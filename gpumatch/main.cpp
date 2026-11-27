// colmap-gpumatch: SIFT feature matching for COLMAP databases on an NVIDIA GPU.
//
// COLMAP's feature_extractor (CPU SIFT, VLFeat) writes each photo's descriptors to database.db. This finds, for every
// feature, the nearest two in the other photo (exact squared L2 distances on the integer descriptors, worked out on the
// GPU), keeps the matches that pass COLMAP's own tests with its default options (ratio test 0.8, maximum distance 0.7,
// cross check), and writes them to the database's matches table as COLMAP's matchers would. COLMAP's
// geometric_verifier then checks them exactly as it checks its own. So the photos, features and verification are
// COLMAP's; only the pairing of features moves to the GPU.
//
//   colmap-gpumatch --check
//       Is there a usable GPU? Prints one JSON line; exit 0 if so, 2 if not.
//   colmap-gpumatch --database DB --pairs PAIRS [--max_ratio R] [--max_distance D] [--cross_check 0|1]
//       Match the photo pairs listed in PAIRS (UTF-8, one pair a line: name1 TAB name2, COLMAP's image names).
//       Pairs already in the matches table are left alone, so a run that stopped carries on. Progress lines:
//       "Matching pair [k/n]". Exit 0 done, 2 a GPU problem (use COLMAP's CPU matcher instead), 1 anything else.
//   colmap-gpumatch --selftest
//       The GPU against a CPU reference on random descriptors (ties included); exit 0 if every result agrees.
//   colmap-gpumatch --bench FEATURES PAIRS
//       Pairs a second, both directions, at that many features a photo.
//
// The database is SQLite through Windows' own winsqlite3.dll, and the CUDA runtime is linked in statically, so this
// needs nothing beside it but an NVIDIA driver.
#include "gpumatch.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsqlite/winsqlite3.h>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr long long kMaxNumImages = 2147483647LL;  // COLMAP's kMaxNumImages: pair_id = id1 * this + id2, id1 < id2
constexpr int kSiftType = 0;                       // COLMAP's FeatureExtractorType::SIFT in the descriptors table
constexpr int kCommitEvery = 200;                  // pairs per transaction, so a stopped run keeps most of its work

std::string utf8(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

std::string jsonEscape(const std::string& s) {
  std::string o;
  for (const char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
    else o += c;
  }
  return o;
}

void out(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  fflush(stdout);
}

// --- SQLite ------------------------------------------------------------------------------------------------------

struct Db {
  sqlite3* h = nullptr;
  explicit Db(const std::string& path) {
    if (sqlite3_open_v2(path.c_str(), &h, SQLITE_OPEN_READWRITE, nullptr) != SQLITE_OK) fail("open the database");
    sqlite3_busy_timeout(h, 15000);
  }
  ~Db() { sqlite3_close(h); }
  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("SQLite: couldn't ") + what + ": " + (h ? sqlite3_errmsg(h) : "no database"));
  }
  void exec(const char* sql) {
    if (sqlite3_exec(h, sql, nullptr, nullptr, nullptr) != SQLITE_OK) fail(sql);
  }
};

struct Stmt {
  sqlite3_stmt* s = nullptr;
  const Db& db;
  Stmt(const Db& d, const char* sql) : db(d) {
    if (sqlite3_prepare_v2(d.h, sql, -1, &s, nullptr) != SQLITE_OK) d.fail(sql);
  }
  ~Stmt() { sqlite3_finalize(s); }
  bool step() {
    const int r = sqlite3_step(s);
    if (r == SQLITE_ROW) return true;
    if (r == SQLITE_DONE) return false;
    db.fail("step");
  }
  void reset() { sqlite3_reset(s); sqlite3_clear_bindings(s); }
};

// --- COLMAP's tests for a match -------------------------------------------------------------------------------------

struct Rules {
  float maxRatio = 0.8f, maxDistance = 0.7f;
  bool crossCheck = true;
};

// FindBestMatchesOneWayIndex in COLMAP's feature/sift.cc: the nearest within the maximum distance, and clearly nearer
// than the second (>= so equal distances are rejected)
bool passes(const Rules& r, int best, int second) {
  const float maxL2 = 512.f * 512.f * r.maxDistance * r.maxDistance;  // kSqSiftDescriptorNorm * max_distance^2
  const float b = static_cast<float>(best), s = static_cast<float>(second);
  if (b > maxL2) return false;
  return !(std::sqrt(b) >= r.maxRatio * std::sqrt(s));
}

// FindBestMatchesIndex: a's feature i and b's feature j match when each is the other's accepted nearest
std::vector<std::pair<uint32_t, uint32_t>> matchesFrom(const Rules& r, const Nearest& ab, const Nearest* ba) {
  std::vector<std::pair<uint32_t, uint32_t>> m;
  for (size_t i = 0; i < ab.idx.size(); ++i) {
    const int j = ab.idx[i];
    if (j < 0 || !passes(r, ab.best[i], ab.second[i])) continue;
    if (r.crossCheck) {
      if (!ba || ba->idx[j] != static_cast<int>(i) || !passes(r, ba->best[j], ba->second[j])) continue;
    }
    m.emplace_back(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
  }
  return m;
}

// --- modes -----------------------------------------------------------------------------------------------------

int runCheck() {
  const GpuStatus s = probeGpu();
  out("{\"ok\":%s,\"why\":\"%s\",\"detail\":\"%s\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"memMB\":%zu,\"driver\":%d,\"runtime\":%d}\n",
      s.ok ? "true" : "false", s.why.c_str(), jsonEscape(s.detail).c_str(), jsonEscape(s.name).c_str(), s.ccMajor, s.ccMinor,
      s.memMB, s.driver, s.runtime);
  return s.ok ? 0 : 2;
}

std::vector<uint8_t> randomDesc(int rows, unsigned& x) {
  std::vector<uint8_t> d(static_cast<size_t>(rows) * kDescBytes);
  for (auto& v : d) { x = x * 1103515245u + 12345u; v = static_cast<uint8_t>((x >> 16) & 0xff); }
  return d;
}

int runSelftest() {
  const GpuStatus s = probeGpu();
  if (!s.ok) { out("GPU not usable (%s): %s\n", s.why.c_str(), s.detail.c_str()); return 2; }
  out("GPU: %s (compute %d.%d, %zu MB)\n", s.name.c_str(), s.ccMajor, s.ccMinor, s.memMB);
  // sizes around the tile edges, one-sided cases, and a realistic one
  const std::vector<std::pair<int, int>> sizes = {{1, 1}, {1, 70}, {65, 1}, {63, 64}, {64, 65}, {100, 130}, {1000, 777}, {3000, 2500}};
  unsigned x = 2026;
  int bad = 0, id = 1;
  GpuMatcher g(size_t(512) << 20);
  for (const auto& [n, m] : sizes) {
    std::vector<uint8_t> a = randomDesc(n, x), b = randomDesc(m, x);
    // ties: copies inside b (equal nearest and second), and features of a copied from b (distance 0)
    for (int k = 0; k + 1 < m && k < 20; k += 2) std::memcpy(&b[static_cast<size_t>(k + 1) * kDescBytes], &b[static_cast<size_t>(k) * kDescBytes], kDescBytes);
    for (int k = 0; k < n && k < m && k < 40; k += 3) std::memcpy(&a[static_cast<size_t>(k) * kDescBytes], &b[static_cast<size_t>(m - 1 - k) * kDescBytes], kDescBytes);
    const int ia = id++, ib = id++;
    g.upload(ia, a.data(), n);
    g.upload(ib, b.data(), m);
    Nearest gab, gba, cab, cba;
    g.nearest(ia, ib, true, &gab, &gba);
    nearestCpu(a.data(), n, b.data(), m, &cab);
    nearestCpu(b.data(), m, a.data(), n, &cba);
    const bool ok = gab.idx == cab.idx && gab.best == cab.best && gab.second == cab.second && gba.idx == cba.idx && gba.best == cba.best && gba.second == cba.second;
    out("%s  %d x %d\n", ok ? "ok  " : "FAIL", n, m);
    if (!ok) ++bad;
  }
  out(bad ? "selftest: %d FAILED\n" : "selftest: all ok\n", bad);
  return bad ? 1 : 0;
}

int runBench(int features, int pairs) {
  const GpuStatus s = probeGpu();
  if (!s.ok) { out("GPU not usable (%s): %s\n", s.why.c_str(), s.detail.c_str()); return 2; }
  const int photos = 16;
  unsigned x = 7;
  GpuMatcher g(size_t(2048) << 20);
  for (int p = 0; p < photos; ++p) { const auto d = randomDesc(features, x); g.upload(p, d.data(), features); }
  Nearest ab, ba;
  g.nearest(0, 1, true, &ab, &ba);  // warm-up
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < pairs; ++k) g.nearest(k % photos, (k + 1 + k / photos) % photos, true, &ab, &ba);
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out("%s: %d features a photo, %d pairs in %.2f s: %.1f pairs/s (%.2f ms a pair)\n", s.name.c_str(), features, pairs, sec, pairs / sec, 1000 * sec / pairs);
  return 0;
}

int runMatch(const std::string& dbPath, const std::wstring& pairsPath, const Rules& rules) {
  const GpuStatus st = probeGpu();
  if (!st.ok) { out("GPU not usable (%s): %s\n", st.why.c_str(), st.detail.c_str()); return 2; }
  out("GPU: %s (compute %d.%d, %zu MB)\n", st.name.c_str(), st.ccMajor, st.ccMinor, st.memMB);

  Db db(dbPath);
  std::unordered_map<std::string, int> idOf;
  {
    Stmt q(db, "SELECT image_id, name FROM images;");
    while (q.step()) idOf[reinterpret_cast<const char*>(sqlite3_column_text(q.s, 1))] = sqlite3_column_int(q.s, 0);
  }
  std::unordered_set<long long> done;
  {
    Stmt q(db, "SELECT pair_id FROM matches;");
    while (q.step()) done.insert(sqlite3_column_int64(q.s, 0));
  }

  // the pairs, as (smaller id, larger id), each once, in order so each photo's pairs come together
  std::ifstream f(pairsPath, std::ios::binary);
  if (!f) throw std::runtime_error("couldn't read the pair list");
  std::set<std::pair<int, int>> pairs;
  size_t unknown = 0, already = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) { ++unknown; continue; }
    const auto i1 = idOf.find(line.substr(0, tab)), i2 = idOf.find(line.substr(tab + 1));
    if (i1 == idOf.end() || i2 == idOf.end()) { ++unknown; continue; }
    int a = i1->second, b = i2->second;
    if (a == b) continue;
    if (a > b) std::swap(a, b);
    if (done.count(kMaxNumImages * a + b)) { ++already; continue; }
    pairs.emplace(a, b);
  }
  if (unknown) out("warning: %zu pair(s) name photos that aren't in the database\n", unknown);
  out("%zu pairs to match (%zu already matched)\n", pairs.size(), already);
  if (pairs.empty()) return 0;

  // GPU memory for descriptors: most of what's free, leaving room for the outputs and the rest of the system
  GpuMatcher g(static_cast<size_t>(gpuFreeBytes() * 0.6));

  Stmt readDesc(db, "SELECT type, rows, cols, data FROM descriptors WHERE image_id = ?;");
  auto ensure = [&](int id) {
    if (g.has(id)) return;
    readDesc.reset();
    sqlite3_bind_int(readDesc.s, 1, id);
    int rows = 0;
    std::vector<uint8_t> d;
    if (readDesc.step()) {
      const int type = sqlite3_column_int(readDesc.s, 0), cols = sqlite3_column_int(readDesc.s, 2);
      rows = sqlite3_column_int(readDesc.s, 1);
      if (rows > 0 && (type != kSiftType || cols != kDescBytes)) throw std::runtime_error("descriptors aren't SIFT (type " + std::to_string(type) + ", " + std::to_string(cols) + " columns)");
      const void* blob = sqlite3_column_blob(readDesc.s, 3);
      const size_t bytes = static_cast<size_t>(sqlite3_column_bytes(readDesc.s, 3));
      if (bytes != static_cast<size_t>(rows) * kDescBytes) throw std::runtime_error("a descriptor blob has the wrong size");
      d.assign(static_cast<const uint8_t*>(blob), static_cast<const uint8_t*>(blob) + bytes);
    }
    g.upload(id, d.data(), rows);
  };

  Stmt write(db, "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, 2, ?);");
  const auto t0 = std::chrono::steady_clock::now();
  const size_t total = pairs.size(), every = std::max<size_t>(1, total / 200);
  size_t k = 0, nMatches = 0;
  db.exec("BEGIN;");
  Nearest ab, ba;
  for (const auto& [a, b] : pairs) {
    ensure(a);
    ensure(b);
    g.nearest(a, b, rules.crossCheck, &ab, &ba);
    const auto m = matchesFrom(rules, ab, rules.crossCheck ? &ba : nullptr);
    nMatches += m.size();
    std::vector<uint32_t> blob(m.size() * 2);
    for (size_t i = 0; i < m.size(); ++i) { blob[2 * i] = m[i].first; blob[2 * i + 1] = m[i].second; }
    static const uint32_t empty = 0;
    write.reset();
    sqlite3_bind_int64(write.s, 1, kMaxNumImages * a + b);
    sqlite3_bind_int(write.s, 2, static_cast<int>(m.size()));
    // an empty but present blob for a pair with no matches, as COLMAP writes one, so the pair isn't matched again
    sqlite3_bind_blob(write.s, 3, blob.empty() ? static_cast<const void*>(&empty) : blob.data(), static_cast<int>(blob.size() * sizeof(uint32_t)), SQLITE_TRANSIENT);
    write.step();
    ++k;
    if (k % kCommitEvery == 0) { db.exec("COMMIT;"); db.exec("BEGIN;"); }
    if (k % every == 0 || k == total) out("Matching pair [%zu/%zu]\n", k, total);
  }
  db.exec("COMMIT;");
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out("Matched %zu pairs in %.1f s (%.1f pairs/s), %zu matches\n", total, sec, total / std::max(sec, 1e-9), nMatches);
  return 0;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  std::map<std::wstring, std::wstring> opt;
  std::vector<std::wstring> rest;
  for (int i = 1; i < argc; ++i) {
    const std::wstring a = argv[i];
    if (a == L"--check" || a == L"--selftest") opt[a] = L"1";
    else if (a.rfind(L"--", 0) == 0 && i + 1 < argc) opt[a] = argv[++i];
    else rest.push_back(a);
  }
  try {
    if (opt.count(L"--check")) return runCheck();
    if (opt.count(L"--selftest")) return runSelftest();
    if (opt.count(L"--bench")) return runBench(std::stoi(opt[L"--bench"]), rest.empty() ? 200 : std::stoi(rest[0]));
    if (opt.count(L"--database") && opt.count(L"--pairs")) {
      Rules r;
      if (opt.count(L"--max_ratio")) r.maxRatio = std::stof(opt[L"--max_ratio"]);
      if (opt.count(L"--max_distance")) r.maxDistance = std::stof(opt[L"--max_distance"]);
      if (opt.count(L"--cross_check")) r.crossCheck = opt[L"--cross_check"] != L"0";
      return runMatch(utf8(opt[L"--database"]), opt[L"--pairs"], r);
    }
    out("usage: colmap-gpumatch --check | --selftest | --bench FEATURES [PAIRS] | --database DB --pairs PAIRS [--max_ratio R] [--max_distance D] [--cross_check 0|1]\n");
    return 1;
  } catch (const GpuError& e) {
    out("GPU error: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    out("error: %s\n", e.what());
    return 1;
  }
}
