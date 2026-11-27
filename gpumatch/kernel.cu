// The GPU side of colmap-gpumatch: the nearest-two search, the descriptor cache on the GPU, and the device check.
#include "gpumatch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <list>
#include <unordered_map>

namespace {

constexpr int kTile = 64;      // descriptors per tile, on each side
constexpr int kWords = 32;     // a descriptor as 32 words of four bytes
constexpr int kThreads = 256;  // 16 x 16: each thread works out 4 x 4 distances per pair of tiles

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw GpuError(std::string(what) + ": " + cudaGetErrorString(e));
}
#define CK(x) check((x), #x)

// a candidate at distance d, index c, for one descriptor's nearest two. Strictly nearer replaces the best (so of equal
// distances the first stays best and the next becomes second), as COLMAP's CPU matcher does
__device__ __forceinline__ void keep(int d, int c, int& best, int& second, int& idx) {
  if (d < best) {
    second = best;
    best = d;
    idx = c;
  } else if (d < second) {
    second = d;
  }
}
// two threads' nearest two, as one: the same result whichever side merges which, and of equal bests the lower index
__device__ __forceinline__ void merge(int ob, int os, int oi, int& best, int& second, int& idx) {
  if (ob < best || (ob == best && oi >= 0 && (idx < 0 || oi < idx))) {
    second = min(os, best);
    best = ob;
    idx = oi;
  } else {
    second = min(second, ob);
  }
}

// One block takes 64 descriptors of a and walks every tile of b. Distances are exact integers:
// |x - y|^2 = |x|^2 + |y|^2 - 2 x.y, with the dot product four bytes at a time (__dp4a, unsigned).
__global__ void nearestTwo(const uint32_t* __restrict__ a, const int* __restrict__ na, int n,
                           const uint32_t* __restrict__ b, const int* __restrict__ nb, int m,
                           int* __restrict__ outIdx, int* __restrict__ outBest, int* __restrict__ outSecond) {
  __shared__ uint32_t sa[kTile][kWords + 1];  // +1: rows fall in different memory banks
  __shared__ uint32_t sb[kTile][kWords + 1];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int row0 = blockIdx.x * kTile;

  for (int t = threadIdx.x; t < kTile * kWords; t += kThreads) {
    const int r = t / kWords, w = t % kWords;
    sa[r][w] = row0 + r < n ? a[static_cast<size_t>(row0 + r) * kWords + w] : 0u;
  }
  // this thread's rows are ty, ty+16, ty+32, ty+48; its columns in each tile tx, tx+16, tx+32, tx+48
  int best[4], second[4], idx[4], normA[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    best[i] = second[i] = INT_MAX;
    idx[i] = -1;
    const int r = row0 + ty + 16 * i;
    normA[i] = r < n ? na[r] : 0;
  }

  for (int col0 = 0; col0 < m; col0 += kTile) {
    __syncthreads();
    for (int t = threadIdx.x; t < kTile * kWords; t += kThreads) {
      const int r = t / kWords, w = t % kWords;
      sb[r][w] = col0 + r < m ? b[static_cast<size_t>(col0 + r) * kWords + w] : 0u;
    }
    __syncthreads();

    unsigned acc[4][4] = {};
#pragma unroll 4
    for (int w = 0; w < kWords; ++w) {
      uint32_t ra[4], rb[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) ra[i] = sa[ty + 16 * i][w];
#pragma unroll
      for (int j = 0; j < 4; ++j) rb[j] = sb[tx + 16 * j][w];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = __dp4a(ra[i], rb[j], acc[i][j]);
    }
    // columns in increasing order, so the first of equal distances stays best
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = col0 + tx + 16 * j;
      if (c >= m) continue;
      const int normB = nb[c];
#pragma unroll
      for (int i = 0; i < 4; ++i) keep(normA[i] + normB - 2 * static_cast<int>(acc[i][j]), c, best[i], second[i], idx[i]);
    }
  }

  // the 16 threads sharing a row (16 neighbouring lanes of one warp) combine their nearest two
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int off = 8; off > 0; off >>= 1) {
      const int ob = __shfl_xor_sync(0xffffffffu, best[i], off, 16);
      const int os = __shfl_xor_sync(0xffffffffu, second[i], off, 16);
      const int oi = __shfl_xor_sync(0xffffffffu, idx[i], off, 16);
      merge(ob, os, oi, best[i], second[i], idx[i]);
    }
    const int r = row0 + ty + 16 * i;
    if (tx == 0 && r < n) {
      outIdx[r] = idx[i];
      outBest[r] = best[i];
      outSecond[r] = second[i];
    }
  }
}

// a photo's descriptors on the GPU, with their squared norms
struct DeviceSet {
  uint32_t* desc = nullptr;
  int* norms = nullptr;
  int rows = 0;
  size_t bytes = 0;
};

// one direction's output: on the GPU, and pinned host memory to copy it back into
struct Output {
  cudaStream_t stream = nullptr;
  int cap = 0;
  int *dIdx = nullptr, *dBest = nullptr, *dSecond = nullptr;
  int *hIdx = nullptr, *hBest = nullptr, *hSecond = nullptr;
  void reserve(int n) {
    if (n <= cap) return;
    release();
    cap = std::max(n, 1024);
    const size_t s = sizeof(int) * cap;
    CK(cudaMalloc(&dIdx, s));
    CK(cudaMalloc(&dBest, s));
    CK(cudaMalloc(&dSecond, s));
    CK(cudaMallocHost(&hIdx, s));
    CK(cudaMallocHost(&hBest, s));
    CK(cudaMallocHost(&hSecond, s));
  }
  void release() {
    cudaFree(dIdx); cudaFree(dBest); cudaFree(dSecond);
    cudaFreeHost(hIdx); cudaFreeHost(hBest); cudaFreeHost(hSecond);
    dIdx = dBest = dSecond = hIdx = hBest = hSecond = nullptr;
    cap = 0;
  }
  void launch(const DeviceSet& a, const DeviceSet& b) {
    reserve(a.rows);
    const int blocks = (a.rows + kTile - 1) / kTile;
    nearestTwo<<<blocks, kThreads, 0, stream>>>(a.desc, a.norms, a.rows, b.desc, b.norms, b.rows, dIdx, dBest, dSecond);
    CK(cudaGetLastError());
    const size_t s = sizeof(int) * a.rows;
    CK(cudaMemcpyAsync(hIdx, dIdx, s, cudaMemcpyDeviceToHost, stream));
    CK(cudaMemcpyAsync(hBest, dBest, s, cudaMemcpyDeviceToHost, stream));
    CK(cudaMemcpyAsync(hSecond, dSecond, s, cudaMemcpyDeviceToHost, stream));
  }
  void collect(int rows, Nearest* out) {
    CK(cudaStreamSynchronize(stream));
    out->idx.assign(hIdx, hIdx + rows);
    out->best.assign(hBest, hBest + rows);
    out->second.assign(hSecond, hSecond + rows);
  }
};

}  // namespace

struct GpuMatcher::Impl {
  size_t budget = 0, used = 0;
  std::list<int> lru;  // most recently used first
  std::unordered_map<int, std::pair<DeviceSet, std::list<int>::iterator>> sets;
  Output out[2];

  void touch(int id) {
    auto& e = sets.at(id);
    lru.splice(lru.begin(), lru, e.second);
  }
  void drop(int id) {
    auto it = sets.find(id);
    cudaFree(it->second.first.desc);  // cudaFree waits for the GPU, so nothing in flight still reads it
    cudaFree(it->second.first.norms);
    used -= it->second.first.bytes;
    lru.erase(it->second.second);
    sets.erase(it);
  }
};

GpuMatcher::GpuMatcher(size_t budgetBytes) : impl_(new Impl) {
  impl_->budget = budgetBytes;
  for (auto& o : impl_->out) CK(cudaStreamCreate(&o.stream));
}

GpuMatcher::~GpuMatcher() {
  for (auto& o : impl_->out) { o.release(); cudaStreamDestroy(o.stream); }
  while (!impl_->lru.empty()) impl_->drop(impl_->lru.back());
  delete impl_;
}

bool GpuMatcher::has(int id) const { return impl_->sets.count(id) != 0; }

void GpuMatcher::upload(int id, const uint8_t* desc, int rows) {
  if (has(id)) { impl_->touch(id); return; }
  DeviceSet s;
  s.rows = rows;
  const size_t descBytes = static_cast<size_t>(std::max(rows, 1)) * kDescBytes, normBytes = sizeof(int) * std::max(rows, 1);
  s.bytes = descBytes + normBytes;
  // keep the two most recently used (the pair being matched) whatever the budget
  while (impl_->used + s.bytes > impl_->budget && impl_->lru.size() > 2) impl_->drop(impl_->lru.back());
  std::vector<int> norms(std::max(rows, 1), 0);
  for (int r = 0; r < rows; ++r) {
    int sum = 0;
    for (int k = 0; k < kDescBytes; ++k) { const int v = desc[static_cast<size_t>(r) * kDescBytes + k]; sum += v * v; }
    norms[r] = sum;
  }
  CK(cudaMalloc(&s.desc, descBytes));
  CK(cudaMalloc(&s.norms, normBytes));
  if (rows) {
    CK(cudaMemcpy(s.desc, desc, static_cast<size_t>(rows) * kDescBytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.norms, norms.data(), sizeof(int) * rows, cudaMemcpyHostToDevice));
  }
  impl_->lru.push_front(id);
  impl_->sets.emplace(id, std::make_pair(s, impl_->lru.begin()));
  impl_->used += s.bytes;
}

void GpuMatcher::nearest(int idA, int idB, bool both, Nearest* ab, Nearest* ba) {
  impl_->touch(idA);
  impl_->touch(idB);
  const DeviceSet& a = impl_->sets.at(idA).first;
  const DeviceSet& b = impl_->sets.at(idB).first;
  if (a.rows == 0 || b.rows == 0) {
    *ab = Nearest{std::vector<int>(a.rows, -1), std::vector<int>(a.rows, INT_MAX), std::vector<int>(a.rows, INT_MAX)};
    if (both) *ba = Nearest{std::vector<int>(b.rows, -1), std::vector<int>(b.rows, INT_MAX), std::vector<int>(b.rows, INT_MAX)};
    return;
  }
  impl_->out[0].launch(a, b);
  if (both) impl_->out[1].launch(b, a);
  impl_->out[0].collect(a.rows, ab);
  if (both) impl_->out[1].collect(b.rows, ba);
}

void nearestCpu(const uint8_t* a, int n, const uint8_t* b, int m, Nearest* out) {
  out->idx.assign(n, -1);
  out->best.assign(n, INT_MAX);
  out->second.assign(n, INT_MAX);
  for (int i = 0; i < n; ++i) {
    int best = INT_MAX, second = INT_MAX, idx = -1;
    for (int j = 0; j < m; ++j) {
      int d = 0;
      for (int k = 0; k < kDescBytes; ++k) {
        const int v = static_cast<int>(a[static_cast<size_t>(i) * kDescBytes + k]) - b[static_cast<size_t>(j) * kDescBytes + k];
        d += v * v;
      }
      if (d < best) { second = best; best = d; idx = j; }
      else if (d < second) { second = d; }
    }
    out->idx[i] = idx;
    out->best[i] = best;
    out->second[i] = second;
  }
}

size_t gpuFreeBytes() {
  size_t freeMem = 0, totalMem = 0;
  return cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess ? freeMem : size_t(1) << 30;
}

GpuStatus probeGpu() {
  GpuStatus s;
  cudaDriverGetVersion(&s.driver);
  cudaRuntimeGetVersion(&s.runtime);
  int count = 0;
  const cudaError_t e = cudaGetDeviceCount(&count);
  if (e == cudaErrorInsufficientDriver) { s.why = "driver"; s.detail = cudaGetErrorString(e); return s; }
  if (e == cudaErrorNoDevice || (e == cudaSuccess && count == 0)) { s.why = "nogpu"; s.detail = "no CUDA device"; return s; }
  if (e != cudaSuccess) { s.why = "cuda"; s.detail = cudaGetErrorString(e); return s; }
  int dev = -1;
  size_t mem = 0;
  for (int d = 0; d < count; ++d) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
    if (dev < 0 || p.totalGlobalMem > mem) { dev = d; mem = p.totalGlobalMem; }
  }
  if (dev < 0) { s.why = "cuda"; s.detail = "no device answered"; return s; }
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, dev);
  s.name = p.name;
  s.ccMajor = p.major;
  s.ccMinor = p.minor;
  s.memMB = p.totalGlobalMem >> 20;
  if (cudaSetDevice(dev) != cudaSuccess) { s.why = "cuda"; s.detail = "cudaSetDevice failed"; return s; }
  // run the real kernel once on something small and compare it with the CPU: a card this build has no code for
  // fails here ("no kernel image"), as does anything else that would make the results wrong
  try {
    const int n = 70, m = 130;
    std::vector<uint8_t> a(n * kDescBytes), b(m * kDescBytes);
    unsigned x = 12345;
    for (auto& v : a) { x = x * 1103515245u + 12345u; v = static_cast<uint8_t>(x >> 24); }
    for (auto& v : b) { x = x * 1103515245u + 12345u; v = static_cast<uint8_t>(x >> 24); }
    GpuMatcher g(64u << 20);
    g.upload(1, a.data(), n);
    g.upload(2, b.data(), m);
    Nearest gab, gba, cab;
    g.nearest(1, 2, false, &gab, &gba);
    nearestCpu(a.data(), n, b.data(), m, &cab);
    if (gab.idx != cab.idx || gab.best != cab.best || gab.second != cab.second) { s.why = "cuda"; s.detail = "the GPU's results don't match the CPU's"; return s; }
  } catch (const GpuError& err) {
    const std::string msg = err.what();
    s.why = msg.find("no kernel image") != std::string::npos ? "arch" : "cuda";
    s.detail = msg;
    return s;
  }
  s.ok = true;
  return s;
}
