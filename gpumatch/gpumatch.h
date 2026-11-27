// colmap-gpumatch: SIFT feature matching for COLMAP databases on an NVIDIA GPU (see main.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kDescBytes = 128;  // a SIFT descriptor as COLMAP stores it: 128 bytes

// A CUDA call failed: the caller falls back to COLMAP's CPU matcher.
struct GpuError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Which GPU, and whether it can run the matcher. why: "nogpu", "driver" (the NVIDIA driver is older than this build
// needs), "arch" (a card this build has no code for) or "cuda" (anything else, in detail).
struct GpuStatus {
  bool ok = false;
  std::string why, detail, name;
  int ccMajor = 0, ccMinor = 0, driver = 0, runtime = 0;
  size_t memMB = 0;
};
GpuStatus probeGpu();  // picks the device with the most memory and runs a check on it
size_t gpuFreeBytes();  // free memory on that device (1 GB if it can't say)

// For each descriptor of a, the nearest two in b by squared L2 distance on the integer descriptors (exact).
struct Nearest {
  std::vector<int> idx, best, second;  // idx -1 when b is empty; distances INT_MAX when there is no such neighbour
};

// Keeps photos' descriptors on the GPU (up to a memory budget, least recently used dropped first) and runs the
// search. id: COLMAP's image_id, so a photo is uploaded once.
class GpuMatcher {
 public:
  explicit GpuMatcher(size_t budgetBytes);
  ~GpuMatcher();
  GpuMatcher(const GpuMatcher&) = delete;
  GpuMatcher& operator=(const GpuMatcher&) = delete;

  bool has(int id) const;
  void upload(int id, const uint8_t* desc, int rows);
  // a -> b, and b -> a too when both is set, the two directions running side by side
  void nearest(int idA, int idB, bool both, Nearest* ab, Nearest* ba);

 private:
  struct Impl;
  Impl* impl_;
};

// The same search on the CPU, for the self-test.
void nearestCpu(const uint8_t* a, int n, const uint8_t* b, int m, Nearest* out);
