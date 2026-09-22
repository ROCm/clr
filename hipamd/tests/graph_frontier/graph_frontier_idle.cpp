// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#define H(x)                                                                      \
  do {                                                                            \
    auto e = (x);                                                                 \
    if (e != hipSuccess) {                                                        \
      std::fprintf(stderr, "line%d %s %s\n", __LINE__, #x, hipGetErrorString(e)); \
      std::exit(2);                                                               \
    }                                                                             \
  } while (0)
#define R(x)                                                    \
  do {                                                          \
    if (!(x)) {                                                 \
      std::fprintf(stderr, "line%d failed %s\n", __LINE__, #x); \
      std::exit(3);                                             \
    }                                                           \
  } while (0)
__global__ void delayed_write(int* output, int branch, unsigned long long ticks) {
  auto begin = wall_clock64();
  while (wall_clock64() - begin < ticks) {
  }
  if (threadIdx.x == 0) output[branch] = 17 + branch;
}
int main() {
  H(hipSetDevice(0));
  int rate = 0;
  H(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));
  R(rate > 0);
  hipStream_t stream;
  H(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  int* output;
  H(hipMalloc(&output, 2 * sizeof(int)));
  H(hipMemset(output, 0, 2 * sizeof(int)));
  hipGraph_t graph;
  hipGraphExec_t exec;
  H(hipGraphCreate(&graph, 0));
  for (int branch = 0; branch < 2; ++branch) {
    auto ticks = static_cast<unsigned long long>(rate) * 200;
    void* args[] = {&output, &branch, &ticks};
    hipKernelNodeParams p{};
    p.func = (void*)delayed_write;
    p.gridDim = dim3(1);
    p.blockDim = dim3(64);
    p.kernelParams = args;
    hipGraphNode_t node;
    H(hipGraphAddKernelNode(&node, graph, nullptr, 0, &p));
  }
  H(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  H(hipGraphLaunch(exec, stream));
  std::fprintf(stderr, "FRONTIER_IDLE phase=before_destroy\n");
  auto start = std::chrono::steady_clock::now();
  H(hipGraphExecDestroy(exec));
  H(hipGraphDestroy(graph));
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  R(ms < 100);
  std::fprintf(stderr, "FRONTIER_IDLE phase=destroy_return\n");
  // No HIP call, host signal poll, or stream observation in this interval.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::fprintf(stderr, "FRONTIER_IDLE phase=idle_end\n");
  int actual[2] = {};
  H(hipMemcpy(actual, output, sizeof(actual), hipMemcpyDeviceToHost));
  R(actual[0] == 17 && actual[1] == 18);
  H(hipStreamDestroy(stream));
  H(hipFree(output));
  std::printf("{\"passed\":true,\"checked_values\":2,\"destroy_ms\":%.6f}\n", ms);
}
