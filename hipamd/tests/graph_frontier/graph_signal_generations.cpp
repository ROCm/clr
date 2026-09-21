// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
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
__global__ void hold(unsigned long long ticks) {
  auto t = wall_clock64();
  while (wall_clock64() - t < ticks) {
  }
}
__global__ void node(int* data, int run, int out, int left, int right, int add) {
  if (threadIdx.x == 0) {
    int* row = data + run * 7;
    row[out] = add + (left < 0 ? 0 : row[left]) + (right < 0 ? 0 : row[right]);
  }
}
int main() {
  constexpr int runs = 64;
  H(hipSetDevice(0));
  int rate = 0;
  H(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));
  R(rate > 0);
  int* data;
  H(hipMalloc(&data, runs * 7 * sizeof(int)));
  H(hipMemset(data, 0, runs * 7 * sizeof(int)));
  hipStream_t streams[2];
  for (auto& s : streams) H(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  const std::array<int, 7> left{{-1, -1, 0, 1, 2, 0, 4}}, right{{-1, -1, -1, -1, 3, -1, 5}},
      expected{{1, 2, 4, 6, 15, 7, 29}};
  hipGraph_t graph;
  hipGraphExec_t exec;
  H(hipGraphCreate(&graph, 0));
  std::array<hipGraphNode_t, 7> nodes{};
  int run = 0;
  for (int i = 0; i < 7; ++i) {
    int l = left[i], r = right[i], add = i + 1;
    void* args[] = {&data, &run, &i, &l, &r, &add};
    hipKernelNodeParams p{};
    p.func = (void*)node;
    p.gridDim = dim3(1);
    p.blockDim = dim3(64);
    p.kernelParams = args;
    std::vector<hipGraphNode_t> deps;
    if (l >= 0) deps.push_back(nodes[l]);
    if (r >= 0) deps.push_back(nodes[r]);
    H(hipGraphAddKernelNode(&nodes[i], graph, deps.data(), deps.size(), &p));
  }
  H(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  // Finite delay ensures many launches are in flight while parameters are updated.
  // It cannot depend on work that has not been submitted or on a CPU polling loop.
  hipLaunchKernelGGL(hold, dim3(1), dim3(1), 0, streams[0],
                     static_cast<unsigned long long>(rate) * 200);
  H(hipGetLastError());
  for (run = 0; run < runs; ++run) {
    for (int i = 0; i < 7; ++i) {
      int l = left[i], r = right[i], add = (i + 1) * (run + 1);
      void* args[] = {&data, &run, &i, &l, &r, &add};
      hipKernelNodeParams p{};
      p.func = (void*)node;
      p.gridDim = dim3(1);
      p.blockDim = dim3(64);
      p.kernelParams = args;
      H(hipGraphExecKernelNodeSetParams(exec, nodes[i], &p));
    }
    H(hipGraphLaunch(exec, streams[run % 2]));
  }
  H(hipGraphExecDestroy(exec));
  H(hipGraphDestroy(graph));
  for (auto s : streams) H(hipStreamSynchronize(s));
  std::vector<int> actual(runs * 7);
  H(hipMemcpy(actual.data(), data, actual.size() * sizeof(int), hipMemcpyDeviceToHost));
  for (int j = 0; j < runs; ++j)
    for (int i = 0; i < 7; ++i) R(actual[j * 7 + i] == expected[i] * (j + 1));
  for (auto s : streams) H(hipStreamDestroy(s));
  H(hipFree(data));
  std::printf(
      "{\"passed\":true,\"launches\":64,\"updates\":448,\"checked_values\":448,\"destroyed_before_"
      "sync\":true}\n");
}
