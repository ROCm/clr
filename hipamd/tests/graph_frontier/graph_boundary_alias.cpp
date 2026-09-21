// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdint>
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
__global__ void previous_pair(uint32_t* data, int row, int branch, unsigned long long delay) {
  auto t = wall_clock64();
  while (wall_clock64() - t < delay) {
  }
  if (threadIdx.x == 0)
    data[2 * row + branch] =
        (data[2 * (row - 1)] + 3 * data[2 * (row - 1) + 1] + branch + 7 * row) & 65535u;
}
int main() {
  constexpr int count = 64;
  H(hipSetDevice(0));
  int rate;
  H(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));
  R(rate > 0);
  uint32_t* data;
  H(hipMalloc(&data, 2 * (count + 1) * sizeof(uint32_t)));
  std::vector<uint32_t> expected(2 * (count + 1), 0);
  expected[0] = 1;
  expected[1] = 2;
  hipStream_t stream;
  H(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  H(hipMemcpyAsync(data, expected.data(), expected.size() * sizeof(uint32_t), hipMemcpyHostToDevice,
                   stream));
  hipGraph_t graph[2];
  hipGraphExec_t exec[2];
  hipGraphNode_t node[2][2];
  auto params = [&](void** args) {
    hipKernelNodeParams p{};
    p.func = (void*)previous_pair;
    p.gridDim = dim3(1);
    p.blockDim = dim3(64);
    p.kernelParams = args;
    return p;
  };
  for (int g = 0; g < 2; ++g) {
    H(hipGraphCreate(&graph[g], 0));
    for (int b = 0; b < 2; ++b) {
      int row = 1;
      unsigned long long delay = 0;
      void* args[] = {&data, &row, &b, &delay};
      auto p = params(args);
      H(hipGraphAddKernelNode(&node[g][b], graph[g], nullptr, 0, &p));
    }
    H(hipGraphInstantiate(&exec[g], graph[g], nullptr, nullptr, 0));
  }
  for (int row = 1; row <= count; ++row) {
    int g = row % 2;
    for (int b = 0; b < 2; ++b) {
      unsigned long long delay = (b == (row % 2)) ? static_cast<unsigned long long>(rate) / 5 : 0;
      void* args[] = {&data, &row, &b, &delay};
      auto p = params(args);
      H(hipGraphExecKernelNodeSetParams(exec[g], node[g][b], &p));
      expected[2 * row + b] =
          (expected[2 * (row - 1)] + 3 * expected[2 * (row - 1) + 1] + b + 7 * row) & 65535u;
    }
    H(hipGraphLaunch(exec[g], stream));
  }
  for (int g = 0; g < 2; ++g) {
    H(hipGraphExecDestroy(exec[g]));
    H(hipGraphDestroy(graph[g]));
  }
  std::vector<uint32_t> actual(expected.size());
  H(hipMemcpyAsync(actual.data(), data, actual.size() * sizeof(uint32_t), hipMemcpyDeviceToHost,
                   stream));
  H(hipStreamSynchronize(stream));
  for (size_t i = 0; i < expected.size(); ++i) R(actual[i] == expected[i]);
  H(hipStreamDestroy(stream));
  H(hipFree(data));
  std::printf(
      "{\"passed\":true,\"launches\":64,\"checked_values\":130,\"alternating_graph_execs\":2}\n");
}
