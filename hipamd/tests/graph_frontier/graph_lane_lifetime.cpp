// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define H(x)                                                                         \
  do {                                                                               \
    auto err = (x);                                                                  \
    if (err != hipSuccess) {                                                         \
      std::fprintf(stderr, "line%d %s: %s\n", __LINE__, #x, hipGetErrorString(err)); \
      std::exit(2);                                                                  \
    }                                                                                \
  } while (0)
#define R(x)                                                     \
  do {                                                           \
    if (!(x)) {                                                  \
      std::fprintf(stderr, "line%d failed: %s\n", __LINE__, #x); \
      std::exit(3);                                              \
    }                                                            \
  } while (0)
__global__ void work(int* data, int out, int lhs, int rhs, int add, unsigned long long delay) {
  auto begin = wall_clock64();
  while (wall_clock64() - begin < delay) {
  }
  if (threadIdx.x == 0) data[out] = add + (lhs < 0 ? 0 : data[lhs]) + (rhs < 0 ? 0 : data[rhs]);
}
struct Node {
  int out, lhs, rhs, add;
  unsigned long long delay;
  hipGraphNode_t node;
};
int main(int argc, char** argv) {
  R(argc == 2);
  bool fail = std::atoi(argv[1]) != 0;
  H(hipSetDevice(0));
  int rate = 0;
  H(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));
  R(rate > 0);
  int* data;
  int* result;
  H(hipMalloc(&data, 9 * sizeof(int)));
  H(hipHostMalloc(&result, 9 * sizeof(int)));
  hipStream_t s[2];
  for (auto& stream : s) H(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  std::puts("step,disabled,updated,stream,expected_error,observed_error,correct");
  for (int step = 0; step < (fail ? 1 : 6); ++step) {
    int disabled_node = step == 1 ? 0 : step == 3 ? 3 : -1;
    bool disabled = disabled_node >= 0, updated = step == 2;
    int root_add = updated ? 7 : 1;
    std::array<Node, 9> n{{{0, -1, -1, 1, 0, {}},
                           {1, 0, -1, 1, 0, {}},
                           {2, 1, -1, 1, 0, {}},
                           {3, -1, -1, 1, static_cast<unsigned long long>(rate), {}},
                           {4, 3, -1, 1, static_cast<unsigned long long>(rate), {}},
                           {5, 2, -1, 1, 0, {}},
                           {6, 5, -1, 1, 0, {}},
                           {7, 2, 4, 1, 0, {}},
                           {8, 6, 7, 1, 0, {}}}};
    hipGraph_t graph;
    hipGraphExec_t exec;
    H(hipGraphCreate(&graph, 0));
    auto params = [&](Node& x, void** args) {
      hipKernelNodeParams p{};
      p.func = (void*)work;
      p.gridDim = dim3(1);
      p.blockDim = dim3(64);
      p.kernelParams = args;
      return p;
    };
    for (auto& x : n) {
      std::vector<hipGraphNode_t> deps;
      if (x.lhs >= 0) deps.push_back(n[x.lhs].node);
      if (x.rhs >= 0) deps.push_back(n[x.rhs].node);
      void* args[] = {&data, &x.out, &x.lhs, &x.rhs, &x.add, &x.delay};
      auto p = params(x, args);
      H(hipGraphAddKernelNode(&x.node, graph, deps.data(), deps.size(), &p));
    }
    H(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    if (disabled) H(hipGraphNodeSetEnabled(exec, n[disabled_node].node, 0));
    if (updated) {
      auto& x = n[0];
      x.add = root_add;
      void* args[] = {&data, &x.out, &x.lhs, &x.rhs, &x.add, &x.delay};
      auto p = params(x, args);
      H(hipGraphExecKernelNodeSetParams(exec, x.node, &p));
    }
    auto stream = s[step % 2];
    H(hipMemsetAsync(data, 0, 9 * sizeof(int), stream));
    std::fprintf(stderr, "LANE_FIXTURE_BEGIN step=%d fail=%d disabled=%d updated=%d\n", step, fail,
                 disabled, updated);
    std::fflush(stderr);
    auto err = hipGraphLaunch(exec, stream);
    R(err == (fail ? hipErrorUnknown : hipSuccess));
    if (fail) {
      auto last = hipGetLastError();
      R(last == hipSuccess || last == hipErrorUnknown);
    }
    // Destroy before completion; graph callback ownership must protect all published arguments.
    H(hipGraphExecDestroy(exec));
    H(hipGraphDestroy(graph));
    H(hipMemcpyAsync(result, data, 9 * sizeof(int), hipMemcpyDeviceToHost, stream));
    H(hipStreamSynchronize(stream));
    std::array<int, 9> expected{};
    for (auto& x : n) {
      if (x.out == disabled_node) continue;
      expected[x.out] =
          x.add + (x.lhs < 0 ? 0 : expected[x.lhs]) + (x.rhs < 0 ? 0 : expected[x.rhs]);
    }
    if (fail) {
      for (int i = 0; i < 5; ++i) R(result[i] == expected[i]);
      R(result[8] == 0);
      for (int i = 5; i < 8; ++i) R(result[i] == 0 || result[i] == expected[i]);
      R((result[5] == 0) == (result[6] == 0));
      R(result[6] != 0 || result[7] != 0);
    } else {
      for (int i = 0; i < 9; ++i) R(result[i] == expected[i]);
    }
    std::fprintf(stderr, "LANE_FIXTURE_END step=%d correct=1 values=", step);
    for (int i = 0; i < 9; ++i) std::fprintf(stderr, "%s%d", i ? "," : "", result[i]);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    std::printf("%d,%d,%d,%d,%d,%d,1\n", step, disabled, updated, step % 2,
                int(fail ? hipErrorUnknown : hipSuccess), int(err));
    std::fflush(stdout);
  }
  for (auto stream : s) H(hipStreamDestroy(stream));
  H(hipHostFree(result));
  H(hipFree(data));
}
