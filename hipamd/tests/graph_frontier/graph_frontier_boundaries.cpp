// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
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
__global__ void write_branch(int* in, int* out, int run, int branch, unsigned long long delay) {
  auto t = wall_clock64();
  while (wall_clock64() - t < delay) {
  }
  if (threadIdx.x == 0) out[2 * run + branch] = in[run] + branch + 1;
}
__global__ void finite_hold(unsigned long long delay) {
  auto t = wall_clock64();
  while (wall_clock64() - t < delay) {
  }
}
struct Callback {
  int* values;
  std::atomic<int> count{0};
};
static void after_copy(void* ptr) {
  auto* c = static_cast<Callback*>(ptr);
  R(c->values[0] == 18 && c->values[1] == 19);
  c->count.fetch_add(1);
}
int main() {
  constexpr int runs = 1105;
  H(hipSetDevice(0));
  int rate = 0;
  H(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));
  R(rate > 0);
  int *input, *output, *host;
  H(hipMalloc(&input, runs * sizeof(int)));
  H(hipMalloc(&output, 2 * runs * sizeof(int)));
  H(hipHostMalloc(&host, 2 * runs * sizeof(int)));
  std::vector<int> expected(runs);
  for (int i = 0; i < runs; ++i) expected[i] = 17 + i;
  hipStream_t stream, observer;
  H(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  H(hipStreamCreateWithFlags(&observer, hipStreamNonBlocking));
  H(hipMemsetAsync(output, 0, 2 * runs * sizeof(int), stream));
  H(hipMemcpyAsync(input, expected.data(), runs * sizeof(int), hipMemcpyHostToDevice, stream));
  hipGraph_t graph;
  hipGraphExec_t exec;
  hipGraphNode_t nodes[2];
  H(hipGraphCreate(&graph, 0));
  int run = 0;
  auto params = [&](int& branch, unsigned long long& delay, void** args) {
    hipKernelNodeParams p{};
    p.func = (void*)write_branch;
    p.gridDim = dim3(1);
    p.blockDim = dim3(64);
    p.kernelParams = args;
    return p;
  };
  for (int b = 0; b < 2; ++b) {
    unsigned long long delay = 0;
    void* args[] = {&input, &output, &run, &b, &delay};
    auto p = params(b, delay, args);
    H(hipGraphAddKernelNode(&nodes[b], graph, nullptr, 0, &p));
  }
  H(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  // The first graph must import pending asynchronous input/copy work.
  H(hipGraphLaunch(exec, stream));
  hipEvent_t first;
  H(hipEventCreateWithFlags(&first, hipEventDisableTiming));
  H(hipEventRecord(first, stream));
  // An earlier public event must not be rebound to later private graph work.
  hipLaunchKernelGGL(finite_hold, dim3(1), dim3(1), 0, stream,
                     static_cast<unsigned long long>(rate) * 200);
  H(hipGetLastError());
  H(hipGraphLaunch(exec, stream));
  H(hipStreamWaitEvent(observer, first, 0));
  H(hipMemcpyAsync(host, output, 2 * sizeof(int), hipMemcpyDeviceToHost, observer));
  Callback callback{host};
  H(hipLaunchHostFunc(observer, after_copy, &callback));
  H(hipStreamSynchronize(observer));
  R(callback.count.load() == 1);
  R(hipEventQuery(first) == hipSuccess);
  R(hipStreamQuery(stream) == hipErrorNotReady);
  (void)hipGetLastError();
  H(hipStreamSynchronize(stream));
  // A long graph chain crosses the normal host-batch watermark and AQL ring.
  // Every launch writes a different row while parameters change in flight.
  for (run = 1; run < runs; ++run) {
    for (int b = 0; b < 2; ++b) {
      unsigned long long delay = 0;
      void* args[] = {&input, &output, &run, &b, &delay};
      auto p = params(b, delay, args);
      H(hipGraphExecKernelNodeSetParams(exec, nodes[b], &p));
    }
    H(hipGraphLaunch(exec, stream));
  }
  // This copy forces the pre-submit public bridge, not a late host-only flush.
  H(hipMemcpyAsync(host, output, 2 * runs * sizeof(int), hipMemcpyDeviceToHost, stream));
  H(hipStreamSynchronize(stream));
  for (int i = 0; i < runs; ++i)
    for (int b = 0; b < 2; ++b) R(host[2 * i + b] == expected[i] + b + 1);
  // Concurrent observation must not deadlock notification against execution.
  std::atomic<bool> stop{false};
  std::atomic<int> queries{0};
  std::thread poll([&] {
    H(hipSetDevice(0));
    while (!stop.load()) {
      auto e = hipStreamQuery(stream);
      R(e == hipSuccess || e == hipErrorNotReady);
      queries.fetch_add(1);
      std::this_thread::yield();
    }
  });
  for (int i = 0; i < 32; ++i) H(hipGraphLaunch(exec, stream));
  stop.store(true);
  poll.join();
  H(hipStreamSynchronize(stream));
  // Destruction of an idle graph tail must retain arguments until stream destroy.
  H(hipGraphLaunch(exec, stream));
  H(hipGraphExecDestroy(exec));
  H(hipGraphDestroy(graph));
  H(hipStreamDestroy(stream));
  H(hipEventDestroy(first));
  H(hipStreamDestroy(observer));
  H(hipFree(input));
  H(hipFree(output));
  H(hipHostFree(host));
  std::printf(
      "{\"passed\":true,\"checked_values\":%d,\"chain_launches\":%d,\"callback_count\":1,"
      "\"queries\":%d}\n",
      2 * runs, runs - 1, queries.load());
}
