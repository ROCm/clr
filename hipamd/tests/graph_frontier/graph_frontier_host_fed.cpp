// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
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
__global__ void caller_released(int* ready, int* output, int slot) {
  while (__hip_atomic_load(ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0)
    __builtin_amdgcn_s_sleep(1);
  if (threadIdx.x == 0) output[slot] = slot + 31;
}
int main() {
  constexpr int launches = 8;
  H(hipSetDevice(0));
  int *host_ready, *device_ready, *output;
  H(hipHostMalloc(&host_ready, sizeof(int), hipHostMallocMapped | hipHostMallocCoherent));
  H(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_ready), host_ready, 0));
  __atomic_store_n(host_ready, 0, __ATOMIC_RELEASE);
  H(hipMalloc(&output, 2 * launches * sizeof(int)));
  H(hipMemset(output, 0, 2 * launches * sizeof(int)));
  hipStream_t stream;
  H(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipGraph_t graph;
  hipGraphExec_t exec;
  hipGraphNode_t nodes[2];
  H(hipGraphCreate(&graph, 0));
  for (int branch = 0; branch < 2; ++branch) {
    int slot = branch;
    void* args[] = {&device_ready, &output, &slot};
    hipKernelNodeParams p{};
    p.func = (void*)caller_released;
    p.gridDim = dim3(1);
    p.blockDim = dim3(64);
    p.kernelParams = args;
    H(hipGraphAddKernelNode(&nodes[branch], graph, nullptr, 0, &p));
  }
  H(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  std::mutex mutex;
  std::condition_variable changed;
  bool submitted = false;
  std::atomic<bool> rescued{false};
  // Rescue deliberately blocked implementations; this is a semantic control,
  // never a performance run and never an indefinitely stuck GPU allocation.
  std::thread watchdog([&] {
    std::unique_lock<std::mutex> lock(mutex);
    if (!changed.wait_for(lock, std::chrono::seconds(2), [&] { return submitted; })) {
      rescued.store(true);
      __atomic_store_n(host_ready, 1, __ATOMIC_RELEASE);
    }
  });
  auto begin = std::chrono::steady_clock::now();
  for (int run = 0; run < launches; ++run) {
    for (int branch = 0; branch < 2; ++branch) {
      int slot = 2 * run + branch;
      void* args[] = {&device_ready, &output, &slot};
      hipKernelNodeParams p{};
      p.func = (void*)caller_released;
      p.gridDim = dim3(1);
      p.blockDim = dim3(64);
      p.kernelParams = args;
      H(hipGraphExecKernelNodeSetParams(exec, nodes[branch], &p));
    }
    H(hipGraphLaunch(exec, stream));
  }
  auto end = std::chrono::steady_clock::now();
  // This action is intentionally AFTER all launch calls on the submitting host.
  __atomic_store_n(host_ready, 1, __ATOMIC_RELEASE);
  {
    std::lock_guard<std::mutex> lock(mutex);
    submitted = true;
  }
  changed.notify_all();
  watchdog.join();
  H(hipGraphExecDestroy(exec));
  H(hipGraphDestroy(graph));
  H(hipStreamSynchronize(stream));
  int result[2 * launches];
  H(hipMemcpy(result, output, sizeof(result), hipMemcpyDeviceToHost));
  for (int i = 0; i < 2 * launches; ++i) R(result[i] == i + 31);
  H(hipStreamDestroy(stream));
  H(hipFree(output));
  H(hipHostFree(host_ready));
  double ms = std::chrono::duration<double, std::milli>(end - begin).count();
  std::printf(
      "{\"passed\":true,\"checked_values\":16,\"launches\":8,\"submit_ms\":%.6f,\"watchdog_"
      "rescued\":%s}\n",
      ms, rescued.load() ? "true" : "false");
}
