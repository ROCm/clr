// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>
#define CHECK(x)                                             \
  do {                                                       \
    auto e = (x);                                            \
    if (e != hipSuccess) {                                   \
      fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); \
      exit(1);                                               \
    }                                                        \
  } while (0)
__global__ void step(unsigned* x, unsigned ticks) {
  unsigned long long t = clock64();
  while (clock64() - t < ticks) {
  }
  ++*x;
}
int main() {
  int devices = 0;
  CHECK(hipGetDeviceCount(&devices));
  if (devices != 1) return 2;
  std::ifstream maps("/proc/self/maps");
  std::string line;
  std::set<std::string> libs;
  while (std::getline(maps, line))
    if (line.find("libamdhip64.so") != std::string::npos ||
        line.find("libhsa-runtime64.so") != std::string::npos)
      libs.insert(line.substr(line.find('/')));
  if (libs.size() != 2) return 3;
  for (auto& p : libs) fprintf(stderr, "MAPPED %s\n", p.c_str());
  hipStream_t stream;
  CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  unsigned* data;
  CHECK(hipMalloc(&data, 2 * sizeof(unsigned)));
  hipEvent_t begin, end;
  CHECK(hipEventCreate(&begin));
  CHECK(hipEventCreate(&end));
  puts("lanes,group,ticks,trial,gpu_us,host_us,submit_us,correct");
  for (int lanes : {1, 2})
    for (int group : {1, 50})
      for (unsigned ticks : {0u, 2000u, 15000u}) {
        hipGraph_t graph;
        hipGraphExec_t exec;
        CHECK(hipGraphCreate(&graph, 0));
        for (int lane = 0; lane < lanes; ++lane) {
          hipGraphNode_t previous = nullptr;
          for (int i = 0; i < group; ++i) {
            hipKernelNodeParams p{};
            auto ptr = data + lane;
            void* args[] = {&ptr, &ticks};
            p.func = (void*)step;
            p.gridDim = dim3(1);
            p.blockDim = dim3(1);
            p.kernelParams = args;
            hipGraphNode_t node;
            CHECK(hipGraphAddKernelNode(&node, graph, previous ? &previous : nullptr,
                                        previous ? 1 : 0, &p));
            previous = node;
          }
        }
        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        const int launches = 1000 / group;
        for (int trial = -2; trial < 8; ++trial) {
          CHECK(hipMemsetAsync(data, 0, 2 * sizeof(unsigned), stream));
          CHECK(hipStreamSynchronize(stream));
          CHECK(hipEventRecord(begin, stream));
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < launches; ++i) CHECK(hipGraphLaunch(exec, stream));
          auto t1 = std::chrono::steady_clock::now();
          CHECK(hipEventRecord(end, stream));
          CHECK(hipEventSynchronize(end));
          auto t2 = std::chrono::steady_clock::now();
          float ms;
          CHECK(hipEventElapsedTime(&ms, begin, end));
          unsigned got[2];
          CHECK(hipMemcpy(got, data, sizeof(got), hipMemcpyDeviceToHost));
          bool correct = got[0] == 1000 && got[1] == (lanes == 2 ? 1000u : 0u);
          if (!correct) return 4;
          if (trial >= 0)
            printf("%d,%d,%u,%d,%.6f,%.6f,%.6f,1\n", lanes, group, ticks, trial, ms,
                   std::chrono::duration<double, std::micro>(t2 - t0).count() / 1000,
                   std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000);
        }
        CHECK(hipGraphExecDestroy(exec));
        CHECK(hipGraphDestroy(graph));
      }
  CHECK(hipEventDestroy(begin));
  CHECK(hipEventDestroy(end));
  CHECK(hipFree(data));
  CHECK(hipStreamDestroy(stream));
}
