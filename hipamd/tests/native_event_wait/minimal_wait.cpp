// MIT License
// 
// Copyright (C) Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// The core reproduction: no host transfers, frameworks, or model state.
// hipcc -O3 -std=c++17 minimal_wait.cpp -o minimal_wait
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#define CHECK(call) do { auto check_status = (call); if (check_status != hipSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(check_status)); std::exit(1); \
} } while (0)
__global__ void increment(unsigned* value) { ++*value; }
int main() {
  int count;
  CHECK(hipGetDeviceCount(&count));
  if (count != 1) { std::fprintf(stderr, "Require one allocated GPU\n"); return 1; }
  std::set<std::string> hip_paths, hsa_paths;
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    auto pos = line.find('/');
    if (pos == std::string::npos) continue;
    auto path = line.substr(pos);
    if (path.find("libamdhip64.so") != std::string::npos) hip_paths.insert(path);
    if (path.find("libhsa-runtime64.so") != std::string::npos) hsa_paths.insert(path);
  }
  if (hip_paths.size() != 1 || hsa_paths.size() != 1) {
    std::fprintf(stderr, "Missing or duplicate HIP/HSA mappings\n");
    return 1;
  }
  int runtime_version = 0;
  CHECK(hipRuntimeGetVersion(&runtime_version));
  std::fprintf(stderr, "RUNTIME_IDENTITY_CPP hip=%s hsa=%s version=%d\n",
               hip_paths.begin()->c_str(), hsa_paths.begin()->c_str(), runtime_version);
  hipStream_t compute, side;
  CHECK(hipStreamCreateWithFlags(&compute, hipStreamNonBlocking));
  CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
  hipEvent_t begin, end;
  CHECK(hipEventCreate(&begin));
  CHECK(hipEventCreate(&end));
  unsigned* value;
  CHECK(hipMalloc(&value, sizeof(unsigned)));
  hipGraph_t graph;
  hipGraphExec_t executable;
  CHECK(hipStreamBeginCapture(compute, hipStreamCaptureModeGlobal));
  for (int i = 0; i < 2048; ++i) increment<<<1, 1, 0, compute>>>(value);
  CHECK(hipStreamEndCapture(compute, &graph));
  CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  const char* names[] = {"alone", "pending_wait", "ready_wait"};
  std::puts("block,case,gpu_us,pending,correct");
  for (int block = -1; block < 5; ++block) {
    for (int j = 0; j < 3; ++j) {
      int mode = (j + block + 3) % 3;
      for (int repeat = 0; repeat < 8; ++repeat) {
        CHECK(hipMemsetAsync(value, 0, sizeof(unsigned), compute));
        CHECK(hipStreamSynchronize(compute));
        CHECK(hipEventRecord(begin, compute));
        CHECK(hipGraphLaunch(executable, compute));
        CHECK(hipEventRecord(end, compute));
        bool pending = false;
        if (mode == 1) {
          auto status = hipEventQuery(end);
          if (status != hipSuccess && status != hipErrorNotReady) CHECK(status);
          pending = status == hipErrorNotReady;
          CHECK(hipStreamWaitEvent(side, end, 0));
        }
        CHECK(hipEventSynchronize(end));
        if (mode == 2) CHECK(hipStreamWaitEvent(side, end, 0));
        CHECK(hipStreamSynchronize(side)); // No dependency survives into next trial.
        float ms;
        CHECK(hipEventElapsedTime(&ms, begin, end));
        unsigned result;
        CHECK(hipMemcpy(&result, value, sizeof(result), hipMemcpyDeviceToHost));
        if (result != 2048) return 2;
        if (block >= 0) std::printf("%d,%s,%.3f,%d,1\n", block, names[mode], ms*1000, pending);
      }
    }
  }
  CHECK(hipGraphExecDestroy(executable));
  CHECK(hipGraphDestroy(graph));
  CHECK(hipEventDestroy(begin)); CHECK(hipEventDestroy(end));
  CHECK(hipStreamDestroy(compute)); CHECK(hipStreamDestroy(side));
  CHECK(hipFree(value));
}
