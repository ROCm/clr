// Bounded diagnostic: keep one valid event pending while submitting many waits.
// The host-controlled gate diagnoses enqueue progress; it is not an application API change.
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#define HIP(call) do { auto error=(call); if(error!=hipSuccess) { std::fprintf(stderr,"HIP line %d: %s\n",__LINE__,hipGetErrorString(error)); std::abort(); } } while(0)
using Clock=std::chrono::steady_clock;
__global__ void hold(volatile unsigned* gate) {
  gate[1]=1;
  __threadfence_system();
  while(gate[0]==0) __builtin_amdgcn_s_sleep(1);
}
__global__ void increment(unsigned* value) { ++*value; }
int main() {
  int devices=0, version=0; HIP(hipGetDeviceCount(&devices)); HIP(hipRuntimeGetVersion(&version));
  if(devices!=1 || version!=70253211) return 3;
  std::set<std::string> hip_paths,hsa_paths;
  std::ifstream maps("/proc/self/maps"); std::string line;
  while(std::getline(maps,line)) {
    auto pos=line.find('/'); if(pos==std::string::npos) continue;
    auto path=line.substr(pos);
    if(path.find("libamdhip64.so")!=std::string::npos) hip_paths.insert(path);
    if(path.find("libhsa-runtime64.so")!=std::string::npos) hsa_paths.insert(path);
  }
  if(hip_paths.size()!=1 || hsa_paths.size()!=1) return 4;
  std::printf("RUNTIME_IDENTITY hip=%s hsa=%s version=%d\n",hip_paths.begin()->c_str(),hsa_paths.begin()->c_str(),version);
  unsigned* host_gate=nullptr; unsigned* device_gate=nullptr; unsigned* value=nullptr;
  HIP(hipHostMalloc(&host_gate,2*sizeof(unsigned),hipHostMallocMapped|hipHostMallocCoherent));
  HIP(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_gate),host_gate,0));
  host_gate[0]=host_gate[1]=0;
  HIP(hipMalloc(&value,sizeof(unsigned))); HIP(hipMemset(value,0,sizeof(unsigned)));
  hipStream_t producer,consumer; hipEvent_t ready;
  HIP(hipStreamCreateWithFlags(&producer,hipStreamNonBlocking));
  HIP(hipStreamCreateWithFlags(&consumer,hipStreamNonBlocking));
  HIP(hipEventCreateWithFlags(&ready,hipEventDisableTiming));
  hold<<<1,1,0,producer>>>(device_gate); HIP(hipGetLastError());
  auto start_deadline=Clock::now()+std::chrono::seconds(5);
  while(__atomic_load_n(host_gate+1,__ATOMIC_ACQUIRE)==0) {
    if(Clock::now()>start_deadline) { __atomic_store_n(host_gate,1u,__ATOMIC_RELEASE); return 5; }
    std::this_thread::yield();
  }
  std::atomic<bool> submitted{false},watchdog{false};
  std::thread release([&] {
    auto deadline=Clock::now()+std::chrono::seconds(2);
    while(!submitted.load(std::memory_order_acquire) && Clock::now()<deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if(!submitted.load(std::memory_order_acquire)) {
      watchdog.store(true);
      __atomic_store_n(host_gate,1u,__ATOMIC_RELEASE);
    }
  });
  HIP(hipEventRecord(ready,producer));
  constexpr int count=2300;
  unsigned pending=0; double max_submit_ms=0; int slowest=-1;
  auto begin=Clock::now();
  for(int i=0;i<count;++i) {
    if(hipEventQuery(ready)==hipErrorNotReady) ++pending;
    auto before=Clock::now();
    HIP(hipStreamWaitEvent(consumer,ready,0));
    increment<<<1,1,0,consumer>>>(value); HIP(hipGetLastError());
    double ms=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
    if(ms>max_submit_ms) {max_submit_ms=ms;slowest=i;}
  }
  double enqueue_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
  submitted.store(true,std::memory_order_release);
  __atomic_store_n(host_gate,1u,__ATOMIC_RELEASE);
  release.join();
  HIP(hipStreamSynchronize(consumer)); HIP(hipStreamSynchronize(producer));
  unsigned actual=0; HIP(hipMemcpy(&actual,value,sizeof(actual),hipMemcpyDeviceToHost));
  std::printf("POOL_PRESSURE mode=%s count=%d pending=%u watchdog=%d enqueue_ms=%.6f max_submit_ms=%.6f slowest=%d actual=%u\n",std::getenv("GPU_NATIVE_EVENT_WAIT"),count,pending,int(watchdog.load()),enqueue_ms,max_submit_ms,slowest,actual);
  HIP(hipEventDestroy(ready)); HIP(hipStreamDestroy(consumer)); HIP(hipStreamDestroy(producer));
  HIP(hipFree(value)); HIP(hipHostFree(host_gate));
  if(actual!=count) return 6;
  return watchdog.load() || pending!=count ? 7 : 0;
}
