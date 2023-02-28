/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef HIP_SRC_HIP_INTERNAL_H
#define HIP_SRC_HIP_INTERNAL_H

#include "vdi_common.hpp"
#include "hip_prof_api.h"
#include "trace_helper.h"
#include "utils/debug.hpp"
#include "hip_formatting.hpp"
#include "hip_graph_capture.hpp"

#include <unordered_set>
#include <thread>
#include <stack>
#include <mutex>
#include <iterator>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

/*! IHIP IPC MEMORY Structure */
#define IHIP_IPC_MEM_HANDLE_SIZE   32
#define IHIP_IPC_MEM_RESERVED_SIZE LP64_SWITCH(20,12)

typedef struct ihipIpcMemHandle_st {
  char ipc_handle[IHIP_IPC_MEM_HANDLE_SIZE];  ///< ipc memory handle on ROCr
  size_t psize;
  size_t poffset;
  int owners_process_id;
  char reserved[IHIP_IPC_MEM_RESERVED_SIZE];
} ihipIpcMemHandle_t;

#define IHIP_IPC_EVENT_HANDLE_SIZE 32
#define IHIP_IPC_EVENT_RESERVED_SIZE LP64_SWITCH(28,24)
typedef struct ihipIpcEventHandle_st {
    //hsa_amd_ipc_signal_t ipc_handle;  ///< ipc signal handle on ROCr
    //char ipc_handle[IHIP_IPC_EVENT_HANDLE_SIZE];
    //char reserved[IHIP_IPC_EVENT_RESERVED_SIZE];
    char shmem_name[IHIP_IPC_EVENT_HANDLE_SIZE];
}ihipIpcEventHandle_t;

const char* ihipGetErrorName(hipError_t hip_error);

static  amd::Monitor g_hipInitlock{"hipInit lock"};
#define HIP_INIT(noReturn) {\
    amd::ScopedLock lock(g_hipInitlock);                     \
    if (!amd::Runtime::initialized()) {                      \
      if (!hip::init() && !noReturn) {                       \
        HIP_RETURN(hipErrorInvalidDevice);                   \
      }                                                      \
    }                                                        \
    if (hip::tls.device_ == nullptr && g_devices.size() > 0) {  \
      hip::tls.device_ = g_devices[0];                          \
      amd::Os::setPreferredNumaNode(g_devices[0]->devices()[0]->getPreferredNumaNode());  \
    }                                                        \
  }

#define HIP_INIT_VOID() {\
    amd::ScopedLock lock(g_hipInitlock);                     \
    if (!amd::Runtime::initialized()) {                      \
      if (hip::init()) {}                                    \
    }                                                        \
    if (hip::tls.device_ == nullptr && g_devices.size() > 0) {  \
      hip::tls.device_ = g_devices[0];                          \
      amd::Os::setPreferredNumaNode(g_devices[0]->devices()[0]->getPreferredNumaNode());  \
    }                                                        \
  }


#define HIP_API_PRINT(...)                                          \
  uint64_t startTimeUs=0;                                           \
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs,       \
                  "%s %s ( %s ) %s", KGRN,                          \
                  __func__, ToString( __VA_ARGS__ ).c_str(), KNRM);

#define HIP_ERROR_PRINT(err, ...)                                                  \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s : %s",                     \
          __func__, ihipGetErrorName(err), ToString( __VA_ARGS__ ).c_str());

#define HIP_INIT_API_INTERNAL(noReturn, cid, ...)            \
  HIP_API_PRINT(__VA_ARGS__)                                 \
  amd::Thread* thread = amd::Thread::current();              \
  if (!VDI_CHECK_THREAD(thread) && !noReturn) {              \
    HIP_RETURN(hipErrorOutOfMemory);                         \
  }                                                          \
  HIP_INIT(noReturn)                                         \
  HIP_CB_SPAWNER_OBJECT(cid);

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(cid, ...)                                                                     \
  HIP_INIT_API_INTERNAL(0, cid, __VA_ARGS__)                                                       \
  if (g_devices.size() == 0) {                                                                     \
    HIP_RETURN(hipErrorNoDevice);                                                                  \
  }

#define HIP_INIT_API_NO_RETURN(cid, ...)                     \
  HIP_INIT_API_INTERNAL(1, cid, __VA_ARGS__)

#define HIP_RETURN_DURATION(ret, ...)                        \
  hip::tls.last_error_ = ret;                                \
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs,                      \
                   "%s: Returned %s : %s",                                         \
                   __func__, ihipGetErrorName(hip::tls.last_error_),               \
                   ToString( __VA_ARGS__ ).c_str());                               \
  return hip::tls.last_error_;

#define HIP_RETURN(ret, ...)                      \
  hip::tls.last_error_ = ret;                         \
  HIP_ERROR_PRINT(hip::tls.last_error_, __VA_ARGS__)  \
  return hip::tls.last_error_;

#define HIP_RETURN_ONFAIL(func)          \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      HIP_RETURN(herror);                \
    }                                    \
  } while (0);

// Cannot be use in place of HIP_RETURN.
// Refrain from using for external HIP APIs
#define IHIP_RETURN_ONFAIL(func)         \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      return herror;                     \
    }                                    \
  } while (0);

#define CHECK_STREAM_CAPTURE_SUPPORTED()                                                           \
  if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeThreadLocal) {                          \
    if (hip::tls.capture_streams_.size() != 0) {                                                   \
      HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                \
    }                                                                                              \
  } else if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeGlobal) {                        \
    if (hip::tls.capture_streams_.size() != 0) {                                                   \
      HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                \
    }                                                                                              \
    amd::ScopedLock lock(g_captureStreamsLock);                                                    \
    if (g_captureStreams.size() != 0) {                                                            \
      HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                \
    }                                                                                              \
  }

// Sync APIs cannot be called when stream capture is active
#define CHECK_STREAM_CAPTURING()                                                                   \
  if (!g_captureStreams.empty()) {                                                                 \
    return hipErrorStreamCaptureImplicit;                                                          \
  }

#define STREAM_CAPTURE(name, stream, ...)                                                          \
  getStreamPerThread(stream);                                                                      \
  if (stream != nullptr &&                                                                         \
      reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus() ==                                \
          hipStreamCaptureStatusActive) {                                                          \
    hipError_t status = capture##name(stream, ##__VA_ARGS__);                                      \
    return status;                                                                                 \
  }

#define EVENT_CAPTURE(name, event, ...)                                                            \
  if (event != nullptr && reinterpret_cast<hip::Event*>(event)->GetCaptureStatus() == true) {      \
    hipError_t status = capture##name(event, ##__VA_ARGS__);                                       \
    HIP_RETURN(status);                                                                            \
  }

#define PER_THREAD_DEFAULT_STREAM(stream)                                                         \
  if (stream == nullptr) {                                                                        \
    stream = getPerThreadDefaultStream();                                                         \
  }

namespace hc {
class accelerator;
class accelerator_view;
};

struct ihipExec_t {
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  hipStream_t hStream_;
  std::vector<char> arguments_;
};

class stream_per_thread {
private:
  std::vector<hipStream_t> m_streams;
public:
  stream_per_thread();
  stream_per_thread(const stream_per_thread& ) = delete;
  void operator=(const stream_per_thread& ) = delete;
  ~stream_per_thread();
  hipStream_t get();
};

namespace hip {
  class Device;
  class MemoryPool;
  class Stream : public amd::HostQueue {
  public:
    enum Priority : int { High = -1, Normal = 0, Low = 1 };

  private:
    mutable amd::Monitor lock_;
    Device* device_;
    Priority priority_;
    unsigned int flags_;
    bool null_;
    const std::vector<uint32_t> cuMask_;

    /// Stream capture related parameters

    /// Current capture status of the stream
    hipStreamCaptureStatus captureStatus_;
    /// Graph that is constructed with capture
    hipGraph_t pCaptureGraph_;
    /// Based on mode stream capture places restrictions on API calls that can be made within or
    /// concurrently
    hipStreamCaptureMode captureMode_{hipStreamCaptureModeGlobal};
    bool originStream_;
    /// Origin sream has no parent. Parent stream for the derived captured streams with event
    /// dependencies
    hipStream_t parentStream_ = nullptr;
    /// Last graph node captured in the stream
    std::vector<hipGraphNode_t> lastCapturedNodes_;
    /// dependencies removed via API hipStreamUpdateCaptureDependencies
    std::vector<hipGraphNode_t> removedDependencies_;
    /// Derived streams/Paralell branches from the origin stream
    std::vector<hipStream_t> parallelCaptureStreams_;
    /// Capture events
    std::unordered_set<hipEvent_t> captureEvents_;
    unsigned long long captureID_;

    static inline CommandQueue::Priority convertToQueuePriority(Priority p){
      return p == Priority::High ? amd::CommandQueue::Priority::High : p == Priority::Low ?
                    amd::CommandQueue::Priority::Low : amd::CommandQueue::Priority::Normal;
    }

  public:
    Stream(Device* dev, Priority p = Priority::Normal, unsigned int f = 0, bool null_stream = false,
           const std::vector<uint32_t>& cuMask = {},
           hipStreamCaptureStatus captureStatus = hipStreamCaptureStatusNone);

    /// Creates the hip stream object, including AMD host queue
    bool Create();
    virtual bool terminate() override;
    /// Get device ID associated with the current stream;
    int DeviceId() const;
    /// Get HIP device associated with the stream
    Device* GetDevice() const { return device_; }
    /// Get device ID associated with a stream;
    static int DeviceId(const hipStream_t hStream);
    /// Returns if stream is null stream
    bool Null() const { return null_; }
    /// Returns the lock object for the current stream
    amd::Monitor& Lock() const { return lock_; }
    /// Returns the creation flags for the current stream
    unsigned int Flags() const { return flags_; }
    /// Returns the priority for the current stream
    Priority GetPriority() const { return priority_; }
    /// Returns the CU mask for the current stream
    const std::vector<uint32_t> GetCUMask() const { return cuMask_; }

    /// Sync all non-blocking streams
    static void syncNonBlockingStreams(int deviceId);

    /// Check whether any blocking stream running
    static bool StreamCaptureBlocking();

    /// Destroy all streams on a given device
    static void destroyAllStreams(int deviceId);

    /// Check Stream Capture status to make sure it is done
    static bool StreamCaptureOngoing(void);

    /// Returns capture status of the current stream
    hipStreamCaptureStatus GetCaptureStatus() const { return captureStatus_; }
    /// Returns capture mode of the current stream
    hipStreamCaptureMode GetCaptureMode() const { return captureMode_; }
    /// Returns if stream is origin stream
    bool IsOriginStream() const { return originStream_; }
    void SetOriginStream() { originStream_ = true; }
    /// Returns captured graph
    hipGraph_t GetCaptureGraph() const { return pCaptureGraph_; }
    /// Returns last captured graph node
    const std::vector<hipGraphNode_t>& GetLastCapturedNodes() const { return lastCapturedNodes_; }
    /// Set last captured graph node
    void SetLastCapturedNode(hipGraphNode_t graphNode) {
      lastCapturedNodes_.clear();
      lastCapturedNodes_.push_back(graphNode);
    }
    /// returns updated dependencies removed
    const std::vector<hipGraphNode_t>& GetRemovedDependencies() {
      return removedDependencies_;
    }
    /// Append captured node via the wait event cross stream
    void AddCrossCapturedNode(std::vector<hipGraphNode_t> graphNodes, bool replace = false) {
      // replace dependencies as per flag hipStreamSetCaptureDependencies
      if (replace == true) {
        for (auto node : lastCapturedNodes_) {
          removedDependencies_.push_back(node);
        }
        lastCapturedNodes_.clear();
      }
      for (auto node : graphNodes) {
        lastCapturedNodes_.push_back(node);
      }
    }
    /// Set graph that is being captured
    void SetCaptureGraph(hipGraph_t pGraph) {
      pCaptureGraph_ = pGraph;
      captureStatus_ = hipStreamCaptureStatusActive;
    }
    void SetCaptureId() {
      // ID is generated in Begin Capture i.e.. when capture status is active
      captureID_ = GenerateCaptureID();
    }
    void SetCaptureId(unsigned long long captureId) {
      // ID is given from parent stream
      captureID_ = captureId;
    }
    /// reset capture parameters
    hipError_t EndCapture();
    /// Set capture status
    void SetCaptureStatus(hipStreamCaptureStatus captureStatus) { captureStatus_ = captureStatus; }
    /// Set capture mode
    void SetCaptureMode(hipStreamCaptureMode captureMode) { captureMode_ = captureMode; }
    /// Set parent stream
    void SetParentStream(hipStream_t parentStream) { parentStream_ = parentStream; }
    /// Get parent stream
    hipStream_t GetParentStream() const { return parentStream_; }
    /// Generate ID for stream capture unique over the lifetime of the process
    static unsigned long long GenerateCaptureID() {
      static std::atomic<unsigned long long> uid(0);
      return ++uid;
    }
    /// Get Capture ID
    unsigned long long GetCaptureID() { return captureID_; }
    void SetCaptureEvent(hipEvent_t e) { captureEvents_.emplace(e); }
    void EraseCaptureEvent(hipEvent_t e) {
      auto it = captureEvents_.find(e);
      if (it != captureEvents_.end()) {
        captureEvents_.erase(it);
      }
    }
    void SetParallelCaptureStream(hipStream_t s) { parallelCaptureStreams_.push_back(s); }
    void EraseParallelCaptureStream(hipStream_t s) {
      auto it = std::find(parallelCaptureStreams_.begin(), parallelCaptureStreams_.end(), s);
      if (it != parallelCaptureStreams_.end()) {
        parallelCaptureStreams_.erase(it);
      }
    }
    static bool existsActiveStreamForDevice(hip::Device* device);

    /// The stream should be destroyed via release() rather than delete
    private:
      ~Stream() {};
  };

  /// HIP Device class
  class Device {
    amd::Monitor lock_{"Device lock", true};
    /// ROCclr context
    amd::Context* context_;
    /// Device's ID
    /// Store it here so we don't have to loop through the device list every time
    int deviceId_;
    /// ROCclr host queue for default streams
    Stream* null_stream_  = nullptr;
    /// Store device flags
    unsigned int flags_;
    /// Maintain list of user enabled peers
    std::list<int> userEnabledPeers;

    /// True if this device is active
    bool isActive_;


    MemoryPool* default_mem_pool_;  //!< Default memory pool for this device
    MemoryPool* current_mem_pool_;
    MemoryPool* graph_mem_pool_;    //!< Memory pool, associated with graphs for this device

    std::set<MemoryPool*> mem_pools_;

  public:
    Device(amd::Context* ctx, int devId): context_(ctx),
        deviceId_(devId),
         flags_(hipDeviceScheduleSpin),
        isActive_(false),
        default_mem_pool_(nullptr),
        current_mem_pool_(nullptr),
        graph_mem_pool_(nullptr)
        { assert(ctx != nullptr); }
    ~Device();

    bool Create();
    amd::Context* asContext() const { return context_; }
    int deviceId() const { return deviceId_; }
    void retain() const { context_->retain(); }
    void release() const { context_->release(); }
    const std::vector<amd::Device*>& devices() const { return context_->devices(); }
    hipError_t EnablePeerAccess(int peerDeviceId){
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        return hipErrorPeerAccessAlreadyEnabled;
      }
      userEnabledPeers.push_back(peerDeviceId);
      return hipSuccess;
    }
    hipError_t DisablePeerAccess(int peerDeviceId) {
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        userEnabledPeers.remove(peerDeviceId);
        return hipSuccess;
      } else {
        return hipErrorPeerAccessNotEnabled;
      }
    }
    unsigned int getFlags() const { return flags_; }
    void setFlags(unsigned int flags) { flags_ = flags; }
    void Reset();

   hip::Stream* NullStream(bool skip_alloc = false);
   Stream* GetNullStream();


    bool GetActiveStatus() {
      amd::ScopedLock lock(lock_);
      if (isActive_) return true;
      if (Stream::existsActiveStreamForDevice(this)) {
        isActive_ = true;
        return true;
      }
      return false;
    }

    /// Set the current memory pool on the device
    void SetCurrentMemoryPool(MemoryPool* pool = nullptr) {
      current_mem_pool_ = (pool == nullptr) ? default_mem_pool_ : pool;
    }

    /// Get the current memory pool on the device
    MemoryPool* GetCurrentMemoryPool() const { return current_mem_pool_; }

    /// Get the default memory pool on the device
    MemoryPool* GetDefaultMemoryPool() const { return default_mem_pool_; }

    /// Get the graph memory pool on the device
    MemoryPool* GetGraphMemoryPool() const { return graph_mem_pool_; }

    /// Add memory pool to the device
    void AddMemoryPool(MemoryPool* pool);

    /// Remove memory pool from the device
    void RemoveMemoryPool(MemoryPool* pool);

    /// Free memory from the device
    bool FreeMemory(amd::Memory* memory, Stream* stream);

    /// Release freed memory from all pools on the current device
    void ReleaseFreedMemory(Stream* stream);

    /// Removes a destroyed stream from the safe list of memory pools
    void RemoveStreamFromPools(Stream* stream);

  };

  /// Thread Local Storage Variables Aggregator Class
  class TlsAggregator {
  public:
    Device* device_;
    std::stack<Device*> ctxt_stack_;
    hipError_t last_error_;
    std::vector<hip::Stream*> capture_streams_;
    hipStreamCaptureMode stream_capture_mode_;
    std::stack<ihipExec_t> exec_stack_;
    stream_per_thread stream_per_thread_obj_;

    TlsAggregator(): device_(nullptr),
      last_error_(hipSuccess),
      stream_capture_mode_(hipStreamCaptureModeGlobal) {
    }
    ~TlsAggregator() {
    }
  };
  extern thread_local TlsAggregator tls;

  /// Device representing the host - for pinned memory
  extern amd::Context* host_context;

  extern bool init();

  extern Device* getCurrentDevice();

  extern void setCurrentDevice(unsigned int index);

  /// Get ROCclr queue associated with hipStream
  /// Note: This follows the CUDA spec to sync with default streams
  ///       and Blocking streams
  extern hip::Stream* getStream(hipStream_t stream);
  /// Get default stream associated with the ROCclr context
  extern hip::Stream* getNullStream(amd::Context&);
  /// Get default stream of the thread
  extern hip::Stream* getNullStream();
  /// Get device ID associated with the ROCclr context
  int getDeviceID(amd::Context& ctx);
  /// Check if stream is valid
  extern bool isValid(hipStream_t& stream);
  extern bool isValid(hipEvent_t event);
  extern amd::Monitor hipArraySetLock;
  extern std::unordered_set<hipArray*> hipArraySet;
}; // namespace hip

extern void WaitThenDecrementSignal(hipStream_t stream, hipError_t status, void* user_data);

/// Wait all active streams on the blocking queue. The method enqueues a wait command and
/// doesn't stall the current thread
extern void iHipWaitActiveStreams(hip::Stream* blocking_stream, bool wait_null_stream = false);

extern std::vector<hip::Device*> g_devices;
extern hipError_t ihipDeviceGetCount(int* count);
extern int ihipGetDevice();

extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset, size_t size = 0);
extern amd::Memory* getMemoryObjectWithOffset(const void* ptr, const size_t size = 0);
extern void getStreamPerThread(hipStream_t& stream);
extern hipStream_t getPerThreadDefaultStream();
extern hipError_t ihipUnbindTexture(textureReference* texRef);
extern hipError_t ihipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
extern hipError_t ihipHostUnregister(void* hostPtr);
extern hipError_t ihipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device);

extern hipError_t ihipDeviceGet(hipDevice_t* device, int deviceId);
extern hipError_t ihipStreamOperation(hipStream_t stream, cl_command_type cmdType, void* ptr,
                                      uint64_t value, uint64_t mask, unsigned int flags, size_t sizeBytes);

constexpr bool kOptionChangeable = true;
constexpr bool kNewDevProg = false;

constexpr bool kMarkerDisableFlush = true;   //!< Avoids command batch flush in ROCclr

extern std::vector<hip::Stream*> g_captureStreams;
extern amd::Monitor g_captureStreamsLock;
extern std::unordered_set<hip::Stream*> g_allCapturingStreams;
#endif // HIP_SRC_HIP_INTERNAL_H
