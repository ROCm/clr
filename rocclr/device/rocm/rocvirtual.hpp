/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#pragma once

#include "platform/commandqueue.hpp"
#include "rocdefs.hpp"
#include "rocdevice.hpp"
#include "utils/util.hpp"
#include "hsa.h"
#include "hsa_ext_image.h"
#include "hsa_ext_amd.h"
#include "rocprintf.hpp"
#include "hsa_ven_amd_aqlprofile.h"
#include "rocsched.hpp"

namespace roc {
class Device;
class Memory;
class Timestamp;

struct ProfilingSignal : public amd::HeapObject {
  hsa_signal_t  signal_;  //!< HSA signal to track profiling information
  Timestamp*    ts_;      //!< Timestamp object associated with the signal
  HwQueueEngine engine_;  //!< Engine used with this signal
  bool          done_;    //!< True if signal is done
  ProfilingSignal()
    : ts_(nullptr)
    , engine_(HwQueueEngine::Compute)
    , done_(true)
    { signal_.handle = 0; }
};

// Initial HSA signal value
constexpr static hsa_signal_value_t kInitSignalValueOne = 1;

// Timeouts for HSA signal wait
constexpr static uint64_t kTimeout30us = 30000;
constexpr static uint64_t kTimeout50us = 50000;
constexpr static uint64_t kUnlimitedWait = std::numeric_limits<uint64_t>::max();

template <uint64_t wait_time = 0>
inline bool WaitForSignal(hsa_signal_t signal) {
  if (wait_time != 0) {
    if (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                  wait_time, HSA_WAIT_STATE_ACTIVE) != 0) {
      return false;
    }
  } else {
    uint64_t timeout = (ROC_ACTIVE_WAIT) ? kUnlimitedWait : kTimeout30us;

    // Active wait with a timeout
    if (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                  timeout, HSA_WAIT_STATE_ACTIVE) != 0) {
      // Wait until the completion with CPU suspend
      if (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                    kUnlimitedWait, HSA_WAIT_STATE_BLOCKED) != 0) {
        return false;
      }
    }
  }
  return true;
}

// Timestamp for keeping track of some profiling information for various commands
// including EnqueueNDRangeKernel and clEnqueueCopyBuffer.
class Timestamp : public amd::HeapObject {
 private:
  static double ticksToTime_;

  uint64_t    start_;
  uint64_t    end_;
  VirtualGPU* gpu_;               //!< Virtual GPU, associated with this timestamp
  const amd::Command& command_;   //!< Command, associated with this timestamp
  std::vector<ProfilingSignal*> signals_;

 public:
  Timestamp(VirtualGPU* gpu, const amd::Command& command)
    : start_(std::numeric_limits<uint64_t>::max())
    , end_(0)
    , gpu_(gpu)
    , command_(command) {}

  ~Timestamp() {}

  uint64_t getStart() {
    checkGpuTime();
    return start_;
  }

  uint64_t getEnd() {
    checkGpuTime();
    return end_;
  }

  void AddProfilingSignal(ProfilingSignal* signal) { signals_.push_back(signal); }

  const std::vector<ProfilingSignal*>& Signals() const { return signals_; }

  const bool HwProfiling() const { return !signals_.empty(); }

  //! Finds execution ticks on GPU
  void checkGpuTime();

  // Start a timestamp (get timestamp from OS)
  void start() { start_ = amd::Os::timeNanos(); }

  // End a timestamp (get timestamp from OS)
  void end() { end_ = amd::Os::timeNanos(); }

  static void setGpuTicksToTime(double ticksToTime) { ticksToTime_ = ticksToTime; }
  static double getGpuTicksToTime() { return ticksToTime_; }

  //! Returns amd::command assigned to this timestamp
  const amd::Command& command() const { return command_; }

  //! Returns virtual GPU device, used with this timestamp
  VirtualGPU* gpu() const { return gpu_; }
};

class VirtualGPU : public device::VirtualDevice {
 public:
  class MemoryDependency : public amd::EmbeddedObject {
   public:
    //! Default constructor
    MemoryDependency()
        : memObjectsInQueue_(nullptr), numMemObjectsInQueue_(0), maxMemObjectsInQueue_(0) {}

    ~MemoryDependency() { delete[] memObjectsInQueue_; }

    //! Creates memory dependecy structure
    bool create(size_t numMemObj);

    //! Notify the tracker about new kernel
    void newKernel() { endMemObjectsInQueue_ = numMemObjectsInQueue_; }

    //! Validates memory object on dependency
    void validate(VirtualGPU& gpu, const Memory* memory, bool readOnly);

    //! Clear memory dependency
    void clear(bool all = true);

    //! Max number of mem objects in the queue
    size_t maxMemObjectsInQueue() const { return maxMemObjectsInQueue_; }

   private:
    struct MemoryState {
      uint64_t start_;  //! Busy memory start address
      uint64_t end_;    //! Busy memory end address
      bool readOnly_;   //! Current GPU state in the queue
    };

    MemoryState* memObjectsInQueue_;  //!< Memory object state in the queue
    size_t endMemObjectsInQueue_;     //!< End of mem objects in the queue
    size_t numMemObjectsInQueue_;     //!< Number of mem objects in the queue
    size_t maxMemObjectsInQueue_;     //!< Maximum number of mem objects in the queue
  };

  class HwQueueTracker : public amd::EmbeddedObject {
   public:
    HwQueueTracker(const VirtualGPU& gpu): gpu_(gpu) {}

    ~HwQueueTracker();

    //! Creates a pool of signals for tracking of HW operations on the queue
    bool Create();

    //! Finds a free signal for the upcomming operation
    hsa_signal_t ActiveSignal(hsa_signal_value_t init_val = kInitSignalValueOne,
                              Timestamp* ts = nullptr, uint32_t queue_size = 0);

    //! Wait for the curent active signal. Can idle the queue
    bool WaitCurrent() { return CpuWaitForSignal(signal_list_[current_id_]); }

    //! Update current active engine
    void SetActiveEngine(HwQueueEngine engine = HwQueueEngine::Compute) { engine_ = engine; }

    //! Returns the last submitted signal for a wait
    hsa_signal_t* WaitingSignal(HwQueueEngine engine = HwQueueEngine::Compute);

    //! Resets current signal back to the previous one. It's necessary in a case of ROCr failure.
    void ResetCurrentSignal();

    //! Inserts an external signal(submission in another queue) for dependency tracking
    void SetExternalSignal(ProfilingSignal* signal) {
      external_signal_ = signal;
      engine_ = HwQueueEngine::External;
    }

    //! Inserts an external signal(submission in another queue) for dependency tracking
    ProfilingSignal* GetLastSignal() const { return signal_list_[current_id_]; }

  private:
    //! Wait for the next active signal
    void WaitNext() {
      size_t next = (current_id_ + 1) % signal_list_.size();
      CpuWaitForSignal(signal_list_[next]);
    }

    //! Wait for the provided signal
    bool CpuWaitForSignal(ProfilingSignal* signal);

    HwQueueEngine engine_ = HwQueueEngine::Unknown; //!< Engine used in the current operations
    std::vector<ProfilingSignal*> signal_list_;     //!< The pool of all signals for processing
    ProfilingSignal*  external_signal_ = nullptr;   //!< Dependency on external signal
    size_t current_id_ = 0;       //!< Last submitted signal
    bool sdma_profiling_ = false; //!< If TRUE, then SDMA profiling is enabled
    const VirtualGPU& gpu_;       //!< VirtualGPU, associated with this tracker
  };

  VirtualGPU(Device& device, bool profiling = false, bool cooperative = false,
             const std::vector<uint32_t>& cuMask = {},
             amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal);
  ~VirtualGPU();

  bool create();
  const Device& dev() const { return roc_device_; }

  void profilingBegin(amd::Command& command, bool drmProfiling = false);
  void profilingEnd(amd::Command& command);

  void updateCommandsState(amd::Command* list) const;

  void submitReadMemory(amd::ReadMemoryCommand& cmd);
  void submitWriteMemory(amd::WriteMemoryCommand& cmd);
  void submitCopyMemory(amd::CopyMemoryCommand& cmd);
  void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd);
  void submitMapMemory(amd::MapMemoryCommand& cmd);
  void submitUnmapMemory(amd::UnmapMemoryCommand& cmd);
  void submitKernel(amd::NDRangeKernelCommand& cmd);
  bool submitKernelInternal(const amd::NDRangeContainer& sizes,  //!< Workload sizes
                            const amd::Kernel& kernel,           //!< Kernel for execution
                            const_address parameters,            //!< Parameters for the kernel
                            void* event_handle,  //!< Handle to OCL event for debugging
                            uint32_t sharedMemBytes = 0, //!< Shared memory size
                            amd::NDRangeKernelCommand* vcmd = nullptr //!< Original launch command
                            );
  void submitNativeFn(amd::NativeFnCommand& cmd);
  void submitMarker(amd::Marker& cmd);

  void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd);
  void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd);
  void submitPerfCounter(amd::PerfCounterCommand& cmd);

  void flush(amd::Command* list = nullptr, bool wait = false);
  void submitFillMemory(amd::FillMemoryCommand& cmd);
  void submitStreamOperation(amd::StreamOperationCommand& cmd);
  void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);

  void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);
  void submitSvmPrefetchAsync(amd::SvmPrefetchAsyncCommand& cmd);

  // { roc OpenCL integration
  // Added these stub (no-ops) implementation of pure virtual methods,
  // when integrating HSA and OpenCL branches.
  // TODO: After inegration, whoever is working on VirtualGPU should write
  // actual implementation.
  virtual void submitSignal(amd::SignalCommand& cmd) {}
  virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& cmd) {}

  virtual void submitTransferBufferFromFile(amd::TransferBufferFileCommand& cmd);

  void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) {}
  void submitThreadTrace(amd::ThreadTraceCommand& vcmd) {}

  /**
   * @brief Waits on an outstanding kernel without regard to how
   * it was dispatched - with or without a signal
   *
   * @return bool true if Wait returned successfully, false otherwise
   */
  bool releaseGpuMemoryFence(bool force_barrier = false, bool skip_copy_wait = false);

  hsa_agent_t gpu_device() const { return gpu_device_; }
  hsa_queue_t* gpu_queue() { return gpu_queue_; }

  // Return pointer to PrintfDbg
  PrintfDbg* printfDbg() const { return printfdbg_; }

  //! Returns memory dependency class
  MemoryDependency& memoryDependency() { return memoryDependency_; }

  //! Detects memory dependency for HSAIL kernels and uses appropriate AQL header
  bool processMemObjects(const amd::Kernel& kernel,  //!< AMD kernel object for execution
                         const_address params,       //!< Pointer to the param's store
                         size_t& ldsAddress,         //!< LDS usage
                         bool cooperativeGroups      //!< Dispatch with cooperative groups
                         );

  //! Adds a stage write buffer into a list
  void addXferWrite(Memory& memory);

  //! Releases stage write buffers
  void releaseXferWrite();

  //! Adds a pinned memory object into a map
  void addPinnedMem(amd::Memory* mem);

  //! Release pinned memory objects
  void releasePinnedMem();

  //! Finds if pinned memory is cached
  amd::Memory* findPinnedMem(void* addr, size_t size);

  void enableSyncBlit() const;

  void hasPendingDispatch() { hasPendingDispatch_ = true; }
  void addSystemScope() { addSystemScope_ = true; }
  void SetCopyCommandType(cl_command_type type) { copy_command_type_ = type; }

  HwQueueTracker& Barriers() { return barriers_; }

  Timestamp* timestamp() const { return timestamp_; }

  // } roc OpenCL integration
 private:
  bool dispatchAqlPacket(hsa_kernel_dispatch_packet_t* packet, uint16_t header,
                         uint16_t rest, bool blocking = true);
  bool dispatchAqlPacket(hsa_barrier_and_packet_t* packet, uint16_t header,
                        uint16_t rest, bool blocking = true);
  template <typename AqlPacket> bool dispatchGenericAqlPacket(AqlPacket* packet, uint16_t header,
                                                              uint16_t rest, bool blocking,
                                                              size_t size = 1);
  void dispatchBarrierPacket(hsa_barrier_and_packet_t* packet, uint16_t packetHeader,
                             bool skipSignal = false);
  bool dispatchCounterAqlPacket(hsa_ext_amd_aql_pm4_packet_t* packet, const uint32_t gfxVersion,
                                bool blocking, const hsa_ven_amd_aqlprofile_1_00_pfn_t* extApi);
  void dispatchBarrierValuePacket(const hsa_amd_barrier_value_packet_t* packet,
                                  hsa_amd_vendor_packet_header_t header);
  void initializeDispatchPacket(hsa_kernel_dispatch_packet_t* packet,
                                amd::NDRangeContainer& sizes);

  bool initPool(size_t kernarg_pool_size);
  void destroyPool();

  void* allocKernArg(size_t size, size_t alignment);
  void resetKernArgPool() { kernarg_pool_cur_offset_ = 0; }

  uint64_t getVQVirtualAddress();

  bool createSchedulerParam();

  //! Returns TRUE if virtual queue was successfully allocatted
  bool createVirtualQueue(uint deviceQueueSize);

  //! Common function for fill memory used by both svm Fill and non-svm fill
  bool fillMemory(cl_command_type type,        //!< the command type
                  amd::Memory* amdMemory,      //!< memory object to fill
                  const void* pattern,         //!< pattern to fill the memory
                  size_t patternSize,          //!< pattern size
                  const amd::Coord3D& origin,  //!< memory origin
                  const amd::Coord3D& size     //!< memory size for filling
                  );

  //! Common function for memory copy used by both svm Copy and non-svm Copy
  bool copyMemory(cl_command_type type,            //!< the command type
                  amd::Memory& srcMem,             //!< source memory object
                  amd::Memory& dstMem,             //!< destination memory object
                  bool entire,                     //!< flag of entire memory copy
                  const amd::Coord3D& srcOrigin,   //!< source memory origin
                  const amd::Coord3D& dstOrigin,   //!< destination memory object
                  const amd::Coord3D& size,        //!< copy size
                  const amd::BufferRect& srcRect,  //!< region of source for copy
                  const amd::BufferRect& dstRect   //!< region of destination for copy
                  );

  //! Updates AQL header for the upcomming dispatch
  void setAqlHeader(uint16_t header) { aqlHeader_ = header; }

  //! Resets the current queue state. Note: should be called after AQL queue becomes idle
  void ResetQueueStates();

  std::vector<Memory*> xferWriteBuffers_;  //!< Stage write buffers
  std::vector<amd::Memory*> pinnedMems_;   //!< Pinned memory list

  //! Queue state flags
  union {
    struct {
      uint32_t hasPendingDispatch_ : 1; //!< A kernel dispatch is outstanding
      uint32_t imageBufferWrtBack_ : 1; //!< Image buffer write back is required
      uint32_t profiling_          : 1; //!< Profiling is enabled
      uint32_t cooperative_        : 1; //!< Cooperative launch is enabled
      uint32_t addSystemScope_     : 1; //!< Insert a system scope to the next aql
      uint32_t tracking_created_   : 1; //!< Enabled if tracking object was properly initialized
    };
    uint32_t  state_;
  };

  std::vector<device::Memory*> wrtBackImageBuffer_;  //!< Array of images for write back

  Timestamp* timestamp_;
  hsa_agent_t gpu_device_;  //!< Physical device
  hsa_queue_t* gpu_queue_;  //!< Queue associated with a gpu
  hsa_barrier_and_packet_t barrier_packet_;

  uint32_t dispatch_id_;  //!< This variable must be updated atomically.
  Device& roc_device_;    //!< roc device object
  PrintfDbg* printfdbg_;
  MemoryDependency memoryDependency_;  //!< Memory dependency class
  uint16_t aqlHeader_;                 //!< AQL header for dispatch

  amd::Memory* virtualQueue_;     //!< Virtual device queue
  uint deviceQueueSize_;          //!< Device queue size
  uint maskGroups_;               //!< The number of mask groups processed in the scheduler by one thread
  uint schedulerThreads_;         //!< The number of scheduler threads

  amd::Memory* schedulerParam_;
  hsa_queue_t* schedulerQueue_;
  hsa_signal_t schedulerSignal_;

  HwQueueTracker  barriers_;      //!< Tracks active barriers in ROCr

  char* kernarg_pool_base_;
  size_t kernarg_pool_size_;
  uint kernarg_pool_cur_offset_;

  friend class Timestamp;

  //  PM4 packet for gfx8 performance counter
  enum {
    SLOT_PM4_SIZE_DW = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE/ sizeof(uint32_t),
    SLOT_PM4_SIZE_AQLP = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE/ 64
  };

  uint16_t dispatchPacketHeaderNoSync_;
  uint16_t dispatchPacketHeader_;

  //!< bit-vector representing the CU mask. Each active bit represents using one CU
  const std::vector<uint32_t> cuMask_;
  amd::CommandQueue::Priority priority_; //!< The priority for the hsa queue

  cl_command_type copy_command_type_;   //!< Type of the copy command, used for ROC profiler
                                        //!< OCL doesn't distinguish diffrent copy types,
                                        //!< but ROC profiler expects D2H or H2D detection
};

template <typename T>
inline void WriteAqlArgAt(
  unsigned char* dst,   //!< The write pointer to the buffer
  const T* src,         //!< The source pointer
  uint size,            //!< The size in bytes to copy
  size_t offset         //!< The alignment to follow while writing to the buffer
) {
  memcpy(dst + offset, src, size);
}

template <>
inline void WriteAqlArgAt(
  unsigned char* dst,   //!< The write pointer to the buffer
  const uint32_t* src,  //!< The source pointer
  uint size,            //!< The size in bytes to copy
  size_t offset         //!< The alignment to follow while writing to the buffer
) {
  *(reinterpret_cast<uint32_t*>(dst + offset)) = *src;
}

template <>
inline void WriteAqlArgAt(
  unsigned char* dst,   //!< The write pointer to the buffer
  const uint64_t* src,  //!< The source pointer
  uint size,            //!< The size in bytes to copy
  size_t offset         //!< The alignment to follow while writing to the buffer
) {
  *(reinterpret_cast<uint64_t*>(dst + offset)) = *src;
}
}
