//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#pragma once

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
  hsa_signal_t signal_;  //!< HSA signal to track profiling information
  Timestamp* ts_;        //!< Timestamp object associated with the signal

  ProfilingSignal() : ts_(nullptr) { signal_.handle = 0; }
};

// Timestamp for keeping track of some profiling information for various commands
// including EnqueueNDRangeKernel and clEnqueueCopyBuffer.
class Timestamp {
 private:
  uint64_t start_;
  uint64_t end_;
  ProfilingSignal* profilingSignal_;
  hsa_agent_t agent_;
  static double ticksToTime_;
  bool splittedDispatch_;
  std::vector<hsa_signal_t> splittedSignals_;

 public:
  uint64_t getStart() {
    checkGpuTime();
    return start_;
  }

  uint64_t getEnd() {
    checkGpuTime();
    return end_;
  }

  void setProfilingSignal(ProfilingSignal* signal) {
    profilingSignal_ = signal;
    if (splittedDispatch_) {
      splittedSignals_.push_back(profilingSignal_->signal_);
    }
  }
  const ProfilingSignal* getProfilingSignal() const { return profilingSignal_; }

  void setAgent(hsa_agent_t agent) { agent_ = agent; }

  Timestamp() : start_(0), end_(0), profilingSignal_(nullptr), splittedDispatch_(false) {
    agent_.handle = 0;
  }

  ~Timestamp() {}

  //! Finds execution ticks on GPU
  void checkGpuTime() {
    if (profilingSignal_ != nullptr) {
      hsa_amd_profiling_dispatch_time_t time;

      if (splittedDispatch_) {
        uint64_t start = UINT64_MAX;
        uint64_t end = 0;
        for (auto it = splittedSignals_.begin(); it < splittedSignals_.end(); it++) {
          hsa_amd_profiling_get_dispatch_time(agent_, *it, &time);
          if (time.start < start) {
            start = time.start;
          }
          if (time.end > end) {
            end = time.end;
          }
        }
        start_ = start * ticksToTime_;
        end_ = end * ticksToTime_;
      } else {
        hsa_amd_profiling_get_dispatch_time(agent_, profilingSignal_->signal_, &time);
        start_ = time.start * ticksToTime_;
        end_ = time.end * ticksToTime_;
      }
      profilingSignal_->ts_ = nullptr;
      profilingSignal_ = nullptr;
    }
  }

  // Start a timestamp (get timestamp from OS)
  void start() { start_ = amd::Os::timeNanos(); }

  // End a timestamp (get timestamp from OS)
  void end() { end_ = amd::Os::timeNanos(); }

  bool isSplittedDispatch() const { return splittedDispatch_; }
  void setSplittedDispatch() { splittedDispatch_ = true; }

  static void setGpuTicksToTime(double ticksToTime) { ticksToTime_ = ticksToTime; }
  static double getGpuTicksToTime() { return ticksToTime_; }
};

class VirtualGPU : public device::VirtualDevice {
 public:
  //! Initial signal value
  static const hsa_signal_value_t InitSignalValue = 1;

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

  VirtualGPU(Device& device);
  ~VirtualGPU();

  bool create(bool profilingEna);
  bool terminate() { return true; }
  const Device& dev() const { return roc_device_; }

  void profilingBegin(amd::Command& command, bool drmProfiling = false);
  void profilingEnd(amd::Command& command);

  void updateCommandsState(amd::Command* list);

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
                            uint32_t sharedMemBytes = 0 //!< Shared memory size
                            );
  void submitNativeFn(amd::NativeFnCommand& cmd);
  void submitMarker(amd::Marker& cmd);

  void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd);
  void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd);
  void submitPerfCounter(amd::PerfCounterCommand& cmd);

  void flush(amd::Command* list = nullptr, bool wait = false);
  void submitFillMemory(amd::FillMemoryCommand& cmd);
  void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);

  void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);

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
   * @return bool true if Wait returned successfully, false
   * otherwise
   */
  bool releaseGpuMemoryFence();

  hsa_agent_t gpu_device() { return gpu_device_; }
  hsa_queue_t* gpu_queue() { return gpu_queue_; }

  // Return pointer to PrintfDbg
  PrintfDbg* printfDbg() const { return printfdbg_; }

  //! Returns memory dependency class
  MemoryDependency& memoryDependency() { return memoryDependency_; }

  //! Detects memory dependency for HSAIL kernels and uses appropriate AQL header
  bool processMemObjects(const amd::Kernel& kernel,  //!< AMD kernel object for execution
                         const_address params,       //!< Pointer to the param's store
                         size_t& ldsAddress          //!< LDS usage
                         );
  // Retun the virtual gpu unique index
  uint index() const { return index_; }

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

  // } roc OpenCL integration
 private:
  bool dispatchAqlPacket(hsa_kernel_dispatch_packet_t* packet, bool blocking = true);
  bool dispatchAqlPacket(hsa_barrier_and_packet_t* packet, bool blocking = true);
  template <typename AqlPacket> bool dispatchGenericAqlPacket(AqlPacket* packet, bool blocking, size_t size = 1);
  void dispatchBarrierPacket(const hsa_barrier_and_packet_t* packet);
  bool dispatchCounterAqlPacket(hsa_ext_amd_aql_pm4_packet_t* packet, const uint32_t gfxVersion, bool blocking, const hsa_ven_amd_aqlprofile_1_00_pfn_t* extApi);
  void initializeDispatchPacket(hsa_kernel_dispatch_packet_t* packet, amd::NDRangeContainer& sizes);

  bool initPool(size_t kernarg_pool_size, uint signal_pool_count);
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

  std::vector<Memory*> xferWriteBuffers_;  //!< Stage write buffers
  std::vector<amd::Memory*> pinnedMems_;   //!< Pinned memory list

  /**
   * @brief Indicates if a kernel dispatch is outstanding. This flag is
   * used to synchronized on kernel outputs.
   */
  bool hasPendingDispatch_;
  Timestamp* timestamp_;
  hsa_agent_t gpu_device_;  //!< Physical device
  hsa_queue_t* gpu_queue_;  //!< Queue associated with a gpu
  hsa_barrier_and_packet_t barrier_packet_;
  hsa_signal_t barrier_signal_;
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

  char* kernarg_pool_base_;
  size_t kernarg_pool_size_;
  uint kernarg_pool_cur_offset_;

  std::vector<ProfilingSignal> signal_pool_;  //!< Pool of signals for profiling
  const uint index_;                          //!< Virtual gpu unique index
  friend class Timestamp;

  //  PM4 packet for gfx8 performance counter
  enum {
    SLOT_PM4_SIZE_DW = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE/ sizeof(uint32_t),
    SLOT_PM4_SIZE_AQLP = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE/ 64
  };

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
