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

#ifndef GPUVIRTUAL_HPP_
#define GPUVIRTUAL_HPP_

#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpuconstbuf.hpp"
#include "device/gpu/gpuprintf.hpp"
#include "device/gpu/gputimestamp.hpp"
#include "device/gpu/gpusched.hpp"
#include "platform/commandqueue.hpp"
#include "device/blit.hpp"

#include "device/gpu/gpudebugger.hpp"


/*! \addtogroup GPU GPU Resource Implementation
 *  @{
 */

//! GPU Device Implementation
namespace gpu {

class Device;
class Kernel;
class Memory;
class CalCounterReference;
class VirtualGPU;
class Program;
class BlitManager;
class ThreadTrace;
class HSAILKernel;

//! Virtual GPU
class VirtualGPU : public device::VirtualDevice, public CALGSLContext {
 public:
  struct CommandBatch : public amd::HeapObject {
    amd::Command* head_;           //!< Command batch head
    GpuEvent events_[AllEngines];  //!< Last known GPU events
    TimeStamp* lastTS_;            //!< TS associated with command batch

    //! Constructor
    CommandBatch(amd::Command* head,      //!< Command batch head
                 const GpuEvent* events,  //!< HW events on all engines
                 TimeStamp* lastTS        //!< Last TS in command batch
                 )
        : head_(head), lastTS_(lastTS) {
      memcpy(&events_, events, AllEngines * sizeof(GpuEvent));
    }
  };

  //! The virtual GPU states
  union State {
    struct {
      uint boundGlobal_ : 1;  //!< Global buffer was bound
      uint profiling_ : 1;    //!< Profiling is enabled
      uint forceWait_ : 1;    //!< Forces wait in flush()
      uint boundCb_ : 1;      //!< Constant buffer was bound
      uint boundPrintf_ : 1;  //!< Printf buffer was bound
      uint hsailKernel_ : 1;  //!< True if HSAIL kernel was used
    };
    uint value_;
    State() : value_(0) {}
  };

  //! CAL descriptor for the GPU virtual device
  struct CalVirtualDesc : public amd::EmbeddedObject {
    gslDomain3D gridBlock;                        //!< size of a block of data
    gslDomain3D gridSize;                         //!< size of 'blocks' to execute
    gslDomain3D partialGridBlock;                 //!< Partial grid block
    CALuint localSize;                            //!< size of OpenCL Local Memory in bytes
    uint memCount_;                               //!< Memory objects count
    GpuEvent events_[AllEngines];                 //!< Last known GPU events
    uint iterations_;                             //!< Number of iterations for the execution
    TimeStamp* lastTS_;                           //!< Last timestamp executed on Virtual GPU
    gslMemObject constBuffers_[MaxConstBuffers];  //!< Constant buffer names
    gslMemObject uavs_[MaxUavArguments];          //!< UAV bindings
    gslMemObject readImages_[MaxReadImage];       //!< Read images
    uint32_t samplersState_[MaxSamplers];         //!< State of all samplers
  };

  typedef std::vector<ConstBuffer*> constbufs_t;

  //! GSL descriptor for the GPU kernel, specific to the virtual device
  struct GslKernelDesc : public amd::HeapObject {
    CALimage image_;         //!< CAL image for the program
    gslProgramObject func_;  //!< GSL program object
    gslMemObject intCb_;     //!< Internal constant buffer
  };

  struct ResourceSlot {
    union State {
      struct {
        uint bound_ : 1;     //!< Resource is bound
        uint constant_ : 1;  //!< Resource is a constant
      };
      uint value_;
      State() : value_(0) {}
    };

    State state_;           //!< slot's state
    const Memory* memory_;  //!< GPU memory object

    ResourceSlot() : memory_(NULL) {}

    //! Copy constructor for the kernel argument
    ResourceSlot(const ResourceSlot& data) { *this = data; }

    //! Overloads operator=
    ResourceSlot& operator=(const ResourceSlot& data) {
      state_.value_ = data.state_.value_;
      memory_ = data.memory_;
      return *this;
    }
  };

  class MemoryDependency : public amd::EmbeddedObject {
   public:
    //! Default constructor
    MemoryDependency()
        : memObjectsInQueue_(NULL), endMemObjectsInQueue_(0), numMemObjectsInQueue_(0), maxMemObjectsInQueue_(0) {}

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


  class DmaFlushMgmt : public amd::EmbeddedObject {
   public:
    DmaFlushMgmt(const Device& dev);

    // Resets DMA command buffer workload
    void resetCbWorkload(const Device& dev);

    // Finds split size for the current dispatch
    void findSplitSize(const Device& dev,  //!< GPU device object
                       uint64_t threads,   //!< Total number of execution threads
                       uint instructions   //!< Number of ALU instructions
                       );

    // Returns TRUE if DMA command buffer is ready for a flush
    bool isCbReady(VirtualGPU& gpu,   //!< Virtual GPU object
                   uint64_t threads,  //!< Total number of execution threads
                   uint instructions  //!< Number of ALU instructions
                   );

    // Returns dispatch split size
    uint dispatchSplitSize() const { return dispatchSplitSize_; }

   private:
    uint64_t maxDispatchWorkload_;  //!< Maximum number of operations for a single dispatch
    uint64_t maxCbWorkload_;        //!< Maximum number of operations for DMA command buffer
    uint64_t cbWorkload_;           //!< Current number of operations in DMA command buffer
    uint aluCnt_;                   //!< All ALUs on the chip
    uint dispatchSplitSize_;        //!< Dispath split size in elements
  };

  typedef std::vector<ResourceSlot> ResourceSlots;

 public:
  explicit VirtualGPU(Device& device);
  bool create(bool profiling, uint rtCUs = amd::CommandQueue::RealTimeDisabled,
              uint deviceQueueSize = 0,
              amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal);
  ~VirtualGPU();

  void submitReadMemory(amd::ReadMemoryCommand& vcmd);
  void submitWriteMemory(amd::WriteMemoryCommand& vcmd);
  void submitCopyMemory(amd::CopyMemoryCommand& vcmd);
  void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& vcmd) {}
  void submitMapMemory(amd::MapMemoryCommand& vcmd);
  void submitUnmapMemory(amd::UnmapMemoryCommand& vcmd);
  void submitKernel(amd::NDRangeKernelCommand& vcmd);
  bool submitKernelInternal(
      const amd::NDRangeContainer& sizes,  //!< Workload sizes
      const amd::Kernel& kernel,           //!< Kernel for execution
      const_address parameters,            //!< Parameters for the kernel
      bool nativeMem = true,               //!< Native memory objects
      amd::Event* enqueueEvent = NULL      //!< Event provided in the enqueue kernel command
      );
  bool submitKernelInternalHSA(
      const amd::NDRangeContainer& sizes,  //!< Workload sizes
      const amd::Kernel& kernel,           //!< Kernel for execution
      const_address parameters,            //!< Parameters for the kernel
      bool nativeMem = true,               //!< Native memory objects
      amd::Event* enqueueEvent = NULL      //!< Event provided in the enqueue kernel command
      );
  void submitNativeFn(amd::NativeFnCommand& vcmd);
  void submitFillMemory(amd::FillMemoryCommand& vcmd);
  void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);
  void submitMarker(amd::Marker& vcmd);
  void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd);
  void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd);
  void submitPerfCounter(amd::PerfCounterCommand& vcmd);
  void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd);
  void submitThreadTrace(amd::ThreadTraceCommand& vcmd);
  void submitSignal(amd::SignalCommand& vcmd);
  void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& vcmd);
  virtual void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  virtual void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  virtual void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  virtual void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  virtual void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);
  virtual void submitTransferBufferFromFile(amd::TransferBufferFileCommand& cmd);

  void releaseMemory(gslMemObject gslResource, bool wait = true);
  void releaseKernel(CALimage calImage);

  void flush(amd::Command* list = NULL, bool wait = false);
  bool terminate() { return true; }

  //! Returns GPU device object associated with this kernel
  const Device& dev() const { return gpuDevice_; }

  //! Returns CAL descriptor of the virtual device
  const CalVirtualDesc* cal() const { return &cal_; }

  //! Returns active kernel descriptor for this virtual device
  const GslKernelDesc* gslKernelDesc() const { return activeKernelDesc_; }

  //! Returns a GPU event, associated with GPU memory
  GpuEvent* getGpuEvent(const gslMemObject gslMem  //!< GSL mem object
                        ) {
    return &gpuEvents_[gslMem];
  }

  //! Assigns a GPU event, associated with GPU memory
  void assignGpuEvent(const gslMemObject gslMem,  //!< GSL mem object
                      GpuEvent gpuEvent) {
    gpuEvents_[gslMem] = gpuEvent;
  }

  //! Set the kernel as active
  bool setActiveKernelDesc(const amd::NDRangeContainer& sizes,  //!< kernel execution work sizes
                           const Kernel* kernel                 //!< GPU kernel object
                           );

  //! Set the last known GPU event
  void setGpuEvent(GpuEvent gpuEvent,  //!< GPU event for tracking
                   bool flush = false  //!< TRUE if flush is required
                   );

  //! Flush DMA buffer on the specified engine
  void flushDMA(uint engineID  //!< Engine ID for DMA flush
                );

  //! Wait for all engines on this Virtual GPU
  //! Returns TRUE if CPU didn't wait for GPU
  bool waitAllEngines(CommandBatch* cb = NULL  //!< Command batch
                      );

  //! Waits for the latest GPU event with a lock to prevent multiple entries
  void waitEventLock(CommandBatch* cb  //!< Command batch
                     );

  //! Returns a resource associated with the constant buffer
  const ConstBuffer* cb(uint idx) const { return constBufs_[idx]; }

  //! Adds CAL objects into the constant buffer vector
  void addConstBuffer(ConstBuffer* cb) { constBufs_.push_back(cb); }

  constbufs_t constBufs_;  //!< constant buffers

  //! Start the command profiling
  void profilingBegin(amd::Command& command,     //!< Command queue object
                      bool drmProfiling = false  //!< Measure DRM time
                      );

  //! End the command profiling
  void profilingEnd(amd::Command& command);

  //! Collect the profiling results
  bool profilingCollectResults(CommandBatch* cb,               //!< Command batch
                               const amd::Event* waitingEvent  //!< Waiting event
                               );

  //! Adds a memory handle into the GSL memory array for Virtual Heap
  bool addVmMemory(const Memory* memory  //!< GPU memory object
                   );

  //! Adds a stage write buffer into a list
  void addXferWrite(Memory& memory);

  //! Adds a pinned memory object into a map
  void addPinnedMem(amd::Memory* mem);

  //! Release pinned memory objects
  void releasePinnedMem();

  //! Finds if pinned memory is cached
  amd::Memory* findPinnedMem(void* addr, size_t size);

  //! Returns gsl memory object for VM
  const gslMemObject* vmMems() const { return vmMems_; }

  //! Get the PrintfDbg object
  PrintfDbg& printfDbg() const { return *printfDbg_; }

  //! Get the PrintfDbgHSA object
  PrintfDbgHSA& printfDbgHSA() const { return *printfDbgHSA_; }

  //! Enables synchronized transfers
  void enableSyncedBlit() const;

  //! Checks if profiling is enabled
  bool profiling() const { return state_.profiling_; }

  //! Returns memory dependency class
  MemoryDependency& memoryDependency() { return memoryDependency_; }

  //! Returns hsaQueueMem_
  const Memory* hsaQueueMem() const { return hsaQueueMem_; }

  //! Returns DMA flush management structure
  const DmaFlushMgmt& dmaFlushMgmt() const { return dmaFlushMgmt_; }

  //! Releases GSL memory objects allocated on this queue
  void releaseMemObjects(bool scratch = true);

  //! Returns the HW ring used on this virtual device
  uint hwRing() const { return hwRing_; }

  //! Returns current timestamp object for profiling
  TimeStamp* currTs() const { return cal_.lastTS_; }

  //! Returns virtual queue object for device enqueuing
  Memory* vQueue() const { return virtualQueue_; }

  //! Update virtual queue header
  void writeVQueueHeader(VirtualGPU& hostQ, uint64_t kernelTable);

  //! Returns TRUE if virtual queue was successfully allocatted
  bool createVirtualQueue(uint deviceQueueSize  //!< Device queue size
                          );

  EngineType engineID_;  //!< Engine ID for this VirtualGPU
  ResourceSlots slots_;  //!< Resource slots for kernel arguments
  State state_;          //!< virtual GPU current state
  CalVirtualDesc cal_;   //!< CAL virtual device descriptor

  void flushCuCaches(HwDbgGpuCacheMask cache_mask);  //!< flush/invalidate SQ cache

 protected:
  virtual void profileEvent(EngineType engine, bool type) const;

  //! Creates buffer object from image
  amd::Memory* createBufferFromImage(
      amd::Memory& amdImage  //! The parent image object(untiled images only)
      );

 private:
  typedef std::unordered_map<CALimage, GslKernelDesc*> GslKernels;
  typedef std::unordered_map<gslMemObject, GpuEvent> GpuEvents;

  //! Finds total amount of necessary iterations
  inline void findIterations(const amd::NDRangeContainer& sizes,  //!< Original workload sizes
                             const amd::NDRange& local,           //!< Local workgroup size
                             amd::NDRange& groups,                //!< Calculated workgroup sizes
                             amd::NDRange& remainder,             //!< Calculated remainder sizes
                             size_t& extra  //!< Amount of extra executions for remainder
                             );

  //! Setups workloads for the current iteration
  inline void setupIteration(
      uint iteration,                      //!< Current iteration
      const amd::NDRangeContainer& sizes,  //!< Original workload sizes
      Kernel& gpuKernel,                   //!< GPU kernel
      amd::NDRange& global,                //!< Global size for the current iteration
      amd::NDRange& offsets,               //!< Offsets for the current iteration
      amd::NDRange& local,                 //!< Local sizes for the current iteration
      amd::NDRange& groups,                //!< Group sizes for the current iteration
      amd::NDRange& groupOffset,           //!< Group offsets for the current iteration
      amd::NDRange& divider,               //!< Group divider
      amd::NDRange& remainder,             //!< Remain workload
      size_t extra                         //!< Extra groups
      );

  //! Allocates constant buffers
  bool allocConstantBuffers();

  //! Allocates CAL kernel descriptor of the virtual device
  GslKernelDesc* allocKernelDesc(const Kernel* kernel,  //!< Kernel object
                                 CALimage calImage);    //!< CAL image

  //! Frees CAL kernel descriptor of the virtual device
  void freeKernelDesc(GslKernelDesc* desc);

  bool gslOpen(uint nEngines, gslEngineDescriptor* engines, uint32_t rtCUs);
  void gslDestroy();

  //! Releases stage write buffers
  void releaseXferWrite();

  //! Allocate hsaQueueMem_
  bool allocHsaQueueMem();

  //! Awaits a command batch with a waiting event
  bool awaitCompletion(CommandBatch* cb,                      //!< Command batch for to wait
                       const amd::Event* waitingEvent = NULL  //!< A waiting event
                       );

  //! Validates the scratch buffer memory for a specified kernel
  void validateScratchBuffer(const Kernel* kernel  //!< Kernel for validaiton
                             );

  //! Detects memory dependency for HSAIL kernels and flushes caches
  bool processMemObjectsHSA(const amd::Kernel& kernel,  //!< AMD kernel object for execution
                            const_address params,       //!< Pointer to the param's store
                            bool nativeMem,             //!< Native memory objects
                            std::vector<const Memory*>* memList  //!< Memory list for KMD tracking
                            );

  //! Common function for fill memory used by both svm Fill and non-svm fill
  bool fillMemory(cl_command_type type,        //!< the command type
                  amd::Memory* amdMemory,      //!< memory object to fill
                  const void* pattern,         //!< pattern to fill the memory
                  size_t patternSize,          //!< pattern size
                  const amd::Coord3D& origin,  //!< memory origin
                  const amd::Coord3D& size     //!< memory size for filling
                  );

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

  void buildKernelInfo(const HSAILKernel& hsaKernel,          //!< hsa kernel
                       hsa_kernel_dispatch_packet_t* aqlPkt,  //!< aql packet for dispatch
                       HwDbgKernelInfo& kernelInfo,           //!< kernel info for the dispatch
                       amd::Event* enqueueEvent  //!< Event provided in the enqueue kernel command
                       );

  void assignDebugTrapHandler(const DebugToolInfo& dbgSetting,  //!< debug settings
                              HwDbgKernelInfo& kernelInfo       //!< kernel info for the dispatch
                              );

  GslKernels gslKernels_;            //!< GSL kernel descriptors
  GslKernelDesc* activeKernelDesc_;  //!< active GSL kernel descriptors
  GpuEvents gpuEvents_;              //!< GPU events

  Device& gpuDevice_;       //!< physical GPU device

  PrintfDbg* printfDbg_;        //!< GPU printf implemenation
  PrintfDbgHSA* printfDbgHSA_;  //!< HSAIL printf implemenation

  TimeStampCache* tsCache_;            //!< TimeStamp cache
  MemoryDependency memoryDependency_;  //!< Memory dependency class

  gslMemObject* vmMems_;  //!< Array of GSL memories for VM mode
  uint numVmMems_;        //!< Number of entries in VM mem array

  DmaFlushMgmt dmaFlushMgmt_;  //!< DMA flush management

  std::list<Memory*> xferWriteBuffers_;  //!< Stage write buffers
  std::list<amd::Memory*> pinnedMems_;   //!< Pinned memory list

  typedef std::list<CommandBatch*> CommandBatchList;
  CommandBatchList cbList_;  //!< List of command batches

  uint hwRing_;  //!< HW ring used on this virtual device

  uint64_t readjustTimeGPU_;  //!< Readjust time between GPU and CPU timestamps
  TimeStamp* currTs_;         //!< current timestamp for command

  AmdVQueueHeader* vqHeader_;  //!< Sysmem copy for virtual queue header
  Memory* virtualQueue_;       //!< Virtual device queue
  Memory* schedParams_;        //!< The scheduler parameters
  uint schedParamIdx_;         //!< Index in the scheduler parameters buffer
  uint deviceQueueSize_;       //!< Device queue size
  uint maskGroups_;  //!< The number of mask groups processed in the scheduler by one thread

  Memory* hsaQueueMem_;  //!< Memory for the amd_queue_t object
  bool profileEnabled_;  //!< Profiling is enabled
};

/*@}*/} // namespace gpu

#endif /*GPUVIRTUAL_HPP_*/
