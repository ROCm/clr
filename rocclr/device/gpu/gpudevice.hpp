/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#ifndef GPU_HPP_
#define GPU_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/perfctr.hpp"
#include "platform/threadtrace.hpp"
#include "platform/memory.hpp"
#include "utils/concurrent.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "device/gpu/gpuvirtual.hpp"
#include "device/gpu/gpumemory.hpp"
#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpusettings.hpp"
#include "device/gpu/gpuappprofile.hpp"


#include "acl.h"
#include "vaminterface.h"

/*! \addtogroup GPU
 *  @{
 */

//! GPU Device Implementation
namespace gpu {

//! A nil device object
class NullDevice : public amd::Device {
 protected:
  static aclCompiler* compiler_;
  static aclCompiler* hsaCompiler_;

 public:
  aclCompiler* amdilCompiler() const { return compiler_; }
  aclCompiler* hsaCompiler() const { return hsaCompiler_; }
  aclCompiler* compiler() const { return hsaCompiler_; }
  Compiler* binCompiler() const { return amdilCompiler(); }

  static bool init(void);

  //! Construct a new identifier
  NullDevice();

  //! Creates an offline device with the specified target
  bool create(CALtarget target  //!< GPU device identifier
              );

  //! Instantiate a new virtual device
  virtual device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = NULL) {
    return NULL;
  }

  //! Create the device program.
  virtual device::Program* createProgram(amd::Program& owner, amd::option::Options* options = NULL);

  //! Just returns NULL for the dummy device
  virtual device::Memory* createMemory(amd::Memory& owner) const { return NULL; }

  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
                             ) const {
    ShouldNotReachHere();
    return true;
  }

  //! Just returns NULL for the dummy device
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
      ) const {
    return NULL;
  }

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device

  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly) {
    return true;
  }

  virtual bool unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                    bool validateOnly) {
    return true;
  }

  //! Releases non-blocking map target memory
  virtual void freeMapTarget(amd::Memory& mem, void* target) {}

  CALtarget calTarget() const { return calTarget_; }

  const AMDDeviceInfo* hwInfo() const { return hwInfo_; }

  //! Empty implementation on Null device
  virtual bool globalFreeMemory(size_t* freeMemory) const { return false; }

  //! Get GPU device settings
  const gpu::Settings& settings() const { return reinterpret_cast<gpu::Settings&>(*settings_); }
  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags, void* svmPtr) const {
    return NULL;
  }
  virtual void svmFree(void* ptr) const { return; }

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput, cl_set_device_clock_mode_output_amd* pSetClockModeOutput) { return true; }

 protected:
  bool usePal() const {
    return (calTarget_ == CAL_TARGET_GREENLAND || calTarget_ == CAL_TARGET_RAVEN ||
            calTarget_ == CAL_TARGET_RAVEN2 || calTarget_ == CAL_TARGET_RENOIR ||
            calTarget_ >= CAL_TARGET_VEGA12);
  }

  //! Answer the question: "Should HSAIL Program be created?",
  //! based on the given options.
  bool isHsailProgram(amd::option::Options* options = NULL);

  //! Fills OpenCL device info structure
  void fillDeviceInfo(const CALdeviceattribs& calAttr,  //!< CAL device attributes info
                      const gslMemInfo& memInfo,        //!< GSL mem info
                      size_t maxTextureSize,            //!< Maximum texture size supported in HW
                      uint numComputeRings,             //!< Number of compute rings
                      uint numComputeRingsRT            //!< Number of RT compute rings
                      );

  CALtarget calTarget_;          //!< GPU device identifier
  const AMDDeviceInfo* hwInfo_;  //!< Device HW info structure
};

//! Forward declarations
class Command;
class Device;
class GpuCommand;
class Heap;
class HeapBlock;
class Program;
class Kernel;
class Memory;
class Resource;
class VirtualDevice;
class PrintfDbg;
class ThreadTrace;

#ifndef CL_FILTER_NONE
#define CL_FILTER_NONE 0x1142
#endif

class Sampler : public device::Sampler {
 public:
  //! Constructor
  Sampler(const Device& dev) : dev_(dev) {}

  //! Default destructor for the device memory object
  virtual ~Sampler();

  //! Creates a device sampler from the OCL sampler state
  bool create(uint32_t oclSamplerState  //!< OCL sampler state
              );

  //! Creates a device sampler from the OCL sampler state
  bool create(const amd::Sampler& owner  //!< AMD sampler object
              );

  const void* hwState() const { return hwState_; }

 private:
  //! Disable default copy constructor
  Sampler& operator=(const Sampler&);

  //! Disable operator=
  Sampler(const Sampler&);

  const Device& dev_;  //!< Device object associated with the sampler
  address hwState_;    //!< GPU HW state (\todo legacy path)
};

//! A GPU device ordinal (physical GPU device)
class Device : public NullDevice, public CALGSLDevice {
 public:
  class Heap : public amd::EmbeddedObject {
   public:
    //! The size of a heap element in bytes
    static const size_t ElementSize = 4;

    //! The type of a heap element in bytes
    static const cmSurfFmt ElementType = CM_SURF_FMT_R32I;

    Heap() : resource_(NULL), baseAddress_(0) {}

    bool create(Device& device  //!< GPU device object
                );

    //! Gets the GPU resource associated with the global heap
    const Memory& resource() const { return *resource_; }

    //! Returns the base virtual address of the heap
    uint64_t baseAddress() const { return baseAddress_; }

   protected:
    Memory* resource_;      //!< GPU resource referencing the heap memory
    uint64_t baseAddress_;  //!< Virtual heap base address
  };

  //! Locks any access to the virtual GPUs
  class ScopedLockVgpus : public amd::StackObject {
   public:
    //! Default constructor
    ScopedLockVgpus(const Device& dev);

    //! Destructor
    ~ScopedLockVgpus();

   private:
    const Device& dev_;  //! Device object
  };

  //! Interop emulation flags
  enum InteropEmulationFlags {
    D3D10Device = 0x00000001,
    GLContext = 0x00000002,
  };

  class Engines : public amd::EmbeddedObject {
   public:
    //! Default constructor
    Engines() : numComputeRings_(0), numComputeRingsRT_(0), numDmaEngines_(0) {
      memset(desc_, 0xff, sizeof(desc_));
    }

    //! Creates engine descriptor for this class
    void create(uint num, gslEngineDescriptor* desc, uint maxNumComputeRings);

    //! Gets engine type mask
    uint getMask(gslEngineID id) const { return (1 << id); }

    //! Gets a descriptor for the requested engines
    uint getRequested(uint engines, gslEngineDescriptor* desc) const;

    //! Returns the number of available compute rings
    uint numComputeRings() const { return numComputeRings_; }

    //! Returns the number of available real time compute rings
    uint numComputeRingsRT() const { return numComputeRingsRT_; }

    //! Returns the number of available DMA engines
    uint numDMAEngines() const { return numDmaEngines_; }

   private:
    uint numComputeRings_;
    uint numComputeRingsRT_;
    uint numDmaEngines_;
    gslEngineDescriptor desc_[GSL_ENGINEID_MAX];  //!< Engine descriptor
  };

  //! Transfer buffers
  class XferBuffers : public amd::HeapObject {
   public:
    static const size_t MaxXferBufListSize = 8;

    //! Default constructor
    XferBuffers(const Device& device, Resource::MemoryType type, size_t bufSize)
        : type_(type), bufSize_(bufSize), acquiredCnt_(0), gpuDevice_(device) {}

    //! Default destructor
    ~XferBuffers();

    //! Creates the xfer buffers object
    bool create();

    //! Acquires an instance of the transfer buffers
    Memory& acquire();

    //! Releases transfer buffer
    void release(VirtualGPU& gpu,  //!< Virual GPU object used with the buffer
                 Memory& buffer    //!< Transfer buffer for release
                 );

    //! Returns the buffer's size for transfer
    size_t bufSize() const { return bufSize_; }

   private:
    //! Disable copy constructor
    XferBuffers(const XferBuffers&);

    //! Disable assignment operator
    XferBuffers& operator=(const XferBuffers&);

    //! Get device object
    const Device& dev() const { return gpuDevice_; }

    Resource::MemoryType type_;       //!< The buffer's type
    size_t bufSize_;                  //!< Staged buffer size
    std::list<Memory*> freeBuffers_;  //!< The list of free buffers
    amd::Atomic<uint> acquiredCnt_;   //!< The total number of acquired buffers
    amd::Monitor lock_;               //!< Staged buffer acquire/release lock
    const Device& gpuDevice_;         //!< GPU device object
  };

  struct ScratchBuffer : public amd::HeapObject {
    uint regNum_;      //!< The number of used scratch registers
    Memory* memObj_;   //!< Memory objects for scratch buffers
    uint64_t offset_;  //!< Offset from the global scratch store
    uint64_t size_;    //!< Scratch buffer size on this queue

    //! Default constructor
    ScratchBuffer() : regNum_(0), memObj_(NULL), offset_(0), size_(0) {}

    //! Default constructor
    ~ScratchBuffer();

    //! Destroys memory objects
    void destroyMemory();
  };


  class SrdManager : public amd::HeapObject {
   public:
    SrdManager(const Device& dev, uint srdSize, uint bufSize)
        : dev_(dev),
          numFlags_(bufSize / (srdSize * MaskBits)),
          srdSize_(srdSize),
          bufSize_(bufSize) {}
    ~SrdManager();

    //! Allocates a new SRD slot for a resource
    uint64_t allocSrdSlot(address* cpuAddr);

    //! Frees a SRD slot
    void freeSrdSlot(uint64_t addr);

    // Fills the memory list for VidMM KMD
    void fillResourceList(std::vector<const Memory*>& memList);

   private:
    //! Disable copy constructor
    SrdManager(const SrdManager&);

    //! Disable assignment operator
    SrdManager& operator=(const SrdManager&);

    struct Chunk {
      Memory* buf_;
      uint* flags_;
      Chunk() : buf_(NULL), flags_(NULL) {}
    };

    static const uint MaskBits = 32;
    const Device& dev_;        //!< GPU device for the chunk manager
    amd::Monitor ml_;          //!< Global lock for the SRD manager
    std::vector<Chunk> pool_;  //!< Pool of SRD buffers
    uint numFlags_;            //!< Total number of flags in array
    uint srdSize_;             //!< SRD size
    uint bufSize_;             //!< Buffer size that holds SRDs
  };

  //! Initialise the whole GPU device subsystem (CAL init, device enumeration, etc).
  static bool init();

  //! Shutdown the whole GPU device subsystem (CAL shutdown).
  static void tearDown();

  //! Construct a new physical GPU device
  Device();

  //! Initialise a device (i.e. all parts of the constructor that could
  //! potentially fail)
  bool create(CALuint ordinal,      //!< GPU device ordinal index. Starts from 0
              CALuint numOfDevices  //!< number of GPU devices in the system
              );

  //! Destructor for the physical GPU device
  virtual ~Device();

  //! Instantiate a new virtual device
  device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = NULL);

  //! Memory allocation
  virtual device::Memory* createMemory(amd::Memory& owner  //!< abstraction layer memory object
                                       ) const;

  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
                             ) const;

  //! Allocates a view object from the device memory
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
      ) const;

  //! Create the device program.
  virtual device::Program* createProgram(amd::Program& owner, amd::option::Options* options = NULL);

  //! Attempt to bind with external graphics API's device/context
  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly);

  //! Attempt to unbind with external graphics API's device/context
  virtual bool unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                    bool validateOnly);

  //! Validates kernel before execution
  virtual bool validateKernel(const amd::Kernel& kernel,  //!< AMD kernel object
                              const device::VirtualDevice* vdev,
                              bool coop_groups = false);

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput, cl_set_device_clock_mode_output_amd* pSetClockModeOutput);

  //! Retrieves information about free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory) const;

  //! Returns a GPU memory object from AMD memory object
  gpu::Memory* getGpuMemory(amd::Memory* mem  //!< Pointer to AMD memory object
                            ) const;

  //! Gets the GPU resource associated with the global heap
  const Memory& globalMem() const { return heap_.resource(); }

  //! Gets the device context object
  amd::Context& context() const { return *context_; }

  //! Gets the global heap object
  const Heap& heap() const { return heap_; }

  //! Gets the memory object for the dummy page
  amd::Memory* dummyPage() const { return dummyPage_; }

  amd::Monitor& lockAsyncOps() const { return *lockAsyncOps_; }

  //! Returns the lock object for the virtual gpus list
  amd::Monitor* vgpusAccess() const { return vgpusAccess_; }

  //! Returns the number of virtual GPUs allocated on this device
  uint numOfVgpus() const { return numOfVgpus_; }
  uint numOfVgpus_;  //!< The number of virtual GPUs (lock protected)

  typedef std::vector<VirtualGPU*> VirtualGPUs;

  //! Returns the list of all virtual GPUs running on this device
  const VirtualGPUs& vgpus() const { return vgpus_; }
  VirtualGPUs vgpus_;  //!< The list of all running virtual gpus (lock protected)

  //! Scratch buffer allocation
  gpu::Memory* createScratchBuffer(size_t size  //!< Size of buffer
                                   ) const;

  //! Returns transfer buffer object
  XferBuffers& xferWrite() const { return *xferWrite_; }

  //! Returns transfer buffer object
  XferBuffers& xferRead() const { return *xferRead_; }

  //! Finds an appropriate map target
  amd::Memory* findMapTarget(size_t size) const;

  //! Adds a map target to the cache
  bool addMapTarget(amd::Memory* memory) const;

  //! Returns resource cache object
  ResourceCache& resourceCache() const { return *resourceCache_; }

  //! Returns engines object
  const Engines& engines() const { return engines_; }

  //! Returns engines object
  const device::BlitManager& xferMgr() const;

  VirtualGPU* xferQueue() const { return xferQueue_; }

  //! Retrieves the internal format from the OCL format
  CalFormat getCalFormat(const amd::Image::Format& format  //! OCL image format
                         ) const;

  //! Retrieves the OCL format from the internal image format
  amd::Image::Format getOclFormat(const CalFormat& format  //! Internal image format
                                  ) const;

  const ScratchBuffer* scratch(uint idx) const { return scratch_[idx]; }

  //! Returns the global scratch buffer
  Memory* globalScratchBuf() const { return globalScratchBuf_; };

  //! Destroys scratch buffer memory
  void destroyScratchBuffers();

  //! Initialize heap resources if uninitialized
  bool initializeHeapResources();

  //! Set GSL sampler to the specified state
  void fillHwSampler(uint32_t state,                       //!< Sampler's OpenCL state
                     void* hwState,                        //!< Sampler's HW state
                     uint32_t hwStateSize,                 //!< Size of sampler's HW state
                     uint32_t mipFilter = CL_FILTER_NONE,  //!< Mip filter
                     float minLod = 0.f,                   //!< Min level of detail
                     float maxLod = CL_MAXFLOAT            //!< Max level of detail
                     ) const;

  //! host memory alloc
  virtual void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const;

  //! SVM allocation
  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags, void* svmPtr) const;

  //! Free host SVM memory
  void hostFree(void* ptr, size_t size) const;

  //! SVM free
  virtual void svmFree(void* ptr) const;

  //! Returns SRD manger object
  SrdManager& srds() const { return *srdManager_; }

  //! Initial the Hardware Debug Manager
  int32_t hwDebugManagerInit(amd::Context* context, uintptr_t messageStorage);

 private:
  //! Disable copy constructor
  Device(const Device&);

  //! Disable assignment
  Device& operator=(const Device&);

  //! Sends the stall command to all queues
  bool stallQueues();

  //! Buffer allocation
  gpu::Memory* createBuffer(amd::Memory& owner,  //!< Abstraction layer memory object
                            bool directAccess    //!< Use direct host memory access
                            ) const;

  //! Image allocation
  gpu::Memory* createImage(amd::Memory& owner,  //!< Abstraction layer memory object
                           bool directAccess    //!< Use direct host memory access
                           ) const;

  //! Allocates/reallocates the scratch buffer, according to the usage
  bool allocScratch(uint regNum,            //!< Number of the scratch registers
                    const VirtualGPU* vgpu  //!< Virtual GPU for the allocation
                    );

  amd::Context* context_;   //!< A dummy context for internal allocations
  Heap heap_;               //!< GPU global heap
  amd::Memory* dummyPage_;  //!< A dummy page for NULL pointer

  amd::Monitor* lockAsyncOps_;             //!< Lock to serialise all async ops on this device
  amd::Monitor* lockAsyncOpsForInitHeap_;  //!< Lock to serialise all async ops on initialization
                                           //!heap operation
  amd::Monitor* vgpusAccess_;              //!< Lock to serialise virtual gpu list access
  amd::Monitor* scratchAlloc_;             //!< Lock to serialise scratch allocation
  amd::Monitor* mapCacheOps_;              //!< Lock to serialise cache for the map resources

  XferBuffers* xferRead_;   //!< Transfer buffers read
  XferBuffers* xferWrite_;  //!< Transfer buffers write

  std::vector<amd::Memory*>* mapCache_;  //!< Map cache info structure
  ResourceCache* resourceCache_;         //!< Resource cache
  Engines engines_;                      //!< Available engines on device
  bool heapInitComplete_;                //!< Keep track of initialization status of heap resources
  VirtualGPU* xferQueue_;                //!< Transfer queue
  std::vector<ScratchBuffer*> scratch_;  //!< Scratch buffers for kernels
  Memory* globalScratchBuf_;             //!< Global scratch buffer
  SrdManager* srdManager_;               //!< SRD manager object

  static AppProfile appProfile_;  //!< application profile
};

/*@}*/} // namespace gpu

#endif /*GPU_HPP_*/
