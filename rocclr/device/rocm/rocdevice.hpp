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

#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "top.hpp"
#include "CL/cl.h"
#include "device/device.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/perfctr.hpp"
#include "platform/memory.hpp"
#include "utils/concurrent.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "utils/versions.hpp"

#include "device/rocm/rocsettings.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rocdefs.hpp"
#include "device/rocm/rocprintf.hpp"
#include "device/rocm/rocglinterop.hpp"

#include "hsa.h"
#include "hsa_ext_image.h"
#include "hsa_ext_amd.h"
#include "hsa_ven_amd_loader.h"

#include <iostream>
#include <vector>
#include <memory>

/*! \addtogroup HSA
 *  @{
 */

//! HSA Device Implementation
namespace roc {

/**
 * @brief List of environment variables that could be used to
 * configure the behavior of Hsa Runtime
 */
#define ENVVAR_HSA_POLL_KERNEL_COMPLETION "HSA_POLL_COMPLETION"

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
class IProDevice;

class Sampler : public device::Sampler {
 public:
  //! Constructor
    Sampler(const Device& dev) : dev_(dev) {}

  //! Default destructor for the device memory object
  virtual ~Sampler();

  //! Creates a device sampler from the OCL sampler state
  bool create(const amd::Sampler& owner  //!< AMD sampler object
              );

 private:
  void fillSampleDescriptor(hsa_ext_sampler_descriptor_t& samplerDescriptor,
                            const amd::Sampler& sampler) const;
  Sampler& operator=(const Sampler&);

  //! Disable operator=
  Sampler(const Sampler&);

  const Device& dev_;  //!< Device object associated with the sampler

  hsa_ext_sampler_t hsa_sampler;
};

// A NULL Device type used only for offline compilation
// Only functions that are used for compilation will be in this device
class NullDevice : public amd::Device {
 public:
  //! constructor
  NullDevice(){};

  //! create the device
  bool create(const AMDDeviceInfo& deviceInfo);

  //! Initialise all the offline devices that can be used for compilation
  static bool init();
  //! Teardown for offline devices
  static void tearDown();

  //! Destructor for the Null device
  virtual ~NullDevice();

  Compiler* compiler() const { return compilerHandle_; }

  const Settings& settings() const { return reinterpret_cast<Settings&>(*settings_); }

  //! Construct an HSAIL program object from the ELF assuming it is valid
  virtual device::Program* createProgram(amd::Program& owner, amd::option::Options* options = nullptr);
  const AMDDeviceInfo& deviceInfo() const { return deviceInfo_; }
  //! Gets the backend device for the Null device type
  virtual hsa_agent_t getBackendDevice() const {
    ShouldNotReachHere();
    const hsa_agent_t kInvalidAgent = {0};
    return kInvalidAgent;
  }

  // List of dummy functions which are disabled for NullDevice

  //! Create a new virtual device environment.
  virtual device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr) {
    ShouldNotReachHere();
    return nullptr;
  }

  virtual bool registerSvmMemory(void* ptr, size_t size) const {
    ShouldNotReachHere();
    return false;
  }

  virtual void deregisterSvmMemory(void* ptr) const { ShouldNotReachHere(); }

  //! Just returns nullptr for the dummy device
  virtual device::Memory* createMemory(amd::Memory& owner) const {
    ShouldNotReachHere();
    return nullptr;
  }

  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
                             ) const {
    ShouldNotReachHere();
    return true;
  }

  //! Just returns nullptr for the dummy device
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
      ) const {
    ShouldNotReachHere();
    return nullptr;
  }

  //! Just returns nullptr for the dummy device
  virtual void* svmAlloc(amd::Context& context,   //!< The context used to create a buffer
                         size_t size,             //!< size of svm spaces
                         size_t alignment,        //!< alignment requirement of svm spaces
                         cl_svm_mem_flags flags,  //!< flags of creation svm spaces
                         void* svmPtr             //!< existing svm pointer for mGPU case
                         ) const {
    ShouldNotReachHere();
    return nullptr;
  }

  //! Just returns nullptr for the dummy device
  virtual void svmFree(void* ptr  //!< svm pointer needed to be freed
                       ) const {
    ShouldNotReachHere();
    return;
  }

  //! Determine if we can use device memory for SVM
  const bool forceFineGrain(amd::Memory* memory) const {
    return !settings().enableCoarseGrainSVM_ || (memory->getContext().devices().size() > 1);
  }

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device

  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly) {
    ShouldNotReachHere();
    return false;
  }

  virtual bool unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                    bool validateOnly) {
    ShouldNotReachHere();
    return false;
  }

  //! Releases non-blocking map target memory
  virtual void freeMapTarget(amd::Memory& mem, void* target) { ShouldNotReachHere(); }

  //! Empty implementation on Null device
  virtual bool globalFreeMemory(size_t* freeMemory) const {
    ShouldNotReachHere();
    return false;
  }

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput, cl_set_device_clock_mode_output_amd* pSetClockModeOutput) { return true; }

 protected:
  //! Initialize compiler instance and handle
  static bool initCompiler(bool isOffline);
  //! destroy compiler instance and handle
  static bool destroyCompiler();
  //! Handle to the the compiler
  static Compiler* compilerHandle_;
  //! Device Id for an HsaDevice
  AMDDeviceInfo deviceInfo_;

 private:
  static const bool offlineDevice_;
};

//! A HSA device ordinal (physical HSA device)
class Device : public NullDevice {
 public:
  //! Transfer buffers
  class XferBuffers : public amd::HeapObject {
   public:
    static const size_t MaxXferBufListSize = 8;

    //! Default constructor
    XferBuffers(const Device& device, size_t bufSize)
        : bufSize_(bufSize), acquiredCnt_(0), gpuDevice_(device) {}

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

    size_t bufSize_;                  //!< Staged buffer size
    std::list<Memory*> freeBuffers_;  //!< The list of free buffers
    amd::Atomic<uint> acquiredCnt_;   //!< The total number of acquired buffers
    amd::Monitor lock_;               //!< Stgaed buffer acquire/release lock
    const Device& gpuDevice_;         //!< GPU device object
  };

  //! Initialise the whole HSA device subsystem (CAL init, device enumeration, etc).
  static bool init();
  static void tearDown();

  //! Lookup all AMD HSA devices and memory regions.
  static hsa_status_t iterateAgentCallback(hsa_agent_t agent, void* data);
  static hsa_status_t iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t region, void* data);
  static hsa_status_t iterateCpuMemoryPoolCallback(hsa_amd_memory_pool_t region, void* data);
  static hsa_status_t loaderQueryHostAddress(const void* device, const void** host);

  static bool loadHsaModules();

  bool create(bool sramEccEnabled);

  //! Construct a new physical HSA device
  Device(hsa_agent_t bkendDevice);
  virtual hsa_agent_t getBackendDevice() const { return _bkendDevice; }

  static const std::vector<hsa_agent_t>& getGpuAgents() { return gpu_agents_; }

  static hsa_agent_t getCpuAgent() { return cpu_agent_; }

  //! Destructor for the physical HSA device
  virtual ~Device();

  // Temporary, delete it later when HSA Runtime and KFD is fully fucntional.
  void fake_device();

  ///////////////////////////////////////////////////////////////////////////////
  // TODO: Below are all mocked up virtual functions from amd::Device, they may
  // need real implementation.
  ///////////////////////////////////////////////////////////////////////////////

  //! Instantiate a new virtual device
  virtual device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr);

  //! Construct an HSAIL program object from the ELF assuming it is valid
  virtual device::Program* createProgram(amd::Program& owner, amd::option::Options* options = nullptr);

  virtual device::Memory* createMemory(amd::Memory& owner) const;

  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
                             ) const;

  //! Just returns nullptr for the dummy device
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
      ) const {
    return nullptr;
  }

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device
  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly);

  /**
   * @brief Removes the external device as an available device.
   *
   * @note: The current implementation is to avoid build break
   * and does not represent actual / correct implementation. This
   * needs to be done.
   */
  bool unbindExternalDevice(
      uint flags,               //!< Enum val. for ext.API type: GL, D3D10, etc.
      void* const gfxDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
      void* gfxContext,         //!< HGLRC/GLXContext handle
      bool validateOnly         //!< Only validate if the device can inter-operate with
                                //!< pDevice/pContext, do not bind.
      );

  //! Gets free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory) const;

  virtual void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const;

  virtual void hostFree(void* ptr, size_t size = 0) const;

  void* deviceLocalAlloc(size_t size, bool atomics = false) const;

  void memFree(void* ptr, size_t size) const;

  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags = CL_MEM_READ_WRITE, void* svmPtr = nullptr) const;

  virtual void svmFree(void* ptr) const;

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput, cl_set_device_clock_mode_output_amd* pSetClockModeOutput);

  //! Returns transfer engine object
  const device::BlitManager& xferMgr() const { return xferQueue()->blitMgr(); }

  const size_t alloc_granularity() const { return alloc_granularity_; }

  const hsa_profile_t agent_profile() const { return agent_profile_; }

  //! Finds an appropriate map target
  amd::Memory* findMapTarget(size_t size) const;

  //! Adds a map target to the cache
  bool addMapTarget(amd::Memory* memory) const;

  //! Returns transfer buffer object
  XferBuffers& xferWrite() const { return *xferWrite_; }

  //! Returns transfer buffer object
  XferBuffers& xferRead() const { return *xferRead_; }

  //! Returns a ROC memory object from AMD memory object
  roc::Memory* getRocMemory(amd::Memory* mem  //!< Pointer to AMD memory object
                            ) const;

  amd::Context& context() const { return *context_; }

  //! Create internal blit program
  bool createBlitProgram();

  // Returns AMD GPU Pro interfaces
  const IProDevice& iPro() const { return *pro_device_; }
  bool ProEna() const  { return pro_ena_; }

  // P2P agents avaialble for this device
  const std::vector<hsa_agent_t>& p2pAgents() const { return p2p_agents_; }

  // Update the global free memory size
  void updateFreeMemory(size_t size, bool free);

  virtual amd::Memory* IpcAttach(const void* handle, size_t mem_size, unsigned int flags, void** dev_ptr) const;
  virtual void IpcDetach (amd::Memory& memory) const;

  bool AcquireExclusiveGpuAccess();
  void ReleaseExclusiveGpuAccess(VirtualGPU& vgpu) const;

  //! Returns the lock object for the virtual gpus list
  amd::Monitor& vgpusAccess() const { return vgpusAccess_; }

  typedef std::vector<VirtualGPU*> VirtualGPUs;
    //! Returns the list of all virtual GPUs running on this device
  const VirtualGPUs& vgpus() const { return vgpus_; }
  VirtualGPUs vgpus_;  //!< The list of all running virtual gpus (lock protected)

  VirtualGPU* xferQueue() const;

  hsa_amd_memory_pool_t SystemSegment() const { return system_segment_; }

  hsa_amd_memory_pool_t SystemCoarseSegment() const { return system_coarse_segment_; }

  //! Acquire HSA queue. This method can create a new HSA queue or
  //! share previously created
  hsa_queue_t* acquireQueue(uint32_t queue_size_hint, bool coop_queue = false);

  //! Release HSA queue
  void releaseQueue(hsa_queue_t*);

  //! For the given HSA queue, return an existing hostcall buffer or create a
  //! new one. queuePool_ keeps a mapping from HSA queue to hostcall buffer.
  void* getOrCreateHostcallBuffer(hsa_queue_t* queue);

  //! Return multi GPU grid launch sync buffer
  address MGSync() const { return mg_sync_; }

  virtual bool findLinkTypeAndHopCount(amd::Device* other_device, uint32_t* link_type,
                                       uint32_t* hop_count);

  //! Returns a GPU memory object from AMD memory object
  roc::Memory* getGpuMemory(amd::Memory* mem  //!< Pointer to AMD memory object
                            ) const;

 private:
  static hsa_ven_amd_loader_1_00_pfn_t amd_loader_ext_table;

  amd::Monitor* mapCacheOps_;            //!< Lock to serialise cache for the map resources
  std::vector<amd::Memory*>* mapCache_;  //!< Map cache info structure

  bool populateOCLDeviceConstants();
  static bool isHsaInitialized_;
  static hsa_agent_t cpu_agent_;
  static std::vector<hsa_agent_t> gpu_agents_;
  std::vector<hsa_agent_t> p2p_agents_;  //!< List of P2P agents available for this device
  hsa_agent_t _bkendDevice;
  hsa_agent_t* p2p_agents_list_;
  hsa_profile_t agent_profile_;
  hsa_amd_memory_pool_t group_segment_;
  hsa_amd_memory_pool_t system_segment_;
  hsa_amd_memory_pool_t system_coarse_segment_;
  hsa_amd_memory_pool_t gpuvm_segment_;
  hsa_amd_memory_pool_t gpu_fine_grained_segment_;
  size_t gpuvm_segment_max_alloc_;
  size_t alloc_granularity_;
  static const bool offlineDevice_;
  amd::Context* context_;  //!< A dummy context for internal data transfer
  VirtualGPU* xferQueue_;  //!< Transfer queue, created on demand

  XferBuffers* xferRead_;   //!< Transfer buffers read
  XferBuffers* xferWrite_;  //!< Transfer buffers write
  const IProDevice* pro_device_;  //!< AMDGPUPro device
  bool  pro_ena_;           //!< Extra functionality with AMDGPUPro device, beyond ROCr
  std::atomic<size_t> freeMem_;   //!< Total of free memory available
  mutable amd::Monitor vgpusAccess_;     //!< Lock to serialise virtual gpu list access
  bool hsa_exclusive_gpu_access_;  //!< TRUE if current device was moved into exclusive GPU access mode
  static address mg_sync_;  //!< MGPU grid launch sync memory (SVM location)

  struct QueueInfo {
    int refCount;
    void* hostcallBuffer_;
  };
  std::map<hsa_queue_t*, QueueInfo> queuePool_;  //!< Pool of HSA queues for recycling

 public:
  amd::Atomic<uint> numOfVgpus_;  //!< Virtual gpu unique index
};                                // class roc::Device
}  // namespace roc

/**
 * @}
 */
#endif /*WITHOUT_HSA_BACKEND*/
