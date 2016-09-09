//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
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
#include "hsa_ext_finalize.h"
#include "hsa_ext_amd.h"

#include <iostream>
#include <vector>

// extern hsa::Runtime* g_hsaruntime;

/*! \addtogroup HSA
 *  @{
 */

//! HSA Device Implementation
namespace roc {

/**
 * @brief List of environment variables that could be used to
 * configure the behavior of Hsa Runtime
 */
#define ENVVAR_HSA_POLL_KERNEL_COMPLETION     "HSA_POLL_COMPLETION"

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

//A NULL Device type used only for offline compilation
// Only functions that are used for compilation will be in this device
class NullDevice : public amd::Device {
public:
    //! constructor
    NullDevice(){};

    //!create the device
    bool create(const AMDDeviceInfo& deviceInfo);

    //! Initialise all the offline devices that can be used for compilation
    static bool init();
    //! Teardown for offline devices
    static void tearDown();

    //! Destructor for the Null device
    virtual ~NullDevice();

    Compiler* compiler() const { return compilerHandle_; }

    //! Construct an HSAIL program object from the ELF assuming it is valid
    virtual device::Program *createProgram(amd::option::Options* options = NULL);
    const AMDDeviceInfo& deviceInfo() const {
        return deviceInfo_;
    }
    //! Gets the backend device for the NULL device type
    virtual hsa_agent_t getBackendDevice() const {
        ShouldNotReachHere();
        const hsa_agent_t kInvalidAgent = { 0 };
        return kInvalidAgent;
    }

    //List of dummy functions which are disabled for NullDevice

    //! Create sub-devices according to the given partition scheme.
    virtual cl_int createSubDevices(
        device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices) {
            ShouldNotReachHere();
            return CL_INVALID_VALUE; };

    //! Create a new virtual device environment.
    virtual device::VirtualDevice* createVirtualDevice(
        amd::CommandQueue* queue = NULL) {
        ShouldNotReachHere();
        return NULL;
    }

    virtual bool registerSvmMemory(void* ptr, size_t size) const {
        ShouldNotReachHere();
        return false;
    }

    virtual void deregisterSvmMemory(void* ptr) const {
        ShouldNotReachHere();
    }

    //! Just returns NULL for the dummy device
    virtual device::Memory* createMemory(amd::Memory& owner) const {
        ShouldNotReachHere();
        return NULL; }

    //! Sampler object allocation
    virtual bool createSampler(
        const amd::Sampler& owner,  //!< abstraction layer sampler object
        device::Sampler**   sampler //!< device sampler object
        ) const
    {
        ShouldNotReachHere();
        return true;
    }

    //! Just returns NULL for the dummy device
    virtual device::Memory* createView(
        amd::Memory& owner,             //!< Owner memory object
        const device::Memory& parent    //!< Parent device memory object for the view
        ) const {
            ShouldNotReachHere();
            return NULL;
    }

    //! Just returns NULL for the dummy device
    virtual void* svmAlloc(
        amd::Context& context,    //!< The context used to create a buffer
        size_t size,                    //!< size of svm spaces
        size_t alignment,               //!< alignment requirement of svm spaces
        cl_svm_mem_flags flags,         //!< flags of creation svm spaces
        void* svmPtr                    //!< existing svm pointer for mGPU case
        ) const {
            ShouldNotReachHere();
            return NULL;
    }

    //! Just returns NULL for the dummy device
    virtual void svmFree(
        void* ptr                    //!< svm pointer needed to be freed
        ) const {
            ShouldNotReachHere();
            return;
    }

    //! Reallocates the provided buffer object
    virtual bool reallocMemory(amd::Memory& owner) const {
        ShouldNotReachHere();
        return false;
    }

    //! Acquire external graphics API object in the host thread
    //! Needed for OpenGL objects on CPU device

    virtual bool bindExternalDevice(
        uint flags, void* const pDevice[], void* pContext, bool validateOnly) {
            ShouldNotReachHere();
            return false;
    }

    virtual bool unbindExternalDevice(
        uint flags, void* const pDevice[], void* pContext, bool validateOnly) {
            ShouldNotReachHere();
            return false;
    }

    //! Releases non-blocking map target memory
    virtual void freeMapTarget(amd::Memory& mem, void* target) { ShouldNotReachHere();}

    //! Empty implementation on Null device
    virtual bool globalFreeMemory(size_t* freeMemory) const {
        ShouldNotReachHere();
        return false;
    }

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
    //! Initialise the whole HSA device subsystem (CAL init, device enumeration, etc).
    static bool init();
    static void tearDown();

    //! Lookup all AMD HSA devices and memory regions.
    static hsa_status_t iterateAgentCallback(hsa_agent_t agent, void *data);
    static hsa_status_t iterateGpuMemoryPoolCallback(
        hsa_amd_memory_pool_t region, void* data);
    static hsa_status_t iterateCpuMemoryPoolCallback(
      hsa_amd_memory_pool_t region, void* data);

    static bool loadHsaModules();

    bool create();

    //! Construct a new physical HSA device
    Device(hsa_agent_t bkendDevice);
    virtual hsa_agent_t getBackendDevice() const { return _bkendDevice; }

    static const std::vector<hsa_agent_t>& getGpuAgents() {
      return gpu_agents_;
    }

    static hsa_agent_t getCpuAgent()
    {
        return cpu_agent_;
    }

    //! Destructor for the physical HSA device
    virtual ~Device();

    bool mapHSADeviceToOpenCLDevice(hsa_agent_t hsadevice);

    // Temporary, delete it later when HSA Runtime and KFD is fully fucntional.
    void fake_device();

    ///////////////////////////////////////////////////////////////////////////////
    // TODO: Below are all mocked up virtual functions from amd::Device, they may
    // need real implementation.
    ///////////////////////////////////////////////////////////////////////////////

// #ifdef cl_ext_device_fission
    //! Create sub-devices according to the given partition scheme.
    virtual cl_int createSubDevices(
        device::CreateSubDevicesInfo &create_inf,
        cl_uint num_entries,
        cl_device_id *devices,
        cl_uint *num_devices)
        { return CL_INVALID_VALUE; }
// #endif // cl_ext_device_fission

    // bool Device::create(CALuint ordinal);

    //! Instantiate a new virtual device
    virtual device::VirtualDevice *createVirtualDevice(
        amd::CommandQueue* queue = NULL);

    //! Construct an HSAIL program object from the ELF assuming it is valid
    virtual device::Program *createProgram(amd::option::Options* options = NULL);

    virtual device::Memory *createMemory(amd::Memory &owner) const;

    //! Sampler object allocation
    virtual bool createSampler(
        const amd::Sampler& owner,  //!< abstraction layer sampler object
        device::Sampler**   sampler //!< device sampler object
        ) const
    {
        //! \todo HSA team has to implement sampler allocation
        *sampler = NULL;
        return true;
    }


    //! Just returns NULL for the dummy device
    virtual device::Memory *createView(
        amd::Memory &owner,     //!< Owner memory object
        const device::Memory &parent //!< Parent device memory object for the view
        ) const { return NULL; }

    //! Reallocates the provided buffer object
    virtual bool reallocMemory(amd::Memory &owner) const {return true; }

    //! Acquire external graphics API object in the host thread
    //! Needed for OpenGL objects on CPU device
    virtual bool bindExternalDevice(
        uint flags, void * const pDevice[], void *pContext, bool validateOnly);

    /**
     * @brief Removes the external device as an available device.
     *
     * @note: The current implementation is to avoid build break
     * and does not represent actual / correct implementation. This
     * needs to be done.
     */
    bool unbindExternalDevice(
        uint flags,      //!< Enum val. for ext.API type: GL, D3D10, etc.
        void * const gfxDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
        void *gfxContext,   //!< HGLRC/GLXContext handle
        bool validateOnly   //!< Only validate if the device can inter-operate with
                            //!< pDevice/pContext, do not bind.
        );

    //! Gets free memory on a GPU device
    virtual bool globalFreeMemory(size_t *freeMemory) const;

    virtual void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const;

    virtual void hostFree(void* ptr, size_t size = 0) const;

    void *deviceLocalAlloc(size_t size) const;

    void memFree(void *ptr, size_t size) const;

    virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags = CL_MEM_READ_WRITE, void* svmPtr = NULL) const;

    virtual void svmFree(void* ptr) const;

    const Settings &settings() const { return reinterpret_cast<Settings &>(*settings_); }

    //! Returns transfer engine object
    const device::BlitManager& xferMgr() const { return xferQueue()->blitMgr(); }

    const size_t alloc_granularity() const { return alloc_granularity_; }

    const hsa_profile_t agent_profile() const { return agent_profile_; }

    const MesaInterop& mesa() const { return mesa_; }

    //! Finds an appropriate map target
    amd::Memory* findMapTarget(size_t size) const;

    //! Adds a map target to the cache
    bool addMapTarget(amd::Memory* memory) const;

private:
    amd::Monitor*   mapCacheOps_;   //!< Lock to serialise cache for the map resources
    std::vector<amd::Memory*>*  mapCache_;  //!< Map cache info structure

    bool populateOCLDeviceConstants();
    static bool isHsaInitialized_;
    static hsa_agent_t cpu_agent_;
    static std::vector<hsa_agent_t> gpu_agents_;
    MesaInterop mesa_;
    hsa_agent_t _bkendDevice;
    hsa_profile_t agent_profile_;
    hsa_amd_memory_pool_t group_segment_;
    hsa_amd_memory_pool_t system_segment_;
    hsa_amd_memory_pool_t system_coarse_segment_;
    hsa_amd_memory_pool_t gpuvm_segment_;
    size_t gpuvm_segment_max_alloc_;
    size_t alloc_granularity_;
    static const bool offlineDevice_;
    amd::Context *context_; //!< A dummy context for internal data transfer
    VirtualGPU *xferQueue_; //!< Transfer queue, created on demand

    VirtualGPU* xferQueue() const;
};  // class roc::Device
}  // namespace roc

/**
 * @}
 */
#endif  /*WITHOUT_HSA_BACKEND*/

