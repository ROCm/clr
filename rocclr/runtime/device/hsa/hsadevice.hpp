//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef _OPENCL_RUNTIME_DEVICE_HSA_HSADEVICE_HPP_
#define _OPENCL_RUNTIME_DEVICE_HSA_HSADEVICE_HPP_

#ifndef WITHOUT_FSA_BACKEND

#include "top.hpp"
#include "device/device.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/perfctr.hpp"
#include "platform/memory.hpp"
#include "utils/concurrent.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "utils/versions.hpp"
#include "aclTypes.h"

#include "device/hsa/hsasettings.hpp"
#include "device/hsa/hsavirtual.hpp"
#include "device/hsa/hsadefs.hpp"

#include "newcore.h"

#include <iostream>

// extern hsa::Runtime* g_hsaruntime;

/*! \addtogroup HSA
 *  @{
 */

//! HSA Device Implementation
namespace oclhsa {

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

    aclCompiler *compiler() const { return compilerHandle_; }

    //! Construct an HSAIL program object from the ELF assuming it is valid
    virtual device::Program *createProgram(int oclVer = 120);

    const AMDDeviceInfo& deviceInfo() const {
        return deviceInfo_;
    }
    //! Gets the backend device for the NULL device type
    virtual const HsaDevice* getBackendDevice() const {
        ShouldNotReachHere();
        return NULL;
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
        bool    profiling,
        bool    interopQueue
#if cl_amd_open_video
        , void*   calVideoProperties = NULL
#endif // cl_amd_open_video
        , uint  deviceQueueSize = 0
        ) {
            ShouldNotReachHere();
            return NULL;
    };

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
        amd::Context& context,          //!< The context used to create a buffer
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
        intptr_t type, void* pDevice, void* pContext, bool validateOnly) {
            ShouldNotReachHere();
            return false;
    }

    virtual bool unbindExternalDevice(
        intptr_t type, void* pDevice, void* pContext, bool validateOnly) {
            ShouldNotReachHere();
            return false;
    }

    //! Gets a pointer to a region of host-visible memory for use as the target
    //! of a non-blocking map for a given memory object
    virtual void* allocMapTarget(
        amd::Memory&    mem,        //!< Abstraction layer memory object
        const amd::Coord3D& origin, //!< The map location in memory
        const amd::Coord3D& region, //!< The map region in memory
        uint    mapFlags,           //!< Map flags
        size_t* rowPitch = NULL,    //!< Row pitch for the mapped memory
        size_t* slicePitch = NULL   //!< Slice for the mapped memory
        ) {
            ShouldNotReachHere();
            return NULL;
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
    static aclCompiler* compilerHandle_;
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

    static bool loadHsaModules();

    bool create();

    //! Construct a new physical HSA device
    Device(const HsaDevice *bkendDevice);
    virtual const HsaDevice *getBackendDevice() const
    {
        return (_bkendDevice);
    }

    //! Destructor for the physical HSA device
    virtual ~Device();

    bool mapHSADeviceToOpenCLDevice(const HsaDevice *hsadevice);

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
        bool profiling, bool interopQueue
#if cl_amd_open_video
        , void *calVideoProperties = NULL
#endif   // cl_amd_open_vide
        , uint  deviceQueueSize = 0
        );
    //! Construct an HSAIL program object from the ELF assuming it is valid
    virtual device::Program *createProgram(int oclVer = 120);

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
        intptr_t type, void *pDevice, void *pContext, bool validateOnly);

    /**
     * @brief Removes the external device as an available device.
     *
     * @note: The current implementation is to avoid build break
     * and does not represent actual / correct implementation. This
     * needs to be done.
     */
    bool unbindExternalDevice(
        intptr_t type,      //!< Enum val. for ext.API type: GL, D3D10, etc.
        void *gfxDevice,    //!< D3D device do D3D, HDC/Display handle of X Window for GL
        void *gfxContext,   //!< HGLRC/GLXContext handle
        bool validateOnly   //!< Only validate if the device can inter-operate with
                            //!< pDevice/pContext, do not bind.
        );

    //! Gets a pointer to a region of host-visible memory for use as the target
    //! of a non-blocking map for a given memory object
    virtual void *allocMapTarget(
        amd::Memory &mem,   //!< Abstraction layer memory object
        const amd::Coord3D &origin, //!< The map location in memory
        const amd::Coord3D &region, //!< The map region in memory
        uint    mapFlags,           //!< Map flags
        size_t *rowPitch = NULL,    //!< Row pitch for the mapped memory
        size_t *slicePitch = NULL   //!< Slice for the mapped memory
        );

    //! Gets free memory on a GPU device
    virtual bool globalFreeMemory(size_t *freeMemory) const;

    virtual void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const;

    virtual void hostFree(void* ptr, size_t size = 0) const;

    virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags = CL_MEM_READ_WRITE, void* svmPtr = NULL) const;

    virtual void svmFree(void* ptr) const;

    //! Returns a OCLHSA memory object from AMD memory object
    oclhsa::Memory* getOclHsaMemory(
        amd::Memory* mem    //!< Pointer to AMD memory object
        ) const;

    const Settings &settings() const { return reinterpret_cast<Settings &>(*settings_); }

    //! Returns transfer engine object
    const device::BlitManager& xferMgr() const { return xferQueue()->blitMgr();}

private:
    bool populateOCLDeviceConstants();

    cl_device_svm_capabilities getSvmCapabilities(const HsaDevice* device);

    VirtualGPU* xferQueue() const;

    static bool isHsaInitialized_;
    const HsaDevice *_bkendDevice;
    static const bool offlineDevice_;
    amd::Context *context_; //!< A dummy context for internal data transfer
    VirtualGPU *xferQueue_; //!< Transfer queue, created on demand
};  // class oclhsa::Device
}  // namespace oclhsa

/**
 * @}
 */
#endif  /*WITHOUT_FSA_BACKEND*/
#endif  /*HSA_HPP_*/
