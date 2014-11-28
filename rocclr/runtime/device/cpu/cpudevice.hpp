//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CPUDEVICE_HPP_
#define CPUDEVICE_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "device/cpu/cpuvirtual.hpp"
#include "device/cpu/cpusettings.hpp"
#include "os/os.hpp"

#if defined(__linux__) && defined(NUMA_SUPPORT)
#include <numa.h>
#endif

#include "acl.h"

//! \namespace cpu CPU Device Implementation
namespace cpu {

//! Maximum number of the supported samplers
const static uint32_t MaxSamplers   = 16;
//! Maximum number of supported read images
const static uint32_t MaxReadImage  = 128;
//! Maximum number of supported write images
const static uint32_t MaxWriteImage = 64;
//! Maximum number of supported read/write images
const static uint32_t MaxReadWriteImage = 64;

/*! \addtogroup CPU CPU Device Implementation
 *  @{
 *
 *  \addtogroup CPUDevice Device
 *
 *  \copydoc cpu::Device
 *
 *  @{
 */

//! A CPU device ordinal
class Device : public amd::Device
{
protected:
    static aclCompiler* compiler_;
public:
    aclCompiler* compiler() const { return compiler_; }

public:
    static bool init(void);

    //! Shutdown CPU device
    static void tearDown();

    //! Construct a new identifier
    Device(Device* parent = NULL) : 
        amd::Device(parent), 
        workerThreadsAffinity_(NULL)
    {}

    virtual ~Device();

    bool create();

    virtual cl_int createSubDevices(
        device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices);

    //! Instantiate a new virtual device
    virtual device::VirtualDevice* createVirtualDevice(
        bool    profiling,
        bool   interopQueue
#if cl_amd_open_video
        , void* calVideoProperties = NULL
#endif // cl_amd_open_video
        , uint  deviceQueueSize = 0
        )
    {
        VirtualCPU* virtualCpu = new VirtualCPU(*this);
        if (virtualCpu != NULL && !virtualCpu->acceptingCommands()) {
            virtualCpu->terminate();
            delete virtualCpu;
            virtualCpu = NULL;
        }
        return virtualCpu;
    }

    //! Compile the given source code.
    virtual device::Program* createProgram(int oclVer = 120);

    //! Just returns NULL as CPU devices use the host memory
    virtual device::Memory* createMemory(amd::Memory& owner) const
    {
        return NULL;
    }

    //! Sampler object allocation
    virtual bool createSampler(
        const amd::Sampler& owner,  //!< abstraction layer sampler object
        device::Sampler**   sampler //!< device sampler object
        ) const
    {
        // Just return NULL on CPU device
        *sampler = NULL;
        return true;
    }

    //! Reallocates device memory obje
    virtual bool reallocMemory(amd::Memory& owner) const
    {
        return true;
    }

    //! Just returns NULL as CPU devices use the host memory
    virtual device::Memory* createView(
        amd::Memory&            owner,  //!< Owner memory object
        const device::Memory&   parent  //!< Parent device memory object for the view
        ) const
    {
        return NULL;
    }

    //! Acquire external graphics API object in the host thread
    //! Needed for OpenGL objects on CPU device

    //! Return true if initialized interoperability, otherwise false
    virtual bool bindExternalDevice(intptr_t type, void* pDevice, void* pContext, bool validateOnly)
    {
        return true;    // On CPU always avail if pD3DDevice is not NULL
    }

    virtual bool unbindExternalDevice(intptr_t type, void* pDevice, void* pContext, bool validateOnly)
    {
        return true;
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
        );

    //! Releases non-blocking map target memory
    virtual void freeMapTarget(amd::Memory& mem, void* target);

    //! Empty implementation on a CPU device
    virtual bool globalFreeMemory(size_t* freeMemory) const { return false; }

     //! Get CPU device settings
    const cpu::Settings& settings() const
        { return reinterpret_cast<cpu::Settings&>(*settings_); }

    bool hasAVXInstructions() const
        { return (settings().cpuFeatures_ & Settings::AVXInstructions) ? true : false; }

    bool hasFMA4Instructions() const
        { return (settings().cpuFeatures_ & Settings::FMA4Instructions) ? true : false; }

    static size_t getMaxWorkerThreadsNumber() { return maxWorkerThreads_; }

    void setWorkerThreadsAffinity(
        cl_uint numWorkerThreads, 
        const amd::Os::ThreadAffinityMask* threadsAffinityMask,
        uint& baseCoreId);

    const amd::Os::ThreadAffinityMask* getWorkerThreadsAffinity() const
    {
        return workerThreadsAffinity_;
    }
    //! host memory alloc
    virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags, void* svmPtr) const
    {
        return NULL;
    }

    //! host memory deallocation
    virtual void svmFree(void* ptr) const
    {
        return;
    }
private:
    bool initSubDevice(
        device::Info& info,
        cl_uint maxComputeUnits,
        const device::CreateSubDevicesInfo& create_info);

    cl_int partitionEqually(
        const device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices);

    cl_int partitionByCounts(
        const device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices);

    cl_int partitionByAffinityDomainNUMA(
        const device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices);

    cl_int partitionByAffinityDomainCacheLevel(
        const device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices);

private:
#if defined(__linux__) && defined(NUMA_SUPPORT)
public:
    const nodemask_t* getNumaMask() const
    {
        return (info_.partitionCreateInfo_.type_ == device::PartitionType::BY_AFFINITY_DOMAIN &&
            info_.partitionCreateInfo_.byAffinityDomain_.numa_) ? 
            numaMask_ : NULL;
    }

private:
    union {
        nodemask_t* numaMask_;
        amd::Os::ThreadAffinityMask* workerThreadsAffinity_; //!< As the number of compute units.
    };
#else
    amd::Os::ThreadAffinityMask* workerThreadsAffinity_; //!< As the number of compute units.
#endif

    static size_t maxWorkerThreads_;  //!< Maximum number of Worker Threads
};

/*! @}
 *  @}
 */

} // namespace cpu

#endif // CPUDEVICE_HPP_
