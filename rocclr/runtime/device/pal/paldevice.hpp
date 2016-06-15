//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

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
#include "device/pal/palvirtual.hpp"
#include "device/pal/palmemory.hpp"
#include "device/pal/paldefs.hpp"
#include "device/pal/palsettings.hpp"
#include "device/pal/palappprofile.hpp"
#include "acl.h"
#include "memory"

/*! \addtogroup PAL
 *  @{
 */

//! PAL Device Implementation
namespace pal {

//! A nil device object
class NullDevice : public amd::Device
{
protected:
    static aclCompiler* compiler_;
public:
    aclCompiler* compiler() const { return compiler_; }

public:
    static bool init(void);

    //! Construct a new identifier
    NullDevice();

    //! Creates an offline device with the specified target
    bool create(
        Pal::AsicRevision asicRevision, //!< GPU ASIC revision
        Pal::GfxIpLevel ipLevel         //!< GPU ip level
        );

    virtual cl_int createSubDevices(
        device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices) {
            return CL_INVALID_VALUE;
    }

    //! Instantiate a new virtual device
    virtual device::VirtualDevice* createVirtualDevice(
        amd::CommandQueue*  queue = NULL
        ) { return NULL; }

    //! Compile the given source code.
    virtual device::Program* createProgram(amd::option::Options* options = NULL);

    //! Just returns NULL for the dummy device
    virtual device::Memory* createMemory(amd::Memory& owner) const { return NULL; }

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
        ) const { return NULL; }

    //! Reallocates the provided buffer object
    virtual bool reallocMemory(amd::Memory& owner) const { return true; }

    //! Acquire external graphics API object in the host thread
    //! Needed for OpenGL objects on CPU device

    virtual bool bindExternalDevice(
        uint flags, void* const pDevice[], void* pContext, bool validateOnly) { return true; }

    virtual bool unbindExternalDevice(
        uint flags, void* const pDevice[], void* pContext, bool validateOnly) { return true; }

    //! Releases non-blocking map target memory
    virtual void freeMapTarget(amd::Memory& mem, void* target) {}

    Pal::GfxIpLevel ipLevel() const { return ipLevel_; }
    Pal::AsicRevision asicRevision() const  { return  asicRevision_; }

    const AMDDeviceInfo* hwInfo() const { return hwInfo_; }

    //! Empty implementation on Null device
    virtual bool globalFreeMemory(size_t* freeMemory) const { return false; }

    //! Get GPU device settings
    const pal::Settings& settings() const
        { return reinterpret_cast<pal::Settings&>(*settings_); }
    virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags, void* svmPtr) const { return NULL; }
    virtual void svmFree(void* ptr) const {return;}

protected:
    Pal::AsicRevision   asicRevision_;  //!< ASIC revision
    Pal::GfxIpLevel     ipLevel_;       //!< Device IP level
    const AMDDeviceInfo* hwInfo_;       //!< Device HW info structure

    //! Fills OpenCL device info structure
    void fillDeviceInfo(
        const Pal::DeviceProperties& palProp,//!< PAL device properties
        const Pal::GpuMemoryHeapProperties heaps[Pal::GpuHeapCount],
        size_t  maxTextureSize,             //!< Maximum texture size supported in HW
        uint    numComputeRings             //!< Number of compute rings
        );
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

class Sampler : public device::Sampler
{
public:
    //! Constructor
    Sampler(const Device& dev): dev_(dev) {}

    //! Default destructor for the device memory object
    virtual ~Sampler();

    //! Creates a device sampler from the OCL sampler state
    bool create(
        uint32_t oclSamplerState    //!< OCL sampler state
        );

    //! Creates a device sampler from the OCL sampler state
    bool create(
        const amd::Sampler& owner   //!< AMD sampler object
        );

    const void* hwState() const { return hwState_; }

private:
    //! Disable default copy constructor
    Sampler& operator=(const Sampler&);

    //! Disable operator=
    Sampler(const Sampler&);

    const Device&   dev_;       //!< Device object associated with the sampler
    address         hwState_;   //!< GPU HW state (\todo legacy path)
};

//! A GPU device ordinal (physical GPU device)
class Device : public NullDevice
{
public:
    //! Locks any access to the virtual GPUs
    class ScopedLockVgpus : public amd::StackObject {
    public:
        //! Default constructor
        ScopedLockVgpus(const Device& dev);

        //! Destructor
        ~ScopedLockVgpus();

    private:
        const Device&   dev_;       //! Device object
    };

    //! Transfer buffers
    class XferBuffers : public amd::HeapObject
    {
    public:
        static const size_t MaxXferBufListSize = 8;

        //! Default constructor
        XferBuffers(const Device& device, Resource::MemoryType type, size_t bufSize)
            : type_(type)
            , bufSize_(bufSize)
            , acquiredCnt_(0)
            , gpuDevice_(device)
            {}

        //! Default destructor
        ~XferBuffers();

        //! Creates the xfer buffers object
        bool create();

        //! Acquires an instance of the transfer buffers
        Memory& acquire();

        //! Releases transfer buffer
        void release(
            VirtualGPU& gpu,    //!< Virual GPU object used with the buffer
            Memory& buffer    //!< Transfer buffer for release
            );

        //! Returns the buffer's size for transfer
        size_t  bufSize() const { return bufSize_; }

    private:
        //! Disable copy constructor
        XferBuffers(const XferBuffers&);

        //! Disable assignment operator
        XferBuffers& operator=(const XferBuffers&);

        //! Get device object
        const Device& dev() const { return gpuDevice_; }

        Resource::MemoryType    type_;          //!< The buffer's type
        size_t                  bufSize_;       //!< Staged buffer size
        std::list<Memory*>      freeBuffers_;   //!< The list of free buffers
        amd::Atomic<uint>       acquiredCnt_;   //!< The total number of acquired buffers
        amd::Monitor            lock_;          //!< Stgaed buffer acquire/release lock
        const Device&           gpuDevice_;     //!< GPU device object
    };

    struct ScratchBuffer : public amd::HeapObject
    {
        uint    regNum_;    //!< The number of used scratch registers
        Memory* memObj_;    //!< Memory objects for scratch buffers
        uint    offset_;    //!< Offset from the global scratch store
        uint    size_;      //!< Scratch buffer size on this queue

        //! Default constructor
        ScratchBuffer(): regNum_(0), memObj_(NULL), offset_(0) {}

        //! Default constructor
        ~ScratchBuffer();

        //! Destroys memory objects
        void destroyMemory();
    };


    class SrdManager : public amd::HeapObject {
    public:
        SrdManager(const Device& dev, uint srdSize, uint bufSize)
            : dev_(dev)
            , numFlags_(bufSize / (srdSize * MaskBits))
            , srdSize_(srdSize)
            , bufSize_(bufSize) {}
        ~SrdManager();

        //! Allocates a new SRD slot for a resource
        uint64_t allocSrdSlot(address* cpuAddr);

        //! Frees a SRD slot
        void freeSrdSlot(uint64_t addr);

        // Fills the memory list for VidMM KMD
        void fillResourceList(std::vector<const Memory*>&   memList);

    private:
        //! Disable copy constructor
        SrdManager(const SrdManager&);

        //! Disable assignment operator
        SrdManager& operator=(const SrdManager&);

        struct Chunk {
            Memory* buf_;
            uint*   flags_;
            Chunk(): buf_(NULL), flags_(NULL) {}
        };

        static const uint MaskBits = 32;
        const Device&   dev_;       //!< GPU device for the chunk manager
        amd::Monitor    ml_;        //!< Global lock for the SRD manager
        std::vector<Chunk>  pool_;  //!< Pool of SRD buffers
        uint            numFlags_;  //!< Total number of flags in array
        uint            srdSize_;   //!< SRD size
        uint            bufSize_;   //!< Buffer size that holds SRDs
    };

    //! Initialise the whole GPU device subsystem
    static bool init();

    //! Shutdown the whole GPU device subsystem
    static void tearDown();

    //! Construct a new physical GPU device
    Device();

    //! Initialise a device (i.e. all parts of the constructor that could
    //! potentially fail)
    bool create(
        Pal::IDevice* device    //!< PAL device interface object
        );

    //! Destructor for the physical GPU device
    virtual ~Device();

    //! Instantiate a new virtual device
    device::VirtualDevice* createVirtualDevice(
        amd::CommandQueue*  queue = NULL
        );

    //! Memory allocation
    virtual device::Memory* createMemory(
        amd::Memory&    owner   //!< abstraction layer memory object
        ) const;

    //! Sampler object allocation
    virtual bool createSampler(
        const amd::Sampler& owner,  //!< abstraction layer sampler object
        device::Sampler**   sampler //!< device sampler object
        ) const;

    //! Reallocates the provided buffer object
    virtual bool reallocMemory(
        amd::Memory&    owner   //!< Buffer for reallocation
        ) const;

    //! Allocates a view object from the device memory
    virtual device::Memory* createView(
        amd::Memory&      owner,        //!< Owner memory object
        const device::Memory&   parent  //!< Parent device memory object for the view
        ) const;

    //! Create the device program.
    virtual device::Program* createProgram(amd::option::Options* options = NULL);

    //! Attempt to bind with external graphics API's device/context
    virtual bool bindExternalDevice(
        uint flags, void* const pDevice[], void* pContext, bool validateOnly);

    //! Attempt to unbind with external graphics API's device/context
    virtual bool unbindExternalDevice(
        uint flags, void* const pDevice[], void* pContext, bool validateOnly);

    //! Validates kernel before execution
    virtual bool validateKernel(
        const amd::Kernel& kernel,      //!< AMD kernel object
        const device::VirtualDevice* vdev
        );

    //! Retrieves information about free memory on a GPU device
    virtual bool globalFreeMemory(size_t* freeMemory) const;

    //! Returns a GPU memory object from AMD memory object
    pal::Memory* getGpuMemory(
        amd::Memory* mem    //!< Pointer to AMD memory object
        ) const;

    amd::Monitor& lockAsyncOps() const { return *lockAsyncOps_; }

    //! Returns the lock object for the virtual gpus list
    amd::Monitor* vgpusAccess() const { return vgpusAccess_; }

    //! Returns the monitor object for PAL
    amd::Monitor& lockPAL() const { return *lockPAL_; }

    //! Returns the number of virtual GPUs allocated on this device
    uint    numOfVgpus() const { return numOfVgpus_; }
    uint    numOfVgpus_;        //!< The number of virtual GPUs (lock protected)

    typedef std::vector<VirtualGPU*> VirtualGPUs;

    //! Returns the list of all virtual GPUs running on this device
    const VirtualGPUs vgpus() const { return vgpus_; }
    VirtualGPUs     vgpus_; //!< The list of all running virtual gpus (lock protected)

    //! Scratch buffer allocation
    pal::Memory* createScratchBuffer(
        size_t size         //!< Size of buffer
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

    //! Returns the number of available compute rings
    uint numComputeEngines() const { return numComputeEngines_; }

    //! Returns the number of available DMA engines
    uint numDMAEngines() const { return numDmaEngines_; }

    //! Returns engines object
    const device::BlitManager& xferMgr() const;

    VirtualGPU* xferQueue() const { return xferQueue_; }

    //! Retrieves the internal format from the OCL format
    Pal::Format getPalFormat(
        const amd::Image::Format& format,   //! OCL image format
        Pal::ChannelMapping* channel
        ) const;

    const ScratchBuffer* scratch(uint idx) const { return scratch_[idx]; }

    //! Returns the global scratch buffer
    Memory* globalScratchBuf() const { return globalScratchBuf_; };

    //! Destroys scratch buffer memory
    void destroyScratchBuffers();

    //! Initialize heap resources if uninitialized
    bool    initializeHeapResources();

    //! Set GSL sampler to the specified state
    void    fillHwSampler(
        uint32_t    state,          //!< Sampler's OpenCL state
        void*       hwState,        //!< Sampler's HW state
        uint32_t    hwStateSize,    //!< Size of sampler's HW state
        uint32_t    mipFilter = CL_FILTER_NONE, //!< Mip filter
        float       minLod = 0.f,           //!< Min level of detail
        float       maxLod = CL_MAXFLOAT    //!< Max level of detail
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
    cl_int hwDebugManagerInit(amd::Context *context, uintptr_t messageStorage);

    //! Returns PAL device properties
    const Pal::DeviceProperties& properties() const { return properties_; }

    //! Returns PAL device interface
    Pal::IDevice* iDev() const { return device_; }

    //! Return private device context for internal allocations
    amd::Context&    context() const { return *context_; }

    //! Update free memory for OCL extension
    void updateFreeMemory(
        Pal::GpuHeap heap,      //!< PAL GPU heap for update
        Pal::gpusize size,      //!< Size of alocated/destroyed memory
        bool free               //!< TRUE if runtime frees memory
        );

    //! Interop for GL device
    bool initGLInteropPrivateExt(void* GLplatformContext, void* GLdeviceContext) const;
    bool glCanInterop(void* GLplatformContext, void* GLdeviceContext) const;
    bool resGLAssociate(void* GLContext, uint name, uint type,
        void** handle, void** mbResHandle, size_t* offset) const;
    bool resGLAcquire(void* GLplatformContext, void* mbResHandle, uint type) const;
    bool resGLRelease(void* GLplatformContext, void* mbResHandle, uint type) const;
    bool resGLFree(void* GLplatformContext, void* mbResHandle, uint type) const;

private:
    //! Disable copy constructor
    Device(const Device&);

    //! Disable assignment
    Device& operator=(const Device&);

    //! Sends the stall command to all queues
    bool stallQueues();

    //! Buffer allocation
    pal::Memory* createBuffer(
        amd::Memory&    owner,          //!< Abstraction layer memory object
        bool            directAccess    //!< Use direct host memory access
        ) const;

    //! Image allocation
    pal::Memory* createImage(
        amd::Memory&    owner,          //!< Abstraction layer memory object
        bool            directAccess    //!< Use direct host memory access
        ) const;

    //! Allocates/reallocates the scratch buffer, according to the usage
    bool allocScratch(
        uint regNum,                //!< Number of the scratch registers
        const VirtualGPU* vgpu      //!< Virtual GPU for the allocation
        );

    //! Interop for D3D devices
    bool associateD3D11Device(
        void* d3d11Device           //!< void* is of type ID3D11Device*
        );
    bool associateD3D10Device(
        void* d3d10Device           //!< void* is of type ID3D10Device*
        );
    bool associateD3D9Device(
        void* d3d9Device            //!< void* is of type IDirect3DDevice9*
        );
    //! Interop for GL device
    bool glAssociate(void* GLplatformContext, void* GLdeviceContext) const;
    bool glDissociate(void* GLplatformContext, void* GLdeviceContext) const;

    amd::Context*   context_;       //!< A dummy context for internal allocations
    amd::Monitor*   lockAsyncOps_;  //!< Lock to serialise all async ops on this device
    amd::Monitor*   lockForInitHeap_;  //!< Lock to serialise all async ops on initialization heap operation
    amd::Monitor*   lockPAL_;       //!< Lock to serialise PAL access
    amd::Monitor*   vgpusAccess_;   //!< Lock to serialise virtual gpu list access
    amd::Monitor*   scratchAlloc_;  //!< Lock to serialise scratch allocation
    amd::Monitor*   mapCacheOps_;   //!< Lock to serialise cache for the map resources
    XferBuffers*    xferRead_;      //!< Transfer buffers read
    XferBuffers*    xferWrite_;     //!< Transfer buffers write
    std::vector<amd::Memory*>*  mapCache_;  //!< Map cache info structure
    ResourceCache*  resourceCache_; //!< Resource cache
    uint            numComputeEngines_; //!< The number of available compute engines
    uint            numDmaEngines_; //!< The number of available compute engines
    bool            heapInitComplete_;  //!< Keep track of initialization status of heap resources
    VirtualGPU*     xferQueue_;     //!< Transfer queue
    std::vector<ScratchBuffer*> scratch_;   //!< Scratch buffers for kernels
    Memory*         globalScratchBuf_;  //!< Global scratch buffer
    SrdManager*     srdManager_;    //!< SRD manager object
    static AppProfile appProfile_;  //!< application profile
    mutable bool freeCPUMem_;       //!< flag to mark GPU free SVM CPU mem
    Pal::DeviceProperties   properties_;    //!< PAL device properties
    Pal::IDevice* device_;          //!< PAL device object
    std::atomic<Pal::gpusize> freeMem[Pal::GpuHeap::GpuHeapCount];    //!< Free memory counter
};

/*@}*/} // namespace pal

