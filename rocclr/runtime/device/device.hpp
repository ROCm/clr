//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef DEVICE_HPP_
#define DEVICE_HPP_

#include "top.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "utils/util.hpp"
#include "amdocl/cl_kernel.h"
#include "elf/elf.hpp"
#include "appprofile.hpp"

#include "acl.h"
#include "hwdebug.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <utility>

namespace amd {
class Command;
class CommandQueue;
class ReadMemoryCommand;
class WriteMemoryCommand;
class FillMemoryCommand;
class CopyMemoryCommand;
class MapMemoryCommand;
class UnmapMemoryCommand;
class MigrateMemObjectsCommand;
class NDRangeKernelCommand;
class NativeFnCommand;
class FlushCommand;
class FinishCommand;
class AcquireExtObjectsCommand;
class ReleaseExtObjectsCommand;
class PerfCounterCommand;
class ReleaseObjectCommand;
class StallQueueCommand;
class Marker;
class KernelSignature;
class ThreadTraceCommand;
class ThreadTraceMemObjectsCommand;
#if cl_amd_open_video
class RunVideoProgramCommand;
class SetVideoSessionCommand;
#endif // cl_amd_open_video
class SignalCommand;
class MakeBuffersResidentCommand;
class SvmFreeMemoryCommand;
class SvmCopyMemoryCommand;
class SvmFillMemoryCommand;
class SvmMapMemoryCommand;
class SvmUnmapMemoryCommand;
class HwDebugManager;
class Device;
struct KernelParameterDescriptor;
struct Coord3D;

namespace option {
class Options;
} // option

struct ProfilingCallback: public amd::HeapObject {
    virtual void callback (ulong duration) = 0;
};
}

enum OclExtensions {
    ClKhrFp64 = 0,
    ClAmdFp64,
    ClKhrSelectFpRoundingMode,
    ClKhrGlobalInt32BaseAtomics,
    ClKhrGlobalInt32ExtendedAtomics,
    ClKhrLocalInt32BaseAtomics,
    ClKhrLocalInt32ExtendedAtomics,
    ClKhrInt64BaseAtomics,
    ClKhrInt64ExtendedAtomics,
    ClKhr3DImageWrites,
    ClKhrByteAddressableStore,
    ClKhrFp16,
    ClKhrGlSharing,
    ClKhrGLDepthImages,
    ClExtDeviceFission,
    ClExtAtomicCounters32,
    ClAmdDeviceAttributeQuery,
    ClAmdVec3,
    ClAmdPrintf,
    ClAmdMediaOps,
    ClAmdMediaOps2,
    ClAmdPopcnt,
#if defined(_WIN32)
    ClKhrD3d10Sharing,
    ClKhrD3d11Sharing,
    ClKhrD3d9Sharing,
#endif
    ClKhrImage2dFromBuffer,
    ClAmdImage2dFromBufferReadOnly,
    ClAmdOpenVideo,
    ClAmdSemaphore,
    ClAMDBusAddressableMemory,
    ClAMDC11Atomics,
    ClKhrSpir,
    ClKhrSubGroups,
    ClKhrGlEvent,
    ClKhrDepthImages,
    ClKhrMipMapImage,
    ClKhrMipMapImageWrites,
    ClExtTotal
};

static const char*
OclExtensionsString[] = {
    "cl_khr_fp64 ", "cl_amd_fp64 ",
    "cl_khr_select_fprounding_mode ",
    "cl_khr_global_int32_base_atomics ",
    "cl_khr_global_int32_extended_atomics ",
    "cl_khr_local_int32_base_atomics ",
    "cl_khr_local_int32_extended_atomics ",
    "cl_khr_int64_base_atomics ",
    "cl_khr_int64_extended_atomics ",
    "cl_khr_3d_image_writes ",
    "cl_khr_byte_addressable_store ",
    "cl_khr_fp16 ",
    "cl_khr_gl_sharing ",
    "cl_khr_gl_depth_images ",
    "cl_ext_device_fission ",
    "cl_ext_atomic_counters_32 ",
    "cl_amd_device_attribute_query ",
    "cl_amd_vec3 ",
    "cl_amd_printf ",
    "cl_amd_media_ops ",
    "cl_amd_media_ops2 ",
    "cl_amd_popcnt ",
#if defined(_WIN32)
    "cl_khr_d3d10_sharing ",
    "cl_khr_d3d11_sharing ",
    "cl_khr_dx9_media_sharing ",
#endif
    "cl_khr_image2d_from_buffer ",
    "cl_amd_image2d_from_buffer_read_only ",
#if cl_amd_open_video
    "cl_amd_open_video ",
#else
    "",
#endif // cl_amd_open_video
    "",
    "cl_amd_bus_addressable_memory ",
    "cl_amd_c11_atomics ",
    "cl_khr_spir ",
    "cl_khr_subgroups ",
    "cl_khr_gl_event ",
    "cl_khr_depth_images ",
    "cl_khr_mipmap_image ",
    "cl_khr_mipmap_image_writes ",
    NULL
};

namespace device {
class ClBinary;
class BlitManager;

struct PartitionType : public amd::EmbeddedObject
{
    enum {
        EQUALLY            = (1 << 0),
        BY_COUNTS          = (1 << 1),
        BY_AFFINITY_DOMAIN = (1 << 2)
    };

    union {
        struct {
            uint equally_          : 1;
            uint byCounts_         : 1;
            uint byAffinityDomain_ : 1;
        };
        uint value_;
    };

    size_t getNumSet() const { return (size_t)amd::countBitsSet(value_); }

    cl_device_partition_property toCL() const;
    size_t toCL(cl_device_partition_property* types) const;
#ifdef cl_ext_device_fission
    cl_device_partition_property_ext toCLExt() const;
    size_t toCLExt(cl_device_partition_property_ext* types) const;
#endif
};

struct AffinityDomain : public amd::EmbeddedObject
{
    enum {
        AFFINITY_DOMAIN_NUMA               = (1 << 0),
        AFFINITY_DOMAIN_L4_CACHE           = (1 << 1),
        AFFINITY_DOMAIN_L3_CACHE           = (1 << 2),
        AFFINITY_DOMAIN_L2_CACHE           = (1 << 3),
        AFFINITY_DOMAIN_L1_CACHE           = (1 << 4),
        AFFINITY_DOMAIN_NEXT_PARTITIONABLE = (1 << 5)
    };

    union {
        struct {
            uint numa_    : 1;
            uint cacheL4_ : 1;
            uint cacheL3_ : 1;
            uint cacheL2_ : 1;
            uint cacheL1_ : 1;
            uint next_    : 1;
        };
        uint value_;
    };

    size_t getNumSet() const { return (size_t)amd::countBitsSet(value_); }

    cl_device_affinity_domain toCL() const;
#ifdef cl_ext_device_fission
    cl_device_partition_property_ext toCLExt() const;
    size_t toCLExt(cl_device_partition_property_ext* affinities) const;
#endif
};

//! Device partition properties.
struct PartitionInfo : public amd::EmbeddedObject
{
    PartitionType type_;
    union {
        struct {
            size_t numComputeUnits_;
        } equally_;

        AffinityDomain byAffinityDomain_;

        struct {
            const cl_uint* countsList_;
            size_t listSize_;
        } byCounts_;
    };
};

//! Create Sub-Devices request properties.
struct CreateSubDevicesInfo : public amd::HeapObject
{
    PartitionInfo p_;
    virtual cl_uint countsListAt(size_t i) const = 0;
    virtual ~CreateSubDevicesInfo() {}
};

template <typename PROP_T>
struct CreateSubDevicesInfoT : public CreateSubDevicesInfo
{
    virtual cl_uint countsListAt(size_t i) const {
        return (cl_uint)reinterpret_cast<const PROP_T*>(p_.byCounts_.countsList_)[i];
    }

    void initCountsList(const PROP_T* props) {
        p_.byCounts_.countsList_ = reinterpret_cast<const cl_uint*>(props);
        p_.byCounts_.listSize_ = 0;
        for (; *props != ((PROP_T)0); ++props) {
            ++p_.byCounts_.listSize_;
        }
    }
};

//! Physical device properties.
struct Info : public amd::EmbeddedObject
{
    //! The OpenCL device type.
    cl_device_type type_;

    //! A unique device vendor identifier.
    cl_uint vendorId_;

    //! The number of parallel compute cores on the compute device.
    cl_uint maxComputeUnits_;

    //! Maximum dimensions that specify the global and local work-item IDs
    //  used by the data-parallel execution model.
    cl_uint maxWorkItemDimensions_;

    //! Maximum number of work-items that can be specified in each dimension
    //  to clEnqueueNDRangeKernel.
    size_t maxWorkItemSizes_[3];

    //! Maximum number of work-items in a work-group executing a kernel
    //  using the data-parallel execution model.
    size_t maxWorkGroupSize_;

    //! Number of shader engines in physical GPU
    size_t numberOfShaderEngines;

    //! cl_uint Preferred native vector width size for built-in scalar types
    //  that can be put into vectors.
    cl_uint preferredVectorWidthChar_;
    cl_uint preferredVectorWidthShort_;
    cl_uint preferredVectorWidthInt_;
    cl_uint preferredVectorWidthLong_;
    cl_uint preferredVectorWidthFloat_;
    cl_uint preferredVectorWidthDouble_;
    cl_uint preferredVectorWidthHalf_;

    //! Returns the native ISA vector width. The vector width is defined as the
    //  number of scalar elements that can be stored in the vector.
    cl_uint nativeVectorWidthChar_;
    cl_uint nativeVectorWidthShort_;
    cl_uint nativeVectorWidthInt_;
    cl_uint nativeVectorWidthLong_;
    cl_uint nativeVectorWidthFloat_;
    cl_uint nativeVectorWidthDouble_;
    cl_uint nativeVectorWidthHalf_;

    //! Maximum configured clock frequency of the device in MHz.
    cl_uint maxClockFrequency_;

    //! Describes the address spaces supported  by the device.
    cl_uint addressBits_;

    //! Max number of simultaneous image objects that can be read by a
    //  kernel.
    cl_uint maxReadImageArgs_;

    //! Max number of simultaneous image objects that can be written to
    //  by a kernel.
    cl_uint maxWriteImageArgs_;

    //! Max number of simultaneous image objects that can be read/written to
    //  by a kernel.
    cl_uint maxReadWriteImageArgs_;

    //! Max size of memory object allocation in bytes.
    cl_ulong maxMemAllocSize_;

    //! Max width of 2D image in pixels.
    size_t image2DMaxWidth_;

    //! Max height of 2D image in pixels.
    size_t image2DMaxHeight_;

    //! Max width of 3D image in pixels.
    size_t image3DMaxWidth_;

    //! Max height of 3D image in pixels.
    size_t image3DMaxHeight_;

    //! Max depth of 3D image in pixels.
    size_t image3DMaxDepth_;

    //! Describes whether images are supported
    cl_bool imageSupport_;

    //! Max size in bytes of the arguments that can be passed to a kernel.
    size_t maxParameterSize_;

    //! Maximum number of samplers that can be used in a kernel.
    cl_uint maxSamplers_;

    //! Describes the alignment in bits of the base address of any
    //  allocated memory object.
    cl_uint memBaseAddrAlign_;

    //! The smallest alignment in bytes which can be used for any data type.
    cl_uint minDataTypeAlignSize_;

    //! Describes single precision floating point capability of the device.
    cl_device_fp_config halfFPConfig_;
    cl_device_fp_config singleFPConfig_;
    cl_device_fp_config doubleFPConfig_;

    //! Type of global memory cache supported.
    cl_device_mem_cache_type globalMemCacheType_;

    //! Size of global memory cache line in bytes.
    cl_uint globalMemCacheLineSize_;

    //! Size of global memory cache in bytes.
    cl_ulong globalMemCacheSize_;

    //! Size of global device memory in bytes.
    cl_ulong globalMemSize_;

    //! Max size in bytes of a constant buffer allocation.
    cl_ulong maxConstantBufferSize_;

    //! Max number of arguments declared
    cl_uint maxConstantArgs_;

    //! This is used to determine the type of local memory that is available
    cl_device_local_mem_type localMemType_;

    //! Size of local memory arena in bytes.
    cl_ulong localMemSize_;

    //! If enabled, implies that all the memories, caches, registers etc. in
    //  the device implement error correction.
    cl_bool errorCorrectionSupport_;

    //! CL_TRUE if the device and the host have a unified memory subsystem and
    //  is CL_FALSE otherwise.
    cl_bool hostUnifiedMemory_;

    //! Describes the resolution of device timer.
    size_t profilingTimerResolution_;

    //! Timer starting point offset to Epoch.
    cl_ulong profilingTimerOffset_;

    //! CL_TRUE if device is a little endian device.
    cl_bool littleEndian_;

    //! If enabled, implies that commands can be submitted to command-queues
    //  created on this device.
    cl_bool available_;

    //! if the implementation does not have a compiler available to compile
    //  the program source.
    cl_bool compilerAvailable_;

    //! Describes the execution capabilities of the device.
    cl_device_exec_capabilities executionCapabilities_;

    //! Describes the SVM capabilities of the device.
    cl_device_svm_capabilities svmCapabilities_;

    //! Preferred alignment for OpenCL fine-grained SVM atomic types.
    cl_uint preferredPlatformAtomicAlignment_;

    //! Preferred alignment for OpenCL global atomic types.
    cl_uint preferredGlobalAtomicAlignment_;

    //! Preferred alignment for OpenCL local atomic types.
    cl_uint preferredLocalAtomicAlignment_;

    //! Describes the command-queue properties supported of the host queue.
    cl_command_queue_properties queueProperties_;

    //! The platform associated with this device
    cl_platform_id platform_;

    //! Device name string
    char name_[0x40];

    //! Vendor name string
    char vendor_[0x20];

    //! OpenCL software driver version string in the form major.minor
    char driverVersion_[0x20];

    //! Returns the profile name supported by the device.
    const char* profile_;

    //! Returns the OpenCL version supported by the device.
    const char* version_;

    //! The highest OpenCL C version supported by the compiler for this device.
    const char* oclcVersion_;

    //! Returns a space separated list of extension names.
    const char* extensions_;

    //! Returns if device linker is available
    cl_bool linkerAvailable_;

    //! Returns the list of built-in kernels, supported by the device
    const char* builtInKernels_;

    //! Returns max number of pixels for a 1D image created from a buffer object
    size_t imageMaxBufferSize_;

    //! Returns max number of images in a 1D or 2D image array
    size_t imageMaxArraySize_;

    //! Returns the list of partition types supported by device
    PartitionType partitionProperties_;

    //! Returns the list of supported affinity domains for
    //! partitioning the device using CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN
    AffinityDomain affinityDomain_;

    //! Returns the properties argument specified in clCreateSubDevices
    //! if device is a subdevice.
    PartitionInfo partitionCreateInfo_;

    //! Returns CL_TRUE if the devices preference is for the user to be
    //! responsible for synchronization
    cl_bool preferredInteropUserSync_;

    //! Returns maximum size of the internal buffer that holds the output
    //! of printf calls from a kernel
    size_t  printfBufferSize_;

    //! Indicates whether UVD interop is supported
    cl_bool openVideo_;

    //! Indicates maximum number of supported global atomic counters
    cl_uint maxAtomicCounters_;

#if cl_amd_open_video
    // Decoder
    //! Maximum number of simultaneous video sessions/streams
    cl_uint     maxVideoSessions_;
    //! List of supported video attributes (profile/format pairs)
    cl_video_attrib_amd* videoAttribs_;
    cl_uint     numVideoAttribs_;
    //Encoder
    cl_video_attrib_encode_amd* videoEncAttribs_;
    cl_uint     numVideoEncAttribs_;
#endif //cl_amd_open_video

    //! Returns the topology for the device
    cl_device_topology_amd deviceTopology_;

    //! Semaphore information
    cl_uint maxSemaphores_;
    cl_uint maxSemaphoreSize_;

    //! Returns the SKU board name for the device
    char boardName_[128];

    //! Number of SIMD (Single Instruction Multiple Data) units per compute unit
    //! that execute in parallel. All work items from the same work group must be
    //! executed by SIMDs in the same compute unit.
    cl_uint     simdPerCU_;
    //! The maximum number of work items from the same work group that can be
    //! executed by a SIMD in parallel
    cl_uint     simdWidth_;
    //! The number of instructions that a SIMD can execute in parallel
    cl_uint     simdInstructionWidth_;
    //! The number of workitems per wavefront
    cl_uint     wavefrontWidth_;
    //! Number of global memory channels
    cl_uint     globalMemChannels_;
    //! Number of banks in each global memory channel
    cl_uint     globalMemChannelBanks_;
    //! Width in bytes of each of global memory bank
    cl_uint     globalMemChannelBankWidth_;
    //! Local memory size per CU
    cl_uint     localMemSizePerCU_;
    //! Number of banks of local memory
    cl_uint     localMemBanks_;
    //! The core engine GFXIP version
    cl_uint     gfxipVersion_;
    //! Number of available async queues
    cl_uint     numAsyncQueues_;
    //! Thread trace enable
    cl_bool     threadTraceEnable_;

    //! Image pitch alignment for image2d_from_buffer
    cl_uint imagePitchAlignment_;
    //! Image base address alignment for image2d_from_buffer
    cl_uint imageBaseAddressAlignment_;

    //! Describes whether buffers from images are supported
    cl_bool bufferFromImageSupport_;

    //! Returns the supported SPIR versions for the device
    const char* spirVersions_;

    //! OpenCL20 device info fields:

    //! The max number of pipe objects that can be passed as arguments to a kernel
    cl_uint     maxPipeArgs_;
    //! The max number of reservations that can be active for a pipe per work-item in a kernel
    cl_uint     maxPipeActiveReservations_;
    //! The max size of pipe packet in bytes
    cl_uint     maxPipePacketSize_;

    //! The command-queue properties supported of the device queue.
    cl_command_queue_properties queueOnDeviceProperties_;
    //! The preferred size of the device queue in bytes
    cl_uint     queueOnDevicePreferredSize_;
    //! The max size of the device queue in bytes
    cl_uint     queueOnDeviceMaxSize_;
    //! The maximum number of device queues
    cl_uint     maxOnDeviceQueues_;
    //! The maximum number of events in use on a device queue
    cl_uint     maxOnDeviceEvents_;

    //! The maximum size of global scope variables
    size_t      maxGlobalVariableSize_;
    size_t      globalVariablePreferredTotalSize_;
};

//! Device settings
class Settings : public amd::HeapObject
{
public:
    uint64_t    extensions_;    //!< Supported OCL extensions
    union {
        struct {
            uint    partialDispatch_: 1;    //!< Enables partial dispatch
            uint    supportRA_: 1;          //!< Support RA channel order format
            uint    largeHostMemAlloc_: 1;  //!< Allow large host mem allocations (> maxSingleAlloc)
            uint    waitCommand_: 1;        //!< Enables a wait for every submitted command
            uint    customHostAllocator_: 1;//!< True if device has custom host allocator
                                            //  that replaces generic OS allocation routines
            uint    supportDepthsRGB_: 1;   //!< Support DEPTH and sRGB channel order format
            uint    assumeAliases_: 1;      //!< Assume aliases in the compilation process
            uint    enableHwDebug_: 1;      //!< Enable HW debug support
            uint    reserved_: 24;
        };
        uint    value_;
    };

    //! Default constructor
    Settings();

    //! Check the specified extension
    bool checkExtension(uint name) const
        { return (extensions_ & (static_cast<uint64_t>(1) << name)) ? true : false; }

    //! Enable the specified extension
    void enableExtension(uint name) { extensions_ |= static_cast<uint64_t>(1) << name; }

private:
    //! Disable copy constructor
    Settings(const Settings&);

    //! Disable assignment
    Settings& operator=(const Settings&);
};

//! Device-independent cache memory, base class for the device-specific
//! memories. One Memory instance refers to one or more of these.
class Memory : public amd::HeapObject
{
public:
    //! Resource map flags
    enum CpuMapFlags
    {
        CpuReadWrite    = 0x00000000,   //!< Lock for CPU read/Write
        CpuReadOnly     = 0x00000001,   //!< Lock for CPU read only operation
        CpuWriteOnly    = 0x00000002,   //!< Lock for CPU write only operation
    };

    union SyncFlags {
        struct {
            uint skipParent_ : 1;   //!< Skip parent synchronization
            uint skipViews_  : 1;   //!< Skip views synchronization
            uint skipEntire_ : 1;   //!< Skip entire synchronization
        };
        uint value_;
        SyncFlags(): value_(0) {}
    };

    struct WriteMapInfo: public amd::EmbeddedObject
    {
        amd::Coord3D    origin_;    //!< Origin of the map location
        amd::Coord3D    region_;    //!< Mapped region
        amd::Image*     baseMip_;   //!< The base mip level for images
        bool            entire_;    //!< True if the enitre memory was mapped
        WriteMapInfo(): origin_(0, 0, 0), region_(0, 0, 0), baseMip_(NULL), entire_(false) {}
    };

    //! Constructor (from an amd::Memory object).
    Memory(amd::Memory& owner)
        : flags_(0)
        , owner_(&owner)
        , version_(0)
        , mapMemory_(NULL)
        , indirectMapCount_(0)
    {
        size_ = owner.getSize();
    }

    //! Constructor (no owner), always eager allocation.
    Memory(size_t size)
        : flags_(0)
        , owner_(NULL)
        , version_(0)
        , mapMemory_(NULL)
        , indirectMapCount_(0)
        , size_(size)
    {
    }

    enum GLResourceOP
    {
        GLDecompressResource = 0, // orders the GL driver to decompress any depth-stencil or MSAA resource to be sampled by a CL kernel.
        GLInvalidateFBO // orders the GL driver to invalidate any FBO the resource may be bound to, since the resource internal state changed.
    };

    //! Default destructor for the device memory object
    virtual ~Memory() {};

    //! Releases virtual objects associated with this memory
    void releaseVirtual();

    //! Read the size
    size_t size() const {return size_;}

    //! Gets the owner Memory instance
    amd::Memory* owner() const { return owner_; }

    //! Immediate blocking write from device cache to owners's backing store.
    //! Marks owner as "current" by resetting the last writer to NULL.
    virtual void syncHostFromCache(
        SyncFlags   syncFlags = SyncFlags()
        ) {}

    //! Allocate memory for API-level maps
    virtual void* allocMapTarget(
        const amd::Coord3D& origin, //!< The map location in memory
        const amd::Coord3D& region, //!< The map region in memory
        uint    mapFlags,           //!< Map flags
        size_t* rowPitch = NULL,    //!< Row pitch for the mapped memory
        size_t* slicePitch = NULL   //!< Slice for the mapped memory
        ) { return NULL; }

    virtual bool pinSystemMemory(
        void*   hostPtr,            //!< System memory address
        size_t  size                //!< Size of allocated system memory
        ) { return true; }

    //! Releases indirect map surface
    virtual void releaseIndirectMap() {}
    //! decompress any MSAA/depth-stencil interop surfaces.
    //! notify GL to invalidate any surfaces touched by a CL kernel
    virtual bool processGLResource(GLResourceOP operation) { return false;}

    //! Map the device memory to CPU visible
    virtual void* cpuMap(
        VirtualDevice& vDev,    //!< Virtual device for map operaiton
        uint flags = 0,         //!< flags for the map operation
        // Optimization for multilayer map/unmap
        uint startLayer = 0,    //!< Start layer for multilayer map
        uint numLayers = 0,     //!< End layer for multilayer map
        size_t* rowPitch = NULL,//!< Row pitch for the device memory
        size_t* slicePitch = NULL   //!< Slice pitch for the device memory
        )
        {
            amd::Image* image = owner()->asImage();
            if (image != NULL) {
                *rowPitch   = image->getRowPitch();
                *slicePitch = image->getSlicePitch();
            }
            // Default behavior uses preallocated host mem for CPU
            return owner()->getHostMem();
        }

    //! Unmap the device memory
    virtual void cpuUnmap(
        VirtualDevice& vDev     //!< Virtual device for unmap operaiton
        ) {}

    //! Saves map info for this object
    //! @note: It's not a thread safe operation, the app must implement
    //! synchronization for the multiple write maps if necessary
    void saveMapInfo(
        const amd::Coord3D  origin, //!< Origin of the map location
        const amd::Coord3D  region, //!< Mapped region
        uint                mapFlags,   //< Map flags
        bool                entire, //!< True if the enitre memory was mapped
        amd::Image*         baseMip = NULL  //!< The base mip level for map
        );

    const WriteMapInfo* writeMapInfo() const { return &writeMapInfo_; }

    //! Clear memory object as mapped read only
    void clearUnmapFlags() { flags_ &= ~(UnmapWrite | UnmapRead); }

    //! Returns state of map write flag
    bool isUnmapWrite() const { return (flags_ & UnmapWrite) ? true : false; }

    //! Returns state of map read flag
    bool isUnmapRead() const { return (flags_ & UnmapRead) ? true : false; }

    //! Returns state of memory direct access flag
    bool isHostMemDirectAccess() const
        { return (flags_ & HostMemoryDirectAccess) ? true : false; }

    //! Returns state of host memory registration flag
    bool isHostMemoryRegistered() const
        { return (flags_ & HostMemoryRegistered) ? true : false; }

protected:
    enum    Flags {
        HostMemoryDirectAccess  = 0x00000001,   //!< GPU has direct access to the host memory
        MapResourceAlloced      = 0x00000002,   //!< Map resource was allocated
        PinnedMemoryAlloced     = 0x00000004,   //!< An extra pinned resource was allocated
        UnmapWrite              = 0x00000008,   //!< Memory was mapped for write
        SubMemoryObject         = 0x00000010,   //!< Memory is sub-memory
        HostMemoryRegistered    = 0x00000020,   //!< Host memory was registered
        UnmapRead               = 0x00000040,   //!< Memory was mapped for read
    };
    uint        flags_;         //!< Memory object flags

    amd::Memory* owner_;    //!< The Memory instance that we cache,
                            //!< or NULL if we're device-private workspace.

    volatile size_t version_;   //!< The version we're currently shadowing

    //! NB, the map data below is for an API-level map (from clEnqueueMapBuffer),
    //! not a physical map. When a memory object does not use USE_HOST_PTR we
    //! can use a remote resource and DMA, avoiding the additional CPU memcpy.
    amd::Memory*    mapMemory_;         //!< Memory used as map target buffer
    volatile size_t indirectMapCount_;  //!< Number of maps
    WriteMapInfo    writeMapInfo_;  //!< Saved write map info for partial unmap

    //! Increment map count
    void incIndMapCount() { ++indirectMapCount_; }

    //! Decrement map count
    virtual void decIndMapCount() {}

private:
    //! Disable default copy constructor
    Memory& operator=(const Memory&);

    //! Disable operator=
    Memory(const Memory&);

    //! Our size
    size_t size_;
};

class Sampler : public amd::HeapObject
{
public:
    //! Constructor
    Sampler() {}

    //! Default destructor for the device memory object
    virtual ~Sampler() {};

    //! Returns device specific HW state for the sampler
    uint64_t    hwSrd() const { return hwSrd_; }

protected:
    uint64_t    hwSrd_;   //!< Device specific HW state for the sampler

private:
    //! Disable default copy constructor
    Sampler& operator=(const Sampler&);

    //! Disable operator=
    Sampler(const Sampler&);
};

//! \class DeviceKernel, which will contain the common fields for any device
class Kernel : public amd::HeapObject
{
public:
    typedef std::vector<amd::KernelParameterDescriptor> parameters_t;

    //! \struct The device kernel workgroup info structure
    struct WorkGroupInfo : public amd::EmbeddedObject
    {
        size_t   size_;                  //!< kernel workgroup size
        size_t   compileSize_[3];        //!< kernel compiled workgroup size
        cl_ulong localMemSize_;          //!< amount of used local memory
        size_t   preferredSizeMultiple_; //!< preferred multiple for launch
        cl_ulong privateMemSize_;        //!< amount of used private memory
        size_t   scratchRegs_;           //!< amount of used scratch registers
        size_t   wavefrontPerSIMD_;      //!< number of wavefronts per SIMD
        size_t   wavefrontSize_;         //!< number of threads per wavefront
        size_t   availableGPRs_;         //!< GPRs available to the program
        size_t   usedGPRs_;              //!< GPRs used by the program
        size_t   availableSGPRs_;        //!< SGPRs available to the program
        size_t   usedSGPRs_;             //!< SGPRs used by the program
        size_t   availableVGPRs_;        //!< VGPRs available to the program
        size_t   usedVGPRs_;             //!< VGPRs used by the program
        size_t   availableLDSSize_;      //!< available LDS size
        size_t   usedLDSSize_;           //!< used LDS size
        size_t   availableStackSize_;    //!< available stack size
        size_t   usedStackSize_;         //!< used stack size
        size_t   compileSizeHint_[3];    //!< kernel compiled workgroup size hint
        std::string compileVecTypeHint_; //!< kernel compiled vector type hint
        bool     uniformWorkGroupSize_;  //!< uniform work group size option
        bool     limitWave_;             //!< adaptively limit waves per SH
    };

    //! Default constructor
    Kernel(const std::string& name): name_(name), signature_(NULL), hsa_(false)
        { memset(&workGroupInfo_, '\0', sizeof(workGroupInfo_)); }

    //! Default destructor
    virtual ~Kernel();

    //! Validates memory argument
    virtual bool validateMemory(
        uint    idx,            //!< Argument's index
        amd::Memory* amdMem     //!< memory object for validation
        ) const { return true; }

    //! Returns the kernel info structure
    const WorkGroupInfo* workGroupInfo() const { return &workGroupInfo_; }

    //! Returns the kernel signature
    const amd::KernelSignature& signature() const { return *signature_; }

    //! Returns the kernel name
    const std::string& name() const { return name_; }

    //! Initializes the kernel parameters for the abstraction layer
    bool createSignature(const parameters_t& params);

    //! Returns TRUE if it's a HSA kernel
    bool hsa() const { return hsa_; }

    void setUniformWorkGroupSize(bool u)
    {
        workGroupInfo_.uniformWorkGroupSize_ = u;
    }

    bool getUniformWorkGroupSize() const {
      return workGroupInfo_.uniformWorkGroupSize_;
    }

    void setReqdWorkGroupSize(size_t x, size_t y, size_t z)
    {
        workGroupInfo_.compileSize_[0] = x;
        workGroupInfo_.compileSize_[1] = y;
        workGroupInfo_.compileSize_[2] = z;
    }

    size_t getReqdWorkGroupSize(int dim) {
      return workGroupInfo_.compileSize_[dim];
    }

    void setWorkGroupSizeHint(size_t x, size_t y, size_t z)
    {
        workGroupInfo_.compileSizeHint_[0] = x;
        workGroupInfo_.compileSizeHint_[1] = y;
        workGroupInfo_.compileSizeHint_[2] = z;
    }

    size_t getWorkGroupSizeHint(int dim) const {
      return workGroupInfo_.compileSizeHint_[dim];
    }

    //! Get profiling callback object
    virtual amd::ProfilingCallback* getProfilingCallback(
        const device::VirtualDevice *vdv) {
        return NULL;
    }

    void setVecTypeHint(const std::string& hint)
    {
        workGroupInfo_.compileVecTypeHint_ = hint;
    }

    void setLocalMemSize(size_t size)
    {
        workGroupInfo_.localMemSize_ = size;
    }

    void setPreferredSizeMultiple(size_t size)
    {
        workGroupInfo_.preferredSizeMultiple_ = size;
    }

    //! Return the build log
    const std::string& buildLog() const { return buildLog_; }

protected:
    std::string     name_;              //!< kernel name
    WorkGroupInfo   workGroupInfo_;     //!< device kernel info structure
    amd::KernelSignature*   signature_; //!< kernel signature
    bool            hsa_;               //!< True if HSA kernel on GPU
    std::string buildLog_;              //!< build log
private:
    //! Disable default copy constructor
    Kernel(const Kernel&);

    //! Disable operator=
    Kernel& operator=(const Kernel&);
};

//! A program object for a specific device.
class Program : public amd::HeapObject
{
public:
    typedef std::pair<const void*, size_t> binary_t;
    typedef std::map<std::string, Kernel*> kernels_t;
    // type of the program
    typedef enum {
        TYPE_NONE = 0,      // uncompiled
        TYPE_COMPILED,      // compiled
        TYPE_LIBRARY,       // linked library
        TYPE_EXECUTABLE,    // linked executable
        TYPE_INTERMEDIATE   // intermediate
    } type_t;

private:
    //! The device target for this binary.
    amd::SharedReference<amd::Device> device_;

    kernels_t   kernels_;       //!< The kernel entry points this binary.

    type_t      type_;          //!< type of this program

protected:
    ClBinary*   clBinary_;      //!< The CL program binary file
    std::string llvmBinary_;    //!< LLVM IR binary code
    bool        llvmBinaryIsSpir_; //!< LLVM IR binary code is in SPIR format
    std::string compileOptions_;//!< compile/build options.
    std::string linkOptions_;   //!< link options.
    //!< the option arg passed in to clCompileProgram(), clLinkProgram(),
    //! or clBuildProgram(), whichever is called last
    std::string lastBuildOptionsArg_;
    std::string buildLog_;      //!< build log.
    cl_int      buildStatus_;   //!< build status.
    cl_int      buildError_;    //!< build error
    //! The info target for this binary.
    aclTargetInfo info_;
    size_t      globalVariableTotalSize_;

public:
    //! Construct a section.
    Program(amd::Device& device);

    //! Destroy this binary image.
    virtual ~Program();

    //! Destroy all the kernels
    void clear();

    //! Return the compiler options passed to build this program
    amd::option::Options* getCompilerOptions() const { return programOptions; }

    //! Compile the device program.
    cl_int compile(const std::string& sourceCode,
                   const std::vector<const std::string*>& headers,
                   const char** headerIncludeNames,
                   const char* origOptions,
                   amd::option::Options* options);

    //! Builds the device program.
    cl_int link(const std::vector<Program*>& inputPrograms,
                const char* origOptions,
                amd::option::Options* options);

    //! Builds the device program.
    cl_int build(const std::string& sourceCode,
                 const char* origOptions,
                 amd::option::Options* options);

    //! Returns the device object, associated with this program.
    const amd::Device& device() { return device_(); }

    //! Return the compiler options used to build the program.
    const std::string& compileOptions() const { return compileOptions_; }

    //! Return the option arg passed in to clCompileProgram(), clLinkProgram(),
    //! or clBuildProgram(), whichever is called last
    const std::string lastBuildOptionsArg() const {
      return lastBuildOptionsArg_;
    }

    //! Return the build log.
    const std::string& buildLog() const { return buildLog_; }

    //! Return the build status.
    cl_build_status buildStatus() const { return buildStatus_; }

    //! Return the build error.
    cl_int buildError() const { return buildError_; }

    //! Return the symbols vector.
    const kernels_t& kernels() const { return kernels_; }
    kernels_t& kernels() { return kernels_; }

    //! Return the binary image.
    inline const binary_t binary() const;
    inline binary_t binary();

    //! Returns the CL program binary file
    ClBinary* clBinary() { return clBinary_; }
    const ClBinary* clBinary() const { return clBinary_; }

    bool setBinary(char* binaryIn, size_t size);

    type_t type() const { return type_; }

    void setGlobalVariableTotalSize(size_t size) {
        globalVariableTotalSize_ = size;
    }

    size_t globalVariableTotalSize() const {
        return globalVariableTotalSize_;
    }

protected:
    //! pre-compile setup
    virtual bool initBuild(amd::option::Options* options);

    //! post-compile cleanup
    virtual bool finiBuild(bool isBuildGood);

    //! Compile the device program.
    virtual bool compileImpl(const std::string& sourceCode,
                             const std::vector<const std::string*>& headers,
                             const char** headerIncludeNames,
                             amd::option::Options* options) = 0;

    //! Link the device program.
    virtual bool linkImpl(amd::option::Options* options) = 0;

    //! Link the device programs.
    virtual bool linkImpl(const std::vector<Program*>& inputPrograms,
                          amd::option::Options* options,
                          bool createLibrary) = 0;

    virtual bool createBinary(amd::option::Options* options) = 0;

    virtual bool createBIFBinary(aclBinary* bin);

    //! Initialize Binary (used only for clCreateProgramWithBinary()).
    bool initClBinary(char* binaryIn, size_t size);

    //! Initialize Binary
    virtual bool initClBinary() = 0;

    //! Release the Binary
    virtual void releaseClBinary() = 0;

    //! return target info
    virtual const aclTargetInfo & info(const char * str = "") = 0;

    virtual bool isElf(const char* bin) const = 0;

    //! At linking time, get the set of compile options to be used from
    //! the set of input program, warn if they have inconsisten compile
    //! options.
    bool getCompileOptionsAtLinking(const std::vector<Program*>& inputPrograms,
                                    const amd::option::Options* linkOptions);

    void setType(type_t newType) { type_ = newType; }

private:
    //! Disable default copy constructor
    Program(const Program&);

    //! Disable operator=
    Program& operator=(const Program&);

public:
    amd::option::Options* programOptions;
};

class ClBinary : public amd::HeapObject
{
public:

    enum BinaryImageFormat {
        BIF_VERSION2 = 0,   //!< Binary Image Format version 2.0 (ELF)
        BIF_VERSION3        //!< Binary Image Format version 3.0 (ELF)
    };

    //! Constructor
    ClBinary(const amd::Device& dev, BinaryImageFormat bifVer = BIF_VERSION2);

    //! Destructor
    virtual ~ClBinary();

    void init(amd::option::Options* optionsObj, bool amdilRequired = false);

    /** called only in loading image routines,
        never called in storing routines */
    bool setBinary(char* theBinary, size_t theBinarySize, bool allocated=false);

    //! setin elfIn_
    bool setElfIn(unsigned char eclass);
    void resetElfIn();

    //! set out elf
    bool setElfOut(unsigned char eclass, const char* outFile);
    void resetElfOut();

    //! Set elf header information
    virtual bool setElfTarget() = 0;

    // class used in for loading images in new format
    amd::OclElf*    elfIn() { return elfIn_; }

    // classes used storing and loading images in new format
    amd::OclElf*    elfOut() { return elfOut_; }
    void elfOut(amd::OclElf* v) { elfOut_ = v; }

    //! Create and save ELF binary image
    bool createElfBinary(bool doencrypt, Program::type_t type);

    //save BIF binary image
    void saveBIFBinary(char* binaryIn, size_t size);

    bool decryptElf(char* binaryIn, size_t size,
                    char** decryptBin, size_t* decryptSize, int* encryptCode);

    //! Returns the binary pair for the abstraction layer
    Program::binary_t data() const;

    //! Loads llvmir binary from OCL binary file
    bool loadLlvmBinary(
        std::string&  llvmBinary,    //!< LLVMIR binary code
        bool& llvmBinaryIsSpir       //!< LLVMIR binary is in SPIR format
        ) const;

    //! Loads compile options from OCL binary file
    bool loadCompileOptions(
        std::string& compileOptions //!< return the compile options loaded
        ) const;

    //! Loads link options from OCL binary file
    bool loadLinkOptions(
        std::string& linkOptions    //!< return the link options loaded
        ) const;

    //! Store compile options into OCL binary file
    void storeCompileOptions(
        const std::string& compileOptions //!< the compile options to be stored
        );

    //! Store link options into OCL binary file
    void storeLinkOptions(
        const std::string& linkOptions    //!< the link options to be stored
        );

    //! Check if the binary is recompilable
    bool isRecompilable(std::string& llvmBinary,
                        amd::OclElf::oclElfPlatform thePlatform);

    void saveOrigBinary(char* origBinary, size_t origSize) {
        origBinary_ = origBinary;
        origSize_   = origSize;
    }

    void restoreOrigBinary() {
        if (origBinary_ != NULL) {
            (void)setBinary(origBinary_, origSize_, false);
        }
    }

    //! Set Binary flags
    void setFlags(int encryptCode);

    bool saveSOURCE () {
        return ((flags_ & BinarySourceMask) == BinarySaveSource);
    }
    bool saveLLVMIR () {
        return ((flags_ & BinaryLlvmirMask) == BinarySaveLlvmir);
    }
    bool saveAMDIL () {
        return ((flags_ & BinaryAmdilMask) == BinarySaveAmdil);
    }
    bool saveISA () {
        return ((flags_ & BinaryIsaMask) == BinarySaveIsa);
    }

    bool saveAS () {
        return ((flags_ & BinaryASMask) == BinarySaveAS);
    }

    // Return the encrypt code for this input binary ( "> 0" means encrypted)
    int getEncryptCode() { return encryptCode_; }

    // Returns TRUE of binary file is SPIR
    bool isSPIR() const;
protected:
    enum Flags {
        BinaryAllocated    = 0x1,    //!< Binary was created

        // Source control
        BinaryNoSaveSource = 0x0,    // 0: default
        BinaryRemoveSource = 0x2,    // for encrypted binary
        BinarySaveSource   = 0x4,
        BinarySourceMask   = 0x6,

        // LLVMIR control
        BinarySaveLlvmir   = 0x0,    // 0: default
        BinaryRemoveLlvmir = 0x8,    // for encrypted binary
        BinaryNoSaveLlvmir = 0x10,
        BinaryLlvmirMask   = 0x18,

        // AMDIL control
        BinarySaveAmdil    = 0x0,    // 0: default
        BinaryRemoveAmdil  = 0x20,   // for encrypted binary
        BinaryNoSaveAmdil  = 0x40,
        BinaryAmdilMask    = 0x60,

        // ISA control
        BinarySaveIsa      = 0x0,    // 0: default
        BinaryRemoveIsa    = 0x80,   // for encrypted binary
        BinaryNoSaveIsa    = 0x100,
        BinaryIsaMask      = 0x180,

        // AS control
        BinaryNoSaveAS     = 0x0,    // 0: default
        BinaryRemoveAS     = 0x200,  // for encrypted binary
        BinarySaveAS       = 0x400,
        BinaryASMask       = 0x600
    };

    //! Returns TRUE if binary file was allocated
    bool isBinaryAllocated() const
        { return (flags_ & BinaryAllocated) ? true : false; }

    //! Returns BIF symbol name by symbolID,
    //! returns empty string if not found or if BIF version is unsupported
    std::string getBIFSymbol(unsigned int symbolID) const;

protected:
    const amd::Device&  dev_;   //!< Device object

private:
    //! Disable default copy constructor
    ClBinary(const ClBinary&);

    //! Disable default operator=
    ClBinary& operator=(const ClBinary&);

    //! Releases the binary data store
    void release();

    char*       binary_;    //!< binary data
    size_t      size_;      //!< binary size
    uint        flags_;     //!< CL binary object flags

    char*       origBinary_;    //!< original binary data
    size_t      origSize_;      //!< original binary size

    int         encryptCode_;   //!< Encryption Code for input binary (0 for not encrypted)

protected:
    amd::OclElf   *elfIn_;      //!< ELF object for input ELF binary
    amd::OclElf   *elfOut_;     //!< ELF object for output ELF binary
    BinaryImageFormat format_;  //!< which binary image format to use
};

inline const Program::binary_t Program::binary() const {
    if (clBinary() == NULL) {
        return std::make_pair((const void*)0, 0);
    }
    return clBinary()->data();
}

inline Program::binary_t Program::binary() {
    if (clBinary() == NULL) {
        return std::make_pair((const void*)0, 0);
    }
    return clBinary()->data();
}

/*! \class PerfCounter
 *
 *  \brief The device interface class for the performance counters
 */
class PerfCounter :  public amd::HeapObject
{
public:
    //! Constructor for the device performance
    PerfCounter() {}

    //! Get the performance counter info
    virtual uint64_t getInfo(uint64_t infoType) const = 0;

    //! Destructor for PerfCounter class
    virtual ~PerfCounter() {}

private:
    //! Disable default copy constructor
    PerfCounter(const PerfCounter&);

    //! Disable default operator=
    PerfCounter& operator=(const PerfCounter&);
};
/*! \class ThreadTrace
 *
 *  \brief The device interface class for the performance counters
 */
class ThreadTrace :  public amd::HeapObject
{
public:
    //! Constructor for the device performance
    ThreadTrace() {}
    //! Update ThreadTrace status to true/false if new buffer was binded/unbinded respectively
    virtual void setNewBufferBinded(bool) = 0;
    //! Get the performance counter info
    virtual bool info(uint infoType, uint* info,uint infoSize) const = 0;
    //! Destructor for PerfCounter class
    virtual ~ThreadTrace() {}

private:
    //! Disable default copy constructor
    ThreadTrace(const ThreadTrace&);

    //! Disable default operator=
    ThreadTrace& operator=(const ThreadTrace&);
};

//! A device execution environment.
class VirtualDevice : public amd::HeapObject
{
public:
    //! Construct a new virtual device for the given physical device.
    VirtualDevice(amd::Device& device)
        : device_(device), blitMgr_(NULL)
    { }

    //! Destroy this virtual device.
    virtual ~VirtualDevice()
    { }

    //! Prepare this virtual device for destruction.
    virtual bool terminate() = 0;

    //! Return the physical device for this virtual device.
    const amd::Device& device() const { return device_(); }

    virtual void submitReadMemory(amd::ReadMemoryCommand& cmd) = 0;
    virtual void submitWriteMemory(amd::WriteMemoryCommand& cmd) = 0;
    virtual void submitCopyMemory(amd::CopyMemoryCommand& cmd) = 0;
    virtual void submitMapMemory(amd::MapMemoryCommand& cmd) = 0;
    virtual void submitUnmapMemory(amd::UnmapMemoryCommand& cmd) = 0;
    virtual void submitKernel(amd::NDRangeKernelCommand& command) = 0;
    virtual void submitNativeFn(amd::NativeFnCommand& cmd) = 0;
    virtual void submitMarker(amd::Marker& cmd) = 0;
    virtual void submitFillMemory(amd::FillMemoryCommand& cmd) = 0;
    virtual void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd) = 0;
    virtual void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd) = 0;
    virtual void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd) = 0;
    virtual void submitPerfCounter(amd::PerfCounterCommand& cmd) = 0;
    virtual void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) = 0;
    virtual void submitThreadTrace(amd::ThreadTraceCommand& cmd) = 0;
    virtual void flush(amd::Command* list = NULL, bool wait = false) = 0;
#if cl_amd_open_video
    virtual void submitRunVideoProgram(amd::RunVideoProgramCommand& cmd) = 0;
    virtual void submitSetVideoSession(amd::SetVideoSessionCommand& cmd) = 0;
#endif // cl_amd_open_video
    virtual void submitSignal(amd::SignalCommand & cmd) = 0;
    virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand & cmd) = 0;
    virtual void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd) = 0;
    virtual void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd) = 0;
    virtual void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd) = 0;
    virtual void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd) = 0;
    virtual void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd) = 0;

    //! Get the blit manager object
    device::BlitManager& blitMgr() const { return *blitMgr_; }

private:
    //! Disable default copy constructor
    VirtualDevice& operator=(const VirtualDevice&);

    //! Disable operator=
    VirtualDevice(const VirtualDevice&);

    //! The physical device that this virtual device utilizes
    amd::SharedReference<amd::Device> device_;

protected:
    device::BlitManager*    blitMgr_;   //!< Blit manager
};

} // namespace device

namespace amd {


//! SvmManager class
class SvmManager : public AllStatic
{
public:

    static size_t size();                                    //!< obtain the size of the container
    static void AddSvmBuffer(const void* k, amd::Memory* v); //!< add the svm pointer and buffer in the container
    static void RemoveSvmBuffer(const void* k);              //!< Remove an entry of svm info from the container
    static amd::Memory* FindSvmBuffer(const void* k);        //!< find the svm buffer based on the input pointer
private:
    static std::map<uintptr_t, amd::Memory*> svmBufferMap_;  //!< the svm space information container
    static amd::Monitor AllocatedLock_;                      //!< amd monitor locker
};

 /*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Device Device Abstraction
 *  @{
 */
class Device : public RuntimeObject
{
public:
    typedef std::list<CommandQueue*> CommandQueues;

    struct BlitProgram : public amd::HeapObject
    {
        Program*    program_;   //!< GPU program obejct
        Context*    context_;   //!< A dummy context

        BlitProgram(Context* context): program_(NULL), context_(context) {}
        ~BlitProgram();

        //! Creates blit program for this device
        bool create(Device* device,         //!< Device object
            const char* extraKernel = NULL, //!< Extra kernels from the device layer
            const char* extraOptions = NULL //!< Extra compilation options
            );
    };

    virtual aclCompiler* compiler() const = 0;

    Device(Device* parent = NULL);
    virtual ~Device();

    //! Increment the reference count
    uint retain() {
        // Only increment the reference count of sub-devices
        return !isRootDevice() ? RuntimeObject::retain() : 0u;
    }

    //! Decrement the reference count
    uint release() {
        // Only decrement the reference count of sub-devices
        return !isRootDevice() ? RuntimeObject::release() : 0u;
    }

    //! Register a device as available
    void registerDevice();

    //! Initialize the device layer (enumerate known devices)
    static bool init();

    //! Shutdown the device layer
    static void tearDown();

    static std::vector<Device*>
    getDevices(
        cl_device_type  type,           //!< Device type
        bool            offlineDevices  //!< Enable offline devices
        );

    static size_t
    numDevices(
        cl_device_type  type,           //!< Device type
        bool            offlineDevices  //!< Enable offline devices
        );

    static bool
    getDeviceIDs(
        cl_device_type  deviceType,     //!< Device type
        cl_uint         numEntries,     //!< Number of entries in the array
        cl_device_id*   devices,        //!< Array of the device ID(s)
        cl_uint*        numDevices,     //!< Number of available devices
        bool            offlineDevices  //!< Report offline devices
        );

    const device::Info& info() const { return info_; }

    //! Return svm support capability.
    bool svmSupport() const {
        return (info().svmCapabilities_ &
            (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER |
            CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
            CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) != 0 ? true : false;
    }

    //! Return this device's type.
    cl_device_type type() const {
        return info().type_ & ~(CL_DEVICE_TYPE_DEFAULT | CL_HSA_ENABLED_AMD
                                  | CL_HSA_DISABLED_AMD);
    }

    //! Create sub-devices according to the given partition scheme.
    virtual cl_int createSubDevices(
        device::CreateSubDevicesInfo& create_info,
        cl_uint num_entries,
        cl_device_id* devices,
        cl_uint* num_devices) = 0;

    //! Create a new virtual device environment.
    virtual device::VirtualDevice* createVirtualDevice(
        bool    profiling,
        bool    interopQueue
#if cl_amd_open_video
        , void*   calVideoProperties = NULL
#endif // cl_amd_open_video
        , uint  deviceQueueSize = 0
        ) = 0;

    //! Compile the given source code.
    virtual device::Program* createProgram(int oclVer = 120) = 0;

    //! Allocate a chunk of device memory as a cache for a CL memory object
    virtual device::Memory* createMemory(Memory& owner) const = 0;

    //! Allocate a device sampler object
    virtual bool createSampler(const Sampler&, device::Sampler**) const = 0;

    //! Allocates a view object from the device memory
    virtual device::Memory* createView(
        amd::Memory&      owner,        //!< Owner memory object
        const device::Memory&   parent  //!< Parent device memory object for the view
        ) const = 0;

    //! Reallocates device memory object
    virtual bool reallocMemory(Memory& owner) const = 0;

    //! Return true if initialized external API interop, otherwise false
    virtual bool bindExternalDevice(
        intptr_t type,  //!< Enum val. for ext.API type: GL, D3D10, etc.
        void* pDevice,  //!< D3D device do D3D, HDC/Display handle of X Window for GL
        void* pContext,  //!< HGLRC/GLXContext handle
        bool validateOnly //! Only validate if the device can inter-operate with pDevice/pContext, do not bind.
        ) = 0;

    virtual bool unbindExternalDevice(
        intptr_t type,  //!< Enum val. for ext.API type: GL, D3D10, etc.
        void* pDevice,  //!< D3D device do D3D, HDC/Display handle of X Window for GL
        void* pContext,  //!< HGLRC/GLXContext handle
        bool validateOnly //! Only validate if the device can inter-operate with pDevice/pContext, do not bind.
        ) = 0;

    //! resolves GL depth/msaa buffer
    virtual bool  resolveGLMemory(device::Memory*) const { return true; }

    //! Gets a pointer to a region of host-visible memory for use as the target
    //! of an indirect map for a given memory object
    virtual void* allocMapTarget(
        amd::Memory&    mem,        //!< Abstraction layer memory object
        const amd::Coord3D& origin, //!< The map location in memory
        const amd::Coord3D& region, //!< The map region in memory
        uint    mapFlags,           //!< Map flags
        size_t* rowPitch = NULL,    //!< Row pitch for the mapped memory
        size_t* slicePitch = NULL   //!< Slice for the mapped memory
        );

    //! Gets free memory on a GPU device
    virtual bool globalFreeMemory(
        size_t* freeMemory      //!< Free memory information on a GPU device
        ) const = 0;

    /**
     * @return True if the device has its own custom host allocator to be used
     * instead of the generic OS allocation routines
     */
    bool customHostAllocator() const {
        return settings().customHostAllocator_ == 1;
    }

    /**
     * @copydoc amd::Context::hostAlloc
     */
    virtual void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const
    {
        ShouldNotCallThis();
        return NULL;
    }

    /**
     * @copydoc amd::Context::hostFree
     */
    virtual void hostFree(void* ptr, size_t size = 0) const
    {
        ShouldNotCallThis();
    }

    /**
     * @copydoc amd::Context::svmAlloc
     */
    virtual void* svmAlloc(Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags, void* svmPtr) const = 0;
    /**
     * @copydoc amd::Context::svmFree
     */
    virtual void svmFree(void* ptr) const = 0;

    //! Validate kernel
    virtual bool validateKernel(
        const amd::Kernel& kernel,
        const device::VirtualDevice* vdev) { return true; };

    //! Returns true if the given binary image is valid for this device.
    bool verifyBinaryImage( const void* image, size_t size) const;

    //! Returns TRUE if the device is available for computations
    bool isOnline() const { return online_; }
    //! Returns TRUE if the device is a root device (as opposed to sub-device)
    bool isRootDevice() const { return parent_ == NULL; }
    //! Returns TRUE if 'this' is an ancestor of the given sub-device.
    bool isAncestor(const Device* sub) const;

    //! Return the parent device.
    Device* parent() const { return parent_; }

    //! Return the root device for this instance;
    Device& rootDevice() {
        Device* root = this;
        while (!root->isRootDevice()) {
            root = root->parent_;
        }
        return *root;
    }

    const Device& rootDevice() const {
        const Device* root = this;
        while (!root->isRootDevice()) {
            root = root->parent_;
        }
        return *root;
    }

    //! Returns device settings
    const device::Settings& settings() const { return *settings_; }

    //! Returns blit program info structure
    BlitProgram* blitProgram() const { return blitProgram_; }

    //! RTTI internal implementation
    virtual ObjectType objectType() const {return ObjectTypeDevice;}

    //! Returns app profile
    static const AppProfile* appProfile() {return &appProfile_;}

    //! Register a hardware debugger manager
    HwDebugManager*  hwDebugMgr() const { return hwDebugMgr_; }

    //! Initialize the Hardware Debug Manager
    virtual cl_int hwDebugManagerInit(amd::Context *context, uintptr_t messageStorage) { return CL_SUCCESS; }

    //! Remove the Hardware Debug Manager
    virtual void hwDebugManagerRemove() {}

protected:
    //! Enable the specified extension
    char* getExtensionString();

    device::Info        info_;      //!< Device info structure
    device::Settings*   settings_;  //!< Device settings
    bool                online_;    //!< The device in online
    BlitProgram*    blitProgram_;   //!< Blit program info
    static AppProfile appProfile_;  //!< application profile
    HwDebugManager* hwDebugMgr_;    //!< Hardware Debug manager

private:
    bool IsHsaCapableDevice();
    bool IsTypeMatching(cl_device_type type, bool offlineDevices);

#if defined(WITH_HSA_DEVICE)
    static AppProfile* oclhsaAppProfile_;
#endif
    static bool isHsaDeviceAvailable_;
    static bool isGpuDeviceAvailable_;

    typedef std::vector<Device*>::iterator device_iterator;
    static std::vector<Device*>* devices_; //!< All known devices

    Device*         parent_;   //!< This device's parent
};

struct KernelParameterDescriptor
{
    const char* name_;      //!< The parameter's name in the source
    clk_value_type_t type_; //!< The parameter's type
    size_t offset_;         //!< Its offset in the parameter's stack
    size_t size_;           //!< Its size in bytes
    //! Argument's address qualifier
    cl_kernel_arg_address_qualifier addressQualifier_;
    //! Argument's access qualifier
    cl_kernel_arg_access_qualifier  accessQualifier_;
    //! Argument's type qualifier
    cl_kernel_arg_type_qualifier    typeQualifier_;
    const char* typeName_;  //!< Argument's type name
};

/*! @}
 *  @}
 */

} // namespace amd

#endif /*DEVICE_HPP_*/
