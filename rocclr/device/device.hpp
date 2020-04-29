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
#include "devprogram.hpp"
#include "devkernel.hpp"
#include "amdocl/cl_profile_amd.h"
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
class CopyMemoryP2PCommand;
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
class ThreadTraceCommand;
class ThreadTraceMemObjectsCommand;
class SignalCommand;
class MakeBuffersResidentCommand;
class SvmFreeMemoryCommand;
class SvmCopyMemoryCommand;
class SvmFillMemoryCommand;
class SvmMapMemoryCommand;
class SvmUnmapMemoryCommand;
class TransferBufferFileCommand;
class HwDebugManager;
class Device;
struct KernelParameterDescriptor;
struct Coord3D;

namespace option {
class Options;
}  // namespace option

}  // namespace amd

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
  ClAmdSemaphore,
  ClAMDBusAddressableMemory,
  ClAMDC11Atomics,
  ClKhrSpir,
  ClKhrSubGroups,
  ClKhrGlEvent,
  ClKhrDepthImages,
  ClKhrMipMapImage,
  ClKhrMipMapImageWrites,
  ClKhrIlProgram,
  ClAMDLiquidFlash,
  ClAmdCopyBufferP2P,
  ClAmdAssemblyProgram,
#if defined(_WIN32)
  ClAmdPlanarYuv,
#endif
  ClExtTotal
};

static const char* OclExtensionsString[] = {"cl_khr_fp64 ",
                                            "cl_amd_fp64 ",
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
                                            "",
                                            "cl_amd_bus_addressable_memory ",
                                            "cl_amd_c11_atomics ",
                                            "cl_khr_spir ",
                                            "cl_khr_subgroups ",
                                            "cl_khr_gl_event ",
                                            "cl_khr_depth_images ",
                                            "cl_khr_mipmap_image ",
                                            "cl_khr_mipmap_image_writes ",
                                            "",
                                            "cl_amd_liquid_flash ",
                                            "cl_amd_copy_buffer_p2p ",
                                            "cl_amd_assembly_program ",
#if defined(_WIN32)
                                            "cl_amd_planar_yuv",
#endif
                                            NULL};

static constexpr int AmdVendor = 0x1002;

namespace device {
class ClBinary;
class BlitManager;
class Program;
class Kernel;

//! Physical device properties.
struct Info : public amd::EmbeddedObject {
  //! The OpenCL device type.
  cl_device_type type_;

  //! A unique device vendor identifier.
  uint32_t vendorId_;

  //! The number of parallel compute cores on the compute device.
  uint32_t maxComputeUnits_;

  //! Maximum dimensions that specify the global and local work-item IDs
  //  used by the data-parallel execution model.
  uint32_t maxWorkItemDimensions_;

  //! Maximum number of work-items that can be specified in each dimension
  //  to clEnqueueNDRangeKernel.
  size_t maxWorkItemSizes_[3];

  //! Maximum number of work-items in a work-group executing a kernel
  //  using the data-parallel execution model.
  size_t maxWorkGroupSize_;

  //! Preferred number of work-items in a work-group executing a kernel
  //  using the data-parallel execution model.
  size_t preferredWorkGroupSize_;

  //! Number of shader engines in physical GPU
  size_t numberOfShaderEngines;

  //! uint32_t Preferred native vector width size for built-in scalar types
  //  that can be put into vectors.
  uint32_t preferredVectorWidthChar_;
  uint32_t preferredVectorWidthShort_;
  uint32_t preferredVectorWidthInt_;
  uint32_t preferredVectorWidthLong_;
  uint32_t preferredVectorWidthFloat_;
  uint32_t preferredVectorWidthDouble_;
  uint32_t preferredVectorWidthHalf_;

  //! Returns the native ISA vector width. The vector width is defined as the
  //  number of scalar elements that can be stored in the vector.
  uint32_t nativeVectorWidthChar_;
  uint32_t nativeVectorWidthShort_;
  uint32_t nativeVectorWidthInt_;
  uint32_t nativeVectorWidthLong_;
  uint32_t nativeVectorWidthFloat_;
  uint32_t nativeVectorWidthDouble_;
  uint32_t nativeVectorWidthHalf_;

  //! Maximum configured engine clock frequency of the device in MHz.
  uint32_t maxEngineClockFrequency_;

  //! Maximum configured memory clock frequency of the device in MHz.
  uint32_t maxMemoryClockFrequency_;

  //! Memory bus width in bits.
  uint32_t vramBusBitWidth_;

  //! Size of L2 Cache in bytes.
  uint32_t l2CacheSize_;

  //! Timestamp frequency in Hz.
  uint32_t timeStampFrequency_;

  //! Describes the address spaces supported  by the device.
  uint32_t addressBits_;

  //! Max number of simultaneous image objects that can be read by a
  //  kernel.
  uint32_t maxReadImageArgs_;

  //! Max number of simultaneous image objects that can be written to
  //  by a kernel.
  uint32_t maxWriteImageArgs_;

  //! Max number of simultaneous image objects that can be read/written to
  //  by a kernel.
  uint32_t maxReadWriteImageArgs_;

  //! Max size of memory object allocation in bytes.
  uint64_t maxMemAllocSize_;

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
  uint32_t imageSupport_;

  //! Max size in bytes of the arguments that can be passed to a kernel.
  size_t maxParameterSize_;

  //! Maximum number of samplers that can be used in a kernel.
  uint32_t maxSamplers_;

  //! Describes the alignment in bits of the base address of any
  //  allocated memory object.
  uint32_t memBaseAddrAlign_;

  //! The smallest alignment in bytes which can be used for any data type.
  uint32_t minDataTypeAlignSize_;

  //! Describes single precision floating point capability of the device.
  cl_device_fp_config halfFPConfig_;
  cl_device_fp_config singleFPConfig_;
  cl_device_fp_config doubleFPConfig_;

  //! Type of global memory cache supported.
  cl_device_mem_cache_type globalMemCacheType_;

  //! Size of global memory cache line in bytes.
  uint32_t globalMemCacheLineSize_;

  //! Size of global memory cache in bytes.
  uint64_t globalMemCacheSize_;

  //! Size of global device memory in bytes.
  uint64_t globalMemSize_;

  //! Max size in bytes of a constant buffer allocation.
  uint64_t maxConstantBufferSize_;

  //! Preferred size in bytes of a constant buffer allocation.
  uint64_t preferredConstantBufferSize_;

  //! Max number of arguments declared
  uint32_t maxConstantArgs_;

  //! This is used to determine the type of local memory that is available
  cl_device_local_mem_type localMemType_;

  //! Size of local memory arena in bytes.
  uint64_t localMemSize_;

  //! If enabled, implies that all the memories, caches, registers etc. in
  //  the device implement error correction.
  uint32_t errorCorrectionSupport_;

  //! CL_TRUE if the device and the host have a unified memory subsystem and
  //  is CL_FALSE otherwise.
  uint32_t hostUnifiedMemory_;

  //! Describes the resolution of device timer.
  size_t profilingTimerResolution_;

  //! Timer starting point offset to Epoch.
  uint64_t profilingTimerOffset_;

  //! CL_TRUE if device is a little endian device.
  uint32_t littleEndian_;

  //! If enabled, implies that commands can be submitted to command-queues
  //  created on this device.
  uint32_t available_;

  //! if the implementation does not have a compiler available to compile
  //  the program source.
  uint32_t compilerAvailable_;

  //! Describes the execution capabilities of the device.
  cl_device_exec_capabilities executionCapabilities_;

  //! Describes the SVM capabilities of the device.
  cl_device_svm_capabilities svmCapabilities_;

  //! Preferred alignment for OpenCL fine-grained SVM atomic types.
  uint32_t preferredPlatformAtomicAlignment_;

  //! Preferred alignment for OpenCL global atomic types.
  uint32_t preferredGlobalAtomicAlignment_;

  //! Preferred alignment for OpenCL local atomic types.
  uint32_t preferredLocalAtomicAlignment_;

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
  uint32_t linkerAvailable_;

  //! Returns the list of built-in kernels, supported by the device
  const char* builtInKernels_;

  //! Returns max number of pixels for a 1D image created from a buffer object
  size_t imageMaxBufferSize_;

  //! Returns max number of images in a 1D or 2D image array
  size_t imageMaxArraySize_;

  //! Returns CL_TRUE if the devices preference is for the user to be
  //! responsible for synchronization
  uint32_t preferredInteropUserSync_;

  //! Returns maximum size of the internal buffer that holds the output
  //! of printf calls from a kernel
  size_t printfBufferSize_;

  //! Indicates maximum number of supported global atomic counters
  uint32_t maxAtomicCounters_;

  //! Returns the topology for the device
  cl_device_topology_amd deviceTopology_;

  //! Semaphore information
  uint32_t maxSemaphores_;
  uint32_t maxSemaphoreSize_;

  //! Returns the SKU board name for the device
  char boardName_[128];

  //! Number of SIMD (Single Instruction Multiple Data) units per compute unit
  //! that execute in parallel. All work items from the same work group must be
  //! executed by SIMDs in the same compute unit.
  uint32_t simdPerCU_;
  uint32_t cuPerShaderArray_;  //!< Number of CUs per shader array
  //! The maximum number of work items from the same work group that can be
  //! executed by a SIMD in parallel
  uint32_t simdWidth_;
  //! The number of instructions that a SIMD can execute in parallel
  uint32_t simdInstructionWidth_;
  //! The number of workitems per wavefront
  uint32_t wavefrontWidth_;
  //! Available number of SGPRs
  uint32_t availableSGPRs_;
  //! Number of global memory channels
  uint32_t globalMemChannels_;
  //! Number of banks in each global memory channel
  uint32_t globalMemChannelBanks_;
  //! Width in bytes of each of global memory bank
  uint32_t globalMemChannelBankWidth_;
  //! Local memory size per CU
  uint32_t localMemSizePerCU_;
  //! Number of banks of local memory
  uint32_t localMemBanks_;
  //! The core engine GFXIP version
  uint32_t gfxipVersion_;
  //! Number of available async queues
  uint32_t numAsyncQueues_;
  //! Number of available real time queues
  uint32_t numRTQueues_;
  //! Number of available real time compute units
  uint32_t numRTCUs_;
  //! Thread trace enable
  uint32_t threadTraceEnable_;
  //! ECC protected GPRs support (only available Vega20+)
  uint32_t sramEccEnabled_;

  //! Image pitch alignment for image2d_from_buffer
  uint32_t imagePitchAlignment_;
  //! Image base address alignment for image2d_from_buffer
  uint32_t imageBaseAddressAlignment_;

  //! Describes whether buffers from images are supported
  uint32_t bufferFromImageSupport_;

  //! Returns the supported SPIR versions for the device
  const char* spirVersions_;

  //! OpenCL20 device info fields:

  //! The max number of pipe objects that can be passed as arguments to a kernel
  uint32_t maxPipeArgs_;
  //! The max number of reservations that can be active for a pipe per work-item in a kernel
  uint32_t maxPipeActiveReservations_;
  //! The max size of pipe packet in bytes
  uint32_t maxPipePacketSize_;

  //! The command-queue properties supported of the device queue.
  cl_command_queue_properties queueOnDeviceProperties_;
  //! The preferred size of the device queue in bytes
  uint32_t queueOnDevicePreferredSize_;
  //! The max size of the device queue in bytes
  uint32_t queueOnDeviceMaxSize_;
  //! The maximum number of device queues
  uint32_t maxOnDeviceQueues_;
  //! The maximum number of events in use on a device queue
  uint32_t maxOnDeviceEvents_;

  //! The maximum size of global scope variables
  size_t maxGlobalVariableSize_;
  size_t globalVariablePreferredTotalSize_;
  //! Driver store location
  char driverStore_[200];
  //! Device ID
  uint32_t pcieDeviceId_;
  //! Revision ID
  uint32_t pcieRevisionId_;

  //! Max numbers of threads per CU
  uint32_t maxThreadsPerCU_;

  //! GPU device supports a launch of cooperative groups
  uint32_t cooperativeGroups_;
  //! GPU device supports a launch of cooperative groups on multiple devices
  uint32_t cooperativeMultiDeviceGroups_;

  //! large bar support.
  bool largeBar_;
};

//! Device settings
class Settings : public amd::HeapObject {
 public:
  uint64_t extensions_;  //!< Supported OCL extensions
  union {
    struct {
      uint overrideLclSet : 3;        //!< Bit mask to override the local size
      uint apuSystem_ : 1;            //!< Device is APU system with shared memory
      uint supportRA_ : 1;            //!< Support RA channel order format
      uint waitCommand_ : 1;          //!< Enables a wait for every submitted command
      uint customHostAllocator_ : 1;  //!< True if device has custom host allocator
                                      //  that replaces generic OS allocation routines
      uint supportDepthsRGB_ : 1;     //!< Support DEPTH and sRGB channel order format
      uint enableHwDebug_ : 1;        //!< Enable HW debug support
      uint reportFMAF_ : 1;           //!< Report FP_FAST_FMAF define in CL program
      uint reportFMA_ : 1;            //!< Report FP_FAST_FMA define in CL program
      uint singleFpDenorm_ : 1;       //!< Support Single FP Denorm
      uint hsailExplicitXnack_ : 1;   //!< Xnack in hsail path for this deivce
      uint useLightning_ : 1;         //!< Enable LC path for this device
      uint enableWgpMode_ : 1;        //!< Enable WGP mode for this device
      uint enableWave32Mode_ : 1;     //!< Enable Wave32 mode for this device
      uint lcWavefrontSize64_ : 1;    //!< Enable Wave64 mode for this device
      uint enableXNACK_ : 1;          //!< Enable XNACK feature
      uint enableCoopGroups_ : 1;     //!< Enable cooperative groups feature
      uint enableCoopMultiDeviceGroups_ : 1; //!< Enable cooperative groups multi device
      uint fenceScopeAgent_ : 1;      //!< Enable fence scope agent in AQL dispatch packet
      uint reserved_ : 11;
    };
    uint value_;
  };

  uint commandQueues_;  //!< Field value for maximum number
                        //!< concurrent Virtual GPUs for each backend

  //! Default constructor
  Settings();

  //! Check the specified extension
  bool checkExtension(uint name) const {
    return (extensions_ & (static_cast<uint64_t>(1) << name)) ? true : false;
  }

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
class Memory : public amd::HeapObject {
 public:
  //! Resource map flags
  enum CpuMapFlags {
    CpuReadWrite = 0x00000000,  //!< Lock for CPU read/Write
    CpuReadOnly = 0x00000001,   //!< Lock for CPU read only operation
    CpuWriteOnly = 0x00000002,  //!< Lock for CPU write only operation
  };

  union SyncFlags {
    struct {
      uint skipParent_ : 1;  //!< Skip parent synchronization
      uint skipViews_ : 1;   //!< Skip views synchronization
      uint skipEntire_ : 1;  //!< Skip entire synchronization
    };
    uint value_;
    SyncFlags() : value_(0) {}
  };

  struct WriteMapInfo : public amd::HeapObject {
    amd::Coord3D origin_;  //!< Origin of the map location
    amd::Coord3D region_;  //!< Mapped region
    amd::Image* baseMip_;  //!< The base mip level for images
    union {
      struct {
        uint32_t count_ : 8;       //!< The same map region counter
        uint32_t unmapWrite_ : 1;  //!< Unmap write operation
        uint32_t unmapRead_ : 1;   //!< Unmap read operation
        uint32_t entire_ : 1;      //!< Process the entire memory
      };
      uint32_t flags_;
    };

    //! Returns the state of entire map
    bool isEntire() const { return (entire_) ? true : false; }

    //! Returns the state of map write flag
    bool isUnmapWrite() const { return (unmapWrite_) ? true : false; }

    //! Returns the state of map read flag
    bool isUnmapRead() const { return (unmapRead_) ? true : false; }

    WriteMapInfo() : origin_(0, 0, 0), region_(0, 0, 0), baseMip_(NULL), flags_(0) {}
  };

  //! Constructor (from an amd::Memory object).
  Memory(amd::Memory& owner)
      : flags_(0), owner_(&owner), version_(0), mapMemory_(NULL), indirectMapCount_(0) {
    size_ = owner.getSize();
  }

  //! Constructor (no owner), always eager allocation.
  Memory(size_t size)
      : flags_(0), owner_(NULL), version_(0), mapMemory_(NULL), indirectMapCount_(0), size_(size) {}

  enum GLResourceOP {
    GLDecompressResource = 0,  // orders the GL driver to decompress any depth-stencil or MSAA
                               // resource to be sampled by a CL kernel.
    GLInvalidateFBO  // orders the GL driver to invalidate any FBO the resource may be bound to,
                     // since the resource internal state changed.
  };

  //! Default destructor for the device memory object
  virtual ~Memory(){};

  //! Releases virtual objects associated with this memory
  void releaseVirtual();

  //! Read the size
  size_t size() const { return size_; }

  //! Gets the owner Memory instance
  amd::Memory* owner() const { return owner_; }

  //! Immediate blocking write from device cache to owners's backing store.
  //! Marks owner as "current" by resetting the last writer to NULL.
  virtual void syncHostFromCache(SyncFlags syncFlags = SyncFlags()) {}

  //! Allocate memory for API-level maps
  virtual void* allocMapTarget(const amd::Coord3D& origin,  //!< The map location in memory
                               const amd::Coord3D& region,  //!< The map region in memory
                               uint mapFlags,               //!< Map flags
                               size_t* rowPitch = NULL,     //!< Row pitch for the mapped memory
                               size_t* slicePitch = NULL    //!< Slice for the mapped memory
  ) {
    return NULL;
  }

  virtual bool pinSystemMemory(void* hostPtr,  //!< System memory address
                               size_t size     //!< Size of allocated system memory
  ) {
    return true;
  }

  //! Releases indirect map surface
  virtual void releaseIndirectMap() {}
  //! decompress any MSAA/depth-stencil interop surfaces.
  //! notify GL to invalidate any surfaces touched by a CL kernel
  virtual bool processGLResource(GLResourceOP operation) { return false; }

  //! Map the device memory to CPU visible
  virtual void* cpuMap(VirtualDevice& vDev,  //!< Virtual device for map operaiton
                       uint flags = 0,       //!< flags for the map operation
                       // Optimization for multilayer map/unmap
                       uint startLayer = 0,       //!< Start layer for multilayer map
                       uint numLayers = 0,        //!< End layer for multilayer map
                       size_t* rowPitch = NULL,   //!< Row pitch for the device memory
                       size_t* slicePitch = NULL  //!< Slice pitch for the device memory
  ) {
    amd::Image* image = owner()->asImage();
    if (image != NULL) {
      *rowPitch = image->getRowPitch();
      *slicePitch = image->getSlicePitch();
    }
    // Default behavior uses preallocated host mem for CPU
    return owner()->getHostMem();
  }

  //! Unmap the device memory
  virtual void cpuUnmap(VirtualDevice& vDev  //!< Virtual device for unmap operaiton
  ) {}

  //! Saves map info for this object
  //! @note: It's not a thread safe operation, the app must implement
  //! synchronization for the multiple write maps if necessary
  void saveMapInfo(const void* mapAddress,        //!< Map cpu address
                   const amd::Coord3D origin,     //!< Origin of the map location
                   const amd::Coord3D region,     //!< Mapped region
                   uint mapFlags,                 //!< Map flags
                   bool entire,                   //!< True if the enitre memory was mapped
                   amd::Image* baseMip = nullptr  //!< The base mip level for map
  );

  const WriteMapInfo* writeMapInfo(const void* mapAddress) const {
    // Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());

    auto it = writeMapInfo_.find(mapAddress);
    if (it == writeMapInfo_.end()) {
      if (writeMapInfo_.size() == 0) {
        LogError("Unmap is a NOP!");
        return nullptr;
      }
      LogWarning("Unknown unmap signature!");
      // Get the first map info
      it = writeMapInfo_.begin();
    }
    return &it->second;
  }

  //! Clear memory object as mapped read only
  void clearUnmapInfo(const void* mapAddress) {
    // Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());
    auto it = writeMapInfo_.find(mapAddress);
    if (it == writeMapInfo_.end()) {
      // Get the first map info
      it = writeMapInfo_.begin();
    }
    if (--it->second.count_ == 0) {
      writeMapInfo_.erase(it);
    }
  }

  //! Returns the state of memory direct access flag
  bool isHostMemDirectAccess() const { return (flags_ & HostMemoryDirectAccess) ? true : false; }

  //! Returns the state of host memory registration flag
  bool isHostMemoryRegistered() const { return (flags_ & HostMemoryRegistered) ? true : false; }

  //! Returns the state of CPU uncached access
  bool isCpuUncached() const { return (flags_ & MemoryCpuUncached) ? true : false; }

  virtual uint64_t virtualAddress() const { return 0; }

  //! Returns CPU pointer to HW state
  virtual const address cpuSrd() const { return nullptr; }

  virtual void IpcCreate(size_t offset, size_t* mem_size, void* handle) const {
    ShouldNotReachHere();
  }

 protected:
  enum Flags {
    HostMemoryDirectAccess = 0x00000001,  //!< GPU has direct access to the host memory
    MapResourceAlloced = 0x00000002,      //!< Map resource was allocated
    PinnedMemoryAlloced = 0x00000004,     //!< An extra pinned resource was allocated
    SubMemoryObject = 0x00000008,         //!< Memory is sub-memory
    HostMemoryRegistered = 0x00000010,    //!< Host memory was registered
    MemoryCpuUncached = 0x00000020        //!< Memory is uncached on CPU access(slow read)
  };
  uint flags_;  //!< Memory object flags

  amd::Memory* owner_;  //!< The Memory instance that we cache,
                        //!< or NULL if we're device-private workspace.

  volatile size_t version_;  //!< The version we're currently shadowing

  //! NB, the map data below is for an API-level map (from clEnqueueMapBuffer),
  //! not a physical map. When a memory object does not use USE_HOST_PTR we
  //! can use a remote resource and DMA, avoiding the additional CPU memcpy.
  amd::Memory* mapMemory_;            //!< Memory used as map target buffer
  volatile size_t indirectMapCount_;  //!< Number of maps
  std::unordered_map<const void*, WriteMapInfo>
      writeMapInfo_;  //!< Saved write map info for partial unmap

  //! Increment map count
  void incIndMapCount() { ++indirectMapCount_; }

  //! Decrement map count
  virtual void decIndMapCount() {}

  size_t size_;  //!< Memory size

 private:
  //! Disable default copy constructor
  Memory& operator=(const Memory&) = delete;

  //! Disable operator=
  Memory(const Memory&) = delete;
};

class Sampler : public amd::HeapObject {
 public:
  //! Constructor
  Sampler() : hwSrd_(0), hwState_(nullptr) {}

  //! Default destructor for the device memory object
  virtual ~Sampler(){};

  //! Returns device specific HW state for the sampler
  uint64_t hwSrd() const { return hwSrd_; }

  //! Returns CPU pointer to HW state
  const address hwState() const { return hwState_; }

 protected:
  uint64_t hwSrd_;   //!< Device specific HW state for the sampler
  address hwState_;  //!< CPU pointer to HW state

 private:
  //! Disable default copy constructor
  Sampler& operator=(const Sampler&);

  //! Disable operator=
  Sampler(const Sampler&);
};

class ClBinary : public amd::HeapObject {
 public:
  enum BinaryImageFormat {
    BIF_VERSION2 = 0,  //!< Binary Image Format version 2.0 (ELF)
    BIF_VERSION3       //!< Binary Image Format version 3.0 (ELF)
  };

  //! Constructor
  ClBinary(const amd::Device& dev, BinaryImageFormat bifVer = BIF_VERSION3);

  //! Destructor
  virtual ~ClBinary();

  void init(amd::option::Options* optionsObj, bool amdilRequired = false);

  /** called only in loading image routines,
      never called in storing routines */
  bool setBinary(const char* theBinary, size_t theBinarySize, bool allocated = false);

  //! setin elfIn_
  bool setElfIn();
  void resetElfIn();

  //! set out elf
  bool setElfOut(unsigned char eclass, const char* outFile);
  void resetElfOut();

  //! Set elf header information
  virtual bool setElfTarget();

  // class used in for loading images in new format
  amd::OclElf* elfIn() { return elfIn_; }

  // classes used storing and loading images in new format
  amd::OclElf* elfOut() { return elfOut_; }
  void elfOut(amd::OclElf* v) { elfOut_ = v; }

  //! Create and save ELF binary image
  bool createElfBinary(bool doencrypt, Program::type_t type);

  // save BIF binary image
  void saveBIFBinary(const char* binaryIn, size_t size);

  bool decryptElf(const char* binaryIn, size_t size, char** decryptBin, size_t* decryptSize,
                  int* encryptCode);

  //! Returns the binary pair for the abstraction layer
  Program::binary_t data() const;

  //! Loads llvmir binary from OCL binary file
  bool loadLlvmBinary(
      std::string& llvmBinary,                     //!< LLVMIR binary code
      amd::OclElf::oclElfSections& elfSectionType  //!< LLVMIR binary is in SPIR format
      ) const;

  //! Loads compile options from OCL binary file
  bool loadCompileOptions(std::string& compileOptions  //!< return the compile options loaded
                          ) const;

  //! Loads link options from OCL binary file
  bool loadLinkOptions(std::string& linkOptions  //!< return the link options loaded
                       ) const;

  //! Store compile options into OCL binary file
  void storeCompileOptions(const std::string& compileOptions  //!< the compile options to be stored
  );

  //! Store link options into OCL binary file
  void storeLinkOptions(const std::string& linkOptions  //!< the link options to be stored
  );

  //! Check if the binary is recompilable
  bool isRecompilable(std::string& llvmBinary, amd::OclElf::oclElfPlatform thePlatform);

  void saveOrigBinary(const char* origBinary, size_t origSize) {
    origBinary_ = origBinary;
    origSize_ = origSize;
  }

  void restoreOrigBinary() {
    if (origBinary_ != NULL) {
      (void)setBinary(origBinary_, origSize_, false);
    }
  }

  //! Set Binary flags
  void setFlags(int encryptCode);

  bool saveSOURCE() { return ((flags_ & BinarySourceMask) == BinarySaveSource); }
  bool saveLLVMIR() { return ((flags_ & BinaryLlvmirMask) == BinarySaveLlvmir); }
  bool saveAMDIL() { return ((flags_ & BinaryAmdilMask) == BinarySaveAmdil); }
  bool saveISA() { return ((flags_ & BinaryIsaMask) == BinarySaveIsa); }

  bool saveAS() { return ((flags_ & BinaryASMask) == BinarySaveAS); }

  // Return the encrypt code for this input binary ( "> 0" means encrypted)
  int getEncryptCode() { return encryptCode_; }

  // Returns TRUE of binary file is SPIR
  bool isSPIR() const;
  // Returns TRUE of binary file is SPIRV
  bool isSPIRV() const;

 protected:
  enum Flags {
    BinaryAllocated = 0x1,  //!< Binary was created

    // Source control
    BinaryNoSaveSource = 0x0,  // 0: default
    BinaryRemoveSource = 0x2,  // for encrypted binary
    BinarySaveSource = 0x4,
    BinarySourceMask = 0x6,

    // LLVMIR control
    BinarySaveLlvmir = 0x0,    // 0: default
    BinaryRemoveLlvmir = 0x8,  // for encrypted binary
    BinaryNoSaveLlvmir = 0x10,
    BinaryLlvmirMask = 0x18,

    // AMDIL control
    BinarySaveAmdil = 0x0,     // 0: default
    BinaryRemoveAmdil = 0x20,  // for encrypted binary
    BinaryNoSaveAmdil = 0x40,
    BinaryAmdilMask = 0x60,

    // ISA control
    BinarySaveIsa = 0x0,     // 0: default
    BinaryRemoveIsa = 0x80,  // for encrypted binary
    BinaryNoSaveIsa = 0x100,
    BinaryIsaMask = 0x180,

    // AS control
    BinaryNoSaveAS = 0x0,    // 0: default
    BinaryRemoveAS = 0x200,  // for encrypted binary
    BinarySaveAS = 0x400,
    BinaryASMask = 0x600
  };

  //! Returns TRUE if binary file was allocated
  bool isBinaryAllocated() const { return (flags_ & BinaryAllocated) ? true : false; }

  //! Returns BIF symbol name by symbolID,
  //! returns empty string if not found or if BIF version is unsupported
  std::string getBIFSymbol(unsigned int symbolID) const;

 protected:
  const amd::Device& dev_;  //!< Device object

 private:
  //! Disable default copy constructor
  ClBinary(const ClBinary&);

  //! Disable default operator=
  ClBinary& operator=(const ClBinary&);

  //! Releases the binary data store
  void release();

  const char* binary_;  //!< binary data
  size_t size_;         //!< binary size
  uint flags_;          //!< CL binary object flags

  const char* origBinary_;  //!< original binary data
  size_t origSize_;         //!< original binary size

  int encryptCode_;  //!< Encryption Code for input binary (0 for not encrypted)

 protected:
  amd::OclElf* elfIn_;        //!< ELF object for input ELF binary
  amd::OclElf* elfOut_;       //!< ELF object for output ELF binary
  BinaryImageFormat format_;  //!< which binary image format to use
};

inline const Program::binary_t Program::binary() const {
  if (clBinary() == NULL) {
    return {(const void*)0, 0};
  }
  return clBinary()->data();
}

inline Program::binary_t Program::binary() {
  if (clBinary() == NULL) {
    return {(const void*)0, 0};
  }
  return clBinary()->data();
}

/*! \class PerfCounter
 *
 *  \brief The device interface class for the performance counters
 */
class PerfCounter : public amd::HeapObject {
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
class ThreadTrace : public amd::HeapObject {
 public:
  //! Constructor for the device performance
  ThreadTrace() {}
  //! Update ThreadTrace status to true/false if new buffer was binded/unbinded respectively
  virtual void setNewBufferBinded(bool) = 0;
  //! Get the performance counter info
  virtual bool info(uint infoType, uint* info, uint infoSize) const = 0;
  //! Destructor for PerfCounter class
  virtual ~ThreadTrace() {}

 private:
  //! Disable default copy constructor
  ThreadTrace(const ThreadTrace&);

  //! Disable default operator=
  ThreadTrace& operator=(const ThreadTrace&);
};

//! A device execution environment.
class VirtualDevice : public amd::HeapObject {
 public:
  //! Construct a new virtual device for the given physical device.
  VirtualDevice(amd::Device& device)
    : device_(device)
    , blitMgr_(NULL)
    , execution_("Virtual device execution lock", true)
    , index_(0) {}

  //! Destroy this virtual device.
  virtual ~VirtualDevice() {}

  //! Prepare this virtual device for destruction.
  virtual bool terminate() = 0;

  //! Return the physical device for this virtual device.
  const amd::Device& device() const { return device_(); }

  virtual void submitReadMemory(amd::ReadMemoryCommand& cmd) = 0;
  virtual void submitWriteMemory(amd::WriteMemoryCommand& cmd) = 0;
  virtual void submitCopyMemory(amd::CopyMemoryCommand& cmd) = 0;
  virtual void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd) = 0;
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
  virtual void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd) = 0;
  virtual void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd) = 0;
  virtual void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd) = 0;
  virtual void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd) = 0;
  virtual void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd) = 0;
  /// Optional extensions
  virtual void submitSignal(amd::SignalCommand& cmd) = 0;
  virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& cmd) = 0;
  virtual void submitTransferBufferFromFile(amd::TransferBufferFileCommand& cmd) {
    ShouldNotReachHere();
  }

  //! Get the blit manager object
  device::BlitManager& blitMgr() const { return *blitMgr_; }

  //! Returns the monitor object for execution access by VirtualGPU
  amd::Monitor& execution() { return execution_; }

  //! Returns the virtual device unique index
  uint index() const { return index_; }

  //! Returns true if device has active wait setting
  bool ActiveWait() const;

 private:
  //! Disable default copy constructor
  VirtualDevice& operator=(const VirtualDevice&);

  //! Disable operator=
  VirtualDevice(const VirtualDevice&);

  //! The physical device that this virtual device utilizes
  amd::SharedReference<amd::Device> device_;

 protected:
  device::BlitManager* blitMgr_;  //!< Blit manager

  amd::Monitor execution_;  //!< Lock to serialise access to all device objects
  uint index_;              //!< The virtual device unique index
};

}  // namespace device

namespace amd {

//! MemoryObject map lookup  class
class MemObjMap : public AllStatic {
 public:
  static size_t size();  //!< obtain the size of the container
  static void AddMemObj(const void* k,
                        amd::Memory* v);  //!< add the host mem pointer and buffer in the container
  static void RemoveMemObj(const void* k);  //!< Remove an entry of mem object from the container
  static amd::Memory* FindMemObj(
      const void* k);  //!< find the mem object based on the input pointer
 private:
  static std::map<uintptr_t, amd::Memory*>
      MemObjMap_;                      //!< the mem object<->hostptr information container
  static amd::Monitor AllocatedLock_;  //!< amd monitor locker
};

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Device Device Abstraction
 *  @{
 */
class Device : public RuntimeObject {
 protected:
  typedef aclCompiler Compiler;

 public:
  // The structures below for MGPU launch match the device library format
  struct MGSyncData {
    uint32_t w0;
    uint32_t w1;
  };

  struct MGSyncInfo {
    struct MGSyncData* mgs;
    uint32_t grid_id;
    uint32_t num_grids;
    uint64_t prev_sum;
    uint64_t all_sum;
  };

  static constexpr size_t kP2PStagingSize = 4 * Mi;
  static constexpr size_t kMGSyncDataSize = sizeof(MGSyncData);
  static constexpr size_t kMGInfoSizePerDevice = kMGSyncDataSize + sizeof(MGSyncInfo);

  typedef std::list<CommandQueue*> CommandQueues;

  struct BlitProgram : public amd::HeapObject {
    Program* program_;  //!< GPU program obejct
    Context* context_;  //!< A dummy context

    BlitProgram(Context* context) : program_(NULL), context_(context) {}
    ~BlitProgram();

    //! Creates blit program for this device
    bool create(Device* device,                  //!< Device object
                const char* extraKernel = NULL,  //!< Extra kernels from the device layer
                const char* extraOptions = NULL  //!< Extra compilation options
    );
  };

  virtual Compiler* compiler() const = 0;
  virtual Compiler* binCompiler() const { return compiler(); }

  Device();
  virtual ~Device();

  //! Initializes abstraction layer device object
  bool create();

  uint retain() {
    // Overwrite the RuntimeObject::retain().
    // There is an issue in the old SHOC11_DeviceMemory test on TC
    return 0u;
  }

  uint release() {
    // Overwrite the RuntimeObject::release().
    // There is an issue in the old SHOC11_DeviceMemory test on TC
    return 0u;
  }

  //! Register a device as available
  void registerDevice();

  //! Initialize the device layer (enumerate known devices)
  static bool init();

  //! Shutdown the device layer
  static void tearDown();

  static std::vector<Device*> getDevices(cl_device_type type,  //!< Device type
                                         bool offlineDevices   //!< Enable offline devices
  );

  static size_t numDevices(cl_device_type type,  //!< Device type
                           bool offlineDevices   //!< Enable offline devices
  );

  static bool getDeviceIDs(cl_device_type deviceType,  //!< Device type
                           uint32_t numEntries,         //!< Number of entries in the array
                           cl_device_id* devices,      //!< Array of the device ID(s)
                           uint32_t* numDevices,        //!< Number of available devices
                           bool offlineDevices         //!< Report offline devices
  );

  const device::Info& info() const { return info_; }

  //! Return svm support capability.
  bool svmSupport() const {
    return (info().svmCapabilities_ &
            (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
             CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) != 0
        ? true
        : false;
  }

  //! check svm FGS support capability.
  inline bool isFineGrainedSystem(bool FGSOPT = false) const {
    return FGSOPT && (info().svmCapabilities_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) != 0 ? true
                                                                                      : false;
  }

  //! Return this device's type.
  cl_device_type type() const { return info().type_ & ~(CL_DEVICE_TYPE_DEFAULT); }

  //! Create a new virtual device environment.
  virtual device::VirtualDevice* createVirtualDevice(CommandQueue* queue = NULL) = 0;

  //! Create a program for device.
  virtual device::Program* createProgram(amd::Program& owner, option::Options* options = NULL) = 0;

  //! Allocate a chunk of device memory as a cache for a CL memory object
  virtual device::Memory* createMemory(Memory& owner) const = 0;

  //! Allocate a device sampler object
  virtual bool createSampler(const Sampler&, device::Sampler**) const = 0;

  //! Allocates a view object from the device memory
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
      ) const = 0;

  //! Return true if initialized external API interop, otherwise false
  virtual bool bindExternalDevice(
      uint flags,             //!< Enum val. for ext.API type: GL, D3D10, etc.
      void* const pDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
      void* pContext,         //!< HGLRC/GLXContext handle
      bool validateOnly  //! Only validate if the device can inter-operate with pDevice/pContext, do
                         //! not bind.
      ) = 0;

  virtual bool unbindExternalDevice(
      uint flags,             //!< Enum val. for ext.API type: GL, D3D10, etc.
      void* const pDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
      void* pContext,         //!< HGLRC/GLXContext handle
      bool validateOnly  //! Only validate if the device can inter-operate with pDevice/pContext, do
                         //! not bind.
      ) = 0;

  //! resolves GL depth/msaa buffer
  virtual bool resolveGLMemory(device::Memory*) const { return true; }

  //! Gets free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory  //!< Free memory information on a GPU device
                                ) const = 0;

  /**
   * @return True if the device has its own custom host allocator to be used
   * instead of the generic OS allocation routines
   */
  bool customHostAllocator() const { return settings().customHostAllocator_ == 1; }

  /**
   * @copydoc amd::Context::hostAlloc
   */
  virtual void* hostAlloc(size_t size, size_t alignment, bool atomics = false) const {
    ShouldNotCallThis();
    return NULL;
  }

  /**
   * @copydoc amd::Context::hostFree
   */
  virtual void hostFree(void* ptr, size_t size = 0) const { ShouldNotCallThis(); }

  /**
   * @copydoc amd::Context::svmAlloc
   */
  virtual void* svmAlloc(Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                         void* svmPtr) const = 0;
  /**
   * @copydoc amd::Context::svmFree
   */
  virtual void svmFree(void* ptr) const = 0;

  //! Validate kernel
  virtual bool validateKernel(const amd::Kernel& kernel,
                              const device::VirtualDevice* vdev,
                              bool coop_groups = false) {
    return true;
  };

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                            cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
    return true;
  };
  //! Returns TRUE if the device is available for computations
  bool isOnline() const { return online_; }

  //! Returns device settings
  const device::Settings& settings() const { return *settings_; }

  //! Returns blit program info structure
  BlitProgram* blitProgram() const { return blitProgram_; }

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeDevice; }

  //! Returns app profile
  static const AppProfile* appProfile() { return &appProfile_; }

  //! Register a hardware debugger manager
  HwDebugManager* hwDebugMgr() const { return hwDebugMgr_; }

  //! Initialize the Hardware Debug Manager
  virtual int32_t hwDebugManagerInit(amd::Context* context, uintptr_t messageStorage) {
    return CL_SUCCESS;
  }

  //! Remove the Hardware Debug Manager
  virtual void hwDebugManagerRemove() {}

  //! Adds GPU memory to the VA cache list
  void addVACache(device::Memory* memory) const;

  //! Removes GPU memory from the VA cache list
  void removeVACache(const device::Memory* memory) const;

  //! Finds GPU memory from virtual address
  device::Memory* findMemoryFromVA(const void* ptr, size_t* offset) const;

  static std::vector<Device*>& devices() { return *devices_; }

  // P2P devices that are accessible from the current device
  std::vector<cl_device_id> p2pDevices_;

  // P2P devices for memory allocation. This list contains devices that can have access to the
  // current device
  std::vector<Device*> p2p_access_devices_;

  //! Checks if OCL runtime can use code object manager for compilation
  bool ValidateComgr();

  virtual amd::Memory* IpcAttach(const void* handle, size_t mem_size, unsigned int flags,
                                 void** dev_ptr) const {
    ShouldNotReachHere();
    return nullptr;
  }

  virtual void IpcDetach(amd::Memory& memory) const { ShouldNotReachHere(); }

  //! Return private global device context for P2P allocations
  amd::Context& GlbCtx() const { return *glb_ctx_; }

  //! Lock protect P2P staging operations
  Monitor& P2PStageOps() const { return p2p_stage_ops_; }

  //! Staging buffer for P2P transfer
  Memory* P2PStage() const { return p2p_stage_; }

  //! Does this device allow P2P access?
  bool P2PAccessAllowed() const { return (p2p_access_devices_.size() > 0) ? true : false; }

  //! Returns the list of devices that can have access to the current
  const std::vector<Device*>& P2PAccessDevices() const { return p2p_access_devices_; }

  //! Returns index of current device
  uint32_t index() const { return index_; }

  virtual bool findLinkTypeAndHopCount(amd::Device* other_device, uint32_t* link_type,
                                       uint32_t* hop_count) {
    ShouldNotReachHere();
    return false;
  }

  // Notifies device about context destroy
  virtual void ContextDestroy() {}

  //! Returns active wait state for this device
  bool ActiveWait() const { return activeWait_; }

  void SetActiveWait(bool state) { activeWait_ = state; }

 protected:
  //! Enable the specified extension
  char* getExtensionString();

  device::Info info_;             //!< Device info structure
  device::Settings* settings_;    //!< Device settings
  union {
    struct {
      uint32_t online_: 1;        //!< The device in online
      uint32_t activeWait_: 1;    //!< If true device requires active wait
    };
    uint32_t  state_;             //!< State bit mask
  };

  BlitProgram* blitProgram_;      //!< Blit program info
  static AppProfile appProfile_;  //!< application profile
  HwDebugManager* hwDebugMgr_;    //!< Hardware Debug manager

  static amd::Context* glb_ctx_;      //!< Global context with all devices
  static amd::Monitor p2p_stage_ops_; //!< Lock to serialise cache for the P2P resources
  static Memory* p2p_stage_;          //!< Staging resources

 private:
  bool IsTypeMatching(cl_device_type type, bool offlineDevices);

#if defined(WITH_HSA_DEVICE)
  static AppProfile* rocAppProfile_;
#endif

  static std::vector<Device*>* devices_;  //!< All known devices

  Monitor* vaCacheAccess_;                            //!< Lock to serialize VA caching access
  std::map<uintptr_t, device::Memory*>* vaCacheMap_;  //!< VA cache map
  uint32_t index_;  //!< Unique device index
};

/*! @}
 *  @}
 */

}  // namespace amd

#endif /*DEVICE_HPP_*/
