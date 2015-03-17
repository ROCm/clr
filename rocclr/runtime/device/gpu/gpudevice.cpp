//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpumemory.hpp"
#include "device/gpu/gpudevice.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"
#include "thread/monitor.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpubinary.hpp"
#include "device/gpu/gpusettings.hpp"
#include "device/gpu/gpublit.hpp"

#include "acl.h"

#include "amdocl/cl_common.hpp"
#include "CL/cl_gl.h"

#ifdef _WIN32
#include <d3d9.h>
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#include "CL/cl_dx9_media_sharing.h"
#endif // _WIN32

#include "os_if.h" // for osInit()

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctype.h>

#include "gpudebugmanager.hpp"

bool DeviceLoad()
{
    bool    ret = false;

    // Create online devices
    ret |= gpu::Device::init();
    // Create offline GPU devices
    ret |= gpu::NullDevice::init();

    return ret;
}

void DeviceUnload()
{
    gpu::Device::tearDown();
}

namespace gpu {

aclCompiler* NullDevice::compiler_;
aclCompiler* NullDevice::hsaCompiler_;
AppProfile Device::appProfile_;

NullDevice::NullDevice()
    : amd::Device(NULL)
    , calTarget_(static_cast<CALtarget>(0))
    , hwInfo_(NULL)
{
}

bool
NullDevice::init()
{
    bool result = false;
    std::vector<Device*> devices;

    devices = getDevices(CL_DEVICE_TYPE_GPU, false);

    // Loop through all supported devices and create each of them
    for (uint id = CAL_TARGET_CYPRESS; id <= CAL_TARGET_LAST; ++id) {
        bool    foundActive = false;

        if (gpu::DeviceInfo[id].targetName_[0] == '\0') {
            continue;
        }

        // Loop through all active devices and see if we match one
        for (uint i = 0; i < devices.size(); ++i) {
            if (static_cast<NullDevice*>(devices[i])->calTarget() ==
                static_cast<CALtarget>(id)) {
                foundActive = true;
                break;
            }
        }

        // Don't report an offline device if it's active
        if (foundActive) {
            continue;
        }

        NullDevice*  dev = new NullDevice();
        if (NULL != dev) {
            if (!dev->create(static_cast<CALtarget>(id))) {
                delete dev;
            }
            else {
                result |= true;
                dev->registerDevice();
            }
        }
    }

    return result;
}

bool
NullDevice::create(CALtarget target)
{
    CALdeviceattribs      calAttr = {0};
    CALdeviceVideoAttribs calVideoAttr = {0};

    online_ = false;

    // Mark the device as GPU type
    info_.type_     = CL_DEVICE_TYPE_GPU;
    info_.vendorId_ = 0x1002;

    calTarget_ = calAttr.target = target;
    hwInfo_ = &DeviceInfo[calTarget_];

    // Report the device name
    ::strcpy(info_.name_, hwInfo()->targetName_);

    // Force double if it could be supported
    switch (target) {
    case CAL_TARGET_CAYMAN:
    case CAL_TARGET_CYPRESS:
    case CAL_TARGET_PITCAIRN:
    case CAL_TARGET_CAPEVERDE:
    case CAL_TARGET_TAHITI:
    case CAL_TARGET_OLAND:
    case CAL_TARGET_HAINAN:
    case CAL_TARGET_DEVASTATOR:
    case CAL_TARGET_SCRAPPER:
        calAttr.doublePrecision = CAL_TRUE;
        break;
    case CAL_TARGET_BONAIRE:
    case CAL_TARGET_SPECTRE:
    case CAL_TARGET_SPOOKY:
    case CAL_TARGET_KALINDI:
    case CAL_TARGET_HAWAII:
    case CAL_TARGET_ICELAND:
    case CAL_TARGET_TONGA:
    case CAL_TARGET_FIJI:
    case CAL_TARGET_GODAVARI:
    case CAL_TARGET_CARRIZO:
    case CAL_TARGET_ELLESMERE:
    case CAL_TARGET_BAFFIN:
        calAttr.doublePrecision = CAL_TRUE;
        calAttr.isOpenCL200Device = CAL_TRUE;
        break;
    default:
        break;
    }

    settings_ = new gpu::Settings();
    gpu::Settings* gpuSettings = reinterpret_cast<gpu::Settings*>(settings_);
    // Create setting for the offline target
    if ((gpuSettings == NULL) || !gpuSettings->create(calAttr
#if cl_amd_open_video
        , calVideoAttr
#endif //cl_amd_open_video
        )) {
        return false;
    }

    info_.maxWorkGroupSize_ = settings().maxWorkGroupSize_;

    // Initialize the extension string for offline devices
    info_.extensions_   = getExtensionString();

    // Fill the version info
    ::strcpy(info_.name_, hwInfo()->targetName_);
    ::strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");
    ::snprintf(info_.driverVersion_, sizeof(info_.driverVersion_) - 1,
        AMD_BUILD_STRING);
    if (settings().hsail_ || (settings().oclVersion_ == OpenCL20)) {
        info_.version_ = "OpenCL 2.0 " AMD_PLATFORM_INFO;
        info_.oclcVersion_ = "OpenCL C 2.0 ";
        // Runtime doesn't know what local size could be on the real board
        info_.maxGlobalVariableSize_ = static_cast<size_t>(512 * Mi);

        if (NULL == hsaCompiler_) {
            const char* library = getenv("HSA_COMPILER_LIBRARY");
            aclCompilerOptions opts = {
                sizeof(aclCompilerOptions_0_8),
                library,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                AMD_OCL_SC_LIB,
                &::malloc,
                &::free
            };
            // Initialize the compiler handle
            acl_error   error;
            hsaCompiler_ = aclCompilerInit(&opts, &error);
            if (error != ACL_SUCCESS) {
                 LogError("Error initializing the compiler");
                 return false;
            }
        }
    }
    else {
        info_.version_ = "OpenCL 1.2 " AMD_PLATFORM_INFO;
        info_.oclcVersion_ = "OpenCL C 1.2 ";
    }

    return true;
}

device::Program*
NullDevice::createProgram(int oclVer)
{
    device::Program* nullProgram;
    if (settings().hsail_ || (oclVer == 200)) {
        nullProgram = new HSAILProgram(*this);
    }
    else {
        nullProgram = new NullProgram(*this);
    }
    if (nullProgram == NULL) {
        LogError("Memory allocation has failed!");
    }

    return nullProgram;
}

void
Device::Engines::create(uint num, gslEngineDescriptor* desc, uint maxNumComputeRings)
{
    numComputeRings_ = 0;

    for (uint i = 0; i < num; ++i) {
        desc_[desc[i].id] = desc[i];
        desc_[desc[i].id].priority = GSL_ENGINEPRIORITY_NEUTRAL;

        if (desc[i].id >= GSL_ENGINEID_COMPUTE0 &&
            desc[i].id <= GSL_ENGINEID_COMPUTE7) {
            numComputeRings_++;
        }
    }

    numComputeRings_ = std::min(numComputeRings_, maxNumComputeRings);
}

uint
Device::Engines::getRequested(uint engines, gslEngineDescriptor* desc) const
{
    uint slot = 0;
    for (uint i = 0; i < GSL_ENGINEID_MAX; ++i) {
        if ((engines & getMask(static_cast<gslEngineID>(i))) &&
            (desc_[i].id == static_cast<gslEngineID>(i))) {
            desc[slot] = desc_[i];
            engines &= ~getMask(static_cast<gslEngineID>(i));
            slot++;
        }
    }
    return (engines == 0) ? slot : 0;
}

Device::XferBuffers::~XferBuffers()
{
    // Destroy temporary buffer for reads
    for (const auto& buf : freeBuffers_) {
        // CPU optimization: unmap staging buffer just once
        if (!buf->cal()->cardMemory_) {
            buf->unmap(NULL);
        }
        delete buf;
    }
    freeBuffers_.clear();
}

bool
Device::XferBuffers::create()
{
    Resource*   xferBuf = NULL;
    bool        result = false;
    // Note: create a 1D resource
    xferBuf = new Resource(dev(), bufSize_ / Heap::ElementSize,
        Heap::ElementType);

    // We will try to creat a CAL resource for the transfer buffer
    if ((NULL == xferBuf) || !xferBuf->create(type_)) {
        delete xferBuf;
        xferBuf = NULL;
        LogError("Couldn't allocate a transfer buffer!");
    }
    else {
        result = true;
        freeBuffers_.push_back(xferBuf);
        // CPU optimization: map staging buffer just once
        if (!xferBuf->cal()->cardMemory_) {
            xferBuf->map(NULL);
        }
    }

    return result;
}

Resource&
Device::XferBuffers::acquire()
{
    Resource*   xferBuf = NULL;
    size_t      listSize;

    // Lock the operations with the staged buffer list
    amd::ScopedLock  l(lock_);
    listSize = freeBuffers_.size();

    // If the list is empty, then attempt to allocate a staged buffer
    if (listSize == 0) {
        // Note: create a 1D resource
        xferBuf = new Resource(dev(), bufSize_ / Heap::ElementSize,
            Heap::ElementType);

        // We will try to create a CAL resource for the transfer buffer
        if ((NULL == xferBuf) || !xferBuf->create(type_)) {
            delete xferBuf;
            xferBuf = NULL;
            LogError("Couldn't allocate a transfer buffer!");
        }
        else {
            ++acquiredCnt_;
            // CPU optimization: map staging buffer just once
            if (!xferBuf->cal()->cardMemory_) {
                xferBuf->map(NULL);
            }
        }
    }

    if (xferBuf == NULL) {
        xferBuf = *(freeBuffers_.begin());
        freeBuffers_.erase(freeBuffers_.begin());
        ++acquiredCnt_;
    }

    return *xferBuf;
}

void
Device::XferBuffers::release(VirtualGPU& gpu, Resource& buffer)
{
    // Make sure buffer isn't busy on the current VirtualGPU, because
    // the next aquire can come from different queue
    buffer.wait(gpu);
    // Lock the operations with the staged buffer list
    amd::ScopedLock  l(lock_);
    freeBuffers_.push_back(&buffer);
    --acquiredCnt_;
}


Device::ScopedLockVgpus::ScopedLockVgpus(const Device& dev)
    : dev_(dev)
{
    // Lock the virtual GPU list
    dev_.vgpusAccess()->lock();

    // Find all available virtual GPUs and lock them
    // from the execution of commands
    for (uint idx = 0; idx < dev_.vgpus().size(); ++idx) {
        dev_.vgpus()[idx]->execution().lock();
    }
}

Device::ScopedLockVgpus::~ScopedLockVgpus()
{
    // Find all available virtual GPUs and unlock them
    // for the execution of commands
    for (uint idx = 0; idx < dev_.vgpus().size(); ++idx) {
        dev_.vgpus()[idx]->execution().unlock();
    }

    // Unock the virtual GPU list
    dev_.vgpusAccess()->unlock();
}

Device::Device()
    : NullDevice()
    , CALGSLDevice()
    , numOfVgpus_(0)
    , context_(NULL)
    , heap_(NULL)
    , dummyPage_(NULL)
    , lockAsyncOps_(NULL)
    , lockAsyncOpsForInitHeap_(NULL)
    , vgpusAccess_(NULL)
    , scratchAlloc_(NULL)
    , mapCacheOps_(NULL)
    , xferRead_(NULL)
    , xferWrite_(NULL)
    , vaCacheAccess_(NULL)
    , vaCacheList_(NULL)
    , mapCache_(NULL)
    , resourceCache_(NULL)
    , heapInitComplete_(false)
    , xferQueue_(NULL)
    , globalScratchBuf_(NULL)
    , srdManager_(NULL)
{
}

Device::~Device()
{
    CondLog(vaCacheList_ == NULL ||
        (vaCacheList_->size() != 0), "Application didn't unmap all host memory!");

    delete srdManager_;

    for (uint s = 0; s < scratch_.size(); ++s) {
        delete scratch_[s];
        scratch_[s] = NULL;
    }

    delete globalScratchBuf_;
    globalScratchBuf_ = NULL;

    // Destroy transfer queue
    delete xferQueue_;

    // Destroy blit program
    delete blitProgram_;

    // Release cached map targets
    for (uint i = 0; mapCache_ != NULL && i < mapCache_->size(); ++i) {
        if ((*mapCache_)[i] != NULL) {
            (*mapCache_)[i]->release();
        }
    }
    delete mapCache_;

    // Destroy temporary buffers for read/write
    delete xferRead_;
    delete xferWrite_;

    if (dummyPage_ != NULL) {
        dummyPage_->release();
    }

    // Destroy global heap
    if (heap_ != NULL) {
        delete heap_;
    }

    // Destroy resource cache
    delete resourceCache_;

    delete lockAsyncOps_;
    delete lockAsyncOpsForInitHeap_;
    delete vgpusAccess_;
    delete scratchAlloc_;
    delete mapCacheOps_;
    delete vaCacheAccess_;
    delete vaCacheList_;

    if (context_ != NULL) {
        context_->release();
    }

    // Close the active device
    close();
}

void Device::fillDeviceInfo(
    const CALdeviceattribs& calAttr,
    const CALdevicestatus& calStatus
#if cl_amd_open_video
    ,
    const CALdeviceVideoAttribs& calVideoAttr
#endif // cl_amd_open_video
    )
{
    info_.type_     = CL_DEVICE_TYPE_GPU;
    info_.vendorId_ = 0x1002;
    info_.maxComputeUnits_          = calAttr.numberOfSIMD;
    info_.maxWorkItemDimensions_    = 3;
    info_.numberOfShaderEngines     = calAttr.numberOfShaderEngines;

    if (settings().siPlus_) {
        // SI parts are scalar.  Also, reads don't need to be 128-bits to get peak rates.
        // For example, float4 is not faster than float as long as all threads fetch the same
        // amount of data and the reads are coalesced.  This is from the H/W team and confirmed
        // through experimentation.  May also be true on EG/NI, but no point in confusing
        // developers now.
        info_.nativeVectorWidthChar_    = info_.preferredVectorWidthChar_   = 4;
        info_.nativeVectorWidthShort_   = info_.preferredVectorWidthShort_  = 2;
        info_.nativeVectorWidthInt_     = info_.preferredVectorWidthInt_    = 1;
        info_.nativeVectorWidthLong_    = info_.preferredVectorWidthLong_   = 1;
        info_.nativeVectorWidthFloat_   = info_.preferredVectorWidthFloat_  = 1;
        info_.nativeVectorWidthDouble_  = info_.preferredVectorWidthDouble_ =
            (settings().checkExtension(ClKhrFp64)) ?  1 : 0;
        info_.nativeVectorWidthHalf_    = info_.preferredVectorWidthHalf_ = 0; // no half support
    }
    else {
        info_.nativeVectorWidthChar_    = info_.preferredVectorWidthChar_   = 16;
        info_.nativeVectorWidthShort_   = info_.preferredVectorWidthShort_  = 8;
        info_.nativeVectorWidthInt_     = info_.preferredVectorWidthInt_    = 4;
        info_.nativeVectorWidthLong_    = info_.preferredVectorWidthLong_   = 2;
        info_.nativeVectorWidthFloat_   = info_.preferredVectorWidthFloat_  = 4;
        info_.nativeVectorWidthDouble_  = info_.preferredVectorWidthDouble_ =
            (settings().checkExtension(ClKhrFp64)) ?  2 : 0;
        info_.nativeVectorWidthHalf_    = info_.preferredVectorWidthHalf_ = 0; // no half support
    }
    info_.maxClockFrequency_    = (calAttr.engineClock != 0) ? calAttr.engineClock : 555;
    info_.maxParameterSize_ = 1024;
     info_.minDataTypeAlignSize_ = sizeof(cl_long16);
    info_.singleFPConfig_       = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
        | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

    if (GPU_FORCE_SINGLE_FP_DENORM) {
        info_.singleFPConfig_ |= CL_FP_DENORM;
    }

    if (settings().checkExtension(ClKhrFp64)) {
        info_.doubleFPConfig_   = info_.singleFPConfig_ | CL_FP_DENORM;
    }

    if (settings().reportFMA_) {
        info_.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }

    info_.globalMemCacheLineSize_   = settings().cacheLineSize_;
    info_.globalMemCacheSize_       = settings().cacheSize_;
    if ((settings().cacheLineSize_ != 0) || (settings().cacheSize_ != 0)) {
        info_.globalMemCacheType_   = CL_READ_WRITE_CACHE;
    }
    else {
        info_.globalMemCacheType_   = CL_NONE;
    }

    if (heap()->isVirtual()) {
#if defined(ATI_OS_LINUX)
        info_.globalMemSize_   =
            (static_cast<cl_ulong>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
            // globalMemSize is the actual available size for app on Linux
            // Because Linux base driver doesn't support paging
            static_cast<cl_ulong>(calStatus.availVisibleHeap +
            calStatus.availInvisibleHeap) / 100u) * Mi;
#else
        info_.globalMemSize_   =
            (static_cast<cl_ulong>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
            static_cast<cl_ulong>(calAttr.localRAM) / 100u) * Mi;
#endif
        if (settings().apuSystem_) {
            info_.globalMemSize_   +=
                (static_cast<cl_ulong>(calAttr.uncachedRemoteRAM) * Mi);
        }

        // We try to calculate the largest available memory size from
        // the largest available block in either heap.  In theory this
        // should be the size we can actually allocate at application
        // start.  Note that it may not be a guarantee still as the
        // application progresses.
        info_.maxMemAllocSize_ = std::max(
            cl_ulong(calStatus.largestBlockVisibleHeap * Mi),
            cl_ulong(calStatus.largestBlockInvisibleHeap * Mi));

#if defined(ATI_OS_WIN)
        if (settings().apuSystem_) {
            info_.maxMemAllocSize_ = std::max(
                (static_cast<cl_ulong>(calAttr.uncachedRemoteRAM) * Mi),
                info_.maxMemAllocSize_);
        }
#endif
        info_.maxMemAllocSize_ = cl_ulong(info_.maxMemAllocSize_ *
            std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);

        //! \note Force max single allocation size.
        //! 4GB limit for the blit kernels and 64 bit optimizations.
        info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_,
                static_cast<cl_ulong>(settings().maxAllocSize_));
    }
    else {
        uint    maxHeapSize = flagIsDefault(GPU_MAX_HEAP_SIZE) ? 50 : GPU_MAX_HEAP_SIZE;
        info_.globalMemSize_   = (std::min(maxHeapSize, 100u)
            * calAttr.localRAM / 100u) * Mi;

        uint    maxAllocSize = flagIsDefault(GPU_SINGLE_ALLOC_PERCENT) ? 25 : GPU_SINGLE_ALLOC_PERCENT;
        info_.maxMemAllocSize_ = cl_ulong(info_.globalMemSize_ *
            std::min(maxAllocSize, 100u) / 100u);
    }

    if (info_.maxMemAllocSize_ < cl_ulong(128 * Mi)) {
        LogError("We are unable to get a heap large enough to support the OpenCL minimum "\
            "requirement for FULL_PROFILE");
    }

    info_.maxMemAllocSize_ = std::max(cl_ulong(128 * Mi),  info_.maxMemAllocSize_);

    // Clamp max single alloc size to the globalMemSize since it's
    // reduced by default
    info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_, info_.globalMemSize_);

    // We need to verify that we are not reporting more global memory
    // that 4x single alloc
    info_.globalMemSize_ = std::min( 4 * info_.maxMemAllocSize_, info_.globalMemSize_);

    // Use 64 bit pointers
    if (settings().use64BitPtr_) {
        info_.addressBits_  = 64;
    }
    else {
        info_.addressBits_  = 32;
        // Limit total size with 3GB for 32 bit
        info_.globalMemSize_ = std::min(info_.globalMemSize_, cl_ulong(3 * Gi));
    }

    // Alignment in BITS of the base address of any allocated memory object
    static const size_t MemBaseAlignment = 256;
    //! @note Force 256 bytes alignment, since currently
    //! calAttr.surface_alignment returns 4KB. For pinned memory runtime
    //! should be able to create a view with 256 bytes alignement
    info_.memBaseAddrAlign_ = 8 * MemBaseAlignment;

    info_.maxConstantBufferSize_ = 64 * Ki;
    info_.maxConstantArgs_       = MaxConstArguments;

    // Image support fields
    if (settings().imageSupport_) {
        info_.imageSupport_      = CL_TRUE;
        info_.maxSamplers_       = MaxSamplers;
        info_.maxReadImageArgs_  = MaxReadImage;
        info_.maxWriteImageArgs_ = MaxWriteImage;
        info_.image2DMaxWidth_   = static_cast<size_t>(getMaxTextureSize());
        info_.image2DMaxHeight_  = static_cast<size_t>(getMaxTextureSize());
        info_.image3DMaxWidth_   = std::min(2 * Ki, static_cast<size_t>(getMaxTextureSize()));
        info_.image3DMaxHeight_  = std::min(2 * Ki, static_cast<size_t>(getMaxTextureSize()));
        info_.image3DMaxDepth_   = std::min(2 * Ki, static_cast<size_t>(getMaxTextureSize()));

        info_.imagePitchAlignment_       = 256; // XXX: 256 pixel pitch alignment for now
        info_.imageBaseAddressAlignment_ = 256; // XXX: 256 byte base address alignment for now

        info_.bufferFromImageSupport_ = (heap()->isVirtual()) ? CL_TRUE : CL_FALSE;
    }

    info_.errorCorrectionSupport_    = CL_FALSE;

    if (settings().apuSystem_) {
        info_.hostUnifiedMemory_ = CL_TRUE;
    }

    info_.profilingTimerResolution_  = 1;
    info_.profilingTimerOffset_      = amd::Os::offsetToEpochNanos();
    info_.littleEndian_              = CL_TRUE;
    info_.available_                 = CL_TRUE;
    info_.compilerAvailable_         = CL_TRUE;
    info_.linkerAvailable_           = CL_TRUE;

    info_.executionCapabilities_     = CL_EXEC_KERNEL;
    info_.preferredPlatformAtomicAlignment_ = 0;
    info_.preferredGlobalAtomicAlignment_ = 0;
    info_.preferredLocalAtomicAlignment_ = 0;
    info_.queueProperties_           = CL_QUEUE_PROFILING_ENABLE;

    info_.platform_ = AMD_PLATFORM;

#if cl_amd_open_video
    // Open Video support
    // Decoder
    info_.openVideo_ = settings().openVideo_;
    info_.maxVideoSessions_ = calVideoAttr.max_decode_sessions;
    info_.numVideoAttribs_ = (calVideoAttr.data_size - 2 * sizeof(CALuint))
        / sizeof(CALvideoAttrib);
    info_.videoAttribs_ = const_cast<cl_video_attrib_amd*>(
        reinterpret_cast<const cl_video_attrib_amd*>(calVideoAttr.video_attribs));

    // Encoder
    info_.numVideoEncAttribs_ = (calVideoAttr.data_size - 2 * sizeof(CALuint))
        / sizeof(CALvideoEncAttrib);
    info_.videoEncAttribs_ = const_cast<cl_video_attrib_encode_amd*>(
        reinterpret_cast<const cl_video_attrib_encode_amd*>(calVideoAttr.video_enc_attribs));
#endif // cl_amd_open_video

    ::strcpy(info_.name_, hwInfo()->targetName_);
    ::strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");
    ::snprintf(info_.driverVersion_, sizeof(info_.driverVersion_) - 1,
         AMD_BUILD_STRING "%s", (heap()->isVirtual()) ? " (VM)": "");

    info_.profile_ = "FULL_PROFILE";
    if (settings().oclVersion_ == OpenCL20) {
        info_.version_ = "OpenCL 2.0 " AMD_PLATFORM_INFO;
        info_.oclcVersion_ = "OpenCL C 2.0 ";
        info_.spirVersions_ = "1.2";
    }
    else if (settings().oclVersion_ == OpenCL12) {
        info_.version_ = "OpenCL 1.2 " AMD_PLATFORM_INFO;
        info_.oclcVersion_ = "OpenCL C 1.2 ";
        info_.spirVersions_ = "1.2";
    }
    else {
        info_.version_ = "OpenCL 1.0 " AMD_PLATFORM_INFO;
        info_.oclcVersion_ = "OpenCL C 1.0 ";
        info_.spirVersions_ = "";
        LogError("Unknown version for support");
    }

    // Fill workgroup info size
    info_.maxWorkGroupSize_     = settings().maxWorkGroupSize_;
    info_.maxWorkItemSizes_[0]  = info_.maxWorkGroupSize_;
    info_.maxWorkItemSizes_[1]  = info_.maxWorkGroupSize_;
    info_.maxWorkItemSizes_[2]  = info_.maxWorkGroupSize_;

    if (settings().hwLDSSize_ != 0) {
        info_.localMemType_ = CL_LOCAL;
        info_.localMemSize_ = settings().hwLDSSize_;
    }
    else {
        info_.localMemType_ = CL_GLOBAL;
        info_.localMemSize_ = 16 * Ki;
    }

    info_.extensions_   = getExtensionString();

    if (settings().checkExtension(ClExtAtomicCounters32)) {
        info_.maxAtomicCounters_    = MaxAtomicCounters;
    }

    info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
    info_.deviceTopology_.pcie.bus = (calAttr.pciTopologyInformation&(0xFF<<8))>>8;
    info_.deviceTopology_.pcie.device = (calAttr.pciTopologyInformation&(0x1F<<3))>>3;
    info_.deviceTopology_.pcie.function = (calAttr.pciTopologyInformation&0x07);

    ::strncpy(info_.boardName_, calAttr.boardName, sizeof(info_.boardName_));

    // OpenCL1.2 device info fields
    info_.builtInKernels_ = "";
    info_.imageMaxBufferSize_ = MaxImageBufferSize;
    info_.imageMaxArraySize_ = MaxImageArraySize;
    info_.preferredInteropUserSync_ = true;
    info_.printfBufferSize_ = PrintfDbg::WorkitemDebugSize * info().maxWorkGroupSize_;

    if (settings().oclVersion_ >= OpenCL20) {
        info_.svmCapabilities_ =
            (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
        if (settings().svmAtomics_) {
            info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
        }
        if (settings().svmFineGrainSystem_) {
            info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
        }
        // OpenCL2.0 device info fields
        info_.maxWriteImageArgs_        = MaxReadWriteImage;    //!< For compatibility
        info_.maxReadWriteImageArgs_    = MaxReadWriteImage;

        info_.maxPipePacketSize_ = info_.maxMemAllocSize_;
        info_.maxPipeActiveReservations_ = 16;
        info_.maxPipeArgs_ = 16;

        info_.queueOnDeviceProperties_ =
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
        info_.queueOnDevicePreferredSize_ = 256 * Ki;
        info_.queueOnDeviceMaxSize_ = 512 * Ki;
        info_.maxOnDeviceQueues_ = 1;
        info_.maxOnDeviceEvents_ = settings().numDeviceEvents_;
        info_.globalVariablePreferredTotalSize_ = static_cast<size_t>(info_.globalMemSize_);
        //! \todo Remove % calculation.
        //! Use 90% of max single alloc size.
        //! Boards with max single alloc size around 4GB will fail allocations
        info_.maxGlobalVariableSize_ = static_cast<size_t>(
            amd::alignDown(info_.maxMemAllocSize_ * 9 / 10, 256));
    }

    if (settings().checkExtension(ClAmdDeviceAttributeQuery)) {
        info_.simdPerCU_            = hwInfo()->simdPerCU_;
        info_.simdWidth_            = hwInfo()->simdWidth_;
        info_.simdInstructionWidth_ = hwInfo()->simdInstructionWidth_;
        info_.wavefrontWidth_       = calAttr.wavefrontSize;
        info_.globalMemChannels_    = calAttr.memBusWidth / 32;
        info_.globalMemChannelBanks_    = calAttr.numMemBanks;
        info_.globalMemChannelBankWidth_ = hwInfo()->memChannelBankWidth_;
        info_.localMemSizePerCU_    = hwInfo()->localMemSizePerCU_;
        info_.localMemBanks_        = hwInfo()->localMemBanks_;
        info_.gfxipVersion_         = hwInfo()->gfxipVersion_;
        info_.numAsyncQueues_       = engines().numComputeRings();
        info_.threadTraceEnable_    = settings().threadTraceEnable_;
    }
}

extern const char* SchedulerSourceCode;

bool
Device::create(CALuint ordinal, CALuint numOfDevices)
{
    appProfile_.init();

    // Open GSL device
    if (!open(ordinal, appProfile_.enableHighPerformanceState(),
        appProfile_.reportAsOCL12Device() || (OPENCL_VERSION < 200))) {
        return false;
    }

    // Update CAL target
    calTarget_ = getAttribs().target;
    hwInfo_ = &DeviceInfo[calTarget_];

    // Creates device settings
    settings_ = new gpu::Settings();
    gpu::Settings* gpuSettings = reinterpret_cast<gpu::Settings*>(settings_);
    if ((gpuSettings == NULL) || !gpuSettings->create(getAttribs()
#if cl_amd_open_video
          , getVideoAttribs()
#endif // cl_amd_open_video
          , appProfile_.reportAsOCL12Device()
        )) {
        return false;
    }

    engines_.create(m_nEngines, m_engines, settings().numComputeRings_);

    amd::Context::Info  info = {0};
    std::vector<amd::Device*> devices;
    devices.push_back(this);

    // Create a dummy context
    context_ = new amd::Context(devices, info);
    if (context_ == NULL) {
        return false;
    }

    // Create the locks
    lockAsyncOps_ = new amd::Monitor("Device Async Ops Lock", true);
    if (NULL == lockAsyncOps_) {
        return false;
    }

    lockAsyncOpsForInitHeap_ = new amd::Monitor("Async Ops Lock For Initialization of Heap Resource", true);
    if (NULL == lockAsyncOpsForInitHeap_) {
        return false;
    }

    vgpusAccess_ = new amd::Monitor("Virtual GPU List Ops Lock", true);
    if (NULL == vgpusAccess_) {
        return false;
    }

    scratchAlloc_ = new amd::Monitor("Scratch Allocation Lock", true);
    if (NULL == scratchAlloc_) {
        return false;
    }

    mapCacheOps_ = new amd::Monitor("Map Cache Lock", true);
    if (NULL == mapCacheOps_) {
        return false;
    }

    vaCacheAccess_ = new amd::Monitor("VA Cache Ops Lock", true);
    if (NULL == vaCacheAccess_) {
        return false;
    }
    vaCacheList_ = new std::list<VACacheEntry*>();
    if (NULL == vaCacheList_) {
        return false;
    }

    mapCache_ = new std::vector<amd::Memory*>();
    if (mapCache_ == NULL) {
        return false;
    }
    // Use just 1 entry by default for the map cache
    mapCache_->push_back(NULL);

    size_t  resourceCacheSize = settings().resourceCacheSize_;

    // Allocate heap
    heapSize_ = settings().heapSize_;

    // Check if BE supports virtual addressing mode
    if (isVmMode()) {
        heap_ = new VirtualHeap(*this);
        gpuSettings->largeHostMemAlloc_ = (NULL != heap_) ? true : false;
    }

    // If virtual heap allocation failed, then try static allocation
    if (heap_ == NULL) {
        heap_ = new Heap(*this);
        // Disable resource cache if VM is disable
        resourceCacheSize = 0;
        if (NULL == heap_) {
            return false;
        }
    }


#ifdef DEBUG
    std::stringstream  message;
    if (settings().remoteAlloc_) {
        message << "Using *Remote* memory";
    }
    else {
        message << "Using *Local* memory";
    }
    if (!heap()->isVirtual()) {
        message << ": " << settings().heapSize_ / Mi << "MB, growth: " <<  \
            settings().heapSizeGrowth_ / Mi << "MB";
    }
    message << std::endl;
    LogInfo(message.str().c_str());
#endif // DEBUG

    // Create resource cache.
    // \note Cache must be created before any resource creation to avoid NULL check
    resourceCache_ = new ResourceCache(resourceCacheSize);
    if (NULL == resourceCache_) {
        return false;
    }

    // Fill the device info structure
    fillDeviceInfo(getAttribs(), getStatus()
#if cl_amd_open_video
        , getVideoAttribs()
#endif //cl_amd_open_video
    );

    if (settings().hsail_ || (settings().oclVersion_ == OpenCL20)) {
        if (NULL == hsaCompiler_) {
            const char* library = getenv("HSA_COMPILER_LIBRARY");
            aclCompilerOptions opts = {
                sizeof(aclCompilerOptions_0_8),
                library,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                AMD_OCL_SC_LIB,
                &::malloc,
                &::free
            };
            // Initialize the compiler handle
            acl_error   error;
            hsaCompiler_ = aclCompilerInit(&opts, &error);
            if (error != ACL_SUCCESS) {
                 LogError("Error initializing the compiler");
                 return false;
            }
        }
    }
    else {
        blitProgram_ = new BlitProgram(context_);
        // Create blit programs
        if (blitProgram_ == NULL || !blitProgram_->create(this)) {
            delete blitProgram_;
            blitProgram_ = NULL;
            LogError("Couldn't create blit kernels!");
            return false;
        }
    }

    // Allocate SRD manager
    srdManager_ = new SrdManager(*this,
        std::max(HsaImageObjectSize, HsaSamplerObjectSize), 64 * Ki);
    if (srdManager_ == NULL) {
        return false;
    }

    return true;
}

bool
Device::initializeHeapResources()
{
    amd::ScopedLock k(lockAsyncOpsForInitHeap_);
    if (!heapInitComplete_) {
        heapInitComplete_ = true;

        PerformFullInitialization();

        uint numComputeRings = engines_.numComputeRings();
        scratch_.resize((settings().useSingleScratch_) ? 1 : (numComputeRings ? numComputeRings : 1));

        // Initialize the number of mem object for the scratch buffer
        for (uint s = 0; s < scratch_.size(); ++s) {
            scratch_[s] = new ScratchBuffer((settings().siPlus_) ? 1 : info_.numberOfShaderEngines);
            if (NULL == scratch_[s]) {
                return false;
            }
        }

        // Complete initialization of the heap and other buffers
        if ((heap_ == NULL) || !heap_->create(heapSize_, settings().remoteAlloc_)) {
            LogError("Failed GPU heap creation");
            return false;
        }

        size_t dummySize = amd::Os::pageSize();

        // Allocate a dummy page for NULL pointer processing
        dummyPage_ = new(*context_) amd::Buffer(*context_, 0, dummySize);
        if ((dummyPage_ != NULL) && !dummyPage_->create()) {
            dummyPage_->release();
            return false;
        }

        Memory* devMemory = reinterpret_cast<Memory*>(dummyPage_->getDeviceMemory(*this));
        if (devMemory == NULL) {
            // Release memory
            dummyPage_->release();
            dummyPage_ = NULL;
            return false;
        }

        if (settings().stagedXferSize_ != 0) {
            // Initialize staged write buffers
            if (settings().stagedXferWrite_) {
                Resource::MemoryType type;
                if (settings().stagingWritePersistent_ && !settings().disablePersistent_) {
                    type = Resource::Persistent;
                } else {
                    type = Resource::RemoteUSWC;
                }
                xferWrite_ = new XferBuffers(*this, type,
                    amd::alignUp(settings().stagedXferSize_, heap()->granularityB()));
                if ((xferWrite_ == NULL) || !xferWrite_->create()) {
                    LogError("Couldn't allocate transfer buffer objects for read");
                    return false;
                }
            }

            // Initialize staged read buffers
            if (settings().stagedXferRead_) {
                xferRead_ = new XferBuffers(*this, Resource::Remote,
                    amd::alignUp(settings().stagedXferSize_, heap()->granularityB()));
                if ((xferRead_ == NULL) || !xferRead_->create()) {
                    LogError("Couldn't allocate transfer buffer objects for write");
                    return false;
                }
            }
        }

        // Delay compilation due to brig_loader memory allocation
        if (settings().hsail_ || (settings().oclVersion_ == OpenCL20)) {
            const char* scheduler = NULL;
            const char* ocl20 = NULL;
            if (settings().oclVersion_ == OpenCL20) {
                scheduler = SchedulerSourceCode;
                ocl20 = "-cl-std=CL2.0";
            }
            blitProgram_ = new BlitProgram(context_);
            // Create blit programs
            if (blitProgram_ == NULL ||
                !blitProgram_->create(this, scheduler, ocl20)) {
                delete blitProgram_;
                blitProgram_ = NULL;
                LogError("Couldn't create blit kernels!");
                return false;
            }
        }

        // Create a synchronized transfer queue
        xferQueue_ = new VirtualGPU(*this);
        if (!(xferQueue_ && xferQueue_->create(
            false,
    #if cl_amd_open_video
            NULL
    #endif // cl_amd_open_video
            ))) {
            delete xferQueue_;
            xferQueue_ = NULL;
        }
        if (NULL == xferQueue_) {
            LogError("Couldn't create the device transfer manager!");
            return false;
        }
        xferQueue_->enableSyncedBlit();
    }
    return true;
}

device::VirtualDevice*
Device::createVirtualDevice(
    bool    profiling,
    bool    interopQueue
#if cl_amd_open_video
    , void* calVideoProperties
#endif // cl_amd_open_video
    , uint  deviceQueueSize
    )
{
    // Not safe to add a queue. So lock the device
    amd::ScopedLock k(lockAsyncOps());
    amd::ScopedLock lock(vgpusAccess());

    // Initialization of heap and other resources occur during the command queue creation time.
    if (!initializeHeapResources()) {
        return NULL;
    }

    VirtualGPU* vgpu = new VirtualGPU(*this);
    if (vgpu && vgpu->create(
        profiling
#if cl_amd_open_video
        , calVideoProperties
#endif // cl_amd_open_video
        , deviceQueueSize
        )) {
        return vgpu;
    } else {
        delete vgpu;
        return NULL;
    }
}

bool
Device::reallocHeap(size_t size, bool remoteAlloc)
{
    size_t  heapSize    =  heapSize_ + ((size != 0) ?
        amd::alignUp(size, settings().heapSizeGrowth_) : 0);
    Heap*   oldHeap     = heap_;
    // Maximum heap limit size = reported size + internal memory
    size_t  maxHeapLimit = static_cast<size_t>(info().globalMemSize_) +
        // an extra 10MB for the alignments of allocations,
        // since the conformance test doesn't expect any
        10 * Mi;

    if ((settings().heapSizeGrowth_ == 0) ||
        // Allow the heap growth up to the global memory limit
        (heapSize_ + size > maxHeapLimit)) {
        return false;
    }
    heapSize = std::min(maxHeapLimit, heapSize);

    heap_ = new Heap(*this);

    // Make sure we have allocated a new global heap
    if (NULL == heap_) {
        heap_ = oldHeap;
        return false;
    }

    if (!heap_->create(heapSize, remoteAlloc)) {
        delete heap_;
        heap_ = oldHeap;
        return false;
    }

    // Copy the old heap to the new one
    if (!oldHeap->copyTo(heap_)) {
        delete heap_;
        heap_ = oldHeap;
        return false;
    }

    delete oldHeap;
    heapSize_ = heapSize;

    return true;
}

device::Program*
Device::createProgram(int oclVer)
{
    device::Program* gpuProgram;
    if (settings().hsail_ || (oclVer == 200)) {
        gpuProgram = new HSAILProgram(*this);
    }
    else {
        gpuProgram = new Program(*this);
    }
    if (gpuProgram == NULL) {
        LogError("We failed memory allocation for program!");
    }

    return gpuProgram;
}

//! Requested devices list as configured by the GPU_DEVICE_ORDINAL
typedef std::map<int, bool> requestedDevices_t;

//! Parses the requested list of devices to be exposed to the user.
static void
parseRequestedDeviceList(requestedDevices_t &requestedDevices) {
    char *pch = NULL;
    int requestedDeviceCount = 0;
    const char* requestedDeviceList = GPU_DEVICE_ORDINAL;

    pch = strtok(const_cast<char*>(requestedDeviceList), ",");
    while (pch != NULL) {
        bool deviceIdValid = true;
        int currentDeviceIndex = atoi(pch);
        // Validate device index.
        for (size_t i = 0; i < strlen(pch); i++) {
            if (!isdigit(pch[i])) {
                deviceIdValid = false;
                break;
            }
        }
        if (currentDeviceIndex < 0) {
            deviceIdValid = false;
        }
        // Get next token.
        pch = strtok(NULL, ",");
        if (!deviceIdValid) {
            continue;
        }

        // Requested device is valid.
        requestedDevices[currentDeviceIndex] = true;
    }
}

#if defined(_WIN32) && defined (DEBUG)
#include <cstdio>
#include <crtdbg.h>
static int reportHook(int reportType, char *message, int *returnValue)
{
    fprintf(stderr, "%s", message);
    ::exit(3);
    return 1;
}
#endif // _WIN32 & DEBUG

bool
Device::init()
{
    CALuint     numDevices = 0;
    bool        result = false;
    bool    useDeviceList = false;
    requestedDevices_t requestedDevices;

    const char *library = getenv("COMPILER_LIBRARY");
    aclCompilerOptions opts = {
        sizeof(aclCompilerOptions_0_8),
        library ? library : LINUX_ONLY("lib") "amdocl12cl" \
            LP64_SWITCH(LINUX_SWITCH("32",""),"64") LINUX_SWITCH(".so",".dll"),
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        AMD_OCL_SC_LIB,
        &::malloc,
        &::free
    };

    hsaCompiler_ = NULL;
    compiler_ = aclCompilerInit(&opts, NULL);

#if defined(_WIN32) && !defined(_WIN64)
    // @toto: FIXME: remove this when CAL is fixed!!!
    unsigned int old, ignored;
    _controlfp_s(&old, 0, 0);
#endif // _WIN32 && !_WIN64
    // FIXME_lmoriche: needs cleanup
    osInit();
#if defined(_WIN32)
    //osAssertSetStyle(OSASSERT_STYLE_LOGANDEXIT);
#endif // WIN32

#if defined(_WIN32) && defined (DEBUG)
    if (::getenv("AMD_OCL_SUPPRESS_MESSAGE_BOX"))
    {
        _CrtSetReportHook(reportHook);
        _set_error_mode(_OUT_TO_STDERR);
   }
#endif // _WIN32 & DEBUG

    calInit();

#if defined(_WIN32) && !defined(_WIN64)
    _controlfp_s(&ignored, old, _MCW_RC | _MCW_PC);
#endif // _WIN32 && !_WIN64

    // Get the total number of active devices
    // Count up all the devices in the system.
    numDevices = calGetDeviceCount();

    CALuint ordinal = 0;
    const char* selectDeviceByName = NULL;
    if (!flagIsDefault(GPU_DEVICE_ORDINAL)) {
        useDeviceList = true;
        parseRequestedDeviceList(requestedDevices);
    }
    else if (!flagIsDefault(GPU_DEVICE_NAME)) {
        selectDeviceByName = GPU_DEVICE_NAME;
    }

    // Loop through all active devices and initialize the device info structure
    for (; ordinal < numDevices; ++ordinal) {
        // Create the GPU device object
        Device *d = new Device();
        result = (NULL != d) && d->create(ordinal, numDevices);
        if (useDeviceList) {
            result &= (requestedDevices.find(ordinal) != requestedDevices.end());
        }
        if (result &&
            ((NULL == selectDeviceByName) || ('\0' == selectDeviceByName[0]) ||
             (strstr(selectDeviceByName, d->info().name_) != NULL))) {
            d->registerDevice();
        }
        else {
            delete d;
        }
    }
    return result;
}

void
Device::tearDown()
{
    osExit();
    calShutdown();
    aclCompilerFini(compiler_);
    if (hsaCompiler_ != NULL) {
        aclCompilerFini(hsaCompiler_);
    }
}

//! @note This funciton must be lock protected from a caller
HeapBlock*
Device::allocHeapBlock(size_t size) const
{
    HeapBlock* hb = NULL;

    // Allocate the underlying heap block
    hb = heap_->alloc(size);

    // Virtual heap should never fail allocation
    if ((hb == NULL) && (!heap_->isVirtual())) {
        // Queues can't process commands,
        // while the global heap reallocation occurs.
        // So stall all queues and then reallocate the global heap
        ScopedLockVgpus lock(*this);

        // Wait for idle
        for (uint idx = 0; idx < vgpus().size(); ++idx) {
            vgpus()[idx]->waitAllEngines();
        }

        // Acount memory alignment for the new allocation
        size_t  extraSpace = heap_->granularityB();
        if (size >= heap_->freeSpace()) {
            // Required extra space = requested size - free space
            extraSpace += size - heap_->freeSpace();
        }

        //! @note the const cast here looks bad, but the device object
        //  is a lock protected above. The rest of the code
        //  doesn't change the device object.
        //  So the const methods can be safly used everywhere else.
        //  In general we should avoid changing the device object after initialization

        // Try to reallocate the heap with the same memory type
        if (const_cast<Device*>(this)->reallocHeap(extraSpace, settings().remoteAlloc_)) {
            hb = heap_->alloc(size);
        }

        if (hb == NULL) {
            // Use reversed memory type as a temporary storage
            bool    remoteAlloc = settings().remoteAlloc_ ^ true;

            // Try to reallocate the heap
            if (const_cast<Device*>(this)->reallocHeap(extraSpace, remoteAlloc)) {
                // Back to the default location of the global heap
                remoteAlloc ^= true;
                if (!const_cast<Device*>(this)->reallocHeap(0, remoteAlloc)) {
                    LogWarning("New memory type for the \
                        global heap after reallocation!");
                }
                hb = heap_->alloc(size);
            }
        }
    }

    return hb;
}

gpu::Memory*
Device::getGpuMemory(amd::Memory* mem) const
{
    return static_cast<gpu::Memory*>(mem->getDeviceMemory(*this));
}


CalFormat
Device::getCalFormat(const amd::Image::Format& format) const
{
    // Find CAL format
    for (uint i = 0; i < sizeof(MemoryFormatMap) / sizeof(MemoryFormat); ++i) {
        if ((format.image_channel_data_type ==
             MemoryFormatMap[i].clFormat_.image_channel_data_type) &&
            (format.image_channel_order ==
             MemoryFormatMap[i].clFormat_.image_channel_order)) {
            return MemoryFormatMap[i].calFormat_;
        }
    }
    osAssert(0 && "We didn't find CAL resource format!");
    return MemoryFormatMap[0].calFormat_;
}

amd::Image::Format
Device::getOclFormat(const CalFormat& format) const
{
    // Find CL format
    for (uint i = 0; i < sizeof(MemoryFormatMap) / sizeof(MemoryFormat); ++i) {
        if ((format.type_ ==
             MemoryFormatMap[i].calFormat_.type_) &&
            (format.channelOrder_ ==
             MemoryFormatMap[i].calFormat_.channelOrder_)) {
            return MemoryFormatMap[i].clFormat_;
        }
    }
    osAssert(0 && "We didn't find OCL resource format!");
    return MemoryFormatMap[0].clFormat_;
}

// Create buffer without an owner (merge common code with createBuffer() ?)
gpu::Memory*
Device::createScratchBuffer(size_t size) const
{
    Memory* gpuMemory = NULL;

    // Use virtual heap allocation
    if (heap()->isVirtual()) {
        // Create a memory object
        gpuMemory = new gpu::Memory(*this, size);
        if (NULL == gpuMemory || !gpuMemory->create(Resource::Local)) {
            delete gpuMemory;
            gpuMemory = NULL;
        }
    }
    else {
        // We have to lock the heap block allocation,
        // so possible reallocation won't occur twice or
        // another thread could destroy a heap block,
        // while we didn't finish allocation
        amd::ScopedLock k(lockAsyncOps());

        HeapBlock* hb = allocHeapBlock(size);
        if (hb != NULL) {
            // wrap it
            gpuMemory = new gpu::Memory(*this, *hb);

            // Create resource
            if (NULL != gpuMemory) {
                Resource::ViewParams   params;
                params.offset_  = hb->offset_;
                params.size_    = hb->size_;
                params.resource_ = &(globalMem());
                params.memory_  = NULL;
                if (!gpuMemory->create(Resource::View, &params)) {
                    delete gpuMemory;
                    gpuMemory = NULL;
                }
            }
        }
    }

    return gpuMemory;
}

gpu::Memory*
Device::createBufferFromHeap(amd::Memory& owner) const
{
    size_t  size = owner.getSize();
    gpu::Memory* gpuMemory;

    // We have to lock the heap block allocation,
    // so possible reallocation won't occur twice or
    // another thread could destroy a heap block,
    // while we didn't finish allocation
    amd::ScopedLock k(lockAsyncOps());

    HeapBlock* hb = allocHeapBlock(size);
    if (hb == NULL) {
        LogError("We don't have enough video memory!");
        return NULL;
    }

    // Create a memory object
    gpuMemory = new gpu::Memory(*this, owner, hb);
    if (NULL == gpuMemory) {
        hb->setMemory(NULL);
        hb->free();
        return NULL;
    }

    Resource::ViewParams params;
    params.owner_       = &owner;
    params.offset_      = hb->offset_;
    params.size_        = hb->size_;
    params.resource_    = &(globalMem());
    params.memory_      = NULL;

    if (!gpuMemory->create(Resource::View, &params)) {
        delete gpuMemory;
        return NULL;
    }

    // Check if owner is interop memory
    if (owner.isInterop()) {
        if (!gpuMemory->createInterop(Memory::InteropHwEmulation)) {
            LogError("HW interop creation failed!");
            delete gpuMemory;
            return NULL;
        }
    }
    return gpuMemory;
}

gpu::Memory*
Device::createBuffer(
    amd::Memory&    owner,
    bool            directAccess,
    bool            bufferAlloc) const
{
    size_t  size = owner.getSize();
    gpu::Memory* gpuMemory;

    // Create resource
    bool result = false;

    if (owner.getType() == CL_MEM_OBJECT_PIPE) {
        // directAccess isnt needed as Pipes shouldnt be host accessible for GPU
        directAccess = false;
    }

    if (NULL != owner.parent()) {
        gpu::Memory*    gpuParent = getGpuMemory(owner.parent());
        if (NULL == gpuParent) {
            LogError("Can't get the owner object for subbuffer allocation");
            return NULL;
        }

        if (!heap()->isVirtual()) {
            bool    uhpAlloc =
                (owner.parent()->getMemFlags() & CL_MEM_USE_HOST_PTR) ? true : false;

            if (owner.parent()->getType() != CL_MEM_OBJECT_IMAGE1D_BUFFER) {
                //! \note This extra line is necessary to make sure that subbuffer
                //! allocation is a synch operation,
                //! due to a possible realloc of heap(no VM) or parent(UHP)
                amd::ScopedLock k(lockAsyncOps());

                //! @note: For now make sure the parent is allocated in the global heap
                //! or if it's the UHP optimization for prepinned memory
                if (((gpuParent->hb() == NULL) || uhpAlloc) &&
                    !owner.parent()->reallocedDeviceMemory(this)) {
                    if (reallocMemory(*owner.parent())) {
                        gpuParent = getGpuMemory(owner.parent());
                    }
                    else {
                        LogError("Can't reallocate the owner object for subbuffer allocation");
                        return NULL;
                    }
                }

                return gpuParent->createBufferView(owner);
            }
            else {
                gpuParent = getGpuMemory(owner.parent()->parent());
                return gpuParent->createBufferView(*owner.parent()->parent());
            }
        }
        else {
            return gpuParent->createBufferView(owner);
        }
    }

    Resource::MemoryType    type = (owner.forceSysMemAlloc() || (owner.getMemFlags() & CL_MEM_SVM_FINE_GRAIN_BUFFER)) ?
        Resource::Remote : Resource::Local;

    if (owner.getMemFlags() & CL_MEM_BUS_ADDRESSABLE_AMD) {
        type = Resource::BusAddressable;
    }
    else if (owner.getMemFlags() & CL_MEM_EXTERNAL_PHYSICAL_AMD) {
        type = Resource::ExternalPhysical;
    }

    // Use direct access if it's possible
    if (bufferAlloc || (type == Resource::Remote)) {
        bool    forceHeapAlloc = false;
        bool    remoteAlloc = false;
        // Internal means VirtualDevice!=NULL
        bool    internalAlloc = ((owner.getMemFlags() & CL_MEM_USE_HOST_PTR) &&
              (owner.getVirtualDevice() != NULL)) ? true : false;

        // Create a memory object
        gpuMemory = new gpu::Buffer(*this, owner, owner.getSize());
        if (NULL == gpuMemory) {
            return NULL;
        }

        // Check if owner is interop memory
        if (owner.isInterop()) {
            result = gpuMemory->createInterop(Memory::InteropDirectAccess);
        }
        else if (owner.getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
            // Attempt to allocate from persistent heap
            result = gpuMemory->create(Resource::Persistent);
        }
        else if (directAccess || (type == Resource::Remote)) {
            // Check for system memory allocations
            if ((owner.getMemFlags() & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_HOST_PTR))
                || (settings().remoteAlloc_)) {
                // Allocate remote memory if AHP allocation and context has just 1 device
                if ((owner.getMemFlags() & CL_MEM_ALLOC_HOST_PTR) &&
                    (owner.getContext().devices().size() == 1)) {
                    if (owner.getMemFlags() & (CL_MEM_READ_ONLY |
                        CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
                        // GPU will be reading from this host memory buffer,
                        // so assume Host write into it
                        type = Resource::RemoteUSWC;
                        remoteAlloc = true;
                    }
                }
                // Make sure owner has a valid hostmem pointer and it's not COPY
                if (!remoteAlloc && (owner.getHostMem() != NULL)) {
                    Resource::PinnedParams params;
                    params.owner_ = &owner;
                    params.gpu_ =
                        reinterpret_cast<VirtualGPU*>(owner.getVirtualDevice());

                    params.hostMemRef_  = owner.getHostMemRef();
                    params.size_        = owner.getHostMemRef()->size();
                    if (0 == params.size_) {
                        params.size_ = owner.getSize();
                    }
                    // Create memory object
                    result = gpuMemory->create(Resource::Pinned, &params);

                    // If direct access failed
                    if (!result) {
                        // and VM off, then force a heap allocation
                        if (!heap()->isVirtual()) {
                            // Internal pinning doesn't need a heap allocation
                            if (!internalAlloc) {
                                forceHeapAlloc = true;
                            }
                        }
                        // Don't use cached allocation
                        // if size is biger than max single alloc
                        if (owner.getSize() > info().maxMemAllocSize_) {
                            delete gpuMemory;
                            return NULL;
                        }
                    }
                }
            }
        }

        if (!result && !forceHeapAlloc &&
            // Make sure it's not internal alloc
            !internalAlloc) {
            Resource::CreateParams  params;
            params.owner_ = &owner;

            // Create memory object
            result = gpuMemory->create(type, &params);

            // If allocation was successful
            if (result) {
                // Initialize if the memory is a pipe object
                if (owner.getType() == CL_MEM_OBJECT_PIPE) {
                    // Pipe initialize in order read_idx, write_idx, end_idx. Refer clk_pipe_t structure.
                    // Init with 3 DWORDS for 32bit addressing and 6 DWORDS for 64bit
                    size_t pipeInit[3] = {0 , 0, owner.asPipe()->getMaxNumPackets()};
                    gpuMemory->writeRawData(*xferQueue_, sizeof(pipeInit), pipeInit, true);
                }
                // If memory has direct access from host, then get CPU address
                if (gpuMemory->isHostMemDirectAccess() &&
                   (type != Resource::ExternalPhysical)) {
                    void* address = gpuMemory->map(NULL);
                    if (address != NULL) {
                        // Copy saved memory
                        if (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) {
                            memcpy(address, owner.getHostMem(), owner.getSize());
                        }
                        // It should be safe to change the host memory pointer,
                        // because it's lock protected from the upper caller
                        owner.setHostMem(address);
                    }
                    else {
                        result = false;
                    }
                }
                // An optimization for CHP. Copy memory and destroy sysmem allocation
                else if ((gpuMemory->memoryType() != Resource::Pinned) &&
                         (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
                         (owner.getContext().devices().size() == 1)) {
                    amd::Coord3D    origin(0, 0, 0);
                    amd::Coord3D    region(owner.getSize());
                    static const bool Entire  = true;
                    if (xferMgr().writeBuffer(owner.getHostMem(),
                        *gpuMemory, origin, region, Entire)) {
                        // Clear CHP memory
                        owner.setHostMem(NULL);
                    }
                }
            }
        }

        if (!result && !forceHeapAlloc) {
            delete gpuMemory;
            return NULL;
        }
    }

    if (!result) {
        assert(!heap()->isVirtual() && "Can't have static heap allocation with VM");
        gpuMemory = createBufferFromHeap(owner);
    }

    return gpuMemory;
}

gpu::Memory*
Device::createImage(amd::Memory& owner, bool directAccess) const
{
    size_t  size = owner.getSize();
    amd::Image& image = *owner.asImage();
    gpu::Memory* gpuImage = NULL;
    CalFormat   format = getCalFormat(image.getImageFormat());

    if ((NULL != owner.parent()) && (owner.parent()->asImage() != NULL)) {
        device::Memory* devParent = owner.parent()->getDeviceMemory(*this);
        if (NULL == devParent) {
            LogError("Can't get the owner object for image view allocation");
            return NULL;
        }
        // Create a view on the specified device
        gpuImage = (gpu::Memory*)createView(owner, *devParent);
        if (heap()->isVirtual() && (NULL != gpuImage) && (gpuImage->owner() != NULL)) {
            gpuImage->owner()->setHostMem((address)(owner.parent()->getHostMem()) + gpuImage->owner()->getOrigin());
        }
        return gpuImage ;
    }

    gpuImage = new gpu::Image(*this, owner,
        image.getWidth(),
        image.getHeight(),
        image.getDepth(),
        format.type_,
        format.channelOrder_,
        image.getType(),
        image.getMipLevels());

    // Create resource
    if (NULL != gpuImage) {
        const bool imageBuffer =
            ((owner.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
             ((owner.getType() == CL_MEM_OBJECT_IMAGE2D) &&
              (owner.parent() != NULL) &&
              (owner.parent()->asBuffer() != NULL)));
        bool result = false;

        // Check if owner is interop memory
        if (owner.isInterop()) {
            result = gpuImage->createInterop(Memory::InteropDirectAccess);
        }
        else if (imageBuffer) {
            Resource::ImageBufferParams  params;
            gpu::Memory* buffer = reinterpret_cast<gpu::Memory*>
                (image.parent()->getDeviceMemory(*this));
            if (buffer == NULL) {
                LogError("Buffer creation for ImageBuffer failed!");
                delete gpuImage;
                return NULL;
            }
            params.owner_       = &owner;
            params.resource_    = buffer;
            params.memory_      = buffer;

            // Create memory object
            result = gpuImage->create(Resource::ImageBuffer, &params);
        }
        else if (directAccess && (owner.getMemFlags() & CL_MEM_ALLOC_HOST_PTR)) {
            Resource::PinnedParams  params;
            params.owner_       = &owner;
            params.hostMemRef_  = owner.getHostMemRef();
            params.size_        = owner.getHostMemRef()->size();

            // Create memory object
            result = gpuImage->create(Resource::Pinned, &params);
        }

        if (!result && !owner.isInterop()) {
            if (owner.getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
                // Attempt to allocate from persistent heap
                result = gpuImage->create(Resource::Persistent);
            }
            else {
                Resource::MemoryType    type = (owner.forceSysMemAlloc()) ?
                    Resource::RemoteUSWC : Resource::Local;
                // Create memory object
                result = gpuImage->create(type);
            }
        }

        if (!result) {
            delete gpuImage;
            return NULL;
        }
        else if ((gpuImage->memoryType() != Resource::Pinned) &&
                 (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
                 (owner.getContext().devices().size() == 1)) {
            // Ignore copy for image1D_buffer, since it was already done for buffer
            if (heap()->isVirtual() && imageBuffer) {
                // Clear CHP memory
                owner.setHostMem(NULL);
            }
            else if (!imageBuffer) {
                amd::Coord3D    origin(0, 0, 0);
                static const bool Entire  = true;
                if (xferMgr().writeImage(owner.getHostMem(),
                    *gpuImage, origin, image.getRegion(), 0, 0, Entire)) {
                    // Clear CHP memory
                    owner.setHostMem(NULL);
                }
            }
        }

        if (result) {
            gslMemObject temp = gpuImage->gslResource();
            size_t bytePitch = gpuImage->elementSize() * temp->getPitch();
            image.setBytePitch(bytePitch);
        }
    }

    return gpuImage;
}

//! Allocates cache memory on the card
device::Memory*
Device::createMemory(
    amd::Memory&    owner) const
{
    bool directAccess   = false;
    bool bufferAlloc    = false;
    gpu::Memory* memory = NULL;

    if (heap()->isVirtual()) {
        bufferAlloc = true;
    }
    //!@todo Remove this code when VM is always on.
    // Use zero-copy transfers for sysmem allocations or persistent memory
    else {
        if (owner.getMemFlags() & (CL_MEM_ALLOC_HOST_PTR |
                                   CL_MEM_USE_HOST_PTR)) {
            bufferAlloc = true;
        }
    }

    if (owner.asBuffer()) {
        directAccess = (settings().hostMemDirectAccess_ & Settings::HostMemBuffer)
            ? true : false;
        memory = createBuffer(owner, directAccess, bufferAlloc);
    }
    else if (owner.asImage()) {
        directAccess = (settings().hostMemDirectAccess_ & Settings::HostMemImage)
            ? true : false;
        memory = createImage(owner, directAccess);
    }
    else {
        LogError("Unknown memory type!");
    }

    // Attempt to pin system memory if runtime didn't use direct access
    if ((memory != NULL) &&
        (memory->memoryType() != Resource::Pinned) &&
        (memory->memoryType() != Resource::Remote) &&
        (memory->memoryType() != Resource::RemoteUSWC) &&
        (memory->memoryType() != Resource::ExternalPhysical) &&
        ((owner.getHostMem() != NULL) ||
         ((NULL != owner.parent()) && (owner.getHostMem() != NULL)))) {
        bool ok = memory->pinSystemMemory(
            owner.getHostMem(), (owner.getHostMemRef()->size()) ?
                owner.getHostMemRef()->size() : owner.getSize());
        //! \note: Ignore the pinning result for now
    }

    return memory;
}

bool
Device::createSampler(const amd::Sampler& owner, device::Sampler** sampler) const
{
    *sampler = NULL;
    if (settings().hsail_ || (settings().oclVersion_ >= OpenCL20)) {
        Sampler* gpuSampler = new Sampler(*this);
        if ((NULL == gpuSampler) || !gpuSampler->create(owner.state())) {
            delete gpuSampler;
            return false;
        }
        *sampler = gpuSampler;
    }
    return true;
}

//! \note reallocMemory() must be called only from outside of
//! VirtualGPU submit commands methods.
//! Otherwise a deadlock in lockVgpus() is possible

bool
Device::reallocMemory(amd::Memory& owner) const
{
    bool directAccess   = false;
    bool bufferAlloc    = heap()->isVirtual();

    // For now we have to serialize reallocation code
    amd::ScopedLock lk(*lockAsyncOps_);

    // Read device memory after the lock,
    // since realloc from another thread can replace the pointer
    gpu::Memory*  gpuMemory = getGpuMemory(&owner);
    if (gpuMemory == NULL) {
        return false;
    }
    if (gpuMemory->hb() != NULL) {
        return true;
    }

    if (bufferAlloc) {
        if (gpuMemory->pinOffset() == 0) {
            return true;
        }
        else if (NULL != owner.parent()) {
            if (!reallocMemory(*owner.parent())) {
                return false;
            }
        }
    }

    if (owner.asBuffer()) {
        // Disable remote allocation if no VM
        if ((gpuMemory != NULL) &&
            ((gpuMemory->memoryType() == Resource::Remote) ||
             (gpuMemory->memoryType() == Resource::RemoteUSWC)) && !bufferAlloc) {
            // Make sure we don't have a stale memory in VA cache before reallocation
            // of system memory.
            // \note: the app must unmap() memory before kernel launch
            removeVACache(gpuMemory);
            static const bool forceAllocHostMem = true;
            static const bool forceCopy = true;
            owner.allocHostMemory(owner.getHostMem(), forceAllocHostMem, forceCopy);
        }
        gpuMemory = createBuffer(owner, directAccess, bufferAlloc);
    }
    else if (owner.asImage()) {
        return true;
    }
    else {
        LogError("Unknown memory type!");
    }

    if (gpuMemory != NULL) {
        gpu::Memory* newMemory = gpuMemory;
        gpu::Memory* oldMemory = getGpuMemory(&owner);

        // Transfer the object
        if (oldMemory != NULL) {
            if (!oldMemory->moveTo(*newMemory)) {
                delete newMemory;
                return false;
            }
        }

        // Attempt to pin system memory
        if ((newMemory->memoryType() != Resource::Pinned) &&
            ((owner.getHostMem() != NULL) ||
             ((NULL != owner.parent()) && (owner.getHostMem() != NULL)))) {
            bool ok = newMemory->pinSystemMemory(
                owner.getHostMem(), (owner.getHostMemRef()->size()) ?
                owner.getHostMemRef()->size() : owner.getSize());
            //! \note: Ignore the pinning result for now
        }

        return true;
    }

    return false;
}

device::Memory*
Device::createView(amd::Memory& owner, const device::Memory& parent) const
{
    size_t  size = owner.getSize();
    assert((owner.asImage() != NULL) && "View supports images only");
    const amd::Image& image = *owner.asImage();
    gpu::Memory* gpuImage = NULL;
    CalFormat   format = getCalFormat(image.getImageFormat());

    gpuImage = new gpu::Image(*this, owner,
        image.getWidth(),
        image.getHeight(),
        image.getDepth(),
        format.type_,
        format.channelOrder_,
        image.getType(),
        image.getMipLevels());

    // Create resource
    if (NULL != gpuImage) {
        bool result = false;
        Resource::ImageViewParams   params;
        const gpu::Memory& gpuMem = static_cast<const gpu::Memory&>(parent);

        params.owner_       = &owner;
        params.level_       = image.getBaseMipLevel();
        params.layer_       = 0;
        params.resource_    = &gpuMem;
        params.gpu_ = reinterpret_cast<VirtualGPU*>(owner.getVirtualDevice());
        params.memory_      = &gpuMem;

        // Create memory object
        result = gpuImage->create(Resource::ImageView, &params);
        if (!result) {
            delete gpuImage;
            return NULL;
        }
    }

    return gpuImage;
}


//! Attempt to bind with external graphics API's device/context
bool
Device::bindExternalDevice(
    intptr_t type, void* pDevice, void* pContext, bool validateOnly)
{
    assert(pDevice);

    switch (type) {
#ifdef _WIN32
    case CL_CONTEXT_D3D10_DEVICE_KHR:
        // There is no need to perform full initialization here
        // if the GSLDevice is still uninitialized.
        // Only adapter initialization is required
        // to validate D3D10 interoperability.
        PerformAdapterInitialization();

        // Associate GSL-D3D
        if (!associateD3D10Device(
            reinterpret_cast<ID3D10Device*>(pDevice))) {
            LogError("Failed gslD3D10Associate()");
            return false;
        }
        break;
    case CL_CONTEXT_D3D11_DEVICE_KHR:
        // There is no need to perform full initialization here
        // if the GSLDevice is still uninitialized.
        // Only adapter initialization is required to validate
        // D3D11 interoperability.
        PerformAdapterInitialization();

        // Associate GSL-D3D
        if (!associateD3D11Device(
            reinterpret_cast<ID3D11Device*>(pDevice))) {
            LogError("Failed gslD3D11Associate()");
            return false;
        }
        break;
    case CL_CONTEXT_ADAPTER_D3D9_KHR:
        PerformAdapterInitialization();

        // Associate GSL-D3D
        if (!associateD3D9Device(
            reinterpret_cast<IDirect3DDevice9*>(pDevice))) {
            LogWarning("D3D9<->OpenCL adapter mismatch or D3D9Associate() failure");
            return false;
        }
        break;
    case CL_CONTEXT_ADAPTER_D3D9EX_KHR:
        PerformAdapterInitialization();

        // Associate GSL-D3D
        if (!associateD3D9Device(
            reinterpret_cast<IDirect3DDevice9Ex*>(pDevice))) {
            LogWarning("D3D9<->OpenCL adapter mismatch or D3D9Associate() failure");
            return false;
        }
        break;
    case CL_CONTEXT_ADAPTER_DXVA_KHR:
        break;
#endif //_WIN32
    case CL_GL_CONTEXT_KHR:
    {

        // There is no need to perform full initialization here
        // if the GSLDevice is still uninitialized.
        // Only adapter initialization is required to validate
        // GL interoperability.
        PerformAdapterInitialization();

        // Attempt to associate GSL-OGL
        if (!glAssociate((CALvoid*)pContext, pDevice)) {
            if (!validateOnly) {
                LogError("Failed gslGLAssociate()");
            }
            return false;
        }
    }
        break;
    default:
        LogError("Unknown external device!");
        return false;
        break;
    }

    return true;
}

bool
Device::unbindExternalDevice(intptr_t type, void* pDevice, void* pContext, bool validateOnly)
{
    if (type != CL_GL_CONTEXT_KHR) {
        return true;
    }

    if (pDevice != NULL) {
        // Dissociate GSL-OGL
        if (true != glDissociate(pContext, pDevice)) {
            if (validateOnly) {
                LogWarning("Failed gslGLDiassociate()");
            }
            return false;
        }
    }
    return true;
}

void*
Device::allocMapTarget(
    amd::Memory&        mem,
    const amd::Coord3D& origin,
    const amd::Coord3D& region,
    uint                mapFlags,
    size_t*             rowPitch,
    size_t*             slicePitch)
{
    // Translate memory references
    gpu::Memory* memory = getGpuMemory(&mem);
    if (memory == NULL) {
        LogError("allocMapTarget failed. Can't allocate video memory");
        return NULL;
    }

    // Pass request over to memory
    return memory->allocMapTarget(origin, region, mapFlags, rowPitch, slicePitch);
}

bool
Device::globalFreeMemory(size_t* freeMemory) const
{
    const uint  TotalFreeMemory = 0;
    const uint  LargestFreeBlock = 1;

    // Initialization of heap and other resources because getMemInfo needs it.
    if (!(const_cast<Device*>(this)->initializeHeapResources())) {
        return false;
    }
    if (heap()->isVirtual()) {
        gslMemInfo memInfo = {0};
        getMemInfo(&memInfo);

         // Fill free memory info
        freeMemory[TotalFreeMemory] = (memInfo.cardMemAvailableBytes +
            memInfo.cardExtMemAvailableBytes) / Ki;
        freeMemory[LargestFreeBlock] = std::max(memInfo.cardLargestFreeBlockBytes,
           memInfo.cardExtLargestFreeBlockBytes) / Ki;
        if (settings().apuSystem_) {
            freeMemory[TotalFreeMemory] += memInfo.agpMemAvailableBytes / Ki;
            freeMemory[LargestFreeBlock] += memInfo.agpLargestFreeBlockBytes / Ki;
        }
    }
    else {
        freeMemory[TotalFreeMemory] = static_cast<size_t>((info().globalMemSize_ -
            static_cast<cl_ulong>(heapSize_) + heap()->freeSpace()) / Ki);
        freeMemory[LargestFreeBlock] = freeMemory[TotalFreeMemory];
    }

    return true;
}

void
Device::addVACache(Memory* memory) const
{
    // Make sure system memory has direct access
    if (memory->isHostMemDirectAccess()) {
        // VA cache access must be serialised
        amd::ScopedLock lk(*vaCacheAccess_);
        void*   start = memory->owner()->getHostMem();
        void*   end = reinterpret_cast<address>(start) + memory->owner()->getSize();
        size_t  offset;
        Memory*   doubleMap = findMemoryFromVA(start, &offset);

        if (doubleMap == NULL) {
            // Allocate a new entry
            VACacheEntry*   entry = new VACacheEntry(start, end, memory);
            if (entry != NULL) {
                vaCacheList_->push_back(entry);
            }
        }
        else {
            LogError("Unexpected double map() call from the app!");
        }
    }
}

void
Device::removeVACache(const Memory* memory) const
{
    // Make sure system memory has direct access
    if (memory->isHostMemDirectAccess() && memory->owner()) {
        // VA cache access must be serialised
        amd::ScopedLock lk(*vaCacheAccess_);
        void*   start = memory->owner()->getHostMem();
        void*   end = reinterpret_cast<address>(start) + memory->owner()->getSize();

        // Find VA cache entry for the specified memory
        for (const auto& entry : *vaCacheList_) {
            if (entry->startAddress_ == start) {
                CondLog((entry->endAddress_ != end), "Incorrect VA range");
                delete entry;
                vaCacheList_->remove(entry);
                break;
            }
        }
    }
}

Memory*
Device::findMemoryFromVA(const void* ptr, size_t* offset) const
{
    // VA cache access must be serialised
    amd::ScopedLock lk(*vaCacheAccess_);
    for (const auto& entry : *vaCacheList_) {
        if ((entry->startAddress_ <= ptr) && (entry->endAddress_ > ptr)) {
            *offset = static_cast<size_t>(reinterpret_cast<const char*>(ptr) -
                reinterpret_cast<char*>(entry->startAddress_));
            return entry->memory_;
        }
    }
    return NULL;
}

amd::Memory*
Device::findMapTarget(size_t size) const
{
    // Must be serialised for access
    amd::ScopedLock lk(*mapCacheOps_);

    amd::Memory*    map = NULL;
    size_t          minSize = 0;
    size_t          maxSize = 0;
    uint            mapId = mapCache_->size();
    uint            releaseId = mapCache_->size();

    // Find if the list has a map target of appropriate size
    for (uint i = 0; i < mapCache_->size(); i++) {
        if ((*mapCache_)[i] != NULL) {
            // Requested size is smaller than the entry size
            if (size < (*mapCache_)[i]->getSize()) {
                if ((minSize == 0) ||
                    (minSize > (*mapCache_)[i]->getSize())) {
                    minSize = (*mapCache_)[i]->getSize();
                    mapId = i;
                }
            }
            // Requeted size matches the entry size
            else if (size == (*mapCache_)[i]->getSize()) {
                mapId = i;
                break;
            }
            else {
                // Find the biggest map target in the list
                if (maxSize < (*mapCache_)[i]->getSize()) {
                    maxSize = (*mapCache_)[i]->getSize();
                    releaseId = i;
                }
            }
        }
    }

    // Check if we found any map target
    if (mapId < mapCache_->size()) {
        map = (*mapCache_)[mapId];
        (*mapCache_)[mapId] = NULL;
        Memory*     gpuMemory = reinterpret_cast<Memory*>
            (map->getDeviceMemory(*this));

        // Get the base pointer for the map resource
        if ((gpuMemory == NULL) || (NULL == gpuMemory->map(NULL))) {
            (*mapCache_)[mapId]->release();
            map = NULL;
        }
    }
    // If cache is full, then release the biggest map target
    else if (releaseId < mapCache_->size()) {
        (*mapCache_)[releaseId]->release();
        (*mapCache_)[releaseId] = NULL;
    }

    return map;
}

bool
Device::addMapTarget(amd::Memory* memory) const
{
    // Must be serialised for access
    amd::ScopedLock lk(*mapCacheOps_);

    //the svm memory shouldn't be cached
    if (!memory->canBeCached()) {
        return false;
    }
    // Find if the list has a map target of appropriate size
    for (uint i = 0; i < mapCache_->size(); ++i) {
        if ((*mapCache_)[i] == NULL) {
            (*mapCache_)[i] = memory;
            return true;
        }
    }

    // Add a new entry
    mapCache_->push_back(memory);

    return true;
}

Device::ScratchBuffer::~ScratchBuffer()
{
    destroyMemory();
}

void
Device::ScratchBuffer::destroyMemory()
{
    for (uint i = 0; i < memObjs_.size(); ++i) {
        // Release memory object
        delete memObjs_[i];
        memObjs_[i] = NULL;
    }
}

bool
Device::allocScratch(uint regNum, const VirtualGPU* vgpu)
{
    if (regNum > 0) {
        // Serialize the scratch buffer allocation code
        amd::ScopedLock lk(*scratchAlloc_);
        uint    sb = vgpu->hwRing();

        // Check if the current buffer isn't big enough
        if (regNum > scratch_[sb]->regNum_) {
            // Stall all command queues, since runtime will reallocate memory
            ScopedLockVgpus lock(*this);

            scratch_[sb]->regNum_ = regNum;
            size_t size = 0;
            uint offset = 0;

            // Destroy all views
            for (uint s = 0; s < scratch_.size(); ++s) {
                ScratchBuffer*  scratchBuf = scratch_[s];
                if (scratchBuf->regNum_ > 0) {
                    scratchBuf->destroyMemory();
                    // Calculate the size of the scratch buffer for a queue
                    scratchBuf->size_ = calcScratchBufferSize(scratchBuf->regNum_);
                    scratchBuf->offset_ = offset;
                    size += scratchBuf->size_ * scratchBuf->memObjs_.size();
                    offset += scratchBuf->size_;
                }
            }

            delete globalScratchBuf_;

            // Allocate new buffer.
            globalScratchBuf_ = new gpu::Memory(*this, size);
            if ((globalScratchBuf_ == NULL) ||
                !globalScratchBuf_->create(Resource::Scratch)) {
                LogError("Couldn't allocate scratch memory");
                for (uint s = 0; s < scratch_.size(); ++s) {
                    scratch_[s]->regNum_ = 0;
                }
                return false;
            }

            for (uint s = 0; s < scratch_.size(); ++s) {
                std::vector<Memory*>& mems = scratch_[s]->memObjs_;

                // Loop through all memory objects and reallocate them
                for (uint i = 0; i < mems.size(); ++i) {
                    if (scratch_[s]->regNum_ > 0) {
                        // Allocate new buffer
                        mems[i] = new gpu::Memory(*this, scratch_[s]->size_);
                        Resource::ViewParams    view;
                        view.resource_ = globalScratchBuf_;
                        view.offset_ = scratch_[s]->offset_ + i * scratch_[s]->size_;
                        view.size_ = scratch_[s]->size_;
                        if ((mems[i] == NULL) || !mems[i]->create(Resource::View, &view)) {
                            LogError("Couldn't allocate a scratch view");
                            scratch_[s]->regNum_ = 0;
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool
Device::validateKernel(const amd::Kernel& kernel, const device::VirtualDevice* vdev)
{
    // Find the number of scratch registers used in the kernel
    const device::Kernel* devKernel = kernel.getDeviceKernel(*this);
    uint regNum = static_cast<uint>(devKernel->workGroupInfo()->scratchRegs_);
    const VirtualGPU* vgpu = static_cast<const VirtualGPU*>(vdev);

    if (!allocScratch(regNum, vgpu)) {
        return false;
    }

    if (devKernel->hsa()) {
        const HSAILKernel* hsaKernel = static_cast<const HSAILKernel*>(devKernel);
        if (hsaKernel->dynamicParallelism()) {
            amd::DeviceQueue*  defQueue =
                kernel.program().context().defDeviceQueue(*this);
            vgpu = static_cast<VirtualGPU*>(defQueue->vDev());
            if (!allocScratch(hsaKernel->prog().maxScratchRegs(), vgpu)) {
                return false;
            }
        }
    }

    return true;
}

void
Device::destroyScratchBuffers()
{
    if (globalScratchBuf_ != NULL) {
        for (uint s = 0; s < scratch_.size(); ++s) {
            scratch_[s]->destroyMemory();
            scratch_[s]->regNum_ = 0;
        }
        delete globalScratchBuf_;
        globalScratchBuf_ = NULL;
    }
}

void
Device::fillHwSampler(uint32_t state, void* hwState, uint32_t hwStateSize) const
{
    // All GSL sampler's parameters are in floats
    uint32_t    gslAddress = GSL_CLAMP_TO_BORDER;
    uint32_t    gslMinFilter = GSL_MIN_NEAREST;
    uint32_t    gslMagFilter = GSL_MAG_NEAREST;
    bool        unnorm = !(state & amd::Sampler::StateNormalizedCoordsMask);

    state &= ~amd::Sampler::StateNormalizedCoordsMask;

    // Program the sampler address mode
    switch (state & amd::Sampler::StateAddressMask) {
        case amd::Sampler::StateAddressRepeat:
            gslAddress = GSL_REPEAT;
            break;
        case amd::Sampler::StateAddressClampToEdge:
            gslAddress = GSL_CLAMP_TO_EDGE;
            break;
        case amd::Sampler::StateAddressMirroredRepeat:
            gslAddress = GSL_MIRRORED_REPEAT;
            break;
        case amd::Sampler::StateAddressClamp:
        case amd::Sampler::StateAddressNone:
        default:
            break;
    }
    state &= ~amd::Sampler::StateAddressMask;

    // Program texture filter mode
    if (state == amd::Sampler::StateFilterLinear) {
        gslMinFilter = GSL_MIN_LINEAR;
        gslMagFilter = GSL_MAG_LINEAR;
    }

    fillSamplerHwState(unnorm, gslMinFilter, gslMagFilter,
        gslAddress, hwState, hwStateSize);
}

void*
Device::hostAlloc(size_t size, size_t alignment, bool atomics) const
{
    //for discrete gpu, we only reserve,no commit yet.
    return amd::Os::reserveMemory(NULL, size, alignment, amd::Os::MEM_PROT_NONE);
}

void
Device::hostFree(void* ptr, size_t size) const
{
    //If we allocate the host memory, we need free, or we have to release
    amd::Os::releaseMemory(ptr, size);
}

void*
Device::svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags, void* svmPtr) const
{
    alignment = std::max(alignment, static_cast<size_t>(info_.memBaseAddrAlign_));

    //VAM for GPU needs 64K alignment for Tahiti and CI+, will pull idnfo from gsl later
    size_t vmBigK = 64 * Ki;
    alignment =  (alignment < vmBigK) ? vmBigK : alignment;

    size = amd::alignUp(size, alignment);
    amd::Memory* mem = NULL;
    if (NULL == svmPtr) {
        //create a hidden buffer, which will allocated on the device later
        mem = new (context)amd::Buffer(context, flags, size, reinterpret_cast<void*>(1));
        if (mem == NULL) {
            LogError("failed to create a svm mem object!");
            return NULL;
        }

        if (!mem->create(NULL, false)) {
            LogError("failed to create a svm hidden buffer!");
            mem->release();
            return NULL;
        }
        gpu::Memory* gpuMem = getGpuMemory(mem);
        //add the information to context so that we can use it later.
        amd::SvmManager::AddSvmBuffer(mem->getSvmPtr(), mem);

    }
    else {
        //find the existing amd::mem object
        mem = amd::SvmManager::FindSvmBuffer(svmPtr);
        if (NULL == mem) {
            return NULL;
        }
        gpu::Memory* gpuMem = getGpuMemory(mem);
    }

    return mem->getSvmPtr();
}

void
Device::svmFree(void *ptr) const
{
    amd::Memory * svmMem = NULL;
    svmMem = amd::SvmManager::FindSvmBuffer(ptr);
    if (NULL != svmMem) {
        svmMem->release();
        amd::SvmManager::RemoveSvmBuffer(ptr);
    }
}


Device::SrdManager::~SrdManager()
{
    for (uint i = 0; i < pool_.size(); ++i) {
        pool_[i].buf_->unmap(NULL);
        delete pool_[i].buf_;
        delete pool_[i].flags_;
    }
}

bool
Sampler::create(
    uint32_t oclSamplerState)
{
    hwSrd_ = dev_.srds().allocSrdSlot(&hwState_);
    if (0 == hwSrd_) {
        return false;
    }
    dev_.fillHwSampler(oclSamplerState, hwState_, HsaSamplerObjectSize);
    return true;
}

Sampler::~Sampler()
{
    dev_.srds().freeSrdSlot(hwSrd_);
}

uint64_t
Device::SrdManager::allocSrdSlot(address* cpuAddr)
{
    amd::ScopedLock lock(ml_);
    // Check all buffers in the pool of chunks
    for (uint i = 0; i < pool_.size(); ++i) {
        const Chunk&    ch = pool_[i];
        // Search for an empty slot
        for (uint s = 0; s < numFlags_; ++s) {
            uint mask = ch.flags_[s];
            // Check if there is an empty slot in this group
            if (mask != 0) {
                uint idx;
                // Find the first empty index
                for (idx = 0; (mask & 0x1) == 0; mask >>= 1, ++idx);
                // Mark the slot as busy
                ch.flags_[s] &= ~(1 << idx);
                // Calculate SRD offset in the buffer
                uint offset = (s * MaskBits + idx) * srdSize_;
                *cpuAddr = ch.buf_->data() + offset;
                return ch.buf_->vmAddress() + offset;
            }
        }
    }
    // At this point the manager doesn't have empty slots
    // and has to allocate a new chunk
    Chunk chunk;
    chunk.flags_ = new uint[numFlags_];
    if (chunk.flags_ == NULL) {
        return 0;
    }
    chunk.buf_ = new Memory(dev_, bufSize_);
    if (chunk.buf_ == NULL || !chunk.buf_->create(Resource::Remote) ||
        (NULL == chunk.buf_->map(NULL))) {
        delete [] chunk.flags_;
        delete chunk.buf_;
        return 0;
    }
    // All slots in the chunk are in "free" state
    memset(chunk.flags_, 0xff, numFlags_ * sizeof(uint));
    // Take the first one...
    chunk.flags_[0] &= ~0x1;
    pool_.push_back(chunk);
    *cpuAddr = chunk.buf_->data();
    return chunk.buf_->vmAddress();
}

void
Device::SrdManager::freeSrdSlot(uint64_t addr) {
    amd::ScopedLock lock(ml_);
    // Check all buffers in the pool of chunks
    for (uint i = 0; i < pool_.size(); ++i) {
        Chunk* ch = &pool_[i];
        // Find the offset
        int64_t offs = static_cast<int64_t>(addr) -
            static_cast<int64_t>(ch->buf_->vmAddress());
        // Check if the offset inside the chunk buffer
        if ((offs >= 0) && (offs < bufSize_)) {
            // Find the index in the chunk
            uint idx  = offs / srdSize_;
            uint s = idx / MaskBits;
            // Free the slot
            ch->flags_[s] |= 1 << (idx % MaskBits);
            return;
        }
    }
    assert(false && "Wrong slot address!");
}

void
Device::SrdManager::fillResourceList(std::vector<const Resource*>&   memList)
{
    for (uint i = 0; i < pool_.size(); ++i) {
        memList.push_back(pool_[i].buf_);
    }
}

cl_int
Device::hwDebugManagerInit(amd::Context *context, uintptr_t messageStorage)
{
    hwDebugMgr_ = new GpuDebugManager(this);
    cl_int status = hwDebugMgr_->registerDebugger(context, messageStorage);

    if (CL_SUCCESS != status) {
        delete hwDebugMgr_;
        hwDebugMgr_ = NULL;
    }

    return status;
}

void
Device::hwDebugManagerRemove()
{
    hwDebugMgr_->unregisterDebugger();

    delete hwDebugMgr_;
    hwDebugMgr_ = NULL;
}

} // namespace gpu
