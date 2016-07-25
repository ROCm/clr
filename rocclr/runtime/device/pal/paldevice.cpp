//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/pal/paldefs.hpp"
#include "device/pal/palmemory.hpp"
#include "device/pal/paldevice.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"
#include "thread/monitor.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palsettings.hpp"
#include "device/pal/palblit.hpp"
#include "device/pal/paldebugmanager.hpp"
#include "palLib.h"
#include "palPlatform.h"
#include "palDevice.h"
#include "cz_id.h"
#include "acl.h"

#include "amdocl/cl_common.hpp"
//#include "CL/cl_gl.h"

#ifdef _WIN32
#include <d3d9.h>
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#include "CL/cl_dx9_media_sharing.h"
#endif // _WIN32

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctype.h>
#include <algorithm>

bool
PalDeviceLoad()
{
    bool    ret = false;

    // Create online devices
    ret |= pal::Device::init();
    // Create offline GPU devices
    ret |= pal::NullDevice::init();

    return ret;
}

void
PalDeviceUnload()
{
    pal::Device::tearDown();
}

namespace pal {

aclCompiler* NullDevice::compiler_;
AppProfile Device::appProfile_;

NullDevice::NullDevice()
    : amd::Device(nullptr)
    , ipLevel_(Pal::GfxIpLevel::None)
    , hwInfo_(nullptr)
{
}

bool
NullDevice::init()
{
    std::vector<Device*> devices;

    devices = getDevices(CL_DEVICE_TYPE_GPU, false);

    // Loop through all supported devices and create each of them
    for (uint id = 0; id < sizeof(DeviceInfo) / sizeof(AMDDeviceInfo); ++id) {
        bool    foundActive = false;
        Pal::AsicRevision revision = static_cast<Pal::AsicRevision>(id);

        if (pal::DeviceInfo[id].targetName_[0] == '\0') {
            continue;
        }

        // Loop through all active devices and see if we match one
        for (uint i = 0; i < devices.size(); ++i) {
            if (static_cast<NullDevice*>(devices[i])->asicRevision() == revision) {
                foundActive = true;
                break;
            }
        }

        // Don't report an offline device if it's active
        if (foundActive) {
            continue;
        }

        NullDevice*  dev = new NullDevice();
        if (nullptr != dev) {
            if (!dev->create(revision, Pal::GfxIpLevel::_None)) {
                delete dev;
            }
            else {
                dev->registerDevice();
            }
        }
    }

    // Loop through all supported devices and create each of them
    for (uint id = static_cast<uint>(Pal::GfxIpLevel::GfxIp7);
        id <= static_cast<uint>(Pal::GfxIpLevel::GfxIp9); ++id) {
        bool    foundActive = false;
        Pal::GfxIpLevel ipLevel = static_cast<Pal::GfxIpLevel>(id);

        if (pal::GfxIpDeviceInfo[id].targetName_[0] == '\0') {
            continue;
        }

        // Loop through all active devices and see if we match one
        for (uint i = 0; i < devices.size(); ++i) {
            if (static_cast<NullDevice*>(devices[i])->ipLevel() == ipLevel) {
                foundActive = true;
                break;
            }
        }

        // Don't report an offline device if it's active
        if (foundActive) {
            continue;
        }

        NullDevice*  dev = new NullDevice();
        if (nullptr != dev) {
            if (!dev->create(Pal::AsicRevision::Unknown, ipLevel)) {
                delete dev;
            }
            else {
                dev->registerDevice();
            }
        }
    }

    return true;
}

bool
NullDevice::create(Pal::AsicRevision asicRevision, Pal::GfxIpLevel ipLevel)
{
    online_ = false;
    Pal::DeviceProperties properties = {};

    // Use fake GFX IP for the device init
    asicRevision_ = asicRevision;
    ipLevel_ = ipLevel;
    properties.revision = asicRevision;
    properties.gfxLevel = ipLevel;

    // Update HW info for the device
    if (ipLevel == Pal::GfxIpLevel::_None) {
        hwInfo_ = &DeviceInfo[static_cast<uint>(asicRevision)];
    }
    else {
        hwInfo_ = &GfxIpDeviceInfo[static_cast<uint>(ipLevel)];
    }

    settings_ = new pal::Settings();
    pal::Settings* palSettings = reinterpret_cast<pal::Settings*>(settings_);

    // Report 512MB for all offline devices
    Pal::GpuMemoryHeapProperties heaps[Pal::GpuHeapCount];
    heaps[Pal::GpuHeapLocal].heapSize = 512 * Mi;

    // Create setting for the offline target
    if ((palSettings == nullptr) || !palSettings->create(properties, heaps)) {
        return false;
    }

    // Fill the device info structure
    fillDeviceInfo(properties, heaps, 4096, 1);

    // Runtime doesn't know what local size could be on the real board
    info_.maxGlobalVariableSize_ = static_cast<size_t>(512 * Mi);

    return true;
}

device::Program*
NullDevice::createProgram(amd::option::Options* options)
{
    device::Program* program;
    program = new HSAILProgram(*this);

    if (program == nullptr) {
        LogError("Memory allocation has failed!");
    }

    return program;
}

void NullDevice::fillDeviceInfo(
    const Pal::DeviceProperties& palProp,
    const Pal::GpuMemoryHeapProperties heaps[Pal::GpuHeapCount],
    size_t  maxTextureSize,
    uint    numComputeRings)
{
    info_.type_     = CL_DEVICE_TYPE_GPU;
    info_.vendorId_ = palProp.vendorId;

    info_.maxWorkItemDimensions_    = 3;
    info_.maxComputeUnits_          =
        palProp.gfxipProperties.shaderCore.numShaderEngines *
        palProp.gfxipProperties.shaderCore.numShaderArrays *
        palProp.gfxipProperties.shaderCore.numCusPerShaderArray;
    info_.numberOfShaderEngines     = palProp.gfxipProperties.shaderCore.numShaderEngines;

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

    info_.maxClockFrequency_    = (palProp.gfxipProperties.performance.maxGpuClock != 0) ?
        palProp.gfxipProperties.performance.maxGpuClock : 555;
    info_.maxParameterSize_ = 1024;
    info_.minDataTypeAlignSize_ = sizeof(cl_long16);
    info_.singleFPConfig_       = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
        | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

    if (settings().singleFpDenorm_) {
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

    uint64_t localRAM = heaps[Pal::GpuHeapLocal].heapSize +
         heaps[Pal::GpuHeapInvisible].heapSize;
#if defined(ATI_OS_LINUX)
    info_.globalMemSize_   =
        (static_cast<cl_ulong>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
        // globalMemSize is the actual available size for app on Linux
        // Because Linux base driver doesn't support paging
        static_cast<cl_ulong>(memInfo.cardMemAvailableBytes + memInfo.cardExtMemAvailableBytes) / 100u);
#else
    info_.globalMemSize_   =
        (static_cast<cl_ulong>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
        static_cast<cl_ulong>(localRAM) / 100u);
#endif
    if (settings().apuSystem_) {
        info_.globalMemSize_   +=
            (static_cast<cl_ulong>(heaps[Pal::GpuHeapGartUswc].heapSize) * Mi * 75)/100;
    }

    // Find the largest heap form FB memory
    info_.maxMemAllocSize_ = std::max(
        cl_ulong(heaps[Pal::GpuHeapLocal].heapSize),
        cl_ulong(heaps[Pal::GpuHeapInvisible].heapSize));

#if defined(ATI_OS_WIN)
    if (settings().apuSystem_) {
        info_.maxMemAllocSize_ = std::max(
            (static_cast<cl_ulong>(heaps[Pal::GpuHeapGartUswc].heapSize) * Mi * 75)/100,
            info_.maxMemAllocSize_);
    }
#endif
    info_.maxMemAllocSize_ = cl_ulong(info_.maxMemAllocSize_ *
        std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);

    //! \note Force max single allocation size.
    //! 4GB limit for the blit kernels and 64 bit optimizations.
    info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_,
            static_cast<cl_ulong>(settings().maxAllocSize_));

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
        info_.image2DMaxWidth_   = maxTextureSize;
        info_.image2DMaxHeight_  = maxTextureSize;
        info_.image3DMaxWidth_   = std::min(2 * Ki, maxTextureSize);
        info_.image3DMaxHeight_  = std::min(2 * Ki, maxTextureSize);
        info_.image3DMaxDepth_   = std::min(2 * Ki, maxTextureSize);

        info_.imagePitchAlignment_       = 1;   // PAL uses LINEAR_GENERAL
        info_.imageBaseAddressAlignment_ = 256; // XXX: 256 byte base address alignment for now

        info_.bufferFromImageSupport_ = CL_TRUE;
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

    if (false && (asicRevision() == Pal::AsicRevision::Carrizo) &&
        ASICREV_IS_CARRIZO_BRISTOL(palProp.revisionId)) {
        const static char* bristol = "Bristol Ridge";
        ::strcpy(info_.name_, bristol);
    }
    else {
        ::strcpy(info_.name_, hwInfo()->targetName_);
    }
    ::strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");
    ::snprintf(info_.driverVersion_, sizeof(info_.driverVersion_) - 1,
         AMD_BUILD_STRING "%s", " (VM)");

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

    info_.localMemType_ = CL_LOCAL;
    info_.localMemSize_ = settings().hwLDSSize_;
    info_.extensions_   = getExtensionString();

    info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
/*    info_.deviceTopology_.pcie.bus  = palProp.pciProperties.busNumber;
    info_.deviceTopology_.pcie.device = palProp.pciProperties.deviceNumber;
    info_.deviceTopology_.pcie.function = palProp.pciProperties.functionNumber;
*/
    ::strncpy(info_.boardName_, palProp.gpuName,
        ::strnlen(palProp.gpuName, sizeof(info_.boardName_)));

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
        info_.queueOnDeviceMaxSize_ = 8 * Mi;
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
        info_.wavefrontWidth_       = palProp.gfxipProperties.shaderCore.wavefrontSize;
        info_.globalMemChannels_    = palProp.gpuMemoryProperties.performance.vramBusBitWidth / 32;
        info_.globalMemChannelBanks_     = 4;
        info_.globalMemChannelBankWidth_ = hwInfo()->memChannelBankWidth_;
        info_.localMemSizePerCU_    = hwInfo()->localMemSizePerCU_;
        info_.localMemBanks_        = hwInfo()->localMemBanks_;
        info_.gfxipVersion_         = hwInfo()->gfxipVersion_;
        info_.numAsyncQueues_       = numComputeRings;
        info_.numRTQueues_          = 2;
        info_.numRTCUs_             = 4;
        info_.threadTraceEnable_    = settings().threadTraceEnable_;
    }
}

Device::XferBuffers::~XferBuffers()
{
    // Destroy temporary buffer for reads
    for (const auto& buf : freeBuffers_) {
        // CPU optimization: unmap staging buffer just once
        if (!buf->desc().cardMemory_) {
            buf->unmap(nullptr);
        }
        delete buf;
    }
    freeBuffers_.clear();
}

bool
Device::XferBuffers::create()
{
    Memory*     xferBuf = nullptr;
    bool        result = false;
    // Create a buffer object
    xferBuf = new Memory(dev(), bufSize_);

    // Try to allocate memory for the transfer buffer
    if ((nullptr == xferBuf) || !xferBuf->create(type_)) {
        delete xferBuf;
        xferBuf = nullptr;
        LogError("Couldn't allocate a transfer buffer!");
    }
    else {
        result = true;
        freeBuffers_.push_back(xferBuf);
        // CPU optimization: map staging buffer just once
        if (!xferBuf->desc().cardMemory_) {
            xferBuf->map(nullptr);
        }
    }

    return result;
}

Memory&
Device::XferBuffers::acquire()
{
    Memory*     xferBuf = nullptr;
    size_t      listSize;

    // Lock the operations with the staged buffer list
    amd::ScopedLock  l(lock_);
    listSize = freeBuffers_.size();

    // If the list is empty, then attempt to allocate a staged buffer
    if (listSize == 0) {
        // Allocate memory
        xferBuf = new Memory(dev(), bufSize_);

        // Allocate memory for the transfer buffer
        if ((nullptr == xferBuf) || !xferBuf->create(type_)) {
            delete xferBuf;
            xferBuf = nullptr;
            LogError("Couldn't allocate a transfer buffer!");
        }
        else {
            ++acquiredCnt_;
            // CPU optimization: map staging buffer just once
            if (!xferBuf->desc().cardMemory_) {
                xferBuf->map(nullptr);
            }
        }
    }

    if (xferBuf == nullptr) {
        xferBuf = *(freeBuffers_.begin());
        freeBuffers_.erase(freeBuffers_.begin());
        ++acquiredCnt_;
    }

    return *xferBuf;
}

void
Device::XferBuffers::release(VirtualGPU& gpu, Memory& buffer)
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
    , numOfVgpus_(0)
    , context_(nullptr)
    , lockAsyncOps_(nullptr)
    , lockForInitHeap_(nullptr)
    , lockPAL_(nullptr)
    , vgpusAccess_(nullptr)
    , scratchAlloc_(nullptr)
    , mapCacheOps_(nullptr)
    , xferRead_(nullptr)
    , xferWrite_(nullptr)
    , mapCache_(nullptr)
    , resourceCache_(nullptr)
    , numComputeEngines_(0)
    , numDmaEngines_(0)
    , heapInitComplete_(false)
    , xferQueue_(nullptr)
    , globalScratchBuf_(nullptr)
    , srdManager_(nullptr)
{
}

Device::~Device()
{
    // remove the HW debug manager
    delete hwDebugMgr_;
    hwDebugMgr_ = nullptr;

    delete srdManager_;

    for (uint s = 0; s < scratch_.size(); ++s) {
        delete scratch_[s];
        scratch_[s] = nullptr;
    }

    delete globalScratchBuf_;
    globalScratchBuf_ = nullptr;

    // Destroy transfer queue
    delete xferQueue_;

    // Destroy blit program
    delete blitProgram_;

    // Release cached map targets
    for (uint i = 0; mapCache_ != nullptr && i < mapCache_->size(); ++i) {
        if ((*mapCache_)[i] != nullptr) {
            (*mapCache_)[i]->release();
        }
    }
    delete mapCache_;

    // Destroy temporary buffers for read/write
    delete xferRead_;
    delete xferWrite_;

    // Destroy resource cache
    delete resourceCache_;

    delete lockAsyncOps_;
    delete lockForInitHeap_;
    delete lockPAL_;
    delete vgpusAccess_;
    delete scratchAlloc_;
    delete mapCacheOps_;

    if (context_ != nullptr) {
        context_->release();
    }

    device_ = nullptr;
}

extern const char* SchedulerSourceCode;

bool
Device::create(Pal::IDevice* device)
{
    if (!amd::Device::create()) {
        return false;
    }

    appProfile_.init();
    device_ = device;
    Pal::Result result;

    // Retrive device properties
    result = iDev()->GetProperties(&properties_);

    // Save the IP level for the offline detection
    ipLevel_ = properties().gfxLevel;
    asicRevision_ = properties().revision;

    // Update HW info for the device
    if (properties().revision == Pal::AsicRevision::Unknown) {
        hwInfo_ = &GfxIpDeviceInfo[static_cast<uint>(properties().gfxLevel)];
    }
    else {
        hwInfo_ = &DeviceInfo[static_cast<uint>(properties().revision)];
    }


    // Find the number of available engines
    numComputeEngines_ =
        properties().engineProperties[Pal::QueueTypeCompute].engineCount -
        properties().engineProperties[Pal::QueueTypeCompute].numExclusiveComputeEngines;
    numDmaEngines_ =
        properties().engineProperties[Pal::QueueTypeDma].engineCount;

    Pal::PalPublicSettings*const palSettings = iDev()->GetPublicSettings();
    // Modify settings here
    // palSettings ...
    palSettings->textureOptLevel = Pal::TextureFilterOptimizationsDisabled;
    palSettings->forceHighClocks = appProfile_.enableHighPerformanceState();
    palSettings->cmdBufBatchedSubmitChainLimit = 0;

    // Commit the new settings for the device
    result = iDev()->CommitSettingsAndInit();
    if (result == Pal::Result::Success) {
        Pal::DeviceFinalizeInfo finalizeInfo = {};

        // Request all compute engines
        finalizeInfo.engineCounts[Pal::QueueTypeCompute] = numComputeEngines_;
        // Request all SDMA engines
        finalizeInfo.engineCounts[Pal::QueueTypeDma] = numDmaEngines_;

        result = iDev()->Finalize(finalizeInfo);
    }

    Pal::GpuMemoryHeapProperties heaps[Pal::GpuHeapCount];
    iDev()->GetGpuMemoryHeapProperties(heaps);

    // Creates device settings
    settings_ = new pal::Settings();
    pal::Settings* gpuSettings = reinterpret_cast<pal::Settings*>(settings_);
    if ((gpuSettings == nullptr) || !gpuSettings->create(properties(), heaps,
        appProfile_.reportAsOCL12Device())) {
        return false;
    }
    numComputeEngines_ = std::min(numComputeEngines_, settings().numComputeRings_);

    amd::Context::Info  info = {0};
    std::vector<amd::Device*> devices;
    devices.push_back(this);

    // Create a dummy context
    context_ = new amd::Context(devices, info);
    if (context_ == nullptr) {
        return false;
    }

    // Create the locks
    lockAsyncOps_ = new amd::Monitor("Device Async Ops Lock", true);
    if (nullptr == lockAsyncOps_) {
        return false;
    }
    lockPAL_ = new amd::Monitor("PAL Ops Lock", true);
    if (nullptr == lockPAL_) {
        return false;
    }

    lockForInitHeap_ = new amd::Monitor("Async Ops Lock For Initialization of Heap Resource", true);
    if (nullptr == lockForInitHeap_) {
        return false;
    }

    vgpusAccess_ = new amd::Monitor("Virtual GPU List Ops Lock", true);
    if (nullptr == vgpusAccess_) {
        return false;
    }

    scratchAlloc_ = new amd::Monitor("Scratch Allocation Lock", true);
    if (nullptr == scratchAlloc_) {
        return false;
    }

    mapCacheOps_ = new amd::Monitor("Map Cache Lock", true);
    if (nullptr == mapCacheOps_) {
        return false;
    }

    mapCache_ = new std::vector<amd::Memory*>();
    if (mapCache_ == nullptr) {
        return false;
    }
    // Use just 1 entry by default for the map cache
    mapCache_->push_back(nullptr);

    size_t  resourceCacheSize = settings().resourceCacheSize_;

#ifdef DEBUG
    std::stringstream  message;
    if (settings().remoteAlloc_) {
        message << "Using *Remote* memory";
    }
    else {
        message << "Using *Local* memory";
    }

    message << std::endl;
    LogInfo(message.str().c_str());
#endif // DEBUG

    // Create resource cache.
    // \note Cache must be created before any resource creation to avoid nullptr check
    resourceCache_ = new ResourceCache(resourceCacheSize);
    if (nullptr == resourceCache_) {
        return false;
    }

    // Fill the device info structure
    fillDeviceInfo(properties(), heaps, 16*Ki, numComputeEngines());

    for (uint i = 0; i < Pal::GpuHeap::GpuHeapCount; ++i) {
        freeMem[i] = heaps[i].heapSize;
    }

    // Allocate SRD manager
    srdManager_ = new SrdManager(*this,
        std::max(HsaImageObjectSize, HsaSamplerObjectSize), 64 * Ki);
    if (srdManager_ == nullptr) {
        return false;
    }

    // create the HW debug manager if needed
    if (settings().enableHwDebug_) {
        hwDebugMgr_ = new GpuDebugManager(this);
    }

    return true;
}

bool
Device::initializeHeapResources()
{
    amd::ScopedLock k(lockForInitHeap_);
    if (!heapInitComplete_) {
        heapInitComplete_ = true;

        scratch_.resize((settings().useSingleScratch_) ?
            1 : (numComputeEngines() ? numComputeEngines() : 1));

        // Initialize the number of mem object for the scratch buffer
        for (uint s = 0; s < scratch_.size(); ++s) {
            scratch_[s] = new ScratchBuffer();
            if (nullptr == scratch_[s]) {
                return false;
            }
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
                    amd::alignUp(settings().stagedXferSize_, 4 * Ki));
                if ((xferWrite_ == nullptr) || !xferWrite_->create()) {
                    LogError("Couldn't allocate transfer buffer objects for read");
                    return false;
                }
            }

            // Initialize staged read buffers
            if (settings().stagedXferRead_) {
                xferRead_ = new XferBuffers(*this, Resource::Remote,
                    amd::alignUp(settings().stagedXferSize_, 4 * Ki));
                if ((xferRead_ == nullptr) || !xferRead_->create()) {
                    LogError("Couldn't allocate transfer buffer objects for write");
                    return false;
                }
            }
        }

        // Delay compilation due to brig_loader memory allocation
        const char* scheduler = nullptr;
        const char* ocl20 = nullptr;
        if (settings().oclVersion_ == OpenCL20) {
            scheduler = SchedulerSourceCode;
            ocl20 = "-cl-std=CL2.0";
        }
        blitProgram_ = new BlitProgram(context_);
        // Create blit programs
        if (blitProgram_ == nullptr ||
            !blitProgram_->create(this, scheduler, ocl20)) {
            delete blitProgram_;
            blitProgram_ = nullptr;
            LogError("Couldn't create blit kernels!");
            return false;
        }

        // Create a synchronized transfer queue
        xferQueue_ = new VirtualGPU(*this);
        if (!(xferQueue_ && xferQueue_->create(
            false
            ))) {
            delete xferQueue_;
            xferQueue_ = nullptr;
        }
        if (nullptr == xferQueue_) {
            LogError("Couldn't create the device transfer manager!");
            return false;
        }
        xferQueue_->enableSyncedBlit();
    }
    return true;
}

device::VirtualDevice*
Device::createVirtualDevice(
    amd::CommandQueue*  queue
    )
{
    bool    profiling = false;
    bool    interopQueue = false;
    uint    rtCUs  = 0;
    uint    deviceQueueSize = 0;

    if (queue != nullptr) {
        profiling = queue->properties().test(CL_QUEUE_PROFILING_ENABLE);
        if (queue->asHostQueue() != nullptr) {
            interopQueue = (0 != (queue->context().info().flags_ &
                (amd::Context::GLDeviceKhr |
                 amd::Context::D3D10DeviceKhr |
                 amd::Context::D3D11DeviceKhr)));
            rtCUs = queue->rtCUs();
        }
        else if (queue->asDeviceQueue() != nullptr) {
            deviceQueueSize = queue->asDeviceQueue()->size();
        }
    }

    // Not safe to add a queue. So lock the device
    amd::ScopedLock k(lockAsyncOps());
    amd::ScopedLock lock(vgpusAccess());

    // Initialization of heap and other resources occur during the command queue creation time.
    if (!initializeHeapResources()) {
        LogError("Heap initializaiton fails!");
        return nullptr;
    }

    VirtualGPU* vgpu = new VirtualGPU(*this);
    if (vgpu && vgpu->create(
        profiling
        , deviceQueueSize
        )) {
        return vgpu;
    } else {
        delete vgpu;
        return nullptr;
    }
}

device::Program*
Device::createProgram(amd::option::Options* options)
{
    device::Program* program;
    program = new HSAILProgram(*this);
    if (program == nullptr) {
        LogError("We failed memory allocation for program!");
    }

    return program;
}

//! Requested devices list as configured by the GPU_DEVICE_ORDINAL
typedef std::map<int, bool> requestedDevices_t;

//! Parses the requested list of devices to be exposed to the user.
static void
parseRequestedDeviceList(requestedDevices_t &requestedDevices) {
    char *pch = nullptr;
    int requestedDeviceCount = 0;
    const char* requestedDeviceList = GPU_DEVICE_ORDINAL;

    pch = strtok(const_cast<char*>(requestedDeviceList), ",");
    while (pch != nullptr) {
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
        pch = strtok(nullptr, ",");
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

static char* platformObj;
static Pal::IPlatform* platform;

bool
Device::init()
{
    uint32_t    numDevices = 0;
    bool        useDeviceList = false;
    requestedDevices_t requestedDevices;

    const char* library = getenv("HSA_COMPILER_LIBRARY");
    aclCompilerOptions opts = {
        sizeof(aclCompilerOptions_0_8),
        library,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        AMD_OCL_SC_LIB
    };
    // Initialize the compiler handle
    acl_error   error;
    compiler_ = aclCompilerInit(&opts, &error);
    if (error != ACL_SUCCESS) {
        LogError("Error initializing the compiler");
        return false;
    }

    size_t size = Pal::GetPlatformSize();
    platformObj = new char[size];
    Pal::PlatformCreateInfo  info = {};
    info.pSettingsPath = "OCL";

    // PAL init
    if (Pal::Result::Success !=
        Pal::CreatePlatform(info, platformObj, &platform)) {
        return false;
    }

    // Get the total number of active devices
    // Count up all the devices in the system.
    Pal::IDevice* deviceList[Pal::MaxDevices] = {};
    platform->EnumerateDevices(&numDevices, &deviceList[0]);

    uint ordinal = 0;
    const char* selectDeviceByName = nullptr;
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
        bool    result = (nullptr != d) && d->create(deviceList[ordinal]);
        if (useDeviceList) {
            result &= (requestedDevices.find(ordinal) != requestedDevices.end());
        }
        if (result &&
            ((nullptr == selectDeviceByName) || ('\0' == selectDeviceByName[0]) ||
             (strstr(selectDeviceByName, d->info().name_) != nullptr))) {
            d->registerDevice();
        }
        else {
            delete d;
        }
    }
    return true;
}

void
Device::tearDown()
{
    platform->Destroy();
    delete platformObj;

    if (compiler_ != nullptr) {
        aclCompilerFini(compiler_);
    }
}

Memory*
Device::getGpuMemory(amd::Memory* mem) const
{
    return static_cast<pal::Memory*>(mem->getDeviceMemory(*this));
}

const device::BlitManager&
Device::xferMgr() const
{
    return xferQueue_->blitMgr();
}

Pal::Format
Device::getPalFormat(const amd::Image::Format& format, Pal::ChannelMapping* channel) const
{
    // Find PAL format
    for (uint i = 0; i < sizeof(MemoryFormatMap) / sizeof(MemoryFormat); ++i) {
        if ((format.image_channel_data_type ==
             MemoryFormatMap[i].clFormat_.image_channel_data_type) &&
            (format.image_channel_order ==
             MemoryFormatMap[i].clFormat_.image_channel_order)) {
            *channel = MemoryFormatMap[i].palChannel_;
            return MemoryFormatMap[i].palFormat_;
        }
    }
    assert(!"We didn't find PAL resource format!");
    *channel = MemoryFormatMap[0].palChannel_;
    return MemoryFormatMap[0].palFormat_;
}

// Create buffer without an owner (merge common code with createBuffer() ?)
pal::Memory*
Device::createScratchBuffer(size_t size) const
{
    Memory* gpuMemory = nullptr;

    // Create a memory object
    gpuMemory = new pal::Memory(*this, size);
    if (nullptr == gpuMemory || !gpuMemory->create(Resource::Local)) {
        delete gpuMemory;
        gpuMemory = nullptr;
    }

    return gpuMemory;
}

pal::Memory*
Device::createBuffer(
    amd::Memory&    owner,
    bool            directAccess) const
{
    size_t  size = owner.getSize();
    pal::Memory* gpuMemory;

    // Create resource
    bool result = false;

    if (owner.getType() == CL_MEM_OBJECT_PIPE) {
        // directAccess isnt needed as Pipes shouldnt be host accessible for GPU
        directAccess = false;
    }

    if (nullptr != owner.parent()) {
        pal::Memory*    gpuParent = getGpuMemory(owner.parent());
        if (nullptr == gpuParent) {
            LogError("Can't get the owner object for subbuffer allocation");
            return nullptr;
        }

        return gpuParent->createBufferView(owner);
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
    bool    remoteAlloc = false;
    // Internal means VirtualDevice!=nullptr
    bool    internalAlloc = ((owner.getMemFlags() & CL_MEM_USE_HOST_PTR) &&
            (owner.getVirtualDevice() != nullptr)) ? true : false;

    // Create a memory object
    gpuMemory = new pal::Buffer(*this, owner, owner.getSize());
    if (nullptr == gpuMemory) {
        return nullptr;
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
            if (!remoteAlloc && (owner.getHostMem() != nullptr)) {
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
                    // Don't use cached allocation
                    // if size is biger than max single alloc
                    if (owner.getSize() > info().maxMemAllocSize_) {
                        delete gpuMemory;
                        return nullptr;
                    }
                }
            }
        }
    }

    if (!result &&
        // Make sure it's not internal alloc
        !internalAlloc) {
        Resource::CreateParams  params;
        params.owner_ = &owner;
        params.gpu_ = static_cast<VirtualGPU*>(owner.getVirtualDevice());

        // Create memory object
        result = gpuMemory->create(type, &params);

        // If allocation was successful
        if (result) {
            // Initialize if the memory is a pipe object
            if (owner.getType() == CL_MEM_OBJECT_PIPE) {
                // Pipe initialize in order read_idx, write_idx, end_idx. Refer clk_pipe_t structure.
                // Init with 3 DWORDS for 32bit addressing and 6 DWORDS for 64bit
                size_t pipeInit[3] = {0 , 0, owner.asPipe()->getMaxNumPackets()};
                static_cast<const KernelBlitManager&>(xferMgr()).writeRawData(
                    *gpuMemory, sizeof(pipeInit), pipeInit);
            }
            // If memory has direct access from host, then get CPU address
            if (gpuMemory->isHostMemDirectAccess() &&
                (type != Resource::ExternalPhysical)) {
                void* address = gpuMemory->map(nullptr);
                if (address != nullptr) {
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
                    owner.setHostMem(nullptr);
                }
            }
        }
    }

    if (!result) {
        delete gpuMemory;
        return nullptr;
    }

    return gpuMemory;
}

pal::Memory*
Device::createImage(amd::Memory& owner, bool directAccess) const
{
    size_t  size = owner.getSize();
    amd::Image& image = *owner.asImage();
    pal::Memory* gpuImage = nullptr;

    if ((nullptr != owner.parent()) && (owner.parent()->asImage() != nullptr)) {
        device::Memory* devParent = owner.parent()->getDeviceMemory(*this);
        if (nullptr == devParent) {
            LogError("Can't get the owner object for image view allocation");
            return nullptr;
        }
        // Create a view on the specified device
        gpuImage = (pal::Memory*)createView(owner, *devParent);
        if ((nullptr != gpuImage) && (gpuImage->owner() != nullptr)) {
            gpuImage->owner()->setHostMem((address)(owner.parent()->getHostMem()) + gpuImage->owner()->getOrigin());
        }
        return gpuImage;
    }

    gpuImage = new pal::Image(*this, owner,
        image.getWidth(),
        image.getHeight(),
        image.getDepth(),
        image.getImageFormat(),
        image.getType(),
        image.getMipLevels());

    // Create resource
    if (nullptr != gpuImage) {
        const bool imageBuffer =
            ((owner.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
             ((owner.getType() == CL_MEM_OBJECT_IMAGE2D) &&
              (owner.parent() != nullptr) &&
              (owner.parent()->asBuffer() != nullptr)));
        bool result = false;

        // Check if owner is interop memory
        if (owner.isInterop()) {
            result = gpuImage->createInterop(Memory::InteropDirectAccess);
        }
        else if (imageBuffer) {
            Resource::ImageBufferParams  params;
            pal::Memory* buffer = reinterpret_cast<pal::Memory*>
                (image.parent()->getDeviceMemory(*this));
            if (buffer == nullptr) {
                LogError("Buffer creation for ImageBuffer failed!");
                delete gpuImage;
                return nullptr;
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
            return nullptr;
        }
        else if ((gpuImage->memoryType() != Resource::Pinned) &&
                 (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
                 (owner.getContext().devices().size() == 1)) {
            // Ignore copy for image1D_buffer, since it was already done for buffer
            if (imageBuffer) {
                // Clear CHP memory
                owner.setHostMem(nullptr);
            }
            else {
                amd::Coord3D    origin(0, 0, 0);
                static const bool Entire  = true;
                if (xferMgr().writeImage(owner.getHostMem(),
                    *gpuImage, origin, image.getRegion(), 0, 0, Entire)) {
                    // Clear CHP memory
                    owner.setHostMem(nullptr);
                }
            }
        }

        if (result) {
            size_t bytePitch = gpuImage->elementSize() * gpuImage->desc().width_;
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
    pal::Memory* memory = nullptr;

    if (owner.asBuffer()) {
        directAccess = (settings().hostMemDirectAccess_ & Settings::HostMemBuffer)
            ? true : false;
        memory = createBuffer(owner, directAccess);
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
    if ((memory != nullptr) &&
        (memory->memoryType() != Resource::Pinned) &&
        (memory->memoryType() != Resource::Remote) &&
        (memory->memoryType() != Resource::RemoteUSWC) &&
        (memory->memoryType() != Resource::ExternalPhysical) &&
        ((owner.getHostMem() != nullptr) ||
         ((nullptr != owner.parent()) && (owner.getHostMem() != nullptr)))) {
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
    *sampler = nullptr;
    Sampler* gpuSampler = new Sampler(*this);
    if ((nullptr == gpuSampler) || !gpuSampler->create(owner)) {
        delete gpuSampler;
        return false;
    }
    *sampler = gpuSampler;
    return true;
}

//! \note reallocMemory() must be called only from outside of
//! VirtualGPU submit commands methods.
//! Otherwise a deadlock in lockVgpus() is possible

bool
Device::reallocMemory(amd::Memory& owner) const
{
    bool directAccess   = false;

    // For now we have to serialize reallocation code
    amd::ScopedLock lk(*lockAsyncOps_);

    // Read device memory after the lock,
    // since realloc from another thread can replace the pointer
    pal::Memory*  gpuMemory = getGpuMemory(&owner);
    if (gpuMemory == nullptr) {
        return false;
    }

    if (gpuMemory->pinOffset() == 0) {
        return true;
    }
    else if (nullptr != owner.parent()) {
        if (!reallocMemory(*owner.parent())) {
            return false;
        }
    }

    if (owner.asBuffer()) {
        gpuMemory = createBuffer(owner, directAccess);
    }
    else if (owner.asImage()) {
        return true;
    }
    else {
        LogError("Unknown memory type!");
    }

    if (gpuMemory != nullptr) {
        pal::Memory* newMemory = gpuMemory;
        pal::Memory* oldMemory = getGpuMemory(&owner);

        // Transfer the object
        if (oldMemory != nullptr) {
            if (!oldMemory->moveTo(*newMemory)) {
                delete newMemory;
                return false;
            }
        }

        // Attempt to pin system memory
        if ((newMemory->memoryType() != Resource::Pinned) &&
            ((owner.getHostMem() != nullptr) ||
             ((nullptr != owner.parent()) && (owner.getHostMem() != nullptr)))) {
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
    assert((owner.asImage() != nullptr) && "View supports images only");
    const amd::Image& image = *owner.asImage();
    pal::Memory* gpuImage = nullptr;

    gpuImage = new pal::Image(*this, owner,
        image.getWidth(),
        image.getHeight(),
        image.getDepth(),
        image.getImageFormat(),
        image.getType(),
        image.getMipLevels());

    // Create resource
    if (nullptr != gpuImage) {
        bool result = false;
        Resource::ImageViewParams   params;
        const pal::Memory& gpuMem = static_cast<const pal::Memory&>(parent);

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
            return nullptr;
        }
    }

    return gpuImage;
}


//! Attempt to bind with external graphics API's device/context
bool
Device::bindExternalDevice(uint flags, void* const pDevice[], void* pContext, bool validateOnly)
{
    assert(pDevice);

#ifdef _WIN32
    if (flags & amd::Context::Flags::D3D10DeviceKhr) {
        if (!associateD3D10Device(pDevice[amd::Context::DeviceFlagIdx::D3D10DeviceKhrIdx])) {
            LogError("Failed gslD3D10Associate()");
            return false;
        }
    }

    if (flags & amd::Context::Flags::D3D11DeviceKhr) {
        if (!associateD3D11Device(pDevice[amd::Context::DeviceFlagIdx::D3D11DeviceKhrIdx])) {
            LogError("Failed gslD3D11Associate()");
            return false;
        }
    }

    if (flags & (amd::Context::Flags::D3D9DeviceKhr |
                      amd::Context::Flags::D3D9DeviceEXKhr)) {
        if (!associateD3D9Device(pDevice[amd::Context::DeviceFlagIdx::D3D9DeviceKhrIdx])) {
            LogWarning("D3D9<->OpenCL adapter mismatch or D3D9Associate() failure");
            return false;
        }
    }
#endif //_WIN32

    if (flags & amd::Context::Flags::GLDeviceKhr) {
        // Attempt to associate GSL-OGL
        if (!glAssociate(pContext, pDevice[amd::Context::DeviceFlagIdx::GLDeviceKhrIdx])) {
            if (!validateOnly) {
                LogError("Failed gslGLAssociate()");
            }
            return false;
        }
    }

    return true;
}

bool
Device::unbindExternalDevice(uint flags, void* const pDevice[], void* pContext, bool validateOnly)
{
    if ((flags & amd::Context::Flags::GLDeviceKhr) == 0) {
        return true;
    }

    void * glDevice = pDevice[amd::Context::DeviceFlagIdx::GLDeviceKhrIdx];
    if (glDevice != nullptr) {
        // Dissociate GSL-OGL
        if (!glDissociate(pContext, glDevice)) {
            if (validateOnly) {
                LogWarning("Failed gslGLDiassociate()");
            }
            return false;
        }
    }
    return true;
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

    Pal::gpusize local = freeMem[Pal::GpuHeapLocal];
    Pal::gpusize invisible = freeMem[Pal::GpuHeapInvisible];

    // Fill free memory info
    freeMemory[TotalFreeMemory] = static_cast<size_t>((local + invisible) / Ki);
    freeMemory[LargestFreeBlock] = static_cast<size_t>(std::max(local, invisible) / Ki);

    if (settings().apuSystem_) {
        Pal::gpusize uswc = freeMem[Pal::GpuHeapGartUswc];
        uswc /= Ki;
        freeMemory[TotalFreeMemory] += static_cast<size_t>(uswc);
        if (freeMemory[LargestFreeBlock] < uswc) {
            freeMemory[LargestFreeBlock] = static_cast<size_t>(uswc);
        }
    }

    return true;
}

amd::Memory*
Device::findMapTarget(size_t size) const
{
    // Must be serialised for access
    amd::ScopedLock lk(*mapCacheOps_);

    amd::Memory*    map = nullptr;
    size_t          minSize = 0;
    size_t          maxSize = 0;
    uint            mapId = mapCache_->size();
    uint            releaseId = mapCache_->size();

    // Find if the list has a map target of appropriate size
    for (uint i = 0; i < mapCache_->size(); i++) {
        if ((*mapCache_)[i] != nullptr) {
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
        (*mapCache_)[mapId] = nullptr;
        Memory*     gpuMemory = reinterpret_cast<Memory*>
            (map->getDeviceMemory(*this));

        // Get the base pointer for the map resource
        if ((gpuMemory == nullptr) || (nullptr == gpuMemory->map(nullptr))) {
            (*mapCache_)[mapId]->release();
            map = nullptr;
        }
    }
    // If cache is full, then release the biggest map target
    else if (releaseId < mapCache_->size()) {
        (*mapCache_)[releaseId]->release();
        (*mapCache_)[releaseId] = nullptr;
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
        if ((*mapCache_)[i] == nullptr) {
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
    // Release memory object
    delete memObj_;
    memObj_ = nullptr;
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
                    uint32_t numTotalCUs = info().maxComputeUnits_;
                    uint32_t numMaxWaves =
                        properties().gfxipProperties.shaderCore.maxScratchWavesPerCu * numTotalCUs;
                    scratchBuf->size_ = properties().gfxipProperties.shaderCore.wavefrontSize *
                        scratchBuf->regNum_ * numMaxWaves * sizeof(uint32_t);
                    scratchBuf->size_ = amd::alignUp(scratchBuf->size_, 0xFFFF);
                    scratchBuf->offset_ = offset;
                    size += scratchBuf->size_;
                    offset += scratchBuf->size_;
                }
            }

            delete globalScratchBuf_;

            // Allocate new buffer.
            globalScratchBuf_ = new pal::Memory(*this, size);
            if ((globalScratchBuf_ == nullptr) ||
                !globalScratchBuf_->create(Resource::Scratch)) {
                LogError("Couldn't allocate scratch memory");
                for (uint s = 0; s < scratch_.size(); ++s) {
                    scratch_[s]->regNum_ = 0;
                }
                return false;
            }

            for (uint s = 0; s < scratch_.size(); ++s) {
                // Loop through all memory objects and reallocate them
                if (scratch_[s]->regNum_ > 0) {
                    // Allocate new buffer
                    scratch_[s]->memObj_ = new pal::Memory(*this, scratch_[s]->size_);
                    Resource::ViewParams    view;
                    view.resource_ = globalScratchBuf_;
                    view.offset_ = scratch_[s]->offset_;
                    view.size_ = scratch_[s]->size_;
                    if ((scratch_[s]->memObj_ == nullptr) ||
                        !scratch_[s]->memObj_->create(Resource::View, &view)) {
                        LogError("Couldn't allocate a scratch view");
                        delete scratch_[s]->memObj_;
                        scratch_[s]->regNum_ = 0;
                        return false;
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
            if (defQueue != nullptr) {
                vgpu = static_cast<VirtualGPU*>(defQueue->vDev());
                if (!allocScratch(hsaKernel->prog().maxScratchRegs(), vgpu)) {
                    return false;
                }
            }
            else {
                return false;
            }
        }
    }

    return true;
}

void
Device::destroyScratchBuffers()
{
    if (globalScratchBuf_ != nullptr) {
        for (uint s = 0; s < scratch_.size(); ++s) {
            scratch_[s]->destroyMemory();
            scratch_[s]->regNum_ = 0;
        }
        delete globalScratchBuf_;
        globalScratchBuf_ = nullptr;
    }
}

void
Device::fillHwSampler(
    uint32_t state, void* hwState, uint32_t hwStateSize,
    uint32_t mipFilter, float minLod, float maxLod) const
{
    Pal::SamplerInfo samplerInfo = {};

    samplerInfo.borderColorType = Pal::BorderColorType::TransparentBlack;

    samplerInfo.filter.zFilter = Pal::XyFilterPoint;

    samplerInfo.flags.unnormalizedCoords = !(state & amd::Sampler::StateNormalizedCoordsMask);
    samplerInfo.maxLod = 4096.0f;

    state &= ~amd::Sampler::StateNormalizedCoordsMask;

    // Program the sampler address mode
    switch (state & amd::Sampler::StateAddressMask) {
        case amd::Sampler::StateAddressRepeat:
            samplerInfo.addressU = Pal::TexAddressMode::Wrap;
            samplerInfo.addressV = Pal::TexAddressMode::Wrap;
            samplerInfo.addressW = Pal::TexAddressMode::Wrap;
            break;
        case amd::Sampler::StateAddressClampToEdge:
            samplerInfo.addressU = Pal::TexAddressMode::Clamp;
            samplerInfo.addressV = Pal::TexAddressMode::Clamp;
            samplerInfo.addressW = Pal::TexAddressMode::Clamp;
            break;
        case amd::Sampler::StateAddressMirroredRepeat:
            samplerInfo.addressU = Pal::TexAddressMode::Mirror;
            samplerInfo.addressV = Pal::TexAddressMode::Mirror;
            samplerInfo.addressW = Pal::TexAddressMode::Mirror;
            break;
        case amd::Sampler::StateAddressClamp:
        case amd::Sampler::StateAddressNone:
            samplerInfo.addressU = Pal::TexAddressMode::ClampBorder;
            samplerInfo.addressV = Pal::TexAddressMode::ClampBorder;
            samplerInfo.addressW = Pal::TexAddressMode::ClampBorder;
        default:
            break;
    }
    state &= ~amd::Sampler::StateAddressMask;

    // Program texture filter mode
    if (state == amd::Sampler::StateFilterLinear) {
        samplerInfo.filter.magnification = Pal::XyFilterLinear;
        samplerInfo.filter.minification = Pal::XyFilterLinear;
        samplerInfo.filter.zFilter = Pal::ZFilterLinear;
    }

    if (mipFilter == CL_FILTER_NEAREST) {
        samplerInfo.filter.mipFilter = Pal::MipFilterPoint;
    }
    else if (mipFilter == CL_FILTER_LINEAR) {
        samplerInfo.filter.mipFilter = Pal::MipFilterLinear;
    }

    iDev()->CreateSamplerSrds(1, &samplerInfo, hwState);
}

void*
Device::hostAlloc(size_t size, size_t alignment, bool atomics) const
{
    //for discrete gpu, we only reserve,no commit yet.
    return amd::Os::reserveMemory(nullptr, size, alignment, amd::Os::MEM_PROT_NONE);
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
    amd::Memory* mem = nullptr;
    freeCPUMem_ = false;
    if (nullptr == svmPtr) {
        if (isFineGrainedSystem()) {
            freeCPUMem_ = true;
            return amd::Os::alignedMalloc(size, alignment);
        }

        //create a hidden buffer, which will allocated on the device later
        mem = new (context)amd::Buffer(context, flags, size, reinterpret_cast<void*>(1));
        if (mem == nullptr) {
            LogError("failed to create a svm mem object!");
            return nullptr;
        }

        if (!mem->create(nullptr, false)) {
            LogError("failed to create a svm hidden buffer!");
            mem->release();
            return nullptr;
        }
        //if the device supports SVM FGS, return the committed CPU address directly.
        pal::Memory* gpuMem = getGpuMemory(mem);

        //add the information to context so that we can use it later.
        amd::SvmManager::AddSvmBuffer(mem->getSvmPtr(), mem);
        svmPtr = mem->getSvmPtr();
    }
    else {
        //find the existing amd::mem object
        mem = amd::SvmManager::FindSvmBuffer(svmPtr);
        if (nullptr == mem) {
            return nullptr;
        }
        //commit the CPU memory for FGS device.
        if (isFineGrainedSystem()) {
            mem->commitSvmMemory();
        }
        else {
            pal::Memory* gpuMem = getGpuMemory(mem);
        }
        svmPtr = mem->getSvmPtr();
    }
    return svmPtr;
}

void
Device::svmFree(void *ptr) const
{
    if (freeCPUMem_) {
        amd::Os::alignedFree(ptr);
    }
    else {
        amd::Memory * svmMem = nullptr;
        svmMem = amd::SvmManager::FindSvmBuffer(ptr);
        if (nullptr != svmMem) {
            svmMem->release();
            amd::SvmManager::RemoveSvmBuffer(ptr);
        }
    }
}


Device::SrdManager::~SrdManager()
{
    for (uint i = 0; i < pool_.size(); ++i) {
        pool_[i].buf_->unmap(nullptr);
        delete pool_[i].buf_;
        delete pool_[i].flags_;
    }
}

bool
Sampler::create(uint32_t oclSamplerState)
{
    hwSrd_ = dev_.srds().allocSrdSlot(&hwState_);
    if (0 == hwSrd_) {
        return false;
    }
    dev_.fillHwSampler(oclSamplerState, hwState_, HsaSamplerObjectSize);
    return true;
}

bool
Sampler::create(const amd::Sampler& owner)
{
    hwSrd_ = dev_.srds().allocSrdSlot(&hwState_);
    if (0 == hwSrd_) {
        return false;
    }
    dev_.fillHwSampler(owner.state(), hwState_, HsaSamplerObjectSize,
        owner.mipFilter(), owner.minLod(), owner.maxLod());
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
    if (chunk.flags_ == nullptr) {
        return 0;
    }
    chunk.buf_ = new Memory(dev_, bufSize_);
    if (chunk.buf_ == nullptr || !chunk.buf_->create(Resource::Remote) ||
        (nullptr == chunk.buf_->map(nullptr))) {
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
    if (addr == 0) return;
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
Device::updateFreeMemory(Pal::GpuHeap heap, Pal::gpusize size, bool free)
{
    if (free) {
        freeMem[heap] += size;
    }
    else {
        freeMem[heap] -= size;
    }
}

void
Device::SrdManager::fillResourceList(std::vector<const Memory*>& memList)
{
    for (uint i = 0; i < pool_.size(); ++i) {
        memList.push_back(pool_[i].buf_);
    }
}

cl_int
Device::hwDebugManagerInit(amd::Context *context, uintptr_t messageStorage)
{
    cl_int status = hwDebugMgr_->registerDebugger(context, messageStorage);

    if (CL_SUCCESS != status) {
        delete hwDebugMgr_;
        hwDebugMgr_ = nullptr;
    }

    return status;
}

} // namespace pal
