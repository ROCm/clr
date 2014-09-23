//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpusettings.hpp"

#include <algorithm>

namespace gpu {

Settings::Settings()
{
    // Initialize the GPU device default settings
    oclVersion_         = OpenCL12;
    debugFlags_         = 0;
    singleHeap_         = false;
    syncObject_         = GPU_USE_SYNC_OBJECTS;
    remoteAlloc_        = REMOTE_ALLOC;

    stagedXferRead_   = true;
    stagedXferWrite_  = true;
    stagedXferSize_   = GPU_STAGING_BUFFER_SIZE * Ki;

    // We will enable staged read/write if we use local memory
    disablePersistent_ = false;

    // By Default persistent writes will be disabled.
    stagingWritePersistent_ = GPU_STAGING_WRITE_PERSISTENT;

    maxRenames_         = 32;
    maxRenameSize_      = 4 * Mi;

    // The global heap settings
    heapSize_           = GPU_INITIAL_HEAP_SIZE * Mi;
    heapSizeGrowth_     = GPU_HEAP_GROWTH_INCREMENT * Mi;

    useAliases_         = false;
    imageSupport_       = false;
    hwLDSSize_          = 0;

    // Set this to true when we drop the flag
    doublePrecision_    = ::CL_KHR_FP64;

    // Fill workgroup info size
    // @todo: revisit the 256 limitation on workgroup size
    maxWorkGroupSize_   = 256;

    hostMemDirectAccess_  = HostMemDisable;

    libSelector_    = amd::LibraryUndefined;

#if cl_amd_open_video
    // By default Open Video extension is not yet supported
    openVideo_ = false;
#endif // cl_amd_open_video

    // Enable workload split by default (for 24 bit arithmetic or timeout)
    workloadSplitSize_  = 1 << GPU_WORKLOAD_SPLIT;

    // By default use host blit
    blitEngine_         = BlitEngineHost;
    const static size_t MaxPinnedXferSize = 32;
    pinnedXferSize_     = std::min(GPU_PINNED_XFER_SIZE, MaxPinnedXferSize) * Mi;
    pinnedMinXferSize_  = std::min(GPU_PINNED_MIN_XFER_SIZE * Ki, pinnedXferSize_);

    // Disable FP_FAST_FMA defines by default
    reportFMAF_ = false;
    reportFMA_  = false;

    // Disable async memory transfers by default
    asyncMemCopy_ = false;

    // GPU device by default
    apuSystem_  = false;

    // Disable 64 bit pointers support by default
    use64BitPtr_ = false;

    // Max alloc size is 16GB
    maxAllocSize_ = 16 * static_cast<uint64_t>(Gi);

    // Disable memory dependency tracking by default
    numMemDependencies_ = 0;

    // By default cache isn't present
    cacheLineSize_  = 0;
    cacheSize_      = 0;

    // Initialize transfer buffer size to 1MB by default
    xferBufSize_    = 1024 * Ki;

    // Use image DMA if requested
    imageDMA_       = GPU_IMAGE_DMA;

    // Disable ASIC specific features by default
    siPlus_         = false;
    ciPlus_         = false;
    viPlus_         = false;

    // Number of compute rings.
    numComputeRings_ = 0;

    // Rectangular Linear DRMDMA
    rectLinearDMA_  = false;

    minWorkloadTime_ = 1;       // 0.1 ms
    maxWorkloadTime_ = 5000;    // 500 ms

    // Preallocates address space
    preallocAddrSpace_ = GPU_PREALLOC_ADDR_SPACE;

    // Controls tiled images in persistent
    //!@note IOL for Linux doesn't setup tiling aperture in CMM/QS
    linearPersistentImage_ = false;

    useSingleScratch_ = GPU_USE_SINGLE_SCRATCH;

    // SDMA profiling is disabled by default
    sdmaProfiling_  = false;

    // Device enqueuing settings
    numDeviceEvents_ = 1024;
    numWaitEvents_   = 8;

    // Disable HSAIL by default
    hsail_ = false;

    // Don't support platform atomics by default.
    svmAtomics_ = false;

    // Use direct SRD by default
    hsailDirectSRD_ = GPU_DIRECT_SRD;

    // Use host queue for device enqueuing by default
    useDeviceQueue_ = GPU_USE_DEVICE_QUEUE;
}

bool
Settings::create(
    const CALdeviceattribs& calAttr
#if cl_amd_open_video
  , const CALdeviceVideoAttribs& calVideoAttr
#endif // cl_amd_open_video
  , bool reportAsOCL12Device
)
{
    CALuint target = calAttr.target;
    uint32_t osVer = 0x0;

    // Disable thread trace by default for all devices
    threadTraceEnable_ = false;

    // Save resource cache size
#ifdef ATI_OS_LINUX
    // Due to EPR#406216, set the default value for Linux for now
    resourceCacheSize_ = GPU_RESOURCE_CACHE_SIZE * Mi;
#else
    resourceCacheSize_ = std::max((calAttr.localRAM / 8) * Mi, GPU_RESOURCE_CACHE_SIZE * Mi);
#endif

    if (calAttr.doublePrecision) {
        // Report FP_FAST_FMA define if double precision HW
        reportFMA_ = true;
        // FMA is 1/4 speed on Pitcairn, Cape Verde, Devastator and Scrapper
        // Bonaire, Kalindi, Spectre and Spooky so disable
        // FP_FMA_FMAF for those parts in switch below
        reportFMAF_ = true;
    }

    // Update GPU specific settings and info structure if we have any
    switch (target) {
    case CAL_TARGET_SUMO:
    case CAL_TARGET_SUPERSUMO:
    case CAL_TARGET_WRESTLER:
        // Treat these like Evergreen parts as far as capabilities go
        // Fall through ...
    case CAL_TARGET_DEVASTATOR:
    case CAL_TARGET_SCRAPPER:
        apuSystem_  = true;
        reportFMAF_ = false;
        // For the system that has APU and Win 8, the work load needs to be smaller
        // This is because KMD doesn't have workaround for TDR in Win 8
        // This is needed only for EG/NI because EG/NI is using graphics ring
#if defined(_WIN32)
        {
            OSVERSIONINFOEX versionInfo = { 0 };
            versionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
            versionInfo.dwMajorVersion = 6;
            versionInfo.dwMinorVersion = 2;

            DWORDLONG conditionMask = 0;
            VER_SET_CONDITION(conditionMask, VER_MAJORVERSION, VER_GREATER_EQUAL);
            VER_SET_CONDITION(conditionMask, VER_MINORVERSION, VER_GREATER_EQUAL);
            if (VerifyVersionInfo(&versionInfo, VER_MAJORVERSION | VER_MINORVERSION, conditionMask)) {
                maxWorkloadTime_ = 500;    // 50 ms
            }
        }
#endif // defined(_WIN32)
        // Add the caps for Trinity here ...
        // Fall through ...
    case CAL_TARGET_CAYMAN:
        // Add the caps for Cayman here ...
    case CAL_TARGET_KAUAI:
    case CAL_TARGET_BARTS:
    case CAL_TARGET_TURKS:
    case CAL_TARGET_CAICOS:
        // Treat these like Evergreen parts as far as capabilities go
        // Fall through ...
    case CAL_TARGET_CYPRESS:
    case CAL_TARGET_JUNIPER:
    case CAL_TARGET_REDWOOD:
    case CAL_TARGET_CEDAR:
        // UAV arena is a pre-SI specific HW feature
        useAliases_ = true;

        if (CAL_TARGET_CEDAR == target) {
            // Workaround for SC spill bugs.
            maxWorkGroupSize_   = 128;
        }
        // Get the link library
        libSelector_ = amd::GPU_Library_Evergreen;

        // Max alloc size
        maxAllocSize_ = 512 * Mi;

        if ((target == CAL_TARGET_CAYMAN) ||
            (target == CAL_TARGET_DEVASTATOR) ||
            (target == CAL_TARGET_SCRAPPER)) {
            rectLinearDMA_  = true;
        }

        // Disable KHR_FP64 for Trinity in the mainline
        if ((target == CAL_TARGET_DEVASTATOR) ||
            (target == CAL_TARGET_SCRAPPER)) {
            doublePrecision_ &= !IS_MAINLINE || !flagIsDefault(CL_KHR_FP64);
        }

        if (target == CAL_TARGET_CYPRESS) {
            // Float FMA is slower than "multiply + add" because we combine
            // "multiply + add" into mad.  MAD is 25% faster than FMA on Cypress,
            // assuming perfect VLIW packing.
            reportFMAF_ = false;
        }
        enableExtension(ClAmdImage2dFromBufferReadOnly);
        break;
    case CAL_TARGET_CARRIZO:
        apuSystem_ = true;
    case CAL_TARGET_ICELAND:
    case CAL_TARGET_TONGA:
    case CAL_TARGET_BERMUDA:
    case CAL_TARGET_FIJI:
        // Disable tiling aperture on VI+
        linearPersistentImage_ = true;
        viPlus_ = true;
        // Fall through to CI ...
    case CAL_TARGET_KALINDI:
    case CAL_TARGET_SPECTRE:
    case CAL_TARGET_SPOOKY:
    case CAL_TARGET_GODAVARI:
        if (!viPlus_) {
            // APU systems for CI
            apuSystem_  = true;
        }
        // Fall through ...
    case CAL_TARGET_BONAIRE:
    case CAL_TARGET_HAWAII:
        ciPlus_ = true;
        sdmaProfiling_ = true;
        hsail_ = GPU_HSAIL_ENABLE;
        // Fall through to SI ...
    case CAL_TARGET_PITCAIRN:
    case CAL_TARGET_CAPEVERDE:
    case CAL_TARGET_OLAND:
    case CAL_TARGET_HAINAN:
        reportFMAF_ = false;
        if (target == CAL_TARGET_HAWAII) {
            reportFMAF_ = true;
        }
        // Fall through ...
    case CAL_TARGET_TAHITI:
        siPlus_ = true;
        // Cache line size is 64 bytes
        cacheLineSize_  = 64;
        // L1 cache size is 16KB
        cacheSize_      = 16 * Ki;

        if (ciPlus_) {
            libSelector_ = amd::GPU_Library_CI;
#if defined(_LP64)
            oclVersion_ = reportAsOCL12Device ? OpenCL12 : XCONCAT(OpenCL,XCONCAT(OPENCL_MAJOR,OPENCL_MINOR));
#endif
            if (calAttr.numOfVpu > 1) {
                oclVersion_ = OpenCL12;
            }
            if (GPU_FORCE_OCL20_32BIT) {
                force32BitOcl20_ = true;
                oclVersion_ = reportAsOCL12Device ? OpenCL12 : XCONCAT(OpenCL,XCONCAT(OPENCL_MAJOR,OPENCL_MINOR));
            }
            if (hsail_ || (OPENCL_VERSION < 200)) {
                oclVersion_ = OpenCL12;
            }
            if (target == CAL_TARGET_CARRIZO) {
                oclVersion_ = OpenCL12;
            }
            numComputeRings_ = 8;
        }
        else {
            numComputeRings_ = 2;
            libSelector_ = amd::GPU_Library_SI;
        }

        // This needs to be cleaned once 64bit addressing is stable
        if (oclVersion_ < OpenCL20) {
            use64BitPtr_ = flagIsDefault(GPU_FORCE_64BIT_PTR) ? LP64_SWITCH(false,
                calAttr.isWorkstation || hsail_) : GPU_FORCE_64BIT_PTR;
        }
        else {
            if (GPU_FORCE_64BIT_PTR || LP64_SWITCH(false, (hsail_ 
                || (oclVersion_ >= OpenCL20)))) {
                use64BitPtr_    = true;
            }
        }

        if (oclVersion_ >= OpenCL20) {
            supportDepthsRGB_ = true;
        }
        if (use64BitPtr_) {
            if (GPU_ENABLE_LARGE_ALLOCATION) {
                maxAllocSize_   = 16ULL * Gi;
            }
            else {
                maxAllocSize_   = 4048 * Mi;
            }
        }
        else {
            maxAllocSize_   = 3ULL * Gi;
        }

        supportRA_  = false;
        partialDispatch_    = GPU_PARTIAL_DISPATCH;
        numMemDependencies_ = GPU_NUM_MEM_DEPENDENCY;

        //! @todo HSAIL doesn't support 64 bit atomic on 32 bit!
        if (LP64_SWITCH(!hsail_, true)) {
            enableExtension(ClKhrInt64BaseAtomics);
            enableExtension(ClKhrInt64ExtendedAtomics);
        }
        enableExtension(ClKhrImage2dFromBuffer);

        rectLinearDMA_  = true;

        if (AMD_DEPTH_MSAA_INTEROP) {
            enableExtension(ClKhrGLDepthImages);
            depthMSAAInterop_ = true;
        }
        if (AMD_THREAD_TRACE_ENABLE) {
            threadTraceEnable_ = true;
        }
        // Disable non-aliased(multiUAV) optimization
        assumeAliases_ = true;
        break;
    default:
        assert(0 && "Unknown ASIC type!");
        return false;
    }

    // Enable atomics support
    enableExtension(ClKhrGlobalInt32BaseAtomics);
    enableExtension(ClKhrGlobalInt32ExtendedAtomics);
    enableExtension(ClKhrLocalInt32BaseAtomics);
    enableExtension(ClKhrLocalInt32ExtendedAtomics);
    enableExtension(ClKhrByteAddressableStore);
    enableExtension(ClKhrGlSharing);
    enableExtension(ClKhrGlEvent);
    enableExtension(ClAmdMediaOps);
    enableExtension(ClAmdMediaOps2);
    enableExtension(ClAmdPopcnt);
#if defined(_WIN32)
    enableExtension(ClKhrD3d9Sharing);
    enableExtension(ClKhrD3d10Sharing);
    enableExtension(ClKhrD3d11Sharing);
#endif // _WIN32
    enableExtension(ClKhr3DImageWrites);
    enableExtension(ClAmdVec3);
    enableExtension(ClAmdPrintf);
    enableExtension(ClExtAtomicCounters32);

    hwLDSSize_      = 32 * Ki;

    imageSupport_       = true;
    singleHeap_         = true;
    customSvmAllocator_ = true;

    // Use kernels for blit if appropriate
    blitEngine_     = BlitEngineKernel;

#if cl_amd_open_video
    // Enable OpenVideo by default if CAL supports it
    if (calVideoAttr.max_decode_sessions > 0) {
        openVideo_ = true;
        if (GPU_OPEN_VIDEO)
            enableExtension(ClAmdOpenVideo);
    }
#endif // cl_amd_open_video

    hostMemDirectAccess_ |= HostMemBuffer;
    // HW doesn't support untiled image writes
    // hostMemDirectAccess_ |= HostMemImage;

    asyncMemCopy_ = true;

    // Make sure device actually supports double precision
    doublePrecision_ = (calAttr.doublePrecision) ? doublePrecision_ : false;
    if (doublePrecision_) {
        // Enable KHR double precision extension
        enableExtension(ClKhrFp64);
    }

    if (calAttr.doublePrecision) {
        // Enable AMD double precision extension
        doublePrecision_ = true;
        enableExtension(ClAmdFp64);
    }

    if (calAttr.totalSDIHeap > 0) {
        //Enable bus addressable memory extension
        enableExtension(ClAMDBusAddressableMemory);
    }

    if (calAttr.longIdleDetect) {
        // KMD is unable to detect if we map the visible memory for CPU access, so
        // accessing persistent staged buffer may fail if LongIdleDetct is enabled.
        disablePersistent_ = true;
    }

    if (calAttr.priSupport) {
        svmAtomics_ = true;
    }

    // Enable some platform extensions
    enableExtension(ClAmdDeviceAttributeQuery);

    enableExtension(ClKhrSpir);

    // Enable some OpenCL 2.0 extensions
    if (oclVersion_ >= OpenCL20) {
        enableExtension(ClKhrSubGroups);
        enableExtension(ClKhrDepthImages);
    }

    // Override current device settings
    override();

    return true;
}

void
Settings::override()
{
    // Limit reported workgroup size
    if (GPU_MAX_WORKGROUP_SIZE != 0) {
        maxWorkGroupSize_ = GPU_MAX_WORKGROUP_SIZE;
    }

    // Override blit engine type
    if (GPU_BLIT_ENGINE_TYPE != BlitEngineDefault) {
        blitEngine_ = GPU_BLIT_ENGINE_TYPE;
    }

    if (!flagIsDefault(DEBUG_GPU_FLAGS)) {
        debugFlags_ = DEBUG_GPU_FLAGS;
    }

    // Check async memory transfer
    if (!flagIsDefault(GPU_ASYNC_MEM_COPY)) {
        asyncMemCopy_ = GPU_ASYNC_MEM_COPY;
    }

    if (!flagIsDefault(DEBUG_GPU_FLAGS)) {
        debugFlags_ = DEBUG_GPU_FLAGS;
    }

    if (!flagIsDefault(GPU_XFER_BUFFER_SIZE)) {
        xferBufSize_ = GPU_XFER_BUFFER_SIZE * Ki;
    }

    if (!flagIsDefault(GPU_USE_SYNC_OBJECTS)) {
        syncObject_ = GPU_USE_SYNC_OBJECTS;
    }

    if (!flagIsDefault(GPU_NUM_COMPUTE_RINGS)) {
        numComputeRings_ = GPU_NUM_COMPUTE_RINGS;
    }

    if (!flagIsDefault(GPU_ASSUME_ALIASES)) {
        assumeAliases_ = GPU_ASSUME_ALIASES;
    }
    if (!flagIsDefault(GPU_RESOURCE_CACHE_SIZE)) {
        resourceCacheSize_ = GPU_RESOURCE_CACHE_SIZE * Mi;
    }
}

} // namespace gpu
