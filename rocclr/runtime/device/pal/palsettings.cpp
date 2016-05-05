//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/pal/paldefs.hpp"
#include "device/pal/palsettings.hpp"

#include <algorithm>

namespace pal {

/*! \brief information for adjusting maximum workload time
 *
 *  This structure contains the time and OS minor version for max workload time
 *  adjustment for Windows 7 or 8.
 */
struct ModifyMaxWorkload
{
    uint32_t time;          //!< max work load time (10x ms)
    uint32_t minorVersion;  //!< OS minor version
};


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

    maxRenames_         = 4;
    maxRenameSize_      = 4 * Mi;

    imageSupport_       = false;
    hwLDSSize_          = 0;

    // Set this to true when we drop the flag
    doublePrecision_    = ::CL_KHR_FP64;

    // Fill workgroup info size
    // @todo: revisit the 256 limitation on workgroup size
    maxWorkGroupSize_   = 256;

    hostMemDirectAccess_  = HostMemDisable;

    libSelector_    = amd::LibraryUndefined;

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
    ciPlus_         = false;
    viPlus_         = false;
    aiPlus_         = false;

    // Number of compute rings.
    numComputeRings_ = 0;

    minWorkloadTime_ = 1;       // 0.1 ms
    maxWorkloadTime_ = 5000;    // 500 ms

    // Controls tiled images in persistent
    //!@note IOL for Linux doesn't setup tiling aperture in CMM/QS
    linearPersistentImage_ = false;

    useSingleScratch_ = GPU_USE_SINGLE_SCRATCH;

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

    // Don't support Denormals for single precision by default
    singleFpDenorm_ = false;
}

bool
Settings::create(
    const Pal::DeviceProperties& palProp,
    const Pal::GpuMemoryHeapProperties* heaps,
    bool reportAsOCL12Device
)
{
//    uint    target = calAttr.target;
    uint32_t osVer = 0x0;

    // Disable thread trace by default for all devices
    threadTraceEnable_ = false;
    bool doublePrecision = true;

    if (doublePrecision) {
        // Report FP_FAST_FMA define if double precision HW
        reportFMA_ = true;
        // FMA is 1/4 speed on Pitcairn, Cape Verde, Devastator and Scrapper
        // Bonaire, Kalindi, Spectre and Spooky so disable
        // FP_FMA_FMAF for those parts in switch below
        reportFMAF_ = true;
    }

    // Update GPU specific settings and info structure if we have any
    ModifyMaxWorkload modifyMaxWorkload = {0};

    switch (palProp.revision) {
/*    case Pal::AsicRevision:::
    case CAL_TARGET_GREENLAND:
        //TODO: specific codes for AI
        aiPlus_ = true;*/
        // Fall through to VI ...
    case Pal::AsicRevision::Carrizo:
    case Pal::AsicRevision::Stoney:
        if (!aiPlus_) {
            // APU systems for VI
            apuSystem_  = true;
        }
    case Pal::AsicRevision::Iceland:
    case Pal::AsicRevision::Tonga:
    case Pal::AsicRevision::Fiji:
    case Pal::AsicRevision::Ellesmere:
    case Pal::AsicRevision::Baffin:
        // Disable tiling aperture on VI+
        linearPersistentImage_ = true;
        // Keep this false even though we have support
        // singleFpDenorm_ = true;
        viPlus_ = true;
        // Fall through to CI ...
    case Pal::AsicRevision::Kalindi:
    case Pal::AsicRevision::Spectre:
        if (!viPlus_) {
            // APU systems for CI
            apuSystem_  = true;
            // Fix BSOD/TDR issues observed on Kaveri Win7 (EPR#416903)
            modifyMaxWorkload.time = 2500;      // 250ms
            modifyMaxWorkload.minorVersion = 1; // Win 7
        }
        // Fall through ...
    case Pal::AsicRevision::Bonaire:
    case Pal::AsicRevision::Hawaii:
        ciPlus_ = true;
        hsail_ = true;
        threadTraceEnable_ = AMD_THREAD_TRACE_ENABLE;
        reportFMAF_ = false;
        if (palProp.revision == Pal::AsicRevision::Hawaii) {
            reportFMAF_ = true;
        }
        // Cache line size is 64 bytes
        cacheLineSize_  = 64;
        // L1 cache size is 16KB
        cacheSize_      = 16 * Ki;

        if (ciPlus_) {
            libSelector_ = amd::GPU_Library_CI;
            if (LP64_SWITCH(WINDOWS_SWITCH(viPlus_, false), true)) {
                oclVersion_ = !reportAsOCL12Device /*&& calAttr.isOpenCL200Device*/ ?
                    XCONCAT(OpenCL, XCONCAT(OPENCL_MAJOR, OPENCL_MINOR)) : OpenCL12;
            }
            if (GPU_FORCE_OCL20_32BIT) {
                force32BitOcl20_ = true;
                oclVersion_ = !reportAsOCL12Device /*&& calAttr.isOpenCL200Device*/ ?
                    XCONCAT(OpenCL, XCONCAT(OPENCL_MAJOR, OPENCL_MINOR)) : OpenCL12;
            }
            if (OPENCL_VERSION < 200) {
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
                /*calAttr.isWorkstation ||*/ hsail_) : GPU_FORCE_64BIT_PTR;
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
            if (GPU_ENABLE_LARGE_ALLOCATION /*&& calAttr.isWorkstation*/) {
                maxAllocSize_   = 64ULL * Gi;
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
        break;
    default:
        assert(0 && "Unknown ASIC type!");
        return false;
    }

    // Enable atomics support
    enableExtension(ClKhrInt64BaseAtomics);
    enableExtension(ClKhrInt64ExtendedAtomics);
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
    enableExtension(ClKhr3DImageWrites);
    enableExtension(ClAmdVec3);
    enableExtension(ClAmdPrintf);
    enableExtension(ClKhrImage2dFromBuffer);

    hwLDSSize_      = 32 * Ki;

    imageSupport_       = true;
    singleHeap_         = true;

    // Use kernels for blit if appropriate
    blitEngine_     = BlitEngineKernel;

    hostMemDirectAccess_ |= HostMemBuffer;
    // HW doesn't support untiled image writes
    // hostMemDirectAccess_ |= HostMemImage;

    // Make sure device actually supports double precision
    doublePrecision_ = (doublePrecision) ? doublePrecision_ : false;
    if (doublePrecision_) {
        // Enable KHR double precision extension
        enableExtension(ClKhrFp64);
    }

    if (doublePrecision) {
        // Enable AMD double precision extension
        doublePrecision_ = true;
        enableExtension(ClAmdFp64);
    }

//! @todo 
/*
    if (calAttr.totalSDIHeap > 0) {
        //Enable bus addressable memory extension
        enableExtension(ClAMDBusAddressableMemory);
    }

    if (calAttr.longIdleDetect) {
        // KMD is unable to detect if we map the visible memory for CPU access, so
        // accessing persistent staged buffer may fail if LongIdleDetct is enabled.
        disablePersistent_ = true;
    }

    svmFineGrainSystem_ = calAttr.isSVMFineGrainSystem;

    svmAtomics_ = (calAttr.svmAtomics || calAttr.isSVMFineGrainSystem) ? true : false;
*/
    // Enable some platform extensions
    enableExtension(ClAmdDeviceAttributeQuery);

    enableExtension(ClKhrSpir);

    // SVM is not currently supported for DX Interop
#if defined(_WIN32)
    enableExtension(ClKhrD3d9Sharing);
    enableExtension(ClKhrD3d10Sharing);
    enableExtension(ClKhrD3d11Sharing);
#endif // _WIN32

    // Enable some OpenCL 2.0 extensions
    if (oclVersion_ >= OpenCL20) {
        enableExtension(ClKhrGLDepthImages);
        enableExtension(ClKhrSubGroups);
        enableExtension(ClKhrDepthImages);

        if (GPU_MIPMAP) {
            enableExtension(ClKhrMipMapImage);
            enableExtension(ClKhrMipMapImageWrites);
        }

        // Enable HW debug
        if (GPU_ENABLE_HW_DEBUG) {
            enableHwDebug_ = true;
        }
    }


    if (apuSystem_ &&
       ((heaps[Pal::GpuHeapLocal].heapSize + heaps[Pal::GpuHeapInvisible].heapSize) < (150*Mi))) {
        remoteAlloc_ = true;
    }

    // Save resource cache size
#ifdef ATI_OS_LINUX
    // Due to EPR#406216, set the default value for Linux for now
    resourceCacheSize_ = GPU_RESOURCE_CACHE_SIZE * Mi;
#else
    if (remoteAlloc_) {
        resourceCacheSize_ = std::max((heaps[Pal::GpuHeapGartUswc].heapSize / 8),
            (uint64_t)GPU_RESOURCE_CACHE_SIZE * Mi);
    }
    else {
        resourceCacheSize_ = std::max(((heaps[Pal::GpuHeapLocal].heapSize +
            heaps[Pal::GpuHeapInvisible].heapSize) / 8),
            (uint64_t)GPU_RESOURCE_CACHE_SIZE * Mi);
    }
    resourceCacheSize_ = std::min(resourceCacheSize_, 512 * Mi);
#endif

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

    if (!flagIsDefault(GPU_RESOURCE_CACHE_SIZE)) {
        resourceCacheSize_ = GPU_RESOURCE_CACHE_SIZE * Mi;
    }

    if (!flagIsDefault(AMD_GPU_FORCE_SINGLE_FP_DENORM)) {
        switch (AMD_GPU_FORCE_SINGLE_FP_DENORM) {
        case 0:
            singleFpDenorm_ = false;
            break;
        case 1:
            singleFpDenorm_ = true;
            break;
        default:
            break;
        }
    }
}

} // namespace pal
