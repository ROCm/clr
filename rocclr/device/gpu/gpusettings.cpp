/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpusettings.hpp"

#include <algorithm>

#if defined(_WIN32)
#include "VersionHelpers.h"
#endif

namespace gpu {

/*! \brief information for adjusting maximum workload time
 *
 *  This structure contains the time and OS minor version for max workload time
 *  adjustment for Windows 7 or 8.
 */
struct ModifyMaxWorkload {
  uint32_t time;          //!< max work load time (10x ms)
  uint32_t minorVersion;  //!< OS minor version
#if defined(_WIN32)
  BYTE comparisonOps;  //!< Comparison option
#endif
};


Settings::Settings() {
  // Initialize the GPU device default settings
  oclVersion_ = OpenCL12;
  debugFlags_ = 0;
  syncObject_ = GPU_USE_SYNC_OBJECTS;
  remoteAlloc_ = REMOTE_ALLOC;

  stagedXferRead_ = true;
  stagedXferWrite_ = true;
  stagedXferSize_ = GPU_STAGING_BUFFER_SIZE * Ki;

  // We will enable staged read/write if we use local memory
  disablePersistent_ = false;

  maxRenames_ = 16;
  maxRenameSize_ = 4 * Mi;

  imageSupport_ = false;
  hwLDSSize_ = 0;

  // Set this to true when we drop the flag
  doublePrecision_ = ::CL_KHR_FP64;

  // Fill workgroup info size
  maxWorkGroupSize_ = 1024;
  preferredWorkGroupSize_ = 256;

  hostMemDirectAccess_ = HostMemDisable;

  libSelector_ = amd::LibraryUndefined;

  // Enable workload split by default (for 24 bit arithmetic or timeout)
  workloadSplitSize_ = 1 << GPU_WORKLOAD_SPLIT;

  // By default use host blit
  blitEngine_ = BlitEngineHost;
  const static size_t MaxPinnedXferSize = 32;
  pinnedXferSize_ = std::min(GPU_PINNED_XFER_SIZE, MaxPinnedXferSize) * Mi;
  pinnedMinXferSize_ = std::min(GPU_PINNED_MIN_XFER_SIZE * Ki, pinnedXferSize_);

  // Disable FP_FAST_FMA defines by default
  reportFMAF_ = false;
  reportFMA_ = false;

  // GPU device by default
  apuSystem_ = false;

  // Disable 64 bit pointers support by default
  use64BitPtr_ = false;

  // Max alloc size is 16GB
  maxAllocSize_ = 16 * static_cast<uint64_t>(Gi);

  // Disable memory dependency tracking by default
  numMemDependencies_ = 0;

  // By default cache isn't present
  cacheLineSize_ = 0;
  cacheSize_ = 0;

  // Initialize transfer buffer size to 1MB by default
  xferBufSize_ = 1024 * Ki;

  // Use image DMA if requested
  imageDMA_ = GPU_IMAGE_DMA;

  // Disable ASIC specific features by default
  ciPlus_ = false;
  viPlus_ = false;
  aiPlus_ = false;

  // Number of compute rings.
  numComputeRings_ = 0;

  minWorkloadTime_ = 100;     // 0.1 ms
  maxWorkloadTime_ = 500000;  // 500 ms

  // Controls tiled images in persistent
  //!@note IOL for Linux doesn't setup tiling aperture in CMM/QS
  linearPersistentImage_ = false;

  useSingleScratch_ = GPU_USE_SINGLE_SCRATCH;

  // SDMA profiling is disabled by default
  sdmaProfiling_ = false;

  // Device enqueuing settings
  numDeviceEvents_ = 1024;
  numWaitEvents_ = 8;

  // Disable HSAIL by default
  hsail_ = false;

  // Don't support platform atomics by default.
  svmAtomics_ = false;

  // Use host queue for device enqueuing by default
  useDeviceQueue_ = GPU_USE_DEVICE_QUEUE;

  // Don't support Denormals for single precision by default
  singleFpDenorm_ = false;
}

bool Settings::create(const CALdeviceattribs& calAttr, bool reportAsOCL12Device,
                      bool smallMemSystem) {
  CALuint target = calAttr.target;
  uint32_t osVer = 0x0;

  // Disable thread trace by default for all devices
  threadTraceEnable_ = false;

  if (calAttr.doublePrecision) {
    // Report FP_FAST_FMA define if double precision HW
    reportFMA_ = true;
    // FMA is 1/4 speed on Pitcairn, Cape Verde, Devastator and Scrapper
    // Bonaire, Kalindi, Spectre and Spooky so disable
    // FP_FMA_FMAF for those parts in switch below
    reportFMAF_ = true;
  }

  // Update GPU specific settings and info structure if we have any
  ModifyMaxWorkload modifyMaxWorkload = {0};

  switch (target) {
    case CAL_TARGET_RAVEN:
    case CAL_TARGET_RAVEN2:
    case CAL_TARGET_RENOIR:
      // APU systems for AI
      apuSystem_ = true;
    case CAL_TARGET_GREENLAND:
    case CAL_TARGET_VEGA12:
    case CAL_TARGET_VEGA20:
      // TODO: specific codes for AI
      aiPlus_ = true;
    // Fall through to VI ...
    case CAL_TARGET_STONEY:
      if (!aiPlus_) {
        // Fix BSOD/TDR issues observed on Stoney Win7/8.1/10
        minWorkloadTime_ = 1000;
        modifyMaxWorkload.time = 1000;       // Decided by experiment
        modifyMaxWorkload.minorVersion = 1;  // Win 7
#if defined(_WIN32)
        modifyMaxWorkload.comparisonOps = VER_EQUAL;  // Limit to Win 7 only
#endif
      }
    case CAL_TARGET_CARRIZO:
      if (!aiPlus_) {
        // APU systems for VI
        apuSystem_ = true;
      }
    case CAL_TARGET_ICELAND:
    case CAL_TARGET_TONGA:
    case CAL_TARGET_FIJI:
    case CAL_TARGET_ELLESMERE:
    case CAL_TARGET_BAFFIN:
    case CAL_TARGET_LEXA:
    case CAL_TARGET_POLARIS22:
      // Disable tiling aperture on VI+
      linearPersistentImage_ = true;
      // Keep this false even though we have support
      // singleFpDenorm_ = true;
      viPlus_ = true;
      enableExtension(ClKhrFp16);
    // Fall through to CI ...
    case CAL_TARGET_KALINDI:
    case CAL_TARGET_SPECTRE:
    case CAL_TARGET_SPOOKY:
    case CAL_TARGET_GODAVARI:
      if (!viPlus_) {
        // APU systems for CI
        apuSystem_ = true;
        // Fix BSOD/TDR issues observed on Kaveri Win7 (EPR#416903)
        modifyMaxWorkload.time = 250000;     // 250ms
        modifyMaxWorkload.minorVersion = 1;  // Win 7
#if defined(_WIN32)
        modifyMaxWorkload.comparisonOps = VER_EQUAL;  // limit to Win 7
#endif
      }
    // Fall through ...
    case CAL_TARGET_BONAIRE:
    case CAL_TARGET_HAWAII:
      ciPlus_ = true;
      sdmaProfiling_ = true;
      hsail_ = GPU_HSAIL_ENABLE;
      threadTraceEnable_ = AMD_THREAD_TRACE_ENABLE;
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
      // Cache line size is 64 bytes
      cacheLineSize_ = 64;
      // L1 cache size is 16KB
      cacheSize_ = 16 * Ki;

      if (ciPlus_) {
        libSelector_ = amd::GPU_Library_CI;
        if (LP64_SWITCH(false, true)) {
          oclVersion_ = !reportAsOCL12Device && calAttr.isOpenCL200Device
              ? XCONCAT(OpenCL, XCONCAT(OPENCL_MAJOR, OPENCL_MINOR))
              : OpenCL12;
        }
        if (smallMemSystem) {  // force the dGPU to be 1.2 device for small memory system.
          if (apuSystem_) {
            return false;
          } else {
            oclVersion_ = OpenCL12;
          }
        }
        if (GPU_FORCE_OCL20_32BIT) {
          force32BitOcl20_ = true;
          oclVersion_ = !reportAsOCL12Device && calAttr.isOpenCL200Device
              ? XCONCAT(OpenCL, XCONCAT(OPENCL_MAJOR, OPENCL_MINOR))
              : OpenCL12;
        }
        if (OPENCL_VERSION < 200) {
          oclVersion_ = OpenCL12;
        }
        numComputeRings_ = 8;
      } else {
        numComputeRings_ = 2;
        libSelector_ = amd::GPU_Library_SI;
      }

      // Cap at OpenCL20 for now
      if (oclVersion_ > OpenCL20) oclVersion_ = OpenCL20;

      // This needs to be cleaned once 64bit addressing is stable
      if (oclVersion_ < OpenCL20) {
        use64BitPtr_ = flagIsDefault(GPU_FORCE_64BIT_PTR)
            ? LP64_SWITCH(false, calAttr.isWorkstation || hsail_)
            : GPU_FORCE_64BIT_PTR;
      } else {
        if (GPU_FORCE_64BIT_PTR || LP64_SWITCH(false, (hsail_ || (oclVersion_ >= OpenCL20)))) {
          use64BitPtr_ = true;
        }
      }

      if (oclVersion_ >= OpenCL20) {
        supportDepthsRGB_ = true;
      }
      if (use64BitPtr_) {
        if (GPU_ENABLE_LARGE_ALLOCATION && (viPlus_ || (oclVersion_ == OpenCL20))) {
          maxAllocSize_ = 64ULL * Gi;
        } else {
          maxAllocSize_ = 4048 * Mi;
        }
      } else {
        maxAllocSize_ = 3ULL * Gi;
      }

      supportRA_ = false;
      numMemDependencies_ = GPU_NUM_MEM_DEPENDENCY;

      enableExtension(ClKhrInt64BaseAtomics);
      enableExtension(ClKhrInt64ExtendedAtomics);
      enableExtension(ClKhrImage2dFromBuffer);
      break;
    default:
      assert(0 && "Unknown ASIC type!");
      return false;
  }

#if defined(_WIN32)
  if (modifyMaxWorkload.time > 0) {
    OSVERSIONINFOEX versionInfo = {0};
    versionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    versionInfo.dwMajorVersion = 6;
    versionInfo.dwMinorVersion = modifyMaxWorkload.minorVersion;

    DWORDLONG conditionMask = 0;
    VER_SET_CONDITION(conditionMask, VER_MAJORVERSION, modifyMaxWorkload.comparisonOps);
    VER_SET_CONDITION(conditionMask, VER_MINORVERSION, modifyMaxWorkload.comparisonOps);
    if (VerifyVersionInfo(&versionInfo, VER_MAJORVERSION | VER_MINORVERSION, conditionMask)) {
      maxWorkloadTime_ = modifyMaxWorkload.time;
    }
  }
  enableExtension(ClAMDLiquidFlash);
#endif  // defined(_WIN32)

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
  enableExtension(ClKhr3DImageWrites);
  enableExtension(ClAmdVec3);
  enableExtension(ClAmdPrintf);
  // Enable some platform extensions
  enableExtension(ClAmdDeviceAttributeQuery);
  enableExtension(ClKhrSpir);

  hwLDSSize_ = 32 * Ki;

  imageSupport_ = true;

  // Use kernels for blit if appropriate
  blitEngine_ = BlitEngineKernel;

  hostMemDirectAccess_ |= HostMemBuffer;
  // HW doesn't support untiled image writes
  // hostMemDirectAccess_ |= HostMemImage;

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
    // Enable bus addressable memory extension
    enableExtension(ClAMDBusAddressableMemory);
  }

  if (calAttr.longIdleDetect) {
    // KMD is unable to detect if we map the visible memory for CPU access, so
    // accessing persistent staged buffer may fail if LongIdleDetct is enabled.
    disablePersistent_ = true;
  }

  svmFineGrainSystem_ = calAttr.isSVMFineGrainSystem;

  svmAtomics_ = (calAttr.svmAtomics || calAttr.isSVMFineGrainSystem) ? true : false;

#if defined(_WIN32)
  enableExtension(ClKhrD3d9Sharing);
  enableExtension(ClKhrD3d10Sharing);
  enableExtension(ClKhrD3d11Sharing);
#endif  // _WIN32

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

#if defined(_WIN32)
    enableExtension(ClAmdPlanarYuv);
#endif
  }

  if (apuSystem_ && ((calAttr.totalVisibleHeap + calAttr.totalInvisibleHeap) < 150)) {
    remoteAlloc_ = true;
  }

// Save resource cache size
#ifdef ATI_OS_LINUX
  // Due to EPR#406216, set the default value for Linux for now
  resourceCacheSize_ = GPU_RESOURCE_CACHE_SIZE * Mi;
#else
  if (remoteAlloc_) {
    resourceCacheSize_ =
        std::max((calAttr.uncachedRemoteRAM / 8) * Mi, GPU_RESOURCE_CACHE_SIZE * Mi);
  } else {
    resourceCacheSize_ = std::max((calAttr.localRAM / 8) * Mi, GPU_RESOURCE_CACHE_SIZE * Mi);
  }
#endif

  // Override current device settings
  override();

  return true;
}

void Settings::override() {
  // Limit reported workgroup size
  if (GPU_MAX_WORKGROUP_SIZE != 0) {
    preferredWorkGroupSize_ = GPU_MAX_WORKGROUP_SIZE;
  }

  // Override blit engine type
  if (GPU_BLIT_ENGINE_TYPE != BlitEngineDefault) {
    blitEngine_ = GPU_BLIT_ENGINE_TYPE;
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

}  // namespace gpu
