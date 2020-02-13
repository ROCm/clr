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

#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpumemory.hpp"
#include "device/gpu/gpudevice.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"
#include "utils/options.hpp"
#include "thread/monitor.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpubinary.hpp"
#include "device/gpu/gpusettings.hpp"
#include "device/gpu/gpublit.hpp"
#include "cz_id.h"

#include "acl.h"

#include "vdi_common.hpp"
#include "CL/cl_gl.h"

#ifdef _WIN32
#include <d3d9.h>
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#include "CL/cl_dx9_media_sharing.h"
#endif  // _WIN32

#include "os_if.h"  // for osInit()

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctype.h>
#include <algorithm>

#include "gpudebugmanager.hpp"

bool DeviceLoad() {
  bool ret = false;

  // Create online devices
  ret |= gpu::Device::init();
  // Create offline GPU devices
  ret |= gpu::NullDevice::init();

  return ret;
}

void DeviceUnload() { gpu::Device::tearDown(); }

namespace gpu {

aclCompiler* NullDevice::compiler_;
aclCompiler* NullDevice::hsaCompiler_;
AppProfile Device::appProfile_;

NullDevice::NullDevice()
    : amd::Device(), calTarget_(static_cast<CALtarget>(0)), hwInfo_(NULL) {}

bool NullDevice::init() {
  std::vector<Device*> devices;

  devices = getDevices(CL_DEVICE_TYPE_GPU, false);

  // Loop through all supported devices and create each of them
  for (uint id = CAL_TARGET_TAHITI; id <= CAL_TARGET_LAST; ++id) {
    bool foundActive = false;
    bool foundDuplicate = false;

    if (gpu::DeviceInfo[id].targetName_[0] == '\0') {
      continue;
    }

    // Loop through all active devices and see if we match one
    for (uint i = 0; i < devices.size(); ++i) {
      if (static_cast<NullDevice*>(devices[i])->calTarget() == static_cast<CALtarget>(id)) {
        foundActive = true;
        break;
      }
    }

    // Don't report an offline device if it's active
    if (foundActive) {
      continue;
    }

    // Loop through all previous devices in the DeviceInfo list and compare them with the
    // current entry to see if the current entry was listed previously in the DeviceInfo,
    // if so, then it means the current entry already has been added in the offline device list
    for (uint j = 0; j < id; ++j) {
      if (gpu::DeviceInfo[j].targetName_[0] == '\0') {
        continue;
      }
      if (strcmp(gpu::DeviceInfo[j].targetName_, gpu::DeviceInfo[id].targetName_) == 0) {
        foundDuplicate = true;
        break;
      }
    }

    // Don't report an offline device twice
    if (foundDuplicate) {
        continue;
    }

    NullDevice* dev = new NullDevice();
    if (NULL != dev) {
      if (!dev->create(static_cast<CALtarget>(id))) {
        delete dev;
      } else {
        dev->registerDevice();
      }
    }
  }

  return true;
}

bool NullDevice::create(CALtarget target) {
  CALdeviceattribs calAttr = {0};
  gslMemInfo memInfo = {0};

  online_ = false;

  calTarget_ = calAttr.target = target;
  hwInfo_ = &DeviceInfo[calTarget_];

  assert((target >= CAL_TARGET_TAHITI) && (target != CAL_TARGET_SCRAPPER) &&
         (target != CAL_TARGET_DEVASTATOR));

  if ((GPU_ENABLE_PAL == 2) && usePal()) {
    return false;
  }

  // Force double if it could be supported
  switch (target) {
    case CAL_TARGET_PITCAIRN:
    case CAL_TARGET_CAPEVERDE:
    case CAL_TARGET_TAHITI:
    case CAL_TARGET_OLAND:
    case CAL_TARGET_HAINAN:
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
    case CAL_TARGET_GREENLAND:
    case CAL_TARGET_STONEY:
    case CAL_TARGET_LEXA:
    case CAL_TARGET_RAVEN:
    case CAL_TARGET_RAVEN2:
    case CAL_TARGET_RENOIR:
    case CAL_TARGET_POLARIS22:
    case CAL_TARGET_VEGA12:
    case CAL_TARGET_VEGA20:
      calAttr.doublePrecision = CAL_TRUE;
      calAttr.isOpenCL200Device = CAL_TRUE;
      break;
    default:
      break;
  }

  settings_ = new gpu::Settings();
  gpu::Settings* gpuSettings = reinterpret_cast<gpu::Settings*>(settings_);
  // Create setting for the offline target
  if ((gpuSettings == NULL) || !gpuSettings->create(calAttr)) {
    return false;
  }

  // Report 512MB for all offline devices
  memInfo.cardMemAvailableBytes = 512 * Mi;
  memInfo.cardLargestFreeBlockBytes = 512 * Mi;
  calAttr.localRAM = 512;

  // Fill the device info structure
  fillDeviceInfo(calAttr, memInfo, 4096, 1, 0);

  if (NULL == compiler_) {
#if !defined(ATI_OS_LINUX)
    char CompilerLibrary[220] = "";
    strcat_s(CompilerLibrary, "amdocl12cl" LP64_SWITCH("", "64") ".dll");
#endif
    const char* library = getenv("COMPILER_LIBRARY");
    aclCompilerOptions opts = {
      sizeof(aclCompilerOptions_0_8),
#if defined(ATI_OS_LINUX)
      library ? library : LINUX_ONLY("lib") "amdocl12cl" LP64_SWITCH(LINUX_SWITCH("32", ""), "64")
                              LINUX_SWITCH(".so", ".dll"),
#else
      library ? library : CompilerLibrary,
#endif
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      AMD_OCL_SC_LIB
    };
    compiler_ = aclCompilerInit(&opts, NULL);
  }

  if (settings().hsail_ || (settings().oclVersion_ >= OpenCL20)) {
    // Runtime doesn't know what local size could be on the real board
    info_.maxGlobalVariableSize_ = static_cast<size_t>(512 * Mi);

    if (NULL == hsaCompiler_) {
      const char* library = getenv("HSA_COMPILER_LIBRARY");
      aclCompilerOptions opts = {
          sizeof(aclCompilerOptions_0_8), library, NULL, NULL, NULL, NULL, NULL, AMD_OCL_SC_LIB};
      // Initialize the compiler handle
      acl_error error;
      hsaCompiler_ = aclCompilerInit(&opts, &error);
      if (error != ACL_SUCCESS) {
        LogError("Error initializing the compiler");
        return false;
      }
    }
  }

  return true;
}

bool NullDevice::isHsailProgram(amd::option::Options* options) {
  bool isCIPlus = settings().ciPlus_;
  bool isHSAILcapable = settings().hsail_;
  bool isBlit = false;
  bool isSPIRV = false;
  bool isClang = false;
  bool isEDG = false;
  bool isLegacy = false;
  bool isOCL20 = false;
  std::vector<amd::option::Options*> optvec;
  bool isInputOptions = false;
  if (options != NULL) {
    optvec.push_back(options);
    isInputOptions = true;
  }
  amd::option::Options parsedOptions;
  constexpr bool OptionChangable = true;
  constexpr bool LinkOptsOnly = false;
  constexpr bool IsLC = false;
  if (!amd::Program::ParseAllOptions("", parsedOptions, OptionChangable, LinkOptsOnly, IsLC)) {
    return NULL;
  }
  optvec.push_back(&parsedOptions);
  for (auto const op : optvec) {
    // TODO: Remove isOCL20 related code from this function along with switching HSAIL by default
    if (isCIPlus && amd::Program::GetOclCVersion(op->oVariables->CLStd) >= 20) {
      isOCL20 = true;
    }
    if (op->oVariables->clInternalKernel) {
      isBlit = true;
      break;
    }
    if (!isLegacy) {
      isLegacy = op->oVariables->Legacy;
    }
    // Checks Frontend option only from input *options, not from Env,
    // because they might be only calculated by RT based on the binaries to link.
    // -frontend is being queried now instead of -cl-std=CL2.0, because the last one
    // is not an indicator for HSAIL path anymore.
    // TODO: Revise these binary's target checks
    // and possibly remove them after switching to HSAIL by default.
    if (isInputOptions) {
      if (!isClang) {
        isClang = op->isCStrOptionsEqual(op->oVariables->Frontend, "clang");
      }
      if (!isEDG) {
        isEDG = op->isCStrOptionsEqual(op->oVariables->Frontend, "edg");
      }
    }
    if (!isSPIRV) {
      isSPIRV = op->oVariables->BinaryIsSpirv;
    }
    isInputOptions = false;
  }
  if (isSPIRV || (isBlit && isCIPlus && isHSAILcapable) || isClang || isOCL20) {
    return true;
  }
  if (isLegacy || !isHSAILcapable || isEDG) {
    return false;
  }
  return true;
}

device::Program* NullDevice::createProgram(amd::Program& owner, amd::option::Options* options) {
  if (isHsailProgram(options)) {
    return new HSAILProgram(*this, owner);
  }
  return new NullProgram(*this, owner);
}

void NullDevice::fillDeviceInfo(const CALdeviceattribs& calAttr, const gslMemInfo& memInfo,
                                size_t maxTextureSize, uint numComputeRings,
                                uint numComputeRingsRT) {
  info_.type_ = CL_DEVICE_TYPE_GPU;
  info_.vendorId_ = 0x1002;
  info_.maxComputeUnits_ = calAttr.numberOfSIMD;
  info_.maxWorkItemDimensions_ = 3;
  info_.numberOfShaderEngines = calAttr.numberOfShaderEngines;

  // SI parts are scalar.  Also, reads don't need to be 128-bits to get peak rates.
  // For example, float4 is not faster than float as long as all threads fetch the same
  // amount of data and the reads are coalesced.  This is from the H/W team and confirmed
  // through experimentation.  May also be true on EG/NI, but no point in confusing
  // developers now.
  info_.nativeVectorWidthChar_ = info_.preferredVectorWidthChar_ = 4;
  info_.nativeVectorWidthShort_ = info_.preferredVectorWidthShort_ = 2;
  info_.nativeVectorWidthInt_ = info_.preferredVectorWidthInt_ = 1;
  info_.nativeVectorWidthLong_ = info_.preferredVectorWidthLong_ = 1;
  info_.nativeVectorWidthFloat_ = info_.preferredVectorWidthFloat_ = 1;
  info_.nativeVectorWidthDouble_ = info_.preferredVectorWidthDouble_ =
      (settings().checkExtension(ClKhrFp64)) ? 1 : 0;
  info_.nativeVectorWidthHalf_ = info_.preferredVectorWidthHalf_ = 0;  // no half support

  info_.maxEngineClockFrequency_ = (calAttr.engineClock != 0) ? calAttr.engineClock : 555;
  info_.maxMemoryClockFrequency_ = (calAttr.memoryClock != 0) ? calAttr.memoryClock : 555;
  info_.timeStampFrequency_ = 1000000;
  info_.vramBusBitWidth_ = calAttr.memBusWidth;
  info_.l2CacheSize_ = 0;
  info_.maxParameterSize_ = 1024;
  info_.minDataTypeAlignSize_ = sizeof(int64_t[16]);
  info_.singleFPConfig_ =
      CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

  if (settings().singleFpDenorm_) {
    info_.singleFPConfig_ |= CL_FP_DENORM;
  }

  if (settings().checkExtension(ClKhrFp64)) {
    info_.doubleFPConfig_ = info_.singleFPConfig_ | CL_FP_DENORM;
  }

  if (settings().reportFMA_) {
    info_.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  }

  info_.globalMemCacheLineSize_ = settings().cacheLineSize_;
  info_.globalMemCacheSize_ = settings().cacheSize_;
  if ((settings().cacheLineSize_ != 0) || (settings().cacheSize_ != 0)) {
    info_.globalMemCacheType_ = CL_READ_WRITE_CACHE;
  } else {
    info_.globalMemCacheType_ = CL_NONE;
  }

#if defined(ATI_OS_LINUX)
  info_.globalMemSize_ =
      (static_cast<uint64_t>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
       // globalMemSize is the actual available size for app on Linux
       // Because Linux base driver doesn't support paging
       static_cast<uint64_t>(memInfo.cardMemAvailableBytes + memInfo.cardExtMemAvailableBytes) /
       100u);
#else
  info_.globalMemSize_ = (static_cast<uint64_t>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
                          static_cast<uint64_t>(calAttr.localRAM) / 100u) *
      Mi;
#endif
  int uswcPercentAvailable = (calAttr.uncachedRemoteRAM > 1536 && IS_WINDOWS) ? 75 : 50;
  if (settings().apuSystem_) {
    info_.globalMemSize_ +=
        (static_cast<uint64_t>(calAttr.uncachedRemoteRAM) * Mi * uswcPercentAvailable) / 100;
  }

// We try to calculate the largest available memory size from
// the largest available block in either heap.  In theory this
// should be the size we can actually allocate at application
// start.  Note that it may not be a guarantee still as the
// application progresses.
#if defined(BRAHMA) && defined(ATI_BITS_64)
  info_.maxMemAllocSize_ =
      std::max(uint64_t(memInfo.cardMemAvailableBytes), uint64_t(memInfo.cardExtMemAvailableBytes));
#else
  info_.maxMemAllocSize_ = std::max(uint64_t(memInfo.cardLargestFreeBlockBytes),
                                    uint64_t(memInfo.cardExtLargestFreeBlockBytes));
#endif

  if (settings().apuSystem_) {
    info_.maxMemAllocSize_ = std::max(
        (static_cast<uint64_t>(calAttr.uncachedRemoteRAM) * Mi * uswcPercentAvailable) / 100,
        info_.maxMemAllocSize_);
  }
  info_.maxMemAllocSize_ =
      uint64_t(info_.maxMemAllocSize_ * std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);

  //! \note Force max single allocation size.
  //! 4GB limit for the blit kernels and 64 bit optimizations.
  info_.maxMemAllocSize_ =
      std::min(info_.maxMemAllocSize_, static_cast<uint64_t>(settings().maxAllocSize_));

  if (info_.maxMemAllocSize_ < uint64_t(128 * Mi)) {
    LogError(
        "We are unable to get a heap large enough to support the OpenCL minimum "
        "requirement for FULL_PROFILE");
  }

  info_.maxMemAllocSize_ = std::max(uint64_t(128 * Mi), info_.maxMemAllocSize_);

  // Clamp max single alloc size to the globalMemSize since it's
  // reduced by default
  info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_, info_.globalMemSize_);

  // We need to verify that we are not reporting more global memory
  // that 4x single alloc
  info_.globalMemSize_ = std::min(4 * info_.maxMemAllocSize_, info_.globalMemSize_);

  // Use 64 bit pointers
  if (settings().use64BitPtr_) {
    info_.addressBits_ = 64;
  } else {
    info_.addressBits_ = 32;
    // Limit total size with 3GB for 32 bit
    info_.globalMemSize_ = std::min(info_.globalMemSize_, uint64_t(3 * Gi));
  }

  // Alignment in BITS of the base address of any allocated memory object
  static const size_t MemBaseAlignment = 256;
  //! @note Force 256 bytes alignment, since currently
  //! calAttr.surface_alignment returns 4KB. For pinned memory runtime
  //! should be able to create a view with 256 bytes alignement
  info_.memBaseAddrAlign_ = 8 * MemBaseAlignment;

  info_.preferredConstantBufferSize_ = 16 * Ki;
  info_.maxConstantBufferSize_ = (settings().ciPlus_) ? info_.maxMemAllocSize_ : 64 * Ki;
  info_.maxConstantArgs_ = MaxConstArguments;

  // Image support fields
  if (settings().imageSupport_) {
    info_.imageSupport_ = CL_TRUE;
    info_.maxSamplers_ = MaxSamplers;
    info_.maxReadImageArgs_ = MaxReadImage;
    info_.maxWriteImageArgs_ = MaxWriteImage;
    info_.image2DMaxWidth_ = maxTextureSize;
    info_.image2DMaxHeight_ = maxTextureSize;
    info_.image3DMaxWidth_ = std::min(2 * Ki, maxTextureSize);
    info_.image3DMaxHeight_ = std::min(2 * Ki, maxTextureSize);
    info_.image3DMaxDepth_ = std::min(2 * Ki, maxTextureSize);

    info_.imagePitchAlignment_ = 256;        // XXX: 256 pixel pitch alignment for now
    info_.imageBaseAddressAlignment_ = 256;  // XXX: 256 byte base address alignment for now

    info_.bufferFromImageSupport_ = CL_TRUE;
  }

  info_.errorCorrectionSupport_ = CL_FALSE;

  if (settings().apuSystem_) {
    info_.hostUnifiedMemory_ = CL_TRUE;
  }

  info_.profilingTimerResolution_ = 1;
  info_.profilingTimerOffset_ = amd::Os::offsetToEpochNanos();
  info_.littleEndian_ = CL_TRUE;
  info_.available_ = CL_TRUE;
  info_.compilerAvailable_ = CL_TRUE;
  info_.linkerAvailable_ = CL_TRUE;

  info_.executionCapabilities_ = CL_EXEC_KERNEL;
  info_.preferredPlatformAtomicAlignment_ = 0;
  info_.preferredGlobalAtomicAlignment_ = 0;
  info_.preferredLocalAtomicAlignment_ = 0;
  info_.queueProperties_ = CL_QUEUE_PROFILING_ENABLE;

  info_.platform_ = AMD_PLATFORM;

  if ((calTarget() == CAL_TARGET_CARRIZO) && ASICREV_IS_CARRIZO_BRISTOL(calAttr.asicRevision)) {
    const static char* bristol = "Bristol Ridge";
    ::strcpy(info_.name_, bristol);
  } else {
    ::strcpy(info_.name_, hwInfo()->targetName_);
  }
  ::strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");
  ::snprintf(info_.driverVersion_, sizeof(info_.driverVersion_) - 1, AMD_BUILD_STRING);

  info_.profile_ = "FULL_PROFILE";
  if (settings().oclVersion_ >= OpenCL20) {
    info_.version_ = "OpenCL 2.0 " AMD_PLATFORM_INFO;
    info_.oclcVersion_ = "OpenCL C 2.0 ";
    info_.spirVersions_ = "1.2";
  } else if (settings().oclVersion_ == OpenCL12) {
    info_.version_ = "OpenCL 1.2 " AMD_PLATFORM_INFO;
    info_.oclcVersion_ = "OpenCL C 1.2 ";
    info_.spirVersions_ = "1.2";
  } else {
    info_.version_ = "OpenCL 1.0 " AMD_PLATFORM_INFO;
    info_.oclcVersion_ = "OpenCL C 1.0 ";
    info_.spirVersions_ = "";
    LogError("Unknown version for support");
  }

  // Fill workgroup info size
  info_.maxWorkGroupSize_ = settings().maxWorkGroupSize_;
  info_.maxWorkItemSizes_[0] = info_.maxWorkGroupSize_;
  info_.maxWorkItemSizes_[1] = info_.maxWorkGroupSize_;
  info_.maxWorkItemSizes_[2] = info_.maxWorkGroupSize_;
  info_.preferredWorkGroupSize_ = settings().preferredWorkGroupSize_;

  if (settings().hwLDSSize_ != 0) {
    info_.localMemType_ = CL_LOCAL;
    info_.localMemSize_ = settings().hwLDSSize_;
  } else {
    info_.localMemType_ = CL_GLOBAL;
    info_.localMemSize_ = 16 * Ki;
  }

  info_.extensions_ = getExtensionString();

  ::strncpy(info_.driverStore_, calAttr.driverStore, sizeof(info_.driverStore_));

  // OpenCL1.2 device info fields
  info_.builtInKernels_ = "";
  info_.imageMaxBufferSize_ = MaxImageBufferSize;
  info_.imageMaxArraySize_ = MaxImageArraySize;
  info_.preferredInteropUserSync_ = true;
  info_.printfBufferSize_ = PrintfDbg::WorkitemDebugSize * info().maxWorkGroupSize_;

  if (settings().oclVersion_ >= OpenCL20) {
    info_.svmCapabilities_ = (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
    if (settings().svmAtomics_) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
    }
    if (settings().svmFineGrainSystem_) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
    }
    // OpenCL2.0 device info fields
    info_.maxWriteImageArgs_ = MaxReadWriteImage;  //!< For compatibility
    info_.maxReadWriteImageArgs_ = MaxReadWriteImage;

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
    info_.maxGlobalVariableSize_ =
        static_cast<size_t>(amd::alignDown(info_.maxMemAllocSize_ * 9 / 10, 256));
  }

  if (settings().checkExtension(ClAmdDeviceAttributeQuery)) {
    ::strncpy(info_.boardName_, calAttr.boardName, sizeof(info_.boardName_));

    info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
    info_.deviceTopology_.pcie.bus = (calAttr.pciTopologyInformation & (0xFF << 8)) >> 8;
    info_.deviceTopology_.pcie.device = (calAttr.pciTopologyInformation & (0x1F << 3)) >> 3;
    info_.deviceTopology_.pcie.function = (calAttr.pciTopologyInformation & 0x07);

    info_.simdPerCU_ = hwInfo()->simdPerCU_;
    info_.cuPerShaderArray_ = calAttr.numberOfCUsperShaderArray;
    info_.simdWidth_ = hwInfo()->simdWidth_;
    info_.simdInstructionWidth_ = hwInfo()->simdInstructionWidth_;
    info_.wavefrontWidth_ = calAttr.wavefrontSize;

    info_.globalMemChannelBanks_ = calAttr.numMemBanks;
    info_.globalMemChannelBankWidth_ = hwInfo()->memChannelBankWidth_;
    info_.localMemSizePerCU_ = hwInfo()->localMemSizePerCU_;
    info_.localMemBanks_ = hwInfo()->localMemBanks_;
    info_.gfxipVersion_ = hwInfo()->gfxipVersion_;

    info_.numAsyncQueues_ = numComputeRings;

    info_.numRTQueues_ = numComputeRingsRT;
    info_.numRTCUs_ = calAttr.maxRTCUs;

    info_.threadTraceEnable_ = settings().threadTraceEnable_;

    info_.pcieDeviceId_ = calAttr.pcieDeviceID;
    info_.pcieRevisionId_ = calAttr.pcieRevisionID;
    info_.maxThreadsPerCU_ = info_.wavefrontWidth_ * hwInfo()->simdPerCU_ * 10;
  }
}

bool Device::Heap::create(Device& device) {
  // Create global GPU heap
  resource_ = new Memory(device, 0);
  if (resource_ == NULL) {
    return false;
  }

  if (!resource_->create(Resource::Heap)) {
    return false;
  }

  baseAddress_ = resource_->gslResource()->getSurfaceAddress();
  return true;
}

void Device::Engines::create(uint num, gslEngineDescriptor* desc, uint maxNumComputeRings) {
  numComputeRings_ = 0;
  numComputeRingsRT_ = 0;
  numDmaEngines_ = 0;

  for (uint i = 0; i < num; ++i) {
    desc_[desc[i].id] = desc[i];
    desc_[desc[i].id].priority = GSL_ENGINEPRIORITY_NEUTRAL;

    if (desc[i].id >= GSL_ENGINEID_COMPUTE0 && desc[i].id <= GSL_ENGINEID_COMPUTE7) {
      numComputeRings_++;
    }

    if (desc[i].id == GSL_ENGINEID_COMPUTE_RT) {
      numComputeRingsRT_++;
    }
    if (desc[i].id == GSL_ENGINEID_COMPUTE_MEDIUM_PRIORITY) {
      numComputeRingsRT_++;
    }

    if (desc[i].id >= GSL_ENGINEID_DRMDMA0 && desc[i].id <= GSL_ENGINEID_DRMDMA1) {
      numDmaEngines_++;
    }
  }

  numComputeRings_ = std::min(numComputeRings_, maxNumComputeRings);
}

uint Device::Engines::getRequested(uint engines, gslEngineDescriptor* desc) const {
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

Device::XferBuffers::~XferBuffers() {
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

bool Device::XferBuffers::create() {
  Memory* xferBuf = NULL;
  bool result = false;
  // Create a buffer object
  xferBuf = new Memory(dev(), bufSize_);

  // Try to allocate memory for the transfer buffer
  if ((NULL == xferBuf) || !xferBuf->create(type_)) {
    delete xferBuf;
    xferBuf = NULL;
    LogError("Couldn't allocate a transfer buffer!");
  } else {
    result = true;
    freeBuffers_.push_back(xferBuf);
    // CPU optimization: map staging buffer just once
    if (!xferBuf->cal()->cardMemory_) {
      xferBuf->map(NULL);
    }
  }

  return result;
}

Memory& Device::XferBuffers::acquire() {
  Memory* xferBuf = NULL;
  size_t listSize;

  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  listSize = freeBuffers_.size();

  // If the list is empty, then attempt to allocate a staged buffer
  if (listSize == 0) {
    // Allocate memory
    xferBuf = new Memory(dev(), bufSize_);

    // Allocate memory for the transfer buffer
    if ((NULL == xferBuf) || !xferBuf->create(type_)) {
      delete xferBuf;
      xferBuf = NULL;
      LogError("Couldn't allocate a transfer buffer!");
    } else {
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

void Device::XferBuffers::release(VirtualGPU& gpu, Memory& buffer) {
  // Make sure buffer isn't busy on the current VirtualGPU, because
  // the next aquire can come from different queue
  buffer.wait(gpu);
  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  freeBuffers_.push_back(&buffer);
  --acquiredCnt_;
}


Device::ScopedLockVgpus::ScopedLockVgpus(const Device& dev) : dev_(dev) {
  // Lock the virtual GPU list
  dev_.vgpusAccess()->lock();

  // Find all available virtual GPUs and lock them
  // from the execution of commands
  for (uint idx = 0; idx < dev_.vgpus().size(); ++idx) {
    dev_.vgpus()[idx]->execution().lock();
  }
}

Device::ScopedLockVgpus::~ScopedLockVgpus() {
  // Find all available virtual GPUs and unlock them
  // for the execution of commands
  for (uint idx = 0; idx < dev_.vgpus().size(); ++idx) {
    dev_.vgpus()[idx]->execution().unlock();
  }

  // Unock the virtual GPU list
  dev_.vgpusAccess()->unlock();
}

Device::Device()
    : NullDevice(),
      CALGSLDevice(),
      numOfVgpus_(0),
      context_(NULL),
      heap_(),
      dummyPage_(NULL),
      lockAsyncOps_(NULL),
      lockAsyncOpsForInitHeap_(NULL),
      vgpusAccess_(NULL),
      scratchAlloc_(NULL),
      mapCacheOps_(NULL),
      xferRead_(NULL),
      xferWrite_(NULL),
      mapCache_(NULL),
      resourceCache_(NULL),
      heapInitComplete_(false),
      xferQueue_(NULL),
      globalScratchBuf_(NULL),
      srdManager_(NULL) {}

Device::~Device() {
  // remove the HW debug manager
  delete hwDebugMgr_;
  hwDebugMgr_ = NULL;

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

  // Destroy resource cache
  delete resourceCache_;

  delete lockAsyncOps_;
  delete lockAsyncOpsForInitHeap_;
  delete vgpusAccess_;
  delete scratchAlloc_;
  delete mapCacheOps_;

  if (context_ != NULL) {
    context_->release();
  }

  // Close the active device
  close();
}

extern const char* SchedulerSourceCode;

bool Device::create(CALuint ordinal, CALuint numOfDevices) {
  if (!amd::Device::create()) {
    return false;
  }

  appProfile_.init();

  bool smallMemSystem = false;
  if (amd::Os::hostTotalPhysicalMemory() < OCL_SYSMEM_REQUIREMENT * Gi) {
    smallMemSystem = true;
  }

  bool noSVM = LP64_SWITCH(true, false) && !GPU_FORCE_OCL20_32BIT;
  // Open GSL device
  CALGSLDevice::OpenParams openData = {0};
  openData.enableHighPerformanceState = appProfile_.enableHighPerformanceState();
  openData.reportAsOCL12Device = (smallMemSystem ||
                                  appProfile_.reportAsOCL12Device() ||
                                  (OPENCL_VERSION < 200) ||
                                  noSVM);
  openData.sclkThreshold = appProfile_.GetSclkThreshold().c_str();
  openData.downHysteresis = appProfile_.GetDownHysteresis().c_str();
  openData.upHysteresis = appProfile_.GetUpHysteresis().c_str();
  openData.powerLimit = appProfile_.GetPowerLimit().c_str();
  openData.mclkThreshold = appProfile_.GetMclkThreshold().c_str();
  openData.mclkUpHyst = appProfile_.GetMclkUpHyst().c_str();
  openData.mclkDownHyst = appProfile_.GetMclkDownHyst().c_str();

  if (!open(ordinal, openData)) {
    return false;
  }

  // Update CAL target
  calTarget_ = getAttribs().target;
  hwInfo_ = &DeviceInfo[calTarget_];

  if ((GPU_ENABLE_PAL == 2) && usePal()) {
    return false;
  }

#if defined(BRAHMA)
  if (calTarget_ == CAL_TARGET_GREENLAND || calTarget_ == CAL_TARGET_RAVEN ||
      calTarget_ == CAL_TARGET_RAVEN2 || calTarget_ == CAL_TARGET_POLARIS22 ||
      calTarget_ == CAL_TARGET_RENOIR) {
    return false;
  }
#endif

  // Creates device settings
  settings_ = new gpu::Settings();
  gpu::Settings* gpuSettings = reinterpret_cast<gpu::Settings*>(settings_);
  if ((gpuSettings == NULL) ||
      !gpuSettings->create(getAttribs(), appProfile_.reportAsOCL12Device(), smallMemSystem)) {
    return false;
  }

  engines_.create(m_nEngines, m_engines, settings().numComputeRings_);

  amd::Context::Info info = {0};
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

  lockAsyncOpsForInitHeap_ =
      new amd::Monitor("Async Ops Lock For Initialization of Heap Resource", true);
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

  mapCache_ = new std::vector<amd::Memory*>();
  if (mapCache_ == NULL) {
    return false;
  }
  // Use just 1 entry by default for the map cache
  mapCache_->push_back(NULL);

  size_t resourceCacheSize = settings().resourceCacheSize_;

#ifdef DEBUG
  std::stringstream message;
  if (settings().remoteAlloc_) {
    message << "Using *Remote* memory";
  } else {
    message << "Using *Local* memory";
  }

  message << std::endl;
  LogInfo(message.str().c_str());
#endif  // DEBUG

  // Create resource cache.
  // \note Cache must be created before any resource creation to avoid NULL check
  resourceCache_ = new ResourceCache(resourceCacheSize);
  if (NULL == resourceCache_) {
    return false;
  }

  // Fill the device info structure
  fillDeviceInfo(getAttribs(), getMemInfo(), static_cast<size_t>(getMaxTextureSize()),
                 engines().numComputeRings(), engines().numComputeRingsRT());

  if (NULL == compiler_) {
#if !defined(ATI_OS_LINUX)
    char CompilerLibrary[220] = "";
    strcat_s(CompilerLibrary, "amdocl12cl" LP64_SWITCH("", "64") ".dll");
#endif

    const char* library = getenv("COMPILER_LIBRARY");
    aclCompilerOptions opts = {
      sizeof(aclCompilerOptions_0_8),
#if defined(ATI_OS_LINUX)
      library ? library : LINUX_ONLY("lib") "amdocl12cl" LP64_SWITCH(LINUX_SWITCH("32", ""), "64")
                              LINUX_SWITCH(".so", ".dll"),
#else
      library ? library : CompilerLibrary,
#endif
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      AMD_OCL_SC_LIB
    };
    compiler_ = aclCompilerInit(&opts, NULL);
  }

  if (settings().hsail_ || (settings().oclVersion_ >= OpenCL20)) {
    if (NULL == hsaCompiler_) {
      const char* library = getenv("HSA_COMPILER_LIBRARY");
      aclCompilerOptions opts = {
          sizeof(aclCompilerOptions_0_8), library, NULL, NULL, NULL, NULL, NULL, AMD_OCL_SC_LIB};
      // Initialize the compiler handle
      acl_error error;
      hsaCompiler_ = aclCompilerInit(&opts, &error);
      if (error != ACL_SUCCESS) {
        LogError("Error initializing the compiler");
        return false;
      }
    }
  } else {
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
  srdManager_ = new SrdManager(*this, std::max(HsaImageObjectSize, HsaSamplerObjectSize), 64 * Ki);
  if (srdManager_ == NULL) {
    return false;
  }

  // create the HW debug manager if needed
  if (settings().enableHwDebug_) {
    hwDebugMgr_ = new GpuDebugManager(this);
  }

  return true;
}

bool Device::initializeHeapResources() {
  amd::ScopedLock k(lockAsyncOpsForInitHeap_);
  if (!heapInitComplete_) {
    heapInitComplete_ = true;

    PerformFullInitialization();

    uint numComputeRings = engines_.numComputeRings() + engines_.numComputeRingsRT();
    scratch_.resize((settings().useSingleScratch_) ? 1 : (numComputeRings ? numComputeRings : 1));

    // Initialize the number of mem object for the scratch buffer
    for (uint s = 0; s < scratch_.size(); ++s) {
      scratch_[s] = new ScratchBuffer();
      if (NULL == scratch_[s]) {
        return false;
      }
    }

    // Complete initialization of the heap and other buffers
    if (!heap_.create(*this)) {
      LogError("Failed GPU heap creation");
      return false;
    }

    size_t dummySize = amd::Os::pageSize();

    // Allocate a dummy page for NULL pointer processing
    dummyPage_ = new (*context_) amd::Buffer(*context_, 0, dummySize);
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
        Resource::MemoryType type = Resource::RemoteUSWC;
        xferWrite_ = new XferBuffers(*this, type, amd::alignUp(settings().stagedXferSize_, 4 * Ki));
        if ((xferWrite_ == NULL) || !xferWrite_->create()) {
          LogError("Couldn't allocate transfer buffer objects for read");
          return false;
        }
      }

      // Initialize staged read buffers
      if (settings().stagedXferRead_) {
        xferRead_ = new XferBuffers(*this, Resource::Remote,
                                    amd::alignUp(settings().stagedXferSize_, 4 * Ki));
        if ((xferRead_ == NULL) || !xferRead_->create()) {
          LogError("Couldn't allocate transfer buffer objects for write");
          return false;
        }
      }
    }

    // Delay compilation due to brig_loader memory allocation
    if (settings().ciPlus_) {
      const char* CL20extraBlits = NULL;
      const char* ocl20 = NULL;
      if (settings().oclVersion_ >= OpenCL20) {
        CL20extraBlits = SchedulerSourceCode;
        ocl20 = "-cl-std=CL2.0";
      }
      blitProgram_ = new BlitProgram(context_);
      // Create blit programs
      if (blitProgram_ == NULL || !blitProgram_->create(this, CL20extraBlits, ocl20)) {
        delete blitProgram_;
        blitProgram_ = NULL;
        LogError("Couldn't create blit kernels!");
        return false;
      }
    }

    // Create a synchronized transfer queue
    xferQueue_ = new VirtualGPU(*this);
    if (!(xferQueue_ && xferQueue_->create(false))) {
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

device::VirtualDevice* Device::createVirtualDevice(amd::CommandQueue* queue) {
  bool profiling = false;
  bool interopQueue = false;
  uint rtCUs = amd::CommandQueue::RealTimeDisabled;
  uint deviceQueueSize = 0;

  if (queue != NULL) {
    profiling = queue->properties().test(CL_QUEUE_PROFILING_ENABLE);
    if (queue->asHostQueue() != NULL) {
      interopQueue = (0 != (queue->context().info().flags_ &
                            (amd::Context::GLDeviceKhr | amd::Context::D3D10DeviceKhr |
                             amd::Context::D3D11DeviceKhr)));
      rtCUs = queue->rtCUs();
    } else if (queue->asDeviceQueue() != NULL) {
      deviceQueueSize = queue->asDeviceQueue()->size();
    }
  }

  // Not safe to add a queue. So lock the device
  amd::ScopedLock k(lockAsyncOps());
  amd::ScopedLock lock(vgpusAccess());

  // Initialization of heap and other resources occur during the command queue creation time.
  if (!initializeHeapResources()) {
    return NULL;
  }

  VirtualGPU* vgpu = new VirtualGPU(*this);
  if (vgpu && vgpu->create(profiling, rtCUs, deviceQueueSize, queue->priority())) {
    return vgpu;
  } else {
    delete vgpu;
    return NULL;
  }
}

device::Program* Device::createProgram(amd::Program& owner, amd::option::Options* options) {
  if (isHsailProgram(options)) {
    return new HSAILProgram(*this, owner);
  }
  return new Program(*this, owner);
}

//! Requested devices list as configured by the GPU_DEVICE_ORDINAL
typedef std::unordered_map<int, bool> requestedDevices_t;

//! Parses the requested list of devices to be exposed to the user.
static void parseRequestedDeviceList(requestedDevices_t& requestedDevices) {
  char* pch = NULL;
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

bool Device::init() {
  CALuint numDevices = 0;
  bool useDeviceList = false;
  requestedDevices_t requestedDevices;

  hsaCompiler_ = NULL;
  compiler_ = NULL;

#if defined(_WIN32) && !defined(_WIN64)
  // @toto: FIXME: remove this when CAL is fixed!!!
  unsigned int old, ignored;
  _controlfp_s(&old, 0, 0);
#endif  // _WIN32 && !_WIN64
  // FIXME_lmoriche: needs cleanup
  osInit();
#if defined(_WIN32)
// osAssertSetStyle(OSASSERT_STYLE_LOGANDEXIT);
#endif  // WIN32

  gslInit();

#if defined(_WIN32) && !defined(_WIN64)
  _controlfp_s(&ignored, old, _MCW_RC | _MCW_PC);
#endif  // _WIN32 && !_WIN64

  // Get the total number of active devices
  // Count up all the devices in the system.
  numDevices = gsAdaptor::enumerateAdaptors();

  const char* selectDeviceByName = NULL;
  if (!flagIsDefault(GPU_DEVICE_ORDINAL)) {
    useDeviceList = true;
    parseRequestedDeviceList(requestedDevices);
  }

  // Loop through all active devices and initialize the device info structure
  for (CALuint ordinal = 0; ordinal < numDevices; ++ordinal) {
    // Create the GPU device object
    Device* d = new Device();
    bool result = (NULL != d) && d->create(ordinal, numDevices);
    if (useDeviceList) {
      result &= (requestedDevices.find(ordinal) != requestedDevices.end());
    }
    if (result) {
      d->registerDevice();
    } else {
      delete d;
    }
  }
  return true;
}

void Device::tearDown() {
  osExit();
  gslExit();
  aclCompilerFini(compiler_);
  if (hsaCompiler_ != NULL) {
    aclCompilerFini(hsaCompiler_);
  }
}

gpu::Memory* Device::getGpuMemory(amd::Memory* mem) const {
  return static_cast<gpu::Memory*>(mem->getDeviceMemory(*this));
}

const device::BlitManager& Device::xferMgr() const { return xferQueue_->blitMgr(); }

CalFormat Device::getCalFormat(const amd::Image::Format& format) const {
  // Find CAL format
  for (uint i = 0; i < sizeof(MemoryFormatMap) / sizeof(MemoryFormat); ++i) {
    if ((format.image_channel_data_type == MemoryFormatMap[i].clFormat_.image_channel_data_type) &&
        (format.image_channel_order == MemoryFormatMap[i].clFormat_.image_channel_order)) {
      return MemoryFormatMap[i].calFormat_;
    }
  }
  osAssert(0 && "We didn't find CAL resource format!");
  return MemoryFormatMap[0].calFormat_;
}

amd::Image::Format Device::getOclFormat(const CalFormat& format) const {
  // Find CL format
  for (uint i = 0; i < sizeof(MemoryFormatMap) / sizeof(MemoryFormat); ++i) {
    if ((format.type_ == MemoryFormatMap[i].calFormat_.type_) &&
        (format.channelOrder_ == MemoryFormatMap[i].calFormat_.channelOrder_)) {
      return MemoryFormatMap[i].clFormat_;
    }
  }
  osAssert(0 && "We didn't find OCL resource format!");
  return MemoryFormatMap[0].clFormat_;
}

// Create buffer without an owner (merge common code with createBuffer() ?)
gpu::Memory* Device::createScratchBuffer(size_t size) const {
  Memory* gpuMemory = NULL;

  // Create a memory object
  gpuMemory = new gpu::Memory(*this, size);
  if (NULL == gpuMemory || !gpuMemory->create(Resource::Local)) {
    delete gpuMemory;
    gpuMemory = NULL;
  }

  return gpuMemory;
}

gpu::Memory* Device::createBuffer(amd::Memory& owner, bool directAccess) const {
  size_t size = owner.getSize();
  gpu::Memory* gpuMemory;

  // Create resource
  bool result = false;

  if (owner.getType() == CL_MEM_OBJECT_PIPE) {
    // directAccess isnt needed as Pipes shouldnt be host accessible for GPU
    directAccess = false;
  }

  if (NULL != owner.parent()) {
    gpu::Memory* gpuParent = getGpuMemory(owner.parent());
    if (NULL == gpuParent) {
      LogError("Can't get the owner object for subbuffer allocation");
      return NULL;
    }

    if (nullptr != owner.parent()->getSvmPtr()) {
      amd::Memory* amdParent = owner.parent();
      {
        // Lock memory object, so only one commitment will occur
        amd::ScopedLock lock(amdParent->lockMemoryOps());
        amdParent->commitSvmMemory();
        amdParent->setHostMem(amdParent->getSvmPtr());
      }
      // Ignore a possible pinning error. Runtime will fallback to SW emulation
      // bool ok = gpuParent->pinSystemMemory(
      //    amdParent->getHostMem(), amdParent->getSize());
    }
    return gpuParent->createBufferView(owner);
  }

  Resource::MemoryType type =
      (owner.forceSysMemAlloc() || (owner.getMemFlags() & CL_MEM_SVM_FINE_GRAIN_BUFFER))
      ? Resource::Remote
      : Resource::Local;
  
  // Check if runtime can force a tiny buffer into USWC memory
  if ((size <= (GPU_MAX_REMOTE_MEM_SIZE * Ki)) && (type == Resource::Local) &&
    (owner.getMemFlags() & CL_MEM_READ_ONLY)) {
    type = Resource::RemoteUSWC;
  }

  if (owner.getMemFlags() & CL_MEM_BUS_ADDRESSABLE_AMD) {
    type = Resource::BusAddressable;
  } else if (owner.getMemFlags() & CL_MEM_EXTERNAL_PHYSICAL_AMD) {
    type = Resource::ExternalPhysical;
  }

  // Use direct access if it's possible
  bool remoteAlloc = false;
  // Internal means VirtualDevice!=NULL
  bool internalAlloc =
      ((owner.getMemFlags() & CL_MEM_USE_HOST_PTR) && (owner.getVirtualDevice() != NULL)) ? true
                                                                                          : false;

  // Create a memory object
  gpuMemory = new gpu::Buffer(*this, owner, owner.getSize());
  if (NULL == gpuMemory) {
    return NULL;
  }

  // Check if owner is interop memory
  if (owner.isInterop()) {
    result = gpuMemory->createInterop(Memory::InteropDirectAccess);
  } else if (owner.getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
    // Attempt to allocate from persistent heap
    result = gpuMemory->create(Resource::Persistent);
  } else if (directAccess || (type == Resource::Remote)) {
    // Check for system memory allocations
    if ((owner.getMemFlags() & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_HOST_PTR)) ||
        (settings().remoteAlloc_)) {
      // Allocate remote memory if AHP allocation and context has just 1 device
      if ((owner.getMemFlags() & CL_MEM_ALLOC_HOST_PTR) &&
          (owner.getContext().devices().size() == 1)) {
        if (owner.getMemFlags() &
            (CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
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
        params.gpu_ = reinterpret_cast<VirtualGPU*>(owner.getVirtualDevice());

        params.hostMemRef_ = owner.getHostMemRef();
        params.size_ = owner.getHostMemRef()->size();
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
            return NULL;
          }
        }
      }
    }
  }

  if (!result &&
      // Make sure it's not internal alloc
      !internalAlloc) {
    Resource::CreateParams params;
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
        size_t pipeInit[3] = {0, 0, owner.asPipe()->getMaxNumPackets()};
        gpuMemory->writeRawData(*xferQueue_, sizeof(pipeInit), pipeInit, true);
      }
      // If memory has direct access from host, then get CPU address
      if (gpuMemory->isHostMemDirectAccess() && (type != Resource::ExternalPhysical)) {
        void* address = gpuMemory->map(NULL);
        if (address != NULL) {
          // Copy saved memory
          // Note: UHP is an optional check if pinning failed and sysmem alloc was forced
          if (owner.getMemFlags() & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) {
            memcpy(address, owner.getHostMem(), owner.getSize());
          }
          // It should be safe to change the host memory pointer,
          // because it's lock protected from the upper caller
          owner.setHostMem(address);
        } else {
          result = false;
        }
      }
      // An optimization for CHP. Copy memory and destroy sysmem allocation
      else if ((gpuMemory->memoryType() != Resource::Pinned) &&
               (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
               (owner.getContext().devices().size() == 1)) {
        amd::Coord3D origin(0, 0, 0);
        amd::Coord3D region(owner.getSize());
        static const bool Entire = true;
        if (xferMgr().writeBuffer(owner.getHostMem(), *gpuMemory, origin, region, Entire)) {
          // Clear CHP memory
          owner.setHostMem(NULL);
        }
      }
    }
  }

  if (!result) {
    delete gpuMemory;
    return NULL;
  }

  return gpuMemory;
}

gpu::Memory* Device::createImage(amd::Memory& owner, bool directAccess) const {
  size_t size = owner.getSize();
  amd::Image& image = *owner.asImage();
  gpu::Memory* gpuImage = NULL;
  CalFormat format = getCalFormat(image.getImageFormat());

  if ((NULL != owner.parent()) && (owner.parent()->asImage() != NULL)) {
    device::Memory* devParent = owner.parent()->getDeviceMemory(*this);
    if (NULL == devParent) {
      LogError("Can't get the owner object for image view allocation");
      return NULL;
    }
    // Create a view on the specified device
    gpuImage = (gpu::Memory*)createView(owner, *devParent);
    if ((NULL != gpuImage) && (gpuImage->owner() != NULL)) {
      gpuImage->owner()->setHostMem((address)(owner.parent()->getHostMem()) +
                                    gpuImage->owner()->getOrigin());
    }
    return gpuImage;
  }

  gpuImage =
      new gpu::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                     format.type_, format.channelOrder_, image.getType(), image.getMipLevels());

  // Create resource
  if (NULL != gpuImage) {
    const bool imageBuffer = ((owner.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) ||
                              ((owner.getType() == CL_MEM_OBJECT_IMAGE2D) &&
                               (owner.parent() != NULL) && (owner.parent()->asBuffer() != NULL)));
    bool result = false;

    // Check if owner is interop memory
    if (owner.isInterop()) {
      result = gpuImage->createInterop(Memory::InteropDirectAccess);
    } else if (imageBuffer) {
      Resource::ImageBufferParams params;
      gpu::Memory* buffer = reinterpret_cast<gpu::Memory*>(image.parent()->getDeviceMemory(*this));
      if (buffer == NULL) {
        LogError("Buffer creation for ImageBuffer failed!");
        delete gpuImage;
        return NULL;
      }
      params.owner_ = &owner;
      params.resource_ = buffer;
      params.memory_ = buffer;

      // Create memory object
      result = gpuImage->create(Resource::ImageBuffer, &params);
    } else if (directAccess && (owner.getMemFlags() & CL_MEM_ALLOC_HOST_PTR)) {
      Resource::PinnedParams params;
      params.owner_ = &owner;
      params.hostMemRef_ = owner.getHostMemRef();
      params.size_ = owner.getHostMemRef()->size();

      // Create memory object
      result = gpuImage->create(Resource::Pinned, &params);
    }

    if (!result && !owner.isInterop()) {
      if (owner.getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
        // Attempt to allocate from persistent heap
        result = gpuImage->create(Resource::Persistent);
      } else {
        Resource::MemoryType type =
            (owner.forceSysMemAlloc()) ? Resource::RemoteUSWC : Resource::Local;
        // Create memory object
        result = gpuImage->create(type);
      }
    }

    if (!result) {
      delete gpuImage;
      return NULL;
    } else if ((gpuImage->memoryType() != Resource::Pinned) &&
               (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
               (owner.getContext().devices().size() == 1)) {
      // Ignore copy for image1D_buffer, since it was already done for buffer
      if (imageBuffer) {
        // Clear CHP memory
        owner.setHostMem(NULL);
      } else {
        amd::Coord3D origin(0, 0, 0);
        static const bool Entire = true;
        if (xferMgr().writeImage(owner.getHostMem(), *gpuImage, origin, image.getRegion(), 0, 0,
                                 Entire)) {
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
device::Memory* Device::createMemory(amd::Memory& owner) const {
  bool directAccess = false;
  gpu::Memory* memory = NULL;

  if (owner.asBuffer()) {
    directAccess = (settings().hostMemDirectAccess_ & Settings::HostMemBuffer) ? true : false;
    memory = createBuffer(owner, directAccess);
  } else if (owner.asImage()) {
    directAccess = (settings().hostMemDirectAccess_ & Settings::HostMemImage) ? true : false;
    memory = createImage(owner, directAccess);
  } else {
    LogError("Unknown memory type!");
  }

  // Attempt to pin system memory if runtime didn't use direct access
  if ((memory != NULL) && (memory->memoryType() != Resource::Pinned) &&
      (memory->memoryType() != Resource::Remote) &&
      (memory->memoryType() != Resource::RemoteUSWC) &&
      (memory->memoryType() != Resource::ExternalPhysical) &&
      ((owner.getHostMem() != NULL) ||
       ((NULL != owner.parent()) && (owner.getHostMem() != NULL)))) {
    bool ok = memory->pinSystemMemory(owner.getHostMem(), (owner.getHostMemRef()->size())
                                          ? owner.getHostMemRef()->size()
                                          : owner.getSize());
    //! \note: Ignore the pinning result for now
  }

  return memory;
}

bool Device::createSampler(const amd::Sampler& owner, device::Sampler** sampler) const {
  *sampler = NULL;
  if (settings().hsail_ || (settings().oclVersion_ >= OpenCL20)) {
    Sampler* gpuSampler = new Sampler(*this);
    if ((NULL == gpuSampler) || !gpuSampler->create(owner)) {
      delete gpuSampler;
      return false;
    }
    *sampler = gpuSampler;
  } else {
    //! Currently allocate the base device class for AMDIL path
    *sampler = new device::Sampler();
    if (*sampler == nullptr) {
      return false;
    }
  }
  return true;
}

device::Memory* Device::createView(amd::Memory& owner, const device::Memory& parent) const {
  size_t size = owner.getSize();
  assert((owner.asImage() != NULL) && "View supports images only");
  const amd::Image& image = *owner.asImage();
  gpu::Memory* gpuImage = NULL;
  CalFormat format = getCalFormat(image.getImageFormat());

  gpuImage =
      new gpu::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                     format.type_, format.channelOrder_, image.getType(), image.getMipLevels());

  // Create resource
  if (NULL != gpuImage) {
    bool result = false;
    Resource::ImageViewParams params;
    const gpu::Memory& gpuMem = static_cast<const gpu::Memory&>(parent);

    params.owner_ = &owner;
    params.level_ = image.getBaseMipLevel();
    params.layer_ = 0;
    params.resource_ = &gpuMem;
    params.gpu_ = reinterpret_cast<VirtualGPU*>(owner.getVirtualDevice());
    params.memory_ = &gpuMem;

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
bool Device::bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                bool validateOnly) {
  assert(pDevice);

  if (flags & amd::Context::Flags::GLDeviceKhr) {
    // There is no need to perform full initialization here
    // if the GSLDevice is still uninitialized.
    // Only adapter initialization is required to validate
    // GL interoperability.
    PerformAdapterInitialization(validateOnly);

    // Attempt to associate GSL-OGL
    if (!glAssociate((CALvoid*)pContext, pDevice[amd::Context::DeviceFlagIdx::GLDeviceKhrIdx])) {
      CloseInitializedAdapter(validateOnly);
      LogError("Failed gslGLAssociate()");
      return false;
    }

    CloseInitializedAdapter(validateOnly);
  }

#ifdef _WIN32
  if (flags & amd::Context::Flags::D3D10DeviceKhr) {
    // There is no need to perform full initialization here
    // if the GSLDevice is still uninitialized.
    // Only adapter initialization is required
    // to validate D3D10 interoperability.
    PerformAdapterInitialization(validateOnly);

    // Associate GSL-D3D
    if (!associateD3D10Device(reinterpret_cast<ID3D10Device*>(
            pDevice[amd::Context::DeviceFlagIdx::D3D10DeviceKhrIdx]))) {
      CloseInitializedAdapter(validateOnly);
      LogError("Failed gslD3D10Associate()");
      return false;
    }

    CloseInitializedAdapter(validateOnly);
  }

  if (flags & amd::Context::Flags::D3D11DeviceKhr) {
    // There is no need to perform full initialization here
    // if the GSLDevice is still uninitialized.
    // Only adapter initialization is required to validate
    // D3D11 interoperability.
    PerformAdapterInitialization(validateOnly);

    // Associate GSL-D3D
    if (!associateD3D11Device(reinterpret_cast<ID3D11Device*>(
            pDevice[amd::Context::DeviceFlagIdx::D3D11DeviceKhrIdx]))) {
      CloseInitializedAdapter(validateOnly);
      LogError("Failed gslD3D11Associate()");
      return false;
    }

    CloseInitializedAdapter(validateOnly);
  }

  if (flags & amd::Context::Flags::D3D9DeviceKhr) {
    PerformAdapterInitialization(validateOnly);

    // Associate GSL-D3D
    if (!associateD3D9Device(reinterpret_cast<IDirect3DDevice9*>(
            pDevice[amd::Context::DeviceFlagIdx::D3D9DeviceKhrIdx]))) {
      CloseInitializedAdapter(validateOnly);
      LogWarning("D3D9<->OpenCL adapter mismatch or D3D9Associate() failure");
      return false;
    }

    CloseInitializedAdapter(validateOnly);
  }

  if (flags & amd::Context::Flags::D3D9DeviceEXKhr) {
    PerformAdapterInitialization(validateOnly);

    // Associate GSL-D3D
    if (!associateD3D9Device(reinterpret_cast<IDirect3DDevice9Ex*>(
            pDevice[amd::Context::DeviceFlagIdx::D3D9DeviceEXKhrIdx]))) {
      CloseInitializedAdapter(validateOnly);
      LogWarning("D3D9<->OpenCL adapter mismatch or D3D9Associate() failure");
      return false;
    }

    CloseInitializedAdapter(validateOnly);
  }

  if (flags & amd::Context::Flags::D3D9DeviceVAKhr) {
  }
#endif  //_WIN32
  return true;
}

bool Device::unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly) {
  if ((flags & amd::Context::Flags::GLDeviceKhr) == 0) {
    return true;
  }

  void* glDevice = pDevice[amd::Context::DeviceFlagIdx::GLDeviceKhrIdx];
  if (glDevice != NULL) {
    // Dissociate GSL-OGL
    if (true != glDissociate(pContext, glDevice)) {
      if (validateOnly) {
        LogWarning("Failed gslGLDiassociate()");
      }
      return false;
    }
  }
  return true;
}

bool Device::globalFreeMemory(size_t* freeMemory) const {
  const uint TotalFreeMemory = 0;
  const uint LargestFreeBlock = 1;

  // Initialization of heap and other resources because getMemInfo needs it.
  if (!(const_cast<Device*>(this)->initializeHeapResources())) {
    return false;
  }

  gslMemInfo memInfo = { 0 };
  gslCtx()->getMemInfo(&memInfo, GSL_MEMINFO_BASIC);

  // Fill free memory info
  freeMemory[TotalFreeMemory] = (memInfo.cardMemAvailableBytes + memInfo.cardExtMemAvailableBytes +
    resourceCache().lclCacheSize()) / Ki;
  freeMemory[LargestFreeBlock] =
    std::max(memInfo.cardLargestFreeBlockBytes, memInfo.cardExtLargestFreeBlockBytes) / Ki;
  if (settings().apuSystem_) {
    uint64_t sysMem = 0;
    if ((memInfo.agpMemAvailableBytes + resourceCache().cacheSize()) > resourceCache().lclCacheSize()) {
      sysMem = (memInfo.agpMemAvailableBytes + resourceCache().cacheSize()) - resourceCache().lclCacheSize();
    }
    sysMem /= Ki;
    freeMemory[TotalFreeMemory] += sysMem;

    if (settings().viPlus_) {
      // for viPlus_, OCL is using remote instead remoteUSWC to avoid extra copy
      freeMemory[LargestFreeBlock] += memInfo.agpCacheableLargestFreeBlockBytes / Ki;
    } else {
      freeMemory[LargestFreeBlock] += memInfo.agpLargestFreeBlockBytes / Ki;
    }
  }
  return true;
}

amd::Memory* Device::findMapTarget(size_t size) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  amd::Memory* map = NULL;
  size_t minSize = 0;
  size_t maxSize = 0;
  uint mapId = mapCache_->size();
  uint releaseId = mapCache_->size();

  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); i++) {
    if ((*mapCache_)[i] != NULL) {
      // Requested size is smaller than the entry size
      if (size < (*mapCache_)[i]->getSize()) {
        if ((minSize == 0) || (minSize > (*mapCache_)[i]->getSize())) {
          minSize = (*mapCache_)[i]->getSize();
          mapId = i;
        }
      }
      // Requeted size matches the entry size
      else if (size == (*mapCache_)[i]->getSize()) {
        mapId = i;
        break;
      } else {
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
    Memory* gpuMemory = reinterpret_cast<Memory*>(map->getDeviceMemory(*this));

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

bool Device::addMapTarget(amd::Memory* memory) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  // the svm memory shouldn't be cached
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

Device::ScratchBuffer::~ScratchBuffer() { destroyMemory(); }

void Device::ScratchBuffer::destroyMemory() {
  // Release memory object
  delete memObj_;
  memObj_ = NULL;
}

bool Device::allocScratch(uint regNum, const VirtualGPU* vgpu) {
  if (regNum > 0) {
    // Serialize the scratch buffer allocation code
    amd::ScopedLock lk(*scratchAlloc_);
    uint sb = vgpu->hwRing();

    static const uint WaveSizeLimit = ((1 << 21) - 256);
    const uint threadSizeLimit = WaveSizeLimit / getAttribs().wavefrontSize;
    if (regNum > threadSizeLimit) {
      LogError("Requested private memory is bigger than HW supports!");
      regNum = threadSizeLimit;
    }

    // Check if the current buffer isn't big enough
    if (regNum > scratch_[sb]->regNum_) {
      // Stall all command queues, since runtime will reallocate memory
      ScopedLockVgpus lock(*this);

      scratch_[sb]->regNum_ = regNum;
      uint64_t size = 0;
      uint64_t offset = 0;

      // Destroy all views
      for (uint s = 0; s < scratch_.size(); ++s) {
        ScratchBuffer* scratchBuf = scratch_[s];
        if (scratchBuf->regNum_ > 0) {
          scratchBuf->destroyMemory();
          // Calculate the size of the scratch buffer for a queue
          scratchBuf->size_ = calcScratchBufferSize(scratchBuf->regNum_);
          scratchBuf->size_ = std::min(scratchBuf->size_, info().maxMemAllocSize_);
          scratchBuf->size_ = std::min(scratchBuf->size_, uint64_t(3 * Gi));
          scratchBuf->size_ = amd::alignUp(scratchBuf->size_, 0xFFFF);
          scratchBuf->offset_ = offset;
          size += scratchBuf->size_;
          offset += scratchBuf->size_;
        }
      }

      delete globalScratchBuf_;

      // Allocate new buffer.
      globalScratchBuf_ = new gpu::Memory(*this, static_cast<size_t>(size));
      if ((globalScratchBuf_ == NULL) || !globalScratchBuf_->create(Resource::Scratch)) {
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
          scratch_[s]->memObj_ = new gpu::Memory(*this, scratch_[s]->size_);
          Resource::ViewParams view;
          view.resource_ = globalScratchBuf_;
          view.offset_ = scratch_[s]->offset_;
          view.size_ = scratch_[s]->size_;
          if ((scratch_[s]->memObj_ == NULL) ||
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

bool Device::validateKernel(
    const amd::Kernel& kernel, const device::VirtualDevice* vdev, bool coop_groups) {
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
      amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(*this);
      if (defQueue != NULL) {
        vgpu = static_cast<VirtualGPU*>(defQueue->vDev());
        if (!allocScratch(hsaKernel->prog().maxScratchRegs(), vgpu)) {
          return false;
        }
      } else {
        return false;
      }
    }
  }

  return true;
}

void Device::destroyScratchBuffers() {
  if (globalScratchBuf_ != NULL) {
    for (uint s = 0; s < scratch_.size(); ++s) {
      scratch_[s]->destroyMemory();
      scratch_[s]->regNum_ = 0;
    }
    delete globalScratchBuf_;
    globalScratchBuf_ = NULL;
  }
}

void Device::fillHwSampler(uint32_t state, void* hwState, uint32_t hwStateSize, uint32_t mipFilter,
                           float minLod, float maxLod) const {
  // All GSL sampler's parameters are in floats
  uint32_t gslAddress = GSL_CLAMP_TO_BORDER;
  uint32_t gslMinFilter = GSL_MIN_NEAREST;
  uint32_t gslMagFilter = GSL_MAG_NEAREST;
  bool unnorm = !(state & amd::Sampler::StateNormalizedCoordsMask);

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

  if (mipFilter == CL_FILTER_NEAREST) {
    if (gslMinFilter == GSL_MIN_NEAREST) {
      gslMinFilter = GSL_MIN_NEAREST_MIPMAP_NEAREST;
    } else {
      gslMinFilter = GSL_MIN_LINEAR_MIPMAP_NEAREST;
    }
  } else if (mipFilter == CL_FILTER_LINEAR) {
    if (gslMinFilter == GSL_MIN_NEAREST) {
      gslMinFilter = GSL_MIN_NEAREST_MIPMAP_LINEAR;
    } else {
      gslMinFilter = GSL_MIN_LINEAR_MIPMAP_LINEAR;
    }
  }

  fillSamplerHwState(unnorm, gslMinFilter, gslMagFilter, gslAddress, minLod, maxLod, hwState,
                     hwStateSize);
}

void* Device::hostAlloc(size_t size, size_t alignment, bool atomics) const {
  // for discrete gpu, we only reserve,no commit yet.
  return amd::Os::reserveMemory(NULL, size, alignment, amd::Os::MEM_PROT_NONE);
}

void Device::hostFree(void* ptr, size_t size) const {
  // If we allocate the host memory, we need free, or we have to release
  amd::Os::releaseMemory(ptr, size);
}

void* Device::svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                       void* svmPtr) const {
  alignment = std::max(alignment, static_cast<size_t>(info_.memBaseAddrAlign_));

  amd::Memory* mem = NULL;
  if (NULL == svmPtr) {
    if (isFineGrainedSystem()) {
      return amd::Os::alignedMalloc(size, alignment);
    }

    // create a hidden buffer, which will allocated on the device later
    mem = new (context) amd::Buffer(context, flags, size, reinterpret_cast<void*>(1));
    if (mem == NULL) {
      LogError("failed to create a svm mem object!");
      return NULL;
    }

    if (!mem->create(NULL, false)) {
      LogError("failed to create a svm hidden buffer!");
      mem->release();
      return NULL;
    }
    // if the device supports SVM FGS, return the committed CPU address directly.
    gpu::Memory* gpuMem = getGpuMemory(mem);

    // add the information to context so that we can use it later.
    amd::MemObjMap::AddMemObj(mem->getSvmPtr(), mem);
    svmPtr = mem->getSvmPtr();
  } else {
    // find the existing amd::mem object
    mem = amd::MemObjMap::FindMemObj(svmPtr);
    if (NULL == mem) {
      return NULL;
    }
    // commit the CPU memory for FGS device.
    if (isFineGrainedSystem()) {
      mem->commitSvmMemory();
    } else {
      gpu::Memory* gpuMem = getGpuMemory(mem);
    }
    svmPtr = mem->getSvmPtr();
  }
  return svmPtr;
}

void Device::svmFree(void* ptr) const {
  if (isFineGrainedSystem()) {
    amd::Os::alignedFree(ptr);
  } else {
    amd::Memory* svmMem = NULL;
    svmMem = amd::MemObjMap::FindMemObj(ptr);
    if (NULL != svmMem) {
      svmMem->release();
      amd::MemObjMap::RemoveMemObj(ptr);
    }
  }
}

Device::SrdManager::~SrdManager() {
  for (uint i = 0; i < pool_.size(); ++i) {
    pool_[i].buf_->unmap(NULL);
    delete pool_[i].buf_;
    delete pool_[i].flags_;
  }
}

bool Sampler::create(uint32_t oclSamplerState) {
  hwSrd_ = dev_.srds().allocSrdSlot(&hwState_);
  if (0 == hwSrd_) {
    return false;
  }
  dev_.fillHwSampler(oclSamplerState, hwState_, HsaSamplerObjectSize);
  return true;
}

bool Sampler::create(const amd::Sampler& owner) {
  hwSrd_ = dev_.srds().allocSrdSlot(&hwState_);
  if (0 == hwSrd_) {
    return false;
  }
  dev_.fillHwSampler(owner.state(), hwState_, HsaSamplerObjectSize, owner.mipFilter(),
                     owner.minLod(), owner.maxLod());
  return true;
}

Sampler::~Sampler() { dev_.srds().freeSrdSlot(hwSrd_); }

uint64_t Device::SrdManager::allocSrdSlot(address* cpuAddr) {
  amd::ScopedLock lock(ml_);
  // Check all buffers in the pool of chunks
  for (uint i = 0; i < pool_.size(); ++i) {
    const Chunk& ch = pool_[i];
    // Search for an empty slot
    for (uint s = 0; s < numFlags_; ++s) {
      uint mask = ch.flags_[s];
      // Check if there is an empty slot in this group
      if (mask != 0) {
        uint idx;
        // Find the first empty index
        for (idx = 0; (mask & 0x1) == 0; mask >>= 1, ++idx)
          ;
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
    delete[] chunk.flags_;
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

void Device::SrdManager::freeSrdSlot(uint64_t addr) {
  amd::ScopedLock lock(ml_);
  // Check all buffers in the pool of chunks
  for (uint i = 0; i < pool_.size(); ++i) {
    Chunk* ch = &pool_[i];
    // Find the offset
    int64_t offs = static_cast<int64_t>(addr) - static_cast<int64_t>(ch->buf_->vmAddress());
    // Check if the offset inside the chunk buffer
    if ((offs >= 0) && (offs < bufSize_)) {
      // Find the index in the chunk
      uint idx = offs / srdSize_;
      uint s = idx / MaskBits;
      // Free the slot
      ch->flags_[s] |= 1 << (idx % MaskBits);
      return;
    }
  }
  assert(false && "Wrong slot address!");
}

void Device::SrdManager::fillResourceList(std::vector<const Memory*>& memList) {
  for (uint i = 0; i < pool_.size(); ++i) {
    memList.push_back(pool_[i].buf_);
  }
}

int32_t Device::hwDebugManagerInit(amd::Context* context, uintptr_t messageStorage) {
  int32_t status = hwDebugMgr_->registerDebugger(context, messageStorage);

  if (CL_SUCCESS != status) {
    delete hwDebugMgr_;
    hwDebugMgr_ = NULL;
  }

  return status;
}

bool Device::SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput, cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
  bool result = true;
  static const bool bValidate = true;
  PerformAdapterInitialization(bValidate);
  GSLClockModeInfo clockModeInfo = {};
  clockModeInfo.clockmode = static_cast<GSLClockMode>(setClockModeInput.clock_mode);
  result = gslSetClockMode(&clockModeInfo);
  CloseInitializedAdapter(bValidate);
  return result;
}

}  // namespace gpu
