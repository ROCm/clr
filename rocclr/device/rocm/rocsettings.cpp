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

#ifndef WITHOUT_GPU_BACKEND

#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "rocsettings.hpp"
#include "device/rocm/rocglinterop.hpp"

namespace roc {

// ================================================================================================
Settings::Settings() {
  // Initialize the HSA device default settings

  // Set this to true when we drop the flag
  doublePrecision_ = ::CL_KHR_FP64;

  enableLocalMemory_ = HSA_LOCAL_MEMORY_ENABLE;
  enableCoarseGrainSVM_ = HSA_ENABLE_COARSE_GRAIN_SVM;

  maxWorkGroupSize_ = 1024;
  preferredWorkGroupSize_ = 256;

  maxWorkGroupSize2DX_ = 16;
  maxWorkGroupSize2DY_ = 16;
  maxWorkGroupSize3DX_ = 4;
  maxWorkGroupSize3DY_ = 4;
  maxWorkGroupSize3DZ_ = 4;

  kernargPoolSize_ = HSA_KERNARG_POOL_SIZE;

  // Determine if user is requesting Non-Coherent mode
  // for system memory. By default system memory is
  // operates or is programmed to be in Coherent mode.
  // Users can turn it off for hardware that does not
  // support this feature naturally
  char* nonCoherentMode = nullptr;
  nonCoherentMode = getenv("OPENCL_USE_NC_MEMORY_POLICY");
  enableNCMode_ = (nonCoherentMode) ? true : false;

  // Disable image DMA by default (ROCM runtime doesn't support it)
  imageDMA_ = false;

  stagedXferRead_ = true;
  stagedXferWrite_ = true;
  stagedXferSize_ = GPU_STAGING_BUFFER_SIZE * Ki;

  // Initialize transfer buffer size to 1MB by default
  xferBufSize_ = 1024 * Ki;

  const static size_t MaxPinnedXferSize = 32;
  pinnedXferSize_ = std::min(GPU_PINNED_XFER_SIZE, MaxPinnedXferSize) * Mi;
  pinnedMinXferSize_ = std::min(GPU_PINNED_MIN_XFER_SIZE * Ki, pinnedXferSize_);

  sdmaCopyThreshold_ = GPU_FORCE_BLIT_COPY_SIZE * Ki;

  // Don't support Denormals for single precision by default
  singleFpDenorm_ = false;

  apuSystem_ = false;

  // Device enqueuing settings
  numDeviceEvents_ = 1024;
  numWaitEvents_ = 8;

  useLightning_ = (!flagIsDefault(GPU_ENABLE_LC)) ? GPU_ENABLE_LC : true;

  lcWavefrontSize64_ = true;
  imageBufferWar_ = false;

  hmmFlags_ = (!flagIsDefault(ROC_HMM_FLAGS)) ? ROC_HMM_FLAGS : Hmm::EnableSvmTracking;

  rocr_backend_ = true;
  barrier_sync_ = (!flagIsDefault(ROC_BARRIER_SYNC)) ? ROC_BARRIER_SYNC : true;

  cpu_wait_for_signal_ = !AMD_DIRECT_DISPATCH;
  cpu_wait_for_signal_ = (!flagIsDefault(ROC_CPU_WAIT_FOR_SIGNAL)) ?
                          ROC_CPU_WAIT_FOR_SIGNAL : cpu_wait_for_signal_;
  system_scope_signal_ = ROC_SYSTEM_SCOPE_SIGNAL;
  skip_copy_sync_      = ROC_SKIP_COPY_SYNC;
}

// ================================================================================================
bool Settings::create(bool fullProfile, uint32_t gfxipMajor, uint32_t gfxipMinor, bool enableXNACK,
                      bool coop_groups) {
  customHostAllocator_ = false;

  if (fullProfile) {
    pinnedXferSize_ = 0;
    stagedXferSize_ = 0;
    xferBufSize_ = 0;
    apuSystem_ = true;
  } else {
    pinnedXferSize_ = std::max(pinnedXferSize_, pinnedMinXferSize_);
    stagedXferSize_ = std::max(stagedXferSize_, pinnedMinXferSize_ + 4 * Ki);
  }
  enableXNACK_ = enableXNACK;
  hsailExplicitXnack_ = enableXNACK;

  // Enable extensions
  enableExtension(ClKhrByteAddressableStore);
  enableExtension(ClKhrGlobalInt32BaseAtomics);
  enableExtension(ClKhrGlobalInt32ExtendedAtomics);
  enableExtension(ClKhrLocalInt32BaseAtomics);
  enableExtension(ClKhrLocalInt32ExtendedAtomics);
  enableExtension(ClKhrInt64BaseAtomics);
  enableExtension(ClKhrInt64ExtendedAtomics);
  enableExtension(ClKhr3DImageWrites);
  enableExtension(ClAmdMediaOps);
  enableExtension(ClAmdMediaOps2);
  enableExtension(ClKhrImage2dFromBuffer);

#ifdef ROCCLR_ENABLE_GL_SHARING
  if (MesaInterop::Supported()) {
    enableExtension(ClKhrGlSharing);
  }
#endif

  // Enable platform extension
  enableExtension(ClAmdDeviceAttributeQuery);

  // Enable KHR double precision extension
  enableExtension(ClKhrFp64);
  enableExtension(ClKhrSubGroups);
  enableExtension(ClKhrDepthImages);
  enableExtension(ClAmdCopyBufferP2P);
  enableExtension(ClKhrFp16);
  supportDepthsRGB_ = true;

  if (useLightning_) {
    enableExtension(ClAmdAssemblyProgram);
    // enable subnormals for gfx900 and later
    if (gfxipMajor >= 9) {
      singleFpDenorm_ = true;
      enableCoopGroups_ = GPU_ENABLE_COOP_GROUPS & coop_groups;
      enableCoopMultiDeviceGroups_ = GPU_ENABLE_COOP_GROUPS & coop_groups;
    }
  } else {
    // Also enable AMD double precision extension?
    enableExtension(ClAmdFp64);
  }

  if (gfxipMajor >= 10) {
     enableWave32Mode_ = true;
     enableWgpMode_ = GPU_ENABLE_WGP_MODE;
     if (gfxipMinor == 1) {
       // GFX10.1 HW doesn't support custom pitch. Enable double copy workaround
       // TODO: This should be updated when ROCr support custom pitch
       imageBufferWar_ = GPU_IMAGE_BUFFER_WAR;
     }
  }
  if (!flagIsDefault(GPU_ENABLE_WAVE32_MODE)) {
    enableWave32Mode_ = GPU_ENABLE_WAVE32_MODE;
  }

  lcWavefrontSize64_ = !enableWave32Mode_;

  // Override current device settings
  override();

  return true;
}

// ================================================================================================
void Settings::override() {
  // Limit reported workgroup size
  if (GPU_MAX_WORKGROUP_SIZE != 0) {
    preferredWorkGroupSize_ = GPU_MAX_WORKGROUP_SIZE;
  }

  if (GPU_MAX_WORKGROUP_SIZE_2D_X != 0) {
    maxWorkGroupSize2DX_ = GPU_MAX_WORKGROUP_SIZE_2D_X;
  }
  if (GPU_MAX_WORKGROUP_SIZE_2D_Y != 0) {
    maxWorkGroupSize2DY_ = GPU_MAX_WORKGROUP_SIZE_2D_Y;
  }

  if (GPU_MAX_WORKGROUP_SIZE_3D_X != 0) {
    maxWorkGroupSize3DX_ = GPU_MAX_WORKGROUP_SIZE_3D_X;
  }
  if (GPU_MAX_WORKGROUP_SIZE_3D_Y != 0) {
    maxWorkGroupSize3DY_ = GPU_MAX_WORKGROUP_SIZE_3D_Y;
  }
  if (GPU_MAX_WORKGROUP_SIZE_3D_Z != 0) {
    maxWorkGroupSize3DZ_ = GPU_MAX_WORKGROUP_SIZE_3D_Z;
  }

  if (!flagIsDefault(GPU_XFER_BUFFER_SIZE)) {
    xferBufSize_ = GPU_XFER_BUFFER_SIZE * Ki;
  }

  if (!flagIsDefault(GPU_PINNED_MIN_XFER_SIZE)) {
    pinnedMinXferSize_ = std::min(GPU_PINNED_MIN_XFER_SIZE * Ki, pinnedXferSize_);
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
  if (!flagIsDefault(GPU_ENABLE_COOP_GROUPS)) {
    enableCoopGroups_ = GPU_ENABLE_COOP_GROUPS;
    enableCoopMultiDeviceGroups_ = GPU_ENABLE_COOP_GROUPS;
  }
}
}  // namespace roc

#endif  // WITHOUT_GPU_BACKEND
