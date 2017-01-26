//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef WITHOUT_GPU_BACKEND

#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "rocsettings.hpp"
#include "device/rocm/rocglinterop.hpp"

namespace roc {

Settings::Settings()
{
    // Initialize the HSA device default settings

    // Set this to true when we drop the flag
    doublePrecision_    = ::CL_KHR_FP64;
    pollCompletion_     = ENVVAR_HSA_POLL_KERNEL_COMPLETION;

    enableLocalMemory_ = HSA_LOCAL_MEMORY_ENABLE;
    enableImageHandle_ = true;

    maxWorkGroupSize_ = 256;
    maxWorkGroupSize2DX_ = 16;
    maxWorkGroupSize2DY_ = 16;
    maxWorkGroupSize3DX_ = 4;
    maxWorkGroupSize3DY_ = 4;
    maxWorkGroupSize3DZ_ = 4;

    kernargPoolSize_ = HSA_KERNARG_POOL_SIZE;
    signalPoolSize_ = HSA_SIGNAL_POOL_SIZE;

    // Determine if user is requesting Non-Coherent mode
    // for system memory. By default system memory is
    // operates or is programmed to be in Coherent mode.
    // Users can turn it off for hardware that does not
    // support this feature naturally
    char *nonCoherentMode = NULL;
    nonCoherentMode = getenv("OPENCL_USE_NC_MEMORY_POLICY");
    enableNCMode_ = (nonCoherentMode)? true : false;

    // Determine if user wishes to disable support for
    // partial dispatch. By default support for partial
    // dispatch is enabled. Users can turn it off for
    // devices that do not support this feature.
    //
    // @note Update appropriate field of device::Settings
    char *partialDispatch = NULL;
    partialDispatch = getenv("OPENCL_DISABLE_PARTIAL_DISPATCH");
    enablePartialDispatch_ = (partialDispatch) ? false : true;
    partialDispatch_ = (partialDispatch) ? false : true;
    commandQueues_ = 100; //!< Field value set to maximum number
                     //!< concurrent Virtual GPUs for ROCm backend

    // Disable image DMA by default (ROCM runtime doesn't support it)
    imageDMA_       = false;

    stagedXferRead_     = true;
    stagedXferWrite_    = true;
    stagedXferSize_     = GPU_STAGING_BUFFER_SIZE * Ki;

    // Initialize transfer buffer size to 1MB by default
    xferBufSize_    = 1024 * Ki;

    const static size_t MaxPinnedXferSize = 32;
    pinnedXferSize_     = std::min(GPU_PINNED_XFER_SIZE, MaxPinnedXferSize) * Mi;
    pinnedMinXferSize_  = std::min(GPU_PINNED_MIN_XFER_SIZE * Ki, pinnedXferSize_);
}

bool
Settings::create(bool fullProfile)
{
    customHostAllocator_ = false;

    if (fullProfile) {
        pinnedXferSize_ = 0;
        stagedXferSize_ = 0;
        xferBufSize_    = 0;
    }
    else {
        pinnedXferSize_ = std::max(pinnedXferSize_, pinnedMinXferSize_);
        stagedXferSize_ = std::max(stagedXferSize_, pinnedMinXferSize_ + 4 * Ki);
    }

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
    if(MesaInterop::Supported()) {
       enableExtension(ClKhrGlSharing);
    }

    // Enable KHR double precision extension
    enableExtension(ClKhrFp64);
#if !defined(WITH_LIGHTNING_COMPILER)
    // Also enable AMD double precision extension?
    enableExtension(ClAmdFp64);
#endif // !defined(WITH_LIGHTNING_COMPILER)
    enableExtension(ClKhrSubGroups);

    enableExtension(ClKhrDepthImages);
    supportDepthsRGB_ = true;

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

    if (!flagIsDefault(GPU_MAX_COMMAND_QUEUES)) {
        commandQueues_ = GPU_MAX_COMMAND_QUEUES;
    }

    if (!flagIsDefault(GPU_IMAGE_DMA)) {
        commandQueues_ = GPU_IMAGE_DMA;
    }

    if (!flagIsDefault(GPU_XFER_BUFFER_SIZE)) {
        xferBufSize_ = GPU_XFER_BUFFER_SIZE * Ki;
    }

    if (!flagIsDefault(GPU_PINNED_MIN_XFER_SIZE)) {
        pinnedMinXferSize_  = std::min(GPU_PINNED_MIN_XFER_SIZE * Ki, pinnedXferSize_);
    }
}

} // namespace roc

#endif // WITHOUT_GPU_BACKEND
