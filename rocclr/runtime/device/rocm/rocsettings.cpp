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
}

bool
Settings::create(bool doublePrecision)
{
    customHostAllocator_ = true;

    // Enable extensions
    enableExtension(ClKhrByteAddressableStore);
    enableExtension(ClKhrGlobalInt32BaseAtomics);
    enableExtension(ClKhrGlobalInt32ExtendedAtomics);
    enableExtension(ClKhrLocalInt32BaseAtomics);
    enableExtension(ClKhrLocalInt32ExtendedAtomics);
    enableExtension(ClKhr3DImageWrites);
    enableExtension(ClAmdMediaOps);
    enableExtension(ClAmdMediaOps2);
    if(MesaInterop::Supported())
      enableExtension(ClKhrGlSharing);

    // Make sure device supports doubles
    doublePrecision_ &= doublePrecision;

    if (doublePrecision_) {
        // Enable KHR double precision extension
        enableExtension(ClKhrFp64);
        // Also enable AMD double precision extension?
        enableExtension(ClAmdFp64);
    }

    enableExtension(ClKhrDepthImages);
    supportDepthsRGB_ = true;

    // Override current device settings
    override();

    return true;
}

void
Settings::override()
{
}

} // namespace roc

#endif // WITHOUT_GPU_BACKEND
