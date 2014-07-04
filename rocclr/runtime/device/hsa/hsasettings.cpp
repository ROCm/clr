//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef WITHOUT_GPU_BACKEND

#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "hsasettings.hpp"

namespace oclhsa {

Settings::Settings()
{
    // Initialize the HSA device default settings

    // Set this to true when we drop the flag
    doublePrecision_    = ::CL_KHR_FP64;
    pollCompletion_     = ENVVAR_HSA_POLL_KERNEL_COMPLETION;

    // Enable "local" memory in HSA
    enableLocalMemory_  = HSA_LOCAL_MEMORY_ENABLE;
    enableSvm32BitsAtomics_ = HSA_ENABLE_ATOMICS_32B;

    maxWorkGroupSize_ = 256;
    maxWorkGroupSize2DX_ = 16;
    maxWorkGroupSize2DY_ = 16;
    maxWorkGroupSize3DX_ = 4;
    maxWorkGroupSize3DY_ = 4;
    maxWorkGroupSize3DZ_ = 4;
}

bool
Settings::create(bool doublePrecision)
{
    largeHostMemAlloc_ = true;
    customHostAllocator_ = true;
    customSvmAllocator_ = true;

    // Enable extensions
    enableExtension(ClKhrByteAddressableStore);
    enableExtension(ClKhrGlobalInt32BaseAtomics);
    enableExtension(ClKhrGlobalInt32ExtendedAtomics);
    enableExtension(ClKhrLocalInt32BaseAtomics);
    enableExtension(ClKhrLocalInt32ExtendedAtomics);
    enableExtension(ClExtAtomicCounters32);
    enableExtension(ClKhr3DImageWrites);
    enableExtension(ClKhrGlSharing);
    enableExtension(ClAmdMediaOps);
    enableExtension(ClAmdMediaOps2);
#if defined(_WIN32)
    enableExtension(ClKhrD3d10Sharing);
    enableExtension(ClKhrD3d11Sharing);
#endif // _WIN32
	enableExtension(ClKhrImage2dFromBuffer);
	enableExtension(ClAmdImage2dFromBufferReadOnly);
    // Make sure device supports doubles
    doublePrecision_ &= doublePrecision;

    if (doublePrecision_) {
        // Enable KHR double precision extension
        enableExtension(ClKhrFp64);
        // Also enable AMD double precision extension?
        enableExtension(ClAmdFp64);
    }

    // Override current device settings
    override();

    return true;
}

void
Settings::override()
{
}

} // namespace oclhsa

#endif // WITHOUT_GPU_BACKEND
