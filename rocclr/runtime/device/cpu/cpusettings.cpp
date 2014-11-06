//
// Copyright 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpusettings.hpp"
#include "os/os.hpp"

namespace cpu {

bool
Settings::create()
{
    largeHostMemAlloc_ = true;

    // This code is temporary until cl_khr_fp64 is unconditional
    if (flagIsDefault(CL_KHR_FP64) || CL_KHR_FP64) {
        enableExtension(ClKhrFp64);
    }

    enableExtension(ClAmdFp64);
    enableExtension(ClKhrGlobalInt32BaseAtomics);
    enableExtension(ClKhrGlobalInt32ExtendedAtomics);
    enableExtension(ClKhrLocalInt32BaseAtomics);
    enableExtension(ClKhrLocalInt32ExtendedAtomics);

#ifdef _LP64
    enableExtension(ClKhrInt64BaseAtomics);
    enableExtension(ClKhrInt64ExtendedAtomics);
#endif // _LP64
    enableExtension(ClKhrByteAddressableStore);
    enableExtension(ClKhrGlSharing);
    enableExtension(ClKhrGlEvent);
    enableExtension(ClExtDeviceFission);
    enableExtension(ClAmdDeviceAttributeQuery);
    enableExtension(ClAmdVec3);
    enableExtension(ClAmdMediaOps);
    enableExtension(ClAmdMediaOps2);
    enableExtension(ClAmdPopcnt);
    enableExtension(ClAmdPrintf);

    // enableExtension(ClKhrSelectFpRoundingMode);
    enableExtension(ClKhr3DImageWrites);

    // enableExtension(ClKhrFp16);

#if defined(_WIN32)
    enableExtension(ClKhrD3d10Sharing);
#endif // _WIN32
    enableExtension(ClKhrSpir);

    // Enable some OpenCL 2.0 extensions
    if ((OPENCL_MAJOR >= 2) && (CPU_OPENCL_VERSION >= 200)) {
        partialDispatch_ = true;
        enableExtension(ClKhrSubGroups);
        supportDepthsRGB_ = true;
        enableExtension(ClKhrDepthImages);
    }

    // Map CPUID feature bits to our own feature bits
    const int sse2_features = CPUFEAT_DX_SSE | CPUFEAT_DX_SSE2;
    const int avx_features = CPUFEAT_CX_SSE3 | CPUFEAT_CX_SSSE3 |
                             CPUFEAT_CX_SSE4_1 | CPUFEAT_CX_SSE4_2 |
                             CPUFEAT_CX_POPCNT | CPUFEAT_CX_AVX |
                             CPUFEAT_CX_OSXSAVE;
    const int fma3_features = INTEL_CPUFEAT_CX_FMA3;
    const int fma4_features = AMD_CPUFEAT_CX_FMA4 | AMD_CPUFEAT_CX_XOP;
    int regs[4];

#if defined(ATI_ARCH_X86)
    amd::Os::cpuid(regs, 0x0);
    bool isAmd = regs[1] == ('A' | ('u' << 8) | ('t' << 16) | ('h' << 24));
    bool isIntel = regs[1] == ('G' | ('e' << 8) | ('n' << 16) | ('u' << 24));

    amd::Os::cpuid(regs, 0x1);

    cpuFeatures_  = (regs[3] & sse2_features) == sse2_features ?
                    SSE2Instructions : 0;

    if ((regs[2] & avx_features) == avx_features) {
        // Check for state support
        uint64_t xcr0 = amd::Os::xgetbv(0);

        // Check for SSE and YMM bits (1 and 2)
        if (((uint32_t)xcr0 & 0x6U) == 0x6U) {
            cpuFeatures_ |= AVXInstructions;

            // Now check for FMA and XOP
            if (isIntel) {
                cpuFeatures_ |= (regs[2] & fma3_features) == fma3_features ?
                                 FMA3Instructions : 0;
            }

            if (isAmd) {
                amd::Os::cpuid(regs, 0x80000001);
                cpuFeatures_ |= (regs[2] & fma4_features) == fma4_features ?
                                FMA4Instructions : 0;
            }
        }
    }
#endif // ATI_ARCH_X86

    return true;
}

} // namespace cpu
