
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include <vector>
#include <string>

#include "top.hpp"
#include "aclTypes.h"
#include "library.hpp"
#include "utils/options.hpp"

namespace amd {

/*
    Integrated bitcode libraries
 */

// GPU libraries
#if defined(WITH_TARGET_AMDIL)
#include "builtins-gpugen-comm.inc"
#include "builtins-gpugen-diff.gpu.inc"
#include "builtins-gpugen-diff.gpu-64.inc"
#include "builtins-gpucommon-comm.inc"
#include "builtins-gpucommon-diff.gpu.inc"
#include "builtins-gpucommon-diff.gpu-64.inc"
#include "builtins-SI-comm.inc"
#include "builtins-SI-diff.gpu.inc"
#include "builtins-SI-diff.gpu-64.inc"
#include "builtins-CI-comm.inc"
#include "builtins-CI-diff.gpu.inc"
#include "builtins-CI-diff.gpu-64.inc"
#endif // WITH_TARGET_AMDIL

// CPU libraries
#if defined(WITH_TARGET_X86)
#include "builtins-cpugen.x86.inc"
#include "builtins-cpucommon.x86.inc"
#include "builtins-avx.x86.inc"
#include "builtins-fma4.x86.inc"
#include "builtins-cpugen.x86-64.inc"
#include "builtins-cpucommon.x86-64.inc"
#include "builtins-avx.x86-64.inc"
#include "builtins-fma4.x86-64.inc"
#endif // WITH_TARGET_X86

#if defined(WITH_TARGET_ARM)
#include "builtins-cpugen.arm.inc"
#include "builtins-cpucommon.arm.inc"
#endif // WITH_TARGET_ARM

#ifdef WITH_TARGET_HSAIL
// HSAIL libraries
#include "builtins-hsail.inc"
#include "builtins-hsail-amd-ci.inc"
#include "builtins-gcn.inc"
#include "builtins-ocml.inc"
#include "builtins-spirv.inc"
#endif

#include <cstdlib>
// getLibsDesc() : returns a list of libraries that need to be linked with the
//                 application.  The max number of libraries is defined by
//                 enum MAX_NUM_LIBRARY_DESCS in class LibraryDescriptor.
//
// Return 0:  successful
//      <n>:  error happened
int
getLibDescs ( 
    LibrarySelector     LibType,    // input
    LibraryDescriptor*  LibDesc,      // output
    int&                LibDescSize   // output -- LibDesc[0:LibDescSize-1]
)
{
    switch (LibType) {
#if defined(WITH_TARGET_AMDIL)
    case GPU_Library_Evergreen:
        // Library order is important!
        LibDesc[0].start = reinterpret_cast<const char*>
            (builtins_gpucommon_comm);
        LibDesc[0].size  = builtins_gpucommon_comm_size;
        LibDesc[1].start = reinterpret_cast<const char*>
            (builtins_gpucommon_diff_gpu);
        LibDesc[1].size  = builtins_gpucommon_diff_gpu_size;
        LibDesc[2].start = reinterpret_cast<const char*>
            (builtins_gpugen_comm);
        LibDesc[2].size  = builtins_gpugen_comm_size;
        LibDesc[3].start = reinterpret_cast<const char*>
            (builtins_gpugen_diff_gpu);
        LibDesc[3].size  = builtins_gpugen_diff_gpu_size;
        LibDescSize = 4;
        break;

    case GPU_Library_SI:
        // Library order is important!
        LibDesc[0].start = reinterpret_cast<const char*>
            (builtins_SI_comm);
        LibDesc[0].size  = builtins_SI_comm_size;
        LibDesc[1].start = reinterpret_cast<const char*>
            (builtins_SI_diff_gpu);
        LibDesc[1].size  = builtins_SI_diff_gpu_size;
        LibDesc[2].start = reinterpret_cast<const char*>
            (builtins_gpucommon_comm);
        LibDesc[2].size  = builtins_gpucommon_comm_size;
        LibDesc[3].start = reinterpret_cast<const char*>
            (builtins_gpucommon_diff_gpu);
        LibDesc[3].size  = builtins_gpucommon_diff_gpu_size;
        LibDesc[4].start = reinterpret_cast<const char*>
            (builtins_gpugen_comm);
        LibDesc[4].size  = builtins_gpugen_comm_size;
        LibDesc[5].start = reinterpret_cast<const char*>
            (builtins_gpugen_diff_gpu);
        LibDesc[5].size  = builtins_gpugen_diff_gpu_size;
        LibDescSize = 6;
        break;

    case GPU64_Library_SI:
        // Library order is important!
        LibDesc[0].start = reinterpret_cast<const char*>
            (builtins_SI_comm);
        LibDesc[0].size  = builtins_SI_comm_size;
        LibDesc[1].start = reinterpret_cast<const char*>
            (builtins_SI_diff_gpu_64);
        LibDesc[1].size  = builtins_SI_diff_gpu_64_size;
        LibDesc[2].start = reinterpret_cast<const char*>
            (builtins_gpucommon_comm);
        LibDesc[2].size  = builtins_gpucommon_comm_size;
        LibDesc[3].start = reinterpret_cast<const char*>
            (builtins_gpucommon_diff_gpu_64);
        LibDesc[3].size  = builtins_gpucommon_diff_gpu_64_size;
        LibDesc[4].start = reinterpret_cast<const char*>
            (builtins_gpugen_comm);
        LibDesc[4].size  = builtins_gpugen_comm_size;
        LibDesc[5].start = reinterpret_cast<const char*>
            (builtins_gpugen_diff_gpu_64);
        LibDesc[5].size  = builtins_gpugen_diff_gpu_64_size;
        LibDescSize = 6;
        break;

    case GPU_Library_CI:
        // Library order is important!
        LibDesc[0].start = reinterpret_cast<const char*>
            (builtins_CI_comm);
        LibDesc[0].size  = builtins_CI_comm_size;
        LibDesc[1].start = reinterpret_cast<const char*>
            (builtins_CI_diff_gpu);
        LibDesc[1].size  = builtins_CI_diff_gpu_size;
        LibDesc[2].start = reinterpret_cast<const char*>
            (builtins_SI_comm);
        LibDesc[2].size  = builtins_SI_comm_size;
        LibDesc[3].start = reinterpret_cast<const char*>
            (builtins_SI_diff_gpu);
        LibDesc[3].size  = builtins_SI_diff_gpu_size;
        LibDesc[4].start = reinterpret_cast<const char*>
            (builtins_gpucommon_comm);
        LibDesc[4].size  = builtins_gpucommon_comm_size;
        LibDesc[5].start = reinterpret_cast<const char*>
            (builtins_gpucommon_diff_gpu);
        LibDesc[5].size  = builtins_gpucommon_diff_gpu_size;
        LibDesc[6].start = reinterpret_cast<const char*>
            (builtins_gpugen_comm);
        LibDesc[6].size  = builtins_gpugen_comm_size;
        LibDesc[7].start = reinterpret_cast<const char*>
            (builtins_gpugen_diff_gpu);
        LibDesc[7].size  = builtins_gpugen_diff_gpu_size;
        LibDescSize = 8;
        break;

    case GPU64_Library_CI:
        // Library order is important!
        LibDesc[0].start = reinterpret_cast<const char*>
            (builtins_CI_comm);
        LibDesc[0].size  = builtins_CI_comm_size;
        LibDesc[1].start = reinterpret_cast<const char*>
            (builtins_CI_diff_gpu_64);
        LibDesc[1].size  = builtins_CI_diff_gpu_64_size;
        LibDesc[2].start = reinterpret_cast<const char*>
            (builtins_SI_comm);
        LibDesc[2].size  = builtins_SI_comm_size;
        LibDesc[3].start = reinterpret_cast<const char*>
            (builtins_SI_diff_gpu_64);
        LibDesc[3].size  = builtins_SI_diff_gpu_64_size;
        LibDesc[4].start = reinterpret_cast<const char*>
            (builtins_gpucommon_comm);
        LibDesc[4].size  = builtins_gpucommon_comm_size;
        LibDesc[5].start = reinterpret_cast<const char*>
            (builtins_gpucommon_diff_gpu_64);
        LibDesc[5].size  = builtins_gpucommon_diff_gpu_64_size;
        LibDesc[6].start = reinterpret_cast<const char*>
            (builtins_gpugen_comm);
        LibDesc[6].size  = builtins_gpugen_comm_size;
        LibDesc[7].start = reinterpret_cast<const char*>
            (builtins_gpugen_diff_gpu_64);
        LibDesc[7].size  = builtins_gpugen_diff_gpu_64_size;
        LibDescSize = 8;
        break;

#endif // WITH_TARGET_AMDIL

#if defined(WITH_TARGET_X86)
    case CPU64_Library_Generic:
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_cpucommon_x86_64);
        LibDesc[0].size  = builtins_cpucommon_x86_64_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_cpugen_x86_64);
        LibDesc[1].size  = builtins_cpugen_x86_64_size;
        LibDescSize = 2;
        break;

    case CPU64_Library_AVX:
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_avx_x86_64);
        LibDesc[0].size  = builtins_avx_x86_64_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_cpucommon_x86_64);
        LibDesc[1].size  = builtins_cpucommon_x86_64_size;
        LibDesc[2].start = reinterpret_cast<const char*>(builtins_cpugen_x86_64);
        LibDesc[2].size  = builtins_cpugen_x86_64_size;
        LibDescSize = 3;
        break;

    case CPU64_Library_FMA4:
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_fma4_x86_64);
        LibDesc[0].size  = builtins_fma4_x86_64_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_avx_x86_64);
        LibDesc[1].size  = builtins_avx_x86_64_size;
        LibDesc[2].start = reinterpret_cast<const char*>(builtins_cpucommon_x86_64);
        LibDesc[2].size  = builtins_cpucommon_x86_64_size;
        LibDesc[3].start = reinterpret_cast<const char*>(builtins_cpugen_x86_64);
        LibDesc[3].size  = builtins_cpugen_x86_64_size;
        LibDescSize = 4;
        break;

    case CPU_Library_Generic:
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_cpucommon_x86);
        LibDesc[0].size  = builtins_cpucommon_x86_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_cpugen_x86);
        LibDesc[1].size  = builtins_cpugen_x86_size;
        LibDescSize = 2;
        break;

    case CPU_Library_AVX:
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_avx_x86);
        LibDesc[0].size  = builtins_avx_x86_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_cpucommon_x86);
        LibDesc[1].size  = builtins_cpucommon_x86_size;
        LibDesc[2].start = reinterpret_cast<const char*>(builtins_cpugen_x86);
        LibDesc[2].size  = builtins_cpugen_x86_size;
        LibDescSize = 3;
        break;

    case CPU_Library_FMA4:
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_fma4_x86);
        LibDesc[0].size  = builtins_fma4_x86_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_avx_x86);
        LibDesc[1].size  = builtins_avx_x86_size;
        LibDesc[2].start = reinterpret_cast<const char*>(builtins_cpucommon_x86);
        LibDesc[2].size  = builtins_cpucommon_x86_size;
        LibDesc[3].start = reinterpret_cast<const char*>(builtins_cpugen_x86);
        LibDesc[3].size  = builtins_cpugen_x86_size;
        LibDescSize = 4;
        break;
#endif // WITH_TARGET_X86

#if defined(WITH_TARGET_ARM)
    case CPU_Library_Generic:
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_cpucommon_arm);
        LibDesc[0].size  = builtins_cpucommon_arm_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_cpugen_arm);
        LibDesc[1].size  = builtins_cpugen_arm_size;
        LibDescSize = 2;
        break;
#endif // WITH_TARGET_ARM

#if defined(WITH_TARGET_HSAIL)
    case GPU_Library_HSAIL:
        // Library order is important!
        LibDesc[0].start = reinterpret_cast<const char*>(builtins_gcn);
        LibDesc[0].size  = builtins_gcn_size;
        LibDesc[1].start = reinterpret_cast<const char*>(builtins_hsail_amd_ci);
        LibDesc[1].size  = builtins_hsail_amd_ci_size;
        LibDesc[2].start = reinterpret_cast<const char*>(builtins_hsail);
        LibDesc[2].size  = builtins_hsail_size;
        LibDesc[3].start = reinterpret_cast<const char*>(builtins_ocml);
        LibDesc[3].size  = builtins_ocml_size;
        LibDesc[4].start = reinterpret_cast<const char*>(builtins_spirv);
        LibDesc[4].size  = builtins_spirv_size;
        LibDescSize = 5;
        break;
#endif // WITH_TARGET_HSAIL

    default:
        // Failed
        return 1;   // 
    }
    return 0;
}

} // namespace amd
