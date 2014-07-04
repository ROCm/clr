//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_X64_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_X64_0_8_H_

#define CPU_MAPPING_LIB(A, B, C, D, E) { #A, #B, #C, D, 0, E, LP64_SWITCH(false, true), LP64_SWITCH(false, true), FAMILY_X64}
#define CPU_MAPPING(A, B, C, D) CPU_MAPPING_LIB(A, B, C, amd::CPU64_Library_Generic, D)
#define NCPU_MAPPING_LIB(A, B, C, D, E) { #A, #B, #C, D, 0, E, false, false, FAMILY_X64}
#define NCPU_MAPPING(A, B, C, D) { #A, #B, #C, amd::CPU64_Library_Generic, 0, D, false, false, FAMILY_X64}
static const TargetMapping X64TargetMapping_0_8[] = {
  UnknownTarget,
  CPU_MAPPING(X64,         Generic,     generic, 0x1),
  CPU_MAPPING(NetBurst,    Prescott,    prescott, 0x1),
  CPU_MAPPING(Xeon,        Nocona,      nocona, 0x1),
  CPU_MAPPING(Core,        Core2,       core2, 0x1),
  CPU_MAPPING(Core,        Penryn,      penryn, 0x1),
  CPU_MAPPING(Nehalem,     Corei7,      corei7, 0x1),
  CPU_MAPPING(Nehalem,     Nehalem,     nehalem, 0x1),
  CPU_MAPPING(Nehalem,     Westmere,    westmere, 0x1),
  NCPU_MAPPING_LIB(SandyBridge, Corei7_AVX,  sandybridge, amd::CPU64_Library_AVX, 0x2 | 0x1),  // LLVM 2.9 only
  CPU_MAPPING_LIB(SandyBridge, Corei7_AVX,  corei7-avx, amd::CPU64_Library_AVX, 0x2 | 0x1),
  CPU_MAPPING(SandyBridge, IvyBridge,   core-avx-i, 0x2 | 0x1), // LLVM 3.0
  CPU_MAPPING(Haswell,     Haswell,     core-avx2, 0x4 | 0x2 | 0x1), // LLVM 3.0
  CPU_MAPPING(K8,          K8,          k8, 0x1),
  CPU_MAPPING(K8,          Opteron,     opteron, 0x1),
  CPU_MAPPING(K8,          Athlon64,    athlon64, 0x1),
  CPU_MAPPING(K8,          AthlonFX,    athlon-fx, 0x1),
  CPU_MAPPING(K8,          K8_SSE3,     k8-sse3, 0x1),
  CPU_MAPPING(K8,          Opteron_SSE3,opteron-sse3, 0x1),
  CPU_MAPPING(K8,          Athlon64SSE3,athlon64-sse3, 0x1),
  CPU_MAPPING(K10,         AMDFAM10,    amdfam10, 0x1),
  NCPU_MAPPING(K10,         Barcelona,  barcelona, 0x1),
  NCPU_MAPPING(K10,         Istanbul,    istanbul, 0x1),
  NCPU_MAPPING(K10,         Shanghai,    shanghai, 0x1),
  CPU_MAPPING(Family14h,   Bobcat, btver1, 0x1),
  CPU_MAPPING_LIB(Family15h, Bulldozer, bdver1, amd::CPU64_Library_FMA4, 0x8 | 0x1),
  CPU_MAPPING_LIB(Family15h, Piledriver, bdver2, amd::CPU64_Library_FMA4, 0x8 | 0x4 | 0x1),
  CPU_MAPPING(Atom, Atom, atom, 0x1),
  InvalidTarget
};
#undef CPU_MAPPING
#undef NCPU_MAPPING
#undef CPU_MAPPING_LIB
#undef NCPU_MAPPING_LIB
#endif // _CL_UTILS_TARGET_MAPPINGS_X64_0_8_H_
