//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_X86_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_X86_0_8_H_
#define CPU_MAPPING_LIB(A, B, C, D, E) { #A, #B, #C, D, 0, E, true, true, FAMILY_X86}
#define CPU_MAPPING(A, B, C, D) CPU_MAPPING_LIB(A, B, C, amd::CPU_Library_Generic, D)
#define NCPU_MAPPING_LIB(A, B, C, D, E) { #A, #B, #C, D, 0, E, false, false, FAMILY_X86}
#define NCPU_MAPPING(A, B, C, D) { #A, #B, #C, amd::CPU_Library_Generic, 0, D, false, false, FAMILY_X86}
static const TargetMapping X86TargetMapping_0_8[] = {
  UnknownTarget,
  CPU_MAPPING(X86,         Generic,     generic, 0),
  // This has to be specified manually since GCC defines i386 as a macro.
  { "X86", "i386", "i386", amd::CPU_Library_Generic, 0, 0, true, true, FAMILY_X86 },
  CPU_MAPPING(X86,         i486,        i486, 0),
  CPU_MAPPING(X86,         i586,        i586, 0),
  CPU_MAPPING(Pentium,     Pentium,     pentium, 0),
  CPU_MAPPING(Pentium_MMX, Pentium_MMX, pentium-mmx, 0),
  CPU_MAPPING(X86,         i686,        i686, 0),
  CPU_MAPPING(PentiumPro,  PentiumPro,  pentiumpro, 0),
  CPU_MAPPING(Pentium2,    Pentium2,    pentium2, 0),
  CPU_MAPPING(Pentium3,    Pentium3,    pentium3, 0),
  CPU_MAPPING(Pentium3m,   Pentium3m,   pentium3m, 0),
  CPU_MAPPING(Pentium_M,   Pentium_M,   pentium-m, 0x1),
  CPU_MAPPING(NetBurst,    Pentium4,    pentium4, 0x1),
  CPU_MAPPING(NetBurst,    Pentium4m,   pentium4m, 0x1),
  CPU_MAPPING(Pentium_M,   Yonah,       yonah, 0x1),
  CPU_MAPPING(Pentium4,    Prescott,    prescott, 0x1),
  CPU_MAPPING(Xeon,        Nocona,      nocona, 0x1),
  CPU_MAPPING(Core,        Core2,       core2, 0x1),
  CPU_MAPPING(Core,        Penryn,      penryn, 0x1),
  CPU_MAPPING(Nehalem,     Corei7,      corei7, 0x1), // Corei3 and Corei5 also
  CPU_MAPPING(Nehalem,     Nehalem,     nehalem, 0x1),
  CPU_MAPPING(Nehalem,     Westmere,    westmere, 0x1),
  NCPU_MAPPING_LIB(SandyBridge, Corei7_AVX,  sandybridge, amd::CPU64_Library_AVX, 0x2 | 0x1),  // LLVM 2.9 only
  CPU_MAPPING(SandyBridge, Corei7_AVX,  corei7-avx, 0x2 | 0x1), // LLVM 3.0 only
  CPU_MAPPING(SandyBridge, IvyBridge,   core-avx-i, 0x2 | 0x1), // LLVM 3.0 only
  CPU_MAPPING(Haswell,     Haswell,     core-avx2, 0x4 | 0x2 | 0x1), // LLVM 3.0 only
  CPU_MAPPING(K6,          K6,          k6, 0),
  CPU_MAPPING(K6,          K6_2,        k6-2, 0),
  CPU_MAPPING(K6,          K6_3,        k6-3, 0),
  CPU_MAPPING(K7,          Athlon,      athlon, 0),
  CPU_MAPPING(K7,          AthlonTBIRD, athlon-tbird, 0),
  CPU_MAPPING(K7,          Athlon4,     athlon-4, 0),
  CPU_MAPPING(K7,          AthlonXP,    athlon-xp, 0),
  CPU_MAPPING(K7,          AthlonMP,    athlon-mp, 0),
  CPU_MAPPING(K8,          K8,          k8, 0x1),
  CPU_MAPPING(K8,          Opteron,     opteron, 0x1),
  CPU_MAPPING(K8,          Athlon64,    athlon64, 0x1),
  CPU_MAPPING(K8,          AthlonFX,    athlon-fx, 0x1),
  CPU_MAPPING(K8,          K8_SSE3,     k8-sse3, 0x1),
  CPU_MAPPING(K8,          Opteron_SSE3,opteron-sse3, 0x1),
  CPU_MAPPING(K8,          Athlon64SSE3,athlon64-sse3, 0x1),
  CPU_MAPPING(K10,         AMDFAM10,    amdfam10, 0x1),
  NCPU_MAPPING(K10,        Barcelona,   barcelona, 0x1),
  NCPU_MAPPING(K10,        Istanbul,    istanbul, 0x1),
  NCPU_MAPPING(K10,        Shanghai,    shanghai, 0x1),
  CPU_MAPPING(Winchip,     Winchip_C6,  winchip-c6, 0),
  CPU_MAPPING(Winchip,     Winchip2,    winchip2, 0),
  CPU_MAPPING(Via,         C3,          c3, 0),
  CPU_MAPPING(Via,         C3_2,        c3-2, 0),
  CPU_MAPPING(Family14h,   Bobcat,      btver1, 0x1), // LLVM 3.1 only
  CPU_MAPPING_LIB(Family15h, Bulldozer, bdver1, amd::CPU_Library_FMA4, 0x8 | 0x1), // LLVM 3.1 only
  CPU_MAPPING_LIB(Family15h, Piledriver, bdver2, amd::CPU_Library_FMA4, 0x8 | 0x4 | 0x1), // LLVM 3.1 only
  CPU_MAPPING(Atom,        Atom,        atom, 0x1), // LLVM 3.1 only
  InvalidTarget
};
#undef CPU_MAPPING
#undef NCPU_MAPPING
#undef CPU_MAPPING_LIB
#undef NCPU_MAPPING_LIB

#endif // _CL_UTILS_TARGET_MAPPINGS_X86_0_8_H_
