//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_0_8_H_
#include "top.hpp"
#include "library.hpp"
#include "aclTypes.h"
#ifndef FAMILY_UNKNOWN
#define FAMILY_UNKNOWN 0
#endif

#ifndef FAMILY_X86
#define FAMILY_X86 1
#endif

#ifndef FAMILY_X64
#define FAMILY_X64 2
#endif

#define ARRAY_SIZEOF(A) (sizeof(A)/sizeof(A[0]))

typedef struct _target_mappings_rec {
  const char* family_name;
  const char* chip_name;
  const char* codegen_name;
  amd::LibrarySelector lib;
  unsigned chip_enum;
  uint64_t chip_options;
  bool supported; // a false value means this device is not supported.
  bool default_chip; // Chip to select if multiple chips with the same name exist.
  unsigned family_enum; // Only used for GPU devices currently, for CPU we should put features.
  bool xnack_supported; // XNACK support as per http://confluence.amd.com/pages/viewpage.action?spaceKey=ASLC&title=AMDGPU+Target+Names 
} TargetMapping;

const TargetMapping UnknownTarget = { "UnknownFamily", "UnknownChip", "UnknownCodeGen",
  amd::LibraryUndefined, 0, 0, false, false, FAMILY_UNKNOWN};
const TargetMapping InvalidTarget = { NULL, NULL, NULL,
  amd::LibraryUndefined, 0, 0, false, false, FAMILY_UNKNOWN};

typedef struct _family_map_rec {
  const TargetMapping*  target;
  const char* architecture;
  const char* triple;
  unsigned children_size;
} FamilyMapping;
const FamilyMapping UnknownFamily = { NULL, "UnknownFamily", "unknown", 0 };
const FamilyMapping InvalidFamily = { NULL, NULL, NULL, 0 };
typedef enum  {
  F_CPU_CMOV      =  1,
  F_CPU_POPCNT    =  2,
  F_CPU_MMX       =  3,
  F_CPU_SSE1      =  4,
  F_CPU_SSE2      =  5,
  F_CPU_SSE3      =  6,
  F_CPU_SSSE3     =  7,
  F_CPU_SSE41     =  8,
  F_CPU_SSE42     =  9,
  F_CPU_SSE4A     = 10,
  F_CPU_3DNow     = 11,
  F_CPU_3DNowA    = 12,
  F_CPU_64Bit     = 13,
  F_CPU_SBTMem    = 14,
  F_CPU_FUAMem    = 15,
  F_CPU_AVX       = 16,
  F_CPU_CLMUL     = 17,
  F_CPU_VUAMem    = 18,
  F_CPU_AES       = 19,
  F_CPU_CXCHG16B  = 20,
  F_CPU_AVX2      = 21,
  F_CPU_FMA3      = 22,
  F_CPU_FMA4      = 23,
  F_CPU_MOVBE     = 24,
  F_CPU_RDRAND    = 25,
  F_CPU_F16C      = 26,
  F_CPU_64BitMode = 27,
  F_CPU_LZCNT     = 28,
  F_CPU_BMI       = 29,
  F_CPU_BMI2      = 30, // LLVM 3.1 only
  F_CPU_LeaForSP  = 31, // LLVM 3.1 only
  F_CPU_FSGSBASE  = 32, // LLVM 3.1 only
  F_CPU_XOP       = 33, // LLVM 3.1 only
  F_CPU_ATOM      = 34, // LLVM 3.1 only
  F_CPU_LAST      = 35
} CPUCodeGenFlags;

static const char* CPUCodeGenFlagTable[] =
{
  "cmov",
  "popcnt",
  "mmx",
  "sse",
  "sse2",
  "sse3",
  "ssse3",
  "sse41",
  "sse42",
  "sse4a",
  "3dnow",
  "3dnowa",
  "64bit",
  "slow-bt-mem",
  "fast-unaligned-mem",
  "avx",
  "clmul",
  "vector-unaligned-mem",
  "aes",
  "cmpxchg16b",
  "avx2",
  "fma3",
  "fma4",
  "movbe",
  "rdrand",
  "f16c",
  "fsgsbase",
  "lzcnt",
  "bmi",
  "bmi2",
  "lea-sp",
  "64bit-mode",
  "xop",
  "atom"
};

typedef enum {
  // Bits for each feature.
  F_FP64              = 0x0001,
  F_BYTE_ADDRESSABLE  = 0x0002,
  F_BARRIER_DETECT    = 0x0004,
  F_IMAGES            = 0x0008,
  F_MULTI_UAV         = 0x0010,
  F_MACRO_DB          = 0x0020,
  F_NO_ALIAS          = 0x0040,
  F_NO_INLINE         = 0x0080,
  F_64BIT_PTR         = 0x0100,
  F_32ON64BIT_PTR     = 0x0200,
  F_DEBUG             = 0x0400,
  F_MWGS_256          = 0x0800,
  F_MWGS_128          = 0x1000,
  F_MWGS_64           = 0x2000,
  F_MWGS_32           = 0x4000,
  F_MWGS_16           = 0x8000,
  F_MD_30             = 0x10000,
  F_STACK_UAV         = 0x20000,
  F_MACRO_CALL        = 0x40000,
  // Bitmasks for each device type.
  F_RV7XX_BASE        = F_MACRO_DB|F_BARRIER_DETECT|F_MD_30,
  F_RV710             = F_RV7XX_BASE|F_MWGS_32,
  F_RV730             = F_RV7XX_BASE|F_MWGS_16,
  F_RV770             = F_RV7XX_BASE|F_MWGS_64|F_FP64,
  F_EG_BASE           = F_BYTE_ADDRESSABLE|F_IMAGES|F_MACRO_DB|F_MD_30,
  F_EG_EXT            = F_EG_BASE|F_FP64|F_MWGS_256,
  F_CEDAR             = F_EG_BASE|F_MWGS_128,
  F_REDWOOD           = F_EG_BASE|F_MWGS_256,
  F_JUNIPER           = F_EG_BASE|F_MWGS_256,
  F_NI_BASE           = F_EG_BASE|F_MWGS_256,
  F_NI_EXT            = F_NI_BASE|F_FP64,
  F_SI_BASE           = F_NI_EXT|F_STACK_UAV|F_MACRO_CALL,
  F_SI_64BIT_PTR      = F_SI_BASE|F_64BIT_PTR
} GPUCodeGenFlags;

typedef enum {
  // Bits for each feature.
  F_FP32_DENORMS      = 0x0001,
  // Bitmasks for each device type.
  F_CI_BASE           = 0,
  F_VI_BASE           = F_CI_BASE | F_FP32_DENORMS,
  F_AI_BASE           = F_VI_BASE
} HSAILCodeGenFlags;

static const char* GPUCodeGenFlagTable[] = {
  "fp64",
  "byte_addressable_store",
  "barrier_detect",
  "images",
  "multi_uav",
  "macrodb",
  "noalias",
  "no-inline",
  "64bitptr",
  "small-global-objects",
  "debug",
  "mwgs-3-256-1-1",
  "mwgs-3-128-1-1",
  "mwgs-3-64-1-1",
  "mwgs-3-32-1-1",
  "mwgs-3-16-1-1",
  "metadata30",
  "stack-uav",
  "macro-call"
};

static const char* HSAILCodeGenFlagTable[] = {
  "fp32-denormals"
};

static const char* calTargetMapping[] = {
      "RV600", "RV610", "RV630", "RV670",
      "RV770", "RV770", "RV710", "RV730",
      "Cypress", "Juniper", "Redwood", "Cedar",
      "WinterPark", "BeaverCreek", "Loveland",
      "Cayman", "Kauai", "Barts", "Turks", "Caicos",
      "Tahiti", "Pitcairn", "Capeverde",
      "Devastator", "Scrapper",
      "Oland", "Bonaire",
      "Spectre", "Spooky", "Kalindi",
      "Hainan", "Hawaii",
      "Iceland", "Tonga", "Mullins", "Fiji",
      "Carrizo", "Ellesmere", "Baffin",
      IF(IS_BRAHMA,"","gfx900"),
      "Stoney",
      IF(IS_BRAHMA,"","gfx804"),
      IF(IS_BRAHMA,"","gfx901"),
      IF(IS_BRAHMA,"","gfx902"),
      IF(IS_BRAHMA,"","gfx903"),
    };

#include "utils/v0_8/target_mappings_amdil.h"
#include "utils/v0_8/target_mappings_hsail.h"
#include "utils/v0_8/target_mappings_x86.h"
#include "utils/v0_8/target_mappings_x64.h"
#include "utils/v0_8/target_mappings_amdil64.h"
#include "utils/v0_8/target_mappings_hsail64.h"
// FIXME: Add static asserts to make sure that all of the arrays for TargetMapping match the enum table.

#define DATA_LAYOUT_64BIT "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16" \
                "-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:32:32" \
                "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64" \
                "-v96:128:128-v128:128:128-v192:256:256-v256:256:256" \
                "-v512:512:512-v1024:1024:1024-v2048:2048:2048-a0:0:64" \
                "-n32"

#define DATA_LAYOUT_32BIT "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16" \
                "-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:32:32" \
                "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64" \
                "-v96:128:128-v128:128:128-v192:256:256-v256:256:256" \
                "-v512:512:512-v1024:1024:1024-v2048:2048:2048-a0:0:64" \
                "-n32"

inline const char* getArchitecture(aclDevType arch_id)
{
  switch (arch_id) {
    case aclX86:
      return "x86";
    case aclAMDIL:
      return "amdil";
    case aclHSAIL:
      return "hsail";
    case aclX64:
      return "x86-64";
    case aclHSAIL64:
      return "hsail64";
    case aclAMDIL64:
      return "amdil64";
    default:
      return NULL;
  }
}

inline const char* getTriple(aclDevType arch_id)
{
  switch (arch_id) {
    case aclX86:
#ifdef _WIN32
      return "i686-pc-mingw32-amdopencl";
#else
      return "i686-pc-linux-amdopencl";
#endif
    case aclAMDIL:
      return "amdil-pc-unknown-amdopencl";
    case aclHSAIL:
      return "hsail-pc-unknown-amdopencl";
    case aclX64:
#ifdef _WIN32
      return "x86_64-pc-mingw32-amdopencl";
#else
      return "x86_64-pc-linux-amdopencl";
#endif
    case aclHSAIL64:
      return "hsail64-pc-unknown-amdopencl";
    case aclAMDIL64:
      return "amdil64-pc-unknown-amdopencl";
    default:
      return NULL;
  }
}

// The contents of this array has to match the sequence defined in
// aclDevType_0_8
static const FamilyMapping familySet[] =
{
  UnknownFamily,
  { (const TargetMapping*)&X86TargetMapping_0_8, getArchitecture(aclX86), getTriple(aclX86), ARRAY_SIZEOF(X86TargetMapping_0_8) },
  { (const TargetMapping*)&AMDILTargetMapping_0_8, getArchitecture(aclAMDIL), getTriple(aclAMDIL), ARRAY_SIZEOF(AMDILTargetMapping_0_8) },
  { (const TargetMapping*)&HSAILTargetMapping_0_8, getArchitecture(aclHSAIL), getTriple(aclHSAIL), ARRAY_SIZEOF(HSAILTargetMapping_0_8) },
  { (const TargetMapping*)&X64TargetMapping_0_8, getArchitecture(aclX64), getTriple(aclX64), ARRAY_SIZEOF(X64TargetMapping_0_8) },
  { (const TargetMapping*)&HSAIL64TargetMapping_0_8, getArchitecture(aclHSAIL64), getTriple(aclHSAIL64), ARRAY_SIZEOF(HSAIL64TargetMapping_0_8) },
  { (const TargetMapping*)&AMDIL64TargetMapping_0_8, getArchitecture(aclAMDIL64), getTriple(aclAMDIL64), ARRAY_SIZEOF(AMDIL64TargetMapping_0_8) },
  InvalidFamily
};

#endif // _CL_UTILS_TARGET_MAPPINGS_0_8_H_
