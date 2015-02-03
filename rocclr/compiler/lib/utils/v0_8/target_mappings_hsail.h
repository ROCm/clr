//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_HSAIL_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_HSAIL_0_8_H_

#include "inc/asic_reg/si_id.h"
#include "inc/asic_reg/kv_id.h"
#include "inc/asic_reg/ci_id.h"
#include "inc/asic_reg/cz_id.h"
#include "inc/asic_reg/atiid.h"

static const TargetMapping HSAILTargetMapping_0_8[] = {
  UnknownTarget,
  { "KV", "Spectre", "generic", amd::GPU_Library_HSAIL, KV_SPECTRE_A0,   0, true, true,  FAMILY_KV },
  { "KV", "Spooky",  "generic", amd::GPU_Library_HSAIL, KV_SPOOKY_A0,    0, true, true,  FAMILY_KV },
  { "KV", "Kalindi", "generic", amd::GPU_Library_HSAIL, KB_KALINDI_A0,   0, true, true,  FAMILY_KV },
  { "KV", "Mullins", "generic", amd::GPU_Library_HSAIL, ML_GODAVARI_A0,  0, true, true,  FAMILY_KV },
  { "CI", "Bonaire", "generic", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A0, 0, true, false, FAMILY_CI },
  { "CI", "Bonaire", "generic", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A1, 0, true, true,  FAMILY_CI },
  { "CI", "Hawaii",  "generic", amd::GPU_Library_HSAIL, CI_HAWAII_P_A0,  0, true, true,  FAMILY_CI },
  { "VI", "Iceland", "generic", amd::GPU_Library_HSAIL, VI_ICELAND_M_A0, 0, true, true,  FAMILY_VI },
  { "VI", "Tonga",   "generic", amd::GPU_Library_HSAIL, VI_TONGA_P_A0,   0, true, true,  FAMILY_VI },

  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  { "CZ", "Carrizo", "generic", amd::GPU_Library_HSAIL, CARRIZO_A0,      0, true, true,  FAMILY_CZ },
  { "VI", "Fiji",    "generic", amd::GPU_Library_HSAIL, VI_FIJI_P_A0,    0, true, true,  FAMILY_VI },
  InvalidTarget
};

#endif // _CL_UTILS_TARGET_MAPPINGS_HSAIL_0_8_H_
