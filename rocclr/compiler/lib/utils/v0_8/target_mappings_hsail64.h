//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_HSAIL64_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_HSAIL64_0_8_H_

#include "inc/asic_reg/si_id.h"
#include "inc/asic_reg/kv_id.h"
#include "inc/asic_reg/ci_id.h"
#include "inc/asic_reg/ai_id.h"
#include "inc/asic_reg/atiid.h"

static const TargetMapping HSAIL64TargetMapping_0_8[] = {
  UnknownTarget,
  { "KV", "Spectre", "CI", amd::GPU_Library_HSAIL, KV_SPECTRE_A0,   F_CI_BASE, true, true,  FAMILY_KV },
  { "KV", "Spooky",  "CI", amd::GPU_Library_HSAIL, KV_SPOOKY_A0,    F_CI_BASE, true, true,  FAMILY_KV },
  { "KV", "Kalindi", "CI", amd::GPU_Library_HSAIL, KB_KALINDI_A0,   F_CI_BASE, true, true,  FAMILY_KV },
  { "KV", "Mullins", "CI", amd::GPU_Library_HSAIL, ML_GODAVARI_A0,  F_CI_BASE, true, true,  FAMILY_KV },
  { "CI", "Bonaire", "CI", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A0, F_CI_BASE, true, false, FAMILY_CI },
  { "CI", "Bonaire", "CI", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A1, F_CI_BASE, true, true,  FAMILY_CI },
  { "CI", "Hawaii",  "CI", amd::GPU_Library_HSAIL, CI_HAWAII_P_A0,  F_CI_BASE, true, true,  FAMILY_CI },
  { "VI", "Iceland", "VI", amd::GPU_Library_HSAIL, VI_ICELAND_M_A0, F_VI_BASE, true, true,  FAMILY_VI },
  { "VI", "Tonga",   "VI", amd::GPU_Library_HSAIL, VI_TONGA_P_A0,   F_VI_BASE, true, true,  FAMILY_VI },

  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  { "VI", "Bee",   "VI",   amd::GPU_Library_HSAIL, VI_LEXA_V_A0,    F_VI_BASE, true, true,  FAMILY_VI },
  { "CZ", "Carrizo", "VI", amd::GPU_Library_HSAIL, CARRIZO_A0,      F_VI_BASE, true, true,  FAMILY_CZ },
  { "VI", "Fiji",    "VI", amd::GPU_Library_HSAIL, VI_FIJI_P_A0,    F_VI_BASE, true, true,  FAMILY_VI },
  { "CZ", "Stoney",  "VI", amd::GPU_Library_HSAIL, STONEY_A0,       F_VI_BASE, true, true,  FAMILY_CZ },
  { "VI", "Baffin",  "VI", amd::GPU_Library_HSAIL, VI_BAFFIN_M_A0,  F_VI_BASE, true, true,  FAMILY_VI },
  { "VI", "Ellesmere", "VI", amd::GPU_Library_HSAIL, VI_ELLESMERE_P_A0, F_VI_BASE, true, true,  FAMILY_VI },
  InvalidTarget
};

#endif // _CL_UTILS_TARGET_MAPPINGS_HSAIL64_0_8_H_
