//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_AMDIL64_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_AMDIL64_0_8_H_

#include "inc/asic_reg/si_id.h"
#include "inc/asic_reg/kv_id.h"
#include "inc/asic_reg/ci_id.h"
#include "inc/asic_reg/vi_id.h"
#include "inc/asic_reg/cz_id.h"
#include "inc/asic_reg/atiid.h"

static const TargetMapping AMDIL64TargetMapping_0_8[] = {
  UnknownTarget,
  { "SI", "Tahiti",    "tahiti",    amd::GPU64_Library_SI, SI_TAHITI_P_A11,    F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",    amd::GPU64_Library_SI, SI_TAHITI_P_A0,     F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",    amd::GPU64_Library_SI, SI_TAHITI_P_A21,    F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",    amd::GPU64_Library_SI, SI_TAHITI_P_B0,     F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",    amd::GPU64_Library_SI, SI_TAHITI_P_A22,    F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",    amd::GPU64_Library_SI, SI_TAHITI_P_B1,     F_SI_64BIT_PTR, true , true,  FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU64_Library_SI, SI_PITCAIRN_PM_A11, F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU64_Library_SI, SI_PITCAIRN_PM_A0,  F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU64_Library_SI, SI_PITCAIRN_PM_A12, F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU64_Library_SI, SI_PITCAIRN_PM_A1,  F_SI_64BIT_PTR, true , true,  FAMILY_SI },
  { "SI", "Capeverde", "capeverde", amd::GPU64_Library_SI, SI_CAPEVERDE_M_A11, F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Capeverde", "capeverde", amd::GPU64_Library_SI, SI_CAPEVERDE_M_A0,  F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Capeverde", "capeverde", amd::GPU64_Library_SI, SI_CAPEVERDE_M_A12, F_SI_64BIT_PTR, true , false, FAMILY_SI },
  { "SI", "Capeverde", "capeverde", amd::GPU64_Library_SI, SI_CAPEVERDE_M_A1,  F_SI_64BIT_PTR, true , true,  FAMILY_SI },
  { "KV", "Spectre",   "spectre",   amd::GPU64_Library_CI, KV_SPECTRE_A0,      F_SI_64BIT_PTR, true,  true,  FAMILY_KV },
  { "KV", "Spooky",    "spooky",    amd::GPU64_Library_CI, KV_SPOOKY_A0,       F_SI_64BIT_PTR, true,  true,  FAMILY_KV },
  { "KV", "Kalindi",   "kalindi",   amd::GPU64_Library_CI, KB_KALINDI_A0,      F_SI_64BIT_PTR, true,  true,  FAMILY_KV },
  { "CI", "Hawaii",    "hawaii",    amd::GPU64_Library_CI, CI_HAWAII_P_A0,     F_SI_64BIT_PTR, true,  true,  FAMILY_CI },
  { "KV", "Mullins",   "mullins",   amd::GPU64_Library_CI, ML_GODAVARI_A0,     F_SI_64BIT_PTR, true,  true,  FAMILY_KV },
  { "SI", "Oland",     "oland",     amd::GPU64_Library_SI, SI_OLAND_M_A0,      F_SI_64BIT_PTR, true,  true,  FAMILY_SI },
  { "CI", "Bonaire",   "bonaire",   amd::GPU64_Library_CI, CI_BONAIRE_M_A0,    F_SI_64BIT_PTR, true,  false, FAMILY_CI },
  { "SI", "Hainan",    "hainan",    amd::GPU64_Library_SI, SI_HAINAN_V_A0,     F_SI_64BIT_PTR, true,  true,  FAMILY_SI },

  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  { "CZ", "Carrizo",   "carrizo",   amd::GPU64_Library_CI, CARRIZO_A0,         F_SI_64BIT_PTR, true,  true,  FAMILY_CZ },
  { "VI", "Iceland",    "iceland",  amd::GPU64_Library_CI, VI_ICELAND_M_A0,    F_SI_64BIT_PTR, true,  true,  FAMILY_VI },
  { "VI", "Tonga",      "tonga",    amd::GPU64_Library_CI, VI_TONGA_P_A0,      F_SI_64BIT_PTR, true,  true,  FAMILY_VI },
  { "CI", "Bonaire",   "bonaire",   amd::GPU64_Library_CI, CI_BONAIRE_M_A0,    F_SI_64BIT_PTR, true,  true,  FAMILY_CI },
  { "VI", "Fiji",      "fiji",      amd::GPU64_Library_CI, VI_FIJI_P_A0,       F_SI_64BIT_PTR, true,  true,  FAMILY_VI },
  InvalidTarget
};

#endif // _CL_UTILS_TARGET_MAPPINGS_AMDIL64_0_8_H_
