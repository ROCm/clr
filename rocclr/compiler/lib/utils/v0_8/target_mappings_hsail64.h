//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_HSAIL64_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_HSAIL64_0_8_H_

#include "si_id.h"
#include "kv_id.h"
#include "ci_id.h"
#include "ai_id.h"
#include "rv_id.h"
#include "atiid.h"

static const TargetMapping HSAIL64TargetMapping_0_8[] = {
  UnknownTarget,
  { "KV", "Spectre",   "GFX7", amd::GPU_Library_HSAIL, KV_SPECTRE_A0,   F_CI_BASE, true, true,  FAMILY_KV },
  { "KV", "Spooky",    "GFX7", amd::GPU_Library_HSAIL, KV_SPOOKY_A0,    F_CI_BASE, true, true,  FAMILY_KV },
  { "KV", "Kalindi",   "GFX7", amd::GPU_Library_HSAIL, KB_KALINDI_A0,   F_CI_BASE, true, true,  FAMILY_KV },
  { "KV", "Mullins",   "GFX7", amd::GPU_Library_HSAIL, ML_GODAVARI_A0,  F_CI_BASE, true, true,  FAMILY_KV },
  { "CI", "Bonaire",   "GFX7", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A0, F_CI_BASE, true, false, FAMILY_CI },
  { "CI", "Bonaire",   "GFX7", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A1, F_CI_BASE, true, true,  FAMILY_CI },
  { "CI", "Hawaii",    "GFX7", amd::GPU_Library_HSAIL, CI_HAWAII_P_A0,  F_CI_BASE, true, true,  FAMILY_CI },
  { "VI", "Iceland",   "GFX8", amd::GPU_Library_HSAIL, VI_ICELAND_M_A0, F_VI_BASE, true, true,  FAMILY_VI },
  { "VI", "Tonga",     "GFX8", amd::GPU_Library_HSAIL, VI_TONGA_P_A0,   F_VI_BASE, true, true,  FAMILY_VI },

  UnknownTarget,
  UnknownTarget,
  { "CZ", "Carrizo",   "GFX8", amd::GPU_Library_HSAIL, CARRIZO_A0,      F_VI_BASE, true, true,  FAMILY_CZ },
  { "VI", "Fiji",      "GFX8", amd::GPU_Library_HSAIL, VI_FIJI_P_A0,    F_VI_BASE, true, true,  FAMILY_VI },
  { "CZ", "Stoney",    "GFX8", amd::GPU_Library_HSAIL, STONEY_A0,       F_VI_BASE, true, true,  FAMILY_CZ },
  { "VI", "Baffin",    "GFX8", amd::GPU_Library_HSAIL, VI_BAFFIN_M_A0,  F_VI_BASE, true, true,  FAMILY_VI },
  { "VI", "Baffin",    "GFX8", amd::GPU_Library_HSAIL, VI_BAFFIN_M_A1,  F_VI_BASE, true, true,  FAMILY_VI },
  { "VI", "Ellesmere", "GFX8", amd::GPU_Library_HSAIL, VI_ELLESMERE_P_A0, F_VI_BASE, true, true,  FAMILY_VI },
  { "VI", "Ellesmere", "GFX8", amd::GPU_Library_HSAIL, VI_ELLESMERE_P_A1, F_VI_BASE, true, true,  FAMILY_VI },
#ifndef BRAHMA
  { "AI", "gfx900",    "GFX9", amd::GPU_Library_HSAIL, AI_GREENLAND_P_A0, F_AI_BASE, true, true,  FAMILY_AI },
  { "VI", "gfx804",    "GFX8", amd::GPU_Library_HSAIL, VI_LEXA_V_A0,    F_VI_BASE, true, true,  FAMILY_VI },
  { "RV", "gfx901",    "GFX9", amd::GPU_Library_HSAIL, RAVEN_A0,        F_AI_BASE, true, true,  FAMILY_RV },
#else
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
#endif
  InvalidTarget
};

#endif // _CL_UTILS_TARGET_MAPPINGS_HSAIL64_0_8_H_
