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
#include "nv_id.h"
#include "atiid.h"

static const TargetMapping HSAIL64TargetMapping_0_8[] = {
  UnknownTarget,
  { "KV", "Spectre",   "GFX7", amd::GPU_Library_HSAIL, KV_SPECTRE_A0,   F_CI_BASE, true, true,  FAMILY_KV, false },
  { "KV", "Spooky",    "GFX7", amd::GPU_Library_HSAIL, KV_SPOOKY_A0,    F_CI_BASE, true, true,  FAMILY_KV, false },
  { "KV", "Kalindi",   "GFX7", amd::GPU_Library_HSAIL, KB_KALINDI_A0,   F_CI_BASE, true, true,  FAMILY_KV, false },
  { "KV", "Mullins",   "GFX7", amd::GPU_Library_HSAIL, ML_GODAVARI_A0,  F_CI_BASE, true, true,  FAMILY_KV, false },
  { "CI", "Bonaire",   "GFX7", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A0, F_CI_BASE, true, false, FAMILY_CI, false },
  { "CI", "Bonaire",   "GFX7", amd::GPU_Library_HSAIL, CI_BONAIRE_M_A1, F_CI_BASE, true, true,  FAMILY_CI, false },
  { "CI", "Hawaii",    "GFX7", amd::GPU_Library_HSAIL, CI_HAWAII_P_A0,  F_CI_BASE, true, true,  FAMILY_CI, false },
  { "VI", "Iceland",   "GFX8", amd::GPU_Library_HSAIL, VI_ICELAND_M_A0, F_VI_BASE, true, true,  FAMILY_VI, false },
  { "VI", "Tonga",     "GFX8", amd::GPU_Library_HSAIL, VI_TONGA_P_A0,   F_VI_BASE, true, true,  FAMILY_VI, false },

  UnknownTarget,
  UnknownTarget,
  { "CZ", "Carrizo",   "GFX8", amd::GPU_Library_HSAIL, CARRIZO_A0,      F_VI_BASE, true, true,  FAMILY_CZ, false },
  { "VI", "Fiji",      "GFX8", amd::GPU_Library_HSAIL, VI_FIJI_P_A0,    F_VI_BASE, true, true,  FAMILY_VI, false },
  { "CZ", "Stoney",    "GFX8", amd::GPU_Library_HSAIL, STONEY_A0,       F_VI_BASE, true, true,  FAMILY_CZ, false },
  { "VI", "Baffin",    "GFX8", amd::GPU_Library_HSAIL, VI_BAFFIN_M_A0,  F_VI_BASE, true, false, FAMILY_VI, false },
  { "VI", "Baffin",    "GFX8", amd::GPU_Library_HSAIL, VI_BAFFIN_M_A1,  F_VI_BASE, true, true,  FAMILY_VI, false },
  { "VI", "Ellesmere", "GFX8", amd::GPU_Library_HSAIL, VI_ELLESMERE_P_A0, F_VI_BASE, true, false, FAMILY_VI, false },
  { "VI", "Ellesmere", "GFX8", amd::GPU_Library_HSAIL, VI_ELLESMERE_P_A1, F_VI_BASE, true, true,  FAMILY_VI, false },
#ifndef BRAHMA
  { "AI", "gfx900",    "GFX9", amd::GPU_Library_HSAIL, AI_GREENLAND_P_A0, F_AI_BASE, true, false,  FAMILY_AI, false },
  { "AI", "gfx900",    "GFX9", amd::GPU_Library_HSAIL, AI_GREENLAND_P_A1, F_AI_BASE, true, true,  FAMILY_AI, false },
  { "VI", "gfx804",    "GFX8", amd::GPU_Library_HSAIL, VI_LEXA_V_A0,    F_VI_BASE, true, true,  FAMILY_VI, false },
  { "AI", "gfx901",    "GFX9",  amd::GPU_Library_HSAIL, AI_GREENLAND_P_A0,  F_AI_BASE, true, false,  FAMILY_AI, true },
  { "AI", "gfx901",    "GFX9",  amd::GPU_Library_HSAIL, AI_GREENLAND_P_A1,  F_AI_BASE, true, true,  FAMILY_AI, true },
  { "RV", "gfx902",    "GFX9",  amd::GPU_Library_HSAIL, RAVEN_A0,           F_AI_BASE, true, true,  FAMILY_RV, false },
  { "RV", "gfx903",    "GFX9",  amd::GPU_Library_HSAIL, RAVEN_A0,           F_AI_BASE, true, true,  FAMILY_RV, true },
  { "AI", "gfx906",    "GFX9",  amd::GPU_Library_HSAIL, AI_VEGA20_P_A0,     F_AI_BASE, true, true,  FAMILY_AI, false },
  { "AI", "gfx907",    "GFX9",  amd::GPU_Library_HSAIL, AI_VEGA20_P_A0,     F_AI_BASE, true, true,  FAMILY_AI, true },
  { "NV", "gfx1000",   "GFX10", amd::GPU_Library_HSAIL, NV_NAVI10_P_A0,     F_AI_BASE, true, true,  FAMILY_NV, false },
  { "NV", "gfx1001",   "GFX10", amd::GPU_Library_HSAIL, NV_NAVI10_P_A0,     F_AI_BASE, true, true,  FAMILY_NV, true },
#else
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
#endif
  InvalidTarget
};

#endif // _CL_UTILS_TARGET_MAPPINGS_HSAIL64_0_8_H_
