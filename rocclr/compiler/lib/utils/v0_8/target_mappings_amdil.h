//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_UTILS_TARGET_MAPPINGS_AMDIL_0_8_H_
#define _CL_UTILS_TARGET_MAPPINGS_AMDIL_0_8_H_

#include "inc/asic_reg/evergreen_id.h"
#include "inc/asic_reg/r700id.h"
#include "inc/asic_reg/tn_id.h"
#include "inc/asic_reg/sumo_id.h"
#include "inc/asic_reg/northernisland_id.h"
#include "inc/asic_reg/si_id.h"
#include "inc/asic_reg/kv_id.h"
#include "inc/asic_reg/ci_id.h"
#include "inc/asic_reg/vi_id.h"
#include "inc/asic_reg/cz_id.h"
#include "inc/asic_reg/ai_id.h"
#include "inc/asic_reg/atiid.h"

static const TargetMapping AMDILTargetMapping_0_8[] = {
  UnknownTarget,
  { "R7XX", "RV770", "rv770", amd::GPU_Library_7xx, WEKIVA_A11,    F_RV770, false, false, FAMILY_RV7XX },
  { "R7XX", "RV770", "rv770", amd::GPU_Library_7xx, WEKIVA_A12,    F_RV770, false, true , FAMILY_RV7XX },
  { "R7XX", "RV790", "rv770", amd::GPU_Library_7xx, WEKIVA_A21,    F_RV770, false, true , FAMILY_RV7XX },
  { "R7XX", "RV730", "rv730", amd::GPU_Library_7xx, MARIO_A11,     F_RV730, false, false, FAMILY_RV7XX },
  { "R7XX", "RV730", "rv730", amd::GPU_Library_7xx, MARIO_A12,     F_RV730, false, false, FAMILY_RV7XX },
  { "R7XX", "RV730", "rv730", amd::GPU_Library_7xx, MARIO_A13,     F_RV730, false, true , FAMILY_RV7XX },
  { "R7XX", "RV710", "rv710", amd::GPU_Library_7xx, LUIGI_A11,     F_RV710, false, false, FAMILY_RV7XX },
  { "R7XX", "RV710", "rv710", amd::GPU_Library_7xx, LUIGI_A12,     F_RV710, false, true , FAMILY_RV7XX },
  { "R7XX", "RV710", "rv710", amd::GPU_Library_7xx, LUIGI_APU_A11, F_RV710, false, false, FAMILY_RV7XX },
  { "R7XX", "RV740", "rv770", amd::GPU_Library_7xx, WALDEN_A11,    F_RV770, false, false, FAMILY_RV7XX },
  { "R7XX", "RV740", "rv770", amd::GPU_Library_7xx, WALDEN_A12,    F_RV770, false, true , FAMILY_RV7XX },
  { "Evergreen", "Cypress", "cypress", amd::GPU_Library_Evergreen, CYPRESS_A11, F_EG_EXT, true , false, FAMILY_EVERGREEN },
  { "Evergreen", "Cypress", "cypress", amd::GPU_Library_Evergreen, CYPRESS_A12, F_EG_EXT, true , true , FAMILY_EVERGREEN },
  { "Evergreen", "Juniper", "juniper", amd::GPU_Library_Evergreen, JUNIPER_A11, F_JUNIPER, true , false, FAMILY_EVERGREEN },
  { "Evergreen", "Juniper", "juniper", amd::GPU_Library_Evergreen, JUNIPER_A12, F_JUNIPER, true , true , FAMILY_EVERGREEN },
  { "Evergreen", "Redwood", "redwood", amd::GPU_Library_Evergreen, REDWOOD_A11, F_REDWOOD, true , false, FAMILY_EVERGREEN },
  { "Evergreen", "Redwood", "redwood", amd::GPU_Library_Evergreen, REDWOOD_A12, F_REDWOOD, true , true , FAMILY_EVERGREEN },
  { "Evergreen", "Cedar",   "cedar",   amd::GPU_Library_Evergreen, CEDAR_A11,   F_CEDAR, true , false, FAMILY_EVERGREEN },
  { "Evergreen", "Cedar",   "cedar",   amd::GPU_Library_Evergreen, CEDAR_A12,   F_CEDAR, true , true , FAMILY_EVERGREEN },
  { "NI", "Cayman", "cayman", amd::GPU_Library_Evergreen, NI_CAYMAN_P_A11,       F_NI_EXT,  true , true , FAMILY_NI },
  { "NI", "Barts",  "barts",  amd::GPU_Library_Evergreen, NI_BARTS_PM_A11,       F_NI_BASE, true , true , FAMILY_NI },
  { "NI", "Turks",  "turks",  amd::GPU_Library_Evergreen, NI_TURKS_M_A11,        F_NI_BASE, true , true , FAMILY_NI },
  { "NI", "Lombok", "turks",  amd::GPU_Library_Evergreen, NI_TURKS_LOMBOK_M_A11, F_NI_BASE, true , true , FAMILY_NI },
  { "NI", "Caicos", "caicos", amd::GPU_Library_Evergreen, NI_CAICOS_V_A11,       F_NI_BASE, true , true , FAMILY_NI },
  { "NI", "Kauai",  "kauai",  amd::GPU_Library_Evergreen, KAUAI_A11,             F_NI_BASE, false, false, FAMILY_NI },
  { "Sumo", "BeaverCreek", "redwood", amd::GPU_Library_Evergreen, SUPERSUMO_A0, F_REDWOOD, true , false, FAMILY_SUMO },
  { "Sumo", "BeaverCreek", "redwood", amd::GPU_Library_Evergreen, SUPERSUMO_B0, F_REDWOOD, true , true , FAMILY_SUMO },
  { "Sumo", "WinterPark",  "redwood", amd::GPU_Library_Evergreen,      SUMO_A0, F_REDWOOD, true , false, FAMILY_SUMO },
  { "Sumo", "WinterPark",  "redwood", amd::GPU_Library_Evergreen,      SUMO_B0, F_REDWOOD, true , true , FAMILY_SUMO },
  { "Sumo", "Loveland",    "cedar",   amd::GPU_Library_Evergreen,  WRESTLER_A0, F_REDWOOD, true , false, FAMILY_SUMO },
  { "Sumo", "Loveland",    "cedar",   amd::GPU_Library_Evergreen,  WRESTLER_A1, F_REDWOOD, true , false, FAMILY_SUMO },
  { "Sumo", "Loveland",    "cedar",   amd::GPU_Library_Evergreen,  WRESTLER_B0, F_REDWOOD, true , false, FAMILY_SUMO },
  { "Sumo", "Loveland",    "cedar",   amd::GPU_Library_Evergreen,  WRESTLER_C0, F_REDWOOD, true , true , FAMILY_SUMO },
  { "Sumo", "Bheem",       "cedar",   amd::GPU_Library_Evergreen,     BHEEM_A0, F_REDWOOD, true , true , FAMILY_SUMO }, 
  { "SI", "Tahiti",    "tahiti",   amd::GPU_Library_SI, SI_TAHITI_P_A11,    F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",   amd::GPU_Library_SI, SI_TAHITI_P_A0,     F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",   amd::GPU_Library_SI, SI_TAHITI_P_A21,    F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",   amd::GPU_Library_SI, SI_TAHITI_P_B0,     F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",   amd::GPU_Library_SI, SI_TAHITI_P_A22,    F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Tahiti",    "tahiti",   amd::GPU_Library_SI, SI_TAHITI_P_B1,     F_SI_BASE, true , true, FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU_Library_SI, SI_PITCAIRN_PM_A11, F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU_Library_SI, SI_PITCAIRN_PM_A0,  F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU_Library_SI, SI_PITCAIRN_PM_A12, F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Pitcairn",  "pitcairn",  amd::GPU_Library_SI, SI_PITCAIRN_PM_A1,  F_SI_BASE, true , true, FAMILY_SI },
  { "SI", "Capeverde", "capeverde",   amd::GPU_Library_SI, SI_CAPEVERDE_M_A11, F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Capeverde", "capeverde",   amd::GPU_Library_SI, SI_CAPEVERDE_M_A0,  F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Capeverde", "capeverde",   amd::GPU_Library_SI, SI_CAPEVERDE_M_A12, F_SI_BASE, true ,false, FAMILY_SI },
  { "SI", "Capeverde", "capeverde",   amd::GPU_Library_SI, SI_CAPEVERDE_M_A1,  F_SI_BASE, true , true, FAMILY_SI },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_M_A0,       F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_M_A1,       F_NI_EXT, true, true,  FAMILY_TN },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_LITE_MV_A0, F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_LITE_MV_A1, F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_V_A0,       F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_V_A1,       F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Scrapper",   "trinity", amd::GPU_Library_Evergreen, TN_SCRAPPER_V_A0,         F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Scrapper",   "trinity", amd::GPU_Library_Evergreen, TN_SCRAPPER_V_A1,         F_NI_EXT, true, true,  FAMILY_TN },
  { "TN", "Scrapper",   "trinity", amd::GPU_Library_Evergreen, TN_DVST_DUO_V_A0,         F_NI_EXT, true, false, FAMILY_TN },
  { "KV", "Spectre",    "spectre", amd::GPU_Library_CI,        KV_SPECTRE_A0,           F_SI_BASE, true, true, FAMILY_KV },
  { "KV", "Spooky",     "spooky",  amd::GPU_Library_CI,        KV_SPOOKY_A0,            F_SI_BASE, true, true, FAMILY_KV },
  { "KV", "Kalindi",    "kalindi", amd::GPU_Library_CI,        KB_KALINDI_A0,           F_SI_BASE, true, true, FAMILY_KV },
  { "CI", "Hawaii",     "hawaii",  amd::GPU_Library_CI,        CI_HAWAII_P_A0,          F_SI_BASE, true, true, FAMILY_CI },
  { "KV", "Mullins",    "mullins", amd::GPU_Library_CI,        ML_GODAVARI_A0,          F_SI_BASE, true, true, FAMILY_KV },
  { "SI", "Oland",      "oland",   amd::GPU_Library_SI,        SI_OLAND_M_A0,           F_SI_BASE, true, true, FAMILY_SI },
  { "CI", "Bonaire",    "bonaire", amd::GPU_Library_CI,        CI_BONAIRE_M_A0,         F_SI_BASE, true, false, FAMILY_CI },
  { "SI", "Hainan",     "hainan",  amd::GPU_Library_SI,        SI_HAINAN_V_A0,          F_SI_BASE, true, true, FAMILY_SI },

  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  UnknownTarget,
  { "CZ", "Dummy",      "dummy",   amd::GPU_Library_CI,        STONEY_A0,               F_SI_BASE, true, true, FAMILY_CZ },
  { "CZ", "Carrizo",    "carrizo", amd::GPU_Library_CI,        CARRIZO_A0,              F_SI_BASE, true,  true, FAMILY_CZ },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_W_A0,       F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Devastator", "trinity", amd::GPU_Library_Evergreen, TN_DEVASTATOR_W_A1,       F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Scrapper",   "trinity", amd::GPU_Library_Evergreen, TN_SCRAPPER_LV_A0,        F_NI_EXT, true, false, FAMILY_TN },
  { "TN", "Scrapper",   "trinity", amd::GPU_Library_Evergreen, TN_SCRAPPER_LV_A1,        F_NI_EXT, true, false, FAMILY_TN },

  { "VI", "Iceland",    "iceland", amd::GPU_Library_CI,        VI_ICELAND_M_A0,         F_SI_BASE, true,  true, FAMILY_VI },
  { "VI", "Tonga",      "tonga",   amd::GPU_Library_CI,        VI_TONGA_P_A0,           F_SI_BASE, true,  true, FAMILY_VI },
  { "CI", "Bonaire",    "bonaire", amd::GPU_Library_CI,        CI_BONAIRE_M_A1,         F_SI_BASE, true, true, FAMILY_CI },
  { "VI", "Fiji",       "fiji",    amd::GPU_Library_CI,        VI_FIJI_P_A0,            F_SI_BASE, true,  true, FAMILY_VI },
  InvalidTarget
};

#endif // _CL_UTILS_TARGET_MAPPINGS_AMDIL_0_8_H_
