#pragma once

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

//! Alignment restriciton for the pinned memory
const static size_t PinnedMemoryAlignment = 4 * Ki;

typedef uint HsaDeviceId;

struct AMDDeviceInfo {
  HsaDeviceId hsaDeviceId_;    //!< Machine id
  const char* targetName_;     //!< Target name for compilation
  const char* machineTarget_;  //!< Machine target
  const char* complibTarget_;  //!< Compiler library target name
  uint simdPerCU_;             //!< Number of SIMDs per CU
  uint simdWidth_;             //!< Number of workitems processed per SIMD
  uint simdInstructionWidth_;  //!< Number of instructions processed per SIMD
  uint memChannelBankWidth_;   //!< Memory channel bank width
  uint localMemSizePerCU_;     //!< Local memory size per CU
  uint localMemBanks_;         //!< Number of banks of local memory
  uint gfxipVersion_;          //!< The core engine GFXIP version
  uint pciDeviceId_;           //!< PCIe device id
};

// The device ID must match with the device's index into DeviceInfo
const HsaDeviceId HSA_SPECTRE_ID = 0;
const HsaDeviceId HSA_SPOOKY_ID = 1;
const HsaDeviceId HSA_TONGA_ID = 2;
const HsaDeviceId HSA_CARRIZO_ID = 3;
const HsaDeviceId HSA_ICELAND_ID = 4;
const HsaDeviceId HSA_FIJI_ID = 5;
const HsaDeviceId HSA_HAWAII_ID = 6;
const HsaDeviceId HSA_ELLESMERE_ID = 7;
const HsaDeviceId HSA_BAFFIN_ID = 8;
const HsaDeviceId HSA_VEGA10_ID = 9;
const HsaDeviceId HSA_INVALID_DEVICE_ID = -1;

static const AMDDeviceInfo DeviceInfo[] = {
    //  targetName  machineTarget
    /* TARGET_KAVERI_SPECTRE */ {HSA_SPECTRE_ID, "", "kaveri", "Spectre", 4, 16, 1, 256, 64 * Ki,
                                 32, 0, 0},
    /* TARGET_KAVERI_SPOOKY */ {HSA_SPOOKY_ID, "", "kaveri", "Spooky", 4, 16, 1, 256, 64 * Ki, 32,
                                0, 0},
    /* TARGET_TONGA */ {HSA_TONGA_ID, "", "tonga", "Tonga", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET_CARRIZO */ {HSA_CARRIZO_ID, "", "carrizo", "Carrizo", 4, 16, 1, 256, 64 * Ki, 32, 0,
                          0},
    /* TARGET_ICELAND */ {HSA_ICELAND_ID, "", "iceland", "Iceland", 4, 16, 1, 256, 64 * Ki, 32, 0,
                          0},
    /* TARGET_FIJI */ {HSA_FIJI_ID, "", "fiji", "Fiji", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET HAWAII */ {HSA_HAWAII_ID, "", "hawaii", "Hawaii", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET ELLESMERE */ {HSA_ELLESMERE_ID, "", "polaris10", "Ellesmere", 4, 16, 1, 256, 64 * Ki,
                            32, 0, 0},
    /* TARGET BAFFIN */ {HSA_BAFFIN_ID, "", "polaris11", "Baffin", 4, 16, 1, 256, 64 * Ki, 32, 0,
                         0},
    /* TARGET VEGA10 */ {HSA_VEGA10_ID, "", "gfx900", "gfx900", 4, 16, 1, 256, 64 * Ki, 32, 0, 0}};
}
#endif
