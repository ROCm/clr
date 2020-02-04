/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

//! Alignment restriciton for the pinned memory
const static size_t PinnedMemoryAlignment = 4 * Ki;

//! Specific defines for images for Dynamic Parallelism
const static uint DeviceQueueMaskSize = 32;

typedef uint HsaDeviceId;

struct AMDDeviceInfo {
  HsaDeviceId hsaDeviceId_;    //!< Machine id
  const char* targetName_;     //!< Target name for compilation
  const char* machineTarget_;  //!< Machine target
  const char* machineTargetLC_;//!< Machine target for LC
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
const HsaDeviceId HSA_VEGA10_HBCC_ID = 10;
const HsaDeviceId HSA_RAVEN_ID = 11;
const HsaDeviceId HSA_VEGA12_ID = 12;
const HsaDeviceId HSA_VEGA20_ID = 13;
const HsaDeviceId HSA_ARIEL_ID = 14;
const HsaDeviceId HSA_NAVI10_ID = 15;
const HsaDeviceId HSA_MI100_ID = 16;
const HsaDeviceId HSA_NAVI12_ID = 17;
const HsaDeviceId HSA_NAVI14_ID = 18;
const HsaDeviceId HSA_INVALID_DEVICE_ID = -1;

static const AMDDeviceInfo DeviceInfo[] = {
    //  targetName  machineTarget
    /* TARGET_KAVERI_SPECTRE */ {HSA_SPECTRE_ID, "", "kaveri", "", "Spectre", 4, 16, 1, 256, 64 * Ki,
                                 32, 0, 0},
    /* TARGET_KAVERI_SPOOKY */ {HSA_SPOOKY_ID, "", "kaveri", "", "Spooky", 4, 16, 1, 256, 64 * Ki, 32,
                                0, 0},
    /* TARGET_TONGA */ {HSA_TONGA_ID, "", "tonga", "gfx802", "Tonga", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET_CARRIZO */ {HSA_CARRIZO_ID, "", "carrizo", "gfx801", "Carrizo", 4, 16, 1, 256, 64 * Ki, 32, 0,
                          0},
    /* TARGET_ICELAND */ {HSA_ICELAND_ID, "", "iceland", "gfx802", "Iceland", 4, 16, 1, 256, 64 * Ki, 32, 0,
                          0},
    /* TARGET_FIJI */ {HSA_FIJI_ID, "", "fiji", "gfx803", "Fiji", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET HAWAII */ {HSA_HAWAII_ID, "", "hawaii", "gfx701", "Hawaii", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET ELLESMERE */ {HSA_ELLESMERE_ID, "", "polaris10", "gfx803", "Ellesmere", 4, 16, 1, 256, 64 * Ki,
                            32, 0, 0},
    /* TARGET BAFFIN */ {HSA_BAFFIN_ID, "", "polaris11", "gfx803", "Baffin", 4, 16, 1, 256, 64 * Ki, 32, 0,
                         0},
    /* TARGET VEGA10 */ {HSA_VEGA10_ID, "", "gfx900", "gfx900", "gfx900", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET VEGA10_HBCC */ {HSA_VEGA10_HBCC_ID, "", "gfx901", "gfx901", "gfx901", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET RAVEN */ {HSA_RAVEN_ID, "", "gfx902", "gfx902", "gfx902", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET VEGA12 */ {HSA_VEGA12_ID, "", "gfx904", "gfx904", "gfx904", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET VEGA20 */ {HSA_VEGA20_ID, "", "gfx906", "gfx906", "gfx906", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET ARIEL */ {HSA_ARIEL_ID, "", "gfx1000", "gfx1000", "gfx1000", 2, 32, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET NAVI10 */ {HSA_NAVI10_ID, "", "gfx1010", "gfx1010", "gfx1010", 2, 32, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET MI100 */ {HSA_MI100_ID, "", "gfx908", "gfx908", "gfx908", 4, 16, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET NAVI12 */ {HSA_NAVI12_ID, "", "gfx1011", "gfx1011", "gfx1011", 2, 32, 1, 256, 64 * Ki, 32, 0, 0},
    /* TARGET NAVI14 */ {HSA_NAVI14_ID, "", "gfx1012", "gfx1012", "gfx1012", 2, 32, 1, 256, 64 * Ki, 32, 0, 0}
};
}

const uint kMaxAsyncQueues = 8;   // set to match the number of pipes, which is 8
#endif
