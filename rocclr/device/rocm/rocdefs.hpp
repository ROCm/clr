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
static constexpr size_t PinnedMemoryAlignment = 4 * Ki;

//! Specific defines for images for Dynamic Parallelism
static constexpr uint DeviceQueueMaskSize = 32;

typedef uint HsaDeviceId;

struct AMDDeviceInfo {
  const char* machineTarget_;  //!< Machine target
  const char* machineTargetLC_;//!< Machine target for LC
  uint simdPerCU_;             //!< Number of SIMDs per CU
  uint simdWidth_;             //!< Number of workitems processed per SIMD
  uint simdInstructionWidth_;  //!< Number of instructions processed per SIMD
  uint memChannelBankWidth_;   //!< Memory channel bank width
  uint localMemSizePerCU_;     //!< Local memory size per CU
  uint localMemBanks_;         //!< Number of banks of local memory
  uint gfxipMajor_;            //!< The core engine GFXIP Major version
  uint gfxipMinor_;            //!< The core engine GFXIP Minor version
  uint gfxipStepping_;         //!< The core engine GFXIP Stepping version
  uint pciDeviceId_;           //!< PCIe device id
};

constexpr HsaDeviceId HSA_INVALID_DEVICE_ID = -1;

static constexpr AMDDeviceInfo DeviceInfo[] = {
  /* KAVERI_SPECTRE */ {"Spectre", "",         4, 16, 1, 256, 64 * Ki, 32, 7, 0, 1, 0},
  /* KAVERI_SPOOKY */  {"Spooky", "",          4, 16, 1, 256, 64 * Ki, 32, 7, 0, 1, 0},
  /* HAWAII */         {"Hawaii", "gfx701",    4, 16, 1, 256, 64 * Ki, 32, 7, 0, 1, 0},
  /* CARRIZO */        {"Carrizo", "gfx801",   4, 16, 1, 256, 64 * Ki, 32, 8, 0, 1, 0},
  /* TONGA */          {"Tonga", "gfx802",     4, 16, 1, 256, 64 * Ki, 32, 8, 0, 2, 0},
  /* ICELAND */        {"Iceland", "gfx802",   4, 16, 1, 256, 64 * Ki, 32, 8, 0, 2, 0},
  /* FIJI */           {"Fiji", "gfx803",      4, 16, 1, 256, 64 * Ki, 32, 8, 0, 3, 0},
  /* ELLESMERE */      {"Ellesmere", "gfx803", 4, 16, 1, 256, 64 * Ki, 32, 8, 0, 3, 0},
  /* BAFFIN */         {"Baffin", "gfx803",    4, 16, 1, 256, 64 * Ki, 32, 8, 0, 3, 0},
  /* VEGA10 */         {"gfx900", "gfx900",    4, 16, 1, 256, 64 * Ki, 32, 9, 0, 0, 0},
  /* VEGA10_HBCC */    {"gfx901", "gfx901",    4, 16, 1, 256, 64 * Ki, 32, 9, 0, 1, 0},
  /* RAVEN */          {"gfx902", "gfx902",    4, 16, 1, 256, 64 * Ki, 32, 9, 0, 2, 0},
  /* VEGA12 */         {"gfx904", "gfx904",    4, 16, 1, 256, 64 * Ki, 32, 9, 0, 4, 0},
  /* VEGA20 */         {"gfx906", "gfx906",    4, 16, 1, 256, 64 * Ki, 32, 9, 0, 6, 0},
  /* ARCTURUS */       {"gfx908", "gfx908",    4, 16, 1, 256, 64 * Ki, 32, 9, 0, 8, 0},
  /* NAVI10 */         {"gfx1010", "gfx1010",  2, 32, 1, 256, 64 * Ki, 32, 10, 1, 0, 0},
  /* NAVI12 */         {"gfx1011", "gfx1011",  2, 32, 1, 256, 64 * Ki, 32, 10, 1, 1, 0},
  /* NAVI14 */         {"gfx1012", "gfx1012",  2, 32, 1, 256, 64 * Ki, 32, 10, 1, 2, 0},
  /* SIENNA_CICHILD */ {"gfx1030", "gfx1030",  2, 32, 1, 256, 64 * Ki, 32, 10, 3, 0, 0},
  /* NAVY_FLOUNDER */  {"gfx1031", "gfx1031",  2, 32, 1, 256, 64 * Ki, 32, 10, 3, 1, 0}
};

}

constexpr uint kMaxAsyncQueues = 8;   // set to match the number of pipes, which is 8
#endif
