/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#ifndef GPUDEFS_HPP_
#define GPUDEFS_HPP_

#include "top.hpp"

#include "gsl_types.h"
#include "gsl_config.h"
#include "gsl_ctx.h"
#include "backend.h"
#include "GSLDevice.h"
#include "GSLContext.h"

extern bool getFuncInfoFromImage(CALimage image, CALfuncInfo* pFuncInfo);

/*! \addtogroup GPU
 *  @{
 */

//! GPU Device Implementation

namespace gpu {

//! Maximum number of the supported global atomic counters
const static uint MaxAtomicCounters = 8;
//! Maximum number of the supported samplers
const static uint MaxSamplers = 16;
//! Maximum number of supported read images
const static uint MaxReadImage = 128;
//! Maximum number of supported write images
const static uint MaxWriteImage = 8;
//! Maximum number of supported read/write images for OCL20
const static uint MaxReadWriteImage = 64;
//! Maximum number of supported constant arguments
const static uint MaxConstArguments = 8;
//! Maximum number of supported kernel UAV arguments
const static uint MaxUavArguments = 1024;
//! Maximum number of pixels for a 1D image created from a buffer
const static size_t MaxImageBufferSize = 1 << 27;
//! Maximum number of pixels for a 1D image created from a buffer
const static size_t MaxImageArraySize = 2048;

//! Maximum number of supported constant buffers
const static uint MaxConstBuffers = MaxConstArguments + 8;

//! Maximum number of constant buffers for arguments
const static uint MaxConstBuffersArguments = 2;

//! Define offline CAL implementation
const static uint CalOfflineImpl = 0xffffffff;

//! Alignment restriciton for the pinned memory
const static size_t PinnedMemoryAlignment = 4 * Ki;

//! HSA path specific defines for images
const static uint HsaImageObjectSize = 48;
const static uint HsaImageObjectAlignment = 16;
const static uint HsaSamplerObjectSize = 32;
const static uint HsaSamplerObjectAlignment = 16;

//! HSA path specific defines for images
const static uint DeviceQueueMaskSize = 32;

//! Defines all supported ASIC families
enum AsicFamilies { Family7xx, Family8xx, FamilyTotal };

struct AMDDeviceInfo {
  uint machine_;               //!< Machine target ID
  const char* targetName_;     //!< Target name
  const char* machineTarget_;  //!< Machine target
  uint simdPerCU_;             //!< Number of SIMDs per CU
  uint simdWidth_;             //!< Number of workitems processed per SIMD
  uint simdInstructionWidth_;  //!< Number of instructions processed per SIMD
  uint memChannelBankWidth_;   //!< Memory channel bank width
  uint localMemSizePerCU_;     //!< Local memory size per CU
  uint localMemBanks_;         //!< Number of banks of local memory
  uint gfxipVersion_;          //!< The core engine GFXIP version
};


static const AMDDeviceInfo DeviceInfo[] = {
    // Machine                      targetName  machineTarget
    /* CAL_TARGET_600 */ {ED_ATI_CAL_MACHINE_R600_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_610 */ {ED_ATI_CAL_MACHINE_R610_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_630 */ {ED_ATI_CAL_MACHINE_R630_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_670 */ {ED_ATI_CAL_MACHINE_R670_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_7XX */ {ED_ATI_CAL_MACHINE_R770_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_770 */ {ED_ATI_CAL_MACHINE_R770_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_710 */ {ED_ATI_CAL_MACHINE_R710_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_730 */ {ED_ATI_CAL_MACHINE_R730_ISA, "", "", 0, 0, 0, 0, 0, 0, 0},
    /* CAL_TARGET_CYPRESS */ {ED_ATI_CAL_MACHINE_CYPRESS_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 32,
                              400},
    /* CAL_TARGET_JUNIPER */ {ED_ATI_CAL_MACHINE_JUNIPER_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 32,
                              400},
    /* CAL_TARGET_REDWOOD */ {ED_ATI_CAL_MACHINE_REDWOOD_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 16,
                              400},
    /* CAL_TARGET_CEDAR */ {ED_ATI_CAL_MACHINE_CEDAR_ISA, "", "", 1, 8, 5, 256, 32 * Ki, 16, 400},
    /* CAL_TARGET_SUMO */ {ED_ATI_CAL_MACHINE_SUMO_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 16, 400},
    /* CAL_TARGET_SUPERSUMO*/ {ED_ATI_CAL_MACHINE_SUPERSUMO_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 16,
                               400},
    /* CAL_TARGET_WRESTLER*/ {ED_ATI_CAL_MACHINE_WRESTLER_ISA, "", "", 1, 8, 5, 256, 32 * Ki, 16,
                              400},
    /* CAL_TARGET_CAYMAN */ {ED_ATI_CAL_MACHINE_CAYMAN_ISA, "", "", 1, 16, 4, 256, 32 * Ki, 32,
                             500},
    /* CAL_TARGET_KAUAI */ {ED_ATI_CAL_MACHINE_KAUAI_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 32, 400},
    /* CAL_TARGET_BARTS */ {ED_ATI_CAL_MACHINE_BARTS_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 32, 400},
    /* CAL_TARGET_TURKS */ {ED_ATI_CAL_MACHINE_TURKS_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 32, 400},
    /* CAL_TARGET_CAICOS */ {ED_ATI_CAL_MACHINE_CAICOS_ISA, "", "", 1, 16, 5, 256, 32 * Ki, 32,
                             400},
    /* CAL_TARGET_TAHITI */ {ED_ATI_CAL_MACHINE_TAHITI_ISA, "Tahiti", "tahiti", 4, 16, 1, 256,
                             64 * Ki, 32, 600},
    /* CAL_TARGET_PITCAIRN */ {ED_ATI_CAL_MACHINE_PITCAIRN_ISA, "Pitcairn", "pitcairn", 4, 16, 1,
                               256, 64 * Ki, 32, 600},
    /* CAL_TARGET_CAPEVERDE */ {ED_ATI_CAL_MACHINE_CAPEVERDE_ISA, "Capeverde", "capeverde", 4, 16,
                                1, 256, 64 * Ki, 32, 600},
    /* CAL_TARGET_DEVASTATOR */ {ED_ATI_CAL_MACHINE_DEVASTATOR_ISA, "", "", 1, 16, 4, 256, 32 * Ki,
                                 32, 500},
    /* CAL_TARGET_SCRAPPER */ {ED_ATI_CAL_MACHINE_SCRAPPER_ISA, "", "", 1, 16, 4, 256, 32 * Ki, 32,
                               500},
    /* CAL_TARGET_OLAND */ {ED_ATI_CAL_MACHINE_OLAND_ISA, "Oland", "oland", 4, 16, 1, 256, 64 * Ki,
                            32, 600},
    /* CAL_TARGET_BONAIRE */ {ED_ATI_CAL_MACHINE_BONAIRE_ISA, "Bonaire", "bonaire", 4, 16, 1, 256,
                              64 * Ki, 32, 702},
    /* CAL_TARGET_SPECTRE */ {ED_ATI_CAL_MACHINE_SPECTRE_ISA, "Spectre", "spectre", 4, 16, 1, 256,
                              64 * Ki, 32, 701},
    /* CAL_TARGET_SPOOKY */ {ED_ATI_CAL_MACHINE_SPOOKY_ISA, "Spooky", "spooky", 4, 16, 1, 256,
                             64 * Ki, 32, 701},
    /* CAL_TARGET_KALINDI */ {ED_ATI_CAL_MACHINE_KALINDI_ISA, "Kalindi", "kalindi", 4, 16, 1, 256,
                              64 * Ki, 32, 702},
    /* CAL_TARGET_HAINAN */ {ED_ATI_CAL_MACHINE_HAINAN_ISA, "Hainan", "hainan", 4, 16, 1, 256,
                             64 * Ki, 32, 600},
    /* CAL_TARGET_HAWAII */ {ED_ATI_CAL_MACHINE_HAWAII_ISA, "Hawaii", "hawaii", 4, 16, 1, 256,
                             64 * Ki, 32, 702},
    /* CAL_TARGET_ICELAND */ {ED_ATI_CAL_MACHINE_ICELAND_ISA, "Iceland", "iceland", 4, 16, 1, 256,
                              64 * Ki, 32, 800},
    /* CAL_TARGET_TONGA */ {ED_ATI_CAL_MACHINE_TONGA_ISA, "Tonga", "tonga", 4, 16, 1, 256, 64 * Ki,
                            32, 800},
    /* CAL_TARGET_MULLINS */ {ED_ATI_CAL_MACHINE_GODAVARI_ISA, "Mullins", "mullins", 4, 16, 1, 256,
                              64 * Ki, 32, 702},
    /* CAL_TARGET_FIJI */ {ED_ATI_CAL_MACHINE_FIJI_ISA, "Fiji", "fiji", 4, 16, 1, 256, 64 * Ki, 32,
                           800},
    /* CAL_TARGET_CARRIZO */ {ED_ATI_CAL_MACHINE_CARRIZO_ISA, "Carrizo", "carrizo", 4, 16, 1, 256,
                              64 * Ki, 32, 800},
    /* CAL_TARGET_ELLESMERE */ {ED_ATI_CAL_MACHINE_ELLESMERE_ISA, "Ellesmere", "ellesmere", 4, 16,
                                1, 256, 64 * Ki, 32, 800},
    /* CAL_TARGET_BAFFIN */ {ED_ATI_CAL_MACHINE_BAFFIN_ISA, "Baffin", "baffin", 4, 16, 1, 256,
                             64 * Ki, 32, 800},
    /* CAL_TARGET_GREENLAND */ {ED_ATI_CAL_MACHINE_GREENLAND_ISA, IF(IS_BRAHMA, "", "gfx900"),
                                IF(IS_BRAHMA, "", "gfx900"), 4, 16, 1, 256, 64 * Ki, 32, 900},
    /* CAL_TARGET_STONEY */ {ED_ATI_CAL_MACHINE_STONEY_ISA, "Stoney", "stoney", 4, 16, 1, 256,
                             64 * Ki, 32, 800},
    /* CAL_TARGET_LEXA */ {ED_ATI_CAL_MACHINE_LEXA_ISA, "gfx804", "gfx804", 4, 16, 1, 256, 64 * Ki,
                             32, 800},
    /* CAL_TARGET_RAVEN */ {ED_ATI_CAL_MACHINE_RAVEN_ISA, IF(IS_BRAHMA, "", "gfx902"),
                            IF(IS_BRAHMA, "", "gfx902"), 4, 16, 1, 256, 64 * Ki, 32, 900},
    /* CAL_TARGET_RAVEN2 */ {ED_ATI_CAL_MACHINE_RAVEN2_ISA, IF(IS_BRAHMA, "", "gfx902"),
                            IF(IS_BRAHMA, "", "gfx902"), 4, 16, 1, 256, 64 * Ki, 32, 900},
    /* CAL_TARGET_RENOIR */{ ED_ATI_CAL_MACHINE_RENOIR_ISA, IF(IS_BRAHMA, "", "gfx902"),
                                IF(IS_BRAHMA, "", "gfx902"), 4, 16, 1, 256, 64 * Ki, 32, 900 },
    /* CAL_TARGET_POLARIS22 */ {ED_ATI_CAL_MACHINE_POLARIS22_ISA, IF(IS_BRAHMA, "", "gfx804"),
                                IF(IS_BRAHMA, "", "gfx804"), 4, 16, 1, 256, 64 * Ki, 32, 800},
    /* CAL_TARGET_VEGA12 */{ ED_ATI_CAL_MACHINE_VEGA12_ISA, IF(IS_BRAHMA, "", "gfx904"),
                                IF(IS_BRAHMA, "", "gfx904"), 4, 16, 1, 256, 64 * Ki, 32, 900 },
    /* CAL_TARGET_VEGA20 */{ ED_ATI_CAL_MACHINE_VEGA20_ISA, IF(IS_BRAHMA, "", "gfx906"),
                                IF(IS_BRAHMA, "", "gfx906"), 4, 16, 1, 256, 64 * Ki, 32, 900 },
};

enum gfx_handle {
  gfx700 = 700,
  gfx701 = 701,
  gfx702 = 702,
  gfx800 = 800,
  gfx801 = 801,
  gfx804 = 804,
  gfx810 = 810,
  gfx900 = 900,
  gfx902 = 902,
  gfx903 = 903,
  gfx904 = 904,
  gfx906 = 906
};

static const char* Gfx700 = "amdgcn-amd-amdhsa--gfx700";
static const char* Gfx701 = "amdgcn-amd-amdhsa--gfx701";
static const char* Gfx800 = "amdgcn-amd-amdhsa--gfx800";
static const char* Gfx801 = "amdgcn-amd-amdhsa--gfx801+xnack";
static const char* Gfx804 = "amdgcn-amd-amdhsa--gfx804";
static const char* Gfx810 = "amdgcn-amd-amdhsa--gfx810+xnack";
static const char* Gfx900 = "amdgcn-amd-amdhsa--gfx900";
static const char* Gfx902 = "amdgcn-amd-amdhsa--gfx902+xnack";
static const char* Gfx903 = "amdgcn-amd-amdhsa--gfx902";  // NOTE: gfx903 is gfx902 with xnack
static const char* Gfx904 = "amdgcn-amd-amdhsa--gfx904";
static const char* Gfx906 = "amdgcn-amd-amdhsa--gfx906";

// Supported OpenCL versions
enum OclVersion { OpenCL10, OpenCL11, OpenCL12, OpenCL20, OpenCL21 };

struct CalFormat {
  gslChannelOrder channelOrder_;  //!< Texel/pixel GSL channel order
  cmSurfFmt type_;                //!< Texel/pixel CAL format
};

struct MemoryFormat {
  cl_image_format clFormat_;  //!< CL image format
  CalFormat calFormat_;       //!< CAL image format
};

static const MemoryFormat MemoryFormatMap[] = {
    // R
    {{CL_R, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_INTENSITY8}},
    {{CL_R, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_R16}},

    {{CL_R, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_sR8}},
    {{CL_R, CL_SNORM_INT16}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_sU16}},

    {{CL_R, CL_SIGNED_INT8}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_sR8I}},
    {{CL_R, CL_SIGNED_INT16}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_sR16I}},
    {{CL_R, CL_SIGNED_INT32}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_sR32I}},
    {{CL_R, CL_UNSIGNED_INT8}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_R8I}},
    {{CL_R, CL_UNSIGNED_INT16}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_R16I}},
    {{CL_R, CL_UNSIGNED_INT32}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_R32I}},

    {{CL_R, CL_HALF_FLOAT}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_R16F}},
    {{CL_R, CL_FLOAT}, {GSL_CHANNEL_ORDER_R, CM_SURF_FMT_R32F}},

    // A
    {{CL_A, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_INTENSITY8}},
    {{CL_A, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_R16}},

    {{CL_A, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_sR8}},
    {{CL_A, CL_SNORM_INT16}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_sU16}},

    {{CL_A, CL_SIGNED_INT8}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_sR8I}},
    {{CL_A, CL_SIGNED_INT16}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_sR16I}},
    {{CL_A, CL_SIGNED_INT32}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_sR32I}},
    {{CL_A, CL_UNSIGNED_INT8}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_R8I}},
    {{CL_A, CL_UNSIGNED_INT16}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_R16I}},
    {{CL_A, CL_UNSIGNED_INT32}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_R32I}},

    {{CL_A, CL_HALF_FLOAT}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_R16F}},
    {{CL_A, CL_FLOAT}, {GSL_CHANNEL_ORDER_A, CM_SURF_FMT_R32F}},

    // RG
    {{CL_RG, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_RG8}},
    {{CL_RG, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_RG16}},

    {{CL_RG, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_sRG8}},
    {{CL_RG, CL_SNORM_INT16}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_sUV16}},

    {{CL_RG, CL_SIGNED_INT8}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_sRG8I}},
    {{CL_RG, CL_SIGNED_INT16}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_sRG16I}},
    {{CL_RG, CL_SIGNED_INT32}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_sRG32I}},
    {{CL_RG, CL_UNSIGNED_INT8}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_RG8I}},
    {{CL_RG, CL_UNSIGNED_INT16}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_RG16I}},
    {{CL_RG, CL_UNSIGNED_INT32}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_RG32I}},

    {{CL_RG, CL_HALF_FLOAT}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_RG16F}},
    {{CL_RG, CL_FLOAT}, {GSL_CHANNEL_ORDER_RG, CM_SURF_FMT_RG32F}},

    // RA
    {{CL_RA, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_RG8}},
    {{CL_RA, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_RG16}},

    {{CL_RA, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_sRG8}},
    {{CL_RA, CL_SNORM_INT16}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_sUV16}},

    {{CL_RA, CL_SIGNED_INT8}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_sRG8I}},
    {{CL_RA, CL_SIGNED_INT16}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_sRG16I}},
    {{CL_RA, CL_SIGNED_INT32}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_sRG32I}},
    {{CL_RA, CL_UNSIGNED_INT8}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_RG8I}},
    {{CL_RA, CL_UNSIGNED_INT16}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_RG16I}},
    {{CL_RA, CL_UNSIGNED_INT32}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_RG32I}},

    {{CL_RA, CL_HALF_FLOAT}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_RG16F}},
    {{CL_RA, CL_FLOAT}, {GSL_CHANNEL_ORDER_RA, CM_SURF_FMT_RG32F}},

    // RGB
    {{CL_RGB, CL_UNORM_INT_101010}, {GSL_CHANNEL_ORDER_RGB, CM_SURF_FMT_BGR10_X2}},
    // RGBA
    {{CL_RGBA, CL_UNORM_INT_101010}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGB10_X2}},

    // RGBA
    {{CL_RGBA, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGBA8}},
    {{CL_RGBA, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGBA16}},

    {{CL_RGBA, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_sRGBA8}},
    {{CL_RGBA, CL_SNORM_INT16}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_sUVWQ16}},

    {{CL_RGBA, CL_SIGNED_INT8}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_sRGBA8I}},
    {{CL_RGBA, CL_SIGNED_INT16}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_sRGBA16I}},
    {{CL_RGBA, CL_SIGNED_INT32}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_sRGBA32I}},
    {{CL_RGBA, CL_UNSIGNED_INT8}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGBA8UI}},
    {{CL_RGBA, CL_UNSIGNED_INT16}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGBA16UI}},
    {{CL_RGBA, CL_UNSIGNED_INT32}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGBA32UI}},

    {{CL_RGBA, CL_HALF_FLOAT}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGBA16F}},
    {{CL_RGBA, CL_FLOAT}, {GSL_CHANNEL_ORDER_RGBA, CM_SURF_FMT_RGBA32F}},

    // ARGB
    {{CL_ARGB, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_ARGB, CM_SURF_FMT_RGBA8}},
    {{CL_ARGB, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_ARGB, CM_SURF_FMT_sRGBA8}},
    {{CL_ARGB, CL_SIGNED_INT8}, {GSL_CHANNEL_ORDER_ARGB, CM_SURF_FMT_sRGBA8I}},
    {{CL_ARGB, CL_UNSIGNED_INT8}, {GSL_CHANNEL_ORDER_ARGB, CM_SURF_FMT_RGBA8UI}},

    // BGRA
    {{CL_BGRA, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_BGRA, CM_SURF_FMT_RGBA8}},
    {{CL_BGRA, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_BGRA, CM_SURF_FMT_sRGBA8}},
    {{CL_BGRA, CL_SIGNED_INT8}, {GSL_CHANNEL_ORDER_BGRA, CM_SURF_FMT_sRGBA8I}},
    {{CL_BGRA, CL_UNSIGNED_INT8}, {GSL_CHANNEL_ORDER_BGRA, CM_SURF_FMT_RGBA8UI}},

    // LUMINANCE
    {{CL_LUMINANCE, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_LUMINANCE, CM_SURF_FMT_sR8}},
    {{CL_LUMINANCE, CL_SNORM_INT16}, {GSL_CHANNEL_ORDER_LUMINANCE, CM_SURF_FMT_sU16}},
    {{CL_LUMINANCE, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_LUMINANCE, CM_SURF_FMT_INTENSITY8}},
    {{CL_LUMINANCE, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_LUMINANCE, CM_SURF_FMT_R16}},
    {{CL_LUMINANCE, CL_HALF_FLOAT}, {GSL_CHANNEL_ORDER_LUMINANCE, CM_SURF_FMT_R16F}},
    {{CL_LUMINANCE, CL_FLOAT}, {GSL_CHANNEL_ORDER_LUMINANCE, CM_SURF_FMT_R32F}},

    // INTENSITY
    {{CL_INTENSITY, CL_SNORM_INT8}, {GSL_CHANNEL_ORDER_INTENSITY, CM_SURF_FMT_sR8}},
    {{CL_INTENSITY, CL_SNORM_INT16}, {GSL_CHANNEL_ORDER_INTENSITY, CM_SURF_FMT_sU16}},
    {{CL_INTENSITY, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_INTENSITY, CM_SURF_FMT_INTENSITY8}},
    {{CL_INTENSITY, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_INTENSITY, CM_SURF_FMT_R16}},
    {{CL_INTENSITY, CL_HALF_FLOAT}, {GSL_CHANNEL_ORDER_INTENSITY, CM_SURF_FMT_R16F}},
    {{CL_INTENSITY, CL_FLOAT}, {GSL_CHANNEL_ORDER_INTENSITY, CM_SURF_FMT_R32F}},

    // sRBGA
    {{CL_sRGBA, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_SRGBA, CM_SURF_FMT_RGBA8_SRGB}},
    {{CL_sRGBA, CL_UNSIGNED_INT8},  // This is used only by blit kernel
     {GSL_CHANNEL_ORDER_SRGBA, CM_SURF_FMT_RGBA8UI}},

    // sRBG
    {{CL_sRGB, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_SRGB, CM_SURF_FMT_RGBX8UI}},
    {{CL_sRGB, CL_UNSIGNED_INT8},  // This is used only by blit kernel
     {GSL_CHANNEL_ORDER_SRGB, CM_SURF_FMT_RGBA8UI}},

    // sRBGx
    {{CL_sRGBx, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_SRGBX, CM_SURF_FMT_RGBX8UI}},
    {{CL_sRGBx, CL_UNSIGNED_INT8},  // This is used only by blit kernel
     {GSL_CHANNEL_ORDER_SRGBX, CM_SURF_FMT_RGBA8UI}},

    // sBGRA
    {{CL_sBGRA, CL_UNORM_INT8}, {GSL_CHANNEL_ORDER_SBGRA, CM_SURF_FMT_RGBA8}},
    {{CL_sBGRA, CL_UNSIGNED_INT8},  // This is used only by blit kernel
     {GSL_CHANNEL_ORDER_SBGRA, CM_SURF_FMT_RGBA8UI}},

    // DEPTH
    {{CL_DEPTH, CL_FLOAT}, {GSL_CHANNEL_ORDER_REPLICATE_R, CM_SURF_FMT_DEPTH32F}},
    {{CL_DEPTH, CL_UNSIGNED_INT32},  // This is used only by blit kernel
     {GSL_CHANNEL_ORDER_REPLICATE_R, CM_SURF_FMT_R32I}},

    {{CL_DEPTH, CL_UNORM_INT16}, {GSL_CHANNEL_ORDER_REPLICATE_R, CM_SURF_FMT_DEPTH16}},
    {{CL_DEPTH, CL_UNSIGNED_INT16},  // This is used only by blit kernel
     {GSL_CHANNEL_ORDER_REPLICATE_R, CM_SURF_FMT_R16I}},

    {{CL_DEPTH_STENCIL, CL_UNORM_INT24},
     {GSL_CHANNEL_ORDER_REPLICATE_R, CM_SURF_FMT_DEPTH24_STEN8}},
    {{CL_DEPTH_STENCIL, CL_FLOAT}, {GSL_CHANNEL_ORDER_REPLICATE_R, CM_SURF_FMT_DEPTH32F_X24_STEN8}}

};

struct MemFormatStruct {
  cmSurfFmt format_;
  uint size_;
  uint components_;
};

static const MemFormatStruct MemoryFormatSize[] = {
    {CM_SURF_FMT_INTENSITY8, 1,
     1}, /**< 1 component, normalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_RG8, 2,
     2}, /**< 2 component, normalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_RGBA8, 4,
     4}, /**< 4 component, normalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_RGBA8_SRGB, 4,
     4}, /**< 4 component, normalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_R16, 2,
     1}, /**< 1 component, normalized unsigned 16-bit integer value per component */
    {CM_SURF_FMT_RG16, 4,
     2}, /**< 2 component, normalized unsigned 16-bit integer value per component */
    {CM_SURF_FMT_RGBA16, 8,
     4}, /**< 4 component, normalized unsigned 16-bit integer value per component */
    {CM_SURF_FMT_sRGBA8, 4,
     4}, /**< 4 component, normalized signed 8-bit integer value per component */
    {CM_SURF_FMT_sU16, 2,
     1}, /**< 1 component, normalized signed 16-bit integer value per component */
    {CM_SURF_FMT_sUV16, 4,
     2}, /**< 2 component, normalized signed 16-bit integer value per component */
    {CM_SURF_FMT_sUVWQ16, 8,
     4}, /**< 4 component, normalized signed 16-bit integer value per component */
    {CM_SURF_FMT_R32F, 4, 1},     /**< A 1 component, 32-bit float value per component */
    {CM_SURF_FMT_RG32F, 8, 2},    /**< A 2 component, 32-bit float value per component */
    {CM_SURF_FMT_RGBA32F, 16, 4}, /**< A 4 component, 32-bit float value per component */
    {CM_SURF_FMT_sR8, 1,
     1}, /**< 1 component, normalized signed 8-bit integer value per component */
    {CM_SURF_FMT_sRG8, 2,
     2}, /**< 2 component, normalized signed 8-bit integer value per component */

    {CM_SURF_FMT_R8I, 1,
     1}, /**< 1 component, unnormalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_RG8I, 2,
     2}, /**< 2 component, unnormalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_RGBA8UI, 4,
     4}, /**< 4 component, unnormalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_RGBX8UI, 4,
     4}, /**< 4 component, unnormalized unsigned 8-bit integer value per component */
    {CM_SURF_FMT_sR8I, 1,
     1}, /**< 1 component, unnormalized signed 8-bit integer value per component */
    {CM_SURF_FMT_sRG8I, 2,
     2}, /**< 2 component, unnormalized signed 8-bit integer value per component */
    {CM_SURF_FMT_sRGBA8I, 4,
     4}, /**< 4 component, unnormalized signed 8-bit integer value per component */
    {CM_SURF_FMT_R16I, 2,
     1}, /**< 1 component, unnormalized unsigned 16-bit integer value per component */
    {CM_SURF_FMT_RG16I, 4,
     2}, /**< 2 component, unnormalized unsigned 16-bit integer value per component */
    {CM_SURF_FMT_RGBA16UI, 8,
     4}, /**< 4 component, unnormalized unsigned 16-bit integer value per component */
    {CM_SURF_FMT_sR16I, 2,
     1}, /**< 1 component, unnormalized signed 16-bit integer value per component */
    {CM_SURF_FMT_sRG16I, 4,
     2}, /**< 2 component, unnormalized signed 16-bit integer value per component */
    {CM_SURF_FMT_sRGBA16I, 8,
     4}, /**< 4 component, unnormalized signed 16-bit integer value per component */
    {CM_SURF_FMT_R32I, 4,
     1}, /**< 1 component, unnormalized unsigned 32-bit integer value per component */
    {CM_SURF_FMT_RG32I, 8,
     2}, /**< 2 component, unnormalized unsigned 32-bit integer value per component */
    {CM_SURF_FMT_RGBA32UI, 16,
     4}, /**< 4 component, unnormalized unsigned 32-bit integer value per component */
    {CM_SURF_FMT_sR32I, 4,
     1}, /**< 1 component, unnormalized signed 32-bit integer value per component */
    {CM_SURF_FMT_sRG32I, 8,
     2}, /**< 2 component, unnormalized signed 32-bit integer value per component */
    {CM_SURF_FMT_sRGBA32I, 16,
     4}, /**< 4 component, unnormalized signed 32-bit integer value per component */

    {CM_SURF_FMT_R16F, 2, 1},    /**< A 1 component, 16-bit float value per component */
    {CM_SURF_FMT_RG16F, 4, 2},   /**< A 2 component, 16-bit float value per component */
    {CM_SURF_FMT_RGBA16F, 8, 4}, /**< A 4 component, 16-bit float value per component */

    {CM_SURF_FMT_BGR10_X2, 4, 4}, /**< 4 component, unnormalized signed 10-bit integer value per
                                     component packed as (@c XXRRRRRRRRRRGGGGGGGGGGBBBBBBBBBB)*/
    {CM_SURF_FMT_RGB10_X2, 4, 4}, /**< 4 component, unnormalized signed 10-bit integer value per
                                     component packed as (@c XXRRRRRRRRRRGGGGGGGGGGBBBBBBBBBB)*/
    {CM_SURF_FMT_DEPTH32F, 4, 1}, /**< A one component, 32 float value per component */
    {CM_SURF_FMT_DEPTH16, 2, 1},  /**< A one component, 16 unsigned int value per component */
    {CM_SURF_FMT_DEPTH24_STEN8, 4, 1}, /**< A one component, 32 float value per component */
    {CM_SURF_FMT_DEPTH32F_X24_STEN8, 8,
     2} /**< depth + stencil, 64 bits per element packed as (@c
           XXXXXXXXXXXXXXXXXXXXXXXXSSSSSSSSDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD) */
};

__inline const MemFormatStruct& memoryFormatSize(cmSurfFmt fmt) {
  for (uint i = 0; i < sizeof(MemoryFormatSize) / sizeof(MemFormatStruct); ++i) {
    if (MemoryFormatSize[i].format_ == fmt) {
      return MemoryFormatSize[i];
    }
  }
  assert(!"Unknown GSL memory format!");
  return MemoryFormatSize[0];
}

}  // namespace gpu

#endif  // GPUDEFS_HPP_
