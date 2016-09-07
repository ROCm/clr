//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "top.hpp"
#include "pal.h"
#include "palGpuMemory.h"
#include "palImage.h"
#include "palFormatInfo.h"

//
/// Memory Object Type
//
enum PalGpuMemoryType {
    PAL_DEPTH_BUFFER = 0,       ///< Depth Buffer
    PAL_BUFFER,                 ///< Pure buffer
    PAL_TEXTURE_3D,             ///< 3D texture
    PAL_TEXTURE_2D,             ///< 2D texture
    PAL_TEXTURE_1D,             ///< 1D texture
    PAL_TEXTURE_1D_ARRAY,       ///< 1D Array texture
    PAL_TEXTURE_2D_ARRAY,       ///< 2D Array texture
    PAL_TEXTURE_BUFFER,         ///< "buffer" texture inside VBO
};

struct HwDbgKernelInfo
{
    uint64_t    scratchBufAddr;             ///< Handle of GPU local memory for kernel private scratch space
    size_t      scratchBufferSizeInBytes;   ///< size of memory pointed to by pScratchBuffer,
    uint64_t    heapBufAddr;                ///< Address of the global heap base
    const void* pAqlDispatchPacket;         ///< Pointer to the dipatch packet
    const void* pAqlQueuePtr;               ///< pointer to the AQL Queue
    void*       trapHandler;                ///< address of the trap handler (TBA)
    void*       trapHandlerBuffer;          ///< address of the trap handler buffer (TMA)
    uint32_t    excpEn;                     ///< excecption mask
    bool        trapPresent;                ///< trap present flag
    bool        sqDebugMode;                ///< debug mode flag (GPU single step mode)
    uint32_t    mgmtSe0Mask;                ///< mask for SE0 (reserving CU for display)
    uint32_t    mgmtSe1Mask;                ///< mask for SE1 (reserving CU for display)
    uint32_t    cacheDisableMask;           ///< cache disable mask
};

//! Engine types
enum EngineType
{
    MainEngine  = 0,
    SdmaEngine,
    AllEngines
};

struct GpuEvent
{
    static const unsigned int InvalidID  = ((1<<30) - 1);

    EngineType      engineId_;  ///< type of the id
    unsigned int    id;         ///< actual event id

    //! GPU event default constructor
    GpuEvent(): engineId_(MainEngine), id(InvalidID) {}

    //! Returns true if the current event is valid
    bool isValid() const { return (id != InvalidID) ? true : false; }

    //! Set invalid event id
    void invalidate() { id = InvalidID; }
};

/*! \addtogroup PAL
 *  @{
 */

//! PAL Device Implementation

namespace pal {

//! Maximum number of the supported global atomic counters
const static uint MaxAtomicCounters = 8;
//! Maximum number of the supported samplers
const static uint MaxSamplers   = 16;
//! Maximum number of supported read images
const static uint MaxReadImage  = 128;
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

//! Alignment restriciton for the pinned memory
const static size_t PinnedMemoryAlignment = 4 * Ki;

//! HSA path specific defines for images
const static uint HsaImageObjectSize = 48;
const static uint HsaImageObjectAlignment = 16;
const static uint HsaSamplerObjectSize = 32;
const static uint HsaSamplerObjectAlignment = 16;

//! HSA path specific defines for images
const static uint DeviceQueueMaskSize = 32;

struct AMDDeviceInfo {
    const char* targetName_;            //!< Target name
    const char* machineTarget_;         //!< Machine target
    uint        simdPerCU_;             //!< Number of SIMDs per CU
    uint        simdWidth_;             //!< Number of workitems processed per SIMD
    uint        simdInstructionWidth_;  //!< Number of instructions processed per SIMD
    uint        memChannelBankWidth_;   //!< Memory channel bank width
    uint        localMemSizePerCU_;     //!< Local memory size per CU
    uint        localMemBanks_;         //!< Number of banks of local memory
    uint        gfxipVersion_;          //!< The core engine GFXIP version
};

static const AMDDeviceInfo DeviceInfo[] = {
/* Unknown */   { "",           "unknown",  4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Tahiti */    { "",           "tahiti",   4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Pitcairn */  { "",           "pitcairn", 4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Capeverde */ { "",           "bonaire",  4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Oland */     { "",           "oland",    4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Hainan */    { "",           "hainan",   4, 16, 1, 256, 64 * Ki, 32, 702 },

/* Bonaire */   { "Bonaire",    "bonaire",  4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Hawaii */    { "Hawaii",     "hawaii",   4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Kalindi */   { "Kalindi",    "kalindi",  4, 16, 1, 256, 64 * Ki, 32, 702 },
/* Spectre */   { "Spectre",    "spectre",  4, 16, 1, 256, 64 * Ki, 32, 701 },

/* Carrizo */   { "Carrizo" ,   "carrizo",  4, 16, 1, 256, 64 * Ki, 32, 800 },
/* Stoney */    { "Stoney",     "stoney",   4, 16, 1, 256, 64 * Ki, 32, 800 },

/* Iceland */   { "Iceland",    "iceland",  4, 16, 1, 256, 64 * Ki, 32, 800 },
/* Tonga */     { "Tonga",      "tonga",    4, 16, 1, 256, 64 * Ki, 32, 800 },
/* Fiji */      { "Fiji",       "fiji",     4, 16, 1, 256, 64 * Ki, 32, 800 },
/* Ellesmere */ { "Ellesmere",  "ellesmere",4, 16, 1, 256, 64 * Ki, 32, 800 },
/* Buffin */    { "Baffin",     "baffin",   4, 16, 1, 256, 64 * Ki, 32, 800 },
};

// The GfxIpDeviceInfo table must match with GfxIpLevel enum
// (located in //depot/stg/pal/inc/core/palDevice.h).
static const AMDDeviceInfo GfxIpDeviceInfo[] = {
/* Unknown  */    { "",         "unknown",     4, 16, 1, 256, 64 * Ki, 32, 000 },
/* GFX6_0_0 */    { "",         "gfx6_0_0",    4, 16, 1, 256, 64 * Ki, 32, 600 },
/* GFX7_0_0 */    { "",         "gfx7_0_0",    4, 16, 1, 256, 64 * Ki, 32, 700 },
/* GFX8_0_0 */    { "",         "gfx8_0_0",    4, 16, 1, 256, 64 * Ki, 32, 800 },
/* GFX8_0_1 */    { "",         "gfx8_0_1",    4, 16, 1, 256, 64 * Ki, 32, 801 },
/* GFX9_0_0 */    { "Rabbit",  "rabbit",   4, 16, 1, 256, 64 * Ki, 32, 900 },
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
    gfx901 = 901
};

static const char* Gfx700 = "AMD:AMDGPU:7:0:0";
static const char* Gfx701 = "AMD:AMDGPU:7:0:1";
static const char* Gfx800 = "AMD:AMDGPU:8:0:0";
static const char* Gfx801 = "AMD:AMDGPU:8:0:1";
static const char* Gfx804 = "AMD:AMDGPU:8:0:4";
static const char* Gfx810 = "AMD:AMDGPU:8:1:0";
static const char* Gfx900 = "AMD:AMDGPU:9:0:0";
static const char* Gfx901 = "AMD:AMDGPU:9:0:1";

// Supported OpenCL versions
enum OclVersion {
    OpenCL10,
    OpenCL11,
    OpenCL12,
    OpenCL20
};

struct MemoryFormat {
    cl_image_format     clFormat_;      //!< CL image format
    Pal::ChNumFormat    palFormat_;     //!< PAL image format
    Pal::ChannelMapping palChannel_;    //!< PAL channel mapping
};

static const MemoryFormat
MemoryFormatMap[] = {
    // R
    { { CL_R,                       CL_UNORM_INT8 },
      Pal::ChNumFormat::X8_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_UNORM_INT16 },
        Pal::ChNumFormat::X16_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },

    { { CL_R,                       CL_SNORM_INT8 },
      Pal::ChNumFormat::X8_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_SNORM_INT16 },
      Pal::ChNumFormat::X16_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },

    { { CL_R,                       CL_SIGNED_INT8 },
      Pal::ChNumFormat::X8_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_SIGNED_INT16 },
      Pal::ChNumFormat::X16_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_SIGNED_INT32 },
      Pal::ChNumFormat::X32_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_UNSIGNED_INT8 },
      Pal::ChNumFormat::X8_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_UNSIGNED_INT16 },
      Pal::ChNumFormat::X16_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_UNSIGNED_INT32 },
      Pal::ChNumFormat::X32_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },

    { { CL_R,                       CL_HALF_FLOAT },
      Pal::ChNumFormat::X16_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_R,                       CL_FLOAT },
      Pal::ChNumFormat::X32_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },

    // A
    { { CL_A,                       CL_UNORM_INT8 },
      Pal::ChNumFormat::X8_Unorm,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_UNORM_INT16 },
      Pal::ChNumFormat::X16_Unorm,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },

    { { CL_A,                       CL_SNORM_INT8 },
      Pal::ChNumFormat::X8_Snorm,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_SNORM_INT16 },
      Pal::ChNumFormat::X16_Snorm,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },

    { { CL_A,                       CL_SIGNED_INT8 },
      Pal::ChNumFormat::X8_Sint,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_SIGNED_INT16 },
      Pal::ChNumFormat::X16_Sint,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_SIGNED_INT32},
      Pal::ChNumFormat::X32_Sint,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_UNSIGNED_INT8 },
      Pal::ChNumFormat::X8_Uint,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_UNSIGNED_INT16 },
      Pal::ChNumFormat::X16_Uint,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_UNSIGNED_INT32},
      Pal::ChNumFormat::X32_Uint,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },

    { { CL_A,                       CL_HALF_FLOAT },
      Pal::ChNumFormat::X16_Float,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },
    { { CL_A,                       CL_FLOAT },
      Pal::ChNumFormat::X32_Float,
        { Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::X } },

    // RG
    { { CL_RG,                      CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_UNORM_INT16 },
      Pal::ChNumFormat::X16Y16_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },

    { { CL_RG,                      CL_SNORM_INT8 },
      Pal::ChNumFormat::X8Y8_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_SNORM_INT16 },
      Pal::ChNumFormat::X16Y16_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },

    { { CL_RG,                      CL_SIGNED_INT8 },
      Pal::ChNumFormat::X8Y8_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_SIGNED_INT16 },
      Pal::ChNumFormat::X16Y16_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_SIGNED_INT32},
      Pal::ChNumFormat::X32Y32_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_UNSIGNED_INT8 },
      Pal::ChNumFormat::X8Y8_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_UNSIGNED_INT16 },
      Pal::ChNumFormat::X16Y16_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_UNSIGNED_INT32},
      Pal::ChNumFormat::X32Y32_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },

    { { CL_RG,                      CL_HALF_FLOAT },
      Pal::ChNumFormat::X16Y16_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
    { { CL_RG,                      CL_FLOAT },
      Pal::ChNumFormat::X32Y32_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::One } },
/*
    // RA
    { { CL_RA,                      CL_UNORM_INT8 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_RG8 } },
    { { CL_RA,                      CL_UNORM_INT16 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_RG16 } },

    { { CL_RA,                      CL_SNORM_INT8 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_sRG8 } },
    { { CL_RA,                      CL_SNORM_INT16 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_sUV16 } },

    { { CL_RA,                      CL_SIGNED_INT8 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_sRG8I } },
    { { CL_RA,                      CL_SIGNED_INT16 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_sRG16I } },
    { { CL_RA,                      CL_SIGNED_INT32},
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_sRG32I } },
    { { CL_RA,                      CL_UNSIGNED_INT8 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_RG8I } },
    { { CL_RA,                      CL_UNSIGNED_INT16 },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_RG16I } },
    { { CL_RA,                      CL_UNSIGNED_INT32},
      { GSL_CHANNEL_ORDER_RA ,      CM_SURF_FMT_RG32I } },

    { { CL_RA,                      CL_HALF_FLOAT },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_RG16F } },
    { { CL_RA,                      CL_FLOAT },
      { GSL_CHANNEL_ORDER_RA,       CM_SURF_FMT_RG32F } },
*/
    // RGB
    { { CL_RGB,                     CL_UNORM_INT_101010 },
      Pal::ChNumFormat::X10Y10Z10W2_Unorm,
        { Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::One } },
    { { CL_RGB,                     CL_UNSIGNED_INT8 },     // This is used only by blit kernel
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::One } },

    // RGBA
    { { CL_RGBA,                    CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_UNORM_INT16 },
      Pal::ChNumFormat::X16Y16Z16W16_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },

    { { CL_RGBA,                    CL_SNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_SNORM_INT16 },
      Pal::ChNumFormat::X16Y16Z16W16_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },

    { { CL_RGBA,                    CL_SIGNED_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_SIGNED_INT16 },
      Pal::ChNumFormat::X16Y16Z16W16_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_SIGNED_INT32 },
      Pal::ChNumFormat::X32Y32Z32W32_Sint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_UNSIGNED_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_UNSIGNED_INT16 },
      Pal::ChNumFormat::X16Y16Z16W16_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_UNSIGNED_INT32},
      Pal::ChNumFormat::X32Y32Z32W32_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },

    { { CL_RGBA,                    CL_HALF_FLOAT },
      Pal::ChNumFormat::X16Y16Z16W16_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_RGBA,                    CL_FLOAT },
      Pal::ChNumFormat::X32Y32Z32W32_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },

    // ARGB
    { { CL_ARGB,                    CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Unorm,
        { Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
          Pal::ChannelSwizzle::W, Pal::ChannelSwizzle::X } },
    { { CL_ARGB,                    CL_SNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Snorm,
        { Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
          Pal::ChannelSwizzle::W, Pal::ChannelSwizzle::X } },
    { { CL_ARGB,                    CL_SIGNED_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Sint,
        { Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
          Pal::ChannelSwizzle::W, Pal::ChannelSwizzle::X } },
    { { CL_ARGB,                    CL_UNSIGNED_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
          Pal::ChannelSwizzle::W, Pal::ChannelSwizzle::X } },

    // BGRA
    { { CL_BGRA,                    CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Unorm,
        { Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::W } },
    { { CL_BGRA,                    CL_SNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Snorm,
        { Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::W } },
    { { CL_BGRA,                    CL_SIGNED_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Sint,
        { Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::W } },
    { { CL_BGRA,                    CL_UNSIGNED_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::W } },

    // LUMINANCE
    { { CL_LUMINANCE,               CL_SNORM_INT8 },
      Pal::ChNumFormat::X8_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::One } },
    { { CL_LUMINANCE,               CL_SNORM_INT16 },
      Pal::ChNumFormat::X16_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::One } },
    { { CL_LUMINANCE,               CL_UNORM_INT8 },
      Pal::ChNumFormat::X8_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::One } },
    { { CL_LUMINANCE,               CL_UNORM_INT16 },
      Pal::ChNumFormat::X16_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::One } },
    { { CL_LUMINANCE,               CL_HALF_FLOAT },
      Pal::ChNumFormat::X16_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::One } },
    { { CL_LUMINANCE,               CL_FLOAT },
      Pal::ChNumFormat::X32_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::One } },

    // INTENSITY
    { { CL_INTENSITY,               CL_SNORM_INT8 },
      Pal::ChNumFormat::X8_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_INTENSITY,               CL_SNORM_INT16 },
      Pal::ChNumFormat::X16_Snorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_INTENSITY,               CL_UNORM_INT8 },
      Pal::ChNumFormat::X8_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_INTENSITY,               CL_UNORM_INT16 },
      Pal::ChNumFormat::X16_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_INTENSITY,               CL_HALF_FLOAT },
      Pal::ChNumFormat::X16_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_INTENSITY,               CL_FLOAT },
      Pal::ChNumFormat::X32_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },

    // sRBGA
    { { CL_sRGBA,                   CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Srgb,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },
    { { CL_sRGBA,                   CL_UNSIGNED_INT8 },     // This is used only by blit kernel
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W } },

    // sRBG
    { { CL_sRGB,                    CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Srgb,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::One } },
    { { CL_sRGB,                    CL_UNSIGNED_INT8 },      // This is used only by blit kernel
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::One } },

    // sRBGx
    { { CL_sRGBx,                   CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Srgb,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::One } },
    { { CL_sRGBx,                   CL_UNSIGNED_INT8 },     // This is used only by blit kernel
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::One } },

    // sBGRA
    { { CL_sBGRA,                   CL_UNORM_INT8 },
      Pal::ChNumFormat::X8Y8Z8W8_Srgb,
        { Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::W } },
    { { CL_sBGRA,                   CL_UNSIGNED_INT8 },     // This is used only by blit kernel
      Pal::ChNumFormat::X8Y8Z8W8_Uint,
        { Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::W } },

    // DEPTH
    { { CL_DEPTH,                   CL_FLOAT },
      Pal::ChNumFormat::X32_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_DEPTH,                   CL_UNSIGNED_INT32 },    // This is used only by blit kernel
      Pal::ChNumFormat::X32_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },

    { { CL_DEPTH,                   CL_UNORM_INT16 },
      Pal::ChNumFormat::X16_Unorm,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_DEPTH,                   CL_UNSIGNED_INT16 },    // This is used only by blit kernel
      Pal::ChNumFormat::X16_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },

    { { CL_DEPTH_STENCIL,           CL_UNORM_INT24 },
      Pal::ChNumFormat::X32_Uint,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } },
    { { CL_DEPTH_STENCIL,           CL_FLOAT },
      Pal::ChNumFormat::X32_Float,
        { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
          Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X } }
};

} // namespace pal

