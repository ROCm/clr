/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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

#include "top.hpp"
#include "pal.h"
#include "palGpuMemory.h"
#include "palImage.h"
#include "palFormatInfo.h"
#include "util/palSysMemory.h"

//
/// Memory Object Type
//
enum PalGpuMemoryType {
  PAL_DEPTH_BUFFER = 0,  ///< Depth Buffer
  PAL_BUFFER,            ///< Pure buffer
  PAL_TEXTURE_3D,        ///< 3D texture
  PAL_TEXTURE_2D,        ///< 2D texture
  PAL_TEXTURE_1D,        ///< 1D texture
  PAL_TEXTURE_1D_ARRAY,  ///< 1D Array texture
  PAL_TEXTURE_2D_ARRAY,  ///< 2D Array texture
  PAL_TEXTURE_BUFFER,    ///< "buffer" texture inside VBO
};

struct HwDbgKernelInfo {
  uint64_t scratchBufAddr;          ///< Handle of GPU local memory for kernel private scratch space
  size_t scratchBufferSizeInBytes;  ///< size of memory pointed to by pScratchBuffer,
  uint64_t heapBufAddr;             ///< Address of the global heap base
  const void* pAqlDispatchPacket;   ///< Pointer to the dipatch packet
  const void* pAqlQueuePtr;         ///< pointer to the AQL Queue
  void* trapHandler;                ///< address of the trap handler (TBA)
  void* trapHandlerBuffer;          ///< address of the trap handler buffer (TMA)
  uint32_t excpEn;                  ///< excecption mask
  bool trapPresent;                 ///< trap present flag
  bool sqDebugMode;                 ///< debug mode flag (GPU single step mode)
  uint32_t mgmtSe0Mask;             ///< mask for SE0 (reserving CU for display)
  uint32_t mgmtSe1Mask;             ///< mask for SE1 (reserving CU for display)
  uint32_t cacheDisableMask;        ///< cache disable mask
};

//! Engine types
enum EngineType { MainEngine = 0, SdmaEngine, AllEngines };

struct GpuEvent {
  static constexpr uint32_t InvalidID = ((1 << 30) - 1);

  struct {
    uint32_t id_ : 30;       ///< Actual event id
    uint32_t modified_ : 1;  ///< Resource associated with the event was modified
    uint32_t engineId_ : 1;  ///< Type of the id
  };
  //! GPU event default constructor
  GpuEvent() : id_(InvalidID), modified_(false), engineId_(MainEngine) {}
  //! GPU event constructor
  GpuEvent(uint evt) : id_(evt), modified_(false), engineId_(MainEngine) {}

  //! Returns true if the current event is valid
  bool isValid() const { return (id_ != InvalidID) ? true : false; }

  //! Set invalid event id
  void invalidate() { id_ = InvalidID; }

  // Overwrite default assign operator to preserve modified_ field
  GpuEvent& operator=(const GpuEvent& evt) {
    id_ = evt.id_;
    engineId_ = evt.engineId_;
    return *this;
  }
};

/*! \addtogroup PAL
 *  @{
 */

//! PAL Device Implementation

namespace pal {

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
  const char* machineTarget_;    //!< Machine target
  const char* machineTargetLC_;  //!< Machine target for LC
  uint simdWidth_;               //!< Number of workitems processed per SIMD
  uint memChannelBankWidth_;     //!< Memory channel bank width
  uint localMemBanks_;           //!< Number of banks of local memory
  uint gfxipVersionLC_;          //!< The core engine GFXIP version for LC
  uint gfxipVersion_;            //!< The core engine GFXIP version
  bool xnackEnabled_;            //!< Enable XNACK feature
};

static constexpr AMDDeviceInfo UnknownDevice = {"", "", 16, 256, 32, 0, 0, false};

static constexpr AMDDeviceInfo DeviceInfo[] = {
    /* Unknown */   UnknownDevice,
    /* Tahiti */    {"", "", 16, 256, 32, 600, 600, false},
    /* Pitcairn */  {"", "", 16, 256, 32, 600, 600, false},
    /* Capeverde */ {"", "", 16, 256, 32, 700, 700, false},
    /* Oland */     {"", "", 16, 256, 32, 600, 600, false},
    /* Hainan */    {"", "", 16, 256, 32, 600, 600, false},

    /* Bonaire */   {"Bonaire", "", 16, 256, 32, 700, 700, false},
    /* Hawaii */    {"Hawaii", "", 16, 256, 32, 701, 701, false},
    /* Hawaii */    {"", "", 16, 256, 32, 701, 701, false},
    /* Hawaii */    {"", "", 16, 256, 32, 701, 701, false},

    /* Kalindi */   {"Kalindi", "", 16, 256, 32, 702, 702, false},
    /* Godavari */  {"Mullins", "", 16, 256, 32, 702, 702, false},
    /* Spectre */   {"Spectre", "", 16, 256, 32, 701, 701, false},
    /* Spooky */    {"Spooky", "", 16, 256, 32, 701, 701, false},

    /* Carrizo */   {"Carrizo", "", 16, 256, 32, 801, 801, false},
    /* Bristol */   {"Bristol Ridge", "", 16, 256, 32, 801, 801, false},
    /* Stoney */    {"Stoney", "", 16, 256, 32, 810, 810, false},

    /* Iceland */   {"Iceland", "gfx802", 16, 256, 32, 802, 800, false},
    /* Tonga */     {"Tonga", "gfx802", 16, 256, 32, 802, 800, false},
    /* Fiji */      {"Fiji", "gfx803", 16, 256, 32, 803, 804, false},
    /* Ellesmere */ {"Ellesmere", "gfx803", 16, 256, 32, 803, 804, false},
    /* Baffin */    {"Baffin", "gfx803", 16, 256, 32, 803, 804, false},
    /* Lexa */      {"gfx804", "gfx803", 16, 256, 32, 803, 804, false},
};

// Ordering as per AsicRevision# in palDevice.h & AMDGPU Target Names

static constexpr AMDDeviceInfo Gfx9PlusSubDeviceInfo[] = {
    /* Vega10          */ {"gfx900", "gfx900", 16, 256, 32, 900, 900, false},
    /* Vega10 XNACK    */ {"gfx901", "gfx900", 16, 256, 32, 900, 901, true},
    /* Vega12          */ {"gfx904", "gfx904", 16, 256, 32, 904, 904, false},
    /* Vega12 XNACK    */ {"gfx905", "gfx904", 16, 256, 32, 904, 905, true},
    /* Vega20          */ {"gfx906", "gfx906", 16, 256, 32, 906, 906, false},
    /* Vega20 XNACK    */ {"gfx907", "gfx906", 16, 256, 32, 906, 907, true},
    /* Raven           */ {"gfx902", "gfx902", 16, 256, 32, 902, 902, false},
    /* Raven XNACK     */ {"gfx903", "gfx902", 16, 256, 32, 902, 903, true},
    /* Raven2          */ {"gfx902", "gfx902", 16, 256, 32, 902, 902, false},
    /* Raven2 XNACK    */ {"gfx903", "gfx902", 16, 256, 32, 902, 903, true},
    /* Renoir          */ {"gfx902", "gfx902", 16, 256, 32, 902, 902, false},
    /* Renoir XNACK    */ {"gfx903", "gfx902", 16, 256, 32, 902, 903, true},
    /* Navi10_A0       */ {"gfx1010", "gfx1010", 32, 256, 32, 1010, 1010, false},
    /* Navi10_A0 XNACK */ {"gfx1010", "gfx1010", 32, 256, 32, 1010, 1010, true},
    /* Navi10          */ {"gfx1010", "gfx1010", 32, 256, 32, 1010, 1010, false},
    /* Navi10 XNACK    */ {"gfx1010", "gfx1010", 32, 256, 32, 1010, 1010, true},
    /* Navi10Lite      */ UnknownDevice,
    /* Navi10LiteXNACK */ UnknownDevice,
    /* Navi12          */ {"gfx1011", "gfx1011", 32, 256, 32, 1011, 1011, false},
    /* Navi12  XNACK   */ {"gfx1011", "gfx1011", 32, 256, 32, 1011, 1011, true},
    /* Navi12Lite      */ UnknownDevice,
    /* Navi12LiteXNACK */ UnknownDevice,
    /* Navi14          */ {"gfx1012", "gfx1012", 32, 256, 32, 1012, 1012, false},
    /* Navi14 XNACK    */ {"gfx1012", "gfx1012", 32, 256, 32, 1012, 1012, true},
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
    /* UnknownDevice       */ UnknownDevice,
    /* UnknownDevice XNACK */ UnknownDevice,
};

// Supported OpenCL versions
enum OclVersion {
  OpenCL10 = 0x10,
  OpenCL11 = 0x11,
  OpenCL12 = 0x12,
  OpenCL20 = 0x20,
  OpenCL21 = 0x21
};

struct MemoryFormat {
  cl_image_format clFormat_;        //!< CL image format
  Pal::ChNumFormat palFormat_;      //!< PAL image format
  Pal::ChannelMapping palChannel_;  //!< PAL channel mapping
};

static const MemoryFormat MemoryFormatMap[] = {
    // R
    {{CL_R, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_R, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_R, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_R, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_R, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    // A
    {{CL_A, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    {{CL_A, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    {{CL_A, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8_Sint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16_Sint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32_Sint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8_Uint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16_Uint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32_Uint,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    {{CL_A, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},
    {{CL_A, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::X}},

    // RG
    {{CL_RG, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNORM_INT16},
     Pal::ChNumFormat::X16Y16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_RG, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_SNORM_INT16},
     Pal::ChNumFormat::X16Y16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_RG, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16Y16_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32Y32_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16Y16_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32Y32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},

    {{CL_RG, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16Y16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
    {{CL_RG, CL_FLOAT},
     Pal::ChNumFormat::X32Y32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Zero,
      Pal::ChannelSwizzle::One}},
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
    {{CL_RGB, CL_UNORM_INT_101010},
     Pal::ChNumFormat::X10Y10Z10W2_Unorm,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    // RGBA
    // Note: special case for GL interops, OCL spec doesn't support 101010 format
    // for GL interop and GL uses real RGB channel order
    {{CL_RGBA, CL_UNORM_INT_101010},
     Pal::ChNumFormat::X10Y10Z10W2_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::One}},

    // RGBA
    {{CL_RGBA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNORM_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    {{CL_RGBA, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_SNORM_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    {{CL_RGBA, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_SIGNED_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_SIGNED_INT32},
     Pal::ChNumFormat::X32Y32Z32W32_Sint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNSIGNED_INT16},
     Pal::ChNumFormat::X16Y16Z16W16_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_UNSIGNED_INT32},
     Pal::ChNumFormat::X32Y32Z32W32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    {{CL_RGBA, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16Y16Z16W16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},
    {{CL_RGBA, CL_FLOAT},
     Pal::ChNumFormat::X32Y32Z32W32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    // ARGB
    {{CL_ARGB, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Unorm,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},
    {{CL_ARGB, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Snorm,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},
    {{CL_ARGB, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Sint,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},
    {{CL_ARGB, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Uint,
     {Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W,
      Pal::ChannelSwizzle::X}},

    // BGRA
    {{CL_BGRA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Unorm,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},
    {{CL_BGRA, CL_SNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Snorm,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},
    {{CL_BGRA, CL_SIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Sint,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},
    {{CL_BGRA, CL_UNSIGNED_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Uint,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},

    // LUMINANCE
    {{CL_LUMINANCE, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},
    {{CL_LUMINANCE, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::One}},

    // INTENSITY
    {{CL_INTENSITY, CL_SNORM_INT8},
     Pal::ChNumFormat::X8_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_SNORM_INT16},
     Pal::ChNumFormat::X16_Snorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_UNORM_INT8},
     Pal::ChNumFormat::X8_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_HALF_FLOAT},
     Pal::ChNumFormat::X16_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_INTENSITY, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},

    // sRBGA
    {{CL_sRGBA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::W}},

    // sRBG
    {{CL_sRGB, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::One}},

    // sRBGx
    {{CL_sRGBx, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::Z,
      Pal::ChannelSwizzle::One}},

    // sBGRA
    {{CL_sBGRA, CL_UNORM_INT8},
     Pal::ChNumFormat::X8Y8Z8W8_Srgb,
     {Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::Y, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::W}},

    // DEPTH
    {{CL_DEPTH, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},

    {{CL_DEPTH, CL_UNORM_INT16},
     Pal::ChNumFormat::X16_Unorm,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},

    {{CL_DEPTH_STENCIL, CL_UNORM_INT24},
     Pal::ChNumFormat::X32_Uint,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}},
    {{CL_DEPTH_STENCIL, CL_FLOAT},
     Pal::ChNumFormat::X32_Float,
     {Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::X,
      Pal::ChannelSwizzle::X}}};

}  // namespace pal
