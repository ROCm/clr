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

#include <cstddef>
#include <cstdint>
#include "hsa.h"
#include "amd_hsa_kernel_code.h"
#include "device/device.hpp"
#include "device/hwdebug.hpp"
#include "acl.h"

static const int NumberReserveVgprs = 4;

namespace pal {

/**
 * \defgroup Services_API OCL Runtime Services API
 * @{
 */

/*!  \brief  Dispatch packet information
 *
 *   This structure contains the packet information for kernel dispatch
 */
struct PacketAmdInfo {
  uint32_t trapReservedVgprIndex_;    //!< reserved VGPR index, -1 when they are not valid
  uint32_t scratchBufferWaveOffset_;  //!< scratch buffer wave offset, -1 when no scratch buffer
  void* pointerToIsaBuffer_;          //!< pointer to the buffer containing ISA
  size_t sizeOfIsaBuffer_;            //!< size of the ISA buffer
  uint32_t numberOfVgprs_;            //!< number of VGPRs used by the kernel
  uint32_t numberOfSgprs_;            //!< number of SGPRs used by the kernel
  size_t sizeOfStaticGroupMemory_;    //!< Static local memory used by the kernel
};

/*! \brief Cache mask for invalidation
 */
struct HwDbgGpuCacheMask {
  HwDbgGpuCacheMask() : ui32All_(0) {}

  HwDbgGpuCacheMask(uint32_t mask) : ui32All_(mask) {}

  union {
    struct {
      uint32_t sqICache_ : 1;  //!< Instruction cache
      uint32_t sqKCache_ : 1;  //!< Data cache
      uint32_t tcL1_ : 1;      //!< tcL1 cache
      uint32_t tcL2_ : 1;      //!< tcL2 cache
      uint32_t reserved_ : 28;
    };
    uint32_t ui32All_;
  };
};

/*!  \brief Address watch information
 *
 *    Information about each watch point - address, mask, mode and event
 */
struct HwDbgAddressWatch {
  void* watchAddress_;                       //! The address of watch point
  uint64_t watchMask_;                       //! The mask for watch point (lower 24 bits)
  cl_dbg_address_watch_mode_amd watchMode_;  //! The watch mode for this watch
  DebugEvent event_;                         //! Event of the watch point (not used for now)
};

/*!  \brief Runtime structure used to communicate debug information
 *          between Ocl services and core for a kernel dispatch.
 */
struct DebugToolInfo {
  uint64_t scratchAddress_;    //! Scratch memory address
  size_t scratchSize_;         //! Scratch memory size
  uint64_t globalAddress_;     //! Global memory address
  uint32_t cacheDisableMask_;  //! Cache mask, indicating caches disabled
  uint32_t exceptionMask_;     //! Exception mask
  uint32_t reservedCuNum_;     //! Number of reserved CUs for display,
                               //!   which ranges from 0 to 7 in the current implementation.
  bool monitorMode_;           //! Debug or profiler mode
  bool gpuSingleStepMode_;     //! SQ debug mode
  amd::Memory* trapHandler_;   //! Trap handler address
  amd::Memory* trapBuffer_;    //! Trap buffer address
  bool sqPerfcounterEnable_;   //! whether SQ perf counters are enabled
  aclBinary* aclBinary_;       //! pointer of the kernel ACL binary
  amd::Event* event_;          //! pointer of the kernel event in the enqueue command
};

/*!  \brief Message used by the KFD wave control for CI
 *
 *   Structure indicates the various information used by the wave control function.
 */
struct HwDebugWaveAddr {
  uint32_t VMID_ : 4;  //! Virtual memory id
  uint32_t wave_ : 4;  //! Wave id
  uint32_t SIMD_ : 2;  //! SIMD id
  uint32_t CU_ : 4;    //! Compute unit
  uint32_t SH_ : 1;    //! Shader array
  uint32_t SE_ : 1;    //! Shader engine
};

/*! \brief Kernel code information
 *
 *   This structure contains the pointer of mapped kernel code for host access
 *   and its size (in bytes)
 */
struct AqlCodeInfo {
  amd_kernel_code_t* aqlCode_;  //! pointer of AQL code to allow host access
  uint32_t aqlCodeSize_;        //! size of AQL code
};

/**@}*/

}  // namespace pal
