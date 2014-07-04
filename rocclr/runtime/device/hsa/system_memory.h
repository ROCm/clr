//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

/** @file */

#ifndef _OPENCL_RUNTIME_DEVICE_HSA_SYSTEM_MEMORY_H_
#define _OPENCL_RUNTIME_DEVICE_HSA_SYSTEM_MEMORY_H_

#include "newcore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 *******************************************************************************
 * @brief System memory types.
 * @details The memory option enumerations are used for specifying the various
 * configurable global system memory allocation options.
 *******************************************************************************
 */
typedef enum {
  /**
   * Memory option used for requesting cacheable system memory.
   */
  kHsaAmdSystemMemoryTypeDefault = 0,

  /**
   * Memory option used for requesting system memory with caching disabled.
   */
  kHsaAmdSystemMemoryTypeUncached = 1,

  /**
   * Memory option used for requesting write-combined system memory.
   */
  kHsaAmdSystemMemoryTypeWriteCombined = 2,

  /**
   * Shortcut to get the number of supported memory type.
   */
  kHsaAmdSystemMemoryTypeCount = 3
} HsaAmdSystemMemoryType;

/**
 ****************************************************************************
 * @brief Allocate system memory accessible by all AMD devices in the platform.
 * @details The HsaAmdAllocateSystemMemory() interface is used for allocating
 * global system memory accessible (read and write) by the host and all AMD
 * devices in the platform.
 *
 * @param size The allocation size in bytes.
 * @param alignment The alignment size in bytes for the address of resulting
 * allocation. If the value is zero, no particular alignment will be applied.
 * If the value is not zero, it needs to be a power of two and minimum of
 * sizeof(void*).
 * @param type Type of system memory.
 * @param address A pointer to the location of where to return the pointer to
 * the base of the allocated region of memory.
 *
 * @return HsaStatus
 * @retval kHsaStatusSuccess The requested amount of memory was successfully
 * allocated.
 * @retval kHsaStatusOutOfMemory The implementation was unable to allocate the
 * requested amount of device memory due to memory constraints.
 * @retval kHsaStatusInvalidArgument An address of NULL was specified, the size
 * is 0 or the alignment is invalid.
 *
 * @see HsaAmdFreeSystemMemory, HsaAmdSystemMemoryType
 **************************************************************************/
COREAPI HsaStatus HsaAmdAllocateSystemMemory(size_t size,
                                             size_t alignment,
                                             HsaAmdSystemMemoryType type,
                                             void **address);

/**
 ****************************************************************************
 * @brief Deallocate system memory.
 * @details The HsaAmdFreeSystemMemory() interface is used for
 * deallocating global system memory that was previously allocated with
 * HsaAmdAllocateSystemMemory().
 *
 * @param address A pointer to the address to be deallocated.
 *
 * @return HsaStatus
 * @retval kHsaStatusSuccess The requested memory was successfully deallocated.
 * @retval kHsaStatusInvalidArguement An address of NULL was specified.
 *
 * @see HsaAmdAllocateSystemMemory
 ***************************************************************************
 */
COREAPI HsaStatus HsaAmdFreeSystemMemory(void *address);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // header guard
