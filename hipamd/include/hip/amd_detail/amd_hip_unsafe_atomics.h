/*
Copyright (c) 2021 - Present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#ifdef __cplusplus
/**
 * @brief Unsafe floating point rmw atomic add for gfx90a.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation is only suppored for the gfx90a target.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 4 bytes aligned
 * - \p addr is a global segment address in a coarse grain allocation.
 * Global segment addresses in fine grain allocations, group segment addresses,
 * and private segment addresses (used for function argument and function local
 * variables) are not supported.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
#if __has_builtin(__builtin_amdgcn_is_shared) &&                               \
    __has_builtin(__builtin_amdgcn_ds_atomic_fadd_f32) &&                      \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)
__device__ inline float unsafeAtomicAdd(float* addr, float value) {
  if (__builtin_amdgcn_is_shared(
          (const __attribute__((address_space(0))) void*)addr))
    return __builtin_amdgcn_ds_atomic_fadd_f32(addr, value);
  else
    return __builtin_amdgcn_global_atomic_fadd_f32(addr, value);
}
#endif

/**
 * @brief Unsafe double precision rmw atomic add for gfx90a.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation is only suppored for the gfx90a target.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - \p addr is a global segment address in a coarse grain allocation.
 * Global segment addresses in fine grain allocations, group segment addresses,
 * and private segment addresses (used for function argument and function local
 * variables) are not supported.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
#if __has_builtin(__builtin_amdgcn_is_shared) &&                               \
    __has_builtin(__builtin_amdgcn_ds_atomic_fadd_f64) &&                      \
    __has_builtin(__builtin_amdgcn_flat_atomic_fadd_f64)
__device__ inline double unsafeAtomicAdd(double* addr, double value) {
  if (__builtin_amdgcn_is_shared(
          (const __attribute__((address_space(0))) void*)addr))
    return __builtin_amdgcn_ds_atomic_fadd_f64(addr, value);
  else
    return __builtin_amdgcn_flat_atomic_fadd_f64(addr, value);
}
#endif
#endif
