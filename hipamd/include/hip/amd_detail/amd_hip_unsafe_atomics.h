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
 * @brief Unsafe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 4 bytes aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and is not supported.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicAdd(float* addr, float value) {
#if defined(__gfx90a__) &&                                                     \
    __has_builtin(__builtin_amdgcn_is_shared) &&                               \
    __has_builtin(__builtin_amdgcn_is_private) &&                              \
    __has_builtin(__builtin_amdgcn_ds_atomic_fadd_f32) &&                      \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)
  if (__builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)addr))
    return __builtin_amdgcn_ds_atomic_fadd_f32(addr, value);
  else if (__builtin_amdgcn_is_private(
              (const __attribute__((address_space(0))) void*)addr)) {
    float temp = *addr;
    *addr = temp + value;
    return temp;
  }
  else
    return __builtin_amdgcn_global_atomic_fadd_f32(addr, value);
#elif __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Unsafe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double unsafeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) &&                                                     \
    __has_builtin(__builtin_amdgcn_flat_atomic_fadd_f64)
  return __builtin_amdgcn_flat_atomic_fadd_f64(addr, value);
#elif defined (__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Safe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicAdd(float* addr, float value) {
#if defined(__gfx908__) ||                                                    \
    (defined(__gfx90a) && !__has_builtin(__hip_atomic_fetch_add))
  // On gfx908, we can generate unsafe FP32 atomic add that does not follow all
  // IEEE rules when -munsafe-fp-atomics is passed. Do a CAS loop emulation instead.
  // On gfx90a, if we do not have the __hip_atomic_fetch_add builtin, we need to
  // force a CAS loop here.
  float old_val;
#if __has_builtin(__hip_atomic_load)
  old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_load)
  old_val = __uint_as_float(__atomic_load_n(reinterpret_cast<unsigned int*>(addr), __ATOMIC_RELAXED));
#endif // __has_builtin(__hip_atomic_load)
  float expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__float_as_uint(temp) != __float_as_uint(old_val));
  return old_val;
#elif defined(__gfx90a__)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Safe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double safeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) &&                                                    \
    __has_builtin(__hip_atomic_fetch_add)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif defined(__gfx90a__)
  // On gfx90a, if we do not have the __hip_atomic_fetch_add builtin, we need to
  // force a CAS loop here.
  double old_val;
#if __has_builtin(__hip_atomic_load)
  old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_load)
  old_val = __longlong_as_double(__atomic_load_n(reinterpret_cast<unsigned long long*>(addr), __ATOMIC_RELAXED));
#endif // __has_builtin(__hip_atomic_load)
  double expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__double_as_longlong(temp) != __double_as_longlong(old_val));
  return old_val;
#else // !defined(__gfx90a__)
#if __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else  // !__has_builtin(__hip_atomic_fetch_add)
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_fetch_add)
#endif
}
#endif
