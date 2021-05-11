/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

/**
 *  @file  amd_detail/llvm_intrinsics.h
 *  @brief Contains declarations for wrapper functions for llvm intrinsics.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_LLVM_INTRINSICS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_LLVM_INTRINSICS_H

#include "hip/amd_detail/host_defines.h"

__device__
inline
unsigned __llvm_amdgcn_groupstaticsize() {
  return __builtin_amdgcn_groupstaticsize();
}

template<typename __T>
__device__
inline
__T __llvm_amdgcn_ds_swizzle(__T index, const int pattern) {
  return __builtin_amdgcn_ds_swizzle(index, pattern);
}

#endif
