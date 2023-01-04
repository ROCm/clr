/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

namespace roc {

#define BLIT_KERNEL(...) #__VA_ARGS__

const char* rocBlitLinearSourceCode = BLIT_KERNEL(

  // Extern
  extern void __amd_streamOpsWrite(__global uint*, __global ulong*, ulong, ulong);

  extern void __amd_streamOpsWait(__global uint*, __global ulong*, ulong, ulong, ulong);

  extern void __ockl_dm_init_v1(ulong, ulong, uint, uint);
  // Implementation
  __kernel void __amd_rocclr_streamOpsWrite(__global uint* ptrInt, __global ulong* ptrUlong,
                                            ulong value, ulong sizeBytes) {
    __amd_streamOpsWrite(ptrInt, ptrUlong, value, sizeBytes);
  }

  __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                           ulong value, ulong flags, ulong mask) {
    __amd_streamOpsWait(ptrInt, ptrUlong, value, flags, mask);
  }

  __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                      uint heap_size, uint number_of_initial_blocks) {
    __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
  }

);

const char* SchedulerSourceCode = BLIT_KERNEL(

  extern void __amd_scheduler_rocm(__global void*);

  __kernel void __amd_rocclr_scheduler(__global void* params) {
    __amd_scheduler_rocm(params);
  }
);

const char* GwsInitSourceCode = BLIT_KERNEL(

  extern void __ockl_gws_init(uint nwm1, uint rid);

  __kernel void __amd_rocclr_gwsInit(uint value) {
    __ockl_gws_init(value, 0);
  }
);

}  // namespace roc
