//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
namespace pal {

#define BLIT_KERNEL(...) #__VA_ARGS__

const char* SchedulerSourceCode = BLIT_KERNEL(
%s
\n
extern void __amd_scheduler(__global void*, __global void*, uint);
\n
__kernel void scheduler(__global void* queue, __global void* params, uint paramIdx) {
  __amd_scheduler(queue, params, paramIdx);
}
\n);

const char* GwsInitSourceCode = BLIT_KERNEL(
\n
extern void __ockl_gws_init(uint nwm1, uint rid);
\n
__kernel void gwsInit(uint value) {
  __ockl_gws_init(value, 0);
}
\n);

}  // namespace pal
