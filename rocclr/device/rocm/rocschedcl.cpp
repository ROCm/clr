//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
//
namespace roc {

#define BLIT_KERNEL(...) #__VA_ARGS__

const char* SchedulerSourceCode = BLIT_KERNEL(
\n
extern void __amd_scheduler_rocm(__global void*);
\n
__kernel void scheduler(__global void* params) {
  __amd_scheduler_rocm(params);
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

}  // namespace roc
