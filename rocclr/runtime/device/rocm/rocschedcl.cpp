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
__kernel void gwsInit(uint value) {
    unsigned int m0_backup, new_m0;
    __asm__ __volatile__(
        "s_mov_b32 %0 m0\n"
        "v_readfirstlane_b32 %1 %2\n"
        "s_nop 0\n"
        "s_mov_b32 m0 %1\n"
        "s_nop 0\n"
        "ds_gws_init %3 offset:0 gds\n"
        "s_waitcnt lgkmcnt(0) expcnt(0)\n"
        "s_mov_b32 m0 %0\n"
        "s_nop 0"
        : "=s"(m0_backup), "=s"(new_m0)
        : "v"(0 << 0x10), "{v0}"(value - 1)
        : "memory");
}
\n);

}  // namespace roc
