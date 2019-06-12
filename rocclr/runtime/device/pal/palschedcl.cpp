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

}  // namespace pal
