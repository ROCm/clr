//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
//
namespace roc {

#define SCHEDULER_KERNEL(...) #__VA_ARGS__

const char* SchedulerSourceCode = SCHEDULER_KERNEL(
\n
extern void __amd_scheduler_rocm(__global void*);
\n
__kernel void scheduler(__global void* params) {
  __amd_scheduler_rocm(params);
}
\n);

}  // namespace roc
