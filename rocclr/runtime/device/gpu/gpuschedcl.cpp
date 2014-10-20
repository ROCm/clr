//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

namespace gpu {

#define SCHEDULER_KERNEL(...) #__VA_ARGS__

const char* SchedulerSourceCode = SCHEDULER_KERNEL(
\n
extern void __amd_scheduler(__global void *, __global void *, uint);
\n
__kernel void
scheduler(
    __global void * queue,
    __global void * params,
    uint paramIdx)
{
    __amd_scheduler(queue, params, paramIdx);
}
\n
);

} // namespace gpu
