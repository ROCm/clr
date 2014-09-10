//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/hsa/oclhsa_common.hpp"
#include "device/hsa/hsacounters.hpp"
#include "device/hsa/hsavirtual.hpp"

namespace oclhsa {

PerfCounter::~PerfCounter()
{
    // Destroy the corresponding HSA counter object
    HsaStatus status;
    status = servicesapi->HsaPmuDestroyCounter(counter_block_, counter_);
    if (status != kHsaStatusSuccess) {
        LogError("Destroy counter failed");
        return;
    }

    // If no enabled counter corresponding to the PMU, 
    // Release the PMU
    uint32_t counter_num;
    if (!getEnabledCounterNum(counter_num)) {
        LogError("getEnabledCounterNum failed");
        return;
    }

    if (counter_num == 0) {
        status = servicesapi->HsaReleasePmu(hsaPmu_);
        if (status != kHsaStatusSuccess) {
            LogError("Destroy pmu failed");
            return;
        }
    }
}

bool
PerfCounter::create(HsaPmu hsaPmu)
{
    HsaStatus status;
    hsaPmu_ = hsaPmu;
    uint32_t blockIndex = static_cast<uint32_t>(info()->blockIndex_);
    status = servicesapi->HsaPmuGetCounterBlockById(hsaPmu_, blockIndex, &counter_block_);
    if (status != kHsaStatusSuccess) {
        LogError("HsaPmuGetCounterBlockById, failed");
        return false;
    }

    status = servicesapi->HsaPmuCreateCounter(counter_block_, &counter_);
    if (status != kHsaStatusSuccess) {
        LogPrintfError("HsaPmuCreateCounter, failed.\
                       Block: %d, counter: #d, event: %d",
                       info()->blockIndex_,
                       info()->counterIndex_,
                       info()->eventIndex_);
  
        return false;
    }

    status = servicesapi->HsaPmuCounterSetEnabled(counter_, true);
    if (status != kHsaStatusSuccess) {
        LogError("HsaPmuCounterSetEnabled, failed");
        return false;
    }

    uint32_t eventIndex = static_cast<uint32_t>(info()->eventIndex_);
    status = servicesapi->HsaPmuCounterSetParameter(counter_,
        kHsaCounterParameterEventIndex,
        sizeof(uint32_t), (void *)&eventIndex);
    if (status != kHsaStatusSuccess) {
        LogError("HsaPmuCounterSetParameter, failed");
        return false;
    }

    return true;
}

uint64_t
PerfCounter::getInfo(uint64_t infoType) const
{
    switch (infoType) {
    case CL_PERFCOUNTER_GPU_BLOCK_INDEX: {
        // Return the GPU block index
        return info()->blockIndex_;
    }
    case CL_PERFCOUNTER_GPU_COUNTER_INDEX: {
        // Return the GPU counter index
        return info()->counterIndex_;
    }
    case CL_PERFCOUNTER_GPU_EVENT_INDEX: {
        // Return the GPU event index
        return info()->eventIndex_;
    }
    case CL_PERFCOUNTER_DATA: {
        HsaStatus status;
        uint64_t counterValue;
        status = servicesapi->HsaPmuCounterGetResult(counter_, &counterValue);
        if (status != kHsaStatusSuccess) {
            LogError("HsaPmuCounterGetResult, failed");
        }
        return counterValue;
    }
    default:
        LogError("Wrong PerfCounter::getInfo parameter");
    }

    return 0;
}

bool 
PerfCounter::getEnabledCounterNum(uint32_t &counter_num) 
{
    // Collect all the program counter blocks
    uint32_t counterblock_num, num;
    uint32_t i;
    HsaStatus status;
    HsaCounter *pp_counters;
    HsaCounterBlock *pp_counterblocks;
    status = servicesapi->HsaPmuGetAllCounterBlocks(hsaPmu_, 
                                       &pp_counterblocks, 
                                       &counterblock_num);
    if (status != kHsaStatusSuccess) {
        LogError("HsaPmuGetAllCounterBlocks, failed");
        return false;
    }

    counter_num = 0;
    for (i = 0; i < counterblock_num; i++) {
        // Retrieve all enabled pp_counters in each counter block
        status = servicesapi->HsaPmuGetEnabledCounters(pp_counterblocks[i], 
                                          &pp_counters, &num);
        if (status != kHsaStatusSuccess) {
            LogError("HsaPmuGetEnabledCounters, failed");
            return false;
        }
        counter_num += num;
    }

    return true;
}


} // namespace oclhsa
