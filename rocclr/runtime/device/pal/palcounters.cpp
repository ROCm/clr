//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/pal/palcounters.hpp"
#include "device/pal/palvirtual.hpp"

namespace pal {

PalCounterReference*
PalCounterReference::Create(VirtualGPU& gpu)
{
    Pal::Result result;

    // Create performance experiment
    Pal::PerfExperimentCreateInfo   createInfo = {};

    createInfo.optionFlags.sampleInternalOperations       = 1;
    createInfo.optionFlags.cacheFlushOnCounterCollection  = 1;
    createInfo.optionFlags.sqShaderMask                   = 1;
    createInfo.optionValues.sampleInternalOperations      = true;
    createInfo.optionValues.cacheFlushOnCounterCollection = true;
    createInfo.optionValues.sqShaderMask = Pal::PerfShaderMaskCs;

    size_t palExperSize = gpu.dev().iDev()->GetPerfExperimentSize(
        createInfo, &result);
    if (result != Pal::Result::Success) {
        return nullptr;
    }

    PalCounterReference*  memRef = new (palExperSize) PalCounterReference(gpu);
    if (memRef != nullptr) {
        result = gpu.dev().iDev()->CreatePerfExperiment(createInfo,
            &memRef[1], &memRef->perfExp_);
        if (result != Pal::Result::Success) {
            memRef->release();
            return nullptr;
        }
    }

    return memRef;
}

PalCounterReference::~PalCounterReference()
{
    // The counter object is always associated with a particular queue,
    // so we have to lock just this queue
    amd::ScopedLock lock(gpu_.execution());

    if (layout_ != nullptr) {
        delete layout_;
    }

    if (memory_ != nullptr) {
        delete memory_;
    }

    if (nullptr != iPerf()) {
        iPerf()->Destroy();
    }
}

uint64_t PalCounterReference::result(uint index)
{
    if (layout_ != nullptr) {
        assert(index <= layout_->sampleCount && "index not in range");
        const Pal::GlobalSampleLayout& sample = layout_->samples[index];
        if (sample.dataType == Pal::PerfCounterDataType::Uint32) {
            uint32_t beginVal = *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.beginValueOffset);
            uint32_t endVal = *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.endValueOffset);
            return (endVal - beginVal);
        }
        else if (sample.dataType == Pal::PerfCounterDataType::Uint64) {
            uint64_t beginVal = *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.beginValueOffset);
            uint64_t endVal = *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.endValueOffset);
            return (endVal - beginVal);
        }
        else {
            assert(0 && "dataType should be either Uint32 or Uint64");
            return 0;
        }
    }

    return 0;
}

bool PalCounterReference::finalize()
{
    Pal::Result result;

    iPerf()->Finalize();

    // Acquire GPU memory for the query from the pool and bind it.
    Pal::GpuMemoryRequirements gpuMemReqs = {};
    iPerf()->GetGpuMemoryRequirements(&gpuMemReqs);

    memory_ = new Memory(gpu().dev(), amd::alignUp(gpuMemReqs.size, gpuMemReqs.alignment));

    if (nullptr == memory_) {
        return false;
    }

    if (!memory_->create(Resource::Remote)) {
        return false;
    }

    cpuAddr_ = memory_->cpuMap(gpu_);

    if (nullptr == cpuAddr_) {
        return false;
    }

    gpu_.queue(gpu_.engineID_).addMemRef(memory_->iMem());

    result = iPerf()->BindGpuMemory(memory_->iMem(), 0);

    if (result == Pal::Result::Success) {
        Pal::GlobalCounterLayout layout = {};
        iPerf()->GetGlobalCounterLayout(&layout);

        size_t size = sizeof(Pal::GlobalCounterLayout) + (sizeof(Pal::GlobalSampleLayout) * (layout.sampleCount - 1));
        layout_ = reinterpret_cast<Pal::GlobalCounterLayout*>(new char[size]);
        if (layout_ != nullptr) {
            layout_->sampleCount = layout.sampleCount;
            iPerf()->GetGlobalCounterLayout(layout_);
        }
        return true;
    }
    else {
        return false;
    }
}

PerfCounter::~PerfCounter()
{
    if (palRef_ == nullptr) {
        return;
    }

    // Release the counter reference object
    palRef_->release();
}

bool
PerfCounter::create()
{
    index_ = palRef_->retain() - 2;

    // Initialize the counter
    Pal::PerfCounterInfo counterInfo = {};
    counterInfo.counterType = Pal::PerfCounterType::Global;
    counterInfo.block       = static_cast<Pal::GpuBlock>(info_.blockIndex_);
    counterInfo.instance    = info_.counterIndex_;
    counterInfo.eventId     = info_.eventIndex_;
    Pal::Result result = iPerf()->AddCounter(counterInfo);
    if (result != Pal::Result::Success) {
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
        return palRef_->result(index_);
    }
    default:
        LogError("Wrong PerfCounter::getInfo parameter");
    }
    return 0;
}

} // namespace pal
