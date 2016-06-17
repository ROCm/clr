//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/pal/paldefs.hpp"
#include "device/pal/palcounters.hpp"
#include "device/pal/palvirtual.hpp"

namespace pal {

PalCounterReference*
PalCounterReference::Create(
   VirtualGPU&     gpu,
    const Pal::PerfExperimentCreateInfo& createInfo)
{
    Pal::Result result;
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
    if (nullptr != iPerf()) {
        iPerf()->Destroy();
    }
}

bool
PalCounterReference::growResultArray(uint index)
{
    if (results_ != nullptr) {
        delete [] results_;
    }
    results_ = new uint64_t [index + 1];
    if (results_ == nullptr) {
        return false;
    }
    return true;
}

bool PalCounterReference::finalize()
{
    iPerf()->Finalize();

    // Acquire GPU memory for the query from the pool and bind it.
    Pal::GpuMemoryRequirements gpuMemReqs = {};
    iPerf()->GetGpuMemoryRequirements(&gpuMemReqs);

    Pal::GpuMemoryCreateInfo createInfo = {};
    createInfo.size      = amd::alignUp(gpuMemReqs.size, gpuMemReqs.alignment);
    createInfo.alignment = gpuMemReqs.alignment;
    createInfo.vaRange   = Pal::VaRange::Default;
    createInfo.heapCount = 1;
    //createInfo.heaps[0]  = Pal::GpuHeapGartCacheable;
    createInfo.heaps[0]  = gpuMemReqs.heaps[0];
    createInfo.priority  = Pal::GpuMemPriority::Normal;

    Pal::Result result;
    size_t gpuMemSize = gpu().dev().iDev()->GetGpuMemorySize(createInfo, &result);
    if (result != Pal::Result::Success) {
        return false;
    }

    char* pMemory = new char[gpuMemSize];

    if (pMemory != nullptr) {
        result = gpu().dev().iDev()->CreateGpuMemory(createInfo, pMemory, &pGpuMemory);

        if (result == Pal::Result::Success) {
            // GPU profiler memory is perma-resident.
            result = gpu().dev().iDev()->AddGpuMemoryReferences(1, &pGpuMemory, nullptr, Pal::GpuMemoryRefCantTrim);

            if (result == Pal::Result::Success) {
                // GPU profiler memory is perma-mapped.
                result = pGpuMemory->Map(&pCpuAddr);
            }
        }
        
        if (result != Pal::Result::Success) {
            if (pGpuMemory != nullptr) {
                pGpuMemory->Destroy();
                pGpuMemory = nullptr;
                pCpuAddr   = nullptr;
            }

            delete pMemory;
            return false;
        }
    }
    else {
        return false;
    }

    result = iPerf()->BindGpuMemory(pGpuMemory, 0);
    if (result == Pal::Result::Success) { 
        return true;
    }
    else {
        return false;
    }
}

PerfCounter::~PerfCounter()
{
    if (calRef_ == nullptr) {
        return;
    }

    // Release the counter reference object
    calRef_->release();
}

bool
PerfCounter::create(
    PalCounterReference*    calRef)
{
    assert(&gpu() == &calRef->gpu());

    calRef_ = calRef;
    counter_ = calRef->iPerf();
    index_ = calRef->retain() - 2;
    calRef->growResultArray(index_);

    // Initialize the counter
    Pal::PerfCounterInfo counterInfo = {};
    counterInfo.counterType = Pal::PerfCounterType::Global;
    counterInfo.block       = static_cast<Pal::GpuBlock>(info_.blockIndex_);
    counterInfo.instance    = info_.counterIndex_;
    counterInfo.eventId     = info_.eventIndex_;
    Pal::Result result = counter_->AddCounter(counterInfo);
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
        Pal::GlobalCounterLayout layout = {};
        counter_->GetGlobalCounterLayout(&layout);

        return calRef_->results()[index_];
    }
    default:
        LogError("Wrong PerfCounter::getInfo parameter");
    }
    return 0;
}

} // namespace pal
