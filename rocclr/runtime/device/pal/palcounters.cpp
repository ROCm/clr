//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/pal/palcounters.hpp"
#include "device/pal/palvirtual.hpp"
#include <array>

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

uint64_t PalCounterReference::result(const std::vector<int>& index)
{
    if (index.size() == 0) {
        // These are counters that have no corresponding PalSample created
        return 0;
    }

    if (layout_ == nullptr) {
        return 0;
    }

    uint64_t result = 0;
    for (auto const& i: index) {
        assert(i <= static_cast<int>(layout_->sampleCount) && "index not in range");
        const Pal::GlobalSampleLayout& sample = layout_->samples[i];
        if (sample.dataType == Pal::PerfCounterDataType::Uint32) {
            uint32_t beginVal = *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.beginValueOffset);
            uint32_t endVal = *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.endValueOffset);
            result += (endVal - beginVal);
        }
        else if (sample.dataType == Pal::PerfCounterDataType::Uint64) {
            uint64_t beginVal = *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.beginValueOffset);
            uint64_t endVal = *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.endValueOffset);
            result += (endVal - beginVal);
        }
        else {
            assert(0 && "dataType should be either Uint32 or Uint64");
            return 0;
        }
    }

    return result;
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

        assert(layout.sampleCount == numExpCounters_);
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

// Converting from ORCA cmndefs.h to PAL palPerfExperiment.h
static const
std::array<std::tuple<int, int, PCIndexSelect>, 83> ciBlockIdOrcaToPal =
{{
    {0x0E, 0, PCIndexSelect::ShaderEngineAndInstance},      // CB0
    {0x0E, 1, PCIndexSelect::ShaderEngineAndInstance},      // CB1
    {0x0E, 2, PCIndexSelect::ShaderEngineAndInstance},      // CB2
    {0x0E, 3, PCIndexSelect::ShaderEngineAndInstance},      // CB3
    {0x00, 0, PCIndexSelect::None},                         // CPF
    {0x0D, 0, PCIndexSelect::ShaderEngineAndInstance},      // DB0
    {0x0D, 1, PCIndexSelect::ShaderEngineAndInstance},      // DB1
    {0x0D, 2, PCIndexSelect::ShaderEngineAndInstance},      // DB2
    {0x0D, 3, PCIndexSelect::ShaderEngineAndInstance},      // DB3
    {0x11, 0, PCIndexSelect::None},                         // GRBM
    {0x12, 0, PCIndexSelect::None},                         // GRBMSE
    {0x03, 0, PCIndexSelect::ShaderEngine},                 // PA_SU
    {0x03, 0, PCIndexSelect::ShaderEngine},                 // PA_SC
    {0x05, 0, PCIndexSelect::ShaderEngine},                 // SPI
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_ES
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_GS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_VS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_PS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_LS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_HS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_CS
    {0x07, 0, PCIndexSelect::ShaderEngine},                 // SX
    {0x08, 0, PCIndexSelect::ShaderEngineAndInstance},      // TA0
    {0x08, 1, PCIndexSelect::ShaderEngineAndInstance},      // TA1
    {0x08, 2, PCIndexSelect::ShaderEngineAndInstance},      // TA2
    {0x08, 3, PCIndexSelect::ShaderEngineAndInstance},      // TA3
    {0x08, 4, PCIndexSelect::ShaderEngineAndInstance},      // TA4
    {0x08, 5, PCIndexSelect::ShaderEngineAndInstance},      // TA5
    {0x08, 6, PCIndexSelect::ShaderEngineAndInstance},      // TA6
    {0x08, 7, PCIndexSelect::ShaderEngineAndInstance},      // TA7
    {0x08, 8, PCIndexSelect::ShaderEngineAndInstance},      // TA8
    {0x08, 9, PCIndexSelect::ShaderEngineAndInstance},      // TA9
    {0x08, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TA10
    {0x0C, 0, PCIndexSelect::Instance},                     // TCA0
    {0x0C, 1, PCIndexSelect::Instance},                     // TCA1
    {0x0B, 0, PCIndexSelect::Instance},                     // TCC0
    {0x0B, 1, PCIndexSelect::Instance},                     // TCC1
    {0x0B, 2, PCIndexSelect::Instance},                     // TCC2
    {0x0B, 3, PCIndexSelect::Instance},                     // TCC3
    {0x0B, 4, PCIndexSelect::Instance},                     // TCC4
    {0x0B, 5, PCIndexSelect::Instance},                     // TCC5
    {0x0B, 6, PCIndexSelect::Instance},                     // TCC6
    {0x0B, 7, PCIndexSelect::Instance},                     // TCC7
    {0x0B, 8, PCIndexSelect::Instance},                     // TCC8
    {0x0B, 9, PCIndexSelect::Instance},                     // TCC9
    {0x0B, 0x0a, PCIndexSelect::Instance},                  // TCC10
    {0x0B, 0x0b, PCIndexSelect::Instance},                  // TCC11
    {0x0B, 0x0c, PCIndexSelect::Instance},                  // TCC12
    {0x0B, 0x0d, PCIndexSelect::Instance},                  // TCC13
    {0x0B, 0x0e, PCIndexSelect::Instance},                  // TCC14
    {0x0B, 0x0f, PCIndexSelect::Instance},                  // TCC15
    {0x09, 0, PCIndexSelect::ShaderEngineAndInstance},      // TD0
    {0x09, 1, PCIndexSelect::ShaderEngineAndInstance},      // TD1
    {0x09, 2, PCIndexSelect::ShaderEngineAndInstance},      // TD2
    {0x09, 3, PCIndexSelect::ShaderEngineAndInstance},      // TD3
    {0x09, 4, PCIndexSelect::ShaderEngineAndInstance},      // TD4
    {0x09, 5, PCIndexSelect::ShaderEngineAndInstance},      // TD5
    {0x09, 6, PCIndexSelect::ShaderEngineAndInstance},      // TD6
    {0x09, 7, PCIndexSelect::ShaderEngineAndInstance},      // TD7
    {0x09, 8, PCIndexSelect::ShaderEngineAndInstance},      // TD8
    {0x09, 9, PCIndexSelect::ShaderEngineAndInstance},      // TD9
    {0x09, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TD10
    {0x0A, 0, PCIndexSelect::ShaderEngineAndInstance},      // TCP0
    {0x0A, 1, PCIndexSelect::ShaderEngineAndInstance},      // TCP1
    {0x0A, 2, PCIndexSelect::ShaderEngineAndInstance},      // TCP2
    {0x0A, 3, PCIndexSelect::ShaderEngineAndInstance},      // TCP3
    {0x0A, 4, PCIndexSelect::ShaderEngineAndInstance},      // TCP4
    {0x0A, 5, PCIndexSelect::ShaderEngineAndInstance},      // TCP5
    {0x0A, 6, PCIndexSelect::ShaderEngineAndInstance},      // TCP6
    {0x0A, 7, PCIndexSelect::ShaderEngineAndInstance},      // TCP7
    {0x0A, 8, PCIndexSelect::ShaderEngineAndInstance},      // TCP8
    {0x0A, 9, PCIndexSelect::ShaderEngineAndInstance},      // TCP9
    {0x0A, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TCP10
    {0x0F, 0, PCIndexSelect::None},                         // GDS
    {0x02, 0, PCIndexSelect::ShaderEngine},                 // VGT
    {0x01, 0, PCIndexSelect::ShaderEngine},                 // IA
    {0x15, 0, PCIndexSelect::None},                         // MC
    {0x10, 0, PCIndexSelect::None},                         // SRBM
    {0x19, 0, PCIndexSelect::None},                         // TCS
    {0x18, 0, PCIndexSelect::None},                         // WD
    {0x16, 0, PCIndexSelect::None},                         // CPG
    {0x17, 0, PCIndexSelect::None},                         // CPC
}};

static const
std::array<std::tuple<int, int, PCIndexSelect>, 97> viBlockIdOrcaToPal =
{{
    {0x0E, 0, PCIndexSelect::ShaderEngineAndInstance},      // CB0
    {0x0E, 1, PCIndexSelect::ShaderEngineAndInstance},      // CB1
    {0x0E, 2, PCIndexSelect::ShaderEngineAndInstance},      // CB2
    {0x0E, 3, PCIndexSelect::ShaderEngineAndInstance},      // CB3
    {0x00, 0, PCIndexSelect::None},                         // CPF
    {0x0D, 0, PCIndexSelect::ShaderEngineAndInstance},      // DB0
    {0x0D, 1, PCIndexSelect::ShaderEngineAndInstance},      // DB1
    {0x0D, 2, PCIndexSelect::ShaderEngineAndInstance},      // DB2
    {0x0D, 3, PCIndexSelect::ShaderEngineAndInstance},      // DB3
    {0x11, 0, PCIndexSelect::None},                         // GRBM
    {0x12, 0, PCIndexSelect::None},                         // GRBMSE
    {0x03, 0, PCIndexSelect::ShaderEngine},                 // PA_SU
    {0x03, 0, PCIndexSelect::ShaderEngine},                 // PA_SC
    {0x05, 0, PCIndexSelect::ShaderEngine},                 // SPI
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_ES
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_GS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_VS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_PS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_LS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_HS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_CS
    {0x07, 0, PCIndexSelect::ShaderEngine},                 // SX
    {0x08, 0, PCIndexSelect::ShaderEngineAndInstance},      // TA0
    {0x08, 1, PCIndexSelect::ShaderEngineAndInstance},      // TA1
    {0x08, 2, PCIndexSelect::ShaderEngineAndInstance},      // TA2
    {0x08, 3, PCIndexSelect::ShaderEngineAndInstance},      // TA3
    {0x08, 4, PCIndexSelect::ShaderEngineAndInstance},      // TA4
    {0x08, 5, PCIndexSelect::ShaderEngineAndInstance},      // TA5
    {0x08, 6, PCIndexSelect::ShaderEngineAndInstance},      // TA6
    {0x08, 7, PCIndexSelect::ShaderEngineAndInstance},      // TA7
    {0x08, 8, PCIndexSelect::ShaderEngineAndInstance},      // TA8
    {0x08, 9, PCIndexSelect::ShaderEngineAndInstance},      // TA9
    {0x08, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TA10
    {0x08, 0x0b, PCIndexSelect::ShaderEngineAndInstance},   // TA11
    {0x08, 0x0c, PCIndexSelect::ShaderEngineAndInstance},   // TA12
    {0x08, 0x0d, PCIndexSelect::ShaderEngineAndInstance},   // TA13
    {0x08, 0x0e, PCIndexSelect::ShaderEngineAndInstance},   // TA14
    {0x08, 0x0f, PCIndexSelect::ShaderEngineAndInstance},   // TA15
    {0x0C, 0, PCIndexSelect::Instance},                     // TCA0
    {0x0C, 1, PCIndexSelect::Instance},                     // TCA1
    {0x0B, 0, PCIndexSelect::Instance},                     // TCC0
    {0x0B, 1, PCIndexSelect::Instance},                     // TCC1
    {0x0B, 2, PCIndexSelect::Instance},                     // TCC2
    {0x0B, 3, PCIndexSelect::Instance},                     // TCC3
    {0x0B, 4, PCIndexSelect::Instance},                     // TCC4
    {0x0B, 5, PCIndexSelect::Instance},                     // TCC5
    {0x0B, 6, PCIndexSelect::Instance},                     // TCC6
    {0x0B, 7, PCIndexSelect::Instance},                     // TCC7
    {0x0B, 8, PCIndexSelect::Instance},                     // TCC8
    {0x0B, 9, PCIndexSelect::Instance},                     // TCC9
    {0x0B, 0x0a, PCIndexSelect::Instance},                  // TCC10
    {0x0B, 0x0b, PCIndexSelect::Instance},                  // TCC11
    {0x0B, 0x0c, PCIndexSelect::Instance},                  // TCC12
    {0x0B, 0x0d, PCIndexSelect::Instance},                  // TCC13
    {0x0B, 0x0e, PCIndexSelect::Instance},                  // TCC14
    {0x0B, 0x0f, PCIndexSelect::Instance},                  // TCC15
    {0x09, 0, PCIndexSelect::ShaderEngineAndInstance},      // TD0
    {0x09, 1, PCIndexSelect::ShaderEngineAndInstance},      // TD1
    {0x09, 2, PCIndexSelect::ShaderEngineAndInstance},      // TD2
    {0x09, 3, PCIndexSelect::ShaderEngineAndInstance},      // TD3
    {0x09, 4, PCIndexSelect::ShaderEngineAndInstance},      // TD4
    {0x09, 5, PCIndexSelect::ShaderEngineAndInstance},      // TD5
    {0x09, 6, PCIndexSelect::ShaderEngineAndInstance},      // TD6
    {0x09, 7, PCIndexSelect::ShaderEngineAndInstance},      // TD7
    {0x09, 8, PCIndexSelect::ShaderEngineAndInstance},      // TD8
    {0x09, 9, PCIndexSelect::ShaderEngineAndInstance},      // TD9
    {0x09, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TD10
    {0x09, 0x0b, PCIndexSelect::ShaderEngineAndInstance},   // TD11
    {0x09, 0x0c, PCIndexSelect::ShaderEngineAndInstance},   // TD12
    {0x09, 0x0d, PCIndexSelect::ShaderEngineAndInstance},   // TD13
    {0x09, 0x0e, PCIndexSelect::ShaderEngineAndInstance},   // TD14
    {0x09, 0x0f, PCIndexSelect::ShaderEngineAndInstance},   // TD15
    {0x0A, 0, PCIndexSelect::ShaderEngineAndInstance},      // TCP0
    {0x0A, 1, PCIndexSelect::ShaderEngineAndInstance},      // TCP1
    {0x0A, 2, PCIndexSelect::ShaderEngineAndInstance},      // TCP2
    {0x0A, 3, PCIndexSelect::ShaderEngineAndInstance},      // TCP3
    {0x0A, 4, PCIndexSelect::ShaderEngineAndInstance},      // TCP4
    {0x0A, 5, PCIndexSelect::ShaderEngineAndInstance},      // TCP5
    {0x0A, 6, PCIndexSelect::ShaderEngineAndInstance},      // TCP6
    {0x0A, 7, PCIndexSelect::ShaderEngineAndInstance},      // TCP7
    {0x0A, 8, PCIndexSelect::ShaderEngineAndInstance},      // TCP8
    {0x0A, 9, PCIndexSelect::ShaderEngineAndInstance},      // TCP9
    {0x0A, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TCP10
    {0x0A, 0x0b, PCIndexSelect::ShaderEngineAndInstance},   // TCP11
    {0x0A, 0x0c, PCIndexSelect::ShaderEngineAndInstance},   // TCP12
    {0x0A, 0x0d, PCIndexSelect::ShaderEngineAndInstance},   // TCP13
    {0x0A, 0x0e, PCIndexSelect::ShaderEngineAndInstance},   // TCP14
    {0x0A, 0x0f, PCIndexSelect::ShaderEngineAndInstance},   // TCP15
    {0x0F, 0, PCIndexSelect::None},                         // GDS
    {0x02, 0, PCIndexSelect::ShaderEngine},                 // VGT
    {0x01, 0, PCIndexSelect::ShaderEngine},                 // IA
    {0x15, 0, PCIndexSelect::None},                         // MC
    {0x10, 0, PCIndexSelect::None},                         // SRBM
    {0x18, 0, PCIndexSelect::None},                         // WD
    {0x16, 0, PCIndexSelect::None},                         // CPG
    {0x17, 0, PCIndexSelect::None},                         // CPC
}};

// The number of counters per block has been increased for gfx9 but this table may not reflect all of them
// as compute may not use all of them.
static const
std::array<std::tuple<int, int, PCIndexSelect>, 104> gfx9BlockIdPal =
{{
    {0x0E, 0, PCIndexSelect::ShaderEngineAndInstance},      // CB0
    {0x0E, 1, PCIndexSelect::ShaderEngineAndInstance},      // CB1
    {0x0E, 2, PCIndexSelect::ShaderEngineAndInstance},      // CB2
    {0x0E, 3, PCIndexSelect::ShaderEngineAndInstance},      // CB3
    {0x00, 0, PCIndexSelect::Instance},                     // CPF0
    {0x00, 1, PCIndexSelect::Instance},                     // CPF1
    {0x0D, 0, PCIndexSelect::ShaderEngineAndInstance},      // DB0
    {0x0D, 1, PCIndexSelect::ShaderEngineAndInstance},      // DB1
    {0x0D, 2, PCIndexSelect::ShaderEngineAndInstance},      // DB2
    {0x0D, 3, PCIndexSelect::ShaderEngineAndInstance},      // DB3
    {0x11, 0, PCIndexSelect::Instance},                     // GRBM0
    {0x11, 1, PCIndexSelect::Instance},                     // GRBM1
    {0x12, 0, PCIndexSelect::Instance},                     // GRBMSE0
    {0x03, 0, PCIndexSelect::ShaderEngine},                 // PA_SU
    {0x03, 0, PCIndexSelect::ShaderEngine},                 // PA_SC
    {0x05, 0, PCIndexSelect::ShaderEngine},                 // SPI
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ0
    {0x06, 1, PCIndexSelect::ShaderEngine},                 // SQ1
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_ES
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_GS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_VS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_PS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_LS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_HS
    {0x06, 0, PCIndexSelect::ShaderEngine},                 // SQ_CS0
    {0x06, 1, PCIndexSelect::ShaderEngine},                 // SQ_CS1
    {0x07, 0, PCIndexSelect::ShaderEngine},                 // SX
    {0x08, 0, PCIndexSelect::ShaderEngineAndInstance},      // TA0
    {0x08, 1, PCIndexSelect::ShaderEngineAndInstance},      // TA1
    {0x08, 2, PCIndexSelect::ShaderEngineAndInstance},      // TA2
    {0x08, 3, PCIndexSelect::ShaderEngineAndInstance},      // TA3
    {0x08, 4, PCIndexSelect::ShaderEngineAndInstance},      // TA4
    {0x08, 5, PCIndexSelect::ShaderEngineAndInstance},      // TA5
    {0x08, 6, PCIndexSelect::ShaderEngineAndInstance},      // TA6
    {0x08, 7, PCIndexSelect::ShaderEngineAndInstance},      // TA7
    {0x08, 8, PCIndexSelect::ShaderEngineAndInstance},      // TA8
    {0x08, 9, PCIndexSelect::ShaderEngineAndInstance},      // TA9
    {0x08, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TA10
    {0x08, 0x0b, PCIndexSelect::ShaderEngineAndInstance},   // TA11
    {0x08, 0x0c, PCIndexSelect::ShaderEngineAndInstance},   // TA12
    {0x08, 0x0d, PCIndexSelect::ShaderEngineAndInstance},   // TA13
    {0x08, 0x0e, PCIndexSelect::ShaderEngineAndInstance},   // TA14
    {0x08, 0x0f, PCIndexSelect::ShaderEngineAndInstance},   // TA15
    {0x0C, 0, PCIndexSelect::Instance},                     // TCA0
    {0x0C, 1, PCIndexSelect::Instance},                     // TCA1
    {0x0B, 0, PCIndexSelect::Instance},                     // TCC0
    {0x0B, 1, PCIndexSelect::Instance},                     // TCC1
    {0x0B, 2, PCIndexSelect::Instance},                     // TCC2
    {0x0B, 3, PCIndexSelect::Instance},                     // TCC3
    {0x0B, 4, PCIndexSelect::Instance},                     // TCC4
    {0x0B, 5, PCIndexSelect::Instance},                     // TCC5
    {0x0B, 6, PCIndexSelect::Instance},                     // TCC6
    {0x0B, 7, PCIndexSelect::Instance},                     // TCC7
    {0x0B, 8, PCIndexSelect::Instance},                     // TCC8
    {0x0B, 9, PCIndexSelect::Instance},                     // TCC9
    {0x0B, 0x0a, PCIndexSelect::Instance},                  // TCC10
    {0x0B, 0x0b, PCIndexSelect::Instance},                  // TCC11
    {0x0B, 0x0c, PCIndexSelect::Instance},                  // TCC12
    {0x0B, 0x0d, PCIndexSelect::Instance},                  // TCC13
    {0x0B, 0x0e, PCIndexSelect::Instance},                  // TCC14
    {0x0B, 0x0f, PCIndexSelect::Instance},                  // TCC15
    {0x09, 0, PCIndexSelect::ShaderEngineAndInstance},      // TD0
    {0x09, 1, PCIndexSelect::ShaderEngineAndInstance},      // TD1
    {0x09, 2, PCIndexSelect::ShaderEngineAndInstance},      // TD2
    {0x09, 3, PCIndexSelect::ShaderEngineAndInstance},      // TD3
    {0x09, 4, PCIndexSelect::ShaderEngineAndInstance},      // TD4
    {0x09, 5, PCIndexSelect::ShaderEngineAndInstance},      // TD5
    {0x09, 6, PCIndexSelect::ShaderEngineAndInstance},      // TD6
    {0x09, 7, PCIndexSelect::ShaderEngineAndInstance},      // TD7
    {0x09, 8, PCIndexSelect::ShaderEngineAndInstance},      // TD8
    {0x09, 9, PCIndexSelect::ShaderEngineAndInstance},      // TD9
    {0x09, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TD10
    {0x09, 0x0b, PCIndexSelect::ShaderEngineAndInstance},   // TD11
    {0x09, 0x0c, PCIndexSelect::ShaderEngineAndInstance},   // TD12
    {0x09, 0x0d, PCIndexSelect::ShaderEngineAndInstance},   // TD13
    {0x09, 0x0e, PCIndexSelect::ShaderEngineAndInstance},   // TD14
    {0x09, 0x0f, PCIndexSelect::ShaderEngineAndInstance},   // TD15
    {0x0A, 0, PCIndexSelect::ShaderEngineAndInstance},      // TCP0
    {0x0A, 1, PCIndexSelect::ShaderEngineAndInstance},      // TCP1
    {0x0A, 2, PCIndexSelect::ShaderEngineAndInstance},      // TCP2
    {0x0A, 3, PCIndexSelect::ShaderEngineAndInstance},      // TCP3
    {0x0A, 4, PCIndexSelect::ShaderEngineAndInstance},      // TCP4
    {0x0A, 5, PCIndexSelect::ShaderEngineAndInstance},      // TCP5
    {0x0A, 6, PCIndexSelect::ShaderEngineAndInstance},      // TCP6
    {0x0A, 7, PCIndexSelect::ShaderEngineAndInstance},      // TCP7
    {0x0A, 8, PCIndexSelect::ShaderEngineAndInstance},      // TCP8
    {0x0A, 9, PCIndexSelect::ShaderEngineAndInstance},      // TCP9
    {0x0A, 0x0a, PCIndexSelect::ShaderEngineAndInstance},   // TCP10
    {0x0A, 0x0b, PCIndexSelect::ShaderEngineAndInstance},   // TCP11
    {0x0A, 0x0c, PCIndexSelect::ShaderEngineAndInstance},   // TCP12
    {0x0A, 0x0d, PCIndexSelect::ShaderEngineAndInstance},   // TCP13
    {0x0A, 0x0e, PCIndexSelect::ShaderEngineAndInstance},   // TCP14
    {0x0A, 0x0f, PCIndexSelect::ShaderEngineAndInstance},   // TCP15
    {0x0F, 0, PCIndexSelect::Instance},                     // GDS0
    {0x0F, 1, PCIndexSelect::Instance},                     // GDS1
    {0x02, 0, PCIndexSelect::ShaderEngine},                 // VGT
    {0x01, 0, PCIndexSelect::ShaderEngine},                 // IA
    {0x15, 0, PCIndexSelect::None},                         // MC
    {0x10, 0, PCIndexSelect::None},                         // SRBM
    {0x18, 0, PCIndexSelect::None},                         // WD
    {0x16, 0, PCIndexSelect::Instance},                     // CPG0
    {0x16, 1, PCIndexSelect::Instance},                     // CPG1
    {0x17, 0, PCIndexSelect::Instance},                     // CPC0
    {0x17, 1, PCIndexSelect::Instance},                     // CPC1
}};

void PerfCounter::convertInfo()
{
    switch (dev().ipLevel()) {
    case Pal::GfxIpLevel::GfxIp7:
        if (info_.blockIndex_ < ciBlockIdOrcaToPal.size()) {
            auto p = ciBlockIdOrcaToPal[info_.blockIndex_];
            info_.blockIndex_ = std::get<0>(p);
            info_.counterIndex_ = std::get<1>(p);
            info_.indexSelect_ = std::get<2>(p);
        }
        break;
    case Pal::GfxIpLevel::GfxIp8:
        if (info_.blockIndex_ < viBlockIdOrcaToPal.size()) {
            auto p = viBlockIdOrcaToPal[info_.blockIndex_];
            info_.blockIndex_ = std::get<0>(p);
            info_.counterIndex_ = std::get<1>(p);
            info_.indexSelect_ = std::get<2>(p);
        }
        break;
    case Pal::GfxIpLevel::GfxIp9:
        if (info_.blockIndex_ < gfx9BlockIdPal.size()) {
            auto p = gfx9BlockIdPal[info_.blockIndex_];
            info_.blockIndex_ = std::get<0>(p);
            info_.counterIndex_ = std::get<1>(p);
            info_.indexSelect_ = std::get<2>(p);
        }
        break;
    default:
        Unimplemented();
        break;
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
    palRef_->retain();

    // Initialize the counter
    Pal::PerfCounterInfo counterInfo = {};
    counterInfo.counterType = Pal::PerfCounterType::Global;
    counterInfo.block       = static_cast<Pal::GpuBlock>(info_.blockIndex_);
    counterInfo.eventId     = info_.eventIndex_;

    Pal::PerfExperimentProperties perfExpProps;
    Pal::Result result;
    result = dev().iDev()->GetPerfExperimentProperties(&perfExpProps);
    if (result != Pal::Result::Success) {
        return false;
    }

    const auto& blockProps = perfExpProps.blocks[static_cast<uint32_t>(counterInfo.block)];
    uint32_t counter_start, counter_step;

    switch (info_.indexSelect_) {
    case PCIndexSelect::ShaderEngine:
    case PCIndexSelect::None:
        counter_start = 0;
        counter_step = 1;
        break;

    case PCIndexSelect::ShaderEngineAndInstance:
        if (info_.counterIndex_ >=
            dev().properties().gfxipProperties.shaderCore.maxCusPerShaderArray) {
            return true;
        }
        counter_start = info_.counterIndex_;
        counter_step = dev().properties().gfxipProperties.shaderCore.maxCusPerShaderArray;
        break;

    case PCIndexSelect::Instance:
        counter_start = info_.counterIndex_;
        counter_step = blockProps.instanceCount;
        break;

    default:
        assert(0 && "Unknown indexSelect_");
        return true;
    }

    for (uint32_t i = counter_start; i < blockProps.instanceCount; i += counter_step) {
        counterInfo.instance = i;
        result = iPerf()->AddCounter(counterInfo);
        if (result == Pal::Result::Success) {
            index_.push_back(palRef_->getPalCounterIndex());
        }
        else {
            // Get here when there's no HW PerfCounter matching the counterInfo
            assert(0 && "AddCounter() failed");
        }
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
