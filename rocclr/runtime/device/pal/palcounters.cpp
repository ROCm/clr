//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/pal/palcounters.hpp"
#include "device/pal/palvirtual.hpp"
#include <array>

namespace pal {

PalCounterReference* PalCounterReference::Create(VirtualGPU& gpu) {
  Pal::Result result;

  // Create performance experiment
  Pal::PerfExperimentCreateInfo createInfo = {};

  createInfo.optionFlags.sampleInternalOperations = 1;
  createInfo.optionFlags.cacheFlushOnCounterCollection = 1;
  createInfo.optionFlags.sqShaderMask = 1;
  createInfo.optionValues.sampleInternalOperations = true;
  createInfo.optionValues.cacheFlushOnCounterCollection = true;
  createInfo.optionValues.sqShaderMask = Pal::PerfShaderMaskCs;

  size_t palExperSize = gpu.dev().iDev()->GetPerfExperimentSize(createInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  PalCounterReference* memRef = new (palExperSize) PalCounterReference(gpu);
  if (memRef != nullptr) {
    result = gpu.dev().iDev()->CreatePerfExperiment(createInfo, &memRef[1], &memRef->perfExp_);
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }

  return memRef;
}

PalCounterReference::~PalCounterReference() {
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

uint64_t PalCounterReference::result(const std::vector<int>& index) {
  if (index.size() == 0) {
    // These are counters that have no corresponding PalSample created
    return 0;
  }

  if (layout_ == nullptr) {
    return 0;
  }

  uint64_t result = 0;
  for (auto const& i : index) {
    assert(i <= static_cast<int>(layout_->sampleCount) && "index not in range");
    const Pal::GlobalSampleLayout& sample = layout_->samples[i];
    if (sample.dataType == Pal::PerfCounterDataType::Uint32) {
      uint32_t beginVal =
          *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.beginValueOffset);
      uint32_t endVal =
          *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.endValueOffset);
      result += (endVal - beginVal);
    } else if (sample.dataType == Pal::PerfCounterDataType::Uint64) {
      uint64_t beginVal =
          *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.beginValueOffset);
      uint64_t endVal =
          *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(cpuAddr_) + sample.endValueOffset);
      result += (endVal - beginVal);
    } else {
      assert(0 && "dataType should be either Uint32 or Uint64");
      return 0;
    }
  }

  return result;
}

bool PalCounterReference::finalize() {
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
    size_t size = sizeof(Pal::GlobalCounterLayout) +
        (sizeof(Pal::GlobalSampleLayout) * (layout.sampleCount - 1));
    layout_ = reinterpret_cast<Pal::GlobalCounterLayout*>(new char[size]);
    if (layout_ != nullptr) {
      layout_->sampleCount = layout.sampleCount;
      iPerf()->GetGlobalCounterLayout(layout_);
    }
    return true;
  } else {
    return false;
  }
}

static const std::array<PCIndexSelect, 0x20> blockIdToIndexSelect = {{
    PCIndexSelect::None,                     // CPF
    PCIndexSelect::ShaderEngine,             // IA
    PCIndexSelect::ShaderEngine,             // VGT
    PCIndexSelect::ShaderEngine,             // PA
    PCIndexSelect::ShaderEngine,             // SC
    PCIndexSelect::ShaderEngine,             // SPI
    PCIndexSelect::ShaderEngine,             // SQ
    PCIndexSelect::ShaderEngine,             // SX
    PCIndexSelect::ShaderEngineAndInstance,  // TA
    PCIndexSelect::ShaderEngineAndInstance,  // TD
    PCIndexSelect::ShaderEngineAndInstance,  // TCP
    PCIndexSelect::Instance,                 // TCC
    PCIndexSelect::Instance,                 // TCA
    PCIndexSelect::ShaderEngineAndInstance,  // DB
    PCIndexSelect::ShaderEngineAndInstance,  // CB
    PCIndexSelect::None,                     // GDS
    PCIndexSelect::None,                     // SRBM
    PCIndexSelect::None,                     // GRBM
    PCIndexSelect::None,                     // GRBMSE
    PCIndexSelect::None,                     // RLC
    PCIndexSelect::None,                     // DMA
    PCIndexSelect::None,                     // MC
    PCIndexSelect::None,                     // CPG
    PCIndexSelect::None,                     // CPC
    PCIndexSelect::None,                     // WD
    PCIndexSelect::None,                     // TCS
    PCIndexSelect::None,                     // UTC12
}};

// Converting from ORCA cmndefs.h to PAL palPerfExperiment.h
static const std::array<std::pair<int, int>, 83> ciBlockIdOrcaToPal = {{
    {0x0E, 0},     // CB0
    {0x0E, 1},     // CB1
    {0x0E, 2},     // CB2
    {0x0E, 3},     // CB3
    {0x00, 0},     // CPF
    {0x0D, 0},     // DB0
    {0x0D, 1},     // DB1
    {0x0D, 2},     // DB2
    {0x0D, 3},     // DB3
    {0x11, 0},     // GRBM
    {0x12, 0},     // GRBMSE
    {0x03, 0},     // PA_SU
    {0x03, 0},     // PA_SC
    {0x05, 0},     // SPI
    {0x06, 0},     // SQ
    {0x06, 0},     // SQ_ES
    {0x06, 0},     // SQ_GS
    {0x06, 0},     // SQ_VS
    {0x06, 0},     // SQ_PS
    {0x06, 0},     // SQ_LS
    {0x06, 0},     // SQ_HS
    {0x06, 0},     // SQ_CS
    {0x07, 0},     // SX
    {0x08, 0},     // TA0
    {0x08, 1},     // TA1
    {0x08, 2},     // TA2
    {0x08, 3},     // TA3
    {0x08, 4},     // TA4
    {0x08, 5},     // TA5
    {0x08, 6},     // TA6
    {0x08, 7},     // TA7
    {0x08, 8},     // TA8
    {0x08, 9},     // TA9
    {0x08, 0x0a},  // TA10
    {0x0C, 0},     // TCA0
    {0x0C, 1},     // TCA1
    {0x0B, 0},     // TCC0
    {0x0B, 1},     // TCC1
    {0x0B, 2},     // TCC2
    {0x0B, 3},     // TCC3
    {0x0B, 4},     // TCC4
    {0x0B, 5},     // TCC5
    {0x0B, 6},     // TCC6
    {0x0B, 7},     // TCC7
    {0x0B, 8},     // TCC8
    {0x0B, 9},     // TCC9
    {0x0B, 0x0a},  // TCC10
    {0x0B, 0x0b},  // TCC11
    {0x0B, 0x0c},  // TCC12
    {0x0B, 0x0d},  // TCC13
    {0x0B, 0x0e},  // TCC14
    {0x0B, 0x0f},  // TCC15
    {0x09, 0},     // TD0
    {0x09, 1},     // TD1
    {0x09, 2},     // TD2
    {0x09, 3},     // TD3
    {0x09, 4},     // TD4
    {0x09, 5},     // TD5
    {0x09, 6},     // TD6
    {0x09, 7},     // TD7
    {0x09, 8},     // TD8
    {0x09, 9},     // TD9
    {0x09, 0x0a},  // TD10
    {0x0A, 0},     // TCP0
    {0x0A, 1},     // TCP1
    {0x0A, 2},     // TCP2
    {0x0A, 3},     // TCP3
    {0x0A, 4},     // TCP4
    {0x0A, 5},     // TCP5
    {0x0A, 6},     // TCP6
    {0x0A, 7},     // TCP7
    {0x0A, 8},     // TCP8
    {0x0A, 9},     // TCP9
    {0x0A, 0x0a},  // TCP10
    {0x0F, 0},     // GDS
    {0x02, 0},     // VGT
    {0x01, 0},     // IA
    {0x15, 0},     // MC
    {0x10, 0},     // SRBM
    {0x19, 0},     // TCS
    {0x18, 0},     // WD
    {0x16, 0},     // CPG
    {0x17, 0},     // CPC
}};

static const std::array<std::pair<int, int>, 97> viBlockIdOrcaToPal = {{
    {0x0E, 0},     // CB0
    {0x0E, 1},     // CB1
    {0x0E, 2},     // CB2
    {0x0E, 3},     // CB3
    {0x00, 0},     // CPF
    {0x0D, 0},     // DB0
    {0x0D, 1},     // DB1
    {0x0D, 2},     // DB2
    {0x0D, 3},     // DB3
    {0x11, 0},     // GRBM
    {0x12, 0},     // GRBMSE
    {0x03, 0},     // PA_SU
    {0x03, 0},     // PA_SC
    {0x05, 0},     // SPI
    {0x06, 0},     // SQ
    {0x06, 0},     // SQ_ES
    {0x06, 0},     // SQ_GS
    {0x06, 0},     // SQ_VS
    {0x06, 0},     // SQ_PS
    {0x06, 0},     // SQ_LS
    {0x06, 0},     // SQ_HS
    {0x06, 0},     // SQ_CS
    {0x07, 0},     // SX
    {0x08, 0},     // TA0
    {0x08, 1},     // TA1
    {0x08, 2},     // TA2
    {0x08, 3},     // TA3
    {0x08, 4},     // TA4
    {0x08, 5},     // TA5
    {0x08, 6},     // TA6
    {0x08, 7},     // TA7
    {0x08, 8},     // TA8
    {0x08, 9},     // TA9
    {0x08, 0x0a},  // TA10
    {0x08, 0x0b},  // TA11
    {0x08, 0x0c},  // TA12
    {0x08, 0x0d},  // TA13
    {0x08, 0x0e},  // TA14
    {0x08, 0x0f},  // TA15
    {0x0C, 0},     // TCA0
    {0x0C, 1},     // TCA1
    {0x0B, 0},     // TCC0
    {0x0B, 1},     // TCC1
    {0x0B, 2},     // TCC2
    {0x0B, 3},     // TCC3
    {0x0B, 4},     // TCC4
    {0x0B, 5},     // TCC5
    {0x0B, 6},     // TCC6
    {0x0B, 7},     // TCC7
    {0x0B, 8},     // TCC8
    {0x0B, 9},     // TCC9
    {0x0B, 0x0a},  // TCC10
    {0x0B, 0x0b},  // TCC11
    {0x0B, 0x0c},  // TCC12
    {0x0B, 0x0d},  // TCC13
    {0x0B, 0x0e},  // TCC14
    {0x0B, 0x0f},  // TCC15
    {0x09, 0},     // TD0
    {0x09, 1},     // TD1
    {0x09, 2},     // TD2
    {0x09, 3},     // TD3
    {0x09, 4},     // TD4
    {0x09, 5},     // TD5
    {0x09, 6},     // TD6
    {0x09, 7},     // TD7
    {0x09, 8},     // TD8
    {0x09, 9},     // TD9
    {0x09, 0x0a},  // TD10
    {0x09, 0x0b},  // TD11
    {0x09, 0x0c},  // TD12
    {0x09, 0x0d},  // TD13
    {0x09, 0x0e},  // TD14
    {0x09, 0x0f},  // TD15
    {0x0A, 0},     // TCP0
    {0x0A, 1},     // TCP1
    {0x0A, 2},     // TCP2
    {0x0A, 3},     // TCP3
    {0x0A, 4},     // TCP4
    {0x0A, 5},     // TCP5
    {0x0A, 6},     // TCP6
    {0x0A, 7},     // TCP7
    {0x0A, 8},     // TCP8
    {0x0A, 9},     // TCP9
    {0x0A, 0x0a},  // TCP10
    {0x0A, 0x0b},  // TCP11
    {0x0A, 0x0c},  // TCP12
    {0x0A, 0x0d},  // TCP13
    {0x0A, 0x0e},  // TCP14
    {0x0A, 0x0f},  // TCP15
    {0x0F, 0},     // GDS
    {0x02, 0},     // VGT
    {0x01, 0},     // IA
    {0x15, 0},     // MC
    {0x10, 0},     // SRBM
    {0x18, 0},     // WD
    {0x16, 0},     // CPG
    {0x17, 0},     // CPC
}};

// The number of counters per block has been increased for gfx9 but this table may not reflect all
// of them as compute may not use all of them.
static const std::array<std::pair<int, int>, 110> gfx9BlockIdPal = {{
    {0x0E, 0},     // CB0       - 0
    {0x0E, 1},     // CB1       - 1
    {0x0E, 2},     // CB2       - 2
    {0x0E, 3},     // CB3       - 3
    {0x00, 0},     // CPF       - 4
    {0x0D, 0},     // DB0       - 5
    {0x0D, 1},     // DB1       - 6
    {0x0D, 2},     // DB2       - 7
    {0x0D, 3},     // DB3       - 8
    {0x11, 0},     // GRBM      - 9
    {0x12, 0},     // GRBMSE    - 10
    {0x03, 0},     // PA_SU     - 11
    {0x03, 0},     // PA_SC     - 12
    {0x05, 0},     // SPI       - 13
    {0x06, 0},     // SQ        - 14
    {0x06, 0},     // SQ_ES     - 15
    {0x06, 0},     // SQ_GS     - 16
    {0x06, 0},     // SQ_VS     - 17
    {0x06, 0},     // SQ_PS     - 18
    {0x06, 0},     // SQ_LS     - 19
    {0x06, 0},     // SQ_HS     - 20
    {0x06, 0},     // SQ_CS     - 21
    {0x07, 0},     // SX        - 22
    {0x08, 0},     // TA0       - 23
    {0x08, 1},     // TA1       - 24
    {0x08, 2},     // TA2       - 25
    {0x08, 3},     // TA3       - 26
    {0x08, 4},     // TA4       - 27
    {0x08, 5},     // TA5       - 28
    {0x08, 6},     // TA6       - 29
    {0x08, 7},     // TA7       - 30
    {0x08, 8},     // TA8       - 31
    {0x08, 9},     // TA9       - 32
    {0x08, 0x0a},  // TA10      - 33
    {0x08, 0x0b},  // TA11      - 34
    {0x08, 0x0c},  // TA12      - 35
    {0x08, 0x0d},  // TA13      - 36
    {0x08, 0x0e},  // TA14      - 37
    {0x08, 0x0f},  // TA15      - 38
    {0x0C, 0},     // TCA0      - 39
    {0x0C, 1},     // TCA1      - 40
    {0x0B, 0},     // TCC0      - 41
    {0x0B, 1},     // TCC1      - 42
    {0x0B, 2},     // TCC2      - 43
    {0x0B, 3},     // TCC3      - 44
    {0x0B, 4},     // TCC4      - 45
    {0x0B, 5},     // TCC5      - 46
    {0x0B, 6},     // TCC6      - 47
    {0x0B, 7},     // TCC7      - 48
    {0x0B, 8},     // TCC8      - 49
    {0x0B, 9},     // TCC9      - 50
    {0x0B, 0x0a},  // TCC10     - 51
    {0x0B, 0x0b},  // TCC11     - 52
    {0x0B, 0x0c},  // TCC12     - 53
    {0x0B, 0x0d},  // TCC13     - 54
    {0x0B, 0x0e},  // TCC14     - 55
    {0x0B, 0x0f},  // TCC15     - 56
    {0x09, 0},     // TD0       - 57
    {0x09, 1},     // TD1       - 58
    {0x09, 2},     // TD2       - 59
    {0x09, 3},     // TD3       - 60
    {0x09, 4},     // TD4       - 61
    {0x09, 5},     // TD5       - 62
    {0x09, 6},     // TD6       - 63
    {0x09, 7},     // TD7       - 64
    {0x09, 8},     // TD8       - 65
    {0x09, 9},     // TD9       - 66
    {0x09, 0x0a},  // TD10      - 67
    {0x09, 0x0b},  // TD11      - 68
    {0x09, 0x0c},  // TD12      - 69
    {0x09, 0x0d},  // TD13      - 70
    {0x09, 0x0e},  // TD14      - 71
    {0x09, 0x0f},  // TD15      - 72
    {0x0A, 0},     // TCP0      - 73
    {0x0A, 1},     // TCP1      - 74
    {0x0A, 2},     // TCP2      - 75
    {0x0A, 3},     // TCP3      - 76
    {0x0A, 4},     // TCP4      - 77
    {0x0A, 5},     // TCP5      - 78
    {0x0A, 6},     // TCP6      - 79
    {0x0A, 7},     // TCP7      - 80
    {0x0A, 8},     // TCP8      - 81
    {0x0A, 9},     // TCP9      - 82
    {0x0A, 0x0a},  // TCP10     - 83
    {0x0A, 0x0b},  // TCP11     - 84
    {0x0A, 0x0c},  // TCP12     - 85
    {0x0A, 0x0d},  // TCP13     - 86
    {0x0A, 0x0e},  // TCP14     - 87
    {0x0A, 0x0f},  // TCP15     - 88
    {0x0F, 0},     // GDS       - 89
    {0x02, 0},     // VGT       - 90
    {0x01, 0},     // IA        - 91
    {0x15, 0},     // MC        - 92
    {0x10, 0},     // SRBM      - 93
    {0x18, 0},     // WD        - 94
    {0x16, 0},     // CPG       - 95
    {0x17, 0},     // CPC       - 96
    {0x1A, 0},     // ATC       - 97
    {0x1B, 0},     // ATCL2     - 98
    {0x1C, 0},     // MCVML2_0  - 99
    {0x1C, 1},     // MCVML2_1  - 100
    {0x1C, 2},     // MCVML2_2  - 101
    {0x1C, 3},     // MCVML2_3  - 102
    {0x1C, 4},     // MCVML2_4  - 103
    {0x1C, 5},     // MCVML2_5  - 104
    {0x1C, 6},     // MCVML2_6  - 105
    {0x1C, 7},     // MCVML2_7  - 106
    {0x1D, 0},     // EA        - 107
    {0x1E, 0},     // RPB       - 108
    {0x1F, 0},     // RMI       - 109
}};

void PerfCounter::convertInfo() {
  switch (dev().ipLevel()) {
    case Pal::GfxIpLevel::GfxIp7:
      if (info_.blockIndex_ < ciBlockIdOrcaToPal.size()) {
        auto p = ciBlockIdOrcaToPal[info_.blockIndex_];
        info_.blockIndex_ = std::get<0>(p);
        info_.counterIndex_ = std::get<1>(p);
      }
      break;
    case Pal::GfxIpLevel::GfxIp8:
      if (info_.blockIndex_ < viBlockIdOrcaToPal.size()) {
        auto p = viBlockIdOrcaToPal[info_.blockIndex_];
        info_.blockIndex_ = std::get<0>(p);
        info_.counterIndex_ = std::get<1>(p);
      }
      break;
    case Pal::GfxIpLevel::GfxIp9:
      if (info_.blockIndex_ < gfx9BlockIdPal.size()) {
        auto p = gfx9BlockIdPal[info_.blockIndex_];
        info_.blockIndex_ = std::get<0>(p);
        info_.counterIndex_ = std::get<1>(p);
      }
      break;
    default:
      Unimplemented();
      break;
  }

  assert(info_.blockIndex_ < blockIdToIndexSelect.size());
  info_.indexSelect_ = blockIdToIndexSelect.at(info_.blockIndex_);
}

PerfCounter::~PerfCounter() {
  if (palRef_ == nullptr) {
    return;
  }

  // Release the counter reference object
  palRef_->release();
}

bool PerfCounter::create() {
  palRef_->retain();

  // Initialize the counter
  Pal::PerfCounterInfo counterInfo = {};
  counterInfo.counterType = Pal::PerfCounterType::Global;
  counterInfo.block = static_cast<Pal::GpuBlock>(info_.blockIndex_);
  counterInfo.eventId = info_.eventIndex_;

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
    } else {
      // Get here when there's no HW PerfCounter matching the counterInfo
      assert(0 && "AddCounter() failed");
    }
  }
  return true;
}

uint64_t PerfCounter::getInfo(uint64_t infoType) const {
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

}  // namespace pal
