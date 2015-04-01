//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/gpu/gpukernel.hpp"
#include "device/gpu/gpuwavelimiter.hpp"
#include "os/os.hpp"
#include "utils/flags.hpp"

namespace gpu {

uint WaveLimiter::MaxWave;
uint WaveLimiter::WarmUpCount;
uint WaveLimiter::AdaptCount;
uint WaveLimiter::RunCount;
uint WaveLimiter::AbandonThresh;

void WaveLimiter::clearData() {
    waves_ = MaxWave;
    countAll_ = 0;
    clear(counts_);
    clear(sum_);
    clear(average_);
    clear(ratio_);
}

void WaveLimiter::enable() {
    if (waves_ > 0) {
        return;
    }
    auto gpuDev = reinterpret_cast<const Device*>(&owner_->dev());
    auto hwInfo = gpuDev->hwInfo();
    // Enable it only for SI+, unless GPU_WAVE_LIMIT_ENABLE is set to 1
    setIfNotDefault(enable_, GPU_WAVE_LIMIT_ENABLE,
         owner_->workGroupInfo()->limitWave_ && gpuDev->settings().siPlus_);
    if (!enable_) {
        return;
    }
    waves_ = MaxWave;
}

WaveLimiter::WaveLimiter(Kernel *owner) :
        owner_(owner), dumper_(owner_->name()) {
    auto gpuDev = reinterpret_cast<const Device*>(&owner_->dev());
    auto attrib = gpuDev->getAttribs();
    auto hwInfo = gpuDev->hwInfo();
    setIfNotDefault(SIMDPerSH_, GPU_WAVE_LIMIT_CU_PER_SH,
            attrib.numberOfCUsperShaderArray * hwInfo->simdPerCU_);

    state_ = WARMUP;
    dynRunCount_ = RunCount;
    auto size = MaxWave + 1;
    counts_.resize(size);
    sum_.resize(size);
    average_.resize(size);
    ratio_.resize(size);
    clearData();
    if (!flagIsDefault(GPU_WAVE_LIMIT_TRACE)) {
        traceStream_.open(std::string(GPU_WAVE_LIMIT_TRACE) + owner_->name() +
            ".txt");
    }

    MaxWave = GPU_WAVE_LIMIT_MAX_WAVE;
    WarmUpCount = GPU_WAVE_LIMIT_WARMUP;
    AdaptCount = GPU_WAVE_LIMIT_ADAPT * MaxWave;
    RunCount = GPU_WAVE_LIMIT_RUN * MaxWave;
    AbandonThresh = GPU_WAVE_LIMIT_ABANDON;

    waves_ = GPU_WAVES_PER_SIMD;
    bestWave_ = MaxWave;
    enable_ = false;
}

WaveLimiter::~WaveLimiter() {
    if (traceStream_.is_open()) {
        traceStream_.close();
    }
}

uint WaveLimiter::getWavesPerSH() const {
    return waves_ * SIMDPerSH_;
}

void WaveLimiter::updateData(ulong time) {
    sum_[waves_] += time;
    counts_[waves_]++;
    average_[waves_] = sum_[waves_] / counts_[waves_];
    ratio_[waves_] = average_[waves_] * 100 / average_[MaxWave];
    if (average_[bestWave_] > average_[waves_]) {
        bestWave_ = waves_;
    }
    outputTrace();
}

void WaveLimiter::outputTrace() {
    if (!traceStream_.is_open()) {
        return;
    }

    traceStream_ << "[WaveLimiter] " << owner_->name() << " state=" << state_
            << " waves=" << waves_ << " bestWave=" << bestWave_ << '\n';
    output(traceStream_, "\n counts = ", counts_);
    output(traceStream_, "\n sum = ", sum_);
    output(traceStream_, "\n average = ", average_);
    output(traceStream_, "\n ratio = ", ratio_);
    traceStream_ << "\n\n";
}

void WaveLimiter::callback(ulong duration) {
    dumper_.addData(duration, waves_, static_cast<char>(state_));

    if (!enable_) {
        return;
    }

    countAll_++;

    switch (state_) {
    case WARMUP:
        if (countAll_ < WarmUpCount) {
            return;
        }
        state_ = ADAPT;
        bestWave_ = MaxWave;
        clearData();
        return;
    case ADAPT:
        updateData(duration);
        if (countAll_ < AdaptCount && ratio_[waves_] < AbandonThresh) {
            waves_ = MaxWave - (countAll_ % MaxWave);
            return;
        }
        waves_ = bestWave_;
        if (countAll_ >= AdaptCount) {
            dynRunCount_ = RunCount;
        } else {
            dynRunCount_ = AdaptCount;
        }
        countAll_ = rand() % MaxWave;
        state_ = RUN;
        return;
    case RUN:
        if (countAll_ < dynRunCount_) {
            return;
        }
        state_ = ADAPT;
        bestWave_ = MaxWave;
        clearData();
        return;
    }
}

WaveLimiter::DataDumper::DataDumper(const std::string &kernelName) {
    enable_ = !flagIsDefault(GPU_WAVE_LIMIT_DUMP);
    if (enable_) {
        fileName_ = std::string(GPU_WAVE_LIMIT_DUMP) + kernelName + ".csv";
    }
}

WaveLimiter::DataDumper::~DataDumper() {
    if (!enable_) {
        return;
    }

    std::ofstream OFS(fileName_);
    for (size_t i = 0, e = time_.size(); i != e; ++i) {
        OFS << i << ',' << time_[i] << ',' << wavePerSIMD_[i] << ','
            << static_cast<uint>(state_[i]) << '\n';
    }
    OFS.close();
}

void WaveLimiter::DataDumper::addData(ulong time, uint wave, char state) {
    if (!enable_) {
        return;
    }

    time_.push_back(time);
    wavePerSIMD_.push_back(wave);
    state_.push_back(state);
}

amd::ProfilingCallback* WaveLimiter::getProfilingCallback() const {
    if (enable_ || dumper_.enabled()) {
        return const_cast<WaveLimiter*>(this);
    }
    return NULL;
}
}

