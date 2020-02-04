/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "platform/command.hpp"
#include "device/devkernel.hpp"
#include "device/devwavelimiter.hpp"
#include "os/os.hpp"
#include "utils/flags.hpp"

#include <cstdlib>
using namespace std;

namespace device {

uint WaveLimiter::MaxWave;
uint WaveLimiter::RunCount;
uint WaveLimiter::AdaptCount;

// ================================================================================================
WaveLimiter::WaveLimiter(WaveLimiterManager* manager, uint seqNum, bool enable, bool enableDump)
    : manager_(manager), dumper_(manager_->name() + "_" + std::to_string(seqNum), enableDump) {
  setIfNotDefault(SIMDPerSH_, GPU_WAVE_LIMIT_CU_PER_SH, manager->getSimdPerSH());
  MaxWave = GPU_WAVE_LIMIT_MAX_WAVE;
  RunCount = GPU_WAVE_LIMIT_RUN * MaxWave;
  AdaptCount = MaxContinuousSamples * 2 * (MaxWave + 1);

  state_ = WARMUP;
  if (!flagIsDefault(GPU_WAVE_LIMIT_TRACE)) {
    traceStream_.open(std::string(GPU_WAVE_LIMIT_TRACE) + manager_->name() + ".txt");
  }

  waves_ = MaxWave;
  enable_ = (SIMDPerSH_ == 0) ? false : enable;
  bestWave_ = (enable_) ? MaxWave : 0;
  worstWave_ = 0;
  sampleCount_ = 0;
  resultCount_ = 0;
  numContinuousSamples_ = 0;
}

// ================================================================================================
WaveLimiter::~WaveLimiter() {
  if (traceStream_.is_open()) {
    traceStream_.close();
  }
}

// ================================================================================================
uint WaveLimiter::getWavesPerSH() {
  // Generate different wave counts in the adaptation mode
  if ((state_ == ADAPT) && (sampleCount_ < AdaptCount)) {
    if (numContinuousSamples_ == 0) {
        ++waves_;
        waves_ %= MaxWave + 1;
        // Don't execute the wave count with the worst performance
        if (waves_ != 0) {
          while (worstWave_ >= waves_) {
            ++waves_;
            waves_ %= MaxWave + 1;
          }
        }
    }
    ++numContinuousSamples_;
    numContinuousSamples_ %= MaxContinuousSamples;
    ++sampleCount_;
  }
  else {
    waves_ = bestWave_;
  }
  return waves_ * SIMDPerSH_;
}

// ================================================================================================
WLAlgorithmSmooth::WLAlgorithmSmooth(WaveLimiterManager* manager, uint seqNum, bool enable,
                                     bool enableDump)
    : WaveLimiter(manager, seqNum, enable, enableDump) {
  dynRunCount_ = RunCount;
  adpMeasure_.resize(MaxWave + 1);
  adpSampleCnt_.resize(MaxWave + 1);
  runMeasure_.resize(MaxWave + 1);
  runSampleCnt_.resize(MaxWave + 1);

  clearData();
}

// ================================================================================================
WLAlgorithmSmooth::~WLAlgorithmSmooth() {}

// ================================================================================================
void WLAlgorithmSmooth::clearData() {
  waves_ = MaxWave;
  countAll_ = 0;
  clear(adpMeasure_);
  clear(adpSampleCnt_);
  dataCount_ = 0;
}

// ================================================================================================
void WLAlgorithmSmooth::updateData(ulong time) {
}

// ================================================================================================
void WLAlgorithmSmooth::outputTrace() {
  if (!traceStream_.is_open()) {
    return;
  }

  traceStream_ << "[WaveLimiter] " << manager_->name() << " state=" << state_ <<
    " waves=" << waves_ << " bestWave=" << bestWave_ << " worstWave=" << worstWave_ << '\n';
  output(traceStream_, "\n adaptive measure = ", adpMeasure_);
  output(traceStream_, "\n adaptive smaple count = ", adpSampleCnt_);
  output(traceStream_, "\n run measure = ", runMeasure_);
  output(traceStream_, "\n run smaple count = ", runSampleCnt_);
  traceStream_ << "\n % time from the previous runs to the best wave: ";
  float min = static_cast<float>(adpMeasure_[bestWave_]) / adpSampleCnt_[bestWave_];
  for (uint i = 0; i < (MaxWave + 1); ++i) {
    runSampleCnt_[i] = (runSampleCnt_[i] == 0) ? 1 : runSampleCnt_[i];
    float average = static_cast<float>(runMeasure_[i]) / runSampleCnt_[i];
    traceStream_ << (average * 100 / min) << " ";
  }
  traceStream_ << "\n run count = " << dynRunCount_;
  traceStream_ << "\n\n";
}

// ================================================================================================
void WLAlgorithmSmooth::callback(ulong duration, uint32_t waves) {
  dumper_.addData(duration, waves, static_cast<char>(state_));

  if (!enable_ || (duration == 0)) {
    return;
  }

  countAll_++;

  waves /= SIMDPerSH_;
  // Collect the time for the current wave count
  runMeasure_[waves] += duration;
  runSampleCnt_[waves]++;

  switch (state_) {
    case ADAPT:
      assert(duration > 0);
      // Wave count 0 indicates the satrt of adaptation
      if ((waves == 0) || (resultCount_ > 0)) {
        // Scale time to us
        adpMeasure_[waves] += duration;
        adpSampleCnt_[waves]++;
        resultCount_++;
        // If the end of adaptation is reached, then analyze the results
        if (resultCount_ == AdaptCount) {
          // Reset the counters
          resultCount_ = sampleCount_ = 0;
          float min = std::numeric_limits<float>::max();
          float max = std::numeric_limits<float>::min();
          uint32_t best = bestWave_;
          // Check performance for the previous run if it's available
          if (runSampleCnt_[bestWave_] > 0) {
            min = static_cast<float>(runMeasure_[bestWave_]) / runSampleCnt_[bestWave_];
          }
          else if (adpSampleCnt_[MaxWave] > 0) {
            min = static_cast<float>(adpMeasure_[MaxWave]) / adpSampleCnt_[MaxWave];
            bestWave_ = MaxWave;
          }
          // Find the fastest average time
          float reference = min;
          for (uint i = MaxWave; i > 0; --i) {
            float average;
            if (adpSampleCnt_[i] > 0) {
              average = static_cast<float>(adpMeasure_[i]) / adpSampleCnt_[i];
            }
            else {
              average = 0.0f;
            }
            // More waves have 5% advantage over the lower number
            if (average * 1.05f < min) {
              min = average;
              bestWave_ = i;
            }
            if (average > max) {
              max = average;
              worstWave_ = i;
            }
          }
          // Check for 5% acceptance
          if ((min * 1.05f > reference) || (bestWave_ == best)) {
            bestWave_ = best;
            // Increase the run time if the same wave count is the best
            dynRunCount_ += RunCount;
            dynRunCount_++;
          }
          else {
            dynRunCount_ = RunCount;
          }
          // Find the middle between the best and the worst
          if (worstWave_ < bestWave_) {
            worstWave_ += ((bestWave_ - worstWave_) >> 1);
          } else {
            worstWave_ = 0;
          }
          state_ = RUN;
          outputTrace();
          // Start to collect the new data for the best wave
          countAll_ = 0;
          runMeasure_[bestWave_] = 0;
          runSampleCnt_[bestWave_] = 0;
        }
      }
      return;
    case WARMUP:
    case RUN:
      if (countAll_ < dynRunCount_) {
        return;
      }
      if (state_ == WARMUP) {
        runSampleCnt_[bestWave_] = 0;
      }
      state_ = ADAPT;
      clearData();
      return;
  }
}

// ================================================================================================
WaveLimiter::DataDumper::DataDumper(const std::string& kernelName, bool enable) {
  enable_ = enable;
  if (enable_) {
    fileName_ = std::string(GPU_WAVE_LIMIT_DUMP) + kernelName + ".csv";
  }
}

// ================================================================================================
WaveLimiter::DataDumper::~DataDumper() {
  if (!enable_) {
    return;
  }

  std::ofstream OFS(fileName_);
  for (size_t i = 0, e = time_.size(); i != e; ++i) {
    OFS << i << ',' << time_[i] << ',' << wavePerSIMD_[i] << ',' << static_cast<uint>(state_[i])
        << '\n';
  }
  OFS.close();
}

// ================================================================================================
void WaveLimiter::DataDumper::addData(ulong time, uint wave, char state) {
  if (!enable_) {
    return;
  }

  time_.push_back(time);
  wavePerSIMD_.push_back(wave);
  state_.push_back(state);
}

// ================================================================================================
WaveLimiterManager::WaveLimiterManager(device::Kernel* kernel, const uint simdPerSH)
    : owner_(kernel), enable_(false), enableDump_(!flagIsDefault(GPU_WAVE_LIMIT_DUMP)) {
  setIfNotDefault(simdPerSH_, GPU_WAVE_LIMIT_CU_PER_SH, ((simdPerSH == 0) ? 1 : simdPerSH));
  fixed_ = GPU_WAVES_PER_SIMD * simdPerSH_;
}

// ================================================================================================
WaveLimiterManager::~WaveLimiterManager() {
  for (auto& I : limiters_) {
    delete I.second;
  }
}

// ================================================================================================
const std::string& WaveLimiterManager::name() const { return owner_->name(); }

// ================================================================================================
uint WaveLimiterManager::getWavesPerSH(const device::VirtualDevice* vdev) const {
  if (fixed_ > 0) {
    return fixed_;
  }
  if (!enable_) {
    return 0;
  }
  auto loc = limiters_.find(vdev);
  if (loc == limiters_.end()) {
    return 0;
  }
  assert(loc->second != nullptr);
  return loc->second->getWavesPerSH();
}

amd::ProfilingCallback* WaveLimiterManager::getProfilingCallback(
    const device::VirtualDevice* vdev) {
  assert(vdev != nullptr);
  if (!enable_ && !enableDump_) {
    return nullptr;
  }

  amd::ScopedLock SL(monitor_);
  auto loc = limiters_.find(vdev);
  if (loc != limiters_.end()) {
    return loc->second;
  }

  auto limiter = new WLAlgorithmSmooth(this, limiters_.size(), enable_, enableDump_);
  if (limiter == nullptr) {
    enable_ = false;
    return nullptr;
  }
  limiters_[vdev] = limiter;
  return limiter;
}

// ================================================================================================
void WaveLimiterManager::enable(bool isSupported) {
  if (fixed_ > 0) {
    return;
  }

  // Enable it only for CI+, unless GPU_WAVE_LIMIT_ENABLE is set to 1
  // Disabled for SI due to bug #10817
  if (!flagIsDefault(GPU_WAVE_LIMIT_ENABLE)) {
    enable_ = GPU_WAVE_LIMIT_ENABLE;
  } else if (isSupported) {
    if (owner_->workGroupInfo()->wavesPerSimdHint_ == 0) {
      enable_ = true;
    } else if (owner_->workGroupInfo()->wavesPerSimdHint_ <= GPU_WAVE_LIMIT_MAX_WAVE) {
      fixed_ = owner_->workGroupInfo()->wavesPerSimdHint_ * getSimdPerSH();
    }
  }
}

}  // namespace pal
