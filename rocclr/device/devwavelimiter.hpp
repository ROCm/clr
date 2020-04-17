/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#pragma once

#include "thread/thread.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <unordered_map>

namespace amd {
  struct ProfilingCallback : public amd::HeapObject {
    virtual void callback(ulong duration, uint32_t waves) = 0;
  };
}

//! \namespace pal PAL Device Implementation
namespace device {

class WaveLimiterManager;
class Kernel;

// Adaptively limit the number of waves per SIMD based on kernel execution time
class WaveLimiter : public amd::ProfilingCallback {
 public:
  explicit WaveLimiter(WaveLimiterManager* manager, uint seqNum, bool enable, bool enableDump);
  virtual ~WaveLimiter();

  //! Get waves per shader array to be used for kernel execution.
  uint getWavesPerSH();

 protected:
  enum StateKind { WARMUP, ADAPT, RUN };

  class DataDumper {
   public:
    explicit DataDumper(const std::string& kernelName, bool enable);
    ~DataDumper();

    //! Record execution time, waves/simd and state of wave limiter.
    void addData(ulong time, uint wave, char state);

    //! Whether this data dumper is enabled.
    bool enabled() const { return enable_; }

   private:
    bool enable_;
    std::string fileName_;
    std::vector<ulong> time_;
    std::vector<uint> wavePerSIMD_;
    std::vector<char> state_;
  };

  bool enable_;
  uint SIMDPerSH_;  // Number of SIMDs per SH
  uint waves_;      // Waves per SIMD to be set
  uint bestWave_;   // Optimal waves per SIMD
  uint worstWave_;  // Wave number with the worst performance 
  uint countAll_;   // Number of kernel executions
  StateKind state_;
  WaveLimiterManager* manager_;
  DataDumper dumper_;
  std::ofstream traceStream_;
  uint32_t sampleCount_;            //!< The number of samples for adaptive mode
  uint32_t resultCount_;            //!< The number of results for adaptive mode
  uint32_t numContinuousSamples_;   //!< The number of samples with the same wave count

  static uint MaxWave;      // Maximum number of waves per SIMD
  static uint RunCount;     // Number of kernel executions for normal run
  static uint AdaptCount;   // Number of kernel executions for adapting
  const static uint MaxContinuousSamples = 2;

  //! Call back from Event::recordProfilingInfo to get execution time.
  virtual void callback(ulong duration, uint32_t waves) = 0;

  //! Output trace of measurement/adaptation.
  virtual void outputTrace() = 0;

  template <class T> void clear(T& A) {
    uint idx = 0;
    for (auto& I : A) {
      if (idx > worstWave_) {
        I = 0;
      }
      ++idx;
    }
  }
  template <class T> void output(std::ofstream& ofs, const std::string& prompt, T& A) {
    ofs << prompt;
    for (auto& I : A) {
      ofs << ' ' << static_cast<ulong>(I);
    }
  }
};

class WLAlgorithmSmooth : public WaveLimiter {
 public:
  explicit WLAlgorithmSmooth(WaveLimiterManager* manager, uint seqNum, bool enable,
                             bool enableDump);
  virtual ~WLAlgorithmSmooth();

 private:
  std::vector<uint64_t> adpMeasure_;    //!< Accumulated performance in the adaptation mode
  std::vector<uint32_t> adpSampleCnt_;  //!< The number of samples in the adaptation mode
  std::vector<uint64_t> runMeasure_;    //!< Accumulated performance in the run mode
  std::vector<uint32_t> runSampleCnt_;  //!< The number of samples in the run mode
  uint dynRunCount_;
  uint dataCount_;

  //! Update measurement data and optimal waves/simd with execution time.
  void updateData(ulong time);

  //! Clear measurement data for the next adaptation.
  void clearData();

  //! Call back from Event::recordProfilingInfo to get execution time.
  void callback(ulong duration, uint32_t waves) override;

  //! Output trace of measurement/adaptation.
  void outputTrace() override;
};

// Create wave limiter for each virtual device for a kernel and manages the wave limiters.
class WaveLimiterManager {
 public:
  explicit WaveLimiterManager(Kernel* owner, const uint simdPerSH);
  virtual ~WaveLimiterManager();

  //! Get waves per shader array for a specific virtual device.
  uint getWavesPerSH(const VirtualDevice*) const;

  //! Provide call back function for a specific virtual device.
  amd::ProfilingCallback* getProfilingCallback(const VirtualDevice*);

  //! Enable wave limiter manager by kernel metadata and flags.
  void enable(bool isSupported = true);

  //! Returns the kernel name
  const std::string& name() const;

  //! Get SimdPerSH.
  uint getSimdPerSH() const { return simdPerSH_; }

 private:
  device::Kernel* owner_;  // The kernel which owns this object
  uint simdPerSH_;         // Simd Per SH
  std::unordered_map<const VirtualDevice*, WaveLimiter*>
    limiters_;            // Maps virtual device to wave limiter
  bool enable_;           // Whether the adaptation is enabled
  bool enableDump_;       // Whether the data dumper is enabled
  uint fixed_;            // The fixed waves/simd value if not zero
  amd::Monitor monitor_;  // The mutex for updating the wave limiter map
};
}
