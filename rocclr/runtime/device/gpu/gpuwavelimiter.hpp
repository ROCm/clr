//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef GPUWAVELIMITER_HPP_
#define GPUWAVELIMITER_HPP_

#include "platform/command.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>

//! \namespace gpu GPU Device Implementation
namespace gpu {

class Kernel;

// Adaptively limit the number of waves per SIMD based on kernel execution time
class WaveLimiter: public amd::ProfilingCallback {
public:
    explicit WaveLimiter(Kernel*);
    ~WaveLimiter();
    uint getWavesPerSH() const;
    amd::ProfilingCallback* getProfilingCallback() const;
    void enable();

private:
    enum StateKind {
        WARMUP, ADAPT, RUN
    };

    class DataDumper {
    public:
        explicit DataDumper(const std::string &kernelName);
        ~DataDumper();
        void addData(ulong time, uint wave, char state);
        bool enabled() const { return enable_;}
    private:
        bool enable_;
        std::string fileName_;
        std::vector<ulong> time_;
        std::vector<uint> wavePerSIMD_;
        std::vector<char> state_;
    };

    std::vector<ulong> measure_;
    std::vector<ulong> reference_;
    std::vector<ulong> trial_;
    std::vector<ulong> ratio_;
    bool discontinuous_; // Measured data is discontinuous

    bool enable_;
    uint SIMDPerSH_;     // Number of SIMDs per SH
    uint waves_;         // waves_ per SIMD
    uint bestWave_;      // Optimal waves per SIMD
    uint countAll_;      // Number of kernel executions
    uint dynRunCount_;
    StateKind state_;
    Kernel *owner_;
    DataDumper dumper_;
    std::ofstream traceStream_;
    mutable bool waveSet_;
    uint dataCount_;

    static uint MaxWave;       // Maximum number of waves per SIMD
    static uint WarmUpCount;   // Number of kernel executions for warm up
    static uint AdaptCount;    // Number of kernel executions for adapting
    static uint RunCount;      // Number of kernel executions for normal run
    static uint AbandonThresh; // Threshold to abandon adaptation
    static uint DscThresh;     // Threshold for identifying discontinuities

    virtual void callback(ulong duration);
    void updateData(ulong time);
    void outputTrace();
    void clearData();

    template<class T> void clear(T& A) {
        for (auto &I : A) {
            I = 0;
        }
    }
    template<class T> void output(std::ofstream &ofs, const std::string &prompt,
            T& A) {
        ofs << prompt;
        for (auto &I : A) {
            ofs << ' ' << static_cast<ulong>(I);
        }
    }
};
}
#endif
