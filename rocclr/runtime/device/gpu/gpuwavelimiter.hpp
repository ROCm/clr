//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef GPUWAVELIMITER_HPP_
#define GPUWAVELIMITER_HPP_

#include "platform/command.hpp"
#include "thread/thread.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <unordered_map>

//! \namespace gpu GPU Device Implementation
namespace gpu {

class Kernel;

// Adaptively limit the number of waves per SIMD based on kernel execution time
class WaveLimiter: public amd::ProfilingCallback {
public:
    explicit WaveLimiter(Kernel*, uint seqNum, bool enable, bool enableDump);
    virtual ~WaveLimiter();

    //! Get waves per shader array to be used for kernel execution.
    uint getWavesPerSH();

protected:
    enum StateKind {
        WARMUP, ADAPT, RUN
    };

    class DataDumper {
    public:
        explicit DataDumper(const std::string &kernelName, bool enable);
        ~DataDumper();

        //! Record execution time, waves/simd and state of wave limiter.
        void addData(ulong time, uint wave, char state);

        //! Whether this data dumper is enabled.
        bool enabled() const { return enable_;}
    private:
        bool enable_;
        std::string fileName_;
        std::vector<ulong> time_;
        std::vector<uint> wavePerSIMD_;
        std::vector<char> state_;
    };

    std::vector<ulong> measure_;
    bool enable_;
    uint SIMDPerSH_;     // Number of SIMDs per SH
    uint waves_;         // Waves per SIMD to be set
    uint bestWave_;      // Optimal waves per SIMD
    uint countAll_;      // Number of kernel executions
    StateKind state_;
    Kernel *owner_;
    DataDumper dumper_;
    std::ofstream traceStream_;
    uint currWaves_;     // Current waves per SIMD

    static uint MaxWave;       // Maximum number of waves per SIMD
    static uint WarmUpCount;   // Number of kernel executions for warm up
    static uint RunCount;      // Number of kernel executions for normal run

    //! Call back from Event::recordProfilingInfo to get execution time.
    virtual void callback(ulong duration)=0;

    //! Output trace of measurement/adaptation.
    virtual void outputTrace()=0;

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

class WLAlgorithmSmooth: public WaveLimiter {
public:
    explicit WLAlgorithmSmooth(Kernel* owner, uint seqNum, bool enable, bool enableDump);
    virtual ~WLAlgorithmSmooth();
private:
    std::vector<ulong> reference_;
    std::vector<ulong> trial_;
    std::vector<ulong> ratio_;
    bool discontinuous_; // Measured data is discontinuous
    uint dynRunCount_;
    uint dataCount_;

    static uint AdaptCount;    // Number of kernel executions for adapting
    static uint AbandonThresh; // Threshold to abandon adaptation
    static uint DscThresh;     // Threshold for identifying discontinuities

    //! Update measurement data and optimal waves/simd with execution time.
    void updateData(ulong time);

    //! Clear measurement data for the next adaptation.
    void clearData();

    //! Call back from Event::recordProfilingInfo to get execution time.
    void callback(ulong duration);

    //! Output trace of measurement/adaptation.
    void outputTrace();
};

class WLAlgorithmAvrg: public WaveLimiter {
public:
    explicit WLAlgorithmAvrg(Kernel* owner, uint seqNum, bool enable, bool enableDump);
    virtual ~WLAlgorithmAvrg();
private:
    //! Call back from Event::recordProfilingInfo to get execution time.
    void callback(ulong duration);

    //! Output trace of measurement/adaptation.
    void outputTrace();
};

// Create wave limiter for each virtual device for a kernel and manages the wave limiters.
class WaveLimiterManager {
public:
    explicit WaveLimiterManager(Kernel* owner);
    virtual ~WaveLimiterManager();

    //! Get waves per shader array for a specific virtual device.
    uint getWavesPerSH(const device::VirtualDevice *) const;

    //! Provide call back function for a specific virtual device.
    amd::ProfilingCallback* getProfilingCallback(const device::VirtualDevice *);

    //! Enable wave limiter manager by kernel metadata and flags.
    void enable();
private:
    Kernel *owner_;                // The kernel which owns this object
    std::unordered_map<const device::VirtualDevice *,
        WaveLimiter*> limiters_;   // Maps virtual device to wave limiter
    bool enable_;                  // Whether the adaptation is enabled
    bool enableDump_;              // Whether the data dumper is enabled
    uint fixed_;                   // The fixed waves/simd value if not zero
    amd::Monitor monitor_;         // The mutex for updating the wave limiter map
};
}
#endif
