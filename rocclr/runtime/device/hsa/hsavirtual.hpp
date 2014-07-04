//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef HSAVIRTUAL_HPP_
#define HSAVIRTUAL_HPP_
#include "hsadevice.hpp"
#include "services.h"
#include "utils/util.hpp"

namespace oclhsa {
class Device;

// Timestamp for keeping track of some profiling information for various events
// including EnqueueNDRangeKernel and clEnqueueCopyBuffer.
class Timestamp {
private:
    HsaSignal signal_;
    uint64_t  start_;
    uint64_t  end_;

public:
    // get-ers
    uint64_t getStart() const { return start_; }
    uint64_t getEnd() const { return end_; }
    HsaSignal getSignal() const { return signal_; }

    // Default constructor
    Timestamp()
        : signal_(0),
          start_(0),
          end_(0) {}

    // Deconstructor, which will delete the signal if we created one
    ~Timestamp();

    // Creates a signal for the timestamp, saves it, and returns it
    HsaSignal createSignal();

    // Start a timestamp (get timestamp from OS)
    void start();

    // End a timestamp (get timestamp from OS)
    void end();
};

class VirtualGPU : public device::VirtualDevice {
public:
    VirtualGPU(Device &device);
    ~VirtualGPU();

    bool create(HsaQueueType queueType);
    bool terminate();

    void profilingBegin(amd::Command &command, bool drmProfiling = false);
    const Device& dev() const { return oclhsa_device_; }
    //! End the command profiling
    void profilingEnd(amd::Command &command);

    //! Collect the profiling results
    bool profilingCollectResults(
        amd::Command* list  //!< List of all commands in the batch.
        );
    void submitReadMemory(amd::ReadMemoryCommand& cmd);
    void submitWriteMemory(amd::WriteMemoryCommand& cmd);
    void submitCopyMemory(amd::CopyMemoryCommand& cmd);
    void submitMapMemory(amd::MapMemoryCommand& cmd);
    void submitUnmapMemory(amd::UnmapMemoryCommand& cmd);
    void submitKernel(amd::NDRangeKernelCommand& cmd);
    bool submitKernelInternal(
        const amd::NDRangeContainer& sizes, //!< Workload sizes
        const amd::Kernel&  kernel,         //!< Kernel for execution
        const_address parameters,           //!< Parameters for the kernel
        void *event_handle                  //!< Handle to OCL event for debugging
        );
    void submitNativeFn(amd::NativeFnCommand& cmd);
    void submitMarker(amd::Marker& cmd);
    void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd);
    void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd);
    void submitPerfCounter(amd::PerfCounterCommand& cmd);
    void flush(amd::Command* list = NULL, bool wait = false);
    void submitFillMemory(amd::FillMemoryCommand& cmd);
    void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);

// { oclhsa OpenCL integration
// Added these stub (no-ops) implementation of pure virtual methods,
// when integrating HSA and OpenCL branches.
// TODO: After inegration, whoever is working on VirtualGPU should write
// actual implemention.
#if cl_amd_open_video
    virtual void submitRunVideoProgram(amd::RunVideoProgramCommand &cmd) {}
    virtual void submitSetVideoSession(amd::SetVideoSessionCommand &cmd) {}
#endif  // cl_amd_open_video
    virtual void submitSignal(amd::SignalCommand &cmd) {}
    virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand &cmd) {}
    virtual void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
    virtual void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
    virtual void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
    virtual void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
    virtual void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);
    void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand &cmd) {}
    void submitThreadTrace(amd::ThreadTraceCommand &vcmd) {}

    /**
     * @brief Waits on an outstanding kernel without regard to how
     * it was dispatched - with or without a signal
     *
     * @return bool true if Wait returned successfully, false
     * otherwise
     */
    bool releaseGpuMemoryFence();
// } oclhsa OpenCL integration
private:
    /**
     * @brief Retrieves the various configuration parameters that could
     * be used to execute a kernel - Enable Profiling, Sizes of Global,
     * Local work spaces, offsets for global Id, etc.
     *
     * @note: The implementation currently does not verify if the input
     * parameters for global, local and offset arrays are valid. For
     * example, it assumes that the values that are passed in conform to
     * openCL properties such as: CL_DEVICE_MAX_WORK_ITEM_SIZES,
     * CL_DEVICE_MAX_WORK_GROUP_SIZE, etc
     *
     * @param lds_size The amount of LDS memory used in the kernel.
     *
     * @param profile_enable Flag to enable kernel profiling.
     *
     * @param config Output parameter updated with various execution
     * policy paramters.
     *
     * @param sizes The work item and work group size.
     *
     * @return HsaStatus ::kHsaStatusSuccess or ::kHsaStatusError
     */
    HsaStatus getDispatchConfig(
        uint32_t lds_size,
        bool profile_enable,
        HsaDispatchConfig* config,
        const amd::NDRangeContainer& sizes,
        const amd::Kernel& kernel);

    /**
     * @brief Synchronize kernel submits across different queue types
     * i.e. a submit to compute kernel should determine that there is no
     * outstanding kernel to another queue type, e.g. interop queue.
     * The same applies for submits to interop queues or queues of
     * another type.
     *
     * @param dispatch_queue Queue object into which the current kernel
     * would be submitted.
     *
     * @return HsaStatus ::kHsaStatusSuccess or ::kHsaStatusError
     */
    HsaStatus synchronizeInterQueueKernels(HsaQueue *dispatchQueue);

    /**
     * @brief Maintains the list of memory blocks allocated
     * for one or more kernel submissions
     */
    std::vector<void *> kernelArgList_;

    /**
     * @brief Indicates if a kernel dispatch is outstanding. This flag is
     * used to synchronized on kernel outputs.
     */
    bool hasPendingDispatch_;

    /**
     * @brief Maintains the queue type of the last kernel submit.
     * Submission of kernels across queue types must be coordinated
     * i.e. all outstanding kernels on one queue type must be finished
     * before kernels can be submitted onto a different queue type.
     */
    HsaQueueType lastSubmitQueue_;

    Timestamp*    timestamp_;
    HsaDevice*    gpu_device_;      //!< Physical device
    HsaQueue*     gpu_queue_;       //!< Queue associated with a gpu
    HsaQueue*     interopQueue_;    //!< Interop queue associated with a gpu
    uint32_t      dispatch_id_;     //!< This variable must be updated atomically.
    Device&       oclhsa_device_;   //!< oclhsa device object
};
}
#endif
