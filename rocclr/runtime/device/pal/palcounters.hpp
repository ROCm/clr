//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "top.hpp"
#include "device/device.hpp"
#include "device/pal/paldevice.hpp"
#include "palPerfExperiment.h"

namespace pal {

class VirtualGPU;

class PalCounterReference : public amd::ReferenceCountedObject
{
public:
    static PalCounterReference* Create(
        VirtualGPU&   gpu,
        const Pal::PerfExperimentCreateInfo& createInfo);

    //! Default constructor
    PalCounterReference(
        VirtualGPU&     gpu //!< Virtual GPU device object
        )
        : perfExp_(nullptr)
        , gpu_(gpu)
        , results_(nullptr) {}

    //! Get PAL counter
    Pal::IPerfExperiment* iPerf() const { return perfExp_; }

    //! Returns the virtual GPU device
    const VirtualGPU& gpu() const { return gpu_; }

    //! Increases the results array for this PAL counter(container)
    bool growResultArray(
        uint maxIndex   //!< the maximum HW counter index in the PAL counter
        );

    bool finalize();

    //! Returns the PAL counter results
    uint64_t*  results() const { return results_; }

    Pal::IPerfExperiment* perfExp_;   //!< PAL performance experiment object

protected:
    //! Default destructor
    ~PalCounterReference();

private:
    //! Disable copy constructor
    PalCounterReference(const PalCounterReference&);

    //! Disable operator=
    PalCounterReference& operator=(const PalCounterReference&);

    VirtualGPU&     gpu_;           //!< The virtual GPU device object
    uint64_t*       results_;       //!< Counter results
    Pal::IGpuMemory*    pGpuMemory;
    void*               pCpuAddr;
};

//! Performance counter implementation on GPU
class PerfCounter : public device::PerfCounter
{
public:
    //! The performance counter info
    struct Info : public amd::EmbeddedObject
    {
        uint        blockIndex_;    //!< Index of the block to configure
        uint        counterIndex_;  //!< Index of the hardware counter
        uint        eventIndex_;    //!< Event you wish to count with the counter
    };

    //! The PerfCounter flags
    enum Flags
    {
        BeginIssued     = 0x00000001,
        EndIssued       = 0x00000002,
        ResultReady     = 0x00000004
    };

    //! Constructor for the GPU PerfCounter object
    PerfCounter(
        const Device&       device,         //!< A GPU device object
        const VirtualGPU&   gpu,            //!< Virtual GPU device object
        cl_uint             blockIndex,     //!< HW block index
        cl_uint             counterIndex,   //!< Counter index within the block
        cl_uint             eventIndex)     //!< Event index for profiling
        : gpuDevice_(device)
        , gpu_(gpu)
        , calRef_(NULL)
        , flags_(0)
        , counter_(0)
        , index_(0)
    {
        info_.blockIndex_   = blockIndex;
        info_.counterIndex_ = counterIndex;
        info_.eventIndex_   = eventIndex;
    }

    //! Destructor for the GPU PerfCounter object
    virtual ~PerfCounter();

    //! Creates the current object
    bool create(
        PalCounterReference* calRef     //!< Reference counter
        );

    //! Returns the specific information about the counter
    uint64_t getInfo(
        uint64_t infoType   //!< The type of returned information
        ) const;

    //! Returns the GPU device, associated with the current object
    const Device& dev() const { return gpuDevice_; }

    //! Returns the virtual GPU device
    const VirtualGPU& gpu() const { return gpu_; }

    //! Returns the CAL performance counter descriptor
    const Info* info() const { return &info_; }

    //! Returns the Info structure for performance counter
    Pal::IPerfExperiment* iPerf() const { return counter_; }

private:
    //! Disable default copy constructor
    PerfCounter(const PerfCounter&);

    //! Disable default operator=
    PerfCounter& operator=(const PerfCounter&);

    const Device&   gpuDevice_; //!< The backend device
    const VirtualGPU&   gpu_;   //!< The virtual GPU device object

    PalCounterReference* calRef_;   //!< Reference counter
    uint                flags_; //!< The perfcounter object state
    Info                info_;  //!< The info structure for perfcounter
    Pal::IPerfExperiment*    counter_;   //!< GSL counter object
    uint                index_; //!< Counter index in the CAL container
};

} // namespace pal


