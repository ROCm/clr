//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef HSACOUNTERS_HPP_
#define HSACOUNTERS_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "device/hsa/hsadevice.hpp"

namespace oclhsa {

class VirtualGPU;

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
        const HsaDevice     *device,         //!< A GPU device object
        const VirtualGPU&   gpu,            //!< Virtual GPU device object
        cl_uint             blockIndex,     //!< HW block index
        cl_uint             counterIndex,   //!< Counter index within the block
        cl_uint             eventIndex)     //!< Event index for profiling
        : gpuDevice_(device)
        , gpu_(gpu)
        , hsaPmu_(NULL)
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
 
    //! Creates the counter object
    bool create(
        HsaPmu hsaPmu     //!< Reference counter
        );
 
    //! Returns the specific information about the counter
    uint64_t getInfo(
        uint64_t infoType   //!< The type of returned information
        ) const;
 
    //! Returns the GPU device, associated with the current object
    const HsaDevice * dev() const { return gpuDevice_; }
 
    //! Returns the virtual GPU device
    const VirtualGPU& gpu() const { return gpu_; }
 
    //! Returns the CAL performance counter descriptor
    const Info* info() const { return &info_; }
 
    //! Returns the Info structure for performance counter
    HsaPmu getCounterPmu() const { return hsaPmu_; }

private:
    //! Disable default copy constructor
    PerfCounter(const PerfCounter&);
 
    //! Disable default operator=
    PerfCounter& operator=(const PerfCounter&);
 
    //! Get enabled counter number
    bool getEnabledCounterNum(uint32_t &counter_num);

    const HsaDevice     *gpuDevice_; //!< The backend device
    const VirtualGPU&   gpu_;   //!< The virtual GPU device object
 
    HsaPmu              hsaPmu_;   //!< Hsa pmu
    uint                flags_; //!< The perfcounter object state
    Info                info_;  //!< The info structure for perfcounter
    HsaCounter          counter_;   //!< HSA counter object
    HsaCounterBlock     counter_block_; //!< counter block that the counter belongs to
    uint                index_; //!< Counter index in the CAL container
};

} // namespace oclhsa

#endif // HSACOUNTERS_HPP_

