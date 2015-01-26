//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef GPUSETTINGS_HPP_
#define GPUSETTINGS_HPP_

#include "top.hpp"
#include "library.hpp"

/*! \addtogroup GPU GPU Resource Implementation
 *  @{
 */

//! GPU Device Implementation
namespace gpu {

//! Device settings
class Settings : public device::Settings
{
public:
    //! Debug GPU flags
    enum DebugGpuFlags
    {
        CheckForILSource        = 0x00000001,
        StubCLPrograms          = 0x00000002,   //!< Enables OpenCL programs stubbing
        LockGlobalMemory        = 0x00000004,
    };

    enum BlitEngineType
    {
        BlitEngineDefault       = 0x00000000,
        BlitEngineHost          = 0x00000001,
        BlitEngineCAL           = 0x00000002,
        BlitEngineKernel        = 0x00000003,
    };

    enum HostMemFlags
    {
        HostMemDisable          = 0x00000000,
        HostMemBuffer           = 0x00000001,
        HostMemImage            = 0x00000002,
    };

    union {
        struct {
            uint    singleHeap_: 1;         //!< Device will use a preallocated heap
            uint    remoteAlloc_: 1;        //!< Allocate remote memory for the heap
            uint    stagedXferRead_: 1;     //!< Uses a staged buffer read
            uint    stagedXferWrite_: 1;    //!< Uses a staged buffer write
            uint    disablePersistent_: 1;  //!< Disables using persistent memory for staging
            uint    useAliases_: 1;         //!< Enables global heap aliases in HW
            uint    imageSupport_: 1;       //!< Report images support
            uint    doublePrecision_: 1;    //!< Enables double precision support
            uint    openVideo_: 1;      //!< Open Video interop support
            uint    reportFMAF_: 1;     //!< Report FP_FAST_FMAF define in CL program
            uint    reportFMA_: 1;      //!< Report FP_FAST_FMA define in CL program
            uint    use64BitPtr_: 1;    //!< Use 64bit pointers on GPU
            uint    force32BitOcl20_: 1;    //!< Force 32bit apps to take CLANG/HSAIL path on GPU
            uint    imageDMA_: 1;       //!< Enable direct image DMA transfers
            uint    syncObject_: 1;     //!< Enable syncobject
            uint    siPlus_: 1;         //!< SI and post SI features
            uint    ciPlus_: 1;         //!< CI and post CI features
            uint    viPlus_: 1;         //!< VI and post VI features
            uint    rectLinearDMA_: 1;  //!< Rectangular linear DRMDMA support
            uint    threadTraceEnable_: 1;  //!< Thread trace enable
            uint    linearPersistentImage_: 1;  //!< Allocates linear images in persistent
            uint    useSingleScratch_: 1;   //!< Allocates single scratch per device
            uint    sdmaProfiling_: 1;  //!< Enables SDMA profiling
            uint    hsail_: 1;          //!< Enables HSAIL compilation
            uint    stagingWritePersistent_: 1; //!< Enables persistent writes
            uint    svmAtomics_: 1;     //!< SVM device atomics
            uint    svmFineGrainSystem_: 1;     //!< SVM fine grain system support
            uint    apuSystem_: 1;      //!< Device is APU system with shared memory
            uint    asyncMemCopy_: 1;   //!< Use async memory transfers
            uint    hsailDirectSRD_: 1; //!< Controls direct SRD for HSAIL
            uint    useDeviceQueue_: 1; //!< Submit to separate device queue
            uint    reserved_: 1;
        };
        uint    value_;
    };

    uint    oclVersion_;        //!< Reported OpenCL version support
    uint    debugFlags_;        //!< Debug GPU flags
    size_t  stagedXferSize_;    //!< Staged buffer size
    uint    maxRenames_;        //!< Maximum number of possible renames
    uint    maxRenameSize_;     //!< Maximum size for all renames
    size_t  heapSize_;          //!< The global heap size
    size_t  heapSizeGrowth_;    //!< The global heap size growth
    uint    hwLDSSize_;         //!< HW local data store size
    uint    maxWorkGroupSize_;  //!< Requested workgroup size for this device
    uint    hostMemDirectAccess_;   //!< Enables direct access to the host memory
    amd::LibrarySelector libSelector_; //!< Select linking libraries for compiler
    uint    workloadSplitSize_; //!< Workload split size
    uint    minWorkloadTime_;   //!< Minimal workload time in 0.1 ms
    uint    maxWorkloadTime_;   //!< Maximum workload time in 0.1 ms
    uint    blitEngine_;        //!< Blit engine type
    size_t  pinnedXferSize_;    //!< Pinned buffer size for transfer
    size_t  pinnedMinXferSize_; //!< Minimal buffer size for pinned transfer
    size_t  resourceCacheSize_; //!< Resource cache size in MB
    uint64_t    maxAllocSize_;  //!< Maximum single allocation size
    size_t  numMemDependencies_;//!< The array size for memory dependencies tracking
    uint    cacheLineSize_;     //!< Cache line size in bytes
    uint    cacheSize_;         //!< L1 cache size in bytes
    size_t  xferBufSize_;       //!< Transfer buffer size for image copy optimization
    uint    numComputeRings_;   //!< 0 - disabled, 1 , 2,.. - the number of compute rings
    uint    numDeviceEvents_;   //!< The number of device events
    uint    numWaitEvents_;     //!< The number of wait events for device enqueue


    //! Default constructor
    Settings();

    //! Creates settings
    bool create(
        const CALdeviceattribs& calAttr //!< CAL attributes structure
#if cl_amd_open_video
      , const CALdeviceVideoAttribs& calVideoAttr //!< CAL video attributes
#endif // cl_amd_open_video
      , bool reportAsOCL12Device = false //!< Report As OpenCL1.2 Device
        );

private:
    //! Disable copy constructor
    Settings(const Settings&);

    //! Disable assignment
    Settings& operator=(const Settings&);

    //! Overrides current settings based on registry/environment
    void override();
};

/*@}*/} // namespace gpu

#endif /*GPUSETTINGS_HPP_*/
