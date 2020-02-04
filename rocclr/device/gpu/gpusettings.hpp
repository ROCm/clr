/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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
class Settings : public device::Settings {
 public:
  //! Debug GPU flags
  enum DebugGpuFlags {
    CheckForILSource = 0x00000001,
    StubCLPrograms = 0x00000002,  //!< Enables OpenCL programs stubbing
    LockGlobalMemory = 0x00000004,
  };

  enum BlitEngineType {
    BlitEngineDefault = 0x00000000,
    BlitEngineHost = 0x00000001,
    BlitEngineCAL = 0x00000002,
    BlitEngineKernel = 0x00000003,
  };

  enum HostMemFlags {
    HostMemDisable = 0x00000000,
    HostMemBuffer = 0x00000001,
    HostMemImage = 0x00000002,
  };

  union {
    struct {
      uint remoteAlloc_ : 1;             //!< Allocate remote memory for the heap
      uint stagedXferRead_ : 1;          //!< Uses a staged buffer read
      uint stagedXferWrite_ : 1;         //!< Uses a staged buffer write
      uint disablePersistent_ : 1;       //!< Disables using persistent memory for staging
      uint imageSupport_ : 1;            //!< Report images support
      uint doublePrecision_ : 1;         //!< Enables double precision support
      uint use64BitPtr_ : 1;             //!< Use 64bit pointers on GPU
      uint force32BitOcl20_ : 1;         //!< Force 32bit apps to take CLANG/HSAIL path on GPU
      uint imageDMA_ : 1;                //!< Enable direct image DMA transfers
      uint syncObject_ : 1;              //!< Enable syncobject
      uint ciPlus_ : 1;                  //!< CI and post CI features
      uint viPlus_ : 1;                  //!< VI and post VI features
      uint aiPlus_ : 1;                  //!< AI and post AI features
      uint threadTraceEnable_ : 1;       //!< Thread trace enable
      uint linearPersistentImage_ : 1;   //!< Allocates linear images in persistent
      uint useSingleScratch_ : 1;        //!< Allocates single scratch per device
      uint sdmaProfiling_ : 1;           //!< Enables SDMA profiling
      uint hsail_ : 1;                   //!< Enables HSAIL compilation
      uint svmAtomics_ : 1;              //!< SVM device atomics
      uint svmFineGrainSystem_ : 1;      //!< SVM fine grain system support
      uint useDeviceQueue_ : 1;          //!< Submit to separate device queue
      uint reserved_ : 11;
    };
    uint value_;
  };

  uint oclVersion_;                   //!< Reported OpenCL version support
  uint debugFlags_;                   //!< Debug GPU flags
  size_t stagedXferSize_;             //!< Staged buffer size
  uint maxRenames_;                   //!< Maximum number of possible renames
  uint maxRenameSize_;                //!< Maximum size for all renames
  uint hwLDSSize_;                    //!< HW local data store size
  uint maxWorkGroupSize_;             //!< Requested workgroup size for this device
  uint preferredWorkGroupSize_;       //!< Requested preferred workgroup size for this device
  uint hostMemDirectAccess_;          //!< Enables direct access to the host memory
  amd::LibrarySelector libSelector_;  //!< Select linking libraries for compiler
  uint workloadSplitSize_;            //!< Workload split size
  uint minWorkloadTime_;              //!< Minimal workload time in 0.1 ms
  uint maxWorkloadTime_;              //!< Maximum workload time in 0.1 ms
  uint blitEngine_;                   //!< Blit engine type
  size_t pinnedXferSize_;             //!< Pinned buffer size for transfer
  size_t pinnedMinXferSize_;          //!< Minimal buffer size for pinned transfer
  size_t resourceCacheSize_;          //!< Resource cache size in MB
  uint64_t maxAllocSize_;             //!< Maximum single allocation size
  size_t numMemDependencies_;         //!< The array size for memory dependencies tracking
  uint cacheLineSize_;                //!< Cache line size in bytes
  uint cacheSize_;                    //!< L1 cache size in bytes
  size_t xferBufSize_;                //!< Transfer buffer size for image copy optimization
  uint numComputeRings_;              //!< 0 - disabled, 1 , 2,.. - the number of compute rings
  uint numDeviceEvents_;              //!< The number of device events
  uint numWaitEvents_;                //!< The number of wait events for device enqueue


  //! Default constructor
  Settings();

  //! Creates settings
  bool create(const CALdeviceattribs& calAttr  //!< CAL attributes structure
              ,
              bool reportAsOCL12Device = false  //!< Report As OpenCL1.2 Device
              ,
              bool smallMemSystem = false  //!< report the sys memory is small
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
