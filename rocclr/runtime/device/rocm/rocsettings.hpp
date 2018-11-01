//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "library.hpp"

/*! \addtogroup HSA OCL Stub Implementation
 *  @{
 */

//! HSA OCL STUB Implementation
namespace roc {

//! Device settings
class Settings : public device::Settings {
 public:
  union {
    struct {
      uint doublePrecision_ : 1;        //!< Enables double precision support
      uint enableLocalMemory_ : 1;      //!< Enable GPUVM memory
      uint enableCoarseGrainSVM_ : 1;   //!< Enable device memory for coarse grain SVM allocations
      uint enableNCMode_ : 1;           //!< Enable Non Coherent mode for system memory
      uint imageDMA_ : 1;               //!< Enable direct image DMA transfers
      uint stagedXferRead_ : 1;         //!< Uses a staged buffer read
      uint stagedXferWrite_ : 1;        //!< Uses a staged buffer write
      uint reserved_ : 25;
    };
    uint value_;
  };

  //! Default max workgroup size for 1D
  int maxWorkGroupSize_;

  //! Preferred workgroup size
  uint preferredWorkGroupSize_;

  //! Default max workgroup sizes for 2D
  int maxWorkGroupSize2DX_;
  int maxWorkGroupSize2DY_;

  //! Default max workgroup sizes for 3D
  int maxWorkGroupSize3DX_;
  int maxWorkGroupSize3DY_;
  int maxWorkGroupSize3DZ_;

  uint kernargPoolSize_;
  uint numDeviceEvents_;      //!< The number of device events
  uint numWaitEvents_;        //!< The number of wait events for device enqueue

  size_t xferBufSize_;        //!< Transfer buffer size for image copy optimization
  size_t stagedXferSize_;     //!< Staged buffer size
  size_t pinnedXferSize_;     //!< Pinned buffer size for transfer
  size_t pinnedMinXferSize_;  //!< Minimal buffer size for pinned transfer

  //! Default constructor
  Settings();

  //! Creates settings
  bool create(bool fullProfile, int gfxipVersion);

 private:
  //! Disable copy constructor
  Settings(const Settings&);

  //! Disable assignment
  Settings& operator=(const Settings&);

  //! Overrides current settings based on registry/environment
  void override();
};

/*@}*/} // namespace roc

#endif /*WITHOUT_HSA_BACKEND*/
