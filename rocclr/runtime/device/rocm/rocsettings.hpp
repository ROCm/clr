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
class Settings : public device::Settings
{
public:
    union {
        struct {
            uint    doublePrecision_: 1;        //!< Enables double precision support
            uint    pollCompletion_: 1;         //!< Enables polling in HSA
            uint    enableLocalMemory_ : 1;     //!< Enable GPUVM memory
            uint    enableImageHandle_: 1;      //!< Use HSAIL image/sampler pointer
            uint    enableNCMode_: 1;           //!< Enable Non Coherent mode for system memory
            uint    enablePartialDispatch_: 1;  //!< Enable support for Partial Dispatch
            uint    reserved_: 26;
        };
        uint    value_;
    };

    //! Default max workgroup size for 1D
    int maxWorkGroupSize_;

    //! Default max workgroup sizes for 2D
    int maxWorkGroupSize2DX_;
    int maxWorkGroupSize2DY_;

    //! Default max workgroup sizes for 3D
    int maxWorkGroupSize3DX_;
    int maxWorkGroupSize3DY_;
    int maxWorkGroupSize3DZ_;

    uint kernargPoolSize_;
    uint signalPoolSize_;

    //! Default constructor
    Settings();

    //! Creates settings
    bool create(bool doublePrecision);

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

