//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef HSASETTINGS_HPP_
#define HSASETTINGS_HPP_

#ifndef WITHOUT_GPU_BACKEND

#include "library.hpp"

/*! \addtogroup HSA OCL Stub Implementation
 *  @{
 */

//! HSA OCL STUB Implementation
namespace oclhsa {

//! Device settings
class Settings : public device::Settings
{
public:
    union {
        struct {
            uint    doublePrecision_: 1;    //!< Enables double precision support
            uint    pollCompletion_: 1;     //!< Enables polling in HSA
            uint    enableLocalMemory_: 1;  //!< Enable HSA device local memory usage
            uint    enableSvm32BitsAtomics_: 1; //!< Enable platform atomics in 32 bits
            uint    reserved_: 27;
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

/*@}*/} // namespace oclhsa

#endif /*WITHOUT_GPU_BACKEND*/
#endif /*HSASETTINGS_HPP_*/
