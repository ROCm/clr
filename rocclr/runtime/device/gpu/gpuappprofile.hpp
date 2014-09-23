//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef GPUAPPPROFILE_HPP_
#define GPUAPPPROFILE_HPP_

#include <string>
#include <map>

namespace gpu {

class AppProfile : public amd::AppProfile
{
public:
    AppProfile();

    //! return the value of enableHighPerformanceState_
    bool enableHighPerformanceState() const { return enableHighPerformanceState_; }
    bool reportAsOCL12Device() const { return reportAsOCL12Device_; }

private:

    bool enableHighPerformanceState_;
    bool reportAsOCL12Device_;
};

}

#endif
