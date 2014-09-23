//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#include "top.hpp"
#include "utils/debug.hpp"
#include "device/appprofile.hpp"
#include "device/gpu/gpuappprofile.hpp"

namespace gpu {

AppProfile::AppProfile()
    : amd::AppProfile()
    , enableHighPerformanceState_(IS_LINUX ? false : true)
    , reportAsOCL12Device_(false)
{
    propertyDataMap_.insert(DataMap::value_type("HighPerfState",
        PropertyData(DataType_Boolean, &enableHighPerformanceState_)));

    propertyDataMap_.insert(DataMap::value_type("OCL12Device",
        PropertyData(DataType_Boolean, &reportAsOCL12Device_)));
}

}

