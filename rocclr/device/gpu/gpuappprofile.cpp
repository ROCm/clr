//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#include "top.hpp"
#include "utils/debug.hpp"
#include "device/appprofile.hpp"
#include "device/gpu/gpuappprofile.hpp"

namespace gpu {

AppProfile::AppProfile()
    : amd::AppProfile(), enableHighPerformanceState_(true), reportAsOCL12Device_(false) {
  propertyDataMap_.insert({"HighPerfState", PropertyData(DataType_Boolean, &enableHighPerformanceState_)});
  propertyDataMap_.insert({"OCL12Device", PropertyData(DataType_Boolean, &reportAsOCL12Device_)});
  propertyDataMap_.insert({"SclkThreshold", PropertyData(DataType_String, &sclkThreshold_)});
  propertyDataMap_.insert({"DownHysteresis", PropertyData(DataType_String, &downHysteresis_)});
  propertyDataMap_.insert({"UpHysteresis", PropertyData(DataType_String, &upHysteresis_)});
  propertyDataMap_.insert({"PowerLimit", PropertyData(DataType_String, &powerLimit_)});
  propertyDataMap_.insert({"MclkThreshold", PropertyData(DataType_String, &mclkThreshold_)});
  propertyDataMap_.insert({"MclkUpHyst", PropertyData(DataType_String, &mclkUpHyst_)});
  propertyDataMap_.insert({"MclkDownHyst", PropertyData(DataType_String, &mclkDownHyst_)});
}
}
