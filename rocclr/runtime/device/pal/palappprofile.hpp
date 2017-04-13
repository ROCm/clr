//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include <string>
#include <map>

namespace pal {

class AppProfile : public amd::AppProfile {
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
