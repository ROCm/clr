//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef GPUAPPPROFILE_HPP_
#define GPUAPPPROFILE_HPP_

#include <string>
#include <map>

namespace gpu {

class AppProfile : public amd::AppProfile {
 public:
  AppProfile();

  //! return the value of enableHighPerformanceState_
  bool enableHighPerformanceState() const { return enableHighPerformanceState_; }
  bool reportAsOCL12Device() const { return reportAsOCL12Device_; }
  const std::string& GetSclkThreshold() const { return sclkThreshold_; }
  const std::string& GetDownHysteresis() const { return downHysteresis_; }
  const std::string& GetUpHysteresis() const { return upHysteresis_; }
  const std::string& GetPowerLimit() const { return powerLimit_; }
  const std::string& GetMclkThreshold() const { return mclkThreshold_; }
  const std::string& GetMclkUpHyst() const { return mclkUpHyst_; }
  const std::string& GetMclkDownHyst() const { return mclkDownHyst_; }

 private:
  bool          enableHighPerformanceState_;
  bool          reportAsOCL12Device_;
  std::string   sclkThreshold_;
  std::string   downHysteresis_;
  std::string   upHysteresis_;
  std::string   powerLimit_;
  std::string   mclkThreshold_;
  std::string   mclkUpHyst_;
  std::string   mclkDownHyst_;
};
}

#endif
