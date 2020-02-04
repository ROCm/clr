/* Copyright (c) 2014-present Advanced Micro Devices, Inc.

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
