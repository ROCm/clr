//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CPUSETTINGS_HPP_
#define CPUSETTINGS_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "device/cpu/cpufeat.hpp"

//! \namespace cpu CPU Device Implementation
namespace cpu {

//! Device settings
class Settings : public device::Settings {
 public:
  enum CpuFeatures {
    SSE2Instructions = 0x01,
    AVXInstructions = 0x02,   // Processor reports SSSE3, SSE4_1, SSE4_2
                              // POPCNT and AVX
    FMA3Instructions = 0x04,  // Intel processor reports FMA3
    FMA4Instructions = 0x08   // AMD processor reports FMA4 and XOP
  };
  uint32_t cpuFeatures_;  //!< CPU features

  //! Default constructor
  Settings() { cpuFeatures_ = 0; }

  //! Creates settings
  bool create();

 private:
  //! Disable copy constructor
  Settings(const Settings&);

  //! Disable assignment
  Settings& operator=(const Settings&);
};

}  // namespace cpu

#endif  // CPUSETTINGS_HPP_
