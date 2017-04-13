//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CPUBINARY_HPP_
#define CPUBINARY_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "device/cpu/cpudevice.hpp"
#include "elf/elf.hpp"

//! \namespace cpu CPU Device Implementation
namespace cpu {

class Device;
class Program;

//! \class CPU binary
class ClBinary : public device::ClBinary {
 public:
  //! Constructor
  ClBinary(const Device& dev) : device::ClBinary(dev) {}

  //! Destructor
  ~ClBinary() {}

  //! Loads x86 executable code
  bool loadX86(Program& prorgam,      //!< CPU Program object
               std::string& dllName,  //!< Dll name of the CPU binary
               bool& hasDLL           //!< indicate if the OCL binary has DLL
               );

  //! Stores x86 executable code
  bool storeX86(Program& program,     //!< CPU Program object
                std::string& dllName  //!< Dll name for the binary
                );

  //! Loads x86 executable in-memory code
  bool loadX86JIT(Program& prorgam,  //!< CPU Program object
                  bool& hasJITBin    //!< indicate if the OCL binary has JIT binary
                  );

  //! Stores x86 executable in-memory code
  bool storeX86JIT(Program& program  //!< CPU Program object
                   );

  //! Set elf header information for CPU target
  bool setElfTarget() {
    uint32_t target = dev().settings().cpuFeatures_;
    assert(((0xFFFF8000 & target) == 0) && "ASIC target ID >= 2^15");
    uint16_t elf_target = (uint16_t)(0x7FFF & target);
    return elfOut()->setTarget(elf_target, amd::OclElf::CPU_PLATFORM);
  }

  bool storeX86Asm(const char* buffer, size_t size);

 private:
  enum FeatureCheckResult { ERROR, RECOMPILE, OK };

  FeatureCheckResult checkFeatures();

  //! Disable default copy constructor
  ClBinary(const ClBinary&);

  //! Disable default operator=
  ClBinary& operator=(const ClBinary&);

  //! Returns the GPU device for this object
  const Device& dev() { return static_cast<const Device&>(dev_); }
};

}  // namespace cpu

#endif  // CPUBINARY_HPP_
