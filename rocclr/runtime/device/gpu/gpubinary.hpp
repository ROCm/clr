//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef GPUBINARY_HPP_
#define GPUBINARY_HPP_

#include "top.hpp"
#include "device/gpu/gpudevice.hpp"
#include "device/gpu/gpukernel.hpp"

namespace gpu {

class ClBinary : public device::ClBinary {
 public:
#pragma pack(push, 8)
  // Kernel version in the ELF header symbol
  enum KernelVersions { VERSION_0 = 0, VERSION_1, VERSION_CURRENT = VERSION_1 };

  /* This is the ELF header symbol */
  struct KernelHeaderSymbol {
    /* VERSION_0
       Version 0 has 8 uint32_t (32 bytes), top 5 are used, the rest zero'ed.
       In Version_0, KernelHeaderSymbol is the same as KernelHeader
     */
    uint32_t privateSize_;    //!< Emulated private memory size
    uint32_t localSize_;      //!< Emulated local memory size
    uint32_t hwPrivateSize_;  //!< HW private memory size
    uint32_t hwLocalSize_;    //!< HW local memory size
    uint32_t flags_;          //!< Kernel's flags

    /* VERSION_1
       VERSION_1 has 6 uint32_t.
     */
    uint32_t version_;       //!< Kernel's version
    uint32_t regionSize_;    //!< Region memory size
    uint32_t hwRegionSize_;  //!< HW region memory size

    /* New entries can be added here, do not change the previous entries */
  };

#pragma pack(pop)

  //! Constructor
  ClBinary(const NullDevice& dev) : device::ClBinary(dev) {}

  //! Destructor
  ~ClBinary() {}

  //! Creates and loads kernels from the OCL ELF binary file into the program
  bool loadKernels(NullProgram& program,  //!< Program object with the binary
                   bool* hasRecompiled    //!< Recompile amdil to isa.
                   );

  //! Stores compiled kernel into the OCL ELF binary file
  bool storeKernel(const std::string& name,       //!< Kernel's name
                   const NullKernel* nullKernel,  //!< The kernel to add
                   Kernel::InitData* initData,    //!< Kernel init data
                   const std::string& metadata,   //!< Kernel's metadata
                   const std::string& ilSource    //!< IL source text
                   );

  //! Loads the program's global data
  bool loadGlobalData(Program& program  //!< The program object for the global data load
                      );

  //! Stores the program's global data
  bool storeGlobalData(const void* globalData,  //!< The program global data
                       size_t dataSize,         //!< The program global data size
                       uint index               //!< The global data storage index
                       );

  //! Set elf header information for GPU target
  bool setElfTarget() {
    uint32_t target = static_cast<uint32_t>(dev().calTarget());
    assert(((0xFFFF8000 & target) == 0) && "ASIC target ID >= 2^15");
    uint16_t elf_target = (uint16_t)(0x7FFF & target);
    return elfOut()->setTarget(elf_target, amd::OclElf::CAL_PLATFORM);
  }

  //! Clear elf out.
  bool clearElfOut();

 private:
  //! Disable default copy constructor
  ClBinary(const ClBinary&);

  //! Disable default operator=
  ClBinary& operator=(const ClBinary&);

  //! Returns the GPU device for this object
  const NullDevice& dev() const { return static_cast<const NullDevice&>(dev_); }
};

class ClBinaryHsa : public device::ClBinary {
 public:
  ClBinaryHsa(const Device& dev, BinaryImageFormat bifVer = BIF_VERSION3)
      : device::ClBinary(dev, bifVer) {}

  //! Destructor
  ~ClBinaryHsa() {}


 protected:
  bool setElfTarget() {
    uint32_t target = static_cast<uint32_t>(21);  // dev().calTarget());
    assert(((0xFFFF8000 & target) == 0) && "ASIC target ID >= 2^15");
    uint16_t elf_target = (uint16_t)(0x7FFF & target);
    return elfOut()->setTarget(elf_target, amd::OclElf::CAL_PLATFORM);
    return true;
  }

 private:
  //! Disable default copy constructor
  ClBinaryHsa(const ClBinaryHsa&);

  //! Disable default operator=
  ClBinaryHsa& operator=(const ClBinaryHsa&);

  //! Returns the HSA device for this object
  const Device& dev() const { return static_cast<const Device&>(dev_); }
};


}  // namespace gpu

#endif  // GPUBINARY_HPP_
