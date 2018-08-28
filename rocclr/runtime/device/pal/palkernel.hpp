//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "device/device.hpp"
#include "utils/macros.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "platform/sampler.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palvirtual.hpp"
#include "amd_hsa_kernel_code.h"
#include "device/pal/palprintf.hpp"
#include "device/pal/palwavelimiter.hpp"
#include "hsa.h"

#if defined(WITH_LIGHTNING_COMPILER)
#include "llvm/Support/AMDGPUMetadata.h"

typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;
#endif  // defined(WITH_LIGHTNING_COMPILER)

namespace amd {
namespace hsa {
namespace loader {
class Symbol;
}  // loader
namespace code {
namespace Kernel {
class Metadata;
}  // Kernel
}  // code
}  // hsa
}  // amd

//! \namespace pal PAL Device Implementation
namespace pal {

class VirtualGPU;
class Device;
class NullDevice;
class HSAILProgram;
class LightningProgram;

/*! \addtogroup pal PAL Device Implementation
 *  @{
 */
class HSAILKernel : public device::Kernel {
 public:

  HSAILKernel(std::string name, HSAILProgram* prog, std::string compileOptions);

  virtual ~HSAILKernel();

  //! Initializes the metadata required for this kernel,
  //! finalizes the kernel if needed
  bool init(amd::hsa::loader::Symbol* sym, bool finalize = false);

  //! Returns GPU device object, associated with this kernel
  const Device& dev() const;

  //! Returns HSA program associated with this kernel
  const HSAILProgram& prog() const;

  //! Returns LDS size used in this kernel
  uint32_t ldsSize() const { return cpuAqlCode_->workgroup_group_segment_byte_size; }

  //! Returns pointer on CPU to AQL code info
  const amd_kernel_code_t* cpuAqlCode() const { return cpuAqlCode_; }

  //! Returns memory object with AQL code
  uint64_t gpuAqlCode() const { return code_; }

  //! Returns size of AQL code
  size_t aqlCodeSize() const { return codeSize_; }

  //! Returns the size of argument buffer
  size_t argsBufferSize() const { return cpuAqlCode_->kernarg_segment_byte_size; }

  //! Returns spill reg size per workitem
  int spillSegSize() const { return amd::alignUp(cpuAqlCode_->workitem_private_segment_byte_size, sizeof(uint32_t)); }

  //! Finds local workgroup size
  void findLocalWorkSize(size_t workDim,                   //!< Work dimension
                         const amd::NDRange& gblWorkSize,  //!< Global work size
                         amd::NDRange& lclWorkSize         //!< Local work size
                         ) const;

  //! Returns AQL packet in CPU memory
  //! if the kernel arguments were successfully loaded, otherwise NULL
  hsa_kernel_dispatch_packet_t* loadArguments(
      VirtualGPU& gpu,                     //!< Running GPU context
      const amd::Kernel& kernel,           //!< AMD kernel object
      const amd::NDRangeContainer& sizes,  //!< NDrange container
      const_address parameters,            //!< Application arguments for the kernel
      size_t ldsAddress,                   //!< LDS address that includes all arguments.
      uint64_t vmDefQueue,                 //!< GPU VM default queue pointer
      uint64_t* vmParentWrap               //!< GPU VM parent aql wrap object
      ) const;


  //! Returns pritnf info array
  const std::vector<PrintfInfo>& printfInfo() const { return printf_; }

  //! Returns the kernel index in the program
  uint index() const { return index_; }

  //! Get profiling callback object
  virtual amd::ProfilingCallback* getProfilingCallback(const device::VirtualDevice* vdev) {
    return waveLimiter_.getProfilingCallback(vdev);
  };

  //! Get waves per shader array to be used for kernel execution.
  virtual uint getWavesPerSH(const device::VirtualDevice* vdev) const {
    return waveLimiter_.getWavesPerSH(vdev);
  };

 private:
  //! Disable copy constructor
  HSAILKernel(const HSAILKernel&);

  //! Disable operator=
  HSAILKernel& operator=(const HSAILKernel&);

 protected:
  //! Creates AQL kernel HW info
  bool aqlCreateHWInfo(amd::hsa::loader::Symbol* sym);

  //! Initializes Hsail Printf metadata and info
  void initPrintf(const aclPrintfFmt* aclPrintf  //!< List of ACL printfs
                  );

  std::string compileOptions_;        //!< compile used for finalizing this kernel
  amd_kernel_code_t* cpuAqlCode_;     //!< AQL kernel code on CPU
  const NullDevice& dev_;             //!< GPU device object
  const HSAILProgram& prog_;          //!< Reference to the parent program
  std::vector<PrintfInfo> printf_;    //!< Format strings for GPU printf support
  uint index_;                        //!< Kernel index in the program

  uint64_t code_;    //!< GPU memory pointer to the kernel
  size_t codeSize_;  //!< Size of ISA code

  WaveLimiterManager waveLimiter_;  //!< adaptively control number of waves
};

#if defined(WITH_LIGHTNING_COMPILER)
class LightningKernel : public HSAILKernel {
 public:
  LightningKernel(const std::string& name, HSAILProgram* prog, const std::string& compileOptions)
      : HSAILKernel(name, prog, compileOptions) {}

  //! Returns Lightning program associated with this kernel
  const LightningProgram& prog() const;

  //! Initializes the metadata required for this kernel,
  bool init(amd::hsa::loader::Symbol* symbol);

  //! Initializes HSAIL Printf metadata and info for LC
  void initPrintf(const std::vector<std::string>& printfInfoStrings);
};
#endif  // defined(WITH_LIGHTNING_COMPILER)

/*@}*/} // namespace pal
