//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "acl.h"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "rocdevice.hpp"

#if defined(WITH_LIGHTNING_COMPILER)
#include "driver/AmdCompiler.h"
#include "llvm/Support/AMDGPUMetadata.h"

typedef llvm::AMDGPU::HSAMD::Metadata CodeObjectMD;
typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;

#endif  // defined(WITH_LIGHTNING_COMPILER)

//! \namespace roc HSA Device Implementation
namespace roc {

class HSAILProgram;
class LightningProgram;

//! \class empty program
class Program : public device::Program {
  friend class ClBinary;

 public:
  //! Default constructor
  Program(roc::NullDevice& device);
  //! Default destructor
  ~Program();

  // Initialize Binary for GPU (used only for clCreateProgramWithBinary()).
  virtual bool initClBinary(char* binaryIn, size_t size);

  //! Return a typecasted GPU device
  const NullDevice& dev() const { return static_cast<const NullDevice&>(device()); }

  //! Returns the hsaBinary associated with the program
  hsa_agent_t hsaDevice() const { return dev().getBackendDevice(); }

  hsa_executable_t hsaExecutable() const { return hsaExecutable_; }

 protected:
  //! pre-compile setup for GPU
  virtual bool initBuild(amd::option::Options* options);

  //! post-compile setup for GPU
  virtual bool finiBuild(bool isBuildGood);

  /*! \brief Compiles LLVM binary to HSAIL code (compiler backend: link+opt+codegen)
   *
   *  \return The build error code
   */
  int compileBinaryToHSAIL(amd::option::Options* options  //!< options for compilation
                           );
  virtual bool createBinary(amd::option::Options* options) = 0;

  virtual const aclTargetInfo& info(const char* str = "") { return info_; }

  virtual bool isElf(const char* bin) const {
    return amd::isElfMagic(bin);
    // return false;
  }

 protected:
  //! Disable default copy constructor
  Program(const Program&) = delete;
  //! Disable operator=
  Program& operator=(const Program&) = delete;

protected:
  /* HSA executable */
  hsa_executable_t hsaExecutable_;               //!< Handle to HSA executable
  hsa_code_object_reader_t hsaCodeObjectReader_; //!< Handle to HSA code reader
};

#if defined(WITH_COMPILER_LIB)
class HSAILProgram : public roc::Program {
 public:
  HSAILProgram(roc::NullDevice& device);
  virtual ~HSAILProgram();

 protected:
  virtual bool linkImpl(amd::option::Options* options) final;

  virtual bool linkImpl(const std::vector<device::Program*>& inputPrograms,
                        amd::option::Options* options, bool createLibrary) final;

  virtual bool createBinary(amd::option::Options* options) final;

private:
  std::string codegenOptions(amd::option::Options* options);

  bool saveBinaryAndSetType(type_t type);
};
#endif // defined(WITH_COMPILER_LIB)

#if defined(WITH_LIGHTNING_COMPILER)
class LightningProgram : public roc::Program {
public:
  LightningProgram(roc::NullDevice& device);
  virtual ~LightningProgram();

  //! Returns the program metadata.
  const CodeObjectMD* metadata() const { return metadata_; }

protected:
  virtual bool linkImpl(amd::option::Options* options) final;

  virtual bool linkImpl(const std::vector<device::Program*>& inputPrograms,
                        amd::option::Options* options, bool createLibrary) final;

  virtual bool createBinary(amd::option::Options* options) final;

  bool saveBinaryAndSetType(type_t type) { return true; }

private:
  bool saveBinaryAndSetType(type_t type, void* rawBinary, size_t size);

  bool setKernels(amd::option::Options* options, void* binary, size_t binSize);

  CodeObjectMD* metadata_;  //!< Runtime metadata
};
#endif // defined(WITH_LIGHTNING_COMPILER)

/*@}*/} // namespace roc

#endif /*WITHOUT_HSA_BACKEND*/
