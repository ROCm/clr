//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "rocbinary.hpp"
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

  //! Returns the aclBinary associated with the program
  const aclBinary* binaryElf() const { return static_cast<const aclBinary*>(binaryElf_); }

  //! Return a typecasted GPU device
  const NullDevice& dev() const { return static_cast<const NullDevice&>(device()); }

  //! Returns the hsaBinary associated with the program
  hsa_agent_t hsaDevice() const { return dev().getBackendDevice(); }

  bool hasGlobalStores() const { return hasGlobalStores_; }

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

  //! Initialize Binary
  virtual bool initClBinary();

  //! Release the Binary
  virtual void releaseClBinary();

  virtual const aclTargetInfo& info(const char* str = "") { return info_; }

  virtual bool isElf(const char* bin) const {
    return amd::isElfMagic(bin);
    // return false;
  }

  //! Returns the binary
  // This should ensure that the binary is updated with all the kernels
  //    ClBinary& clBinary() { return binary_; }
  ClBinary* clBinary() { return static_cast<ClBinary*>(device::Program::clBinary()); }
  const ClBinary* clBinary() const {
    return static_cast<const ClBinary*>(device::Program::clBinary());
  }

 protected:
  /* \brief Returns the next stage to compile from, based on sections in binary,
   *  also returns completeStages in a vector, which contains at least ACL_TYPE_DEFAULT,
   *  sets needOptionsCheck to true if options check is needed to decide whether or not to recompile
   */
  virtual aclType getCompilationStagesFromBinary(std::vector<aclType>& completeStages,
                                         bool& needOptionsCheck) = 0;

  /* \brief Returns the next stage to compile from, based on sections and options in binary
   */
  aclType getNextCompilationStageFromBinary(amd::option::Options* options);

  //! Disable default copy constructor
  Program(const Program&) = delete;
  //! Disable operator=
  Program& operator=(const Program&) = delete;

protected:
  //! Returns all the options to be appended while passing to the
  // compiler
  std::string preprocessorOptions(amd::option::Options* options);

  // aclBinary and aclCompiler - for the compiler library
  aclBinary* binaryElf_;      //!< Binary for the new compiler library
  aclBinaryOptions binOpts_;  //!< Binary options to create aclBinary
  bool hasGlobalStores_;      //!< program has writable program scope variables

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
  virtual bool compileImpl(const std::string& sourceCode,  //!< the program's source code
                           const std::vector<const std::string*>& headers,
                           const char** headerIncludeNames,
                           amd::option::Options* options  //!< compile options's object
                           ) final;

  virtual bool linkImpl(amd::option::Options* options) final;

  virtual bool linkImpl(const std::vector<device::Program*>& inputPrograms,
                        amd::option::Options* options, bool createLibrary) final;

  virtual bool createBinary(amd::option::Options* options) final;

private:
  std::string codegenOptions(amd::option::Options* options);

  virtual aclType getCompilationStagesFromBinary(std::vector<aclType>& completeStages,
                                                 bool& needOptionsCheck) final;

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
  virtual bool compileImpl(const std::string& sourceCode,  //!< the program's source code
                           const std::vector<const std::string*>& headers,
                           const char** headerIncludeNames,
                           amd::option::Options* options  //!< compile options's object
                           ) final;

  virtual bool linkImpl(amd::option::Options* options) final;

  virtual bool linkImpl(const std::vector<device::Program*>& inputPrograms,
                        amd::option::Options* options, bool createLibrary) final;

  virtual bool createBinary(amd::option::Options* options) final;

private:
  bool saveBinaryAndSetType(type_t type, void* rawBinary, size_t size);

  virtual aclType getCompilationStagesFromBinary(std::vector<aclType>& completeStages,
                                                 bool& needOptionsCheck) final;

  bool setKernels(amd::option::Options* options, void* binary, size_t binSize);

  CodeObjectMD* metadata_;  //!< Runtime metadata
  //! Return a new transient compiler instance.
  static amd::opencl_driver::Compiler* newCompilerInstance();
};
#endif // defined(WITH_LIGHTNING_COMPILER)

/*@}*/} // namespace roc

#endif /*WITHOUT_HSA_BACKEND*/
