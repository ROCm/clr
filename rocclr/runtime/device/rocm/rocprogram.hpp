//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "rocbinary.hpp"
#if !defined(WITH_LIGHTNING_COMPILER)
#include "roccompilerlib.hpp"
#endif  // !defined(WITH_LIGHTNING_COMPILER)
#include "acl.h"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "rocdevice.hpp"

#if defined(WITH_LIGHTNING_COMPILER)
#include "llvm/Support/AMDGPUCodeObjectMetadata.h"
#include "driver/AmdCompiler.h"

typedef llvm::AMDGPU::CodeObject::Metadata CodeObjectMD;
typedef llvm::AMDGPU::CodeObject::Kernel::Metadata KernelMD;
typedef llvm::AMDGPU::CodeObject::Kernel::Arg::Metadata KernelArgMD;
#endif  // defined(WITH_LIGHTNING_COMPILER)

//! \namespace roc HSA Device Implementation
namespace roc {

//! \class empty program
class HSAILProgram : public device::Program {
  friend class ClBinary;

 public:
  //! Default constructor
  HSAILProgram(roc::NullDevice& device);
  //! Default destructor
  ~HSAILProgram();

  // Initialize Binary for GPU (used only for clCreateProgramWithBinary()).
  virtual bool initClBinary(char* binaryIn, size_t size);

  //! Returns the aclBinary associated with the program
  const aclBinary* binaryElf() const { return static_cast<const aclBinary*>(binaryElf_); }

#if defined(WITH_LIGHTNING_COMPILER)
  //! Returns the program metadata.
  const CodeObjectMD* metadata() const { return metadata_; }
#endif  // defined(WITH_LIGHTNING_COMPILER)

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

  /*! \brief Compiles GPU CL program to LLVM binary (compiler frontend)
   *
   *  \return True if we successfully compiled a GPU program
   */
  virtual bool compileImpl(const std::string& sourceCode,  //!< the program's source code
                           const std::vector<const std::string*>& headers,
                           const char** headerIncludeNames,
                           amd::option::Options* options  //!< compile options's object
                           );
#if defined(WITH_LIGHTNING_COMPILER)
  virtual bool compileImpl_LC(const std::string& sourceCode,  //!< the program's source code
                              const std::vector<const std::string*>& headers,
                              const char** headerIncludeNames,
                              amd::option::Options* options  //!< compile options's object
                              );
#endif  // defined(WITH_LIGHTNING_COMPILER)

  /*! \brief Compiles LLVM binary to HSAIL code (compiler backend: link+opt+codegen)
   *
   *  \return The build error code
   */
  int compileBinaryToHSAIL(amd::option::Options* options  //!< options for compilation
                           );


  virtual bool linkImpl(amd::option::Options* options);
#if defined(WITH_LIGHTNING_COMPILER)
  virtual bool linkImpl_LC(amd::option::Options* options);
  bool setKernels_LC(amd::option::Options* options, void* binary, size_t binSize);
#endif  // defined(WITH_LIGHTNING_COMPILER)

  //! Link the device programs.
  virtual bool linkImpl(const std::vector<Program*>& inputPrograms, amd::option::Options* options,
                        bool createLibrary);
#if defined(WITH_LIGHTNING_COMPILER)
  virtual bool linkImpl_LC(const std::vector<Program*>& inputPrograms,
                           amd::option::Options* options, bool createLibrary);
#endif  // defined(WITH_LIGHTNING_COMPILER)

  virtual bool createBinary(amd::option::Options* options);

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

 private:
  /* \brief Returns the next stage to compile from, based on sections in binary,
   *  also returns completeStages in a vector, which contains at least ACL_TYPE_DEFAULT,
   *  sets needOptionsCheck to true if options check is needed to decide whether or not to recompile
   */
  aclType getCompilationStagesFromBinary(std::vector<aclType>& completeStages,
                                         bool& needOptionsCheck);

  /* \brief Returns the next stage to compile from, based on sections and options in binary
   */
  aclType getNextCompilationStageFromBinary(amd::option::Options* options);
  bool saveBinaryAndSetType(type_t type, void* binary = nullptr, size_t size = 0);

  //! Disable default copy constructor
  HSAILProgram(const HSAILProgram&) = delete;
  //! Disable operator=
  HSAILProgram& operator=(const HSAILProgram&) = delete;

  //! Returns all the options to be appended while passing to the
  // compiler
  std::string preprocessorOptions(amd::option::Options* options);
  std::string codegenOptions(amd::option::Options* options);

  // aclBinary and aclCompiler - for the compiler library
  aclBinary* binaryElf_;      //!< Binary for the new compiler library
  aclBinaryOptions binOpts_;  //!< Binary options to create aclBinary
  bool hasGlobalStores_;      //!< program has writable program scope variables

  /* HSA executable */
  hsa_ext_program_t hsaProgramHandle_;  //!< Handle to HSA runtime program
  hsa_executable_t hsaExecutable_;      //!< Handle to HSA executable

#if defined(WITH_LIGHTNING_COMPILER)
  CodeObjectMD* metadata_;  //!< Runtime metadata
  //! Return a new transient compiler instance.
  static amd::opencl_driver::Compiler* newCompilerInstance();
#endif  // defined(WITH_LIGHTNING_COMPILER)
};

/*@}*/} // namespace roc

#endif /*WITHOUT_HSA_BACKEND*/
