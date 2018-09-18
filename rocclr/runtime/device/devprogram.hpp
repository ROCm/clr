//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "include/aclTypes.h"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "devwavelimiter.hpp"

#if defined(WITH_LIGHTNING_COMPILER)
namespace llvm {
  namespace AMDGPU {
    namespace HSAMD {
      namespace Kernel {
        struct Metadata;
}}}}
typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
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

namespace amd {

class Device;
class Program;

namespace option {
  class Options;
}  // option
}

namespace device {
class ClBinary;
class Kernel;

//! A program object for a specific device.
class Program : public amd::HeapObject {
 public:
  typedef std::pair<const void*, size_t> binary_t;
  typedef std::unordered_map<std::string, Kernel*> kernels_t;
  // type of the program
  typedef enum {
    TYPE_NONE = 0,     // uncompiled
    TYPE_COMPILED,     // compiled
    TYPE_LIBRARY,      // linked library
    TYPE_EXECUTABLE,   // linked executable
    TYPE_INTERMEDIATE  // intermediate
  } type_t;

 private:
  //! The device target for this binary.
  amd::SharedReference<amd::Device> device_;

  kernels_t kernels_;  //!< The kernel entry points this binary.

  type_t type_;  //!< type of this program

 protected:
   union {
     struct {
       uint32_t isNull_ : 1;          //!< Null program no memory allocations
       uint32_t internal_ : 1;        //!< Internal blit program
       uint32_t isLC_ : 1;            //!< LC was used for the program compilation
       uint32_t hasGlobalStores_ : 1; //!< Program has writable program scope variables
     };
     uint32_t flags_;  //!< Program flags
   };

  ClBinary* clBinary_;                          //!< The CL program binary file
  std::string llvmBinary_;                      //!< LLVM IR binary code
  amd::OclElf::oclElfSections elfSectionType_;  //!< LLVM IR binary code is in SPIR format
  std::string compileOptions_;                  //!< compile/build options.
  std::string linkOptions_;                     //!< link options.
                                                //!< the option arg passed in to clCompileProgram(), clLinkProgram(),
                                                //! or clBuildProgram(), whichever is called last
  aclBinaryOptions binOpts_;           //!< Binary options to create aclBinary
  aclBinary* binaryElf_;               //!< Binary for the new compiler library

  std::string lastBuildOptionsArg_;
  std::string buildLog_;  //!< build log.
  cl_int buildStatus_;    //!< build status.
  cl_int buildError_;     //!< build error
                          //! The info target for this binary.
  aclTargetInfo info_;
  size_t globalVariableTotalSize_;
  amd::option::Options* programOptions_;

 public:
  //! Construct a section.
  Program(amd::Device& device);

  //! Destroy this binary image.
  virtual ~Program();

  //! Destroy all the kernels
  void clear();

  //! Return the compiler options passed to build this program
  amd::option::Options* getCompilerOptions() const { return programOptions_; }

  //! Compile the device program.
  cl_int compile(const std::string& sourceCode, const std::vector<const std::string*>& headers,
    const char** headerIncludeNames, const char* origOptions,
    amd::option::Options* options);

  //! Builds the device program.
  cl_int link(const std::vector<Program*>& inputPrograms, const char* origOptions,
    amd::option::Options* options);

  //! Builds the device program.
  cl_int build(const std::string& sourceCode, const char* origOptions,
    amd::option::Options* options);

  //! Returns the device object, associated with this program.
  const amd::Device& device() const { return device_(); }

  //! Return the compiler options used to build the program.
  const std::string& compileOptions() const { return compileOptions_; }

  //! Return the option arg passed in to clCompileProgram(), clLinkProgram(),
  //! or clBuildProgram(), whichever is called last
  const std::string lastBuildOptionsArg() const { return lastBuildOptionsArg_; }

  //! Return the build log.
  const std::string& buildLog() const { return buildLog_; }

  //! Return the build status.
  cl_build_status buildStatus() const { return buildStatus_; }

  //! Return the build error.
  cl_int buildError() const { return buildError_; }

  //! Return the symbols vector.
  const kernels_t& kernels() const { return kernels_; }
  kernels_t& kernels() { return kernels_; }

  //! Return the binary image.
  inline const binary_t binary() const;
  inline binary_t binary();

  //! Returns the CL program binary file
  ClBinary* clBinary() { return clBinary_; }
  const ClBinary* clBinary() const { return clBinary_; }

  bool setBinary(const char* binaryIn, size_t size);

  type_t type() const { return type_; }

  void setGlobalVariableTotalSize(size_t size) { globalVariableTotalSize_ = size; }

  size_t globalVariableTotalSize() const { return globalVariableTotalSize_; }

  //! Returns the aclBinary associated with the program
  aclBinary* binaryElf() const { return static_cast<aclBinary*>(binaryElf_); }

  //! Returns TRUE if the program just compiled
  bool isNull() const { return isNull_; }

  //! Returns TRUE if the program used internally by runtime
  bool isInternal() const { return internal_; }

  //! Returns TRUE if Lightning compiler was used for this program
  bool isLC() const { return isLC_; }

  //! Global variables are a part of the code segment
  bool hasGlobalStores() const { return hasGlobalStores_; }

 protected:
  //! pre-compile setup
  virtual bool initBuild(amd::option::Options* options);

  //! post-compile cleanup
  virtual bool finiBuild(bool isBuildGood);

  //! Compile the device program.
  virtual bool compileImpl(const std::string& sourceCode,
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames, amd::option::Options* options) = 0;

  //! Link the device program.
  virtual bool linkImpl(amd::option::Options* options) = 0;

  //! Link the device programs.
  virtual bool linkImpl(const std::vector<Program*>& inputPrograms, amd::option::Options* options,
    bool createLibrary) = 0;

  virtual bool createBinary(amd::option::Options* options) = 0;

  //! Initialize Binary (used only for clCreateProgramWithBinary()).
  bool initClBinary(const char* binaryIn, size_t size);

  //! Initialize Binary
  virtual bool initClBinary();

  //! Release the Binary
  void releaseClBinary();

  //! return target info
  virtual const aclTargetInfo& info(const char* str = "") = 0;

  virtual bool isElf(const char* bin) const = 0;

  //! Returns all the options to be appended while passing to the compiler library
  std::string ProcessOptions(amd::option::Options* options);

  //! At linking time, get the set of compile options to be used from
  //! the set of input program, warn if they have inconsisten compile options.
  bool getCompileOptionsAtLinking(const std::vector<Program*>& inputPrograms,
    const amd::option::Options* linkOptions);

  void setType(type_t newType) { type_ = newType; }

 private:
  //! Disable default copy constructor
  Program(const Program&);

  //! Disable operator=
  Program& operator=(const Program&);
};

} // namespace device