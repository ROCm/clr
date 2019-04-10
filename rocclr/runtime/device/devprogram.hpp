//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "include/aclTypes.h"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "devwavelimiter.hpp"
#include "comgrctx.hpp"

#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
#ifndef USE_COMGR_LIBRARY
#include "driver/AmdCompiler.h"
#endif
//#include "llvm/Support/AMDGPUMetadata.h"

namespace llvm {
  namespace AMDGPU {
    namespace HSAMD {
      struct Metadata;
      namespace Kernel {
        struct Metadata;
}}}}

#define LC_METADATA 1
typedef llvm::AMDGPU::HSAMD::Metadata CodeObjectMD;
typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
//typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;
#endif  // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

#ifndef LC_METADATA
typedef char CodeObjectMD;
#endif

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

  kernels_t kernels_; //!< The kernel entry points this binary.
  type_t type_;       //!< type of this program

 protected:
   union {
     struct {
       uint32_t isNull_ : 1;          //!< Null program no memory allocations
       uint32_t internal_ : 1;        //!< Internal blit program
       uint32_t isLC_ : 1;            //!< LC was used for the program compilation
       uint32_t hasGlobalStores_ : 1; //!< Program has writable program scope variables
       uint32_t xnackEnabled_ : 1;    //!< Xnack was enabled during compilation
       uint32_t sramEccEnabled_ : 1;  //!< SRAM ECC was enabled during compilation
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
  aclBinaryOptions binOpts_;        //!< Binary options to create aclBinary
  aclBinary* binaryElf_;            //!< Binary for the new compiler library

  std::string lastBuildOptionsArg_;
  mutable std::string buildLog_;    //!< build log.
  cl_int buildStatus_;              //!< build status.
  cl_int buildError_;               //!< build error

  const char* machineTarget_;       //!< Machine target for this program
  aclTargetInfo info_;              //!< The info target for this binary.
  size_t globalVariableTotalSize_;
  amd::option::Options* programOptions_;


#if defined(USE_COMGR_LIBRARY)
  amd_comgr_metadata_node_t* metadata_;   //!< COMgr metadata
  uint32_t codeObjectVer_;                //!< version of code object
  std::map<std::string,amd_comgr_metadata_node_t> kernelMetadataMap_; //!< Map of kernel metadata
#else
  CodeObjectMD* metadata_;  //!< Runtime metadata
#endif

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

#if defined(USE_COMGR_LIBRARY)
  const amd_comgr_metadata_node_t* metadata() const { return metadata_; }

  //! Get the kernel metadata
  const amd_comgr_metadata_node_t* getKernelMetadata(const std::string name) const {
    auto it = kernelMetadataMap_.find(name);
    return (it == kernelMetadataMap_.end()) ? nullptr : &(it->second);
  }

  const uint32_t codeObjectVer() const { return codeObjectVer_; }
#else
  const CodeObjectMD* metadata() const { return metadata_; }
#endif

  //! Get the machine target for the program
  const char* machineTarget() const { return machineTarget_; }

  //! Check if xnack is enable
  const bool xnackEnable() const { return (xnackEnabled_ == 1); }

  //! Check if SRAM ECC is enable
  const bool sramEccEnable() const { return (sramEccEnabled_ == 1); }

  virtual bool createGlobalVarObj(amd::Memory** amd_mem_obj, void** dptr,
                                  size_t* bytes, const char* globalName) const {
    ShouldNotReachHere();
    return false;
  }

 protected:
  //! pre-compile setup
  bool initBuild(amd::option::Options* options);

  //! post-compile cleanup
  bool finiBuild(bool isBuildGood);

  /*! \brief Compiles GPU CL program to LLVM binary (compiler frontend)
  *
  *  \return True if we successefully compiled a GPU program
  */
  virtual bool compileImpl(
    const std::string& sourceCode,  //!< the program's source code
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames,
    amd::option::Options* options   //!< compile options's object
  );

  //! Link the device program.
  virtual bool linkImpl(amd::option::Options* options);

  //! Link the device programs.
  virtual bool linkImpl(const std::vector<Program*>& inputPrograms, amd::option::Options* options,
    bool createLibrary);

  virtual bool createBinary(amd::option::Options* options) = 0;

  //! Initialize Binary (used only for clCreateProgramWithBinary()).
  bool initClBinary(const char* binaryIn, size_t size);

  //! Initialize Binary
  virtual bool initClBinary();

  virtual bool saveBinaryAndSetType(type_t type) = 0;

  //! Release the Binary
  void releaseClBinary();

  //! return target info
  virtual const aclTargetInfo& info(const char* str = "") = 0;

  virtual bool setKernels(
    amd::option::Options* options, void* binary, size_t binSize) { return true; }

  //! Returns all the options to be appended while passing to the compiler library
  std::string ProcessOptions(amd::option::Options* options);

  //! At linking time, get the set of compile options to be used from
  //! the set of input program, warn if they have inconsisten compile options.
  bool getCompileOptionsAtLinking(const std::vector<Program*>& inputPrograms,
    const amd::option::Options* linkOptions);

  void setType(type_t newType) { type_ = newType; }

#if defined(WITH_LIGHTNING_COMPILER) && !defined(USE_COMGR_LIBRARY)
  //! Return a new transient compiler instance.
  static std::unique_ptr<amd::opencl_driver::Compiler> newCompilerInstance();
#endif // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

  /* \brief Returns the next stage to compile from, based on sections in binary,
  *  also returns completeStages in a vector, which contains at least ACL_TYPE_DEFAULT,
  *  sets needOptionsCheck to true if options check is needed to decide whether or not to recompile
  */
  aclType getCompilationStagesFromBinary(
    std::vector<aclType>& completeStages,
    bool& needOptionsCheck);

  /* \brief Returns the next stage to compile from, based on sections and options in binary
  */
  aclType getNextCompilationStageFromBinary(amd::option::Options* options);

  //! Finds the total size of all global variables in the program
  bool FindGlobalVarSize(void* binary, size_t binSize);

  bool isElf(const char* bin) const { return amd::isElfMagic(bin); }

 private:
  //! Compile the device program with LC path
  bool compileImplLC(const std::string& sourceCode,
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames, amd::option::Options* options);

  //! Compile the device program with HSAIL path
  bool compileImplHSAIL(const std::string& sourceCode,
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames, amd::option::Options* options);

  //! Link the device programs with LC path
  bool linkImplLC(const std::vector<Program*>& inputPrograms,
    amd::option::Options* options, bool createLibrary);

  //! Link the device programs with HSAIL path
  bool linkImplHSAIL(const std::vector<Program*>& inputPrograms,
    amd::option::Options* options, bool createLibrary);

  //! Link the device program with LC path
  bool linkImplLC(amd::option::Options* options);

  //! Link the device program with HSAIL path
  bool linkImplHSAIL(amd::option::Options* options);

#if defined(USE_COMGR_LIBRARY)
  //! Dump the log data object to the build log, if both are present
  void extractBuildLog(const char* buildLog, amd_comgr_data_set_t dataSet);
  //! Dump the code object data
  amd_comgr_status_t extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
    const amd_comgr_data_kind_t dataKind, const std::string& outFileName,
    char* outBinary[] = nullptr, size_t* outSize = nullptr);

  //! Set the OCL language and target triples with feature
  void setLangAndTargetStr(const char* clStd, amd_comgr_language_t* oclver,
                           std::string& targetIdent);

  //! Create code object and add it into the data set
  amd_comgr_status_t addCodeObjData(const char *source,
    const size_t size, const amd_comgr_data_kind_t type,
    const char* name, amd_comgr_data_set_t* dataSet);

  //! Create action for the specified language, target and options
  amd_comgr_status_t createAction(const amd_comgr_language_t oclvar,
    const std::string& targetIdent, const std::string& options,
    amd_comgr_action_info_t* action, bool* hasAction);

  //! Create the bitcode of the linked input dataset
  bool linkLLVMBitcode(const amd_comgr_data_set_t inputs,
    const std::string& options, const bool requiredDump,
    amd::option::Options* amdOptions, amd_comgr_data_set_t* output,
    char* binary[] = nullptr, size_t* binarySize = nullptr);

  //! Create the bitcode of the compiled input dataset
  bool compileToLLVMBitcode(const amd_comgr_data_set_t inputs,
    const std::string& options, amd::option::Options* amdOptions,
    char* binary[], size_t* binarySize);

  //! Compile and create the excutable of the input dataset
  bool compileAndLinkExecutable(const amd_comgr_data_set_t inputs,
    const std::string& options, amd::option::Options* amdOptions,
    char* executable[], size_t* executableSize);

  //! Create the map for the kernel name and its metadata for fast access
  bool createKernelMetadataMap();
#endif

  //! Disable default copy constructor
  Program(const Program&);

  //! Disable operator=
  Program& operator=(const Program&);
};

} // namespace device
