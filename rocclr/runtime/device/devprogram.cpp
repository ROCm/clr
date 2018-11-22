//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "platform/ndrange.hpp"
#include "devprogram.hpp"
#include "devkernel.hpp"
#include "utils/macros.hpp"
#include "utils/options.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"

#if defined(WITH_LIGHTNING_COMPILER)
#include "driver/AmdCompiler.h"
#include "libraries.amdgcn.inc"
#include "opencl1.2-c.amdgcn.inc"
#include "opencl2.0-c.amdgcn.inc"
#endif  // defined(WITH_LIGHTNING_COMPILER)

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>


#if defined(ATI_OS_LINUX)
#include <dlfcn.h>
#include <libgen.h>
#endif  // defined(ATI_OS_LINUX)

#include "spirv/spirvUtils.h"
#include "acl.h"

#if defined(WITH_LIGHTNING_COMPILER)
#include "llvm/Support/AMDGPUMetadata.h"

typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;
#endif  // defined(WITH_LIGHTNING_COMPILER)

namespace device {

// ================================================================================================
Program::Program(amd::Device& device)
    : device_(device),
      type_(TYPE_NONE),
      flags_(0),
      clBinary_(nullptr),
      llvmBinary_(),
      elfSectionType_(amd::OclElf::LLVMIR),
      compileOptions_(),
      linkOptions_(),
      binaryElf_(nullptr),
      lastBuildOptionsArg_(),
      buildStatus_(CL_BUILD_NONE),
      buildError_(CL_SUCCESS),
      machineTarget_(nullptr),
      globalVariableTotalSize_(0),
      programOptions_(nullptr),
      metadata_(nullptr)
{
  memset(&binOpts_, 0, sizeof(binOpts_));
  binOpts_.struct_size = sizeof(binOpts_);
  binOpts_.elfclass = LP64_SWITCH(ELFCLASS32, ELFCLASS64);
  binOpts_.bitness = ELFDATA2LSB;
  binOpts_.alloc = &::malloc;
  binOpts_.dealloc = &::free;
}

// ================================================================================================
Program::~Program() {
  clear();
  delete metadata_;
}

// ================================================================================================
void Program::clear() {
  // Destroy all device kernels
  for (const auto& it : kernels_) {
    delete it.second;
  }
  kernels_.clear();
}

// ================================================================================================
bool Program::compileImpl(const std::string& sourceCode,
  const std::vector<const std::string*>& headers,
  const char** headerIncludeNames, amd::option::Options* options) {
  if (isLC()) {
    return compileImplLC(sourceCode, headers, headerIncludeNames, options);
  } else {
    return compileImplHSAIL(sourceCode, headers, headerIncludeNames, options);
  }
}

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static std::string llvmBin_(amd::Os::getEnvironment("LLVM_BIN"));

#if defined(ATI_OS_WIN)
static BOOL CALLBACK checkLLVM_BIN(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContex) {
  if (llvmBin_.empty()) {
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      (LPCSTR)&amd::Device::init, &hm)) {
      char path[1024];
      GetModuleFileNameA(hm, path, sizeof(path));
      llvmBin_ = path;
      size_t pos = llvmBin_.rfind('\\');
      if (pos != std::string::npos) {
        llvmBin_.resize(pos);
      }
    }
  }
  return TRUE;
}
#endif  // defined (ATI_OS_WINDOWS)

#if defined(ATI_OS_LINUX)
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void checkLLVM_BIN() {
  if (llvmBin_.empty()) {
    Dl_info info;
    if (dladdr((const void*)&amd::Device::init, &info)) {
      char* str = strdup(info.dli_fname);
      if (str) {
        llvmBin_ = dirname(str);
        free(str);
        size_t pos = llvmBin_.rfind("lib");
        if (pos != std::string::npos) {
          llvmBin_.replace(pos, 3, "bin");
        }
      }
    }
  }
#if defined(DEBUG)
  static const std::string tools[] = { "clang", "llvm-link", "ld.lld" };

  for (const std::string tool : tools) {
    std::string exePath(llvmBin_ + "/" + tool);
    struct stat buf;
    if (stat(exePath.c_str(), &buf)) {
      std::string msg(exePath + " not found");
      LogWarning(msg.c_str());
    }
    else if ((buf.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0) {
      std::string msg("Cannot execute " + exePath);
      LogWarning(msg.c_str());
    }
  }
#endif  // defined(DEBUG)
}
#endif  // defined(ATI_OS_LINUX)

std::unique_ptr<amd::opencl_driver::Compiler> Program::newCompilerInstance() {
#if defined(ATI_OS_WIN)
  static INIT_ONCE initOnce;
  InitOnceExecuteOnce(&initOnce, checkLLVM_BIN, nullptr, nullptr);
#endif  // defined(ATI_OS_WIN)
#if defined(ATI_OS_LINUX)
  pthread_once(&once, checkLLVM_BIN);
#endif  // defined(ATI_OS_LINUX)
#if defined(DEBUG)
  std::string clangExe(llvmBin_ + LINUX_SWITCH("/clang", "\\clang.exe"));
  struct stat buf;
  if (stat(clangExe.c_str(), &buf)) {
    std::string msg("Could not find the Clang binary in " + llvmBin_);
    LogWarning(msg.c_str());
  }
#endif  // defined(DEBUG)

  return std::unique_ptr<amd::opencl_driver::Compiler>(
    amd::opencl_driver::CompilerFactory().CreateAMDGPUCompiler(llvmBin_));
}
#endif // defined(WITH_LIGHTNING_COMPILER)

// ================================================================================================

#if defined(USE_COMGR_LIBRARY)
//  Extract the byte code binary from the data set.  The binary will be saved to an output
//  file if the file name is provided. If buffer pointer, outBinary, is provided, the
//  binary will be passed back to the caller.
//
void Program::extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
                                    const amd_comgr_data_kind_t dataKind,
                                    const std::string& outFileName,
                                    char* outBinary[], size_t* outSize) {
  amd_comgr_data_t  binaryData;

  amd_comgr_status_t status = amd_comgr_create_data(dataKind, &binaryData);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_action_data_get_data(inDataSet, dataKind, 0, &binaryData);
  }

  size_t binarySize = 0;
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_get_data(binaryData, &binarySize, NULL);
  }

  char* binary = static_cast<char *>(malloc(binarySize));
  if (binary == nullptr) {
    amd_comgr_release_data(binaryData);
    return;
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_get_data(binaryData, &binarySize, binary);
  }

  amd_comgr_release_data(binaryData);

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    free(binary);
    return;
  }

  // save the binary to the file as output file name is specified
  if (!outFileName.empty()) {
    std::ofstream f(outFileName.c_str(), std::ios::trunc);
    if (f.is_open()) {
      f.write(binary, binarySize);
      f.close();
    } else {
      buildLog_ += "Warning: opening the file to dump the code failed.\n";
    }
  }

  if (outBinary != nullptr) {
    // Pass the dump binary and its size back to the caller
    *outBinary = binary;
    *outSize = binarySize;
  }
  else {
    free(binary);
  }
}

amd_comgr_status_t Program::addCodeObjData(const char *source,
                                           const size_t size,
                                           const amd_comgr_data_kind_t type,
                                           const char* name,
                                           amd_comgr_data_set_t* dataSet)
{
  amd_comgr_data_t data;
  amd_comgr_status_t status;

  status = amd_comgr_create_data(type, &data);
  if (status  != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  status = amd_comgr_set_data(data, size, source);

  if ((name != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd_comgr_set_data_name(data, name);
  }

  if ((dataSet != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd_comgr_data_set_add(*dataSet, data);
  }

  amd_comgr_release_data(data);

  return status;
}

void Program::setLangAndTargetStr(const char* clStd, amd_comgr_language_t* oclver,
                                  std::string& targetIdent) {

  uint clcStd = (clStd[2] - '0') * 100 + (clStd[4] - '0') * 10;

  if (oclver != nullptr) {
    switch (clcStd) {
      case 100:
      case 110:
      case 120:
        *oclver = AMD_COMGR_LANGUAGE_OPENCL_1_2;
        break;
      case 200:
        *oclver = AMD_COMGR_LANGUAGE_OPENCL_2_0;
        break;
      default:
        *oclver = AMD_COMGR_LANGUAGE_NONE;
        break;
    }
  }

  // Set target triple and CPU
  targetIdent = std::string("amdgcn-amd-amdhsa--") + machineTarget_;

  // Set xnack option if needed
  if (xnackEnabled_) {
    targetIdent.append("+xnack");
  }
}


amd_comgr_status_t Program::createAction(const amd_comgr_language_t oclver,
                                         const std::string& targetIdent,
                                         const std::string& options,
                                         amd_comgr_action_info_t* action) {

  amd_comgr_status_t status = amd_comgr_create_action_info(action);

  if ((oclver != AMD_COMGR_LANGUAGE_NONE) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd_comgr_action_info_set_language(*action, oclver);
  }

  if (!targetIdent.empty() && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd_comgr_action_info_set_isa_name(*action, targetIdent.c_str());
  }

  if (!options.empty() && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd_comgr_action_info_set_options(*action, options.c_str());
  }

  return status;
}

bool Program::linkLLVMBitcode(const amd_comgr_data_set_t inputs,
                              const std::string& options, const bool requiredDump,
                              amd::option::Options* amdOptions, amd_comgr_data_set_t* output,
                              char* binaryData[], size_t* binarySize) {

  // get the language and target name
  std::string targetIdent;
  amd_comgr_language_t oclver;
  setLangAndTargetStr(amdOptions->oVariables->CLStd, &oclver, targetIdent);
  if (oclver == AMD_COMGR_LANGUAGE_NONE) {
    return false;
  }

  //  Create the action for linking
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t dataSetDevLibs;

  amd_comgr_status_t status = createAction(oclver, targetIdent, options, &action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_create_data_set(&dataSetDevLibs);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES, action, inputs,
                                 dataSetDevLibs);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, action, dataSetDevLibs, *output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string dumpFileName;
    if (requiredDump && amdOptions != nullptr &&
        amdOptions->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
      dumpFileName = amdOptions->getDumpFileName("_linked.bc");
    }
    extractByteCodeBinary(*output, AMD_COMGR_DATA_KIND_BC, dumpFileName, binaryData, binarySize);
  }

  amd_comgr_destroy_action_info(action);
  amd_comgr_destroy_data_set(dataSetDevLibs);

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Program::compileToLLVMBitcode(const amd_comgr_data_set_t inputs,
                                   const std::string& options, amd::option::Options* amdOptions,
                                   char* binaryData[], size_t* binarySize) {

  //  get the lanuage and target name
  std::string targetIdent;
  amd_comgr_language_t oclver;
  setLangAndTargetStr(amdOptions->oVariables->CLStd, &oclver, targetIdent);
  if (oclver == AMD_COMGR_LANGUAGE_NONE) {
    return false;
  }

  //  Create the output data set
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t dataSetPCH;

  amd_comgr_status_t status = createAction(oclver, targetIdent, options, &action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_create_data_set(&output);
  }

  //  Adding Precompiled Headers
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_create_data_set(&dataSetPCH);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS,
                                 action, inputs, dataSetPCH);
  }

  //  Compiling the source codes with precompiled headers
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                                 action, dataSetPCH, output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string outFileName;
    if (amdOptions->isDumpFlagSet(amd::option::DUMP_BC_ORIGINAL)) {
       outFileName = amdOptions->getDumpFileName("_original.bc");
    }
    extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, outFileName, binaryData, binarySize);
  }

  amd_comgr_destroy_action_info(action);
  amd_comgr_destroy_data_set(dataSetPCH);
  amd_comgr_destroy_data_set(output);

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

//  Create an executable from an input data set.  To generate the executable,
//  the input data set is converted to relocatable code, then executable binary.
//  If assembly code is required, the input data set is converted to assembly.
bool Program::compileAndLinkExecutable(const amd_comgr_data_set_t inputs,
                                       const std::string& options, amd::option::Options* amdOptions,
                                       char* executable[], size_t* executableSize) {

  //  get the language and target name
  std::string targetIdent;
  setLangAndTargetStr(amdOptions->oVariables->CLStd, nullptr, targetIdent);

  // create the linked output
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t relocatableData;

  amd_comgr_status_t status = createAction(AMD_COMGR_LANGUAGE_NONE, targetIdent, options, &action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_create_data_set(&output);
  }

  if ((amdOptions->isDumpFlagSet(amd::option::DUMP_ISA)) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    //  create the assembly data set
    amd_comgr_data_set_t assemblyData;
    status = amd_comgr_create_data_set(&assemblyData);
    if (status == AMD_COMGR_STATUS_SUCCESS) {
      status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                                   action, inputs, assemblyData);
    }

    // dump the ISA
    if (status == AMD_COMGR_STATUS_SUCCESS) {
      std::string dumpIsaName = amdOptions->getDumpFileName(".s");
      extractByteCodeBinary(assemblyData, AMD_COMGR_DATA_KIND_SOURCE, dumpIsaName);
    }
    amd_comgr_destroy_data_set(assemblyData);
  }

  //  Create the relocatiable data set
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_create_data_set(&relocatableData);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                                 action, inputs, relocatableData);
  }

  // Create executable from the relocatable data set
  amd_comgr_action_info_set_options(action, "");
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                                 action, relocatableData, output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    // Extract the executable binary
    std::string outFileName;
    if (amdOptions->isDumpFlagSet(amd::option::DUMP_O)) {
      outFileName = amdOptions->getDumpFileName(".so");
    }
    extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_EXECUTABLE, outFileName, executable,
                          executableSize);
  }

  amd_comgr_destroy_action_info(action);
  amd_comgr_destroy_data_set(relocatableData);
  amd_comgr_destroy_data_set(output);

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Program::compileImplLC(const std::string& sourceCode,
                            const std::vector<const std::string*>& headers,
                            const char** headerIncludeNames, amd::option::Options* options) {
  const char* xLang = options->oVariables->XLang;
  if (xLang != nullptr) {
    if (strcmp(xLang,"asm") == 0) {
      clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, sourceCode.data(), sourceCode.size());
      return true;
    } else if (!strcmp(xLang,"cl")) {
      buildLog_ += "Unsupported language: \"" + std::string(xLang) + "\".\n";
      return false;
    }
  }

  // add CL source to input data set
  amd_comgr_data_set_t inputs;

  if (amd_comgr_create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create output buffer for LLVM bitcode.\n";
    return false;
  }

  if (addCodeObjData(sourceCode.c_str(), sourceCode.length(), AMD_COMGR_DATA_KIND_SOURCE,
                     "CompileCLSource", &inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data from CL source.\n";
    amd_comgr_destroy_data_set(inputs);
    return false;
  }

  // Set the options for the compiler
  // Some options are set in Clang AMDGPUToolChain (like -m64)
  std::ostringstream ostrstr;
  std::copy(options->clangOptions.begin(), options->clangOptions.end(),
            std::ostream_iterator<std::string>(ostrstr, " "));

  std::string driverOptions(ostrstr.str());

  // Set the -O#
  std::ostringstream optLevel;
  optLevel << " -O" << options->oVariables->OptLevel;
  driverOptions.append(optLevel.str());

  driverOptions.append(options->llvmOptions);
  driverOptions.append(ProcessOptions(options));

  // Set whole program mode
  driverOptions.append(" -mllvm -amdgpu-early-inline-all -mllvm -amdgpu-prelink");

  // Iterate through each source code and dump it into tmp
  std::fstream f;
  std::vector<std::string> headerFileNames(headers.size());
  std::vector<std::string> newDirs;

  if (!headers.empty()) {
    for (size_t i = 0; i < headers.size(); ++i) {
      std::string headerName="Header" + std::to_string(i);
      if (addCodeObjData(headers[i]->c_str(), headers[i]->length(), AMD_COMGR_DATA_KIND_INCLUDE,
                         headerName.c_str(), &inputs) != AMD_COMGR_STATUS_SUCCESS) {
        buildLog_ += "Error: COMGR fails to add headers into inputs.\n";
        amd_comgr_destroy_data_set(inputs);
        return false;
      }
    }
  }

  if (options->isDumpFlagSet(amd::option::DUMP_CL)) {
    std::ofstream f(options->getDumpFileName(".cl").c_str(), std::ios::trunc);
    if (f.is_open()) {
      f << "/* Compiler options:\n"
           "-c -emit-llvm -target amdgcn-amd-amdhsa -x cl "
        << driverOptions << " -include opencl-c.h "
        << "\n*/\n\n"
        << sourceCode;
      f.close();
    } else {
      buildLog_ += "Warning: opening the file to dump the OpenCL source failed.\n";
    }
  }

  // Compile source to IR
  char* binaryData = nullptr;
  size_t binarySize = 0;
  bool ret = compileToLLVMBitcode(inputs, driverOptions, options, &binaryData, &binarySize);
  if (ret) {
    llvmBinary_.assign(binaryData, binarySize);
    elfSectionType_ = amd::OclElf::LLVMIR;

    if (clBinary()->saveSOURCE()) {
      clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, sourceCode.data(), sourceCode.size());
    }
    if (clBinary()->saveLLVMIR()) {
      clBinary()->elfOut()->addSection(amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(),
                                       false);
      // store the original compile options
      clBinary()->storeCompileOptions(compileOptions_);
    }
  }
  else {
    buildLog_ += "Error: Failed to compile opencl source (from CL to LLVM IR).\n";
  }

  amd_comgr_destroy_data_set(inputs);
  return ret;
}
#else // not using COMgr
bool Program::compileImplLC(const std::string& sourceCode,
  const std::vector<const std::string*>& headers,
  const char** headerIncludeNames, amd::option::Options* options) {
#if defined(WITH_LIGHTNING_COMPILER)
  const char* xLang = options->oVariables->XLang;
  if (xLang != nullptr) {
    if (strcmp(xLang, "asm") == 0) {
      clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, sourceCode.data(), sourceCode.size());
      return true;
    }
    else if (!strcmp(xLang, "cl")) {
      buildLog_ += "Unsupported language: \"" + std::string(xLang) + "\".\n";
      return false;
    }
  }

  using namespace amd::opencl_driver;
  std::unique_ptr<Compiler> C(newCompilerInstance());
  std::vector<Data*> inputs;

  Data* input = C->NewBufferReference(DT_CL, sourceCode.c_str(), sourceCode.length());
  if (input == nullptr) {
    buildLog_ += "Error while creating data from source code";
    return false;
  }

  inputs.push_back(input);

  amd::opencl_driver::Buffer* output = C->NewBuffer(DT_LLVM_BC);
  if (output == nullptr) {
    buildLog_ += "Error while creating buffer for the LLVM bitcode";
    return false;
  }

  // Set the options for the compiler
  // Some options are set in Clang AMDGPUToolChain (like -m64)
  std::ostringstream ostrstr;
  std::copy(options->clangOptions.begin(), options->clangOptions.end(),
    std::ostream_iterator<std::string>(ostrstr, " "));

  std::string driverOptions(ostrstr.str());

  // Setting the language
  driverOptions.append(" -cl-std=").append(options->oVariables->CLStd);

  // Set the -O#
  std::ostringstream optLevel;
  optLevel << " -O" << options->oVariables->OptLevel;
  driverOptions.append(optLevel.str());

  // Set the machine target
  driverOptions.append(" -mcpu=");
  driverOptions.append(machineTarget_);

  // Set xnack option if needed
  if (xnackEnabled_) {
    driverOptions.append(" -mxnack");
  }

  driverOptions.append(options->llvmOptions);
  driverOptions.append(ProcessOptions(options));

  // Set whole program mode
  driverOptions.append(" -mllvm -amdgpu-early-inline-all -mllvm -amdgpu-prelink");

  // Find the temp folder for the OS
  std::string tempFolder = amd::Os::getTempPath();

  // Iterate through each source code and dump it into tmp
  std::fstream f;
  std::vector<std::string> headerFileNames(headers.size());
  std::vector<std::string> newDirs;
  for (size_t i = 0; i < headers.size(); ++i) {
    std::string headerPath = tempFolder;
    std::string headerIncludeName(headerIncludeNames[i]);
    // replace / in path with current os's file separator
    if (amd::Os::fileSeparator() != '/') {
      for (auto& it : headerIncludeName) {
        if (it == '/') it = amd::Os::fileSeparator();
      }
    }
    size_t pos = headerIncludeName.rfind(amd::Os::fileSeparator());
    if (pos != std::string::npos) {
      headerPath += amd::Os::fileSeparator();
      headerPath += headerIncludeName.substr(0, pos);
      headerIncludeName = headerIncludeName.substr(pos + 1);
    }
    if (!amd::Os::pathExists(headerPath)) {
      bool ret = amd::Os::createPath(headerPath);
      assert(ret && "failed creating path!");
      newDirs.push_back(headerPath);
    }
    std::string headerFullName = headerPath + amd::Os::fileSeparator() + headerIncludeName;
    headerFileNames[i] = headerFullName;
    f.open(headerFullName.c_str(), std::fstream::out);
    // Should we allow asserts
    assert(!f.fail() && "failed creating header file!");
    f.write(headers[i]->c_str(), headers[i]->length());
    f.close();

    Data* inc = C->NewFileReference(DT_CL_HEADER, headerFileNames[i]);
    if (inc == nullptr) {
      buildLog_ += "Error while creating data from headers";
      return false;
    }
    inputs.push_back(inc);
  }

  // Set the include path for the temp folder that contains the includes
  if (!headers.empty()) {
    driverOptions.append(" -I");
    driverOptions.append(tempFolder);
  }

  if (options->isDumpFlagSet(amd::option::DUMP_CL)) {
    std::ofstream f(options->getDumpFileName(".cl").c_str(), std::ios::trunc);
    if (f.is_open()) {
      f << "/* Compiler options:\n"
        "-c -emit-llvm -target amdgcn-amd-amdhsa -x cl "
        << driverOptions << " -include opencl-c.h "
        << "\n*/\n\n"
        << sourceCode;
      f.close();
    }
    else {
      buildLog_ += "Warning: opening the file to dump the OpenCL source failed.\n";
    }
  }

  uint clcStd =
    (options->oVariables->CLStd[2] - '0') * 100 + (options->oVariables->CLStd[4] - '0') * 10;

  std::pair<const void*, size_t> hdr;
  switch (clcStd) {
  case 100:
  case 110:
  case 120:
    hdr = { opencl1_2_c_amdgcn, opencl1_2_c_amdgcn_size };
    break;
  case 200:
    hdr = { opencl2_0_c_amdgcn, opencl2_0_c_amdgcn_size };
    break;
  default:
    buildLog_ += "Unsupported requested OpenCL C version (-cl-std).\n";
    return false;
  }

  File* pch = C->NewTempFile(DT_CL_HEADER);
  if (pch == nullptr || !pch->WriteData((const char*)hdr.first, hdr.second)) {
    buildLog_ += "Error while opening the opencl-c header ";
    return false;
  }

  driverOptions.append(" -include-pch " + pch->Name());
  driverOptions.append(" -Xclang -fno-validate-pch");
  driverOptions.append(" -Xclang -target-feature -Xclang -code-object-v3");

  // Tokenize the options string into a vector of strings
  std::istringstream istrstr(driverOptions);
  std::istream_iterator<std::string> sit(istrstr), end;
  std::vector<std::string> params(sit, end);

  // Compile source to IR
  bool ret =
    device().cacheCompilation()->compileToLLVMBitcode(C.get(), inputs, output, params, buildLog_);
  buildLog_ += C->Output();
  if (!ret) {
    buildLog_ += "Error: Failed to compile opencl source (from CL to LLVM IR).\n";
    return false;
  }

  llvmBinary_.assign(output->Buf().data(), output->Size());
  elfSectionType_ = amd::OclElf::LLVMIR;

  if (options->isDumpFlagSet(amd::option::DUMP_BC_ORIGINAL)) {
    std::ofstream f(options->getDumpFileName("_original.bc").c_str(),
      std::ios::binary | std::ios::trunc);
    if (f.is_open()) {
      f.write(llvmBinary_.data(), llvmBinary_.size());
      f.close();
    }
    else {
      buildLog_ += "Warning: opening the file to dump the compiled IR failed.\n";
    }
  }

  if (clBinary()->saveSOURCE()) {
    clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, sourceCode.data(), sourceCode.size());
  }
  if (clBinary()->saveLLVMIR()) {
    clBinary()->elfOut()->addSection(amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(),
      false);
    // store the original compile options
    clBinary()->storeCompileOptions(compileOptions_);
  }
#endif // defined(WITH_LIGHTNING_COMPILER)
  return true;
}
#endif // defined(USE_COMGR_LIBRARY)

// ================================================================================================
static void logFunction(const char* msg, size_t size) {
  std::cout << "Compiler Log: " << msg << std::endl;
}

// ================================================================================================
bool Program::compileImplHSAIL(const std::string& sourceCode,
  const std::vector<const std::string*>& headers,
  const char** headerIncludeNames, amd::option::Options* options) {
#if defined(WITH_COMPILER_LIB)
  acl_error errorCode;
  aclTargetInfo target;

  std::string arch = LP64_SWITCH("hsail", "hsail64");
  target = aclGetTargetInfo(arch.c_str(), machineTarget_, &errorCode);

  // end if asic info is ready
  // We dump the source code for each program (param: headers)
  // into their filenames (headerIncludeNames) into the TEMP
  // folder specific to the OS and add the include path while
  // compiling

  // Find the temp folder for the OS
  std::string tempFolder = amd::Os::getTempPath();

  // Iterate through each source code and dump it into tmp
  std::fstream f;
  std::vector<std::string> newDirs;
  for (size_t i = 0; i < headers.size(); ++i) {
    std::string headerPath = tempFolder;
    std::string headerIncludeName(headerIncludeNames[i]);
    // replace / in path with current os's file separator
    if (amd::Os::fileSeparator() != '/') {
      for (auto& it : headerIncludeName) {
        if (it == '/') it = amd::Os::fileSeparator();
      }
    }
    size_t pos = headerIncludeName.rfind(amd::Os::fileSeparator());
    if (pos != std::string::npos) {
      headerPath += amd::Os::fileSeparator();
      headerPath += headerIncludeName.substr(0, pos);
      headerIncludeName = headerIncludeName.substr(pos + 1);
    }
    if (!amd::Os::pathExists(headerPath)) {
      bool ret = amd::Os::createPath(headerPath);
      assert(ret && "failed creating path!");
      newDirs.push_back(headerPath);
    }
    std::string headerFullName = headerPath + amd::Os::fileSeparator() + headerIncludeName;
    f.open(headerFullName.c_str(), std::fstream::out);
    // Should we allow asserts
    assert(!f.fail() && "failed creating header file!");
    f.write(headers[i]->c_str(), headers[i]->length());
    f.close();
  }

  // Create Binary
  binaryElf_ = aclBinaryInit(sizeof(aclBinary), &target, &binOpts_, &errorCode);
  if (errorCode != ACL_SUCCESS) {
    buildLog_ += "Error: aclBinary init failure\n";
    LogWarning("aclBinaryInit failed");
    return false;
  }

  // Insert opencl into binary
  errorCode = aclInsertSection(device().compiler(), binaryElf_, sourceCode.c_str(),
    strlen(sourceCode.c_str()), aclSOURCE);
  if (errorCode != ACL_SUCCESS) {
    buildLog_ += "Error: Inserting openCl Source to binary\n";
  }

  // Set the options for the compiler
  // Set the include path for the temp folder that contains the includes
  if (!headers.empty()) {
    compileOptions_.append(" -I");
    compileOptions_.append(tempFolder);
  }

#if !defined(_LP64) && defined(ATI_OS_LINUX)
  if (options->origOptionStr.find("-cl-std=CL2.0") != std::string::npos) {
    errorCode = ACL_UNSUPPORTED;
    LogWarning("aclCompile failed");
    return false;
  }
#endif

  // Compile source to IR
  compileOptions_.append(ProcessOptions(options));
  errorCode = aclCompile(device().compiler(), binaryElf_, compileOptions_.c_str(), ACL_TYPE_OPENCL,
    ACL_TYPE_LLVMIR_BINARY, nullptr /* logFunction */);
  buildLog_ += aclGetCompilerLog(device().compiler());
  if (errorCode != ACL_SUCCESS) {
    LogWarning("aclCompile failed");
    buildLog_ += "Error: Compiling CL to IR\n";
    return false;
  }

  clBinary()->storeCompileOptions(compileOptions_);

  // Save the binary in the interface class
  saveBinaryAndSetType(TYPE_COMPILED);
#endif  // defined(WITH_COMPILER_LIB)
  return true;
}

// ================================================================================================
bool Program::linkImpl(const std::vector<device::Program*>& inputPrograms,
  amd::option::Options* options, bool createLibrary) {
  if (isLC()) {
    return linkImplLC(inputPrograms, options, createLibrary);
  }
  else {
    return linkImplHSAIL(inputPrograms, options, createLibrary);
  }
}

// ================================================================================================
#if defined(USE_COMGR_LIBRARY)
bool Program::linkImplLC(const std::vector<Program*>& inputPrograms,
                         amd::option::Options* options, bool createLibrary) {
  using namespace amd::opencl_driver;
  std::unique_ptr<Compiler> C(newCompilerInstance());

  amd_comgr_data_set_t inputs;

  if (amd_comgr_create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data set.\n";
    return false;
  }

  size_t idx = 0;
  for (auto program : inputPrograms) {
    bool result = true;
    if (program->llvmBinary_.empty()) {
      result = (program->clBinary() != nullptr);
      if (result) {
        // We are using CL binary directly.
        // Setup elfIn() and try to load llvmIR from binary
        // This elfIn() will be released at the end of build by finiBuild().
        result = program->clBinary()->setElfIn();
      }

      if (result) {
        result = program->clBinary()->loadLlvmBinary(program->llvmBinary_,
                                                     program->elfSectionType_);
      }
    }

    if (result) {
      result = (program->elfSectionType_ == amd::OclElf::LLVMIR);
    }

    if (result) {
      result = (addCodeObjData(program->llvmBinary_.data(), program->llvmBinary_.size(),
                               AMD_COMGR_DATA_KIND_BC, "LLVM Binary " + idx, &inputs) ==
                               AMD_COMGR_STATUS_SUCCESS);
    }

    if (!result) {
      amd_comgr_destroy_data_set(inputs);
      buildLog_ += "Error: Linking bitcode failed: failing to generate LLVM binary.\n";
      return false;
    }

    idx++;

    // release elfIn() for the program
    program->clBinary()->resetElfIn();
  }

  // create the linked output
  amd_comgr_data_set_t output;
  if (amd_comgr_create_data_set(&output) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create output buffer for LLVM bitcode.\n";
    amd_comgr_destroy_data_set(inputs);
    return false;
  }

  // NOTE: The options parameter is also used to identy cached code object.
  //       This parameter should not contain any dyanamically generated filename.
  char* binaryData = nullptr;
  size_t binarySize = 0;
  std::string linkOptions;
  bool ret = linkLLVMBitcode(inputs, linkOptions, false, options, &output, &binaryData,
                             &binarySize);

  amd_comgr_destroy_data_set(output);
  amd_comgr_destroy_data_set(inputs);

  if (!ret) {
    buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
    return false;
  }

  llvmBinary_.assign(binaryData, binarySize);
  elfSectionType_ = amd::OclElf::LLVMIR;

  if (clBinary()->saveLLVMIR()) {
    clBinary()->elfOut()->addSection(amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(),
                                     false);
    // store the original link options
    clBinary()->storeLinkOptions(linkOptions_);
    // store the original compile options
    clBinary()->storeCompileOptions(compileOptions_);
  }

  // skip the rest if we are building an opencl library
  if (createLibrary) {
    setType(TYPE_LIBRARY);
    if (!createBinary(options)) {
      buildLog_ += "Internal error: creating OpenCL binary failed\n";
      return false;
    }
    return true;
  }

  return linkImpl(options);
}

#else // not using COMgr
bool Program::linkImplLC(const std::vector<Program*>& inputPrograms,
  amd::option::Options* options, bool createLibrary) {
#if defined(WITH_LIGHTNING_COMPILER)
  using namespace amd::opencl_driver;
  std::unique_ptr<Compiler> C(newCompilerInstance());

  std::vector<Data*> inputs;
  for (auto program : inputPrograms) {
    if (program->llvmBinary_.empty()) {
      if (program->clBinary() == NULL) {
        buildLog_ += "Internal error: Input program not compiled!\n";
        return false;
      }

      // We are using CL binary directly.
      // Setup elfIn() and try to load llvmIR from binary
      // This elfIn() will be released at the end of build by finiBuild().
      if (!program->clBinary()->setElfIn()) {
        buildLog_ += "Internal error: Setting input OCL binary failed!\n";
        return false;
      }
      if (!program->clBinary()->loadLlvmBinary(program->llvmBinary_, program->elfSectionType_)) {
        buildLog_ += "Internal error: Failed loading compiled binary!\n";
        return false;
      }
    }

    if (program->elfSectionType_ != amd::OclElf::LLVMIR) {
      buildLog_ += "Error: Input binary format is not supported\n.";
      return false;
    }

    Data* input = C->NewBufferReference(DT_LLVM_BC, (const char*)program->llvmBinary_.data(),
      program->llvmBinary_.size());

    if (!input) {
      buildLog_ += "Internal error: Failed to open the compiled programs.\n";
      return false;
    }

    // release elfIn() for the program
    program->clBinary()->resetElfIn();

    inputs.push_back(input);
  }

  // open the linked output
  amd::opencl_driver::Buffer* output = C->NewBuffer(DT_LLVM_BC);

  if (!output) {
    buildLog_ += "Error: Failed to open the linked program.\n";
    return false;
  }

  std::vector<std::string> linkOptions;

  // NOTE: The params is also used to identy cached code object.  This parameter
  //       should not contain any dyanamically generated filename.
  bool ret = device().cacheCompilation()->linkLLVMBitcode(
    C.get(), inputs, output, linkOptions, buildLog_);
  buildLog_ += C->Output();
  if (!ret) {
    buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
    return false;
  }

  llvmBinary_.assign(output->Buf().data(), output->Size());
  elfSectionType_ = amd::OclElf::LLVMIR;

  if (clBinary()->saveLLVMIR()) {
    clBinary()->elfOut()->addSection(amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(),
      false);
    // store the original link options
    clBinary()->storeLinkOptions(linkOptions_);
    // store the original compile options
    clBinary()->storeCompileOptions(compileOptions_);
  }

  // skip the rest if we are building an opencl library
  if (createLibrary) {
    setType(TYPE_LIBRARY);
    if (!createBinary(options)) {
      buildLog_ += "Internal error: creating OpenCL binary failed\n";
      return false;
    }
    return true;
  }

  return linkImpl(options);
#else
  return false;
#endif // defined(WITH_LIGHTNING_COMPILER)
}
#endif // defined(USE_COMGR_LIBRARY)

// ================================================================================================
bool Program::linkImplHSAIL(const std::vector<Program*>& inputPrograms,
  amd::option::Options* options, bool createLibrary) {
#if  defined(WITH_COMPILER_LIB)
  acl_error errorCode;

  // For each program we need to extract the LLVMIR and create
  // aclBinary for each
  std::vector<aclBinary*> binaries_to_link;

  for (auto program : inputPrograms) {
    // Check if the program was created with clCreateProgramWIthBinary
    binary_t binary = program->binary();
    if ((binary.first != nullptr) && (binary.second > 0)) {
      // Binary already exists -- we can also check if there is no
      // opencl source code
      // Need to check if LLVMIR exists in the binary
      // If LLVMIR does not exist then is it valid
      // We need to pull out all the compiled kernels
      // We cannot do this at present because we need at least
      // Hsail text to pull the kernels oout
      void* mem = const_cast<void*>(binary.first);
      binaryElf_ = aclReadFromMem(mem, binary.second, &errorCode);
      if (errorCode != ACL_SUCCESS) {
        LogWarning("Error while linking : Could not read from raw binary");
        return false;
      }
    }

    // At this stage each Program contains a valid binary_elf
    // Check if LLVMIR is in the binary
    size_t boolSize = sizeof(bool);
    bool containsLLLVMIR = false;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_LLVMIR,
      nullptr, &containsLLLVMIR, &boolSize);

    if (errorCode != ACL_SUCCESS || !containsLLLVMIR) {
      bool spirv = false;
      size_t boolSize = sizeof(bool);
      errorCode = aclQueryInfo(
        device().compiler(), binaryElf_, RT_CONTAINS_SPIRV, nullptr, &spirv, &boolSize);
      if (errorCode != ACL_SUCCESS) {
        spirv = false;
      }
      if (spirv) {
        errorCode = aclCompile(
          device().compiler(), binaryElf_, options->origOptionStr.c_str(),
          ACL_TYPE_SPIRV_BINARY, ACL_TYPE_LLVMIR_BINARY, nullptr);
        buildLog_ += aclGetCompilerLog(device().compiler());
        if (errorCode != ACL_SUCCESS) {
          buildLog_ += "Error while linking: Could not load SPIR-V";
          return false;
        }
      }
      else {
        buildLog_ += "Error while linking : Invalid binary (Missing LLVMIR section)";
        return false;
      }
    }
    // Create a new aclBinary for each LLVMIR and save it in a list
    aclBIFVersion ver = aclBinaryVersion(binaryElf_);
    aclBinary* bin = aclCreateFromBinary(binaryElf_, ver);
    binaries_to_link.push_back(bin);
  }

  errorCode = aclLink(device().compiler(), binaries_to_link[0], binaries_to_link.size() - 1,
    binaries_to_link.size() > 1 ? &binaries_to_link[1] : nullptr,
    ACL_TYPE_LLVMIR_BINARY, "-create-library", nullptr);
  if (errorCode != ACL_SUCCESS) {
    buildLog_ += aclGetCompilerLog(device().compiler());
    buildLog_ += "Error while linking : aclLink failed";
    return false;
  }
  // Store the newly linked aclBinary for this program.
  binaryElf_ = binaries_to_link[0];
  // Free all the other aclBinaries
  for (size_t i = 1; i < binaries_to_link.size(); i++) {
    aclBinaryFini(binaries_to_link[i]);
  }
  if (createLibrary) {
    saveBinaryAndSetType(TYPE_LIBRARY);
    buildLog_ += aclGetCompilerLog(device().compiler());
    return true;
  }

  // Now call linkImpl with the new options
  return linkImpl(options);
#else
  return false;
#endif  // defined(WITH_COMPILER_LIB)
}

// ================================================================================================
bool Program::linkImpl(amd::option::Options* options) {
  if (isLC()) {
    return linkImplLC(options);
  }
  else {
    return linkImplHSAIL(options);
  }
}

// ================================================================================================
#if defined(USE_COMGR_LIBRARY)
bool Program::linkImplLC(amd::option::Options* options) {
  acl_error errorCode;
  aclType continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;

  internal_ = (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ?
    true : false;

  amd_comgr_data_set_t inputs;
  if (amd_comgr_create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data set for linking.\n";
    return false;
  }

  bool bLinkLLVMBitcode = true;
  if (llvmBinary_.empty()) {
    continueCompileFrom = getNextCompilationStageFromBinary(options);
  }

  switch (continueCompileFrom) {
    case ACL_TYPE_CG:
    case ACL_TYPE_LLVMIR_BINARY: {
      break;
    }
    case ACL_TYPE_ASM_TEXT: {
      char* section;
      size_t sz;
      clBinary()->elfOut()->getSection(amd::OclElf::SOURCE, &section, &sz);

      if (addCodeObjData(section, sz, AMD_COMGR_DATA_KIND_BC, "Assembly Text",
                         &inputs) != AMD_COMGR_STATUS_SUCCESS) {
        buildLog_ += "Error: COMGR fails to create assembly input.\n";
        amd_comgr_destroy_data_set(inputs);
        return false;
      }

      bLinkLLVMBitcode = false;
      break;
    }
    case ACL_TYPE_ISA: {
      amd_comgr_destroy_data_set(inputs);
      binary_t isaBinary = binary();
      return setKernels(options, const_cast<void *>(isaBinary.first), isaBinary.second);
      break;
    }
    default:
      buildLog_ += "Error while Codegen phase: the binary is incomplete \n";
      amd_comgr_destroy_data_set(inputs);
      return false;
  }

  // call LinkLLVMBitcode
  if (bLinkLLVMBitcode) {
    // open the bitcode libraries
    std::string linkOptions;

    if (options->oVariables->FP32RoundDivideSqrt) {
        linkOptions += "correctly_rounded_sqrt,";
    }
    if (options->oVariables->DenormsAreZero || AMD_GPU_FORCE_SINGLE_FP_DENORM == 0 ||
        (device().info().gfxipVersion_ < 900 && AMD_GPU_FORCE_SINGLE_FP_DENORM < 0)) {
        linkOptions += "daz_opt,";
    }
    if (options->oVariables->FiniteMathOnly || options->oVariables->FastRelaxedMath) {
        linkOptions += "finite_only,";
    }
    if (options->oVariables->UnsafeMathOpt || options->oVariables->FastRelaxedMath) {
        linkOptions += "unsafe_math,";
    }
    if (!linkOptions.empty()) {
        linkOptions.pop_back();     // remove the last comma
    }

    amd_comgr_status_t status = addCodeObjData(llvmBinary_.data(), llvmBinary_.size(),
                                               AMD_COMGR_DATA_KIND_BC,
                                               "LLVM Binary", &inputs);

    amd_comgr_data_set_t linked_bc;
    if (status == AMD_COMGR_STATUS_SUCCESS) {
      status = amd_comgr_create_data_set(&linked_bc);
    }

    bool ret = (status == AMD_COMGR_STATUS_SUCCESS);
    if (ret) {
      ret = linkLLVMBitcode(inputs, linkOptions, true, options, &linked_bc);
    }

    amd_comgr_destroy_data_set(inputs);

    if (!ret) {
      amd_comgr_destroy_data_set(linked_bc);
      buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
      return false;
    }

    inputs = linked_bc;
  }

  std::string codegenOptions(options->llvmOptions);

  // Set the -O#
  std::ostringstream optLevel;
  optLevel << "-O" << options->oVariables->OptLevel;
  codegenOptions.append(" ").append(optLevel.str());

  // Pass clang options
  std::ostringstream ostrstr;
  std::copy(options->clangOptions.begin(), options->clangOptions.end(),
            std::ostream_iterator<std::string>(ostrstr, " "));
  codegenOptions.append(" ").append(ostrstr.str());

  // Set whole program mode
  codegenOptions.append(" -mllvm -amdgpu-internalize-symbols -mllvm -amdgpu-early-inline-all");
  // Force object code v2.
  codegenOptions.append(" -mno-code-object-v3");

  // NOTE: The params is also used to identy cached code object. This parameter
  //       should not contain any dyanamically generated filename.
  char* executable = nullptr;
  size_t executableSize = 0;
  bool ret = compileAndLinkExecutable(inputs, codegenOptions, options, &executable,
                                      &executableSize);
  amd_comgr_destroy_data_set(inputs);

  if (!ret) {
    if (continueCompileFrom == ACL_TYPE_ASM_TEXT) {
      buildLog_ += "Error: Creating the executable from ISA assembly text failed.\n";
    } else {
      buildLog_ += "Error: Creating the executable from LLVM IRs failed.\n";
    }
    return false;
  }

  if (!setKernels(options, executable, executableSize)) {
    return false;
  }

  // Save the binary and type
  clBinary()->saveBIFBinary(reinterpret_cast<const char*>(executable), executableSize);
  setType(TYPE_EXECUTABLE);

  return true;
}
#else // not using COMgr
bool Program::linkImplLC(amd::option::Options* options) {
#if defined(WITH_LIGHTNING_COMPILER)
  using namespace amd::opencl_driver;
  internal_ = (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ?
    true : false;
  std::vector<Data*> inputs;
  std::unique_ptr<Compiler> C(newCompilerInstance());
  bool bLinkLLVMBitcode = true;
  aclType continueCompileFrom = llvmBinary_.empty() ?
    getNextCompilationStageFromBinary(options) : ACL_TYPE_LLVMIR_BINARY;

  switch (continueCompileFrom) {
  case ACL_TYPE_CG:
  case ACL_TYPE_LLVMIR_BINARY: {
    break;
  }
  case ACL_TYPE_ASM_TEXT: {
    char* section;
    size_t sz;
    clBinary()->elfOut()->getSection(amd::OclElf::SOURCE, &section, &sz);
    Data* input = C->NewBufferReference(DT_ASSEMBLY, section, sz);
    if (!input) {
      buildLog_ += "Error: Failed to open the assembler text.\n";
      return false;
    }
    inputs.push_back(input);
    bLinkLLVMBitcode = false;
    break;
  }
  case ACL_TYPE_ISA: {
    binary_t isaBinary = binary();
    return setKernels(options, (void*)isaBinary.first, isaBinary.second);
    break;
  }
  default:
    buildLog_ += "Error while Codegen phase: the binary is incomplete \n";
    return false;
  }

  // call LinkLLVMBitcode
  if (bLinkLLVMBitcode) {
    // open the input IR source
    Data* input = C->NewBufferReference(DT_LLVM_BC, llvmBinary_.data(), llvmBinary_.size());

    if (!input) {
      buildLog_ += "Error: Failed to open the compiled program.\n";
      return false;
    }

    inputs.push_back(input);  // must be the first input
                              // open the bitcode libraries
    Data* opencl_bc =
      C->NewBufferReference(DT_LLVM_BC, (const char*)opencl_amdgcn, opencl_amdgcn_size);
    Data* ocml_bc = C->NewBufferReference(DT_LLVM_BC, (const char*)ocml_amdgcn, ocml_amdgcn_size);
    Data* ockl_bc = C->NewBufferReference(DT_LLVM_BC, (const char*)ockl_amdgcn, ockl_amdgcn_size);

    if (!opencl_bc || !ocml_bc || !ockl_bc) {
      buildLog_ += "Error: Failed to open the bitcode library.\n";
      return false;
    }

    inputs.push_back(opencl_bc);  // depends on oclm & ockl
    inputs.push_back(ockl_bc);
    inputs.push_back(ocml_bc);

    // open the control functions
    auto isa_version = get_oclc_isa_version(device().info().gfxipVersion_);
    if (!isa_version.first) {
      buildLog_ += "Error: Linking for this device is not supported\n";
      return false;
    }

    Data* isa_version_bc =
      C->NewBufferReference(DT_LLVM_BC, (const char*)isa_version.first, isa_version.second);

    if (!isa_version_bc) {
      buildLog_ += "Error: Failed to open the control functions.\n";
      return false;
    }

    inputs.push_back(isa_version_bc);

    auto correctly_rounded_sqrt =
      get_oclc_correctly_rounded_sqrt(options->oVariables->FP32RoundDivideSqrt);
    Data* correctly_rounded_sqrt_bc = C->NewBufferReference(DT_LLVM_BC, correctly_rounded_sqrt.first,
      correctly_rounded_sqrt.second);

    auto daz_opt = get_oclc_daz_opt(options->oVariables->DenormsAreZero ||
      AMD_GPU_FORCE_SINGLE_FP_DENORM == 0 ||
      (device().info().gfxipVersion_ < 900 && AMD_GPU_FORCE_SINGLE_FP_DENORM < 0));
    Data* daz_opt_bc = C->NewBufferReference(DT_LLVM_BC, daz_opt.first, daz_opt.second);

    auto finite_only = get_oclc_finite_only(options->oVariables->FiniteMathOnly ||
      options->oVariables->FastRelaxedMath);
    Data* finite_only_bc = C->NewBufferReference(DT_LLVM_BC, finite_only.first, finite_only.second);

    auto unsafe_math = get_oclc_unsafe_math(options->oVariables->UnsafeMathOpt ||
      options->oVariables->FastRelaxedMath);
    Data* unsafe_math_bc = C->NewBufferReference(DT_LLVM_BC, unsafe_math.first, unsafe_math.second);

    if (!correctly_rounded_sqrt_bc || !daz_opt_bc || !finite_only_bc || !unsafe_math_bc) {
      buildLog_ += "Error: Failed to open the control functions.\n";
      return false;
    }

    inputs.push_back(correctly_rounded_sqrt_bc);
    inputs.push_back(daz_opt_bc);
    inputs.push_back(finite_only_bc);
    inputs.push_back(unsafe_math_bc);

    // open the linked output
    std::vector<std::string> linkOptions;
    Buffer* linked_bc = C->NewBuffer(DT_LLVM_BC);

    if (!linked_bc) {
      buildLog_ += "Error: Failed to open the linked program.\n";
      return false;
    }

    // NOTE: The linkOptions parameter is also used to identy cached code object. This parameter
    //       should not contain any dyanamically generated filename.
    bool ret = device().cacheCompilation()->linkLLVMBitcode(
      C.get(), inputs, linked_bc, linkOptions, buildLog_);
    buildLog_ += C->Output();
    if (!ret) {
      buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
      return false;
    }

    if (options->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
      std::ofstream f(options->getDumpFileName("_linked.bc").c_str(),
        std::ios::binary | std::ios::trunc);
      if (f.is_open()) {
        f.write(linked_bc->Buf().data(), linked_bc->Size());
        f.close();
      }
      else {
        buildLog_ += "Warning: opening the file to dump the linked IR failed.\n";
      }
    }

    inputs.clear();
    inputs.push_back(linked_bc);
  }

  Buffer* out_exec = C->NewBuffer(DT_EXECUTABLE);
  if (!out_exec) {
    buildLog_ += "Error: Failed to create the linked executable.\n";
    return false;
  }

  std::string codegenOptions(options->llvmOptions);

  // Set the machine target
  codegenOptions.append(" -mcpu=");
  codegenOptions.append(machineTarget_);

  // Set xnack option if needed
  if (xnackEnabled_) {
    codegenOptions.append(" -mxnack");
  }

  // Set the -O#
  std::ostringstream optLevel;
  optLevel << "-O" << options->oVariables->OptLevel;
  codegenOptions.append(" ").append(optLevel.str());

  // Pass clang options
  std::ostringstream ostrstr;
  std::copy(options->clangOptions.begin(), options->clangOptions.end(),
    std::ostream_iterator<std::string>(ostrstr, " "));
  codegenOptions.append(" ").append(ostrstr.str());

  // Set whole program mode
  codegenOptions.append(" -mllvm -amdgpu-internalize-symbols -mllvm -amdgpu-early-inline-all");
  // Force object code v2.
  codegenOptions.append(" -mno-code-object-v3");

  // Tokenize the options string into a vector of strings
  std::istringstream strstr(codegenOptions);
  std::istream_iterator<std::string> sit(strstr), end;
  std::vector<std::string> params(sit, end);

  // NOTE: The params is also used to identy cached code object. This parameter
  //       should not contain any dyanamically generated filename.
  bool ret = device().cacheCompilation()->compileAndLinkExecutable(C.get(), inputs, out_exec, params,
    buildLog_);
  buildLog_ += C->Output();
  if (!ret) {
    if (continueCompileFrom == ACL_TYPE_ASM_TEXT) {
      buildLog_ += "Error: Creating the executable from ISA assembly text failed.\n";
    }
    else {
      buildLog_ += "Error: Creating the executable from LLVM IRs failed.\n";
    }
    return false;
  }

  if (options->isDumpFlagSet(amd::option::DUMP_O)) {
    std::ofstream f(options->getDumpFileName(".so").c_str(), std::ios::binary | std::ios::trunc);
    if (f.is_open()) {
      f.write(out_exec->Buf().data(), out_exec->Size());
      f.close();
    }
    else {
      buildLog_ += "Warning: opening the file to dump the code object failed.\n";
    }
  }

  if (options->isDumpFlagSet(amd::option::DUMP_ISA)) {
    std::string name = options->getDumpFileName(".s");
    File* dump = C->NewFile(DT_INTERNAL, name);
    if (!C->DumpExecutableAsText(out_exec, dump)) {
      buildLog_ += "Warning: failed to dump code object.\n";
    }
  }

  // Call the device layer to setup all available kernels on the actual device
  if (!setKernels(options, out_exec->Buf().data(), out_exec->Size())) {
      return false;
  }

  // Save the binary and type
  clBinary()->saveBIFBinary(reinterpret_cast<const char*>(out_exec->Buf().data()), out_exec->Size());
  setType(TYPE_EXECUTABLE);

  return true;
#else
  return false;
#endif // defined(WITH_LIGHTNING_COMPILER)
}
#endif // defined(USE_COMGR_LIBRARY)


// ================================================================================================
bool Program::linkImplHSAIL(amd::option::Options* options) {
#if  defined(WITH_COMPILER_LIB)
  acl_error errorCode;
  bool finalize = true;
  internal_ = (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ? true : false;
  // If !binaryElf_ then program must have been created using clCreateProgramWithBinary
  aclType continueCompileFrom = (!binaryElf_) ?
    getNextCompilationStageFromBinary(options) : ACL_TYPE_LLVMIR_BINARY;

  switch (continueCompileFrom) {
  case ACL_TYPE_SPIRV_BINARY:
  case ACL_TYPE_SPIR_BINARY:
    // Compilation from ACL_TYPE_LLVMIR_BINARY to ACL_TYPE_CG in cases:
    // 1. if the program is not created with binary;
    // 2. if the program is created with binary and contains only .llvmir & .comment
    // 3. if the program is created with binary, contains .llvmir, .comment, brig sections,
    //    but the binary's compile & link options differ from current ones (recompilation);
  case ACL_TYPE_LLVMIR_BINARY:
    // Compilation from ACL_TYPE_HSAIL_BINARY to ACL_TYPE_CG in cases:
    // 1. if the program is created with binary and contains only brig sections
  case ACL_TYPE_HSAIL_BINARY:
    // Compilation from ACL_TYPE_HSAIL_TEXT to ACL_TYPE_CG in cases:
    // 1. if the program is created with binary and contains only hsail text
  case ACL_TYPE_HSAIL_TEXT: {
    std::string curOptions =
      options->origOptionStr + ProcessOptions(options);
    errorCode = aclCompile(device().compiler(), binaryElf_, curOptions.c_str(),
      continueCompileFrom, ACL_TYPE_CG, logFunction);
    buildLog_ += aclGetCompilerLog(device().compiler());
    if (errorCode != ACL_SUCCESS) {
      buildLog_ += "Error while BRIG Codegen phase: compilation error \n";
      return false;
    }
    break;
  }
  case ACL_TYPE_CG:
    break;
  case ACL_TYPE_ISA:
    finalize = false;
    break;
  default:
    buildLog_ += "Error while BRIG Codegen phase: the binary is incomplete \n";
    return false;
  }

  if (finalize) {
    std::string fin_options(options->origOptionStr + ProcessOptions(options));
    // Append an option so that we can selectively enable a SCOption on CZ
    // whenever IOMMUv2 is enabled.
    if (device().isFineGrainedSystem(true)) {
      fin_options.append(" -sc-xnack-iommu");
    }
    if (device().settings().gfx10Hsail_) {
      if (GPU_FORCE_WAVE_SIZE_32) {
        fin_options.append(" -force-wave-size-32");
      }
      if (xnackEnabled_) {
        fin_options.append(" -xnack");
      }
    }

    errorCode = aclCompile(device().compiler(), binaryElf_, fin_options.c_str(), ACL_TYPE_CG,
      ACL_TYPE_ISA, logFunction);
    buildLog_ += aclGetCompilerLog(device().compiler());
    if (errorCode != ACL_SUCCESS) {
      buildLog_ += "Error: BRIG finalization to ISA failed.\n";
      return false;
    }
  }

  size_t binSize;
  void* binary = const_cast<void*>(aclExtractSection(
    device().compiler(), binaryElf_, &binSize, aclTEXT, &errorCode));
  if (errorCode != ACL_SUCCESS) {
    buildLog_ += "Error: cannot extract ISA from compiled binary.\n";
    return false;
  }

  // Call the device layer to setup all available kernels on the actual device
  if (!setKernels(options, binary, binSize)) {
    return false;
  }

  // Save the binary in the interface class
  saveBinaryAndSetType(TYPE_EXECUTABLE);
  buildLog_ += aclGetCompilerLog(device().compiler());

  return true;
#else
  return false;
#endif // defined(WITH_COMPILER_LIB)
}

// ================================================================================================
bool Program::initClBinary() {
  if (clBinary_ == nullptr) {
    clBinary_ = new ClBinary(device());
    if (clBinary_ == nullptr) {
      return false;
    }
  }
  return true;
}

// ================================================================================================
void Program::releaseClBinary() {
  if (clBinary_ != nullptr) {
    delete clBinary_;
    clBinary_ = nullptr;
  }
}

// ================================================================================================
bool Program::initBuild(amd::option::Options* options) {
  compileOptions_ = options->origOptionStr;
  programOptions_ = options;

  if (options->oVariables->DumpFlags > 0) {
    static amd::Atomic<unsigned> build_num = 0;
    options->setBuildNo(build_num++);
  }
  buildLog_.clear();
  if (!initClBinary()) {
    return false;
  }

  const char* devName = machineTarget_;
  options->setPerBuildInfo((devName && (devName[0] != '\0')) ? devName : "gpu",
    clBinary()->getEncryptCode(), true);

  // Elf Binary setup
  std::string outFileName;

  // true means hsail required
  clBinary()->init(options, true);
  if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
    outFileName = options->getDumpFileName(".bin");
  }

  if (!clBinary()->setElfOut(LP64_SWITCH(ELFCLASS32, ELFCLASS64),
    (outFileName.size() > 0) ? outFileName.c_str() : nullptr)) {
    LogError("Setup elf out for gpu failed");
    return false;
  }

  return true;
}

// ================================================================================================
bool Program::finiBuild(bool isBuildGood) {
  clBinary()->resetElfOut();
  clBinary()->resetElfIn();

  if (!isBuildGood) {
    // Prevent the encrypted binary form leaking out
    clBinary()->setBinary(nullptr, 0);
  }

  return true;
}

// ================================================================================================
cl_int Program::compile(const std::string& sourceCode,
                        const std::vector<const std::string*>& headers,
                        const char** headerIncludeNames, const char* origOptions,
                        amd::option::Options* options) {
  uint64_t start_time = 0;
  if (options->oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  lastBuildOptionsArg_ = origOptions ? origOptions : "";
  if (options) {
    compileOptions_ = options->origOptionStr;
  }

  buildStatus_ = CL_BUILD_IN_PROGRESS;
  if (!initBuild(options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation init failed.";
    }
  }

  if (options->oVariables->FP32RoundDivideSqrt &&
      !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
    buildStatus_ = CL_BUILD_ERROR;
    buildLog_ +=
        "Error: -cl-fp32-correctly-rounded-divide-sqrt "
        "specified without device support";
  }

  // Compile the source code if any
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !sourceCode.empty() &&
      !compileImpl(sourceCode, headers, headerIncludeNames, options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation failed.";
    }
  }

  setType(TYPE_COMPILED);

  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !createBinary(options)) {
    buildLog_ += "Internal Error: creating OpenCL binary failed!\n";
  }

  if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation fini failed.";
    }
  }

  if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
    buildStatus_ = CL_BUILD_SUCCESS;
  } else {
    buildError_ = CL_COMPILE_PROGRAM_FAILURE;
  }

  if (options->oVariables->EnableBuildTiming) {
    std::stringstream tmp_ss;
    tmp_ss << "\nTotal Compile Time: " << (amd::Os::timeNanos() - start_time) / 1000ULL << " us\n";
    buildLog_ += tmp_ss.str();
  }

  if (options->oVariables->BuildLog && !buildLog_.empty()) {
    if (strcmp(options->oVariables->BuildLog, "stderr") == 0) {
      fprintf(stderr, "%s\n", options->optionsLog().c_str());
      fprintf(stderr, "%s\n", buildLog_.c_str());
    } else if (strcmp(options->oVariables->BuildLog, "stdout") == 0) {
      printf("%s\n", options->optionsLog().c_str());
      printf("%s\n", buildLog_.c_str());
    } else {
      std::fstream f;
      std::stringstream tmp_ss;
      std::string logs = options->optionsLog() + buildLog_;
      tmp_ss << options->oVariables->BuildLog << "." << options->getBuildNo();
      f.open(tmp_ss.str().c_str(), (std::fstream::out | std::fstream::binary));
      f.write(logs.data(), logs.size());
      f.close();
    }
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
cl_int Program::link(const std::vector<Program*>& inputPrograms, const char* origLinkOptions,
                     amd::option::Options* linkOptions) {
  lastBuildOptionsArg_ = origLinkOptions ? origLinkOptions : "";
  if (linkOptions) {
    linkOptions_ = linkOptions->origOptionStr;
  }

  buildStatus_ = CL_BUILD_IN_PROGRESS;

  amd::option::Options options;
  if (!getCompileOptionsAtLinking(inputPrograms, linkOptions)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Get compile options failed.";
    }
  } else {
    if (!amd::option::parseAllOptions(compileOptions_, options)) {
      buildStatus_ = CL_BUILD_ERROR;
      buildLog_ += options.optionsLog();
      LogError("Parsing compile options failed.");
    }
  }

  uint64_t start_time = 0;
  if (options.oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  // initBuild() will clear buildLog_, so store it in a temporary variable
  std::string tmpBuildLog = buildLog_;

  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !initBuild(&options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Compilation init failed.";
    }
  }

  buildLog_ += tmpBuildLog;

  if (options.oVariables->FP32RoundDivideSqrt &&
      !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
    buildStatus_ = CL_BUILD_ERROR;
    buildLog_ +=
        "Error: -cl-fp32-correctly-rounded-divide-sqrt "
        "specified without device support";
  }

  bool createLibrary = linkOptions ? linkOptions->oVariables->clCreateLibrary : false;
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !linkImpl(inputPrograms, &options, createLibrary)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Link failed.\n";
      buildLog_ += "Make sure the system setup is correct.";
    }
  }

  if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation fini failed.";
    }
  }

  if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
    buildStatus_ = CL_BUILD_SUCCESS;
  } else {
    buildError_ = CL_LINK_PROGRAM_FAILURE;
  }

  if (options.oVariables->EnableBuildTiming) {
    std::stringstream tmp_ss;
    tmp_ss << "\nTotal Link Time: " << (amd::Os::timeNanos() - start_time) / 1000ULL << " us\n";
    buildLog_ += tmp_ss.str();
  }

  if (options.oVariables->BuildLog && !buildLog_.empty()) {
    if (strcmp(options.oVariables->BuildLog, "stderr") == 0) {
      fprintf(stderr, "%s\n", options.optionsLog().c_str());
      fprintf(stderr, "%s\n", buildLog_.c_str());
    } else if (strcmp(options.oVariables->BuildLog, "stdout") == 0) {
      printf("%s\n", options.optionsLog().c_str());
      printf("%s\n", buildLog_.c_str());
    } else {
      std::fstream f;
      std::stringstream tmp_ss;
      std::string logs = options.optionsLog() + buildLog_;
      tmp_ss << options.oVariables->BuildLog << "." << options.getBuildNo();
      f.open(tmp_ss.str().c_str(), (std::fstream::out | std::fstream::binary));
      f.write(logs.data(), logs.size());
      f.close();
    }
  }

  if (!buildLog_.empty()) {
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
cl_int Program::build(const std::string& sourceCode, const char* origOptions,
                      amd::option::Options* options) {
  uint64_t start_time = 0;
  if (options->oVariables->EnableBuildTiming) {
    buildLog_ = "\nStart timing major build components.....\n\n";
    start_time = amd::Os::timeNanos();
  }

  lastBuildOptionsArg_ = origOptions ? origOptions : "";
  if (options) {
    compileOptions_ = options->origOptionStr;
  }

  buildStatus_ = CL_BUILD_IN_PROGRESS;
  if (!initBuild(options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation init failed.";
    }
  }

  if (options->oVariables->FP32RoundDivideSqrt &&
      !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
    buildStatus_ = CL_BUILD_ERROR;
    buildLog_ +=
        "Error: -cl-fp32-correctly-rounded-divide-sqrt "
        "specified without device support";
  }

  // Compile the source code if any
  std::vector<const std::string*> headers;
  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !sourceCode.empty() &&
      !compileImpl(sourceCode, headers, nullptr, options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation failed.";
    }
  }

  if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !linkImpl(options)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ += "Internal error: Link failed.\n";
      buildLog_ += "Make sure the system setup is correct.";
    }
  }

  if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
    buildStatus_ = CL_BUILD_ERROR;
    if (buildLog_.empty()) {
      buildLog_ = "Internal error: Compilation fini failed.";
    }
  }

  if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
    buildStatus_ = CL_BUILD_SUCCESS;
  } else {
    buildError_ = CL_BUILD_PROGRAM_FAILURE;
  }

  if (options->oVariables->EnableBuildTiming) {
    std::stringstream tmp_ss;
    tmp_ss << "\nTotal Build Time: " << (amd::Os::timeNanos() - start_time) / 1000ULL << " us\n";
    buildLog_ += tmp_ss.str();
  }

  if (options->oVariables->BuildLog && !buildLog_.empty()) {
    if (strcmp(options->oVariables->BuildLog, "stderr") == 0) {
      fprintf(stderr, "%s\n", options->optionsLog().c_str());
      fprintf(stderr, "%s\n", buildLog_.c_str());
    } else if (strcmp(options->oVariables->BuildLog, "stdout") == 0) {
      printf("%s\n", options->optionsLog().c_str());
      printf("%s\n", buildLog_.c_str());
    } else {
      std::fstream f;
      std::stringstream tmp_ss;
      std::string logs = options->optionsLog() + buildLog_;
      tmp_ss << options->oVariables->BuildLog << "." << options->getBuildNo();
      f.open(tmp_ss.str().c_str(), (std::fstream::out | std::fstream::binary));
      f.write(logs.data(), logs.size());
      f.close();
    }
  }

  if (!buildLog_.empty()) {
    LogError(buildLog_.c_str());
  }

  return buildError();
}

// ================================================================================================
std::string Program::ProcessOptions(amd::option::Options* options) {
  std::string optionsStr;

#ifndef WITH_LIGHTNING_COMPILER
  optionsStr.append(" -D__AMD__=1");

  optionsStr.append(" -D__").append(machineTarget_).append("__=1");
  optionsStr.append(" -D__").append(machineTarget_).append("=1");
#endif

#ifdef WITH_LIGHTNING_COMPILER
  int major, minor;
  ::sscanf(device().info().version_, "OpenCL %d.%d ", &major, &minor);

  std::stringstream ss;
  ss << " -D__OPENCL_VERSION__=" << (major * 100 + minor * 10);
  optionsStr.append(ss.str());
#endif

  if (device().info().imageSupport_ && options->oVariables->ImageSupport) {
    optionsStr.append(" -D__IMAGE_SUPPORT__=1");
  }

#ifndef WITH_LIGHTNING_COMPILER
  // Set options for the standard device specific options
  // All our devices support these options now
  if (device().settings().reportFMAF_) {
    optionsStr.append(" -DFP_FAST_FMAF=1");
  }
  if (device().settings().reportFMA_) {
    optionsStr.append(" -DFP_FAST_FMA=1");
  }
#endif

  uint clcStd =
    (options->oVariables->CLStd[2] - '0') * 100 + (options->oVariables->CLStd[4] - '0') * 10;

  if (clcStd >= 200) {
    std::stringstream opts;
    // Add only for CL2.0 and later
    opts << " -D"
      << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=" << device().info().maxGlobalVariableSize_;
    optionsStr.append(opts.str());
  }

  if (!device().settings().useLightning_) {
    if (!device().settings().singleFpDenorm_) {
      optionsStr.append(" -cl-denorms-are-zero");
    }

    // Check if the host is 64 bit or 32 bit
    LP64_ONLY(optionsStr.append(" -m64"));
  }

  // Tokenize the extensions string into a vector of strings
  std::istringstream istrstr(device().info().extensions_);
  std::istream_iterator<std::string> sit(istrstr), end;
  std::vector<std::string> extensions(sit, end);

  if (IS_LIGHTNING && !options->oVariables->Legacy) {
    // FIXME_lmoriche: opencl-c.h defines 'cl_khr_depth_images', so
    // remove it from the command line. Should we fix opencl-c.h?
    auto found = std::find(extensions.begin(), extensions.end(), "cl_khr_depth_images");
    if (found != extensions.end()) {
      extensions.erase(found);
    }

    if (!extensions.empty()) {
      std::ostringstream clext;

      clext << " -Xclang -cl-ext=+";
      std::copy(extensions.begin(), extensions.end() - 1,
        std::ostream_iterator<std::string>(clext, ",+"));
      clext << extensions.back();

      optionsStr.append(clext.str());
    }
  } else {
    for (auto e : extensions) {
      optionsStr.append(" -D").append(e).append("=1");
    }
  }

  return optionsStr;
}

// ================================================================================================
bool Program::getCompileOptionsAtLinking(const std::vector<Program*>& inputPrograms,
                                         const amd::option::Options* linkOptions) {
  amd::option::Options compileOptions;
  auto it = inputPrograms.cbegin();
  const auto itEnd = inputPrograms.cend();
  for (size_t i = 0; it != itEnd; ++it, ++i) {
    Program* program = *it;

    amd::option::Options compileOptions2;
    amd::option::Options* thisCompileOptions = i == 0 ? &compileOptions : &compileOptions2;
    if (!amd::option::parseAllOptions(program->compileOptions_, *thisCompileOptions)) {
      buildLog_ += thisCompileOptions->optionsLog();
      LogError("Parsing compile options failed.");
      return false;
    }

    if (i == 0) compileOptions_ = program->compileOptions_;

    // if we are linking a program executable, and if "program" is a
    // compiled module or a library created with "-enable-link-options",
    // we can overwrite "program"'s compile options with linking options
    if (!linkOptions_.empty() && !linkOptions->oVariables->clCreateLibrary) {
      bool linkOptsCanOverwrite = false;
      if (program->type() != TYPE_LIBRARY) {
        linkOptsCanOverwrite = true;
      } else {
        amd::option::Options thisLinkOptions;
        if (!amd::option::parseLinkOptions(program->linkOptions_, thisLinkOptions)) {
          buildLog_ += thisLinkOptions.optionsLog();
          LogError("Parsing link options failed.");
          return false;
        }
        if (thisLinkOptions.oVariables->clEnableLinkOptions) linkOptsCanOverwrite = true;
      }
      if (linkOptsCanOverwrite) {
        if (!thisCompileOptions->setOptionVariablesAs(*linkOptions)) {
          buildLog_ += thisCompileOptions->optionsLog();
          LogError("Setting link options failed.");
          return false;
        }
      }
      if (i == 0) compileOptions_ += " " + linkOptions_;
    }
    // warn if input modules have inconsistent compile options
    if (i > 0) {
      if (!compileOptions.equals(*thisCompileOptions, true /*ignore clc options*/)) {
        buildLog_ +=
            "Warning: Input OpenCL binaries has inconsistent"
            " compile options. Using compile options from"
            " the first input binary!\n";
      }
    }
  }
  return true;
}

// ================================================================================================
bool isSPIRVMagicL(const void* Image, size_t Length) {
  const unsigned SPRVMagicNumber = 0x07230203;
  if (Image == nullptr || Length < sizeof(unsigned))
    return false;
  auto Magic = static_cast<const unsigned*>(Image);
  return *Magic == SPRVMagicNumber;
}

// ================================================================================================
bool Program::initClBinary(const char* binaryIn, size_t size) {
  if (!initClBinary()) {
    return false;
  }

  // Save the original binary that isn't owned by ClBinary
  clBinary()->saveOrigBinary(binaryIn, size);

  const char* bin = binaryIn;
  size_t sz = size;

  // unencrypted
  int encryptCode = 0;
  char* decryptedBin = nullptr;
  bool isSPIRV = false;
  bool isBc = false;

#if defined(WITH_COMPILER_LIB)
  if (!device().settings().useLightning_) {
    isSPIRV = isSPIRVMagicL(binaryIn, size);
    isBc = isBcMagic(binaryIn);
  }
#endif  // defined(WITH_COMPILER_LIB)

  if (isSPIRV || isBc) {
#if defined(WITH_COMPILER_LIB)
    acl_error err = ACL_SUCCESS;
    aclBinaryOptions binOpts = {0};
    binOpts.struct_size = sizeof(binOpts);
    binOpts.elfclass =
        (info().arch_id == aclX64 || info().arch_id == aclAMDIL64 || info().arch_id == aclHSAIL64)
        ? ELFCLASS64
        : ELFCLASS32;
    binOpts.bitness = ELFDATA2LSB;
    binOpts.alloc = &::malloc;
    binOpts.dealloc = &::free;
    aclBinary* aclbin_v30 = aclBinaryInit(sizeof(aclBinary), &info(), &binOpts, &err);
    if (err != ACL_SUCCESS) {
      LogWarning("aclBinaryInit failed");
      aclBinaryFini(aclbin_v30);
      return false;
    }
    err = aclInsertSection(device().compiler(), aclbin_v30, binaryIn, size,
                           isSPIRV ? aclSPIRV : aclSPIR);
    if (ACL_SUCCESS != err) {
      LogWarning("aclInsertSection failed");
      aclBinaryFini(aclbin_v30);
      return false;
    }
    if (info().arch_id == aclHSAIL || info().arch_id == aclHSAIL64) {
      err = aclWriteToMem(aclbin_v30, (void**)const_cast<char**>(&bin), &sz);
      if (err != ACL_SUCCESS) {
        LogWarning("aclWriteToMem failed");
        aclBinaryFini(aclbin_v30);
        return false;
      }
      aclBinaryFini(aclbin_v30);
    } else {
      aclBinary* aclbin_v21 = aclCreateFromBinary(aclbin_v30, aclBIFVersion21);
      err = aclWriteToMem(aclbin_v21, (void**)const_cast<char**>(&bin), &sz);
      if (err != ACL_SUCCESS) {
        LogWarning("aclWriteToMem failed");
        aclBinaryFini(aclbin_v30);
        aclBinaryFini(aclbin_v21);
        return false;
      }
      aclBinaryFini(aclbin_v30);
      aclBinaryFini(aclbin_v21);
    }
#endif  // defined(WITH_COMPILER_LIB)
  } else {
    size_t decryptedSize;
    if (!clBinary()->decryptElf(binaryIn, size, &decryptedBin, &decryptedSize, &encryptCode)) {
      return false;
    }
    if (decryptedBin != nullptr) {
      // It is decrypted binary.
      bin = decryptedBin;
      sz = decryptedSize;
    }

    if (!isElf(bin)) {
      // Invalid binary.
      if (decryptedBin != nullptr) {
        delete[] decryptedBin;
      }
      return false;
    }
  }

  clBinary()->setFlags(encryptCode);

  return clBinary()->setBinary(bin, sz, (decryptedBin != nullptr));
}

// ================================================================================================
bool Program::setBinary(const char* binaryIn, size_t size) {
  if (!initClBinary(binaryIn, size)) {
    return false;
  }

  if (!clBinary()->setElfIn()) {
    LogError("Setting input OCL binary failed");
    return false;
  }
  uint16_t type;
  if (!clBinary()->elfIn()->getType(type)) {
    LogError("Bad OCL Binary: error loading ELF type!");
    return false;
  }
  switch (type) {
    case ET_NONE: {
      setType(TYPE_NONE);
      break;
    }
    case ET_REL: {
      if (clBinary()->isSPIR() || clBinary()->isSPIRV()) {
        setType(TYPE_INTERMEDIATE);
      } else {
        setType(TYPE_COMPILED);
      }
      break;
    }
    case ET_DYN: {
      char* sect = nullptr;
      size_t sz = 0;
      // FIXME: we should look for the e_machine to detect an HSACO.
      if (clBinary()->elfIn()->getSection(amd::OclElf::TEXT, &sect, &sz) && sect && sz > 0) {
        setType(TYPE_EXECUTABLE);
      } else {
        setType(TYPE_LIBRARY);
      }
      break;
    }
    case ET_EXEC: {
      setType(TYPE_EXECUTABLE);
      break;
    }
    default:
      LogError("Bad OCL Binary: bad ELF type!");
      return false;
  }

  clBinary()->loadCompileOptions(compileOptions_);
  clBinary()->loadLinkOptions(linkOptions_);

  clBinary()->resetElfIn();
  return true;
}

// ================================================================================================
aclType Program::getCompilationStagesFromBinary(std::vector<aclType>& completeStages,
  bool& needOptionsCheck) {
  aclType from = ACL_TYPE_DEFAULT;
  if (isLC()) {
#if defined(WITH_LIGHTNING_COMPILER)
    completeStages.clear();
    needOptionsCheck = true;
    //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
    // Checking llvmir in .llvmir section
    bool containsLlvmirText = (type() == TYPE_COMPILED);
    bool containsShaderIsa = (type() == TYPE_EXECUTABLE);
    bool containsOpts = !(compileOptions_.empty() && linkOptions_.empty());

    if (containsLlvmirText && containsOpts) {
      completeStages.push_back(from);
      from = ACL_TYPE_LLVMIR_BINARY;
    }
    if (containsShaderIsa) {
      completeStages.push_back(from);
      from = ACL_TYPE_ISA;
    }
    std::string sCurOptions = compileOptions_ + linkOptions_;
    amd::option::Options curOptions;
    if (!amd::option::parseAllOptions(sCurOptions, curOptions)) {
      buildLog_ += curOptions.optionsLog();
      LogError("Parsing compile options failed.");
      return ACL_TYPE_DEFAULT;
    }
    switch (from) {
    case ACL_TYPE_CG:
    case ACL_TYPE_ISA:
      // do not check options, if LLVMIR is absent or might be absent or options are absent
      if (!curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
        needOptionsCheck = false;
      }
      break;
      // recompilation might be needed
    case ACL_TYPE_LLVMIR_BINARY:
    case ACL_TYPE_DEFAULT:
    default:
      break;
    }
#endif   // defined(WITH_LIGHTNING_COMPILER)
  } else {
#if defined(WITH_COMPILER_LIB)
    acl_error errorCode;
    size_t secSize = 0;
    completeStages.clear();
    needOptionsCheck = true;
    size_t boolSize = sizeof(bool);
    // Checking llvmir in .llvmir section
    bool containsSpirv = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_SPIRV, nullptr,
      &containsSpirv, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsSpirv = false;
    }
    if (containsSpirv) {
      completeStages.push_back(from);
      from = ACL_TYPE_SPIRV_BINARY;
    }
    bool containsSpirText = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_SPIR, nullptr,
      &containsSpirText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsSpirText = false;
    }
    if (containsSpirText) {
      completeStages.push_back(from);
      from = ACL_TYPE_SPIR_BINARY;
    }
    bool containsLlvmirText = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_LLVMIR, nullptr,
      &containsLlvmirText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsLlvmirText = false;
    }
    // Checking compile & link options in .comment section
    bool containsOpts = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_OPTIONS, nullptr,
      &containsOpts, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsOpts = false;
    }
    if (containsLlvmirText && containsOpts) {
      completeStages.push_back(from);
      from = ACL_TYPE_LLVMIR_BINARY;
    }
    // Checking HSAIL in .cg section
    bool containsHsailText = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_HSAIL, nullptr,
      &containsHsailText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsHsailText = false;
    }
    // Checking BRIG sections
    bool containsBrig = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_BRIG, nullptr,
      &containsBrig, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsBrig = false;
    }
    if (containsBrig) {
      completeStages.push_back(from);
      from = ACL_TYPE_HSAIL_BINARY;
    }
    else if (containsHsailText) {
      completeStages.push_back(from);
      from = ACL_TYPE_HSAIL_TEXT;
    }
    // Checking Loader Map symbol from CG section
    bool containsLoaderMap = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_LOADER_MAP, nullptr,
      &containsLoaderMap, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsLoaderMap = false;
    }
    if (containsLoaderMap) {
      completeStages.push_back(from);
      from = ACL_TYPE_CG;
    }
    // Checking ISA in .text section
    bool containsShaderIsa = true;
    errorCode = aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_ISA, nullptr,
      &containsShaderIsa, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsShaderIsa = false;
    }
    if (containsShaderIsa) {
      completeStages.push_back(from);
      from = ACL_TYPE_ISA;
    }
    std::string sCurOptions = compileOptions_ + linkOptions_;
    amd::option::Options curOptions;
    if (!amd::option::parseAllOptions(sCurOptions, curOptions)) {
      buildLog_ += curOptions.optionsLog();
      LogError("Parsing compile options failed.");
      return ACL_TYPE_DEFAULT;
    }
    switch (from) {
      // compile from HSAIL text, no matter prev. stages and options
    case ACL_TYPE_HSAIL_TEXT:
      needOptionsCheck = false;
      break;
    case ACL_TYPE_HSAIL_BINARY:
      // do not check options, if LLVMIR is absent or might be absent or options are absent
      if (!curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
        needOptionsCheck = false;
      }
      break;
    case ACL_TYPE_CG:
    case ACL_TYPE_ISA:
      // do not check options, if LLVMIR is absent or might be absent or options are absent
      if (!curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
        needOptionsCheck = false;
      }
      // do not check options, if BRIG is absent or might be absent or LoaderMap is absent
      if (!curOptions.oVariables->BinCG || !containsBrig || !containsLoaderMap) {
        needOptionsCheck = false;
      }
      break;
      // recompilation might be needed
    case ACL_TYPE_LLVMIR_BINARY:
    case ACL_TYPE_DEFAULT:
    default:
      break;
    }
#endif  // #if defined(WITH_COMPILER_LIB)
  }
  return from;
}

// ================================================================================================
aclType Program::getNextCompilationStageFromBinary(amd::option::Options* options) {
  aclType continueCompileFrom = ACL_TYPE_DEFAULT;
  binary_t binary = this->binary();
  // If the binary already exists
  if ((binary.first != nullptr) && (binary.second > 0)) {
#if defined(WITH_COMPILER_LIB)
    if (aclValidateBinaryImage(binary.first, binary.second, BINARY_TYPE_ELF)) {
      acl_error errorCode;
      binaryElf_ = aclReadFromMem(binary.first, binary.second, &errorCode);
      if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Error while BRIG Codegen phase: aclReadFromMem failure \n";
        return continueCompileFrom;
      }
    }
#endif // defined(WITH_COMPILER_LIB)

    // save the current options
    std::string sCurCompileOptions = compileOptions_;
    std::string sCurLinkOptions = linkOptions_;
    std::string sCurOptions = compileOptions_ + linkOptions_;

    // Saving binary in the interface class,
    // which also load compile & link options from binary
    setBinary(static_cast<const char*>(binary.first), binary.second);

    // Calculate the next stage to compile from, based on sections in binaryElf_;
    // No any validity checks here
    std::vector<aclType> completeStages;
    bool needOptionsCheck = true;
    continueCompileFrom = getCompilationStagesFromBinary(completeStages, needOptionsCheck);
    if (!options || !needOptionsCheck) {
      return continueCompileFrom;
    }
    bool recompile = false;
    //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
    switch (continueCompileFrom) {
    case ACL_TYPE_HSAIL_BINARY:
    case ACL_TYPE_CG:
    case ACL_TYPE_ISA: {
      // Compare options loaded from binary with current ones, recompile if differ;
      // If compile options are absent in binary, do not compare and recompile
      if (compileOptions_.empty()) break;

      std::string sBinOptions;
#if defined(WITH_COMPILER_LIB)
      if (binaryElf_ != nullptr) {
        const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symOpenclCompilerOptions);
        assert(symbol && "symbol not found");
        std::string symName =
          std::string(symbol->str[bif::PRE]) + std::string(symbol->str[bif::POST]);
        size_t symSize = 0;
        acl_error errorCode;

        const void* opts = aclExtractSymbol(device().compiler(), binaryElf_, &symSize,
          aclCOMMENT, symName.c_str(), &errorCode);
        if (errorCode != ACL_SUCCESS) {
          recompile = true;
          break;
        }
        sBinOptions = std::string((char*)opts, symSize);
      }
      else
#endif // defined(WITH_COMPILER_LIB)
      {
        sBinOptions = sCurOptions;
      }

      compileOptions_ = sCurCompileOptions;
      linkOptions_ = sCurLinkOptions;

      amd::option::Options curOptions, binOptions;
      if (!amd::option::parseAllOptions(sBinOptions, binOptions)) {
        buildLog_ += binOptions.optionsLog();
        LogError("Parsing compile options from binary failed.");
        return ACL_TYPE_DEFAULT;
      }
      if (!amd::option::parseAllOptions(sCurOptions, curOptions)) {
        buildLog_ += curOptions.optionsLog();
        LogError("Parsing compile options failed.");
        return ACL_TYPE_DEFAULT;
      }
      if (!curOptions.equals(binOptions)) {
        recompile = true;
      }
      break;
    }
    default:
      break;
    }
    if (recompile) {
      while (!completeStages.empty()) {
        continueCompileFrom = completeStages.back();
        if (continueCompileFrom == ACL_TYPE_SPIRV_BINARY ||
            continueCompileFrom == ACL_TYPE_LLVMIR_BINARY ||
            continueCompileFrom == ACL_TYPE_SPIR_BINARY ||
            continueCompileFrom == ACL_TYPE_DEFAULT) {
          break;
        }
        completeStages.pop_back();
      }
    }
  }
  else {
    const char* xLang = options->oVariables->XLang;
    if (xLang != nullptr && strcmp(xLang, "asm") == 0) {
      continueCompileFrom = ACL_TYPE_ASM_TEXT;
    }
  }
  return continueCompileFrom;
}

// ================================================================================================
bool Program::FindGlobalVarSize(void* binary, size_t binSize) {
#if defined(WITH_LIGHTNING_COMPILER)
  size_t progvarsTotalSize = 0;
  size_t dynamicSize = 0;
  size_t progvarsWriteSize = 0;

  // Begin the Elf image from memory
  Elf* e = elf_memory((char*)binary, binSize, nullptr);
  if (elf_kind(e) != ELF_K_ELF) {
    buildLog_ += "Error while reading the ELF program binary\n";
    return false;
  }

  size_t numpHdrs;
  if (elf_getphdrnum(e, &numpHdrs) != 0) {
    buildLog_ += "Error while reading the ELF program binary\n";
    return false;
  }

  for (size_t i = 0; i < numpHdrs; ++i) {
    GElf_Phdr pHdr;
    if (gelf_getphdr(e, i, &pHdr) != &pHdr) {
      continue;
    }
    // Look for the runtime metadata note
    if (pHdr.p_type == PT_NOTE && pHdr.p_align >= sizeof(int)) {
      // Iterate over the notes in this segment
      address ptr = (address)binary + pHdr.p_offset;
      address segmentEnd = ptr + pHdr.p_filesz;

      while (ptr < segmentEnd) {
        Elf_Note* note = (Elf_Note*)ptr;
        address name = (address)&note[1];
        address desc = name + amd::alignUp(note->n_namesz, sizeof(int));

        if (note->n_type == 7 ||
            note->n_type == 8) {
          buildLog_ += "Error: object code with old metadata is not supported\n";
          return false;
        }
        else if (note->n_type == 10 /* NT_AMD_AMDGPU_HSA_METADATA */ &&
          note->n_namesz == sizeof "AMD" &&
          !memcmp(name, "AMD", note->n_namesz)) {
#if defined(USE_COMGR_LIBRARY)
          amd_comgr_status_t status;
          amd_comgr_data_t binaryData;

          status  = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &binaryData);
          if (status == AMD_COMGR_STATUS_SUCCESS) {
            status = amd_comgr_set_data(binaryData, binSize,
                                        reinterpret_cast<const char*>(binary));
          }

          if (status == AMD_COMGR_STATUS_SUCCESS) {
            metadata_ = new amd_comgr_metadata_node_t;
            status = amd_comgr_get_data_metadata(binaryData, metadata_);
          }

          amd_comgr_release_data(binaryData);

          if (status != AMD_COMGR_STATUS_SUCCESS) {
            buildLog_ += "Error: COMGR fails to get the metadata.\n";
            return false;
          }
#else
          std::string metadataStr((const char*)desc, (size_t)note->n_descsz);
          metadata_ = new CodeObjectMD();
          if (llvm::AMDGPU::HSAMD::fromString(metadataStr, *metadata_)) {
            buildLog_ += "Error: failed to process metadata\n";
            return false;
          }
          // We've found and loaded the runtime metadata, exit the
          // note record loop now.
#endif
          break;
        }
        ptr += sizeof(*note) + amd::alignUp(note->n_namesz, sizeof(int)) +
          amd::alignUp(note->n_descsz, sizeof(int));
      }
    }
    // Accumulate the size of R & !X loadable segments
    else if (pHdr.p_type == PT_LOAD && !(pHdr.p_flags & PF_X)) {
      if (pHdr.p_flags & PF_R) {
        progvarsTotalSize += pHdr.p_memsz;
      }
      if (pHdr.p_flags & PF_W) {
        progvarsWriteSize += pHdr.p_memsz;
      }
    }
    else if (pHdr.p_type == PT_DYNAMIC) {
      dynamicSize += pHdr.p_memsz;
    }
  }

  elf_end(e);

  if (!metadata_) {
    buildLog_ +=
      "Error: runtime metadata section not present in ELF program binary\n";
    return false;
  }

  progvarsTotalSize -= dynamicSize;
  setGlobalVariableTotalSize(progvarsTotalSize);

  if (progvarsWriteSize != dynamicSize) {
    hasGlobalStores_ = true;
  }
#endif // defined(WITH_LIGHTNING_COMPILER)
  return true;
}
}
