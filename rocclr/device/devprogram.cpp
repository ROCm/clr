/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "platform/ndrange.hpp"
#include "devprogram.hpp"
#include "devkernel.hpp"
#include "utils/macros.hpp"
#include "utils/options.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"
#include "comgrctx.hpp"

#include <algorithm>
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

#ifdef EARLY_INLINE
#define AMDGPU_EARLY_INLINE_ALL_OPTION " -mllvm -amdgpu-early-inline-all"
#else
#define AMDGPU_EARLY_INLINE_ALL_OPTION
#endif

namespace device {

// TODO: Can this be unified with the copies in:
// runtime/device/pal/palprogram.cpp, runtime/device/gpu/gpuprogram.cpp,
// compiler/lib/utils/v0_8/libUtils.h, compiler/lib/backends/gpu/hsail_be.cpp,
// compiler/legacy-lib/utils/v0_8/libUtils.h,
// and compiler/legacy-lib/backends/gpu/hsail_be.cpp ?
inline static std::vector<std::string> splitSpaceSeparatedString(const char *str) {
  std::string s(str);
  std::stringstream ss(s);
  std::istream_iterator<std::string> beg(ss), end;
  std::vector<std::string> vec(beg, end);
  return vec;
}

// ================================================================================================
Program::Program(amd::Device& device, amd::Program& owner)
    : device_(device),
      owner_(owner),
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
      programOptions_(nullptr)
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

  /* Delete the undefined memory object */
  for (auto it = undef_mem_obj_.begin(); it != undef_mem_obj_.end(); ++it) {
    (*it)->release();
  }

  if (isLC()) {
#if defined(USE_COMGR_LIBRARY)
    for (auto const& kernelMeta : kernelMetadataMap_) {
      amd::Comgr::destroy_metadata(kernelMeta.second);
    }
    amd::Comgr::destroy_metadata(metadata_);
#endif
  }
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
#if defined(USE_COMGR_LIBRARY)
static std::string llvmBin_(amd::Os::getEnvironment("LLVM_BIN"));

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

#endif // defined(USE_COMGR_LIBRARY)

// ================================================================================================

#if defined(USE_COMGR_LIBRARY)
// If buildLog is not null, and dataSet contains a log object, extract the
// first log data object from dataSet and process it with
// extractByteCodeBinary.
void Program::extractBuildLog(amd_comgr_data_set_t dataSet) {
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  size_t count;
  status = amd::Comgr::action_data_count(dataSet, AMD_COMGR_DATA_KIND_LOG, &count);

  if (status == AMD_COMGR_STATUS_SUCCESS && count > 0) {
    char* logData = nullptr;
    size_t logSize;
    status = extractByteCodeBinary(dataSet, AMD_COMGR_DATA_KIND_LOG, "", &logData, &logSize);
    buildLog_ += logData;
    delete[] logData;
  }
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Warning: extracting build log failed.\n";
  }
}

//  Extract the byte code binary from the data set.  The binary will be saved to an output
//  file if the file name is provided. If buffer pointer, outBinary, is provided, the
//  binary will be passed back to the caller.
//
amd_comgr_status_t Program::extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
                                                  const amd_comgr_data_kind_t dataKind,
                                                  const std::string& outFileName,
                                                  char* outBinary[], size_t* outSize) {
  amd_comgr_data_t  binaryData;

  amd_comgr_status_t status = amd::Comgr::action_data_get_data(inDataSet, dataKind, 0, &binaryData);

  size_t binarySize = 0;
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_data(binaryData, &binarySize, NULL);
  }

  size_t bufSize = (dataKind == AMD_COMGR_DATA_KIND_LOG) ? binarySize + 1 : binarySize;

  char* binary = new char[bufSize];
  if (binary == nullptr) {
    amd::Comgr::release_data(binaryData);
    return AMD_COMGR_STATUS_ERROR;
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_data(binaryData, &binarySize, binary);
  }

  if (dataKind == AMD_COMGR_DATA_KIND_LOG) {
    binary[binarySize] = '\0';
  }

  amd::Comgr::release_data(binaryData);

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    delete[] binary;
    return status;
  }

  // save the binary to the file as output file name is specified
  if (!outFileName.empty()) {
    std::ofstream f(outFileName.c_str(), std::ios::trunc | std::ios::binary);
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
    delete[] binary;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t Program::addCodeObjData(const char *source,
                                           const size_t size,
                                           const amd_comgr_data_kind_t type,
                                           const char* name,
                                           amd_comgr_data_set_t* dataSet)
{
  amd_comgr_data_t data;
  amd_comgr_status_t status;

  status = amd::Comgr::create_data(type, &data);
  if (status  != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  status = amd::Comgr::set_data(data, size, source);

  if ((name != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd::Comgr::set_data_name(data, name);
  }

  if ((dataSet != nullptr) && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd::Comgr::data_set_add(*dataSet, data);
  }

  amd::Comgr::release_data(data);

  return status;
}

void Program::setLangAndTargetStr(const char* clStd, amd_comgr_language_t* langver,
                                  std::string& targetIdent) {

  if (isHIP()) {
    if (langver != nullptr) {
      *langver = AMD_COMGR_LANGUAGE_HIP;
    }
  } else {
    uint clcStd = (clStd[2] - '0') * 100 + (clStd[4] - '0') * 10;

    if (langver != nullptr) {
      switch (clcStd) {
        case 100:
        case 110:
        case 120:
          *langver = AMD_COMGR_LANGUAGE_OPENCL_1_2;
          break;
        case 200:
          *langver = AMD_COMGR_LANGUAGE_OPENCL_2_0;
          break;
        default:
          *langver = AMD_COMGR_LANGUAGE_NONE;
          break;
      }
    }
  }
  // Set target triple and CPU
  targetIdent = std::string("amdgcn-amd-amdhsa--") + machineTarget_;
  // Set xnack option if needed
  if (xnackEnabled_) {
    targetIdent.append("+xnack");
  }
  if (sramEccEnabled_) {
    targetIdent.append("+sram-ecc");
  }
}


amd_comgr_status_t Program::createAction(const amd_comgr_language_t oclver,
                                         const std::string& targetIdent,
                                         const std::vector<std::string>& options,
                                         amd_comgr_action_info_t* action,
                                         bool* hasAction) {

  *hasAction = false;
  amd_comgr_status_t status = amd::Comgr::create_action_info(action);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    *hasAction = true;
    if (oclver != AMD_COMGR_LANGUAGE_NONE) {
      status = amd::Comgr::action_info_set_language(*action, oclver);
    }
  }

  if (!targetIdent.empty() && (status == AMD_COMGR_STATUS_SUCCESS)) {
    status = amd::Comgr::action_info_set_isa_name(*action, targetIdent.c_str());
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::vector<const char *> optionsArgv;
    optionsArgv.reserve(options.size());
    for (auto &option : options) {
      optionsArgv.push_back(option.c_str());
    }
    status = amd::Comgr::action_info_set_option_list(*action, optionsArgv.data(), optionsArgv.size());
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::action_info_set_logging(*action, true);
  }

  return status;
}

bool Program::linkLLVMBitcode(const amd_comgr_data_set_t inputs,
                              const std::vector<std::string>& options, const bool requiredDump,
                              amd::option::Options* amdOptions, amd_comgr_data_set_t* output,
                              char* binaryData[], size_t* binarySize) {

  // get the language and target name
  std::string targetIdent;
  amd_comgr_language_t langver;
  setLangAndTargetStr(amdOptions->oVariables->CLStd, &langver, targetIdent);
  if (langver == AMD_COMGR_LANGUAGE_NONE) {
    DevLogPrintfError("Cannot set Langauge version for %s \n",
                      amdOptions->oVariables->CLStd);
    return false;
  }

  //  Create the action for linking
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t dataSetDevLibs;
  bool hasAction = false;
  bool hasDataSetDevLibs = false;

  amd_comgr_status_t status = createAction(langver, targetIdent, options, &action, &hasAction);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&dataSetDevLibs);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasDataSetDevLibs = true;
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES, action, inputs,
                                   dataSetDevLibs);
    extractBuildLog(dataSetDevLibs);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, action, dataSetDevLibs, *output);
    extractBuildLog(*output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string dumpFileName;
    if (requiredDump && amdOptions != nullptr &&
        amdOptions->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
      dumpFileName = amdOptions->getDumpFileName("_linked.bc");
    }
    status = extractByteCodeBinary(*output, AMD_COMGR_DATA_KIND_BC, dumpFileName, binaryData,
                                   binarySize);
  }

  if (hasAction) {
    amd::Comgr::destroy_action_info(action);
  }

  if (hasDataSetDevLibs) {
    amd::Comgr::destroy_data_set(dataSetDevLibs);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Program::compileToLLVMBitcode(const amd_comgr_data_set_t compileInputs,
                                   const std::vector<std::string>& options, amd::option::Options* amdOptions,
                                   char* binaryData[], size_t* binarySize) {

  //  get the lanuage and target name
  std::string targetIdent;
  amd_comgr_language_t langver;
  setLangAndTargetStr(amdOptions->oVariables->CLStd, &langver, targetIdent);
  if (langver == AMD_COMGR_LANGUAGE_NONE) {
    DevLogPrintfError("Cannot set Langauge version for %s \n",
                      amdOptions->oVariables->CLStd);
    return false;
  }

  //  Create the output data set
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t dataSetPCH;
  amd_comgr_data_set_t input = compileInputs ;

  bool hasAction = false;
  bool hasOutput = false;
  bool hasDataSetPCH = false;

  amd_comgr_status_t status = createAction(langver, targetIdent, options, &action, &hasAction);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&output);
  }

  //  Adding Precompiled Headers
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasOutput = true;
    status = amd::Comgr::create_data_set(&dataSetPCH);
  }

  // Preprocess the source
  // FIXME: This must happen before the precompiled headers are added, as they
  // do not embed the source text of the header, and so reference paths in the
  // filesystem which do not exist at runtime.
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasDataSetPCH = true;

    if (amdOptions->isDumpFlagSet(amd::option::DUMP_I)){
      amd_comgr_data_set_t dataSetPreprocessor;
      bool hasDataSetPreprocessor = false;

      status = amd::Comgr::create_data_set(&dataSetPreprocessor);

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        hasDataSetPreprocessor = true;
        status = amd::Comgr::do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR,
                                       action, input, dataSetPreprocessor);
        extractBuildLog(dataSetPreprocessor);
      }

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        std::string outFileName = amdOptions->getDumpFileName(".i");
        status = extractByteCodeBinary(dataSetPreprocessor,
                                       AMD_COMGR_DATA_KIND_SOURCE, outFileName);
      }

      if (hasDataSetPreprocessor) {
        amd::Comgr::destroy_data_set(dataSetPreprocessor);
      }
    }
  }

  if (!isHIP()) {
    if (status == AMD_COMGR_STATUS_SUCCESS) {
      status = amd::Comgr::do_action(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS,
                                     action, input, dataSetPCH);
      extractBuildLog(dataSetPCH);
    }

    // Set input for the next stage
    input = dataSetPCH;
  }

  //  Compiling the source codes with precompiled headers or directly compileInputs
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                                 action, input, output);
    extractBuildLog(output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string outFileName;
    if (amdOptions->isDumpFlagSet(amd::option::DUMP_BC_ORIGINAL)) {
       outFileName = amdOptions->getDumpFileName("_original.bc");
    }
    status = extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, outFileName, binaryData,
                                   binarySize);
  }

  if (hasAction) {
    amd::Comgr::destroy_action_info(action);
  }

  if (hasDataSetPCH) {
    amd::Comgr::destroy_data_set(dataSetPCH);
  }

  if (hasOutput) {
    amd::Comgr::destroy_data_set(output);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

//  Create an executable from an input data set.  To generate the executable,
//  the input data set is converted to relocatable code, then executable binary.
//  If assembly code is required, the input data set is converted to assembly.
bool Program::compileAndLinkExecutable(const amd_comgr_data_set_t inputs,
                                       const std::vector<std::string>& options, amd::option::Options* amdOptions,
                                       char* executable[], size_t* executableSize) {

  //  get the language and target name
  std::string targetIdent;
  setLangAndTargetStr(amdOptions->oVariables->CLStd, nullptr, targetIdent);

  // create the linked output
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t relocatableData;
  bool hasAction = false;
  bool hasOutput = false;
  bool hasRelocatableData = false;

  amd_comgr_status_t status = createAction(AMD_COMGR_LANGUAGE_NONE, targetIdent, options,
                                           &action, &hasAction);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasOutput = true;

    if (amdOptions->isDumpFlagSet(amd::option::DUMP_ISA)){
      //  create the assembly data set
      amd_comgr_data_set_t assemblyData;
      bool hasAssemblyData = false;

      status = amd::Comgr::create_data_set(&assemblyData);
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        hasAssemblyData = true;
        status = amd::Comgr::do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                                       action, inputs, assemblyData);
        extractBuildLog(assemblyData);
      }

      // dump the ISA
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        std::string dumpIsaName = amdOptions->getDumpFileName(".s");
        status = extractByteCodeBinary(assemblyData, AMD_COMGR_DATA_KIND_SOURCE, dumpIsaName);
      }

      if (hasAssemblyData) {
        amd::Comgr::destroy_data_set(assemblyData);
      }
    }
  }

  //  Create the relocatiable data set
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::create_data_set(&relocatableData);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasRelocatableData = true;
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                                 action, inputs, relocatableData);
    extractBuildLog(relocatableData);
  }

  // Create executable from the relocatable data set
  amd::Comgr::action_info_set_option_list(action, nullptr, 0);
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                                 action, relocatableData, output);
    extractBuildLog(output);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    // Extract the executable binary
    std::string outFileName;
    if (amdOptions->isDumpFlagSet(amd::option::DUMP_O)) {
      outFileName = amdOptions->getDumpFileName(".so");
    }
    status = extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_EXECUTABLE, outFileName, executable,
                                   executableSize);
  }

  if (hasAction) {
    amd::Comgr::destroy_action_info(action);
  }

  if (hasRelocatableData) {
    amd::Comgr::destroy_data_set(relocatableData);
  }

  if (hasOutput) {
    amd::Comgr::destroy_data_set(output);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}
#endif  // defined(USE_COMGR_LIBRARY)

bool Program::compileImplLC(const std::string& sourceCode,
                            const std::vector<const std::string*>& headers,
                            const char** headerIncludeNames, amd::option::Options* options) {
#if  defined(USE_COMGR_LIBRARY)
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

  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create output buffer for LLVM bitcode.\n";
    return false;
  }

  if (addCodeObjData(sourceCode.c_str(), sourceCode.length(), AMD_COMGR_DATA_KIND_SOURCE,
                     "CompileSource", &inputs) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create data from source.\n";
    amd::Comgr::destroy_data_set(inputs);
    return false;
  }

  std::vector<std::string> driverOptions;
  // Set the -O#
  std::ostringstream optLevel;
  optLevel << "-O" << options->oVariables->OptLevel;
  driverOptions.push_back(optLevel.str());

  if(!isHIP()) {
    driverOptions.insert(driverOptions.end(), options->clangOptions.begin(), options->clangOptions.end());
    // TODO: Can this be fixed at the source? options->llvmOptions is a flat
    // string, but should really be a vector of strings.
    std::vector<std::string> splitLlvmOptions = splitSpaceSeparatedString(options->llvmOptions.c_str());
    driverOptions.insert(driverOptions.end(), splitLlvmOptions.begin(), splitLlvmOptions.end());
  }

  std::vector<std::string> processedOptions = ProcessOptions(options);
  driverOptions.insert(driverOptions.end(), processedOptions.begin(), processedOptions.end());

  // Set whole program mode
#ifdef EARLY_INLINE
  driverOptions.push_back("-mllvm");
  driverOptions.push_back("-amdgpu-early-inline-all");
#endif
  driverOptions.push_back("-mllvm");
  driverOptions.push_back("-amdgpu-prelink");

  if (!device().settings().enableWgpMode_) {
    driverOptions.push_back("-mcumode");
  }

  if (device().settings().lcWavefrontSize64_) {
    driverOptions.push_back("-mwavefrontsize64");
  }

  // Iterate through each source code and dump it into tmp
  std::fstream f;
  std::vector<std::string> headerFileNames(headers.size());

  if (!headers.empty()) {
    for (size_t i = 0; i < headers.size(); ++i) {
      std::string headerIncludeName(headerIncludeNames[i]);
      // replace / in path with current os's file separator
      if (amd::Os::fileSeparator() != '/') {
        for (auto& it : headerIncludeName) {
          if (it == '/') it = amd::Os::fileSeparator();
        }
      }
      if (addCodeObjData(headers[i]->c_str(), headers[i]->length(), AMD_COMGR_DATA_KIND_INCLUDE,
                         headerIncludeName.c_str(), &inputs) != AMD_COMGR_STATUS_SUCCESS) {
        buildLog_ += "Error: COMGR fails to add headers into inputs.\n";
        amd::Comgr::destroy_data_set(inputs);
        return false;
      }
    }
  }

  if (!isHIP() && options->isDumpFlagSet(amd::option::DUMP_CL)) {
    std::ostringstream driverOptionsOStrStr;
    std::copy(driverOptions.begin(), driverOptions.end(),
      std::ostream_iterator<std::string>(driverOptionsOStrStr, " "));

    std::ofstream f(options->getDumpFileName(".cl").c_str(), std::ios::trunc);
    if (f.is_open()) {
      f << "/* Compiler options:\n"
           "-c -emit-llvm -target amdgcn-amd-amdhsa -x cl "
        << driverOptionsOStrStr.str() << " -include opencl-c.h "
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
    // Destroy the original LLVM binary, received after compilation
    delete[] binaryData;

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
    buildLog_ += "Error: Failed to compile opencl source (from CL or HIP source to LLVM IR).\n";
  }

  amd::Comgr::destroy_data_set(inputs);
  return ret;
#else   // defined(USE_COMGR_LIBRARY)
  return false;
#endif  // defined(USE_COMGR_LIBRARY)
}


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
  compileOptions_.append(ProcessOptionsFlattened(options));
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
bool Program::linkImplLC(const std::vector<Program*>& inputPrograms,
                         amd::option::Options* options, bool createLibrary) {
#if defined(USE_COMGR_LIBRARY)
  amd_comgr_data_set_t inputs;

  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
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
      std::string llvmName = "LLVM Binary " + std::to_string(idx);
      result = (addCodeObjData(program->llvmBinary_.data(), program->llvmBinary_.size(),
                               AMD_COMGR_DATA_KIND_BC, llvmName.c_str(), &inputs) ==
                               AMD_COMGR_STATUS_SUCCESS);
    }

    if (!result) {
      amd::Comgr::destroy_data_set(inputs);
      buildLog_ += "Error: Linking bitcode failed: failing to generate LLVM binary.\n";
      return false;
    }

    idx++;

    // release elfIn() for the program
    program->clBinary()->resetElfIn();
  }

  // create the linked output
  amd_comgr_data_set_t output;
  if (amd::Comgr::create_data_set(&output) != AMD_COMGR_STATUS_SUCCESS) {
    buildLog_ += "Error: COMGR fails to create output buffer for LLVM bitcode.\n";
    amd::Comgr::destroy_data_set(inputs);
    return false;
  }

  // NOTE: The options parameter is also used to identy cached code object.
  //       This parameter should not contain any dyanamically generated filename.
  char* binaryData = nullptr;
  size_t binarySize = 0;
  std::vector<std::string> linkOptions;
  bool ret = linkLLVMBitcode(inputs, linkOptions, false, options, &output, &binaryData,
                             &binarySize);

  amd::Comgr::destroy_data_set(output);
  amd::Comgr::destroy_data_set(inputs);

  if (!ret) {
    buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
    return false;
  }

  llvmBinary_.assign(binaryData, binarySize);

  // Destroy llvm binary, received after compilation
  delete[] binaryData;

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
#else   // defined(USE_COMGR_LIBRARY)
  return false;
#endif  // defined(USE_COMGR_LIBRARY)
}

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

static void dumpCodeObject(const std::string& image) {
    char fname[30];
    static std::atomic<int> index;
    sprintf(fname, "_code_object%04d.o", index++);
    ClPrint(amd::LOG_INFO, amd::LOG_CODE, "Code object saved in %s\n", fname);
    std::ofstream ofs;
    ofs.open(fname, std::ios::binary);
    ofs << image;
    ofs.close();
}

// ================================================================================================
bool Program::linkImplLC(amd::option::Options* options) {
#if defined(USE_COMGR_LIBRARY)
  aclType continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;

  internal_ = (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ?
    true : false;

  amd_comgr_data_set_t inputs;
  if (amd::Comgr::create_data_set(&inputs) != AMD_COMGR_STATUS_SUCCESS) {
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
        amd::Comgr::destroy_data_set(inputs);
        return false;
      }

      bLinkLLVMBitcode = false;
      break;
    }
    case ACL_TYPE_ISA: {
      amd::Comgr::destroy_data_set(inputs);
      binary_t isaBinary = binary();
      if (GPU_DUMP_CODE_OBJECT) {
        dumpCodeObject(std::string{(const char*)isaBinary.first, isaBinary.second});
      }
      return setKernels(options, const_cast<void *>(isaBinary.first), isaBinary.second);
      break;
    }
    default:
      buildLog_ += "Error while Codegen phase: the binary is incomplete \n";
      amd::Comgr::destroy_data_set(inputs);
      return false;
  }

  // call LinkLLVMBitcode
  if (bLinkLLVMBitcode) {
    // open the bitcode libraries
    std::vector<std::string> linkOptions;

    if (options->oVariables->FP32RoundDivideSqrt) {
        linkOptions.push_back("correctly_rounded_sqrt");
    }
    if (options->oVariables->DenormsAreZero || AMD_GPU_FORCE_SINGLE_FP_DENORM == 0 ||
        (device().info().gfxipVersion_ < 900 && AMD_GPU_FORCE_SINGLE_FP_DENORM < 0)) {
        linkOptions.push_back("daz_opt");
    }
    if (options->oVariables->FiniteMathOnly || options->oVariables->FastRelaxedMath) {
        linkOptions.push_back("finite_only");
    }
    if (options->oVariables->UnsafeMathOpt || options->oVariables->FastRelaxedMath) {
        linkOptions.push_back("unsafe_math");
    }
    if (device().settings().lcWavefrontSize64_) {
        linkOptions.push_back("wavefrontsize64");
    }

    amd_comgr_status_t status = addCodeObjData(llvmBinary_.data(), llvmBinary_.size(),
                                               AMD_COMGR_DATA_KIND_BC,
                                               "LLVM Binary", &inputs);

    amd_comgr_data_set_t linked_bc;
    bool hasLinkedBC = false;

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      status = amd::Comgr::create_data_set(&linked_bc);
    }

    bool ret = (status == AMD_COMGR_STATUS_SUCCESS);
    if (ret) {
      hasLinkedBC = true;
      ret = linkLLVMBitcode(inputs, linkOptions, true, options, &linked_bc);
    }

    amd::Comgr::destroy_data_set(inputs);

    if (!ret) {
      if (hasLinkedBC) {
        amd::Comgr::destroy_data_set(linked_bc);
      }
      buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
      return false;
    }

    inputs = linked_bc;
  }

  std::vector<std::string> codegenOptions;

  // TODO: Can this be fixed at the source? options->llvmOptions is a flat
  // string, but should really be a vector of strings.
  std::vector<std::string> splitLlvmOptions = splitSpaceSeparatedString(options->llvmOptions.c_str());
  codegenOptions.insert(codegenOptions.end(), splitLlvmOptions.begin(), splitLlvmOptions.end());

  // Set the -O#
  std::ostringstream optLevel;
  optLevel << "-O" << options->oVariables->OptLevel;
  codegenOptions.push_back(optLevel.str());

  // Pass clang options
  codegenOptions.insert(codegenOptions.end(),
      options->clangOptions.begin(), options->clangOptions.end());

  // Set SRAM ECC option if needed
  if (sramEccEnabled_) {
    codegenOptions.push_back("-msram-ecc");
  }
  else {
    codegenOptions.push_back("-mno-sram-ecc");
  }

  // Set whole program mode
  codegenOptions.push_back("-mllvm");
  codegenOptions.push_back("-amdgpu-internalize-symbols");
#ifdef EARLY_INLINE
  codegenOptions.push_back("-mllvm");
  codegenOptions.push_back("-amdgpu-early-inline-all");
#endif

  if (!device().settings().enableWgpMode_) {
    codegenOptions.push_back("-mcumode");
  }

  if (device().settings().lcWavefrontSize64_) {
    codegenOptions.push_back("-mwavefrontsize64");
  }

  // NOTE: The params is also used to identy cached code object. This parameter
  //       should not contain any dyanamically generated filename.
  char* executable = nullptr;
  size_t executableSize = 0;
  bool ret = compileAndLinkExecutable(inputs, codegenOptions, options, &executable,
                                      &executableSize);
  amd::Comgr::destroy_data_set(inputs);

  if (!ret) {
    if (continueCompileFrom == ACL_TYPE_ASM_TEXT) {
      buildLog_ += "Error: Creating the executable from ISA assembly text failed.\n";
    } else {
      buildLog_ += "Error: Creating the executable from LLVM IRs failed.\n";
    }
    return false;
  }

  // Save the binary and type
  clBinary()->saveBIFBinary(executable, executableSize);

  // Destroy original memory with executable after compilation
  delete[] executable;

  if (!setKernels(options,  const_cast<void*>(clBinary()->data().first),
                  clBinary()->data().second)) {
    return false;
  }

  setType(TYPE_EXECUTABLE);

  return true;
#else   // defined(USE_COMGR_LIBRARY)
  return false;
#endif  // defined(USE_COMGR_LIBRARY)
}


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
      options->origOptionStr + ProcessOptionsFlattened(options);
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
    std::string fin_options(options->origOptionStr + ProcessOptionsFlattened(options));
    // Append an option so that we can selectively enable a SCOption on CZ
    // whenever IOMMUv2 is enabled.
    if (device().isFineGrainedSystem(true)) {
      fin_options.append(" -sc-xnack-iommu");
    }

  if (device().settings().enableWave32Mode_) {
    fin_options.append(" -force-wave-size-32");
  }

  if (device().settings().enableWgpMode_) {
    fin_options.append(" -force-wgp-mode");
  }

  if (device().settings().hsailExplicitXnack_) {
    fin_options.append(" -xnack");
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
    buildLog_ += "Error: Cannot set kernel \n";
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
    DevLogError("Init CL Binary failed \n");
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
int32_t Program::compile(const std::string& sourceCode,
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
int32_t Program::link(const std::vector<Program*>& inputPrograms, const char* origLinkOptions,
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
    if (!amd::option::parseAllOptions(compileOptions_, options, false, isLC())) {
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
int32_t Program::build(const std::string& sourceCode, const char* origOptions,
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
std::vector<std::string> Program::ProcessOptions(amd::option::Options* options) {
  std::string scratchStr;
  std::vector<std::string> optionsVec;

  if (!isLC()) {
    optionsVec.push_back("-D__AMD__=1");

    scratchStr.clear();
    optionsVec.push_back(scratchStr.append("-D__").append(machineTarget_).append("__=1"));

    scratchStr.clear();
    optionsVec.push_back(scratchStr.append("-D__").append(machineTarget_).append("=1"));

    // Set options for the standard device specific options
    // All our devices support these options now
    if (device().settings().reportFMAF_) {
      optionsVec.push_back("-DFP_FAST_FMAF=1");
    }
    if (device().settings().reportFMA_) {
      optionsVec.push_back("-DFP_FAST_FMA=1");
    }
  } else {

    if (isHIP()) {
      optionsVec.push_back("-D__HIP_ROCclr__=1");
      scratchStr.clear();
      std::string target(machineTarget_);
      std::transform(target.begin(), target.end(), target.begin(), ::toupper);
      optionsVec.push_back(scratchStr.append("-D__HIP_ARCH_").append(target).append("__=1"));
    } else {
      int major, minor;
      ::sscanf(device().info().version_, "OpenCL %d.%d ", &major, &minor);
      std::stringstream ss;
      ss << "-D__OPENCL_VERSION__=" << (major * 100 + minor * 10);
      optionsVec.push_back(ss.str());
    }
  }

  if (!isHIP()) {
    if (device().info().imageSupport_ && options->oVariables->ImageSupport) {
      optionsVec.push_back("-D__IMAGE_SUPPORT__=1");
    }

      uint clcStd =
        (options->oVariables->CLStd[2] - '0') * 100 + (options->oVariables->CLStd[4] - '0') * 10;

      if (clcStd >= 200) {
        std::stringstream opts;
        // Add only for CL2.0 and later
        opts << "-D"
          << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=" << device().info().maxGlobalVariableSize_;
        optionsVec.push_back(opts.str());
      }

    if (!device().settings().useLightning_) {
      if (!device().settings().singleFpDenorm_) {
        optionsVec.push_back("-cl-denorms-are-zero");
      }

      // Check if the host is 64 bit or 32 bit
      LP64_ONLY(optionsVec.push_back("-m64"));
    }

    // Tokenize the extensions string into a vector of strings
    std::istringstream istrstr(device().info().extensions_);
    std::istream_iterator<std::string> sit(istrstr), end;
    std::vector<std::string> extensions(sit, end);

    if (isLC()) {
      // FIXME_lmoriche: opencl-c.h defines 'cl_khr_depth_images', so
      // remove it from the command line. Should we fix opencl-c.h?
      auto found = std::find(extensions.begin(), extensions.end(), "cl_khr_depth_images");
      if (found != extensions.end()) {
        extensions.erase(found);
      }

      if (!extensions.empty()) {
        std::ostringstream clext;

        clext << "-cl-ext=+";
        std::copy(extensions.begin(), extensions.end() - 1,
          std::ostream_iterator<std::string>(clext, ",+"));
        clext << extensions.back();

        optionsVec.push_back("-Xclang");
        optionsVec.push_back(clext.str());
      }
    } else {
      for (auto e : extensions) {
        scratchStr.clear();
        optionsVec.push_back(scratchStr.append("-D").append(e).append("=1"));
      }
    }
  }

  return optionsVec;
}

std::string Program::ProcessOptionsFlattened(amd::option::Options* options) {
  std::vector<std::string> processOptions = ProcessOptions(options);
  std::ostringstream processOptionsOStrStr;
  processOptionsOStrStr << " ";
  std::copy(processOptions.begin(), processOptions.end(),
    std::ostream_iterator<std::string>(processOptionsOStrStr, " "));
  return processOptionsOStrStr.str();
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
    if (!amd::option::parseAllOptions(program->compileOptions_, *thisCompileOptions, false, isLC())) {
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
        if (!amd::option::parseLinkOptions(program->linkOptions_, thisLinkOptions, isLC())) {
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
  {
    DevLogPrintfError("Invalid Argument, Image: 0x%x Length: %u \n",
                      Image, Length);
    return false;
  }
  auto Magic = static_cast<const unsigned*>(Image);
  return *Magic == SPRVMagicNumber;
}

// ================================================================================================
bool Program::initClBinary(const char* binaryIn, size_t size) {
  if (!initClBinary()) {
    DevLogError("Init CL Binary failed \n");
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
      DevLogError("Cannot Decrypt Elf \n");
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
      DevLogError("Bin is not ELF \n");
      return false;
    }
  }

  clBinary()->setFlags(encryptCode);

  return clBinary()->setBinary(bin, sz, (decryptedBin != nullptr));
}

// ================================================================================================
bool Program::setBinary(const char* binaryIn, size_t size) {
  if (!initClBinary(binaryIn, size)) {
    DevLogError("Init CL Binary failed \n");
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
      if (clBinary()->elfIn()->isHsaCo()) {
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
#if defined(USE_COMGR_LIBRARY)
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
    if (!amd::option::parseAllOptions(sCurOptions, curOptions, false, isLC())) {
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
#endif   // defined(USE_COMGR_LIBRARY)
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
    if (!amd::option::parseAllOptions(sCurOptions, curOptions, false, isLC())) {
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
      if (!amd::option::parseAllOptions(sBinOptions, binOptions, false, isLC())) {
        buildLog_ += binOptions.optionsLog();
        LogError("Parsing compile options from binary failed.");
        return ACL_TYPE_DEFAULT;
      }
      if (!amd::option::parseAllOptions(sCurOptions, curOptions, false, isLC())) {
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
#if defined(USE_COMGR_LIBRARY)
bool Program::createKernelMetadataMap() {

  amd_comgr_status_t status;
  amd_comgr_metadata_node_t kernelsMD;
  bool hasKernelMD = false;
  size_t size = 0;

  status = amd::Comgr::metadata_lookup(metadata_, "Kernels", &kernelsMD);
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    ClPrint(amd::LOG_INFO, amd::LOG_CODE, "Using Code Object V2.");
    hasKernelMD = true;
    codeObjectVer_ = 2;
  }
  else {
    status = amd::Comgr::metadata_lookup(metadata_, "amdhsa.kernels", &kernelsMD);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "Using Code Object V3.");
      hasKernelMD = true;
      codeObjectVer_ = 3;
    }
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_metadata_list_size(kernelsMD, &size);
  } else if (amd::IS_HIP) {
    // Assume an empty binary. HIP may have binaries with just global variables
    return true;
  }

  for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
    amd_comgr_metadata_node_t nameMeta;
    bool hasNameMeta = false;
    bool hasKernelNode = false;

    amd_comgr_metadata_node_t kernelNode;

    std::string kernelName;
    status = amd::Comgr::index_list_metadata(kernelsMD, i, &kernelNode);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      hasKernelNode = true;
      status = amd::Comgr::metadata_lookup(kernelNode,
                                           (codeObjectVer() == 2) ? "Name" : ".name",
                                           &nameMeta);
    }

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      hasNameMeta = true;
      status  = getMetaBuf(nameMeta, &kernelName);
    }

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      kernelMetadataMap_[kernelName] = kernelNode;
    }
    else {
      if (hasKernelNode) {
        amd::Comgr::destroy_metadata(kernelNode);
      }
      for (auto const& kernelMeta : kernelMetadataMap_) {
        amd::Comgr::destroy_metadata(kernelMeta.second);
      }
      kernelMetadataMap_.clear();
    }

    if (hasNameMeta) {
      amd::Comgr::destroy_metadata(nameMeta);
    }
  }

  if (hasKernelMD) {
    amd::Comgr::destroy_metadata(kernelsMD);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}
#endif

bool Program::FindGlobalVarSize(void* binary, size_t binSize) {
#if defined(USE_COMGR_LIBRARY)
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
  bool metadata_found = false;
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
        else if ((note->n_type == 10 /* NT_AMD_AMDGPU_HSA_METADATA V2 */ &&
                  note->n_namesz == sizeof "AMD" && !memcmp(name, "AMD", note->n_namesz)) ||
                (note->n_type == 32 /* NT_AMD_AMDGPU_HSA_METADATA V3 */ &&
                  note->n_namesz == sizeof "AMDGPU" && !memcmp(name, "AMDGPU", note->n_namesz))) {
          amd_comgr_status_t status;
          amd_comgr_data_t binaryData;

          status  = amd::Comgr::create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &binaryData);
          if (status == AMD_COMGR_STATUS_SUCCESS) {
            status = amd::Comgr::set_data(binaryData, binSize,
                                          reinterpret_cast<const char*>(binary));
          }

          if (status == AMD_COMGR_STATUS_SUCCESS) {
            status = amd::Comgr::get_data_metadata(binaryData, &metadata_);
          }

          amd::Comgr::release_data(binaryData);

          if (status != AMD_COMGR_STATUS_SUCCESS) {
            buildLog_ += "Error: COMGR fails to get the metadata.\n";
            return false;
          }
          metadata_found = true;
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

  if (!metadata_found) {
    buildLog_ += "Error: runtime metadata section not present in ELF program binary\n";
    return false;
  }

  if (!createKernelMetadataMap()) {
    buildLog_ +=
      "Error: create kernel metadata map using COMgr\n";
    return false;
  }

  progvarsTotalSize -= dynamicSize;
  setGlobalVariableTotalSize(progvarsTotalSize);

  if (progvarsWriteSize != dynamicSize) {
    hasGlobalStores_ = true;
  }
#endif // defined(USE_COMGR_LIBRARY)
  return true;
}

#if defined(USE_COMGR_LIBRARY)
amd_comgr_status_t getSymbolFromModule(amd_comgr_symbol_t symbol, void* userData) {
  size_t nlen = 0;
  size_t* userDataInfo = nullptr;
  amd_comgr_status_t status;
  amd_comgr_symbol_type_t type;
  std::vector<std::string>* var_names = nullptr;

  /* Unpack the user data */
  SymbolInfo* sym_info = reinterpret_cast<SymbolInfo*>(userData);

  if (!sym_info) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  /* Retrieve the symbol info */
  status = amd::Comgr::symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, &nlen);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  /* Retrieve the symbol name */
  char* name = new char[nlen + 1];
  status = amd::Comgr::symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME, name);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  /* Retrieve the symbol type*/
  status = amd::Comgr::symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_TYPE, &type);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return status;
  }

  /* If symbol type is object(Variable) add it to vector */
  if ((std::strcmp(name, "") != 0) && (type == sym_info->sym_type)) {
    sym_info->var_names->push_back(std::string(name));
  }

  delete[] name;
  return status;
}

bool Program::getSymbolsFromCodeObj(std::vector<std::string>* var_names, amd_comgr_symbol_type_t sym_type) const {
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
  amd_comgr_data_t dataObject;
  SymbolInfo sym_info;
  bool ret_val = true;

  do {
    /* Create comgr data */
    status = amd::Comgr::create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &dataObject);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot create comgr data \n";
      ret_val = false;
      break;
    }

    /* Set the binary as a dataObject */
    status = amd::Comgr::set_data(dataObject,static_cast<size_t>(clBinary_->data().second),
                                  reinterpret_cast<const char*>(clBinary_->data().first));
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot set comgr data \n";
      ret_val = false;
      break;
    }

    /* Pack the user data */
    sym_info.sym_type = sym_type;
    sym_info.var_names = var_names;

  /* Iterate through list of symbols */
    status = amd::Comgr::iterate_symbols(dataObject, getSymbolFromModule, &sym_info);
    if (status != AMD_COMGR_STATUS_SUCCESS) {
      buildLog_ += "COMGR:  Cannot iterate comgr symbols \n";
      ret_val = false;
      break;
    }
  } while (0);

  return ret_val;
}
#endif /* USE_COMGR_LIBRARY */

const bool Program::getLoweredNames(std::vector<std::string>* mangledNames) const {
#if defined (USE_COMGR_LIBRARY)
  /* Iterate thru kernel names first */
  for (auto const& kernelMeta : kernelMetadataMap_) {
    mangledNames->emplace_back(kernelMeta.first);
  }

  /* Itrate thru global vars */
  if (!getSymbolsFromCodeObj(mangledNames, AMD_COMGR_SYMBOL_TYPE_OBJECT)) {
    DevLogError("Cannot get Symbols from Code Obj \n");
    return false;
  }

  return true;

#else
  assert("No COMGR loaded");
  return false;
#endif
}

bool Program::getGlobalFuncFromCodeObj(std::vector<std::string>* func_names) const {
#if defined(USE_COMGR_LIBRARY)
  return getSymbolsFromCodeObj(func_names, AMD_COMGR_SYMBOL_TYPE_FUNC);
#else
  return true;
#endif
}

bool Program::getGlobalVarFromCodeObj(std::vector<std::string>* var_names) const {
#if defined(USE_COMGR_LIBRARY)
  return getSymbolsFromCodeObj(var_names, AMD_COMGR_SYMBOL_TYPE_OBJECT);
#else
  return true;
#endif
}

bool Program::getUndefinedVarFromCodeObj(std::vector<std::string>* var_names) const {
#if defined(USE_COMGR_LIBRARY)
  return getSymbolsFromCodeObj(var_names, AMD_COMGR_SYMBOL_TYPE_NOTYPE);
#else
  return true;
#endif
}

bool Program::getUndefinedVarInfo(std::string var_name, void** var_addr, size_t* var_size) {
  if (owner()->varcallback != nullptr) {
    return owner()->varcallback(as_cl(owner()), var_name.c_str(), var_addr, var_size);
  } else {
    buildLog_ += "SVAR HIP Call back is not set \n";
    return false;
  }
}

bool Program::defineUndefinedVars() {
  size_t address = 0;
  size_t hsize = 0;
  void* dptr = nullptr;
  void* hptr = nullptr;
  device::Memory* dev_mem = nullptr;
  amd::Memory* amd_mem_obj = nullptr;
  std::vector<std::string> var_names;

  if (!getUndefinedVarFromCodeObj(&var_names)) {
    DevLogError("Cannot get Undefined Var from Code Object \n");
    return false;
  }

  for (auto it = var_names.begin(); it != var_names.end(); ++it) {
    if (!getUndefinedVarInfo(*it, &hptr, &hsize)) {
      continue;
    }

    amd_mem_obj = new (owner()->context()) amd::Buffer(const_cast<amd::Context&>(owner()->context()),
                                                       CL_MEM_USE_HOST_PTR, hsize);
    if (amd_mem_obj == nullptr) {
      LogError("[OCL] failed to create a mem object!");
      return false;
    }

    if (!amd_mem_obj->create(hptr)) {
      LogError("[OCL] failed to create a svm hidden buffer!");
      amd_mem_obj->release();
      return false;
    }

    undef_mem_obj_.push_back(amd_mem_obj);

    dev_mem = amd_mem_obj->getDeviceMemory(device());
    if (dev_mem == nullptr) {
      LogError("[OCL] failed to create a mem object!");
      return false;
    }

    dptr = reinterpret_cast<void*>(dev_mem->virtualAddress());
    if (dev_mem == nullptr) {
      LogError("[OCL] failed to create a mem object!");
      return false;
    }

    if(!defineGlobalVar(it->c_str(), dptr)) {
      LogError("[OCL] failed to define global var");
      return false;
    }
  }

  return true;
}

} /* namespace device*/
