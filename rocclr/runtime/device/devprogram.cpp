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

#include "spirv/spirvUtils.h"

#include <string>
#include <sstream>

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
Program::~Program() { clear(); }

// ================================================================================================
void Program::clear() {
  // Destroy all device kernels
  for (const auto& it : kernels_) {
    delete it.second;
  }
  kernels_.clear();
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
  programOptions_ = options;

  if (options->oVariables->DumpFlags > 0) {
    static amd::Atomic<unsigned> build_num = 0;
    options->setBuildNo(build_num++);
  }
  buildLog_.clear();
  if (!initClBinary()) {
    return false;
  }
  return true;
}

// ================================================================================================
bool Program::finiBuild(bool isBuildGood) { return true; }

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

  optionsStr.append(" -D__").append(device().info().name_).append("__=1");
  optionsStr.append(" -D__").append(device().info().name_).append("=1");
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

#if !defined(WITH_LIGHTNING_COMPILER)
  if (!device().settings().singleFpDenorm_) {
    optionsStr.append(" -cl-denorms-are-zero");
  }

  // Check if the host is 64 bit or 32 bit
  LP64_ONLY(optionsStr.append(" -m64"));
#endif  // !defined(WITH_LIGHTNING_COMPILER)

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

#if !defined(WITH_LIGHTNING_COMPILER)
  bool isSPIRV = isSPIRVMagic(binaryIn, size);
  if (isSPIRV || isBcMagic(binaryIn)) {
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
  } else
#endif  // !defined(WITH_LIGHTNING_COMPILER)
  {
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

}
