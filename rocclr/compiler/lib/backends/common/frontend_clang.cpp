//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "OpenCLFE.h"

#include "bif/bifbase.hpp"
#include "frontend.hpp"
#include "os/os.hpp"
#include "top.hpp"
#include "utils/options.hpp"
#include "utils/target_mappings.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SPIRV.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <sstream>

amdcl::ClangOCLFrontend::ClangOCLFrontend(aclCompiler* cl, aclBinary* elf,
                                          aclLogFunction log)
       : Frontend(cl, elf, log){}

/// @brief This function generates the required command-line options to
/// call the ClangOCLFE library.
int amdcl::ClangOCLFrontend::compileCommand(const std::string& src) {

  std::vector<const char*> argsToClang;
  std::string tempFileName = amd::Os::getTempFileName();
  std::string logFileName = tempFileName + ".log";
  std::string inpCLFileName = tempFileName + ".cl";
  std::string logFromClang;
  int ret = 0;

  aclBinary *elf = Elf();
  amd::option::Options* amdOpts = (amd::option::Options*)elf->options;

  // Following are the options passed to the ClangOCLFE library
  // and then to Clang itself.

  // Passing the compiler FE options to clang.
  if (amdOpts) {
    for (std::vector<std::string>::const_iterator it = amdOpts->clangOptions.begin();
         it != amdOpts->clangOptions.end(); ++it) {
      argsToClang.push_back((*it).c_str());
    }
  }
  if (Options()->oVariables->ImageSupport) {
    argsToClang.push_back ("-D__IMAGE_SUPPORT__=1 ");
  }

  if (Options()->oVariables->FastFMA) {
    argsToClang.push_back ("-DFP_FAST_FMA=1 ");
  }

  if (Options()->oVariables->FastFMAF) {
    argsToClang.push_back ("-DFP_FAST_FMAF=1 ");
  }

  argsToClang.push_back ("-D__AMD__=1 ");

  if (Options()->oVariables->FEGenSPIRV) {
    argsToClang.push_back("-D__AMD_SPIRV__ ");
  }

  // Other options are passed using OptionsInfo structure.
  clc2::OptionsInfo ClangOptions;

  ClangOptions.InFilename = inpCLFileName;

  // Generate target triple.
  // TODO: Refine the triple as necessary.
  uint32_t chipName = elf->target.chip_id;
  assert(chipName < familySet[elf->target.arch_id].children_size &&
         "Cannot index past end of array!");
  switch (elf->target.arch_id) {
    default:
      log_ += "\nerror: Unknown target device ID!\n";
      ret |= 1;
      return ret;
      break;
    case aclX86:
    case aclAMDIL:
    case aclHSAIL:
      // See bug: http://ocltc.amd.com/bugs/show_bug.cgi?id=9631
      if (sizeof(void*) != 4) {
        log_ += "\nerror: 32-bit kernels not supported on a 64-bit executable\n";
        ret |= 1;
        return ret;
      }
      ClangOptions.TargetArch = llvm::Triple::spir;
      break;
    case aclX64:
    case aclAMDIL64:
    case aclHSAIL64:
      // See bug: http://ocltc.amd.com/bugs/show_bug.cgi?id=9631
      if (sizeof(void*) != 8) {
        log_ += "\nerror: 64-bit kernels not supported on a 32-bit executable\n";
        ret |= 1;
        return ret;
      }
      ClangOptions.TargetArch = llvm::Triple::spir64;
      break;
  };

  // Copy the source to a buffer. Note that the input
  // file itself is not passed to the ClangOCLFE library. It is a passed
  // as a string for compilation.

  std::unique_ptr<llvm::MemoryBuffer> srcBufferPtr =
       llvm::MemoryBuffer::getMemBuffer(src, inpCLFileName.c_str(),
                                        true);
  ClangOptions.Src.swap(srcBufferPtr);
  assert(ClangOptions.Src.get() && "ClangOCLFE: Memory Buffer"
                    " initialization error\n");

  // Set Pre-processor output if user asks for it.
  if (amdOpts && amdOpts->isDumpFlagSet(amd::option::DUMP_I)) {
    ClangOptions.PreProcOut = amdOpts->getDumpFileName(".i");
  }

  // Set the LLVMContext for the front-end compilation.
  ClangOptions.CompilerContext = &Context();

  if (amdOpts && amdOpts->isDumpFlagSet(amd::option::DUMP_CL)) {
    dumpSource(src, amdOpts);
  }

  //Start the compilation
  uint64_t start_time = 0, stop_time = 0;

  if (Options()->oVariables->EnableBuildTiming) {
    start_time = amd::Os::timeNanos();
  }

  if (!checkFlag(aclutGetCaps(Elf()), capSaveSOURCE)) {
    CL()->clAPI.remSec(CL(), Elf(), aclSOURCE);
  }

  // Pass OpenCL version option to Clang
  llvm::StringRef OCLVer(amdOpts->oVariables->CLStd);
  if (OCLVer.equals("CL1.2")) {
    ClangOptions.OCLVer = clc2::OCL_12;
  } else if (OCLVer.equals("CL2.0")) {
    ClangOptions.OCLVer = clc2::OCL_20;
  } else {
    llvm_unreachable("Unknown OpenCL version");
  }

  // Call the Clang Front-end to generate serialized llvm::Module
  // from the OpenCL source.
#ifdef ANDROID
  // We will not exercise Clang for RenderScript.
  log_ += "\nerror: Clang front-end compilation unsupported on Android!\n";
  ret |= 1;
  return ret;
#else
  if (!parseOCLSource(ClangOptions, argsToClang, &Source(), &logFromClang)) {
      log_ += logFromClang;
      log_ += "\nerror: Clang front-end compilation failed!\n";
      ret |= 1;
      return ret;
  }
#endif

  if (Options()->oVariables->EnableBuildTiming) {
    stop_time = amd::Os::timeNanos();
    std::stringstream tmp_ss;
    tmp_ss << "    OpenCL FE time: "
      << (stop_time - start_time)/1000ULL
      << "us\n";
    appendLogToCL(CL(), tmp_ss.str());
  }

  llvmbinary_ = loadBitcode(Source());

  if (!llvmbinary_) {
    ret |= 1;
  }

  if (!ret) {
    CL()->clAPI.insSec(CL(), Elf(), Source().data(), Source().size(), aclLLVMIR);
  }

  if (Options()->oVariables->FEGenSPIRV) {
    std::ostringstream ss;
    std::string err;

    if (Options()->getLLVMArgc()) {
      llvm::cl::ParseCommandLineOptions(Options()->getLLVMArgc(),
        Options()->getLLVMArgv(), "LLVM/SPIRV converter");
    }
    if (WriteSPRV(llvmbinary_, ss, err)) {
      std::string img = ss.str();
      CL()->clAPI.insSec(CL(), Elf(), img.data(), img.size(), aclSPIRV);
    }

    if (!log_.empty())
      log_ += std::string(" ");
    log_ += err;
  }

  log_ += logFromClang;
  if (isCpuTarget(Elf()->target)
      && Options()->oVariables->EnableDebug) {
    Options()->sourceFileName_ = inpCLFileName;
  } else {
    amd::Os::unlink(inpCLFileName.c_str());
  }
  return ret;
}
