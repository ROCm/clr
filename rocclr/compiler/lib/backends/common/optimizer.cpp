//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "optimizer.hpp"
#include "opt_level.hpp"
#include "os/os.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"
#include "utils/options.hpp"

#if defined(LEGACY_COMPLIB)
#include "llvm/DataLayout.h"
#else
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/FileSystem.h"
#include "AMDPasses.h"
#endif
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/LinkAllPasses.h"
#include <cassert>
#include <sstream>
using namespace amdcl;
using namespace llvm;

static OptLevel* getOptLevel(amd::option::Options* Options, bool isGPU) {
  switch(Options->oVariables->OptLevel) {
    case amd::option::OPT_O0:
      return (isGPU) ? new GPUO0OptLevel(Options) : new O0OptLevel(Options);
    case amd::option::OPT_O1:
      return new O1OptLevel(Options);
    default:
      assert(!"Found an invalid optimization level!");
    case amd::option::OPT_O2:
      return new O2OptLevel(Options);
    case amd::option::OPT_O3:
      return new O3OptLevel(Options);
    case amd::option::OPT_O4:
      return new O4OptLevel(Options);
    case amd::option::OPT_OS:
      return new OsOptLevel(Options);
  }
  assert(!"Unreachable!");
  return NULL;
}
int
CPUOptimizer::preOptimizer(llvm::Module* M)
{
#if defined(LEGACY_COMPLIB)
    llvm::PassManager Passes;
    Passes.add(new llvm::DataLayout(M));
#else
    llvm::legacy::PassManager Passes;
#endif

    Passes.add(createAMDExportKernelNaturePass());

    Passes.run(*M);

    return 0;
}

  int
CPUOptimizer::optimize(llvm::Module *input)
{
  if (!input) {
    return 1;
  }
  int ret = 0;
  uint64_t start_time = 0ULL, time_opt = 0ULL;
  llvmbinary_ = input;
  setWholeProgram(true);

  setGPU(false);
  if (Options()->oVariables->EnableBuildTiming) {
    start_time = amd::Os::timeNanos();
  }
  ret = preOptimizer(LLVMBinary());
  setUniformWorkGroupSize(Options()->oVariables->UniformWorkGroupSize);
  OptLevel* cpuOpt = getOptLevel(Options(), false);
  if (Options()->oVariables->EnableBuildTiming) {
    time_opt = amd::Os::timeNanos();
  }
  ret = cpuOpt->optimize(Elf(), LLVMBinary(), false);
  if (Options()->oVariables->EnableBuildTiming) {
    time_opt = amd::Os::timeNanos() - time_opt;
    std::stringstream tmp_ss;
    tmp_ss << "    LLVM Opt time: "
      << time_opt/1000ULL
      << "us\n";
    appendLogToCL(CL(), tmp_ss.str());
  }
  delete cpuOpt;

  if ( ret ) {
    BuildLog() += "Internal Error: optimizer failed!\n";
    return 1;
  }
  if (Options()->isDumpFlagSet(amd::option::DUMP_BC_OPTIMIZED)) {
    std::string fileName = Options()->getDumpFileName("_optimized.bc");
#if defined(LEGACY_COMPLIB)
    std::string MyErrorInfo;
    raw_fd_ostream outs (fileName.c_str(), MyErrorInfo, raw_fd_ostream::F_Binary);
    // FIXME: Need to add this to the elf binary!
    if (MyErrorInfo.empty())
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(MyErrorInfo.c_str());
#else
    std::error_code EC;
    llvm::raw_fd_ostream outs(fileName.c_str(), EC, llvm::sys::fs::F_None);
    // FIXME: Need to add this to the elf binary!
    if (!EC)
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(EC.message().c_str());
#endif
  }
  return ret;
}

  int
GPUOptimizer::optimize(llvm::Module *input)
{
  if (!input) {
    return 1;
  }
  int ret = 0;
  uint64_t start_time = 0ULL, time_opt = 0ULL;
  llvmbinary_ = input;

  setGPU(true);
  setWholeProgram(true);
#ifdef WITH_TARGET_HSAIL
  if (isHSAILTarget(Elf()->target)) {
    if (Options()->NumAvailGPRs == -1)
      Options()->NumAvailGPRs = 128; // Default HSAIL number of GPRs
    if (hookup_.amdoptions.NumAvailGPRs == ~0u)
      hookup_.amdoptions.NumAvailGPRs = Options()->NumAvailGPRs;
  }
#endif
  setUniformWorkGroupSize(Options()->oVariables->UniformWorkGroupSize);
  OptLevel* gpuOpt = getOptLevel(Options(), true);
  if (Options()->oVariables->EnableBuildTiming) {
    time_opt = amd::Os::timeNanos();
  }
  ret = gpuOpt->optimize(Elf(), LLVMBinary(), true);
  if (Options()->oVariables->EnableBuildTiming) {
    time_opt = amd::Os::timeNanos() - time_opt;
    std::stringstream tmp_ss;
    tmp_ss << "    LLVM Opt time: "
      << time_opt/1000ULL
      << "us\n";
    appendLogToCL(CL(), tmp_ss.str());
  }
  delete gpuOpt;

  if ( ret ) {
    BuildLog() += "Internal Error: optimizer failed!\n";
    return 1;
  }
  if (Options()->isDumpFlagSet(amd::option::DUMP_BC_OPTIMIZED)) {
    std::string fileName = Options()->getDumpFileName("_optimized.bc");
#if defined(LEGACY_COMPLIB)
    std::string MyErrorInfo;
    raw_fd_ostream outs (fileName.c_str(), MyErrorInfo, raw_fd_ostream::F_Binary);
    // FIXME: Need to add this to the elf binary!
    if (MyErrorInfo.empty())
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(MyErrorInfo.c_str());
#else
    std::error_code EC;
    llvm::raw_fd_ostream outs(fileName.c_str(), EC, llvm::sys::fs::F_None);
    // FIXME: Need to add this to the elf binary!
    if (!EC)
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(EC.message().c_str());
#endif
  }
  return ret;
}
