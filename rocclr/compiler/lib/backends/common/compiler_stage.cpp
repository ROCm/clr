//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#include "compiler_stage.hpp"
#include "utils/libUtils.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/LLVMContext.h"

using namespace amdcl;

CompilerStage::CompilerStage(aclCompiler* cl, aclBinary* elf,
    aclLogFunction callback)
: cl_(cl), elf_(elf), binary_(NULL),
  source_(""), log_(""), callback_(callback)
{
  opts_ = (amd::option::Options*)Elf()->options;
}

CompilerStage::~CompilerStage()
{ }

LLVMCompilerStage::LLVMCompilerStage(aclCompiler *cl, aclBinary *elf,
    aclLogFunction callback)
: CompilerStage(cl, elf, callback),
  llvmbinary_(NULL),
  context_(NULL)
{
  if (!Options()->oVariables->DisableAllWarnings) {
    hookup_.LLVMBuildLog = &log_;
  }
  // Expose some options to LLVM.
  llvm::AMDOptions *amdopts = &hookup_.amdoptions;
  amdopts->OptLiveness = Options()->oVariables->OptLiveness;
  if (isHSAILTarget(Elf()->target)) {
    if ((amdopts->NumAvailGPRs == ~0u) || (Options()->NumAvailGPRs != -1))
      amdopts->NumAvailGPRs = Options()->NumAvailGPRs;
  } else {
    amdopts->OptPrintLiveness = Options()->oVariables->OptPrintLiveness;
    amdopts->OptMem2reg = Options()->oVariables->OptMem2reg;
    amdopts->UseJIT = Options()->oVariables->UseJIT;
    amdopts->APThreshold = Options()->oVariables->APThreshold;
    amdopts->AAForBarrier = Options()->oVariables->AAForBarrier;
    amdopts->UnrollScratchThreshold = 500;
    amdopts->AmdilUseDefaultResId = Options()->oVariables->DefaultResourceId;
  }
  amdopts->OptSimplifyLibCall = Options()->oVariables->OptSimplifyLibCall;
  amdopts->EnableFDiv2FMul = Options()->oVariables->EnableFDiv2FMul;
  amdopts->SRThreshold = Options()->oVariables->SRThreshold;
  amdopts->OptMemCombineMaxVecGen = Options()->oVariables->OptMemCombineMaxVecGen;
  amdopts->OptLICM = Options()->oVariables->OptLICM;

  // math-related options
  amdopts->UnsafeMathOpt = Options()->oVariables->UnsafeMathOpt;
  amdopts->NoSignedZeros = Options()->oVariables->NoSignedZeros;
  amdopts->FiniteMathOnly = Options()->oVariables->FiniteMathOnly;
  amdopts->FastRelaxedMath = Options()->oVariables->FastRelaxedMath;

  amdopts->LUThreshold = Options()->oVariables->LUThreshold;
  amdopts->LUCount = Options()->oVariables->LUCount;
  amdopts->LUAllowPartial = Options()->oVariables->LUAllowPartial;
  amdopts->GPUArch = (uint32_t)getLibraryType(&elf->target);
}

  void
LLVMCompilerStage::setContext(aclContext *ctx)
{
  context_ = reinterpret_cast<llvm::LLVMContext*>(ctx);
  if (ctx) {
    Context().setAMDLLVMContextHook(&hookup_);
  }
}

LLVMCompilerStage::~LLVMCompilerStage()
{
  if (context_) {
    Context().setAMDLLVMContextHook(NULL);
  }
}

  llvm::Module*
LLVMCompilerStage::loadBitcode(std::string& llvmBinary)
{
  if (!llvm::isBitcode(reinterpret_cast<const unsigned char*>(llvmBinary.data()),
        reinterpret_cast<const unsigned char*>(llvmBinary.data()
          + llvmBinary.length()))) {
    llvm::SMDiagnostic diags;
    return ParseAssemblyString(llvmBinary.c_str(), llvmbinary_, diags, Context());

  }
  // Use getMemBuffer() ?
  if (llvm::MemoryBuffer *Buffer =
      llvm::MemoryBuffer::getMemBufferCopy(
        llvm::StringRef(llvmBinary), "input.bc")) {
    std::string ErrorMessage;
    llvm::Module* M =
      llvm::ParseBitcodeFile(Buffer, Context(), &ErrorMessage);
    delete Buffer;
    return M;
  }
  return NULL;
}
