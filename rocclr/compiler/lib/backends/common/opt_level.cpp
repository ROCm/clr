//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "opt_level.hpp"
#include "library.hpp"
#include "acl.h"
#include "utils/options.hpp"
#include "utils/target_mappings.h"
#include "utils/libUtils.h"
#include "llvm/Analysis/Passes.h"
#if defined(LEGACY_COMPLIB)
#include "llvm/DataLayout.h"
#include "llvm/Module.h"
#endif
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Transforms/IPO/AMDOptOptions.h"
#include "compiler_stage.hpp"
using namespace amdcl;
using namespace llvm;

void
OptLevel::setup(bool isGPU, uint32_t OptLevel)
{
  // Add an appropriate DataLayout instance for this module.
#if defined(LEGACY_COMPLIB)
  Passes().add(new DataLayout(module_));
#else
  Passes().add(new DataLayoutPass());
#endif
  fpasses_ = new FunctionPassManager(module_);
#if defined(LEGACY_COMPLIB)
  fpasses_->add(new DataLayout(module_));
#else
  fpasses_->add(new DataLayoutPass());
#endif

  PassManagerBuilder Builder;
  Builder.OptLevel = OptLevel;

  if (Options()->libraryType_ == amd::GPU_Library_HSAIL) {
    if (OptLevel == 0) return;
  }

  if (!Options()->oVariables->Inline) {
    // No inlining pass
  } else if (isGPU) {
#ifdef WITH_TARGET_HSAIL
    if (Options()->libraryType_ == amd::GPU_Library_HSAIL) {
      if (HLC_HSAIL_Enable_Calls) {
        HLC_Disable_Amd_Inline_All = true;
      } else {
        HLC_Disable_Amd_Inline_All = false;
      }
      // Always create Inliner regardless of OptLevel
      if (HLC_Force_Always_Inliner_Pass) {
        Builder.Inliner = createAlwaysInlinerPass();
      } else {
        Builder.Inliner = createAMDFunctionInliningPass(HLC_HSAIL_Inline_Threshold);
      }
    } else
#endif
    {
      HLC_Disable_Amd_Inline_All = false;
      // Always create Inliner regardless of OptLevel
      Builder.Inliner = createAMDFunctionInliningPass(500);
    }
  } else if (OptLevel > 1) {
    unsigned Threshold = 225;
    if (OptLevel > 2)
      Threshold = 275;
#ifdef WITH_TARGET_HSAIL
    if (Options()->libraryType_ == amd::GPU_Library_HSAIL) {
      // Don't do inlining (including createAlwaysInlinerPass()) if OptimizationLevel
      // is zero becaue we are generating code for -g
      if (OptLevel > 0) {
        Builder.Inliner = createAMDFunctionInliningPass(Threshold);
      }
    } else
#endif
    {
      Builder.Inliner = createAMDFunctionInliningPass(Threshold);
    }
  }
  Builder.SizeLevel = 0;
  Builder.DisableUnitAtATime = false;
  Builder.DisableUnrollLoops = OptLevel == 0;
#if defined(LEGACY_COMPLIB)
  if (Options()->libraryType_ != amd::GPU_Library_HSAIL)
    Builder.DisableSimplifyLibCalls = true;
#endif
  Builder.AMDpopulateFunctionPassManager(*fpasses_, &module_->getContext());
  Builder.AMDpopulateModulePassManager(passes_, &module_->getContext(), module_);
}

void
OptLevel::run(aclBinary *elf)
{
#if !defined(LEGACY_COMPLIB)
  const aclTargetInfo* trg = aclutGetTargetInfo(elf);
  TargetMachine *Machine = nullptr;
  if (trg) {
    llvm::Triple TheTriple(getTriple(trg->arch_id));
    if (TheTriple.getArch()) {
      std::string Error;
      llvm::StringRef MArch(aclGetArchitecture(*trg));
      const Target *TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple,
                                                             Error);
      if (TheTarget) {
        llvm::TargetOptions targetOptions;
        targetOptions.NoFramePointerElim = false;
        targetOptions.StackAlignmentOverride = Options()->oVariables->CPUStackAlignment;
#ifdef WITH_TARGET_HSAIL
        if (Options()->libraryType_ == amd::GPU_Library_HSAIL)
          targetOptions.UnsafeFPMath = Options()->oVariables->UnsafeMathOpt;
#endif
        targetOptions.LessPreciseFPMADOption = Options()->oVariables->MadEnable ||
                                               Options()->oVariables->EnableMAD;
        targetOptions.NoInfsFPMath = Options()->oVariables->FiniteMathOnly;
        targetOptions.NoNaNsFPMath = Options()->oVariables->FastRelaxedMath;

        llvm::CodeGenOpt::Level OLvl = CodeGenOpt::None;
        switch (Options()->oVariables->OptLevel) {
        case 0: // -O0
          OLvl = CodeGenOpt::None;
          break;
        case 1: // -O1
          OLvl = CodeGenOpt::Less;
          break;
        case 2: // -O2
        case 5: // -O5(-Os)
          OLvl = CodeGenOpt::Default;
          break;
        case 3: // -O3
        case 4: // -O4
          OLvl = CodeGenOpt::Aggressive;
          break;
        default:
          assert(!"Error with optimization level");
        };

        Machine = TheTarget->createTargetMachine(TheTriple.getTriple(),
                                                 aclutGetCodegenName(elf->target),
                                                 "", targetOptions,
                                                 WINDOWS_SWITCH(Reloc::DynamicNoPIC, Reloc::PIC_),
                                                 CodeModel::Default, OLvl);
      }
    }
  }
  std::unique_ptr<TargetMachine> TM(Machine);
  if (TM.get())
    TM->addAnalysisPasses(passes_);
#endif

  if (Options()->oVariables->OptPrintLiveness) {
    Passes().add(createAMDLivenessPrinterPass());
  }
  fpasses_->doInitialization();
  for (Module::iterator I = module_->begin(), E = module_->end(); I != E; ++I)
    fpasses_->run(*I);
  fpasses_->doFinalization();
  // Now that we have all of the passes ready, run them.
  passes_.run(*module_);

  delete fpasses_;
}

int
O0OptLevel::optimize(aclBinary *elf, Module *input, bool isGPU)
{
  // With -O0, we don't do anything
  module_ = input;
#ifdef WITH_TARGET_HSAIL
  if (Options()->libraryType_ == amd::GPU_Library_HSAIL) {
    // Mark all non-kernel functions as having internal linkage
    Passes().add(createAMDSymbolLinkagePass(true, NULL));
  } else
#endif
  {
    setup(false, 0);
    run(elf);
  }
  return 0;
}

int
GPUO0OptLevel::optimize(aclBinary *elf, Module *input, bool isGPU)
{
  module_ = input;
  assert(isGPU && "Only a GPU can use GPUO0OptLevel!\n");
  setup(true, 0);
#ifdef WITH_TARGET_HSAIL
  if (Options()->libraryType_ == amd::GPU_Library_HSAIL) {
    // On the GPU, even with -O0, we must do some optimizations. One
    // goal is to ensure that all functions are inlined. This requires
    // three steps in that order:
    //
    // 1. Mark all non-kernel functions as having internal linkage.
    // 2. Invoke the GlobalOptimizer to resolve function aliases.
    // 3. Force inlining using our custom inliner pass.
    if (Options()->oVariables->EnableDebug) {
      HLC_HSAIL_Enable_Calls = false;
      HLC_Disable_Amd_Inline_All = false;
    }
    else if (HLC_HSAIL_Enable_Calls) {
      HLC_Disable_Amd_Inline_All = true;
    }
    else {
      HLC_Disable_Amd_Inline_All = false;
    }
    Passes().add(createAMDSymbolLinkagePass(true, NULL));
    Passes().add(createGlobalOptimizerPass());
    if (!HLC_Disable_Amd_Inline_All &&
        !DisableInline &&
        !HLC_Force_Always_Inliner_Pass) {
      Passes().add(createAMDInlineAllPass(true));
    } else {
      Passes().add(createAlwaysInlinerPass());
    }
  }
#endif
  run(elf);
  return 0;
}

int
O1OptLevel::optimize(aclBinary *elf, Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 1);
  run(elf);
  return 0;
}

int
O2OptLevel::optimize(aclBinary *elf, Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 2);
  run(elf);
  return 0;
}

int
O3OptLevel::optimize(aclBinary *elf, Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 3);
  run(elf);
  return 0;
}

int
O4OptLevel::optimize(aclBinary *elf, Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 4);
  run(elf);
  return 0;
}

int
OsOptLevel::optimize(aclBinary *elf, Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 5);
  run(elf);
  return 0;
}
