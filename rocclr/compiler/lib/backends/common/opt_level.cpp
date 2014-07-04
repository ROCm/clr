//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "opt_level.hpp"
#include "library.hpp"
#include "utils/options.hpp"
#include "llvm/Module.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/DataLayout.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Transforms/IPO/AMDOptOptions.h"
#include "compiler_stage.hpp"
using namespace amdcl;
using namespace llvm;

void
OptLevel::setup(bool isGPU, uint32_t OptLevel)
{
  // Add an appropriate DataLayout instance for this module.
  Passes().add(new DataLayout(module_));
  fpasses_ = new FunctionPassManager(module_);
  fpasses_->add(new DataLayout(module_));

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
      if (HLC_Experimental_Enable_Calls) {
        HLC_Disable_Amd_Inline_All = true;
      }
      // Always create Inliner regardless of OptLevel
      if (HLC_Force_Always_Inliner_Pass) {
        Builder.Inliner = createAlwaysInlinerPass();
      } else {
        Builder.Inliner = createFunctionInliningPass(500);
      }
    } else 
#endif
    {
      // Always create Inliner regardless of OptLevel
      Builder.Inliner = createFunctionInliningPass(500);
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
        Builder.Inliner = createFunctionInliningPass(Threshold);
      }
    } else 
#endif
    {
      Builder.Inliner = createFunctionInliningPass(Threshold);
    }
  }
  Builder.SizeLevel = 0;
  Builder.DisableUnitAtATime = false;
  Builder.DisableUnrollLoops = OptLevel == 0;
  if (Options()->libraryType_ != amd::GPU_Library_HSAIL)
    Builder.DisableSimplifyLibCalls = true;
  Builder.AMDpopulateFunctionPassManager(*fpasses_, &module_->getContext());
  Builder.AMDpopulateModulePassManager(passes_, &module_->getContext(), module_);
}

void
OptLevel::run()
{
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
O0OptLevel::optimize(Module *input, bool isGPU)
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
    run();
  }
  return 0;
}

int
GPUO0OptLevel::optimize(Module *input, bool isGPU)
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
    Passes().add(createAMDSymbolLinkagePass(true, NULL));
    Passes().add(createGlobalOptimizerPass());
    if (!HLC_Disable_Amd_Inline_All && !DisableInline ) {
      if (HLC_Force_Always_Inliner_Pass) {
        Passes().add(createAlwaysInlinerPass());
      } else {
        Passes().add(createAMDInlineAllPass(true));
      }
    }
  }
#endif
  run();
  return 0;
}

int
O1OptLevel::optimize(Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 1);
  run();
  return 0;
}

int
O2OptLevel::optimize(Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 2);
  run();
  return 0;
}

int
O3OptLevel::optimize(Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 3);
  run();
  return 0;
}

int
O4OptLevel::optimize(Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 4);
  run();
  return 0;
}

int
OsOptLevel::optimize(Module *input, bool isGPU)
{
  module_ = input;
  setup(isGPU, 5);
  run();
  return 0;
}
