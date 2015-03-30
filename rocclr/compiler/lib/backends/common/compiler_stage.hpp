//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_COMPILER_STAGE_HPP_
#define _BE_COMPILER_STAGE_HPP_
#include "aclTypes.h"
#include "utils/options.hpp"
#include "llvm/AMDLLVMContextHook.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Bitcode/ReaderWriter.h"

#include <cassert>
#include <string>

namespace llvm
{
  class LLVMContext;
  class Module;
}

namespace amdcl
{
  /*! \addtogroup CompilerLibrary
   *
   * \copydoc amdcl::CompilerStage
   *
   *  @{
   */
  class CompilerStage {
    private:
      CompilerStage(); // DO NOT IMPLEMENT.
      CompilerStage(CompilerStage&); // DO NOT IMPLEMENT.
    public:
      CompilerStage(aclCompiler* cl, aclBinary* elf, aclLogFunction callback);

      virtual ~CompilerStage();

      /*! Returns the Compiler */
      aclCompiler* CL() const { return cl_; }

      /*! Returns the elf binary */
      aclBinary* Elf() const { return elf_; }

      /*! Returns the callback */
      aclLogFunction Callback() const { return callback_; }

      /*! Returns the options */
      amd::option::Options* Options() const {
          assert(opts_ && "Options should not be null");
          return opts_;
      }


      /*! Returns the source file */
      std::string& Source() { return source_; }

      /*! Returns the build log */
      std::string& BuildLog() { return log_; }

    protected:
      aclCompiler  *cl_;
      aclBinary       *elf_;
      void         *binary_;
      amd::option::Options* opts_;
      std::string   source_;
      std::string   log_;
      aclLogFunction callback_;
  }; // class CompilerStage

  class LLVMCompilerStage : public CompilerStage {
    public:
      LLVMCompilerStage(aclCompiler *cl, aclBinary *elf,
          aclLogFunction callback);
      virtual ~LLVMCompilerStage();
      void setContext(aclContext *ctx);

      /*! Returns the local context */
      llvm::LLVMContext& Context() { return (*context_); }

      /*! Loads bitcode in either text or binary format and return
       * and LLVM module. */
      virtual llvm::Module* loadBitcode(std::string& llvmBinary);

      void setGPU(bool isForGPU) { hookup_.amdoptions.IsGPU = isForGPU; }
      void setWholeProgram(bool Val) { hookup_.amdoptions.WholeProgram = Val; }
      void setNoSignedZeros(bool Val) { hookup_.amdoptions.NoSignedZeros = Val; }
      void setFastRelaxedMath(bool Val) { hookup_.amdoptions.FastRelaxedMath = Val; }
      void setOptSimplifyLibCall(bool Val) { hookup_.amdoptions.OptSimplifyLibCall = Val; }
      void setUnsafeMathOpt(bool Val) { hookup_.amdoptions.UnsafeMathOpt = Val; }
      void setFiniteMathOnly(bool Val) { hookup_.amdoptions.FiniteMathOnly = Val; }
      void setIsPreLinkOpt(bool Val) { hookup_.amdoptions.IsPreLinkOpt = Val; }
      void setFP32RoundDivideSqrt(bool Val) { hookup_.amdoptions.FP32RoundDivideSqrt = Val; }
      void setUseNative(const char * Val) { if(Val) hookup_.amdoptions.OptUseNative = Val; }
      void setDenormsAreZero(bool Val) { hookup_.amdoptions.DenormsAreZero = Val; }
      void setUniformWorkGroupSize(bool Val) { hookup_.amdoptions.UniformWorkGroupSize = Val; }
      void setHaveFastFMA32(bool Val) { hookup_.amdoptions.HaveFastFMA32 = Val; }

      /*! Returns the llvm binary */
      llvm::Module* LLVMBinary() const { return llvmbinary_; }
      aclModule* Module() const { return reinterpret_cast<aclModule*>(llvmbinary_);}
    protected:
      llvm::Module *llvmbinary_;
      llvm::LLVMContext         *context_;
      llvm::AMDLLVMContextHook  hookup_;
  }; // class CompilerStage
  /*@}*/
}
#endif // _BE_COMPILER_STAGE_HPP_
