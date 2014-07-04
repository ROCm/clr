//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_OPTIMIZER_HPP_
#define _BE_OPTIMIZER_HPP_
#include "aclTypes.h"
#include "compiler_stage.hpp"
#include "llvm/Module.h"

namespace amdcl
{
  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::Optimizer
   *
   * @{
   */
  class Optimizer : public LLVMCompilerStage {
    Optimizer(Optimizer&); // DO NOT IMPLEMENT.
    Optimizer(); // DO NOT IMPLEMENT.
    public:
    Optimizer(aclCompiler *cl, aclBinary* elf, aclLogFunction log)
      : LLVMCompilerStage(cl, elf, log) {
        // Expose some options to LLVM
        llvm::AMDOptions *amdopts = &hookup_.amdoptions;
        if (opts_) {
            amdopts->OptLiveness = opts_->oVariables->OptLiveness;
            amdopts->NumAvailGPRs = opts_->NumAvailGPRs;
        }
      }

    virtual ~Optimizer() {}

    /*! Function that takes in the LLVM module as input
     * and optimizes it. 
     * Returns 0 on success and non-zero on failure.
     */
    virtual int optimize(llvm::Module *input) = 0;

  }; // class Optimizer
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::CLOptimizer
   *
   * @{
   */
  class CLOptimizer : public Optimizer {
   public:
      CLOptimizer(aclCompiler *cl, aclBinary *elf, aclLogFunction log)
      : Optimizer(cl, elf, log) {}
      virtual ~CLOptimizer() {}

    /*! Function that takes in the LLVM module as input
     * and optimizes it. 
     * Returns 0 on success and non-zero on failure.
     */
    virtual int optimize(llvm::Module *input) = 0;
  }; // class CLOptimizer
  /*@}*/
  
  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::GPUOptimizer
   *
   * @{
   */
  class GPUOptimizer : public CLOptimizer {
   public:
      GPUOptimizer(aclCompiler *cl, aclBinary *elf, aclLogFunction log)
      : CLOptimizer(cl, elf, log) {}
      virtual ~GPUOptimizer() {}

    /*! Function that takes in the LLVM module as input
     * and optimizes it. 
     * Returns 0 on success and non-zero on failure.
     */
    virtual int optimize(llvm::Module *input);
  }; // class GPUOptimizer
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::CPUOptimizer
   *
   * @{
   */
  class CPUOptimizer : public CLOptimizer {
    public:
      CPUOptimizer(aclCompiler *cl, aclBinary *elf, aclLogFunction log)
      : CLOptimizer(cl, elf, log) {}
      virtual ~CPUOptimizer() {}

    /*! Function that takes in the LLVM module as input
     * and optimizes it. 
     * Returns 0 on success and non-zero on failure.
     */
    virtual int optimize(llvm::Module *input);
    protected:
    int preOptimizer(llvm::Module *m);
  }; // class CPUOptimizer
  /*@}*/

}; // amdcl namespace
#endif // _BE_OPTIMIZER_HPP_
