//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_CODEGEN_HPP_
#define _BE_CODEGEN_HPP_
#include "compiler_stage.hpp"

namespace amdcl
{
  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::CodeGen
   *
   * @{
   */
  class CodeGen : public LLVMCompilerStage {
    CodeGen(CodeGen&); // DO NOT IMPLEMENT.
    CodeGen(); // DO NOT IMPLEMENT.
    public:
    CodeGen(aclCompiler *cl, aclBinary *elf, aclLogFunction log)
      : LLVMCompilerStage(cl, elf, log) {}

    virtual ~CodeGen() {}

    /*! Function that takes in an LLVM module as input
     * and generates code for it based on the target
     * device.
     * Returns 0 on success and non-zero on failure.
     */
    virtual int codegen(llvm::Module *input) = 0;

  }; // class CodeGen
  /*@}*/ 

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::CLCodeGen
   *
   * @{
   */
  class CLCodeGen : public CodeGen {
    CLCodeGen(CLCodeGen&); // DO NOT IMPLEMENT.
    CLCodeGen(); // DO NOT IMPLEMENT.
    public:
    CLCodeGen(aclCompiler *cl, aclBinary *elf, aclLogFunction log)
      : CodeGen(cl, elf, log) {}

    virtual ~CLCodeGen() {}

    /*! Function that takes in an LLVM module as input
     * and generates code for it based on the target
     * device.
     * Returns 0 on success and non-zero on failure.
     */
    virtual int codegen(llvm::Module *input);

  }; // class CLCodeGen
  /*@}*/  

#if 0
  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::AMDILCodeGen
   *
   * @{
   */
  class AMDILCodeGen : public CodeGen {
    AMDILCodeGen(AMDILCodeGen&); // DO NOT IMPLEMENT.
    AMDILCodeGen(); // DO NOT IMPLEMENT.
    public:
    AMDILCodeGen(aclCompiler *cl, aclBinary *elf, llvm::LLVMContext *ctx)
      : CLCodeGen(cl, elf, ctx) {}

    virtual ~AMDILCodeGen() {}

    /*! Function that takes in an LLVM module as input
     * and generates code for it based on the target
     * device.
     * Returns 0 on success and non-zero on failure.
     */
    int codegen(llvm::Module *input) = 0;

  }; // class AMDILCodeGen
  /*@}*/
#endif
} // amdcl namespace
#endif // _BE_CODEGEN_HPP_
