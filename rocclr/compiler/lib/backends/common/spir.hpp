//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_SPIR_HPP_
#define _BE_SPIR_HPP_
#include <string>
#include "aclTypes.h"
#include "compiler_stage.hpp"
namespace amdcl
{
  /*@}*/

  /*! \addtogroup CompilerLibrary
   *
   * \copydoc amdcl::SPIR
   *
   * @{
   * \brief Implementation of the Frontend interface to compile
   * from OpenCL C to LLVM-IR.
   */
  class SPIR : public LLVMCompilerStage {
      SPIR(SPIR&); // DO NOT IMPLEMENT.
      SPIR(); // DO NOT IMPLEMENT.

    public:
      SPIR(aclCompiler* cl, aclBinary* elf, aclLogFunction log)
        : LLVMCompilerStage(cl, elf, log) {}

      virtual ~SPIR() {}
      virtual llvm::Module* loadBitcode(std::string &spirBinary);
      virtual llvm::Module* loadSPIR(std::string &spirBinary);
      const void*
        toBinary(const void *text, size_t text_size, size_t *binary_size);
      const void*
        toText(const void *binary, size_t binary_size, size_t *text_size);

  }; // class SPIR
  /*@}*/
} // namespac amdcl
#endif // _BE_SPIR_HPP_
