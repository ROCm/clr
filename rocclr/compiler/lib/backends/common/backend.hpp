//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_BACKEND_HPP_
#define _BE_BACKEND_HPP_
#include "compiler_stage.hpp"

namespace amdcl
{
  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::Backend
   *
   * @{
   */
  class Backend : public CompilerStage {
    Backend(Backend&); // DO NOT IMPLEMENT.
    Backend(); // DO NOT IMPLEMENT.
    public:
    Backend(aclCompiler *cl, aclBinary *elf, aclLogFunction log)
      : CompilerStage(cl, elf, log) {}

    virtual ~Backend() {}

    /*! Function that takes in a string that is a source file
     * and generates the backend binary that is then
     * inserted into the elf file at the correct location.
     */
    virtual int jit(const std::string &source) = 0;
  }; // class Backend
  /*@}*/ 
}; // amdcl namespace
#endif // _BE_BACKEND_HPP
