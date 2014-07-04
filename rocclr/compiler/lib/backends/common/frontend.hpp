//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_FRONTEND_HPP_
#define _BE_FRONTEND_HPP_
#include <string>
#include "aclTypes.h"
#include "compiler_stage.hpp"

namespace amdcl
{
  /*! \addtogroup CompilerLibrary
   *
   * \copydoc amdcl::Frontend
   *
   * @{
   * \brief Interface parent class for the frontend child classes.
   * This class should never be instantiated directly.
   */
  class Frontend : public LLVMCompilerStage {
      Frontend(Frontend&); // DO NOT IMPLEMENT.
      Frontend(); // DO NOT IMPLEMENT.
    public:
      Frontend(aclCompiler* cl, aclBinary* elf, aclLogFunction log)
        : LLVMCompilerStage(cl, elf, log) {}
      //! Virtual destructer that makes sure everything is cleaned up.
      virtual ~Frontend() {}

      //! Function that converts from OpenCL singleSrc into
      // OpenCL formatted LLVM-IR stored as a std::string.
      // This function generates a command string for clc to execute.
      virtual int compileCommand(const std::string& singleSrc) = 0;

  }; // class Frontend
  /*@}*/

  /*! \addtogroup CompilerLibrary
   *
   * \copydoc amdcl::OCLFrontend
   *
   * @{
   * \brief Implementation of the Frontend interface to compile
   * from OpenCL C to LLVM-IR.
   */
  class OCLFrontend : public Frontend {
      OCLFrontend(OCLFrontend&); // DO NOT IMPLEMENT.
      OCLFrontend(); // DO NOT IMPLEMENT.

      void appendCLVersionFlag(
          std::stringstream &ss,
          const amd::option::Options *opts);

      std::string getFrontendCommand(
          aclBinary *elf,
          const std::string &src,
          std::string &logFile,
          std::string &clFile,
          bool preprocessOnly);

    public:
      OCLFrontend(aclCompiler* cl, aclBinary* elf, aclLogFunction log)
        : Frontend(cl, elf, log) {}

      virtual ~OCLFrontend() {}

      //! Function that converts from OpenCL singleSrc into
      // OpenCL formatted LLVM-IR stored as a std::string.
      // This function generates a command string for clc to execute.
      virtual int compileCommand(const std::string& singleSrc);

  }; // class OCLFrontend
  /*@}*/


  /*! \addtogroup CompilerLibrary
   *
   * \copydoc amdcl::Frontend
   *
   * @{
   * \brief  This is the class  which calls the clang front-end.
   * This class will be used if user asks for it (By default EDG will be
   * called).
   */
  class ClangOCLFrontend : public Frontend {
      //! Options to be passed to the ClangOCLFE library.

  public:
      ClangOCLFrontend(aclCompiler* cl, aclBinary* elf, aclLogFunction log);

      //! Virtual destructer that makes sure everything is cleaned up.
      virtual ~ClangOCLFrontend() {}

      //! This function generates a command string for ClangOCLFE to execute.
      virtual int compileCommand(const std::string& singleSrc);

  }; // class Frontend
  /*@}*/
} // namespac amdcl
#endif // _BE_FRONTEND_HPP_
