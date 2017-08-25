//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_OPT_LEVEL_HPP_
#define _BE_OPT_LEVEL_HPP_
#include "top.hpp"
#include "utils/options.hpp"
#include "aclTypes.h"

#if defined DEBUG
#undef DEBUG
#endif

#include "llvm/Analysis/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class Module;

}; // llvm namespace

#if defined(LEGACY_COMPLIB)
#define LLVM_LEGACY_NAMESPACE llvm
#else
#define LLVM_LEGACY_NAMESPACE llvm::legacy
#endif

namespace amdcl
{
  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::OptLevel
   *
   * @{
   */
  class OptLevel {
    OptLevel(OptLevel&); // DO NOT IMPLEMENT.
    OptLevel(); // DO NOT IMPLEMENT.

    public:
    OptLevel(amd::option::Options *OptionsObj)
    : opts_(OptionsObj), TM(nullptr) {}

    virtual ~OptLevel() { delete TM; }

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU) = 0;
    protected:
    void setup(aclBinary *elf, bool isGPU, uint32_t OptLevel);
    void run(aclBinary *elf);
    LLVM_LEGACY_NAMESPACE::PassManager& Passes() { return passes_; }
    LLVM_LEGACY_NAMESPACE::FunctionPassManager& FPasses() { return (*fpasses_); }
    amd::option::Options* Options() { return opts_; }
    llvm::Module* module_;
    private:
    LLVM_LEGACY_NAMESPACE::FunctionPassManager *fpasses_;
    LLVM_LEGACY_NAMESPACE::PassManager passes_;
    amd::option::Options *opts_;
    llvm::TargetMachine* TM;
  }; // class OptLevel
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::O0OptLevel
   *
   * @{
   */
  class O0OptLevel : public OptLevel {
    O0OptLevel(O0OptLevel&); // DO NOT IMPLEMENT.
    O0OptLevel(); // DO NOT IMPLEMENT.

    public:
    O0OptLevel(amd::option::Options *opts)
    : OptLevel(opts) {}

    virtual ~O0OptLevel() {}

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class O0OptLevel
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::GPUO0OptLevel
   *
   * @{
   */
  class GPUO0OptLevel : public O0OptLevel {
    GPUO0OptLevel(GPUO0OptLevel&); // DO NOT IMPLEMENT.
    GPUO0OptLevel(); // DO NOT IMPLEMENT.

    public:
    GPUO0OptLevel(amd::option::Options *opts) 
    : O0OptLevel(opts) {}

    virtual ~GPUO0OptLevel() {}

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class O0OptLevel
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::O1OptLevel
   *
   * @{
   */
  class O1OptLevel : public OptLevel {
    O1OptLevel(O1OptLevel&); // DO NOT IMPLEMENT.
    O1OptLevel(); // DO NOT IMPLEMENT.

    public:
    O1OptLevel(amd::option::Options *opts) 
    : OptLevel(opts) {}

    virtual ~O1OptLevel() {}

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class O1OptLevel
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::O2OptLevel
   *
   * @{
   */
  class O2OptLevel : public OptLevel {
    O2OptLevel(O2OptLevel&); // DO NOT IMPLEMENT.
    O2OptLevel(); // DO NOT IMPLEMENT.

    public:
    O2OptLevel(amd::option::Options *opts) 
    : OptLevel(opts) {}

    virtual ~O2OptLevel() {}

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class O2OptLevel
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::O3OptLevel
   *
   * @{
   */
  class O3OptLevel : public OptLevel {
    O3OptLevel(O3OptLevel&); // DO NOT IMPLEMENT.
    O3OptLevel(); // DO NOT IMPLEMENT.

    public:
    O3OptLevel(amd::option::Options *opts) 
    : OptLevel(opts) {}

    virtual ~O3OptLevel() {}

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class O3OptLevel
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::O4OptLevel
   *
   * @{
   */
  class O4OptLevel : public OptLevel {
    O4OptLevel(O4OptLevel&); // DO NOT IMPLEMENT.
    O4OptLevel(); // DO NOT IMPLEMENT.

    public:
    O4OptLevel(amd::option::Options *opts) 
    : OptLevel(opts) {}

    virtual ~O4OptLevel() {}

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class O4OptLevel
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::OsOptLevel
   *
   * @{
   */
  class OsOptLevel : public OptLevel {
    OsOptLevel(OsOptLevel&); // DO NOT IMPLEMENT.
    OsOptLevel(); // DO NOT IMPLEMENT.

    public:
    OsOptLevel(amd::option::Options *opts) 
    : OptLevel(opts) {}

    virtual ~OsOptLevel() {}

    virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class OsOptLevel
  /*@}*/

  class OgOptLevel : public OptLevel {
      OgOptLevel(OgOptLevel&); // DO NOT IMPLEMENT.
      OgOptLevel(); // DO NOT IMPLEMENT.

  public:
      OgOptLevel(amd::option::Options *opts)
          : OptLevel(opts) {}

      virtual ~OgOptLevel() {}

      virtual int optimize(aclBinary *elf, llvm::Module *input, bool isGPU);
  }; // class OgOptLevel
     /*@}*/

}; // amdcl namespace
#endif // _BE_OPT_LEVEL_HPP_
