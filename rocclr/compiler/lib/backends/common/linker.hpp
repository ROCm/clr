//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_LINKER_HPP_
#define _BE_LINKER_HPP_
#include "compiler_stage.hpp"
#include "aclTypes.h"
#include <string>
#include <map>


namespace llvm {
  class Module;
  class Value;
}; // namespace llvm

namespace amdcl
{
  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::Linker
   *
   * @{
   */
  class Linker : public LLVMCompilerStage{
    Linker(Linker&); // DO NOT IMPLEMENT.
    Linker(); // DO NOT IMPLEMENT.

    public:
    Linker(aclCompiler *cl, aclBinary* elf, aclLogFunction log)
      : LLVMCompilerStage(cl, elf, log) {}
    
    virtual ~Linker() {}

   

    /*! Function that takes as in a llvm::Module that contains LLVM-IR
     * binary and links in a vector of libraries.  
     * Returns 0 on success, non-zero on failure.
     */
    virtual int link(llvm::Module* input, std::vector<llvm::Module*> &libs) = 0;

  }; // class Linker
  /*@}*/

  /*! \addtogroup Compiler Library
   *
   * \copydoc amdcl::OCLLinker
   *
   * @{
   * \brief Linker that is unique to OpenCL.
   */
  class OCLLinker : public Linker {
    enum {
      MASK_NO_SIGNED_ZEROES          = 0x1,
      MASK_UNSAFE_MATH_OPTIMIZATIONS = 0x2,
      MASK_FINITE_MATH_ONLY          = 0x4,
      MASK_FAST_RELAXED_MATH         = 0x8,
      MASK_UNIFORM_WORK_GROUP_SIZE   = 0x10
    };

    public:
      OCLLinker(aclCompiler* cl, aclBinary* bin, aclLogFunction log)
        : Linker(cl, bin, log) {}

      virtual ~OCLLinker() {
        for (unsigned j = 0, i = (unsigned)mathLibs_.size(); j < i; ++j) {
          if (mathLibs_[j]) {
            delete mathLibs_[j];
          }
        }
      };
      void setPreLinkOpt(bool Val) { hookup_.amdoptions.IsPreLinkOpt = Val; }
      void setUnrollScratchThreshold(uint32_t ust) { hookup_.amdoptions.UnrollScratchThreshold = ust; }

      bool getWholeProgram() { return hookup_.amdoptions.WholeProgram; }
      uint32_t getUnrollScratchThreshold() { return hookup_.amdoptions.UnrollScratchThreshold; }


    /*! Function that takes as input a std::string which
     * contains LLVM-IR binary and links in a vector of libraries.
     * This version also links in the OpenCL math libraries along with
     * the list of libraries that are passed in.
     */
      int link(llvm::Module* input, std::vector<llvm::Module*> &libs);
    protected:
      void createOptionMaskFunction(llvm::Module* module);
      void createASICIDFunctions(llvm::Module* module);
      bool linkLLVMModules(std::vector<llvm::Module*> &libs);
      bool linkWithModule(llvm::Module* Dst, llvm::Module* Src,
    std::map<const llvm::Value*, bool> *ModuleRefMap);


    private:
      static void fixupOldTriple(llvm::Module* module);
      /*! Vector of modules that stores the math libraries. 
       */
      std::vector<llvm::Module*> mathLibs_;
  }; // class OCLLinker
  /*@}*/

}; // namespace amdcl
#endif // _BE_LINKER_HPP_
