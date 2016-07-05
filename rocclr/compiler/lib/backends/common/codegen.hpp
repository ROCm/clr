//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _BE_CODEGEN_HPP_
#define _BE_CODEGEN_HPP_
#include "compiler_stage.hpp"
#if defined(LEGACY_COMPLIB)
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#else
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#endif

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

//!--------------------------------------------------------------------------!//
// JIT Memory manager
//!--------------------------------------------------------------------------!//
class OCLMCJITMemoryManager : public
#if defined(LEGACY_COMPLIB)
  llvm::JITMemoryManager
#else
  llvm::RTDyldMemoryManager
#endif
{
public:
  typedef std::pair<llvm::sys::MemoryBlock, unsigned> Allocation;

private:
  llvm::SmallVector<Allocation, 16> AllocatedDataMem;
  llvm::SmallVector<Allocation, 16> AllocatedCodeMem;

  // FIXME: This is part of a work around to keep sections near one another
  // when MCJIT performs relocations after code emission but before
  // the generated code is moved to the remote target.
  llvm::sys::MemoryBlock Near;
  uint8_t * reservedAlloc(uintptr_t Size, unsigned Alignment);
  llvm::sys::MemoryBlock allocateSection(uintptr_t Size);

  uint8_t *allocPtr;
  uint8_t *allocMaxPtr;

public:
  OCLMCJITMemoryManager() : allocPtr(NULL), allocMaxPtr(NULL) {}
  virtual ~OCLMCJITMemoryManager();

  typedef llvm::SmallVectorImpl<Allocation>::const_iterator const_data_iterator;
  typedef llvm::SmallVectorImpl<Allocation>::const_iterator const_code_iterator;

  const_data_iterator data_begin() const { return AllocatedDataMem.begin(); }
  const_data_iterator   data_end() const { return AllocatedDataMem.end(); }
  const_code_iterator code_begin() const { return AllocatedCodeMem.begin(); }
  const_code_iterator   code_end() const { return AllocatedCodeMem.end(); }

  virtual void reserveMemory(uint64_t size);

#if defined(LEGACY_COMPLIB)
  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID);

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, bool isReadOnly);

  bool applyPermissions(std::string *ErrMsg) { return false; }

#else
  virtual bool needsToReserveAllocationSpace() override { return true; }

  virtual void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                                      uintptr_t RODataSize,
                                      uint32_t RODataAlign,
                                      uintptr_t RWDataSize,
                                      uint32_t RWDataAlign) override;

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, llvm::StringRef SectionName) override;

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, llvm::StringRef SectionName,
                               bool isReadOnly) override;

  bool finalizeMemory(std::string *ErrMsg = nullptr) override { return false; }
#endif

  void *getPointerToNamedFunction(const std::string &Name,
                                  bool AbortOnFailure = true);

  // The following obsolete JITMemoryManager calls are stubbed out for
  // this model.
  void setMemoryWritable();
  void setMemoryExecutable();
  void setPoisonMemory(bool poison);
  void AllocateGOT();
  uint8_t *getGOTBase() const;
  uint8_t *startFunctionBody(const llvm::Function *F, uintptr_t &ActualSize);
  uint8_t *allocateStub(const llvm::GlobalValue* F, unsigned StubSize,
                        unsigned Alignment);
  void endFunctionBody(const llvm::Function *F, uint8_t *FunctionStart,
                       uint8_t *FunctionEnd);
  uint8_t *allocateSpace(intptr_t Size, unsigned Alignment);
  uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment);
  void deallocateFunctionBody(void *Body);
  uint8_t* startExceptionTable(const llvm::Function* F, uintptr_t &ActualSize);
  void endExceptionTable(const llvm::Function *F, uint8_t *TableStart,
                         uint8_t *TableEnd, uint8_t* FrameRegister);
  void deallocateExceptionTable(void *ET);
  void deallocateSection(uint8_t* BasePtr);
};

// The jitCodeGen function creates a string where the '\0' characters
// have been encoded. decodeObjectImage puts the '\0' characters back.
void decodeObjectImage(std::string encodedObjectImage, std::string &decodedObjectImage);

#endif // _BE_CODEGEN_HPP_
