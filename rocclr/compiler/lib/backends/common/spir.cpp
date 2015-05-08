//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "spir.hpp"
#include "aclTypes.h"
#include "bif/bifbase.hpp"
#include "utils/libUtils.h"
#include "utils/options.hpp"
#include "utils/target_mappings.h"
#include "os/os.hpp"

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#if defined(LEGACY_COMPLIB)
#include "llvm/DataLayout.h"
#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/PrintModulePass.h"
#else
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#endif
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/SPIRVerifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

#if defined(LEGACY_COMPLIB)
#define LLVMReturnStatusAction ReturnStatusAction
#endif

using namespace llvm;
using namespace amdcl;

Module*
amdcl::SPIR::loadSPIR(std::string &spirBinary)
{
  // Need to use the namespace here since a parent function is called Module().
  llvm::Module *bc = NULL;
  std::string errors;
  source_ = spirBinary;
  SPIRState State = {"", "", 1, 0, 1, 2};
  bc = amdcl::LLVMCompilerStage::loadBitcode(source_);
  if (!bc)
  {
    errors = "loadBitcode failed";
    log_ += errors;
    return NULL;
  }
#if defined(LEGACY_COMPLIB)
  verifyModule(*bc, ReturnStatusAction, &errors);
#else
  raw_string_ostream errorsOS(errors);
  verifyModule(*bc, &errorsOS);
#endif
  if (!errors.empty()) {
    log_ += errors;
    errors.clear();
  }
  FunctionPassManager FPM(bc);
  if (Options()->oVariables->verifyHWSpir) {
    if (!isHSAILTarget(Elf()->target)) {
      verifySPIRModule(*bc, LLVMReturnStatusAction, State, false, &errors);
    }
    if (!errors.empty()) {
      log_ += errors;
      errors.clear();
      delete bc;
      return NULL;
    }
  }
  if (Options()->oVariables->verifyLWSpir) {
    if (!isHSAILTarget(Elf()->target)) {
      verifySPIRModule(*bc, LLVMReturnStatusAction, State, true, &errors);
    }
    if (!errors.empty()) {
      log_ += errors;
      errors.clear();
      delete bc;
      return NULL;
    }
  }
  return bc;
}
Module*
amdcl::SPIR::loadBitcode(std::string &binary)
{
  llvm::Module *bc = loadSPIR(binary);
  if (!bc) return NULL;

  // FIXME: It is not clear why SPIRLoader is invoked so early here.
  // The current view is to keep SPIRLoader as a pure pre-link pass to
  // be called only by the linker.
  StringRef LayoutStr = is64BitTarget(Elf()->target) ?
    DATA_LAYOUT_64BIT : DATA_LAYOUT_32BIT;
  bc->setDataLayout(LayoutStr);
  bc->setTargetTriple(familySet[Elf()->target.arch_id].triple);

  llvm::PassManager SPIRPasses;
#if defined(LEGACY_COMPLIB)
  SPIRPasses.add(new llvm::DataLayout(bc));
#else
  SPIRPasses.add(new llvm::DataLayoutPass());
#endif
  SPIRPasses.add(createSPIRLoader(/*demangleBuiltin=*/ true));
  SPIRPasses.run(*bc);
  return bc;
}

const void*
SPIR::toBinary(const void *text, size_t text_size, size_t *binary_size)
{
  std::string text_buf(reinterpret_cast<const char*>(text), text_size);
  // Need to use the namespace here since a parent function is called Module().
  llvm::Module *mod = loadSPIR(text_buf);
  SmallString<256> char_buf;
  raw_svector_ostream outstream(char_buf);
  WriteBitcodeToFile(mod, outstream);
  std::string str_buf(char_buf.begin(), char_buf.end());
  (*binary_size) = char_buf.size();
  void *ptr = aclutAlloc(CL())(*binary_size);
  std::copy(char_buf.begin(), char_buf.end(), reinterpret_cast<char*>(ptr));
  return ptr;
}

const void*
SPIR::toText(const void *binary, size_t binary_size, size_t *text_size)
{
  std::string text_buf(reinterpret_cast<const char*>(binary), binary_size);
  // Need to use the namespace here since a parent function is called Module().
  llvm::Module *mod = loadSPIR(text_buf);
  std::string errors;
  if (!mod)
  {
    errors = "loadSPIR failed";
    log_ += errors;
    return NULL;
  }
  std::string bin_buf;
  raw_string_ostream buf(bin_buf);
  mod->print(buf, NULL);
  (*text_size) = bin_buf.size();
  void *ptr = aclutAlloc(CL())(*text_size);
  std::copy(bin_buf.begin(), bin_buf.end(), reinterpret_cast<char*>(ptr));
  return ptr;
}

