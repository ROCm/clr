//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifdef WITH_TARGET_HSAIL
#include "libHSAIL/HSAILBrigContainer.h"
#include "libHSAIL/HSAILDisassembler.h"
#include "libHSAIL/HSAILBrigObjectFile.h"

//prevent macro redefinition in drivers\hsa\compiler\lib\promotions\oclutils\top.hpp
//as it's already defined in drivers\hsa\compiler\llvm\include\llvm\Support\Format.h
#undef snprintf
#endif

#include "acl.h"
#include "aclTypes.h"
#include "compiler_stage.hpp"
#include "frontend.hpp"
#include "spir.hpp"
#include "codegen.hpp"
#include "library.hpp"
#include "linker.hpp"
#include "optimizer.hpp"
#include "amdil_be.hpp"
#include "hsail_be.hpp"
#include "x86_be.hpp"
#include "scCompileBase.h"
#include "bif/bifbase.hpp"
#include "os/os.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"
#include "utils/options.hpp"
#include "utils/target_mappings.h"
#include "utils/versions.hpp"

#include "llvm/LLVMContext.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Transforms/Scalar.h"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cassert>

aclLoaderData * ACL_API_ENTRY
if_aclCompilerInit(aclCompiler *cl, aclBinary *bin,
    aclLogFunction log, acl_error *error)
{
  llvm::llvm_acquire_global_lock();
  char* timing = ::getenv("AMD_DEBUG_HLC_ENABLE_TIMING");
  if (timing && (timing[0] == '1')) 
     llvm::TimePassesIsEnabled = true;
  else
     llvm::TimePassesIsEnabled = false;
  if (cl->llvm_shutdown == NULL) {
     cl->llvm_shutdown = reinterpret_cast<void*>
       (new llvm::llvm_shutdown_obj(false));
  }
  // Initialize targets first.
  llvm::InitializeAllTargets();

  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  // Initialize passes
  llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeCore(Registry);
  llvm::initializeTransformUtils(Registry);
  llvm::initializeScalarOpts(Registry);
  llvm::initializeInstCombine(Registry);
  llvm::initializeIPO(Registry);
  llvm::initializeInstrumentation(Registry);
  llvm::initializeAnalysis(Registry);
  llvm::initializeIPA(Registry);
  llvm::initializeCodeGen(Registry);
  llvm::initializeTarget(Registry);
  llvm::initializeVerifierPass(Registry);
  llvm::initializeDominatorTreePass(Registry);
  llvm::initializePreVerifierPass(Registry);
  llvm::llvm_release_global_lock();
  if (error) (*error) = ACL_SUCCESS;
  return reinterpret_cast<aclLoaderData*>(cl);
}
acl_error  ACL_API_ENTRY
if_aclCompilerFini(aclLoaderData *ald)
{
  if (ald == NULL) return ACL_INVALID_ARG;
  aclCompiler *cl = reinterpret_cast<aclCompiler *>(ald);
  return ACL_SUCCESS;
}


#define LOADER_FUNCS(NAME, TYPE) \
  aclLoaderData* ACL_API_ENTRY \
NAME##Init(aclCompiler *cl,\
    aclBinary *bin, \
    aclLogFunction callback,\
    acl_error *error)\
{\
  acl_error error_code = ACL_SUCCESS;\
  TYPE *acl = new TYPE(cl, bin, callback);\
  if (acl == NULL) {\
    error_code = ACL_OUT_OF_MEM;\
  }\
  if (error != NULL) (*error) = error_code;\
  return reinterpret_cast<aclLoaderData*>(acl);\
}\
acl_error ACL_API_ENTRY \
NAME##Fini(aclLoaderData *ald)\
{\
  acl_error error_code = ACL_SUCCESS;\
  TYPE *acl = reinterpret_cast<TYPE *>(ald);\
  if (acl == NULL) {\
    error_code = ACL_INVALID_ARG;\
  } else {\
    delete acl;\
  }\
  return error_code;\
}

#define LOADER_FUNCS_ERROR(NAME, TYPE) \
  aclLoaderData* ACL_API_ENTRY \
NAME##Init(aclCompiler *cl,\
    aclBinary *bin, \
    aclLogFunction callback,\
    acl_error *error)\
{\
  assert(!"Cannot go down this path without enabling support!"); \
  if (error) (*error) = ACL_SYS_ERROR; \
  return NULL; \
}\
acl_error ACL_API_ENTRY \
NAME##Fini(aclLoaderData *ald)\
{\
  assert(!"Cannot go down this path without enabling support!"); \
  return ACL_SYS_ERROR; \
}

#if defined(WITH_TARGET_AMDIL)
LOADER_FUNCS(AMDIL, amdcl::AMDIL);
LOADER_FUNCS(AMDILOpt, amdcl::GPUOptimizer);
#else
LOADER_FUNCS_ERROR(AMDIL, amdcl::AMDIL);
LOADER_FUNCS_ERROR(AMDILOpt, amdcl::GPUOptimizer);
#endif

#if defined(WITH_TARGET_HSAIL)
LOADER_FUNCS(HSAILAsm, amdcl::HSAIL);
LOADER_FUNCS(HSAILFE, amdcl::OCLFrontend);
LOADER_FUNCS(HSAILOpt, amdcl::GPUOptimizer);
#else
LOADER_FUNCS_ERROR(HSAILAsm, amdcl::HSAIL);
LOADER_FUNCS_ERROR(HSAILFE, amdcl::OCLFrontend);
LOADER_FUNCS_ERROR(HSAILOpt, amdcl::GPUOptimizer);
#endif

#if defined(WITH_TARGET_X86)
LOADER_FUNCS(X86Asm, amdcl::X86);
LOADER_FUNCS(X86Opt, amdcl::CPUOptimizer);
#else
LOADER_FUNCS_ERROR(X86Asm, amdcl::X86);
LOADER_FUNCS_ERROR(X86Opt, amdcl::CPUOptimizer);
#endif

LOADER_FUNCS(OCL, amdcl::OCLFrontend);
LOADER_FUNCS(OCLClang, amdcl::ClangOCLFrontend);
LOADER_FUNCS(Link, amdcl::OCLLinker);
LOADER_FUNCS(Codegen, amdcl::CLCodeGen);
LOADER_FUNCS(SPIR, amdcl::SPIR);
#undef LOADER_FUNCS


// CLC Frontend phase
aclModule* ACL_API_ENTRY
OCLFEToLLVMIR(
    aclLoaderData *ald,
    const char *source,
    size_t data_size,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_SUCCESS;
  amdcl::Frontend *aclFE = reinterpret_cast<amdcl::Frontend*>(ald);
  aclFE->setContext(ctx);
  int ret;
  std::string src_str(source, data_size);
  ret = aclFE->compileCommand(src_str);
  if (!aclFE->BuildLog().empty()) {
    appendLogToCL(aclFE->CL(), aclFE->BuildLog());
  }
  if (ret) {
    if (error != NULL) (*error) = ACL_FRONTEND_FAILURE;
    return NULL;
  }
  return aclFE->Module();
}

aclModule* ACL_API_ENTRY
OCLFEToSPIR(
    aclLoaderData *ald,
    const char *source,
    size_t data_size,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_SUCCESS;
  amdcl::Frontend *aclFE = reinterpret_cast<amdcl::Frontend*>(ald);
  aclFE->setContext(ctx);
  int ret;
  std::string src_str(source, data_size);
  ret = aclFE->compileCommand(src_str);
  if (!aclFE->BuildLog().empty()) {
    appendLogToCL(aclFE->CL(), aclFE->BuildLog());
  }
  if (ret) {
    if (error != NULL) (*error) = ACL_FRONTEND_FAILURE;
    return NULL;
  }
  return aclFE->Module();
}
aclModule* ACL_API_ENTRY
SPIRToModule(
    aclLoaderData *ald,
    const char *source,
    size_t data_size,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_SUCCESS;
  amdcl::SPIR *aclSPIR = reinterpret_cast<amdcl::SPIR*>(ald);
  aclSPIR->setContext(ctx);
  std::string dataStr(source, data_size);
  aclModule *module = reinterpret_cast<aclModule*>(aclSPIR->loadBitcode(dataStr));
  if (!aclSPIR->BuildLog().empty()) {
    appendLogToCL(aclSPIR->CL(), aclSPIR->BuildLog());
  }
  if (module == NULL) {
    if (error != NULL) (*error) = ACL_FRONTEND_FAILURE;
    return NULL;
  }
  return module;
}

aclModule * ACL_API_ENTRY
RSLLVMIRToModule(
    aclLoaderData *ald,
    const char *source,
    size_t data_size,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_SUCCESS;
  std::string llvmBinary(source, data_size);
  std::string ErrorMessage;
  llvm::LLVMContext * Context = reinterpret_cast<llvm::LLVMContext*>(ctx);
  llvm::MemoryBuffer *Buffer =
  llvm::MemoryBuffer::getMemBufferCopy(
    llvm::StringRef(llvmBinary), "input.bc");
  llvm::Module *M = NULL;

  if (llvm::isBitcode((const unsigned char *)Buffer->getBufferStart(),
                (const unsigned char *)Buffer->getBufferEnd())) {
    M = llvm::ParseBitcodeFile(Buffer, *Context, &ErrorMessage);
  }

  if (M == NULL) {
    if (error != NULL) (*error) = ACL_INVALID_BINARY;
    return NULL;
  }

  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(ald);
  aclDevType arch_id = cs->Elf()->target.arch_id;
  if ((arch_id != aclAMDIL) && (arch_id != aclHSAIL)) {
    assert("Unsupported architecture, expect amdil.");
    return NULL;
  }

  const char * NewTriple = familySet[aclAMDIL].triple;
  std::string OldTriple = M->getTargetTriple();

  if (OldTriple.compare("armv7-none-linux-gnueabi")) {
    assert("Input target is unknown, expect armv7-none-linux-gnueabi.");
    return NULL;
  }

  M->setTargetTriple(NewTriple);
  const char * LayoutStr = is64BitTarget(cs->Elf()->target) ?
    DATA_LAYOUT_64BIT : DATA_LAYOUT_32BIT;
  M->setDataLayout(LayoutStr);
  llvm::PassManager TransformPasses;
  TransformPasses.add(llvm::createOpenCLIRTransform());
  if (!TransformPasses.run(*M)) {
    if (error != NULL) (*error) = ACL_FRONTEND_FAILURE;
    return NULL;
  }
  
  aclModule *module = reinterpret_cast<aclModule*>(M);
  return module;
}

aclModule* ACL_API_ENTRY
OCLFEToModule(
    aclLoaderData *ald,
    const char *source,
    size_t data_size,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_SUCCESS;
  amdcl::Frontend *aclFE = reinterpret_cast<amdcl::Frontend*>(ald);
  aclFE->setContext(ctx);
  std::string dataStr(source, data_size);
  aclModule *module = reinterpret_cast<aclModule*>(aclFE->loadBitcode(dataStr));
  if (!aclFE->BuildLog().empty()) {
    appendLogToCL(aclFE->CL(), aclFE->BuildLog());
  }
  if (module == NULL) {
    if (error != NULL) (*error) = ACL_FRONTEND_FAILURE;
    return NULL;
  }
  return module;
}
acl_error ACL_API_ENTRY
AMDILFEToISA(
    aclLoaderData *ald,
    const char *source,
    size_t data_size)
{
#ifdef WITH_TARGET_AMDIL
  acl_error error_code = ACL_SUCCESS;
  amdcl::AMDIL *acl = reinterpret_cast<amdcl::AMDIL*>(ald);
  if (acl == NULL) {
    error_code = ACL_FRONTEND_FAILURE;
  }
  else {
    amd::option::Options* Opts = acl->Options();
    const char *kernel = Opts->getCurrKernelName();
    const char *name = (kernel == NULL) ? "main" : kernel;
    if (acl->compile(source, name)) {
      error_code = ACL_FRONTEND_FAILURE;
    }
  }
  if (!acl->BuildLog().empty()) {
    appendLogToCL(acl->CL(), acl->BuildLog());
  }
  if (!checkFlag(aclutGetCaps(acl->Elf()), capSaveAMDIL)) {
    acl->CL()->clAPI.remSec(acl->CL(), acl->Elf(), aclSOURCE);
  }
  return error_code;
#else
  assert(!"Cannot go down this path without AMDIL support!");
  return ACL_SYS_ERROR;
#endif
}

acl_error ACL_API_ENTRY
OCLFEToISA(
    aclLoaderData *ald,
    const char *source,
    size_t data_size)
{
  assert(!"Not implemented!");
  return ACL_UNSUPPORTED;
}

aclModule* ACL_API_ENTRY
OCLLinkToLLVMIR(
    aclLoaderData *data,
    aclModule *llvmBin,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_UNSUPPORTED;
  assert(!"Not implemented!");
  return NULL;
}
aclModule* ACL_API_ENTRY
OCLLinkToSPIR(
    aclLoaderData *data,
    aclModule *llvmBin,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_UNSUPPORTED;
  assert(!"Not implemented!");
  return NULL;
}

// LLVM Link phase
aclModule* ACL_API_ENTRY
OCLLinkPhase(
    aclLoaderData *data,
    aclModule *llvmBin,
    unsigned int numLibs,
    aclModule **libs,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_SUCCESS;
  amdcl::OCLLinker *aclLink = reinterpret_cast<amdcl::OCLLinker*>(data);
  if (aclLink == NULL || llvmBin == NULL || ctx == NULL) {
    if (error != NULL) (*error) = ACL_INVALID_ARG;
    return NULL;
  }
  const char* argv[] = { "",
    "-loop-unswitch-threshold=0",
    "-binomial-coefficient-limit-bitwidth=64"
  };

  aclLink->setContext(ctx);
  amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(aclLink->Elf()->options);
  int args = sizeof(argv) / sizeof(argv[0]);
  llvm::cl::ParseCommandLineOptions(args, (char**)argv, "OpenCL");

  if (Opts->getLLVMArgc())
    llvm::cl::ParseCommandLineOptions(Opts->getLLVMArgc(),
        Opts->getLLVMArgv(), "OpenCL");

  // LLVM Link phase
  std::vector<llvm::Module*> libvec;
  for (unsigned x = 0; x < numLibs; ++x) {
    if (libs[x] != NULL) {
      libvec.push_back(reinterpret_cast<llvm::Module*>(libs[x]));
    }
  }
  int ret = aclLink->link(reinterpret_cast<llvm::Module*>(llvmBin), libvec);
  if (!aclLink->BuildLog().empty()) {
    appendLogToCL(aclLink->CL(), aclLink->BuildLog());
  }
  if (ret) {
    if (error != NULL) (*error) = ACL_LINKER_ERROR;
    return NULL;
  }
  return aclLink->Module();
}

aclModule* ACL_API_ENTRY
GPUOptPhase(aclLoaderData *data,
    aclModule *llvmBin,
    aclContext *ctx,
    acl_error *error)
{
#if defined(WITH_TARGET_AMDIL) || defined(WITH_TARGET_HSAIL)
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(data);
  if (isGpuTarget(cs->Elf()->target)) {
    if (error != NULL) (*error) = ACL_SUCCESS;
    amdcl::GPUOptimizer *aclOpt = reinterpret_cast<amdcl::GPUOptimizer*>(data);
    if (aclOpt == NULL || llvmBin == NULL || ctx == NULL) {
      if (error != NULL) (*error) = ACL_INVALID_ARG;
      return NULL;
    }
    // LLVM Optimize phase
    aclOpt->setContext(ctx);
    amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(aclOpt->Elf()->options);
    if (Opts->getLLVMArgc())
      llvm::cl::ParseCommandLineOptions(Opts->getLLVMArgc(),
          Opts->getLLVMArgv(), "OpenCL");

    int ret = aclOpt->optimize(reinterpret_cast<llvm::Module*>(llvmBin));
    if (!aclOpt->BuildLog().empty()) {
      appendLogToCL(aclOpt->CL(), aclOpt->BuildLog());
    }
    if (ret) {
      if (error != NULL) (*error) = ACL_OPTIMIZER_ERROR;
      return NULL;
    }
    return aclOpt->Module();
  } else {
    assert(!"GPUOptPhase should be called only for AMDIL or HSAIL target.");
    if (error) (*error) = ACL_SYS_ERROR;
    return NULL;
  }
#else
  assert(!"Cannot go down this path without GPU support!");
  if (error) (*error) = ACL_SYS_ERROR;
  return NULL;
#endif
}

aclModule* ACL_API_ENTRY
X86OptPhase(aclLoaderData *data,
    aclModule *llvmBin,
    aclContext *ctx,
    acl_error *error)
{
#if defined(WITH_TARGET_X86)
  if (error != NULL) (*error) = ACL_SUCCESS;
  amdcl::CPUOptimizer *aclOpt = reinterpret_cast<amdcl::CPUOptimizer*>(data);
  if (aclOpt == NULL || llvmBin == NULL || ctx == NULL) {
    if (error != NULL) (*error) = ACL_INVALID_ARG;
    return NULL;
  }
  // LLVM Optimize phase
  aclOpt->setContext(ctx);
  amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(aclOpt->Elf()->options);
  if (Opts->getLLVMArgc())
    llvm::cl::ParseCommandLineOptions(Opts->getLLVMArgc(),
        Opts->getLLVMArgv(), "OpenCL");
  int ret = aclOpt->optimize(reinterpret_cast<llvm::Module*>(llvmBin));
  if (!aclOpt->BuildLog().empty()) {
    appendLogToCL(aclOpt->CL(), aclOpt->BuildLog());
  }
  if (ret) {
    if (error != NULL) (*error) = ACL_OPTIMIZER_ERROR;
    return NULL;
  }
  return aclOpt->Module();
#else
  assert(!"Cannot go down this path without X86 support!");
  if (error) (*error) = ACL_SYS_ERROR;
  return NULL;
#endif
}

const void* ACL_API_ENTRY
CodegenPhase(aclLoaderData *data,
    aclModule *llvmBin,
    aclContext *ctx,
    acl_error *error)
{
  if (error != NULL) (*error) = ACL_SUCCESS;
  amdcl::CLCodeGen *aclCG = reinterpret_cast<amdcl::CLCodeGen*>(data);
  if (aclCG == NULL || llvmBin == NULL || ctx == NULL) {
    if (error != NULL) (*error) = ACL_INVALID_ARG;
    return NULL;
  }
  aclCG->setContext(ctx);
  amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(aclCG->Elf()->options);
  if (Opts->getLLVMArgc())
    llvm::cl::ParseCommandLineOptions(Opts->getLLVMArgc(),
        Opts->getLLVMArgv(), "OpenCL");
  // LLVM Codegen phase
  int ret = aclCG->codegen(reinterpret_cast<llvm::Module*>(llvmBin));
  if (!aclCG->BuildLog().empty()) {
    appendLogToCL(aclCG->CL(), aclCG->BuildLog());
  }
  if (ret) {
    if (error != NULL) (*error) = ACL_CODEGEN_ERROR;
    return NULL;
  }
#ifdef WITH_TARGET_HSAIL
  if (isHSAILTarget(aclCG->Elf()->target)) {
    if (checkFlag(&aclCG->Elf()->caps, capSaveCG)) {
      const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symHSAILText);
      assert(symbol && "symbol not found");
      std::string name = symbol->str[PRE] + std::string("main") + 
                         symbol->str[POST];

      /*@todo r=Leonid, modify hsa\drivers\opencl\runtime\device\hsa\hsaprogram.cpp to parse BRIG section
       * for kernel names instead searching for "kernel &" in text HSAIL.
       */
      HSAIL_ASM::BrigContainer c;

      int result=HSAIL_ASM::BrigStreamer::load(c, aclCG->Source().data(), aclCG->Source().length());
      if (result!=0)
      {
        (*error) = ACL_CODEGEN_ERROR;
        return NULL;
      }

      HSAIL_ASM::Disassembler disasm(c);
      std::ostringstream hsail_stream;
      int ret = disasm.run(hsail_stream);
      if (ret) {
        (*error) = ACL_CODEGEN_ERROR;
        return NULL;
      }
      hsail_stream.flush();
      // TBD extra string copy
      const std::string& hsail = hsail_stream.str();
      amdcl::HSAIL *acl = reinterpret_cast<amdcl::HSAIL*>(data);
      // Dumping of HSAIL to file if needed
      if (acl)
        acl->dumpHSAIL(hsail, Opts, ".hsail");
      aclCG->CL()->clAPI.insSym(aclCG->CL(), aclCG->Elf(),
        hsail.data(), hsail.size(), aclCODEGEN, name.c_str());
    }
    return reinterpret_cast<const void*>(&(aclCG->Source()));
  } else
#endif
  {
    if (checkFlag(aclutGetCaps(aclCG->Elf()), capSaveCG)) {
      aclCG->CL()->clAPI.insSec(aclCG->CL(), aclCG->Elf(),
          aclCG->Source().data(),
          aclCG->Source().size(), aclCODEGEN);
    }
    return reinterpret_cast<const void*>(aclCG->Source().c_str());
  }
}

acl_error ACL_API_ENTRY
AMDILAsmPhase(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
#ifdef WITH_TARGET_AMDIL
  acl_error error_code = ACL_SUCCESS;
  if (source == NULL) {
    return ACL_INVALID_BINARY;
  }
  amdcl::AMDIL *acl = reinterpret_cast<amdcl::AMDIL*>(data);
  if (acl == NULL || acl->jit(source)) {
    error_code = ACL_CODEGEN_ERROR;
  }
  if (!acl->BuildLog().empty()) {
    appendLogToCL(acl->CL(), acl->BuildLog());
  }
  return error_code;
#else
  assert(!"Cannot go down this path without AMDIL support!");
  return ACL_CODEGEN_ERROR;
#endif
}
acl_error ACL_API_ENTRY
AMDILDisassemble(aclLoaderData *data,
    const char *kernel,
    const void *isa_code,
    size_t isa_size)
{
#ifdef WITH_TARGET_AMDIL
  std::string isaDump = "";
  std::string isaName = "";
  acl_error error_code = ACL_SUCCESS;
  if (isa_code == NULL || isa_size == 0 || kernel == NULL) {
    return ACL_INVALID_ARG;
  }
  amdcl::AMDIL *acl = reinterpret_cast<amdcl::AMDIL*>(data);
  if (acl == NULL) {
    error_code = ACL_INVALID_ARG;
  }
  isaDump = acl->disassemble(isa_code, isa_size);
  const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symISAText);
  assert(symbol && "symbol not found");
  isaName = symbol->str[PRE] + std::string(kernel) + symbol->str[POST];
  if (!isaDump.empty()) {
    error_code = acl->CL()->clAPI.insSym(acl->CL(), acl->Elf(),
        isaDump.data(), isaDump.size(),
        symbol->sections[0], isaName.c_str());
  }
  if (acl->Options()) {
    std::string kernelFileName = acl->Options()->getDumpFileName("_" + std::string(kernel) + ".isa");
    amdcl::dumpISA(kernelFileName, isaDump, acl->Options());
  }
  if (acl->Callback()) {
    acl->Callback()(isaDump.data(), isaDump.size());
  }
  return error_code;
#else
  assert(!"Cannot go down this path without AMDIL support!");
  return ACL_SYS_ERROR;
#endif
}

acl_error ACL_API_ENTRY
AMDILAssemble(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
#ifdef WITH_TARGET_AMDIL
  assert(!"Not implemented!");
  return ACL_UNSUPPORTED;
#else
  assert(!"Cannot go down this path without AMDIL support!");
  return ACL_SYS_ERROR;
#endif
}

acl_error ACL_API_ENTRY
HSAILAsmPhase(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
#ifdef WITH_TARGET_HSAIL
  acl_error error_code = ACL_SUCCESS;
  if (source == NULL) {
    return ACL_INVALID_BINARY;
  }
  amdcl::HSAIL *acl = reinterpret_cast<amdcl::HSAIL*>(data);
  if (acl == NULL) {
    error_code = ACL_CODEGEN_ERROR;
  }
  SC_EXPORT_FUNCTIONS* scef = reinterpret_cast<SC_EXPORT_FUNCTIONS*>(acl->CL()->scAPI.scef);
  if (scef[SC_HSAIL].SCCreate == NULL) {
    // Fail if table has not been initialized, probably because dynamic SC has not been loaded.
    // In this case, aclSCLoaderInit returns ACL_SUCCESS.
    return ACL_CODEGEN_ERROR;
  }
  if (acl->finalize()) {
    error_code = ACL_CODEGEN_ERROR;
  }
  if (!acl->BuildLog().empty()) {
    appendLogToCL(acl->CL(), acl->BuildLog());
  }
  return error_code;
#else
  assert(!"Cannot go down this path without HSAIL support!");
  return ACL_SYS_ERROR;
#endif
}

acl_error ACL_API_ENTRY
HSAILAssemble(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
#ifdef WITH_TARGET_HSAIL
  acl_error error_code = ACL_SUCCESS;
  amdcl::HSAIL *acl = reinterpret_cast<amdcl::HSAIL*>(data);
  if (acl == NULL || !acl->assemble(source)) {
    // TODO_HSA: Should this be tagged as an assembler error?
    //           needs ACL_ASSEMBLER_ERROR
    error_code = ACL_CODEGEN_ERROR;
    appendLogToCL(acl->CL(), "Error assembling HSAIL text.");
  }
  if (!acl->BuildLog().empty())
    appendLogToCL(acl->CL(), acl->BuildLog());
  return error_code;
#else
  assert(!"Cannot go down this path without HSAIL support!");
  return ACL_SYS_ERROR;
#endif
}

acl_error ACL_API_ENTRY
HSAILDisassemble(aclLoaderData *data,
    const char *kernel,
    const void *isa_code,
    size_t isa_size)
{
#ifdef WITH_TARGET_HSAIL
  std::string isaDump = "";
  std::string isaName = "";
  acl_error error_code = ACL_SUCCESS;
  if (isa_code == NULL || isa_size == 0 || kernel == NULL) {
    return ACL_INVALID_ARG;
  }
  amdcl::HSAIL *acl = reinterpret_cast<amdcl::HSAIL*>(data);
  if (acl == NULL) {
    return ACL_INVALID_ARG;
  }
  isaDump = acl->disassemble(isa_code, isa_size);
  const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symISAText);
  assert(symbol && "symbol not found");
  isaName = symbol->str[PRE] + std::string(kernel) + symbol->str[POST];
  if (!isaDump.empty()) {
    error_code = acl->CL()->clAPI.insSym(acl->CL(), acl->Elf(),
        isaDump.c_str(), isaDump.size(),
        aclINTERNAL, isaName.c_str());
  }
  if (acl->Options()) {
    std::string kernelFileName = acl->Options()->getDumpFileName("_" + std::string(kernel) + ".isa");
    acl->dumpISA(kernelFileName, isaDump, acl->Options());
  }
  if (acl->Callback()) {
    acl->Callback()(isaDump.c_str(), isaDump.size());
  }
  return error_code;
#else
  assert(!"Cannot go down this path without HSAIL support!");
  return ACL_SYS_ERROR;
#endif
}

acl_error ACL_API_ENTRY
X86AsmPhase(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
#ifdef WITH_TARGET_X86
  acl_error error_code = ACL_SUCCESS;
  if (source == NULL) {
    return ACL_INVALID_BINARY;
  }
  amdcl::X86 *acl = reinterpret_cast<amdcl::X86*>(data);
  if (acl == NULL || acl->jit(source)) {
    error_code = ACL_CODEGEN_ERROR;
  }
  if (!acl->BuildLog().empty()) {
    appendLogToCL(acl->CL(), acl->BuildLog());
  }
  return error_code;
#else
  assert(!"Cannot go down this path without X86 support!");
  return ACL_SYS_ERROR;
#endif
}

  acl_error ACL_API_ENTRY
X86Assemble(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
#ifdef WITH_TARGET_X86
  assert(!"Not implemented!");
  return ACL_UNSUPPORTED;
#else
  assert(!"Cannot go down this path without X86 support!");
  return ACL_SYS_ERROR;
#endif
}

acl_error ACL_API_ENTRY
X86Disassemble(aclLoaderData *data,
    const char *kernel,
    const void *isa_code,
    size_t isa_size)
{
#ifdef WITH_TARGET_X86
  assert(!"Not implemented!");
  return ACL_UNSUPPORTED;
#else
  assert(!"Cannot go down this path without X86 support!");
  return ACL_SYS_ERROR;
#endif
}

static void
saveOptionsToComments(aclCompiler *cl, aclBinary *curElf, const char *str, std::string &symbol)
{
  if (str != NULL && !checkFlag(aclutGetCaps(curElf), capEncrypted)
      && strlen(str)) {
    size_t test = 0;
    const void* ptr = cl->clAPI.extSym(cl, curElf, &test, aclCOMMENT, symbol.c_str(), NULL);
    if (ptr == NULL || (ptr != NULL && (test != strlen(str)
            || strcmp(reinterpret_cast<const char*>(ptr), str)))) {
      if (ptr != NULL) {
        cl->clAPI.remSym(cl, curElf, aclCOMMENT, symbol.c_str());
      }
      cl->clAPI.insSym(cl, curElf, str, strlen(str), aclCOMMENT, symbol.c_str());
    }
  }
}

aclLoaderData* ACL_API_ENTRY
OptInit(aclCompiler *cl,
    aclBinary *bin,
    aclLogFunction log,
    acl_error *err)
{
  if (!bin) return NULL;
  switch(bin->target.arch_id)
  {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86OptInit(cl, bin, log, err);
    case aclHSAIL64:
    case aclHSAIL: return HSAILOptInit(cl, bin, log, err);
    case aclAMDIL64:
    case aclAMDIL: return AMDILOptInit(cl, bin, log, err);
  }
  return NULL;
}

acl_error ACL_API_ENTRY
OptFini(aclLoaderData *ptr) {
  if (!ptr) return ACL_ERROR;
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(ptr);
  switch (cs->Elf()->target.arch_id) {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86OptFini(ptr);
    case aclHSAIL64:
    case aclHSAIL: return HSAILOptFini(ptr);
    case aclAMDIL64:
    case aclAMDIL: return AMDILOptFini(ptr);
  }
  return ACL_ERROR;
}

aclModule* ACL_API_ENTRY
OptOptimize(aclLoaderData *data,
    aclModule *llvmBin,
    aclContext *ctx,
    acl_error *error)
{
  if (!data) return NULL;
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(data);
  switch (cs->Elf()->target.arch_id) {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86OptPhase(data, llvmBin, ctx, error);
    case aclHSAIL64:
    case aclHSAIL: return GPUOptPhase(data, llvmBin, ctx, error);
    case aclAMDIL64:
    case aclAMDIL: return GPUOptPhase(data, llvmBin, ctx, error);
  }
  return NULL;
}

aclLoaderData* ACL_API_ENTRY
BEInit(aclCompiler *cl,
    aclBinary *bin,
    aclLogFunction log,
    acl_error *err)
{
  if (!bin) return NULL;
  switch(bin->target.arch_id)
  {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86AsmInit(cl, bin, log, err);
    case aclHSAIL64:
    case aclHSAIL: return HSAILAsmInit(cl, bin, log, err);
    case aclAMDIL64:
    case aclAMDIL: return AMDILInit(cl, bin, log, err);
  }
  return NULL;
}

acl_error ACL_API_ENTRY
BEFini(aclLoaderData *ptr)
{
  if (!ptr) return ACL_ERROR;
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(ptr);
  switch (cs->Elf()->target.arch_id) {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86AsmFini(ptr);
    case aclHSAIL64:
    case aclHSAIL: return HSAILAsmFini(ptr);
    case aclAMDIL64:
    case aclAMDIL: return AMDILFini(ptr);
  }
  return ACL_ERROR;
}

acl_error ACL_API_ENTRY
BEAsmPhase(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
  if (!data) return ACL_ERROR;
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(data);
  switch (cs->Elf()->target.arch_id) {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86AsmPhase(data, source, data_size);
    case aclHSAIL64:
    case aclHSAIL: return HSAILAsmPhase(data, source, data_size);
    case aclAMDIL64:
    case aclAMDIL: return AMDILAsmPhase(data, source, data_size);
  }
  return ACL_ERROR;

}


acl_error ACL_API_ENTRY
BEAssemble(aclLoaderData *data,
    const char *source,
    size_t data_size)
{
  if (!data) return ACL_ERROR;
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(data);
  switch (cs->Elf()->target.arch_id) {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86Assemble(data, source, data_size);
    case aclHSAIL64:
    case aclHSAIL: return HSAILAssemble(data, source, data_size);
    case aclAMDIL64:
    case aclAMDIL: return AMDILAssemble(data, source, data_size);
  }
  return ACL_ERROR;

}

acl_error ACL_API_ENTRY
BEDisassemble(aclLoaderData *data,
    const char *kernel,
    const void *isa_code,
    size_t data_size)
{
  if (!data) return ACL_ERROR;
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(data);
  switch (cs->Elf()->target.arch_id) {
    default:
      assert(!"Found an unhandled architecture!");
    case aclX64:
    case aclX86:   return X86Disassemble(data, kernel, isa_code, data_size);
    case aclHSAIL64:
    case aclHSAIL: return HSAILDisassemble(data, kernel, isa_code, data_size);
    case aclAMDIL64:
    case aclAMDIL: return AMDILDisassemble(data, kernel, isa_code, data_size);
  }
  return ACL_ERROR;

}

acl_error
finalizeBinary(aclCompiler *cl, aclBinary *bin)
{
  if (!bin || !bin->bin || !bin->options) return ACL_INVALID_ARG;
  if (cl) {
    size_t test = 0;
    const void* ptr = cl->clAPI.extSym(cl, bin, &test, aclCOMMENT, "acl_version_string", NULL);
    if (ptr == NULL || (ptr != NULL && (test != strlen(AMD_COMPILER_INFO)
        || strcmp(reinterpret_cast<const char*>(ptr), "acl_version_string")))) {
      if (ptr != NULL) {
        cl->clAPI.remSym(cl, bin, aclCOMMENT, "acl_version_string");
      }
      cl->clAPI.insSym(cl, bin,
        reinterpret_cast<const void*>(AMD_COMPILER_INFO),
        strlen(AMD_COMPILER_INFO), aclCOMMENT,
        "acl_version_string");
    }
#ifdef WITH_TARGET_HSAIL
    if (isHSAILTarget(bin->target)) {
      // Dumping of BIF to file if needed
      amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(bin->options);
      if (Opts && Opts->isDumpFlagSet(amd::option::DUMP_BIF)) {
        std::string fileName = Opts->getDumpFileName(".bif");
        if (aclWriteToFile(bin, fileName.c_str()) != ACL_SUCCESS)
          printf("Error - Failure in saving BIF file %s.\n", fileName.c_str());
      }
    }
#endif
  }
  return ACL_SUCCESS;
}

acl_error ACL_API_ENTRY
HSAILFEToISA(
    aclLoaderData *ald,
    const char *source,
    size_t data_size)
{
  acl_error error_code = HSAILAssemble(ald, source, data_size);
  if (error_code != ACL_SUCCESS)
    return error_code;
  return BEAsmPhase(ald, source, data_size);
}

static char * readFile(const char *source, size_t& size) {
  FILE *fp = ::fopen( source, "rb" );
  unsigned int length;
  size_t offset = 0;
  char *ptr;

  if (!fp) {
    return NULL;
  }

  // obtain file size.
  ::fseek (fp , 0 , SEEK_END);
  length = ::ftell (fp);
  ::rewind (fp);

  ptr = new char[offset + length + 1];

  if (length != fread(&ptr[offset], 1, length, fp))
  {
    delete [] ptr;
    return NULL;
  }

  ptr[offset + length] = '\0';
  size = offset + length;
  ::fclose(fp);

  return ptr;
}

static acl_error
aclCompileInternal(
    aclCompiler *cl,
    aclBinary *bin,
    const char *data,
    size_t data_size,
    aclLogFunction compile_callback,
    bool useFE,
    bool useLinker,
    bool useOpt,
    bool useCG,
    bool useISA)
{
  llvm::LLVMContext myCtx;
  aclContext *context = reinterpret_cast<aclContext*>(&myCtx);
  aclModule *module = NULL;
  std::string dataStr = std::string(data, data_size);
  acl_error error_code = ACL_SUCCESS;
  aclLoaderData *ald;


  // Load the frontend to convert from Source to LLVM-IR
  if (useFE) {
    ald = cl->feAPI.init(cl, bin, compile_callback, &error_code);
    if (!useLinker && !useCG && !useOpt && !useISA && cl->feAPI.toISA != NULL) {
      error_code = cl->feAPI.toISA(ald, data, data_size);
    } else {
      if (cl->feAPI.toIR == NULL) {
        error_code = ACL_SYS_ERROR;
        goto internal_compile_failure;
      }
      module = cl->feAPI.toIR(ald, data, data_size, context, &error_code);
    }
    cl->feAPI.fini(ald);
    if (error_code != ACL_SUCCESS) {
      goto internal_compile_failure;
    }
  } else if (useLinker || useOpt || useCG) {
    // Load a temp frontend object to convert from string LLVM-IR to LLVM Module.
    ald = cl->feAPI.init(cl, bin, compile_callback, &error_code);
    module = cl->feAPI.toModule(ald, data, data_size, context, &error_code);
    cl->feAPI.fini(ald);
    if (error_code != ACL_SUCCESS) {
      goto internal_compile_failure;
    }
  }

  // Use the linker to link in the libraries to the current module.
  if (useLinker) {
    ald = cl->linkAPI.init(cl, bin, compile_callback, &error_code);
    module = cl->linkAPI.link(ald, module, 0, NULL, context, &error_code);
    cl->linkAPI.fini(ald);
    if (error_code != ACL_SUCCESS) {
      goto internal_compile_failure;
    }
  }

  // Use the optimizer on the module at the given optimization level.
  if (useOpt) {
    ald = cl->optAPI.init(cl, bin, compile_callback, &error_code);
    module = cl->optAPI.optimize(ald, module, context, &error_code);
    cl->optAPI.fini(ald);
    if (error_code != ACL_SUCCESS) {
      goto internal_compile_failure;
    }
  }

  // Use the code generators to generate the ISA/IL string.
  if (useCG) {
    ald = cl->cgAPI.init(cl, bin, compile_callback, &error_code);
#ifdef WITH_TARGET_HSAIL
    amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(ald);
    if (isHSAILTarget(cs->Elf()->target)) {
      bool bHsailTextInput = false;
      const char *hsail_text_input = getenv("AMD_DEBUG_HSAIL_TEXT_INPUT");
      if (hsail_text_input != NULL && strcmp(hsail_text_input, "") != 0) {
        bHsailTextInput = true;
        if (bin && bin->options) {
          amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(bin->options);
          // Verify that the internal (blit) kernel is not being compiled 
          size_t ifind = Opts->origOptionStr.find("-cl-internal-kernel");
          if (ifind != std::string::npos)
            bHsailTextInput = false;
        }
      }
      if (!bHsailTextInput)
      {
        std::string* output = (std::string*) cl->cgAPI.codegen(ald, module, context, &error_code);

        if (error_code != ACL_SUCCESS) {
          goto internal_compile_failure;
        } 

        amdcl::HSAIL *acl = reinterpret_cast<amdcl::HSAIL*>(ald);
        if (acl == NULL || !acl->insertBRIG(*output)) {
          assert(!"Inserting BRIG failed\n");
        }
      }
      else
      {
        if (bHsailTextInput) {
          static std::string sHsailFileNames;
          if (sHsailFileNames.empty())
            sHsailFileNames = hsail_text_input;
          std::string sCurHsailFileName;
          size_t iFind = sHsailFileNames.find_first_not_of(";");
          if (iFind == std::string::npos) {
            sCurHsailFileName = sHsailFileNames;
            sHsailFileNames.clear();
          }
          else {
            size_t iFindEnd = sHsailFileNames.find_first_of(";", iFind+1);
            size_t iCount = sHsailFileNames.size();
            if (iFindEnd == std::string::npos) {
              sCurHsailFileName = sHsailFileNames.substr(iFind, iCount-iFind);
              sHsailFileNames.clear();
            }
            else {
              sCurHsailFileName = sHsailFileNames.substr(iFind, iFindEnd-iFind);
              sHsailFileNames = sHsailFileNames.substr(iFindEnd+1, iCount-iFindEnd-1);
            }
          }
          size_t size = 0;
          char * str = readFile(sCurHsailFileName.c_str(), size);
          dataStr = (str == NULL) ? "" : str;
          if (size == 0 || dataStr.length() == 0) {
            const char* error = "ERROR: AMD_DEBUG_HSAIL_TEXT_INPUT file does not exist.";
            appendLogToCL(cl, error);
            error_code = ACL_ERROR;
            goto internal_compile_failure;
          }
          // adding symHSAILText section to binary
          // TODO_HSA: remove it after changing RT's hsaprogram.cpp to parse BRIG section
          // for kernel names instead searching for "kernel &" in text HSAIL. 
          amdcl::CLCodeGen *aclCG = reinterpret_cast<amdcl::CLCodeGen*>(ald);
          if (checkFlag(&aclCG->Elf()->caps, capSaveCG)) {
            const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symHSAILText);
            assert(symbol && "symbol not found");
            std::string name = symbol->str[PRE] + std::string("main") + 
                               symbol->str[POST];
            aclCG->CL()->clAPI.insSym(aclCG->CL(), aclCG->Elf(),
                dataStr.data(), dataStr.size(), aclCODEGEN, name.c_str());
          }
        }
        else {
          const char *ptr = reinterpret_cast<const char*>(
              cl->cgAPI.codegen(ald, module, context, &error_code));
          if (error_code == ACL_SUCCESS) {
            dataStr = ptr;
          } 
          else {
            goto internal_compile_failure;
          }
        }
        // Use the assembler to generate the binary format of the IL string.
        if (HSAILAssemble(ald, dataStr.c_str(), dataStr.length()) != ACL_SUCCESS) {
          const char* error = "ASSEMBLER_FAILURE";
          appendLogToCL(cl, error);
          error_code = ACL_ERROR;
          goto internal_compile_failure;
        }
      }
      bifbase *elfBin = reinterpret_cast<bifbase*>(bin->bin);
      elfBin->setType(ET_EXEC);
    } else
#endif
    {
      const char *ptr = reinterpret_cast<const char*>(
          cl->cgAPI.codegen(ald, module, context, &error_code));
      if (error_code == ACL_SUCCESS) {
        dataStr = ptr;
      }
    }

    cl->cgAPI.fini(ald);
    if (error_code != ACL_SUCCESS) {
      goto internal_compile_failure;
    }
  }

  // Convert the input string into the device ISA binary.
  if (useISA) {
    ald = cl->beAPI.init(cl, bin, compile_callback, &error_code);
    error_code = cl->beAPI.finalize(ald, dataStr.data(), dataStr.length());
    cl->beAPI.fini(ald);
    if (error_code != ACL_SUCCESS) {
      goto internal_compile_failure;
    }
  }
internal_compile_failure:
  if (module) {
    delete reinterpret_cast<llvm::Module*>(module);
  }

  return error_code;
}
#define CONDITIONAL_ASSIGN(A, B) A = (A) ? (A) : (B)

#define CONDITIONAL_CMP_ASSIGN(A, B, C) A = (A && B != A) ? (A) : (C)

acl_error
IsValidCompilationOptions(aclBinary *bin, aclLogFunction compile_callback)
{
  acl_error error_code = ACL_SUCCESS;
#if defined(WITH_TARGET_HSAIL) && defined(WITH_TARGET_AMDIL)
  amd::option::Options* opts = reinterpret_cast<amd::option::Options*>(bin->options);
  std::string major = std::string(opts->oVariables->CLStd).substr(2,1);
  std::string error_msg;
  if (isHSAILTarget(bin->target) && major == "1") {
    error_msg = "Error: HSAIL doesn't support OpenCL version < 2.0.";
    error_code = ACL_INVALID_OPTION;
  }
  if (isAMDILTarget(bin->target) && major == "2") {
    error_msg = "Error: AMDIL doesn't support OpenCL version >= 2.0.";
    error_code = ACL_INVALID_OPTION;
  }
  if (ACL_SUCCESS != error_code && compile_callback) {
    compile_callback(error_msg.c_str(), error_msg.size());
  }
#endif
  return error_code;
}

acl_error  ACL_API_ENTRY
if_aclCompile(aclCompiler *cl,
    aclBinary *bin,
    const char *options,
    aclType from,
    aclType to,
    aclLogFunction compile_callback)
{
  if (bin == NULL || cl == NULL) {
    return ACL_INVALID_ARG;
  }
  acl_error error_code = IsValidCompilationOptions(bin, compile_callback);
  if (error_code != ACL_SUCCESS) {
    return error_code;
  }
#ifdef WITH_TARGET_HSAIL
  if (isHSAILTarget(bin->target)) {
#ifndef DEBUG
    // Do not install signal handlers for the pretty stack trace.
    llvm::DisablePrettyStackTrace = true;
#else
    llvm::sys::PrintStackTraceOnErrorSignal();
#endif
  } else 
#endif
  {
    llvm::InitializeAllAsmParsers();
    llvm::DisablePrettyStackTrace = true;
    llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeSPIRVerifierPass(Registry);
  }
  // Default 'to' is ISA.
  if (to == ACL_TYPE_DEFAULT) {
    to = ACL_TYPE_ISA;
  }
 if ((from == ACL_TYPE_X86_TEXT
    || from == ACL_TYPE_X86_BINARY)
      && (bin->target.arch_id != aclX86
        && bin->target.arch_id != aclX64)) {
    return ACL_INVALID_BINARY;
  }

  if ((from == ACL_TYPE_AMDIL_TEXT
    || from == ACL_TYPE_AMDIL_BINARY)
      && !isAMDILTarget(bin->target)) {
    return ACL_INVALID_BINARY;
  }

  if ((from == ACL_TYPE_HSAIL_TEXT
    || from == ACL_TYPE_HSAIL_BINARY)
      && bin->target.arch_id != aclHSAIL
      && bin->target.arch_id != aclHSAIL64) {
    return ACL_INVALID_BINARY;
  }
  if ((from == ACL_TYPE_HSAIL_TEXT && to == ACL_TYPE_HSAIL_BINARY)
      || (from == ACL_TYPE_HSAIL_BINARY && to == ACL_TYPE_HSAIL_TEXT)
      || (from == ACL_TYPE_AMDIL_TEXT && to == ACL_TYPE_AMDIL_BINARY)
      || (from == ACL_TYPE_AMDIL_BINARY && to == ACL_TYPE_AMDIL_TEXT)
      || (from == ACL_TYPE_SPIR_TEXT && to == ACL_TYPE_SPIR_BINARY)
      || (from == ACL_TYPE_SPIR_BINARY && to == ACL_TYPE_SPIR_TEXT)
      || (from == ACL_TYPE_LLVMIR_TEXT && to == ACL_TYPE_LLVMIR_BINARY)
      || (from == ACL_TYPE_LLVMIR_BINARY && to == ACL_TYPE_LLVMIR_TEXT)
      || (from == ACL_TYPE_X86_TEXT && to == ACL_TYPE_X86_BINARY)
      || (from == ACL_TYPE_X86_BINARY && to == ACL_TYPE_X86_TEXT)
      ) {
    amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(bin->options);
    const char *kernel = Opts->oVariables->Kernel;
    return aclConvertType(cl, bin, kernel, from);
  }

  if ((from == ACL_TYPE_AMDIL_TEXT
    || from == ACL_TYPE_AMDIL_BINARY
    || from == ACL_TYPE_HSAIL_TEXT
    || from == ACL_TYPE_HSAIL_BINARY
    || from == ACL_TYPE_X86_TEXT
    || from == ACL_TYPE_X86_BINARY)
    && to != ACL_TYPE_ISA) {
    return ACL_INVALID_ARG;
  }

  bool stages[5] = {false};
  uint8_t sectable[ACL_TYPE_LAST] =
    { 0, 0, 1, 1, 1, 1, 0, 6, 0, 6, 4, 4, 4, 0, 5, 0, 1 };
  aclSections d_section[7] =
  { aclSOURCE, aclLLVMIR, aclSPIR, aclSOURCE, aclCODEGEN, aclTEXT, aclINTERNAL };
  uint8_t start = sectable[from];
  uint8_t stop = sectable[to];
  const void* data = NULL;
  size_t data_size = 0;
  if (from == ACL_TYPE_DEFAULT) {
    aclSections sections[] = { aclSOURCE, aclSPIR, aclLLVMIR, aclCODEGEN, aclTEXT };
    uint8_t table[] = { 0, 1, 1, 4, 5 };
    aclType type[] = { ACL_TYPE_SOURCE, ACL_TYPE_SPIR_BINARY, ACL_TYPE_LLVMIR_BINARY,
      ACL_TYPE_CG, ACL_TYPE_ISA };
    for (int y = 0, x = sizeof(sections) / sizeof(sections[0]) - 1;
        x >= y; --x) {
      data = (const char*)cl->clAPI.extSec(cl, bin, &data_size,
          sections[x], &error_code);
      if (data != NULL && data_size > 0 && error_code == ACL_SUCCESS) {
        start = table[x];
        from = type[x];
        break;
      }
    }
  } else {
    if (from == ACL_TYPE_SPIR_BINARY ||
        from == ACL_TYPE_SPIR_TEXT) {
      data = cl->clAPI.extSec(cl, bin, &data_size, aclSPIR, &error_code);
    }
    else if (from == ACL_TYPE_RSLLVMIR_BINARY) {
      data = cl->clAPI.extSec(cl, bin, &data_size, aclLLVMIR, &error_code);
    }
    else {
      data = cl->clAPI.extSec(cl, bin, &data_size, d_section[start], &error_code);
    }
  }
  if (error_code != ACL_SUCCESS) {
    return error_code;
  }
  // Based on our compiler options, we need to change the functors to use
  // the correct pointers unless they are custom loaded, then we should
  // not modify them. This code is ugly and needs to be designed better.
  if (start == 0) {
    if (from == ACL_TYPE_OPENCL
        || from == ACL_TYPE_SOURCE
        || from == ACL_TYPE_DEFAULT) {
      const oclBIFSymbolStruct* symbol
        = findBIF30SymStruct(symOpenclCompilerOptions);
      assert(symbol && "symbol not found");
      std::string optSec = std::string(symbol->str[PRE]) + std::string(symbol->str[POST]);
      assert(symbol->sections[0] == aclCOMMENT
             && symbol->sections[0] == symbol->sections[1]
             && "not in comment section");
      saveOptionsToComments(cl, bin, options, optSec);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &SPIRInit, &OCLInit);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &AMDILInit, &OCLInit);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &HSAILFEInit, &OCLInit);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &SPIRFini, &OCLFini);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &AMDILFini, &OCLFini);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &HSAILFEFini, &OCLFini);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.toISA, &AMDILFEToISA, NULL);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.toISA, &HSAILFEToISA, NULL);
      if (to == ACL_TYPE_LLVMIR_BINARY
          || to == ACL_TYPE_LLVMIR_TEXT) {
        cl->feAPI.toISA = NULL;
        cl->feAPI.toIR = &OCLFEToLLVMIR;
      } else if(to == ACL_TYPE_SPIR_BINARY
          || to == ACL_TYPE_SPIR_TEXT) {
        cl->feAPI.toISA = NULL;
        cl->feAPI.toIR = &OCLFEToSPIR;
      }
    } else if (from == ACL_TYPE_AMDIL_TEXT) {
      amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(bin->options);
      const char *kernel = Opts->oVariables->Kernel;
      const oclBIFSymbolStruct* symbol
        = findBIF30SymStruct(symAMDILCompilerOptions);
      assert(symbol && "symbol not found");
      std::string optSec = std::string(symbol->str[PRE])
        + std::string((kernel == NULL) ? "main" : kernel)
        + std::string(symbol->str[POST]);
      assert(symbol->sections[0] == aclCOMMENT && "not in comment section");
      saveOptionsToComments(cl, bin, options, optSec);
      if (to == ACL_TYPE_ISA || to == ACL_TYPE_DEFAULT) {
        stop = 1;
        cl->feAPI.init = &AMDILInit;
        cl->feAPI.fini = &AMDILFini;
        cl->feAPI.toISA = &AMDILFEToISA;
        cl->feAPI.toIR = NULL;
        cl->feAPI.toModule = NULL;
      } else {
        return ACL_UNSUPPORTED;
      }
    } else if (from == ACL_TYPE_HSAIL_TEXT) {
      amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(bin->options);
      const char *kernel = Opts->oVariables->Kernel;
      const oclBIFSymbolStruct* symbol
        = findBIF30SymStruct(symHSACompilerOptions);
      assert(symbol && "symbol not found");
      std::string optSec = std::string(symbol->str[PRE])
        + std::string((kernel == NULL) ? "main" : kernel)
        + std::string(symbol->str[POST]);
      assert(symbol->sections[0] == aclCOMMENT && "not in comment section");
      saveOptionsToComments(cl, bin, options, optSec);
      if (to == ACL_TYPE_ISA || to == ACL_TYPE_DEFAULT) {
        stop = 1;
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &OCLInit, &HSAILFEInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &OCLFini, &HSAILFEFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.toISA, &OCLFEToISA, &HSAILFEToISA);
        cl->feAPI.toIR = NULL;
        cl->feAPI.toModule = NULL;
      } else {
        return ACL_UNSUPPORTED;
      }
    }
  } else if (start == 1) {
      if ((from == ACL_TYPE_SPIR_BINARY || from == ACL_TYPE_SPIR_TEXT) &&
          (to == ACL_TYPE_LLVMIR_BINARY || to == ACL_TYPE_LLVMIR_TEXT)) {
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &OCLInit, &SPIRInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &AMDILInit, &SPIRInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &HSAILFEInit, &SPIRInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &OCLFini, &SPIRFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &AMDILFini, &SPIRFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &HSAILFEFini, &SPIRFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.toModule, &OCLFEToModule, &SPIRToModule);
      } else if (from == ACL_TYPE_LLVMIR_BINARY || from == ACL_TYPE_LLVMIR_TEXT ||
          from == ACL_TYPE_SPIR_BINARY || from == ACL_TYPE_SPIR_TEXT ||
          from == ACL_TYPE_RSLLVMIR_BINARY) {
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &SPIRInit, &OCLInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &AMDILInit, &OCLInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &HSAILFEInit, &OCLInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &SPIRFini, &OCLFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &AMDILFini, &OCLFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &HSAILFEFini, &OCLFini);

	if (from == ACL_TYPE_RSLLVMIR_BINARY) {
	  cl->feAPI.toModule = &RSLLVMIRToModule;
	}
      }
  }

  if (start > stop) {
    return ACL_INVALID_ARG;
  }
  if (start == stop) {
    return ACL_SUCCESS;
  }
  for (uint8_t x = start; x < stop; ++x) {
    stages[x] = true;
  }
  error_code = aclCompileInternal(cl, bin,
      reinterpret_cast<const char*>(data),
      data_size, compile_callback,
      stages[0], stages[1], stages[2], stages[3], stages[4]);
  if (error_code == ACL_SUCCESS) {
    return finalizeBinary(cl, bin);
  }
  return error_code;
}
#undef CONDITIONAL_ASSIGN
#undef CONDITIONAL_CMP_ASSIGN

acl_error  ACL_API_ENTRY
if_aclLink(aclCompiler *cl,
    aclBinary *src_bin,
    unsigned int num_libs,
    aclBinary **libs,
    aclType link_mode,
    const char *options,
    aclLogFunction link_callback)
{
  aclLoaderData *ald;
  size_t data_size = 0;
  aclModule *module = NULL, *dst_module = NULL;
  llvm::LLVMContext myCtx;
  aclContext *context = reinterpret_cast<aclContext*>(&myCtx);

  acl_error error_code = ACL_SUCCESS;
  aclModule **mod_libs = NULL;
  if (num_libs > 0) {
    mod_libs = new aclModule*[num_libs];
    memset(mod_libs, 0, num_libs * sizeof(*mod_libs));
  }

  switch(link_mode) {
    default: error_code = ACL_UNSUPPORTED; break;
    case ACL_TYPE_LLVMIR_BINARY:
    case ACL_TYPE_RSLLVMIR_BINARY:
     {
       ald = cl->feAPI.init(cl, src_bin, link_callback, &error_code);
       const void *ptr = cl->clAPI.extSec(cl, src_bin, &data_size, aclLLVMIR, &error_code);
       if (ptr == NULL)
         ptr = cl->clAPI.extSec(cl, src_bin, &data_size, aclSPIR, &error_code);
       char *mod = new char[data_size];
       memcpy(mod, ptr, data_size);
       module = cl->feAPI.toModule(ald, mod, data_size, context, &error_code);
       for (unsigned x = 0; x < num_libs; ++x) {
         const void *ptr = cl->clAPI.extSec(cl, libs[x], &data_size, aclLLVMIR, NULL);
         if (ptr == NULL)
           ptr = cl->clAPI.extSec(cl, libs[x], &data_size, aclSPIR, NULL);
         if (ptr == NULL) continue;
         mod = new char[data_size];
         memcpy(mod, ptr, data_size);
         mod_libs[x] = cl->feAPI.toModule(ald, mod, data_size, context, &error_code);
       }
       cl->feAPI.fini(ald);
     }
     break;
  }
  if (error_code != ACL_SUCCESS) {
    goto internal_link_failure;
  }
  ald = cl->linkAPI.init(cl, src_bin, link_callback, &error_code);
  dst_module = cl->linkAPI.link(ald, module, num_libs, mod_libs,
      context, &error_code);
  cl->linkAPI.fini(ald);
  if (error_code == ACL_SUCCESS) {
    switch (link_mode) {
      default: error_code = ACL_UNSUPPORTED; break;
      case ACL_TYPE_LLVMIR_BINARY:
      case ACL_TYPE_RSLLVMIR_BINARY:
      {
#if 1 || LLVM_TRUNK_INTEGRATION_CL >= 7710
        llvm::SmallVector<char, 4096> array;
        llvm::raw_svector_ostream outstream(array);
        llvm::WriteBitcodeToFile(reinterpret_cast<llvm::Module*>(dst_module), outstream);
        cl->clAPI.remSec(cl, src_bin, aclLLVMIR);
        outstream.flush();
        error_code = cl->clAPI.insSec(cl, src_bin,
            &array[0], array.size(), aclLLVMIR);
#else
        std::vector<unsigned char> array;
        array.reserve(4096);
        llvm::BitstreamWriter stream(array);
        llvm::WriteBitcodeToStream(reinterpret_cast<llvm::Module*>(dst_module),
          stream);
        cl->clAPI.remSec(cl, src_bin, aclLLVMIR);
        error_code = cl->clAPI.insSec(cl, src_bin,
            &array[0], array.size(), aclLLVMIR);
#endif
        if (dst_module != NULL && dst_module != module) {
          delete reinterpret_cast<llvm::Module*>(dst_module);
        }
      }
      bifbase *elfBin = reinterpret_cast<bifbase*>(src_bin->bin);
      elfBin->setType(ET_DYN);
      break;
    }
    return finalizeBinary(cl, src_bin);
  }
internal_link_failure:
  const char *error = aclGetErrorString(error_code);
  appendLogToCL(cl, error);
  if (link_callback) {
    link_callback(cl->buildLog, cl->logSize);
  }
  if (!error && module) {
    delete reinterpret_cast<llvm::Module*>(module);
  }
  if (mod_libs) {
    for (unsigned x = 0; x < num_libs; ++x) {
      if (!error && mod_libs[x]) {
        delete reinterpret_cast<llvm::Module*>(mod_libs[x]);
      }
    }
    delete [] mod_libs;
  }
  return error_code;
}

const char*  ACL_API_ENTRY
if_aclGetCompilerLog(aclCompiler *cl)
{
  return (cl->buildLog == 0) ? "" : cl->buildLog;
}

static std::string getSymbolName(aclType type, const char *name, aclSections &id)
{
  const oclBIFSymbolStruct* symbol = NULL;
  uint8_t targetType = 0;
  std::string tmpname(name);
  std::string prefix = "";
  std::string postfix = "";
  switch (type) {
    default:
      assert(!"Invalid type detected!");
      return tmpname;
    case ACL_TYPE_AMDIL_TEXT:
      symbol = findBIF30SymStruct(symAMDILText);
      assert(symbol && "symbol not found");
      break;
    case ACL_TYPE_HSAIL_TEXT:
      symbol = findBIF30SymStruct(symHSAILText);
      assert(symbol && "symbol not found");
      break;
    case ACL_TYPE_LLVMIR_TEXT:
      id = aclLLVMIR;
      break;
    case ACL_TYPE_SPIR_TEXT:
      id = aclSPIR;
      break;
    case ACL_TYPE_X86_TEXT:
      id = aclCODEGEN;
      break;
    case ACL_TYPE_AMDIL_BINARY:
      symbol = findBIF30SymStruct(symAMDILBinary);
      assert(symbol && "symbol not found");
      break;
    case ACL_TYPE_HSAIL_BINARY:
      symbol = findBIF30SymStruct(symHSABinary);
      assert(symbol && "symbol not found");
      break;
    case ACL_TYPE_LLVMIR_BINARY:
      id = aclLLVMIR;
      break;
    case ACL_TYPE_RSLLVMIR_BINARY:
      id = aclLLVMIR;
      break;
    case ACL_TYPE_SPIR_BINARY:
      id = aclSPIR;
      break;
    case ACL_TYPE_X86_BINARY:
      id = aclCODEGEN;
      break;
  };
  if (symbol) {
    prefix = symbol->str[PRE];
    postfix = symbol->str[POST];
    id = symbol->sections[0];
  }
  return prefix + tmpname + postfix;
}

const void*  ACL_API_ENTRY
if_aclRetrieveType(aclCompiler *cl,
    const aclBinary *bin,
    const char *name,
    size_t *data_size,
    aclType type,
    acl_error *error_code)
{
  aclSections sec_id;
  std::string symbol_name = getSymbolName(type, name, sec_id);
  return cl->clAPI.extSym(cl, bin, data_size, sec_id, symbol_name.c_str(), error_code);
}

acl_error  ACL_API_ENTRY
if_aclSetType(aclCompiler *cl,
    aclBinary *bin,
    const char *name,
    aclType type,
    const void *data,
    size_t size)
{
  aclSections sec_id;
  std::string symbol_name = getSymbolName(type, name, sec_id);
  return cl->clAPI.insSym(cl, bin, data, size, sec_id, symbol_name.c_str());
}

acl_error  ACL_API_ENTRY
if_aclConvertType(aclCompiler *cl,
    aclBinary *bin,
    const char *name,
    aclType type)
{
  acl_error error_code = ACL_SUCCESS;
  aclType to;
  aclSections sec = aclSOURCE;
  bool need_name = true;
  size_t from_data_size = 0;
  const void *from_data = NULL;
  switch (type) {
    default:
      return ACL_UNSUPPORTED;
    case ACL_TYPE_LLVMIR_TEXT:
      to = ACL_TYPE_LLVMIR_BINARY;
      need_name = false;
      sec = aclLLVMIR;
      break;
    case ACL_TYPE_LLVMIR_BINARY:
      to = ACL_TYPE_LLVMIR_TEXT;
      need_name = false;
      sec = aclLLVMIR;
      break;
    case ACL_TYPE_SPIR_TEXT:
      to = ACL_TYPE_SPIR_BINARY;
      need_name = false;
      sec = aclSPIR;
      break;
    case ACL_TYPE_SPIR_BINARY:
      to = ACL_TYPE_SPIR_TEXT;
      need_name = false;
      sec = aclSPIR;
      break;
    case ACL_TYPE_AMDIL_TEXT:
    {
      to = ACL_TYPE_AMDIL_BINARY;
      // extract from symbol __debugil_text in .internal section
      const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symDebugilText);
      assert(symbol && "symbol not found");
      std::string debugilSym
        = std::string(symbol->str[PRE] + std::string(symbol->str[POST]));
      from_data = cl->clAPI.extSym(cl, bin, &from_data_size,
                                   symbol->sections[0],
                                   debugilSym.c_str(), &error_code);
      break;
    }
    case ACL_TYPE_AMDIL_BINARY:
    {
      to = ACL_TYPE_AMDIL_TEXT;
      // extract from symbol __debugil_binary in .internal section
      const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symDebugilBinary);
      assert(symbol && "symbol not found");
      std::string debugilSym
        = std::string(symbol->str[PRE] + std::string(symbol->str[POST]));
      from_data = cl->clAPI.extSym(cl, bin, &from_data_size,
                                   symbol->sections[0],
                                   debugilSym.c_str(), &error_code);
      break;
    }
    case ACL_TYPE_HSAIL_TEXT:
      to = ACL_TYPE_HSAIL_BINARY;
      break;
    case ACL_TYPE_HSAIL_BINARY:
      to = ACL_TYPE_HSAIL_TEXT;
      break;
    case ACL_TYPE_X86_TEXT:
      to = ACL_TYPE_X86_BINARY;
      break;
    case ACL_TYPE_X86_BINARY:
      to = ACL_TYPE_X86_TEXT;
      break;
  }
  if (from_data == NULL) {
    if (name == NULL || !need_name) {
      if (need_name) {
        return ACL_INVALID_ARG;
      }
      from_data = cl->clAPI.extSec(cl, bin,
          &from_data_size, sec, &error_code);
    } else {
      from_data = cl->clAPI.retrieveType(cl, bin, name,
        &from_data_size, type, &error_code);
    }
  }
  if (error_code != ACL_SUCCESS) {
    return error_code;
  }
  const void *to_data = from_data;
  size_t to_data_size = from_data_size;
  switch (to) {
    default:
      return ACL_UNSUPPORTED;
    case ACL_TYPE_SPIR_TEXT:
      {
        amdcl::SPIR *spir = new amdcl::SPIR(cl, bin, NULL);
        llvm::LLVMContext myCtx;
        aclContext *context = reinterpret_cast<aclContext*>(&myCtx);
        spir->setContext(context);
        if (spir == NULL) {
          return ACL_OUT_OF_MEM;
        }
        to_data = spir->toText(from_data, from_data_size, &to_data_size);
        if (!spir->BuildLog().empty()) {
          appendLogToCL(cl, spir->BuildLog());
        }
        if (to_data == NULL) {
          return ACL_INVALID_SPIR;
        }
        delete spir;
      }
      break;
    case ACL_TYPE_SPIR_BINARY:
      {
        amdcl::SPIR *spir = new amdcl::SPIR(cl, bin, NULL);
        llvm::LLVMContext myCtx;
        aclContext *context = reinterpret_cast<aclContext*>(&myCtx);
        spir->setContext(context);
        if (spir == NULL) {
          return ACL_OUT_OF_MEM;
        }
        to_data = spir->toBinary(from_data, from_data_size, &to_data_size);
        if (!spir->BuildLog().empty()) {
          appendLogToCL(cl, spir->BuildLog());
        }
        if (to_data == NULL) {
          return ACL_INVALID_SPIR;
        }
        delete spir;
      }
      break;
    case ACL_TYPE_AMDIL_TEXT:
      {
#if defined(WITH_TARGET_AMDIL)
        if (isAMDILTarget(bin->target)) {
          amdcl::AMDIL *acl = new amdcl::AMDIL(cl, bin, NULL);
          if (acl == NULL)  {
            return ACL_OUT_OF_MEM;
          }
          to_data = acl->toText(from_data, from_data_size);
          to_data_size = strlen(reinterpret_cast<const char*>(to_data));
          delete acl;
          // insert into .internal section under symbol __debugil_text
          const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symDebugilText);
          assert(symbol && "symbol not found");
          std::string debugilSym
            = std::string(symbol->str[PRE] + std::string(symbol->str[POST]));
          return cl->clAPI.insSym(cl, bin, to_data, to_data_size,
                                  symbol->sections[0], debugilSym.c_str());
        } else {
          assert(!"Unsupported architecture, expect amdil.");
          return ACL_SYS_ERROR;
        }
#else
        assert(!"Cannot go down this path without AMDIL support!");
        return ACL_SYS_ERROR;
#endif
      }
      break;
    case ACL_TYPE_AMDIL_BINARY:
      {
#if defined(WITH_TARGET_AMDIL)
        if (isAMDILTarget(bin->target)) {
          amdcl::AMDIL *acl = new amdcl::AMDIL(cl, bin, NULL);
          if (acl == NULL)  {
            return ACL_OUT_OF_MEM;
          }
          to_data = acl->toBinary(reinterpret_cast<const char*>(from_data),
              &to_data_size);
          delete acl;
          // insert into .internal section under symbol __debugil_binary
          const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symDebugilBinary);
          assert(symbol && "symbol not found");
          std::string debugilSym
            = std::string(symbol->str[PRE] + std::string(symbol->str[POST]));
          return cl->clAPI.insSym(cl, bin, to_data, to_data_size,
                                  symbol->sections[0], debugilSym.c_str());
        } else {
          assert(!"Unsupported architecture, expect amdil.");
          return ACL_SYS_ERROR;
        }
#else
        assert(!"Cannot go down this path without AMDIL support!");
        return ACL_SYS_ERROR;
#endif
      }
      break;
  }

  if (name == NULL || !need_name) {
    return cl->clAPI.insSec(cl, bin, to_data, to_data_size, sec);
  } else {
    return cl->clAPI.setType(cl, bin, name, to, to_data, to_data_size);
  }
}

acl_error  ACL_API_ENTRY
if_aclDisassemble(aclCompiler *cl,
    aclBinary *bin,
    const char *kernel,
    aclLogFunction disasm_callback)
{
  acl_error error_code = ACL_SUCCESS;
  size_t size = 0;
  const void *code = NULL;
  aclLoaderData *data = cl->beAPI.init(cl, bin, disasm_callback, &error_code);
  if (error_code != ACL_SUCCESS) {
    goto internal_disasm_failure;
  }
  code = cl->clAPI.devBinary(cl, bin, kernel, &size, &error_code);
  if (error_code != ACL_SUCCESS) {
    goto internal_disasm_failure;
  }
  error_code = cl->beAPI.disassemble(data, kernel, code, size);
  if (error_code != ACL_SUCCESS) {
    goto internal_disasm_failure;
  }
#ifdef WITH_TARGET_HSAIL
  {
    amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(data);
    if (isHSAILTarget(cs->Elf()->target)) {
      amdcl::HSAIL *hsail_be = reinterpret_cast<amdcl::HSAIL*>(data);
      if (!hsail_be) {
        goto internal_disasm_failure;
      }
      hsail_be->disassembleBRIG(cl, bin);
    }
  }
#endif
  error_code = cl->beAPI.fini(data);
  if (error_code != ACL_SUCCESS) {
    goto internal_disasm_failure;
  }
  return error_code;
internal_disasm_failure:
  const char *error = aclGetErrorString(error_code);
  appendLogToCL(cl, error);
  if (disasm_callback) {
    disasm_callback(cl->buildLog, cl->logSize);
  }
  return error_code;
}

const void*  ACL_API_ENTRY
if_aclGetDeviceBinary(aclCompiler *cl,
    const aclBinary *bin,
    const char *kernel,
    size_t *size,
    acl_error *error_code)
{
  const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symISABinary);
  assert(symbol && "symbol not found");
  std::string kernelName = symbol->str[PRE] + std::string(kernel) + symbol->str[POST];
  return cl->clAPI.extSym(cl, bin, size,
      symbol->sections[0], kernelName.c_str(), error_code);
}

acl_error  ACL_API_ENTRY
if_aclInsertSection(aclCompiler *cl,
    aclBinary *binary,
    const void *data,
    size_t data_size,
    aclSections id)
{
  bifbase *elfBin = reinterpret_cast<bifbase*>(binary->bin);
  if (!elfBin) {
    return ACL_ELF_ERROR;
  }
  if (!elfBin->addSection(id, data, data_size)) {
    return ACL_ELF_ERROR;
  }
  return ACL_SUCCESS;

}

acl_error  ACL_API_ENTRY
if_aclInsertSymbol(aclCompiler *cl,
    aclBinary *binary,
    const void *data,
    size_t data_size,
    aclSections id,
    const char *symbol)
{
  bifbase *elfBin = reinterpret_cast<bifbase*>(binary->bin);
  if (!elfBin) {
    return ACL_ELF_ERROR;
  }
  if (!elfBin->addSymbol(id, symbol,
        reinterpret_cast<const char*>(data), data_size)) {
    return ACL_ELF_ERROR;
  }
  return ACL_SUCCESS;

}

const void*  ACL_API_ENTRY
if_aclExtractSection(aclCompiler *cl,
    const aclBinary *binary,
    size_t *size,
    aclSections id,
    acl_error *error_code)
{
  bifbase *elfBin = reinterpret_cast<bifbase*>(binary->bin);
  if (!elfBin) {
    if (error_code) (*error_code) = ACL_ELF_ERROR;
    return NULL;
  }
  const void* a = elfBin->getSection(id, size);
  if (a == NULL) {
    if (error_code) (*error_code) = ACL_ELF_ERROR;
    return NULL;
  }
  if (error_code) (*error_code) = ACL_SUCCESS;
  return a;

}

const void*  ACL_API_ENTRY
if_aclExtractSymbol(aclCompiler *cl,
    const aclBinary *binary,
    size_t *size,
    aclSections id,
    const char *symbol,
    acl_error *error_code)
{
  bifbase *elfBin = reinterpret_cast<bifbase*>(binary->bin);
  if (!elfBin) {
    if (error_code) (*error_code) = ACL_ELF_ERROR;
    return NULL;
  }
  const void* a = elfBin->getSymbol(id, symbol, size);
  if (a == NULL) {
    if (error_code) (*error_code) = ACL_ELF_ERROR;
    return NULL;
  }
  if (error_code) (*error_code) = ACL_SUCCESS;
  return a;

}

acl_error  ACL_API_ENTRY
if_aclRemoveSection(aclCompiler *cl,
    aclBinary *binary,
    aclSections id)
{
  bifbase *elfBin = reinterpret_cast<bifbase*>(binary->bin);
  if (!elfBin) {
    return ACL_ELF_ERROR;
  }
  return elfBin->removeSection(id) ? ACL_SUCCESS : ACL_ELF_ERROR;
}

acl_error  ACL_API_ENTRY
if_aclRemoveSymbol(aclCompiler *cl,
    aclBinary *binary,
    aclSections id,
    const char *symbol)
{
  bifbase *elfBin = reinterpret_cast<bifbase*>(binary->bin);
  if (!elfBin) {
    return ACL_ELF_ERROR;
  }
  return elfBin->removeSymbol(id, symbol) ? ACL_SUCCESS : ACL_ELF_ERROR;
}

// Function performs deserialization of aclMetadata into *md 
// instead of changing source .rodata section in memory pointed by *ptr.
// Deserialization includes restoring of pointers, whereas
// serialized .rodata has pointers set to NULL by serializeMetadata function.
// We should leave serialized metaData unchanged (e.g. w/o garbage pointers)
// due to obtain the same binary from one compilation to another.
// Otherwise, OpenCL conformance "binary_create" test would fail on comparison 
// of OpenCL "binaries" (bifs in our case).
void deserializeCLMetadata(const char* ptr, aclMetadata * const md, const size_t size)
{
  memcpy(md,ptr,size);
  char *tmp_ptr = reinterpret_cast<char*>(md);
  tmp_ptr += md->struct_size;
  // de-serialize the kernel name
  md->kernelName = tmp_ptr;
  tmp_ptr += md->kernelNameSize + 1;

  // de-serialize the device name
  md->deviceName = tmp_ptr;
  tmp_ptr += md->deviceNameSize + 1;

  // de-serailize the arguments
  md->args = reinterpret_cast<aclArgData*>(tmp_ptr);
  tmp_ptr += (md->numArgs + 1) * sizeof(aclArgData);

  for (unsigned x = 0; x < md->numArgs; ++x) {
    // Get a pointer to the structure
    aclArgData *argPtr = md->args + x;

    // de-serialize the argument name string
    argPtr->argStr = tmp_ptr;
    tmp_ptr += argPtr->argNameSize + 1;

    // de-serialize the argument type string
    argPtr->typeStr = tmp_ptr;
    tmp_ptr += argPtr->typeStrSize + 1;
  }

  // de-serialize the printf strings
  md->printf = reinterpret_cast<aclPrintfFmt*>(tmp_ptr);
  tmp_ptr += sizeof(aclPrintfFmt) * (md->numPrintf + 1);
  for (unsigned x = 0; x < md->numPrintf; ++x) {
    // Get a pointer to the printf structure
    aclPrintfFmt *fmtPtr = md->printf + x;

    // de-serialize the arguments
    fmtPtr->argSizes = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(tmp_ptr));
    tmp_ptr += sizeof(uint32_t) * fmtPtr->numSizes;

    // de-serialize the format string
    fmtPtr->fmtStr = tmp_ptr;
    tmp_ptr += fmtPtr->fmtStrSize + 1;
  }
  assert(md->data_size == size && "The size and data size calculations are off!");
  assert((size_t)(tmp_ptr - reinterpret_cast<char*>(md)) 
    == size && "Size of data and calculated sizes differ!");
}

acl_error  ACL_API_ENTRY
if_aclQueryInfo(aclCompiler *cl,
    const aclBinary *binary,
    aclQueryType query,
    const char *kernel,
    void *ptr,
    size_t *size)
{
  const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclMeta);
  assert(sym && "symbol not found");
  std::string symbol = sym->str[PRE] + std::string(kernel) + sym->str[POST];
  size_t roSize;
  acl_error error_code;
  const void* roSec = cl->clAPI.extSym(cl, binary, &roSize,
      sym->sections[0], symbol.c_str(), &error_code);
  if (error_code != ACL_SUCCESS) return error_code;
  if (roSec == NULL || roSize == 0) {
    return ACL_ELF_ERROR;
  }
  bool success = true;

  switch (query) {
    default: break;
    case RT_CPU_BARRIER_NAMES:
             if (size != NULL && ptr == NULL) {
               (*size) = 0;
             } else if (ptr != NULL && size != NULL) {
               assert(!"Not implemented!");
               success = false;
             } else {
               success = false;
             }
             break;
    case RT_ABI_VERSION:
             if (size != NULL && ptr == NULL) {
               (*size) = sizeof(uint32_t) * 3;
             } else if (ptr != NULL && (*size) >= (sizeof(uint32_t) * 3)) {
               uint32_t *tmp = reinterpret_cast<uint32_t*>(ptr);
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               tmp[0] = md->major;
               tmp[1] = md->minor;
               tmp[2] = md->revision;
             } else {
               success = false;
             }
             break;
    case RT_DEVICE_NAME:
             if (size != NULL && ptr == NULL) {
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               (*size) = md->deviceNameSize;
             } else if (ptr != NULL) {
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               success = false;
               if ((*size) >= md->deviceNameSize) {
                 strncpy(reinterpret_cast<char*>(ptr), reinterpret_cast<const char*>(roSec)
                     + md->struct_size + md->kernelNameSize + 1, md->deviceNameSize);
                 success = true;
               }
             } else {
               success = false;
             }
             break;
    case RT_MEM_SIZES:
             if (size != NULL && ptr == NULL) {
               (*size) = sizeof(size_t) * RT_MEM_LAST;
             } else if (ptr != NULL && (*size) >= (sizeof(size_t) * RT_MEM_LAST)) {
               size_t *tmp = reinterpret_cast<size_t*>(ptr);
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               memcpy(tmp, md->mem, sizeof(size_t) * RT_MEM_LAST);
             } else {
               success = false;
             }
             break;
    case RT_GPU_FUNC_CAPS:
             if (binary->target.arch_id == aclX86) success = false;
             if (size != NULL && ptr == NULL) {
               (*size) = sizeof(uint32_t);
             } else if (ptr != NULL && (*size) >= sizeof(uint32_t)) {
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               (*reinterpret_cast<uint32_t*>(ptr)) = md->gpuCaps;
             } else {
               success = false;
             }
             break;
    case RT_GPU_FUNC_ID:
             if (binary->target.arch_id == aclX86) success = false;
             if (size != NULL && ptr == NULL) {
               (*size) = sizeof(uint32_t);
             } else if (ptr != NULL && (*size) >= sizeof(uint32_t)) {
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               (*reinterpret_cast<uint32_t*>(ptr)) = md->funcID;
             } else {
               success = false;
             }
             break;
    case RT_GPU_DEFAULT_ID:
             if (binary->target.arch_id == aclX86) success = false;
             if (size != NULL && ptr == NULL) {
               (*size) = sizeof(uint32_t) * RT_RES_LAST;
             } else if (ptr != NULL && (*size) >= (sizeof(uint32_t) * RT_RES_LAST)) {
               uint32_t *tmp = reinterpret_cast<uint32_t*>(ptr);
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               memcpy(tmp, md->gpuRes, sizeof(uint32_t) * RT_RES_LAST);
             } else {
               success = false;
             }
             break;
    case RT_WORK_GROUP_SIZE:
             if (size != NULL && ptr == NULL) {
               (*size) = sizeof(size_t) * 3;
             } else if (ptr != NULL && (*size) >= (sizeof(size_t) * 3)) {
               size_t *tmp = reinterpret_cast<size_t*>(ptr);
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               memcpy(tmp, md->wgs, 3 * sizeof(size_t));
             } else {
               success = false;
             }
             break;
    case RT_WORK_REGION_SIZE:
             if (size != NULL && ptr == NULL) {
               (*size) = sizeof(uint32_t) * 3;
             } else if (ptr != NULL && (*size) >= (sizeof(uint32_t) * 3)) {
               uint32_t *tmp = reinterpret_cast<uint32_t*>(ptr);
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
               memcpy(tmp, md->wrs, 3 * sizeof(uint32_t));
             } else {
               success = false;
             }
             break;
    case RT_ARGUMENT_ARRAY: {
             aclMetadata *md = static_cast<aclMetadata*>(malloc(roSize));
             if (size != NULL && ptr == NULL) {
               deserializeCLMetadata(reinterpret_cast<const char*>(roSec), md, roSize);
               (*size) = sizeof(aclArgData) * (md->numArgs + 1);
               for (unsigned x = 0; x < md->numArgs; ++x) {
                 (*size) += md->args[x].typeStrSize + md->args[x].argNameSize + 2;
               }
             } else if (ptr) {
               deserializeCLMetadata(reinterpret_cast<const char*>(roSec), md, roSize);
               unsigned totSize = sizeof(aclArgData) * (md->numArgs + 1);
               for (unsigned x = 0; x < md->numArgs; ++x) {
                 totSize += md->args[x].typeStrSize + md->args[x].argNameSize + 2;
               }
               if ((*size) >= totSize) {
                 char *tmp = reinterpret_cast<char*>(ptr);
                 memset(ptr, 0, (*size));
                 memcpy(ptr, md->args, sizeof(aclArgData) * (md->numArgs + 1));
                 tmp += (sizeof(aclArgData) * (md->numArgs + 1));
                 for (unsigned x = 0; x < md->numArgs; ++x) {
                   memcpy(tmp, md->args[x].argStr, md->args[x].argNameSize);
                   reinterpret_cast<aclArgData*>(ptr)[x].argStr = tmp;
                   tmp += md->args[x].argNameSize + 1;
                   tmp[-1] = '\0';
                   memcpy(tmp, md->args[x].typeStr, md->args[x].typeStrSize);
                   reinterpret_cast<aclArgData*>(ptr)[x].typeStr = tmp;
                   tmp += md->args[x].typeStrSize + 1;
                   tmp[-1] = '\0';
                 }
               } else {
                 success = false;
               }
             } else {
               success = false;
             }
             free(md);
             break;
           }
    case RT_GPU_PRINTF_ARRAY: {
             aclMetadata *md = static_cast<aclMetadata*>(malloc(roSize));
             if (size != NULL && ptr == NULL) {
               deserializeCLMetadata(reinterpret_cast<const char*>(roSec), md, roSize);
               (*size) = 0;
               if (md->numPrintf > 0) {
                 (*size) = sizeof(aclPrintfFmt) * (md->numPrintf + 1);
                 for (unsigned x = 0; x < md->numPrintf; ++x) {
                   (*size) += sizeof(uint32_t) * md->printf[x].numSizes;
                   (*size) += md->printf[x].fmtStrSize + 1;
                 }
               }
             } else if (ptr != NULL) {
               deserializeCLMetadata(reinterpret_cast<const char*>(roSec), md, roSize);
               unsigned totSize = sizeof(aclPrintfFmt) * (md->numPrintf + 1);
               for (unsigned x = 0; x < md->numPrintf; ++ x) {
                 totSize += sizeof(uint32_t) + md->printf[x].fmtStrSize + 1;
               }
               if ((*size) >= totSize) {
                 char *tmp = reinterpret_cast<char*>(ptr);
                 memcpy(ptr, md->printf, sizeof(aclPrintfFmt) * (md->numPrintf + 1));
                 tmp += (sizeof(aclPrintfFmt) * (md->numPrintf + 1));
                 for (unsigned x = 0; x < md->numPrintf; ++x) {
                   memcpy(tmp, md->printf[x].argSizes, sizeof(uint32_t) * md->printf[x].numSizes);
                   reinterpret_cast<aclPrintfFmt*>(ptr)[x].argSizes = reinterpret_cast<uint32_t*>(tmp);
                   tmp += sizeof(uint32_t) * md->printf[x].numSizes;
                   memcpy(tmp, md->printf[x].fmtStr, md->printf[x].fmtStrSize);
                   reinterpret_cast<aclPrintfFmt*>(ptr)[x].fmtStr = tmp;
                   tmp += md->printf[x].fmtStrSize + 1;
                   tmp[-1] = '\0';
                 }
               } else {
                 success = false;
               }
             } else {
               success = false;
             }
             free(md);
             break;
           }
    case RT_DEVICE_ENQUEUE: {
           if (size != NULL && ptr == NULL) {
             (*size) = sizeof(uint32_t);
           } else if (ptr != NULL && (*size) >= (sizeof(uint32_t))) {
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
              (*reinterpret_cast<uint32_t*>(ptr)) = md->enqueue_kernel;
           } else {
             success = false;
           }
           break;
        }
    // Temporary approach till the "ldk" instruction is supported.
    case RT_KERNEL_INDEX: {
           if (size != NULL && ptr == NULL) {
             (*size) = sizeof(uint32_t);
           } else if (ptr != NULL && (*size) >= (sizeof(uint32_t))) {
               const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
              (*reinterpret_cast<uint32_t*>(ptr)) = md->kernel_index;
           } else {
             success = false;
           }
           break;
        }
  }
  return (success) ? ACL_SUCCESS : ACL_ERROR;
}
static unsigned getSize(aclArgDataType data)
{
  switch(data) {
    default:
      return 4;
    case DATATYPE_i64:
    case DATATYPE_u64:
    case DATATYPE_f64:
      return 8;
    case DATATYPE_f80:
    case DATATYPE_f128:
      return 16;
  }
  return 4;
}
acl_error  ACL_API_ENTRY
if_aclDbgAddArgument(aclCompiler *cl,
    aclBinary *bin,
    const char *kernel,
    const char *name,
    bool byVal)
{
  if (!isAMDILTarget(bin->target)) {
    return ACL_UNSUPPORTED;
  }
  const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclMeta);
  assert(sym && "symbol not found");
  std::string symbol = sym->str[PRE] + std::string(kernel) + sym->str[POST];
  size_t roSize;
  acl_error error_code;
  aclMetadata *md = NULL;
  {
    const char* roSec = reinterpret_cast<const char*>(cl->clAPI.extSym(
      cl, bin, &roSize, sym->sections[0], symbol.c_str(), &error_code));
    if (error_code != ACL_SUCCESS) return error_code;
    if (roSec == NULL || roSize == 0) {
      return ACL_ELF_ERROR;
    }
    md = static_cast<aclMetadata*>(malloc(roSize));
    if (md == NULL) return ACL_OUT_OF_MEM;
    deserializeCLMetadata(roSec, md, roSize);
  }
  std::string dbg_name = name;
  size_t newSize = roSize + sizeof(aclArgData) + dbg_name.size() + 9;
  char *newMDptr = new char[newSize];
  char *tmp_ptr = newMDptr;
  memset(newMDptr, 0, newSize);
  aclMetadata *newMD = reinterpret_cast<aclMetadata*>(newMDptr);
  memcpy(tmp_ptr, md, md->struct_size
      + (md->kernelNameSize + 1)
      + (md->deviceNameSize + 1));
  tmp_ptr += md->struct_size;
  tmp_ptr += md->kernelNameSize + 1;
  tmp_ptr[-1] = '\0';
  tmp_ptr += md->deviceNameSize + 1;
  tmp_ptr[-1] = '\0';
  newMD->args = reinterpret_cast<aclArgData*>(tmp_ptr);
  unsigned cb_offset = 0;
  const aclArgData *c_argPtr = reinterpret_cast<const aclArgData*>(
      reinterpret_cast<const char*>(md) + (tmp_ptr - newMDptr));
  for (unsigned x = 0; x < md->numArgs; ++x) {
    switch (c_argPtr[x].type) {
      default:
      case ARG_TYPE_ERROR:
        assert(!"Unknown type!");
        break;
      case ARG_TYPE_SAMPLER:
        break;
      case ARG_TYPE_COUNTER:
        if (c_argPtr[x].arg.counter.cbOffset >= cb_offset) {
          cb_offset = c_argPtr[x].arg.counter.cbOffset + 16;
        }
        break;
      case ARG_TYPE_POINTER:
        if (c_argPtr[x].arg.pointer.cbOffset >= cb_offset) {
          cb_offset = c_argPtr[x].arg.pointer.cbOffset + 16;
        }
        break;
      case ARG_TYPE_SEMAPHORE:
        if (c_argPtr[x].arg.sema.cbOffset >= cb_offset) {
          cb_offset = c_argPtr[x].arg.sema.cbOffset + 16;
        }
        break;
      case ARG_TYPE_IMAGE:
        if (c_argPtr[x].arg.image.cbOffset >= cb_offset) {
          cb_offset = c_argPtr[x].arg.image.cbOffset + 16;
        }
        break;
      case ARG_TYPE_VALUE:
        if (c_argPtr[x].arg.value.cbOffset >= cb_offset) {
          unsigned offs = c_argPtr[x].arg.value.numElements * getSize(c_argPtr[x].arg.value.data);
          cb_offset = c_argPtr[x].arg.value.cbOffset + (offs > 16 ? offs : 16);
        }
        break;
    }
    size_t arg_size = c_argPtr[x].struct_size;
    memcpy(tmp_ptr, &c_argPtr[x], arg_size);
    tmp_ptr += arg_size;
  }
  // Skip the new one and the sentinal one.
  tmp_ptr += (sizeof(aclArgData) * 2);
  // Copy all of the name/type strings.
  for (unsigned x = 0; x < md->numArgs; ++x) {
    memcpy(tmp_ptr, md->args[x].argStr, md->args[x].argNameSize);
    tmp_ptr += md->args[x].argNameSize + 1;
    tmp_ptr[-1] = '\0';
    memcpy(tmp_ptr, md->args[x].typeStr, md->args[x].typeStrSize);
    tmp_ptr += md->args[x].typeStrSize + 1;
    tmp_ptr[-1] = '\0';
  }
  size_t printf_offset = reinterpret_cast<const char*>(md->printf)
    - reinterpret_cast<const char*>(md);
  aclArgData *argPtr = &newMD->args[newMD->numArgs];
  newMD->numArgs++;
  if (byVal) {
    argPtr->type = ARG_TYPE_VALUE;
    argPtr->arg.value.data = DATATYPE_u32;
    argPtr->arg.value.numElements = 4;
    argPtr->arg.value.cbNum = 2;
    argPtr->arg.value.cbOffset = cb_offset;
  } else {
    argPtr->type = ARG_TYPE_POINTER;
    argPtr->arg.pointer.data = DATATYPE_u32;
    argPtr->arg.pointer.numElements = 1;
    argPtr->arg.pointer.cbNum = 2;
    argPtr->arg.pointer.cbOffset = cb_offset;
    argPtr->arg.pointer.memory = PTR_MT_GLOBAL;
    argPtr->arg.pointer.bufNum = md->gpuRes[RT_RES_UAV];
    argPtr->arg.pointer.align = 4;
    argPtr->arg.pointer.type = ACCESS_TYPE_RW;
    argPtr->arg.pointer.isVolatile = false;
    argPtr->arg.pointer.isRestrict = false;
  }
  argPtr->argNameSize = dbg_name.size() + 7;
  argPtr->typeStrSize = 0;
  argPtr->typeStr = "";
  argPtr->isConst = false;
  argPtr->struct_size = sizeof(aclArgData);
  argPtr->argStr = tmp_ptr;
  memcpy(tmp_ptr, "_debug_", 7);
  tmp_ptr += 7;
  memcpy(tmp_ptr, dbg_name.data(), dbg_name.size());
  tmp_ptr += dbg_name.size() + 1;
  tmp_ptr[-1] = '\0';
  memcpy(tmp_ptr, argPtr->typeStr, argPtr->typeStrSize);
  tmp_ptr += argPtr->typeStrSize + 1;
  tmp_ptr[-1] = '\0';
  newMD->printf = reinterpret_cast<aclPrintfFmt*>(tmp_ptr);
  newMD->data_size = newSize;
  memcpy(tmp_ptr, reinterpret_cast<const char*>(md) + printf_offset, roSize - printf_offset);
  tmp_ptr += (roSize - printf_offset);
  cl->clAPI.remSym(cl, bin, aclRODATA, symbol.c_str());
  error_code = cl->clAPI.insSym(cl, bin, newMDptr, newSize,
      aclRODATA, symbol.c_str());
  assert((size_t)(tmp_ptr - newMDptr) == newSize && "allocated memory does not equal the amount of memory copied!");
  free(md);
  delete [] newMDptr;
  return error_code;
}

acl_error  ACL_API_ENTRY
if_aclDbgRemoveArgument(aclCompiler *cl,
    aclBinary *bin,
    const char* kernel,
    const char* name)
{
  if (!isAMDILTarget(bin->target)) {
    return ACL_UNSUPPORTED;
  }
  const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclMeta);
  assert(sym && "symbol not found");
  std::string symbol = sym->str[PRE] + std::string(kernel) + sym->str[POST];
  size_t roSize;
  acl_error error_code;
  aclMetadata *md = NULL;
  {
    const char* roSec = reinterpret_cast<const char*>(cl->clAPI.extSym(cl, bin, &roSize,
        sym->sections[0], symbol.c_str(), &error_code));
    if (error_code != ACL_SUCCESS) return error_code;
    if (roSec == NULL || roSize == 0) {
      return ACL_ELF_ERROR;
    }
    md = static_cast<aclMetadata*>(malloc(roSize));
    if (md == NULL) return ACL_OUT_OF_MEM;
    deserializeCLMetadata(roSec, md, roSize);
  }
  const char* ro_ptr = reinterpret_cast<const char*>(md);
  ro_ptr += md->struct_size;
  ro_ptr += md->kernelNameSize + 1;
  ro_ptr += md->deviceNameSize + 1;
  const aclArgData *argPtr = reinterpret_cast<const aclArgData*>(ro_ptr);
  const aclArgData *delArg = 0;
  for (unsigned x = 0; x < md->numArgs; ++x) {
    if (0 != argPtr[x].argStr
        && !strncmp("_debug_", argPtr[x].argStr, 7)
        && !strcmp(name, argPtr[x].argStr + 7)) {
      delArg = &argPtr[x];
      break;
    }
  }
  if (0 == delArg) {
    return ACL_INVALID_ARG;
  }
  size_t newSize = roSize - (delArg->struct_size + delArg->argNameSize + delArg->typeStrSize + 2);
  char *newMDptr = new char[newSize];
  memset(newMDptr, 0, newSize);
  aclMetadata *newMD = reinterpret_cast<aclMetadata*>(newMDptr);
  char *tmp_ptr = newMDptr;
  memcpy(tmp_ptr, reinterpret_cast<const char*>(md), md->struct_size
      + (md->kernelNameSize + 1)
      + (md->deviceNameSize + 1));
  tmp_ptr += md->struct_size;
  tmp_ptr += md->kernelNameSize + 1;
  tmp_ptr[-1] = '\0';
  tmp_ptr += md->deviceNameSize + 1;
  tmp_ptr[-1] = '\0';
  unsigned cb_offset = ((delArg->type == ARG_TYPE_VALUE)
      ? delArg->arg.value.cbOffset : delArg->arg.pointer.cbOffset);
  size_t printf_offset = reinterpret_cast<const char*>(md->printf)
    - reinterpret_cast<const char*>(md);
  newMD->numArgs--;
  for (unsigned x = 0; x < md->numArgs; ++x) {
    size_t arg_size = argPtr[x].struct_size;
    if (strcmp(argPtr[x].argStr, delArg->argStr)) {
      memcpy(tmp_ptr, &argPtr[x], arg_size);
      aclArgData *tmpArg = reinterpret_cast<aclArgData*>(tmp_ptr);
      tmp_ptr += arg_size;
      switch (argPtr[x].type) {
        default:
        case ARG_TYPE_ERROR:
          assert(!"Unknown type!");
          break;
        case ARG_TYPE_SAMPLER:
          break;
        case ARG_TYPE_COUNTER:
          if (tmpArg->arg.counter.cbOffset >= cb_offset) {
            tmpArg->arg.counter.cbOffset -= 16;
          }
          break;
        case ARG_TYPE_POINTER:
          if (tmpArg->arg.pointer.cbOffset >= cb_offset) {
            tmpArg->arg.pointer.cbOffset -= 16;
          }
          break;
        case ARG_TYPE_SEMAPHORE:
          if (tmpArg->arg.sema.cbOffset >= cb_offset) {
            tmpArg->arg.sema.cbOffset -= 16;
          }
          break;
        case ARG_TYPE_IMAGE:
          if (tmpArg->arg.image.cbOffset >= cb_offset) {
            tmpArg->arg.image.cbOffset -= 16;
          }
          break;
        case ARG_TYPE_VALUE:
          if (tmpArg->arg.value.cbOffset >= cb_offset) {
            tmpArg->arg.value.cbOffset -= 16;
          }
          break;
      }
    }
  }
  memset(tmp_ptr, 0, delArg->struct_size);
  tmp_ptr += delArg->struct_size;
  for (unsigned x = 0; x < md->numArgs; ++x) {
    size_t arg_size = argPtr[x].struct_size;
    if (strcmp(argPtr[x].argStr, delArg->argStr)) {
      memcpy(tmp_ptr, argPtr[x].argStr, argPtr[x].argNameSize);
      tmp_ptr += argPtr[x].argNameSize + 1;
      tmp_ptr[-1] = '\0';
      memcpy(tmp_ptr, argPtr[x].typeStr, argPtr[x].typeStrSize);
      tmp_ptr += argPtr[x].typeStrSize + 1;
      tmp_ptr[-1] = '\0';
    }
  }

  memcpy(tmp_ptr, reinterpret_cast<const char*>(md) + printf_offset, roSize - printf_offset);
  tmp_ptr += (roSize - printf_offset);
  newMD->data_size = newSize;
  cl->clAPI.remSym(cl, bin, aclRODATA, symbol.c_str());
  error_code = cl->clAPI.insSym(cl, bin, newMDptr, newSize,
      aclRODATA, symbol.c_str());
  assert((size_t)(tmp_ptr - newMDptr) == newSize && "allocated memory does not equal the amount of memory copied!");
  free(md);
  delete [] newMDptr;
  return error_code;
}

void myLogFunc(const char * msg, size_t size)
{
  printf("%s\n", msg);
}

extern "C" {
bool aclRenderscriptCompile(
  char * srcFile,
  char ** outBuf,
  size_t * outLen
)
{
#if 0
  // Consider using code here if aoc2 is not used.
  llvm::Module *bc = NULL;
  llvm::LLVMContext &Context = llvm::getGlobalContext();
  llvm::SMDiagnostic Err;
  std::string Str(srcFile);

  bc = llvm::ParseIRFile(Str, Err, Context);
  if (!bc)
    return false;

  llvm::PassManager TransformPasses;
  TransformPasses.add(llvm::createOpenCLIRTransform());
  TransformPasses.run(*bc);
#endif

  size_t size = 0;
  acl_error error_code;
  char * source = readFile(srcFile, size);
  if (!size)
    return false;

  aclCompiler *aoc = aclCompilerInit(NULL, &error_code);
  if ((aoc == NULL) || (error_code != ACL_SUCCESS))
    return false;

  aclTargetInfo target = aclGetTargetInfo("hsail", "Bonaire", &error_code);
  if (error_code != ACL_SUCCESS) 
    return false;
  
  aclBinary *aoe = aclBinaryInit(sizeof(aclBinary), &target, NULL, &error_code);
  if (error_code != ACL_SUCCESS) 
    return false;
  
  error_code = aclInsertSection(aoc, aoe, source, size, aclLLVMIR);
  if (error_code != ACL_SUCCESS) 
    return false;

#if 1
  // Dump HSAIL and ISA to a temporary file in the working directory.
  error_code = aclCompile(aoc, aoe, "-save-temps=tmp", ACL_TYPE_RSLLVMIR_BINARY, ACL_TYPE_ISA, myLogFunc);
#else
  error_code = aclCompile(aoc, aoe, NULL, ACL_TYPE_RSLLVMIR_BINARY, ACL_TYPE_ISA, myLogFunc);
#endif
	
  if (error_code == ACL_FRONTEND_FAILURE) {
    printf("ACL_FRONTEND_FAILURE.\n");
    return true;
  }

  if (error_code != ACL_SUCCESS)
    return false;

  if ((aoe == NULL) || (aoe->bin == NULL))
    return false;
  
  char *buffer = NULL;
  size_t len;
  acl_error errCode = aclWriteToMem(aoe, reinterpret_cast<void**>(&buffer), &len);
  if (errCode != ACL_SUCCESS) 
    return false;

  *outLen = len;
  *outBuf = buffer;
  return true;
}
}
