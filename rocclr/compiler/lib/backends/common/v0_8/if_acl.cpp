//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifdef WITH_TARGET_HSAIL
#include "HSAILBrigContainer.h"
#include "HSAILDisassembler.h"
#include "HSAILBrigObjectFile.h"

//prevent macro redefinition in drivers\hsa\compiler\lib\promotions\oclutils\top.hpp
//as it's already defined in drivers\hsa\compiler\llvm\include\llvm\Support\Format.h
#undef snprintf
#endif

#include "acl.h"
#include "aclTypes.h"
#include "compiler_stage.hpp"
#include "frontend.hpp"
#include "spir.hpp"

#if defined DEBUG
#undef DEBUG
#endif

#include "codegen.hpp"
#include "library.hpp"
#include "linker.hpp"
#include "optimizer.hpp"
#include "amdil_be.hpp"
#include "hsail_be.hpp"
#include "x86_be.hpp"
#include "os/os.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"
#include "utils/options.hpp"
#include "utils/target_mappings.h"
#include "utils/versions.hpp"
#include "sync.hpp"

#include "llvm/Analysis/Passes.h"
#if defined(LEGACY_COMPLIB)
#include "Disassembler.h"
#include "llvm/LLVMContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/ExecutionEngine/ObjectBuffer.h"
#else
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/SPIRV.h"
#endif
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"

#include "bif/bifbase.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cassert>
#include <iomanip>

aclLoaderData * ACL_API_ENTRY
if_aclCompilerInit(aclCompiler *cl, aclBinary *bin,
    aclLogFunction log, acl_error *error)
{
  amdcl::acquire_global_lock();
  char* timing = ::getenv("AMD_DEBUG_HLC_ENABLE_TIMING");
  if (timing && (timing[0] == '1'))
     llvm::TimePassesIsEnabled = true;
  else
     llvm::TimePassesIsEnabled = false;
  if (cl->llvm_shutdown == NULL) {
     cl->llvm_shutdown = reinterpret_cast<void*>
       (new llvm::llvm_shutdown_obj(
#if defined(LEGACY_COMPLIB)
                                    false
#endif
                                   ));
  }
  static const char *DumpStackTrace = getenv("AMD_DUMP_STACK_TRACE");
  if (DumpStackTrace) {
    llvm::EnablePrettyStackTrace();
    llvm::sys::PrintStackTraceOnErrorSignal();
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
  llvm::initializeCodeGen(Registry);
  llvm::initializeTarget(Registry);
#if defined(LEGACY_COMPLIB)
  llvm::initializeVerifierPass(Registry);
  llvm::initializeDominatorTreePass(Registry);
  llvm::initializePreVerifierPass(Registry);
#endif
  amdcl::release_global_lock();
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
LOADER_FUNCS(HSAILFE, amdcl::ClangOCLFrontend);
LOADER_FUNCS(HSAILOpt, amdcl::GPUOptimizer);
#else
LOADER_FUNCS_ERROR(HSAILAsm, amdcl::HSAIL);
LOADER_FUNCS_ERROR(HSAILFE, amdcl::ClangOCLFrontend);
LOADER_FUNCS_ERROR(HSAILOpt, amdcl::GPUOptimizer);
#endif

#if defined(WITH_TARGET_X86)
LOADER_FUNCS(X86Asm, amdcl::X86);
LOADER_FUNCS(X86Opt, amdcl::CPUOptimizer);
#else
LOADER_FUNCS_ERROR(X86Asm, amdcl::X86);
LOADER_FUNCS_ERROR(X86Opt, amdcl::CPUOptimizer);
#endif

#if defined(LEGACY_COMPLIB)
LOADER_FUNCS(OCL, amdcl::OCLFrontend);
#else
LOADER_FUNCS(OCL, amdcl::ClangOCLFrontend);
#endif
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
#if defined(LEGACY_COMPLIB)
  llvm::MemoryBuffer *Buffer =
  llvm::MemoryBuffer::getMemBufferCopy(
    llvm::StringRef(llvmBinary), "input.bc");
  llvm::Module *M = NULL;
#else
  std::unique_ptr<llvm::MemoryBuffer> Buffer =
    llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(llvmBinary), "input.bc");
  llvm::ErrorOr<std::unique_ptr<llvm::Module>> ErrOrM(nullptr);
#endif

  if (llvm::isBitcode((const unsigned char *)Buffer->getBufferStart(),
                (const unsigned char *)Buffer->getBufferEnd())) {
#if defined(LEGACY_COMPLIB)
    M = llvm::ParseBitcodeFile(Buffer, *Context, &ErrorMessage);
#else
    ErrOrM = llvm::parseBitcodeFile(Buffer->getMemBufferRef(), *Context);
#endif
  }

#if defined(LEGACY_COMPLIB)
  if (M == NULL) {
#else
  if (ErrOrM.getError()) {
#endif
    if (error != NULL) (*error) = ACL_INVALID_BINARY;
    return NULL;
  }
#if !defined(LEGACY_COMPLIB)
  auto M = ErrOrM.get().release();
#endif
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
#if defined(LEGACY_COMPLIB)
  llvm::PassManager TransformPasses;
#else
  llvm::legacy::PassManager TransformPasses;
#endif
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

/// Update elf e_rawfile buffer.
static acl_error
updateElfRawFile(aclBinary *bin)
{
  if (bin == NULL
      || bin->bin == NULL) {
    return ACL_INVALID_ARG;
  }
  bifbase *elfBin = reinterpret_cast<bifbase*>(bin->bin);
  return elfBin->updateRawFile() ? ACL_SUCCESS : ACL_ELF_ERROR;
}

aclModule* ACL_API_ENTRY
SPIRVToModule(
    aclLoaderData *ald,
    const char *image,
    size_t length,
    aclContext *ctx,
    acl_error *error)
{
  auto compiler = reinterpret_cast<amdcl::LLVMCompilerStage*>(ald);
  auto cl = compiler->CL();
  auto bin = compiler->Elf();
#ifdef LEGACY_COMPLIB
  llvm::report_fatal_error("SPIR-V not supported on legacy compiler lib");
  appendLogToCL(cl, "SPIR-V not supported on legacy compiler lib");
  if (error != nullptr) (*error) = ACL_SPIRV_LOAD_FAIL;
  return nullptr;
#else
  std::string spvImg(image, length);
  /// ToDo: When there are multiple binaries, compiler->Options()
  /// cannot carry options specified by environment variables to here
  /// but bin->options can. This seems to be related to how runtime
  /// sets up aclCompiler options and BIF options.
  auto opt = reinterpret_cast<amd::option::Options*>(bin->options);
  if (opt->isDumpFlagSet(amd::option::DUMP_SPIRV)) {
    std::ofstream ofs(opt->getDumpFileName(".spv"), std::ios::binary);
    ofs << spvImg;
    ofs.close();
  }

  std::stringstream ss(spvImg);
  std::string errMsg;
  auto llCtx = reinterpret_cast<llvm::LLVMContext*>(ctx);
  llvm::Module *llMod = nullptr;
  if (opt->getLLVMArgc()) {
    llvm::cl::ParseCommandLineOptions(opt->getLLVMArgc(), opt->getLLVMArgv(),
      "SPIRV/LLVM converter");
  }
  bool success = llvm::ReadSPIRV(*llCtx, ss, llMod, errMsg);

  if (success && llMod && opt->isDumpFlagSet(amd::option::DUMP_BC_SPIRV)) {
    auto bcDump = opt->getDumpFileName("_fromspv.bc");
    std::error_code ec;
    llvm::raw_fd_ostream outS(bcDump.c_str(), ec, llvm::sys::fs::F_None);
    if (!ec)
      WriteBitcodeToFile(llMod, outS);
    else
      errMsg = ec.message();
  }

  if (!errMsg.empty()) {
    appendLogToCL(cl, errMsg);
  }
  if (!success || llMod == nullptr) {
    if (error != nullptr) (*error) = ACL_SPIRV_LOAD_FAIL;
    return nullptr;
  }

  llvm::SmallVector<char, 4096> array;
  llvm::raw_svector_ostream outstream(array);
  llvm::WriteBitcodeToFile(reinterpret_cast<llvm::Module*>(llMod), outstream);
  auto errCode = cl->clAPI.insSec(cl, bin, &array[0], array.size(), aclLLVMIR);
  if (error != nullptr) (*error) = errCode;
  if (errCode != ACL_SUCCESS)
    return reinterpret_cast<aclModule*>(llMod);

  errCode = updateElfRawFile(bin);
  if (error != nullptr) (*error) = errCode;
  return reinterpret_cast<aclModule*>(llMod);
#endif // LEGACY_COMPLIB
}

aclModule * ACL_API_ENTRY
LLVMToSPIRV(
    aclLoaderData *ald,
    const char *source,
    size_t data_size,
    aclContext *ctx,
    acl_error *error)
{
  auto compiler = reinterpret_cast<amdcl::LLVMCompilerStage*>(ald);
#ifdef LEGACY_COMPLIB
  llvm::report_fatal_error("SPIR-V not supported on legacy compiler lib");
  appendLogToCL(compiler->CL(), "SPIR-V not supported on legacy compiler lib");
  if (error != nullptr) (*error) = ACL_SPIRV_LOAD_FAIL;
  return nullptr;
#else

  std::string errMsg;
  auto opt = compiler->Options();
  llvm::Module *llMod = reinterpret_cast<llvm::Module *>(OCLFEToModule(
      ald, source, data_size, ctx, error));
  if (!llMod)
    return nullptr;

  if (opt->isDumpFlagSet(amd::option::DUMP_BC_SPIRV)) {
    auto bcDump = opt->getDumpFileName("_tospv.bc");
    std::error_code ec;
    llvm::raw_fd_ostream outS(bcDump.c_str(), ec, llvm::sys::fs::F_None);
    if (!ec)
      WriteBitcodeToFile(llMod, outS);
    else
      errMsg = ec.message();
  }

  std::string spvImg;
  llvm::raw_string_ostream ss(spvImg);
  bool success = llvm::WriteSPIRV(llMod, ss, errMsg);

  if (opt->isDumpFlagSet(amd::option::DUMP_SPIRV)) {
    std::ofstream ofs(opt->getDumpFileName(".spv"), std::ios::binary);
    ofs << ss.str();
    ofs.close();
  }

  if (!errMsg.empty()) {
    appendLogToCL(compiler->CL(), errMsg);
  }
  if (!success) {
    if (error != nullptr) (*error) = ACL_SPIRV_SAVE_FAIL;
    return nullptr;
  }

  if (error != nullptr) (*error) = ACL_SUCCESS;
  return reinterpret_cast<aclModule*>(llMod);
#endif // LEGACY_COMPLIB
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
    "-binomial-coefficient-limit-bitwidth=64",
    "-hsail-max-wg-size=2048"
  };

  aclLink->setContext(ctx);
  amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(aclLink->Elf()->options);
  int args = sizeof(argv) / sizeof(argv[0]);
  llvm::cl::ParseCommandLineOptions(args, (char**)argv, "OpenCL");

  if (Opts->getLLVMArgc())
    llvm::cl::ParseCommandLineOptions(Opts->getLLVMArgc(),
        Opts->getLLVMArgv(), "OpenCL");

  // LLVM Link phase
  std::vector<std::unique_ptr<llvm::Module>> libvec;
  for (unsigned x = 0; x < numLibs; ++x) {
    if (libs[x] != NULL) {
      libvec.push_back(std::unique_ptr<llvm::Module>(reinterpret_cast<llvm::Module*>(libs[x])));
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
  if (!isHSAILTarget(aclCG->Elf()->target)) {
    if (checkFlag(aclutGetCaps(aclCG->Elf()), capSaveCG)) {
      aclCG->CL()->clAPI.insSec(aclCG->CL(), aclCG->Elf(),
          aclCG->Source().data(),
          aclCG->Source().size(), aclCODEGEN);
    }
  }
  return reinterpret_cast<const void*>(&(aclCG->Source()));
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
  isaDump = acl->disassemble(isa_code, isa_size, kernel);
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
  } else if (useLinker || useOpt) {
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
    amdcl::CompilerStage *acs = reinterpret_cast<amdcl::CompilerStage*>(ald);
    if (isHSAILTarget(acs->Elf()->target)) {
      amdcl::HSAIL *acl = reinterpret_cast<amdcl::HSAIL*>(ald);
      bool bHsailTextInput = false;
      const char *hsail_text_input = getenv("AMD_DEBUG_HSAIL_TEXT_INPUT");
      // Verify that the internal (blit) kernel is not being compiled
      if (hsail_text_input && strcmp(hsail_text_input, "") != 0 && !acl->Options()->oVariables->clInternalKernel) {
        bHsailTextInput = true;
      }
      if (!bHsailTextInput) {
        // from ACL_TYPE_HSAIL_BINARY
        if (!useFE && !useLinker && !useOpt) {
          int result = 0;
          HSAIL_ASM::BrigContainer c;
          // BRIG is in aclSOURCE section
          if (data) {
            if (0 != HSAIL_ASM::BrigStreamer::load(c, data, data_size)) {
              appendLogToCL(cl, "ERROR: BRIG loading failed.");
              error_code = ACL_CODEGEN_ERROR;
              goto internal_compile_failure;
            }
            if (!acl->insertBRIG(c)) {
              appendLogToCL(cl, "ERROR: BRIG inserting failed.");
              error_code = ACL_CODEGEN_ERROR;
              goto internal_compile_failure;
            }
          // Only check that BRIG is in the binary
          } else {
            bool containsBRIG = false;
            size_t boolSise = sizeof(bool);
            error_code = aclQueryInfo(cl, bin, RT_CONTAINS_BRIG, NULL, &containsBRIG, &boolSise);
            if (!containsBRIG || error_code != ACL_SUCCESS) {
              appendLogToCL(cl, "ERROR: BRIG is absent or incomplete.");
              error_code = ACL_CODEGEN_ERROR;
              goto internal_compile_failure;
            }
          }
        // from ACL_TYPE_LLVMIR_BINARY
        } else {
          std::string* cg = (std::string*) cl->cgAPI.codegen(ald, module, context, &error_code);
          if (!cg || error_code != ACL_SUCCESS) {
            goto internal_compile_failure;
          }
          if (!acl->insertBRIG(*cg)) {
            appendLogToCL(cl, "ERROR: BRIG inserting failed.");
            error_code = ACL_CODEGEN_ERROR;
            goto internal_compile_failure;
          }
        }
      }
      // HSAIL substitution from AMD_DEBUG_HSAIL_TEXT_INPUT
      else {
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
          appendLogToCL(cl, "ERROR: AMD_DEBUG_HSAIL_TEXT_INPUT file does not exist.");
          error_code = ACL_CODEGEN_ERROR;
          goto internal_compile_failure;
        }
        if (!acl->insertHSAIL(dataStr)) {
          appendLogToCL(cl, "ERROR: HSAIL inserting failed.");
          error_code = ACL_CODEGEN_ERROR;
          goto internal_compile_failure;
        }
        // Use the assembler to generate the binary format of the IL string.
        if (HSAILAssemble(ald, dataStr.c_str(), dataStr.length()) != ACL_SUCCESS) {
          appendLogToCL(cl, "ERROR: HSAIL assembling failed.");
          error_code = ACL_CODEGEN_ERROR;
          goto internal_compile_failure;
        }
      }
      char* dumpFileName = ::getenv("AMD_DEBUG_DUMP_HSAIL_ALL_KERNELS");
      if (acl->Options()->isDumpFlagSet(amd::option::DUMP_CGIL) || dumpFileName) {
        acl->dumpHSAIL(acl->disassembleBRIG(), ".hsail");
      }
      bifbase *elfBin = reinterpret_cast<bifbase*>(bin->bin);
      elfBin->setType(ET_EXEC);
    } else if(isCpuTarget(acs->Elf()->target)) {
      std::string* cg = (std::string*) cl->cgAPI.codegen(ald, module, context, &error_code);
      if (!cg || error_code != ACL_SUCCESS) {
        goto internal_compile_failure;
      }
      dataStr = *cg;
    } else {
      assert("Unsupported architecture.");
    }
    if (!checkFlag(aclutGetCaps(bin), capSaveLLVMIR) || !acs->Options()->oVariables->BinLLVMIR) {
      cl->clAPI.remSec(cl, bin, aclLLVMIR);
    }
    cl->cgAPI.fini(ald);
    if (error_code != ACL_SUCCESS) {
      goto internal_compile_failure;
    }
  }

  if (useISA) {
    ald = cl->beAPI.init(cl, bin, compile_callback, &error_code);
    error_code = cl->beAPI.finalize(ald, dataStr.data(), dataStr.length());
    if (isHSAILTarget(bin->target) && error_code == ACL_SUCCESS) {
      amdcl::HSAIL *acl = reinterpret_cast<amdcl::HSAIL*>(cl->cgAPI.init(cl, bin, compile_callback, &error_code));
      acl->deleteBRIG();
      cl->cgAPI.fini(reinterpret_cast<aclLoaderData*>(acl));
    }
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
  return ACL_SUCCESS;
}

acl_error  ACL_API_ENTRY
if_aclCompile(aclCompiler *cl,
    aclBinary *bin,
    const char *options,
    aclType from,
    aclType to,
    aclLogFunction compile_callback)
{
  if (!bin || !cl) {
    return ACL_INVALID_ARG;
  }
  if (((from == ACL_TYPE_X86_TEXT   || from == ACL_TYPE_X86_BINARY)   && !isCpuTarget(bin->target))   ||
      ((from == ACL_TYPE_AMDIL_TEXT || from == ACL_TYPE_AMDIL_BINARY) && !isAMDILTarget(bin->target)) ||
      ((from == ACL_TYPE_HSAIL_TEXT || from == ACL_TYPE_HSAIL_BINARY) && !isHSAILTarget(bin->target))) {
    return ACL_INVALID_BINARY;
  }
  acl_error error_code = IsValidCompilationOptions(bin, compile_callback);
  if (error_code != ACL_SUCCESS) {
    return error_code;
  }
#ifdef WITH_TARGET_HSAIL
  if (isHSAILTarget(bin->target)) {
  } else
#endif
  {
    llvm::InitializeAllAsmParsers();
    llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeSPIRVerifierPass(Registry);
  }
  amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(bin->options);
  // Default 'to' is ACL_TYPE_ISA
  if (to == ACL_TYPE_DEFAULT) {
    to = ACL_TYPE_ISA;
  }
  if ((from == ACL_TYPE_HSAIL_TEXT    && (to == ACL_TYPE_HSAIL_BINARY ||
                                          to == ACL_TYPE_CG           ||
                                          to == ACL_TYPE_ISA))        ||
      (from == ACL_TYPE_HSAIL_BINARY  && to == ACL_TYPE_HSAIL_TEXT)   ||
      (from == ACL_TYPE_AMDIL_TEXT    && to == ACL_TYPE_AMDIL_BINARY) ||
      (from == ACL_TYPE_AMDIL_BINARY  && to == ACL_TYPE_AMDIL_TEXT)   ||
      (from == ACL_TYPE_SPIR_TEXT     && to == ACL_TYPE_SPIR_BINARY)  ||
      (from == ACL_TYPE_SPIR_BINARY   && to == ACL_TYPE_SPIR_TEXT)    ||
      (from == ACL_TYPE_LLVMIR_TEXT   && to == ACL_TYPE_LLVMIR_BINARY)||
      (from == ACL_TYPE_LLVMIR_BINARY && to == ACL_TYPE_LLVMIR_TEXT)  ||
      (from == ACL_TYPE_X86_TEXT      && to == ACL_TYPE_X86_BINARY)   ||
      (from == ACL_TYPE_X86_BINARY    && to == ACL_TYPE_X86_TEXT)) {
    const char *kernel = Opts->oVariables->Kernel;
    error_code = aclConvertType(cl, bin, kernel, from);
    // if compilation to ACL_TYPE_ISA, then continue from ACL_TYPE_CG
    if (to == ACL_TYPE_ISA && error_code == ACL_SUCCESS) {
      from = ACL_TYPE_CG;
    } else {
      return error_code;
    }
  }
  if (((from == ACL_TYPE_AMDIL_TEXT   || from == ACL_TYPE_AMDIL_BINARY  ||
        from == ACL_TYPE_X86_TEXT     || from == ACL_TYPE_X86_BINARY    ||
        from == ACL_TYPE_HSAIL_TEXT)  && to != ACL_TYPE_ISA) ||
       (from == ACL_TYPE_HSAIL_BINARY && to != ACL_TYPE_ISA && to != ACL_TYPE_CG)) {
    return ACL_INVALID_ARG;
  }
  if (to == ACL_TYPE_SPIRV_BINARY) {
    if (from == ACL_TYPE_OPENCL) {
      to = ACL_TYPE_LLVMIR_BINARY;
      Opts->oVariables->FEGenSPIRV = true;
    } else {
      return ACL_INVALID_ARG;
    }
  }
  uint8_t sectable[ACL_TYPE_LAST] = {0, 0, 1, 1, 1, 1, 0, 6, 0, 3, 4, 4, 4, 0,
      5, 0, 1, 1};
  aclSections d_section[7] = {aclSOURCE, aclLLVMIR, aclSPIR, aclSOURCE,
      aclCODEGEN, aclTEXT, aclINTERNAL};
  uint8_t start = sectable[from];
  uint8_t stop = sectable[to];
  const void* data = NULL;
  size_t data_size = 0;
  switch (from) {
  default:
    data = cl->clAPI.extSec(cl, bin, &data_size, d_section[start], &error_code);
    break;
  case ACL_TYPE_DEFAULT: {
    aclSections sections[] = {aclSOURCE, aclSPIR, aclLLVMIR, aclCODEGEN, aclTEXT};
    uint8_t table[] = {0, 1, 1, 4, 5};
    aclType type[] = {ACL_TYPE_SOURCE, ACL_TYPE_SPIR_BINARY, ACL_TYPE_LLVMIR_BINARY, ACL_TYPE_CG, ACL_TYPE_ISA};
    for (int y = 0, x = sizeof(sections) / sizeof(sections[0]) - 1; x >= y; --x) {
      data = (const char*)cl->clAPI.extSec(cl, bin, &data_size, sections[x], &error_code);
      if (data && data_size > 0 && error_code == ACL_SUCCESS) {
        start = table[x];
        from = type[x];
        break;
      }
    }
    break;
  }
  case ACL_TYPE_SPIRV_BINARY:
    data = cl->clAPI.extSec(cl, bin, &data_size, aclSPIRV, &error_code);
    break;
  case ACL_TYPE_SPIR_BINARY:
  case ACL_TYPE_SPIR_TEXT:
    data = cl->clAPI.extSec(cl, bin, &data_size, aclSPIR, &error_code);
    break;
  case ACL_TYPE_RSLLVMIR_BINARY:
    data = cl->clAPI.extSec(cl, bin, &data_size, aclLLVMIR, &error_code);
    break;
  case ACL_TYPE_HSAIL_BINARY:
    data = cl->clAPI.extSec(cl, bin, &data_size, aclSOURCE, &error_code);
    // if for ACL_TYPE_HSAIL_BINARY stage BRIG (data) is not presented in aclSOURCE (.source) section of BIF,
    // then it should be in multiple corresponding .brig_ sections in BIF, so continue to compile (data might be NULL)
    if (error_code == ACL_ELF_ERROR) {
      error_code = ACL_SUCCESS;
    }
    break;
  case ACL_TYPE_CG:
    // there is no data for codegen phase (data might be NULL),
    // BRIG should be in its multiple corresponding .brig_ sections in BIF
    if (isHSAILTarget(bin->target)) {
      from = ACL_TYPE_CG;
    } else {
      data = cl->clAPI.extSec(cl, bin, &data_size, d_section[start], &error_code);
    }
    break;
  }
  if (error_code != ACL_SUCCESS) {
    return error_code;
  }
  // Based on our compiler options, we need to change the functors to use
  // the correct pointers unless they are custom loaded, then we should
  // not modify them. This code is ugly and needs to be designed better.
  if (start == 0) {
    if (from == ACL_TYPE_OPENCL || from == ACL_TYPE_SOURCE || from == ACL_TYPE_DEFAULT) {
      const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclCompilerOptions);
      assert(sym && "symbol not found");
      assert(sym->sections[0] == aclCOMMENT && sym->sections[0] == sym->sections[1] &&
             "not in comment section");
      std::string optSec = std::string(sym->str[PRE]) + std::string(sym->str[POST]);
      saveOptionsToComments(cl, bin, options, optSec);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &SPIRInit, &OCLInit);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &AMDILInit, &OCLInit);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &HSAILFEInit, &OCLInit);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &SPIRFini, &OCLFini);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &AMDILFini, &OCLFini);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &HSAILFEFini, &OCLFini);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.toISA, &AMDILFEToISA, NULL);
      CONDITIONAL_CMP_ASSIGN(cl->feAPI.toISA, &HSAILFEToISA, NULL);
      if (to == ACL_TYPE_LLVMIR_BINARY || to == ACL_TYPE_LLVMIR_TEXT) {
        cl->feAPI.toISA = NULL;
        cl->feAPI.toIR = &OCLFEToLLVMIR;
      } else if(to == ACL_TYPE_SPIR_BINARY || to == ACL_TYPE_SPIR_TEXT) {
        cl->feAPI.toISA = NULL;
        cl->feAPI.toIR = &OCLFEToSPIR;
      }
    } else if (from == ACL_TYPE_AMDIL_TEXT || from == ACL_TYPE_HSAIL_TEXT) {
      const oclBIFSymbolStruct* sym = findBIF30SymStruct(symAMDILCompilerOptions);
      assert(sym && "symbol not found");
      assert(sym->sections[0] == aclCOMMENT && "not in comment section");
      amd::option::Options* Opts = reinterpret_cast<amd::option::Options*>(bin->options);
      const char *kernel = Opts->oVariables->Kernel;
      std::string optSec = std::string(sym->str[PRE]) +
                           std::string((!kernel) ? "main" : kernel) +
                           std::string(sym->str[POST]);
      saveOptionsToComments(cl, bin, options, optSec);
      if (to == ACL_TYPE_ISA || to == ACL_TYPE_DEFAULT) {
        stop = 1;
        if (from == ACL_TYPE_AMDIL_TEXT) {
          cl->feAPI.init = &AMDILInit;
          cl->feAPI.fini = &AMDILFini;
          cl->feAPI.toISA = &AMDILFEToISA;
        } else {
          CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &OCLInit, &HSAILFEInit);
          CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &OCLFini, &HSAILFEFini);
          CONDITIONAL_CMP_ASSIGN(cl->feAPI.toISA, &OCLFEToISA, &HSAILFEToISA);
        }
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
                 from == ACL_TYPE_SPIR_BINARY   || from == ACL_TYPE_SPIR_TEXT   ||
                 from == ACL_TYPE_RSLLVMIR_BINARY || from == ACL_TYPE_SPIRV_BINARY) {
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &SPIRInit, &OCLInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &AMDILInit, &OCLInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.init, &HSAILFEInit, &OCLInit);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &SPIRFini, &OCLFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &AMDILFini, &OCLFini);
        CONDITIONAL_CMP_ASSIGN(cl->feAPI.fini, &HSAILFEFini, &OCLFini);
        if (from == ACL_TYPE_SPIRV_BINARY) {
          if (to != ACL_TYPE_LLVMIR_BINARY)
            cl->feAPI.toModule = &SPIRVToModule;
          else {
            cl->feAPI.toISA = NULL;
            cl->feAPI.toIR = &SPIRVToModule;
            start = 0;
            stop = 1;
          }
        } else if (from == ACL_TYPE_RSLLVMIR_BINARY) {
          cl->feAPI.toModule = &RSLLVMIRToModule;
        } else {
          cl->feAPI.toModule = &OCLFEToModule;
        }
      }
  }
  if (start > stop) {
    return ACL_INVALID_ARG;
  }
  if (start == stop) {
    return ACL_SUCCESS;
  }
  bool stages[5] = {false};
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
       if (ptr == NULL) {
         error_code = ACL_INVALID_FILE;
         goto internal_link_failure;
       }
       char *mod = new char[data_size];
       memcpy(mod, ptr, data_size);
       module = cl->feAPI.toModule(ald, mod, data_size, context, &error_code);
       for (unsigned x = 0; x < num_libs; ++x) {
         const void *ptr = cl->clAPI.extSec(cl, libs[x], &data_size, aclLLVMIR, NULL);
         if (ptr == NULL)
           ptr = cl->clAPI.extSec(cl, libs[x], &data_size, aclSPIR, NULL);
         if (ptr == NULL) {
           error_code = ACL_INVALID_FILE;
           goto internal_link_failure;
         }
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
        llvm::SmallVector<char, 4096> array;
        llvm::raw_svector_ostream outstream(array);
        llvm::WriteBitcodeToFile(reinterpret_cast<llvm::Module*>(dst_module), outstream);
        cl->clAPI.remSec(cl, src_bin, aclLLVMIR);
        error_code = cl->clAPI.insSec(cl, src_bin,
            &array[0], array.size(), aclLLVMIR);
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
      symbol = findBIF30SymStruct(symBRIG);
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
    {
      to = ACL_TYPE_HSAIL_BINARY;
      const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symHSAILText);
      assert(symbol && "symbol not found");
      std::string symbolName = symbol->str[PRE] + std::string("main") + symbol->str[POST];
      from_data = cl->clAPI.extSym(cl, bin, &from_data_size,
                                   symbol->sections[0],
                                   symbolName.c_str(), &error_code);
      // HSAIL was inserted into bif as section only without corresponding symbol
      if (!from_data) {
        from_data = cl->clAPI.extSec(cl, bin, &from_data_size,
                                     symbol->sections[0], &error_code);
      }
      // HSAIL is in aclSOURCE section (might be used while compiling from HSAIL by -hsail option)
      if (!from_data) {
        from_data = cl->clAPI.extSec(cl, bin, &from_data_size, aclSOURCE, &error_code);
      }
      break;
    }
    case ACL_TYPE_HSAIL_BINARY:
    {
#if defined(WITH_TARGET_HSAIL)
      // BRIG to HSAIL disassembling
      if (isHSAILTarget(bin->target)) {
        amdcl::HSAIL *acl = new amdcl::HSAIL(cl, bin, NULL);
        if (acl == NULL)  {
          return ACL_OUT_OF_MEM;
        }
        std::string hsail = acl->disassembleBRIG();
        // If HSAIL was not disassembled from multiple .brig_ sections in BIF, then:
        // 1. try to extract BRIG from aclSOURCE section
        if (hsail.empty()) {
          from_data = cl->clAPI.extSec(cl, bin, &from_data_size, aclSOURCE, &error_code);
          HSAIL_ASM::BrigContainer c;
          // 2. load BRIG in BrigContainer
          int result = HSAIL_ASM::BrigStreamer::load(c,
              reinterpret_cast<const char*>(from_data), from_data_size);
          if (result != 0) {
            error_code = ACL_INVALID_BINARY;
            delete acl;
            return error_code;
          }
          // 3. insert BRIG into multiple .brig_ sections in BIF +
          // insert matadata symbols for every kernel
          if (!acl->insertBRIG(c)) {
            assert(!"Inserting BRIG failed\n");
            error_code = ACL_INVALID_BINARY;
            delete acl;
            return error_code;
          }
          // 4. second attempt to disassemble BRIG
          hsail = acl->disassembleBRIG();
        }
        delete acl;
        if (hsail.empty()) {
            return ACL_ELF_ERROR;
        }
        const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symHSAILText);
        assert(symbol && "symbol not found");
        std::string symbolName = symbol->str[PRE] + std::string("main") +
                            symbol->str[POST];
        return cl->clAPI.insSym(cl, bin, hsail.data(), hsail.size(),
                                symbol->sections[0], symbolName.c_str());
      } else {
        assert(!"Unsupported architecture, expect hsail.");
        return ACL_SYS_ERROR;
      }
#else
      assert(!"Cannot go down this path without HSAIL support!");
      return ACL_SYS_ERROR;
#endif
      break;
    }
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
    case ACL_TYPE_HSAIL_BINARY:
      {
#if defined(WITH_TARGET_HSAIL)
        if (isHSAILTarget(bin->target)) {
          amdcl::HSAIL *acl = new amdcl::HSAIL(cl, bin, NULL);
          if (acl == NULL)  {
            return ACL_OUT_OF_MEM;
          }
          // while assembling BRIG insertion into BIF (bin) performs,
          // so no need in any symbol/section insertion here
          bool bRet = acl->assemble(std::string(reinterpret_cast<const char*>(from_data)));
          delete acl;
          if (!bRet) {
            return ACL_CODEGEN_ERROR;
          }
          return ACL_SUCCESS;
        } else {
          assert(!"Unsupported architecture, expect hsail.");
          return ACL_SYS_ERROR;
        }
#else
        assert(!"Cannot go down this path without HSAIL support!");
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
      hsail_be->disassembleBRIG();
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
#ifdef WITH_TARGET_HSAIL
  if (isHSAILTarget(bin->target)) {
    return cl->clAPI.extSec(cl, bin, size, aclTEXT, error_code);
  } else
#endif
  {
    const oclBIFSymbolStruct* sym = findBIF30SymStruct(symISABinary);
    assert(sym && "symbol not found");
    std::string name = sym->str[PRE] + std::string(kernel) + sym->str[POST];
    return cl->clAPI.extSym(cl, bin, size, sym->sections[0], name.c_str(), error_code);
  }
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

  // de-serialize the vec type hint
  md->vth = tmp_ptr;
  tmp_ptr += md->vecTypeHintSize + 1;

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
  if (!size) {
    return ACL_ERROR;
  }
  bifbase *elfBin = reinterpret_cast<bifbase*>(binary->bin);
  if (!elfBin) {
    return ACL_ELF_ERROR;
  }
  const oclBIFSymbolStruct* sym = findBIF30SymStruct(symOpenclMeta);
  assert(sym && "symbol not found");
  aclSections secID = sym->sections[0];
  std::string pre = std::string(sym->str[PRE]);
  std::string post = std::string(sym->str[POST]);
  switch (query) {
    default:
      break;
    case RT_CONTAINS_LLVMIR:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        bool contains = elfBin->isSection(aclLLVMIR);
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_CONTAINS_SPIR:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        bool contains = elfBin->isSection(aclSPIR);
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_CONTAINS_SPIRV:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        bool contains = elfBin->isSection(aclSPIRV);
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_CONTAINS_OPTIONS:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        bool contains = elfBin->isSection(aclCOMMENT);
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_CONTAINS_HSAIL:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        const oclBIFSymbolStruct* sym = findBIF30SymStruct(symHSAILText);
        assert(sym && "symbol not found");
        std::string symbolName = sym->str[PRE] + std::string("main") + sym->str[POST];
        bool contains = elfBin->isSymbol(aclCODEGEN, symbolName.c_str());
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_CONTAINS_BRIG:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        bool contains = elfBin->isSection(aclBRIG);
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_CONTAINS_LOADER_MAP:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        const oclBIFSymbolStruct* sym = findBIF30SymStruct(symBRIGLoaderMap);
        assert(sym && "symbol not found");
        std::string symbolName = sym->str[PRE];
        bool contains = elfBin->isSymbol(aclCODEGEN, symbolName.c_str());
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_CONTAINS_ISA:
      if (!ptr) {
        *size = sizeof(bool);
        return ACL_SUCCESS;
      } else if (*size >= sizeof(bool)) {
        bool contains = elfBin->isSection(aclTEXT);
        memcpy(ptr, &contains, sizeof(bool));
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    case RT_KERNEL_NAMES:{
      bifbase::SymbolVector symbols, kernels;
      elfBin->getSectionSymbols(secID, symbols);
      size_t totSize = 0;
      if (!symbols.empty()) {
        std::size_t beg = 0, begKernel = 0, end = 0, endKernel = 0, endSize = 0;
        const oclBIFSymbolStruct* symKernel = findBIF30SymStruct(symOpenclKernel);
        assert(symKernel && "symbol not found");
        std::string preKernel = std::string(symKernel->str[PRE]);
        std::string postKernel = std::string(symKernel->str[POST]);
        for (bifbase::SymbolVector::iterator it = symbols.begin(); it != symbols.end(); ++it) {
          beg = (*it).find(pre);
          if (std::string::npos == beg) continue;
          beg += pre.size();
          begKernel = (*it).find(preKernel, beg);
          if (std::string::npos != begKernel) {
            beg = begKernel + preKernel.size();
            end = (*it).rfind(postKernel);
            endSize = postKernel.size();
          } else {
            end = (*it).rfind(post);
          }
          if (std::string::npos == end) continue;
          endSize += post.size();
          if (end <= beg || end != (*it).size() - endSize) continue;
          std::string kernel((*it).substr(beg, (*it).size() - beg - endSize) + " ");
          totSize += kernel.size();
          kernels.push_back(kernel);
        }
      }
      if (!ptr) {
        *size = totSize > 0 ? totSize + 1 : 0;
        return ACL_SUCCESS;
      } else if (*size >= totSize && totSize > 0) {
        char* tmp = reinterpret_cast<char*>(ptr);
        for (bifbase::SymbolVector::iterator it = kernels.begin(); it != kernels.end(); ++it) {
          memcpy(tmp, (*it).c_str(), (*it).size());
          tmp += (*it).size();
        }
        *(tmp++) = '\0';
        return ACL_SUCCESS;
      }
      return ACL_ERROR;
    }
  }
  size_t roSize;
  acl_error error_code;
  if (!kernel) {
    return ACL_INVALID_ARG;
  }
  std::string symbol = pre + std::string(kernel) + post;
  const void* roSec = cl->clAPI.extSym(cl, binary, &roSize, secID, symbol.c_str(), &error_code);
  if (error_code != ACL_SUCCESS) return error_code;
  if (roSec == NULL || roSize == 0) {
    return ACL_ELF_ERROR;
  }
  const aclMetadata *md = reinterpret_cast<const aclMetadata*>(roSec);
  bool success = false;
  switch (query) {
    default: break;
    case RT_CPU_BARRIER_NAMES:
            if (!ptr) {
              *size = 0;
              success = true;
            } else {
              assert(!"Not implemented");
            }
            break;
    case RT_ABI_VERSION: {
            size_t majorSize = sizeof(md->major);
            size_t minorSize = sizeof(md->minor);
            size_t revisionSize = sizeof(md->revision);
            size_t verSize = majorSize + minorSize + revisionSize;
            if (!ptr) {
              *size = verSize;
              success = true;
            } else if (*size >= verSize) {
              char *tmp = reinterpret_cast<char*>(ptr);
              memcpy(tmp, &md->major, majorSize);
              tmp += majorSize;
              memcpy(tmp, &md->minor, minorSize);
              tmp += minorSize;
              memcpy(tmp, &md->revision, revisionSize);
              success = true;
            }
            break;
          }
    case RT_DEVICE_NAME:
            if (!ptr) {
              *size = md->deviceNameSize;
              success = true;
            } else if (*size >= md->deviceNameSize) {
              // deviceName is a pointer, which is serialized by serializeMetadata() to NULL
              // in binary; to get the data deserializeCLMetadata() is needed
              aclMetadata *deserializedMd = static_cast<aclMetadata*>(alloca(roSize));
              deserializeCLMetadata(reinterpret_cast<const char*>(roSec), deserializedMd, roSize);
              if (deserializedMd->deviceName && deserializedMd->deviceNameSize == md->deviceNameSize) {
                strncpy(reinterpret_cast<char*>(ptr), deserializedMd->deviceName, deserializedMd->deviceNameSize);
                success = true;
              }
            }
            break;
    case RT_KERNEL_NAME:
            if (!ptr) {
              *size = md->kernelNameSize;
              success = true;
            } else if (*size >= md->kernelNameSize) {
              // kernelName is a pointer, which is serialized by serializeMetadata() to NULL
              // in binary; to get the data deserializeCLMetadata() is needed
              aclMetadata *deserializedMd = static_cast<aclMetadata*>(alloca(roSize));
              deserializeCLMetadata(reinterpret_cast<const char*>(roSec), deserializedMd, roSize);
              if (deserializedMd->kernelName && deserializedMd->kernelNameSize == md->kernelNameSize) {
                strncpy(reinterpret_cast<char*>(ptr), deserializedMd->kernelName, deserializedMd->kernelNameSize);
                success = true;
              }
            }
            break;
    case RT_MEM_SIZES: {
            size_t memSize = sizeof(md->mem);
            if (!ptr) {
              *size = memSize;
              success = true;
            } else if (*size >= memSize) {
              memcpy(ptr, md->mem, memSize);
              success = true;
            }
            break;
          }
    case RT_GPU_FUNC_CAPS: {
            if (binary->target.arch_id == aclX86) {
              break;
            }
            size_t gpuCapsSize = sizeof(md->gpuCaps);
            if (!ptr) {
              *size = gpuCapsSize;
              success = true;
            } else if (*size >= gpuCapsSize) {
              memcpy(ptr, &md->gpuCaps, gpuCapsSize);
              success = true;
            }
            break;
          }
    case RT_GPU_FUNC_ID: {
            if (binary->target.arch_id == aclX86) {
              break;
            }
            size_t funcIDSize = sizeof(md->funcID);
            if (!ptr) {
              *size = funcIDSize;
              success = true;
            } else if (*size >= funcIDSize) {
              memcpy(ptr, &md->funcID, funcIDSize);
              success = true;
            }
            break;
          }
    case RT_GPU_DEFAULT_ID: {
            if (binary->target.arch_id == aclX86) {
              break;
            }
            size_t gpuResSize = sizeof(md->gpuRes);
            if (!ptr) {
              *size = gpuResSize;
              success = true;
            } else if (*size >= gpuResSize) {
              memcpy(ptr, &md->gpuRes, gpuResSize);
              success = true;
            }
            break;
          }
    case RT_WORK_GROUP_SIZE: {
            size_t wgsSize = sizeof(md->wgs);
            if (!ptr) {
              *size = wgsSize;
              success = true;
            } else if (md->wgs && *size >= wgsSize) {
              memcpy(ptr, md->wgs, wgsSize);
              success = true;
            }
            break;
          }
    case RT_WORK_REGION_SIZE: {
            size_t wrsSize = sizeof(md->wrs);
            if (!ptr) {
              *size = wrsSize;
              success = true;
            } else if (md->wrs && *size >= wrsSize) {
              memcpy(ptr, md->wrs, wrsSize);
              success = true;
            }
            break;
          }
    case RT_ARGUMENT_ARRAY: {
            // args is a pointer, which is serialized by serializeMetadata() to NULL
            // in binary; to get the data deserializeCLMetadata() is needed
            aclMetadata *deserializedMd = static_cast<aclMetadata*>(alloca(roSize));
            deserializeCLMetadata(reinterpret_cast<const char*>(roSec), deserializedMd, roSize);
            size_t totSize = 0;
            if (deserializedMd->numArgs > 0) {
              // 1 additional elemet is the array's end marker,
              // which points to the structure with struct_size == 0
              totSize = sizeof(aclArgData) * (deserializedMd->numArgs + 1);
              for (unsigned x = 0; x < deserializedMd->numArgs; ++x) {
                totSize += deserializedMd->args[x].typeStrSize + deserializedMd->args[x].argNameSize + 2;
              }
            }
            if (!ptr) {
              *size = totSize;
              success = true;
            } else if (*size >= totSize) {
              char *tmp = reinterpret_cast<char*>(ptr);
              size_t sizeToCopy = sizeof(aclArgData) * (deserializedMd->numArgs + 1);
              memcpy(ptr, deserializedMd->args, sizeToCopy);
              // shift pointer at the end of the POD struct aclArgData
              tmp += sizeToCopy;
              for (unsigned x = 0; x < deserializedMd->numArgs; ++x) {
                sizeToCopy = deserializedMd->args[x].argNameSize;
                // copying argStr data
                memcpy(tmp, deserializedMd->args[x].argStr, sizeToCopy);
                // copying pointer to argStr data
                reinterpret_cast<aclArgData*>(ptr)[x].argStr = tmp;
                tmp += sizeToCopy;
                *(tmp++) = '\0';
                sizeToCopy = deserializedMd->args[x].typeStrSize;
                // copying typeStr data
                memcpy(tmp, deserializedMd->args[x].typeStr, sizeToCopy);
                // copying pointer to typeStr data
                reinterpret_cast<aclArgData*>(ptr)[x].typeStr = tmp;
                tmp += sizeToCopy;
                *(tmp++) = '\0';
                success = true;
              }
            }
            break;
          }
    case RT_GPU_PRINTF_ARRAY: {
            // Printf is a pointer, which is serialized by serializeMetadata() to NULL
            // in binary; to get the data deserializeCLMetadata() is needed
            aclMetadata *deserializedMd = static_cast<aclMetadata*>(alloca(roSize));
            deserializeCLMetadata(reinterpret_cast<const char*>(roSec), deserializedMd, roSize);
            size_t totSize = 0;
            if (deserializedMd->numPrintf > 0) {
              // 1 additional elemet is the array's end marker,
              // which points to the structure with struct_size == 0
              totSize = sizeof(aclPrintfFmt) * (deserializedMd->numPrintf + 1);
              for (unsigned x = 0; x < deserializedMd->numPrintf; ++x) {
                totSize += sizeof(*aclPrintfFmt().argSizes) * deserializedMd->printf[x].numSizes;
                totSize += deserializedMd->printf[x].fmtStrSize + 1;
              }
            }
            if (!ptr) {
              *size = totSize;
              success = true;
            } else if (*size >= totSize) {
              char *tmp = reinterpret_cast<char*>(ptr);
              size_t sizeToCopy = sizeof(aclPrintfFmt) * (deserializedMd->numPrintf + 1);
              memcpy(ptr, deserializedMd->printf, sizeToCopy);
              // shift pointer at the end of the POD struct aclPrintfFmt
              tmp += sizeToCopy;
              for (unsigned x = 0; x < deserializedMd->numPrintf; ++x) {
                sizeToCopy = sizeof(*aclPrintfFmt().argSizes) * deserializedMd->printf[x].numSizes;
                // copying argSizes data
                memcpy(tmp, deserializedMd->printf[x].argSizes, sizeToCopy);
                // copying pointer to argSizes data
                memcpy(&reinterpret_cast<aclPrintfFmt*>(ptr)[x].argSizes, &tmp, sizeof(void*));
                tmp += sizeToCopy;
                sizeToCopy = deserializedMd->printf[x].fmtStrSize;
                // copying fmtStr data
                memcpy(tmp, deserializedMd->printf[x].fmtStr, sizeToCopy);
                // copying pointer to fmtStr data
                reinterpret_cast<aclPrintfFmt*>(ptr)[x].fmtStr = tmp;
                tmp += sizeToCopy;
                *(tmp++) = '\0';
              }
              success = true;
            }
            break;
          }
    case RT_DEVICE_ENQUEUE: {
            size_t enqueue_kernelSize = sizeof(md->enqueue_kernel);
            if (!ptr) {
              *size = enqueue_kernelSize;
              success = true;
            } else if (*size >= enqueue_kernelSize) {
              memcpy(ptr, &md->enqueue_kernel, enqueue_kernelSize);
              success = true;
            }
            break;
          }
    // Temporary approach till the "ldk" instruction is supported.
    case RT_KERNEL_INDEX: {
            size_t kernel_indexSize = sizeof(md->kernel_index);
            if (!ptr) {
              *size = kernel_indexSize;
              success = true;
            } else if (*size >= kernel_indexSize) {
              memcpy(ptr, &md->kernel_index, kernel_indexSize);
              success = true;
            }
            break;
          }
    case RT_NUM_KERNEL_HIDDEN_ARGS: {
            size_t hidden_kernargs_size = sizeof(md->numHiddenKernelArgs);
            if (!ptr) {
              *size = hidden_kernargs_size;
              success = true;
            } else if (*size >= hidden_kernargs_size) {
              memcpy(ptr, &md->numHiddenKernelArgs, hidden_kernargs_size);
              success = true;
            }
            break;
          }
    case RT_WAVES_PER_SIMD_HINT: {
            size_t waves_per_simd_hint_size = sizeof(md->wavesPerSimdHint);
            if (!ptr) {
              *size = waves_per_simd_hint_size;
              success = true;
            } else if (*size >= waves_per_simd_hint_size) {
              memcpy(ptr, &md->wavesPerSimdHint, waves_per_simd_hint_size);
              success = true;
            }
            break;
          }
    case RT_WORK_GROUP_SIZE_HINT: {
            size_t work_group_size_hint_size = sizeof(md->wsh);
            if (!ptr) {
              *size = work_group_size_hint_size;
              success = true;
            } else if (*size >= work_group_size_hint_size) {
              memcpy(ptr, md->wsh, work_group_size_hint_size);
              success = true;
            }
            break;
          }
    case RT_VEC_TYPE_HINT: {
            if (!ptr) {
              *size = md->vecTypeHintSize;
              success = true;
            } else if (*size >= md->vecTypeHintSize) {
              // vecTypeHint is a pointer, which is serialized by serializeMetadata() to NULL
              // in binary; to get the data deserializeCLMetadata() is needed
              aclMetadata *deserializedMd = static_cast<aclMetadata*>(alloca(roSize));
              deserializeCLMetadata(reinterpret_cast<const char*>(roSec), deserializedMd, roSize);
              if (deserializedMd->vth && deserializedMd->vecTypeHintSize == md->vecTypeHintSize) {
                strncpy(reinterpret_cast<char*>(ptr), deserializedMd->vth, deserializedMd->vecTypeHintSize);
                success = true;
              }
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
      + (md->deviceNameSize + 1)
      + (md->vecTypeHintSize + 1));
  tmp_ptr += md->struct_size;
  tmp_ptr += md->kernelNameSize + 1;
  tmp_ptr[-1] = '\0';
  tmp_ptr += md->deviceNameSize + 1;
  tmp_ptr[-1] = '\0';
  tmp_ptr += md->vecTypeHintSize + 1;
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
  ro_ptr += md->vecTypeHintSize + 1;
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
      + (md->deviceNameSize + 1)
      + (md->vecTypeHintSize +1));
  tmp_ptr += md->struct_size;
  tmp_ptr += md->kernelNameSize + 1;
  tmp_ptr[-1] = '\0';
  tmp_ptr += md->deviceNameSize + 1;
  tmp_ptr[-1] = '\0';
  tmp_ptr += md->vecTypeHintSize + 1;
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

#if defined(LEGACY_COMPLIB)
static OCLMCJITMemoryManager* memMgr = NULL;

OCLMCJITMemoryManager* createJITMemoryManager() {
  if (!memMgr) {
    memMgr = new OCLMCJITMemoryManager();
  }
  return memMgr;
}
#else
typedef llvm::DenseMap<llvm::object::ObjectFile*, OCLMCJITMemoryManager*> MemMgrTableT;
typedef llvm::DenseMap<llvm::object::ObjectFile*, llvm::RuntimeDyld*>     DyLdTableT;
static MemMgrTableT MemMgrTable;
static DyLdTableT   DyLdTable;

static llvm::RuntimeDyld* GetOrCreateDyld(llvm::object::ObjectFile* obj) {
  DyLdTableT::iterator DI = DyLdTable.find(obj);
  if (DI != DyLdTable.end())
    return DI->second;
  OCLMCJITMemoryManager *memMgr = new OCLMCJITMemoryManager();
  MemMgrTable.insert(std::make_pair(obj, memMgr));
  llvm::RuntimeDyld *rtdyld = new llvm::RuntimeDyld(*memMgr, *memMgr);
  DyLdTable.insert(std::make_pair(obj, rtdyld));
  return rtdyld;
}

static void ReleaseDyld(llvm::object::ObjectFile* obj) {
  DyLdTableT::iterator DI = DyLdTable.find(obj);
  if (DI != DyLdTable.end()) {
    delete DI->second;
    DyLdTable.erase(DI);
  }
  MemMgrTableT::iterator MI = MemMgrTable.find(obj);
  if (MI != MemMgrTable.end()) {
    delete MI->second;
    MemMgrTable.erase(MI);
  }
}
#endif

aclJITObjectImage ACL_API_ENTRY
if_aclJITObjectImageCreate(const void* buffer, size_t length,
                           aclBinary* bin, acl_error* error_code) {
  llvm::StringRef dataString((const char*)buffer, length);
#if defined(LEGACY_COMPLIB)
  llvm::MemoryBuffer* memBuf = llvm::MemoryBuffer::getMemBufferCopy(dataString);
  llvm::ObjectBuffer* objBuf = new llvm::ObjectBuffer(memBuf);
  llvm::RuntimeDyld rtdyld(createJITMemoryManager());
  llvm::ObjectImage* objectImage = rtdyld.loadObject(objBuf);
  rtdyld.resolveRelocations();
  amd::option::Options* options = reinterpret_cast<amd::option::Options*>(bin->options);
  if (options && options->isDumpFlagSet(amd::option::DUMP_O)) {
    llvm::StringRef finalData = objectImage->getData();
    std::string finalDataString = finalData.str();
    std::string objname = options->getDumpFileName(".elf");
    std::ofstream out(objname.c_str(), std::fstream::binary | std::fstream::trunc);
    out << finalDataString;
    out.close();
  }
  return objectImage;
#else
  std::unique_ptr<llvm::MemoryBuffer> memBuf = llvm::MemoryBuffer::getMemBufferCopy(dataString);
  llvm::ErrorOr<std::unique_ptr<llvm::object::ObjectFile>> objBuf =
    llvm::object::ObjectFile::createObjectFile(memBuf->getMemBufferRef());
  llvm::RuntimeDyld *rtdyld = GetOrCreateDyld(objBuf->get());

  auto objectImage = rtdyld->loadObject(*(objBuf.get()));
  rtdyld->resolveRelocations();

  amd::option::Options* options =   (amd::option::Options*)bin->options;
  if (options->isDumpFlagSet(amd::option::DUMP_O)) {
    llvm::StringRef finalData = objBuf.get()->getData();
    std::string finalDataString = finalData.str();
    std::string objname = options->getDumpFileName(".elf");
    std::ofstream out(objname.c_str(),
                      (std::fstream::binary | std::fstream::trunc));
    out << finalDataString;
    out.close();
  }

  memBuf.release();
  llvm::object::ObjectFile* result = objBuf.get().release();

  return result;
#endif
}

aclJITObjectImage ACL_API_ENTRY
if_aclJITObjectImageCopy(const void* buffer, size_t length, acl_error* error_code) {
  llvm::StringRef dataString((const char*)buffer, length);
#if defined(LEGACY_COMPLIB)
  llvm::MemoryBuffer* memBuf = llvm::MemoryBuffer::getMemBufferCopy(dataString);
  llvm::ObjectBuffer* objBuf = new llvm::ObjectBuffer(memBuf);
  llvm::RuntimeDyld rtdyld(createJITMemoryManager());
  llvm::ObjectImage* objectImage = rtdyld.loadObject(objBuf);
  rtdyld.resolveRelocations();
  return objectImage;
#else
  std::unique_ptr<llvm::MemoryBuffer> memBuf = llvm::MemoryBuffer::getMemBufferCopy(dataString);
  auto objBuf = llvm::object::ObjectFile::createObjectFile(memBuf->getMemBufferRef());
  llvm::RuntimeDyld *rtdyld = GetOrCreateDyld(objBuf->get());
  auto objectImage = rtdyld->loadObject(*(objBuf.get()));
  rtdyld->resolveRelocations();
  memBuf.release();
  llvm::object::ObjectFile* result = objBuf.get().release();

  return result;
#endif
}

acl_error ACL_API_ENTRY
if_aclJITObjectImageDestroy(aclJITObjectImage image) {
#if defined(LEGACY_COMPLIB)
  llvm::ObjectImage* objectImage(reinterpret_cast<llvm::ObjectImage*>(image));
  llvm::object::section_iterator end = objectImage->end_sections();
  llvm::error_code err;
  for (llvm::object::section_iterator iter = objectImage->begin_sections();
       iter != end; iter.increment(err)) {
    llvm::object::SectionRef sectionRef = *iter;
    uint64_t address;
    sectionRef.getAddress(address);
    memMgr->deallocateSection((uint8_t*)address);
  }
#else
  llvm::object::ObjectFile* objectImage(reinterpret_cast<llvm::object::ObjectFile*>(image));
  ReleaseDyld(objectImage);
#endif
  delete objectImage;
  return ACL_SUCCESS;
}

size_t ACL_API_ENTRY
if_aclJITObjectImageSize(aclJITObjectImage image, acl_error* error_code) {
#if defined(LEGACY_COMPLIB)
  return (reinterpret_cast<llvm::ObjectImage*>(image))->getData().size();
#else
  return (reinterpret_cast<llvm::object::ObjectFile*>(image))->getData().size();
#endif
}

const char* ACL_API_ENTRY
if_aclJITObjectImageData(aclJITObjectImage image, acl_error* error_code) {
#if defined(LEGACY_COMPLIB)
  return (reinterpret_cast<llvm::ObjectImage*>(image))->getData().data();
#else
  return (reinterpret_cast<llvm::object::ObjectFile*>(image))->getData().data();
#endif
}

acl_error ACL_API_ENTRY
if_aclJITObjectImageFinalize(aclJITObjectImage image) {
  return ACL_SUCCESS;
}

size_t ACL_API_ENTRY
if_aclJITObjectImageGetGlobalsSize(aclJITObjectImage image, acl_error* error_code) {
  size_t totalSize = 0;
#if defined(LEGACY_COMPLIB)
  llvm::ObjectImage* objectImage(reinterpret_cast<llvm::ObjectImage*>(image));
  llvm::object::section_iterator end = objectImage->end_sections();
  llvm::error_code err;
  for (llvm::object::section_iterator iter = objectImage->begin_sections();
       iter != end; iter.increment(err)) {
    llvm::object::SectionRef sectionRef = *iter;
    llvm::StringRef name;
    uint64_t size;
    bool isBSS, isData, isText;
    sectionRef.getName(name);
    sectionRef.getSize(size);
    sectionRef.isBSS(isBSS);
    sectionRef.isData(isData);
    sectionRef.isText(isText);
    if ((isBSS || isData) && !isText) {
      totalSize += (size_t)size;
    }
  }
#else
  llvm::object::ObjectFile* objectImage(reinterpret_cast<llvm::object::ObjectFile*>(image));
  for (auto iter: objectImage->sections()) {
    uint64_t size = iter.getSize();
    if ((iter.isBSS() || iter.isData()) && !iter.isText()) {
      totalSize += (size_t)iter.getSize();
    }
  }
#endif
  return totalSize;
}

acl_error ACL_API_ENTRY
if_aclJITObjectImageIterateSymbols(aclJITObjectImage image,
                                   JITSymbolCallback jit_callback, void* data) {
#if defined(LEGACY_COMPLIB)
  llvm::ObjectImage* objectImage(reinterpret_cast<llvm::ObjectImage*>(image));
  llvm::object::symbol_iterator end = objectImage->end_symbols();
  llvm::StringRef name;
  uint64_t address;
  llvm::error_code err;
  for (llvm::object::symbol_iterator iter = objectImage->begin_symbols();
       iter != end; iter.increment(err)) {
    llvm::object::SymbolRef symRef = *iter;
    symRef.getName(name);
    symRef.getAddress(address);
    jit_callback(name.str().c_str(), (const void*)address, data);
  }
#else
  llvm::object::ObjectFile* objectImage = reinterpret_cast<llvm::object::ObjectFile*>(image);
  llvm::RuntimeDyld *rtdyld = GetOrCreateDyld(objectImage);
  for (const auto &S: objectImage->symbols()) {
    auto Ret = S.getName();
    if (!Ret) {
      auto InternalSymbol = rtdyld->getSymbol(Ret.get());
      uint64_t address = (uint64_t)(InternalSymbol ? InternalSymbol.getAddress() : 0);
      jit_callback(Ret.get().data(), (const void*)address, data);
    }
  }
#endif
  return ACL_SUCCESS;
}

#if defined(LEGACY_COMPLIB)
#if 0
static std::string getFeaturesString(llvm::StringMap<bool>& Features)
{
  std::string FeatureString;
  llvm::raw_string_ostream FeatureStream(FeatureString);
  llvm::SubtargetFeatures TargetFeatures("");
  llvm::StringMapConstIterator<bool> iterEnd = Features.end();
  for(llvm::StringMapConstIterator<bool> I = Features.begin();
      I != iterEnd; ++I) {
    const llvm::StringMapEntry<bool> entry = *I;
    TargetFeatures.AddFeature(entry.getKey(), entry.getValue());
  }
  TargetFeatures.print(FeatureStream);
  return FeatureString;
}
#endif

static std::string getTripleName()
{
#ifdef _WIN32
  return LP64_SWITCH("i686-pc-mingw32-amdopencl",
                     "x86_64-pc-mingw32-amdopencl");
#else
  return LP64_SWITCH("i686-pc-linux-amdopencl",
                     "x86_64-pc-linux-amdopencl");
#endif
}

static std::string bytesToHexString(const char* data, size_t size) {
  std::stringstream hexstring;
  hexstring << std::hex << std::setfill('0');
  for(size_t i = 0; i < size; ++i) {
    hexstring << "0x" << std::setw(2) << unsigned((unsigned char)data[i])
              << std::endl;
  }
  hexstring << std::endl;
  return hexstring.str();
}

char* ACL_API_ENTRY
if_aclJITObjectImageDisassembleKernel(constAclJITObjectImage image,
                                      const char* kernel, acl_error* error_code) {
  const llvm::ObjectImage* objectImage(reinterpret_cast<const llvm::ObjectImage*>(image));
  llvm::object::symbol_iterator end = objectImage->end_symbols();
  llvm::error_code err;
  llvm::StringRef name;
  std::stringstream disas;
  for (llvm::object::symbol_iterator iter = objectImage->begin_symbols();
       iter != end; iter.increment(err)) {
    llvm::object::SymbolRef symRef = *iter;
    symRef.getName(name);
    std::string kernelStr(kernel);
    if(name == kernelStr) {
      uint64_t start;
      uint64_t size;
      symRef.getSize(size);
      symRef.getAddress(start);
      const char *bytes = (const char *)start;
      const uint64_t extent = 0x10000;
      uint64_t max_pc = 0;

      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllDisassemblers();

      std::string TripleName = getTripleName();
      std::string Error;
      const llvm::Target *TheTarget =
        llvm::TargetRegistry::lookupTarget(TripleName, Error);

      std::string hexstring = bytesToHexString(bytes,  size);
      llvm::StringRef kernelMem(hexstring);
      llvm::MemoryBuffer *Buffer =
        llvm::MemoryBuffer::getMemBuffer(kernelMem, "", false);
      llvm::SourceMgr SrcMgr;

      SrcMgr.AddNewSourceBuffer(Buffer, llvm::SMLoc());

      llvm::OwningPtr<llvm::MCAsmInfo>
        MAI(TheTarget->createMCAsmInfo(TripleName));
      assert(MAI && "Unable to create target asm info!");

      llvm::OwningPtr<llvm::MCRegisterInfo>
        MRI(TheTarget->createMCRegInfo(TripleName));
      assert(MRI && "Unable to create target register info!");

      llvm::OwningPtr<llvm::MCObjectFileInfo>
        MOFI(new llvm::MCObjectFileInfo());
      llvm::MCContext Ctx(*MAI, *MRI, MOFI.get(), &SrcMgr);
      MOFI->InitMCObjectFileInfo(TripleName, llvm::Reloc::Default,
                                 llvm::CodeModel::Default, Ctx);

      Ctx.setAllowTemporaryLabels(true);
      Ctx.setGenDwarfForAssembly(true);

      std::string MCPU = "corei7-avx";
      std::string FeaturesStr;

      std::string DisasResultString;
      llvm::raw_string_ostream OutputStream(DisasResultString);
      OutputStream.SetUnbuffered();
      llvm::formatted_raw_ostream FOS(OutputStream);
      llvm::OwningPtr<llvm::MCStreamer> Str;
      llvm::OwningPtr<llvm::MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
      llvm::OwningPtr<llvm::MCSubtargetInfo>
        STI(TheTarget->createMCSubtargetInfo(TripleName, MCPU, FeaturesStr));
      llvm::MCInstPrinter *IP =
        TheTarget->createMCInstPrinter(0 /* OutputAsmVariant */,  *MAI, *MCII, *MRI,
                                       *STI);
      llvm::MCCodeEmitter *CE = 0;
      llvm::MCAsmBackend *MAB = 0;
      if (false) {
        CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, *STI, Ctx);
        MAB = TheTarget->createMCAsmBackend(TripleName, MCPU);
      }
      Str.reset(TheTarget->createAsmStreamer(Ctx, FOS, /*asmverbose*/true,
                                             /*useLoc*/ true,
                                             /*useCFI*/ true,
                                             /*useDwarfDirectory*/ true,
                                             IP, CE, MAB, false));
      // int Res = llvm::Disassembler::disassemble(*TheTarget,
      //                                           TripleName, *STI, *Str,
      //                                           *Buffer, SrcMgr, OutputStream);

            int Res =
                   llvm::Disassembler::disassembleEnhanced(TripleName, *Buffer, SrcMgr,
                                             OutputStream);
      const char* result = DisasResultString.c_str();
      return strdup(result);
    }
  }
  return NULL;
}
#endif

void myLogFunc(const char * msg, size_t size)
{
  printf("%s\n", msg);
}

#define CONDITIONAL_ASSIGN(A, B) A = (A) ? (A) : (B)
acl_error  ACL_API_ENTRY
if_aclSetupLoaderObject(aclCompiler *cl) {
  /* setup the loader objects here now that we have parsed the
   * options and know the target. */
  CONDITIONAL_ASSIGN(cl->cgAPI.init, &CodegenInit);
  CONDITIONAL_ASSIGN(cl->cgAPI.fini, &CodegenFini);
  CONDITIONAL_ASSIGN(cl->cgAPI.codegen, &CodegenPhase);
  CONDITIONAL_ASSIGN(cl->linkAPI.init, &LinkInit);
  CONDITIONAL_ASSIGN(cl->linkAPI.fini, &LinkFini);
  CONDITIONAL_ASSIGN(cl->linkAPI.link, &OCLLinkPhase);
  CONDITIONAL_ASSIGN(cl->linkAPI.toLLVMIR, &OCLLinkToLLVMIR);
  CONDITIONAL_ASSIGN(cl->linkAPI.toSPIR, &OCLLinkToSPIR);

  CONDITIONAL_ASSIGN(cl->feAPI.init, &OCLInit);
  CONDITIONAL_ASSIGN(cl->feAPI.fini, &OCLFini);
#if !defined(LEGACY_COMPLIB)
  CONDITIONAL_ASSIGN(cl->feAPI.toIR, &OCLFEToSPIR);
#else
  CONDITIONAL_ASSIGN(cl->feAPI.toIR, &OCLFEToLLVMIR);
#endif

  CONDITIONAL_ASSIGN(cl->feAPI.toModule, &OCLFEToModule);
  CONDITIONAL_ASSIGN(cl->feAPI.toISA, &OCLFEToISA);
  CONDITIONAL_ASSIGN(cl->optAPI.init, &OptInit);
  CONDITIONAL_ASSIGN(cl->optAPI.fini, &OptFini);
  CONDITIONAL_ASSIGN(cl->optAPI.optimize, &OptOptimize);
  CONDITIONAL_ASSIGN(cl->beAPI.init, &BEInit);
  CONDITIONAL_ASSIGN(cl->beAPI.fini, &BEFini);
  CONDITIONAL_ASSIGN(cl->beAPI.finalize, &BEAsmPhase);
  CONDITIONAL_ASSIGN(cl->beAPI.assemble, &BEAssemble);
  CONDITIONAL_ASSIGN(cl->beAPI.disassemble, &BEDisassemble);
  return ACL_SUCCESS;
}

#undef CONDITIONAL_ASSIGN

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
  error_code = aclCompile(aoc, aoe, "-save-temps=tmp", ACL_TYPE_RSLLVMIR_BINARY, ACL_TYPE_HSAIL_BINARY, myLogFunc);
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
