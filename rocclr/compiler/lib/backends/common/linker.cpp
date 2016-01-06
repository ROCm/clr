//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
// TODO: The entire linker implementation should be a pass in LLVM and
// the code in the compiler library should only call this pass.

#include "top.hpp"
#include "library.hpp"
#include "linker.hpp"
#include "os/os.hpp"
#include "thread/monitor.hpp"
#include "utils/libUtils.h"
#include "utils/options.hpp"
#include "utils/target_mappings.h"

#include "acl.h"
#if !defined(LEGACY_COMPLIB)
#define HAS_SPIRV
#endif

#if defined(LEGACY_COMPLIB)
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/GlobalValue.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/system_error.h"
#include "llvm/DataLayout.h"
#include "llvm/ValueSymbolTable.h"
#ifdef _DEBUG
#include "llvm/Assembly/Writer.h"
#endif
#else
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#endif

#ifndef LEGACY_COMPLIB
#include "AMDFixupKernelModule.h"
#include "AMDResolveLinker.h"
#include "AMDPrelink.h"
#else
#include "llvm/AMDFixupKernelModule.h"
#include "llvm/AMDResolveLinker.h"
#include "llvm/AMDPrelink.h"
#endif

#include "llvm/AMDUtils.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/AMDLocalArrayUsage.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Bitcode/ReaderWriter.h"

#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/Config/config.h"

#include "llvm/MC/SubtargetFeature.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/AMDLLVMContextHook.h"

#if defined(LEGACY_COMPLIB)
#include "llvm/AMDILFuncSupport.h"
#endif

#ifdef HAS_SPIRV
#include "llvm/Support/SPIRV.h"
#endif

// need to undef DEBUG before using DEBUG macro in llvm/Support/Debug.h
#ifdef DEBUG
#undef DEBUG
#endif
#include "llvm/Support/Debug.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <set>

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "ocl_linker"

#if !defined(LEGACY_COMPLIB)
namespace llvm {
  extern bool HLC_FlushF32Denorms;
}
#endif

namespace AMDSpir {
  extern void replaceTrivialFunc(llvm::Module& M);
}
namespace amd {

namespace {

using namespace llvm;

// LoadFile - Read the specified bitcode file in and return it.  This routine
// searches the link path for the specified file to try to find it...
//
inline llvm::Module*
  LoadFile(const std::string &Filename, LLVMContext& Context)
  {
    if (!sys::fs::exists(Filename)) {
      //    dbgs() << "Bitcode file: '" << Filename.c_str() << "' does not exist.\n";
      return 0;
    }

#if defined(LEGACY_COMPLIB)
    llvm::Module* M;
    std::string ErrorMessage;
    OwningPtr<MemoryBuffer> Buffer;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buffer)) {
      // Error
      M = NULL;
    }
    else {
      M = ParseBitcodeFile(Buffer.get(), Context, &ErrorMessage);
    }

    return M;
#else
    ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr = MemoryBuffer::getFileOrSTDIN(Filename);
    if (!FileOrErr) {
      llvm::ErrorOr<llvm::Module *> M = llvm::parseBitcodeFile(FileOrErr.get()->getMemBufferRef(), Context);
      if (M) return M.get();
    }

    return nullptr;
#endif
  }

#if defined(LEGACY_COMPLIB)
inline llvm::Module*
  LoadLibrary(const std::string& libFile, LLVMContext& Context, MemoryBuffer** Buffer) {
    if (!sys::fs::exists(libFile)) {
      //    dbgs() << "Bitcode file: '" << Filename.c_str() << "' does not exist.\n";
      return 0;
    }

    llvm::Module* M = NULL;
    std::string ErrorMessage;

    static Monitor mapLock;
    static std::map<std::string, void*> FileMap;
    MemoryBuffer* statBuffer;
    {
      ScopedLock sl(mapLock);
      statBuffer = (MemoryBuffer*) FileMap[libFile];
      if (statBuffer == NULL) {
        OwningPtr<MemoryBuffer> PtrBuffer;
        if (error_code ec = MemoryBuffer::getFileOrSTDIN(libFile, PtrBuffer)) {
          // Error
          return NULL;
        }
        else
          statBuffer = PtrBuffer.take();
        M = ParseBitcodeFile(statBuffer, Context, &ErrorMessage);
        FileMap[libFile] = statBuffer;
      }
    }
    *Buffer = MemoryBuffer::getMemBufferCopy(StringRef(statBuffer->getBufferStart(), statBuffer->getBufferSize()), "");
    if ( *Buffer ) {
      M = getLazyBitcodeModule(*Buffer, Context, &ErrorMessage);
      if (!M) {
        delete *Buffer;
        *Buffer = 0;
      }
    }
    return M;
  }
#endif

// Load bitcode libary from an array of const char. This assumes that
// the array has a valid ending zero !
#if defined(LEGACY_COMPLIB)
llvm::Module*
  LoadLibrary(const char* libBC, size_t libBCSize,
      LLVMContext& Context, MemoryBuffer** Buffer)
  {
    llvm::Module* M = 0;
    std::string ErrorMessage;

    *Buffer = MemoryBuffer::getMemBuffer(StringRef(libBC, libBCSize), "");
    if ( *Buffer ) {
      M = getLazyBitcodeModule(*Buffer, Context, &ErrorMessage);
      if (!M) {
        delete *Buffer;
        *Buffer = 0;
      }
    }
    return M;
  }
#else
llvm::Module*
  LoadLibrary(const char* libBC, size_t libBCSize,
      LLVMContext& Context)
  {
    llvm::ErrorOr<llvm::Module*> M(nullptr);
    std::string ErrorMessage;

    auto Buffer = MemoryBuffer::getMemBuffer(StringRef(libBC, libBCSize), "");
    if ( Buffer ) {
      M = llvm::getLazyBitcodeModule(std::move(Buffer), Context);
      if (!M) return nullptr;
    }
    return *M;
  }
#endif

static std::set<std::string> *getAmdRtFunctions()
{
  std::set<std::string> *result = new std::set<std::string>();
  for (size_t i = 0; i < sizeof(amdRTFuns)/sizeof(amdRTFuns[0]); ++i)
    result->insert(amdRTFuns[i]);
  return result;
}

}


} // namespace amd


// Create functions that returns true or false for some features which
// are used by the built-in library
void amdcl::OCLLinker::createASICIDFunctions(llvm::Module* module)
{
  uint64_t features = aclGetChipOptions(Elf()->target);

  llvm::StringRef chip(aclGetChip(Elf()->target));
  llvm::StringRef family(aclGetFamily(Elf()->target));

  createConstIntFunc("__amdil_have_hw_fma32",
                        chip == "Cypress"
                     || chip == "Cayman"
                     || family == "SI"
                     || family == "CI"
                     || family == "KV"
                     || family == "TN"
                     || family == "VI"
                     || family == "CZ"
                     || family == "AI",
                     module);
  createConstIntFunc("__amdil_have_fast_fma32",
                        chip == "Cypress"
                     || chip == "Cayman"
                     || chip == "Tahiti"
                     || chip == "Hawaii"
                     || chip == "Carrizo",
                     module);
  createConstIntFunc("__amdil_have_bitalign", !!(features & F_EG_BASE), module);
  createConstIntFunc("__amdil_is_cypress", chip == "Cypress", module);
  createConstIntFunc("__amdil_is_ni",
                        chip == "Cayman"
                     || family == "TN",
                     module);
  createConstIntFunc("__amdil_is_gcn",
                        family == "SI"
                     || family == "CI"
                     || family == "VI"
                     || family == "KV"
                     || family == "CZ"
                     || family == "AI",
                     module);
}

bool
amdcl::OCLLinker::linkWithModule(
    llvm::Module* Dst, llvm::Module* Src,
    std::map<const llvm::Value*, bool> *ModuleRefMap)
{
#ifndef NDEBUG
  if (Options()->oVariables->EnableDebugLinker) {
      llvm::DebugFlag = true;
      llvm::setCurrentDebugType(DEBUG_TYPE);
  }
#endif
  std::string ErrorMessage;
  if (llvm::linkWithModule(Dst, Src, ModuleRefMap, &ErrorMessage)) {
    DEBUG(llvm::dbgs() << "Error: " << ErrorMessage << "\n");
    BuildLog() += "\nInternal Error: linking libraries failed!\n";
    LogError("linkWithModule(): linking bc libraries failed!");
    return true;
  }
  return false;
}



static void delete_llvm_module(llvm::Module *a)
{
  delete a;
}
  bool
amdcl::OCLLinker::linkLLVMModules(std::vector<llvm::Module*> &libs)
{
  // Load input modules first
  bool Failed = false;
  for (size_t i = 0; i < libs.size(); ++i) {
    std::string ErrorMsg;
    if (!libs[i]) {
      char ErrStr[128];
      sprintf(ErrStr,
          "Error: cannot load input %d bc for linking: %s\n",
          (int)i, ErrorMsg.c_str());
      BuildLog() += ErrStr;
      Failed = true;
      break;
    }

    if (Options()->isDumpFlagSet(amd::option::DUMP_BC_ORIGINAL)) {
      std::string MyErrorInfo;
      char buf[128];
      sprintf(buf, "_original%d.bc", (int)i);
      std::string fileName = Options()->getDumpFileName(buf);
#if defined(LEGACY_COMPLIB)
      llvm::raw_fd_ostream outs(fileName.c_str(), MyErrorInfo,
          llvm::raw_fd_ostream::F_Binary);
      if (MyErrorInfo.empty())
        llvm::WriteBitcodeToFile(libs[i], outs);
      else
        printf(MyErrorInfo.c_str());
#else
      std::error_code EC;
      llvm::raw_fd_ostream outs(fileName.c_str(), EC, llvm::sys::fs::F_None);
      if (!EC)
        llvm::WriteBitcodeToFile(libs[i], outs);
      else
        printf(EC.message().c_str());
#endif
    }
  }

  if (!Failed) {
    // Link input modules together
    for (size_t i = 0; i < libs.size(); ++i) {
      DEBUG(llvm::dbgs() << "LinkWithModule " << i << ":\n");
      if (amdcl::OCLLinker::linkWithModule(LLVMBinary(), libs[i], NULL)) {
        Failed = true;
      }
    }
  }

  if (Failed) {
    delete LLVMBinary();
  }
  std::for_each(libs.begin(), libs.end(), std::ptr_fun(delete_llvm_module));
  libs.clear();
  return Failed;

}

void amdcl::OCLLinker::fixupOldTriple(llvm::Module *module)
{
  llvm::Triple triple(module->getTargetTriple());

  // Bug 9357: "amdopencl" used to be a hacky "OS" that was Linux or Windows
  // depending on the host. It only really matters for x86. If we are trying to
  // use an old binary module still using the old triple, replace it with a new
  // one.
  if (triple.getOSName() == "amdopencl") {
    if (triple.getArch() == llvm::Triple::amdil ||
        triple.getArch() == llvm::Triple::amdil64) {
      triple.setOS(llvm::Triple::UnknownOS);
    } else {
      llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
      triple.setOS(hostTriple.getOS());
    }

    triple.setEnvironment(llvm::Triple::AMDOpenCL);
    module->setTargetTriple(triple.str());
  }
}

// On 64 bit device, aclBinary target is set to 64 bit by default. When 32 bit
// LLVM or SPIR binary is loaded, aclBinary target needs to be modified to
// match LLVM or SPIR bitness.
// Returns false on error.
static bool
checkAndFixAclBinaryTarget(llvm::Module* module, aclBinary* elf,
    std::string& buildLog) {
  if (module->getTargetTriple().empty()) {
    LogWarning("Module has no target triple");
    return true;
  }

  llvm::Triple triple(module->getTargetTriple());
  const char* newArch = NULL;
  if (elf->target.arch_id == aclAMDIL64 &&
     (triple.getArch() == llvm::Triple::amdil ||
     triple.getArch() == llvm::Triple::spir))
       newArch = "amdil";
  else if (elf->target.arch_id == aclX64 &&
      (triple.getArch() == llvm::Triple::x86 ||
      triple.getArch() == llvm::Triple::spir))
      newArch = "x86";
  else if (elf->target.arch_id == aclHSAIL64 &&
      (triple.getArch() == llvm::Triple::hsail ||
      triple.getArch() == llvm::Triple::spir))
      newArch = "hsail";
  if (newArch != NULL) {
    acl_error errorCode;
    elf->target = aclGetTargetInfo(newArch, aclGetChip(elf->target),
        &errorCode);
    if (errorCode != ACL_SUCCESS) {
      assert(0 && "Invalid arch id or chip id in elf target");
      buildLog += "Internal Error: failed to link modules correctlty.\n";
      return false;
    }
  }

  reinterpret_cast<amd::option::Options*>(elf->options)->libraryType_ =
      getLibraryType(&elf->target);

  // Check consistency between module triple and aclBinary target
  if (elf->target.arch_id == aclAMDIL64 &&
      (triple.getArch() == llvm::Triple::amdil64 ||
      triple.getArch() == llvm::Triple::spir64))
    return true;
  if (elf->target.arch_id == aclAMDIL &&
      (triple.getArch() == llvm::Triple::amdil ||
      triple.getArch() == llvm::Triple::spir))
    return true;
  if (elf->target.arch_id == aclHSAIL64 &&
      (triple.getArch() == llvm::Triple::hsail64 ||
      triple.getArch() == llvm::Triple::spir64))
    return true;
  if (elf->target.arch_id == aclHSAIL &&
      (triple.getArch() == llvm::Triple::hsail ||
      triple.getArch() == llvm::Triple::spir))
    return true;
  if (elf->target.arch_id == aclX64 &&
      (triple.getArch() == llvm::Triple::x86_64 ||
      triple.getArch() == llvm::Triple::spir64))
    return true;
  if (elf->target.arch_id == aclX86 &&
      (triple.getArch() == llvm::Triple::x86 ||
      triple.getArch() == llvm::Triple::spir))
    return true;
  DEBUG_WITH_TYPE("linkTriple", llvm::dbgs() <<
      "[checkAndFixAclBinaryTarget] " <<
      " aclBinary target: " << elf->target.arch_id <<
      " chipId: " << elf->target.chip_id <<
      " module triple: " << module->getTargetTriple() <<
      '\n');

  //ToDo: There is bug 9996 in compiler library about converting BIF30 to BIF21
  //which causes regressions in ocltst if the following check is enabled.
  //Fix the bugs then enable the following check
#if 0
  assert(0 && "Inconsistent LLVM target and elf target");
  buildLog += "Internal Error: failed to link modules correctlty.\n";
  return false;
#else
  LogWarning("Inconsistent LLVM target and elf target");
  return true;
#endif
}

#ifdef HAS_SPIRV
bool
translateSpirv(llvm::Module *&M, const std::string &DumpSpirv,
    const std::string &DumpLlvm, bool Timing, std::string &TimeStr){
  uint64_t ReadTime = 0;
  uint64_t WriteTime = 0;
  std::string S;
  llvm::raw_string_ostream RSS(S);
  std::string Err;

  if (Timing)
    WriteTime = amd::Os::timeNanos();
  if (!llvm::WriteSPIRV(M, RSS, Err)) {
    llvm::errs() << "Fails to save LLVM as SPIR-V: " << Err << '\n';
    return false;
  }
  if (Timing)
    WriteTime = amd::Os::timeNanos() - WriteTime;

  if (!DumpSpirv.empty()) {
    std::ofstream OFS(DumpSpirv, std::ios::binary);
    OFS << RSS.str();
    OFS.close();
  }

  RSS.flush();
  std::stringstream SS(S);

  auto &Ctx = M->getContext();
  delete M;
  M = nullptr;
  if (Timing)
    ReadTime = amd::Os::timeNanos();
  if (!llvm::ReadSPIRV(Ctx, SS, M, Err)) {
    llvm::errs() << "Fails to load SPIR-V as LLVM Module: " << Err << '\n';
    return false;
  }

  if (Timing) {
    ReadTime = amd::Os::timeNanos() - ReadTime;
    std::stringstream tmp_ss;
    tmp_ss << "    LLVM/SPIRV translation time: "
           << WriteTime/1000ULL << " us\n"
           << "    SPIRV/LLVM translation time: "
           << ReadTime/1000ULL << " us\n";
    TimeStr = tmp_ss.str();
  }

  if (!DumpLlvm.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream outs(DumpLlvm.c_str(), EC, llvm::sys::fs::F_None);
    if (!EC)
      WriteBitcodeToFile(M, outs);
    else {
      llvm::errs() << EC.message();
      return false;
    }
  }
  return true;
}
#endif

int
amdcl::OCLLinker::link(llvm::Module* input, std::vector<llvm::Module*> &libs)
{
  bool IsGPUTarget = isGpuTarget(Elf()->target);
  uint64_t start_time = 0ULL, time_link = 0ULL, time_prelinkopt = 0ULL;
  if (Options()->oVariables->EnableBuildTiming) {
    start_time = amd::Os::timeNanos();
  }

  fixupOldTriple(input);

  if (!checkAndFixAclBinaryTarget(input, Elf(), BuildLog()))
    return 1;

  int ret = 0;
  if (Options()->oVariables->UseJIT) {
    delete hookup_.amdrtFunctions;
    hookup_.amdrtFunctions = amd::getAmdRtFunctions();
  } else {
    hookup_.amdrtFunctions = NULL;
  }
  if (Options()->isOptionSeen(amd::option::OID_LUThreshold) || !IsGPUTarget) {
    setUnrollScratchThreshold(Options()->oVariables->LUThreshold);
  } else {
    setUnrollScratchThreshold(500);
  }

  llvmbinary_ = input;

  if ( !LLVMBinary() ) {
    BuildLog() += "Internal Error: cannot load bc application for linking\n";
    return 1;
  }

  if (linkLLVMModules(libs)) {
    BuildLog() += "Internal Error: failed to link modules correctlty.\n";
    return 1;
  }

  // Don't link in built-in libraries if we are only creating the library.
  if (Options()->oVariables->clCreateLibrary) {
    return 0;
  }

  if (Options()->isDumpFlagSet(amd::option::DUMP_BC_ORIGINAL)) {
#if defined(LEGACY_COMPLIB)
    std::string MyErrorInfo;
    std::string fileName = Options()->getDumpFileName("_original.bc");
    llvm::raw_fd_ostream outs(fileName.c_str(), MyErrorInfo, llvm::raw_fd_ostream::F_Binary);
    if (MyErrorInfo.empty())
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(MyErrorInfo.c_str());
#else
    std::string fileName = Options()->getDumpFileName("_original.bc");
    std::error_code EC;
    llvm::raw_fd_ostream outs(fileName.c_str(), EC, llvm::sys::fs::F_None);
    if (!EC)
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(EC.message().c_str());
#endif
  }

#ifdef HAS_SPIRV
  if (Options()->oVariables->RoundTripSPIRV && isSPIRModule(*llvmbinary_)) {
    std::string DumpSpirv;
    std::string DumpLlvm;
    if (Options()->isDumpFlagSet(amd::option::DUMP_BC_ORIGINAL)) {
      DumpSpirv = Options()->getDumpFileName(".spv");
      DumpLlvm = Options()->getDumpFileName("_fromspv.bc");
    }
    std::string TimeStr;
    translateSpirv(llvmbinary_, DumpSpirv, DumpLlvm,
        Options()->oVariables->EnableBuildTiming, TimeStr);
    if (!TimeStr.empty())
      appendLogToCL(CL(), TimeStr);
  }
#endif

  std::vector<llvm::Module*> LibMs;

  // The AMDIL GPU libraries include 32 bit specific, 64 bit specific and common
  // libraries. The common libraries do not have target triple. A search is
  // performed to find the first library containing non-empty target triple
  // and use it for translating SPIR.
  amd::LibraryDescriptor  LibDescs[
    amd::LibraryDescriptor::MAX_NUM_LIBRARY_DESCS];
  int sz;
  std::string LibTargetTriple;
  std::string LibDataLayout;
  if (amd::getLibDescs(Options()->libraryType_, LibDescs, sz) != 0) {
    // FIXME: If we error here, we don't clean up, so we crash in debug build
    // on compilerfini().
    BuildLog() += "Internal Error: finding libraries failed!\n";
    return 1;
  }
  for (int i=0; i < sz; i++) {
#if defined(LEGACY_COMPLIB)
    llvm::MemoryBuffer* Buffer = 0;
    llvm::Module* Library = amd::LoadLibrary(LibDescs[i].start, LibDescs[i].size, Context(), &Buffer);
#else
    llvm::Module *Library = amd::LoadLibrary(LibDescs[i].start, LibDescs[i].size, Context());
#endif
    DEBUG(llvm::dbgs() << "Loaded library " << i << "\n");
    if ( !Library ) {
      BuildLog() += "Internal Error: cannot load library!\n";
      delete LLVMBinary();
      for (int j = 0; j < i; ++j) {
        delete LibMs[j];
      }
      LibMs.clear();
      return 1;
#ifndef NDEBUG
    } else {
      if ( llvm::verifyModule( *Library ) ) {
        BuildLog() += "Internal Error: library verification failed!\n";
        exit(1);
      }
#endif
    }
    DEBUG_WITH_TYPE("linkTriple", llvm::dbgs() << "Library[" << i << "] " <<
        Library->getTargetTriple() << ' ' << Library->getDataLayout() << '\n');
    // Find the first library whose target triple is not empty.
    if (LibTargetTriple.empty() && !Library->getTargetTriple().empty()) {
        LibTargetTriple = Library->getTargetTriple();
#if defined(LEGACY_COMPLIB)
        LibDataLayout = Library->getDataLayout();
#else
        LibDataLayout = Library->getDataLayoutStr();
#endif
    }
    LibMs.push_back(Library);
  }

  // Check consistency of target and data layout
  assert (!LibTargetTriple.empty() && "At least one library should have triple");
#ifndef NDEBUG
  for (size_t i = 0, e = LibMs.size(); i < e; ++i) {
    if (LibMs[i]->getTargetTriple().empty())
      continue;
    assert (LibMs[i]->getTargetTriple() == LibTargetTriple &&
        "Library target triple should match");
#if defined(LEGACY_COMPLIB)
    assert (LibMs[i]->getDataLayout() == LibDataLayout &&
        "Library data layout should match");
#else
    assert (LibMs[i]->getDataLayoutStr() == LibDataLayout &&
        "Library data layout should match");
#endif
  }
#endif


  AMDSpir::replaceTrivialFunc(*LLVMBinary());

  if (!llvm::fixupKernelModule(LLVMBinary(), LibTargetTriple, LibDataLayout))
    return 1;

  // Before doing anything else, quickly optimize Module
  if (Options()->oVariables->EnableBuildTiming) {
    time_prelinkopt = amd::Os::timeNanos();
  }
  llvm::StringRef chip(aclGetChip(Elf()->target));
  setGPU(IsGPUTarget);
  setFiniteMathOnly(Options()->oVariables->FiniteMathOnly);
  setNoSignedZeros(Options()->oVariables->NoSignedZeros);
  setFastRelaxedMath(Options()->oVariables->FastRelaxedMath);
  setWholeProgram(true);
  setOptSimplifyLibCall(Options()->oVariables->OptSimplifyLibCall);
  setUnsafeMathOpt(Options()->oVariables->UnsafeMathOpt);
  setIsPreLinkOpt(Options()->oVariables->OptLevel);
  setFP32RoundDivideSqrt(Options()->oVariables->FP32RoundDivideSqrt);
  setUseNative(Options()->oVariables->OptUseNative);
  setDenormsAreZero(Options()->oVariables->DenormsAreZero);
#if !defined(LEGACY_COMPLIB)
  llvm::HLC_FlushF32Denorms = Options()->oVariables->DenormsAreZero;
#endif
  setUniformWorkGroupSize(Options()->oVariables->UniformWorkGroupSize);
  setHaveFastFMA32(chip == "Cypress"
                || chip == "Cayman"
                || chip == "Tahiti"
                || chip == "Hawaii"
                || chip == "Carrizo");
  LLVMBinary()->getContext().setAMDLLVMContextHook(&hookup_);

  std::string clp_errmsg;
  llvm::Module *OnFlyLib = AMDPrelink(LLVMBinary(), clp_errmsg);

  if (!clp_errmsg.empty()) {
    delete LLVMBinary();
    for (unsigned int i = 0; i < LibMs.size(); ++ i) {
      delete LibMs[i];
    }
    LibMs.clear();
    BuildLog() += clp_errmsg;
    BuildLog() += "Internal Error: on-fly library generation failed\n";
    return 1;
  }

  if (OnFlyLib) {
    // OnFlyLib must be the last!
    LibMs.push_back(OnFlyLib);
  }

  if (Options()->oVariables->EnableBuildTiming) {
    time_prelinkopt = amd::Os::timeNanos() - time_prelinkopt;
  }
  // Now, do linking by extracting from the builtins library only those
  // functions that are used in the kernel(s).
  if (Options()->oVariables->EnableBuildTiming) {
    time_link = amd::Os::timeNanos();
  }

  std::string ErrorMessage;
#ifdef LEGACY_COMPLIB
  createASICIDFunctions(LLVMBinary());
#endif // LEGACY_COMPLIB
  // Link libraries to get every functions that are referenced.
  std::string ErrorMsg;
  if (resolveLink(LLVMBinary(), LibMs, &ErrorMsg)) {
      BuildLog() += ErrorMsg;
      BuildLog() += "\nInternal Error: linking libraries failed!\n";
      return 1;
  }
  LibMs.clear();


  if (Options()->oVariables->EnableBuildTiming) {
    time_link = amd::Os::timeNanos() - time_link;
    std::stringstream tmp_ss;
    tmp_ss << "    LLVM time (link+opt): "
      << (amd::Os::timeNanos() - start_time)/1000ULL
          << " us\n"
          << "      prelinkopt: "  << time_prelinkopt/1000ULL << " us\n"
          << "      link: "  << time_link/1000ULL << " us\n"
            ;
    appendLogToCL(CL(), tmp_ss.str());
  }

#if defined(LEGACY_COMPLIB)
  // Disable outline macro for mem2reg=0 unless -fdebug-call
  // is on.
  if (!Options()->oVariables->OptMem2reg && !Options()->oVariables->DebugCall)
    Options()->oVariables->UseMacroForCall = false;

  if (isAMDILTarget(Elf()->target) &&
      getFamilyEnum(&Elf()->target) >= FAMILY_SI &&
      !Options()->oVariables->clInternalKernel &&
      (Options()->oVariables->OptMem2reg ||
      Options()->oVariables->DebugCall)) {
    auto OV = Options()->oVariables;
    AMDILFuncSupport::PostLinkProcForFuncSupport(
        OV->AddUserNoInline,
        OV->AddLibNoInline,
        OV->InlineCostThreshold,
        OV->InlineSizeThreshold,
        OV->InlineKernelSizeThreshold,
        OV->AllowMultiLevelCall && OV->UseMacroForCall,
        LLVMBinary(), LibMs);
  }
#endif

  if (Options()->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
    std::string MyErrorInfo;
    std::string fileName = Options()->getDumpFileName("_linked.bc");
#if defined(LEGACY_COMPLIB)
    llvm::raw_fd_ostream outs(fileName.c_str(), MyErrorInfo, llvm::raw_fd_ostream::F_Binary);
    // FIXME: Need to add this to the elf binary!
    if (MyErrorInfo.empty())
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(MyErrorInfo.c_str());
#else
    std::error_code EC;
    llvm::raw_fd_ostream outs(fileName.c_str(), EC, llvm::sys::fs::F_None);
    // FIXME: Need to add this to the elf binary!
    if (!EC)
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(EC.message().c_str());
#endif
  }

    // Check if kernels containing local arrays are called by other kernels.
    std::string localArrayUsageError;
    if (!llvm::AMDCheckLocalArrayUsage(*LLVMBinary(), &localArrayUsageError)) {
      BuildLog() += "Error: " + localArrayUsageError + '\n';
      return 1;
    }

    // check undefined function
#ifndef NDEBUG
    {
      auto M = LLVMBinary();
      for (auto I = M->begin(), E = M->end(); I != E; ++I) {
        if (!I->isDeclaration() || I->use_empty() || (I->hasName() &&
            (I->getName().startswith("__") ||
             I->getName().startswith("llvm."))))
          continue;
        llvm::errs() << "Warning: Undefined function: " << *I << '\n';
      }
    }
#endif

  return 0;
}
