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

#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/GlobalValue.h"
#include "llvm/GlobalVariable.h"

#include "llvm/AMDResolveLinker.h"
#include "llvm/AMDPrelinkOpt.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/AMDLocalArrayUsage.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"

#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#if 1 || LLVM_TRUNK_INTEGRATION_CL >= 2270
#else
#include "llvm/CodeGen/ObjectCodeEmitter.h"
#endif
#include "llvm/Config/config.h"

#include "llvm/MC/SubtargetFeature.h"

#include "llvm/Support/CallSite.h"
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
#include "llvm/Support/system_error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ValueSymbolTable.h"

#ifdef _DEBUG
#include "llvm/Assembly/Writer.h"
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

static const char* OptionMaskFName = "__option_mask";
extern  llvm::Module*
clpVectorExpansion(llvm::Module *srcModules[], std::string &errorMsg);
namespace amd {

namespace {

using namespace llvm;

// LoadFile - Read the specified bitcode file in and return it.  This routine
// searches the link path for the specified file to try to find it...
//
inline llvm::Module*
  LoadFile(const std::string &Filename, LLVMContext& Context)
  {
    bool Exists;
    if (sys::fs::exists(Filename, Exists) || !Exists) {
      //    dbgs() << "Bitcode file: '" << Filename.c_str() << "' does not exist.\n";
      return 0;
    }

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
  }

inline llvm::Module*
  LoadLibrary(const std::string& libFile, LLVMContext& Context, MemoryBuffer** Buffer) {
    bool Exists;
    if (sys::fs::exists(libFile, Exists) || !Exists) {
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

// Load bitcode libary from an array of const char. This assumes that
// the array has a valid ending zero !
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


static std::set<std::string> *getAmdRtFunctions()
{
  std::set<std::string> *result = new std::set<std::string>();
  for (size_t i = 0; i < sizeof(amdRTFuns)/sizeof(amdRTFuns[0]); ++i)
    result->insert(amdRTFuns[i]);
  return result;
}

// Remove NoInline attribute to functions in a module
void
RemoveNoInlineAttr(llvm::Module* M)
{
  LLVMContext &Context = M->getContext();
  for (llvm::Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
    I->removeFnAttr(Attributes::get(Context, Attributes::NoInline));
  }
}

bool
IsKernel(llvm::Function* F)
{
  return F->getName().startswith("__OpenCL_") &&
      F->getName().endswith("_kernel");
}

// Add NoInline attribute to functions in a module
void
AddNoInlineAttr(llvm::Module* M)
{
  LLVMContext &Context = M->getContext();
  for (llvm::Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
    if (I->hasName() &&
        !I->isDeclaration() &&
        !I->isIntrinsic() &&
        !I->getName().startswith("__amdil") &&
        !I->getFnAttributes().hasAttribute(Attributes::AlwaysInline) &&
        !IsKernel(I)) {
      DEBUG_WITH_TYPE("noinline",
                      dbgs() << "[Candidate] " << I->getName() << '\n');
      I->addFnAttr(Attributes::NoInline);
    }
  }
}

unsigned
CountCallSites(llvm::Function* F, llvm::Module* M,
    std::map<llvm::Function*, unsigned>& counts) {
  std::map<llvm::Function*, unsigned>::iterator iter = counts.find(F);
  if (iter != counts.end())
    return iter->second;

  unsigned numCalled = 0;
  for (Function::use_iterator I = F->use_begin(), E = F->use_end(); I != E;
      ++I) {
    User *UI = *I;
    if (isa<CallInst>(UI) || isa<InvokeInst>(UI)) {
      ImmutableCallSite CS(cast<Instruction>(UI));
      Function* caller = const_cast<llvm::Function*>(CS.getCaller());
      unsigned callerCount = CountCallSites(caller, M, counts);
      if (caller->getFnAttributes().hasAttribute(Attributes::NoInline) &&
          callerCount > 0)
        numCalled++;
      else
        numCalled += callerCount;
    }
  }
  if (numCalled == 0 && IsKernel(F))
    numCalled = 1;

  counts[F] = numCalled;
  return numCalled;
}

unsigned
CalculateSize(llvm::Function* F, llvm::Module* M,
    std::map<llvm::Function*, unsigned>& sizes) {
  std::map<llvm::Function*, unsigned>::iterator iter = sizes.find(F);
  if (iter != sizes.end())
    return iter->second;

  CodeMetrics metrics;
  metrics.analyzeFunction(F);
  unsigned size = metrics.NumInsts;
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      if (CallInst* callInst = dyn_cast<CallInst>(BI)) {
        Function* called = callInst->getCalledFunction();
        if (called &&
            !called->getFnAttributes().hasAttribute(Attributes::NoInline))
          size += CalculateSize(called, M, sizes);
      }
    }
  }
  sizes[F] = size;
  return size;
}

// Identify functions with image arguments.
// Callers may pass images with different resource ids to the callee.
// Currently pointer manager cannot handle this.
// ToDo: Should remove this after we find a way to handle image in function.
bool
IsImageFunc(Function* F) {
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
      ++I) {
    if (PointerType *PT = dyn_cast<PointerType>(I->getType())) {
      if (PT->getAddressSpace() != 1) {
        continue;
      }
      if (StructType *ST = dyn_cast<StructType>(PT->getElementType())) {
        if (ST->getName().startswith("struct._image")) {
          DEBUG_WITH_TYPE("noinline", dbgs() << "[image function] " <<
              F->getName() << " inline\n");
          return true;
        }
      }
    }
  }
  return false;
}

bool
MustInline(Function* F) {
  if (F->getFnAttributes().hasAttribute(Attributes::AlwaysInline))
    return true;
  return IsImageFunc(F);
}

bool
CallerMustInline(Function* F) {
  return IsImageFunc(F);
}

bool
CallsNoInlineFunc(Function* F, std::map<Function*, bool>& work) {
  DEBUG_WITH_TYPE("noinline", dbgs() << "[CallsNoInlineFunc:" << F->getName() << " ");
  std::map<Function*, bool>::iterator loc = work.find(F);
  if (loc != work.end()) {
    DEBUG_WITH_TYPE("noinline", dbgs() << loc->second << "(cached)]\n");
    return loc->second;
  }
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      if (CallInst* callInst = dyn_cast<CallInst>(BI)) {
        Function* called = callInst->getCalledFunction();
        if (called) {
          if (called->getFnAttributes().hasAttribute(Attributes::NoInline) ||
              CallerMustInline(called) ||
              CallsNoInlineFunc(called, work)) {
            work[F] = true;
            DEBUG_WITH_TYPE("noinline", dbgs() << "1(" << called->getName() <<")]\n");
            return true;
          }
        }
      }
    }
  }
  work[F] = false;
  DEBUG_WITH_TYPE("noinline", dbgs() << "0]\n");
  return false;
}

bool
CalledByNoInlineFunc(Function* F, std::map<Function*, bool>& work) {
  DEBUG_WITH_TYPE("noinline", dbgs() << "[CalledByNoInlineFunc: " << F->getName() << " ");
  std::map<Function*, bool>::iterator loc = work.find(F);
  if (loc != work.end()) {
    DEBUG_WITH_TYPE("noinline", dbgs() << loc->second << "]\n");
    return loc->second;
  }
  for (Function::use_iterator I = F->use_begin(), E = F->use_end(); I != E;
      ++I) {
    User *UI = *I;
    if (isa<CallInst>(UI) || isa<InvokeInst>(UI)) {
      ImmutableCallSite CS(cast<Instruction>(UI));
      Function* caller = const_cast<llvm::Function*>(CS.getCaller());
      if (caller->getFnAttributes().hasAttribute(Attributes::NoInline) ||
          CalledByNoInlineFunc(caller, work)) {
        work[F] = true;
        DEBUG_WITH_TYPE("noinline", dbgs() << "1(" << caller->getName() <<")]\n");
        return true;
      }
    }
  }
  work[F] = false;
  DEBUG_WITH_TYPE("noinline", dbgs() << "0]\n");
  return false;
}

bool
CanBeNoInline(Function* F, std::map<Function*, bool>& callsNoInline,
    std::map<Function*, bool>& calledByNoInline, bool allowMultiLevelCall) {
  return !MustInline(F) && (allowMultiLevelCall ||
      (!CallsNoInlineFunc(F, callsNoInline) &&
      !CalledByNoInlineFunc(F, calledByNoInline)));
}

struct CostInfo {
  unsigned count;
  unsigned size;
  unsigned cost;
};

unsigned
CalculateMaxKernelSize(llvm::Module* M) {
  std::map<llvm::Function*, unsigned> sizes;
  unsigned maxSize = 0;
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
    if (IsKernel(I)) {
      unsigned kernelSize = CalculateSize(I, M, sizes);
      DEBUG_WITH_TYPE("noinlines", dbgs() << "[Kernel size] " <<
          I->getName() << " : " << kernelSize << '\n');
      if (maxSize < kernelSize)
        maxSize = kernelSize;
    }
  }
  return maxSize;
}

void
RefineNoInlineAttr(llvm::Module* M, int thresh, int sizeThresh,
    int kernelSizeThresh, bool allowMultiLevelCall)
{
  if (thresh == 0 && sizeThresh == 0)
    return;

  std::set<Function*> candidates;
  LLVMContext &Context = M->getContext();

  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
    if (I->getFnAttributes().hasAttribute(Attributes::NoInline)) {
      candidates.insert(I);
      I->removeFnAttr(Attributes::get(Context, Attributes::NoInline));
    }
  }

  unsigned maxKernelSize = CalculateMaxKernelSize(M);
  if (maxKernelSize < unsigned(kernelSizeThresh))
    return;

  while (true) {
    std::map<Function*, unsigned> counts;
    std::map<Function*, unsigned> sizes;
    std::map<Function*, CostInfo> costInfos;
    std::map<Function*, bool > callsNoInline;
    std::map<Function*, bool > calledByNoInline;
    for (std::set<Function*>::iterator I = candidates.begin(),
        E = candidates.end(); I != E; ++I) {
      Function* F = *I;
      unsigned count = CountCallSites(F, M, counts);
      if (count > 0 && CanBeNoInline(F, callsNoInline, calledByNoInline,
          allowMultiLevelCall)) {
        unsigned size = CalculateSize(F, M, sizes);
        if (size > unsigned(sizeThresh)) {
          CostInfo& info = costInfos[F];
          info.count = count;
          info.size = size;
          info.cost = (count - 1) * size;
          DEBUG_WITH_TYPE("noinline", dbgs() << F->getName() <<
            " : " << count - 1 << " * " << size << " = " << (count-1) * size <<
            "\n");
        }
      }
    }

    int maxCost = -1;
    Function* select = NULL;
    for (std::map<Function*, CostInfo>::iterator I = costInfos.begin(),
        E = costInfos.end(); I != E; ++I) {
      CostInfo& info = I->second;
      if (int(info.cost) > maxCost) {
        maxCost = int(info.cost);
        select = I->first;
      }
    }
    if (select == NULL || maxCost < thresh)
      break;
    CostInfo& info = costInfos[select];
    DEBUG_WITH_TYPE("noinlines", llvm::dbgs() << "select " << select->getName().str()
        << " cost = " << info.count << " x " << info.size << " = " <<
        info.cost << "\n");

    select->addFnAttr(Attributes::NoInline);
    candidates.erase(select);
    if (candidates.empty())
      break;
  }

  if (getenv("AMD_OCL_INLINE")) {
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
      if(I->hasName() && strstr(getenv("AMD_OCL_INLINE"),
          I->getName().str().c_str())) {
        I->removeFnAttr(Attributes::get(Context, Attributes::NoInline));
        printf("force inline %s\n", I->getName().data());
      }
    }
  }

  if (getenv("AMD_OCL_NOINLINE")) {
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
      if(I->hasName() && strstr(getenv("AMD_OCL_NOINLINE"),
          I->getName().str().c_str())) {
        I->addFnAttr(Attributes::NoInline);
        printf("force noinline %s\n", I->getName().data());
      }
    }
  }

}

} // unnamed namespace
} // namespace amd

// create a llvm function which simply returns the given mask
static void createConstIntFunc(const char* fname,
                               int mask,
                               llvm::Module* module)
{
  llvm::LLVMContext& context = module->getContext();

  llvm::Type* int32Ty = llvm::Type::getInt32Ty(context);
  llvm::FunctionType* fType = llvm::FunctionType::get(int32Ty, false);
  llvm::Function* function
      = llvm::cast<llvm::Function>(module->getOrInsertFunction(fname, fType));
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  function->addFnAttr(llvm::Attributes::AlwaysInline);
  llvm::BasicBlock* bb = llvm::BasicBlock::Create(context, "entry", function);
  llvm::Value* retVal = llvm::ConstantInt::get(int32Ty, mask);
  llvm::ReturnInst* retInst = llvm::ReturnInst::Create(context, retVal);
  bb->getInstList().push_back(retInst);
  assert(!verifyFunction(*function) && "verifyFunction failed");
}

// create a llvm function that returns a mask of several compile options
// which are used by the built-in library
void amdcl::OCLLinker::createOptionMaskFunction(llvm::Module* module)
{
  unsigned mask = 0;
  if (Options()->oVariables->NoSignedZeros) {
    mask |= MASK_NO_SIGNED_ZEROES;
  }
  if (Options()->oVariables->UnsafeMathOpt) {
    mask |= MASK_UNSAFE_MATH_OPTIMIZATIONS;
    mask |= MASK_NO_SIGNED_ZEROES;
  }
  if (Options()->oVariables->FiniteMathOnly) {
    mask |= MASK_FINITE_MATH_ONLY;
  }
  if (Options()->oVariables->FastRelaxedMath) {
    mask |= MASK_FAST_RELAXED_MATH;
    mask |= MASK_FINITE_MATH_ONLY;
    mask |= MASK_UNSAFE_MATH_OPTIMIZATIONS;
    mask |= MASK_NO_SIGNED_ZEROES;
  }

  if (Options()->oVariables->UniformWorkGroupSize) {
    mask |= MASK_UNIFORM_WORK_GROUP_SIZE;
  }

  createConstIntFunc(OptionMaskFName, mask, module);
}

// Create functions that returns true or false for some features which
// are used by the built-in library
void amdcl::OCLLinker::createASICIDFunctions(llvm::Module* module)
{
  if (!isAMDILTarget(Elf()->target))
    return;

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
                     || family == "CZ",
                     module);
  createConstIntFunc("__amdil_have_fast_fma32",
                        chip == "Cypress"
                     || chip == "Cayman"
                     || chip == "Tahiti"
                     || chip == "Hawaii",
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
                     || family == "CZ",
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
      llvm::raw_fd_ostream outs(fileName.c_str(), MyErrorInfo,
          llvm::raw_fd_ostream::F_Binary);
      if (MyErrorInfo.empty())
        llvm::WriteBitcodeToFile(libs[i], outs);
      else
        printf(MyErrorInfo.c_str());
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

static unsigned getSPIRVersion(const llvm::Module *M) {
  const llvm::NamedMDNode *SPIRVersion
    = M->getNamedMetadata("opencl.spir.version");

  if (!SPIRVersion) return 0; // not SPIR

  // When multiple llvm modules are linked together to create a single module
  // Metadata's of llvm modules are added into destination module and
  // it results in a more than one SPIR MDNode value.
  // Marking this fix as temporary and it will be tracked in bugzilla id 9775
  // FIXME: Uncomment the line below
  // assert(SPIRVersion->getNumOperands() == 1);
  assert(SPIRVersion->getNumOperands() > 0);
  if (SPIRVersion->getNumOperands() > 1) {
    DEBUG_WITH_TYPE("linkTriple",
                    llvm::dbgs() << "[CheckSPIRVersion] "
                    "Too many arguments to SPIR version MDNode\n");
  }

  const llvm::MDNode *VersionMD = SPIRVersion->getOperand(0);
  assert(VersionMD->getNumOperands() == 2);

  const llvm::ConstantInt *CMajor
    = llvm::cast<llvm::ConstantInt>(VersionMD->getOperand(0));
  assert(CMajor->getType()->getIntegerBitWidth() == 32);
  unsigned VersionMajor = CMajor->getZExtValue();

  const llvm::ConstantInt *CMinor
    = llvm::cast<llvm::ConstantInt>(VersionMD->getOperand(1));
  assert(CMinor->getType()->getIntegerBitWidth() == 32);
  unsigned VersionMinor = CMinor->getZExtValue();

  return (VersionMajor * 100) + (VersionMinor * 10);
}

//Modify module for targets before linking.
//Report error by buildLog.
//Return false on error.
static bool fixUpModule(llvm::Module *M,
                        llvm::StringRef TargetTriple,
                        llvm::StringRef TargetLayout,
                        bool RunSPIRLoader,
                        bool DemangleBuiltins,
                        bool RunEDGAdapter,
                        bool SetSPIRCallingConv,
                        bool RunX86Adpater) {
  llvm::PassManager Passes;

  DEBUG_WITH_TYPE("linkTriple", llvm::dbgs() <<
      "[fixUpModule] module triple: " << M->getTargetTriple() <<
      " target triple: " << TargetTriple);
  llvm::Triple triple(M->getTargetTriple());
#if OPENCL_MAJOR < 2
  if (triple.getArch() == llvm::Triple::spir ||
      triple.getArch() == llvm::Triple::spir64 ||
      triple.getArch() == llvm::Triple::x86 ||
      triple.getArch() == llvm::Triple::x86_64 ||
      M->getTargetTriple().empty())
#endif
  {
    M->setTargetTriple(TargetTriple);
    M->setDataLayout(TargetLayout);
  }
#if OPENCL_MAJOR < 2
  if (M->getTargetTriple() != TargetTriple) {
    //ToDo: There is bug 9996 in compiler library about converting BIF30 to BIF21
    //which causes regressions in ocltst if the following check is enabled.
    //Fix the bugs then enable the following check
  #if 0
    llvm::dbgs() << "Internal Error: Inconsistent module and library target\n";
    return false;
  #else
    llvm::dbgs() << "WARNING: Inconsistent module and library target\n";
    return true;
  #endif
  }
#endif

  Passes.add(new llvm::DataLayout(M));

  Passes.add(llvm::createAMDLowerAtomicsPass());

  if (getSPIRVersion(M) >= 200) {
    Passes.add(llvm::createAMDPrintfRuntimeBinding());
    Passes.add(llvm::createAMDLowerPipeBuiltinsPass());
    Passes.add(llvm::createAMDLowerEnqueueKernelPass());
    Passes.add(llvm::createAMDGenerateDevEnqMetadataPass());
  }

  if (RunEDGAdapter) {
    assert(!RunSPIRLoader);
    Passes.add(llvm::createAMDEDGToIA64TranslatorPass(SetSPIRCallingConv));
  }

  if (RunSPIRLoader) {
    assert(!RunEDGAdapter);
    Passes.add(llvm::createSPIRLoader(DemangleBuiltins));
  }

  if (RunX86Adpater) {
    // One of them should run before the AMDX86Adapter Pass.
    assert(RunSPIRLoader || RunEDGAdapter);
    Passes.add(llvm::createAMDX86AdapterPass());
  }

  Passes.run(*M);
  return true;
}

static bool isSPIRTriple(const llvm::Triple &Triple) {
  return Triple.getArch() == llvm::Triple::spir
    || Triple.getArch() == llvm::Triple::spir64;
}

static bool isAMDILTriple(const llvm::Triple &Triple) {
  return Triple.getArch() == llvm::Triple::amdil
    || Triple.getArch() == llvm::Triple::amdil64;
}

static bool isX86Triple(const llvm::Triple &Triple) {
  return Triple.getArch() == llvm::Triple::x86
    || Triple.getArch() == llvm::Triple::x86_64;
}

static bool isHSAILTriple(const llvm::Triple &Triple) {
  return Triple.getArch() == llvm::Triple::hsail
    || Triple.getArch() == llvm::Triple::hsail_64;
}

static void CheckSPIRVersionForTarget(const llvm::Module *M,
                                      const llvm::Triple &TargetTriple) {
  unsigned SPIRVersion = getSPIRVersion(M);
  if (SPIRVersion >= 200)
    assert(!isAMDILTriple(TargetTriple));
  else
    assert(SPIRVersion == 120);
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
      (triple.getArch() == llvm::Triple::hsail_64 ||
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
    hookup_.amdrtFunctions = amd::getAmdRtFunctions();
  } else {
    hookup_.amdrtFunctions = NULL;
  }
  if (Options()->isOptionSeen(amd::option::OID_LUThreshold) || !IsGPUTarget) {
    setUnrollScratchThreshold(Options()->oVariables->LUThreshold);
  } else {
    setUnrollScratchThreshold(500);
  }
  setGPU(IsGPUTarget);

  setPreLinkOpt(false);

  // We are doing whole program optimization
  setWholeProgram(true);

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
    std::string MyErrorInfo;
    std::string fileName = Options()->getDumpFileName("_original.bc");
    llvm::raw_fd_ostream outs(fileName.c_str(), MyErrorInfo, llvm::raw_fd_ostream::F_Binary);
    if (MyErrorInfo.empty())
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(MyErrorInfo.c_str());
  }
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
    llvm::MemoryBuffer* Buffer = 0;
    llvm::Module* Library = amd::LoadLibrary(LibDescs[i].start, LibDescs[i].size, Context(), &Buffer);
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
        LibDataLayout = Library->getDataLayout();
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
    assert (LibMs[i]->getDataLayout() == LibDataLayout &&
        "Library data layout should match");
  }
#endif


  // Under various situations, the LLVM dialect used in the kernel
  // module does not match the dialect used in the builtin library. We
  // need to fix-up the kernel module to eliminate this mismatch.
  //
  // SPIRLoader is required to consume a SPIR kernel:
  // SPIR 1.2 on all targets.
  // SPIR 2.0 on x86 and HSAIL only.
  //
  // The AMDIL libary is compiled by EDG, and hence it does not use
  // the SPIR mangling scheme. To allow a SPIR 1.2 kernel to link with
  // this library, the SPIRLoader must fix the mangling in the kernel.
  //
  // EDGAdapter is required to consume a non-SPIR (EDG) kernel on x86
  // and HSAIL targets. The builtins library for these targets are
  // built by Clang, but OpenCL 1.2 kernels are compiled by EDG.
  //
  // A non-SPIR kernel module is not expected on the HSAIL target in a
  // normal OpenCL 2.0 build. We should actually flag an error if this
  // occurs, but we let it through to facilitate custom builds created
  // to test this combination. In this situation, the EDGAdapter must
  // additionally set the calling conventions correctly, because the
  // HSAIL library is in SPIR format.
  //
  // RunX86Adpater is required to run only on the CPU path. It is
  // expected to the solve the link issues between the user kernel
  // (SPIR/EDG) vs. Clang compiled x86 builtins library.

                                    // Enabled for:
  bool RunSPIRLoader = false;       // SPIR     -> x86/HSAIL/AMDIL
  bool DemangleBuiltins = false;    // SPIR     -> AMDIL
  bool RunEDGAdapter = false;       // EDG      -> x86/HSAIL
  bool SetSPIRCallingConv = false;  // EDG      -> HSAIL
  bool RunX86Adapter = false;       // SPIR/EDG -> x86
  bool LowerToPreciseFunctions = false;

  llvm::Triple ModuleTriple(LLVMBinary()->getTargetTriple());
  llvm::Triple TargetTriple(LibTargetTriple);


  if (isSPIRTriple(ModuleTriple)) {
    CheckSPIRVersionForTarget(LLVMBinary(), TargetTriple);
    RunSPIRLoader = true;
#if OPENCL_MAJOR >= 2 // this will become default
    DemangleBuiltins |= isAMDILTriple(TargetTriple);
#ifdef BUILD_HSA_TARGET // special case for HSA build
    DemangleBuiltins |= isHSAILTriple(TargetTriple);
#endif
    // Never demangle for x86 target on 200 build.
#else // OpenCL 1.2 build (this will go away)
    DemangleBuiltins = true;
#endif
  } else {
#if OPENCL_MAJOR >= 2
    // Decide if we need to adapt the non-SPIR (EDG) kernel module.
    //
    // FIXME: Remove the #ifdef when x86 and HSAIL libraries are
    // always built by Clang.
#ifndef BUILD_HSA_TARGET
    // Run the adapter for HSAIL, only if this is an ORCA build!
    //
    // On an HSA build, the HSAIL library is always built with EDG.
    // This assumption must match the settings in
    // "opencl/library/hsa/hsail/build/Makefile.hsail"
    RunEDGAdapter |= isHSAILTriple(TargetTriple);
#endif
    // HSAIL requires SPIR calling conventions since the library is in
    // SPIR format. This doesn't matter if the EDGAdapter is not run.
    SetSPIRCallingConv = isHSAILTriple(TargetTriple);

    // Run the EDG Adapter if OPENCL_MAJOR >= 2 and for x86 target.
    RunEDGAdapter |= isX86Triple(TargetTriple);
#endif // OPENCL_MAJOR >= 2
  }

// X86Adapter should run for both EDG generated LLVM IR and SPIR for x86 path.
// FIXME: Remove the #ifdef when x86 is always built by Clang on
// OpenCL 1.2 builds.
#if OPENCL_MAJOR >=2
  RunX86Adapter = isX86Triple(TargetTriple);
  // For HSAIL targets, when the option -cl-fp32-correctly-rounded-divide-sqrt
  // lower divide and sqrt functions to precise HSAIL builtin library functions.
  LowerToPreciseFunctions = (isHSAILTriple(TargetTriple)
		  && Options()->oVariables->FP32RoundDivideSqrt);
#endif

  if (!fixUpModule(LLVMBinary(), LibTargetTriple, LibDataLayout,
                   RunSPIRLoader, DemangleBuiltins,
                   RunEDGAdapter, SetSPIRCallingConv,
                   RunX86Adapter))
    return 1;

  // Before doing anything else, quickly optimize Module
  if (Options()->oVariables->OptLevel) {
    if (Options()->oVariables->EnableBuildTiming) {
      time_prelinkopt = amd::Os::timeNanos();
    }

    AMDPrelinkOpt(LLVMBinary(), true /*Whole*/,
      !Options()->oVariables->OptSimplifyLibCall,
      Options()->oVariables->UnsafeMathOpt,
      Options()->oVariables->OptUseNative,
      LowerToPreciseFunctions);

    if (Options()->oVariables->EnableBuildTiming) {
      time_prelinkopt = amd::Os::timeNanos() - time_prelinkopt;
    }
  }
  // Now, do linking by extracting from the builtins library only those
  // functions that are used in the kernel(s).
  if (Options()->oVariables->EnableBuildTiming) {
    time_link = amd::Os::timeNanos();
  }

  std::string ErrorMessage;

  // CL pre-link processing
  llvm::Module *clp_inputs[2];
  clp_inputs[0] = LLVMBinary();
  clp_inputs[1] = NULL;
  std::string clp_errmsg;
  llvm::Module *OnFlyLib = clpVectorExpansion (clp_inputs, clp_errmsg);
  if (clp_errmsg.empty() == false) {
    delete LLVMBinary();
    for (unsigned int i = 0; i < LibMs.size(); ++ i) {
      delete LibMs[i];
    }
    LibMs.clear();
    BuildLog() += clp_errmsg;
    BuildLog() += "Internal Error: on-fly library generation failed\n";
    return 1;
  }

  unsigned int offset = (unsigned int)LibMs.size();

  if (OnFlyLib) {
    // OnFlyLib must be the last!
    LibMs.push_back(OnFlyLib);
  }

  // build the reference map
  llvm::ReferenceMapBuilder RefMapBuilder(LLVMBinary(), LibMs);

  RefMapBuilder.InitReferenceMap();

  if (IsGPUTarget && RefMapBuilder.isInExternFuncs("printf")) {
    DEBUG(llvm::dbgs() << "Adding printf funs:\n");
    // The following functions need forcing as printf-conversion happens
    // after this link stage
    static const char* forcedRefs[] = {
      "___initDumpBuf",
      "___dumpBytes_v1b8",
      "___dumpBytes_v1b16",
      "___dumpBytes_v1b32",
      "___dumpBytes_v1b64",
      "___dumpBytes_v1b128",
      "___dumpBytes_v1b256",
      "___dumpBytes_v1b512",
      "___dumpBytes_v1b1024",
      "___dumpBytes_v1bs",
      "___dumpStringID"
    };
    RefMapBuilder.AddForcedReferences(forcedRefs,
      sizeof(forcedRefs)/sizeof(forcedRefs[0]));
  }
  if (!IsGPUTarget && Options()->oVariables->UseJIT) {
    RefMapBuilder.AddForcedReferences(amd::amdRTFuns,
      sizeof(amd::amdRTFuns)/sizeof(amd::amdRTFuns[0]));
  }

  RefMapBuilder.AddReferences();

  // inject an llvm function that returns the mask of several compile
  // options, which are used by the built-in library
  const std::list<std::string>& ExternFuncs
    = RefMapBuilder.getExternFunctions();
  const std::list<std::string>::const_iterator it
    = std::find(ExternFuncs.begin(), ExternFuncs.end(), OptionMaskFName);
  if (it != ExternFuncs.end()) {
    createOptionMaskFunction(LLVMBinary());
  }

  createASICIDFunctions(LLVMBinary());

  if (!isHSAILTarget(Elf()->target)) {
    // Add NoInline attribute to user functions
    llvm::StringRef family(aclGetFamily(Elf()->target));
    llvm::StringRef chip(aclGetChip(Elf()->target));

    // Add NoInline attribute to library functions so that they
    // can be considered for not inlining in codegen.
    if (IsGPUTarget &&
        (Options()->oVariables->OptMem2reg || Options()->oVariables->DebugCall) &&
        !Options()->oVariables->clInternalKernel &&
        !(family == "NI" || family == "Evergreen" || family == "Sumo" ||
          family == "TN")) {
      if (Options()->oVariables->AddUserNoInline)
        amd::AddNoInlineAttr(LLVMBinary());
      if (Options()->oVariables->AddLibNoInline)
        for (unsigned int i=0; i < LibMs.size(); i++)
          amd::AddNoInlineAttr(LibMs[i]);
    }

    // Disable outline macro for mem2reg=0 unless -fdebug-call
    // is on.
    if (!Options()->oVariables->OptMem2reg && !Options()->oVariables->DebugCall) {
      Options()->oVariables->UseMacroForCall = false;
    }
  }

  // Link libraries to get every functions that are referenced.
  std::string ErrorMsg;
  if (resolveLink(LLVMBinary(), LibMs, RefMapBuilder.getModuleRefMaps(),
                  &ErrorMsg)) {
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

  if (!isHSAILTarget(Elf()->target)) {
    // Refine NoInline attribute of functions
    if (IsGPUTarget && !Options()->oVariables->clInternalKernel) {
      amd::RefineNoInlineAttr(LLVMBinary(),
          Options()->oVariables->InlineCostThreshold,
          Options()->oVariables->InlineSizeThreshold,
          Options()->oVariables->InlineKernelSizeThreshold,
          Options()->oVariables->AllowMultiLevelCall &&
          Options()->oVariables->UseMacroForCall );
    }
  }

  if (Options()->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
    std::string MyErrorInfo;
    std::string fileName = Options()->getDumpFileName("_linked.bc");
    llvm::raw_fd_ostream outs(fileName.c_str(), MyErrorInfo, llvm::raw_fd_ostream::F_Binary);
    // FIXME: Need to add this to the elf binary!
    if (MyErrorInfo.empty())
      WriteBitcodeToFile(LLVMBinary(), outs);
    else
      printf(MyErrorInfo.c_str());
  }

    // Check if kernels containing local arrays are called by other kernels.
    std::string localArrayUsageError;
    if (!llvm::AMDCheckLocalArrayUsage(*LLVMBinary(), &localArrayUsageError)) {
      BuildLog() += "Error: " + localArrayUsageError + '\n';
      return 1;
    }

  return 0;
}
