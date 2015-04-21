//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "codegen.hpp"
#include "utils/libUtils.h"
#include "os/os.hpp"
#include "utils/target_mappings.h"
#ifdef _MSC_VER
/* for disabling warning in llvm/ADT/Statistic.h */
#pragma warning(disable:4146)
#endif
#include "llvm/ADT/Statistic.h"
#ifdef _MSC_VER
#pragma warning(default:4146)
#endif
#include "llvm/DataLayout.h"
#include "llvm/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>

using namespace amdcl;
using namespace llvm;

/*! Function that modifies the code gen level based on the
 * function size threshhold.
 */
static CodeGenOpt::Level
AdjustCGOptLevel(Module& M, CodeGenOpt::Level OrigOLvl)
{
  const unsigned int FuncSizeThreshold = 10000;
  if (OrigOLvl == CodeGenOpt::None)
    return OrigOLvl;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Function *F = (Function *)I;
    if (F->size() > FuncSizeThreshold) {
      return CodeGenOpt::None;
    }
  }
  return OrigOLvl;
}

//!--------------------------------------------------------------------------!//
// JIT Memory manager
//!--------------------------------------------------------------------------!//
OCLMCJITMemoryManager::~OCLMCJITMemoryManager() {
  for (llvm::SmallVectorImpl<Allocation>::iterator
         I = AllocatedCodeMem.begin(), E = AllocatedCodeMem.end();
       I != E; ++I)
    llvm::sys::Memory::releaseMappedMemory(I->first);
  for (llvm::SmallVectorImpl<Allocation>::iterator
         I = AllocatedDataMem.begin(), E = AllocatedDataMem.end();
       I != E; ++I)
    llvm::sys::Memory::releaseMappedMemory(I->first);
}

void
OCLMCJITMemoryManager::deallocateSection(uint8_t* BasePtr) {
  for (llvm::SmallVectorImpl<Allocation>::iterator
         I = AllocatedCodeMem.begin(), E = AllocatedCodeMem.end();
       I != E; ++I)
    if (I->first.base() == BasePtr) {
      llvm::sys::Memory::releaseMappedMemory(I->first);
      AllocatedCodeMem.erase(I);
      return;
    }
  for (llvm::SmallVectorImpl<Allocation>::iterator
         I = AllocatedDataMem.begin(), E = AllocatedDataMem.end();
       I != E; ++I)
    if (I->first.base() == BasePtr) {
      llvm::sys::Memory::releaseMappedMemory(I->first);
      AllocatedDataMem.erase(I);
      return;
    }
}

void OCLMCJITMemoryManager::reserveMemory(uint64_t Size) {
  llvm::sys::MemoryBlock Block = allocateSection(Size);
  AllocatedCodeMem.push_back(Allocation(Block, 64));
  allocPtr = (uint8_t*)Block.base();
  allocMaxPtr = allocPtr + Block.size();
}

uint8_t *OCLMCJITMemoryManager::
allocateCodeSection(uintptr_t Size, unsigned Alignment, unsigned SectionID) {
  // The recording memory manager is just a local copy of the remote target.
  // The alignment requirement is just stored here for later use. Regular
  // heap storage is sufficient here, but we're using mapped memory to work
  // around a bug in MCJIT.
  uint8_t* address = reservedAlloc(Size, Alignment);
  if(address != NULL) {
    return address;
  } else {
    llvm::sys::MemoryBlock Block = allocateSection(Size);
    AllocatedCodeMem.push_back(Allocation(Block, Alignment));
    return (uint8_t*)Block.base();
  }
}

uint8_t *OCLMCJITMemoryManager::
allocateDataSection(uintptr_t Size, unsigned Alignment,
                    unsigned SectionID, bool isReasOnly) {
  bool IsReadOnly = false;
  // The recording memory manager is just a local copy of the remote target.
  // The alignment requirement is just stored here for later use. Regular
  // heap storage is sufficient here, but we're using mapped memory to work
  // around a bug in MCJIT.
  uint8_t* address = reservedAlloc(Size, Alignment);
  if(address != NULL) {
    return address;
  } else {
    llvm::sys::MemoryBlock Block = allocateSection(Size);
    AllocatedDataMem.push_back(Allocation(Block, Alignment));
    return (uint8_t*)Block.base();
  }
}

uint8_t * OCLMCJITMemoryManager::reservedAlloc(uintptr_t Size, unsigned Alignment) {
  if(allocPtr != NULL) {
    uint8_t *allocPtrAligned =
      (uint8_t*)(((uintptr_t)allocPtr +
                  ((uintptr_t)Alignment-1)) & ~((uintptr_t)Alignment-1));
    uint8_t *allocPtrNext = allocPtrAligned + Size;
    if(allocPtrNext < allocMaxPtr) {
      allocPtr = allocPtrNext;
      return allocPtrAligned;
    }
  }
  return NULL;
}

llvm::sys::MemoryBlock OCLMCJITMemoryManager::allocateSection(uintptr_t Size) {
  llvm::error_code ec;
  llvm::sys::MemoryBlock MB =
    llvm::sys::Memory::allocateMappedMemory(Size,
                                            &Near,
                                            llvm::sys::Memory::MF_READ |
                                            llvm::sys::Memory::MF_WRITE |
                                            llvm::sys::Memory::MF_EXEC,
                                            ec);
  assert(!ec && MB.base());

  // FIXME: This is part of a work around to keep sections near one another
  // when MCJIT performs relocations after code emission but before
  // the generated code is moved to the remote target.
  // Save this address as the basis for our next request
  Near = MB;
  return MB;
}

void OCLMCJITMemoryManager::setMemoryWritable() {
  assert(!"Unexpected");
}

void OCLMCJITMemoryManager::setMemoryExecutable() {
  assert(!"Unexpected");
}

void OCLMCJITMemoryManager::setPoisonMemory(bool poison) {
  assert(!"Unexpected");
}

void OCLMCJITMemoryManager::AllocateGOT() {
  assert(!"Unexpected");
}

uint8_t *OCLMCJITMemoryManager::getGOTBase() const {
  assert(!"Unexpected");
  return 0;
}
uint8_t *OCLMCJITMemoryManager::startFunctionBody(const llvm::Function *F,
                                                  uintptr_t &ActualSize) {
  assert(!"Unexpected");
  return 0;
}
uint8_t *OCLMCJITMemoryManager::allocateStub(const llvm::GlobalValue* F,
                                             unsigned StubSize,
                                             unsigned Alignment) {
  assert(!"Unexpected");
  return 0;
}
void OCLMCJITMemoryManager::endFunctionBody(const llvm::Function *F,
                                            uint8_t *FunctionStart,
                                            uint8_t *FunctionEnd) {
  assert(!"Unexpected");
}
uint8_t *OCLMCJITMemoryManager::allocateSpace(intptr_t Size,
                                              unsigned Alignment) {
  assert(!"Unexpected");
  return 0;
}
uint8_t *OCLMCJITMemoryManager::allocateGlobal(uintptr_t Size,
                                               unsigned Alignment) {
  assert(!"Unexpected");
  return 0;
}
void OCLMCJITMemoryManager::deallocateFunctionBody(void *Body) {
  assert(!"Unexpected");
}
uint8_t* OCLMCJITMemoryManager::startExceptionTable(const llvm::Function* F,
                                                    uintptr_t &ActualSize) {
  assert(!"Unexpected");
  return 0;
}
void OCLMCJITMemoryManager::endExceptionTable(const llvm::Function *F,
                                              uint8_t *TableStart,
                                               uint8_t *TableEnd,
                                              uint8_t* FrameRegister) {
  assert(!"Unexpected");
}
void OCLMCJITMemoryManager::deallocateExceptionTable(void *ET) {
  assert(!"Unexpected");
}

static int jit_noop() {
  return 0;
}

void *OCLMCJITMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                        bool AbortOnFailure) {
  // We should not invoke parent's ctors/dtors from generated main()!
  // On Mingw and Cygwin, the symbol __main is resolved to
  // callee's(eg. tools/lli) one, to invoke wrong duplicated ctors
  // (and register wrong callee's dtors with atexit(3)).
  // We expect ExecutionEngine::runStaticConstructorsDestructors()
  // is called before ExecutionEngine::runFunctionAsMain() is called.
  if (Name == "__main") return (void*)(intptr_t)&jit_noop;

  return NULL;
}

//!--------------------------------------------------------------------------!//
// JIT Event Listener
//!--------------------------------------------------------------------------!//
class OclJITEventListener : public llvm::JITEventListener
{
private:
  std::string* output_;

public:
  OclJITEventListener(std::string &output) {
    output_ = &output;
  }

  virtual void NotifyObjectEmitted(const llvm::ObjectImage &Obj) {
    encodeObjectImage(Obj.getData(), *output_);
  }

  // Encoding and decoding are used to eliminate 0x00 ('\0') from the
  // string so it is safe to use it as a null terminated c string.
  // Translate:
  //    0x00 -> 0xaa 0x55
  //    0xaa -> 0xaa 0xaa
  static void encodeObjectImage(std::string objectImage, std::string &encodedObjectImage) {
    size_t length =  objectImage.length();
    for (size_t i = 0; i < length; ++i) {
      unsigned char c = objectImage[i];
      switch (c) {
      case 0x00U:
        encodedObjectImage.push_back(0xaaU);
        encodedObjectImage.push_back(0x55U);
        break;
      case 0xaaU:
        encodedObjectImage.push_back(0xaaU);
        encodedObjectImage.push_back(0xaaU);
        break;
      default:
        encodedObjectImage.push_back(c);
        break;
      }
    }
  }

  // Translate:
  //    0xaa 0x55 -> 0x00
  //    0xaa 0xaa -> 0xaa
  static void decodeObjectImage(std::string encodedObjectImage, std::string &decodedObjectImage) {
    size_t length =  encodedObjectImage.length();
    for (size_t i = 0; i < length; ++i) {
      unsigned char c = encodedObjectImage[i];
      switch (c) {
      case 0xaaU:
        {
          i = i + 1; // Increment to advance two characters
          unsigned char cnext = encodedObjectImage[i];
          if (cnext == 0xaaU) {
            decodedObjectImage.push_back(0xaaU);
          } else if (cnext == 0x55U) {
            decodedObjectImage.push_back(0x00U);
          } else {
            assert(!"Bad encoding encountered");
          }
        }
        break;
      default:
        decodedObjectImage.push_back(c);
        break;
      }
    }
  }

};

void decodeObjectImage(std::string encodedObjectImage, std::string &decodedObjectImage) {
  OclJITEventListener::decodeObjectImage(encodedObjectImage, decodedObjectImage);
}

// Returns empty string if code generation was successful,
// otherwise the return string contains the error the MCJIT encountered.
std::string
jitCodeGen(llvm::Module* Composite,
           llvm::TargetMachine* TargetMachine,
           llvm::CodeGenOpt::Level OLvl,
           std::string& output) {
  std::string ErrStr;
  OclJITEventListener Listener(output);
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::JITMemoryManager* MemMgr = new OCLMCJITMemoryManager();
  llvm::EngineBuilder builder(Composite);
  builder.setOptLevel(OLvl);
  builder.setErrorStr(&ErrStr);
  builder.setJITMemoryManager(MemMgr);
  builder.setUseMCJIT(true);
  // builder.setRelocationModel(llvm::Reloc::PIC_)
  // builder.setCodeModel(llvm::CodeModel::Large)
#ifndef ANDROID
  std::unique_ptr<llvm::ExecutionEngine>
    TheExecutionEngine(builder.create(TargetMachine));

  TheExecutionEngine->RegisterJITEventListener(&Listener);
  TheExecutionEngine->finalizeObject();
  TheExecutionEngine->removeModule(Composite);
#endif
  return ErrStr;
}

int
llvmCodeGen(
    Module* Composite,
    amd::option::Options *OptionsObj,
    std::string& output,
    aclBinary* binary)
{
  const FamilyMapping &familyMap = familySet[binary->target.arch_id];
  const bool optimize = (OptionsObj ? (OptionsObj->oVariables->OptLevel > 0) : true);
  const TargetMapping* targetMap = familyMap.target;
  unsigned famID = binary->target.chip_id;
  if (!targetMap || !targetMap[famID].supported) {
    LogError("Device is not supported by code generator!");
    return 1;
  }

#if 1 || LLVM_TRUNK_INTEGRATION_CL >= 1463
#else
  // a dirty way to guarantee "push bp" inserted by CodeGen in prologue
  llvm::NoFramePointerElim = !optimize;
#endif
  // Load the module to be compiled...
  Module &mod = *Composite;

  // FIXME: The triple given in this map is wrong and isn't really
  // useful. Only need the architecture.
  const std::string TargetTriple = std::string(familyMap.triple);
  Triple TheTriple(TargetTriple);
  if (TheTriple.getTriple().empty()) {
    TheTriple.setTriple(sys::getDefaultTargetTriple());
  }

  Triple::ArchType arch = TheTriple.getArch();

  bool isGPU = (arch == Triple::amdil || arch == Triple::amdil64 || 
                arch == Triple::hsail || arch == Triple::hsail64);

  if (isGPU) {
    TheTriple.setOS(Triple::UnknownOS);
  } else { // CPUs
    // FIXME: This should come from somewhere else.
#ifdef __linux__
    TheTriple.setOS(Triple::Linux);
#else
    TheTriple.setOS(Triple::MinGW32);
#endif
  }

  TheTriple.setEnvironment(Triple::AMDOpenCL);
  // FIXME: need to make AMDOpenCL be the same as ELF
  if (OptionsObj->oVariables->UseJIT)
    TheTriple.setEnvironment(Triple::ELF);
  mod.setTargetTriple(TheTriple.getTriple());

  // Allocate target machine.  First, check whether the user has explicitly
  // specified an architecture to compile for. If so we have to look it up by
  // name, because it might be a backend that has no mapping to a target triple.
  const Target *TheTarget = 0;
  assert(binary->target.arch_id != aclError && "Cannot have the error device!");

  std::string MArch = familyMap.architecture;

#ifdef WITH_TARGET_HSAIL
  if (MArch == "hsail" && OptionsObj->oVariables->GPU64BitIsa) {
    MArch = std::string("hsail-64");
  }
#endif

  for (TargetRegistry::iterator it = TargetRegistry::begin(),
      ie = TargetRegistry::end(); it != ie; ++it) {
    if (MArch == it->getName()) {
      TheTarget = &*it;
      break;
    }
  }

  if (!TheTarget) {
    errs() << ": ERROR: invalid target '" << MArch << "'.\n";
    return 1;
  }

  CodeGenOpt::Level OLvl = CodeGenOpt::None;
  switch (OptionsObj->oVariables->OptLevel) {
    case 0: // -O0
      OLvl = CodeGenOpt::None;
      break;
    case 1: // -O1
      OLvl = CodeGenOpt::Less;
      break;
    default:
      assert(!"Error with optimization level");
    case 2: // -O2
    case 5: // -O5(-Os)
      OLvl = CodeGenOpt::Default;
      break;
    case 3: // -O3
    case 4: // -O4
      OLvl = CodeGenOpt::Aggressive;
      break;
  };

  // If there is a very big function, lower the optimization level.
  OLvl = AdjustCGOptLevel(mod, OLvl);

  // Adjust the triple to match (if known), otherwise stick with the
  // module/host triple.
  Triple::ArchType Type = Triple::getArchTypeForLLVMName(MArch);
  if (Type != Triple::UnknownArch)
    TheTriple.setArch(Type);

  // Package up features to be passed to target/subtarget
  std::string FeatureStr;
  if ((Type == Triple::amdil || Type == Triple::amdil64) &&
      targetMap[famID].chip_options) {
    uint64_t y = targetMap[famID].chip_options;
    for (uint64_t x = 0; y != 0; y >>= 1, ++x) {
      if (!(y & 0x1) && (x >= 11 && x < 16)) {
        continue;
      }

      if ((1 << x) == F_NO_ALIAS) {
        FeatureStr += (!OptionsObj->oVariables->AssumeAlias ? '+' : '-');
      } else if ((1 << x) == F_STACK_UAV) {
        FeatureStr += (OptionsObj->oVariables->UseStackUAV ? '+' : '-');
      } else if ((1 << x) == F_MACRO_CALL) {
        FeatureStr += (OptionsObj->oVariables->UseMacroForCall ? '+' : '-');
      } else if ((1 << x) == F_64BIT_PTR) {
        FeatureStr += (binary->target.arch_id == aclAMDIL64) ? '+' : '-';
      } else {
        FeatureStr += ((y & 0x1) ? '+' : '-');
      }

      FeatureStr += GPUCodeGenFlagTable[x];
      if (y != 0x1) {
        FeatureStr += ',';
      }
    }
  }

  if (Type == Triple::amdil64) {
      if (OptionsObj->oVariables->SmallGlobalObjects)
          FeatureStr += ",+small-global-objects";
  }

#if 1 || LLVM_TRUNK_INTEGRATION_CL >= 1463
    llvm::TargetOptions targetOptions;
    targetOptions.NoFramePointerElim = false;
    targetOptions.StackAlignmentOverride =
      OptionsObj->oVariables->CPUStackAlignment;
    // jgolds
    //targetOptions.EnableEBB = (optimize && OptionsObj->oVariables->CGEBB);
    //targetOptions.EnableBFO = OptionsObj->oVariables->CGBFO;
    //targetOptions.NoExcessFPPrecision = !OptionsObj->oVariables->EnableFMA;

    // Don't allow unsafe optimizations for CPU because the library
    // contains code that is not safe.  See bug 9567.
    if (isGPU)
        targetOptions.UnsafeFPMath = OptionsObj->oVariables->UnsafeMathOpt;
    targetOptions.LessPreciseFPMADOption = OptionsObj->oVariables->MadEnable ||
                                           OptionsObj->oVariables->EnableMAD;
    targetOptions.NoInfsFPMath = OptionsObj->oVariables->FiniteMathOnly;
    // Need to add a support for OptionsObj->oVariables->NoSignedZeros,
    targetOptions.NoNaNsFPMath = OptionsObj->oVariables->FastRelaxedMath;

    std::auto_ptr<TargetMachine>
        target(TheTarget->createTargetMachine(TheTriple.getTriple(),
	       aclutGetCodegenName(binary->target), FeatureStr, targetOptions,
        WINDOWS_SWITCH(Reloc::DynamicNoPIC, Reloc::PIC_),
        CodeModel::Default, OLvl));
#else
  std::auto_ptr<TargetMachine>
  target(TheTarget->createTargetMachine(TheTriple.getTriple(),
        aclutGetCodegenName(binary->target), FeatureStr,
        WINDOWS_SWITCH(Reloc::DynamicNoPIC, Reloc::PIC_),
        CodeModel::Default));
  assert(target.get() && "Could not allocate target machine!");
#endif

  // MCJIT(Jan)
  if(!isGPU && OptionsObj->oVariables->UseJIT) {
    TargetMachine* jittarget(TheTarget->createTargetMachine(TheTriple.getTriple(),
        aclutGetCodegenName(binary->target), FeatureStr, targetOptions,
        WINDOWS_SWITCH(Reloc::DynamicNoPIC, Reloc::PIC_),
        CodeModel::Default, OLvl));

    std::string ErrStr = jitCodeGen(Composite, jittarget, OLvl, output);

    if (!ErrStr.empty()) {
      LogError("MCJIT failed to generate code");
      LogError(ErrStr.c_str());
      return 1;
    }
    return 0;
  }


  TargetMachine &Target = *target;

  // Figure out where we are going to send the output...
  raw_string_ostream *RSOut = new raw_string_ostream(output);
  formatted_raw_ostream *Out = new formatted_raw_ostream(*RSOut, formatted_raw_ostream::DELETE_STREAM);
  if (Out == 0) {
    LogError("llvmCodeGen couldn't create an output stream");
    return 1;
  }

  // Build up all of the passes that we want to do to the module or function or
  // Basic Block.
  PassManager Passes;

  // Add the target data from the target machine, if it exists, or the module.
  if (const DataLayout *TD = Target.getDataLayout())
    Passes.add(new DataLayout(*TD));
  else
    Passes.add(new DataLayout(&mod));

  // Override default to generate verbose assembly, if the device is not the GPU.
  // The GPU sets this in AMDILTargetMachine.cpp.
  if (familyMap.target == (const TargetMapping*)&X86TargetMapping ||
#if WITH_VERSION_0_9
      familyMap.target == (const TargetMapping*)&A32TargetMapping ||
      familyMap.target == (const TargetMapping*)&A32TargetMapping ||
#elif WITH_VERSION_0_8
#else
#error "The current version implementation was not implemented here."
#endif
      familyMap.target == (const TargetMapping*)&X64TargetMapping
      ) {
    Target.setAsmVerbosityDefault(true);
  }

#ifdef WITH_TARGET_HSAIL
  if (isHSAILTarget(binary->target)) {
    if (Target.addPassesToEmitFile(Passes, *Out, TargetMachine::CGFT_ObjectFile, true)) {
      delete Out;
      return 1;
    }
  } else 
#endif
  {
#ifndef NDEBUG
#if 1 || LLVM_TRUNK_INTEGRATION_CL >= 1144
    if (Target.addPassesToEmitFile(Passes, *Out, TargetMachine::CGFT_AssemblyFile, false))
#else
    if (Target.addPassesToEmitFile(Passes, *Out, TargetMachine::CGFT_AssemblyFile, OLvl, false))
#endif
#else
#if 1 || LLVM_TRUNK_INTEGRATION_CL >= 1144
    if (Target.addPassesToEmitFile(Passes, *Out, TargetMachine::CGFT_AssemblyFile, true))
#else
    if (Target.addPassesToEmitFile(Passes, *Out, TargetMachine::CGFT_AssemblyFile, OLvl, true))
#endif
#endif
    {
      delete Out;
      return 1;
    }
  }

  Passes.run(mod);
  llvm::PrintStatistics();
  delete Out;
  return 0;
}

  int
CLCodeGen::codegen(llvm::Module *input)
{
  uint64_t time_cg = 0ULL;
  if (Options()->oVariables->EnableBuildTiming) {
    time_cg = amd::Os::timeNanos();
  }
  llvmbinary_ = input;
  amdcl::CompilerStage *cs = reinterpret_cast<amdcl::CompilerStage*>(this);
  if (!isHSAILTarget(cs->Elf()->target)) {
    setWholeProgram(true);
  }

  int ret = llvmCodeGen(LLVMBinary(), Options(), Source(), Elf());

  if (Options()->oVariables->EnableBuildTiming) {
    time_cg = amd::Os::timeNanos() - time_cg;
    std::stringstream tmp_ss;
    tmp_ss << "    LLVM CodeGen time: "
      << time_cg/1000ULL
      << "us\n";
    appendLogToCL(CL(), tmp_ss.str());
  }
  if (!Source().empty() && Options()->isDumpFlagSet(amd::option::DUMP_CGIL)) {
    std::string ilFileName = Options()->getDumpFileName(".il");
    std::fstream f;
    f.open(ilFileName.c_str(), (std::fstream::out | std::fstream::binary));
    f.write(Source().data(), Source().length());
    f.close();
  }

  return ret;
}
