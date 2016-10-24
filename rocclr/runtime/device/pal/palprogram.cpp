//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#include "os/os.hpp"
#include "utils/flags.hpp"
#include "include/aclTypes.h"
#include "utils/amdilUtils.hpp"
#include "utils/bif_section_labels.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palblit.hpp"
#include "macrodata.h"
#include "MDParser/AMDILMDInterface.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include "utils/options.hpp"
#include "hsa.h"
#include "hsa_ext_image.h"
#include "amd_hsa_loader.hpp"
#if defined(WITH_LIGHTNING_COMPILER)
#include "libamdhsacode/amdgpu_metadata.hpp"
#include "driver/AmdCompiler.h"
#include "libraries.amdgcn.inc"
#include "gelf.h"
#endif // !defined(WITH_LIGHTNING_COMPILER)

namespace pal {

HSAILProgram::HSAILProgram(Device& device)
    : Program(device)
    , llvmBinary_()
    , binaryElf_(nullptr)
    , rawBinary_(nullptr)
    , kernels_(nullptr)
    , codeSegGpu_(nullptr)
    , codeSegCpu_(nullptr)
    , maxScratchRegs_(0)
    , flags_(0)
    , executable_(nullptr)
    , loaderContext_(this)
{
    memset(&binOpts_, 0, sizeof(binOpts_));
    binOpts_.struct_size = sizeof(binOpts_);
    binOpts_.elfclass = LP64_SWITCH(ELFCLASS32, ELFCLASS64);
    binOpts_.bitness = ELFDATA2LSB;
    binOpts_.alloc = &::malloc;
    binOpts_.dealloc = &::free;
    loader_ = amd::hsa::loader::Loader::Create(&loaderContext_);
}

HSAILProgram::HSAILProgram(NullDevice& device)
    : Program(device)
    , llvmBinary_()
    , binaryElf_(nullptr)
    , rawBinary_(nullptr)
    , kernels_(nullptr)
    , codeSegGpu_(nullptr)
    , codeSegCpu_(nullptr)
    , maxScratchRegs_(0)
    , flags_(0)
    , executable_(nullptr)
    , loaderContext_(this)
{
    memset(&binOpts_, 0, sizeof(binOpts_));
    binOpts_.struct_size = sizeof(binOpts_);
    binOpts_.elfclass = LP64_SWITCH(ELFCLASS32, ELFCLASS64);
    binOpts_.bitness = ELFDATA2LSB;
    binOpts_.alloc = &::malloc;
    binOpts_.dealloc = &::free;
    loader_ = amd::hsa::loader::Loader::Create(&loaderContext_);
}

HSAILProgram::~HSAILProgram()
{
    // Destroy internal static samplers
    for (auto& it : staticSamplers_) {
        delete it;
    }
#if !defined(WITH_LIGHTNING_COMPILER)
    if (rawBinary_ != nullptr) {
        aclFreeMem(binaryElf_, rawBinary_);
    }
    acl_error error;
    // Free the elf binary
    if (binaryElf_ != nullptr) {
        error = aclBinaryFini(binaryElf_);
        if (error != ACL_SUCCESS) {
            LogWarning( "Error while destroying the acl binary \n" );
        }
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)
    releaseClBinary();
    if (executable_ != nullptr) {
        loader_->DestroyExecutable(executable_);
    }
    delete kernels_;
    amd::hsa::loader::Loader::Destroy(loader_);
    assert((codeSegGpu_ == nullptr) && "Loader didn't destroy code!");
    assert((codeSegCpu_ == nullptr) && "Loader didn't destroy code!");
}

bool
HSAILProgram::initBuild(amd::option::Options *options)
{
    if (!device::Program::initBuild(options)) {
        return false;
    }

    const char* devName = dev().hwInfo()->machineTarget_;
    options->setPerBuildInfo(
        (devName && (devName[0] != '\0')) ? devName : "gpu",
        clBinary()->getEncryptCode(), true);

    // Elf Binary setup
    std::string outFileName;

    // true means fsail required
    clBinary()->init(options, true);
    if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
        outFileName = options->getDumpFileName(".bin");
    }

    if (!clBinary()->setElfOut(LP64_SWITCH(ELFCLASS32, ELFCLASS64),
        (outFileName.size() > 0) ? outFileName.c_str() : nullptr)) {
        LogError("Setup elf out for gpu failed");
        return false;
    }
    return true;
}

bool
HSAILProgram::finiBuild(bool isBuildGood)
{
    clBinary()->resetElfOut();
    clBinary()->resetElfIn();

    if (!isBuildGood) {
        // Prevent the encrypted binary form leaking out
        clBinary()->setBinary(nullptr, 0);
    }

    return device::Program::finiBuild(isBuildGood);
}

bool
HSAILProgram::linkImpl(
    const std::vector<device::Program *> &inputPrograms,
    amd::option::Options *options,
    bool createLibrary)
{
#if defined(WITH_LIGHTNING_COMPILER)
    assert(!"Should not reach here");
    return false;
#else // !defined(WITH_LIGHTNING_COMPILER)
    std::vector<device::Program *>::const_iterator it
        = inputPrograms.begin();
    std::vector<device::Program *>::const_iterator itEnd
        = inputPrograms.end();
    acl_error errorCode;

    // For each program we need to extract the LLVMIR and create
    // aclBinary for each
    std::vector<aclBinary *> binaries_to_link;

    for (size_t i = 0; it != itEnd; ++it, ++i) {
        HSAILProgram *program = (HSAILProgram *)*it;
        // Check if the program was created with clCreateProgramWIthBinary
        binary_t binary = program->binary();
        if ((binary.first != nullptr) && (binary.second > 0)) {
            // Binary already exists -- we can also check if there is no
            // opencl source code
            // Need to check if LLVMIR exists in the binary
            // If LLVMIR does not exist then is it valid
            // We need to pull out all the compiled kernels
            // We cannot do this at present because we need at least
            // Hsail text to pull the kernels oout
            void *mem = const_cast<void *>(binary.first);
            binaryElf_ = aclReadFromMem(mem, binary.second, &errorCode);
            if (errorCode != ACL_SUCCESS) {
                LogWarning("Error while linking : Could not read from raw binary");
                return false;
            }
        }
        // At this stage each HSAILProgram contains a valid binary_elf
        // Check if LLVMIR is in the binary
        // @TODO - Memory leak , cannot free this buffer
        // need to fix this.. File EPR on compiler library
        size_t llvmirSize = 0;
        const void *llvmirText = aclExtractSection(dev().compiler(),
            binaryElf_, &llvmirSize, aclLLVMIR, &errorCode);
        if (errorCode != ACL_SUCCESS) {
            bool spirv = false;
            size_t boolSize = sizeof(bool);
            errorCode = aclQueryInfo(dev().compiler(), binaryElf_,
                RT_CONTAINS_SPIRV, nullptr, &spirv, &boolSize);
            if (errorCode != ACL_SUCCESS) {
                spirv = false;
            }
            if (spirv) {
                errorCode = aclCompile(dev().compiler(), binaryElf_,
                    options->origOptionStr.c_str(), ACL_TYPE_SPIRV_BINARY,
                    ACL_TYPE_LLVMIR_BINARY, nullptr);
                buildLog_ += aclGetCompilerLog(dev().compiler());
                if (errorCode != ACL_SUCCESS) {
                    buildLog_ += "Error while linking: Could not load SPIR-V" ;
                    return false;
                }
            } else {
                buildLog_ +="Error while linking : \
                        Invalid binary (Missing LLVMIR section)" ;
                return false;
            }
        }
        // Create a new aclBinary for each LLVMIR and save it in a list
        aclBIFVersion ver = aclBinaryVersion(binaryElf_);
        aclBinary *bin = aclCreateFromBinary(binaryElf_, ver);
        binaries_to_link.push_back(bin);
    }

    errorCode = aclLink(dev().compiler(),
        binaries_to_link[0], binaries_to_link.size() - 1,
        binaries_to_link.size() > 1 ? &binaries_to_link[1] : NULL,
        ACL_TYPE_LLVMIR_BINARY, "-create-library", NULL);
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += aclGetCompilerLog(dev().compiler());
        buildLog_ +="Error while linking : aclLink failed" ;
        return false;
    }
    // Store the newly linked aclBinary for this program.
    binaryElf_ = binaries_to_link[0];
    // Free all the other aclBinaries
    for (size_t i = 1; i < binaries_to_link.size(); i++) {
        aclBinaryFini(binaries_to_link[i]);
    }
    if (createLibrary) {
        saveBinaryAndSetType(TYPE_LIBRARY);
        buildLog_ += aclGetCompilerLog(dev().compiler());
        return true;
    }
    // Now call linkImpl with the new options
    return linkImpl(options);
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

aclType
HSAILProgram::getCompilationStagesFromBinary(std::vector<aclType>& completeStages, bool& needOptionsCheck)
{
#if defined(WITH_LIGHTNING_COMPILER)
    assert(!"Should not reach here");
    return ACL_TYPE_DEFAULT;
#else // !defined(WITH_LIGHTNING_COMPILER)
    acl_error errorCode;
    size_t secSize = 0;
    completeStages.clear();
    aclType from = ACL_TYPE_DEFAULT;
    needOptionsCheck = true;
    size_t boolSize = sizeof(bool);
    // Checking llvmir in .llvmir section
    bool containsSpirv = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_,
            RT_CONTAINS_SPIRV, nullptr, &containsSpirv, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsSpirv = false;
    }
    if (containsSpirv) {
        completeStages.push_back(from);
        from = ACL_TYPE_SPIRV_BINARY;
    }
    bool containsSpirText = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_CONTAINS_SPIR, nullptr, &containsSpirText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsSpirText = false;
    }
    if (containsSpirText) {
        completeStages.push_back(from);
        from = ACL_TYPE_SPIR_BINARY;
    }
    bool containsLlvmirText = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_CONTAINS_LLVMIR, nullptr, &containsLlvmirText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsLlvmirText = false;
    }
    // Checking compile & link options in .comment section
    bool containsOpts = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_CONTAINS_OPTIONS, nullptr, &containsOpts, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsOpts = false;
    }
    if (containsLlvmirText && containsOpts) {
        completeStages.push_back(from);
        from = ACL_TYPE_LLVMIR_BINARY;
    }
    // Checking HSAIL in .cg section
    bool containsHsailText = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_CONTAINS_HSAIL, nullptr, &containsHsailText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsHsailText = false;
    }
    // Checking BRIG sections
    bool containsBrig = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_CONTAINS_BRIG, nullptr, &containsBrig, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsBrig = false;
    }
    if (containsBrig) {
        completeStages.push_back(from);
        from = ACL_TYPE_HSAIL_BINARY;
    } else if (containsHsailText) {
        completeStages.push_back(from);
        from = ACL_TYPE_HSAIL_TEXT;
    }
    // Checking Loader Map symbol from CG section
    bool containsLoaderMap = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_CONTAINS_LOADER_MAP, nullptr, &containsLoaderMap, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsLoaderMap = false;
    }
    if (containsLoaderMap) {
        completeStages.push_back(from);
        from = ACL_TYPE_CG;
    }
    // Checking ISA in .text section
    bool containsShaderIsa = true;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_CONTAINS_ISA, nullptr, &containsShaderIsa, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsShaderIsa = false;
    }
    if (containsShaderIsa) {
        completeStages.push_back(from);
        from = ACL_TYPE_ISA;
    }
    std::string sCurOptions = compileOptions_ + linkOptions_;
    amd::option::Options curOptions;
    if (!amd::option::parseAllOptions(sCurOptions, curOptions)) {
        buildLog_ += curOptions.optionsLog();
        LogError("Parsing compile options failed.");
        return ACL_TYPE_DEFAULT;
    }
    switch (from) {
    // compile from HSAIL text, no matter prev. stages and options
    case ACL_TYPE_HSAIL_TEXT:
        needOptionsCheck = false;
        break;
    case ACL_TYPE_HSAIL_BINARY:
        // do not check options, if LLVMIR is absent or might be absent or options are absent
        if (!curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
            needOptionsCheck = false;
        }
        break;
    case ACL_TYPE_CG:
    case ACL_TYPE_ISA:
        // do not check options, if LLVMIR is absent or might be absent or options are absent
        if (!curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
            needOptionsCheck = false;
        }
        // do not check options, if BRIG is absent or might be absent or LoaderMap is absent
        if (!curOptions.oVariables->BinCG || !containsBrig || !containsLoaderMap) {
            needOptionsCheck = false;
        }
        break;
    // recompilation might be needed
    case ACL_TYPE_LLVMIR_BINARY:
    case ACL_TYPE_DEFAULT:
    default:
        break;
    }
    return from;
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

aclType
HSAILProgram::getNextCompilationStageFromBinary(amd::option::Options* options) {
#if defined(WITH_LIGHTNING_COMPILER)
    assert(!"Should not reach here");
    return ACL_TYPE_DEFAULT;
#else // !defined(WITH_LIGHTNING_COMPILER)
    aclType continueCompileFrom = ACL_TYPE_DEFAULT;
    binary_t binary = this->binary();
    // If the binary already exists
    if ((binary.first != nullptr) && (binary.second > 0)) {
        void *mem = const_cast<void *>(binary.first);
        acl_error errorCode;
        binaryElf_ = aclReadFromMem(mem, binary.second, &errorCode);
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error: Reading the binary from memory failed.\n";
            return continueCompileFrom;
      }
      // Calculate the next stage to compile from, based on sections in binaryElf_;
      // No any validity checks here
      std::vector<aclType> completeStages;
      bool needOptionsCheck = true;
      continueCompileFrom = getCompilationStagesFromBinary(completeStages, needOptionsCheck);
      // Saving binary in the interface class,
      // which also load compile & link options from binary
      setBinary(static_cast<char*>(mem), binary.second);
      if (!options || !needOptionsCheck) {
          return continueCompileFrom;
      }
      bool recompile = false;
      switch (continueCompileFrom) {
      case ACL_TYPE_HSAIL_BINARY:
      case ACL_TYPE_CG:
      case ACL_TYPE_ISA: {
          // Compare options loaded from binary with current ones, recompile if differ;
          // If compile options are absent in binary, do not compare and recompile
          if (compileOptions_.empty())
              break;
          const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symOpenclCompilerOptions);
          assert(symbol && "symbol not found");
          std::string symName = std::string(symbol->str[bif::PRE]) + std::string(symbol->str[bif::POST]);
          size_t symSize = 0;
          const void *opts = aclExtractSymbol(dev().compiler(),
              binaryElf_, &symSize, aclCOMMENT, symName.c_str(), &errorCode);
          if (errorCode != ACL_SUCCESS) {
              recompile = true;
              break;
          }
          std::string sBinOptions = std::string((char*)opts, symSize);
          std::string sCurOptions = compileOptions_ + linkOptions_;
          amd::option::Options curOptions, binOptions;
          if (!amd::option::parseAllOptions(sBinOptions, binOptions)) {
              buildLog_ += binOptions.optionsLog();
              LogError("Parsing compile options from binary failed.");
              return ACL_TYPE_DEFAULT;
          }
          if (!amd::option::parseAllOptions(sCurOptions, curOptions)) {
              buildLog_ += curOptions.optionsLog();
              LogError("Parsing compile options failed.");
              return ACL_TYPE_DEFAULT;
          }
          if (!curOptions.equals(binOptions)) {
              recompile = true;
          }
          break;
      }
      default:
          break;
      }
      if (recompile) {
          while (!completeStages.empty()) {
              continueCompileFrom = completeStages.back();
              if (continueCompileFrom == ACL_TYPE_SPIRV_BINARY ||
                  continueCompileFrom == ACL_TYPE_LLVMIR_BINARY ||
                  continueCompileFrom == ACL_TYPE_SPIR_BINARY ||
                  continueCompileFrom == ACL_TYPE_DEFAULT) {
                  break;
              }
              completeStages.pop_back();
          }
      }
    }
    return continueCompileFrom;
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

inline static std::vector<std::string>
splitSpaceSeparatedString(char *str)
{
  std::string s(str);
  std::stringstream ss(s);
  std::istream_iterator<std::string> beg(ss), end;
  std::vector<std::string> vec(beg, end);
  return vec;
}

bool
HSAILProgram::linkImpl(amd::option::Options* options)
{
#if defined(WITH_LIGHTNING_COMPILER)
    assert(!"Should not reach here");
    return false;
#else // !defined(WITH_LIGHTNING_COMPILER)
    acl_error errorCode;
    aclType continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;
    bool finalize = true;
    bool hsaLoad = true;
    internal_ = (compileOptions_.find("-cl-internal-kernel") !=
        std::string::npos) ? true : false;


    // If !binaryElf_ then program must have been created using clCreateProgramWithBinary
    if (!binaryElf_) {
        continueCompileFrom = getNextCompilationStageFromBinary(options);
    }
    switch (continueCompileFrom) {
    case ACL_TYPE_SPIRV_BINARY:
    case ACL_TYPE_SPIR_BINARY:
    // Compilation from ACL_TYPE_LLVMIR_BINARY to ACL_TYPE_CG in cases:
    // 1. if the program is not created with binary;
    // 2. if the program is created with binary and contains only .llvmir & .comment
    // 3. if the program is created with binary, contains .llvmir, .comment, brig sections,
    //    but the binary's compile & link options differ from current ones (recompilation);
    case ACL_TYPE_LLVMIR_BINARY:
    // Compilation from ACL_TYPE_HSAIL_BINARY to ACL_TYPE_CG in cases:
    // 1. if the program is created with binary and contains only brig sections
    case ACL_TYPE_HSAIL_BINARY:
    // Compilation from ACL_TYPE_HSAIL_TEXT to ACL_TYPE_CG in cases:
    // 1. if the program is created with binary and contains only hsail text
    case ACL_TYPE_HSAIL_TEXT: {
        std::string curOptions = options->origOptionStr + hsailOptions();
        errorCode = aclCompile(dev().compiler(), binaryElf_,
            curOptions.c_str(), continueCompileFrom, ACL_TYPE_CG, nullptr);
        buildLog_ += aclGetCompilerLog(dev().compiler());
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error: BRIG code generation failed.\n";
            return false;
        }
        break;
    }
    case ACL_TYPE_CG:
        break;
    case ACL_TYPE_ISA:
        finalize = false;
        break;
    default:
        buildLog_ += "Error: The binary is incorrect or incomplete. Finalization to ISA couldn't be performed.\n";
        return false;
    }
    if (finalize) {
        std::string fin_options(options->origOptionStr + hsailOptions());
        // Append an option so that we can selectively enable a SCOption on CZ
        // whenever IOMMUv2 is enabled.
        if (dev().settings().svmFineGrainSystem_) {
            fin_options.append(" -sc-xnack-iommu");
        }
        errorCode = aclCompile(dev().compiler(), binaryElf_,
            fin_options.c_str(), ACL_TYPE_CG, ACL_TYPE_ISA, nullptr);
        buildLog_ += aclGetCompilerLog(dev().compiler());
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error: BRIG finalization to ISA failed.\n";
            return false;
        }
    }
    // ACL_TYPE_CG stage is not performed for offline compilation
    hsa_agent_t agent;
    agent.handle = 1;
    if (hsaLoad) {
        executable_ = loader_->CreateExecutable(HSA_PROFILE_FULL, NULL);
        if (executable_ == nullptr) {
            buildLog_ += "Error: Executable for AMD HSA Code Object isn't created.\n";
            return false;
        }
        size_t size = 0;
        hsa_code_object_t code_object;
        code_object.handle = reinterpret_cast<uint64_t>(aclExtractSection(dev().compiler(), binaryElf_, &size, aclTEXT, &errorCode));
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error: Extracting AMD HSA Code Object from binary failed.\n";
            return false;
        }
        hsa_status_t status = executable_->LoadCodeObject(agent, code_object, nullptr);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: AMD HSA Code Object loading failed.\n";
            return false;
        }
        status  = executable_->Freeze(nullptr);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: AMD HSA Code Object freeze failed.\n";
            return false;
        }
    }
    size_t kernelNamesSize = 0;
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_KERNEL_NAMES, nullptr, nullptr, &kernelNamesSize);
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Error: Querying of kernel names size from the binary failed.\n";
        return false;
    }
    if (kernelNamesSize > 0) {
        char* kernelNames = new char[kernelNamesSize];
        errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_KERNEL_NAMES, nullptr, kernelNames, &kernelNamesSize);
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error: Querying of kernel names from the binary failed.\n";
            delete kernelNames;
            return false;
        }
        std::vector<std::string> vKernels = splitSpaceSeparatedString(kernelNames);
        delete kernelNames;
        std::vector<std::string>::iterator it = vKernels.begin();
        bool dynamicParallelism = false;
        aclMetadata md;
        md.numHiddenKernelArgs = 0;
        size_t sizeOfnumHiddenKernelArgs = sizeof(md.numHiddenKernelArgs);
        for (it; it != vKernels.end(); ++it) {
            std::string kernelName(*it);
            std::string openclKernelName = device::Kernel::openclMangledName(kernelName);
            errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_NUM_KERNEL_HIDDEN_ARGS,
                openclKernelName.c_str(), &md.numHiddenKernelArgs, &sizeOfnumHiddenKernelArgs);
            if (errorCode != ACL_SUCCESS) {
                buildLog_ += "Error: Querying of kernel '" + openclKernelName +
                    "' extra arguments count from AMD HSA Code Object failed. Kernel initialization failed.\n";
                return false;
            }
            HSAILKernel *aKernel = new HSAILKernel(kernelName, this, options->origOptionStr + hsailOptions(),
                md.numHiddenKernelArgs);
            kernels()[kernelName] = aKernel;
            amd::hsa::loader::Symbol *sym = executable_->GetSymbol("", openclKernelName.c_str(), agent, 0);
            if (!sym) {
                buildLog_ += "Error: Getting kernel ISA code symbol '" + openclKernelName +
                    "' from AMD HSA Code Object failed. Kernel initialization failed.\n";
                return false;
            }
            if (!aKernel->init(sym, false)) {
                buildLog_ += "Error: Kernel '" + openclKernelName + "' initialization failed.\n";
                return false;
            }
            buildLog_ += aKernel->buildLog();
            aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
            dynamicParallelism |= aKernel->dynamicParallelism();
            // Find max scratch regs used in the program. It's used for scratch buffer preallocation
            // with dynamic parallelism, since runtime doesn't know which child kernel will be called
            maxScratchRegs_ = std::max(static_cast<uint>(aKernel->workGroupInfo()->scratchRegs_), maxScratchRegs_);
        }
        // Allocate kernel table for device enqueuing
        if (!isNull() && dynamicParallelism && !allocKernelTable()) {
            return false;
        }
    }
    // Save the binary in the interface class
    saveBinaryAndSetType(TYPE_EXECUTABLE);
    buildLog_ += aclGetCompilerLog(dev().compiler());
    return true;
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

bool
HSAILProgram::createBinary(amd::option::Options *options)
{
    return true;
}

bool
HSAILProgram::initClBinary()
{
    if (clBinary_ == nullptr) {
        clBinary_ = new ClBinaryHsa(static_cast<const Device &>(device()));
        if (clBinary_ == nullptr) {
            return false;
        }
    }
    return true;
}

void
HSAILProgram::releaseClBinary()
{
    if (clBinary_ != nullptr) {
        delete clBinary_;
        clBinary_ = nullptr;
    }
}

std::string
HSAILProgram::hsailOptions()
{
    std::string hsailOptions;
    // Set options for the standard device specific options
    // All our devices support these options now
    if (dev().settings().reportFMAF_) {
        hsailOptions.append(" -DFP_FAST_FMAF=1");
    }
    if (dev().settings().reportFMA_) {
        hsailOptions.append(" -DFP_FAST_FMA=1");
    }
#if !defined(WITH_LIGHTNING_COMPILER)
    if (!dev().settings().singleFpDenorm_) {
        hsailOptions.append(" -cl-denorms-are-zero");
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)

    // Check if the host is 64 bit or 32 bit
    LP64_ONLY(hsailOptions.append(" -m64"));

    // Append each extension supported by the device
    std::string token;
    std::istringstream iss("");
    iss.str(device().info().extensions_);
    while (getline(iss, token, ' ')) {
        if (!token.empty()) {
            hsailOptions.append(" -D");
            hsailOptions.append(token);
            hsailOptions.append("=1");
        }
    }
    return hsailOptions;
}

bool
HSAILProgram::allocKernelTable()
{
    uint size = kernels().size() * sizeof(size_t);

    kernels_ = new pal::Memory(dev(), size);
    // Initialize kernel table
    if ((kernels_ == nullptr) || !kernels_->create(Resource::RemoteUSWC)) {
        delete kernels_;
        return false;
    }
    else {
        size_t* table = reinterpret_cast<size_t*>(
            kernels_->map(nullptr, pal::Resource::WriteOnly));
        for (auto& it : kernels()) {
            HSAILKernel* kernel = static_cast<HSAILKernel*>(it.second);
            table[kernel->index()] = static_cast<size_t>(kernel->gpuAqlCode());
        }
        kernels_->unmap(nullptr);
    }
    return true;
}

void
HSAILProgram::fillResListWithKernels(
    std::vector<const Memory*>& memList) const
{
    memList.push_back(&codeSegGpu());
}

const aclTargetInfo &
HSAILProgram::info(const char * str)
{
#if defined(WITH_LIGHTNING_COMPILER)
    assert(!"Should not reach here");
#else // !defined(WITH_LIGHTNING_COMPILER)
    acl_error err;
    std::string arch = "hsail";
    if (dev().settings().use64BitPtr_) {
        arch = "hsail64";
    }
    info_ = aclGetTargetInfo(arch.c_str(), ( str && str[0] == '\0' ?
        dev().hwInfo()->targetName_ : str ), &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclGetTargetInfo failed");
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)
    return info_;
}

bool
HSAILProgram::saveBinaryAndSetType(type_t type)
{
#if defined(WITH_LIGHTNING_COMPILER)
    assert(!"Should not reach here");
#else // !defined(WITH_LIGHTNING_COMPILER)
    //Write binary to memory
    if (rawBinary_ != nullptr) {
        //Free memory containing rawBinary
        aclFreeMem(binaryElf_, rawBinary_);
        rawBinary_ = nullptr;
    }
    size_t  size = 0;
    if (aclWriteToMem(binaryElf_, &rawBinary_, &size) != ACL_SUCCESS) {
        buildLog_ += "Failed to write binary to memory \n";
        return false;
    }
    setBinary(static_cast<char*>(rawBinary_), size);
    //Set the type of binary
    setType(type);
#endif // !defined(WITH_LIGHTNING_COMPILER)
    return true;
}

hsa_isa_t PALHSALoaderContext::IsaFromName(const char *name) {
    hsa_isa_t isa = {0};
    if (!strcmp(Gfx700, name)) { isa.handle = gfx700; return isa; }
    if (!strcmp(Gfx701, name)) { isa.handle = gfx701; return isa; }
    if (!strcmp(Gfx800, name)) { isa.handle = gfx800; return isa; }
    if (!strcmp(Gfx801, name)) { isa.handle = gfx801; return isa; }
    if (!strcmp(Gfx804, name)) { isa.handle = gfx804; return isa; }
    if (!strcmp(Gfx810, name)) { isa.handle = gfx810; return isa; }
    if (!strcmp(Gfx900, name)) { isa.handle = gfx900; return isa; }
    if (!strcmp(Gfx901, name)) { isa.handle = gfx901; return isa; }
    return isa;
}

bool PALHSALoaderContext::IsaSupportedByAgent(hsa_agent_t agent, hsa_isa_t isa) {
    switch (program_->dev().hwInfo()->gfxipVersion_) {
    default:
        LogError("Unsupported gfxip version");
        return false;
    case gfx700: case gfx701: case gfx702:
        // gfx701 only differs from gfx700 by faster fp operations and can be loaded on either device.
        return isa.handle == gfx700 || isa.handle == gfx701;
    case gfx800:
        return isa.handle == gfx800;
    case gfx801:
        return isa.handle == gfx801;
    case gfx804:
        // gfx800 ISA has only sgrps limited and can be loaded.
        // gfx801 ISA has XNACK limitations and can be loaded.
        return isa.handle == gfx800 || isa.handle == gfx801 || isa.handle == gfx804;
    case gfx810:
            return isa.handle == gfx810;
    case gfx900: case gfx901:
        return isa.handle == gfx900 || isa.handle == gfx901;
    }
}

void* PALHSALoaderContext::SegmentAlloc(amdgpu_hsa_elf_segment_t segment,
    hsa_agent_t agent, size_t size, size_t align, bool zero) {
    assert(size);
    assert(align);
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT:
        return AgentGlobalAlloc(agent, size, align, zero);
    case AMDGPU_HSA_SEGMENT_CODE_AGENT:
        return KernelCodeAlloc(size, align, zero);
    default:
        assert(false); return 0;
    }
}

bool PALHSALoaderContext::SegmentCopy(amdgpu_hsa_elf_segment_t segment,
    hsa_agent_t agent, void* dst, size_t offset, const void* src, size_t size) {
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT:
      return AgentGlobalCopy(dst, offset, src, size);
    case AMDGPU_HSA_SEGMENT_CODE_AGENT:
      return KernelCodeCopy(dst, offset, src, size);
    default:
      assert(false); return false;
    }
}

void PALHSALoaderContext::SegmentFree(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t size)
{
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT: AgentGlobalFree(seg, size); break;
    case AMDGPU_HSA_SEGMENT_CODE_AGENT: KernelCodeFree(seg, size); break;
    default:
        assert(false); return;
    }
}

void* PALHSALoaderContext::SegmentAddress(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t offset)
{
    assert(seg);
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT: {
    case AMDGPU_HSA_SEGMENT_CODE_AGENT:
        if (!program_->isNull()) {
            pal::Memory *gpuMem = reinterpret_cast<pal::Memory*>(seg);
            return reinterpret_cast<void*>(gpuMem->vmAddress() + offset);
        }
        else {
            return reinterpret_cast<address>(seg) + offset;
        }
    }
    default:
        assert(false); return nullptr;
    }
}

void* PALHSALoaderContext::SegmentHostAddress(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t offset)
{
    void* host = nullptr;
    assert(seg);
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_CODE_AGENT:
        host = program_->codeSegCpu() + offset;
        break;
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT:
    default:
        break;
    }
    return host;
}

bool PALHSALoaderContext::SegmentFreeze(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t size)
{
    assert(seg);
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT:
        return true;
    case AMDGPU_HSA_SEGMENT_CODE_AGENT: {
        if (program_->isNull()) {
            return true;
        }

        const pal::Memory& mem = program_->codeSegGpu();
        constexpr bool WaitForCopy = true;
        mem.writeRawData(*mem.dev().xferQueue(), 0, size, program_->codeSegCpu(), WaitForCopy);
        return true;
    }
    default:
        assert(false); 
        return false;
    }
}

hsa_status_t PALHSALoaderContext::SamplerCreate(
    hsa_agent_t agent,
    const hsa_ext_sampler_descriptor_t *sampler_descriptor,
    hsa_ext_sampler_t *sampler_handle)
{
    if (!agent.handle) {
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    if (!sampler_descriptor || !sampler_handle) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    if (program_->isNull()) {
        // Offline compilation. Provide a fake handle to avoid an assert
        sampler_handle->handle = 1;
        return HSA_STATUS_SUCCESS;
    }
    uint32_t state = 0;
    switch (sampler_descriptor->coordinate_mode) {
        case HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED: state = amd::Sampler::StateNormalizedCoordsFalse; break;
        case HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED:   state = amd::Sampler::StateNormalizedCoordsTrue; break;
        default:
            assert(false);
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    switch (sampler_descriptor->filter_mode) {
        case HSA_EXT_SAMPLER_FILTER_MODE_NEAREST: state |= amd::Sampler::StateFilterNearest; break;
        case HSA_EXT_SAMPLER_FILTER_MODE_LINEAR:  state |= amd::Sampler::StateFilterLinear; break;
        default:
            assert(false);
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    }
    switch (sampler_descriptor->address_mode) {
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:   state |= amd::Sampler::StateAddressClampToEdge; break;
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER: state |= amd::Sampler::StateAddressClamp; break;
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT:          state |= amd::Sampler::StateAddressRepeat; break;
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT: state |= amd::Sampler::StateAddressMirroredRepeat; break;
        case HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED: state |= amd::Sampler::StateAddressNone; break;
        default:
            assert(false);
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    pal::Sampler* sampler = new pal::Sampler(program_->dev());
    if (!sampler || !sampler->create(state)) {
        delete sampler;
        return HSA_STATUS_ERROR;
    }
    program_->addSampler(sampler);
    sampler_handle->handle = sampler->hwSrd();
    return HSA_STATUS_SUCCESS;
}

hsa_status_t PALHSALoaderContext::SamplerDestroy(
    hsa_agent_t agent, hsa_ext_sampler_t sampler_handle)
{
    if (!agent.handle) {
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    if (!sampler_handle.handle) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    return HSA_STATUS_SUCCESS;
}

address PALHSALoaderContext::CpuMemAlloc(size_t size, size_t align, bool zero)
{
    assert(size);
    assert(align);
    assert(sizeof(void*) == 8 || sizeof(void*) == 4);

    void* ptr = amd::Os::alignedMalloc(size, align);
    if (zero) {
        memset(ptr, 0, size);
    }
    return reinterpret_cast<address>(ptr);
}

bool PALHSALoaderContext::CpuMemCopy(void *dst, size_t offset, const void* src, size_t size)
{
    amd::Os::fastMemcpy((char*)dst + offset, src, size);
    return true;
}

void* PALHSALoaderContext::GpuMemAlloc(size_t size, size_t align, bool zero) {
    assert(size);
    assert(align);
    assert(sizeof(void*) == 8 || sizeof(void*) == 4);
    if (program_->isNull()) {
        return CpuMemAlloc(size, align, zero);
    }

    pal::Memory* mem = new pal::Memory(program_->dev(), amd::alignUp(size, align));
    if (!mem || !mem->create(pal::Resource::Local)) {
        delete mem;
        return nullptr;
    }
    assert(program_->dev().xferQueue());
    if (zero && !program_->isInternal()) {
        char pattern = 0;
        program_->dev().xferMgr().fillBuffer(*mem, &pattern, sizeof(pattern), amd::Coord3D(0), amd::Coord3D(size));
    }
    program_->addGlobalStore(mem);
    program_->setGlobalVariableTotalSize(program_->globalVariableTotalSize() + size);
    return mem;
}

bool PALHSALoaderContext::GpuMemCopy(void *dst, size_t offset, const void *src, size_t size) {
    if (!dst || !src || dst == src) {
        return false;
    }
    if (0 == size) {
        return true;
    }
    if (program_->isNull()) {
        CpuMemCopy(dst, offset, src, size);
        return true;
    }
    assert(program_->dev().xferQueue());
    pal::Memory* mem = reinterpret_cast<pal::Memory*>(dst);
    constexpr bool WaitForCopy = true;
    mem->writeRawData(*mem->dev().xferQueue(), offset, size, src, WaitForCopy);
    return true;
}

void PALHSALoaderContext::GpuMemFree(void *ptr, size_t size)
{
    if (program_->isNull()) {
       CpuMemFree(ptr, size);
    }
    else {
        delete reinterpret_cast<pal::Memory*>(ptr);
    }
}

void* PALHSALoaderContext::KernelCodeAlloc(
    size_t size, size_t align, bool zero)
{
    address host = CpuMemAlloc(size, align, zero);
    pal::Memory* mem = nullptr;

    if (!program_->isNull()) {
        mem = new pal::Memory(program_->dev(), amd::alignUp(size, align));
        if (!mem || !mem->create(pal::Resource::Local)) {
            delete mem;
            mem = nullptr;
        }
    }
    program_->setCodeObjects(mem, host);
    return ((host == nullptr || mem == nullptr) ? nullptr : mem);
}

bool PALHSALoaderContext::KernelCodeCopy(void *dst, size_t offset, const void *src, size_t size)
{
    if (!dst || !src || dst == src) {
        return false;
    }
    if (0 == size) {
        return true;
    }
    if (program_->isNull()) {
        return CpuMemCopy(dst, offset, src, size);
    }
    assert(program_->dev().xferQueue());
    pal::Memory* mem = reinterpret_cast<pal::Memory*>(dst);
    if (mem == &program_->codeSegGpu()) {
        return CpuMemCopy(program_->codeSegCpu(), offset, src, size);
    }
    assert(!"The segement doesn't match code segment in the program!");
    return false;
}

void PALHSALoaderContext::KernelCodeFree(void *ptr, size_t size)
{
    CpuMemFree(program_->codeSegCpu(), size);
    if (!program_->isNull()) {
        delete reinterpret_cast<pal::Memory*>(ptr);
    }
    program_->setCodeObjects(nullptr, nullptr);
}

#if defined(WITH_LIGHTNING_COMPILER)

static hsa_status_t
GetKernelNamesCallback(
    hsa_executable_t hExec,
    hsa_executable_symbol_t hSymbol,
    void *data)
{
    auto symbol = Symbol::Object(hSymbol);
    auto symbolNameList = reinterpret_cast<std::vector<std::string>*>(data);

    hsa_symbol_kind_t type;
    if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &type)) {
        return HSA_STATUS_ERROR;
    }

    if (type == HSA_SYMBOL_KIND_KERNEL) {
        uint32_t length;
        if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &length)) {
            return HSA_STATUS_ERROR;
        }

        char* name = (char*) alloca(length+1);
        if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_NAME, name)) {
            return HSA_STATUS_ERROR;
        }
        name[length] = '\0';

        symbolNameList->push_back(std::string(name));
    }
    return HSA_STATUS_SUCCESS;
}

bool
LightningProgram::linkImpl(amd::option::Options *options)
{
    using namespace amd::opencl_driver;
    internal_ = (compileOptions_.find("-cl-internal-kernel") !=
        std::string::npos) ? true : false;

    aclType continueCompileFrom = llvmBinary_.empty()
        ? getNextCompilationStageFromBinary(options)
        : ACL_TYPE_LLVMIR_BINARY;

    if (continueCompileFrom == ACL_TYPE_ISA) {
        binary_t isa = binary();
        if ((isa.first != NULL) && (isa.second > 0)) {
            return setKernels(options, (void*) isa.first, isa.second );
        }
        else {
            buildLog_ += "Error: code object is empty \n" ;
            return false;
        }
        return true;
    }
    if (continueCompileFrom != ACL_TYPE_LLVMIR_BINARY) {
        buildLog_ += "Error while Codegen phase: the binary is incomplete \n" ;
        return false;
    }

    std::auto_ptr<Compiler> C(newCompilerInstance());
    // call LinkLLVMBitcode
    std::vector<Data*> inputs;

    // open the input IR source
    Data* input = C->NewBufferReference(
        DT_LLVM_BC, llvmBinary_.data(), llvmBinary_.size());

    if (!input) {
        buildLog_ += "Error: Failed to open the compiled program.\n";
        return false;
    }

    inputs.push_back(input); //< must be the first input

    // open the bitcode libraries
    Data* opencl_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) opencl_amdgcn, opencl_amdgcn_size);
    Data* ocml_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) ocml_amdgcn, ocml_amdgcn_size);
    Data* ockl_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) ockl_amdgcn, ockl_amdgcn_size);
    Data* irif_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) irif_amdgcn, irif_amdgcn_size);

    if (!opencl_bc || !ocml_bc || !ockl_bc || !irif_bc) {
        buildLog_ += "Error: Failed to open the bitcode library.\n";
        return false;
    }

    inputs.push_back(opencl_bc); // depends on oclm & ockl
    inputs.push_back(ockl_bc); // depends on irif
    inputs.push_back(ocml_bc); // depends on irif
    inputs.push_back(irif_bc);

    // open the control functions
    std::pair<const void*, size_t> isa_version;
    switch (dev().hwInfo()->gfxipVersion_) {
    case 700: isa_version = std::make_pair(
            oclc_isa_version_700_amdgcn, oclc_isa_version_700_amdgcn_size);
        break;
    case 701: isa_version = std::make_pair(
            oclc_isa_version_701_amdgcn, oclc_isa_version_701_amdgcn_size);
        break;
    case 800: isa_version = std::make_pair(
            oclc_isa_version_800_amdgcn, oclc_isa_version_800_amdgcn_size);
        break;
    case 801: isa_version = std::make_pair(
            oclc_isa_version_801_amdgcn, oclc_isa_version_801_amdgcn_size);
        break;
    case 802: isa_version = std::make_pair(
            oclc_isa_version_802_amdgcn, oclc_isa_version_802_amdgcn_size);
        break;
    case 803: case 804: isa_version = std::make_pair(
            oclc_isa_version_803_amdgcn, oclc_isa_version_803_amdgcn_size);
        break;
    case 810: isa_version = std::make_pair(
            oclc_isa_version_810_amdgcn, oclc_isa_version_810_amdgcn_size);
        break;
    default:
        buildLog_ += "Error: Linking for this device is not supported\n";
        return false;
    }

    Data* isa_version_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) isa_version.first, isa_version.second);

    if (!isa_version_bc) {
        buildLog_ += "Error: Failed to open the control functions.\n";
        return false;
    }

    inputs.push_back(isa_version_bc);

    auto correctly_rounded_sqrt = (options->oVariables->FP32RoundDivideSqrt)
        ? std::make_pair(
            oclc_correctly_rounded_sqrt_on_amdgcn,
            oclc_correctly_rounded_sqrt_on_amdgcn_size)
        : std::make_pair(
            oclc_correctly_rounded_sqrt_off_amdgcn,
            oclc_correctly_rounded_sqrt_off_amdgcn_size);

    auto daz_opt = (dev().hwInfo()->gfxipVersion_ < 900
                 || options->oVariables->DenormsAreZero)
        ? std::make_pair(
            oclc_daz_opt_on_amdgcn,
            oclc_daz_opt_on_amdgcn_size)
        : std::make_pair(
            oclc_daz_opt_off_amdgcn,
            oclc_daz_opt_off_amdgcn_size);

    auto finite_only = (options->oVariables->FiniteMathOnly
                     || options->oVariables->FastRelaxedMath)
        ? std::make_pair(
            oclc_finite_only_on_amdgcn,
            oclc_finite_only_on_amdgcn_size)
        : std::make_pair(
            oclc_finite_only_off_amdgcn,
            oclc_finite_only_off_amdgcn_size);

    auto unsafe_math = (options->oVariables->UnsafeMathOpt
                     || options->oVariables->FastRelaxedMath)
        ? std::make_pair(
            oclc_unsafe_math_on_amdgcn,
            oclc_unsafe_math_on_amdgcn_size)
        : std::make_pair(
            oclc_unsafe_math_off_amdgcn,
            oclc_unsafe_math_off_amdgcn_size);

    Data* correctly_rounded_sqrt_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) correctly_rounded_sqrt.first, correctly_rounded_sqrt.second);
    Data* daz_opt_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) daz_opt.first, daz_opt.second);
    Data* finite_only_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) finite_only.first, finite_only.second);
    Data* unsafe_math_bc = C->NewBufferReference(DT_LLVM_BC,
        (const char*) unsafe_math.first, unsafe_math.second);

    if (!correctly_rounded_sqrt_bc || !daz_opt_bc || !finite_only_bc || !unsafe_math_bc) {
        buildLog_ += "Error: Failed to open the control functions.\n";
        return false;
    }

    inputs.push_back(correctly_rounded_sqrt_bc);
    inputs.push_back(daz_opt_bc);
    inputs.push_back(finite_only_bc);
    inputs.push_back(unsafe_math_bc);

    // open the linked output
    std::vector<std::string> linkOptions;
    amd::opencl_driver::Buffer* linked_bc = C->NewBuffer(DT_LLVM_BC);

    if (!linked_bc) {
        buildLog_ += "Error: Failed to open the linked program.\n";
        return false;
    }

    bool ret = C->LinkLLVMBitcode(inputs, linked_bc, linkOptions);
    buildLog_ += C->Output();
    if (!ret) {
        buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
        return false;
    }

    if (options->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
        std::ofstream f(options->getDumpFileName("_linked.bc").c_str(), std::ios::trunc);
        if(f.is_open()) {
            f.write(linked_bc->Buf().data(), linked_bc->Size());
        } else {
            buildLog_ +=
                "Warning: opening the file to dump the linked IR failed.\n";
        }
    }

    inputs.clear();
    inputs.push_back(linked_bc);

    amd::opencl_driver::Buffer* out_exec = C->NewBuffer(DT_EXECUTABLE);
    if (!out_exec) {
        buildLog_ += "Error: Failed to create the linked executable.\n";
        return false;
    }

    std::string codegenOptions(options->llvmOptions);

    // Set the machine target
#if 0
    std::ostringstream mCPU;
    mCPU << " -mcpu=gfx" << dev().hwInfo()->gfxipVersion_;
    codegenOptions.append(mCPU.str());
#else
    codegenOptions.append(" -mcpu=");
    codegenOptions.append(dev().hwInfo()->machineTarget_);
#endif

    // Set the -O#
    std::ostringstream optLevel;
    optLevel << "-O" << options->oVariables->OptLevel;
    codegenOptions.append(" ").append(optLevel.str());

    // Tokenize the options string into a vector of strings
    std::istringstream strstr(codegenOptions);
    std::istream_iterator<std::string> sit(strstr), end;
    std::vector<std::string> params(sit, end);

    ret = C->CompileAndLinkExecutable(inputs, out_exec, params);
    buildLog_ += C->Output();
    if (!ret) {
        buildLog_ += "Error: Creating the executable failed: Compiling LLVM IRs to exe.\n";
        return false;
    }

    if (options->isDumpFlagSet(amd::option::DUMP_O)) {
        std::ofstream f(options->getDumpFileName(".so").c_str(), std::ios::trunc);
        if(f.is_open()) {
            f.write(out_exec->Buf().data(), out_exec->Size());
        } else {
            buildLog_ +=
                "Warning: opening the file to dump the code object failed.\n";
        }
    }

    return setKernels(options, out_exec->Buf().data(), out_exec->Size());
}

bool
LightningProgram::setKernels(
    amd::option::Options *options,
    void* binary,
    size_t size
    )
{
    hsa_agent_t agent;
    agent.handle = 1;

    executable_ = loader_->CreateExecutable(HSA_PROFILE_FULL, NULL);
    if (executable_ == nullptr) {
        buildLog_ += "Error: Executable for AMD HSA Code Object isn't created.\n";
        return false;
    }

    hsa_code_object_t code_object;
    code_object.handle = reinterpret_cast<uint64_t>(binary);

    hsa_status_t status = executable_->LoadCodeObject(agent, code_object, nullptr);
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: AMD HSA Code Object loading failed.\n";
        return false;
    }

    status = executable_->Freeze(nullptr);
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Freezing the executable failed: ";
        return false;
    }

    size_t progvarsTotalSize = 0;

    // Begin the Elf image from memory
    Elf* e = elf_memory((char*) binary, size, NULL);
    if (elf_kind(e) != ELF_K_ELF) {
        buildLog_ += "Error while reading the ELF program binary\n";
        return false;
    }
    size_t shstrndx;
    if (elf_getshdrstrndx(e, &shstrndx) != 0) {
        buildLog_ += "Error while reading the ELF program binary\n";
        return false;
    }

    // Iterate over the sections
    for (Elf_Scn* scn = elf_nextscn(e, 0); scn; scn = elf_nextscn(e, scn)) {
        GElf_Shdr shdr;
        if (gelf_getshdr(scn, &shdr) != &shdr) {
            continue;
        }
        // Skip non-program sections
        if (shdr.sh_type != SHT_PROGBITS) {
            continue;
        }
        // Accumulate the size of A & !X sections
        if ((shdr.sh_flags & SHF_ALLOC) && !(shdr.sh_flags & SHF_EXECINSTR)) {
            progvarsTotalSize += shdr.sh_size;
        }
        // Check if this is the metadata section
        const char* name = elf_strptr(e, shstrndx , shdr.sh_name);
        if (name && !strcmp(name, ".AMDGPU.runtime_metadata")) {
            // Assume a single Elf_Data, the parser will fail if it isn't
            Elf_Data* data = elf_getdata(scn, NULL);
            if (!data) {
                buildLog_ += "Error while reading ELF program binary " \
                    "runtime metadata section\n";
                return false;
            }

            metadata_ = new amd::hsa::code::Program::Metadata();
            if (!metadata_ || !metadata_->ReadFrom(data->d_buf, data->d_size)) {
                buildLog_ += "Error while parsing ELF program binary " \
                    "runtime metadata section\n";
                return false;
            }
        }
    }

    elf_end(e);

    if (!metadata_) {
        buildLog_ += "Error: runtime metadata section not present in " \
            "ELF program binary\n";
        return false;
    }

    setGlobalVariableTotalSize(progvarsTotalSize);

    // Get the list of kernels
    std::vector<std::string> kernelNameList;
    status = executable_->IterateSymbols(GetKernelNamesCallback,
        (void *) &kernelNameList );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Failed to get kernel names\n";
        return false;
    }

    for (auto &kernelName : kernelNameList) {
        auto kernel = new LightningKernel(
            kernelName, this, options->origOptionStr + hsailOptions());

        kernels()[kernelName] = kernel;

        auto symbol = executable_->GetSymbol("", kernelName.c_str(), agent, 0);
        if (!symbol) {
            buildLog_ += "Error: Getting kernel symbol '" + kernelName
                + "' from AMD HSA Code Object failed. " \
                "Kernel initialization failed.\n";
            return false;
        }
        if (!kernel->init(symbol)) {
            buildLog_ += "Error: Kernel '" + kernelName
                + "' initialization failed.\n";
            return false;
        }
        buildLog_ += kernel->buildLog();

        kernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);

        // Find max scratch regs used in the program. It's used for scratch buffer preallocation
        // with dynamic parallelism, since runtime doesn't know which child kernel will be called
        maxScratchRegs_ = std::max(static_cast<uint>(kernel->workGroupInfo()->scratchRegs_), maxScratchRegs_);
    }

    // Allocate kernel table for device enqueuing
    if (!isNull() && false/*dynamicParallelism*/ && !allocKernelTable()) {
        return false;
    }

    // Save the binary and type
    clBinary()->saveBIFBinary((char*)binary, size);
    setType(TYPE_EXECUTABLE);

    return true;
}

LightningProgram::~LightningProgram()
{
    delete metadata_;
}

#endif // defined(WITH_LIGHTNING_COMPILER)

} // namespace pal
