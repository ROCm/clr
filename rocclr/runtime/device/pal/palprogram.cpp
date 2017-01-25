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
#include "AMDGPUPTNote.h"
#include "AMDGPURuntimeMetadata.h"
#include "driver/AmdCompiler.h"
#include "libraries.amdgcn.inc"
#include "gelf.h"
#endif // !defined(WITH_LIGHTNING_COMPILER)

namespace pal {

Segment::Segment()
    : gpuAccess_(nullptr)
    , cpuAccess_(nullptr)
{}

Segment::~Segment()
{
    delete gpuAccess_;
    if (cpuAccess_ != nullptr) {
        cpuAccess_->unmap(nullptr);
        delete cpuAccess_;
    }
}

bool
Segment::alloc(
    HSAILProgram& prog, amdgpu_hsa_elf_segment_t segment,
    size_t size, size_t align, bool zero)
{
    align = amd::alignUp(align, sizeof(uint32_t));
    gpuAccess_ = new pal::Memory(prog.dev(), amd::alignUp(size, align));
    if ((gpuAccess_ == nullptr) || !gpuAccess_->create(pal::Resource::Local)) {
        delete gpuAccess_;
        gpuAccess_ = nullptr;
        return false;
    }
    if (segment == AMDGPU_HSA_SEGMENT_CODE_AGENT) {
        cpuAccess_ = new pal::Memory(prog.dev(), amd::alignUp(size, align));
        if ((cpuAccess_ == nullptr) || !cpuAccess_->create(pal::Resource::Remote)) {
            delete cpuAccess_;
            cpuAccess_ = nullptr;
            return false;
        }
        void* ptr = cpuAccess_->map(nullptr, 0);
        if (zero) {
            memset(ptr, 0, size);
        }
    }

    if (zero && !prog.isInternal()) {
        char pattern = 0;
        prog.dev().xferMgr().fillBuffer(*gpuAccess_, &pattern, sizeof(pattern),
            amd::Coord3D(0), amd::Coord3D(size));
    }

    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT:
        prog.addGlobalStore(gpuAccess_);
        prog.setGlobalVariableTotalSize(prog.globalVariableTotalSize() + size);
        break;
    case AMDGPU_HSA_SEGMENT_CODE_AGENT:
        prog.setCodeObjects(gpuAccess_, cpuAccess_->data());
        break;
    default:
        break;
    }
    return true;
}

void 
Segment::copy(size_t offset, const void* src, size_t size)
{
    if (cpuAccess_ != nullptr) {
        amd::Os::fastMemcpy(cpuAddress(offset), src, size);
    }
    else {
        VirtualGPU& gpu = *gpuAccess_->dev().xferQueue();
        Memory& xferBuf = gpuAccess_->dev().xferWrite().acquire();
        size_t tmpSize = std::min(static_cast<size_t>(xferBuf.vmSize()), size);
        size_t srcOffs = 0;
        while (size != 0) {
            xferBuf.hostWrite(&gpu,
                reinterpret_cast<const_address>(src) + srcOffs, 0, tmpSize);
            bool result = xferBuf.partialMemCopyTo(gpu,
                0, (offset + srcOffs), tmpSize, *gpuAccess_, false, true);
            size -= tmpSize;
            srcOffs += tmpSize;
            tmpSize = std::min(static_cast<size_t>(xferBuf.vmSize()), size);
        }
        gpu.releaseMemObjects();
        gpu.waitAllEngines();
    }
}

bool
Segment::freeze(bool destroySysmem)
{
    VirtualGPU& gpu = *gpuAccess_->dev().xferQueue();
    bool result = true;
    if (cpuAccess_ != nullptr) {
        assert(gpuAccess_->size() == cpuAccess_->size() && "Backing store size mismatch!");
        result = cpuAccess_->partialMemCopyTo(gpu,
            0, 0, gpuAccess_->size(), *gpuAccess_, false, true);
        gpu.releaseMemObjects();
        gpu.waitAllEngines();
    }
    assert(!destroySysmem || (cpuAccess_ == nullptr));
    return result;
}

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
    isNull_ = true;
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
        std::string curOptions = options->origOptionStr + hsailOptions(options);
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
        std::string fin_options(options->origOptionStr + hsailOptions(options));
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
        for (it; it != vKernels.end(); ++it) {
            std::string kernelName(*it);
            std::string openclKernelName = device::Kernel::openclMangledName(kernelName);

            HSAILKernel *aKernel = new HSAILKernel(kernelName, this, options->origOptionStr + hsailOptions(options));
            kernels()[kernelName] = aKernel;

            amd::hsa::loader::Symbol *sym = executable_->GetSymbol(openclKernelName.c_str(), &agent);
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
HSAILProgram::hsailOptions(amd::option::Options* options)
{
    std::string hsailOptions;

    hsailOptions.append(" -D__AMD__=1");

    hsailOptions.append(" -D__").append(device().info().name_).append("__=1");
    hsailOptions.append(" -D__").append(device().info().name_).append("=1");

    int major, minor;
    ::sscanf(device().info().version_, "OpenCL %d.%d ", &major, &minor);

#ifdef WITH_LIGHTNING_COMPILER
    std::stringstream ss;
    ss << " -D__OPENCL_VERSION__=" << (major * 100 + minor * 10);
    hsailOptions.append(ss.str());
#endif

    if (device().info().imageSupport_ && options->oVariables->ImageSupport) {
        hsailOptions.append(" -D__IMAGE_SUPPORT__=1");
    }

    // Set options for the standard device specific options
    // All our devices support these options now
    if (dev().settings().reportFMAF_) {
        hsailOptions.append(" -DFP_FAST_FMAF=1");
    }
    if (dev().settings().reportFMA_) {
        hsailOptions.append(" -DFP_FAST_FMA=1");
    }

    uint clcStd = (options->oVariables->CLStd[2] - '0') * 100
        + (options->oVariables->CLStd[4] - '0') * 10;

    if (clcStd >= 200) {
        std::stringstream opts;
        //Add only for CL2.0 and later
        opts << " -D" << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE="
            << device().info().maxGlobalVariableSize_;
        hsailOptions.append(opts.str());
    }

#if !defined(WITH_LIGHTNING_COMPILER)
    if (!dev().settings().singleFpDenorm_) {
        hsailOptions.append(" -cl-denorms-are-zero");
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)

    // Check if the host is 64 bit or 32 bit
    LP64_ONLY(hsailOptions.append(" -m64"));

    // Tokenize the extensions string into a vector of strings
    std::istringstream istrstr(device().info().extensions_);
    std::istream_iterator<std::string> sit(istrstr), end;
    std::vector<std::string> extensions(sit, end);

#if defined(WITH_LIGHTNING_COMPILER)
    // FIXME_lmoriche: opencl-c.h defines 'cl_khr_depth_images', so
    // remove it from the command line. Should we fix opencl-c.h?
    auto found = std::find(extensions.begin(), extensions.end(),
        "cl_khr_depth_images");
    if (found != extensions.end()) {
        extensions.erase(found);
    }

    if (!extensions.empty()) {
        std::ostringstream clext;

        clext << " -Xclang -cl-ext=+";
        std::copy(extensions.begin(), extensions.end() - 1,
            std::ostream_iterator<std::string>(clext, ",+"));
        clext << extensions.back();

        hsailOptions.append(clext.str());
    }
#else // !defined(WITH_LIGHTNING_COMPILER)
    for (auto e : extensions) {
        hsailOptions.append(" -D").append(e).append("=1");
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)

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

void* PALHSALoaderContext::SegmentAlloc(
    amdgpu_hsa_elf_segment_t segment,
    hsa_agent_t agent, size_t size, size_t align, bool zero)
{
    assert(size);
    assert(align);
    if (program_->isNull()) {
        void* ptr = amd::Os::alignedMalloc(size, align);
        if (zero) {
            memset(ptr, 0, size);
        }
        return ptr;
    }
    Segment* seg  = new Segment();
    if (seg != nullptr && !seg->alloc(*program_, segment, size, align, zero)) {
        return nullptr;
    }
    return seg;
}

bool PALHSALoaderContext::SegmentCopy(amdgpu_hsa_elf_segment_t segment,
    hsa_agent_t agent, void* dst, size_t offset, const void* src, size_t size)
{
    if (program_->isNull()) {
        amd::Os::fastMemcpy(reinterpret_cast<address>(dst) + offset, src, size);
        return true;
    }
    Segment* s = reinterpret_cast<Segment*>(dst);
    s->copy(offset, src, size);
    return true;
}

void PALHSALoaderContext::SegmentFree(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t size)
{
    if (program_->isNull()) {
        amd::Os::alignedFree(seg);
    }
    else {
        Segment* s = reinterpret_cast<Segment*>(seg);
        delete s ;
    }
}

void* PALHSALoaderContext::SegmentAddress(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t offset)
{
    assert(seg);
    if (program_->isNull()) {
        return (reinterpret_cast<address>(seg) + offset);
    }
    Segment* s = reinterpret_cast<Segment*>(seg);
    return reinterpret_cast<void*>(s->gpuAddress(offset));
}

void* PALHSALoaderContext::SegmentHostAddress(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t offset)
{
    assert(seg);
    if (program_->isNull()) {
        return (reinterpret_cast<address>(seg) + offset);
    }
    Segment* s = reinterpret_cast<Segment*>(seg);
    return s ->cpuAddress(offset);
}

bool PALHSALoaderContext::SegmentFreeze(
    amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg, size_t size)
{
    if (program_->isNull()) {
        return true;
    }
    Segment* s = reinterpret_cast<Segment*>(seg);
    return s->freeze((segment == AMDGPU_HSA_SEGMENT_CODE_AGENT) ? false : true);
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

aclType
LightningProgram::getCompilationStagesFromBinary(
    std::vector<aclType>& completeStages,
    bool& needOptionsCheck
    )
{
    completeStages.clear();
    aclType from = ACL_TYPE_DEFAULT;
    needOptionsCheck = true;

    bool containsLlvmirText = (type() == TYPE_COMPILED);
    bool containsShaderIsa = (type() == TYPE_EXECUTABLE);
    bool containsOpts = !(compileOptions_.empty() && linkOptions_.empty());

    if (containsLlvmirText && containsOpts) {
        completeStages.push_back(from);
        from = ACL_TYPE_LLVMIR_BINARY;
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
    case ACL_TYPE_ISA:
        // do not check options, if LLVMIR is absent or might be absent or options are absent
        if (curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
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
}


aclType
LightningProgram::getNextCompilationStageFromBinary(amd::option::Options* options)
{
    aclType continueCompileFrom = ACL_TYPE_DEFAULT;
    binary_t binary = this->binary();

    // If the binary already exists
    if ((binary.first != NULL) && (binary.second > 0)) {
        void *mem = const_cast<void *>(binary.first);

        // save the current options
        std::string sCurCompileOptions = compileOptions_;
        std::string sCurLinkOptions = linkOptions_;
        std::string sCurOptions = compileOptions_ + linkOptions_;

        // Saving binary in the interface class,
        // which also load compile & link options from binary
        setBinary(static_cast<char*>(mem), binary.second);

        // Calculate the next stage to compile from, based on sections in binaryElf_;
        // No any validity checks here
        std::vector<aclType> completeStages;
        bool needOptionsCheck = true;
        continueCompileFrom = getCompilationStagesFromBinary(completeStages, needOptionsCheck);
        if (!options || !needOptionsCheck) {
            return continueCompileFrom;
        }
        bool recompile = false;
        //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
        switch (continueCompileFrom) {
        case ACL_TYPE_ISA: {
            // Compare options loaded from binary with current ones, recompile if differ;
            // If compile options are absent in binary, do not compare and recompile
            if (compileOptions_.empty())
                break;

            std::string sBinOptions = compileOptions_ + linkOptions_;

            compileOptions_ = sCurCompileOptions;
            linkOptions_ = sCurLinkOptions;

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
                if (continueCompileFrom == ACL_TYPE_LLVMIR_BINARY ||
                    continueCompileFrom == ACL_TYPE_DEFAULT) {
                    break;
                }
                completeStages.pop_back();
            }
        }
    }
    return continueCompileFrom;
}

bool
LightningProgram::createBinary(amd::option::Options *options)
{
    if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt, type())) {
        LogError("Failed to create ELF binary image!");
        return false;
    }
    return true;
}

bool
LightningProgram::linkImpl(
    const std::vector<Program *> &inputPrograms,
    amd::option::Options *options,
    bool createLibrary)
{
    using namespace amd::opencl_driver;
    std::auto_ptr<Compiler> C(newCompilerInstance());

    std::vector<Data*> inputs;
    for (auto program : (const std::vector<LightningProgram*>&)inputPrograms) {
        if (program->llvmBinary_.empty()) {
            if (program->clBinary() == NULL) {
                buildLog_ += "Internal error: Input program not compiled!\n";
                return false;
            }

            // We are using CL binary directly.
            // Setup elfIn() and try to load llvmIR from binary
            // This elfIn() will be released at the end of build by finiBuild().
            if (!program->clBinary()->setElfIn(ELFCLASS64)) {
                buildLog_ += "Internal error: Setting input OCL binary failed!\n";
                return false;
            }
            if (!program->clBinary()->loadLlvmBinary(program->llvmBinary_,
                program->elfSectionType_)) {
                buildLog_ += "Internal error: Failed loading compiled binary!\n";
                return false;
            }
        }

        if (program->elfSectionType_ != amd::OclElf::LLVMIR) {
            buildLog_ += "Error: Input binary format is not supported\n.";
            return false;
        }

        Data* input = C->NewBufferReference(DT_LLVM_BC,
            (const char*) program->llvmBinary_.data(),
            program->llvmBinary_.size());

        if (!input) {
            buildLog_ += "Internal error: Failed to open the compiled programs.\n";
            return false;
        }

        // release elfIn() for the program
        program->clBinary()->resetElfIn();

        inputs.push_back(input);
    }

    // open the linked output
    amd::opencl_driver::Buffer* output = C->NewBuffer(DT_LLVM_BC);

    if (!output) {
        buildLog_ += "Error: Failed to open the linked program.\n";
        return false;
    }

    std::vector<std::string> linkOptions;

    // NOTE: The params is also used to identy cached code object.  This parameter
    //       should not contain any dyanamically generated filename.
    bool ret = dev().cacheCompilation()->linkLLVMBitcode(C.get(), inputs, output, linkOptions, buildLog_);
    buildLog_ += C->Output();
    if (!ret) {
        buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
        return false;
    }

    llvmBinary_.assign(output->Buf().data(), output->Size());
    elfSectionType_ = amd::OclElf::LLVMIR;


    if (clBinary()->saveLLVMIR()) {
        clBinary()->elfOut()->addSection(
            amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(), false);
        // store the original link options
        clBinary()->storeLinkOptions(linkOptions_);
        // store the original compile options
        clBinary()->storeCompileOptions(compileOptions_);
    }

    // skip the rest if we are building an opencl library
    if (createLibrary) {
        setType(TYPE_LIBRARY);
        if (!createBinary(options)) {
            buildLog_ += "Internal error: creating OpenCL binary failed\n";
            return false;
       }
        return true;
    }

    return linkImpl(options);
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
    auto isa_version = get_oclc_isa_version(dev().hwInfo()->gfxipVersion_);
    if (!isa_version.first) {
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

    auto correctly_rounded_sqrt = get_oclc_correctly_rounded_sqrt(
        options->oVariables->FP32RoundDivideSqrt);
    Data* correctly_rounded_sqrt_bc = C->NewBufferReference(DT_LLVM_BC,
        correctly_rounded_sqrt.first, correctly_rounded_sqrt.second);

    auto daz_opt = get_oclc_daz_opt(dev().hwInfo()->gfxipVersion_ < 900
        || options->oVariables->DenormsAreZero);
    Data* daz_opt_bc = C->NewBufferReference(DT_LLVM_BC,
        daz_opt.first, daz_opt.second);

    auto finite_only = get_oclc_finite_only(options->oVariables->FiniteMathOnly
        || options->oVariables->FastRelaxedMath);
    Data* finite_only_bc = C->NewBufferReference(DT_LLVM_BC,
        finite_only.first, finite_only.second);

    auto unsafe_math = get_oclc_unsafe_math(options->oVariables->UnsafeMathOpt
        || options->oVariables->FastRelaxedMath);
    Data* unsafe_math_bc = C->NewBufferReference(DT_LLVM_BC,
        unsafe_math.first, unsafe_math.second);

    if (!correctly_rounded_sqrt_bc || !daz_opt_bc
        || !finite_only_bc || !unsafe_math_bc) {
        buildLog_ += "Error: Failed to open the control functions.\n";
        return false;
    }


    if (!correctly_rounded_sqrt_bc || !daz_opt_bc
        || !finite_only_bc || !unsafe_math_bc) {
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

    // NOTE: The linkOptions parameter is also used to identy cached code object.  This parameter
    //       should not contain any dyanamically generated filename.
    bool ret = dev().cacheCompilation()->linkLLVMBitcode(C.get(), inputs, linked_bc, linkOptions, buildLog_);
    buildLog_ += C->Output();
    if (!ret) {
        buildLog_ += "Error: Linking bitcode failed: linking source & IR libraries.\n";
        return false;
    }

    if (options->isDumpFlagSet(amd::option::DUMP_BC_LINKED)) {
        std::ofstream f(options->getDumpFileName("_linked.bc").c_str(),
            std::ios::binary | std::ios::trunc);
        if(f.is_open()) {
            f.write(linked_bc->Buf().data(), linked_bc->Size());
            f.close();
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
    std::ostringstream mCPU;
    mCPU << " -mcpu=gfx" << dev().hwInfo()->gfxipVersion_;
    codegenOptions.append(mCPU.str());

    // Set the -O#
    std::ostringstream optLevel;
    optLevel << "-O" << options->oVariables->OptLevel;
    codegenOptions.append(" ").append(optLevel.str());

    // Tokenize the options string into a vector of strings
    std::istringstream strstr(codegenOptions);
    std::istream_iterator<std::string> sit(strstr), end;
    std::vector<std::string> params(sit, end);

    // NOTE: The params is also used to identy cached code object.  This parameter
    //       should not contain any dyanamically generated filename.
    ret = dev().cacheCompilation()->compileAndLinkExecutable(C.get(), inputs, out_exec, params, buildLog_);
    buildLog_ += C->Output();
    if (!ret) {
        buildLog_ += "Error: Creating the executable failed: Compiling LLVM IRs to exeutable\n";
        return false;
    }

    if (options->isDumpFlagSet(amd::option::DUMP_O)) {
        std::ofstream f(options->getDumpFileName(".so").c_str(),
            std::ios::binary | std::ios::trunc);
        if(f.is_open()) {
            f.write(out_exec->Buf().data(), out_exec->Size());
            f.close();
        } else {
            buildLog_ +=
                "Warning: opening the file to dump the code object failed.\n";
        }
    }

    if (options->isDumpFlagSet(amd::option::DUMP_ISA)) {
        std::string name = options->getDumpFileName(".s");
        File *dump = C->NewFile(DT_INTERNAL, name);
        if (!C->DumpExecutableAsText(out_exec, dump)) {
            buildLog_ += "Warning: failed to dump code object.\n";
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

    size_t numpHdrs;
    if (elf_getphdrnum(e, &numpHdrs) != 0) {
        buildLog_ += "Error while reading the ELF program binary\n";
        return false;
    }

    for (size_t i = 0; i < numpHdrs; ++i) {
        GElf_Phdr pHdr;
        if (gelf_getphdr(e, i, &pHdr) != &pHdr) {
            continue;
        }
        // Look for the runtime metadata note
        if (pHdr.p_type == PT_NOTE && pHdr.p_align >= sizeof(int)) {
            // Iterate over the notes in this segment
            address ptr = (address) binary + pHdr.p_offset;
            address segmentEnd = ptr + pHdr.p_filesz;

            while (ptr < segmentEnd) {
                Elf_Note* note = (Elf_Note*) ptr;
                address name = (address) &note[1];
                address desc = name + amd::alignUp(note->n_namesz, sizeof(int));

                if (note->n_type == AMDGPU::PT_NOTE::NT_AMDGPU_HSA_RUNTIME_METADATA_V_1) {
                    buildLog_ += "Error: object code with metadata v1 is not " \
                      "supported\n";
                    return false;
                }
                else if (note->n_type == AMDGPU::PT_NOTE::NT_AMDGPU_HSA_RUNTIME_METADATA
                         && note->n_namesz == sizeof AMDGPU::PT_NOTE::NoteName
                         && !memcmp(name, AMDGPU::PT_NOTE::NoteName, note->n_namesz)) {
                    std::string metadataStr((const char *) desc, (size_t) note->n_descsz);
                    metadata_ = new AMDGPU::RuntimeMD::Program::Metadata(metadataStr);
                    // We've found and loaded the runtime metadata, exit the
                    // note record loop now.
                    break;
                }
                ptr += sizeof(*note)
                    + amd::alignUp(note->n_namesz, sizeof(int))
                    + amd::alignUp(note->n_descsz, sizeof(int));
            }
        }
        // Accumulate the size of R & !X loadable segments
        else if (pHdr.p_type == PT_LOAD
                 && (pHdr.p_flags & PF_R) && !(pHdr.p_flags & PF_X)) {
            progvarsTotalSize += pHdr.p_memsz;
        }
    }

    elf_end(e);

    if (!metadata_) {
        buildLog_ += "Error: runtime metadata section not present in " \
            "ELF program binary\n";
        return false;
    }

    // note: The global variable size is updated in the context loader
    //setGlobalVariableTotalSize(progvarsTotalSize);

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
            kernelName, this, options->origOptionStr + hsailOptions(options));

        kernels()[kernelName] = kernel;

        auto symbol = executable_->GetSymbol(kernelName.c_str(), &agent);
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
