//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//


#ifndef WITHOUT_HSA_BACKEND

#include "rocprogram.hpp"

#include "compiler/lib/utils/options.hpp"
#include "rockernel.hpp"
#if defined(WITH_LIGHTNING_COMPILER)
#include "AMDGPUPTNote.h"
#include "libelf/gelf.h"
#include "driver/AmdCompiler.h"
#include "libraries.amdgcn.inc"
#else // !defined(WITH_LIGHTNING_COMPILER)
#include "roccompilerlib.hpp"
#include "amd_hsa_code.hpp"
#endif // !defined(WITH_LIGHTNING_COMPILER)
#include "utils/bif_section_labels.hpp"

#include "amd_hsa_kernel_code.h"

#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <istream>

namespace roc {

static hsa_status_t
GetKernelNamesCallback(
    hsa_executable_t exec,
    hsa_agent_t agent,
    hsa_executable_symbol_t symbol,
    void *data)
{
    std::vector<std::string>* symNameList = reinterpret_cast<std::vector<std::string>*>(data);

    hsa_symbol_kind_t sym_type;
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &sym_type);

    if (sym_type == HSA_SYMBOL_KIND_KERNEL) {
        uint32_t  len;
        hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);

        char* symName = (char*) alloca(len+1);
        hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, symName);
        symName[len] = '\0';

        std::string kernelName(symName);
        symNameList->push_back(kernelName);
    }

    return HSA_STATUS_SUCCESS;
}

/* Temporary log function for the compiler library */
static void
logFunction(const char *msg, size_t size)
{
    std::cout << "Compiler Library log :" << msg << std::endl;
}

HSAILProgram::~HSAILProgram()
{
#if !defined(WITH_LIGHTNING_COMPILER)
    acl_error error;
    // Free the elf binary
    if (binaryElf_ != NULL) {
        error = g_complibApi._aclBinaryFini(binaryElf_);
        if (error != ACL_SUCCESS) {
            LogWarning( "Error while destroying the acl binary \n" );
        }
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)
    // Destroy the executable.
    if (hsaExecutable_.handle != 0) {
        hsa_executable_destroy(hsaExecutable_);
    }
    // Destroy the program handle.
    if (hsaProgramHandle_.handle != 0) {
        hsa_ext_program_destroy(hsaProgramHandle_);
    }
    releaseClBinary();

#if defined(WITH_LIGHTNING_COMPILER)
    delete metadata_;
#endif // defined(WITH_LIGHTNING_COMPILER)
}

HSAILProgram::HSAILProgram(roc::NullDevice& device)
    : Program(device),
      binaryElf_(NULL)
{
    memset(&binOpts_, 0, sizeof(binOpts_));
    binOpts_.struct_size = sizeof(binOpts_);
    //binOpts_.elfclass = LP64_SWITCH( ELFCLASS32, ELFCLASS64 );
    //Setting as 32 bit because hsail64 returns an invalid aclTargetInfo
    //when aclGetTargetInfo is called - EPR# 377910
    binOpts_.elfclass = ELFCLASS32;
    binOpts_.bitness = ELFDATA2LSB;
    binOpts_.alloc = &::malloc;
    binOpts_.dealloc = &::free;

    hsaProgramHandle_.handle = 0;
    hsaExecutable_.handle = 0;

    hasGlobalStores_ = false;

#if defined(WITH_LIGHTNING_COMPILER)
    metadata_ = NULL;
#endif // defined(WITH_LIGHTNING_COMPILER)
}

bool
HSAILProgram::initClBinary(char *binaryIn, size_t size)
{
    // Save the original binary that isn't owned by ClBinary
    clBinary()->saveOrigBinary(binaryIn, size);

    char *bin = binaryIn;
    size_t sz = size;

    int encryptCode;

    char *decryptedBin;
    size_t decryptedSize;
    if (!clBinary()->decryptElf(binaryIn, size,
        &decryptedBin, &decryptedSize, &encryptCode)) {
        return false;
    }
    if (decryptedBin != NULL) {
        // It is decrypted binary.
        bin = decryptedBin;
        sz = decryptedSize;
    }

    // Both 32-bit and 64-bit are allowed!
    if (!amd::isElfMagic(bin)) {
        // Invalid binary.
        if (decryptedBin != NULL) {
            delete[]decryptedBin;
        }
        return false;
    }

    clBinary()->setFlags(encryptCode);

    return clBinary()->setBinary(bin, sz, (decryptedBin != NULL));
}


bool
HSAILProgram::initBuild(amd::option::Options *options)
{
    compileOptions_ = options->origOptionStr;

    if (!device::Program::initBuild(options)) {
        return false;
    }

    const char* devName = dev().deviceInfo().machineTarget_;
    options->setPerBuildInfo(
        (devName && (devName[0] != '\0')) ? devName : "gpu",
            clBinary()->getEncryptCode(), true);

    // Elf Binary setup
    std::string outFileName;

    // true means hsail required
    clBinary()->init(options, true);
    if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
        outFileName = options->getDumpFileName(".bin");
    }

#if defined(WITH_LIGHTNING_COMPILER)
    bool useELF64 = true;
#else // !defined(WITH_LIGHTNING_COMPILER)
    bool useELF64 = getCompilerOptions()->oVariables->EnableGpuElf64;
#endif // !defined(WITH_LIGHTNING_COMPILER)
    if (!clBinary()->setElfOut(useELF64 ? ELFCLASS64 : ELFCLASS32,
        (outFileName.size() >
    0) ? outFileName.c_str() : NULL)) {
        LogError("Setup elf out for gpu failed");
        return false;
    }
    return true;
}

// ! post-compile setup for GPU
bool
HSAILProgram::finiBuild(bool isBuildGood)
{
    clBinary()->resetElfOut();
    clBinary()->resetElfIn();

    if (!isBuildGood) {
        // Prevent the encrypted binary form leaking out
        clBinary()->setBinary(NULL, 0);

    }

    return device::Program::finiBuild(isBuildGood);
}

aclType
HSAILProgram::getCompilationStagesFromBinary(std::vector<aclType>& completeStages, bool& needOptionsCheck)
{
    acl_error errorCode;
    size_t secSize = 0;
    completeStages.clear();
    aclType from = ACL_TYPE_DEFAULT;
    needOptionsCheck = true;
    size_t boolSize = sizeof(bool);
    //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
    // Checking llvmir in .llvmir section
    bool containsHsailText = false;
    bool containsBrig = false;
    bool containsLlvmirText = (type() == TYPE_COMPILED);
    bool containsShaderIsa = (type() == TYPE_EXECUTABLE);
    bool containsOpts = !(compileOptions_.empty() && linkOptions_.empty());
#if !defined(WITH_LIGHTNING_COMPILER) // !defined(WITH_LIGHTNING_COMPILER)
    errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_LLVMIR, NULL, &containsLlvmirText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsLlvmirText = false;
    }
    // Checking compile & link options in .comment section
    errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_OPTIONS, NULL, &containsOpts, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsOpts = false;
    }
    // Checking HSAIL in .cg section
    containsHsailText = true;
    errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_HSAIL, NULL, &containsHsailText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsHsailText = false;
    }
    // Checking BRIG sections
    containsBrig = true;
    errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_BRIG, NULL, &containsBrig, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsBrig = false;
    }
    if (containsBrig) {
        completeStages.push_back(from);
        from = ACL_TYPE_HSAIL_BINARY;
        // Here we should check that CG stage was done.
        // Right now there are 2 criterions to check it (besides BRIG itself):
        // 1. matadata symbols symOpenclKernel for every kernel.
        // 2. HSAIL text in aclCODEGEN section.
        // Unfortunately there is no appropriate way in Compiler Lib to check 1.
        // because kernel names are unknown here, therefore only 2.
        if (containsHsailText) {
            completeStages.push_back(from);
            from = ACL_TYPE_CG;
        }
    }
    else if (containsHsailText) {
        completeStages.push_back(from);
        from = ACL_TYPE_HSAIL_TEXT;
    }
    errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_ISA, NULL, &containsShaderIsa, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsShaderIsa = false;
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)

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
    // compile from HSAIL text, no matter prev. stages and options
    case ACL_TYPE_HSAIL_TEXT:
        needOptionsCheck = false;
        break;
    case ACL_TYPE_HSAIL_BINARY:
    case ACL_TYPE_CG:
        // do not check options, if LLVMIR is absent or might be absent or options are absent
        if (curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
            needOptionsCheck = false;
        }
        break;
    case ACL_TYPE_ISA:
        // do not check options, if LLVMIR is absent or might be absent or options are absent
        if (curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
            needOptionsCheck = false;
        }
#if !defined(WITH_LIGHTNING_COMPILER)
        if (containsBrig && containsHsailText && curOptions.oVariables->BinHSAIL) {
            needOptionsCheck = false;
            // recompile from prev. stage, if BRIG || HSAIL are absent
        } else {
            from = completeStages.back();
            completeStages.pop_back();
            needOptionsCheck = true;
        }
#endif
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
HSAILProgram::getNextCompilationStageFromBinary(amd::option::Options* options)
{
    aclType continueCompileFrom = ACL_TYPE_DEFAULT;
    binary_t binary = this->binary();
    // If the binary already exists
    if ((binary.first != NULL) && (binary.second > 0)) {
#if defined(WITH_LIGHTNING_COMPILER)
        void *mem = (void *) binary.first;
#else // !defined(WITH_LIGHTNING_COMPILER)
        void *mem = const_cast<void *>(binary.first);
        acl_error errorCode;
        binaryElf_ = g_complibApi._aclReadFromMem(mem, binary.second, &errorCode);
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error while BRIG Codegen phase: aclReadFromMem failure \n" ;
            return continueCompileFrom;
        }
#endif // !defined(WITH_LIGHTNING_COMPILER)

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
        case ACL_TYPE_HSAIL_BINARY:
        case ACL_TYPE_CG:
        case ACL_TYPE_ISA: {
            // Compare options loaded from binary with current ones, recompile if differ;
            // If compile options are absent in binary, do not compare and recompile
            if (compileOptions_.empty())
                break;

#if defined(WITH_LIGHTNING_COMPILER)
            std::string sBinOptions = compileOptions_ + linkOptions_;
#else // !defined(WITH_LIGHTNING_COMPILER)
            const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symOpenclCompilerOptions);
            assert(symbol && "symbol not found");
            std::string symName = std::string(symbol->str[bif::PRE]) + std::string(symbol->str[bif::POST]);
            size_t symSize = 0;

            const void *opts = g_complibApi._aclExtractSymbol(device().compiler(),
                binaryElf_, &symSize, aclCOMMENT, symName.c_str(), &errorCode);
            if (errorCode != ACL_SUCCESS) {
                recompile = true;
                break;
            }
            std::string sBinOptions = std::string((char*)opts, symSize);
#endif // !defined(WITH_LIGHTNING_COMPILER)

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

static hsa_status_t
allocFunc(size_t size, hsa_callback_data_t data, void **address)
{
    if (!address || 0 == size) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    *address = (char*) malloc(size);
    if (!*address) {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    return HSA_STATUS_SUCCESS;
}

bool
HSAILProgram::saveBinaryAndSetType(type_t type, void* rawBinary, size_t size)
{
    //Write binary to memory
#if defined(WITH_LIGHTNING_COMPILER)
    if (type == TYPE_EXECUTABLE) {     // handle code object binary
        assert(rawBinary != NULL && size != 0 && "must pass in the binary");
    }
    else {                  // handle LLVM binary
        if (llvmBinary_.empty()) {
            buildLog_ += "ERROR: Tried to save emtpy LLVM binary \n";
            return false;
        }
        rawBinary = (void*) llvmBinary_.data();
        size = llvmBinary_.size();
    }
#else // !defined(WITH_LIGHTNING_COMPILER)
    if (g_complibApi._aclWriteToMem(binaryElf_, &rawBinary, &size)
        != ACL_SUCCESS) {
        buildLog_ += "Failed to write binary to memory \n";
        return false;
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)
    clBinary()->saveBIFBinary((char*)rawBinary, size);
    //Set the type of binary
    setType(type);

    //Free memory containing rawBinary
#if !defined(WITH_LIGHTNING_COMPILER)
    binaryElf_->binOpts.dealloc(rawBinary);
#endif
    return true;
}

#if defined(WITH_LIGHTNING_COMPILER)
bool
HSAILProgram::linkImpl_LC(
    const std::vector<Program *> &inputPrograms,
    amd::option::Options *options,
    bool createLibrary)
{
    using namespace amd::opencl_driver;
    std::auto_ptr<Compiler> C(newCompilerInstance());

    std::vector<Data*> inputs;
    for (auto program : (const std::vector<HSAILProgram*>&)inputPrograms) {
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
    Buffer* output = C->NewBuffer(DT_LLVM_BC);

    if (!output) {
        buildLog_ += "Error: Failed to open the linked program.\n";
        return false;
    }

    std::vector<std::string> linkOptions;

    // NOTE: The linkOptions parameter is also used to identy cached code object.  This parameter
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

    return linkImpl_LC(options);
}
#endif // defined(WITH_LIGHTNING_COMPILER)

bool
HSAILProgram::linkImpl(
    const std::vector<Program *> &inputPrograms,
    amd::option::Options *options,
    bool createLibrary)
{
#if defined(WITH_LIGHTNING_COMPILER)
    return linkImpl_LC(inputPrograms, options, createLibrary);
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
        if ((binary.first != NULL) && (binary.second > 0)) {
            // Binary already exists -- we can also check if there is no
            // opencl source code
            // Need to check if LLVMIR exists in the binary
            // If LLVMIR does not exist then is it valid
            // We need to pull out all the compiled kernels
            // We cannot do this at present because we need at least
            // Hsail text to pull the kernels oout
            void *mem = const_cast<void *>(binary.first);
            binaryElf_ = g_complibApi._aclReadFromMem(mem,
                binary.second,
                &errorCode);

            if (errorCode != ACL_SUCCESS) {
                LogWarning("Error while linking : Could not read from raw binary");
                return false;
            }
        }
        // At this stage each HSAILProgram contains a valid binary_elf
        // Check if LLVMIR is in the binary
        size_t boolSize = sizeof(bool);
        bool containsLLLVMIR = false;
        errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_,
            RT_CONTAINS_LLVMIR, NULL, &containsLLLVMIR, &boolSize);
        if (errorCode != ACL_SUCCESS || !containsLLLVMIR) {
            buildLog_ +="Error while linking : Invalid binary (Missing LLVMIR section)";
            return false;
        }
        // Create a new aclBinary for each LLVMIR and save it in a list
        aclBIFVersion ver = g_complibApi._aclBinaryVersion(binaryElf_);
        aclBinary *bin = g_complibApi._aclCreateFromBinary(binaryElf_, ver);
        binaries_to_link.push_back(bin);
    }

    // At this stage each HSAILProgram in the list has an aclBinary initialized
    // and contains LLVMIR
    // We can now go ahead and link them.
    if (binaries_to_link.size() > 1) {
        errorCode = g_complibApi._aclLink(device().compiler(),
            binaries_to_link[0],
            binaries_to_link.size() - 1,
            &binaries_to_link[1],
            ACL_TYPE_LLVMIR_BINARY,
            "-create-library",
            NULL);
    }
    else {
        errorCode = g_complibApi._aclLink(device().compiler(),
            binaries_to_link[0],
            0,
            NULL,
            ACL_TYPE_LLVMIR_BINARY,
            "-create-library",
            NULL);
    }
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Failed to link programs";
        return false;
    }
    // Store the newly linked aclBinary for this program.
    binaryElf_ = binaries_to_link[0];
    // Free all the other aclBinaries
    for (size_t i = 1; i < binaries_to_link.size(); i++) {
        g_complibApi._aclBinaryFini(binaries_to_link[i]);
    }
    if (createLibrary) {
        saveBinaryAndSetType(TYPE_LIBRARY);
        return true;
    }

    // Now call linkImpl with the new options
    return linkImpl(options);
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

static inline const char*
hsa_strerror(hsa_status_t status)
{
    const char* str = NULL;
    if (hsa_status_string(status, &str) == HSA_STATUS_SUCCESS) {
        return str;
    }
    return "Unknown error";
}

#if defined(WITH_LIGHTNING_COMPILER)
bool
HSAILProgram::linkImpl_LC(amd::option::Options *options)
{
    using namespace amd::opencl_driver;
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
    auto isa_version = get_oclc_isa_version(dev().deviceInfo().gfxipVersion_);
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

    auto daz_opt = get_oclc_daz_opt(dev().deviceInfo().gfxipVersion_ < 900
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

    inputs.push_back(correctly_rounded_sqrt_bc);
    inputs.push_back(daz_opt_bc);
    inputs.push_back(finite_only_bc);
    inputs.push_back(unsafe_math_bc);

    // open the linked output
    std::vector<std::string> linkOptions;
    Buffer* linked_bc = C->NewBuffer(DT_LLVM_BC);

    if (!linked_bc) {
        buildLog_ += "Error: Failed to open the linked program.\n";
        return false;
    }

    // NOTE: The linkOptions parameter is also used to identy cached code object. This parameter
    //       should not contain any dyanamically generated filename.
    bool ret = dev().cacheCompilation()->linkLLVMBitcode(C.get(), inputs, linked_bc, linkOptions, buildLog_);
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

    Buffer* out_exec = C->NewBuffer(DT_EXECUTABLE);
    if (!out_exec) {
        buildLog_ += "Error: Failed to create the linked executable.\n";
        return false;
    }

    std::string codegenOptions(options->llvmOptions);

    // Set the machine target
    codegenOptions.append(" -mcpu=");
    codegenOptions.append(dev().deviceInfo().machineTarget_);

    // Set the -O#
    std::ostringstream optLevel;
    optLevel << "-O" << options->oVariables->OptLevel;
    codegenOptions.append(" ").append(optLevel.str());

    // Set whole program mode
    codegenOptions.append(" -mllvm -amdgpu-internalize-symbols");

    // Tokenize the options string into a vector of strings
    std::istringstream strstr(codegenOptions);
    std::istream_iterator<std::string> sit(strstr), end;
    std::vector<std::string> params(sit, end);

    // NOTE: The params is also used to identy cached code object.  This paramete
    //       should not contain any dyanamically generated filename.
    ret = dev().cacheCompilation()->compileAndLinkExecutable(C.get(), inputs, out_exec, params, buildLog_);
    buildLog_ += C->Output();
    if (!ret) {
        buildLog_ += "Error: Creating the executable failed: Compiling LLVM IRs to executable\n";
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

    if (options->isDumpFlagSet(amd::option::DUMP_ISA)) {
        std::string name = options->getDumpFileName(".s");
        File *dump = C->NewFile(DT_INTERNAL, name);
        if (!C->DumpExecutableAsText(out_exec, dump)) {
            buildLog_ += "Warning: failed to dump code object.\n";
        }
    }

    return setKernels_LC( options, out_exec->Buf().data(), out_exec->Size() );
}

bool
HSAILProgram::setKernels_LC(amd::option::Options *options, void* binary, size_t binSize)
{
    hsa_agent_t agent = dev().getBackendDevice();
    hsa_status_t status;

    status = hsa_executable_create_alt(
        HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
        NULL, &hsaExecutable_ );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Executable for AMD HSA Code Object isn't created: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    // Load the code object.
    hsa_code_object_reader_t codeObjectReader;
    status = hsa_code_object_reader_create_from_memory(
        binary, binSize, &codeObjectReader);
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: AMD HSA Code Object Reader create failed: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    status = hsa_executable_load_agent_code_object(
        hsaExecutable_, agent, codeObjectReader, NULL, NULL );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: AMD HSA Code Object loading failed: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    hsa_code_object_reader_destroy(codeObjectReader);

    // Freeze the executable.
    status = hsa_executable_freeze( hsaExecutable_, NULL );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Freezing the executable failed: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    size_t progvarsTotalSize = 0;
    size_t dynamicSize = 0;
    size_t progvarsWriteSize = 0;

    // Begin the Elf image from memory
    Elf* e = elf_memory((char*) binary, binSize, NULL);
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

                if (note->n_type == AMDGPU::ElfNote::NT_AMDGPU_HSA_RUNTIME_METADATA_V_1) {
                    buildLog_ += "Error: object code with metadata v1 is not " \
                      "supported\n";
                    return false;
                }
                else if (note->n_type == AMDGPU::ElfNote::NT_AMDGPU_HSA_RUNTIME_METADATA
                         && note->n_namesz == sizeof AMDGPU::ElfNote::NoteName
                         && !memcmp(name, AMDGPU::ElfNote::NoteName, note->n_namesz)) {
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
        else if (pHdr.p_type == PT_LOAD && !(pHdr.p_flags & PF_X)) {
            if (pHdr.p_flags & PF_R) {
                progvarsTotalSize += pHdr.p_memsz;
            }
            if (pHdr.p_flags & PF_W) {
                progvarsWriteSize += pHdr.p_memsz;
            }
        }
        else if (pHdr.p_type == PT_DYNAMIC) {
            dynamicSize += pHdr.p_memsz;
        }
    }

    elf_end(e);

    if (!metadata_) {
        buildLog_ += "Error: runtime metadata section not present in " \
            "ELF program binary\n";
        return false;
    }

    if (progvarsWriteSize != dynamicSize) {
        hasGlobalStores_ = true;
    }
    progvarsTotalSize -= dynamicSize;
    setGlobalVariableTotalSize(progvarsTotalSize);

    saveBinaryAndSetType(TYPE_EXECUTABLE, binary, binSize);

    // Get the list of kernels
    std::vector<std::string> kernelNameList;
    status = hsa_executable_iterate_agent_symbols( hsaExecutable_,
        agent, GetKernelNamesCallback, (void *) &kernelNameList );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Failed to get kernel names: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    for (auto &kernelName : kernelNameList) {
        hsa_executable_symbol_t kernelSymbol;

        status = hsa_executable_get_symbol_by_name(
            hsaExecutable_, kernelName.c_str(), &agent, &kernelSymbol);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get the symbol: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint64_t kernelCodeHandle;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
            &kernelCodeHandle);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get the kernel code: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint32_t workgroupGroupSegmentByteSize;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
            &workgroupGroupSegmentByteSize);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get group segment size info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint32_t workitemPrivateSegmentByteSize;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
            &workitemPrivateSegmentByteSize);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get private segment size info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint32_t kernargSegmentByteSize;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            &kernargSegmentByteSize);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get kernarg segment size info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint32_t kernargSegmentAlignment;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
            &kernargSegmentAlignment);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get kernarg segment alignment info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        // FIME_lmoriche: the compiler should set the kernarg alignment based
        // on the alignment requirement of the parameters. For now, bump it to
        // the worse case: 128byte aligned.
        kernargSegmentAlignment = std::max(kernargSegmentAlignment, 128u);

        Kernel *aKernel = new roc::Kernel(
            kernelName,
            this,
            kernelCodeHandle,
            workgroupGroupSegmentByteSize,
            workitemPrivateSegmentByteSize,
            kernargSegmentByteSize,
            amd::alignUp(kernargSegmentAlignment,device().info().globalMemCacheLineSize_));
        if (!aKernel->init()) {
            return false;
        }
        aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
        aKernel->setInternalKernelFlag(compileOptions_.find("-cl-internal-kernel") != std::string::npos);
        kernels()[kernelName] = aKernel;
    }

    return true;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

bool
HSAILProgram::linkImpl(amd::option::Options *options)
{
    acl_error errorCode;
    aclType continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;
    bool finalize = true;
#if !defined(WITH_LIGHTNING_COMPILER)
    // If !binaryElf_ then program must have been created using clCreateProgramWithBinary
    if (!binaryElf_)
#else // defined(WITH_LIGHTNING_COMPILER)
        if (llvmBinary_.empty())
#endif // defined(WITH_LIGHTNING_COMPILER)
        {
            continueCompileFrom = getNextCompilationStageFromBinary(options);
        }
    switch (continueCompileFrom) {
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
#if defined(WITH_LIGHTNING_COMPILER)
        if (!linkImpl_LC(options)) {
            return false;
        }
#else // !defined(WITH_LIGHTNING_COMPILER)
        std::string curOptions = options->origOptionStr
            + preprocessorOptions(options) + codegenOptions(options);
        errorCode = g_complibApi._aclCompile(device().compiler(), binaryElf_,
            curOptions.c_str(), continueCompileFrom, ACL_TYPE_CG, logFunction);
        buildLog_ += g_complibApi._aclGetCompilerLog(device().compiler());
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error while BRIG Codegen phase: compilation error \n" ;
            return false;
        }
#endif // !defined(WITH_LIGHTNING_COMPILER)
        break;
    }
    case ACL_TYPE_CG:
        break;
    case ACL_TYPE_ISA: {
#if defined(WITH_LIGHTNING_COMPILER)
       binary_t isaBinary = binary();
       if ((isaBinary.first != NULL) && (isaBinary.second > 0)) {
            return setKernels_LC(options, (void*) isaBinary.first, isaBinary.second );
        }
       else {
            buildLog_ += "Error: code object is empty \n" ;
            return false;
        }
#endif // !defined(WITH_LIGHTNING_COMPILER)
        finalize = false;
        break;
    }
    default:
        buildLog_ += "Error while BRIG Codegen phase: the binary is incomplete \n" ;
        return false;
    }
    //Stop compilation if it is an offline device - HSA runtime does not
    //support ISA compiled offline
    if (!dev().isOnline()) {
        return true;
    }

#if !defined(WITH_LIGHTNING_COMPILER)
    hsa_agent_t hsaDevice = dev().getBackendDevice();

    std::string fin_options(options->origOptionStr);
    // Append an option so that we can selectively enable a SCOption on CZ
    // whenever IOMMUv2 is enabled.
    if (dev().isFineGrainedSystem(true)) {
        fin_options.append(" -sc-xnack-iommu");
    }
    errorCode = aclCompile(dev().compiler(), binaryElf_,
        fin_options.c_str(), ACL_TYPE_CG, ACL_TYPE_ISA, logFunction);
    buildLog_ += aclGetCompilerLog(dev().compiler());
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Error: BRIG finalization to ISA failed.\n";
        return false;
    }
    size_t secSize;
    void *data = (void*)aclExtractSection(device().compiler(),
        binaryElf_, &secSize, aclTEXT, &errorCode);
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Error: cannot extract ISA from compiled binary.\n";
        return false;
    }

    // Create an executable.
    hsa_status_t status = hsa_executable_create_alt(
        HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
        NULL, &hsaExecutable_
    );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Failed to create executable: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    // Load the code object.
    hsa_code_object_reader_t codeObjectReader;
    status = hsa_code_object_reader_create_from_memory(
        data, secSize, &codeObjectReader);
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: AMD HSA Code Object Reader create failed: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    status = hsa_executable_load_agent_code_object(
        hsaExecutable_, hsaDevice, codeObjectReader, NULL, NULL );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: AMD HSA Code Object loading failed: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    hsa_code_object_reader_destroy(codeObjectReader);

    // Freeze the executable.
    status = hsa_executable_freeze(hsaExecutable_, NULL);
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Failed to freeze executable: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    // Get the list of kernels
    std::vector<std::string> kernelNameList;
    status = hsa_executable_iterate_agent_symbols( hsaExecutable_,
        hsaDevice, GetKernelNamesCallback, (void *) &kernelNameList );
    if (status != HSA_STATUS_SUCCESS) {
        buildLog_ += "Error: Failed to get kernel names: ";
        buildLog_ += hsa_strerror(status);
        buildLog_ += "\n";
        return false;
    }

    for (auto &kernelName : kernelNameList) {
        // Query symbol handle for this symbol.
        hsa_executable_symbol_t kernelSymbol;
        status = hsa_executable_get_symbol_by_name(
            hsaExecutable_, kernelName.c_str(), &hsaDevice, &kernelSymbol
        );
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get executable symbol: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        // Query code handle for this symbol.
        uint64_t kernelCodeHandle;
        status = hsa_executable_symbol_get_info(
            kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle
        );
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get executable symbol info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        std::string openclKernelName = kernelName;
        // Strip the opencl and kernel name
        kernelName = kernelName.substr(strlen("&__OpenCL_"), kernelName.size());
        kernelName = kernelName.substr(0,kernelName.size() - strlen("_kernel"));
        aclMetadata md;
        md.numHiddenKernelArgs = 0;

        size_t sizeOfnumHiddenKernelArgs = sizeof(md.numHiddenKernelArgs);
        errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_NUM_KERNEL_HIDDEN_ARGS,
            openclKernelName.c_str(), &md.numHiddenKernelArgs, &sizeOfnumHiddenKernelArgs);
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error while Finalization phase: Kernel extra arguments count querying from the ELF failed\n";
            return false;
        }

        uint32_t workgroupGroupSegmentByteSize;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
            &workgroupGroupSegmentByteSize);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get group segment size info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint32_t workitemPrivateSegmentByteSize;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
            &workitemPrivateSegmentByteSize);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get private segment size info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint32_t kernargSegmentByteSize;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            &kernargSegmentByteSize);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get kernarg segment size info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        uint32_t kernargSegmentAlignment;
        status = hsa_executable_symbol_get_info(
            kernelSymbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
            &kernargSegmentAlignment);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: Failed to get kernarg segment alignment info: ";
            buildLog_ += hsa_strerror(status);
            buildLog_ += "\n";
            return false;
        }

        Kernel *aKernel = new roc::Kernel(
            kernelName,
            this,
            kernelCodeHandle,
            workgroupGroupSegmentByteSize,
            workitemPrivateSegmentByteSize,
            kernargSegmentByteSize,
            kernargSegmentAlignment);
        if (!aKernel->init()) {
            return false;
        }
        aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
        aKernel->setInternalKernelFlag(compileOptions_.find("-cl-internal-kernel") != std::string::npos);
        kernels()[kernelName] = aKernel;
    }
    saveBinaryAndSetType(TYPE_EXECUTABLE);
    buildLog_ += g_complibApi._aclGetCompilerLog(device().compiler());
#endif // !defined(WITH_LIGHTNING_COMPILER)
    return true;
}

bool
HSAILProgram::createBinary(amd::option::Options *options)
{
#if defined(WITH_LIGHTNING_COMPILER)
    if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt, type())) {
        LogError("Failed to create ELF binary image!");
        return false;
    }
    return true;
#else // !defined(WITH_LIGHTNING_COMPILER)
    return false;
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

bool
HSAILProgram::initClBinary()
{
    if (clBinary_ == NULL) {
        clBinary_ = new ClBinary(static_cast<const Device &>(device()));
        if (clBinary_ == NULL) {
            return false;
        }
    }
    return true;
}

void
HSAILProgram::releaseClBinary()
{
    if (clBinary_ != NULL) {
        delete clBinary_;
        clBinary_ = NULL;
    }
}

std::string
HSAILProgram::codegenOptions(amd::option::Options* options)
{
    std::string optionsStr;

#if !defined(WITH_LIGHTNING_COMPILER)
    if (dev().deviceInfo().gfxipVersion_ < 900) {
        optionsStr.append(" -cl-denorms-are-zero");
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)

    //check if the host is 64 bit or 32 bit
    LP64_ONLY(optionsStr.append(" -m64"));

    return optionsStr;
}

std::string
HSAILProgram::preprocessorOptions(amd::option::Options* options)
{
    std::string optionsStr;

    //Set options for the standard device specific options

    optionsStr.append(" -D__AMD__=1");

    optionsStr.append(" -D__").append(device().info().name_).append("__=1");
    optionsStr.append(" -D__").append(device().info().name_).append("=1");

    int major, minor;
    ::sscanf(device().info().version_, "OpenCL %d.%d ", &major, &minor);

    std::stringstream ss;
    ss << " -D__OPENCL_VERSION__=" << (major * 100 + minor * 10);
    optionsStr.append(ss.str());

    if (device().info().imageSupport_ && options->oVariables->ImageSupport) {
        optionsStr.append(" -D__IMAGE_SUPPORT__=1");
    }

    //This is just for legacy compiler code
    // All our devices support these options now
    if (options->oVariables->FastFMA) {
        optionsStr.append(" -DFP_FAST_FMA=1");
    }
    if (options->oVariables->FastFMAF) {
        optionsStr.append(" -DFP_FAST_FMAF=1");
    }

    uint clcStd = (options->oVariables->CLStd[2] - '0') * 100
        + (options->oVariables->CLStd[4] - '0') * 10;

    if (clcStd >= 200) {
        std::stringstream opts;
        //Add only for CL2.0 and later
        opts << " -D" << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE="
            << device().info().maxGlobalVariableSize_;
        optionsStr.append(opts.str());
    }

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

        optionsStr.append(clext.str());
    }
#else // !defined(WITH_LIGHTNING_COMPILER)
    for (auto e : extensions) {
        optionsStr.append(" -D").append(e).append("=1");
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)

    return optionsStr;
}

}  // namespace roc

#endif  // WITHOUT_HSA_BACKEND
