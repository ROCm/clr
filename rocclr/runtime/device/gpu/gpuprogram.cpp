//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "os/os.hpp"
#include "utils/flags.hpp"
#include "include/aclTypes.h"
#include "utils/amdilUtils.hpp"
#include "utils/bif_section_labels.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpublit.hpp"
#include "macrodata.h"
#include "MDParser/AMDILMDInterface.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include "utils/options.hpp"
#include "hsa.h"
#include "hsa_ext_image.h"
#include "amd_hsa_loader.hpp"

namespace gpu {

bool
NullProgram::initBuild(amd::option::Options* options)
{
    if (!device::Program::initBuild(options)) {
        return false;
    }

    const char* devname = dev().hwInfo()->machineTarget_;
    options->setPerBuildInfo(
        (devname && (devname[0] != '\0')) ? devname : "gpu",
        clBinary()->getEncryptCode(),
        true // FIXME: the dev ptr is used to query the wavefront size.
        );

    // Elf Binary setup
    std::string outFileName;

    // Recompile from IL may happen (invoking Kernel::recompil()) to generate correct
    // isa code for 7xx. Because of this, force saving AMDIL into the binary.
    clBinary()->init(options, (dev().calTarget() <= CAL_TARGET_730));
    if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
      outFileName = options->getDumpFileName(".bin");
    }

    bool useELF64 = dev().settings().use64BitPtr_;
    if (!clBinary()->setElfOut(useELF64 ? ELFCLASS64 : ELFCLASS32,
                               (outFileName.size() > 0) ? outFileName.c_str() : NULL)) {
        LogError("Setup elf out for gpu failed");
        return false;
    }
    return true;
}

bool
NullProgram::finiBuild(bool isBuildGood)
{
    clBinary()->resetElfOut();
    clBinary()->resetElfIn();

    if (!isBuildGood) {
        // Prevent the encrypted binary form leaking out
        clBinary()->setBinary(NULL, 0);
    }

    return device::Program::finiBuild(isBuildGood);
}

const aclTargetInfo &
NullProgram::info(const char * str) {
    acl_error err;
    std::string arch = GPU_TARGET_INFO_ARCH;
    if (dev().settings().use64BitPtr_) {
      arch += "64";
    }
    info_ = aclGetTargetInfo(arch.c_str(), ( str && str[0] == '\0' ? dev().hwInfo()->targetName_ : str ), &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclGetTargetInfo failed");
    }
    return info_;
}

NullProgram::~NullProgram()
{
    // Destroy all ILFunc objects
    freeAllILFuncs();
    releaseClBinary();
}

bool
NullProgram::isCalled(const ILFunc* base, const ILFunc* func)
{
    // Loop through all functions, which will be called from the base one
    for (size_t i = 0; i < base->calls_.size(); ++i) {
        assert(base->calls_[i] != base && "recursion");
        // Check if the current function is the one
        if (base->calls_[i] == func) {
            return true;
        }
        // We have to use a recursive method to make sure it's not called inside
        else if (isCalled(base->calls_[i], func)) {
            return true;
        }
    }
    return false;
}

uint
ILFunc::totalHwPrivateUsage() {
  if (totalHwPrivateSize_ >= 0)
    return totalHwPrivateSize_;

  uint maxChildUsage = 0;
  for (size_t i = 0; i < calls_.size(); ++i) {
    uint childUsage = calls_[i]->totalHwPrivateUsage();
    if (childUsage > maxChildUsage)
      maxChildUsage = childUsage;
  }
  totalHwPrivateSize_ = hwPrivateSize_ + maxChildUsage;
  return totalHwPrivateSize_;
}

void
NullProgram::patchMain(std::string& kernel, uint index)
{
    std::string callPatch = "call ";
    char        sym;

    // Create the patch string
    while (index) {
        sym     = (index % 10) + 0x30;
        callPatch.insert(5, &sym, 1);
        index   /=  10;
    }
    callPatch += ";";

    // Patch the program
    kernel.replace(patch_, callPatch.size(), callPatch);
}

NullKernel*
Program::createKernel(
    const std::string&  name, const Kernel::InitData* initData,
    const std::string&  code, const std::string&  metadata, bool* created,
    const void* binaryCode, size_t binarySize)
{
    amd::option::Options *options = getCompilerOptions();
    uint64_t start_time = 0;
    if (options->oVariables->EnableBuildTiming) {
        start_time = amd::Os::timeNanos();
    }

    *created = false;
    // Create a GPU kernel
    Kernel* gpuKernel = new Kernel(name,
        static_cast<const gpu::Device&>(device()), *this, initData);

    if (gpuKernel == NULL) {
        buildLog_ += "new Kernel() failed";
        LogPrintfError("new Kernel() failed for kernel %s!", name.c_str());
        return NULL;
    }
    else if (gpuKernel->create(code, metadata, binaryCode, binarySize)) {
        // Add kernel to the program
        kernels()[gpuKernel->name()] = gpuKernel;
        buildLog_ += gpuKernel->buildLog();
    }
    else {
        buildError_ = gpuKernel->buildError();
        buildLog_ += gpuKernel->buildLog();
        delete gpuKernel;
        LogPrintfError("Kernel creation failed for kernel %s!", name.c_str());
        return NULL;
    }

    if (options->oVariables->EnableBuildTiming) {
        std::stringstream tmp_ss;
        tmp_ss << "    Time for creating kernel ("
               << name << ") : "
               << (amd::Os::timeNanos() - start_time)/1000ULL
               << " us\n";
        buildLog_ += tmp_ss.str();
    }

    *created = true;
    return static_cast<NullKernel*>(gpuKernel);
}

bool
NullProgram::linkImpl(amd::option::Options* options)
{
    if (llvmBinary_.empty()) {
        // We are using either CL binary or IL directly.
        bool hasRecompiled;
        if (ilProgram_.empty()) {
            // Setup elfIn() and try to load ISA from binary
            // This elfIn() will be released at the end of build by finiBuild().
            if (!clBinary()->setElfIn(ELFCLASS32)) {
                buildLog_ += "Internal error: Setting input OCL binary failed!\n";
                LogError("Setting input OCL binary failed");
                return false;
            }
            bool loadSuccess = false;
            if (!options->oVariables->ForceLLVM) {
                loadSuccess = loadBinary(&hasRecompiled);
            }
            if (!loadSuccess &&
                (options->oVariables->UseDebugIL &&
                 !options->oVariables->ForceLLVM)) {
                buildLog_ += "Internal error: Loading OpenCL binary under -use-debugil failed!\n";
                LogError("Loading OCL binary failed under -use-debugil");
                return false;
            }
            if (loadSuccess) {
                if (hasRecompiled) {
                    char *section;
                    size_t sz;
                    if (clBinary()->saveSOURCE() &&
                        clBinary()->elfIn()->getSection(amd::OclElf::SOURCE, &section, &sz)) {
                        clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, section, sz);
                    }
                    if (clBinary()->saveLLVMIR()) {
                        if (clBinary()->loadLlvmBinary(llvmBinary_, llvmBinaryIsSpir_) && (!llvmBinary_.empty())) {
                            clBinary()->elfOut()->addSection(llvmBinaryIsSpir_?amd::OclElf::SPIR:amd::OclElf::LLVMIR,
                                                  llvmBinary_.data(), llvmBinary_.size(), false);
                        }
                    }

                    setType(TYPE_EXECUTABLE);
                    if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt, type())) {
                        buildLog_ += "Internal error: Failed to create OpenCL binary!\n";
                        LogError("Failed to create OpenCL binary");
                        return false;
                    }
                }
                else {
                    // The original binary is good and reuse it.
                    // Release the new binary if there is.
                    clBinary()->restoreOrigBinary();
                }
                return true;
            }
            else if (clBinary()->loadLlvmBinary(llvmBinary_, llvmBinaryIsSpir_) &&
                     clBinary()->isRecompilable(llvmBinary_, amd::OclElf::CAL_PLATFORM)) {
                char *section;
                size_t sz;

                // Clean up and remove all the content generated before
                if (!clBinary()->clearElfOut()) {
                    buildLog_ += "Internal error: Resetting OpenCL Binary failed!\n";
                    LogError("Resetting output OCL binary failed");
                    return false;
                }

                if (clBinary()->saveSOURCE() &&
                    clBinary()->elfIn()->getSection(amd::OclElf::SOURCE, &section, &sz)) {
                    clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, section, sz);
                }
                if (clBinary()->saveLLVMIR()) {
                    clBinary()->elfOut()->addSection(llvmBinaryIsSpir_?amd::OclElf::SPIR:amd::OclElf::LLVMIR,
                                         llvmBinary_.data(), llvmBinary_.size(), false);
                }
            }
            else {
                buildLog_ += "Internal error: Input OpenCL binary is not for the target!\n";
                LogError("OCL Binary isn't good for the target");
                return false;
            }
        }
    }

    if (!llvmBinary_.empty()) {
        // Compile llvm binary to the IL source code
        // This is link/OPT/Codegen part of compiler.
        cl_int iErr = compileBinaryToIL(options);
        if (iErr != CL_SUCCESS) {
            buildLog_ += "Error: Compilation from LLVMIR binary to IL text failed!";
            LogError(buildLog_.c_str());
            return false;
        }
    }

    if (!ilProgram_.empty() && options->oVariables->EnableDebug) {
        // Lets parse out the dwarf debug information and store it in the elf
        llvm::CompUnit compilation(ilProgram_);
        std::string debugILStr = compilation.getILStr();
        const char* dbgSec = debugILStr.c_str();
        size_t dbgSize = debugILStr.size();
        // Add an IL section that contains debug information and is the
        // output of LLVM codegen.
        clBinary()->elfOut()->addSection(amd::OclElf::ILDEBUG, dbgSec, dbgSize);

        if ((dbgSize > 0) && options->isDumpFlagSet(amd::option::DUMP_DEBUGIL)) {
            std::string debugilWithLine;
            size_t b = 1;
            size_t e;
            int linenum=0;
            char cstr[9];
            cstr[8] = 0;
            while (b != std::string::npos) {
                e = debugILStr.find_first_of("\n", b);
                if (e != std::string::npos) {
                    ++e;
                }
                sprintf(&cstr[0], "%5x:  ", linenum);
                debugilWithLine.append(cstr);
                debugilWithLine.append(debugILStr.substr(b,e-b));
                b = e;
                ++linenum;
            }
            std::string debugilFileName = options->getDumpFileName(".debugil");
            std::fstream f;
            f.open(debugilFileName.c_str(), (std::fstream::out | std::fstream::binary));
            f.write(debugilWithLine.c_str(), debugilWithLine.size());
            f.close();
        }

        for (unsigned x = 0; x < llvm::AMDILDwarf::DEBUG_LAST; ++x) {
            dbgSec = compilation.getDebugData()->getDwarfBitstream(
                    static_cast<llvm::AMDILDwarf::DwarfSection>(x), dbgSize);
            // Do not create an elf section if the size of the section is
            // 0.
            if (!dbgSize) {
                continue;
            }
            clBinary()->elfOut()->addSection(
                    static_cast<amd::OclElf::oclElfSections>(x
                        + amd::OclElf::DEBUG_INFO), dbgSec, dbgSize);
        }

    }

    // Create kernel objects
    if (!ilProgram_.empty() && parseKernels(ilProgram_)) {
        // Loop through all possible kernels
        for (size_t i = 0; i < funcs_.size(); ++i) {
            ILFunc* baseFunc = funcs_[i];
            // Make sure we have a Kernel function, but not Intrinsic or Simple
            if (baseFunc->state_ == ILFunc::Kernel) {
                size_t      metadataSize =
                    baseFunc->metadata_.end_ - baseFunc->metadata_.begin_;
                std::string             kernel = ilProgram_;
                std::string             metadataStr;
                std::vector<ILFunc*>    notCalled;
                std::vector<ILFunc*>    called;
                std::map<int, const char**>   macros;
                size_t  j;
                Kernel::InitData    initData = {0};

                // Fill the list of not used functions, relativly to the current
                for (j = 0; j < funcs_.size(); ++j) {
                    if ((i != j) &&
                        ((funcs_[j]->state_ == ILFunc::Regular) ||
                         (funcs_[j]->state_ == ILFunc::Kernel))) {
                        if (!isCalled(baseFunc, funcs_[j])) {
                            notCalled.push_back(funcs_[j]);
                        }
                        else {
                            called.push_back(funcs_[j]);
                        }
                    }
                }

                // Get the metadata string for the current kernel
                metadataStr.insert(0, kernel,
                    baseFunc->metadata_.begin_, metadataSize);

                std::vector<ILFunc::SourceRange*> rangeList;
                // Remove unused kernels, starting from the end
                for (j = notCalled.size(); j > 0; --j) {
                    ILFunc* func = notCalled[j-1];
                    std::vector<ILFunc::SourceRange*>::iterator it;
                    for (it = rangeList.begin(); it != rangeList.end(); ++it) {
                        if ((*it)->begin_ < func->metadata_.begin_) {
                            assert((*it)->begin_ < func->code_.begin_
                                && "code and metadata not next to each other");
                            break;
                        }
                        assert((*it)->begin_ >= func->code_.begin_
                            && "code and metadata not next to each other");
                    }
                    assert(func->metadata_.begin_ > func->code_.begin_
                        && "code after metadata");
                    if (it == rangeList.end()) {
                        rangeList.push_back(&func->metadata_);
                        rangeList.push_back(&func->code_);
                    }
                    else {
                        it = rangeList.insert(it, &func->code_);
                        rangeList.insert(it, &func->metadata_);
                    }
                }
                for (j = 0; j < rangeList.size(); ++j) {
                    const ILFunc::SourceRange* range = rangeList[j];
                    kernel.erase(range->begin_, range->end_ - range->begin_);
                }

                // Patch the main program with a call to the current kernel
                patchMain(kernel, baseFunc->index_);

                // Add macros at the top, loop through all available functions
                // for this kernel
                for (j = 0; j <= called.size(); ++j) {
                    ILFunc* func = (j < called.size()) ? called[j] : baseFunc;
                    for (size_t l = func->macros_.size(); l > 0 ; --l) {
                        int lines;
                        int idx = static_cast<int>(func->macros_[l - 1]);
                        const char** macro = amd::MacroDBGetMacro(&lines, idx);

                        // Make sure we didn't place this macro already
                        if (macros[idx] == NULL) {
                            macros[idx] = macro;
                            // Do we have a valid macro?
                            if ((lines == 0) || (macro == NULL)) {
                                buildLog_ += "Error: undefined macro!\n";
                                LogPrintfError(
                                    "Metadata reports undefined macro %d!", idx);
                                return false;
                            }
                            else {
                                // Add the macro to the IL source
                                for (int k = 0; k < lines; ++k) {
                                    kernel.insert(0, macro[k], strlen(macro[k]));
                                }
                            }
                        }
                    }
                    // Accumulate all emulated local and private sizes,
                    // necessary for the kernel execution
                    initData.localSize_   += func->localSize_;

                    // Accumulate all HW local and private sizes,
                    // necessary for the kernel execution
                    initData.hwLocalSize_   += func->hwLocalSize_;
                    initData.hwPrivateSize_ += func->hwPrivateSize_;
                    initData.flags_         |= func->flags_;
                }
                initData.privateSize_ = baseFunc->totalHwPrivateUsage();
                amdilUtils::changePrivateUAVLength(kernel,
                    initData.privateSize_);

                // Create a GPU kernel
                bool    created;
                NullKernel* gpuKernel = createKernel(baseFunc->name_,
                    &initData, kernel.data(), metadataStr, &created);
                if (!created) {
                    buildLog_ += "Error: Creating kernel " +
                        baseFunc->name_ + " failed!\n";
                    LogError(buildLog_.c_str());
                    return false;
                }

                // Add the current kernel to the binary
                if (!clBinary()->storeKernel(baseFunc->name_, gpuKernel,
                    &initData, metadataStr, kernel)) {
                    buildLog_ += "Internal error: adding a kernel into OpenCL binary failed!\n";
                    return false;
                }
            }
            else {
                // Non-kernel function, save metadata symbols for recompilation
                if (clBinary()->saveAMDIL()) {
                    size_t      metadataSize =
                        baseFunc->metadata_.end_ - baseFunc->metadata_.begin_;
                    if (metadataSize <= 0) {
                        continue;
                    }
                    std::string metadataStr;
                    // Get the metadata string
                    metadataStr.insert(0, ilProgram_, baseFunc->metadata_.begin_,
                        metadataSize);

                    std::stringstream  aStream;
                    aStream << "__OpenCL_" << baseFunc->name_ << "_fmetadata";
                    std::string metaName = aStream.str();
                    // Save metadata symbols in .rodata
                    if (!clBinary()->elfOut()->addSymbol(amd::OclElf::RODATA,
                                                        metaName.c_str(),
                                                        metadataStr.data(),
                                                        metadataStr.size())) {
                        buildLog_ += "Internal error: addSymbol failed!\n";
                        LogError ("AddSymbol failed");
                        return false;
                    }
                }
            }
        }

        setType(TYPE_EXECUTABLE);
        if (!createBinary(options)) {
            buildLog_ += "Intenral error: creating OpenCL binary failed\n";
            return false;
        }

        // Destroy all ILFunc objects
        freeAllILFuncs();
        ilProgram_.clear();
        return true;
    }
    return false;
}

bool
NullProgram::linkImpl(const std::vector<device::Program*>& inputPrograms,
                      amd::option::Options* options,
                      bool createLibrary)
{
    std::vector<std::string*> llvmBinaries(inputPrograms.size());
    std::vector<bool> llvmBinaryIsSpir(inputPrograms.size());
    std::vector<device::Program*>::const_iterator it
        = inputPrograms.begin();
    std::vector<device::Program*>::const_iterator itEnd
        = inputPrograms.end();
    for (size_t i = 0; it != itEnd; ++it, ++i) {
        NullProgram* program = (NullProgram*)*it;

        if (program->llvmBinary_.empty()) {
            if (program->clBinary() == NULL) {
                buildLog_ += "Internal error: Input program not compiled!\n";
                LogError("Loading compiled input object failed");
                return false;
            }

            // We are using CL binary directly.
            // Setup elfIn() and try to load llvmIR from binary
            // This elfIn() will be released at the end of build by finiBuild().
            if (!program->clBinary()->setElfIn(ELFCLASS32)) {
                buildLog_ += "Internal error: Setting input OCL binary failed!\n";
                LogError("Setting input OCL binary failed");
                return false;
            }
            if (!program->clBinary()->loadLlvmBinary(program->llvmBinary_,
                    program->llvmBinaryIsSpir_)) {
                buildLog_
                    += "Internal error: Failed loading compiled binary!\n";
                LogError("Bad OCL Binary");
                return false;
            }

            if (!program->clBinary()->isRecompilable(program->llvmBinary_,
                                                 amd::OclElf::CAL_PLATFORM)) {
                buildLog_ += "Internal error: Input OpenCL binary is not"
                             " for the target!\n";
                LogError("OCL Binary isn't good for the target");
                return false;
            }
#if 0
                // TODO: copy .source over to output program
                char *section;
                size_t sz;

                if (clBinary()->saveSOURCE() &&
                    clBinary()->elfIn()->getSection(amd::OclElf::SOURCE, &section, &sz)) {
                    clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, section, sz);
                }
#endif
        }

        llvmBinaries[i] = &program->llvmBinary_;
        llvmBinaryIsSpir[i] = program->llvmBinaryIsSpir_;
    }

    acl_error err;
    aclTargetInfo aclinfo = info();
    aclBinaryOptions binOpts = {0};
    binOpts.struct_size = sizeof(binOpts);
    binOpts.elfclass = aclinfo.arch_id == aclAMDIL64 ? ELFCLASS64 : ELFCLASS32;
    binOpts.bitness = ELFDATA2LSB;
    binOpts.alloc = &::malloc;
    binOpts.dealloc = &::free;

    std::vector<aclBinary*> libs(llvmBinaries.size(), NULL);
    for (size_t i = 0; i < libs.size(); ++i) {
        libs[i] = aclBinaryInit(sizeof(aclBinary), &aclinfo, &binOpts, &err);
        if (err != ACL_SUCCESS) {
            LogWarning("aclBinaryInit failed");
            break;
        }

        err = aclInsertSection(dev().compiler(), libs[i],
            llvmBinaries[i]->data(), llvmBinaries[i]->size(),
            llvmBinaryIsSpir[i]?aclSPIR:aclLLVMIR);
        if (err != ACL_SUCCESS) {
            LogWarning("aclInsertSection failed");
            break;
        }

        // temporary solution to synchronize buildNo between runtime and complib
        // until we move runtime inside complib
        ((amd::option::Options*)libs[i]->options)->setBuildNo(
            options->getBuildNo());
        }


    if (libs.size() > 0 && err == ACL_SUCCESS) do {
        unsigned int numLibs = libs.size() - 1;
        bool resultIsSPIR = (llvmBinaryIsSpir[0] && numLibs == 0);

        if (numLibs > 0) {
            err = aclLink(dev().compiler(), libs[0], numLibs, &libs[1],
                ACL_TYPE_LLVMIR_BINARY, "-create-library", NULL);

            buildLog_ += aclGetCompilerLog(dev().compiler());

            if (err != ACL_SUCCESS) {
                LogWarning("aclLink failed");
                break;
            }
        }

        size_t size = 0;
        const void* llvmir = aclExtractSection(dev().compiler(), libs[0],
            &size, resultIsSPIR?aclSPIR:aclLLVMIR, &err);
        if (err != ACL_SUCCESS) {
            LogWarning("aclExtractSection failed");
            break;
        }

        llvmBinary_.assign(reinterpret_cast<const char*>(llvmir), size);
        llvmBinaryIsSpir_ = false;
    } while(0);

    std::for_each(libs.begin(), libs.end(), std::ptr_fun(aclBinaryFini));

    if (err != ACL_SUCCESS) {
        buildLog_ += "Error: linking llvm modules failed!";
        return false;
    }

    if (clBinary()->saveLLVMIR()) {
        clBinary()->elfOut()->addSection(amd::OclElf::LLVMIR,
                                        llvmBinary_.data(), llvmBinary_.size(),
                                       false);
        // store the original link options
        clBinary()->storeLinkOptions(linkOptions_);

        clBinary()->storeCompileOptions(compileOptions_);
    }

    // skip the rest if we are building an opencl library
    if (createLibrary) {
        setType(TYPE_LIBRARY);
        if (!createBinary(options)) {
            buildLog_ += "Intenral error: creating OpenCL binary failed\n";
            return false;
        }

        return true;
    }

    // Compile llvm binary to the IL source code
    // This is link/OPT/Codegen part of compiler.
    cl_int iErr = compileBinaryToIL(options);
    if (iErr != CL_SUCCESS) {
        buildLog_ += "Error: Compilation from LLVMIR binary to IL text failed!";
        LogError(buildLog_.c_str());
        return false;
    }

    if (!ilProgram_.empty() && options->oVariables->EnableDebug) {
        // Lets parse out the dwarf debug information and store it in the elf
        llvm::CompUnit compilation(ilProgram_);
        std::string debugILStr = compilation.getILStr();
        const char* dbgSec = debugILStr.c_str();
        size_t dbgSize = debugILStr.size();
        // Add an IL section that contains debug information and is the
        // output of LLVM codegen.
        clBinary()->elfOut()->addSection(amd::OclElf::ILDEBUG, dbgSec, dbgSize);

        if ((dbgSize > 0) && options->isDumpFlagSet(amd::option::DUMP_DEBUGIL)) {
            std::string debugilWithLine;
            size_t b = 1;
            size_t e;
            int linenum=0;
            char cstr[9];
            cstr[8] = 0;
            while (b != std::string::npos) {
                e = debugILStr.find_first_of("\n", b);
                if (e != std::string::npos) {
                    ++e;
                }
                sprintf(&cstr[0], "%5x:  ", linenum);
                debugilWithLine.append(cstr);
                debugilWithLine.append(debugILStr.substr(b,e-b));
                b = e;
                ++linenum;
            }
            std::string debugilFileName = options->getDumpFileName(".debugil");
            std::fstream f;
            f.open(debugilFileName.c_str(), (std::fstream::out | std::fstream::binary));
            f.write(debugilWithLine.c_str(), debugilWithLine.size());
            f.close();
        }

        for (unsigned x = 0; x < llvm::AMDILDwarf::DEBUG_LAST; ++x) {
            dbgSec = compilation.getDebugData()->getDwarfBitstream(
                    static_cast<llvm::AMDILDwarf::DwarfSection>(x), dbgSize);
            // Do not create an elf section if the size of the section is
            // 0.
            if (!dbgSize) {
                continue;
            }
            clBinary()->elfOut()->addSection(
                    static_cast<amd::OclElf::oclElfSections>(x
                        + amd::OclElf::DEBUG_INFO), dbgSec, dbgSize);
        }

    }

    // Create kernel objects
    if (!ilProgram_.empty() && parseKernels(ilProgram_)) {
        // Loop through all possible kernels
        for (size_t i = 0; i < funcs_.size(); ++i) {
            ILFunc* baseFunc = funcs_[i];
            // Make sure we have a Kernel function, but not Intrinsic or Simple
            if (baseFunc->state_ == ILFunc::Kernel) {
                size_t      metadataSize =
                    baseFunc->metadata_.end_ - baseFunc->metadata_.begin_;
                std::string             kernel = ilProgram_;
                std::string             metadataStr;
                std::vector<ILFunc*>    notCalled;
                std::vector<ILFunc*>    called;
                std::map<int, const char**>   macros;
                size_t  j;
                Kernel::InitData    initData = {0};

                // Fill the list of not used functions, relativly to the current
                for (j = 0; j < funcs_.size(); ++j) {
                    if ((i != j) &&
                        ((funcs_[j]->state_ == ILFunc::Regular) ||
                         (funcs_[j]->state_ == ILFunc::Kernel))) {
                        if (!isCalled(baseFunc, funcs_[j])) {
                            notCalled.push_back(funcs_[j]);
                        }
                        else {
                            called.push_back(funcs_[j]);
                        }
                    }
                }

                // Get the metadata string for the current kernel
                metadataStr.insert(0, kernel,
                    baseFunc->metadata_.begin_, metadataSize);

                std::vector<ILFunc::SourceRange*> rangeList;
                // Remove unused kernels, starting from the end
                for (j = notCalled.size(); j > 0; --j) {
                    ILFunc* func = notCalled[j-1];
                    std::vector<ILFunc::SourceRange*>::iterator it;
                    for (it = rangeList.begin(); it != rangeList.end(); ++it) {
                        if ((*it)->begin_ < func->metadata_.begin_) {
                            assert((*it)->begin_ < func->code_.begin_
                                && "code and metadata not next to each other");
                            break;
                        }
                        assert((*it)->begin_ >= func->code_.begin_
                            && "code and metadata not next to each other");
                    }
                    assert(func->metadata_.begin_ > func->code_.begin_
                           && "code after metadata");
                    if (it == rangeList.end()) {
                        rangeList.push_back(&func->metadata_);
                        rangeList.push_back(&func->code_);
                    }
                    else {
                        it = rangeList.insert(it, &func->code_);
                        rangeList.insert(it, &func->metadata_);
                    }
                }
                for (j = 0; j < rangeList.size(); ++j) {
                    const ILFunc::SourceRange* range = rangeList[j];
                    kernel.erase(range->begin_, range->end_ - range->begin_);
                }

                // Patch the main program with a call to the current kernel
                patchMain(kernel, baseFunc->index_);

                // Add macros at the top, loop through all available functions
                // for this kernel
                for (j = 0; j <= called.size(); ++j) {
                    ILFunc* func = (j < called.size()) ? called[j] : baseFunc;
                    for (size_t l = func->macros_.size(); l > 0 ; --l) {
                        int lines;
                        int idx = static_cast<int>(func->macros_[l - 1]);
                        const char** macro = amd::MacroDBGetMacro(&lines, idx);

                        // Make sure we didn't place this macro already
                        if (macros[idx] == NULL) {
                            macros[idx] = macro;
                            // Do we have a valid macro?
                            if ((lines == 0) || (macro == NULL)) {
                                buildLog_ += "Error: undefined macro!\n";
                                LogPrintfError(
                                    "Metadata reports undefined macro %d!", idx);
                                return false;
                            }
                            else {
                                // Add the macro to the IL source
                                for (int k = 0; k < lines; ++k) {
                                    kernel.insert(0, macro[k], strlen(macro[k]));
                                }
                            }
                        }
                    }
                    // Accumulate all emulated local and private sizes,
                    // necessary for the kernel execution
                    initData.localSize_   += func->localSize_;

                    // Accumulate all HW local and private sizes,
                    // necessary for the kernel execution
                    initData.hwLocalSize_   += func->hwLocalSize_;
                    initData.hwPrivateSize_ += func->hwPrivateSize_;
                    initData.flags_         |= func->flags_;
                }
                initData.privateSize_ = baseFunc->totalHwPrivateUsage();
                amdilUtils::changePrivateUAVLength(kernel,
                    initData.privateSize_);

                // Create a GPU kernel
                bool    created;
                NullKernel* gpuKernel = createKernel(baseFunc->name_,
                    &initData, kernel.data(), metadataStr, &created);
                if (!created) {
                    buildLog_ += "Error: Creating kernel " +
                        baseFunc->name_ + " failed!\n";
                    LogError(buildLog_.c_str());
                    return false;
                }

                // Add the current kernel to the binary
                if (!clBinary()->storeKernel(baseFunc->name_, gpuKernel,
                    &initData, metadataStr, kernel)) {
                    buildLog_ += "Internal error: adding a kernel into OpenCL binary failed!\n";
                    return false;
                }
            }
            else {
                // Non-kernel function, save metadata symbols for recompilation
                if (clBinary()->saveAMDIL()) {
                    size_t      metadataSize =
                        baseFunc->metadata_.end_ - baseFunc->metadata_.begin_;
                    if (metadataSize <= 0) {
                        continue;
                    }
                    std::string metadataStr;
                    // Get the metadata string
                    metadataStr.insert(0, ilProgram_, baseFunc->metadata_.begin_,
                        metadataSize);

                    std::stringstream  aStream;
                    aStream << "__OpenCL_" << baseFunc->name_ << "_fmetadata";
                    std::string metaName = aStream.str();
                    // Save metadata symbols in .rodata
                    if (!clBinary()->elfOut()->addSymbol(amd::OclElf::RODATA,
                                                        metaName.c_str(),
                                                        metadataStr.data(),
                                                        metadataStr.size())) {
                        buildLog_ += "Internal error: addSymbol failed!\n";
                        LogError ("AddSymbol failed");
                        return false;
                    }
                }
            }
        }

        setType(TYPE_EXECUTABLE);
        if (!createBinary(options)) {
            buildLog_ += "Intenral error: creating OpenCL binary failed\n";
            return false;
        }

        // Destroy all ILFunc objects
        freeAllILFuncs();
        ilProgram_.clear();
        return true;
    }
    return false;
}

bool
NullProgram::initClBinary()
{
    if (clBinary_ == NULL) {
        clBinary_ = new ClBinary(static_cast<const Device&>(device()));
        if (clBinary_ == NULL) {
            return false;
        }
    }
    return true;
}

void
NullProgram::releaseClBinary()
{
    if (clBinary_ != NULL) {
        delete clBinary_;
        clBinary_ = NULL;
    }
}

bool
NullProgram::loadBinary(bool* hasRecompiled)
{
    if (!clBinary()->loadKernels(*this, hasRecompiled)) {
        clear();
        return false;
    }
    return true;
}

bool
NullProgram::initGlobalData(const std::string& source, size_t start)
{
    size_t  pos, dataStart;

    // Find the global data store
    dataStart= source.find(";#DATASTART", start);
    if (dataStart!= std::string::npos) {
        uint    index = 0;
        pos = dataStart + 2;
        while (expect(source, &pos, "DATASTART:")) {
            uint    dataSize = 0;
            uint    offset;
            uint    numElements;
            size_t  posStart;
            bool failed = false;

            // Kernel has the global constants
            if (!getuint(source, &pos, &index)) {
                return false;
            }
            pos--;
            if (expect(source, &pos, ":")) {
                // Read the size
                if (!getuint(source, &pos, &dataSize)) {
                    return false;
                }
            }
            else {
                // Emulated global data store
                pos++;
                dataSize = index;
                index = 0;
            }

            if (dataSize == 0) {
                return false;
            }

            posStart = pos = source.find_first_not_of(";# \n\r", pos);

            char* globalData = new char[dataSize];
            if (globalData == NULL) {
                return false;
            }

            // Find the global data size
            while (!expect(source, &pos, "DATAEND")) {
                for (uint i = 0; i < DataTypeTotal; ++i) {
                    if (expect(source, &pos, DataType[i].tagName_)) {
                        // Read the offset
                        if (!getuint(source, &pos, &offset)) {
                            return false;
                        }
                        if (!getuint(source, &pos, &numElements)) {
                            return false;
                        }
                        for (uint j = 0; j < numElements; ++j) {
                            switch (DataType[i].type_) {
                            case KernelArg::Float: {
                                uint32_t* tmp = reinterpret_cast<uint32_t*>(globalData + offset);
                                if (!getuintHex(source, &pos, &tmp[j])) {
                                    failed = true;
                                }
                            }
                                break;
                            case KernelArg::Double: {
                                uint64_t* tmp = reinterpret_cast<uint64_t*>(globalData + offset);
                                if (!getuint64Hex(source, &pos, &tmp[j])) {
                                    failed = true;
                                }
                            }
                                break;
                            case KernelArg::Struct:
                            case KernelArg::Union:
                                // Struct and Union should be presented as bytes
                                // Fall through...
                            case KernelArg::Char: {
                                uint8_t* tmp = reinterpret_cast<uint8_t*>(globalData + offset);
                                uint    value;
                                if (!getuintHex(source, &pos, &value)) {
                                    failed = true;
                                }
                                tmp[j] = static_cast<uint8_t>(value);
                            }
                                break;
                            case KernelArg::Short: {
                                uint16_t* tmp = reinterpret_cast<uint16_t*>(globalData + offset);
                                uint    value;
                                if (!getuintHex(source, &pos, &value)) {
                                    failed = true;
                                }
                                tmp[j] = static_cast<uint16_t>(value);
                            }
                                break;
                            case KernelArg::Int:
                            case KernelArg::UInt: {
                                uint32_t* tmp = reinterpret_cast<uint32_t*>(globalData + offset);
                                if (!getuintHex(source, &pos, &tmp[j])) {
                                    failed = true;
                                }
                            }
                                break;
                            case KernelArg::Long:
                            case KernelArg::ULong: {
                                uint64_t* tmp = reinterpret_cast<uint64_t*>(globalData + offset);
                                if (!getuint64Hex(source, &pos, &tmp[j])) {
                                    failed = true;
                                }
                            }
                                break;
                            case KernelArg::None:
                            default:
                                break;
                            }
                            if (failed) {
                                delete [] globalData;
                                return false;
                            }
                        }
                        break;
                    }
                }
                if (posStart == pos) {
                    delete [] globalData;
                    return false;
                }
                posStart = pos = source.find_first_not_of(";# \n\r", pos);
            }

            if (!allocGlobalData(globalData, dataSize, index)) {
                failed = true;
            }

            if (!clBinary()->storeGlobalData(globalData, dataSize, index)) {
                failed = true;
            }

            delete [] globalData;

            // Erase the global store information
            if (index != 0) {
                if (expect(source, &pos, ":")) {
                    // Read the size
                    if (!getuint(source, &pos, &index)) {
                        return false;
                    }
                }
            }
            pos = source.find_first_not_of(";# \n\r", pos);
            (const_cast<std::string&>(source)).erase(dataStart, pos - dataStart);
            pos = dataStart;
            if (failed) {
                return false;
            }
        }
    }

    return true;
}

bool
NullProgram::findILFuncs(const std::string& source,
                         const std::string &func_start,
                         const std::string &func_end,
                         size_t& lastFuncPos)
{
    lastFuncPos = 0;

    // Find first tag
    size_t  pos = source.find(func_start);

    // Loop through all provided program arguments
    while (pos != std::string::npos) {
        char    funcName[256];
        ILFunc  func;

        func.code_.begin_ = pos;
        if (!expect(source, &pos, func_start)) {
            break;
        }

        pos = source.find_first_not_of(" \n\r", pos);
        // Read the function index
        if (!getuint(source, &pos, &func.index_)) {
            LogError("Error reading function index");
            return false;
        }

        pos = source.find_first_of(";\n\r", pos);
        if (source[pos] == '\r' || source[pos] == '\n') {
            // this is the dummy macro
            func.name_ = std::string("");
        }
        else {
            pos = source.find_first_not_of("; \n\r", pos);
            // Read the function's name
            if (!getword(source, &pos, funcName)) {
                LogError("Error reading function name");
                return false;
            }
            func.name_ = funcName;
        }

        // Find the function end
        pos = source.find(func_end, pos);
        if (!expect(source, &pos, func_end)) {
            break;
        }
        if (source[pos] == '\r' || source[pos] == '\n') {
            if (!func.name_.empty()) {
                LogError("Missing function name");
                return false;
            }
        }
        else {
            // this is the dummy macro
            pos = source.find_first_not_of("; \n\r", pos);
            if (!expect(source, &pos, funcName)) {
                LogError("Error reading function name");
                return false;
            }
        }
        // Save the function end
        func.code_.end_ = pos;

        if (!func.name_.empty()) {
            // Create a new function
            ILFunc*  clFunc = new ILFunc(func);
            if (clFunc != NULL) {
                addFunc(clFunc);
            }
            else {
                return false;
            }
        }
        lastFuncPos = pos;
        // Next function
        pos = source.find(func_start, pos);
    }

    return true;
}

bool
NullProgram::findAllILFuncs(const std::string& source, size_t& lastFuncPos)
{
    // find all functions defined using "func"
    size_t lastPos1;
    bool ret = findILFuncs(source, "func ", "endfunc ", lastPos1);
    if (!ret) return false;

    // find all functions defined using outlined macro
    size_t lastPos2;
    ret = findILFuncs(source, "mdef(", "mend", lastPos2);
    if (!ret) return false;

    lastFuncPos = std::max(lastPos1, lastPos2);
    return true;
}

bool
NullProgram::parseAllILFuncs(const std::string& source)
{
    bool doPatch = true;
    amd::option::Options *opts = getCompilerOptions();
    if (opts->isCStrOptionsEqual(opts->oVariables->XLang, "il")) {
        doPatch = false;
    }
    // Find the patch position
    if (doPatch) {
        patch_ = source.find(";$$$$$$$$$$");
        if (patch_ == std::string::npos) {
            return false;
        }
    }

    size_t lastFuncPos = 0;
    if (!findAllILFuncs(source, lastFuncPos)) {
        return false;
    }

    // Initialize the global data if available
    if (!initGlobalData(source, lastFuncPos)) {
        LogError("We failed the global constants detection/initialization!");
        return false;
    }

    return true;
}

bool
NullProgram::parseFuncMetadata(const std::string& source, size_t posBegin, size_t posEnd)
{
    ILFunc* baseFunc = NULL;
    uint    index;
    size_t  pos = posBegin;
    while (pos < posEnd) {
        if (!expect(source, &pos, ";")) {
            break;
        }
        for (uint k = 0; k < DescTotal; ++k) {
            uint    funcIndex;
            uint    j;

            if (expect(source, &pos, ArgState[k].typeName_)) {
                if (ArgState[k].type_ == KernelArg::ErrorMessage) {
                    // Next argument
                    size_t posNext = source.find(";", pos);
                    buildLog_.append("Error:");
                    buildLog_.append(source.substr(pos, posNext - pos));
                    return false;
                }
                else if (ArgState[k].type_ == KernelArg::WarningMessage) {
                    // Next argument
                    size_t posNext = source.find(";", pos);
                    buildLog_.append("Warning:");
                    buildLog_.append(source.substr(pos, posNext - pos));
                    continue;
                }
                else if (ArgState[k].type_ == KernelArg::PrivateFixed) {
                    baseFunc->flags_ |= Kernel::PrivateFixed;
                    continue;
                }
                else if (ArgState[k].type_ == KernelArg::ABI64Bit) {
                    baseFunc->flags_ |= Kernel::ABI64bit;
                    continue;
                }
                else if (ArgState[k].type_ == KernelArg::Wavefront) {
                    baseFunc->flags_ |= Kernel::LimitWorkgroup;
                    continue;
                }
                else if (ArgState[k].type_ == KernelArg::PrintfFormatStr) {
                    uint    tmp;
                    uint    arguments;
                    PrintfInfo  info;

                    // Read index
                    if (!getuint(source, &pos, &index)) {
                        return false;
                    }
                    if (printf_.size() <= index) {
                        printf_.resize(index + 1);
                    }
                    // Read the number of arguments
                    if (!getuint(source, &pos, &arguments)) {
                        return false;
                    }
                    for (uint j = 0; j < arguments; ++j) {
                        // Read the argument's size in bytes
                        if (!getuint(source, &pos, &tmp)) {
                            return false;
                        }
                        info.arguments_.push_back(tmp);
                    }

                    // Read length
                    if (!getuint(source, &pos, &tmp)) {
                        return false;
                    }
                    // Read string (uses length so all possible chars are valid)
                    for (size_t i = 0; i < tmp; ++i) {
                        char symbol = source[pos++];
                        if (symbol == '\\') {
                            // Rest of the C escape sequences (e.g. \') are handled correctly
                            // by the MDParser, we are not sure exactly how!
                            switch (source[pos]) {
                            case 'n':
                                pos++;
                                symbol = '\n';
                                break;
                            case 'r':
                                pos++;
                                symbol = '\r';
                                break;
                            case 'a':
                                pos++;
                                symbol = '\a';
                                break;
                            case 'b':
                                pos++;
                                symbol = '\b';
                                break;
                            case 'f':
                                pos++;
                                symbol = '\f';
                                break;
                            case 'v':
                                pos++;
                                symbol = '\v';
                                break;
                            default:
                                break;
                            }
                        }
                        info.fmtString_.push_back(symbol);
                    }
                    if (!expect(source, &pos, ";")) {
                        return false;
                    }
                    printf_[index] = info;
                    baseFunc->flags_ |= Kernel::PrintfOutput;
                    // Process next token ...
                    continue;
                }
                else if (ArgState[k].type_ == KernelArg::MetadataVersion) {
                    continue;
                }

                // Read the index
                if (!getuint(source, &pos, &index)) {
                    return false;
                }

                switch (ArgState[k].type_) {
                case KernelArg::PrivateSize:
                    baseFunc->privateSize_ = index;
                    continue;
                case KernelArg::LocalSize:
                    baseFunc->localSize_ = index;
                    continue;
                case KernelArg::HwPrivateSize:
                    baseFunc->hwPrivateSize_ = index;
                    continue;
                case KernelArg::HwLocalSize:
                    baseFunc->hwLocalSize_ = index;
                    continue;
                default:
                    break;
                }

                if (!ArgState[k].size_) {
                    // Find the base function
                    baseFunc = findILFunc(index);
                    if (baseFunc == NULL) {
                        return false;
                    }
                    // Sanity check
                    if (baseFunc->state_ != ILFunc::Unknown) {
                        buildLog_ = "Error: Creating kernel ";
                        buildLog_ += baseFunc->name_;
                        buildLog_ += " failed!\n";
                        LogError(buildLog_.c_str());
                        continue;
                    }
                    // If we have __OpenCL_ prefix in the name
                    // and _kernel suffix, then this is a kernel function
                    const std::string prefix = "__OpenCL_";
                    const std::string postfix = "_kernel";
                    const std::string &fname = baseFunc->name_;
                    size_t namelen = fname.size();
                    size_t postfixPos = namelen - postfix.size();
                    if (fname.compare(0, prefix.size(), prefix) == 0 &&
                        fname.compare(postfixPos, namelen, postfix) == 0) {
                        baseFunc->state_ = ILFunc::Kernel;
                        baseFunc->name_.erase(postfixPos, postfix.size());
                        baseFunc->name_.erase(0, prefix.size());
                    }
                    else {
                        baseFunc->state_ = ILFunc::Regular;
                    }
                    baseFunc->metadata_.begin_ = posBegin;
                    baseFunc->metadata_.end_   = posEnd;
                    continue;
                }

                // Process metadata
                for (j = 0; j < index; ++j) {
                    // Read the index
                    if (getuint(source, &pos, &funcIndex)) {
                        bool    error = false;
                        if (ArgState[k].name_) {
                            ILFunc* func = findILFunc(funcIndex);
                            if (NULL != func) {
                                baseFunc->calls_.push_back(func);
                            }
                            else {
                                buildLog_ += "Error: Undeclared function index ";
                                error = true;
                            }
                        }
                        else {
                            if (funcIndex != 0xffffffff) {
                                baseFunc->macros_.push_back(funcIndex);
                            }
                            else {
                                buildLog_ += "Error: Undeclared macro index ";
                                error = true;
                            }
                        }
                        if (error) {
                            char    str[8];
                            intToStr(funcIndex, str, 8);
                            buildLog_ += str;
                            buildLog_ += "\n";
                            LogError("Undeclared index!");
                            return false;
                        }
                    }
                    else {
                        return false;
                    }
                }
            }
        }
        // Next argument
        pos = source.find(";", pos);
    }
    return true;
}

bool
NullProgram::parseKernels(const std::string& source)
{
    size_t  pos = 0;

    // Strip out all the debug tokens as these are
    // not needed yet, but will be used later.
    while(1) {
        pos = source.find(";DEBUGSTART", pos);
        if (pos == std::string::npos) {
            break;
        }
        size_t last = source.find(";DEBUGEND", pos);
        const_cast<std::string&>(source).erase(pos, last - pos + 10);
        pos = last;
    }
    // Create a list of all functions in the program
    if (!parseAllILFuncs(source)) {
        return false;
    }
    pos = 0;
    // Find all available metadata structures
    for (size_t i = 0; i < funcs_.size(); ++i) {
        char    funcName[256];
        ILFunc::SourceRange range;

        // Find function metadata start
        range.begin_ = pos = source.find(";ARGSTART:", pos);
        if (pos == std::string::npos) {
            break;
        }

        // Find function metadata end
        pos = source.find(";ARGEND:", pos);
        if (!expect(source, &pos, ";ARGEND:")) {
            break;
        }
        // Read the function's name
        if (!getword(source, &pos, funcName)) {
            return false;
        }
        pos = source.find_first_not_of(" \n\r", pos);
        range.end_ = pos;
        if (!parseFuncMetadata(source, range.begin_, range.end_)) {
            return false;
        }
    }
    return true;
}

void NullProgram::freeAllILFuncs()
{
    for (size_t i = 0; i < funcs_.size(); ++i) {
        delete funcs_[i];
    }
    funcs_.clear();
}

ILFunc*
NullProgram::findILFunc(uint index)
{
    for (size_t i = 0; i < funcs_.size(); ++i) {
        if (funcs_[i]->index_ == index) {
            return funcs_[i];
        }
    }
    return NULL;
}

NullKernel*
NullProgram::createKernel(
    const std::string&  name, const Kernel::InitData* initData,
    const std::string&  code, const std::string&  metadata, bool* created,
    const void* binaryCode, size_t binarySize)
{
    amd::option::Options *options = getCompilerOptions();
    uint64_t start_time = 0;
    if (options->oVariables->EnableBuildTiming) {
        start_time = amd::Os::timeNanos();
    }

    *created = false;
    // Create a GPU kernel
    NullKernel* gpuKernel = new NullKernel(name,
        static_cast<const gpu::NullDevice&>(device()), *this);

    if (gpuKernel == NULL) {
        buildLog_ += "new Kernel() failed";
        LogPrintfError("new Kernel() failed for kernel %s!",
            name.c_str());
        return NULL;
    }
    else if (gpuKernel->create(code, metadata, binaryCode, binarySize)) {
        // Add kernel to the program
        kernels()[gpuKernel->name()] = gpuKernel;
        buildLog_ += gpuKernel->buildLog();
    }
    else {
        buildError_ = gpuKernel->buildError();
        buildLog_ += gpuKernel->buildLog();
        delete gpuKernel;
        LogPrintfError("Kernel creation failed for kernel %s!", name.c_str());
        return NULL;
    }

    if (options->oVariables->EnableBuildTiming) {
        std::stringstream tmp_ss;
        tmp_ss << "    Time for creating kernel ("
               << name << ") : "
               << (amd::Os::timeNanos() - start_time)/1000ULL
               << " us\n";
        buildLog_ += tmp_ss.str();
    }

    *created = true;
    return gpuKernel;
}

// Invoked from ClBinary
bool
NullProgram::getAllKernelILs(std::map<std::string, std::string>& allKernelILs,
                             std::string& programIL, const char* ilKernelName)
{
    llvm::CompUnit compunit (programIL);
    if (ilKernelName != NULL) {
        std::string MangeledName("__OpenCL_");
        MangeledName.append(ilKernelName);
        MangeledName.append("_kernel");
        for (int i=0; i < static_cast<int>(compunit.getNumKernels()); ++i) {
            std::string kernelname = compunit.getKernelName(i);
            if (kernelname.compare(MangeledName) == 0) {
                allKernelILs[kernelname] = compunit.getKernelStr(i);
                break;
            }
        }
    }
    else {
        for (int i=0; i < static_cast<int>(compunit.getNumKernels()); ++i) {
            std::string kernelname = compunit.getKernelName(i);
            allKernelILs[kernelname] = compunit.getKernelStr(i);
        }
    }
    return true;
}

bool
NullProgram::createBinary(amd::option::Options* options)
{
    if (options->oVariables->BinBIF30) {
        return true;
    }

    if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt,
                                      type())) {
        LogError("Failed to create ELF binary image!");
        return false;
    }
    return true;
}

Program::~Program()
{
    // Destroy the global HW constant buffers
    const Program::HwConstBuffers& gds = glbHwCb();
    for (Program::HwConstBuffers::const_iterator it = gds.begin(); it != gds.end(); ++it) {
        delete it->second;
    }

    // Destroy the global data store
    if (glbData_ != NULL) {
        delete glbData_;
    }
}

bool
Program::allocGlobalData(const void* globalData, size_t dataSize, uint index)
{
    bool result = false;
    gpu::Memory* dataStore = NULL;

    if (index == 0) {
        // We have to lock the heap block allocation,
        // so possible reallocation won't occur twice or
        // another thread could destroy a heap block,
        // while we didn't finish allocation
        amd::ScopedLock k(dev().lockAsyncOps());

        // Allocate memory for the global data store
        glbData_ = dev().createScratchBuffer(amd::alignUp(dataSize, 0x1000));
        dataStore = glbData_;
    }
    else {
        dataStore = new Memory(dev(), amd::alignUp(dataSize, ConstBuffer::VectorSize));

        // Initialize constant buffer
        if ((dataStore == NULL) || !dataStore->create(Resource::RemoteUSWC)) {
            delete dataStore;
        }
        else {
            constBufs_[index] = dataStore;
            glbCb_.push_back(index);
        }
    }

    if (dataStore != NULL) {
        // Upload data to GPU memory
        static const bool Entire  = true;
        amd::Coord3D    origin(0, 0, 0);
        amd::Coord3D    region(dataSize);
        result = dev().xferMgr().writeBuffer(globalData,
            *dataStore, origin, region, Entire);
    }

    return result;
}

bool
Program::loadBinary(bool* hasRecompile)
{
    if (clBinary()->loadKernels(*this, hasRecompile)) {
        // Load the global data
        if (clBinary()->loadGlobalData(*this)) {
            return true;
        }
    }

    // Make sure that kernels that have been generated so far shall be deleted.
    clear();

    return false;
}

HSAILProgram::HSAILProgram(Device& device)
    : Program(device)
    , llvmBinary_()
    , binaryElf_(NULL)
    , rawBinary_(NULL)
    , kernels_(NULL)
    , maxScratchRegs_(0)
    , isNull_(false)
    , executable_(NULL)
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
    , binaryElf_(NULL)
    , rawBinary_(NULL)
    , kernels_(NULL)
    , maxScratchRegs_(0)
    , isNull_(true)
    , executable_(NULL)
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
    if (rawBinary_ != NULL) {
        free(rawBinary_);
    }
    acl_error error;
    // Free the elf binary
    if (binaryElf_ != NULL) {
        error = aclBinaryFini(binaryElf_);
        if (error != ACL_SUCCESS) {
            LogWarning( "Error while destroying the acl binary \n" );
        }
    }
    releaseClBinary();
    if (executable_ != NULL) {
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
        (outFileName.size() > 0) ? outFileName.c_str() : NULL)) {
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
        clBinary()->setBinary(NULL, 0);
    }

    return device::Program::finiBuild(isBuildGood);
}

bool
HSAILProgram::linkImpl(
    const std::vector<device::Program *> &inputPrograms,
    amd::option::Options *options,
    bool createLibrary)
{
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
        const void *llvmirText = aclExtractSection(dev().hsaCompiler(),
            binaryElf_, &llvmirSize, aclLLVMIR, &errorCode);
        if (errorCode != ACL_SUCCESS) {
            bool spirv = false;
            size_t boolSize = sizeof(bool);
            errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_,
                RT_CONTAINS_SPIRV, NULL, &spirv, &boolSize);
            if (errorCode != ACL_SUCCESS) {
                spirv = false;
            }
            if (spirv) {
                errorCode = aclCompile(dev().hsaCompiler(), binaryElf_,
                    options->origOptionStr.c_str(), ACL_TYPE_SPIRV_BINARY,
                    ACL_TYPE_LLVMIR_BINARY, NULL);
                buildLog_ += aclGetCompilerLog(dev().hsaCompiler());
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

    // At this stage each HSAILProgram in the list has an aclBinary initialized
    // and contains LLVMIR
    // We can now go ahead and link them.
    if (binaries_to_link.size() > 1) {
        errorCode = aclLink(dev().hsaCompiler(),
            binaries_to_link[0], binaries_to_link.size() - 1,
            &binaries_to_link[1], ACL_TYPE_LLVMIR_BINARY, "-create-library", NULL);
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += aclGetCompilerLog(dev().hsaCompiler());
            buildLog_ +="Error while linking : \
                        aclLink failed" ;
            return false;
        }
    }
    // Store the newly linked aclBinary for this program.
    binaryElf_ = binaries_to_link[0];
    // Free all the other aclBinaries
    for (size_t i = 1; i < binaries_to_link.size(); i++) {
        aclBinaryFini(binaries_to_link[i]);
    }
    // Uncomment when CL1.2 support on HSAIL is enabled,
    // Otherwise ocltst's OCLSeparateCompile will fail
    // on the non-legacy path with CL1.2
    // if (createLibrary || options->oVariables->EnableDebug) {
    //     // Save the binary in the interface class
    //     size_t size = 0;
    //     void *mem = NULL;
    //     aclWriteToMem(binaryElf_, &mem, &size);
    //     setBinary(static_cast<char*>(mem), size);
    //     if (createLibrary)
    //         setType(TYPE_LIBRARY);
    //     buildLog_ += aclGetCompilerLog(dev().hsaCompiler());
    //     return true;
    // }
    // Now call linkImpl with the new options
    return linkImpl(options);
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
    bool containsSpirv = true;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_,
            RT_CONTAINS_SPIRV, NULL, &containsSpirv, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsSpirv = false;
    }
    if (containsSpirv) {
        completeStages.push_back(from);
        from = ACL_TYPE_SPIRV_BINARY;
    }
    bool containsSpirText = true;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_CONTAINS_SPIR, NULL, &containsSpirText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsSpirText = false;
    }
    if (containsSpirText) {
        completeStages.push_back(from);
        from = ACL_TYPE_SPIR_BINARY;
    }
    bool containsLlvmirText = true;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_CONTAINS_LLVMIR, NULL, &containsLlvmirText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsLlvmirText = false;
    }
    // Checking compile & link options in .comment section
    bool containsOpts = true;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_CONTAINS_OPTIONS, NULL, &containsOpts, &boolSize);
    if (errorCode != ACL_SUCCESS) {
      containsOpts = false;
    }
    if (containsLlvmirText && containsOpts) {
        completeStages.push_back(from);
        from = ACL_TYPE_LLVMIR_BINARY;
    }
    // Checking HSAIL in .cg section
    bool containsHsailText = true;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_CONTAINS_HSAIL, NULL, &containsHsailText, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsHsailText = false;
    }
    // Checking BRIG sections
    bool containsBrig = true;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_CONTAINS_BRIG, NULL, &containsBrig, &boolSize);
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
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_CONTAINS_LOADER_MAP, NULL, &containsLoaderMap, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsLoaderMap = false;
    }
    if (containsLoaderMap) {
        completeStages.push_back(from);
        from = ACL_TYPE_CG;
    }
    // Checking ISA in .text section
    bool containsShaderIsa = true;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_CONTAINS_ISA, NULL, &containsShaderIsa, &boolSize);
    if (errorCode != ACL_SUCCESS) {
        containsShaderIsa = false;
    }
    if (containsShaderIsa) {
        completeStages.push_back(from);
        from = ACL_TYPE_ISA;
    }
    std::string sCurOptions = compileOptions_ + linkOptions_;
    amd::option::Options curOptions;
    amd::option::parseAllOptions(sCurOptions, curOptions);
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
}

aclType
HSAILProgram::getNextCompilationStageFromBinary(amd::option::Options* options) {
    aclType continueCompileFrom = ACL_TYPE_DEFAULT;
    binary_t binary = this->binary();
    // If the binary already exists
    if ((binary.first != NULL) && (binary.second > 0)) {
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
      //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
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
          const void *opts = aclExtractSymbol(dev().hsaCompiler(),
              binaryElf_, &symSize, aclCOMMENT, symName.c_str(), &errorCode);
          if (errorCode != ACL_SUCCESS) {
              recompile = true;
              break;
          }
          std::string sBinOptions = std::string((char*)opts, symSize);
          std::string sCurOptions = compileOptions_ + linkOptions_;
          amd::option::Options curOptions, binOptions;
          amd::option::parseAllOptions(sBinOptions, binOptions);
          amd::option::parseAllOptions(sCurOptions, curOptions);
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
    acl_error errorCode;
    aclType continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;
    bool finalize = true;
    bool hsaLoad = true;
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
        errorCode = aclCompile(dev().hsaCompiler(), binaryElf_,
            curOptions.c_str(), continueCompileFrom, ACL_TYPE_CG, NULL);
        buildLog_ += aclGetCompilerLog(dev().hsaCompiler());
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
        errorCode = aclCompile(dev().hsaCompiler(), binaryElf_,
            fin_options.c_str(), ACL_TYPE_CG, ACL_TYPE_ISA, NULL);
        buildLog_ += aclGetCompilerLog(dev().hsaCompiler());
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error: BRIG finalization to ISA failed.\n";
            return false;
        }
    }
    // ACL_TYPE_CG stage is not performed for offline compilation
    hsa_agent_t agent;
    agent.handle = 1;
    if (!isNull() && hsaLoad) {
        executable_ = loader_->CreateExecutable(HSA_PROFILE_BASE, NULL);
        if (executable_ == NULL) {
            buildLog_ += "Error: Executable for AMD HSA Code Object isn't created.\n";
            return false;
        }
        size_t size = 0;
        hsa_code_object_t code_object;
        code_object.handle = reinterpret_cast<uint64_t>(aclExtractSection(dev().hsaCompiler(), binaryElf_, &size, aclTEXT, &errorCode));
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error: Extracting AMD HSA Code Object from binary failed.\n";
            return false;
        }
        hsa_status_t status = executable_->LoadCodeObject(agent, code_object, NULL);
        if (status != HSA_STATUS_SUCCESS) {
            buildLog_ += "Error: AMD HSA Code Object loading failed.\n";
            return false;
        }
    }
    size_t kernelNamesSize = 0;
    errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_KERNEL_NAMES, NULL, NULL, &kernelNamesSize);
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Error: Querying of kernel names size from the binary failed.\n";
        return false;
    }
    if (!isNull() && kernelNamesSize > 0) {
        char* kernelNames = new char[kernelNamesSize];
        errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_KERNEL_NAMES, NULL, kernelNames, &kernelNamesSize);
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
            std::string openclKernelName = Kernel::openclMangledName(kernelName);
            errorCode = aclQueryInfo(dev().hsaCompiler(), binaryElf_, RT_NUM_KERNEL_HIDDEN_ARGS,
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
    size_t size = 0;
    void *mem = NULL;
    aclWriteToMem(binaryElf_, &mem, &size);
    setBinary(static_cast<char*>(mem), size);
    buildLog_ += aclGetCompilerLog(dev().hsaCompiler());
    return true;
}

bool
HSAILProgram::createBinary(amd::option::Options *options)
{
    return true;
}

bool
HSAILProgram::initClBinary()
{
    if (clBinary_ == NULL) {
        clBinary_ = new ClBinaryHsa(static_cast<const Device &>(device()));
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
    if (!dev().settings().singleFpDenorm_) {
        hsailOptions.append(" -cl-denorms-are-zero");
    }

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

    kernels_ = new gpu::Memory(dev(), size);
    // Initialize kernel table
    if ((kernels_ == NULL) || !kernels_->create(Resource::RemoteUSWC)) {
        delete kernels_;
        return false;
    }
    else {
        size_t* table = reinterpret_cast<size_t*>(
            kernels_->map(NULL, gpu::Resource::WriteOnly));
        for (auto& it : kernels()) {
            HSAILKernel* kernel = static_cast<HSAILKernel*>(it.second);
            table[kernel->index()] = static_cast<size_t>(
                kernel->gpuAqlCode()->vmAddress());
        }
        kernels_->unmap(NULL);
    }
    return true;
}

void
HSAILProgram::fillResListWithKernels(
    std::vector<const Memory*>& memList) const
{
    for (auto& it : kernels()) {
        memList.push_back(
            static_cast<HSAILKernel*>(it.second)->gpuAqlCode());
    }
}

const aclTargetInfo &
HSAILProgram::info(const char * str) {
    acl_error err;
    std::string arch = "hsail";
    if (dev().settings().use64BitPtr_) {
      arch = "hsail-64";
    }
    info_ = aclGetTargetInfo(arch.c_str(), ( str && str[0] == '\0' ?
        dev().hwInfo()->targetName_ : str ), &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclGetTargetInfo failed");
    }
    return info_;
}

hsa_isa_t ORCAHSALoaderContext::IsaFromName(const char *name) {
    hsa_isa_t isa = {0};
    if (!strcmp(Gfx700, name)) { isa.handle = gfx700; return isa; }
    if (!strcmp(Gfx701, name)) { isa.handle = gfx701; return isa; }
    if (!strcmp(Gfx800, name)) { isa.handle = gfx800; return isa; }
    if (!strcmp(Gfx801, name)) { isa.handle = gfx801; return isa; }
    if (!strcmp(Gfx810, name)) { isa.handle = gfx810; return isa; }
    if (!strcmp(Gfx900, name)) { isa.handle = gfx900; return isa; }
    return isa;
}

bool ORCAHSALoaderContext::IsaSupportedByAgent(hsa_agent_t agent, hsa_isa_t isa) {
    switch (program_->dev().hwInfo()->gfxipVersion_) {
    default:
        LogError("Unsupported gfxip version");
        return false;
    case gfx700:
    case gfx701:
    case gfx702:
        // gfx701 only differs from gfx700 by faster fp operations and can be loaded on either device.
        return isa.handle == gfx700 || isa.handle == gfx701;
    case gfx800:
        if (ED_ATI_CAL_MACHINE_ICELAND_ISA == program_->dev().hwInfo()->machine_ ||
            ED_ATI_CAL_MACHINE_TONGA_ISA == program_->dev().hwInfo()->machine_ ) {
            return isa.handle == gfx800;
        } else {
            // gfx800 has only sgrps limited and can be loaded on later chips.
            return isa.handle == gfx800 || isa.handle == gfx801;
        }
    case gfx900:
        return isa.handle == gfx900;
    }
}

void* ORCAHSALoaderContext::SegmentAlloc(amdgpu_hsa_elf_segment_t segment,
    hsa_agent_t agent, size_t size, size_t align, bool zero) {
    assert(size);
    assert(align);
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT:
        return AgentGlobalAlloc(agent, size, align, zero);
    case AMDGPU_HSA_SEGMENT_CODE_AGENT:
        return KernelCodeAlloc(agent, size, align, zero);
    default:
        assert(false); return 0;
    }
}

bool ORCAHSALoaderContext::SegmentCopy(amdgpu_hsa_elf_segment_t segment,
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

void ORCAHSALoaderContext::SegmentFree(amdgpu_hsa_elf_segment_t segment,
    hsa_agent_t agent, void* seg, size_t size) {
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT: AgentGlobalFree(seg, size); break;
    case AMDGPU_HSA_SEGMENT_CODE_AGENT: KernelCodeFree(seg, size); break;
    default:
        assert(false); return;
    }
}

void* ORCAHSALoaderContext::SegmentAddress(amdgpu_hsa_elf_segment_t segment,
    hsa_agent_t agent, void* seg, size_t offset) {
    assert(seg);
    switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT: {
        gpu::Memory *gpuMem = reinterpret_cast<gpu::Memory*>(seg);
        return reinterpret_cast<void*>(gpuMem->vmAddress() + offset);
    }
    case AMDGPU_HSA_SEGMENT_CODE_AGENT: return (char*) seg + offset;
    default:
        assert(false); return NULL;
    }
}

hsa_status_t ORCAHSALoaderContext::SamplerCreate(
    hsa_agent_t agent,
    const hsa_ext_sampler_descriptor_t *sampler_descriptor,
    hsa_ext_sampler_t *sampler_handle) {
    if (!agent.handle) {
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    if (!sampler_descriptor || !sampler_handle) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
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
    assert(!program_->dev().settings().hsailDirectSRD_);
    gpu::Sampler* sampler = new gpu::Sampler(program_->dev());
    if (!sampler || !sampler->create(state)) {
        delete sampler;
        return HSA_STATUS_ERROR;
    }
    program_->addSampler(sampler);
    sampler_handle->handle = sampler->hwSrd();
    return HSA_STATUS_SUCCESS;
}

hsa_status_t ORCAHSALoaderContext::SamplerDestroy(
    hsa_agent_t agent, hsa_ext_sampler_t sampler_handle) {
    if (!agent.handle) {
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    if (!sampler_handle.handle) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    return HSA_STATUS_SUCCESS;
}

void* ORCAHSALoaderContext::CpuMemAlloc(size_t size, size_t align, bool zero) {
    assert(size);
    assert(align);
    assert(sizeof(void*) == 8 || sizeof(void*) == 4);
    void* ptr = amd::Os::alignedMalloc(size, align);
    if (zero) {
        memset(ptr, 0, size);
    }
    return ptr;
}

bool ORCAHSALoaderContext::CpuMemCopy(void *dst, size_t offset, const void* src, size_t size) {
  if (!dst || !src || dst == src) {
      return false;
  }
  if (0 == size) {
      return true;
  }
  amd::Os::fastMemcpy((char*)dst + offset, src, size);
  return true;
}

void* ORCAHSALoaderContext::GpuMemAlloc(size_t size, size_t align, bool zero) {
    assert(size);
    assert(align);
    assert(sizeof(void*) == 8 || sizeof(void*) == 4);
    gpu::Memory* mem = new gpu::Memory(program_->dev(), amd::alignUp(size, align));
    if (!mem || !mem->create(gpu::Resource::Local)) {
        delete mem;
        return NULL;
    }
    assert(program_->dev().xferQueue());
    if (zero) {
        char pattern = 0;
        program_->dev().xferMgr().fillBuffer(*mem, &pattern, sizeof(pattern), amd::Coord3D(0), amd::Coord3D(size));
    }
    program_->addGlobalStore(mem);
    program_->setGlobalVariableTotalSize(program_->globalVariableTotalSize() + size);
    return mem;
}

bool ORCAHSALoaderContext::GpuMemCopy(void *dst, size_t offset, const void *src, size_t size) {
    if (!dst || !src || dst == src) {
        return false;
    }
    if (0 == size) {
        return true;
    }
    assert(program_->dev().xferQueue());
    gpu::Memory* mem = reinterpret_cast<gpu::Memory*>(dst);
    return program_->dev().xferMgr().writeBuffer(src, *mem, amd::Coord3D(offset), amd::Coord3D(size), true);
    return true;
}

} // namespace gpu
