//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef WITHOUT_HSA_BACKEND

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "os/os.hpp"
#include "rocdevice.hpp"
#include "rocprogram.hpp"
#if defined(WITH_LIGHTNING_COMPILER)
#include "opencl1.2-c.amdgcn.inc"
#include "opencl2.0-c.amdgcn.inc"
#else // !defined(WITH_LIGHTNING_COMPILER)
#include "roccompilerlib.hpp"
#endif // !defined(WITH_LIGHTNING_COMPILER)
#include "utils/options.hpp"
#include <cstdio>

#if defined(ATI_OS_LINUX)
#include <dlfcn.h>
#include <libgen.h>
#endif // defined(ATI_OS_LINUX)

#if defined(WITH_LIGHTNING_COMPILER)
static std::string llvmBin_(amd::Os::getEnvironment("LLVM_BIN"));
#endif // defined(WITH_LIGHTNING_COMPILER)

//CLC_IN_PROCESS_CHANGE
extern int openclFrontEnd(const char* cmdline, std::string*, std::string* typeInfo = nullptr);

namespace roc {

/* Temporary log function for the compiler library */
static void
logFunction(const char* msg, size_t size)
{
    std::cout<< "Compiler Log: " << msg << std::endl;
}

static int programsCount = 0;

#if defined(WITH_LIGHTNING_COMPILER)
bool
HSAILProgram::compileImpl_LC(
    const std::string& sourceCode,
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames,
    amd::option::Options* options)
{
    using namespace amd::opencl_driver;
    std::auto_ptr<Compiler> C(newCompilerInstance());
    std::vector<Data*> inputs;

    Data* input = C->NewBufferReference(DT_CL,
        sourceCode.c_str(), sourceCode.length());
    if (input == nullptr) {
        buildLog_ += "Error while creating data from source code";
        return false;
    }

    inputs.push_back(input);

    Buffer* output = C->NewBuffer(DT_LLVM_BC);
    if (output == nullptr) {
        buildLog_ += "Error while creating buffer for the LLVM bitcode";
        return false;
    }

    //Set the options for the compiler
    std::ostringstream ostrstr;
    std::copy(options->clangOptions.begin(), options->clangOptions.end(),
        std::ostream_iterator<std::string>(ostrstr, " "));

    ostrstr << " -m" << sizeof(void*) * 8;
    std::string driverOptions(ostrstr.str());

    const char* xLang = options->oVariables->XLang;
    if (xLang != nullptr && strcmp(xLang, "cl")) {
        buildLog_ += "Unsupported OpenCL language.\n";
    }

    //FIXME_Nikolay: the program manager should be setting the language
    //driverOptions.append(" -x cl");

    driverOptions.append(" -cl-std=").append(options->oVariables->CLStd);

    // Set the -O#
    std::ostringstream optLevel;
    optLevel << " -O" << options->oVariables->OptLevel;
    driverOptions.append(optLevel.str());

    // Set the machine target
    driverOptions.append(" -mcpu=");
    driverOptions.append(dev().deviceInfo().machineTarget_);

    driverOptions.append(options->llvmOptions);

    // Set whole program mode
    driverOptions.append(" -mllvm -amdgpu-internalize-symbols");

    driverOptions.append(preprocessorOptions(options));

    //Find the temp folder for the OS
    std::string tempFolder = amd::Os::getEnvironment("TEMP");
    if (tempFolder.empty()) {
        tempFolder = amd::Os::getEnvironment("TMP");
        if (tempFolder.empty()) {
            tempFolder = WINDOWS_SWITCH(".","/tmp");;
        }
    }
    //Iterate through each source code and dump it into tmp
    std::fstream f;
    std::vector<std::string> headerFileNames(headers.size());
    std::vector<std::string> newDirs;
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string headerPath = tempFolder;
        std::string headerIncludeName(headerIncludeNames[i]);
        // replace / in path with current os's file separator
        if ( amd::Os::fileSeparator() != '/') {
            for (std::string::iterator it = headerIncludeName.begin(),
                end = headerIncludeName.end();
                it != end;
                ++it) {
                if (*it == '/') *it = amd::Os::fileSeparator();
            }
        }
        size_t pos = headerIncludeName.rfind(amd::Os::fileSeparator());
        if (pos != std::string::npos) {
            headerPath += amd::Os::fileSeparator();
            headerPath += headerIncludeName.substr(0, pos);
            headerIncludeName = headerIncludeName.substr(pos+1);
        }
        if (!amd::Os::pathExists(headerPath)) {
            bool ret = amd::Os::createPath(headerPath);
            assert(ret && "failed creating path!");
            newDirs.push_back(headerPath);
        }
        std::string headerFullName
        = headerPath + amd::Os::fileSeparator() + headerIncludeName;
        headerFileNames[i] = headerFullName;
        f.open(headerFullName.c_str(), std::fstream::out);
        //Should we allow asserts
        assert(!f.fail() && "failed creating header file!");
        f.write(headers[i]->c_str(), headers[i]->length());
        f.close();

        Data* inc = C->NewFileReference(DT_CL_HEADER, headerFileNames[i]);
        if (inc == nullptr) {
            buildLog_ += "Error while creating data from headers";
            return false;
        }
        inputs.push_back(inc);
    }

    //Set the include path for the temp folder that contains the includes
    if(!headers.empty()) {
        driverOptions.append(" -I");
        driverOptions.append(tempFolder);
    }

    if (options->isDumpFlagSet(amd::option::DUMP_CL)) {
        std::ofstream f(options->getDumpFileName(".cl").c_str(), std::ios::trunc);
        if(f.is_open()) {
            f << "/* Compiler options:\n" \
                   "-c -emit-llvm -target amdgcn-amd-amdhsa-opencl -x cl "
                   << driverOptions << " -include opencl-c.h "
                << "\n*/\n\n" << sourceCode;
        } else {
            buildLog_ +=
                "Warning: opening the file to dump the OpenCL source failed.\n";
        }
    }

    //FIXME_lmoriche: has the CL option been validated?
    uint clcStd = (options->oVariables->CLStd[2] - '0') * 100
        + (options->oVariables->CLStd[4] - '0') * 10;

    std::pair<const void*, size_t> hdr;
    switch(clcStd) {
    case 100: case 110: case 120:
        hdr = std::make_pair(opencl1_2_c_amdgcn, opencl1_2_c_amdgcn_size);
        break;
    case 200:
        hdr = std::make_pair(opencl2_0_c_amdgcn, opencl2_0_c_amdgcn_size);
        break;
    default:
        buildLog_ += "Unsupported requested OpenCL C version (-cl-std).\n";
        return false;
    }

    File* pch = C->NewTempFile(DT_CL_HEADER);
    if (pch == nullptr || !pch->WriteData((const char*) hdr.first, hdr.second)) {
        buildLog_ += "Error while opening the opencl-c header ";
        return false;
    }

    driverOptions.append(" -include-pch " + pch->Name());
    driverOptions.append(" -Xclang -fno-validate-pch");

    // Tokenize the options string into a vector of strings
    std::istringstream istrstr(driverOptions);
    std::istream_iterator<std::string> sit(istrstr), end;
    std::vector<std::string> params(sit, end);

    // Compile source to IR
    bool ret = dev().cacheCompilation()->compileToLLVMBitcode(C.get(), inputs, output, params, buildLog_);
    buildLog_ += C->Output();
    if (!ret) {
        buildLog_ += "Error: Failed to compile opencl source (from CL to LLVM IR).\n";
        return false;
    }

    llvmBinary_.assign(output->Buf().data(), output->Size());
    elfSectionType_ = amd::OclElf::LLVMIR;

    if (options->isDumpFlagSet(amd::option::DUMP_BC_ORIGINAL)) {
        std::ofstream f(options->getDumpFileName("_original.bc").c_str(), std::ios::trunc);
        if(f.is_open()) {
            f.write(llvmBinary_.data(), llvmBinary_.size());
        } else {
            buildLog_ +=
                "Warning: opening the file to dump the compiled IR failed.\n";
        }
    }

    if (clBinary()->saveSOURCE()) {
        clBinary()->elfOut()->addSection(
            amd::OclElf::SOURCE, sourceCode.data(), sourceCode.size());
    }
    if (clBinary()->saveLLVMIR()) {
        clBinary()->elfOut()->addSection(
            amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(), false);
        // store the original compile options
        clBinary()->storeCompileOptions(compileOptions_);
    }
    return true;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

bool
HSAILProgram::compileImpl(
    const std::string& sourceCode,
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames,
    amd::option::Options* options)
{
#if defined(WITH_LIGHTNING_COMPILER)
    return compileImpl_LC(sourceCode, headers, headerIncludeNames, options);
#else // !defined(WITH_LIGHTNING_COMPILER)
    acl_error errorCode;
    aclTargetInfo target;

    target = g_complibApi._aclGetTargetInfo(LP64_SWITCH("hsail","hsail64"),
        dev().deviceInfo().complibTarget_, &errorCode);

    //end if asic info is ready
    // We dump the source code for each program (param: headers)
    // into their filenames (headerIncludeNames) into the TEMP
    // folder specific to the OS and add the include path while
    // compiling

    //Find the temp folder for the OS
    std::string tempFolder = amd::Os::getEnvironment("TEMP");
    if (tempFolder.empty()) {
        tempFolder = amd::Os::getEnvironment("TMP");
        if (tempFolder.empty()) {
            tempFolder = WINDOWS_SWITCH(".","/tmp");;
        }
    }
    //Iterate through each source code and dump it into tmp
    std::fstream f;
    std::vector<std::string> headerFileNames(headers.size());
    std::vector<std::string> newDirs;
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string headerPath = tempFolder;
        std::string headerIncludeName(headerIncludeNames[i]);
        // replace / in path with current os's file separator
        if ( amd::Os::fileSeparator() != '/') {
            for (std::string::iterator it = headerIncludeName.begin(),
                end = headerIncludeName.end();
                it != end;
                ++it) {
                if (*it == '/') *it = amd::Os::fileSeparator();
            }
        }
        size_t pos = headerIncludeName.rfind(amd::Os::fileSeparator());
        if (pos != std::string::npos) {
            headerPath += amd::Os::fileSeparator();
            headerPath += headerIncludeName.substr(0, pos);
            headerIncludeName = headerIncludeName.substr(pos+1);
        }
        if (!amd::Os::pathExists(headerPath)) {
            bool ret = amd::Os::createPath(headerPath);
            assert(ret && "failed creating path!");
            newDirs.push_back(headerPath);
        }
        std::string headerFullName
        = headerPath + amd::Os::fileSeparator() + headerIncludeName;
        headerFileNames[i] = headerFullName;
        f.open(headerFullName.c_str(), std::fstream::out);
        //Should we allow asserts
        assert(!f.fail() && "failed creating header file!");
        f.write(headers[i]->c_str(), headers[i]->length());
        f.close();
    }

    //Create Binary
    binaryElf_ = g_complibApi._aclBinaryInit(sizeof(aclBinary),
        &target,
        &binOpts_,
        &errorCode);

    if( errorCode!=ACL_SUCCESS ) {
        buildLog_ += "Error while compiling opencl source:\
                     aclBinary init failure \n";
        LogWarning("aclBinaryInit failed");
        return false;
    }

    //Insert opencl into binary
    errorCode = g_complibApi._aclInsertSection(device().compiler(),
        binaryElf_,
        sourceCode.c_str(),
        strlen(sourceCode.c_str()),
        aclSOURCE);

    if ( errorCode != ACL_SUCCESS )    {
        buildLog_ += "Error while converting to BRIG: \
                     Inserting openCl Source \n";
    }

    //Set the options for the compiler
    //Set the include path for the temp folder that contains the includes
    if(!headers.empty()) {
        this->compileOptions_.append(" -I");
        this->compileOptions_.append(tempFolder);
    }

    //Add only for CL2.0 and later
    if (options->oVariables->CLStd[2] >= '2') {
        std::stringstream opts;
        opts << " -D" << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE="
            << device().info().maxGlobalVariableSize_;
        compileOptions_.append(opts.str());
    }

    //Compile source to IR
    this->compileOptions_.append(preprocessorOptions(options));
    this->compileOptions_.append(codegenOptions(options));

    errorCode = g_complibApi._aclCompile(device().compiler(),
        binaryElf_,
        //"-Wf,--support_all_extensions",
        this->compileOptions_.c_str(),
        ACL_TYPE_OPENCL,
        ACL_TYPE_LLVMIR_BINARY,
        logFunction);
    buildLog_ += g_complibApi._aclGetCompilerLog(device().compiler());
    if( errorCode!=ACL_SUCCESS ) {
        LogWarning("aclCompile failed");
        buildLog_ += "Error while compiling \
                     opencl source: Compiling CL to IR";
        return false;
    }
    // Save the binary in the interface class
    saveBinaryAndSetType(TYPE_COMPILED);
    return true;
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

#if defined(WITH_LIGHTNING_COMPILER)
#if defined(ATI_OS_LINUX)
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void
checkLLVM_BIN()
{
    if (llvmBin_.empty()) {
        Dl_info info;
        if (dladdr((const void*)&amd::Device::init, &info)) {
            llvmBin_ = dirname(strdup(info.dli_fname));
            size_t pos = llvmBin_.rfind("lib");
            if (pos != std::string::npos) {
                llvmBin_.replace(pos, 3, "bin");
            }
        }
    }
#if defined(DEBUG)
    std::string clangExe(llvmBin_ + "/clang");
    struct stat buf;
    if (stat(clangExe.c_str(), &buf)) {
        std::string msg("Could not find the Clang binary in " + llvmBin_);
        LogWarning(msg.c_str());
    }
#endif // defined(DEBUG)
}
#endif // defined(ATI_OS_LINUX)

std::auto_ptr<amd::opencl_driver::Compiler>
HSAILProgram::newCompilerInstance()
{
#if defined(ATI_OS_LINUX)
    pthread_once(&once, checkLLVM_BIN);
#endif // defined(ATI_OS_LINUX)
    return std::auto_ptr<amd::opencl_driver::Compiler>(
        amd::opencl_driver::CompilerFactory().CreateAMDGPUCompiler(llvmBin_));
}
#endif // defined(WITH_LIGHTNING_COMPILER)

} // namespace roc
#endif // WITHOUT_GPU_BACKEND
