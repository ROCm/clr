//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "os/os.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palkernel.hpp"
#include "utils/options.hpp"
#include <cstdio>

//CLC_IN_PROCESS_CHANGE
extern int openclFrontEnd(const char* cmdline, std::string*, std::string* typeInfo = nullptr);

namespace pal {

bool
HSAILProgram::compileImpl(
    const std::string& sourceCode,
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames,
    amd::option::Options* options)
{
    acl_error errorCode;
    aclTargetInfo target;

    std::string arch = "hsail";
    if (dev().settings().use64BitPtr_) {
        arch += "64";
    }
    target = aclGetTargetInfo(arch.c_str(),
        dev().info().name_, &errorCode);

    // end if asic info is ready
    // We dump the source code for each program (param: headers)
    // into their filenames (headerIncludeNames) into the TEMP
    // folder specific to the OS and add the include path while
    // compiling

    // Find the temp folder for the OS
    std::string tempFolder = amd::Os::getTempPath();
    std::string tempFileName = amd::Os::getTempFileName();

    // Iterate through each source code and dump it into tmp
    std::fstream f;
    std::vector<std::string> headerFileNames(headers.size());
    std::vector<std::string> newDirs;
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string headerPath = tempFolder;
        std::string headerIncludeName(headerIncludeNames[i]);
        // replace / in path with current os's file separator
        if (amd::Os::fileSeparator() != '/') {
            for (std::string::iterator it = headerIncludeName.begin(),
                end = headerIncludeName.end(); it != end; ++it) {
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
        std::string headerFullName =
            headerPath + amd::Os::fileSeparator() + headerIncludeName;
        headerFileNames[i] = headerFullName;
        f.open(headerFullName.c_str(), std::fstream::out);
        // Should we allow asserts
        assert(!f.fail() && "failed creating header file!");
        f.write(headers[i]->c_str(), headers[i]->length());
        f.close();
    }

    // Create Binary
    binaryElf_ = aclBinaryInit(sizeof(aclBinary),
        &target, &binOpts_, &errorCode);
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Error: aclBinary init failure\n";
        LogWarning("aclBinaryInit failed");
        return false;
    }

    // Insert opencl into binary
    errorCode = aclInsertSection(dev().compiler(), binaryElf_,
        sourceCode.c_str(), strlen(sourceCode.c_str()), aclSOURCE);
    if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Error: Inserting openCl Source to binary\n";
    }

    // Set the options for the compiler
    // Set the include path for the temp folder that contains the includes
    if (!headers.empty()) {
        compileOptions_.append(" -I");
        compileOptions_.append(tempFolder);
    }

    //Add only for CL2.0 and above
    if (options->oVariables->CLStd[2] >= '2') {
        std::stringstream opts;
        opts << " -D" << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE="
            << device().info().maxGlobalVariableSize_;
        compileOptions_.append(opts.str());
    }

#if !defined(_LP64) && defined(ATI_OS_LINUX)
    if (options->origOptionStr.find("-cl-std=CL2.0") != std::string::npos && !dev().settings().force32BitOcl20_) {
        errorCode = ACL_UNSUPPORTED;
        LogWarning("aclCompile failed");
        return false;
    }
#endif

    // Compile source to IR
    compileOptions_.append(hsailOptions());
    errorCode = aclCompile(dev().compiler(), binaryElf_, compileOptions_.c_str(),
        ACL_TYPE_OPENCL, ACL_TYPE_LLVMIR_BINARY, nullptr);
    buildLog_ += aclGetCompilerLog(dev().compiler());
    if (errorCode != ACL_SUCCESS) {
        LogWarning("aclCompile failed");
        buildLog_ += "Error: Compiling CL to IR\n";
        return false;
    }

    clBinary()->storeCompileOptions(compileOptions_);
    // Save the binary in the interface class
    size_t size = 0;
    void* mem = nullptr;
    aclWriteToMem(binaryElf_, &mem, &size);
    setBinary(static_cast<char*>(mem), size);

    // Save the binary inside the program
    // The FSAILProgram will be responsible to free it during destruction
    rawBinary_ = mem;
    return true;
}

}   // namespace pal
