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
#include "opencl-c.amdgcn.inc"
#else // !defined(WITH_LIGHTNING_COMPILER)
#include "roccompilerlib.hpp"
#endif // !defined(WITH_LIGHTNING_COMPILER)
#include "utils/options.hpp"
#include <cstdio>

//CLC_IN_PROCESS_CHANGE
extern int openclFrontEnd(const char* cmdline, std::string*, std::string* typeInfo = NULL);

namespace roc {

/* Temporary log function for the compiler library */
static void logFunction(const char* msg, size_t size)
{
	std::cout<< "Compiler Log: " << msg << std::endl;
}

static int programsCount = 0;

#if defined(WITH_LIGHTNING_COMPILER)
bool
HSAILProgram::compileImpl_LC(const std::string& sourceCode,
                             const std::vector<const std::string*>& headers,
		                     const char** headerIncludeNames,
		                     amd::option::Options* options)
{
    std::vector<amd::opencl_driver::Data*> inputs;
    amd::opencl_driver::Data* src = device().compiler()->NewBufferReference(
                                                amd::opencl_driver::DT_CL,sourceCode.c_str(),
                                                sourceCode.length());
    if (src == NULL) {
        buildLog_ += "Error while creating data from source code";
        return false;
    }
    inputs.push_back(src);

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

        amd::opencl_driver::Data* inc = device().compiler()->NewFileReference(
                                                amd::opencl_driver::DT_CL_HEADER,
                                                headerFileNames[i]);
        inputs.push_back(inc);
    }


    //Set the options for the compiler
    //Set the include path for the temp folder that contains the includes
    if(!headers.empty()) {
        compileOptions_.append(" -I");
        compileOptions_.append(tempFolder);
    }

    //Add only for CL2.0 and later
    if (options->oVariables->CLStd[2] >= '2') {
        std::stringstream opts;
        opts << " -D" << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE="
            << device().info().maxGlobalVariableSize_;
        compileOptions_.append(opts.str());
    }

    amd::opencl_driver::File* pch = device().compiler()->NewTempFile(
            amd::opencl_driver::DT_CL_HEADER);
    if (pch == NULL || !pch->WriteData((const char*) opencl_c_amdgcn,
            opencl_c_amdgcn_size)) {
        buildLog_ += "Error while opening the opencl-c header ";
        return false;
    }

    // FIXME_lmoriche: Force OpenCL-C 2.0, since the built-ins are built that way.
    compileOptions_.append(" -Xclang -cl-std=CL2.0");
    compileOptions_.append(" -Xclang -include-pch -Xclang " + pch->Name());

    compileOptions_.append(hsailOptions());

    amd::opencl_driver::Buffer* output = device().compiler()->NewBuffer(amd::opencl_driver::DT_LLVM_BC);
    if (output == NULL) {
        buildLog_ += "Error while creating buffer for the LLVM bitcode";
        return false;
    }

    // Tokenize the options string into a vector of strings
    std::istringstream strstr(compileOptions_);
    std::istream_iterator<std::string> sit(strstr), end;
    std::vector<std::string> optionsvec(sit, end);

    // Compile source to IR
    bool ret = device().compiler()->CompileToLLVMBitcode(inputs, output, optionsvec);
    buildLog_ += device().compiler()->Output().c_str();
    if (!ret) {
        buildLog_ += "Error while compiling \
                     opencl source: Compiling CL to IR";
        return false;
    }

    //  save the source code
    //Create Binary
    codeObjBinary_ = new CodeObjBinary();

    openCLSource_ = sourceCode;
    codeObjBinary_->saveIR(std::string(output->Buf().begin(), output->Buf().end()));
    return true;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

bool
HSAILProgram::compileImpl(const std::string& sourceCode,
                       const std::vector<const std::string*>& headers,
		       const char** headerIncludeNames,
		       amd::option::Options* options)
{
#if defined(WITH_LIGHTNING_COMPILER)
    return compileImpl_LC(sourceCode, headers, headerIncludeNames, options);
#else // !defined(WITH_LIGHTNING_COMPILER)
    acl_error errorCode;
    aclTargetInfo target;

    //Defaulting to bonaire
    //Todo (sramalin) : Query the device for asic type-
    //Defaulting to Bonair for now.
    target = g_complibApi._aclGetTargetInfo(LP64_SWITCH("hsail","hsail64"), "Bonaire",
        &errorCode);

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
    this->compileOptions_.append(hsailOptions());
    
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
}
#endif // WITHOUT_GPU_BACKEND
