/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "os/os.hpp"
#include "device/gpu/gpudevice.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpukernel.hpp"
#include "utils/options.hpp"
#include <cstdio>

// CLC_IN_PROCESS_CHANGE
extern int openclFrontEnd(const char* cmdline, std::string*, std::string* typeInfo = NULL);

namespace gpu {

static int programsCount = 0;

bool NullProgram::compileImpl(const std::string& src,
                              const std::vector<const std::string*>& headers,
                              const char** headerIncludeNames, amd::option::Options* options) {
  std::string sourceCode = src;

  if (dev().settings().debugFlags_ & Settings::CheckForILSource) {
    size_t inc = sourceCode.find("il_cs_", 0);
    if (inc != std::string::npos) {
      // CL program is an IL program
      ilProgram_ = sourceCode;
      return true;
    }
  }


  std::string tempFolder = amd::Os::getTempPath();
  std::string tempFileName = amd::Os::getTempFileName();

  if (dev().settings().debugFlags_ & Settings::StubCLPrograms) {
    std::stringstream fileName;
    std::fstream stubRead;
    // Dump the IL function
    fileName << "program_" << programsCount++ << ".cl";
    stubRead.open(fileName.str().c_str(), (std::fstream::in | std::fstream::binary));
    // Check if we have OpenCL program
    if (stubRead.is_open()) {
      // Find the stream size
      stubRead.seekg(0, std::fstream::end);
      size_t size = stubRead.tellg();
      stubRead.seekg(0, std::ios::beg);

      char* data = new char[size];
      stubRead.read(data, size);
      stubRead.close();

      sourceCode.assign(data, size);
      delete[] data;
    } else {
      std::fstream stubWrite;
      stubWrite.open(fileName.str().c_str(), (std::fstream::out | std::fstream::binary));
      stubWrite << sourceCode;
      stubWrite.close();
    }
  }

  std::fstream f;
  std::vector<std::string> headerFileNames(headers.size());
  std::vector<std::string> newDirs;
  for (size_t i = 0; i < headers.size(); ++i) {
    std::string headerPath = tempFolder;
    std::string headerIncludeName(headerIncludeNames[i]);
    // replace / in path with current os's file separator
    if (amd::Os::fileSeparator() != '/') {
      for (auto& it : headerIncludeName) {
        if (it == '/') it = amd::Os::fileSeparator();
      }
    }
    size_t pos = headerIncludeName.rfind(amd::Os::fileSeparator());
    if (pos != std::string::npos) {
      headerPath += amd::Os::fileSeparator();
      headerPath += headerIncludeName.substr(0, pos);
      headerIncludeName = headerIncludeName.substr(pos + 1);
    }
    if (!amd::Os::pathExists(headerPath)) {
      bool ret = amd::Os::createPath(headerPath);
      assert(ret && "failed creating path!");
      newDirs.push_back(headerPath);
    }
    std::string headerFullName = headerPath + amd::Os::fileSeparator() + headerIncludeName;
    headerFileNames[i] = headerFullName;
    f.open(headerFullName.c_str(), std::fstream::out);
    assert(!f.fail() && "failed creating header file!");
    f.write(headers[i]->c_str(), headers[i]->length());
    f.close();
  }

  acl_error err;
  const aclTargetInfo& targInfo = info();

  aclBinaryOptions binOpts = {0};
  binOpts.struct_size = sizeof(binOpts);
  binOpts.elfclass = targInfo.arch_id == aclAMDIL64 ? ELFCLASS64 : ELFCLASS32;
  binOpts.bitness = ELFDATA2LSB;
  binOpts.alloc = &::malloc;
  binOpts.dealloc = &::free;

  aclBinary* bin = aclBinaryInit(sizeof(aclBinary), &targInfo, &binOpts, &err);
  if (err != ACL_SUCCESS) {
    LogWarning("aclBinaryInit failed");
    return false;
  }

  if (ACL_SUCCESS !=
      aclInsertSection(dev().amdilCompiler(), bin, sourceCode.c_str(), sourceCode.size(), aclSOURCE)) {
    LogWarning("aclInsertSection failed");
    aclBinaryFini(bin);
    return false;
  }

  // temporary solution to synchronize buildNo between runtime and complib
  // until we move runtime inside complib
  ((amd::option::Options*)bin->options)->setBuildNo(options->getBuildNo());

  std::stringstream opts;
  std::string token;
  opts << options->origOptionStr.c_str();

  if (options->origOptionStr.find("-cl-std=CL") == std::string::npos) {
    switch (dev().settings().oclVersion_) {
      case OpenCL10:
        opts << " -cl-std=CL1.0";
        break;
      case OpenCL11:
        opts << " -cl-std=CL1.1";
        break;
      case OpenCL20:
      case OpenCL21:
      default:
      case OpenCL12:
        opts << " -cl-std=CL1.2";
        break;
    }
  }

  // FIXME: Should we prefix everything with -Wf,?
  std::istringstream iss(options->clcOptions);
  while (getline(iss, token, ' ')) {
    if (!token.empty()) {
      // Check if this is a -D option
      if (token.compare("-D") == 0) {
        // It is, skip payload
        getline(iss, token, ' ');
        continue;
      }
      opts << " -Wf," << token;
    }
  }

  if (!headers.empty()) {
    opts << " -I" << tempFolder;
  }

  if (!dev().settings().imageSupport_) {
    opts << " -fno-image-support";
  }

  if (dev().settings().reportFMAF_) {
    opts << " -mfast-fmaf";
  }

  if (dev().settings().reportFMA_) {
    opts << " -mfast-fma";
  }

  iss.clear();
  iss.str(device().info().extensions_);
  while (getline(iss, token, ' ')) {
    if (!token.empty()) {
      opts << " -D" << token << "=1";
    }
  }

  std::string newOpt = opts.str();
  size_t pos = newOpt.find("-fno-bin-llvmir");
  while (pos != std::string::npos) {
    newOpt.erase(pos, 15);
    pos = newOpt.find("-fno-bin-llvmir");
  }

  err = aclCompile(dev().amdilCompiler(), bin, newOpt.c_str(), ACL_TYPE_OPENCL, ACL_TYPE_LLVMIR_BINARY,
                   NULL);

  buildLog_ += aclGetCompilerLog(dev().amdilCompiler());

  if (err != ACL_SUCCESS) {
    LogWarning("aclCompile failed");
    aclBinaryFini(bin);
    return false;
  }

  size_t len = 0;
  const void* ir = aclExtractSection(dev().amdilCompiler(), bin, &len, aclLLVMIR, &err);
  if (err != ACL_SUCCESS) {
    LogWarning("aclExtractSection failed");
    aclBinaryFini(bin);
    return false;
  }

  llvmBinary_.assign(reinterpret_cast<const char*>(ir), len);
  elfSectionType_ = amd::OclElf::LLVMIR;
  aclBinaryFini(bin);

  for (size_t i = 0; i < headerFileNames.size(); ++i) {
    amd::Os::unlink(headerFileNames[i].c_str());
  }
  for (size_t i = 0; i < newDirs.size(); ++i) {
    amd::Os::removePath(newDirs[i]);
  }

#ifdef _WIN32
  amd::Os::unlink(tempFileName);
#endif

  if (clBinary()->saveSOURCE()) {
    clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, sourceCode.data(), sourceCode.size());
  }
  if (clBinary()->saveLLVMIR()) {
    clBinary()->elfOut()->addSection(amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(),
                                     false);
    // store the original compile options
    clBinary()->storeCompileOptions(compileOptions_);
  }

  return true;
}

int NullProgram::compileBinaryToIL(amd::option::Options* options) {
  acl_error err;
  const aclTargetInfo& targInfo = info();

  aclBinaryOptions binOpts = {0};
  binOpts.struct_size = sizeof(binOpts);
  binOpts.elfclass = targInfo.arch_id == aclAMDIL64 ? ELFCLASS64 : ELFCLASS32;
  binOpts.bitness = ELFDATA2LSB;
  binOpts.alloc = &::malloc;
  binOpts.dealloc = &::free;

  aclBinary* bin = aclBinaryInit(sizeof(aclBinary), &targInfo, &binOpts, &err);
  if (err != ACL_SUCCESS) {
    LogWarning("aclBinaryInit failed");
    return CL_BUILD_PROGRAM_FAILURE;
  }
  aclSections_0_8 spirFlag;
  _acl_type_enum_0_8 aclTypeBinaryUsed;
  if (std::string::npos != options->clcOptions.find("--spirv") ||
      elfSectionType_ == amd::OclElf::SPIRV) {
    spirFlag = aclSPIRV;
    aclTypeBinaryUsed = ACL_TYPE_SPIRV_BINARY;
  } else if (std::string::npos != options->clcOptions.find("--spir") ||
             elfSectionType_ == amd::OclElf::SPIR) {
    spirFlag = aclSPIR;
    aclTypeBinaryUsed = ACL_TYPE_SPIR_BINARY;
  } else {
    spirFlag = aclLLVMIR;
    aclTypeBinaryUsed = ACL_TYPE_LLVMIR_BINARY;
  }

  if (ACL_SUCCESS !=
      aclInsertSection(dev().amdilCompiler(), bin, llvmBinary_.data(), llvmBinary_.size(), spirFlag)) {
    LogWarning("aclInsertSection failed");
    aclBinaryFini(bin);
    return CL_BUILD_PROGRAM_FAILURE;
  }

  // pass kernel argument alignment info to compiler lib through option str
  std::string optionStr = options->origOptionStr;
  if (options->origOptionStr.find("kernel-arg-alignment") == std::string::npos) {
    char s[256];
    sprintf(s, " -Wb,-kernel-arg-alignment=%d", dev().info().memBaseAddrAlign_ / 8);
    optionStr += s;
  }

  // temporary solution to synchronize buildNo between runtime and complib
  // until we move runtime inside complib
  ((amd::option::Options*)bin->options)->setBuildNo(options->getBuildNo());

  aclType type = ACL_TYPE_CG;
  // If option bin-bif30 is set, generate BIF 3.0 binary
  if (options->oVariables->BinBIF30) {
    type = ACL_TYPE_ISA;
  }

  err = aclCompile(dev().amdilCompiler(), bin, optionStr.c_str(), aclTypeBinaryUsed, type, NULL);
  buildLog_ += aclGetCompilerLog(dev().amdilCompiler());

  if (err != ACL_SUCCESS) {
    LogWarning("aclCompile failed");
    aclBinaryFini(bin);
    return CL_BUILD_PROGRAM_FAILURE;
  }

  if (options->oVariables->BinBIF30) {
    acl_error err;
    char* binaryIn = nullptr;
    size_t size;
    err = aclWriteToMem(bin, reinterpret_cast<void**>(&binaryIn), &size);
    if (err != ACL_SUCCESS) {
      LogWarning("aclWriteToMem failed");
      aclBinaryFini(bin);
      return CL_BUILD_PROGRAM_FAILURE;
    }
    clBinary()->saveBIFBinary(binaryIn, size);
    aclFreeMem(bin, binaryIn);
  }

  size_t len = 0;
  const void* amdil = aclExtractSection(dev().amdilCompiler(), bin, &len, aclCODEGEN, &err);
  if (err != ACL_SUCCESS) {
    LogWarning("aclExtractSection failed");
    aclBinaryFini(bin);
    return CL_BUILD_PROGRAM_FAILURE;
  }

  ilProgram_.assign(reinterpret_cast<const char*>(amdil), len);
  aclBinaryFini(bin);

  return CL_SUCCESS;
}

}  // namespace gpu
