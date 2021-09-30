/*
Copyright (c) 2022 - Present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hiprtcInternal.hpp"

#include "vdi_common.hpp"
#include "utils/flags.hpp"

namespace hiprtc {
using namespace helpers;
RTCProgram::RTCProgram(std::string name_) : name(name_) {
  std::call_once(amd::Comgr::initialized, amd::Comgr::LoadLib);
  if (amd::Comgr::create_data_set(&compileInput) != AMD_COMGR_STATUS_SUCCESS ||
      amd::Comgr::create_data_set(&linkInput) != AMD_COMGR_STATUS_SUCCESS ||
      amd::Comgr::create_data_set(&execInput) != AMD_COMGR_STATUS_SUCCESS) {
    crashWithMessage("Failed to allocate internal hiprtc structure");
  }

  // Add internal header
  if (!addBuiltinHeader()) {
    crashWithMessage("Unable to add internal header");
  }

  // Add compile options
  const std::string hipVerOpt{"--hip-version=" + std::to_string(HIP_VERSION_MAJOR) + '.' +
                              std::to_string(HIP_VERSION_MINOR) + '.' +
                              std::to_string(HIP_VERSION_PATCH)};
  const std::string hipVerMajor{"-DHIP_VERSION_MAJOR=" + std::to_string(HIP_VERSION_MAJOR)};
  const std::string hipVerMinor{"-DHIP_VERSION_MINOR=" + std::to_string(HIP_VERSION_MINOR)};
  const std::string hipVerPatch{"-DHIP_VERSION_PATCH=" + std::to_string(HIP_VERSION_PATCH)};

  compileOptions.reserve(18);  // count of options below
  compileOptions.push_back("-O3");

#ifdef HIPRTC_EARLY_INLINE
  compileOptions.push_back("-mllvm");
  compileOptions.push_back("-amdgpu-early-inline-all");
#endif
  compileOptions.push_back("-mllvm");
  compileOptions.push_back("-amdgpu-prelink");

  if (GPU_ENABLE_WGP_MODE) compileOptions.push_back("-mcumode");

  if (!GPU_ENABLE_WAVE32_MODE) compileOptions.push_back("-mwavefrontsize64");

  compileOptions.push_back(hipVerOpt);
  compileOptions.push_back(hipVerMajor);
  compileOptions.push_back(hipVerMinor);
  compileOptions.push_back(hipVerPatch);
  compileOptions.push_back("-D__HIPCC_RTC__");
  compileOptions.push_back("-include");
  compileOptions.push_back("hiprtc_runtime.h");
  compileOptions.push_back("-std=c++14");
  compileOptions.push_back("-nogpuinc");
#ifdef _WIN32
  compileOptions.push_back("-target x86_64-pc-windows-msvc");
  compileOptions.push_back("-fms-extensions");
  compileOptions.push_back("-fms-compatibility");
#endif

  if (!GPU_ENABLE_WAVE32_MODE) linkOptions.push_back("wavefrontsize64");

  exeOptions.push_back("-O3");
  exeOptions.push_back("-mllvm");
  exeOptions.push_back("-amdgpu-internalize-symbols");
  exeOptions.push_back("-mcumode");
  if (!GPU_ENABLE_WAVE32_MODE) exeOptions.push_back("-mwavefrontsize64");
}

bool RTCProgram::addSource(const std::string& source, const std::string& name) {
  if (source.size() == 0 || name.size() == 0) {
    LogError("Error in hiprtc: source or name is of size 0 in addSource");
    return false;
  }
  sourceCode += source;
  sourceName = name;
  return true;
}

// addSource_impl is a different function because we need to add source when we track mangled
// objects
bool RTCProgram::addSource_impl() {
  std::vector<char> vsource(sourceCode.begin(), sourceCode.end());
  if (!addCodeObjData(compileInput, vsource, sourceName, AMD_COMGR_DATA_KIND_SOURCE)) {
    return false;
  }
  return true;
}

bool RTCProgram::addHeader(const std::string& source, const std::string& name) {
  if (source.size() == 0 || name.size() == 0) {
    LogError("Error in hiprtc: source or name is of size 0 in addHeader");
    return false;
  }
  std::vector<char> vsource(source.begin(), source.end());
  if (!addCodeObjData(compileInput, vsource, name, AMD_COMGR_DATA_KIND_INCLUDE)) {
    return false;
  }
  return true;
}

bool RTCProgram::addBuiltinHeader() {
  std::vector<char> source(__hipRTC_header, __hipRTC_header + __hipRTC_header_size);
  std::string name{"hiprtc_runtime.h"};
  if (!addCodeObjData(compileInput, source, name, AMD_COMGR_DATA_KIND_INCLUDE)) {
    return false;
  }
  return true;
}

bool RTCProgram::transformOptions() {
  auto getValueOf = [](const std::string& option) {
    std::string res;
    auto f = std::find(option.begin(), option.end(), '=');
    if (f != option.end()) res = std::string(f + 1, option.end());
    return res;
  };

  for (auto& i : compileOptions) {
    if (i == "-hip-pch") {
      LogInfo(
          "-hip-pch is deprecated option, has no impact on execution of new hiprtc programs, it "
          "can be removed");
      i.clear();
      continue;
    }
    // Some rtc samples use --gpu-architecture
    if (i.rfind("--gpu-architecture=", 0) == 0) {
      LogInfo("--gpu-architecture is nvcc option, transforming it to --offload-arch option");
      auto val = getValueOf(i);
      i = "--offload-arch=" + val;
      continue;
    }
    if (i == "--save-temps") {
      settings.dumpISA = true;
      continue;
    }
  }

  if (auto res = std::find_if(
          compileOptions.begin(), compileOptions.end(),
          [](const std::string& str) { return str.find("--offload-arch=") != std::string::npos; });
      res != compileOptions.end()) {
    auto isaName = getValueOf(*res);
    isa = "amdgcn-amd-amdhsa--" + isaName;
    settings.offloadArchProvided = true;
    return true;
  }
  buildLog +=
      "Error: Please provide architecture for which code is to be "
      "generated.\n";
  return false;
}

amd::Monitor RTCProgram::lock_("HIPRTC Program", true);

bool RTCProgram::compile(const std::vector<std::string>& options) {
  amd::ScopedLock lock(lock_); // Lock, because LLVM is not multi threaded

  if (!addSource_impl()) {
    LogError("Error in hiprtc: unable to add source code");
    return false;
  }

  // Append compile options
  compileOptions.reserve(compileOptions.size() + options.size());
  compileOptions.insert(compileOptions.end(), options.begin(), options.end());

  if (!transformOptions()) {
    LogError("Error in hiprtc: unable to transform options");
    return false;
  }

  std::vector<char> LLVMBitcode;
  if (!compileToBitCode(compileInput, isa, compileOptions, buildLog, LLVMBitcode)) {
    LogError("Error in hiprtc: unable to compile source to bitcode");
    return false;
  }

  std::string linkFileName = "linked";
  if (!addCodeObjData(linkInput, LLVMBitcode, linkFileName, AMD_COMGR_DATA_KIND_BC)) {
    LogError("Error in hiprtc: unable to add linked code object");
    return false;
  }

  std::vector<char> LinkedLLVMBitcode;
  if (!linkLLVMBitcode(linkInput, isa, linkOptions, buildLog, LinkedLLVMBitcode)) {
    LogError("Error in hiprtc: unable to add device libs to linked bitcode");
    return false;
  }

  std::string linkedFileName = "LLVMBitcode.bc";
  if (!addCodeObjData(execInput, LinkedLLVMBitcode, linkedFileName, AMD_COMGR_DATA_KIND_BC)) {
    LogError("Error in hiprtc: unable to add device libs linked code object");
    return false;
  }

  if (settings.dumpISA) {
    if (!dumpIsaFromBC(execInput, isa, exeOptions, name, buildLog)) {
      LogError("Error in hiprtc: unable to dump isa code");
      return false;
    }
  }

  if (!createExecutable(execInput, isa, exeOptions, buildLog, executable)) {
    LogError("Error in hiprtc: unable to create executable");
    return false;
  }

  std::vector<std::string> mangledNames;
  if (!fillDemangledNames(executable, mangledNames)) {
    LogError("Error in hiprtc: unable to fill demangled names");
    return false;
  }

  if (!getMangledNames(mangledNames, strippedNames, demangledNames)) {
    LogError("Error in hiprtc: unable to get mangled names");
    return false;
  }

  return true;
}

bool RTCProgram::trackMangledName(std::string& name) {
  amd::ScopedLock lock(lock_);

  if (name.size() == 0) return false;

  std::string strippedName = name;
  if (strippedName.back() == ')') {
    strippedName.pop_back();
    strippedName.erase(0, strippedName.find('('));
  }
  if (strippedName.front() == '&') {
    strippedName.erase(0, 1);
  }

  std::string strippedNameNoSpace = strippedName;
  strippedNameNoSpace.erase(std::remove_if(strippedNameNoSpace.begin(),
                                           strippedNameNoSpace.end(),
                                           [](unsigned char c) {
                                             return std::isspace(c);
                                           }), strippedNameNoSpace.end());

  strippedNames.insert(std::pair<std::string, std::string>(name, strippedNameNoSpace));
  demangledNames.insert(std::pair<std::string, std::string>(strippedName, ""));

  const auto var{"__hiprtc_" + std::to_string(strippedNames.size())};
  const auto code{"\nextern \"C\" constexpr auto " + var + " = " + name + ";\n"};

  sourceCode += code;
  return true;
}

bool RTCProgram::getDemangledName(const char* name_expression, const char** loweredName) {
  std::string name = name_expression;
  if (auto res = strippedNames.find(name); res != strippedNames.end()) {
    if (auto dres = demangledNames.find(res->second); dres != demangledNames.end()) {
      if (dres->second.size() != 0) {
        *loweredName = dres->second.c_str();
        return true;
      } else
        return false;
    }
  }
  if (auto dres = demangledNames.find(name); dres != demangledNames.end()) {
    if (dres->second.size() != 0) {
      *loweredName = dres->second.c_str();
      return true;
    }
    return false;
  }
  return false;
}

}  // namespace hiprtc
