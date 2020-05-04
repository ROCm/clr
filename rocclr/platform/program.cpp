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

#include "top.hpp"
#include "device/appprofile.hpp"
#include "platform/program.hpp"
#include "platform/context.hpp"
#include "utils/options.hpp"
#include "utils/libUtils.h"
#include "utils/bif_section_labels.hpp"
#include "acl.h"

#include <cstdlib>  // for malloc
#include <cstring>  // for strcmp
#include <sstream>
#include <fstream>
#include <iostream>
#include <utility>

namespace amd {

static void remove_g_option(std::string &option)
{
  // Remove " -g " option from application.
  // People can still add -g in AMD_OCL_BUILD_OPTIONS_APPEND, if it is so desired.
  std::string g_str("-g");
  std::size_t g_pos = 0;
  while ((g_pos = option.find(g_str, g_pos)) != std::string::npos) {
    if ((g_pos == 0 || option[g_pos - 1] == ' ') &&
       (g_pos + 2 == option.size() || option[g_pos + 2] == ' ')) {
       option.erase(g_pos, g_str.size());
    } else {
       g_pos += g_str.size();
    }
  }

  return;
}

Program::~Program() {
  // Destroy all device programs
  for (const auto& it : devicePrograms_) {
    delete it.second;
  }

  for (const auto& it : binary_) {
    const binary_t& Bin = it.second;
    if (std::get<2>(Bin)) {
      delete[] std::get<0>(Bin);
    }
  }

  if (std::get<1>(mmap_) > 0) {
    amd::Os::MemoryUnmapFile(std::get<0>(mmap_), std::get<1>(mmap_));
  }

  delete symbolTable_;
  //! @todo Make sure we have destroyed all CPU specific objects
}

const Symbol* Program::findSymbol(const char* kernelName) const {
  // avoid seg. fault if the program has not built yet
  if (symbolTable_ == NULL) {
    return NULL;
  }

  const auto it = symbolTable_->find(kernelName);
  return (it == symbolTable_->cend()) ? NULL : &it->second;
}

int32_t Program::addDeviceProgram(Device& device, const void* image, size_t length,
                                 bool make_copy, amd::option::Options* options) {
  if (image != NULL &&  !amd::isElfMagic((const char*)image)) {
    if (device.settings().useLightning_) {
      return CL_INVALID_BINARY;
    }
#if defined(WITH_COMPILER_LIB)
    else if (!aclValidateBinaryImage(
          image, length, language_ == SPIRV ? BINARY_TYPE_SPIRV : BINARY_TYPE_ELF | BINARY_TYPE_LLVM)) {
      return CL_INVALID_BINARY;
    }
#endif // !defined(WITH_COMPILER_LIB)
  }

  // Check if the device is already associated with this program
  if (deviceList_.find(&device) != deviceList_.end()) {
    return CL_INVALID_VALUE;
  }

  Device& rootDev = device;

  // if the rootDev is already associated with a program
  if (devicePrograms_[&rootDev] != NULL) {
    return CL_SUCCESS;
  }
  bool emptyOptions = false;
  amd::option::Options emptyOpts;
  if (options == NULL) {
    options = &emptyOpts;
    emptyOptions = true;
  }

#if defined(WITH_COMPILER_LIB)
  if (image != NULL && length != 0 && aclValidateBinaryImage(image, length, BINARY_TYPE_ELF)) {
    acl_error errorCode;
    aclBinary* binary = aclReadFromMem(image, length, &errorCode);
    if (errorCode != ACL_SUCCESS) {
      return CL_INVALID_BINARY;
    }
    const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symOpenclCompilerOptions);
    assert(symbol && "symbol not found");
    std::string symName = std::string(symbol->str[bif::PRE]) + std::string(symbol->str[bif::POST]);
    size_t symSize = 0;
    const void* opts = aclExtractSymbol(device.binCompiler(), binary, &symSize, aclCOMMENT,
                                        symName.c_str(), &errorCode);
    // if we have options from binary and input options was not specified
    if (opts != NULL && emptyOptions) {
      std::string sBinOptions = std::string((char*)opts, symSize);
      if (!amd::option::parseAllOptions(sBinOptions, *options, false, false)) {
        programLog_ = options->optionsLog();
        LogError("Parsing compilation options from binary failed.");
        return CL_INVALID_COMPILER_OPTIONS;
      }
    }
    options->oVariables->Legacy = !device.settings().useLightning_ ?
                                     isAMDILTarget(*aclutGetTargetInfo(binary)) :
                                     isHSAILTarget(*aclutGetTargetInfo(binary));
    aclBinaryFini(binary);
  }
#endif // defined(WITH_COMPILER_LIB)
  options->oVariables->BinaryIsSpirv = language_ == SPIRV;
  device::Program* program = rootDev.createProgram(*this, options);
  if (program == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  if (image != NULL) {
    const uint8_t* memory = std::get<0>(binary(rootDev));
    // clone 'binary' (it is owned by the host thread).
    if (memory == NULL) {
      if (make_copy) {
        auto *image_copy = new (std::nothrow) uint8_t[length];
        if (image_copy == NULL) {
          delete program;
          return CL_OUT_OF_HOST_MEMORY;
        }

        ::memcpy(image_copy, image, length);
        memory = image_copy;
      }
      else {
        memory = static_cast<const uint8_t*>(image);
      }

      // Save the original image
      binary_[&rootDev] = std::make_tuple(memory, length, make_copy);
    }

    if (!program->setBinary(reinterpret_cast<const char*>(memory), length)) {
      delete program;
      return CL_INVALID_BINARY;
    }
  }

  devicePrograms_[&rootDev] = program;

  deviceList_.insert(&device);
  return CL_SUCCESS;
}

device::Program* Program::getDeviceProgram(const Device& device) const {
  const auto it = devicePrograms_.find(&device);
  if (it == devicePrograms_.cend()) {
    return NULL;
  }
  return it->second;
}

Monitor Program::buildLock_("OCL build program", true);

int32_t Program::compile(const std::vector<Device*>& devices, size_t numHeaders,
                        const std::vector<const Program*>& headerPrograms,
                        const char** headerIncludeNames, const char* options,
                        void(CL_CALLBACK* notifyFptr)(cl_program, void*), void* data,
                        bool optionChangable) {
  ScopedLock sl(buildLock_);

  int32_t retval = CL_SUCCESS;

  // Clear the program object
  clear();

  // Process build options.
  std::string cppstr(options ? options : "");

  // if there is a -ignore-env,  adjust options.
  if (cppstr.size() > 0) {
    // Set the options to be the string after -ignore-env
    size_t pos = cppstr.find("-ignore-env");
    if (pos != std::string::npos) {
      cppstr = cppstr.substr(pos + sizeof("-ignore-env"));
      optionChangable = false;
    }
    remove_g_option(cppstr);
  }

  std::vector<const std::string*> headers(numHeaders);
  for (size_t i = 0; i < numHeaders; ++i) {
    const std::string& header = headerPrograms[i]->sourceCode();
    headers[i] = &header;
  }

  // Compile the program programs associated with the given devices.
  for (const auto& it : devices) {
    option::Options parsedOptions;
    constexpr bool LinkOptsOnly = false;
    if (!ParseAllOptions(cppstr, parsedOptions, optionChangable, LinkOptsOnly,
                         it->settings().useLightning_)) {
      programLog_ = parsedOptions.optionsLog();
      LogError("Parsing compile options failed.");
      return CL_INVALID_COMPILER_OPTIONS;
    }
    device::Program* devProgram = getDeviceProgram(*it);
    if (devProgram == NULL) {
      const binary_t& bin = binary(*it);
      retval = addDeviceProgram(*it, std::get<0>(bin), std::get<1>(bin), false, &parsedOptions);
      if (retval != CL_SUCCESS) {
        return retval;
      }
      devProgram = getDeviceProgram(*it);
    }

    if (devProgram->type() == device::Program::TYPE_INTERMEDIATE || language_ == SPIRV) {
      continue;
    }
    // We only build a Device-Program once
    if (devProgram->buildStatus() != CL_BUILD_NONE) {
      continue;
    }
    if (sourceCode_.empty()) {
      return CL_INVALID_OPERATION;
    }
    int32_t result =
        devProgram->compile(sourceCode_, headers, headerIncludeNames, options, &parsedOptions);

    // Check if the previous device failed a build
    if ((result != CL_SUCCESS) && (retval != CL_SUCCESS)) {
      retval = CL_INVALID_OPERATION;
    }
    // Update the returned value with a build error
    else if (result != CL_SUCCESS) {
      retval = result;
    }
  }

  if (notifyFptr != NULL) {
    notifyFptr(as_cl(this), data);
  }

  return retval;
}

int32_t Program::link(const std::vector<Device*>& devices, size_t numInputs,
                     const std::vector<Program*>& inputPrograms, const char* options,
                     void(CL_CALLBACK* notifyFptr)(cl_program, void*), void* data,
                     bool optionChangable) {
  ScopedLock sl(buildLock_);
  int32_t retval = CL_SUCCESS;

  if (symbolTable_ == NULL) {
    symbolTable_ = new symbols_t;
    if (symbolTable_ == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
  }

  // Clear the program object
  clear();

  // Process build options.
  std::string cppstr(options ? options : "");

  // if there is a -ignore-env,  adjust options.
  if (cppstr.size() > 0) {
    // Set the options to be the string after -ignore-env
    size_t pos = cppstr.find("-ignore-env");
    if (pos != std::string::npos) {
      cppstr = cppstr.substr(pos + sizeof("-ignore-env"));
      optionChangable = false;
    }
    remove_g_option(cppstr);
  }

  // Link the program programs associated with the given devices.
  for (const auto& it : devices) {
    option::Options parsedOptions;
    constexpr bool LinkOptsOnly = true;
    if (!ParseAllOptions(cppstr, parsedOptions, optionChangable, LinkOptsOnly,
                         it->settings().useLightning_)) {
      programLog_ = parsedOptions.optionsLog();
      LogError("Parsing link options failed.");
      return CL_INVALID_LINKER_OPTIONS;
    }
    // find the corresponding device program in each input program
    std::vector<device::Program*> inputDevPrograms(numInputs);
    bool found = false;
    for (size_t i = 0; i < numInputs; ++i) {
      Program& inputProgram = *inputPrograms[i];
      if (inputProgram.language_ == SPIRV) {
        parsedOptions.oVariables->BinaryIsSpirv = true;
      }
      deviceprograms_t inputDevProgs = inputProgram.devicePrograms();
      const auto findIt = inputDevProgs.find(it);
      if (findIt == inputDevProgs.cend()) {
        if (found) break;
        continue;
      }
      inputDevPrograms[i] = findIt->second;
      device::Program::binary_t binary = inputDevPrograms[i]->binary();
// Check the binary's target for the first found device program.
// TODO: Revise these binary's target checks
// and possibly remove them after switching to HSAIL by default.
#if defined(WITH_COMPILER_LIB)
      if (!found && binary.first != NULL && binary.second > 0 &&
          aclValidateBinaryImage(binary.first, binary.second, BINARY_TYPE_ELF)) {
        acl_error errorCode = ACL_SUCCESS;
        void* mem = const_cast<void*>(binary.first);
        aclBinary* aclBin = aclReadFromMem(mem, binary.second, &errorCode);
        if (errorCode != ACL_SUCCESS) {
          LogWarning("Error while linking: Could not read from raw binary.");
          return CL_INVALID_BINARY;
        }
        if (isHSAILTarget(*aclutGetTargetInfo(aclBin))) {
          parsedOptions.oVariables->Frontend = "clang";
          parsedOptions.oVariables->Legacy = it->settings().useLightning_;
        } else if (isAMDILTarget(*aclutGetTargetInfo(aclBin))) {
          parsedOptions.oVariables->Frontend = "edg";
        }
        aclBinaryFini(aclBin);
      }
#endif // defined(WITH_COMPILER_LIB)
      found = true;
    }
    if (inputDevPrograms.size() == 0) {
      continue;
    }
    if (inputDevPrograms.size() < numInputs) {
      return CL_INVALID_VALUE;
    }

    device::Program* devProgram = getDeviceProgram(*it);
    if (devProgram == NULL) {
      const binary_t& bin = binary(*it);
      retval = addDeviceProgram(*it, std::get<0>(bin), std::get<1>(bin), false, &parsedOptions);
      if (retval != CL_SUCCESS) {
        return retval;
      }
      devProgram = getDeviceProgram(*it);
    }

    // We only build a Device-Program once
    if (devProgram->buildStatus() != CL_BUILD_NONE) {
      continue;
    }
    int32_t result = devProgram->link(inputDevPrograms, options, &parsedOptions);

    // Check if the previous device failed a build
    if ((result != CL_SUCCESS) && (retval != CL_SUCCESS)) {
      retval = CL_INVALID_OPERATION;
    }
    // Update the returned value with a build error
    else if (result != CL_SUCCESS) {
      retval = result;
    }
  }

  if (retval != CL_SUCCESS) {
    return retval;
  }

  // Rebuild the symbol table
  for (const auto& sit : devicePrograms_) {
    const Device& device = *(sit.first);
    const device::Program& program = *(sit.second);

    const device::Program::kernels_t& kernels = program.kernels();
    for (const auto& it : kernels) {
      const std::string& name = it.first;
      const device::Kernel* devKernel = it.second;

      Symbol& symbol = (*symbolTable_)[name];
      if (!symbol.setDeviceKernel(device, devKernel)) {
        retval = CL_LINK_PROGRAM_FAILURE;
      }
    }
  }

  // Create a string with all kernel names from the program
  if (kernelNames_.length() == 0) {
    for (auto it = symbols().cbegin(); it != symbols().cend(); ++it) {
      if (it != symbols().cbegin()) {
        kernelNames_.append(1, ';');
      }
      kernelNames_.append(it->first.c_str());
    }
  }

  if (notifyFptr != NULL) {
    notifyFptr(as_cl(this), data);
  }

  return retval;
}

void Program::StubProgramSource(const std::string& app_name) {
  static uint program_counter = 0;
  std::fstream stub_read;
  std::stringstream file_name;
  std::string app_name_no_ext;

  std::size_t length = app_name.rfind(".exe");
  if (length == std::string::npos) {
    length = app_name.size();
  }
  app_name_no_ext.assign(app_name.c_str(), length);

  // Construct a unique file name for the CL program
  file_name << app_name_no_ext << "_program_" << program_counter << ".cl";

  stub_read.open(file_name.str().c_str(), (std::fstream::in | std::fstream::binary));
  // Check if we have OpenCL program
  if (stub_read.is_open()) {
    // Find the stream size
    stub_read.seekg(0, std::fstream::end);
    size_t size = stub_read.tellg();
    stub_read.seekg(0, std::ios::beg);

    char* data = new char[size];
    stub_read.read(data, size);
    stub_read.close();

    sourceCode_.assign(data, size);
    delete[] data;
  } else {
    std::fstream stub_write;
    stub_write.open(file_name.str().c_str(), (std::fstream::out | std::fstream::binary));
    stub_write << sourceCode_;
    stub_write.close();
  }
  program_counter++;
}

int32_t Program::build(const std::vector<Device*>& devices, const char* options,
                      void(CL_CALLBACK* notifyFptr)(cl_program, void*), void* data,
                      bool optionChangable) {
  ScopedLock sl(buildLock_);
  int32_t retval = CL_SUCCESS;

  if (symbolTable_ == NULL) {
    symbolTable_ = new symbols_t;
    if (symbolTable_ == NULL) {
      return CL_OUT_OF_HOST_MEMORY;
    }
  }

  if (OCL_STUB_PROGRAMS && !sourceCode_.empty()) {
    // The app name should be the samme for all device
    StubProgramSource(devices[0]->appProfile()->appFileName());
  }

  // Clear the program object
  clear();

  // Process build options.
  std::string cppstr(options ? options : "");

  // if there is a -ignore-env,  adjust options.
  if (cppstr.size() > 0) {
    // Set the options to be the string after -ignore-env
    size_t pos = cppstr.find("-ignore-env");
    if (pos != std::string::npos) {
      cppstr = cppstr.substr(pos + sizeof("-ignore-env"));
      optionChangable = false;
    }
    remove_g_option(cppstr);
  }

  // Build the program programs associated with the given devices.
  for (const auto& it : devices) {
    option::Options parsedOptions;
    constexpr bool LinkOptsOnly = false;
    if ((language_ != HIP) && !ParseAllOptions(cppstr, parsedOptions, optionChangable, LinkOptsOnly,
                         it->settings().useLightning_)) {
      programLog_ = parsedOptions.optionsLog();
      LogError("Parsing compile options failed.");
      return CL_INVALID_COMPILER_OPTIONS;
    }

    device::Program* devProgram = getDeviceProgram(*it);
    if (devProgram == NULL) {
      const binary_t& bin = binary(*it);
      if (sourceCode_.empty() && (std::get<0>(bin) == NULL)) {
        retval = false;
        continue;
      }
      retval = addDeviceProgram(*it, std::get<0>(bin), std::get<1>(bin), false, &parsedOptions);
      if (retval != CL_SUCCESS) {
        return retval;
      }
      devProgram = getDeviceProgram(*it);
    }

    parsedOptions.oVariables->AssumeAlias = true;

    if (language_ == Assembly) {
      constexpr char asmLang[] = "asm";
      parsedOptions.oVariables->XLang = asmLang;
    }

    if (language_ == HIP) {
      constexpr char hipLang[] = "HIP";
      parsedOptions.oVariables->CLStd = hipLang;
      parsedOptions.origOptionStr = options;
      parsedOptions.oVariables->DumpPrefix = "_hip_";
      parsedOptions.oVariables->OptLevel = '3';
    }

    // We only build a Device-Program once
    if (devProgram->buildStatus() != CL_BUILD_NONE) {
      continue;
    }
    int32_t result = devProgram->build(sourceCode_, options, &parsedOptions);

    // Check if the previous device failed a build
    if ((result != CL_SUCCESS) && (retval != CL_SUCCESS)) {
      retval = CL_INVALID_OPERATION;
    }
    // Update the returned value with a build error
    else if (result != CL_SUCCESS) {
      retval = result;
    }
  }

  if (retval != CL_SUCCESS) {
    return retval;
  }

  // Rebuild the symbol table
  for (const auto& it : devicePrograms_) {
    const Device& device = *(it.first);
    const device::Program& program = *(it.second);

    const device::Program::kernels_t& kernels = program.kernels();
    for (const auto& kit : kernels) {
      const std::string& name = kit.first;
      const device::Kernel* devKernel = kit.second;

      Symbol& symbol = (*symbolTable_)[name];
      if (!symbol.setDeviceKernel(device, devKernel)) {
        retval = CL_BUILD_PROGRAM_FAILURE;
      }
    }
  }

  // Create a string with all kernel names from the program
  if (kernelNames_.length() == 0) {
    for (auto it = symbols().cbegin(); it != symbols().cend(); ++it) {
      if (it != symbols().cbegin()) {
        kernelNames_.append(1, ';');
      }
      kernelNames_.append(it->first.c_str());
    }
  }

  if (notifyFptr != NULL) {
    notifyFptr(as_cl(this), data);
  }

  return retval;
}

void Program::clear() {
  // Destroy old programs if we have any
  for (const auto& it : devicePrograms_) {
    // Destroy device program
    delete it.second;
  }

  devicePrograms_.clear();
  deviceList_.clear();
  if (symbolTable_) symbolTable_->clear();
  kernelNames_.clear();
}

int Program::GetOclCVersion(const char* clVer) {
  // default version
  int version = 12;
  if (clVer == NULL) {
    return version;
  }
  std::string clStd(clVer);
  if (clStd.size() != 5) {
    return version;
  }
  clStd.erase(0, 2);
  clStd.erase(1, 1);
  return std::stoi(clStd);
}

bool Program::ParseAllOptions(const std::string& options, option::Options& parsedOptions,
                              bool optionChangable, bool linkOptsOnly, bool isLC) {
  std::string allOpts = options;
  if (optionChangable) {
    if (linkOptsOnly) {
      if (AMD_OCL_LINK_OPTIONS != NULL) {
        allOpts.append(" ");
        allOpts.append(AMD_OCL_LINK_OPTIONS);
      }
      if (AMD_OCL_LINK_OPTIONS_APPEND != NULL) {
        allOpts.append(" ");
        allOpts.append(AMD_OCL_LINK_OPTIONS_APPEND);
      }
    } else {
      if (AMD_OCL_BUILD_OPTIONS != NULL) {
        allOpts.append(" ");
        allOpts.append(AMD_OCL_BUILD_OPTIONS);
      }
      if (!Device::appProfile()->GetBuildOptsAppend().empty()) {
        allOpts.append(" ");
        allOpts.append(Device::appProfile()->GetBuildOptsAppend());
      }
      if (AMD_OCL_BUILD_OPTIONS_APPEND != NULL) {
        allOpts.append(" ");
        allOpts.append(AMD_OCL_BUILD_OPTIONS_APPEND);
      }
    }
  }
  return amd::option::parseAllOptions(allOpts, parsedOptions, linkOptsOnly, isLC);
}

bool Symbol::setDeviceKernel(const Device& device, const device::Kernel* func) {
  if (deviceKernels_.size() == 0 ||
      // Always pick the most recent version in MGPU case
      (func->signature().version() > signature_.version())) {
    signature_ = func->signature();
  }
  deviceKernels_[&device] = func;
  return true;
}

const device::Kernel* Symbol::getDeviceKernel(const Device& device) const {
  auto it = deviceKernels_.find(&device);
  if (it != deviceKernels_.cend()) {
    return it->second;
  }
  return nullptr;
}

}  // namespace amd
