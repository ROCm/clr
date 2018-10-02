//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef WITHOUT_HSA_BACKEND

#include "rocprogram.hpp"

#include "utils/options.hpp"
#include "rockernel.hpp"
#if defined(WITH_LIGHTNING_COMPILER)
#include <gelf.h>
#include "driver/AmdCompiler.h"
#include "libraries.amdgcn.inc"
#endif  // defined(WITH_LIGHTNING_COMPILER)

#include "utils/bif_section_labels.hpp"
#include "amd_hsa_kernel_code.h"

#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>

namespace roc {

static hsa_status_t GetKernelNamesCallback(hsa_executable_t exec, hsa_agent_t agent,
                                           hsa_executable_symbol_t symbol, void* data) {
  std::vector<std::string>* symNameList = reinterpret_cast<std::vector<std::string>*>(data);

  hsa_symbol_kind_t sym_type;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &sym_type);

  if (sym_type == HSA_SYMBOL_KIND_KERNEL) {
    uint32_t len;
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);

    char* symName = (char*)alloca(len + 1);
    hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, symName);
    symName[len] = '\0';

    std::string kernelName(symName);
    symNameList->push_back(kernelName);
  }

  return HSA_STATUS_SUCCESS;
}

/* Temporary log function for the compiler library */
static void logFunction(const char* msg, size_t size) {
  std::cout << "Compiler Library log :" << msg << std::endl;
}

static inline const char* hsa_strerror(hsa_status_t status) {
  const char* str = nullptr;
  if (hsa_status_string(status, &str) == HSA_STATUS_SUCCESS) {
    return str;
  }
  return "Unknown error";
}

Program::~Program() {
  // Destroy the executable.
  if (hsaExecutable_.handle != 0) {
    hsa_executable_destroy(hsaExecutable_);
  }
  if (hsaCodeObjectReader_.handle != 0) {
    hsa_code_object_reader_destroy(hsaCodeObjectReader_);
  }
  releaseClBinary();
}

Program::Program(roc::NullDevice& device) : device::Program(device) {
  hsaExecutable_.handle = 0;
  hsaCodeObjectReader_.handle = 0;
}

bool Program::initClBinary(char* binaryIn, size_t size) {
  // Save the original binary that isn't owned by ClBinary
  clBinary()->saveOrigBinary(binaryIn, size);

  char* bin = binaryIn;
  size_t sz = size;

  int encryptCode;

  char* decryptedBin;
  size_t decryptedSize;
  if (!clBinary()->decryptElf(binaryIn, size, &decryptedBin, &decryptedSize, &encryptCode)) {
    return false;
  }
  if (decryptedBin != nullptr) {
    // It is decrypted binary.
    bin = decryptedBin;
    sz = decryptedSize;
  }

  // Both 32-bit and 64-bit are allowed!
  if (!amd::isElfMagic(bin)) {
    // Invalid binary.
    if (decryptedBin != nullptr) {
      delete[] decryptedBin;
    }
    return false;
  }

  clBinary()->setFlags(encryptCode);

  return clBinary()->setBinary(bin, sz, (decryptedBin != nullptr));
}

#if defined(WITH_COMPILER_LIB)
HSAILProgram::HSAILProgram(roc::NullDevice& device) : roc::Program(device) {
  xnackEnabled_ = dev().deviceInfo().xnackEnabled_;
  machineTarget_ = dev().deviceInfo().complibTarget_;
}

HSAILProgram::~HSAILProgram() {
  acl_error error;
  // Free the elf binary
  if (binaryElf_ != nullptr) {
    error = aclBinaryFini(binaryElf_);
    if (error != ACL_SUCCESS) {
      LogWarning("Error while destroying the acl binary \n");
    }
  }
}

bool HSAILProgram::createBinary(amd::option::Options* options) {
  return true;
}

bool HSAILProgram::saveBinaryAndSetType(type_t type) {
  void* rawBinary;
  size_t size;

  // Write binary to memory
  if (aclWriteToMem(binaryElf_, &rawBinary, &size) != ACL_SUCCESS) {
    buildLog_ += "Failed to write binary to memory \n";
    return false;
  }
  clBinary()->saveBIFBinary((char*)rawBinary, size);
  // Set the type of binary
  setType(type);

// Free memory containing rawBinary
  binaryElf_->binOpts.dealloc(rawBinary);
  return true;
}

bool HSAILProgram::setKernels(amd::option::Options* options, void* binary, size_t binSize) {
  // Stop compilation if it is an offline device - HSA runtime does not
  // support ISA compiled offline
  if (!dev().isOnline()) {
    return true;
  }

  size_t secSize = binSize;
  void* data = binary;

  // Create an executable.
  hsa_status_t status = hsa_executable_create_alt(
      HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &hsaExecutable_);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Failed to create executable: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Load the code object.
  hsa_code_object_reader_t codeObjectReader;
  status = hsa_code_object_reader_create_from_memory(data, secSize, &codeObjectReader);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object Reader create failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  hsa_agent_t hsaDevice = dev().getBackendDevice();
  status = hsa_executable_load_agent_code_object(hsaExecutable_, hsaDevice, codeObjectReader,
                                                 nullptr, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object loading failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  hsa_code_object_reader_destroy(codeObjectReader);

  // Freeze the executable.
  status = hsa_executable_freeze(hsaExecutable_, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Failed to freeze executable: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Get the list of kernels
  std::vector<std::string> kernelNameList;
  status = hsa_executable_iterate_agent_symbols(hsaExecutable_, hsaDevice, GetKernelNamesCallback,
                                                (void*)&kernelNameList);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Failed to get kernel names: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  for (auto& kernelName : kernelNameList) {
    // Query symbol handle for this symbol.
    hsa_executable_symbol_t kernelSymbol;
    status = hsa_executable_get_symbol_by_name(hsaExecutable_, kernelName.c_str(), &hsaDevice,
                                               &kernelSymbol);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get executable symbol: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    // Query code handle for this symbol.
    uint64_t kernelCodeHandle;
    status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                            &kernelCodeHandle);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get executable symbol info: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    std::string openclKernelName = kernelName;
    // Strip the opencl and kernel name
    kernelName = kernelName.substr(strlen("&__OpenCL_"), kernelName.size());
    kernelName = kernelName.substr(0, kernelName.size() - strlen("_kernel"));
    aclMetadata md;
    md.numHiddenKernelArgs = 0;

    size_t sizeOfnumHiddenKernelArgs = sizeof(md.numHiddenKernelArgs);
    acl_error errorCode = aclQueryInfo(device().compiler(), binaryElf_,
                             RT_NUM_KERNEL_HIDDEN_ARGS, openclKernelName.c_str(),
                             &md.numHiddenKernelArgs, &sizeOfnumHiddenKernelArgs);
    if (errorCode != ACL_SUCCESS) {
      buildLog_ +=
          "Error while Finalization phase: Kernel extra arguments count querying from the ELF "
          "failed\n";
      return false;
    }

    uint32_t workgroupGroupSegmentByteSize;
    status = hsa_executable_symbol_get_info(kernelSymbol,
                                            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                            &workgroupGroupSegmentByteSize);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get group segment size info: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    uint32_t workitemPrivateSegmentByteSize;
    status = hsa_executable_symbol_get_info(kernelSymbol,
                                            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                            &workitemPrivateSegmentByteSize);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get private segment size info: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    uint32_t kernargSegmentByteSize;
    status = hsa_executable_symbol_get_info(kernelSymbol,
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
        kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
        &kernargSegmentAlignment);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get kernarg segment alignment info: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    Kernel* aKernel = new roc::HSAILKernel(kernelName, this, kernelCodeHandle,
                                           workgroupGroupSegmentByteSize,
                                           workitemPrivateSegmentByteSize,
                                           kernargSegmentByteSize, kernargSegmentAlignment);
    if (!aKernel->init()) {
      return false;
    }
    aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
    aKernel->setInternalKernelFlag(compileOptions_.find("-cl-internal-kernel") !=
                                   std::string::npos);
    kernels()[kernelName] = aKernel;
  }
  return true;
}
#endif // defined(WITH_COMPILER_LIB)

#if defined(WITH_LIGHTNING_COMPILER)
LightningProgram::LightningProgram(roc::NullDevice& device)
  : roc::Program(device) {
  isLC_ = true;
  xnackEnabled_ = dev().deviceInfo().xnackEnabled_;
  machineTarget_ = dev().deviceInfo().machineTarget_;
}

bool LightningProgram::createBinary(amd::option::Options* options) {
  if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt, type())) {
    LogError("Failed to create ELF binary image!");
    return false;
  }
  return true;
}

bool LightningProgram::saveBinaryAndSetType(type_t type, void* rawBinary, size_t size) {
  // Write binary to memory
  if (type == TYPE_EXECUTABLE) {  // handle code object binary
    assert(rawBinary != nullptr && size != 0 && "must pass in the binary");
  }
  else {  // handle LLVM binary
    if (llvmBinary_.empty()) {
      buildLog_ += "ERROR: Tried to save emtpy LLVM binary \n";
      return false;
    }
    rawBinary = (void*)llvmBinary_.data();
    size = llvmBinary_.size();
  }
  clBinary()->saveBIFBinary((char*)rawBinary, size);

  // Set the type of binary
  setType(type);
  return true;
}

bool LightningProgram::setKernels(amd::option::Options* options, void* binary, size_t binSize) {
  // Find the size of global variables from the binary
  if (!FindGlobalVarSize(binary, binSize)) {
    return false;
  }

  hsa_agent_t agent = dev().getBackendDevice();
  hsa_status_t status;

  status = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     nullptr, &hsaExecutable_);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Executable for AMD HSA Code Object isn't created: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Load the code object.
  status = hsa_code_object_reader_create_from_memory(binary, binSize, &hsaCodeObjectReader_);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object Reader create failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  status = hsa_executable_load_agent_code_object(hsaExecutable_, agent, hsaCodeObjectReader_, nullptr,
                                                 nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object loading failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Freeze the executable.
  status = hsa_executable_freeze(hsaExecutable_, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Freezing the executable failed: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  // Get the list of kernels
  std::vector<std::string> kernelNameList;
  status = hsa_executable_iterate_agent_symbols(hsaExecutable_, agent, GetKernelNamesCallback,
                                                (void*)&kernelNameList);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Failed to get kernel names: ";
    buildLog_ += hsa_strerror(status);
    buildLog_ += "\n";
    return false;
  }

  for (auto& kernelName : kernelNameList) {
    hsa_executable_symbol_t kernelSymbol;

    status = hsa_executable_get_symbol_by_name(hsaExecutable_, kernelName.c_str(), &agent,
                                               &kernelSymbol);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get the symbol: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    uint64_t kernelCodeHandle;
    status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                            &kernelCodeHandle);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get the kernel code: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    uint32_t workgroupGroupSegmentByteSize;
    status = hsa_executable_symbol_get_info(kernelSymbol,
                                            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                            &workgroupGroupSegmentByteSize);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get group segment size info: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    uint32_t workitemPrivateSegmentByteSize;
    status = hsa_executable_symbol_get_info(kernelSymbol,
                                            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                            &workitemPrivateSegmentByteSize);
    if (status != HSA_STATUS_SUCCESS) {
      buildLog_ += "Error: Failed to get private segment size info: ";
      buildLog_ += hsa_strerror(status);
      buildLog_ += "\n";
      return false;
    }

    uint32_t kernargSegmentByteSize;
    status = hsa_executable_symbol_get_info(kernelSymbol,
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
        kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
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

    Kernel* aKernel = new roc::LightningKernel(
        kernelName, this, kernelCodeHandle, workgroupGroupSegmentByteSize,
        workitemPrivateSegmentByteSize, kernargSegmentByteSize,
        amd::alignUp(kernargSegmentAlignment, device().info().globalMemCacheLineSize_));
    if (!aKernel->init()) {
      return false;
    }
    aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
    aKernel->setInternalKernelFlag(compileOptions_.find("-cl-internal-kernel") !=
                                   std::string::npos);
    kernels()[kernelName] = aKernel;
  }

  return true;
}
#endif  // defined(WITH_LIGHTNING_COMPILER)

}  // namespace roc

#endif  // WITHOUT_HSA_BACKEND
