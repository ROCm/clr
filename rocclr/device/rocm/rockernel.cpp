/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#include "rockernel.hpp"
#include "amd_hsa_kernel_code.h"

#include <algorithm>

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

Kernel::Kernel(std::string name, Program* prog, const uint64_t& kernelCodeHandle,
               const uint32_t workgroupGroupSegmentByteSize,
               const uint32_t workitemPrivateSegmentByteSize, const uint32_t kernargSegmentByteSize,
               const uint32_t kernargSegmentAlignment)
    : device::Kernel(prog->device(), name, *prog) {
  kernelCodeHandle_ = kernelCodeHandle;
  workgroupGroupSegmentByteSize_ = workgroupGroupSegmentByteSize;
  workitemPrivateSegmentByteSize_ = workitemPrivateSegmentByteSize;
  kernargSegmentByteSize_ = kernargSegmentByteSize;
  kernargSegmentAlignment_ = kernargSegmentAlignment;
}

Kernel::Kernel(std::string name, Program* prog)
    : device::Kernel(prog->device(), name, *prog) {
}

#if defined(USE_COMGR_LIBRARY)
bool LightningKernel::init() {
  if (!GetAttrCodePropMetadata()) {
    LogError("[ROC][Kernel] Could not get Code Prop Meta Data \n");
    return false;
  }

  // Set the kernel symbol name and size/alignment based on the kernel metadata
  // NOTE: kernel name is used to get the kernel code handle in V2,
  //       but kernel symbol name is used in V3
  if (codeObjectVer() == 2) {
    symbolName_ = name();
  }
  kernargSegmentAlignment_ =
      amd::alignUp(std::max(kernargSegmentAlignment_, 128u), device().info().globalMemCacheLineSize_);

  // Set the workgroup information for the kernel
  workGroupInfo_.availableLDSSize_ = device().info().localMemSizePerCU_;
  assert(workGroupInfo_.availableLDSSize_ > 0);

  if (!SetAvailableSgprVgpr()) {
    DevLogError("Cannot set available SGPR/VGPR\n");
    return false;
  }

  // Get the kernel code handle
  hsa_status_t hsaStatus;
  hsa_executable_symbol_t symbol;
  hsa_agent_t agent = program()->rocDevice().getBackendDevice();
  hsaStatus = hsa_executable_get_symbol_by_name(program()->hsaExecutable(),
                                                symbolName().c_str(),
                                                &agent, &symbol);
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    DevLogPrintfError("Cannot Get Symbol : %s, failed with hsa_status: %d \n",
                      symbolName().c_str(), hsaStatus);
    return false;
  }

  hsaStatus = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                             &kernelCodeHandle_);
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    DevLogPrintfError(" Cannot Get Symbol Info: %s, failed with hsa_status: %d \n ",
                      symbolName().c_str(), hsaStatus);
    return false;
  }

  if (!RuntimeHandle().empty()) {
    hsa_executable_symbol_t kernelSymbol;
    int                     variable_size;
    uint64_t                variable_address;

    // Only kernels that could be enqueued by another kernel has the RuntimeHandle metadata. The RuntimeHandle
    // metadata is a string that represents a variable from which the library code can retrieve the kernel code
    // object handle of such a kernel. The address of the variable and the kernel code object handle are known
    // only after the hsa executable is loaded. The below code copies the kernel code object handle to the
    // address of the variable.
    hsaStatus = hsa_executable_get_symbol_by_name(program()->hsaExecutable(),
                                                  RuntimeHandle().c_str(),
                                                  &agent, &kernelSymbol);
    if (hsaStatus != HSA_STATUS_SUCCESS) {
      DevLogPrintfError("Cannot get Kernel Symbol by name: %s, failed with hsa_status: %d \n",
                        RuntimeHandle().c_str(), hsaStatus);
      return false;
    }

    hsaStatus = hsa_executable_symbol_get_info(kernelSymbol,
                                               HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE,
                                               &variable_size);
    if (hsaStatus != HSA_STATUS_SUCCESS) {
      DevLogPrintfError("[ROC][Kernel] Cannot get Kernel Symbol Info, failed with hsa_status: %d \n",
                        hsaStatus);
      return false;
    }

    hsaStatus = hsa_executable_symbol_get_info(kernelSymbol,
                                               HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                                               &variable_address);
    if (hsaStatus != HSA_STATUS_SUCCESS) {
      DevLogPrintfError("[ROC][Kernel] Cannot get Kernel Address, failed with hsa_status: %d \n",
                        hsaStatus);
      return false;
    }

    const struct RuntimeHandle runtime_handle = {
      kernelCodeHandle_,
      WorkitemPrivateSegmentByteSize(),
      WorkgroupGroupSegmentByteSize()
    };
    hsaStatus = hsa_memory_copy(reinterpret_cast<void*>(variable_address),
                                &runtime_handle, variable_size);

    if (hsaStatus != HSA_STATUS_SUCCESS) {
      DevLogPrintfError("[ROC][Kernel] HSA Memory copy failed, failed with hsa_status: %d \n",
                        hsaStatus);
      return false;
    }
  }

  uint32_t wavefront_size = 0;
  if (hsa_agent_get_info(program()->rocDevice().getBackendDevice(), HSA_AGENT_INFO_WAVEFRONT_SIZE,
                         &wavefront_size) != HSA_STATUS_SUCCESS) {
    DevLogPrintfError("[ROC][Kernel] Cannot get Wavefront Size, failed with hsa_status: %d \n",
                      hsaStatus);
    return false;
  }
  assert(wavefront_size > 0);

  workGroupInfo_.privateMemSize_ = workitemPrivateSegmentByteSize_;
  workGroupInfo_.localMemSize_ = workgroupGroupSegmentByteSize_;
  workGroupInfo_.usedLDSSize_ = workgroupGroupSegmentByteSize_;
  workGroupInfo_.preferredSizeMultiple_ = wavefront_size;
  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.wavefrontPerSIMD_ = program()->rocDevice().info().maxWorkItemSizes_[0] / wavefront_size;
  workGroupInfo_.wavefrontSize_ = wavefront_size;
  if (workGroupInfo_.size_ == 0) {
    return false;
  }

  // handle the printf metadata if any
  std::vector<std::string> printfStr;
  if (!GetPrintfStr(&printfStr)) {
    return false;
  }

  if (!printfStr.empty()) {
    InitPrintf(printfStr);
  }
  return true;
}
#endif  // defined(USE_COMGR_LIBRARY)

}  // namespace roc
#endif  // WITHOUT_HSA_BACKEND
