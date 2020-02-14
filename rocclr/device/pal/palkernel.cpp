/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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

#include "device/pal/palkernel.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palblit.hpp"
#include "device/pal/palconstbuf.hpp"
#include "device/pal/palsched.hpp"
#include "platform/commandqueue.hpp"
#include "utils/options.hpp"
#include "acl.h"
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <algorithm>

namespace pal {

void HSAILKernel::setWorkGroupInfo(const uint32_t privateSegmentSize,
                                   const uint32_t groupSegmentSize, const uint16_t numSGPRs,
                                   const uint16_t numVGPRs) {
  workGroupInfo_.scratchRegs_ = amd::alignUp(privateSegmentSize, 16) / sizeof(uint32_t);
  // Make sure runtime matches HW alignment, which is 256 scratch regs (DWORDs) per wave
  constexpr uint32_t ScratchRegAlignment = 256;
  workGroupInfo_.scratchRegs_ =
      amd::alignUp((workGroupInfo_.scratchRegs_ * dev().info().wavefrontWidth_),
                   ScratchRegAlignment) / dev().info().wavefrontWidth_;
  workGroupInfo_.privateMemSize_ = workGroupInfo_.scratchRegs_ * sizeof(uint32_t);
  workGroupInfo_.localMemSize_ = workGroupInfo_.usedLDSSize_ = groupSegmentSize;
  workGroupInfo_.usedSGPRs_ = numSGPRs;
  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.usedVGPRs_ = numVGPRs;

  if (!prog().isNull()) {
    workGroupInfo_.availableLDSSize_ = dev().properties().gfxipProperties.shaderCore.ldsSizePerCu;
    workGroupInfo_.availableSGPRs_ =
        dev().properties().gfxipProperties.shaderCore.numAvailableSgprs;
    workGroupInfo_.availableVGPRs_ =
        dev().properties().gfxipProperties.shaderCore.numAvailableVgprs;
    workGroupInfo_.preferredSizeMultiple_ = workGroupInfo_.wavefrontPerSIMD_ =
        dev().info().wavefrontWidth_;
  } else {
    workGroupInfo_.availableLDSSize_ = 64 * Ki;
    workGroupInfo_.availableSGPRs_ = 104;
    workGroupInfo_.availableVGPRs_ = 256;
    workGroupInfo_.preferredSizeMultiple_ = workGroupInfo_.wavefrontPerSIMD_ = 64;
  }
}

bool HSAILKernel::setKernelCode(amd::hsa::loader::Symbol* sym, amd_kernel_code_t* akc) {
  if (!sym) {
    return false;
  }
  if (!sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, reinterpret_cast<void*>(&code_))) {
    return false;
  }

  // Copy code object of this kernel from the program CPU segment
  memcpy(akc, reinterpret_cast<void*>(prog().findHostKernelAddress(code_)),
         sizeof(amd_kernel_code_t));

  return true;
}

bool HSAILKernel::aqlCreateHWInfo(amd::hsa::loader::Symbol* sym) {
  amd_kernel_code_t* akc = &akc_;

  if (!setKernelCode(sym, akc)) {
    return false;
  }

  if (!sym->GetInfo(HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_SIZE,
                    reinterpret_cast<void*>(&codeSize_))) {
    return false;
  }

  // Setup the the workgroup info
  setWorkGroupInfo(akc->workitem_private_segment_byte_size, akc->workgroup_group_segment_byte_size,
                   akc->wavefront_sgpr_count, akc->workitem_vgpr_count);

  workgroupGroupSegmentByteSize_ = workGroupInfo_.usedLDSSize_;
  kernargSegmentByteSize_ = akc->kernarg_segment_byte_size;

  return true;
}

HSAILKernel::HSAILKernel(std::string name, HSAILProgram* prog, std::string compileOptions)
    : device::Kernel(prog->dev(), name, *prog),
      compileOptions_(compileOptions),
      index_(0),
      code_(0),
      codeSize_(0) {
  flags_.hsa_ = true;
}

HSAILKernel::~HSAILKernel() {}

bool HSAILKernel::init(amd::hsa::loader::Symbol* sym, bool finalize) {
#if defined(WITH_COMPILER_LIB)
  acl_error error = ACL_SUCCESS;
  std::string openClKernelName = openclMangledName(name());
  flags_.internalKernel_ =
      (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ? true : false;
  // compile kernel down to ISA
  if (finalize) {
    std::string options(compileOptions_.c_str());
    options.append(" -just-kernel=");
    options.append(openClKernelName.c_str());
    // Append an option so that we can selectively enable a SCOption on CZ
    // whenever IOMMUv2 is enabled.
    if (dev().settings().svmFineGrainSystem_) {
      options.append(" -sc-xnack-iommu");
    }
    error = aclCompile(dev().compiler(), prog().binaryElf(), options.c_str(), ACL_TYPE_CG,
                       ACL_TYPE_ISA, nullptr);
    buildLog_ += aclGetCompilerLog(dev().compiler());
    if (error != ACL_SUCCESS) {
      LogError("Failed to finalize kernel");
      return false;
    }
  }

  aqlCreateHWInfo(sym);

  // Pull out metadata from the ELF
  size_t sizeOfArgList;
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_ARGUMENT_ARRAY,
                       openClKernelName.c_str(), nullptr, &sizeOfArgList);
  if (error != ACL_SUCCESS) {
    return false;
  }

  char* aclArgList = new char[sizeOfArgList];
  if (nullptr == aclArgList) {
    return false;
  }
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_ARGUMENT_ARRAY,
                       openClKernelName.c_str(), aclArgList, &sizeOfArgList);
  if (error != ACL_SUCCESS) {
    return false;
  }
  // Set the argList
  InitParameters(reinterpret_cast<const aclArgData*>(aclArgList), argsBufferSize());
  delete[] aclArgList;

  size_t sizeOfWorkGroupSize;
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_WORK_GROUP_SIZE,
                       openClKernelName.c_str(), nullptr, &sizeOfWorkGroupSize);
  if (error != ACL_SUCCESS) {
    return false;
  }
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_WORK_GROUP_SIZE,
                       openClKernelName.c_str(), workGroupInfo_.compileSize_, &sizeOfWorkGroupSize);
  if (error != ACL_SUCCESS) {
    return false;
  }

  // Copy wavefront size
  workGroupInfo_.wavefrontSize_ = dev().info().wavefrontWidth_;
  // Find total workgroup size
  if (workGroupInfo_.compileSize_[0] != 0) {
    workGroupInfo_.size_ = workGroupInfo_.compileSize_[0] * workGroupInfo_.compileSize_[1] *
        workGroupInfo_.compileSize_[2];
  } else {
    workGroupInfo_.size_ = dev().info().preferredWorkGroupSize_;
  }

  // Pull out printf metadata from the ELF
  size_t sizeOfPrintfList;
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_GPU_PRINTF_ARRAY,
                       openClKernelName.c_str(), nullptr, &sizeOfPrintfList);
  if (error != ACL_SUCCESS) {
    return false;
  }

  // Make sure kernel has any printf info
  if (0 != sizeOfPrintfList) {
    char* aclPrintfList = new char[sizeOfPrintfList];
    if (nullptr == aclPrintfList) {
      return false;
    }
    error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_GPU_PRINTF_ARRAY,
                         openClKernelName.c_str(), aclPrintfList, &sizeOfPrintfList);
    if (error != ACL_SUCCESS) {
      return false;
    }

    // Set the PrintfList
    InitPrintf(reinterpret_cast<aclPrintfFmt*>(aclPrintfList));
    delete[] aclPrintfList;
  }

  aclMetadata md;
  md.enqueue_kernel = false;
  size_t sizeOfDeviceEnqueue = sizeof(md.enqueue_kernel);
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_DEVICE_ENQUEUE,
                       openClKernelName.c_str(), &md.enqueue_kernel, &sizeOfDeviceEnqueue);
  if (error != ACL_SUCCESS) {
    return false;
  }
  flags_.dynamicParallelism_ = md.enqueue_kernel;

  md.kernel_index = -1;
  size_t sizeOfIndex = sizeof(md.kernel_index);
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_KERNEL_INDEX,
                       openClKernelName.c_str(), &md.kernel_index, &sizeOfIndex);
  if (error != ACL_SUCCESS) {
    return false;
  }
  index_ = md.kernel_index;

  size_t sizeOfWavesPerSimdHint = sizeof(workGroupInfo_.wavesPerSimdHint_);
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_WAVES_PER_SIMD_HINT,
                       openClKernelName.c_str(), &workGroupInfo_.wavesPerSimdHint_,
                       &sizeOfWavesPerSimdHint);
  if (error != ACL_SUCCESS) {
    return false;
  }

  waveLimiter_.enable();

  size_t sizeOfWorkGroupSizeHint = sizeof(workGroupInfo_.compileSizeHint_);
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_WORK_GROUP_SIZE_HINT,
                       openClKernelName.c_str(), workGroupInfo_.compileSizeHint_,
                       &sizeOfWorkGroupSizeHint);
  if (error != ACL_SUCCESS) {
    return false;
  }

  size_t sizeOfVecTypeHint;
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_VEC_TYPE_HINT,
                       openClKernelName.c_str(), NULL, &sizeOfVecTypeHint);
  if (error != ACL_SUCCESS) {
    return false;
  }

  if (0 != sizeOfVecTypeHint) {
    char* VecTypeHint = new char[sizeOfVecTypeHint + 1];
    if (NULL == VecTypeHint) {
      return false;
    }
    error = aclQueryInfo(dev().compiler(), prog().binaryElf(), RT_VEC_TYPE_HINT,
                         openClKernelName.c_str(), VecTypeHint, &sizeOfVecTypeHint);
    if (error != ACL_SUCCESS) {
      return false;
    }
    VecTypeHint[sizeOfVecTypeHint] = '\0';
    workGroupInfo_.compileVecTypeHint_ = std::string(VecTypeHint);
    delete[] VecTypeHint;
  }

#endif  // defined(WITH_COMPILER_LIB)
  return true;
}

const Device& HSAILKernel::dev() const { return reinterpret_cast<const Device&>(dev_); }

const HSAILProgram& HSAILKernel::prog() const {
  return reinterpret_cast<const HSAILProgram&>(prog_);
}

hsa_kernel_dispatch_packet_t* HSAILKernel::loadArguments(VirtualGPU& gpu, const amd::Kernel& kernel,
                                                         const amd::NDRangeContainer& sizes,
                                                         const_address params,
                                                         size_t ldsAddress, uint64_t vmDefQueue,
                                                         uint64_t* vmParentWrap) const {
  const_address parameters = params;
  uint64_t argList;
  address aqlArgBuf = gpu.managedBuffer().reserve(
      argsBufferSize() + sizeof(hsa_kernel_dispatch_packet_t), &argList);
  gpu.addVmMemory(gpu.managedBuffer().activeMemory());

  if (dynamicParallelism()) {
    // Provide the host parent AQL wrap object to the kernel
    AmdAqlWrap wrap = {};
    wrap.state = AQL_WRAP_BUSY;
    *vmParentWrap = gpu.cb(1)->UploadDataToHw(&wrap, sizeof(AmdAqlWrap));
    gpu.addVmMemory(gpu.cb(1)->ActiveMemory());
  }

  // The check below handles a special case of single context with multiple devices
  // when the devices use different compilers(HSAIL and LC) and have different signatures
  const amd::KernelSignature& signature =
    (this->signature().version() == kernel.signature().version()) ?
    kernel.signature() : this->signature();

  // If signatures don't match, then patch the parameters
  if (signature.version() != kernel.signature().version()) {
    WriteAqlArgAt(aqlArgBuf, parameters, signature.paramsSize() - signature.at(0).offset_,
                  signature.at(0).offset_);
    parameters = aqlArgBuf;
  }

  // Check if runtime has to setup hidden arguments
  for (uint32_t i = signature.numParameters(); i < signature.numParametersAll(); ++i) {
    const auto it = signature.at(i);
    size_t offset;
    switch (it.info_.oclObject_) {
      case amd::KernelParameterDescriptor::HiddenNone:
        // void* zero = 0;
        // WriteAqlArgAt(const_cast<address>(parameters), &zero, it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetX:
        offset = sizes.offset()[0];
        WriteAqlArgAt(const_cast<address>(parameters), &offset, it.size_, it.offset_);
        break;
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetY:
        if (sizes.dimensions() >= 2) {
          offset = sizes.offset()[1];
          WriteAqlArgAt(const_cast<address>(parameters), &offset, it.size_, it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenGlobalOffsetZ:
        if (sizes.dimensions() >= 3) {
          offset = sizes.offset()[2];
          WriteAqlArgAt(const_cast<address>(parameters), &offset, it.size_, it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenPrintfBuffer:
        if ((printfInfo().size() > 0) &&
            // and printf buffer was allocated
            (gpu.printfDbgHSA().dbgBuffer() != nullptr)) {
          // and set the fourth argument as the printf_buffer pointer
          size_t bufferPtr = static_cast<size_t>(gpu.printfDbgHSA().dbgBuffer()->vmAddress());
          gpu.addVmMemory(gpu.printfDbgHSA().dbgBuffer());
          WriteAqlArgAt(const_cast<address>(parameters), &bufferPtr, it.size_, it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenDefaultQueue:
        if (vmDefQueue != 0) {
          WriteAqlArgAt(const_cast<address>(parameters), &vmDefQueue, it.size_, it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenCompletionAction:
        if (*vmParentWrap != 0) {
          WriteAqlArgAt(const_cast<address>(parameters), vmParentWrap, it.size_, it.offset_);
        }
        break;
      case amd::KernelParameterDescriptor::HiddenMultiGridSync:
        break;
    }
  }

  // Load all kernel arguments
  if (signature.version() == kernel.signature().version()) {
    WriteAqlArgAt(aqlArgBuf, parameters, argsBufferSize(), 0);
  }

  // Note: In a case of structs the size won't match,
  // since HSAIL compiler expects a reference...
  assert(argsBufferSize() <= signature.paramsSize() &&
         "A mismatch of sizes of arguments between compiler and runtime!");

  // hsa_kernel_dispatch_packet_t disp;
  hsa_kernel_dispatch_packet_t* hsaDisp =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(gpu.cb(0)->SysMemCopy());

  amd::NDRange local(sizes.local());
  const amd::NDRange& global = sizes.global();

  // Check if runtime has to find local workgroup size
  FindLocalWorkSize(sizes.dimensions(), sizes.global(), local);

  constexpr uint16_t kDispatchPacketHeader =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

  hsaDisp->header = kDispatchPacketHeader;
  hsaDisp->setup = sizes.dimensions();

  hsaDisp->workgroup_size_x = local[0];
  hsaDisp->workgroup_size_y = (sizes.dimensions() > 1) ? local[1] : 1;
  hsaDisp->workgroup_size_z = (sizes.dimensions() > 2) ? local[2] : 1;

  hsaDisp->grid_size_x = global[0];
  hsaDisp->grid_size_y = (sizes.dimensions() > 1) ? global[1] : 1;
  hsaDisp->grid_size_z = (sizes.dimensions() > 2) ? global[2] : 1;
  hsaDisp->reserved2 = 0;

  // Initialize kernel ISA and execution buffer requirements
  hsaDisp->private_segment_size = spillSegSize();
  hsaDisp->group_segment_size = ldsAddress;
  hsaDisp->kernel_object = gpuAqlCode();

  hsaDisp->kernarg_address = reinterpret_cast<void*>(argList);
  hsaDisp->reserved2 = 0;
  hsaDisp->completion_signal.handle = 0;
  memcpy(aqlArgBuf + argsBufferSize(), hsaDisp, sizeof(hsa_kernel_dispatch_packet_t));

  if (AMD_HSA_BITS_GET(akc_.kernel_code_properties,
                       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)) {
    gpu.addVmMemory(gpu.hsaQueueMem());
  }

  return hsaDisp;
}

const LightningProgram& LightningKernel::prog() const {
  return reinterpret_cast<const LightningProgram&>(prog_);
}

#if defined(USE_COMGR_LIBRARY)
bool LightningKernel::init() {
  flags_.internalKernel_ =
      (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ? true : false;

  if (!GetAttrCodePropMetadata()) {
    return false;
  }

  if (codeObjectVer() == 2) {
    symbolName_ =  name();
  }

  // Copy codeobject of this kernel from the program CPU segment
  hsa_agent_t agent;
  agent.handle = 1;

  auto sym = prog().GetSymbol(symbolName().c_str(), const_cast<hsa_agent_t*>(&agent));

  if (!setKernelCode(sym, &akc_)) {
    return false;
  }

  if (!prog().isNull()) {
    codeSize_ = prog().codeSegGpu().owner()->getSize();

    // handle device enqueue
    if (!RuntimeHandle().empty()) {
      hsa_agent_t agent;
      agent.handle = 1;
      amd::hsa::loader::Symbol* rth_symbol;

      // Get the runtime handle symbol GPU address
      rth_symbol = prog().GetSymbol(const_cast<char*>(RuntimeHandle().c_str()),
                                    const_cast<hsa_agent_t*>(&agent));
      uint64_t symbol_address;
      rth_symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &symbol_address);

      // Copy the kernel_object pointer to the runtime handle symbol GPU address
      const Memory& codeSegGpu = prog().codeSegGpu();
      uint64_t offset = symbol_address - codeSegGpu.vmAddress();
      uint64_t kernel_object = gpuAqlCode();
      VirtualGPU* gpu = codeSegGpu.dev().xferQueue();

      const struct RuntimeHandle runtime_handle = {gpuAqlCode(), spillSegSize(), ldsSize()};

      codeSegGpu.writeRawData(*gpu, offset, sizeof(runtime_handle), &runtime_handle, true);
    }
  }

  // Setup the the workgroup info
  setWorkGroupInfo(WorkitemPrivateSegmentByteSize(), WorkgroupGroupSegmentByteSize(),
                   workGroupInfo()->usedSGPRs_, workGroupInfo()->usedVGPRs_);

  // Copy wavefront size
  workGroupInfo_.wavefrontSize_ = dev().info().wavefrontWidth_;

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

}  // namespace pal
