//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//

#include "rockernel.hpp"
#include "amd_hsa_kernel_code.h"

#include <algorithm>

#ifndef WITHOUT_HSA_BACKEND

#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
#ifndef USE_COMGR_LIBRARY
#include "driver/AmdCompiler.h"
#endif
#include "llvm/Support/AMDGPUMetadata.h"

typedef llvm::AMDGPU::HSAMD::Metadata CodeObjectMD;
typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
#endif  // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

namespace roc {

Kernel::Kernel(std::string name, Program* prog, const uint64_t& kernelCodeHandle,
               const uint32_t workgroupGroupSegmentByteSize,
               const uint32_t workitemPrivateSegmentByteSize, const uint32_t kernargSegmentByteSize,
               const uint32_t kernargSegmentAlignment)
    : device::Kernel(prog->dev(), name, *prog),
      kernelCodeHandle_(kernelCodeHandle),
      workgroupGroupSegmentByteSize_(workgroupGroupSegmentByteSize),
      workitemPrivateSegmentByteSize_(workitemPrivateSegmentByteSize),
      kernargSegmentByteSize_(kernargSegmentByteSize),
      kernargSegmentAlignment_(kernargSegmentAlignment) {}

Kernel::Kernel(std::string name, Program* prog)
    : device::Kernel(prog->dev(), name, *prog),
      kernelCodeHandle_(0),
      workgroupGroupSegmentByteSize_(0),
      workitemPrivateSegmentByteSize_(0),
      kernargSegmentByteSize_(0),
      kernargSegmentAlignment_(0) {}

#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
#if defined(USE_COMGR_LIBRARY)
bool LightningKernel::init() {

  hsa_agent_t hsaDevice = program()->hsaDevice();

  const amd_comgr_metadata_node_t* kernelMetaNode =
              static_cast<const LightningProgram*>(program())->getKernelMetadata(name());
  if (kernelMetaNode == nullptr) {
    return false;
  }

  KernelMD  kernelMD;
  if (!GetAttrCodePropMetadata(*kernelMetaNode, &kernelMD)) {
    return false;
  }

  // Set the kernel symbol name and size/alignment based on the kernel metadata
  // NOTE: kernel name is used to get the kernel code handle in V2,
  //       but kernel symbol name is used in V3
  symbolName_ = (codeObjectVer() == 2) ? name() : kernelMD.mSymbolName;
  workgroupGroupSegmentByteSize_ = kernelMD.mCodeProps.mGroupSegmentFixedSize;
  workitemPrivateSegmentByteSize_ = kernelMD.mCodeProps.mPrivateSegmentFixedSize;
  kernargSegmentByteSize_ = kernelMD.mCodeProps.mKernargSegmentSize;
  kernargSegmentAlignment_ = amd::alignUp(std::max(kernelMD.mCodeProps.mKernargSegmentAlign, 128u),
                                          dev().info().globalMemCacheLineSize_);

  // Set the workgroup information for the kernel
  workGroupInfo_.availableLDSSize_ = dev().info().localMemSizePerCU_;
  assert(workGroupInfo_.availableLDSSize_ > 0);

  // Get the available SGPRs and VGPRs
  std::string targetIdent = std::string("amdgcn-amd-amdhsa--")+program()->machineTarget();
  if (program()->xnackEnable()) {
    targetIdent.append("+xnack");
  }
  if (program()->sramEccEnable()) {
    targetIdent.append("+sram-ecc");
  }

  if (!SetAvailableSgprVgpr(targetIdent)) {
    return false;
  }

  // Get the kernel code handle
  hsa_status_t hsaStatus;
  hsa_executable_symbol_t symbol;
  hsa_agent_t agent = program()->hsaDevice();
  hsaStatus = hsa_executable_get_symbol_by_name(program()->hsaExecutable(),
                                                symbolName().c_str(),
                                                &agent, &symbol);
  if (hsaStatus == HSA_STATUS_SUCCESS) {
    hsaStatus = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                               &kernelCodeHandle_);
  }
  if (hsaStatus != HSA_STATUS_SUCCESS) {
    return false;
  }

  if (!kernelMD.mAttrs.mRuntimeHandle.empty()) {
    hsa_executable_symbol_t kernelSymbol;
    int                     variable_size;
    uint64_t                variable_address;

    // Only kernels that could be enqueued by another kernel has the RuntimeHandle metadata. The RuntimeHandle
    // metadata is a string that represents a variable from which the library code can retrieve the kernel code
    // object handle of such a kernel. The address of the variable and the kernel code object handle are known
    // only after the hsa executable is loaded. The below code copies the kernel code object handle to the
    // address of the variable.
    hsaStatus = hsa_executable_get_symbol_by_name(program()->hsaExecutable(),
                                                  kernelMD.mAttrs.mRuntimeHandle.c_str(),
                                                  &agent, &kernelSymbol);
    if (hsaStatus == HSA_STATUS_SUCCESS) {
      hsaStatus = hsa_executable_symbol_get_info(kernelSymbol,
                                                 HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE,
                                                 &variable_size);
    }
    if (hsaStatus == HSA_STATUS_SUCCESS) {
      hsaStatus = hsa_executable_symbol_get_info(kernelSymbol,
                                                 HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                                                 &variable_address);
    }

    if (hsaStatus == HSA_STATUS_SUCCESS) {
      const struct RuntimeHandle runtime_handle = {
        kernelCodeHandle_,
        workitemPrivateSegmentByteSize(),
        WorkgroupGroupSegmentByteSize()
      };
      hsaStatus = hsa_memory_copy(reinterpret_cast<void*>(variable_address),
                                  &runtime_handle, variable_size);
    }

    if (hsaStatus != HSA_STATUS_SUCCESS) {
      return false;
    }
  }

  uint32_t wavefront_size = 0;
  if (hsa_agent_get_info(program()->hsaDevice(), HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size) !=
      HSA_STATUS_SUCCESS) {
    return false;
  }
  assert(wavefront_size > 0);

  workGroupInfo_.privateMemSize_ = workitemPrivateSegmentByteSize_;
  workGroupInfo_.localMemSize_ = workgroupGroupSegmentByteSize_;
  workGroupInfo_.usedLDSSize_ = workgroupGroupSegmentByteSize_;
  workGroupInfo_.preferredSizeMultiple_ = wavefront_size;
  workGroupInfo_.usedSGPRs_ = kernelMD.mCodeProps.mNumSGPRs;
  workGroupInfo_.usedVGPRs_ = kernelMD.mCodeProps.mNumVGPRs;
  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.wavefrontPerSIMD_ = program()->dev().info().maxWorkItemSizes_[0] / wavefront_size;
  workGroupInfo_.wavefrontSize_ = wavefront_size;
  workGroupInfo_.size_ = kernelMD.mCodeProps.mMaxFlatWorkGroupSize;
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
#else
static const KernelMD* FindKernelMetadata(const CodeObjectMD* programMD, const std::string& name) {
  for (const KernelMD& kernelMD : programMD->mKernels) {
    if (kernelMD.mName == name) {
      return &kernelMD;
    }
  }
  return nullptr;
}

bool LightningKernel::init() {
  hsa_agent_t hsaDevice = program()->hsaDevice();

  // Pull out metadata from the ELF
  const CodeObjectMD* programMD = static_cast<const LightningProgram*>(program())->metadata();
  assert(programMD != nullptr);

  const KernelMD* kernelMD = FindKernelMetadata(programMD, name());
  if (kernelMD == nullptr) {
    return false;
  }
  InitParameters(*kernelMD, KernargSegmentByteSize());

  // Set the workgroup information for the kernel
  workGroupInfo_.availableLDSSize_ = program()->dev().info().localMemSizePerCU_;
  assert(workGroupInfo_.availableLDSSize_ > 0);
  workGroupInfo_.availableSGPRs_ = 104;
  workGroupInfo_.availableVGPRs_ = 256;

  if (!kernelMD->mAttrs.mReqdWorkGroupSize.empty()) {
    const auto& requiredWorkgroupSize = kernelMD->mAttrs.mReqdWorkGroupSize;
    workGroupInfo_.compileSize_[0] = requiredWorkgroupSize[0];
    workGroupInfo_.compileSize_[1] = requiredWorkgroupSize[1];
    workGroupInfo_.compileSize_[2] = requiredWorkgroupSize[2];
  }

  if (!kernelMD->mAttrs.mWorkGroupSizeHint.empty()) {
    const auto& workgroupSizeHint = kernelMD->mAttrs.mWorkGroupSizeHint;
    workGroupInfo_.compileSizeHint_[0] = workgroupSizeHint[0];
    workGroupInfo_.compileSizeHint_[1] = workgroupSizeHint[1];
    workGroupInfo_.compileSizeHint_[2] = workgroupSizeHint[2];
  }

  if (!kernelMD->mAttrs.mVecTypeHint.empty()) {
    workGroupInfo_.compileVecTypeHint_ = kernelMD->mAttrs.mVecTypeHint.c_str();
  }

  if (!kernelMD->mAttrs.mRuntimeHandle.empty()) {
    hsa_agent_t             agent = program()->hsaDevice();
    hsa_executable_symbol_t kernelSymbol;
    hsa_status_t            status;
    int                     variable_size;
    uint64_t                variable_address;

    // Only kernels that could be enqueued by another kernel has the RuntimeHandle metadata. The RuntimeHandle
    // metadata is a string that represents a variable from which the library code can retrieve the kernel code
    // object handle of such a kernel. The address of the variable and the kernel code object handle are known
    // only after the hsa executable is loaded. The below code copies the kernel code object handle to the
    // address of the variable.

    status = hsa_executable_get_symbol_by_name(program()->hsaExecutable(), kernelMD->mAttrs.mRuntimeHandle.c_str(),
                                               &agent, &kernelSymbol);
    if (status != HSA_STATUS_SUCCESS) {
      return false;
    }

    status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE,
                                            &variable_size);
    if (status != HSA_STATUS_SUCCESS) {
      return false;
    }

    status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                                            &variable_address);
    if (status != HSA_STATUS_SUCCESS) {
      return false;
    }

    const struct RuntimeHandle runtime_handle = {
        kernelCodeHandle_,
        workitemPrivateSegmentByteSize(),
        WorkgroupGroupSegmentByteSize()
    };

    status = hsa_memory_copy(reinterpret_cast<void*>(variable_address), &runtime_handle, variable_size);
    if (status != HSA_STATUS_SUCCESS) {
      return false;
    }
  }

  uint32_t wavefront_size = 0;
  if (hsa_agent_get_info(program()->hsaDevice(), HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size) !=
      HSA_STATUS_SUCCESS) {
    return false;
  }
  assert(wavefront_size > 0);

  workGroupInfo_.privateMemSize_ = workitemPrivateSegmentByteSize_;
  workGroupInfo_.localMemSize_ = workgroupGroupSegmentByteSize_;
  workGroupInfo_.usedLDSSize_ = workgroupGroupSegmentByteSize_;

  workGroupInfo_.preferredSizeMultiple_ = wavefront_size;

  /// TODO: Are there any other fields that are getting queried from akc?
  /// If so, code properties metadata should be used instead.
  workGroupInfo_.usedSGPRs_ = kernelMD->mCodeProps.mNumSGPRs;
  workGroupInfo_.usedVGPRs_ = kernelMD->mCodeProps.mNumVGPRs;

  workGroupInfo_.usedStackSize_ = 0;

  workGroupInfo_.wavefrontPerSIMD_ = program()->dev().info().maxWorkItemSizes_[0] / wavefront_size;

  workGroupInfo_.wavefrontSize_ = wavefront_size;

  workGroupInfo_.size_ = kernelMD->mCodeProps.mMaxFlatWorkGroupSize;
  if (workGroupInfo_.size_ == 0) {
    return false;
  }

  InitPrintf(programMD->mPrintf);

  return true;
}
#endif  // defined(USE_COMGR_LIBRARY)
#endif  // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

#if defined(WITH_COMPILER_LIB)
bool HSAILKernel::init() {
  acl_error errorCode;
  // compile kernel down to ISA
  hsa_agent_t hsaDevice = program()->hsaDevice();
  // Pull out metadata from the ELF
  size_t sizeOfArgList;
  aclCompiler* compileHandle = program()->dev().compiler();
  std::string openClKernelName("&__OpenCL_" + name() + "_kernel");
  errorCode = aclQueryInfo(compileHandle, program()->binaryElf(), RT_ARGUMENT_ARRAY,
                                         openClKernelName.c_str(), nullptr, &sizeOfArgList);
  if (errorCode != ACL_SUCCESS) {
    return false;
  }
  std::unique_ptr<char[]> argList(new char[sizeOfArgList]);
  errorCode = aclQueryInfo(compileHandle, program()->binaryElf(), RT_ARGUMENT_ARRAY,
                                         openClKernelName.c_str(), argList.get(), &sizeOfArgList);
  if (errorCode != ACL_SUCCESS) {
    return false;
  }

  // Set the argList
  InitParameters((const aclArgData*)argList.get(), KernargSegmentByteSize());

  // Set the workgroup information for the kernel
  memset(&workGroupInfo_, 0, sizeof(workGroupInfo_));
  workGroupInfo_.availableLDSSize_ = program()->dev().info().localMemSizePerCU_;
  assert(workGroupInfo_.availableLDSSize_ > 0);
  workGroupInfo_.availableSGPRs_ = 104;
  workGroupInfo_.availableVGPRs_ = 256;
  size_t sizeOfWorkGroupSize;
  errorCode = aclQueryInfo(compileHandle, program()->binaryElf(), RT_WORK_GROUP_SIZE,
                                         openClKernelName.c_str(), nullptr, &sizeOfWorkGroupSize);
  if (errorCode != ACL_SUCCESS) {
    return false;
  }
  errorCode = aclQueryInfo(compileHandle, program()->binaryElf(), RT_WORK_GROUP_SIZE,
                                         openClKernelName.c_str(), workGroupInfo_.compileSize_,
                                         &sizeOfWorkGroupSize);
  if (errorCode != ACL_SUCCESS) {
    return false;
  }

  uint32_t wavefront_size = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(program()->hsaDevice(), HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size)) {
    return false;
  }
  assert(wavefront_size > 0);

  // Setting it the same as used LDS.
  workGroupInfo_.localMemSize_ = workgroupGroupSegmentByteSize_;
  workGroupInfo_.privateMemSize_ = workitemPrivateSegmentByteSize_;
  workGroupInfo_.usedLDSSize_ = workgroupGroupSegmentByteSize_;
  workGroupInfo_.preferredSizeMultiple_ = wavefront_size;

  // Query kernel header object to initialize the number of
  // SGPR's and VGPR's used by the kernel
  const void* kernelHostPtr = nullptr;
  if (Device::loaderQueryHostAddress(reinterpret_cast<const void*>(kernelCodeHandle_),
                                     &kernelHostPtr) == HSA_STATUS_SUCCESS) {
    auto akc = reinterpret_cast<const amd_kernel_code_t*>(kernelHostPtr);
    workGroupInfo_.usedSGPRs_ = akc->wavefront_sgpr_count;
    workGroupInfo_.usedVGPRs_ = akc->workitem_vgpr_count;
  } else {
    workGroupInfo_.usedSGPRs_ = 0;
    workGroupInfo_.usedVGPRs_ = 0;
  }

  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.wavefrontPerSIMD_ = program()->dev().info().maxWorkItemSizes_[0] / wavefront_size;
  workGroupInfo_.wavefrontSize_ = wavefront_size;
  if (workGroupInfo_.compileSize_[0] != 0) {
    workGroupInfo_.size_ = workGroupInfo_.compileSize_[0] * workGroupInfo_.compileSize_[1] *
        workGroupInfo_.compileSize_[2];
  } else {
    workGroupInfo_.size_ = program()->dev().info().preferredWorkGroupSize_;
  }

  // Pull out printf metadata from the ELF
  size_t sizeOfPrintfList;
  errorCode = aclQueryInfo(compileHandle, program()->binaryElf(), RT_GPU_PRINTF_ARRAY,
                                         openClKernelName.c_str(), nullptr, &sizeOfPrintfList);
  if (errorCode != ACL_SUCCESS) {
    return false;
  }

  // Make sure kernel has any printf info
  if (0 != sizeOfPrintfList) {
    std::unique_ptr<char[]> aclPrintfList(new char[sizeOfPrintfList]);
    if (!aclPrintfList) {
      return false;
    }
    errorCode = aclQueryInfo(compileHandle, program()->binaryElf(),
                                           RT_GPU_PRINTF_ARRAY, openClKernelName.c_str(),
                                           aclPrintfList.get(), &sizeOfPrintfList);
    if (errorCode != ACL_SUCCESS) {
      return false;
    }

    // Set the Printf List
    InitPrintf(reinterpret_cast<aclPrintfFmt*>(aclPrintfList.get()));
  }
  return true;
}
#endif // defined(WITH_COMPILER_LIB)

}  // namespace roc
#endif  // WITHOUT_HSA_BACKEND
