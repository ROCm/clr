//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
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

bool HSAILKernel::aqlCreateHWInfo(amd::hsa::loader::Symbol* sym) {
  if (!sym) {
    return false;
  }
  if (!sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, reinterpret_cast<void*>(&code_))) {
    return false;
  }

  amd_kernel_code_t* akc =
      reinterpret_cast<amd_kernel_code_t*>(prog().findHostKernelAddress(code_));
  cpuAqlCode_ = akc;
  if (!sym->GetInfo(HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_SIZE,
                    reinterpret_cast<void*>(&codeSize_))) {
    return false;
  }
  size_t akc_align = 0;
  if (!sym->GetInfo(HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_ALIGN,
                    reinterpret_cast<void*>(&akc_align))) {
    return false;
  }

  workGroupInfo_.scratchRegs_ =
      amd::alignUp(akc->workitem_private_segment_byte_size, 16) / sizeof(uint);
  workGroupInfo_.privateMemSize_ = akc->workitem_private_segment_byte_size;
  workGroupInfo_.localMemSize_ = workGroupInfo_.usedLDSSize_ =
      akc->workgroup_group_segment_byte_size;
  workGroupInfo_.usedSGPRs_ = akc->wavefront_sgpr_count;
  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.usedVGPRs_ = akc->workitem_vgpr_count;

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
  return true;
}

void HSAILKernel::initPrintf(const aclPrintfFmt* aclPrintf) {
  PrintfInfo info;
  uint index = 0;
  for (; aclPrintf->struct_size != 0; aclPrintf++) {
    index = aclPrintf->ID;
    if (printf_.size() <= index) {
      printf_.resize(index + 1);
    }
    std::string pfmt = aclPrintf->fmtStr;
    info.fmtString_.clear();
    bool need_nl = true;
    for (size_t pos = 0; pos < pfmt.size(); ++pos) {
      char symbol = pfmt[pos];
      need_nl = true;
      if (symbol == '\\') {
        // Rest of the C escape sequences (e.g. \') are handled correctly
        // by the MDParser, we are not sure exactly how!
        switch (pfmt[pos + 1]) {
          case 'a':
            pos++;
            symbol = '\a';
            break;
          case 'b':
            pos++;
            symbol = '\b';
            break;
          case 'f':
            pos++;
            symbol = '\f';
            break;
          case 'n':
            pos++;
            symbol = '\n';
            need_nl = false;
            break;
          case 'r':
            pos++;
            symbol = '\r';
            break;
          case 'v':
            pos++;
            symbol = '\v';
            break;
          case '7':
            if (pfmt[pos + 2] == '2') {
              pos += 2;
              symbol = '\72';
            }
            break;
          default:
            break;
        }
      }
      info.fmtString_.push_back(symbol);
    }
    if (need_nl) {
      info.fmtString_ += "\n";
    }
    uint32_t* tmp_ptr = const_cast<uint32_t*>(aclPrintf->argSizes);
    for (uint i = 0; i < aclPrintf->numSizes; i++, tmp_ptr++) {
      info.arguments_.push_back(*tmp_ptr);
    }
    printf_[index] = info;
    info.arguments_.clear();
  }
}

HSAILKernel::HSAILKernel(std::string name, HSAILProgram* prog, std::string compileOptions)
    : device::Kernel(name),
      compileOptions_(compileOptions),
      dev_(prog->dev()),
      prog_(*prog),
      index_(0),
      code_(0),
      codeSize_(0),
      waveLimiter_(
          this,
          (prog->isNull() ? 1
                          : dev().properties().gfxipProperties.shaderCore.numCusPerShaderArray) *
              dev().hwInfo()->simdPerCU_) {
  flags_.hsa_ = true;
}

HSAILKernel::~HSAILKernel() {
}

bool HSAILKernel::init(amd::hsa::loader::Symbol* sym, bool finalize) {
#if defined(WITH_LIGHTNING_COMPILER)
  assert(!"Should not reach here");
#else  // !defined(WITH_LIGHTNING_COMPILER)
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
    initPrintf(reinterpret_cast<aclPrintfFmt*>(aclPrintfList));
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

#endif  // !defined(WITH_LIGHTNING_COMPILER)
  return true;
}

const Device& HSAILKernel::dev() const { return reinterpret_cast<const Device&>(dev_); }

const HSAILProgram& HSAILKernel::prog() const {
  return reinterpret_cast<const HSAILProgram&>(prog_);
}

void HSAILKernel::findLocalWorkSize(size_t workDim, const amd::NDRange& gblWorkSize,
                                    amd::NDRange& lclWorkSize) const {
  // Initialize the default workgoup info
  // Check if the kernel has the compiled sizes
  if (workGroupInfo()->compileSize_[0] == 0) {
    // Find the default local workgroup size, if it wasn't specified
    if (lclWorkSize[0] == 0) {
      bool b1DOverrideSet = !flagIsDefault(GPU_MAX_WORKGROUP_SIZE);
      bool b2DOverrideSet = !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_2D_X) ||
          !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_2D_Y);
      bool b3DOverrideSet = !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_3D_X) ||
          !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_3D_Y) ||
          !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_3D_Z);

      bool overrideSet = ((workDim == 1) && b1DOverrideSet) || ((workDim == 2) && b2DOverrideSet) ||
          ((workDim == 3) && b3DOverrideSet);
      if (!overrideSet) {
        // Find threads per group
        size_t thrPerGrp = workGroupInfo()->size_;

        // Check if kernel uses images
        if (flags_.imageEna_ &&
            // and thread group is a multiple value of wavefronts
            ((thrPerGrp % workGroupInfo()->wavefrontSize_) == 0) &&
            // and it's 2 or 3-dimensional workload
            (workDim > 1) && ((dev().settings().partialDispatch_) ||
                              (((gblWorkSize[0] % 16) == 0) && ((gblWorkSize[1] % 16) == 0)))) {
          // Use 8x8 workgroup size if kernel has image writes
          if (flags_.imageWriteEna_ || (thrPerGrp != dev().info().preferredWorkGroupSize_)) {
            lclWorkSize[0] = 8;
            lclWorkSize[1] = 8;
          } else {
            lclWorkSize[0] = 16;
            lclWorkSize[1] = 16;
          }
          if (workDim == 3) {
            lclWorkSize[2] = 1;
          }
        } else {
          size_t tmp = thrPerGrp;
          // Split the local workgroup into the most efficient way
          for (uint d = 0; d < workDim; ++d) {
            size_t div = tmp;
            for (; (gblWorkSize[d] % div) != 0; div--)
              ;
            lclWorkSize[d] = div;
            tmp /= div;
          }

          // Assuming DWORD access
          const uint cacheLineMatch = dev().settings().cacheLineSize_ >> 2;

          // Check if partial dispatch is enabled and
          if (dev().settings().partialDispatch_ &&
            // we couldn't find optimal workload
            (((lclWorkSize.product() % workGroupInfo()->wavefrontSize_) != 0) ||
                // or size is too small for the cache line
            (lclWorkSize[0] < cacheLineMatch))) {
            size_t maxSize = 0;
            size_t maxDim = 0;
            for (uint d = 0; d < workDim; ++d) {
              if (maxSize < gblWorkSize[d]) {
                maxSize = gblWorkSize[d];
                maxDim = d;
              }
            }
            // Use X dimension as high priority. Runtime will assume that
            // X dimension is more important for the address calculation
            if ((maxDim != 0) && (gblWorkSize[0] >= (cacheLineMatch / 2))) {
              lclWorkSize[0] = cacheLineMatch;
              thrPerGrp /= cacheLineMatch;
              lclWorkSize[maxDim] = thrPerGrp;
              for (uint d = 1; d < workDim; ++d) {
                if (d != maxDim) {
                  lclWorkSize[d] = 1;
                }
              }
            }
            else {
              // Check if a local workgroup has the most optimal size
              if (thrPerGrp > maxSize) {
                thrPerGrp = maxSize;
              }
              lclWorkSize[maxDim] = thrPerGrp;
              for (uint d = 0; d < workDim; ++d) {
                if (d != maxDim) {
                  lclWorkSize[d] = 1;
                }
              }
            }
          }
        }
      } else {
        // Use overrides when app doesn't provide workgroup dimensions
        if (workDim == 1) {
          lclWorkSize[0] = GPU_MAX_WORKGROUP_SIZE;
        } else if (workDim == 2) {
          lclWorkSize[0] = GPU_MAX_WORKGROUP_SIZE_2D_X;
          lclWorkSize[1] = GPU_MAX_WORKGROUP_SIZE_2D_Y;
        } else if (workDim == 3) {
          lclWorkSize[0] = GPU_MAX_WORKGROUP_SIZE_3D_X;
          lclWorkSize[1] = GPU_MAX_WORKGROUP_SIZE_3D_Y;
          lclWorkSize[2] = GPU_MAX_WORKGROUP_SIZE_3D_Z;
        } else {
          assert(0 && "Invalid workDim!");
        }
      }
    }
  } else {
    for (uint d = 0; d < workDim; ++d) {
      lclWorkSize[d] = workGroupInfo()->compileSize_[d];
    }
  }
}

hsa_kernel_dispatch_packet_t* HSAILKernel::loadArguments(
    VirtualGPU& gpu, const amd::Kernel& kernel, const amd::NDRangeContainer& sizes,
    const_address parameters, size_t ldsAddress, uint64_t vmDefQueue, uint64_t* vmParentWrap) const {
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

  const amd::KernelSignature& signature = kernel.signature();

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
          size_t bufferPtr = static_cast<size_t>(gpu.printfDbgHSA().
            dbgBuffer()->vmAddress());
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
    }
  }

  // Load all kernel arguments
  WriteAqlArgAt(aqlArgBuf, parameters, argsBufferSize(), 0);
  // Note: In a case of structs the size won't match,
  // since HSAIL compiler expects a reference...
  assert(argsBufferSize() <= signature.paramsSize() &&
    "A mismatch of sizes of arguments between compiler and runtime!");

  //hsa_kernel_dispatch_packet_t disp;
  hsa_kernel_dispatch_packet_t* hsaDisp = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
    gpu.cb(0)->SysMemCopy());

  amd::NDRange local(sizes.local());
  const amd::NDRange& global = sizes.global();

  // Check if runtime has to find local workgroup size
  findLocalWorkSize(sizes.dimensions(), sizes.global(), local);

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
  hsaDisp->group_segment_size = ldsAddress - ldsSize();
  hsaDisp->kernel_object = gpuAqlCode();

  hsaDisp->kernarg_address = reinterpret_cast<void*>(argList);
  hsaDisp->reserved2 = 0;
  hsaDisp->completion_signal.handle = 0;
  memcpy(aqlArgBuf + argsBufferSize(), hsaDisp, sizeof(hsa_kernel_dispatch_packet_t));

  if (AMD_HSA_BITS_GET(cpuAqlCode_->kernel_code_properties,
      AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)) {
    gpu.addVmMemory(gpu.hsaQueueMem());
  }

  return hsaDisp;
}

#if defined(WITH_LIGHTNING_COMPILER)

const LightningProgram& LightningKernel::prog() const {
  return reinterpret_cast<const LightningProgram&>(prog_);
}

void LightningKernel::initPrintf(const std::vector<std::string>& printfInfoStrings) {
  for (auto str : printfInfoStrings) {
    std::vector<std::string> tokens;

    size_t end, pos = 0;
    do {
      end = str.find_first_of(':', pos);
      tokens.push_back(str.substr(pos, end - pos));
      pos = end + 1;
    } while (end != std::string::npos);

    if (tokens.size() < 2) {
      LogPrintfWarning("Invalid PrintInfo string: \"%s\"", str.c_str());
      continue;
    }

    pos = 0;
    size_t printfInfoID = std::stoi(tokens[pos++]);
    if (printf_.size() <= printfInfoID) {
      printf_.resize(printfInfoID + 1);
    }
    PrintfInfo& info = printf_[printfInfoID];

    size_t numSizes = std::stoi(tokens[pos++]);
    end = pos + numSizes;

    // ensure that we have the correct number of tokens
    if (tokens.size() < end + 1 /*last token is the fmtString*/) {
      LogPrintfWarning("Invalid PrintInfo string: \"%s\"", str.c_str());
      continue;
    }

    // push the argument sizes
    while (pos < end) {
      info.arguments_.push_back(std::stoi(tokens[pos++]));
    }

    // FIXME: We should not need this! [
    std::string& fmt = tokens[pos];
    bool need_nl = true;

    for (pos = 0; pos < fmt.size(); ++pos) {
      char symbol = fmt[pos];
      need_nl = true;
      if (symbol == '\\') {
        switch (fmt[pos + 1]) {
          case 'a':
            pos++;
            symbol = '\a';
            break;
          case 'b':
            pos++;
            symbol = '\b';
            break;
          case 'f':
            pos++;
            symbol = '\f';
            break;
          case 'n':
            pos++;
            symbol = '\n';
            need_nl = false;
            break;
          case 'r':
            pos++;
            symbol = '\r';
            break;
          case 'v':
            pos++;
            symbol = '\v';
            break;
          case '7':
            if (fmt[pos + 2] == '2') {
              pos += 2;
              symbol = '\72';
            }
            break;
          default:
            break;
        }
      }
      info.fmtString_.push_back(symbol);
    }
    if (need_nl) {
      info.fmtString_ += "\n";
    }
    // ]
  }
}

static const KernelMD* FindKernelMetadata(const CodeObjectMD* programMD, const std::string& name) {
  for (const KernelMD& kernelMD : programMD->mKernels) {
    if (kernelMD.mName == name) {
      return &kernelMD;
    }
  }
  return nullptr;
}

bool LightningKernel::init(amd::hsa::loader::Symbol* symbol) {
  flags_.internalKernel_ =
      (compileOptions_.find("-cl-internal-kernel") != std::string::npos) ? true : false;

  aqlCreateHWInfo(symbol);

  const CodeObjectMD* programMD = prog().metadata();
  assert(programMD != nullptr);

  const KernelMD* kernelMD = FindKernelMetadata(programMD, name());

  if (kernelMD == nullptr) {
    return false;
  }

  // Set the argList
  InitParameters(*kernelMD, argsBufferSize());

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

  // Copy wavefront size
  workGroupInfo_.wavefrontSize_ = dev().info().wavefrontWidth_;

  workGroupInfo_.size_ = kernelMD->mCodeProps.mMaxFlatWorkGroupSize;
  if (workGroupInfo_.size_ == 0) {
    return false;
  }

  initPrintf(programMD->mPrintf);

  /*FIXME_lmoriche:
  size_t sizeOfWavesPerSimdHint = sizeof(workGroupInfo_.wavesPerSimdHint_);
  error = aclQueryInfo(dev().compiler(), prog().binaryElf(),
      RT_WAVES_PER_SIMD_HINT, openClKernelName.c_str(),
      &workGroupInfo_.wavesPerSimdHint_, &sizeOfWavesPerSimdHint);
  if (error != ACL_SUCCESS) {
      return false;
  }

  waveLimiter_.enable();
  */

  return true;
}
#endif  // defined(WITH_LIGHTNING_COMPILER)


}  // namespace pal
