//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "platform/ndrange.hpp"
#include "devkernel.hpp"
#include "utils/macros.hpp"
#include "utils/options.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"

#include <string>
#include <sstream>

#include "acl.h"

#if defined(WITH_LIGHTNING_COMPILER)
#include "llvm/Support/AMDGPUMetadata.h"

typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;
#endif  // defined(WITH_LIGHTNING_COMPILER)

namespace device {

 // ================================================================================================
bool Kernel::createSignature(
  const parameters_t& params, uint32_t numParameters,
  uint32_t version) {
  std::stringstream attribs;
  if (workGroupInfo_.compileSize_[0] != 0) {
    attribs << "reqd_work_group_size(";
    for (size_t i = 0; i < 3; ++i) {
      if (i != 0) {
        attribs << ",";
      }

      attribs << workGroupInfo_.compileSize_[i];
    }
    attribs << ")";
  }
  if (workGroupInfo_.compileSizeHint_[0] != 0) {
    attribs << " work_group_size_hint(";
    for (size_t i = 0; i < 3; ++i) {
      if (i != 0) {
        attribs << ",";
      }

      attribs << workGroupInfo_.compileSizeHint_[i];
    }
    attribs << ")";
  }

  if (!workGroupInfo_.compileVecTypeHint_.empty()) {
    attribs << " vec_type_hint(" << workGroupInfo_.compileVecTypeHint_ << ")";
  }

  // Destroy old signature if it was allocated before
  // (offline devices path)
  delete signature_;
  signature_ = new amd::KernelSignature(params, attribs.str(), numParameters, version);
  if (NULL != signature_) {
    return true;
  }
  return false;
}

// ================================================================================================
Kernel::~Kernel() { delete signature_; }

// ================================================================================================
std::string Kernel::openclMangledName(const std::string& name) {
  const oclBIFSymbolStruct* bifSym = findBIF30SymStruct(symOpenclKernel);
  assert(bifSym && "symbol not found");
  return std::string("&") + bifSym->str[bif::PRE] + name + bifSym->str[bif::POST];
}

// ================================================================================================
void Kernel::FindLocalWorkSize(size_t workDim, const amd::NDRange& gblWorkSize,
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
          }
          else {
            lclWorkSize[0] = 16;
            lclWorkSize[1] = 16;
          }
          if (workDim == 3) {
            lclWorkSize[2] = 1;
          }
        }
        else {
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
          const uint cacheLineMatch = dev().info().globalMemCacheLineSize_ >> 2;

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
      }
      else {
        // Use overrides when app doesn't provide workgroup dimensions
        if (workDim == 1) {
          lclWorkSize[0] = GPU_MAX_WORKGROUP_SIZE;
        }
        else if (workDim == 2) {
          lclWorkSize[0] = GPU_MAX_WORKGROUP_SIZE_2D_X;
          lclWorkSize[1] = GPU_MAX_WORKGROUP_SIZE_2D_Y;
        }
        else if (workDim == 3) {
          lclWorkSize[0] = GPU_MAX_WORKGROUP_SIZE_3D_X;
          lclWorkSize[1] = GPU_MAX_WORKGROUP_SIZE_3D_Y;
          lclWorkSize[2] = GPU_MAX_WORKGROUP_SIZE_3D_Z;
        }
        else {
          assert(0 && "Invalid workDim!");
        }
      }
    }
  }
  else {
    for (uint d = 0; d < workDim; ++d) {
      lclWorkSize[d] = workGroupInfo()->compileSize_[d];
    }
  }
}
// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
using llvm::AMDGPU::HSAMD::AccessQualifier;
using llvm::AMDGPU::HSAMD::AddressSpaceQualifier;
using llvm::AMDGPU::HSAMD::ValueKind;
using llvm::AMDGPU::HSAMD::ValueType;

static inline uint32_t GetOclArgumentTypeOCL(const KernelArgMD& lcArg, bool* isHidden) {
  switch (lcArg.mValueKind) {
  case ValueKind::GlobalBuffer:
  case ValueKind::DynamicSharedPointer:
  case ValueKind::Pipe:
    return amd::KernelParameterDescriptor::MemoryObject;
  case ValueKind::ByValue:
    return amd::KernelParameterDescriptor::ValueObject;
  case ValueKind::Image:
    return amd::KernelParameterDescriptor::ImageObject;
  case ValueKind::Sampler:
    return amd::KernelParameterDescriptor::SamplerObject;
  case ValueKind::HiddenGlobalOffsetX:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenGlobalOffsetX;
  case ValueKind::HiddenGlobalOffsetY:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenGlobalOffsetY;
  case ValueKind::HiddenGlobalOffsetZ:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenGlobalOffsetZ;
  case ValueKind::HiddenPrintfBuffer:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenPrintfBuffer;
  case ValueKind::HiddenDefaultQueue:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenDefaultQueue;
  case ValueKind::HiddenCompletionAction:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenCompletionAction;
  case ValueKind::HiddenNone:
  default:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenNone;
  }
}
#endif
// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline uint32_t GetOclArgumentTypeOCL(const aclArgData* argInfo, bool* isHidden) {
  if (argInfo->argStr[0] == '_' && argInfo->argStr[1] == '.') {
    *isHidden = true;
    if (strcmp(&argInfo->argStr[2], "global_offset_0") == 0) {
      return amd::KernelParameterDescriptor::HiddenGlobalOffsetX;
    }
    else if (strcmp(&argInfo->argStr[2], "global_offset_1") == 0) {
      return amd::KernelParameterDescriptor::HiddenGlobalOffsetY;
    }
    else if (strcmp(&argInfo->argStr[2], "global_offset_2") == 0) {
      return amd::KernelParameterDescriptor::HiddenGlobalOffsetZ;
    }
    else if (strcmp(&argInfo->argStr[2], "printf_buffer") == 0) {
      return amd::KernelParameterDescriptor::HiddenPrintfBuffer;
    }
    else if (strcmp(&argInfo->argStr[2], "vqueue_pointer") == 0) {
      return amd::KernelParameterDescriptor::HiddenDefaultQueue;
    }
    else if (strcmp(&argInfo->argStr[2], "aqlwrap_pointer") == 0) {
      return amd::KernelParameterDescriptor::HiddenCompletionAction;
    }
    return amd::KernelParameterDescriptor::HiddenNone;
  }
  switch (argInfo->type) {
  case ARG_TYPE_POINTER:
    return amd::KernelParameterDescriptor::MemoryObject;
  case ARG_TYPE_QUEUE:
    return amd::KernelParameterDescriptor::QueueObject;
  case ARG_TYPE_VALUE:
    return (argInfo->arg.value.data == DATATYPE_struct) ?
      amd::KernelParameterDescriptor::ReferenceObject :
      amd::KernelParameterDescriptor::ValueObject;
  case ARG_TYPE_IMAGE:
    return amd::KernelParameterDescriptor::ImageObject;
  case ARG_TYPE_SAMPLER:
    return amd::KernelParameterDescriptor::SamplerObject;
  case ARG_TYPE_ERROR:
  default:
    return amd::KernelParameterDescriptor::HiddenNone;
}
}
#endif

// ================================================================================================
static const clk_value_type_t ClkValueMapType[6][6] = {
  { T_CHAR, T_CHAR2, T_CHAR3, T_CHAR4, T_CHAR8, T_CHAR16 },
  { T_SHORT, T_SHORT2, T_SHORT3, T_SHORT4, T_SHORT8, T_SHORT16 },
  { T_INT, T_INT2, T_INT3, T_INT4, T_INT8, T_INT16 },
  { T_LONG, T_LONG2, T_LONG3, T_LONG4, T_LONG8, T_LONG16 },
  { T_FLOAT, T_FLOAT2, T_FLOAT3, T_FLOAT4, T_FLOAT8, T_FLOAT16 },
  { T_DOUBLE, T_DOUBLE2, T_DOUBLE3, T_DOUBLE4, T_DOUBLE8, T_DOUBLE16 },
};

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline clk_value_type_t GetOclTypeOCL(const KernelArgMD& lcArg, size_t size = 0) {
  uint sizeType;
  uint numElements;

  if (lcArg.mValueKind != ValueKind::ByValue) {
    switch (lcArg.mValueKind) {
    case ValueKind::GlobalBuffer:
    case ValueKind::DynamicSharedPointer:
    case ValueKind::Pipe:
    case ValueKind::Image:
      return T_POINTER;
    case ValueKind::Sampler:
      return T_SAMPLER;
    default:
      return T_VOID;
    }
  }
  else {
    switch (lcArg.mValueType) {
    case ValueType::I8:
    case ValueType::U8:
      sizeType = 0;
      numElements = size;
      break;
    case ValueType::I16:
    case ValueType::U16:
      sizeType = 1;
      numElements = size / 2;
      break;
    case ValueType::I32:
    case ValueType::U32:
      sizeType = 2;
      numElements = size / 4;
      break;
    case ValueType::I64:
    case ValueType::U64:
      sizeType = 3;
      numElements = size / 8;
      break;
    case ValueType::F16:
      sizeType = 4;
      numElements = size / 2;
      break;
    case ValueType::F32:
      sizeType = 4;
      numElements = size / 4;
      break;
    case ValueType::F64:
      sizeType = 5;
      numElements = size / 8;
      break;
    case ValueType::Struct:
    default:
      return T_VOID;
    }
    switch (numElements) {
    case 1:
      return ClkValueMapType[sizeType][0];
    case 2:
      return ClkValueMapType[sizeType][1];
    case 3:
      return ClkValueMapType[sizeType][2];
    case 4:
      return ClkValueMapType[sizeType][3];
    case 8:
      return ClkValueMapType[sizeType][4];
    case 16:
      return ClkValueMapType[sizeType][5];
    default:
      return T_VOID;
    }
  }
  return T_VOID;
}
#endif
// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline clk_value_type_t GetOclTypeOCL(const aclArgData* argInfo, size_t size = 0) {
  uint sizeType;
  uint numElements;
  if (argInfo->type == ARG_TYPE_QUEUE) {
    return T_QUEUE;
  }
  else if (argInfo->type == ARG_TYPE_POINTER || argInfo->type == ARG_TYPE_IMAGE) {
    return T_POINTER;
  }
  else if (argInfo->type == ARG_TYPE_VALUE) {
    switch (argInfo->arg.value.data) {
    case DATATYPE_i8:
    case DATATYPE_u8:
      sizeType = 0;
      numElements = size;
      break;
    case DATATYPE_i16:
    case DATATYPE_u16:
      sizeType = 1;
      numElements = size / 2;
      break;
    case DATATYPE_i32:
    case DATATYPE_u32:
      sizeType = 2;
      numElements = size / 4;
      break;
    case DATATYPE_i64:
    case DATATYPE_u64:
      sizeType = 3;
      numElements = size / 8;
      break;
    case DATATYPE_f16:
      sizeType = 4;
      numElements = size / 2;
      break;
    case DATATYPE_f32:
      sizeType = 4;
      numElements = size / 4;
      break;
    case DATATYPE_f64:
      sizeType = 5;
      numElements = size / 8;
      break;
    case DATATYPE_struct:
    case DATATYPE_opaque:
    case DATATYPE_ERROR:
    default:
      return T_VOID;
    }

    switch (numElements) {
    case 1:
      return ClkValueMapType[sizeType][0];
    case 2:
      return ClkValueMapType[sizeType][1];
    case 3:
      return ClkValueMapType[sizeType][2];
    case 4:
      return ClkValueMapType[sizeType][3];
    case 8:
      return ClkValueMapType[sizeType][4];
    case 16:
      return ClkValueMapType[sizeType][5];
    default:
      return T_VOID;
    }
  }
  else if (argInfo->type == ARG_TYPE_SAMPLER) {
    return T_SAMPLER;
  }
  else {
    return T_VOID;
  }
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline size_t GetArgAlignmentOCL(const KernelArgMD& lcArg) { return lcArg.mAlign; }
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline size_t GetArgAlignmentOCL(const aclArgData* argInfo) {
  switch (argInfo->type) {
  case ARG_TYPE_POINTER:
    return sizeof(void*);
  case ARG_TYPE_VALUE:
    switch (argInfo->arg.value.data) {
    case DATATYPE_i8:
    case DATATYPE_u8:
      return 1;
    case DATATYPE_u16:
    case DATATYPE_i16:
    case DATATYPE_f16:
      return 2;
    case DATATYPE_u32:
    case DATATYPE_i32:
    case DATATYPE_f32:
      return 4;
    case DATATYPE_i64:
    case DATATYPE_u64:
    case DATATYPE_f64:
      return 8;
    case DATATYPE_struct:
      return 128;
    case DATATYPE_ERROR:
    default:
      return -1;
    }
  case ARG_TYPE_IMAGE:
    return sizeof(cl_mem);
  case ARG_TYPE_SAMPLER:
    return sizeof(cl_sampler);
  default:
    return -1;
  }
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline size_t GetArgPointeeAlignmentOCL(const KernelArgMD& lcArg) {
  if (lcArg.mValueKind == ValueKind::DynamicSharedPointer) {
    uint32_t align = lcArg.mPointeeAlign;
    if (align == 0) {
      LogWarning("Missing DynamicSharedPointer alignment");
      align = 128; /* worst case alignment */
    }
    return align;
  }
  return 1;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline size_t GetArgPointeeAlignmentOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_POINTER) {
    return argInfo->arg.pointer.align;
  }
  return 1;
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline bool GetReadOnlyOCL(const KernelArgMD& lcArg) {
  if ((lcArg.mValueKind == ValueKind::GlobalBuffer) || (lcArg.mValueKind == ValueKind::Image)) {
    switch (lcArg.mAccQual) {
    case AccessQualifier::ReadOnly:
      return true;
    case AccessQualifier::WriteOnly:
    case AccessQualifier::ReadWrite:
    default:
      return false;
    }
  }
  return false;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline bool GetReadOnlyOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_POINTER) {
    return (argInfo->arg.pointer.type == ACCESS_TYPE_RO) ? true : false;
  }
  else if (argInfo->type == ARG_TYPE_IMAGE) {
    return (argInfo->arg.image.type == ACCESS_TYPE_RO) ? true : false;
  }
  return false;
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline int GetArgSizeOCL(const KernelArgMD& lcArg) { return lcArg.mSize; }
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
inline static int GetArgSizeOCL(const aclArgData* argInfo) {
  switch (argInfo->type) {
  case ARG_TYPE_POINTER:
    return sizeof(void*);
  case ARG_TYPE_VALUE:
    switch (argInfo->arg.value.data) {
    case DATATYPE_i8:
    case DATATYPE_u8:
    case DATATYPE_struct:
      return 1 * argInfo->arg.value.numElements;
    case DATATYPE_u16:
    case DATATYPE_i16:
    case DATATYPE_f16:
      return 2 * argInfo->arg.value.numElements;
    case DATATYPE_u32:
    case DATATYPE_i32:
    case DATATYPE_f32:
      return 4 * argInfo->arg.value.numElements;
    case DATATYPE_i64:
    case DATATYPE_u64:
    case DATATYPE_f64:
      return 8 * argInfo->arg.value.numElements;
    case DATATYPE_ERROR:
    default:
      return -1;
    }
  case ARG_TYPE_IMAGE:
  case ARG_TYPE_SAMPLER:
  case ARG_TYPE_QUEUE:
    return sizeof(void*);
  default:
    return -1;
  }
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline cl_kernel_arg_address_qualifier GetOclAddrQualOCL(const KernelArgMD& lcArg) {
  if (lcArg.mValueKind == ValueKind::DynamicSharedPointer) {
    return CL_KERNEL_ARG_ADDRESS_LOCAL;
  }
  else if (lcArg.mValueKind == ValueKind::GlobalBuffer) {
    if (lcArg.mAddrSpaceQual == AddressSpaceQualifier::Global ||
        lcArg.mAddrSpaceQual == AddressSpaceQualifier::Generic) {
      return CL_KERNEL_ARG_ADDRESS_GLOBAL;
    }
    else if (lcArg.mAddrSpaceQual == AddressSpaceQualifier::Constant) {
      return CL_KERNEL_ARG_ADDRESS_CONSTANT;
    }
    LogError("Unsupported address type");
    return CL_KERNEL_ARG_ADDRESS_PRIVATE;
  }
  else if (lcArg.mValueKind == ValueKind::Image || lcArg.mValueKind == ValueKind::Pipe) {
    return CL_KERNEL_ARG_ADDRESS_GLOBAL;
  }
  // default for all other cases
  return CL_KERNEL_ARG_ADDRESS_PRIVATE;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline cl_kernel_arg_address_qualifier GetOclAddrQualOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_POINTER) {
    switch (argInfo->arg.pointer.memory) {
    case PTR_MT_UAV_CONSTANT:
    case PTR_MT_CONSTANT_EMU:
    case PTR_MT_CONSTANT:
      return CL_KERNEL_ARG_ADDRESS_CONSTANT;
    case PTR_MT_UAV:
    case PTR_MT_GLOBAL:
    case PTR_MT_SCRATCH_EMU:
      return CL_KERNEL_ARG_ADDRESS_GLOBAL;
    case PTR_MT_LDS_EMU:
    case PTR_MT_LDS:
      return CL_KERNEL_ARG_ADDRESS_LOCAL;
    case PTR_MT_ERROR:
    default:
      LogError("Unsupported address type");
      return CL_KERNEL_ARG_ADDRESS_PRIVATE;
    }
  }
  else if ((argInfo->type == ARG_TYPE_IMAGE) || (argInfo->type == ARG_TYPE_QUEUE)) {
    return CL_KERNEL_ARG_ADDRESS_GLOBAL;
  }

  // default for all other cases
  return CL_KERNEL_ARG_ADDRESS_PRIVATE;
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline cl_kernel_arg_access_qualifier GetOclAccessQualOCL(const KernelArgMD& lcArg) {
  if (lcArg.mValueKind == ValueKind::Image) {
    switch (lcArg.mAccQual) {
    case AccessQualifier::ReadOnly:
      return CL_KERNEL_ARG_ACCESS_READ_ONLY;
    case AccessQualifier::WriteOnly:
      return CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
    case AccessQualifier::ReadWrite:
    default:
      return CL_KERNEL_ARG_ACCESS_READ_WRITE;
    }
  }
  return CL_KERNEL_ARG_ACCESS_NONE;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline cl_kernel_arg_access_qualifier GetOclAccessQualOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_IMAGE) {
    switch (argInfo->arg.image.type) {
    case ACCESS_TYPE_RO:
      return CL_KERNEL_ARG_ACCESS_READ_ONLY;
    case ACCESS_TYPE_WO:
      return CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
    default:
      return CL_KERNEL_ARG_ACCESS_READ_WRITE;
    }
  }
  return CL_KERNEL_ARG_ACCESS_NONE;
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
static inline cl_kernel_arg_type_qualifier GetOclTypeQualOCL(const KernelArgMD& lcArg) {
  cl_kernel_arg_type_qualifier rv = CL_KERNEL_ARG_TYPE_NONE;
  if (lcArg.mValueKind == ValueKind::GlobalBuffer ||
    lcArg.mValueKind == ValueKind::DynamicSharedPointer) {
    if (lcArg.mIsVolatile) {
      rv |= CL_KERNEL_ARG_TYPE_VOLATILE;
    }
    if (lcArg.mIsRestrict) {
      rv |= CL_KERNEL_ARG_TYPE_RESTRICT;
    }
    if (lcArg.mIsConst) {
      rv |= CL_KERNEL_ARG_TYPE_CONST;
    }
  }
  else if (lcArg.mIsPipe) {
    assert(lcArg.mValueKind == ValueKind::Pipe);
    rv |= CL_KERNEL_ARG_TYPE_PIPE;
  }
  return rv;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
static inline cl_kernel_arg_type_qualifier GetOclTypeQualOCL(const aclArgData* argInfo) {
  cl_kernel_arg_type_qualifier rv = CL_KERNEL_ARG_TYPE_NONE;
  if (argInfo->type == ARG_TYPE_POINTER) {
    if (argInfo->arg.pointer.isVolatile) {
      rv |= CL_KERNEL_ARG_TYPE_VOLATILE;
    }
    if (argInfo->arg.pointer.isRestrict) {
      rv |= CL_KERNEL_ARG_TYPE_RESTRICT;
    }
    if (argInfo->arg.pointer.isPipe) {
      rv |= CL_KERNEL_ARG_TYPE_PIPE;
    }
    if (argInfo->isConst) {
      rv |= CL_KERNEL_ARG_TYPE_CONST;
    }
    switch (argInfo->arg.pointer.memory) {
    case PTR_MT_CONSTANT:
    case PTR_MT_UAV_CONSTANT:
    case PTR_MT_CONSTANT_EMU:
      rv |= CL_KERNEL_ARG_TYPE_CONST;
      break;
    default:
      break;
    }
  }
  return rv;
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
void Kernel::InitParameters(const KernelMD& kernelMD, uint32_t argBufferSize) {
  // Iterate through the arguments and insert into parameterList
  device::Kernel::parameters_t params;
  device::Kernel::parameters_t hiddenParams;
  amd::KernelParameterDescriptor desc;
  size_t offset = 0;
  size_t offsetStruct = argBufferSize;

  for (size_t i = 0; i < kernelMD.mArgs.size(); ++i) {
    const KernelArgMD& lcArg = kernelMD.mArgs[i];

    size_t size = GetArgSizeOCL(lcArg);
    size_t alignment = GetArgAlignmentOCL(lcArg);
    bool isHidden = false;
    desc.info_.oclObject_ = GetOclArgumentTypeOCL(lcArg, &isHidden);

    // Allocate the hidden arguments, but abstraction layer will skip them
    if (isHidden) {
      offset = amd::alignUp(offset, alignment);
      desc.offset_ = offset;
      desc.size_ = size;
      offset += size;
      hiddenParams.push_back(desc);
      continue;
    }

    desc.name_ = lcArg.mName.c_str();
    desc.type_ = GetOclTypeOCL(lcArg, size);
    desc.typeName_ = lcArg.mTypeName.c_str();

    desc.addressQualifier_ = GetOclAddrQualOCL(lcArg);
    desc.accessQualifier_ = GetOclAccessQualOCL(lcArg);
    desc.typeQualifier_ = GetOclTypeQualOCL(lcArg);
    desc.info_.arrayIndex_ = GetArgPointeeAlignmentOCL(lcArg);
    desc.size_ = size;

    // These objects have forced data size to uint64_t
    if ((desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) ||
      (desc.info_.oclObject_ == amd::KernelParameterDescriptor::SamplerObject) ||
      (desc.info_.oclObject_ == amd::KernelParameterDescriptor::QueueObject)) {
      offset = amd::alignUp(offset, sizeof(uint64_t));
      desc.offset_ = offset;
      offset += sizeof(uint64_t);
    }
    else {
      offset = amd::alignUp(offset, alignment);
      desc.offset_ = offset;
      offset += size;
    }

    // Update read only flag
    desc.info_.readOnly_ = GetReadOnlyOCL(lcArg);

    params.push_back(desc);

    if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
      flags_.imageEna_ = true;
      if (desc.accessQualifier_ != CL_KERNEL_ARG_ACCESS_READ_ONLY) {
        flags_.imageWriteEna_ = true;
      }
    }
  }

  // Save the number of OCL arguments
  uint32_t numParams = params.size();
  // Append the hidden arguments to the OCL arguments
  params.insert(params.end(), hiddenParams.begin(), hiddenParams.end());
  createSignature(params, numParams, amd::KernelSignature::ABIVersion_1);
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
void Kernel::InitParameters(const aclArgData* aclArg, uint32_t argBufferSize) {
  // Iterate through the arguments and insert into parameterList
  device::Kernel::parameters_t params;
  device::Kernel::parameters_t hiddenParams;
  amd::KernelParameterDescriptor desc;
  size_t offset = 0;
  size_t offsetStruct = argBufferSize;

  for (uint i = 0; aclArg->struct_size != 0; i++, aclArg++) {
    size_t size = GetArgSizeOCL(aclArg);
    size_t alignment = GetArgAlignmentOCL(aclArg);
    bool isHidden = false;
    desc.info_.oclObject_ = GetOclArgumentTypeOCL(aclArg, &isHidden);

    // Allocate the hidden arguments, but abstraction layer will skip them
    if (isHidden) {
      offset = amd::alignUp(offset, alignment);
      desc.offset_ = offset;
      desc.size_ = size;
      offset += size;
      hiddenParams.push_back(desc);
      continue;
    }

    desc.name_ = aclArg->argStr;
    desc.typeName_ = aclArg->typeStr;
    desc.type_ = GetOclTypeOCL(aclArg, size);

    desc.addressQualifier_ = GetOclAddrQualOCL(aclArg);
    desc.accessQualifier_ = GetOclAccessQualOCL(aclArg);
    desc.typeQualifier_ = GetOclTypeQualOCL(aclArg);
    desc.info_.arrayIndex_ = GetArgPointeeAlignmentOCL(aclArg);
    desc.size_ = size;

    // Check if HSAIL expects data by reference and allocate it behind
    if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ReferenceObject) {
      desc.offset_ = offsetStruct;
      // Align the offset reference
      offset = amd::alignUp(offset, sizeof(size_t));
      patchReferences_.insert({ desc.offset_, offset });
      offsetStruct += size;
      // Adjust the offset of arguments
      offset += sizeof(size_t);
    }
    else {
      // These objects have forced data size to uint64_t
      if ((desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) ||
        (desc.info_.oclObject_ == amd::KernelParameterDescriptor::SamplerObject) ||
        (desc.info_.oclObject_ == amd::KernelParameterDescriptor::QueueObject)) {
        offset = amd::alignUp(offset, sizeof(uint64_t));
        desc.offset_ = offset;
        offset += sizeof(uint64_t);
      }
      else {
        offset = amd::alignUp(offset, alignment);
        desc.offset_ = offset;
        offset += size;
      }
    }
    // Update read only flag
    desc.info_.readOnly_ = GetReadOnlyOCL(aclArg);

    params.push_back(desc);

    if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
      flags_.imageEna_ = true;
      if (desc.accessQualifier_ != CL_KERNEL_ARG_ACCESS_READ_ONLY) {
        flags_.imageWriteEna_ = true;
      }
    }
  }
  // Save the number of OCL arguments
  uint32_t numParams = params.size();
  // Append the hidden arguments to the OCL arguments
  params.insert(params.end(), hiddenParams.begin(), hiddenParams.end());
  createSignature(params, numParams, amd::KernelSignature::ABIVersion_1);
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER)
void Kernel::InitPrintf(const std::vector<std::string>& printfInfoStrings) {
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
#endif  // defined(WITH_LIGHTNING_COMPILER)

// ================================================================================================
#if defined(WITH_COMPILER_LIB) || !defined(WITH_LIGHTNING_COMPILER)
void Kernel::InitPrintf(const aclPrintfFmt* aclPrintf) {
  PrintfInfo info;
  uint index = 0;
  for (; aclPrintf->struct_size != 0; aclPrintf++) {
    index = aclPrintf->ID;
    if (printf_.size() <= index) {
      printf_.resize(index + 1);
    }
    std::string pfmt = aclPrintf->fmtStr;
    bool need_nl = true;
    for (size_t pos = 0; pos < pfmt.size(); ++pos) {
      char symbol = pfmt[pos];
      need_nl = true;
      if (symbol == '\\') {
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
#endif // defined(WITH_COMPILER_LIB)

}