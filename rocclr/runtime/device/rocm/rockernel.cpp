//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//

#include "rockernel.hpp"
#include "SCHSAInterface.h"
#include "amd_hsa_kernel_code.h"

#include <algorithm>

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

#if defined(WITH_LIGHTNING_COMPILER)

using llvm::AMDGPU::CodeObject::AccessQualifier;
using llvm::AMDGPU::CodeObject::AddressSpaceQualifier;
using llvm::AMDGPU::CodeObject::ValueKind;
using llvm::AMDGPU::CodeObject::ValueType;

static inline ROC_ARG_TYPE
GetKernelArgType(const KernelArgMD& lcArg)
{
    switch (lcArg.mValueKind) {
    case ValueKind::GlobalBuffer:
    case ValueKind::DynamicSharedPointer:
        return ROC_ARGTYPE_POINTER;
    case ValueKind::ByValue:
        return ROC_ARGTYPE_VALUE;
    case ValueKind::Image:
        return ROC_ARGTYPE_IMAGE;
    case ValueKind::Sampler:
        return ROC_ARGTYPE_SAMPLER;
    case ValueKind::HiddenGlobalOffsetX:
        return ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_X;
    case ValueKind::HiddenGlobalOffsetY:
        return ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Y;
    case ValueKind::HiddenGlobalOffsetZ:
        return ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Z;
    case ValueKind::HiddenPrintfBuffer:
        return ROC_ARGTYPE_HIDDEN_PRINTF_BUFFER;
    case ValueKind::HiddenDefaultQueue:
        return ROC_ARGTYPE_HIDDEN_DEFAULT_QUEUE;
    case ValueKind::HiddenCompletionAction:
        return ROC_ARGTYPE_HIDDEN_COMPLETION_ACTION;
    case ValueKind::HiddenNone:
        return ROC_ARGTYPE_HIDDEN_NONE;
    default:
        return ROC_ARGTYPE_ERROR;
    }
}
#endif // defined(WITH_LIGHTNING_COMPILER)

static inline ROC_ARG_TYPE
GetKernelArgType(const aclArgData* argInfo)
{
    if (argInfo->argStr[0] == '_' && argInfo->argStr[1] == '.') {
        if (strcmp(&argInfo->argStr[2], "global_offset_0") == 0) {
            return ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_X;
        }
        else if (strcmp(&argInfo->argStr[2], "global_offset_1") == 0) {
            return ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Y;
        }
        else if (strcmp(&argInfo->argStr[2], "global_offset_2") == 0) {
            return ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Z;
        }
        else if (strcmp(&argInfo->argStr[2], "printf_buffer") == 0) {
            return ROC_ARGTYPE_HIDDEN_PRINTF_BUFFER;
        }
        else if (strcmp(&argInfo->argStr[2], "vqueue_pointer") == 0) {
            return ROC_ARGTYPE_HIDDEN_DEFAULT_QUEUE;
        }
        else if (strcmp(&argInfo->argStr[2], "aqlwrap_pointer") == 0) {
            return ROC_ARGTYPE_HIDDEN_COMPLETION_ACTION;
        }
        return ROC_ARGTYPE_HIDDEN_NONE;
    }

    switch (argInfo->type) {
    case ARG_TYPE_POINTER:
        return ROC_ARGTYPE_POINTER;
    case ARG_TYPE_VALUE:
        return (argInfo->arg.value.data == DATATYPE_struct)
            ? ROC_ARGTYPE_REFERENCE : ROC_ARGTYPE_VALUE;
    case ARG_TYPE_IMAGE:
        return ROC_ARGTYPE_IMAGE;
    case ARG_TYPE_SAMPLER:
        return ROC_ARGTYPE_SAMPLER;
    case ARG_TYPE_ERROR:
    default:
        return ROC_ARGTYPE_ERROR;
    }
}

#if defined(WITH_LIGHTNING_COMPILER)
static inline size_t
GetKernelArgAlignment(const KernelArgMD& lcArg)
{
    return lcArg.mAlign;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

static inline size_t
GetKernelArgAlignment(const aclArgData* argInfo)
{
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
    case ARG_TYPE_IMAGE: return sizeof(cl_mem);
    case ARG_TYPE_SAMPLER: return sizeof(cl_sampler);
    default: return -1;
    }
}

#if defined(WITH_LIGHTNING_COMPILER)
static inline size_t
GetKernelArgPointeeAlignment(const KernelArgMD& lcArg)
{
    if (lcArg.mValueKind == ValueKind::DynamicSharedPointer) {
         uint32_t align = lcArg.mPointeeAlign;
         if (align == 0) {
             LogWarning("Missing DynamicSharedPointer alignment");
             align = 128; /* worst case alignment */;
         }
         return align;
    }
    return 1;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

static inline size_t
GetKernelArgPointeeAlignment(const aclArgData* argInfo)
{
    if (argInfo->type == ARG_TYPE_POINTER) {
        return argInfo->arg.pointer.align;
    }
    return 1;
}

#if defined(WITH_LIGHTNING_COMPILER)
static inline ROC_ACCESS_TYPE
GetKernelArgAccessType(const KernelArgMD& lcArg)
{
    if (lcArg.mValueKind == ValueKind::GlobalBuffer ||
        lcArg.mValueKind == ValueKind::Image) {
        switch (lcArg.mAccQual) {
        case AccessQualifier::ReadOnly:
            return ROC_ACCESS_TYPE_RO;
        case AccessQualifier::WriteOnly:
            return ROC_ACCESS_TYPE_WO;
        case AccessQualifier::ReadWrite:
        default:
            return ROC_ACCESS_TYPE_RW;
        }
    }
    return ROC_ACCESS_TYPE_NONE;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

static inline ROC_ACCESS_TYPE
GetKernelArgAccessType(const aclArgData* argInfo)
{
    aclAccessType accessType;

    if (argInfo->type == ARG_TYPE_POINTER) {
        accessType = argInfo->arg.pointer.type;
    }
    else if (argInfo->type == ARG_TYPE_IMAGE) {
        accessType = argInfo->arg.image.type;
    }
    else {
        return ROC_ACCESS_TYPE_NONE;
    }
    if (accessType == ACCESS_TYPE_RO) {
        return ROC_ACCESS_TYPE_RO;
        }
    else if (accessType == ACCESS_TYPE_WO) {
        return ROC_ACCESS_TYPE_WO;
    }

    return ROC_ACCESS_TYPE_RW;
}

#if defined(WITH_LIGHTNING_COMPILER)
static inline ROC_ADDRESS_QUALIFIER
GetKernelAddrQual(const KernelArgMD& lcArg)
{
    if (lcArg.mValueKind == ValueKind::DynamicSharedPointer) {
        return ROC_ADDRESS_LOCAL;
    }
    else if (lcArg.mValueKind == ValueKind::GlobalBuffer) {
        if (lcArg.mAddrSpaceQual == AddressSpaceQualifier::Global) {
            return ROC_ADDRESS_GLOBAL;
        }
        else if (lcArg.mAddrSpaceQual == AddressSpaceQualifier::Constant) {
            return ROC_ADDRESS_CONSTANT;
        }
        LogError("Unsupported address type");
        return ROC_ADDRESS_ERROR;
    }
    else if (lcArg.mValueKind == ValueKind::Image
        || lcArg.mValueKind == ValueKind::Sampler) {
        return ROC_ADDRESS_GLOBAL;
    }
    return ROC_ADDRESS_ERROR;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

static inline ROC_ADDRESS_QUALIFIER
GetKernelAddrQual(const aclArgData* argInfo)
{
    if (argInfo->type == ARG_TYPE_POINTER) {
        switch (argInfo->arg.pointer.memory) {
        case PTR_MT_CONSTANT_EMU:
        case PTR_MT_UAV_CONSTANT:
        case PTR_MT_CONSTANT:
            return ROC_ADDRESS_CONSTANT;
        case PTR_MT_UAV:
        case PTR_MT_GLOBAL:
            return ROC_ADDRESS_GLOBAL;
        case PTR_MT_LDS_EMU:
        case PTR_MT_LDS:
            return ROC_ADDRESS_LOCAL;
        case PTR_MT_ERROR:
        default:
            LogError("Unsupported address type");
            return ROC_ADDRESS_ERROR;
        }
    }
    else if ((argInfo->type == ARG_TYPE_IMAGE) ||
             (argInfo->type == ARG_TYPE_SAMPLER)) {
        return ROC_ADDRESS_GLOBAL;
    }
    return ROC_ADDRESS_ERROR;
}

#if defined(WITH_LIGHTNING_COMPILER)
static inline ROC_DATA_TYPE
GetKernelDataType(const KernelArgMD& lcArg)
{
    aclArgDataType dataType;

    if (lcArg.mValueKind != ValueKind::ByValue) {
        return ROC_DATATYPE_ERROR;
    }

    switch (lcArg.mValueType) {
    case ValueType::I8:
        return ROC_DATATYPE_S8;
    case ValueType::I16:
        return ROC_DATATYPE_S16;
    case ValueType::I32:
        return ROC_DATATYPE_S32;
    case ValueType::I64:
        return ROC_DATATYPE_S64;
    case ValueType::U8:
        return ROC_DATATYPE_U8;
    case ValueType::U16:
        return ROC_DATATYPE_U16;
    case ValueType::U32:
        return ROC_DATATYPE_U32;
    case ValueType::U64:
        return ROC_DATATYPE_U64;
    case ValueType::F16:
        return ROC_DATATYPE_F16;
    case ValueType::F32:
        return ROC_DATATYPE_F32;
    case ValueType::F64:
        return ROC_DATATYPE_F64;
    case ValueType::Struct:
        return ROC_DATATYPE_STRUCT;
    default:
        return ROC_DATATYPE_ERROR;
    }
}
#endif // defined(WITH_LIGHTNING_COMPILER)

/* f16 returns f32 - workaround due to comp lib */
static inline ROC_DATA_TYPE
GetKernelDataType(const aclArgData* argInfo)
{
    aclArgDataType dataType;

    if (argInfo->type == ARG_TYPE_POINTER) {
        dataType = argInfo->arg.pointer.data;
    }
    else if (argInfo->type == ARG_TYPE_VALUE) {
        dataType = argInfo->arg.value.data;
    }
    else {
        return ROC_DATATYPE_ERROR;
    }
    switch (dataType) {
    case DATATYPE_i1:
        return ROC_DATATYPE_B1;
    case DATATYPE_i8:
        return ROC_DATATYPE_S8;
    case DATATYPE_i16:
        return ROC_DATATYPE_S16;
    case DATATYPE_i32:
        return ROC_DATATYPE_S32;
    case DATATYPE_i64:
        return ROC_DATATYPE_S64;
    case DATATYPE_u8:
        return ROC_DATATYPE_U8;
    case DATATYPE_u16:
        return ROC_DATATYPE_U16;
    case DATATYPE_u32:
        return ROC_DATATYPE_U32;
    case DATATYPE_u64:
        return ROC_DATATYPE_U64;
    case DATATYPE_f16:
        return ROC_DATATYPE_F32;
    case DATATYPE_f32:
        return ROC_DATATYPE_F32;
    case DATATYPE_f64:
        return ROC_DATATYPE_F64;
    case DATATYPE_struct:
        return ROC_DATATYPE_STRUCT;
    case DATATYPE_opaque:
        return ROC_DATATYPE_OPAQUE;
    case DATATYPE_ERROR:
    default:
        return ROC_DATATYPE_ERROR;
    }
}

static inline int
GetKernelArgSize(const aclArgData* argInfo)
{
    switch (argInfo->type) {
        case ARG_TYPE_POINTER: return sizeof(void *);
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
            default: return -1;
            }
        case ARG_TYPE_IMAGE: return sizeof(cl_mem);
        case ARG_TYPE_SAMPLER: return sizeof(cl_sampler);
        default: return -1;
    }
}

static inline clk_value_type_t
GetOclType(const Kernel::Argument* arg)
{
    static const clk_value_type_t   ClkValueMapType[6][6] = {
        { T_CHAR,   T_CHAR2,    T_CHAR3,    T_CHAR4,    T_CHAR8,    T_CHAR16   },
        { T_SHORT,  T_SHORT2,   T_SHORT3,   T_SHORT4,   T_SHORT8,   T_SHORT16  },
        { T_INT,    T_INT2,     T_INT3,     T_INT4,     T_INT8,     T_INT16    },
        { T_LONG,   T_LONG2,    T_LONG3,    T_LONG4,    T_LONG8,    T_LONG16   },
        { T_FLOAT,  T_FLOAT2,   T_FLOAT3,   T_FLOAT4,   T_FLOAT8,   T_FLOAT16  },
        { T_DOUBLE, T_DOUBLE2,  T_DOUBLE3,  T_DOUBLE4,  T_DOUBLE8,  T_DOUBLE16 },
    };

    uint sizeType;
    uint numElements;
    if (arg->type_ == ROC_ARGTYPE_POINTER || arg->type_ == ROC_ARGTYPE_IMAGE) {
            return T_POINTER;
    }
    else if (arg->type_ == ROC_ARGTYPE_VALUE
         || arg->type_ == ROC_ARGTYPE_REFERENCE) {
        switch (arg->dataType_) {
        case ROC_DATATYPE_S8:
        case ROC_DATATYPE_U8:
            sizeType = 0;
            numElements  = arg->size_;
            break;
        case ROC_DATATYPE_S16:
        case ROC_DATATYPE_U16:
            sizeType = 1;
            numElements  = arg->size_ / 2;
            break;
        case ROC_DATATYPE_S32:
        case ROC_DATATYPE_U32:
            sizeType = 2;
            numElements  = arg->size_ / 4;
            break;
        case ROC_DATATYPE_S64:
        case ROC_DATATYPE_U64:
            sizeType = 3;
            numElements  = arg->size_ / 8;
            break;
        case ROC_DATATYPE_F16:
            sizeType = 4;
            numElements  = arg->size_ / 2;
            break;
        case ROC_DATATYPE_F32:
            sizeType = 4;
            numElements  = arg->size_ / 4;
            break;
        case ROC_DATATYPE_F64:
            sizeType = 5;
            numElements  = arg->size_ / 8;
            break;
        default:
            return T_VOID;
        }

        switch (numElements) {
            case 1: return ClkValueMapType[sizeType][0];
            case 2: return ClkValueMapType[sizeType][1];
            case 3: return ClkValueMapType[sizeType][2];
            case 4: return ClkValueMapType[sizeType][3];
            case 8: return ClkValueMapType[sizeType][4];
            case 16: return ClkValueMapType[sizeType][5];
            default: return T_VOID;
        }
    }
    else if (arg->type_ == ROC_ARGTYPE_SAMPLER) {
        return T_SAMPLER;
    }
    else {
        return T_VOID;
    }
}

static inline cl_kernel_arg_address_qualifier
GetOclAddrQual(const Kernel::Argument* arg)
{
    if (arg->type_ == ROC_ARGTYPE_POINTER) {
        switch (arg->addrQual_) {
        case ROC_ADDRESS_GLOBAL:
            return CL_KERNEL_ARG_ADDRESS_GLOBAL;
        case ROC_ADDRESS_CONSTANT:
            return CL_KERNEL_ARG_ADDRESS_CONSTANT;
        case ROC_ADDRESS_LOCAL:
            return CL_KERNEL_ARG_ADDRESS_LOCAL;
        default:
            return CL_KERNEL_ARG_ADDRESS_PRIVATE;
        }
    }
    else if (arg->type_ == ROC_ARGTYPE_IMAGE) {
        return CL_KERNEL_ARG_ADDRESS_GLOBAL;
    }
    //default for all other cases
    return CL_KERNEL_ARG_ADDRESS_PRIVATE;
}

static inline cl_kernel_arg_access_qualifier
GetOclAccessQual(const Kernel::Argument* arg)
{
    if (arg->type_ == ROC_ARGTYPE_IMAGE) {
        switch (arg->access_) {
        case ROC_ACCESS_TYPE_RO:
            return CL_KERNEL_ARG_ACCESS_READ_ONLY;
        case ROC_ACCESS_TYPE_WO:
             return CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
        case ROC_ACCESS_TYPE_RW:
            return CL_KERNEL_ARG_ACCESS_READ_WRITE;
        default:
            return CL_KERNEL_ARG_ACCESS_NONE;
        }
    }
    return CL_KERNEL_ARG_ACCESS_NONE;
}

#if defined(WITH_LIGHTNING_COMPILER)
static inline cl_kernel_arg_type_qualifier
GetOclTypeQual(const KernelArgMD& lcArg)
{
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
    return rv;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

static inline cl_kernel_arg_type_qualifier
GetOclTypeQual(const aclArgData* argInfo)
{
    cl_kernel_arg_type_qualifier rv = CL_KERNEL_ARG_TYPE_NONE;
    if (argInfo->type == ARG_TYPE_POINTER) {
        if (argInfo->arg.pointer.isVolatile) {
            rv |= CL_KERNEL_ARG_TYPE_VOLATILE;
        }
        if (argInfo->arg.pointer.isRestrict) {
            rv |= CL_KERNEL_ARG_TYPE_RESTRICT;
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

void
Kernel::initArguments(const aclArgData* aclArg)
{
    device::Kernel::parameters_t params;

    // Iterate through the arguments and insert into parameterList
    for (size_t offset = 0; aclArg->struct_size != 0; aclArg++) {

        // Initialize HSAIL kernel argument
        Kernel::Argument* arg = new Kernel::Argument;
        arg->name_      = aclArg->argStr;
        arg->typeName_  = aclArg->typeStr;
        arg->size_      = GetKernelArgSize(aclArg);
        arg->type_      = GetKernelArgType(aclArg);
        arg->addrQual_  = GetKernelAddrQual(aclArg);
        arg->dataType_  = GetKernelDataType(aclArg);
        arg->alignment_ = GetKernelArgAlignment(aclArg);
        arg->access_    = GetKernelArgAccessType(aclArg);
        arg->pointeeAlignment_ = GetKernelArgPointeeAlignment(aclArg);

        bool isHidden = arg->type_ == ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_X
            || arg->type_ == ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Y
            || arg->type_ == ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Z
            || arg->type_ == ROC_ARGTYPE_HIDDEN_PRINTF_BUFFER
            || arg->type_ == ROC_ARGTYPE_HIDDEN_DEFAULT_QUEUE
            || arg->type_ == ROC_ARGTYPE_HIDDEN_COMPLETION_ACTION
            || arg->type_ == ROC_ARGTYPE_HIDDEN_NONE;

        arg->index_ = isHidden ? uint(-1) : params.size();
        hsailArgList_.push_back(arg);

        if (isHidden) {
            continue;
        }

        amd::KernelParameterDescriptor desc;
        desc.name_ = arg->name_.c_str();
        desc.type_ = GetOclType(arg);
        desc.addressQualifier_ = GetOclAddrQual(arg);
        desc.accessQualifier_ = GetOclAccessQual(arg);
        desc.typeQualifier_ = GetOclTypeQual(aclArg);
        desc.typeName_ = arg->typeName_.c_str();

        // Make a check if it is local or global
        if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
            desc.size_ = 0;
        }
        else {
            desc.size_ = arg->size_;
        }

        // Make offset alignment to match CPU metadata, since
        // in multidevice config abstraction layer has a single signature
        // and CPU sends the parameters as they are allocated in memory
        size_t  size = desc.size_;
        if (size == 0) {
            // Local memory for CPU
            size = sizeof(cl_mem);
        }
        offset  = amd::alignUp(offset, std::min(size, size_t(16)));
        desc.offset_ = offset;
        offset += amd::alignUp(size, sizeof(uint32_t));

        params.push_back(desc);
    }
    createSignature(params);
}

#if defined(WITH_LIGHTNING_COMPILER)
void
Kernel::initArguments_LC(const KernelMD& kernelMD)
{
    device::Kernel::parameters_t params;

    size_t offset = 0;

    for (size_t i = 0; i < kernelMD.mArgs.size(); ++i) {
        const KernelArgMD& lcArg = kernelMD.mArgs[i];

        // Initialize HSAIL kernel argument
        Kernel::Argument* arg = new Kernel::Argument;
        arg->name_      = lcArg.mName;
        arg->typeName_  = lcArg.mTypeName;
        arg->size_      = lcArg.mSize;
        arg->type_      = GetKernelArgType(lcArg);
        arg->addrQual_  = GetKernelAddrQual(lcArg);
        arg->dataType_  = GetKernelDataType(lcArg);
        arg->alignment_ = GetKernelArgAlignment(lcArg);
        arg->access_    = GetKernelArgAccessType(lcArg);
        arg->pointeeAlignment_ = GetKernelArgPointeeAlignment(lcArg);

        bool isHidden = arg->type_ == ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_X
            || arg->type_ == ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Y
            || arg->type_ == ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Z
            || arg->type_ == ROC_ARGTYPE_HIDDEN_PRINTF_BUFFER
            || arg->type_ == ROC_ARGTYPE_HIDDEN_DEFAULT_QUEUE
            || arg->type_ == ROC_ARGTYPE_HIDDEN_COMPLETION_ACTION
            || arg->type_ == ROC_ARGTYPE_HIDDEN_NONE;

        arg->index_ = isHidden ? uint(-1) : params.size();
        hsailArgList_.push_back(arg);

        if (isHidden) {
            continue;
        }

        // Initialize Device kernel parameters
        amd::KernelParameterDescriptor desc;

        desc.name_ = lcArg.mName.c_str();
        desc.type_ = GetOclType(arg);
        desc.addressQualifier_ = GetOclAddrQual(arg);
        desc.accessQualifier_ = GetOclAccessQual(arg);
        desc.typeQualifier_ = GetOclTypeQual(lcArg);
        desc.typeName_ = lcArg.mTypeName.c_str();

        // Make a check if it is local or global
        if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
            desc.size_ = 0;
        }
        else {
            desc.size_ = arg->size_;
        }

        // Make offset alignment to match CPU metadata, since
        // in multidevice config abstraction layer has a single signature
        // and CPU sends the parameters as they are allocated in memory
        size_t  size = desc.size_;
        if (size == 0) {
            // Local memory for CPU
            size = sizeof(cl_mem);
        }
        offset = (size_t) amd::alignUp(offset, std::min(size, size_t(16)));
        desc.offset_ = offset;
        offset += amd::alignUp(size, sizeof(uint32_t));

        params.push_back(desc);
    }

    createSignature(params);
}
#endif // defined(WITH_LIGHTNING_COMPILER)

Kernel::Kernel(
    std::string name, HSAILProgram* prog,
               const uint64_t& kernelCodeHandle,
               const uint32_t workgroupGroupSegmentByteSize,
               const uint32_t workitemPrivateSegmentByteSize,
               const uint32_t kernargSegmentByteSize,
               const uint32_t kernargSegmentAlignment)
    : device::Kernel(name),
      program_(prog),
      kernelCodeHandle_(kernelCodeHandle),
      workgroupGroupSegmentByteSize_(workgroupGroupSegmentByteSize),
      workitemPrivateSegmentByteSize_(workitemPrivateSegmentByteSize),
      kernargSegmentByteSize_(kernargSegmentByteSize),
      kernargSegmentAlignment_(kernargSegmentAlignment) {}

#if defined(WITH_LIGHTNING_COMPILER)

static const KernelMD* FindKernelMetadata(const CodeObjectMD* programMD,
                                          const std::string& name) {
    for (const KernelMD& kernelMD : programMD->mKernels) {
        if (kernelMD.mName == name) { return &kernelMD; }
    }
    return nullptr;
}

bool Kernel::init_LC()
{
    hsa_agent_t hsaDevice = program_->hsaDevice();

    // Pull out metadata from the ELF
    const CodeObjectMD* programMD = program_->metadata();
    assert(programMD != nullptr);

    const KernelMD* kernelMD = FindKernelMetadata(programMD, name());
    if (kernelMD == nullptr) {
        return false;
    }
    initArguments_LC(*kernelMD);

    //Set the workgroup information for the kernel
    memset(&workGroupInfo_, 0, sizeof(workGroupInfo_));
    workGroupInfo_.availableLDSSize_ = program_->dev().info().localMemSizePerCU_;
    assert(workGroupInfo_.availableLDSSize_ > 0);
    workGroupInfo_.availableSGPRs_ = 0;
    workGroupInfo_.availableVGPRs_ = 0;

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
        workGroupInfo_.compileVecTypeHint_ =
            kernelMD->mAttrs.mVecTypeHint.c_str();
    }

    uint32_t wavefront_size = 0;
    if (hsa_agent_get_info(
            program_->hsaDevice(),
            HSA_AGENT_INFO_WAVEFRONT_SIZE,
            &wavefront_size) != HSA_STATUS_SUCCESS) {
        return false;
    }
    assert(wavefront_size > 0);

    workGroupInfo_.privateMemSize_ = workitemPrivateSegmentByteSize_;
    workGroupInfo_.localMemSize_ = workgroupGroupSegmentByteSize_;
    workGroupInfo_.usedLDSSize_ = workgroupGroupSegmentByteSize_;

    workGroupInfo_.preferredSizeMultiple_ = wavefront_size;

    /// TODO: Are there any other fields that are getting queried from akc?
    /// If so, code properties metadata should be used instead.
    workGroupInfo_.usedSGPRs_ = kernelMD->mCodeProps.mWavefrontNumSGPRs;
    workGroupInfo_.usedVGPRs_ = kernelMD->mCodeProps.mWorkitemNumVGPRs;

    workGroupInfo_.usedStackSize_ = 0;

    workGroupInfo_.wavefrontPerSIMD_ =
        program_->dev().info().maxWorkItemSizes_[0] / wavefront_size;

    workGroupInfo_.wavefrontSize_ = wavefront_size;

    if (workGroupInfo_.compileSize_[0] != 0) {
        workGroupInfo_.size_ =
        workGroupInfo_.compileSize_[0] *
        workGroupInfo_.compileSize_[1] *
        workGroupInfo_.compileSize_[2];
    }
    else {
        workGroupInfo_.size_ = program_->dev().info().maxWorkGroupSize_;
    }

    initPrintf_LC(programMD->mPrintf);

    return true;
}
#endif // defined(WITH_LIGHTNING_COMPILER)

bool Kernel::init()
{
#if defined(WITH_LIGHTNING_COMPILER)
    return init_LC();
#else // !defined(WITH_LIGHTNING_COMPILER)
    acl_error errorCode;
    //compile kernel down to ISA
    hsa_agent_t hsaDevice = program_->hsaDevice();
    // Pull out metadata from the ELF
    size_t sizeOfArgList;
    aclCompiler* compileHandle = program_->dev().compiler();
    std::string openClKernelName("&__OpenCL_" + name() + "_kernel");
    errorCode = g_complibApi._aclQueryInfo(compileHandle,
        program_->binaryElf(),
        RT_ARGUMENT_ARRAY,
        openClKernelName.c_str(),
        nullptr,
        &sizeOfArgList);
    if (errorCode != ACL_SUCCESS) {
        return false;
    }
    std::unique_ptr<char[]> argList(new char[sizeOfArgList]);
    errorCode = g_complibApi._aclQueryInfo(compileHandle,
        program_->binaryElf(),
        RT_ARGUMENT_ARRAY,
        openClKernelName.c_str(),
        argList.get(),
        &sizeOfArgList);
    if (errorCode != ACL_SUCCESS) {
        return false;
    }

    //Set the argList
    initArguments((const aclArgData *) argList.get());

    //Set the workgroup information for the kernel
    memset(&workGroupInfo_, 0, sizeof(workGroupInfo_));
    workGroupInfo_.availableLDSSize_ = program_->dev().info().localMemSizePerCU_;
    assert(workGroupInfo_.availableLDSSize_ > 0);
    workGroupInfo_.availableSGPRs_ = 0;
    workGroupInfo_.availableVGPRs_ = 0;
    size_t sizeOfWorkGroupSize;
    errorCode = g_complibApi._aclQueryInfo(compileHandle,
        program_->binaryElf(),
        RT_WORK_GROUP_SIZE,
        openClKernelName.c_str(),
        nullptr,
        &sizeOfWorkGroupSize);
    if (errorCode != ACL_SUCCESS) {
        return false;
    }
    errorCode = g_complibApi._aclQueryInfo(compileHandle,
        program_->binaryElf(),
        RT_WORK_GROUP_SIZE,
        openClKernelName.c_str(),
        workGroupInfo_.compileSize_,
        &sizeOfWorkGroupSize);
    if (errorCode != ACL_SUCCESS) {
        return false;
    }

    uint32_t wavefront_size = 0;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(
        program_->hsaDevice(), HSA_AGENT_INFO_WAVEFRONT_SIZE,
        &wavefront_size)) {
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
    if (Device::loaderQueryHostAddress(
            reinterpret_cast<const void*>(kernelCodeHandle_), &kernelHostPtr
            ) == HSA_STATUS_SUCCESS) {
        auto akc = reinterpret_cast<const amd_kernel_code_t*>(kernelHostPtr);
        workGroupInfo_.usedSGPRs_ = akc->wavefront_sgpr_count;
        workGroupInfo_.usedVGPRs_ = akc->workitem_vgpr_count;
    }
    else {
        workGroupInfo_.usedSGPRs_ = 0;
        workGroupInfo_.usedVGPRs_ = 0;
    }

    workGroupInfo_.usedStackSize_ = 0;
    workGroupInfo_.wavefrontPerSIMD_ =
        program_->dev().info().maxWorkItemSizes_[0] / wavefront_size;
    workGroupInfo_.wavefrontSize_ = wavefront_size;
    if (workGroupInfo_.compileSize_[0] != 0) {
        workGroupInfo_.size_ =
        workGroupInfo_.compileSize_[0] *
        workGroupInfo_.compileSize_[1] *
        workGroupInfo_.compileSize_[2];
    }
    else {
        workGroupInfo_.size_ = program_->dev().info().maxWorkGroupSize_;
    }

    // Pull out printf metadata from the ELF
    size_t sizeOfPrintfList;
    errorCode = g_complibApi._aclQueryInfo(compileHandle, program_->binaryElf(), RT_GPU_PRINTF_ARRAY,
                                            openClKernelName.c_str(), nullptr, &sizeOfPrintfList);
    if (errorCode != ACL_SUCCESS){
    return false;
    }

    // Make sure kernel has any printf info
  if (0 != sizeOfPrintfList) {
    std::unique_ptr<char[]> aclPrintfList(new char[sizeOfPrintfList]);
    if (!aclPrintfList) {
      return false;
    }
    errorCode = g_complibApi._aclQueryInfo(
        compileHandle, program_->binaryElf(), RT_GPU_PRINTF_ARRAY,
        openClKernelName.c_str(), aclPrintfList.get(), &sizeOfPrintfList);
    if (errorCode != ACL_SUCCESS) {
      return false;
    }

    // Set the Printf List
    initPrintf(reinterpret_cast<aclPrintfFmt*>(aclPrintfList.get()));
  }
  return true;
#endif // !defined(WITH_LIGHTNING_COMPILER)
}

#if defined(WITH_LIGHTNING_COMPILER)
void
Kernel::initPrintf_LC(const std::vector<std::string>& printfInfoStrings)
{
    for (auto str : printfInfoStrings) {
        std::vector<std::string> tokens;

        size_t end, pos = 0;
        do {
            end = str.find_first_of(':', pos);
            tokens.push_back(str.substr(pos, end-pos));
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
        if (tokens.size() < end + 1/*last token is the fmtString*/) {
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
                switch (fmt[pos+1]) {
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
                    if (fmt[pos+2] == '2') {
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
#endif // defined(WITH_LIGHTNING_COMPILER)

void
Kernel::initPrintf(const aclPrintfFmt* aclPrintf)
{
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
                switch (pfmt[pos+1]) {
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
                    if (pfmt[pos+2] == '2') {
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


Kernel::~Kernel()
{
    while (!hsailArgList_.empty()) {
        Argument* kernelArgPointer = hsailArgList_.back();
        delete kernelArgPointer;
        hsailArgList_.pop_back();
    }
}

}  // namespace roc
#endif  // WITHOUT_HSA_BACKEND
