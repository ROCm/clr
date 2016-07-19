//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//

#include "rockernel.hpp"
#include "SCHSAInterface.h"
#include "amd_hsa_kernel_code.h"

#include <algorithm>

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

inline static HSAIL_ARG_TYPE
GetHSAILArgType(const aclArgData* argInfo)
{
    switch (argInfo->type) {
        case ARG_TYPE_POINTER:
            return HSAIL_ARGTYPE_POINTER;
        case ARG_TYPE_VALUE:
            return HSAIL_ARGTYPE_VALUE;
        case ARG_TYPE_IMAGE:
            return HSAIL_ARGTYPE_IMAGE;
        case ARG_TYPE_SAMPLER:
            return HSAIL_ARGTYPE_SAMPLER;
        case ARG_TYPE_ERROR:
        default:
            return HSAIL_ARGTYPE_ERROR;
    }
}

inline static size_t
GetHSAILArgAlignment(const aclArgData* argInfo)
{
    switch (argInfo->type) {
        case ARG_TYPE_POINTER:
            return argInfo->arg.pointer.align;
        default:
            return 1;
    }
}

inline static HSAIL_ACCESS_TYPE
GetHSAILArgAccessType(const aclArgData* argInfo)
{
    if (argInfo->type == ARG_TYPE_POINTER) {
        switch (argInfo->arg.pointer.type) {
        case ACCESS_TYPE_RO:
            return HSAIL_ACCESS_TYPE_RO;
        case ACCESS_TYPE_WO:
            return HSAIL_ACCESS_TYPE_WO;
        case ACCESS_TYPE_RW:
        default:
            return HSAIL_ACCESS_TYPE_RW;
        }
    }
    return HSAIL_ACCESS_TYPE_NONE;
}

inline static HSAIL_ADDRESS_QUALIFIER
GetHSAILAddrQual(const aclArgData* argInfo)
{
    if (argInfo->type == ARG_TYPE_POINTER) {
        switch (argInfo->arg.pointer.memory) {
            case PTR_MT_CONSTANT_EMU:
            case PTR_MT_CONSTANT:
            case PTR_MT_UAV:
            case PTR_MT_GLOBAL:
                return HSAIL_ADDRESS_GLOBAL;
            case PTR_MT_LDS_EMU:
            case PTR_MT_LDS:
                return HSAIL_ADDRESS_LOCAL;
            case PTR_MT_ERROR:
            default:
                LogError("Unsupported address type");
                return HSAIL_ADDRESS_ERROR;
        }
    }
    else if ((argInfo->type == ARG_TYPE_IMAGE) ||
             (argInfo->type == ARG_TYPE_SAMPLER)) {
        return HSAIL_ADDRESS_GLOBAL;
    }
    return HSAIL_ADDRESS_ERROR;
}

/* f16 returns f32 - workaround due to comp lib */
inline static HSAIL_DATA_TYPE
GetHSAILDataType(const aclArgData* argInfo)
{
    aclArgDataType dataType;

    if (argInfo->type == ARG_TYPE_POINTER) {
        dataType = argInfo->arg.pointer.data;
    }
    else if (argInfo->type == ARG_TYPE_VALUE) {
        dataType = argInfo->arg.value.data;
    }
    else {
        return HSAIL_DATATYPE_ERROR;
    }
    switch (dataType) {
        case DATATYPE_i1:
            return HSAIL_DATATYPE_B1;
        case DATATYPE_i8:
            return HSAIL_DATATYPE_S8;
        case DATATYPE_i16:
            return HSAIL_DATATYPE_S16;
        case DATATYPE_i32:
            return HSAIL_DATATYPE_S32;
        case DATATYPE_i64:
            return HSAIL_DATATYPE_S64;
        case DATATYPE_u8:
            return HSAIL_DATATYPE_U8;
        case DATATYPE_u16:
            return HSAIL_DATATYPE_U16;
        case DATATYPE_u32:
            return HSAIL_DATATYPE_U32;
        case DATATYPE_u64:
            return HSAIL_DATATYPE_U64;
        case DATATYPE_f16:
            return HSAIL_DATATYPE_F32;
        case DATATYPE_f32:
            return HSAIL_DATATYPE_F32;
        case DATATYPE_f64:
            return HSAIL_DATATYPE_F64;
        case DATATYPE_struct:
            return HSAIL_DATATYPE_STRUCT;
        case DATATYPE_opaque:
            return HSAIL_DATATYPE_OPAQUE;
        case DATATYPE_ERROR:
        default:
            return HSAIL_DATATYPE_ERROR;
    }
}

// returns size in number of bytes
inline static int
GetHSAILArgSize(const aclArgData *argInfo)
{
    switch (argInfo->type) {
        case ARG_TYPE_VALUE:
            switch (GetHSAILDataType(argInfo)) {
                case HSAIL_DATATYPE_B1:
                    return 1;
                case HSAIL_DATATYPE_B8:
                case HSAIL_DATATYPE_S8:
                case HSAIL_DATATYPE_U8:
                    return 1;
                case HSAIL_DATATYPE_B16:
                case HSAIL_DATATYPE_U16:
                case HSAIL_DATATYPE_S16:
                case HSAIL_DATATYPE_F16:
                    return 2;
                case HSAIL_DATATYPE_B32:
                case HSAIL_DATATYPE_U32:
                case HSAIL_DATATYPE_S32:
                case HSAIL_DATATYPE_F32:
                    return 4;
                case HSAIL_DATATYPE_B64:
                case HSAIL_DATATYPE_U64:
                case HSAIL_DATATYPE_S64:
                case HSAIL_DATATYPE_F64:
                    return 8;
                case HSAIL_DATATYPE_STRUCT:
                    return argInfo->arg.value.numElements;
                default:
                    return -1;
            }
        case ARG_TYPE_POINTER:
        case ARG_TYPE_IMAGE:
        case ARG_TYPE_SAMPLER:
            return sizeof(void*);
        default:
            return -1;
    }
}

inline static clk_value_type_t
GetOclType(const aclArgData* argInfo)
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
    if ((argInfo->type == ARG_TYPE_POINTER) || (argInfo->type == ARG_TYPE_IMAGE)) {
        return T_POINTER;
    }
    else if (argInfo->type == ARG_TYPE_VALUE) {
        switch (argInfo->arg.value.data) {
            case DATATYPE_i8:
            case DATATYPE_u8:
                sizeType = 0;
                break;
            case DATATYPE_i16:
            case DATATYPE_u16:
                sizeType = 1;
                break;
            case DATATYPE_i32:
            case DATATYPE_u32:
                sizeType = 2;
                break;
            case DATATYPE_i64:
            case DATATYPE_u64:
                sizeType = 3;
                break;
            case DATATYPE_f16:
            case DATATYPE_f32:
                sizeType = 4;
                break;
            case DATATYPE_f64:
                sizeType = 5;
                break;
            default:
                return T_VOID;
        }
        switch (argInfo->arg.value.numElements) {
            case 1: return ClkValueMapType[sizeType][0];
            case 2: return ClkValueMapType[sizeType][1];
            case 3: return ClkValueMapType[sizeType][2];
            case 4: return ClkValueMapType[sizeType][3];
            case 8: return ClkValueMapType[sizeType][4];
            case 16: return ClkValueMapType[sizeType][5];
            default: return T_VOID;
        }
    }
    else if (argInfo->type == ARG_TYPE_SAMPLER) {
        return T_SAMPLER;
    }
    else {
        return T_VOID;
    }
}

inline static cl_kernel_arg_address_qualifier
GetOclAddrQual(const aclArgData* argInfo)
{
    if (argInfo->type == ARG_TYPE_POINTER) {
        switch (argInfo->arg.pointer.memory) {
        case PTR_MT_UAV:
        case PTR_MT_GLOBAL:
            return CL_KERNEL_ARG_ADDRESS_GLOBAL;
        case PTR_MT_CONSTANT:
        case PTR_MT_UAV_CONSTANT:
        case PTR_MT_CONSTANT_EMU:
            return CL_KERNEL_ARG_ADDRESS_CONSTANT;
        case PTR_MT_LDS_EMU:
        case PTR_MT_LDS:
            return CL_KERNEL_ARG_ADDRESS_LOCAL;
        default:
            return CL_KERNEL_ARG_ADDRESS_PRIVATE;
        }
    }
    else if (argInfo->type == ARG_TYPE_IMAGE) {
        return CL_KERNEL_ARG_ADDRESS_GLOBAL;
    }
    //default for all other cases
    return CL_KERNEL_ARG_ADDRESS_PRIVATE;
}

inline static cl_kernel_arg_access_qualifier
GetOclAccessQual(const aclArgData* argInfo)
{
    if (argInfo->type == ARG_TYPE_IMAGE) {
        switch (argInfo->arg.image.type) {
        case ACCESS_TYPE_RO:
            return CL_KERNEL_ARG_ACCESS_READ_ONLY;
        case ACCESS_TYPE_WO:
             return CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
        case ACCESS_TYPE_RW:
            return CL_KERNEL_ARG_ACCESS_READ_WRITE;
        default:
            return CL_KERNEL_ARG_ACCESS_NONE;
        }
    }
    return CL_KERNEL_ARG_ACCESS_NONE;
}

inline static cl_kernel_arg_type_qualifier
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

static int
GetOclSize(const aclArgData* argInfo)
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

KernelArg::KernelArg(aclArgData *argInfo) {
    argInfo_ = argInfo;
    name_ = argInfo_->argStr;
    typeName_ = argInfo->typeStr;
}

int KernelArg::size() {
  switch (argInfo_->type) {
    case ARG_TYPE_POINTER: {
      return sizeof(void *);
    }
    case ARG_TYPE_VALUE: {
      switch (argInfo_->arg.value.data) {
        case DATATYPE_ERROR: {
          return -1;
        }
        case DATATYPE_i8:
        case DATATYPE_u8:
        case DATATYPE_struct: {
          return 1 * argInfo_->arg.value.numElements;
        }
        case DATATYPE_u16:
        case DATATYPE_i16:
        case DATATYPE_f16: {
          return 2 * argInfo_->arg.value.numElements;
        }
        case DATATYPE_u32:
        case DATATYPE_i32:
        case DATATYPE_f32: {
          return 4 * argInfo_->arg.value.numElements;
        }
        case DATATYPE_i64:
        case DATATYPE_u64:
        case DATATYPE_f64: {
          return 8 * argInfo_->arg.value.numElements;
        }
        default:
          return -1;
      }
    }
    case ARG_TYPE_IMAGE: {
        return sizeof(cl_mem);
    }
    case ARG_TYPE_SAMPLER: {
        return sizeof(cl_sampler);
    }
    default:
      return -1;
  }
}

std::string& KernelArg::name() {
    return name_;
}

std::string& KernelArg::typeName()
{
    return typeName_;
}

void
Kernel::initArgList(const aclArgData* aclArg)
{
    // Initialize the hsail argument list too
    initHsailArgs(aclArg);

    // Iterate through the arguments and insert into parameterList
    device::Kernel::parameters_t params;
    amd::KernelParameterDescriptor desc;
    size_t offset = 0;

    // Reserved arguments for HSAIL launch
    aclArg += MaxExtraArgumentsNum;
    for (uint i = 0; aclArg->struct_size != 0; i++, aclArg++) {
        desc.name_ = hsailArgList_[i]->name_.c_str();
        desc.type_ = GetOclType(aclArg);
        desc.addressQualifier_ = GetOclAddrQual(aclArg);
        desc.accessQualifier_ = GetOclAccessQual(aclArg);
        desc.typeQualifier_ = GetOclTypeQual(aclArg);
        desc.typeName_ = hsailArgList_[i]->typeName_.c_str();

        // Make a check if it is local or global
        if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
            desc.size_ = 0;
        }
        else {
            desc.size_ = GetOclSize(aclArg);
        }

        // Make offset alignment to match CPU metadata, since
        // in multidevice config abstraction layer has a single signature
        // and CPU sends the paramaters as they are allocated in memory
        size_t  size = desc.size_;
        if (size == 0) {
            // Local memory for CPU
            size = sizeof(cl_mem);
        }
        offset  = amd::alignUp(offset, std::min(size, size_t(16)));
        desc.offset_    = offset;
        offset          += amd::alignUp(size, sizeof(uint32_t));
        params.push_back(desc);
    }
    createSignature(params);
}

void
Kernel::initHsailArgs(const aclArgData* aclArg)
{
    int offset = 0;

    // Reserved arguments for HSAIL launch
    aclArg += MaxExtraArgumentsNum;

    // Iterate through the each kernel argument
    for (; aclArg->struct_size != 0; aclArg++) {
        HsailKernelArg* arg = new HsailKernelArg;
        // Initialize HSAIL kernel argument
        arg->name_      = aclArg->argStr;
        arg->typeName_  = aclArg->typeStr;
        arg->size_      = GetHSAILArgSize(aclArg);
        arg->offset_    = offset;
        arg->type_      = GetHSAILArgType(aclArg);
        arg->addrQual_  = GetHSAILAddrQual(aclArg);
        arg->dataType_  = GetHSAILDataType(aclArg);
        // If vector of args we add additional arguments to flatten it out
        arg->numElem_   = ((aclArg->type == ARG_TYPE_VALUE) &&
             (aclArg->arg.value.data != DATATYPE_struct)) ?
             aclArg->arg.value.numElements : 1;
        arg->alignment_ = GetHSAILArgAlignment(aclArg);
        arg->access_    = GetHSAILArgAccessType(aclArg);
        offset += GetHSAILArgSize(aclArg);
        hsailArgList_.push_back(arg);
    }
}

Kernel::Kernel(std::string name, HSAILProgram* prog,
               const uint64_t& kernelCodeHandle,
               const uint32_t workgroupGroupSegmentByteSize,
               const uint32_t workitemPrivateSegmentByteSize,
               const uint32_t kernargSegmentByteSize,
               const uint32_t kernargSegmentAlignment,
               uint extraArgsNum)
    : device::Kernel(name),
      program_(prog),
      kernelCodeHandle_(kernelCodeHandle),
      workgroupGroupSegmentByteSize_(workgroupGroupSegmentByteSize),
      workitemPrivateSegmentByteSize_(workitemPrivateSegmentByteSize),
      kernargSegmentByteSize_(kernargSegmentByteSize),
      kernargSegmentAlignment_(kernargSegmentAlignment),
      extraArgumentsNum_(extraArgsNum) {}

bool Kernel::init(){
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
        NULL,
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
    initArgList((const aclArgData *) argList.get());
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
        NULL,
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
    workGroupInfo_.usedSGPRs_ = 0;
    workGroupInfo_.usedStackSize_ = 0;
    workGroupInfo_.usedVGPRs_ = 0;
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
                                            openClKernelName.c_str(), NULL, &sizeOfPrintfList);
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
}

void Kernel::initPrintf(const aclPrintfFmt* aclPrintf) {
  PrintfInfo info;
  uint index = 0;
  for (; aclPrintf->struct_size != 0; aclPrintf++) {
    index = aclPrintf->ID;
    if (printf_.size() <= index) {
      printf_.resize(index + 1);
    }
    std::string pfmt = aclPrintf->fmtStr;
    size_t pos = 0;
    for (size_t i = 0; i < pfmt.size(); ++i) {
      char symbol = pfmt[pos++];
      if (symbol == '\\') {
        // Rest of the C escape sequences (e.g. \') are handled correctly
        // by the MDParser, we are not sure exactly how!
        switch (pfmt[pos]) {
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
            if (pfmt[++pos] == '2') {
              pos++;
              i++;
              symbol = '\72';
            }
            break;
          default:
            break;
        }
      }
      info.fmtString_.push_back(symbol);
    }
    info.fmtString_ += "\n";
    uint32_t* tmp_ptr = const_cast<uint32_t*>(aclPrintf->argSizes);
    for (uint i = 0; i < aclPrintf->numSizes; i++, tmp_ptr++) {
      info.arguments_.push_back(*tmp_ptr);
    }
    printf_[index] = info;
    info.arguments_.clear();
  }
}


Kernel::~Kernel() {
    while (!hsailArgList_.empty()) {
        HsailKernelArg* kernelArgPointer = hsailArgList_.back();
        delete kernelArgPointer;
        hsailArgList_.pop_back();
    }
}

}  // namespace roc
#endif  // WITHOUT_HSA_BACKEND
