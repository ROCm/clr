//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include <memory>
#include "acl.h"
#include "rocprogram.hpp"
#include "top.hpp"
#include "rocprintf.hpp"

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

#define MAX_INFO_STRING_LEN 0x40
enum HSAIL_ADDRESS_QUALIFIER{
HSAIL_ADDRESS_ERROR=0,
HSAIL_ADDRESS_GLOBAL,
HSAIL_ADDRESS_LOCAL,
HSAIL_MAX_ADDRESS_QUALIFIERS
} ;

enum HSAIL_ARG_TYPE{
HSAIL_ARGTYPE_ERROR=0,
HSAIL_ARGTYPE_POINTER,
HSAIL_ARGTYPE_VALUE,
HSAIL_ARGTYPE_IMAGE,
HSAIL_ARGTYPE_SAMPLER,
HSAIL_ARGMAX_ARG_TYPES
};

enum HSAIL_DATA_TYPE{
HSAIL_DATATYPE_ERROR=0,
HSAIL_DATATYPE_B1,
HSAIL_DATATYPE_B8,
HSAIL_DATATYPE_B16,
HSAIL_DATATYPE_B32,
HSAIL_DATATYPE_B64,
HSAIL_DATATYPE_S8,
HSAIL_DATATYPE_S16,
HSAIL_DATATYPE_S32,
HSAIL_DATATYPE_S64,
HSAIL_DATATYPE_U8,
HSAIL_DATATYPE_U16,
HSAIL_DATATYPE_U32,
HSAIL_DATATYPE_U64,
HSAIL_DATATYPE_F16,
HSAIL_DATATYPE_F32,
HSAIL_DATATYPE_F64,
HSAIL_DATATYPE_STRUCT,
HSAIL_DATATYPE_OPAQUE,
HSAIL_DATATYPE_MAX_TYPES
};

enum HSAIL_ACCESS_TYPE {
    HSAIL_ACCESS_TYPE_NONE = 0,
    HSAIL_ACCESS_TYPE_RO,
    HSAIL_ACCESS_TYPE_WO,
    HSAIL_ACCESS_TYPE_RW
};

struct HsailKernelArg
{
    std::string name_;          //!< Argument's name
    std::string typeName_;      //!< Argument's type name
    uint        size_;          //!< Size in bytes
    uint        offset_;        //!< Argument's offset
    uint        alignment_;     //!< Argument's alignment
    HSAIL_ARG_TYPE type_;       //!< Type of the argument
    HSAIL_ADDRESS_QUALIFIER addrQual_;  //!< Address qualifier of the argument
    HSAIL_DATA_TYPE dataType_;  //!< The type of data
    uint        numElem_;       //!< Number of elements
    HSAIL_ACCESS_TYPE access_;  //!< Access type for the argument
};

class KernelArg
{
public:
    KernelArg(aclArgData* argInfo);
    //! Return type of the argument 
    clk_value_type_t amdoclType();
    //! Global, local etc - returns amdocl types
    clk_address_space_t amdoclAddrQual();
    //! Global,localetc - returns opencl type 
    cl_kernel_arg_address_qualifier oclAddrQual();
    //! read , write etc - returns amdocl type
    clk_arg_qualifier_t amdoclAccessQual();
    //! read , write etc - returns opencl type type
    cl_kernel_arg_access_qualifier oclAccessQual();
    //! const,volatile,restrict etc - returns opencl type type
    cl_kernel_arg_type_qualifier oclTypeQual();

    //! Name of the argument
    std::string& name();
    //! Name of the argument
    std::string& typeName();
    //! reflection 
    std::string reflection(){ return name(); };
    //! Returns the size of the argument 
    int size();
    //! returns the offset
    int offset();

    void setOffset();

private:
    aclArgData* argInfo_;
    int offset_;
    std::string name_;
    std::string typeName_;
};

class Kernel : public device::Kernel
{
public:
    Kernel(std::string name,
        HSAILProgram* prog,
        const uint64_t &kernelCodeHandle,
        const uint32_t workgroupGroupSegmentByteSize,
        const uint32_t workitemPrivateSegmentByteSize,
        const uint32_t kernargSegmentByteSize,
        const uint32_t kernargSegmentAlignment,
        uint extraArgsNum);

    const uint64_t& KernelCodeHandle() {
        return kernelCodeHandle_;
    }

    const uint32_t WorkgroupGroupSegmentByteSize() const {
      return workgroupGroupSegmentByteSize_;
    }

    const uint32_t workitemPrivateSegmentByteSize() const {
      return workitemPrivateSegmentByteSize_;
    }

    const uint64_t KernargSegmentByteSize() const {
      return kernargSegmentByteSize_;
    }

    const uint8_t KernargSegmentAlignment() const {
      return kernargSegmentAlignment_;
    }

    ~Kernel();

    //! Initializes the metadata required for this kernel
    bool init();

#if defined(WITH_LIGHTNING_COMPILER)
    //! Initializes the metadata required for this kernel
    bool init_LC();
#endif // defined(WITH_LIGHTNING_COMPILER)

    const HSAILProgram* program() {
        return static_cast<const HSAILProgram*>(program_);
    }

    //! Returns a pointer to the hsail argument at the specified index
    HsailKernelArg* hsailArgAt(size_t index) const {
        return hsailArgList_[index];
    }

    //! Max number of possible extra (hidden) kernel arguments
    static const uint MaxExtraArgumentsNum = 6;

    uint extraArgumentsNum() const { return extraArgumentsNum_; }

    //! Return printf info array
    const std::vector<PrintfInfo>& printfInfo() const {return printf_;}

private:
    //! Populates hsailArgList_
    void initArgList(const aclArgData* aclArg);

    //! Initializes Hsail Argument metadata and info ;
    void initHsailArgs(const aclArgData* aclArg);

#if defined(WITH_LIGHTNING_COMPILER)
    //! Initializes Hsail Argument metadata and info for LC
    void initArgsParams( const amd::hsa::code::KernelArg::Metadata* lcArg, size_t* kOffset,
                         device::Kernel::parameters_t& params, size_t* pOffset );
#endif // defined(WITH_LIGHTNING_COMPILER)

    //! Initializes HSAIL Printf metadata and info
    void initPrintf(const aclPrintfFmt* aclPrintf);

    HSAILProgram *program_; //!< The roc::HSAILProgram context
    std::vector<HsailKernelArg*> hsailArgList_; //!< Vector list of HSAIL Arguments
    std::string compileOptions_; //!< compile used for finalizing this kernel
    uint64_t kernelCodeHandle_; //!< Kernel code handle (aka amd_kernel_code_t)
    const uint32_t workgroupGroupSegmentByteSize_;
    const uint32_t workitemPrivateSegmentByteSize_;
    const uint32_t kernargSegmentByteSize_;
    const uint32_t kernargSegmentAlignment_;
    size_t kernelDirectiveOffset_;
    const uint extraArgumentsNum_; // Number of arguments in Kernenv
    std::vector<PrintfInfo> printf_;
};

} // namespace roc

#endif // WITHOUT_HSA_BACKEND


