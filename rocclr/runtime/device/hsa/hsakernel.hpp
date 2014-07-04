//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef HSAKERNEL_HPP_
#define HSAKERNEL_HPP_

#include "acl.h"
#include "device/hsa/hsaprogram.hpp"
#include "newcore.h"
#include "top.hpp"

#ifndef WITHOUT_FSA_BACKEND

namespace oclhsa {

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
    // Global offsets located in the first 3 elements
    static const uint ExtraArguments = 3;

    Kernel(std::string name,
        FSAILProgram* prog,
        HsaBrig* brig,
        std::string compileOptions);

    ~Kernel();

    //! Initializes the metadata required for this kernel
    bool init();

    const FSAILProgram* program() {
        return static_cast<const FSAILProgram*>(program_);
    }

    //! Returns the AqlKernel associated with this Kernel
    const HsaKernelCode* kernelCode() { return
        static_cast<const HsaKernelCode*>(kernelCode_);
    }

    //! Returns the BRIG that was used to compile this kernel
    const HsaBrig* brig() {
        return static_cast<const HsaBrig*>(brig_);
    }

    //!returns a pointer to the hsail argument at the specified index
    HsailKernelArg* hsailArgAt(size_t index) {
        return hsailArgList_[index];
    }

private:
    //! Populates hsailArgList_
    void initArgList(const aclArgData* aclArg);

    //! Initializes Hsail Argument metadata and info ;
    void initHsailArgs(const aclArgData* aclArg);

    FSAILProgram *program_; //!< The oclhsa::FSAILProgram context
    std::vector<HsailKernelArg*> hsailArgList_; //!< Vector list of HSAIL Arguments
    std::string compileOptions_; //!< compile used for finalizing this kernel
    HsaBrig* brig_; //!< The brig used to generate ISA for this kernel
    HsaKernelCode* kernelCode_; //!< AQL kernel code for this kernel
    HsaKernelDebug* debugInfo_; //!< Dwarf info for this kernel
};

} // namespace oclhsa

#endif // WITHOUT_FSA_BACKEND

#endif // HSAKERNEL_HPP_

