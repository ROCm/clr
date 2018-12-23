//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "include/aclTypes.h"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "devwavelimiter.hpp"

#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
namespace llvm {
  namespace AMDGPU {
    namespace HSAMD {
      namespace Kernel {
        struct Metadata;
}}}}
typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;

//! Runtime handle structure for device enqueue
struct RuntimeHandle {
  uint64_t kernel_handle;             //!< Pointer to amd_kernel_code_s or kernel_descriptor_t
  uint32_t private_segment_size;      //!< From PRIVATE_SEGMENT_FIXED_SIZE
  uint32_t group_segment_size;        //!< From GROUP_SEGMENT_FIXED_SIZE
};

#if defined(USE_COMGR_LIBRARY)
#include "llvm/Support/AMDGPUMetadata.h"
typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;

using llvm::AMDGPU::HSAMD::AccessQualifier;
using llvm::AMDGPU::HSAMD::AddressSpaceQualifier;
using llvm::AMDGPU::HSAMD::ValueKind;
using llvm::AMDGPU::HSAMD::ValueType;

enum class ArgField : uint8_t {
  Name          = 0,
  TypeName      = 1,
  Size          = 2,
  Align         = 3,
  ValueKind     = 4,
  ValueType     = 5,
  PointeeAlign  = 6,
  AddrSpaceQual = 7,
  AccQual       = 8,
  ActualAccQual = 9,
  IsConst       = 10,
  IsRestrict    = 11,
  IsVolatile    = 12,
  IsPipe        = 13
};

enum class AttrField : uint8_t {
  ReqdWorkGroupSize  = 0,
  WorkGroupSizeHint = 1,
  VecTypeHint       = 2,
  RuntimeHandle     = 3
};

enum class CodePropField : uint8_t {
  KernargSegmentSize      = 0,
  GroupSegmentFixedSize   = 1,
  PrivateSegmentFixedSize = 2,
  KernargSegmentAlign     = 3,
  WavefrontSize           = 4,
  NumSGPRs                = 5,
  NumVGPRs                = 6,
  MaxFlatWorkGroupSize    = 7,
  IsDynamicCallStack      = 8,
  IsXNACKEnabled          = 9,
  NumSpilledSGPRs         = 10,
  NumSpilledVGPRs         = 11
};


static const std::map<std::string,ArgField> ArgFieldMap =
{
  {"Name",          ArgField::Name},
  {"TypeName",      ArgField::TypeName},
  {"Size",          ArgField::Size},
  {"Align",         ArgField::Align},
  {"ValueKind",     ArgField::ValueKind},
  {"ValueType",     ArgField::ValueType},
  {"PointeeAlign",  ArgField::PointeeAlign},
  {"AddrSpaceQual", ArgField::AddrSpaceQual},
  {"AccQual",       ArgField::AccQual},
  {"ActualAccQual", ArgField::ActualAccQual},
  {"IsConst",       ArgField::IsConst},
  {"IsRestrict",    ArgField::IsRestrict},
  {"IsVolatile",    ArgField::IsVolatile},
  {"IsPipe",        ArgField::IsPipe}
};

static const std::map<std::string,ValueKind> ArgValueKind =
{
  {"ByValue",                 ValueKind::ByValue},
  {"GlobalBuffer",            ValueKind::GlobalBuffer},
  {"DynamicSharedPointer",    ValueKind::DynamicSharedPointer},
  {"Sampler",                 ValueKind::Sampler},
  {"Image",                   ValueKind::Image},
  {"Pipe",                    ValueKind::Pipe},
  {"Queue",                   ValueKind::Queue},
  {"HiddenGlobalOffsetX",     ValueKind::HiddenGlobalOffsetX},
  {"HiddenGlobalOffsetY",     ValueKind::HiddenGlobalOffsetY},
  {"HiddenGlobalOffsetZ",     ValueKind::HiddenGlobalOffsetZ},
  {"HiddenNone",              ValueKind::HiddenNone},
  {"HiddenPrintfBuffer",      ValueKind::HiddenPrintfBuffer},
  {"HiddenDefaultQueue",      ValueKind::HiddenDefaultQueue},
  {"HiddenCompletionAction",  ValueKind::HiddenCompletionAction}
};

static const std::map<std::string,ValueType> ArgValueType =
{
  {"Struct",  ValueType::Struct},
  {"I8",      ValueType::I8},
  {"U8",      ValueType::U8},
  {"I16",     ValueType::I16},
  {"U16",     ValueType::U16},
  {"F16",     ValueType::F16},
  {"I32",     ValueType::I32},
  {"U32",     ValueType::U32},
  {"F32",     ValueType::F32},
  {"I64",     ValueType::I64},
  {"U64",     ValueType::U64},
  {"F64",     ValueType::F64}
};

static const std::map<std::string,AccessQualifier> ArgAccQual =
{
  {"Default",   AccessQualifier::Default},
  {"ReadOnly",  AccessQualifier::ReadOnly},
  {"WriteOnly", AccessQualifier::WriteOnly},
  {"ReadWrite", AccessQualifier::ReadWrite}
};

static const std::map<std::string,AddressSpaceQualifier> ArgAddrSpaceQual =
{
  {"Private",   AddressSpaceQualifier::Private},
  {"Global",    AddressSpaceQualifier::Global},
  {"Constant",  AddressSpaceQualifier::Constant},
  {"Local",     AddressSpaceQualifier::Local},
  {"Generic",   AddressSpaceQualifier::Generic},
  {"Region",    AddressSpaceQualifier::Region}
};

static const std::map<std::string,AttrField> AttrFieldMap =
{
  {"ReqdWorkGroupSize",   AttrField::ReqdWorkGroupSize},
  {"WorkGroupSizeHint",   AttrField::WorkGroupSizeHint},
  {"VecTypeHint",         AttrField::VecTypeHint},
  {"RuntimeHandle",       AttrField::RuntimeHandle}
};

static const std::map<std::string,CodePropField> CodePropFieldMap =
{
  {"KernargSegmentSize",      CodePropField::KernargSegmentSize},
  {"GroupSegmentFixedSize",   CodePropField::GroupSegmentFixedSize},
  {"PrivateSegmentFixedSize", CodePropField::PrivateSegmentFixedSize},
  {"KernargSegmentAlign",     CodePropField::KernargSegmentAlign},
  {"WavefrontSize",           CodePropField::WavefrontSize},
  {"NumSGPRs",                CodePropField::NumSGPRs},
  {"NumVGPRs",                CodePropField::NumVGPRs},
  {"MaxFlatWorkGroupSize",    CodePropField::MaxFlatWorkGroupSize},
  {"IsDynamicCallStack",      CodePropField::IsDynamicCallStack},
  {"IsXNACKEnabled",          CodePropField::IsXNACKEnabled},
  {"NumSpilledSGPRs",         CodePropField::NumSpilledSGPRs},
  {"NumSpilledVGPRs",         CodePropField::NumSpilledVGPRs}
};
#endif  // defined(USE_COMGR_LIBRARY)
#endif  // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

namespace amd {
  namespace hsa {
    namespace loader {
      class Symbol;
    }  // loader
    namespace code {
      namespace Kernel {
        class Metadata;
      }  // Kernel
    }  // code
  }  // hsa
}  // amd

namespace amd {

class Device;
class KernelSignature;
class NDRange;

struct KernelParameterDescriptor {
  enum {
    Value = 0,
    HiddenNone = 1,
    HiddenGlobalOffsetX = 2,
    HiddenGlobalOffsetY = 3,
    HiddenGlobalOffsetZ = 4,
    HiddenPrintfBuffer = 5,
    HiddenDefaultQueue = 6,
    HiddenCompletionAction = 7,
    MemoryObject = 8,
    ReferenceObject = 9,
    ValueObject = 10,
    ImageObject = 11,
    SamplerObject = 12,
    QueueObject = 13
  };
  clk_value_type_t type_;  //!< The parameter's type
  size_t offset_;          //!< Its offset in the parameter's stack
  size_t size_;            //!< Its size in bytes
  union InfoData {
    struct {
      uint32_t oclObject_ : 4;   //!< OCL object type
      uint32_t readOnly_ : 1;   //!< OCL object is read only, applied to memory only
      uint32_t rawPointer_ : 1;   //!< Arguments have a raw GPU VA
      uint32_t defined_ : 1;   //!< The argument was defined by the app
      uint32_t reserved_ : 1;   //!< reserved
      uint32_t arrayIndex_ : 24;  //!< Index in the objects array or LDS alignment
    };
    uint32_t allValues_;
    InfoData() : allValues_(0) {}
  } info_;

  cl_kernel_arg_address_qualifier addressQualifier_;  //!< Argument's address qualifier
  cl_kernel_arg_access_qualifier accessQualifier_;    //!< Argument's access qualifier
  cl_kernel_arg_type_qualifier typeQualifier_;        //!< Argument's type qualifier

  std::string name_;      //!< The parameter's name in the source
  std::string typeName_;  //!< Argument's type name
};

}

namespace device {

//! Printf info structure
struct PrintfInfo {
  std::string fmtString_;        //!< formated string for printf
  std::vector<uint> arguments_;  //!< passed arguments to the printf() call
};

//! \class DeviceKernel, which will contain the common fields for any device
class Kernel : public amd::HeapObject {
 public:
  typedef std::vector<amd::KernelParameterDescriptor> parameters_t;

  //! \struct The device kernel workgroup info structure
  struct WorkGroupInfo : public amd::EmbeddedObject {
    size_t size_;                     //!< kernel workgroup size
    size_t compileSize_[3];           //!< kernel compiled workgroup size
    cl_ulong localMemSize_;           //!< amount of used local memory
    size_t preferredSizeMultiple_;    //!< preferred multiple for launch
    cl_ulong privateMemSize_;         //!< amount of used private memory
    size_t scratchRegs_;              //!< amount of used scratch registers
    size_t wavefrontPerSIMD_;         //!< number of wavefronts per SIMD
    size_t wavefrontSize_;            //!< number of threads per wavefront
    size_t availableGPRs_;            //!< GPRs available to the program
    size_t usedGPRs_;                 //!< GPRs used by the program
    size_t availableSGPRs_;           //!< SGPRs available to the program
    size_t usedSGPRs_;                //!< SGPRs used by the program
    size_t availableVGPRs_;           //!< VGPRs available to the program
    size_t usedVGPRs_;                //!< VGPRs used by the program
    size_t availableLDSSize_;         //!< available LDS size
    size_t usedLDSSize_;              //!< used LDS size
    size_t availableStackSize_;       //!< available stack size
    size_t usedStackSize_;            //!< used stack size
    size_t compileSizeHint_[3];       //!< kernel compiled workgroup size hint
    std::string compileVecTypeHint_;  //!< kernel compiled vector type hint
    bool uniformWorkGroupSize_;       //!< uniform work group size option
    size_t wavesPerSimdHint_;         //!< waves per simd hit
  };

  //! Default constructor
  Kernel(const amd::Device& dev, const std::string& name);

  //! Default destructor
  virtual ~Kernel();

  //! Returns the kernel info structure
  const WorkGroupInfo* workGroupInfo() const { return &workGroupInfo_; }

  //! Returns the kernel signature
  const amd::KernelSignature& signature() const { return *signature_; }

  //! Returns the kernel name
  const std::string& name() const { return name_; }

  //! Initializes the kernel parameters for the abstraction layer
  bool createSignature(
    const parameters_t& params, uint32_t numParameters,
    uint32_t version);

  void setUniformWorkGroupSize(bool u) { workGroupInfo_.uniformWorkGroupSize_ = u; }

  bool getUniformWorkGroupSize() const { return workGroupInfo_.uniformWorkGroupSize_; }

  void setReqdWorkGroupSize(size_t x, size_t y, size_t z) {
    workGroupInfo_.compileSize_[0] = x;
    workGroupInfo_.compileSize_[1] = y;
    workGroupInfo_.compileSize_[2] = z;
  }

  size_t getReqdWorkGroupSize(int dim) { return workGroupInfo_.compileSize_[dim]; }

  void setWorkGroupSizeHint(size_t x, size_t y, size_t z) {
    workGroupInfo_.compileSizeHint_[0] = x;
    workGroupInfo_.compileSizeHint_[1] = y;
    workGroupInfo_.compileSizeHint_[2] = z;
  }

  size_t getWorkGroupSizeHint(int dim) const { return workGroupInfo_.compileSizeHint_[dim]; }

  //! Get profiling callback object
  amd::ProfilingCallback* getProfilingCallback(const device::VirtualDevice* vdev) {
    return waveLimiter_.getProfilingCallback(vdev);
  };

  //! Get waves per shader array to be used for kernel execution.
  uint getWavesPerSH(const device::VirtualDevice* vdev) const {
    return waveLimiter_.getWavesPerSH(vdev);
  };

  //! Returns GPU device object, associated with this kernel
  const amd::Device& dev() const { return dev_; }

  void setVecTypeHint(const std::string& hint) { workGroupInfo_.compileVecTypeHint_ = hint; }

  void setLocalMemSize(size_t size) { workGroupInfo_.localMemSize_ = size; }

  void setPreferredSizeMultiple(size_t size) { workGroupInfo_.preferredSizeMultiple_ = size; }

  //! Return the build log
  const std::string& buildLog() const { return buildLog_; }

  static std::string openclMangledName(const std::string& name);

  const std::unordered_map<size_t, size_t>& patch() const { return patchReferences_; }

  //! Returns TRUE if kernel uses dynamic parallelism
  bool dynamicParallelism() const { return (flags_.dynamicParallelism_) ? true : false; }

  //! set dynamic parallelism flag
  void setDynamicParallelFlag(bool flag) { flags_.dynamicParallelism_ = flag; }

  //! Returns TRUE if kernel is internal kernel
  bool isInternalKernel() const { return (flags_.internalKernel_) ? true : false; }

  //! set internal kernel flag
  void setInternalKernelFlag(bool flag) { flags_.internalKernel_ = flag; }

  //! Return TRUE if kernel uses images
  bool imageEnable() const { return (flags_.imageEna_) ? true : false; }

  //! Return TRUE if kernel wirtes images
  bool imageWrite() const { return (flags_.imageWriteEna_) ? true : false; }

  //! Returns TRUE if it's a HSA kernel
  bool hsa() const { return (flags_.hsa_) ? true : false; }

  //! Return printf info array
  const std::vector<PrintfInfo>& printfInfo() const { return printf_; }

  //! Finds local workgroup size
  void FindLocalWorkSize(
    size_t workDim,                   //!< Work dimension
    const amd::NDRange& gblWorkSize,  //!< Global work size
    amd::NDRange& lclWorkSize         //!< Calculated local work size
  ) const;

 protected:
  //! Initializes the abstraction layer kernel parameters
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
#if defined(USE_COMGR_LIBRARY)
  void InitParameters(const amd_comgr_metadata_node_t kernelMD, uint32_t argBufferSize);

  //! Get ther kernel metadata
  bool GetKernelMetadata(const amd_comgr_metadata_node_t programMD,
                         const std::string& name,
                         amd_comgr_metadata_node_t* kernelNode);

  //! Retrieve kernel attribute and code properties metadata
  bool GetAttrCodePropMetadata(const amd_comgr_metadata_node_t programMD,
                               const uint32_t kernargSegmentByteSize,
                               KernelMD* kernelMD);

  //! Retrieve the available SGPRs and VGPRs
  bool SetAvailableSgprVgpr(const std::string& targetIdent);

  //! Retrieve the printf string metadata
  bool GetPrintfStr(const amd_comgr_metadata_node_t programMD,
                    std::vector<std::string>* printfStr);
#else
  void InitParameters(const KernelMD& kernelMD, uint32_t argBufferSize);
#endif
  //! Initializes HSAIL Printf metadata and info for LC
  void InitPrintf(const std::vector<std::string>& printfInfoStrings);
#endif
#if defined(WITH_COMPILER_LIB)
  void InitParameters(
    const aclArgData* aclArg,   //!< List of ACL arguments
    uint32_t argBufferSize
  );
  //! Initializes HSAIL Printf metadata and info
  void InitPrintf(const aclPrintfFmt* aclPrintf);
#endif
  const amd::Device& dev_;          //!< GPU device object
  std::string name_;                //!< kernel name
  WorkGroupInfo workGroupInfo_;     //!< device kernel info structure
  amd::KernelSignature* signature_; //!< kernel signature
  std::string buildLog_;            //!< build log
  std::vector<PrintfInfo> printf_;  //!< Format strings for GPU printf support
  WaveLimiterManager waveLimiter_;  //!< adaptively control number of waves

  union Flags {
    struct {
      uint imageEna_ : 1;           //!< Kernel uses images
      uint imageWriteEna_ : 1;      //!< Kernel uses image writes
      uint dynamicParallelism_ : 1; //!< Dynamic parallelism enabled
      uint internalKernel_ : 1;     //!< True: internal kernel
      uint hsa_ : 1;                //!< HSA kernel
    };
    uint value_;
    Flags() : value_(0) {}
  } flags_;

 private:
  //! Disable default copy constructor
  Kernel(const Kernel&);

  //! Disable operator=
  Kernel& operator=(const Kernel&);

  std::unordered_map<size_t, size_t> patchReferences_;  //!< Patch table for references
};

#if defined(USE_COMGR_LIBRARY)
static amd_comgr_status_t getMetaBuf(const amd_comgr_metadata_node_t meta,
                                     std::string* str) {
  size_t size = 0;
  amd_comgr_status_t status = amd::Comgr::get_metadata_string(meta, &size, NULL);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    str->resize(size-1);    // minus one to discount the null character
    status = amd::Comgr::get_metadata_string(meta, &size, &((*str)[0]));
  }

  return status;
}

static amd_comgr_status_t populateArgs(const amd_comgr_metadata_node_t key,
                                       const amd_comgr_metadata_node_t value,
                                       void *data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  std::string buf;

  // get the key of the argument field
  size_t size = 0;
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  auto itArgField = ArgFieldMap.find(buf);
  if (itArgField == ArgFieldMap.end()) {
    return AMD_COMGR_STATUS_ERROR;
  }

  // get the value of the argument field
  status = getMetaBuf(value, &buf);

  KernelArgMD* lcArg = static_cast<KernelArgMD*>(data);

  switch (itArgField->second) {
    case ArgField::Name:
      lcArg->mName = buf;
      break;
    case ArgField::TypeName:
      lcArg->mTypeName = buf;
      break;
    case ArgField::Size:
      lcArg->mSize = atoi(buf.c_str());
      break;
    case ArgField::Align:
      lcArg->mAlign = atoi(buf.c_str());
      break;
    case ArgField::ValueKind:
      {
        auto itValueKind = ArgValueKind.find(buf);
        if (itValueKind == ArgValueKind.end()) {
          return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mValueKind = itValueKind->second;
      }
      break;
    case ArgField::ValueType:
      {
        auto itValueType = ArgValueType.find(buf);
        if (itValueType == ArgValueType.end()) {
          return AMD_COMGR_STATUS_ERROR;
       }
       lcArg->mValueType = itValueType->second;
      }
      break;
    case ArgField::PointeeAlign:
      lcArg->mPointeeAlign = atoi(buf.c_str());
      break;
    case ArgField::AddrSpaceQual:
      {
        auto itAddrSpaceQual = ArgAddrSpaceQual.find(buf);
        if (itAddrSpaceQual == ArgAddrSpaceQual.end()) {
          return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mAddrSpaceQual = itAddrSpaceQual->second;
      }
      break;
    case ArgField::AccQual:
      {
        auto itAccQual = ArgAccQual.find(buf);
        if (itAccQual == ArgAccQual.end()) {
          return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mAccQual = itAccQual->second;
      }
      break;
    case ArgField::ActualAccQual:
      {
        auto itAccQual = ArgAccQual.find(buf);
        if (itAccQual == ArgAccQual.end()) {
            return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mActualAccQual = itAccQual->second;
      }
      break;
    case ArgField::IsConst:
      lcArg->mIsConst = (buf.compare("true") == 0);
      break;
    case ArgField::IsRestrict:
      lcArg->mIsRestrict = (buf.compare("true") == 0);
      break;
    case ArgField::IsVolatile:
      lcArg->mIsVolatile = (buf.compare("true") == 0);
      break;
    case ArgField::IsPipe:
      lcArg->mIsPipe = (buf.compare("true") == 0);
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateAttrs(const amd_comgr_metadata_node_t key,
                                        const amd_comgr_metadata_node_t value,
                                        void *data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  size_t size = 0;
  std::string buf;

  // get the key of the argument field
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  auto itAttrField = AttrFieldMap.find(buf);
  if (itAttrField == AttrFieldMap.end()) {
    return AMD_COMGR_STATUS_ERROR;
  }

  KernelMD* kernelMD = static_cast<KernelMD*>(data);
  switch (itAttrField->second) {
    case AttrField::ReqdWorkGroupSize:
      {
        status = amd::Comgr::get_metadata_list_size(value, &size);
        if (size == 3 && status == AMD_COMGR_STATUS_SUCCESS) {
          for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
            amd_comgr_metadata_node_t workgroupSize;
            status = amd::Comgr::index_list_metadata(value, i, &workgroupSize);

            if (status == AMD_COMGR_STATUS_SUCCESS &&
                getMetaBuf(workgroupSize, &buf) == AMD_COMGR_STATUS_SUCCESS) {
              kernelMD->mAttrs.mReqdWorkGroupSize.push_back(atoi(buf.c_str()));
            }
            amd::Comgr::destroy_metadata(workgroupSize);
          }
        }
      }
      break;
    case AttrField::WorkGroupSizeHint:
      {
        status = amd::Comgr::get_metadata_list_size(value, &size);
        if (status == AMD_COMGR_STATUS_SUCCESS && size == 3) {
          for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
            amd_comgr_metadata_node_t workgroupSizeHint;
            status = amd::Comgr::index_list_metadata(value, i, &workgroupSizeHint);

            if (status == AMD_COMGR_STATUS_SUCCESS &&
                getMetaBuf(workgroupSizeHint, &buf) == AMD_COMGR_STATUS_SUCCESS) {
              kernelMD->mAttrs.mWorkGroupSizeHint.push_back(atoi(buf.c_str()));
            }
            amd::Comgr::destroy_metadata(workgroupSizeHint);
          }
        }
      }
      break;
    case AttrField::VecTypeHint:
      {
        if (getMetaBuf(value,&buf) == AMD_COMGR_STATUS_SUCCESS) {
          kernelMD->mAttrs.mVecTypeHint = buf;
        }
      }
      break;
    case AttrField::RuntimeHandle:
      {
        if (getMetaBuf(value,&buf) == AMD_COMGR_STATUS_SUCCESS) {
          kernelMD->mAttrs.mRuntimeHandle = buf;
        }
      }
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }

  return status;
}

static amd_comgr_status_t populateCodeProps(const amd_comgr_metadata_node_t key,
                                            const amd_comgr_metadata_node_t value,
                                            void *data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  std::string buf;

  // get the key of the argument field
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  auto itCodePropField = CodePropFieldMap.find(buf);
  if (itCodePropField == CodePropFieldMap.end()) {
    return AMD_COMGR_STATUS_ERROR;
  }

  // get the value of the argument field
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(value, &buf);
  }

  KernelMD*  kernelMD = static_cast<KernelMD*>(data);
  switch (itCodePropField->second) {
    case CodePropField::KernargSegmentSize:
      kernelMD->mCodeProps.mKernargSegmentSize = atoi(buf.c_str());
      break;
    case CodePropField::GroupSegmentFixedSize:
      kernelMD->mCodeProps.mKernargSegmentSize = atoi(buf.c_str());
      break;
    case CodePropField::PrivateSegmentFixedSize:
      kernelMD->mCodeProps.mPrivateSegmentFixedSize = atoi(buf.c_str());
      break;
    case CodePropField::KernargSegmentAlign:
      kernelMD->mCodeProps.mKernargSegmentAlign = atoi(buf.c_str());
      break;
    case CodePropField::WavefrontSize:
      kernelMD->mCodeProps.mWavefrontSize = atoi(buf.c_str());
      break;
    case CodePropField::NumSGPRs:
      kernelMD->mCodeProps.mNumSGPRs = atoi(buf.c_str());
      break;
    case CodePropField::NumVGPRs:
      kernelMD->mCodeProps.mNumVGPRs = atoi(buf.c_str());
      break;
    case CodePropField::MaxFlatWorkGroupSize:
      kernelMD->mCodeProps.mMaxFlatWorkGroupSize = atoi(buf.c_str());
      break;
    case CodePropField::IsDynamicCallStack:
        kernelMD->mCodeProps.mIsDynamicCallStack = (buf.compare("true") == 0);
      break;
    case CodePropField::IsXNACKEnabled:
      kernelMD->mCodeProps.mIsXNACKEnabled = (buf.compare("true") == 0);
      break;
    case CodePropField::NumSpilledSGPRs:
      kernelMD->mCodeProps.mNumSpilledSGPRs = atoi(buf.c_str());
      break;
    case CodePropField::NumSpilledVGPRs:
      kernelMD->mCodeProps.mNumSpilledVGPRs = atoi(buf.c_str());
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}
#endif

} // namespace device
