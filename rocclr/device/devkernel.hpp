/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#pragma once

#include "include/aclTypes.h"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "devwavelimiter.hpp"

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
    QueueObject = 13,
    HiddenMultiGridSync = 14,
    HiddenHostcallBuffer = 15,
  };
  clk_value_type_t type_;  //!< The parameter's type
  size_t offset_;          //!< Its offset in the parameter's stack
  size_t size_;            //!< Its size in bytes
  union InfoData {
    struct {
      uint32_t oclObject_ : 4;    //!< OCL object type
      uint32_t readOnly_ : 1;     //!< OCL object is read only, applied to memory only
      uint32_t rawPointer_ : 1;   //!< Arguments have a raw GPU VA
      uint32_t defined_ : 1;      //!< The argument was defined by the app
      uint32_t hidden_ : 1;       //!< It's a hidden argument
      uint32_t shared_ : 1;       //!< Dynamic shared memory
      uint32_t reserved_ : 3;     //!< Reserved       
      uint32_t arrayIndex_ : 20;  //!< Index in the objects array or LDS alignment
    };
    uint32_t allValues_;
    InfoData() : allValues_(0) {}
  } info_;

  cl_kernel_arg_address_qualifier addressQualifier_ =
      CL_KERNEL_ARG_ADDRESS_PRIVATE;                  //!< Argument's address qualifier
  cl_kernel_arg_access_qualifier accessQualifier_ =
      CL_KERNEL_ARG_ACCESS_NONE;                      //!< Argument's access qualifier
  cl_kernel_arg_type_qualifier typeQualifier_;        //!< Argument's type qualifier

  std::string name_;      //!< The parameter's name in the source
  std::string typeName_;  //!< Argument's type name
  uint32_t alignment_;    //!< Argument's alignment
};
}

#if defined(USE_COMGR_LIBRARY)
//! Runtime handle structure for device enqueue
struct RuntimeHandle {
  uint64_t kernel_handle;             //!< Pointer to amd_kernel_code_s or kernel_descriptor_t
  uint32_t private_segment_size;      //!< From PRIVATE_SEGMENT_FIXED_SIZE
  uint32_t group_segment_size;        //!< From GROUP_SEGMENT_FIXED_SIZE
};

#include "amd_comgr.h"

//  for Code Object V3
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
  IsPipe        = 13,
  Offset        = 14
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


static const std::map<std::string, ArgField> ArgFieldMap =
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

static const std::map<std::string, uint32_t> ArgValueKind = {
  {"ByValue",                 amd::KernelParameterDescriptor::ValueObject},
  {"GlobalBuffer",            amd::KernelParameterDescriptor::MemoryObject},
  {"DynamicSharedPointer",    amd::KernelParameterDescriptor::MemoryObject},
  {"Sampler",                 amd::KernelParameterDescriptor::SamplerObject},
  {"Image",                   amd::KernelParameterDescriptor::ImageObject },
  {"Pipe",                    amd::KernelParameterDescriptor::MemoryObject},
  {"Queue",                   amd::KernelParameterDescriptor::QueueObject},
  {"HiddenGlobalOffsetX",     amd::KernelParameterDescriptor::HiddenGlobalOffsetX},
  {"HiddenGlobalOffsetY",     amd::KernelParameterDescriptor::HiddenGlobalOffsetY},
  {"HiddenGlobalOffsetZ",     amd::KernelParameterDescriptor::HiddenGlobalOffsetZ},
  {"HiddenNone",              amd::KernelParameterDescriptor::HiddenNone},
  {"HiddenPrintfBuffer",      amd::KernelParameterDescriptor::HiddenPrintfBuffer},
  {"HiddenDefaultQueue",      amd::KernelParameterDescriptor::HiddenDefaultQueue},
  {"HiddenCompletionAction",  amd::KernelParameterDescriptor::HiddenCompletionAction},
  {"HiddenMultigridSyncArg",  amd::KernelParameterDescriptor::HiddenMultiGridSync},
  {"HiddenHostcallBuffer",    amd::KernelParameterDescriptor::HiddenHostcallBuffer}
};

static const std::map<std::string, std::pair<uint32_t, uint32_t>> ArgValueType = {
  {"Struct",  {0, 0}},
  {"I8",      {0, 1}},
  {"U8",      {0, 1}},
  {"I16",     {1, 2}},
  {"U16",     {1, 2}},
  {"F16",     {4, 2}},
  {"I32",     {2, 4}},
  {"U32",     {2, 4}},
  {"F32",     {4, 4}},
  {"I64",     {3, 8}},
  {"U64",     {3, 8}},
  {"F64",     {5, 8}}
};

static const std::map<std::string, cl_kernel_arg_access_qualifier> ArgAccQual = {
  {"Default",   CL_KERNEL_ARG_ACCESS_NONE},
  {"ReadOnly",  CL_KERNEL_ARG_ACCESS_READ_ONLY},
  {"WriteOnly", CL_KERNEL_ARG_ACCESS_WRITE_ONLY},
  {"ReadWrite", CL_KERNEL_ARG_ACCESS_READ_WRITE}
};

static const std::map<std::string, cl_kernel_arg_address_qualifier> ArgAddrSpaceQual = {
  {"Private",   CL_KERNEL_ARG_ADDRESS_PRIVATE},
  {"Global",    CL_KERNEL_ARG_ADDRESS_GLOBAL},
  {"Constant",  CL_KERNEL_ARG_ADDRESS_CONSTANT},
  {"Local",     CL_KERNEL_ARG_ADDRESS_LOCAL},
  {"Generic",   CL_KERNEL_ARG_ADDRESS_GLOBAL},
  {"Region",    CL_KERNEL_ARG_ADDRESS_PRIVATE}
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

//  for Code Object V3
enum class KernelField : uint8_t {
  SymbolName              = 0,
  ReqdWorkGroupSize       = 1,
  WorkGroupSizeHint       = 2,
  VecTypeHint             = 3,
  DeviceEnqueueSymbol     = 4,
  KernargSegmentSize      = 5,
  GroupSegmentFixedSize   = 6,
  PrivateSegmentFixedSize = 7,
  KernargSegmentAlign     = 8,
  WavefrontSize           = 9,
  NumSGPRs                = 10,
  NumVGPRs                = 11,
  MaxFlatWorkGroupSize    = 12,
  NumSpilledSGPRs         = 13,
  NumSpilledVGPRs         = 14
};

static const std::map<std::string,ArgField> ArgFieldMapV3 =
{
  {".name",           ArgField::Name},
  {".type_name",      ArgField::TypeName},
  {".size",           ArgField::Size},
  {".offset",         ArgField::Offset},
  {".value_kind",     ArgField::ValueKind},
  {".value_type",     ArgField::ValueType},
  {".pointee_align",  ArgField::PointeeAlign},
  {".address_space",  ArgField::AddrSpaceQual},
  {".access",         ArgField::AccQual},
  {".actual_access",  ArgField::ActualAccQual},
  {".is_const",       ArgField::IsConst},
  {".is_restrict",    ArgField::IsRestrict},
  {".is_volatile",    ArgField::IsVolatile},
  {".is_pipe",        ArgField::IsPipe}
};

static const std::map<std::string, uint32_t> ArgValueKindV3 = {
  {"by_value",                  amd::KernelParameterDescriptor::ValueObject},
  {"global_buffer",             amd::KernelParameterDescriptor::MemoryObject},
  {"dynamic_shared_pointer",    amd::KernelParameterDescriptor::MemoryObject},
  {"sampler",                   amd::KernelParameterDescriptor::SamplerObject},
  {"image",                     amd::KernelParameterDescriptor::ImageObject },
  {"pipe",                      amd::KernelParameterDescriptor::MemoryObject},
  {"queue",                     amd::KernelParameterDescriptor::QueueObject},
  {"hidden_global_offset_x",    amd::KernelParameterDescriptor::HiddenGlobalOffsetX},
  {"hidden_global_offset_y",    amd::KernelParameterDescriptor::HiddenGlobalOffsetY},
  {"hidden_global_offset_z",    amd::KernelParameterDescriptor::HiddenGlobalOffsetZ},
  {"hidden_none",               amd::KernelParameterDescriptor::HiddenNone},
  {"hidden_printf_buffer",      amd::KernelParameterDescriptor::HiddenPrintfBuffer},
  {"hidden_default_queue",      amd::KernelParameterDescriptor::HiddenDefaultQueue},
  {"hidden_completion_action",  amd::KernelParameterDescriptor::HiddenCompletionAction},
  {"hidden_multigrid_sync_arg", amd::KernelParameterDescriptor::HiddenMultiGridSync},
  {"hidden_hostcall_buffer",    amd::KernelParameterDescriptor::HiddenHostcallBuffer}
};

static const std::map<std::string, std::pair<uint32_t, uint32_t>> ArgValueTypeV3 = {
  {"struct",  {0, 0}},
  {"i8",      {0, 1}},
  {"u8",      {0, 1}},
  {"i16",     {1, 2}},
  {"u16",     {1, 2}},
  {"f16",     {4, 2}},
  {"i32",     {2, 4}},
  {"u32",     {2, 4}},
  {"f32",     {4, 4}},
  {"i64",     {3, 8}},
  {"u64",     {3, 8}},
  {"f64",     {5, 8}}
};

static const std::map<std::string, cl_kernel_arg_access_qualifier> ArgAccQualV3 = {
  {"default",    CL_KERNEL_ARG_ACCESS_NONE},
  {"read_only",  CL_KERNEL_ARG_ACCESS_READ_ONLY},
  {"write_only", CL_KERNEL_ARG_ACCESS_WRITE_ONLY},
  {"read_write", CL_KERNEL_ARG_ACCESS_READ_WRITE}
};

static const std::map<std::string, cl_kernel_arg_address_qualifier> ArgAddrSpaceQualV3 = {
  {"private",   CL_KERNEL_ARG_ADDRESS_PRIVATE},
  {"global",    CL_KERNEL_ARG_ADDRESS_GLOBAL},
  {"constant",  CL_KERNEL_ARG_ADDRESS_CONSTANT},
  {"local",     CL_KERNEL_ARG_ADDRESS_LOCAL},
  {"generic",   CL_KERNEL_ARG_ADDRESS_GLOBAL},
  {"region",    CL_KERNEL_ARG_ADDRESS_PRIVATE}
};

static const std::map<std::string,KernelField> KernelFieldMapV3 =
{
  {".symbol",                     KernelField::SymbolName},
  {".reqd_workgroup_size",        KernelField::ReqdWorkGroupSize},
  {".workgorup_size_hint",        KernelField::WorkGroupSizeHint},
  {".vec_type_hint",              KernelField::VecTypeHint},
  {".device_enqueue_symbol",      KernelField::DeviceEnqueueSymbol},
  {".kernarg_segment_size",       KernelField::KernargSegmentSize},
  {".group_segment_fixed_size",   KernelField::GroupSegmentFixedSize},
  {".private_segment_fixed_size", KernelField::PrivateSegmentFixedSize},
  {".kernarg_segment_align",      KernelField::KernargSegmentAlign},
  {".wavefront_size",             KernelField::WavefrontSize},
  {".sgpr_count",                 KernelField::NumSGPRs},
  {".vgpr_count",                 KernelField::NumVGPRs},
  {".max_flat_workgroup_size",    KernelField::MaxFlatWorkGroupSize},
  {".sgpr_spill_count",           KernelField::NumSpilledSGPRs},
  {".vgpr_spill_count",           KernelField::NumSpilledVGPRs}
};

#endif  // defined(USE_COMGR_LIBRARY)

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

namespace device {

class Program;

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
    uint64_t localMemSize_;           //!< amount of used local memory
    size_t preferredSizeMultiple_;    //!< preferred multiple for launch
    uint64_t privateMemSize_;         //!< amount of used private memory
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
    int maxOccupancyPerCu_;           //!< Max occupancy per compute unit in threads
  };

  //! Default constructor
  Kernel(const amd::Device& dev, const std::string& name, const Program& prog);

  //! Default destructor
  virtual ~Kernel();

  //! Returns the kernel info structure
  const WorkGroupInfo* workGroupInfo() const { return &workGroupInfo_; }
  //! Returns the kernel info structure for filling in
  WorkGroupInfo* workGroupInfo() { return &workGroupInfo_; }

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

  const std::string& RuntimeHandle() const { return runtimeHandle_; }
  void setRuntimeHandle(const std::string& handle) { runtimeHandle_ = handle; }

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

  const uint64_t KernelCodeHandle() const { return kernelCodeHandle_; }

  const uint32_t WorkgroupGroupSegmentByteSize() const { return workgroupGroupSegmentByteSize_; }
  void SetWorkgroupGroupSegmentByteSize(uint32_t size) { workgroupGroupSegmentByteSize_ = size; }

  const uint32_t WorkitemPrivateSegmentByteSize() const { return workitemPrivateSegmentByteSize_; }
  void SetWorkitemPrivateSegmentByteSize(uint32_t size) { workitemPrivateSegmentByteSize_ = size; }
  
  const uint32_t KernargSegmentByteSize() const { return kernargSegmentByteSize_; }
  void SetKernargSegmentByteSize(uint32_t size) { kernargSegmentByteSize_ = size; }

  const uint8_t KernargSegmentAlignment() const { return kernargSegmentAlignment_; }
  void SetKernargSegmentAlignment(uint32_t align) { kernargSegmentAlignment_ = align; }

  void SetSymbolName(const std::string& name) { symbolName_ = name; }

 protected:
  //! Initializes the abstraction layer kernel parameters
#if defined(USE_COMGR_LIBRARY)
  void InitParameters(const amd_comgr_metadata_node_t kernelMD);

  //! Retrieve kernel attribute and code properties metadata
  bool GetAttrCodePropMetadata();

  //! Retrieve the available SGPRs and VGPRs
  bool SetAvailableSgprVgpr(const std::string& targetIdent);

  //! Retrieve the printf string metadata
  bool GetPrintfStr(std::vector<std::string>* printfStr);

  //! Returns the kernel symbol name
  const std::string& symbolName() const { return symbolName_; }

  //! Returns the kernel code object version
  const uint32_t codeObjectVer() const { return prog().codeObjectVer(); }
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
  //! Returns program associated with this kernel
  const Program& prog() const { return prog_; }

  const amd::Device& dev_;          //!< GPU device object
  std::string name_;                //!< kernel name
  const Program& prog_;             //!< Reference to the parent program
  std::string symbolName_;          //!< kernel symbol name
  WorkGroupInfo workGroupInfo_;     //!< device kernel info structure
  amd::KernelSignature* signature_; //!< kernel signature
  std::string buildLog_;            //!< build log
  std::vector<PrintfInfo> printf_;  //!< Format strings for GPU printf support
  WaveLimiterManager waveLimiter_;  //!< adaptively control number of waves
  std::string runtimeHandle_;       //!< Runtime handle for context loader

  uint64_t kernelCodeHandle_ = 0;   //!< Kernel code handle (aka amd_kernel_code_t)
  uint32_t workgroupGroupSegmentByteSize_ = 0;
  uint32_t workitemPrivateSegmentByteSize_ = 0;
  uint32_t kernargSegmentByteSize_ = 0;   //!< Size of kernel argument buffer
  uint32_t kernargSegmentAlignment_ = 0;

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
amd_comgr_status_t getMetaBuf(const amd_comgr_metadata_node_t meta, std::string* str);
#endif // defined(USE_COMGR_LIBRARY)
} // namespace device
