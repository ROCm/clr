//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "include/aclTypes.h"
#include "platform/context.hpp"
#include "platform/object.hpp"
#include "platform/memory.hpp"
#include "devwavelimiter.hpp"

#if defined(WITH_LIGHTNING_COMPILER)
namespace llvm {
  namespace AMDGPU {
    namespace HSAMD {
      namespace Kernel {
        struct Metadata;
}}}}
typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
#endif  // defined(WITH_LIGHTNING_COMPILER)

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
#if defined(WITH_LIGHTNING_COMPILER)
  void InitParameters(const KernelMD& kernelMD, uint32_t argBufferSize);
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

} // namespace device