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

#ifndef GPUKERNEL_HPP_
#define GPUKERNEL_HPP_

#include "device/device.hpp"
#include "utils/macros.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "platform/sampler.hpp"
#include "device/gpu/gpudevice.hpp"
#include "device/gpu/gpuvirtual.hpp"
#include "amd_hsa_kernel_code.h"
#include "device/gpu/gpuprintf.hpp"
#include "device/devwavelimiter.hpp"
#include "hsa.h"

namespace amd {
namespace hsa {
namespace loader {
class Symbol;
}  // loader
}  // hsa
}  // amd

//! \namespace gpu GPU Device Implementation
namespace gpu {

class VirtualGPU;
class Device;
class NullDevice;
class HSAILProgram;

struct HWSHADER_Helper {
  template <typename S, typename T> static T Get(S base, T offset) {
    return reinterpret_cast<T>(reinterpret_cast<intptr_t>(base) + reinterpret_cast<size_t>(offset));
  }
};

#define HWSHADER_Get(shader, field) HWSHADER_Helper::Get((shader), (shader)->field)

template <typename D, typename S>
static void CalcPtr(D& dst, const S src, size_t structSize, size_t size) {
  dst = reinterpret_cast<D>(reinterpret_cast<const intptr_t>(src) + structSize * size);
}

/*! \addtogroup GPU GPU Device Implementation
 *  @{
 */

/*! \brief Helper function for the std::string processing.
 *  Finds the name in the std::string
 *
 *  \return True if we found the entry of the symbols
 */
bool expect(const std::string& str,  //!< The original std::string
            size_t* pos,             //!< Position to start
            const std::string& sym   //!< The sympols to expect
            );

/*! \brief Helper function for the std::string processing.
 *  Gets a word from the std::string
 *
 *  \return True if we successfully received a word
 */
bool getword(const std::string& str,  //!< The original std::string
             size_t* pos,             //!< Position to start
             std::string& sym         //!< Returned word
             );

/*! \brief Helper function for the std::string processing.
 *  Loads numbers from the metadata
 *
 *  \return True if we loaded a number
 */
bool getuint(const std::string& str,  //!< The original std::string
             size_t* pos,             //!< Position to start
             uint* val                //!< Returned number
             );

/*! \brief Helper function for the std::string processing.
 *  Loads numbers from the metadata in HEX format
 *
 *  \return True if we loaded a number
 */
bool getuintHex(const std::string& str,  //!< The original std::string
                size_t* pos,             //!< Position to start
                uint* val                //!< Returned number
                );

/*! \brief Helper function for the std::string processing.
 *  Loads numbers from the metadata in HEX format
 *
 *  \return True if we loaded a number
 */
bool getuint64Hex(const std::string& str,  //!< The original std::string
                  size_t* pos,             //!< Position to start
                  uint64_t* val            //!< Returned number
                  );

/*! \brief Helper function for the std::string processing.
 *  Converts unsigned integer to string
 *
 *  \return None
 */
void intToStr(size_t value,  //!< Value for conversion
              char* str,     //!< Pointer to the converted string
              size_t size    //!< String size
              );

//! Image constant data from ABI specification
struct ImageConstants : public amd::EmbeddedObject {
  uint32_t width_;         //!< Image surface width
  uint32_t height_;        //!< Image surface height
  uint32_t depth_;         //!< Image surface depth (1 for 2D images)
  uint32_t dataType_;      //!< Image surface data type
  float widthFloat_;       //!< Image surface width
  float heightFloat_;      //!< Image surface height
  float depthFloat_;       //!< Image surface depth (1 for 2D images)
  uint32_t channelOrder_;  //!< Image surface texels channel order
};

//! Kernel arguments
struct KernelArg : public amd::HeapObject {
 public:
  //! \enum Kernel argument type
  enum ArgumentType {
    NoType = 0,
    PointerGlobal,
    Value,
    Image,
    PointerLocal,
    PointerHwLocal,
    PointerPrivate,
    PointerHwPrivate,
    PointerConst,
    PointerHwConst,
    Float,
    Double,
    Half,
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    Struct,
    Union,
    Opaque,
    Event,
    Image1D,  //!< first image
    Image2D,
    Image1DB,
    Image1DA,
    Image2DA,
    Image3D,  //!< last image
    Counter,
    Sampler,
    PrivateSize,
    LocalSize,
    HwPrivateSize,
    HwLocalSize,
    Grouping,
    WrkgrpSize,
    Wavefront,
    PrivateFixed,
    ErrorMessage,
    WarningMessage,
    PrintfFormatStr,
    MetadataVersion,
    UavId,
    ABI64Bit,
    GWS,
    SWGWS,
    Reflection,
    ConstArg,
    ConstBufId,
    PrintfBufId,
    GroupingHint,
    VecTypeHint,
    WavesPerSimdHint,
    TotalTypes
  };

  // The compiler metadata fields
  std::string name_;   //!< parameters name
  ArgumentType type_;  //!< type of argument
  union {
    uint size_;      //!< number of arguments (for values and pointers only)
    uint location_;  //!< sampler's location (for samplers only)
  };
  uint cbIdx_;             //!< constant buffer index
  uint cbPos_;             //!< dword address in CB for the argument
  std::string buf_;        //!< buffer tag
  uint index_;             //!< buffer/image/sampler index
  uint alignment_;         //!< the required argument's alignment
  ArgumentType dataType_;  //!< data type of the argument
  union {
    struct {
      uint uavBuf_ : 1;     //!< UAV memory, no global heap
      uint realloc_ : 1;    //!< argument has to be reallocatedin the global heap
      uint readOnly_ : 1;   //!< Read only memory object
      uint writeOnly_ : 1;  //!< Write only memory object
      uint readWrite_ : 1;  //!< Read/Write memory object
    };
    uint value_;
  } memory_;

  std::string typeName_;  //!< argument's type name
  uint typeQualifier_;    //!< argument's type qualifier

  //! Default constructor for the kernel argument
  KernelArg();

  //! Copy constructor for the kernel argument
  KernelArg(const KernelArg& data);

  //! Overloads operator=
  KernelArg& operator=(const KernelArg& data);

  //! Destructor of the kernel argument
  ~KernelArg() { name_.clear(); }

  /*! \brief Checks if this arguments requires a place in constant buffer
   *
   *  \return True if we need CB
   */
  bool isCbNeeded() const;

  /*! \brief Retrieves the argument's size
   *
   *  \return Size of the current argument
   */
  size_t size(bool gpuLayer  //!< True if we want the argument's size for the GPU layer
              ) const;

  /*! \brief Retrieves the argument's type for the abstraction layer
   *
   *  \return The argument's type in the abstraction layer format
   */
  clk_value_type_t type() const;

  /*! \brief Retrieves the argument's address qualifier for the abstraction layer
   *
   *  \return The argument's address qualifier in the abstraction layer format
   */
  cl_kernel_arg_address_qualifier addressQualifier() const;

  /*! \brief Retrieves the argument's access qualifier for the abstraction layer
   *
   *  \return The argument's access qualifier in the abstraction layer format
   */
  cl_kernel_arg_access_qualifier accessQualifier() const;

  /*! \brief Retrieves the argument's type name for the abstraction layer
   *
   *  \return The argument's type name
   */
  const char* typeName() const { return typeName_.c_str(); }

  /*! \brief Retrieves the argument's type qualifier for the abstraction layer
   *
   *  \return The argument's type qualifier
   */
  cl_kernel_arg_type_qualifier typeQualifier() const {
    switch (type_) {
      case PointerConst:
      case PointerHwConst:
        return static_cast<cl_kernel_arg_type_qualifier>(typeQualifier_ | CL_KERNEL_ARG_TYPE_CONST);
      default:
        return static_cast<cl_kernel_arg_type_qualifier>(typeQualifier_);
    }
  }

  //! Special case for vectors with component size <= 16bit
  const static uint VectorSizeLimit = 4;
  size_t specialVector() const;
};

struct DataTypeConst {
  const char* tagName_;           //!< data type's name
  KernelArg::ArgumentType type_;  //!< data type
};

//!  Metadata description for parsing
struct MetaDataConst {
  const char* typeName_;          //!< parameters name
  KernelArg::ArgumentType type_;  //!< type of argument
  struct {
    uint size_ : 1;      //!< number of arguments
    uint name_ : 1;      //!< argument's name
    uint resType_ : 1;   //!< argument's type
    uint cbIdx_ : 1;     //!< resource index CB, sampler or image
    uint cbPos_ : 1;     //!< dword address in CB for the argument
    uint buf_ : 1;       //!< buffer tag
    uint reserved : 26;  //!< reserved
  };
};

const uint DescTotal = 15;
const uint BasicTypeTotal = 15;
const uint ArgStateTotal = DescTotal + BasicTypeTotal;

//! The constant array that describes different metadata properties
extern const MetaDataConst ArgState[ArgStateTotal];

extern const DataTypeConst DataType[];

extern const uint DataTypeTotal;

// Forward declaration
class Program;
class NullProgram;

class CalImageReference : public amd::ReferenceCountedObject {
 public:
  //! Default constructor
  CalImageReference(CALimage calImage) : image_(calImage) {}

  //! Get CAL image
  CALimage calImage() const { return image_; }

 protected:
  //! Default destructor
  ~CalImageReference();

 private:
  //! Disable copy constructor
  CalImageReference(const CalImageReference&);

  //! Disable operator=
  CalImageReference& operator=(const CalImageReference&);

  CALimage image_;  //!< CAL kernel image
};

//! \class GPU NullKernel - Kernel for offline device
class NullKernel : public device::Kernel {
 public:
  typedef std::vector<KernelArg*> arguments_t;

  const static uint UavIdUndefined = 0xffff;

  enum Flags {
    LimitWorkgroup = 1 << 0,  //!< Limits the workgroup size
    PrintfOutput = 1 << 1,    //!< Kernel has printf output
    PrivateFixed = 1 << 2,    //!< Kernel has printf output
    ABI64bit = 1 << 3,        //!< Kernel has 64 bit ABI
    Unused0 = 1 << 4,         //!< Unused
    Unused1 = 1 << 5,         //!< Unused
    ImageEnable = 1 << 6,     //!< Kernel uses images
    ImageWrite = 1 << 7,      //!< Kernel writes images
  };

  //! \enum Resource type for binding
  enum ResourceType {
    Undefined = 0x00000000,            //!< resource type will be detected
    ConstantBuffer = 0x00000001,       //!< resource is a constant buffer
    GlobalBuffer = 0x00000002,         //!< resource is a global buffer
    ArgumentHeapBuffer = 0x00000004,   //!< resource is an argument buffer
    ArgumentBuffer = 0x00000005,       //!< resource is an argument buffer
    ArgumentImageRead = 0x00000006,    //!< resource is an argument image read
    ArgumentImageWrite = 0x00000007,   //!< resource is an argument image write
    ArgumentConstBuffer = 0x00000008,  //!< resource is an argument const buffer
    ArgumentCounter = 0x00000009,      //!< resource is a global counter
    ArgumentUavID = 0x0000000a,        //!< resource is a dummy ID read
    ArgumentCbID = 0x0000000b,         //!< resource is a constant buffer
    ArgumentPrintfID = 0x0000000c,     //!< resource is a printf buffer
  };

  //! GPU kernel constructor
  NullKernel(const std::string& name,       //!< The kernel's name
             const NullDevice& gpuNullDev,  //!< GPU device object
             const NullProgram& nullProg    //!< Reference to the program
             );

  virtual ~NullKernel();

  /*! \brief Creates a GPU kernel in CAL
   *
   *  \return True if we successfully created a kernel in CAL
   */
  bool create(const std::string& code,        //!< IL source code
              const std::string& metadata,    //!< the kernel metadata structure
              const void* binaryCode = NULL,  //!< binary machine code for CAL
              size_t binarySize = 0           //!< the machine code size
              );

  //! Returns CAL function descriptor
  CALimage calImage() const { return calRef_->calImage(); }

  //! Returns TRUE if we successfully retrieved the binary from CAL
  bool getCalBinary(void* binary,  //!< ISA binary code
                    size_t size    //!< ISA binary size
                    ) const;

  //! Returns CAL image size
  size_t getCalBinarySize() const;

  //! Returns GPU device object, associated with this kernel
  const NullDevice& nullDev() const { return gpuDev_; }

  //! Returns GPU device object, associated with this kernel
  const NullProgram& nullProg() const { return reinterpret_cast<const NullProgram&>(prog_); }

  //! Returns the kernel's build error
  const int32_t buildError() const { return buildError_; }

  //! Returns the kernel's flags
  uint flags() const { return flags_; }

  //! Returns TRUE if ABI is for 64 bits
  bool abi64Bit() const { return (flags_ & ABI64bit) ? true : false; }

  //! Returns the total number of all arguments
  size_t argSize() const { return arguments_.size(); }

  //! Returns instruction count of the current kernel
  uint instructionCnt() const { return instructionCnt_; }

 protected:
  /*! \brief Parses the metadata structure for the kernel,
   *   provided by the OpenCL compiler
   *
   *  \return True if we succefully parsed all arguments
   */
  bool parseArguments(const std::string& metaData,  //!< the program for parsing
                      uint* uavRefCount  //!< an array of reference counters for used UAVs
                      );

  //! Returns the argument for the specified index
  const KernelArg* argument(uint idx) const { return arguments_[idx]; }

  //! Adds the kernel argument into the list
  void addArgument(KernelArg* arg) { arguments_.push_back(arg); }

  //! Returns the argument for the specified sampler's index
  const KernelArg* sampler(uint idx) const { return intSamplers_[idx]; }

  //! Returns the total number of all internal samplers
  size_t samplerSize() const { return intSamplers_.size(); }

  //! Adds the kernel sampler into the sampler's list
  void addSampler(KernelArg* arg) { intSamplers_.push_back(arg); }

  //! Returns UAV raw index for this kernel
  uint uavRaw() const { return uavRaw_; }

  int32_t buildError_;     //!< Kernel's build error
  std::string ilSource_;  //!< IL source code of this kernel

  const NullDevice& gpuDev_;  //!< GPU device object

  CalImageReference* calRef_;  //!< CAL image reference for this kernel
  bool internal_;              //!< Runtime internal ker

  uint flags_;               //!< kernel object flags
  arguments_t arguments_;    //!< kernel arguments for the execution
  arguments_t intSamplers_;  //!< predefined intenal kernel samplers

  size_t* cbSizes_;  //!< real constant buffer sizes for this kernel
  uint numCb_;       //!< total number of constant buffers

  uint uavRaw_;  //!< UAV used for RAW access

  bool rwAttributes_;  //!< backend provides RW attributes for arguments

  uint instructionCnt_;  //!< Instruction count

  uint cbId_;      //!< UAV used for constant buffer access
  uint printfId_;  //!< UAV used for printf buffer access

 private:
  //! Disable copy constructor
  NullKernel(const NullKernel&);

  //! Disable operator=
  NullKernel& operator=(const NullKernel&);

  //! Creates a filename for ISA/IL dumps
  std::string mkDumpName(const char* extension  //!< File extension to append
                         ) const;

  bool createMultiBinary(uint* imageSize,  //!< Multibinary image size
                         void** image,     //!< Multibinary image
                         const void* isa   //!< Kernel HW info
                         );

  //! SI HW specific setup for kernels
  bool siCreateHwInfo(const void* shader,          //!< HW info shader
                      AMUabiAddEncoding& encoding  //!< ABI encoding structure
                      );

  //! r800 HW specific setup for kernels
  bool r800CreateHwInfo(const void* shader,          //!< HW info shader
                        AMUabiAddEncoding& encoding  //!< ABI encoding structure
                        );
};

//! \class GPU kernel
class Kernel : public NullKernel {
 public:
  struct InitData {
    uint privateSize_;    //!< Private ring initial size
    uint localSize_;      //!< Local ring initial size
    uint hwPrivateSize_;  //!< HW private ring initial size
    uint hwLocalSize_;    //!< HW local ring initial size
    uint flags_;          //!< Kernel initialization flags
  };

  //! GPU kernel constructor
  Kernel(const std::string& name,   //!< The kernel's name
         const Device& gpuDev,      //!< GPU device object
         const Program& prog,       //!< Reference to the program
         const InitData* initData_  //!< Initialization data
         );

  //! GPU kernel destructor
  virtual ~Kernel();

  /*! \brief Creates a GPU kernel in CAL
   *
   *  \return True if we successfully created a kernel in CAL
   */
  bool create(const std::string& code,        //!< IL source code
              const std::string& metadata,    //!< the kernel metadata structure
              const void* binaryCode = NULL,  //!< binary machine code for CAL
              size_t binarySize = 0           //!< the machine code size
              );

  //! Initializes the CAL program grid for the kernel execution
  void setupProgramGrid(VirtualGPU& gpu,                    //!< virtual GPU device object
                        size_t workDim,                     //!< work dimension
                        const amd::NDRange& glbWorkOffset,  //!< global work offset
                        const amd::NDRange& gblWorkSize,    //!< global work size
                        amd::NDRange& lclWorkSize,          //!< local work size
                        const amd::NDRange& groupOffset,    //!< group offsets
                        const amd::NDRange& glbWorkOffsetOrg,
                        const amd::NDRange& glbWorkSizeOrg  //!< original global work size
                        ) const;

  /*! \brief Detects if runtime has to disable cache optimization and
   *   recompiles the kernel
   *
   *  \return True if aliases were detected in the kernel arguments
   */
  void processMemObjects(VirtualGPU& gpu,            //!< Virtual GPU objects - queue
                         const amd::Kernel& kernel,  //!< AMD kernel object for execution
                         const_address params,       //!< pointer to the param's store
                         bool nativeMem              //!< Native memory objects
                         ) const;

  /*! \brief Loads all kernel arguments, so we could run the kernel in HW.
   *  This includes CB update and resource binding
   *
   *  \return True if we succefully loaded the arguments
   */
  bool loadParameters(VirtualGPU& gpu,            //!< virtual GPU device object
                      const amd::Kernel& kernel,  //!< AMD kernel object for execution
                      const_address params,       //!< pointer to the param's store
                      bool nativeMem              //!< Native memory objects
                      ) const;

  //! Binds the constant buffers associated with the kernel
  bool bindConstantBuffers(VirtualGPU& gpu) const;

  /*! \brief Runs the kernel on HW
   *
   *  \return True if we succefully executed the kernel
   */
  bool run(VirtualGPU& gpu,     //!< virtual GPU device object
           GpuEvent* gpuEvent,  //!< Pointer to the GPU event
           bool lastRun,        //!< Last run in the split execution
           bool lastDoppCmd,    //!< for last dopp submission kernel dispatch
           bool pfpaDoppCmd     //!< for PFPA dopp submission kernel dispatch
           ) const;

  //! Help function to debug the kernel output
  void debug(VirtualGPU& gpu  //!< virtual GPU device object
             ) const;

  //! Programs internal samplers defined inside the kernel
  bool setInternalSamplers(VirtualGPU& gpu  //!< Virtual GPU device object
                           ) const;

  //! Returns TRUE if we successfully retrieved the binary from CAL
  bool getCalBinary(void* binary,  //!< ISA binary code
                    size_t size    //!< ISA binary size
                    ) const;

  //! Returns CAL image size
  size_t getCalBinarySize() const;

  //! Returns GPU device object, associated with this kernel
  const Device& dev() const;

  //! Returns GPU device object, associated with this kernel
  const Program& prog() const;

  //! Binds global HW constant buffers
  bool bindGlobalHwCb(VirtualGPU& gpu,                 //!< Virtual GPU device object
                      VirtualGPU::GslKernelDesc* desc  //!< Kernel descriptor
                      ) const;

 protected:
  //! Initializes the kernel parameters for the abstraction layer
  bool initParameters();

  /*! \brief Creates constant buffer resources, associated with the kernel
   *
   *  \return TRUE if we succefully created constant buffers
   */
  bool initConstBuffers();

 private:
  //! Disable copy constructor
  Kernel(const Kernel&);

  //! Disable operator=
  Kernel& operator=(const Kernel&);

  //! \enum Fixed Metadata offsets
  enum MetadataOffsets {
    GlobalWorkitemOffset = 0,
    LocalWorkitemOffset = 1,
    GroupsOffset = 2,
    PrivateRingOffset = 3,
    LocalRingOffset = 4,
    MathLibOffset = 5,
    GlobalWorkOffsetOffset = 6,
    GroupWorkOffsetOffset = 7,
    GlobalDataStoreOffset = 8,
    DebugOffset = 8,
    NDRangeGlobalWorkOffsetOffset = 9,

    // The total number of constants reserved for ABI
    TotalABIVectors
  };

  /*! \brief Sets the kernel argument
   *
   *  \return True if we succefully updated the arguments
   */
  bool setArgument(VirtualGPU& gpu,     //!< Virtual GPU device object
                   const amd::Kernel& kernel, //!< AMD kernel object
                   uint idx,            //!< the argument index
                   const_address params,//!< the arguments data
                   const amd::KernelParameterDescriptor& desc, //!< Argument's descriptor
                   bool nativeMem      //!< Native memory objects
                   ) const;

  /*! \brief Initializes local and private buffer ranges
   *
   *  \return True if we succefully initialized the ranges
   */
  bool initLocalPrivateRanges(VirtualGPU& gpu  //!< Virtual GPU device object
                              ) const;

  //! Sets local and private buffer ranges
  void setLocalPrivateRanges(VirtualGPU& gpu  //!< Virtual GPU device object
                             ) const;

  //! Sets the sampler's parameters for the image look-up
  void setSampler(VirtualGPU& gpu,  //!< virtual GPU device object
                  uint32_t state,   //!< sampler state
                  uint physUnit     //!< sampler's number
                  ) const;

  /*! \brief Binds resource
   *
   *  \return True if we succefully created constant buffers
   */
  bool bindResource(VirtualGPU& gpu,       //!< virtual GPU device object
                    const Memory& memory,  //!< memory for binding
                    uint paramIdx,         //!< index of the parameter
                    ResourceType type,     //!< resource type
                    uint physUnit,         //!< PhysUnit
                    size_t offset = 0) const;

  //! Unbinds all resources for the kernel
  void unbindResources(VirtualGPU& gpu,    //!< virtual GPU device object
                       GpuEvent gpuEvent,  //!< GPU event that will be associated with the resources
                       bool lastRun        //!< last run in the split execution
                       ) const;

  //! Copies image constants to the constant buffer
  void copyImageConstants(const amd::Image* amdImage,  //!< Abstraction layer image object
                          ImageConstants* imageData    //!< Pointer in CB to the image constants
                          ) const;

  //! Finds local workgroup size
  void findLocalWorkSize(size_t workDim,                   //!< Work dimension
                         const amd::NDRange& gblWorkSize,  //!< Global work size
                         amd::NDRange& lclWorkSize         //!< Local work size
                         ) const;

  uint hwPrivateSize_;  //!< initial HW private size
  uint hwLocalSize_;    //!< initial HW local size
};

enum HSAIL_ADDRESS_QUALIFIER {
  HSAIL_ADDRESS_ERROR = 0,
  HSAIL_ADDRESS_GLOBAL,
  HSAIL_ADDRESS_LOCAL,
  HSAIL_MAX_ADDRESS_QUALIFIERS
};

enum HSAIL_ARG_TYPE {
  HSAIL_ARGTYPE_ERROR = 0,
  HSAIL_ARGTYPE_POINTER,
  HSAIL_ARGTYPE_VALUE,
  HSAIL_ARGTYPE_IMAGE,
  HSAIL_ARGTYPE_SAMPLER,
  HSAIL_ARGTYPE_QUEUE,
  HSAIL_ARGMAX_ARG_TYPES
};

enum HSAIL_DATA_TYPE {
  HSAIL_DATATYPE_ERROR = 0,
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

class HSAILKernel : public device::Kernel {
 public:
  struct Argument {
    std::string name_;                  //!< Argument's name
    std::string typeName_;              //!< Argument's type name
    uint size_;                         //!< Size in bytes
    uint offset_;                       //!< Argument's offset
    uint alignment_;                    //!< Argument's alignment
    HSAIL_ARG_TYPE type_;               //!< Type of the argument
    HSAIL_ADDRESS_QUALIFIER addrQual_;  //!< Address qualifier of the argument
    HSAIL_DATA_TYPE dataType_;          //!< The type of data
    uint numElem_;                      //!< Number of elements
    HSAIL_ACCESS_TYPE access_;          //!< Access type for the argument
  };

  // Max number of possible extra (hidden) kernel arguments
  static const uint MaxExtraArgumentsNum = 6;

  HSAILKernel(std::string name, HSAILProgram* prog, std::string compileOptions, uint extraArgsNum);

  virtual ~HSAILKernel();

  //! Initializes the metadata required for this kernel,
  //! finalizes the kernel if needed
  bool init(amd::hsa::loader::Symbol* sym, bool finalize = false);

  //! Returns a pointer to the hsail argument
  const Argument* argument(size_t i) const { return arguments_[i]; }

  //! Returns the number of hsail arguments
  size_t numArguments() const { return arguments_.size(); }

  //! Returns GPU device object, associated with this kernel
  const Device& dev() const;

  //! Returns HSA program associated with this kernel
  const HSAILProgram& prog() const;

  //! Returns LDS size used in this kernel
  uint32_t ldsSize() const { return cpuAqlCode_->workgroup_group_segment_byte_size; }

  //! Returns pointer on CPU to AQL code info
  const void* cpuAqlCode() const { return cpuAqlCode_; }

  //! Returns memory object with AQL code
  gpu::Memory* gpuAqlCode() const { return code_; }

  //! Returns size of AQL code
  size_t aqlCodeSize() const { return codeSize_; }

  //! Returns the size of argument buffer
  size_t argsBufferSize() const { return cpuAqlCode_->kernarg_segment_byte_size; }

  //! Returns spill reg size per workitem
  int spillSegSize() const { return cpuAqlCode_->workitem_private_segment_byte_size; }

  //! Returns AQL packet in CPU memory
  //! if the kerenl arguments were successfully loaded, otherwise NULL
  hsa_kernel_dispatch_packet_t* loadArguments(
      VirtualGPU& gpu,                     //!< Running GPU context
      const amd::Kernel& kernel,           //!< AMD kernel object
      const amd::NDRangeContainer& sizes,  //!< NDrange container
      const_address parameters,            //!< Application arguments for the kernel
      bool nativeMem,                      //!< Native memory objectes are passed
      uint64_t vmDefQueue,                 //!< GPU VM default queue pointer
      uint64_t* vmParentWrap,              //!< GPU VM parent aql wrap object
      std::vector<const Memory*>& memList  //!< Memory list for GSL/VidMM handles
      ) const;

  //! Returns the kernel index in the program
  uint index() const { return index_; }

  //! Returns kernel's extra argument count
  uint extraArgumentsNum() const { return extraArgumentsNum_; }

 private:
  //! Disable copy constructor
  HSAILKernel(const HSAILKernel&);

  //! Disable operator=
  HSAILKernel& operator=(const HSAILKernel&);

  //! Creates AQL kernel HW info
  bool aqlCreateHWInfo(amd::hsa::loader::Symbol* sym);

  //! Initializes arguments_ and the abstraction layer kernel parameters
  void initArgList(const aclArgData* aclArg  //!< List of ACL arguments
                   );

  //! Initializes Hsail Argument metadata and info
  void initHsailArgs(const aclArgData* aclArg  //!< List of ACL arguments
                     );

  std::vector<Argument*> arguments_;  //!< Vector list of HSAIL Arguments
  std::string compileOptions_;        //!< compile used for finalizing this kernel
  amd_kernel_code_t* cpuAqlCode_;     //!< AQL kernel code on CPU
  uint index_;                        //!< Kernel index in the program

  gpu::Memory* code_;  //!< Memory object with ISA code
  size_t codeSize_;    //!< Size of ISA code

  char* hwMetaData_;  //!< SI metadata

  uint extraArgumentsNum_;  //! Number of extra (hidden) kernel arguments
};

/*@}*/} // namespace gpu

#endif /*GPUKERNEL_HPP_*/
