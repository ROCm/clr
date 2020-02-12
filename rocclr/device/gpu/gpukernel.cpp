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

#include "device/gpu/gpukernel.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpublit.hpp"
#include "device/gpu/gpuconstbuf.hpp"
#include "device/gpu/gpusched.hpp"
#include "platform/commandqueue.hpp"
#include "shader/ComputeProgramObject.h"
#include "utils/options.hpp"

#include "acl.h"
#include "SCCommon.h"

#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <algorithm>

namespace gpu {

const MetaDataConst ArgState[ArgStateTotal] = {
    // Note: the order is important
    // Name                 Type                    Properties
    // Kernel description (special properties)
    {"memory:compilerwrite", KernelArg::PrivateFixed, {0, 0, 0, 0, 0, 0, 0}},
    {"uniqueid:", KernelArg::NoType, {0, 0, 0, 0, 0, 0, 0}},
    {"memory:private:", KernelArg::PrivateSize, {0, 0, 0, 0, 0, 0, 0}},
    {"memory:local:", KernelArg::LocalSize, {0, 0, 0, 0, 0, 0, 0}},
    {"memory:hwprivate:", KernelArg::HwPrivateSize, {0, 0, 0, 0, 0, 0, 0}},
    {"memory:uavprivate:", KernelArg::HwPrivateSize, {0, 0, 0, 0, 0, 0, 0}},
    {"memory:hwlocal:", KernelArg::HwLocalSize, {0, 0, 0, 0, 0, 0, 0}},
    {"memory:64bitABI", KernelArg::ABI64Bit, {0, 0, 0, 0, 0, 0, 0}},
    {"limitgroupsize", KernelArg::Wavefront, {0, 0, 0, 0, 0, 0, 0}},
    {"function:", KernelArg::NoType, {1, 1, 0, 0, 0, 0, 0}},
    {"intrinsic:", KernelArg::NoType, {1, 0, 0, 0, 0, 0, 0}},
    {"error:", KernelArg::ErrorMessage, {0, 0, 0, 0, 0, 0, 0}},
    {"warning:", KernelArg::WarningMessage, {0, 0, 0, 0, 0, 0, 0}},
    {"printf_fmt:", KernelArg::PrintfFormatStr, {0, 0, 0, 0, 0, 0, 0}},
    {"version:", KernelArg::MetadataVersion, {0, 0, 0, 0, 0, 0, 0}},
    // Kernel basic types
    {"pointer:", KernelArg::PointerGlobal, {1, 1, 1, 1, 1, 1, 0}},
    {"value:", KernelArg::Value, {1, 1, 1, 1, 1, 0, 0}},
    {"image:", KernelArg::Image, {1, 1, 1, 1, 1, 0, 0}},
    {"sampler:", KernelArg::Sampler, {0, 1, 0, 0, 0, 0, 0}},
    {"counter:", KernelArg::Counter, {1, 1, 0, 1, 1, 0, 0}},
    {"cws:", KernelArg::Grouping, {0, 0, 0, 0, 0, 0, 0}},
    {"lws:", KernelArg::WrkgrpSize, {0, 0, 0, 0, 0, 0, 0}},
    {"uavid:", KernelArg::UavId, {0, 0, 0, 0, 0, 0, 0}},
    {"reflection:", KernelArg::Reflection, {0, 0, 0, 0, 0, 0, 0}},
    {"constarg:", KernelArg::ConstArg, {0, 0, 0, 0, 0, 0, 0}},
    {"cbid:", KernelArg::ConstBufId, {0, 0, 0, 0, 0, 0, 0}},
    {"printfid:", KernelArg::PrintfBufId, {0, 0, 0, 0, 0, 0, 0}},
    {"wsh:", KernelArg::GroupingHint, {0, 0, 0, 0, 0, 0, 0}},
    {"vth:", KernelArg::VecTypeHint, {0, 0, 0, 0, 0, 0, 0}},
    {"WavesPerSimdHint:", KernelArg::WavesPerSimdHint, {0, 0, 0, 0, 0, 0, 0}},
};

const DataTypeConst DataType[] = {
    {
        "i8:", KernelArg::Char,
    },
    {
        "i16:", KernelArg::Short,
    },
    {
        "i32:", KernelArg::Int,
    },
    {
        "i64:", KernelArg::Long,
    },
    {
        "u8:", KernelArg::UChar,
    },
    {
        "u16:", KernelArg::UShort,
    },
    {
        "u32:", KernelArg::UInt,
    },
    {
        "u64:", KernelArg::ULong,
    },
    {
        "float:", KernelArg::Float,
    },
    {
        "double:", KernelArg::Double,
    },
    {
        "struct:", KernelArg::Struct,
    },
    {
        "union:", KernelArg::Union,
    },
    {
        "1D:", KernelArg::Image1D,
    },
    {
        "2D:", KernelArg::Image2D,
    },
    {
        "3D:", KernelArg::Image3D,
    },
    {
        "1DB:", KernelArg::Image1DB,
    },
    {
        "1DA:", KernelArg::Image1DA,
    },
    {
        "2DA:", KernelArg::Image2DA,
    },
    {
        "opaque:", KernelArg::Opaque,
    },
    {
        "event:", KernelArg::Event,
    },
    {
        "sampler:", KernelArg::Sampler,
    },
    {
        "half:", KernelArg::Half,
    },
};

const uint DataTypeTotal = sizeof(DataType) / sizeof(DataTypeConst);

struct BufDataConst {
  const char* tagName_;           //!< buffer's name
  KernelArg::ArgumentType type_;  //!< type of argument
  struct {
    uint number_ : 1;     //!< buffer's number
    uint alignment_ : 1;  //!< buffer's alignment
    uint attribute_ : 1;  //!< buffer's read/write attribute
    uint reserved : 29;   //!< reserved
  };
};

static const BufDataConst BufType[] = {{"g", KernelArg::PointerGlobal, {1, 0, 0, 0}},
                                       {"p", KernelArg::PointerPrivate, {1, 1, 1, 0}},
                                       {"l", KernelArg::PointerLocal, {1, 1, 1, 0}},
                                       {"uav", KernelArg::PointerGlobal, {1, 1, 1, 0}},
                                       {"c", KernelArg::PointerConst, {1, 1, 1, 0}},
                                       {"hl", KernelArg::PointerHwLocal, {1, 1, 1, 0}},
                                       {"hp", KernelArg::PointerHwPrivate, {1, 1, 1, 0}},
                                       {"hc", KernelArg::PointerHwConst, {1, 1, 1, 0}}};
static const uint BufTypeTotal = sizeof(BufType) / sizeof(BufDataConst);

//! The mathlib constants for each kernel execution
static const float MathLibConst[4] = {0.0f, 0.5f, 1.0f, 2.0f};

bool expect(const std::string& str, size_t* pos, const std::string& sym) {
  bool result = true;
  uint i;

  if (*pos == std::string::npos) {
    return false;
  }

  // Check if we have expected symbols
  for (i = 0; i < sym.size(); ++i) {
    char deb = str[*pos + i];
    if (deb != sym[i]) {
      result = false;
      break;
    }
  }

  if (result) *pos += i;

  return result;
}

bool getword(const std::string& str, size_t* pos, std::string& sym) {
  if (*pos == std::string::npos) {
    return false;
  }

  *pos = str.find_first_not_of(" \n\r", *pos);
  size_t posEnd = str.find_first_of(": \n\r;", *pos);
  size_t count = posEnd - *pos;

  if (count != 0) {
    sym = str.substr(*pos, count);
  }
  sym[count] = 0;
  *pos = posEnd + 1;
  return true;
}

bool getstring(const std::string& str, size_t* pos, std::string* out) {
  if (*pos == std::string::npos) {
    return false;
  }

  *pos = str.find_first_not_of(" \n\r", *pos);
  size_t posEnd = str.find_first_of(":\n\r;", *pos);
  size_t count = posEnd - *pos;

  char* sym = new char[count + 1];
  if (count != 0) {
    if (!str.copy(sym, count, *pos)) {
      return false;
    }
  }
  sym[count] = 0;
  *out = sym;
  delete[] sym;
  *pos = posEnd + 1;
  return true;
}

bool getuint(const std::string& str, size_t* pos, uint* val) {
  if (*pos == std::string::npos) {
    return false;
  }

  char sym[16];
  *pos = str.find_first_not_of(" \n\r", *pos);
  size_t posEnd = str.find_first_of(": \n\r;)", *pos);

  if (!str.copy(sym, posEnd - *pos, *pos)) {
    return false;
  }
  *val = 0;
  for (size_t i = 0; i < (posEnd - *pos); ++i) {
    *val = (*val * 10) + (sym[i] - 0x30);
  }
  *pos = posEnd + 1;
  return true;
}

bool getuintHex(const std::string& str, size_t* pos, uint* val) {
  if (*pos == std::string::npos) {
    return false;
  }

  char sym[16];
  *pos = str.find_first_not_of(" \n\r", *pos);
  size_t posEnd = str.find_first_of(": \n\r;)", *pos);

  if (!str.copy(sym, posEnd - *pos, *pos)) {
    return false;
  }
  *val = 0;
  for (size_t i = 0; i < (posEnd - *pos); ++i) {
    if (sym[i] >= '0' && sym[i] <= 'F') {
      *val = (*val * 16) + (sym[i] - '0');
    } else if (sym[i] >= 'a' && sym[i] <= 'f') {
      *val = (*val * 16) + (sym[i] - 'a' + 10);
    } else {
      return false;
    }
  }
  *pos = posEnd + 1;
  return true;
}

bool getuint64Hex(const std::string& str, size_t* pos, uint64_t* val) {
  if (*pos == std::string::npos) {
    return false;
  }

  char sym[16];
  *pos = str.find_first_not_of(" \n\r", *pos);
  size_t posEnd = str.find_first_of(": \n\r;)", *pos);

  if (!str.copy(sym, posEnd - *pos, *pos)) {
    return false;
  }
  *val = 0;
  for (size_t i = 0; i < (posEnd - *pos); ++i) {
    if (sym[i] >= '0' && sym[i] <= 'F') {
      *val = (*val * 16) + (sym[i] - '0');
    } else if (sym[i] >= 'a' && sym[i] <= 'f') {
      *val = (*val * 16) + (sym[i] - 'a' + 10);
    } else {
      return false;
    }
  }
  *pos = posEnd + 1;
  return true;
}

void intToStr(size_t value, char* str, size_t size) {
  static const uint MaxDigits32bit = 10;
  char result[MaxDigits32bit];
  uint idx = MaxDigits32bit;

  do {
    idx--;
    result[idx] = static_cast<char>((value % 10) + '0');
    value /= 10;
  } while ((value != 0) && (idx > 0));
  size_t len = MaxDigits32bit - idx;
  size_t n = std::min<size_t>(len, size - 1);
  memcpy(str, &result[idx], n);
  str[n] = '\0';
}

//! Default destructor
CalImageReference::~CalImageReference() {
  // Free CAL image
  free(image_);
}

KernelArg::KernelArg()
    : type_(KernelArg::NoType),
      size_(0),
      cbIdx_(0),
      cbPos_(0),
      index_(0),
      alignment_(1),
      dataType_(KernelArg::NoType) {
  name_ = "";
  buf_ = "";
  memory_.value_ = 0;
  typeQualifier_ = CL_KERNEL_ARG_TYPE_NONE;
}

KernelArg::KernelArg(const KernelArg& data) {
  // Fill the new object
  *this = data;
}

KernelArg& KernelArg::operator=(const KernelArg& data) {
  // Fill the fields of the current object
  name_ = data.name_;
  typeName_ = data.typeName_;
  typeQualifier_ = data.typeQualifier_;
  type_ = data.type_;
  size_ = data.size_;
  cbIdx_ = data.cbIdx_;
  cbPos_ = data.cbPos_;
  buf_ = data.buf_;
  index_ = data.index_;
  alignment_ = data.alignment_;
  dataType_ = data.dataType_;
  memory_.value_ = data.memory_.value_;
  return *this;
}

bool KernelArg::isCbNeeded() const {
  //! \note not a safe way
  bool result = ((type_ > NoType) && (type_ < Sampler)) ? true : false;
  if ((type_ == Sampler) && (location_ == 0)) {
    // Sampler is defined outside the kernel
    result = true;
  }
  return result;
}

size_t KernelArg::size(bool gpuLayer) const {
  switch (type_) {
    case NoType:
      return 0;
    case PointerConst:
    case PointerHwConst:
    case PointerGlobal:
      return (gpuLayer) ? sizeof(uint32_t) * size_ : sizeof(cl_mem);
    case Image1D:
    case Image2D:
    case Image3D:
    case Image1DB:
    case Image1DA:
    case Image2DA:
      return (gpuLayer) ? sizeof(ImageConstants) : sizeof(cl_mem);
    case Sampler:
      return (gpuLayer) ? 2 * sizeof(uint32_t) : sizeof(cl_sampler);
    case Counter:
      return (gpuLayer) ? 0 : sizeof(cl_mem);
    case PointerLocal:
    case PointerHwLocal:
      return (gpuLayer) ? sizeof(uint32_t) * size_ : sizeof(cl_mem);
    case PointerPrivate:
    case PointerHwPrivate:
      return (gpuLayer) ? sizeof(uint32_t) * size_ : 0;
    case Float:
      return sizeof(float) * amd::nextPowerOfTwo(size_);
    case Double:
      return sizeof(double) * amd::nextPowerOfTwo(size_);
    case Char:
    case UChar:
      return sizeof(cl_char) * amd::nextPowerOfTwo(size_);
    case Short:
    case UShort:
      return sizeof(cl_short) * amd::nextPowerOfTwo(size_);
    case Int:
    case UInt:
      return sizeof(uint32_t) * amd::nextPowerOfTwo(size_);
    case Long:
    case ULong:
      return sizeof(uint64_t) * amd::nextPowerOfTwo(size_);
    case Struct:
    case Union:
      return (gpuLayer) ? amd::alignUp(size_, 16) : size_;
    default:
      return 0;
  }
}

cl_kernel_arg_address_qualifier KernelArg::addressQualifier() const {
  switch (type_) {
    case PointerGlobal:
    case Image1D:
    case Image2D:
    case Image3D:
    case Image1DB:
    case Image1DA:
    case Image2DA:
      return CL_KERNEL_ARG_ADDRESS_GLOBAL;
    case PointerLocal:
    case PointerHwLocal:
      return CL_KERNEL_ARG_ADDRESS_LOCAL;
    case PointerConst:
    case PointerHwConst:
      return CL_KERNEL_ARG_ADDRESS_CONSTANT;
    default:
      return CL_KERNEL_ARG_ADDRESS_PRIVATE;
  }
}

cl_kernel_arg_access_qualifier KernelArg::accessQualifier() const {
  switch (type_) {
    case Image1D:
    case Image2D:
    case Image3D:
    case Image1DB:
    case Image1DA:
    case Image2DA:
      if (memory_.readOnly_) {
        return CL_KERNEL_ARG_ACCESS_READ_ONLY;
      } else if (memory_.writeOnly_) {
        return CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
      } else if (memory_.readWrite_) {
        return CL_KERNEL_ARG_ACCESS_READ_WRITE;
      }
    // Fall through ...
    default:
      return CL_KERNEL_ARG_ACCESS_NONE;
  }
}

//! temporary solution for the vectors handling in compiler
size_t KernelArg::specialVector() const {
  if (size_ > VectorSizeLimit) {
    switch (type_) {
      case Char:
      case UChar:
        return sizeof(cl_char);
      case Short:
      case UShort:
        return sizeof(cl_short);
      default:
        return 0;
    }
  }
  return 0;
}
clk_value_type_t KernelArg::type() const {
  switch (type_) {
    case PointerGlobal:
    case PointerLocal:
    case PointerHwLocal:
    case PointerConst:
    case PointerHwConst:
    case Image1D:
    case Image2D:
    case Image3D:
    case Image1DB:
    case Image1DA:
    case Image2DA:
    case Counter:
      return T_POINTER;
    case Float:
      return T_FLOAT;
    case Double:
      return T_DOUBLE;
    case Char:
    case UChar:
      return T_CHAR;
    case Short:
    case UShort:
      return T_SHORT;
    case Int:
      return T_INT;
    case UInt:
      //! \note No UINT type
      return T_INT;
    case Long:
      return T_LONG;
    case ULong:
      //! \note No ULONG type
      return T_LONG;
    case Struct:
    case Union:
      //! @todo What should we report?
      return T_CHAR;
    case Sampler:
      return T_SAMPLER;
    case PointerPrivate:
    case PointerHwPrivate:
    case NoType:
    default:
      return T_VOID;
  }
}

NullKernel::NullKernel(const std::string& name, const NullDevice& gpuNullDev,
                       const NullProgram& nullprog)
    : device::Kernel(gpuNullDev, name, nullprog),
      buildError_(CL_BUILD_PROGRAM_FAILURE),
      gpuDev_(gpuNullDev),
      calRef_(NULL),
      internal_(false),
      flags_(0),
      cbSizes_(NULL),
      numCb_(0),
      rwAttributes_(false),
      instructionCnt_(4) {
  // UAV raw index will be detected
  uavRaw_ = UavIdUndefined;
  // CB index will be detected
  cbId_ = UavIdUndefined;
  // Printf index will be detected
  printfId_ = UavIdUndefined;
}

NullKernel::~NullKernel() {
  uint idx;

  if (calRef_ == NULL) {
    return;
  }
  calRef_->release();

  // Destroy all kernel arguments
  for (idx = 0; idx < arguments_.size(); ++idx) {
    delete arguments_[idx];
  }
  arguments_.clear();

  // Destroy all sampler kernel arguments
  for (idx = 0; idx < intSamplers_.size(); ++idx) {
    delete intSamplers_[idx];
  }
  intSamplers_.clear();
}


static int scComponentToArrayIndex(E_SC_COMPONENT dstComp) {
  switch (dstComp) {
    case SC_COMPONENT_X:
      return 0;
    case SC_COMPONENT_Y:
      return 1;
    case SC_COMPONENT_Z:
      return 2;
    case SC_COMPONENT_W:
      return 3;
  }

  return 0;
}

static void addLoopConst(const SC_HWSHADER* shader, AMUabiAddEncoding& encoding) {
  uint count = shader->dep.NumIntrlIConstants;
  encoding.litConstsCount = shader->dep.NumIntrlIConstants;

  // only suppport loop consts (int consts)
  if (count) {
    AMUabiLiteralConst* allocatedconsts = encoding.litConsts;
    memset(allocatedconsts, 0, count * sizeof(AMUabiLiteralConst));
    uint usedConsts = 0;
    for (uint i = 0; i < count; ++i) {
      uint currentConst;
      for (currentConst = 0; currentConst < usedConsts; ++currentConst) {
        if (allocatedconsts[currentConst].addr ==
            HWSHADER_Get(shader, dep.IntrlIConstants)[i].uDstNumber) {
          break;
        }
      }
      if (currentConst == usedConsts) {
        usedConsts++;
        assert(usedConsts <= count);
      }
      allocatedconsts[currentConst].addr = HWSHADER_Get(shader, dep.IntrlIConstants)[i].uDstNumber;
      allocatedconsts[currentConst].type = AMU_ABI_INT32;
      allocatedconsts[currentConst].value.int32[scComponentToArrayIndex(
          HWSHADER_Get(shader, dep.IntrlIConstants)[i].eDstComp)] =
          HWSHADER_Get(shader, dep.IntrlIConstants)[i].iValue;
    }
    encoding.litConstsCount = usedConsts;
  }
}

bool NullKernel::create(const std::string& code, const std::string& metadata,
                        const void* binaryCode, size_t binarySize) {
  std::auto_ptr<uint> uavRefCount(new uint[MaxUavArguments]);
  if (NULL == uavRefCount.get()) {
    return false;
  }

  // Set all ref counts to 0
  memset(uavRefCount.get(), 0, sizeof(uavRefCount.get()[0]) * MaxUavArguments);

  // parse the metadata fields
  if (!parseArguments(metadata, uavRefCount.get())) {
    return false;
  }

  CALimage calImage;
// Save source if DEBUG build
#if DEBUG
  ilSource_ = code;
#endif  // DEBUG

  amd::option::Options* options = nullProg().getCompilerOptions();
  internal_ = options->oVariables->clInternalKernel;

  if ((binaryCode == NULL) && (binarySize == 0) && !code.empty()) {
    acl_error err;
    std::string arch = "amdil";
    if (nullDev().settings().use64BitPtr_) {
      arch += "64";
    }
    aclTargetInfo info = aclGetTargetInfo(arch.c_str(), nullDev().hwInfo()->targetName_, &err);
    if (err != ACL_SUCCESS) {
      LogWarning("aclGetTargetInfo failed");
      return false;
    }

    aclBinaryOptions binOpts = {0};
    binOpts.struct_size = sizeof(binOpts);
    binOpts.elfclass = info.arch_id == aclAMDIL64 ? ELFCLASS64 : ELFCLASS32;
    binOpts.bitness = ELFDATA2LSB;
    binOpts.alloc = &::malloc;
    binOpts.dealloc = &::free;

    aclBinary* bin = aclBinaryInit(sizeof(aclBinary), &info, &binOpts, &err);
    if (err != ACL_SUCCESS) {
      LogWarning("aclBinaryInit failed");
      return false;
    }

    if (ACL_SUCCESS !=
        aclInsertSection(nullDev().amdilCompiler(), bin, code.data(), code.size(), aclSOURCE)) {
      LogWarning("aclInsertSection failed");
      aclBinaryFini(bin);
      return false;
    }

    amd::option::Options* Opts = (amd::option::Options*)bin->options;

    // Append an option so that we can selectively enable a SCOption on CZ
    // whenever IOMMUv2 is enabled.
    if (nullDev().settings().svmFineGrainSystem_) {
      options->origOptionStr.append(" -sc-xnack-iommu");
    }
    // temporary solution to synchronize buildNo between runtime and complib
    // until we move runtime inside complib
    Opts->setBuildNo(options->getBuildNo());

    // pass kernel name to compiler
    Opts->setCurrKernelName(name().c_str());

    err = aclCompile(nullDev().amdilCompiler(), bin, options->origOptionStr.c_str(), ACL_TYPE_AMDIL_TEXT,
                     ACL_TYPE_ISA, NULL);

    buildLog_ += aclGetCompilerLog(nullDev().amdilCompiler());

    if (err != ACL_SUCCESS) {
      LogWarning("aclCompile failed");
      aclBinaryFini(bin);
      return false;
    }
    if (!options->oVariables->BinEXE) {
      // Early exit if binary doesn't contain EXE
      aclBinaryFini(bin);
      return true;
    }
    size_t len;
    const void* isa = aclExtractSection(nullDev().amdilCompiler(), bin, &len, aclTEXT, &err);
    if (err != ACL_SUCCESS) {
      LogWarning("aclExtractSection failed");
      aclBinaryFini(bin);
      return false;
    }

    uint calImageSize;
    if (!createMultiBinary(&calImageSize, reinterpret_cast<void**>(&calImage), isa)) {
      LogWarning("initSrcEncoding failed");
      aclBinaryFini(bin);
      return false;
    }

    aclBinaryFini(bin);
  } else if ((binaryCode != NULL) && (binarySize != 0)) {
    uint size = 0;
    if (!amuABIMultiBinaryGetSize(&size, const_cast<void*>(binaryCode)) || size > binarySize) {
      buildLog_ += "Invalid binary image";
      LogError("amuABIMultiBinaryGetSize failed!");
      return false;
    }

    calImage = static_cast<CALimage>(malloc(size));
    ::memcpy(calImage, binaryCode, size);
  } else {
    LogError("Incorrect initialization parameters!");
    return false;
  }

  calRef_ = new CalImageReference(calImage);
  if (calRef_ == NULL) {
    LogError("Memory allocation failure!");
    // Free CAL image
    free(calImage);
    return false;
  }

  CALfuncInfo calFuncInfo;

  // Get kernel compiled information
  getFuncInfoFromImage(calImage, &calFuncInfo);
  if (calFuncInfo.maxScratchRegsNeeded > 0) {
    LogPrintfInfo(
        "%s kernel has register spilling."
        "Lower performance is expected.",
        name().c_str());
  }

  workGroupInfo_.scratchRegs_ = calFuncInfo.maxScratchRegsNeeded;
  workGroupInfo_.wavefrontPerSIMD_ = calFuncInfo.numWavefrontPerSIMD;
  workGroupInfo_.wavefrontSize_ = calFuncInfo.wavefrontSize;
  workGroupInfo_.availableGPRs_ = calFuncInfo.numGPRsAvailable;
  workGroupInfo_.usedGPRs_ = calFuncInfo.numGPRsUsed;
  workGroupInfo_.availableSGPRs_ = calFuncInfo.numSGPRsAvailable;
  workGroupInfo_.usedSGPRs_ = calFuncInfo.numSGPRsUsed;
  workGroupInfo_.availableVGPRs_ = calFuncInfo.numVGPRsAvailable;
  workGroupInfo_.usedVGPRs_ = calFuncInfo.numVGPRsUsed;
  workGroupInfo_.availableLDSSize_ = calFuncInfo.LDSSizeAvailable;
  workGroupInfo_.usedLDSSize_ = calFuncInfo.LDSSizeUsed;
  workGroupInfo_.availableStackSize_ = calFuncInfo.stackSizeAvailable;
  workGroupInfo_.usedStackSize_ = calFuncInfo.stackSizeUsed;

  device::Kernel::parameters_t params;
  if (!createSignature(params, params.size(), amd::KernelSignature::ABIVersion_0)) {
    return false;
  }

  return true;
}

size_t NullKernel::getCalBinarySize() const {
  CALuint imageSize;
  if (!amuABIMultiBinaryGetSize(&imageSize, calImage())) {
    LogError("Failed to get the image size!");
    return 0;
  }
  return static_cast<size_t>(imageSize);
}

bool NullKernel::getCalBinary(void* binary, size_t size) const {
  uint calImageSize = 0;
  if (!amuABIMultiBinaryGetSize(&calImageSize, calImage()) || size < calImageSize) {
    LogError("CAL failed to save the kernel binary!");
    return false;
  }
  ::memcpy(binary, calImage(), calImageSize);

  return true;
}

bool Kernel::create(const std::string& code, const std::string& metadata, const void* binaryCode,
                    size_t binarySize) {
  setPreferredSizeMultiple(dev().getAttribs().wavefrontSize);

  if (!NullKernel::create(code, metadata, binaryCode, binarySize)) {
    return false;
  }

  // initialize constant buffer sizes
  if (!initConstBuffers()) {
    return false;
  }

  // Initialize the kernel parameters
  bool result = initParameters();

  // Wave limiter needs to be initialized after kernel metadata is parsed
  // Since it depends on it.
  waveLimiter_.enable(dev().settings().ciPlus_);

  if (result) {
    buildError_ = CL_SUCCESS;
  } else {
    result = false;
  }

  return result;
}

Kernel::Kernel(const std::string& name, const Device& gpuDev, const Program& prog,
               const InitData* initData)
    : NullKernel(name, gpuDev, prog) {
  hwPrivateSize_ = 0;
  if (NULL != initData) {
    flags_ = initData->flags_;
    hwPrivateSize_ = initData->hwPrivateSize_;
    hwLocalSize_ = initData->hwLocalSize_;
  }
  // Workgroup info private memory size
  workGroupInfo_.privateMemSize_ = hwPrivateSize_;
  // Default wavesPerSimdHint_
  workGroupInfo_.wavesPerSimdHint_ = ~0U;
}

Kernel::~Kernel() {
  if (calRef_ == NULL) {
    return;
  }

  {
    Device::ScopedLockVgpus lock(dev());

    // Release all virtual image objects on all virtual GPUs
    for (uint idx = 0; idx < dev().vgpus().size(); ++idx) {
      dev().vgpus()[idx]->releaseKernel(calImage());
    }
  }

  if (0 != numCb_) {
    delete[] cbSizes_;
  }
}

const Device& Kernel::dev() const { return reinterpret_cast<const Device&>(gpuDev_); }

const Program& Kernel::prog() const { return reinterpret_cast<const Program&>(prog_); }

bool NullKernel::createMultiBinary(uint* imageSize, void** image, const void* isa) {
  const SC_HWSHADER* shader = reinterpret_cast<const SC_HWSHADER*>(isa);

  bool result = false;
  AMUabiAddEncoding encoding;
  memset(&encoding, 0, sizeof(AMUabiAddEncoding));

  size_t allocSize = sizeof(uint) * MaxReadImage + sizeof(CALUavEntry) * MaxUavArguments +
      sizeof(CALSamplerMapEntry) * MaxSamplers + sizeof(CALConstantBufferMask) * MaxConstBuffers +
      sizeof(AMUabiLiteralConst) * shader->dep.NumIntrlIConstants;
  char* tmpMem = new char[allocSize];
  if (tmpMem == NULL) {
    LogError("Error allocating memory");
    return false;
  }

  CalcPtr(encoding.inputs, tmpMem, 0, 0);
  CalcPtr(encoding.uav, encoding.inputs, sizeof(uint), MaxReadImage);
  CalcPtr(encoding.inputSamplerMaps, encoding.uav, sizeof(CALUavEntry), MaxUavArguments);
  CalcPtr(encoding.constBuffers, encoding.inputSamplerMaps, sizeof(CALSamplerMapEntry),
          MaxSamplers);
  if (shader->dep.NumIntrlIConstants != 0) {
    CalcPtr(encoding.litConsts, encoding.constBuffers, sizeof(CALConstantBufferMask),
            MaxConstBuffers);
  }
  AMUabiMultiBinary amuBinary;
  amuABIMultiBinaryCreate(&amuBinary);

  result = siCreateHwInfo(shader, encoding);
  if (!result) {
    delete[] tmpMem;
    LogWarning("Error Creating program info");
    return false;
  }

  addLoopConst(shader, encoding);

  unsigned int outputCount = 0, condOut = 0, earlyExit = 0, globalCount = 0, persistentCount = 0;
  unsigned int symbolCount = 0;
  CALOutputEntry* outputs = 0;
  unsigned int* globalBuffers = 0;
  unsigned int* persistentBuffers = 0;
  AMUabiUserSymbol* symbols = 0;

  CALSamplerMapEntry* inputSamplers = encoding.inputSamplerMaps;
  CALConstantBufferMask* constBuffers = encoding.constBuffers;
  uint* inputResources = encoding.inputs;
  CALUavEntry* uav = encoding.uav;

  uint inputSamplerCount = samplerSize();
  for (uint i = 0; i < inputSamplerCount; ++i) {
    inputSamplers[i].resource = 0;
    inputSamplers[i].sampler = sampler(i)->index_;
  }

  uint constBufferCount = 2;

  constBuffers[0].index = 0;
  constBuffers[1].index = 1;

  uint inputResourceCount = 0;

  uint uavCount = 0;
  bool cbBound = false;
  bool printfBound = false;
  for (uint i = 0; i < arguments_.size(); ++i) {
    const KernelArg* arg = argument(i);
    switch (arg->type_) {
      case KernelArg::PointerConst:
      case KernelArg::PointerHwConst:
        constBuffers[constBufferCount++].index = arg->index_;
        break;
      case KernelArg::PointerGlobal:
        uav[uavCount].offset = arg->index_;
        uav[uavCount].type = AMU_ABI_UAV_TYPE_TYPELESS;
        uav[uavCount].dimension = AMU_ABI_DIM_BUFFER;
        uav[uavCount].format = AMU_ABI_UAV_FORMAT_TYPELESS;
        uavCount++;
        break;
      case KernelArg::ConstBufId:
        if (!cbBound) {
          uav[uavCount].offset = cbId_;
          uav[uavCount].type = AMU_ABI_UAV_TYPE_RAW;
          uav[uavCount].dimension = AMU_ABI_DIM_BUFFER;
          uav[uavCount].format = AMU_ABI_UAV_FORMAT_TYPELESS;
          uavCount++;
        }
        cbBound = true;
        break;
      case KernelArg::PrintfBufId:
        if (!printfBound) {
          uav[uavCount].offset = printfId_;
          uav[uavCount].type = AMU_ABI_UAV_TYPE_RAW;
          uav[uavCount].dimension = AMU_ABI_DIM_BUFFER;
          uav[uavCount].format = AMU_ABI_UAV_FORMAT_TYPELESS;
          uavCount++;
        }
        printfBound = true;
        break;
      case KernelArg::UavId:
        if ((UavIdUndefined != uavRaw_) && !(flags() & PrintfOutput)) {
          uav[uavCount].offset = arg->index_;
          uav[uavCount].type = AMU_ABI_UAV_TYPE_TYPELESS;
          uav[uavCount].dimension = AMU_ABI_DIM_BUFFER;
          uav[uavCount].format = AMU_ABI_UAV_FORMAT_TYPELESS;
          uavCount++;
        } else {
          if (UavIdUndefined != uavRaw_) {
            uav[uavCount].offset = uavRaw_;
            uav[uavCount].type = AMU_ABI_UAV_TYPE_RAW;
            uav[uavCount].dimension = AMU_ABI_DIM_BUFFER;
            uav[uavCount].format = AMU_ABI_UAV_FORMAT_TYPELESS;
            uavCount++;
          }
        }
        break;
      case KernelArg::Sampler:
        inputSamplers[inputSamplerCount].resource = 0;
        inputSamplers[inputSamplerCount].sampler = arg->index_;
        inputSamplerCount++;
        break;
      case KernelArg::Image1D:
      case KernelArg::Image2D:
      case KernelArg::Image3D:
      case KernelArg::Image1DB:
      case KernelArg::Image1DA:
      case KernelArg::Image2DA:
        if (arg->memory_.readOnly_) {
          inputResources[inputResourceCount++] = arg->index_;
        } else {
          uav[uavCount].offset = arg->index_;
          uav[uavCount].type = AMU_ABI_UAV_TYPE_TYPED;
          uav[uavCount].dimension = AMU_ABI_DIM_2D;
          uav[uavCount].format = AMU_ABI_UAV_FORMAT_TYPELESS;
          uavCount++;
        }
        break;
      default:
        break;
    }
  }

  for (uint i = 0; i < nullProg().glbCb().size(); ++i) {
    constBuffers[constBufferCount++].index = nullProg().glbCb()[i];
  }

  encoding.machine = nullDev().hwInfo()->machine_;
  encoding.type = ED_ATI_CAL_TYPE_COMPUTE;
  encoding.inputCount = inputResourceCount;
  encoding.outputCount = outputCount;
  encoding.outputs = outputs;
  encoding.condOut = condOut;
  encoding.earlyExit = earlyExit;
  encoding.globalBuffersCount = globalCount;
  encoding.globalBuffers = globalBuffers;
  encoding.persistentBuffersCount = persistentCount;
  encoding.persistentBuffers = persistentBuffers;
  encoding.constBuffersCount = constBufferCount;
  encoding.inputSamplerMapCount = inputSamplerCount;
  encoding.symbolsCount = symbolCount;
  encoding.symbols = symbols;
  encoding.uavCount = uavCount;

  amuABIMultiBinaryAddEncoding(amuBinary, &encoding);

  uint success = amuABIMultiBinaryPack(imageSize, image, amuBinary);

  amuABIMultiBinaryDestroy(amuBinary);

  delete[] tmpMem;
  delete[] encoding.progInfos;

  return (success == 0) ? false : true;
}

void Kernel::findLocalWorkSize(size_t workDim, const amd::NDRange& gblWorkSize,
                               amd::NDRange& lclWorkSize) const {
  // Initialize the default workgoup info
  // Check if the kernel has the compiled sizes
  if (workGroupInfo()->compileSize_[0] == 0) {
    // Find the default local workgroup size, if it wasn't specified
    if (lclWorkSize[0] == 0) {
      if ((dev().settings().overrideLclSet & (1 << (workDim - 1))) == 0) {
        // Find threads per group
        size_t thrPerGrp = workGroupInfo()->size_;

        // Check if kernel uses images
        if ((flags() & ImageEnable) &&
            // and thread group is a multiple value of wavefronts
            ((thrPerGrp % workGroupInfo()->wavefrontSize_) == 0) &&
            // and it's 2 or 3-dimensional workload
            (workDim > 1) && ((gblWorkSize[0] % 16) == 0) && ((gblWorkSize[1] % 16) == 0)) {
          // Use 8x8 workgroup size if kernel has image writes
          if ((flags() & ImageWrite) || (thrPerGrp != nullDev().info().preferredWorkGroupSize_)) {
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

          // Check if we couldn't find optimal workload
          if (((lclWorkSize.product() % workGroupInfo()->wavefrontSize_) != 0) ||
               // or size is too small for the cache line
               (lclWorkSize[0] < cacheLineMatch)) {
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
            } else {
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

void Kernel::setupProgramGrid(VirtualGPU& gpu, size_t workDim, const amd::NDRange& glbWorkOffset,
                              const amd::NDRange& gblWorkSize, amd::NDRange& lclWorkSize,
                              const amd::NDRange& groupOffset, const amd::NDRange& glbWorkOffsetOrg,
                              const amd::NDRange& glbWorkSizeOrg) const {
  // ABI is always in CB0
  address cbBuf = gpu.cb(0)->sysMemCopy();
  uint* pGlobalSize =
      reinterpret_cast<uint*>(cbBuf + GlobalWorkitemOffset * ConstBuffer::VectorSize);
  uint* pLocalSize = reinterpret_cast<uint*>(cbBuf + LocalWorkitemOffset * ConstBuffer::VectorSize);
  uint* pNumGroups = reinterpret_cast<uint*>(cbBuf + GroupsOffset * ConstBuffer::VectorSize);
  uint* pGlobalOffset =
      reinterpret_cast<uint*>(cbBuf + GlobalWorkOffsetOffset * ConstBuffer::VectorSize);
  uint* pGroupOffset =
      reinterpret_cast<uint*>(cbBuf + GroupWorkOffsetOffset * ConstBuffer::VectorSize);
  uint32_t* debugInfo = reinterpret_cast<uint*>(cbBuf + DebugOffset * ConstBuffer::VectorSize);
  uint* pNDRangeGlobalOffset =
      reinterpret_cast<uint*>(cbBuf + NDRangeGlobalWorkOffsetOffset * ConstBuffer::VectorSize);

  // Check for 64-bit metadata
  uint glbABIShift = (abi64Bit()) ? 1 : 0;

  VirtualGPU::CalVirtualDesc* progGrid = &gpu.cal_;

  // Finds local workgroup size
  findLocalWorkSize(workDim, gblWorkSize, lclWorkSize);

  // Initialize the execution grid block and size/offset
  pGlobalSize[0] = pGlobalSize[1] = pGlobalSize[2] = 1;
  pGlobalSize[3] = static_cast<uint>(workDim);

  pLocalSize[0] = pLocalSize[1] = pLocalSize[2] = 1;
  pLocalSize[3] = 0;

  pNumGroups[0] = pNumGroups[1] = pNumGroups[2] = 1;
  pNumGroups[3] = 0;

  pGlobalOffset[2] = pGlobalOffset[1] = pGlobalOffset[0] = 0;
  pGroupOffset[2] = pGroupOffset[1] = pGroupOffset[0] = 0;

  progGrid->gridBlock.width = progGrid->gridBlock.height = progGrid->gridBlock.depth = 1;

  progGrid->gridSize.width = progGrid->gridSize.height = progGrid->gridSize.depth = 1;

  progGrid->partialGridBlock.width = progGrid->partialGridBlock.height =
      progGrid->partialGridBlock.depth = 1;

  bool partialGrid = false;

  // Fill the right values, based on the application request
  switch (workDim) {
    case 3:
      pLocalSize[2] = progGrid->gridBlock.depth = static_cast<CALuint>(lclWorkSize[2]);

      pGlobalSize[2] = static_cast<CALuint>(glbWorkSizeOrg[2]);
      progGrid->gridSize.depth = static_cast<CALuint>(gblWorkSize[2]);
      progGrid->gridSize.depth /= progGrid->gridBlock.depth;
      pNumGroups[2] = pGlobalSize[2] / progGrid->gridBlock.depth;

      pGlobalOffset[2] = glbWorkOffset[2];
      pGroupOffset[2] = groupOffset[2];
      pNDRangeGlobalOffset[2 + glbABIShift] = glbWorkOffsetOrg[2];

      // Check if partial workgroup dispatch is required
      progGrid->partialGridBlock.depth = gblWorkSize[2] % lclWorkSize[2];
      if (progGrid->partialGridBlock.depth != 0) {
        partialGrid = true;
        // Increment the number of groups
        progGrid->gridSize.depth++;
        pNumGroups[2]++;
      } else {
        progGrid->partialGridBlock.depth = lclWorkSize[2];
      }
    // Fall through to fill 2D and 1D dimensions...
    case 2:
      pLocalSize[1] = progGrid->gridBlock.height = static_cast<CALuint>(lclWorkSize[1]);

      pGlobalSize[1] = static_cast<CALuint>(glbWorkSizeOrg[1]);
      progGrid->gridSize.height = static_cast<CALuint>(gblWorkSize[1]);
      progGrid->gridSize.height /= progGrid->gridBlock.height;
      pNumGroups[1] = pGlobalSize[1] / progGrid->gridBlock.height;

      pGlobalOffset[1] = glbWorkOffset[1];
      pGroupOffset[1] = groupOffset[1];
      pNDRangeGlobalOffset[1 + glbABIShift] = glbWorkOffsetOrg[1];

      // Check if partial workgroup dispatch is required
      progGrid->partialGridBlock.height = gblWorkSize[1] % lclWorkSize[1];
      if (progGrid->partialGridBlock.height != 0) {
        partialGrid = true;
        // Increment the number of groups
        progGrid->gridSize.height++;
        pNumGroups[1]++;
      } else {
        progGrid->partialGridBlock.height = lclWorkSize[1];
      }
    // Fall through to fill 1D dimension...
    case 1:
      pLocalSize[0] = progGrid->gridBlock.width = static_cast<CALuint>(lclWorkSize[0]);

      pGlobalSize[0] = static_cast<CALuint>(glbWorkSizeOrg[0]);
      progGrid->gridSize.width = static_cast<CALuint>(gblWorkSize[0]);
      progGrid->gridSize.width /= progGrid->gridBlock.width;
      pNumGroups[0] = pGlobalSize[0] / progGrid->gridBlock.width;

      pGlobalOffset[0] = glbWorkOffset[0];
      pGroupOffset[0] = groupOffset[0];
      pNDRangeGlobalOffset[0 + glbABIShift] = glbWorkOffsetOrg[0];

      // Check if partial workgroup dispatch is required
      progGrid->partialGridBlock.width = gblWorkSize[0] % lclWorkSize[0];
      if (progGrid->partialGridBlock.width != 0) {
        partialGrid = true;
        // Increment the number of groups
        progGrid->gridSize.width++;
        pNumGroups[0]++;
      } else {
        progGrid->partialGridBlock.width = lclWorkSize[0];
      }
      break;
    default:
      LogWarning("Wrong dimensions. Force to 1x1x1!");
      break;
  }

  if (!partialGrid) {
    progGrid->partialGridBlock.width = progGrid->partialGridBlock.height =
        progGrid->partialGridBlock.depth = 0;
  }

  // Calculate the total number of workitems and workgroups
  pGlobalOffset[3] = pGroupOffset[3] = 1;
  for (uint i = 0; i < workDim; ++i) {
    pGlobalOffset[3] *= pGlobalOffset[i];
    pGroupOffset[3] *= pGroupOffset[i];
  }

  // Setup debug output buffer (if printf is active)
  if (flags() & PrintfOutput) {
    if (abi64Bit()) {
      // Setup the debug info in constant buffer
      reinterpret_cast<uint64_t*>(debugInfo)[1] = gpu.printfDbg().bufOffset();
      // Size in DWORDs
      debugInfo[4] = static_cast<uint32_t>(gpu.printfDbg().wiDbgSize());
      debugInfo[4] /= sizeof(uint32_t);
    } else {
      // Setup the debug info in constant buffer
      debugInfo[1] = static_cast<uint32_t>(gpu.printfDbg().bufOffset());
      // Size in DWORDs
      debugInfo[2] = static_cast<uint32_t>(gpu.printfDbg().wiDbgSize());
      debugInfo[2] /= sizeof(uint32_t);
    }
  }
}

bool Kernel::initParameters() {
  size_t offset = 0;
  device::Kernel::parameters_t params;
  amd::KernelParameterDescriptor desc;

  for (uint i = 0; i < arguments_.size(); ++i) {
    const KernelArg* arg = argument(i);

    // Initialize the arguments for the abstraction layer
    if (arg->isCbNeeded()) {
      desc.name_ = arg->name_.data();
      desc.type_ = arg->type();
      desc.size_ = arg->size(false);
      desc.addressQualifier_ = arg->addressQualifier();
      desc.accessQualifier_ = arg->accessQualifier();
      desc.typeName_ = arg->typeName();
      desc.typeQualifier_ = arg->typeQualifier();

      // Make offset alignment to match CPU metadata, since
      // in multidevice config abstraction layer has a single signature
      // and CPU sends the paramaters as they are allocated in memory
      size_t size = desc.size_;
      if (size == 0) {
        // Local memory for CPU
        size = sizeof(cl_mem);
      }
      offset = amd::alignUp(offset, std::min(size, size_t(16)));
      desc.offset_ = offset;
      offset += amd::alignUp(size, sizeof(uint32_t));
      params.push_back(desc);
    }
  }

  // Report the allocated local memory size (emulated and hw)
  if (hwLocalSize_ != 0) {
    CondLog((dev().info().localMemSize_ < hwLocalSize_),
            "Requested local size is bigger than reported");
    workGroupInfo_.localMemSize_ = hwLocalSize_;
  }

  if (!createSignature(params, params.size(), amd::KernelSignature::ABIVersion_0)) {
    return false;
  }

  return true;
}

bool Kernel::bindGlobalHwCb(VirtualGPU& gpu, VirtualGPU::GslKernelDesc* desc) const {
  bool result = true;

  // Bind HW constant buffers used for the global data store
  const Program::HwConstBuffers& gds = prog().glbHwCb();
  for (const auto& it : gds) {
    uint idx = it.first;
    result = bindResource(gpu, *(it.second), idx, ConstantBuffer, idx);
  }

  return result;
}

bool Kernel::bindConstantBuffers(VirtualGPU& gpu) const {
  bool result = true;

  assert((numCb_ <= MaxConstBuffersArguments) && "Runtime doesn't support more CBs for arguments!");

  // Upload the parameters to HW and bind all constant buffers
  for (uint i = 0; i < numCb_; i++) {
    ConstBuffer* cb = gpu.constBufs_[i];
    result &= cb->uploadDataToHw(cbSizes_[i]) &&
        bindResource(gpu, *cb, i, ConstantBuffer, i, cb->wrtOffset());
  }

  return result;
}

void Kernel::processMemObjects(VirtualGPU& gpu, const amd::Kernel& kernel, const_address params,
                               bool nativeMem) const {
  // Mark the tracker with a new kernel,
  // so we can avoid checks of the aliased objects
  gpu.memoryDependency().newKernel();

  // Check all parameters for the current kernel
  const amd::KernelSignature& signature = kernel.signature();
  amd::Memory* const* memories =
      reinterpret_cast<amd::Memory* const*>(params + kernel.parameters().memoryObjOffset());

  for (size_t i = 0; i < signature.numParameters(); ++i) {
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    const KernelArg* arg = argument(i);
    Memory* memory = NULL;

    // Find if current argument is a buffer
    if ((desc.type_ == T_POINTER) && (arg->type_ != KernelArg::PointerLocal) &&
        (arg->type_ != KernelArg::PointerHwLocal)) {
      uint32_t index = desc.info_.arrayIndex_;
      if (nativeMem) {
        memory = reinterpret_cast<Memory* const*>(memories)[index];
      } else if (*reinterpret_cast<amd::Memory* const*>(params + desc.offset_) != NULL) {
        memory = dev().getGpuMemory(memories[index]);
        // Synchronize data with other memory instances if necessary
        memory->syncCacheFromHost(gpu);
      }

      if (memory != NULL) {
        // Validate memory for a dependency in the queue
        gpu.memoryDependency().validate(gpu, memory, arg->memory_.readOnly_);
      }
    }
  }
}

bool Kernel::loadParameters(VirtualGPU& gpu, const amd::Kernel& kernel, const_address params,
                            bool nativeMem) const {
  bool result = true;
  uint i;

  // Initialize local private ranges
  if (!initLocalPrivateRanges(gpu)) {
    return false;
  }

  if ((UavIdUndefined != uavRaw_) && (!(flags() & PrintfOutput) || (printfId_ != UavIdUndefined))) {
    Memory* gpuMemory = dev().getGpuMemory(dev().dummyPage());
    // Bind a buffer for a dummy read
    result = bindResource(gpu, *gpuMemory, 0, ArgumentUavID, uavRaw_);
  }

  // Find all parameters for the current kernel
  const amd::KernelSignature& signature = kernel.signature();
  for (i = 0; i != signature.numParameters(); ++i) {
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    // Set current argument
    if (!setArgument(gpu, kernel, i, params, desc, nativeMem)) {
      result = false;
      break;
    }
  }

  if (result) {
    // Update the ring ranges and math constant
    setLocalPrivateRanges(gpu);

    result = bindConstantBuffers(gpu);

    if (flags() & PrivateFixed) {
      result &= bindResource(gpu, dev().globalMem(), 0, GlobalBuffer, uavRaw_);
    }

    // Setup debug output buffer (if printf is active)
    if (flags() & PrintfOutput) {
      gpu.addVmMemory(gpu.printfDbg().dbgBuffer());
    }
  }

  return result;
}

bool Kernel::run(VirtualGPU& gpu, GpuEvent* calEvent, bool lastRun, bool lastDoppCmd,
                 bool pfpaDoppCmd) const {
  const VirtualGPU::CalVirtualDesc* dispatch = gpu.cal();

  auto compProg = static_cast<gsl::ComputeProgramObject*>(gpu.gslKernelDesc()->func_);
  compProg->setWavesPerSH(waveLimiter_.getWavesPerSH(&gpu));

  gpu.eventBegin(MainEngine);
  gpu.rs()->Dispatch(gpu.cs(), &dispatch->gridBlock, &dispatch->partialGridBlock,
                     &dispatch->gridSize, dispatch->localSize, gpu.vmMems(), dispatch->memCount_,
                     lastDoppCmd, pfpaDoppCmd);

  gpu.flushCUCaches();

  gpu.eventEnd(MainEngine, *calEvent);

  // Unbind all resources
  unbindResources(gpu, *calEvent, lastRun);

  return true;
}

static size_t counter = 0;
void Kernel::debug(VirtualGPU& gpu) const {
  std::fstream stubWrite;
  address src = NULL;

  std::cerr << "--- " << name_ << " ---" << std::endl;
  for (uint i = 0; i < arguments_.size(); ++i) {
    const KernelArg* arg = argument(i);
    const Memory* gpuMem = gpu.slots_[i].memory_;
    std::stringstream fileName;
    bool bufferObj =
        ((arg->type_ == KernelArg::PointerGlobal) || (arg->type_ == KernelArg::PointerConst) ||
         (arg->type_ == KernelArg::PointerHwConst));

    if ((src != NULL) && arg->isCbNeeded() && bufferObj) {
      address memory = gpu.cb(arg->cbIdx_)->sysMemCopy();
      std::cerr.setf(std::ios::hex);
      uint* location =
          reinterpret_cast<uint*>(src + *reinterpret_cast<uint*>(memory + arg->cbPos_));
      std::cerr << " > " << arg->name_ << ": 0x" << location << std::endl;

      // Dump the data
      fileName << counter << "_kernel_" << name() << "_" << arg->name_ << "_" << location << ".bin";
      stubWrite.open(fileName.str().c_str(), (std::fstream::out | std::fstream::binary));

      // Write data to a file
      if (stubWrite.is_open()) {
        stubWrite.write(reinterpret_cast<char*>(location), gpuMem->size());
        stubWrite.close();
      }
    }
    if (((arg->type_ >= KernelArg::Image1D) && (arg->type_ <= KernelArg::Image3D)) ||
        ((src == NULL) && bufferObj)) {
      //@todo Replace the current map
      Memory* resource = const_cast<Memory*>(gpu.slots_[i].memory_);
      void* memory = resource->map(&gpu);
      uint* location = reinterpret_cast<uint*>(memory);
      std::cerr << " > " << arg->name_ << (bufferObj ? ": buffer" : ": image") << std::endl;
      // Dump the data
      fileName << counter << "_kernel_" << name() << "_" << arg->name_ << "_" << location << ".bin";
      stubWrite.open(fileName.str().c_str(), (std::fstream::out | std::fstream::binary));

      // Write data to a file
      if (stubWrite.is_open()) {
        stubWrite.write(reinterpret_cast<char*>(location), gpuMem->size());
        stubWrite.close();
      }
      resource->unmap(&gpu);
    }
  }

  for (uint i = 0; i < gpu.constBufs_.size(); ++i) {
    std::stringstream fileName;
    fileName << counter++ << "_kernel_" << name() << "_const" << i << ".bin";
    stubWrite.open(fileName.str().c_str(), (std::fstream::out | std::fstream::binary));
    if (stubWrite.is_open()) {
      address memory = reinterpret_cast<address>(gpu.constBufs_[i]->map(&gpu, Resource::ReadOnly));
      // Check if we have OpenCL program
      stubWrite.write(reinterpret_cast<char*>(memory + gpu.cb(i)->wrtOffset()),
                      gpu.cb(i)->lastWrtSize());
      gpu.constBufs_[i]->unmap(&gpu);
      stubWrite.close();
    }
  }
  const Program::HwConstBuffers& gds = prog().glbHwCb();
  for (const auto& it : gds) {
    uint idx = it.first;
    std::stringstream fileName;
    fileName << counter++ << "_kernel_" << name() << "_const" << idx << ".bin";
    stubWrite.open(fileName.str().c_str(), (std::fstream::out | std::fstream::binary));
    if (stubWrite.is_open()) {
      address memory = reinterpret_cast<address>(it.second->map(&gpu, Resource::ReadOnly));
      // Check if we have OpenCL program
      stubWrite.write(reinterpret_cast<char*>(memory), it.second->size());
      it.second->unmap(&gpu);
      stubWrite.close();
    }
  }
}

bool Kernel::initConstBuffers() {
  bool result = true;
  size_t i;

  assert((numCb_ != 0) && "We have 0 constant buffers!");

  // Allocate an array for CB sizes
  cbSizes_ = new size_t[numCb_];
  if (cbSizes_ == NULL) {
    return false;
  }
  memset(cbSizes_, 0, sizeof(size_t) * numCb_);

  // CB0 is reserved for ABI data
  cbSizes_[0] = TotalABIVectors * ConstBuffer::VectorSize;

  // Find sizes of all constant buffers
  for (i = 0; i < arguments_.size(); ++i) {
    const KernelArg* arg = argument(i);
    size_t size = arg->cbPos_ + arg->size(true);
    size_t specVec = arg->specialVector();
    if (specVec != 0) {
      size = arg->cbPos_ + (arg->size_ / KernelArg::VectorSizeLimit) * ConstBuffer::VectorSize;
    }
    // Do we need a CB?
    if (arg->isCbNeeded() && (cbSizes_[arg->cbIdx_] < size)) {
      cbSizes_[arg->cbIdx_] = size;
    }
  }

  return result;
}

bool Kernel::setInternalSamplers(VirtualGPU& gpu) const {
  for (uint i = 0; i < samplerSize(); ++i) {
    const KernelArg* arg = sampler(i);
    uint state = arg->cbPos_;
    uint idx = arg->index_;

    if (gpu.cal()->samplersState_[idx] != state) {
      setSampler(gpu, state, idx);
      gpu.cal_.samplersState_[idx] = state;
    }
  }

  return true;
}

bool Kernel::setArgument(VirtualGPU& gpu, const amd::Kernel& kernel,
                         uint idx, const_address params,
                         const amd::KernelParameterDescriptor& desc,
                         bool nativeMem) const {
  size_t size = desc.size_;
  const void* param = params + desc.offset_;
  bool result = true;
  const KernelArg* arg;
  address memory;
  size_t argSize;
  static const bool waitOnBusyEngine = true;

  assert((idx < arguments_.size()) && "Param index is out of range!");

  arg = argument(idx);
  assert((arg->cbIdx_ == 1) && "Runtime supports CB1 only for the arguments buffer!");
  memory = gpu.cb(1)->sysMemCopy();
  argSize = arg->size(true);

  // Bind the global heap for emulation mode
  switch (arg->type_) {
    case KernelArg::PointerLocal:
    case KernelArg::PointerPrivate:
      if (!bindResource(gpu, dev().globalMem(), 0, GlobalBuffer, uavRaw_)) {
        return false;
      }
    // Fall through ...
    default:
      break;
  }

  switch (arg->type_) {
    case KernelArg::PointerConst:
    case KernelArg::PointerHwConst:
    case KernelArg::PointerGlobal: {
      gpu::Memory* gpuMem = NULL;
      amd::Memory* const* memories =
        reinterpret_cast<amd::Memory* const*>(params + kernel.parameters().memoryObjOffset());
      uint32_t index = desc.info_.arrayIndex_;
      if (nativeMem) {
        gpuMem = reinterpret_cast<Memory*>(memories[index]);
      } else if (memories[index] != nullptr) {
        gpuMem = dev().getGpuMemory(memories[index]);
      }
      bool forceZeroOffset = false;

      if (gpuMem == NULL) {
        forceZeroOffset = true;
        gpuMem = dev().getGpuMemory(dev().dummyPage());
      }
      uint64_t offset = gpuMem->pinOffset();

      // Make sure the passed argument is a buffer object
      if (!gpuMem->cal()->buffer_) {
        LogError("The kernel buffer argument isn't a buffer object!");
        return false;
      }

      if (arg->type_ == KernelArg::PointerHwConst) {
        // Bind current memory object with the kernel
        if (!bindResource(gpu, *gpuMem, idx, ArgumentConstBuffer, arg->index_)) {
          return false;
        }
        assert((offset == 0) && "No offset for HW CB");
        // Add a fake offset to make sure (ptr != NULL) is TRUE
        offset = 1;
      } else {
        ResourceType type = ArgumentHeapBuffer;

        // Check if kernel expects UAV binding
        if (arg->memory_.uavBuf_) {
          type = ArgumentBuffer;
        } else {
          // Bind global buffer to UAV this buffer is bound to
          if (!bindResource(gpu, dev().globalMem(), 0, GlobalBuffer, uavRaw_)) {
            return false;
          }
        }

        // Bind current memory object with the kernel
        // Note: it's a fake binding, if the buffer is part of
        // the global heap
        if (!bindResource(gpu, *gpuMem, idx, type, arg->index_)) {
          return false;
        }

        // Update offset only if we bind HeapBuffer or
        // it's global address space in UAV setup on SI+
        offset += gpuMem->hbOffset();
        if (!forceZeroOffset) {
          assert((offset != 0) && "Offset 0 with a real allocation!");
        }
        gpu.addVmMemory(gpuMem);
      }

      // Wait for resource if it was used on an inactive engine
      //! \note syncCache may call DRM transfer
      gpuMem->wait(gpu, waitOnBusyEngine);

      if (forceZeroOffset) {
        offset = 0;
      }

      // Copy memory offset into the constant buffer
      if (abi64Bit()) {
        *(reinterpret_cast<uint64_t*>(memory + arg->cbPos_)) = offset;
      } else {
        *(reinterpret_cast<uint*>(memory + arg->cbPos_)) = static_cast<uint>(offset);
      }
    } break;
    case KernelArg::Image1D:
    case KernelArg::Image2D:
    case KernelArg::Image3D:
    case KernelArg::Image1DB:
    case KernelArg::Image1DA:
    case KernelArg::Image2DA: {
      gpu::Memory* gpuMem = NULL;
      amd::Memory* const* memories =
        reinterpret_cast<amd::Memory* const*>(params + kernel.parameters().memoryObjOffset());
      uint32_t index = desc.info_.arrayIndex_;
      if (nativeMem) {
        gpuMem = reinterpret_cast<Memory*>(memories[index]);
      } else if (memories[index] != nullptr) {
        gpuMem = dev().getGpuMemory(memories[index]);
      }

      if (gpuMem == NULL) {
        return false;
      }
      // Make sure the passed argument is an image object
      if (gpuMem->cal()->buffer_) {
        LogError("The kernel image argument isn't an image object!");
        return false;
      }

      ResourceType resType = arg->memory_.readOnly_ ? ArgumentImageRead : ArgumentImageWrite;

      // Bind current memory object with the shader.
      if (!bindResource(gpu, *gpuMem, idx, resType, arg->index_)) {
        return false;
      }

      // Wait for resource if it was used on an inactive engine
      //! \note syncCache may call DRM transfer
      gpuMem->wait(gpu, waitOnBusyEngine);

      // Copy image constants into the constant buffer
      if (gpuMem->owner() != NULL) {
        copyImageConstants(gpuMem->owner()->asImage(),
                           reinterpret_cast<ImageConstants*>(memory + arg->cbPos_));
      }

      // Handle DOPP texture resource
      gslMemObject gslMem = gpuMem->gslResource();
      if (gslMem->getAttribs().isDOPPDesktopTexture) {
        gpu.addVmMemory(gpuMem);
      }
    } break;
    case KernelArg::Sampler: {
      uint32_t index = desc.info_.arrayIndex_;
      const amd::Sampler* amdSampler = reinterpret_cast<amd::Sampler* const*>(params +
        kernel.parameters().samplerObjOffset())[index];
      uint idx = arg->index_;
      uint32_t state = amdSampler->state();

      if (state != gpu.cal()->samplersState_[idx]) {
        setSampler(gpu, state, idx);
        gpu.cal_.samplersState_[idx] = state;
      }

      // Copy sampler state into the constant buffer
      *(reinterpret_cast<uint32_t*>(memory + arg->cbPos_)) = state;
    } break;
    case KernelArg::Counter: {
      gpu::Memory* gpuMem = NULL;
      if (nativeMem) {
        gpuMem = *reinterpret_cast<Memory* const*>(param);
      } else if (*reinterpret_cast<amd::Memory* const*>(param) != NULL) {
        gpuMem = dev().getGpuMemory(*reinterpret_cast<amd::Memory* const*>(param));
      }

      // Wait for resource if it was used on an inactive engine
      //! \note syncCache may call DRM transfer
      gpuMem->wait(gpu, waitOnBusyEngine);

      // Bind current memory object with the shader.
      if (!bindResource(gpu, *gpuMem, idx, ArgumentCounter, idx)) {
        return false;
      }
    } break;
    case KernelArg::PointerHwLocal: {
      // Calculate current offset in the local ring
      uint offset = gpu.cal_.localSize;
      uint extra = amd::alignUp(offset, arg->alignment_) - offset;

      offset = amd::alignUp(offset, arg->alignment_);
      size_t memSize = *static_cast<const uintptr_t*>(param);

      // Allocate new memory from the local ring
      gpu.cal_.localSize += static_cast<uint>(memSize) + extra;
      // Copy current local argument's offset into the CB
      *(reinterpret_cast<uint*>(memory + arg->cbPos_)) = offset;

      CondLog((gpu.cal_.localSize > dev().info().localMemSize_),
              "Requested local size is bigger than reported!");
    } break;
    case KernelArg::Float:
    case KernelArg::Double:
    case KernelArg::Char:
    case KernelArg::UChar:
    case KernelArg::Short:
    case KernelArg::UShort:
    case KernelArg::Int:
    case KernelArg::UInt:
    case KernelArg::Long:
    case KernelArg::ULong:
      if (size != argSize) {
        LogWarning("Parameter's sizes are unmatched!");
      }
    // Fall through ...
    case KernelArg::Struct:
    case KernelArg::Union: {
      size_t specVec = arg->specialVector();
      if (specVec != 0) {
        uint iter = (arg->size_ / KernelArg::VectorSizeLimit);
        for (uint i = 0; i < iter; ++i) {
          amd::Os::fastMemcpy(
              (memory + arg->cbPos_ + i * ConstBuffer::VectorSize),
              reinterpret_cast<const char*>(param) + i * KernelArg::VectorSizeLimit * specVec,
              specVec * KernelArg::VectorSizeLimit);
        }
      } else {
        // Copy data into the CB
        amd::Os::fastMemcpy((memory + arg->cbPos_), param, size);
      }
    } break;
    default:
      LogError("Unhandled argument's type!");
      break;
  }

  return result;
}

bool Kernel::initLocalPrivateRanges(VirtualGPU& gpu) const {
  // Initialize HW local
  gpu.cal_.localSize = hwLocalSize_;

  // Bind the global buffer if emulated local or private memory
  // was allocated by the kernel
  if ((flags() & PrintfOutput && (printfId_ == UavIdUndefined)) && (uavRaw_ != UavIdUndefined)) {
    if (!bindResource(gpu, dev().globalMem(), 0, GlobalBuffer, uavRaw_)) {
      return false;
    }
  }

  // Bind the global buffer if emulated constant buffers are enabled
  if (cbId_ != UavIdUndefined) {
    if (!bindResource(gpu, dev().globalMem(), 0, ArgumentCbID, cbId_)) {
      return false;
    }
  }

  // Bind the printf buffer
  if (printfId_ != UavIdUndefined) {
    if (!bindResource(gpu, dev().globalMem(), 0, ArgumentPrintfID, printfId_)) {
      return false;
    }
  }
  // Initialize the iterations count
  gpu.cal_.iterations_ = 1;

  return true;
}

void Kernel::setLocalPrivateRanges(VirtualGPU& gpu) const {
  address cbBuf = gpu.cb(0)->sysMemCopy();
  uint* data;
  uint gridSize =
      gpu.cal()->gridSize.width * gpu.cal()->gridSize.height * gpu.cal()->gridSize.depth;
  uint blockSize =
      gpu.cal()->gridBlock.width * gpu.cal()->gridBlock.height * gpu.cal()->gridBlock.depth;

  //! \todo validate if the compiler still generates PrivateFixed
  if (flags() & PrivateFixed) {
    // Update private ring
    data = reinterpret_cast<uint*>(cbBuf + PrivateRingOffset * ConstBuffer::VectorSize);
    Memory* gpuMemory = dev().getGpuMemory(dev().dummyPage());

    if (abi64Bit()) {
      reinterpret_cast<uint64_t*>(data)[0] = gpuMemory->hbOffset();
      data[2] = 0;
      data[3] = 0;
    } else {
      data[0] = static_cast<uint>(gpuMemory->hbOffset());
      data[1] = 0;
      data[2] = data[3] = 0;
    }
    gpu.addVmMemory(gpuMemory);
  }

  // Copy the math lib constants
  amd::Os::fastMemcpy((cbBuf + MathLibOffset * ConstBuffer::VectorSize), MathLibConst,
                      sizeof(MathLibConst));

  // Update the offset to the global data
  if (prog().glbData() != NULL) {
    gpu.addVmMemory(prog().glbData());
    uint64_t glbDataOffset = prog().glbData()->hbOffset();
    if (abi64Bit()) {
      *reinterpret_cast<uint64_t*>(cbBuf + GlobalDataStoreOffset * ConstBuffer::VectorSize) =
          glbDataOffset;
    } else {
      *reinterpret_cast<uint*>(cbBuf + GlobalDataStoreOffset * ConstBuffer::VectorSize) =
          static_cast<uint>(glbDataOffset);
    }
  }

  // Split workload if it was requested
  if ((gpu.cal_.iterations_ < 2) && gpu.dmaFlushMgmt().dispatchSplitSize() != 0) {
    uint totalSize = gridSize * blockSize;
    if (totalSize > gpu.dmaFlushMgmt().dispatchSplitSize()) {
      gpu.cal_.iterations_ =
          std::max(gpu.cal_.iterations_, (totalSize / gpu.dmaFlushMgmt().dispatchSplitSize()));
    }
  }

  // Initialize the number of iterations to the grid size
  if (flags() & PrintfOutput) {
    gpu.cal_.iterations_ = gridSize;
  }
}

void Kernel::setSampler(VirtualGPU& gpu, uint32_t state, uint physUnit) const {
  // All CAL sampler's parameters are in floats
  float gslAddress = GSL_CLAMP_TO_BORDER;
  float gslMinFilter = GSL_MIN_NEAREST;
  float gslMagFilter = GSL_MAG_NEAREST;

  state &= ~amd::Sampler::StateNormalizedCoordsMask;

  // Program the sampler address mode
  switch (state & amd::Sampler::StateAddressMask) {
    case amd::Sampler::StateAddressRepeat:
      gslAddress = GSL_REPEAT;
      break;
    case amd::Sampler::StateAddressClampToEdge:
      gslAddress = GSL_CLAMP_TO_EDGE;
      break;
    case amd::Sampler::StateAddressMirroredRepeat:
      gslAddress = GSL_MIRRORED_REPEAT;
      break;
    case amd::Sampler::StateAddressClamp:
    case amd::Sampler::StateAddressNone:
    default:
      break;
  }
  state &= ~amd::Sampler::StateAddressMask;

  gpu.setSamplerParameter(physUnit, GSL_TEXTURE_WRAP_S, &gslAddress);
  gpu.setSamplerParameter(physUnit, GSL_TEXTURE_WRAP_T, &gslAddress);
  gpu.setSamplerParameter(physUnit, GSL_TEXTURE_WRAP_R, &gslAddress);

  // Program texture filter mode
  if (state == amd::Sampler::StateFilterLinear) {
    gslMinFilter = GSL_MIN_LINEAR;
    gslMagFilter = GSL_MAG_LINEAR;
  }

  gpu.setSamplerParameter(physUnit, GSL_TEXTURE_MIN_FILTER, &gslMinFilter);
  gpu.setSamplerParameter(physUnit, GSL_TEXTURE_MAG_FILTER, &gslMagFilter);
}

bool Kernel::bindResource(VirtualGPU& gpu, const Memory& memory, uint paramIdx, ResourceType type,
                          uint physUnit, size_t offset) const {
  gslUAVType uavType = GSL_UAV_TYPE_UNKNOWN;

  // Find the original resource name from the IL program
  switch (type) {
    case GlobalBuffer:
      if (gpu.state_.boundGlobal_) {
        return true;
      }
      gpu.state_.boundGlobal_ = true;
      physUnit = uavRaw_;
      uavType = GSL_UAV_TYPE_TYPELESS;
      break;
    case ArgumentCbID:
      if (gpu.state_.boundCb_) {
        return true;
      }
      gpu.state_.boundCb_ = true;
      physUnit = cbId_;
      uavType = GSL_UAV_TYPE_TYPELESS;
      break;
    case ArgumentPrintfID:
      if (gpu.state_.boundPrintf_) {
        return true;
      }
      gpu.state_.boundPrintf_ = true;
      physUnit = printfId_;
      uavType = GSL_UAV_TYPE_TYPELESS;
      break;
    case ArgumentHeapBuffer:
    case ArgumentBuffer:
    case ArgumentImageRead:
    case ArgumentImageWrite:
    case ArgumentConstBuffer:
    case ArgumentCounter:
      // Early exit if resource is bound already
      if (gpu.slots_[paramIdx].state_.bound_) {
        return true;
      }

      // Associate resource with the slot
      gpu.slots_[paramIdx].memory_ = &memory;

      // Mark resource as bound
      gpu.slots_[paramIdx].state_.bound_ = true;

      if (type == ArgumentCounter) {
        GpuEvent calEvent;

        // Bind memory with atomic counter
        gpu.cs()->bindAtomicCounter(argument(paramIdx)->index_, memory.gslResource());

        // Copy the counter value into GDS
        gpu.eventBegin(MainEngine);
        gpu.cs()->syncAtomicCounter(argument(paramIdx)->index_, false);
        gpu.eventEnd(MainEngine, calEvent);

        // Mark resource as busy
        memory.setBusy(gpu, calEvent);
        return true;
      } else if (type == ArgumentHeapBuffer) {
        // We return here, since we just have to bind the global heap
        return true;
      } else if (type == ArgumentConstBuffer) {
        gpu.slots_[paramIdx].state_.constant_ = true;
      }
      break;
    case ArgumentUavID:
    case ConstantBuffer:
      break;
    default:
      LogPrintfError("Unspecified argument type ()!", type);
      return false;
  }

  gslMemObject gslMem = NULL;
  // Use global address space on SI+ for UAV setup
  if ((type == ArgumentBuffer) || (type == ArgumentCbID) || (type == ArgumentUavID) ||
      (type == ArgumentPrintfID)) {
    gslMem = dev().heap().resource().gslResource();
  } else {
    gslMem = memory.gslResource();
  }

  // Associate memory with the physical unit, the actual binding
  bool result = true;
  switch (type) {
    case GlobalBuffer:
    case ArgumentBuffer:
    case ArgumentImageWrite:
    case ArgumentUavID:
    case ArgumentCbID:
    case ArgumentPrintfID:
      if (type == ArgumentImageWrite) {
        uavType = GSL_UAV_TYPE_TYPED;
      } else if ((type == ArgumentBuffer) || (type == ArgumentUavID)) {
        uavType = GSL_UAV_TYPE_TYPELESS;
      }
      if (gpu.cal_.uavs_[physUnit] != gslMem) {
        result = gpu.setUAVBuffer(physUnit, gslMem, uavType);
        gpu.setUAVChannelOrder(physUnit, gslMem);
        gpu.cal_.uavs_[physUnit] = gslMem;
      }
      break;
    case ConstantBuffer:
    case ArgumentConstBuffer:
      if ((gpu.cal_.constBuffers_[physUnit] != gslMem) || (offset != 0)) {
        result = gpu.setConstantBuffer(physUnit, gslMem, offset, memory.hbSize());
        gpu.cal_.constBuffers_[physUnit] = gslMem;
      }
      break;
    case ArgumentImageRead:
      if (gpu.cal_.readImages_[physUnit] != gslMem) {
        result = gpu.setInput(physUnit, gslMem);
        gpu.cal_.readImages_[physUnit] = gslMem;
      }
      break;
    default:
      result = false;
      assert(false);
      break;
  }
  if (!result) {
    LogPrintfError("setMem failed unit:%d mem:0x%08x!", physUnit, gslMem);
    return false;
  }

  return true;
}

void Kernel::unbindResources(VirtualGPU& gpu, GpuEvent calEvent, bool lastRun) const {
  // Make sure unbind occurs on the last run, in case the execution had a split
  if (lastRun) {
    for (uint i = 0; i < arguments_.size(); ++i) {
      if (gpu.slots_[i].state_.bound_) {
        GpuEvent calEventTmp = calEvent;

        if (KernelArg::Counter == argument(i)->type_) {
          // Copy the counter value from GDS
          gpu.eventBegin(MainEngine);
          gpu.cs()->syncAtomicCounter(argument(i)->index_, true);
          gpu.eventEnd(MainEngine, calEventTmp);
        } else if (!(gpu.slots_[i].state_.constant_ || argument(i)->memory_.readOnly_)) {
          // Signal the abstraction layer that GPU memory is dirty
          if (gpu.slots_[i].memory_->owner() != NULL) {
            gpu.slots_[i].memory_->owner()->signalWrite(&gpu.dev());
          }
        }
        // Mark resource as busy
        gpu.slots_[i].memory_->setBusy(gpu, calEventTmp);

        gpu.slots_[i].state_.value_ = 0;
      }
    }

    // Unbind the global buffer
    gpu.state_.boundGlobal_ = false;

    // Unbind the constant buffer
    gpu.state_.boundCb_ = false;

    // Unbind the pritnf buffer
    gpu.state_.boundPrintf_ = false;
  }

  // Mark CB busy
  for (uint i = 0; i < numCb_; ++i) {
    gpu.constBufs_[i]->setBusy(gpu, calEvent);
  }

  // Set the event object for the scratch buffer
  if (workGroupInfo()->scratchRegs_ > 0) {
    dev().scratch(gpu.hwRing())->memObj_->setBusy(gpu, calEvent);
  }
}

void Kernel::copyImageConstants(const amd::Image* amdImage, ImageConstants* imageData) const {
  imageData->width_ = static_cast<uint32_t>(amdImage->getWidth());
  imageData->height_ = static_cast<uint32_t>(amdImage->getHeight());
  imageData->depth_ = static_cast<uint32_t>(amdImage->getDepth());
  imageData->dataType_ = static_cast<uint32_t>(amdImage->getImageFormat().image_channel_data_type);

  imageData->widthFloat_ = 1.f / static_cast<float>(amdImage->getWidth());
  imageData->heightFloat_ = 1.f / static_cast<float>(amdImage->getHeight());
  imageData->depthFloat_ = 1.f / static_cast<float>(amdImage->getDepth());
  imageData->channelOrder_ = static_cast<uint32_t>(amdImage->getImageFormat().image_channel_order);
}

union MetadataVersion {
  struct {
    uint64_t revision_ : 16;      //!< LLVM metadata revision
    uint64_t minorVersion_ : 16;  //!< LLVM metadata minor verison
    uint64_t majorVersion_ : 16;  //!< LLVM metadata major version
  };
  uint64_t value_;
  MetadataVersion(uint mj, uint mi, uint rev) : value_(0) {
    revision_ = rev;
    minorVersion_ = mi;
    majorVersion_ = mj;
  }
  MetadataVersion() : value_(0) {}
};

//! Version of metadata with buffer attributes
const MetadataVersion MetadataBufferAttributes = MetadataVersion(2, 0, 88);

//! Version of metadata with type qualifiers
const MetadataVersion MetadataTypeQualifiers = MetadataVersion(3, 1, 103);

bool NullKernel::parseArguments(const std::string& metaData, uint* uavRefCount) {
  // Initialize workgroup info
  workGroupInfo_.size_ = nullDev().info().preferredWorkGroupSize_;
  MetadataVersion mdVersion;

  // Find first tag
  size_t pos = metaData.find(";");

  // Loop through all provided program arguments
  while (pos != std::string::npos) {
    KernelArg arg;

    if (!expect(metaData, &pos, ";")) {
      break;
    }

    arg.type_ = KernelArg::NoType;

    // Loop through all available metadata types
    for (uint i = 0; i < ArgStateTotal; ++i) {
      uint tmpValue;
      // Find the name tag
      if (expect(metaData, &pos, ArgState[i].typeName_)) {
        switch (ArgState[i].type_) {
          case KernelArg::NoType:
            // Process next ...
            continue;
          case KernelArg::Reflection: {
            uint argIdx;
            // Read the argument's index
            if (!getuint(metaData, &pos, &argIdx)) {
              LogWarning("Couldn't get the argument index!");
              return false;
            }
            KernelArg* tmpArg = arguments_[argIdx];
            if (!getstring(metaData, &pos, &tmpArg->typeName_)) {
              LogWarning("Couldn't get the argument type!");
              return false;
            }
          }
            continue;
          case KernelArg::ConstArg: {
            uint argIdx;
            // Read the argument's index
            if (!getuint(metaData, &pos, &argIdx)) {
              LogWarning("Couldn't get the argument index!");
              return false;
            }
            KernelArg* tmpArg = arguments_[argIdx];
            tmpArg->typeQualifier_ |= CL_KERNEL_ARG_TYPE_CONST;
          }
            continue;
          case KernelArg::Grouping:
            for (uint j = 0; j < 3; ++j) {
              uint temp;
              // Read the compile workgroup size
              if (!getuint(metaData, &pos, &temp)) {
                LogWarning("Couldn't get the compile workgroup size!");
                return false;
              }
              workGroupInfo_.compileSize_[j] = temp;
            }
            // Process next ...
            continue;
          case KernelArg::WrkgrpSize: {
            uint temp;
            // Read the workgroup size
            if (!getuint(metaData, &pos, &temp)) {
              LogWarning("Couldn't get the workgroup size!");
              return false;
            }
            workGroupInfo_.size_ = temp;
          }
            // Process next ...
            continue;
          case KernelArg::Wavefront:
            // Process next ...
            continue;
          case KernelArg::UavId:
            // Read index
            if (!getuint(metaData, &pos, &arg.index_)) {
              return false;
            }
            break;
          case KernelArg::ConstBufId:
            // Read index
            if (!getuint(metaData, &pos, &cbId_)) {
              return false;
            }
            continue;
          case KernelArg::PrintfBufId:
            // Read index
            if (!getuint(metaData, &pos, &printfId_)) {
              return false;
            }
            continue;
          case KernelArg::MetadataVersion:
            // Read metadata version
            if (!getuint(metaData, &pos, &tmpValue)) {
              return false;
            }
            mdVersion.majorVersion_ = tmpValue;
            if (!getuint(metaData, &pos, &tmpValue)) {
              return false;
            }
            mdVersion.minorVersion_ = tmpValue;
            if (!getuint(metaData, &pos, &tmpValue)) {
              return false;
            }
            mdVersion.revision_ = tmpValue;
            // Process next ...
            continue;
          case KernelArg::GroupingHint:
            for (uint j = 0; j < 3; ++j) {
              uint temp;
              // Read the compile workgroup size hint
              if (!getuint(metaData, &pos, &temp)) {
                LogWarning("Couldn't get the compile workgroup size hint!");
                return false;
              }
              workGroupInfo_.compileSizeHint_[j] = temp;
            }
            // Process next ...
            continue;
          case KernelArg::VecTypeHint: {
            std::string temp;
            // Read the compile vector type hint
            if (!getstring(metaData, &pos, &temp)) {
              LogWarning("Couldn't get the compile vector type hint!");
              return false;
            }
            workGroupInfo_.compileVecTypeHint_ = temp;
          }
            // Process next ...
            continue;
          case KernelArg::WavesPerSimdHint: {
            uint tmp;
            if (!getuint(metaData, &pos, &tmp)) {
              return false;
            }
            workGroupInfo_.wavesPerSimdHint_ = tmp;
          }
            continue;
          default:
            break;
        }

        std::string argName;
        // Save the argument type
        arg.type_ = ArgState[i].type_;

        // Check if we should expect the name
        if (ArgState[i].name_) {
          // Read the parameter's name
          if (!getword(metaData, &pos, argName)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }
          arg.name_ = argName;
        }

        if (arg.type_ == KernelArg::Sampler) {
          if (!getuint(metaData, &pos, &arg.index_)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }
          if (!getuint(metaData, &pos, &arg.location_)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }
          if (!getuint(metaData, &pos, &arg.cbPos_)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }
        }

        // Check if we should expect the resource data type
        if (ArgState[i].resType_) {
          uint k;
          // Search for the data type
          for (k = 0; k < DataTypeTotal; k++) {
            if (expect(metaData, &pos, DataType[k].tagName_)) {
              arg.dataType_ = DataType[k].type_;
              if (arg.type_ == KernelArg::Image) {
                flags_ |= ImageEnable;
                if (expect(metaData, &pos, "RO:")) {
                  arg.memory_.readOnly_ = 1;
                } else if (expect(metaData, &pos, "RW:")) {
                  arg.memory_.readWrite_ = 1;
                  flags_ |= ImageWrite;
                } else if (expect(metaData, &pos, "WO:")) {
                  arg.memory_.writeOnly_ = 1;
                  flags_ |= ImageWrite;
                }
              } else if (arg.type_ == KernelArg::Value) {
                arg.type_ = DataType[k].type_;
              }
              break;
            }
          }
          if (k == DataTypeTotal) {
            LogWarning("We couldn't find the argument's type.");
            if ((arg.type_ == KernelArg::Value) || !getword(metaData, &pos, argName)) {
              LogWarning("Couldn't get a kernel argument!");
              return false;
            }
          }
          //! @todo temporary condition
          if ((arg.type_ == KernelArg::Opaque) || (arg.type_ == KernelArg::Sampler)) {
            assert(false);
            continue;
          }
        }

        // Check if we should expect the data size
        if (ArgState[i].size_) {
          uint tmpData;
          // Read the data size
          if (!getuint(metaData, &pos, &tmpData)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }
          if (arg.type_ == KernelArg::Image) {
            arg.type_ = arg.dataType_;
            arg.index_ = tmpData;
          } else {
            arg.size_ = tmpData;
          }
        }

        if (arg.type_ == KernelArg::Counter) {
          // Read a counter index
          if (!getuint(metaData, &pos, &arg.index_)) {
            LogWarning("Couldn't get a counter index!");
            return false;
          }
        }

        // Check if we should expect a resource index
        if (ArgState[i].cbIdx_) {
          // Read resource index
          if (!getuint(metaData, &pos, &arg.cbIdx_)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }

          if (arg.isCbNeeded() && (numCb_ < arg.cbIdx_)) {
            numCb_ = arg.cbIdx_;
          }
        }
        // Check if we should expect the CB offset
        if (ArgState[i].cbPos_) {
          // Read position in the constant buffer
          if (!getuint(metaData, &pos, &arg.cbPos_)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }
        }
        // Check if we should expect the buffer type
        if (ArgState[i].buf_) {
          // Read the buffer type
          if (!getword(metaData, &pos, argName)) {
            LogWarning("Couldn't get a kernel argument!");
            return false;
          }
          arg.buf_ = argName;

          for (uint k = 0; k < BufTypeTotal; ++k) {
            if (0 == arg.buf_.compare(BufType[k].tagName_)) {
              // Update the parameter type
              arg.type_ = BufType[k].type_;
              // Check if we should expect a buffer index
              if (BufType[k].number_) {
                // Read a buffer index
                if (!getuint(metaData, &pos, &arg.index_)) {
                  LogWarning("Couldn't get a kernel argument!");
                  return false;
                }
              }
              // Check for the required alignment
              if (BufType[k].alignment_) {
                // Read data alignment
                if (!getuint(metaData, &pos, &arg.alignment_)) {
                  LogWarning("Couldn't get a kernel argument!");
                  return false;
                }
              }
              // Check for the buffer's attribute
              if ((mdVersion.value_ >= MetadataBufferAttributes.value_) && BufType[k].attribute_) {
                if (expect(metaData, &pos, "RO")) {
                  arg.memory_.readOnly_ = 1;
                } else if (expect(metaData, &pos, "RW")) {
                  arg.memory_.readWrite_ = 1;
                } else if (expect(metaData, &pos, "WO")) {
                  arg.memory_.writeOnly_ = 1;
                }
              }
              // Check for the type qualifier
              if ((mdVersion.value_ >= MetadataTypeQualifiers.value_) && BufType[k].attribute_) {
                uint tmp;
                pos += 1;
                if (!getuint(metaData, &pos, &tmp)) {
                  LogWarning("Couldn't get volatile type!");
                  return false;
                }
                if (tmp == 1) {
                  arg.typeQualifier_ |= CL_KERNEL_ARG_TYPE_VOLATILE;
                }
                if (!getuint(metaData, &pos, &tmp)) {
                  LogWarning("Couldn't get restrict type!");
                  return false;
                }
                if (tmp == 1) {
                  arg.typeQualifier_ |= CL_KERNEL_ARG_TYPE_RESTRICT;
                }
              }
            }
          }
        }
        // Find multiple UAV references
        switch (arg.type_) {
          case KernelArg::PointerGlobal:
          case KernelArg::PointerConst:
          case KernelArg::PointerLocal:
          case KernelArg::PointerPrivate:
          case KernelArg::UavId:
            uavRefCount[arg.index_]++;
            break;
          default:
            break;
        }
        // Check if this argument will be passed in constant buffer
        if (arg.isCbNeeded() || (arg.type_ == KernelArg::UavId)) {
          if (arg.type_ == KernelArg::Sampler) {
            // Serach for the passed by value sampler
            for (uint i = 0; i < argSize(); ++i) {
              KernelArg* value = arguments_[i];
              if (0 == value->name_.compare(arg.name_)) {
                value->type_ = arg.type_;
                value->index_ = arg.index_;
                value->location_ = 0;
                break;
              }
            }
          } else {
            KernelArg* argument = new KernelArg(arg);
            if (argument != NULL) {
              addArgument(argument);
            } else {
              LogError("Couldn't allocate memory!");
              return false;
            }
          }
        }
        // Check if we have a pre-defined sampler
        else if (arg.type_ == KernelArg::Sampler) {
          KernelArg* sampler = new KernelArg(arg);
          if (sampler != NULL) {
            addSampler(sampler);
          } else {
            LogError("Couldn't allocate memory!");
            return false;
          }
        }
        break;
      }
    }

    // Next argument
    pos = metaData.find(";", pos);
  }

  // Find arguments that will require a reallocation
  for (uint i = 0; i < arguments_.size(); ++i) {
    KernelArg* arg = arguments_[i];
    switch (arg->type_) {
      case KernelArg::PointerGlobal:
      case KernelArg::PointerConst:
      case KernelArg::PointerLocal:
      case KernelArg::PointerPrivate:
        // Check if can't use a dedicated UAV,
        // so realloc memory in the heap
        arg->memory_.realloc_ = false;
        arg->memory_.uavBuf_ = true;
        break;
      case KernelArg::PointerHwConst:
        arg->memory_.realloc_ = true;
        break;
      case KernelArg::UavId:
        uavRaw_ = arg->index_;
        break;
      default:
        break;
    }
    // If argument marked with the const qualifier, then overwrite
    // Read-Write attributes, since compiler doesn't mark it properly
    if (arg->typeQualifier() & CL_KERNEL_ARG_TYPE_CONST) {
      arg->memory_.readOnly_ = 1;
      arg->memory_.readWrite_ = 0;
      arg->memory_.writeOnly_ = 0;
    }
  }

  if ((uavRaw_ != UavIdUndefined) && !(flags() & PrintfOutput)) {
    // Find if default UAV is already assigned to an argument
    for (uint i = 0; i < arguments_.size(); ++i) {
      KernelArg* arg = arguments_[i];
      switch (arg->type_) {
        case KernelArg::PointerGlobal:
        case KernelArg::PointerConst:
        case KernelArg::PointerLocal:
        case KernelArg::PointerPrivate:
          if (uavRaw_ == arg->index_) {
            uavRaw_ = UavIdUndefined;
          }
          break;
        default:
          break;
      }
    }
  }

  // There is always 1 constant buffer, associated with the kernel
  numCb_++;
  assert((numCb_ <= MaxConstBuffersArguments) &&
         "Runtime doesn't support more than max CBs for arguments!");

  // Limit workgroup size if requested
  if ((flags() & LimitWorkgroup) && (GPU_MAX_WORKGROUP_SIZE == 0)) {
    size_t temp = 1;
    workGroupInfo_.size_ = workGroupInfo()->wavefrontSize_;
    for (uint j = 0; j < 3; ++j) {
      if (workGroupInfo()->compileSize_[j] != 0) {
        temp *= workGroupInfo_.compileSize_[j];
      }
    }
    // Report a compilation error if requested compile size doesn't
    // match the required workgroup size
    if (workGroupInfo()->size_ < temp) {
      char str[8];
      intToStr(workGroupInfo_.size_, str, 8);
      buildError_ = CL_OUT_OF_RESOURCES;
      buildLog_ += "Error: Requested compile size is bigger than the required workgroup size of ";
      buildLog_ += str;
      buildLog_ += " elements\n";
      LogError(buildLog().c_str());
      return false;
    }
  }

  // Read/Write attributes are provided in metadata
  if (mdVersion.value_ >= MetadataBufferAttributes.value_) {
    rwAttributes_ = true;
  }

  return true;
}

inline static HSAIL_ARG_TYPE GetHSAILArgType(const aclArgData* argInfo) {
  switch (argInfo->type) {
    case ARG_TYPE_POINTER:
      return HSAIL_ARGTYPE_POINTER;
    case ARG_TYPE_QUEUE:
      return HSAIL_ARGTYPE_QUEUE;
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

inline static size_t GetHSAILArgAlignment(const aclArgData* argInfo) {
  switch (argInfo->type) {
    case ARG_TYPE_POINTER:
      return argInfo->arg.pointer.align;
    default:
      return 1;
  }
}

inline static HSAIL_ACCESS_TYPE GetHSAILArgAccessType(const aclArgData* argInfo) {
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

inline static HSAIL_ADDRESS_QUALIFIER GetHSAILAddrQual(const aclArgData* argInfo) {
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
      case PTR_MT_SCRATCH_EMU:
        return HSAIL_ADDRESS_GLOBAL;
      case PTR_MT_ERROR:
      default:
        LogError("Unsupported address type");
        return HSAIL_ADDRESS_ERROR;
    }
  } else if ((argInfo->type == ARG_TYPE_IMAGE) || (argInfo->type == ARG_TYPE_SAMPLER)) {
    return HSAIL_ADDRESS_GLOBAL;
  } else if (argInfo->type == ARG_TYPE_QUEUE) {
    return HSAIL_ADDRESS_GLOBAL;
  }
  return HSAIL_ADDRESS_ERROR;
}

/* f16 returns f32 - workaround due to comp lib */
inline static HSAIL_DATA_TYPE GetHSAILDataType(const aclArgData* argInfo) {
  aclArgDataType dataType;

  if (argInfo->type == ARG_TYPE_POINTER) {
    dataType = argInfo->arg.pointer.data;
  } else if (argInfo->type == ARG_TYPE_VALUE) {
    dataType = argInfo->arg.value.data;
  } else {
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

inline static int GetHSAILArgSize(const aclArgData* argInfo) {
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
    case ARG_TYPE_QUEUE:
      return sizeof(void*);
    default:
      return -1;
  }
}

inline static clk_value_type_t GetOclType(const aclArgData* argInfo) {
  static const clk_value_type_t ClkValueMapType[6][6] = {
      {T_CHAR, T_CHAR2, T_CHAR3, T_CHAR4, T_CHAR8, T_CHAR16},
      {T_SHORT, T_SHORT2, T_SHORT3, T_SHORT4, T_SHORT8, T_SHORT16},
      {T_INT, T_INT2, T_INT3, T_INT4, T_INT8, T_INT16},
      {T_LONG, T_LONG2, T_LONG3, T_LONG4, T_LONG8, T_LONG16},
      {T_FLOAT, T_FLOAT2, T_FLOAT3, T_FLOAT4, T_FLOAT8, T_FLOAT16},
      {T_DOUBLE, T_DOUBLE2, T_DOUBLE3, T_DOUBLE4, T_DOUBLE8, T_DOUBLE16},
  };

  uint sizeType;
  if (argInfo->type == ARG_TYPE_QUEUE) {
    return T_QUEUE;
  }
  if ((argInfo->type == ARG_TYPE_POINTER) || (argInfo->type == ARG_TYPE_IMAGE)) {
    return T_POINTER;
  } else if (argInfo->type == ARG_TYPE_VALUE) {
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
  } else if (argInfo->type == ARG_TYPE_SAMPLER) {
    return T_SAMPLER;
  } else {
    return T_VOID;
  }
}

inline static cl_kernel_arg_address_qualifier GetOclAddrQual(const aclArgData* argInfo) {
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
  } else if (argInfo->type == ARG_TYPE_IMAGE) {
    return CL_KERNEL_ARG_ADDRESS_GLOBAL;
  }
  // default for all other cases
  return CL_KERNEL_ARG_ADDRESS_PRIVATE;
}

inline static cl_kernel_arg_access_qualifier GetOclAccessQual(const aclArgData* argInfo) {
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

inline static cl_kernel_arg_type_qualifier GetOclTypeQual(const aclArgData* argInfo) {
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

static int GetOclSize(const aclArgData* argInfo) {
  switch (argInfo->type) {
    case ARG_TYPE_POINTER:
      return sizeof(void*);
    case ARG_TYPE_VALUE:
      //! \note OCL 6.1.5. For 3-component vector data types,
      //! the size of the data type is 4 * sizeof(component).
      switch (argInfo->arg.value.data) {
        case DATATYPE_struct:
          return 1 * argInfo->arg.value.numElements;
        case DATATYPE_i8:
        case DATATYPE_u8:
          return 1 * amd::nextPowerOfTwo(argInfo->arg.value.numElements);
        case DATATYPE_u16:
        case DATATYPE_i16:
        case DATATYPE_f16:
          return 2 * amd::nextPowerOfTwo(argInfo->arg.value.numElements);
        case DATATYPE_u32:
        case DATATYPE_i32:
        case DATATYPE_f32:
          return 4 * amd::nextPowerOfTwo(argInfo->arg.value.numElements);
        case DATATYPE_i64:
        case DATATYPE_u64:
        case DATATYPE_f64:
          return 8 * amd::nextPowerOfTwo(argInfo->arg.value.numElements);
        case DATATYPE_ERROR:
        default:
          return -1;
      }
    case ARG_TYPE_IMAGE:
      return sizeof(cl_mem);
    case ARG_TYPE_SAMPLER:
      return sizeof(cl_sampler);
    case ARG_TYPE_QUEUE:
      return sizeof(cl_command_queue);
    default:
      return -1;
  }
}

void HSAILKernel::initArgList(const aclArgData* aclArg) {
  // Initialize the hsail argument list too
  initHsailArgs(aclArg);

  // Iterate through the arguments and insert into parameterList
  device::Kernel::parameters_t params;
  amd::KernelParameterDescriptor desc;
  size_t offset = 0;

  // Reserved arguments for HSAIL launch
  aclArg += MaxExtraArgumentsNum;
  for (uint i = 0; aclArg->struct_size != 0; i++, aclArg++) {
    desc.name_ = arguments_[i]->name_.c_str();
    desc.type_ = GetOclType(aclArg);
    desc.addressQualifier_ = GetOclAddrQual(aclArg);
    desc.accessQualifier_ = GetOclAccessQual(aclArg);
    desc.typeQualifier_ = GetOclTypeQual(aclArg);
    desc.typeName_ = arguments_[i]->typeName_.c_str();

    // Make a check if it is local or global
    if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
      desc.size_ = sizeof(cl_mem);
    } else {
      desc.size_ = GetOclSize(aclArg);
    }

    // Make offset alignment to match CPU metadata, since
    // in multidevice config abstraction layer has a single signature
    // and CPU sends the paramaters as they are allocated in memory
    size_t size = desc.size_;

    offset = amd::alignUp(offset, std::min(size, size_t(16)));
    desc.offset_ = offset;
    offset += amd::alignUp(size, sizeof(uint32_t));
    params.push_back(desc);

    if (arguments_[i]->type_ == HSAIL_ARGTYPE_IMAGE) {
      flags_.imageEna_ = true;
      if (desc.accessQualifier_ != CL_KERNEL_ARG_ACCESS_READ_ONLY) {
        flags_.imageWriteEna_ = true;
      }
    }
  }

  createSignature(params, params.size(), amd::KernelSignature::ABIVersion_0);
}

void HSAILKernel::initHsailArgs(const aclArgData* aclArg) {
  int offset = 0;

  // Reserved arguments for HSAIL launch
  aclArg += MaxExtraArgumentsNum;

  // Iterate through the each kernel argument
  for (; aclArg->struct_size != 0; aclArg++) {
    Argument* arg = new Argument;
    // Initialize HSAIL kernel argument
    arg->name_ = aclArg->argStr;
    arg->typeName_ = aclArg->typeStr;
    arg->size_ = GetHSAILArgSize(aclArg);
    arg->offset_ = offset;
    arg->type_ = GetHSAILArgType(aclArg);
    arg->addrQual_ = GetHSAILAddrQual(aclArg);
    arg->dataType_ = GetHSAILDataType(aclArg);
    // If vector of args we add additional arguments to flatten it out
    arg->numElem_ =
        ((aclArg->type == ARG_TYPE_VALUE) && (aclArg->arg.value.data != DATATYPE_struct))
        ? aclArg->arg.value.numElements
        : 1;
    arg->alignment_ = GetHSAILArgAlignment(aclArg);
    arg->access_ = GetHSAILArgAccessType(aclArg);
    offset += GetHSAILArgSize(aclArg);
    arguments_.push_back(arg);
  }
}

HSAILKernel::HSAILKernel(std::string name, HSAILProgram* prog, std::string compileOptions,
                         uint extraArgsNum)
    : device::Kernel(prog->dev(), name, *prog),
      compileOptions_(compileOptions),
      index_(0),
      code_(NULL),
      codeSize_(0),
      hwMetaData_(NULL),
      extraArgumentsNum_(extraArgsNum) {
  flags_.hsa_ = true;
}

HSAILKernel::~HSAILKernel() {
  while (!arguments_.empty()) {
    Argument* arg = arguments_.back();
    delete arg;
    arguments_.pop_back();
  }

  delete[] hwMetaData_;

  delete code_;
}

bool HSAILKernel::init(amd::hsa::loader::Symbol* sym, bool finalize) {
  if (extraArgumentsNum_ > MaxExtraArgumentsNum) {
    LogError("Failed to initialize kernel: extra arguments number is bigger than is supported");
    return false;
  }
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
    error = aclCompile(dev().hsaCompiler(), prog().binaryElf(), options.c_str(), ACL_TYPE_CG,
                       ACL_TYPE_ISA, NULL);
    buildLog_ += aclGetCompilerLog(dev().hsaCompiler());
    if (error != ACL_SUCCESS) {
      LogError("Failed to finalize kernel");
      return false;
    }
  }

  aqlCreateHWInfo(sym);

  // Pull out metadata from the ELF
  size_t sizeOfArgList;
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_ARGUMENT_ARRAY,
                       openClKernelName.c_str(), NULL, &sizeOfArgList);
  if (error != ACL_SUCCESS) {
    return false;
  }

  char* aclArgList = new char[sizeOfArgList];
  if (NULL == aclArgList) {
    return false;
  }
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_ARGUMENT_ARRAY,
                       openClKernelName.c_str(), aclArgList, &sizeOfArgList);
  if (error != ACL_SUCCESS) {
    return false;
  }

  size_t sizeOfWorkGroupSize;
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_WORK_GROUP_SIZE,
                       openClKernelName.c_str(), NULL, &sizeOfWorkGroupSize);
  if (error != ACL_SUCCESS) {
    return false;
  }
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_WORK_GROUP_SIZE,
                       openClKernelName.c_str(), workGroupInfo_.compileSize_, &sizeOfWorkGroupSize);
  if (error != ACL_SUCCESS) {
    return false;
  }

  // Copy wavefront size
  workGroupInfo_.wavefrontSize_ = prog().isNull() ? 64 : dev().getAttribs().wavefrontSize;

  // Find total workgroup size
  if (workGroupInfo_.compileSize_[0] != 0) {
    workGroupInfo_.size_ = workGroupInfo_.compileSize_[0] * workGroupInfo_.compileSize_[1] *
        workGroupInfo_.compileSize_[2];
  } else {
    workGroupInfo_.size_ = dev().info().preferredWorkGroupSize_;
  }

  // Pull out printf metadata from the ELF
  size_t sizeOfPrintfList;
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_GPU_PRINTF_ARRAY,
                       openClKernelName.c_str(), NULL, &sizeOfPrintfList);
  if (error != ACL_SUCCESS) {
    return false;
  }

  // Make sure kernel has any printf info
  if (0 != sizeOfPrintfList) {
    char* aclPrintfList = new char[sizeOfPrintfList];
    if (NULL == aclPrintfList) {
      return false;
    }
    error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_GPU_PRINTF_ARRAY,
                         openClKernelName.c_str(), aclPrintfList, &sizeOfPrintfList);
    if (error != ACL_SUCCESS) {
      return false;
    }

    // Set the PrintfList
    InitPrintf(reinterpret_cast<aclPrintfFmt*>(aclPrintfList));
    delete[] aclPrintfList;
  }

  aclMetadata md;
  md.enqueue_kernel = false;
  size_t sizeOfDeviceEnqueue = sizeof(md.enqueue_kernel);
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_DEVICE_ENQUEUE,
                       openClKernelName.c_str(), &md.enqueue_kernel, &sizeOfDeviceEnqueue);
  if (error != ACL_SUCCESS) {
    return false;
  }
  flags_.dynamicParallelism_ = md.enqueue_kernel;

  md.kernel_index = -1;
  size_t sizeOfIndex = sizeof(md.kernel_index);
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_KERNEL_INDEX,
                       openClKernelName.c_str(), &md.kernel_index, &sizeOfIndex);
  if (error != ACL_SUCCESS) {
    return false;
  }
  index_ = md.kernel_index;

  size_t sizeOfWavesPerSimdHint = sizeof(workGroupInfo_.wavesPerSimdHint_);
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_WAVES_PER_SIMD_HINT,
                       openClKernelName.c_str(), &workGroupInfo_.wavesPerSimdHint_,
                       &sizeOfWavesPerSimdHint);
  if (error != ACL_SUCCESS) {
    return false;
  }

  waveLimiter_.enable(dev().settings().ciPlus_);

  size_t sizeOfWorkGroupSizeHint = sizeof(workGroupInfo_.compileSizeHint_);
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_WORK_GROUP_SIZE_HINT,
                       openClKernelName.c_str(), workGroupInfo_.compileSizeHint_,
                       &sizeOfWorkGroupSizeHint);
  if (error != ACL_SUCCESS) {
    return false;
  }

  size_t sizeOfVecTypeHint;
  error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_VEC_TYPE_HINT,
                       openClKernelName.c_str(), NULL, &sizeOfVecTypeHint);
  if (error != ACL_SUCCESS) {
    return false;
  }

  if (0 != sizeOfVecTypeHint) {
    char* VecTypeHint = new char[sizeOfVecTypeHint + 1];
    if (NULL == VecTypeHint) {
      return false;
    }
    error = aclQueryInfo(dev().hsaCompiler(), prog().binaryElf(), RT_VEC_TYPE_HINT,
                         openClKernelName.c_str(), VecTypeHint, &sizeOfVecTypeHint);
    if (error != ACL_SUCCESS) {
      return false;
    }
    VecTypeHint[sizeOfVecTypeHint] = '\0';
    workGroupInfo_.compileVecTypeHint_ = std::string(VecTypeHint);
    delete[] VecTypeHint;
  }

  // Set the argList
  initArgList(reinterpret_cast<const aclArgData*>(aclArgList));
  delete[] aclArgList;

  return true;
}

const Device& HSAILKernel::dev() const { return reinterpret_cast<const Device&>(dev_); }

const HSAILProgram& HSAILKernel::prog() const {
  return reinterpret_cast<const HSAILProgram&>(prog_);
}

inline static void WriteAqlArg(
    unsigned char** dst,  //!< The write pointer to the buffer
    const void* src,      //!< The source pointer
    uint size,            //!< The size in bytes to copy
    uint alignment = 0    //!< The alignment to follow while writing to the buffer
    ) {
  if (alignment == 0) {
    *dst = amd::alignUp(*dst, size);
  } else {
    *dst = amd::alignUp(*dst, alignment);
  }
  memcpy(*dst, src, size);
  *dst += size;
}

const uint16_t kDispatchPacketHeader = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

hsa_kernel_dispatch_packet_t* HSAILKernel::loadArguments(
    VirtualGPU& gpu, const amd::Kernel& kernel, const amd::NDRangeContainer& sizes,
    const_address parameters, bool nativeMem, uint64_t vmDefQueue, uint64_t* vmParentWrap,
    std::vector<const Memory*>& memList) const {
  static const bool WaitOnBusyEngine = true;
  uint64_t ldsAddress = ldsSize();
  address aqlArgBuf = gpu.cb(0)->sysMemCopy();
  address aqlStruct = gpu.cb(1)->sysMemCopy();
  bool srdResource = false;

  if (extraArgumentsNum_ > 0) {
    assert(MaxExtraArgumentsNum >= 6 &&
           "MaxExtraArgumentsNum has changed, the below algorithm should be changed accordingly");
    size_t extraArgs[MaxExtraArgumentsNum] = {0, 0, 0, 0, 0, 0};
    // The HLC generates up to 3 additional arguments for the global offsets
    for (uint i = 0; i < sizes.dimensions(); ++i) {
      extraArgs[i] = sizes.offset()[i];
    }
    // Check if the kernel may have printf output
    if ((printfInfo().size() > 0) &&
        // and printf buffer was allocated
        (gpu.printfDbgHSA().dbgBuffer() != NULL)) {
      // and set the fourth argument as the printf_buffer pointer
      extraArgs[3] = static_cast<size_t>(gpu.printfDbgHSA().dbgBuffer()->vmAddress());
      memList.push_back(gpu.printfDbgHSA().dbgBuffer());
    }
    if (dynamicParallelism()) {
      // Provide the host parent AQL wrap object to the kernel
      AmdAqlWrap* wrap = reinterpret_cast<AmdAqlWrap*>(aqlStruct);
      memset(wrap, 0, sizeof(AmdAqlWrap));
      wrap->state = AQL_WRAP_BUSY;
      ConstBuffer* cb = gpu.constBufs_[1];
      cb->uploadDataToHw(sizeof(AmdAqlWrap));
      *vmParentWrap = cb->vmAddress() + cb->wrtOffset();
      // and set 5th & 6th arguments
      extraArgs[4] = vmDefQueue;
      extraArgs[5] = *vmParentWrap;
      memList.push_back(cb);
    }
    WriteAqlArg(&aqlArgBuf, extraArgs, sizeof(size_t) * extraArgumentsNum_, sizeof(size_t));
  }

  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  amd::Memory* const* memories =
    reinterpret_cast<amd::Memory* const*>(parameters + kernelParams.memoryObjOffset());

  // Find all parameters for the current kernel
  for (uint i = 0; i != signature.numParameters(); ++i) {
    const HSAILKernel::Argument* arg = argument(i);
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    const_address paramaddr = parameters + desc.offset_;

    switch (arg->type_) {
      case HSAIL_ARGTYPE_POINTER:
        // If it is a global pointer
        if (arg->addrQual_ == HSAIL_ADDRESS_GLOBAL) {
          Memory* gpuMem = NULL;
          amd::Memory* mem = NULL;

          uint32_t index = signature.at(i).info_.arrayIndex_;
          if (nativeMem) {
            gpuMem = reinterpret_cast<Memory* const*>(memories)[index];
            if (nullptr != gpuMem) {
              mem = gpuMem->owner();
            }
          } else {
            mem = memories[index];
            if (mem != nullptr) {
              gpuMem = dev().getGpuMemory(mem);
            }
          }

          WriteAqlArg(&aqlArgBuf, paramaddr, sizeof(paramaddr), sizeof(paramaddr));
          if (gpuMem == nullptr) {
            break;
          }

          // Wait for resource if it was used on an inactive engine
          //! \note syncCache may call DRM transfer
          gpuMem->wait(gpu, WaitOnBusyEngine);

          //! @todo Compiler has to return read/write attributes
          if ((NULL != mem) && ((mem->getMemFlags() & CL_MEM_READ_ONLY) == 0)) {
            mem->signalWrite(&dev());
          }
          memList.push_back(gpuMem);

          // save the memory object pointer to allow global memory access
          if (NULL != dev().hwDebugMgr()) {
            dev().hwDebugMgr()->assignKernelParamMem(i, gpuMem->owner());
          }
        }
        // If it is a local pointer
        else {
          assert((arg->addrQual_ == HSAIL_ADDRESS_LOCAL) && "Unsupported address type");
          ldsAddress = amd::alignUp(ldsAddress, arg->alignment_);
          WriteAqlArg(&aqlArgBuf, &ldsAddress, desc.size_);
          if (desc.size_ == 8) {
            ldsAddress += *reinterpret_cast<const uint64_t*>(paramaddr);
          } else {
            ldsAddress += *reinterpret_cast<const uint32_t*>(paramaddr);
          }
        }
        break;
      case HSAIL_ARGTYPE_VALUE:
        // Special case for structrues
        if (arg->dataType_ == HSAIL_DATATYPE_STRUCT) {
          // Copy the current structre into CB1
          memcpy(aqlStruct, paramaddr, arg->size_);
          ConstBuffer* cb = gpu.constBufs_[1];
          cb->uploadDataToHw(arg->size_);
          // Then use a pointer in aqlArgBuffer to CB1
          uint64_t gpuPtr = cb->vmAddress() + cb->wrtOffset();
          WriteAqlArg(&aqlArgBuf, &gpuPtr, sizeof(void*));
          memList.push_back(cb);
        } else {
          WriteAqlArg(&aqlArgBuf, paramaddr, arg->numElem_ * arg->size_, arg->size_);
        }
        break;
      case HSAIL_ARGTYPE_IMAGE: {
        Image* image = nullptr;
        amd::Memory* mem = nullptr;
        uint32_t index = signature.at(i).info_.arrayIndex_;
        if (nativeMem) {
          image = reinterpret_cast<Image* const*>(memories)[index];
          if (nullptr != image) {
            mem = image->owner();
          }
        } else {
          mem = memories[index];
          if (mem == NULL) {
            LogError("The kernel image argument isn't an image object!");
            return nullptr;
          }
          image = static_cast<Image*>(dev().getGpuMemory(mem));
        }

        // Wait for resource if it was used on an inactive engine
        //! \note syncCache may call DRM transfer
        image->wait(gpu, WaitOnBusyEngine);

        //! \note Special case for the image views.
        //! Copy SRD to CB1, so blit manager will be able to release
        //! this view without a wait for SRD resource.
        if (image->memoryType() == Resource::ImageView) {
          // Copy the current structre into CB1
          memcpy(aqlStruct, image->hwState(), HsaImageObjectSize);
          ConstBuffer* cb = gpu.constBufs_[1];
          cb->uploadDataToHw(HsaImageObjectSize);
          // Then use a pointer in aqlArgBuffer to CB1
          uint64_t srd = cb->vmAddress() + cb->wrtOffset();
          WriteAqlArg(&aqlArgBuf, &srd, sizeof(srd));
          memList.push_back(cb);
        } else {
          uint64_t srd = image->hwSrd();
          WriteAqlArg(&aqlArgBuf, &srd, sizeof(srd));
          srdResource = true;
        }

        //! @todo Compiler has to return read/write attributes
        if ((NULL != mem) && ((mem->getMemFlags() & CL_MEM_READ_ONLY) == 0)) {
          mem->signalWrite(&dev());
        }

        memList.push_back(image);
        break;
      }
      case HSAIL_ARGTYPE_SAMPLER: {
        uint32_t index = signature.at(i).info_.arrayIndex_;
        const amd::Sampler* sampler = reinterpret_cast<amd::Sampler* const*>(parameters +
            kernelParams.samplerObjOffset())[index];
        const Sampler* gpuSampler = static_cast<Sampler*>(sampler->getDeviceSampler(dev()));
        uint64_t srd = gpuSampler->hwSrd();
        WriteAqlArg(&aqlArgBuf, &srd, sizeof(srd));
        srdResource = true;
        break;
      }
      case HSAIL_ARGTYPE_QUEUE: {
        uint32_t index = signature.at(i).info_.arrayIndex_;
        const amd::DeviceQueue* queue = reinterpret_cast<amd::DeviceQueue* const*>(
          parameters + kernelParams.queueObjOffset())[index];
        VirtualGPU* gpuQueue = static_cast<VirtualGPU*>(queue->vDev());
        uint64_t vmQueue;
        if (dev().settings().useDeviceQueue_) {
          vmQueue = gpuQueue->vQueue()->vmAddress();
        } else {
          if (!gpu.createVirtualQueue(queue->size())) {
            LogError("Virtual queue creation failed!");
            return nullptr;
          }
          vmQueue = gpu.vQueue()->vmAddress();
        }
        WriteAqlArg(&aqlArgBuf, &vmQueue, sizeof(void*));
        break;
      }
      default:
        LogError(" Unsupported address type ");
        return NULL;
    }
  }

  if (ldsAddress > dev().info().localMemSize_) {
    LogError("No local memory available\n");
    return NULL;
  }

  // HSAIL kernarg segment size is rounded up to multiple of 16.
  aqlArgBuf = amd::alignUp(aqlArgBuf, 16);
  assert((aqlArgBuf == (gpu.cb(0)->sysMemCopy() + argsBufferSize())) &&
         "Size and the number of arguments don't match!");
  hsa_kernel_dispatch_packet_t* hsaDisp =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(aqlArgBuf);

  amd::NDRange local(sizes.local());
  const amd::NDRange& global = sizes.global();

  // Check if runtime has to find local workgroup size
  FindLocalWorkSize(sizes.dimensions(), sizes.global(), local);

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
  hsaDisp->group_segment_size = ldsAddress;
  hsaDisp->kernel_object = gpuAqlCode()->vmAddress();

  ConstBuffer* cb = gpu.constBufs_[0];
  cb->uploadDataToHw(argsBufferSize() + sizeof(hsa_kernel_dispatch_packet_t));
  uint64_t argList = cb->vmAddress() + cb->wrtOffset();

  hsaDisp->kernarg_address = reinterpret_cast<void*>(argList);
  hsaDisp->reserved2 = 0;
  hsaDisp->completion_signal.handle = 0;

  memList.push_back(cb);
  memList.push_back(gpuAqlCode());
  for (gpu::Memory* mem : prog().globalStores()) {
    memList.push_back(mem);
  }
  if (AMD_HSA_BITS_GET(cpuAqlCode_->kernel_code_properties,
                       AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)) {
    memList.push_back(gpu.hsaQueueMem());
  }

  if (srdResource || prog().isStaticSampler()) {
    dev().srds().fillResourceList(memList);
  }

  return hsaDisp;
}

}  // namespace gpu
