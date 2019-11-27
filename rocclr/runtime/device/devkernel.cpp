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
#include "comgrctx.hpp"

#include <map>
#include <string>
#include <sstream>

#include "acl.h"

#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
#include "llvm/Support/AMDGPUMetadata.h"

typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;

using llvm::AMDGPU::HSAMD::AccessQualifier;
using llvm::AMDGPU::HSAMD::AddressSpaceQualifier;
using llvm::AMDGPU::HSAMD::ValueKind;
using llvm::AMDGPU::HSAMD::ValueType;
#endif  // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

namespace device {

#if defined(USE_COMGR_LIBRARY)
amd_comgr_status_t getMetaBuf(const amd_comgr_metadata_node_t meta,
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
      kernelMD->mCodeProps.mGroupSegmentFixedSize = atoi(buf.c_str());
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

static amd_comgr_status_t populateArgsV3(const amd_comgr_metadata_node_t key,
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

  auto itArgField = ArgFieldMapV3.find(buf);
  if (itArgField == ArgFieldMapV3.end()) {
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
    case ArgField::Offset:
      lcArg->mOffset = atoi(buf.c_str());
      break;
    case ArgField::ValueKind:
      {
        auto itValueKind = ArgValueKindV3.find(buf);
        if (itValueKind == ArgValueKindV3.end()) {
          return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mValueKind = itValueKind->second;
      }
      break;
    case ArgField::ValueType:
      {
        auto itValueType = ArgValueTypeV3.find(buf);
        if (itValueType == ArgValueTypeV3.end()) {
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
        auto itAddrSpaceQual = ArgAddrSpaceQualV3.find(buf);
        if (itAddrSpaceQual == ArgAddrSpaceQualV3.end()) {
          return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mAddrSpaceQual = itAddrSpaceQual->second;
      }
      break;
    case ArgField::AccQual:
      {
        auto itAccQual = ArgAccQualV3.find(buf);
        if (itAccQual == ArgAccQualV3.end()) {
          return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mAccQual = itAccQual->second;
      }
      break;
    case ArgField::ActualAccQual:
      {
        auto itAccQual = ArgAccQualV3.find(buf);
        if (itAccQual == ArgAccQualV3.end()) {
            return AMD_COMGR_STATUS_ERROR;
        }
        lcArg->mActualAccQual = itAccQual->second;
      }
      break;
    case ArgField::IsConst:
      lcArg->mIsConst = (buf.compare("1") == 0);
      break;
    case ArgField::IsRestrict:
      lcArg->mIsRestrict = (buf.compare("1") == 0);
      break;
    case ArgField::IsVolatile:
      lcArg->mIsVolatile = (buf.compare("1") == 0);
      break;
    case ArgField::IsPipe:
      lcArg->mIsPipe = (buf.compare("1") == 0);
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateKernelMetaV3(const amd_comgr_metadata_node_t key,
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

  auto itKernelField = KernelFieldMapV3.find(buf);
  if (itKernelField == KernelFieldMapV3.end()) {
    return AMD_COMGR_STATUS_ERROR;
  }

  if (itKernelField->second != KernelField::ReqdWorkGroupSize &&
      itKernelField->second != KernelField::WorkGroupSizeHint) {
      status = getMetaBuf(value,&buf);
  }
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  KernelMD* kernelMD = static_cast<KernelMD*>(data);
  switch (itKernelField->second) {
    case KernelField::ReqdWorkGroupSize:
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
      break;
    case KernelField::WorkGroupSizeHint:
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
      break;
    case KernelField::VecTypeHint:
      kernelMD->mAttrs.mVecTypeHint = buf;
      break;
    case KernelField::DeviceEnqueueSymbol:
      kernelMD->mAttrs.mRuntimeHandle = buf;
      break;
    case KernelField::KernargSegmentSize:
      kernelMD->mCodeProps.mKernargSegmentSize = atoi(buf.c_str());
      break;
    case KernelField::GroupSegmentFixedSize:
      kernelMD->mCodeProps.mGroupSegmentFixedSize = atoi(buf.c_str());
      break;
    case KernelField::PrivateSegmentFixedSize:
      kernelMD->mCodeProps.mPrivateSegmentFixedSize = atoi(buf.c_str());
      break;
    case KernelField::KernargSegmentAlign:
      kernelMD->mCodeProps.mKernargSegmentAlign = atoi(buf.c_str());
      break;
    case KernelField::WavefrontSize:
      kernelMD->mCodeProps.mWavefrontSize = atoi(buf.c_str());
      break;
    case KernelField::NumSGPRs:
      kernelMD->mCodeProps.mNumSGPRs = atoi(buf.c_str());
      break;
    case KernelField::NumVGPRs:
      kernelMD->mCodeProps.mNumVGPRs = atoi(buf.c_str());
      break;
    case KernelField::MaxFlatWorkGroupSize:
      kernelMD->mCodeProps.mMaxFlatWorkGroupSize = atoi(buf.c_str());
      break;
    case KernelField::NumSpilledSGPRs:
      kernelMD->mCodeProps.mNumSpilledSGPRs = atoi(buf.c_str());
      break;
    case KernelField::NumSpilledVGPRs:
      kernelMD->mCodeProps.mNumSpilledVGPRs = atoi(buf.c_str());
      break;
    case KernelField::SymbolName:
      kernelMD->mSymbolName = buf;
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }

  return status;
}
#endif

// ================================================================================================
Kernel::Kernel(const amd::Device& dev, const std::string& name, const Program& prog)
  : dev_(dev)
  , name_(name)
  , prog_(prog)
  , signature_(nullptr)
  , waveLimiter_(this, dev.info().cuPerShaderArray_ * dev.info().simdPerCU_) {
  // Instead of memset(&workGroupInfo_, '\0', sizeof(workGroupInfo_));
  // Due to std::string not being able to be memset to 0
  workGroupInfo_.size_ = 0;
  workGroupInfo_.compileSize_[0] = 0;
  workGroupInfo_.compileSize_[1] = 0;
  workGroupInfo_.compileSize_[2] = 0;
  workGroupInfo_.localMemSize_ = 0;
  workGroupInfo_.preferredSizeMultiple_ = 0;
  workGroupInfo_.privateMemSize_ = 0;
  workGroupInfo_.scratchRegs_ = 0;
  workGroupInfo_.wavefrontPerSIMD_ = 0;
  workGroupInfo_.wavefrontSize_ = 0;
  workGroupInfo_.availableGPRs_ = 0;
  workGroupInfo_.usedGPRs_ = 0;
  workGroupInfo_.availableSGPRs_ = 0;
  workGroupInfo_.usedSGPRs_ = 0;
  workGroupInfo_.availableVGPRs_ = 0;
  workGroupInfo_.usedVGPRs_ = 0;
  workGroupInfo_.availableLDSSize_ = 0;
  workGroupInfo_.usedLDSSize_ = 0;
  workGroupInfo_.availableStackSize_ = 0;
  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.compileSizeHint_[0] = 0;
  workGroupInfo_.compileSizeHint_[1] = 0;
  workGroupInfo_.compileSizeHint_[2] = 0;
  workGroupInfo_.compileVecTypeHint_ = "";
  workGroupInfo_.uniformWorkGroupSize_ = false;
  workGroupInfo_.wavesPerSimdHint_ = 0;
}

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
      if ((dev().settings().overrideLclSet & (1 << (workDim - 1))) == 0) {
        // Find threads per group
        size_t thrPerGrp = workGroupInfo()->size_;

        // Check if kernel uses images
        if (flags_.imageEna_ &&
          // and thread group is a multiple value of wavefronts
          ((thrPerGrp % workGroupInfo()->wavefrontSize_) == 0) &&
          // and it's 2 or 3-dimensional workload
          (workDim > 1) && (((gblWorkSize[0] % 16) == 0) && ((gblWorkSize[1] % 16) == 0))) {
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
  case ValueKind::Queue:
    return amd::KernelParameterDescriptor::QueueObject;
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
  case ValueKind::HiddenHostcallBuffer:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenHostcallBuffer;
  case ValueKind::HiddenDefaultQueue:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenDefaultQueue;
  case ValueKind::HiddenCompletionAction:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenCompletionAction;
  case ValueKind::HiddenMultiGridSyncArg:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenMultiGridSync;
  case ValueKind::HiddenNone:
  default:
    *isHidden = true;
    return amd::KernelParameterDescriptor::HiddenNone;
  }
}
#endif
// ================================================================================================
#if defined(WITH_COMPILER_LIB)
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
    else if (strcmp(&argInfo->argStr[2], "hostcall_buffer") == 0) {
      return amd::KernelParameterDescriptor::HiddenHostcallBuffer;
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
    case ValueKind::Queue:
      return T_QUEUE;
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
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
static inline size_t GetArgOffsetOCL(const KernelArgMD& lcArg) { return lcArg.mOffset; }

static inline size_t GetArgAlignmentOCL(const KernelArgMD& lcArg) { return lcArg.mAlign; }
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
#if defined(WITH_COMPILER_LIB)
static inline size_t GetArgPointeeAlignmentOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_POINTER) {
    return argInfo->arg.pointer.align;
  }
  return 1;
}
#endif

// ================================================================================================
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
static inline int GetArgSizeOCL(const KernelArgMD& lcArg) { return lcArg.mSize; }
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
#if defined(USE_COMGR_LIBRARY)
bool Kernel::GetAttrCodePropMetadata( const amd_comgr_metadata_node_t kernelMetaNode,
                                      KernelMD* kernelMD) {

  InitParameters(kernelMetaNode);

  // Set the workgroup information for the kernel
  workGroupInfo_.availableLDSSize_ = dev().info().localMemSizePerCU_;
  workGroupInfo_.availableSGPRs_ = 104;
  workGroupInfo_.availableVGPRs_ = 256;

  // extract the attribute metadata if there is any
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;

  switch (codeObjectVer()) {
    case 2: {
        amd_comgr_metadata_node_t symbolName;
        status = amd::Comgr::metadata_lookup(kernelMetaNode, "SymbolName", &symbolName);
        if (status == AMD_COMGR_STATUS_SUCCESS) {
          status = getMetaBuf(symbolName, &(kernelMD->mSymbolName));
          amd::Comgr::destroy_metadata(symbolName);
        }

        amd_comgr_metadata_node_t attrMeta;
        if (status == AMD_COMGR_STATUS_SUCCESS) {
          if (amd::Comgr::metadata_lookup(kernelMetaNode, "Attrs", &attrMeta) ==
              AMD_COMGR_STATUS_SUCCESS) {
            status = amd::Comgr::iterate_map_metadata(attrMeta, populateAttrs,
                                                      static_cast<void*>(kernelMD));
            amd::Comgr::destroy_metadata(attrMeta);
          }
        }

        // extract the code properties metadata
        amd_comgr_metadata_node_t codePropsMeta;
        if (status == AMD_COMGR_STATUS_SUCCESS) {
          status = amd::Comgr::metadata_lookup(kernelMetaNode, "CodeProps", &codePropsMeta);
        }

        if (status == AMD_COMGR_STATUS_SUCCESS) {
          status = amd::Comgr::iterate_map_metadata(codePropsMeta, populateCodeProps,
                                                    static_cast<void*>(kernelMD));
          amd::Comgr::destroy_metadata(codePropsMeta);
        }
      }
      break;
    case 3: {
        status = amd::Comgr::iterate_map_metadata(kernelMetaNode, populateKernelMetaV3,
                                                  static_cast<void*>(kernelMD));
      }
      break;
    default:
      return false;
  }


  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }

  // Setup the workgroup info based on the attributes and code properties
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
    workGroupInfo_.compileVecTypeHint_ = kernelMD->mAttrs.mVecTypeHint.c_str();
  }

  return true;
}

bool Kernel::SetAvailableSgprVgpr(const std::string& targetIdent) {
  std::string buf;

  amd_comgr_metadata_node_t isaMeta;
  amd_comgr_metadata_node_t sgprMeta;
  amd_comgr_metadata_node_t vgprMeta;
  bool hasIsaMeta = false;
  bool hasSgprMeta = false;
  bool hasVgprMeta = false;

  amd_comgr_status_t status = amd::Comgr::get_isa_metadata(targetIdent.c_str(), &isaMeta);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasIsaMeta = true;
    status = amd::Comgr::metadata_lookup(isaMeta, "AddressableNumSGPRs", &sgprMeta);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasSgprMeta = true;
    status = getMetaBuf(sgprMeta, &buf);
  }

  workGroupInfo_.availableSGPRs_ = (status == AMD_COMGR_STATUS_SUCCESS) ? atoi(buf.c_str()) : 0;

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::metadata_lookup(isaMeta, "AddressableNumVGPRs", &vgprMeta);
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hasVgprMeta = true;
    status = getMetaBuf(vgprMeta, &buf);
  }
  workGroupInfo_.availableVGPRs_ = (status == AMD_COMGR_STATUS_SUCCESS) ? atoi(buf.c_str()) : 0;

  if (hasVgprMeta) {
    amd::Comgr::destroy_metadata(vgprMeta);
  }

  if (hasSgprMeta) {
    amd::Comgr::destroy_metadata(sgprMeta);
  }

  if (hasIsaMeta) {
    amd::Comgr::destroy_metadata(isaMeta);
  }

  return (status == AMD_COMGR_STATUS_SUCCESS);
}

bool Kernel::GetPrintfStr(std::vector<std::string>* printfStr) {
  const amd_comgr_metadata_node_t programMD = prog().metadata();
  amd_comgr_metadata_node_t printfMeta;

  amd_comgr_status_t status = amd::Comgr::metadata_lookup(programMD,
                                codeObjectVer() == 2 ? "Printf" : "amdhsa.printf", &printfMeta);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return true;   // printf string metadata is not provided so just exit
  }

  // handle the printf string
  size_t printfSize = 0;
  status = amd::Comgr::get_metadata_list_size(printfMeta, &printfSize);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string buf;
    for (size_t i = 0; i < printfSize; ++i) {
      amd_comgr_metadata_node_t str;
      status = amd::Comgr::index_list_metadata(printfMeta, i, &str);

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(str, &buf);
        amd::Comgr::destroy_metadata(str);
      }

      if (status != AMD_COMGR_STATUS_SUCCESS) {
        amd::Comgr::destroy_metadata(printfMeta);
        return false;
      }

      printfStr->push_back(buf);
    }
  }

  amd::Comgr::destroy_metadata(printfMeta);
  return (status == AMD_COMGR_STATUS_SUCCESS);
}

void Kernel::InitParameters(const amd_comgr_metadata_node_t kernelMD) {
  // Iterate through the arguments and insert into parameterList
  device::Kernel::parameters_t params;
  device::Kernel::parameters_t hiddenParams;
  amd::KernelParameterDescriptor desc;
  size_t offset = 0;

  amd_comgr_metadata_node_t argsMeta;
  bool hsaArgsMeta = false;
  size_t argsSize = 0;

  amd_comgr_status_t status =  amd::Comgr::metadata_lookup(
                                          kernelMD,
                                          (codeObjectVer() == 2) ? "Args" : ".args",
                                          &argsMeta);
  // Assume no arguments if lookup fails.
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hsaArgsMeta = true;
    status = amd::Comgr::get_metadata_list_size(argsMeta, &argsSize);
  }

  for (size_t i = 0; i < argsSize; ++i) {
    KernelArgMD lcArg;

    amd_comgr_metadata_node_t argsNode;
    amd_comgr_metadata_kind_t kind;
    bool hsaArgsNode = false;

    status = amd::Comgr::index_list_metadata(argsMeta, i, &argsNode);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      hsaArgsNode = true;
      status = amd::Comgr::get_metadata_kind(argsNode, &kind);
    }
    if (kind != AMD_COMGR_METADATA_KIND_MAP) {
      status = AMD_COMGR_STATUS_ERROR;
    }
    if (status == AMD_COMGR_STATUS_SUCCESS) {
      void *data = static_cast<void*>(&lcArg);
      if (codeObjectVer() == 2) {
        status = amd::Comgr::iterate_map_metadata(argsNode, populateArgs, data);
      }
      else if (codeObjectVer() == 3) {
        status = amd::Comgr::iterate_map_metadata(argsNode, populateArgsV3, data);
      }
    }

    if (hsaArgsNode) {
      amd::Comgr::destroy_metadata(argsNode);
    }

    if (status != AMD_COMGR_STATUS_SUCCESS) {
      if (hsaArgsMeta) {
        amd::Comgr::destroy_metadata(argsMeta);
      }
      return;
    }

    size_t size = GetArgSizeOCL(lcArg);
    size_t alignment = (codeObjectVer() == 2) ? GetArgAlignmentOCL(lcArg) : 0;
    bool isHidden = false;
    desc.info_.oclObject_ = GetOclArgumentTypeOCL(lcArg, &isHidden);

    // Allocate the hidden arguments, but abstraction layer will skip them
    if (isHidden) {
      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::HiddenCompletionAction) {
        setDynamicParallelFlag(true);
      }
      offset = (codeObjectVer() == 2) ? amd::alignUp(offset, alignment) : GetArgOffsetOCL(lcArg);
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
      offset = (codeObjectVer() == 2) ? amd::alignUp(offset, alignment) : GetArgOffsetOCL(lcArg);
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

  if (hsaArgsMeta) {
    amd::Comgr::destroy_metadata(argsMeta);
  }

  // Save the number of OCL arguments
  uint32_t numParams = params.size();
  // Append the hidden arguments to the OCL arguments
  params.insert(params.end(), hiddenParams.begin(), hiddenParams.end());
  createSignature(params, numParams, amd::KernelSignature::ABIVersion_2);
}
#else // not define USE_COMGR_LIBRARY
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

      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::HiddenCompletionAction) {
        setDynamicParallelFlag(true);
      }

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
  createSignature(params, numParams, amd::KernelSignature::ABIVersion_2);
}
#endif  // defined(USE_COMGR_LIBRARY)
#endif  // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
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
#if defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)
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
      ClPrint(amd::LOG_WARNING, amd::LOG_KERN, "Invalid PrintInfo string: \"%s\"", str.c_str());
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
      ClPrint(amd::LOG_WARNING, amd::LOG_KERN, "Invalid PrintInfo string: \"%s\"", str.c_str());
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
#endif  // defined(WITH_LIGHTNING_COMPILER) || defined(USE_COMGR_LIBRARY)

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
void Kernel::InitPrintf(const aclPrintfFmt* aclPrintf) {
  uint index = 0;
  for (; aclPrintf->struct_size != 0; aclPrintf++) {
    index = aclPrintf->ID;
    if (printf_.size() <= index) {
      printf_.resize(index + 1);
    }
    PrintfInfo& info = printf_[index];
    const std::string& pfmt = aclPrintf->fmtStr;
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
  }
}
#endif // defined(WITH_COMPILER_LIB)

}
