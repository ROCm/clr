/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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

#include "os/os.hpp"
#include "utils/flags.hpp"
#include "include/aclTypes.h"
#include "device/pal/palprogram.hpp"
#include "device/pal/palblit.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include "utils/options.hpp"
#include "hsa.h"
#include "hsa_ext_image.h"
#include "amd_hsa_loader.hpp"
#if defined(USE_COMGR_LIBRARY)
#include "gelf.h"
#endif  // defined(USE_COMGR_LIBRARY)

namespace pal {

Segment::Segment() : gpuAccess_(nullptr), cpuAccess_(nullptr), cpuMem_(nullptr) {}

Segment::~Segment() {
  if (gpuAccess_ != nullptr) {
    gpuAccess_->owner()->release();
  }
  DestroyCpuAccess();
}

void Segment::DestroyCpuAccess() {
  if (cpuAccess_ != nullptr) {
    cpuAccess_->unmap(nullptr);
    delete cpuAccess_;
    cpuAccess_ = nullptr;
  }
  if (cpuMem_ != nullptr) {
    delete[] cpuMem_;
    cpuMem_ = nullptr;
  }
}

bool Segment::gpuAddressOffset(uint64_t offAddr, size_t* offset) {
  uint64_t baseAddr = gpuAccess_->vmAddress();
  if (baseAddr > offAddr) {
    return false;
  }
  *offset = (offAddr - baseAddr);
  return true;
}

bool Segment::alloc(HSAILProgram& prog, amdgpu_hsa_elf_segment_t segment, size_t size, size_t align,
                    bool zero) {
  align = amd::alignUp(align, sizeof(uint32_t));

  amd::Memory* amd_mem_obj = new (prog.dev().context())
      amd::Buffer(prog.dev().context(), 0, amd::alignUp(size, align),
                  // HIP requires SVM allocation for segment code due to possible global variable
                  // access and global variables are a part of code segment with the latest loader
                  amd::IS_HIP ? reinterpret_cast<void*>(1) : nullptr);

  if (amd_mem_obj == nullptr) {
    LogError("[OCL] failed to create a mem object!");
    return false;
  }

  if (!amd_mem_obj->create(nullptr)) {
    LogError("[OCL] failed to create a svm hidden buffer!");
    amd_mem_obj->release();
    return false;
  }

  gpuAccess_ = static_cast<pal::Memory*>(amd_mem_obj->getDeviceMemory(prog.dev(), false));

  if (segment == AMDGPU_HSA_SEGMENT_CODE_AGENT) {
    void* ptr = nullptr;
    cpuAccess_ = new pal::Memory(prog.dev(), amd::alignUp(size, align));
    if ((cpuAccess_ == nullptr) || !cpuAccess_->create(pal::Resource::Remote)) {
      delete cpuAccess_;
      cpuAccess_ = nullptr;
      ptr = cpuMem_ = reinterpret_cast<address>(new char[amd::alignUp(size, align)]);
      if (cpuMem_ == nullptr) {
        return false;
      }
    } else {
      ptr = cpuAccess_->map(nullptr, 0);
    }
    if (zero) {
      memset(ptr, 0, size);
    }
  }

  // Don't clear GPU memory if CPU backing store is available
  if ((cpuAccess_ == nullptr) && zero && !prog.isInternal()) {
    uint64_t pattern = 0;
    size_t patternSize = ((size % sizeof(pattern)) == 0) ? sizeof(pattern) : 1;
    prog.dev().xferMgr().fillBuffer(*gpuAccess_, &pattern, patternSize, amd::Coord3D(0),
                                    amd::Coord3D(size));
  }

  switch (segment) {
    case AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM:
    case AMDGPU_HSA_SEGMENT_GLOBAL_AGENT:
    case AMDGPU_HSA_SEGMENT_READONLY_AGENT:
      prog.addGlobalStore(gpuAccess_);
      prog.setGlobalVariableTotalSize(prog.globalVariableTotalSize() + size);
      break;
    case AMDGPU_HSA_SEGMENT_CODE_AGENT:
      prog.setCodeObjects(this, gpuAccess_, reinterpret_cast<address>(cpuAddress(0)));
      break;
    default:
      break;
  }
  return true;
}

void Segment::copy(size_t offset, const void* src, size_t size) {
  if (cpuAccess_ != nullptr) {
    amd::Os::fastMemcpy(cpuAddress(offset), src, size);
  } else {
    if (cpuMem_ != nullptr) {
      amd::Os::fastMemcpy(cpuAddress(offset), src, size);
    }
    amd::ScopedLock k(gpuAccess_->dev().xferMgr().lockXfer());
    VirtualGPU& gpu = *gpuAccess_->dev().xferQueue();
    Memory& xferBuf = gpu.xferWrite().Acquire(size);
    size_t tmpSize = std::min(static_cast<size_t>(xferBuf.size()), size);
    size_t srcOffs = 0;
    while (size != 0) {
      xferBuf.hostWrite(&gpu, reinterpret_cast<const_address>(src) + srcOffs, 0, tmpSize);
      xferBuf.partialMemCopyTo(gpu, 0, (offset + srcOffs), tmpSize, *gpuAccess_, false, true);
      size -= tmpSize;
      srcOffs += tmpSize;
      tmpSize = std::min(static_cast<size_t>(xferBuf.size()), size);
    }
    gpu.xferWrite().Release(xferBuf);
    gpu.waitAllEngines();
  }
}

bool Segment::freeze(bool destroySysmem) {
  VirtualGPU& gpu = *gpuAccess_->dev().xferQueue();
  bool result = true;
  if (cpuAccess_ != nullptr) {
    assert(gpuAccess_->size() == cpuAccess_->size() && "Backing store size mismatch!");
    amd::ScopedLock k(gpuAccess_->dev().xferMgr().lockXfer());
    result = cpuAccess_->partialMemCopyTo(gpu, 0, 0, gpuAccess_->size(), *gpuAccess_, false, true);
    gpu.waitAllEngines();
  }
  assert(!destroySysmem || (cpuAccess_ == nullptr));
  return result;
}

const static char* Carrizo = "Carrizo";
HSAILProgram::HSAILProgram(Device& device, amd::Program& owner)
    : Program(device, owner),
      rawBinary_(nullptr),
      kernels_(nullptr),
      codeSegGpu_(nullptr),
      codeSegment_(nullptr),
      maxScratchRegs_(0),
      executable_(nullptr),
      loaderContext_(this) {
  xnackEnabled_ = dev().hwInfo()->xnackEnabled_;
  sramEccEnabled_ = dev().info().sramEccEnabled_;
  if (dev().asicRevision() == Pal::AsicRevision::Bristol) {
    machineTarget_ = Carrizo;
  } else {
    machineTarget_ = dev().hwInfo()->machineTarget_;
  }
  loader_ = amd::hsa::loader::Loader::Create(&loaderContext_);
}

HSAILProgram::HSAILProgram(NullDevice& device, amd::Program& owner)
    : Program(device, owner),
      rawBinary_(nullptr),
      kernels_(nullptr),
      codeSegGpu_(nullptr),
      codeSegment_(nullptr),
      maxScratchRegs_(0),
      executable_(nullptr),
      loaderContext_(this) {
  isNull_ = true;
  xnackEnabled_ = dev().hwInfo()->xnackEnabled_;
  sramEccEnabled_ = dev().info().sramEccEnabled_;
  if (dev().asicRevision() == Pal::AsicRevision::Bristol) {
    machineTarget_ = Carrizo;
  } else {
    machineTarget_ = dev().hwInfo()->machineTarget_;
  }
  loader_ = amd::hsa::loader::Loader::Create(&loaderContext_);
}

HSAILProgram::~HSAILProgram() {
  // Destroy internal static samplers
  for (auto& it : staticSamplers_) {
    delete it;
  }
#if defined(WITH_COMPILER_LIB)
  if (rawBinary_ != nullptr) {
    aclFreeMem(binaryElf_, rawBinary_);
  }
  acl_error error;
  // Free the elf binary
  if (binaryElf_ != nullptr) {
    error = aclBinaryFini(binaryElf_);
    if (error != ACL_SUCCESS) {
      LogWarning("Error while destroying the acl binary \n");
    }
  }
#endif  // defined(WITH_COMPILER_LIB)
  releaseClBinary();
  if (executable_ != nullptr) {
    loader_->DestroyExecutable(executable_);
  }
  delete kernels_;
  amd::hsa::loader::Loader::Destroy(loader_);
}


inline static std::vector<std::string> splitSpaceSeparatedString(char* str) {
  std::string s(str);
  std::stringstream ss(s);
  std::istream_iterator<std::string> beg(ss), end;
  std::vector<std::string> vec(beg, end);
  return vec;
}

bool HSAILProgram::setKernels(amd::option::Options* options, void* binary, size_t binSize) {
#if defined(WITH_COMPILER_LIB)
  // ACL_TYPE_CG stage is not performed for offline compilation
  hsa_agent_t agent;
  agent.handle = 1;
  executable_ = loader_->CreateExecutable(HSA_PROFILE_FULL, nullptr);
  if (executable_ == nullptr) {
    buildLog_ += "Error: Executable for AMD HSA Code Object isn't created.\n";
    return false;
  }
  size_t size = binSize;
  hsa_code_object_t code_object;
  code_object.handle = reinterpret_cast<uint64_t>(binary);

  if (amd::IS_HIP) {
    defineUndefinedVars();
  }

  hsa_status_t status = executable_->LoadCodeObject(agent, code_object, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object loading failed.\n";
    return false;
  }
  status = executable_->Freeze(nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object freeze failed.\n";
    return false;
  }

  size_t kernelNamesSize = 0;
  acl_error errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_KERNEL_NAMES, nullptr,
                                     nullptr, &kernelNamesSize);
  if (errorCode != ACL_SUCCESS) {
    buildLog_ += "Error: Querying of kernel names size from the binary failed.\n";
    return false;
  }
  if (kernelNamesSize > 0) {
    char* kernelNames = new char[kernelNamesSize];
    errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_KERNEL_NAMES, nullptr, kernelNames,
                             &kernelNamesSize);
    if (errorCode != ACL_SUCCESS) {
      buildLog_ += "Error: Querying of kernel names from the binary failed.\n";
      delete[] kernelNames;
      return false;
    }
    std::vector<std::string> vKernels = splitSpaceSeparatedString(kernelNames);
    delete[] kernelNames;
    bool dynamicParallelism = false;
    for (const auto& it : vKernels) {
      std::string kernelName(it);
      std::string openclKernelName = device::Kernel::openclMangledName(kernelName);

      HSAILKernel* aKernel =
          new HSAILKernel(kernelName, this, options->origOptionStr + ProcessOptionsFlattened(options));
      kernels()[kernelName] = aKernel;

      amd::hsa::loader::Symbol* sym = executable_->GetSymbol(openclKernelName.c_str(), &agent);
      if (!sym) {
        buildLog_ += "Error: Getting kernel ISA code symbol '" + openclKernelName +
            "' from AMD HSA Code Object failed. Kernel initialization failed.\n";
        return false;
      }
      if (!aKernel->init(sym, false)) {
        buildLog_ += "Error: Kernel '" + openclKernelName + "' initialization failed.\n";
        return false;
      }
      buildLog_ += aKernel->buildLog();
      aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
      dynamicParallelism |= aKernel->dynamicParallelism();
      // Find max scratch regs used in the program. It's used for scratch buffer preallocation
      // with dynamic parallelism, since runtime doesn't know which child kernel will be called
      maxScratchRegs_ =
          std::max(static_cast<uint>(aKernel->workGroupInfo()->scratchRegs_), maxScratchRegs_);
    }
    // Allocate kernel table for device enqueuing
    if (!isNull() && dynamicParallelism && !allocKernelTable()) {
      return false;
    }
  }

  DestroySegmentCpuAccess();
#endif  // defined(WITH_COMPILER_LIB)
  return true;
}

bool HSAILProgram::createBinary(amd::option::Options* options) { return true; }

bool HSAILProgram::allocKernelTable() {
  uint size = kernels().size() * sizeof(size_t);

  kernels_ = new pal::Memory(dev(), size);
  // Initialize kernel table
  if ((kernels_ == nullptr) || !kernels_->create(Resource::RemoteUSWC)) {
    delete kernels_;
    return false;
  } else {
    size_t* table = reinterpret_cast<size_t*>(kernels_->map(nullptr, pal::Resource::WriteOnly));
    for (auto& it : kernels()) {
      HSAILKernel* kernel = static_cast<HSAILKernel*>(it.second);
      table[kernel->index()] = static_cast<size_t>(kernel->gpuAqlCode());
    }
    kernels_->unmap(nullptr);
  }
  return true;
}

void HSAILProgram::fillResListWithKernels(VirtualGPU& gpu) const { gpu.addVmMemory(&codeSegGpu()); }

const aclTargetInfo& HSAILProgram::info(const char* str) {
#if defined(WITH_COMPILER_LIB)
  acl_error err;
  std::string arch = "hsail";
  if (dev().settings().use64BitPtr_) {
    arch = "hsail64";
  }
  info_ = aclGetTargetInfo(arch.c_str(),
                           (str && str[0] == '\0' ? dev().hwInfo()->machineTarget_ : str), &err);
  if (err != ACL_SUCCESS) {
    LogWarning("aclGetTargetInfo failed");
  }
#endif  // defined(WITH_COMPILER_LIB)
  return info_;
}

bool HSAILProgram::saveBinaryAndSetType(type_t type) {
#if defined(WITH_COMPILER_LIB)
  // Write binary to memory
  if (rawBinary_ != nullptr) {
    // Free memory containing rawBinary
    aclFreeMem(binaryElf_, rawBinary_);
    rawBinary_ = nullptr;
  }
  size_t size = 0;
  if (aclWriteToMem(binaryElf_, &rawBinary_, &size) != ACL_SUCCESS) {
    buildLog_ += "Failed to write binary to memory \n";
    return false;
  }
  setBinary(static_cast<char*>(rawBinary_), size);
  // Set the type of binary
  setType(type);
#endif  // defined(WITH_COMPILER_LIB)
  return true;
}

bool HSAILProgram::defineGlobalVar(const char* name, void* dptr) {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  hsa_agent_t agent;

  agent.handle = 1;
  hsa_status = executable_->DefineAgentExternalVariable(name, agent, HSA_VARIABLE_SEGMENT_GLOBAL, dptr);
  if(HSA_STATUS_SUCCESS != hsa_status) {
    buildLog_ += "Could not define Program External Variable";
    buildLog_ += "\n";
  }

  return (hsa_status == HSA_STATUS_SUCCESS);
}

bool HSAILProgram::createGlobalVarObj(amd::Memory** amd_mem_obj, void** device_pptr, size_t* bytes,
                                      const char* global_name) const {
  uint32_t length = 0;
  size_t offset = 0;
  uint32_t flags = 0;
  amd::Memory* parent = nullptr;
  hsa_agent_t agent;
  hsa_symbol_kind_t type;
  hsa_status_t status = HSA_STATUS_SUCCESS;
  amd::hsa::loader::Symbol* symbol = nullptr;

  if (amd_mem_obj == nullptr) {
    buildLog_ += "amd_mem_obj is null";
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the Symbol obj from global name*/
  agent.handle = 1;
  symbol = executable_->GetSymbol(global_name, &agent);
  if (!symbol) {
    buildLog_ += "Error: Getting Global Var Symbol";
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the symbol type */
  if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &type)) {
    buildLog_ += "Error: Getting Global Var Symbol Type";
    buildLog_ += "\n";
    return false;
  }

  /* Make sure the symbol is of type VARIABLE */
  if (type != HSA_SYMBOL_KIND_VARIABLE) {
    buildLog_ += "Error: Retrieve Symbol type is not Variable ";
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the symbol Name Length */
  if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &length)) {
    buildLog_ += "Error: Getting Global Var Symbol length";
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the symbol name */
  char* name = reinterpret_cast<char*>(alloca(length + 1));
  if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_NAME, name)) {
    buildLog_ += "Error: Getting Global Var Symbol name";
    buildLog_ += "\n";
    return false;
  }
  name[length] = '\0';

  /* Make sure the name matches with the global name */
  if (std::string(name) != std::string(global_name)) {
    buildLog_ += "Error: Global Var Name mismatch";
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the Symbol address */
  if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, device_pptr)) {
    buildLog_ += "Error: Getting Global Var Symbol Address";
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the Symbol size */
  if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, bytes)) {
    buildLog_ += "Error: Getting Global Var Symbol size";
    buildLog_ += "\n";
    return false;
  }

  /* Retrieve the Offset from global pal::Memory created @ segment::alloc */
  if (!codeSegment_->gpuAddressOffset(reinterpret_cast<uint64_t>(*device_pptr), &offset)) {
    buildLog_ += "Error: Cannot Retrieve the Address Offset";
    buildLog_ += "\n";
    return false;
  }

  /* Create a View from the global pal::Memory */
  parent = codeSegGpu_->owner();
  *amd_mem_obj = new (parent->getContext()) amd::Buffer(*parent, flags, offset, *bytes);

  if (*amd_mem_obj == nullptr) {
    buildLog_ += "[OCL] Failed to create a mem object!";
    buildLog_ += "\n";
    return false;
  }

  if (!((*amd_mem_obj)->create(nullptr))) {
    buildLog_ += "[OCL] failed to create a svm hidden buffer!";
    buildLog_ += "\n";
    (*amd_mem_obj)->release();
    return false;
  }

  return true;
}

hsa_isa_t PALHSALoaderContext::IsaFromName(const char* name) {
  hsa_isa_t isa = {0};
  uint32_t gfxip = 0;
  std::string gfx_target(name);
  if (gfx_target.find("amdgcn-") == 0) {
    std::string gfxip_version_str = gfx_target.substr(gfx_target.find("gfx") + 3);
    gfxip = std::atoi(gfxip_version_str.c_str());
  } else {
    // FIXME: Old way. To be remove.
    uint32_t shift = 1;
    size_t last = gfx_target.length();
    std::string ver;
    do {
      size_t first = gfx_target.find_last_of(':', last);
      ver = gfx_target.substr(first + 1, last - first);
      last = first - 1;
      gfxip += static_cast<uint32_t>(atoi(ver.c_str())) * shift;
      shift *= 10;
    } while (shift <= 100);
  }
  isa.handle = gfxip;
  return isa;
}

bool PALHSALoaderContext::IsaSupportedByAgent(hsa_agent_t agent, hsa_isa_t isa) {
  uint32_t gfxipVersion = program_->dev().settings().useLightning_
      ? program_->dev().hwInfo()->gfxipVersionLC_
      : program_->dev().hwInfo()->gfxipVersion_;
  uint32_t majorSrc = gfxipVersion / 10;
  uint32_t minorSrc = gfxipVersion % 10;

  uint32_t majorTrg = isa.handle / 10;
  uint32_t minorTrg = isa.handle % 10;

  if (majorSrc != majorTrg) {
    return false;
  } else if (minorTrg == minorSrc) {
    return true;
  } else if (minorTrg < minorSrc) {
    LogWarning("ISA downgrade for execution!");
    return true;
  }

  return false;
}

void* PALHSALoaderContext::SegmentAlloc(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                                        size_t size, size_t align, bool zero) {
  assert(size);
  assert(align);
  if (program_->isNull()) {
    // Note: In Linux ::posix_memalign() requires at least 16 bytes for the alignment.
    align = amd::alignUp(align, 16);
    void* ptr = amd::Os::alignedMalloc(size, align);
    if ((ptr != nullptr) && zero) {
      memset(ptr, 0, size);
    }
    return ptr;
  }
  Segment* seg = new Segment();
  if (seg != nullptr && !seg->alloc(*program_, segment, size, align, zero)) {
    return nullptr;
  }
  return seg;
}

bool PALHSALoaderContext::SegmentCopy(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                                      void* dst, size_t offset, const void* src, size_t size) {
  if (program_->isNull()) {
    amd::Os::fastMemcpy(reinterpret_cast<address>(dst) + offset, src, size);
    return true;
  }
  Segment* s = reinterpret_cast<Segment*>(dst);
  s->copy(offset, src, size);
  return true;
}

void PALHSALoaderContext::SegmentFree(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                                      void* seg, size_t size) {
  if (program_->isNull()) {
    amd::Os::alignedFree(seg);
  } else {
    Segment* s = reinterpret_cast<Segment*>(seg);
    delete s;
  }
}

void* PALHSALoaderContext::SegmentAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                                          void* seg, size_t offset) {
  assert(seg);
  if (program_->isNull()) {
    return (reinterpret_cast<address>(seg) + offset);
  }
  Segment* s = reinterpret_cast<Segment*>(seg);
  return reinterpret_cast<void*>(s->gpuAddress(offset));
}

void* PALHSALoaderContext::SegmentHostAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                                              void* seg, size_t offset) {
  assert(seg);
  if (program_->isNull()) {
    return (reinterpret_cast<address>(seg) + offset);
  }
  Segment* s = reinterpret_cast<Segment*>(seg);
  return s->cpuAddress(offset);
}

bool PALHSALoaderContext::SegmentFreeze(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent,
                                        void* seg, size_t size) {
  if (program_->isNull()) {
    return true;
  }
  Segment* s = reinterpret_cast<Segment*>(seg);
  return s->freeze((segment == AMDGPU_HSA_SEGMENT_CODE_AGENT) ? false : true);
}

hsa_status_t PALHSALoaderContext::SamplerCreate(
    hsa_agent_t agent, const hsa_ext_sampler_descriptor_t* sampler_descriptor,
    hsa_ext_sampler_t* sampler_handle) {
  if (!agent.handle) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (!sampler_descriptor || !sampler_handle) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (program_->isNull()) {
    // Offline compilation. Provide a fake handle to avoid an assert
    sampler_handle->handle = 1;
    return HSA_STATUS_SUCCESS;
  }
  uint32_t state = 0;
  switch (sampler_descriptor->coordinate_mode) {
    case HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED:
      state = amd::Sampler::StateNormalizedCoordsFalse;
      break;
    case HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED:
      state = amd::Sampler::StateNormalizedCoordsTrue;
      break;
    default:
      assert(false);
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  switch (sampler_descriptor->filter_mode) {
    case HSA_EXT_SAMPLER_FILTER_MODE_NEAREST:
      state |= amd::Sampler::StateFilterNearest;
      break;
    case HSA_EXT_SAMPLER_FILTER_MODE_LINEAR:
      state |= amd::Sampler::StateFilterLinear;
      break;
    default:
      assert(false);
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  switch (sampler_descriptor->address_mode) {
    case HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
      state |= amd::Sampler::StateAddressClampToEdge;
      break;
    case HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER:
      state |= amd::Sampler::StateAddressClamp;
      break;
    case HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT:
      state |= amd::Sampler::StateAddressRepeat;
      break;
    case HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:
      state |= amd::Sampler::StateAddressMirroredRepeat;
      break;
    case HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED:
      state |= amd::Sampler::StateAddressNone;
      break;
    default:
      assert(false);
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  pal::Sampler* sampler = new pal::Sampler(program_->dev());
  if (!sampler || !sampler->create(state)) {
    delete sampler;
    return HSA_STATUS_ERROR;
  }
  program_->addSampler(sampler);
  sampler_handle->handle = sampler->hwSrd();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t PALHSALoaderContext::SamplerDestroy(hsa_agent_t agent,
                                                 hsa_ext_sampler_t sampler_handle) {
  if (!agent.handle) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }
  if (!sampler_handle.handle) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

#if defined(USE_COMGR_LIBRARY)

static hsa_status_t GetKernelNamesCallback(hsa_executable_t hExec, hsa_executable_symbol_t hSymbol,
                                           void* data) {
  auto symbol = Symbol::Object(hSymbol);
  auto symbolNameList = reinterpret_cast<std::vector<std::string>*>(data);

  hsa_symbol_kind_t type;
  if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &type)) {
    return HSA_STATUS_ERROR;
  }

  if (type == HSA_SYMBOL_KIND_KERNEL) {
    uint32_t length;
    if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &length)) {
      return HSA_STATUS_ERROR;
    }

    char* name = reinterpret_cast<char*>(alloca(length + 1));
    if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_NAME, name)) {
      return HSA_STATUS_ERROR;
    }
    name[length] = '\0';

    symbolNameList->push_back(std::string(name));
  }
  return HSA_STATUS_SUCCESS;
}

#endif  // defined(USE_COMGR_LIBRARY)

bool LightningProgram::createBinary(amd::option::Options* options) {
#if defined(USE_COMGR_LIBRARY)
  if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt, type())) {
    LogError("Failed to create ELF binary image!");
    return false;
  }
#endif  // defined(USE_COMGR_LIBRARY)
  return true;
}

bool LightningProgram::setKernels(amd::option::Options* options, void* binary, size_t binSize) {
#if defined(USE_COMGR_LIBRARY)
  hsa_agent_t agent;
  agent.handle = 1;

  executable_ = loader_->CreateExecutable(HSA_PROFILE_FULL, nullptr);
  if (executable_ == nullptr) {
    buildLog_ += "Error: Executable for AMD HSA Code Object isn't created.\n";
    return false;
  }

  hsa_code_object_t code_object;
  code_object.handle = reinterpret_cast<uint64_t>(binary);

  if (amd::IS_HIP) {
    defineUndefinedVars();
  }

  hsa_status_t status = executable_->LoadCodeObject(agent, code_object, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: AMD HSA Code Object loading failed.\n";
    return false;
  }

  status = executable_->Freeze(nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog_ += "Error: Freezing the executable failed: ";
    return false;
  }

  // Find the size of global variables from the binary
  if (!FindGlobalVarSize(binary, binSize)) {
    return false;
  }

  for (const auto& kernelMeta : kernelMetadataMap_) {
    auto kernelName = kernelMeta.first;
    auto kernel =
        new LightningKernel(kernelName, this, options->origOptionStr + ProcessOptionsFlattened(options));
    kernels()[kernelName] = kernel;

    if (!kernel->init()) {
      return false;
    }

    kernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);

    // Find max scratch regs used in the program. It's used for scratch buffer preallocation
    // with dynamic parallelism, since runtime doesn't know which child kernel will be called
    maxScratchRegs_ =
        std::max(static_cast<uint>(kernel->workGroupInfo()->scratchRegs_), maxScratchRegs_);
  }
  DestroySegmentCpuAccess();
#endif  // defined(USE_COMGR_LIBRARY)
  return true;
}

}  // namespace pal
