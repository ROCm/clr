//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/device.hpp"
#include "thread/atomic.hpp"
#include "thread/monitor.hpp"
#include "utils/options.hpp"
#include "comgrctx.hpp"

#if defined(WITH_HSA_DEVICE)
#include "device/rocm/rocdevice.hpp"
extern amd::AppProfile* rocCreateAppProfile();
#endif

#if defined(WITH_PAL_DEVICE)
// namespace pal {
extern bool PalDeviceLoad();
extern void PalDeviceUnload();
//}
#endif  // WITH_PAL_DEVICE

#if defined(WITH_GPU_DEVICE)
extern bool DeviceLoad();
extern void DeviceUnload();
#endif  // WITH_GPU_DEVICE

#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "thread/monitor.hpp"
#include "amdocl/cl_common.hpp"
#include "utils/options.hpp"
#include "utils/versions.hpp"  // AMD_PLATFORM_INFO

#if defined(HAVE_BLOWFISH_H)
#include "blowfish/oclcrypt.hpp"
#endif

#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"
#include "spirv/spirvUtils.h"

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <set>
#include <algorithm>
#include <numeric>


namespace device {
extern const char* BlitSourceCode;
}

namespace amd {

std::vector<Device*>* Device::devices_ = nullptr;
AppProfile Device::appProfile_;

Context* Device::glb_ctx_ = nullptr;
Monitor Device::p2p_stage_ops_("P2P Staging Lock", true);
Memory* Device::p2p_stage_ = nullptr;

amd::Monitor MemObjMap::AllocatedLock_("Guards SVM allocation list");
std::map<uintptr_t, amd::Memory*> MemObjMap::MemObjMap_;

size_t MemObjMap::size() {
  amd::ScopedLock lock(AllocatedLock_);
  return MemObjMap_.size();
}

void MemObjMap::AddMemObj(const void* k, amd::Memory* v) {
  amd::ScopedLock lock(AllocatedLock_);
  MemObjMap_.insert({reinterpret_cast<uintptr_t>(k), v});
}

void MemObjMap::RemoveMemObj(const void* k) {
  amd::ScopedLock lock(AllocatedLock_);
  MemObjMap_.erase(reinterpret_cast<uintptr_t>(k));
}

amd::Memory* MemObjMap::FindMemObj(const void* k) {
  amd::ScopedLock lock(AllocatedLock_);
  uintptr_t key = reinterpret_cast<uintptr_t>(k);
  auto it = MemObjMap_.upper_bound(key);
  if (it == MemObjMap_.begin()) {
    return nullptr;
  }

  --it;
  amd::Memory* mem = it->second;
  if (key >= it->first && key < (it->first + mem->getSize())) {
    // the k is in the range
    return mem;
  } else {
    return nullptr;
  }
}


Device::BlitProgram::~BlitProgram() {
  if (program_ != nullptr) {
    program_->release();
  }
}

bool Device::BlitProgram::create(amd::Device* device, const char* extraKernels,
                                 const char* extraOptions) {
  std::vector<amd::Device*> devices;
  devices.push_back(device);
  std::string kernels(device::BlitSourceCode);

  if (extraKernels != nullptr) {
    kernels += extraKernels;
  }

  // Create a program with all blit kernels
  program_ = new Program(*context_, kernels.c_str(), Program::OpenCL_C);
  if (program_ == nullptr) {
    return false;
  }

  // Build all kernels
  std::string opt = "-cl-internal-kernel ";
  if (!device->settings().useLightning_) {
    opt += "-Wf,--force_disable_spir -fno-lib-no-inline -fno-sc-keep-calls ";
  }

  if (extraOptions != nullptr) {
    opt += extraOptions;
  }
  if (!GPU_DUMP_BLIT_KERNELS) {
    opt += " -fno-enable-dump";
  }
  if (CL_SUCCESS !=
      program_->build(devices, opt.c_str(), nullptr, nullptr, GPU_DUMP_BLIT_KERNELS)) {
    return false;
  }

  return true;
}

bool Device::init() {
  assert(!Runtime::initialized() && "initialize only once");
  bool ret = false;
  devices_ = nullptr;
  appProfile_.init();


// IMPORTANT: Note that we are initialiing HSA stack first and then
// GPU stack. The order of initialization is signiicant and if changed
// amd::Device::registerDevice() must be accordingly modified.
#if defined(WITH_HSA_DEVICE)
  if ((GPU_ENABLE_PAL != 1) || flagIsDefault(GPU_ENABLE_PAL)) {
    // Return value of roc::Device::init()
    // If returned false, error initializing HSA stack.
    // If returned true, either HSA not installed or HSA stack
    //                   successfully initialized.
    if (!roc::Device::init()) {
      // abort() commentted because this is the only indication
      // that KFD is not installed.
      // Ignore the failure and assume KFD is not installed.
      // abort();
    }
    ret |= roc::NullDevice::init();
  }
#endif  // WITH_HSA_DEVICE
#if defined(WITH_GPU_DEVICE)
  if (GPU_ENABLE_PAL != 1) {
    ret |= DeviceLoad();
  }
#endif  // WITH_GPU_DEVICE
#if defined(WITH_PAL_DEVICE)
  if (GPU_ENABLE_PAL != 0) {
    ret |= PalDeviceLoad();
  }
#endif  // WITH_PAL_DEVICE
  return ret;
}

void Device::tearDown() {
  if (devices_ != nullptr) {
    for (uint i = 0; i < devices_->size(); ++i) {
      delete devices_->at(i);
    }
    devices_->clear();
    delete devices_;
  }
#if defined(WITH_HSA_DEVICE)
  roc::Device::tearDown();
#endif  // WITH_HSA_DEVICE
#if defined(WITH_GPU_DEVICE)
  if (GPU_ENABLE_PAL != 1) {
    DeviceUnload();
  }
#endif  // WITH_GPU_DEVICE
#if defined(WITH_PAL_DEVICE)
  if (GPU_ENABLE_PAL != 0) {
    PalDeviceUnload();
  }
#endif  // WITH_PAL_DEVICE
}

Device::Device()
    : settings_(nullptr),
      online_(true),
      blitProgram_(nullptr),
      hwDebugMgr_(nullptr),
      vaCacheAccess_(nullptr),
      vaCacheMap_(nullptr),
      index_(0) {
  memset(&info_, '\0', sizeof(info_));
}

Device::~Device() {
  CondLog((vaCacheMap_ != nullptr) && (vaCacheMap_->size() != 0),
          "Application didn't unmap all host memory!");
  delete vaCacheMap_;
  delete vaCacheAccess_;

  // Destroy device settings
  if (settings_ != nullptr) {
    delete settings_;
  }

  if (info_.extensions_ != nullptr) {
    delete[] info_.extensions_;
  }
}

bool Device::ValidateComgr() {
#if defined(USE_COMGR_LIBRARY)
  // Check if Lightning compiler was requested
  if (settings_->useLightning_) {
    std::call_once(amd::Comgr::initialized, amd::Comgr::LoadLib);
    // Use Lightning only if it's available
    settings_->useLightning_ = amd::Comgr::IsReady();
    return settings_->useLightning_;
  }
#endif
  return true;
}

bool Device::create() {
  vaCacheAccess_ = new amd::Monitor("VA Cache Ops Lock", true);
  if (nullptr == vaCacheAccess_) {
    return false;
  }
  vaCacheMap_ = new std::map<uintptr_t, device::Memory*>();
  if (nullptr == vaCacheMap_) {
    return false;
  }
  return true;
}

void Device::registerDevice() {
  assert(Runtime::singleThreaded() && "this is not thread-safe");

  static bool defaultIsAssigned = false;

  if (devices_ == nullptr) {
    devices_ = new std::vector<Device*>;
  }

  if (info_.available_) {
    if (!defaultIsAssigned && online_) {
      defaultIsAssigned = true;
      info_.type_ |= CL_DEVICE_TYPE_DEFAULT;
    }
  }
  if (isOnline()) {
    for (const auto& dev : devices()) {
      if (dev->isOnline()) {
        index_++;
      }
    }
  }
  devices_->push_back(this);
}

void Device::addVACache(device::Memory* memory) const {
  // Make sure system memory has direct access
  if (memory->isHostMemDirectAccess()) {
    // VA cache access must be serialised
    amd::ScopedLock lk(*vaCacheAccess_);
    void* start = memory->owner()->getHostMem();
    size_t offset;
    device::Memory* doubleMap = findMemoryFromVA(start, &offset);

    if (doubleMap == nullptr) {
      // Insert the new entry
      vaCacheMap_->insert(
          std::pair<uintptr_t, device::Memory*>(reinterpret_cast<uintptr_t>(start), memory));
    } else {
      LogError("Unexpected double map() call from the app!");
    }
  }
}

void Device::removeVACache(const device::Memory* memory) const {
  // Make sure system memory has direct access
  if (memory->isHostMemDirectAccess() && memory->owner()) {
    // VA cache access must be serialised
    amd::ScopedLock lk(*vaCacheAccess_);
    void* start = memory->owner()->getHostMem();
    vaCacheMap_->erase(reinterpret_cast<uintptr_t>(start));
  }
}

device::Memory* Device::findMemoryFromVA(const void* ptr, size_t* offset) const {
  // VA cache access must be serialised
  amd::ScopedLock lk(*vaCacheAccess_);

  uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
  auto it = vaCacheMap_->upper_bound(reinterpret_cast<uintptr_t>(ptr));
  if (it == vaCacheMap_->begin()) {
    return nullptr;
  }

  --it;
  device::Memory* mem = it->second;
  if (key >= it->first && key < (it->first + mem->size())) {
    // ptr is in the range
    *offset = key - it->first;
    return mem;
  }
  return nullptr;
}

bool Device::IsTypeMatching(cl_device_type type, bool offlineDevices) {
  if (!(isOnline() || offlineDevices)) {
    return false;
  }

  return (info_.type_ & type) != 0;
}

std::vector<Device*> Device::getDevices(cl_device_type type, bool offlineDevices) {
  std::vector<Device*> result;

  if (devices_ == nullptr) {
    return result;
  }

  // Create the list of available devices
  for (const auto& it : *devices_) {
    // Check if the device type is matched
    if (it->IsTypeMatching(type, offlineDevices)) {
      result.push_back(it);
    }
  }

  return result;
}

size_t Device::numDevices(cl_device_type type, bool offlineDevices) {
  size_t result = 0;

  if (devices_ == nullptr) {
    return 0;
  }

  for (const auto& it : *devices_) {
    // Check if the device type is matched
    if (it->IsTypeMatching(type, offlineDevices)) {
      ++result;
    }
  }

  return result;
}

bool Device::getDeviceIDs(cl_device_type deviceType, cl_uint numEntries, cl_device_id* devices,
                          cl_uint* numDevices, bool offlineDevices) {
  if (numDevices != nullptr && devices == nullptr) {
    *numDevices = (cl_uint)amd::Device::numDevices(deviceType, offlineDevices);
    return (*numDevices > 0) ? true : false;
  }
  assert(devices != nullptr && "check the code above");

  std::vector<amd::Device*> ret = amd::Device::getDevices(deviceType, offlineDevices);
  if (ret.size() == 0) {
    *not_null(numDevices) = 0;
    return false;
  }

  auto it = ret.cbegin();
  cl_uint count = std::min(numEntries, (cl_uint)ret.size());

  while (count--) {
    *devices++ = as_cl(*it++);
    --numEntries;
  }
  while (numEntries--) {
    *devices++ = (cl_device_id)0;
  }

  *not_null(numDevices) = (cl_uint)ret.size();
  return true;
}

char* Device::getExtensionString() {
  std::stringstream extStream;
  size_t size;
  char* result = nullptr;

  // Generate the extension string
  for (uint i = 0; i < ClExtTotal; ++i) {
    if (settings().checkExtension(i)) {
      extStream << OclExtensionsString[i];
    }
  }

  size = extStream.str().size() + 1;

  // Create a single string with all extensions
  result = new char[size];
  if (result != nullptr) {
    memcpy(result, extStream.str().data(), (size - 1));
    result[size - 1] = 0;
  }

  return result;
}

#if defined(WITH_LIGHTNING_COMPILER) && !defined(USE_COMGR_LIBRARY)
CacheCompilation::CacheCompilation(std::string targetStr, std::string postfix, bool enableCache,
                                   bool resetCache)
    : codeCache_(targetStr, 0, AMD_PLATFORM_BUILD_NUMBER, postfix),
      isCodeCacheEnabled_(enableCache) {
  if (resetCache) {
    // clean up the cached data of the target device
    StringCache emptyCache(targetStr, 0, 0, postfix);
  }
}

bool CacheCompilation::linkLLVMBitcode(amd::opencl_driver::Compiler* C,
                                       std::vector<amd::opencl_driver::Data*>& inputs,
                                       amd::opencl_driver::Buffer* output,
                                       std::vector<std::string>& options, std::string& buildLog) {
  std::string cacheOpt;
  cacheOpt = std::accumulate(begin(options), end(options), cacheOpt);

  bool ret = false;
  bool cachedCodeExist = false;
  std::vector<StringCache::CachedData> bcSet;
  if (isCodeCacheEnabled_) {
    using namespace amd::opencl_driver;

    for (auto& input : inputs) {
      assert(input->Type() == DT_LLVM_BC);

      BufferReference* bc = reinterpret_cast<BufferReference*>(input);
      StringCache::CachedData cachedData = {bc->Ptr(), bc->Size()};
      bcSet.push_back(cachedData);
    }

    std::string dstData = "";
    if (codeCache_.getCacheEntry(isCodeCacheEnabled_, bcSet.data(), bcSet.size(), cacheOpt, dstData,
                                 "Link LLVM Bitcodes")) {
      std::copy(dstData.begin(), dstData.end(), std::back_inserter(output->Buf()));
      cachedCodeExist = true;
    }
  }

  if (!cachedCodeExist) {
    if (!C->LinkLLVMBitcode(inputs, output, options)) {
      return false;
    }

    if (isCodeCacheEnabled_) {
      std::string dstData(output->Buf().data(), output->Buf().size());
      if (!codeCache_.makeCacheEntry(bcSet.data(), bcSet.size(), cacheOpt, dstData)) {
        buildLog += "Warning: Failed to caching codes.\n";
        LogWarning("Caching codes failed!");
      }
    }
  }

  return true;
}

bool CacheCompilation::compileToLLVMBitcode(amd::opencl_driver::Compiler* C,
                                            std::vector<amd::opencl_driver::Data*>& inputs,
                                            amd::opencl_driver::Buffer* output,
                                            std::vector<std::string>& options,
                                            std::string& buildLog) {
  std::string cacheOpt;
  for (uint i = 0; i < options.size(); i++) {
    // skip the header file option, which is associated with the -cl-std=<CLstd> option
    if (options[i].compare("-include-pch") == 0) {
      i++;
      continue;
    }
    cacheOpt += options[i];
  }

  bool ret = false;
  bool cachedCodeExist = false;
  std::vector<StringCache::CachedData> bcSet;
  if (isCodeCacheEnabled_) {
    using namespace amd::opencl_driver;

    bool checkCache = true;
    for (auto& input : inputs) {
      if (input->Type() == DT_CL) {
        BufferReference* bc = reinterpret_cast<BufferReference*>(input);
        StringCache::CachedData cachedData = {bc->Ptr(), bc->Size()};
        bcSet.push_back(cachedData);
      } else if (input->Type() == DT_CL_HEADER) {
        FileReference* bcFile = reinterpret_cast<FileReference*>(input);
        std::string bc;
        bcFile->ReadToString(bc);
        StringCache::CachedData cachedData = {bc.c_str(), bc.size()};
        bcSet.push_back(cachedData);
      } else {
        buildLog += "Error: unsupported bitcode type for checking cache.\n";
        checkCache = false;
        break;
      }
    }

    std::string dstData = "";
    if (checkCache &&
        codeCache_.getCacheEntry(isCodeCacheEnabled_, bcSet.data(), bcSet.size(), cacheOpt, dstData,
                                 "Compile to LLVM Bitcodes")) {
      std::copy(dstData.begin(), dstData.end(), std::back_inserter(output->Buf()));
      cachedCodeExist = true;
    }
  }

  if (!cachedCodeExist) {
    if (!C->CompileToLLVMBitcode(inputs, output, options)) {
      return false;
    }

    if (isCodeCacheEnabled_) {
      std::string dstData(output->Buf().data(), output->Buf().size());
      if (!codeCache_.makeCacheEntry(bcSet.data(), bcSet.size(), cacheOpt, dstData)) {
        buildLog += "Warning: Failed to caching codes.\n";
        LogWarning("Caching codes failed!");
      }
    }
  }

  return true;
}

bool CacheCompilation::compileAndLinkExecutable(amd::opencl_driver::Compiler* C,
                                                std::vector<amd::opencl_driver::Data*>& inputs,
                                                amd::opencl_driver::Buffer* output,
                                                std::vector<std::string>& options,
                                                std::string& buildLog) {
  std::string cacheOpt;
  cacheOpt = std::accumulate(begin(options), end(options), cacheOpt);

  bool ret = false;
  bool cachedCodeExist = false;
  std::vector<StringCache::CachedData> bcSet;
  if (isCodeCacheEnabled_) {
    for (auto& input : inputs) {
      assert(input->Type() == amd::opencl_driver::DT_LLVM_BC);

      amd::opencl_driver::Buffer* bc = (amd::opencl_driver::Buffer*)input;
      StringCache::CachedData cachedData = {bc->Buf().data(), bc->Size()};
      bcSet.push_back(cachedData);
    }

    std::string dstData = "";
    if (codeCache_.getCacheEntry(isCodeCacheEnabled_, bcSet.data(), bcSet.size(), cacheOpt, dstData,
                                 "Compile and Link Executable")) {
      std::copy(dstData.begin(), dstData.end(), std::back_inserter(output->Buf()));
      cachedCodeExist = true;
    }
  }

  if (!cachedCodeExist) {
    if (!C->CompileAndLinkExecutable(inputs, output, options)) {
      return false;
    }

    if (isCodeCacheEnabled_) {
      std::string dstData(output->Buf().data(), output->Buf().size());
      if (!codeCache_.makeCacheEntry(bcSet.data(), bcSet.size(), cacheOpt, dstData)) {
        buildLog += "Warning: Failed to caching codes.\n";
        LogWarning("Caching codes failed!");
      }
    }
  }

  return true;
}
#endif  // defined(WITH_LIGHTNING_COMPILER) && ! defined(USE_COMGR_LIBRARY)

}  // namespace amd

namespace device {

Settings::Settings() : value_(0) {
  assert((ClExtTotal < (8 * sizeof(extensions_))) && "Too many extensions!");
  extensions_ = 0;
  supportRA_ = true;
  customHostAllocator_ = false;
  waitCommand_ = AMD_OCL_WAIT_COMMAND;
  supportDepthsRGB_ = false;
  enableHwDebug_ = false;
  commandQueues_ = 200;  //!< Field value set to maximum number
                         //!< concurrent Virtual GPUs for default

  overrideLclSet = (!flagIsDefault(GPU_MAX_WORKGROUP_SIZE)) ? 1 : 0;
  overrideLclSet |=
      (!flagIsDefault(GPU_MAX_WORKGROUP_SIZE_2D_X) || !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_2D_Y))
      ? 2
      : 0;
  overrideLclSet |=
      (!flagIsDefault(GPU_MAX_WORKGROUP_SIZE_3D_X) || !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_3D_Y) ||
       !flagIsDefault(GPU_MAX_WORKGROUP_SIZE_3D_Z))
      ? 4
      : 0;

  if (amd::IS_HIP) {
    GPU_SINGLE_ALLOC_PERCENT = 100;
  }

  fenceScopeAgent_ = AMD_OPT_FLUSH;
}

void Memory::saveMapInfo(const void* mapAddress, const amd::Coord3D origin,
                         const amd::Coord3D region, uint mapFlags, bool entire,
                         amd::Image* baseMip) {
  // Map/Unmap must be serialized.
  amd::ScopedLock lock(owner()->lockMemoryOps());

  WriteMapInfo info = {};
  WriteMapInfo* pInfo = &info;
  auto it = writeMapInfo_.find(mapAddress);
  if (it != writeMapInfo_.end()) {
    LogWarning("Double map of the same or overlapped region!");
    pInfo = &it->second;
  }

  if (mapFlags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)) {
    pInfo->origin_ = origin;
    pInfo->region_ = region;
    pInfo->entire_ = entire;
    pInfo->unmapWrite_ = true;
  }
  if (mapFlags & CL_MAP_READ) {
    pInfo->unmapRead_ = true;
  }
  pInfo->baseMip_ = baseMip;

  // Insert into the map if it's the first region
  if (++pInfo->count_ == 1) {
    writeMapInfo_.insert({mapAddress, info});
  }
}

ClBinary::ClBinary(const amd::Device& dev, BinaryImageFormat bifVer)
    : dev_(dev),
      binary_(nullptr),
      size_(0),
      flags_(0),
      origBinary_(nullptr),
      origSize_(0),
      encryptCode_(0),
      elfIn_(nullptr),
      elfOut_(nullptr),
      format_(bifVer) {}

ClBinary::~ClBinary() {
  release();

  if (elfIn_) {
    delete elfIn_;
  }
  if (elfOut_) {
    delete elfOut_;
  }
}

bool ClBinary::setElfTarget() {
  static const uint32_t Target = 21;
  assert(((0xFFFF8000 & Target) == 0) && "ASIC target ID >= 2^15");
  uint16_t elf_target = static_cast<uint16_t>(0x7FFF & Target);
  return elfOut()->setTarget(elf_target, amd::OclElf::CAL_PLATFORM);
  return true;
}

std::string ClBinary::getBIFSymbol(unsigned int symbolID) const {
  size_t nSymbols = 0;
  // Due to PRE & POST defines in bif_section_labels.hpp conflict with
  // PRE & POST struct members in sp3-si-chip-registers.h
  // unable to include bif_section_labels.hpp in device.hpp
  //! @todo: resolve conflict by renaming defines,
  // then include bif_section_labels.hpp in device.hpp &
  // use oclBIFSymbolID instead of unsigned int as a parameter
  const oclBIFSymbolID symID = static_cast<oclBIFSymbolID>(symbolID);
  switch (format_) {
    case BIF_VERSION2: {
      nSymbols = sizeof(BIF20) / sizeof(oclBIFSymbolStruct);
      const oclBIFSymbolStruct* symb = findBIFSymbolStruct(BIF20, nSymbols, symID);
      assert(symb && "BIF20 symbol with symbolID not found");
      if (symb) {
        return std::string(symb->str[bif::PRE]) + std::string(symb->str[bif::POST]);
      }
      break;
    }
    case BIF_VERSION3: {
      nSymbols = sizeof(BIF30) / sizeof(oclBIFSymbolStruct);
      const oclBIFSymbolStruct* symb = findBIFSymbolStruct(BIF30, nSymbols, symID);
      assert(symb && "BIF30 symbol with symbolID not found");
      if (symb) {
        return std::string(symb->str[bif::PRE]) + std::string(symb->str[bif::POST]);
      }
      break;
    }
    default:
      assert(0 && "unexpected BIF type");
      return "";
  }
  return "";
}

void ClBinary::init(amd::option::Options* optionsObj, bool amdilRequired) {
  // option has higher priority than environment variable.
  if ((flags_ & BinarySourceMask) != BinaryRemoveSource) {
    // set to zero
    flags_ = (flags_ & (~BinarySourceMask));

    flags_ |= (optionsObj->oVariables->BinSOURCE ? BinarySaveSource : BinaryNoSaveSource);
  }

  if ((flags_ & BinaryLlvmirMask) != BinaryRemoveLlvmir) {
    // set to zero
    flags_ = (flags_ & (~BinaryLlvmirMask));

    flags_ |= (optionsObj->oVariables->BinLLVMIR ? BinarySaveLlvmir : BinaryNoSaveLlvmir);
  }

  // If amdilRequired is true, force to save AMDIL (for correctness)
  if ((flags_ & BinaryAmdilMask) != BinaryRemoveAmdil || amdilRequired) {
    // set to zero
    flags_ = (flags_ & (~BinaryAmdilMask));
    flags_ |=
        ((optionsObj->oVariables->BinAMDIL || amdilRequired) ? BinarySaveAmdil : BinaryNoSaveAmdil);
  }

  if ((flags_ & BinaryIsaMask) != BinaryRemoveIsa) {
    // set to zero
    flags_ = (flags_ & (~BinaryIsaMask));
    flags_ |= ((optionsObj->oVariables->BinEXE) ? BinarySaveIsa : BinaryNoSaveIsa);
  }

  if ((flags_ & BinaryASMask) != BinaryRemoveAS) {
    // set to zero
    flags_ = (flags_ & (~BinaryASMask));
    flags_ |= ((optionsObj->oVariables->BinAS) ? BinarySaveAS : BinaryNoSaveAS);
  }
}

bool ClBinary::isRecompilable(std::string& llvmBinary, amd::OclElf::oclElfPlatform thePlatform) {
  /* It is recompilable if there is llvmir that was generated for
     the same platform (CPU or GPU) and with the same bitness.

     Note: the bitness has been checked in initClBinary(), no need
           to check it here.
   */
  if (llvmBinary.empty()) {
    return false;
  }

  uint16_t elf_target;
  amd::OclElf::oclElfPlatform platform;
  if (elfIn()->getTarget(elf_target, platform)) {
    if (platform == thePlatform) {
      return true;
    }
    if ((platform == amd::OclElf::COMPLIB_PLATFORM) &&
        (((thePlatform == amd::OclElf::CAL_PLATFORM) &&
          ((elf_target == (uint16_t)EM_AMDIL) || (elf_target == (uint16_t)EM_HSAIL) ||
           (elf_target == (uint16_t)EM_HSAIL_64))) ||
         ((thePlatform == amd::OclElf::CPU_PLATFORM) &&
          ((elf_target == (uint16_t)EM_386) || (elf_target == (uint16_t)EM_X86_64))))) {
      return true;
    }
  }

  return false;
}

void ClBinary::release() {
  if (isBinaryAllocated() && (binary_ != nullptr)) {
    delete[] binary_;
    binary_ = nullptr;
    flags_ &= ~BinaryAllocated;
  }
}

void ClBinary::saveBIFBinary(const char* binaryIn, size_t size) {
  char* image = new char[size];
  memcpy(image, binaryIn, size);

  setBinary(image, size, true);
  return;
}

bool ClBinary::createElfBinary(bool doencrypt, Program::type_t type) {
  release();

  size_t imageSize;
  char* image;
  assert(elfOut_ && "elfOut_ should be initialized in ClBinary::data()");

  // Insert Version string that builds this binary into .comment section
  const device::Info& devInfo = dev_.info();
  std::string buildVerInfo("@(#) ");
  if (devInfo.version_ != nullptr) {
    buildVerInfo.append(devInfo.version_);
    buildVerInfo.append(".  Driver version: ");
    buildVerInfo.append(devInfo.driverVersion_);
  } else {
    // char OpenCLVersion[256];
    // size_t sz;
    // cl_int ret= clGetPlatformInfo(AMD_PLATFORM, CL_PLATFORM_VERSION, 256, OpenCLVersion, &sz);
    // if (ret == CL_SUCCESS) {
    //     buildVerInfo.append(OpenCLVersion, sz);
    // }

    // If CAL is unavailable, just hard-code the OpenCL driver version
    buildVerInfo.append("OpenCL 1.1" AMD_PLATFORM_INFO);
  }

  elfOut_->addSection(amd::OclElf::COMMENT, buildVerInfo.data(), buildVerInfo.size());
  switch (type) {
    case Program::TYPE_NONE: {
      elfOut_->setType(ET_NONE);
      break;
    }
    case Program::TYPE_COMPILED: {
      elfOut_->setType(ET_REL);
      break;
    }
    case Program::TYPE_LIBRARY: {
      elfOut_->setType(ET_DYN);
      break;
    }
    case Program::TYPE_EXECUTABLE: {
      elfOut_->setType(ET_EXEC);
      break;
    }
    default:
      assert(0 && "unexpected elf type");
  }

  if (!elfOut_->dumpImage(&image, &imageSize)) {
    return false;
  }

#if defined(HAVE_BLOWFISH_H)
  if (doencrypt) {
    // Increase the size by 64 to accomodate extra headers
    int outBufSize = (int)(imageSize + 64);
    char* outBuf = new char[outBufSize];
    if (outBuf == nullptr) {
      return false;
    }
    memset(outBuf, '\0', outBufSize);

    int outBytes = 0;
    bool success = amd::oclEncrypt(0, image, imageSize, outBuf, outBufSize, &outBytes);
    delete[] image;
    if (!success) {
      delete[] outBuf;
      return false;
    }
    image = outBuf;
    imageSize = outBytes;
  }
#endif

  setBinary(image, imageSize, true);
  return true;
}

Program::binary_t ClBinary::data() const { return {binary_, size_}; }

bool ClBinary::setBinary(const char* theBinary, size_t theBinarySize, bool allocated) {
  release();

  size_ = theBinarySize;
  binary_ = theBinary;
  if (allocated) {
    flags_ |= BinaryAllocated;
  }
  return true;
}

void ClBinary::setFlags(int encryptCode) {
  encryptCode_ = encryptCode;
  if (encryptCode != 0) {
    flags_ =
        (flags_ &
         (~(BinarySourceMask | BinaryLlvmirMask | BinaryAmdilMask | BinaryIsaMask | BinaryASMask)));
    flags_ |= (BinaryRemoveSource | BinaryRemoveLlvmir | BinaryRemoveAmdil | BinarySaveIsa |
               BinaryRemoveAS);
  }
}

bool ClBinary::decryptElf(const char* binaryIn, size_t size, char** decryptBin, size_t* decryptSize,
                          int* encryptCode) {
  *decryptBin = nullptr;
#if defined(HAVE_BLOWFISH_H)
  int outBufSize = 0;
  if (amd::isEncryptedBIF(binaryIn, (int)size, &outBufSize)) {
    char* outBuf = new (std::nothrow) char[outBufSize];
    if (outBuf == nullptr) {
      return false;
    }

    // Decrypt
    int outDataSize = 0;
    if (!amd::oclDecrypt(binaryIn, (int)size, outBuf, outBufSize, &outDataSize)) {
      delete[] outBuf;
      return false;
    }

    *decryptBin = reinterpret_cast<char*>(outBuf);
    *decryptSize = outDataSize;
    *encryptCode = 1;
  }
#endif
  return true;
}

bool ClBinary::setElfIn() {
  if (elfIn_) return true;

  if (binary_ == nullptr) {
    return false;
  }
  elfIn_ = new amd::OclElf(ELFCLASSNONE, binary_, size_, nullptr, ELF_C_READ);
  if ((elfIn_ == nullptr) || elfIn_->hasError()) {
    if (elfIn_) {
      delete elfIn_;
      elfIn_ = nullptr;
    }
    LogError("Creating input ELF object failed");
    return false;
  }

  return true;
}

void ClBinary::resetElfIn() {
  if (elfIn_) {
    delete elfIn_;
    elfIn_ = nullptr;
  }
}

bool ClBinary::setElfOut(unsigned char eclass, const char* outFile) {
  elfOut_ = new amd::OclElf(eclass, nullptr, 0, outFile, ELF_C_WRITE);
  if ((elfOut_ == nullptr) || elfOut_->hasError()) {
    if (elfOut_) {
      delete elfOut_;
      elfOut_ = nullptr;
    }
    LogError("Creating ouput ELF object failed");
    return false;
  }

  return setElfTarget();
}

void ClBinary::resetElfOut() {
  if (elfOut_) {
    delete elfOut_;
    elfOut_ = nullptr;
  }
}

bool ClBinary::loadLlvmBinary(std::string& llvmBinary,
                              amd::OclElf::oclElfSections& elfSectionType) const {
  // Check if current binary already has LLVMIR
  char* section = nullptr;
  size_t sz = 0;
  const amd::OclElf::oclElfSections SectionTypes[] = {amd::OclElf::LLVMIR, amd::OclElf::SPIR,
                                                      amd::OclElf::SPIRV};

  for (int i = 0; i < 3; ++i) {
    if (elfIn_->getSection(SectionTypes[i], &section, &sz) && section && sz > 0) {
      llvmBinary.append(section, sz);
      elfSectionType = SectionTypes[i];
      return true;
    }
  }

  return false;
}

bool ClBinary::loadCompileOptions(std::string& compileOptions) const {
  char* options = nullptr;
  size_t sz;
  compileOptions.clear();
  if (elfIn_->getSymbol(amd::OclElf::COMMENT, getBIFSymbol(symOpenclCompilerOptions).c_str(),
                        &options, &sz)) {
    if (sz > 0) {
      compileOptions.append(options, sz);
    }
    return true;
  }
  return false;
}

bool ClBinary::loadLinkOptions(std::string& linkOptions) const {
  char* options = nullptr;
  size_t sz;
  linkOptions.clear();
  if (elfIn_->getSymbol(amd::OclElf::COMMENT, getBIFSymbol(symOpenclLinkerOptions).c_str(),
                        &options, &sz)) {
    if (sz > 0) {
      linkOptions.append(options, sz);
    }
    return true;
  }
  return false;
}

void ClBinary::storeCompileOptions(const std::string& compileOptions) {
  elfOut()->addSymbol(amd::OclElf::COMMENT, getBIFSymbol(symOpenclCompilerOptions).c_str(),
                      compileOptions.c_str(), compileOptions.length());
}

void ClBinary::storeLinkOptions(const std::string& linkOptions) {
  elfOut()->addSymbol(amd::OclElf::COMMENT, getBIFSymbol(symOpenclLinkerOptions).c_str(),
                      linkOptions.c_str(), linkOptions.length());
}

bool ClBinary::isSPIR() const {
  char* section = nullptr;
  size_t sz = 0;
  if (elfIn_->getSection(amd::OclElf::LLVMIR, &section, &sz) && section && sz > 0) return false;

  if (elfIn_->getSection(amd::OclElf::SPIR, &section, &sz) && section && sz > 0) return true;

  return false;
}

bool ClBinary::isSPIRV() const {
  char* section = nullptr;
  size_t sz = 0;

  if (elfIn_->getSection(amd::OclElf::SPIRV, &section, &sz) && section && sz > 0) {
    return true;
  }
  return false;
}

}  // namespace device
