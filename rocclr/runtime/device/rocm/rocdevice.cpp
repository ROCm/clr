//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef WITHOUT_HSA_BACKEND

#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"
#include "thread/monitor.hpp"
#include "CL/cl_ext.h"

#include "amdocl/cl_common.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rocprogram.hpp"
#if defined(WITH_LIGHTNING_COMPILER)
#include "driver/AmdCompiler.h"
#else  // !defined(WITH_LIGHTNING_COMPILER)
#include "device/rocm/roccompilerlib.hpp"
#endif  // !defined(WITH_LIGHTNING_COMPILER)
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocglinterop.hpp"
#ifdef WITH_AMDGPU_PRO
#include "pro/prodriver.hpp"
#endif
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#endif  // WITHOUT_HSA_BACKEND

#define OPENCL_VERSION_STR XSTR(OPENCL_MAJOR) "." XSTR(OPENCL_MINOR)
#define OPENCL_C_VERSION_STR XSTR(OPENCL_C_MAJOR) "." XSTR(OPENCL_C_MINOR)

#ifndef WITHOUT_HSA_BACKEND
namespace device {
extern const char* BlitSourceCode;
}

namespace roc {
amd::Device::Compiler* NullDevice::compilerHandle_;
bool roc::Device::isHsaInitialized_ = false;
hsa_agent_t roc::Device::cpu_agent_ = {0};
std::vector<hsa_agent_t> roc::Device::gpu_agents_;
amd::Monitor* roc::Device::p2p_stage_ops_ = nullptr;
std::vector<Memory*> roc::Device::p2p_stages_;
const bool roc::Device::offlineDevice_ = false;
const bool roc::NullDevice::offlineDevice_ = true;


static HsaDeviceId getHsaDeviceId(hsa_agent_t device, uint32_t& pci_id) {
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(device, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CHIP_ID, &pci_id)) {
    return HSA_INVALID_DEVICE_ID;
  }

  char agent_name[64] = {0};

  if (HSA_STATUS_SUCCESS != hsa_agent_get_info(device, HSA_AGENT_INFO_NAME, agent_name)) {
    return HSA_INVALID_DEVICE_ID;
  }

  if (strncmp(agent_name, "gfx", 3) != 0) {
    return HSA_INVALID_DEVICE_ID;
  }

  uint gfxipVersion = atoi(&agent_name[3]);
  if (gfxipVersion < 900 && GPU_VEGA10_ONLY) {
    return  HSA_INVALID_DEVICE_ID;
  }

  switch (gfxipVersion) {
    case 701:
      return HSA_HAWAII_ID;
    case 801:
      return HSA_CARRIZO_ID;
    case 802:
      return HSA_TONGA_ID;
    case 803:
      return HSA_FIJI_ID;
    case 900:
      return HSA_VEGA10_ID;
    case 901:
      return HSA_VEGA10_HBCC_ID;
    default:
      return HSA_INVALID_DEVICE_ID;
  }
}

bool NullDevice::create(const AMDDeviceInfo& deviceInfo) {
  online_ = false;
  deviceInfo_ = deviceInfo;
  // Mark the device as GPU type
  info_.type_ = CL_DEVICE_TYPE_GPU;
  info_.vendorId_ = 0x1002;

  settings_ = new Settings();
  roc::Settings* hsaSettings = static_cast<roc::Settings*>(settings_);
  if ((hsaSettings == nullptr) || !hsaSettings->create(false, deviceInfo_.gfxipVersion_)) {
    LogError("Error creating settings for nullptr HSA device");
    return false;
  }
  // Report the device name
  ::strcpy(info_.name_, "AMD HSA Device");
  info_.extensions_ = getExtensionString();
  info_.maxWorkGroupSize_ = hsaSettings->maxWorkGroupSize_;
  ::strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");
  info_.oclcVersion_ = "OpenCL C " IF(IS_LIGHTNING, OPENCL_C_VERSION_STR, "1.2") " ";
  info_.spirVersions_ = "";
  strcpy(info_.driverVersion_, "1.0 Provisional (hsa)");
  info_.version_ = "OpenCL " OPENCL_VERSION_STR " ";
  return true;
}

Device::Device(hsa_agent_t bkendDevice)
    : mapCacheOps_(nullptr)
    , mapCache_(nullptr)
    , _bkendDevice(bkendDevice)
    , gpuvm_segment_max_alloc_(0)
    , alloc_granularity_(0)
    , context_(nullptr)
    , xferQueue_(nullptr)
    , xferRead_(nullptr)
    , xferWrite_(nullptr)
    , pro_device_(nullptr)
    , pro_ena_(false)
    , freeMem_(0)
    , numOfVgpus_(0) {
  group_segment_.handle = 0;
  system_segment_.handle = 0;
  system_coarse_segment_.handle = 0;
  gpuvm_segment_.handle = 0;
}

Device::~Device() {
#ifdef WITH_AMDGPU_PRO
  delete pro_device_;
#endif

  // Release cached map targets
  for (uint i = 0; mapCache_ != nullptr && i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] != nullptr) {
      (*mapCache_)[i]->release();
    }
  }
  delete mapCache_;
  delete mapCacheOps_;

  delete p2p_stage_ops_;
  p2p_stage_ops_ = nullptr;

  for (auto buf: p2p_stages_) {
    delete buf;
  }
  p2p_stages_.clear();

  // Destroy temporary buffers for read/write
  delete xferRead_;
  delete xferWrite_;

  // Destroy transfer queue
  if (xferQueue_ && xferQueue_->terminate()) {
    delete xferQueue_;
    xferQueue_ = nullptr;
  }

  if (blitProgram_) {
    delete blitProgram_;
    blitProgram_ = nullptr;
  }

  if (context_ != nullptr) {
    context_->release();
  }

  if (info_.extensions_) {
    delete[] info_.extensions_;
    info_.extensions_ = nullptr;
  }

  if (settings_) {
    delete settings_;
    settings_ = nullptr;
  }
}
bool NullDevice::initCompiler(bool isOffline) {
#if !defined(WITH_LIGHTNING_COMPILER)
  // Initializes g_complibModule and g_complibApi if they were not initialized
  if (g_complibModule == nullptr) {
    if (!LoadCompLib(isOffline)) {
      if (!isOffline) {
        LogError("Error - could not find the compiler library");
      }
      return false;
    }
  }
  // Initialize the compiler handle if has already not been initialized
  // This is destroyed in Device::teardown
  acl_error error;
  if (!compilerHandle_) {
    compilerHandle_ = g_complibApi._aclCompilerInit(nullptr, &error);
    if (error != ACL_SUCCESS) {
      LogError("Error initializing the compiler handle");
      return false;
    }
  }
#endif  // !defined(WITH_LIGHTNING_COMPILER)
  return true;
}

bool NullDevice::destroyCompiler() {
#if defined(WITH_LIGHTNING_COMPILER)
  delete compilerHandle_;
  compilerHandle_ = nullptr;
#else   // !defined(WITH_LIGHTNING_COMPILER)
  if (compilerHandle_ != nullptr) {
    acl_error error = g_complibApi._aclCompilerFini(compilerHandle_);
    if (error != ACL_SUCCESS) {
      LogError("Error closing the compiler");
      return false;
    }
  }
  if (g_complibModule != nullptr) {
    UnloadCompLib();
  }
#endif  // !defined(WITH_LIGHTNING_COMPILER)
  return true;
}

void NullDevice::tearDown() { destroyCompiler(); }
bool NullDevice::init() {
  // Initialize the compiler
  if (!initCompiler(offlineDevice_)) {
    return false;
  }

  // Return without initializing offline device list
  return true;

#if !defined(WITH_LIGHTNING_COMPILER)
  // If there is an HSA enabled device online then skip any offline device
  std::vector<Device*> devices;
  devices = getDevices(CL_DEVICE_TYPE_GPU, false);

  // Load the offline devices
  // Iterate through the set of available offline devices
  for (uint id = 0; id < sizeof(DeviceInfo) / sizeof(AMDDeviceInfo); id++) {
    bool isOnline = false;
    // Check if the particular device is online
    for (unsigned int i = 0; i < devices.size(); i++) {
      if (static_cast<NullDevice*>(devices[i])->deviceInfo_.hsaDeviceId_ ==
          DeviceInfo[id].hsaDeviceId_) {
        isOnline = true;
      }
    }
    if (isOnline) {
      continue;
    }
    NullDevice* nullDevice = new NullDevice();
    if (!nullDevice->create(DeviceInfo[id])) {
      LogError("Error creating new instance of Device.");
      delete nullDevice;
      return false;
    }
    nullDevice->registerDevice();
  }
#endif  // !defined(WITH_LIGHTNING_COMPILER)
  return true;
}
NullDevice::~NullDevice() {
  if (info_.extensions_) {
    delete[] info_.extensions_;
    info_.extensions_ = nullptr;
  }

  if (settings_) {
    delete settings_;
    settings_ = nullptr;
  }
}

hsa_status_t Device::iterateAgentCallback(hsa_agent_t agent, void* data) {
  hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

  hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  if (dev_type == HSA_DEVICE_TYPE_CPU) {
    Device::cpu_agent_ = agent;
  } else if (dev_type == HSA_DEVICE_TYPE_GPU) {
    gpu_agents_.push_back(agent);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_ven_amd_loader_1_00_pfn_t Device::amd_loader_ext_table = {nullptr};

hsa_status_t Device::loaderQueryHostAddress(const void* device, const void** host) {
  return amd_loader_ext_table.hsa_ven_amd_loader_query_host_address
      ? amd_loader_ext_table.hsa_ven_amd_loader_query_host_address(device, host)
      : HSA_STATUS_ERROR;
}

Device::XferBuffers::~XferBuffers() {
  // Destroy temporary buffer for reads
  for (const auto& buf : freeBuffers_) {
    delete buf;
  }
  freeBuffers_.clear();
}

bool Device::XferBuffers::create() {
  Memory* xferBuf = nullptr;
  bool result = false;

  // Create a buffer object
  xferBuf = new Buffer(dev(), bufSize_);

  // Try to allocate memory for the transfer buffer
  if ((nullptr == xferBuf) || !xferBuf->create()) {
    delete xferBuf;
    xferBuf = nullptr;
    LogError("Couldn't allocate a transfer buffer!");
  } else {
    result = true;
    freeBuffers_.push_back(xferBuf);
  }

  return result;
}

Memory& Device::XferBuffers::acquire() {
  Memory* xferBuf = nullptr;
  size_t listSize;

  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  listSize = freeBuffers_.size();

  // If the list is empty, then attempt to allocate a staged buffer
  if (listSize == 0) {
    // Allocate memory
    xferBuf = new Buffer(dev(), bufSize_);

    // Allocate memory for the transfer buffer
    if ((nullptr == xferBuf) || !xferBuf->create()) {
      delete xferBuf;
      xferBuf = nullptr;
      LogError("Couldn't allocate a transfer buffer!");
    } else {
      ++acquiredCnt_;
    }
  }

  if (xferBuf == nullptr) {
    xferBuf = *(freeBuffers_.begin());
    freeBuffers_.erase(freeBuffers_.begin());
    ++acquiredCnt_;
  }

  return *xferBuf;
}

void Device::XferBuffers::release(VirtualGPU& gpu, Memory& buffer) {
  // Make sure buffer isn't busy on the current VirtualGPU, because
  // the next aquire can come from different queue
  //    buffer.wait(gpu);
  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  freeBuffers_.push_back(&buffer);
  --acquiredCnt_;
}

bool Device::init() {
  LogInfo("Initializing HSA stack.");

  // Initialize the compiler
  if (!initCompiler(offlineDevice_)) {
    return false;
  }

  if (HSA_STATUS_SUCCESS != hsa_init()) {
    LogError("hsa_init failed.");
    return false;
  }

  hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(amd_loader_ext_table),
                                       &amd_loader_ext_table);

  if (HSA_STATUS_SUCCESS != hsa_iterate_agents(iterateAgentCallback, nullptr)) {
    return false;
  }

  std::vector<bool> selectedDevices;
  selectedDevices.resize(gpu_agents_.size(), true);

  if (!flagIsDefault(GPU_DEVICE_ORDINAL)) {
    std::fill(selectedDevices.begin(), selectedDevices.end(), false);

    std::string ordinals(GPU_DEVICE_ORDINAL);
    size_t end, pos = 0;
    do {
      end = ordinals.find_first_of(',', pos);
      size_t index = atoi(ordinals.substr(pos, end - pos).c_str());
      selectedDevices.resize(index + 1);
      selectedDevices[index] = true;
      pos = end + 1;
    } while (end != std::string::npos);
  }

  size_t ordinal = 0;
  for (auto agent : gpu_agents_) {
    std::unique_ptr<Device> roc_device(new Device(agent));

    if (!roc_device) {
      LogError("Error creating new instance of Device on then heap.");
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    uint32_t pci_id;
    HsaDeviceId deviceId = getHsaDeviceId(agent, pci_id);
    if (deviceId == HSA_INVALID_DEVICE_ID) {
      LogPrintfError("Invalid HSA device %x", pci_id);
      continue;
    }
    // Find device id in the table
    uint id = HSA_INVALID_DEVICE_ID;
    for (uint i = 0; i < sizeof(DeviceInfo) / sizeof(AMDDeviceInfo); ++i) {
      if (DeviceInfo[i].hsaDeviceId_ == deviceId) {
        id = i;
        break;
      }
    }
    // If the AmdDeviceInfo for the HsaDevice Id could not be found return false
    if (id == HSA_INVALID_DEVICE_ID) {
      LogPrintfWarning("Could not find a DeviceInfo entry for %d", deviceId);
      continue;
    }
    roc_device->deviceInfo_ = DeviceInfo[id];
    roc_device->deviceInfo_.pciDeviceId_ = pci_id;

    // Query the agent's ISA name to fill deviceInfo.gfxipVersion_. We can't
    // have a static mapping as some marketing names cover multiple gfxip.
    hsa_isa_t isa = {0};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) != HSA_STATUS_SUCCESS) {
      continue;
    }

    uint32_t isaNameLength = 0;
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &isaNameLength) != HSA_STATUS_SUCCESS) {
      continue;
    }

    char* isaName = (char*)alloca((size_t)isaNameLength + 1);
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isaName) != HSA_STATUS_SUCCESS) {
      continue;
    }
    isaName[isaNameLength] = '\0';

    std::string str(isaName);
    std::vector<std::string> tokens;
    size_t end, pos = 0;
    do {
      end = str.find_first_of(':', pos);
      tokens.push_back(str.substr(pos, end - pos));
      pos = end + 1;
    } while (end != std::string::npos);

    if (tokens.size() != 5 || tokens[0] != "AMD" || tokens[1] != "AMDGPU") {
      LogError("Not an AMD:AMDGPU ISA name");
      continue;
    }

    uint major = atoi(tokens[2].c_str());
    uint minor = atoi(tokens[3].c_str());
    uint stepping = atoi(tokens[4].c_str());
    if (minor >= 10 && stepping >= 10) {
      LogError("Invalid ISA string");
      continue;
    }

    roc_device->deviceInfo_.gfxipVersion_ = major * 100 + minor * 10 + stepping;

    if (!roc_device->create()) {
      LogError("Error creating new instance of Device.");
      continue;
    }

    // Setup System Memory to be Non-Coherent per user
    // request via environment variable. By default the
    // System Memory is setup to be Coherent
    if (roc_device->settings().enableNCMode_) {
      hsa_status_t err = hsa_amd_coherency_set_type(agent, HSA_AMD_COHERENCY_TYPE_NONCOHERENT);
      if (err != HSA_STATUS_SUCCESS) {
        LogError("Unable to set NC memory policy!");
        continue;
      }
    }

    if (selectedDevices[ordinal++] &&
        (flagIsDefault(GPU_DEVICE_NAME) || GPU_DEVICE_NAME == 0 || GPU_DEVICE_NAME[0] == '\0' ||
         !strcmp(GPU_DEVICE_NAME, roc_device->info_.name_))) {
      roc_device.release()->registerDevice();
    }
  }

  if (0 != Device::numDevices(CL_DEVICE_TYPE_GPU, false)) {
    // Loop through all available devices
    for (auto device1: Device::devices()) {
      // Find all agents that can have access to the current device
      for (auto agent: static_cast<Device*>(device1)->p2pAgents()) {
        // Find cl_device_id associated with the current agent
        for (auto device2: Device::devices()) {
          if (agent.handle == static_cast<Device*>(device2)->getBackendDevice().handle) {
            // Device2 can have access to device1
            device2->p2pDevices_.push_back(as_cl(device1));
          }
        }
      }
    }
  }

  return true;
}

void Device::tearDown() {
  NullDevice::tearDown();
  hsa_shut_down();
}

bool Device::create() {
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_PROFILE, &agent_profile_)) {
    return false;
  }

  // Create HSA settings
  settings_ = new Settings();
  roc::Settings* hsaSettings = static_cast<roc::Settings*>(settings_);
  if ((hsaSettings == nullptr) ||
      !hsaSettings->create((agent_profile_ == HSA_PROFILE_FULL), deviceInfo_.gfxipVersion_)) {
    return false;
  }

  if (!amd::Device::create()) {
    return false;
  }

  uint32_t hsa_bdf_id = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &hsa_bdf_id)) {
    return false;
  }

  info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
  info_.deviceTopology_.pcie.bus = (hsa_bdf_id & (0xFF << 8)) >> 8;
  info_.deviceTopology_.pcie.device = (hsa_bdf_id & (0x1F << 3)) >> 3;
  info_.deviceTopology_.pcie.function = (hsa_bdf_id & 0x07);

#ifdef WITH_AMDGPU_PRO
  // Create amdgpu-pro device interface for SSG support
  pro_device_ = IProDevice::Init(
      info_.deviceTopology_.pcie.bus,
      info_.deviceTopology_.pcie.device,
      info_.deviceTopology_.pcie.function);
  if (pro_device_ != nullptr) {
    pro_ena_ = true;
    settings_->enableExtension(ClAMDLiquidFlash);
  }
#endif

  if (populateOCLDeviceConstants() == false) {
    return false;
  }

#if defined(WITH_LIGHTNING_COMPILER)
  //  create compilation object with cache support
  int gfxipMajor = deviceInfo_.gfxipVersion_ / 100;
  int gfxipMinor = deviceInfo_.gfxipVersion_ / 10 % 10;
  int gfxipStepping = deviceInfo_.gfxipVersion_ % 10;

  // Use compute capability as target (AMD:AMDGPU:major:minor:stepping)
  // with dash as delimiter to be compatible with Windows directory name
  std::ostringstream cacheTarget;
  cacheTarget << "AMD-AMDGPU-" << gfxipMajor << "-" << gfxipMinor << "-" << gfxipStepping;

  amd::CacheCompilation* compObj = new amd::CacheCompilation(
      cacheTarget.str(), "_rocm", OCL_CODE_CACHE_ENABLE, OCL_CODE_CACHE_RESET);
  if (!compObj) {
    LogError("Unable to create cache compilation object!");
    return false;
  }

  cacheCompilation_.reset(compObj);
#endif

  amd::Context::Info info = {0};
  std::vector<amd::Device*> devices;
  devices.push_back(this);

  // Create a dummy context
  context_ = new amd::Context(devices, info);
  if (context_ == nullptr) {
    return false;
  }

  blitProgram_ = new BlitProgram(context_);
  // Create blit programs
  if (blitProgram_ == nullptr || !blitProgram_->create(this)) {
    delete blitProgram_;
    blitProgram_ = nullptr;
    LogError("Couldn't create blit kernels!");
    return false;
  }

  mapCacheOps_ = new amd::Monitor("Map Cache Lock", true);
  if (nullptr == mapCacheOps_) {
    return false;
  }

  mapCache_ = new std::vector<amd::Memory*>();
  if (mapCache_ == nullptr) {
    return false;
  }
  // Use just 1 entry by default for the map cache
  mapCache_->push_back(nullptr);

  if (p2p_stage_ops_ == nullptr) {
    p2p_stage_ops_ = new amd::Monitor("P2P Staging Lock", true);
    if (nullptr == p2p_stage_ops_) {
      return false;
    }
    for (uint i = 0; i < 2; i++) {
      Memory* buf = new Buffer(*this, kP2PStagingSize);
      if ((buf != nullptr) && buf->create()) {
        p2p_stages_.push_back(buf);
      } else {
        delete buf;
        return false;
      }
    }
  }

  if (settings().stagedXferSize_ != 0) {
    // Initialize staged write buffers
    if (settings().stagedXferWrite_) {
      xferWrite_ = new XferBuffers(*this, amd::alignUp(settings().stagedXferSize_, 4 * Ki));
      if ((xferWrite_ == nullptr) || !xferWrite_->create()) {
        LogError("Couldn't allocate transfer buffer objects for read");
        return false;
      }
    }

    // Initialize staged read buffers
    if (settings().stagedXferRead_) {
      xferRead_ = new XferBuffers(*this, amd::alignUp(settings().stagedXferSize_, 4 * Ki));
      if ((xferRead_ == nullptr) || !xferRead_->create()) {
        LogError("Couldn't allocate transfer buffer objects for write");
        return false;
      }
    }
  }

  xferQueue();

  return true;
}

device::Program* NullDevice::createProgram(amd::option::Options* options) {
  return new roc::HSAILProgram(*this);
}

device::Program* Device::createProgram(amd::option::Options* options) {
  return new roc::HSAILProgram(*this);
}

hsa_status_t Device::iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  // TODO: system and device local segment
  Device* dev = reinterpret_cast<Device*>(data);
  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      if (dev->settings().enableLocalMemory_) {
        dev->gpuvm_segment_ = pool;
      }
      break;
    }
    case HSA_REGION_SEGMENT_GROUP:
      dev->group_segment_ = pool;
      break;
    default:
      break;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t Device::iterateCpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  Device* dev = reinterpret_cast<Device*>(data);
  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      uint32_t global_flag = 0;
      hsa_status_t stat =
          hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
      if (stat != HSA_STATUS_SUCCESS) {
        return stat;
      }

      if ((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
        dev->system_segment_ = pool;
      } else {
        dev->system_coarse_segment_ = pool;
      }
      break;
    }
    default:
      break;
  }

  return HSA_STATUS_SUCCESS;
}

bool Device::populateOCLDeviceConstants() {
  info_.available_ = true;

  roc::Settings* hsa_settings = static_cast<roc::Settings*>(settings_);

  int gfxipMajor = deviceInfo_.gfxipVersion_ / 100;
  int gfxipMinor = deviceInfo_.gfxipVersion_ / 10 % 10;
  int gfxipStepping = deviceInfo_.gfxipVersion_ % 10;

  std::ostringstream oss;
  oss << "gfx" << gfxipMajor << gfxipMinor << gfxipStepping;
  ::strcpy(info_.name_, oss.str().c_str());

  char device_name[64] = {0};
  if (HSA_STATUS_SUCCESS == hsa_agent_get_info(_bkendDevice,
                                               (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME,
                                               device_name)) {
    ::strcpy(info_.boardName_, device_name);
  }

  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                         &info_.maxComputeUnits_)) {
    return false;
  }
  assert(info_.maxComputeUnits_ > 0);

  if (HSA_STATUS_SUCCESS != hsa_agent_get_info(_bkendDevice,
                                               (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                                               &info_.globalMemCacheLineSize_)) {
    return false;
  }
  assert(info_.globalMemCacheLineSize_ > 0);

  uint32_t cachesize[4] = {0};
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_CACHE_SIZE, cachesize)) {
    return false;
  }
  assert(cachesize[0] > 0);
  info_.globalMemCacheSize_ = cachesize[0];

  info_.globalMemCacheType_ = CL_READ_WRITE_CACHE;

  info_.type_ = CL_DEVICE_TYPE_GPU;

  info_.extensions_ = getExtensionString();
  info_.nativeVectorWidthDouble_ = info_.preferredVectorWidthDouble_ =
      (settings().doublePrecision_) ? 1 : 0;

  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
                         &info_.maxClockFrequency_)) {
    return false;
  }
  assert(info_.maxClockFrequency_ > 0);

  if (HSA_STATUS_SUCCESS !=
      hsa_amd_agent_iterate_memory_pools(cpu_agent_, Device::iterateCpuMemoryPoolCallback, this)) {
    return false;
  }

  assert(system_segment_.handle != 0);

  if (HSA_STATUS_SUCCESS != hsa_amd_agent_iterate_memory_pools(
                                _bkendDevice, Device::iterateGpuMemoryPoolCallback, this)) {
    return false;
  }

  assert(group_segment_.handle != 0);

  for (auto agent: gpu_agents_) {
    if (agent.handle != _bkendDevice.handle) {
      hsa_status_t err;
      // Can current GPU have access to another GPU memory pool
      hsa_amd_memory_pool_access_t access;
      err = hsa_amd_agent_memory_pool_get_info(agent, gpuvm_segment_, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
      if (err != HSA_STATUS_SUCCESS) {
        continue;
      }

      // Find accessible p2p agents - i.e != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED
      if (HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT == access ||
          HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT == access) {
        p2p_agents_.push_back(agent);
      }
    }
  }

  size_t group_segment_size = 0;
  if (HSA_STATUS_SUCCESS != hsa_amd_memory_pool_get_info(group_segment_,
                                                         HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                         &group_segment_size)) {
    return false;
  }
  assert(group_segment_size > 0);

  info_.localMemSizePerCU_ = group_segment_size;
  info_.localMemSize_ = group_segment_size;

  info_.maxWorkItemDimensions_ = 3;

  if (settings().enableLocalMemory_ && gpuvm_segment_.handle != 0) {
    size_t global_segment_size = 0;
    if (HSA_STATUS_SUCCESS != hsa_amd_memory_pool_get_info(gpuvm_segment_,
                                                           HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                           &global_segment_size)) {
      return false;
    }

    assert(global_segment_size > 0);
    info_.globalMemSize_ = static_cast<cl_ulong>(global_segment_size);

    gpuvm_segment_max_alloc_ =
        cl_ulong(info_.globalMemSize_ * std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);
    assert(gpuvm_segment_max_alloc_ > 0);

    info_.maxMemAllocSize_ = static_cast<cl_ulong>(gpuvm_segment_max_alloc_);

    if (HSA_STATUS_SUCCESS !=
        hsa_amd_memory_pool_get_info(gpuvm_segment_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                     &alloc_granularity_)) {
      return false;
    }

    assert(alloc_granularity_ > 0);
  } else {
    static const cl_ulong kDefaultGlobalMemSize = cl_ulong(1 * Gi);
    info_.globalMemSize_ = kDefaultGlobalMemSize;
    info_.maxMemAllocSize_ = info_.globalMemSize_ / 4;

    if (HSA_STATUS_SUCCESS !=
        hsa_amd_memory_pool_get_info(
            system_segment_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &alloc_granularity_)) {
      return false;
    }
  }

  freeMem_ = info_.globalMemSize_;

  // Make sure the max allocation size is not larger than the available
  // memory size.
  info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_, info_.globalMemSize_);

  /*make sure we don't run anything over 8 params for now*/
  info_.maxParameterSize_ = 1024;  // [TODO]: CAL stack values: 1024*
  // constant

  uint32_t max_work_group_size = 0;
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &max_work_group_size)) {
    return false;
  }
  assert(max_work_group_size > 0);
  max_work_group_size =
      std::min(max_work_group_size, static_cast<uint32_t>(settings().maxWorkGroupSize_));
  info_.maxWorkGroupSize_ = max_work_group_size;

  uint16_t max_workgroup_size[3] = {0, 0, 0};
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &max_workgroup_size)) {
    return false;
  }
  assert(max_workgroup_size[0] != 0 && max_workgroup_size[1] != 0 && max_workgroup_size[2] != 0);

  uint16_t max_work_item_size = static_cast<uint16_t>(max_work_group_size);
  info_.maxWorkItemSizes_[0] = std::min(max_workgroup_size[0], max_work_item_size);
  info_.maxWorkItemSizes_[1] = std::min(max_workgroup_size[1], max_work_item_size);
  info_.maxWorkItemSizes_[2] = std::min(max_workgroup_size[2], max_work_item_size);
  info_.preferredWorkGroupSize_ = settings().preferredWorkGroupSize_;

  info_.nativeVectorWidthChar_ = info_.preferredVectorWidthChar_ = 4;
  info_.nativeVectorWidthShort_ = info_.preferredVectorWidthShort_ = 2;
  info_.nativeVectorWidthInt_ = info_.preferredVectorWidthInt_ = 1;
  info_.nativeVectorWidthLong_ = info_.preferredVectorWidthLong_ = 1;
  info_.nativeVectorWidthFloat_ = info_.preferredVectorWidthFloat_ = 1;

  if (agent_profile_ == HSA_PROFILE_FULL) {  // full-profile = participating in coherent memory,
                                             // base-profile = NUMA based non-coherent memory
    info_.hostUnifiedMemory_ = CL_TRUE;
  }
  info_.memBaseAddrAlign_ =
      8 * (flagIsDefault(MEMOBJ_BASE_ADDR_ALIGN) ? sizeof(cl_long16) : MEMOBJ_BASE_ADDR_ALIGN);
  info_.minDataTypeAlignSize_ = sizeof(cl_long16);

  info_.maxConstantArgs_ = 8;
  info_.preferredConstantBufferSize_ = 16 * Ki;
  info_.maxConstantBufferSize_ = info_.maxMemAllocSize_;
  info_.localMemType_ = CL_LOCAL;
  info_.errorCorrectionSupport_ = false;
  info_.profilingTimerResolution_ = 1;
  info_.littleEndian_ = true;
  info_.compilerAvailable_ = true;
  info_.executionCapabilities_ = CL_EXEC_KERNEL;
  info_.queueProperties_ = CL_QUEUE_PROFILING_ENABLE;
  info_.platform_ = AMD_PLATFORM;
  info_.profile_ = "FULL_PROFILE";
  strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");

  info_.addressBits_ = LP64_SWITCH(32, 64);
  info_.maxSamplers_ = 16;
  info_.bufferFromImageSupport_ = CL_FALSE;
  info_.oclcVersion_ = "OpenCL C " OPENCL_C_VERSION_STR " ";
  info_.spirVersions_ = "";

  uint16_t major, minor;
  if (hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_VERSION_MAJOR, &major) !=
          HSA_STATUS_SUCCESS ||
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_VERSION_MINOR, &minor) !=
          HSA_STATUS_SUCCESS) {
    return false;
  }
  std::stringstream ss;
  ss << AMD_BUILD_STRING " (HSA" << major << "." << minor << "," IF(IS_LIGHTNING, "LC", "HSAIL") ")";

  strcpy(info_.driverVersion_, ss.str().c_str());
  info_.version_ = "OpenCL " /*OPENCL_VERSION_STR*/"1.2" " ";

  info_.builtInKernels_ = "";
  info_.linkerAvailable_ = true;
  info_.preferredInteropUserSync_ = true;
  info_.printfBufferSize_ = PrintfDbg::WorkitemDebugSize * info().maxWorkGroupSize_;
  info_.vendorId_ = 0x1002;  // AMD's PCIe vendor id

  info_.maxGlobalVariableSize_ = static_cast<size_t>(info_.maxMemAllocSize_);
  info_.globalVariablePreferredTotalSize_ = static_cast<size_t>(info_.globalMemSize_);

  // Populate the single config setting.
  info_.singleFPConfig_ =
      CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

  if (hsa_settings->doublePrecision_) {
    info_.doubleFPConfig_ = info_.singleFPConfig_ | CL_FP_DENORM;
    info_.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  }

  if (hsa_settings->singleFpDenorm_) {
    info_.singleFPConfig_ |= CL_FP_DENORM;
  }

  info_.preferredPlatformAtomicAlignment_ = 0;
  info_.preferredGlobalAtomicAlignment_ = 0;
  info_.preferredLocalAtomicAlignment_ = 0;

  uint8_t hsa_extensions[128];
  if (HSA_STATUS_SUCCESS !=
      hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_EXTENSIONS, hsa_extensions)) {
    return false;
  }

  assert(HSA_EXTENSION_IMAGES < 8);
  const bool image_is_supported = ((hsa_extensions[0] & (1 << HSA_EXTENSION_IMAGES)) != 0);
  if (image_is_supported) {
    // Images
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS),
                           &info_.maxSamplers_)) {
      return false;
    }

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES),
                           &info_.maxReadImageArgs_)) {
      return false;
    }

    // TODO: no attribute for write image.
    info_.maxWriteImageArgs_ = 8;

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES),
                           &info_.maxReadWriteImageArgs_)) {
      return false;
    }

    uint32_t image_max_dim[3];
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS),
                           &image_max_dim)) {
      return false;
    }

    info_.image2DMaxWidth_ = image_max_dim[0];
    info_.image2DMaxHeight_ = image_max_dim[1];

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS),
                           &image_max_dim)) {
      return false;
    }

    info_.image3DMaxWidth_ = image_max_dim[0];
    info_.image3DMaxHeight_ = image_max_dim[1];
    info_.image3DMaxDepth_ = image_max_dim[2];

    uint32_t max_array_size = 0;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS),
                           &max_array_size)) {
      return false;
    }

    info_.imageMaxArraySize_ = max_array_size;

    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice,
                           static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS),
                           &image_max_dim)) {
      return false;
    }
    info_.imageMaxBufferSize_ = image_max_dim[0];

    info_.imagePitchAlignment_ = 256;

    info_.imageBaseAddressAlignment_ = 256;

    info_.bufferFromImageSupport_ = CL_FALSE;

    info_.imageSupport_ = (info_.maxReadWriteImageArgs_ > 0) ? CL_TRUE : CL_FALSE;
  }

  // Enable SVM Capabilities of Hsa device. Ensure
  // user has not setup memory to be non-coherent
  info_.svmCapabilities_ = 0;
  if (hsa_settings->enableNCMode_ == false) {
    info_.svmCapabilities_ = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
    info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
    // Report fine-grain system only on full profile
    if (agent_profile_ == HSA_PROFILE_FULL) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
    }
#if !defined(WITH_LIGHTNING_COMPILER)
    // Report atomics capability based on GFX IP, control on Hawaii
    // and Vega10.
    if (info_.hostUnifiedMemory_ ||
        ((deviceInfo_.gfxipVersion_ >= 800) && (deviceInfo_.gfxipVersion_ < 900))) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
    }
#endif  // !defined(WITH_LIGHTNING_COMPILER)
  }

  if (settings().checkExtension(ClAmdDeviceAttributeQuery)) {
    info_.simdPerCU_ = deviceInfo_.simdPerCU_;
    info_.simdWidth_ = deviceInfo_.simdWidth_;
    info_.simdInstructionWidth_ = deviceInfo_.simdInstructionWidth_;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice, HSA_AGENT_INFO_WAVEFRONT_SIZE, &info_.wavefrontWidth_)) {
      return false;
    }
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_WIDTH, &info_.globalMemChannels_)) {
      return false;
    }
    info_.globalMemChannels_ /= 32;
    info_.globalMemChannelBanks_ = 4;
    info_.globalMemChannelBankWidth_ = deviceInfo_.memChannelBankWidth_;
    info_.localMemSizePerCU_ = deviceInfo_.localMemSizePerCU_;
    info_.localMemBanks_ = deviceInfo_.localMemBanks_;
    info_.gfxipVersion_ = deviceInfo_.gfxipVersion_;
    info_.numAsyncQueues_ = kMaxAsyncQueues;
    info_.numRTQueues_ = info_.numAsyncQueues_;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(_bkendDevice, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &info_.numRTCUs_)) {
      return false;
    }
    //TODO: set to true once thread trace support is available
    info_.threadTraceEnable_ = false;
  }

  info_.maxPipePacketSize_ = info_.maxMemAllocSize_;
  info_.maxPipeActiveReservations_ = 16;
  info_.maxPipeArgs_ = 16;

  return true;
}

device::VirtualDevice* Device::createVirtualDevice(amd::CommandQueue* queue) {
  bool profiling = (queue != nullptr) && queue->properties().test(CL_QUEUE_PROFILING_ENABLE);

  // Initialization of heap and other resources occur during the command
  // queue creation time.
  VirtualGPU* virtualDevice = new VirtualGPU(*this);

  if (!virtualDevice->create(profiling)) {
    delete virtualDevice;
    return nullptr;
  }

  if (profiling) {
    hsa_amd_profiling_set_profiler_enabled(virtualDevice->gpu_queue(), 1);
  }

  return virtualDevice;
}

bool Device::globalFreeMemory(size_t* freeMemory) const {
  const uint TotalFreeMemory = 0;
  const uint LargestFreeBlock = 1;

  freeMemory[TotalFreeMemory] = freeMem_ / Ki;

  // since there is no memory heap on ROCm, the biggest free block is
  // equal to total free local memory
  freeMemory[LargestFreeBlock] = freeMemory[TotalFreeMemory];

  return true;
}

bool Device::bindExternalDevice(uint flags, void* const gfxDevice[], void* gfxContext,
                                bool validateOnly) {
#if defined(_WIN32)
  return false;
#else
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;

  MesaInterop::MESA_INTEROP_KIND kind = MesaInterop::MESA_INTEROP_NONE;
  MesaInterop::DisplayHandle display;
  MesaInterop::ContextHandle context;

  if ((flags & amd::Context::EGLDeviceKhr) != 0) {
    kind = MesaInterop::MESA_INTEROP_EGL;
    display.eglDisplay = reinterpret_cast<EGLDisplay>(gfxDevice[amd::Context::GLDeviceKhrIdx]);
    context.eglContext = reinterpret_cast<EGLContext>(gfxContext);
  } else {
    kind = MesaInterop::MESA_INTEROP_GLX;
    display.glxDisplay = reinterpret_cast<Display*>(gfxDevice[amd::Context::GLDeviceKhrIdx]);
    context.glxContext = reinterpret_cast<GLXContext>(gfxContext);
  }

  mesa_glinterop_device_info info;
  info.size = sizeof(mesa_glinterop_device_info);
  MesaInterop temp;
  if (!temp.Bind(kind, display, context)) {
    assert(false && "Failed mesa interop bind.");
    return false;
  }

  if (!temp.GetInfo(info)) {
    assert(false && "Failed to get mesa interop device info.");
    return false;
  }

  bool match = true;
  match &= info_.deviceTopology_.pcie.bus == info.pci_bus;
  match &= info_.deviceTopology_.pcie.device == info.pci_device;
  match &= info_.deviceTopology_.pcie.function == info.pci_function;
  match &= info_.vendorId_ == info.vendor_id;
  match &= deviceInfo_.pciDeviceId_ == info.device_id;

  if (!validateOnly) mesa_ = temp;

  return match;
#endif
}

bool Device::unbindExternalDevice(uint flags, void* const gfxDevice[], void* gfxContext,
                                  bool validateOnly) {
#if defined(_WIN32)
  return false;
#else
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;
  if (!validateOnly) mesa_.Unbind();
  return true;
#endif
}

amd::Memory* Device::findMapTarget(size_t size) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  amd::Memory* map = nullptr;
  size_t minSize = 0;
  size_t maxSize = 0;
  uint mapId = mapCache_->size();
  uint releaseId = mapCache_->size();

  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); i++) {
    if ((*mapCache_)[i] != nullptr) {
      // Requested size is smaller than the entry size
      if (size < (*mapCache_)[i]->getSize()) {
        if ((minSize == 0) || (minSize > (*mapCache_)[i]->getSize())) {
          minSize = (*mapCache_)[i]->getSize();
          mapId = i;
        }
      }
      // Requeted size matches the entry size
      else if (size == (*mapCache_)[i]->getSize()) {
        mapId = i;
        break;
      } else {
        // Find the biggest map target in the list
        if (maxSize < (*mapCache_)[i]->getSize()) {
          maxSize = (*mapCache_)[i]->getSize();
          releaseId = i;
        }
      }
    }
  }

  // Check if we found any map target
  if (mapId < mapCache_->size()) {
    map = (*mapCache_)[mapId];
    (*mapCache_)[mapId] = nullptr;
  }
  // If cache is full, then release the biggest map target
  else if (releaseId < mapCache_->size()) {
    (*mapCache_)[releaseId]->release();
    (*mapCache_)[releaseId] = nullptr;
  }

  return map;
}

bool Device::addMapTarget(amd::Memory* memory) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  // the svm memory shouldn't be cached
  if (!memory->canBeCached()) {
    return false;
  }
  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] == nullptr) {
      (*mapCache_)[i] = memory;
      return true;
    }
  }

  // Add a new entry
  mapCache_->push_back(memory);

  return true;
}

Memory* Device::getRocMemory(amd::Memory* mem) const {
  return static_cast<roc::Memory*>(mem->getDeviceMemory(*this));
}


device::Memory* Device::createMemory(amd::Memory& owner) const {
  roc::Memory* memory = nullptr;
  if (owner.asBuffer()) {
    memory = new roc::Buffer(*this, owner);
  } else if (owner.asImage()) {
    memory = new roc::Image(*this, owner);
  } else {
    LogError("Unknown memory type");
  }

  if (memory == nullptr) {
    return nullptr;
  }

  bool result = memory->create();

  if (!result) {
    LogError("Failed creating memory");
    delete memory;
    return nullptr;
  }

  // Initialize if the memory is a pipe object
  if (owner.getType() == CL_MEM_OBJECT_PIPE) {
    // Pipe initialize in order read_idx, write_idx, end_idx. Refer clk_pipe_t structure.
    // Init with 3 DWORDS for 32bit addressing and 6 DWORDS for 64bit
    size_t pipeInit[3] = { 0, 0, owner.asPipe()->getMaxNumPackets() };
    xferMgr().writeBuffer((void *)pipeInit, *memory, amd::Coord3D(0), amd::Coord3D(sizeof(pipeInit)));
  }

  // Transfer data only if OCL context has one device.
  // Cache coherency layer will update data for multiple devices
  if (!memory->isHostMemDirectAccess() && owner.asImage() && (owner.parent() == nullptr) &&
      (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) && (owner.getContext().devices().size() == 1)) {
    // To avoid recurssive call to Device::createMemory, we perform
    // data transfer to the view of the image.
    amd::Image* imageView = owner.asImage()->createView(
        owner.getContext(), owner.asImage()->getImageFormat(), xferQueue());

    if (imageView == nullptr) {
      LogError("[OCL] Fail to allocate view of image object");
      return nullptr;
    }

    Image* devImageView = new roc::Image(static_cast<const Device&>(*this), *imageView);
    if (devImageView == nullptr) {
      LogError("[OCL] Fail to allocate device mem object for the view");
      imageView->release();
      return nullptr;
    }

    if (devImageView != nullptr && !devImageView->createView(static_cast<roc::Image&>(*memory))) {
      LogError("[OCL] Fail to create device mem object for the view");
      delete devImageView;
      imageView->release();
      return nullptr;
    }

    imageView->replaceDeviceMemory(this, devImageView);

    result = xferMgr().writeImage(owner.getHostMem(), *devImageView, amd::Coord3D(0, 0, 0),
                                  imageView->getRegion(), 0, 0, true);

    // Release host memory, since runtime copied data
    owner.setHostMem(nullptr);

    imageView->release();
  }

  // Prepin sysmem buffer for possible data synchronization between CPU and GPU
  if (!memory->isHostMemDirectAccess() && (owner.getHostMem() != nullptr)) {
    memory->pinSystemMemory(owner.getHostMem(), owner.getSize());
  }

  if (!result) {
    delete memory;
    return nullptr;
  }

  return memory;
}

void* Device::hostAlloc(size_t size, size_t alignment, bool atomics) const {
  void* ptr = nullptr;
  const hsa_amd_memory_pool_t segment = (!atomics)
      ? (system_coarse_segment_.handle != 0) ? system_coarse_segment_ : system_segment_
      : system_segment_;
  assert(segment.handle != 0);
  hsa_status_t stat = hsa_amd_memory_pool_allocate(segment, size, 0, &ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail allocation host memory");
    return nullptr;
  }

  stat = hsa_amd_agents_allow_access(gpu_agents_.size(), &gpu_agents_[0], nullptr, ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail hsa_amd_agents_allow_access");
    return nullptr;
  }

  return ptr;
}

void Device::hostFree(void* ptr, size_t size) const { memFree(ptr, size); }

void* Device::deviceLocalAlloc(size_t size) const {
  if (gpuvm_segment_.handle == 0 || gpuvm_segment_max_alloc_ == 0) {
    return nullptr;
  }

  void* ptr = nullptr;
  hsa_status_t stat = hsa_amd_memory_pool_allocate(gpuvm_segment_, size, 0, &ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail allocation local memory");
    return nullptr;
  }

  if (p2pAgents().size() > 0) {
    stat = hsa_amd_agents_allow_access(p2pAgents().size(), p2pAgents().data(), nullptr, ptr);
    if (stat != HSA_STATUS_SUCCESS) {
      LogError("Allow p2p acces for memory allocation");
      memFree(ptr, size);
      return nullptr;
    }
  }

  return ptr;
}

void Device::memFree(void* ptr, size_t size) const {
  hsa_status_t stat = hsa_amd_memory_pool_free(ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail freeing local memory");
  }
}

void Device::updateFreeMemory(size_t size, bool free) {
  if (free) {
    freeMem_ += size;
  }
  else {
    freeMem_ -= size;
  }
}

void* Device::svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                       void* svmPtr) const {
  amd::Memory* mem = nullptr;
  if (nullptr == svmPtr) {
    bool atomics = (flags & CL_MEM_SVM_ATOMICS) != 0;
    void* ptr = hostAlloc(size, alignment, atomics);

    if (ptr != nullptr) {
      // Copy paste from ORCA code.
      // create a hidden buffer, which will allocated on the device later
      mem = new (context) amd::Buffer(context, CL_MEM_USE_HOST_PTR, size, ptr);
      if (mem == nullptr) {
        LogError("failed to create a svm mem object!");
        return nullptr;
      }

      if (!mem->create(ptr)) {
        LogError("failed to create a svm hidden buffer!");
        mem->release();
        return nullptr;
      }

      // add the information to context so that we can use it later.
      amd::SvmManager::AddSvmBuffer(ptr, mem);

      return ptr;
    } else {
      return nullptr;
    }
  } else {
    // Copy paste from ORCA code.
    // Find the existing amd::mem object
    mem = amd::SvmManager::FindSvmBuffer(svmPtr);

    if (nullptr == mem) {
      return nullptr;
    }

    return svmPtr;
  }
}

void Device::svmFree(void* ptr) const {
  amd::Memory* svmMem = nullptr;
  svmMem = amd::SvmManager::FindSvmBuffer(ptr);
  if (nullptr != svmMem) {
    svmMem->release();
    amd::SvmManager::RemoveSvmBuffer(ptr);
    hostFree(ptr);
  }
}

VirtualGPU* Device::xferQueue() const {
  if (!xferQueue_) {
    // Create virtual device for internal memory transfer
    Device* thisDevice = const_cast<Device*>(this);
    thisDevice->xferQueue_ = reinterpret_cast<VirtualGPU*>(thisDevice->createVirtualDevice());
    if (!xferQueue_) {
      LogError("Couldn't create the device transfer manager!");
    }
  }
  xferQueue_->enableSyncBlit();
  return xferQueue_;
}
}
#endif  // WITHOUT_HSA_BACKEND
