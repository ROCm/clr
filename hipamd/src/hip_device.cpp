/* Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc.

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

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"
#include "hip_mempool_impl.hpp"

namespace hip {

// ================================================================================================
amd::HostQueue* Device::NullStream(bool skip_alloc) {
  amd::HostQueue* null_queue = null_stream_.asHostQueue(skip_alloc);
  if (null_queue == nullptr) {
    return nullptr;
  }
  // Wait for all active streams before executing commands on the default
  iHipWaitActiveStreams(null_queue);
  return null_queue;
}

// ================================================================================================
bool Device::Create() {
  // Create default memory pool
  default_mem_pool_ = new MemoryPool(this);
  if (default_mem_pool_ == nullptr) {
    return false;
  }
  // Current is default pool after device creation
  current_mem_pool_ = default_mem_pool_;
  return true;
}

// ================================================================================================
void Device::AddMemoryPool(MemoryPool* pool) {
  if (auto it = mem_pools_.find(pool); it == mem_pools_.end()) {
    mem_pools_.insert(pool);
  }
}

// ================================================================================================
void Device::RemoveMemoryPool(MemoryPool* pool) {
  if (auto it = mem_pools_.find(pool); it != mem_pools_.end()) {
    mem_pools_.erase(it);
  }
}

// ================================================================================================
bool Device::FreeMemory(amd::Memory* memory, Stream* stream) {
  // Search for memory in the entire list of pools
  for (auto& it : mem_pools_) {
    if (it->FreeMemory(memory, stream)) {
      return true;
    }
  }
  return false;
}

// ================================================================================================
void Device::ReleaseFreedMemory(Stream* stream) {
  // Search for memory in the entire list of pools
  for (auto& it : mem_pools_) {
    it->ReleaseFreedMemory(stream);
  }
}

// ================================================================================================
Device::~Device() {
  if (default_mem_pool_ != nullptr) {
    default_mem_pool_->release();
  }
}

}

void ihipDestroyDevice() {
  for (auto deviceHandle : g_devices) {
    delete deviceHandle;
  }
}

hipError_t ihipDeviceGet(hipDevice_t* device, int deviceId) {
  if (deviceId < 0 || static_cast<size_t>(deviceId) >= g_devices.size() || device == nullptr) {
    return hipErrorInvalidDevice;
  }
  *device = deviceId;
  return hipSuccess;
}

hipError_t hipDeviceGet(hipDevice_t* device, int deviceId) {
  HIP_INIT_API(hipDeviceGet, device, deviceId);

  HIP_RETURN(ihipDeviceGet(device, deviceId));
}

hipError_t hipDeviceTotalMem (size_t *bytes, hipDevice_t device) {

  HIP_INIT_API(hipDeviceTotalMem, bytes, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (bytes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();

  *bytes = info.globalMemSize_;

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device) {

  HIP_INIT_API(hipDeviceComputeCapability, major, minor, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (major == nullptr || minor == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& isa = deviceHandle->isa();
  *major = isa.versionMajor();
  *minor = isa.versionMinor();

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetCount(int* count) {
  HIP_INIT_API(hipDeviceGetCount, count);

  HIP_RETURN(ihipDeviceGetCount(count));
}

hipError_t ihipDeviceGetCount(int* count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }

  // Get all available devices
  *count = g_devices.size();

  if (*count < 1) {
    return hipErrorNoDevice;
  }

  return hipSuccess;
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t device) {

  HIP_INIT_API(hipDeviceGetName, (void*)name, len, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (name == nullptr || len <= 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();
  const auto nameLen = ::strlen(info.boardName_);

  // Make sure that the size of `dest` is big enough to hold `src` including
  // trailing zero byte
  if (nameLen > (cl_uint)(len - 1)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  ::memcpy(name, info.boardName_, (nameLen + 1));

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device) {
  HIP_INIT_API(hipDeviceGetUuid, reinterpret_cast<void*>(uuid), device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (uuid == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();

  ::strncpy(uuid->bytes, info.uuid_, 16);

  HIP_RETURN(hipSuccess);
}

hipError_t ihipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device) {
  if (props == nullptr) {
    return hipErrorInvalidValue;
  }

  if (unsigned(device) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }
  auto* deviceHandle = g_devices[device]->devices()[0];

  constexpr auto int32_max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  hipDeviceProp_t deviceProps = {0};

  const auto& info = deviceHandle->info();
  const auto& isa = deviceHandle->isa();
  ::strncpy(deviceProps.name, info.boardName_, 128);
  deviceProps.totalGlobalMem = info.globalMemSize_;
  deviceProps.sharedMemPerBlock = info.localMemSizePerCU_;
  deviceProps.regsPerBlock = info.availableRegistersPerCU_;
  deviceProps.warpSize = info.wavefrontWidth_;
  deviceProps.maxThreadsPerBlock = info.maxWorkGroupSize_;
  deviceProps.maxThreadsDim[0] = info.maxWorkItemSizes_[0];
  deviceProps.maxThreadsDim[1] = info.maxWorkItemSizes_[1];
  deviceProps.maxThreadsDim[2] = info.maxWorkItemSizes_[2];
  deviceProps.maxGridSize[0] = int32_max;
  deviceProps.maxGridSize[1] = int32_max;
  deviceProps.maxGridSize[2] = int32_max;
  deviceProps.clockRate = info.maxEngineClockFrequency_ * 1000;
  deviceProps.memoryClockRate = info.maxMemoryClockFrequency_ * 1000;
  deviceProps.memoryBusWidth = info.globalMemChannels_;
  deviceProps.totalConstMem = std::min(info.maxConstantBufferSize_, int32_max);
  deviceProps.major = isa.versionMajor();
  deviceProps.minor = isa.versionMinor();
  deviceProps.multiProcessorCount = info.maxComputeUnits_;
  deviceProps.l2CacheSize = info.l2CacheSize_;
  deviceProps.maxThreadsPerMultiProcessor = info.maxThreadsPerCU_;
  deviceProps.computeMode = 0;
  deviceProps.clockInstructionRate = info.timeStampFrequency_;
  deviceProps.arch.hasGlobalInt32Atomics = 1;
  deviceProps.arch.hasGlobalFloatAtomicExch = 1;
  deviceProps.arch.hasSharedInt32Atomics = 1;
  deviceProps.arch.hasSharedFloatAtomicExch = 1;
  deviceProps.arch.hasFloatAtomicAdd = 1;
  deviceProps.arch.hasGlobalInt64Atomics = 1;
  deviceProps.arch.hasSharedInt64Atomics = 1;
  deviceProps.arch.hasDoubles = 1;
  deviceProps.arch.hasWarpVote = 1;
  deviceProps.arch.hasWarpBallot = 1;
  deviceProps.arch.hasWarpShuffle = 1;
  deviceProps.arch.hasFunnelShift = 0;
  deviceProps.arch.hasThreadFenceSystem = 1;
  deviceProps.arch.hasSyncThreadsExt = 0;
  deviceProps.arch.hasSurfaceFuncs = 0;
  deviceProps.arch.has3dGrid = 1;
  deviceProps.arch.hasDynamicParallelism = 0;
  deviceProps.concurrentKernels = 1;
  deviceProps.pciDomainID = info.pciDomainID;
  deviceProps.pciBusID = info.deviceTopology_.pcie.bus;
  deviceProps.pciDeviceID = info.deviceTopology_.pcie.device;
  deviceProps.maxSharedMemoryPerMultiProcessor = info.localMemSizePerCU_;
  deviceProps.canMapHostMemory = 1;
  // FIXME: This should be removed, targets can have character names as well.
  deviceProps.gcnArch = isa.versionMajor() * 100 + isa.versionMinor() * 10 + isa.versionStepping();
  sprintf(deviceProps.gcnArchName, "%s", isa.targetId());
  deviceProps.cooperativeLaunch = info.cooperativeGroups_;
  deviceProps.cooperativeMultiDeviceLaunch = info.cooperativeMultiDeviceGroups_;

  deviceProps.cooperativeMultiDeviceUnmatchedFunc = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedGridDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedBlockDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedSharedMem = info.cooperativeMultiDeviceGroups_;

  deviceProps.maxTexture1DLinear = std::min(16 * info.imageMaxBufferSize_, int32_max);  // Max pixel size is 16 bytes
  deviceProps.maxTexture1D    = std::min(info.image1DMaxWidth_, int32_max);
  deviceProps.maxTexture2D[0] = std::min(info.image2DMaxWidth_, int32_max);
  deviceProps.maxTexture2D[1] = std::min(info.image2DMaxHeight_, int32_max);
  deviceProps.maxTexture3D[0] = std::min(info.image3DMaxWidth_, int32_max);
  deviceProps.maxTexture3D[1] = std::min(info.image3DMaxHeight_, int32_max);
  deviceProps.maxTexture3D[2] = std::min(info.image3DMaxDepth_, int32_max);
  deviceProps.hdpMemFlushCntl = info.hdpMemFlushCntl;
  deviceProps.hdpRegFlushCntl = info.hdpRegFlushCntl;

  deviceProps.memPitch = std::min(info.maxMemAllocSize_, int32_max);
  deviceProps.textureAlignment = info.imageBaseAddressAlignment_;
  deviceProps.texturePitchAlignment = info.imagePitchAlignment_;
  deviceProps.kernelExecTimeoutEnabled = 0;
  deviceProps.ECCEnabled = info.errorCorrectionSupport_ ? 1 : 0;
  deviceProps.isLargeBar = info.largeBar_ ? 1 : 0;
  deviceProps.asicRevision = info.asicRevision_;

  // HMM capabilities
  deviceProps.managedMemory = info.hmmSupported_;
  deviceProps.concurrentManagedAccess = info.hmmSupported_;
  deviceProps.directManagedMemAccessFromHost = info.hmmDirectHostAccess_;
  deviceProps.pageableMemoryAccess = info.hmmCpuMemoryAccessible_;
  deviceProps.pageableMemoryAccessUsesHostPageTables = info.hostUnifiedMemory_;

  *props = deviceProps;
  return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device) {
  HIP_INIT_API(hipGetDeviceProperties, props, device);

  HIP_RETURN(ihipGetDeviceProperties(props, device));
}
