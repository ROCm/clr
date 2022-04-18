/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

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
#include "hip_vm.hpp"

hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  HIP_INIT_API(hipMemAddressFree, devPtr, size);

  if (devPtr == nullptr || size == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (auto& dev: g_devices) {
    dev->devices()[0]->virtualFree(devPtr);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags) {
  HIP_INIT_API(hipMemAddressReserve, ptr, size, alignment, addr, flags);

  if (ptr == nullptr ||
      flags !=0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *ptr = nullptr;

  void* startAddress = addr;

  for (auto& dev : g_devices) {
    *ptr = dev->devices()[0]->virtualAlloc(startAddress, size, alignment);

    // if addr==0 we generate the va and use it for other devices
    if (startAddress == nullptr) {
      startAddress = *ptr;
    } else if (*ptr != startAddress) {
      // if we cannot reserve the same VA on other devices, just fail
      for (auto& d : g_devices) {
        if (d == dev) HIP_RETURN(hipErrorOutOfMemory);
        d->devices()[0]->virtualFree(startAddress);
      }
    }
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop, unsigned long long flags) {
  HIP_INIT_API(hipMemCreate, handle, size, prop, flags);

  if (handle == nullptr ||
      size == 0 ||
      flags != 0 ||
      prop == nullptr ||
      prop->type != hipMemAllocationTypePinned ||
      prop->location.type != hipMemLocationTypeDevice ||
      prop->location.id >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Currently only support non-IPC allocations
  if (prop->requestedHandleType != hipMemHandleTypeNone) {
    HIP_RETURN(hipErrorNotSupported);
  }

  const auto& dev_info = g_devices[prop->location.id]->devices()[0]->info();

  if (dev_info.maxPhysicalMemAllocSize_ < size) {
    HIP_RETURN(hipErrorOutOfMemory);
  }
  if (size % dev_info.memBaseAddrAlign_ != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Context* amdContext = g_devices[prop->location.id]->asContext();

  void* ptr = amd::SvmBuffer::malloc(*amdContext, 0, size, dev_info.memBaseAddrAlign_,
                nullptr);

  if (ptr == nullptr) {
    size_t free = 0, total =0;
    hipError_t err = hipMemGetInfo(&free, &total);
    if (err == hipSuccess) {
      LogPrintfError("Allocation failed : Device memory : required :%zu | free :%zu | total :%zu \n", size, free, total);
    }
    HIP_RETURN(hipErrorOutOfMemory);
  }
  size_t offset = 0; //this is ignored
  amd::Memory* memObj = getMemoryObject(ptr, offset);
  //saves the current device id so that it can be accessed later
  memObj->getUserData().deviceId = prop->location.id;
  memObj->getUserData().data = new hip::GenericAllocation(ptr, size, *prop);

  *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(memObj->getUserData().data);

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemExportToShareableHandle(void* shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags) {
  HIP_INIT_API(hipMemExportToShareableHandle, shareableHandle, handle, handleType, flags);

  if (flags != 0 ||
      handle == nullptr ||
      shareableHandle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr) {
  HIP_INIT_API(hipMemGetAccess, flags, location, ptr);

  if (flags == nullptr ||
      location == nullptr ||
      ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop, hipMemAllocationGranularity_flags option) {
  HIP_INIT_API(hipMemGetAllocationGranularity, granularity, prop, option);

  if (granularity == nullptr ||
      prop == nullptr ||
      prop->type != hipMemAllocationTypePinned ||
      prop->location.type != hipMemLocationTypeDevice ||
      prop->location.id >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const auto& dev_info = g_devices[prop->location.id]->devices()[0]->info();

  // Default to that for now.
  *granularity = dev_info.memBaseAddrAlign_;

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle) {
  HIP_INIT_API(hipMemGetAllocationPropertiesFromHandle, prop, handle);

  if (handle == nullptr || prop == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *prop = reinterpret_cast<hip::GenericAllocation*>(handle)->GetProperties();

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle, hipMemAllocationHandleType shHandleType) {
  HIP_INIT_API(hipMemImportFromShareableHandle, handle, osHandle, shHandleType);

  if (handle == nullptr || osHandle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags) {
  HIP_INIT_API(hipMemMap, ptr, size, offset, handle, flags);

  if (ptr == nullptr ||
      handle == nullptr ||
      size == 0 ||
      offset != 0 ||
      flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int  count, hipStream_t stream) {
  HIP_INIT_API(hipMemMapArrayAsync, mapInfoList, count, stream);

  if (mapInfoList == nullptr || count == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) {
  HIP_INIT_API(hipMemRelease, handle);

  if (handle == nullptr) HIP_RETURN(hipErrorInvalidValue);

  hip::GenericAllocation* ga = reinterpret_cast<hip::GenericAllocation*>(handle);

  delete ga;

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr) {
  HIP_INIT_API(hipMemRetainAllocationHandle, handle, addr);

  if (handle == nullptr || addr == nullptr) HIP_RETURN(hipErrorInvalidValue);

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
  HIP_INIT_API(hipMemSetAccess, ptr, size, desc, count);

  if (ptr == nullptr ||
      size == 0 ||
      desc == nullptr ||
      count == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemUnmap(void* ptr, size_t size) {
  HIP_INIT_API(hipMemUnmap, ptr, size);

  if (ptr == nullptr) HIP_RETURN(hipErrorInvalidValue);

  HIP_RETURN(hipSuccess);
}

