/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include "hip_mempool_impl.hpp"

/**
 * API interfaces
 */
extern hipError_t ihipFree(void* ptr);

// ================================================================================================
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
  HIP_INIT_API(hipDeviceGetDefaultMemPool, mem_pool, device);
  if ((mem_pool == nullptr) || (device >= g_devices.size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *mem_pool = reinterpret_cast<hipMemPool_t>(g_devices[device]->GetDefaultMemoryPool());
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool) {
  HIP_INIT_API(hipDeviceSetMemPool, device, mem_pool);
  if ((mem_pool == nullptr) || (device >= g_devices.size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  g_devices[device]->SetCurrentMemoryPool(reinterpret_cast<hip::MemoryPool*>(mem_pool));
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device) {
  HIP_INIT_API(hipDeviceGetMemPool, mem_pool, device);
    if ((mem_pool == nullptr) || (device >= g_devices.size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *mem_pool = reinterpret_cast<hipMemPool_t>(g_devices[device]->GetCurrentMemoryPool());
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream) {
  HIP_INIT_API(hipMallocAsync, dev_ptr, size, stream);
  if ((dev_ptr == nullptr) || (size == 0) || (stream == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto device = reinterpret_cast<hip::Stream*>(stream)->GetDevice();
  auto mem_pool = device->GetCurrentMemoryPool();
  *dev_ptr = reinterpret_cast<hip::MemoryPool*>(mem_pool)->AllocateMemory(
      size, reinterpret_cast<hip::Stream*>(stream));
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream) {
  HIP_INIT_API(hipFreeAsync, dev_ptr, stream);
  if ((dev_ptr == nullptr) || (stream == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  size_t offset = 0;
  auto memory = getMemoryObject(dev_ptr, offset);
  auto id = memory->getUserData().deviceId;
  if (!g_devices[id]->FreeMemory(memory, reinterpret_cast<hip::Stream*>(stream))) {
    //! @todo It's not the most optimal logic. The current implementation has unconditional waits
    HIP_RETURN(ihipFree(dev_ptr));
   }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold) {
  HIP_INIT_API(hipMemPoolTrimTo, mem_pool, min_bytes_to_hold);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::MemoryPool* hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  hip_mem_pool->TrimTo(min_bytes_to_hold);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  HIP_INIT_API(hipMemPoolSetAttribute, mem_pool, attr, value);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  HIP_RETURN(hip_mem_pool->SetAttribute(attr, value));
}

// ================================================================================================
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
  HIP_INIT_API(hipMemPoolGetAttribute, mem_pool, attr, value);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  HIP_RETURN(hip_mem_pool->GetAttribute(attr, value));
}

// ================================================================================================
hipError_t hipMemPoolSetAccess(
    hipMemPool_t mem_pool,
    const hipMemAccessDesc* desc_list,
    size_t count) {
  HIP_INIT_API(hipMemPoolSetAccess, mem_pool, desc_list, count);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolGetAccess(
    hipMemAccessFlags* flags,
    hipMemPool_t mem_pool,
    hipMemLocation* location) {
  HIP_INIT_API(hipMemPoolGetAccess, flags, mem_pool, location);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props) {
  HIP_INIT_API(hipMemPoolCreate, mem_pool, pool_props);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Make sure the pool creation occurs on a valid device
  if ((pool_props->location.type != hipMemLocationTypeDevice) ||
      (pool_props->location.id >= g_devices.size())) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto device = g_devices[pool_props->location.id];
  auto pool = new hip::MemoryPool(device);
  if (pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *mem_pool = reinterpret_cast<hipMemPool_t>(pool);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool) {
  HIP_INIT_API(hipMemPoolDestroy, mem_pool);
  if (mem_pool == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::MemoryPool* hip_mem_pool = reinterpret_cast<hip::MemoryPool*>(mem_pool);
  hip_mem_pool->ReleaseFreedMemory();

  auto device = hip_mem_pool->Device();

  // Force default pool if the current one is destroyed
  if (hip_mem_pool == device->GetCurrentMemoryPool()) {
    device->SetCurrentMemoryPool(device->GetDefaultMemoryPool());
  }

  hip_mem_pool->release();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMallocFromPoolAsync(
    void** dev_ptr,
    size_t size,
    hipMemPool_t mem_pool,
    hipStream_t stream) {
  HIP_INIT_API(hipMallocFromPoolAsync, dev_ptr, size, mem_pool, stream);
  if ((dev_ptr == nullptr) || (size == 0) || (mem_pool == nullptr) || (stream == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *dev_ptr = reinterpret_cast<hip::MemoryPool*>(mem_pool)->AllocateMemory(
      size, reinterpret_cast<hip::Stream*>(stream));
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolExportToShareableHandle(
    void*                      shared_handle,
    hipMemPool_t               mem_pool,
    hipMemAllocationHandleType handle_type,
    unsigned int               flags) {
  HIP_INIT_API(hipMemPoolExportToShareableHandle, shared_handle, mem_pool, handle_type, flags);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolImportFromShareableHandle(
    hipMemPool_t*              mem_pool,
    void*                      shared_handle,
    hipMemAllocationHandleType handle_type,
    unsigned int               flags) {
  HIP_INIT_API(hipMemPoolImportFromShareableHandle, mem_pool, shared_handle, handle_type, flags);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* ptr) {
  HIP_INIT_API(hipMemPoolExportPointer, export_data, ptr);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemPoolImportPointer(
    void**                   ptr,
    hipMemPool_t             mem_pool,
    hipMemPoolPtrExportData* export_data) {
  HIP_INIT_API(hipMemPoolImportPointer, ptr, mem_pool, export_data);
  HIP_RETURN(hipSuccess);
}
