//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef WITHOUT_HSA_BACKEND

#include "hsa_ext_amd.h"
#include "lnxheaders.h"
#include "prodevice.hpp"
#include "amdgpu_drm.h"

namespace roc {

constexpr uint32_t kMaxDevices  = 32;
constexpr uint32_t kAtiVendorId = 0x1002;

IProDevice* IProDevice::Init(uint32_t bus, uint32_t dev, uint32_t func)
{
  ProDevice* pro_device = new ProDevice();

  if (pro_device == nullptr || !pro_device->Create(bus, dev, func)) {
    delete pro_device;
    return nullptr;
  }
  return pro_device;
}

ProDevice::~ProDevice() {
  delete alloc_ops_;

  if (dev_handle_ != nullptr) {
    amdgpu_device_deinitialize(dev_handle_);
  }
  if (file_desc_ > 0) {
    close(file_desc_);
  }
}

#ifndef AMDGPU_CAPABILITY_SSG_FLAG
#define AMDGPU_CAPABILITY_SSG_FLAG 4
#endif

// ================================================================================================
// Open drm device and initialize it. And also get the drm information.
bool ProDevice::Create(uint32_t bus, uint32_t device, uint32_t func) {
  drmDevicePtr  devices[kMaxDevices] = { };
  int32_t       device_count        = drmGetDevices(devices, kMaxDevices);
  bool          result = false;

  for (int32_t i = 0; i < device_count; i++) {
    // Check if the device vendor is AMD
    if (devices[i]->deviceinfo.pci->vendor_id != kAtiVendorId) {
      continue;
    }
    if ((devices[i]->businfo.pci->bus == bus) &&
        (devices[i]->businfo.pci->dev == device) &&
			  (devices[i]->businfo.pci->func == func)) {

      // pDevices[i]->nodes[DRM_NODE_PRIMARY];
      // Using render node here so that we can do the off-screen rendering without authentication
      file_desc_ = open(devices[i]->nodes[DRM_NODE_RENDER], O_RDWR, 0);

      if (file_desc_ > 0) {
        void* data, *file, *cap;

        // Initialize the admgpu device.
        if (amdgpu_device_initialize(file_desc_, &major_ver_,
                                     &minor_ver_, &dev_handle_) == 0) {
          uint32_t version = 0;
          // amdgpu_query_gpu_info will never fail only if it is initialized
          amdgpu_query_gpu_info(dev_handle_, &gpu_info_);

          drm_amdgpu_capability cap = {};
          amdgpu_query_info(dev_handle_, AMDGPU_INFO_CAPABILITY, sizeof(drm_amdgpu_capability), &cap);

          // Check if DGMA and SSG are available
          if ((cap.flag & (AMDGPU_CAPABILITY_DIRECT_GMA_FLAG | AMDGPU_CAPABILITY_SSG_FLAG)) == 
              (AMDGPU_CAPABILITY_DIRECT_GMA_FLAG | AMDGPU_CAPABILITY_SSG_FLAG)) {
            result = true;
            break;
          }
        }
      }
    }
  }

  if (result) {
    alloc_ops_ = new amd::Monitor("DGMA mem alloc lock", true);
    if (nullptr == alloc_ops_) {
      return true;
    }
  }

  return result;
}

void* ProDevice::AllocDmaBuffer(hsa_agent_t agent, size_t size, void** host_ptr) const
{
  amd::ScopedLock l(alloc_ops_);
  void* ptr = nullptr;
  amdgpu_bo_handle buf_handle = 0;
  amdgpu_bo_alloc_request req = {0};
  *host_ptr = nullptr;

  req.alloc_size = size;
  req.phys_alignment = 64 * Ki;
  req.preferred_heap = AMDGPU_GEM_DOMAIN_DGMA;

  // Allocate buffer in DGMA heap
  if (0 == amdgpu_bo_alloc(dev_handle_, &req, &buf_handle)) {
    amdgpu_bo_handle_type type = amdgpu_bo_handle_type_dma_buf_fd;
    uint32_t shared_handle = 0;
    // Find the base driver handle
    if (0 == amdgpu_bo_export(buf_handle, type, &shared_handle)) {
      uint32_t  flags = 0;
      size_t    buf_size = 0;
      // Map memory object to HSA device
      if (0 == hsa_amd_interop_map_buffer(1, &agent, shared_handle,
                                          flags, &buf_size, &ptr, nullptr, nullptr)) {
        // Ask GPUPro driver to provide CPU access to allocation
        if (0 == amdgpu_bo_cpu_map(buf_handle, host_ptr)) {
          allocs_.insert(std::pair<void*, std::pair<amdgpu_bo_handle, uint32_t>>(
                         ptr, std::pair<amdgpu_bo_handle, uint32_t>(buf_handle, shared_handle)));
        }
        else {
          hsa_amd_interop_unmap_buffer(ptr);
          close(shared_handle);
          amdgpu_bo_free(buf_handle);
        }
      }
      else {
        close(shared_handle);
        amdgpu_bo_free(buf_handle);
      }
    }
    else {
      amdgpu_bo_free(buf_handle);
    }
  }

  return ptr;
}

void ProDevice::FreeDmaBuffer(void* ptr) const
{
  amd::ScopedLock l(alloc_ops_);
  auto it = allocs_.find(ptr);
  if (it != allocs_.end()) {
    amdgpu_bo_cpu_unmap(it->second.first);
    // Unmap memory from HSA device
    hsa_amd_interop_unmap_buffer(ptr);
    // Close shared handle
    close(it->second.second);
    int error = amdgpu_bo_free(it->second.first);
    allocs_.erase(it);
  }
}

}

#endif  // WITHOUT_HSA_BACKEND

