/* Copyright (c) 2017-present Advanced Micro Devices, Inc.

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

#ifndef WITHOUT_HSA_BACKEND

#include "hsa_ext_amd.h"
#include "lnxheaders.h"
#include "prodevice.hpp"
#include "amdgpu_drm.h"

namespace roc {

constexpr uint32_t kMaxDevices  = 32;
constexpr uint32_t kAtiVendorId = 0x1002;

void*      ProDevice::lib_drm_handle_ = nullptr;
bool       ProDevice::initialized_ = false;
drm::Funcs ProDevice::funcs_;

IProDevice* IProDevice::Init(uint32_t bus, uint32_t dev, uint32_t func)
{
  // Make sure DRM lib is initialized
  if (!ProDevice::DrmInit()) {
    return nullptr;
  }

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
    Funcs().AmdgpuDeviceDeinitialize(dev_handle_);
  }
  if (file_desc_ > 0) {
    close(file_desc_);
  }
}

bool ProDevice::DrmInit()
{
  if (initialized_ == false) {
    // Find symbols in libdrm_amdgpu.so.1
    lib_drm_handle_ = dlopen("libdrm_amdgpu.so.1", RTLD_NOW);
    if (lib_drm_handle_ == nullptr) {
      return false;
    } else {
      funcs_.DrmGetDevices = reinterpret_cast<drm::DrmGetDevices>(dlsym(
                             lib_drm_handle_,
                             "drmGetDevices"));
      if (funcs_.DrmGetDevices == nullptr) return false;
      funcs_.AmdgpuDeviceInitialize = reinterpret_cast<drm::AmdgpuDeviceInitialize>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_device_initialize"));
      if (funcs_.AmdgpuDeviceInitialize == nullptr) return false;
      funcs_.AmdgpuDeviceDeinitialize = reinterpret_cast<drm::AmdgpuDeviceDeinitialize>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_device_deinitialize"));
      if (funcs_.AmdgpuDeviceDeinitialize == nullptr) return false;
      funcs_.AmdgpuQueryGpuInfo = reinterpret_cast<drm::AmdgpuQueryGpuInfo>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_query_gpu_info"));
      if (funcs_.AmdgpuQueryGpuInfo == nullptr) return false;
      funcs_.AmdgpuQueryInfo = reinterpret_cast<drm::AmdgpuQueryInfo>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_query_info"));
      if (funcs_.AmdgpuQueryInfo == nullptr) return false;
      funcs_.AmdgpuBoAlloc = reinterpret_cast<drm::AmdgpuBoAlloc>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_bo_alloc"));
      if (funcs_.AmdgpuBoAlloc == nullptr) return false;
      funcs_.AmdgpuBoExport = reinterpret_cast<drm::AmdgpuBoExport>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_bo_export"));
      if (funcs_.AmdgpuBoExport == nullptr) return false;
      funcs_.AmdgpuBoFree = reinterpret_cast<drm::AmdgpuBoFree>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_bo_free"));
      if (funcs_.AmdgpuBoFree == nullptr) return false;
      funcs_.AmdgpuBoCpuMap = reinterpret_cast<drm::AmdgpuBoCpuMap>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_bo_cpu_map"));
      if (funcs_.AmdgpuBoCpuMap == nullptr) return false;
      funcs_.AmdgpuBoCpuUnmap = reinterpret_cast<drm::AmdgpuBoCpuUnmap>(dlsym(
                             lib_drm_handle_,
                             "amdgpu_bo_cpu_unmap"));
      if (funcs_.AmdgpuBoCpuUnmap == nullptr) return false;
    }
  }

  initialized_ = true;
  return true;
}

#ifndef AMDGPU_CAPABILITY_SSG_FLAG
#define AMDGPU_CAPABILITY_SSG_FLAG 4
#endif

// ================================================================================================
// Open drm device and initialize it. And also get the drm information.
bool ProDevice::Create(uint32_t bus, uint32_t device, uint32_t func) {
  drmDevicePtr  devices[kMaxDevices] = { };
  int32_t device_count = Funcs().DrmGetDevices(devices, kMaxDevices);
  bool    result = false;

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
        if (Funcs().AmdgpuDeviceInitialize(file_desc_, &major_ver_,
                                           &minor_ver_, &dev_handle_) == 0) {
          uint32_t version = 0;
          // amdgpu_query_gpu_info will never fail only if it is initialized
          Funcs().AmdgpuQueryGpuInfo(dev_handle_, &gpu_info_);

          drm_amdgpu_capability cap = {};
          Funcs().AmdgpuQueryInfo(dev_handle_, AMDGPU_INFO_CAPABILITY, sizeof(drm_amdgpu_capability), &cap);

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
  if (0 == Funcs().AmdgpuBoAlloc(dev_handle_, &req, &buf_handle)) {
    amdgpu_bo_handle_type type = amdgpu_bo_handle_type_dma_buf_fd;
    uint32_t shared_handle = 0;
    // Find the base driver handle
    if (0 == Funcs().AmdgpuBoExport(buf_handle, type, &shared_handle)) {
      uint32_t  flags = 0;
      size_t    buf_size = 0;
      // Map memory object to HSA device
      if (0 == hsa_amd_interop_map_buffer(1, &agent, shared_handle,
                                          flags, &buf_size, &ptr, nullptr, nullptr)) {
        // Ask GPUPro driver to provide CPU access to allocation
        if (0 == Funcs().AmdgpuBoCpuMap(buf_handle, host_ptr)) {
          allocs_.insert({ptr, {buf_handle, shared_handle}});
        }
        else {
          hsa_amd_interop_unmap_buffer(ptr);
          close(shared_handle);
          Funcs().AmdgpuBoFree(buf_handle);
        }
      }
      else {
        close(shared_handle);
        Funcs().AmdgpuBoFree(buf_handle);
      }
    }
    else {
      Funcs().AmdgpuBoFree(buf_handle);
    }
  }

  return ptr;
}

void ProDevice::FreeDmaBuffer(void* ptr) const
{
  amd::ScopedLock l(alloc_ops_);
  auto it = allocs_.find(ptr);
  if (it != allocs_.end()) {
    Funcs().AmdgpuBoCpuUnmap(it->second.first);
    // Unmap memory from HSA device
    hsa_amd_interop_unmap_buffer(ptr);
    // Close shared handle
    close(it->second.second);
    int error = Funcs().AmdgpuBoFree(it->second.first);
    allocs_.erase(it);
  }
}

}

#endif  // WITHOUT_HSA_BACKEND

