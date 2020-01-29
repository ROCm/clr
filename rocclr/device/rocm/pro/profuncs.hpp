//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

namespace roc
{
namespace drm
{
typedef int (*DrmGetDevices)(
            drmDevicePtr*     pDevices,
            int               maxDevices);

typedef int (*AmdgpuDeviceInitialize)(
            int                       fd,
            uint32_t*                 pMajorVersion,
            uint32_t*                 pMinorVersion,
            amdgpu_device_handle*     pDeviceHandle);

typedef int (*AmdgpuDeviceDeinitialize)(
            amdgpu_device_handle  hDevice);

typedef int (*AmdgpuQueryGpuInfo)(
            amdgpu_device_handle      hDevice,
            struct amdgpu_gpu_info*   pInfo);

typedef int (*AmdgpuQueryInfo)(
            amdgpu_device_handle  hDevice,
            unsigned              infoId,
            unsigned              size,
            void*                 pValue);

typedef int (*AmdgpuBoAlloc)(
            amdgpu_device_handle              hDevice,
            struct amdgpu_bo_alloc_request*   pAllocBuffer,
            amdgpu_bo_handle*                 pBufferHandle);

typedef int (*AmdgpuBoExport)(
            amdgpu_bo_handle              hBuffer,
            enum amdgpu_bo_handle_type    type,
            uint32_t*                     pFd);

typedef int (*AmdgpuBoFree)(
            amdgpu_bo_handle  hBuffer);

typedef int (*AmdgpuBoCpuMap)(
            amdgpu_bo_handle  hBuffer,
            void**            ppCpuAddress);

typedef int (*AmdgpuBoCpuUnmap)(
            amdgpu_bo_handle  hBuffer);

struct Funcs
{
    DrmGetDevices                     DrmGetDevices;
    AmdgpuDeviceInitialize            AmdgpuDeviceInitialize;
    AmdgpuDeviceDeinitialize          AmdgpuDeviceDeinitialize;
    AmdgpuQueryGpuInfo                AmdgpuQueryGpuInfo;
    AmdgpuQueryInfo                   AmdgpuQueryInfo;
    AmdgpuBoAlloc                     AmdgpuBoAlloc;
    AmdgpuBoExport                    AmdgpuBoExport;
    AmdgpuBoFree                      AmdgpuBoFree;
    AmdgpuBoCpuMap                    AmdgpuBoCpuMap;
    AmdgpuBoCpuUnmap                  AmdgpuBoCpuUnmap;
};

} //namespace drm
} //namespace roc
