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
