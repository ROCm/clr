 /* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#include "gsl_ctx.h"
#include "GSLDevice.h"

#if defined(ATI_OS_WIN)

#include <D3D10_1.h>

/**************************************************************************************************************
* Note: ideally the DXX extension interfaces should be mapped from the DXX perforce branch.
* This means CAL client spec will need to change to include headers directly from the DXX perforce tree.
* However, CAL only cares about the DXX OpenCL extension interface class. The spec cannot change
* without notification. So it is safe to use a local copy of the relevant DXX extension interface classes.
**************************************************************************************************************/
#include "DxxOpenCLInteropExt.h"

static bool
queryD3D10DeviceGPUMask(ID3D10Device* pd3d10Device, UINT* pd3d10DeviceGPUMask)
{
    HMODULE             hDLL = NULL;
    IAmdDxExt*          pExt = NULL;
    IAmdDxExtCLInterop* pCLExt = NULL;
    PFNAmdDxExtCreate   AmdDxExtCreate;
    HRESULT             hr = S_OK;

    // Get a handle to the DXX DLL with extension API support
#if defined _WIN64
    static const CHAR dxxModuleName[13] = "atidxx64.dll";
#else
    static const CHAR dxxModuleName[13] = "atidxx32.dll";
#endif

    hDLL = GetModuleHandle(dxxModuleName);

    if (hDLL == NULL)
    {
        hr = E_FAIL;
    }

    // Get the exported AmdDxExtCreate() function pointer
    if (SUCCEEDED(hr))
    {
        AmdDxExtCreate = reinterpret_cast<PFNAmdDxExtCreate>(GetProcAddress(hDLL, "AmdDxExtCreate"));
        if (AmdDxExtCreate == NULL)
        {
            hr = E_FAIL;
        }
    }

    // Create the extension object
    if (SUCCEEDED(hr))
    {
        hr = AmdDxExtCreate(pd3d10Device, &pExt);
    }

    // Get the extension version information
    if (SUCCEEDED(hr))
    {
        AmdDxExtVersion extVersion;
        hr = pExt->GetVersion(&extVersion);

        if (extVersion.majorVersion == 0)
        {
            hr = E_FAIL;
        }
    }

    // Get the OpenCL Interop interface
    if (SUCCEEDED(hr))
    {
        pCLExt = static_cast<IAmdDxExtCLInterop*>(pExt->GetExtInterface(AmdDxExtCLInteropID));
        if (pCLExt != NULL)
        {
            // Get the GPU mask using the CL Interop extension.
            pCLExt->QueryInteropGpuMask(pd3d10DeviceGPUMask);
        }
        else
        {
            hr = E_FAIL;
        }
    }

    if (pCLExt != NULL)
    {
        pCLExt->Release();
    }

    if (pExt != NULL)
    {
        pExt->Release();
    }

    return (SUCCEEDED(hr));
}

bool
CALGSLDevice::associateD3D10Device(void* d3d10Device)
{
    bool canInteroperate = false;

    LUID calDevAdapterLuid = {0, 0};
    UINT calDevChainBitMask = 0;
    UINT d3d10DeviceGPUMask = 0;

    ID3D10Device* pd3d10Device = static_cast<ID3D10Device*>(d3d10Device);

    IDXGIDevice* pDXGIDevice;
    pd3d10Device->QueryInterface(__uuidof(IDXGIDevice), (void **)&pDXGIDevice);

    IDXGIAdapter* pDXGIAdapter;
    pDXGIDevice->GetAdapter(&pDXGIAdapter);

    DXGI_ADAPTER_DESC adapterDesc;
    pDXGIAdapter->GetDesc(&adapterDesc);

    // match the adapter
    if (m_adp->getMVPUinfo(&calDevAdapterLuid, &calDevChainBitMask))
    {
        canInteroperate = ((calDevAdapterLuid.HighPart == adapterDesc.AdapterLuid.HighPart) &&
                           (calDevAdapterLuid.LowPart == adapterDesc.AdapterLuid.LowPart));
    }

    // match the chain ID
    if (canInteroperate)
    {
        if (queryD3D10DeviceGPUMask(pd3d10Device, &d3d10DeviceGPUMask))
        {
            canInteroperate = (calDevChainBitMask & d3d10DeviceGPUMask) != 0;
        }
        else
        {
            // special handling for Intel iGPU + AMD dGPU in LDA mode (only occurs on a PX platform) where
            // the D3D10Device object is created on the Intel iGPU and passed to AMD dGPU (secondary) to interoperate.
            if (calDevChainBitMask > 1)
            {
                canInteroperate = false;
            }
        }
    }

    pDXGIDevice->Release();
    pDXGIAdapter->Release();

    return canInteroperate;
}

gslMemObject
CALGSLDevice::resMapD3DResource(const CALresourceDesc* desc, uint64 sharedhandle, bool displayable) const
{
    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps_);

    gslMemObject mem = NULL;

    gslMemObjectAttribs attribs(
        GSL_MOA_TEXTURE_2D,      // type
        GSL_MOA_MEMORY_ALIAS,    // location
        GSL_MOA_TILING_TILED,    // tiling
        GSL_MOA_DISPLAYABLE_NO,  // displayable
        ATIGL_FALSE,             // mipmap
        1,                       // samples
        0,                       // cpu_address
        GSL_MOA_SIGNED_NO,       // signed_format
        GSL_MOA_FORMAT_DERIVED,  // numFormat
        DRIVER_MODULE_GLL,       // module
        GSL_ALLOCATION_INSTANCED // alloc_type
    );

    HANDLE h = (HANDLE)sharedhandle;
    attribs.cpu_address = h;
    attribs.alias_swizzle = 0;
    attribs.channelOrder = desc->channelOrder;
    attribs.type = desc->dimension;

    switch (desc->dimension)
    {
        case GSL_MOA_BUFFER:
            attribs.tiling = GSL_MOA_TILING_LINEAR;
            mem = m_cs->createMemObject1D(desc->format, desc->size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_1D:
            attribs.tiling = GSL_MOA_TILING_LINEAR;
            mem = m_cs->createMemObject1D(desc->format, desc->size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_2D:
        {
            uint32 height = (uint32)desc->size.height;
            if (displayable)
            {
                attribs.displayable = GSL_MOA_DISPLAYABLE_YES;
            }
            mem = m_cs->createMemObject2D(desc->format, desc->size.width, height, &attribs);
        }
            break;
        case GSL_MOA_TEXTURE_3D:
            mem = m_cs->createMemObject3D(desc->format, desc->size.width,
                (uint32)desc->size.height, (uint32)desc->size.depth, &attribs);
            break;
        case GSL_MOA_TEXTURE_BUFFER:
            attribs.type = GSL_MOA_TEXTURE_BUFFER;
            mem = m_cs->createMemObject1D(desc->format, desc->size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_1D_ARRAY:
            mem = m_cs->createMemObject3D(desc->format, desc->size.width,
                1, (uint32)desc->size.height, &attribs);
            break;
        case GSL_MOA_TEXTURE_2D_ARRAY:
            mem = m_cs->createMemObject3D(desc->format, desc->size.width,
                (uint32)desc->size.height, (uint32)desc->size.depth, &attribs);
            break;
        default:
            break;
    }

    return mem;
}

#else // !ATI_OS_WIN

bool
CALGSLDevice::associateD3D10Device(void* d3d10Device)
{
    return false;
}

gslMemObject
CALGSLDevice::resMapD3DResource(const CALresourceDesc* desc, uint64 sharedhandle, bool displayable) const
{
    return 0;
}

#endif // !ATI_OS_WIN
