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

#include <d3d9.h>
#include <dxgi.h>

/**************************************************************************************************************
* Note: ideally the DXX extension interfaces should be mapped from the DXX perforce branch. 
* This means CAL client spec will need to change to include headers directly from the DXX perforce tree. 
* However, CAL only cares about the DXX OpenCL extension interface class. The spec cannot change
* without notification. So it is safe to use a local copy of the relevant DXX extension interface classes.
**************************************************************************************************************/
#include "DxxOpenCLInteropExt.h"


bool
CALGSLDevice::associateD3D9Device(void* d3d9Device)
{
    bool canInteroperate = false;
    D3DCAPS9 pCaps;
    LUID calDevAdapterLuid = {0, 0};
    UINT calDevChainBitMask = 0;
    IDirect3D9* p3d9dev;
    LUID d3d9deviceLuid = {0, 0};

    IDirect3DDevice9* pd3d9Device = static_cast<IDirect3DDevice9*>(d3d9Device);

    // Get D3D9 Device caps
    pd3d9Device->GetDeviceCaps(&pCaps);
    // Get 3D9 Device
    pd3d9Device->GetDirect3D(&p3d9dev);

    IDirect3D9Ex* p3d9devEx = static_cast<IDirect3D9Ex*>(p3d9dev);
    p3d9devEx->GetAdapterLUID(pCaps.AdapterOrdinal, &d3d9deviceLuid);
    p3d9dev->Release();

    // match the adapter
    if (m_adp->getMVPUinfo(&calDevAdapterLuid, &calDevChainBitMask))
    {
        canInteroperate = ((calDevAdapterLuid.HighPart == d3d9deviceLuid.HighPart) &&
                           (calDevAdapterLuid.LowPart == d3d9deviceLuid.LowPart));
    }

    return canInteroperate;
}

#else // !ATI_OS_WIN

bool
CALGSLDevice::associateD3D9Device(void* d3dDevice)
{
    return false;
}

#endif // !ATI_OS_WIN
