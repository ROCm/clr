/* ============================================================

Copyright (c) 2010 Advanced Micro Devices, Inc.  All rights reserved.

Redistribution and use of this material is permitted under the following
conditions:

Redistributions must retain the above copyright notice and all terms of this
license.

In no event shall anyone redistributing or accessing or using this material
commence or participate in any arbitration or legal action relating to this
material against Advanced Micro Devices, Inc. or any copyright holders or
contributors. The foregoing shall survive any expiration or termination of
this license or any agreement or access or use related to this material.

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERATION, OR THAT IT IS FREE
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT.
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES,
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES,
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S.
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS,
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS,
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS.
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to
computer software and technical data, respectively. Use, duplication,
distribution or disclosure by the U.S. Government and/or DOD agencies is
subject to the full extent of restrictions in all applicable regulations,
including those found at FAR52.227 and DFARS252.227 et seq. and any successor
regulations thereof. Use of this material by the U.S. Government and/or DOD
agencies is acknowledgment of the proprietary rights of any copyright holders
and contributors, including those of Advanced Micro Devices, Inc., as well as
the provisions of FAR52.227-14 through 23 regarding privately developed and/or
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and
supersedes all proposals and prior discussions and writings between the parties
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be
modified or waived, and no breach of this license can be excused, unless done
so in a writing signed by all affected parties. Each term of this license is
separately enforceable. If any term of this license is determined to be or
becomes unenforceable or illegal, such term shall be reformed to the minimum
extent necessary in order for this license to remain in effect in accordance
with its terms as modified by such reformation. This license shall be governed
by and construed in accordance with the laws of the State of Texas without
regard to rules on conflicts of law of any state or jurisdiction or the United
Nations Convention on the International Sale of Goods. All disputes arising out
of this license shall be subject to the jurisdiction of the federal and state
courts in Austin, Texas, and all defenses are hereby waived concerning personal
jurisdiction and venue of these courts.

============================================================ */

/* $Revision$ on $Date$ */

#ifndef __OPENCL_CL_D3D9_AMD_H
#define __OPENCL_CL_D3D9_AMD_H

#include "CL/cl_dx9_media_sharing.h"
#include <d3d9.h>
#include "platform/context.hpp"
#include "platform/memory.hpp"

#include <utility>

/* cl_amd_d3d9_sharing extension    */
#define cl_amd_d3d9_sharing 1

/* cl_amd_d3d9_sharing error codes */
#define CL_INVALID_D3D9_DEVICE_KHR              -1021
#define CL_INVALID_D3D9_RESOURCE_KHR            -1022

/* cl_amd_d3d9_sharing enumerations */
#define CL_CONTEXT_D3D9_DEVICE_KHR              0x4039

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDsFromDX9MediaAdapterKHR(
    cl_platform_id,
    cl_uint,
    cl_dx9_media_adapter_type_khr *,
    void *,
    cl_dx9_media_adapter_set_khr,
    cl_uint,
    cl_device_id *,
    cl_uint *);

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateFromDX9MediaSurfaceKHR(
    cl_context,
    cl_mem_flags,
    cl_dx9_media_adapter_type_khr,
    void *,
    cl_uint,                                               
    cl_int *);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAcquireDX9MediaSurfacesKHR(
    cl_command_queue,
    cl_uint,
    const cl_mem *,
    cl_uint,
    const cl_event *,
    cl_event *);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReleaseDX9MediaSurfacesKHR(
    cl_command_queue,
    cl_uint,
    const cl_mem *,
    cl_uint,
    const cl_event *,
    cl_event *);

namespace amd
{
typedef struct
{
    union
    {
        UINT ByteWidth;
        UINT Width;
    };
    UINT Height;
    UINT Depth;
} D3D9ObjSize_t;

typedef struct
{
    D3D9ObjSize_t       objSize_;
    D3DFORMAT           d3dFormat_;
    D3DRESOURCETYPE     resType_;
    UINT                usage_;
    D3DPOOL             d3dPool_;
    D3DMULTISAMPLE_TYPE msType_;
    UINT                msQuality_;
    UINT                mipLevels_;
    UINT                fvf_;
    RECT                surfRect_;
} D3D9ObjDesc_t;

typedef struct d3d9ResInfo {
    cl_dx9_surface_info_khr surfInfo;
    cl_uint                 surfPlane;
} TD3D9RESINFO;


//typedef std::pair<cl_dx9_surface_info_khr, D3D9Object*> TD3D9OBJINFO;

//! Class D3D9Object keeps all the info about the D3D9 object
//! from which the CL object is created
class D3D9Object : public InteropObject
{
private:
    IDirect3DSurface9* pD3D9Aux_;
    cl_int  cliChecksum_;
    bool releaseResources_;
    static bool createSharedResource(D3D9Object& obj);
    static std::vector<std::pair<TD3D9RESINFO, TD3D9RESINFO>> resources_;

    //!Global lock
    static Monitor              resLock_;
    cl_uint                     surfPlane_;
    cl_dx9_surface_info_khr     surfInfo_;

protected:
    IDirect3DSurface9*  pD3D9Res_;
    IDirect3DSurface9*  pD3D9ResOrig_;
    IDirect3DQuery9*    pQuery_;
    D3D9ObjDesc_t       objDesc_;
    D3D9ObjDesc_t       objDescOrig_;
    HANDLE              handleOrig_;
    HANDLE              handleShared_;
    RECT                srcSurfRect;
    RECT                SharedSurfRect;
    cl_dx9_media_adapter_type_khr adapterType_;

public:
//! D3D9Object constructor initializes memeber variables
    D3D9Object()
        : releaseResources_(false),
        pQuery_(NULL)
    {
        // @todo Incorrect initialization!!!
        memset(this, 0, sizeof(D3D9Object));
    }
    //copy constructor
    D3D9Object(D3D9Object& d3d9obj)
        :pQuery_(NULL)
    {
        *this = d3d9obj;
        this->releaseResources_ = true;
    }

    //virtual destructor
    virtual ~D3D9Object()
    {
        ScopedLock sl(resLock_);
        if(releaseResources_) {
            if(pD3D9ResOrig_) pD3D9ResOrig_->Release();
            if(pD3D9Res_) pD3D9Res_->Release();
            if(pD3D9Aux_) pD3D9Aux_->Release();
            if(pQuery_) pQuery_->Release();
            //if the resouce is being used
            if(resources_.size()) {
                for(auto& it = resources_.cbegin(); it != resources_.cend(); it++) {
                    if( surfInfo_.resource && 
                        ((*it).first.surfInfo.resource == surfInfo_.resource) &&
                        ((*it).first.surfPlane == surfPlane_)) {
                            resources_.erase(it);
                            break;
                    }
                }
            }
        }
    }
    static int initD3D9Object(const Context& amdContext, cl_dx9_media_adapter_type_khr adapter_type, 
        cl_dx9_surface_info_khr* cl_surf_info, cl_uint plane, D3D9Object& obj);
    cl_uint getMiscFlag(void);

    D3D9Object* asD3D9Object() {return this;}
    IDirect3DSurface9* getD3D9Resource() const {return pD3D9Res_;}
    HANDLE getD3D9SharedHandle() const {return handleShared_;}
    IDirect3DSurface9* getD3D9ResOrig() const {return pD3D9ResOrig_;}
    RECT* getSrcSurfRect() {return &objDesc_.surfRect_;}
    RECT* getSharedSurfRect() {return &objDescOrig_.surfRect_;}
    void setD3D9AuxRes(IDirect3DSurface9* pAux) {pD3D9Aux_ = pAux;}
    IDirect3DSurface9* getD3D9AuxRes() {return pD3D9Aux_;}
    IDirect3DQuery9* getQuery() const {return pQuery_;}
    Monitor & getResLock() { return resLock_;}
    UINT getWidth() const {return objDesc_.objSize_.Width;}
    UINT getHeight() const {return objDesc_.objSize_.Height;}
    cl_uint getPlane() const {return surfPlane_;}
    cl_dx9_media_adapter_type_khr getAdapterType() const { return adapterType_;};
    const cl_dx9_surface_info_khr& getSurfInfo() const {return surfInfo_;};
    size_t getElementBytes(D3DFORMAT d3d9Format, cl_uint plane);
    size_t getElementBytes() {return getElementBytes(objDesc_.d3dFormat_, surfPlane_);}
    D3DFORMAT getD3D9Format() {return objDesc_.d3dFormat_;}
    D3D9ObjDesc_t* getObjDesc() {return &objDesc_;}
    cl_image_format getCLFormatFromD3D9();
    cl_image_format getCLFormatFromD3D9(D3DFORMAT d3d9Fmt, cl_uint plane);
    // On acquire copy data from original resource to shared resource
    virtual bool copyOrigToShared();
    // On release copy data from shared copy to the original resource
    virtual bool copySharedToOrig();
};

class Image2DD3D9 : public D3D9Object , public Image
{
protected:
    //! Initializes the device memory array which is nested
    // after'Image2DD3D9' object in memory layout.
    virtual void initDeviceMemory();
public:
//! Image2DD3D9 constructor just calls constructors of base classes
//! to pass down the parameters
    Image2DD3D9(
        Context&            amdContext,
        cl_mem_flags        clFlags,
        D3D9Object&         d3d9obj)
        : // Call base classes constructors
        D3D9Object(d3d9obj),
        Image(
            amdContext,
            CL_MEM_OBJECT_IMAGE2D,
            clFlags,
            d3d9obj.getCLFormatFromD3D9(),
            d3d9obj.getWidth(),
            d3d9obj.getHeight(),
            1,
            d3d9obj.getWidth() * d3d9obj.getElementBytes(), //rowPitch),
            0)
        {
            setInteropObj(this);
        }
    virtual ~Image2DD3D9() {}
};

cl_mem clCreateImage2DFromD3D9ResourceAMD(
    Context&        amdContext,
    cl_mem_flags    flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    cl_dx9_surface_info_khr*  surface_info,
    cl_uint         plane,
    int*            errcode_ret);

void SyncD3D9Objects(std::vector<amd::Memory*>& memObjects);

} //namespace amd

#endif  /* __OPENCL_CL_D3D9_AMD_H   */
