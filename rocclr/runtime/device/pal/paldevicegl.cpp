#include "platform/context.hpp"
#include "device/device.hpp"
#include "platform/runtime.hpp"
#include "platform/agent.hpp"
#ifdef _WIN32
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#endif // _WIN32

#include <GL/gl.h>
#include <GL/glext.h>
#include "CL/cl_gl.h"
#include "paldevice.hpp"
//#include "cwddeci.h"
#include <GL/gl.h>
#include "GL/glATIInternal.h"
#ifdef ATI_OS_LINUX
#include <stdlib.h>
#include <dlfcn.h>
#include "GL/glx.h"
#include "GL/glxext.h"
#include "GL/glXATIPrivate.h"
#else
#include "GL/wglATIPrivate.h"
#endif

#ifdef ATI_OS_LINUX
typedef void* (*PFNGlxGetProcAddress)(const GLubyte* procName);
static PFNGlxGetProcAddress    pfnGlxGetProcAddress=NULL;
static PFNGLXBEGINCLINTEROPAMD glXBeginCLInteropAMD = NULL;
static PFNGLXENDCLINTEROPAMD glXEndCLInteropAMD = NULL;
static PFNGLXRESOURCEATTACHAMD glXResourceAttachAMD = NULL;
static PFNGLXRESOURCEDETACHAMD glxResourceAcquireAMD = NULL;
static PFNGLXRESOURCEDETACHAMD glxResourceReleaseAMD = NULL;
static PFNGLXRESOURCEDETACHAMD glXResourceDetachAMD = NULL;
static PFNGLXGETCONTEXTMVPUINFOAMD glXGetContextMVPUInfoAMD = NULL;
#else
static PFNWGLBEGINCLINTEROPAMD wglBeginCLInteropAMD = NULL;
static PFNWGLENDCLINTEROPAMD wglEndCLInteropAMD = NULL;
static PFNWGLRESOURCEATTACHAMD wglResourceAttachAMD = NULL;
static PFNWGLRESOURCEDETACHAMD wglResourceAcquireAMD = NULL;
static PFNWGLRESOURCEDETACHAMD wglResourceReleaseAMD = NULL;
static PFNWGLRESOURCEDETACHAMD wglResourceDetachAMD = NULL;
static PFNWGLGETCONTEXTGPUINFOAMD wglGetContextGPUInfoAMD = NULL;
#endif

namespace pal {

bool
Device::initGLInteropPrivateExt(void* GLplatformContext, void* GLdeviceContext) const
{
#ifdef ATI_OS_LINUX
    GLXContext ctx = (GLXContext)GLplatformContext;
    void * pModule = dlopen("libGL.so.1",RTLD_NOW);

    if(NULL == pModule) {
        return false;
    }
    pfnGlxGetProcAddress = (PFNGlxGetProcAddress) dlsym(pModule,"glXGetProcAddress");

    if (NULL == pfnGlxGetProcAddress) {
        return false;
    }

    if (!glXBeginCLInteropAMD || !glXEndCLInteropAMD || !glXResourceAttachAMD ||
        !glXResourceDetachAMD || !glXGetContextMVPUInfoAMD) {
        glXBeginCLInteropAMD = (PFNGLXBEGINCLINTEROPAMD) pfnGlxGetProcAddress ((const GLubyte *)"glXBeginCLInteroperabilityAMD");
        glXEndCLInteropAMD = (PFNGLXENDCLINTEROPAMD) pfnGlxGetProcAddress ((const GLubyte *)"glXEndCLInteroperabilityAMD");
        glXResourceAttachAMD = (PFNGLXRESOURCEATTACHAMD) pfnGlxGetProcAddress ((const GLubyte *)"glXResourceAttachAMD");
        glxResourceAcquireAMD = (PFNGLXRESOURCEDETACHAMD) pfnGlxGetProcAddress ((const GLubyte *)"glXResourceAcquireAMD");
        glxResourceReleaseAMD = (PFNGLXRESOURCEDETACHAMD) pfnGlxGetProcAddress ((const GLubyte *)"glXResourceReleaseAMD");
        glXResourceDetachAMD = (PFNGLXRESOURCEDETACHAMD) pfnGlxGetProcAddress ((const GLubyte *)"glXResourceDetachAMD");
        glXGetContextMVPUInfoAMD = (PFNGLXGETCONTEXTMVPUINFOAMD) pfnGlxGetProcAddress ((const GLubyte *)"glXGetContextMVPUInfoAMD");
    }

    if (!glXBeginCLInteropAMD || !glXEndCLInteropAMD || !glXResourceAttachAMD ||
        !glXResourceDetachAMD
#ifndef BRAHMA
        || !glXGetContextMVPUInfoAMD
#endif
        ) {
        return false;
    }
#else
    if (!wglBeginCLInteropAMD || !wglEndCLInteropAMD || !wglResourceAttachAMD ||
        !wglResourceDetachAMD || !wglGetContextGPUInfoAMD) {
        HGLRC fakeRC = NULL;

        if (!wglGetCurrentContext()) {
            fakeRC = wglCreateContext((HDC)GLdeviceContext);
            wglMakeCurrent((HDC)GLdeviceContext, fakeRC);
        }

        wglBeginCLInteropAMD = (PFNWGLBEGINCLINTEROPAMD) wglGetProcAddress ("wglBeginCLInteroperabilityAMD");
        wglEndCLInteropAMD = (PFNWGLENDCLINTEROPAMD) wglGetProcAddress ("wglEndCLInteroperabilityAMD");
        wglResourceAttachAMD = (PFNWGLRESOURCEATTACHAMD) wglGetProcAddress ("wglResourceAttachAMD");
        wglResourceAcquireAMD = (PFNWGLRESOURCEDETACHAMD) wglGetProcAddress ("wglResourceAcquireAMD");
        wglResourceReleaseAMD = (PFNWGLRESOURCEDETACHAMD) wglGetProcAddress ("wglResourceReleaseAMD");
        wglResourceDetachAMD = (PFNWGLRESOURCEDETACHAMD) wglGetProcAddress ("wglResourceDetachAMD");
        wglGetContextGPUInfoAMD = (PFNWGLGETCONTEXTGPUINFOAMD) wglGetProcAddress ("wglGetContextGPUInfoAMD");

        if (fakeRC) {
            wglMakeCurrent(NULL, NULL);
            wglDeleteContext(fakeRC);
        }
    }
    if (!wglBeginCLInteropAMD || !wglEndCLInteropAMD || !wglResourceAttachAMD ||
        !wglResourceDetachAMD || !wglGetContextGPUInfoAMD) {
        return false;
    }
#endif
    return true;
}

bool
Device::glCanInterop(void* GLplatformContext, void* GLdeviceContext) const
{
    bool canInteroperate = false;

#ifdef ATI_OS_WIN
    LUID glAdapterLuid = {0, 0};
    UINT glChainBitMask = 0;
    HGLRC hRC = (HGLRC)GLplatformContext;

    //get GL context's LUID and chainBitMask from UGL
    if (wglGetContextGPUInfoAMD(hRC, &glAdapterLuid, &glChainBitMask)) {
        // match the adapter
        canInteroperate =
            (properties().osProperties.luidHighPart == glAdapterLuid.HighPart) &&
            (properties().osProperties.luidLowPart == glAdapterLuid.LowPart) &&
            ((1 << properties().gpuIndex) == glChainBitMask);
    }
#else
#ifdef BRAHMA
    canInteroperate = true;
#else
    GLuint glDeviceId = 0 ;
    GLuint glChainMask = 0 ;
    GLXContext ctx = (GLXContext)GLplatformContext;
    
    if (glXGetContextMVPUInfoAMD(ctx, &glDeviceId, &glChainMask)) {
        // we allow intoperability only with GL context reside on a single GPU
        canInteroperate =
            (properties().deviceId == glDeviceId) &&
            ((1 << properties().gpuIndex) == glChainBitMask);

        }
    }
#endif
#endif
    return canInteroperate;
}

bool
Device::glAssociate(void* GLplatformContext, void* GLdeviceContext) const
{
    //initialize pointers to the gl extension that supports interoperability
    if (!initGLInteropPrivateExt(GLplatformContext, GLdeviceContext) ||
        !glCanInterop(GLplatformContext, GLdeviceContext)) {
        return false;
    }

    int flags = 0;
/*
    if (m_adp->pAsicInfo->svmFineGrainSystem)
    {
        flags = GL_INTEROP_SVM;
    }
*/
#ifdef ATI_OS_LINUX
    GLXContext ctx = (GLXContext)GLplatformContext;
    return (glXBeginCLInteropAMD(ctx, 0)) ? true : false;
#else
    HGLRC hRC = (HGLRC)GLplatformContext;
    return (wglBeginCLInteropAMD(hRC, flags)) ? true : false;
#endif
}

bool
Device::glDissociate(void* GLplatformContext, void* GLdeviceContext) const
{
    int flags = 0;
/*
    if (m_adp->pAsicInfo->svmFineGrainSystem)
    {
        flags = GL_INTEROP_SVM;
    }
*/
#ifdef ATI_OS_LINUX
    GLXContext ctx = (GLXContext)GLplatformContext;
    return (glXEndCLInteropAMD(ctx, 0)) ? true : false;
#else
    HGLRC hRC = (HGLRC)GLplatformContext;
    return (wglEndCLInteropAMD(hRC, flags)) ? true : false;
#endif
}

bool
Device::resGLAssociate(
    void*   GLContext,
    uint    name,
    uint    type,
    void**  handle,
    void**  mbResHandle,
    size_t* offset) const
{
    amd::ScopedLock lk(lockPAL());

    GLResource hRes = {};
    GLResourceData hData = {};

    bool status = false;

    hRes.type = type;
    hRes.name = name;

    hData.version = GL_RESOURCE_DATA_VERSION;
#ifdef ATI_OS_LINUX
    GLXContext ctx = (GLXContext)GLContext;
    if (glXResourceAttachAMD(ctx, &hRes, &hData)) {
        attribs.dynamicSharedBufferID = hData->sharedBufferID;
        status = true;
    }
#else
    HGLRC hRC = (HGLRC)GLContext;
    if (wglResourceAttachAMD(hRC, &hRes, &hData)) {
        status =  true;
    }
#endif

    if (!status) {
        return false;
    }

    *handle = reinterpret_cast<void*>(hData.handle);
    *mbResHandle = reinterpret_cast<void*>(hData.mbResHandle);
    *offset = static_cast<size_t>(hData.offset);

    return status;
}

bool
Device::resGLAcquire(void* GLplatformContext, void* mbResHandle, uint type) const
{
    amd::ScopedLock lk(lockPAL());

    GLResource hRes = {};
    hRes.mbResHandle = (GLuintp)mbResHandle;
    hRes.type = type;

#ifdef ATI_OS_LINUX
    GLXContext ctx = (GLXContext) GLplatformContext;
    return (glxResourceAcquireAMD(ctx, &hRes)) ? true : false;
#else
    HGLRC hRC = wglGetCurrentContext();
    //! @todo A temporary workaround for MT issue in conformance fence_sync
    if (0 == hRC) {
        return true;
    }
    return (wglResourceAcquireAMD(hRC, &hRes)) ? true : false;
#endif
}

bool
Device::resGLRelease(void* GLplatformContext, void* mbResHandle, uint type) const
{
    amd::ScopedLock lk(lockPAL());

    GLResource hRes = {};
    hRes.mbResHandle = (GLuintp)mbResHandle;
    hRes.type = type;
#ifdef ATI_OS_LINUX
    //TODO : make sure the application GL context is current. if not no
    // point calling into the GL RT.
    GLXContext ctx = (GLXContext) GLplatformContext;
    return (glxResourceReleaseAMD(ctx, &hRes)) ? true : false;
#else
    // Make the call into the GL driver only if the application GL context is current
    HGLRC hRC = wglGetCurrentContext();
    //! @todo A temporary workaround for MT issue in conformance fence_sync
    if (0 == hRC) {
        return true;
    }
    return (wglResourceReleaseAMD(hRC, &hRes)) ? true : false;
#endif
}

bool
Device::resGLFree(void* GLplatformContext, void* mbResHandle, uint type) const
{
    amd::ScopedLock lk(lockPAL());

    GLResource hRes = {};
    hRes.mbResHandle = (GLuintp)mbResHandle;
    hRes.type = type;
#ifdef ATI_OS_LINUX
    GLXContext ctx = (GLXContext)GLplatformContext;
    return (glXResourceDetachAMD(ctx, &hRes)) ? true : false;
#else
    HGLRC hRC = (HGLRC)GLplatformContext;
    return (wglResourceDetachAMD(hRC, &hRes)) ? true : false;
#endif
}

} // pal
