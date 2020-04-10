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

#ifndef __GSLDevice_h__
#define __GSLDevice_h__

#include "backend.h"
#include "atitypes.h"
#include "gsl_types.h"
#include "gsl_config.h"
#include "thread/monitor.hpp"
#include "gsl_types_internal.h"

#ifdef ATI_OS_LINUX
typedef unsigned int IDirect3DDevice9;
typedef unsigned int IDirect3DSurface9;
typedef unsigned int IDirect3DQuery9;
typedef unsigned int RECT;
#else
#undef APIENTRY
#include <d3d9.h>
#endif

#include <map>

namespace gsl
{
    class gsAdaptor;
};

typedef enum
{
    USE_NONE,
    USE_CPDMA,
    USE_DRMDMA,
    USE_DRMDMA_L2T,
    USE_DRMDMA_T2L,
} CopyType;

class CALGSLDevice
{
public:
    struct GLResAssociate {
        void*   GLContext;          //(IN) handle to HGLRC or GLXContext
        void*   GLdeviceContext;    //(IN) a handle to device context
        uint    name;               //(IN) gl identifier of the object
        uint    type;               // (IN) type of the interop object .
        uint    flags;              // (IN) flags assigned  to 'GLResource' struct
        void*   mbResHandle;        // (OUT) Internal GL driver handle for the resource
        gslMemObject  mem_base;     // (OUT) Base memory object for the resource
        gslMemObject  memObject;    //(OUT)  Alias gsl memory object for the resource
        gslMemObject  fMaskObject;  //(OUT) gsl memobject of the an MSAA resource F-mask.
    };

    struct OpenParams {
        bool        enableHighPerformanceState;
        bool        reportAsOCL12Device;
        const char* sclkThreshold;
        const char* downHysteresis;
        const char* upHysteresis;
        const char* powerLimit;
        const char* mclkThreshold;
        const char* mclkUpHyst;
        const char* mclkDownHyst;
    };

    CALGSLDevice();
    ~CALGSLDevice();

    bool open(uint32 gpuIndex, OpenParams& openData);
    void close();

    gslMemObject     resAlloc(const CALresourceDesc* desc) const;
    void*            resMapLocal(size_t& pitch, gslMemObject res, gslMapAccessType flags);
    void             resUnmapLocal(gslMemObject res);

    void             resFree(gslMemObject mem) const;
    void*            resMapRemote(size_t& pitch, gslMemObject res, gslMapAccessType flags) const;
    void             resUnmapRemote(gslMemObject res) const;

    gslMemObject     resGetHeap(size_t size) const;
    gslMemObject     resAllocView(gslMemObject res, gslResource3D size,
        size_t offset, cmSurfFmt format, gslChannelOrder channelOrder,
        gslMemObjectAttribType resType, uint32 level, uint32 layer,
        uint32 flags, uint64 bytePitch = (uint64)-1) const;

    bool             associateD3D11Device(void* d3d11Device); //void* is of type ID3D11Device*
    bool             associateD3D10Device(void* d3d10Device); //void* is of type ID3D10Device*
    bool             associateD3D9Device(void* d3d9Device); //void* is of type IDirect3DDevice9*

    gslMemObject     resMapD3DResource(
        const CALresourceDesc* desc, uint64 sharedhandle, bool displayable) const;

    bool             glAssociate(CALvoid *GLplatformContext, CALvoid* GLdeviceContext);
    bool             glDissociate(CALvoid *GLplatformContext, CALvoid* GLdeviceContext);
    //! @brief This function is called once for every interop resource on the first clEnqeueuAcquireGL.
    bool             resGLAssociate(GLResAssociate & resData) const;
    //! @brief This function is called once for every interop resource on resource destruction.
    bool             resGLFree (CALvoid* GLplatformContext,
        CALvoid* GLdeviceContext, gslMemObject mem, gslMemObject mem_base,
        CALvoid* mbResHandle, CALuint type) const;
    //! @brief Decompresses depth/MSAA surfaces.This function is called on every 'clEnqeueuAcquireGLObject'.
    bool             resGLAcquire( CALvoid* GLplatformContext, CALvoid* mbResHandle, CALuint type) const;
    //! @brief This function is called on every 'clEnqeueuReleaseGLObject'.
    bool             resGLRelease(CALvoid* GLplatformContext, CALvoid* mbResHandle, CALuint type) const;

    gsl::gsAdaptor*  getNative() const;
    CALuint          getElfMachine() const { return m_elfmachine; };
    uint32           getGpuIndex() const { return m_gpuIndex; };

    uint32           getMaxTextureSize() const;
    const CALdeviceattribs& getAttribs() const { return m_attribs; }
    const gslMemInfo& getMemInfo() const { return m_memInfo; }

    uint32           getVPUMask() const { return m_vpuMask; }
    bool             canDMA() const { return m_canDMA; }
    gslMemObject     m_srcDRMDMAMem, m_dstDRMDMAMem;    // memory object of flush buffer, used for DRMDMA flush

    void             PerformAdapterInitialization(bool ValidateOnly);
    void             PerformFullInitialization() const;
    void             CloseInitializedAdapter(bool ValidateOnly);

    CopyType         GetCopyType(gslMemObject srcMem, gslMemObject destMem, size_t* srcOffset,
                                     size_t* destOffset, bool allowDMA, uint32 flags, size_t size, bool enableCopyRect) const;

    uint64           calcScratchBufferSize(uint32 regNum) const;

    amd::Monitor& gslDeviceOps() const { return *gslDeviceOps_; }

    void fillImageHwState(gslMemObject mem, void* hwState, uint32 hwStateSize) const;

    void fillSamplerHwState(bool unnorm, uint32 min, uint32 mag, uint32 addr,
        float minLod, float maxLod, void* hwState, uint32 hwStateSize) const;

    gslSamplerObject txSampler() const { return m_textureSampler; }

    void convertInputChannelOrder(intp *channelOrder) const;

    gsl::gsCtx* gslCtx() const { return m_cs; }

    bool isComputeRingIDForced() const { return m_isComputeRingIDForced; }
    gslEngineID getforcedComputeEngineID() const { return m_forcedComputeEngineID; }

    gslEngineID getFirstAvailableComputeEngineID() const { return static_cast<gslEngineID>(
                                                           m_adp->findFirstAvailableComputeEngineID()); }

    virtual bool gslSetClockMode(GSLClockModeInfo * clockModeInfo);
protected:
    //
    /// channel order enumerants
    //
    //channelSwizzleMode and channelSwizzle match the hwl equivalent hwtxSwizzleMode and hwtxUnitSwizzle in hwl_tx_if.h.
    enum channelSwizzleMode {
        SWIZZLE_COMPONENT0,                          ///< Select Component0
        SWIZZLE_COMPONENT1,                          ///< Select Component1
        SWIZZLE_COMPONENT2,                          ///< Select Component2
        SWIZZLE_COMPONENT3,                          ///< Select Component3
        SWIZZLE_ZERO,                                ///< Select Zero
        SWIZZLE_ONE,                                 ///< Select One
    };

    //
    /// channel order swizzle type
    //
    typedef struct channelSwizzleRec
    {
        channelSwizzleMode r : 8;  ///< Red channel of texture
        channelSwizzleMode g : 8;  ///< Green channel of texture
        channelSwizzleMode b : 8;  ///< Blue channel of texture
        channelSwizzleMode a : 8;  ///< Alpha channel of texture
    } channelSwizzle;

    uint                m_nEngines;
    gslEngineDescriptor m_engines[GSL_ENGINEID_MAX];

private:
    gsl::gsAdaptor*  m_adp;
    gsl::gsCtx*      m_cs;
    gslRenderState   m_rs;
    CALtarget        m_target;
    CALuint          m_elfmachine;
    uint32           m_vpuMask;
    uint32           m_chainIndex;
    int32            m_maxtexturesize;
    uint32           m_gpuIndex;
    void*            m_nativeDisplayHandle;

    gslDeviceModeEnum   m_deviceMode;

    typedef std::map<gslMemObject, intp> Hack;
    Hack             m_hack;
    gslQueryObject   m_mapQuery;
    gslQueryObject   m_mapDMAQuery;

    gslStaticRuntimeConfig  m_scfg;
    gslDynamicRuntimeConfig m_dcfg;

    //GL Extension specific
    bool             initGLInteropPrivateExt(CALvoid* GLplatformContext, CALvoid* GLdeviceContext) const;
    bool             glCanInterop(CALvoid* GLplatformContext, CALvoid* GLdeviceContext);

    bool             PerformDMACopy(gslMemObject srcMem, gslMemObject destMem, cmSurfFmt format, CALuint flags, bool isHwDebug = false);
    void             Initialize(void);

    bool             SetupAdapter(int32 &asic_id);
    bool             SetupContext(int32 &asic_id);
    void             PerformAdapterInitialization_int(bool initLite);
    void             PerformFullInitialization_int();

    void             getAttribs_int(gsl::gsCtx* cs);
    bool             ResolveAperture(const gslMemObjectAttribTiling tiling) const;

    void             parsePowerParam(const char* element, gslRuntimeConfigUint32Value& pwrCount, gslRuntimeConfigUint32pValue& pwrPointer);

    CALdeviceattribs      m_attribs;
    gslMemInfo            m_memInfo;
    gslTextureResourceObject m_textureResource;
    gslSamplerObject      m_textureSampler;
    gslEngineID           m_forcedComputeEngineID;

    union {
        struct {
            uint    m_canDMA                : 1;
            uint    m_allowDMA              : 1;
            uint    m_computeRing           : 1;
            uint    m_usePerVPUAdapterModel : 1;
            uint    m_PerformLazyDeviceInit : 1;
            uint    m_isComputeRingIDForced : 1;
            uint    m_isSDMAL2TConstrained  : 1;
            uint    m_initLite              : 1;
            uint    m_fullInitialized       : 1;
        };

        uint m_flags;
    };

    amd::Monitor*   gslDeviceOps_;  //!< Lock to serialize GSL device
};

#endif // __GSLDevice_h__

