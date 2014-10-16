#ifndef __GSLDevice_h__
#define __GSLDevice_h__

#include "cal.h"
#include "calcl.h"
#include "atitypes.h"
#include "gsl_types.h"
#include "gsl_config.h"
#include "gsl_vid_if.h"
#include "thread/monitor.hpp"

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

    CALGSLDevice();
    ~CALGSLDevice();

    bool open(uint32 gpuIndex, bool enableHighPerformanceState, bool reportAsOCL12Device);
    void close();

    gslMemObject     resAlloc(const CALresourceDesc* desc) const;
    bool             resMapLocal(void*& pPtr, size_t& pitch, gslMemObject res, gslMapAccessType flags);
    bool             resUnmapLocal(gslMemObject res);

    void             resFree(gslMemObject mem) const;
    bool             resMapRemote(void*& pPtr, size_t& pitch, gslMemObject res, gslMapAccessType flags) const;
    bool             resUnmapRemote(gslMemObject res) const;

    gslMemObject     resGetHeap(size_t size) const;
    gslMemObject     resAllocView(gslMemObject res, gslResource3D size,
        CALdomain offset, cmSurfFmt format, gslChannelOrder channelOrder,
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
    const CALdeviceVideoAttribs& getVideoAttribs() const { return m_videoAttribs; }
    const CALdevicestatus& getStatus() const {return m_deviceStatus; }
    void             getMemInfo(gslMemInfo* memInfo) const;

    bool             isVmMode() const { return m_vmMode; };

    void             closeNativeDisplayHandle();

    uint32           getVPUCount();
    void             setVPUMask(uint32 mask);
    uint32           getVPUMask() const { return m_vpuMask; }
    bool             uavInCB() const { return m_uavInCB; }
    bool             canDMA() const { return m_canDMA; }
    gslMemObject     m_srcDRMDMAMem, m_dstDRMDMAMem;    // memory object of flush buffer, used for DRMDMA flush

    void             resCopy(gslMemObject srcRes, gslMemObject dstRes, uint32 flags) const;

    void             PerformAdapterInitialization() const;
    void             PerformFullInitialization() const;
    void             queryDeviceEngines(uint32* nEngines, gslEngineDescriptor* engines);

    CopyType         GetCopyType(gslMemObject srcMem, gslMemObject destMem, size_t* srcOffset,
                                     size_t* destOffset, bool allowDMA, uint32 flags, uint64& surfaceSize,
                                     size_t size, bool enableCopyRect) const;

    uint32          calcScratchBufferSize(uint32 regNum) const;

    amd::Monitor& gslDeviceOps() const { return *gslDeviceOps_; }

    void fillImageHwState(gslMemObject mem, void* hwState, uint32 hwStateSize) const;

    void fillSamplerHwState(bool unnorm, uint32 min, uint32 mag, uint32 addr, void* hwState, uint32 hwStateSize) const;

    gslSamplerObject txSampler() const { return m_textureSampler; }

    void convertInputChannelOrder(intp *channelOrder) const;

    gsl::gsCtx* gslCtx() const { return m_cs; }

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

private:
    gsl::gsAdaptor*  m_adp;
    gsl::gsCtx*      m_cs;
    gslRenderState   m_rs;
    CALtarget        m_target;
    CALuint          m_elfmachine;
    uint32           m_revision;
    uint32           m_vpuMask;
    uint32           m_chainIndex;
    int32            m_vpucount;
    int32            m_maxtexturesize;
    uint32           m_gpuIndex;
    void*            m_nativeDisplayHandle;

    gslDeviceModeEnum   m_deviceMode;

    typedef std::map<gslMemObject, intp> Hack;
    Hack             m_hack;
    gslQueryObject   m_mapQuery;
    gslQueryObject   m_mapDMAQuery;
    gslQueryObject   m_mapUVDQuery;

    gslQueryObject   m_mapVCEQuery;

    gslStaticRuntimeConfig  m_scfg;
    gslDynamicRuntimeConfig m_dcfg;

    //GL Extension specific
    bool             initGLInteropPrivateExt(CALvoid* GLplatformContext, CALvoid* GLdeviceContext) const;
    bool             glCanInterop(CALvoid* GLplatformContext, CALvoid* GLdeviceContext);

    bool             PerformDMACopy(gslMemObject srcMem, gslMemObject destMem, cmSurfFmt format, CALuint flags);
    void             Initialize(void);

    bool             SetupAdapter(int32 &asic_id);
    bool             SetupContext(int32 &asic_id);
    void             PerformAdapterInitialization_int();
    void             PerformFullInitialization_int();

    void             getAttribs_int(gsl::gsCtx* cs);
    void             getVideoAttribs_int(gslVideoContext* vsHandle);
    void             getStatus_int(gsl::gsCtx* cs);
    bool             ResolveAperture(const gslMemObjectAttribTiling tiling) const;

    CALdeviceattribs      m_attribs;
    CALdeviceVideoAttribs m_videoAttribs;
    CALdevicestatus       m_deviceStatus;
    gslTextureResourceObject m_textureResource;
    gslSamplerObject      m_textureSampler;
    bool                  m_isOpenCL200Device;

    union {
        struct {
            uint    m_canDMA                : 1;
            uint    m_allowDMA              : 1;
            uint    m_computeRing           : 1;
            uint    m_usePerVPUAdapterModel : 1;
            uint    m_PerformLazyDeviceInit : 1;
            uint    m_vmMode                : 1;
            uint    m_uavInCB               : 1;
        };
    };

    amd::Monitor*   gslDeviceOps_;  //!< Lock to serialize GSL device
};

#endif // __GSLDevice_h__

