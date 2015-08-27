#ifndef __GSLContext_h__
#define __GSLContext_h__

#include "atitypes.h"
#include "gsl_types.h"
#include "backend.h"

#include "EventQueue.h"
#include "amuABI.h"

#define SC_INFO_CONSTANTBUFFER    (147-128)
#define SC_SR_INIT_CONSTANTBUFFER 0

#define HW_R800_MAX_UAV             12
#define SC_R800_ARENA_UAV_SHORT_ID   9
#define SC_R800_ARENA_UAV_BYTE_ID   10
#define SC_R800_ARENA_UAV_DWORD_ID  11

class CALGSLDevice;

namespace gsl
{
    class gsAdaptor;
};

class CALGSLContext
{
public:
    CALGSLContext();
    ~CALGSLContext();

    bool open(const CALGSLDevice* pDeviceObject, uint32 nEngines, gslEngineDescriptor *engines);
    void close(gsl::gsAdaptor* native);

    bool             setInput(uint32 physUnit, gslMemObject mem);
    bool             setOutput(uint32 physUnit, gslMemObject mem);
    bool             setConstantBuffer(uint32 physUnit, gslMemObject mem, CALuint offset, size_t size);
    bool             setUAVBuffer(uint32 physUnit, gslMemObject mem, gslUAVType uavType);
    void             setUavMask(const CALUavMask& uavMask);
    void             setUAVChannelOrder(uint32 physUnit, gslMemObject mem);
    bool             isDone(GpuEvent* event);
    void             waitForEvent(GpuEvent* event);
    void             flushCUCaches(bool flushL2 = false) const;
    void             eventBegin(EngineType engId)
    {
        m_eventQueue[engId].begin();
        const static bool Begin = true;
        profileEvent(engId, Begin);
    }
    void             eventEnd(EngineType engId, GpuEvent& event)
    {
        const static bool End = false;
        profileEvent(engId, End);
        event.id    = m_eventQueue[engId].end();
        event.engineId_ = engId;
    }

    bool             copyPartial(GpuEvent& event, gslMemObject srcMem, size_t* srcOffset,
                        gslMemObject destMem, size_t* destOffset, size_t* size, CALmemcopyflags flags, bool enableCopyRect, uint32 bytesPerElement);

    void             setSamplerParameter(uint32 sampler, gslTexParameterPname param, CALvoid* vals);

    bool             recompileShader(CALimage srcImage, CALimage* newImage, const CALuint type);
    bool             getMachineType(CALuint* pMachine, CALuint* pType, CALimage image);

    bool             moduleLoad(CALimage image, gslProgramObject* func, gslMemObject* constants, CALUavMask* uavMask);

    gsl::gsCtx*     cs() const { return m_cs; }
    gslRenderState  rs() const { return m_rs; }

    /// HW Debug support functions
    void            InvalidateSqCaches(bool instInvalidate = true, bool dataInvalidate = true, bool tcL1 = true, bool tcL2 = true);

protected:
    void setScratchBuffer(gslMemObject mem, int32 engineId);
    virtual void profileEvent(EngineType engine, bool type) const {}

    CALwaitType m_waitType;     //!< Wait type

private:
    enum {
        MAX_OUTPUTS         = 12,
        MAX_CONSTANTBUFFERS = 20,
        MAX_APICONSTANTBUFFERS = 16,
        MAX_SAMPLERS        = 16,
        MAX_RESOURCES       = 128,
        MAX_SCRATCHBUFFERS  = 1,
        MAX_SHADERENGINES   = 4,
        MAX_UAVS            = 1024,
    };

    const CALGSLDevice*     m_Dev;
    const CALGSLDevice*     dev() const { return m_Dev; }

    gsl::gsCtx*             m_cs;
    gslRenderState          m_rs;
    gslConstantBufferObject m_constantBuffers[MAX_CONSTANTBUFFERS];
    gslUAVObject            m_uavResources[MAX_UAVS];
    gslTextureResourceObject m_textureResources[MAX_RESOURCES];
    gslSamplerObject        m_textureSamplers[MAX_SAMPLERS];
    gslDrawBuffers          m_drawBuffers;
    gslFramebufferObject    m_fb;
    gslScratchBufferObject  m_scratchBuffers;
    EventQueue              m_eventQueue[AllEngines];
    bool                    m_allowDMA;
};

#endif // __GSLContext_h__
