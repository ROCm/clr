#ifndef __GSLContext_h__
#define __GSLContext_h__

#include "atitypes.h"
#include "gsl_types.h"
#include "gsl_vid_if.h"
#include "cal.h"
#include "calcl.h"

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
    void             setProgram(gslProgramObject func);
    void             setWavesPerSH(gslProgramObject func, uint32 wavesPerSH)const;
    bool             runProgramGrid(GpuEvent& event, const ProgramGrid* pProgramGrid, const gslMemObject* mems, uint32 numMems);
    bool             runProgramVideoDecode(GpuEvent& event, gslMemObject mo, const CALprogramVideoDecode& decode);
    void             runAqlDispatch(GpuEvent& event, const void* aqlPacket, const gslMemObject* mems,
                        uint32 numMems, gslMemObject scratch, uint32 scratchOffset, const void* cpuKernelCode,
                        uint64 hsaQueueVA, const void* kernelInfo);
    mcaddr           virtualQueueDispatcherStart();
    void             virtualQueueDispatcherEnd(GpuEvent& event, const gslMemObject* mems, uint32 numMems,
                        mcaddr signal, mcaddr loopStart, uint32 numTemplates);
    void             virtualQueueHandshake(GpuEvent& event, const gslMemObject mem, mcaddr parentState,
                        uint32 newStateValue, mcaddr parentChildCounter, mcaddr signal, bool dedicatedQueue);
    bool             isDone(GpuEvent* event);
    void             waitForEvent(GpuEvent* event);
    void             flushIOCaches() const;
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

    gslProgramObject createProgramObject(uint32 type);
    void             destroyProgramObject(gslProgramObject func);

    bool             copyPartial(GpuEvent& event, gslMemObject srcMem, size_t* srcOffset,
                        gslMemObject destMem, size_t* destOffset, size_t* size, CALmemcopyflags flags, bool enableCopyRect, uint32 bytesPerElement);

    void             setSamplerParameter(uint32 sampler, gslTexParameterPname param, CALvoid* vals);

    gslQueryObject   createCounter(gslQueryTarget target) const;
    void             configPerformanceCounter(gslQueryObject counter, CALuint block, CALuint index, CALuint event) const;
    void             destroyCounter(gslQueryObject counter) const;
    void             beginCounter(gslQueryObject counter, gslQueryTarget target) const;
    void             endCounter(gslQueryObject counter, GpuEvent& event);
    void             getCounter(uint64* result, gslQueryObject counter) const;

    gslMemObject     createConstants(uint32 count) const;
    void             setConstants(gslMemObject constants) const;
    void             destroyConstants(gslMemObject constants) const;

    bool             recompileShader(CALimage srcImage, CALimage* newImage, const CALuint type);
    bool             getMachineType(CALuint* pMachine, CALuint* pType, CALimage image);

    void             getFuncInfo(gslProgramObject func, gslProgramTarget target, CALfuncInfo* pInfo);

    void             bindAtomicCounter(uint32 index, gslMemObject obj);
    void             syncAtomicCounter(GpuEvent& event, uint32 index, bool read);

    void             setGWSResource(uint32 index, uint32 value);

    void             createVCE(CALEncodeCreateVCE* pEncodeVCE, CALuint flags);
    void             destroyVCE(CALuint flags);
    void             getDeviceInfoVCE(CALuint *num_device, CALEncodeGetDeviceInfo* pEncodeDeviceInfo, CALuint flags);
    void             getNumberOfModesVCE(CALEncodeGetNumberOfModes* pEncodeNumberOfModes, CALuint flags);
    void             getModesVCE(CALuint device_id, CALuint NumEncodeModesToRetrieve, CALEncodeGetModes* pEncodeMode, CALuint flags);
    void             getDeviceCAPVCE(CALuint device_id, CALuint encode_cap_total_size, CALEncodeGetDeviceCAP *pEncodeCAP, CALuint flags);

    void             createEncodeSession(CALuint device_id, CALencodeMode encode_mode, CAL_VID_PROFILE_LEVEL encode_profile_level,
                              CAL_VID_PICTURE_FORMAT encode_formatm, CALuint encode_width, CALuint encode_height,
                              CALuint frameRateNum, CALuint frameRateDenom, CAL_VID_ENCODE_JOB_PRIORITY  encode_priority_level);
    void             closeVideoEncodeSession(CALuint device_id);

    void             setState(CALEncodeSetState state, CALuint flags);
    void             getPictureConfig(CALEncodeGetPictureControlConfig *pPictureControlConfig, CALuint flags);
    void             getRateControlConfig(CALEncodeGetRateControlConfig *pRateControConfig, CALuint flags);
    void             getMotionEstimationConfig(CALEncodeGetMotionEstimationConfig *pMotionEstimationConfig, CALuint flags);
    void             getRDOConfig(CALEncodeGetRDOControlConfig *pRODConfig, CALuint flags);

    void             SendConfig(CALuint num_of_config_buffers, CAL_VID_CONFIG *pConfigBuffers, CALuint flags);
    void             EncodeePicture(GpuEvent& event, CALuint num_of_encode_task_input_buffer, CAL_VID_BUFFER_DESCRIPTION *encode_task_input_buffer_list, void *picture_parameter, CALuint *pTaskID, gslMemObject input_NV12_surface, CALuint flags);
    void             QueryTaskDescription(CALuint num_of_task_description_request, CALuint *num_of_task_description_return, CAL_VID_OUTPUT_DESCRIPTION *task_description_list, CALuint flags);
    void             ReleaseOutputResource(CALuint taskID, CALuint flags);
    bool             moduleLoad(CALimage image, gslProgramObject* func, gslMemObject* constants, CALUavMask* uavMask);
    bool             WaitSignal(gslMemObject mem, CALuint value);
    bool             WriteSignal(gslMemObject mem, CALuint value, CALuint64 offset);
    bool             MakeBuffersResident(CALuint numObjects, gslMemObject* pMemObjects, CALuint64* surfBusAddress, CALuint64* markerBusAddress);

    gslQueryObject  createThreadTrace(void) const;
    void            destroyThreadTrace(gslQueryObject) const;
    gslShaderTraceBufferObject CreateThreadTraceBuffer(void) const;
    void            DestroyThreadTraceBuffer(gslShaderTraceBufferObject,uint32) const;
    uint32          getThreadTraceQueryRes(gslQueryObject) const;
    void            configMemThreadTrace(gslShaderTraceBufferObject,gslMemObject,uint32,uint32) const;
    void            beginThreadTrace(gslQueryObject,gslQueryObject, gslQueryTarget,uint32,CALthreadTraceConfig&) const;
    void            endThreadTrace(gslQueryObject,uint32) const;
    void            pauseThreadTrace(uint32) const;
    void            resumeThreadTrace(uint32) const;
    void            writeTimer(bool sdma, const gslMemObject mem, uint32 offset) const;
    void            writeSurfRaw(GpuEvent& event, gslMemObject mem, size_t size, const void* data);

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
