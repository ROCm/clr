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
#include "gsl_adaptor.h"
#include "GSLContext.h"
#include "GSLDevice.h"
#include "cm_if.h"
#include "amuABI.h"
#include "shader/ProgramObject.h"
#include "shader/ComputeProgramObject.h"
#include "query/QueryObject.h"
#include "query/PerformanceQueryObject.h"
#include "constbuffer/ConstantBufferObject.h"
#include "sampler/SamplerObject.h"
#include "texture/TextureResourceObject.h"
#include "uav/UAVObject.h"
#include "RenderStateObject.h"
#include "shadertracebuffer/ShaderTraceBufferObject.h"
#include "scratchbuffer/ScratchBufferObject.h"
#include "memory/MemObject.h"

#include <algorithm>

CALGSLContext::CALGSLContext()
{
    m_cs = 0;
    m_rs = 0;
    m_allowDMA = false;

    COMPILE_TIME_ASSERT((int)MAX_OUTPUTS <= (int)GSL_MAX_OUTPUT);

    memset(m_textureSamplers, 0, sizeof(m_textureSamplers));
    memset(m_textureResources, 0, sizeof(m_textureResources));
    memset(m_uavResources, 0, sizeof(m_uavResources));
    memset(m_constantBuffers, 0, sizeof(m_constantBuffers));

    m_scratchBuffers = 0;

    m_waitType = CAL_WAIT_LOW_CPU_UTILIZATION;
}

CALGSLContext::~CALGSLContext()
{
    assert(m_cs == 0);
}

bool
CALGSLContext::open(
    const CALGSLDevice* pDeviceObject,
    uint32              nEngines,
    gslEngineDescriptor* engines,
    uint32              rtCUs)
{
    m_Dev = pDeviceObject;

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(dev()->gslDeviceOps());

    gsl::gsAdaptor* native = dev()->getNative();
    assert(native != 0);

    EQManagerConfig EQConfig = EQManager_HIGH;

    gslEngineID mainEngineOrdinal = GSL_ENGINEID_INVALID;
    gslEngineID sdmaOrdinal = GSL_ENGINEID_INVALID;
    for (uint i = 0; i < nEngines; i++)
    {
        if (engines[i].id >= GSL_ENGINEID_3DCOMPUTE0 &&
            engines[i].id <= GSL_ENGINEID_COMPUTE_MEDIUM_PRIORITY)
        {
            mainEngineOrdinal = engines[i].id;
        }

        if (engines[i].id == GSL_ENGINEID_DRMDMA0||
            engines[i].id == GSL_ENGINEID_DRMDMA1)
        {
            sdmaOrdinal = engines[i].id;
            m_allowDMA = dev()->canDMA();
        }
    }

    m_cs = native->createComputeContext(mainEngineOrdinal, sdmaOrdinal, false, rtCUs);

    if (m_cs == 0)
    {
        return false;
    }
    m_cs->getMainSubCtx()->setVPUMask(dev()->getVPUMask());

    m_cs->makeCurrent(0);

    m_rs = m_cs->createRenderState();
    if (m_rs == 0)
    {
        native->deleteContext(m_cs);
        m_cs = 0;
        return false;
    }

    m_cs->setRenderState(m_rs);
    m_cs->createSubAllocDesc();

    //
    //
    // configure the default compute mode
    //
    m_rs->setComputeShader(m_cs, true);

    m_eventQueue[MainEngine].open(m_cs, GSL_SYNC_ATI, EQConfig);
    m_eventQueue[SdmaEngine].open(m_cs, GSL_DRMDMA_SYNC_ATI, EQConfig, GSL_ENGINE_MASK(GSL_ENGINEID_DRMDMA0) | GSL_ENGINE_MASK(GSL_ENGINEID_DRMDMA1));

    m_cs->setGPU((gslGPUMask)dev()->getVPUMask());

    m_cs->setDMAFlushBuf(dev()->m_srcDRMDMAMem, dev()->m_dstDRMDMAMem, 4 /* size of CM_SURF_FMT_R32F*/);

    // Create the GSL scratch buffer object
    m_scratchBuffers = m_cs->createScratchBuffer();
    if (m_scratchBuffers == NULL)
    {
        return false;
    }

    if (m_textureSamplers[0] == 0)
    {
        // Special case. GSL validation requires a sampler with any texture setup.
        // In OCL kernel may have an image argument, but doesn't use it. So a sampler
        // can be undefined.
        //! @note HSAIL will need a sampler as well
        m_textureSamplers[0] = m_cs->createSampler();
        m_rs->setSamplerObject(GSL_COMPUTE_PROGRAM, m_textureSamplers[0], 0);
    }

    return true;
}

void
CALGSLContext::close(gsl::gsAdaptor* native)
{
    if (m_cs == 0)
    {
        return;
    }

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(dev()->gslDeviceOps());

    m_cs->Flush();

    assert(m_rs != 0);

    m_cs->setRenderState(m_rs);

    m_rs->setCurrentProgramObject(GSL_COMPUTE_PROGRAM, 0);

    for (int i = 0; i < MAX_SAMPLERS; i++)
    {
        m_rs->setSamplerObject(GSL_COMPUTE_PROGRAM, 0, i);
        if (m_textureSamplers[i] != 0)
        {
            m_cs->destroySampler(m_textureSamplers[i]);
        }
    }

    for (int i = 0; i < MAX_RESOURCES; i++)
    {
        m_rs->setTextureResourceObject(m_cs, GSL_COMPUTE_PROGRAM, 0, i);
        if (m_textureResources[i] != 0)
        {
            m_cs->destroyTextureResource(m_textureResources[i]);
        }
    }

    for (int i = 0; i < MAX_UAVS; i++)
    {
        m_rs->setUavObject(m_cs, GSL_COMPUTE_PROGRAM, 0, static_cast<uint32>(GSL_UAV0 + i));
        if (m_uavResources[i] != 0)
        {
            m_cs->destroyUAVObject(m_uavResources[i]);
        }
    }

    for (int i = 0; i < MAX_CONSTANTBUFFERS; i++)
    {
        m_rs->setConstantBufferObject(GSL_COMPUTE_PROGRAM, 0, i);
        if (m_constantBuffers[i])
        {
            m_cs->destroyConstantBuffer(m_constantBuffers[i]);
        }
    }

    if (m_scratchBuffers != NULL)
    {
        //!@todo it should be GSL_COMPUTE_PROGRAM
        m_rs->setScratchBufferObject(GSL_FRAGMENT_PROGRAM, 0);
        m_scratchBuffers->setMemObject(m_cs, 0, 0);
        m_cs->destroyScratchBuffer(m_scratchBuffers);
        m_scratchBuffers = 0;
    }

    m_cs->setRenderState(0);

    m_cs->destroyRenderState(m_rs);
    m_cs->destroySubAllocDesc();

    m_rs = 0;

    for (uint32 i = 0; i < AllEngines; ++i)
    {
        m_eventQueue[i].close();
    }

    native->deleteContext(m_cs);

    m_cs = 0;
}

bool
CALGSLContext::setInput(uint32 physUnit, gslMemObject mem)
{
    assert(physUnit < MAX_RESOURCES);

    //if there is no texture resource object associated with this unit, then allocate one.
    if (m_textureResources[physUnit] == 0)
    {
        m_textureResources[physUnit] = m_cs->createTextureResource();
        m_rs->setTextureResourceObject(m_cs, GSL_COMPUTE_PROGRAM,
            m_textureResources[physUnit], physUnit);
    }

    m_textureResources[physUnit]->updateDepthTextureParam(mem);
    m_textureResources[physUnit]->setMemObject(m_cs, GSL_COMPUTE_PROGRAM, mem, NULL);

    if (mem != NULL)
    {
        intp channelOrder = mem->getAttribs().channelOrder;
        dev()->convertInputChannelOrder(&channelOrder);
        m_rs->setTextureResourceSwizzle(GSL_COMPUTE_PROGRAM, physUnit,
            reinterpret_cast<const int32 *>(&channelOrder));
    }

    return true;
}

bool
CALGSLContext::setConstantBuffer(uint32 physUnit, gslMemObject mem, uint32 offset, size_t size)
{
    assert(physUnit < MAX_CONSTANTBUFFERS);
    assert((physUnit < MAX_APICONSTANTBUFFERS) || (physUnit == SC_INFO_CONSTANTBUFFER));

    //if there is no constant buffer object associated with this unit, then allocate one.
    if (m_constantBuffers[physUnit] == 0)
    {
        m_constantBuffers[physUnit] = m_cs->createConstantBuffer();
        m_rs->setConstantBufferObject(GSL_COMPUTE_PROGRAM, m_constantBuffers[physUnit], physUnit);
    }

    return m_constantBuffers[physUnit]->bindMemory(m_cs, mem, static_cast<mcoffset>(offset), (uint32)size);
}

bool
CALGSLContext::setUAVBuffer(uint32 physUnit, gslMemObject mem, gslUAVType uavType)
{
    assert(physUnit < MAX_UAVS);

    if (m_uavResources[physUnit] == 0)
    {
        m_uavResources[physUnit] = m_cs->createUAVObject();
        m_rs->setUavObject(m_cs, GSL_COMPUTE_PROGRAM, m_uavResources[physUnit], GSL_UAV0 + physUnit);
    }
    m_uavResources[physUnit]->setMemObject(m_cs, mem, uavType);
    m_uavResources[physUnit]->setRSOBindings(m_cs, GSL_COMPUTE_PROGRAM);

    return true;
}

void
CALGSLContext::setUAVChannelOrder(uint32 physUnit, gslMemObject mem)
{
    assert(physUnit < MAX_UAVS);
    intp channelOrder = mem->getAttribs().channelOrder;
    dev()->convertInputChannelOrder(&channelOrder);
    m_uavResources[physUnit]->setParameter(GSL_UAV_RESOURCE_SWIZZLE, &channelOrder);
}

bool
CALGSLContext::isDone(GpuEvent* event)
{
    if (event->isValid())
    {
        assert(event->engineId_ < AllEngines);
        if (m_eventQueue[event->engineId_].isDone(event->id))
        {
            event->invalidate();
            return true;
        }
        return false;
    }
    return true;
}

void
CALGSLContext::waitForEvent(GpuEvent* event)
{
    if (event->isValid())
    {
        assert(event->engineId_ < AllEngines);
        m_eventQueue[event->engineId_].waitForEvent(event->id, m_waitType);
        event->invalidate();
    }
}

void
CALGSLContext::flushCUCaches(bool flushL2) const
{
    m_cs->FlushCUCaches(flushL2);
}

void
CALGSLContext::setScratchBuffer(gslMemObject mem, int32 engineId)
{
    // This card has global scratch buffer, so we only manage one resource,
    // independent of program type and number of shader engineers.
    // For consistency with GSL, We will store the buffer under the
    // fragment program type for shader engine 0.
    //!@tod should be GSL_COMPUTE_PROGRAM
    gslProgramTargetEnum target = GSL_FRAGMENT_PROGRAM;
    gslScratchBufferObject  scratchBuff = (mem != NULL) ? m_scratchBuffers : NULL;

    m_rs->setScratchBufferObject(target, m_scratchBuffers);

    m_scratchBuffers->setMemObject(m_cs, mem, engineId);
}

bool
CALGSLContext::copyPartial(GpuEvent&      event,
                           gslMemObject     srcMem,
                           size_t*          srcOffset,
                           gslMemObject     destMem,
                           size_t*          destOffset,
                           size_t*          size,
                           CALmemcopyflags  flags,
                           bool             enableRectCopy,
                           uint32           bytesPerElement)
{
    uint32      mode = GSL_SYNCUPLOAD_IGNORE_ELEMENTSIZE;
    EngineType  engineId = MainEngine;
    assert(m_cs != 0);
    CopyType    type = USE_NONE;
    uint64  linearBytePitch = 0;
    intp bpp = 0;

    type = dev()->GetCopyType(srcMem, destMem, srcOffset, destOffset, m_allowDMA, flags, size[0], enableRectCopy);

    if(type == USE_NONE)
    {
        return false;
    }

    switch (flags)
    {
    case CAL_MEMCOPY_DEFAULT:
    case CAL_MEMCOPY_SYNC:
        mode |= GSL_SYNCUPLOAD_SYNC_START | GSL_SYNCUPLOAD_SYNC_WAIT;
        break;

    case CAL_MEMCOPY_ASYNC:
        if ((type == USE_DRMDMA) || (type == USE_DRMDMA_T2L) || (type == USE_DRMDMA_L2T))
        {
            engineId = SdmaEngine;
        }
        break;

    default:
        break;
    }

    gslErrorCode gslErr = GSL_NO_ERROR;

    switch (type)
    {
    case USE_DRMDMA:
        mode |= GSL_SYNCUPLOAD_DMA;
        eventBegin(engineId);
        if(enableRectCopy)
        {
            if ((*srcOffset%4 != 0) || (*destOffset%4 != 0) || (size[0]%4 !=0))
            {
                return false;
            }
            m_cs->syncUploadRawRect(srcMem, srcOffset[0], (uint32)srcOffset[1], (uint32)srcOffset[2],
                                    destMem, destOffset[0], (uint32)destOffset[1], (uint32)destOffset[2],
                                    size[0], (uint32)size[1], (uint32)size[2], mode, bytesPerElement);
        }
        else
        {
            m_cs->syncUploadRaw(srcMem, srcOffset[0], destMem, destOffset[0], size[0], mode);
        }
        eventEnd(engineId, event);
        break;

    case USE_DRMDMA_T2L:
        mode |= GSL_SYNCUPLOAD_DMA;
        eventBegin(engineId);
        bpp = srcMem->getBitsPerElement();
        linearBytePitch = size[0] * (bpp / 8);
        gslErr = m_cs->DMACopySubSurface(srcOffset[0], (uint32)srcOffset[1], size[0], (uint32)size[1],
            destMem, destOffset[0], linearBytePitch, srcMem, 0, 0, ATIGL_FALSE, mode);
        eventEnd(engineId, event);
        break;

    case USE_DRMDMA_L2T:
        mode |= GSL_SYNCUPLOAD_DMA;
        eventBegin(engineId);
        bpp = destMem->getBitsPerElement();
        linearBytePitch = size[0] * (bpp / 8);
        gslErr = m_cs->DMACopySubSurface(destOffset[0], (uint32)destOffset[1], size[0], (uint32)size[1],
            srcMem, srcOffset[0], linearBytePitch, destMem, 0, 0, ATIGL_TRUE, mode);
        eventEnd(engineId, event);
        break;

    case USE_CPDMA:
        eventBegin(MainEngine);
        m_cs->syncUploadRaw(srcMem, srcOffset[0], destMem, destOffset[0], size[0], mode);
        eventEnd(MainEngine, event);
        break;

    default:
        assert(0);
        //
        // XXX - should never be here
        //
        return false;
    }

    if (gslErr != GSL_NO_ERROR)
    {
        return false;
    }

    return true;
}

void
CALGSLContext::setSamplerParameter(uint32 sampler, gslTexParameterPname param, void* vals)
{
    if (m_textureSamplers[sampler] == 0)
    {
        m_textureSamplers[sampler] = m_cs->createSampler();
        m_rs->setSamplerObject(GSL_COMPUTE_PROGRAM, m_textureSamplers[sampler], sampler);
    }

    float* params = reinterpret_cast<float*>(vals);
    switch (param)
    {
    case GSL_TEXTURE_MIN_FILTER:
        m_textureSamplers[sampler]->setMinFilter(m_cs,
            static_cast<gslTexParameterParamMinFilter>((uint32)params[0]));
        break;
    case GSL_TEXTURE_MAG_FILTER:
        m_textureSamplers[sampler]->setMagFilter(m_cs,
            static_cast<gslTexParameterParamMagFilter>((uint32)params[0]));
        break;
    case GSL_TEXTURE_WRAP_S:
    case GSL_TEXTURE_WRAP_T:
    case GSL_TEXTURE_WRAP_R:
        m_textureSamplers[sampler]->setWrap(m_cs, param,
            static_cast<gslTexParameterParamWrap>((uint32)params[0]));
        break;
    case GSL_TEXTURE_BORDER_COLOR:
        m_textureSamplers[sampler]->setBorderColor(m_cs, params);
        break;
    default:
        assert(!"Unknown sampler state");
        break;
    }
}

bool
CALGSLContext::moduleLoad(CALimage image,
    gslProgramObject* func, gslMemObject* constants)
{
    AMUabiMultiBinary binary;
    AMUabiEncoding encoding;

    amuABIMultiBinaryCreate(&binary);
    amuABIMultiBinaryUnpack(binary, image);

    CALuint machine, type, count = 0;
    amuABIMultiBinaryGetEncodingCount(&count, binary);
    bool binaryFound = false;
    for (CALuint i = 0; i < count; ++i)
    {
        if (amuABIMultiBinaryGetEncoding(&encoding, binary, i) &&
            amuABIEncodingGetSignature(&machine, &type, encoding) &&
            (machine == dev()->getElfMachine()) && (type == (CALuint)ED_ATI_CAL_TYPE_COMPUTE))
        {
            binaryFound = true;
            break;
        }
    }

    if (!binaryFound)
    {
        amuABIMultiBinaryDestroy(binary);
        return false;
    }

    *func = m_cs->createProgramObject(GSL_COMPUTE_PROGRAM);
    if (*func == 0)
    {
        amuABIMultiBinaryDestroy(binary);
        return false;
    }
    (*func)->programStringARB(m_cs, GSL_COMPUTE_PROGRAM, GSL_PROGRAM_FORMAT_ELF_BINARY, 0, image);

    // Setup the loop constants from the ELF binary int const area.
    CALuint numConstants = 0;
    CALuint maxPhysical = 0;

    AMUabiLiteralConst* litConsts;
    CALuint litConstsCount = 0;
    amuABIEncodingGetLitConsts(&litConstsCount, &litConsts, encoding);
    for (CALuint i = 0; i < litConstsCount; ++i)
    {
        if (litConsts[i].type == AMU_ABI_INT32)
        {
            maxPhysical = std::max(maxPhysical, litConsts[i].addr);
            ++numConstants;
        }
    }

    if (numConstants > 0)
    {
        const gslMemObjectAttribs attribs(
            GSL_MOA_CONSTANT_STORE,      // type
            GSL_MOA_MEMORY_CARD,     // location
            GSL_MOA_TILING_LINEAR,   // tiling
            GSL_MOA_DISPLAYABLE_NO,  // displayable
            ATIGL_FALSE,             // mipmap
            1,                       // samples
            0,                       // cpu_address
            GSL_MOA_SIGNED_NO,       // signed_format
            GSL_MOA_FORMAT_NORM,     // numFormat
            DRIVER_MODULE_GLL,       // module
            GSL_ALLOCATION_INSTANCED // alloc_type
        );

        *constants = m_cs->createMemObject1D(CM_SURF_FMT_RGBX8, ++maxPhysical, &attribs);

        CALuint* ptr = static_cast<CALuint*>((*constants)->map(m_cs, GSL_MAP_READ_WRITE));
        assert(ptr != 0 && "gslMapMemImage failed!");

        for (CALuint i = 0; i < litConstsCount; ++i)
        {
            if (litConsts[i].type == AMU_ABI_INT32)
            {
                ptr[litConsts[i].addr] = litConsts[i].value.int32[0];
            }
        }

        (*constants)->unmap(m_cs);
    }

    amuABIMultiBinaryDestroy(binary);

    // FIXME Until we get everything right, return an error or we'll hang the HW
    return true;
}

void
CALGSLContext::InvalidateSqCaches(bool instInvalidate, bool dataInvalidate, bool tcL1, bool tcL2)
{
    // invalidating instruction/data L1 caches using Escape
    if (instInvalidate || dataInvalidate) {
        m_cs->invalidateSqCaches(instInvalidate, dataInvalidate);
    }

    if (tcL1) {
        flushCUCaches(tcL2);
    }

}

