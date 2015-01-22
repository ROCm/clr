/*******************************************************************************
 *
 *  Copyright (c) 2014 Advanced Micro Devices, Inc. (unpublished)
 *
 *  All rights reserved.  This notice is intended as a precaution against
 *  inadvertent publication and does not imply publication or any waiver
 *  of confidentiality.  The year included in the foregoing notice is the
 *  year of creation of the work.
 *
 ******************************************************************************/

#include "gpudebugmanager.hpp"
#include "gpudevice.hpp"
#include "platform/commandqueue.hpp"

#include "device/device.hpp"
#include "device/gpu/gpumemory.hpp"
#include <iostream>
#include <sstream>
#include <fstream>

namespace gpu {

class VirtualGPU;
class Device;
class Memory;

/*
 ***************************************************************************
 *                  Implementation of GPU Debug Manager class
 ***************************************************************************
 */

GpuDebugManager::GpuDebugManager(amd::Device* device)
    : HwDebugManager(device)
    , vGpu_(NULL)
    , debugMessages_(0)
    , addressWatch_(NULL)
    , addressWatchSize_(0)
    , oclEventHandle_(NULL)
{
    // Initialize the exception info and the kernel execution mode
    excpPolicy_.exceptionMask = 0x0;
    excpPolicy_.waveAction =  CL_DBG_WAVES_RESUME;
    excpPolicy_.hostAction = CL_DBG_HOST_IGNORE;
    excpPolicy_.waveMode = CL_DBG_WAVEMODE_BROADCAST;

    execMode_.ui32All = 0;

    rtTrapHandlerInfo_.trap_.trapHandler_ = NULL;
    rtTrapHandlerInfo_.trap_.trapBuffer_  = NULL;

    aqlPacket_ = (hsa_kernel_dispatch_packet_t *) NULL;

    return;
}

GpuDebugManager::~GpuDebugManager()
{
    if (NULL != addressWatch_) {
        delete [] addressWatch_;
    }
}

void
GpuDebugManager::executePreDispatchCallBack(void*  aqlPacket,
                                            void*  toolInfo)
{
    DebugToolInfo* info = reinterpret_cast<DebugToolInfo*>(toolInfo);

    aqlPacket_ = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(aqlPacket);

    // Only if the pre-dispatch callback is set, will we update cache
    // flush configuration and build the memory descriptor.
    if (NULL != preDispatchCallBackFunc_) {
        // Build the scratch memory descriptor
        device()->gslCtx()->BuildScratchBufferResource(debugInfo_.scratchMemoryDescriptor_,
                                          info->scratchAddress_,
                                          info->scratchSize_);

        // Build the global memory descriptor
        device()->gslCtx()->BuildHeapBufferResource(debugInfo_.globalMemoryDescriptor_,
                                       info->globalAddress_);

//      // for invalidate cache (BuildEndOfKernelNotifyCommands)
//        aqlPacket->release_fence_scope = 2;

        aclBinary_ = reinterpret_cast<void*>(info->aclBinary_);
        oclEventHandle_ = reinterpret_cast<void*>(as_cl(info->event_));

        cl_device_id clDeviceId = as_cl(device_);
        preDispatchCallBackFunc_(clDeviceId,
                                 oclEventHandle_,
                                 aqlPacket_,
                                 aclBinary_,
                                 deviceTrapInfo_,
                                 preDispatchCallBackArgs_);
    }

    // Copy the various info set by the debugger/profiler to the tool info structure
    setupTrapInformation(info);
}

void
GpuDebugManager::executePostDispatchCallBack()
{
    if (NULL != postDispatchCallBackFunc_) {
        cl_device_id clDeviceId = as_cl(device_);
        postDispatchCallBackFunc_(clDeviceId,
                                  aqlPacket_->completion_signal.handle,
                                  postDispatchCallBackArgs_);
    }
}


cl_int
GpuDebugManager::registerDebugger(amd::Context* context, uintptr_t messageStorage)
{
    //! @todo: obtain the global mutex of HW debug to make sure only one debugger process exist

    if (!device()->settings().enableHwDebug_) {
        LogError("debugmanager: Register debugger error - HW DEBUG is not enable");
        return CL_DEBUGGER_REGISTER_FAILURE_AMD;
    }

    // first time register - set the message storage, flush queue and enable hw debug
    if (!isRegistered()) {
        debugMessages_ = messageStorage;
        dbgMsgBufferReady_ = true;
        isRegistered_ = false;
    }

    context_ = context;

    return CL_SUCCESS;
}

void
GpuDebugManager::unregisterDebugger()
{
    if (isRegistered()) {
        //! @todo: release the global mutex of HW debug

        // reset the debugger registration flag
        isRegistered_ = false;
        dbgMsgBufferReady_ = false;

        context_ = NULL;
    }
}

cl_int
GpuDebugManager::registerDebuggerOnQueue(device::VirtualDevice* vDevice)
{
    if (!isMsgBufferReady()) {
        return CL_DEBUGGER_REGISTER_FAILURE_AMD;
    }

    if (isRegistered()) {               // The debugger has already been registered,
        return CL_SUCCESS;              //   nothing to be done
    }

    VirtualGPU* vGpu = reinterpret_cast<gpu::VirtualGPU*>(vDevice);

    //  populate the fields in the debugMessages structure used by the GPU exception notification
    if (vGpu->RegisterHwDebugger(debugMessages_)) {
        vGpu_ = vGpu;
        isRegistered_ = true;
        return CL_SUCCESS;
    }

    return CL_DEBUGGER_REGISTER_FAILURE_AMD;
}

void
GpuDebugManager::flushCache(uint32_t mask)
{
    HwDbgGpuCacheMask cacheMask(mask);
    device()->xferQueue()->flushCuCaches(cacheMask);
}


void
GpuDebugManager::setupTrapInformation(DebugToolInfo* toolInfo)
{
    toolInfo->scratchAddress_       = 0;
    toolInfo->scratchSize_          = 0;
    toolInfo->globalAddress_        = 0;
    toolInfo->sqPerfcounterEnable_  = false;

    // Set up trap related info in the kernel info structure to be
    // used in the kernel dispatch.
    toolInfo->exceptionMask_ = excpPolicy_.exceptionMask;
    toolInfo->gpuSingleStepMode_ = execMode_.gpuSingleStepMode;
    toolInfo->monitorMode_ = execMode_.monitorMode;

    // The order of these three bits is determined by the definition
    // of the register COMPUTE_DISPATCH_INITIATOR
    toolInfo->cacheDisableMask_ = ((execMode_.disableL1Scalar << 2)
                                   |  (execMode_.disableL2Cache << 1)
                                   |  (execMode_.disableL1Vector));

    toolInfo->reservedCuNum_ = execMode_.reservedCuNum;

    toolInfo->trapHandler_ =
                as_amd(reinterpret_cast<cl_mem>(deviceTrapInfo_[kDebugTrapHandlerLocation]));
    toolInfo->trapBuffer_ =
                as_amd(reinterpret_cast<cl_mem>(deviceTrapInfo_[kDebugTrapBufferLocation]));
}


void
GpuDebugManager::getPacketAmdInfo(
    const void* aqlCodeInfo,
    void* packetInfo) const

{
    const AqlCodeInfo* codeInfo =
                    reinterpret_cast<const AqlCodeInfo*>(aqlCodeInfo);

    const amd_kernel_code_t* hostAqlCode = codeInfo->aqlCode_;

    PacketAmdInfo* packet =
                    reinterpret_cast<PacketAmdInfo*>(packetInfo);

    const amd_kernel_code_t* akc = hostAqlCode;

    packet->numberOfSgprs_ = akc->wavefront_sgpr_count;
    packet->numberOfVgprs_ = akc->workitem_vgpr_count;

    //  use mapped kernel_object_address for host accessing of ISA buffer
    packet->pointerToIsaBuffer_ = (char*) (hostAqlCode) +
                                            akc->kernel_code_entry_byte_offset;

    packet->scratchBufferWaveOffset_ =
                                akc->debug_wavefront_private_segment_offset_sgpr;

    packet->sizeOfIsaBuffer_ = codeInfo->aqlCodeSize_;

    packet->sizeOfStaticGroupMemory_ = akc->workgroup_group_segment_byte_size;

    // The trap_reserved_vgpr_index will be 4 less the original
    // This value must be used only by the debugger
    packet->trapReservedVgprIndex_ = akc->workitem_vgpr_count - NumberReserveVgprs;
}

DebugEvent
GpuDebugManager::createDebugEvent(
    const bool  autoReset)
{
    if (!isRegistered()) {
        LogError("debugmanager: Failed to flush cache - hw debug is not available");
        return 0;
    }


    // create the event object
    osEventHandle shaderEvent = osEventCreate(!autoReset);

    // event object has been created, set the initial state
    if (shaderEvent != 0) {

        osEventReset(shaderEvent);   // initial state is non-signaled

        if (vGpu_->ExceptionNotification(shaderEvent)) {
            isRegistered_ = true;
            return shaderEvent;
        }
    }

    return 0;
}

cl_int
GpuDebugManager::waitDebugEvent(
    DebugEvent    pEvent,
    uint32_t      timeOut) const
{
    if (osEventTimedWait(pEvent, timeOut)) {
        return CL_SUCCESS;
    }
    else {
        return CL_EVENT_TIMEOUT_AMD;
    }
}

void
GpuDebugManager::destroyDebugEvent(DebugEvent* pEvent)
{
    osEventDestroy(*pEvent);
    *pEvent = 0;

    vGpu_->ExceptionNotification(0);

}

void
GpuDebugManager::wavefrontControl(
    uint32_t waveAction,
    uint32_t waveMode,
    uint32_t trapId,
    void*    waveAddr) const
{
    device()->gslCtx()->executeSqCommand(waveAction, waveMode, trapId, waveAddr);
}

void
GpuDebugManager::setAddressWatch(
    uint32_t    numWatchPoints,
    void**      watchAddress,
    uint64_t*   watchMask,
    uint64_t*   watchMode,
    DebugEvent* event)
{
    size_t  requiredSize = numWatchPoints * sizeof(HwDbgAddressWatch);

    //  previously allocated size is not big enough, allocate new memory
    if (addressWatchSize_ < requiredSize) {
        if (NULL != addressWatch_) {    // free the smaller address watch storage
            delete [] addressWatch_;
        }
        addressWatch_ = new HwDbgAddressWatch[numWatchPoints];
        addressWatchSize_ = requiredSize;
    }

    //  fill in the address watch structure
    memset(addressWatch_, 0, addressWatchSize_);

    for (uint32_t i = 0; i < numWatchPoints; i++)
    {
        amd::Memory* watchMem = as_amd(reinterpret_cast<cl_mem>(watchAddress[i]));
        Memory* watchMemAddress = device()->getGpuMemory(watchMem);

        addressWatch_[i].watchAddress_ = reinterpret_cast<void*>(watchMemAddress->vmAddress());
        addressWatch_[i].watchMask_ = watchMask[i];
        addressWatch_[i].watchMode_ = (cl_dbg_address_watch_mode_amd) watchMode[i];
        addressWatch_[i].event_ = (0 != event) ? event[i] : 0;
    }

    //  setup the watch addresses
    device()->gslCtx()->setAddressWatch(numWatchPoints, (void*) addressWatch_);

}

void
GpuDebugManager::setGlobalMemory(
    amd::Memory* memObj,
    uint32_t offset,
    void* srcPtr,
    uint32_t size)
{
    gpu::Memory* globalMem = device()->getGpuMemory(memObj);

    address  mappedMem = static_cast<address>(globalMem->map(NULL,0));
    assert(mappedMem != 0);

    void* dest_ptr = reinterpret_cast<void*>(mappedMem + offset);
    memcpy(dest_ptr, srcPtr, size);

    globalMem->unmap(NULL);
}


}  // namespace gpu
