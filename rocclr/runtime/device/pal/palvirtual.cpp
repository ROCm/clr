//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#include "platform/perfctr.hpp"
#include "platform/threadtrace.hpp"
#include "platform/kernel.hpp"
#include "platform/commandqueue.hpp"
#include "device/pal/palconstbuf.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/palkernel.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palcounters.hpp"
#include "device/pal/palthreadtrace.hpp"
#include "device/pal/paltimestamp.hpp"
#include "device/pal/palblit.hpp"
#include "device/pal/paldebugger.hpp"
#include "hsa.h"
#include "amd_hsa_kernel_code.h"
#include "amd_hsa_queue.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include "palQueue.h"
#include "palFence.h"

#ifdef _WIN32
#include <d3d10_1.h>
#include "amdocl/cl_d3d9_amd.hpp"
#include "amdocl/cl_d3d10_amd.hpp"
#include "amdocl/cl_d3d11_amd.hpp"
#endif // _WIN32

namespace pal {

VirtualGPU::Queue*
VirtualGPU::Queue::Create(
    Pal::IDevice*   palDev,
    Pal::QueueType  queueType,
    uint            engineIdx,
    Pal::ICmdAllocator* cmdAllocator)
{
    Pal::Result result;
    Pal::QueueCreateInfo        qCreateInfo = {};
    qCreateInfo.engineType = queueType;
    qCreateInfo.engineIndex = engineIdx;
    qCreateInfo.aqlQueue = true;

    // Find queue object size
    size_t qSize = palDev->GetQueueSize(qCreateInfo, &result);
    if (result != Pal::Result::Success) {
        return nullptr;
    }

    Pal::CmdBufferCreateInfo    cmdCreateInfo = {};
    cmdCreateInfo.pCmdAllocator = cmdAllocator;
    cmdCreateInfo.queueType = queueType;

    // Find command buffer object size
    size_t cmdSize = palDev->GetCmdBufferSize(cmdCreateInfo, &result);
    if (result != Pal::Result::Success) {
        return nullptr;
    }

    // Find fence object size
    size_t fSize = palDev->GetFenceSize(&result);
    if (result != Pal::Result::Success) {
        return nullptr;
    }

    size_t allocSize = qSize + MaxCmdBuffers * (cmdSize + fSize);
    VirtualGPU::Queue*  queue = new (allocSize) VirtualGPU::Queue(palDev);
    if (queue != nullptr) {
        address addrQ = reinterpret_cast<address>(&queue[1]);
        // Create PAL queue object
        result = palDev->CreateQueue(qCreateInfo, addrQ,  &queue->iQueue_);
        if (result != Pal::Result::Success) {
            delete queue;
            return nullptr;
        }

        address addrCmd = addrQ + qSize;
        address addrF = addrCmd + MaxCmdBuffers * cmdSize;
        Pal::CmdBufferBuildInfo cmdBuildInfo = {};

        for (uint i = 0; i < MaxCmdBuffers; ++i) {
            result = palDev->CreateCmdBuffer(cmdCreateInfo,
                &addrCmd[i*cmdSize], &queue->iCmdBuffs_[i]);
            if (result != Pal::Result::Success) {
                delete queue;
                return nullptr;
            }
            static const bool InitiallySignaled = false;
            result = palDev->CreateFence(InitiallySignaled, &addrF[i*fSize],
                &queue->iCmdFences_[i]);
            if (result != Pal::Result::Success) {
                delete queue;
                return nullptr;
            }
            if (i == StartCmdBufIdx) {
                result = queue->iCmdBuffs_[i]->Begin(cmdBuildInfo);
                if (result != Pal::Result::Success) {
                    delete queue;
                    return nullptr;
                }
            }
        }
    }
    return queue;
}

VirtualGPU::Queue::~Queue()
{
    std::vector<Pal::IGpuMemory*>   memRef;
    // Remove all memory references
    for (auto it: memReferences_) {
        memRef.push_back(it.first);
    }
    if (memRef.size() != 0) {
        iDev_->RemoveGpuMemoryReferences(memRef.size(), &memRef[0], NULL);
    }
    memReferences_.clear();

    for (uint i = 0; i < MaxCmdBuffers; ++i) {
        if (nullptr != iCmdBuffs_[i]) {
            iCmdBuffs_[i]->Destroy();
        }
        if (nullptr != iCmdFences_[i]) {
            iCmdFences_[i]->Destroy();
        }
    }

    if (nullptr != iQueue_) {
        iQueue_->Destroy();
    }
}

void
VirtualGPU::Queue::addCmdMemRef(Pal::IGpuMemory* iMem)
{            
    auto it = memReferences_.find(iMem);
    if (it != memReferences_.end()) {
        it->second = (it->second & FirstMemoryReference) | cmdBufIdSlot_;
    }
    else {
        memReferences_[iMem] = FirstMemoryReference | cmdBufIdSlot_;
    }
}

void
VirtualGPU::Queue::removeCmdMemRef(Pal::IGpuMemory* iMem)
{
    if (0 != memReferences_.erase(iMem)) {
        iDev_->RemoveGpuMemoryReferences(1, &iMem, iQueue_);
    }
}

uint
VirtualGPU::Queue::submit(bool forceFlush)
{
    cmdCnt_++;
    uint id = cmdBufIdCurrent_;
    if ((cmdCnt_ > MaxCommands) || forceFlush) {
        if (!flush()) {
            return GpuEvent::InvalidID;
        }
    }
    return id;
}

bool
VirtualGPU::Queue::flush()
{
    std::vector<Pal::IGpuMemory*>   memRef;
    // Stop commands building
    if (Pal::Result::Success != iCmdBuffs_[cmdBufIdSlot_]->End()) {
        LogError("PAL failed to finalize a command buffer!");
        return false;
    }
    // Add memory references
    for (auto it = memReferences_.begin(); it != memReferences_.end(); ++it) {
        if (it->second & FirstMemoryReference) {
            it->second &= ~FirstMemoryReference;
            memRef.push_back(it->first);
        }
    }

    if (memRef.size() != 0) {
        iDev_->AddGpuMemoryReferences(memRef.size(), &memRef[0], iQueue_,
             Pal::GpuMemoryRefCantTrim);
    }

    Pal::SubmitInfo submitInfo = {};
    submitInfo.cmdBufferCount = 1;
    submitInfo.ppCmdBuffers = &iCmdBuffs_[cmdBufIdSlot_];
    submitInfo.pFence = iCmdFences_[cmdBufIdSlot_];

    // Submit command buffer to OS
    if (Pal::Result::Success != iQueue_->Submit(submitInfo)) {
        LogError("PAL failed to submit CMD!");
        return false;
    }
    if (GPU_FLUSH_ON_EXECUTION) {
        if (Pal::Result::Success !=
            iDev_->WaitForFences(1, &iCmdFences_[cmdBufIdSlot_], true, WaitTimeoutInNsec)) {
            LogError("PAL wait for a fence failed!");
            return false;
        }
    }

    // Reset the counter of commands
    cmdCnt_ = 0;

    // Find the next command buffer
    cmdBufIdCurrent_++;

    if (cmdBufIdCurrent_ == GpuEvent::InvalidID) {
        ///@todo handle wrapping
        cmdBufIdCurrent_ = 1;
        cmbBufIdRetired_ = 0;
    }

    // Wrap current slot
    cmdBufIdSlot_ = cmdBufIdCurrent_ % MaxCmdBuffers;

    // Make sure the slot isn't busy
    if (Pal::Result::NotReady == iCmdFences_[cmdBufIdSlot_]->GetStatus()) {
        if (Pal::Result::Success !=
            iDev_->WaitForFences(1, &iCmdFences_[cmdBufIdSlot_], true, WaitTimeoutInNsec)) {
            LogError("PAL wait for a fence failed!");
            return false;
        }
    }
    // Progress retired TS 
    if ((cmdBufIdCurrent_ > MaxCmdBuffers) &&
        (cmbBufIdRetired_ < (cmdBufIdCurrent_ - MaxCmdBuffers))) {
        cmbBufIdRetired_ = cmdBufIdCurrent_ - MaxCmdBuffers;
    }

    if (Pal::Result::Success !=
        iDev_->ResetFences(1, &iCmdFences_[cmdBufIdSlot_])) {
        LogError("PAL failed to reset a fence!");
        return false;
    }

    // Reset command buffer, so CB chunks could be reused
    if (Pal::Result::Success != iCmdBuffs_[cmdBufIdSlot_]->Reset(nullptr, false)) {
        LogError("PAL failed CB reset!");
        return false;
    }
    // Start command buffer building
    Pal::CmdBufferBuildInfo cmdBuildInfo = {};
    if (Pal::Result::Success != iCmdBuffs_[cmdBufIdSlot_]->Begin(cmdBuildInfo)) {
        LogError("PAL failed CB building initialization!");
        return false;
    }

    memRef.clear();
    // Remove old memory references
    for (auto it = memReferences_.begin(); it != memReferences_.end();) {
        if (it->second == cmdBufIdSlot_) {
            memRef.push_back(it->first);
            it = memReferences_.erase(it);
        }
        else {
            ++it;
        }
    }
    if (memRef.size() != 0) {
        iDev_->RemoveGpuMemoryReferences(memRef.size(), &memRef[0], iQueue_);
    }

    return true;
}

bool
VirtualGPU::Queue::waitForEvent(uint id)
{
    if (isDone(id)) {
        return true;
    }

    uint slotId = id % MaxCmdBuffers;

    // Wait for the specified fence
    if (Pal::Result::Success != iCmdFences_[slotId]->GetStatus()) {
        if (Pal::Result::Success !=
            iDev_->WaitForFences(1, &iCmdFences_[slotId], true, WaitTimeoutInNsec)) {
            LogError("PAL wait for a fence failed!");
            return false;
        }
    }
    cmbBufIdRetired_ = id; 
    return true;
}

bool
VirtualGPU::Queue::isDone(uint id)
{
    if ((id <= cmbBufIdRetired_) || (id > cmdBufIdCurrent_)) {
        return true;
    }

    if (id == cmdBufIdCurrent_) {
        // Flush the current command buffer
        flush();
    }

    if (Pal::Result::Success != iCmdFences_[id % MaxCmdBuffers]->GetStatus()) {
        return false;
    }
    cmbBufIdRetired_ = id; 
    return true;
}

bool
VirtualGPU::MemoryDependency::create(size_t numMemObj)
{
    if (numMemObj > 0) {
        // Allocate the array of memory objects for dependency tracking
        memObjectsInQueue_ = new MemoryState[numMemObj];
        if (nullptr == memObjectsInQueue_) {
            return false;
        }
        memset(memObjectsInQueue_, 0, sizeof(MemoryState) * numMemObj);
        maxMemObjectsInQueue_ = numMemObj;
    }

    return true;
}

void
VirtualGPU::MemoryDependency::validate(
    VirtualGPU&     gpu,
    const Memory*   memory,
    bool            readOnly)
{
    bool    flushL1Cache = false;

    if (maxMemObjectsInQueue_ == 0) {
        // Flush cache
        gpu.flushCUCaches();
        return;
    }

    uint64_t curStart = memory->vmAddress();
    uint64_t curEnd = curStart + memory->vmSize();

    // Loop through all memory objects in the queue and find dependency
    // @note don't include objects from the current kernel
    for (size_t j = 0; j < endMemObjectsInQueue_; ++j) {
        // Check if the queue already contains this mem object and
        // GPU operations aren't readonly
        uint64_t busyStart = memObjectsInQueue_[j].start_;
        uint64_t busyEnd = memObjectsInQueue_[j].end_;

        // Check if the start inside the busy region
        if ((((curStart >= busyStart) && (curStart < busyEnd)) ||
            // Check if the end inside the busy region
             ((curEnd > busyStart) && (curEnd <= busyEnd)) ||
            // Check if the start/end cover the busy region
             ((curStart <= busyStart) && (curEnd >= busyEnd))) &&
            // If the buys region was written or the current one is for write
            (!memObjectsInQueue_[j].readOnly_ || !readOnly)) {
            flushL1Cache = true;
            break;
        }
    }

    // Did we reach the limit?
    if (maxMemObjectsInQueue_ <= (numMemObjectsInQueue_ + 1)) {
        flushL1Cache = true;
    }

    if (flushL1Cache) {
        // Flush cache
        gpu.flushCUCaches();
    
        // Clear memory dependency state
        const static bool All = true;
        clear(!All);
    }

    // Insert current memory object into the queue always,
    // since runtime calls flush before kernel execution and it has to keep
    // current kernel in tracking
    memObjectsInQueue_
        [numMemObjectsInQueue_].start_ = curStart;
    memObjectsInQueue_
        [numMemObjectsInQueue_].end_ = curEnd;
    memObjectsInQueue_
        [numMemObjectsInQueue_].readOnly_ = readOnly;
    numMemObjectsInQueue_++;
}

void
VirtualGPU::MemoryDependency::clear(bool all)
{
    if (numMemObjectsInQueue_ > 0) {
        size_t  i, j;
        if (all) {
            endMemObjectsInQueue_ = numMemObjectsInQueue_;
        }

        // Preserve all objects from the current kernel
        for (i = 0, j = endMemObjectsInQueue_; j < numMemObjectsInQueue_; i++, j++) {
            memObjectsInQueue_[i].start_ = memObjectsInQueue_[j].start_;
            memObjectsInQueue_[i].end_ = memObjectsInQueue_[j].end_;
            memObjectsInQueue_[i].readOnly_ = memObjectsInQueue_[j].readOnly_;
        }
        // Clear all objects except current kernel
        memset(&memObjectsInQueue_[i], 0, sizeof(amd::Memory*) * numMemObjectsInQueue_);
        numMemObjectsInQueue_ -= endMemObjectsInQueue_;
        endMemObjectsInQueue_ = 0;
    }
}

VirtualGPU::DmaFlushMgmt::DmaFlushMgmt(const Device& dev)
    : cbWorkload_(0)
    , dispatchSplitSize_(0)
{
    aluCnt_ = dev.info().simdPerCU_ * dev.info().simdWidth_ * dev.info().maxComputeUnits_;
    maxDispatchWorkload_ = static_cast<uint64_t>(dev.info().maxClockFrequency_) *
        // find time in us
        100 * dev.settings().maxWorkloadTime_ *
        aluCnt_;
    resetCbWorkload(dev);
}

void
VirtualGPU::DmaFlushMgmt::resetCbWorkload(const Device& dev)
{
    cbWorkload_ = 0;
    maxCbWorkload_ = static_cast<uint64_t>(dev.info().maxClockFrequency_) *
        // find time in us
        100 * dev.settings().minWorkloadTime_ * aluCnt_;
}

void
VirtualGPU::DmaFlushMgmt::findSplitSize(
    const Device& dev, uint64_t threads, uint instructions)
{
    uint64_t workload = threads * instructions;
    if (maxDispatchWorkload_ < workload) {
        dispatchSplitSize_ = static_cast<uint>(maxDispatchWorkload_ / instructions);
        uint    fullLoad = dev.info().maxComputeUnits_ * dev.info().maxWorkGroupSize_;
        if ((dispatchSplitSize_ % fullLoad) != 0) {
            dispatchSplitSize_ = (dispatchSplitSize_ / fullLoad + 1) * fullLoad;
        }
    }
    else {
        dispatchSplitSize_ = (threads > dev.settings().workloadSplitSize_) ?
            dev.settings().workloadSplitSize_ : 0;
    }
}

bool
VirtualGPU::DmaFlushMgmt::isCbReady(
    VirtualGPU& gpu, uint64_t threads, uint instructions)
{
    bool    cbReady = false;
    uint64_t workload = amd::alignUp(threads, 4 * aluCnt_) * instructions;
    // Add current workload to the overall workload in the current DMA
    cbWorkload_ += workload;
    // Did it exceed maximum?
    if (cbWorkload_ > maxCbWorkload_) {
        // Reset DMA workload
        cbWorkload_ = 0;
        // Increase workload of the next DMA buffer by 50%
        maxCbWorkload_ = maxCbWorkload_ * 3 / 2;
        if (maxCbWorkload_ > maxDispatchWorkload_) {
            maxCbWorkload_ = maxDispatchWorkload_;
        }
        cbReady = true;
    }
    return cbReady;
}

void
VirtualGPU::addXferWrite(Memory& memory)
{
    if (xferWriteBuffers_.size() > 7) {
        dev().xferWrite().release(*this, *xferWriteBuffers_.front());
        xferWriteBuffers_.pop_front();
    }

    // Delay destruction
    xferWriteBuffers_.push_back(&memory);
}

void
VirtualGPU::releaseXferWrite()
{
    for (auto& memory : xferWriteBuffers_) {
        dev().xferWrite().release(*this, *memory);
    }
    xferWriteBuffers_.clear();
}

void
VirtualGPU::addPinnedMem(amd::Memory* mem)
{
    if (nullptr == findPinnedMem(mem->getHostMem(), mem->getSize())) {
        if (pinnedMems_.size() > 7) {
            pinnedMems_.front()->release();
            pinnedMems_.pop_front();
        }

        // Start operation, since we should release mem object
        flushDMA(getGpuEvent(dev().getGpuMemory(mem)->iMem())->engineId_);

        // Delay destruction
        pinnedMems_.push_back(mem);
    }
}

void
VirtualGPU::releasePinnedMem()
{
    for (auto& amdMemory : pinnedMems_) {
        amdMemory->release();
    }
    pinnedMems_.clear();
}

amd::Memory*
VirtualGPU::findPinnedMem(void* addr, size_t size)
{
    for (auto& amdMemory : pinnedMems_) {
        if ((amdMemory->getHostMem() == addr) && (size <= amdMemory->getSize())) {
            return amdMemory;
        }
    }
    return nullptr;
}

bool
VirtualGPU::createVirtualQueue(uint deviceQueueSize)
{
    uint MinDeviceQueueSize = 16 * 1024;
    deviceQueueSize = std::max(deviceQueueSize, MinDeviceQueueSize);

    maskGroups_      = deviceQueueSize / (512 * Ki);
    maskGroups_      = (maskGroups_== 0) ? 1 : maskGroups_;

    // Align the queue size for the multiple dispatch scheduler.
    // Each thread works with 32 entries * maskGroups
    uint extra = deviceQueueSize % (sizeof(AmdAqlWrap) *
            DeviceQueueMaskSize * maskGroups_);
    if (extra != 0) {
        deviceQueueSize += (sizeof(AmdAqlWrap) *
            DeviceQueueMaskSize * maskGroups_) - extra;
    }

    if (deviceQueueSize_ == deviceQueueSize) {
        return true;
    }
    else {
        //! @todo Temporarily keep the buffer mapped for debug purpose
        if (nullptr != schedParams_) {
            schedParams_->unmap(this);
        }
        delete vqHeader_;
        delete virtualQueue_;
        delete schedParams_;
        vqHeader_ = nullptr;
        virtualQueue_ = nullptr;
        schedParams_ = nullptr;
        schedParamIdx_ = 0;
        deviceQueueSize_ = 0;
    }
    uint    numSlots = deviceQueueSize / sizeof(AmdAqlWrap);
    uint    allocSize = deviceQueueSize;

    // Add the virtual queue header
    allocSize += sizeof(AmdVQueueHeader);
    allocSize = amd::alignUp(allocSize, sizeof(AmdAqlWrap));

    uint    argOffs = allocSize;

    // Add the kernel arguments and wait events
    uint singleArgSize = amd::alignUp(dev().info().maxParameterSize_ + 64 +
        dev().settings().numWaitEvents_ * sizeof(uint64_t), sizeof(AmdAqlWrap));
    allocSize += singleArgSize * numSlots;

    uint    eventsOffs = allocSize;
    // Add the device events
    allocSize += dev().settings().numDeviceEvents_ * sizeof(AmdEvent);

    uint    eventMaskOffs = allocSize;
    // Add mask array for events
    allocSize += amd::alignUp(dev().settings().numDeviceEvents_, DeviceQueueMaskSize) / 8;

    uint    slotMaskOffs = allocSize;
    // Add mask array for AmdAqlWrap slots
    allocSize += amd::alignUp(numSlots, DeviceQueueMaskSize) / 8;

    virtualQueue_ = new Memory(dev(), allocSize);
    Resource::MemoryType type = (GPU_PRINT_CHILD_KERNEL == 0) ?
        Resource::Local : Resource::Remote;
    if  ((virtualQueue_ == nullptr) || !virtualQueue_->create(type)) {
        return false;
    }

    if (GPU_PRINT_CHILD_KERNEL != 0) {
        address ptr  = reinterpret_cast<address>(
            virtualQueue_->map(this, Resource::WriteOnly));
        if (nullptr == ptr) {
            return false;
        }
    }

    uint64_t        vaBase = virtualQueue_->vmAddress();
    AmdVQueueHeader header = {};
    // Initialize the virtual queue header
    header.aql_slot_num    = numSlots;
    header.event_slot_num  = dev().settings().numDeviceEvents_;
    header.event_slot_mask = vaBase + eventMaskOffs;
    header.event_slots     = vaBase + eventsOffs;
    header.aql_slot_mask   = vaBase + slotMaskOffs;
    header.wait_size       = dev().settings().numWaitEvents_;
    header.arg_size        = dev().info().maxParameterSize_ + 64;
    header.mask_groups     = maskGroups_;

    vqHeader_ = new AmdVQueueHeader;
    if (nullptr == vqHeader_) {
        return false;
    }
    *vqHeader_ = header;

    virtualQueue_->writeRawData(*this, 0, sizeof(AmdVQueueHeader), &header, false);

    // Go over all slots and perform initialization
    AmdAqlWrap  slot = {};
    size_t      offset = sizeof(AmdVQueueHeader);
    for (uint i = 0; i < numSlots; ++i) {
        uint64_t argStart = vaBase + argOffs + i * singleArgSize;
        slot.aql.kernarg_address = reinterpret_cast<void*>(argStart);
        slot.wait_list = argStart + dev().info().maxParameterSize_ + 64;
        virtualQueue_->writeRawData(*this, offset, sizeof(AmdAqlWrap), &slot, false);
        offset += sizeof(AmdAqlWrap);
    }

    schedParams_ = new Memory(dev(), 64 * Ki);
    if ((schedParams_ == nullptr) || !schedParams_->create(Resource::RemoteUSWC)) {
        return false;
    }

    address ptr  = reinterpret_cast<address>(schedParams_->map(this));

    deviceQueueSize_ = deviceQueueSize;

    return true;
}

VirtualGPU::VirtualGPU(
    Device&  device)
    : device::VirtualDevice(device)
    , engineID_(MainEngine)
    , gpuDevice_(static_cast<Device&>(device))
    , execution_("Virtual GPU execution lock", true)
    , printfDbg_(nullptr)
    , printfDbgHSA_(nullptr)
    , tsCache_(nullptr)
    , dmaFlushMgmt_(device)
    , hwRing_(0)
    , readjustTimeGPU_(0)
    , currTs_(nullptr)
    , vqHeader_(nullptr)
    , virtualQueue_(nullptr)
    , schedParams_(nullptr)
    , schedParamIdx_(0)
    , deviceQueueSize_(0)
    , maskGroups_(1)
    , hsaQueueMem_(nullptr)
    , cmdAllocator_(nullptr)
{
    memset(&cal_, 0, sizeof(CalVirtualDesc));
    for (uint i = 0; i < AllEngines; ++i) {
        cal_.events_[i].invalidate();
    }

    // Note: Virtual GPU device creation must be a thread safe operation
    index_ = gpuDevice_.numOfVgpus_++;
    gpuDevice_.vgpus_.resize(gpuDevice_.numOfVgpus());
    gpuDevice_.vgpus_[index()] = this;
    queues_[MainEngine] = nullptr;
    queues_[SdmaEngine] = nullptr;
}

bool
VirtualGPU::create(bool profiling, uint  deviceQueueSize)
{
    device::BlitManager::Setup  blitSetup;

    if (index() >= GPU_MAX_COMMAND_QUEUES) {
        // Cap the maximum number of concurrent Virtual GPUs
        return false;
    }

    // Virtual GPU will have profiling enabled
    state_.profiling_ = profiling;

    Pal::CmdAllocatorCreateInfo createInfo = {};
    createInfo.flags.threadSafe = true;
    // \todo forces PAL to reuse CBs, but requires postamble
    createInfo.flags.autoMemoryReuse = false;
    createInfo.allocInfo[Pal::CommandDataAlloc].allocHeap =
        Pal::GpuHeapGartCacheable;
    createInfo.allocInfo[Pal::CommandDataAlloc].allocSize = 128 * Ki;
    createInfo.allocInfo[Pal::CommandDataAlloc].suballocSize = 128 * Ki;

    createInfo.allocInfo[Pal::EmbeddedDataAlloc].allocHeap =
        Pal::GpuHeapGartCacheable;
    createInfo.allocInfo[Pal::EmbeddedDataAlloc].allocSize = 64 * Ki;
    createInfo.allocInfo[Pal::EmbeddedDataAlloc].suballocSize = 64 * Ki;

    Pal::Result result;
    size_t cmdAllocSize = dev().iDev()->GetCmdAllocatorSize(createInfo, &result);
    if (Pal::Result::Success != result) {
        return false;
    }
    char* addr = new char [cmdAllocSize];
    if (Pal::Result::Success !=
        dev().iDev()->CreateCmdAllocator(createInfo, addr, &cmdAllocator_)) {
        return false;
    }

    if (dev().numComputeEngines()) {
        uint    idx = index() % dev().numComputeEngines();

        // hwRing_ should be set 0 if forced to have single scratch buffer
        hwRing_ = (dev().settings().useSingleScratch_) ? 0 : idx;

        queues_[MainEngine] = Queue::Create(
            dev().iDev(), Pal::QueueTypeCompute, idx, cmdAllocator_);
        if (nullptr == queues_[MainEngine]) {
            return false;
        }

        // Check if device has SDMA engines
        if (dev().numDMAEngines() != 0) {
            queues_[SdmaEngine] = Queue::Create(
                dev().iDev(), Pal::QueueTypeDma,
                idx % dev().numDMAEngines(), cmdAllocator_);
            if (nullptr == queues_[SdmaEngine]) {
                return false;
            }
        }
        else {
            Unimplemented();
        }
    }
    else {
        Unimplemented();
    }

    // Diable double copy optimization,
    // since UAV read from nonlocal is fast enough
    blitSetup.disableCopyBufferToImageOpt_ = true;
    if (!allocConstantBuffers()) {
        return false;
    }

    // Create Printf class
    printfDbg_ = new PrintfDbg(gpuDevice_);
    if ((nullptr == printfDbg_) || !printfDbg_->create()) {
        delete printfDbg_;
        LogError("Could not allocate debug buffer for printf()!");
        return false;
    }

    // Create HSAILPrintf class
    printfDbgHSA_ = new PrintfDbgHSA(gpuDevice_);
    if (nullptr == printfDbgHSA_) {
        delete printfDbgHSA_;
        LogError("Could not create PrintfDbgHSA class!");
        return false;
    }

    // Choose the appropriate class for blit engine
    switch (dev().settings().blitEngine_) {
        default:
            // Fall through ...
        case Settings::BlitEngineHost:
            blitSetup.disableAll();
            // Fall through ...
        case Settings::BlitEngineCAL:
        case Settings::BlitEngineKernel:
            // use host blit for HW debug
            if (dev().settings().enableHwDebug_) {
                blitSetup.disableCopyImageToBuffer_   = true;
                blitSetup.disableCopyBufferToImage_   = true;
            }
            blitMgr_ = new KernelBlitManager(*this, blitSetup);
            break;
    }
    if ((nullptr == blitMgr_) || !blitMgr_->create(gpuDevice_)) {
        LogError("Could not create BlitManager!");
        return false;
    }

    tsCache_ = new TimeStampCache(*this);
    if (nullptr == tsCache_) {
        LogError("Could not create TimeStamp cache!");
        return false;
    }

    if (!memoryDependency().create(dev().settings().numMemDependencies_)) {
        LogError("Could not create the array of memory objects!");
        return false;
    }

    if(!allocHsaQueueMem()) {
        LogError("Could not create hsaQueueMem object!");
        return false;
    }

    // Check if the app requested a device queue creation
    if (dev().settings().useDeviceQueue_ &&
        (0 != deviceQueueSize) && !createVirtualQueue(deviceQueueSize)) {
        LogError("Could not create a virtual queue!");
        return false;
    }

    return true;
}

bool
VirtualGPU::allocHsaQueueMem()
{
    // Allocate a dummy HSA queue
    hsaQueueMem_ = new Memory(dev(), sizeof(amd_queue_t));
    if ((hsaQueueMem_ == nullptr) ||
        (!hsaQueueMem_->create(Resource::RemoteUSWC))) {
        delete hsaQueueMem_;
        return false;
    }
    amd_queue_t* queue = reinterpret_cast<amd_queue_t*>
        (hsaQueueMem_->map(nullptr, Resource::WriteOnly));
    if (nullptr == queue) {
        delete hsaQueueMem_;
        return false;
    }
    memset(queue, 0, sizeof(amd_queue_t));

    // Provide private and local heap addresses
    const static uint addressShift = LP64_SWITCH(0, 32);
    queue->private_segment_aperture_base_hi = static_cast<uint32_t>(
        dev().properties().gpuMemoryProperties.privateApertureBase >> addressShift);
    queue->group_segment_aperture_base_hi = static_cast<uint32_t>(
        dev().properties().gpuMemoryProperties.sharedApertureBase >> addressShift);

    hsaQueueMem_->unmap(nullptr);
    return true;
}

VirtualGPU::~VirtualGPU()
{
    // Not safe to remove a queue. So lock the device
    amd::ScopedLock k(dev().lockAsyncOps());
    amd::ScopedLock lock(dev().vgpusAccess());

    // Destroy all memories
    static const bool SkipScratch = false;
    releaseMemObjects(SkipScratch);

    // Destroy printf object
    delete printfDbg_;

    // Destroy printfHSA object
    delete printfDbgHSA_;

    // Destroy BlitManager object
    delete blitMgr_;

    // Destroy TimeStamp cache
    delete tsCache_;

    // Destroy resource list with the constant buffers
    for (uint i = 0; i < constBufs_.size(); ++i) {
        delete constBufs_[i];
    }

    // Destroy queues
    if (nullptr != queues_[MainEngine]) {
        // Make sure the queues are idle
        // It's unclear why PAL could still have a busy queue
        queues_[MainEngine]->iQueue_->WaitIdle();
        delete queues_[MainEngine];
    }

    if (nullptr != queues_[SdmaEngine]) {
        queues_[SdmaEngine]->iQueue_->WaitIdle();
        delete queues_[SdmaEngine];
    }

    if (nullptr != cmdAllocator_) {
        cmdAllocator_->Destroy();
        delete [] reinterpret_cast<char*>(cmdAllocator_);
    }

    gpuDevice_.numOfVgpus_--;
    gpuDevice_.vgpus_.erase(gpuDevice_.vgpus_.begin() + index());
    for (uint idx = index(); idx < dev().vgpus().size(); ++idx) {
        dev().vgpus()[idx]->index_--;
    }

    // Release scratch buffer memory to reduce memory pressure
    //!@note OCLtst uses single device with multiple tests
    //! Release memory only if it's the last command queue.
    //! The first queue is reserved for the transfers on device
    if (gpuDevice_.numOfVgpus_ <= 1) {
        gpuDevice_.destroyScratchBuffers();
    }

    //! @todo Temporarily keep the buffer mapped for debug purpose
    if (nullptr != schedParams_) {
        schedParams_->unmap(this);
    }
    delete vqHeader_;
    delete virtualQueue_;
    delete schedParams_;
    delete hsaQueueMem_;
}

void
VirtualGPU::submitReadMemory(amd::ReadMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    // Translate memory references and ensure cache up-to-date
    pal::Memory* memory = dev().getGpuMemory(&vcmd.source());

    size_t offset = 0;
    // Find if virtual address is a CL allocation
    device::Memory* hostMemory = dev().findMemoryFromVA(vcmd.destination(), &offset);

    profilingBegin(vcmd, true);

    memory->syncCacheFromHost(*this);
    cl_command_type type = vcmd.type();
    bool result = false;
    amd::Memory* bufferFromImage = nullptr;

    // Force buffer read for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_READ_IMAGE) &&
        (vcmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        bufferFromImage = createBufferFromImage(vcmd.source());
        if (nullptr == bufferFromImage) {
            LogError("We should not fail buffer creation from image_buffer!");
        }
        else {
            type = CL_COMMAND_READ_BUFFER;
            bufferFromImage->setVirtualDevice(this);
            memory = dev().getGpuMemory(bufferFromImage);
        }
    }

    // Process different write commands
    switch (type) {
    case CL_COMMAND_READ_BUFFER: {
        amd::Coord3D    origin(vcmd.origin()[0]);
        amd::Coord3D    size(vcmd.size()[0]);
        if (nullptr != bufferFromImage) {
            size_t  elemSize =
                vcmd.source().asImage()->getImageFormat().getElementSize();
            origin.c[0] *= elemSize;
            size.c[0]   *= elemSize;
        }
        if (hostMemory != nullptr) {
            // Accelerated transfer without pinning
            amd::Coord3D dstOrigin(offset);
            result = blitMgr().copyBuffer(*memory, *hostMemory,
                origin, dstOrigin, size, vcmd.isEntireMemory());
        }
        else {
            result = blitMgr().readBuffer(
                *memory, vcmd.destination(),
                origin, size, vcmd.isEntireMemory());
        }
        if (nullptr != bufferFromImage) {
            bufferFromImage->release();
        }
    }
        break;
    case CL_COMMAND_READ_BUFFER_RECT: {
        amd::BufferRect hostbufferRect;
        amd::Coord3D    region(0);
        amd::Coord3D hostOrigin(vcmd.hostRect().start_+ offset);
        hostbufferRect.create(hostOrigin.c, vcmd.size().c , vcmd.hostRect().rowPitch_, vcmd.hostRect().slicePitch_);
        if (hostMemory != nullptr) {
            result = blitMgr().copyBufferRect(*memory, *hostMemory,
                vcmd.bufRect(), hostbufferRect, vcmd.size(),
                vcmd.isEntireMemory());
        }
        else {
            result = blitMgr().readBufferRect(*memory,
                vcmd.destination(), vcmd.bufRect(), vcmd.hostRect(), vcmd.size(),
                vcmd.isEntireMemory());
        }
    }
        break;
    case CL_COMMAND_READ_IMAGE:
        if (hostMemory != nullptr) {
            // Accelerated image to buffer transfer without pinning
            amd::Coord3D dstOrigin(offset);
            result = blitMgr().copyImageToBuffer(*memory, *hostMemory,
                vcmd.origin(), dstOrigin, vcmd.size(),
                vcmd.isEntireMemory(),
                vcmd.rowPitch(), vcmd.slicePitch());
        }
        else {
            result = blitMgr().readImage(*memory, vcmd.destination(),
                vcmd.origin(), vcmd.size(), vcmd.rowPitch(), vcmd.slicePitch(),
                vcmd.isEntireMemory());
        }
        break;
    default:
        LogError("Unsupported type for the read command");
        break;
    }

    if (!result) {
        LogError("submitReadMemory failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitWriteMemory(amd::WriteMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    // Translate memory references and ensure cache up to date
    pal::Memory* memory = dev().getGpuMemory(&vcmd.destination());
    size_t offset = 0;
    // Find if virtual address is a CL allocation
    device::Memory* hostMemory = dev().findMemoryFromVA(vcmd.source(), &offset);

    profilingBegin(vcmd, true);

    bool    entire  = vcmd.isEntireMemory();

    // Synchronize memory from host if necessary
    device::Memory::SyncFlags syncFlags;
    syncFlags.skipEntire_ = entire;
    memory->syncCacheFromHost(*this, syncFlags);

    cl_command_type type = vcmd.type();
    bool result = false;
    amd::Memory* bufferFromImage = nullptr;

    // Force buffer write for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_WRITE_IMAGE) &&
        (vcmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        bufferFromImage = createBufferFromImage(vcmd.destination());
        if (nullptr == bufferFromImage) {
            LogError("We should not fail buffer creation from image_buffer!");
        }
        else {
            type = CL_COMMAND_WRITE_BUFFER;
            bufferFromImage->setVirtualDevice(this);
            memory = dev().getGpuMemory(bufferFromImage);
        }
    }

    // Process different write commands
    switch (type) {
    case CL_COMMAND_WRITE_BUFFER: {
        amd::Coord3D    origin(vcmd.origin()[0]);
        amd::Coord3D    size(vcmd.size()[0]);
        if (nullptr != bufferFromImage) {
            size_t  elemSize =
                vcmd.destination().asImage()->getImageFormat().getElementSize();
            origin.c[0] *= elemSize;
            size.c[0]   *= elemSize;
        }
        if (hostMemory != nullptr) {
            // Accelerated transfer without pinning
            amd::Coord3D srcOrigin(offset);
            result = blitMgr().copyBuffer(*hostMemory, *memory,
                srcOrigin, origin, size, vcmd.isEntireMemory());
        }
        else {
            result = blitMgr().writeBuffer(vcmd.source(), *memory,
                origin, size, vcmd.isEntireMemory());
        }
        if (nullptr != bufferFromImage) {
            bufferFromImage->release();
        }
    }
        break;
    case CL_COMMAND_WRITE_BUFFER_RECT: {
        amd::BufferRect hostbufferRect;
        amd::Coord3D    region(0);
        amd::Coord3D hostOrigin(vcmd.hostRect().start_+ offset);
        hostbufferRect.create(hostOrigin.c, vcmd.size().c , vcmd.hostRect().rowPitch_, vcmd.hostRect().slicePitch_);
        if (hostMemory != nullptr) {
            result = blitMgr().copyBufferRect(*hostMemory, *memory,
                hostbufferRect, vcmd.bufRect(), vcmd.size(),
                vcmd.isEntireMemory());
        }
        else {
            result = blitMgr().writeBufferRect(vcmd.source(), *memory,
                vcmd.hostRect(), vcmd.bufRect(), vcmd.size(),
                vcmd.isEntireMemory());
        }
    }
        break;
    case CL_COMMAND_WRITE_IMAGE:
        if (hostMemory != nullptr) {
            // Accelerated buffer to image transfer without pinning
            amd::Coord3D srcOrigin(offset);
            result = blitMgr().copyBufferToImage(*hostMemory, *memory,
                srcOrigin, vcmd.origin(), vcmd.size(),
                vcmd.isEntireMemory(),
                vcmd.rowPitch(), vcmd.slicePitch());
        }
        else {
            result = blitMgr().writeImage(vcmd.source(), *memory,
                vcmd.origin(), vcmd.size(), vcmd.rowPitch(), vcmd.slicePitch(),
                vcmd.isEntireMemory());
        }
        break;
    default:
        LogError("Unsupported type for the write command");
        break;
    }

    if (!result) {
        LogError("submitWriteMemory failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        // Mark this as the most-recently written cache of the destination
        vcmd.destination().signalWrite(&gpuDevice_);
    }
    profilingEnd(vcmd);
}

bool
VirtualGPU::copyMemory(cl_command_type type
            , amd::Memory& srcMem
            , amd::Memory& dstMem
            , bool entire
            , const amd::Coord3D& srcOrigin
            , const amd::Coord3D& dstOrigin
            , const amd::Coord3D& size
            , const amd::BufferRect& srcRect
            , const amd::BufferRect& dstRect
            )
{
    // Translate memory references and ensure cache up-to-date
    pal::Memory* dstMemory = dev().getGpuMemory(&dstMem);
    pal::Memory* srcMemory = dev().getGpuMemory(&srcMem);

    // Synchronize source and destination memory
    device::Memory::SyncFlags syncFlags;
    syncFlags.skipEntire_ = entire;
    dstMemory->syncCacheFromHost(*this, syncFlags);
    srcMemory->syncCacheFromHost(*this);

    amd::Memory* bufferFromImageSrc = nullptr;
    amd::Memory* bufferFromImageDst = nullptr;

    // Force buffer read for IMAGE1D_BUFFER
    if ((srcMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        bufferFromImageSrc = createBufferFromImage(srcMem);
        if (nullptr == bufferFromImageSrc) {
            LogError("We should not fail buffer creation from image_buffer!");
        }
        else {
            type = CL_COMMAND_COPY_BUFFER;
            bufferFromImageSrc->setVirtualDevice(this);
            srcMemory = dev().getGpuMemory(bufferFromImageSrc);
       }
    }
    // Force buffer write for IMAGE1D_BUFFER
    if ((dstMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        bufferFromImageDst = createBufferFromImage(dstMem);
        if (nullptr == bufferFromImageDst) {
            LogError("We should not fail buffer creation from image_buffer!");
        }
        else {
            type = CL_COMMAND_COPY_BUFFER;
            bufferFromImageDst->setVirtualDevice(this);
            dstMemory = dev().getGpuMemory(bufferFromImageDst);
        }
    }

    bool result = false;

    // Check if HW can be used for memory copy
    switch (type) {
    case CL_COMMAND_SVM_MEMCPY:
    case CL_COMMAND_COPY_BUFFER: {
        amd::Coord3D    realSrcOrigin(srcOrigin[0]);
        amd::Coord3D    realDstOrigin(dstOrigin[0]);
        amd::Coord3D    realSize(size.c[0],size.c[1],size.c[2]);

        if (nullptr != bufferFromImageSrc) {
            size_t  elemSize =
                srcMem.asImage()->getImageFormat().getElementSize();
            realSrcOrigin.c[0] *= elemSize;
            if (nullptr != bufferFromImageDst) {
                realDstOrigin.c[0] *= elemSize;
            }
            realSize.c[0] *= elemSize;
        }
        else if (nullptr != bufferFromImageDst) {
            size_t  elemSize =
                dstMem.asImage()->getImageFormat().getElementSize();
            realDstOrigin.c[0] *= elemSize;
            realSize.c[0]   *= elemSize;
        }

        result = blitMgr().copyBuffer(*srcMemory, *dstMemory,
            realSrcOrigin, realDstOrigin, realSize, entire);

        if (nullptr != bufferFromImageSrc) {
            bufferFromImageSrc->release();
        }
        if (nullptr != bufferFromImageDst) {
            bufferFromImageDst->release();
        }
    }
        break;
    case CL_COMMAND_COPY_BUFFER_RECT:
        result = blitMgr().copyBufferRect(*srcMemory, *dstMemory,
            srcRect, dstRect, size, entire);
        break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
        result = blitMgr().copyImageToBuffer(*srcMemory, *dstMemory,
            srcOrigin, dstOrigin, size, entire);
        break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
        result = blitMgr().copyBufferToImage(*srcMemory, *dstMemory,
            srcOrigin, dstOrigin, size, entire);
        break;
    case CL_COMMAND_COPY_IMAGE:
        result = blitMgr().copyImage(*srcMemory, *dstMemory,
            srcOrigin, dstOrigin, size, entire);
        break;
    default:
        LogError("Unsupported command type for memory copy!");
        break;
    }

    if (!result) {
        LogError("submitCopyMemory failed!");
        return false;
    }
    else {
        // Mark this as the most-recently written cache of the destination
        dstMem.signalWrite(&gpuDevice_);
    }
    return true;
}

void
VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);

    cl_command_type type = vcmd.type();
    bool entire  = vcmd.isEntireMemory();

    if (!copyMemory(type, vcmd.source(), vcmd.destination(), entire,
            vcmd.srcOrigin(), vcmd.dstOrigin(), vcmd.size(), vcmd.srcRect(),
            vcmd.dstRect())) {
        vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());
    profilingBegin(vcmd);

    cl_command_type type = vcmd.type();
    //no op for FGS supported device
    if (!dev().isFineGrainedSystem()) {

        amd::Memory* srcMem = amd::SvmManager::FindSvmBuffer(vcmd.src());
        amd::Memory* dstMem = amd::SvmManager::FindSvmBuffer(vcmd.dst());
        if (nullptr == srcMem || nullptr == dstMem) {
            vcmd.setStatus(CL_INVALID_OPERATION);
            return;
        }

        amd::Coord3D srcOrigin(0, 0, 0);
        amd::Coord3D dstOrigin(0, 0, 0);
        amd::Coord3D size(vcmd.srcSize(), 1, 1);
        amd::BufferRect srcRect;
        amd::BufferRect dstRect;

        srcOrigin.c[0] = static_cast<const_address>(vcmd.src()) - static_cast<address>(srcMem->getSvmPtr());
        dstOrigin.c[0] = static_cast<const_address>(vcmd.dst()) - static_cast<address>(dstMem->getSvmPtr());

        if (!(srcMem->validateRegion(srcOrigin, size)) || !(dstMem->validateRegion(dstOrigin, size))) {
            vcmd.setStatus(CL_INVALID_OPERATION);
            return;
        }

        bool entire = srcMem->isEntirelyCovered(srcOrigin, size) &&
            dstMem->isEntirelyCovered(dstOrigin, size);

        if (!copyMemory(type, *srcMem, *dstMem, entire,
            srcOrigin, dstOrigin, size, srcRect, dstRect)) {
            vcmd.setStatus(CL_INVALID_OPERATION);
        }
    }
    else {
        //direct memcpy for FGS enabled system
        amd::SvmBuffer::memFill(vcmd.dst(), vcmd.src(), vcmd.srcSize(), 1);
    }
    profilingEnd(vcmd);
}

void
VirtualGPU::submitMapMemory(amd::MapMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd, true);

    pal::Memory* memory = dev().getGpuMemory(&vcmd.memory());

    // Save map info for unmap operation
    memory->saveMapInfo(vcmd.mapPtr(), vcmd.origin(), vcmd.size(),
        vcmd.mapFlags(), vcmd.isEntireMemory());

    // If we have host memory, use it
    if ((memory->owner()->getHostMem() != nullptr) && memory->isDirectMap()) {
        if (!memory->isHostMemDirectAccess()) {
            // Make sure GPU finished operation before
            // synchronization with the backing store
            memory->wait(*this);
        }

        // Target is the backing store, so just ensure that owner is up-to-date
        memory->owner()->cacheWriteBack();

        // Add memory to VA cache, so rutnime can detect direct access to VA
        dev().addVACache(memory);
    }
    else if (memory->isPersistentDirectMap()) {
        // Nothing to do here
    }
    else if (memory->mapMemory() != nullptr) {
        // Target is a remote resource, so copy
        assert(memory->mapMemory() != nullptr);
        if (vcmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
            amd::Coord3D dstOrigin(0, 0, 0);
            if (memory->desc().buffer_) {
                if (!blitMgr().copyBuffer(*memory,
                    *memory->mapMemory(), vcmd.origin(), vcmd.origin(),
                    vcmd.size(), vcmd.isEntireMemory())) {
                    LogError("submitMapMemory() - copy failed");
                    vcmd.setStatus(CL_MAP_FAILURE);
                }
            }
            else if ((vcmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
                amd::Memory* bufferFromImage = nullptr;
                Memory* memoryBuf = memory;
                amd::Coord3D    origin(vcmd.origin()[0]);
                amd::Coord3D    size(vcmd.size()[0]);
                size_t  elemSize =
                    vcmd.memory().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0]   *= elemSize;

                bufferFromImage = createBufferFromImage(vcmd.memory());
                if (nullptr == bufferFromImage) {
                    LogError("We should not fail buffer creation from image_buffer!");
                }
                else {
                    bufferFromImage->setVirtualDevice(this);
                    memoryBuf = dev().getGpuMemory(bufferFromImage);
                }
                if (!blitMgr().copyBuffer(*memoryBuf,
                    *memory->mapMemory(), origin, dstOrigin,
                    size, vcmd.isEntireMemory())) {
                    LogError("submitMapMemory() - copy failed");
                    vcmd.setStatus(CL_MAP_FAILURE);
                }
                if (nullptr != bufferFromImage) {
                    bufferFromImage->release();
                }
            }
            else {
                // Validate if it's a view for a map of mip level
                if (vcmd.memory().parent() != nullptr) {
                    amd::Image* amdImage = vcmd.memory().parent()->asImage();
                    if ((amdImage != nullptr) && (amdImage->getMipLevels() > 1)) {
                        // Save map write info in the parent object
                        dev().getGpuMemory(amdImage)->saveMapInfo(vcmd.mapPtr(),
                            vcmd.origin(), vcmd.size(),
                            vcmd.mapFlags(), vcmd.isEntireMemory(),
                            vcmd.memory().asImage());
                    }
                }
                if (!blitMgr().copyImageToBuffer(*memory,
                    *memory->mapMemory(), vcmd.origin(), dstOrigin,
                    vcmd.size(), vcmd.isEntireMemory())) {
                    LogError("submitMapMemory() - copy failed");
                    vcmd.setStatus(CL_MAP_FAILURE);
                }
            }
        }
    }
    else {
        LogError("Unhandled map!");
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitUnmapMemory(amd::UnmapMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    pal::Memory* memory = dev().getGpuMemory(&vcmd.memory());
    amd::Memory* owner = memory->owner();
    bool    unmapMip = false;
    const device::Memory::WriteMapInfo* writeMapInfo =
        memory->writeMapInfo(vcmd.mapPtr());
    if (nullptr == writeMapInfo) {
        LogError("Unmap without map call");
        return;
    }
    profilingBegin(vcmd, true);

    // Check if image is a mipmap and assign a saved view
    amd::Image* amdImage = owner->asImage();
    if ((amdImage != nullptr) && (amdImage->getMipLevels() > 1) &&
        (writeMapInfo->baseMip_ != nullptr)) {
        // Assign mip level view
        amdImage = writeMapInfo->baseMip_;
        // Clear unmap flags from the parent image
        memory->clearUnmapInfo(vcmd.mapPtr());
        memory = dev().getGpuMemory(amdImage);
        unmapMip = true;
        writeMapInfo = memory->writeMapInfo(vcmd.mapPtr());
    }

    // We used host memory
    if ((owner->getHostMem() != nullptr) && memory->isDirectMap()) {
        if (writeMapInfo->isUnmapWrite() && !owner->usesSvmPointer()) {
            // Target is the backing store, so sync
            owner->signalWrite(nullptr);
            memory->syncCacheFromHost(*this);
        }
        // Remove memory from VA cache
        dev().removeVACache(memory);
    }
    // data check was added for persistent memory that failed to get aperture
    // and therefore are treated like a remote resource
    else if (memory->isPersistentDirectMap() && (memory->data() != nullptr)) {
        memory->unmap(this);
    }
    else if (memory->mapMemory() != nullptr) {
        if (writeMapInfo->isUnmapWrite()) {
            amd::Coord3D srcOrigin(0, 0, 0);
            // Target is a remote resource, so copy
            assert(memory->mapMemory() != nullptr);
            if (memory->desc().buffer_) {
                if (!blitMgr().copyBuffer(
                    *memory->mapMemory(), *memory,
                    writeMapInfo->origin_,
                    writeMapInfo->origin_,
                    writeMapInfo->region_,
                    writeMapInfo->isEntire())) {
                    LogError("submitUnmapMemory() - copy failed");
                    vcmd.setStatus(CL_OUT_OF_RESOURCES);
                }
            }
            else if ((vcmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
                amd::Memory* bufferFromImage = nullptr;
                Memory* memoryBuf = memory;
                amd::Coord3D    origin(writeMapInfo->origin_[0]);
                amd::Coord3D    size(writeMapInfo->region_[0]);
                size_t  elemSize =
                    vcmd.memory().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0]   *= elemSize;

                bufferFromImage = createBufferFromImage(vcmd.memory());
                if (nullptr == bufferFromImage) {
                    LogError("We should not fail buffer creation from image_buffer!");
                }
                else {
                    bufferFromImage->setVirtualDevice(this);
                    memoryBuf = dev().getGpuMemory(bufferFromImage);
                }
                if (!blitMgr().copyBuffer(
                    *memory->mapMemory(), *memoryBuf,
                    srcOrigin, origin, size,
                    writeMapInfo->isEntire())) {
                    LogError("submitUnmapMemory() - copy failed");
                    vcmd.setStatus(CL_OUT_OF_RESOURCES);
                }
                if (nullptr != bufferFromImage) {
                    bufferFromImage->release();
                }
            }
            else {
                if (!blitMgr().copyBufferToImage(
                    *memory->mapMemory(), *memory,
                    srcOrigin,
                    writeMapInfo->origin_,
                    writeMapInfo->region_,
                    writeMapInfo->isEntire())) {
                    LogError("submitUnmapMemory() - copy failed");
                    vcmd.setStatus(CL_OUT_OF_RESOURCES);
                }
            }
        }
    }
    else {
        LogError("Unhandled unmap!");
        vcmd.setStatus(CL_INVALID_VALUE);
    }

    // Clear unmap flags
    memory->clearUnmapInfo(vcmd.mapPtr());

    // Release a view for a mipmap map
    if (unmapMip) {
        amdImage->release();
    }
    profilingEnd(vcmd);
}

bool
VirtualGPU::fillMemory(cl_command_type type, amd::Memory* amdMemory, const void* pattern,
                       size_t patternSize, const amd::Coord3D& origin, const amd::Coord3D& size)
{
    pal::Memory* memory = dev().getGpuMemory(amdMemory);
    bool    entire = amdMemory->isEntirelyCovered(origin, size);

    // Synchronize memory from host if necessary
    device::Memory::SyncFlags syncFlags;
    syncFlags.skipEntire_ = entire;
    memory->syncCacheFromHost(*this, syncFlags);

    bool result = false;
    amd::Memory* bufferFromImage = nullptr;
    float fillValue[4];

    // Force fill buffer for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_FILL_IMAGE) &&
        (amdMemory->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        bufferFromImage = createBufferFromImage(*amdMemory);
        if (nullptr == bufferFromImage) {
            LogError("We should not fail buffer creation from image_buffer!");
        }
        else {
            type = CL_COMMAND_FILL_BUFFER;
            bufferFromImage->setVirtualDevice(this);
            memory = dev().getGpuMemory(bufferFromImage);
        }
    }

    // Find the the right fill operation
    switch (type) {
    case CL_COMMAND_FILL_BUFFER :
    case CL_COMMAND_SVM_MEMFILL : {
        amd::Coord3D    realOrigin(origin[0]);
        amd::Coord3D    realSize(size[0]);
        // Reprogram fill parameters if it's an IMAGE1D_BUFFER object
        if (nullptr != bufferFromImage) {
            size_t  elemSize =
                amdMemory->asImage()->getImageFormat().getElementSize();
            realOrigin.c[0] *= elemSize;
            realSize.c[0]   *= elemSize;
            memset(fillValue, 0, sizeof(fillValue));
            amdMemory->asImage()->getImageFormat().formatColor(pattern, fillValue);
            pattern = fillValue;
            patternSize = elemSize;
        }
        result = blitMgr().fillBuffer(*memory, pattern,
            patternSize, realOrigin, realSize, amdMemory->isEntirelyCovered(origin, size));
        if (nullptr != bufferFromImage) {
            bufferFromImage->release();
        }
    }
        break;
    case CL_COMMAND_FILL_IMAGE:
        result = blitMgr().fillImage(*memory, pattern,
            origin, size, amdMemory->isEntirelyCovered(origin, size));
        break;
    default:
        LogError("Unsupported command type for FillMemory!");
        break;
    }

    if (!result) {
        LogError("fillMemory failed!");
        return false;
    }

    // Mark this as the most-recently written cache of the destination
    amdMemory->signalWrite(&gpuDevice_);
    return true;
}

void
VirtualGPU::submitFillMemory(amd::FillMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd, true);

    if (!fillMemory(vcmd.type(), &vcmd.memory(),vcmd.pattern(),
        vcmd.patternSize(), vcmd.origin(), vcmd.size())) {
        vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd, true);

    //no op for FGS supported device
    if (!dev().isFineGrainedSystem()) {
        // Make sure we have memory for the command execution
        pal::Memory* memory = dev().getGpuMemory(vcmd.getSvmMem());

        memory->saveMapInfo(vcmd.svmPtr(), vcmd.origin(), vcmd.size(),
            vcmd.mapFlags(), vcmd.isEntireMemory());

        if (memory->mapMemory() != nullptr) {
            if (vcmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
                assert(memory->desc().buffer_ && "SVM memory can't be an image");
                if (!blitMgr().copyBuffer(*memory, *memory->mapMemory(),
                    vcmd.origin(), vcmd.origin(), vcmd.size(), vcmd.isEntireMemory())) {
                    LogError("submitSVMMapMemory() - copy failed");
                    vcmd.setStatus(CL_MAP_FAILURE);
                }
            }
        }
        else {
            LogError("Unhandled svm map!");
        }
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());
    profilingBegin(vcmd, true);

    //no op for FGS supported device
    if (!dev().isFineGrainedSystem()) {
        pal::Memory* memory = dev().getGpuMemory(vcmd.getSvmMem());
        const device::Memory::WriteMapInfo* writeMapInfo =
            memory->writeMapInfo(vcmd.svmPtr());

        if (memory->mapMemory() != nullptr) {
            if (writeMapInfo->isUnmapWrite()) {
                amd::Coord3D srcOrigin(0, 0, 0);
                // Target is a remote resource, so copy
                assert(memory->desc().buffer_ && "SVM memory can't be an image");
                if (!blitMgr().copyBuffer(*memory->mapMemory(), *memory,
                    writeMapInfo->origin_, writeMapInfo->origin_,
                    writeMapInfo->region_, writeMapInfo->isEntire())) {
                    LogError("submitSvmUnmapMemory() - copy failed");
                    vcmd.setStatus(CL_OUT_OF_RESOURCES);
                }
            }
        }
        memory->clearUnmapInfo(vcmd.svmPtr());
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd, true);

    if (!dev().isFineGrainedSystem()) {
        size_t patternSize = vcmd.patternSize();
        size_t fillSize = patternSize * vcmd.times();
        size_t offset = 0;
        amd::Memory* dstMemory = amd::SvmManager::FindSvmBuffer(vcmd.dst());
        assert(dstMemory&&"No svm Buffer to fill with!");
        offset = reinterpret_cast<uintptr_t>(vcmd.dst())
            - reinterpret_cast<uintptr_t>(dstMemory->getSvmPtr());
        assert((offset >= 0) && "wrong svm ptr to fill with!");

        pal::Memory* memory = dev().getGpuMemory(dstMemory);

        amd::Coord3D    origin(offset, 0, 0);
        amd::Coord3D    size(fillSize, 1, 1);
        assert((dstMemory->validateRegion(origin, size)) && "The incorrect fill size!");

        if (!fillMemory(vcmd.type(), dstMemory, vcmd.pattern(),
            vcmd.patternSize(), origin, size)) {
            vcmd.setStatus(CL_INVALID_OPERATION);
        }
    }
    else {
        // for FGS capable device, fill CPU memory directly
        amd::SvmBuffer::memFill(vcmd.dst(), vcmd.pattern(), vcmd.patternSize(), vcmd.times());
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd, true);

    std::vector<amd::Memory*>::const_iterator itr;
    for (itr = vcmd.memObjects().begin(); itr != vcmd.memObjects().end(); itr++) {
        // Find device memory
        pal::Memory* memory = dev().getGpuMemory(*itr);

        if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_HOST) {
            memory->mgpuCacheWriteBack();
        }
        else if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) {
            // Synchronize memory from host if necessary.
            // The sync function will perform memory migration from
            // another device if necessary
            device::Memory::SyncFlags syncFlags;
            memory->syncCacheFromHost(*this, syncFlags);
        }
        else {
            LogWarning("Unknown operation for memory migration!");
        }
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& vcmd)
{
    // in-order semantics: previous commands need to be done before we start
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);
    std::vector<void*>& svmPointers = vcmd.svmPointers();
    if (vcmd.pfnFreeFunc() == nullptr) {
        // pointers allocated using clSVMAlloc
        for (cl_uint i = 0; i < svmPointers.size(); i++) {
            dev().svmFree(svmPointers[i]);
        }
    }
    else {
        vcmd.pfnFreeFunc()(as_cl(vcmd.queue()->asCommandQueue()), svmPointers.size(),
                static_cast<void**>(&(svmPointers[0])), vcmd.userData());
    }
    profilingEnd(vcmd);
}

void
VirtualGPU::findIterations(
    const amd::NDRangeContainer& sizes,
    const amd::NDRange&   local,
    amd::NDRange&   groups,
    amd::NDRange&   remainder,
    size_t&         extra)
{
    size_t  dimensions = sizes.dimensions();

    if (cal()->iterations_ > 1) {
        size_t  iterations = cal()->iterations_;
        cal_.iterations_ = 1;

        // Find the total amount of all groups
        groups = sizes.global() / local;
        if (dev().settings().partialDispatch_) {
            for (uint j = 0; j < dimensions; ++j) {
                if ((sizes.global()[j] % local[j]) != 0) {
                    groups[j]++;
                }
            }
        }

        // Calculate the real number of required iterations and
        // the workgroup size of each iteration
        for (int j = (dimensions - 1); j >= 0; --j) {
            // Find possible size of each iteration
            size_t tmp = (groups[j] / iterations);
            // Make sure the group size is more than 1
            if (tmp > 0) {
                remainder = groups;
                remainder[j] = (groups[j] % tmp);

                extra = ((groups[j] / tmp) +
                    // Check for the remainder
                    ((remainder[j] != 0) ? 1 : 0));
                // Recalculate the number of iterations
                cal_.iterations_ *= extra;
                if (remainder[j] == 0) {
                    extra = 0;
                }
                groups[j] = tmp;
                break;
            }
            else {
                iterations = ((iterations / groups[j]) +
                    (((iterations % groups[j]) != 0) ? 1 : 0));
                cal_.iterations_ *= groups[j];
                groups[j] = 1;
            }
        }
    }
}

void
VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);

    // Submit kernel to HW
    if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(), false,
                              &vcmd.event())) {
        vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
}

bool
VirtualGPU::submitKernelInternal(
    const amd::NDRangeContainer& sizes,
    const amd::Kernel&  kernel,
    const_address parameters,
    bool    nativeMem,
    amd::Event* enqueueEvent)
{
    uint64_t    vmParentWrap = 0;
    uint64_t    vmDefQueue = 0;
    amd::DeviceQueue*  defQueue = kernel.program().context().defDeviceQueue(dev());
    VirtualGPU*  gpuDefQueue = nullptr;
    amd::HwDebugManager * dbgManager = dev().hwDebugMgr();

    // Get the HSA kernel object
    const HSAILKernel& hsaKernel =
        static_cast<const HSAILKernel&>(*(kernel.getDeviceKernel(dev())));
    std::vector<const Memory*>    memList;

    bool printfEnabled = (hsaKernel.printfInfo().size() > 0) ? true:false;
    if (!printfDbgHSA().init(*this, printfEnabled )) {
        LogError( "Printf debug buffer initialization failed!");
        return false;
    }

    // Check memory dependency and SVM objects
    if (!processMemObjectsHSA(kernel, parameters, nativeMem, &memList)) {
        LogError("Wrong memory objects!");
        return false;
    }

    if (hsaKernel.dynamicParallelism()) {
        if (nullptr == defQueue) {
            LogError("Default device queue wasn't allocated");
            return false;
        }
        else {
            if (dev().settings().useDeviceQueue_) {
                gpuDefQueue = static_cast<VirtualGPU*>(defQueue->vDev());
                if (gpuDefQueue->hwRing() == hwRing()) {
                    LogError("Can't submit the child kernels to the same HW ring as the host queue!");
                    return false;
                }
            }
            else {
                createVirtualQueue(defQueue->size());
                gpuDefQueue = this;
            }
        }
        vmDefQueue = gpuDefQueue->virtualQueue_->vmAddress();

        // Add memory handles before the actual dispatch
        memList.push_back(gpuDefQueue->virtualQueue_);
        memList.push_back(gpuDefQueue->schedParams_);
        memList.push_back(hsaKernel.prog().kernelTable());
        gpuDefQueue->writeVQueueHeader(*this,
            hsaKernel.prog().kernelTable()->vmAddress());
    }

    //  setup the storage for the memory pointers of the kernel parameters
    uint numParams = kernel.signature().numParameters();
    if (dbgManager) {
        dbgManager->allocParamMemList(numParams);
    }

    size_t newOffset[3] = {0, 0, 0};
    size_t newGlobalSize[3] = {0, 0, 0};

    int dim = -1;
    int iteration = 1;
    size_t globalStep = 0;
    for (uint i = 0; i < sizes.dimensions(); i++) {
        newGlobalSize[i] = sizes.global()[i];
        newOffset[i] = sizes.offset()[i];
    }
    // Check if it is blit kernel. If it is, then check if split is needed.
    if (hsaKernel.isInternalKernel()) {
        // Calculate new group size for each submission
        for (uint i = 0; i < sizes.dimensions(); i++) {
            if (sizes.global()[i] > static_cast<size_t>(0xffffffff)) {
                dim = i;
                iteration = sizes.global()[i] / 0xC0000000
                            + ((sizes.global()[i] % 0xC0000000) ? 1: 0);
                globalStep = (sizes.global()[i] / sizes.local()[i]) / iteration
                             * sizes.local()[dim];
                break;
            }
        }
    }

    for (int j = 0; j < iteration; j++) {
        // Reset global size for dimension dim if split is needed
        if (dim != -1) {
            newOffset[dim] = sizes.offset()[dim] + globalStep * j;
            if (((newOffset[dim] + globalStep) < sizes.global()[dim]) &&
                (j != (iteration - 1))) {
                newGlobalSize[dim] = globalStep;
            }
            else {
                newGlobalSize[dim] = sizes.global()[dim] - newOffset[dim];
            }
        }

        amd::NDRangeContainer  tmpSizes(sizes.dimensions(),
            &newOffset[0], &newGlobalSize[0],
            &(const_cast<amd::NDRangeContainer&>(sizes).local()[0]));

        // Program the kernel arguments for the GPU execution
        hsa_kernel_dispatch_packet_t*  aqlPkt =
            hsaKernel.loadArguments(*this, kernel, tmpSizes, parameters, nativeMem,
            vmDefQueue, &vmParentWrap, memList);
        if (nullptr == aqlPkt) {
            LogError("Couldn't load kernel arguments");
            return false;
        }

        const Device::ScratchBuffer* scratch = nullptr;
        // Check if the device allocated more registers than the old setup
        if (hsaKernel.workGroupInfo()->scratchRegs_ > 0) {
            scratch = dev().scratch(hwRing());
            memList.push_back(scratch->memObj_);
        }

        // Add GSL handle to the memory list for VidMM
        for (uint i = 0; i < memList.size(); ++i) {
            addVmMemory(memList[i]);
        }

        // HW Debug for the kernel?
        HwDbgKernelInfo kernelInfo;
        HwDbgKernelInfo *pKernelInfo = nullptr;

        if (dbgManager) {
            buildKernelInfo(hsaKernel, aqlPkt, kernelInfo, enqueueEvent);
            pKernelInfo = &kernelInfo;
        }

        GpuEvent    gpuEvent;

        // Run AQL dispatch in HW
        eventBegin(MainEngine);
        if (nullptr == scratch) {
            iCmd()->CmdDispatchAql(aqlPkt, 0, 0, 0,
                hsaKernel.cpuAqlCode(), hsaQueueMem_->vmAddress(), hsaKernel.getWavesPerSH(this));
        }
        else {
            iCmd()->CmdDispatchAql(aqlPkt, scratch->memObj_->vmAddress(),
                scratch->size_, scratch->offset_,
                hsaKernel.cpuAqlCode(), hsaQueueMem_->vmAddress(), hsaKernel.getWavesPerSH(this));
        }
        eventEnd(MainEngine, gpuEvent);

        if (dbgManager && (nullptr != dbgManager->postDispatchCallBackFunc())) {
            dbgManager->executePostDispatchCallBack();
        }

        if (hsaKernel.dynamicParallelism()) {
            // Make sure exculsive access to the device queue
            amd::ScopedLock(defQueue->lock());

            if (GPU_PRINT_CHILD_KERNEL != 0) {
                waitForEvent(&gpuEvent);

                AmdAqlWrap* wraps =  (AmdAqlWrap*)(&((AmdVQueueHeader*)gpuDefQueue->virtualQueue_->data())[1]);
                uint p = 0;
                for (uint i = 0; i < gpuDefQueue->vqHeader_->aql_slot_num; ++i) {
                    if (wraps[i].state != 0) {
                        uint j;
                        if (p == GPU_PRINT_CHILD_KERNEL) {
                            break;
                        }
                        p++;
                        std::stringstream print;
                        print.flags(std::ios::right | std::ios_base::hex | std::ios_base::uppercase);
                        print << "Slot#: "  << i << "\n";
                        print << "\tenqueue_flags: "  << wraps[i].enqueue_flags   << "\n";
                        print << "\tcommand_id: "     << wraps[i].command_id      << "\n";
                        print << "\tchild_counter: "  << wraps[i].child_counter   << "\n";
                        print << "\tcompletion: "     << wraps[i].completion      << "\n";
                        print << "\tparent_wrap: "    << wraps[i].parent_wrap     << "\n";
                        print << "\twait_list: "      << wraps[i].wait_list       << "\n";
                        print << "\twait_num: "       << wraps[i].wait_num        << "\n";
                        uint offsEvents = wraps[i].wait_list -
                            gpuDefQueue->virtualQueue_->vmAddress();
                        size_t* events = reinterpret_cast<size_t*>(
                            gpuDefQueue->virtualQueue_->data() + offsEvents);
                        for (j = 0; j < wraps[i].wait_num; ++j) {
                            uint offs = static_cast<uint64_t>(events[j]) -
                                gpuDefQueue->virtualQueue_->vmAddress();
                            AmdEvent* eventD = (AmdEvent*)(gpuDefQueue->virtualQueue_->data() + offs);
                            print << "Wait Event#: " << j << "\n";
                            print << "\tState: " << eventD->state <<
                                     "; Counter: " << eventD->counter << "\n";
                        }
                        print << "WorkGroupSize[ " << wraps[i].aql.workgroup_size_x << ", ";
                        print << wraps[i].aql.workgroup_size_y << ", ";
                        print << wraps[i].aql.workgroup_size_z << "]\n";
                        print << "GridSize[ " << wraps[i].aql.grid_size_x << ", ";
                        print << wraps[i].aql.grid_size_y << ", ";
                        print << wraps[i].aql.grid_size_z << "]\n";

                        uint64_t* kernels = (uint64_t*)(
                            const_cast<Memory*>(hsaKernel.prog().kernelTable())->map(this));
                        for (j = 0; j < hsaKernel.prog().kernels().size(); ++j) {
                            if (kernels[j] == wraps[i].aql.kernel_object) {
                                break;
                            }
                        }
                        const_cast<Memory*>(hsaKernel.prog().kernelTable())->unmap(this);
                        HSAILKernel* child = nullptr;
                        for (auto it = hsaKernel.prog().kernels().begin();
                             it != hsaKernel.prog().kernels().end(); ++it) {
                            if (j == static_cast<HSAILKernel*>(it->second)->index()) {
                                child = static_cast<HSAILKernel*>(it->second);
                            }
                        }
                        if (child == nullptr) {
                            printf("Error: couldn't find child kernel!\n");
                            continue;
                        }
                        const uint64_t kernarg_address =
                          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wraps[i].aql.kernarg_address));
                        uint offsArg = kernarg_address -
                            gpuDefQueue->virtualQueue_->vmAddress();
                        address argum = gpuDefQueue->virtualQueue_->data() + offsArg;
                        print << "Kernel: " << child->name() << "\n";
                        static const char* Names[HSAILKernel::MaxExtraArgumentsNum] = {
                        "Offset0: ", "Offset1: ","Offset2: ","PrintfBuf: ", "VqueuePtr: ", "AqlWrap: "};
                        for (j = 0; j < child->extraArgumentsNum(); ++j) {
                            print << "\t" << Names[j] << *(size_t*)argum;
                            print << "\n";
                            argum += sizeof(size_t);
                        }
                        for (j = 0; j < child->numArguments(); ++j) {
                            print << "\t" << child->argument(j)->name_ << ": ";
                            for (int s = child->argument(j)->size_ - 1; s >= 0; --s) {
                                print.width(2);
                                print.fill('0');
                                print << (uint32_t)(argum[s]);
                            }
                            argum += child->argument(j)->size_;
                            print << "\n";
                        }
                        printf("%s", print.str().c_str());
                    }
                }
            }

            if (!dev().settings().useDeviceQueue_) {
                // Add the termination handshake to the host queue
                eventBegin(MainEngine);
                //iCmd()->CmdVirtualQueueHandshake(*gpuDefQueue->schedParams_->iMem(),
                //    vmParentWrap + offsetof(AmdAqlWrap, state), AQL_WRAP_DONE,
                //    vmParentWrap + offsetof(AmdAqlWrap, child_counter),
                //    0, dev().settings().useDeviceQueue_);
                eventEnd(MainEngine, gpuEvent);
            }

            // Get the global loop start before the scheduler
            //Pal::gpusize loopStart = gpuDefQueue->iCmd()->CmdVirtualQueueDispatcherStart();
            //static_cast<KernelBlitManager&>(gpuDefQueue->blitMgr()).runScheduler(
            //    *gpuDefQueue->virtualQueue_,
            //    *gpuDefQueue->schedParams_, gpuDefQueue->schedParamIdx_,
            //    gpuDefQueue->vqHeader_->aql_slot_num / (DeviceQueueMaskSize * maskGroups_));
            const static bool FlushL2 = true;
            gpuDefQueue->flushCUCaches(FlushL2);

            // Get the address of PM4 template and add write it to params
            //! @note DMA flush must not occur between patch and the scheduler
            Pal::gpusize patchStart = 0;
            //Pal::gpusize patchStart = gpuDefQueue->iCmd()->CmdVirtualQueueDispatcherStart();
            // Program parameters for the scheduler
            SchedulerParam* param = &reinterpret_cast<SchedulerParam*>
                (gpuDefQueue->schedParams_->data())[gpuDefQueue->schedParamIdx_];
            param->signal = 1;
            // Scale clock to 1024 to avoid 64 bit div in the scheduler
            param->eng_clk = (1000 * 1024) / dev().info().maxClockFrequency_;
            param->hw_queue = patchStart + sizeof(uint32_t)/* Rewind packet*/;
            param->hsa_queue = gpuDefQueue->hsaQueueMem()->vmAddress();
            param->releaseHostCP = 0;
            param->parentAQL = vmParentWrap;
            param->dedicatedQueue = dev().settings().useDeviceQueue_;
            param->useATC = dev().settings().svmFineGrainSystem_;

            // Fill the scratch buffer information
            if (hsaKernel.prog().maxScratchRegs() > 0) {
                pal::Memory* scratchBuf = dev().scratch(gpuDefQueue->hwRing())->memObj_;
                param->scratchSize = scratchBuf->size();
                param->scratch = scratchBuf->vmAddress();
                param->numMaxWaves = 32 * dev().info().maxComputeUnits_;
                param->scratchOffset = dev().scratch(gpuDefQueue->hwRing())->offset_;
                memList.push_back(scratchBuf);
            }
            else {
                param->numMaxWaves = 0;
                param->scratchSize = 0;
                param->scratch = 0;
                param->scratchOffset = 0;
            }

            // Add all kernels in the program to the mem list.
            //! \note Runtime doesn't know which one will be called
            hsaKernel.prog().fillResListWithKernels(memList);

            // Add GPU memory handle to the memory list for VidMM
            for (uint i = 0; i < memList.size(); ++i) {
                gpuDefQueue->addVmMemory(memList[i]);
            }

            Pal::gpusize  signalAddr = gpuDefQueue->schedParams_->vmAddress() +
                gpuDefQueue->schedParamIdx_ * sizeof(SchedulerParam);
            gpuDefQueue->eventBegin(MainEngine);
            //gpuDefQueue->iCmd()->CmdVirtualQueueDispatcherEnd(
            //    signalAddr, loopStart, gpuDefQueue->vqHeader_->aql_slot_num /
            //    (DeviceQueueMaskSize * maskGroups_));
            // Note: Device enqueue can't have extra commands after INDIRECT_BUFFER call.
            // Thus TS command for profiling has to follow in the next CB.
            constexpr bool ForceSubmitFirst = true;
            gpuDefQueue->eventEnd(MainEngine, gpuEvent, ForceSubmitFirst);

            // Set GPU event for the used resources
            for (uint i = 0; i < memList.size(); ++i) {
                memList[i]->setBusy(*gpuDefQueue, gpuEvent);
            }

            if (dev().settings().useDeviceQueue_) {
                // Add the termination handshake to the host queue
                eventBegin(MainEngine);
                //iCmd()->CmdVirtualQueueHandshake(*gpuDefQueue->schedParams_->iMem(),
                //    vmParentWrap + offsetof(AmdAqlWrap, state), AQL_WRAP_DONE,
                //    vmParentWrap + offsetof(AmdAqlWrap, child_counter),
                //    signalAddr, dev().settings().useDeviceQueue_);
                eventEnd(MainEngine, gpuEvent);
            }

            ++gpuDefQueue->schedParamIdx_ %=
                gpuDefQueue->schedParams_->size() / sizeof(SchedulerParam);
            //! \todo optimize the wrap around
            if (gpuDefQueue->schedParamIdx_ == 0) {
                gpuDefQueue->schedParams_->wait(*gpuDefQueue);
            }
        }

        // Set GPU event for the used resources
        for (uint i = 0; i < memList.size(); ++i) {
            memList[i]->setBusy(*this, gpuEvent);
        }

        // Update the global GPU event
        setGpuEvent(gpuEvent);

        if (!printfDbgHSA().output(*this, printfEnabled, hsaKernel.printfInfo())) {
            LogError("Couldn't read printf data from the buffer!\n");
            return false;
        }
    }

    return true;
}

void
VirtualGPU::submitNativeFn(amd::NativeFnCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    Unimplemented();    //!< @todo: Unimplemented
}

void
VirtualGPU::submitMarker(amd::Marker& vcmd)
{
    //!@note runtime doesn't need to lock this command on execution

    if (vcmd.waitingEvent() != nullptr) {
        bool foundEvent = false;

        // Loop through all outstanding command batches
        while (!cbList_.empty()) {
            CommandBatchList::const_iterator it = cbList_.begin();
            // Wait for completion
            foundEvent = awaitCompletion(*it, vcmd.waitingEvent());
            // Release a command batch
            delete *it;
            // Remove command batch from the list
            cbList_.pop_front();
            // Early exit if we found a command
            if (foundEvent) break;
        }

        // Event should be in the current command batch
        if (!foundEvent) {
            state_.forceWait_ = true;
        }
        // If we don't have any more batches, then assume GPU is idle
        else if (cbList_.empty()) {
            dmaFlushMgmt_.resetCbWorkload(dev());
        }
    }
}

GpuEvent*
VirtualGPU::getGpuEvent(Pal::IGpuMemory* iMem)
{
    return &gpuEvents_[iMem];
}

void 
VirtualGPU::assignGpuEvent(Pal::IGpuMemory* iMem, GpuEvent gpuEvent)
{ 
    auto it = gpuEvents_.find(iMem);

    if (it != gpuEvents_.end()) {
        it->second = gpuEvent;
    }
    else {
        gpuEvents_[iMem] = gpuEvent;
    }
}

void
VirtualGPU::releaseMemory(Pal::IGpuMemory* iMem, bool wait)
{
    auto it = gpuEvents_.find(iMem);
    //! @note if there is no wait, then it's a view release
    if (wait &&  (it != gpuEvents_.end())) {
        waitForEvent(&it->second);
        queues_[MainEngine]->removeCmdMemRef(iMem);
        queues_[SdmaEngine]->removeCmdMemRef(iMem);
        gpuEvents_.erase(it);
    }
}

void
VirtualGPU::submitPerfCounter(amd::PerfCounterCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    const amd::PerfCounterCommand::PerfCounterList counters = vcmd.getCounters();

    // Create performance experiment
    Pal::PerfExperimentCreateInfo   createInfo = {};
    createInfo.optionValues.sqShaderMask = Pal::PerfShaderMaskCs;

    PalCounterReference* palRef = PalCounterReference::Create(*this, createInfo);
    if (palRef == nullptr) {
        LogError("We failed to allocate memory for the GPU perfcounter");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
    }

    bool newExperiment = false;

    for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
        amd::PerfCounter* amdCounter =
            static_cast<amd::PerfCounter*>(counters[i]);
        const PerfCounter* counter =
            static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());

        // Make sure we have a valid gpu performance counter
        if (nullptr == counter) {
            amd::PerfCounter::Properties prop = amdCounter->properties();
            PerfCounter* gpuCounter = new PerfCounter(
                gpuDevice_,
                *this,
                prop[CL_PERFCOUNTER_GPU_BLOCK_INDEX],
                prop[CL_PERFCOUNTER_GPU_COUNTER_INDEX],
                prop[CL_PERFCOUNTER_GPU_EVENT_INDEX]);
            if (nullptr == gpuCounter) {
                LogError("We failed to allocate memory for the GPU perfcounter");
                vcmd.setStatus(CL_INVALID_OPERATION);
                return;
            }
            else if (gpuCounter->create(palRef)) {
                amdCounter->setDeviceCounter(gpuCounter);
                newExperiment = true;
            }
            else {
                LogPrintfError("We failed to allocate a perfcounter in CAL.\
                    Block: %d, counter: #d, event: %d",
                    gpuCounter->info()->blockIndex_,
                    gpuCounter->info()->counterIndex_,
                    gpuCounter->info()->eventIndex_);
                delete gpuCounter;
                vcmd.setStatus(CL_INVALID_OPERATION);
                return;
            }
            counter = gpuCounter;
        }
    }

    if (newExperiment) {
        palRef->finalize();
    }

    palRef->release();

    Pal::IPerfExperiment* palPerf = nullptr;
    for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
        amd::PerfCounter* amdCounter =
            static_cast<amd::PerfCounter*>(counters[i]);
        const PerfCounter* counter =
            static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());

        if (palPerf != counter->iPerf()) {
            palPerf = counter->iPerf();
            // Find the state and sends the command to PAL
            if (vcmd.getState() == amd::PerfCounterCommand::Begin) {
                iCmd()->CmdBeginPerfExperiment(palPerf);
            }
            else if (vcmd.getState() == amd::PerfCounterCommand::End) {
                GpuEvent event;
                eventBegin(MainEngine);
                iCmd()->CmdEndPerfExperiment(palPerf);
                eventEnd(MainEngine, event);
                setGpuEvent(event);
            }
            else {
                LogError("Unsupported performance counter state");
                vcmd.setStatus(CL_INVALID_OPERATION);
                return;
            }
        }
    }
}

void
VirtualGPU::submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(cmd);

    switch(cmd.type()) {
    case CL_COMMAND_THREAD_TRACE_MEM:
        {
            amd::ThreadTrace* amdThreadTrace = &cmd.getThreadTrace();
            ThreadTrace* threadTrace =
                static_cast<ThreadTrace*>(amdThreadTrace->getDeviceThreadTrace());
            Unimplemented();
/*
            if (threadTrace == nullptr) {
                gslQueryObject  gslThreadTrace;
                // Create a HW thread trace query object
                gslThreadTrace = cs()->createQuery(GSL_SHADER_TRACE_BYTES_WRITTEN);
                if (0 == gslThreadTrace) {
                    LogError("Failure in memory allocation for the GPU threadtrace");
                    cmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }
                CalThreadTraceReference* palRef = new CalThreadTraceReference(*this,gslThreadTrace);
                if (palRef == nullptr) {
                    LogError("Failure in memory allocation for the GPU threadtrace");
                    cmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }
                size_t seNum = amdThreadTrace->deviceSeNumThreadTrace();
                ThreadTrace* gpuThreadTrace = new ThreadTrace(
                    gpuDevice_,
                    *this,
                    seNum);
                if (nullptr == gpuThreadTrace) {
                    LogError("Failure in memory allocation for the GPU threadtrace");
                    cmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }
                if (gpuThreadTrace->create(palRef)) {
                    amdThreadTrace->setDeviceThreadTrace(gpuThreadTrace);
                }
                else {
                    LogError("Failure in memory allocation for the GPU threadtrace");
                    delete gpuThreadTrace;
                    cmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }
                threadTrace = gpuThreadTrace;
                palRef->release();
            }
            gslShaderTraceBufferObject* threadTraceBufferObjects = threadTrace->getThreadTraceBufferObjects();
            const size_t memObjSize = cmd.getMemoryObjectSize();
            const std::vector<amd::Memory*>& memObj = cmd.getMemList();
            size_t se = 0;
            for (std::vector<amd::Memory*>::const_iterator itMemObj = memObj.begin();itMemObj != memObj.end();++itMemObj,++se) {
                // Find GSL Mem Object
                Pal::IGpuMemory* gslMemObj = dev().getGpuMemory(*itMemObj)->iMem();

                // Bind GSL MemObject to the appropriate SE Thread Trace Buffer Object
                threadTraceBufferObjects[se]->attachMemObject(cs(), gslMemObj, 0, 0, memObjSize, se);
            }
*/
            break;
        }
    default:
        LogError("Unsupported command type for ThreadTraceMemObjects!");
        break;
    }
}

void
VirtualGPU::submitThreadTrace(amd::ThreadTraceCommand& cmd)
{
     // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(cmd);

    switch(cmd.type()) {
    case CL_COMMAND_THREAD_TRACE:
        {
            amd::ThreadTrace* amdThreadTrace =
                static_cast<amd::ThreadTrace*>(&cmd.getThreadTrace());
            ThreadTrace* threadTrace =
                static_cast<ThreadTrace*>(amdThreadTrace->getDeviceThreadTrace());

            // gpu thread trace object had to be generated prior to begin/end/pause/resume due
            // to ThreadTraceMemObjectsCommand execution
            if (threadTrace == nullptr) {
                return;
            }
            else {
                Unimplemented();
/*
                gslQueryObject  gslThreadTrace;
                gslThreadTrace = threadTrace->gslThreadTrace();
                uint32_t seNum = amdThreadTrace->deviceSeNumThreadTrace();

                // Find the state and sends the commands to GSL
                if (cmd.getState() == amd::ThreadTraceCommand::Begin) {
                    amd::ThreadTrace::ThreadTraceConfig* traceCfg =
                        static_cast<amd::ThreadTrace::ThreadTraceConfig*>(cmd.threadTraceConfig());
                    const gslErrorCode ec = gslThreadTrace->BeginQuery(cs(),
                        GSL_SHADER_TRACE_BYTES_WRITTEN, 0);
                    assert(ec == GSL_NO_ERROR);

                    for (uint32_t idx = 0; idx < seNum; ++idx) {
                        rs()->enableShaderTrace(cs(), idx, true);
                        rs()->setShaderTraceComputeUnit (idx, traceCfg->cu_);
                        rs()->setShaderTraceShaderArray (idx, traceCfg->sh_);
                        rs()->setShaderTraceSIMDMask    (idx, traceCfg->simdMask_);
                        rs()->setShaderTraceVmIdMask    (idx, traceCfg->vmIdMask_);
                        rs()->setShaderTraceTokenMask   (idx, traceCfg->tokenMask_);
                        rs()->setShaderTraceRegisterMask(idx, traceCfg->regMask_);
                        rs()->setShaderTraceIssueMask   (idx, traceCfg->instMask_);
                        rs()->setShaderTraceRandomSeed  (idx, traceCfg->randomSeed_);
                        rs()->setShaderTraceCaptureMode (idx, traceCfg->captureMode_);
                        rs()->setShaderTraceWrap        (idx, traceCfg->isWrapped_);
                        rs()->setShaderTraceUserData    (idx,
                            (traceCfg->isUserData_) ? traceCfg->userData_ : 0);
                    }
                }
                else if (cmd.getState() == amd::ThreadTraceCommand::End) {
                    for (uint32_t idx = 0; idx < seNum; ++idx) {
                        rs()->enableShaderTrace(cs(), idx, false);
                    }
                    gslThreadTrace->EndQuery(cs(), 0);
                }
                else if (cmd.getState() == amd::ThreadTraceCommand::Pause) {
                    for (uint32_t idx = 0; idx < seNum; ++idx) {
                        rs()->setShaderTraceIsPaused(cs(), idx, true);
                    }
                }
                else if (cmd.getState() == amd::ThreadTraceCommand::Resume) {
                    for (uint32_t idx = 0; idx < seNum; ++idx) {
                        rs()->setShaderTraceIsPaused(cs(), idx, false);
                    }
                }
*/
            }
            break;
        }
    default:
        LogError("Unsupported command type for ThreadTrace!");
        break;
    }
}

void
VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);

    for (std::vector<amd::Memory*>::const_iterator it = vcmd.getMemList().begin();
         it != vcmd.getMemList().end(); it++) {
        // amd::Memory object should never be nullptr
        assert(*it && "Memory object for interop is nullptr");
        pal::Memory* memory = dev().getGpuMemory(*it);

        // If resource is a shared copy of original resource, then
        // runtime needs to copy data from original resource
        (*it)->getInteropObj()->copyOrigToShared();

        // Check if OpenCL has direct access to the interop memory
        if (memory->interopType() == Memory::InteropDirectAccess) {
            continue;
        }

        // Does interop use HW emulation?
        if (memory->interopType() == Memory::InteropHwEmulation) {
            static const bool Entire  = true;
            amd::Coord3D    origin(0, 0, 0);
            amd::Coord3D    region(memory->size());

            // Synchronize the object
            if (!blitMgr().copyBuffer(*memory->interop(),
                *memory, origin, origin, region, Entire)) {
                LogError("submitAcquireExtObjects - Interop synchronization failed!");
                vcmd.setStatus(CL_INVALID_OPERATION);
                return;
            }
        }
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);

    for (std::vector<amd::Memory*>::const_iterator it = vcmd.getMemList().begin();
         it != vcmd.getMemList().end(); it++) {
        // amd::Memory object should never be nullptr
        assert(*it && "Memory object for interop is nullptr");
        pal::Memory* memory = dev().getGpuMemory(*it);

        // Check if we can use HW interop
        if (memory->interopType() == Memory::InteropHwEmulation) {
            static const bool Entire  = true;
            amd::Coord3D    origin(0, 0, 0);
            amd::Coord3D    region(memory->size());

            // Synchronize the object
            if (!blitMgr().copyBuffer(*memory, *memory->interop(),
                origin, origin, region, Entire)) {
                LogError("submitReleaseExtObjects interop synchronization failed!");
                vcmd.setStatus(CL_INVALID_OPERATION);
                return;
            }
        }
        else {
            if (memory->interopType() != Memory::InteropDirectAccess) {
                LogError("None interop release!");
            }
        }

        // If resource is a shared copy of original resource, then
        // runtime needs to copy data back to original resource
        (*it)->getInteropObj()->copySharedToOrig();
    }

    profilingEnd(vcmd);
}

void
VirtualGPU::submitSignal(amd::SignalCommand & vcmd)
{
    amd::ScopedLock lock(execution());
    profilingBegin(vcmd);
    pal::Memory* gpuMemory = dev().getGpuMemory(&vcmd.memory());
    Unimplemented();
/*
    if (vcmd.type() == CL_COMMAND_WAIT_SIGNAL_AMD) {
        uint64_t surfAddr = gpuMemory->iMem()->getPhysicalAddress(cs());
        uint64_t markerAddr = gpuMemory->iMem()->getMarkerAddress(cs());
        uint64_t markerOffset = markerAddr - surfAddr;
        cs()->p2pMarkerOp(gpuMemory->iMem(), vcmd.markerValue(),
            markerOffset, false);
    }
    else if (vcmd.type() == CL_COMMAND_WRITE_SIGNAL_AMD) {
        GpuEvent    gpuEvent;
        eventBegin(MainEngine);
        cs()->p2pMarkerOp(gpuMemory->iMem(), vcmd.markerValue(),  vcmd.markerOffset(), true);
        //! @todo We don't need flush if an event is tracked.
        cs()->Flush();
        eventEnd(MainEngine, gpuEvent);
        gpuMemory->setBusy(*this, gpuEvent);
        // Update the global GPU event
        setGpuEvent(gpuEvent);
    }
*/
    profilingEnd(vcmd);
}

void
VirtualGPU::submitMakeBuffersResident(amd::MakeBuffersResidentCommand & vcmd)
{
    amd::ScopedLock lock(execution());
    profilingBegin(vcmd);
    std::vector<amd::Memory*> memObjects = vcmd.memObjects();
    cl_uint numObjects = memObjects.size();
    Pal::IGpuMemory** pGpuMemObjects = new Pal::IGpuMemory*[numObjects];

    for(cl_uint i = 0; i < numObjects; ++i)
    {
        pal::Memory* gpuMemory = dev().getGpuMemory(memObjects[i]);
        pGpuMemObjects[i] = gpuMemory->iMem();
        gpuMemory->syncCacheFromHost(*this);
    }

    uint64_t* surfBusAddr = new uint64_t[numObjects];
    uint64_t* markerBusAddr = new uint64_t[numObjects];
    Unimplemented();
/*
    gslErrorCode res = cs()->makeBuffersResident(numObjects, pGpuMemObjects,
        surfBusAddr, markerBusAddr);
    if(res != GSL_NO_ERROR) {
        LogError("MakeBuffersResident failed");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        cl_bus_address_amd* busAddr = vcmd.busAddress();
        for(cl_uint i = 0; i < numObjects; ++i)
        {
            busAddr[i].surface_bus_address = surfBusAddr[i];
            busAddr[i].marker_bus_address = markerBusAddr[i];
        }
    }
*/
    delete[] pGpuMemObjects;
    delete[] surfBusAddr;
    delete[] markerBusAddr;
    profilingEnd(vcmd);
}


bool
VirtualGPU::awaitCompletion(CommandBatch* cb, const amd::Event* waitingEvent)
{
    bool found = false;
    amd::Command*   current;
    amd::Command*   head = cb->head_;

    // Make sure that profiling is enabled
    if (state_.profileEnabled_) {
        return profilingCollectResults(cb, waitingEvent);
    }
    // Mark the first command in the batch as running
    if (head != nullptr) {
        head->setStatus(CL_RUNNING);
    }
    else {
        return found;
    }

    // Wait for the last known GPU event
    waitEventLock(cb);

    while (nullptr != head) {
        current = head->getNext();
        if (head->status() == CL_SUBMITTED) {
            head->setStatus(CL_RUNNING);
            head->setStatus(CL_COMPLETE);
        }
        else if (head->status() == CL_RUNNING) {
            head->setStatus(CL_COMPLETE);
        }
        else if ((head->status() != CL_COMPLETE) && (current != nullptr)) {
            LogPrintfError("Unexpected command status - %d!", head->status());
        }

        // Check if it's a waiting command
        if (head == waitingEvent) {
            found = true;
        }

        head->release();
        head = current;
    }

    return found;
}

void
VirtualGPU::flush(amd::Command* list, bool wait)
{
    CommandBatch* cb = nullptr;
    bool    gpuCommand = false;

    for (uint i = 0; i < AllEngines; ++i) {
        if (cal_.events_[i].isValid()) {
            gpuCommand = true;
        }
    }

    // If the batch doesn't have any GPU command and the list is empty
    if (!gpuCommand && cbList_.empty()) {
        state_.forceWait_ = true;
    }

    // Insert the current batch into a list
    if (nullptr != list) {
        cb = new CommandBatch(list, cal()->events_, cal()->lastTS_);
    }

    {
        //! @todo: Check if really need a lock
        amd::ScopedLock lock(execution());
        for (uint i = 0; i < AllEngines; ++i) {
            flushDMA(i);
            // Reset event so we won't try to wait again,
            // if runtime didn't submit any commands
            //! @note: it's safe to invalidate events, since
            //! we already saved them with the batch creation step above
            cal_.events_[i].invalidate();
        }
    }

    // Mark last TS as nullptr, so runtime won't process empty batches with the old TS
    cal_.lastTS_ = nullptr;
    if (nullptr != cb) {
        cbList_.push_back(cb);
    }

    wait |= state_.forceWait_;
    // Loop through all outstanding command batches
    while (!cbList_.empty()) {
        CommandBatchList::const_iterator it = cbList_.begin();
        // Check if command batch finished without a wait
        bool    finished = true;
        for (uint i = 0; i < AllEngines; ++i) {
            finished &= isDone(&(*it)->events_[i]);
        }
        if (finished || wait) {
            // Wait for completion
            awaitCompletion(*it);
            // Release a command batch
            delete *it;
            // Remove command batch from the list
            cbList_.pop_front();
        }
        else {
            // Early exit if no finished
            break;
        }
    }
    state_.forceWait_ = false;
}

void
VirtualGPU::enableSyncedBlit() const
{
    return blitMgr_->enableSynchronization();
}

void
VirtualGPU::releaseMemObjects(bool scratch)
{
    for (GpuEvents::const_iterator it = gpuEvents_.begin();
            it != gpuEvents_.end(); ++it) {
        GpuEvent event = it->second;
        waitForEvent(&event);
        queues_[MainEngine]->removeCmdMemRef(const_cast<Pal::IGpuMemory*>(it->first));
        queues_[SdmaEngine]->removeCmdMemRef(const_cast<Pal::IGpuMemory*>(it->first));
    }

    gpuEvents_.clear();
}

void
VirtualGPU::setGpuEvent(
    GpuEvent    gpuEvent,
    bool        flush)
{
    cal_.events_[engineID_] = gpuEvent;

    // Flush current DMA buffer if requested
    if (flush) {
        flushDMA(engineID_);
    }
}

void
VirtualGPU::flushDMA(uint engineID)
{
    if (engineID == MainEngine) {
        // Clear memory dependency state, since runtime flushes compute
        // memoryDependency().clear();
        //!@todo Keep memory dependency alive even if we flush DMA,
        //! since only L2 cache is flushed in KMD frame,
        //! but L1 still has to be invalidated.
    }

    isDone(&cal_.events_[engineID]);
}

bool
VirtualGPU::waitAllEngines(CommandBatch* cb)
{
    uint i;
    GpuEvent*   events;    //!< GPU events for the batch

    // If command batch is nullptr then wait for the current
    if (nullptr == cb) {
        events = cal_.events_;
    }
    else {
        events = cb->events_;
    }

    bool earlyDone = true;
    // The first loop is to flush all engines and/or check if
    // engines are idle already
    for (i = 0; i < AllEngines; ++i) {
        earlyDone &= isDone(&events[i]);
    }

    // Release all transfer buffers on this command queue
    releaseXferWrite();

    // Rlease all pinned memory
    releasePinnedMem();

    // The second loop is to wait all engines
    for (i = 0; i < AllEngines; ++i) {
        waitForEvent(&events[i]);
    }

    return earlyDone;
}

void
VirtualGPU::waitEventLock(CommandBatch* cb)
{
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    bool earlyDone = waitAllEngines(cb);

    // Free resource cache if we have too many entries
    //! \note we do it here, when all engines are idle,
    // because Vista/Win7 idles GPU on a resource destruction
    static const size_t MinCacheEntries = 4096;
    dev().resourceCache().free(MinCacheEntries);

    // Find the timestamp object of the last command in the batch
    if (cb->lastTS_ != nullptr) {
        // If earlyDone is TRUE, then CPU didn't wait for GPU.
        // Thus the sync point between CPU and GPU is unclear and runtime
        // will use an older adjustment value to maintain the same timeline
        if (!earlyDone ||
            //! \note Workaround for APU(s).
            //! GPU-CPU timelines may go off too much, thus always
            //! force calibration with the last batch in the list
            (cbList_.size() <= 1) ||
            (readjustTimeGPU_ == 0)) {
            uint64_t    startTimeStampGPU = 0;
            uint64_t    endTimeStampGPU = 0;

            // Get the timestamp value of the last command in the batch
            cb->lastTS_->value(&startTimeStampGPU, &endTimeStampGPU);

            uint64_t    endTimeStampCPU = amd::Os::timeNanos();
            // Make sure the command batch has a valid GPU TS
            if (!GPU_RAW_TIMESTAMP) {
                // Adjust the base time by the execution time
                readjustTimeGPU_ = endTimeStampGPU - endTimeStampCPU;
            }
        }
    }
}

bool
VirtualGPU::allocConstantBuffers()
{
    // Allocate/reallocate constant buffers
    size_t minCbSize;
    // GCN doesn't really have a limit
    minCbSize = 256 * Ki;
    uint    i;

    // Create/reallocate constant buffer resources
    for (i = 0; i < MaxConstBuffersArguments; ++i) {
        ConstBuffer* constBuf = new ConstBuffer(*this, ((minCbSize +
            ConstBuffer::VectorSize - 1) / ConstBuffer::VectorSize));

        if ((constBuf != nullptr) && constBuf->create()) {
            addConstBuffer(constBuf);
        }
        else {
            // We failed to create a constant buffer
            delete constBuf;
            return false;
        }
    }

    return true;
}

void
VirtualGPU::profilingBegin(amd::Command& command, bool drmProfiling)
{
    // Is profiling enabled?
    if (command.profilingInfo().enabled_) {
        // Allocate a timestamp object from the cache
        TimeStamp* ts = tsCache_->allocTimeStamp();
        if (nullptr == ts) {
            return;
        }
        // Save the TimeStamp object in the current OCL event
        command.setData(ts);
        currTs_ = ts;
        state_.profileEnabled_ = true;
    }
}

void
VirtualGPU::profilingEnd(amd::Command& command)
{
    // Get the TimeStamp object associated witht the current command
    TimeStamp* ts = reinterpret_cast<TimeStamp*>(command.data());
    if (ts != nullptr) {
        // Check if the command actually did any GPU submission
        if (ts->isValid()) {
            cal_.lastTS_ = ts;
        }
        else {
            // Destroy the TimeStamp object
            tsCache_->freeTimeStamp(ts);
            command.setData(nullptr);
        }
    }
}

bool
VirtualGPU::profilingCollectResults(CommandBatch* cb, const amd::Event* waitingEvent)
{
    bool    found = false;
    amd::Command*   current;
    amd::Command*   first = cb->head_;

    // If the command list is, empty then exit
    if (nullptr == first) {
        return found;
    }

    // Wait for the last known GPU events on all engines
    waitEventLock(cb);

    // Find the CPU base time of the entire command batch execution
    uint64_t    endTimeStamp = amd::Os::timeNanos();
    uint64_t    startTimeStamp = endTimeStamp;

    // First step, walk the command list to find the first valid command
    //! \note The batch may have empty markers at the beginning.
    //! So the start/end of the empty commands is equal to
    //! the start of the first valid command in the batch.
    first = cb->head_;
    while (nullptr != first) {
        // Get the TimeStamp object associated witht the current command
        TimeStamp* ts = reinterpret_cast<TimeStamp*>(first->data());

        if (ts != nullptr) {
            ts->value(&startTimeStamp, &endTimeStamp);
            endTimeStamp -= readjustTimeGPU_;
            startTimeStamp -= readjustTimeGPU_;
            // Assign to endTimeStamp the start of the first valid command
            endTimeStamp = startTimeStamp;
            break;
        }
        first = first->getNext();
    }

    // Second step, walk the command list to construct the time line
    first = cb->head_;
    while (nullptr != first) {
        // Get the TimeStamp object associated witht the current command
        TimeStamp* ts = reinterpret_cast<TimeStamp*>(first->data());

        current = first->getNext();

        if (ts != nullptr) {
            ts->value(&startTimeStamp, &endTimeStamp);
            endTimeStamp -= readjustTimeGPU_;
            startTimeStamp -= readjustTimeGPU_;
            // Destroy the TimeStamp object
            tsCache_->freeTimeStamp(ts);
            first->setData(nullptr);
        }
        else {
            // For empty commands start/end is equal to
            // the end of the last valid command
            startTimeStamp = endTimeStamp;
        }

        // Update the command status with the proper timestamps
        if (first->status() == CL_SUBMITTED) {
            first->setStatus(CL_RUNNING, startTimeStamp);
            first->setStatus(CL_COMPLETE, endTimeStamp);
        }
        else if (first->status() == CL_RUNNING) {
            first->setStatus(CL_COMPLETE, endTimeStamp);
        }
        else if ((first->status() != CL_COMPLETE) && (current != nullptr)) {
            LogPrintfError("Unexpected command status - %d!", first->status());
        }

        // Do we wait this event?
        if (first == waitingEvent) {
            found = true;
        }

        first->release();
        first = current;
    }

    return found;
}

bool
VirtualGPU::addVmMemory(const Memory* memory)
{
    queues_[MainEngine]->addCmdMemRef(memory->iMem());
    return true;
}

void
VirtualGPU::profileEvent(EngineType engine, bool type) const
{
    if (nullptr == currTs_) {
        return;
    }
    if (type) {
        currTs_->begin((engine == SdmaEngine) ? true : false);
    }
    else {
        currTs_->end((engine == SdmaEngine) ? true : false);
    }
}

bool
VirtualGPU::processMemObjectsHSA(
    const amd::Kernel&  kernel,
    const_address       params,
    bool                nativeMem,
    std::vector<const Memory*>* memList)
{
    static const bool NoAlias = true;
    const HSAILKernel& hsaKernel = static_cast<const HSAILKernel&>
        (*(kernel.getDeviceKernel(dev(), NoAlias)));
    const amd::KernelSignature& signature = kernel.signature();
    const amd::KernelParameters& kernelParams = kernel.parameters();

    // Mark the tracker with a new kernel,
    // so we can avoid checks of the aliased objects
    memoryDependency().newKernel();

    bool deviceSupportFGS = 0 != (dev().info().svmCapabilities_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM);
    bool supportFineGrainedSystem = deviceSupportFGS;
    FGSStatus status = kernelParams.getSvmSystemPointersSupport();
    switch (status) {
        case FGS_YES:
            if (!deviceSupportFGS) {
                return false;
            }
            supportFineGrainedSystem = true;
            break;
        case FGS_NO:
            supportFineGrainedSystem = false;
            break;
        case FGS_DEFAULT:
        default:
            break;
    }

    size_t count = kernelParams.getNumberOfSvmPtr();
    size_t execInfoOffset = kernelParams.getExecInfoOffset();
    bool sync = true;

    amd::Memory* memory = nullptr;
    //get svm non arugment information
    void* const* svmPtrArray =
        reinterpret_cast<void* const*>(params + execInfoOffset);
    for (size_t i = 0; i < count; i++) {
        memory =  amd::SvmManager::FindSvmBuffer(svmPtrArray[i]);
        if (nullptr == memory) {
            if (!supportFineGrainedSystem) {
                return false;
            }
            else if (sync) {
                Unimplemented();
                //flushCUCaches();
                // Clear memory dependency state
                const static bool All = true;
                memoryDependency().clear(!All);
            }
        }
        else {
            Memory* gpuMemory = dev().getGpuMemory(memory);
            if (nullptr != gpuMemory) {
                // Synchronize data with other memory instances if necessary
                gpuMemory->syncCacheFromHost(*this);

                const static bool IsReadOnly = false;
                // Validate SVM passed in the non argument list
                memoryDependency().validate(*this, gpuMemory, IsReadOnly);

                memList->push_back(gpuMemory);
            }
            else {
                return false;
            }
        }
    }

    // Check all parameters for the current kernel
    for (size_t i = 0; i < signature.numParameters(); ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        const HSAILKernel::Argument*  arg = hsaKernel.argument(i);
        Memory* memory = nullptr;
        bool    readOnly = false;
        amd::Memory* svmMem = nullptr;

        // Find if current argument is a buffer
        if ((desc.type_ == T_POINTER) && (arg->addrQual_ != HSAIL_ADDRESS_LOCAL)) {
            if (kernelParams.boundToSvmPointer(dev(), params, i)) {
                svmMem = amd::SvmManager::FindSvmBuffer(
                    *reinterpret_cast<void* const*>(params + desc.offset_));
                if (!svmMem) {
                    Unimplemented();
                    //flushCUCaches();
                    // Clear memory dependency state
                    const static bool All = true;
                    memoryDependency().clear(!All);
                }
            }

            if (nativeMem) {
                memory = *reinterpret_cast<Memory* const*>(params + desc.offset_);
            }
            else if (*reinterpret_cast<amd::Memory* const*>
                    (params + desc.offset_) != nullptr) {
                if (nullptr == svmMem) {
                    memory = dev().getGpuMemory(*reinterpret_cast<amd::Memory* const*>
                            (params + desc.offset_));
                }
                else {
                    memory = dev().getGpuMemory(svmMem);
                }
                // Synchronize data with other memory instances if necessary
                memory->syncCacheFromHost(*this);
            }

            if (memory != nullptr) {
                // Check image
                readOnly = (desc.accessQualifier_ ==
                    CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
                // Check buffer
                readOnly |= (arg->access_ == HSAIL_ACCESS_TYPE_RO) ? true : false;
                // Validate memory for a dependency in the queue
                memoryDependency().validate(*this, memory, readOnly);
            }
        }
    }

    for (pal::Memory* mem : hsaKernel.prog().globalStores()) {
        const static bool IsReadOnly = false;
        // Validate global store for a dependency in the queue
        memoryDependency().validate(*this, mem, IsReadOnly);
    }

    return true;
}

amd::Memory*
VirtualGPU::createBufferFromImage(amd::Memory& amdImage) const
{
    amd::Memory* mem = new(amdImage.getContext())
        amd::Buffer(amdImage, 0, 0, amdImage.getSize());

    if ((mem != nullptr) && !mem->create()) {
        mem->release();
    }

    return mem;
}

void
VirtualGPU::writeVQueueHeader(VirtualGPU& hostQ, uint64_t kernelTable)
{
    const static bool Wait = true;
    vqHeader_->kernel_table = kernelTable;
    virtualQueue_->writeRawData(hostQ, 0, sizeof(AmdVQueueHeader), vqHeader_, Wait);
}

void
VirtualGPU::flushCuCaches(HwDbgGpuCacheMask cache_mask)
{
    Unimplemented();
/*
    //! @todo:  fix issue of no event available for the flush/invalidate cache command
    InvalidateSqCaches(cache_mask.sqICache_,
                       cache_mask.sqKCache_,
                       cache_mask.tcL1_,
                       cache_mask.tcL2_);
*/
    flushDMA(engineID_);

    return;
}

void
VirtualGPU::buildKernelInfo(const HSAILKernel& hsaKernel,
                            hsa_kernel_dispatch_packet_t* aqlPkt,
                            HwDbgKernelInfo& kernelInfo,
                            amd::Event* enqueueEvent)
{
    amd::HwDebugManager * dbgManager = dev().hwDebugMgr();
    assert (dbgManager && "No HW Debug Manager!");

    // Initialize structure with default values

    if (hsaKernel.prog().maxScratchRegs() > 0) {
        pal::Memory* scratchBuf = dev().scratch(hwRing())->memObj_;
        kernelInfo.scratchBufAddr = scratchBuf->vmAddress();
        kernelInfo.scratchBufferSizeInBytes = scratchBuf->size();

        // Get the address of the scratch buffer and its size for CPU access
        address scratchRingAddr = nullptr;
        scratchRingAddr = static_cast<address>(scratchBuf->map(nullptr, 0));
        dbgManager->setScratchRing(scratchRingAddr,scratchBuf->size());
        scratchBuf->unmap(nullptr);
    }
    else {
        kernelInfo.scratchBufAddr = 0;
        kernelInfo.scratchBufferSizeInBytes = 0;
        dbgManager->setScratchRing(nullptr, 0);
    }

    //! @todo:  need to verify what is wanted for the global memory
    Unimplemented();
    kernelInfo.heapBufAddr = 0;

    kernelInfo.pAqlDispatchPacket = aqlPkt;
    kernelInfo.pAqlQueuePtr = reinterpret_cast<void*>(hsaQueueMem_->vmAddress());

    // Get the address of the kernel code and its size for CPU access
    pal::Memory* aqlCode = hsaKernel.gpuAqlCode();
    if (nullptr != aqlCode) {
        address aqlCodeAddr = static_cast<address>(aqlCode->map(nullptr, 0));
        dbgManager->setKernelCodeInfo(aqlCodeAddr, hsaKernel.aqlCodeSize());
        aqlCode->unmap(nullptr);
    }
    else {
        dbgManager->setKernelCodeInfo(nullptr, 0);
    }

    kernelInfo.trapPresent = false;
    kernelInfo.trapHandler = nullptr;
    kernelInfo.trapHandlerBuffer = nullptr;

    kernelInfo.excpEn = 0;
    kernelInfo.cacheDisableMask = 0;
    kernelInfo.sqDebugMode = 0;

    kernelInfo.mgmtSe0Mask = 0xFFFFFFFF;
    kernelInfo.mgmtSe1Mask = 0xFFFFFFFF;

    // set kernel info for HW debug and call the callback function
    if (nullptr != dbgManager->preDispatchCallBackFunc()) {
        DebugToolInfo dbgSetting = {0};
        dbgSetting.scratchAddress_ = kernelInfo.scratchBufAddr;
        dbgSetting.scratchSize_ = kernelInfo.scratchBufferSizeInBytes;
        dbgSetting.globalAddress_ = kernelInfo.heapBufAddr;
        dbgSetting.aclBinary_ = hsaKernel.prog().binaryElf();
        dbgSetting.event_ = enqueueEvent;

        // Call the predispatch callback function & set the trap info
        AqlCodeInfo  aqlCodeInfo;
        aqlCodeInfo.aqlCode_ = (amd_kernel_code_t *) hsaKernel.cpuAqlCode();
        aqlCodeInfo.aqlCodeSize_ = hsaKernel.aqlCodeSize();

        // Execute the pre-dispatch call back function
        dbgManager->executePreDispatchCallBack(reinterpret_cast<void*>(aqlPkt), &dbgSetting);

        // assign the debug TMA and TBA for kernel dispatch
        if (nullptr != dbgSetting.trapHandler_ && nullptr != dbgSetting.trapBuffer_) {
            assignDebugTrapHandler(dbgSetting, kernelInfo);
        }

        kernelInfo.trapPresent = (kernelInfo.trapHandler) ? true : false;

        // Execption policy
        kernelInfo.excpEn = dbgSetting.exceptionMask_;
        kernelInfo.cacheDisableMask = dbgSetting.cacheDisableMask_;
        kernelInfo.sqDebugMode = dbgSetting.gpuSingleStepMode_;

        // Compute the mask for reserved CUs. These two dwords correspond to
        // two registers used for reserving CUs for display. In the current
        // implementation, the number of CUs reserved can be 0 to 7, and it
        // is set by debugger users.
        if (dbgSetting.monitorMode_) {
            uint32_t i = dbgSetting.reservedCuNum_ / 2;
            kernelInfo.mgmtSe0Mask <<= i;
            i = dbgSetting.reservedCuNum_ - i;
            kernelInfo.mgmtSe1Mask <<= i;
        }
        Unimplemented();
/*
        // flush/invalidate the instruction, data, L1 and L2 caches
        InvalidateSqCaches();
*/
    }
}

void
VirtualGPU::assignDebugTrapHandler(const DebugToolInfo& dbgSetting,
                                   HwDbgKernelInfo& kernelInfo)
{
    // setup the runtime trap handler code and trap buffer to be assigned before kernel dispatching
    //
    Memory* rtTrapHandlerMem = static_cast<Memory*>(dev().hwDebugMgr()->runtimeTBA());
    Memory* rtTrapBufferMem = static_cast<Memory*>(dev().hwDebugMgr()->runtimeTMA());

    kernelInfo.trapHandler = reinterpret_cast<void *>(rtTrapHandlerMem->vmAddress() + TbaStartOffset);
    // With the TMA corruption hw bug workaround, the trap handler buffer can be set to zero.
    // However, by setting the runtime trap buffer (TMA) correct, the runtime trap hander
    // without the workaround can still function correctly.
    kernelInfo.trapHandlerBuffer = reinterpret_cast<void *>(rtTrapBufferMem->vmAddress());

    address rtTrapBufferAddress = static_cast<address>(rtTrapBufferMem->map(this));

    Memory* trapHandlerMem = dev().getGpuMemory(dbgSetting.trapHandler_);
    Memory* trapBufferMem = dev().getGpuMemory(dbgSetting.trapBuffer_);

    // Address of the trap handler code/buffer should be 256-byte aligned
    uint64_t tbaAddress = trapHandlerMem->vmAddress();
    uint64_t tmaAddress = trapBufferMem->vmAddress();
    if ((tbaAddress & 0xFF) != 0 || (tmaAddress & 0xFF) != 0) {
        assert(false && "Trap handler/buffer is not 256-byte aligned");
    }

    // The addresses of the debug trap handler code (TBA) and buffer (TMA) are
    // stored in the runtime trap handler buffer with offset location of 0x18-19
    // and 0x20-21, respectively.
    uint64_t * rtTmaPtr = reinterpret_cast<uint64_t *>(rtTrapBufferAddress + 0x18);
    rtTmaPtr[0] = tbaAddress;
    rtTmaPtr[1] = tmaAddress;

    rtTrapBufferMem->unmap(nullptr);

    // Add GPU mem handles to the memory list for VidMM
    addVmMemory(trapHandlerMem);
    addVmMemory(trapBufferMem);
    addVmMemory(rtTrapHandlerMem);
    addVmMemory(rtTrapBufferMem);

}

bool
VirtualGPU::validateSdmaOverlap(const Resource& src, const Resource& dst)
{
    uint64_t    srcVmEnd = src.vmAddress() + src.vmSize();
    if (((src.vmAddress() >= sdmaRange_.start_) &&
        (src.vmAddress() <= sdmaRange_.end_)) ||
        ((srcVmEnd >= sdmaRange_.start_) &&
         (srcVmEnd <= sdmaRange_.end_)) ||
        ((src.vmAddress() <= sdmaRange_.start_) &&
         (srcVmEnd >= sdmaRange_.end_))) {
        sdmaRange_.start_ = dst.vmAddress();
        sdmaRange_.end_ = dst.vmAddress() + dst.vmSize();
        return true;
    }

    sdmaRange_.start_ = std::min(sdmaRange_.start_, dst.vmAddress());
    sdmaRange_.end_ = std::max(sdmaRange_.end_, dst.vmAddress() + dst.vmSize());
    return false;
}

void
VirtualGPU::submitTransferBufferFromFile(amd::TransferBufferFileCommand& cmd)
{
    size_t copySize = cmd.size()[0];
    size_t fileOffset = cmd.fileOffset();
    size_t srcDstOffset = cmd.origin()[0];
    Memory* mem = dev().getGpuMemory(&cmd.memory());
    uint    idx = 0;

    assert((cmd.type() == CL_COMMAND_WRITE_BUFFER_FROM_FILE_AMD) ||
        (cmd.type() == CL_COMMAND_READ_BUFFER_FROM_FILE_AMD));
    bool writeBuffer(cmd.type() == CL_COMMAND_WRITE_BUFFER_FROM_FILE_AMD);

    while (copySize > 0) {
        Memory* staging = dev().getGpuMemory(&cmd.staging(idx));
        size_t srcDstSize = amd::TransferBufferFileCommand::StagingBufferSize;
        srcDstSize = std::min(srcDstSize, copySize);
        void* srcDstBuffer = staging->cpuMap(*this);
        if (!cmd.file()->transferBlock(writeBuffer, srcDstBuffer, fileOffset, 0, srcDstSize)) {
            return;
        }
        staging->cpuUnmap(*this);

        bool result = blitMgr().copyBuffer(*staging, *mem,
            fileOffset, srcDstOffset, srcDstSize, false);
        flushDMA(getGpuEvent(staging->iMem())->engineId_);
        srcDstOffset += srcDstSize;
        copySize -= srcDstSize;
    }
}
} // namespace pal
