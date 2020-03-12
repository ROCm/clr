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

#include "platform/perfctr.hpp"
#include "platform/threadtrace.hpp"
#include "platform/kernel.hpp"
#include "platform/commandqueue.hpp"
#include "device/gpu/gpuconstbuf.hpp"
#include "device/gpu/gpuvirtual.hpp"
#include "device/gpu/gpukernel.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpucounters.hpp"
#include "device/gpu/gputhreadtrace.hpp"
#include "device/gpu/gputimestamp.hpp"
#include "device/gpu/gpublit.hpp"
#include "device/gpu/gpudebugger.hpp"
#include "shader/ComputeProgramObject.h"
#include "hsa.h"
#include "amd_hsa_kernel_code.h"
#include "amd_hsa_queue.h"
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#include <d3d10_1.h>
#include "amdocl/cl_d3d9_amd.hpp"
#include "amdocl/cl_d3d10_amd.hpp"
#include "amdocl/cl_d3d11_amd.hpp"
#endif  // _WIN32

namespace gpu {

bool VirtualGPU::MemoryDependency::create(size_t numMemObj) {
  if (numMemObj > 0) {
    // Allocate the array of memory objects for dependency tracking
    memObjectsInQueue_ = new MemoryState[numMemObj];
    if (NULL == memObjectsInQueue_) {
      return false;
    }
    memset(memObjectsInQueue_, 0, sizeof(MemoryState) * numMemObj);
    maxMemObjectsInQueue_ = numMemObj;
  }

  return true;
}

void VirtualGPU::MemoryDependency::validate(VirtualGPU& gpu, const Memory* memory, bool readOnly) {
  bool flushL1Cache = false;

  if (maxMemObjectsInQueue_ == 0) {
    // Flush cache
    gpu.flushCUCaches();
    return;
  }

  uint64_t curStart = memory->hbOffset();
  uint64_t curEnd = curStart + memory->hbSize();

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
  if (maxMemObjectsInQueue_ <= numMemObjectsInQueue_) {
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
  memObjectsInQueue_[numMemObjectsInQueue_].start_ = curStart;
  memObjectsInQueue_[numMemObjectsInQueue_].end_ = curEnd;
  memObjectsInQueue_[numMemObjectsInQueue_].readOnly_ = readOnly;
  numMemObjectsInQueue_++;
}

void VirtualGPU::MemoryDependency::clear(bool all) {
  if (numMemObjectsInQueue_ > 0) {
    size_t i, j;
    if (all) {
      endMemObjectsInQueue_ = numMemObjectsInQueue_;
    }

    // If the current launch didn't start from the beginning, then move the data
    if (0 != endMemObjectsInQueue_) {
      // Preserve all objects from the current kernel
      for (i = 0, j = endMemObjectsInQueue_; j < numMemObjectsInQueue_; i++, j++) {
        memObjectsInQueue_[i].start_ = memObjectsInQueue_[j].start_;
        memObjectsInQueue_[i].end_ = memObjectsInQueue_[j].end_;
        memObjectsInQueue_[i].readOnly_ = memObjectsInQueue_[j].readOnly_;
      }
    } else if (numMemObjectsInQueue_ >= maxMemObjectsInQueue_) {
      // note: The array growth shouldn't occur under the normal conditions,
      // but in a case when SVM path sends the amount of SVM ptrs over
      // the max size of kernel arguments
      MemoryState* ptr  = new MemoryState[maxMemObjectsInQueue_ << 1];
      if (nullptr == ptr) {
        numMemObjectsInQueue_ = 0;
        return;
      }
      maxMemObjectsInQueue_ <<= 1;
      memcpy(ptr, memObjectsInQueue_, sizeof(MemoryState) * numMemObjectsInQueue_);
      delete[] memObjectsInQueue_;
      memObjectsInQueue_= ptr;
    }
    numMemObjectsInQueue_ -= endMemObjectsInQueue_;
    endMemObjectsInQueue_ = 0;
  }
}

VirtualGPU::DmaFlushMgmt::DmaFlushMgmt(const Device& dev) : cbWorkload_(0), dispatchSplitSize_(0) {
  aluCnt_ = dev.info().simdPerCU_ * dev.info().simdWidth_ * dev.info().maxComputeUnits_;
  maxDispatchWorkload_ = static_cast<uint64_t>(dev.info().maxEngineClockFrequency_) *
      // find time in us
      dev.settings().maxWorkloadTime_ * aluCnt_;
  resetCbWorkload(dev);
}

void VirtualGPU::DmaFlushMgmt::resetCbWorkload(const Device& dev) {
  cbWorkload_ = 0;
  maxCbWorkload_ = static_cast<uint64_t>(dev.info().maxEngineClockFrequency_) *
      // find time in us
      dev.settings().minWorkloadTime_ * aluCnt_;
}

void VirtualGPU::DmaFlushMgmt::findSplitSize(const Device& dev, uint64_t threads,
                                             uint instructions) {
  uint64_t workload = threads * instructions;
  if (maxDispatchWorkload_ < workload) {
    dispatchSplitSize_ = static_cast<uint>(maxDispatchWorkload_ / instructions);
    uint fullLoad = dev.info().maxComputeUnits_ * dev.info().preferredWorkGroupSize_;
    if ((dispatchSplitSize_ % fullLoad) != 0) {
      dispatchSplitSize_ = (dispatchSplitSize_ / fullLoad + 1) * fullLoad;
    }
  } else {
    dispatchSplitSize_ =
        (threads > dev.settings().workloadSplitSize_) ? dev.settings().workloadSplitSize_ : 0;
  }
}

bool VirtualGPU::DmaFlushMgmt::isCbReady(VirtualGPU& gpu, uint64_t threads, uint instructions) {
  bool cbReady = false;
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

bool VirtualGPU::gslOpen(uint nEngines, gslEngineDescriptor* engines, uint32_t rtCUs) {
  // GSL device initialization
  dev().PerformFullInitialization();

  // Wait the event
  m_waitType = dev().settings().syncObject_ ? CAL_WAIT_LOW_CPU_UTILIZATION : CAL_WAIT_POLLING;

  if (!open(&dev(), nEngines, engines, rtCUs)) {
    return false;
  }

  return true;
}

void VirtualGPU::gslDestroy() { close(dev().getNative()); }

void VirtualGPU::addXferWrite(Memory& memory) {
  if (xferWriteBuffers_.size() > 7) {
    dev().xferWrite().release(*this, *xferWriteBuffers_.front());
    xferWriteBuffers_.pop_front();
  }

  // Delay destruction
  xferWriteBuffers_.push_back(&memory);
}

void VirtualGPU::releaseXferWrite() {
  for (auto& memory : xferWriteBuffers_) {
    dev().xferWrite().release(*this, *memory);
  }
  xferWriteBuffers_.clear();
}

void VirtualGPU::addPinnedMem(amd::Memory* mem) {
  if (NULL == findPinnedMem(mem->getHostMem(), mem->getSize())) {
    if (pinnedMems_.size() > 7) {
      pinnedMems_.front()->release();
      pinnedMems_.pop_front();
    }

    // Start operation, since we should release mem object
    flushDMA(getGpuEvent(dev().getGpuMemory(mem)->gslResource())->engineId_);

    // Delay destruction
    pinnedMems_.push_back(mem);
  }
}

void VirtualGPU::releasePinnedMem() {
  for (auto& amdMemory : pinnedMems_) {
    amdMemory->release();
  }
  pinnedMems_.clear();
}

amd::Memory* VirtualGPU::findPinnedMem(void* addr, size_t size) {
  for (auto& amdMemory : pinnedMems_) {
    if ((amdMemory->getHostMem() == addr) && (size <= amdMemory->getSize())) {
      return amdMemory;
    }
  }
  return NULL;
}

bool VirtualGPU::createVirtualQueue(uint deviceQueueSize) {
  uint MinDeviceQueueSize = 16 * 1024;
  deviceQueueSize = std::max(deviceQueueSize, MinDeviceQueueSize);

  maskGroups_ = deviceQueueSize / (512 * Ki);
  maskGroups_ = (maskGroups_ == 0) ? 1 : maskGroups_;

  // Align the queue size for the multiple dispatch scheduler.
  // Each thread works with 32 entries * maskGroups
  uint extra = deviceQueueSize % (sizeof(AmdAqlWrap) * DeviceQueueMaskSize * maskGroups_);
  if (extra != 0) {
    deviceQueueSize += (sizeof(AmdAqlWrap) * DeviceQueueMaskSize * maskGroups_) - extra;
  }

  if (deviceQueueSize_ == deviceQueueSize) {
    return true;
  } else {
    //! @todo Temporarily keep the buffer mapped for debug purpose
    if (NULL != schedParams_) {
      schedParams_->unmap(this);
    }
    delete vqHeader_;
    delete virtualQueue_;
    delete schedParams_;
    vqHeader_ = NULL;
    virtualQueue_ = NULL;
    schedParams_ = NULL;
    schedParamIdx_ = 0;
    deviceQueueSize_ = 0;
  }
  uint numSlots = deviceQueueSize / sizeof(AmdAqlWrap);
  uint allocSize = deviceQueueSize;

  // Add the virtual queue header
  allocSize += sizeof(AmdVQueueHeader);
  allocSize = amd::alignUp(allocSize, sizeof(AmdAqlWrap));

  uint argOffs = allocSize;

  // Add the kernel arguments and wait events
  uint singleArgSize = amd::alignUp(
      dev().info().maxParameterSize_ + 64 + dev().settings().numWaitEvents_ * sizeof(uint64_t),
      sizeof(AmdAqlWrap));
  allocSize += singleArgSize * numSlots;

  uint eventsOffs = allocSize;
  // Add the device events
  allocSize += dev().settings().numDeviceEvents_ * sizeof(AmdEvent);

  uint eventMaskOffs = allocSize;
  // Add mask array for events
  allocSize += amd::alignUp(dev().settings().numDeviceEvents_, DeviceQueueMaskSize) / 8;

  uint slotMaskOffs = allocSize;
  // Add mask array for AmdAqlWrap slots
  allocSize += amd::alignUp(numSlots, DeviceQueueMaskSize) / 8;

  virtualQueue_ = new Memory(dev(), allocSize);
  Resource::MemoryType type = (GPU_PRINT_CHILD_KERNEL == 0) ? Resource::Local : Resource::Remote;
  if ((virtualQueue_ == NULL) || !virtualQueue_->create(type)) {
    return false;
  }
  address ptr = reinterpret_cast<address>(virtualQueue_->map(this, Resource::WriteOnly));
  if (NULL == ptr) {
    return false;
  }
  // Clear memory
  memset(ptr, 0, allocSize);
  uint64_t vaBase = virtualQueue_->vmAddress();
  AmdVQueueHeader* header = reinterpret_cast<AmdVQueueHeader*>(ptr);

  // Initialize the virtual queue header
  header->aql_slot_num = numSlots;
  header->event_slot_num = dev().settings().numDeviceEvents_;
  header->event_slot_mask = vaBase + eventMaskOffs;
  header->event_slots = vaBase + eventsOffs;
  header->aql_slot_mask = vaBase + slotMaskOffs;
  header->wait_size = dev().settings().numWaitEvents_;
  header->arg_size = dev().info().maxParameterSize_ + 64;
  header->mask_groups = maskGroups_;
  vqHeader_ = new AmdVQueueHeader;
  if (NULL == vqHeader_) {
    return false;
  }
  *vqHeader_ = *header;

  // Go over all slots and perform initialization
  AmdAqlWrap* slots = reinterpret_cast<AmdAqlWrap*>(&header[1]);
  for (uint i = 0; i < numSlots; ++i) {
    uint64_t argStart = vaBase + argOffs + i * singleArgSize;
    slots[i].aql.kernarg_address = reinterpret_cast<void*>(argStart);
    slots[i].wait_list = argStart + dev().info().maxParameterSize_ + 64;
  }
  // Upload data back to local memory
  if (GPU_PRINT_CHILD_KERNEL == 0) {
    virtualQueue_->unmap(this);
  }

  schedParams_ = new Memory(dev(), 64 * Ki);
  if ((schedParams_ == NULL) || !schedParams_->create(Resource::RemoteUSWC)) {
    return false;
  }

  ptr = reinterpret_cast<address>(schedParams_->map(this));

  deviceQueueSize_ = deviceQueueSize;

  return true;
}

VirtualGPU::VirtualGPU(Device& device)
    : device::VirtualDevice(device),
      CALGSLContext(),
      engineID_(MainEngine),
      activeKernelDesc_(NULL),
      gpuDevice_(static_cast<Device&>(device)),
      printfDbg_(NULL),
      printfDbgHSA_(NULL),
      tsCache_(NULL),
      vmMems_(NULL),
      numVmMems_(0),
      dmaFlushMgmt_(device),
      hwRing_(0),
      readjustTimeGPU_(0),
      currTs_(NULL),
      vqHeader_(NULL),
      virtualQueue_(NULL),
      schedParams_(NULL),
      schedParamIdx_(0),
      deviceQueueSize_(0),
      maskGroups_(1),
      hsaQueueMem_(NULL),
      profileEnabled_(false) {
  memset(&cal_, 0, sizeof(CalVirtualDesc));
  for (uint i = 0; i < AllEngines; ++i) {
    cal_.events_[i].invalidate();
  }
  memset(&cal_.samplersState_, 0xff, sizeof(cal_.samplersState_));

  // Note: Virtual GPU device creation must be a thread safe operation
  index_ = gpuDevice_.numOfVgpus_++;
  gpuDevice_.vgpus_.resize(gpuDevice_.numOfVgpus());
  gpuDevice_.vgpus_[index()] = this;
}

bool VirtualGPU::create(bool profiling, uint rtCUs, uint deviceQueueSize,
                        amd::CommandQueue::Priority priority) {
  device::BlitManager::Setup blitSetup;
  gslEngineDescriptor engines[2];
  uint engineMask = 0;
  uint32_t num = 0;

  if (index() >= GPU_MAX_COMMAND_QUEUES) {
    // Cap the maximum number of concurrent Virtual GPUs.
    return false;
  }

  // Virtual GPU will have profiling enabled
  state_.profiling_ = profiling;

  {
    if (dev().engines().numComputeRings()) {
      uint idx;

      if ((amd::CommandQueue::RealTimeDisabled == rtCUs) &&
          (priority == amd::CommandQueue::Priority::Normal)) {
        idx = index() % dev().engines().numComputeRings();
        engineMask = dev().engines().getMask((gslEngineID)(
            dev().isComputeRingIDForced() ? dev().getforcedComputeEngineID()
                                          : (dev().getFirstAvailableComputeEngineID() + idx)));

      } else {
        if ((priority == amd::CommandQueue::Priority::Medium) &&
            (amd::CommandQueue::RealTimeDisabled == rtCUs)) {
          engineMask = dev().engines().getMask((gslEngineID)(GSL_ENGINEID_COMPUTE_MEDIUM_PRIORITY));
        } else {
          if (priority == amd::CommandQueue::Priority::Medium) {
            engineMask = dev().engines().getMask((gslEngineID)(GSL_ENGINEID_COMPUTE_RT1));
          } else {
            engineMask = dev().engines().getMask((gslEngineID)(GSL_ENGINEID_COMPUTE_RT));
          }
        }
        //!@todo This is not a generic solution and
        // may have issues with > 8 queues
        idx = index() % (dev().engines().numComputeRings() + dev().engines().numComputeRingsRT());
      }
      // hwRing_ should be set 0 if forced to have single scratch buffer
      hwRing_ = (dev().settings().useSingleScratch_) ? 0 : idx;

      if (dev().canDMA()) {
        // If only 1 DMA engine is available then use that one
        if (dev().engines().numDMAEngines() < 2) {
          engineMask |= dev().engines().getMask(GSL_ENGINEID_DRMDMA0);
        } else if (index() & 0x1) {
          engineMask |= dev().engines().getMask(GSL_ENGINEID_DRMDMA0);
        } else {
          engineMask |= dev().engines().getMask(GSL_ENGINEID_DRMDMA1);
        }
      }
    } else {
      engineMask = dev().engines().getMask(GSL_ENGINEID_3DCOMPUTE0);
      if (dev().canDMA()) {
        engineMask |= dev().engines().getMask(GSL_ENGINEID_DRMDMA0);
      }
    }
  }
  num = dev().engines().getRequested(engineMask, engines);

  // Open GSL context
  if ((num == 0) || !gslOpen(num, engines, rtCUs)) {
    return false;
  }

  // Diable double copy optimization,
  // since UAV read from nonlocal is fast enough
  blitSetup.disableCopyBufferToImageOpt_ = true;
  if (!allocConstantBuffers()) {
    return false;
  }

  // Create Printf class
  printfDbg_ = new PrintfDbg(gpuDevice_);
  if ((NULL == printfDbg_) || !printfDbg_->create()) {
    delete printfDbg_;
    LogError("Could not allocate debug buffer for printf()!");
    return false;
  }

  // Create HSAILPrintf class
  printfDbgHSA_ = new PrintfDbgHSA(gpuDevice_);
  if (NULL == printfDbgHSA_) {
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
        blitSetup.disableCopyImageToBuffer_ = true;
        blitSetup.disableCopyBufferToImage_ = true;
      }
      blitMgr_ = new KernelBlitManager(*this, blitSetup);
      break;
  }
  if ((NULL == blitMgr_) || !blitMgr_->create(gpuDevice_)) {
    LogError("Could not create BlitManager!");
    return false;
  }

  tsCache_ = new TimeStampCache(*this);
  if (NULL == tsCache_) {
    LogError("Could not create TimeStamp cache!");
    return false;
  }

  if (!memoryDependency().create(dev().settings().numMemDependencies_)) {
    LogError("Could not create the array of memory objects!");
    return false;
  }

  if (!allocHsaQueueMem()) {
    LogError("Could not create hsaQueueMem object!");
    return false;
  }

  // Check if the app requested a device queue creation
  if (dev().settings().useDeviceQueue_ && (0 != deviceQueueSize) &&
      !createVirtualQueue(deviceQueueSize)) {
    LogError("Could not create a virtual queue!");
    return false;
  }

  return true;
}

bool VirtualGPU::allocHsaQueueMem() {
  // Allocate a dummy HSA queue
  hsaQueueMem_ = new Memory(dev(), sizeof(amd_queue_t));
  if ((hsaQueueMem_ == NULL) || (!hsaQueueMem_->create(Resource::Local))) {
    delete hsaQueueMem_;
    return false;
  }
  amd_queue_t* queue = reinterpret_cast<amd_queue_t*>(hsaQueueMem_->map(NULL, Resource::WriteOnly));
  if (NULL == queue) {
    delete hsaQueueMem_;
    return false;
  }
  memset(queue, 0, sizeof(amd_queue_t));
  // Provide private and local heap addresses
  const static uint addressShift = LP64_SWITCH(0, 32);
  queue->private_segment_aperture_base_hi =
      static_cast<uint32>(dev().gslCtx()->getPrivateApertureBase() >> addressShift);
  queue->group_segment_aperture_base_hi =
      static_cast<uint32>(dev().gslCtx()->getSharedApertureBase() >> addressShift);
  hsaQueueMem_->unmap(NULL);
  return true;
}

VirtualGPU::~VirtualGPU() {
  // Not safe to remove a queue. So lock the device
  amd::ScopedLock k(dev().lockAsyncOps());
  amd::ScopedLock lock(dev().vgpusAccess());

  uint i;
  // Destroy all kernels
  for (const auto& it : gslKernels_) {
    if (it.first != 0) {
      freeKernelDesc(it.second);
    }
  }
  gslKernels_.clear();

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
  for (i = 0; i < constBufs_.size(); ++i) {
    delete constBufs_[i];
  }

  gslDestroy();

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

  delete[] vmMems_;
  //! @todo Temporarily keep the buffer mapped for debug purpose
  if (NULL != schedParams_) {
    schedParams_->unmap(this);
  }
  delete vqHeader_;
  delete virtualQueue_;
  delete schedParams_;
  delete hsaQueueMem_;
}

void VirtualGPU::submitReadMemory(amd::ReadMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  // Translate memory references and ensure cache up-to-date
  gpu::Memory* memory = dev().getGpuMemory(&vcmd.source());

  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(vcmd.destination(), &offset);

  profilingBegin(vcmd, true);

  memory->syncCacheFromHost(*this);
  cl_command_type type = vcmd.type();
  bool result = false;
  amd::Memory* bufferFromImage = NULL;

  // Force buffer read for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_READ_IMAGE) &&
      (vcmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImage = createBufferFromImage(vcmd.source());
    if (NULL == bufferFromImage) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_READ_BUFFER;
      memory = dev().getGpuMemory(bufferFromImage);
    }
  }

  // Process different write commands
  switch (type) {
    case CL_COMMAND_READ_BUFFER: {
      amd::Coord3D origin(vcmd.origin()[0]);
      amd::Coord3D size(vcmd.size()[0]);
      if (NULL != bufferFromImage) {
        size_t elemSize = vcmd.source().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if (hostMemory != NULL) {
        // Accelerated transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result = blitMgr().copyBuffer(*memory, *hostMemory, origin, dstOrigin, size,
                                      vcmd.isEntireMemory());
      } else {
        // The logic below will perform 2 step copy to make sure memory pinning doesn't
        // occur on the first unaligned page, because in Windows memory manager can
        // have CPU access to the allocation header in another thread
        // and a race condition is possible.
        char* tmpHost =
            amd::alignUp(reinterpret_cast<char*>(vcmd.destination()), PinnedMemoryAlignment);

        // Find the partial size for unaligned copy
        size_t partial = tmpHost - reinterpret_cast<char*>(vcmd.destination());
        result = true;
        // Check if it's staging copy, then ignore unaligned address
        if (size[0] <= dev().settings().pinnedMinXferSize_) {
          partial = size[0];
        }
        // Make first step transfer
        if (partial > 0) {
          result = blitMgr().readBuffer(*memory, vcmd.destination(), origin, partial);
        }
        // Second step transfer if something left to copy
        if (partial < size[0]) {
          result &= blitMgr().readBuffer(*memory, tmpHost, origin[0] + partial, size[0] - partial);
        }
      }
      if (NULL != bufferFromImage) {
        bufferFromImage->release();
      }
    } break;
    case CL_COMMAND_READ_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(vcmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, vcmd.size().c, vcmd.hostRect().rowPitch_,
                            vcmd.hostRect().slicePitch_);
      if (hostMemory != NULL) {
        result = blitMgr().copyBufferRect(*memory, *hostMemory, vcmd.bufRect(), hostbufferRect,
                                          vcmd.size(), vcmd.isEntireMemory());
      } else {
        result = blitMgr().readBufferRect(*memory, vcmd.destination(), vcmd.bufRect(),
                                          vcmd.hostRect(), vcmd.size(), vcmd.isEntireMemory());
      }
    } break;
    case CL_COMMAND_READ_IMAGE:
      if (hostMemory != NULL) {
        // Accelerated image to buffer transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result =
            blitMgr().copyImageToBuffer(*memory, *hostMemory, vcmd.origin(), dstOrigin, vcmd.size(),
                                        vcmd.isEntireMemory(), vcmd.rowPitch(), vcmd.slicePitch());
      } else {
        result = blitMgr().readImage(*memory, vcmd.destination(), vcmd.origin(), vcmd.size(),
                                     vcmd.rowPitch(), vcmd.slicePitch(), vcmd.isEntireMemory());
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

void VirtualGPU::submitWriteMemory(amd::WriteMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  // Translate memory references and ensure cache up to date
  gpu::Memory* memory = dev().getGpuMemory(&vcmd.destination());
  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(vcmd.source(), &offset);

  profilingBegin(vcmd, true);

  bool entire = vcmd.isEntireMemory();

  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  memory->syncCacheFromHost(*this, syncFlags);

  cl_command_type type = vcmd.type();
  bool result = false;
  amd::Memory* bufferFromImage = NULL;

  // Force buffer write for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_WRITE_IMAGE) &&
      (vcmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImage = createBufferFromImage(vcmd.destination());
    if (NULL == bufferFromImage) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_WRITE_BUFFER;
      memory = dev().getGpuMemory(bufferFromImage);
    }
  }

  // Process different write commands
  switch (type) {
    case CL_COMMAND_WRITE_BUFFER: {
      amd::Coord3D origin(vcmd.origin()[0]);
      amd::Coord3D size(vcmd.size()[0]);
      if (NULL != bufferFromImage) {
        size_t elemSize = vcmd.destination().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if (hostMemory != NULL) {
        // Accelerated transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result = blitMgr().copyBuffer(*hostMemory, *memory, srcOrigin, origin, size,
                                      vcmd.isEntireMemory());
      } else {
        // The logic below will perform 2 step copy to make sure memory pinning doesn't
        // occur on the first unaligned page, because in Windows memory manager can
        // have CPU access to the allocation header in another thread
        // and a race condition is possible.
        const char* tmpHost =
            amd::alignUp(reinterpret_cast<const char*>(vcmd.source()), PinnedMemoryAlignment);

        // Find the partial size for unaligned copy
        size_t partial = tmpHost - reinterpret_cast<const char*>(vcmd.source());
        result = true;
        // Check if it's staging copy, then ignore unaligned address
        if (size[0] <= dev().settings().pinnedMinXferSize_) {
          partial = size[0];
        }
        // Make first step transfer
        if (partial > 0) {
          result = blitMgr().writeBuffer(vcmd.source(), *memory, origin, partial);
        }
        // Second step transfer if something left to copy
        if (partial < size[0]) {
          result &= blitMgr().writeBuffer(tmpHost, *memory, origin[0] + partial, size[0] - partial);
        }
      }
      if (NULL != bufferFromImage) {
        bufferFromImage->release();
      }
    } break;
    case CL_COMMAND_WRITE_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(vcmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, vcmd.size().c, vcmd.hostRect().rowPitch_,
                            vcmd.hostRect().slicePitch_);
      if (hostMemory != NULL) {
        result = blitMgr().copyBufferRect(*hostMemory, *memory, hostbufferRect, vcmd.bufRect(),
                                          vcmd.size(), vcmd.isEntireMemory());
      } else {
        result = blitMgr().writeBufferRect(vcmd.source(), *memory, vcmd.hostRect(), vcmd.bufRect(),
                                           vcmd.size(), vcmd.isEntireMemory());
      }
    } break;
    case CL_COMMAND_WRITE_IMAGE:
      if (hostMemory != NULL) {
        // Accelerated buffer to image transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result =
            blitMgr().copyBufferToImage(*hostMemory, *memory, srcOrigin, vcmd.origin(), vcmd.size(),
                                        vcmd.isEntireMemory(), vcmd.rowPitch(), vcmd.slicePitch());
      } else {
        result = blitMgr().writeImage(vcmd.source(), *memory, vcmd.origin(), vcmd.size(),
                                      vcmd.rowPitch(), vcmd.slicePitch(), vcmd.isEntireMemory());
      }
      break;
    default:
      LogError("Unsupported type for the write command");
      break;
  }

  if (!result) {
    LogError("submitWriteMemory failed!");
    vcmd.setStatus(CL_INVALID_OPERATION);
  } else {
    // Mark this as the most-recently written cache of the destination
    vcmd.destination().signalWrite(&gpuDevice_);
  }
  profilingEnd(vcmd);
}

bool VirtualGPU::copyMemory(cl_command_type type, amd::Memory& srcMem, amd::Memory& dstMem,
                            bool entire, const amd::Coord3D& srcOrigin,
                            const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                            const amd::BufferRect& srcRect, const amd::BufferRect& dstRect) {
  // Translate memory references and ensure cache up-to-date
  gpu::Memory* dstMemory = dev().getGpuMemory(&dstMem);
  gpu::Memory* srcMemory = dev().getGpuMemory(&srcMem);

  // Synchronize source and destination memory
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  dstMemory->syncCacheFromHost(*this, syncFlags);
  srcMemory->syncCacheFromHost(*this);

  amd::Memory* bufferFromImageSrc = NULL;
  amd::Memory* bufferFromImageDst = NULL;

  // Force buffer read for IMAGE1D_BUFFER
  if ((srcMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImageSrc = createBufferFromImage(srcMem);
    if (NULL == bufferFromImageSrc) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_COPY_BUFFER;
      srcMemory = dev().getGpuMemory(bufferFromImageSrc);
    }
  }
  // Force buffer write for IMAGE1D_BUFFER
  if ((dstMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImageDst = createBufferFromImage(dstMem);
    if (NULL == bufferFromImageDst) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_COPY_BUFFER;
      dstMemory = dev().getGpuMemory(bufferFromImageDst);
    }
  }

  bool result = false;

  // Check if HW can be used for memory copy
  switch (type) {
    case CL_COMMAND_SVM_MEMCPY:
    case CL_COMMAND_COPY_BUFFER: {
      amd::Coord3D realSrcOrigin(srcOrigin[0]);
      amd::Coord3D realDstOrigin(dstOrigin[0]);
      amd::Coord3D realSize(size.c[0], size.c[1], size.c[2]);

      if (NULL != bufferFromImageSrc) {
        size_t elemSize = srcMem.asImage()->getImageFormat().getElementSize();
        realSrcOrigin.c[0] *= elemSize;
        if (NULL != bufferFromImageDst) {
          realDstOrigin.c[0] *= elemSize;
        }
        realSize.c[0] *= elemSize;
      } else if (NULL != bufferFromImageDst) {
        size_t elemSize = dstMem.asImage()->getImageFormat().getElementSize();
        realDstOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
      }

      result = blitMgr().copyBuffer(*srcMemory, *dstMemory, realSrcOrigin, realDstOrigin, realSize,
                                    entire);

      if (NULL != bufferFromImageSrc) {
        bufferFromImageSrc->release();
      }
      if (NULL != bufferFromImageDst) {
        bufferFromImageDst->release();
      }
    } break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      result = blitMgr().copyBufferRect(*srcMemory, *dstMemory, srcRect, dstRect, size, entire);
      break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      result =
          blitMgr().copyImageToBuffer(*srcMemory, *dstMemory, srcOrigin, dstOrigin, size, entire);
      break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      result =
          blitMgr().copyBufferToImage(*srcMemory, *dstMemory, srcOrigin, dstOrigin, size, entire);
      break;
    case CL_COMMAND_COPY_IMAGE:
      result = blitMgr().copyImage(*srcMemory, *dstMemory, srcOrigin, dstOrigin, size, entire);
      break;
    default:
      LogError("Unsupported command type for memory copy!");
      break;
  }

  if (!result) {
    LogError("submitCopyMemory failed!");
    return false;
  } else {
    // Mark this as the most-recently written cache of the destination
    dstMem.signalWrite(&gpuDevice_);
  }
  return true;
}

void VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  cl_command_type type = vcmd.type();
  bool entire = vcmd.isEntireMemory();

  if (!copyMemory(type, vcmd.source(), vcmd.destination(), entire, vcmd.srcOrigin(),
                  vcmd.dstOrigin(), vcmd.size(), vcmd.srcRect(), vcmd.dstRect())) {
    vcmd.setStatus(CL_INVALID_OPERATION);
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);

  cl_command_type type = vcmd.type();
  // no op for FGS supported device
  if (!dev().isFineGrainedSystem()) {
    amd::Coord3D srcOrigin(0, 0, 0);
    amd::Coord3D dstOrigin(0, 0, 0);
    amd::Coord3D size(vcmd.srcSize(), 1, 1);
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    bool result = false;
    amd::Memory* srcMem = amd::MemObjMap::FindMemObj(vcmd.src());
    amd::Memory* dstMem = amd::MemObjMap::FindMemObj(vcmd.dst());

    device::Memory::SyncFlags syncFlags;
    if (nullptr != srcMem) {
      srcMem->commitSvmMemory();
      srcOrigin.c[0] =
          static_cast<const_address>(vcmd.src()) - static_cast<address>(srcMem->getSvmPtr());
      if (!(srcMem->validateRegion(srcOrigin, size))) {
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }
    if (nullptr != dstMem) {
      dstMem->commitSvmMemory();
      dstOrigin.c[0] =
          static_cast<const_address>(vcmd.dst()) - static_cast<address>(dstMem->getSvmPtr());
      if (!(dstMem->validateRegion(dstOrigin, size))) {
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }

    if (nullptr == srcMem && nullptr == dstMem) { // both not in svm space
      amd::Os::fastMemcpy(vcmd.dst(), vcmd.src(), vcmd.srcSize());
      result = true;
    } else if (nullptr == srcMem && nullptr != dstMem) {  // src not in svm space
      Memory* memory = dev().getGpuMemory(dstMem);
      // Synchronize source and destination memory
      syncFlags.skipEntire_ = dstMem->isEntirelyCovered(dstOrigin, size);
      memory->syncCacheFromHost(*this, syncFlags);

      result = blitMgr().writeBuffer(vcmd.src(), *memory, dstOrigin, size,
                                     dstMem->isEntirelyCovered(dstOrigin, size));
      // Mark this as the most-recently written cache of the destination
      dstMem->signalWrite(&gpuDevice_);
    } else if (nullptr != srcMem && nullptr == dstMem) {  // dst not in svm space
      Memory* memory = dev().getGpuMemory(srcMem);
      // Synchronize source and destination memory
      memory->syncCacheFromHost(*this);

      result = blitMgr().readBuffer(*memory, vcmd.dst(), srcOrigin, size,
                                    srcMem->isEntirelyCovered(srcOrigin, size));
    } else if (nullptr != srcMem && nullptr != dstMem) {  // both in svm space
      bool entire =
          srcMem->isEntirelyCovered(srcOrigin, size) && dstMem->isEntirelyCovered(dstOrigin, size);
      result =
          copyMemory(type, *srcMem, *dstMem, entire, srcOrigin, dstOrigin, size, srcRect, dstRect);
    }

    if (!result) {
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    // direct memcpy for FGS enabled system
    amd::SvmBuffer::memFill(vcmd.dst(), vcmd.src(), vcmd.srcSize(), 1);
  }
  profilingEnd(vcmd);
}

void VirtualGPU::submitMapMemory(amd::MapMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd, true);

  gpu::Memory* memory = dev().getGpuMemory(&vcmd.memory());

  // Save map info for unmap operation
  memory->saveMapInfo(vcmd.mapPtr(), vcmd.origin(), vcmd.size(), vcmd.mapFlags(),
                      vcmd.isEntireMemory());

  // If we have host memory, use it
  if ((memory->owner()->getHostMem() != NULL) && memory->isDirectMap()) {
    if (!memory->isHostMemDirectAccess()) {
      // Make sure GPU finished operation before
      // synchronization with the backing store
      memory->wait(*this);
    }

    // Target is the backing store, so just ensure that owner is up-to-date
    memory->owner()->cacheWriteBack();

    // Add memory to VA cache, so rutnime can detect direct access to VA
    dev().addVACache(memory);
  } else if (memory->isPersistentDirectMap()) {
    // Nothing to do here
  } else if (memory->mapMemory() != NULL) {
    // Target is a remote resource, so copy
    assert(memory->mapMemory() != NULL);
    if (vcmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
      amd::Coord3D dstOrigin(0, 0, 0);
      if (memory->cal()->buffer_) {
        if (!blitMgr().copyBuffer(*memory, *memory->mapMemory(), vcmd.origin(), vcmd.origin(),
                                  vcmd.size(), vcmd.isEntireMemory())) {
          LogError("submitMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
      } else if ((vcmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        amd::Memory* bufferFromImage = NULL;
        Memory* memoryBuf = memory;
        amd::Coord3D origin(vcmd.origin()[0]);
        amd::Coord3D size(vcmd.size()[0]);
        size_t elemSize = vcmd.memory().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;

        bufferFromImage = createBufferFromImage(vcmd.memory());
        if (NULL == bufferFromImage) {
          LogError("We should not fail buffer creation from image_buffer!");
        } else {
          memoryBuf = dev().getGpuMemory(bufferFromImage);
        }
        if (!blitMgr().copyBuffer(*memoryBuf, *memory->mapMemory(), origin, dstOrigin, size,
                                  vcmd.isEntireMemory())) {
          LogError("submitMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
        if (NULL != bufferFromImage) {
          bufferFromImage->release();
        }
      } else {
        // Validate if it's a view for a map of mip level
        if (vcmd.memory().parent() != NULL) {
          amd::Image* amdImage = vcmd.memory().parent()->asImage();
          if ((amdImage != NULL) && (amdImage->getMipLevels() > 1)) {
            // Save map write info in the parent object
            dev().getGpuMemory(amdImage)->saveMapInfo(vcmd.mapPtr(), vcmd.origin(), vcmd.size(),
                                                      vcmd.mapFlags(), vcmd.isEntireMemory(),
                                                      vcmd.memory().asImage());
          }
        }
        if (!blitMgr().copyImageToBuffer(*memory, *memory->mapMemory(), vcmd.origin(), dstOrigin,
                                         vcmd.size(), vcmd.isEntireMemory())) {
          LogError("submitMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
      }
    }
  } else {
    LogError("Unhandled map!");
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitUnmapMemory(amd::UnmapMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  gpu::Memory* memory = dev().getGpuMemory(&vcmd.memory());
  amd::Memory* owner = memory->owner();
  bool unmapMip = false;
  const device::Memory::WriteMapInfo* writeMapInfo = memory->writeMapInfo(vcmd.mapPtr());
  if (nullptr == writeMapInfo) {
    LogError("Unmap without map call");
    return;
  }
  profilingBegin(vcmd, true);

  // Check if image is a mipmap and assign a saved view
  amd::Image* amdImage = owner->asImage();
  if ((amdImage != NULL) && (amdImage->getMipLevels() > 1) && (writeMapInfo->baseMip_ != NULL)) {
    // Assign mip level view
    amdImage = writeMapInfo->baseMip_;
    // Clear unmap flags from the parent image
    memory->clearUnmapInfo(vcmd.mapPtr());
    memory = dev().getGpuMemory(amdImage);
    unmapMip = true;
    writeMapInfo = memory->writeMapInfo(vcmd.mapPtr());
  }

  // We used host memory
  if ((owner->getHostMem() != NULL) && memory->isDirectMap()) {
    if (writeMapInfo->isUnmapWrite()) {
      // Target is the backing store, so sync
      owner->signalWrite(NULL);
      memory->syncCacheFromHost(*this);
    }
    // Remove memory from VA cache
    dev().removeVACache(memory);
  }
  // data check was added for persistent memory that failed to get aperture
  // and therefore are treated like a remote resource
  else if (memory->isPersistentDirectMap() && (memory->data() != NULL)) {
    memory->unmap(this);
  } else if (memory->mapMemory() != NULL) {
    if (writeMapInfo->isUnmapWrite()) {
      amd::Coord3D srcOrigin(0, 0, 0);
      // Target is a remote resource, so copy
      assert(memory->mapMemory() != NULL);
      if (memory->cal()->buffer_) {
        if (!blitMgr().copyBuffer(*memory->mapMemory(), *memory, writeMapInfo->origin_,
                                  writeMapInfo->origin_, writeMapInfo->region_,
                                  writeMapInfo->isEntire())) {
          LogError("submitUnmapMemory() - copy failed");
          vcmd.setStatus(CL_OUT_OF_RESOURCES);
        }
      } else if ((vcmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        amd::Memory* bufferFromImage = NULL;
        Memory* memoryBuf = memory;
        amd::Coord3D origin(writeMapInfo->origin_[0]);
        amd::Coord3D size(writeMapInfo->region_[0]);
        size_t elemSize = vcmd.memory().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;

        bufferFromImage = createBufferFromImage(vcmd.memory());
        if (NULL == bufferFromImage) {
          LogError("We should not fail buffer creation from image_buffer!");
        } else {
          memoryBuf = dev().getGpuMemory(bufferFromImage);
        }
        if (!blitMgr().copyBuffer(*memory->mapMemory(), *memoryBuf, srcOrigin, origin, size,
                                  writeMapInfo->isEntire())) {
          LogError("submitUnmapMemory() - copy failed");
          vcmd.setStatus(CL_OUT_OF_RESOURCES);
        }
        if (NULL != bufferFromImage) {
          bufferFromImage->release();
        }
      } else {
        if (!blitMgr().copyBufferToImage(*memory->mapMemory(), *memory, srcOrigin,
                                         writeMapInfo->origin_, writeMapInfo->region_,
                                         writeMapInfo->isEntire())) {
          LogError("submitUnmapMemory() - copy failed");
          vcmd.setStatus(CL_OUT_OF_RESOURCES);
        }
      }
    }
  } else {
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

bool VirtualGPU::fillMemory(cl_command_type type, amd::Memory* amdMemory, const void* pattern,
                            size_t patternSize, const amd::Coord3D& origin,
                            const amd::Coord3D& size) {
  gpu::Memory* memory = dev().getGpuMemory(amdMemory);
  bool entire = amdMemory->isEntirelyCovered(origin, size);

  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  memory->syncCacheFromHost(*this, syncFlags);

  bool result = false;
  amd::Memory* bufferFromImage = NULL;
  float fillValue[4];

  // Force fill buffer for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_FILL_IMAGE) && (amdMemory->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    bufferFromImage = createBufferFromImage(*amdMemory);
    if (NULL == bufferFromImage) {
      LogError("We should not fail buffer creation from image_buffer!");
    } else {
      type = CL_COMMAND_FILL_BUFFER;
      memory = dev().getGpuMemory(bufferFromImage);
    }
  }

  // Find the the right fill operation
  switch (type) {
    case CL_COMMAND_FILL_BUFFER:
    case CL_COMMAND_SVM_MEMFILL: {
      amd::Coord3D realOrigin(origin[0]);
      amd::Coord3D realSize(size[0]);
      // Reprogram fill parameters if it's an IMAGE1D_BUFFER object
      if (NULL != bufferFromImage) {
        size_t elemSize = amdMemory->asImage()->getImageFormat().getElementSize();
        realOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
        memset(fillValue, 0, sizeof(fillValue));
        amdMemory->asImage()->getImageFormat().formatColor(pattern, fillValue);
        pattern = fillValue;
        patternSize = elemSize;
      }
      result = blitMgr().fillBuffer(*memory, pattern, patternSize, realOrigin, realSize,
                                    amdMemory->isEntirelyCovered(origin, size));
      if (NULL != bufferFromImage) {
        bufferFromImage->release();
      }
    } break;
    case CL_COMMAND_FILL_IMAGE:
      result = blitMgr().fillImage(*memory, pattern, origin, size,
                                   amdMemory->isEntirelyCovered(origin, size));
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

void VirtualGPU::submitFillMemory(amd::FillMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd, true);

  if (!fillMemory(vcmd.type(), &vcmd.memory(), vcmd.pattern(), vcmd.patternSize(), vcmd.origin(),
                  vcmd.size())) {
    vcmd.setStatus(CL_INVALID_OPERATION);
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd, true);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem()) {
    // Make sure we have memory for the command execution
    gpu::Memory* memory = dev().getGpuMemory(vcmd.getSvmMem());
    memory->saveMapInfo(vcmd.svmPtr(), vcmd.origin(), vcmd.size(), vcmd.mapFlags(),
                        vcmd.isEntireMemory());

    if (memory->mapMemory() != NULL) {
      if (vcmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
        assert(memory->cal()->buffer_ && "SVM memory can't be an image");
        if (!blitMgr().copyBuffer(*memory, *memory->mapMemory(), vcmd.origin(), vcmd.origin(),
                                  vcmd.size(), vcmd.isEntireMemory())) {
          LogError("submitSVMMapMemory() - copy failed");
          vcmd.setStatus(CL_MAP_FAILURE);
        }
      }
    } else if ((memory->owner()->getHostMem() != nullptr) && memory->isDirectMap()) {
      if (!memory->isHostMemDirectAccess()) {
        // Make sure GPU finished operation before
        // synchronization with the backing store
        memory->wait(*this);
      }

      // Target is the backing store, so just ensure that owner is up-to-date
      memory->owner()->cacheWriteBack();
    } else {
      LogError("Unhandled svm map!");
    }
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd, true);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem()) {
    gpu::Memory* memory = dev().getGpuMemory(vcmd.getSvmMem());
    const device::Memory::WriteMapInfo* writeMapInfo = memory->writeMapInfo(vcmd.svmPtr());

    if (memory->mapMemory() != NULL) {
      if (writeMapInfo->isUnmapWrite()) {
        // Target is a remote resource, so copy
        assert(memory->cal()->buffer_ && "SVM memory can't be an image");
        if (!blitMgr().copyBuffer(*memory->mapMemory(), *memory, writeMapInfo->origin_,
                                  writeMapInfo->origin_, writeMapInfo->region_,
                                  writeMapInfo->isEntire())) {
          LogError("submitSvmUnmapMemory() - copy failed");
          vcmd.setStatus(CL_OUT_OF_RESOURCES);
        }
      }
    } else if ((memory->owner()->getHostMem() != nullptr) && memory->isDirectMap()) {
      if (writeMapInfo->isUnmapWrite()) {
        // Target is the backing store, so sync
        memory->owner()->signalWrite(nullptr);
        memory->syncCacheFromHost(*this);
      }
    }
    memory->clearUnmapInfo(vcmd.svmPtr());
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd, true);

  if (!dev().isFineGrainedSystem()) {
    size_t patternSize = vcmd.patternSize();
    size_t fillSize = patternSize * vcmd.times();
    size_t offset = 0;
    amd::Memory* dstMemory = amd::MemObjMap::FindMemObj(vcmd.dst());
    assert(dstMemory && "No svm Buffer to fill with!");
    offset = reinterpret_cast<uintptr_t>(vcmd.dst()) -
        reinterpret_cast<uintptr_t>(dstMemory->getSvmPtr());
    assert((offset >= 0) && "wrong svm ptr to fill with!");

    gpu::Memory* memory = dev().getGpuMemory(dstMemory);

    amd::Coord3D origin(offset, 0, 0);
    amd::Coord3D size(fillSize, 1, 1);
    assert((dstMemory->validateRegion(origin, size)) && "The incorrect fill size!");

    if (!fillMemory(vcmd.type(), dstMemory, vcmd.pattern(), vcmd.patternSize(), origin, size)) {
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
    // Mark this as the most-recently written cache of the destination
    dstMemory->signalWrite(&gpuDevice_);
  } else {
    // for FGS capable device, fill CPU memory directly
    amd::SvmBuffer::memFill(vcmd.dst(), vcmd.pattern(), vcmd.patternSize(), vcmd.times());
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd, true);

  for (const auto& it : vcmd.memObjects()) {
    // Find device memory
    gpu::Memory* memory = dev().getGpuMemory(it);

    if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_HOST) {
      memory->mgpuCacheWriteBack();
    } else if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) {
      // Synchronize memory from host if necessary.
      // The sync function will perform memory migration from
      // another device if necessary
      device::Memory::SyncFlags syncFlags;
      memory->syncCacheFromHost(*this, syncFlags);
    } else {
      LogWarning("Unknown operation for memory migration!");
    }
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& vcmd) {
  // in-order semantics: previous commands need to be done before we start
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);
  std::vector<void*>& svmPointers = vcmd.svmPointers();
  if (vcmd.pfnFreeFunc() == NULL) {
    // pointers allocated using clSVMAlloc
    for (uint32_t i = 0; i < svmPointers.size(); ++i) {
      dev().svmFree(svmPointers[i]);
    }
  } else {
    vcmd.pfnFreeFunc()(as_cl(vcmd.queue()->asCommandQueue()), svmPointers.size(),
                       static_cast<void**>(&(svmPointers[0])), vcmd.userData());
  }
  profilingEnd(vcmd);
}

void VirtualGPU::findIterations(const amd::NDRangeContainer& sizes, const amd::NDRange& local,
                                amd::NDRange& groups, amd::NDRange& remainder, size_t& extra) {
  size_t dimensions = sizes.dimensions();

  if (cal()->iterations_ > 1) {
    size_t iterations = cal()->iterations_;
    cal_.iterations_ = 1;

    // Find the total amount of all groups
    groups = sizes.global() / local;
    for (uint j = 0; j < dimensions; ++j) {
      if ((sizes.global()[j] % local[j]) != 0) {
        groups[j]++;
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
      } else {
        iterations = ((iterations / groups[j]) + (((iterations % groups[j]) != 0) ? 1 : 0));
        cal_.iterations_ *= groups[j];
        groups[j] = 1;
      }
    }
  }
}

void VirtualGPU::setupIteration(uint iteration, const amd::NDRangeContainer& sizes,
                                Kernel& gpuKernel, amd::NDRange& global, amd::NDRange& offsets,
                                amd::NDRange& local, amd::NDRange& groups,
                                amd::NDRange& groupOffset, amd::NDRange& divider,
                                amd::NDRange& remainder, size_t extra) {
  size_t dimensions = sizes.dimensions();

  // Calculate the workload size for the remainder
  if ((extra != 0) && ((iteration % extra) == 0)) {
    groups = remainder;
  } else {
    groups = divider;
  }
  global = groups * local;

  for (uint j = 0; j < dimensions; ++j) {
    size_t offset = groupOffset[j] * local[j];
    if ((offset + global[j]) > sizes.global()[j]) {
      global[j] = sizes.global()[j] - offset;
    }
  }

  // Reprogram the kernel parameters for the GPU execution
  gpuKernel.setupProgramGrid(*this, dimensions, offsets, global, local, groupOffset, sizes.offset(),
                             sizes.global());

  // Update the constant buffers
  gpuKernel.bindConstantBuffers(*this);

  uint sub = 0;
  // Find the offsets for the next execution
  for (uint j = 0; j < dimensions; ++j) {
    groupOffset[j] += groups[j];
    // Make sure the offset doesn't go over the size limit
    if (sizes.global()[j] <= groupOffset[j] * local[j]) {
      // Check if we counted a group in one dimension already
      if (sub) {
        groupOffset[j] -= groups[j];
      } else {
        groupOffset[j] = 0;
      }
    } else {
      groupOffset[j] -= sub;
      // We already counted elements in one dimension
      sub = 1;
    }

    offsets[j] = groupOffset[j] * local[j] + sizes.offset()[j];
  }
}

void VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  // Submit kernel to HW
  if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(), false, &vcmd.event())) {
    vcmd.setStatus(CL_INVALID_OPERATION);
  }

  profilingEnd(vcmd);
}

bool VirtualGPU::submitKernelInternalHSA(const amd::NDRangeContainer& sizes,
                                         const amd::Kernel& kernel, const_address parameters,
                                         bool nativeMem, amd::Event* enqueueEvent) {
  uint64_t vmParentWrap = 0;
  uint64_t vmDefQueue = 0;
  amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(dev());
  VirtualGPU* gpuDefQueue = NULL;
  amd::HwDebugManager* dbgManager = dev().hwDebugMgr();

  // Get the HSA kernel object
  const HSAILKernel& hsaKernel = static_cast<const HSAILKernel&>(*(kernel.getDeviceKernel(dev())));
  std::vector<const Memory*> memList;

  bool printfEnabled = (hsaKernel.printfInfo().size() > 0) ? true : false;
  if (!printfDbgHSA().init(*this, printfEnabled)) {
    LogError("Printf debug buffer initialization failed!");
    return false;
  }

  // Check memory dependency and SVM objects
  if (!processMemObjectsHSA(kernel, parameters, nativeMem, &memList)) {
    LogError("Wrong memory objects!");
    return false;
  }

  cal_.memCount_ = 0;

  if (hsaKernel.dynamicParallelism()) {
    if (NULL == defQueue) {
      LogError("Default device queue wasn't allocated");
      return false;
    } else {
      if (dev().settings().useDeviceQueue_) {
        gpuDefQueue = static_cast<VirtualGPU*>(defQueue->vDev());
        if (gpuDefQueue->hwRing() == hwRing()) {
          LogError("Can't submit the child kernels to the same HW ring as the host queue!");
          return false;
        }
      } else {
        createVirtualQueue(defQueue->size());
        gpuDefQueue = this;
      }
    }
    vmDefQueue = gpuDefQueue->virtualQueue_->vmAddress();

    // Add memory handles before the actual dispatch
    memList.push_back(gpuDefQueue->virtualQueue_);
    memList.push_back(gpuDefQueue->schedParams_);
    memList.push_back(hsaKernel.prog().kernelTable());
    gpuDefQueue->writeVQueueHeader(*this, hsaKernel.prog().kernelTable()->vmAddress());
  }

  //  setup the storage for the memory pointers of the kernel parameters
  uint numParams = kernel.signature().numParameters();
  if (dbgManager) {
    dbgManager->allocParamMemList(numParams);
  }

  bool needFlush = false;
  dmaFlushMgmt_.findSplitSize(dev(), sizes.global().product(), hsaKernel.aqlCodeSize());
  if (dmaFlushMgmt().dispatchSplitSize() != 0) {
    needFlush = true;
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
        iteration = sizes.global()[i] / 0xC0000000 + ((sizes.global()[i] % 0xC0000000) ? 1 : 0);
        globalStep = (sizes.global()[i] / sizes.local()[i]) / iteration * sizes.local()[dim];
        break;
      }
    }
  }

  for (int j = 0; j < iteration; j++) {
    // Reset global size for dimension dim if split is needed
    if (dim != -1) {
      newOffset[dim] = sizes.offset()[dim] + globalStep * j;
      if (((newOffset[dim] + globalStep) < sizes.global()[dim]) && (j != (iteration - 1))) {
        newGlobalSize[dim] = globalStep;
      } else {
        newGlobalSize[dim] = sizes.global()[dim] - newOffset[dim];
      }
    }

    amd::NDRangeContainer tmpSizes(sizes.dimensions(), &newOffset[0], &newGlobalSize[0],
                                   &(const_cast<amd::NDRangeContainer&>(sizes).local()[0]));

    // Program the kernel arguments for the GPU execution
    hsa_kernel_dispatch_packet_t* aqlPkt = hsaKernel.loadArguments(
        *this, kernel, tmpSizes, parameters, nativeMem, vmDefQueue, &vmParentWrap, memList);
    if (NULL == aqlPkt) {
      LogError("Couldn't load kernel arguments");
      return false;
    }

    gslMemObject scratch = NULL;
    uint scratchOffset = 0;
    // Check if the device allocated more registers than the old setup
    if (hsaKernel.workGroupInfo()->scratchRegs_ > 0) {
      const Device::ScratchBuffer* scratchObj = dev().scratch(hwRing());
      scratch = scratchObj->memObj_->gslResource();
      memList.push_back(scratchObj->memObj_);
      scratchOffset = scratchObj->offset_;
    }

    // Add GSL handle to the memory list for VidMM
    for (uint i = 0; i < memList.size(); ++i) {
      addVmMemory(memList[i]);
    }

    // HW Debug for the kernel?
    HwDbgKernelInfo kernelInfo;
    HwDbgKernelInfo* pKernelInfo = NULL;

    if (dbgManager) {
      buildKernelInfo(hsaKernel, aqlPkt, kernelInfo, enqueueEvent);
      pKernelInfo = &kernelInfo;
    }

    // Set up the dispatch information
    KernelDispatchInfo dispatchInfo;
    dispatchInfo.aqlPacket = aqlPkt;
    dispatchInfo.mems = vmMems();
    dispatchInfo.numMems = cal_.memCount_;
    dispatchInfo.scratch = scratch;
    dispatchInfo.scratchOffset = scratchOffset;
    dispatchInfo.cpuAqlCode = hsaKernel.cpuAqlCode();
    dispatchInfo.hsaQueueVA = hsaQueueMem_->vmAddress();
    dispatchInfo.kernelInfo = pKernelInfo;
    dispatchInfo.wavesPerSH = hsaKernel.getWavesPerSH(this);
    dispatchInfo.lastDoppSubmission = kernel.parameters().getExecNewVcop();
    dispatchInfo.pfpaDoppSubmission = kernel.parameters().getExecPfpaVcop();

    GpuEvent gpuEvent;
    // Run AQL dispatch in HW
    eventBegin(MainEngine);
    cs()->AqlDispatch(&dispatchInfo);
    eventEnd(MainEngine, gpuEvent);

    if (dbgManager && (NULL != dbgManager->postDispatchCallBackFunc())) {
      dbgManager->executePostDispatchCallBack();
    }

    if (hsaKernel.dynamicParallelism()) {
      // Make sure exculsive access to the device queue
      amd::ScopedLock(defQueue->lock());

      if (GPU_PRINT_CHILD_KERNEL != 0) {
        waitForEvent(&gpuEvent);

        AmdAqlWrap* wraps =
            (AmdAqlWrap*)(&((AmdVQueueHeader*)gpuDefQueue->virtualQueue_->data())[1]);
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
            print << "Slot#: " << i << "\n";
            print << "\tenqueue_flags: " << wraps[i].enqueue_flags << "\n";
            print << "\tcommand_id: " << wraps[i].command_id << "\n";
            print << "\tchild_counter: " << wraps[i].child_counter << "\n";
            print << "\tcompletion: " << wraps[i].completion << "\n";
            print << "\tparent_wrap: " << wraps[i].parent_wrap << "\n";
            print << "\twait_list: " << wraps[i].wait_list << "\n";
            print << "\twait_num: " << wraps[i].wait_num << "\n";
            uint offsEvents = wraps[i].wait_list - gpuDefQueue->virtualQueue_->vmAddress();
            size_t* events =
                reinterpret_cast<size_t*>(gpuDefQueue->virtualQueue_->data() + offsEvents);
            for (j = 0; j < wraps[i].wait_num; ++j) {
              uint offs =
                  static_cast<uint64_t>(events[j]) - gpuDefQueue->virtualQueue_->vmAddress();
              AmdEvent* eventD = (AmdEvent*)(gpuDefQueue->virtualQueue_->data() + offs);
              print << "Wait Event#: " << j << "\n";
              print << "\tState: " << eventD->state << "; Counter: " << eventD->counter << "\n";
            }
            print << "WorkGroupSize[ " << wraps[i].aql.workgroup_size_x << ", ";
            print << wraps[i].aql.workgroup_size_y << ", ";
            print << wraps[i].aql.workgroup_size_z << "]\n";
            print << "GridSize[ " << wraps[i].aql.grid_size_x << ", ";
            print << wraps[i].aql.grid_size_y << ", ";
            print << wraps[i].aql.grid_size_z << "]\n";

            uint64_t* kernels =
                (uint64_t*)(const_cast<Memory*>(hsaKernel.prog().kernelTable())->map(this));
            for (j = 0; j < hsaKernel.prog().kernels().size(); ++j) {
              if (kernels[j] == wraps[i].aql.kernel_object) {
                break;
              }
            }
            const_cast<Memory*>(hsaKernel.prog().kernelTable())->unmap(this);
            HSAILKernel* child = NULL;
            for (auto it = hsaKernel.prog().kernels().begin();
                 it != hsaKernel.prog().kernels().end(); ++it) {
              if (j == static_cast<HSAILKernel*>(it->second)->index()) {
                child = static_cast<HSAILKernel*>(it->second);
              }
            }
            if (child == NULL) {
              printf("Error: couldn't find child kernel!\n");
              continue;
            }
            const uint64_t kernarg_address =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wraps[i].aql.kernarg_address));
            uint offsArg = kernarg_address - gpuDefQueue->virtualQueue_->vmAddress();
            address argum = gpuDefQueue->virtualQueue_->data() + offsArg;
            print << "Kernel: " << child->name() << "\n";
            static const char* Names[HSAILKernel::MaxExtraArgumentsNum] = {
                "Offset0: ", "Offset1: ", "Offset2: ", "PrintfBuf: ", "VqueuePtr: ", "AqlWrap: "};
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
        cs()->VirtualQueueHandshake(gpuDefQueue->schedParams_->gslResource(),
                                    vmParentWrap + offsetof(AmdAqlWrap, state), AQL_WRAP_DONE,
                                    vmParentWrap + offsetof(AmdAqlWrap, child_counter), 0,
                                    dev().settings().useDeviceQueue_);
        eventEnd(MainEngine, gpuEvent);
      }

      // Get the global loop start before the scheduler
      mcaddr loopStart = gpuDefQueue->cs()->VirtualQueueDispatcherStart();
      static_cast<KernelBlitManager&>(gpuDefQueue->blitMgr())
          .runScheduler(*gpuDefQueue->virtualQueue_, *gpuDefQueue->schedParams_,
                        gpuDefQueue->schedParamIdx_,
                        gpuDefQueue->vqHeader_->aql_slot_num / (DeviceQueueMaskSize * maskGroups_));
      const static bool FlushL2 = true;
      gpuDefQueue->flushCUCaches(FlushL2);

      // Get the address of PM4 template and add write it to params
      //! @note DMA flush must not occur between patch and the scheduler
      mcaddr patchStart = gpuDefQueue->cs()->VirtualQueueDispatcherStart();

      // Program parameters for the scheduler
      SchedulerParam* param = &reinterpret_cast<SchedulerParam*>(
          gpuDefQueue->schedParams_->data())[gpuDefQueue->schedParamIdx_];
      param->signal = 1;
      // Scale clock to 1024 to avoid 64 bit div in the scheduler
      param->eng_clk = (1000 * 1024) / dev().info().maxEngineClockFrequency_;
      param->hw_queue = patchStart + sizeof(uint32_t) /* Rewind packet*/;
      param->hsa_queue = gpuDefQueue->hsaQueueMem()->vmAddress();
      param->releaseHostCP = 0;
      param->parentAQL = vmParentWrap;
      param->dedicatedQueue = dev().settings().useDeviceQueue_;
      param->useATC = dev().settings().svmFineGrainSystem_;

      // Fill the scratch buffer information
      if (hsaKernel.prog().maxScratchRegs() > 0) {
        gpu::Memory* scratchBuf = dev().scratch(gpuDefQueue->hwRing())->memObj_;
        param->scratchSize = scratchBuf->size();
        param->scratch = scratchBuf->vmAddress();
        param->numMaxWaves = 32 * dev().info().maxComputeUnits_;
        param->scratchOffset = dev().scratch(gpuDefQueue->hwRing())->offset_;
        memList.push_back(scratchBuf);
      } else {
        param->numMaxWaves = 0;
        param->scratchSize = 0;
        param->scratch = 0;
        param->scratchOffset = 0;
      }

      // Add all kernels in the program to the mem list.
      //! \note Runtime doesn't know which one will be called
      hsaKernel.prog().fillResListWithKernels(memList);

      // Add GSL handle to the memory list for VidMM
      for (uint i = 0; i < memList.size(); ++i) {
        gpuDefQueue->addVmMemory(memList[i]);
      }

      mcaddr signalAddr = gpuDefQueue->schedParams_->vmAddress() +
          gpuDefQueue->schedParamIdx_ * sizeof(SchedulerParam);
      gpuDefQueue->eventBegin(MainEngine);
      gpuDefQueue->cs()->VirtualQueueDispatcherEnd(
          gpuDefQueue->vmMems(), gpuDefQueue->cal_.memCount_, signalAddr, loopStart,
          gpuDefQueue->vqHeader_->aql_slot_num / (DeviceQueueMaskSize * maskGroups_));
      gpuDefQueue->eventEnd(MainEngine, gpuEvent);

      // Set GPU event for the used resources
      for (uint i = 0; i < memList.size(); ++i) {
        memList[i]->setBusy(*gpuDefQueue, gpuEvent);
      }

      if (dev().settings().useDeviceQueue_) {
        // Add the termination handshake to the host queue
        eventBegin(MainEngine);
        cs()->VirtualQueueHandshake(gpuDefQueue->schedParams_->gslResource(),
                                    vmParentWrap + offsetof(AmdAqlWrap, state), AQL_WRAP_DONE,
                                    vmParentWrap + offsetof(AmdAqlWrap, child_counter), signalAddr,
                                    dev().settings().useDeviceQueue_);
        eventEnd(MainEngine, gpuEvent);
      }

      ++gpuDefQueue->schedParamIdx_ %= gpuDefQueue->schedParams_->size() / sizeof(SchedulerParam);
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
    setGpuEvent(gpuEvent, needFlush);

    if (!printfDbgHSA().output(*this, printfEnabled, hsaKernel.printfInfo())) {
      LogError("Couldn't read printf data from the buffer!\n");
      return false;
    }
  }

  // Runtime submitted a HSAIL kernel
  state_.hsailKernel_ = true;

  return true;
}

bool VirtualGPU::submitKernelInternal(const amd::NDRangeContainer& sizes, const amd::Kernel& kernel,
                                      const_address parameters, bool nativeMem,
                                      amd::Event* enqueueEvent) {
  bool result = true;
  uint i;
  size_t dimensions = sizes.dimensions();
  amd::NDRange local(sizes.local());
  amd::NDRange groupOffset(dimensions);
  GpuEvent gpuEvent;
  groupOffset = 0;

  // Get the GPU kernel object
  device::Kernel* devKernel = const_cast<device::Kernel*>(kernel.getDeviceKernel(dev()));
  Kernel& gpuKernelOpt = static_cast<gpu::Kernel&>(*devKernel);

  if (gpuKernelOpt.hsa()) {
    return submitKernelInternalHSA(sizes, kernel, parameters, nativeMem, enqueueEvent);
  } else if (state_.hsailKernel_) {
    // Reload GSL state to HW, so runtime could run AMDIL kernel
    flushDMA(MainEngine);
    // Reset HSAIL state
    state_.hsailKernel_ = false;
  }

  // Find if arguments contain memory aliases or a dependency in the queue
  gpuKernelOpt.processMemObjects(*this, kernel, parameters, nativeMem);

  Kernel& gpuKernel = static_cast<gpu::Kernel&>(*devKernel);
  bool printfEnabled = (gpuKernel.flags() & gpu::NullKernel::PrintfOutput) ? true : false;
  // Set current kernel CAL descriptor as active
  if (!setActiveKernelDesc(sizes, &gpuKernel) ||
      // Initialize printf support
      !printfDbg().init(*this, printfEnabled, sizes.global())) {
    LogPrintfError("We couldn't set \"%s\" kernel as active!", gpuKernel.name().data());
    return false;
  }

  // Find if we have to split workload
  dmaFlushMgmt_.findSplitSize(dev(), sizes.global().product(), gpuKernel.instructionCnt());

  // Program the kernel parameters for the GPU execution
  cal_.memCount_ = 0;
  gpuKernel.setupProgramGrid(*this, dimensions, sizes.offset(), sizes.global(), local, groupOffset,
                             sizes.offset(), sizes.global());

  // Load kernel arguments
  if (gpuKernel.loadParameters(*this, kernel, parameters, nativeMem)) {
    amd::NDRange global(sizes.global());
    amd::NDRange groups(dimensions);
    amd::NDRange offsets(sizes.offset());
    amd::NDRange divider(dimensions);
    amd::NDRange remainder(dimensions);
    size_t extra = 0;

    // Split the workload if necessary for local/private emulation or printf
    findIterations(sizes, local, groups, remainder, extra);

    divider = groups;
    i = 0;
    do {
      bool lastRun = (i == (cal()->iterations_ - 1)) ? true : false;
      // Reprogram the CAL grid and constant buffers if
      // the workload split is on
      if (cal()->iterations_ > 1) {
        // Initialize printf support
        if (!printfDbg().init(*this, printfEnabled, local)) {
          result = false;
          break;
        }

        // Reprogram the CAL grid and constant buffers
        setupIteration(i, sizes, gpuKernel, global, offsets, local, groups, groupOffset, divider,
                       remainder, extra);
      }

      // Execute the kernel
      if (gpuKernel.run(*this, &gpuEvent, lastRun, kernel.parameters().getExecNewVcop(),
                        kernel.parameters().getExecPfpaVcop())) {
        //! @todo A flush is necessary to make sure
        // that 2 consecutive runs won't access to the same
        // private/local memory. CAL has to generate cache flush
        // and wait for idle commands
        bool flush = ((cal()->iterations_ > 1) ||
                      dmaFlushMgmt_.isCbReady(*this, global.product(), gpuKernel.instructionCnt()))
            ? true
            : false;

        // Update the global GPU event
        setGpuEvent(gpuEvent, flush);

        // This code for the kernel execution debugging
        if (dev().settings().debugFlags_ & Settings::LockGlobalMemory) {
          gpuKernel.debug(*this);
        }
      } else {
        result = false;
        break;
      }

      // Print the debug buffer output result
      if (printfDbg().output(*this, printfEnabled,
                             (cal()->iterations_ > 1) ? local : sizes.global(),
                             gpuKernel.prog().printfInfo())) {
        // Go to the next iteration
        ++i;
      } else {
        result = false;
        break;
      }
    }
    // Check if we have to make multiple iterations
    while (i < cal()->iterations_);
  } else {
    result = false;
  }

  if (!result) {
    LogPrintfError("submitKernel failed to execute the \"%s\" kernel on HW!",
                   gpuKernel.name().data());
  }

  return result;
}

void VirtualGPU::submitNativeFn(amd::NativeFnCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  Unimplemented();  //!< @todo: Unimplemented
}

void VirtualGPU::submitMarker(amd::Marker& vcmd) {
  //!@note runtime doesn't need to lock this command on execution

  if (vcmd.waitingEvent() != NULL) {
    bool foundEvent = false;

    // Loop through all outstanding command batches
    while (!cbList_.empty()) {
      const auto it = cbList_.cbegin();
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

void VirtualGPU::releaseMemory(gslMemObject gslResource, bool wait) {
  bool result = true;
  if (wait) {
    waitForEvent(&gpuEvents_[gslResource]);
  }

  // Unbind resource if it's active kernel desc
  for (uint i = 0; i < MaxUavArguments; ++i) {
    if (gslResource == cal_.uavs_[i]) {
      result = setUAVBuffer(i, 0, GSL_UAV_TYPE_UNKNOWN);
      cal_.uavs_[i] = 0;
    }
  }
  for (uint i = 0; i < MaxReadImage; ++i) {
    if (gslResource == cal_.readImages_[i]) {
      result = setInput(i, 0);
      cal_.readImages_[i] = 0;
    }
  }
  for (uint i = 0; i < MaxConstBuffers; ++i) {
    if (gslResource == cal_.constBuffers_[i]) {
      result = setConstantBuffer(i, 0, 0, 0);
      cal_.constBuffers_[i] = 0;
    }
  }

  if ((dev().scratch(hwRing()) != NULL) && (dev().scratch(hwRing())->regNum_ > 0)) {
    // Unbind scratch memory
    const Device::ScratchBuffer* scratch = dev().scratch(hwRing());
    if ((scratch->memObj_ != NULL) && (scratch->memObj_->gslResource() == gslResource)) {
      setScratchBuffer(NULL, 0);
    }
  }

  gpuEvents_.erase(gslResource);
}

void VirtualGPU::releaseKernel(CALimage calImage) {
  GslKernelDesc* desc = gslKernels_[calImage];
  if (desc != NULL) {
    freeKernelDesc(desc);
  }
  gslKernels_.erase(calImage);
}

void VirtualGPU::submitPerfCounter(amd::PerfCounterCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  gslQueryObject gslCounter;

  const amd::PerfCounterCommand::PerfCounterList counters = vcmd.getCounters();

  // Create a HW counter
  gslCounter = cs()->createQuery(GSL_PERFORMANCE_COUNTERS_ATI);
  if (0 == gslCounter) {
    LogError("We failed to allocate memory for the GPU perfcounter");
    vcmd.setStatus(CL_INVALID_OPERATION);
    return;
  }
  CalCounterReference* calRef = new CalCounterReference(*this, gslCounter);
  if (calRef == NULL) {
    LogError("We failed to allocate memory for the GPU perfcounter");
    vcmd.setStatus(CL_INVALID_OPERATION);
    return;
  }
  gslCounter = 0;

  for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
    amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[i]);
    const PerfCounter* counter = static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());

    // Make sure we have a valid gpu performance counter
    if (NULL == counter) {
      amd::PerfCounter::Properties prop = amdCounter->properties();
      PerfCounter* gpuCounter = new PerfCounter(
          gpuDevice_, *this, prop[CL_PERFCOUNTER_GPU_BLOCK_INDEX],
          prop[CL_PERFCOUNTER_GPU_COUNTER_INDEX], prop[CL_PERFCOUNTER_GPU_EVENT_INDEX]);
      if (NULL == gpuCounter) {
        LogError("We failed to allocate memory for the GPU perfcounter");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      } else if (gpuCounter->create(calRef)) {
        amdCounter->setDeviceCounter(gpuCounter);
      } else {
        LogPrintfError(
            "We failed to allocate a perfcounter in CAL.\
                    Block: %d, counter: #d, event: %d",
            gpuCounter->info()->blockIndex_, gpuCounter->info()->counterIndex_,
            gpuCounter->info()->eventIndex_);
        delete gpuCounter;
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
      counter = gpuCounter;
    }
  }

  calRef->release();

  for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
    amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[i]);
    const PerfCounter* counter = static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());

    if (gslCounter != counter->gslCounter()) {
      gslCounter = counter->gslCounter();
      // Find the state and sends the command to CAL
      if (vcmd.getState() == amd::PerfCounterCommand::Begin) {
        gslCounter->BeginQuery(cs(), GSL_PERFORMANCE_COUNTERS_ATI, 0);
      } else if (vcmd.getState() == amd::PerfCounterCommand::End) {
        GpuEvent event;
        eventBegin(MainEngine);
        gslCounter->EndQuery(cs(), 0);
        eventEnd(MainEngine, event);
        setGpuEvent(event);
      } else {
        LogError("Unsupported performance counter state");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }
  }
}
void VirtualGPU::submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  switch (cmd.type()) {
    case CL_COMMAND_THREAD_TRACE_MEM: {
      amd::ThreadTrace* amdThreadTrace = &cmd.getThreadTrace();
      ThreadTrace* threadTrace = static_cast<ThreadTrace*>(amdThreadTrace->getDeviceThreadTrace());

      if (threadTrace == NULL) {
        gslQueryObject gslThreadTrace;
        // Create a HW thread trace query object
        gslThreadTrace = cs()->createQuery(GSL_SHADER_TRACE_BYTES_WRITTEN);
        if (0 == gslThreadTrace) {
          LogError("Failure in memory allocation for the GPU threadtrace");
          cmd.setStatus(CL_INVALID_OPERATION);
          return;
        }
        CalThreadTraceReference* calRef = new CalThreadTraceReference(*this, gslThreadTrace);
        if (calRef == NULL) {
          LogError("Failure in memory allocation for the GPU threadtrace");
          cmd.setStatus(CL_INVALID_OPERATION);
          return;
        }
        size_t seNum = amdThreadTrace->deviceSeNumThreadTrace();
        ThreadTrace* gpuThreadTrace = new ThreadTrace(gpuDevice_, *this, seNum);
        if (NULL == gpuThreadTrace) {
          LogError("Failure in memory allocation for the GPU threadtrace");
          cmd.setStatus(CL_INVALID_OPERATION);
          return;
        }
        if (gpuThreadTrace->create(calRef)) {
          amdThreadTrace->setDeviceThreadTrace(gpuThreadTrace);
        } else {
          LogError("Failure in memory allocation for the GPU threadtrace");
          delete gpuThreadTrace;
          cmd.setStatus(CL_INVALID_OPERATION);
          return;
        }
        threadTrace = gpuThreadTrace;
        calRef->release();
      }
      gslShaderTraceBufferObject* threadTraceBufferObjects =
          threadTrace->getThreadTraceBufferObjects();
      const size_t memObjSize = cmd.getMemoryObjectSize();
      const std::vector<amd::Memory*>& memObj = cmd.getMemList();
      size_t se = 0;
      for (auto itMemObj = memObj.cbegin();
           itMemObj != memObj.cend(); ++itMemObj, ++se) {
        // Find GSL Mem Object
        gslMemObject gslMemObj = dev().getGpuMemory(*itMemObj)->gslResource();

        // Bind GSL MemObject to the appropriate SE Thread Trace Buffer Object
        threadTraceBufferObjects[se]->attachMemObject(cs(), gslMemObj, 0, 0, memObjSize, se);
      }
      break;
    }
    default:
      LogError("Unsupported command type for ThreadTraceMemObjects!");
      break;
  }
}

void VirtualGPU::submitThreadTrace(amd::ThreadTraceCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  switch (cmd.type()) {
    case CL_COMMAND_THREAD_TRACE: {
      amd::ThreadTrace* amdThreadTrace = static_cast<amd::ThreadTrace*>(&cmd.getThreadTrace());
      ThreadTrace* threadTrace = static_cast<ThreadTrace*>(amdThreadTrace->getDeviceThreadTrace());

      // gpu thread trace object had to be generated prior to begin/end/pause/resume due
      // to ThreadTraceMemObjectsCommand execution
      if (threadTrace == NULL) {
        return;
      } else {
        gslQueryObject gslThreadTrace;
        gslThreadTrace = threadTrace->gslThreadTrace();
        uint32_t seNum = amdThreadTrace->deviceSeNumThreadTrace();

        // Find the state and sends the commands to GSL
        if (cmd.getState() == amd::ThreadTraceCommand::Begin) {
          amd::ThreadTrace::ThreadTraceConfig* traceCfg =
              static_cast<amd::ThreadTrace::ThreadTraceConfig*>(cmd.threadTraceConfig());
          const gslErrorCode ec =
              gslThreadTrace->BeginQuery(cs(), GSL_SHADER_TRACE_BYTES_WRITTEN, 0);
          assert(ec == GSL_NO_ERROR);

          for (uint32_t idx = 0; idx < seNum; ++idx) {
            rs()->enableShaderTrace(cs(), idx, true);
            rs()->setShaderTraceComputeUnit(idx, traceCfg->cu_);
            rs()->setShaderTraceShaderArray(idx, traceCfg->sh_);
            rs()->setShaderTraceSIMDMask(idx, traceCfg->simdMask_);
            rs()->setShaderTraceVmIdMask(idx, traceCfg->vmIdMask_);
            rs()->setShaderTraceTokenMask(idx, traceCfg->tokenMask_);
            rs()->setShaderTraceRegisterMask(idx, traceCfg->regMask_);
            rs()->setShaderTraceIssueMask(idx, traceCfg->instMask_);
            rs()->setShaderTraceRandomSeed(idx, traceCfg->randomSeed_);
            rs()->setShaderTraceCaptureMode(idx, traceCfg->captureMode_);
            rs()->setShaderTraceWrap(idx, traceCfg->isWrapped_);
            rs()->setShaderTraceUserData(idx, (traceCfg->isUserData_) ? traceCfg->userData_ : 0);
          }
        } else if (cmd.getState() == amd::ThreadTraceCommand::End) {
          for (uint32_t idx = 0; idx < seNum; ++idx) {
            rs()->enableShaderTrace(cs(), idx, false);
          }
          gslThreadTrace->EndQuery(cs(), 0);
        } else if (cmd.getState() == amd::ThreadTraceCommand::Pause) {
          for (uint32_t idx = 0; idx < seNum; ++idx) {
            rs()->setShaderTraceIsPaused(cs(), idx, true);
          }
        } else if (cmd.getState() == amd::ThreadTraceCommand::Resume) {
          for (uint32_t idx = 0; idx < seNum; ++idx) {
            rs()->setShaderTraceIsPaused(cs(), idx, false);
          }
        }
      }
      break;
    }
    default:
      LogError("Unsupported command type for ThreadTrace!");
      break;
  }
}

void VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  for (const auto& it : vcmd.getMemList()) {
    // amd::Memory object should never be NULL
    assert(it && "Memory object for interop is NULL");
    gpu::Memory* memory = dev().getGpuMemory(it);

    // If resource is a shared copy of original resource, then
    // runtime needs to copy data from original resource
    it->getInteropObj()->copyOrigToShared();

    // Check if OpenCL has direct access to the interop memory
    if (memory->interopType() == Memory::InteropDirectAccess) {
      continue;
    }

    // Does interop use HW emulation?
    if (memory->interopType() == Memory::InteropHwEmulation) {
      static const bool Entire = true;
      amd::Coord3D origin(0, 0, 0);
      amd::Coord3D region(memory->size());

      // Synchronize the object
      if (!blitMgr().copyBuffer(*memory->interop(), *memory, origin, origin, region, Entire)) {
        LogError("submitAcquireExtObjects - Interop synchronization failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  for (const auto& it : vcmd.getMemList()) {
    // amd::Memory object should never be NULL
    assert(it && "Memory object for interop is NULL");
    gpu::Memory* memory = dev().getGpuMemory(it);

    // Check if we can use HW interop
    if (memory->interopType() == Memory::InteropHwEmulation) {
      static const bool Entire = true;
      amd::Coord3D origin(0, 0, 0);
      amd::Coord3D region(memory->size());

      // Synchronize the object
      if (!blitMgr().copyBuffer(*memory, *memory->interop(), origin, origin, region, Entire)) {
        LogError("submitReleaseExtObjects interop synchronization failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    } else {
      if (memory->interopType() != Memory::InteropDirectAccess) {
        LogError("None interop release!");
      }
    }

    // If resource is a shared copy of original resource, then
    // runtime needs to copy data back to original resource
    it->getInteropObj()->copySharedToOrig();
  }

  profilingEnd(vcmd);
}

void VirtualGPU::submitSignal(amd::SignalCommand& vcmd) {
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);
  gpu::Memory* gpuMemory = dev().getGpuMemory(&vcmd.memory());
  GpuEvent gpuEvent;
  eventBegin(MainEngine);
  if (vcmd.type() == CL_COMMAND_WAIT_SIGNAL_AMD) {
    uint64_t surfAddr = gpuMemory->gslResource()->getPhysicalAddress(cs());
    uint64_t markerAddr = gpuMemory->gslResource()->getMarkerAddress(cs());
    uint64_t markerOffset = markerAddr - surfAddr;
    cs()->p2pMarkerOp(gpuMemory->gslResource(), vcmd.markerValue(), markerOffset, false);
  } else if (vcmd.type() == CL_COMMAND_WRITE_SIGNAL_AMD) {
    static constexpr bool FlushL2 = true;
    flushCUCaches(FlushL2);
    cs()->p2pMarkerOp(gpuMemory->gslResource(), vcmd.markerValue(), vcmd.markerOffset(), true);
  }
  eventEnd(MainEngine, gpuEvent);
  gpuMemory->setBusy(*this, gpuEvent);
  // Update the global GPU event
  setGpuEvent(gpuEvent);

  profilingEnd(vcmd);
}

void VirtualGPU::submitMakeBuffersResident(amd::MakeBuffersResidentCommand& vcmd) {
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);
  std::vector<amd::Memory*> memObjects = vcmd.memObjects();
  uint32_t numObjects = memObjects.size();
  gslMemObject* pGSLMemObjects = new gslMemObject[numObjects];

  for (uint32_t i = 0; i < numObjects; ++i) {
    gpu::Memory* gpuMemory = dev().getGpuMemory(memObjects[i]);
    pGSLMemObjects[i] = gpuMemory->gslResource();
    gpuMemory->syncCacheFromHost(*this);
  }

  uint64* surfBusAddr = new uint64[numObjects];
  uint64* markerBusAddr = new uint64[numObjects];
  gslErrorCode res =
      cs()->makeBuffersResident(numObjects, pGSLMemObjects, surfBusAddr, markerBusAddr);
  if (res != GSL_NO_ERROR) {
    LogError("MakeBuffersResident failed");
    vcmd.setStatus(CL_INVALID_OPERATION);
  } else {
    cl_bus_address_amd* busAddr = vcmd.busAddress();
    for (uint32_t i = 0; i < numObjects; ++i) {
      busAddr[i].surface_bus_address = surfBusAddr[i];
      busAddr[i].marker_bus_address = markerBusAddr[i];
    }
  }
  delete[] pGSLMemObjects;
  delete[] surfBusAddr;
  delete[] markerBusAddr;
  profilingEnd(vcmd);
}


bool VirtualGPU::awaitCompletion(CommandBatch* cb, const amd::Event* waitingEvent) {
  bool found = false;
  amd::Command* current;
  amd::Command* head = cb->head_;

  // Make sure that profiling is enabled
  if (profileEnabled_) {
    return profilingCollectResults(cb, waitingEvent);
  }
  // Mark the first command in the batch as running
  if (head != NULL) {
    head->setStatus(CL_RUNNING);
  } else {
    return found;
  }

  // Wait for the last known GPU event
  waitEventLock(cb);

  while (NULL != head) {
    current = head->getNext();
    if (head->status() == CL_SUBMITTED) {
      head->setStatus(CL_RUNNING);
      head->setStatus(CL_COMPLETE);
    } else if (head->status() == CL_RUNNING) {
      head->setStatus(CL_COMPLETE);
    } else if ((head->status() != CL_COMPLETE) && (current != NULL)) {
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

void VirtualGPU::flush(amd::Command* list, bool wait) {
  CommandBatch* cb = NULL;
  bool gpuCommand = false;

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
  if (NULL != list) {
    cb = new CommandBatch(list, cal()->events_, cal()->lastTS_);
  }

  {
    //! @note: flushDMA() requires a lock, because GSL can
    //! defer destruction of internal memory objects and releases them
    //! on GSL flush. If runtime calls another GSL flush at the same time,
    //! then double release can occur.
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

  // Mark last TS as NULL, so runtime won't process empty batches with the old TS
  cal_.lastTS_ = NULL;
  if (NULL != cb) {
    cbList_.push_back(cb);
  }

  wait |= state_.forceWait_;
  // Loop through all outstanding command batches
  while (!cbList_.empty()) {
    const auto it = cbList_.cbegin();
    // Check if command batch finished without a wait
    bool finished = true;
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
    } else {
      // Early exit if no finished
      break;
    }
  }
  state_.forceWait_ = false;
}

void VirtualGPU::enableSyncedBlit() const { return blitMgr_->enableSynchronization(); }

void VirtualGPU::releaseMemObjects(bool scratch) {
  for (const auto& it : gpuEvents_) {
    GpuEvent event = it.second;
    waitForEvent(&event);
  }
  // Unbind all resources.So the queue won't have any bound mem objects
  for (uint i = 0; i < MaxUavArguments; ++i) {
    if (NULL != cal_.uavs_[i]) {
      setUAVBuffer(i, 0, GSL_UAV_TYPE_UNKNOWN);
      cal_.uavs_[i] = 0;
    }
  }
  for (uint i = 0; i < MaxReadImage; ++i) {
    if (NULL != cal_.readImages_[i]) {
      setInput(i, 0);
      cal_.readImages_[i] = 0;
    }
  }
  for (uint i = 0; i < MaxConstBuffers; ++i) {
    if (NULL != cal_.constBuffers_[i]) {
      setConstantBuffer(i, 0, 0, 0);
      cal_.constBuffers_[i] = 0;
    }
  }

  if (scratch) {
    setScratchBuffer(NULL, 0);
  }

  gpuEvents_.clear();
}

void VirtualGPU::setGpuEvent(GpuEvent gpuEvent, bool flush) {
  cal_.events_[engineID_] = gpuEvent;

  // Flush current DMA buffer if requested
  if (flush || GPU_FLUSH_ON_EXECUTION) {
    flushDMA(engineID_);
  }
}

void VirtualGPU::flushDMA(uint engineID) {
  if (engineID == MainEngine) {
    // Clear memory dependency state, since runtime flushes compute
    // memoryDependency().clear();
    //!@todo Keep memory dependency alive even if we flush DMA,
    //! since only L2 cache is flushed in KMD frame,
    //! but L1 still has to be invalidated.
  }
  //! \note Use CtxIsEventDone, so we won't flush compute for DRM engine
  isDone(&cal_.events_[engineID]);
}

bool VirtualGPU::waitAllEngines(CommandBatch* cb) {
  uint i;
  GpuEvent* events;  //!< GPU events for the batch
  // If command batch is NULL then wait for the current
  if (NULL == cb) {
    events = cal_.events_;
  } else {
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

void VirtualGPU::waitEventLock(CommandBatch* cb) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  bool earlyDone = waitAllEngines(cb);

  // Free resource cache if we have too many entries
  //! \note we do it here, when all engines are idle,
  // because Vista/Win7 idles GPU on a resource destruction
  static const size_t MinCacheEntries = 4096;
  dev().resourceCache().free(MinCacheEntries);

  // Find the timestamp object of the last command in the batch
  if (cb->lastTS_ != NULL) {
    // If earlyDone is TRUE, then CPU didn't wait for GPU.
    // Thus the sync point between CPU and GPU is unclear and runtime
    // will use an older adjustment value to maintain the same timeline
    if (!earlyDone ||
        //! \note Workaround for APU(s).
        //! GPU-CPU timelines may go off too much, thus always
        //! force calibration with the last batch in the list
        (cbList_.size() <= 1) || (readjustTimeGPU_ == 0)) {
      uint64_t startTimeStampGPU = 0;
      uint64_t endTimeStampGPU = 0;

      // Get the timestamp value of the last command in the batch
      cb->lastTS_->value(&startTimeStampGPU, &endTimeStampGPU);

      uint64_t endTimeStampCPU = amd::Os::timeNanos();
      // Make sure the command batch has a valid GPU TS
      if (!GPU_RAW_TIMESTAMP) {
        // Adjust the base time by the execution time
        readjustTimeGPU_ = endTimeStampGPU - endTimeStampCPU;
      }
    }
  }
}

void VirtualGPU::validateScratchBuffer(const Kernel* kernel) {
  // Check if a scratch buffer is required
  if (dev().scratch(hwRing())->regNum_ > 0) {
    // Setup scratch buffer
    setScratchBuffer(dev().scratch(hwRing())->memObj_->gslResource(), 0);
  }
}

bool VirtualGPU::setActiveKernelDesc(const amd::NDRangeContainer& sizes, const Kernel* kernel) {
  bool result = true;
  CALimage calImage = kernel->calImage();

  GslKernelDesc* desc = gslKernels_[calImage];

  validateScratchBuffer(kernel);

  // Early exit
  if ((activeKernelDesc_ == desc) && (desc != NULL)) {
    return result;
  }

  // Does the kernel descriptor for this virtual device exist?
  if (desc == NULL) {
    desc = allocKernelDesc(kernel, calImage);
    if (desc == NULL) {
      return false;
    }
    gslKernels_[calImage] = desc;
  }

  // Set the descriptor as active
  activeKernelDesc_ = desc;

  // Program the samplers defined in the kernel
  if (!kernel->setInternalSamplers(*this)) {
    result = false;
  }

  // Bind global HW constant buffers
  if (!kernel->bindGlobalHwCb(*this, desc)) {
    result = false;
  }

  if (result) {
    // Set program in GSL
    rs()->setCurrentProgramObject(GSL_COMPUTE_PROGRAM, desc->func_);

    // Update internal constant buffer
    if (desc->intCb_ != 0) {
      cs()->setIntConstants(GSL_COMPUTE_PROGRAM, desc->intCb_);
    }
  }

  return result;
}

bool VirtualGPU::allocConstantBuffers() {
  // Allocate/reallocate constant buffers
  size_t minCbSize;
  // GCN doesn't really have a limit
  minCbSize = 128 * Ki;
  uint i;

  // Create/reallocate constant buffer resources
  for (i = 0; i < MaxConstBuffersArguments; ++i) {
    ConstBuffer* constBuf = new ConstBuffer(
        *this, ((minCbSize + ConstBuffer::VectorSize - 1) / ConstBuffer::VectorSize));

    if ((constBuf != NULL) && constBuf->create()) {
      addConstBuffer(constBuf);
    } else {
      // We failed to create a constant buffer
      delete constBuf;
      return false;
    }
  }

  return true;
}

VirtualGPU::GslKernelDesc* VirtualGPU::allocKernelDesc(const Kernel* kernel, CALimage calImage) {
  // Sanity checks
  assert(kernel != NULL);
  GslKernelDesc* desc = new GslKernelDesc;

  if (desc != NULL) {
    memset(desc, 0, sizeof(GslKernelDesc));

    if (kernel->calImage() != calImage) {
      desc->image_ = calImage;
    }

    if (!moduleLoad(calImage, &desc->func_, &desc->intCb_)) {
      LogPrintfError("calModuleLoad failed for \"%s\" kernel!", kernel->name().c_str());
      delete desc;
      return NULL;
    }
  }

  if (kernel->argSize() > slots_.size()) {
    slots_.resize(kernel->argSize());
  }

  return desc;
}

void VirtualGPU::freeKernelDesc(VirtualGPU::GslKernelDesc* desc) {
  if (desc) {
    if (gslKernelDesc() == desc) {
      // Clear active kernel desc
      activeKernelDesc_ = NULL;
      rs()->setCurrentProgramObject(GSL_COMPUTE_PROGRAM, 0);
    }

    if (desc->image_ != 0) {
      // Free CAL image
      free(desc->image_);
    }

    if (desc->func_ != 0) {
      if (desc->intCb_ != 0) {
        cs()->setIntConstants(GSL_COMPUTE_PROGRAM, 0);
        cs()->destroyMemObject(desc->intCb_);
      }
      cs()->destroyProgramObject(desc->func_);
    }

    delete desc;
  }
}

void VirtualGPU::profilingBegin(amd::Command& command, bool drmProfiling) {
  // Is profiling enabled?
  if (command.profilingInfo().enabled_) {
    // Allocate a timestamp object from the cache
    TimeStamp* ts = tsCache_->allocTimeStamp();
    if (NULL == ts) {
      return;
    }
    // Save the TimeStamp object in the current OCL event
    command.setData(ts);
    currTs_ = ts;
    profileEnabled_ = true;
  }
}

void VirtualGPU::profilingEnd(amd::Command& command) {
  // Get the TimeStamp object associated witht the current command
  TimeStamp* ts = reinterpret_cast<TimeStamp*>(command.data());
  if (ts != NULL) {
    // Check if the command actually did any GPU submission
    if (ts->isValid()) {
      cal_.lastTS_ = ts;
    } else {
      // Destroy the TimeStamp object
      tsCache_->freeTimeStamp(ts);
      command.setData(NULL);
    }
  }
}

bool VirtualGPU::profilingCollectResults(CommandBatch* cb, const amd::Event* waitingEvent) {
  bool found = false;
  amd::Command* current;
  amd::Command* first = cb->head_;

  // If the command list is, empty then exit
  if (NULL == first) {
    return found;
  }

  // Wait for the last known GPU events on all engines
  waitEventLock(cb);

  // Find the CPU base time of the entire command batch execution
  uint64_t endTimeStamp = amd::Os::timeNanos();
  uint64_t startTimeStamp = endTimeStamp;

  // First step, walk the command list to find the first valid command
  //! \note The batch may have empty markers at the beginning.
  //! So the start/end of the empty commands is equal to
  //! the start of the first valid command in the batch.
  first = cb->head_;
  while (NULL != first) {
    // Get the TimeStamp object associated witht the current command
    TimeStamp* ts = reinterpret_cast<TimeStamp*>(first->data());

    if (ts != NULL) {
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
  while (NULL != first) {
    // Get the TimeStamp object associated witht the current command
    TimeStamp* ts = reinterpret_cast<TimeStamp*>(first->data());

    current = first->getNext();

    if (ts != NULL) {
      ts->value(&startTimeStamp, &endTimeStamp);
      endTimeStamp -= readjustTimeGPU_;
      startTimeStamp -= readjustTimeGPU_;
      // Destroy the TimeStamp object
      tsCache_->freeTimeStamp(ts);
      first->setData(NULL);
    } else {
      // For empty commands start/end is equal to
      // the end of the last valid command
      startTimeStamp = endTimeStamp;
    }

    // Update the command status with the proper timestamps
    if (first->status() == CL_SUBMITTED) {
      first->setStatus(CL_RUNNING, startTimeStamp);
      first->setStatus(CL_COMPLETE, endTimeStamp);
    } else if (first->status() == CL_RUNNING) {
      first->setStatus(CL_COMPLETE, endTimeStamp);
    } else if ((first->status() != CL_COMPLETE) && (current != NULL)) {
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

bool VirtualGPU::addVmMemory(const Memory* memory) {
  uint* cnt = &cal_.memCount_;
  (*cnt)++;
  // Reallocate array if kernel uses more memory objects
  if (numVmMems_ < *cnt) {
    gslMemObject* tmp;
    tmp = new gslMemObject[*cnt];
    if (tmp == NULL) {
      return false;
    }
    memcpy(tmp, vmMems_, sizeof(gslMemObject) * numVmMems_);
    delete[] vmMems_;
    vmMems_ = tmp;
    numVmMems_ = *cnt;
  }
  vmMems_[*cnt - 1] = memory->gslResource();

  return true;
}

void VirtualGPU::profileEvent(EngineType engine, bool type) const {
  if (NULL == currTs_) {
    return;
  }
  if (type) {
    currTs_->begin((engine == SdmaEngine) ? true : false);
  } else {
    currTs_->end((engine == SdmaEngine) ? true : false);
  }
}

bool VirtualGPU::processMemObjectsHSA(const amd::Kernel& kernel, const_address params,
                                      bool nativeMem, std::vector<const Memory*>* memList) {
  const HSAILKernel& hsaKernel =
      static_cast<const HSAILKernel&>(*(kernel.getDeviceKernel(dev())));
  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  // Mark the tracker with a new kernel,
  // so we can avoid checks of the aliased objects
  memoryDependency().newKernel();

  bool deviceSupportFGS = 0 != dev().isFineGrainedSystem(true);
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

  amd::Memory* memory = NULL;
  // get svm non arugment information
  void* const* svmPtrArray = reinterpret_cast<void* const*>(params + execInfoOffset);
  for (size_t i = 0; i < count; i++) {
    memory = amd::MemObjMap::FindMemObj(svmPtrArray[i]);
    if (NULL == memory) {
      if (!supportFineGrainedSystem) {
        return false;
      } else if (sync) {
        flushCUCaches();
        // Clear memory dependency state
        const static bool All = true;
        memoryDependency().clear(!All);
        continue;
      }
    } else {
      Memory* gpuMemory = dev().getGpuMemory(memory);
      if (NULL != gpuMemory) {
        // Synchronize data with other memory instances if necessary
        gpuMemory->syncCacheFromHost(*this);

        const static bool IsReadOnly = false;
        // Validate SVM passed in the non argument list
        memoryDependency().validate(*this, gpuMemory, IsReadOnly);

        // Mark signal write for cache coherency,
        // since this object isn't a part of kernel arg setup
        if ((memory->getMemFlags() & CL_MEM_READ_ONLY) == 0) {
          memory->signalWrite(&dev());
        }

        memList->push_back(gpuMemory);
      } else {
        return false;
      }
    }
  }

  amd::Memory* const* memories =
      reinterpret_cast<amd::Memory* const*>(params + kernelParams.memoryObjOffset());
  // Check all parameters for the current kernel
  for (size_t i = 0; i < signature.numParameters(); ++i) {
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    const HSAILKernel::Argument* arg = hsaKernel.argument(i);
    Memory* gpuMem = nullptr;
    bool readOnly = false;
    amd::Memory* mem = nullptr;

    // Find if current argument is a buffer
    if ((desc.type_ == T_POINTER) && (arg->addrQual_ != HSAIL_ADDRESS_LOCAL)) {
      uint32_t index = desc.info_.arrayIndex_;
      if (nativeMem) {
        gpuMem = reinterpret_cast<Memory* const*>(memories)[index];
        if (nullptr != gpuMem) {
          mem = gpuMem->owner();
        }
      } else {
        mem = memories[index];
        if (mem != nullptr) {
          gpuMem = dev().getGpuMemory(mem);
          // Synchronize data with other memory instances if necessary
          gpuMem->syncCacheFromHost(*this);
        }
      }
      //! This condition is for SVM fine-grain
      if ((gpuMem == nullptr) && dev().isFineGrainedSystem(true)) {
        flushCUCaches();
        // Clear memory dependency state
        const static bool All = true;
        memoryDependency().clear(!All);
        continue;
      } else if (gpuMem != nullptr) {
        // Check image
        readOnly = (desc.accessQualifier_ == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
        // Check buffer
        readOnly |= (arg->access_ == HSAIL_ACCESS_TYPE_RO) ? true : false;
        // Validate memory for a dependency in the queue
        memoryDependency().validate(*this, gpuMem, readOnly);
      }
    }
  }

  for (gpu::Memory* mem : hsaKernel.prog().globalStores()) {
    const static bool IsReadOnly = false;
    // Validate global store for a dependency in the queue
    memoryDependency().validate(*this, mem, IsReadOnly);
  }

  return true;
}

amd::Memory* VirtualGPU::createBufferFromImage(amd::Memory& amdImage) {
  amd::Memory* mem = new (amdImage.getContext()) amd::Buffer(amdImage, 0, 0, amdImage.getSize());
  mem->setVirtualDevice(this);
  if ((mem != NULL) && !mem->create()) {
    mem->release();
  }

  return mem;
}

void VirtualGPU::writeVQueueHeader(VirtualGPU& hostQ, uint64_t kernelTable) {
  const static bool Wait = true;
  vqHeader_->kernel_table = kernelTable;
  virtualQueue_->writeRawData(hostQ, sizeof(AmdVQueueHeader), vqHeader_, !Wait);
}

void VirtualGPU::flushCuCaches(HwDbgGpuCacheMask cache_mask) {
  //! @todo:  fix issue of no event available for the flush/invalidate cache command
  InvalidateSqCaches(cache_mask.sqICache_, cache_mask.sqKCache_, cache_mask.tcL1_,
                     cache_mask.tcL2_);

  flushDMA(engineID_);

  return;
}

void VirtualGPU::buildKernelInfo(const HSAILKernel& hsaKernel, hsa_kernel_dispatch_packet_t* aqlPkt,
                                 HwDbgKernelInfo& kernelInfo, amd::Event* enqueueEvent) {
  amd::HwDebugManager* dbgManager = dev().hwDebugMgr();
  assert(dbgManager && "No HW Debug Manager!");

  // Initialize structure with default values

  if (hsaKernel.prog().maxScratchRegs() > 0) {
    gpu::Memory* scratchBuf = dev().scratch(hwRing())->memObj_;
    kernelInfo.scratchBufAddr = scratchBuf->vmAddress();
    kernelInfo.scratchBufferSizeInBytes = scratchBuf->size();

    // Get the address of the scratch buffer and its size for CPU access
    address scratchRingAddr = NULL;
    scratchRingAddr = static_cast<address>(scratchBuf->map(NULL, 0));
    dbgManager->setScratchRing(scratchRingAddr, scratchBuf->size());
    scratchBuf->unmap(NULL);
  } else {
    kernelInfo.scratchBufAddr = 0;
    kernelInfo.scratchBufferSizeInBytes = 0;
    dbgManager->setScratchRing(NULL, 0);
  }


  //! @todo:  need to verify what is wanted for the global memory
  kernelInfo.heapBufAddr = (dev().globalMem()).vmAddress();

  kernelInfo.pAqlDispatchPacket = aqlPkt;
  kernelInfo.pAqlQueuePtr = reinterpret_cast<void*>(hsaQueueMem_->vmAddress());

  // Get the address of the kernel code and its size for CPU access
  gpu::Memory* aqlCode = hsaKernel.gpuAqlCode();
  if (NULL != aqlCode) {
    address aqlCodeAddr = static_cast<address>(aqlCode->map(NULL, 0));
    dbgManager->setKernelCodeInfo(aqlCodeAddr, hsaKernel.aqlCodeSize());
    aqlCode->unmap(NULL);
  } else {
    dbgManager->setKernelCodeInfo(NULL, 0);
  }

  kernelInfo.trapPresent = false;
  kernelInfo.trapHandler = NULL;
  kernelInfo.trapHandlerBuffer = NULL;

  kernelInfo.excpEn = 0;
  kernelInfo.cacheDisableMask = 0;
  kernelInfo.sqDebugMode = 0;

  kernelInfo.mgmtSe0Mask = 0xFFFFFFFF;
  kernelInfo.mgmtSe1Mask = 0xFFFFFFFF;

  // set kernel info for HW debug and call the callback function
  if (NULL != dbgManager->preDispatchCallBackFunc()) {
    DebugToolInfo dbgSetting = {0};
    dbgSetting.scratchAddress_ = kernelInfo.scratchBufAddr;
    dbgSetting.scratchSize_ = kernelInfo.scratchBufferSizeInBytes;
    dbgSetting.globalAddress_ = kernelInfo.heapBufAddr;
    dbgSetting.aclBinary_ = hsaKernel.prog().binaryElf();
    dbgSetting.event_ = enqueueEvent;

    // Call the predispatch callback function & set the trap info
    AqlCodeInfo aqlCodeInfo;
    aqlCodeInfo.aqlCode_ = (amd_kernel_code_t*)hsaKernel.cpuAqlCode();
    aqlCodeInfo.aqlCodeSize_ = hsaKernel.aqlCodeSize();

    // Execute the pre-dispatch call back function
    dbgManager->executePreDispatchCallBack(reinterpret_cast<void*>(aqlPkt), &dbgSetting);

    // assign the debug TMA and TBA for kernel dispatch
    if (NULL != dbgSetting.trapHandler_ && NULL != dbgSetting.trapBuffer_) {
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

    // flush/invalidate the instruction, data, L1 and L2 caches
    InvalidateSqCaches();
  }
}

void VirtualGPU::assignDebugTrapHandler(const DebugToolInfo& dbgSetting,
                                        HwDbgKernelInfo& kernelInfo) {
  // setup the runtime trap handler code and trap buffer to be assigned before kernel dispatching
  //
  Memory* rtTrapHandlerMem = static_cast<Memory*>(dev().hwDebugMgr()->runtimeTBA());
  Memory* rtTrapBufferMem = static_cast<Memory*>(dev().hwDebugMgr()->runtimeTMA());

  kernelInfo.trapHandler = reinterpret_cast<void*>(rtTrapHandlerMem->vmAddress() + TbaStartOffset);
  // With the TMA corruption hw bug workaround, the trap handler buffer can be set to zero.
  // However, by setting the runtime trap buffer (TMA) correct, the runtime trap hander
  // without the workaround can still function correctly.
  kernelInfo.trapHandlerBuffer = reinterpret_cast<void*>(rtTrapBufferMem->vmAddress());

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
  uint64_t* rtTmaPtr = reinterpret_cast<uint64_t*>(rtTrapBufferAddress + 0x18);
  rtTmaPtr[0] = tbaAddress;
  rtTmaPtr[1] = tmaAddress;

  rtTrapBufferMem->unmap(NULL);

  // Add GSL handle to the memory list for VidMM
  addVmMemory(trapHandlerMem);
  addVmMemory(trapBufferMem);
  addVmMemory(rtTrapHandlerMem);
  addVmMemory(rtTrapBufferMem);
}

void VirtualGPU::submitTransferBufferFromFile(amd::TransferBufferFileCommand& cmd) {
  size_t copySize = cmd.size()[0];
  size_t fileOffset = cmd.fileOffset();
  Memory* mem = dev().getGpuMemory(&cmd.memory());
  uint idx = 0;

  assert((cmd.type() == CL_COMMAND_READ_SSG_FILE_AMD) ||
         (cmd.type() == CL_COMMAND_WRITE_SSG_FILE_AMD));
  const bool writeBuffer(cmd.type() == CL_COMMAND_READ_SSG_FILE_AMD);

  if (writeBuffer) {
    size_t dstOffset = cmd.origin()[0];
    while (copySize > 0) {
      Memory* staging = dev().getGpuMemory(&cmd.staging(idx));
      size_t dstSize = amd::TransferBufferFileCommand::StagingBufferSize;
      dstSize = std::min(dstSize, copySize);
      void* dstBuffer = staging->cpuMap(*this);
      if (!cmd.file()->transferBlock(writeBuffer, dstBuffer, staging->size(), fileOffset, 0,
                                     dstSize)) {
        cmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
      staging->cpuUnmap(*this);

      bool result = blitMgr().copyBuffer(*staging, *mem, 0, dstOffset, dstSize, false);
      flushDMA(getGpuEvent(staging->gslResource())->engineId_);
      fileOffset += dstSize;
      dstOffset += dstSize;
      copySize -= dstSize;
    }
  } else {
    size_t srcOffset = cmd.origin()[0];
    while (copySize > 0) {
      Memory* staging = dev().getGpuMemory(&cmd.staging(idx));
      size_t srcSize = amd::TransferBufferFileCommand::StagingBufferSize;
      srcSize = std::min(srcSize, copySize);
      bool result = blitMgr().copyBuffer(*mem, *staging, srcOffset, 0, srcSize, false);

      void* srcBuffer = staging->cpuMap(*this);
      if (!cmd.file()->transferBlock(writeBuffer, srcBuffer, staging->size(), fileOffset, 0,
                                     srcSize)) {
        cmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
      staging->cpuUnmap(*this);

      fileOffset += srcSize;
      srcOffset += srcSize;
      copySize -= srcSize;
    }
  }
}

}  // namespace gpu
