/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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

#include "platform/commandqueue.hpp"
#include "device/device.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palmemory.hpp"
#include "device/pal/paltrap.hpp"
#include "device/pal/paldebugmanager.hpp"
#include <iostream>
#include <sstream>
#include <fstream>

namespace pal {

class VirtualGPU;
class Device;
class Memory;

/*
 ***************************************************************************
 *                  Implementation of GPU Debug Manager class
 ***************************************************************************
 */

GpuDebugManager::GpuDebugManager(amd::Device* device)
    : HwDebugManager(device),
      vGpu_(nullptr),
      debugMessages_(0),
      addressWatch_(nullptr),
      addressWatchSize_(0),
      oclEventHandle_(nullptr) {
  // Initialize the exception info and the kernel execution mode
  excpPolicy_.exceptionMask = 0x0;
  excpPolicy_.waveAction = CL_DBG_WAVES_RESUME;
  excpPolicy_.hostAction = CL_DBG_HOST_IGNORE;
  excpPolicy_.waveMode = CL_DBG_WAVEMODE_BROADCAST;

  execMode_.ui32All = 0;

  rtTrapHandlerInfo_.trap_.trapHandler_ = nullptr;
  rtTrapHandlerInfo_.trap_.trapBuffer_ = nullptr;

  aqlPacket_ = (hsa_kernel_dispatch_packet_t*)nullptr;

  return;
}

GpuDebugManager::~GpuDebugManager() {
  if (nullptr != addressWatch_) {
    delete[] addressWatch_;
  }
}

void GpuDebugManager::executePreDispatchCallBack(void* aqlPacket, void* toolInfo) {
  DebugToolInfo* info = reinterpret_cast<DebugToolInfo*>(toolInfo);

  aqlPacket_ = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(aqlPacket);
  Unimplemented();
  // Only if the pre-dispatch callback is set, will we update cache
  // flush configuration and build the memory descriptor.
  if (nullptr != preDispatchCallBackFunc_) {
    /*
            // Build the scratch memory descriptor
            device()->gslCtx()->BuildScratchBufferResource(debugInfo_.scratchMemoryDescriptor_,
                                              info->scratchAddress_,
                                              info->scratchSize_);

            // Build the global memory descriptor
            device()->gslCtx()->BuildHeapBufferResource(debugInfo_.globalMemoryDescriptor_,
                                           info->globalAddress_);
    */
    //      // for invalidate cache (BuildEndOfKernelNotifyCommands)
    //        aqlPacket->release_fence_scope = 2;

    aclBinary_ = reinterpret_cast<void*>(info->aclBinary_);
    oclEventHandle_ = reinterpret_cast<void*>(as_cl(info->event_));

    cl_device_id clDeviceId = as_cl(device_);
    preDispatchCallBackFunc_(clDeviceId, oclEventHandle_, aqlPacket_, aclBinary_,
                             preDispatchCallBackArgs_);
  }

  // setup the trap handler information only if the debugger has been registered
  if (isRegistered()) {
    // Copy the various info set by the debugger/profiler to the tool info structure
    setupTrapInformation(info);
  }
}

void GpuDebugManager::executePostDispatchCallBack() {
  if (nullptr != postDispatchCallBackFunc_) {
    cl_device_id clDeviceId = as_cl(device_);
    postDispatchCallBackFunc_(clDeviceId, aqlPacket_->completion_signal.handle,
                              postDispatchCallBackArgs_);
  }
}

//!  Map the kernel code for host access
void GpuDebugManager::mapKernelCode(void* aqlCodeInfo) const {
  AqlCodeInfo* codeInfo = reinterpret_cast<AqlCodeInfo*>(aqlCodeInfo);

  codeInfo->aqlCode_ = reinterpret_cast<amd_kernel_code_t*>(aqlCodeAddr_);
  codeInfo->aqlCodeSize_ = aqlCodeSize_;
}

int32_t GpuDebugManager::registerDebugger(amd::Context* context, uintptr_t messageStorage) {
  if (!device()->settings().enableHwDebug_) {
    LogError("debugmanager: Register debugger error - HW DEBUG is not enable");
    return CL_DEBUGGER_REGISTER_FAILURE_AMD;
  }

  // first time register - set the message storage, flush queue and enable hw debug
  if (!isRegistered()) {
    debugMessages_ = messageStorage;
    Unimplemented();
    /*
            if (!device()->gslCtx()->registerHwDebugger(debugMessages_)) {
                LogError("debugmanager: Register debugger failed");
                return CL_OUT_OF_RESOURCES;
            }
    */
    isRegistered_ = true;

    if (CL_SUCCESS != createRuntimeTrapHandler()) {
      LogError("debugmanager: Create runtime trap handler failed");
      return CL_OUT_OF_RESOURCES;
    }
  }

  context_ = context;

  return CL_SUCCESS;
}

void GpuDebugManager::unregisterDebugger() {
  if (isRegistered()) {
    // reset the debugger registration flag
    isRegistered_ = false;
    context_ = nullptr;
  }
}

void GpuDebugManager::flushCache(uint32_t mask) {
  HwDbgGpuCacheMask cacheMask(mask);
  // device()->xferQueue()->flushCuCaches(cacheMask);
}


void GpuDebugManager::setupTrapInformation(DebugToolInfo* toolInfo) {
  toolInfo->scratchAddress_ = 0;
  toolInfo->scratchSize_ = 0;
  toolInfo->globalAddress_ = 0;
  toolInfo->sqPerfcounterEnable_ = false;

  // Set up trap related info in the kernel info structure to be
  // used in the kernel dispatch.
  toolInfo->exceptionMask_ = excpPolicy_.exceptionMask;
  toolInfo->gpuSingleStepMode_ = execMode_.gpuSingleStepMode;
  toolInfo->monitorMode_ = execMode_.monitorMode;

  // The order of these three bits is determined by the definition
  // of the register COMPUTE_DISPATCH_INITIATOR
  toolInfo->cacheDisableMask_ = ((execMode_.disableL1Scalar << 2) |
                                 (execMode_.disableL2Cache << 1) | (execMode_.disableL1Vector));

  toolInfo->reservedCuNum_ = execMode_.reservedCuNum;

  toolInfo->trapHandler_ = rtTrapInfo_[kDebugTrapHandlerLocation];
  toolInfo->trapBuffer_ = rtTrapInfo_[kDebugTrapBufferLocation];
}

void GpuDebugManager::getPacketAmdInfo(const void* aqlCodeInfo, void* packetInfo) const

{
  const AqlCodeInfo* codeInfo = reinterpret_cast<const AqlCodeInfo*>(aqlCodeInfo);

  const amd_kernel_code_t* hostAqlCode = codeInfo->aqlCode_;

  PacketAmdInfo* packet = reinterpret_cast<PacketAmdInfo*>(packetInfo);

  const amd_kernel_code_t* akc = hostAqlCode;

  packet->numberOfSgprs_ = akc->wavefront_sgpr_count;
  packet->numberOfVgprs_ = akc->workitem_vgpr_count;

  //  use mapped kernel_object_address for host accessing of ISA buffer
  packet->pointerToIsaBuffer_ = (char*)(hostAqlCode) + akc->kernel_code_entry_byte_offset;

  packet->scratchBufferWaveOffset_ = akc->debug_wavefront_private_segment_offset_sgpr;

  packet->sizeOfIsaBuffer_ = codeInfo->aqlCodeSize_;

  packet->sizeOfStaticGroupMemory_ = akc->workgroup_group_segment_byte_size;

  // The trap_reserved_vgpr_index will be 4 less the original
  // This value must be used only by the debugger
  packet->trapReservedVgprIndex_ = akc->workitem_vgpr_count - NumberReserveVgprs;
}

DebugEvent GpuDebugManager::createDebugEvent(const bool autoReset) {
  Unimplemented();
  /*
      // create the event object
      osEventHandle shaderEvent = osEventCreate(!autoReset);

      // event object has been created, set the initial state
      if (shaderEvent != 0) {

          osEventReset(shaderEvent);   // initial state is non-signaled

          if (device()->gslCtx()->exceptionNotification(shaderEvent)) {
              return shaderEvent;
          }
      }
  */
  return 0;
}

int32_t GpuDebugManager::waitDebugEvent(DebugEvent pEvent, uint32_t timeOut) const {
  Unimplemented();
  /*
      if (osEventTimedWait(pEvent, timeOut)) {
          return CL_SUCCESS;
      }
      else {
          return CL_EVENT_TIMEOUT_AMD;
      }
  */
  return CL_SUCCESS;
}

void GpuDebugManager::destroyDebugEvent(DebugEvent* pEvent) {
  Unimplemented();
  /*
      osEventDestroy(*pEvent);
      *pEvent = 0;

      device()->gslCtx()->exceptionNotification(0);
  */
}

void GpuDebugManager::wavefrontControl(uint32_t waveAction, uint32_t waveMode, uint32_t trapId,
                                       void* waveAddr) const {
  Unimplemented();
  // device()->gslCtx()->executeSqCommand(waveAction, waveMode, trapId, waveAddr);
}

void GpuDebugManager::setAddressWatch(uint32_t numWatchPoints, void** watchAddress,
                                      uint64_t* watchMask, uint64_t* watchMode, DebugEvent* event) {
  size_t requiredSize = numWatchPoints * sizeof(HwDbgAddressWatch);

  //  previously allocated size is not big enough, allocate new memory
  if (addressWatchSize_ < requiredSize) {
    if (nullptr != addressWatch_) {  // free the smaller address watch storage
      delete[] addressWatch_;
    }
    addressWatch_ = new HwDbgAddressWatch[numWatchPoints];
    addressWatchSize_ = requiredSize;
  }

  //  fill in the address watch structure
  memset(addressWatch_, 0, addressWatchSize_);

  for (uint32_t i = 0; i < numWatchPoints; i++) {
    amd::Memory* watchMem = as_amd(reinterpret_cast<cl_mem>(watchAddress[i]));
    Memory* watchMemAddress = device()->getGpuMemory(watchMem);

    addressWatch_[i].watchAddress_ = reinterpret_cast<void*>(watchMemAddress->vmAddress());
    addressWatch_[i].watchMask_ = watchMask[i];
    addressWatch_[i].watchMode_ = (cl_dbg_address_watch_mode_amd)watchMode[i];
    addressWatch_[i].event_ = (0 != event) ? event[i] : 0;
  }

  Unimplemented();
  //  setup the watch addresses
  // device()->gslCtx()->setAddressWatch(numWatchPoints, (void*) addressWatch_);
}

void GpuDebugManager::setGlobalMemory(amd::Memory* memObj, uint32_t offset, void* srcPtr,
                                      uint32_t size) {
  Memory* globalMem = device()->getGpuMemory(memObj);

  address mappedMem = static_cast<address>(globalMem->map(nullptr, 0));
  assert(mappedMem != 0);

  void* dest_ptr = reinterpret_cast<void*>(mappedMem + offset);
  memcpy(dest_ptr, srcPtr, size);

  globalMem->unmap(nullptr);
}

int32_t GpuDebugManager::createRuntimeTrapHandler() {
  size_t codeSize = 0;
  const uint32_t* rtTrapCode = nullptr;

  if (device()->settings().viPlus_) {
    codeSize = sizeof(RuntimeTrapCodeVi);
    rtTrapCode = RuntimeTrapCodeVi;
  } else {
    codeSize = sizeof(RuntimeTrapCode);
    rtTrapCode = RuntimeTrapCode;
  }

  uint32_t numCodes = codeSize / sizeof(uint32_t);

  // Handle TMA corruption hw bug workaround -
  //   The trap handler buffer has extra 256 bytes allocated, the TMA address
  //   is stored in the first two DWORDs and the actual trap handler code
  //   is stored starting at the location of 256 bytes (TbaStartOffset).
  //
  // allocate memory for the runtime trap handler (TBA) + TMA address
  uint32_t allocSize = codeSize + TbaStartOffset;

  Memory* rtTBA = new Memory(*device(), allocSize);
  runtimeTBA_ = rtTBA;

  if ((rtTBA == nullptr) || !rtTBA->create(Resource::RemoteUSWC)) {
    return CL_OUT_OF_RESOURCES;
  }
  address tbaAddress = reinterpret_cast<address>(rtTBA->map(nullptr));

  // allocate buffer for the runtime trap handler buffer (TMA)
  uint32_t tmaSize = 0x100;
  Memory* rtTMA = new Memory(*device(), tmaSize);
  runtimeTMA_ = rtTMA;

  if ((rtTMA == nullptr) || !rtTMA->create(Resource::RemoteUSWC)) {
    return CL_OUT_OF_RESOURCES;
  }

  uint64_t rtTmaAddress = rtTMA->vmAddress();
  if ((rtTBA->vmAddress() & 0xFF) != 0 || (rtTmaAddress & 0xFF) != 0) {
    LogError("debugmanager: Trap handler/buffer is not 256-byte aligned");
    return CL_INVALID_VALUE;
  }

  // store the TMA address at the beginning of trap handler buffer
  uint64_t* tbaStorage = reinterpret_cast<uint64_t*>(tbaAddress);
  tbaStorage[0] = rtTmaAddress;

  // save the trap handler code
  uint32_t* trapHandlerPtr = (uint32_t*)(tbaAddress + TbaStartOffset);
  for (uint32_t i = 0; i < numCodes; i++) {
    trapHandlerPtr[i] = rtTrapCode[i];
  }

  rtTBA->unmap(nullptr);

  return CL_SUCCESS;
}

}  // namespace pal
