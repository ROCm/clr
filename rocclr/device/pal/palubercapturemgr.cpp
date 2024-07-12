/* Copyright (c) 2024 Advanced Micro Devices, Inc.

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

#include "device/pal/palubercapturemgr.hpp"
#include "device/pal/paldevice.hpp"

#include "palPlatform.h"
#include "palTraceSession.h"
#include "palRenderOpTraceController.h"
#include "palCodeObjectTraceSource.h"
#include "palQueueTimingsTraceSource.h"

#include "devDriverServer.h"
#include "protocols/driverControlServer.h"
#include "util/ddStructuredReader.h"

namespace amd::pal {

// ================================================================================================
// Returns true if the given device properties/settings support tracing.
static inline bool GpuSupportsTracing(const Pal::DeviceProperties& props,
                                      const Settings& settings) {
  return props.gfxipProperties.flags.supportRgpTraces && !settings.rgpSqttForceDisable_;
}

// ================================================================================================
// Creates the GPU Open Developer Mode manager class.
UberTraceCaptureMgr* UberTraceCaptureMgr::Create(Pal::IPlatform* platform, const Device& device) {
  UberTraceCaptureMgr* mgr = new UberTraceCaptureMgr(platform, device);

  if (mgr != nullptr && !mgr->Init(platform)) {
    delete mgr;
    mgr = nullptr;
  }

  return mgr;
}

// ================================================================================================
UberTraceCaptureMgr::UberTraceCaptureMgr(Pal::IPlatform* platform, const Device& device)
  : device_(device),
    dev_driver_server_(platform->GetDevDriverServer()),
    global_disp_count_(1), // Must start from 1 according to RGP spec
    trace_session_(platform->GetTraceSession()),
    trace_controller_(nullptr),
    code_object_trace_source_(nullptr),
    queue_timings_trace_source_(nullptr) {
}

// ================================================================================================
UberTraceCaptureMgr::~UberTraceCaptureMgr() {
  DestroyUberTraceResources();
}

// ================================================================================================
bool UberTraceCaptureMgr::CreateUberTraceResources(Pal::IPlatform* platform) {
  bool success = false;

  do {
    // Initialize the renderop trace controller
    trace_controller_ = new GpuUtil::RenderOpTraceController(platform, device_.iDev());
    if (trace_controller_ == nullptr) {
      break;
    }

    Pal::Result result = trace_session_->RegisterController(trace_controller_);
    if (result != Pal::Result::Success) {
      break;
    }

    // Initialize the code object trace source
    code_object_trace_source_ = new GpuUtil::CodeObjectTraceSource(platform);
    if (code_object_trace_source_ == nullptr) {
      break;
    }

    result = trace_session_->RegisterSource(code_object_trace_source_);
    if (result != Pal::Result::Success) {
      break;
    }

    // Initialize the queue timings trace source
    queue_timings_trace_source_ = new GpuUtil::QueueTimingsTraceSource(platform);
    if (queue_timings_trace_source_ == nullptr) {
      break;
    }

    result = trace_session_->RegisterSource(queue_timings_trace_source_);
    if (result != Pal::Result::Success) {
      break;
    }

    success = true;
  } while (false);

  return success;
}

// ================================================================================================
void UberTraceCaptureMgr::DestroyUberTraceResources() {
  // Deallocate and unregister all created trace controllers & trace sources

  // RenderOp TraceController
  if (trace_controller_ != nullptr) {
    trace_session_->UnregisterController(trace_controller_);
    delete trace_controller_;
    trace_controller_ = nullptr;
  }

  // CodeObjects TraceSource
  if (code_object_trace_source_ != nullptr) {
    trace_session_->UnregisterSource(code_object_trace_source_);
    delete code_object_trace_source_;
    code_object_trace_source_ = nullptr;
  }

  // QueueTimings TraceSource
  if (queue_timings_trace_source_ != nullptr) {
    trace_session_->UnregisterSource(queue_timings_trace_source_);
    delete queue_timings_trace_source_;
    queue_timings_trace_source_ = nullptr;
  }
}

// ================================================================================================
bool UberTraceCaptureMgr::Init(Pal::IPlatform* platform) {
  // Finalize the devmode manager
  if (dev_driver_server_ == nullptr) {
    return false;
  }
  dev_driver_server_->Finalize();

  // Initialize the trace sources & controllers owned by the compute driver
  const bool success = CreateUberTraceResources(platform);

  if (!success) {
    DestroyUberTraceResources();
    return false;
  }

  return true;
}

// ================================================================================================
void UberTraceCaptureMgr::PreDispatch(VirtualGPU* gpu, const HSAILKernel& kernel,
                                      size_t x, size_t y, size_t z) {
  // Wait for the driver to be resumed in case it's been paused.
  WaitForDriverResume();

  // Increment dispatch count in RenderOp trace controller
  Pal::IQueue* pQueue = gpu->queue(MainEngine).iQueue_;
  trace_controller_->RecordRenderOp(pQueue, GpuUtil::RenderOpTraceController::RenderOpDispatch);

  // Increment the global dispatch counter
  global_disp_count_++;
}

// ================================================================================================
void UberTraceCaptureMgr::PostDispatch(VirtualGPU* gpu) {
}

// ================================================================================================
// Waits for the driver to be resumed if it's currently paused.
void UberTraceCaptureMgr::WaitForDriverResume() {
  auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();

  assert(pDriverControlServer != nullptr);
  pDriverControlServer->DriverTick();
}

// ================================================================================================
void UberTraceCaptureMgr::PreDeviceDestroy() {
  if (trace_session_->GetTraceSessionState() == GpuUtil::TraceSessionState::Ready) {
    DestroyUberTraceResources();
  }
}

// ================================================================================================
void UberTraceCaptureMgr::FinishRGPTrace(VirtualGPU* gpu, bool aborted) {
  // Nothing to be done
}

// ================================================================================================
bool UberTraceCaptureMgr::IsQueueTimingActive() const {
  return ((queue_timings_trace_source_ != nullptr) &&
          (queue_timings_trace_source_->IsTimingInProgress()));
}

// ================================================================================================
void UberTraceCaptureMgr::WriteBarrierStartMarker(const VirtualGPU* gpu,
                                                  const Pal::Developer::BarrierData& data) const {
}

// ================================================================================================
void UberTraceCaptureMgr::WriteBarrierEndMarker(const VirtualGPU* gpu,
                                                const Pal::Developer::BarrierData& data) const {
}

// ================================================================================================
bool UberTraceCaptureMgr::RegisterTimedQueue(uint32_t queue_id,
                                             Pal::IQueue* iQueue,
                                             bool* debug_vmid) const {
  // Get the OS context handle for this queue (this is a thing that RGP needs on DX clients;
  // it may be optional for Vulkan, but we provide it anyway if available).
  Pal::KernelContextInfo kernelContextInfo = {};
  Pal::Result result = iQueue->QueryKernelContextInfo(&kernelContextInfo);

  // QueryKernelContextInfo may fail.
  // If so, just use a context identifier of 0.
  uint64_t queueContext = (result == Pal::Result::Success)
                        ? kernelContextInfo.contextIdentifier
                        : 0;

  // Register the queue with the GPA session class for timed queue operation support.
  result = queue_timings_trace_source_->RegisterTimedQueue(iQueue, queue_id, queueContext);

  return (result == Pal::Result::Success);
}

// ================================================================================================
Pal::Result UberTraceCaptureMgr::TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
                                                  const Pal::SubmitInfo& submitInfo) const {
  // Fill in extra meta-data information to associate the API command buffer data with
  // the generated timing information.
  GpuUtil::TimedSubmitInfo timedSubmitInfo = {};
  Pal::uint64 apiCmdBufIds = cmdId;
  Pal::uint32 sqttCmdBufIds = 0;

  timedSubmitInfo.pApiCmdBufIds = &apiCmdBufIds;
  timedSubmitInfo.pSqttCmdBufIds = &sqttCmdBufIds;
  timedSubmitInfo.frameIndex = 0;

  // Do a timed submit of all the command buffers
  Pal::Result result = queue_timings_trace_source_->TimedSubmit(queue,
                                                                submitInfo,
                                                                timedSubmitInfo);

  // Punt to non-timed submit if a timed submit fails (or is not supported)
  if (result != Pal::Result::Success) {
    result = queue->Submit(submitInfo);
  }

  return result;
}

// ================================================================================================
bool UberTraceCaptureMgr::Update(Pal::IPlatform* platform) {
  Pal::Result result = queue_timings_trace_source_->Init(device_.iDev());
  return (result == Pal::Result::Success);
}

// ================================================================================================
uint64_t UberTraceCaptureMgr::AddElfBinary(const void* exe_binary, size_t exe_binary_size,
                                           const void* elf_binary, size_t elf_binary_size,
                                           Pal::IGpuMemory* pGpuMemory, size_t offset) {
  GpuUtil::ElfBinaryInfo elfBinaryInfo = {};
  elfBinaryInfo.pBinary = exe_binary;
  elfBinaryInfo.binarySize = exe_binary_size; ///< FAT Elf binary size.
  elfBinaryInfo.pGpuMemory = pGpuMemory;      ///< GPU Memory where the compiled ISA resides.
  elfBinaryInfo.offset = static_cast<Pal::gpusize>(offset);

  elfBinaryInfo.originalHash = DevDriver::MetroHash::MetroHash64(
      reinterpret_cast<const DevDriver::uint8*>(elf_binary), elf_binary_size);

  elfBinaryInfo.compiledHash = DevDriver::MetroHash::MetroHash64(
      reinterpret_cast<const DevDriver::uint8*>(exe_binary), exe_binary_size);

  assert(code_object_trace_source_ != nullptr);

  code_object_trace_source_->RegisterElfBinary(elfBinaryInfo);

  return elfBinaryInfo.originalHash;
}

} // namespace amd::pal
