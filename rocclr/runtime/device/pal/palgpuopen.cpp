/*
 **************************************************************************************************
 *
 *  Trade secret of Advanced Micro Devices, Inc.
 *  Copyright (c) 2016, Advanced Micro Devices, Inc., (unpublished)
 *
 *  All rights reserved. This notice is intended as a precaution against inadvertent publication
 *  and does not imply  publication or any waiver of confidentiality. The year included in
 *  the foregoing notice is the year of creation of the work.
 *
 **************************************************************************************************
 */
#include "device/pal/palgpuopen.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palvirtual.hpp"

// PAL headers
#include "palCmdAllocator.h"
#include "palFence.h"
#include "palQueueSemaphore.h"

// gpuutil headers
#include "gpuUtil/palGpaSession.h"

// gpuopen headers
#include "devDriverServer.h"
#include "msgChannel.h"
#include "msgTransport.h"
#include "protocols/rgpServer.h"
#include "protocols/driverControlServer.h"

namespace pal
{
// ================================================================================================
RgpCaptureMgr::RgpCaptureMgr(Pal::IPlatform* platform, const Device& device)
  :
  device_(device),
  dev_driver_server_(platform->GetDevDriverServer()),
  num_prep_disp_(0),
  trace_gpu_mem_limit_(0),
  global_disp_count_(1),      // Must start from 1 according to RGP spec
  trace_enabled_(false),
  inst_tracing_enabled_(false)
{
  memset(&trace_, 0, sizeof(trace_));
}

// ================================================================================================
RgpCaptureMgr::~RgpCaptureMgr()
{
  DestroyRGPTracing();
}

// ================================================================================================
// Creates the GPU Open Developer Mode manager class.
RgpCaptureMgr* RgpCaptureMgr::Create(Pal::IPlatform* platform, const Device& device)
{
  RgpCaptureMgr* mgr = new RgpCaptureMgr(platform, device);

  if (mgr != nullptr && !mgr->Init(platform)) {
    delete mgr;
    mgr = nullptr;
  }

  return mgr;
}

// ================================================================================================
bool RgpCaptureMgr::Init(Pal::IPlatform* platform)
{
  if (dev_driver_server_ == nullptr) {
    return false;
  }
  const Settings& settings = device_.settings();
  // Tell RGP that the server (i.e. the driver) supports tracing if requested.
  rgp_server_ = dev_driver_server_->GetRGPServer();
  if (rgp_server_ == nullptr) {
    return false;
  }

  // Finalize RGP settings
  Finalize();

  bool result = true;

  // Fail initialization of trace resources if SQTT tracing has been force-disabled from
  // the panel (this will consequently fail the trace), or if the chosen device's gfxip
  // does not support SQTT.
  //
  // It's necessary to check this during RGP tracing init in addition to devmode init because
  // during the earlier devmode init we may be in a situation where some enumerated physical
  // devices support tracing and others do not.
  if (GpuSupportsTracing(device_.properties(), settings) == false) {
    result = false;
  }

  // Create a GPA session object for this trace session
  if (result) {
    assert(trace_.gpa_session_ == nullptr);

    const uint32_t api_version = settings.oclVersion_;

    trace_.gpa_session_ = new GpuUtil::GpaSession(
        platform,
        device_.iDev(),
        api_version >> 4,   // OCL API version major
        api_version & 0xf,  // OCL API version minor
        RgpSqttInstrumentationSpecVersion,
        RgpSqttInstrumentationApiVersion);

    if (trace_.gpa_session_ == nullptr) {
      result = false;
    }
  }

  // Initialize the GPA session
  if (result &&  (trace_.gpa_session_->Init() != Pal::Result::Success)) {
    result = false;
  }

  // Initialize trace resources required by each queue (and queue family)
  bool hasDebugVmid = true;

  if (result) {
    //result = InitTraceQueueResources(trace_, &hasDebugVmid);
  }

  // If we've failed to acquire the debug VMID, fail to trace
  if (hasDebugVmid == false) {
    result = false;
  }

  if (!result) {
    // If we've failed to initialize tracing, permanently disable traces
    if (rgp_server_ != nullptr) {
        rgp_server_->DisableTraces();

        trace_enabled_ = false;
    }

    // Clean up if we failed
    DestroyRGPTracing();
  } else {
    PostDeviceCreate();
  }

  return result;
}

// ================================================================================================
// Called during initial device enumeration prior to calling Pal::IDevice::CommitSettingsAndInit().
//
// This finalizes the developer driver manager.
void RgpCaptureMgr::Finalize()
{
  // Figure out if the gfxip supports tracing.  We decide tracing if there is at least one
  // enumerated GPU that can support tracing.  Since we don't yet know if that GPU will be
  // picked as the target of an eventual VkDevice, this check is imperfect.
  // In mixed-GPU situations where an unsupported GPU is picked for tracing,
  // trace capture will fail with an error.
  bool hw_support_tracing = false;

  if ((rgp_server_->EnableTraces() == DevDriver::Result::Success)) {
   if (GpuSupportsTracing(device_.properties(), device_.settings())) {
     hw_support_tracing = true;
    }
  }

  if (hw_support_tracing == false) {
    rgp_server_->DisableTraces();
  }

  // Finalize the devmode manager
  dev_driver_server_->Finalize();

  // Figure out if tracing support should be enabled or not
  trace_enabled_ = (rgp_server_ != nullptr) && rgp_server_->TracesEnabled();
}


// ================================================================================================
// Waits for the driver to be resumed if it's currently paused.
void RgpCaptureMgr::WaitForDriverResume()
{
    auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();

    assert(pDriverControlServer != nullptr);

    pDriverControlServer->WaitForDriverResume();
}

// ================================================================================================
// Called before a swap chain presents.  This signals a frame-end boundary and
// is used to coordinate RGP trace start/stop.
void RgpCaptureMgr::PostDispatch(VirtualGPU* pQueue)
{
  if (rgp_server_->TracesEnabled()) {
    // If there's currently a trace running, submit the trace-end command buffer
    if (trace_.status_ == TraceStatus::Running) {
      amd::ScopedLock traceLock(&trace_mutex_);
      trace_.sqtt_disp_count_++;
      if (trace_.sqtt_disp_count_ >= device_.settings().rgpSqttDispCount_) {
        if (EndRGPHardwareTrace(pQueue) != Pal::Result::Success) {
        FinishRGPTrace(true);
        }
      }
    }

    if (IsQueueTimingActive()) {
      // Call TimedQueuePresent() to insert commands that collect GPU timestamp.
      Pal::IQueue* pPalQueue = pQueue->queue(MainEngine).iQueue_;

      // Currently nothing in the PresentInfo struct is used for inserting a timed present marker.
      GpuUtil::TimedQueuePresentInfo timedPresentInfo = {};
      //Pal::Result result = trace_.gpa_session_->TimedQueuePresent(pPalQueue, timedPresentInfo);
      //assert(result == Pal::Result::Success);
    }
  }
}

// ================================================================================================
Pal::Result RgpCaptureMgr::CheckForTraceResults()
{
  assert(trace_.status_ == TraceStatus::WaitingForResults);

  Pal::Result result = Pal::Result::NotReady;

  // Check if trace results are ready
  if (trace_.gpa_session_->IsReady() && // GPA session is ready
      (trace_.trace_begin_queue_->isDone(&trace_.end_event_)))   // "Trace end" cmdbuf has retired
  {
    bool success = false;

    // Fetch required trace data size from GPA session
    size_t traceDataSize = 0;
    void* pTraceData     = nullptr;

    trace_.gpa_session_->GetResults(trace_.gpa_sample_id_, &traceDataSize, nullptr);

    // Allocate memory for trace data
    if (traceDataSize > 0) {
        pTraceData = amd::AlignedMemory::allocate(traceDataSize, 256);
    }

    if (pTraceData != nullptr) {
      // Get trace data from GPA session
      if (trace_.gpa_session_->GetResults(trace_.gpa_sample_id_, &traceDataSize, pTraceData) ==
        Pal::Result::Success) {
        // Transmit trace data to anyone who's listening
        auto devResult = rgp_server_->WriteTraceData(
            static_cast<Pal::uint8*>(pTraceData), traceDataSize);

        success = (devResult == DevDriver::Result::Success);
      }

      amd::AlignedMemory::deallocate(pTraceData);
    }

    if (success) {
        result = Pal::Result::Success;
    }
  }

  return result;
}

// ================================================================================================
// Called after a swap chain presents.  This signals a (next) frame-begin boundary and is
// used to coordinate RGP trace start/stop.
void RgpCaptureMgr::PreDispatch(VirtualGPU* pQueue)
{
  // Wait for the driver to be resumed in case it's been paused.
  WaitForDriverResume();

  if (rgp_server_->TracesEnabled()) {
    amd::ScopedLock traceLock(&trace_mutex_);

    // Check if there's an RGP trace request pending and we're idle
    if ((trace_.status_ == TraceStatus::Idle) && rgp_server_->IsTracePending()) {
      // Attempt to start preparing for a trace
      if (PrepareRGPTrace(pQueue) == Pal::Result::Success) {
          // Attempt to start the trace immediately if we do not need to prepare
          if (num_prep_disp_ == 0) {
              if (BeginRGPTrace(pQueue) != Pal::Result::Success) {
                  FinishRGPTrace(true);
              }
          }
      }
    }
    else if (trace_.status_ == TraceStatus::Preparing) {
      // Wait some number of "preparation frames" before starting the trace in order to get enough
      // timer samples to sync CPU/GPU clock domains.
      trace_.prepared_disp_count_++;

      // Take a calibration timing measurement sample for this frame.
      trace_.gpa_session_->SampleTimingClocks();

      // Start the SQTT trace if we've waited a sufficient number of preparation frames
      if (trace_.prepared_disp_count_ >= num_prep_disp_) {
          Pal::Result result = BeginRGPTrace(pQueue);

          if (result != Pal::Result::Success) {
              FinishRGPTrace(true);
          }
      }
    }
    // Check if we're ending a trace waiting for SQTT to turn off.
    // If SQTT has turned off, end the trace
    else if (trace_.status_ == TraceStatus::WaitingForSqtt) {
      Pal::Result result      = Pal::Result::Success;

      if (trace_.trace_begin_queue_->isDone(&trace_.end_sqtt_event_)) {
        result = EndRGPTrace(pQueue);
      } else {
        // todo: There is a wait inside the trace end for now
        result = EndRGPTrace(pQueue);
      }

      if (result != Pal::Result::Success) {
          FinishRGPTrace(true);
      }
    }
    // Check if we're waiting for final trace results.
    else if (trace_.status_ == TraceStatus::WaitingForResults) {
      Pal::Result result = CheckForTraceResults();

      // Results ready: finish trace
      if (result == Pal::Result::Success) {
          FinishRGPTrace(false);
      }
      // Error while computing results: abort trace
      else if (result != Pal::Result::NotReady) {
          FinishRGPTrace(true);
      }
    }
  }

  global_disp_count_++;
}

// ================================================================================================
// This function starts preparing for an RGP trace.  Preparation involves some N frames of
// lead-up time during which timing samples are accumulated to synchronize CPU and GPU clock domains.
//
// This function transitions from the Idle state to the Preparing state.
Pal::Result RgpCaptureMgr::PrepareRGPTrace(VirtualGPU* pQueue)
{
  assert(trace_.status_ == TraceStatus::Idle);

  // We can only trace using a single device at a time currently, so recreate RGP trace
  // resources against this new one if the device is changing.
  Pal::Result result = Pal::Result::Success;

  const auto traceParameters = rgp_server_->QueryTraceParameters();

  num_prep_disp_        = traceParameters.numPreparationFrames;
  trace_gpu_mem_limit_  = traceParameters.gpuMemoryLimitInMb * 1024 * 1024;
  inst_tracing_enabled_ = traceParameters.flags.enableInstructionTokens;

  // Notify the RGP server that we are starting a trace
  if (rgp_server_->BeginTrace() != DevDriver::Result::Success) {
      result = Pal::Result::ErrorUnknown;
  }

  // Tell the GPA session class we're starting a trace
  if (result == Pal::Result::Success) {
    GpuUtil::GpaSessionBeginInfo info = {};

    info.flags.enableQueueTiming   = false;// trace_.queueTimingEnabled;

    result = trace_.gpa_session_->Begin(info);
  }

  trace_.prepared_disp_count_ = 0;
  trace_.sqtt_disp_count_     = 0;

  // Sample the timing clocks prior to starting a trace.
  if (result == Pal::Result::Success) {
    trace_.gpa_session_->SampleTimingClocks();
  }

  if (result == Pal::Result::Success) {
    // Remember which queue started the trace
    trace_.trace_prepare_queue_ = pQueue;
    trace_.trace_begin_queue_   = nullptr;

    trace_.status_ = TraceStatus::Preparing;
  } else {
    // We failed to prepare for the trace so abort it.
    if (rgp_server_ != nullptr) {
      const DevDriver::Result devDriverResult = rgp_server_->AbortTrace();

      // AbortTrace should always succeed unless we've used the api incorrectly.
      assert(devDriverResult == DevDriver::Result::Success);
    }
  }

  return result;
}

// ================================================================================================
// This function begins an RGP trace by initializing all dependent resources and submitting
// the "begin trace" information command buffer.
//
// This function transitions from the Preparing state to the Running state.
Pal::Result RgpCaptureMgr::BeginRGPTrace(VirtualGPU* pQueue)
{
  assert(trace_.status_ == TraceStatus::Preparing);
  assert(trace_enabled_);

  // We can only trace using a single device at a time currently, so recreate RGP trace
  // resources against this new one if the device is changing.
  Pal::Result result = Pal::Result::Success;

  if (result == Pal::Result::Success) {
    // Only allow trace to start if the queue family at prep-time matches the queue
    // family at begin time because the command buffer engine type must match
    if (trace_.trace_prepare_queue_ != pQueue) {
        result = Pal::Result::ErrorIncompatibleQueue;
    }
  }

  // Start a GPA tracing sample with SQTT enabled
  if (result == Pal::Result::Success) {
    GpuUtil::GpaSampleConfig sampleConfig = {};

    sampleConfig.type = GpuUtil::GpaSampleType::Trace;
    sampleConfig.sqtt.gpuMemoryLimit = trace_gpu_mem_limit_;
    sampleConfig.sqtt.flags.enable = true;
    sampleConfig.sqtt.flags.supressInstructionTokens = (inst_tracing_enabled_ == false);

    // Fill GPU commands
    pQueue->eventBegin(MainEngine);
    trace_.gpa_sample_id_ = trace_.gpa_session_->BeginSample(
        pQueue->queue(MainEngine).iCmd(), sampleConfig);
    pQueue->eventEnd(MainEngine, trace_.begin_sqtt_event_);
  }

  // Submit the trace-begin command buffer
  if (result == Pal::Result::Success) {
    static constexpr bool NeedFlush = true;
    // Update the global GPU event
    pQueue->setGpuEvent(trace_.begin_sqtt_event_, NeedFlush);
  }

  // Make the trace active and remember which queue started it
  if (result == Pal::Result::Success) {
    trace_.status_            = TraceStatus::Running;
    trace_.trace_begin_queue_ = pQueue;
  }

  return result;
}

// ================================================================================================
// This function submits the command buffer to stop SQTT tracing.  Full tracing still continues.
//
// This function transitions from the Running state to the WaitingForSqtt state.
Pal::Result RgpCaptureMgr::EndRGPHardwareTrace(VirtualGPU* pQueue)
{
  assert(trace_.status_ == TraceStatus::Running);

  Pal::Result result = Pal::Result::Success;

  // Only allow SQTT trace to start and end on the same queue because it's critical that these are
  // in the same order
  if (pQueue != trace_.trace_begin_queue_) {
    result = Pal::Result::ErrorIncompatibleQueue;
  }

  // Tell the GPA session to insert any necessary commands to end the tracing sample and
  // end the session itself
  if (result == Pal::Result::Success) {
    assert(trace_.gpa_session_ != nullptr);

    // Write CB commands to finish the SQTT
    pQueue->eventBegin(MainEngine);
    trace_.gpa_session_->EndSample(pQueue->queue(MainEngine).iCmd(), trace_.gpa_sample_id_);
    pQueue->eventEnd(MainEngine, trace_.end_sqtt_event_);

    static constexpr bool NeedFlush = true;
    // Update the global GPU event
    pQueue->setGpuEvent(trace_.end_sqtt_event_, NeedFlush);

    trace_.status_ = TraceStatus::WaitingForSqtt;

    // Execute a device wait idle
    if (device_.settings().rgpSqttWaitIdle_) {
      // Make sure the trace is done. Note: required for SDMA data write back
      pQueue->waitForEvent(&trace_.end_sqtt_event_);
    }
  }

  return result;
}

// ================================================================================================
// This function ends a running RGP trace.
//
// This function transitions from the WaitingForSqtt state to WaitingForResults state.
Pal::Result RgpCaptureMgr::EndRGPTrace(VirtualGPU* pQueue)
{
  assert(trace_.status_ == TraceStatus::WaitingForSqtt);

  Pal::Result result = Pal::Result::Success;

  // Tell the GPA session to insert any necessary commands to end the tracing sample and
  // end the session itself
  if (result == Pal::Result::Success) {
    assert(trace_.gpa_session_ != nullptr);
    // Initiate SDMA copy
    pQueue->eventBegin(SdmaEngine);
    result = trace_.gpa_session_->End(pQueue->queue(SdmaEngine).iCmd());
    pQueue->eventEnd(SdmaEngine, trace_.end_event_);
  }

  // Submit the trace-end command buffer
  if (result == Pal::Result::Success) {
    static constexpr bool NeedFlush = true;
    // Update the global GPU event
    pQueue->setGpuEvent(trace_.end_event_, NeedFlush);

    trace_.status_ = TraceStatus::WaitingForResults;

    if (device_.settings().rgpSqttWaitIdle_) {
      // Make sure the transfer is done
      pQueue->waitForEvent(&trace_.end_event_);
    }
  }

  return result;
}

// ================================================================================================
// This function resets and possibly cancels a currently active (between begin/end) RGP trace.
// It frees any dependent resources.
void RgpCaptureMgr::FinishRGPTrace(bool aborted)
{
  if (trace_.trace_prepare_queue_ == nullptr) {
    return;
  }

  // Inform RGP protocol that we're done with the trace, either by aborting it or finishing normally
  if (aborted) {
    rgp_server_->AbortTrace();
  } else {
    rgp_server_->EndTrace();
  }

  if (trace_.gpa_session_ != nullptr) {
    trace_.gpa_session_->Reset();
  }

  // Reset tracing state to idle
  trace_.prepared_disp_count_ = 0;
  trace_.sqtt_disp_count_     = 0;
  trace_.gpa_sample_id_       = 0;
  trace_.status_              = TraceStatus::Idle;
  trace_.trace_prepare_queue_ = nullptr;
  trace_.trace_begin_queue_   = nullptr;
}

// ================================================================================================
// Destroys device-persistent RGP resources
void RgpCaptureMgr::DestroyRGPTracing()
{
  if (trace_.status_ != TraceStatus::Idle) {
   FinishRGPTrace(true);
  }

  // Destroy the GPA session
  if (trace_.gpa_session_ != nullptr) {
    //Util::Destructor(trace_.gpa_session_);
    delete trace_.gpa_session_;
    trace_.gpa_session_ = nullptr;
  }

  memset(&trace_, 0, sizeof(trace_));
}

// ================================================================================================
// Returns true if the given device properties/settings support tracing.
bool RgpCaptureMgr::GpuSupportsTracing(
    const Pal::DeviceProperties& props,
    const Settings&       settings)
{
  return props.gfxipProperties.flags.supportRgpTraces && !settings.rgpSqttForceDisable_;
}

// ================================================================================================
// Called when a new device is created.  This will preallocate reusable RGP trace resources
// for that device.
void RgpCaptureMgr::PostDeviceCreate()
{
  amd::ScopedLock traceLock(&trace_mutex_);

  auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();

  assert(pDriverControlServer != nullptr);

  // If the driver hasn't been marked as fully initialized yet, mark it now.
  // We consider the time after the logical device creation to be the fully initialized driver
  // position. This is mainly because PAL is fully initialized at this point and we also know
  // whether or not the debug vmid has been acquired. External tools use this information to
  // decide when it's reasonable to make certain requests of the driver through protocol functions.
  if (pDriverControlServer->IsDriverInitialized() == false) {
    pDriverControlServer->FinishDriverInitialization();
  }
}

// ================================================================================================
// Called prior to a device's being destroyed.  This will free persistent RGP trace resources for
// that device.
void RgpCaptureMgr::PreDeviceDestroy()
{
  amd::ScopedLock traceLock(&trace_mutex_);
  // If we are idle, we can re-initialize trace resources based on the new device.
  if (trace_.status_ == TraceStatus::Idle) {
    DestroyRGPTracing();
  }
}

}; // namespace vk
