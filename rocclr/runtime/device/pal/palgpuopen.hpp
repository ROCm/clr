/*
***********************************************************************************************************************
*
*  Trade secret of Advanced Micro Devices, Inc.
*  Copyright (c) 2016 Advanced Micro Devices, Inc. (unpublished)
*
*  All rights reserved.  This notice is intended as a precaution against inadvertent publication and
*  does not imply publication or any waiver of confidentiality.  The year included in the foregoing
*  notice is the year of creation of the work.
*
***********************************************************************************************************************
*/
#pragma once

#include <queue>
#include "device/pal/paldefs.hpp"
#include "platform/commandqueue.hpp"
#include "protocols/rgpServer.h"
#include "device/blit.hpp"

// PAL headers
#include "palUtil.h"
#include "palPlatform.h"
#include "palCmdBuffer.h"
#include "palCmdAllocator.h"
#include "palQueue.h"
#include "palFence.h"
#include "palLinearAllocator.h"
#include "palHashMap.h"
#include "palQueue.h"
#include "palUtil.h"

// gpuopen headers
#include "gpuopen.h"

// PAL forward declarations
namespace Pal
{
class  ICmdBuffer;
class  IFence;
class  IQueueSemaphore;
struct PalPublicSettings;
}

// GpuUtil forward declarations
namespace GpuUtil
{                     
class GpaSession;
};

// GPUOpen forward declarations
namespace DevDriver
{
class DevDriverServer;
class IMsgChannel;
struct MessageBuffer;

namespace DriverControlProtocol
{
enum struct DeviceClockMode : uint32_t;
class HandlerServer;
}

namespace SettingsProtocol
{
class HandlerServer;
}

}

namespace pal
{
class Settings;
class Device;
class VirtualGPU;

// RGP SQTT Instrumentation Specification version (API-independent)
constexpr uint32_t RgpSqttInstrumentationSpecVersion = 1;

// RGP SQTT Instrumentation Specification version for Vulkan-specific tables
constexpr uint32_t RgpSqttInstrumentationApiVersion  = 0;

// =====================================================================================================================
// This class provides functionality to interact with the GPU Open Developer Mode message passing service and the rest
// of the driver.
class RgpCaptureMgr
{
public:
  ~RgpCaptureMgr();

  static RgpCaptureMgr* Create(Pal::IPlatform* platform, const Device& device);

  void Finalize();

  void PreDispatch(VirtualGPU* pQueue);
  void PostDispatch(VirtualGPU* pQueue);
  void WaitForDriverResume();

  void PostDeviceCreate();
  void PreDeviceDestroy();
  void FinishRGPTrace(bool aborted);

  bool IsQueueTimingActive() const;

private:
  // Steps that an RGP trace goes through
  enum class TraceStatus
  {
      Idle = 0,           // No active trace and none requested
      Preparing,          // A trace has been requested but is not active yet because we are
                          // currently sampling timing information over some number of lead frames.
      Running,            // SQTT and queue timing is currently active for all command buffer submits.
      WaitingForSqtt,
      WaitingForResults   // Tracing is no longer active, but all results are not yet ready.
  };

  // All per-device state to support RGP tracing
  struct TraceState
  {
    TraceStatus           status_;              // Current trace status (idle, running, etc.)

    GpuEvent              begin_sqtt_event_;    // Event that is signaled when a trace-end cmdbuf retires
    GpuEvent              end_sqtt_event_;      // Event that is signaled when a trace-end cmdbuf retires
    GpuEvent              end_event_;           // Event that is signaled when a trace-end cmdbuf retires

    VirtualGPU*           trace_prepare_queue_; // The queue that triggered the full start of a trace
    VirtualGPU*           trace_begin_queue_;   // The queue that triggered starting SQTT

    GpuUtil::GpaSession*  gpa_session_;         // GPA session helper object for building RGP data
    uint32_t              gpa_sample_id_;       // Sample ID associated with the current trace
    bool                  queue_timing_;        // Queue timing is enabled

    uint32_t              prepared_disp_count_; // Number of dispatches counted while preparing for a trace
    uint32_t              sqtt_disp_count_;     // Number of dispatches counted while SQTT tracing is active
  };

  RgpCaptureMgr(Pal::IPlatform* platform, const Device& device);

  bool Init(Pal::IPlatform* platform);
  Pal::Result PrepareRGPTrace(VirtualGPU* pQueue);
  Pal::Result BeginRGPTrace(VirtualGPU* pQueue);
  Pal::Result EndRGPHardwareTrace(VirtualGPU* pQueue);
  Pal::Result EndRGPTrace(VirtualGPU* pQueue);
  void DestroyRGPTracing();
  Pal::Result CheckForTraceResults();
  static bool GpuSupportsTracing(const Pal::DeviceProperties& props, const Settings& settings);

  const Device&               device_;
  DevDriver::DevDriverServer* dev_driver_server_;
  DevDriver::RGPProtocol::RGPServer* rgp_server_;
  amd::Monitor                trace_mutex_;
  TraceState                  trace_;
  uint32_t                    num_prep_disp_;
  uint32_t                    trace_gpu_mem_limit_;
  uint32_t                    global_disp_count_;
  bool                        trace_enabled_;         // True if tracing is currently enabled (master flag)
  bool                        inst_tracing_enabled_;  // Enable instruction-level SQTT tokens

  PAL_DISALLOW_DEFAULT_CTOR(RgpCaptureMgr);
  PAL_DISALLOW_COPY_AND_ASSIGN(RgpCaptureMgr);
};

// =====================================================================================================================
// Returns true if queue operations are currently being timed by RGP traces.
inline bool RgpCaptureMgr::IsQueueTimingActive() const
{
  return (trace_.queue_timing_ &&
          (trace_.status_ == TraceStatus::Running ||
           trace_.status_ == TraceStatus::Preparing ||
           trace_.status_ == TraceStatus::WaitingForSqtt));
}
};
