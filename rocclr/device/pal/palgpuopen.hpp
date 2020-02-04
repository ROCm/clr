/* Copyright (c) 2016-present Advanced Micro Devices, Inc.

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

namespace pal {
class Settings;
class Device;
class VirtualGPU;
class HSAILKernel;

// ================================================================================================
enum class RgpSqqtBarrierReason : uint32_t {
  Invalid = 0,
  MemDependency = 0xC0000000,
  ProfilingControl = 0xC0000001,
  SignalSubmit = 0xC0000002,
  PostDeviceEnqueue = 0xC0000003,
  Unknown = 0xffffffff
};

}

#ifdef PAL_GPUOPEN_OCL
// gpuopen headers
#include "gpuopen.h"

// PAL forward declarations
namespace Pal {
class ICmdBuffer;
class IFence;
class IQueueSemaphore;
struct PalPublicSettings;
}  // namespace Pal

// GpuUtil forward declarations
namespace GpuUtil {
class GpaSession;
};

// GPUOpen forward declarations
namespace DevDriver {
class DevDriverServer;
class IMsgChannel;
struct MessageBuffer;

namespace DriverControlProtocol {
enum struct DeviceClockMode : uint32_t;
class HandlerServer;
}  // namespace DriverControlProtocol

namespace SettingsProtocol {
class HandlerServer;
}

}  // namespace DevDriver

namespace pal {
// ================================================================================================
// RgpSqttMarkerIdentifier - Identifiers for RGP SQ thread-tracing markers (Table 1)
enum RgpSqttMarkerIdentifier : uint32_t {
  RgpSqttMarkerIdentifierEvent = 0x0,
  RgpSqttMarkerIdentifierCbStart = 0x1,
  RgpSqttMarkerIdentifierCbEnd = 0x2,
  RgpSqttMarkerIdentifierBarrierStart = 0x3,
  RgpSqttMarkerIdentifierBarrierEnd = 0x4,
  RgpSqttMarkerIdentifierUserEvent = 0x5,
  RgpSqttMarkerIdentifierGeneralApi = 0x6,
  RgpSqttMarkerIdentifierSync = 0x7,
  RgpSqttMarkerIdentifierPresent = 0x8,
  RgpSqttMarkerIdentifierLayoutTransition = 0x9,
  RgpSqttMarkerIdentifierRenderPass = 0xA,
  RgpSqttMarkerIdentifierReserved2 = 0xB,
  RgpSqttMarkerIdentifierReserved3 = 0xC,
  RgpSqttMarkerIdentifierReserved4 = 0xD,
  RgpSqttMarkerIdentifierReserved5 = 0xE,
  RgpSqttMarkerIdentifierReserved6 = 0xF
};

// ================================================================================================
enum class RgpSqttMarkerEventType : uint32_t {
  CmdNDRangeKernel = 0,
  CmdScheduler = 1,
  CmdCopyBuffer = 2,
  CmdCopyImageToBuffer = 3,
  CmdCopyBufferToImage = 4,
  CmdFillBuffer = 5,
  CmdCopyImage = 6,
  CmdFillImage = 7,
  CmdPipelineBarrier = 8,
  InternalUnknown = 26,
  Invalid = 0xffffffff
};

// ================================================================================================
// RgpSqttMarkerEvent - "Event (Per-draw/dispatch)" RGP SQ thread-tracing marker.
// These are generated ahead of draws or dispatches for commands that trigger generation of waves
//  i.e. draws/dispatches (Table 4).
struct RgpSqttMarkerEvent {
  union {
    struct {
      uint32_t identifier : 4;     // Identifier for this marker
      uint32_t extDwords : 3;      // Number of extra dwords following this marker
      uint32_t apiType : 24;       // The API type for this command
      uint32_t hasThreadDims : 1;  // Whether thread dimensions are included
    };

    uint32_t dword01;  // The first dword
  };

  union {
    // Some information about the vertex/instance/draw register indices.  These values are not
    // always valid because they are not available for one reason or another:
    //
    // - If vertex offset index or instance offset index are not (together) valid, they are both
    //  equal to 0
    // - If draw index is not valid, it is equal to the vertex offset index
    struct {
      uint32_t cbID : 20;               // Command buffer ID for this marker
      uint32_t vertexOffsetRegIdx : 4;  // SPI userdata register index for the first vertex offset
      uint32_t
          instanceOffsetRegIdx : 4;  // SPI userdata register index for the first instance offset
      uint32_t drawIndexRegIdx : 4;  // SPI userdata register index for the draw index (multi draw
                                     // indirect)
    };
    uint32_t dword02;  // The second dword
  };

  union {
    uint32_t cmdID;    // Command index within the command buffer
    uint32_t dword03;  // The third dword
  };
};

// ================================================================================================
// RgpSqttMarkerEventWithDims - Per-dispatch specific marker where workgroup dims are included
struct RgpSqttMarkerEventWithDims {
  RgpSqttMarkerEvent
      event;         // Per-draw/dispatch marker.  API type should be Dispatch, threadDim = 1
  uint32_t threadX;  // Work group count in X
  uint32_t threadY;  // Work group count in Y
  uint32_t threadZ;  // Work group count in Z
};

// ================================================================================================
// RgpSqttMarkerBarrierStart - "Barrier Start" RGP SQTT instrumentation marker (Table 5)
struct RgpSqttMarkerBarrierStart {
  union {
    struct {
      uint32_t identifier : 4;  // Identifier for this marker
      uint32_t extDwords : 3;   // Number of extra dwords following this marker
      uint32_t cbId : 20;       // Command buffer ID within queue
      uint32_t reserved : 5;    // Reserved
    };

    uint32_t dword01;  // The first dword
  };

  union {
    struct {
      uint32_t driverReason : 31;
      uint32_t internal : 1;
    };

    uint32_t dword02;  // The second dword
  };
};

// ================================================================================================
// RgpSqttMarkerBarrierEnd - "Barrier End" RGP SQTT instrumentation marker (Table 6)
struct RgpSqttMarkerBarrierEnd {
  union {
    struct {
      uint32_t identifier : 4;   // Identifier for this marker
      uint32_t extDwords : 3;    // Number of extra dwords following this marker
      uint32_t cbId : 20;        // Command buffer ID within queue
      uint32_t waitOnEopTs : 1;  // Issued EOP_TS VGT event followed by a WAIT_REG_MEM for that
                                 // timestamp to be written.  Quintessential full pipeline stall.
      uint32_t vsPartialFlush : 1;  // Stall at ME waiting for all prior VS waves to complete.
      uint32_t psPartialFlush : 1;  // Stall at ME waiting for all prior PS waves to complete.
      uint32_t csPartialFlush : 1;  // Stall at ME waiting for all prior CS waves to complete.
      uint32_t pfpSyncMe : 1;       // Stall PFP until ME is at same point in command stream.
    };

    uint32_t dword01;  // The first dword
  };

  union {
    struct {
      uint32_t
          syncCpDma : 1;  // Issue dummy CP-DMA command to confirm all prior CP-DMAs have completed.
      uint32_t invalTcp : 1;  // Invalidate the L1 vector caches.
      uint32_t invalSqI : 1;  // Invalidate the SQ instruction caches
      uint32_t invalSqK : 1;  // Invalidate the SQ constant caches (i.e. L1 scalar caches)
      uint32_t flushTcc : 1;  // Flush L2
      uint32_t invalTcc : 1;  // Invalidate L2
      uint32_t flushCb : 1;   // Flush CB caches (including DCC, cmask, fmask)
      uint32_t invalCb : 1;   // Invalidate CB caches (including DCC, cmask, fmask)
      uint32_t flushDb : 1;   // Flush DB caches (including htile)
      uint32_t invalDb : 1;   // Invalidate DB caches (including htile)
      uint32_t numLayoutTransitions : 16;  // Number of layout transitions following this packet
      uint32_t reserved : 6;               // Reserved for future expansion.  Always 0
    };

    uint32_t dword02;  // The second dword
  };
};

// RGP SQTT Instrumentation Specification version (API-independent)
constexpr uint32_t RgpSqttInstrumentationSpecVersion = 1;

// RGP SQTT Instrumentation Specification version for Vulkan-specific tables
constexpr uint32_t RgpSqttInstrumentationApiVersion = 0;

// RgpSqttMarkeUserEventDataType - Data types used in RGP SQ thread-tracing markers for an user
// event
enum RgpSqttMarkerUserEventType : uint32_t {
  RgpSqttMarkerUserEventTrigger = 0x0,
  RgpSqttMarkerUserEventPop = 0x1,
  RgpSqttMarkerUserEventPush = 0x2,
  RgpSqttMarkerUserEventObjectName = 0x3,
  RgpSqttMarkerUserEventReserved1 = 0x4,
  RgpSqttMarkerUserEventReserved2 = 0x5,
  RgpSqttMarkerUserEventReserved3 = 0x6,
  RgpSqttMarkerUserEventReserved4 = 0x7,
};

// RgpSqttMarkerUserEvent - RGP SQ thread-tracing marker for an user event.
union RgpSqttMarkerUserEvent {
  struct {
    uint32_t identifier : 4;  // Identifier for this marker
    uint32_t extDwords : 8;   // Number of extra dwords following this marker
    uint32_t dataType : 8;    // The type for this marker
    uint32_t reserved : 12;   // reserved
  };

  uint32_t dword01;  // The first dword
};

constexpr uint32_t RgpSqttMarkerUserEventWordCount = 1;

// The max lengths of frame marker strings
static constexpr size_t RgpSqttMaxUserEventStringLengthInDwords = 1024;

// RgpSqttMarkerUserEvent - RGP SQ thread-tracing marker for an user event with a string (push and
// trigger data types)
struct RgpSqttMarkerUserEventWithString {
  RgpSqttMarkerUserEvent header;

  uint32_t stringLength;  // Length of the string (in characters)
  uint32_t stringData[RgpSqttMaxUserEventStringLengthInDwords];  // String data in UTF-8 format
};

// ================================================================================================
// This class provides functionality to interact with the GPU Open Developer Mode message passing
// service and the rest of the driver.
class RgpCaptureMgr {
 public:
  ~RgpCaptureMgr();

  static RgpCaptureMgr* Create(Pal::IPlatform* platform, const Device& device);

  void Finalize();

  void PreDispatch(VirtualGPU* gpu, const HSAILKernel& kernel, size_t x, size_t y, size_t z);
  void PostDispatch(VirtualGPU* gpu);

  void WaitForDriverResume();

  void PostDeviceCreate();
  void PreDeviceDestroy();
  void FinishRGPTrace(VirtualGPU* gpu, bool aborted);

  bool IsQueueTimingActive() const;

  void WriteBarrierStartMarker(const VirtualGPU* gpu,
                               const Pal::Developer::BarrierData& data) const;
  void WriteBarrierEndMarker(const VirtualGPU* gpu, const Pal::Developer::BarrierData& data) const;
  bool RegisterTimedQueue(uint32_t queue_id, Pal::IQueue* iQueue, bool* debug_vmid) const;
  Pal::Result TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
                               const Pal::SubmitInfo& submitInfo) const;

 private:
  // Steps that an RGP trace goes through
  enum class TraceStatus {
    Idle = 0,   // No active trace and none requested
    Preparing,  // A trace has been requested but is not active yet because we are
                // currently sampling timing information over some number of lead frames.
    Running,    // SQTT and queue timing is currently active for all command buffer submits.
    WaitingForSqtt,
    WaitingForResults  // Tracing is no longer active, but all results are not yet ready.
  };

  // All per-device state to support RGP tracing
  struct TraceState {
    TraceStatus status_;  // Current trace status (idle, running, etc.)

    GpuEvent begin_sqtt_event_;  // Event that is signaled when a trace-end cmdbuf retires
    GpuEvent end_sqtt_event_;    // Event that is signaled when a trace-end cmdbuf retires
    GpuEvent end_event_;         // Event that is signaled when a trace-end cmdbuf retires

    VirtualGPU* prepare_queue_;  // The queue that triggered the full start of a trace
    VirtualGPU* begin_queue_;    // The queue that triggered starting SQTT

    GpuUtil::GpaSession* gpa_session_;  // GPA session helper object for building RGP data
    uint32_t gpa_sample_id_;            // Sample ID associated with the current trace
    bool queue_timing_;                 // Queue timing is enabled

    uint32_t prepared_disp_count_;  // Number of dispatches counted while preparing for a trace
    uint32_t sqtt_disp_count_;      // Number of dispatches counted while SQTT tracing is active
    mutable uint32_t current_event_id_;  // Current event ID
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
  RgpSqttMarkerEvent BuildEventMarker(const VirtualGPU* gpu, RgpSqttMarkerEventType api_type) const;
  void WriteMarker(const VirtualGPU* gpu, const void* data, size_t data_size) const;
  void WriteEventWithDimsMarker(const VirtualGPU* gpu, RgpSqttMarkerEventType apiType, uint32_t x,
                                uint32_t y, uint32_t z) const;
  void WriteUserEventMarker(const VirtualGPU* gpu, RgpSqttMarkerUserEventType eventType,
                            const std::string& name) const;

  const Device& device_;
  DevDriver::DevDriverServer* dev_driver_server_;
  DevDriver::RGPProtocol::RGPServer* rgp_server_;
  mutable amd::Monitor trace_mutex_;
  TraceState trace_;
  RgpSqttMarkerUserEventWithString* user_event_;

  uint32_t num_prep_disp_;
  uint32_t max_sqtt_disp_;  // Maximum number of the dispatches allowed in the trace
  uint32_t trace_gpu_mem_limit_;
  uint32_t global_disp_count_;

  bool trace_enabled_;         // True if tracing is currently enabled (master flag)
  bool inst_tracing_enabled_;  // Enable instruction-level SQTT tokens

  PAL_DISALLOW_DEFAULT_CTOR(RgpCaptureMgr);
  PAL_DISALLOW_COPY_AND_ASSIGN(RgpCaptureMgr);
};

// ================================================================================================
// Returns true if queue operations are currently being timed by RGP traces.
inline bool RgpCaptureMgr::IsQueueTimingActive() const {
  return (trace_.queue_timing_ &&
          (trace_.status_ == TraceStatus::Running || trace_.status_ == TraceStatus::Preparing ||
           trace_.status_ == TraceStatus::WaitingForSqtt));
}
}  // namespace pal
#else // PAL_GPUOPEN_OCL
namespace pal {
class RgpCaptureMgr {
 public:
  static RgpCaptureMgr* Create(Pal::IPlatform* platform, const Device& device) { return nullptr; }
  Pal::Result TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
                               const Pal::SubmitInfo& submitInfo) const {}
  void PreDispatch(VirtualGPU* gpu, const HSAILKernel& kernel, size_t x, size_t y, size_t z) {}
  void PostDispatch(VirtualGPU* gpu) {}
  void FinishRGPTrace(VirtualGPU* gpu, bool aborted) {}
  bool RegisterTimedQueue(uint32_t queue_id, Pal::IQueue* iQueue, bool* debug_vmid) const { return true; }
};
}  // namespace pal
#endif // PAL_GPUOPEN_OCL
