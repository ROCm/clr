/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include <hip/hip_runtime.h>

#include "hip_event.hpp"
#if !defined(_MSC_VER)
#include <unistd.h>
#endif

namespace hip {

bool Event::ready() {
  if (event_->status() != CL_COMPLETE) {
    event_->notifyCmdQueue();
  }
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  bool ready = g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_);
  if (!ready) {
    ready = (event_->status() == CL_COMPLETE);
  }
  return ready;
}

bool EventDD::ready() {
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  bool ready = g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_);
  // FIXME: Remove status check entirely
  if (!ready) {
    ready = (event_->status() == CL_COMPLETE);
  }
  return ready;
}

hipError_t Event::query() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  return ready() ? hipSuccess : hipErrorNotReady;
}

hipError_t Event::synchronize() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  static constexpr bool kWaitCompletion = true;
  if (!g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_, kWaitCompletion)) {
    event_->awaitCompletion();
  }

  return hipSuccess;
}

bool Event::awaitEventCompletion() {
  return event_->awaitCompletion();
}

bool EventDD::awaitEventCompletion() {
  return g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_, true);
}

hipError_t Event::elapsedTime(Event& eStop, float& ms) {
  amd::ScopedLock startLock(lock_);
  if (this == &eStop) {
    ms = 0.f;
    if (event_ == nullptr) {
      return hipErrorInvalidHandle;
    }

    if (flags & hipEventDisableTiming) {
      return hipErrorInvalidHandle;
    }

    if (!ready()) {
      return hipErrorNotReady;
    }

    return hipSuccess;
  }
  amd::ScopedLock stopLock(eStop.lock());

  if (event_ == nullptr || eStop.event() == nullptr) {
    return hipErrorInvalidHandle;
  }

  if ((flags | eStop.flags) & hipEventDisableTiming) {
    return hipErrorInvalidHandle;
  }

  if (!ready() || !eStop.ready()) {
    return hipErrorNotReady;
  }

  if (event_ == eStop.event_ && recorded_ && eStop.isRecorded()) {
    // Events are the same, which indicates the stream is empty and likely
    // eventRecord is called on another stream. For such cases insert and measure a
    // marker.
    amd::Command* command = new amd::Marker(*event_->command().queue(), kMarkerDisableFlush);
    command->enqueue();
    command->awaitCompletion();
    ms = static_cast<float>(static_cast<int64_t>(command->event().profilingInfo().end_) - time()) /
        1000000.f;
    command->release();
  } else {
    // Note: with direct dispatch eStop.ready() relies on HW event, but CPU status can be delayed.
    // Hence for now make sure CPU status is updated by calling awaitCompletion();
    awaitEventCompletion();
    eStop.awaitEventCompletion();
    ms = static_cast<float>(eStop.time() - time()) / 1000000.f;
  }
  return hipSuccess;
}

int64_t Event::time() const {
  assert(event_ != nullptr);
  if (recorded_) {
    return static_cast<int64_t>(event_->profilingInfo().end_);
  } else {
    return static_cast<int64_t>(event_->profilingInfo().start_);
  }
}

int64_t EventDD::time() const {
  uint64_t start = 0, end = 0;
  assert(event_ != nullptr);
  g_devices[deviceId()]->devices()[0]->getHwEventTime(*event_, &start, &end);
  // FIXME: This is only needed if the command had to wait CL_COMPLETE status
  if (start == 0 || end == 0) {
    return Event::time();
  }
  if (recorded_) {
    return static_cast<int64_t>(end);
  } else {
    return static_cast<int64_t>(start);
  }
}

hipError_t Event::streamWaitCommand(amd::Command*& command, amd::HostQueue* queue) {
  amd::Command::EventWaitList eventWaitList;
  if (event_ != nullptr) {
    eventWaitList.push_back(event_);
  }
  command = new amd::Marker(*queue, kMarkerDisableFlush, eventWaitList);

  if (command == NULL) {
    return hipErrorOutOfMemory;
  }
  return hipSuccess;
}

hipError_t Event::enqueueStreamWaitCommand(hipStream_t stream, amd::Command* command) {
  command->enqueue();
  return hipSuccess;
}

hipError_t Event::streamWait(hipStream_t stream, uint flags) {
  amd::HostQueue* queue = hip::getQueue(stream);
  // Access to event_ object must be lock protected
  amd::ScopedLock lock(lock_);
  if ((event_ == nullptr) || (event_->command().queue() == queue) || ready()) {
    return hipSuccess;
  }
  if (!event_->notifyCmdQueue()) {
    return hipErrorLaunchOutOfResources;
  }
  amd::Command* command;
  hipError_t status = streamWaitCommand(command, queue);
  if (status != hipSuccess) {
    return status;
  }
  status = enqueueStreamWaitCommand(stream, command);
  if (status != hipSuccess) {
    return status;
  }
  command->release();
  return hipSuccess;
}

hipError_t Event::recordCommand(amd::Command*& command, amd::HostQueue* queue) {
  if (command == nullptr) {
    static constexpr bool kRecordExplicitGpuTs = true;
    // Always submit a EventMarker.
    command = new hip::EventMarker(*queue, !kMarkerDisableFlush, kRecordExplicitGpuTs);
  }
  return hipSuccess;
}

hipError_t Event::enqueueRecordCommand(hipStream_t stream, amd::Command* command, bool record) {
  command->enqueue();
  if (event_ == &command->event()) return hipSuccess;
  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &command->event();
  recorded_ = record;

  return hipSuccess;
}

hipError_t Event::addMarker(hipStream_t stream, amd::Command* command, bool record) {
  amd::HostQueue* queue = hip::getQueue(stream);
  // Keep the lock always at the beginning of this to avoid a race. SWDEV-277847
  amd::ScopedLock lock(lock_);
  hipError_t status = recordCommand(command, queue);
  if (status != hipSuccess) {
    return hipSuccess;
  }
  status = enqueueRecordCommand(stream, command, record);
  return status;
}

}  // namespace hip
// ================================================================================================
hipError_t ihipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  if (event == nullptr) {
    return hipErrorInvalidValue;
  }
#if !defined(_MSC_VER)
  unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
      hipEventReleaseToDevice | hipEventReleaseToSystem | hipEventInterprocess;
#else
  unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
      hipEventReleaseToDevice | hipEventReleaseToSystem;
#endif
  const unsigned releaseFlags = (hipEventReleaseToDevice | hipEventReleaseToSystem);
  // can't set any unsupported flags.
  // can't set both release flags
  const bool illegalFlags = (flags & ~supportedFlags) || (flags & releaseFlags) == releaseFlags;
  if (!illegalFlags) {
    hip::Event* e = nullptr;
    if (flags & hipEventInterprocess) {
      e = new hip::IPCEvent();
    } else {
      if (AMD_DIRECT_DISPATCH) {
        e = new hip::EventDD(flags);
      } else {
        e = new hip::Event(flags);
      }
    }
    if (e == nullptr) {
      return hipErrorOutOfMemory;
    }
    *event = reinterpret_cast<hipEvent_t>(e);
  } else {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  HIP_INIT_API(hipEventCreateWithFlags, event, flags);
  HIP_RETURN(ihipEventCreateWithFlags(event, flags), *event);
}

hipError_t hipEventCreate(hipEvent_t* event) {
  HIP_INIT_API(hipEventCreate, event);
  HIP_RETURN(ihipEventCreateWithFlags(event, 0), *event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  HIP_INIT_API(hipEventDestroy, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  delete e;
  HIP_RETURN(hipSuccess);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  HIP_INIT_API(hipEventElapsedTime, ms, start, stop);

  if (ms == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (start == nullptr || stop == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* eStart = reinterpret_cast<hip::Event*>(start);
  hip::Event* eStop = reinterpret_cast<hip::Event*>(stop);

  if (eStart->deviceId() != eStop->deviceId()) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  HIP_RETURN(eStart->elapsedTime(*eStop, *ms), "Elapsed Time = ", *ms);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(hipEventRecord, event, stream);

  STREAM_CAPTURE(hipEventRecord, stream, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  amd::HostQueue* queue = hip::getQueue(stream);

  if (g_devices[e->deviceId()]->devices()[0] != &queue->device()) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  HIP_RETURN(e->addMarker(stream, nullptr, true));
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(hipEventSynchronize, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  HIP_RETURN(e->synchronize());
}

hipError_t ihipEventQuery(hipEvent_t event) {
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  return e->query();
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(hipEventQuery, event);
  HIP_RETURN(ihipEventQuery(event));
}
