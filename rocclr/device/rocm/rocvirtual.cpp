/* Copyright (c) 2013 - 2022 Advanced Micro Devices, Inc.

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

#include "device/devhostcall.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/roccounters.hpp"
#include "platform/activity.hpp"
#include "platform/kernel.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/command_utils.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"
#include "hsa/amd_hsa_kernel_code.h"
#include "hsa/amd_hsa_queue.h"

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>


/**
* HSA image object size in bytes (see HSAIL spec)
*/
#define HSA_IMAGE_OBJECT_SIZE 48

/**
* HSA image object alignment in bytes (see HSAIL spec)
*/
#define HSA_IMAGE_OBJECT_ALIGNMENT 16

/**
* HSA sampler object size in bytes (see HSAIL spec)
*/
#define HSA_SAMPLER_OBJECT_SIZE 32

/**
* HSA sampler object alignment in bytes (see HSAIL spec)
*/
#define HSA_SAMPLER_OBJECT_ALIGNMENT 16

namespace roc {
// (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) invalidates I, K and L1
// (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE) invalidates L1, L2 and flushes
// L2

static constexpr uint16_t kInvalidAql =
    (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);

static constexpr uint16_t kBarrierPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierPacketAgentScopeHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kNopPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierPacketAcquireHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierPacketReleaseHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierVendorPacketHeader =
    (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr uint16_t kBarrierVendorPacketAgentScopeHeader =
    (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static constexpr hsa_barrier_and_packet_t kBarrierAcquirePacket = {
    kBarrierPacketAcquireHeader, 0, 0, {{0}}, 0, {0}};

static constexpr hsa_barrier_and_packet_t kBarrierReleasePacket = {
    kBarrierPacketReleaseHeader, 0, 0, {{0}}, 0, {0}};

double Timestamp::ticksToTime_ = 0;

static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
  return (v >> pos) & ((1 << width) - 1);
};

// ================================================================================================
void Timestamp::checkGpuTime() {
  amd::ScopedLock s(lock_);
  if (HwProfiling()) {
    uint64_t  start = std::numeric_limits<uint64_t>::max();
    uint64_t  end = 0;

    for (auto it : signals_) {
      amd::ScopedLock lock(it->LockSignalOps());

      // Ignore the wait if runtime processes API callback, because the signal value is bigger
      // than expected and the value reset will occur after API callback is done
      if (GetCallbackSignal().handle == 0) {
        WaitForSignal(it->signal_);
      }
      // Avoid profiling data for the sync barrier, in tiny performance tests the first call
      // to ROCr is very slow and that also affects the overall performance of the callback thread
      if (command().GetBatchHead() == nullptr || command().profilingInfo().marker_ts_) {
        hsa_amd_profiling_dispatch_time_t time = {};
        if (it->engine_ == HwQueueEngine::Compute) {
          hsa_amd_profiling_get_dispatch_time(gpu()->gpu_device(), it->signal_, &time);
        } else {
          hsa_amd_profiling_async_copy_time_t time_sdma = {};
          hsa_amd_profiling_get_async_copy_time(it->signal_, &time_sdma);
          time.start = time_sdma.start;
          time.end = time_sdma.end;
        }

        start = std::min(time.start, start);
        end = std::max(time.end, end);
        ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Signal = (0x%lx), start = %ld, "
          "end = %ld time taken= %ld ns", it->signal_.handle, start, end, end - start);
      }
      it->done_ = true;
    }
    signals_.clear();
    if (end != 0) {
      // Check if it's the first execution and update start time
      if (!accum_ena_) {
        start_ = start * ticksToTime_;
        accum_ena_ = true;
      }
      // Progress the end time always
      end_ = end * ticksToTime_;
    }
  }
}

// ================================================================================================
bool HsaAmdSignalHandler(hsa_signal_value_t value, void* arg) {
  Timestamp* ts = reinterpret_cast<Timestamp*>(arg);

  amd::Thread* thread = amd::Thread::current();
  if (!(thread != nullptr ||
      ((thread = new amd::HostThread()) != nullptr && thread == amd::Thread::current()))) {
    return false;
  }

  if (activity_prof::IsEnabled(OP_ID_DISPATCH)) {
    amd::Command* head = ts->getParsedCommand();
    if (head == nullptr) {
      head = ts->command().GetBatchHead();
    }
    while (head != nullptr) {
      if (head->data() != nullptr) {
        Timestamp* headTs  = reinterpret_cast<Timestamp*>(head->data());
        ts->setParsedCommand(head);
        for (auto it : headTs->Signals()) {
          hsa_signal_value_t complete_val = (headTs->GetCallbackSignal().handle != 0) ? 1 : 0;
          if (int64_t val = hsa_signal_load_relaxed(it->signal_) > complete_val) {
            hsa_status_t result = hsa_amd_signal_async_handler(headTs->Signals()[0]->signal_,
                                 HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                 &HsaAmdSignalHandler, ts);
            if (HSA_STATUS_SUCCESS != result) {
              LogError("hsa_amd_signal_async_handler() failed to requeue the handler!");
            } else {
              ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Requeue handler : value(%d), timestamp(%p),"
                      "handle(0x%lx)", static_cast<uint32_t>(val), headTs,
                      headTs->HwProfiling() ? headTs->Signals()[0]->signal_.handle : 0);
            }
            return false;
          }
        }
      }
      head = head->getNext();
    }
  }
  ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Handler: value(%d), timestamp(%p), handle(0x%lx)",
    static_cast<uint32_t>(value), arg, ts->HwProfiling() ? ts->Signals()[0]->signal_.handle : 0);

  // Save callback signal
  hsa_signal_t callback_signal = ts->GetCallbackSignal();

  // Update the batch, since signal is complete
  ts->gpu()->updateCommandsState(ts->command().GetBatchHead());

  // Reset API callback signal. It will release AQL queue and start commands processing
  if (callback_signal.handle != 0) {
    hsa_signal_subtract_relaxed(callback_signal, 1);
  }

  // Return false, so the callback will not be called again for this signal
  return false;
}

// ================================================================================================
bool VirtualGPU::MemoryDependency::create(size_t numMemObj) {
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

// ================================================================================================
void VirtualGPU::MemoryDependency::validate(VirtualGPU& gpu, const Memory* memory, bool readOnly) {
  bool flushL1Cache = false;

  if (maxMemObjectsInQueue_ == 0) {
    // Sync AQL packets
    gpu.setAqlHeader(gpu.dispatchPacketHeader_);
    return;
  }

  uint64_t curStart = reinterpret_cast<uint64_t>(memory->getDeviceMemory());
  uint64_t curEnd = curStart + memory->size();

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
    // Sync AQL packets
    gpu.setAqlHeader(gpu.dispatchPacketHeader_);

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

// ================================================================================================
void VirtualGPU::MemoryDependency::clear(bool all) {
  if (numMemObjectsInQueue_ > 0) {
    if (all) {
      endMemObjectsInQueue_ = numMemObjectsInQueue_;
    }

    // If the current launch didn't start from the beginning, then move the data
    if (0 != endMemObjectsInQueue_) {
      // Preserve all objects from the current kernel
      size_t i, j;
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

// ================================================================================================
VirtualGPU::HwQueueTracker::~HwQueueTracker() {
  for (auto& signal: signal_list_) {
    signal->release();
  }
}

// ================================================================================================
bool VirtualGPU::HwQueueTracker::Create() {
  uint kSignalListSize = ROC_SIGNAL_POOL_SIZE;
  signal_list_.resize(kSignalListSize);

  hsa_agent_t agent = gpu_.gpu_device();
  const Settings& settings = gpu_.dev().settings();
  hsa_agent_t* agents = (settings.system_scope_signal_) ? nullptr : &agent;
  uint32_t num_agents = (settings.system_scope_signal_) ? 0 : 1;

  for (uint i = 0; i < kSignalListSize; ++i) {
    std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
    if ((signal == nullptr) ||
        (HSA_STATUS_SUCCESS != hsa_signal_create(0, num_agents, agents, &signal->signal_))) {
      return false;
    }
    signal_list_[i] = signal.release();
  }
  return true;
}

// ================================================================================================
hsa_signal_t VirtualGPU::HwQueueTracker::ActiveSignal(
    hsa_signal_value_t init_val, Timestamp* ts) {
  bool new_signal = false;

  // Peep signal +2 ahead to see if its done
  auto temp_id = (current_id_ + 2) % signal_list_.size();
  // If GPU is still busy with processing, then add more signals to avoid more frequent stalls
  if (hsa_signal_load_relaxed(signal_list_[temp_id]->signal_) > 0) {
    std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
    if (signal != nullptr) {
      hsa_agent_t agent = gpu_.gpu_device();
      const Settings& settings = gpu_.dev().settings();
      hsa_agent_t* agents = (settings.system_scope_signal_) ? nullptr : &agent;
      uint32_t num_agents = (settings.system_scope_signal_) ? 0 : 1;

      if (HSA_STATUS_SUCCESS == hsa_signal_create(0, num_agents, agents, &signal->signal_)) {
        // Find valid new index
        ++current_id_ %= signal_list_.size();
        // Insert the new signal into the current slot and ignore any wait
        signal_list_.insert(signal_list_.begin() + current_id_, signal.release());
        new_signal = true;
      }
    }
  }

  // If it's the new signal, then the wait can be avoided.
  // That will allow to grow the list of signals without stalls
  if (!new_signal) {
    // Find valid index
    ++current_id_ %= signal_list_.size();

    // Make sure the previous operation on the current signal is done
    WaitCurrent();

    // Have to wait the next signal in the queue to avoid a race condition between
    // a GPU waiter(which may be not triggered yet) and CPU signal reset below
    WaitNext();
  }

  if (signal_list_[current_id_]->referenceCount() > 1) {
    // The signal was assigned to the global marker's event, hence runtime can't reuse it
    // and needs a new signal
    std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
    if (signal != nullptr) {
      hsa_agent_t agent = gpu_.gpu_device();
      const Settings& settings = gpu_.dev().settings();
      hsa_agent_t* agents = (settings.system_scope_signal_) ? nullptr : &agent;
      uint32_t num_agents = (settings.system_scope_signal_) ? 0 : 1;

      if (HSA_STATUS_SUCCESS == hsa_signal_create(0, num_agents, agents, &signal->signal_)) {
        signal_list_[current_id_]->release();
        signal_list_[current_id_] = signal.release();
      } else {
        assert(!"ProfilingSignal reallocation failed! Marker has a conflict with signal reuse!");
      }
    } else {
      assert(!"ProfilingSignal reallocation failed! Marker has a conflict with signal reuse!");
    }
  }
  ProfilingSignal* prof_signal = signal_list_[current_id_];
  // Reset the signal and return
  hsa_signal_silent_store_relaxed(prof_signal->signal_, init_val);
  prof_signal->done_ = false;
  prof_signal->engine_ = engine_;
  if (ts != 0) {
    // Save HSA signal earlier to make sure the possible callback will have a valid
    // value for processing
    ts->retain();
    prof_signal->ts_ = ts;
    ts->AddProfilingSignal(prof_signal);
    if (AMD_DIRECT_DISPATCH) {
      bool enqueHandler= false;
      uint32_t init_value = kInitSignalValueOne;
      enqueHandler = (ts->command().Callback() != nullptr ||
                      ts->command().GetBatchHead() != nullptr )  &&
                      !ts->command().CpuWaitRequested();
      // If direct dispatch is enabled and the batch head isn't null, then it's a marker and
      // requires the batch update upon HSA signal completion
      if (enqueHandler) {
        uint32_t init_value = kInitSignalValueOne;
        // If API callback is enabled, then use a blocking signal for AQL queue.
        // HSA signal will be acquired in SW and released after HSA signal callback
        if (ts->command().Callback() != nullptr) {
          ts->SetCallbackSignal(prof_signal->signal_);
          // Blocks AQL queue from further processing
          hsa_signal_add_relaxed(prof_signal->signal_, 1);
          init_value += 1;
        }
        hsa_status_t result = hsa_amd_signal_async_handler(prof_signal->signal_,
            HSA_SIGNAL_CONDITION_LT, init_value, &HsaAmdSignalHandler, ts);
        if (HSA_STATUS_SUCCESS != result) {
          LogError("hsa_amd_signal_async_handler() failed to set the handler!");
        } else {
          ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Set Handler: handle(0x%lx), timestamp(%p)",
            prof_signal->signal_.handle, prof_signal);
        }
        SetHandlerPending(false);
        // Update the current command/marker with HW event
        prof_signal->retain();
        ts->command().SetHwEvent(prof_signal);
      } else if (ts->command().profilingInfo().marker_ts_) {
        SetHandlerPending(true);
        // Update the current command/marker with HW event
        prof_signal->retain();
        ts->command().SetHwEvent(prof_signal);
      }
    }
    if (!sdma_profiling_) {
      hsa_amd_profiling_async_copy_enable(true);
      sdma_profiling_ = true;
    }
  }
  return prof_signal->signal_;
}

// ================================================================================================
std::vector<hsa_signal_t>& VirtualGPU::HwQueueTracker::WaitingSignal(HwQueueEngine engine) {
  bool explicit_wait = false;
  bool sdma_wait = false;
  // Reset all current waiting signals
  waiting_signals_.clear();

  if(engine != HwQueueEngine::Compute)
    sdma_wait = true;

  // Does runtime switch the active engine?
  if (engine != engine_) {
    // Yes, return the signal from the previous operation for a wait
    engine_ = engine;
    explicit_wait = true;
  } else {
    // Unknown engine in use, hence return a wait signal always
    if (engine == HwQueueEngine::Unknown) {
      explicit_wait = true;
    } else {
      // Check if skip wait optimization is enabled. It will try to predict the same engine in ROCr
      // and ignore the signal wait, relying on in-order engine execution
      const Settings& settings = gpu_.dev().settings();
      if (!settings.skip_copy_sync_ && (engine != HwQueueEngine::Compute)) {
        explicit_wait = true;
      }
    }
  }
  // Check if a wait is required
  if (explicit_wait) {
    bool skip_internal_signal = false;

    for (uint32_t i = 0; i < external_signals_.size(); ++i) {
      // If external signal matches internal one, then skip it
      if (external_signals_[i]->signal_.handle ==
          signal_list_[current_id_]->signal_.handle) {
        skip_internal_signal = true;
      }
    }
    // Add the oldest signal into the tracking for a wait
    if (!skip_internal_signal) {
      external_signals_.push_back(signal_list_[current_id_]);
    }

    // Validate all signals for the wait and skip already completed
    for (uint32_t i = 0; i < external_signals_.size(); ++i) {
      // Early signal status check
      if (hsa_signal_load_relaxed(external_signals_[i]->signal_) > 0) {
        const Settings& settings = gpu_.dev().settings();
        // Actively wait on CPU to avoid extra overheads of signal tracking on GPU
        if (!WaitForSignal<true>(external_signals_[i]->signal_, false, sdma_wait)) {
          if (settings.cpu_wait_for_signal_) {
            // Wait on CPU for completion if requested
            CpuWaitForSignal(external_signals_[i]);
          } else {
            // Add HSA signal for tracking on GPU
            waiting_signals_.push_back(external_signals_[i]->signal_);
          }
        }
      }
    }
    external_signals_.clear();
  }

  // Return the array of waiting HSA signals
  return waiting_signals_;
}

// ================================================================================================
bool VirtualGPU::HwQueueTracker::CpuWaitForSignal(ProfilingSignal* signal) {
  // Wait for the current signal
  if (signal->ts_ != nullptr) {
    // Update timestamp values if requested
    auto ts = signal->ts_;
    ts->checkGpuTime();
    ts->release();
    signal->ts_ = nullptr;
  } else if (hsa_signal_load_relaxed(signal->signal_) > 0) {
    amd::ScopedLock lock(signal->LockSignalOps());
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Host wait on completion_signal=0x%zx",
            signal->signal_.handle);
    if (!WaitForSignal(signal->signal_, gpu_.ActiveWait())) {
      LogPrintfError("Failed signal [0x%lx] wait", signal->signal_);
      return false;
    }
    signal->done_ = true;
  }
  return true;
}

// ================================================================================================
void VirtualGPU::HwQueueTracker::ResetCurrentSignal() {
  // Reset the signal and return
  hsa_signal_silent_store_relaxed(signal_list_[current_id_]->signal_, 0);
  // Fallback to the previous signal
  current_id_ = (current_id_ == 0) ? (signal_list_.size() - 1) : (current_id_ - 1);
}

// ================================================================================================
bool VirtualGPU::processMemObjects(const amd::Kernel& kernel, const_address params,
  size_t& ldsAddress, bool cooperativeGroups, bool& imageBufferWrtBack,
  std::vector<device::Memory*>& wrtBackImageBuffer) {
  Kernel& hsaKernel = const_cast<Kernel&>(static_cast<const Kernel&>(*(kernel.getDeviceKernel(dev()))));
  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  if (!cooperativeGroups && memoryDependency().maxMemObjectsInQueue() != 0) {
    // AQL packets
    setAqlHeader(dispatchPacketHeaderNoSync_);
  }

  amd::Memory* const* memories =
    reinterpret_cast<amd::Memory* const*>(params + kernelParams.memoryObjOffset());

  // HIP shouldn't use cache coherency layer at any time
  if (!amd::IS_HIP) {
    // Process cache coherency first, since the extra transfers may affect
    // other mem dependency tracking logic: TS and signalWrite()
    for (uint i = 0; i < signature.numMemories(); ++i) {
      amd::Memory* mem = memories[i];
      if (mem != nullptr) {
        roc::Memory* gpuMem = dev().getGpuMemory(mem);
        // Don't sync for internal objects, since they are not shared between devices
        if (gpuMem->owner()->getVirtualDevice() == nullptr) {
          // Synchronize data with other memory instances if necessary
          gpuMem->syncCacheFromHost(*this);
        }
      }
    }
  }

  // Mark the tracker with a new kernel, so it can avoid checks of the aliased objects
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

  amd::Memory* memory = nullptr;
  // get svm non arugment information
  void* const* svmPtrArray = reinterpret_cast<void* const*>(params + execInfoOffset);
  for (size_t i = 0; i < count; i++) {
    memory = amd::MemObjMap::FindMemObj(svmPtrArray[i]);
    if (nullptr == memory) {
      if (!supportFineGrainedSystem) {
        return false;
      } else if (sync) {
        // Sync AQL packets
        setAqlHeader(dispatchPacketHeader_);
        // Clear memory dependency state
        const static bool All = true;
        memoryDependency().clear(!All);
        continue;
      }
    } else {
      Memory* rocMemory = static_cast<Memory*>(memory->getDeviceMemory(dev()));
      if (nullptr != rocMemory) {
        // Synchronize data with other memory instances if necessary
        rocMemory->syncCacheFromHost(*this);

        const static bool IsReadOnly = false;
        // Validate SVM passed in the non argument list
        memoryDependency().validate(*this, rocMemory, IsReadOnly);
      } else {
        return false;
      }
    }
  }

  // Check all parameters for the current kernel
  for (size_t i = 0; i < signature.numParameters(); ++i) {
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    Memory* gpuMem = nullptr;
    amd::Memory* mem = nullptr;

    // Find if current argument is a buffer
    if (desc.type_ == T_POINTER) {
      if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_LOCAL) {
        // Align the LDS on the alignment requirement of type pointed to
        ldsAddress = amd::alignUp(ldsAddress, desc.info_.arrayIndex_);
        if (desc.size_ == 8) {
          // Save the original LDS size
          uint64_t ldsSize = *reinterpret_cast<const uint64_t*>(params + desc.offset_);
          // Patch the LDS address in the original arguments with an LDS address(offset)
          WriteAqlArgAt(const_cast<address>(params), ldsAddress, desc.size_, desc.offset_);
          // Add the original size
          ldsAddress += ldsSize;
        } else {
          // Save the original LDS size
          uint32_t ldsSize = *reinterpret_cast<const uint32_t*>(params + desc.offset_);
          // Patch the LDS address in the original arguments with an LDS address(offset)
          uint32_t ldsAddr = ldsAddress;
          WriteAqlArgAt(const_cast<address>(params), ldsAddr, desc.size_, desc.offset_);
          // Add the original size
          ldsAddress += ldsSize;
        }
      }
      else {
        uint32_t index = desc.info_.arrayIndex_;
        mem = memories[index];
        if (mem == nullptr) {
          //! This condition is for SVM fine-grain
          if (dev().isFineGrainedSystem(true)) {
            // Sync AQL packets
            setAqlHeader(dispatchPacketHeader_);
            // Clear memory dependency state
            const static bool All = true;
            memoryDependency().clear(!All);
          }
        }
        else {
          gpuMem = static_cast<Memory*>(mem->getDeviceMemory(dev()));

          const void* globalAddress = *reinterpret_cast<const void* const*>(params + desc.offset_);
          ClPrint(amd::LOG_INFO, amd::LOG_KERN,
            "Arg%d: %s %s = ptr:%p obj:[%p-%p]", i, desc.typeName_.c_str(),
             desc.name_.c_str(), globalAddress, gpuMem->getDeviceMemory(),
            reinterpret_cast<address>(gpuMem->getDeviceMemory()) + mem->getSize());

          // Validate memory for a dependency in the queue
          memoryDependency().validate(*this, gpuMem, (desc.info_.readOnly_ == 1));

          assert((desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_GLOBAL ||
                  desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_CONSTANT) &&
                 "Unsupported address qualifier");

          const bool readOnly =
#if defined(USE_COMGR_LIBRARY)
          desc.typeQualifier_ == CL_KERNEL_ARG_TYPE_CONST ||
#endif // defined(USE_COMGR_LIBRARY)
            (mem->getMemFlags() & CL_MEM_READ_ONLY) != 0;

          if (!readOnly) {
            mem->signalWrite(&dev());
          }

          if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
            Image* image = static_cast<Image*>(mem->getDeviceMemory(dev()));

            const uint64_t image_srd = image->getHsaImageObject().handle;
            assert(amd::isMultipleOf(image_srd, sizeof(image_srd)));
            WriteAqlArgAt(const_cast<address>(params), image_srd, sizeof(image_srd), desc.offset_);

              // Check if synchronization has to be performed
            if (image->CopyImageBuffer() != nullptr) {
              Memory* devBuf = dev().getGpuMemory(mem->parent());
              amd::Coord3D offs(0);
              Image* devCpImg = static_cast<Image*>(dev().getGpuMemory(image->CopyImageBuffer()));
              amd::Image* img = mem->asImage();

              // Copy memory from the original image buffer into the backing store image
              bool result = blitMgr().copyBufferToImage(
                  *devBuf, *devCpImg, offs, offs, img->getRegion(), true,
                  img->getRowPitch(), img->getSlicePitch());
              // Make sure the copy operation is done
              setAqlHeader(dispatchPacketHeader_);
              // Use backing store SRD as the replacment
              const uint64_t srd = devCpImg->getHsaImageObject().handle;
              WriteAqlArgAt(const_cast<address>(params), srd, sizeof(srd), desc.offset_);

              // If it's not a read only resource, then runtime has to write back
              if (!desc.info_.readOnly_) {
                wrtBackImageBuffer.push_back(mem->getDeviceMemory(dev()));
                imageBufferWrtBack = true;
              }
            }
          }
        }
      }
    }
    else if (desc.type_ == T_QUEUE) {
      uint32_t index = desc.info_.arrayIndex_;
      const amd::DeviceQueue* queue = reinterpret_cast<amd::DeviceQueue* const*>(
        params + kernelParams.queueObjOffset())[index];

      if (!createVirtualQueue(queue->size()) || !createSchedulerParam()) {
         return false;
      }
      uint64_t vqVA = getVQVirtualAddress();
      WriteAqlArgAt(const_cast<address>(params), vqVA, sizeof(vqVA), desc.offset_);
    }
    else if (desc.type_ == T_VOID) {
      const_address srcArgPtr = params + desc.offset_;
      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ReferenceObject) {
        void* mem = allocKernArg(desc.size_, 128);
        memcpy(mem, srcArgPtr, desc.size_);
        const auto it = hsaKernel.patch().find(desc.offset_);
        WriteAqlArgAt(const_cast<address>(params), mem, sizeof(void*), it->second);
      }
      ClPrint(amd::LOG_INFO, amd::LOG_KERN,
        "Arg%d: %s %s = val:%lld", i, desc.typeName_.c_str(), desc.name_.c_str(),
        *reinterpret_cast<const long long*>(srcArgPtr));
    }
    else if (desc.type_ == T_SAMPLER) {
      uint32_t index = desc.info_.arrayIndex_;
      const amd::Sampler* sampler = reinterpret_cast<amd::Sampler* const*>(params +
        kernelParams.samplerObjOffset())[index];

      device::Sampler* devSampler = sampler->getDeviceSampler(dev());

      uint64_t sampler_srd = devSampler->hwSrd();
      WriteAqlArgAt(const_cast<address>(params), sampler_srd, sizeof(sampler_srd), desc.offset_);
    }
  }

  if (hsaKernel.program()->hasGlobalStores()) {
    // Sync AQL packets
    setAqlHeader(dispatchPacketHeader_);
    // Clear memory dependency state
    const static bool All = true;
    memoryDependency().clear(!All);
  }

  return true;
}

// ================================================================================================
static inline void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
}

// ================================================================================================
template <typename AqlPacket>
bool VirtualGPU::dispatchGenericAqlPacket(
  AqlPacket* packet, uint16_t header, uint16_t rest, bool blocking, size_t size) {
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;
  const uint32_t sw_queue_size = queueMask;

  // Check for queue full and wait if needed.
  uint64_t index = hsa_queue_add_write_index_screlease(gpu_queue_, size);
  uint64_t read = hsa_queue_load_read_index_relaxed(gpu_queue_);

  auto expected_fence_state = extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);

  if (fence_state_ == amd::Device::kCacheStateSystem &&
      expected_fence_state == amd::Device::kCacheStateSystem) {
    header = dispatchPacketHeader_;
    fence_dirty_ = true;
  }

  fence_state_ = static_cast<Device::CacheState>(expected_fence_state);

  if (timestamp_ != nullptr) {
    // Get active signal for current dispatch if profiling is necessary
    packet->completion_signal = Barriers().ActiveSignal(kInitSignalValueOne, timestamp_);
  }

  // Make sure the slot is free for usage
  while ((index - hsa_queue_load_read_index_scacquire(gpu_queue_)) >= sw_queue_size) {
    amd::Os::yield();
  }

  // Add blocking command if the original value of read index was behind of the queue size.
  // Note: direct dispatch relies on the slot stall above to keep the forward progress
  // of the app if a dispatched kernel requires some CPU input for completion
  if (blocking || (!AMD_DIRECT_DISPATCH && (index - read) >= sw_queue_size)) {
    if (packet->completion_signal.handle == 0) {
      packet->completion_signal = Barriers().ActiveSignal();
    }
    blocking = true;
  }

  // Insert packet(s)
  // NOTE: need multiple packets to dispatch the performance counter
  //       packet blob of the legacy devices (gfx8)
  for (uint i = 0; i < size; i++, index++, packet++) {
    AqlPacket* aql_loc = &((AqlPacket*)(gpu_queue_->base_address))[index & queueMask];
    *aql_loc = *packet;
    if (header != 0) {
      packet_store_release(reinterpret_cast<uint32_t*>(aql_loc), header, rest);
    }
    ClPrint(amd::LOG_DEBUG, amd::LOG_AQL,
            "HWq=0x%zx, Dispatch Header = "
            "0x%x (type=%d, barrier=%d, acquire=%d, release=%d), "
            "setup=%d, grid=[%zu, %zu, %zu], workgroup=[%zu, %zu, %zu], private_seg_size=%zu, "
            "group_seg_size=%zu, kernel_obj=0x%zx, kernarg_address=0x%zx, completion_signal=0x%zx",
            gpu_queue_->base_address, header,
            extractAqlBits(header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
            extractAqlBits(header, HSA_PACKET_HEADER_BARRIER,
                           HSA_PACKET_HEADER_WIDTH_BARRIER),
            extractAqlBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                           HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
            extractAqlBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                           HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
            rest, reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->grid_size_x,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->grid_size_y,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->grid_size_z,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->workgroup_size_x,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->workgroup_size_y,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->workgroup_size_z,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->private_segment_size,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->group_segment_size,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->kernel_object,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->kernarg_address,
            reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet)->completion_signal);
  }

  hsa_signal_store_screlease(gpu_queue_->doorbell_signal, index - 1);

  // Wait on signal ?
  if (blocking) {
    LogInfo("Runtime reachead the AQL queue limit. SW is much ahead of HW. Blocking AQL queue!");
    if (!Barriers().WaitCurrent()) {
      LogPrintfError("Failed blocking queue wait with signal [0x%lx]",
                     packet->completion_signal.handle);
      return false;
    }
  }

  return true;
}

// ================================================================================================
void VirtualGPU::dispatchBlockingWait() {
  auto wait_signals = Barriers().WaitingSignal();
  // AQL dispatch doesn't support dependent signals and extra barrier packet must be generated
  for (uint32_t i = 0; i < wait_signals.size(); ++i) {
    uint32_t j = i % 5;
    barrier_packet_.dep_signal[j] = wait_signals[i];
    constexpr bool kSkipSignal = true;
    // If runtime reached the packet limit or the count limit, then flush the barrier
    if ((j == 4) || ((i + 1) == wait_signals.size())) {
      dispatchBarrierPacket(kNopPacketHeader, kSkipSignal);
    }
  }
}

// ================================================================================================
bool VirtualGPU::dispatchAqlPacket(
  hsa_kernel_dispatch_packet_t* packet, uint16_t header, uint16_t rest, bool blocking) {
  dispatchBlockingWait();

  return dispatchGenericAqlPacket(packet, header, rest, blocking);
}

// ================================================================================================
bool VirtualGPU::dispatchAqlPacket(
  hsa_barrier_and_packet_t* packet, uint16_t header, uint16_t rest, bool blocking) {
  return dispatchGenericAqlPacket(packet, header, rest, blocking);
}

// ================================================================================================
bool VirtualGPU::dispatchCounterAqlPacket(hsa_ext_amd_aql_pm4_packet_t* packet,
                                          const uint32_t gfxVersion, bool blocking,
                                          const hsa_ven_amd_aqlprofile_1_00_pfn_t* extApi) {


  // PM4 IB packet submission is different between GFX8 and GFX9:
  //  In GFX8 the PM4 IB packet blob is writing directly to AQL queue
  //  In GFX9 the PM4 IB is submitting by AQL Vendor Specific packet and
  switch (gfxVersion) {
    case PerfCounter::ROC_GFX8:
      { // Create legacy devices PM4 data
        hsa_ext_amd_aql_pm4_packet_t pm4Packet[SLOT_PM4_SIZE_AQLP];
        extApi->hsa_ven_amd_aqlprofile_legacy_get_pm4(packet, static_cast<void*>(&pm4Packet[0]));
        return dispatchGenericAqlPacket(&pm4Packet[0], 0, 0, blocking, SLOT_PM4_SIZE_AQLP);
      }
      break;
    case PerfCounter::ROC_GFX9:
    case PerfCounter::ROC_GFX10:
      {
        packet->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        return dispatchGenericAqlPacket(packet, 0, 0, blocking);
      }
      break;
  }

  return false;
}

// ================================================================================================
void VirtualGPU::dispatchBarrierPacket(uint16_t packetHeader, bool skipSignal,
                                       hsa_signal_t signal) {
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;

  if (!skipSignal) {
    // Make sure the wait is issued before queue index reservation
    auto wait_signals = Barriers().WaitingSignal();
    for (uint32_t i = 0; i < wait_signals.size(); ++i) {
      uint32_t j = i % 5;
      barrier_packet_.dep_signal[j] = wait_signals[i];
      constexpr bool kSkipSignal = true;
      // If runtime reached the packet limit and signals left, then flush the barrier
      if ((j == 4) && ((i + 1) < wait_signals.size())) {
        dispatchBarrierPacket(kNopPacketHeader, kSkipSignal);
      }
    }
  }

  uint64_t index = hsa_queue_add_write_index_screlease(gpu_queue_, 1);
  uint64_t read = hsa_queue_load_read_index_relaxed(gpu_queue_);

  auto cache_state = extractAqlBits(packetHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
  if (!skipSignal) {
    // Get active signal for current dispatch if profiling is necessary
    barrier_packet_.completion_signal =
      Barriers().ActiveSignal(kInitSignalValueOne, timestamp_);
  } else {
    // Attach external signal to the packet
    barrier_packet_.completion_signal = signal;
  }

  // Reset fence_dirty_ flag if we submit a barrier
  fence_dirty_ = false;

  while ((index - hsa_queue_load_read_index_scacquire(gpu_queue_)) >= queueMask);
  hsa_barrier_and_packet_t* aql_loc =
    &(reinterpret_cast<hsa_barrier_and_packet_t*>(gpu_queue_->base_address))[index & queueMask];
  *aql_loc = barrier_packet_;
  __atomic_store_n(reinterpret_cast<uint32_t*>(aql_loc), packetHeader, __ATOMIC_RELEASE);

  hsa_signal_store_screlease(gpu_queue_->doorbell_signal, index);
  ClPrint(amd::LOG_DEBUG, amd::LOG_AQL,
          "HWq=0x%zx, BarrierAND Header = 0x%x (type=%d, barrier=%d, acquire=%d,"
          " release=%d), "
          "dep_signal=[0x%zx, 0x%zx, 0x%zx, 0x%zx, 0x%zx], completion_signal=0x%zx",
          gpu_queue_->base_address, packetHeader,
          extractAqlBits(packetHeader, HSA_PACKET_HEADER_TYPE,
                         HSA_PACKET_HEADER_WIDTH_TYPE),
          extractAqlBits(packetHeader, HSA_PACKET_HEADER_BARRIER,
                         HSA_PACKET_HEADER_WIDTH_BARRIER),
          extractAqlBits(packetHeader, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
          extractAqlBits(packetHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
          barrier_packet_.dep_signal[0], barrier_packet_.dep_signal[1],
          barrier_packet_.dep_signal[2], barrier_packet_.dep_signal[3],
          barrier_packet_.dep_signal[4], barrier_packet_.completion_signal);

  // Clear dependent signals for the next packet
  barrier_packet_.dep_signal[0] = hsa_signal_t{};
  barrier_packet_.dep_signal[1] = hsa_signal_t{};
  barrier_packet_.dep_signal[2] = hsa_signal_t{};
  barrier_packet_.dep_signal[3] = hsa_signal_t{};
  barrier_packet_.dep_signal[4] = hsa_signal_t{};
}

// ================================================================================================
void VirtualGPU::dispatchBarrierValuePacket(uint16_t packetHeader, bool resolveDepSignal,
                                            hsa_signal_t signal, hsa_signal_value_t value,
                                            hsa_signal_value_t mask, hsa_signal_condition32_t cond,
                                            bool skipTs, hsa_signal_t completionSignal) {
  hsa_amd_barrier_value_packet_t barrier_value_packet_ = {0};
  uint16_t rest = HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;

  barrier_value_packet_.signal = signal;
  barrier_value_packet_.value = value;
  barrier_value_packet_.mask = mask;
  barrier_value_packet_.cond = cond;

  // Dependent signal and external signal cant be true at the same time
  assert(resolveDepSignal & (signal.handle != 0) == 0);
  if (resolveDepSignal) {
    auto wait_signal = Barriers().WaitingSignal();
    if (wait_signal.size() > 0) {
      assert(wait_signal.size() == 1 && "Only one dep signal allowed for BarrierValue");
      barrier_value_packet_.signal = wait_signal[0];
      barrier_value_packet_.value = kInitSignalValueOne;
      barrier_value_packet_.mask = std::numeric_limits<int64_t>::max();
      barrier_value_packet_.cond = HSA_SIGNAL_CONDITION_LT;
    }
  }

  if (completionSignal.handle == 0) {
    // Get active signal for current dispatch if profiling is necessary
    barrier_value_packet_.completion_signal =
      Barriers().ActiveSignal(kInitSignalValueOne, skipTs ? nullptr : timestamp_);
  } else {
    // Attach external signal to the packet
    barrier_value_packet_.completion_signal = completionSignal;
  }

  uint64_t index = hsa_queue_add_write_index_screlease(gpu_queue_, 1);
  while ((index - hsa_queue_load_read_index_scacquire(gpu_queue_)) >= queueMask);
  hsa_amd_barrier_value_packet_t* aql_loc = &(reinterpret_cast<hsa_amd_barrier_value_packet_t*>(
      gpu_queue_->base_address))[index & queueMask];
  *aql_loc = barrier_value_packet_;
  packet_store_release(reinterpret_cast<uint32_t*>(aql_loc), packetHeader, rest);

  auto cache_state = extractAqlBits(packetHeader, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);

  hsa_signal_store_screlease(gpu_queue_->doorbell_signal, index);

  ClPrint(amd::LOG_DEBUG, amd::LOG_AQL,
          "HWq=0x%zx, BarrierValue Header = 0x%x AmdFormat = 0x%x "
          "(type=%d, barrier=%d, acquire=%d, release=%d), "
          "signal=0x%zx, value = 0x%llx mask = 0x%llx cond: %s, completion_signal=0x%zx",
          gpu_queue_, packetHeader, rest,
          extractAqlBits(packetHeader, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE),
          extractAqlBits(packetHeader, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER),
          extractAqlBits(packetHeader, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                         HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
          cache_state,
          barrier_value_packet_.signal,
          barrier_value_packet_.value,
          barrier_value_packet_.mask,
          barrier_value_packet_.cond == 0 ? "EQ" : barrier_value_packet_.cond == 1 ?
                                        "NE" : barrier_value_packet_.cond == 2 ? "LT" : "GTE",
          barrier_value_packet_.completion_signal);
}

// ================================================================================================
void VirtualGPU::ResetQueueStates() {
  // Release all transfer buffers on this command queue
  releaseXferWrite();

  // Release all memory dependencies
  memoryDependency().clear();

  // Release the pool, since runtime just completed a barrier
  // @note: Runtime can reset kernel arg pool only if the barrier with L2 invalidation was issued
  resetKernArgPool();
}

// ================================================================================================
bool VirtualGPU::releaseGpuMemoryFence(bool skip_cpu_wait) {
  if (hasPendingDispatch_ || !Barriers().IsExternalSignalListEmpty()) {
    // Dispatch barrier packet into the queue
    dispatchBarrierPacket(kBarrierPacketHeader);
    hasPendingDispatch_ = false;
    retainExternalSignals_ = false;
  }

  // Check if runtime could skip CPU wait
  if (!skip_cpu_wait) {
    Barriers().WaitCurrent();

    ResetQueueStates();
  }
  return true;
}

// ================================================================================================
VirtualGPU::VirtualGPU(Device& device, bool profiling, bool cooperative,
                       const std::vector<uint32_t>& cuMask,
                       amd::CommandQueue::Priority priority)
    : device::VirtualDevice(device),
      state_(0),
      gpu_queue_(nullptr),
      roc_device_(device),
      virtualQueue_(nullptr),
      deviceQueueSize_(0),
      maskGroups_(0),
      schedulerThreads_(0),
      schedulerParam_(nullptr),
      schedulerQueue_(nullptr),
      schedulerSignal_({0}),
      barriers_(*this),
      kernarg_pool_signal_(KernelArgPoolNumSignal),
      cuMask_(cuMask),
      priority_(priority),
      copy_command_type_(0),
      fence_state_(Device::CacheState::kCacheStateInvalid),
      fence_dirty_(false)
{
  index_ = device.numOfVgpus_++;
  gpu_device_ = device.getBackendDevice();
  printfdbg_ = nullptr;

  // Initialize the last signal and dispatch flags
  timestamp_ = nullptr;
  hasPendingDispatch_ = false;
  profiling_ = profiling;
  cooperative_ = cooperative;

  kernarg_pool_base_ = nullptr;
  kernarg_pool_size_ = 0;
  kernarg_pool_cur_offset_ = 0;

  if (device.settings().fenceScopeAgent_) {
    dispatchPacketHeaderNoSync_ =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    dispatchPacketHeader_=
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  } else {
    dispatchPacketHeaderNoSync_ =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    dispatchPacketHeader_=
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  }

  aqlHeader_ = dispatchPacketHeader_;

  // Note: Virtual GPU device creation must be a thread safe operation
  roc_device_.vgpus_.resize(roc_device_.numOfVgpus_);
  roc_device_.vgpus_[index()] = this;

}

// ================================================================================================
VirtualGPU::~VirtualGPU() {
  delete blitMgr_;

  if (tracking_created_) {
    // Release the resources of signal
    releaseGpuMemoryFence();
  }

  destroyPool();

  releasePinnedMem();

  if (timestamp_ != nullptr) {
    timestamp_->release();
    timestamp_ = nullptr;
    LogError("There was a timestamp that was not used; deleting.");
  }

  delete printfdbg_;

  if (0 != schedulerSignal_.handle) {
    hsa_signal_destroy(schedulerSignal_);
  }

  if (nullptr != schedulerQueue_) {
    hsa_queue_destroy(schedulerQueue_);
  }

  if (nullptr != schedulerParam_) {
    schedulerParam_->release();
  }

  if (nullptr != virtualQueue_) {
    virtualQueue_->release();
  }

  // Lock the device to make the following thread safe
  amd::ScopedLock lock(roc_device_.vgpusAccess());

  --roc_device_.numOfVgpus_;  // Virtual gpu unique index decrementing
  roc_device_.vgpus_.erase(roc_device_.vgpus_.begin() + index());
  for (uint idx = index(); idx < roc_device_.vgpus().size(); ++idx) {
    roc_device_.vgpus()[idx]->index_--;
  }

  if (gpu_queue_) {
    roc_device_.releaseQueue(gpu_queue_, cuMask_);
  }
}

// ================================================================================================
bool VirtualGPU::create() {
  // Pick a reasonable queue size
  uint32_t queue_size = ROC_AQL_QUEUE_SIZE;
  gpu_queue_ = roc_device_.acquireQueue(queue_size, cooperative_, cuMask_, priority_);
  if (!gpu_queue_) return false;

  if (!initPool(dev().settings().kernargPoolSize_)) {
    LogError("Couldn't allocate arguments/signals for the queue");
    return false;
  }

  device::BlitManager::Setup blitSetup;
  blitMgr_ = new KernelBlitManager(*this, blitSetup);
  if ((nullptr == blitMgr_) || !blitMgr_->create(roc_device_)) {
    LogError("Could not create BlitManager!");
    return false;
  }

  // Initialize barrier and barrier value packets
  memset(&barrier_packet_, 0, sizeof(barrier_packet_));
  barrier_packet_.header = kInvalidAql;

  // Create a object of PrintfDbg
  printfdbg_ = new PrintfDbg(roc_device_);
  if (nullptr == printfdbg_) {
    LogError("\nCould not create printfDbg Object!");
    return false;
  }

  // Initialize timestamp conversion factor
  if (Timestamp::getGpuTicksToTime() == 0) {
    uint64_t frequency;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
    Timestamp::setGpuTicksToTime(1e9 / double(frequency));
  }

  if (!memoryDependency().create(GPU_NUM_MEM_DEPENDENCY)) {
    LogError("Could not create the array of memory objects!");
    return false;
  }

  // Allocate signal tracker for ROCr copy queue
  tracking_created_ = Barriers().Create();
  if (!tracking_created_) {
    LogError("Could not create signal for copy queue!");
    return false;
  }
  return true;
}

// ================================================================================================
bool VirtualGPU::initPool(size_t kernarg_pool_size) {
  kernarg_pool_size_ = kernarg_pool_size;
  kernarg_pool_chunk_end_ = kernarg_pool_size_ / KernelArgPoolNumSignal;
  active_chunk_ = 0;
  kernarg_pool_base_ = reinterpret_cast<address>(roc_device_.hostAlloc(kernarg_pool_size_, 0,
                                                 Device::MemorySegment::kKernArg));
  if (kernarg_pool_base_ == nullptr) {
    return false;
  }
  hsa_agent_t agent = gpu_device();
  for (auto& it : kernarg_pool_signal_) {
    if (HSA_STATUS_SUCCESS != hsa_signal_create(0, 1, &agent, &it)) {
      return false;
    }
  }
  return true;
}

// ================================================================================================
void VirtualGPU::destroyPool() {
  for (auto& it : kernarg_pool_signal_) {
    if (it.handle != 0) {
      hsa_signal_destroy(it);
    }
  }
  if (kernarg_pool_base_ != nullptr) {
    roc_device_.hostFree(kernarg_pool_base_, kernarg_pool_size_);
  }
}

// ================================================================================================
void* VirtualGPU::allocKernArg(size_t size, size_t alignment) {
  assert(alignment != 0);
  address result = nullptr;
  result = amd::alignUp(kernarg_pool_base_ + kernarg_pool_cur_offset_, alignment);
  const size_t pool_new_usage = (result + size) - kernarg_pool_base_;
  if (pool_new_usage <= kernarg_pool_chunk_end_) {
    kernarg_pool_cur_offset_ = pool_new_usage;
    return result;
  } else {
    //! We run out of the arguments space!
    //! That means the app didn't call clFlush/clFinish for very long time.
    // Reset the signal for the barrier packet
    hsa_signal_silent_store_relaxed(kernarg_pool_signal_[active_chunk_], kInitSignalValueOne);
    // Dispatch a barrier packet into the queue
    dispatchBarrierPacket(kBarrierPacketHeader, true, kernarg_pool_signal_[active_chunk_]);
    // Get the next chunk
    active_chunk_ = ++active_chunk_ % KernelArgPoolNumSignal;
    // Make sure the new active chunk is free
    bool test = WaitForSignal(kernarg_pool_signal_[active_chunk_], ActiveWait());
    assert(test && "Runtime can't fail a wait for chunk!");
    // Make sure the current offset matches the new chunk to avoid possible overlaps
    // between chunks and issues during recycle
    kernarg_pool_cur_offset_ = (active_chunk_ == 0) ? 0 : kernarg_pool_chunk_end_;
    kernarg_pool_chunk_end_ = kernarg_pool_cur_offset_ +
                              kernarg_pool_size_ / KernelArgPoolNumSignal;
    result = amd::alignUp(kernarg_pool_base_ + kernarg_pool_cur_offset_, alignment);
    kernarg_pool_cur_offset_ = (result + size) - kernarg_pool_base_;
  }

  return result;
}

// ================================================================================================
address VirtualGPU::allocKernelArguments(size_t size, size_t alignment) {
  if (ROC_SKIP_KERNEL_ARG_COPY) {
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());
    return reinterpret_cast<address>(allocKernArg(size, alignment));
  } else {
    return nullptr;
  }
}

// ================================================================================================
/* profilingBegin, when profiling is enabled, creates a timestamp to save in
* virtualgpu's timestamp_, and calls start() to get the current host
* timestamp.
*/
void VirtualGPU::profilingBegin(amd::Command& command, bool drmProfiling) {
  if (command.profilingInfo().enabled_) {
    if (timestamp_ != nullptr) {
      LogWarning("Trying to create a second timestamp in VirtualGPU. \
                  This could have unintended consequences.");
      return;
    }
    // Without barrier profiling will wait for each individual signal
    timestamp_ = new Timestamp(this, command);
    timestamp_->start();
  }

  if (AMD_DIRECT_DISPATCH) {
    if (!retainExternalSignals_) {
      Barriers().ClearExternalSignals();
    }
    for (auto it = command.eventWaitList().begin(); it < command.eventWaitList().end(); ++it) {
      void* hw_event = ((*it)->NotifyEvent() != nullptr) ?
        (*it)->NotifyEvent()->HwEvent() : (*it)->HwEvent();
      if (hw_event != nullptr) {
        Barriers().AddExternalSignal(reinterpret_cast<ProfilingSignal*>(hw_event));
      } else if (static_cast<amd::Command*>(*it)->queue() != command.queue() &&
                 ((*it)->status() != CL_COMPLETE)) {
        LogPrintfError("Waiting event(%p) doesn't have a HSA signal!\n", *it);
      } else {
        // Assume serialization on the same queue...
      }
    }
  }
}

// ================================================================================================
/* profilingEnd, when profiling is enabled, checks to see if a signal was
* created for whatever command we are running and calls end() to get the
* current host timestamp if no signal is available. It then saves the pointer
* timestamp_ to the command's data.
*/
void VirtualGPU::profilingEnd(amd::Command& command) {
  if (command.profilingInfo().enabled_) {
    if (!timestamp_->HwProfiling()) {
      timestamp_->end();
    }
    command.setData(timestamp_);
    timestamp_ = nullptr;
  }
  if (AMD_DIRECT_DISPATCH) {
    assert(retainExternalSignals_ || Barriers().IsExternalSignalListEmpty());
  }
}

// ================================================================================================
void VirtualGPU::updateCommandsState(amd::Command* list) const {
  Timestamp* ts = nullptr;

  amd::Command* current = list;
  amd::Command* next = nullptr;

  if (current == nullptr) {
    return;
  }

  uint64_t endTimeStamp = 0;
  uint64_t startTimeStamp = endTimeStamp;

  if (current->profilingInfo().enabled_) {
    // TODO: use GPU timestamp when available.
    endTimeStamp = amd::Os::timeNanos();
    startTimeStamp = endTimeStamp;

    // This block gets the first valid timestamp from the first command
    // that has one. This timestamp is used below to mark any command that
    // came before it to start and end with this first valid start time.
    current = list;
    while (current != nullptr) {
      if (current->data() != nullptr) {
        ts = reinterpret_cast<Timestamp*>(current->data());
        ts->getTime(&startTimeStamp, &endTimeStamp);
        break;
      }
      current = current->getNext();
    }
  }

  // Iterate through the list of commands, and set timestamps as appropriate
  // Note, if a command does not have a timestamp, it does one of two things:
  // - if the command (without a timestamp), A, precedes another command, C,
  // that _does_ contain a valid timestamp, command A will set RUNNING and
  // COMPLETE with the RUNNING (start) timestamp from command C. This would
  // also be true for command B, which is between A and C. These timestamps
  // are actually retrieved in the block above (startTimeStamp, endTimeStamp).
  // - if the command (without a timestamp), C, follows another command, A,
  // that has a valid timestamp, command C will be set RUNNING and COMPLETE
  // with the COMPLETE (end) timestamp of the previous command, A. This is
  // also true for any command B, which falls between A and C.
  current = list;
  while (current != nullptr) {
    if (current->profilingInfo().enabled_) {
      if (current->data() != nullptr) {
        // Since this is a valid command to get a timestamp, we use the
        // timestamp provided by the runtime (saved in the data())
        ts = reinterpret_cast<Timestamp*>(current->data());
        ts->getTime(&startTimeStamp, &endTimeStamp);
        ts->release();
        current->setData(nullptr);
      } else {
        // If we don't have a command that contains a valid timestamp,
        // we simply use the end timestamp of the previous command.
        // Note, if this is a command before the first valid timestamp,
        // this will be equal to the start timestamp of the first valid
        // timestamp at this point.
        startTimeStamp = endTimeStamp;
      }
    }

    if (current->status() == CL_SUBMITTED) {
      current->setStatus(CL_RUNNING, startTimeStamp);
      current->setStatus(CL_COMPLETE, endTimeStamp);
    } else if (current->status() != CL_COMPLETE) {
      LogPrintfError("Unexpected command status - %d.", current->status());
    }

    next = current->getNext();
    current->release();
    current = next;
  }
}

// ================================================================================================

void VirtualGPU::submitReadMemory(amd::ReadMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(cmd.destination(), &offset);

  Memory* devMem = dev().getRocMemory(&cmd.source());
  // Synchronize data with other memory instances if necessary
  devMem->syncCacheFromHost(*this);

  void* dst = cmd.destination();
  amd::Coord3D size = cmd.size();

  //! @todo: add multi-devices synchronization when supported.

  cl_command_type type = cmd.type();
  bool result = false;
  bool imageBuffer = false;

  // Force buffer read for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_READ_IMAGE) && (cmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    type = CL_COMMAND_READ_BUFFER;
    imageBuffer = true;
  }

  switch (type) {
    case CL_COMMAND_READ_BUFFER: {
      amd::Coord3D origin(cmd.origin()[0]);
      if (imageBuffer) {
        size_t elemSize = cmd.source().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if (hostMemory != nullptr) {
        // Accelerated transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result = blitMgr().copyBuffer(*devMem, *hostMemory, origin, dstOrigin, size,
                                      cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().readBuffer(*devMem, dst, origin, size, cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_READ_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(cmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, size.c, cmd.hostRect().rowPitch_,
                            cmd.hostRect().slicePitch_);
      if (hostMemory != nullptr) {
        result = blitMgr().copyBufferRect(*devMem, *hostMemory, cmd.bufRect(), hostbufferRect,
                                          size, cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().readBufferRect(*devMem, dst, cmd.bufRect(), cmd.hostRect(), size,
                                          cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_READ_IMAGE: {
      if ((cmd.source().parent() != nullptr) && (cmd.source().parent()->getType() == CL_MEM_OBJECT_BUFFER)) {
        Image* imageBuffer = static_cast<Image*>(devMem);
        // Check if synchronization has to be performed
        if (nullptr != imageBuffer->CopyImageBuffer()) {
          amd::Memory* memory = imageBuffer->CopyImageBuffer();
          devMem = dev().getGpuMemory(memory);
          Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
          amd::Image* image = imageBuffer->owner()->asImage();
          amd::Coord3D offs(0);
          // Copy memory from the original image buffer into the backing store image
          result = blitMgr().copyBufferToImage(*buffer, *devMem, offs,
                                               offs, image->getRegion(), true,
                                               image->getRowPitch(), image->getSlicePitch());
        }
      }
      if (hostMemory != nullptr) {
        // Accelerated image to buffer transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result =
            blitMgr().copyImageToBuffer(*devMem, *hostMemory, cmd.origin(), dstOrigin, size,
                                        cmd.isEntireMemory(), cmd.rowPitch(), cmd.slicePitch(), cmd.copyMetadata());
      } else {
        result = blitMgr().readImage(*devMem, dst, cmd.origin(), size, cmd.rowPitch(),
                                     cmd.slicePitch(), cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitReadMemory failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  }

  profilingEnd(cmd);
}

void VirtualGPU::submitWriteMemory(amd::WriteMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  size_t offset = 0;
  // Find if virtual address is a CL allocation
  device::Memory* hostMemory = dev().findMemoryFromVA(cmd.source(), &offset);

  Memory* devMem = dev().getRocMemory(&cmd.destination());

  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = cmd.isEntireMemory();
  devMem->syncCacheFromHost(*this, syncFlags);

  const char* src = static_cast<const char*>(cmd.source());
  amd::Coord3D size = cmd.size();

  //! @todo add multi-devices synchronization when supported.

  cl_command_type type = cmd.type();
  bool result = false;
  bool imageBuffer = false;

  // Force buffer write for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_WRITE_IMAGE) &&
      (cmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    type = CL_COMMAND_WRITE_BUFFER;
    imageBuffer = true;
  }

  switch (type) {
    case CL_COMMAND_WRITE_BUFFER: {
      amd::Coord3D origin(cmd.origin()[0]);
      if (imageBuffer) {
        size_t elemSize = cmd.destination().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }
      if (hostMemory != nullptr) {
        // Accelerated transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result = blitMgr().copyBuffer(*hostMemory, *devMem, srcOrigin, origin, size,
                                      cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().writeBuffer(src, *devMem, origin, size, cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_WRITE_BUFFER_RECT: {
      amd::BufferRect hostbufferRect;
      amd::Coord3D region(0);
      amd::Coord3D hostOrigin(cmd.hostRect().start_ + offset);
      hostbufferRect.create(hostOrigin.c, size.c, cmd.hostRect().rowPitch_,
                            cmd.hostRect().slicePitch_);
      if (hostMemory != nullptr) {
        result = blitMgr().copyBufferRect(*hostMemory, *devMem, hostbufferRect, cmd.bufRect(),
                                          size, cmd.isEntireMemory(), cmd.copyMetadata());
      } else {
        result = blitMgr().writeBufferRect(src, *devMem, cmd.hostRect(), cmd.bufRect(), size,
                                          cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    case CL_COMMAND_WRITE_IMAGE: {
      if (hostMemory != nullptr) {
        // Accelerated buffer to image transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result =
            blitMgr().copyBufferToImage(*hostMemory, *devMem, srcOrigin, cmd.origin(), size,
                                        cmd.isEntireMemory(), cmd.rowPitch(), cmd.slicePitch(),
                                        cmd.copyMetadata());
      } else {
        result = blitMgr().writeImage(src, *devMem, cmd.origin(), size, cmd.rowPitch(),
                                      cmd.slicePitch(), cmd.isEntireMemory(), cmd.copyMetadata());
      }
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitWriteMemory failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  } else {
    cmd.destination().signalWrite(&dev());
  }

  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  // in-order semantics: previous commands need to be done before we start
  releaseGpuMemoryFence();

  profilingBegin(cmd);
  const std::vector<void*>& svmPointers = cmd.svmPointers();
  if (cmd.pfnFreeFunc() == nullptr) {
    // pointers allocated using clSVMAlloc
    for (uint32_t i = 0; i < svmPointers.size(); i++) {
      amd::SvmBuffer::free(cmd.context(), svmPointers[i]);
    }
  } else {
    cmd.pfnFreeFunc()(as_cl(cmd.queue()->asCommandQueue()), svmPointers.size(),
                      (void**)(&(svmPointers[0])), cmd.userData());
  }
  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitSvmPrefetchAsync(amd::SvmPrefetchAsyncCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(cmd);

  if (dev().info().hmmSupported_) {
    // Initialize signal for the barrier
    auto wait_events = Barriers().WaitingSignal(HwQueueEngine::Unknown);
    hsa_signal_t active = Barriers().ActiveSignal(kInitSignalValueOne, timestamp_);

    // Find the requested agent for the transfer
    hsa_agent_t agent = (cmd.cpu_access() ||
        (dev().settings().hmmFlags_ & Settings::Hmm::EnableSystemMemory)) ?
        dev().getCpuAgent() : (static_cast<const roc::Device*>(cmd.device()))->getBackendDevice();

    // Initiate a prefetch command
    hsa_status_t status = hsa_amd_svm_prefetch_async(
        const_cast<void*>(cmd.dev_ptr()), cmd.count(), agent,
        wait_events.size(), wait_events.data(), active);

    // Wait for the prefetch. Should skip wait, but may require extra tracking for kernel execution
    if ((status != HSA_STATUS_SUCCESS) || !Barriers().WaitCurrent()) {
      Barriers().ResetCurrentSignal();
      LogError("hsa_amd_svm_prefetch_async failed");
      cmd.setStatus(CL_INVALID_OPERATION);
    }

    // Add system scope, since the prefetch scope is unclear
    addSystemScope();
  } else {
    LogWarning("hsa_amd_svm_prefetch_async is ignored, because no HMM support");
  }
  profilingEnd(cmd);
}

// ================================================================================================
bool VirtualGPU::copyMemory(cl_command_type type, amd::Memory& srcMem, amd::Memory& dstMem,
                            bool entire, const amd::Coord3D& srcOrigin,
                            const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                            const amd::BufferRect& srcRect, const amd::BufferRect& dstRect,
                            amd::CopyMetadata copyMetadata) {
  Memory* srcDevMem = dev().getRocMemory(&srcMem);
  Memory* dstDevMem = dev().getRocMemory(&dstMem);

  // Synchronize source and destination memory
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  dstDevMem->syncCacheFromHost(*this, syncFlags);
  srcDevMem->syncCacheFromHost(*this);

  bool result = false;
  bool srcImageBuffer = false;
  bool dstImageBuffer = false;

  // Force buffer copy for IMAGE1D_BUFFER
  if (srcMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    srcImageBuffer = true;
    type = CL_COMMAND_COPY_BUFFER;
  }
  if (dstMem.getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
    dstImageBuffer = true;
    type = CL_COMMAND_COPY_BUFFER;
  }

  switch (type) {
    case CL_COMMAND_SVM_MEMCPY:
    case CL_COMMAND_COPY_BUFFER: {
      amd::Coord3D realSrcOrigin(srcOrigin[0]);
      amd::Coord3D realDstOrigin(dstOrigin[0]);
      amd::Coord3D realSize(size.c[0], size.c[1], size.c[2]);

      if (srcImageBuffer) {
        const size_t elemSize = srcMem.asImage()->getImageFormat().getElementSize();
        realSrcOrigin.c[0] *= elemSize;
        if (dstImageBuffer) {
          realDstOrigin.c[0] *= elemSize;
        }
        realSize.c[0] *= elemSize;
      } else if (dstImageBuffer) {
        const size_t elemSize = dstMem.asImage()->getImageFormat().getElementSize();
        realDstOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
      }

      result = blitMgr().copyBuffer(*srcDevMem, *dstDevMem, realSrcOrigin, realDstOrigin,
                                    realSize, entire, copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_RECT: {
      result = blitMgr().copyBufferRect(*srcDevMem, *dstDevMem, srcRect, dstRect, size, entire,
                                        copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_IMAGE: {
      result = blitMgr().copyImage(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size, entire,
                                   copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
        result = blitMgr().copyImageToBuffer(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size, entire,
                                           dstRect.rowPitch_, dstRect.slicePitch_, copyMetadata);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
        result = blitMgr().copyBufferToImage(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size, entire,
                                           srcRect.rowPitch_, srcRect.slicePitch_, copyMetadata);
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitCopyMemory failed!");
    return false;
  }

  // Mark this as the most-recently written cache of the destination
  dstMem.signalWrite(&dev());
  return true;
}

// ================================================================================================
void VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  cl_command_type type = cmd.type();
  bool entire = cmd.isEntireMemory();

  if (!copyMemory(type, cmd.source(), cmd.destination(), entire, cmd.srcOrigin(),
                  cmd.dstOrigin(), cmd.size(), cmd.srcRect(), cmd.dstRect(), cmd.copyMetadata())) {
    cmd.setStatus(CL_INVALID_OPERATION);
  }

  // Runtime may change the command type to report a more accurate info in ROC profiler
  if (copy_command_type_ != 0) {
    cmd.OverrrideCommandType(copy_command_type_);
    copy_command_type_ = 0;
  }
  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);
  // no op for FGS supported device
  if (!dev().isFineGrainedSystem(true)) {
    amd::Coord3D srcOrigin(0, 0, 0);
    amd::Coord3D dstOrigin(0, 0, 0);
    amd::Coord3D size(cmd.srcSize(), 1, 1);
    amd::BufferRect srcRect;
    amd::BufferRect dstRect;

    bool result = false;
    amd::Memory* srcMem = amd::MemObjMap::FindMemObj(cmd.src());
    amd::Memory* dstMem = amd::MemObjMap::FindMemObj(cmd.dst());

    device::Memory::SyncFlags syncFlags;
    if (nullptr != srcMem) {
      srcOrigin.c[0] =
          static_cast<const_address>(cmd.src()) - static_cast<address>(srcMem->getSvmPtr());
      if (!(srcMem->validateRegion(srcOrigin, size))) {
        cmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }
    if (nullptr != dstMem) {
      dstOrigin.c[0] =
          static_cast<const_address>(cmd.dst()) - static_cast<address>(dstMem->getSvmPtr());
      if (!(dstMem->validateRegion(dstOrigin, size))) {
        cmd.setStatus(CL_INVALID_OPERATION);
        return;
      }
    }

    if ((nullptr == srcMem && nullptr == dstMem) || // both not in svm space
        (nullptr != srcMem && dev().forceFineGrain(srcMem)) ||
        (nullptr != dstMem && dev().forceFineGrain(dstMem))) {
      // Wait on a kernel if one is outstanding
      releaseGpuMemoryFence();

      // If these are from different contexts, then one of them could be in the device memory
      // This is fine, since spec doesn't allow for copies with pointers from different contexts
      amd::Os::fastMemcpy(cmd.dst(), cmd.src(), cmd.srcSize());
      result = true;
    } else if (nullptr == srcMem && nullptr != dstMem) {  // src not in svm space
      Memory* memory = dev().getRocMemory(dstMem);
      // Synchronize source and destination memory
      syncFlags.skipEntire_ = dstMem->isEntirelyCovered(dstOrigin, size);
      memory->syncCacheFromHost(*this, syncFlags);

      result = blitMgr().writeBuffer(cmd.src(), *memory, dstOrigin, size,
                                     dstMem->isEntirelyCovered(dstOrigin, size));
      // Mark this as the most-recently written cache of the destination
      dstMem->signalWrite(&dev());
    } else if (nullptr != srcMem && nullptr == dstMem) {  // dst not in svm space
      Memory* memory = dev().getRocMemory(srcMem);
      // Synchronize source and destination memory
      memory->syncCacheFromHost(*this);

      result = blitMgr().readBuffer(*memory, cmd.dst(), srcOrigin, size,
                                    srcMem->isEntirelyCovered(srcOrigin, size));
    } else if (nullptr != srcMem && nullptr != dstMem) {  // both in svm space
      bool entire =
          srcMem->isEntirelyCovered(srcOrigin, size) && dstMem->isEntirelyCovered(dstOrigin, size);
      result =
          copyMemory(cmd.type(), *srcMem, *dstMem, entire, srcOrigin, dstOrigin, size, srcRect, dstRect);
    }

    if (!result) {
      cmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    // Stall GPU for CPU access to memory
    releaseGpuMemoryFence();
    // direct memcpy for FGS enabled system
    amd::SvmBuffer::memFill(cmd.dst(), cmd.src(), cmd.srcSize(), 1);
  }
  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  Memory* srcDevMem = static_cast<roc::Memory*>(
    cmd.source().getDeviceMemory(*cmd.source().getContext().devices()[0]));
  Memory* dstDevMem = static_cast<roc::Memory*>(
    cmd.destination().getDeviceMemory(*cmd.destination().getContext().devices()[0]));

  bool p2pAllowed = false;
  // Loop through all available P2P devices for the destination buffer
  for (auto agent: dstDevMem->dev().p2pAgents()) {
    // Find the device, which is matching the current
    if (agent.handle == dev().getBackendDevice().handle) {
      p2pAllowed = true;
      break;
    }

    for (auto agent: srcDevMem->dev().p2pAgents()) {
      if (agent.handle == dev().getBackendDevice().handle) {
        p2pAllowed = true;
        break;
      }
    }
  }

  // Synchronize source and destination memory
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = cmd.isEntireMemory();
  amd::Coord3D size = cmd.size();

  bool result = false;
  switch (cmd.type()) {
    case CL_COMMAND_COPY_BUFFER: {
      amd::Coord3D srcOrigin(cmd.srcOrigin()[0]);
      amd::Coord3D dstOrigin(cmd.dstOrigin()[0]);

      if (p2pAllowed) {
          result = blitMgr().copyBuffer(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin,
                                        size, cmd.isEntireMemory());
      }
      else {
          // Sync the current queue, since P2P staging uses the device queues for transfer
          releaseGpuMemoryFence();

          amd::ScopedLock lock(dev().P2PStageOps());
          Memory* dstStgMem = static_cast<Memory*>(
              dev().P2PStage()->getDeviceMemory(*cmd.source().getContext().devices()[0]));
          Memory* srcStgMem = static_cast<Memory*>(
              dev().P2PStage()->getDeviceMemory(*cmd.destination().getContext().devices()[0]));

          size_t copy_size = Device::kP2PStagingSize;
          size_t left_size = size[0];
          result = true;
          do {
            if (left_size <= copy_size) {
              copy_size = left_size;
            }
            left_size -= copy_size;
            amd::Coord3D stageOffset(0);
            amd::Coord3D cpSize(copy_size);

            // Perform 2 step transfer with staging buffer
            result &= srcDevMem->dev().xferMgr().copyBuffer(
              *srcDevMem, *dstStgMem, srcOrigin, stageOffset, cpSize);
            srcOrigin.c[0] += copy_size;
            result &= dstDevMem->dev().xferMgr().copyBuffer(
              *srcStgMem, *dstDevMem, stageOffset, dstOrigin, cpSize);
            dstOrigin.c[0] += copy_size;
          } while (left_size > 0);
      }
      break;
    }
    case CL_COMMAND_COPY_BUFFER_RECT:
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      LogError("Unsupported P2P type!");
      break;
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitCopyMemoryP2P failed!");
    cmd.setStatus(CL_OUT_OF_RESOURCES);
  }

  cmd.destination().signalWrite(&dstDevMem->dev());

  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem(true) &&
      !dev().forceFineGrain(cmd.getSvmMem())) {
    // Make sure we have memory for the command execution
    Memory* memory = dev().getRocMemory(cmd.getSvmMem());

    memory->saveMapInfo(cmd.svmPtr(), cmd.origin(), cmd.size(), cmd.mapFlags(),
                        cmd.isEntireMemory());

    if (memory->mapMemory() != nullptr) {
      if (cmd.mapFlags() & (CL_MAP_READ | CL_MAP_WRITE)) {
        Memory* hsaMapMemory = dev().getRocMemory(memory->mapMemory());

        if (!blitMgr().copyBuffer(*memory, *hsaMapMemory, cmd.origin(), cmd.origin(),
                                  cmd.size(), cmd.isEntireMemory())) {
          LogError("submitSVMMapMemory() - copy failed");
          cmd.setStatus(CL_MAP_FAILURE);
        }
        // Wait on a kernel if one is outstanding
        releaseGpuMemoryFence();
        const void* mappedPtr = hsaMapMemory->owner()->getHostMem();
        amd::Os::fastMemcpy(cmd.svmPtr(), mappedPtr, cmd.size()[0]);
      }
    } else {
      LogError("Unhandled svm map!");
    }
  }

  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  // no op for FGS supported device
  if (!dev().isFineGrainedSystem(true) &&
      !dev().forceFineGrain(cmd.getSvmMem())) {
    Memory* memory = dev().getRocMemory(cmd.getSvmMem());
    const device::Memory::WriteMapInfo* writeMapInfo = memory->writeMapInfo(cmd.svmPtr());

    if (memory->mapMemory() != nullptr) {
      if (writeMapInfo->isUnmapWrite()) {
        // Wait on a kernel if one is outstanding
        releaseGpuMemoryFence();
        amd::Coord3D srcOrigin(0, 0, 0);
        Memory* hsaMapMemory = dev().getRocMemory(memory->mapMemory());

        void* mappedPtr = hsaMapMemory->owner()->getHostMem();
        amd::Os::fastMemcpy(mappedPtr, cmd.svmPtr(), writeMapInfo->region_[0]);
        // Target is a remote resource, so copy
        if (!blitMgr().copyBuffer(*hsaMapMemory, *memory, writeMapInfo->origin_,
                                  writeMapInfo->origin_, writeMapInfo->region_,
                                  writeMapInfo->isEntire())) {
          LogError("submitSvmUnmapMemory() - copy failed");
          cmd.setStatus(CL_OUT_OF_RESOURCES);
        }
      }
    } else {
      LogError("Unhandled svm map!");
    }

    memory->clearUnmapInfo(cmd.svmPtr());
  }

  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitMapMemory(amd::MapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  //! @todo add multi-devices synchronization when supported.

  roc::Memory* devMemory =
      reinterpret_cast<roc::Memory*>(cmd.memory().getDeviceMemory(dev(), false));

  cl_command_type type = cmd.type();
  bool imageBuffer = false;

  // Save map requirement.
  cl_map_flags mapFlag = cmd.mapFlags();

  // Treat no map flag as read-write.
  if (mapFlag == 0) {
    mapFlag = CL_MAP_READ | CL_MAP_WRITE;
  }

  devMemory->saveMapInfo(cmd.mapPtr(), cmd.origin(), cmd.size(), mapFlag, cmd.isEntireMemory());

  // Sync to the map target.
  // If we have host memory, use it
  if ((devMemory->owner()->getHostMem() != nullptr) &&
      (devMemory->owner()->getSvmPtr() == nullptr)) {
    if (!devMemory->isHostMemDirectAccess()) {
      // Make sure GPU finished operation before synchronization with the backing store
      releaseGpuMemoryFence();
    }
    // Target is the backing store, so just ensure that owner is up-to-date
    devMemory->owner()->cacheWriteBack(this);

    if (devMemory->isHostMemDirectAccess()) {
      // Add memory to VA cache, so rutnime can detect direct access to VA
      dev().addVACache(devMemory);
    }
  } else if (devMemory->IsPersistentDirectMap()) {
    // Persistent memory - NOP map
  } else if (mapFlag & (CL_MAP_READ | CL_MAP_WRITE)) {
    bool result = false;
    roc::Memory* hsaMemory = static_cast<roc::Memory*>(devMemory);

    amd::Memory* mapMemory = hsaMemory->mapMemory();
    void* hostPtr =
        mapMemory == nullptr ? hsaMemory->owner()->getHostMem() : mapMemory->getHostMem();

    if (type == CL_COMMAND_MAP_BUFFER) {
      amd::Coord3D origin(cmd.origin()[0]);
      amd::Coord3D size(cmd.size()[0]);
      amd::Coord3D dstOrigin(cmd.origin()[0], 0, 0);
      if (imageBuffer) {
        size_t elemSize = cmd.memory().asImage()->getImageFormat().getElementSize();
        origin.c[0] *= elemSize;
        size.c[0] *= elemSize;
      }

      if (mapMemory != nullptr) {
        roc::Memory* hsaMapMemory =
            static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));
        result = blitMgr().copyBuffer(*hsaMemory, *hsaMapMemory, origin, dstOrigin, size,
                                      cmd.isEntireMemory());
        void* svmPtr = devMemory->owner()->getSvmPtr();
        if ((svmPtr != nullptr) && (hostPtr != svmPtr)) {
          // Wait on a kernel if one is outstanding
          releaseGpuMemoryFence();
          amd::Os::fastMemcpy(svmPtr, hostPtr, size[0]);
        }
      } else {
        result = blitMgr().readBuffer(*hsaMemory, static_cast<char*>(hostPtr) + origin[0], origin,
                                      size, cmd.isEntireMemory());
      }
    } else if (type == CL_COMMAND_MAP_IMAGE) {
      amd::Image* image = cmd.memory().asImage();
      if (mapMemory != nullptr) {
        roc::Memory* hsaMapMemory =
            static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));
        result =
            blitMgr().copyImageToBuffer(*hsaMemory, *hsaMapMemory, cmd.origin(), amd::Coord3D(0, 0, 0),
                                        cmd.size(), cmd.isEntireMemory());
      } else {
        result = blitMgr().readImage(*hsaMemory, hostPtr, amd::Coord3D(0), image->getRegion(),
                                     image->getRowPitch(), image->getSlicePitch(), true);
      }
    } else {
      ShouldNotReachHere();
    }

    if (!result) {
      LogError("submitMapMemory failed!");
      cmd.setStatus(CL_OUT_OF_RESOURCES);
    }
  }

  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitUnmapMemory(amd::UnmapMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  roc::Memory* devMemory = static_cast<roc::Memory*>(cmd.memory().getDeviceMemory(dev(), false));

  const device::Memory::WriteMapInfo* mapInfo = devMemory->writeMapInfo(cmd.mapPtr());
  if (nullptr == mapInfo) {
    LogError("Unmap without map call");
    return;
  }

  profilingBegin(cmd);

  // Force buffer write for IMAGE1D_BUFFER
  bool imageBuffer = (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER);

  // We used host memory
  if ((devMemory->owner()->getHostMem() != nullptr) &&
      (devMemory->owner()->getSvmPtr() == nullptr)) {
    if (mapInfo->isUnmapWrite()) {
      // Target is the backing store, so sync
      devMemory->owner()->signalWrite(nullptr);
      devMemory->syncCacheFromHost(*this);
    }
    if (devMemory->isHostMemDirectAccess()) {
      // Remove memory from VA cache
      dev().removeVACache(devMemory);
    }
  } else if (devMemory->IsPersistentDirectMap()) {
    // Persistent memory - NOP unmap
  } else if (mapInfo->isUnmapWrite()) {
    // Commit the changes made by the user.
    if (!devMemory->isHostMemDirectAccess()) {
      bool result = false;

      amd::Memory* mapMemory = devMemory->mapMemory();
      if (cmd.memory().asImage() && !imageBuffer) {
        amd::Image* image = cmd.memory().asImage();
        if (mapMemory != nullptr) {
          roc::Memory* hsaMapMemory =
              static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));
          result =
              blitMgr().copyBufferToImage(*hsaMapMemory, *devMemory, amd::Coord3D(0, 0, 0),
                                          mapInfo->origin_, mapInfo->region_, mapInfo->isEntire());
        } else {
          void* hostPtr = devMemory->owner()->getHostMem();

          result = blitMgr().writeImage(hostPtr, *devMemory, amd::Coord3D(0), image->getRegion(),
                                        image->getRowPitch(), image->getSlicePitch(), true);
        }
      } else {
        amd::Coord3D origin(mapInfo->origin_[0]);
        amd::Coord3D size(mapInfo->region_[0]);
        if (imageBuffer) {
          size_t elemSize = cmd.memory().asImage()->getImageFormat().getElementSize();
          origin.c[0] *= elemSize;
          size.c[0] *= elemSize;
        }
        if (mapMemory != nullptr) {
          roc::Memory* hsaMapMemory =
              static_cast<roc::Memory*>(mapMemory->getDeviceMemory(dev(), false));

          const void* svmPtr = devMemory->owner()->getSvmPtr();
          void* hostPtr = mapMemory->getHostMem();
          if ((svmPtr != nullptr) && (hostPtr != svmPtr)) {
            // Wait on a kernel if one is outstanding
            releaseGpuMemoryFence();
            amd::Os::fastMemcpy(hostPtr, svmPtr, size[0]);
          }
          result = blitMgr().copyBuffer(*hsaMapMemory, *devMemory, mapInfo->origin_, mapInfo->origin_,
                                        mapInfo->region_, mapInfo->isEntire());
        } else {
          result = blitMgr().writeBuffer(cmd.mapPtr(), *devMemory, origin, size);
        }
      }
      if (!result) {
        LogError("submitMapMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
      }
    }

    cmd.memory().signalWrite(&dev());
  }

  devMemory->clearUnmapInfo(cmd.mapPtr());

  profilingEnd(cmd);
}

// ================================================================================================
bool VirtualGPU::fillMemory(cl_command_type type, amd::Memory* amdMemory, const void* pattern,
                            size_t patternSize, const amd::Coord3D& surface,
                            const amd::Coord3D& origin, const amd::Coord3D& size,
                            bool forceBlit) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  Memory* memory = dev().getRocMemory(amdMemory);

  bool entire = amdMemory->isEntirelyCovered(origin, size);
  // Synchronize memory from host if necessary
  device::Memory::SyncFlags syncFlags;
  syncFlags.skipEntire_ = entire;
  memory->syncCacheFromHost(*this, syncFlags);

  bool result = false;
  bool imageBuffer = false;
  float fillValue[4];

  // Force fill buffer for IMAGE1D_BUFFER
  if ((type == CL_COMMAND_FILL_IMAGE) && (amdMemory->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
    type = CL_COMMAND_FILL_BUFFER;
    imageBuffer = true;
  }

  // Find the the right fill operation
  switch (type) {
    case CL_COMMAND_SVM_MEMFILL:
    case CL_COMMAND_FILL_BUFFER: {
      amd::Coord3D realSurf(surface[0], surface[1], surface[2]);
      amd::Coord3D realOrigin(origin[0], origin[1], origin[2]);
      amd::Coord3D realSize(size[0], size[1], size[2]);
      // Reprogram fill parameters if it's an IMAGE1D_BUFFER object
      if (imageBuffer) {
        size_t elemSize = amdMemory->asImage()->getImageFormat().getElementSize();
        realOrigin.c[0] *= elemSize;
        realSize.c[0] *= elemSize;
        memset(fillValue, 0, sizeof(fillValue));
        amdMemory->asImage()->getImageFormat().formatColor(pattern, fillValue);
        pattern = fillValue;
        patternSize = elemSize;
      }
      result = blitMgr().fillBuffer(*memory, pattern, patternSize, realSurf, realOrigin,
                                    realSize, entire, forceBlit);
      break;
    }
    case CL_COMMAND_FILL_IMAGE: {
      result = blitMgr().fillImage(*memory, pattern, origin, size, entire);
      break;
    }
    default:
      ShouldNotReachHere();
      break;
  }

  if (!result) {
    LogError("submitFillMemory failed!");
  }

  amdMemory->signalWrite(&dev());
  return true;
}

// ================================================================================================
void VirtualGPU::submitFillMemory(amd::FillMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  bool force_blit = false;
  if (amd::IS_HIP) {
    constexpr uint32_t kManagedAlloc = (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR);
    // In case of HMM, use blit kernel instead of CPU memcpy
    if ((cmd.memory().getMemFlags() & kManagedAlloc) == kManagedAlloc) {
      force_blit = true;
    }
  }

  if (!fillMemory(cmd.type(), &cmd.memory(), cmd.pattern(), cmd.patternSize(),
    cmd.surface(), cmd.origin(), cmd.size(), force_blit)) {
    cmd.setStatus(CL_INVALID_OPERATION);
  }
  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitStreamOperation(amd::StreamOperationCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(cmd);

  const cl_command_type type = cmd.type();
  const uint64_t value = cmd.value();
  const uint64_t mask = cmd.mask();
  const unsigned int flags = cmd.flags();
  const size_t sizeBytes = cmd.sizeBytes();
  const size_t offset = cmd.offset();

  amd::Memory* amdMemory = &cmd.memory();
  Memory* memory = dev().getRocMemory(amdMemory);

  if (type == ROCCLR_COMMAND_STREAM_WAIT_VALUE) {
    if (GPU_STREAMOPS_CP_WAIT) {
      uint16_t header = kBarrierVendorPacketHeader;
      Buffer* buff = static_cast<Buffer*>(memory);
      hsa_signal_t signal = buff->getSignal();

      // mask is always applied on value at signal before performing
      // the comparision defiend by 'condition'
      switch (flags) {
        case ROCCLR_STREAM_WAIT_VALUE_GTE: {
          dispatchBarrierValuePacket(header, false, signal, value, mask,
                                     HSA_SIGNAL_CONDITION_GTE, true);
          break;
        }
        case ROCCLR_STREAM_WAIT_VALUE_EQ: {
          dispatchBarrierValuePacket(header,false, signal, value, mask,
                                     HSA_SIGNAL_CONDITION_EQ, true);
          break;
        }
        case ROCCLR_STREAM_WAIT_VALUE_AND: {
          dispatchBarrierValuePacket(header, false, signal, 0, (value & mask),
                                     HSA_SIGNAL_CONDITION_NE, true);
          break;
        }
        case ROCCLR_STREAM_WAIT_VALUE_NOR: {
          uint64_t norValue = ~value & mask;
          dispatchBarrierValuePacket(header, false, signal, norValue, norValue,
                                    HSA_SIGNAL_CONDITION_NE, true);
          break;
        }
        default:
          ShouldNotReachHere();
          break;
      }
    }
    // Use a blit kernel to perform the wait operation
    else {
    // mask is applied on value before performing
    // the comparision defined by 'condition'
      bool result = static_cast<KernelBlitManager&>(blitMgr()).streamOpsWait(*memory, value, offset,
                                                                              sizeBytes, flags, mask);
      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Waiting for value: 0x%lx."
              " Flags: 0x%lx mask: 0x%lx", value, flags, mask);
      if (!result) {
        LogError("submitStreamOperation: Wait failed!");
      }
    }
  } else if (type == ROCCLR_COMMAND_STREAM_WRITE_VALUE) {
    amd::Coord3D origin(offset);
    amd::Coord3D size(sizeBytes);
    // Ensure memory ordering preceding the write
    dispatchBarrierPacket(kBarrierPacketReleaseHeader);

    bool result = static_cast<KernelBlitManager&>(blitMgr()).streamOpsWrite(*memory, value,
                                                                            offset, sizeBytes);
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Writing value: 0x%lx", value);
    if (!result) {
      LogError("submitStreamOperation: Write failed!");
    }
  } else {
    ShouldNotReachHere();
  }
  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  amd::Memory* dstMemory = amd::MemObjMap::FindMemObj(cmd.dst());

  if (!dev().isFineGrainedSystem(true) ||
      ((dstMemory != nullptr) &&
       !dev().forceFineGrain(dstMemory))) {
    size_t patternSize = cmd.patternSize();
    size_t fillSize = patternSize * cmd.times();

    size_t offset = reinterpret_cast<uintptr_t>(cmd.dst()) -
        reinterpret_cast<uintptr_t>(dstMemory->getSvmPtr());

    Memory* memory = dev().getRocMemory(dstMemory);

    amd::Coord3D origin(offset, 0, 0);
    amd::Coord3D size(fillSize, 1, 1);

    assert((dstMemory->validateRegion(origin, size)) && "The incorrect fill size!");

    if (!fillMemory(cmd.type(), dstMemory, cmd.pattern(), cmd.patternSize(), size, origin, size,
                    true)) {
      cmd.setStatus(CL_INVALID_OPERATION);
    }
  } else {
    // Stall GPU for CPU access to memory
    releaseGpuMemoryFence();
    // for FGS capable device, fill CPU memory directly
    amd::SvmBuffer::memFill(cmd.dst(), cmd.pattern(), cmd.patternSize(), cmd.times());
  }

  profilingEnd(cmd);
}

// ================================================================================================
void VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  for (auto itr : vcmd.memObjects()) {
    // Find device memory
    Memory* memory = dev().getRocMemory(&(*itr));

    if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_HOST) {
      if (!memory->isHostMemDirectAccess()) {
        // Make sure GPU finished operation before synchronization with the backing store
        releaseGpuMemoryFence();
      }
      memory->mgpuCacheWriteBack(*this);
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

// ================================================================================================
static void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data) {
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    // Abort on device exceptions.
    const char* errorMsg = 0;
    hsa_status_string(status, &errorMsg);
    if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      size_t global_available_mem = 0;
      VirtualGPU* vgpu = reinterpret_cast<VirtualGPU*>(data);
      if (HSA_STATUS_SUCCESS != hsa_agent_get_info(vgpu->gpu_device(),
                         static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_AVAIL),
                         &global_available_mem)) {
        LogError("HSA_AMD_AGENT_INFO_MEMORY_AVAIL query failed.");
      }
      ClPrint(amd::LOG_NONE, amd::LOG_ALWAYS,
              "Callback: Queue %p Aborting with error : %s Code: 0x%x Available Free mem : %zu MB",
              queue->base_address, errorMsg, status, global_available_mem/Mi);
    } else {
      ClPrint(amd::LOG_NONE, amd::LOG_ALWAYS,
        "Callback: Queue %p aborting with error : %s code: 0x%x", queue->base_address,
        errorMsg, status);
    }
    abort();
  }
}

// ================================================================================================
bool VirtualGPU::createSchedulerParam()
{
  if (nullptr != schedulerParam_) {
    return true;
  }

  while(true) {
    schedulerParam_ = new (dev().context()) amd::Buffer(dev().context(),
      CL_MEM_ALLOC_HOST_PTR, sizeof(SchedulerParam) + sizeof(AmdAqlWrap));

    if ((nullptr != schedulerParam_) && !schedulerParam_->create(nullptr)) {
      break;
    }

    // The queue is written by multiple threads of the scheduler kernel
    if (HSA_STATUS_SUCCESS != hsa_queue_create(gpu_device(), 2048, HSA_QUEUE_TYPE_MULTI,
        callbackQueue, this, std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(),
        &schedulerQueue_)) {
      break;
    }

    hsa_signal_t  signal0 = {0};

    if (HSA_STATUS_SUCCESS != hsa_signal_create(0, 0, nullptr, &signal0)) {
      break;
    }

    schedulerSignal_ = signal0;

    Memory* schedulerMem = dev().getRocMemory(schedulerParam_);

    if (nullptr == schedulerMem) {
      break;
    }

    schedulerParam_->setVirtualDevice(this);
    return true;
  }

  if (0 != schedulerSignal_.handle) {
    hsa_signal_destroy(schedulerSignal_);
    schedulerSignal_.handle = 0;
  }

  if (nullptr != schedulerQueue_) {
    hsa_queue_destroy(schedulerQueue_);
    schedulerQueue_ = nullptr;
  }

  if (nullptr != schedulerParam_) {
    schedulerParam_->release();
    schedulerParam_ = nullptr;
  }

  return false;
}

// ================================================================================================
uint64_t VirtualGPU::getVQVirtualAddress()
{
  Memory* vqMem = dev().getRocMemory(virtualQueue_);
  return reinterpret_cast<uint64_t>(vqMem->getDeviceMemory());
}

// ================================================================================================
bool VirtualGPU::createVirtualQueue(uint deviceQueueSize)
{
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
    if (0 != deviceQueueSize_) {
      virtualQueue_->release();
      virtualQueue_ = nullptr;
      deviceQueueSize_ = 0;
      schedulerThreads_ = 0;
    }
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

  // Make sure the allocation size aligns with DWORD.
  allocSize = amd::alignUp(allocSize, sizeof(uint64_t));

  // CL_MEM_ALLOC_HOST_PTR/CL_MEM_READ_WRITE
  virtualQueue_ = new (dev().context()) amd::Buffer(dev().context(), CL_MEM_READ_WRITE, allocSize);

  if ((nullptr != virtualQueue_) && !virtualQueue_->create(nullptr)) {
    virtualQueue_->release();
    return false;
  }

  Memory* vqMem = dev().getRocMemory(virtualQueue_);

  if (nullptr == vqMem) {
    return false;
  }

  uint64_t vqVA = reinterpret_cast<uint64_t>(vqMem->getDeviceMemory());
  uint64_t pattern = 0;
  amd::Coord3D origin(0, 0, 0);
  amd::Coord3D region(virtualQueue_->getSize(), 1, 1);

  if (!dev().xferMgr().fillBuffer(*vqMem, &pattern, sizeof(pattern), region, origin, region)) {
    return false;
  }

  AmdVQueueHeader header = {};
  // Initialize the virtual queue header
  header.aql_slot_num = numSlots;
  header.event_slot_num = dev().settings().numDeviceEvents_;
  header.event_slot_mask = vqVA + eventMaskOffs;
  header.event_slots = vqVA + eventsOffs;
  header.aql_slot_mask = vqVA + slotMaskOffs;
  header.wait_size = dev().settings().numWaitEvents_;
  header.arg_size = dev().info().maxParameterSize_ + 64;
  header.mask_groups = maskGroups_;

  amd::Coord3D origin_header(0);
  amd::Coord3D region_header(sizeof(AmdVQueueHeader));

  if (!dev().xferMgr().writeBuffer(&header, *vqMem, origin_header, region_header)) {
    return false;
  }

  // Go over all slots and perform initialization
  AmdAqlWrap slot = {};
  size_t offset = sizeof(AmdVQueueHeader);
  for (uint i = 0; i < numSlots; ++i) {
    uint64_t argStart = vqVA + argOffs + i * singleArgSize;
    amd::Coord3D origin_slot(offset);
    amd::Coord3D region_slot(sizeof(AmdAqlWrap));

    slot.aql.kernarg_address = reinterpret_cast<void*>(argStart);
    slot.wait_list = argStart + dev().info().maxParameterSize_ + 64;

    if (!dev().xferMgr().writeBuffer(&slot, *vqMem, origin_slot, region_slot)) {
      return false;
    }

    offset += sizeof(AmdAqlWrap);
  }

  deviceQueueSize_ = deviceQueueSize;
  schedulerThreads_ = numSlots / (DeviceQueueMaskSize * maskGroups_);

  return true;
}

// ================================================================================================
bool VirtualGPU::submitKernelInternal(const amd::NDRangeContainer& sizes,
    const amd::Kernel& kernel, const_address parameters, void* eventHandle,
    uint32_t sharedMemBytes, amd::NDRangeKernelCommand* vcmd,
    hsa_kernel_dispatch_packet_t* aql_packet) {
  device::Kernel* devKernel = const_cast<device::Kernel*>(kernel.getDeviceKernel(dev()));
  Kernel& gpuKernel = static_cast<Kernel&>(*devKernel);
  size_t ldsUsage = gpuKernel.WorkgroupGroupSegmentByteSize();
  bool imageBufferWrtBack = false; // Image buffer write back is required
  std::vector<device::Memory*> wrtBackImageBuffer; // Array of images for write back

  // Check memory dependency and SVM objects
  bool coopGroups = (vcmd != nullptr) ? vcmd->cooperativeGroups() : false;
  if (!processMemObjects(kernel, parameters, ldsUsage, coopGroups,
                         imageBufferWrtBack, wrtBackImageBuffer)) {
    LogError("Wrong memory objects!");
    return false;
  }

  // Init PrintfDbg object if printf is enabled.
  bool printfEnabled = (gpuKernel.printfInfo().size() > 0) ? true : false;
  if (!printfDbg()->init(printfEnabled)) {
    LogError("\nPrintfDbg object initialization failed!");
    return false;
  }

  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  size_t newOffset[3] = {0, 0, 0};
  size_t newGlobalSize[3] = {0, 0, 0};

  int dim = -1;
  int iteration = 1;
  size_t globalStep = 0;
  for (uint i = 0; i < sizes.dimensions(); i++) {
    newGlobalSize[i] = sizes.global()[i];
    newOffset[i] = sizes.offset()[i];
  }

  if (gpuKernel.isInternalKernel()) {
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

  amd::Memory* const* memories =
      reinterpret_cast<amd::Memory* const*>(parameters + kernelParams.memoryObjOffset());

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

    ClPrint(amd::LOG_INFO, amd::LOG_KERN, "ShaderName : %s", gpuKernel.name().c_str());

    amd::NDRange local(sizes.local());
    address hidden_arguments = const_cast<address>(parameters);
    // Calculate local size if it wasn't provided
    devKernel->FindLocalWorkSize(sizes.dimensions(), sizes.global(), local);

    // Check if runtime has to setup hidden arguments
    for (uint32_t i = signature.numParameters(); i < signature.numParametersAll(); ++i) {
      const auto& it = signature.at(i);
      switch (it.info_.oclObject_) {
        case amd::KernelParameterDescriptor::HiddenNone:
          break;
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetX: {
          WriteAqlArgAt(hidden_arguments, newOffset[0], it.size_, it.offset_);
          break;
        }
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetY: {
          if (sizes.dimensions() >= 2) {
            WriteAqlArgAt(hidden_arguments, newOffset[1], it.size_, it.offset_);
          }
          break;
        }
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetZ: {
          if (sizes.dimensions() >= 3) {
            WriteAqlArgAt(hidden_arguments, newOffset[2], it.size_, it.offset_);
          }
          break;
        }
        case amd::KernelParameterDescriptor::HiddenPrintfBuffer: {
          uintptr_t bufferPtr = reinterpret_cast<uintptr_t>(printfDbg()->dbgBuffer());
          if (printfEnabled && bufferPtr) {
            WriteAqlArgAt(hidden_arguments, bufferPtr, it.size_, it.offset_);
          }
          break;
        }
        case amd::KernelParameterDescriptor::HiddenHostcallBuffer: {
          if (amd::IS_HIP) {
            if (dev().info().pcie_atomics_) {
              uintptr_t buffer = reinterpret_cast<uintptr_t>(
                roc_device_.getOrCreateHostcallBuffer(gpu_queue_, coopGroups, cuMask_));
              if (!buffer) {
                ClPrint(amd::LOG_ERROR, amd::LOG_KERN,
                        "Kernel expects a hostcall buffer, but none found");
                return false;
              }
              WriteAqlArgAt(hidden_arguments, buffer, it.size_, it.offset_);
            } else {
              ClPrint(amd::LOG_ERROR, amd::LOG_KERN,
                      "Pcie atomics not enabled, hostcall not supported");
              return false;
            }
          }
          break;
        }
        case amd::KernelParameterDescriptor::HiddenDefaultQueue: {
          uint64_t vqVA = 0;
          amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(dev());
          if (nullptr != defQueue) {
            if (!createVirtualQueue(defQueue->size()) || !createSchedulerParam()) {
              return false;
            }
            vqVA = getVQVirtualAddress();
          }
          WriteAqlArgAt(hidden_arguments, vqVA, it.size_, it.offset_);
          break;
        }
        case amd::KernelParameterDescriptor::HiddenCompletionAction: {
          uint64_t spVA = 0;
          if (nullptr != schedulerParam_) {
            Memory* schedulerMem = dev().getRocMemory(schedulerParam_);
            AmdAqlWrap* wrap = reinterpret_cast<AmdAqlWrap*>(
                               reinterpret_cast<uint64_t>(schedulerParam_->getHostMem()) + sizeof(SchedulerParam));
            memset(wrap, 0, sizeof(AmdAqlWrap));
            wrap->state = AQL_WRAP_DONE;

            spVA = reinterpret_cast<uint64_t>(schedulerMem->getDeviceMemory()) + sizeof(SchedulerParam);
          }
          WriteAqlArgAt(hidden_arguments, spVA, it.size_, it.offset_);
          break;
        }
        case amd::KernelParameterDescriptor::HiddenMultiGridSync: {
          uint64_t gridSync = coopGroups ? 1 : 0;
          bool multiGrid = (vcmd != nullptr) ? vcmd->cooperativeMultiDeviceGroups() : false;
          Device::MGSyncInfo* syncInfo = nullptr;
          if (multiGrid) {
            // Find CPU pointer to the right sync info structure. It should be after MGSyncData
            syncInfo = reinterpret_cast<Device::MGSyncInfo*>(
              dev().MGSync() + Device::kMGInfoSizePerDevice * dev().index() + Device::kMGSyncDataSize);
            // Update sync data address. Use the offset adjustment to the right location
            syncInfo->mgs = reinterpret_cast<Device::MGSyncData*>(dev().MGSync() +
                            Device::kMGInfoSizePerDevice * vcmd->firstDevice());
          }
          else {
            syncInfo = reinterpret_cast<Device::MGSyncInfo*>(allocKernArg(Device::kSGInfoSize, 64));
            syncInfo->mgs = nullptr;
          }
          // Update sync data address.
          syncInfo->sgs = {0};
          // Fill all sync info fields
          syncInfo->grid_id = vcmd->gridId();
          syncInfo->num_grids = vcmd->numGrids();
          syncInfo->prev_sum = vcmd->prevGridSum();
          syncInfo->all_sum = vcmd->allGridSum();
          syncInfo->num_wg = vcmd->numWorkgroups();
          // Update GPU address for grid sync info. Use the offset adjustment for the right
          // location
          gridSync = reinterpret_cast<uint64_t>(syncInfo);
          WriteAqlArgAt(hidden_arguments, gridSync, it.size_, it.offset_);
          break;
        }
        case amd::KernelParameterDescriptor::HiddenHeap:
          // Allocate hidden heap for HIP applications only
          if ((amd::IS_HIP) && (dev().HeapBuffer() == nullptr)) {
            const_cast<Device&>(dev()).HiddenHeapAlloc(*this);
          }
          if (dev().HeapBuffer() != nullptr) {
            // Add heap pointer to the code
            size_t heap_ptr = static_cast<size_t>(dev().HeapBuffer()->virtualAddress());
            WriteAqlArgAt(hidden_arguments, heap_ptr, it.size_, it.offset_);
          }
          break;
        case amd::KernelParameterDescriptor::HiddenBlockCountX:
          WriteAqlArgAt(hidden_arguments, static_cast<uint32_t>(newGlobalSize[0] / local[0]),
                        it.size_, it.offset_);
          break;
        case amd::KernelParameterDescriptor::HiddenBlockCountY:
          if (sizes.dimensions() >= 2) {
            WriteAqlArgAt(hidden_arguments, static_cast<uint32_t>(newGlobalSize[1] / local[1]),
                          it.size_, it.offset_);
          } else {
            WriteAqlArgAt(hidden_arguments, static_cast<uint32_t>(1), it.size_, it.offset_);
          }
          break;
        case amd::KernelParameterDescriptor::HiddenBlockCountZ:
          if (sizes.dimensions() >= 3) {
            WriteAqlArgAt(hidden_arguments, static_cast<uint32_t>(newGlobalSize[2] / local[2]),
                          it.size_, it.offset_);
          } else {
            WriteAqlArgAt(hidden_arguments, static_cast<uint32_t>(1), it.size_, it.offset_);
          }
          break;
        case amd::KernelParameterDescriptor::HiddenGroupSizeX:
          WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(local[0]), it.size_, it.offset_);
          break;
        case amd::KernelParameterDescriptor::HiddenGroupSizeY:
          if (sizes.dimensions() >= 2) {
            WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(local[1]), it.size_, it.offset_);
          } else {
            WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(1), it.size_, it.offset_);
          }
          break;
        case amd::KernelParameterDescriptor::HiddenGroupSizeZ:
          if (sizes.dimensions() >= 3) {
            WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(local[2]), it.size_, it.offset_);
          } else {
            WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(1), it.size_, it.offset_);
          }
          break;
        case amd::KernelParameterDescriptor::HiddenRemainderX:
          WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(newGlobalSize[0] % local[0]),
                        it.size_, it.offset_);
          break;
        case amd::KernelParameterDescriptor::HiddenRemainderY:
          if (sizes.dimensions() >= 2) {
            WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(newGlobalSize[1] % local[1]),
                          it.size_, it.offset_);
          }
          break;
        case amd::KernelParameterDescriptor::HiddenRemainderZ:
          if (sizes.dimensions() >= 3) {
            WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(newGlobalSize[2] % local[2]),
                          it.size_, it.offset_);
          }
          break;
        case amd::KernelParameterDescriptor::HiddenGridDims:
          WriteAqlArgAt(hidden_arguments, static_cast<uint16_t>(sizes.dimensions()),
                        it.size_, it.offset_);
          break;
        case amd::KernelParameterDescriptor::HiddenPrivateBase:
          WriteAqlArgAt(hidden_arguments,
                        reinterpret_cast<amd_queue_t*>(gpu_queue_)->private_segment_aperture_base_hi,
                        it.size_, it.offset_);
          break;
        case amd::KernelParameterDescriptor::HiddenSharedBase:
          WriteAqlArgAt(hidden_arguments,
                        reinterpret_cast<amd_queue_t*>(gpu_queue_)->group_segment_aperture_base_hi,
                        it.size_, it.offset_);
          break;
        case amd::KernelParameterDescriptor::HiddenQueuePtr:
          WriteAqlArgAt(hidden_arguments, gpu_queue_, it.size_, it.offset_);
          break;
      }
    }

    address argBuffer = hidden_arguments;
    // Find all parameters for the current kernel
    if (!kernel.parameters().deviceKernelArgs() || gpuKernel.isInternalKernel()) {
      // Allocate buffer to hold kernel arguments
      argBuffer = reinterpret_cast<address>(allocKernArg(gpuKernel.KernargSegmentByteSize(),
                                            gpuKernel.KernargSegmentAlignment()));
      // Load all kernel arguments
      memcpy(argBuffer, parameters, std::min(gpuKernel.KernargSegmentByteSize(),
                                             signature.paramsSize()));
    }

    // Check for group memory overflow
    //! @todo Check should be in HSA - here we should have at most an assert
    assert(dev().info().localMemSizePerCU_ > 0);
    if (ldsUsage > dev().info().localMemSizePerCU_) {
      LogError("No local memory available\n");
      return false;
    }

    // Initialize the dispatch Packet
    hsa_kernel_dispatch_packet_t dispatchPacket;
    memset(&dispatchPacket, 0, sizeof(dispatchPacket));

    dispatchPacket.header = kInvalidAql;
    dispatchPacket.kernel_object = gpuKernel.KernelCodeHandle();

   // dispatchPacket.header = aqlHeader_;
    // dispatchPacket.setup |= sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    dispatchPacket.grid_size_x = sizes.dimensions() > 0 ? newGlobalSize[0] : 1;
    dispatchPacket.grid_size_y = sizes.dimensions() > 1 ? newGlobalSize[1] : 1;
    dispatchPacket.grid_size_z = sizes.dimensions() > 2 ? newGlobalSize[2] : 1;

    dispatchPacket.workgroup_size_x = sizes.dimensions() > 0 ? local[0] : 1;
    dispatchPacket.workgroup_size_y = sizes.dimensions() > 1 ? local[1] : 1;
    dispatchPacket.workgroup_size_z = sizes.dimensions() > 2 ? local[2] : 1;

    dispatchPacket.kernarg_address = argBuffer;
    dispatchPacket.group_segment_size = ldsUsage + sharedMemBytes;
    dispatchPacket.private_segment_size = devKernel->workGroupInfo()->privateMemSize_;

    if ((devKernel->workGroupInfo()->usedStackSize_ & 0x1) == 0x1) {
      dispatchPacket.private_segment_size =
              std::max<uint64_t>(dev().StackSize(), dispatchPacket.private_segment_size);
      if (dispatchPacket.private_segment_size > 16 * Ki) {
        dispatchPacket.private_segment_size = 16 * Ki;
      }
    }

    // Pass the header accordingly
    auto aqlHeaderWithOrder = aqlHeader_;
    if (vcmd != nullptr && vcmd->getAnyOrderLaunchFlag()) {
      constexpr uint32_t kAqlHeaderMask = ~(1 << HSA_PACKET_HEADER_BARRIER);
      aqlHeaderWithOrder &= kAqlHeaderMask;
    }

    if (addSystemScope_ || (vcmd != nullptr &&
        vcmd->getEventScope() == amd::Device::kCacheStateSystem)) {
      aqlHeaderWithOrder &= ~(HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE |
                              HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      aqlHeaderWithOrder |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE |
                             HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      addSystemScope_ = false;
    }

    // If profiling is enabled, store the correlation ID in the dispatch packet. The profiler can
    // retrieve this correlation ID to attribute waves to specific dispatch locations.
    if (vcmd != nullptr && vcmd->profilingInfo().enabled_) {
      dispatchPacket.reserved2 = vcmd->profilingInfo().correlation_id_;
    }

    // Copy scheduler's AQL packet for possible relaunch from the scheduler itself
    if (aql_packet != nullptr) {
      *aql_packet = dispatchPacket;
      aql_packet->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                           (1 << HSA_PACKET_HEADER_BARRIER) |
                           (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                           (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
      aql_packet->setup = sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    }

    // Dispatch the packet
    if (!dispatchAqlPacket(
            &dispatchPacket, aqlHeaderWithOrder,
            (sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS),
            GPU_FLUSH_ON_EXECUTION)) {
      return false;
    }
  }

  // Mark the flag indicating if a dispatch is outstanding.
  // We are not waiting after every dispatch.
  hasPendingDispatch_ = true;

  // Output printf buffer
  if (!printfDbg()->output(*this, printfEnabled, gpuKernel.printfInfo())) {
    LogError("\nCould not print data from the printf buffer!");
    return false;
  }

  if (gpuKernel.dynamicParallelism()) {
    dispatchBarrierPacket(kBarrierPacketHeader, true);
    static_cast<KernelBlitManager&>(blitMgr()).runScheduler(
        getVQVirtualAddress(), schedulerParam_, schedulerQueue_, schedulerSignal_, schedulerThreads_);
  }

  // Check if image buffer write back is required
  if (imageBufferWrtBack) {
    // Make sure the original kernel execution is done
    releaseGpuMemoryFence();
    for (const auto imageBuffer : wrtBackImageBuffer) {
      Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
      amd::Image* image = imageBuffer->owner()->asImage();
      Image* devImage = static_cast<Image*>(dev().getGpuMemory(imageBuffer->owner()));
      Memory* cpyImage = dev().getGpuMemory(devImage->CopyImageBuffer());
      amd::Coord3D offs(0);
      // Copy memory from the the backing store image into original buffer
      bool result = blitMgr().copyImageToBuffer(*cpyImage, *buffer, offs,
                                                offs, image->getRegion(), true,
                                                image->getRowPitch(), image->getSlicePitch());
    }
  }
  return true;
}

/**
 * @brief Api to dispatch a kernel for execution. The implementation
 * parses the input object, an instance of virtual command to obtain
 * the parameters of global size, work group size, offsets of work
 * items, enable/disable profiling, etc.
 *
 * It also parses the kernel arguments buffer to inject into Hsa Runtime
 * the list of kernel parameters.
 */
 // ================================================================================================
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd) {
  if (vcmd.cooperativeGroups()) {
    // Wait for the execution on the current queue, since the coop groups will use the device queue
    releaseGpuMemoryFence(kSkipCpuWait);

    // Get device queue for exclusive GPU access
    VirtualGPU* queue = dev().xferQueue();
    if (!queue) {
      LogError("Runtime failed to acquire a cooperative queue!");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }

    // Lock the queue, using the blit manager lock
    amd::ScopedLock lock(queue->blitMgr().lockXfer());

    queue->profilingBegin(vcmd);

    // Add a dependency into the device queue on the current queue
    queue->Barriers().AddExternalSignal(Barriers().GetLastSignal());

    if (!Settings().coop_sync_) {
      uint32_t workgroups = vcmd.numWorkgroups();
      static_cast<KernelBlitManager&>(queue->blitMgr()).RunGwsInit(workgroups - 1);
    }

    // Sync AQL packets
    queue->setAqlHeader(dispatchPacketHeader_);

    // Submit kernel to HW
    if (!queue->submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
      static_cast<void*>(as_cl(&vcmd.event())), vcmd.sharedMemBytes(), &vcmd)) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
    // Wait for the execution on the device queue. Keep the current queue in-order
    queue->releaseGpuMemoryFence(kSkipCpuWait);

    // Add a dependency into the current queue on the coop queue
    Barriers().AddExternalSignal(queue->Barriers().GetLastSignal());
    hasPendingDispatch_ = true;
    retainExternalSignals_ = true;

    queue->profilingEnd(vcmd);
  } else {
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);

    // Submit kernel to HW
    if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
      static_cast<void*>(as_cl(&vcmd.event())), vcmd.sharedMemBytes(), &vcmd)) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
  }
}

// ================================================================================================
void VirtualGPU::submitNativeFn(amd::NativeFnCommand& cmd) {
  // std::cout<<__FUNCTION__<<" not implemented"<<"*********"<<std::endl;
}

// ================================================================================================
void VirtualGPU::submitMarker(amd::Marker& vcmd) {
  if (AMD_DIRECT_DISPATCH || vcmd.profilingInfo().marker_ts_) {
    // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());
    if (vcmd.CpuWaitRequested()) {
      // It should be safe to call flush directly if there are not pending dispatches without
      // HSA signal callback
      flush(vcmd.GetBatchHead());
    } else {
      profilingBegin(vcmd);
      if (timestamp_ != nullptr) {
        const Settings& settings = dev().settings();
        int32_t releaseFlags = vcmd.getEventScope();
        if (releaseFlags == Device::CacheState::kCacheStateAgent) {
          if (settings.barrier_value_packet_ && vcmd.profilingInfo().marker_ts_) {
            dispatchBarrierValuePacket(kBarrierVendorPacketAgentScopeHeader, true);
          } else {
            dispatchBarrierPacket(kBarrierPacketAgentScopeHeader, false);
          }
        } else {
          // Submit a barrier with a cache flushes.
          if (settings.barrier_value_packet_ && vcmd.profilingInfo().marker_ts_) {
            dispatchBarrierValuePacket(kBarrierVendorPacketHeader, true);
          } else {
            dispatchBarrierPacket(kBarrierPacketHeader, false);
          }
          hasPendingDispatch_ = false;
        }
      }
      profilingEnd(vcmd);
    }

  }
}

// ================================================================================================
void VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);
  addSystemScope();
  profilingEnd(vcmd);
}

// ================================================================================================
void VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);
  profilingEnd(vcmd);
}

// ================================================================================================
void VirtualGPU::flush(amd::Command* list, bool wait) {
  // If barrier is requested, then wait for everything, otherwise
  // a per disaptch wait will occur later in updateCommandsState()
  releaseGpuMemoryFence();
  updateCommandsState(list);

  // Release all pinned memory
  releasePinnedMem();
}

// ================================================================================================
void VirtualGPU::addXferWrite(Memory& memory) {
  //! @note: ROCr backend doesn't have per resource busy tracking, hence runtime has to wait
  //!        unconditionally, before it can release pinned memory
  releaseGpuMemoryFence();
  if (xferWriteBuffers_.size() > 7) {
    dev().xferWrite().release(*this, *xferWriteBuffers_.front());
    xferWriteBuffers_.erase(xferWriteBuffers_.begin());
  }

  // Delay destruction
  xferWriteBuffers_.push_back(&memory);
}

// ================================================================================================
void VirtualGPU::releaseXferWrite() {
  for (auto& memory : xferWriteBuffers_) {
    dev().xferWrite().release(*this, *memory);
  }
  xferWriteBuffers_.resize(0);
}

// ================================================================================================
void VirtualGPU::addPinnedMem(amd::Memory* mem) {
  //! @note: ROCr backend doesn't have per resource busy tracking, hence runtime has to wait
  //!        unconditionally, before it can release pinned memory
  releaseGpuMemoryFence();
  if (!AMD_DIRECT_DISPATCH) {
    if (nullptr == findPinnedMem(mem->getHostMem(), mem->getSize())) {
      if (pinnedMems_.size() > 7) {
        pinnedMems_.front()->release();
        pinnedMems_.erase(pinnedMems_.begin());
      }

      // Delay destruction
      pinnedMems_.push_back(mem);
    }
  } else {
    mem->release();
  }
}

// ================================================================================================
void VirtualGPU::releasePinnedMem() {
  for (auto& amdMemory : pinnedMems_) {
    amdMemory->release();
  }
  pinnedMems_.resize(0);
}

// ================================================================================================
amd::Memory* VirtualGPU::findPinnedMem(void* addr, size_t size) {
  for (auto& amdMemory : pinnedMems_) {
    if ((amdMemory->getHostMem() == addr) && (size <= amdMemory->getSize())) {
      return amdMemory;
    }
  }
  return nullptr;
}

// ================================================================================================
void VirtualGPU::enableSyncBlit() const { blitMgr_->enableSynchronization(); }

// ================================================================================================
void VirtualGPU::submitPerfCounter(amd::PerfCounterCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  const amd::PerfCounterCommand::PerfCounterList counters = vcmd.getCounters();

  if (vcmd.getState() == amd::PerfCounterCommand::Begin) {
    // Create a profile for the profiling AQL packet
    PerfCounterProfile* profileRef =  new PerfCounterProfile(roc_device_);
    if (profileRef == nullptr || !profileRef->Create()) {
      LogError("Failed to create performance counter profile");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }

    // Make sure all performance counter objects to use the same profile
    PerfCounter* counter = nullptr;
    for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
      amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[i]);
      counter = static_cast<PerfCounter*>(amdCounter->getDeviceCounter());

      if (nullptr == counter) {
        amd::PerfCounter::Properties prop = amdCounter->properties();
        PerfCounter* rocCounter = new PerfCounter(
            roc_device_, prop[CL_PERFCOUNTER_GPU_BLOCK_INDEX],
            prop[CL_PERFCOUNTER_GPU_COUNTER_INDEX], prop[CL_PERFCOUNTER_GPU_EVENT_INDEX]);

        if (nullptr == rocCounter || rocCounter->gfxVersion() == PerfCounter::ROC_UNSUPPORTED) {
          LogError("Failed to create the performance counter");
          vcmd.setStatus(CL_INVALID_OPERATION);
          delete rocCounter;
          return;
        }

        amdCounter->setDeviceCounter(rocCounter);
        counter = rocCounter;
      }

      counter->setProfile(profileRef);
    }

    if (!profileRef->initialize()) {
      LogError("Failed to initialize performance counter");
      vcmd.setStatus(CL_INVALID_OPERATION);
    } else if (profileRef->createStartPacket() == nullptr) {
      LogError("Failed to create AQL packet for start profiling");
      vcmd.setStatus(CL_INVALID_OPERATION);
    } else {
        dispatchCounterAqlPacket(profileRef->prePacket(), counter->gfxVersion(), false,
                                 profileRef->api());
    }

    profileRef->release();
  } else if (vcmd.getState() == amd::PerfCounterCommand::End) {
    // Since all performance counters should use the same profile, use the 1st
    // one to get the profile object
    amd::PerfCounter* amdCounter = static_cast<amd::PerfCounter*>(counters[0]);
    PerfCounter* counter = static_cast<PerfCounter*>(amdCounter->getDeviceCounter());
    PerfCounterProfile* profileRef =  counter->profileRef();

    // create the AQL packet for stop profiling
    if (profileRef->createStopPacket() == nullptr) {
      LogError("Failed to create AQL packet for stop profiling");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }
    dispatchCounterAqlPacket(profileRef->postPacket(), counter->gfxVersion(), true,
                             profileRef->api());
  } else {
    LogError("Unsupported performance counter state");
    vcmd.setStatus(CL_INVALID_OPERATION);
  }

}

}  // End of roc namespace
