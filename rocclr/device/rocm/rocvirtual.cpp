/* Copyright (c) 2013-present Advanced Micro Devices, Inc.

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

#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/roccounters.hpp"
#include "platform/kernel.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "rochostcall.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"
#include "amd_hsa_kernel_code.h"

#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <thread>

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

static const uint16_t kInvalidAql =
    (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);

static const uint16_t kBarrierPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kBarrierPacketAcquireHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kBarrierPacketReleaseHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const hsa_barrier_and_packet_t kBarrierAcquirePacket = {
    kBarrierPacketAcquireHeader, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static const hsa_barrier_and_packet_t kBarrierReleasePacket = {
    kBarrierPacketReleaseHeader, 0, 0, 0, 0, 0, 0, 0, 0, 0};

double Timestamp::ticksToTime_ = 0;

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

bool VirtualGPU::processMemObjects(const amd::Kernel& kernel, const_address params,
  size_t& ldsAddress, bool cooperativeGroups) {
  Kernel& hsaKernel = const_cast<Kernel&>(static_cast<const Kernel&>(*(kernel.getDeviceKernel(dev()))));
  const amd::KernelSignature& signature = kernel.signature();
  const amd::KernelParameters& kernelParams = kernel.parameters();

  if (!cooperativeGroups && memoryDependency().maxMemObjectsInQueue() != 0) {
    // AQL packets
    setAqlHeader(dispatchPacketHeaderNoSync_);
  }

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

  amd::Memory* const* memories =
    reinterpret_cast<amd::Memory* const*>(params + kernelParams.memoryObjOffset());

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
          WriteAqlArgAt(const_cast<address>(params), &ldsAddress, desc.size_, desc.offset_);
          // Add the original size
          ldsAddress += ldsSize;
        } else {
          // Save the original LDS size
          uint32_t ldsSize = *reinterpret_cast<const uint32_t*>(params + desc.offset_);
          // Patch the LDS address in the original arguments with an LDS address(offset)
          uint32_t ldsAddr = ldsAddress;
          WriteAqlArgAt(const_cast<address>(params), &ldsAddr, desc.size_, desc.offset_);
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
          // Don't sync for internal objects,
          // since they are not shared between devices
          if (gpuMem->owner()->getVirtualDevice() == nullptr) {
            // Synchronize data with other memory instances if necessary
            gpuMem->syncCacheFromHost(*this);
          }
          const void* globalAddress = *reinterpret_cast<const void* const*>(params + desc.offset_);
          ClPrint(amd::LOG_INFO, amd::LOG_KERN, "!\targ%d: %s %s = ptr:%p obj:[%p-%p] threadId : %zx\n", index,
            desc.typeName_.c_str(), desc.name_.c_str(),
            globalAddress, gpuMem->getDeviceMemory(),
            reinterpret_cast<address>(gpuMem->getDeviceMemory()) + mem->getSize(),
            std::this_thread::get_id());

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
            WriteAqlArgAt(const_cast<address>(params), &image_srd, sizeof(image_srd), desc.offset_);

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
              WriteAqlArgAt(const_cast<address>(params), &srd, sizeof(srd), desc.offset_);

              // If it's not a read only resource, then runtime has to write back
              if (!desc.info_.readOnly_) {
                wrtBackImageBuffer_.push_back(mem->getDeviceMemory(dev()));
                imageBufferWrtBack_ = true;
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
      WriteAqlArgAt(const_cast<address>(params), &vqVA, sizeof(vqVA), desc.offset_);
    }
    else if (desc.type_ == T_VOID) {
      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ReferenceObject) {
        const_address srcArgPtr = params + desc.offset_;
        void* mem = allocKernArg(desc.size_, 128);
        if (mem == nullptr) {
          LogError("Out of memory");
          return false;
        }
        memcpy(mem, srcArgPtr, desc.size_);
        const auto it = hsaKernel.patch().find(desc.offset_);
        WriteAqlArgAt(const_cast<address>(params), &mem, sizeof(void*), it->second);
      }
    }
    else if (desc.type_ == T_SAMPLER) {
      uint32_t index = desc.info_.arrayIndex_;
      const amd::Sampler* sampler = reinterpret_cast<amd::Sampler* const*>(params +
        kernelParams.samplerObjOffset())[index];

      device::Sampler* devSampler = sampler->getDeviceSampler(dev());

      uint64_t sampler_srd = devSampler->hwSrd();
      WriteAqlArgAt(const_cast<address>(params), &sampler_srd, sizeof(sampler_srd), desc.offset_);
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

static inline void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
}

template <typename AqlPacket>
bool VirtualGPU::dispatchGenericAqlPacket(
  AqlPacket* packet, uint16_t header, uint16_t rest, bool blocking, size_t size) {
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;

  // Check for queue full and wait if needed.
  uint64_t index = hsa_queue_add_write_index_screlease(gpu_queue_, size);
  uint64_t read = hsa_queue_load_read_index_relaxed(gpu_queue_);
  hsa_signal_t signal;

  if (addSystemScope_) {
    header &= ~(HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
    addSystemScope_ = false;
  }

  // TODO: placeholder to setup the kernel to populate start and end timestamp.
  if (timestamp_ != nullptr) {
    // Find signal slot
    ProfilingSignal* profilingSignal = &signal_pool_[index & queueMask];
    // Make sure we save the old results in the TS structure
    if (profilingSignal->ts_ != nullptr) {
      profilingSignal->ts_->checkGpuTime();
    }
    // Update the new TS with the signal info
    timestamp_->setProfilingSignal(profilingSignal);
    packet->completion_signal = profilingSignal->signal_;
    profilingSignal->ts_ = timestamp_;
    timestamp_->setAgent(gpu_device_);
  }

  // Make sure the slot is free for usage
  while ((index - hsa_queue_load_read_index_scacquire(gpu_queue_)) >= queueMask) {
    amd::Os::yield();
  }

  // Add blocking command if the original value of read index was behind of the queue size
  if (blocking || (index - read) >= queueMask) {
    if (packet->completion_signal.handle == 0) {
      packet->completion_signal = barrier_signal_;
    }
    signal = packet->completion_signal;
    // Initialize signal for a wait
    hsa_signal_store_relaxed(signal, InitSignalValue);
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
  }
  //hsa_queue_store_write_index_release(gpu_queue_, index);
  hsa_signal_store_release(gpu_queue_->doorbell_signal, index - 1);

  // Wait on signal ?
  if (blocking) {
    if (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                HSA_WAIT_STATE_BLOCKED) != 0) {
      LogPrintfError("Failed signal [0x%lx] wait", signal.handle);
      return false;
    }
  }

  return true;
}

bool VirtualGPU::dispatchAqlPacket(
  hsa_kernel_dispatch_packet_t* packet, uint16_t header, uint16_t rest, bool blocking) {
  return dispatchGenericAqlPacket(packet, header, rest, blocking);
}

bool VirtualGPU::dispatchAqlPacket(
  hsa_barrier_and_packet_t* packet, uint16_t header, uint16_t rest, bool blocking) {
  return dispatchGenericAqlPacket(packet, header, rest, blocking);
}

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

void VirtualGPU::dispatchBarrierPacket(const hsa_barrier_and_packet_t* packet) {
  assert(packet->completion_signal.handle != 0);
  const uint32_t queueSize = gpu_queue_->size;
  const uint32_t queueMask = queueSize - 1;
  uint32_t header = kBarrierPacketHeader;

  uint64_t index = hsa_queue_add_write_index_screlease(gpu_queue_, 1);
  while ((index - hsa_queue_load_read_index_scacquire(gpu_queue_)) >= queueMask);
  hsa_barrier_and_packet_t* aql_loc =
    &(reinterpret_cast<hsa_barrier_and_packet_t*>(gpu_queue_->base_address))[index & queueMask];
  *aql_loc = *packet;
 __atomic_store_n(reinterpret_cast<uint32_t*>(aql_loc), kBarrierPacketHeader, __ATOMIC_RELEASE);

 hsa_signal_store_release(gpu_queue_->doorbell_signal, index);
}

/**
 * @brief Waits on an outstanding kernel without regard to how
 * it was dispatched - with or without a signal
 *
 * @return bool true if Wait returned successfully, false
 * otherwise
 */
bool VirtualGPU::releaseGpuMemoryFence() {
  // Return if there is no pending dispatch
  if (!hasPendingDispatch_) {
    return false;
  }

  // Initialize signal for the barrier packet.
  hsa_signal_store_relaxed(barrier_signal_, InitSignalValue);

  // Dispatch barrier packet into the queue and wait till it finishes.
  dispatchBarrierPacket(&barrier_packet_);
  if (hsa_signal_wait_acquire(barrier_signal_, HSA_SIGNAL_CONDITION_EQ, 0, uint64_t(-1),
                              HSA_WAIT_STATE_BLOCKED) != 0) {
    LogError("Barrier packet submission failed");
    return false;
  }

  hasPendingDispatch_ = false;

  // Release all transfer buffers on this command queue
  releaseXferWrite();

  // Release all memory dependencies
  memoryDependency().clear();

  // Release the pool, since runtime just completed a barrier
  resetKernArgPool();

  return true;
}

VirtualGPU::VirtualGPU(Device& device, bool profiling, bool cooperative)
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
      schedulerSignal_({0})
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
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    dispatchPacketHeader_=
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  } else {
    dispatchPacketHeaderNoSync_ =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    dispatchPacketHeader_=
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  }

  aqlHeader_ = dispatchPacketHeader_;
  barrier_signal_.handle = 0;

  // Note: Virtual GPU device creation must be a thread safe operation
  roc_device_.vgpus_.resize(roc_device_.numOfVgpus_);
  roc_device_.vgpus_[index()] = this;
}

VirtualGPU::~VirtualGPU() {
  delete blitMgr_;

  // Release the resources of signal
  releaseGpuMemoryFence();

  if (barrier_signal_.handle != 0) {
    hsa_signal_destroy(barrier_signal_);
  }

  destroyPool();

  releasePinnedMem();

  if (timestamp_ != nullptr) {
    delete timestamp_;
    timestamp_ = nullptr;
    LogError("There was a timestamp that was not used; deleting.");
  }
  if (printfdbg_ != nullptr) {
    delete printfdbg_;
    printfdbg_ = nullptr;
  }

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
    roc_device_.releaseQueue(gpu_queue_);
  }
}

bool VirtualGPU::create() {
  // Pick a reasonable queue size
  uint32_t queue_size = 1024;
  gpu_queue_ = roc_device_.acquireQueue(queue_size, cooperative_);
  if (!gpu_queue_) return false;

  if (!initPool(dev().settings().kernargPoolSize_, (profiling_) ? queue_size : 0)) {
    LogError("Couldn't allocate arguments/signals for the queue");
    return false;
  }

  device::BlitManager::Setup blitSetup;
  blitMgr_ = new KernelBlitManager(*this, blitSetup);
  if ((nullptr == blitMgr_) || !blitMgr_->create(roc_device_)) {
    LogError("Could not create BlitManager!");
    return false;
  }

  // Create signal for the barrier packet.
  hsa_signal_t signal = {0};
  if (HSA_STATUS_SUCCESS != hsa_signal_create(InitSignalValue, 0, nullptr, &signal)) {
    return false;
  }
  barrier_signal_ = signal;

  // Initialize barrier packet.
  memset(&barrier_packet_, 0, sizeof(barrier_packet_));
  barrier_packet_.header = kInvalidAql;
  barrier_packet_.completion_signal = barrier_signal_;

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

  return true;
}

bool VirtualGPU::initPool(size_t kernarg_pool_size, uint signal_pool_count) {
  kernarg_pool_size_ = kernarg_pool_size;
  kernarg_pool_base_ = reinterpret_cast<char*>(roc_device_.hostAlloc(kernarg_pool_size_, false));
  if (kernarg_pool_base_ == nullptr) {
    return false;
  }

  if (signal_pool_count != 0) {
    signal_pool_.resize(signal_pool_count);
    for (uint i = 0; i < signal_pool_count; ++i) {
      ProfilingSignal profilingSignal;
      if (HSA_STATUS_SUCCESS != hsa_signal_create(0, 0, nullptr, &profilingSignal.signal_)) {
        return false;
      }
      signal_pool_[i] = profilingSignal;
    }
  }

  return true;
}

void VirtualGPU::destroyPool() {
  if (kernarg_pool_base_ != nullptr) {
    roc_device_.hostFree(kernarg_pool_base_, kernarg_pool_size_);
  }

  if (signal_pool_.size() > 0) {
    for (uint i = 0; i < signal_pool_.size(); ++i) {
      hsa_signal_destroy(signal_pool_[i].signal_);
    }
  }
}

void* VirtualGPU::allocKernArg(size_t size, size_t alignment) {
  char* result = nullptr;
  do {
    result = amd::alignUp(kernarg_pool_base_ + kernarg_pool_cur_offset_, alignment);
    const size_t pool_new_usage = (result + size) - kernarg_pool_base_;
    if (pool_new_usage <= kernarg_pool_size_) {
      kernarg_pool_cur_offset_ = pool_new_usage;
      return result;
    } else {
      //! We run out of the arguments space!
      //! That means the app didn't call clFlush/clFinish for very long time.
      //! We can issue a barrier to avoid expensive extra memory allocations.

      // Initialize signal for the barrier packet.
      hsa_signal_store_relaxed(barrier_signal_, InitSignalValue);

      // Dispatch barrier packet into the queue and wait till it finishes.
      dispatchBarrierPacket(&barrier_packet_);
      if (hsa_signal_wait_acquire(barrier_signal_, HSA_SIGNAL_CONDITION_EQ, 0, uint64_t(-1),
                                  HSA_WAIT_STATE_BLOCKED) != 0) {
        LogError("Kernel arguments reset failed");
      }

      resetKernArgPool();
    }
  } while (true);

  return result;
}

/* profilingBegin, when profiling is enabled, creates a timestamp to save in
* virtualgpu's timestamp_, and calls start() to get the current host
* timestamp.
*/
void VirtualGPU::profilingBegin(amd::Command& command, bool drmProfiling) {
  if (command.profilingInfo().enabled_) {
    if (timestamp_ != nullptr) {
      LogWarning(
          "Trying to create a second timestamp in VirtualGPU. \
                        This could have unintended consequences.");
      return;
    }
    timestamp_ = new Timestamp;
    timestamp_->start();
  }
}

/* profilingEnd, when profiling is enabled, checks to see if a signal was
* created for whatever command we are running and calls end() to get the
* current host timestamp if no signal is available. It then saves the pointer
* timestamp_ to the command's data.
*/
void VirtualGPU::profilingEnd(amd::Command& command) {
  if (command.profilingInfo().enabled_) {
    if (timestamp_->getProfilingSignal() == nullptr) {
      timestamp_->end();
    }
    command.setData(reinterpret_cast<void*>(timestamp_));
    timestamp_ = nullptr;
  }
}

void VirtualGPU::updateCommandsState(amd::Command* list) {
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
        startTimeStamp = ts->getStart();
        endTimeStamp = ts->getStart();
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
        startTimeStamp = ts->getStart();
        endTimeStamp = ts->getEnd();
        delete ts;
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
                                      cmd.isEntireMemory());
      } else {
        result = blitMgr().readBuffer(*devMem, dst, origin, size, cmd.isEntireMemory());
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
                                          size, cmd.isEntireMemory());
      } else {
        result = blitMgr().readBufferRect(*devMem, dst, cmd.bufRect(), cmd.hostRect(), size,
                                          cmd.isEntireMemory());
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
          if (nullptr == imageBuffer->owner()->getLastWriter()) {
            Memory* buffer = dev().getGpuMemory(imageBuffer->owner()->parent());
            amd::Image* image = imageBuffer->owner()->asImage();
            amd::Coord3D offs(0);
            // Copy memory from the original image buffer into the backing store image
            result = blitMgr().copyBufferToImage(*buffer, *devMem, offs,
                                                 offs, image->getRegion(), true,
                                                 image->getRowPitch(), image->getSlicePitch());
          }
        }
      }
      if (hostMemory != nullptr) {
        // Accelerated image to buffer transfer without pinning
        amd::Coord3D dstOrigin(offset);
        result =
            blitMgr().copyImageToBuffer(*devMem, *hostMemory, cmd.origin(), dstOrigin, size,
                                        cmd.isEntireMemory(), cmd.rowPitch(), cmd.slicePitch());
      } else {
        result = blitMgr().readImage(*devMem, dst, cmd.origin(), size, cmd.rowPitch(),
                                     cmd.slicePitch(), cmd.isEntireMemory());
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
                                      cmd.isEntireMemory());
      } else {
        result = blitMgr().writeBuffer(src, *devMem, origin, size, cmd.isEntireMemory());
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
                                          size, cmd.isEntireMemory());
      } else {
        result = blitMgr().writeBufferRect(src, *devMem, cmd.hostRect(), cmd.bufRect(), size,
                                          cmd.isEntireMemory());
      }
      break;
    }
    case CL_COMMAND_WRITE_IMAGE: {
      if (hostMemory != nullptr) {
        // Accelerated buffer to image transfer without pinning
        amd::Coord3D srcOrigin(offset);
        result =
            blitMgr().copyBufferToImage(*hostMemory, *devMem, srcOrigin, cmd.origin(), size,
                                        cmd.isEntireMemory(), cmd.rowPitch(), cmd.slicePitch());
      } else {
        result = blitMgr().writeImage(src, *devMem, cmd.origin(), size, cmd.rowPitch(),
                                      cmd.slicePitch(), cmd.isEntireMemory());
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

bool VirtualGPU::copyMemory(cl_command_type type, amd::Memory& srcMem, amd::Memory& dstMem,
                            bool entire, const amd::Coord3D& srcOrigin,
                            const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                            const amd::BufferRect& srcRect, const amd::BufferRect& dstRect) {
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

      result = blitMgr().copyBuffer(*srcDevMem, *dstDevMem, realSrcOrigin, realDstOrigin, realSize, entire);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_RECT: {
      result = blitMgr().copyBufferRect(*srcDevMem, *dstDevMem, srcRect, dstRect, size, entire);
      break;
    }
    case CL_COMMAND_COPY_IMAGE: {
      result = blitMgr().copyImage(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size, entire);
      break;
    }
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
      result = blitMgr().copyImageToBuffer(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size, entire);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
      result = blitMgr().copyBufferToImage(*srcDevMem, *dstDevMem, srcOrigin, dstOrigin, size, entire);
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

void VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  cl_command_type type = cmd.type();
  bool entire = cmd.isEntireMemory();

  if (!copyMemory(type, cmd.source(), cmd.destination(), entire, cmd.srcOrigin(),
                  cmd.dstOrigin(), cmd.size(), cmd.srcRect(), cmd.dstRect())) {
    cmd.setStatus(CL_INVALID_OPERATION);
  }

  profilingEnd(cmd);
}

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
        dev().forceFineGrain(srcMem) ||
        dev().forceFineGrain(dstMem)) {
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
    // direct memcpy for FGS enabled system
    amd::SvmBuffer::memFill(cmd.dst(), cmd.src(), cmd.srcSize(), 1);
  }
  profilingEnd(cmd);
}

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
    // Target is the backing store, so just ensure that owner is up-to-date
    devMemory->owner()->cacheWriteBack();

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

bool VirtualGPU::fillMemory(cl_command_type type, amd::Memory* amdMemory, const void* pattern,
                            size_t patternSize, const amd::Coord3D& origin,
                            const amd::Coord3D& size) {
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
      amd::Coord3D realOrigin(origin[0]);
      amd::Coord3D realSize(size[0]);
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
      result = blitMgr().fillBuffer(*memory, pattern, patternSize, realOrigin, realSize, entire);
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

void VirtualGPU::submitFillMemory(amd::FillMemoryCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(cmd);

  if (!fillMemory(cmd.type(), &cmd.memory(), cmd.pattern(), cmd.patternSize(), cmd.origin(),
                  cmd.size())) {
    cmd.setStatus(CL_INVALID_OPERATION);
  }
  profilingEnd(cmd);
}

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
    // Synchronize memory from host if necessary
    device::Memory::SyncFlags syncFlags;
    syncFlags.skipEntire_ = dstMemory->isEntirelyCovered(origin, size);
    memory->syncCacheFromHost(*this, syncFlags);

    if (!fillMemory(cmd.type(), dstMemory, cmd.pattern(), cmd.patternSize(), origin, size)) {
      cmd.setStatus(CL_INVALID_OPERATION);
    }
    // Mark this as the most-recently written cache of the destination
    dstMemory->signalWrite(&dev());
  } else {
    // for FGS capable device, fill CPU memory directly
    amd::SvmBuffer::memFill(cmd.dst(), cmd.pattern(), cmd.patternSize(), cmd.times());
  }

  profilingEnd(cmd);
}

void VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);

  for (auto itr : vcmd.memObjects()) {
    // Find device memory
    Memory* memory = dev().getRocMemory(&(*itr));

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

static void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data) {
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    // Abort on device exceptions.
    abort();
  }
}

bool VirtualGPU::createSchedulerParam()
{
  if (nullptr != schedulerParam_) {
    return true;
  }

  while(true) {
    schedulerParam_ = new (dev().context()) amd::Buffer(dev().context(), CL_MEM_ALLOC_HOST_PTR, sizeof(SchedulerParam) + sizeof(AmdAqlWrap));

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

uint64_t VirtualGPU::getVQVirtualAddress()
{
  Memory* vqMem = dev().getRocMemory(virtualQueue_);
  return reinterpret_cast<uint64_t>(vqMem->getDeviceMemory());
}

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
  amd::Coord3D region(virtualQueue_->getSize(), 0, 0);

  if (!dev().xferMgr().fillBuffer(*vqMem, &pattern, sizeof(pattern), origin, region)) {
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

bool VirtualGPU::submitKernelInternal(const amd::NDRangeContainer& sizes, const amd::Kernel& kernel,
  const_address parameters, void* eventHandle, uint32_t sharedMemBytes, amd::NDRangeKernelCommand* vcmd) {
  device::Kernel* devKernel = const_cast<device::Kernel*>(kernel.getDeviceKernel(dev()));
  Kernel& gpuKernel = static_cast<Kernel&>(*devKernel);
  size_t ldsUsage = gpuKernel.WorkgroupGroupSegmentByteSize();

  // Check memory dependency and SVM objects
  bool coopGroups = (vcmd != nullptr) ? vcmd->cooperativeGroups() : false;
  if (!processMemObjects(kernel, parameters, ldsUsage, coopGroups)) {
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
        if (timestamp_ != nullptr) {
          timestamp_->setSplittedDispatch();
        }
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

    // Find all parameters for the current kernel

    // Allocate buffer to hold kernel arguments
    address argBuffer = (address)allocKernArg(gpuKernel.KernargSegmentByteSize(),
                                              gpuKernel.KernargSegmentAlignment());

    if (argBuffer == nullptr) {
      LogError("Out of memory");
      return false;
    }

    ClPrint(amd::LOG_INFO, amd::LOG_KERN, "[%zx]!\tShaderName : %s\n", std::this_thread::get_id(), gpuKernel.name().c_str());

    // Check if runtime has to setup hidden arguments
    for (uint32_t i = signature.numParameters(); i < signature.numParametersAll(); ++i) {
      const auto it = signature.at(i);
      size_t offset;
      switch (it.info_.oclObject_) {
        case amd::KernelParameterDescriptor::HiddenNone:
          break;
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetX: {
          offset = newOffset[0];
          assert(it.size_ == sizeof(offset) && "check the sizes");
          WriteAqlArgAt(const_cast<address>(parameters), &offset, it.size_, it.offset_);
          break;
        }
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetY: {
          if (sizes.dimensions() >= 2) {
            offset = newOffset[1];
            assert(it.size_ == sizeof(offset) && "check the sizes");
            WriteAqlArgAt(const_cast<address>(parameters), &offset, it.size_, it.offset_);
          }
          break;
        }
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetZ: {
          if (sizes.dimensions() >= 3) {
            offset = newOffset[2];
            assert(it.size_ == sizeof(offset) && "check the sizes");
            WriteAqlArgAt(const_cast<address>(parameters), &offset, it.size_, it.offset_);
          }
          break;
        }
        case amd::KernelParameterDescriptor::HiddenPrintfBuffer: {
          address bufferPtr = printfDbg()->dbgBuffer();
          if (printfEnabled &&
            // and printf buffer was allocated
            (bufferPtr != nullptr)) {
            assert(it.size_ == sizeof(bufferPtr) && "check the sizes");
            WriteAqlArgAt(const_cast<address>(parameters), &bufferPtr, it.size_, it.offset_);
          }
          break;
        }
        case amd::KernelParameterDescriptor::HiddenHostcallBuffer: {
          if (amd::IS_HIP) {
            auto buffer = roc_device_.getOrCreateHostcallBuffer(gpu_queue_);
            if (!buffer) {
              ClPrint(amd::LOG_ERROR, amd::LOG_KERN,
                      "Kernel expects a hostcall buffer, but none found");
              return false;
            }
            assert(it.size_ == sizeof(buffer) && "check the sizes");
            WriteAqlArgAt(const_cast<address>(parameters), &buffer, it.size_, it.offset_);
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
          WriteAqlArgAt(const_cast<address>(parameters), &vqVA, it.size_, it.offset_);
          break;
        }
        case amd::KernelParameterDescriptor::HiddenCompletionAction: {
          uint64_t spVA = 0;
          if (nullptr != schedulerParam_) {
            Memory* schedulerMem = dev().getRocMemory(schedulerParam_);
            AmdAqlWrap* wrap = reinterpret_cast<AmdAqlWrap*>(reinterpret_cast<uint64_t>(schedulerParam_->getHostMem()) + sizeof(SchedulerParam));
            memset(wrap, 0, sizeof(AmdAqlWrap));
            wrap->state = AQL_WRAP_DONE;

            spVA = reinterpret_cast<uint64_t>(schedulerMem->getDeviceMemory()) + sizeof(SchedulerParam);
          }
          WriteAqlArgAt(const_cast<address>(parameters), &spVA, it.size_, it.offset_);
          break;
        }
        case amd::KernelParameterDescriptor::HiddenMultiGridSync: {
          uint64_t gridSync = coopGroups ? 1 : 0;
          bool multiGrid = (vcmd != nullptr) ? vcmd->cooperativeMultiDeviceGroups() : false;
          if (multiGrid) {
            // Find CPU pointer to the right sync info structure. It should be after MGSyncData
            Device::MGSyncInfo* syncInfo = reinterpret_cast<Device::MGSyncInfo*>(
                dev().MGSync() + Device::kMGInfoSizePerDevice * dev().index() + Device::kMGSyncDataSize);
            // Update sync data address. Use the offset adjustment to the right location
            syncInfo->mgs = reinterpret_cast<Device::MGSyncData*>(dev().MGSync() +
              Device::kMGInfoSizePerDevice * vcmd->firstDevice());
            // Fill all sync info fields
            syncInfo->grid_id = vcmd->gridId();
            syncInfo->num_grids = vcmd->numGrids();
            syncInfo->prev_sum = vcmd->prevGridSum();
            syncInfo->all_sum = vcmd->allGridSum();
            // Update GPU address for grid sync info. Use the offset adjustment for the right location
            gridSync = reinterpret_cast<uint64_t>(syncInfo);
          }
          WriteAqlArgAt(const_cast<address>(parameters), &gridSync, it.size_, it.offset_);
          break;
        }
      }
    }

    // Load all kernel arguments
    WriteAqlArgAt(argBuffer, parameters, gpuKernel.KernargSegmentByteSize(), 0);
    // Note: In a case of structs the size won't match,
    // since HSAIL compiler expects a reference...
    assert(gpuKernel.KernargSegmentByteSize() <= signature.paramsSize() &&
      "A mismatch of sizes of arguments between compiler and runtime!");

    // Check for group memory overflow
    //! @todo Check should be in HSA - here we should have at most an assert
    assert(roc_device_.info().localMemSizePerCU_ > 0);
    if (ldsUsage > roc_device_.info().localMemSizePerCU_) {
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

    amd::NDRange local(sizes.local());
    devKernel->FindLocalWorkSize(sizes.dimensions(), sizes.global(), local);
    dispatchPacket.workgroup_size_x = sizes.dimensions() > 0 ? local[0] : 1;
    dispatchPacket.workgroup_size_y = sizes.dimensions() > 1 ? local[1] : 1;
    dispatchPacket.workgroup_size_z = sizes.dimensions() > 2 ? local[2] : 1;

    dispatchPacket.kernarg_address = argBuffer;
    dispatchPacket.group_segment_size = ldsUsage + sharedMemBytes;
    dispatchPacket.private_segment_size = devKernel->workGroupInfo()->privateMemSize_;

    // Dispatch the packet
    if (!dispatchAqlPacket(
            &dispatchPacket, aqlHeader_,
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
    dispatchBarrierPacket(&barrier_packet_);
    static_cast<KernelBlitManager&>(blitMgr()).runScheduler(
      getVQVirtualAddress(), schedulerParam_, schedulerQueue_, schedulerSignal_, schedulerThreads_);
  }

  // Check if image buffer write back is required
  if (imageBufferWrtBack_) {
    // Avoid recursive write back
    imageBufferWrtBack_ = false;
    // Make sure the original kernel execution is done
    releaseGpuMemoryFence();
    for (const auto imageBuffer : wrtBackImageBuffer_) {
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
    wrtBackImageBuffer_.clear();
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
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd) {
  if (vcmd.cooperativeGroups() || vcmd.cooperativeMultiDeviceGroups()) {
    // Wait for the execution on the current queue, since the coop groups will use the device queue
    releaseGpuMemoryFence();

    // Get device queue for exclusive GPU access
    VirtualGPU* queue = dev().xferQueue();

    // Lock the queue, using the blit manager lock
    amd::ScopedLock lock(queue->blitMgr().lockXfer());

    queue->profilingBegin(vcmd);

    if (vcmd.cooperativeGroups()) {
      // Initialize GWS if it's cooperative groups launch
      uint32_t workgroups = 0;
      for (uint i = 0; i < vcmd.sizes().dimensions(); i++) {
        if ((vcmd.sizes().local()[i] != 0) && (vcmd.sizes().global()[i] != 1)) {
          workgroups += (vcmd.sizes().global()[i] / vcmd.sizes().local()[i]);
        }
      }

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
    queue->releaseGpuMemoryFence();

    queue->profilingEnd(vcmd);
  } else {
  // Make sure VirtualGPU has an exclusive access to the resources
    amd::ScopedLock lock(execution());

    profilingBegin(vcmd);

    // Submit kernel to HW
    if (!submitKernelInternal(vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
      static_cast<void*>(as_cl(&vcmd.event())), vcmd.sharedMemBytes())) {
      LogError("AQL dispatch failed!");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
  }
}

void VirtualGPU::submitNativeFn(amd::NativeFnCommand& cmd) {
  // std::cout<<__FUNCTION__<<" not implemented"<<"*********"<<std::endl;
}

void VirtualGPU::submitMarker(amd::Marker& cmd) {
  // std::cout<<__FUNCTION__<<" not implemented"<<"*********"<<std::endl;
}

void VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  profilingBegin(vcmd);
  auto fence = kBarrierAcquirePacket;
  dispatchAqlPacket(&fence, 0, 0, false);
  profilingEnd(vcmd);
}

void VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());
  profilingBegin(vcmd);
  auto fence = kBarrierReleasePacket;
  dispatchAqlPacket(&fence, 0, 0, false);
  profilingEnd(vcmd);
}

void VirtualGPU::flush(amd::Command* list, bool wait) {
  releaseGpuMemoryFence();
  updateCommandsState(list);
  // Release all pinned memory
  releasePinnedMem();
}

void VirtualGPU::addXferWrite(Memory& memory) {
  if (xferWriteBuffers_.size() > 7) {
    dev().xferWrite().release(*this, *xferWriteBuffers_.front());
    xferWriteBuffers_.erase(xferWriteBuffers_.begin());
  }

  // Delay destruction
  xferWriteBuffers_.push_back(&memory);
}

void VirtualGPU::releaseXferWrite() {
  for (auto& memory : xferWriteBuffers_) {
    dev().xferWrite().release(*this, *memory);
  }
  xferWriteBuffers_.resize(0);
}

void VirtualGPU::addPinnedMem(amd::Memory* mem) {
  if (nullptr == findPinnedMem(mem->getHostMem(), mem->getSize())) {
    if (pinnedMems_.size() > 7) {
      pinnedMems_.front()->release();
      pinnedMems_.erase(pinnedMems_.begin());
    }

    // Delay destruction
    pinnedMems_.push_back(mem);
  }
}

void VirtualGPU::releasePinnedMem() {
  for (auto& amdMemory : pinnedMems_) {
    amdMemory->release();
  }
  pinnedMems_.resize(0);
}

amd::Memory* VirtualGPU::findPinnedMem(void* addr, size_t size) {
  for (auto& amdMemory : pinnedMems_) {
    if ((amdMemory->getHostMem() == addr) && (size <= amdMemory->getSize())) {
      return amdMemory;
    }
  }
  return nullptr;
}

void VirtualGPU::enableSyncBlit() const { blitMgr_->enableSynchronization(); }

void VirtualGPU::submitTransferBufferFromFile(amd::TransferBufferFileCommand& cmd) {
  // Make sure VirtualGPU has an exclusive access to the resources
  amd::ScopedLock lock(execution());

  size_t copySize = cmd.size()[0];
  size_t fileOffset = cmd.fileOffset();
  Memory* mem = dev().getRocMemory(&cmd.memory());
  uint idx = 0;

  assert((cmd.type() == CL_COMMAND_READ_SSG_FILE_AMD) ||
         (cmd.type() == CL_COMMAND_WRITE_SSG_FILE_AMD));
  const bool writeBuffer(cmd.type() == CL_COMMAND_READ_SSG_FILE_AMD);

  if (writeBuffer) {
    size_t dstOffset = cmd.origin()[0];
    while (copySize > 0) {
      Memory* staging = dev().getRocMemory(&cmd.staging(idx));
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
      fileOffset += dstSize;
      dstOffset += dstSize;
      copySize -= dstSize;
    }
  } else {
    size_t srcOffset = cmd.origin()[0];
    while (copySize > 0) {
      Memory* staging = dev().getRocMemory(&cmd.staging(idx));
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
    }

    // create the AQL packet for start profiling
    if (profileRef->createStartPacket() == nullptr) {
      LogError("Failed to create AQL packet for start profiling");
      vcmd.setStatus(CL_INVALID_OPERATION);
    }

    dispatchCounterAqlPacket(profileRef->prePacket(), counter->gfxVersion(), false,
                             profileRef->api());

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
