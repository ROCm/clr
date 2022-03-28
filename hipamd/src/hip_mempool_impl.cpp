/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include "hip_mempool_impl.hpp"

namespace hip {

// ================================================================================================
void Heap::AddMemory(amd::Memory* memory, hip::Stream* stream) {
  allocations_.insert({memory, {stream, nullptr}});
  total_size_ += memory->getSize();
  max_total_size_ = std::max(max_total_size_, total_size_);
}

// ================================================================================================
void Heap::AddMemory(amd::Memory* memory, const MemoryTimestamp& ts) {
  allocations_.insert({memory, ts});
  total_size_ += memory->getSize();
  max_total_size_ = std::max(max_total_size_, total_size_);
}

// ================================================================================================
amd::Memory* Heap::FindMemory(size_t size, hip::Stream* stream, bool opportunistic) {
  amd::Memory* memory = nullptr;
  for (auto it = allocations_.begin(); it != allocations_.end();) {
    // Check if size can match and it's safe to use this resource
    if ((it->first->getSize() >= size) && (it->second.IsSafeFind(stream, opportunistic))) {
      memory = it->first;
      total_size_ -= memory->getSize();
      // Remove found allocation from the map
      it = allocations_.erase(it);
      break;
    } else {
      ++it;
    }
  }
  return memory;
}

// ================================================================================================
bool Heap::RemoveMemory(amd::Memory* memory, MemoryTimestamp* ts) {
  if (auto it = allocations_.find(memory); it != allocations_.end()) {
    if (ts != nullptr) {
      // Preserve timestamp info for possible reuse later
      *ts = it->second;
    } else {
      // Runtime will delete the timestamp object, hence make sure HIP event is released
      it->second.Wait();
      it->second.SetEvent(nullptr);
    }
    total_size_ -= memory->getSize();
    allocations_.erase(it);
    return true;
  }
  return false;
}

// ================================================================================================
std::unordered_map<amd::Memory*, MemoryTimestamp>::iterator
Heap::EraseAllocaton(std::unordered_map<amd::Memory*, MemoryTimestamp>::iterator& it) {
  const device::Memory* dev_mem = it->first->getDeviceMemory(*device_->devices()[0]);
  amd::SvmBuffer::free(it->first->getContext(), reinterpret_cast<void*>(dev_mem->virtualAddress()));
  total_size_ -= it->first->getSize();
  // Clear HIP event
  it->second.SetEvent(nullptr);
  // Remove the allocation from the map
  return allocations_.erase(it);
}

// ================================================================================================
bool Heap::ReleaseAllMemory(size_t min_bytes_to_hold, bool safe_release) {
  for (auto it = allocations_.begin(); it != allocations_.end();) {
    // Make sure the heap is smaller than the minimum value to hold
    if (total_size_ <= min_bytes_to_hold) {
      return true;
    }
    // Safe release forces unconditional wait for memory
    if (safe_release) {
      it->second.Wait();
    }
    if (it->second.IsSafeRelease()) {
      it = EraseAllocaton(it);
    } else {
      ++it;
    }
  }
  return true;
}

// ================================================================================================
bool Heap::ReleaseAllMemory(hip::Stream* stream) {
  for (auto it = allocations_.begin(); it != allocations_.end();) {
    // Make sure the heap holds the minimum number of bytes
    if (total_size_ <= release_threshold_) {
      return true;
    }
    if (it->second.IsSafeRelease()) {
      it = EraseAllocaton(it);
    } else {
      ++it;
    }
  }
  return true;
}

// ================================================================================================
void* MemoryPool::AllocateMemory(size_t size, hip::Stream* stream) {
  amd::ScopedLock lock(lock_pool_ops_);

  void* dev_ptr = nullptr;
  amd::Memory* memory = free_heap_.FindMemory(size, stream, Opportunistic());
  if (memory == nullptr) {
    amd::Context* context = device_->asContext();
    const auto& dev_info = context->devices()[0]->info();
    if (dev_info.maxMemAllocSize_ < size) {
      return nullptr;
    }

    dev_ptr = amd::SvmBuffer::malloc(*context, 0, size, dev_info.memBaseAddrAlign_, nullptr);
    if (dev_ptr == nullptr) {
      size_t free = 0, total =0;
      hipError_t err = hipMemGetInfo(&free, &total);
      if (err == hipSuccess) {
        LogPrintfError("Allocation failed : Device memory : required :%zu | free :%zu | total :%zu \n",
          size, free, total);
      }
      return nullptr;
    }

    size_t offset = 0;
    memory = getMemoryObject(dev_ptr, offset);
    // Saves the current device id so that it can be accessed later
    memory->getUserData().deviceId = device_->deviceId();
  } else {
    free_heap_.RemoveMemory(memory);
    const device::Memory* dev_mem = memory->getDeviceMemory(*device_->devices()[0]);
    dev_ptr = reinterpret_cast<void*>(dev_mem->virtualAddress());
  }
  // Place the allocated memory into the busy heap
  busy_heap_.AddMemory(memory, stream);

  // Increment the reference counter on the pool
  retain();

  return dev_ptr;
}

// ================================================================================================
bool MemoryPool::FreeMemory(amd::Memory* memory, hip::Stream* stream) {
  amd::ScopedLock lock(lock_pool_ops_);

  MemoryTimestamp ts;
  // Remove memory object fro the busy pool
  if (!busy_heap_.RemoveMemory(memory, &ts)) {
    // This pool doesn't contain memory
    return false;
  }
  // The stream of destruction is a safe stream, because the app must handle sync
  ts.AddSafeStream(stream);

  // Add a marker to the stream to trace availability of this memory
  Event* e = new hip::Event(0);
  if (e != nullptr) {
    if (hipSuccess == e->addMarker(reinterpret_cast<hipStream_t>(stream), nullptr, true)) {
      ts.SetEvent(e);
    }
  }
  free_heap_.AddMemory(memory, ts);

  // Decrement the reference counter on the pool
  release();

  return true;
}

// ================================================================================================
void MemoryPool::ReleaseFreedMemory(hip::Stream* stream) {
  amd::ScopedLock lock(lock_pool_ops_);

  free_heap_.ReleaseAllMemory(stream);
}

// ================================================================================================
void MemoryPool::TrimTo(size_t min_bytes_to_hold) {
  amd::ScopedLock lock(lock_pool_ops_);

  free_heap_.ReleaseAllMemory(min_bytes_to_hold);
}

// ================================================================================================
hipError_t MemoryPool::SetAttribute(hipMemPoolAttr attr, void* value) {
  amd::ScopedLock lock(lock_pool_ops_);
  uint64_t reset;

  switch (attr) {
    case hipMemPoolReuseFollowEventDependencies:
      // Enable/disable HIP events tracking from the app's dependencies
      state_.event_dependencies_ = *reinterpret_cast<int32_t*>(value);
      break;
    case hipMemPoolReuseAllowOpportunistic:
      // Enable/disable HIP event check for freed memory
      state_.opportunistic_ = *reinterpret_cast<int32_t*>(value);
      break;
    case  hipMemPoolReuseAllowInternalDependencies:
      // Enable/disable internal extra dependencies introduced in runtime
      state_.internal_dependencies_ = *reinterpret_cast<int32_t*>(value);
      break;
    case hipMemPoolAttrReleaseThreshold:
      free_heap_.SetReleaseThreshold(*reinterpret_cast<uint64_t*>(value));
      break;
    case hipMemPoolAttrReservedMemCurrent:
      // Should be GetAttribute only
      return hipErrorInvalidValue;
      break;
    case hipMemPoolAttrReservedMemHigh:
      reset = *reinterpret_cast<uint64_t*>(value);
      // Only 0 is accepted
      if (reset != 0) {
        return hipErrorInvalidValue;
      }
      free_heap_.SetMaxTotalSize(reset);
    case hipMemPoolAttrUsedMemCurrent:
      // Should be GetAttribute only
      return hipErrorInvalidValue;
      break;
    case hipMemPoolAttrUsedMemHigh:
      reset = *reinterpret_cast<uint64_t*>(value);
      // Only 0 is accepted
      if (reset != 0) {
        return hipErrorInvalidValue;
      }
      busy_heap_.SetMaxTotalSize(reset);
      break;
    default:
      return hipErrorInvalidValue;
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t MemoryPool::GetAttribute(hipMemPoolAttr attr, void* value) {
  amd::ScopedLock lock(lock_pool_ops_);

  switch (attr) {
    case hipMemPoolReuseFollowEventDependencies:
      // Enable/disable HIP events tracking from the app's dependencies
      *reinterpret_cast<int32_t*>(value) = EventDependencies();
      break;
    case hipMemPoolReuseAllowOpportunistic:
      // Enable/disable HIP event check for freed memory
      *reinterpret_cast<int32_t*>(value) = Opportunistic();
      break;
    case  hipMemPoolReuseAllowInternalDependencies:
      // Enable/disable internal extra dependencies introduced in runtime
      *reinterpret_cast<int32_t*>(value) = InternalDependencies();
      break;
    case hipMemPoolAttrReleaseThreshold:
      *reinterpret_cast<uint64_t*>(value) = free_heap_.GetReleaseThreshold();
      break;
    case hipMemPoolAttrReservedMemCurrent:
      // All allocate memory by the pool in OS
      *reinterpret_cast<uint64_t*>(value) = busy_heap_.GetTotalSize() + free_heap_.GetTotalSize();
      break;
    case hipMemPoolAttrReservedMemHigh:
      // High watermark of all allocated memory in OS, since the last reset
      *reinterpret_cast<uint64_t*>(value) = busy_heap_.GetTotalSize() + free_heap_.GetMaxTotalSize();
      break;
    case hipMemPoolAttrUsedMemCurrent:
      // Total currently used memory by the pool
      *reinterpret_cast<uint64_t*>(value) = busy_heap_.GetTotalSize();
      break;
    case hipMemPoolAttrUsedMemHigh:
      // High watermark of all used memoryS, since the last reset
      *reinterpret_cast<uint64_t*>(value) = busy_heap_.GetMaxTotalSize();
      break;
    default:
      return hipErrorInvalidValue;
  }
  return hipSuccess;
}

}
