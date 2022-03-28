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

#pragma once

#include <hip/hip_runtime.h>
#include "hip_event.hpp"
#include "hip_internal.hpp"
#include <unordered_map>
#include <unordered_set>

namespace hip {

class Device;
class Stream;

struct MemoryTimestamp {
  MemoryTimestamp(hip::Stream* stream, hip::Event* event = nullptr): event_(event) {
    safe_streams_.insert(stream);
  }
  MemoryTimestamp(): event_(nullptr) {}

  /// Adds a safe stream to the list of stream for possible reuse
  void AddSafeStream(hip::Stream* stream) {
    if (safe_streams_.find(stream) != safe_streams_.end()) {
      safe_streams_.insert(stream);
    }
  }
  /// Changes last known valid event asociated with memory
  void SetEvent(hip::Event* event) {
    delete event_;
    event_ = event;
  }
  /// Wait for memory to be available
  void Wait() {
    if (event_ != nullptr) {
      auto hip_error = event_->synchronize();
    }
  }
  /// Returns if memory object is safe for reuse
  bool IsSafeFind(hip::Stream* stream = nullptr, bool opportunistic = true) {
    bool result = false;
    if (safe_streams_.find(stream) != safe_streams_.end()) {
      // A safe stream doesn't require TS validation
      result = true;
    } else if (opportunistic && (event_ != nullptr)) {
      // Check HIP event for a retired status
      result = (event_->query() == hipSuccess) ? true : false;
    }
    return result;
  }
  /// Returns if memory object is safe for reuse
  bool IsSafeRelease() {
    bool result = true;
    if (event_ != nullptr) {
      // Check HIP event for a retired status
      result = (event_->query() == hipSuccess) ? true : false;
    }
    return result;
  }

  std::unordered_set<hip::Stream*>  safe_streams_;  //!< Safe streams for memory reuse
  hip::Event*   event_;   //!< Last known HIP event, associated with the memory object
};

class Heap : public amd::EmbeddedObject {
public:
  Heap(hip::Device* device):
    total_size_(0), max_total_size_(0), release_threshold_(0), device_(device) {}
  ~Heap() {}

  /// Adds allocation into the heap on a specific stream
  void AddMemory(amd::Memory* memory, hip::Stream* stream);

  /// Adds allocation into the heap with specific TS
  void AddMemory(amd::Memory* memory, const MemoryTimestamp& ts);

  /// Finds memory object with the specified size
  amd::Memory* FindMemory(size_t size, hip::Stream* stream, bool opportunistic);

  /// Removes allocation from the map
  bool RemoveMemory(amd::Memory* memory, MemoryTimestamp* ts = nullptr);

  /// Releases all memory, until the threshold value is met
  bool ReleaseAllMemory(size_t min_bytes_to_hold = std::numeric_limits<size_t>::max(), bool safe_release = false);

  /// Releases all memory, safe to the provided stream, until the threshold value is met
  bool ReleaseAllMemory(hip::Stream* stream);

  /// Heap doesn't have any allocations
  bool IsEmpty() const { return (allocations_.size() == 0) ? true : false; }

  /// Set the memory release threshold
  void SetReleaseThreshold(uint64_t value) { release_threshold_ = value; }

  /// Set the memory release threshold
  uint64_t GetReleaseThreshold() const { return release_threshold_; }

  /// Get the size of all allocations in the heap
  uint64_t GetTotalSize() const { return total_size_; }

  /// Get the size of all allocations in the heap
  uint64_t GetMaxTotalSize() const { return max_total_size_; }

  /// Set maximum total, allocated by the heap
  void SetMaxTotalSize(uint64_t value) { max_total_size_ = value; }

  std::unordered_map<amd::Memory*, MemoryTimestamp>::iterator EraseAllocaton(
    std::unordered_map<amd::Memory*, MemoryTimestamp>::iterator& it);

private:
  Heap() = delete;
  Heap(const Heap&) = delete;
  Heap& operator=(const Heap&) = delete;

  std::unordered_map<amd::Memory*, MemoryTimestamp> allocations_;   //!< Map of allocations on a specific stream
  uint64_t total_size_;         //!< Size of all allocations in the heap
  uint64_t max_total_size_;     //!< Maximum heap allocation size
  uint64_t release_threshold_;  //!< Threshold size in bytes for memory release from heap, default 0

  hip::Device*  device_;    //!< Hip device the allocations will reside
};

/// Allocates memory in the pool on the specified stream and places the allocation into busy_heap_
/// @note: the logic also will look in free_heap for possible reuse.
/// hipMemPoolReuseAllowOpportunistic option will validate if HIP event,
/// associated with memory is done, then reuse can be performed.
class MemoryPool : public amd::ReferenceCountedObject {
public:
  MemoryPool(hip::Device* device):
    busy_heap_(device),
    free_heap_(device),
    lock_pool_ops_("Pool operations", true), device_(device) {
      device_->AddMemoryPool(this);
      state_.event_dependencies_ = 1;
      state_.opportunistic_ = 1;
      state_.internal_dependencies_ = 1;
    }
  virtual ~MemoryPool() {
    assert(busy_heap_.IsEmpty() && "Can't destroy pool with busy allocations!");
    constexpr bool kSafeRelease = true;
    free_heap_.ReleaseAllMemory(0, kSafeRelease);
    // Remove memory pool from the list of all pool on the current device
    device_->RemoveMemoryPool(this);
  }

  /// The same stream can reuse memory without HIP event validation
  void* AllocateMemory(size_t size, hip::Stream* stream);

  /// Frees memory by placing memory object with HIP event into free_heap_
  bool FreeMemory(amd::Memory* memory, hip::Stream* stream);

  /// Releases all allocations from free_heap_. It can be called on Stream or Device synchronization
  /// @note The caller must make sure it's safe to release memory
  void ReleaseFreedMemory(hip::Stream* stream = nullptr);

  /// Releases all allocations in MemoryPool
  void ReleaseAllMemory();

  /// Trims the pool until it has only min_bytes_to_hold
  void TrimTo(size_t min_bytes_to_hold);

  /// Trims the pool until it has only min_bytes_to_hold
  hip::Device* Device() const { return device_; }

  /// Set memory pool control attributes
  hipError_t SetAttribute(hipMemPoolAttr attr, void* value);

  /// Get memory pool control attributes
  hipError_t GetAttribute(hipMemPoolAttr attr, void* value);

  /// Accessors for the pool state
  bool EventDependencies() const { return (state_.event_dependencies_) ? true : false; }
  bool Opportunistic() const { return (state_.opportunistic_) ? true : false; }
  bool InternalDependencies() const { return (state_.internal_dependencies_) ? true : false; }

private:
  MemoryPool() = delete;
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;


  Heap busy_heap_;    //!< Heap of busy allocations
  Heap free_heap_;    //!< Heap of freed allocations
  struct {
    uint32_t event_dependencies_ : 1;     //!< Event dependencies tracking is enabled
    uint32_t opportunistic_ : 1;          //!< HIP event check is enabled
    uint32_t internal_dependencies_ : 1;  //!< Runtime adds internal events to handle memory dependencies
  } state_;

  amd::Monitor  lock_pool_ops_;  //!< Access to the pool must be lock protected
  hip::Device*  device_;    //!< Hip device the heap will reside
};

} // Mamespace hip
