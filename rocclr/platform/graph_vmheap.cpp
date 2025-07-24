/* Copyright (c) 2025 Advanced Micro Devices, Inc.

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

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include "graph_vmheap.hpp"
#include "command.hpp"

namespace amd {

// ================================================================================================
address GraphVmHeap::ReserveAddressRange(address start, size_t size) {
  // Reserve a virtual address range on the device
  void* ptr = device_->virtualAlloc(start, size, 4096);
  // Save base memory object to accelerate access in the future
  base_memory_ = MemObjMap::FindVirtualMemObj(ptr);
  return reinterpret_cast<address>(ptr);
}

// ================================================================================================
bool GraphVmHeap::ReleaseAddressRange(void* addr) {
  Memory* memObj = MemObjMap::FindVirtualMemObj(addr);
  assert(memObj != nullptr && "Cannot find the Virtual MemObj entry");

  // Frees address range on the device
  device_->virtualFree(addr);
  memObj->release();

  return true;
}

size_t RoundUpPow2(size_t val) {
  if (val == 0) {
    return 1;
  }
  val--;
  val |= val >> 1;
  val |= val >> 2;
  val |= val >> 4;
  val |= val >> 8;
  val |= val >> 16;
  val |= val >> 32;
  val++;
  return val;
}

size_t GetPow2(size_t val) {
  return __builtin_ctzll(val);
}

// ================================================================================================
GraphVmHeap::GraphVmHeap(Device* device, size_t va_size, GetQueueFunc get_queue) :
   lock_(true)
  , device_(device)
  , get_vm_queue_(get_queue)
  , va_size_(RoundUpPow2(va_size)) {
  assert((double) va_size_ / va_size < 1.1 && "Rounding up GraphVmHeap size to a power of two would result in significant size increase");
  unmap_threshold_ = va_size_ / 2;
  free_size_ = va_size_;
}

// ================================================================================================
GraphVmHeap::~GraphVmHeap() {
  if (created_) {
    ScopedLock k(lock_);

    FreeAllMemory();

    // Destroy virtual address space
    if (base_address_ != nullptr) {
      ReleaseAddressRange(base_address_);
    }
  }
}

// ================================================================================================
bool GraphVmHeap::Create() {
  // Create a new GPU resource
  base_address_ = ReserveAddressRange(0, va_size_);
  if (base_address_ == nullptr) {
    return false;
  }
  free_size_ = va_size_;

  max_bin_idx_ = GetPow2(va_size_) + 1 - kMinPow2_;
  free_bins_ = std::vector<GraphSlab*>(max_bin_idx_, nullptr);
  busy_bins_ = std::vector<GraphSlab*>(max_bin_idx_, nullptr);
  cache_bins_ = std::vector<GraphSlab*>(max_bin_idx_, nullptr);

  auto initial_slab = new GraphSlab(this, nullptr, va_size_, 0, base_address_ + block_alignment_);
  PushBin(free_bins_, max_bin_idx_ - 1, initial_slab);

  // Ensures that NullStream exists before VmHeap destructor is called
  GetVmQueue();

  return true;
}

// ================================================================================================
void GraphVmHeap::TrimPhysMemory(size_t unmap_threshold) {
  if (!created_) {
    return;
  }

  ScopedLock k(lock_);
  auto unmap_org = unmap_threshold_;
  unmap_threshold_ = unmap_threshold;

  for (size_t i = 0; i < max_bin_idx_; ++i) {
    auto current = free_bins_[i];
    bool done = false;
    while (current != nullptr) {
      auto busy_size = va_size_ - free_size_;
      uint64_t free_mapped = mapped_size_.load() - busy_size;
      if (free_mapped <= unmap_threshold_) {
	done = true;
	break;
      }

      if (current->mapped_.load()) {
	auto cmd = GetSlabUnmapCommand(current, GetVmQueue());
	cmd->enqueue();
	cmd->awaitCompletion();
	cmd->release();
      }
      current = current->next_;
    }
    if (done) {
      break;
    }
  }

  unmap_threshold_ = unmap_org;
}

// ================================================================================================
void GraphVmHeap::FreeAllMemory() {
  ScopedLock k(lock_);

  for (size_t i = 0; i < max_bin_idx_; ++i)  {
    while (cache_bins_[i]) {
      auto slab = PopBin(cache_bins_, i);
      UnmapSlab(slab);
      ReturnSlab(slab, false);
    }
    while (busy_bins_[i]) {
      UnmapSlab(busy_bins_[i]);
      ReturnSlab(busy_bins_[i]);
    }
  }
}

// ================================================================================================
Command* GraphVmHeap::GetSlabMapCommand(GraphSlab* slab, HostQueue& queue) {
  // Slab is already mapped - do nothing
  if (slab->mapped_.load()) {
    return nullptr;
  }

  const auto& dev_info = device_->info();
  size_t granularity = dev_info.virtualMemAllocGranularity_;
  auto padded_size = alignUp(slab->size_, granularity);
  auto addr = base_address_ + slab->offset_;
  return new CommitMemoryCommand(queue, Command::EventWaitList{}, addr, padded_size, nullptr, slab, this);
}

// ================================================================================================
Command* GraphVmHeap::GetSlabUnmapCommand(GraphSlab* slab, HostQueue& queue, bool unmap_guaranteed) {
  // Slab is already unmapped or it shouldn't be unmapped - do nothing
  if (!slab->mapped_.load() || (!unmap_guaranteed && !ShouldUnmap())) {
    return nullptr;
  }

  const auto& dev_info = device_->info();
  size_t granularity = dev_info.virtualMemAllocGranularity_;
  auto padded_size = alignUp(slab->size_, granularity);
  auto addr = base_address_ + slab->offset_;
  return new UncommitMemoryCommand(queue, Command::EventWaitList{}, addr, padded_size, nullptr, slab, this, unmap_guaranteed);
}

// ================================================================================================
Command* GraphVmHeap::GetAllocationCommand(GraphSlab* slab, HostQueue& queue, Command* wait_command, size_t size, size_t offset) {
  size_t full_offset = block_alignment_ + slab->offset_ + offset;
  auto ptr = (void*) ((size_t) slab->mem_ptr_ + offset);
  return new CreateMemoryCommand(queue, wait_command ? Command::EventWaitList{wait_command} : Command::EventWaitList{}, ptr, slab, base_memory_, size, full_offset);
}

void GraphVmHeap::UnmapSlab(GraphSlab* slab) {
  auto unmap_command = GetSlabUnmapCommand(slab, GetVmQueue(), true);
  if (!unmap_command) {
      return;
  }
  unmap_command->enqueue();
  unmap_command->awaitCompletion();
  unmap_command->release();
}

// ================================================================================================
GraphSlab* GraphVmHeap::AllocateSlab(size_t size, size_t num_peers, size_t slab_id) {
  ScopedLock k(lock_);

  if (!created_) {
    // Create VM heap if it's not created
    created_ = Create();
    if (!created_) {
      return nullptr;
    }
  }

  if (size > va_size_) {
    return nullptr;
  }

  auto slab_by_id = FetchSlab(slab_id);
  if (slab_by_id != nullptr) {
    return slab_by_id;
  }

  auto slab = GetSlab(size);
  slab->refcount_.store(num_peers);

  slab_ids[slab_id] = slab;

  max_total_size_ = std::max(max_total_size_, va_size_ - free_size_ + size);

  return slab;
}

// ================================================================================================
GraphSlab* GraphVmHeap::FetchSlab(size_t slab_id) {
  auto slab_by_id = slab_ids.find(slab_id);
  if (slab_by_id != slab_ids.end()) {
    return slab_by_id->second;
  }
  return nullptr;
}

// ================================================================================================
void GraphVmHeap::FreeSlab(GraphSlab* slab) {
  ScopedLock k(lock_);

  assert(created_);

  if (!slab->busy_.load()) {
    return;
  }

  if (!slab->mapped_.load()) {
    ReturnSlab(slab);
  } else if (ShouldUnmap()) {
    UnmapSlab(slab);
    ReturnSlab(slab);
  } else {
    CacheSlab(slab);
  }

  for (auto it = slab_ids.begin(); it != slab_ids.end(); ++it) {
    if (it->second == slab) {
      slab_ids.erase(it);
      break;
    }
  }

  max_total_size_ = std::max(max_total_size_, va_size_ - free_size_ + slab->size_);
}

// ================================================================================================
GraphSlab* GraphVmHeap::GetSlab(size_t size) {
  // Get the bin index based on size
  size_t bin_idx = 0;
  if (size < kMinSplitSize_) {
    bin_idx = 0;
  } else {
    bin_idx = GetPow2(RoundUpPow2(size + block_alignment_)) - kMinPow2_;
  }

  GraphSlab* curr_slab = nullptr;

  // If there is an already-mapped slab in the cache, reuse it
  if (cache_bins_[bin_idx] != nullptr) {
    curr_slab = PopBin(cache_bins_, bin_idx);
  } else {
    // Find the smallest matching unallocated slab
    size_t curr_bin_idx = max_bin_idx_;
    for (curr_bin_idx = bin_idx; curr_bin_idx < max_bin_idx_; ++curr_bin_idx) {
      if (free_bins_[curr_bin_idx]) {
	break;
      }
    }
    assert(curr_bin_idx < max_bin_idx_);

    curr_slab = PopBin(free_bins_, curr_bin_idx);

    // Split slab until the size is the smallest possible
    for (; curr_bin_idx != bin_idx; --curr_bin_idx) {
      curr_slab = SplitSlab(curr_slab, curr_bin_idx - 1);
    }
    curr_slab->mapped_.store(false);
  }

  curr_slab->busy_.store(true);
  PushBin(busy_bins_, bin_idx, curr_slab);

  free_size_ -= curr_slab->size_;

  return curr_slab;
}

// ================================================================================================
void GraphVmHeap::ReturnSlab(GraphSlab* slab, bool in_busy) {
  size_t bin_idx = GetPow2(slab->size_) - kMinPow2_;
  if (in_busy) {
    RemoveBin(busy_bins_, bin_idx, slab);
  }
  GraphSlab* curr_slab = slab;

  free_size_ += slab->size_;

  while (curr_slab->parent_ && !curr_slab->buddy_->busy_.load() && !curr_slab->buddy_->mapped_.load()) {
    RemoveBin(free_bins_, bin_idx, curr_slab->buddy_);
    curr_slab = CoalesceSlab(curr_slab, curr_slab->buddy_);
    ++bin_idx;
  }
  curr_slab->busy_.store(false);
  PushBin(free_bins_, bin_idx, curr_slab);
}

// ================================================================================================
void GraphVmHeap::CacheSlab(GraphSlab* slab) {
  size_t bin_idx = GetPow2(slab->size_) - kMinPow2_;
  RemoveBin(busy_bins_, bin_idx, slab);

  free_size_ += slab->size_;
  
  slab->busy_.store(false);
  PushBin(cache_bins_, bin_idx, slab);
}

// ================================================================================================
GraphSlab* GraphVmHeap::PopBin(std::vector<GraphSlab*>& bins, size_t bin_idx) {
  assert(bins[bin_idx] != nullptr);
  auto slab = bins[bin_idx];
  bins[bin_idx] = slab->next_;
  if (bins[bin_idx] != nullptr) {
    bins[bin_idx]->prev_ = nullptr;
  }
  slab->next_ = nullptr;
  return slab;
}

// ================================================================================================
void GraphVmHeap::RemoveBin(std::vector<GraphSlab*>& bins, size_t bin_idx, GraphSlab* slab) {
  if (slab->prev_ != nullptr) {
    slab->prev_->next_ = slab->next_;
  }
  if (slab->next_ != nullptr) {
    slab->next_->prev_ = slab->prev_;
  }
  if (bins[bin_idx] == slab) {
    bins[bin_idx] = slab->next_;
  }
  slab->prev_ = nullptr;
  slab->next_ = nullptr;
}

// ================================================================================================
void GraphVmHeap::PushBin(std::vector<GraphSlab*>& bins, size_t bin_idx, GraphSlab* slab) {
  slab->next_ = bins[bin_idx];
  slab->prev_ = nullptr;
  if (bins[bin_idx]) {
    bins[bin_idx]->prev_ = slab;
  }
  bins[bin_idx] = slab;
}

// ================================================================================================
GraphSlab* GraphVmHeap::SplitSlab(GraphSlab* slab, size_t bin_idx) {
  slab->busy_.store(true);

  GraphSlab* left_slab = new GraphSlab(slab->owner_, slab, slab->size_ / 2, slab->offset_, slab->mem_ptr_);
  GraphSlab* right_slab = new GraphSlab(slab->owner_, slab, slab->size_ / 2, slab->offset_ + (slab->size_ / 2), (void*) ((size_t) slab->mem_ptr_ + (slab->size_ / 2)));

  left_slab->buddy_ = right_slab;
  right_slab->buddy_ = left_slab;

  PushBin(free_bins_, bin_idx, right_slab);
  return left_slab;
}

// ================================================================================================
GraphSlab* GraphVmHeap::CoalesceSlab(GraphSlab* slab1, GraphSlab* slab2) {
  assert(slab1->size_ == slab2->size_);
  assert(slab1->parent_ == slab2->parent_);

  auto parent = slab1->parent_;
  parent->busy_.store(false);

  delete slab1;
  delete slab2;

  return parent;
}

// ================================================================================================
GraphTemporaryHeap::GraphTemporaryHeap(Device* device, size_t num_handles) :
  device_(device),
  num_handles_(num_handles), 
  base_ptr_(nullptr),
  lock_(true) {
    bump_counter_.store(0);
    invalidated_handles_.store(0);
    created_.store(false);
  }

// ================================================================================================
void GraphTemporaryHeap::Create() {
  ScopedLock k(lock_);

  if (created_.load()) {
    return;
  }

  base_ptr_ = reinterpret_cast<char*>(device_->virtualAlloc(nullptr, num_handles_ / 8, 8));
  created_.store(true);
}

// ================================================================================================
void* GraphTemporaryHeap::Allocate() {
  if (!created_.load()) {
    Create();
  }

  size_t my_offset = bump_counter_.fetch_add(1);
  if (my_offset > num_handles_) {
    LogError("Vm temporary heap ran out of handles");
    return nullptr;
  }

  size_t my_offset_bytes = my_offset / 8;
  size_t my_offset_bits = my_offset % 8;
  return (void*) (((size_t) (base_ptr_ + my_offset_bytes * 8)) | my_offset_bits);
}

// ================================================================================================
void GraphTemporaryHeap::Invalidate() {
  size_t handles_returned = invalidated_handles_.fetch_add(1);
  size_t handles_allocated = bump_counter_.load();
  if (handles_allocated == handles_returned && bump_counter_.compare_exchange_weak(handles_allocated, 0)) {
    invalidated_handles_.store(0);
  }
}

// ================================================================================================
bool GraphTemporaryHeap::InRange(void* ptr) {
  return base_ptr_ <= ptr && base_ptr_ + num_handles_ / 8 > ptr;
}

// ================================================================================================
void* GraphVmHeapArray::AllocateTemporaryHandle() {
  return tmp_heap_.Allocate();
}

// ================================================================================================
void GraphVmHeapArray::InvalidateTemporaryHandle() {
  return tmp_heap_.Invalidate();
}

// ================================================================================================
Command* GraphVmHeapArray::GetSlabMapCommand(GraphSlab* slab, HostQueue& queue) {
  assert(slab->Owner() != nullptr);
  return slab->Owner()->GetSlabMapCommand(slab, queue);
}

// ================================================================================================
Command* GraphVmHeapArray::GetSlabUnmapCommand(GraphSlab* slab, HostQueue& queue) {
  assert(slab->Owner() != nullptr);
  return slab->Owner()->GetSlabUnmapCommand(slab, queue);
}

// ================================================================================================
Command* GraphVmHeapArray::GetAllocationCommand(GraphSlab* slab, HostQueue& queue, Command* wait_command, size_t size, size_t offset) {
  assert(slab->Owner() != nullptr);
  return slab->Owner()->GetAllocationCommand(slab, queue, wait_command, size, offset);
}

// ================================================================================================
GraphSlab* GraphVmHeapArray::AllocateSlab(size_t size, size_t num_peers, size_t slab_id) {
  size_t my_tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  uint32_t my_heap = my_tid % kMaxArraySize;
  size_t free_device_memory[2];
  bool should_reuse_physical = false;
  device_->globalFreeMemory(free_device_memory);
  free_device_memory[0] *= 1024;

  if (size > free_device_memory[0]) {
    // Attempt to salvage #1: try trimming my heap
    vm_heaps_[my_heap]->TrimPhysMemory(0);
    device_->globalFreeMemory(free_device_memory);
    free_device_memory[0] *= 1024;

    if (size > free_device_memory[0]) {
      // Attempt to salvage #2: try trimming everyone
      for (uint32_t i = 0; i < kMaxArraySize; ++i) {
        vm_heaps_[my_heap]->TrimPhysMemory(0);
      }
      large_heap_.TrimPhysMemory(0);

      device_->globalFreeMemory(free_device_memory);
      free_device_memory[0] *= 1024;
      if (size > free_device_memory[0]) {
        // No way to get more physical memory, vm heap has to reuse existing physical buffer
        should_reuse_physical = true;
      }
    }
  }

  GraphSlab* slab = nullptr;
  // Try allocating from this thread's heap
  if (vm_heaps_[my_heap]->free_size_ > size) {
    slab = vm_heaps_[my_heap]->AllocateSlab(size, num_peers, my_tid + slab_id);
  }
  // If that fails, try allocating from large heap
  if (slab == nullptr) {
    slab = large_heap_.AllocateSlab(size, num_peers, my_tid + slab_id);
  }
  // If that fails, try allocating from other heaps
  if (slab == nullptr) {
    for (uint i = 0; i < kMaxArraySize; ++i) {
      if (i == my_heap) {
	continue;
      }

      slab = vm_heaps_[i]->AllocateSlab(size, num_peers, my_tid + slab_id);
      if (slab != nullptr) {
	break;
      }
    }
  }
  return slab;
}

// ================================================================================================
GraphSlab* GraphVmHeapArray::FetchSlab(size_t slab_id) {
  size_t my_tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  uint32_t my_heap = my_tid % kMaxArraySize;
  GraphSlab* slab = vm_heaps_[my_heap]->FetchSlab(my_tid + slab_id);
  if (slab == nullptr) {
    slab = large_heap_.FetchSlab(my_tid + slab_id);
  }
  if (slab == nullptr) {
    for (uint i = 0; i < kMaxArraySize; ++i) {
      if (i == my_heap) {
	continue;
      }

      slab = vm_heaps_[i]->FetchSlab(my_tid + slab_id);
      if (slab != nullptr) {
	break;
      }
    }
  }
  return slab;
}

// ================================================================================================
bool GraphVmHeapArray::FreeSlab(GraphSlab* slab) {
  assert(slab->Owner() != nullptr);

  for (uint i = 0; i < kMaxArraySize; ++i) {
    if (vm_heaps_[i] == slab->Owner()) {
      slab->Owner()->FreeSlab(slab);
      return true;
    }
  }
  if (&large_heap_ == slab->Owner()) {
    slab->Owner()->FreeSlab(slab);
    return true;
  }

  return false;
}


// ================================================================================================
bool GraphVmHeapArray::IsValidAllocation(void* ptr) {
  if (tmp_heap_.Created() && tmp_heap_.InRange(ptr)) {
    return true;
  }

  for (uint i = 0; i < kMaxArraySize; ++i) {
    if (vm_heaps_[i]->created_ && vm_heaps_[i]->InRange(ptr)) {
      return true;
    }
  }
  if (large_heap_.created_ && large_heap_.InRange(ptr)) {
    return true;
  }
  return false;
}

// ================================================================================================
void GraphVmHeapArray::FreeAllMemory(HostQueue& queue) {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    if (vm_heaps_[i]->created_) {
      vm_heaps_[i]->FreeAllMemory();
    }
  }

  if (large_heap_.created_) {
    large_heap_.FreeAllMemory();
  }
}

// ================================================================================================
void GraphVmHeapArray::TrimPhysMemory(size_t unmap_threshold) {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    // Check the threshold against the accumulated sizes in all heaps
    if (vm_heaps_[i]->created_ && [this]() {
      uint64_t size = 0;
      for (uint i = 0; i < kMaxArraySize; ++i) {
        size += vm_heaps_[i]->FreeMappedSize();
      }
      return size;
     }() > unmap_threshold) {
      vm_heaps_[i]->TrimPhysMemory(unmap_threshold);
    } else {
      break;
    }
  }

  if (large_heap_.created_ && [this]() {
    uint64_t size = 0;
    for (uint i = 0; i < kMaxArraySize; ++i) {
      size += large_heap_.FreeMappedSize();
    }
    return size;
   }() > unmap_threshold) {
    large_heap_.TrimPhysMemory(unmap_threshold);
  }
}

// ================================================================================================
void GraphVmHeapArray::SetUnmapThreshold(uint64_t threshold) {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    // Note: it's not precisely correct to use the same threshold in all heaps,
    // but the logic will trim heaps in Free()
    if (vm_heaps_[i]->created_) {
      vm_heaps_[i]->SetUnmapThreshold(threshold);
    }
  }
  if (large_heap_.created_) {
    large_heap_.SetUnmapThreshold(threshold);
  }
  unmap_threshold_ = threshold;
}

// ================================================================================================
uint64_t GraphVmHeapArray::MappedSize() const {
  uint64_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->MappedSize();
  }
  size += large_heap_.MappedSize();
  return size;
}

// ================================================================================================
uint64_t GraphVmHeapArray::FreeMappedSize() const {
  uint64_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->FreeMappedSize();
  }
  size += large_heap_.FreeMappedSize();
  return size;
}

// ================================================================================================
uint64_t GraphVmHeapArray::MaxMappedSize() const {
  uint64_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->max_mapped_size_.load();
  }
  size += large_heap_.max_mapped_size_.load();
  return size;
}

// ================================================================================================
void GraphVmHeapArray::ResetMaxMappedSize() {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    vm_heaps_[i]->max_mapped_size_.store(0);
  }
  large_heap_.max_mapped_size_.store(0);
}

// ================================================================================================
size_t GraphVmHeapArray::TotalSize() const {
  size_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->TotalSize();
  }
  size += large_heap_.TotalSize();
  return size;
}

// ================================================================================================
void GraphVmHeapArray::SetMaxTotalSize(size_t value) {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    vm_heaps_[i]->SetMaxTotalSize(value);
  }
  large_heap_.SetMaxTotalSize(value);
}

// ================================================================================================
size_t GraphVmHeapArray::MaxTotalSize() const {
  size_t size = 0;
  for (uint i = 0; i < kMaxArraySize; ++i) {
    size += vm_heaps_[i]->MaxTotalSize();
  }
  size += large_heap_.MaxTotalSize();
  return size;
}

} // namespace amd
