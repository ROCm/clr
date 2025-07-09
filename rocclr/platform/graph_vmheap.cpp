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
address GraphVmHeap::ReserveAddressRange(address start, size_t size, size_t alignment) {
  // Reserve a virtual address range on the device
  void* ptr = device_->virtualAlloc(start, size, alignment);
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

// ================================================================================================
std::vector<Command*> GraphVmHeap::GetAllocationCommands(void* ptr, HostQueue& queue) {
  ScopedLock k(lock_);

  if (heap_block_lookup.find(ptr) == heap_block_lookup.end()) {
    return {};
  }

  auto block = heap_block_lookup[ptr];

  std::vector<Command*> out;

  const auto& dev_info = device_->info();
  size_t granularity = dev_info.virtualMemAllocGranularity_;
  auto padded_size = alignUp(chunk_size_, granularity);

  for (auto chunk_idx : blocks_to_map[block]) {
    resident_blocks_[chunk_idx].fetch_add(1);
    auto addr = base_address_ + chunk_idx * chunk_size_;
    if (!mapped_mem_[chunk_idx].load()) {
      out.push_back(new CommitMemoryCommand(queue, Command::EventWaitList{}, addr, padded_size, nullptr, this, chunk_idx));
    }
  }

  blocks_to_map.erase(block);

  std::vector<Event*> wait_list;
  for (auto e : out) {
    wait_list.push_back(e);
  }

  out.push_back(new CreateMemoryCommand(queue, wait_list, ptr, block, base_memory_, block_alignment_, this));

  return out;
}

std::vector<Command*> GraphVmHeap::GetDeallocationCommands(HostQueue& queue) {
  ScopedLock k(lock_);

  std::vector<Command*> out;
  
  for (auto chunk_idx : chunks_to_unmap) {
    auto addr = base_address_ + chunk_idx * chunk_size_;
    if (resident_blocks_[chunk_idx].load() == 0) {
      out.push_back(new UncommitMemoryCommand(queue, Command::EventWaitList{}, addr, chunk_size_, nullptr, this, chunk_idx));
    }
  }

  chunks_to_unmap = {};

  return out;
}

// ================================================================================================
bool GraphVmHeap::UncommitMemory(void* addr, size_t size) {
  Memory* vaddr_sub_obj = MemObjMap::FindMemObj(addr);

  if (vaddr_sub_obj == nullptr) {
    return true;
  }

  Memory* phys_mem_obj = vaddr_sub_obj->getUserData().phys_mem_obj;

  // Unmap the physical memory from a virtual address
  Command* cmd = new VirtualMapCommand(
    GetVmQueue(), Command::EventWaitList{}, addr, size, nullptr);
  cmd->enqueue();
  cmd->awaitCompletion();
  cmd->release();
  vaddr_sub_obj->release();
  if (phys_mem_obj) {
    SvmBuffer::free(device_->context(), phys_mem_obj->getSvmPtr());
  }
  return true;
}

// ================================================================================================
GraphVmHeap::GraphVmHeap(Device* device, size_t va_size, size_t chunk_size, GetQueueFunc get_queue)
  : block_alignment_(kMinBlockAlignment)
  , chunk_size_(chunk_size)
  , lock_(true)
  , device_(device)
  , get_vm_queue_(get_queue) {
  va_size_ = alignUp(va_size, chunk_size);
  unmap_threshold_ = va_size / 2;
  free_size_ = va_size_;
}

// ================================================================================================
GraphVmHeap::~GraphVmHeap() {
  if (created_) {
    ScopedLock k(lock_);

    FreeAllMemory();

    if (mapped_mem_.size() > 0) {
      // Unmap the entire memory range
      UnmapPhysMemory(0, va_size_);
    }
    // Destroy virtual address space
    if (base_address_ != nullptr) {
      ReleaseAddressRange(base_address_);
    }

    for (auto l : chunk_locks_) {
      delete l;
    }
  }
}

// ================================================================================================
bool GraphVmHeap::Create() {
  // Create a new GPU resource
  base_address_ = ReserveAddressRange(0, va_size_, kChunkSize);
  if (base_address_ == nullptr) {
    return false;
  }
  free_size_ = va_size_;
  // Set up initial free list
  free_list_ = new GraphHeapBlock(this, va_size_, 0);
  if (free_list_ == nullptr) {
    return false;
  }
  mapped_mem_.resize(va_size_ / chunk_size_);

  resident_blocks_.resize(va_size_ / chunk_size_);

  chunk_locks_.resize(va_size_ / chunk_size_);
  for (size_t i = 0; i < chunk_locks_.size(); ++i) {
    mapped_mem_[i].store(false);
    resident_blocks_[i].store(0);
    chunk_locks_[i] = new Monitor(true);
  }

  // Ensures that NullStream exists before VmHeap destructor is called
  GetVmQueue();

  return true;
}

// ================================================================================================
void GraphVmHeap::MapPhysMemory(GraphHeapBlock* block, void* ptr) {
  size_t offset = block->Offset();
  size_t size = block->Size();

  auto start_chunk = offset / chunk_size_;
  auto end_chunk = alignUp(offset + size, chunk_size_) / chunk_size_;

  blocks_to_map[block] = {};
  for (auto i = start_chunk; i < end_chunk; ++i) {
    blocks_to_map[block].push_back(i);
  }

  heap_block_lookup[ptr] = block;
}

// ================================================================================================
void GraphVmHeap::UnmapPhysMemory(GraphHeapBlock* block) {
  size_t offset = block->Offset();
  size_t size = block->Size();
  
  auto start_chunk = offset / chunk_size_;
  auto end_chunk = alignUp(offset + size, chunk_size_) / chunk_size_;

  for (auto i = start_chunk; i < end_chunk; ++i) {
    if (resident_blocks_[i].fetch_sub(1) == 0) {
      chunks_to_unmap.insert(i);
    }
  }
}

// ================================================================================================
void GraphVmHeap::UnmapPhysMemory(size_t offset, size_t size) {
  auto busy_size = va_size_ - free_size_;
  uint64_t free_mapped = alignDown(mapped_size_.load() - busy_size, kChunkSize);

  int start_chunk = alignUp(offset, chunk_size_) / chunk_size_;
  int end_chunk = alignDown(offset + size, chunk_size_) / chunk_size_;

  for (int i = end_chunk - 1; i >= start_chunk; i--) {
    // If free mapped memory lower than the threshold, then stop unmapping
    if (free_mapped <= unmap_threshold_) {
      return;
    }
    if (i >= mapped_mem_.size()) {
      assert(false);
      LogError("VM heap allocation is beyond the range!");
      return;
    }
    if (mapped_mem_[i].load()) {
      auto address = base_address_ + i * chunk_size_;
      if (UncommitMemory(address, chunk_size_)) {
        mapped_size_.fetch_sub(chunk_size_);
        free_mapped -= chunk_size_;
        mapped_mem_[i].store(false);
      }
      else {
        assert(false);
      }
    }
  }
}

// ================================================================================================
bool GraphVmHeap::BlockFullyMapped(GraphHeapBlock* block) {
  size_t offset = block->Offset();
  size_t size = block->Size();

  auto start_chunk = offset / chunk_size_;
  auto end_chunk = alignUp(offset + size, chunk_size_) / chunk_size_;

  for (auto i = start_chunk; i < end_chunk; ++i) {
    if (!mapped_mem_[i].load()) {
      return false;
    }
  }
  return true;
}

// ================================================================================================
void GraphVmHeap::TrimPhysMemory(size_t unmap_threshold, bool immediate) {
  ScopedLock k(lock_);
  auto current = free_list_;
  auto unmap_org = unmap_threshold_;
  unmap_threshold_ = unmap_threshold;
  while (current != nullptr) {
    UnmapPhysMemory(current);
    current = current->next_;
  }
  if (immediate) {
    auto deallocation_commands = GetDeallocationCommands(GetVmQueue());
    for (auto cmd : deallocation_commands) {
      cmd->enqueue();
      cmd->awaitCompletion();
      cmd->release();
    }
  }

  unmap_threshold_ = unmap_org;
}

// ================================================================================================
address GraphVmHeap::Alloc(size_t size, bool should_reuse_physical) {
  ScopedLock k(lock_);

  if (!created_) {
    // Create VM heap if it's not created
    created_ = Create();
    if (!created_) {
      return nullptr;
    }
  }
  address ptr = nullptr;
  size_t offset = 0;
  auto hb = AllocBlock(size + block_alignment_, should_reuse_physical);
  if (hb != nullptr) {
    offset = ((hb->Offset() & ~kChunkSize) == 0) ? hb->Offset() + block_alignment_ : hb->Offset();
    ptr = base_address_ + offset;
  } else {
    return nullptr;
  }
  MapPhysMemory(hb, ptr);
  max_total_size_ = std::max(max_total_size_, va_size_ - free_size_ + size);
  return ptr;
}

// ================================================================================================
void GraphVmHeap::Free(Memory* memory) {
  const device::Memory* dev_mem = memory->getDeviceMemory(*device_);
  void* addr = reinterpret_cast<void*>(dev_mem->virtualAddress());
  if (addr == nullptr) {
    addr = memory->getSvmPtr();
  }

  if (!created_ || (addr < base_address_)) {
    return;
  }
  ScopedLock k(lock_);
  if (memory->getUserData().data != nullptr) {
    auto hb = reinterpret_cast<GraphHeapBlock*>(memory->getUserData().data);
    ClPrint(LOG_INFO, LOG_MEM_POOL, "GraphVmHeap Free: %p offset(%zx + %zx) hb(%p)",
      addr, hb->Offset(), memory->getSize(), hb);
    FreeBlock(hb);
    max_total_size_ = std::max(max_total_size_, va_size_ - free_size_ + memory->getSize());
  }
  MemObjMap::RemoveMemObj(addr);
  memory->release();
}

// ================================================================================================
void GraphVmHeap::FreeAllMemory() {
  ScopedLock k(lock_);

  // Release all heap blocks
  GraphHeapBlock* walk, * next;
  walk = busy_list_;
  while (walk) {
    next = walk->next_;
    FreeBlock(walk);
    walk = next;
  }

  walk = free_list_;
  while (walk) {
    next = walk->next_;
    delete walk;
    walk = next;
  }
}

// ================================================================================================
GraphHeapBlock* GraphVmHeap::AllocBlock(size_t un_size, bool should_reuse_physical) {
  assert(un_size != 0);
  ScopedLock k(lock_);
  GraphHeapBlock* walk = free_list_;
  GraphHeapBlock* best = nullptr;

  // Round size
  auto size = alignUp(un_size, block_alignment_);

  // Walk the free list looking for a suitable block (currently best-fit)
  while (walk) {
    if ((walk->size_ > size) &&
	(best == nullptr || walk->size_ < best->size_) &&
	(!should_reuse_physical || BlockFullyMapped(walk))) {
      best = walk;
    } else if (walk->size_ == size && (!should_reuse_physical || BlockFullyMapped(walk))) {
      // No need to split, just move to busy list
      DetachBlock(&free_list_, walk);
      walk->busy_ = true;
      InsertBlock(&busy_list_, walk);
      free_size_ -= size;
      return walk;
    }
    walk = walk->next_;
  }

  if (best != nullptr) {
    // Got one, but need to split it. Keep first part in free list,
    // put second part into busy list
    GraphHeapBlock *newblock = SplitBlock(best, size);
    newblock->busy_ = true;
    InsertBlock(&busy_list_, newblock);
    free_size_ -= size;
    return newblock;
  }

  return nullptr;
}

// ================================================================================================
void GraphVmHeap::FreeBlock(GraphHeapBlock* blk) {
  UnmapPhysMemory(blk);
  DetachBlock(&busy_list_, blk);
  blk->busy_ = false;
  free_size_ += blk->size_;
  MergeBlock(&free_list_, blk);
}

// ================================================================================================
void GraphVmHeap::DetachBlock(GraphHeapBlock** list, GraphHeapBlock* blk) {
  if (*list == blk) {
    *list = blk->next_;
  }
  if (blk->prev_) {
    blk->prev_->next_ = blk->next_;
  }
  if (blk->next_) {
    blk->next_->prev_ = blk->prev_;
  }
}

// ================================================================================================
void GraphVmHeap::InsertBlock(GraphHeapBlock** head, GraphHeapBlock* blk) {
  if (nullptr == *head) {
    *head = blk;
    blk->prev_ = nullptr;
    blk->next_ = nullptr;
    return;
  }

  // Find the place to insert it at
  GraphHeapBlock* walk = *head;
  while (walk->next_ && walk->next_->offset_ < blk->offset_) {
    walk = walk->next_;
  }

  // Insert it
  if (walk == *head) {
    if (walk->offset_ >= blk->offset_) {
      *head = blk;
      blk->prev_ = nullptr;
      blk->next_ = walk;
      walk->prev_ = *head;
      return;
    }
  }

  blk->next_ = walk->next_;
  blk->prev_ = walk;
  if (walk->next_) {
      walk->next_->prev_ = blk;
  }
  walk->next_ = blk;
}

// ================================================================================================
GraphHeapBlock* GraphVmHeap::SplitBlock(GraphHeapBlock* blk, size_t tailsize) {
  // Create a new block from the beginning of the current
  GraphHeapBlock* nb = new GraphHeapBlock(blk->owner_, tailsize, blk->offset_);

  // Resize the old block
  blk->offset_ += tailsize;
  blk->size_ -= tailsize;
  return nb;
}

// ================================================================================================
void GraphVmHeap::Join2Blocks(GraphHeapBlock* first, GraphHeapBlock* second) const {
  // Do the join
  first->size_ = first->size_ + second->size_;
  first->next_ = second->next_;
  if (second->next_) {
      second->next_->prev_ = first;
  }
  delete second;
}

// ================================================================================================
void GraphVmHeap::MergeBlock(GraphHeapBlock** head, GraphHeapBlock* blk) {
  InsertBlock(head, blk);

  // Merge with successor if possible
  if ((blk->next_ != nullptr) && (blk->offset_ + blk->size_ == blk->next_->offset_)) {
      Join2Blocks(blk, blk->next_);
  }

  // Merge with predecessor if possible
  if ((blk->prev_ != nullptr) && (blk->prev_->offset_ + blk->prev_->size_ == blk->offset_)) {
      Join2Blocks(blk->prev_, blk);
  }
}

// ================================================================================================
std::vector<Command*> GraphVmHeapArray::GetAllocationCommands(void* ptr, HostQueue& queue) {
  uint32_t my_heap = std::hash<std::thread::id>{}(std::this_thread::get_id()) % kMaxArraySize;
  auto out = vm_heaps_[my_heap]->GetAllocationCommands(ptr, queue);
  if (!out.empty()) {
    return out;
  }
  if (large_heap_.created_) {
    return large_heap_.GetAllocationCommands(ptr, queue);
  }
  return {};
}

// ================================================================================================
std::vector<Command*> GraphVmHeapArray::GetDeallocationCommands(HostQueue& queue) {
  uint32_t my_heap = std::hash<std::thread::id>{}(std::this_thread::get_id()) % kMaxArraySize;
  auto out = vm_heaps_[my_heap]->GetDeallocationCommands(queue);
  if (!out.empty()) {
    return out;
  }
  if (large_heap_.created_) {
    return large_heap_.GetDeallocationCommands(queue);
  }
  return {};
}

// ================================================================================================
address GraphVmHeapArray::Alloc(size_t size) {
  uint32_t my_heap = std::hash<std::thread::id>{}(std::this_thread::get_id()) % kMaxArraySize;
  size_t free_device_memory[2];
  bool should_reuse_physical = false;
  device_->globalFreeMemory(free_device_memory);
  free_device_memory[0] *= 1024;

  if (size + GraphVmHeap::kChunkSize > free_device_memory[0]) {
    // Attempt to salvage #1: try trimming my heap
    vm_heaps_[my_heap]->TrimPhysMemory(0, true);
    device_->globalFreeMemory(free_device_memory);
    free_device_memory[0] *= 1024;

    if (size + GraphVmHeap::kChunkSize > free_device_memory[0]) {
      // Attempt to salvage #2: try trimming everyone
      for (uint32_t i = 0; i < kMaxArraySize; ++i) {
	vm_heaps_[my_heap]->TrimPhysMemory(0, true);
      }
      large_heap_.TrimPhysMemory(0, true);

      device_->globalFreeMemory(free_device_memory);
      free_device_memory[0] *= 1024;
      if (size + GraphVmHeap::kChunkSize > free_device_memory[0]) {
	// No way to get more physical memory, vm heap has to reuse existing physical buffer
	should_reuse_physical = true;
      }
    }
  }

  address addr = nullptr;
  // Try allocating on this thread's heap
  if (vm_heaps_[my_heap]->free_size_ > (size + GraphVmHeap::kChunkSize)) {
    addr = vm_heaps_[my_heap]->Alloc(size, should_reuse_physical);
  }
  // If that fails, try allocating on the large heap
  if (addr == nullptr) {
    addr = large_heap_.Alloc(size, should_reuse_physical);
  }
  // And if that fails, try allocating from another thread heap.
  // May introduce contention but that's better than failing
  if (addr == nullptr) {
    for (uint i = 0; i < kMaxArraySize; ++i) {
      if (i == my_heap) {
	continue;
      }

      addr = vm_heaps_[i]->Alloc(size, should_reuse_physical);
      if (addr != nullptr) {
	break;
      }
    }
  }
  return addr;
}

// ================================================================================================
bool GraphVmHeapArray::IsValidAllocation(void* ptr) {
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
bool GraphVmHeapArray::Free(amd::Memory* memory) {
  const device::Memory* dev_mem = memory->getDeviceMemory(*device_);
  void* addr = reinterpret_cast<void*>(dev_mem->virtualAddress());
  if (addr == nullptr) {
    addr = memory->getSvmPtr();
  }

  uint32_t my_heap = std::hash<std::thread::id>{}(std::this_thread::get_id()) % kMaxArraySize;
  if (vm_heaps_[my_heap]->created_ && vm_heaps_[my_heap]->InRange(addr)) {
    vm_heaps_[my_heap]->Free(memory);
    return true;
  } else if (large_heap_.created_ && large_heap_.InRange(addr)) {
    large_heap_.Free(memory);
    return true;
  } else {
    for (uint i = 0; i < kMaxArraySize; ++i) {
      if (i == my_heap) {
	continue;
      }

      if (vm_heaps_[i]->created_ && vm_heaps_[my_heap]->InRange(addr)) {
	vm_heaps_[i]->Free(memory);
	return true;
      }
    }
  }
  return false;
}

// ================================================================================================
void GraphVmHeapArray::FreeAllMemory(HostQueue& queue) {
  for (uint i = 0; i < kMaxArraySize; ++i) {
    if (vm_heaps_[i]->created_) {
      vm_heaps_[i]->FreeAllMemory();
      auto deallocation_commands = vm_heaps_[i]->GetDeallocationCommands(queue);
      for (auto cmd : deallocation_commands) {
	cmd->enqueue();
	cmd->awaitCompletion();
	cmd->release();
      }
    }
  }

  if (large_heap_.created_) {
    large_heap_.FreeAllMemory();
    auto deallocation_commands = large_heap_.GetDeallocationCommands(queue);
    for (auto cmd : deallocation_commands) {
      cmd->enqueue();
      cmd->awaitCompletion();
      cmd->release();
    }
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
bool GraphVmHeapArray::IsBusyMemory(Memory* memory) const {
  if (memory->getUserData().data != nullptr) {
    auto hb = reinterpret_cast<GraphHeapBlock*>(memory->getUserData().data);
    return hb->Busy();
  }
  return false;
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
