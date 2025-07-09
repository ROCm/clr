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

#pragma once

#include <atomic>
#include <functional>
#include "top.hpp"
#include "device/device.hpp"
#include "object.hpp"
#include "commandqueue.hpp"

namespace amd {

class GraphHeapBlock;
class GraphVmHeap;
class GraphVmHeapArray;

class GraphHeapBlock : public amd::HeapObject {
public:
  friend GraphVmHeap;
  //! Constructor
  GraphHeapBlock(
      GraphVmHeap* owner = nullptr,  //!< GraphVmHeap object that owns this heap block
      size_t size = 0,          //!< Heap block size for allocation
      size_t offset = 0)        //!< Heap block offset
      : owner_(owner)
      , size_(size)
      , offset_(offset)
      , next_(nullptr)
      , prev_(nullptr)
      , busy_(false)
      {}

  //! Destructor does some sanity checks
  ~GraphHeapBlock() { assert(!busy_ && "The blocked must be destroyed explicitly!"); }

  //! Gets the offset
  size_t Offset() const { return offset_; }

  //! Gets the size
  size_t Size() const { return size_; }

  //! Gets the busy flag
  bool Busy() const { return busy_; }

private:
  GraphHeapBlock() = delete;
  GraphHeapBlock(const GraphHeapBlock&) = delete;
  GraphHeapBlock& operator=(const GraphHeapBlock&) = delete;

  GraphVmHeap*     owner_;   //!< Heap that owns this block
  size_t      size_;    //!< Size of the block in bytes
  size_t      offset_;  //!< Offset of this block in the heap
  GraphHeapBlock*  next_;    //!< Next block on the list, or nullptr
  GraphHeapBlock*  prev_;    //!< Previous block on the list, or nullptr
  bool        busy_;    //!< True if the block is in use
};

class GraphVmHeap {
  // Wrapper around STL's atomic that allows to have vectors of atomics
  template <typename T>
  struct atomwrapper
  {
    std::atomic<T> _a;
    
    atomwrapper() : _a() {}

    atomwrapper(const std::atomic<T> &a) : _a(a.load()) {}

    atomwrapper(const atomwrapper &other) : _a(other._a.load()) {}

    atomwrapper &operator=(const atomwrapper &other) { _a.store(other._a.load()); }

    T load() { return _a.load(); }
    
    void store(T val) { return _a.store(val); }

    T fetch_add(T val) { return _a.fetch_add(val); }

    T fetch_sub(T val) { return _a.fetch_sub(val); }
  };

  // A command that creates an amd::Buffer for a GraphHeapBlock. Runs after all the physical chunks of the block have been mapped.
  class CreateMemoryCommand : public Command {
    public:
      CreateMemoryCommand(HostQueue& queue, const Event::EventWaitList& eventWaitList, void* ptr, GraphHeapBlock* block, Memory* base_memory, size_t block_alignment, GraphVmHeap *vm_heap)
	:
	  Command(queue, 0, eventWaitList, 0, nullptr),
	  ptr_(ptr),
	  block_(block),
	  base_memory_(base_memory),
	  block_alignment_(block_alignment),
	  vm_heap_(vm_heap) {}

      virtual void submit(device::VirtualDevice& device) final {
	size_t size = block_->Size() - block_alignment_;
	size_t offset = ((block_->Offset() & ~kChunkSize) == 0) ? block_->Offset() + block_alignment_ : block_->Offset();
	auto memory = new (base_memory_->getContext()) Buffer(*base_memory_, 0, offset, size);
	if (nullptr == memory || !memory->create(nullptr)) {
	  // FIXME: FreeBlock() ?
	  return;
	}
	MemObjMap::AddMemObj(ptr_, memory);
	if (memory->getUserData().data == nullptr) {
	  memory->getUserData().data = block_;
	}
      }
    private:
      void* ptr_;
      GraphHeapBlock* block_;
      Memory* base_memory_;
      size_t block_alignment_;
      GraphVmHeap* vm_heap_;
  };

  // A command that allocates and maps physical chunks of memory  
  class CommitMemoryCommand : public VirtualMapCommand {
    public:
      CommitMemoryCommand(HostQueue& queue, const Event::EventWaitList& eventWaitList, void* ptr, size_t size, Memory* memory, GraphVmHeap* vm_heap, size_t chunk_idx)
	: VirtualMapCommand(queue, eventWaitList, ptr, size, memory),
	  vm_heap_(vm_heap),
	  chunk_idx_(chunk_idx),
	  mptr_(ptr) {}

      virtual void submit(device::VirtualDevice& device) final {
	// This command lost the race, bail
	if (vm_heap_->mapped_mem_[chunk_idx_].load()) {
	  return;
	}

	ScopedLock(*vm_heap_->chunk_locks_[chunk_idx_]);

	if (vm_heap_->mapped_mem_[chunk_idx_].load()) {
	  return;
	}

	const auto& dev_info = vm_heap_->device_->info();

	// Allocate physical memory
	void* dptr = SvmBuffer::malloc(vm_heap_->device_->context(), ROCCLR_MEM_PHYMEM, size_,
				       dev_info.memBaseAddrAlign_, nullptr);
	if (dptr == nullptr) {
	  LogPrintfError("Failed to allocate physical memory %zd", size_);
	  return;
	}

	size_t offset = 0;
	// Find physical memory in the map of all objects
	memory_ = MemObjMap::FindMemObj(dptr, &offset);

	// Map the physical memory to a virtual address
	VirtualMapCommand::submit(device);

	// Enable memory access
	if (!vm_heap_->device_->SetMemAccess(mptr_, size_, Device::VmmAccess::kReadWrite)) {
	  LogError("SetAccess failed for the commited memory in GraphVmHeap!");
	}

	// Update mapped size in GraphVmHeap, signal that mapping succeeded
	auto mapped_size = vm_heap_->mapped_size_.fetch_add(size_);
	auto prev_max_mapped_size = vm_heap_->max_mapped_size_.load();
	while (prev_max_mapped_size < prev_max_mapped_size + size_ &&
	  !vm_heap_->max_mapped_size_.compare_exchange_weak(prev_max_mapped_size, prev_max_mapped_size + size_)) {}
	vm_heap_->mapped_mem_[chunk_idx_].store(true);
      }
    private:
      GraphVmHeap* vm_heap_;
      size_t chunk_idx_;
      void* mptr_;
  };

  // A command that unmaps and deallocates physical chunks of memory
  class UncommitMemoryCommand : public VirtualMapCommand {
    public:
      UncommitMemoryCommand(HostQueue& queue, const Event::EventWaitList& eventWaitList, void* ptr, size_t size, Memory* memory, GraphVmHeap* vm_heap, size_t chunk_idx)
	: VirtualMapCommand(queue, eventWaitList, ptr, size, memory),
	  vm_heap_(vm_heap),
	  chunk_idx_(chunk_idx),
	  mptr_(ptr) {}

      virtual void submit(device::VirtualDevice& device) final {
	// This command either lost the race or a block was put in the chunk, bail
	if (!vm_heap_->mapped_mem_[chunk_idx_].load() || vm_heap_->resident_blocks_[chunk_idx_].load() != 0) {
	  return;
	}

	ScopedLock k(*vm_heap_->chunk_locks_[chunk_idx_]);

	// This command either lost the race or a block was put in the chunk, bail
	if (!vm_heap_->mapped_mem_[chunk_idx_].load() || vm_heap_->resident_blocks_[chunk_idx_].load() != 0) {
	  return;
	}

	auto busy_size = vm_heap_->va_size_ - vm_heap_->free_size_;
	uint64_t free_mapped = alignDown(vm_heap_->mapped_size_.load() - busy_size, vm_heap_->chunk_size_);
	// If free mapped memory lower than the threshold, then stop unmapping
	if (free_mapped <= vm_heap_->unmap_threshold_) {
	  return;
	}

	Memory* vaddr_sub_obj = MemObjMap::FindMemObj(mptr_);
	Memory* phys_mem_obj = vaddr_sub_obj->getUserData().phys_mem_obj;

	// Unmap the physical memory from a virtual address
	VirtualMapCommand::submit(device);

	vaddr_sub_obj->release();

	// Deallocate physical memory
	SvmBuffer::free(vm_heap_->device_->context(), phys_mem_obj->getSvmPtr());

	vm_heap_->mapped_size_.fetch_sub(vm_heap_->chunk_size_);
	vm_heap_->mapped_mem_[chunk_idx_].store(false);
      }
    private:
      GraphVmHeap* vm_heap_;
      size_t chunk_idx_;
      void* mptr_;
  };

public:
  friend GraphVmHeapArray;
  static const size_t kChunkSize = 32 * Mi; //!< Chunk size, must be power of 2
  static const size_t kMinBlockAlignment = 256;
  typedef std::function<amd::HostQueue&()> GetQueueFunc;

  GraphVmHeap(Device* device,        //!< GPU device object
         GetQueueFunc get_queue //!< Function to retrieve a map queue
         )
      : GraphVmHeap(device, device->info().globalMemSize_ / 8, kChunkSize, get_queue) {
  }

  GraphVmHeap(Device* device,        //!< GPU device object
         size_t  va_size,       //!< The size of the allocated heap (bytes).Virtual address space
         size_t  chunk_size,    //!< The size of single chunk for physical memory growth
         GetQueueFunc get_queue //!< Function to retrieve a map/unmap queue
         );

  //! Heap destructor
  virtual ~GraphVmHeap();

  //! Returns a pointer to the allocated device memory from a heap
  address Alloc(
      size_t size,     //! The allocation size
      bool should_reuse_physical
      );

  //! Release memory back to the VM heap
  void Free(amd::Memory* memory);

  //! Release all heap blocks
  void FreeAllMemory();

  //! Unmaps freed memory based on the threshold
  void TrimPhysMemory(size_t unmap_threshold, bool immediate = false);

  //! Enable memory unmap threashold (default 0 unmap always)
  void SetUnmapThreshold(uint64_t threshold) { unmap_threshold_ = threshold; }

  //! Returns mapped memory size (total allocated physical memory)
  uint64_t MappedSize() const { return mapped_size_.load(); }

  //! Returns mapped memory size (allocated physical memory) without actual allocations
  uint64_t FreeMappedSize() const { return mapped_size_.load() - (va_size_ - free_size_); }

  //! Gets current size of the heap (both mapped and unmapped)
  size_t TotalSize() const { return va_size_ - free_size_; }

  //! Sets maximum size of the heap (both mapped and unmapped) 
  void SetMaxTotalSize(size_t value) { max_total_size_ = value; }

  //! Gets maximum size of the heap (both mapped and unmapped)  
  size_t MaxTotalSize() const { return max_total_size_; }

  //! Returns true if the address is in the range of this heap
  bool InRange(void* addr) {
    return ((addr >= base_address_) && (addr <= (base_address_ + va_size_))) ? true : false;
  }

  //! Creates commands to map physical memory chunks corresponding to object at ptr
  std::vector<Command*> GetAllocationCommands(void* ptr, HostQueue& queue);

  //! Creates commands to unmap physical memory chunks
  std::vector<Command*> GetDeallocationCommands(amd::HostQueue& queue);

private:
  GraphVmHeap() = delete;
  GraphVmHeap(const GraphVmHeap&) = delete;
  GraphVmHeap& operator=(const GraphVmHeap&) = delete;

  //! Ceates heap object. Reserves virtual address range for the heap operation
  bool Create();

  //! Reseves address range for memory allocations
  address ReserveAddressRange(address start, size_t size, size_t alignment);

  //! Releases address range specified by the address
  bool ReleaseAddressRange(void* addr);

  //! Uncommits physical memory from the spcified address
  bool UncommitMemory(void* addr, size_t size);

  GraphHeapBlock* AllocBlock(size_t size,  //! The allocation size
			     bool should_reuse_physical
                        );

  //! Release memory back to a heap
  void FreeBlock(GraphHeapBlock* blk);

  //! Insert a block into a list
  void InsertBlock(GraphHeapBlock** list, GraphHeapBlock* node);

  //! Merge a block into a list
  void MergeBlock(GraphHeapBlock** list, GraphHeapBlock* node);

  //! Remove a block from a list
  void DetachBlock(GraphHeapBlock** list, GraphHeapBlock* node);

  //! Splits a block into two pieces
  GraphHeapBlock* SplitBlock(GraphHeapBlock* node, size_t size);

  //! Gets ready to map physical memory into specified block
  void MapPhysMemory(GraphHeapBlock* block, void* ptr);

  //! Gets ready to unmap physical memory from the specified block
  void UnmapPhysMemory(GraphHeapBlock* block);

  //! Unmaps physical memory from the specified address
  void UnmapPhysMemory(size_t offset, size_t size);

  //! Checks that the block already has all necessary physical memory
  bool BlockFullyMapped(GraphHeapBlock* block, size_t size);

  //! Join two blocks, transferring the size of the second into the first and deleting the second
  void Join2Blocks(GraphHeapBlock* first, GraphHeapBlock* second) const;

  //! Returns a queue for VM map/unmap operations
  amd::HostQueue& GetVmQueue() const { return get_vm_queue_(); }

  address       base_address_ = nullptr;  //!< GPU virtual address base of the heap
  Memory*  base_memory_ = nullptr;   //!< VA space base object, used in the view creation
  GraphHeapBlock*    free_list_ = nullptr;     //!< Head block for free list
  GraphHeapBlock*    busy_list_ = nullptr;     //!< Head block for busy list
  size_t        free_size_ = 0;           //!< Total free size of the heap (both mapped and unmapped)
  size_t        va_size_ = 0;             //!< Heap virtual address space size
  size_t        block_alignment_ = 1;     //!< Size of an allocation page
  size_t        chunk_size_ = 0;          //!< Chunk size (min physical allocation for the growth)
  uint64_t      unmap_threshold_ = 0;     //!< Unmap threshold in bytes,used to release phys memory
  std::atomic<uint64_t>      mapped_size_ = 0;         //!< Size of mapped memory
  std::atomic<uint64_t>      max_mapped_size_ = 0;     //!< Max size of mapped memory in this heap
  uint64_t      max_total_size_ = 0;	  //!< Max allowed size in this heap
  bool          created_ = false;         //!< Used for deferred VM heap allocation
  amd::Monitor  lock_;                    //!< Lock to serialise heap accesses
  Device*       device_;                  //!< Device that owns this heap
  GetQueueFunc  get_vm_queue_;            //!< Queue for VM operations

  std::map<GraphHeapBlock*, std::vector<size_t>> blocks_to_map; //!< Blocks that need to be mapped
  std::set<size_t> chunks_to_unmap;                             //!< Chunks that need to be unmapped
  std::map<void*, GraphHeapBlock*> heap_block_lookup;           //!< A map from memory pointers to heap blocks
  std::vector<atomwrapper<uint32_t>> resident_blocks_;          //!< Number of resident blocks per chunk, the size is total_size/chunk_size
  std::vector<amd::Monitor *> chunk_locks_;                     //!< One lock per physical memory chunk, the size is total_size/chunk_size
  std::vector<atomwrapper<bool>> mapped_mem_;                   //!< A map of mapped memory, the size is total_size/chunk_size
};

//! Implements an array of vm heaps of different sizes for more efficient management
class GraphVmHeapArray {
public:
  GraphVmHeapArray(Device* device,    //!< GPU device object
              GraphVmHeap::GetQueueFunc get_queue  //!< Function to retrieve a map queue
  ) : heap0_(device, device->info().globalMemSize_ / 4, GraphVmHeap::kChunkSize, get_queue)
    , heap1_(device, device->info().globalMemSize_ / 4, GraphVmHeap::kChunkSize, get_queue)
    , heap2_(device, device->info().globalMemSize_ / 4, GraphVmHeap::kChunkSize, get_queue)
    , heap3_(device, device->info().globalMemSize_ / 4, GraphVmHeap::kChunkSize, get_queue)
    , large_heap_(device, device->info().globalMemSize_ / 4, GraphVmHeap::kChunkSize, get_queue)
    , device_(device) {}

  //! Returns a pointer to the allocated device memory from a heap
  address Alloc(
    size_t size     //! The allocation size
    );

  //! Checks that ptr was allocated from this GraphVmHeapArray
  bool IsValidAllocation(void* ptr);

  //! Creates commands to map physical memory chunks corresponding to object at ptr
  std::vector<Command*> GetAllocationCommands(void* ptr, HostQueue& queue);

  //! Creates commands to unmap physical memory chunks
  std::vector<Command*> GetDeallocationCommands(HostQueue& queue);

  //! Release memory back to the VM heap
  bool Free(amd::Memory* memory);

  //! Immediately frees and trims all memory
  void FreeAllMemory(HostQueue& queue);

  //! Unmaps freed memory based on the threshold
  void TrimPhysMemory(size_t unmap_threshold);

  //! Enable memory unmap threashold (default 0 unmap always)
  void SetUnmapThreshold(uint64_t threshold);

  //! Returns mapped memory size (total allocated physical memory)
  uint64_t MappedSize() const;

  //! Returns mapped memory size (allocated physical memory) without actual allocations
  uint64_t FreeMappedSize() const;

  //! Returns the maximum mapped memory size
  uint64_t MaxMappedSize() const;

  //! Returns the maximum mapped memory size
  void ResetMaxMappedSize();

  //! Checks if memory is active and belongs to the busy heap
  bool IsBusyMemory(Memory* memory) const;

  //! Gets current size of all heaps (both mapped and unmapped)
  size_t TotalSize() const;

  //! Sets maximum size of all heaps (both mapped and unmapped) 
  void SetMaxTotalSize(size_t value); 

  //! Gets maximum size of all heaps (both mapped and unmapped)  
  size_t MaxTotalSize() const;

private:
  GraphVmHeapArray() = delete;
  GraphVmHeapArray(const GraphVmHeapArray&) = delete;
  GraphVmHeapArray& operator=(const GraphVmHeapArray&) = delete;

  static const uint32_t kMaxArraySize = 4;  //!< The number of vm heap in the array
  // @note: gcc10.2 or lower wrongly uses copy constructor in the initialization
  // of GraphVmHeap array of objects. Hence, use an array of GraphVmHeap pointers for now
  GraphVmHeap* vm_heaps_[kMaxArraySize] = {&heap0_, &heap1_, &heap2_, &heap3_};  //!< The array of heaps
  GraphVmHeap  heap0_;
  GraphVmHeap  heap1_;
  GraphVmHeap  heap2_;
  GraphVmHeap  heap3_;
  GraphVmHeap  large_heap_;

  uint64_t unmap_threshold_ = 0;  //!< Unmap threshold in bytes,used to release phys memory
  Device* device_;                //!< Device that owns this heap
};

} // namespace amd
