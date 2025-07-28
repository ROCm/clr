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

class GraphVmHeap;
class GraphVmHeapArray;

class GraphSlab : public amd::HeapObject {
public:
  friend GraphVmHeap;

  GraphSlab(GraphVmHeap* owner, GraphSlab* parent, size_t size, size_t offset, void* mem_ptr)
    : owner_(owner),
      size_(size),
      offset_(offset),
      mem_ptr_(mem_ptr),
      parent_(parent),
      buddy_(nullptr),
      next_(nullptr),
      prev_(nullptr),
      lock_(true),
      refcount_(0),
      busy_(false),
      mapped_(false) {}

  GraphVmHeap* Owner() const {
    return owner_;
  }

  void* MemPtr() const {
    return mem_ptr_;
  }

  size_t Size() const {
    return size_;
  }

  bool IsMapped() const {
    return mapped_.load();
  }

  void Mapped() {
    mapped_.store(true);
  }

  void Unmapped() {
    mapped_.store(false);
  }

  size_t DecrementRefcount() {
    return refcount_.fetch_sub(1);
  }

  void Lock() {
      lock_.lock();
  }

  void Unlock() {
      lock_.unlock();
  }

  bool Busy() const {
    return busy_.load();
  }
private:
  GraphSlab() = delete;
  GraphSlab(const GraphSlab&) = delete;
  GraphSlab& operator=(const GraphSlab&) = delete;

  GraphVmHeap* owner_;
  size_t size_;
  size_t offset_; //!< stores offset from the owner's base_address_
  void* mem_ptr_; //!< stores ptr to base_address_ + offset_ + block_alignment_
  GraphSlab* parent_;
  GraphSlab* buddy_;
  GraphSlab* next_;
  GraphSlab* prev_;

  Monitor lock_;

  std::atomic<size_t> refcount_;
  std::atomic<bool> busy_;
  std::atomic<bool> mapped_;
};

class GraphVmHeap {
  // A command that creates an amd::Buffer for a GraphHeapBlock. Runs after all the physical chunks of the block have been mapped.
  class CreateMemoryCommand : public Command {
    public:
      CreateMemoryCommand(HostQueue& queue, const Event::EventWaitList& eventWaitList, void* ptr, GraphSlab* slab, Memory* base_memory, size_t size, size_t offset)
	:
	  Command(queue, 0, eventWaitList, 0, nullptr),
	  ptr_(ptr),
	  slab_(slab),
	  base_memory_(base_memory),
	  size_(size),
	  offset_(offset) {}

      virtual void submit(device::VirtualDevice& device) final {
	size_t offset = 0;
	auto prev_memory = MemObjMap::FindMemObj(ptr_, &offset);
	if (prev_memory != nullptr && offset == 0) {
	  return;
	}

	auto memory = new (base_memory_->getContext()) Buffer(*base_memory_, 0, offset_, size_);
	if (nullptr == memory || !memory->create(nullptr)) {
	  LogError("Creating memory inside slab failed.");
	  return;
	}
	MemObjMap::AddMemObj(ptr_, memory);
	if (memory->getUserData().data == nullptr) {
	  memory->getUserData().data = slab_;
	}
      }
    private:
      void* ptr_;
      GraphSlab* slab_;
      Memory* base_memory_;
      size_t size_;
      size_t offset_;
  };

  // A command that allocates and maps physical chunks of memory  
  class CommitMemoryCommand : public VirtualMapCommand {
    public:
      CommitMemoryCommand(HostQueue& queue, const Event::EventWaitList& eventWaitList, void* ptr, size_t size, Memory* memory, GraphSlab* slab, GraphVmHeap* vm_heap)
	: VirtualMapCommand(queue, eventWaitList, ptr, size, memory),
	  vm_heap_(vm_heap),
	  slab_(slab),
	  mptr_(ptr) {}

      virtual void submit(device::VirtualDevice& device) final {
	if (slab_->IsMapped()) {
	  return;
	}

	slab_->Lock();

	if (slab_->IsMapped()) {
	  slab_->Unlock();
	  return;
	}

	const auto& dev_info = vm_heap_->device_->info();

	// Allocate physical memory
	void* dptr = SvmBuffer::malloc(vm_heap_->device_->context(), ROCCLR_MEM_PHYMEM, size_,
				       dev_info.memBaseAddrAlign_, nullptr);
	if (dptr == nullptr) {
	  LogPrintfError("Failed to allocate physical memory %zd", size_);
	  slab_->Unlock();
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

	slab_->Mapped();
	slab_->Unlock();

	// Update mapped size in GraphVmHeap
	auto mapped_size = vm_heap_->mapped_size_.fetch_add(size_);
	auto prev_max_mapped_size = vm_heap_->max_mapped_size_.load();
	while (prev_max_mapped_size < prev_max_mapped_size + size_ &&
	  !vm_heap_->max_mapped_size_.compare_exchange_weak(prev_max_mapped_size, prev_max_mapped_size + size_)) {}
      }
    private:
      GraphVmHeap* vm_heap_;
      GraphSlab* slab_;
      void* mptr_;
  };

  // A command that unmaps and deallocates physical chunks of memory
  class UncommitMemoryCommand : public VirtualMapCommand {
    public:
      UncommitMemoryCommand(HostQueue& queue, const Event::EventWaitList& eventWaitList, void* ptr, size_t size, Memory* memory, GraphSlab* slab, GraphVmHeap* vm_heap, bool unmap_guaranteed)
        : VirtualMapCommand(queue, eventWaitList, ptr, size, memory),
          vm_heap_(vm_heap),
	  slab_(slab),
          mptr_(ptr),
	  unmap_guaranteed_(unmap_guaranteed) {}

      virtual void submit(device::VirtualDevice& device) final {
        // If free mapped memory lower than the threshold, then stop unmapping
        if (!unmap_guaranteed_ && !vm_heap_->ShouldUnmap()) {
          return;
        }

        Memory* vaddr_sub_obj = MemObjMap::FindMemObj(mptr_);
        Memory* phys_mem_obj = vaddr_sub_obj->getUserData().phys_mem_obj;

        // Unmap the physical memory from a virtual address
        VirtualMapCommand::submit(device);

        vaddr_sub_obj->release();

        // Deallocate physical memory
        SvmBuffer::free(vm_heap_->device_->context(), phys_mem_obj->getSvmPtr());

        vm_heap_->mapped_size_.fetch_sub(size_);

	slab_->Unmapped();
      }
    private:
      GraphVmHeap* vm_heap_;
      GraphSlab* slab_;
      void* mptr_;
      bool unmap_guaranteed_;
  };
public:
  friend GraphVmHeapArray;
  typedef std::function<amd::HostQueue&()> GetQueueFunc;

  GraphVmHeap(Device* device,        //!< GPU device object
         GetQueueFunc get_queue //!< Function to retrieve a map queue
         )
      : GraphVmHeap(device, device->info().globalMemSize_ / 8, get_queue) {
  }

  GraphVmHeap(Device* device,        //!< GPU device object
         size_t  va_size,       //!< The size of the allocated heap (bytes).Virtual address space
         GetQueueFunc get_queue //!< Function to retrieve a map/unmap queue
         );

  //! Heap destructor
  virtual ~GraphVmHeap();

  //! Release all heap blocks
  void FreeAllMemory();

  //! Unmaps freed memory based on the threshold
  void TrimPhysMemory(size_t unmap_threshold);

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

  Command* GetSlabMapCommand(GraphSlab* slab, HostQueue& queue);

  Command* GetSlabUnmapCommand(GraphSlab* slab, HostQueue& queue, bool unmap_guaranteed = false);

  Command* GetAllocationCommand(GraphSlab* slab, HostQueue& queue, Command* wait_command, size_t size, size_t offset);

  GraphSlab* AllocateSlab(size_t size, size_t num_peers, size_t slab_id);

  GraphSlab* FetchSlab(size_t slab_id);

  void FreeSlab(GraphSlab* slab);

  bool ShouldUnmap() {
    auto busy_size = va_size_ - free_size_;
    uint64_t free_mapped = mapped_size_.load() - busy_size;
    return free_mapped > unmap_threshold_;
  }
private:
  GraphVmHeap() = delete;
  GraphVmHeap(const GraphVmHeap&) = delete;
  GraphVmHeap& operator=(const GraphVmHeap&) = delete;

  //! Ceates heap object. Reserves virtual address range for the heap operation
  bool Create();

  //! Reseves address range for memory allocations
  address ReserveAddressRange(address start, size_t size);

  //! Releases address range specified by the address
  bool ReleaseAddressRange(void* addr);

  void UnmapSlab(GraphSlab* slab);

  GraphSlab* GetSlab(size_t size);

  void ReturnSlab(GraphSlab* slab, bool in_busy=true);

  void CacheSlab(GraphSlab* slab);

  GraphSlab* PopBin(std::vector<GraphSlab*>& bins, size_t bin_idx);

  void RemoveBin(std::vector<GraphSlab*>& bins, size_t bin_idx, GraphSlab* slab);

  void PushBin(std::vector<GraphSlab*>& bins, size_t bin_idx, GraphSlab* slab);

  GraphSlab* SplitSlab(GraphSlab* slab, size_t bin_idx, bool cached);

  GraphSlab* CoalesceSlab(GraphSlab* slab1, GraphSlab* slab2);

  //! Returns a queue for VM map/unmap operations
  amd::HostQueue& GetVmQueue() const { return get_vm_queue_(); }

  static constexpr size_t kMinPow2_ = (12); // 4KB
  static constexpr size_t kMinSplitSize_ = (1 << kMinPow2_);
  size_t max_bin_idx_;
  std::vector<GraphSlab*> free_bins_;
  std::vector<GraphSlab*> busy_bins_;
  std::vector<GraphSlab*> cache_bins_;
  std::map<size_t, GraphSlab*> slab_ids;

  address       base_address_ = nullptr;  //!< GPU virtual address base of the heap
  Memory*  base_memory_ = nullptr;   //!< VA space base object, used in the view creation
  size_t        free_size_ = 0;           //!< Total free size of the heap (both mapped and unmapped)
  size_t        va_size_ = 0;             //!< Heap virtual address space size
  size_t        block_alignment_ = 8;     //!< Size of an allocation page
  uint64_t      unmap_threshold_ = 0;     //!< Unmap threshold in bytes,used to release phys memory
  std::atomic<uint64_t>      mapped_size_ = 0;         //!< Size of mapped memory
  std::atomic<uint64_t>      max_mapped_size_ = 0;     //!< Max size of mapped memory in this heap
  uint64_t      max_total_size_ = 0;	  //!< Max allowed size in this heap
  bool          created_ = false;         //!< Used for deferred VM heap allocation
  amd::Monitor  lock_;                    //!< Lock to serialise heap accesses
  Device*       device_;                  //!< Device that owns this heap
  GetQueueFunc  get_vm_queue_;            //!< Queue for VM operations
};

class GraphTemporaryHeap {
public:
  GraphTemporaryHeap(Device* device_, size_t num_handles);

  void* Allocate();

  void Invalidate();

  bool InRange(void* ptr);

  bool Created() {
    return created_.load();
  }
private:
  void Create();

  Device* device_;
  std::atomic<size_t> bump_counter_;
  std::atomic<size_t> invalidated_handles_;
  size_t num_handles_;
  char* base_ptr_;
  amd::Monitor lock_;
  std::atomic<bool> created_;
};

//! Implements an array of vm heaps of different sizes for more efficient management
class GraphVmHeapArray {
public:
  GraphVmHeapArray(Device* device,    //!< GPU device object
              GraphVmHeap::GetQueueFunc get_queue  //!< Function to retrieve a map queue
  ) : heap0_(device, device->info().globalMemSize_ / 4, get_queue)
    , heap1_(device, device->info().globalMemSize_ / 4, get_queue)
    , heap2_(device, device->info().globalMemSize_ / 4, get_queue)
    , heap3_(device, device->info().globalMemSize_ / 4, get_queue)
    , large_heap_(device, device->info().globalMemSize_, get_queue)
    , tmp_heap_(device, 32768) //!< 32K handles, should be more than enough
    , device_(device) {}

  //! Checks that ptr was allocated from this GraphVmHeapArray
  bool IsValidAllocation(void* ptr);

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

  //! Gets current size of all heaps (both mapped and unmapped)
  size_t TotalSize() const;

  //! Sets maximum size of all heaps (both mapped and unmapped) 
  void SetMaxTotalSize(size_t value); 

  //! Gets maximum size of all heaps (both mapped and unmapped)  
  size_t MaxTotalSize() const;

  Command* GetSlabMapCommand(GraphSlab* slab, HostQueue& queue);

  Command* GetSlabUnmapCommand(GraphSlab* slab, HostQueue& queue);

  Command* GetAllocationCommand(GraphSlab* slab, HostQueue& queue, Command* wait_command, size_t size, size_t offset);

  void* AllocateTemporaryHandle();

  void InvalidateTemporaryHandle();

  GraphSlab* AllocateSlab(size_t size, size_t num_peers, size_t slab_id);

  GraphSlab* FetchSlab(size_t slab_id);

  bool FreeSlab(GraphSlab* slab);

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
  GraphTemporaryHeap tmp_heap_;

  uint64_t unmap_threshold_ = 0;  //!< Unmap threshold in bytes,used to release phys memory
  Device* device_;                //!< Device that owns this heap
};

} // namespace amd
