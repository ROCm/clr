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

#include "top.hpp"
#include "device/device.hpp"
#include "object.hpp"
#include "commandqueue.hpp"

namespace amd {
class VmHeap;
class HeapBlock;

class HeapBlock : public amd::HeapObject {
public:
  friend VmHeap;
  //! Constructor
  HeapBlock(
      VmHeap* owner = nullptr,  //!< VmHeap object that owns this heap block
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
  ~HeapBlock() { assert(!busy_ && "The blocked must be destroyed explicitly!"); }

  //! Gets the offset
  size_t Offset() const { return offset_; }

private:
  HeapBlock() = delete;
  HeapBlock(const HeapBlock&) = delete;
  HeapBlock& operator=(const HeapBlock&) = delete;

  VmHeap*     owner_;   //!< Heap that owns this block
  size_t      size_;    //!< Size of the block in bytes
  size_t      offset_;  //!< Offset of this block in the heap
  HeapBlock*  next_;    //!< Next block on the list, or nullptr
  HeapBlock*  prev_;    //!< Previous block on the list, or nullptr
  bool        busy_;    //!< True if the block is in use
};

class VmHeap {
public:
  static const size_t kChunkSize = 32 * Mi; //!< Chunk size, must be power of 2
  static const size_t kMinBlockAlignment = 256;
  VmHeap(Device* device,    //!< GPU device object
         HostQueue& queue   //!< Queue, used for map/unmap of physical memory
         )
      : VmHeap(device, queue, device->info().globalMemSize_ / 8, kChunkSize) {}

  VmHeap(Device* device,    //!< GPU device object
         HostQueue& queue,  //!< Queue, usde for map/unmap of physical memory
         size_t  va_size,   //!< The size of the allocated heap (bytes).Virtual address space
         size_t  chunk_size //!< The size of single chunk for physical memory growth
         );

  //! Ceates heap object. Reserves virtual address range for the heap operation
  bool Create();

  //! Heap destructor
  virtual ~VmHeap();

  //! Returns a pointer to the allocated device memory from a heap
  address Alloc(
      size_t size     //! The allocation size
      );

  //! Release memory back to the VM heap
  void Free(amd::Memory* memory);

  //! Unmaps freed memory based on the threshold
  void TrimPhysMemory(size_t unmap_threshold);

  //! Enable memory unmap threashold (default 0 unmap always)
  void SetUnmapThreshold(uint64_t threshold) { unmap_threshold_ = threshold; }

  //! Returns mapped memory size (total allocated physical memory)
  uint64_t MappedSize() const { return mapped_size_; }

  //! Returns mapped memory size (allocated physical memory) without actual allocations
  uint64_t FreeMappedSize() const { return mapped_size_ - (va_size_ - free_size_); }

private:
  VmHeap() = delete;
  VmHeap(const VmHeap&) = delete;
  VmHeap& operator=(const VmHeap&) = delete;

  //! Reseves address range for memory allocations
  address ReserveAddressRange(address start, size_t size, size_t alignment);

  //! Releases address range specified by the address
  bool ReleaseAddressRange(void* addr);

  //! Commits actual physical memory on the specified address
  bool CommitMemory(void* addr, size_t size);

  //! Uncommits physical memory from the spcified address
  bool UncommitMemory(void* addr, size_t size);

  HeapBlock* AllocBlock(size_t size  //! The allocation size
                        );

  //! Release memory back to a heap
  void FreeBlock(HeapBlock* blk);

  //! Insert a block into a list
  void InsertBlock(HeapBlock** list, HeapBlock* node);

  //! Merge a block into a list
  void MergeBlock(HeapBlock** list, HeapBlock* node);

  //! Remove a block from a list
  void DetachBlock(HeapBlock** list, HeapBlock* node);

  //! Splits a block into two pieces
  HeapBlock* SplitBlock(HeapBlock* node, size_t size);

  //! Maps physical memory into specified address space
  bool MapPhysMemory(size_t offset, size_t size);

  //! Unmaps physical memory from the specified address
  void UnmapPhysMemory(size_t offset, size_t size);

  //! Join two blocks, transferring the size of the second into the first and deleting the second
  void Join2Blocks(HeapBlock* first, HeapBlock* second) const;

  address       base_address_ = nullptr;  //!< GPU virtual address base of the heap
  amd::Memory*  base_memory_ = nullptr;   //!< VA space base object, used in the view creation
  HeapBlock*    free_list_ = nullptr;     //!< Head block for free list
  HeapBlock*    busy_list_ = nullptr;     //!< Head block for busy list
  size_t        free_size_ = 0;           //!< Total free size of the heap (both mapped and unmapped)
  size_t        va_size_ = 0;             //!< Heap virtual address space size
  size_t        block_alignment_ = 1;     //!< Size of an allocation page
  size_t        chunk_size_ = 0;          //!< Chunk size (min physical allocation for the growth)
  uint64_t      unmap_threshold_ = 0;     //!< Unmap threshold in bytes,used to release phys memory
  uint64_t      mapped_size_ = 0;         //!< Size of mapped memory
  uint64_t      max_mapped_size_ = 0;     //!< Max size of mapped memory in this heap
  bool          created_ = false;         //!< Used for deferred VM heap allocation
  amd::Monitor  lock_;                    //!< Lock to serialise heap accesses
  Device*       device_;                  //!< Device that owns this heap
  HostQueue&    map_queue_;               //!< Queue, used to map/unmap

  std::vector<bool> mapped_mem_;  //!< A map of mapped memory, the size is total_size/chunk_size
};

} // namespace amd
