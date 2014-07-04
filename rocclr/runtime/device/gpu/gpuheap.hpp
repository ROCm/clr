//! Declarations for GPU memory management

#ifndef GPUHEAP_HPP_
#define GPUHEAP_HPP_

#include "top.hpp"
#include "thread/atomic.hpp"
#include "device/gpu/gpudefs.hpp"

/*! \addtogroup GPU
 *  @{
 */

//! GPU Device Implementation

namespace gpu {

class Device;
class Heap;
class Resource;
class Memory;
class VirtualGPU;

//! @todo:dgladdin: The heap list should be singly-linked

//! \brief A block on the GPU heap.
//!
//! Note that no code outside of the gpumemory.hpp/.cpp pair should touch this
//! class directly as it is not thread-safe. In general, this class should be
//! pretty much a struct and contain as little functionality as possible - just
//!  a constructor, destructor.
//!
//! Any other methods - in particular, anything that talks to CAL - should be no
//! more than proxies for functionality implemented in Heap, as Heap is aware
//! of the lock state.

class HeapBlock : public amd::HeapObject
{
public:
    //! Constructor
    HeapBlock(
        Heap* owner = NULL,
        size_t size = 0,
        size_t offset = 0,
        HeapBlock* next=NULL,
        HeapBlock* prev=NULL)
        : owner_(owner)
        , size_(size)
        , offset_(offset)
        , next_(next)
        , prev_(prev)
        , inUse_(false)
        , parent_(NULL)
        , memory_(NULL)
        {}

    //! Destructor does some sanity checks.
    ~HeapBlock();

    //! Frees a heap block, returning its memory to the owning heap (proxy)
    void free();

    //! Sets the GPU memory object associated with the heap block
    void setMemory(Memory* memory) { memory_ = memory; }

    //! Gets the GPU memory object associated with the heap block
    Memory* getMemory() const { return memory_; }

    //! Adds a heapblock view to the list of views
    void addView(HeapBlock* hb)
        { views_.push_back(hb);  hb->parent_ = this; }

    //! Removes a heapblock view from the list of views
    void removeView(HeapBlock* hb) { views_.remove(hb); }

    //! Destroys all views
    void destroyViewsMemory();

    //! Creates all new views
    bool reallocateViews(
        HeapBlock*  parent,     //!< Parent heap block
        size_t      shift       //!< The new HeapBlock shift
        );

    //! Gets the offset
    size_t offset() const { return offset_; }

    Heap*       owner_;     //!< Heap that owns this block
    size_t      size_;      //!< Size of the block in bytes
    size_t      offset_;    //!< Offset of this block in the heap
    HeapBlock*  next_;      //!< Next block on the list, or NULL
    HeapBlock*  prev_;      //!< Previous block on the list, or NULL
    bool        inUse_;     //!< true if the block is in use
    HeapBlock*  parent_;    //!< The parent heap block for a view

private:
    //! Disable copy constructor
    HeapBlock(const HeapBlock&);

    //! Disable assignment
    HeapBlock& operator=(const HeapBlock&);

    Memory*     memory_;    //!< Memory object associated with the heap block
    std::list<HeapBlock*>   views_; //!< The list of all allocated views
};

class Heap : public amd::HeapObject
{
public:
    //! Minimal supported CAL granularity = 256 bytes / ElementSize
    static const size_t MinGranularity = 64;

    //! The size of a heap element in bytes
    static const size_t ElementSize = 4;

    //! The type of a heap element in bytes
    static const cmSurfFmt ElementType = CM_SURF_FMT_R32I;

    Heap(
        Device& device      //!< GPU device object
        );

    virtual bool create(
        size_t  totalSize,  //!< total size of the allocated heap (bytes)
        bool    remoteAlloc //!< allocate the heap in remote memory
        );

    //! Heap destructor
    virtual ~Heap();

    /*!
     * \brief Allocates memory from a heap (best-fit).
     * We round up to 4k granularity for alignment.
     *
     * \return A pointer to allocated heap block object.
     */
    virtual HeapBlock* alloc(
        size_t size     //! The allocation size
        );

    //! Release memory back to a heap.
    virtual void free(HeapBlock* blk);

    //! Copies this heap to another
    virtual bool copyTo(Heap* heap);

    //! Gets the GPU resource associated with the global heap
    const Resource& resource() const { return *resource_; }

    //! Read the page size (bytes)
    size_t granularityB() const;

    //! Read the total free space (bytes)
    size_t freeSpace() const { return freeSize_; }

    virtual bool isSane(void) const;    //!< Checks heap sanity

    //! Returns true if we have a virtual heap
    bool isVirtual() const { return virtualMode_; }

    //! Returns the base virtual address of the heap
    uint64_t baseAddress() const { return baseAddress_; }

private:
    //! Insert a block into a list. Must be called under a lock.
    void insertBlock(HeapBlock** list, HeapBlock* node);

    //! Merge a block into a list. Must be called under a lock.
    void mergeBlock(HeapBlock** list, HeapBlock* node);

    //! Remove a block from a list. Must be called under a lock.
    void detachBlock(HeapBlock** list, HeapBlock* node);

    //! Split a block into two pieces
    HeapBlock* splitBlock(HeapBlock* node, size_t size);

protected:
    Resource*   resource_;      //!< GPU resource referencing the heap memory
    HeapBlock*  freeList_;      //!< Head block for free list
    HeapBlock*  busyList_;      //!< Head block for busy list
    size_t      freeSize_;      //!< total free size of the heap
    Device&     device_;        //!< Device that owns this heap
    size_t      granularity_;   //!< Size of an allocation page
    amd::Monitor    lock_;      //!< Lock to serialise heap accesses
    bool        virtualMode_;   //!< Virtual mode
    uint64_t    baseAddress_;   //!< Virtual heap base address
};

class VirtualHeap : public Heap
{
public:
    VirtualHeap(
        Device& device      //!< GPU device object
        );

    virtual bool create(
        size_t  totalSize,  //!< total size of the allocated heap (bytes)
        bool    remoteAlloc //!< allocate the heap in remote memory
        );

    //! Heap destructor
    virtual ~VirtualHeap();

    /*!
     * \brief Allocates memory from a heap (best-fit).
     * We round up to 4k granularity for alignment.
     *
     * \return A pointer to allocated heap block object.
     */
    virtual HeapBlock* alloc(
        size_t size     //! The allocation size
        );

    //! Release memory back to a heap.
    virtual void free(HeapBlock* blk);

    //! Copies this heap to another
    virtual bool copyTo(Heap* heap);

    virtual bool isSane(void) const;    //!< Checks heap sanity
};

} // namespace gpu

#endif // GPUHEAP_HPP_
