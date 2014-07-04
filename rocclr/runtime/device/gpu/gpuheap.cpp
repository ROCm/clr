//! Implementation of GPU device memory management

#include "top.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "device/device.hpp"
#include "device/gpu/gpuheap.hpp"
#include "device/gpu/gpudevice.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

//! Turn this on to enable sanity checks before and after every heap operation.
#if DEBUG
#define EXTRA_HEAP_CHECKS   1
#endif // DEBUG

namespace gpu {

// The GPU heap. Very simple implementation for now.
Heap::Heap(
    Device& device)
    : resource_(NULL)
    , freeList_(NULL)
    , busyList_(NULL)
    , freeSize_(0)
    , device_(device)
    , granularity_(Heap::MinGranularity)
    , lock_("GPU heap lock", true)
    , virtualMode_(false)
    , baseAddress_(0)
{
}

size_t
Heap::granularityB() const
{
    return granularity_ * Heap::ElementSize;
}

bool
Heap::create(size_t totalSize, bool remoteAlloc)
{
    Resource::MemoryType    memType;
    size_t  maxHeight = device_.info().image2DMaxHeight_;
    size_t  sizeInElements;
    size_t  npages;

    freeSize_ = totalSize;

    sizeInElements = (totalSize + Heap::ElementSize - 1) / Heap::ElementSize;

    // Calculate best granularity given the size and device characteristics
    npages = amd::alignUp(sizeInElements, granularity_) / granularity_;

    // Create a new GPU resource
    resource_ = new Resource(device_, sizeInElements, Heap::ElementType);

    if (resource_ == NULL) {
        return false;
    }

    memType = (remoteAlloc) ? Resource::RemoteUSWC : Resource::Local;

    if (!resource_->create(memType, NULL, true)) {
        return false;
    }

    // Set up initial free list
    freeList_ = new HeapBlock(this, npages * granularityB(), 0, NULL, NULL);
    if (freeList_ == NULL) {
        return false;
    }

    guarantee(isSane());
    return true;
}

Heap::~Heap()
{
    amd::ScopedLock k(lock_);

    guarantee(isSane());

    // Release all heap blocks
    HeapBlock *walk, *next;
    walk = busyList_;
    while (walk) {
        next = walk->next_;
        walk->free();
        walk = next;
    }

    walk = freeList_;
    while (walk) {
        next = walk->next_;
        delete walk;
        walk = next;
    }

    // Release resource
    delete resource_;
}

HeapBlock*
Heap::alloc(size_t size)
{
    amd::ScopedLock k(lock_);
    HeapBlock* walk = freeList_;
    HeapBlock* best = NULL;

    guarantee(isSane());

    // Round size
    size = amd::alignUp(size, granularityB());

    // Walk the free list looking for a suitable block (currently best-fit)
    //! @todo:dgladdin: experiment with switching back to first-fit

    while (walk) {
        if ((walk->size_ > size) &&
            (best == NULL || walk->size_ < best->size_)) {
                best = walk;
        }
        else if (walk->size_ == size) {
            // No need to split, just move to busy list
            detachBlock(&freeList_, walk);
            walk->inUse_ = true;
            insertBlock(&busyList_, walk);
            guarantee(isSane());
            freeSize_ -= size;
            return walk;
        }
    walk = walk->next_;
    }

    if (best != NULL) {
        // Got one, but need to split it. Keep first part in free list,
        // put second part into busy list.
        HeapBlock *newblock = splitBlock(best, size);
        newblock->inUse_ = true;
        insertBlock(&busyList_, newblock);
        guarantee(isSane());
        freeSize_ -= size;
        return newblock;
    }

    // No free block available
    guarantee(isSane());
    return NULL;
}

bool
Heap::copyTo(Heap* heap)
{
    HeapBlock    *walk;

    walk = busyList_;
    while (walk) {
        if (walk->getMemory() != NULL) {
            HeapBlock* hb = heap->alloc(walk->size_);
            if (hb == NULL) {
                return false;
            }
            hb->setMemory(walk->getMemory());

            walk->destroyViewsMemory();
            if (!walk->getMemory()->reallocate(hb, &(heap->resource()))) {
                return false;
            }

            if (!walk->reallocateViews(hb,
                    static_cast<size_t>(hb->offset_ - walk->offset_))) {
                return false;
            }
        }
        walk = walk->next_;
    }

    return true;
}

void
Heap::free(HeapBlock* blk)
{
    amd::ScopedLock k(lock_);
    guarantee(isSane());
    detachBlock(&busyList_, blk);
    blk->inUse_ = false;
    freeSize_ += blk->size_;
    mergeBlock(&freeList_, blk);
    guarantee(isSane());
}

void
Heap::detachBlock(HeapBlock** list, HeapBlock* blk)
{
    // Sanity checks
    guarantee(isSane());

    if (*list == blk) {
        *list = blk->next_;
    }

    if (blk->prev_) {
       blk->prev_->next_ = blk->next_;
    }
    if (blk->next_) {
        blk->next_->prev_ = blk->prev_;
    }
    // no heap sanity check as blk is now floating
}

void
Heap::insertBlock(HeapBlock** head, HeapBlock* blk)
{
     if (NULL == *head) {
        *head = blk;
        blk->prev_ = NULL;
        blk->next_ = NULL;
        guarantee(isSane());
        return;
    }

    // Find the place to insert it at
    HeapBlock* walk = *head;
    while (walk->next_ && walk->next_->offset_ < blk->offset_) {
        walk = walk->next_;
    }

    // Insert it
    if (walk == *head) {
        if (walk->offset_ >= blk->offset_) {
            *head = blk;
            blk->prev_ = NULL;
            blk->next_ = walk;
            walk->prev_ = *head;
            guarantee(isSane());
            return;
        }
    }

    blk->next_ = walk->next_;
    blk->prev_ = walk;
    if (walk->next_) {
        walk->next_->prev_ = blk;
    }
    walk->next_ = blk;
    guarantee(isSane());
}

HeapBlock*
Heap::splitBlock(HeapBlock* blk, size_t tailsize)
{
    // Sanity checks

    guarantee(isSane());
    guarantee(blk->size_ > tailsize && "block too small to split as requested");
    guarantee(!blk->inUse_ && "can't split in-use block");

    // Create a new block

    HeapBlock* nb = new HeapBlock(blk->owner_, tailsize,
                                  blk->offset_ + blk->size_ - tailsize);

    // Resize the old block

    blk->size_ = blk->size_ - tailsize;
    return nb;  // no heap sanity check here as the new block hasn't been plugged in yet
}

//! Join two blocks, transferring the size of the second into the first and deleting
//! the second. Utility fn for mergeBlock()

static void
join2Blocks(HeapBlock* first, HeapBlock* second)
{
    // Sanity checks

    guarantee(first->size_ > 0 && "first block invalid");
    guarantee(!first->inUse_ && "can't join  an in-use block");
    guarantee(second->size_ > 0 && "second block invalid");
    guarantee(first->offset_ + first->size_ == second->offset_);

    // Do the join
    first->size_ = first->size_ + second->size_;
    first->next_ = second->next_;
    if (second->next_) {
        second->next_->prev_ = first;
    }
    delete second;
}

//! Insert a block into a list, merging it with adjacent blocks if possible. Must be called
//! under a lock, cannot be used on in-use blocks or blocks with an associated resource alias.

void
Heap::mergeBlock(HeapBlock** head, HeapBlock* blk)
{
    insertBlock(head, blk);

    // Merge with successor if possible
    if ((blk->next_ != NULL) &&
        (blk->offset_ + blk->size_ == blk->next_->offset_)) {
        join2Blocks(blk, blk->next_);
    }

    // Merge with predecessor if possible
    if ((blk->prev_ != NULL) &&
        (blk->prev_->offset_ + blk->prev_->size_ == blk->offset_)) {
        join2Blocks(blk->prev_, blk);
    }

    guarantee(isSane());
}

//! Sanity check for both types of block (helper function for Heap::isSane())

static bool
isBlockSane(HeapBlock* b)
{
    return (b->owner_ != NULL
        && (b->next_ == NULL || b->next_->prev_ == b)
        && (b->prev_ == NULL || b->prev_->next_ == b));
}

//! Sanity check for an individual free block (helper function for Heap::isSane())
static bool
isFreeBlockSane(HeapBlock* b)
{
    if (isBlockSane(b) && !b->inUse_) {
        return true;
    } else {
        return false;
    }
}

//! Sanity check for an individual busy block (helper function for Heap::isSane())
static bool
isBusyBlockSane(HeapBlock* b)
{
    if (isBlockSane(b) && b->inUse_) {
        return true;
    } else {
        return false;
    }
}

//! Sanity check for the heap.

bool
Heap::isSane() const
{
    // If we got this far, everything is (probably) OK
#if EXTRA_HEAP_CHECKS
    HeapBlock* walkFree = freeList_;    // Free list position
    HeapBlock* walkBusy = busyList_;    // Busy list position
    size_t offset = 0;                  // Current offset

    // We can have zero lists if Heap allocation fails
    if (walkFree == NULL && walkBusy == NULL) {
        return true;
    }

    // Walk both lists in parallel
    while (walkFree != NULL || walkBusy != NULL) {
        if (walkFree != NULL && walkFree->offset_ == offset) {
            if (!isFreeBlockSane(walkFree)) {
                return false;
            }
            offset += walkFree->size_;
            walkFree = walkFree->next_;
        }
        else if (walkBusy != NULL && walkBusy->offset_ == offset) {
            if (!isBusyBlockSane(walkBusy)) {
                return false;
            }
            offset += walkBusy->size_;
            walkBusy = walkBusy->next_;
        }
        else {
            return false;
        }
    }

#endif // EXTRA_HEAP_CHECKS
    return true;
}

void
HeapBlock::destroyViewsMemory()
{
    if ((parent_ != NULL) && (0 == views_.size())) {
        memory_->free();
    }
    else if (views_.size() != 0) {
        std::list<HeapBlock*>::const_iterator it;
        for (it = views_.begin(); it != views_.end(); ++it) {
            (*it)->destroyViewsMemory();
        }
    }
}

bool
HeapBlock::reallocateViews(HeapBlock* parent, size_t shift)
{
    if (views_.size() != 0) {
        std::list<HeapBlock*>::const_iterator it;

        // Loop through all views and reallocate them
        for (it = views_.begin(); it != views_.end(); ++it) {
            // Get the view HeapBlock
            HeapBlock* hb = (*it);

            // Readjust the offset
            hb->offset_ += shift;
            // Add to the list if we have a new parent
            if (parent != this) {
                parent->addView(hb);
            }

            // Reallocate memory
            hb->memory_->reallocate(hb, parent->getMemory());

            // Process a view on view if available
            if (!hb->reallocateViews(hb, shift)) {
                return false;
            }
        }

        // Destroy old list
        if (parent != this) {
            views_.clear();
        }
    }
    return true;
}

//! Destructor. Frees the block if in use and does some final sanity checks.
HeapBlock::~HeapBlock()
{
    if (NULL != owner_) {
        if (inUse_) {
            owner_->free(this);
        }
    }
    else {
        // View destruction
        if (parent_ != NULL) {
            assert(((parent_->getMemory() != NULL) && (parent_->getMemory()->owner() != NULL)));
            amd::ScopedLock lock(parent_->getMemory()->owner()->lockMemoryOps());
            parent_->removeView(this);
        }
    }
    guarantee(size_ > 0 && "destructor called for zero-size heap block (destructor called twice?)");
    size_ = 0; // Mark as invalid

    if (views_.size() != 0) {
        LogError("Can't destroy a resource if we still have views!");
    }
}

void
HeapBlock::free()
{
    if (NULL != owner_) {
        owner_->free(this);
    }
    else {
        // It's a view. Destroy the object
        delete this;
    }
}

VirtualHeap::VirtualHeap(
    Device& device)
    : Heap(device)
{
    virtualMode_ = true;
}

bool
VirtualHeap::create(
    size_t  totalSize,
    bool    remoteAlloc)
{
    // Create a new GPU resource
    resource_ = new Resource(device_, 0, Heap::ElementType);
    if (resource_ == NULL) {
        return false;
    }

    if (!resource_->create(Resource::Heap)) {
        return false;
    }

    if (!device_.settings().hsail_) {
        baseAddress_ = resource_->gslResource()->getSurfaceAddress();
    }
    return true;
}

VirtualHeap::~VirtualHeap()
{
}

HeapBlock*
VirtualHeap::alloc(size_t size)
{
    assert(false && "Dead branch!");
    return NULL;
}

void
VirtualHeap::free(HeapBlock* blk)
{
    assert(false && "Dead branch!");
}

bool
VirtualHeap::copyTo(Heap* heap)
{
    assert(false && "Dead branch!");
    return false;
}

bool
VirtualHeap::isSane(void) const
{
    assert(false && "Dead branch!");
    return true;
}

} // namespace gpu
