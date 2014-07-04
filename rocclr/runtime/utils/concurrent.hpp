//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CONCURRENT_HPP_
#define CONCURRENT_HPP_

#include "top.hpp"
#include "thread/atomic.hpp"
#include "os/alloc.hpp"

#include <new>

//! \addtogroup Utils

namespace amd {/*@{*/

namespace details {

template <typename T, int N>
struct TaggedPointerHelper
{
    static const uintptr_t TagMask = (1u << N) - 1;

private:
    TaggedPointerHelper();        // Cannot instantiate
    void* operator new(size_t);   // allocate or
    void operator delete(void*);  // delete a TaggedPointerHelper.

public:
    //! Create a tagged pointer.
    static TaggedPointerHelper* make(T* ptr, size_t tag)
    {
        return reinterpret_cast<TaggedPointerHelper*>(
            (reinterpret_cast<uintptr_t>(ptr) & ~TagMask) | (tag & TagMask));
    }

    //! Return the pointer value.
    T* ptr()
    {
        return reinterpret_cast<T*>(
            reinterpret_cast<uintptr_t>(this) & ~TagMask);
    }

    //! Return the tag value.
    size_t tag() const
    {
        return reinterpret_cast<uintptr_t>(this) & TagMask;
    }
};

} // namespace details

/*! \brief An unbounded thread-safe queue.
 *
 * This queue orders elements first-in-first-out. It is based on the algorithm
 * "Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue
 * Algorithms by Maged M. Michael and Michael L. Scott.".
 *
 * FIXME_lmoriche: Implement the new/delete operators for SimplyLinkedNode
 * using thread-local allocation buffers.
 */
template <typename T, int N = 5>
class ConcurrentLinkedQueue : public HeapObject
{
    //! A simply-linked node
    struct Node
    {
        typedef details::TaggedPointerHelper<Node,N> TaggedPointerHelper;
        typedef TaggedPointerHelper* Ptr;

        T value_;          //!< The value stored in that node.
        Atomic<Ptr> next_; //!< Pointer to the next node

        //! Create a Node::Ptr
        static inline Ptr ptr(Node* ptr, size_t counter = 0)
        {
            return TaggedPointerHelper::make(ptr, counter);
        }
    };

private:
    Atomic<typename Node::Ptr> head_; //! Pointer to the oldest element.
    Atomic<typename Node::Ptr> tail_; //! Pointer to the most recent element.

private:
    //! \brief Allocate a free node.
    static inline Node* allocNode()
    {
        return new(AlignedMemory::allocate(sizeof(Node), 1 << N)) Node();
    }

    //! \brief Return a node to the free list.
    static inline void reclaimNode(Node* node)
    {
        AlignedMemory::deallocate(node);
    }

public:
    //! \brief Initialize a new concurrent linked queue.
    ConcurrentLinkedQueue();

    //! \brief Destroy this concurrent linked queue.
    ~ConcurrentLinkedQueue();

    //! \brief Enqueue an element to this queue.
    inline void enqueue(T elem);

    //! \brief Dequeue an element from this queue.
    inline T dequeue();
};

/*@}*/

template <typename T, int N>
inline
ConcurrentLinkedQueue<T,N>::ConcurrentLinkedQueue()
{
    // Create the first "dummy" node.
    Node* dummy = allocNode();
    dummy->next_ = NULL;
    DEBUG_ONLY(dummy->value_ = NULL);

    // Head and tail should now point to it (empty list).
    head_ = tail_ = Node::ptr(dummy);

    // Make sure the instance is fully initialized before it becomes
    // globally visible.
    MemoryOrder::sfence();
}

template <typename T, int N>
inline
ConcurrentLinkedQueue<T,N>::~ConcurrentLinkedQueue()
{
    typename Node::Ptr head = head_;
    typename Node::Ptr tail = tail_;
    while (head->ptr() != tail->ptr()) {
        Node* node = head->ptr();
        head = head->ptr()->next_;
        reclaimNode(node);
    }
    reclaimNode(head->ptr());
}

template <typename T, int N>
inline void
ConcurrentLinkedQueue<T,N>::enqueue(T elem)
{
    Node* node = allocNode();
    node->value_ = elem;
    node->next_ = NULL;

    while (true) {
        typename Node::Ptr tail = tail_;
        typename Node::Ptr next = tail->ptr()->next_;
        MemoryOrder::lfence();
        if (tail == tail_) {
            if (next->ptr() == NULL) {
                if (tail->ptr()->next_.compareAndSet(
                        next, Node::ptr(node, next->tag()+1))) {
                    tail_.compareAndSet(tail, Node::ptr(node, tail->tag()+1));
                    return;
                }
            }
            else {
                tail_.compareAndSet(
                    tail, Node::ptr(next->ptr(), tail->tag()+1));
            }
        }
    }
}

template <typename T, int N>
inline T
ConcurrentLinkedQueue<T,N>::dequeue()
{
    while (true) {
        typename Node::Ptr head = head_;
        typename Node::Ptr tail = tail_;
        typename Node::Ptr next = head->ptr()->next_;
        MemoryOrder::lfence();
        if (head == head_) {
            if (head->ptr() == tail->ptr()) {
                if (next->ptr() == NULL) {
                    return NULL;
                }
                tail_.compareAndSet(
                    tail, Node::ptr(next->ptr(), tail->tag()+1));
            }
            else {
                T value = next->ptr()->value_;
                if (head_.compareAndSet(
                        head, Node::ptr(next->ptr(), head->tag()+1))) {
                    // we can reclaim head now
                    reclaimNode(head->ptr());
                    return value;
                }
            }
        }
    }
}

} // namespace amd

#endif /*CONCURRENT_HPP_*/
