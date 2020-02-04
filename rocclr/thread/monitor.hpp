/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#ifndef MONITOR_HPP_
#define MONITOR_HPP_

#include "top.hpp"
#include "thread/atomic.hpp"
#include "thread/semaphore.hpp"
#include "thread/thread.hpp"

#include <atomic>
#include <tuple>
#include <utility>

namespace amd {

/*! \addtogroup Threads
 *  @{
 *
 *  \addtogroup Synchronization
 *  @{
 */

namespace details {

template <class T, class AllocClass = HeapObject> struct SimplyLinkedNode : public AllocClass {
  typedef SimplyLinkedNode<T, AllocClass> Node;

 protected:
  std::atomic<Node*> next_; /*!< \brief The next element. */
  T volatile item_;

 public:
  //! \brief Return the next element in the linked-list.
  Node* next() const { return next_; }
  //! \brief Return the item.
  T item() const { return item_; }

  //! \brief Set the next element pointer.
  void setNext(Node* next) { next_ = next; }
  //! \brief Set the item.
  void setItem(T item) { item_ = item; }

  //! \brief Swap the next element pointer.
  Node* swapNext(Node* next) { return next_.swap(next); }

  //! \brief Compare and set the next element pointer.
  bool compareAndSetNext(Node* compare, Node* next) {
    return next_.compare_exchange_strong(compare, next);
  }
};

}  // namespace details

class Monitor : public HeapObject {
  typedef details::SimplyLinkedNode<Semaphore*, StackObject> LinkedNode;

 private:
  static const intptr_t kLockBit = 0x1;

  static const int kMaxSpinIter = 55;      //!< Total number of spin iterations.
  static const int kMaxReadSpinIter = 50;  //!< Read iterations before yielding

  /*! Linked list of semaphores the contending threads are waiting on
   *  and main lock.
   */
  std::atomic_intptr_t contendersList_;
  //! The Mutex's name
  char name_[64];

  //! Semaphore of the next thread to contend for the lock.
  std::atomic_intptr_t onDeck_;
  //! Linked list of the suspended threads resume semaphores.
  LinkedNode* volatile waitersList_;

  //! Thread owning this monitor.
  Thread* volatile owner_;
  //! The amount of times this monitor was acquired by the owner.
  uint32_t lockCount_;
  //! True if this is a recursive mutex, false otherwise.
  const bool recursive_;

 private:
  //! Finish locking the mutex (contented case).
  void finishLock();
  //! Finish unlocking the mutex (contented case).
  void finishUnlock();

 protected:
  //! Try to spin-acquire the lock, return true if successful.
  bool trySpinLock();

  /*! \brief Return true if the lock is owned.
   *
   *  \note The user is responsible for the memory ordering.
   */
  bool isLocked() const { return (contendersList_ & kLockBit) != 0; }

  //! Return this monitor's owner thread (NULL if unlocked).
  Thread* owner() const { return owner_; }

  //! Set the owner.
  void setOwner(Thread* thread) { owner_ = thread; }

 public:
  explicit Monitor(const char* name = NULL, bool recursive = false);
  ~Monitor() {}

  //! Try to acquire the lock, return true if successful.
  inline bool tryLock();

  //! Acquire the lock or suspend the calling thread.
  inline void lock();

  //! Release the lock and wake a single waiting thread if any.
  inline void unlock();

  /*! \brief Give up the lock and go to sleep.
   *
   *  Calling wait() causes the current thread to go to sleep until
   *  another thread calls notify()/notifyAll().
   *
   *  \note The monitor must be owned before calling wait().
   */
  void wait();
  /*! \brief Wake up a single thread waiting on this monitor.
   *
   *  \note The monitor must be owned before calling notify().
   */
  void notify();
  /*! \brief Wake up all threads that are waiting on this monitor.
   *
   *  \note The monitor must be owned before calling notifyAll().
   */
  void notifyAll();

  //! Return this lock's name.
  const char* name() const { return name_; }
};

class ScopedLock : StackObject {
 private:
  Monitor* lock_;

 public:
  ScopedLock(Monitor& lock) : lock_(&lock) { lock_->lock(); }

  ScopedLock(Monitor* lock) : lock_(lock) {
    if (lock_) lock_->lock();
  }

  ~ScopedLock() {
    if (lock_) lock_->unlock();
  }
};

/*! @}
 *  @}
 */

inline bool Monitor::tryLock() {
  Thread* thread = Thread::current();
  assert(thread != NULL && "cannot lock() from (null)");

  intptr_t ptr = contendersList_.load(std::memory_order_acquire);

  if (unlikely((ptr & kLockBit) != 0)) {
    if (recursive_ && thread == owner_) {
      // Recursive lock: increment the lock count and return.
      ++lockCount_;
      return true;
    }
    return false;  // Already locked!
  }

  if (unlikely(!contendersList_.compare_exchange_weak(
          ptr, ptr | kLockBit, std::memory_order_acq_rel, std::memory_order_acquire))) {
    return false;  // We failed the CAS from unlocked to locked.
  }

  setOwner(thread);  // cannot move above the CAS.
  lockCount_ = 1;

  return true;
}

inline void Monitor::lock() {
  if (unlikely(!tryLock())) {
    // The lock is contented.
    finishLock();
  }

  // This is the beginning of the critical region. From now-on, everything
  // executes single-threaded!
  //
}

inline void Monitor::unlock() {
  assert(isLocked() && owner_ == Thread::current() && "invariant");

  if (recursive_ && --lockCount_ > 0) {
    // was a recursive lock case, simply return.
    return;
  }

  setOwner(NULL);

  // Clear the lock bit.
  intptr_t ptr = contendersList_.load(std::memory_order_acquire);
  while (!contendersList_.compare_exchange_weak(ptr, ptr & ~kLockBit, std::memory_order_acq_rel,
                                                std::memory_order_acquire))
    ;
  //
  // We succeeded the CAS from locked to unlocked.
  // This is the end of the critical region.

  // Check if we have an on-deck thread that needs signaling.
  intptr_t onDeck = onDeck_;
  if (onDeck != 0) {
    if ((onDeck & kLockBit) == 0) {
      // Only signal if it is unmarked.
      reinterpret_cast<Semaphore*>(onDeck)->post();
    }
    return;  // We are done.
  }

  // We do not have an on-deck thread yet, we might have to walk the list in
  // order to select the next onDeck_. Only one thread needs to fill onDeck_,
  // so return if the list is empty or if the lock got acquired again (it's
  // somebody else's problem now!)

  intptr_t head = contendersList_;
  if (head == 0 || (head & kLockBit) != 0) {
    return;
  }

  // Finish the unlock operation: find a thread to wake up.
  finishUnlock();
}

}  // namespace amd

#endif /*MONITOR_HPP_*/
