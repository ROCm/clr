/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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
#include "utils/flags.hpp"
#include "thread/semaphore.hpp"
#include "thread/thread.hpp"
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <tuple>
#include <utility>
#include <variant>

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

namespace legacy_monitor {
class Monitor {
  typedef details::SimplyLinkedNode<Semaphore*, StackObject> LinkedNode;

 private:
  static constexpr intptr_t kLockBit = 0x1;

  static constexpr int kMaxSpinIter = 55;      //!< Total number of spin iterations.
  static constexpr int kMaxReadSpinIter = 50;  //!< Read iterations before yielding

  /*! Linked list of semaphores the contending threads are waiting on
   *  and main lock.
   */
  std::atomic_intptr_t contendersList_;

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
  explicit Monitor(bool recursive = false);
  ~Monitor() {}

  //! Try to acquire the lock, return true if successful.
  bool tryLock();

  //! Acquire the lock or suspend the calling thread.
  void lock();

  //! Release the lock and wake a single waiting thread if any.
  void unlock();

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
};


} // namespace legacy_monitor

namespace mutex_monitor {
class Monitor {
 public:
  explicit Monitor(bool recursive = false) : recursive_(recursive) {
    waits_.store(0); // 0 waiting thread initially
    notifyState_.store(notifyState::notNotified); // initially not notified
    if (recursive) {
      mutex_.emplace<std::recursive_mutex>();
    } else {
      mutex_.emplace<std::mutex>();
    }
  }

  //! Try to acquire the lock, return true if successful, false if failed.
  bool tryLock() {
    return recursive_ ? std::get<std::recursive_mutex>(mutex_).try_lock() :
                        std::get<std::mutex>(mutex_).try_lock();
  }

  //! Acquire the lock or suspend the calling thread.
  void lock() {
    recursive_ ? std::get<std::recursive_mutex>(mutex_).lock() :
                 std::get<std::mutex>(mutex_).lock();
  }

  //! Release the lock and wake a single waiting thread if any.
  void unlock() {
    recursive_ ? std::get<std::recursive_mutex>(mutex_).unlock() :
                 std::get<std::mutex>(mutex_).unlock();
  }

  /*! \brief Give up the lock and go to sleep.
   *
   *  Calling wait() causes the current thread to go to sleep until
   *  another thread calls notify()/notifyAll().
   *
   *  \note The monitor must be owned before calling wait().
   */
  void wait() {
    assert(recursive_ == false && "Error: wait() doesn't support recursive mode");
    assert(waits_.load(std::memory_order_acquire) >= 0 && "Error: waits_.load() < 0");
    std::mutex& mut = std::get<std::mutex>(mutex_);
    std::unique_lock lk(mut, std::adopt_lock);

    int c = 0;
    while (unlikely(notifyState_.load(std::memory_order_acquire) == notifyState::allNotified)) {
      lk.unlock();
      // NotifyAll() processing already in progress, don't enter now.
      // The new wait() shoule be processed by next notifyAll().
      if (c < maxReadSpinIter_) {
        Os::spinPause();
        c++;
      }
      // and then SMP friendly
      else {
        Thread::yield();
      }
      lk.lock();
    }
    waits_.fetch_add(1, std::memory_order_acq_rel);

    lk.unlock();
    notifyState expextedNotifyState  = notifyState::oneNotified; // expect that notify() is called
    // fast path
    c = 0;
    while (c < maxCount_ &&
      (notifyState_.load(std::memory_order_acquire) != notifyState::allNotified &&
      !notifyState_.compare_exchange_weak(expextedNotifyState, notifyState::notNotified,
                   std::memory_order_acq_rel,  std::memory_order_acquire))) {
      // First, be SMT friendly
      if (c < maxReadSpinIter_) {
        Os::spinPause();
      }
      // and then SMP friendly
      else {
        Thread::yield();
      }
      c++;
      expextedNotifyState = notifyState::oneNotified;
    }
    assert(c <= maxCount_ && "Error: c > maxCount_");

    lk.lock();

    if (c == maxCount_) {
      // In case notify() is called between loop and here
      expextedNotifyState = notifyState::oneNotified;
      if (notifyState_.load(std::memory_order_acquire) != notifyState::allNotified &&
        !notifyState_.compare_exchange_strong(expextedNotifyState,
         notifyState::notNotified, std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Still not notified, so enter slow path
        cv_.wait(lk); // slow path
        expextedNotifyState = notifyState::oneNotified;
        // To reset notifyState::oneNotified to notifyState::notNotified state if notifyState_ is
        // notifyState::oneNotified.
        // This will happen when notify() is called during cv_.wait(lk). Will do nothing
        // if notifyState_ is notifyState::notNoftifed or notifyState::allNotified.
        notifyState_.compare_exchange_strong(expextedNotifyState, notifyState::notNotified,
          std::memory_order_acq_rel, std::memory_order_acquire);
      }
    }
    // the mutex is locked again before exiting...
    lk.release();  // Release the ownership so that the caller should unlock the mutex
    if (waits_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (notifyState_.load(std::memory_order_acquire) == notifyState::allNotified) {
        // No waiter indicates that notifyAll() processing has ended
        notifyState_.store(notifyState::notNotified, std::memory_order_release);
      }
    }
  }

  /*! \brief Wake up a single thread waiting on this monitor.
   *
   *  \note The monitor need be owned before calling notify().
   */
  void notify() {
    // If notifyState_ is notifyState::oneNotified or notifyState::allNotified, this will be
    // skipped.
    if (notifyState_.load(std::memory_order_acquire) == notifyState::notNotified &&
        waits_.load(std::memory_order_acquire) > 0 ) {
      notifyState_.store(notifyState::oneNotified, std::memory_order_release);
      cv_.notify_one();
    }
  }

  /*! \brief Wake up all threads that are waiting on this monitor.
   *
   *  \note The monitor need be owned before calling notifyAll().
   */
  void notifyAll() {
    // If notifyState_ is notifyState::allNotified, this will be skipped.  So notifyAll()
    // can still be called if notify() is just called as notifyAll() covers notify()
    if ( notifyState_.load(std::memory_order_acquire) != notifyState::allNotified &&
         waits_.load(std::memory_order_acquire) > 0 ) {
      // One notification is enough
      notifyState_.store(notifyState::allNotified, std::memory_order_release);
      cv_.notify_all();
    }
  }

 private:

  std::variant<std::monostate, std::mutex, std::recursive_mutex> mutex_;

  enum class notifyState{
    notNotified = 0,
    oneNotified = 1,
    allNotified = 2
  };
  std::condition_variable cv_; //!< The condition variable for sync on the mutex
  const bool recursive_; //!< True if this is a recursive mutex, false otherwise.
  std::atomic<int> waits_;
  std::atomic<notifyState> notifyState_;
  const int maxCount_{ 55 }; //!< Max count of spins in wait()
  const int maxReadSpinIter_{ 50 };
};
} // namespace mutex_monitor

// Monitor API wrapper to user
class Monitor {
public:
  explicit Monitor(bool recursive = false){
    if (mode_) {
      monitor_.emplace<mutex_monitor::Monitor>(recursive);
    } else {
      monitor_.emplace<legacy_monitor::Monitor>(recursive);
    }
  }
  inline bool tryLock() {
    return mode_ ? std::get<mutex_monitor::Monitor>(monitor_).tryLock() :
                   std::get<legacy_monitor::Monitor>(monitor_).tryLock();
  }
  inline void lock() {
    mode_ ? std::get<mutex_monitor::Monitor>(monitor_).lock() :
            std::get<legacy_monitor::Monitor>(monitor_).lock();
  }
  inline void unlock() {
    mode_ ? std::get<mutex_monitor::Monitor>(monitor_).unlock() :
            std::get<legacy_monitor::Monitor>(monitor_).unlock();
  }
  inline void wait() {
    mode_ ? std::get<mutex_monitor::Monitor>(monitor_).wait() :
            std::get<legacy_monitor::Monitor>(monitor_).wait();
  }
  inline void notify() {
    mode_ ? std::get<mutex_monitor::Monitor>(monitor_).notify() :
            std::get<legacy_monitor::Monitor>(monitor_).notify();
  }
  inline void notifyAll() {
    mode_ ? std::get<mutex_monitor::Monitor>(monitor_).notifyAll() :
            std::get<legacy_monitor::Monitor>(monitor_).notifyAll();
  }

private:
  std::variant<std::monostate, legacy_monitor::Monitor, mutex_monitor::Monitor> monitor_;
  const bool mode_{DEBUG_CLR_USE_STDMUTEX_IN_AMD_MONITOR};
};

class ScopedLock : StackObject {
 public:
  ScopedLock(Monitor& lock) : lock_(&lock) { lock_->lock(); }

  ScopedLock(Monitor* lock) : lock_(lock) {
    if (lock_) lock_->lock();
  }

  ~ScopedLock() {
    if (lock_) lock_->unlock();
  }

 private:
  Monitor* lock_;
};

}  // namespace amd

#endif /*MONITOR_HPP_*/
