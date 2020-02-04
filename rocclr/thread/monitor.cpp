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

#include "thread/monitor.hpp"
#include "thread/atomic.hpp"
#include "thread/semaphore.hpp"
#include "thread/thread.hpp"
#include "utils/util.hpp"

#include <cstring>
#include <tuple>
#include <utility>

namespace amd {

Monitor::Monitor(const char* name, bool recursive)
    : contendersList_(0), onDeck_(0), waitersList_(NULL), owner_(NULL), recursive_(recursive) {
  const size_t maxNameLen = sizeof(name_);
  if (name == NULL) {
    const char* unknownName = "@unknown@";
    assert(sizeof(unknownName) < maxNameLen && "just checking");
    strcpy(name_, unknownName);
  } else {
    strncpy(name_, name, maxNameLen - 1);
    name_[maxNameLen - 1] = '\0';
  }
}

bool Monitor::trySpinLock() {
  if (tryLock()) {
    return true;
  }

  for (int s = kMaxSpinIter; s > 0; --s) {
    // First, be SMT friendly
    if (s >= (kMaxSpinIter - kMaxReadSpinIter)) {
      Os::spinPause();
    }
    // and then SMP friendly
    else {
      Thread::yield();
    }
    if (!isLocked()) {
      return tryLock();
    }
  }

  // We could not acquire the lock in the spin loop.
  return false;
}

void Monitor::finishLock() {
  Thread* thread = Thread::current();
  assert(thread != NULL && "cannot lock() from (null)");

  if (trySpinLock()) {
    return;  // We succeeded, we are done.
  }

  /* The lock is contended. Push the thread's semaphore onto
   * the contention list.
   */
  Semaphore& semaphore = thread->lockSemaphore();
  semaphore.reset();

  LinkedNode newHead;
  newHead.setItem(&semaphore);

  intptr_t head = contendersList_.load(std::memory_order_acquire);
  for (;;) {
    // The assumption is that lockWord is locked. Make sure we do not
    // continue unless the lock bit is set.
    if ((head & kLockBit) == 0) {
      if (tryLock()) {
        return;
      }
      continue;
    }

    // Set the new contention list head if lockWord is unchanged.
    newHead.setNext(reinterpret_cast<LinkedNode*>(head & ~kLockBit));
    if (contendersList_.compare_exchange_weak(head, reinterpret_cast<intptr_t>(&newHead) | kLockBit,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      break;
    }

    // We failed the CAS. yield/pause before trying again.
    Thread::yield();
  }

  int32_t spinCount = 0;
  // Go to sleep until we become the on-deck thread.
  while ((onDeck_ & ~kLockBit) != reinterpret_cast<intptr_t>(&semaphore)) {
    // First, be SMT friendly
    if (spinCount < kMaxReadSpinIter) {
      Os::spinPause();
    }
    // and then SMP friendly
    else if (spinCount < kMaxSpinIter) {
      Thread::yield();
    }
    // now go to sleep
    else {
      semaphore.wait();
    }
    spinCount++;
  }

  spinCount = 0;
  //
  // From now-on, we are the on-deck thread. It will stay that way until
  // we successfuly acquire the lock.
  //
  for (;;) {
    assert((onDeck_ & ~kLockBit) == reinterpret_cast<intptr_t>(&semaphore) && "just checking");
    if (tryLock()) {
      break;
    }

    // Somebody beat us to it. Since we are on-deck, we can just go
    // back to sleep.
    // First, be SMT friendly
    if (spinCount < kMaxReadSpinIter) {
      Os::spinPause();
    }
    // and then SMP friendly
    else if (spinCount < kMaxSpinIter) {
      Thread::yield();
    }
    // now go to sleep
    else {
      semaphore.wait();
    }
    spinCount++;
  }

  assert(newHead.next() == NULL && "Should not be linked");
  onDeck_ = 0;
}

void Monitor::finishUnlock() {
  // If we get here, it means that there might be a thread in the contention
  // list waiting to acquire the lock. We need to select a successor and
  // place it on-deck.

  for (;;) {
    // Grab the onDeck_ microlock to protect the next loop (make sure only
    // one semaphore is removed from the contention list).
    //
    intptr_t ptr = 0;
    if (!onDeck_.compare_exchange_strong(ptr, ptr | kLockBit, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
      return;  // Somebody else has the microlock, let him select onDeck_
    }

    intptr_t head = contendersList_.load(std::memory_order_acquire);
    for (;;) {
      if (head == 0) {
        break;  // There's nothing else to do.
      }

      if ((head & kLockBit) != 0) {
        // Somebody could have acquired then released the lock
        // and failed to grab the onDeck_ microlock.
        head = 0;
        break;
      }

      if (contendersList_.compare_exchange_weak(
              head, reinterpret_cast<intptr_t>(reinterpret_cast<LinkedNode*>(head)->next()),
              std::memory_order_acq_rel, std::memory_order_acquire)) {
#ifdef ASSERT
        reinterpret_cast<LinkedNode*>(head)->setNext(NULL);
#endif  // ASSERT
        break;
      }
    }

    Semaphore* semaphore = (head != 0) ? reinterpret_cast<LinkedNode*>(head)->item() : NULL;

    onDeck_.store(reinterpret_cast<intptr_t>(semaphore), std::memory_order_release);
    //
    // Release the onDeck_ microlock (end of critical region);

    if (semaphore != NULL) {
      semaphore->post();
      return;
    }

    // We do not have an on-deck thread (semaphore == NULL). Return if
    // the contention list is empty or if the lock got acquired again.
    head = contendersList_;
    if (head == 0 || (head & kLockBit) != 0) {
      return;
    }
  }
}

void Monitor::wait() {
  Thread* thread = Thread::current();
  assert(isLocked() && owner_ == thread && "just checking");

  // Add the thread's resume semaphore to the list.
  Semaphore& suspend = thread->suspendSemaphore();
  suspend.reset();

  LinkedNode newHead;
  newHead.setItem(&suspend);
  newHead.setNext(waitersList_);
  waitersList_ = &newHead;

  // Preserve the lock count (for recursive mutexes)
  uint32_t lockCount = lockCount_;
  lockCount_ = 1;

  // Release the lock and go to sleep.
  unlock();

  // Go to sleep until we become the on-deck thread.
  int32_t spinCount = 0;
  while ((onDeck_ & ~kLockBit) != reinterpret_cast<intptr_t>(&suspend)) {
    // First, be SMT friendly
    if (spinCount < kMaxReadSpinIter) {
      Os::spinPause();
    }
    // and then SMP friendly
    else if (spinCount < kMaxSpinIter) {
      Thread::yield();
    }
    // now go to sleep
    else {
      suspend.wait();
    }
    spinCount++;
  }

  spinCount = 0;
  for (;;) {
    assert((onDeck_ & ~kLockBit) == reinterpret_cast<intptr_t>(&suspend) && "just checking");

    if (trySpinLock()) {
      break;
    }

    // Somebody beat us to it. Since we are on-deck, we can just go
    // back to sleep.
    // First, be SMT friendly
    if (spinCount < kMaxReadSpinIter) {
      Os::spinPause();
    }
    // and then SMP friendly
    else if (spinCount < kMaxSpinIter) {
      Thread::yield();
    }
    // now go to sleep
    else {
      suspend.wait();
    }
    spinCount++;
  }

  // Restore the lock count (for recursive mutexes)
  lockCount_ = lockCount;

  onDeck_.store(0, std::memory_order_release);
}

void Monitor::notify() {
  assert(isLocked() && owner_ == Thread::current() && "just checking");

  LinkedNode* waiter = waitersList_;
  if (waiter == NULL) {
    return;
  }

  // Dequeue a waiter from the wait list and add it to the contention list.
  waitersList_ = waiter->next();

  intptr_t node = contendersList_.load(std::memory_order_acquire);
  for (;;) {
    waiter->setNext(reinterpret_cast<LinkedNode*>(node & ~kLockBit));
    if (contendersList_.compare_exchange_weak(node, reinterpret_cast<intptr_t>(waiter) | kLockBit,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      break;
    }
  }
}

void Monitor::notifyAll() {
  // NOTE: We could CAS the whole list in 1 shot but this is
  // not critical code. Optimize this if it becomes hot.
  while (waitersList_ != NULL) {
    notify();
  }
}

}  // namespace amd
