//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "thread/monitor.hpp"
#include "thread/atomic.hpp"
#include "thread/semaphore.hpp"
#include "thread/thread.hpp"
#include "utils/util.hpp"

#include <cstring>
#include <tuple>
#include <utility>

namespace amd {

Monitor::Monitor(const char* name, bool recursive) :
    contendersList_(NULL), onDeck_(NULL), waitersList_(NULL),
    owner_(NULL), recursive_(recursive)
{
    const size_t maxNameLen = sizeof(name_);
    if (name == NULL) {
        const char* unknownName = "@unknown@";
        assert(sizeof(unknownName) < maxNameLen && "just checking");
        strcpy(name_, unknownName);
    }
    else {
        strncpy(name_, name, maxNameLen - 1);
        name_[maxNameLen - 1] = '\0';
    }
}

bool
Monitor::trySpinLock()
{
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

void
Monitor::finishLock()
{
    Thread* thread = Thread::current();
    assert(thread != NULL && "cannot lock() from (null)");

    if (trySpinLock()) {
        return; // We succeeded, we are done.
    }

    /* The lock is contended. Push the thread's semaphore onto
     * the contention list.
     */
    Semaphore& sem = thread->lockSemaphore();
    sem.reset();

    LinkedNode newHead;
    newHead.setItem(&sem);

    while (true) {
        LinkedNode* head; bool isLocked;

        // The assumption is that lockWord is locked. Make sure we do not
        // continue unless the lock bit is set.
        std::tie(head, isLocked) = contendersList_.get();
        if (!isLocked) {
            if (tryLock()) {
                return;
            }
            continue;
        }

        // Set the new contention list head if lockWord is unchanged.
        newHead.setNext(head);
        if (contendersList_.compareAndSet(head, &newHead, kLocked, kLocked)) {
            break;
        }

        // We failed the CAS. yield/pause before trying again.
        Thread::yield();
    }

    int32_t spinCount = 0;
    // Go to sleep until we become the on-deck thread.
    while (onDeck_.getReference() != &sem) {
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
            sem.wait();
        }
        spinCount++;
    }

    spinCount = 0;
    //
    // From now-on, we are the on-deck thread. It will stay that way until
    // we successfuly acquire the lock.
    //
    while (true) {
        assert(onDeck_.getReference() == &sem && "just checking");
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
            sem.wait();
        }
        spinCount++;
    }

    assert(newHead.next() == NULL && "Should not be linked");
    onDeck_ = NULL;
}

void
Monitor::finishUnlock()
{
    // If we get here, it means that there might be a thread in the contention
    // list waiting to acquire the lock. We need to select a successor and
    // place it on-deck.

    while (true) {
        // Grab the onDeck_ microlock to protect the next loop (make sure only
        // one semaphore is removed from the contention list).
        //
        if (!onDeck_.compareAndSet(NULL, NULL, kUnlocked, kLocked)) {
            return; // Somebody else has the microlock, let him select onDeck_
        }

        LinkedNode* head; bool isLocked;
        while (true) {
            std::tie(head, isLocked) = contendersList_.get();

            if (head == NULL) {
                break; // There's nothing else to do.
            }

            if (isLocked) {
                // Somebody could have acquired then released the lock
                // and failed to grab the onDeck_ microlock.
                head = NULL;
                break;
            }

            if (contendersList_.compareAndSet(
                    head, head->next(), kUnlocked, kUnlocked)) {
            #ifdef ASSERT
                head->setNext(NULL);
            #endif // ASSERT
                break;
            }
        }

        Semaphore* sem = (head != NULL) ? head->item() : NULL;
        onDeck_ = sem;
        MemoryOrder::fence();
        //
        // Release the onDeck_ microlock (end of critical region);

        if (sem != NULL) {
            sem->post();
            return;
        }

        // We do not have an on-deck thread (sem == NULL). Return if
        // the contention list is empty or if the lock got acquired again.
        std::tie(head, isLocked) = contendersList_.get();
        if (isLocked || head == NULL) {
            return;
        }
    }
}

void
Monitor::wait()
{
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
    while (onDeck_.getReference() != &suspend) {
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
    while (true) {
        assert(onDeck_.getReference() == &suspend && "just checking");

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

    onDeck_ = NULL;
    MemoryOrder::fence();
}

void
Monitor::notify()
{
    assert(isLocked() && owner_ == Thread::current() && "just checking");

    LinkedNode* waiter = waitersList_;
    if (waiter == NULL) {
        return;
    }

    // Dequeue a waiter from the wait list and add it to the contention list.
    waitersList_ = waiter->next();
    while (true) {
        LinkedNode* node = contendersList_.getReference();

        waiter->setNext(node);
        if (contendersList_.compareAndSet(node, waiter, kLocked, kLocked)) {
            break;
        }
    }
}

void
Monitor::notifyAll()
{
    // NOTE: We could CAS the whole list in 1 shot but this is
    // not critical code. Optimize this if it becomes hot.
    while (waitersList_ != NULL) {
        notify();
    }
}

} // namespace amd
