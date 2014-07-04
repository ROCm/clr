//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef MONITOR_HPP_
#define MONITOR_HPP_

#include "top.hpp"
#include "atomic.hpp"
#include "thread/semaphore.hpp"
#include "thread/thread.hpp"

namespace amd {

/*! \addtogroup Threads
 *  @{
 *
 *  \addtogroup Synchronization
 *  @{
 */

class Monitor : public HeapObject
{
    typedef SimplyLinkedNode<Semaphore*,StackObject> LinkedNode;

private:
    static const bool kUnlocked = false;
    static const bool kLocked = true;

    static const int kMaxSpinIter = 55; //!< Total number of spin iterations.
    static const int kMaxReadSpinIter = 50; //!< Read iterations before yielding

    /*! Linked list of semaphores the contending threads are waiting on
     *  and main lock.
     */
    AtomicMarkableReference<LinkedNode> contendersList_;
    //! The Mutex's name
    char name_[64];

    //! Semaphore of the next thread to contend for the lock.
    AtomicMarkableReference<Semaphore> onDeck_;
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
    bool isLocked() const { return contendersList_.isMarked(); }

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

class ScopedLock : StackObject
{
private:
    Monitor* lock_;

public:
    ScopedLock(Monitor& lock)
        : lock_(&lock)
    {
        lock_->lock();
    }

    ScopedLock(Monitor* lock)
        : lock_(lock)
    {
        if (lock_) lock_->lock();
    }

    ~ScopedLock()
    {
        if (lock_) lock_->unlock();
    }
};

/*! @}
 *  @}
 */

inline bool
Monitor::tryLock()
{
    Thread* thread = Thread::current();
    assert(thread != NULL && "cannot lock() from (null)");

    LinkedNode* ptr; bool isLocked;
    tie(ptr, isLocked) = contendersList_.get();

    if (unlikely(isLocked)) {
        if (recursive_ && thread == owner_) {
            // Recursive lock: increment the lock count and return.
            ++lockCount_;
            return true;
        }
        return false; // Already locked!
    }

    if (unlikely(!contendersList_.compareAndSet(
            ptr, ptr, kUnlocked, kLocked))) {
        return false; // We failed the CAS from unlocked to locked.
    }

    setOwner(thread); // cannot move above the CAS.
    lockCount_ = 1;

    return true;
}

inline void
Monitor::lock()
{
    if (unlikely(!tryLock())) {
        // The lock is contented.
        finishLock();
    }

    // This is the beginning of the critical region. From now-on, everything
    // executes single-threaded!
    //
}

inline void
Monitor::unlock()
{
    assert(isLocked() && owner_ == Thread::current() && "invariant");

    if (recursive_ && --lockCount_ > 0) {
        // was a recursive lock case, simply return.
        return;
    }

    setOwner(NULL);

    while (true) {
        LinkedNode* ptr = contendersList_.getReference();
        // Clear the lock bit.
        if (contendersList_.compareAndSet(ptr, ptr, kLocked, kUnlocked)) {
            break; // We succeeded the CAS from locked to unlocked.
        }
    }
    //
    // This is the end of the critical region.

    // Check if we have an on-deck thread that needs signaling.
    Semaphore* onDeck; bool isMarked;
    tie(onDeck, isMarked) = onDeck_.get();
    if (onDeck != NULL) {
        if (!isMarked) {
            // Only signal if it is unmarked.
            onDeck->post();
        }
        return; // We are done.
    }

    // We do not have an on-deck thread yet, we might have to walk the list in
    // order to select the next onDeck_. Only one thread needs to fill onDeck_,
    // so return if the list is empty or if the lock got acquired again (it's
    // somebody else's problem now!)

    LinkedNode* head; bool isLocked;
    amd::tie(head, isLocked) = contendersList_.get();
    if (isLocked || head == NULL) {
        return;
    }

    // Finish the unlock operation: find a thread to wake up.
    finishUnlock();
}

} // namespace amd

#endif /*MONITOR_HPP_*/
