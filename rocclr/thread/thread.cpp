//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "thread/thread.hpp"
#include "thread/semaphore.hpp"
#include "thread/monitor.hpp"
#include "os/os.hpp"

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#endif  // _WIN32

namespace amd {

HostThread::HostThread() : Thread("HostThread", 0, false) {
  setCurrent();
  Os::currentStackInfo(&stackBase_, &stackSize_);
  setState(RUNNABLE);
}

void Thread::create() {
  created_ = new Semaphore();
  lock_ = new Semaphore();
  suspend_ = new Semaphore();

  selfSuspendLock_ = new Monitor();

  data_ = NULL;
  handle_ = NULL;
  setState(CREATED);
}

Thread::Thread(const std::string& name, size_t stackSize, bool spawn)
    : handle_(NULL), name_(name), stackSize_(stackSize) {
  create();

  if (!spawn) return;

  if ((handle_ = Os::createOsThread(this))) {
    // Now we need to wait for Thread::main to report back.
    while (state() != Thread::INITIALIZED) {
      created_->wait();
    }
  }
}

Thread::~Thread() {
#if defined(_WIN32)
  if (handle_ != NULL) {
    ::CloseHandle((HANDLE)handle_);
  }
#endif
  delete created_;
  delete lock_;
  delete suspend_;

  delete selfSuspendLock_;
}

void* Thread::main() {
#ifdef DEBUG
  Os::setCurrentThreadName(name().c_str());
#endif  // DEBUG
  Os::currentStackInfo(&stackBase_, &stackSize_);
  setCurrent();

  // Notify the parent thread that we are up and running.
  {
    ScopedLock sl(selfSuspendLock_);
    setState(INITIALIZED);
    created_->post();
    selfSuspendLock_->wait();
  }

  if (state() == RUNNABLE) {
    run(data_);
  }

  setState(FINISHED);
  return NULL;
}

bool Thread::start(void* data) {
  if (state() != INITIALIZED) {
    return false;
  }

  data_ = data;
  {
    ScopedLock sl(selfSuspendLock_);
    setState(RUNNABLE);
    selfSuspendLock_->notify();
  }

  return true;
}

void Thread::resume() {
  ScopedLock sl(selfSuspendLock_);
  selfSuspendLock_->notify();
}

#if defined(__linux__)

namespace details {

__thread Thread* thread_ __attribute__((tls_model("initial-exec")));

}  // namespace details

void Thread::registerStack(address base, address top) {
  // Nothing to do.
}

void Thread::setCurrent() { details::thread_ = this; }

#elif defined(_WIN32)

namespace details {

#if defined(USE_DECLSPEC_THREAD)
__declspec(thread) Thread* thread_;
#else   // !USE_DECLSPEC_THREAD
DWORD threadIndex_ = TlsAlloc();
#endif  // !USE_DECLSPEC_THREAD

}  // namespace details

void Thread::registerStack(address base, address top) {
  // Nothing to do.
}

void Thread::setCurrent() {
#if defined(USE_DECLSPEC_THREAD)
  details::thread_ = this;
#else   // !USE_DECLSPEC_THREAD
  TlsSetValue(details::threadIndex_, this);
#endif  // !USE_DECLSPEC_THREAD
}

#endif

bool Thread::init() {
  static bool initialized_ = false;

  // We could use InitOnceExecuteOnce/pthread_once here:
  if (initialized_) {
    return true;
  }
  initialized_ = true;

  // Register the main thread
  return NULL != new HostThread();
}

void Thread::tearDown() {
#if defined(_WIN32) && !defined(USE_DECLSPEC_THREAD)
  if (details::threadIndex_ != TLS_OUT_OF_INDEXES) {
    TlsFree(threadIndex_);
  }
#endif  // _WIN32 && !USE_DECLSPEC_THREAD
}

}  // namespace amd
