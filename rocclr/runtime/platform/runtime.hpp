//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef RUNTIME_HPP_
#define RUNTIME_HPP_

#include "top.hpp"
#include "thread/thread.hpp"

namespace amd {

/*! \addtogroup Runtime The OpenCL Runtime
 *  @{
 */

class Runtime : AllStatic {
  static volatile bool initialized_;

 public:
  //! Return true if the OpencCL runtime is already initialized
  inline static bool initialized();

  //! Initialize the OpenCL runtime.
  static bool init();

  //! Tear down the runtime.
  static void tearDown();

  //! Return true if the Runtime is still single-threaded.
  static bool singleThreaded() { return !initialized(); }
};

#if 0
class HostThread : public Thread
{
private:
    virtual void run(void* data) { ShouldNotCallThis(); }

public:
    HostThread() : Thread("HostThread", 0, false)
    {
        setHandle(NULL);
        setCurrent();

        if (!amd::Runtime::initialized() && !amd::Runtime::init()) {
            return;
        }

        Os::currentStackInfo(&stackBase_, &stackSize_);
        setState(RUNNABLE);
    }

    bool isHostThread() const { return true; };

    static inline HostThread* current()
    {
        Thread* thread = Thread::current();
        assert(thread->isHostThread() && "just checking");
        return (HostThread*) thread;
    }
};
#endif

/*@}*/

inline bool Runtime::initialized() { return initialized_; }

}  // namespace amd

#endif /*RUNTIME_HPP_*/
