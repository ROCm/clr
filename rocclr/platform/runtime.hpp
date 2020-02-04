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
