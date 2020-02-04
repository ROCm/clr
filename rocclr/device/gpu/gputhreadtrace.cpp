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

#include "device/gpu/gputhreadtrace.hpp"
#include "device/gpu/gpuvirtual.hpp"

namespace gpu {

CalThreadTraceReference::~CalThreadTraceReference() {
  // The thread trace object is always associated with a particular queue,
  // so we have to lock just this queue
  amd::ScopedLock lock(gpu_.execution());

  if (0 != threadTrace_) {
    // gpu().cs()->destroyQuery(gslThreadTrace());
  }
}


ThreadTrace::~ThreadTrace() {
  if (calRef_ == NULL) {
    return;
  }
  for (uint i = 0; i < amdThreadTraceMemObjsNum_; ++i) {
    threadTraceBufferObjs_[i]->attachMemObject(gpu().cs(), NULL, 0, 0, 0, i);
    gpu().cs()->destroyShaderTraceBuffer(threadTraceBufferObjs_[i]);
  }

  // Release the thread trace reference object
  // calRef_->release();
}

bool ThreadTrace::create(CalThreadTraceReference* calRef) {
  assert(&gpu() == &calRef->gpu());

  calRef_ = calRef;
  threadTrace_ = calRef->gslThreadTrace();

  return true;
}

bool ThreadTrace::info(uint infoType, uint* info, uint infoSize) const {
  switch (infoType) {
    case CL_THREAD_TRACE_BUFFERS_SIZE: {
      if (infoSize < amdThreadTraceMemObjsNum_) {
        LogError("The amount of buffers should be equal to the amount of Shader Engines");
        return false;
      } else {
        gslThreadTrace()->GetResultAll(gpu().cs(), info);
      }
      break;
    }
    default:
      LogError("Wrong ThreadTrace::getInfo parameter");
      return false;
  }
  return true;
}

}  // namespace gpu
