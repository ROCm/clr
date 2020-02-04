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

#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpucounters.hpp"
#include "device/gpu/gpuvirtual.hpp"
#include "query/PerformanceQueryObject.h"

namespace gpu {

CalCounterReference::~CalCounterReference() {
  // The counter object is always associated with a particular queue,
  // so we have to lock just this queue
  amd::ScopedLock lock(gpu_.execution());

  if (0 != counter_) {
    gpu().cs()->destroyQuery(gslCounter());
  }
}

bool CalCounterReference::growResultArray(uint index) {
  if (results_ != NULL) {
    delete[] results_;
  }
  results_ = new uint64_t[index + 1];
  if (results_ == NULL) {
    return false;
  }
  return true;
}

PerfCounter::~PerfCounter() {
  if (calRef_ == NULL) {
    return;
  }

  // Release the counter reference object
  calRef_->release();
}

bool PerfCounter::create(CalCounterReference* calRef) {
  assert(&gpu() == &calRef->gpu());

  calRef_ = calRef;
  counter_ = calRef->gslCounter();
  index_ = calRef->retain() - 2;
  calRef->growResultArray(index_);

  // Initialize the counter
  gslCounter()->getAsPerformanceQueryObject()->setCounterState(
      info()->blockIndex_, info()->counterIndex_, info()->eventIndex_);

  return true;
}

uint64_t PerfCounter::getInfo(uint64_t infoType) const {
  switch (infoType) {
    case CL_PERFCOUNTER_GPU_BLOCK_INDEX: {
      // Return the GPU block index
      return info()->blockIndex_;
    }
    case CL_PERFCOUNTER_GPU_COUNTER_INDEX: {
      // Return the GPU counter index
      return info()->counterIndex_;
    }
    case CL_PERFCOUNTER_GPU_EVENT_INDEX: {
      // Return the GPU event index
      return info()->eventIndex_;
    }
    case CL_PERFCOUNTER_DATA: {
      gslCounter()->GetResult(gpu().cs(), reinterpret_cast<uint64*>(calRef_->results()));
      return calRef_->results()[index_];
    }
    default:
      LogError("Wrong PerfCounter::getInfo parameter");
  }
  return 0;
}

}  // namespace gpu
