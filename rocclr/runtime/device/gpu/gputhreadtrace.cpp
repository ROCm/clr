//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/gpu/gputhreadtrace.hpp"
#include "device/gpu/gpuvirtual.hpp"

namespace gpu {

CalThreadTraceReference::~CalThreadTraceReference() {
    // The thread trace object is always associated with a particular queue,
    // so we have to lock just this queue
    amd::ScopedLock lock(gpu_.execution());

    if (0 != threadTrace_) {
        //gpu().destroyThreadTrace(gslThreadTrace());
    }
}


ThreadTrace::~ThreadTrace()
{
    if (calRef_ == NULL) {
        return;
    }
    for(uint i = 0; i < amdThreadTraceMemObjsNum_;++i) {
        gpu().DestroyThreadTraceBuffer(threadTraceBufferObjs_[i],i);
    }

    // Release the thread trace reference object
    //calRef_->release();
}

bool
ThreadTrace::create(CalThreadTraceReference* calRef)
{
    assert(&gpu() == &calRef->gpu());

    calRef_ = calRef;
    threadTrace_ = calRef->gslThreadTrace();

    return true;
}

bool
ThreadTrace::info(uint infoType, uint* info, uint infoSize) const
{
    switch (infoType) {
    case CL_THREAD_TRACE_BUFFERS_SIZE: {
        if (infoSize < amdThreadTraceMemObjsNum_) {
            LogError("The amount of buffers should be equal to the amount of Shader Engines");
            return false;
        }
        else {
            gpu().getThreadTraceQueryRes(gslThreadTrace(), info);
        }
        break;
    }
    default:
        LogError("Wrong ThreadTrace::getInfo parameter");
        return false;
    }
    return true;
}

} // namespace gpu
