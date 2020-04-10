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

#ifndef __EventQueue_h__
#define __EventQueue_h__

#include "backend.h"
#include "atitypes.h"
#include "gsl_types.h"
#include "gsl_config.h"

namespace gsl
{
    class gsCtx;
};

enum EQManagerConfig
{
    EQManager_HIGH = 512,
    EQManager_LOW  = 32
};

class EventQueue {
public:
    static const unsigned int  c_staticQueueSize  = EQManager_HIGH;
    EventQueue();
    ~EventQueue();

    bool open(gsl::gsCtx* cs, gslQueryTarget target, EQManagerConfig config, uint32 engineMask = GSL_ENGINEMASK_ALL_BUT_UVD_VCE);
    void close();

    void        begin();
    uint32      end();
    bool        isDone(uint32 event);
    bool        waitForEvent(uint32 event, uint32 waitType);
    bool        flush();

private:

    gsl::gsCtx*      m_cs;

    uint32           m_queueSize;
    gslQueryTarget   m_target;
    uint32           m_engineMask;   // EngineMask for this Query
    uint32           m_tail; //represents the oldest event we have
    uint32           m_headId;
    uint32           m_latestRetired; //!< most recentyl retired event.
    gslQueryObject   m_queries[c_staticQueueSize];
    bool             m_flushed[c_staticQueueSize];
    ///////////////////////
    // private functions //
    ///////////////////////
    void             setSlotCount(uint32 slotCount);
};

#endif

