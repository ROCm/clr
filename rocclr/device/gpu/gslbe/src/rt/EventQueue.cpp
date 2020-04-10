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


#include "EventQueue.h"
#include "query/QueryObject.h"
#include "gsl_ctx.h"

EventQueue::EventQueue()
{
    m_cs            = NULL;
    m_queueSize     = c_staticQueueSize;

    memset(m_queries,0,sizeof(m_queries));
    memset(m_flushed,0,sizeof(m_flushed));

    m_latestRetired = 0;
    m_headId   = m_queueSize - 1 ;
    m_tail = 0;
}

EventQueue::~EventQueue()
{
    for (unsigned int i = 0; i < c_staticQueueSize; i++)
    {
        assert(m_queries[i] == 0);
    }
}

bool
EventQueue::open(gsCtx* cs, gslQueryTarget target, EQManagerConfig config, uint32 engineMask)
{
    assert((config == EQManager_HIGH) || (config == EQManager_LOW));
    setSlotCount((int) config);
    assert((GpuEvent::InvalidID+1) % m_queueSize  == 0);

    m_cs = cs;
    
    m_headId   = m_queueSize - 1 ;
    m_tail = 0;
    m_latestRetired = 0;
    m_target = target;
    m_engineMask = engineMask;

    for (unsigned int i = 0; i < m_queueSize; i++)
    {
        m_queries[i] = cs->createQuery(target);
    }

    return true;
}

void
EventQueue::close()
{
    if (!m_cs) // the queue is unintialized.
    {
        return;
    }
    
    for (unsigned int i = 0; i < m_queueSize; i++)
    {
        m_cs->destroyQuery(m_queries[i]);
    }
    memset(m_queries, 0, sizeof(m_queries));
    memset(m_flushed, 0, sizeof(m_flushed));
    m_latestRetired = 0;
    m_headId   = m_queueSize - 1 ;
    m_tail = 0;
    m_cs = NULL;
}

void
EventQueue::begin()
{
    const CALuint slot = m_headId % m_queueSize;
    gslErrorCode ec =  m_queries[slot]->BeginQuery(m_cs, m_target, 0, m_engineMask);
    assert(ec == GSL_NO_ERROR);

    m_flushed[slot] = false;  // we've started a query, but it hasn't been checked yet...
}

uint32
EventQueue::end()
{
    uint32 ret = m_headId;
    const uint32 slot = m_headId % m_queueSize;

    m_queries[slot]->EndQuery(m_cs, 0);

    m_headId++;
    m_tail++;

    if (GpuEvent::InvalidID == m_headId)
    {
        // Flush on an event ID wrap around or when the Queue is going to wrap in
         flush();
         //roll numbers back to the beginning
         m_latestRetired = 0;
         m_headId = m_headId % m_queueSize;
         m_tail = m_tail % m_queueSize;
    }
    return ret;
}

bool
EventQueue::isDone(uint32 event)
{
    assert((event < GpuEvent::InvalidID) && "illegal event handle");
    // if the event is older the the last known retired event we
    // do not need to process it.
    if (event <= m_latestRetired)
    {
        return true;
    }

    // if the event is older than the oldest event handle we have
    //  we synchronize with the oldest event.
    if (event < m_tail)
    {
        return waitForEvent(m_tail, CAL_WAIT_LOW_CPU_UTILIZATION);
    }

    //
    //  If we've never called flush on the query object, go ahead flush the first time to ensure
    //  we never infinite loop
    //
    const uint32 slot = event % m_queueSize;
    if (!m_flushed[slot])
    {
        flush();
    }

    //
    // Since we're in between, we actually have to check to see if things are truely done
    //
    bool retVal = m_queries[slot]->IsResultAvailable(m_cs);

    // cache the most recently retired event
    if (retVal && (event < m_headId) && (event > m_latestRetired))
    {
        m_latestRetired = event;
    }

    return retVal;
}

bool
EventQueue::waitForEvent(uint32 event, uint32 waitType)
{
    // if we already retired a younger event we don't to process current events
    if (event <= m_latestRetired)
    {
        return true;
    }

    // if the event is older than the oldest event handle we have
    //  we synchronize with the oldest event
    if (event < m_tail)
    {
        event = m_tail;
    }

    //
    //  If we've never called flush on the query object, go ahead flush the first time to ensure
    //  we never infinite loop
    //
    const uint32 slot = event % m_queueSize;
    if (!m_flushed[slot])
    {
        flush();
    }
    uint64 param;
    m_queries[slot]->GetResult(m_cs, &param, (IOSyncWaitType) waitType);

    // cache the most recently retired event
    if ((event < m_headId) && (event > m_latestRetired))
    {
        m_latestRetired = event;
    }

    return (param != 0);
}

bool
EventQueue::flush()
{
    m_cs->Flush(false, m_engineMask);
    memset(m_flushed, 1, sizeof(m_flushed));
    return true;
}


void
EventQueue::setSlotCount(uint32 slotCount)
{
    if (slotCount < c_staticQueueSize)
    {
        m_queueSize = slotCount;
    }
    else
    {
        m_queueSize = c_staticQueueSize;
    }
}
