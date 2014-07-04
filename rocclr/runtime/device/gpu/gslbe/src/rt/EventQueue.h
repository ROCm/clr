#ifndef __EventQueue_h__
#define __EventQueue_h__

#include "cal.h"
#include "backend.h"
#include "atitypes.h"
#include "gsl_types.h"
#include "gsl_config.h"

//#define USE_3D_SYNC 1

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

