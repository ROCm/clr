#ifndef __BACKEND_H__
#define __BACKEND_H__

#include "cal.h"
#include "calcl.h"

//internal
#include <vector>
#include <cassert>

class CALGSLDevice;

//! Engine types
enum EngineType
{
    MainEngine  = 0,
    SdmaEngine,
    AllEngines
};

struct GpuEvent
{
    static const unsigned int InvalidID  = ((1<<30) - 1);

    EngineType      engineId_;  ///< type of the id
    unsigned int    id;         ///< actual event id

    //! GPU event default constructor
    GpuEvent(): engineId_(MainEngine), id(InvalidID) {}

    //! Returns true if the current event is valid
    bool isValid() const { return (id != InvalidID) ? true : false; }

    //! Set invalid event id
    void invalidate() { id = InvalidID; }
};

typedef enum CALBEtilingEnum
{
    CALBE_TILING_DEFAULT,
    CALBE_TILING_LINEAR,
    CALBE_TILING_TILED,
    CALBEtiling_FIRST = CALBE_TILING_DEFAULT,
    CALBEtiling_LAST  = CALBE_TILING_TILED,
} CALBEtiling;

/*
 * GPU Backend functions
 */
void calInit(void);
void calShutdown(void);
uint32 calGetDeviceCount();

#endif
