/*****************************************************************************
 *
 *
 *
 *  Trade secret of ATI Technologies, Inc.
 *  Copyright 2006, ATI Technologies, Inc., (unpublished)
 *
 *  All rights reserved.  This notice is intended as a precaution against
 *  inadvertent publication and does not imply publication or any waiver
 *  of confidentiality.  The year included in the foregoing notice is the
 *  year of creation of the work.
 *
 *
 ****************************************************************************
 */

#ifndef __CALIF_H__
#define __CALIF_H__

#define CALIF_VERSION_MAJOR        1
#define CALIF_VERSION_MINOR        1

#define CALIF_HELPER_SURF_WIDTH   256
#define CALIF_HELPER_SURF_HEIGHT  8

#define CALIF_SEMAPHORE_SURF_WIDTH   8
#define CALIF_SEMAPHORE_SURF_HEIGHT  1



// Structure for commuticating with driver through Lock backdoor
typedef struct _CALIF_LOCK_COMM_HEADER
{
    UINT    uCmd;
    UINT    *puRes;

    PVOID   pInputBuffer;
    UINT    uInputBufferSize;

    PVOID   pOutputBuffer;
    UINT    uOutputBufferSize;

} CALIF_LOCK_COMM_HEADER, *PCALIF_LOCK_COMM_HEADER;



// Commands for LOCK backdoor
typedef enum _CALIF_LOCK_CMD
{
    CALIF_LOCK_CMD_GET_VERSION        = 1,
    CALIF_LOCK_CMD_NEXT_SURF_INFO     = 2,
    CALIF_LOCK_CMD_GET_SURF_INFO      = 3,
    CALIF_LOCK_CMD_SET_ALIAS_INFO     = 4,
    CALIF_LOCK_CMD_SET_CAL_TARGET     = 5,
    CALIF_LOCK_CMD_GET_CAL_STATUS     = 6,
    CALIF_LOCK_CMD_GET_RENDER_STATUS  = 7,
    CALIF_LOCK_CMD_INVALID            = 0xFFFFFFFF,
} CALIF_LOCK_CMD;


typedef enum _CALIF_LOCK_CMD_RES
{
    CALIF_LOCK_CMD_RES_OK             = 0,
    CALIF_LOCK_CMD_RES_ERROR          = 1,
    CALIF_LOCK_CMD_RES_INVALID        = 0xFFFFFFFF,
} CALIF_LOCK_CMD_RES;



// Input structure
typedef struct _CALIF_CAL_TARGET
{
    ULONG ulSize;
    ULONG ulFlags;

    ULONG ulNumTargets;
    ULONG ulTargets[MAX_CAL_TARGETS];

    ULONG ulReserved[1];    // 16 byte alignment

} CALIF_CAL_TARGET, *PCALIF_CAL_TARGET;




#define CALIF_DEV_CAP_CAPABLE        0x00000001
#define CALIF_DEV_CAP_ENABLE         0x00000002
#define CALIF_DEV_CAP_PRIMARY        0x80000000

typedef struct _CALIF_DEV_INFO
{
    ULONG ulIndex;

    ULONG ulCaps;   // CALIF_DEV_CAP_XXX

    ULONG ulFBSize;
    LONGLONG llFBSharedSize;

    UCHAR ucDevicePath[MAX_REGISTRY_PATH];

    ULONG ulReserved[2];

} CALIF_DEV_INFO, *PCALIF_DEV_INFO;



// Output structure
typedef struct _CALIF_CAL_STATUS
{
    ULONG ulSize;
    ULONG ulFlags;

    ULONG ulCurrentIndex;
    ULONG ulAdapterCount;

    LONGLONG llSharedCacheableSize;
    LONGLONG llSharedUSWCSize;

    ULONG ulLinkCount;
    ULONG ulLinkAdaper[MAX_CAL_TARGETS];

    BOOL bP2PCap[MAX_CAL_DEVICE][MAX_CAL_DEVICE];

    CALIF_DEV_INFO devInfo[MAX_CAL_DEVICE];

    ULONG ulReserved[3];

} CALIF_CAL_STATUS, *PCALIF_CAL_STATUS;



// Output structure
typedef struct _CALIF_VERSION
{
    ULONG ulSize;
    ULONG ulFlags;

    UINT uMajor;  // Major version
    UINT uMinor;  // Minor version 

} CALIF_VERSION, *PCALIF_VERSION;



// Surface heap choice
typedef enum _CALIF_SURF_HEAP
{
    CALIF_SURF_HEAP_UNKNOWN             = 0,  // VCAM real mode or dummy surf
    CALIF_SURF_HEAP_LOCAL               = 1,  // Local Visible + Local Invisible
    CALIF_SURF_HEAP_LOCALIF_VISIBLE     = 2,
    CALIF_SURF_HEAP_USWC                = 3,
    CALIF_SURF_HEAP_CACHEABLE           = 4,
    CALIF_SURF_HEAP_SHARED_USWC         = 5,
    CALIF_SURF_HEAP_SHARED_CACHEABLE    = 6,
    
    CALIF_SURF_HEAP_INVALID             = 0xFFFFFFFF,
} CALIF_SURF_HEAP;


// Surface flag
#define CALIF_NEXT_SURF_FLAG_DUMMY      0x80000000
#define CALIF_NEXT_SURF_FLAG_LINEAR     0x40000000
#define CALIF_NEXT_SURF_FLAG_ARENA      0x20000000

// Input structure
typedef struct _CALIF_NEXT_SURF_INFO
{
    ULONG ulSize;
    ULONG ulFlags;

    // to match it later at surface creation time
    UINT            uWidth;
    UINT            uHeight;
    D3DFORMAT       d3dFormat;
    ULONG_PTR       lpProcessID;

    // info
    CALIF_SURF_HEAP uHeap;
    UINT            uFlags;

#if _WIN64
    ULONG           ulReserved[3];
#endif

} CALIF_NEXT_SURF_INFO, *PCALIF_NEXT_SURF_INFO;



// Output structure
typedef struct _CALIF_SURF_INFO
{
    ULONG               ulSize;
    ULONG               ulFlags;

    ULONG               ulDeviceIndex;      // current device id
    ULONG_PTR           lpSurfHandle;       // VCAM handle if VCAM is on
    LARGE_INTEGER       gpuDevAddr;         // mc address of the surface
    LONGLONG            llHeapOffset;       // offset from the beginning of the heap
    UINT                uMemSize;           // total memory size
    CALIF_SURF_HEAP     uHeap;              // memory pool
    UINT                uGranularity;       // minimum RT aligment
    UINT                uBitsPerPixel;      // bits per pixel
    UINT                uActualWidth;       // padded width pixel pitch
    UINT                uActualHeight;      // padded height pitch
    UINT                uPitch;             // padded width byte pitch

    UINT                uTile;              // Tiling of surface
    UINT                uTileSwizzle;       // Tile swizzle of surface

#if !_WIN64
    ULONG           ulReserved[1];
#endif

} CALIF_SURF_INFO, *PCALIF_SURF_INFO;



// Input structure
typedef struct _CALIF_ALIAS_SURF_INFO
{
    ULONG           ulSize;
    ULONG           ulFlags;

    ULONG           ulDeviceIndex;      // device id we want to alias to
    ULONG_PTR       lpSurfHandle;
    LONGLONG        llHeapOffset;       // offset from the beginning of the heap
    UINT            uMemSize;           // total memory size
    CALIF_SURF_HEAP uHeap;              // memory pool
    UINT            uGranularity;       // minimum RT aligment
    UINT            uBitsPerPixel;      // bits per pixel
    UINT            uActualWidth;       // padded width pixel pitch
    UINT            uActualHeight;      // padded height pitch
    UINT            uPitch;             // padded width byte pitch

#if _WIN64
    ULONG           ulReserved[3];
#endif

} CALIF_ALIAS_SURF_INFO, *PCALIF_ALIAS_SURF_INFO;



// Output structure
typedef struct _CALIF_RENDER_STATUS
{
    ULONG ulSize;
    ULONG ulFlags;

    BOOL  bSurfBusy;

    ULONG ulReserved[1];

} CALIF_RENDER_STATUS, *PCALIF_RENDER_STATUS;



// Commands for StretchBlt backdoor
typedef enum _CALIF_SBLT_CMD
{
    CALIF_SBLT_CMD_SURF_MARK_HELPER         = 0x2200,
    CALIF_SBLT_CMD_SURF_GET_SURF_INFO       = 0x2400,
    CALIF_SBLT_CMD_SURF_ALIAS               = 0x2600,
    CALIF_SBLT_CMD_SEMAPHORE_WAIT           = 0x4200,
    CALIF_SBLT_CMD_SEMAPHORE_SIGNAL         = 0x4400,
    CALIF_SBLT_CMD_OUTPUT_CACHE_FLUSH       = 0x4600,
    CALIF_SBLT_CMD_INPUT_CACHE_INVALIDATE   = 0x6200,
    CALIF_SBLT_CMD_GET_RENDER_STATUS        = 0x6400,
    CALIF_SBLT_CMD_PIN_SURF                 = 0x6600,
    CALIF_SBLT_CMD_INVALID                  = 0xFFFF,

} CALIF_SBLT_CMD;


#define CALIF_SBLT_CMD_RECT_MASK__LEFT            0x000F
#define CALIF_SBLT_CMD_RECT_MASK__TOP             0x00F0
#define CALIF_SBLT_CMD_RECT_MASK__RIGHT           0x0F00
#define CALIF_SBLT_CMD_RECT_MASK__BOTTOM          0xF000

#define CALIF_SBLT_CMD_RECT_SHIFT__LEFT           0
#define CALIF_SBLT_CMD_RECT_SHIFT__TOP            4
#define CALIF_SBLT_CMD_RECT_SHIFT__RIGHT          8
#define CALIF_SBLT_CMD_RECT_SHIFT__BOTTOM         12

#endif//__CALIF_H__
