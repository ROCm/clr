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

#ifndef __BACKEND_H__
#define __BACKEND_H__

#include <vector>
#include <cassert>

//internal
#include "gsl_enum.h"
#include "gsl_types.h"
#include "cm_enum.h"
#include "caltarget.h"

typedef void           CALvoid;       /**< void type                        */
typedef char           CALchar;       /**< ASCII character                  */
typedef signed   char  CALbyte;       /**< 1 byte signed integer value      */
typedef unsigned char  CALubyte;      /**< 1 byte unsigned integer value    */
typedef signed   short CALshort;      /**< 2 byte signed integer value      */
typedef unsigned short CALushort;     /**< 2 byte unsigned integer value    */
typedef signed   int   CALint;        /**< 4 byte signed integer value      */
typedef unsigned int   CALuint;       /**< 4 byte unsigned intger value     */
typedef float          CALfloat;      /**< 32-bit IEEE floating point value */
typedef double         CALdouble;     /**< 64-bit IEEE floating point value */
typedef signed   long  CALlong;       /**< long value                       */
typedef unsigned long  CALulong;      /**< unsigned long value              */

#if defined(_MSC_VER)

typedef signed __int64     CALint64;  /**< 8 byte signed integer value */
typedef unsigned __int64   CALuint64; /**< 8 byte unsigned integer value */

#elif defined(__GNUC__)

typedef signed long long   CALint64;  /**< 8 byte signed integer value */
typedef unsigned long long CALuint64; /**< 8 byte unsigned integer value */

#else
#error "Unsupported compiler type."
#endif

/** Boolean type */
typedef enum CALbooleanEnum {
    CAL_FALSE          = 0,           /**< Boolean false value */
    CAL_TRUE           = 1            /**< Boolean true value */
} CALboolean;

/** CAL image container */
typedef struct CALimageRec*  CALimage;

#define CAL_ASIC_INFO_MAX_LEN 128
#define CAL_DRIVER_STORE_MAX_LEN    200

/** CAL device attributes */
typedef struct CALdeviceattribsRec {
    CALuint    struct_size;                         /**< Client filled out size of CALdeviceattribs struct */
    CALtarget  target;                              /**< Asic identifier */
    CALuint    localRAM;                            /**< Amount of local GPU RAM in megabytes */
    CALuint    uncachedRemoteRAM;                   /**< Amount of uncached remote GPU memory in megabytes */
    CALuint    cachedRemoteRAM;                     /**< Amount of cached remote GPU memory in megabytes */
    CALuint    engineClock;                         /**< GPU device clock rate in megahertz */
    CALuint    memoryClock;                         /**< GPU memory clock rate in megahertz */
    CALuint    wavefrontSize;                       /**< Wavefront size */
    CALuint    numberOfSIMD;                        /**< Number of SIMDs */
    CALuint    numberOfCUsperShaderArray;           /**< Number of CUs per shader array */
    bool       doublePrecision;                     /**< double precision supported */
    CALuint    numberOfShaderEngines;               /**< Number of shader engines */
    CALuint    totalVisibleHeap;                    /**< Amount of visible local GPU RAM in megabytes */
    CALuint    totalInvisibleHeap;                  /**< Amount of invisible local GPU RAM in megabytes */
    CALuint    totalDirectHeap;                     /**< Amount of direct GPU memory in megabytes */
    CALuint    totalCoherentHeap;                   /**< Amount of coherent GPU memory in megabytes */
    CALuint    totalRemoteSharedHeap;               /**< Amount of remote Shared GPU memory in megabytes */
    CALuint    totalCachedRemoteSharedHeap;         /**< Amount of cached remote Shared GPU memory in megabytes */
    CALuint    totalSDIHeap;                        /**< Amount of SDI memory allocated in CCC */
    CALuint    pciTopologyInformation;              /**< PCI topology information contains: bus, device and function number. */
    CALchar    boardName[CAL_ASIC_INFO_MAX_LEN];    /**< Actual ASIC board name and not the internal name. */
    CALuint    memBusWidth;                         /**< Memory busw width */
    CALuint    numMemBanks;                         /**< Number of memory banks */
    CALuint    counterFreq;                         /**< Ref clock counter frequency */
    double     nanoSecondsPerTick;                  /**< Nano seconds per GPU tick */
    bool       longIdleDetect;                      /**< Whether LongIdleDetect enabled */
    bool       svmAtomics;                          /**< check if svm atomics support */    
    CALuint64  vaStart;                             /**< VA start address */
    CALuint64  vaEnd;                               /**< VA end address */
    bool       isWorkstation;                       /**< Whether Device is a Workstation/Server part */
    CALuint    numOfVpu;                            /**< number of vpu in the device*/
    bool       isOpenCL200Device;                   /**< the flag to mark if the device is OpenCL 200 */
    bool       isSVMFineGrainSystem;                /**< check if SVM finegrainsystem */
    bool       isWDDM2Enabled;                      /**< check if WDDM2 is enabled */
    CALuint    maxRTCUs;                            /**< The maximum number of RT CUs for RT queues */
    CALuint    asicRevision;                        /**< The ASIC revision ID */
    CALchar    driverStore[CAL_DRIVER_STORE_MAX_LEN];/**< Driver store location. */
    CALuint    pcieDeviceID;                        /**< The ASIC PCIE device ID */
    CALuint    pcieRevisionID;                      /**< The ASIC PCIE revision ID */
} CALdeviceattribs;


/** CAL resource allocation flags **/
typedef enum CALresallocflagsEnum {
    CAL_RESALLOC_GLOBAL_BUFFER  = 1, /**< used for global import/export buffer */
} CALresallocflags;


/** CAL function information **/
typedef struct CALfuncInfoRec
{
    CALuint    maxScratchRegsNeeded;    /**< Maximum number of scratch regs needed */
    CALuint    numSharedGPRUser;        /**< Number of shared GPRs */
    CALuint    numSharedGPRTotal;       /**< Number of shared GPRs including ones used by SC */
    CALuint    numThreadPerGroup;       /**< Flattend umber of threads per group */
    CALuint    numThreadPerGroupX;      /**< x dimension of numThreadPerGroup */
    CALuint    numThreadPerGroupY;      /**< y dimension of numThreadPerGroup */
    CALuint    numThreadPerGroupZ;      /**< z dimension of numThreadPerGroup */
    CALuint    totalNumThreadGroup;     /**< Total number of thread groups */
    CALuint    numWavefrontPerSIMD;     /**< Number of wavefronts per SIMD */
    bool       setBufferForNumGroup;    /**< Need to set up buffer for info on number of thread groups? */
    CALuint    wavefrontSize;           /**< number of threads per wavefront. */
    CALuint    numGPRsAvailable;        /**< number of GPRs available to the program */
    CALuint    numGPRsUsed;             /**< number of GPRs used by the program */
    CALuint    LDSSizeAvailable;        /**< LDS size available to the program */
    CALuint    LDSSizeUsed;             /**< LDS size used by the program */
    CALuint    stackSizeAvailable;      /**< stack size availabe to the program */
    CALuint    stackSizeUsed;           /**< stack size use by the program */
    CALuint    numSGPRsAvailable;       /**< number of SGPRs available to the program */
    CALuint    numSGPRsUsed;            /**< number of SGPRs used by the program */
    CALuint    numVGPRsAvailable;       /**< number of VGPRs available to the program */
    CALuint    numVGPRsUsed;            /**< number of VGPRs used by the program */
} CALfuncInfo;

// flags for calCtxWaitForEvents
typedef enum CALwaitTypeEnum
{
    CAL_WAIT_POLLING                = 0,
    CAL_WAIT_LOW_CPU_UTILIZATION    = 1,
} CALwaitType;

//
// calResAllocView typedefs
//
typedef enum CALresallocviewflagsRec {
    CAL_RESALLOCVIEW_GLOBAL_BUFFER    = CAL_RESALLOC_GLOBAL_BUFFER, /**< used for global import/export buffer */
    CAL_RESALLOCVIEW_LINEAR_ALIGNED   = CAL_RESALLOC_GLOBAL_BUFFER, /**< 256 byte alignment restriction. */
    CAL_RESALLOCVIEW_LINEAR_UNALIGNED = 3,                          /**< no alignment restrictions */
} CALresallocviewflags;

typedef struct CALresourceDescRec {
    gslMemObjectAttribLocation  type;
    gslResource3D   size;
    cmSurfFmt       format;
    gslChannelOrder channelOrder;
    gslMemObjectAttribType    dimension;
    CALuint         mipLevels;
    CALvoid*        systemMemory;
    CALuint         flags;
    CALuint         systemMemorySize;
    CALuint64       busAddress[2];
    mcaddr          vaBase;
    gslMemObjectAttribSection section;
    CALuint         minAlignment;
    bool            isAllocSVM;
    bool            isAllocExecute;
} CALresourceDesc;

typedef enum CALresallocsliceviewflagsRec {
    CAL_RESALLOCSLICEVIEW_GLOBAL_BUFFER    = CAL_RESALLOC_GLOBAL_BUFFER, /**< used for global import/export buffer */
    CAL_RESALLOCSLICEVIEW_LINEAR_ALIGNED   = CAL_RESALLOC_GLOBAL_BUFFER, /**< 256 byte alignment restriction. */
    CAL_RESALLOCSLICEVIEW_LINEAR_UNALIGNED = CAL_RESALLOCVIEW_LINEAR_UNALIGNED,                          /**< no alignment restrictions */
    CAL_RESALLOCSLICEVIEW_LEVEL = 0x10, /**< sliceDesc.layer is not used, the whole level is only*/
    CAL_RESALLOCSLICEVIEW_LAYER = 0x20, /**< sliceDesc.layer is not used, the whole level is only*/
    CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER = CAL_RESALLOCSLICEVIEW_LEVEL | CAL_RESALLOCSLICEVIEW_LAYER,
} CALresallocsliceviewflags;

typedef enum CALmemcopyflagsEnum
{
    CAL_MEMCOPY_DEFAULT = 0, /**< default CAL behavior of partial sync */
    CAL_MEMCOPY_SYNC    = 1, /**< used to synchronize with the specified CAL context */
    CAL_MEMCOPY_ASYNC   = 2, /**< used to indicate completely asynchronous behavior */
} CALmemcopyflags;

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

#endif
