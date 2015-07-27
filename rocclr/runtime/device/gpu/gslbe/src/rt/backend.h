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

/** CAL computational domain */
typedef struct CALdomainRec {
    CALuint x;                 /**< x origin of domain */
    CALuint y;                 /**< y origin of domain */
    CALuint width;             /**< width of domain */
    CALuint height;            /**< height of domain */
} CALdomain;

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
    bool       localDataShare;                      /**< local data share supported */
    bool       globalDataShare;                     /**< global data share supported */
    bool       globalGPR;                           /**< global GPR supported */
    bool       computeShader;                       /**< compute shader supported */
    bool       memExport;                           /**< memexport supported */
    CALuint    pitch_alignment;                     /**< Required alignment for calCreateRes allocations (in data elements) */
    CALuint    surface_alignment;                   /**< Required start address alignment for calCreateRes allocations (in bytes) */
    CALuint    numberOfUAVs;                        /**< Number of UAVs */
    bool       bUAVMemExport;                       /**< Hw only supports mem export to simulate 1 UAV */
    CALuint    numberOfShaderEngines;               /**< Number of shader engines */
    CALuint    targetRevision;                      /**< Asic family revision */
    CALuint    totalVisibleHeap;                    /**< Amount of visible local GPU RAM in megabytes */
    CALuint    totalInvisibleHeap;                  /**< Amount of invisible local GPU RAM in megabytes */
    CALuint    totalDirectHeap;                     /**< Amount of direct GPU memory in megabytes */
    CALuint    totalCoherentHeap;                   /**< Amount of coherent GPU memory in megabytes */
    CALuint    totalRemoteSharedHeap;               /**< Amount of remote Shared GPU memory in megabytes */
    CALuint    totalCachedRemoteSharedHeap;         /**< Amount of cached remote Shared GPU memory in megabytes */
    CALuint    totalSDIHeap;                        /**< Amount of SDI memory allocated in CCC */
    CALuint    pciTopologyInformation;              /**< PCI topology information contains: bus, device and function number. */
    CALchar    boardName[CAL_ASIC_INFO_MAX_LEN];    /**< Actual ASIC board name and not the internal name. */
    bool       vectorBufferInstructionAddr64;       /**< Vector buffer instructions support ADDR64 mode */
    bool       memRandomAccessTargetInstructions;   /**< hw/sc supports memory RAT (Random Access Target) instructions e.g. mem0.x_z_ supported */
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
} CALdeviceattribs;

/** CAL device status */
typedef struct CALdevicestatusRec {
    CALuint   struct_size;                          /**< Client filled out size of CALdevicestatus struct */
    CALuint   availLocalRAM;                        /**< Amount of available local GPU RAM in megabytes */
    CALuint   availUncachedRemoteRAM;               /**< Amount of available uncached remote GPU memory in megabytes */
    CALuint   availCachedRemoteRAM;                 /**< Amount of available cached remote GPU memory in megabytes */
    CALuint   availVisibleHeap;                     /**< Amount of available visible local GPU RAM in megabytes */
    CALuint   availInvisibleHeap;                   /**< Amount of available invisible local GPU RAM in megabytes */
    CALuint   availDirectHeap;                      /**< Amount of available direct GPU memory in megabytes */
    CALuint   availCoherentHeap;                    /**< Amount of available coherent GPU memory in megabytes */
    CALuint   availRemoteSharedHeap;                /**< Amount of available remote Shared GPU memory in megabytes */
    CALuint   availCachedRemoteSharedHeap;          /**< Amount of available cached remote Shared GPU memory in megabytes */
    CALuint   largestBlockVisibleHeap;              /**< Largest block available visible local GPU RAM in megabytes */
    CALuint   largestBlockInvisibleHeap;            /**< Largest block available invisible local GPU RAM in megabytes */
    CALuint   largestBlockRemoteHeap;               /**< Largest block available remote GPU memory in megabytes */
    CALuint   largestBlockCachedRemoteHeap;         /**< Largest block available cached remote GPU memory in megabytes */
    CALuint   largestBlockDirectHeap;               /**< Largest block available direct GPU memory in megabytes */
    CALuint   largestBlockCoherentHeap;             /**< Largest block available coherent GPU memory in megabytes */
    CALuint   largestBlockRemoteSharedHeap;         /**< Largest block available remote Shared GPU memory in megabytes */
    CALuint   largestBlockCachedRemoteSharedHeap;   /**< Largest block available cached remote Shared GPU memory in megabytes */
} CALdevicestatus;

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
    bool       eCsSetupMode;            /**< Slow mode */
    CALuint    numThreadPerGroup;       /**< Flattend umber of threads per group */
    CALuint    numThreadPerGroupX;      /**< x dimension of numThreadPerGroup */
    CALuint    numThreadPerGroupY;      /**< y dimension of numThreadPerGroup */
    CALuint    numThreadPerGroupZ;      /**< z dimension of numThreadPerGroup */
    CALuint    totalNumThreadGroup;     /**< Total number of thread groups */
    CALuint    numWavefrontPerSIMD;     /**< Number of wavefronts per SIMD */
    bool       isMaxNumWavePerSIMD;     /**< Is this the max num active wavefronts per SIMD */
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

typedef struct ProgramGridRec
{
    gslDomain3D     gridBlock;       /**< size of a block of data */
    gslDomain3D     gridSize;        /**< size of 'blocks' to execute. */
    gslDomain3D     partialGridBlock;/** Partial grid block */
    CALuint         localSize;       /** size of OpenCL Local Memory in bytes */
} ProgramGrid;

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

//
// Thread Trace Extension
//

typedef struct CALthreadTraceConfigRec    CALthreadTraceConfig;

struct CALthreadTraceConfigRec
{
    CALuint          cu;             // target compute unit [cu]
    CALuint          sh;             // target shader array [sh],that contains target cu
    CALuint          simd_mask;      // bitmask to enable or disable target tokens for different SIMDs
    CALuint          vm_id_mask;     // virtual memory [vm] IDs to capture
    CALuint          token_mask;     // bitmask indicating which trace token IDs will be included in the trace
    CALuint          reg_mask;       // bitmask indicating which register types should be included in the trace
    CALuint          inst_mask;      // types of instruction scheduling updates which should be recorded
    CALuint          random_seed;    // linear feedback shift register [LFSR] seed
    CALuint          user_data;      // user data ,which is written as payload
    CALuint          capture_mode;   // indicator for the way how THREAD_TRACE_START / STOP events affect token collection
    CALboolean       is_user_data;   // indicator if user_data is set
    CALboolean       is_wrapped;     // indicator if the memory buffer should be wrapped around instead of stopping at the end
};

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


/*
 * GPU Backend functions
 */
void calInit(void);
void calShutdown(void);
uint32 calGetDeviceCount();

#endif
