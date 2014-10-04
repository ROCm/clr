/**
 *  @file     cal.h
 *  @brief    CAL Interface Header
 *  @version  1.00.0 Beta
 */


/* ============================================================

Copyright (c) 2007 Advanced Micro Devices, Inc.  All rights reserved.

Redistribution and use of this material is permitted under the following
conditions:

Redistributions must retain the above copyright notice and all terms of this
license.

In no event shall anyone redistributing or accessing or using this material
commence or participate in any arbitration or legal action relating to this
material against Advanced Micro Devices, Inc. or any copyright holders or
contributors. The foregoing shall survive any expiration or termination of
this license or any agreement or access or use related to this material.

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERATION, OR THAT IT IS FREE
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT.
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES,
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES,
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S.
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS,
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS,
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS.
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to
computer software and technical data, respectively. Use, duplication,
distribution or disclosure by the U.S. Government and/or DOD agencies is
subject to the full extent of restrictions in all applicable regulations,
including those found at FAR52.227 and DFARS252.227 et seq. and any successor
regulations thereof. Use of this material by the U.S. Government and/or DOD
agencies is acknowledgment of the proprietary rights of any copyright holders
and contributors, including those of Advanced Micro Devices, Inc., as well as
the provisions of FAR52.227-14 through 23 regarding privately developed and/or
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and
supersedes all proposals and prior discussions and writings between the parties
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be
modified or waived, and no breach of this license can be excused, unless done
so in a writing signed by all affected parties. Each term of this license is
separately enforceable. If any term of this license is determined to be or
becomes unenforceable or illegal, such term shall be reformed to the minimum
extent necessary in order for this license to remain in effect in accordance
with its terms as modified by such reformation. This license shall be governed
by and construed in accordance with the laws of the State of Texas without
regard to rules on conflicts of law of any state or jurisdiction or the United
Nations Convention on the International Sale of Goods. All disputes arising out
of this license shall be subject to the jurisdiction of the federal and state
courts in Austin, Texas, and all defenses are hereby waived concerning personal
jurisdiction and venue of these courts.

============================================================ */

#ifndef __CAL_H__
#define __CAL_H__

#ifdef __cplusplus
extern "C" {
#endif

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

/** Device Kernel ISA */
typedef enum CALtargetEnum {
    CAL_TARGET_600,                 /**< R600 GPU ISA */
    CAL_TARGET_610,                 /**< RV610 GPU ISA */
    CAL_TARGET_630,                 /**< RV630 GPU ISA */
    CAL_TARGET_670,                 /**< RV670 GPU ISA */
    CAL_TARGET_7XX,                 /**< R700 class GPU ISA */
    CAL_TARGET_770,                 /**< RV770 GPU ISA */
    CAL_TARGET_710,                 /**< RV710 GPU ISA */
    CAL_TARGET_730,                 /**< RV730 GPU ISA */
    CAL_TARGET_CYPRESS,             /**< CYPRESS GPU ISA */
    CAL_TARGET_JUNIPER,             /**< JUNIPER GPU ISA */
    CAL_TARGET_REDWOOD,             /**< REDWOOD GPU ISA */
    CAL_TARGET_CEDAR,               /**< CEDAR GPU ISA */
//##BEGIN_PRIVATE##
    CAL_TARGET_SUMO,                /**< SUMO GPU ISA */
    CAL_TARGET_SUPERSUMO,           /**< SUPERSUMO GPU ISA */
    CAL_TARGET_WRESTLER,            /**< WRESTLER GPU ISA */
    CAL_TARGET_CAYMAN,              /**< CAYMAN GPU ISA */
    CAL_TARGET_KAUAI,               /**< KAUAI GPU ISA */
    CAL_TARGET_BARTS ,              /**< BARTS GPU ISA */
    CAL_TARGET_TURKS ,              /**< TURKS GPU ISA */
    CAL_TARGET_CAICOS,              /**< CAICOS GPU ISA */
    CAL_TARGET_TAHITI,              /**< TAHITI GPU ISA*/
    CAL_TARGET_PITCAIRN,            /**< PITCAIRN GPU ISA*/
    CAL_TARGET_CAPEVERDE,           /**< CAPE VERDE GPU ISA*/
    CAL_TARGET_DEVASTATOR,          /**< DEVASTATOR GPU ISA*/
    CAL_TARGET_SCRAPPER,            /**< SCRAPPER GPU ISA*/
    CAL_TARGET_OLAND,               /**< OLAND GPU ISA*/
    CAL_TARGET_BONAIRE,             /**< BONAIRE GPU ISA*/
    CAL_TARGET_SPECTRE,             /**< KAVERI1 GPU ISA*/
    CAL_TARGET_SPOOKY,              /**< KAVERI2 GPU ISA*/
    CAL_TARGET_KALINDI,             /**< KALINDI GPU ISA*/
    CAL_TARGET_HAINAN,              /**< HAINAN GPU ISA*/
    CAL_TARGET_HAWAII,              /**< HAWAII GPU ISA*/
    CAL_TARGET_ICELAND,             /**< ICELAND GPU ISA*/
    CAL_TARGET_TONGA,               /**< TONGA GPU ISA*/
    CAL_TARGET_GODAVARI,            /**< MULLINS GPU ISA*/
    CAL_TARGET_BERMUDA,             /**< BERMUDA GPU ISA*/
    CAL_TARGET_FIJI,                /**< FIJI GPU ISA*/
    CAL_TARGET_CARRIZO,             /**< CARRIZO GPU ISA*/
    CAL_TARGET_LAST = CAL_TARGET_CARRIZO, /**< last */
//##END_PRIVATE##
} CALtarget;

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
    bool       priSupport;                          /**< IOMMUv2 ATS/PRI support */
    CALuint64  vaStart;                             /**< VA start address */
    CALuint64  vaEnd;                               /**< VA end address */
    bool       isWorkstation;                       /**< Whether Device is a Workstation/Server part */
    CALuint    numOfVpu;                            /**< number of vpu in the device*/
    bool       isOpenCL200Device;                    /**< the flag to mark if the device is OpenCL 200*/
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

#ifdef __cplusplus
}      /* extern "C" { */
#endif


#endif /* __CAL_H__ */

