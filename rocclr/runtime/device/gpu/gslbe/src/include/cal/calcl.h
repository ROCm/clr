/**
 *  @file     calcl.h
 *  @brief    CAL Compiler Interface Header
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

#ifndef __CALCL_H__
#define __CALCL_H__

#include "cal.h"
#include "gsl_enum.h"
#include "gsl_types.h"
#include "cm_enum.h"

#ifdef __cplusplus
extern "C" {
#endif

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

typedef  struct CALDeviceGLParamsRec {
    CALvoid            *GLplatformContext;
    CALvoid            *GLdeviceContext;
    CALuint            flags;
} CALDeviceGLParams;

#ifdef __cplusplus
}      /* extern "C" { */
#endif


#endif /* __CALCL_H__ */
