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

//
// Video Extension
//

typedef struct CALvideoPropertiesRec    CALvideoProperties;
typedef struct CALprogramVideoRec       CALprogramVideo;
typedef struct CALdeviceVideoAttribsRec CALdeviceVideoAttribs;
typedef struct CALcontextPropertiesRec  CALcontextProperties;
typedef struct CALprogramVideoDecodeRec CALprogramVideoDecode;
typedef struct CALprogramVideoEncodeRec CALprogramVideoEncode;
typedef struct CALvideoAttribRec        CALvideoAttrib;
typedef struct CALvideoEncAttribRec     CALvideoEncAttrib;

// VCE
typedef struct CALEncodeCreateVCERec                     CALEncodeCreateVCE;
typedef struct CALEncodeGetDeviceInfoRec                 CALEncodeGetDeviceInfo;
typedef struct CALEncodeGetNumberOfModesRec              CALEncodeGetNumberOfModes;
typedef struct CALEncodeGetModesRec                      CALEncodeGetModes;
typedef struct CALEncodeGetDeviceCAPRec                  CALEncodeGetDeviceCAP;
typedef struct CALEncodeSetStateRec                      CALEncodeSetState;
typedef struct CALEncodeGetPictureControlConfigRec       CALEncodeGetPictureControlConfig;
typedef struct CALEncodeGetRateControlConfigRec          CALEncodeGetRateControlConfig;
typedef struct CALEncodeGetMotionEstimationConfigRec     CALEncodeGetMotionEstimationConfig;
typedef struct CALEncodeGetRDOControlConfigRec           CALEncodeGetRDOControlConfig;

typedef enum
{
    CAL_VID_NV12_INTERLEAVED = 1,// NV12
    CAL_VID_YV12_INTERLEAVED,   // YV12
} CALdecodeFormat;

typedef enum
{
    CAL_VID_H264_BASELINE = 1,  // H.264 bitstream acceleration baseline profile
    CAL_VID_H264_MAIN,          // H.264 bitstream acceleration main profile
    CAL_VID_H264_HIGH,          // H.264 bitstream acceleration high profile
    CAL_VID_VC1_SIMPLE,         // VC-1 bitstream acceleration simple profile
    CAL_VID_VC1_MAIN,           // VC-1 bitstream acceleration main profile
    CAL_VID_VC1_ADVANCED,       // VC-1 bitstream acceleration advanced profile
    CAL_VID_MPEG2_VLD,          // MPEG2 bitstream acceleration VLD profile
} CALdecodeProfile;

typedef enum
{
    CAL_VID_ENC_H264_BASELINE = 1,  // H.264 bitstream acceleration baseline profile
    CAL_VID_ENC_H264_MAIN,          // H.264 bitstream acceleration main profile
    CAL_VID_ENC_H264_HIGH,          // H.264 bitstream acceleration high profile
} CALencodeProfile;

typedef enum
{
    CAL_CONTEXT_VIDEO      = 1,
    CAL_CONTEXT_3DCOMPUTE  = 2,
    CAL_CONTEXT_COMPUTE0   = 3,
    CAL_CONTEXT_COMPUTE1   = 4,
    CAL_CONTEXT_DRMDMA0    = 5,
    CAL_CONTEXT_DRMDMA1    = 6,
    CAL_CONTEXT_VIDEO_VCE,
    CALcontextEnum_FIRST = CAL_CONTEXT_VIDEO,
    CALcontextEnum_LAST  = CAL_CONTEXT_VIDEO_VCE,
} CALcontextEnum;

typedef enum
{
    CAL_PRIORITY_NEUTRAL = 0,
    CAL_PRIORITY_HIGH    = 1,
    CAL_PRIORITY_LOW     = 2
} CALpriorityEnum;


typedef enum
{
    CAL_VIDEO_DECODE = 1,
    CAL_VIDEO_ENCODE = 2
} CALvideoType;

struct CALcontextPropertiesRec
{
    CALcontextEnum name;
    CALpriorityEnum priority;
    CALvoid*       data;
};

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

struct CALvideoPropertiesRec
{
    CALuint          size;
    CALuint          flags;
    CALdecodeProfile profile;
    CALdecodeFormat  format;
    CALuint          width;
    CALuint          height;
    CALcontextEnum   VideoEngine_name;
};

struct CALprogramVideoRec
{
    CALuint      size;
    CALvideoType type;
    CALuint      flags;
};

struct CALprogramVideoDecodeRec
{
    CALprogramVideo videoType;
    void*           picture_parameter_1;
    void*           picture_parameter_2;
    CALuint         picture_parameter_2_size;
    void*           bitstream_data;
    CALuint         bitstream_data_size;
    void*           slice_data_control;
    CALuint         slice_data_size;
};

struct CALprogramVideoEncodeRec
{
    CALprogramVideo videoType;
    CALuint         pictureParam1Size;
    CALuint         pictureParam2Size;
    void*           pictureParam1;
    void*           pictureParam2;
    CALuint         uiTaskID;
};

struct CALvideoAttribRec
{
    CALdecodeProfile decodeProfile;
    CALdecodeFormat  decodeFormat;
};

struct CALvideoEncAttribRec
{
    CALencodeProfile encodeProfile;
    CALdecodeFormat  encodeFormat;  // decode format is the same as the encode format
};

struct CALdeviceVideoAttribsRec
{
    CALuint                 data_size;  // in - size of the struct,
                                        // out - bytes of data incl. pointed to
    CALuint                 max_decode_sessions;
    const CALvideoAttrib*   video_attribs;  // list of supported
                                            // profile/format pairs
    const CALvideoEncAttrib*      video_enc_attribs;
};


////// VCE
struct CALEncodeCreateVCERec
{
    CALvoid*    VCEsession;
};

struct CALEncodeGetDeviceInfoRec
{
    unsigned int     device_id;
    unsigned int     max_encode_stream;
    unsigned int     encode_cap_list_size;
};

struct CALEncodeGetNumberOfModesRec
{
    unsigned int num_of_encode_Mode;
};

typedef enum
{
    CAL_VID_encode_MODE_NONE      = 0,
    CAL_VID_encode_AVC_FULL       = 1,
    CAL_VID_encode_AVC_ENTROPY    = 2,
} CALencodeMode;

struct CALEncodeGetModesRec
{
    CALuint          NumEncodeModesToRetrieve;
    CALencodeMode *pEncodeModes;
};

typedef enum
{
    CAL_VID_ENCODE_JOB_PRIORITY_NONE   = 0,
    CAL_VID_ENCODE_JOB_PRIORITY_LEVEL1 = 1,      // Always in normal queue
    CAL_VID_ENCODE_JOB_PRIORITY_LEVEL2 = 2       // possibly in low-latency queue
}   CAL_VID_ENCODE_JOB_PRIORITY;

typedef struct _CAL_VID_PROFILE_LEVEL
{
    CALuint      profile;        //based on  H.264 standard
    CALuint      level;
} CAL_VID_PROFILE_LEVEL;

typedef enum
{
    CAL_VID_PICTURE_NONOE  = 0,
    CAL_VID_PICTURE_NV12   = 1,
} CAL_VID_PICTURE_FORMAT;

#define CAL_VID_MAX_NUM_PICTURE_FORMATS_H264_AVC      10
#define CAL_VID_MAX_NUM_PROFILE_LEVELS_H264_AVC    20

typedef struct
{
    CALuint                       maxPicSizeInMBs;    // Max picture size in MBs
    CALuint                       minPicSizeInMBs;     // Min picture size in MBs
    CALuint                       numPictureFormats;   // number of supported picture formats
    CAL_VID_PICTURE_FORMAT        supportedPictureFormats[CAL_VID_MAX_NUM_PICTURE_FORMATS_H264_AVC];
    CALuint                       numProfileLevels;     // number of supported profiles/levels returne;
    CAL_VID_PROFILE_LEVEL         supportedProfileLevel[CAL_VID_MAX_NUM_PROFILE_LEVELS_H264_AVC];
    CALuint                       maxBitRate;               // Max bit rate
    CALuint                       minBitRate;                // min bit rate
    CAL_VID_ENCODE_JOB_PRIORITY   supportedJobPriority;// supported max level of job priority
}CAL_VID_ENCODE_CAPS_FULL;

typedef struct
{
    CAL_VID_ENCODE_JOB_PRIORITY  supportedJobPriority;// supported max level of job priority
    CALuint                      maxJobQueueDepth;    // Max job queue depth
}CAL_VID_ENCODE_CAPS_ENTROPY;

typedef struct
{
    CALencodeMode  EncodeModes;
    CALuint        encode_cap_size;
    union
    {
       CAL_VID_ENCODE_CAPS_FULL      *encode_cap_full;
       CAL_VID_ENCODE_CAPS_ENTROPY   *encode_cap_entropy;
       void                          *encode_cap;
    }  caps;
} CAL_VID_ENCODE_CAPS;

struct CALEncodeGetDeviceCAPRec
{
    CALuint               num_of_encode_cap;
    CAL_VID_ENCODE_CAPS  *encode_caps;
};



typedef enum
{
    CAL_VID__ENCODE_STATE_START       = 1,
    CAL_VID__ENCODE_STATE_PAUSE       = 2,
    CAL_VID__ENCODE_STATE_RESUME      = 3,
    CAL_VID__ENCODE_STATE_STOP        = 4
} CAL_VID_ENCODE_STATE   ;

typedef struct
{
    CALuint             size;                                  // structure size

    CALuint             useConstrainedIntraPred;        // binary var - force the use of constrained intra prediction when set to 1
    //CABAC options
    CALuint             cabacEnable;                    // Enable CABAC entropy coding
    CALuint             cabacIDC;                       // cabac_init_id = 0; cabac_init_id = 1; cabac_init_id = 2;

    CALuint             loopFilterDisable;              // binary var - disable loop filter when 1 - enable loop filter when 0 (0 and 1 are the only two supported cases)
    int                encLFBetaOffset;                // -- move with loop control flag , Loop filter control, slice_beta_offset (N.B. only used if deblocking filter is not disabled, and there is no div2 as defined in the h264 bitstream syntax)
    int                encLFAlphaC0Offset;             // Loop filter control, slice_alpha_c0_offset (N.B. only used if deblocking filter is not disabled, and there is no div2 as defined in the h264 bitstream syntax)
    CALuint             encIDRPeriod;
    CALuint             encIPicPeriod;                  // spacing for I pictures, in case driver doesnt force/select a picture type, this will be used for inference
    int                encHeaderInsertionSpacing;      // spacing for inserting SPS/PPS. Example usage cases are: 0 for inserting at the beginning of the stream only, 1 for every picture, "GOP size" to align it with GOP boundaries etc. For compliance reasons, these headers might be inserted when SPS/PPS parameters change from the config packages.
    CALuint             encCropLeftOffset;
    CALuint             encCropRightOffset;
    CALuint             encCropTopOffset;
    CALuint             encCropBottomOffset;
    CALuint             encNumMBsPerSlice;              // replaces encSliceArgument - Slice control - number of MBs per slice
    CALuint             encNumSlicesPerFrame;           // Slice control - number of slices in this frame, pre-calculated to avoid DIV operation in firmware
    CALuint             encForceIntraRefresh;           // 1 serves to load intra refresh bitmap from address force_intra_refresh_bitmap_mc_addr when equal to 1, 3 also loads dirty clean bitmap on top of the intra refresh
    CALuint             encForceIMBPeriod;              // --- package with intra referesh -Intra MB spacing. if encForceIntraRefresh = 2, shifts intra refreshed MBs by frame number
    CALuint             encInsertVUIParam;              // insert VUI params in SPS
    CALuint             encInsertSEIMsg;                // insert SEI messages (bit 0 for buffering period; bit 1 for picture timing; bit 2 for pan scan)
} CAL_VID_ENCODE_PICTURE_CONTROL;

typedef struct
{
    CALuint        size;                       // structure size
    CALuint        encRateControlMethod;           // rate control method to be used
    CALuint        encRateControlTargetBitRate;    // target bit rate
    CALuint        encRateControlPeakBitRate;      // peak bit rate
    CALuint        encRateControlFrameRateNumerator;  // target frame rate
    CALuint        encGOPSize;                     // RC GOP size
    CALuint        encRCOptions;                   // packed bitfield definition for extending options here, bit 0: RC will not generate skipped frames in order to meet GOP target, bits 1-30: up for grabs by the RC alg designer
    CALuint        encQP_I;                        // I frame quantization only if rate control is disabled
    CALuint        encQP_P;                        // P frame quantization if rate control is disabled
    CALuint        encQP_B;                        // B frame quantization if rate control is disabled
    CALuint        encVBVBufferSize;               // VBV buffer size - this is CPB Size, and the default is per Table A-1 of the spec
    CALuint        encRateControlFrameRateDenominator;// target frame rate
} CAL_VID_ENCODE_RATE_CONTROL;

 // mode estimation control options
typedef struct
{
    CALuint              size;                   // structure size
    CALuint             imeDecimationSearch;            // decimation search is on
    CALuint             motionEstHalfPixel;             // enable half pel motion estimation
    CALuint             motionEstQuarterPixel;          // enable quarter pel motion estimation
    CALuint             disableFavorPMVPoint;           // disable favorization of PMV point
    CALuint             forceZeroPointCenter;           // force [0,0] point as search window center in IME
    CALuint             lsmVert;                        //  Luma Search window in MBs, set to either VCE_ENC_SEARCH_WIND_5x3 or VCE_ENC_SEARCH_WIND_9x5 or VCE_ENC_SEARCH_WIND_13x7
    CALuint             encSearchRangeX;                // forward prediction - Manual limiting of horizontal motion vector range (for performance) in pel resolution
    CALuint             encSearchRangeY;                // forward prediction - Manual limiting of vertical motion vector range (for performance)
    CALuint             encSearch1RangeX;               // for 2nd ref - curr IME_SEARCH_SIZE doesn't have SIZE__SEARCH1_X bitfield
    CALuint             encSearch1RangeY;               // for 2nd ref
    CALuint             disable16x16Frame1;             // second reference (B frame) limitation
    CALuint             disableSATD;                    // Disable SATD cost calculation (SAD only)
    CALuint             enableAMD;                      // FME advanced mode decision
    CALuint             encDisableSubMode;              // --- FME
    CALuint             encIMESkipX;                    // sub sample search window horz --- UENC_IME_OPTIONS.SKIP_POINT_X
    CALuint             encIMESkipY;                    // sub sample search window vert --- UENC_IME_OPTIONS.SKIP_POINT_Y
    CALuint             encEnImeOverwDisSubm;           // Enable overwriting of fme_disable_submode in IME with enabled mode number equal to ime_overw_dis_subm_no (only 8x8 and above could be enabled)
    CALuint             encImeOverwDisSubmNo;           // Numbers of mode IME will pick if en_ime_overw_dis_subm equal to 1.
    CALuint             encIME2SearchRangeX;            // IME Additional Search Window Size: horizontal 1-4 (+- this value left and right from center)
    CALuint             encIME2SearchRangeY;            // IME Additional Search Window Size: vertical not-limited (+- this value up and down from center)
                                                //   (+- this value up and down from center)
} CAL_VID_ENCODE_MOTION_ESTIMATION_CONTROL;                  // structure aligned to 88 bytes

typedef struct
{
    CALuint         size;                                  // structure size
    CALuint         encDisableTbePredIFrame;            // Disable Prediction Modes For I-Frames
    CALuint         encDisableTbePredPFrame;            // same as above for P frames
    CALuint         useFmeInterpolY;                    // zero_residues_luma
    CALuint         useFmeInterpolUV;                   // zero_residues_chroma
    CALuint         enc16x16CostAdj;                    // --- UENC_FME_MD.M16x16_COST_ADJ
    CALuint         encSkipCostAdj;                     // --- UENC_FME_MD.MSkip_COST_ADJ
    unsigned char   encForce16x16skip;
} CAL_VID_ENCODE_RDO_CONTROL;


struct CALEncodeSetStateRec
{
    CAL_VID_ENCODE_STATE  encode_states;
};
struct CALEncodeGetPictureControlConfigRec
{
    CAL_VID_ENCODE_PICTURE_CONTROL  encode_picture_control;
};
struct CALEncodeGetRateControlConfigRec
{
    CAL_VID_ENCODE_RATE_CONTROL  encode_rate;
};
struct CALEncodeGetMotionEstimationConfigRec
{
    CAL_VID_ENCODE_MOTION_ESTIMATION_CONTROL  encode_motion_estimation;
};
struct CALEncodeGetRDOControlConfigRec
{
    CAL_VID_ENCODE_RDO_CONTROL  encode_RDO;
};

typedef enum
{
    CAL_VID_CONFIG_TYPE_NONE               = 0,
    CAL_VID_CONFIG_TYPE_PICTURECONTROL       = 1,
    CAL_VID_CONFIG_TYPE_RATECONTROL          = 2,
    CAL_VID_CONFIG_TYPE_MOTIONSESTIMATION = 3,
    CAL_VID_CONFIG_TYPE_RDO               = 4
} CAL_VID_CONFIG_TYPE;

typedef struct
{
    CAL_VID_CONFIG_TYPE                    configType;
    union
    {
        CAL_VID_ENCODE_PICTURE_CONTROL*            pPictureControl;
        CAL_VID_ENCODE_RATE_CONTROL*               pRateControl;
        CAL_VID_ENCODE_MOTION_ESTIMATION_CONTROL*  pMotionEstimation;
        CAL_VID_ENCODE_RDO_CONTROL*                pRDO;
    }    config;
} CAL_VID_CONFIG;

typedef enum
{
    CAL_VID_PICTURE_STRUCTURE_H264_NONE         = 0,
    CAL_VID_PICTURE_STRUCTURE_H264_FRAME        = 1,
    CAL_VID_PICTURE_STRUCTURE_H264_TOP_FIELD    = 2,
    CAL_VID_PICTURE_STRUCTURE_H264_BOTTOM_FIELD = 3
} CAL_VID_PICTURE_STRUCTURE_H264;

// Used to force picture type
typedef enum _CU_VID_PICTURE_TYPE_H264
{
    CAL_VID_PICTURE_TYPE_H264_NONE                  = 0,
    CAL_VID_PICTURE_TYPE_H264_SKIP                  = 1,
    CAL_VID_PICTURE_TYPE_H264_IDR                   = 2,
    CAL_VID_PICTURE_TYPE_H264_I                     = 3,
    CAL_VID_PICTURE_TYPE_H264_P                     = 4
} CAL_VID_PICTURE_TYPE_H264;

typedef union _CAL_VID_ENCODE_PARAMETERS_H264_FLAGS
{
    struct
    {
        // enable/disable features
        unsigned int                            reserved    : 32;   // reserved fields must be set to zero
    }                                           flags;
    unsigned int                                value;
}CAL_VID_ENCODE_PARAMETERS_H264_FLAGS;

typedef struct
{
    CALuint                              size;  // structure size. Must be always set to the size of AVE_ENCODE_PARAMETERS_H264.

    CAL_VID_ENCODE_PARAMETERS_H264_FLAGS flags; // enable/disable any supported features

    CALboolean                           insertSPS;
    CAL_VID_PICTURE_STRUCTURE_H264       pictureStructure;
    CALboolean                           forceRefreshMap;
    CALuint                              forceIMBPeriod;
    CAL_VID_PICTURE_TYPE_H264            forcePicType;
} CAL_VID_ENCODE_PARAMETERS_H264;

typedef enum
{
    CAL_VID_BUFFER_TYPE_NONE                            = 0,
    CAL_VID_BUFFER_TYPE_ENCODE_PARAM_H264               = 1,
    CAL_VID_BUFFER_TYPE_PICTURE                         = 2,
    CAL_VID_BUFFER_TYPE_SLICE_HEADER                    = 3,
    CAL_VID_BUFFER_TYPE_SLICE                           = 4,
    CAL_VID_BUFFER_TYPE_RECONSTRUCTED_PICTURE_OUTPUT    = 5
} CAL_VID_BUFFER_TYPE;

#define CAL_VID_SURFACE_HANDLE                      void*

typedef struct
{
    CAL_VID_BUFFER_TYPE           bufferType;
    union
    {
        CAL_VID_ENCODE_PARAMETERS_H264*  pEncodeParamH264;
        CAL_VID_SURFACE_HANDLE           pPicture;
        CAL_VID_SURFACE_HANDLE           pSliceHeader;
        CAL_VID_SURFACE_HANDLE           pSlice;
        CAL_VID_SURFACE_HANDLE           pReconstructedPictureOutput;

    }   buffer;
} CAL_VID_BUFFER_DESCRIPTION;

typedef enum
{
    CAL_VID_TASK_STATUS_NONE        = 0,
    CAL_VID_TASK_STATUS_COMPLETE    = 1,    // encoding task has finished successfully.
    CAL_VID_TASK_STATUS_FAILED      = 2     // encoding task has finished but failed.
} CAL_VID_TASK_STATUS;

typedef struct
{
    CALuint             size;                   // structure size
    CALuint             taskID;                 // task ID
    CAL_VID_TASK_STATUS status;                 // Task status. May be duplicated if current task has multiple output blocks.
    CALuint             size_of_bitstream_data; // data size of the output block
    void*               bitstream_data;         // read pointer the top portion of the generated bitstream data for the current task
} CAL_VID_OUTPUT_DESCRIPTION;

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
