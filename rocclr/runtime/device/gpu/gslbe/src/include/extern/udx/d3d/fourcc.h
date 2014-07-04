//    
//  Workfile: fourcc.h
//
//  Description: FourCC definitions
//
//  Trade secret of ATI Technologies, Inc.
//  Copyright 1999, ATI Technologies, Inc., (unpublished)
//
//  All rights reserved.  This notice is intended as a precaution against
//  inadvertent publication and does not imply publication or any waiver
//  of confidentiality.  The year included in the foregoing notice is the
//  year of creation of the work.
//
//

#ifndef _FOURCC_H_
#define _FOURCC_H_

//#include "atidxinc.h"
 
#define FOURCC_YUY2  MAKEFOURCC('Y','U','Y','2')
#define FOURCC_UYVY  MAKEFOURCC('U','Y','V','Y')
#define FOURCC_YV12  MAKEFOURCC('Y','V','1','2')
#define FOURCC_YUV12 FOURCC_YV12
#define FOURCC_YVU9  MAKEFOURCC('Y','V','U','9')
#define FOURCC_IF09  MAKEFOURCC('I','F','0','9')
#define FOURCC_IMC4  MAKEFOURCC('I','M','C','4')
#define FOURCC_IYUV  MAKEFOURCC('I','Y','U','V')
#define FOURCC_NV11  MAKEFOURCC('N','V','1','1')
#define FOURCC_NV12  MAKEFOURCC('N','V','1','2')
#define FOURCC_NV21  MAKEFOURCC('N','V','2','1')

//Microsoft specific format for WebTV
#define FOURCC_VBID  MAKEFOURCC('V','B','I','D')

#define FOURCC_MCAM  MAKEFOURCC('M','C','A','M')
#define FOURCC_MC12  MAKEFOURCC('M','C','1','2')
#define FOURCC_MCR4  MAKEFOURCC('M','C','R','4')
#define FOURCC_M2IA  MAKEFOURCC('M','2','I','A')
#define FOURCC_M2AM  MAKEFOURCC('M','2','A','M')
#define FOURCC_M2R4  MAKEFOURCC('M','2','R','4')
#define FOURCC_AYUV  MAKEFOURCC('A','Y','U','V')
#define FOURCC_AI44  MAKEFOURCC('A','I','4','4')
#define FOURCC_XENC  MAKEFOURCC('X','E','N','C')

// OpenGL Surfaces
#define FOURCC_OGLZ  MAKEFOURCC('O','G','L','Z')
#define FOURCC_OGNZ  MAKEFOURCC('O','G','N','Z')
#define FOURCC_OGLS  MAKEFOURCC('O','G','L','S')
#define FOURCC_OGNS  MAKEFOURCC('O','G','N','S')
#define FOURCC_OGLT  MAKEFOURCC('O','G','L','T')
#define FOURCC_OGNT  MAKEFOURCC('O','G','N','T')
#define FOURCC_OGLB  MAKEFOURCC('O','G','L','B')

#define FOURCC_DDES  MAKEFOURCC('D','D','E','S')
#define FOURCC_PBSM  MAKEFOURCC('P','B','S','M')

#define FOURCC_ATI1  MAKEFOURCC('A','T','I','1')
#define FOURCC_ATI2  MAKEFOURCC('A','T','I','2')

// Alias of ARGB-8888 for special MM app. to store security content 
#define FOURCC_SORT  MAKEFOURCC('S','O','R','T')
// Alias of YUY2 for special MM app. to store security content 
#define FOURCC_SYV2  MAKEFOURCC('S','Y','V','2')
// Communication surface for special MM app. to enable security content playback
#define FOURCC_EAPI  MAKEFOURCC('E','A','P','I')

// Communication surface
#define FOURCC_ATIC  MAKEFOURCC('A','T','I','C')

// Fake format for exposing DX9c geometry instancing
#define FOURCC_INST  MAKEFOURCC('I','N','S','T')

// Fake format for exposing R2VB support
// must match FOURCC_R2VB in d3d/atir2vb.h
#define FOURCC_R2VB  MAKEFOURCC('R','2','V','B')

// Depth Stencil Texture formats.
#define FOURCC_DF16  MAKEFOURCC('D','F','1','6')
#define FOURCC_DF24  MAKEFOURCC('D','F','2','4')

// FP_11_11_10 format - used internally for optimization
#define FOURCC_FP11  MAKEFOURCC('F','P','1','1')


// Fetch4:
// GET4 is used both as fake format for exposing Fetch4 and as enable value.
// GET1 is used only as disable value.
#define FOURCC_GET4  MAKEFOURCC('G','E','T','4')
#define FOURCC_GET1  MAKEFOURCC('G','E','T','1')

// ATI Compute Abstraction Layer (CAL)
// U8X1 stands for unsigned 8 bits by 1 component
// S6X4 stands for signed 16 bits by 4 components
#define FOURCC_ATIP  MAKEFOURCC('A','T','I','P')
#define FOURCC_U8X1  MAKEFOURCC('U','8','X','1')
#define FOURCC_S8X1  MAKEFOURCC('S','8','X','1')
#define FOURCC_U8X2  MAKEFOURCC('U','8','X','2')
#define FOURCC_S8X2  MAKEFOURCC('S','8','X','2')
#define FOURCC_S8X4  MAKEFOURCC('S','8','X','4')
#define FOURCC_U6X1  MAKEFOURCC('U','6','X','1')
#define FOURCC_S6X1  MAKEFOURCC('S','6','X','1')
#define FOURCC_S6X2  MAKEFOURCC('S','6','X','2')
#define FOURCC_S6X4  MAKEFOURCC('S','6','X','4')

// ATI semaphore, currently used by CAL 
#define FOURCC_SEMA  MAKEFOURCC('S','E','M','A')

#endif // _FOURCC_H_

