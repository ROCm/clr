/*****************************************************************************
 *
 *
 *
 *  Trade secret of ATI Technologies, Inc.
 *  Copyright 2000, ATI Technologies, Inc., (unpublished)
 *
 *  All rights reserved.  This notice is intended as a precaution against
 *  inadvertent publication and does not imply publication or any waiver
 *  of confidentiality.  The year included in the foregoing notice is the
 *  year of creation of the work.
 *
 *
 ****************************************************************************
 */

#ifndef __D3DSHADERDEFS_H__
#define __D3DSHADERDEFS_H__

#define D3DSI_OPCODE_PARAM           (1 << 31)

#define D3DSI_GETCOMMENTSIZE(token)  (((token) & D3DSI_COMMENTSIZE_MASK) >>  \
                                       D3DSI_COMMENTSIZE_SHIFT)

#define D3DSI_GETDSTSHIFT(token)     (((token) & D3DSP_DSTSHIFT_MASK) >> D3DSP_DSTSHIFT_SHIFT)

// D3D uses 2 swizzle bits per component. Define them since they are not 
// available in d3d header files.
#define D3DSP_SWIZZLE_BITS_PER_COMP  2
#define D3DSP_SWIZZLE_XYZW_MASK      0x3

// DST related: Parameter definition writemask shifts - missing from D3D header 
#define D3DSP_WRITEMASK_SHIFT        16
#define D3DSP_WRITEMASK_ASHIFT       19

// DX9 Ref uses 7. But if only upto _X8 & _D8 are supported, the mask should be 3
#define D3DSP_D3D_DSTSHIFT_MASK      3

#define D3DSP_SHADER_TYPE_MASK       0xFFFF0000
#define D3DSP_PS_TYPE                0xFFFF0000
#define D3DSP_VS_TYPE                0xFFFE0000

// This is necessary to avoid a duplicate definition of these functions
// in C++ source files that use this header. These functions are already
// defined in d3dhal.h inside a "#ifdef __cplusplus" block.
#ifndef __cplusplus

// This gets regtype, and also maps D3DSPR_CONSTn to D3DSPR_CONST
// (for easier parsing)
ATI_INLINE DWORD D3DSI_GETREGTYPE_RESOLVING_CONSTANTS(DWORD token) 
{
    DWORD RegType = D3DSI_GETREGTYPE(token);
    switch (RegType)
    {
        case D3DSPR_CONST4:
        case D3DSPR_CONST3:
        case D3DSPR_CONST2:
            return D3DSPR_CONST;
        default:
            return RegType;
    }
}

// The inline function below retrieves register number for an opcode, 
// taking into account that: if the type is a 
// D3DSPR_CONSTn, the register number needs to be remapped.
//
//           D3DSPR_CONST  is for c0-c2047
//           D3DSPR_CONST2 is for c2048-c4095
//           D3DSPR_CONST3 is for c4096-c6143
//           D3DSPR_CONST4 is for c6144-c8191
//
// For example if the instruction token specifies type D3DSPR_CONST4, reg# 3,
// the register number retrieved is 6147.
// For other register types, the register number is just returned unchanged.
ATI_INLINE DWORD D3DSI_GETREGNUM_RESOLVING_CONSTANTS(DWORD   token) 
{
    DWORD RegType = D3DSI_GETREGTYPE(token);
    DWORD RegNum = D3DSI_GETREGNUM(token);

    switch(RegType)
    {
        case D3DSPR_CONST4:
            return RegNum + 6144;
        case D3DSPR_CONST3:
            return RegNum + 4096;
        case D3DSPR_CONST2:
            return RegNum + 2048;
        default:
            return RegNum;
    }
}

#endif // __cplusplus

#define PSTR_MAX_NUMSRCPARAMS 6
#define PSTR_NUM_COMPONENTS_IN_REGISTER 4

#endif // __D3DSHADERDEFS_H__
