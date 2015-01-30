/*
 * Copyright (c) 2012 The Khronos Group Inc.  All rights reserved.
 *
 * NOTICE TO KHRONOS MEMBER:
 *
 * AMD has assigned the copyright for this object code to Khronos.
 * This object code is subject to Khronos ownership rights under U.S. and
 * international Copyright laws.
 *
 * Permission is hereby granted, free of charge, to any Khronos Member
 * obtaining a copy of this software and/or associated documentation files
 * (the "Materials"), to use, copy, modify and merge the Materials in object
 * form only and to publish, distribute and/or sell copies of the Materials
 * solely in object code form as part of conformant OpenCL API implementations,
 * subject to the following conditions:
 *
 * Khronos Members shall ensure that their respective ICD implementation,
 * that is installed over another Khronos Members' ICD implementation, will
 * continue to support all OpenCL devices (hardware and software) supported
 * by the replaced ICD implementation. For the purposes of this notice, "ICD"
 * shall mean a library that presents an implementation of the OpenCL API for
 * the purpose routing API calls to different vendor implementation.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Materials.
 *
 * KHRONOS AND AMD MAKE NO REPRESENTATION ABOUT THE SUITABILITY OF THIS
 * SOURCE CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  KHRONOS AND AMD DISCLAIM ALL WARRANTIES
 * WITH REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL KHRONOS OR AMD BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH
 * THE USE OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */
/* $Revision: 11708 $ on $Date: 2010-06-13 23:36:24 -0700 (Sun, 13 Jun 2010) $ */
#ifndef __SPIR_H
#define __SPIR_H

#ifdef __cplusplus
extern "C" {
#endif
// CLS is for CL SPIR.
typedef enum {
  CLS_PRIVATE_AS       = 0,
  CLS_GLOBAL_AS        = 1,
  CLS_CONSTANT_AS      = 2,
  CLS_LOCAL_AS         = 3,
  CLS_GLOBAL_HOST_AS   = 4,
  CLS_CONSTANT_HOST_AS = 5,
  CLS_REGION_AS_AMD    = 16,
} CLS_ADDRESS_SPACES;

// Table 13
typedef enum {
  CLS_ADDRESS_MIRRORED_REPEAT = 0,
  CLS_ADDRESS_REPEAT          = 1,
  CLS_ADDRESS_CLAMP_TO_EDGE   = 2,
  CLS_ADDRESS_CLAMP           = 3,
  CLS_ADDRESS_NONE            = 4
} CLS_ADDRESSING_MODES;

// Table 13
typedef enum {
  CLS_FILTER_NEAREST = 0,
  CLS_FILTER_LINEAR  = 1
} CLS_FILTER_MODE;

// Table 13
typedef enum {
  CLS_NORMALIZED_COORDS_TRUE  = 0,
  CLS_NORMALIZED_COORDS_FALSE = 1
} CLS_NORMALIZED_COORDS;

// Section 2.3
typedef enum {
  CLS_READ_ONLY  = 0,
  CLS_WRITE_ONLY = 1,
  CLS_READ_WRITE = 2,
  CLS_NONE       = 3
} CLS_ACCESS_QUALIFIERS;

// Table 14
typedef enum {
  CLS_ARG_CONST    = 1,
  CLS_ARG_RESTRICT = 2,
  CLS_ARG_VOLATILE = 4,
  CLS_ARG_NONE     = 0,
  CLS_ARG_MASK     = 0xf,
  CLS_ARG_PIPE     = 8
} CLS_ARGTYPE_QUALIFIERS;

static const char* SPIR_TRIPLE_32BIT = "spir-unknown-unknown";
static const char* SPIR_TRIPLE_64BIT = "spir64-unknown-unknown";
static const char* SPIR_DATA_LAYOUT_32BIT = "p:32:32:32-i1:8:8-i8:8:8"
    "-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16"
    "-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
    "-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024";
static const char* SPIR_DATA_LAYOUT_64BIT = "p:64:64:64-i1:8:8-i8:8:8"
    "-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16"
    "-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
    "-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024";

#ifdef __cplusplus
}
#endif
#endif // __SPIR_H
