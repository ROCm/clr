//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _ACL_DEFS_0_8_H_
#define _ACL_DEFS_0_8_H_

#ifndef ACL_API_ENTRY
#if defined(_WIN32) || defined(__CYGWIN__)
#define ACL_API_ENTRY __stdcall
#else
#define ACL_API_ENTRY
#endif
#endif

#ifndef ACL_API_0_8
#define ACL_API_0_8
#endif

#ifndef BIF_API_2_0
#define BIF_API_2_0
#endif

#ifndef BIF_API_2_1
#define BIF_API_2_1
#endif

#ifndef BIF_API_3_0
#define BIF_API_3_0
#endif

#ifndef MAX_HIDDEN_KERNARGS_NUM
#define MAX_HIDDEN_KERNARGS_NUM 6
#else
#error "MAX_HIDDEN_KERNARGS_NUM is already defined"
#endif

#endif // _ACL_DEFS_0_8_H_
