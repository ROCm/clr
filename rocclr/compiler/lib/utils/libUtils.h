//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
// This is a compatibility header file. Either define the version
// of the compiler library to use or include the version specific
// header file directly.
#ifndef _CL_LIB_UTILS_H_
#define _CL_LIB_UTILS_H_
#if WITH_VERSION_0_8
#include "v0_8/libUtils.h"
#elif WITH_VERSION_0_9
#include "v0_9/libUtils.h"
#else
#error "The compiler library version was not defined."
#include "v0_8/libUtils.h"
#endif
#endif // _CL_LIB_UTILS_H_
