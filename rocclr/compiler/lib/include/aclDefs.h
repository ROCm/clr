//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
// This is a compatibility header file. Either define the version
// of the compiler library that is to be used or include the
// header file for that version directly.
#ifndef ACL_DEFS_H_
#define ACL_DEFS_H_
#if WITH_VERSION_0_8
#include "v0_8/aclDefs.h"
#elif WITH_VERSION_0_9
#else
#error "The compiler library version was not defined."
#include "v0_8/aclDefs.h"
#endif
#endif // ACL_DEFS_H_
