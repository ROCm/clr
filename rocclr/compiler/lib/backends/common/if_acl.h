//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _IF_ACL_H_
#define _IF_ACL_H_
#include "aclTypes.h"
#if WITH_VERSION_0_8
#include "v0_8/if_acl.h"
#elif WITH_VERSION_0_9
#include "v0_9/if_acl.h"
#else
#error "The compiler library version was not defined."
#include "v0_8/if_acl.h"
#endif
#endif // _IF_ACL_H_
