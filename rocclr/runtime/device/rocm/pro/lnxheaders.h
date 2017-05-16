//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//

#pragma once

// NOTE: Some of the Linux driver stack's headers don't wrap their C-style interface names in 'extern "C" { ... }'
// blocks when building with a C++ compiler, so we need to add that ourselves.
#if __cplusplus
extern "C"
{
#endif

#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <amdgpu_shared.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

constexpr int32_t InvalidFd = -1; // value representing a invalid file descriptor for Linux

#if __cplusplus
} // extern "C"
#endif
