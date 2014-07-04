//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
// This is a compatibility header file. Either define the version
// of the compiler library to use or include the version specific
// header file directly.
#ifndef TARGET_MAPPINGS_H_
#define TARGET_MAPPINGS_H_
#if WITH_VERSION_0_8
#include "v0_8/target_mappings.h"

#define X86TargetMapping X86TargetMapping_0_8
#define X64TargetMapping X64TargetMapping_0_8
#define AMDILTargetMapping AMDILTargetMapping_0_8
#define HSAILTargetMapping HSAILTargetMapping_0_8
#define AMDIL64TargetMapping AMDIL64TargetMapping_0_8
#define HSAIL64TargetMapping HSAIL64TargetMapping_0_8
#elif WITH_VERSION_0_9
#include "v0_9/target_mappings.h"

#define X86TargetMapping X86TargetMapping_0_9
#define X64TargetMapping X64TargetMapping_0_9
#define AMDILTargetMapping AMDILTargetMapping_0_9
#define HSAILTargetMapping HSAILTargetMapping_0_9
#define AMDIL64TargetMapping AMDIL64TargetMapping_0_9
#define HSAIL64TargetMapping HSAIL64TargetMapping_0_9
#define A32TargetMapping A32TargetMapping_0_9
#define A64TargetMapping A64TargetMapping_0_9
#else
#error "The compiler library version was not defined."
#include "v0_8/target_mappings.h"

#define X86TargetMapping X86TargetMapping_0_8
#define X64TargetMapping X64TargetMapping_0_8
#define AMDILTargetMapping AMDILTargetMapping_0_8
#define HSAILTargetMapping HSAILTargetMapping_0_8
#define AMDIL64TargetMapping AMDIL64TargetMapping_0_8
#define HSAIL64TargetMapping HSAIL64TargetMapping_0_8
#endif
#endif // TARGET_MAPPINGS_H_
