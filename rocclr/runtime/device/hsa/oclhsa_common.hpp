//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef _OPENCL_RUNTIME_DEVICE_HSA_OCLHSA_COMMON_HPP_
#define _OPENCL_RUNTIME_DEVICE_HSA_OCLHSA_COMMON_HPP_

#include "hsacore_symbol_loader.hpp"
#include "services_symbol_loader.hpp"

#include "hsacoreagent.h"
#include "hsaagent.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const HsaCoreApiTable *hsacoreapi;
extern const HsaServicesApiTable *servicesapi;


#ifdef __cplusplus
}
#endif

#endif // header guard
