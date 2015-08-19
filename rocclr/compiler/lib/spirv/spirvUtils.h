#ifndef _COMPLIB_SPIRV_UTILS_H
#define _COMPLIB_SPIRV_UTILS_H

#include <cstddef>

bool validateSPIRV(const void *image, size_t length);
bool isSPIRVMagic(const void* image, size_t length);

#endif
