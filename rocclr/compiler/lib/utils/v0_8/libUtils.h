//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef _CL_LIB_UTILS_0_8_H_
#define _CL_LIB_UTILS_0_8_H_
#include "v0_8/aclTypes.h"
#include <string>
#include <sstream>
#include <iterator>
#include "library.hpp"
// Utility function to set a flag in option structure
// of the aclDevCaps.
void
setFlag(aclDevCaps *elf, compDeviceCaps option);

// Utility function to flip a flag in option structure
// of the aclDevCaps.
void
flipFlag(aclDevCaps *elf, compDeviceCaps option);

// Utility function to clear a flag in option structure
// of the aclDevCaps.
void
clearFlag(aclDevCaps *elf, compDeviceCaps option);

// Utility function to check that a flag in option structure
// of the aclDevCaps is set.
bool
checkFlag(aclDevCaps *elf, compDeviceCaps option);

// Utility function to initialize and elf device capabilities
void
initElfDeviceCaps(aclBinary *elf);

// Append the string to the aclCompiler log string.
void
appendLogToCL(aclCompiler *cl, const std::string &logStr);

const char *getDeviceName(const aclTargetInfo &Target);

// Select the correct library from the target information.
amd::LibrarySelector getLibraryType(const aclTargetInfo *Target);

// get family_enum from the target information.
unsigned getFamilyEnum(const aclTargetInfo *Target);

// get chip_enum from the target information.
unsigned getChipEnum(const aclTargetInfo *Target);

// Create a copy of an ELF and duplicate all sections/symbols
aclBinary*
createELFCopy(aclBinary *src);

// Create a BIF2.1 elf from a BIF 2.0 elf
aclBinary*
convertBIF20ToBIF21(aclBinary *src);

// Create a BIF3.0 elf from a BIF 2.0 elf
aclBinary*
convertBIF20ToBIF30(aclBinary *src);

// Create a BIF2.0 elf from a BIF 2.1 elf
aclBinary*
convertBIF21ToBIF20(aclBinary *src);

// Create a BIF3.0 elf from a BIF 2.1 elf
aclBinary*
convertBIF21ToBIF30(aclBinary *src);

// Create a BIF2.0 elf from a BIF 3.0 elf
aclBinary*
convertBIF30ToBIF20(aclBinary *src);

// Create a BIF2.1 elf from a BIF 3.0 elf
aclBinary*
convertBIF30ToBIF21(aclBinary *src);

// get a pointer to the aclBIF irrespective of the
// binary version.
aclBIF*
aclutGetBIF(aclBinary*);

// Get a pointer to the aclOptions irrespective of
// the binary version.
aclOptions*
aclutGetOptions(aclBinary*);

// Get a pointer to the aclBinaryOptions struct
// irrespective of the binary version.
aclBinaryOptions*
aclutGetBinOpts(aclBinary*);

// Get a pointer to the target info struct
// irrespective of the binary version.
aclTargetInfo*
aclutGetTargetInfo(aclBinary*);

// Get a pointer to the device caps
// irrespective of the binary version.
aclDevCaps*
aclutGetCaps(aclBinary*);

// Copy two binary option structures irrespective
// of the binary version and uses defaults when
// things don't match up.
void
aclutCopyBinOpts(aclBinaryOptions *dst,
    const aclBinaryOptions *src,
    bool is64bit);

// Retrieve kernel statistics from binary
// and insert to elf as symbol
acl_error aclutInsertKernelStatistics(aclCompiler*, aclBinary*);

// Helper function that returns the
// allocation function from the binary.
AllocFunc
aclutAlloc(const aclBinary *bin);

// Helper function that returns the
// de-allocation function from the binary.
FreeFunc
aclutFree(const aclBinary *bin);


// Helper function that returns the
// allocation function from the compiler.
AllocFunc
aclutAlloc(const aclCompiler *bin);

// Helper function that returns the
// de-allocation function from the compiler.
FreeFunc
aclutFree(const aclCompiler *bin);

// Helper function that returns the
// allocation function from the compiler options.
AllocFunc
aclutAlloc(const aclCompilerOptions *bin);

// Helper function that returns the
// de-allocation function from the compiler options.
FreeFunc
aclutFree(const aclCompilerOptions *bin);

// Helper predicate
// returns true if p starts with bit code signature.
bool
isBcMagic(const char* p);

inline bool is64BitTarget(const aclTargetInfo& target)
{
  return (target.arch_id == aclX64 || 
          target.arch_id == aclAMDIL64 || 
          target.arch_id == aclHSAIL64);
}

inline bool isCpuTarget(const aclTargetInfo& target)
{
  return (target.arch_id == aclX64 || target.arch_id == aclX86);
}

inline bool isGpuTarget(const aclTargetInfo& target)
{
  return (target.arch_id == aclAMDIL || target.arch_id == aclAMDIL64 ||
          target.arch_id == aclHSAIL || target.arch_id == aclHSAIL64);
}

inline bool isAMDILTarget(const aclTargetInfo& target)
{
  return (target.arch_id == aclAMDIL || target.arch_id == aclAMDIL64);
}

inline bool isHSAILTarget(const aclTargetInfo& target)
{
  return (target.arch_id == aclHSAIL || target.arch_id == aclHSAIL64);
}

enum scId {
  SC_AMDIL = 0,
  SC_HSAIL = 1,
  SC_LAST,
};

inline std::vector<std::string> splitSpaceSeparatedString(char *str)
{
  std::string s(str);
  std::stringstream ss(s);
  std::istream_iterator<std::string> beg(ss), end;
  std::vector<std::string> vec(beg, end);
  return vec;
}

// Helper function that allocates an aligned memory.
inline void*
alignedMalloc(size_t size, size_t alignment)
{
#if defined(_WIN32)
  return ::_aligned_malloc(size, alignment);
#else
  void * ptr = NULL;
  if (0 == ::posix_memalign(&ptr, alignment, size)) {
    return ptr;
  }
  return NULL;
#endif
}

// Helper function that frees an aligned memory.
inline void
alignedFree(void *ptr)
{
#if defined(_WIN32)
  ::_aligned_free(ptr);
#else
  free(ptr);
#endif
}

#endif // _CL_LIB_UTILS_0_8_H_
