//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#include "api/v0_8/aclValidation.h"
#include "libUtils.h"
#include "bif/bifbase.hpp"
#include "utils/target_mappings.h"
#include "utils/versions.hpp"
#include "utils/options.hpp"
#include <cstring>
#include "bif/bif.hpp"
extern aclBinary* constructBinary(size_t struct_version,
    const aclTargetInfo *target,
    const aclBinaryOptions *opts);

static const std::string sgfx700 = "AMD:AMDGPU:7:0:0";
static const std::string sgfx701 = "AMD:AMDGPU:7:0:1";
static const std::string sgfx800 = "AMD:AMDGPU:8:0:0";
static const std::string sgfx801 = "AMD:AMDGPU:8:0:1";
static const std::string sgfx900 = "AMD:AMDGPU:9:0:0";

// Utility function to set a flag in option structure
// of the aclDevCaps.
void
setFlag(aclDevCaps *caps, compDeviceCaps option)
{
  assert((uint32_t)option < ((1 << FLAG_SHIFT_VALUE)  *FLAG_ARRAY_SIZE)
    && "The index passed in is outside of the range of valid values!");
  caps->flags[option >> FLAG_SHIFT_VALUE] |= FLAG_BITLOC(option);
}

// Utility function to flip a flag in option structure
// of the aclDevCaps.
void
flipFlag(aclDevCaps *caps, compDeviceCaps option)
{
  assert((uint32_t)option < ((1 << FLAG_SHIFT_VALUE)  *FLAG_ARRAY_SIZE)
    && "The index passed in is outside of the range of valid values!");
  caps->flags[option >> FLAG_SHIFT_VALUE] ^= FLAG_BITLOC(option);
}

// Utility function to clear a flag in option structure
// of the aclDevCaps.
void
clearFlag(aclDevCaps *caps, compDeviceCaps option)
{
  assert((uint32_t)option < ((1 << FLAG_SHIFT_VALUE)  *FLAG_ARRAY_SIZE)
    && "The index passed in is outside of the range of valid values!");
  caps->flags[option >> FLAG_SHIFT_VALUE] &= ~FLAG_BITLOC(option);
}

// Utility function to check that a flag in option structure
// of the aclDevCaps is set.
bool
checkFlag(aclDevCaps *caps, compDeviceCaps option)
{
  assert((uint32_t)option < ((1 << FLAG_SHIFT_VALUE)  *FLAG_ARRAY_SIZE)
    && "The index passed in is outside of the range of valid values!");
  return ((uint32_t)(caps->flags[option >> FLAG_SHIFT_VALUE]
      & FLAG_BITLOC(option))) == (uint32_t)FLAG_BITLOC(option);
}
void setEncryptCaps(aclDevCaps_0_8 *ptr)
{
  clearFlag(ptr, capSaveSOURCE);
  clearFlag(ptr, capSaveLLVMIR);
  clearFlag(ptr, capSaveCG);
  clearFlag(ptr, capSaveSPIR);
  clearFlag(ptr, capSaveAMDIL);
  clearFlag(ptr, capSaveHSAIL);
  clearFlag(ptr, capSaveDISASM);
  clearFlag(ptr, capSaveAS);
  setFlag(ptr, capSaveEXE);
  setFlag(ptr, capEncrypted);
}
void setOptionCaps(amd::option::Options *opts, aclDevCaps_0_8 *ptr)
{
#define COND_SET_FLAG(A) \
      (((opts)->oVariables->Bin##A) ? setFlag(ptr, capSave##A) : clearFlag(ptr, capSave##A))
      COND_SET_FLAG(SOURCE);
      COND_SET_FLAG(LLVMIR);
      COND_SET_FLAG(CG);
      COND_SET_FLAG(DISASM);
      COND_SET_FLAG(AMDIL);
      COND_SET_FLAG(HSAIL);
      COND_SET_FLAG(AS);
      COND_SET_FLAG(SPIR);
      COND_SET_FLAG(EXE);

#undef COND_SET_FLAG
}

aclBIF *aclutGetBIF(aclBinary *binary)
{
  aclBIF *bif = NULL;
  if (binary->struct_size == sizeof(aclBinary_0_8)) {
    bif = reinterpret_cast<aclBinary_0_8*>(binary)->bin;
  } else if (binary->struct_size == sizeof(aclBinary_0_8_1)) {
    bif = reinterpret_cast<aclBinary_0_8_1*>(binary)->bin;
  } else {
    assert(!"Binary format not supported!");
    bif = reinterpret_cast<aclBinary*>(binary)->bin;
  }
  return bif;
}

aclOptions *aclutGetOptions(aclBinary *binary)
{
  aclOptions *opt = NULL;
  if (binary->struct_size == sizeof(aclBinary_0_8)) {
    opt = reinterpret_cast<aclBinary_0_8*>(binary)->options;
  } else if (binary->struct_size == sizeof(aclBinary_0_8_1)) {
    opt = reinterpret_cast<aclBinary_0_8_1*>(binary)->options;
  } else {
    assert(!"Binary format not supported!");
    opt = binary->options;
  }
  return opt;
}

aclBinaryOptions *aclutGetBinOpts(aclBinary *binary)
{
  aclBinaryOptions *opt = NULL;
  if (binary->struct_size == sizeof(aclBinary_0_8)) {
    opt = reinterpret_cast<aclBinaryOptions*>(
        &reinterpret_cast<aclBinary_0_8*>(binary)->binOpts);
  } else if (binary->struct_size == sizeof(aclBinary_0_8_1)) {
    opt = &reinterpret_cast<aclBinary_0_8_1*>(binary)->binOpts;
  } else {
    assert(!"Binary format not supported!");
    opt = &binary->binOpts;
  }
  return opt;
}

aclTargetInfo *aclutGetTargetInfo(aclBinary *binary)
{
  aclTargetInfo *tgt = NULL;
  if (binary->struct_size == sizeof(aclBinary_0_8)) {
    tgt = &reinterpret_cast<aclBinary_0_8*>(binary)->target;
  } else if (binary->struct_size == sizeof(aclBinary_0_8_1)) {
    tgt = &reinterpret_cast<aclBinary_0_8_1*>(binary)->target;
  } else {
    assert(!"Binary format not supported!");
    tgt = &binary->target;
  }
  return tgt;
}

aclDevCaps* aclutGetCaps(aclBinary *binary)
{
  aclDevCaps *caps = NULL;
  if (binary->struct_size == sizeof(aclBinary_0_8)) {
    caps = &reinterpret_cast<aclBinary_0_8*>(binary)->caps;
  } else if (binary->struct_size == sizeof(aclBinary_0_8_1)) {
    caps = &reinterpret_cast<aclBinary_0_8_1*>(binary)->caps;
  } else {
    assert(!"Binary format not supported!");
    caps = &binary->caps;
  }
  return caps;
}
// Helper function that returns the
// allocation function from the binary.
AllocFunc
aclutAlloc(const aclBinary *bin)
{
  size_t size = (bin ? bin->struct_size : 0);
  AllocFunc m = NULL;
  switch(size) {
    case 0:
    case sizeof(aclBinary_0_8):
      break;
    case sizeof(aclBinary_0_8_1):
      m = reinterpret_cast<const aclBinary_0_8_1*>(bin)->binOpts.alloc;
      break;
    default:
      assert(!"Found an unsupported binary!");
      m = bin->binOpts.alloc;
      break;
  }
  return (m) ? m : &::malloc;
}

// Helper function that returns the
// allocation function from the compiler.
AllocFunc
aclutAlloc(const aclCompiler *bin)
{
  size_t size = (bin ? bin->struct_size : 0);
  AllocFunc m = NULL;
  switch(size) {
    case 0:
    case sizeof(aclCompilerHandle_0_8):
      break;
    case sizeof(aclCompilerHandle_0_8_1):
      m = reinterpret_cast<const aclCompilerHandle_0_8_1*>(bin)->alloc;
      break;
    default:
      assert(!"Found an unsupported compiler!");
      m = bin->alloc;
      break;
  }
  return (m) ? m : &::malloc;
}

AllocFunc
aclutAlloc(const aclCompilerOptions *opts)
{
  size_t size = (opts ? opts->struct_size : 0);
  AllocFunc m = NULL;
  switch (size) {
    case 0:
    case sizeof(aclCompilerOptions_0_8):
      break;
    case sizeof(aclCompilerOptions_0_8_1):
      m = reinterpret_cast<const aclCompilerOptions_0_8_1*>(opts)->alloc;
      break;
    default:
      assert(!"Found an unsupported compiler options struct!");
      m = opts->alloc;
      break;
  }
  return (m) ? m : &::malloc;
}


// Helper function that returns the
// de-allocation function from the compiler.
FreeFunc
aclutFree(const aclCompiler *bin)
{
  size_t size = (bin ? bin->struct_size : 0);
  FreeFunc f = NULL;
  switch(size) {
    case 0:
    case sizeof(aclCompilerHandle_0_8):
      break;
    case sizeof(aclCompilerHandle_0_8_1):
      f = reinterpret_cast<const aclCompilerHandle_0_8_1*>(bin)->dealloc;
      break;
    default:
      assert(!"Found an unsupported compiler!");
      f = bin->dealloc;
      break;
  }
  return (f) ? f : &::free;
}

// Helper function that returns the
// de-allocation function from the binary.
FreeFunc
aclutFree(const aclBinary *bin)
{
  size_t size = (bin ? bin->struct_size : 0);
  FreeFunc f = NULL;
  switch(size) {
    case 0:
    case sizeof(aclBinary_0_8):
      break;
    case sizeof(aclBinary_0_8_1):
      f = reinterpret_cast<const aclBinary_0_8_1*>(bin)->binOpts.dealloc;
      break;
    default:
      assert(!"Found an unsupported binary!");
      f = bin->binOpts.dealloc;
      break;
  }
  return (f) ? f : &::free;
}

FreeFunc
aclutFree(const aclCompilerOptions *opts)
{
  size_t size = (opts ? opts->struct_size : 0);
  FreeFunc f = NULL;
  switch (size) {
    case 0:
    case sizeof(aclCompilerOptions_0_8):
      break;
    case sizeof(aclCompilerOptions_0_8_1):
      f = reinterpret_cast<const aclCompilerOptions_0_8_1*>(opts)->dealloc;
      break;
    default:
      assert(!"Found an unsupported compiler options struct!");
      f = opts->dealloc;
      break;
  }
  return (f) ? f : &::free;
}


void
aclutCopyBinOpts(aclBinaryOptions *dst, const aclBinaryOptions *src, bool is64)
{
  if (dst == src) return;
  aclBinaryOptions_0_8 *dst08;
  aclBinaryOptions_0_8_1 *dst081;
  const aclBinaryOptions_0_8 *src08;
  const aclBinaryOptions_0_8_1 *src081;
  dst08 = reinterpret_cast<aclBinaryOptions_0_8*>(dst);
  dst081 = reinterpret_cast<aclBinaryOptions_0_8_1*>(dst);
  src08 = reinterpret_cast<const aclBinaryOptions_0_8*>(src);
  src081 = reinterpret_cast<const aclBinaryOptions_0_8_1*>(src);
  unsigned size = (src ? src->struct_size : 0);
  switch (size) {
    case 0:
      switch (dst->struct_size) {
        case sizeof(aclBinary_0_8):
          dst08->elfclass = (is64) ? ELFCLASS64 : ELFCLASS32;
          dst08->bitness = ELFDATA2LSB;
          dst08->temp_file = "";
          dst08->kernelArgAlign = 4;
          break;
        case sizeof(aclBinary_0_8_1):
          dst081->elfclass = (is64) ? ELFCLASS64 : ELFCLASS32;
          dst081->bitness = ELFDATA2LSB;
          dst081->temp_file = "";
          dst081->kernelArgAlign = 4;
          dst081->alloc = &::malloc;
          dst081->dealloc = &::free;
          break;
        default:
          dst->elfclass = (is64) ? ELFCLASS64 : ELFCLASS32;
          dst->bitness = ELFDATA2LSB;
          dst->temp_file = "";
          dst->kernelArgAlign = 4;
          dst->alloc = &::malloc;
          dst->dealloc = &::free;
          break;
       }
      break;
    case sizeof(aclBinaryOptions_0_8):
      switch (dst->struct_size) {
        case sizeof(aclBinaryOptions_0_8):
          memcpy(dst08, src08, src08->struct_size);
          break;
        case sizeof(aclBinaryOptions_0_8_1):
          dst081->elfclass = src08->elfclass;
          dst081->bitness = src08->bitness;
          dst081->temp_file = src08->temp_file;
          dst081->kernelArgAlign = src08->kernelArgAlign;
          dst081->alloc = &::malloc;
          dst081->dealloc = &::free;
          break;
        default:
          assert(!"aclBinary format is not supported!");
          memcpy(dst, src08, src08->struct_size);
          if (!dst->alloc) dst->alloc = &::malloc;
          if (!dst->dealloc) dst->dealloc = &::free;
      }
      break;
    case sizeof(aclBinaryOptions_0_8_1):
      switch (dst->struct_size) {
        case sizeof(aclBinary_0_8):
          dst08->elfclass = src081->elfclass;
          dst08->bitness = src081->bitness;
          dst08->temp_file = src081->temp_file;
          dst08->kernelArgAlign = src081->kernelArgAlign;
          break;
        case sizeof(aclBinaryOptions_0_8_1):
          memcpy(dst081, src081, src081->struct_size);
          if (!dst->alloc) dst->alloc = &::malloc;
          if (!dst->dealloc) dst->dealloc = &::free;
          break;
        default:
          assert(!"aclBinary format is not supported!");
          memcpy(dst, src081, src081->struct_size);
          if (!dst->alloc) dst->alloc = &::malloc;
          if (!dst->dealloc) dst->dealloc = &::free;
      }
      break;
    default:
      assert(!"aclBinary format is not supported!");
      memcpy(dst, src, src->struct_size);
  }
}

acl_error
aclutInsertKernelStatistics(aclCompiler *cl, aclBinary *bin)
{
    if (!aclValidateCompiler(cl, true)) {
        return ACL_INVALID_COMPILER;
    }
    if (!aclValidateBinary(bin)) {
        return ACL_INVALID_BINARY;
    }
    size_t len = 0;
    acl_error err = ACL_SUCCESS;
    const void *isa = aclExtractSection(cl, bin, &len, aclTEXT, &err);
    if (err != ACL_SUCCESS)
        return err;
    aclTargetInfo *tgtInfo = aclutGetTargetInfo(bin);
    const char* chipName = aclGetChip(*tgtInfo);
    unsigned family = getFamilyEnum(tgtInfo);
    unsigned chip = getChipEnum(tgtInfo);
    // Non-GPU devices have family_enum set to 1 and do not qualify. Need to update.
    if (family >= FAMILY_R600 &&
	    family <= FAMILY_CZ) {
	  aclKernelStats kstats = {0};
      if (family < FAMILY_SI) {
        aclGetKstatsR800(isa, kstats, chipName);
      }
      else {
        aclGetKstatsSI(isa, kstats);
      }
	  kstats.wavefrontsize = 64; // FIXME: Hardcoded for now.
	  const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symKernelStats);
	  assert(symbol && "symbol not found");
	  std::string symName = std::string(symbol->str[PRE]) + std::string(symbol->str[POST]);
	  err = aclInsertSymbol(cl, bin, reinterpret_cast<void*>(&kstats), sizeof(kstats), aclKSTATS, symName.c_str());
    }
    return err;
}

std::string aclutGetCodegenName(const aclTargetInfo &tgtInfo)
{
  assert(tgtInfo.arch_id <= aclLast && "Unknown device id!");
  const FamilyMapping *family = familySet + tgtInfo.arch_id;
  if (!family) return "";

  assert((tgtInfo.chip_id) < family->children_size && "Unknown family id!");
  const TargetMapping *target = &family->target[tgtInfo.chip_id];
  return (target) ? target->codegen_name : "";
}

void initElfDeviceCaps(aclBinary *elf)
{
  if (aclutGetCaps(elf)->encryptCode) {
    setEncryptCaps(aclutGetCaps(elf));
    return;
  }
  if (aclutGetOptions(elf)) {
    setOptionCaps(reinterpret_cast<amd::option::Options*>(
          aclutGetOptions(elf)), aclutGetCaps(elf));
  }
}

const char *getDeviceName(const aclTargetInfo &target)
{
  if (target.chip_id) {
    return aclGetChip(target);
  } else if (target.arch_id) {
    return aclGetArchitecture(target);
  }
  return NULL;
}

/*! Function that returns the TargetMapping for
 *the specific target device.
 */
static const TargetMapping& getTargetMapping(const aclTargetInfo &target)
{
  switch(target.arch_id) {
    default:
      assert(!"Passed a device id that is invalid!");
      break;
    case aclX64:
      return X64TargetMapping[target.chip_id];
      break;
    case aclX86:
      return X86TargetMapping[target.chip_id];
      break;
    case aclHSAIL:
      return HSAILTargetMapping[target.chip_id];
      break;
    case aclHSAIL64:
      return HSAIL64TargetMapping[target.chip_id];
      break;
    case aclAMDIL:
      return AMDILTargetMapping[target.chip_id];
      break;
    case aclAMDIL64:
      return AMDIL64TargetMapping[target.chip_id];
      break;
  };
  return UnknownTarget;
}

/*! Function that returns the library type from the TargetMapping table for
 *the specific target device id.
 */
amd::LibrarySelector getLibraryType(const aclTargetInfo *target)
{
  const TargetMapping& Mapping = getTargetMapping(*target);
  return Mapping.lib;
}

/*! Function that returns family_enum from the TargetMapping table for
 *the specific target device id.
 */
unsigned getFamilyEnum(const aclTargetInfo *target)
{
  const TargetMapping& Mapping = getTargetMapping(*target);
  return Mapping.family_enum;
}

/*! Function that returns chip_enum from the TargetMapping table for
 *the specific target device id.
 */
unsigned getChipEnum(const aclTargetInfo *target)
{
  const TargetMapping& Mapping = getTargetMapping(*target);
  return Mapping.chip_enum;
}

/*! Function that returns isa type name (compute capability) from
 *the TargetMapping table for the specific target device id.
 */
const std::string &getIsaTypeName(const aclTargetInfo *target)
{
  const TargetMapping& Mapping = getTargetMapping(*target);
  switch (Mapping.family_enum) {
    default: return sgfx700;
    case FAMILY_KV:
      switch (Mapping.chip_enum) {
        default: return sgfx700;
        case KV_SPECTRE_A0:
        case KV_SPOOKY_A0:
        case KB_KALINDI_A0:
        // ???
        case ML_GODAVARI_A0: return sgfx700;
      }
    case FAMILY_CI:
      switch (Mapping.chip_enum) {
        default: return sgfx700;
        case CI_BONAIRE_M_A0:
        case CI_BONAIRE_M_A1: return sgfx700;
        case CI_HAWAII_P_A0: return sgfx701;
        case CI_TIRAN_P_A0:
        case CI_MAUI_P_A0: return sgfx700;
      }
    case FAMILY_VI:
      switch (Mapping.chip_enum) {
        default: return sgfx800;
        case VI_ICELAND_M_A0:
        case VI_TONGA_P_A0: return sgfx800;
        case VI_ELLESMERE_P_A0:
        case VI_BAFFIN_M_A0:
        case VI_FIJI_P_A0: return sgfx801;
      }
    case FAMILY_CZ:
      switch (Mapping.chip_enum) {
        default: return sgfx801;
        case CARRIZO_A0: return sgfx801;
      }
    case FAMILY_AI:
      switch (Mapping.chip_enum) {
        default: return sgfx900;
        case AI_GREENLAND_P_A0: return sgfx900;
      }
  }
}

void
appendLogToCL(aclCompiler *cl, const std::string &logStr)
{
  if (logStr.empty()) {
    return;
  }
  std::string log = logStr;
  if ('\n' != log[log.size()-1]) {
    log.append("\n");
  }
  unsigned size = cl->logSize + log.size();
  if (!size) {
    return;
  }
  char *tmpBuildLog = reinterpret_cast<char*>(aclutAlloc(cl)(size + 2));
  memset(tmpBuildLog, 0, size + 2);
  if (cl->logSize) {
    std::copy(cl->buildLog, cl->buildLog + cl->logSize, tmpBuildLog);
    std::copy(log.begin(), log.end(), tmpBuildLog + cl->logSize);
  } else {
    std::copy(log.begin(), log.end(), tmpBuildLog);
  }
  cl->logSize += (unsigned int)log.size();
  if (cl->buildLog) {
    aclutFree(cl)(cl->buildLog);
  }
  cl->buildLog = tmpBuildLog;
}

static void
setElfTarget(bifbase *elfBin, const aclTargetInfo *tgtInfo)
{
  uint16_t elf_target = 0;
  switch (tgtInfo->arch_id) {
    default:
      assert(!"creating an elf for an invalid architecture!");
    case aclX86:
      elfBin->setTarget(EM_386, aclPlatformCompLib);
      break;
    case aclX64:
      elfBin->setTarget(EM_X86_64, aclPlatformCompLib);
      break;
    case aclHSAIL:
      elfBin->setTarget(EM_HSAIL, aclPlatformCompLib);
      break;
    case aclHSAIL64:
      elfBin->setTarget(EM_HSAIL_64, aclPlatformCompLib);
      break;
    case aclAMDIL:
      elfBin->setTarget(EM_AMDIL, aclPlatformCompLib);
      break;
    case aclAMDIL64:
      elfBin->setTarget(EM_AMDIL_64, aclPlatformCompLib);
      break;
  }
}
// FIXME: this needs to be moved into the BIF classes.
static void
convertBIF30MachineTo2X(bifbase *elfBin, const aclTargetInfo *tgtInfo)
{
  uint16_t machine = 0;
  uint32_t flags = 0;
  aclPlatform pform = aclPlatformLast;
  if (elfBin == NULL) return;
  elfBin->getTarget(machine, pform);
  assert(pform == aclPlatformCompLib
      && "Platform is specified incorrectly!");
  if (isCpuTarget(*tgtInfo)) {
    assert(!"Not implemented/supported family detected!");
    pform = aclPlatformCPU;
  } else if (isAMDILTarget(*tgtInfo)) {
    const char* chip = aclGetChip(*tgtInfo);
    for (unsigned x = 0, y = sizeof(calTargetMapping)/sizeof(calTargetMapping[0]);
        x < y; ++x) {
      if (!strcmp(chip, calTargetMapping[x])) {
        machine = x;
        break;
      }
    }
    pform = aclPlatformCAL;
  } else {
    assert(!"Not implemented/supported family detected!");
  }
  elfBin->setTarget(machine, pform);
}
// FIXME: This needs to be moved into the elf classes
static void
convertBIF2XMachineTo31(bifbase *elfBin)
{
  uint16_t machine = 0;
  aclPlatform pform = aclPlatformLast;
  if (elfBin == NULL) return;
  elfBin->getTarget(machine, pform);
  assert(pform != aclPlatformCompLib
      && "Platform is specified incorrectly!");
  if (pform == aclPlatformCPU) {
      uint16_t type;
      elfBin->getType(type);
    machine = (type == ELFCLASS32 ? EM_386 : EM_X86_64);
  } else if (pform == aclPlatformCAL) {
    machine = EM_AMDIL;
  } else {
    assert(!"Unknown platform found!");
  }
  pform = aclPlatformCompLib;
  elfBin->setTarget(machine, pform);
}

static void
convertBIF2XMachineTo30(bifbase *elfBin)
{
  uint16_t machine = 0;
  aclPlatform pform = aclPlatformLast;
  if (elfBin == NULL) return;
  elfBin->getTarget(machine, pform);
  assert(pform != aclPlatformCompLib
      && "Platform is specified incorrectly!");
  if (pform == aclPlatformCPU) {
      uint16_t type;
      elfBin->getType(type);
    machine = (type == ELFCLASS32 ? EM_386 : EM_X86_64);
  } else if (pform == aclPlatformCAL) {
    machine = EM_AMDIL;
  } else {
    assert(!"Unknown platform found!");
  }
  pform = aclPlatformCompLib;
  elfBin->setTarget(machine, pform);
}

static void
setElfFlags(bifbase *elfBin, const aclTargetInfo *tgtInfo)
{
  uint32_t flags = 0;
  elfBin->getFlags(flags);
  flags &= 0xFFFF0000;
  const FamilyMapping *family = familySet + tgtInfo->arch_id;
  flags = tgtInfo->chip_id & 0xFFFF;
  elfBin->setFlags(flags);
}

static aclBinary*
cloneOclElfNoBIF(const aclBinary *src) {
  if (src == NULL) return NULL;
  if (src->struct_size == sizeof(aclBinary_0_8_1)) {
    aclBinary *dst = constructBinary(src->struct_size,
        aclutGetTargetInfo(const_cast<aclBinary*>(src)),
        aclutGetBinOpts(const_cast<aclBinary*>(src)));
    if (dst == NULL) {
      return NULL;
    }
    aclBinary_0_8_1 *dptr = reinterpret_cast<aclBinary_0_8_1*>(dst);
    const aclBinary_0_8_1 *sptr = reinterpret_cast<const aclBinary_0_8_1*>(src);
    dptr->target.struct_size = sizeof(aclTargetInfo_0_8);
    if (sptr->target.struct_size == sizeof(aclTargetInfo_0_8)) {
      memcpy(&dptr->target, &sptr->target, sptr->target.struct_size);
    } else {
      assert(!"Unsupported target info detected!");
    }

    memcpy(&dptr->caps, &sptr->caps, sptr->caps.struct_size);
    assert(sizeof(aclDevCaps_0_8) == dptr->caps.struct_size);
    amd::option::Options *Opts = reinterpret_cast<amd::option::Options*>(
            aclutAlloc(src)(sizeof(amd::option::Options)));
    Opts = new (Opts) amd::option::Options;
    amd::option::Options *sOpts = reinterpret_cast<amd::option::Options*>(
        sptr->options);
    if (sOpts) {
      parseAllOptions(sOpts->origOptionStr, *Opts);
    }
    dptr->options = reinterpret_cast<aclOptions*>(Opts);
    dptr->bin = NULL;
    return dst;
  } else if (src->struct_size == sizeof(aclBinary_0_8)) {
    aclBinary *dst = constructBinary(src->struct_size,
        &src->target,
        &src->binOpts);
    if (dst == NULL) {
      return NULL;
    }
    aclBinary_0_8 *dptr = reinterpret_cast<aclBinary_0_8*>(dst);
    const aclBinary_0_8 *sptr = reinterpret_cast<const aclBinary_0_8*>(src);
    dptr->target.struct_size = sizeof(aclTargetInfo_0_8);
    if (sptr->target.struct_size == sizeof(aclTargetInfo_0_8)) {
      memcpy(&dptr->target, &sptr->target, sptr->target.struct_size);
    } else {
      assert(!"Unsupported target info detected!");
    }

    memcpy(&dptr->caps, &sptr->caps, sptr->caps.struct_size);
    assert(sizeof(aclDevCaps_0_8) == dptr->caps.struct_size
        && "The caps struct is not version 0.7!");
    amd::option::Options *Opts = reinterpret_cast<amd::option::Options*>(
            aclutAlloc(src)(sizeof(amd::option::Options)));
    Opts = new (Opts) amd::option::Options;
    amd::option::Options *sOpts = reinterpret_cast<amd::option::Options*>(
        sptr->options);
    if (sOpts) {
      parseAllOptions(sOpts->origOptionStr, *Opts);
    }
    dptr->options = reinterpret_cast<aclOptions*>(Opts);
    dptr->bin = NULL;
    return dst;
  } else {
    assert(!"Elf version not supported!");
  }
  return NULL;
}

// Create a copy of an ELF and duplicate all sections/symbols
// All sections are copied verbatim.
aclBinary*
createELFCopy(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    bifbase* dstBin = NULL;
    switch (srcBin->getVersion()) {
      default:
        assert(!"New/unknown version detected!");
        dstBin = reinterpret_cast<bifbase*>(aclutAlloc(src)(sizeof(bifbase)));
        dstBin = new (dstBin) bifbase(srcBin->getBase());
        break;
      case aclBIFVersion20:
        dstBin = reinterpret_cast<bifbase*>(aclutAlloc(src)(sizeof(bif20)));
        dstBin = new (dstBin) bif20(srcBin->get20()); break;
      case aclBIFVersion21:
        dstBin = reinterpret_cast<bifbase*>(aclutAlloc(src)(sizeof(bif21)));
        dstBin = new (dstBin) bif21(srcBin->get21()); break;
      case aclBIFVersion30:
        dstBin = reinterpret_cast<bifbase*>(aclutAlloc(src)(sizeof(bif30)));
        dstBin = new (dstBin) bif30(srcBin->get30()); break;
      case aclBIFVersion31:
        dstBin = reinterpret_cast<bifbase*>(aclutAlloc(src)(sizeof(bif31)));
        dstBin = new (dstBin) bif31(srcBin->get31()); break;
    }
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF2.1 elf from a BIF 2.0 elf.
// All sections are copied and then if
// CAL/DLL or JITBINARY sections are found,
// the type is set to EXEC.
aclBinary*
convertBIF20ToBIF21(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get20() != NULL && "Passed in an invalid binary!");
    bif21 *dstBin = NULL;
    dstBin = reinterpret_cast<bif21*>(aclutAlloc(src)(sizeof(bif21)));
    dstBin = new (dstBin) bif21(srcBin->get20());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF3.0 elf from a BIF 2.0 elf.
aclBinary*
convertBIF20ToBIF30(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get20() != NULL && "Passed in an invalid binary!");
    bif30 *dstBin = NULL;
    dstBin = reinterpret_cast<bif30*>(aclutAlloc(src)(sizeof(bif30)));
    dstBin = new (dstBin) bif30(srcBin->get20());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
    convertBIF2XMachineTo30(dstBin);
  }
  return dst;
}

// Create a BIF3.1 elf from a BIF 2.0 elf.
aclBinary*
convertBIF20ToBIF31(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get20() != NULL && "Passed in an invalid binary!");
    bif31 *dstBin = NULL;
    dstBin = reinterpret_cast<bif31*>(aclutAlloc(src)(sizeof(bif31)));
    dstBin = new (dstBin) bif31(srcBin->get20());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
    convertBIF2XMachineTo31(dstBin);
  }
  return dst;
}

// Create a BIF2.0 elf from a BIF 2.1 elf.
// All sections except for the COMMENT section is copied
// verbatim and the section is set to NONE.
aclBinary*
convertBIF21ToBIF20(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get21() != NULL && "Passed in an invalid binary!");
    bif20 *dstBin = NULL;
    dstBin = reinterpret_cast<bif20*>(aclutAlloc(src)(sizeof(bif20)));
    dstBin = new (dstBin) bif20(srcBin->get21());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF3.0 elf from a BIF 2.1 elf.
// See BIF spec for 2.1 to 3.0 conversion
// and also include the comment section.
aclBinary*
convertBIF21ToBIF30(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get21() != NULL && "Passed in an invalid binary!");
    bif30 *dstBin = NULL;
    dstBin = reinterpret_cast<bif30*>(aclutAlloc(src)(sizeof(bif30)));
    dstBin = new (dstBin) bif30(srcBin->get21());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
    convertBIF2XMachineTo30(dstBin);
  }
  return dst;
}

// Create a BIF3.1 elf from a BIF 2.1 elf.
// See BIF spec for 3.1 to 2.1 conversion.
aclBinary*
convertBIF21ToBIF31(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get21() != NULL && "Passed in an invalid binary!");
    bif31 *dstBin = NULL;
    dstBin = reinterpret_cast<bif31*>(aclutAlloc(src)(sizeof(bif31)));
    dstBin = new (dstBin) bif31(srcBin->get21());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
    convertBIF2XMachineTo31(dstBin);
  }
  return dst;
}

// Create a BIF2.0 elf from a BIF 3.0 elf.
// See BIF spec for 3.0 to 2.0 conversion.
aclBinary*
convertBIF30ToBIF20(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get30() != NULL && "Passed in an invalid binary!");
    bif20 *dstBin = NULL;
    dstBin = reinterpret_cast<bif20*>(aclutAlloc(src)(sizeof(bif20)));
    dstBin = new (dstBin) bif20(srcBin->get30());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF2.1 elf from a BIF 3.0 elf
// See BIF spec for 3.0 to 2.1 conversion
// but also include the COMMENT section.
aclBinary*
convertBIF30ToBIF21(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get30() != NULL && "Passed in an invalid binary!");
    bif21 *dstBin = NULL;
    dstBin = reinterpret_cast<bif21*>(aclutAlloc(src)(sizeof(bif21)));
    dstBin = new (dstBin) bif21(srcBin->get30());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF3.1 elf from a BIF 3.0 elf
// See BIF spec for 3.0 to 3.1 conversion.
aclBinary*
convertBIF30ToBIF31(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get30() != NULL && "Passed in an invalid binary!");
    bif31 *dstBin = NULL;
    dstBin = reinterpret_cast<bif31*>(aclutAlloc(src)(sizeof(bif31)));
    dstBin = new (dstBin) bif31(srcBin->get30());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF2.0 elf from a BIF 3.1 elf.
// See BIF spec for 3.1 to 2.0 conversion.
aclBinary*
convertBIF31ToBIF20(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get31() != NULL && "Passed in an invalid binary!");
    bif20 *dstBin = NULL;
    dstBin = reinterpret_cast<bif20*>(aclutAlloc(src)(sizeof(bif20)));
    dstBin = new (dstBin) bif20(srcBin->get31());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF2.1 elf from a BIF 3.1 elf
// See BIF spec for 3.1 to 2.1 conversion.
aclBinary*
convertBIF31ToBIF21(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get31() != NULL && "Passed in an invalid binary!");
    bif21 *dstBin = NULL;
    dstBin = reinterpret_cast<bif21*>(aclutAlloc(src)(sizeof(bif21)));
    dstBin = new (dstBin) bif21(srcBin->get31());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

// Create a BIF3.1 elf from a BIF 3.0 elf
// See BIF spec for 3.0 to 3.1 conversion.
aclBinary*
convertBIF31ToBIF30(aclBinary *src) {
  aclBinary *dst = cloneOclElfNoBIF(src);
  if (dst != NULL) {
    bifbase *srcBin = reinterpret_cast<bifbase*>(aclutGetBIF(src));
    assert(srcBin->get31() != NULL && "Passed in an invalid binary!");
    bif30 *dstBin = NULL;
    dstBin = reinterpret_cast<bif30*>(aclutAlloc(src)(sizeof(bif30)));
    dstBin = new (dstBin) bif30(srcBin->get31());
    if (dstBin->hasError()) {
      aclBinaryFini(dst);
      return NULL;
    }
    dst->bin = reinterpret_cast<aclBIF*>(dstBin);
  }
  return dst;
}

void dump(aclBinary *bin) {
  bifbase *elfBin = reinterpret_cast<bifbase*>(bin->bin);
  elfBin->dump();
}
