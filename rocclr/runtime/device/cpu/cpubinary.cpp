//
// Copyright 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpubinary.hpp"
#include "device/cpu/cpudevice.hpp"
#include "device/cpu/cpuprogram.hpp"
#include "utils/versions.hpp"
#include "os/os.hpp"

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace cpu {

ClBinary::FeatureCheckResult
ClBinary::checkFeatures()
{
    /* Validate that all cpu features of loaded binary target (i.e. elf_target) exists in current target.
     * If some of elf_target features doesn't exist in current target we fail the build since we assume that elf LLVM-IR and binary are
     * target specific and can't be recompiled to current target*/
    uint16_t target = (uint16_t)dev().settings().cpuFeatures_;
    uint16_t elf_target;
    amd::OclElf::oclElfPlatform platform;
    if (!elfIn()->getTarget(elf_target, platform)){
       LogError("Loading OCL CPU binary: incorrect format");
       return ERROR;
    }
    uint64_t chip_options=0x0;
    if (platform == amd::OclElf::COMPLIB_PLATFORM) {
        // BIF 3.0
        uint32_t flag;
        if (!elfIn()->getFlags(flag)) {
            LogError("Loading OCL CPU binary: incorrect format");
            return ERROR;
        }
        aclTargetInfo tgtInfo = aclGetTargetInfoFromChipID(LP64_SWITCH("x86", "x86-64"), flag, NULL);
        chip_options = aclGetChipOptions(tgtInfo) ;
        if (((target & chip_options) != chip_options) ||
            ((elf_target == EM_386) && (strcmp(LP64_SWITCH("x86", "x86-64"), "x86") != 0)) ||
            ((elf_target == EM_X86_64) && (strcmp(LP64_SWITCH("x86", "x86-64"), "x86-64") != 0))){
            LogError("Loading OCL CPU binary: different target");
            return ERROR;
        }
    }
    else {
        // BIF 2.0
        if ((platform != amd::OclElf::CPU_PLATFORM) ||
            ((target & elf_target) != elf_target)) {
            LogError("Loading OCL CPU binary: different target");
            return ERROR;
        }
    }
    char* section;
    size_t sz;

    /* If current target has more cpu features than the one for which the binary was (notice it must have all features as in elf_target 
     * due to previous check), we can benefit from recompiling the LLVM-IR if exists in binary (if there are errors, ignore them !).*/
    if (((platform == amd::OclElf::CPU_PLATFORM) && 
         ((target ^ elf_target) != 0)) ||
        ((platform == amd::OclElf::COMPLIB_PLATFORM) && 
         ((target ^ chip_options) != 0))) {
        if (elfIn_->getSection(amd::OclElf::LLVMIR, &section, &sz)) {
            if ((section != NULL) && (sz > 0)) {
                // hasDLL being false to force recompiling
                RECOMPILE;
            }
        }
    }
    return OK;
}

bool
ClBinary::loadX86(Program& program, std::string& dllName, bool& hasDLL)
{
    hasDLL = false;

    std::string tempName = amd::Os::getTempFileName();

    dllName = tempName
        + "." WINDOWS_SWITCH("dll",MACOS_SWITCH("dyld","so"));

    switch (checkFeatures()) {
    case ERROR:
      return false;
    case RECOMPILE:
      return true;
    case OK:
      // Fallthrough
      break;
    }

    char* section;
    size_t sz;

    if (!elfIn_->getSection(amd::OclElf::DLL, &section, &sz)) {
        LogError("Loading OCL CPU binary: error occured!");
        return false;
    }

    if ((section == NULL) || (sz == 0)) {
        // hasDLL being false to force recompiling
        return true;
    }

    std::fstream f;
    f.open(dllName.c_str(), (std::fstream::out | std::fstream::binary));

    if (!f.is_open()) {
#ifdef _WIN32
        amd::Os::unlink(tempName.c_str());
#endif // _WIN32
        LogError("Loading OCL CPU binary: cannot open a file!");
        return false;
    }
    f.write(section, sz);
    f.close();

    hasDLL = true;
    return true;
}

bool
ClBinary::storeX86(Program& program, std::string& dllName)
{
    std::fstream f;
    f.open(dllName.c_str(), (std::fstream::in | std::fstream::binary));
    if (!f.is_open()) {
        return false;
    }

    f.seekg(0, std::fstream::end);
    size_t x86CodeSize = f.tellg();
    f.seekg(0, std::fstream::beg);

    if (saveISA()) {
        char* x86Code = new char[x86CodeSize];
        f.read(x86Code, x86CodeSize);
        elfOut_->addSection(amd::OclElf::DLL, x86Code, x86CodeSize);
        delete [] x86Code;
    }
    f.close();
    return true;
}

bool
ClBinary::loadX86JIT(Program& program, bool& hasJITBinary)
{
  hasJITBinary = false;

    switch (checkFeatures()) {
    case ERROR:
      return false;
    case RECOMPILE:
      return true;
    case OK:
      // Fallthrough
      break;
    }

    char* section;
    size_t sz;

    if (!elfIn_->getSection(amd::OclElf::JITBINARY, &section, &sz)) {
        LogError("Loading OCL CPU JIT binary: error occured!");
        return false;
    }

    if ((section == NULL) || (sz == 0)) {
        // force recompiling
        return true;
    }
    acl_error err = ACL_SUCCESS;
    program.setJITBinary(aclJITObjectImageCopy(program.compiler(), section, sz, &err));
    if (err != ACL_SUCCESS) {
        LogWarning("aclJITObjectImageCopy failed");
        return false;
    }
    hasJITBinary = true;
    return true;
}

void checkDifference(const char* buf1, const char* buf2, size_t size) {
  for(size_t i = 0; i < size; ++i) {
    if(buf1[i] != buf2[i]) {
      printf("Index %d different",(int)i);
      return;
    }
  }
}
bool
ClBinary::storeX86JIT(Program& program)
{
  if (saveISA()) {
    acl_error err = ACL_SUCCESS;
    aclJITObjectImage objectImage = program.getJITBinary();
    size_t x86CodeSize = aclJITObjectImageSize(program.compiler(), objectImage, &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclJITObjectImageSize failed");
        return false;
    }
    const char* x86CodePtr = aclJITObjectImageData(program.compiler(), objectImage, &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclJITObjectImageData failed");
        return false;
    }
    elfOut_->addSection(amd::OclElf::JITBINARY, x86CodePtr, x86CodeSize);
  }
  return true;
}

bool
ClBinary::storeX86Asm(const char* buffer, size_t size)
{
  if (saveAS()) {
    elfOut_->addSection(amd::OclElf::ASTEXT, buffer, size);
  }
  return true;
}

} // namespace cpu
