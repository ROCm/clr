//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "frontend.hpp"
#include "bif/bifbase.hpp"
#include "utils/target_mappings.h"
#include "utils/options.hpp"
#include "os/os.hpp"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

bool Is64bitMachine() {return sizeof(void*) == 8;}

void
amdcl::OCLFrontend::appendCLVersionFlag(std::stringstream &ss,
                                        const amd::option::Options *opts)
{
  llvm::StringRef clStd(opts->oVariables->CLStd);

  if (clStd == "CL1.0") {
    ss << "--opencl=1.0 ";
  } else if (clStd == "CL1.1") {
    ss << "--opencl=1.1 ";
  } else if (clStd == "CL1.2") {
    ss << "--opencl=1.2 ";
  } else {
    if (clStd != "CL2.0") {
      appendLogToCL(CL(), "Warning: invalid value for -cl-std, defaulting to CL1.2");
      ss << "--opencl=1.2 ";
      return;
    }

    ss << "--opencl=2.0 ";
  }
}

///
/// @brief Function that converts elf + src combo into the correct
/// sequence of commands to call the CLC frontend.
///
/// FIXME: This needs to be modified so writing to a file is
/// not necessary!
std::string
amdcl::OCLFrontend::getFrontendCommand(aclBinary *elf,
                                       const std::string &src,
                                       std::string &logFile,
                                       std::string &clFile,
                                       bool preprocessOnly)
{
  std::stringstream systemPath;
  std::fstream f;
  amd::option::Options* Opts = (amd::option::Options*)elf->options;

  f.open(clFile.c_str(), (std::fstream::out | std::fstream::binary));
  f.write(src.data(), src.length());
  f.close();

  bool enableSpir = false;
#ifdef DEBUG
  enableSpir = getenv("AMD_OCL_ENABLE_SPIR");
#endif

  if (enableSpir)
    systemPath << "clc --spir --emit=spirbc ";
  else
    systemPath << "clc --emit=llvmbc ";

  appendCLVersionFlag(systemPath, Opts);

  if (enableSpir)
    systemPath << "--amd-options-begin " << Opts->origOptionStr << " --amd-options-end ";

  if (checkFlag(aclutGetCaps(elf), capImageSupport)) {
    systemPath << "-D__IMAGE_SUPPORT__=1 ";
  }

  if (checkFlag(aclutGetCaps(elf), capFMA)) {
    systemPath << "-DFP_FAST_FMAF=1 ";
    systemPath << "-DFP_FAST_FMA=1 ";
  }

  // F_IMAGES
  if (Options()->oVariables->ImageSupport) {
    systemPath << "-D__IMAGE_SUPPORT__=1 ";
  }

  if (Options()->oVariables->FastFMA) {
    systemPath << "-DFP_FAST_FMA=1 ";
  }

  if (Options()->oVariables->FastFMAF) {
    systemPath << "-DFP_FAST_FMAF=1 ";
  }

  systemPath << "-D__AMD__=1 ";
  uint32_t chipName = elf->target.chip_id;
  assert(chipName < familySet[elf->target.arch_id].children_size && "Cannot index past end of array!");
  switch(elf->target.arch_id) {
    default:
      assert(!"Unknown target device ID!");
    case aclX64:
      systemPath << "--march=x86-64 -D__X86_64__=1 -D__" << X64TargetMapping[chipName].chip_name << "__=1 ";
      break;
    case aclX86:
      systemPath << "--march=x86 -D__X86__=1 -D__" << X86TargetMapping[chipName].chip_name << "__=1 ";
      break;
    case aclAMDIL:
      systemPath << "-D__AMDIL__ -D__" << AMDILTargetMapping[chipName].chip_name << "__=1 ";
      break;
     case aclAMDIL64:
      systemPath << "--march=gpu-64 -D__AMDIL_64__ -D__" << AMDIL64TargetMapping[chipName].chip_name << "__=1 ";
      break;
    case aclHSAIL:
      systemPath << "--march=hsail -D__HSAIL__ -D__" << HSAILTargetMapping[chipName].chip_name << "__=1 ";
      break;
    case aclHSAIL64:
      systemPath << "--march=hsail64 -D__HSAIL__ -D__" << HSAIL64TargetMapping[chipName].chip_name << "__=1 ";
      break;
  };
  // AMDIL and non CPU HSAIL targets get the GPU define, everything
  // else gets CPU define.
  if (!isCpuTarget(elf->target)) {
    systemPath << "-D__GPU__=1 ";
  } else {
    systemPath << "-D__CPU__=1 ";
  }

  if (elf->target.arch_id == aclAMDIL
      && AMDILTargetMapping[chipName].family_enum == FAMILY_RV7XX) {
    systemPath << "-Dcl_amd_vec3=1 -Dcl_amd_printf=1 --opencl=1.0";
  }

  if (Opts) {
    systemPath << Opts->clcOptions;
  }

#ifdef WITH_TARGET_HSAIL
  if ((Is64bitMachine() && isHSAILTarget(elf->target)) ||
      (Opts->oVariables->GPU64BitIsa && (elf->target.arch_id == aclHSAIL)))
    systemPath << " --march=hsail64 ";
#endif

#ifdef DEBUG
  const char* env = getenv("AMD_EDG_OPTIONS");
  if (env)
    systemPath << env << " ";
#endif

#ifdef DEBUG
  if (!getenv("AMD_OCL_SHOW_COMPILER_OUTPUT"))
#endif
    systemPath << " --error_output \"" << logFile << "\" ";
  if(preprocessOnly) {
    std::string clppFileName = Opts->getDumpFileName(".i");
    systemPath << " -E -o \"" << clppFileName << "\"";
  }
  systemPath << " \"" << clFile << "\" ";

  LogPrintfDebug("Invoking CL to LLVM binary compilation:\n %s",
    systemPath.str().c_str());

#ifdef DEBUG
  if(getenv("AMD_OCL_SHOW_CMD_LINE"))
    std::cout << "command line: " << systemPath.str() << std::endl;
#endif

  if (Opts && Opts->isDumpFlagSet(amd::option::DUMP_CL) && !preprocessOnly) {
    dumpSource(src, Opts);
  }
  std::string clcCmd = systemPath.str();
  return clcCmd;
}
// CLC_IN_PROCESS_CHANGE
extern int openclFrontEnd(const char* cmdline, std::string*, std::string* typeInfo = NULL);

static std::string
loadFileToStr(std::string file)
{
  std::string str = "";
  std::ifstream log(file.c_str(), std::ios::in|std::ios::ate);
  if (log.is_open()) {
    size_t size = (size_t)log.tellg();
    log.seekg(0, std::ios::beg);
    std::vector<char> buffer(size+1);
    log.read(&buffer[0],size);
    log.close();
    //for safety
    buffer[size] = '\0';
    str += &buffer[0];
  }
  return str;
}

int
amdcl::OCLFrontend::compileCommand(const std::string& singleSrc)
{
  std::string tempFileName = amd::Os::getTempFileName();
  std::string logFile = tempFileName + ".log";
  std::string clFile = tempFileName + ".cl";
  std::string frontendCmd = getFrontendCommand(Elf(), singleSrc, logFile,
                                               clFile, false);
  std::string logStr;
  uint64_t start_time = 0, stop_time = 0;
  amd::option::Options* Opts = (amd::option::Options*)Elf()->options;

  if (Options()->oVariables->EnableBuildTiming) {
    start_time = amd::Os::timeNanos();
  }
  if (!checkFlag(aclutGetCaps(Elf()), capSaveSOURCE)) {
    CL()->clAPI.remSec(CL(), Elf(), aclSOURCE);
  }
  int ret = openclFrontEnd(frontendCmd.c_str(), &Source(), NULL);

  // We dump the preprocessed code by invoking clc a second time after the
  // original call, just in case somthing really bad happens in the original
  // call.
  if (Opts && Opts->isDumpFlagSet(amd::option::DUMP_I)) {
    std::string pplogFile = tempFileName + "preprocess.log";
    std::string ppFrontendCmd =
      getFrontendCommand(Elf(), singleSrc, pplogFile, clFile, true);
    (void) openclFrontEnd(ppFrontendCmd.c_str(), &Source(), NULL);
    amd::Os::unlink(pplogFile.c_str());
  }
  if (Options()->oVariables->EnableBuildTiming) {
    stop_time = amd::Os::timeNanos();
    std::stringstream tmp_ss;
    tmp_ss << "    OpenCL FE time: " << (stop_time - start_time)/1000ULL
      << "us\n";
    appendLogToCL(CL(), tmp_ss.str());
  }
  llvmbinary_ = loadBitcode(Source());
  if (!llvmbinary_) {
    ret |= 1;
  }
  if (!ret) {
    CL()->clAPI.insSec(CL(), Elf(), Source().data(), Source().size(), aclLLVMIR);
  }
  log_ += loadFileToStr(logFile);
  amd::Os::unlink(logFile.c_str());
  if (isCpuTarget(Elf()->target) && Options()->oVariables->EnableDebug) {
    Options()->sourceFileName_.assign(clFile);
  } else {
    amd::Os::unlink(clFile.c_str());
  }
  return ret;
}
