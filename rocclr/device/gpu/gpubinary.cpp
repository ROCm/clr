/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "device/gpu/gpubinary.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "utils/options.hpp"
#include "os/os.hpp"
#include <string>
#include <sstream>


namespace {

enum { NDX_KERNEL = 0, NDX_METADATA = 1, NDX_HEADER = 2, NDX_AMDIL = 3, NDX_LAST };
typedef struct {
  bool IsKernel;  // whether the entry is for kernel

  /*
     SymInfo[NDX_KERNEL]   :  SymbolInfo for kernel isa (cal image)
     SymInfo[NDX_METADATA] :  SymbolInfo for kernel metadata
     SymInfo[NDX_HEADER]   :  SymbolInfo for kernel header
     SymInfo[NDX_AMDIL]    :  SymbolInfo for kernel's amdil
   */
  amd::OclElf::SymbolInfo SymInfo[NDX_LAST];
} ElfSymbol_t;
}

namespace gpu {

bool ClBinary::loadKernels(NullProgram& program, bool* hasRecompiled) {
  const char __OpenCL_[] = "__OpenCL_";
  const char _kernel[] = "_kernel";
  const char _data[] = "_metadata";    // metadata for kernel function
  const char _fdata[] = "_fmetadata";  // metadata for non-kernel function
  const char _header[] = "_header";
  const char _amdil[] = "_amdil";

  *hasRecompiled = false;

  // TODO : jugu
  // Target should be 15 bit maximum. Should check this somewhere.
  uint32_t target = static_cast<uint32_t>(dev().calTarget());
  uint16_t elf_target;
  amd::OclElf::oclElfPlatform platform;
  if (!elfIn()->getTarget(elf_target, platform)) {
    LogError("The OCL binary image loading failed: incorrect format");
    return false;
  }
  if (platform == amd::OclElf::COMPLIB_PLATFORM) {
    // BIF 3.0
    uint32_t flag;
    aclTargetInfo tgtInfo = aclGetTargetInfo("amdil", dev().hwInfo()->targetName_, NULL);
    if (!elfIn()->getFlags(flag)) {
      LogError("The OCL binary image loading failed: incorrect format");
      return false;
    }
    if ((elf_target != EM_AMDIL) || (tgtInfo.chip_id != flag)) {
      LogError("The OCL binary image loading failed: different target");
      return false;
    }
  } else {
    if (((platform != amd::OclElf::CAL_PLATFORM) || ((uint32_t)target != elf_target))) {
      LogError("The OCL binary image loading failed: different target");
      return false;
    }
  }

  /* Using class so that dtor() can be invoked to do clean-up */
  class TempWrapper {
   public:
    /*
       functionNameMap[] maps from a function name (linkage name in the generated code)
       to ElfSymbol_t, which is defined as above.
     */
    std::unordered_map<std::string, ElfSymbol_t*> functionNameMap;

    // Keep all kernel ILs if -use-debugil is present (gpu debugging)
    std::unordered_map<std::string, std::string> kernelILs;

    ~TempWrapper() {
      for (const auto& it : functionNameMap) {
        delete[] it.second;
      }

      kernelILs.clear();
    }
  } tempObj;

  /*
      If usedebugil is true,  we will load IL from .debugil section. We will ignore
      _kernel, _amdil, _header in the binary.
   */
  bool usedebugil = program.getCompilerOptions()->oVariables->UseDebugIL;

  for (amd::Sym_Handle sym = elfIn()->nextSymbol(NULL); sym != NULL;
       sym = elfIn()->nextSymbol(sym)) {
    amd::OclElf::SymbolInfo symInfo;
    if (!elfIn()->getSymbolInfo(sym, &symInfo)) {
      LogError("LoadKernelFromElf: getSymbolInfo() fails");
      return false;
    }

    std::string elfSymName(symInfo.sym_name);
    const size_t offset = sizeof(__OpenCL_) - 1;
    if (elfSymName.compare(0, offset, __OpenCL_) != 0) {
      continue;
    }

    // Assume this elfSymName is associated with a kernel name. The following code will adjust
    // it if it isn't.
    const size_t suffixPos = elfSymName.rfind('_');
    bool isKernel = true;  // assume it is a kernel
    std::string FName = elfSymName.substr(0, suffixPos);
    FName.append("_kernel");  // make the kernel's linkage name

    ElfSymbol_t* elfsymbol = tempObj.functionNameMap[FName];
    amd::OclElf::SymbolInfo* sinfo = (elfsymbol != NULL) ? &(elfsymbol->SymInfo[0]) : NULL;

    // Add info for this elf symbol into tempobj's functionNameMap[]
    int index = -1;
    if (!usedebugil && (elfSymName.compare(suffixPos, sizeof(_kernel) - 1, _kernel) == 0)) {
      index = NDX_KERNEL;
      assert(((sinfo == NULL) || (sinfo[index].size == 0)) &&
             "More than one kernel symbol for the same kernel");
    } else if (!usedebugil && (elfSymName.compare(suffixPos, sizeof(_header) - 1, _header) == 0)) {
      index = NDX_HEADER;
      assert(((sinfo == NULL) || (sinfo[index].size == 0)) &&
             "More than one header symbol for a kernel");
    } else if (!usedebugil && (elfSymName.compare(suffixPos, sizeof(_amdil) - 1, _amdil) == 0)) {
      index = NDX_AMDIL;
      assert(((sinfo == NULL) || (sinfo[index].size == 0)) &&
             "More than one amdil symbol for a kernel");
    } else if (elfSymName.compare(suffixPos, sizeof(_data) - 1, _data) == 0) {
      index = NDX_METADATA;
      assert(((sinfo == NULL) || (sinfo[index].size == 0)) &&
             "More than one metadata symbol for the same kernel");
    } else if (elfSymName.compare(suffixPos, sizeof(_fdata) - 1, _fdata) == 0) {
      index = NDX_METADATA;
      isKernel = false;

      FName = elfSymName.substr(offset, suffixPos - offset);

      elfsymbol = tempObj.functionNameMap[FName];
      sinfo = (elfsymbol != NULL) ? &(elfsymbol->SymInfo[0]) : NULL;

      assert(((sinfo == NULL) || (sinfo[index].size == 0)) &&
             "More than one metadata symbol for a non-kernel function");
    }

    if (index >= 0) {
      if (elfsymbol == NULL) {
        elfsymbol = new ElfSymbol_t();
        sinfo = &(elfsymbol->SymInfo[0]);
        ::memset(sinfo, 0, NDX_LAST * sizeof(amd::OclElf::SymbolInfo));
        tempObj.functionNameMap[FName] = elfsymbol;

        elfsymbol->IsKernel = isKernel;
      }
      sinfo[index] = symInfo;
    }
  }

  std::string programil;
  if (usedebugil) {
    char* section;
    size_t sz;

    if (elfIn_->getSection(amd::OclElf::ILDEBUG, &section, &sz)) {
      // Get debugIL
      programil.append(section, sz);
    } else {
      LogError("LoadKernelFromElf(): reading .debugil failed");
      return false;
    }

    // Append all function metadata to debugIL
    for (const auto& it : tempObj.functionNameMap) {
      ElfSymbol_t* elfsymbol = it.second;
      if (elfsymbol == NULL) {
        // Not valid, skip
        continue;
      }
      if ((elfsymbol->SymInfo[NDX_METADATA].address != 0) &&
          (elfsymbol->SymInfo[NDX_METADATA].size > 0)) {
        std::string mdString = std::string(elfsymbol->SymInfo[NDX_METADATA].address,
                                           elfsymbol->SymInfo[NDX_METADATA].size);
        assert((mdString.find_first_of('\0') == std::string::npos) &&
               "Metadata string has NULL inside !");
        programil.append(mdString);
      }
    }

    const char* ilKernelName = program.getCompilerOptions()->oVariables->JustKernel;
    if (!program.getAllKernelILs(tempObj.kernelILs, programil, ilKernelName)) {
      LogError("LoadKernelFromElf(): MDParser failed generating kernel ILs");
      return false;
    }

    // Now, patch the IL from debugIL into functionNameMap[]
    for (const auto& it : tempObj.kernelILs) {
      const std::string& kn = it.first;
      const std::string& ilstr = it.second;

      ElfSymbol_t* elfsymbol = tempObj.functionNameMap[kn];
      if (elfsymbol == NULL) {
        elfsymbol = new ElfSymbol_t();
        ::memset(elfsymbol->SymInfo, 0, NDX_LAST * sizeof(amd::OclElf::SymbolInfo));
        tempObj.functionNameMap[kn] = elfsymbol;
      }
      amd::OclElf::SymbolInfo* sinfo = &(elfsymbol->SymInfo[0]);

      elfsymbol->IsKernel = true;
      sinfo[NDX_AMDIL].address = const_cast<char*>(ilstr.data());
      sinfo[NDX_AMDIL].size = ilstr.size();
      // All the other fields in SymInfo is unused
    }
  }

  bool recompiled = false;
  bool hasKernels = false;
  for (const auto& it : tempObj.functionNameMap) {
    ElfSymbol_t* elfsymbol = it.second;
    if (elfsymbol == NULL) {
      // Not valid, skip
      continue;
    } else if (!elfsymbol->IsKernel) {
      // Not a kernel. Add its metadata to the OCL binary in case recompilation happens
      // and the new binary is needed.
      if (saveAMDIL() && (elfsymbol->SymInfo[NDX_METADATA].size > 0)) {
        std::string fmetadata = "__OpenCL_";
        fmetadata.append(it.first);
        fmetadata.append("_fmetadata");

        if (!elfOut()->addSymbol(amd::OclElf::RODATA, fmetadata.c_str(),
                                 elfsymbol->SymInfo[NDX_METADATA].address,
                                 elfsymbol->SymInfo[NDX_METADATA].size)) {
          LogError("AddSymbol() failed to add fmetadata");
          return false;
        }
      }
      continue;
    }
    amd::OclElf::SymbolInfo* sinfo = &(elfsymbol->SymInfo[0]);
    std::string FName = it.first;

    // For this kernel, get the demangled kernel name, which is used to identify each kernel.
    const size_t name_sz = FName.size() - (sizeof(_kernel) - 1) - (sizeof(__OpenCL_) - 1);
    std::string demangledKName = FName.substr(sizeof(__OpenCL_) - 1, name_sz);

    // Check if the current entry is valid
    if (((sinfo[NDX_HEADER].size <= 0) || (sinfo[NDX_KERNEL].size <= 0)) &&
        (sinfo[NDX_AMDIL].size <= 0)) {
      std::string tlog =
          "Warning: both IL and CAL Image are not available for kernel " + demangledKName;
      LogWarning(tlog.c_str());
      continue;
    }
    hasKernels = true;

    Kernel::InitData initData = {0};
    std::string ilSource(sinfo[NDX_AMDIL].address, sinfo[NDX_AMDIL].size);
    std::string metadata(sinfo[NDX_METADATA].address, sinfo[NDX_METADATA].size);
    if ((sinfo[NDX_HEADER].size <= 0) || (sinfo[NDX_KERNEL].size <= 0)) {
      // IL recompilation
      // TODO:  global data recompilation as well.
      // 1) parse IL; 2) parse metadata to set up kernel header
      size_t pos;
      if (!program.findAllILFuncs((programil.size() ? programil : ilSource), pos)) {
        program.freeAllILFuncs();
        return false;
      }

      bool isFailed = false;
      for (uint32_t i = 0; i < program.funcs_.size(); ++i) {
        ILFunc* func = program.funcs_[i];
        ElfSymbol_t* sym = tempObj.functionNameMap[func->name_];
        if (sym == NULL) {
          // No metadata for this function.
          continue;
        }

        assert((func->metadata_.end_ == 0) && "ILFunc init failed");
        amd::OclElf::SymbolInfo* si = &(sym->SymInfo[0]);
        if (si[NDX_METADATA].size > 0) {
          std::string meta(si[NDX_METADATA].address, si[NDX_METADATA].size);
          if (!program.parseFuncMetadata(meta, 0, std::string::npos)) {
            isFailed = true;
            break;
          }
          if (func->metadata_.end_ != std::string::npos) {
            assert(false && "ILFunc name and index does not match");
            isFailed = true;
            break;
          }

          // Accumulate all emulated local, region and private sizes,
          // necessary for the kernel execution
          initData.localSize_ += func->localSize_;
          initData.privateSize_ += func->privateSize_;

          // Accumulate all HW local, region and private sizes,
          // necessary for the kernel execution
          initData.hwLocalSize_ += func->hwLocalSize_;
          initData.hwPrivateSize_ += func->hwPrivateSize_;
          initData.flags_ |= func->flags_;
        }
      }

      program.freeAllILFuncs();
      if (isFailed) {
        return false;
      }
    } else {
      KernelHeaderSymbol kHeader = {0};
      ::memcpy(&kHeader, sinfo[NDX_HEADER].address, (sizeof(kHeader) < sinfo[NDX_HEADER].size)
                   ? sizeof(kHeader)
                   : sinfo[NDX_HEADER].size);

      if (kHeader.version_ > VERSION_CURRENT) {
        LogError("LoadKernelFromElf: cannot handle the newer version of the binary");
        return false;
      }

      // VERSION_0
      initData.localSize_ = kHeader.localSize_;
      initData.hwLocalSize_ = kHeader.hwLocalSize_;
      initData.privateSize_ = kHeader.privateSize_;
      initData.hwPrivateSize_ = kHeader.hwPrivateSize_;
      initData.flags_ = kHeader.flags_;
    }

    bool created;
    NullKernel* gpuKernel =
        program.createKernel(demangledKName, &initData, ilSource, metadata, &created,
                             sinfo[NDX_KERNEL].address, sinfo[NDX_KERNEL].size);
    if (!created) {
      std::string tlog =
          "Error: Creating kernel during loading OCL binary " + demangledKName + " failed!";
      LogError(tlog.c_str());
      return false;
    }

    recompiled = recompiled || (sinfo[NDX_KERNEL].size == 0);

    // Add the current kernel to the OCL binary in case recompilation happens and
    // the new binary is needed.
    if (!storeKernel(demangledKName, gpuKernel, &initData, metadata, ilSource)) {
      return false;
    }
  }

  *hasRecompiled = recompiled;
  return hasKernels;
}

bool ClBinary::storeKernel(const std::string& name, const NullKernel* nullKernel,
                           Kernel::InitData* initData, const std::string& metadata,
                           const std::string& ilSource) {
  if (!saveISA() && !saveAMDIL()) {
    return true;
  }

  // should we save kernel metadata only under saveAMDIL()?
  bool kernelMetaStored = false;

  if (saveAMDIL() && (ilSource.size() > 0)) {
    // Save IL (this is the per-kernel IL)
    std::string ilName = "__OpenCL_" + name + "_amdil";
    if (!elfOut()->addSymbol(amd::OclElf::ILTEXT, ilName.c_str(), ilSource.data(),
                             ilSource.size())) {
      LogError("AddElfSymbol failed");
      return false;
    }

    std::string metaName = "__OpenCL_" + name + "_metadata";
    // Save metadata symbols in .rodata
    if (!elfOut()->addSymbol(amd::OclElf::RODATA, metaName.c_str(), metadata.data(),
                             metadata.size())) {
      LogError("AddElfSymbol failed");
      return false;
    }
    kernelMetaStored = true;
  }

  if (!saveISA()) {
    return true;
  }

  size_t binarySize = (nullKernel != NULL) ? nullKernel->getCalBinarySize() : 0;
  if (binarySize != 0) {
    if (!kernelMetaStored) {
      std::string metaName = "__OpenCL_" + name + "_metadata";
      // Save metadata symbols in .rodata
      if (!elfOut()->addSymbol(amd::OclElf::RODATA, metaName.c_str(), metadata.data(),
                               metadata.size())) {
        LogError("AddSymbol failed");
        return false;
      }
    }
    // Save kernel symbol that is associated with GPU ISA
    std::string kernelName = "__OpenCL_" + name + "_kernel";
    uint8_t* isacode = new uint8_t[binarySize];
    if (!nullKernel->getCalBinary(reinterpret_cast<void*>(isacode), binarySize)) {
      LogError("Failed to read GPU kernel isa");
      delete[] isacode;
      return false;
    }
    if (!elfOut()->addSymbol(amd::OclElf::CAL, kernelName.c_str(), isacode, binarySize)) {
      LogError("AddElfSymbol failed");
      return false;
    }
    delete[] isacode;

    // Save kernel header information into a pseudo symbol
    //    __OpenCL_<kernelName>_header
    // for example, given a kernel foo, this pseudo symbol
    // would be  __OpenCL_foo_header
    std::string headerName = "__OpenCL_" + name + "_header";
    KernelHeaderSymbol kHeader;
    // VERSION_0
    kHeader.privateSize_ = initData->privateSize_;
    kHeader.localSize_ = initData->localSize_;
    kHeader.regionSize_ = 0;
    kHeader.hwPrivateSize_ = initData->hwPrivateSize_;
    kHeader.hwLocalSize_ = initData->hwLocalSize_;
    kHeader.hwRegionSize_ = 0;
    kHeader.flags_ = initData->flags_;

    // VERSION_1
    kHeader.version_ = VERSION_CURRENT;

    if (!elfOut()->addSymbol(amd::OclElf::RODATA, headerName.c_str(), &kHeader, sizeof(kHeader))) {
      LogError("AddElfSymbol failed");
      return false;
    }
  }
  return true;
}

bool ClBinary::loadGlobalData(Program& program) {
  const char __OpenCL_[] = "__OpenCL_";
  const char _global[] = "_global";

  for (amd::Sym_Handle sym = elfIn()->nextSymbol(NULL); sym != NULL;
       sym = elfIn()->nextSymbol(sym)) {
    amd::OclElf::SymbolInfo symInfo;
    if (!elfIn()->getSymbolInfo(sym, &symInfo)) {
      LogError("LoadGlobalDataFromElf: getSymbolInfo() fails");
      return false;
    }

    std::string globalName(symInfo.sym_name);
    const size_t offset = sizeof(__OpenCL_) - 1;
    if (globalName.compare(0, offset, __OpenCL_) != 0) {
      continue;
    }
    const size_t suffixPos = globalName.rfind('_');
    if (globalName.compare(suffixPos, sizeof(_global) - 1, _global) != 0) {
      continue;
    }

    // Get index for this global
    std::string indexString = globalName.substr(offset, suffixPos - offset);
    uint index = ::atoi(indexString.c_str());

    if (!program.allocGlobalData(symInfo.address, symInfo.size, index)) {
      LogError("Couldn't load global data");
      return false;
    }
  }

  return true;
}

bool ClBinary::storeGlobalData(const void* globalData, size_t dataSize, uint index) {
  // For each global, use "__OpenCL_<globalname>" as its name
  // Since there is no name in amdil, just use "__OpenCL_<index>_global" for now.
  std::stringstream glbName;
  glbName << "__OpenCL_" << index << "_global";

  if (!elfOut()->addSymbol(amd::OclElf::RODATA, glbName.str().c_str(), globalData, dataSize)) {
    LogError("addSymbol() failed");
    return false;
  }
  return true;
}

bool ClBinary::clearElfOut() {
  // Recreate libelf elf object
  if (!elfOut()->Clear()) {
    return false;
  }

  // Need to re-setup target
  return setElfTarget();
}

}  // namespace gpu
