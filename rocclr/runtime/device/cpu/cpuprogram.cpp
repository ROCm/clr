
//
// Copyright 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpuprogram.hpp"
#include "device/cpu/cpudevice.hpp"
#include "device/cpu/cpukernel.hpp"
#include "platform/program.hpp"
#include "utils/options.hpp"
#include "os/os.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
# include <windows.h>
#endif

// amdrt.o
#if defined(WITH_ONLINE_COMPILER) && !defined(_LP64) && !defined(ATI_ARCH_ARM)
#include "amdrt.inc"
#endif

//CLC_IN_PROCESS_CHANGE
extern int openclFrontEnd(const char* cmdline, std::string*, std::string* typInfo = NULL);

namespace cpu {

static inline bool
isScalar(clk_value_type_t type)
{
    switch (type) {
    case T_CHAR: case T_SHORT: case T_INT:
    case T_LONG: case T_FLOAT: case T_DOUBLE:
    case T_POINTER:
        return true;
    default:
        return false;
    }
}


static cl_kernel_arg_address_qualifier
getParamAddressQualifier(const clk_parameter_descriptor_t* desc)
{
    switch (desc->space) {
    case A_LOCAL:
        return CL_KERNEL_ARG_ADDRESS_LOCAL;
        break;
    case A_CONSTANT:
        return CL_KERNEL_ARG_ADDRESS_CONSTANT;
        break;
    case A_GLOBAL:
        return  CL_KERNEL_ARG_ADDRESS_GLOBAL;
        break;
    default:
        return CL_KERNEL_ARG_ADDRESS_PRIVATE;
        break;
    }
}

static cl_kernel_arg_type_qualifier
getParamTypeQualifier(const clk_parameter_descriptor_t* desc)
{
    cl_kernel_arg_type_qualifier typeQualifier = CL_KERNEL_ARG_TYPE_NONE;

    if (desc->space == A_CONSTANT) {
        typeQualifier |= CL_KERNEL_ARG_TYPE_CONST;
    }

    if ((desc->qualifier & Q_CONST) != 0) {
        typeQualifier |= CL_KERNEL_ARG_TYPE_CONST;
    }
    if ((desc->qualifier & Q_RESTRICT) != 0) {
        typeQualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
    }
    if ((desc->qualifier & Q_VOLATILE) != 0) {
        typeQualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
    }

    if ((desc->qualifier & Q_PIPE) != 0) {
        typeQualifier = CL_KERNEL_ARG_TYPE_PIPE;
    }

    return typeQualifier;
}

static cl_kernel_arg_access_qualifier
getParamAccessQualifier(const clk_parameter_descriptor_t* desc)
{
    uint access = desc->qualifier & (Q_READ | Q_WRITE);
    switch (access) {
    case Q_READ:
        return CL_KERNEL_ARG_ACCESS_READ_ONLY;
        break;
    case Q_WRITE:
        return CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
        break;
    case (Q_READ | Q_WRITE):
        return CL_KERNEL_ARG_ACCESS_READ_WRITE;
        break;
    default:
        return CL_KERNEL_ARG_ACCESS_NONE;
        break;
    }
}

static size_t
getScalarParamSize(bool cpuLayer, const clk_value_type_t type,
                   cl_kernel_arg_address_qualifier qualifier)
{
    size_t size = 0;

    if (qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL) {
      return cpuLayer ? sizeof(void*) : 0;
    }

    switch (type) {
    case T_CHAR:
        size = 1;
        break;
    case T_SHORT:  case T_CHAR2:
        size = 2;
        break;
    case T_FLOAT:  case T_INT:   case T_CHAR4:
    case T_SHORT2: case T_CHAR3:
        size = 4;
        break;
    case T_SAMPLER:
        size  = cpuLayer ? sizeof(uint32_t) : sizeof(cl_sampler);
        break;
    case T_LONG:   case T_DOUBLE: case T_CHAR8:
    case T_SHORT4: case T_INT2:   case T_FLOAT2:
    case T_SHORT3:
        size = 8;
        break;
    case T_INT3:   case T_FLOAT3:
    case T_CHAR16: case T_SHORT8: case T_INT4:
    case T_FLOAT4: case T_LONG2:  case T_DOUBLE2:
        size = 16;
        break;
    case T_LONG3:  case T_DOUBLE3:
    case T_SHORT16: case T_INT8:  case T_FLOAT8:
    case T_LONG4:   case T_DOUBLE4:
        size = 32;
        break;
    case T_INT16: case T_FLOAT16: case T_LONG8:
    case T_DOUBLE8:
        size = 64;
        break;
    case T_LONG16: case T_DOUBLE16:
        size = 128;
        break;
    case T_POINTER: case T_VOID:
        size = sizeof(void*);
        break;
    default:
        ShouldNotReachHere();
        break;
    }
    return size;
}

static size_t
getParamSizeImpl(bool cpuLayer, const clk_parameter_descriptor_t* desc,
                 unsigned index,  cl_kernel_arg_address_qualifier qualifier,
                 size_t* alignment, unsigned* index_out)
{
    size_t size = 0;
    if(desc[index].type == T_STRUCT || desc[index].type == T_PAD) {
        size_t maxAlignment = 0;
        size_t structSize = 0;
        size_t structAlignment = 0;
        index++;
        while(desc[index].type != T_VOID) {
            size_t elementAlignment = 0;
            size_t elementSize =
              getParamSizeImpl(cpuLayer, desc, index, qualifier,
                               &elementAlignment, index_out);
            #if defined(_WIN32)
              maxAlignment = std::max(maxAlignment, elementAlignment);
            #else
              // In Linux, the alignment of long field is 4 for GCC,
              // but it is 8 on LLVM side
              if (desc[index].type == T_LONG)
                structAlignment = cpuLayer? LP64_SWITCH(4, 8) : 8;
              else
                structAlignment = std::max(maxAlignment, elementAlignment);
              maxAlignment = std::max(maxAlignment, structAlignment);
            #endif
            index = *index_out;
            structSize =
              amd::alignUp(structSize,
                           std::min(elementAlignment, size_t(16))) +
              elementSize;
        }
        *index_out = index + 1;
        *alignment = maxAlignment;
        size = amd::alignUp(structSize, std::min(maxAlignment, size_t(16)));
    } else {
      size = getScalarParamSize(cpuLayer, desc[index].type, qualifier);
      if (desc[index].type == T_DOUBLE) {
          #if defined(_WIN32)
          *alignment = 8;
          #else
          *alignment = LP64_SWITCH(4, 8);
          #endif
      } else if (desc[index].type == T_LONG) {
          *alignment = 8;
      } else {
          *alignment = size;
      }
      *index_out = index + 1;
    }
    return size;
}

size_t
getParamSize(bool cpuLayer, const clk_parameter_descriptor_t* desc,
             cl_kernel_arg_address_qualifier qualifier,
             size_t* alignment)
{
   unsigned index_out = 0;
   return getParamSizeImpl(cpuLayer, desc, 0, qualifier, alignment,
                          &index_out);
}


static unsigned
getNumTypeDescs(const clk_parameter_descriptor_t* desc)
{
  int numStruct = 0;
  unsigned i;
    for(i = 0; desc[i].type != T_VOID || numStruct > 0; ++i) {
        if (desc[i].type == T_STRUCT || desc[i].type == T_PAD)
            numStruct++;
        if (desc[i].type == T_VOID)
            numStruct--;
    }
    return i + 1;
}

static clk_value_type_t
getFirstScalarType(const clk_parameter_descriptor_t* desc)
{
  int i = 0;
  while(desc[i].type == T_STRUCT)
    i++;

  return desc[i].type;
}

static const clk_value_type_t
getParamType(const clk_parameter_descriptor_t* desc,
             const clk_parameter_descriptor_t** desc_out,
             const char** type_name)
{
    unsigned numDescs = getNumTypeDescs(desc);
    *desc_out = desc + numDescs;
    *type_name = desc[numDescs-1].name;
    // Use old behaviour and return first scalar type in case of a struct.
    return getFirstScalarType(desc);

}

static amd::KernelParameterDescriptor
getParam(bool cpuLayer, const clk_parameter_descriptor_t* desc,
         size_t offset_in, const clk_parameter_descriptor_t ** desc_out)
{
    size_t alignment;

    amd::KernelParameterDescriptor param;
    param.name_ = desc->name;
    param.type_ = getParamType(desc, desc_out, &(param.typeName_));
    param.addressQualifier_ = getParamAddressQualifier(desc);
    param.typeQualifier_ = getParamTypeQualifier(desc);
    param.accessQualifier_ = getParamAccessQualifier(desc);
    param.size_ = getParamSize(cpuLayer, desc, param.addressQualifier_,
                               &alignment);
    if(param.size_ == 0) {
        param.offset_ = amd::alignUp(offset_in,
                                     std::min(sizeof(cl_mem), size_t(16)));
    } else {
      param.offset_ = amd::alignUp(offset_in,
                                   std::min(alignment, size_t(16)));
    }
    return param;
}

static bool
setKernelInfoCallback(std::string symbol, const void* value, void* data)
{
    cpu::Program* program = reinterpret_cast<cpu::Program*>(data);
    device::Program::kernels_t& kernels = program->kernels();
    const char __OpenCL_[] = "__OpenCL_";
    const char _kernel[] = "_stub";
    const char _data[] = "_metadata";
    const char _nature[] = "_nature";

    const size_t offset = sizeof(__OpenCL_) - 1;
    if (symbol.compare(0, offset, __OpenCL_) != 0) {
        return false;
    }

    size_t suffixPos = symbol.rfind('_');
    if (suffixPos == std::string::npos) {
        return false;
    }

    std::string name = symbol.substr(offset, suffixPos - offset);
    cpu::Kernel* kernel = reinterpret_cast<cpu::Kernel*>(kernels[name]);
    if (NULL == kernel) {
        kernel = new Kernel(name);
        kernels[name] = kernel;
    }

    if (symbol.compare(suffixPos, sizeof(_kernel) - 1, _kernel) == 0) {
        kernel->setEntryPoint(value);
        return true;
    }
    else if (symbol.compare(suffixPos, sizeof(_data) - 1, _data) == 0) {
        device::Kernel::parameters_t params;

        size_t* recordPtr = (size_t*) value;
        size_t* recordEnd = recordPtr + (*recordPtr)/sizeof(size_t);
        ++recordPtr; // skip struct_length

        kernel->setLocalMemSize(*recordPtr++);
        kernel->setPreferredSizeMultiple(1);

        kernel->setUniformWorkGroupSize(program->getCompilerOptions()
          ->oVariables->UniformWorkGroupSize);

        kernel->setReqdWorkGroupSize(recordPtr[0], recordPtr[1], recordPtr[2]);
        recordPtr += 3;

        kernel->setWorkGroupSizeHint(recordPtr[0], recordPtr[1], recordPtr[2]);
        recordPtr += 3;

        const clk_parameter_descriptor_t* desc =
            reinterpret_cast<const clk_parameter_descriptor_t*>(recordPtr);

        size_t offset = 0;
        while (desc->type != T_VOID) {
          const clk_parameter_descriptor_t* next_desc = NULL;
          amd::KernelParameterDescriptor param = getParam(false, desc, offset,
                                                          &next_desc);

          size_t cpuSize, cpuAlignment;
          cpuSize =
            getParamSize(true, desc, param.addressQualifier_, &cpuAlignment);
          kernel->addArg(cpuSize, cpuAlignment);

          //Init for HCtoDCmap
          unsigned int init_offset = 0;
          unsigned int align = 0;
          int inStruct = 0;
          int end_index = 0;
          HCtoDCmap *map_p = new HCtoDCmap(desc, align, 0, init_offset);
          map_p->dc_size = map_p->compute_map(desc, map_p->hc_alignment, map_p->dc_alignment, init_offset, inStruct, end_index);
          map_p->align_map(map_p->hc_alignment, map_p->dc_alignment, map_p->hc_size, map_p->dc_size, inStruct);
          if (CPU_USE_ALIGNMENT_MAP == 0) {
             kernel->addHCtoDCmap(map_p);
             if (map_p->internal_field_map != NULL) {
                  kernel->addInternalMap(map_p->internal_field_map);
              }
          }
          else {
              delete(map_p);
          }
          //End of HCtoDCmap

          desc = next_desc;
          params.push_back(param);
          size_t size = param.size_ == 0 ? sizeof(cl_mem) : param.size_;
#if defined(USE_NATIVE_ABI)
          size  = amd::alignUp(size, sizeof(size_t));
#endif // USE_NATIVE_ABI
            offset = param.offset_ + size;
        }

        // retrieve vector type hint metadata
        const clk_parameter_descriptor_t* vth_desc = NULL;
        getParam(false, desc, offset, &vth_desc);
        const size_t* vthPtr = reinterpret_cast<const size_t*>(vth_desc);
        if (vthPtr < recordEnd && *vthPtr != 0) {
          const char* vecTypeHint = reinterpret_cast<const char*>(*vthPtr);
          kernel->setVecTypeHint(vecTypeHint);
        }

        if (kernel->createSignature(params)) {
            return true;
        }
    }
    else if (symbol.compare(suffixPos, sizeof(_nature) - 1, _nature) == 0) {
        uint32_t* recordPtr = (uint32_t*) value;
        kernel->nature_ = (uint)recordPtr[0];
        kernel->privateSize_ = (uint)recordPtr[1];
        return true;
    }

    return false;
}

static bool
setKernelInfoCallbackCStr(const char* symbol, const void* value, void* data) {
  std::string symbolString(symbol);
  return setKernelInfoCallback(symbolString, value, data);
}

static bool
setSymbolsCallback(std::string symbol, const void* value, void* data)
{
    device::ClBinary* clbinary = (device::ClBinary*) data;
    const char __OpenCL_[] = "__OpenCL_";
    const char _stub[] = "_stub";
    const char _kernel[] = "_kernel";
    const char _data[] = "_metadata";

    const size_t offset = sizeof(__OpenCL_) - 1;
    if (symbol.compare(0, offset, __OpenCL_) != 0) {
        return false;
    }

    size_t suffixPos = symbol.rfind('_');
    if (suffixPos == std::string::npos) {
        return false;
    }

    if ((symbol.compare(suffixPos, sizeof(_stub) - 1, _stub) == 0) ||
        (symbol.compare(suffixPos, sizeof(_kernel) - 1, _kernel) == 0) ||
        (symbol.compare(suffixPos, sizeof(_data) - 1, _data) == 0)) {

        return clbinary->elfOut()->addSymbol(amd::OclElf::DLL,
                                             const_cast<char*>(symbol.c_str()),
                                             0, false);
    }
    return false;
}

static bool
setSymbolsCallbackCStr(const char* symbol, const void* value, void* data) {
  std::string symbolString(symbol);
  return setSymbolsCallback(symbolString, value, data);
}

// Some helper functions to simplify testing the disassembler
struct DisasData {
public:
  DisasData(std::stringstream *stream,
            aclJITObjectImage im, aclCompiler* cmpl)
            : asmstream(stream), image(im), compiler(cmpl) {};
  std::stringstream *asmstream;
  aclJITObjectImage image;
  aclCompiler* compiler;
};

#if defined(LEGACY_COMPLIB)
static bool
disasSymbolsCallback(std::string symbol, const void* value, void* data)
{
    DisasData* disasData = (DisasData*) data;
    std::stringstream &asmstream = *(disasData->asmstream);
    aclJITObjectImage image = disasData->image;
    aclCompiler* compiler = disasData->compiler;
    const char __OpenCL_[] = "__OpenCL_";
    const char _stub[] = "_stub";
    const char _kernel[] = "_kernel";
    const char _data[] = "_metadata";

    const size_t offset = sizeof(__OpenCL_) - 1;
    if (symbol.compare(0, offset, __OpenCL_) != 0) {
        return false;
    }

    size_t suffixPos = symbol.rfind('_');
    if (suffixPos == std::string::npos) {
        return false;
    }

    if ((symbol.compare(suffixPos, sizeof(_stub) - 1, _stub) == 0) ||
        (symbol.compare(suffixPos, sizeof(_kernel) - 1, _kernel) == 0)) {
      acl_error err = ACL_SUCCESS;
      char* kernelDisas =
        aclJITObjectImageDisassembleKernel(compiler, image, symbol.c_str(), &err);
      if (err != ACL_SUCCESS) {
          LogWarning("aclJITObjectImageDisassembleKernel failed");
          return false;
      }
      asmstream << kernelDisas;
      free(kernelDisas);
    }
    return false;
}

static bool
disasSymbolsCallbackCStr(const char* symbol, const void* value, void* data) {
  std::string symbolString(symbol);
  return disasSymbolsCallback(symbolString, value, data);
}
#endif

bool
Program::compileBinaryToISA(amd::option::Options* options)
{
    const bool has_avx = !options->oVariables->DisableAVX
        && device().hasAVXInstructions();
    const bool has_fma4 = device().hasFMA4Instructions();

#if defined(WITH_ONLINE_COMPILER)
    std::string tempName = amd::Os::getTempFileName();
    dllFileName_ = tempName + "dbg" + "." IF(IS_WINDOWS, "dll", "so");

    acl_error err = ACL_SUCCESS;
    aclTargetInfo aclinfo = info(has_avx ?
      /*has_fma4 ? "Bulldozer" :*/
            "Corei7_AVX" :
            "Athlon64");

    aclBinaryOptions binOpts = {0};
    binOpts.struct_size = sizeof(binOpts);
    binOpts.elfclass = aclinfo.arch_id == aclX64 ? ELFCLASS64 : ELFCLASS32;
    binOpts.bitness = ELFDATA2LSB;
    binOpts.alloc = &::malloc;
    binOpts.dealloc = &::free;

    aclBinary* bin = aclBinaryInit(sizeof(aclBinary), &aclinfo, &binOpts, &err);
    if (err != ACL_SUCCESS) {
        buildLog_ += "Internal error: Setting up input OpenCL binary failed!\n";
        LogWarning("aclBinaryInit failed");
        return false;
    }

    bool spirFlag = std::string::npos != options->clcOptions.find("--spir")
        || llvmBinaryIsSpir_;
    if (ACL_SUCCESS != aclInsertSection(compiler(), bin,
            llvmBinary_.data(), llvmBinary_.size(),
              spirFlag ? aclSPIR : aclLLVMIR )) {
        LogWarning("aclInsertSection failed");
        aclBinaryFini(bin);
        return false;
    }

    // temporary solution to synchronize buildNo between runtime and complib
    // until we move runtime inside complib
    ((amd::option::Options*)bin->options)->setBuildNo(options->getBuildNo());

    err = aclCompile(compiler(), bin, options->origOptionStr.c_str(),
        spirFlag ? ACL_TYPE_SPIR_BINARY : ACL_TYPE_LLVMIR_BINARY,
          ACL_TYPE_ISA, NULL);

    buildLog_ += aclGetCompilerLog(compiler());

    if (err != ACL_SUCCESS) {
        LogWarning("aclCompile failed");
        aclBinaryFini(bin);
        return false;
    }

    if (options->oVariables->BinBIF30) {
        if (!createBIFBinary(bin)) {
            aclBinaryFini(bin);
            return false;
        }
    }

    if (options->oVariables->BinAS  && !options->oVariables->UseJIT) {
      size_t len = 0;
      const char* asmtext =
        static_cast<const char*>(aclExtractSection(compiler(), bin,
                                                   &len, aclCODEGEN, &err));
      if (err != ACL_SUCCESS) {
        LogWarning("aclExtractSection failed");
        aclBinaryFini(bin);
        return false;
      }

      // Store the Asm text in ASTEXT section unless the JIT is used

      if (!clBinary()->storeX86Asm(asmtext, len)) {
        buildLog_ += "Internal Error:  Storing X86 ASM failed!\n";
        return false;
      }
    }

    size_t len = 0;
    const void* isa = aclExtractSection(compiler(), bin,
        &len, aclTEXT, &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclExtractSection failed");
        aclBinaryFini(bin);
        return false;
    }

    if (options->oVariables->UseJIT) {
      //      printf("Using the jit!\n");
      aclJITObjectImage objectImage = aclJITObjectImageCreate(compiler(), isa, len, bin, &err);
      if (err != ACL_SUCCESS) {
          LogWarning("aclJITObjectImageCreate failed");
          aclBinaryFini(bin);
          return false;
      }
      err = aclJITObjectImageFinalize(compiler(), objectImage);
      if (err != ACL_SUCCESS) {
          LogWarning("aclJITObjectImageFinalize failed");
          aclBinaryFini(bin);
          return false;
      }
      setJITBinary(objectImage);
      aclBinaryFini(bin);

    // Store the object image binary in the CL binary;
      if (!clBinary()->storeX86JIT(*this)) {
        buildLog_ += "Internal Error:  Storing X86 DLL failed!\n";
        return false;
    }

#if 0
      // Debug stuff. Try and disassemble all kernels and stubs
      std::stringstream asmtext;
      DisasData disasData(&asmtext, objectImage, compiler());
      err = aclJITObjectImageIterateSymbols(compiler(), objectImage,
                                      disasSymbolsCallbackCStr,
                                      &disasData);
      if (err != ACL_SUCCESS) {
          LogWarning("aclJITObjectImageIterateSymbols failed");
          return false;
      }
      printf("DisasSize: %d\nDisas: %s\n", (int)asmtext.str().size(),
             asmtext.str().c_str());

#endif
      return true;
    }

    std::fstream f;
    f.open(dllFileName_.c_str(), std::fstream::out | std::fstream::binary);
    f.write(static_cast<const char*>(isa), len);
    f.close();

    aclBinaryFini(bin);

    if (f.fail() || f.bad()) {
        buildLog_ += "Internal error: fail to create an internal file!\n";
        return false;
    }

    // Store the dll binary in the CL binary;
    if (!clBinary()->storeX86(*this, dllFileName_)) {
        buildLog_ += "Internal Error:  Storing X86 DLL failed!\n";
        return false;
    }

    return true;
#endif // WITH_ONLINE_COMPILER
    return false;
}

bool
Program::initBuild(amd::option::Options* options)
{
    if (!this->::device::Program::initBuild(options)) {
        return false;
    }

    options->setPerBuildInfo("cpu",
        clBinary()->getEncryptCode(), false);

    /*
       -f[no-]bin-source    : control .source
       -f[no-]bin-llvmir    : control .llvmir
       -f[no-]bin-amdil     : control .amdil
       -f[no-]bin-exe       : control .text

       Default:  -fno-bin-source -fbin-llvmir -fno-bin-amdil -fbin-exe
     */
    // Elf Binary setup
    clBinary()->init(options);

    std::string outFileName;
    if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
      outFileName = options->getDumpFileName(".bin");
    }
    if (!clBinary()->setElfOut(LP64_SWITCH(ELFCLASS32, ELFCLASS64),
                               (outFileName.size() > 0)
                               ? outFileName.c_str() : NULL)) {
        LogError("setup elfout for CPU failed");
        return false;
    }

    return true;
}

bool
Program::finiBuild(bool isBuildGood)
{
    clBinary()->resetElfOut();
    clBinary()->resetElfIn();

    if (!isBuildGood) {
        // Prevent the encrypted binary form leaking out
        clBinary()->setBinary(NULL, 0);
    }

    return this->::device::Program::finiBuild(isBuildGood);
}

bool
Program::compileImpl(
    const std::string& sourceCode,
    const std::vector<const std::string*>& headers,
    const char** headerIncludeNames,
    amd::option::Options* options)
{
#if defined(WITH_ONLINE_COMPILER)
    std::string tempFolder = amd::Os::getTempPath();

    std::fstream f;
    std::vector<std::string> headerFileNames(headers.size());
    std::vector<std::string> newDirs;
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string headerPath = tempFolder;
        std::string headerIncludeName(headerIncludeNames[i]);
        // replace / in path with current os's file separator
        if (amd::Os::fileSeparator() != '/') {
            for (std::string::iterator it = headerIncludeName.begin(),
                 end = headerIncludeName.end();
                 it != end;
                 ++it) {
                if (*it == '/') *it = amd::Os::fileSeparator();
            }
        }
        size_t pos = headerIncludeName.rfind(amd::Os::fileSeparator());
        if (pos != std::string::npos) {
            headerPath += amd::Os::fileSeparator();
            headerPath += headerIncludeName.substr(0, pos);
            headerIncludeName = headerIncludeName.substr(pos+1);
        }
        if (!amd::Os::pathExists(headerPath)) {
            bool ret = amd::Os::createPath(headerPath);
            assert(ret && "failed creating path!");
            newDirs.push_back(headerPath);
        }
        std::string headerFullName
            = headerPath + amd::Os::fileSeparator() + headerIncludeName;
        headerFileNames[i] = headerFullName;
        f.open(headerFullName.c_str(), std::fstream::out);
        assert(!f.fail() && "failed creating header file!");
        f.write(headers[i]->c_str(), headers[i]->length());
        f.close();
    }

    acl_error err = ACL_SUCCESS;
    aclTargetInfo aclinfo = info();

    aclBinaryOptions binOpts = {0};
    binOpts.struct_size = sizeof(binOpts);
    binOpts.elfclass = aclinfo.arch_id == aclX64 ? ELFCLASS64 : ELFCLASS32;
    binOpts.bitness = ELFDATA2LSB;
    binOpts.alloc = &::malloc;
    binOpts.dealloc = &::free;

    aclBinary* bin = aclBinaryInit(sizeof(aclBinary), &aclinfo, &binOpts, &err);
    if (err != ACL_SUCCESS) {
        buildLog_ += "Internal error: Setting up input OpenCL binary failed!\n";
        LogWarning("aclBinaryInit failed");
        return false;
    }

    if (ACL_SUCCESS != aclInsertSection(compiler(), bin,
            sourceCode.c_str(), sourceCode.size(), aclSOURCE)) {
        LogWarning("aclInsertSection failed");
        aclBinaryFini(bin);
        return false;
    }

    // temporary solution to synchronize buildNo between runtime and complib
    // until we move runtime inside complib
    ((amd::option::Options*)bin->options)->setBuildNo(options->getBuildNo());

    std::stringstream opts;
    std::string token;
    opts << options->origOptionStr.c_str();

    if (options->origOptionStr.find("-cl-std=CL") == std::string::npos) {
        switch(OPENCL_MAJOR*100 + OPENCL_MINOR*10) {
        case 100: opts << " -cl-std=CL1.0"; break;
        case 110: opts << " -cl-std=CL1.1"; break;         
        case 200: default:
        case 120: opts << " -cl-std=CL1.2"; break;
        }
    }

    //Add only for CL2.0 and later
    bool spirFlag = false;
    if (options->oVariables->CLStd[2] >= '2') {
        opts << " -D" << "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE="
            << device().info().maxGlobalVariableSize_;
        spirFlag = true;
    }

    // FIXME: Should we prefix everything with -Wf,?
    std::istringstream iss(options->clcOptions);
    while (getline(iss, token, ' ')) {
        if (!token.empty()) {
            // Check if this is a -D option
            if (token.compare("-D") == 0) {
                // It is, skip payload
                getline(iss, token, ' ');
                continue;
            }
            opts << " -Wf," << token;
        }
    }

    if (!headers.empty()) {
      opts << " -I" << tempFolder;
    }

    if (device().info().imageSupport_) {
        opts << " -D__IMAGE_SUPPORT__=1";
    }
    if (device().hasFMA4Instructions()) {
        opts << " -DFP_FAST_FMA=1 -DFP_FAST_FMAF=1";
    }

    iss.clear();
    iss.str(device().info().extensions_);
    while (getline(iss, token, ' ')) {
        if (!token.empty()) {
            opts << " -D" << token << "=1";
        }
    }

    std::string newOpt = opts.str();
    size_t pos = newOpt.find("-fno-bin-llvmir");
    while (pos != std::string::npos) {
      newOpt.erase(pos, 15);
      pos = newOpt.find("-fno-bin-llvmir");
    }

    err = aclCompile(compiler(), bin, newOpt.c_str(),
        ACL_TYPE_OPENCL, spirFlag ? ACL_TYPE_SPIR_BINARY : ACL_TYPE_LLVMIR_BINARY, NULL);

    buildLog_ += aclGetCompilerLog(compiler());

    if (err != ACL_SUCCESS) {
        LogWarning("aclCompile failed");
        aclBinaryFini(bin);
        return false;
    }

    size_t size = 0;
    const void* llvmir = aclExtractSection(compiler(), bin,
        &size, aclLLVMIR, &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclExtractSection failed");
        aclBinaryFini(bin);
        return false;
    }

    llvmBinary_.assign(reinterpret_cast<const char*>(llvmir), size);
    llvmBinaryIsSpir_ = false;
    aclBinaryFini(bin);

    if (clBinary()->saveSOURCE()) {
        clBinary()->elfOut()->addSection(
            amd::OclElf::SOURCE, sourceCode.data(), sourceCode.length());
    }
    if (clBinary()->saveLLVMIR()) {
        clBinary()->elfOut()->addSection(
            amd::OclElf::LLVMIR, llvmBinary_.data(), llvmBinary_.size(), false);
        // store the original compile options
        clBinary()->storeCompileOptions(compileOptions_);
    }

    return true;
#else // WITH_ONLINE_COMPILER
    return false;
#endif
}

bool
Program::loadDllCode(amd::option::Options* options, bool addElfSymbols)
{
    if(options->oVariables->UseJIT) {
      acl_error err = ACL_SUCCESS;
      aclJITObjectImage objectImage = getJITBinary();
      err = aclJITObjectImageIterateSymbols(compiler(), objectImage,
                                      setKernelInfoCallbackCStr, this);
      if (err != ACL_SUCCESS) {
          LogWarning("aclJITObjectImageIterateSymbols failed");
          return false;
      }
      err = aclJITObjectImageIterateSymbols(compiler(), objectImage,
                                      setSymbolsCallbackCStr, clBinary());
      if (err != ACL_SUCCESS) {
          LogWarning("aclJITObjectImageIterateSymbols failed");
          return false;
      }
      size_t size = aclJITObjectImageGetGlobalsSize(compiler(), objectImage, &err);
      if (err != ACL_SUCCESS) {
          LogWarning("aclJITObjectImageGetGlobalsSize failed");
          return false;
      }
      setGlobalVariableTotalSize(size);
      return true;
    }
    // Check if we have a URI
#if defined(_WIN32)
    UINT prevMode = ::SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
                                             
    handle_ = ::LoadLibraryEx(
        dllFileName_.c_str(), NULL,DONT_RESOLVE_DLL_REFERENCES);

    ::SetErrorMode(prevMode);
#else
    handle_ = amd::Os::loadLibrary(dllFileName_.c_str());
#endif
    if (!handle_) {
        return false;
    }

    if (!amd::Os::iterateSymbols(handle_, setKernelInfoCallback, this)) {
        return false;
    }

    // Add cpu symbols into elf
    if (addElfSymbols) {
        if (!amd::Os::iterateSymbols(handle_, setSymbolsCallback, clBinary())) {
            return false;
        }
    }

    return true;
}

bool
Program::linkImpl(amd::option::Options* options)
{
#if defined(WITH_ONLINE_COMPILER)
    // If we don't have LLVM binary then attempt to use OCL binary
    if (llvmBinary_.empty()) {
        // Load ISA
        // For elf format, setup elfIn() and this elfIn() will be released
        // at the end of build by finiBuild().
        if (!clBinary()->setElfIn(LP64_SWITCH(ELFCLASS32, ELFCLASS64))) {
            buildLog_ += "Internal error: Setting up input OpenCL binary failed!\n";
            LogError("Setting up input binary failed");
            return false;
        }

        if (options->oVariables->UseJIT) {
          bool hasJITBinary;
          if (!clBinary()->loadX86JIT(*this, hasJITBinary)) {
            return false;
          } else if (hasJITBinary) {
            aclJITObjectImage objectImage = getJITBinary();
            acl_error err = aclJITObjectImageIterateSymbols(compiler(), objectImage,
                                            setKernelInfoCallbackCStr, this);
            if (err != ACL_SUCCESS) {
                LogWarning("aclJITObjectImageIterateSymbols failed");
                return false;
            }
            err = aclJITObjectImageIterateSymbols(compiler(), objectImage,
                                            setSymbolsCallbackCStr, clBinary());
            if (err != ACL_SUCCESS) {
                LogWarning("aclJITObjectImageIterateSymbols failed");
                return false;
            }
            size_t size = aclJITObjectImageGetGlobalsSize(compiler(), objectImage, &err);
            if (err != ACL_SUCCESS) {
                LogWarning("aclJITObjectImageGetGlobalsSize failed");
                return false;
            }
            setGlobalVariableTotalSize(size);
            return true;
          }
          // Fall-through to recompile
        }  else {
            // Trying to load DLL that was generated by out-process as/ld before
            bool hasDLL = false;
            bool loadSuccess = clBinary()->loadX86(*this, dllFileName_, hasDLL);
            if (!loadSuccess) {
                buildLog_ += "Error: loading a kernel from OpenCL binary failed!\n";
                return false;
            }
            else if (hasDLL) {
                if (loadDllCode(options)) {
                    // No rebuid and use the original binary. Release any new binary if there is.
                    clBinary()->restoreOrigBinary();
                    return true;
                }
            }
            // Fall-through to recompile
        }

        // Need to try recompile, check to see if if LLVM IR is present
        if (clBinary()->loadLlvmBinary(llvmBinary_, llvmBinaryIsSpir_) &&
            clBinary()->isRecompilable(llvmBinary_, amd::OclElf::CPU_PLATFORM)) {
            // Copy both .source and .llvmir into the elfout_
            char *section;
            size_t sz;
            if (clBinary()->saveSOURCE() &&
                clBinary()->elfIn()->getSection(amd::OclElf::SOURCE, &section, &sz)) {
                if ((section != NULL) && (sz > 0)) {
                    clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, section, sz);
                }
            }

            if (clBinary()->saveLLVMIR()) {
                clBinary()->elfOut()->addSection(llvmBinaryIsSpir_?amd::OclElf::SPIR:amd::OclElf::LLVMIR,
                                                 llvmBinary_.data(),
                                                 llvmBinary_.size(), false);
            }
        }
        // We failed kernels loading (wrong ASIC?)
        else {
            buildLog_ += "Error: Runtime failed to load kernels from OCL binary!\n";
            LogError(buildLog_.c_str());
            return false;
        }
    }

    // Do we have llvm binary?
    if (!llvmBinary_.empty()) {
        // Compile llvm binary to x86 source code
        if (!compileBinaryToISA(options)) {
            LogError("We failed to compile LLVMIR binary to ASM text!");
            return false;
        }
    }

    setType(TYPE_EXECUTABLE);

    /////////////////////////////////////////////////////////////
    //////////////// check, there is a good place to finish elf objects
    //////////////////////////////////////////////////////////////

    // Load dll executable
    if (loadDllCode(options, clBinary()->saveISA())) {
        if (!createBinary(options)) {
            buildLog_ += "Internal Error: creating OpenCL binary failed!\n";
            return false;
        }
        return true;
    }
    buildLog_ += "Internal Error: loading shared library failed!\n";
#endif // WITH_ONLINE_COMPILER
    return false;
}

bool
Program::linkImpl(
    const std::vector<device::Program*>& inputPrograms,
    amd::option::Options* options,
    bool createLibrary)
{
#if defined(WITH_ONLINE_COMPILER)
    std::vector<std::string*> llvmBinaries(inputPrograms.size());
    std::vector<bool> llvmBinaryIsSpir(inputPrograms.size());
    std::vector<device::Program*>::const_iterator it
        = inputPrograms.begin();
    std::vector<device::Program*>::const_iterator itEnd
        = inputPrograms.end();
    for (size_t i = 0; it != itEnd; ++it, ++i) {
        Program* program = (Program*)*it;

        if (program->llvmBinary_.empty()) {
            if (program->clBinary() == NULL) {
                buildLog_ += "Internal error: Input program not compiled!\n";
                LogError("Loading compiled input object failed");
                return false;
            }

            // If we don't have LLVM binary then attempt to use OCL binary
            // Load ISA
            // For elf format, setup elfIn() and this elfIn() will be released
            // at the end of build by finiBuild().
            if (!program->clBinary()->setElfIn(LP64_SWITCH(ELFCLASS32,
                                                           ELFCLASS64))) {
                buildLog_ += "Internal error: Setting up input OpenCL binary"
                    " failed!\n";
                LogError("Setting up input binary failed");
                return false;
            }

            // Need to try recompile, check to see if if LLVM IR is present
            if (program->clBinary()->loadLlvmBinary(program->llvmBinary_, program->llvmBinaryIsSpir_) &&
                program->clBinary()->isRecompilable(program->llvmBinary_,
                amd::OclElf::CPU_PLATFORM)) {
                // Copy both .source and .llvmir into the elfout_
#if 0
                // TODO: copy source into .source section of elfout_
                char *section;
                size_t sz;
                if (clBinary()->saveSOURCE() &&
                    clBinary()->elfIn()->getSection(amd::OclElf::SOURCE, &section, &sz)) {
                        if ((section != NULL) && (sz > 0)) {
                            clBinary()->elfOut()->addSection(amd::OclElf::SOURCE, section, sz);
                        }
                }
#endif
            }
            // We failed kernels loading (wrong ASIC?)
            else {
                buildLog_ += "Error: Runtime failed to load kernels from OCL "
                    "binary!\n";
                LogError(buildLog_.c_str());
                return false;
            }
        }

        llvmBinaries[i] = &program->llvmBinary_;
        llvmBinaryIsSpir[i] = program->llvmBinaryIsSpir_;
    }

    acl_error err = ACL_SUCCESS;
    aclTargetInfo aclinfo = info();

    aclBinaryOptions binOpts = {0};
    binOpts.struct_size = sizeof(binOpts);
    binOpts.elfclass = aclinfo.arch_id == aclX64 ? ELFCLASS64 : ELFCLASS32;
    binOpts.bitness = ELFDATA2LSB;
    binOpts.alloc = &::malloc;
    binOpts.dealloc = &::free;

    std::vector<aclBinary*> libs(llvmBinaries.size(), NULL);
    for (size_t i = 0; i < libs.size(); ++i) {
        libs[i] = aclBinaryInit(sizeof(aclBinary), &aclinfo, &binOpts, &err);
        if (err != ACL_SUCCESS) {
            buildLog_ += "Internal error: Setting up input OpenCL binary failed!\n";
            LogWarning("aclBinaryInit failed");
            break;
        }

        err = aclInsertSection(compiler(), libs[i],
            llvmBinaries[i]->data(), llvmBinaries[i]->size(),
            llvmBinaryIsSpir[i]?aclSPIR:aclLLVMIR);
        if (err != ACL_SUCCESS) {
            LogWarning("aclInsertSection failed");
            break;
        }

        // temporary solution to synchronize buildNo between runtime and complib
        // until we move runtime inside complib
        ((amd::option::Options*)libs[i]->options)->setBuildNo(
            options->getBuildNo());
    }

    if (libs.size() > 0 && err == ACL_SUCCESS) do {
        unsigned int numLibs = libs.size() - 1;
        bool resultIsSPIR = (llvmBinaryIsSpir[0] && numLibs == 0);
        if (numLibs > 0) {
            err = aclLink(compiler(), libs[0], libs.size() - 1, &libs[1],
                ACL_TYPE_LLVMIR_BINARY, "-create-library", NULL);

            buildLog_ += aclGetCompilerLog(compiler());

            if (err != ACL_SUCCESS) {
                LogWarning("aclLink failed");
                break;
            }
        }

        size_t size = 0;
        const void* llvmir = aclExtractSection(compiler(), libs[0],
            &size, resultIsSPIR?aclSPIR:aclLLVMIR, &err);
        if (err != ACL_SUCCESS) {
            LogWarning("aclExtractSection failed");
            break;
        }

        llvmBinary_.assign(reinterpret_cast<const char*>(llvmir), size);
    } while(0);

    std::for_each(libs.begin(), libs.end(), std::ptr_fun(aclBinaryFini));

    if (err != ACL_SUCCESS) {
        buildLog_ += "Error: linking llvm modules failed!";
        return false;
    }

    if (clBinary()->saveLLVMIR()) {
        clBinary()->elfOut()->addSection(llvmBinaryIsSpir_?amd::OclElf::SPIR:amd::OclElf::LLVMIR,
                                         llvmBinary_.data(),
                                         llvmBinary_.size(),
                                         false);
        // store the original link options
        clBinary()->storeLinkOptions(linkOptions_);
        clBinary()->storeCompileOptions(compileOptions_);
    }

    // skip the rest if we are building an opencl library
    if (createLibrary) {
        setType(TYPE_LIBRARY);
        if (!createBinary(options)) {
            buildLog_ += "Intenral error: creating OpenCL binary failed\n";
            return false;
        }

        return true;
    }

    // Compile llvm binary to x86 source code
    if (!compileBinaryToISA(options)) {
        LogError("We failed to compile LLVMIR binary to ASM text!");
        return false;
    }

    setType(TYPE_EXECUTABLE);

    /////////////////////////////////////////////////////////////
    //////////////// check, there is a good place to finish elf objects
    //////////////////////////////////////////////////////////////

    // Load dll executable
    if (loadDllCode(options, clBinary()->saveISA())) {
        if (!createBinary(options)) {
            buildLog_ += "Internal Error: creating OpenCL binary failed!\n";
            return false;
        }
        return true;
    }
    buildLog_ += "Internal Error: loading shared library failed!\n";
#endif // WITH_ONLINE_COMPILER
    return false;
}

bool
Program::initClBinary()
{
    if (clBinary_ == NULL) {
        clBinary_ = new ClBinary(device());
        if (clBinary_ == NULL) {
            return false;
        }
    }
    return true;
}

void
Program::releaseClBinary()
{
    if (clBinary_ != NULL) {
        delete clBinary_;
        clBinary_ = NULL;
    }
}

bool
Program::createBinary(amd::option::Options* options)
{
    if (options->oVariables->BinBIF30) {
        return true;
    }

    if (!clBinary()->createElfBinary(options->oVariables->BinEncrypt,
                                     type())) {
        buildLog_ += "Internal Error: creating OpenCL binary failed!\n";
        LogError("Failed to create ELF binary image!");
        return false;
    }
    return true;
}

const aclTargetInfo &
Program::info(const char * str) {
    acl_error err = ACL_SUCCESS;
    info_ = aclGetTargetInfo(LP64_SWITCH("x86", "x86-64"), ( str && str[0] == '\0' ? "Generic" : str ), &err);
    if (err != ACL_SUCCESS) {
        LogWarning("aclGetTargetInfo failed");
    }
    return info_;
}

Program::~Program()
{
    if(getJITBinary() != NULL) {
      aclJITObjectImageDestroy(compiler(), getJITBinary());
    }

    if (!sourceFileName_.empty()) {
        amd::Os::unlink(sourceFileName_.c_str());
    }

    if (handle_ != NULL) {
        amd::Os::unloadLibrary(handle_);
        amd::Os::unlink(dllFileName_);
        char dllName[256];
#ifdef _WIN32
        memcpy(dllName, dllFileName_.data(), dllFileName_.size());
        char* tempName = strrchr(dllName, '.');
        if (tempName != NULL) {
            *tempName = '\0';
            amd::Os::unlink(dllName);
        }
#endif // _WIN32
    }

#if defined(WITH_ONLINE_COMPILER)
    releaseClBinary();
#endif
}

} // namespace cpu
