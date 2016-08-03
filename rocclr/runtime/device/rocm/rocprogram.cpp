//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//


#ifndef WITHOUT_HSA_BACKEND

#include "rocprogram.hpp"

#include "compiler/lib/loaders/elf/elf.hpp"
#include "compiler/lib/utils/options.hpp"
#include "rockernel.hpp"
#include "roccompilerlib.hpp"
#include "utils/bif_section_labels.hpp"

#if defined(WITH_LIGHTNING_COMPILER)
#include "driver/AmdCompiler.h"
#endif // defined(WITH_LIGHTNING_COMPILER)

#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <istream>


#endif  // WITHOUT_HSA_BACKEND

namespace roc {
#ifndef WITHOUT_HSA_BACKEND
    /* Temporary log function for the compiler library */
    static void logFunction(const char *msg, size_t size) {
        std::cout << "Compiler Library log :" << msg << std::endl;
    }

    HSAILProgram::~HSAILProgram() {
        acl_error error;
        // Free the elf binary
        if (binaryElf_ != NULL) {
            error = g_complibApi._aclBinaryFini(binaryElf_);
            if (error != ACL_SUCCESS) {
                LogWarning( "Error while destroying the acl binary \n" );
            }
        }
        // Destroy the executable.
        if (hsaExecutable_.handle != 0) {
          hsa_executable_destroy(hsaExecutable_);
        }
        // Destroy the code object.
        if (hsaProgramCodeObject_.handle != 0) {
          hsa_code_object_destroy(hsaProgramCodeObject_);
        }
        // Destroy the program handle.
        if (hsaProgramHandle_.handle != 0) {
            hsa_ext_program_destroy(hsaProgramHandle_);
        }
        destroyBrigModule();
        destroyBrigContainer();
        releaseClBinary();
   }

    HSAILProgram::HSAILProgram(roc::NullDevice& device): device::Program(device),
        llvmBinary_(),
        binaryElf_(NULL),
        device_(device),
        brigModule_(NULL),
        hsaBrigContainer_(NULL)
    {
        memset(&binOpts_, 0, sizeof(binOpts_));
        binOpts_.struct_size = sizeof(binOpts_);
        //binOpts_.elfclass = LP64_SWITCH( ELFCLASS32, ELFCLASS64 );
        //Setting as 32 bit because hsail64 returns an invalid aclTargetInfo
        //when aclGetTargetInfo is called - EPR# 377910
        binOpts_.elfclass = ELFCLASS32;
        binOpts_.bitness = ELFDATA2LSB;
        binOpts_.alloc = &::malloc;
        binOpts_.dealloc = &::free;
        hsaProgramHandle_.handle = 0;
        hsaProgramCodeObject_.handle = 0;
        hsaExecutable_.handle = 0;
    }

    bool HSAILProgram::initClBinary(char *binaryIn, size_t size) {  // Save the
        // original
        // binary that
        // isn't owned
        // by ClBinary
        clBinary()->saveOrigBinary(binaryIn, size);

        char *bin = binaryIn;
        size_t sz = size;

        int encryptCode;

        char *decryptedBin;
        size_t decryptedSize;
        if (!clBinary()->decryptElf(binaryIn, size,
            &decryptedBin, &decryptedSize, &encryptCode)) {
                return false;
        }
        if (decryptedBin != NULL) {
            // It is decrypted binary.
            bin = decryptedBin;
            sz = decryptedSize;
        }

        // Both 32-bit and 64-bit are allowed!
        if (!amd::isElfMagic(bin)) {
            // Invalid binary.
            if (decryptedBin != NULL) {
                delete[]decryptedBin;
            }
            return false;
        }

        clBinary()->setFlags(encryptCode);

        return clBinary()->setBinary(bin, sz, (decryptedBin != NULL));
    }


    bool HSAILProgram::initBuild(amd::option::Options *options) {
        compileOptions_ = options->origOptionStr;
        
        if (!device::Program::initBuild(options)) {
            return false;
        }
        // Elf Binary setup
        std::string outFileName;

        // true means hsail required
        clBinary()->init(options, true);
        if (options->isDumpFlagSet(amd::option::DUMP_BIF)) {
            outFileName = options->getDumpFileName(".bin");
        }

        bool useELF64 = getCompilerOptions()->oVariables->EnableGpuElf64;
        if (!clBinary()->setElfOut(useELF64 ? ELFCLASS64 : ELFCLASS32,
            (outFileName.size() >
            0) ? outFileName.c_str() : NULL)) {
                LogError("Setup elf out for gpu failed");
                return false;
        }
        return true;
    }

    // ! post-compile setup for GPU
    bool HSAILProgram::finiBuild(bool isBuildGood) {
        clBinary()->resetElfOut();
        clBinary()->resetElfIn();

        if (!isBuildGood) {
            // Prevent the encrypted binary form leaking out
            clBinary()->setBinary(NULL, 0);
            
        }

        return device::Program::finiBuild(isBuildGood);
    }

    static char *readFile(std::string source_filename, size_t &size) {
        FILE *fp = ::fopen(source_filename.c_str(), "rb");
        unsigned int length;
        size_t offset = 0;
        char *ptr;

        if (!fp) {
            return NULL;
        }

        // obtain file size.
        ::fseek(fp, 0, SEEK_END);
        length = ::ftell(fp);
        ::rewind(fp);

        ptr = reinterpret_cast<char *>(malloc(offset + length + 1));
        if (length != fread(&ptr[offset], 1, length, fp)) {
            free(ptr);
            return NULL;
        }

        ptr[offset + length] = '\0';
        size = offset + length;
        ::fclose(fp);
        return ptr;
    }

    aclType HSAILProgram::getCompilationStagesFromBinary(std::vector<aclType>& completeStages, bool& needOptionsCheck)
    {
        acl_error errorCode;
        size_t secSize = 0;
        completeStages.clear();
        aclType from = ACL_TYPE_DEFAULT;
        needOptionsCheck = true;
        size_t boolSize = sizeof(bool);
        //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
        // Checking llvmir in .llvmir section
        bool containsLlvmirText = true;
        errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_LLVMIR, NULL, &containsLlvmirText, &boolSize);
        if (errorCode != ACL_SUCCESS) {
            containsLlvmirText = false;
        }
        // Checking compile & link options in .comment section
        bool containsOpts = true;
        errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_OPTIONS, NULL, &containsOpts, &boolSize);
        if (errorCode != ACL_SUCCESS) {
          containsOpts = false;
        }
        if (containsLlvmirText && containsOpts) {
            completeStages.push_back(from);
            from = ACL_TYPE_LLVMIR_BINARY;
        }
        // Checking HSAIL in .cg section
        bool containsHsailText = true;
        errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_HSAIL, NULL, &containsHsailText, &boolSize);
        if (errorCode != ACL_SUCCESS) {
            containsHsailText = false;
        }
        // Checking BRIG sections
        bool containsBrig = true;
        errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_BRIG, NULL, &containsBrig, &boolSize);
        if (errorCode != ACL_SUCCESS) {
            containsBrig = false;
        }
        if (containsBrig) {
            completeStages.push_back(from);
            from = ACL_TYPE_HSAIL_BINARY;
            // Here we should check that CG stage was done.
            // Right now there are 2 criterions to check it (besides BRIG itself):
            // 1. matadata symbols symOpenclKernel for every kernel.
            // 2. HSAIL text in aclCODEGEN section.
            // Unfortunately there is no appropriate way in Compiler Lib to check 1.
            // because kernel names are unknown here, therefore only 2.
            if (containsHsailText) {
                completeStages.push_back(from);
                from = ACL_TYPE_CG;
            }
        }
        else if (containsHsailText) {
            completeStages.push_back(from);
            from = ACL_TYPE_HSAIL_TEXT;
        }
        // Checking ISA in .text section
        bool containsShaderIsa = true;
        errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_CONTAINS_ISA, NULL, &containsShaderIsa, &boolSize);
        if (errorCode != ACL_SUCCESS) {
            containsShaderIsa = false;
        }
        if (containsShaderIsa) {
            completeStages.push_back(from);
            from = ACL_TYPE_ISA;
        }
        std::string sCurOptions = compileOptions_ + linkOptions_;
        amd::option::Options curOptions;
        if (!amd::option::parseAllOptions(sCurOptions, curOptions)) {
            buildLog_ += curOptions.optionsLog();
            LogError("Parsing compile options failed.");
            return ACL_TYPE_DEFAULT;
        }
        switch (from) {
        // compile from HSAIL text, no matter prev. stages and options
        case ACL_TYPE_HSAIL_TEXT:
            needOptionsCheck = false;
            break;
        case ACL_TYPE_HSAIL_BINARY:
        case ACL_TYPE_CG:
            // do not check options, if LLVMIR is absent or might be absent or options are absent
            if (curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
                needOptionsCheck = false;
            }
            break;
        case ACL_TYPE_ISA:
            // do not check options, if LLVMIR is absent or might be absent or options are absent
            if (curOptions.oVariables->BinLLVMIR || !containsLlvmirText || !containsOpts) {
                needOptionsCheck = false;
            }
            if (containsBrig && containsHsailText && curOptions.oVariables->BinHSAIL) {
                needOptionsCheck = false;
            // recompile from prev. stage, if BRIG || HSAIL are absent
            } else {
                from = completeStages.back();
                completeStages.pop_back();
                needOptionsCheck = true;
            }
            break;
        // recompilation might be needed
        case ACL_TYPE_LLVMIR_BINARY:
        case ACL_TYPE_DEFAULT:
        default:
            break;
        }
        return from;
    }

    aclType HSAILProgram::getNextCompilationStageFromBinary(amd::option::Options* options) {
        aclType continueCompileFrom = ACL_TYPE_DEFAULT;
        binary_t binary = this->binary();
        // If the binary already exists
        if ((binary.first != NULL) && (binary.second > 0)) {
            void *mem = const_cast<void *>(binary.first);
            acl_error errorCode;
            binaryElf_ = g_complibApi._aclReadFromMem(mem, binary.second, &errorCode);
            if (errorCode != ACL_SUCCESS) {
                buildLog_ += "Error while BRIG Codegen phase: aclReadFromMem failure \n" ;
                LogWarning("aclReadFromMem failed");
                return continueCompileFrom;
          }
          // Calculate the next stage to compile from, based on sections in binaryElf_;
          // No any validity checks here
          std::vector<aclType> completeStages;
          bool needOptionsCheck = true;
          continueCompileFrom = getCompilationStagesFromBinary(completeStages, needOptionsCheck);
          // Saving binary in the interface class,
          // which also load compile & link options from binary
          setBinary(static_cast<char*>(mem), binary.second);
          if (!options || !needOptionsCheck) {
              return continueCompileFrom;
          }
          bool recompile = false;
          //! @todo Should we also check for ACL_TYPE_OPENCL & ACL_TYPE_LLVMIR_TEXT?
          switch (continueCompileFrom) {
          case ACL_TYPE_HSAIL_BINARY:
          case ACL_TYPE_CG:
          case ACL_TYPE_ISA: {
              // Compare options loaded from binary with current ones, recompile if differ;
              // If compile options are absent in binary, do not compare and recompile
              if (compileOptions_.empty())
                  break;
              const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symOpenclCompilerOptions);
              assert(symbol && "symbol not found");
              std::string symName = std::string(symbol->str[bif::PRE]) + std::string(symbol->str[bif::POST]);
              size_t symSize = 0;
              const void *opts = g_complibApi._aclExtractSymbol(device().compiler(),
                  binaryElf_, &symSize, aclCOMMENT, symName.c_str(), &errorCode);
              if (errorCode != ACL_SUCCESS) {
                  recompile = true;
                  break;
              }
              std::string sBinOptions = std::string((char*)opts, symSize);
              std::string sCurOptions = compileOptions_ + linkOptions_;
              amd::option::Options curOptions, binOptions;
              if (!amd::option::parseAllOptions(sBinOptions, binOptions)) {
                  buildLog_ += binOptions.optionsLog();
                  LogError("Parsing compile options from binary failed.");
                  return ACL_TYPE_DEFAULT;
              }
              if (!amd::option::parseAllOptions(sCurOptions, curOptions)) {
                  buildLog_ += curOptions.optionsLog();
                  LogError("Parsing compile options failed.");
                  return ACL_TYPE_DEFAULT;
              }
              if (!curOptions.equals(binOptions)) {
                  recompile = true;
              }
              break;
          }
          default:
              break;
          }
          if (recompile) {
              while (!completeStages.empty()) {
                  continueCompileFrom = completeStages.back();
                  if (continueCompileFrom == ACL_TYPE_LLVMIR_BINARY ||
                      continueCompileFrom == ACL_TYPE_DEFAULT) {
                      break;
                  }
                  completeStages.pop_back();
              }
          }
        }
        return continueCompileFrom;
    }

    bool HSAILProgram::saveBinaryAndSetType(type_t type) {
        //Write binary to memory
        void *rawBinary = NULL;
        size_t size;
        if (g_complibApi._aclWriteToMem(binaryElf_, &rawBinary, &size)
            != ACL_SUCCESS) {
                buildLog_ += "Failed to write binary to memory \n";
                return false;
        }
        clBinary()->saveBIFBinary((char*)rawBinary, size);
        //Set the type of binary
        setType(type);
        //Free memory containing rawBinary
        binaryElf_->binOpts.dealloc(rawBinary);
        return true;
    }

    bool HSAILProgram::linkImpl(const std::vector<Program *> &inputPrograms,
        amd::option::Options *options,
        bool createLibrary) {
            std::vector<device::Program *>::const_iterator it
                = inputPrograms.begin();
            std::vector<device::Program *>::const_iterator itEnd
                = inputPrograms.end();
            acl_error errorCode;

            // For each program we need to extract the LLVMIR and create
            // aclBinary for each
            std::vector<aclBinary *> binaries_to_link;

            for (size_t i = 0; it != itEnd; ++it, ++i) {
                HSAILProgram *program = (HSAILProgram *)*it;
                // Check if the program was created with clCreateProgramWIthBinary
                binary_t binary = program->binary();
                if ((binary.first != NULL) && (binary.second > 0)) {
                    // Binary already exists -- we can also check if there is no
                    // opencl source code
                    // Need to check if LLVMIR exists in the binary
                    // If LLVMIR does not exist then is it valid
                    // We need to pull out all the compiled kernels
                    // We cannot do this at present because we need at least
                    // Hsail text to pull the kernels oout
                    void *mem = const_cast<void *>(binary.first);
                    binaryElf_ = g_complibApi._aclReadFromMem(mem,
                        binary.second,
                        &errorCode);

                    if (errorCode != ACL_SUCCESS) {
                        LogWarning("Error while linking : Could not read from raw binary");
                        return false;
                    }
                }
                // At this stage each HSAILProgram contains a valid binary_elf
                // Check if LLVMIR is in the binary
                size_t boolSize = sizeof(bool);
                bool containsLLLVMIR = false;
                errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_,
                            RT_CONTAINS_LLVMIR, NULL, &containsLLLVMIR, &boolSize);
                if (errorCode != ACL_SUCCESS || !containsLLLVMIR) {
                    buildLog_ +="Error while linking : Invalid binary (Missing LLVMIR section)";
                    return false;
                }
                // Create a new aclBinary for each LLVMIR and save it in a list
                aclBIFVersion ver = g_complibApi._aclBinaryVersion(binaryElf_);
                aclBinary *bin = g_complibApi._aclCreateFromBinary(binaryElf_, ver);
                binaries_to_link.push_back(bin);
            }

            // At this stage each HSAILProgram in the list has an aclBinary initialized
            // and contains LLVMIR
            // We can now go ahead and link them.
            if (binaries_to_link.size() > 1) {
                errorCode = g_complibApi._aclLink(device().compiler(),
                    binaries_to_link[0],
                    binaries_to_link.size() - 1,
                    &binaries_to_link[1],
                    ACL_TYPE_LLVMIR_BINARY,
                    "-create-library",
                    NULL);
            }
            else {
                errorCode = g_complibApi._aclLink(device().compiler(),
                    binaries_to_link[0],
                    0,
                    NULL,
                    ACL_TYPE_LLVMIR_BINARY,
                    "-create-library",
                    NULL);
            }
            if (errorCode != ACL_SUCCESS) {
                buildLog_ += "Failed to link programs";
                return false;
            }
            // Store the newly linked aclBinary for this program.
            binaryElf_ = binaries_to_link[0];
            // Free all the other aclBinaries
            for (size_t i = 1; i < binaries_to_link.size(); i++) {
                g_complibApi._aclBinaryFini(binaries_to_link[i]);
            }
            if (createLibrary) {
                saveBinaryAndSetType(TYPE_LIBRARY);
                return true;
            }

            // Now call linkImpl with the new options
            return linkImpl(options);
    }

    bool HSAILProgram::initBrigModule() {
        const char *symbol_name = "__BRIG__";
        BrigModuleHeader* brig; 
        acl_error error_code;
        size_t size;
        const void* symbol_data = g_complibApi._aclExtractSymbol(
            device().compiler(),
            binaryElf_,
            &size,
            aclBRIG,
            symbol_name,
            &error_code);
        if (error_code != ACL_SUCCESS) {
           std::string error = "Could not find Brig in BIF: ";
           error += symbol_name;
           LogError(error.c_str());
           buildLog_ +=  error;
           return false;
        }
        brig = (BrigModuleHeader*)malloc(size);
        memcpy(brig, symbol_data, size);
        brigModule_ = brig;
        return true;
    }
   void HSAILProgram::destroyBrigModule() {
    if (brigModule_ != NULL) {
        free(brigModule_);
    }
   }
    bool HSAILProgram::initBrigContainer() {
        assert(brigModule_ != NULL);

        //Create a BRIG container
        hsaBrigContainer_ = new BrigContainer(brigModule_);
        if (!hsaBrigContainer_) {
            return false;
        }
        return true;
    }

    void HSAILProgram::destroyBrigContainer() {
        delete (hsaBrigContainer_);
    }

    
    void HSAILProgram::hsaError(const char *msg, hsa_status_t status) {
      std::string fmsg;
      fmsg += msg;
      if (status != HSA_STATUS_SUCCESS) {
        const char *hmsg = 0;
        hsa_status_string(status, &hmsg);
        if (hmsg) {
          fmsg += ": ";
          fmsg += hmsg;
        }
      }
      LogError(fmsg.c_str());
      buildLog_ += fmsg;
    }

    bool HSAILProgram::linkImpl(amd::option::Options *options) {
        acl_error errorCode;
        aclType continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;
        bool finalize = true;
        // If !binaryElf_ then program must have been created using clCreateProgramWithBinary
        if (!binaryElf_) {
            continueCompileFrom = getNextCompilationStageFromBinary(options);
        }
        switch (continueCompileFrom) {
        // Compilation from ACL_TYPE_LLVMIR_BINARY to ACL_TYPE_CG in cases:
        // 1. if the program is not created with binary;
        // 2. if the program is created with binary and contains only .llvmir & .comment
        // 3. if the program is created with binary, contains .llvmir, .comment, brig sections,
        //    but the binary's compile & link options differ from current ones (recompilation);
        case ACL_TYPE_LLVMIR_BINARY:
        // Compilation from ACL_TYPE_HSAIL_BINARY to ACL_TYPE_CG in cases:
        // 1. if the program is created with binary and contains only brig sections
        case ACL_TYPE_HSAIL_BINARY:
        // Compilation from ACL_TYPE_HSAIL_TEXT to ACL_TYPE_CG in cases:
        // 1. if the program is created with binary and contains only hsail text
        case ACL_TYPE_HSAIL_TEXT: {
            std::string curOptions = options->origOptionStr + hsailOptions();
            errorCode = g_complibApi._aclCompile(device().compiler(), binaryElf_,
                curOptions.c_str(), continueCompileFrom, ACL_TYPE_CG, logFunction);
            buildLog_ += g_complibApi._aclGetCompilerLog(device().compiler());
            if (errorCode != ACL_SUCCESS) {
                buildLog_ += "Error while BRIG Codegen phase: compilation error \n" ;
                return false;
            }
            break;
        }
        case ACL_TYPE_CG:
            break;
        case ACL_TYPE_ISA:
            finalize = false;
            break;
        default:
            buildLog_ += "Error while BRIG Codegen phase: the binary is incomplete \n" ;
            return false;
        }
        //Stop compilation if it is an offline device - HSA runtime does not
        //support ISA compiled offline
        if (!dev().isOnline()) {
            return true;
        }

        hsa_agent_t hsaDevice = dev().getBackendDevice();
        if (!initBrigModule()) {
            hsaError("Failed to create Brig Module");
            return false;
        }

        // Create a BrigContainer.
        if (!initBrigContainer()) {
            hsaError("Failed to create Brig Container");
            return false;
        }
        // Create a program.
        hsa_status_t status = hsa_ext_program_create(
          HSA_MACHINE_MODEL_LARGE,
          HSA_PROFILE_FULL,
          HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO,
          NULL,
          &hsaProgramHandle_
        );
        if (status != HSA_STATUS_SUCCESS) {
            hsaError("Failed to create hsail program", status);
            return false;
        }

        // Add module to a program.
        hsa_ext_module_t programModule = 
          reinterpret_cast<hsa_ext_module_t>(brigModule_);
        status = hsa_ext_program_add_module(
          hsaProgramHandle_, programModule
        );
        if (status != HSA_STATUS_SUCCESS) {
            hsaError("Failed to add a module to the program", status);
            return false;
        }

        // Obtain agent's Isa.
        hsa_isa_t hsaDeviceIsa;
        status = hsa_agent_get_info(
          hsaDevice, HSA_AGENT_INFO_ISA, &hsaDeviceIsa
        );
        if (status != HSA_STATUS_SUCCESS) {
            hsaError("Failed to create hsail program", status);
            return false;
        }

        // Finalize a program.
        hsa_ext_control_directives_t hsaControlDirectives;
        memset(&hsaControlDirectives, 0, sizeof(hsa_ext_control_directives_t));
        status = hsa_ext_program_finalize(
          hsaProgramHandle_,
          hsaDeviceIsa,
          0,
          hsaControlDirectives,
          NULL,
          HSA_CODE_OBJECT_TYPE_PROGRAM,
          &hsaProgramCodeObject_
        );
        if (status != HSA_STATUS_SUCCESS) {
            hsaError("Failed to finalize hsail program", status);
            return false;
        }

        // HLC always generates full profile
        hsa_profile_t profile = HSA_PROFILE_FULL;

        // Create an executable.
        status = hsa_executable_create(
          profile,
          HSA_EXECUTABLE_STATE_UNFROZEN,
          "",
          &hsaExecutable_
        );
        if (status != HSA_STATUS_SUCCESS) {
            hsaError("Failed to create executable", status);
            return false;
        }

        // Load the code object.
        status = hsa_executable_load_code_object(
          hsaExecutable_, hsaDevice, hsaProgramCodeObject_, NULL
        );
        if (status != HSA_STATUS_SUCCESS) {
            hsaError("Failed to load code object", status);
            return false;
        }

        // Freeze the executable.
        status = hsa_executable_freeze(hsaExecutable_, NULL);
        if (status != HSA_STATUS_SUCCESS) {
            hsaError("Failed to freeze executable", status);
            return false;
        }

        Code first_d = hsaBrigContainer_->code().begin();
        Code last_d = hsaBrigContainer_->code().end();
        //Iterate through the symbols using brig assembler
        for (;first_d != last_d;first_d = first_d.next()) {
            if (DirectiveExecutable de = first_d) {
                // Disable function compilation unconditionally.
                // TODO: May remove this after the finalizer supports function compilation.
                if (DirectiveFunction df = first_d) {
                    continue;
                }

                std::string kernelName = (SRef)de.name();
                if (de.linkage() != BRIG_LINKAGE_PROGRAM) {
                  kernelName.insert(0, "am::");
                }
                // Query symbol handle for this symbol.
                hsa_executable_symbol_t kernelSymbol;
                status = hsa_executable_get_symbol(
                  hsaExecutable_, NULL, kernelName.c_str(), hsaDevice, 0, &kernelSymbol
                );
                if (status != HSA_STATUS_SUCCESS) {
                    hsaError("Failed to get executable symbol", status);
                    return false;
                }

                // Query code handle for this symbol.
                uint64_t kernelCodeHandle;
                status = hsa_executable_symbol_get_info(
                  kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle
                );
                if (status != HSA_STATUS_SUCCESS) {
                    hsaError("Failed to get executable symbol info", status);
                    return false;
                }

                std::string openclKernelName = kernelName;
                // Strip the opencl and kernel name
                kernelName = kernelName.substr(strlen("&__OpenCL_"), kernelName.size());
                kernelName = kernelName.substr(0,kernelName.size() - strlen("_kernel"));
                aclMetadata md;
                md.numHiddenKernelArgs = 0;
                size_t sizeOfnumHiddenKernelArgs = sizeof(md.numHiddenKernelArgs);
                errorCode = g_complibApi._aclQueryInfo(device().compiler(), binaryElf_, RT_NUM_KERNEL_HIDDEN_ARGS,
                    openclKernelName.c_str(), &md.numHiddenKernelArgs, &sizeOfnumHiddenKernelArgs);
                if (errorCode != ACL_SUCCESS) {
                    buildLog_ += "Error while Finalization phase: Kernel extra arguments count querying from the ELF failed\n";
                    return false;
                }

                uint32_t workgroupGroupSegmentByteSize;
                status = hsa_executable_symbol_get_info(
                    kernelSymbol,
                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                    &workgroupGroupSegmentByteSize);
                if (status != HSA_STATUS_SUCCESS) {
                  hsaError("Failed to get group segment size info", status);
                  return false;
                }

                uint32_t workitemPrivateSegmentByteSize;
                status = hsa_executable_symbol_get_info(
                    kernelSymbol,
                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                    &workitemPrivateSegmentByteSize);
                if (status != HSA_STATUS_SUCCESS) {
                    hsaError("Failed to get private segment size info", status);
                    return false;
                }

                uint32_t kernargSegmentByteSize;
                status = hsa_executable_symbol_get_info(
                    kernelSymbol,
                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                    &kernargSegmentByteSize);
                if (status != HSA_STATUS_SUCCESS) {
                  hsaError("Failed to get kernarg segment size info", status);
                  return false;
                }

                uint32_t kernargSegmentAlignment;
                status = hsa_executable_symbol_get_info(
                    kernelSymbol,
                    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                    &kernargSegmentAlignment);
                if (status != HSA_STATUS_SUCCESS) {
                  hsaError("Failed to get kernarg segment alignment info", status);
                  return false;
                }

                Kernel *aKernel = new roc::Kernel(
                  kernelName,
                  this,
                  kernelCodeHandle,
                  workgroupGroupSegmentByteSize,
                  workitemPrivateSegmentByteSize,
                  kernargSegmentByteSize,
                  kernargSegmentAlignment,
                  md.numHiddenKernelArgs
                );
                if (!aKernel->init()) {
                    return false;
                }
                aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
                kernels()[kernelName] = aKernel;
            }
        }
        saveBinaryAndSetType(TYPE_EXECUTABLE);
        buildLog_ += g_complibApi._aclGetCompilerLog(device().compiler());
        return true;
    }

    bool HSAILProgram::createBinary(amd::option::Options *options) {
        return false;
    }

    bool HSAILProgram::initClBinary() {
        if (clBinary_ == NULL) {
            clBinary_ = new ClBinary(static_cast<const Device &>(device()));
            if (clBinary_ == NULL) {
                return false;
            }
        }
        return true;
    }

    void HSAILProgram::releaseClBinary() {
        if (clBinary_ != NULL) {
            delete clBinary_;
            clBinary_ = NULL;
        }
    }

    std::string HSAILProgram::hsailOptions() {
        std::string hsailOptions;
        //Set options for the standard device specific options
        //This is just for legacy compiler code
        // All our devices support these options now
        hsailOptions.append(" -DFP_FAST_FMAF=1");
        hsailOptions.append(" -DFP_FAST_FMA=1");
        //TODO: this is a quick fix to restore original f32 denorm flushing
        //Make this target/option dependent
        hsailOptions.append(" -cl-denorms-are-zero");
        //TODO(sramalin) : Query the device for opencl version
        //                 and only set if -cl-std wasn't specified in
        //                 original build options (app)
        //hsailOptions.append(" -cl-std=CL1.2");
        //check if the host is 64 bit or 32 bit
        LP64_ONLY(hsailOptions.append(" -m64"));
        //Now append each extension supported by the device 
        // one by one
        std::string token;
        std::istringstream iss("");
        iss.str(device().info().extensions_); 	  
        while (getline(iss, token, ' ')) {
            if (!token.empty()) {
                hsailOptions.append(" -D");
                hsailOptions.append(token);
                hsailOptions.append("=1");
            }
        }
        return hsailOptions;
    }

#endif  // WITHOUT_HSA_BACKEND
}  // namespace hsa

