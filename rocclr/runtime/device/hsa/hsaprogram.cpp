//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//


#ifndef WITHOUT_FSA_BACKEND

#include "device/hsa/hsaprogram.hpp"

#include "compiler/lib/loaders/elf/elf.hpp"
#include "compiler/lib/utils/options.hpp"
#include "runtime/device/hsa/hsakernel.hpp"
#include "runtime/device/hsa/hsacompilerlib.hpp"
#include "runtime/device/hsa/oclhsa_common.hpp"
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"

#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <istream>


#endif  // WITHOUT_FSA_BACKEND

namespace oclhsa {
#ifndef WITHOUT_FSA_BACKEND
    /* Temporary log function for the compiler library */
    static void logFunction(const char *msg, size_t size) {
        std::cout << "Compiler Library log :" << msg << std::endl;
    }

    FSAILProgram::~FSAILProgram() {
        unloadBrig();
        acl_error error;
        // Free the elf binary
        if (binaryElf_ != NULL) {
            error = g_complibApi._aclBinaryFini(binaryElf_);
            if (error != ACL_SUCCESS) {
                LogWarning( "Error while destroying the acl binary \n" );
            }
        }
   }

    FSAILProgram::FSAILProgram(oclhsa::NullDevice& device): device::Program(device),
        llvmBinary_(),
        binaryElf_(NULL),
        device_(device),
        isBrigLoaded_(false)
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
    }

    bool FSAILProgram::initClBinary(char *binaryIn, size_t size) {  // Save the
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

    bool FSAILProgram::initBuild(amd::option::Options *options) {
        if (!device::Program::initBuild(options)) {
            return false;
        }

        // Need to get device information from CAL !?!?
        // Needs the device pointer from CAL to send to options class
        //
        // Shreyas: Commenting this might cause a bug - keeping this fro now
        // options->setPerBuildInfo("hsa",
        // binary_.getEncryptCode()
        // );

        // Elf Binary setup
        std::string outFileName;

        // true means fsail required
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
    bool FSAILProgram::finiBuild(bool isBuildGood) {
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

    aclType FSAILProgram::getNextCompilationStageFromBinary() {
        acl_error errorCode;
        size_t secSize = 0;
        aclType from = ACL_TYPE_DEFAULT;
        // Checking llvmir in .llvmir section
        bool isLlvmirText = true;
        const void *llvmirText = g_complibApi._aclExtractSection(device().compiler(),
            binaryElf_,
            &secSize,
            aclLLVMIR,
            &errorCode);
        if (errorCode != ACL_SUCCESS) {
            isLlvmirText = false;
        }
        // Checking compile & link options in .comment section
        bool isOpts = true;
        const void *opts = g_complibApi._aclExtractSection(device().compiler(),
            binaryElf_,
            &secSize,
            aclCOMMENT,
            &errorCode);
        if (errorCode != ACL_SUCCESS) {
            isOpts = false;
        }
        if (isLlvmirText) {
            from = ACL_TYPE_LLVMIR_BINARY;
        } else {
            if (!isLlvmirText) {
                buildLog_ +="Error while linking : \
                            Invalid binary (Missing LLVMIR section)\n" ;
            }
            if (!isOpts) {
                buildLog_ +="Warning while linking : \
                            Invalid binary (Missing COMMENT section)\n" ;
            }
            return ACL_TYPE_DEFAULT;
        }
        bool isHsailText = true;
        // Checking HSAIL in .cg section
        const void *hsailText = g_complibApi._aclExtractSection(device().compiler(),
            binaryElf_,
            &secSize,
            aclCODEGEN,
            &errorCode);
        if (errorCode != ACL_SUCCESS) {
            isHsailText = false;
        }
        // Checking BRIG STRTAB in .brig_strtab section
        bool isBrigStrtab = true;
        const void *brigStrtab = g_complibApi._aclExtractSection(device().compiler(),
            binaryElf_,
            &secSize,
            aclBRIGstrs,
            &errorCode);
        if (errorCode != ACL_SUCCESS) {
            isBrigStrtab = false;
        }
        // Checking BRIG CODE in .brig_code section
        bool isBrigCode = true;
        const void *brigCode = g_complibApi._aclExtractSection(device().compiler(),
            binaryElf_,
            &secSize,
            aclBRIGcode,
            &errorCode);
        if (errorCode != ACL_SUCCESS) {
            isBrigCode = false;
        }
        // Checking BRIG OPERANDS in .brig_operands section
        bool isBrigOps = true;
        const void *brigOps = g_complibApi._aclExtractSection(device().compiler(),
            binaryElf_,
            &secSize,
            aclBRIGoprs,
            &errorCode);
        if (errorCode != ACL_SUCCESS) {
            isBrigOps = false;
        }
        if (isHsailText && isBrigStrtab && isBrigCode && isBrigOps) {
            from = ACL_TYPE_HSAIL_BINARY;
        } else if (!isHsailText && !isBrigStrtab && !isBrigCode && !isBrigOps) {
            from = ACL_TYPE_LLVMIR_BINARY;
        } else {
            if (!isHsailText) {
                buildLog_ +="Error while linking : \
                            Invalid binary (Missing CG section)\n" ;
            }
            if (!isBrigStrtab) {
                buildLog_ +="Error while linking : \
                            Invalid binary (Missing BRIG_STRTAB section)\n" ;
            }
            if (!isBrigCode) {
                buildLog_ +="Error while linking : \
                            Invalid binary (Missing BRIG_CODE section)\n" ;
            }
            if (!isBrigOps) {
                buildLog_ +="Error while linking : \
                            Invalid binary (Missing BRIG_OPERANDS section)\n" ;
            }
            return ACL_TYPE_DEFAULT;
        }
        // Checking ISA in .text section
        bool isShaderIsa = true;
        const void *shaderIsa = g_complibApi._aclExtractSection(device().compiler(),
            binaryElf_,
            &secSize,
            aclTEXT,
            &errorCode);
        if (errorCode != ACL_SUCCESS) {
            isShaderIsa = false;
        }
        if (isShaderIsa && from == ACL_TYPE_LLVMIR_BINARY) {
            from = ACL_TYPE_DEFAULT;
        }
        return from;
    }
    bool FSAILProgram::updateAclBinaryWithKernelIsaAndDebug(std::string kernelName){
      assert(brig_.loadmap_section != NULL);
      aclBinary * internalAclBinary = reinterpret_cast<aclBinary*>(brig_.loadmap_section);

      std::string openClKernelName("&__OpenCL_" + kernelName + "_kernel");
      const oclBIFSymbolStruct* isaSymbolStruct = findBIF30SymStruct(symISABinary);
      assert(isaSymbolStruct && "symbol not found");
      std::string kernelIsaSymbol = isaSymbolStruct->str[PRE] +
        openClKernelName +  isaSymbolStruct->str[POST];

      const oclBIFSymbolStruct* debugSymbolStruct = findBIF30SymStruct(symDebugInfo);
      assert(debugSymbolStruct && "symbol not found");
      //For debug symbols, the PRE is used for BRIG debug and the POST is used for
      //ISA debug
      std::string kernelIsaDebugSymbol = debugSymbolStruct->str[POST] + openClKernelName;

      //Extract the ISA section
      size_t symbolSize;
      acl_error errorCode;
      const void* isaSymbol = g_complibApi._aclExtractSymbol(device().compiler(),
        internalAclBinary,
        &symbolSize,
        aclTEXT,
        kernelIsaSymbol.c_str(),
        &errorCode);
      if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Failed to extract ISA for kernel";
        return false;
      }
      //Insert the ISA section
      errorCode = g_complibApi._aclInsertSymbol(device().compiler(),
        binaryElf_,
        isaSymbol,
        symbolSize,
        aclTEXT,
        kernelIsaSymbol.c_str());
      if (errorCode != ACL_SUCCESS) {
        buildLog_ += "Failed to insert ISA for kernel";
        return false;
      }
      const void* debugSymbol = g_complibApi._aclExtractSymbol(device().compiler(),
        internalAclBinary,
        &symbolSize,
        aclHSADEBUG,
        kernelIsaDebugSymbol.c_str(),
        &errorCode);
      //If debug information is available
      if (errorCode == ACL_SUCCESS) {
        //Update binary with the debug section for the kernel
        errorCode = g_complibApi._aclInsertSymbol(device().compiler(),
          binaryElf_,
          debugSymbol,
          symbolSize,
          aclHSADEBUG,
          kernelIsaDebugSymbol.c_str());
        if (errorCode != ACL_SUCCESS) {
          buildLog_ += "Failed to insert debug information for kernel";
          return false;
        }
      }
      return true;
    }
    bool FSAILProgram::ExtractSymbolAndCopy(aclSections id,
        const char *symbol_name,
        void** address_to_copy,
        size_t* symbol_size_bytes,
        bool verify) {
            acl_error error_code;
            *symbol_size_bytes = 0;
            const void* symbol_data = g_complibApi._aclExtractSymbol(
                device().compiler(),
                binaryElf_,
                symbol_size_bytes,
                id,
                symbol_name,
                &error_code);
            //If the section is not mandatory and the section does not exist
            //skip this section
            if (error_code != ACL_SUCCESS) {
              if (!verify) {
                return true;
              }
              std::string error = "Could not find Brig Directive in BIFF: ";
              error += symbol_name;
              LogError(error.c_str());
              buildLog_ +=  error;
              return false;
            }
            *address_to_copy = malloc(*symbol_size_bytes);
            if (*address_to_copy == NULL) {
                LogError(" Failed to allocate memory");
                return false;
            }
            memcpy(*address_to_copy, symbol_data, *symbol_size_bytes);

            return true;
    }

    bool FSAILProgram::saveBinaryAndSetType(type_t type) {
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

    bool FSAILProgram::linkImpl(const std::vector<Program *> &inputPrograms,
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
                FSAILProgram *program = (FSAILProgram *)*it;
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
                // At this stage each FSAILProgram contains a valid binary_elf
                // Check if LLVMIR is in the binary
                // @TODO - Memory leak , cannot free this buffer
                // need to fix this.. File EPR on compiler library
                size_t llvmirSize = 0;
                const void *llvmirText = g_complibApi._aclExtractSection(device().compiler(),
                    binaryElf_,
                    &llvmirSize,
                    aclLLVMIR,
                    &errorCode);
                if (errorCode != ACL_SUCCESS) {
                    buildLog_ +="Error while linking : \
                                Invalid binary (Missing LLVMIR section)" ;
                    return false;
                }
                // Create a new aclBinary for each LLVMIR and save it in a list
                aclBIFVersion ver = g_complibApi._aclBinaryVersion(binaryElf_);
                aclBinary *bin = g_complibApi._aclCreateFromBinary(binaryElf_, ver);
                binaries_to_link.push_back(bin);
            }

            // At this stage each FSAILProgram in the list has an aclBinary initialized
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

    bool FSAILProgram::loadBrig() {
        //Copy all the sections into BRIG
        memset(&brig_, 0 ,sizeof(HsaBrig));
        bool codeStatus = ExtractSymbolAndCopy(aclBRIGcode,
            "__BRIG__code",
            &brig_.code_section,
            &brig_.code_section_byte_size,
            true
            );
        bool oprStatus = ExtractSymbolAndCopy(aclBRIGoprs,
            "__BRIG__operands",
            &brig_.operand_section,
            &brig_.operand_section_byte_size,
            true
            );
        bool strStatus = ExtractSymbolAndCopy(aclBRIGstrs,
            "__BRIG__strtab",
            &brig_.string_section,
            &brig_.string_section_byte_size,
            true
            );
        bool dbgStatus = ExtractSymbolAndCopy(aclHSADEBUG ,
            "__debug_brig__",
            &brig_.debug_section,
            &brig_.debug_section_byte_size,
            false
            );
        if (!codeStatus || !oprStatus || !strStatus || !dbgStatus) {
            LogError("Failed to Extract one or more BRIG sections");
            buildLog_ += "Error: Failed to Extract one or more BRIG sections";
            return false;
        }
        if(hsacoreapi->HsaLoadBrig(device_.getBackendDevice(), &brig_)
            != kHsaStatusSuccess){
            return false;
        }
        isBrigLoaded_ = true;
        return true;
    }

    bool FSAILProgram::unloadBrig() {
        if (isBrigLoaded_ == true) {
            HsaStatus status = hsacoreapi->HsaUnloadBrig(&brig_);
            if (status != kHsaStatusSuccess){
                return false;
            }
            //Destroy the BRIG
            free(brig_.code_section);
            free(brig_.operand_section);
            free(brig_.string_section);
            free(brig_.debug_section);
        }
        return true;
    }

    bool FSAILProgram::linkImpl(amd::option::Options *options) {
        acl_error errorCode;
        aclType continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;
        //If the binaryElf_ is not set then program must have been created
        // using clCreateProgramWithBinary
        if (!binaryElf_) {
            binary_t binary = this->binary();
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
                    buildLog_ += "Error while converting to BRIG: aclBinary init failure \n" ;
                    LogWarning("aclBinaryInit failed");
                    return false;
                }
                // Check that all needed section also exist in binaryElf_
                // No any validity checks here
                continueCompileFrom = getNextCompilationStageFromBinary();
                if (ACL_TYPE_DEFAULT == continueCompileFrom) {
                    return false;
                }
                if (ACL_TYPE_HSAIL_BINARY == continueCompileFrom) {
                    // Save binary in the interface class
                    // Also load compile & link options from binary into Program class members:
                    // compileOptions_ & linkOptions_
                    setBinary(static_cast<char*>(mem), binary.second);
                    // Compare options loaded from binary with current ones
                    // If they differ then recompile from ACL_TYPE_LLVMIR_BINARY
                    // @TODO It is needed to compare options taking into account that:
                    // 1. options are order independent;
                    // 2. (may be not trivial) compare only options that affect binary
                    std::string curOptions = options->origOptionStr + hsailOptions();
                    if (compileOptions_ + linkOptions_ != curOptions) {
                        continueCompileFrom = ACL_TYPE_LLVMIR_BINARY;
                    }
                }
            }
        }
        // Compilation from ACL_TYPE_LLVMIR_BINARY to ACL_TYPE_CG in cases:
        // 1. if the program is not created with binary;
        // 2. if the program is created with binary and contains only .llvmir & .comment
        // 3. if the program is created with binary, contains all brig sections,
        //    but the binary's compile & link options differ from current ones (recompilation);
        if (ACL_TYPE_LLVMIR_BINARY == continueCompileFrom) {
            std::string curOptions = options->origOptionStr + hsailOptions();
            errorCode = g_complibApi._aclCompile(device().compiler(),
                binaryElf_,
                curOptions.c_str(),
                ACL_TYPE_LLVMIR_BINARY,
                ACL_TYPE_CG,
                logFunction);
        }
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error while converting to BRIG: Compiling LLVMIR to BRIG \n" ;
            return false;
        }
        //Stop compilation if it is an offline device - HSA runtime does not
        //support ISA compiled offline
        if (!dev().isOnline()) {
            return true;
        }

        const HsaDevice *hsaDevice = dev().getBackendDevice();
        if (!loadBrig()) {
            buildLog_ += "Error while loading BRIG" ;
            return false;
        }

        size_t kernelNamesSize = 0;
        errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_KERNEL_NAMES, NULL, NULL, &kernelNamesSize);
        if (errorCode != ACL_SUCCESS) {
            buildLog_ += "Error while Finalization phase: kernel names query from the ELF failed\n";
            return false;
        }
        if (kernelNamesSize > 0) {
            char* kernelNames = new char[kernelNamesSize];
            errorCode = aclQueryInfo(dev().compiler(), binaryElf_, RT_KERNEL_NAMES, NULL, kernelNames, &kernelNamesSize);
            if (errorCode != ACL_SUCCESS) {
                buildLog_ += "Error while Finalization phase: kernel's Metadata is corrupted in the ELF\n";
                delete kernelNames;
                return false;
            }
            std::vector<std::string> vKernels = splitSpaceSeparatedString(kernelNames);
            delete kernelNames;
            std::vector<std::string>::iterator it = vKernels.begin();
            bool dynamicParallelism = false;
            for (it; it != vKernels.end(); ++it) {
                std::string kernelName = *it;
                Kernel *aKernel = new oclhsa::Kernel(kernelName,
                    this,
                    &brig_,
                    options->origOptionStr + hsailOptions());
                if (!aKernel->init() ) {
                    return false;
                }
                aKernel->setUniformWorkGroupSize(options->oVariables->UniformWorkGroupSize);
                // Update the binary in the FSAILProgram to save the ISA and debug information.
                // This is so the debugger and the profiler can use the a single aclBinary for all their needs.
                if (!updateAclBinaryWithKernelIsaAndDebug(kernelName)) {
                    return false;
                }
                kernels()[kernelName] = aKernel;
            }
        }
        saveBinaryAndSetType(TYPE_EXECUTABLE);
        buildLog_ += g_complibApi._aclGetCompilerLog(device().compiler());
        return true;
    }

    bool FSAILProgram::createBinary(amd::option::Options *options) {
        return false;
    }

    bool FSAILProgram::initClBinary() {
        if (clBinary_ == NULL) {
            clBinary_ = new ClBinary(static_cast<const Device &>(device()));
            if (clBinary_ == NULL) {
                return false;
            }
        }
        return true;
    }

    void FSAILProgram::releaseClBinary() {
        if (clBinary_ != NULL) {
            delete clBinary_;
            clBinary_ = NULL;
        }
    }

    std::string FSAILProgram::hsailOptions() {
        std::string hsailOptions;
        //Set options for the standard device specific options
        //This is just for legacy compiler code
        // All our devices support these options now
        hsailOptions.append(" -DFP_FAST_FMAF=1");
        hsailOptions.append(" -DFP_FAST_FMA=1");
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

#endif  // WITHOUT_FSA_BACKEND
}  // namespace hsa

