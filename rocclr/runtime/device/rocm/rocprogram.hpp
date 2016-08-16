//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "rocbinary.hpp"
#if !defined(WITH_LIGHTNING_COMPILER)
#include "roccompilerlib.hpp"
#endif // !defined(WITH_LIGHTNING_COMPILER)
#include "acl.h"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "rocdevice.hpp"
#include "HSAILItems.h"

#if defined(WITH_LIGHTNING_COMPILER)
#include "rocmetadata.hpp"
#include "driver/AmdCompiler.h"
#endif // defined(WITH_LIGHTNING_COMPILER)

using namespace HSAIL_ASM;
//! \namespace roc HSA Device Implementation
namespace roc {

#if defined(WITH_LIGHTNING_COMPILER)
    class CodeObjBinary {
    public:
        CodeObjBinary()
          : target_(""), kernelArgAlign_(0), capFlags_(0), encryptCode_(0),
            binary_(NULL), binarySize_(0), llvmIR_(""), oclElf_(NULL), runtimeMD_(NULL) {}

        void init(std::string& target, void* binary, size_t binarySize);
        void fini();

        std::string Target() const { return target_; }
        uint32_t    KernelArgAlign() const { return kernelArgAlign_; }
        void*       Binary() const { return binary_; }
        size_t      BinarySize() const { return binarySize_; }

        void saveIR(std::string llvmIR) { llvmIR_ = llvmIR; }
        std::string   getLlvmIR() const { return llvmIR_; }

        amd::OclElf*  oclElf() const { return oclElf_; }
        RuntimeMD::Program::Metadata* runtimeMD() const { return runtimeMD_; }

        const RuntimeMD::Program::Metadata* GetProgramMetadata() const;

    private:
        enum CapFlag {
            capSaveSource   = 0,
            capSaveLLVMIR   = 1,
            capSaveCG       = 2,
            capSaveEXE      = 3,
            capSaveHSAIL    = 4,
            capSaveISASM    = 5,
            capEncryted     = 6
        };

        std::string target_;         // target device
        uint32_t    kernelArgAlign_;
        uint32_t    capFlags_;
        uint32_t    encryptCode_;

        void *      binary_;        //!< code object binary (ISA)
        size_t      binarySize_;    //!< size of the code object binary

        std::string llvmIR_;        //!< LLVM IR binary code

        amd::OclElf*  oclElf_;      //!< ELF object to access runtime metadata

        roc::RuntimeMD::Program::Metadata* runtimeMD_;   //!< runtime metadata
    };
#endif // defined(WITH_LIGHTNING_COMPILER)

    //! \class empty program
    class HSAILProgram : public device::Program
    {
        friend class ClBinary;
    public:
        //! Default constructor
        HSAILProgram(roc::NullDevice& device);
        //! Default destructor
        ~HSAILProgram();

        // Initialize Binary for GPU (used only for clCreateProgramWithBinary()).
        virtual bool initClBinary(char *binaryIn, size_t size);

        //! Returns the aclBinary associated with the progrm
        const aclBinary* binaryElf() const {
            return static_cast<const aclBinary*>(binaryElf_); }

#if defined(WITH_LIGHTNING_COMPILER)
        //! Returns the code object binary associated with the progrm
        const CodeObjBinary*  codeObjBinary() const {       //! Binary for the code object
            return static_cast<const CodeObjBinary*>(codeObjBinary_); }
#endif // defined(WITH_LIGHTNING_COMPILER)

        const std::string& HsailText() {
            return hsailProgram_;
        }

        const NullDevice& dev() const { return device_; }
        //! Returns the hsaBinary associated with the progrm
        hsa_agent_t hsaDevice() const {
            return dev().getBackendDevice();
        }

    protected:
        //! pre-compile setup for GPU
        virtual bool initBuild(amd::option::Options* options);

        //! post-compile setup for GPU
        virtual bool finiBuild(bool isBuildGood);

        /*! \brief Compiles GPU CL program to LLVM binary (compiler frontend)
        *
        *  \return True if we successefully compiled a GPU program
        */
        virtual bool compileImpl(
            const std::string& sourceCode,  //!< the program's source code
            const std::vector<const std::string*>& headers,
            const char** headerIncludeNames,
            amd::option::Options* options   //!< compile options's object
            );
#if defined(WITH_LIGHTNING_COMPILER)
        virtual bool compileImpl_LC(
            const std::string& sourceCode,  //!< the program's source code
            const std::vector<const std::string*>& headers,
            const char** headerIncludeNames,
            amd::option::Options* options   //!< compile options's object
            );
#endif // defined(WITH_LIGHTNING_COMPILER)

        /*! \brief Compiles LLVM binary to HSAIL code (compiler backend: link+opt+codegen)
        *
        *  \return The build error code
        */
        int compileBinaryToHSAIL(
            amd::option::Options* options   //!< options for compilation
            );


        virtual bool linkImpl(amd::option::Options* options);
#if defined(WITH_LIGHTNING_COMPILER)
        virtual bool linkImpl_LC(amd::option::Options* options);
#endif // defined(WITH_LIGHTNING_COMPILER)

        //! Link the device programs.
        virtual bool linkImpl (const std::vector<Program*>& inputPrograms,
            amd::option::Options* options,
            bool createLibrary);

        virtual bool createBinary(amd::option::Options* options);

        //! Initialize Binary
        virtual bool initClBinary();

        //! Release the Binary
        virtual void releaseClBinary();

        virtual const aclTargetInfo & info(const char * str = ""){
            return info_;
        }

        virtual bool isElf(const char* bin) const {
            return amd::isElfMagic(bin);
            //return false;
        }

        //! Returns the binary
        // This should ensure that the binary is updated with all the kernels
        //    ClBinary& clBinary() { return binary_; }
        ClBinary* clBinary() {
            return static_cast<ClBinary*>(device::Program::clBinary());
        }
        const ClBinary* clBinary() const {
            return static_cast<const ClBinary*>(device::Program::clBinary());
        }
    private:
        /* \brief Returns the next stage to compile from, based on sections in binary,
        *  also returns completeStages in a vector, which contains at least ACL_TYPE_DEFAULT,
        *  sets needOptionsCheck to true if options check is needed to decide whether or not to recompile
        */
        aclType getCompilationStagesFromBinary(std::vector<aclType>& completeStages, bool& needOptionsCheck);

        /* \brief Returns the next stage to compile from, based on sections and options in binary
        */
        aclType getNextCompilationStageFromBinary(amd::option::Options* options);
        bool saveBinaryAndSetType(type_t type);
        bool initBrigContainer();
        void destroyBrigContainer();
        //Initializes BRIG module
        bool initBrigModule();
        void destroyBrigModule();
        //! Disable default copy constructor
        HSAILProgram(const HSAILProgram&);

        //! Disable operator=
        HSAILProgram& operator=(const HSAILProgram&);

        //! Returns all the options to be appended while passing to the
        //compiler
        std::string hsailOptions();

        std::string     openCLSource_; //!< Original OpenCL source
        std::string     hsailProgram_;     //!< HSAIL program after compilation.
        std::string     llvmBinary_;    //!< LLVM IR binary code
        //!< aclBinary and aclCompiler - for the compiler libray
        aclBinary*      binaryElf_; //!<Binary for the new compiler library - shreyas edit
        aclBinaryOptions binOpts_; //!<Binary options to create aclBinary
        roc::NullDevice& device_; //!< Device related to the program
        /* Brig and Brig modules */
        BrigModule_t brigModule_; //!< Brig that should be used in the HSA runtime
        BrigContainer* hsaBrigContainer_; //!< Container for the BRIG;
        hsa_ext_program_t hsaProgramHandle_; //!< Handle to HSA runtime program
        hsa_code_object_t hsaProgramCodeObject_; //!< Handle to HSA code object
        hsa_executable_t hsaExecutable_; //!< Handle to HSA executable

#if defined(WITH_LIGHTNING_COMPILER)
        CodeObjBinary*    codeObjBinary_;       //! Binary for the code object
#endif // defined(WITH_LIGHTNING_COMPILER)
    };

    /*@}*/} // namespace roc

#endif /*WITHOUT_HSA_BACKEND*/

