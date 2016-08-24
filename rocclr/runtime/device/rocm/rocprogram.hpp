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

        //! Returns the aclBinary associated with the program
        const aclBinary* binaryElf() const {
            return static_cast<const aclBinary*>(binaryElf_); }

#if defined(WITH_LIGHTNING_COMPILER)
        //! Returns the program metadata.
        const RuntimeMD::Program::Metadata* metadata() const { return metadata_; }
#endif // defined(WITH_LIGHTNING_COMPILER)

        //! Return a typecasted GPU device
        const NullDevice& dev() const
            { return static_cast<const NullDevice&>(device()); }

        //! Returns the hsaBinary associated with the program
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
        *  \return True if we successfully compiled a GPU program
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
#if defined(WITH_LIGHTNING_COMPILER)
        virtual bool linkImpl_LC(const std::vector<Program*>& inputPrograms,
            amd::option::Options* options,
            bool createLibrary);
#endif // defined(WITH_LIGHTNING_COMPILER)

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
        std::string hsailOptions(amd::option::Options* options);

        // aclBinary and aclCompiler - for the compiler library
        aclBinary*      binaryElf_; //!< Binary for the new compiler library
        aclBinaryOptions binOpts_; //!< Binary options to create aclBinary

        /* Brig and Brig modules */
        BrigModule_t brigModule_; //!< Brig that should be used in the HSA runtime
        BrigContainer* hsaBrigContainer_; //!< Container for the BRIG;
        hsa_ext_program_t hsaProgramHandle_; //!< Handle to HSA runtime program
        hsa_code_object_t hsaProgramCodeObject_; //!< Handle to HSA code object
        hsa_executable_t hsaExecutable_; //!< Handle to HSA executable

#if defined(WITH_LIGHTNING_COMPILER)
        RuntimeMD::Program::Metadata* metadata_; //!< Runtime metadata
#endif // defined(WITH_LIGHTNING_COMPILER)
    };

    /*@}*/} // namespace roc

#endif /*WITHOUT_HSA_BACKEND*/

