//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "rocbinary.hpp"
#include "roccompilerlib.hpp"
#include "acl.h"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "rocdevice.hpp"
#include "HSAILItems.h"

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

        //! Returns the aclBinary associated with the progrm
        const aclBinary* binaryElf() const {
            return static_cast<const aclBinary*>(binaryElf_); }

        const std::string& HsailText() {
            return hsailProgram_;
        }

        const NullDevice& dev() const { return device_; }
        //! Returns the hsaBinary associated with the progrm
        hsa_agent_t hsaDevice() const {
            return dev().getBackendDevice();
        }

    protected:
        //! log and append to build log an error from runtime
        void hsaError(const char *msg, hsa_status_t status = HSA_STATUS_SUCCESS);

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

        /*! \brief Compiles LLVM binary to HSAIL code (compiler backend: link+opt+codegen)
        *
        *  \return The build error code
        */
        int compileBinaryToHSAIL(
            amd::option::Options* options   //!< options for compilation
            );


        virtual bool linkImpl(amd::option::Options* options);

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
        //compiler library
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
    };

    /*@}*/} // namespace roc

#endif /*WITHOUT_HSA_BACKEND*/

