//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef HSAPROGRAM_HPP_
#define HSAPROGRAM_HPP_

#ifndef WITHOUT_FSA_BACKEND

#include "hsabinary.hpp"
#include "hsacompilerlib.hpp"
#include "services.h"
#include "acl.h"
#include "oclhsa_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "hsadevice.hpp"

//! \namespace oclhsa HSA Device Implementation
namespace oclhsa {

    //! \class empty program
    class FSAILProgram : public device::Program
    {
        friend class ClBinary;
    public:
        //! Default constructor
        FSAILProgram(oclhsa::NullDevice& device);
        //! Default destructor
        ~FSAILProgram();

        // Initialize Binary for GPU (used only for clCreateProgramWithBinary()).
        virtual bool initClBinary(char *binaryIn, size_t size);

        //! Returns the aclBinary associated with the progrm
        const aclBinary* binaryElf() const {
            return static_cast<const aclBinary*>(binaryElf_); }

        //! Returns the brig associated with the progrm
        const HsaBrig* brig() {
            return static_cast<const HsaBrig*>(&brig_); }

        const NullDevice& dev() const { return device_; }
        //! Returns the hsaBinary associated with the progrm
        const HsaDevice* hsaDevice() const {
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

        /*! \brief Compiles LLVM binary to FSAIL code (compiler backend: link+opt+codegen)
        *
        *  \return The build error code
        */
        int compileBinaryToFSAIL(
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

        //! Extracts a symbol from the binaryElf_
        //  and copies it to a buffer allocated
        //  by the function
        bool ExtractSymbolAndCopy(aclSections id,
            const char *symbol_name,
            void** address_to_copy,
            size_t* symbol_size_bytes,
            bool verify);
        //! Extracts the aclBinary used internally within the brig
        // and pulls the debug and ISA section for a particular kernel
        // and inserts it into aclBinary contained in the program
        bool updateAclBinaryWithKernelIsaAndDebug(std::string kernelName);
        //! Checks the existence of sections in binaryElf_
        // and calculates the next stage of compilation;
        // if set of the section is impossible, then
        // binary is invalid and function returns ACL_TYPE_DEFAULT
        aclType getNextCompilationStageFromBinary();
        //! Loads the global variables for the BRIG
        bool loadBrig();
        //! Unloads the global variables for the BRIG
        bool unloadBrig();
        bool saveBinaryAndSetType(type_t type);
        //! Disable default copy constructor
        FSAILProgram(const FSAILProgram&);

        //! Disable operator=
        FSAILProgram& operator=(const FSAILProgram&);

        //! Returns all the options to be appended while passing to the
        //compiler library
        std::string hsailOptions();

        std::string     openCLSource_; //!< Original OpenCL source
        std::string     fsailProgram_;     //!< FSAIL program after compilation.
        std::string     llvmBinary_;    //!< LLVM IR binary code
        //!< aclBinary and aclCompiler - for the compiler libray
        aclBinary*      binaryElf_; //!<Binary for the new compiler library - shreyas edit
        aclBinaryOptions binOpts_; //!<Binary options to create aclBinary
        oclhsa::NullDevice& device_; //!< Device related to the program
        HsaBrig brig_; //!< Brig for the program
        bool isBrigLoaded_; //!< Boolean to verify is the Brig has been loaded
    };

    /*@}*/} // namespace oclhsa

#endif /*WITHOUT_FSA_BACKEND*/
#endif /* HSAPROGRAM_HPP_*/
