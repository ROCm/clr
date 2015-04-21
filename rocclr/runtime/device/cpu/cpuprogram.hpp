//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CPUPROGRAM_HPP_
#define CPUPROGRAM_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "device/cpu/cpubinary.hpp"
#include <string>

// forward declaration
namespace amd {
namespace option {
class Options;
} // option
} // amd

//! \namespace cpu CPU Device Implementation
namespace cpu {

//! \class CPU program
class Program : public device::Program
{
private:
    aclJITObjectImage JITBinary;
    std::string sourceFileName_; //!< The source image.
    void* handle_; // @todo: remove me

public:
    //! Default constructor
    Program(Device& cpuDev)
      : device::Program(cpuDev), JITBinary(NULL), handle_(NULL) {}

    //! Default destructor
    ~Program();

    //! pre-compile setup for CPU
    virtual bool initBuild(amd::option::Options* options);

    //! post-compile setup for CPU
    virtual bool finiBuild(bool isBuildGood);

    //! Compiles CPU program
    virtual bool compileImpl(
        const std::string& sourceCode,
        const std::vector<const std::string*>& headers,
        const char** headerIncludeNames,
        amd::option::Options* options );

    //! Links CPU program
    virtual bool linkImpl(amd::option::Options* options = NULL);

    //! Links CPU programs
    virtual bool linkImpl(
        const std::vector<device::Program*>& inputPrograms,
        amd::option::Options* options = NULL,
        bool createLibrary = false);

    virtual bool createBinary(amd::option::Options* options);

    //! Returns the device object, associated with this program.
    const Device& device() {
        return static_cast<const Device&>(device::Program::device());
    }

    /*! \brief Invokes the LLC compiler for the LLVM binary compilation
     *  to x86 ASM text source code and ISA binary
     *
     *  \return True if we successefully compiled a CPU program
     */
    bool compileBinaryToISA(
        amd::option::Options* options     //!< options for compilation
        );

    //! Load the library into memory
  bool loadDllCode(amd::option::Options* options, bool addElfSymbols=false);

    //! Initialize binary for CPU
    virtual bool initClBinary();

    //! Release binary for CPU
    virtual void releaseClBinary();

    ClBinary* clBinary() {
        return static_cast<ClBinary*>(device::Program::clBinary());
    }
    const ClBinary* clBinary() const {
        return static_cast<const ClBinary*>(device::Program::clBinary());
    }

    aclJITObjectImage getJITBinary() { return this->JITBinary; }
    void setJITBinary(aclJITObjectImage JITBinary) { this->JITBinary = JITBinary; }

    //! Returns the pointer to the Compiler struct
    //! Became public (prev. private) due to use in cpubinary for aclJIT functionality
    aclCompiler* compiler() { return static_cast<const Device&>(device()).compiler(); }

private:

    //! Disable default copy constructor
    Program(const Program&);

    //! Disable operator=
    Program& operator=(const Program&);

    std::string dllFileName_;   //!< File name of the dll with kernels
protected:
    virtual bool isElf(const char* bin) const {
        return amd::isElfHeader(bin, LP64_SWITCH(ELFCLASS32, ELFCLASS64));
    }

    virtual const aclTargetInfo & info(const char * str = "");
};

} // namespace cpu

#endif // CPUPROGRAM_HPP_
