//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "device/pal/palkernel.hpp"
#include "device/pal/palbinary.hpp"
#include "amd_hsa_loader.hpp"

#if defined(WITH_LIGHTNING_COMPILER)
#include "libamdhsacode/amdgpu_metadata.hpp"
#endif // defined(WITH_LIGHTNING_COMPILER)

namespace amd {
namespace option {
class Options;
} // option
namespace hsa {
namespace loader {
class Loader;
class Executable;
class Context;
} // loader
} // hsa
} // amd

//! \namespace pal PAL Device Implementation
namespace pal {

/*! \addtogroup pal PAL Device Implementation
 *  @{
 */

using namespace amd::hsa::loader;
class HSAILProgram;

class Segment : public amd::HeapObject {
public:
    Segment();
    ~Segment();

    //! Allocates a segment
    bool alloc(HSAILProgram& prog, amdgpu_hsa_elf_segment_t segment,
        size_t size, size_t align, bool zero);

    //! Copies data from host to the segment
    void copy(size_t offset, const void* src, size_t size);

    //! Segment freeze
    bool freeze(bool destroySysmem);

    //! Returns address for GPU access in the segment
    uint64_t    gpuAddress(size_t offset) const { return gpuAccess_->vmAddress() + offset; }

    //! Returns address for CPU access in the segment
    void*       cpuAddress(size_t offset) const { return cpuAccess_->data() + offset; }

private:
    Memory*     gpuAccess_;     //!< GPU memory for segment access
    Memory*     cpuAccess_;     //!< CPU memory for segment (backing store)
};

class PALHSALoaderContext final: public Context {
public:
    PALHSALoaderContext(HSAILProgram* program): program_(program) {}

    virtual ~PALHSALoaderContext() {}

    hsa_isa_t IsaFromName(const char *name) override;

    bool IsaSupportedByAgent(hsa_agent_t agent, hsa_isa_t isa) override;

    void* SegmentAlloc(amdgpu_hsa_elf_segment_t segment,
        hsa_agent_t agent, size_t size, size_t align, bool zero) override;

    bool SegmentCopy(amdgpu_hsa_elf_segment_t segment,
        hsa_agent_t agent, void* dst, size_t offset,
        const void* src, size_t size) override;

    void SegmentFree(amdgpu_hsa_elf_segment_t segment,
        hsa_agent_t agent, void* seg, size_t size = 0) override;

    void* SegmentAddress(amdgpu_hsa_elf_segment_t segment,
        hsa_agent_t agent, void* seg, size_t offset) override;

    void* SegmentHostAddress(amdgpu_hsa_elf_segment_t segment,
        hsa_agent_t agent, void* seg, size_t offset) override;

    bool SegmentFreeze(amdgpu_hsa_elf_segment_t segment,
        hsa_agent_t agent, void* seg, size_t size) override;

    bool ImageExtensionSupported() override { return false; }

    hsa_status_t ImageCreate(
        hsa_agent_t agent,
        hsa_access_permission_t image_permission,
        const hsa_ext_image_descriptor_t *image_descriptor,
        const void *image_data,
        hsa_ext_image_t *image_handle) override {
        // not supported
        assert(false);
        return HSA_STATUS_ERROR;
    }

    hsa_status_t ImageDestroy(
        hsa_agent_t agent, hsa_ext_image_t image_handle) override {
        // not supported
        assert(false);
        return HSA_STATUS_ERROR;
    }

    hsa_status_t SamplerCreate(
        hsa_agent_t agent,
        const hsa_ext_sampler_descriptor_t *sampler_descriptor,
        hsa_ext_sampler_t *sampler_handle) override;

    //! All samplers are owned by HSAILProgram and are deleted in its destructor.
    hsa_status_t SamplerDestroy(
        hsa_agent_t agent, hsa_ext_sampler_t sampler_handle) override;

private:
    PALHSALoaderContext(const PALHSALoaderContext &c);
    PALHSALoaderContext& operator=(const PALHSALoaderContext &c);

    pal::HSAILProgram* program_;
};

//! \class HSAIL program
class HSAILProgram : public device::Program
{
    friend class ClBinary;
public:
    //! Default constructor
    HSAILProgram(Device& device);
    HSAILProgram(NullDevice& device);
    //! Default destructor
    virtual ~HSAILProgram();

    //! Returns the aclBinary associated with the progrm
    aclBinary* binaryElf() const {
        return static_cast<aclBinary*>(binaryElf_); }

    void addGlobalStore(Memory* mem) { globalStores_.push_back(mem); }

    void setCodeObjects(Memory* codeGpu, address codeCpu)
        { codeSegGpu_ = codeGpu; codeSegCpu_ = codeCpu; }

    const std::vector<Memory*>& globalStores() const { return globalStores_; }

    //! Return a typecasted GPU device
    pal::Device& dev()
        { return const_cast<pal::Device&>(
            static_cast<const pal::Device&>(device())); }

    //! Returns GPU kernel table
    const Memory* kernelTable() const { return kernels_; }

    //! Adds all kernels to the mem handle lists
    void fillResListWithKernels(std::vector<const Memory*>& memList) const;

    //! Returns the maximum number of scratch regs used in the program
    uint    maxScratchRegs() const { return maxScratchRegs_; }

    //! Add internal static sampler
    void addSampler(Sampler* sampler) { staticSamplers_.push_back(sampler); }

    //! Returns TRUE if the program just compiled
    bool isNull() const { return isNull_; }

    //! Returns TRUE if the program used internally by runtime
    bool isInternal() const { return internal_; }

    //! Returns TRUE if the program contains static samplers
    bool isStaticSampler() const { return (staticSamplers_.size() != 0); }

    //! Returns code segement on GPU
    const Memory& codeSegGpu()  const { return *codeSegGpu_; }

    //! Returns code segement on CPU
    address codeSegCpu() const { return codeSegCpu_; }

    //! Returns CPU address for a kernel
    uint64_t findHostKernelAddress(uint64_t devAddr) const
    {
        return loader_->FindHostAddress(devAddr);
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

    /* \brief Returns the next stage to compile from, based on sections in binary,
    *  also returns completeStages in a vector, which contains at least ACL_TYPE_DEFAULT,
    *  sets needOptionsCheck to true if options check is needed to decide whether or not to recompile
    */
    aclType getCompilationStagesFromBinary(std::vector<aclType>& completeStages, bool& needOptionsCheck);

    /* \brief Returns the next stage to compile from, based on sections and options in binary
    */
    aclType getNextCompilationStageFromBinary(amd::option::Options* options);
    
    bool saveBinaryAndSetType(type_t type);

    virtual bool linkImpl(amd::option::Options* options);

    //! Link the device programs.
    virtual bool linkImpl (const std::vector<device::Program*>& inputPrograms,
        amd::option::Options* options,
        bool createLibrary);

    virtual bool createBinary(amd::option::Options* options);

    //! Initialize Binary
    virtual bool initClBinary();

    //! Release the Binary
    virtual void releaseClBinary();

    virtual const aclTargetInfo & info(const char * str = "");

    virtual bool isElf(const char* bin) const {
        return amd::isElfMagic(bin);
        //return false;
    }

    //! Returns the binary
    // This should ensure that the binary is updated with all the kernels
    //    ClBinary& clBinary() { return binary_; }
    ClBinaryHsa* clBinary() {
        return static_cast<ClBinaryHsa*>(device::Program::clBinary());
    }
    const ClBinaryHsa* clBinary() const {
        return static_cast<const ClBinaryHsa*>(device::Program::clBinary());
    }

private:
    //! Disable default copy constructor
    HSAILProgram(const HSAILProgram&);

    //! Disable operator=
    HSAILProgram& operator=(const HSAILProgram&);

protected:
    //! Returns all the options to be appended while passing to the
    //compiler library
    std::string hsailOptions();

    //! Allocate kernel table
    bool allocKernelTable();

    std::string     openCLSource_;  //!< Original OpenCL source
    std::string     HSAILProgram_;  //!< FSAIL program after compilation
    std::string     llvmBinary_;    //!< LLVM IR binary code
    aclBinary*      binaryElf_;     //!< Binary for the new compiler library
    void*           rawBinary_;     //!< Pointer to the raw binary
    aclBinaryOptions binOpts_;      //!< Binary options to create aclBinary
    std::vector<Memory*>    globalStores_;   //!< Global memory for the program
    Memory*         kernels_;       //!< Table with kernel object pointers
    Memory*         codeSegGpu_;    //!< GPU memory with code objects
    address         codeSegCpu_;    //!< CPU memory with code objects
    uint    maxScratchRegs_;    //!< Maximum number of scratch regs used in the program by individual kernel
    std::list<Sampler*>     staticSamplers_;    //!< List od internal static samplers
    union {
        struct {
            uint32_t    isNull_     : 1;    //!< Null program no memory allocations
            uint32_t    internal_   : 1;    //!< Internal blit program
        };
        uint32_t    flags_;  //!< Program flags
    };
    amd::hsa::loader::Loader* loader_; //!< Loader object
    amd::hsa::loader::Executable* executable_;    //!< Executable for HSA Loader
    PALHSALoaderContext loaderContext_;    //!< Context for HSA Loader
};

#if defined(WITH_LIGHTNING_COMPILER)
//! \class Lightning Compiler Program
class LightningProgram : public HSAILProgram
{
public:
    LightningProgram(NullDevice& device)
        : HSAILProgram(device),
          metadata_(nullptr)
    {}

    const amd::hsa::code::Program::Metadata* metadata() const {
        return metadata_;
    }
private:
    virtual ~LightningProgram();

protected:
    virtual bool compileImpl(
        const std::string& sourceCode,  //!< the program's source code
        const std::vector<const std::string*>& headers,
        const char** headerIncludeNames,
        amd::option::Options* options   //!< compile options's object
    ) override;

    virtual bool linkImpl(amd::option::Options* options) override;

    bool setKernels(amd::option::Options *options, void* binary, size_t size);

    //! Return a new transient compiler instance.
    static std::auto_ptr<amd::opencl_driver::Compiler> newCompilerInstance();

private:
    amd::hsa::code::Program::Metadata* metadata_; //!< Runtime metadata
};
#endif // defined(WITH_LIGHTNING_COMPILER)

/*@}*/} // namespace pal
