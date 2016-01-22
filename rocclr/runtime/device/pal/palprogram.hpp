//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef PALPROGRAM_HPP_
#define PALPROGRAM_HPP_

#include "device/pal/palkernel.hpp"
#include "device/pal/palbinary.hpp"
#include "amd_hsa_loader.hpp"

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
class ClBinaryHsa;

class ORCAHSALoaderContext final: public Context {
public:
    ORCAHSALoaderContext(HSAILProgram* program): program_(program) {}

    virtual ~ORCAHSALoaderContext() {}

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

    bool SegmentFreeze(amdgpu_hsa_elf_segment_t segment,
        hsa_agent_t agent, void* seg, size_t size) override { return false; }

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

    void* AgentGlobalAlloc(
        hsa_agent_t agent, size_t size, size_t align, bool zero) {
        return GpuMemAlloc(size, align, zero);
    }

    bool AgentGlobalCopy(void *dst, size_t offset, const void *src, size_t size) {
        return GpuMemCopy(dst, offset, src, size);
    }

    void AgentGlobalFree(void *ptr, size_t size) {
        GpuMemFree(ptr, size);
    }

    void* KernelCodeAlloc(
        hsa_agent_t agent, size_t size, size_t align, bool zero) {
        return CpuMemAlloc(size, align, zero);
    }

    bool KernelCodeCopy(void *dst, size_t offset, const void *src, size_t size) {
        return CpuMemCopy(dst, offset, src, size);
    }

    void KernelCodeFree(void *ptr, size_t size) {
        CpuMemFree(ptr, size);
    }

    void* CpuMemAlloc(size_t size, size_t align, bool zero);

    bool CpuMemCopy(void *dst, size_t offset, const void* src, size_t size);

    void CpuMemFree(void *ptr, size_t size) {
        amd::Os::alignedFree(ptr);
    }

    void* GpuMemAlloc(size_t size, size_t align, bool zero);

    bool GpuMemCopy(void *dst, size_t offset, const void *src, size_t size);

    void GpuMemFree(void *ptr, size_t size = 0) {
        delete reinterpret_cast<pal::Memory*>(ptr);
    }

    ORCAHSALoaderContext(const ORCAHSALoaderContext &c);

    ORCAHSALoaderContext& operator=(const ORCAHSALoaderContext &c);

    enum gfx_handle {
        gfx700 = 700,
        gfx701 = 701,
        gfx702 = 702,
        gfx800 = 800,
        gfx801 = 801,
        gfx804 = 804,
        gfx810 = 810,
        gfx900 = 900,
        gfx901 = 901
    };

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
    ~HSAILProgram();

    //! Returns the aclBinary associated with the progrm
    aclBinary* binaryElf() const {
        return static_cast<aclBinary*>(binaryElf_); }

    void addGlobalStore(Memory* mem) { globalStores_.push_back(mem); }

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

    /*! \brief Compiles LLVM binary to FSAIL code (compiler backend: link+opt+codegen)
    *
    *  \return The build error code
    */
    int compileBinaryToFSAIL(
        amd::option::Options* options   //!< options for compilation
        );

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
    std::vector<Memory*>         globalStores_;   //!< Global memory for the program
    Memory*         kernels_;       //!< Table with kernel object pointers
    uint    maxScratchRegs_;    //!< Maximum number of scratch regs used in the program by individual kernel
    std::list<Sampler*>   staticSamplers_;    //!< List od internal static samplers
    bool            isNull_;        //!< Null program no memory allocations
    amd::hsa::loader::Loader* loader_; //!< Loader object
    amd::hsa::loader::Executable* executable_;    //!< Executable for HSA Loader
    ORCAHSALoaderContext loaderContext_;    //!< Context for HSA Loader
};

/*@}*/} // namespace pal

#endif /*PALPROGRAM_HPP_*/
