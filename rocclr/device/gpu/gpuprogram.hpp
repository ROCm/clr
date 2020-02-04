/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef GPUPROGRAM_HPP_
#define GPUPROGRAM_HPP_

#include "device/gpu/gpukernel.hpp"
#include "device/gpu/gpubinary.hpp"
#include "amd_hsa_loader.hpp"

namespace amd {
namespace option {
class Options;
}  // option
namespace hsa {
namespace loader {
class Loader;
class Executable;
class Context;
}  // loader
}  // hsa
}  // amd

//! \namespace gpu GPU Device Implementation
namespace gpu {

/*! \addtogroup GPU GPU Device Implementation
 *  @{
 */

//! \struct ILFunc for the opencl program processing
struct ILFunc : public amd::HeapObject {
 public:
  //! \struct CodeRange for the code ranges
  struct SourceRange : public amd::EmbeddedObject {
    size_t begin_;  //!< start code position
    size_t end_;    //!< end code position
  };

  //! \enum IL function state
  enum State {
    Unknown = 0x00000000,  //! unknown function
    Regular = 0x00000001,  //! regular function from the program
    Kernel = 0x00000002    //! kernel function from the program
  };

  //! Default constructor
  ILFunc()
      : name_(""),
        index_(0),
        state_(Unknown),
        privateSize_(0),
        localSize_(0),
        hwPrivateSize_(0),
        hwLocalSize_(0),
        flags_(0),
        totalHwPrivateSize_(-1) {
    code_.begin_ = code_.end_ = 0;
    metadata_.begin_ = metadata_.end_ = 0;
  }

  //! Copy constructor
  ILFunc(const ILFunc& func) { *this = func; }

  //! Destructor
  ~ILFunc() {}

  //! Overloads operator=
  ILFunc& operator=(const ILFunc& func) {
    name_ = func.name_;
    index_ = func.index_;
    code_ = func.code_;
    metadata_ = func.metadata_;
    state_ = func.state_;
    privateSize_ = func.privateSize_;
    localSize_ = func.localSize_;
    hwPrivateSize_ = func.hwPrivateSize_;
    hwLocalSize_ = func.hwLocalSize_;
    flags_ = func.flags_;
    totalHwPrivateSize_ = func.totalHwPrivateSize_;

    // Note: we don't copy calls_ and macros_
    return *this;
  }

  std::string name_;              //!< kernel's name
  uint index_;                    //!< kernel's index
  SourceRange code_;              //!< the entire function range in the source
  SourceRange metadata_;          //!< the metadata range
  State state_;                   //!< the function is real, and not intrinsic
  uint privateSize_;              //!< private ring allocation by the function
  uint localSize_;                //!< local ring allocation by the function
  uint hwPrivateSize_;            //!< HW private ring allocation by the function
  uint hwLocalSize_;              //!< HW local ring allocation by the function
  uint flags_;                    //!< The IL func flags/properties
  long long totalHwPrivateSize_;  //!< total HW private usage including called functions
  std::vector<ILFunc*> calls_;    //! Functions called from the current
  std::vector<uint> macros_;      //! Macros, used in the IL function

  uint totalHwPrivateUsage();  //!< total HW private usage including called functions
};

//! \class empty program
class NullProgram : public device::Program {
  friend class ClBinary;

 public:
  //! Default constructor
  NullProgram(NullDevice& nullDev, amd::Program& owner)
    : device::Program(nullDev, owner), patch_(0) {}

  //! Default destructor
  ~NullProgram();

  // Initialize Binary for GPU
  virtual bool initClBinary();

  //! Returns global constant buffers
  const std::vector<uint>& glbCb() const { return glbCb_; }

 protected:
  /*! \brief Compiles GPU CL program to LLVM binary (compiler frontend)
   *
   *  \return True if we successefully compiled a GPU program
   */
  virtual bool compileImpl(const std::string& sourceCode,  //!< the program's source code
                           const std::vector<const std::string*>& headers,  //!< header souce codes
                           const char** headerIncludeNames,  //!< include names of headers
                           amd::option::Options* options     //!< compile options's object
                           );

  /*! \brief Compiles LLVM binary to IL code (compiler backend: link+opt+codegen)
   *
   *  \return The build error code
   */
  int compileBinaryToIL(amd::option::Options* options  //!< options for compilation
                        );

  /*! \brief Links the compiled IL program with HW
   *
   *  \return True if we successefully linked a GPU program
   */
  virtual bool linkImpl(amd::option::Options* options = NULL  //!< options object
                        );
  virtual bool linkImpl(const std::vector<device::Program*>& inputPrograms,
                        amd::option::Options* options = NULL,  //!< options object
                        bool createLibrary = false);

  virtual bool createBinary(amd::option::Options* options);


  /*! \brief Parses the GPU program and finds all available kernels
   *
   *  \return True if we successefully parsed the GPU program
   */
  bool parseKernels(const std::string& source  //! the program's source code
                    );

  /*! \brief Parse all functions in the program
   *
   *  \return True if we successefully parsed all functions
   */
  bool parseAllILFuncs(const std::string& source  //! the program's source code
                       );

  /*! \brief Parse a function's metadata given as source[posBegin:posEnd-1]
   *
   *  \return True if we successefully parsed the given metadata
   */
  bool parseFuncMetadata(const std::string& source,  //! string that contains metadata
                         size_t posBegin,            //! begin of metadata in 'source'
                         size_t posEnd               //! end of metadata in 'source'
                         );

  /*! \brief Finds functions with the given start and end string in the
   * program
   *
   *  \return True if we successefully found all functions
   */
  bool findILFuncs(const std::string& source,      //! the program's source code
                   const std::string& func_start,  //! the start string of a function
                   const std::string& func_end,    //! the end string of a function
                   size_t& lastFuncPos             //! pos to the end of the last func in 'source'
                   );


  /*! \brief Finds all functions in the program
   *
   *  \return True if we successefully found all functions
   */
  bool findAllILFuncs(const std::string& source,  //! the program's source code
                      size_t& lastFuncPos         //! pos to the end of the last func in 'source'
                      );

  /*! \brief Finds function, corresponded to the provided unique index
   *
   *  \return Pointer to the ILFunc structure
   */
  ILFunc* findILFunc(uint index  //! the function unique index
                     );

  //! Destroys all objects, associated with the IL functions
  void freeAllILFuncs();

  /*! \brief Finds if a provided function is called from the base function
   *
   *  \return True if a function is used from the base one
   */
  bool isCalled(const ILFunc* base,  //!< The base function
                const ILFunc* func   //!< Function to check for usage
                );

  //! Patches the "main" function with the call to the current kernel
  void patchMain(std::string& kernel,  //! The current kernel's code for compilation
                 uint index            //! Index of the current kernel in the program
                 );

  //! Adds the IL function object into the list of functions
  void addFunc(ILFunc* func) { funcs_.push_back(func); }

  //! Empty implementation, since we don't have real HW
  virtual bool allocGlobalData(const void* globalData,  //!< Pointer to the global data
                               size_t dataSize,         //!< The global data size
                               uint index  //!< Index for the global data store (0 - global heap)
                               ) {
    glbCb_.push_back(index);
    return true;
  }

  //! Load binary for offline device.
  virtual bool loadBinary(bool* hasRecompiled);

  //! Create NullKernel for compiling to isa.
  virtual NullKernel* createKernel(const std::string& name,           //!< The kernel's name
                                   const Kernel::InitData* initData,  //!< Initialization data
                                   const std::string& code,           //!< IL source code
                                   const std::string& metadata,  //!< the kernel metadata structure
                                   bool* created,                //!< True if the object was created
                                   const void* binaryCode = NULL,  //!< binary machine code for CAL
                                   size_t binarySize = 0           //!< the machine code size
                                   );

  ClBinary* clBinary() { return static_cast<ClBinary*>(device::Program::clBinary()); }
  const ClBinary* clBinary() const {
    return static_cast<const ClBinary*>(device::Program::clBinary());
  }

  /*! Get all per-kernel IL from programIL, where programIL is the IL for the
   *  whole compilation unit.
   */
  bool getAllKernelILs(std::unordered_map<std::string, std::string>& allKernelILs, std::string& programIL,
                       const char* ilKernelName);

 protected:
  std::vector<device::PrintfInfo> printf_;  //!< Format strings for GPU printf support
  std::vector<uint> glbCb_;         //!< Global constant buffers

  virtual const aclTargetInfo& info(const char* str = "");

  virtual bool saveBinaryAndSetType(type_t type) { return true; }

 private:
  //! Disable default copy constructor
  NullProgram(const NullProgram&);

  //! Disable operator=
  NullProgram& operator=(const NullProgram&);

  //! Initializes the global data store
  bool initGlobalData(const std::string& source,  //!< the program's source code
                      size_t start                //!< start position for the global data search
                      );

  //! Return a typecasted GPU device
  gpu::NullDevice& dev() {
    return const_cast<gpu::NullDevice&>(static_cast<const gpu::NullDevice&>(device()));
  }

  size_t patch_;                //!< Patch call position in the source code.
  std::vector<ILFunc*> funcs_;  //!< list of all functions.

  std::string ilProgram_;  //!< IL program after compilation
};

//! \class GPU program
class Program : public NullProgram {
 public:
  //! GPU program constructor
  Program(Device& gpuDev, amd::Program& owner) : NullProgram(gpuDev, owner), glbData_(NULL) {}

  //! GPU program destructor
  ~Program();

  //! Get the global data store for this program
  gpu::Memory* glbData() const { return glbData_; }

  //! Returns TRUE if we successfully allocated the global data store
  //! in video memory
  bool allocGlobalData(const void* globalData,  //!< Pointer to the global data
                       size_t dataSize,         //!< The global data size
                       uint index  //!< Index for the global data store (0 - global heap)
                       );

  //! Returns TRUE if we could
  virtual bool loadBinary(bool* hasRecompiled);

  //! Creates the GPU kernel (return base type)
  virtual NullKernel* createKernel(const std::string& name,           //!< The kernel's name
                                   const Kernel::InitData* initData,  //!< Initialization data
                                   const std::string& code,           //!< IL source code
                                   const std::string& metadata,  //!< the kernel metadata structure
                                   bool* created,                //!< True if the object was created
                                   const void* binaryCode = NULL,  //!< binary machine code for CAL
                                   size_t binarySize = 0           //!< the machine code size
                                   );

  typedef std::unordered_map<uint, gpu::Memory*> HwConstBuffers;

  //! Global HW constant buffers
  const HwConstBuffers& glbHwCb() const { return constBufs_; }

  //! Returns pritnf info array
  const std::vector<device::PrintfInfo>& printfInfo() const { return printf_; }

  //! Return a typecasted GPU device
  gpu::Device& dev() { return const_cast<gpu::Device&>(static_cast<const gpu::Device&>(device())); }

 protected:
 private:
  //! Disable copy constructor
  Program(const Program&);

  //! Disable operator=
  Program& operator=(const Program&);

  HwConstBuffers constBufs_;  //!< Constant buffers for the global store
  gpu::Memory* glbData_;      //!< Global data store
};

using namespace amd::hsa::loader;
class HSAILProgram;

class ORCAHSALoaderContext final : public Context {
 public:
  ORCAHSALoaderContext(HSAILProgram* program) : program_(program) {}

  virtual ~ORCAHSALoaderContext() {}

  hsa_isa_t IsaFromName(const char* name) override;

  bool IsaSupportedByAgent(hsa_agent_t agent, hsa_isa_t isa) override;

  void* SegmentAlloc(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, size_t size, size_t align,
                     bool zero) override;

  bool SegmentCopy(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* dst, size_t offset,
                   const void* src, size_t size) override;

  void SegmentFree(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                   size_t size = 0) override;

  void* SegmentAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                       size_t offset) override;

  void* SegmentHostAddress(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                           size_t offset) override {
    return nullptr;
  }

  bool SegmentFreeze(amdgpu_hsa_elf_segment_t segment, hsa_agent_t agent, void* seg,
                     size_t size) override {
    return false;
  }

  bool ImageExtensionSupported() override { return false; }

  hsa_status_t ImageCreate(hsa_agent_t agent, hsa_access_permission_t image_permission,
                           const hsa_ext_image_descriptor_t* image_descriptor,
                           const void* image_data, hsa_ext_image_t* image_handle) override {
    // not supported
    assert(false);
    return HSA_STATUS_ERROR;
  }

  hsa_status_t ImageDestroy(hsa_agent_t agent, hsa_ext_image_t image_handle) override {
    // not supported
    assert(false);
    return HSA_STATUS_ERROR;
  }

  hsa_status_t SamplerCreate(hsa_agent_t agent,
                             const hsa_ext_sampler_descriptor_t* sampler_descriptor,
                             hsa_ext_sampler_t* sampler_handle) override;

  //! All samplers are owned by HSAILProgram and are deleted in its destructor.
  hsa_status_t SamplerDestroy(hsa_agent_t agent, hsa_ext_sampler_t sampler_handle) override;

 private:
  void* AgentGlobalAlloc(hsa_agent_t agent, size_t size, size_t align, bool zero) {
    return GpuMemAlloc(size, align, zero);
  }

  bool AgentGlobalCopy(void* dst, size_t offset, const void* src, size_t size) {
    return GpuMemCopy(dst, offset, src, size);
  }

  void AgentGlobalFree(void* ptr, size_t size) { GpuMemFree(ptr, size); }

  void* KernelCodeAlloc(hsa_agent_t agent, size_t size, size_t align, bool zero) {
    return CpuMemAlloc(size, align, zero);
  }

  bool KernelCodeCopy(void* dst, size_t offset, const void* src, size_t size) {
    return CpuMemCopy(dst, offset, src, size);
  }

  void KernelCodeFree(void* ptr, size_t size) { CpuMemFree(ptr, size); }

  void* CpuMemAlloc(size_t size, size_t align, bool zero);

  bool CpuMemCopy(void* dst, size_t offset, const void* src, size_t size);

  void CpuMemFree(void* ptr, size_t size) { amd::Os::alignedFree(ptr); }

  void* GpuMemAlloc(size_t size, size_t align, bool zero);

  bool GpuMemCopy(void* dst, size_t offset, const void* src, size_t size);

  void GpuMemFree(void* ptr, size_t size = 0);

  ORCAHSALoaderContext(const ORCAHSALoaderContext& c);

  ORCAHSALoaderContext& operator=(const ORCAHSALoaderContext& c);

  gpu::HSAILProgram* program_;
};

//! \class HSAIL program
class HSAILProgram : public device::Program {
  friend class ClBinary;

 public:
  //! Default constructor
  HSAILProgram(Device& device, amd::Program& owner);
  HSAILProgram(NullDevice& device, amd::Program& owner);
  //! Default destructor
  ~HSAILProgram();

  void addGlobalStore(Memory* mem) { globalStores_.push_back(mem); }

  const std::vector<Memory*>& globalStores() const { return globalStores_; }

  //! Return a typecasted GPU device
  gpu::Device& dev() { return const_cast<gpu::Device&>(static_cast<const gpu::Device&>(device())); }

  //! Returns GPU kernel table
  const Memory* kernelTable() const { return kernels_; }

  //! Adds all kernels to the mem handle lists
  void fillResListWithKernels(std::vector<const Memory*>& memList) const;

  //! Returns the maximum number of scratch regs used in the program
  uint maxScratchRegs() const { return maxScratchRegs_; }

  //! Add internal static sampler
  void addSampler(Sampler* sampler) { staticSamplers_.push_back(sampler); }

  //! Returns TRUE if the program contains static samplers
  bool isStaticSampler() const { return (staticSamplers_.size() != 0); }

 protected:
  bool saveBinaryAndSetType(type_t type);

  virtual bool linkImpl(amd::option::Options* options);

  virtual bool createBinary(amd::option::Options* options);

  virtual const aclTargetInfo& info(const char* str = "");

 private:
  //! Disable default copy constructor
  HSAILProgram(const HSAILProgram&);

  //! Disable operator=
  HSAILProgram& operator=(const HSAILProgram&);

  //! Returns all the options to be appended while passing to the
  // compiler library
  std::string hsailOptions();

  //! Allocate kernel table
  bool allocKernelTable();

  void* rawBinary_;                    //!< Pointer to the raw binary
  std::vector<Memory*> globalStores_;  //!< Global memory for the program
  Memory* kernels_;                    //!< Table with kernel object pointers
  uint
      maxScratchRegs_;  //!< Maximum number of scratch regs used in the program by individual kernel
  std::list<Sampler*> staticSamplers_;        //!< List od internal static samplers
  amd::hsa::loader::Loader* loader_;          //!< Loader object
  amd::hsa::loader::Executable* executable_;  //!< Executable for HSA Loader
  ORCAHSALoaderContext loaderContext_;        //!< Context for HSA Loader
};

/*@}*/} // namespace gpu

#endif /*GPUPROGRAM_HPP_*/
