/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#ifndef FLAGS_HPP_
#define FLAGS_HPP_


#define RUNTIME_FLAGS(debug,release,release_on_stg)                           \
                                                                              \
release(int, LOG_LEVEL, 0,                                                    \
        "The default log level")                                              \
release(uint, GPU_LOG_MASK, 0X7FFFFFFF,                                       \
        "The mask to enable specific kinds of logs")                          \
debug(uint, DEBUG_GPU_FLAGS, 0,                                               \
        "The debug options for GPU device")                                   \
release(uint, GPU_MAX_COMMAND_QUEUES, 300,                                    \
        "The maximum number of concurrent Virtual GPUs")                      \
release(size_t, CQ_THREAD_STACK_SIZE, 256*Ki, /* @todo: that much! */         \
        "The default command queue thread stack size")                        \
release(int, GPU_MAX_WORKGROUP_SIZE, 0,                                       \
        "Maximum number of workitems in a workgroup for GPU, 0 -use default") \
release(int, GPU_MAX_WORKGROUP_SIZE_2D_X, 0,                                  \
        "Maximum number of workitems in a 2D workgroup for GPU, x component, 0 -use default") \
release(int, GPU_MAX_WORKGROUP_SIZE_2D_Y, 0,                                  \
        "Maximum number of workitems in a 2D workgroup for GPU, y component, 0 -use default") \
release(int, GPU_MAX_WORKGROUP_SIZE_3D_X, 0,                                  \
        "Maximum number of workitems in a 3D workgroup for GPU, x component, 0 -use default") \
release(int, GPU_MAX_WORKGROUP_SIZE_3D_Y, 0,                                  \
        "Maximum number of workitems in a 3D workgroup for GPU, y component, 0 -use default") \
release(int, GPU_MAX_WORKGROUP_SIZE_3D_Z, 0,                                  \
        "Maximum number of workitems in a 3D workgroup for GPU, z component, 0 -use default") \
debug(bool, CPU_MEMORY_GUARD_PAGES, false,                                    \
        "Use guard pages for CPU memory")                                     \
debug(size_t, CPU_MEMORY_GUARD_PAGE_SIZE, 64,                                 \
        "Size in KB of CPU memory guard page")                                \
debug(size_t, CPU_MEMORY_ALIGNMENT_SIZE, 256,                                 \
        "Size in bytes for the default alignment for guarded memory on CPU")  \
debug(size_t, PARAMETERS_MIN_ALIGNMENT, 16,                                   \
        "Minimum alignment required for the abstract parameters stack")       \
debug(size_t, MEMOBJ_BASE_ADDR_ALIGN, 4*Ki,                                   \
        "Alignment of the base address of any allocate memory object")        \
release(cstring, GPU_DEVICE_ORDINAL, "",                                      \
        "Select the device ordinal (comma seperated list of available devices)") \
release(bool, REMOTE_ALLOC, false,                                            \
        "Use remote memory for the global heap allocation")                   \
release(uint, GPU_MAX_HEAP_SIZE, 100,                                         \
        "Set maximum size of the GPU heap to % of board memory")              \
release(uint, GPU_STAGING_BUFFER_SIZE, 1024,                                  \
        "Size of the GPU staging buffer in KiB")                              \
release(bool, GPU_DUMP_BLIT_KERNELS, false,                                   \
        "Dump the kernels for blit manager")                                  \
release(uint, GPU_BLIT_ENGINE_TYPE, 0x0,                                      \
        "Blit engine type: 0 - Default, 1 - Host, 2 - CAL, 3 - Kernel")       \
release(bool, GPU_FLUSH_ON_EXECUTION, false,                                  \
        "Submit commands to HW on every operation. 0 - Disable, 1 - Enable")  \
release(bool, GPU_USE_SYNC_OBJECTS, true,                                     \
        "If enabled, use sync objects instead of polling")                    \
release(size_t, GPU_FORCE_BLIT_COPY_SIZE, 0,                                  \
        "Size in KB of the threshold below which to force blit instead for sdma") \
release(bool, CL_KHR_FP64, true,                                              \
        "Enable/Disable support for double precision")                        \
release(cstring, AMD_OCL_BUILD_OPTIONS, 0,                                    \
        "Set clBuildProgram() and clCompileProgram()'s options (override)")   \
release(cstring, AMD_OCL_BUILD_OPTIONS_APPEND, 0,                             \
        "Append clBuildProgram() and clCompileProgram()'s options")           \
release(cstring, AMD_OCL_LINK_OPTIONS, 0,                                     \
        "Set clLinkProgram()'s options (override)")                           \
release(cstring, AMD_OCL_LINK_OPTIONS_APPEND, 0,                              \
        "Append clLinkProgram()'s options")                                   \
release(cstring, AMD_OCL_SC_LIB, 0,                                           \
        "Set shader compiler shared library name or path")                    \
debug(bool, AMD_OCL_ENABLE_MESSAGE_BOX, false,                                \
        "Enable the error dialog on Windows")                                 \
release(size_t, GPU_PINNED_XFER_SIZE, 32,                                     \
        "The pinned buffer size for pinning in read/write transfers")         \
release(size_t, GPU_PINNED_MIN_XFER_SIZE, 1024,                               \
        "The minimal buffer size for pinned read/write transfers in KBytes")  \
release(size_t, GPU_RESOURCE_CACHE_SIZE, 64,                                  \
        "The resource cache size in MB")                                      \
release(size_t, GPU_MAX_SUBALLOC_SIZE, 4096,                                  \
        "The maximum size accepted for suballocaitons in KB")                 \
release(bool, GPU_FORCE_64BIT_PTR, 0,                                         \
        "Forces 64 bit pointers on GPU")                                      \
release(bool, GPU_FORCE_OCL20_32BIT, 0,                                       \
        "Forces 32 bit apps to take CLANG\HSAIL path")                        \
release(bool, GPU_RAW_TIMESTAMP, 0,                                           \
        "Reports GPU raw timestamps in GPU timeline")                         \
release(size_t, GPU_NUM_MEM_DEPENDENCY, 256,                                  \
        "Number of memory objects for dependency tracking")                   \
release(size_t, GPU_XFER_BUFFER_SIZE, 0,                                      \
        "Transfer buffer size for image copy optimization in KB")             \
release(bool, GPU_IMAGE_DMA, true,                                            \
        "Enable DRM DMA for image transfers")                                 \
release(uint, GPU_SINGLE_ALLOC_PERCENT, 85,                                   \
        "Maximum size of a single allocation as percentage of total")         \
release(uint, GPU_NUM_COMPUTE_RINGS, 2,                                       \
        "GPU number of compute rings. 0 - disabled, 1 , 2,.. - the number of compute rings") \
release(int, GPU_SELECT_COMPUTE_RINGS_ID, -1,                                 \
        "GPU select the compute rings ID -1 - disabled, 0 , 1,.. - the forced compute rings ID for submission") \
release(uint, GPU_WORKLOAD_SPLIT, 22,                                         \
        "Workload split size")                                                \
release(bool, GPU_USE_SINGLE_SCRATCH, false,                                  \
        "Use single scratch buffer per device instead of per HW ring")        \
release(bool, AMD_OCL_WAIT_COMMAND, false,                                    \
        "1 = Enable a wait for every submitted command")                      \
/* HSAIL is by default, except Linux 32bit, because of known Catalyst 32bit issue */  \
release(bool, GPU_HSAIL_ENABLE, LP64_SWITCH(LINUX_SWITCH(false,true),true),   \
        "Enable HSAIL on dGPU stack (requires CI+ HW)")                       \
release(uint, GPU_PRINT_CHILD_KERNEL, 0,                                      \
        "Prints the specified number of the child kernels")                   \
release(bool, GPU_USE_DEVICE_QUEUE, false,                                    \
        "Use a dedicated device queue for the actual submissions")            \
release(bool, GPU_ENABLE_LARGE_ALLOCATION, true,                              \
        "Enable >4GB single allocations")                                     \
release(bool, AMD_THREAD_TRACE_ENABLE, true,                                  \
        "Enable thread trace extension")                                      \
release(uint, OPENCL_VERSION, (IS_BRAHMA ? 120 : 200),                        \
        "Force GPU opencl verison")                                           \
release(bool, HSA_LOCAL_MEMORY_ENABLE, true,                                  \
        "Enable HSA device local memory usage")                               \
release(uint, HSA_KERNARG_POOL_SIZE, 512 * 1024,                              \
        "Kernarg pool size")                                                  \
release(bool, HSA_ENABLE_COARSE_GRAIN_SVM, true,                              \
        "Enable device memory for coarse grain SVM allocations")              \
release(bool, GPU_IFH_MODE, false,                                            \
        "1 = Enable GPU IFH (infinitely fast hardware) mode. Any other value keeps setting disabled.") \
release(bool, GPU_MIPMAP, true,                                               \
        "Enables GPU mipmap extension")                                       \
release(uint, GPU_ENABLE_PAL, 2,                                              \
        "Enables PAL backend. 0 - GSL(default), 1 - PAL, 2 - GSL and PAL")    \
release(bool, DISABLE_DEFERRED_ALLOC, false,                                  \
        "Disables deferred memory allocation on device")                      \
release(int, AMD_GPU_FORCE_SINGLE_FP_DENORM, -1,                              \
        "Force denorm for single precision: -1 - don't force, 0 - disable, 1 - enable") \
release(uint, OCL_SET_SVM_SIZE, 4*16384,                                      \
        "set SVM space size for discrete GPU")                                \
debug(uint, OCL_SYSMEM_REQUIREMENT, 2,                                        \
        "Use flag to change the minimum requirement of system memory not to downgrade")        \
debug(bool, GPU_ENABLE_HW_DEBUG, false,                                       \
        "Enable HW DEBUG for GPU")                                            \
release(uint, GPU_WAVES_PER_SIMD, 0,                                          \
        "Force the number of waves per SIMD (1-10)")                          \
release(bool, GPU_WAVE_LIMIT_ENABLE, false,                                   \
        "1 = Enable adaptive wave limiter")                                   \
release(bool, OCL_STUB_PROGRAMS, false,                                       \
        "1 = Enables OCL programs stubing")                                   \
release(bool, GPU_ANALYZE_HANG, false,                                        \
        "1 = Enables GPU hang analysis")                                      \
release(uint, GPU_MAX_REMOTE_MEM_SIZE, 2,                                     \
        "Maximum size (in Ki) that allows device memory substitution with system") \
release(bool, GPU_ADD_HBCC_SIZE, false,                                        \
        "Add HBCC size to the reported device memory")                        \
release_on_stg(uint, GPU_WAVE_LIMIT_CU_PER_SH, 0,                             \
        "Assume the number of CU per SH for wave limiter")                    \
release_on_stg(uint, GPU_WAVE_LIMIT_MAX_WAVE, 10,                             \
        "Set maximum waves per SIMD to try for wave limiter")                 \
release_on_stg(uint, GPU_WAVE_LIMIT_RUN, 20,                                  \
        "Set running factor for wave limiter")                                \
release_on_stg(cstring, GPU_WAVE_LIMIT_DUMP, "",                              \
        "File path prefix for dumping wave limiter output")                   \
release_on_stg(cstring, GPU_WAVE_LIMIT_TRACE, "",                             \
        "File path prefix for tracing wave limiter")                          \
release(bool, OCL_CODE_CACHE_ENABLE, false,                                   \
        "1 = Enable compiler code cache")                                     \
release(bool, OCL_CODE_CACHE_RESET, false,                                    \
        "1 =  Reset the compiler code cache storage")                         \
release(bool, GPU_VEGA10_ONLY, VEGA10_ONLY,                                   \
        "1 = Report vega10 only on OCL/ROCR")                                 \
release_on_stg(bool, PAL_DISABLE_SDMA, false,                                 \
        "1 = Disable SDMA for PAL")                                           \
release(uint, PAL_RGP_DISP_COUNT, 50,                                         \
        "The number of dispatches for RGP capture with SQTT")                 \
release(uint, PAL_MALL_POLICY, 0,                                             \
        "Controls the behaviour of allocations with respect to the MALL"      \
        "0 = MALL policy is decided by KMD"                                   \
        "1 = Allocations are never put through the MALL"                      \
        "2 = Allocations will always be put through the MALL")                \
release(bool, GPU_ENABLE_WAVE32_MODE, true,                                   \
        "Enables Wave32 compilation in HW if available")                      \
release(bool, GPU_ENABLE_LC, true,                                            \
        "Enables LC path")                                                    \
release(bool, GPU_ENABLE_HW_P2P, false,                                       \
        "Enables HW P2P path")                                                \
release(bool, GPU_ENABLE_COOP_GROUPS, false,                                  \
         "Enables cooperative group launch")                                  \
release(uint, GPU_MAX_COMMAND_BUFFERS, 8,                                     \
         "The maximum number of command buffers allocated per queue")         \
release(uint, GPU_MAX_HW_QUEUES, 4,                                           \
         "The maximum number of HW queues allocated per device")              \
release(bool, GPU_IMAGE_BUFFER_WAR, true,                                     \
        "Enables image buffer workaround")                                    \
release(cstring, HIP_VISIBLE_DEVICES, "",                                     \
        "Only devices whose index is present in the sequence are visible to HIP")  \
release(cstring, CUDA_VISIBLE_DEVICES, "",                                    \
        "Only devices whose index is present in the sequence are visible to HIP")    \
release(bool, GPU_ENABLE_WGP_MODE, true,                                      \
        "Enables WGP Mode in HW if available")                                \
release(bool, GPU_DUMP_CODE_OBJECT, false,                                    \
        "Enable dump code object")                                            \
release(uint, GPU_MAX_USWC_ALLOC_SIZE, 2048,                                  \
        "Set a limit in Mb on the maximum USWC allocation size"               \
        "-1 = No limit")                                                      \
release(uint, AMD_SERIALIZE_KERNEL, 0,                                        \
        "Serialize kernel enqueue, 0x1 = Wait for completion before enqueue"  \
        "0x2 = Wait for completion after enqueue 0x3 = both")                 \
release(uint, AMD_SERIALIZE_COPY, 0,                                          \
        "Serialize copies, 0x1 = Wait for completion before enqueue"          \
        "0x2 = Wait for completion after enqueue 0x3 = both")                 \
release(bool, PAL_ALWAYS_RESIDENT, false,                                     \
        "Force memory resources to become resident at allocation time")       \
release(uint, HIP_HOST_COHERENT, 0,                                           \
        "Coherent memory in hipHostMalloc, 0x1 = memory is coherent with host"\
        "0x0 = memory is not coherent between host and GPU")                  \
release(uint, AMD_OPT_FLUSH, 1,                                               \
        "Kernel flush option , 0x0 = Use system-scope fence operations."      \
        "0x1 = Use device-scope fence operations when possible.")             \
release(uint, HIP_HIDDEN_FREE_MEM, 0,                                         \
        "Reserve free mem reporting in Mb"                                    \
        "0 = Disable")

namespace amd {

extern bool IS_HIP;

extern bool IS_LEGACY;

//! \addtogroup Utils
//  @{

struct Flag {
  enum Type {
    Tinvalid = 0,
    Tbool,    //!< A boolean type flag (true, false).
    Tint,     //!< An integer type flag (signed).
    Tuint,    //!< An integer type flag (unsigned).
    Tsize_t,  //!< A size_t type flag.
    Tcstring  //!< A string type flag.
  };

#define DEFINE_FLAG_NAME(type, name, value, help) k##name,
  enum Name {
    RUNTIME_FLAGS(DEFINE_FLAG_NAME, DEFINE_FLAG_NAME, DEFINE_FLAG_NAME)
    numFlags_
  };
#undef DEFINE_FLAG_NAME

#define CAN_SET(type, name, v, h)    static const bool cannotSet##name = false;
#define CANNOT_SET(type, name, v, h) static const bool cannotSet##name = true;

#ifdef DEBUG
  RUNTIME_FLAGS(CAN_SET, CAN_SET, CAN_SET)
#else // !DEBUG
  RUNTIME_FLAGS(CANNOT_SET, CAN_SET, CANNOT_SET)
#endif // !DEBUG

#undef CAN_SET
#undef CANNOT_SET

 private:
  static Flag flags_[];

 public:
  static char* envstr_;
  const char* name_;
  const void* value_;
  Type type_;
  bool isDefault_;

 public:
  static bool init();

  static void tearDown();

  bool setValue(const char* value);

  static bool isDefault(Name name) { return flags_[name].isDefault_; }
};

#define flagIsDefault(name) \
  (amd::Flag::cannotSet##name || amd::Flag::isDefault(amd::Flag::k##name))

#define setIfNotDefault(var, opt, other) \
  if (!flagIsDefault(opt)) \
    var = (opt);\
  else \
    var = (other);

//  @}

} // namespace amd

#ifdef _WIN32
# define EXPORT_FLAG extern "C" __declspec(dllexport)
#else // !_WIN32
# define EXPORT_FLAG extern "C"
#endif // !_WIN32

#define DECLARE_RELEASE_FLAG(type, name, value, help) EXPORT_FLAG type name;
#ifdef DEBUG
# define DECLARE_DEBUG_FLAG(type, name, value, help) EXPORT_FLAG type name;
#else // !DEBUG
# define DECLARE_DEBUG_FLAG(type, name, value, help) const type name = value;
#endif // !DEBUG

RUNTIME_FLAGS(DECLARE_DEBUG_FLAG, DECLARE_RELEASE_FLAG, DECLARE_DEBUG_FLAG);

#undef DECLARE_DEBUG_FLAG
#undef DECLARE_RELEASE_FLAG

#endif /*FLAGS_HPP_*/
