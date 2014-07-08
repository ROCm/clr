//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef FLAGS_HPP_
#define FLAGS_HPP_


#define RUNTIME_FLAGS(debug,release,release_on_stg)                           \
                                                                              \
debug(int, LOG_LEVEL, 0,                                                      \
        "The default log level")                                              \
debug(bool, BREAK_ON_LOG_WARNING, false,                                      \
        "Break each time an error is logged")                                 \
debug(bool, BREAK_ON_LOG_ERROR, false,                                        \
        "Break each time an error is logged")                                 \
debug(uint, DEBUG_GPU_FLAGS, 0,                                               \
        "The debug options for GPU device")                                   \
debug(uint, GPU_MAX_COMMAND_QUEUES, 80,                                       \
        "The maximum number of concurrent Virtual GPUs")                      \
release(size_t, CQ_THREAD_STACK_SIZE, 256*Ki, /* @todo: that much! */         \
        "The default command queue thread stack size")                        \
release(size_t, CPU_WORKER_THREAD_STACK_SIZE, 64*Ki,                          \
        "The default CPU worker thread stack size")                           \
release(int, CPU_MAX_COMPUTE_UNITS, -1,                                       \
        "Override the number of computation units per CPU device")            \
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
release(int, CPU_MAX_WORKGROUP_SIZE, 1024,                                    \
        "Maximum number of workitems in a workgroup for CPU")                 \
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
release(cstring, GPU_DEVICE_NAME, "",                                         \
        "Select the device ordinal (will only report a single device)")       \
release(cstring, GPU_DEVICE_ORDINAL, "",                                      \
        "Select the device ordinal (comma seperated list of available devices)") \
release(bool, REMOTE_ALLOC, false,                                            \
        "Use remote memory for the global heap allocation")                   \
release(int, GPU_INITIAL_HEAP_SIZE, 16,                                       \
        "Initial size of the GPU heap in MiB")                                \
release(uint, GPU_MAX_HEAP_SIZE, 100,                                         \
        "Set maximum size of the GPU heap to % of board memory")              \
release(int, GPU_HEAP_GROWTH_INCREMENT, 8,                                    \
        "Amount to grow the GPU heap by in MiB")                              \
release(uint, GPU_STAGING_BUFFER_SIZE, 512,                                   \
        "Size of the GPU staging buffer in KiB")                              \
release(bool, GPU_DUMP_BLIT_KERNELS, false,                                   \
        "Dump the kernels for blit manager")                                  \
release(uint, GPU_BLIT_ENGINE_TYPE, 0x0,                                      \
        "Blit engine type: 0 - Default, 1 - Host, 2 - CAL, 3 - Kernel")       \
release(bool, GPU_FLUSH_ON_EXECUTION, false,                                  \
        "Submit commands to HW on every operation. 0 - Disable, 1 - Enable")  \
release(bool, GPU_USE_SYNC_OBJECTS, true,                                     \
        "If enabled, use sync objects instead of polling")                    \
release(bool, ENABLE_CAL_SHUTDOWN, false,                                     \
        "Enable explicit CAL shutdown (for PM4 capture)")                     \
release(bool, CL_KHR_FP64, true,                                              \
        "Enable/Disable support for double precision")                        \
release(uint, GPU_OPEN_VIDEO, 0,                                              \
        "Non-zero value allows to report Open Video extension on GPU")        \
release(cstring, AMD_OCL_BUILD_OPTIONS, 0,                                    \
        "Set clBuildProgram() and clCompileProgram()'s options (override)")   \
release(cstring, AMD_OCL_BUILD_OPTIONS_APPEND, 0,                             \
        "Append clBuildProgram() and clCompileProgram()'s options")           \
release(cstring, AMD_OCL_LINK_OPTIONS, 0,                                     \
        "Set clLinkProgram()'s options (override)")                           \
release(cstring, AMD_OCL_LINK_OPTIONS_APPEND, 0,                              \
        "Append clLinkProgram()'s options")                                   \
debug(bool, AMD_OCL_SUPPRESS_MESSAGE_BOX, false,                              \
        "Suppress the error dialog on Windows")                               \
debug(bool, OCL_STRESS_BINARY_IMAGE, false,                                   \
        "Exercise the binary image producer and consumer")                    \
release(cstring, GPU_PRE_RA_SCHED, "default",                                 \
        "Allows setting of alternate pre-RA-sched")                           \
release(size_t, GPU_PINNED_XFER_SIZE, 16,                                     \
        "The pinned buffer size for pinning in read/write transfers")         \
release(size_t, GPU_PINNED_MIN_XFER_SIZE, 512,                                \
        "The minimal buffer size for pinned read/write transfers in KBytes")  \
release(size_t, GPU_RESOURCE_CACHE_SIZE, 64,                                  \
        "The resource cache size in MB")                                      \
release(uint, GPU_ASYNC_MEM_COPY, 0,                                          \
        "Enables async memory transfers with DRM engine")                     \
release(bool, GPU_FORCE_64BIT_PTR, 0,                                         \
        "Forces 64 bit pointers on GPU")                                      \
release(bool, GPU_FORCE_OCL20_32BIT, 0,                                       \
        "Forces 32 bit apps to take CLANG\HSAIL path")                        \
release(bool, GPU_PREALLOC_ADDR_SPACE, 0,                                     \
        "Preallocates 4GB address space. Valid for boards > 4GB")             \
release(bool, GPU_RAW_TIMESTAMP, 0,                                           \
        "Reports GPU raw timestamps in GPU timeline")                         \
release(bool, CPU_IMAGE_SUPPORT, true,                                        \
        "Turn on image support on the CPU device")                            \
release(bool, GPU_PARTIAL_DISPATCH, true,                                     \
        "Enables partial dispatch on GPU")                                    \
release(size_t, GPU_NUM_MEM_DEPENDENCY, 256,                                  \
        "Number of memory objects for dependency tracking")                   \
release(size_t, GPU_XFER_BUFFER_SIZE, 0,                                      \
        "Transfer buffer size for image copy optimization in KB")             \
release(bool, GPU_IMAGE_DMA, true,                                            \
        "Enable DRM DMA for image transfers")                                 \
release(uint, CPU_MAX_ALLOC_PERCENT, 25,                                      \
        "Maximum size of a single allocation in MiB")                         \
release(uint, GPU_MAX_ALLOC_PERCENT, 75,                                      \
        "Maximum size of a single allocation as percentage of total")         \
release(uint, GPU_NUM_COMPUTE_RINGS, 2,                                       \
        "GPU number of compute rings. 0 - disabled, 1 , 2,.. - the number of compute rings") \
release_on_stg(bool, C1X_ATOMICS, !IS_MAINLINE,                               \
        "Runtime will report c1x atomics support")                            \
release(uint, GPU_WORKLOAD_SPLIT, 22,                                         \
        "Workload split size")                                                \
release(bool, GPU_USE_SINGLE_SCRATCH, false,                                  \
        "Use single scratch buffer per device instead of per HW ring")        \
release_on_stg(cstring, GPU_TARGET_INFO_ARCH, "amdil",                        \
        "Select the GPU TargetInfo arch (amdil|hsail)")                       \
release(bool, HSA_RUNTIME, 0,                                                 \
        "1 = Enable HSA Runtime, any other value or absence disables it.")    \
release(bool, AMD_OCL_WAIT_COMMAND, false,                                    \
        "1 = Enable a wait for every submitted command")                      \
debug(bool, AMD_OCL_DEBUG_LINKER, false,                                      \
        "Enable debug output in linker")                                      \
debug(bool, GPU_SPLIT_LIB, true,                                              \
        "Enable splitting GPU 32/64 bit library")                             \
release(bool, GPU_STAGING_WRITE_PERSISTENT, false,                            \
        "Enable Persistent writes")                                           \
release(bool, DRMDMA_FOR_LNX_CF, false,                                       \
        "Enable DRMDMA for Linux CrossFire")                                  \
release(bool, GPU_HSAIL_ENABLE, false,                                        \
        "Enable HSAIL on dGPU stack (requires CI+ HW)")                       \
release(bool, GPU_ASSUME_ALIASES, false,                                      \
        "Assume memory aliases in the compilation process")                   \
release(uint, GPU_PRINT_CHILD_KERNEL, 0,                                      \
        "Prints the specified number of the child kernels")                   \
release(bool, GPU_DIRECT_SRD, true,                                           \
        "Use indirect SRD access in HSAIL")                                   \
release(bool, AMD_DEPTH_MSAA_INTEROP, false,                                  \
        "Enable depth stencil and MSAA buffer interop")                       \
release(bool, AMD_THREAD_TRACE_ENABLE, false,                                 \
        "Enable thread trace extension")                                      \
release(bool, ENVVAR_HSA_POLL_KERNEL_COMPLETION, false,                       \
        "Determines if Hsa runtime should use polling scheme")                \
release(bool, HSA_LOCAL_MEMORY_ENABLE, false,                                 \
        "Enable HSA device local memory usage")                               \
release(bool, HSA_ENABLE_ATOMICS_32B, false,                                  \
        "1 = Enable SVM atomics in 32 bits (HSA backend-only). Any other value keeps then disabled.") \
release(bool, ENABLE_PLATFORM_ATOMICS, false,                \
        "Enable platform atomics")

namespace amd {

//! \addtogroup Utils
//  @{

struct Flag
{
    enum Type
    {
        Tinvalid = 0,
        Tbool,    //!< A boolean type flag (true, false).
        Tint,     //!< An integer type flag (signed).
        Tuint,    //!< An integer type flag (unsigned).
        Tsize_t,  //!< A size_t type flag.
        Tcstring  //!< A string type flag.
    };

#define DEFINE_FLAG_NAME(type, name, value, help) k##name,
    enum Name
    {
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
