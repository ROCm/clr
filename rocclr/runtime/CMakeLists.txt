set(CMAKE_CXX_FLAGS "-std=c++11")

add_definitions(-DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DAMD_LIBELF -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DWITH_VERSION_0_8=1 -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER -DBSD_LIBELF)

include_directories(/opt/rocm/include/hsa)

include_directories(${CMAKE_SOURCE_DIR}/runtime)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.0)
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/lib/Target/AMDGPU/MCTargetDesc)
include_directories(${CMAKE_SOURCE_DIR}/compiler/driver/src)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/include)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/backends/common)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/common)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/libelf)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders)
include_directories(${CMAKE_SOURCE_DIR}/compiler/tools)

add_subdirectory(device/rocm)

add_library(oclruntime OBJECT
  thread/thread.cpp
  thread/monitor.cpp
  thread/semaphore.cpp
  utils/flags.cpp
  utils/debug.cpp
  device/appprofile.cpp
  device/device.cpp
  device/hwdebug.cpp
  device/blitcl.cpp
  device/blit.cpp
  platform/kernel.cpp
  platform/context.cpp
  platform/command.cpp
  platform/ndrange.cpp
  platform/runtime.cpp
  platform/object.cpp
  platform/memory.cpp
  platform/program.cpp
  platform/commandqueue.cpp
  platform/agent.cpp
  os/os_win32.cpp
  os/alloc.cpp
  os/os.cpp
  os/os_posix.cpp
  ${CMAKE_SOURCE_DIR}/compiler/lib/utils/options.cpp
  ${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/elf.cpp
  ${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/elf_utils.cpp
  ${CMAKE_SOURCE_DIR}/compiler/tools/caching/cache.cpp
)
set_target_properties(oclruntime PROPERTIES POSITION_INDEPENDENT_CODE ON)

