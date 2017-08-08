set(CMAKE_CXX_FLAGS "-std=c++11")

option(GENERIC_IS_ZERO "Teach LLVM/Clang that generic address space IS address space 0 for AMDGPU target" OFF)
if (GENERIC_IS_ZERO)
  add_definitions(-DAMDGCN_TRIPLE=amdgcn-amd-amdhsa-amdgizcl)
else (GENERIC_IS_ZERO)
  add_definitions(-DAMDGCN_TRIPLE=amdgcn-amd-amdhsa-opencl)
endif (GENERIC_IS_ZERO)

add_definitions(-DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

include_directories(${ROCR_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/runtime)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.0)
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/include)
include_directories(${CMAKE_SOURCE_DIR}/compiler/driver/src)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/include)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/backends/common)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders)
include_directories(${CMAKE_SOURCE_DIR}/compiler/tools)

add_definitions(-DBSD_LIBELF)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/common)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/libelf)

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

