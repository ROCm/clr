project(ROCclr)
cmake_minimum_required(VERSION 3.5)
#example command:

#cmake   -DOPENCL_DIR=/path to/opencl ..

set(ROCCLR_EXPORTS_FILE "${CMAKE_CURRENT_BINARY_DIR}/amdrocclr_staticTargets.cmake")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# FIXME: Should be option and ON/OFF, not yes/no variable.
set(USE_COMGR_LIBRARY "yes")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("-Wno-ignored-attributes")
endif()

find_package(amd_comgr REQUIRED CONFIG
   PATHS
     /opt/rocm/
   PATH_SUFFIXES
     cmake/amd_comgr
     lib/cmake/amd_comgr
)
message(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

# FIXME: Use find_package(OpenCL)
find_library( OpenCL REQUIRED CONFIG
   PATHS
      /opt/rocm
)

if( NOT OPENCL_DIR )
   find_path(OPENCL_INCLUDE_DIR
      NAMES OpenCL/cl.h CL/cl.h
      PATH_SUFFIXES include opencl/include inc include/x86_64 include/x64
      PATHS /opt/rocm
      DOC "OpenCL include header OpenCL/cl.h or CL/cl.h"
   )

   if( NOT OPENCL_INCLUDE_DIR )
      unset(OPENCL_INCLUDE_DIR CACHE)
      set(OPENCL_INCLUDE_DIR "" CACHE PATH "" FORCE )
   endif()
      message(STATUS "Opencl found at ${OPENCL_INCLUDE_DIR}.")
endif()



find_package(ROCT REQUIRED)
find_package(ROCR REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-D__x86_64__ -DOPENCL_MAJOR=2 -DOPENCL_MINOR=1 -DCL_TARGET_OPENCL_VERSION=220 -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER)
add_definitions(-DOPENCL_C_MAJOR=2 -DOPENCL_C_MINOR=0)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

if(DEV_LOG_ENABLE MATCHES "yes")
  add_definitions(-DDEV_LOG_ENABLE=1)
endif()

if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")
  include_directories(${LLVM_INCLUDES})
endif() # if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/device/rocm)
include_directories(${OPENCL_DIR})
include_directories(${OPENCL_DIR}/amdocl)
include_directories(${OPENCL_DIR}/khronos)
include_directories(${OPENCL_DIR}/khronos/headers)
include_directories(${OPENCL_DIR}/khronos/headers/opencl2.2)
include_directories(${OPENCL_INCLUDE_DIR})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/compiler/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/compiler/lib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/compiler/lib/backends/common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/elf/utils/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/elf/utils/libelf)

option(BUILD_PAL "Build PAL backend" OFF)
if (BUILD_PAL)
  add_subdirectory(device/pal)

  add_definitions(-DAMD_LIBELF)
  add_subdirectory(compiler/sc/HSAIL)
else ()
  add_subdirectory(device/rocm)
endif()

set(COMGR_CPP device/comgrctx.cpp)

add_subdirectory(elf/utils/libelf "${CMAKE_CURRENT_BINARY_DIR}/libelf" )

set(oclruntime_src
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
  device/devkernel.cpp
  device/devwavelimiter.cpp
  device/devprogram.cpp
  platform/activity.cpp
  platform/kernel.cpp
  platform/context.cpp
  platform/command.cpp
  platform/ndrange.cpp
  platform/runtime.cpp
  platform/memory.cpp
  platform/program.cpp
  platform/commandqueue.cpp
  platform/agent.cpp
  os/os_win32.cpp
  os/alloc.cpp
  os/os.cpp
  os/os_posix.cpp
  compiler/lib/utils/options.cpp
  elf/elf.cpp
  elf/elf_utils.cpp
  #${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools/caching/cache.cpp
)

add_library(amdrocclr_static STATIC
  ${oclruntime_src} ${COMGR_CPP}
  $<TARGET_OBJECTS:oclelf_obj>)

set_target_properties(amdrocclr_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(amdrocclr_static
  PUBLIC
    $<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>
  PRIVATE
    $<TARGET_PROPERTY:oclelf_obj,INTERFACE_INCLUDE_DIRECTORIES>
    ${ROCR_INCLUDES}
    ${ROCT_INCLUDES})

target_compile_definitions(amdrocclr_static PRIVATE BSD_LIBELF)

if(${USE_COMGR_LIBRARY} MATCHES "yes")
  # FIXME: This should not be part of the public interface. Downstream
  # users need to add these definitions. This should be defined in a
  # config header here so other builds don't need to be aware of this.
  target_compile_definitions(amdrocclr_static PUBLIC USE_COMGR_LIBRARY COMGR_DYN_DLL)
endif()

target_link_libraries(amdrocclr_static PUBLIC Threads::Threads)
target_link_libraries(amdrocclr_static PUBLIC dl ${ROCT_LIBRARIES} ${ROCR_LIBRARIES} )

if (BUILD_PAL)
  target_sources(amdrocclr_static PRIVATE $<TARGET_OBJECTS:rocclrpal>)
  target_link_libraries(amdrocclr_static PRIVATE amdhsaloader)
  target_link_libraries(amdrocclr_static PRIVATE pal)

  target_compile_definitions(amdrocclr_static PRIVATE WITH_PAL_DEVICE)

  export(TARGETS amdrocclr_static rocclrpal amdhsaloader amdhsacode pal addrlib vam metrohash cwpack gpuopen ddCore mpack rapidjson
    FILE ${ROCCLR_EXPORTS_FILE}
  )
else()
  target_compile_definitions(amdrocclr_static PRIVATE WITH_HSA_DEVICE)

  target_sources(amdrocclr_static PRIVATE $<TARGET_OBJECTS:oclrocm>)
  export(TARGETS amdrocclr_static
    FILE ${ROCCLR_EXPORTS_FILE}
  )
endif()

if (UNIX)
    set(ROCclr_DEFAULT_INSTALL_PREFIX "/opt/rocm/rocclr")
endif()
#comment out as it's not available in cmake 3.5
#if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
if(NOT CMAKE_INSTALL_PREFIX)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Installation path for rocclr" FORCE)
    #elseif(CMAKE_BUILD_TYPE MATCHES Release)
    else()
        set(CMAKE_INSTALL_PREFIX ${ROCclr_DEFAULT_INSTALL_PREFIX} CACHE PATH "Installation path for rocclr" FORCE)
   #    message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE specified. Valid values are Debug and Release")
   endif()
#endif()
endif()

install( TARGETS amdrocclr_static ARCHIVE DESTINATION lib LIBRARY DESTINATION lib COMPONENT applications)
file(GLOB headerfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")
install (
    FILES ${headerfiles}
    DESTINATION include)
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/elf"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/compiler"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/platform"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thread"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/os"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
install (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/device"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
