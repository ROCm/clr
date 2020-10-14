cmake_minimum_required(VERSION 3.5)

project(ROCclr VERSION "1.0.0" LANGUAGES C CXX)

include(CMakePackageConfigHelpers)

#decide whether .so is to be build or .a
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared library (.so) or not.")

#example command:

#cmake   -DOPENCL_DIR=/path to/opencl ..

set(ROCCLR_CONFIG_NAME ROCclrConfig.cmake)
set(ROCCLR_TARGETS_NAME rocclr-targets.cmake)
set(ROCCLR_VERSION_NAME rocclr-config-version.cmake)
set(ROCCLR_PACKAGE_PREFIX lib/cmake/rocclr)
set(ROCCLR_PREFIX_CODE)
set(ROCCLR_TARGETS_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/${ROCCLR_PACKAGE_PREFIX}/${ROCCLR_TARGETS_NAME}")
set(ROCCLR_VERSION_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/${ROCCLR_PACKAGE_PREFIX}/${ROCCLR_VERSION_NAME}")

# Generate the build-tree package.
configure_file("cmake/${ROCCLR_CONFIG_NAME}.in"
  "${ROCCLR_PACKAGE_PREFIX}/${ROCCLR_CONFIG_NAME}"
  @ONLY)

write_basic_package_version_file("${ROCCLR_VERSION_PATH}"
  VERSION "${ROCclr_VERSION}"
  COMPATIBILITY SameMajorVersion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# FIXME: Handling legacy custom boolean matching "yes" or "no"
if(DEFINED USE_COMGR_LIBRARY)
  if(${USE_COMGR_LIBRARY} MATCHES "yes")
    set(USE_COMGR_LIBRARY ON)
  elseif(${USE_COMGR_LIBRARY} MATCHES "no")
    set(USE_COMGR_LIBRARY OFF)
  endif()
endif()

option(USE_COMGR_LIBRARY "Use comgr library" ON)

find_package(amd_comgr REQUIRED CONFIG
   PATHS
     /opt/rocm/
   PATH_SUFFIXES
     cmake/amd_comgr
     lib/cmake/amd_comgr
)
message(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

find_package(hsa-runtime64 REQUIRED CONFIG
   PATHS
     /opt/rocm/
   PATH_SUFFIXES
     cmake/hsa-runtime64
     lib/cmake/hsa-runtime64
     lib64/cmake/hsa-runtime64
)
message(STATUS "HSA Runtime found at ${hsa-runtime64_DIR}.")

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



set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  add_definitions(
    # Enabling -Wextra or -pedantic will cause
    # thousands of warnings. Keep things simple for now.
    -Wall
    # Makefile build adds -fno-strict-aliasing instead.
    -Wno-strict-aliasing
    # This one seems impossible to fix for now.
    # There are hundreds of instances of unused vars/functions
    # throughout the code base.
    -Wno-unused-variable)
endif()

add_definitions(-D__x86_64__ -DOPENCL_MAJOR=2 -DOPENCL_MINOR=1 -DCL_TARGET_OPENCL_VERSION=220 -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER)
add_definitions(-DOPENCL_C_MAJOR=2 -DOPENCL_C_MINOR=0)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

if(DEV_LOG_ENABLE MATCHES "yes")
  add_definitions(-DDEV_LOG_ENABLE)
endif()

option(BUILD_PAL "Build PAL backend" OFF)
if (BUILD_PAL)
  add_subdirectory(device/pal)

  add_subdirectory(compiler/sc/HSAIL)
else ()
  add_subdirectory(device/rocm)
endif()

set(COMGR_CPP device/comgrctx.cpp)

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
  device/devhcprintf.cpp
  device/devhcmessages.cpp
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
  #${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools/caching/cache.cpp
)

add_library(amdrocclr_static STATIC
  ${oclruntime_src} ${COMGR_CPP})

set_target_properties(amdrocclr_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(amdrocclr_static
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/device/rocm>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/compiler/lib/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/compiler/lib>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/compiler/lib/backends/common>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/elf>
    $<BUILD_INTERFACE:${OPENCL_DIR}>
    # GL and EGL headers.
    $<BUILD_INTERFACE:${OPENCL_DIR}/khronos/headers>
    $<BUILD_INTERFACE:${OPENCL_DIR}/khronos/headers/opencl2.2>
    $<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>)

if(USE_COMGR_LIBRARY)
  # FIXME: This should not be part of the public interface. Downstream
  # users need to add these definitions. This should be defined in a
  # config header here so other builds don't need to be aware of this.
  if(${BUILD_SHARED_LIBS})
     target_compile_definitions(amdrocclr_static PUBLIC USE_COMGR_LIBRARY COMGR_DYN_DLL)
  else()
     target_compile_definitions(amdrocclr_static PUBLIC USE_COMGR_LIBRARY)
  endif()
  #Needed here to export as transitive dependency in rocclr-targets.cmake
  target_link_libraries(amdrocclr_static PRIVATE amd_comgr)
endif()

target_link_libraries(amdrocclr_static PUBLIC Threads::Threads)
target_link_libraries(amdrocclr_static PUBLIC dl)


if (BUILD_PAL)
  target_sources(amdrocclr_static PRIVATE $<TARGET_OBJECTS:rocclrpal>)
  target_link_libraries(amdrocclr_static PRIVATE amdhsaloader)
  target_link_libraries(amdrocclr_static PRIVATE pal)

  target_compile_definitions(amdrocclr_static PRIVATE WITH_PAL_DEVICE)

  export(TARGETS amdrocclr_static rocclrpal amdhsaloader amdhsacode pal addrlib vam metrohash cwpack gpuopen ddCore tiny_printf mpack rapidjson
    FILE ${ROCCLR_TARGETS_PATH})
else()
  target_compile_definitions(amdrocclr_static PRIVATE WITH_HSA_DEVICE)

  #ROCclr being static lib shall not link hsa-runtime64.
  #Needed here to export as transitive dependency in rocclr-targets.cmake
  target_link_libraries(amdrocclr_static PRIVATE hsa-runtime64::hsa-runtime64)
  target_sources(amdrocclr_static PRIVATE $<TARGET_OBJECTS:oclrocm>)
  export(TARGETS amdrocclr_static
         FILE ${ROCCLR_TARGETS_PATH})
endif()

if (UNIX)
    set(ROCclr_DEFAULT_INSTALL_PREFIX "/opt/rocm/rocclr")

    find_library(LIBRT rt)
    if (LIBRT)
        target_link_libraries(amdrocclr_static PUBLIC ${LIBRT})
    endif()

    find_library(LIBNUMA numa)
    if (LIBNUMA)
        target_compile_definitions(amdrocclr_static PUBLIC ROCCLR_SUPPORT_NUMA_POLICY)
        target_link_libraries(amdrocclr_static PUBLIC ${LIBNUMA})
        message(STATUS "Found: ${LIBNUMA}")
    endif()
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


install( TARGETS amdrocclr_static
  EXPORT rocclr
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  COMPONENT applications)

install (
   DIRECTORY include/
   DESTINATION include)
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

# Generate the install-tree package.
set(ROCCLR_PREFIX_CODE "
# Derive absolute install prefix from config file path.
get_filename_component(ROCCLR_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
string(REGEX REPLACE "/" ";" count "${ROCCLR_PACKAGE_PREFIX}")
foreach(p ${count})
  set(ROCCLR_PREFIX_CODE "${ROCCLR_PREFIX_CODE}
get_filename_component(ROCCLR_PREFIX \"\${ROCCLR_PREFIX}\" PATH)")
endforeach()

configure_file("cmake/${ROCCLR_CONFIG_NAME}.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${ROCCLR_CONFIG_NAME}.install"
  @ONLY)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${ROCCLR_CONFIG_NAME}.install"
  DESTINATION "${ROCCLR_PACKAGE_PREFIX}"
  RENAME "${ROCCLR_CONFIG_NAME}")
install(EXPORT rocclr
  DESTINATION "${ROCCLR_PACKAGE_PREFIX}"
  FILE "${ROCCLR_TARGETS_NAME}")
install(FILES
  "${ROCCLR_VERSION_PATH}"
  DESTINATION "${ROCCLR_PACKAGE_PREFIX}")
