project(vdi)
cmake_minimum_required(VERSION 3.5)
#example command:

#cmake   -DOPENCL_DIR=/path to/opencl ..

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
set(USE_COMGR_LIBRARY "yes")
add_definitions(-DUSE_COMGR_LIBRARY -DCOMGR_DYN_DLL)
add_definitions(-DBSD_LIBELF)

find_package(amd_comgr REQUIRED CONFIG
   PATHS
     /opt/rocm/
   PATH_SUFFIXES
     cmake/amd_comgr
     lib/cmake/amd_comgr
)
MESSAGE(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

find_path(COMGR_INCLUDE
   NAMES amd_comgr.h
   PATH_SUFFIXES include
   PATHS /opt/rocm
)

find_library( OpenCL REQUIRED CONFIG
   PATHS
      /opt/rocm
)

find_path(OPENCL_INCLUDE_DIR
  NAMES OpenCL/cl.h CL/cl.h
  PATH_SUFFIXES include opencl/include inc include/x86_64 include/x64
  PATHS /opt/rocm 
  DOC "OpenCL include header OpenCL/cl.h or CL/cl.h"
)

if( NOT OPENCL_INCLUDE_DIR )
   unset(OPENCL_INCLUDE_DIR CACHE)
   set(OPENCL_INCLUDE_DIR "" CACHE PATH "" FORCE )
else()
   set(OPENCL_ICD_LOADER_HEADERS_DIR "${OPENCL_INCLUDE_DIR}")
endif()
   MESSAGE(STATUS "Opencl found at ${OPENCL_INCLUDE_DIR}.")


find_package(ROCT REQUIRED)
find_package(ROCR REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

add_definitions(-DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=1 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER)
add_definitions(-DOPENCL_C_MAJOR=2 -DOPENCL_C_MINOR=0)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

include_directories(${ROCR_INCLUDES})
if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")
  include_directories(${LLVM_INCLUDES})
endif() # if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/device/rocm)
include_directories(${OPENCL_DIR})
include_directories(${OPENCL_DIR}/amdocl)
include_directories(${OPENCL_DIR}/khronos)
include_directories(${OPENCL_DIR}/khronos/headers)
include_directories(${OPENCL_DIR}/khronos/headers/opencl2.2)
include_directories(${OPENCL_INCLUDE_DIR})
include_directories(${COMGR_INCLUDE})


include_directories(${CMAKE_SOURCE_DIR}/compiler/lib)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/include)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/backends/common)

include_directories(${CMAKE_SOURCE_DIR}/elf/utils/common)
include_directories(${CMAKE_SOURCE_DIR}/elf/utils/libelf)

add_subdirectory(device/rocm)




#find_package(amd_comgr REQUIRED CONFIG)
include_directories("$<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>")
set(COMGR_CPP device/comgrctx.cpp)

add_subdirectory( elf/utils/libelf "${CMAKE_CURRENT_BINARY_DIR}/libelf" )
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
  device/devkernel.cpp
  device/devwavelimiter.cpp
  device/devprogram.cpp
  platform/activity.cpp
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
  compiler/lib/utils/options.cpp
  elf/elf.cpp
  elf/elf_utils.cpp
  #${CMAKE_SOURCE_DIR}/compiler/tools/caching/cache.cpp
  ${COMGR_CPP}
)

set_target_properties(oclruntime PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(amdvdi_static STATIC 
  	   $<TARGET_OBJECTS:oclruntime>
  	   $<TARGET_OBJECTS:oclrocm>
  	   $<TARGET_OBJECTS:oclelf_obj>
  	   )
set_target_properties(amdvdi_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
#add_library(oclelf STATIC IMPORTED)
#set_target_properties(oclelf PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/libelf/liboclelf.a")
#target_link_libraries(oclruntime_static PRIVATE oclelf)
target_link_libraries(amdvdi_static PUBLIC Threads::Threads)
target_link_libraries(amdvdi_static PUBLIC dl ${ROCT_LIBRARIES} ${ROCR_LIBRARIES} )

if (UNIX)
    set(VDI_DEFAULT_INSTALL_PREFIX "/opt/rocm/vdi")
endif()
#comment out as it's not available in cmake 3.5
#if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
if(NOT CMAKE_INSTALL_PREFIX)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Installation path for vdi" FORCE)
    #elseif(CMAKE_BUILD_TYPE MATCHES Release)
    else()
        set(CMAKE_INSTALL_PREFIX ${VDI_DEFAULT_INSTALL_PREFIX} CACHE PATH "Installation path for vdi" FORCE)
   #    message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE specified. Valid values are Debug and Release")
   endif()
#endif()
endif()
message("cmake install path  ${CMAKE_INSTALL_PREFIX}")

install( TARGETS amdvdi_static ARCHIVE DESTINATION lib LIBRARY DESTINATION lib COMPONENT applications)
FILE(GLOB headerfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")
INSTALL (
    FILES ${headerfiles}
    DESTINATION include)
INSTALL (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/compiler"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
INSTALL (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
INSTALL (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/platform"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
INSTALL (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thread"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
INSTALL (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/os"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
INSTALL (
   DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/device"
   DESTINATION include
   FILES_MATCHING PATTERN "*.h*")
   #set(AMDGCN_DEP_LIST)
   #if(AMDGCN_TARGETS_LIB_DEPS)
   #set(${AMDGCN_TARGETS_LIB_DEPS} ${AMDGCN_DEP_LIST} PARENT_SCOPE)
   #endif()

   #foreach(AMDGCN_DEP_TARGET ${AMDGCN_DEP_TARGETS})
   #add_dependencies(oclruntime ${AMDGCN_DEP_TARGET})
   #endforeach()
