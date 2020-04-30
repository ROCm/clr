
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Some linker flags, -Wl,--unresolved-symbols=report-all being one of them, requires -shared-libasan if using -fsanitize=address.
# If -Wl,--unresolved-symbols=report-all is removed from CMAKE_SHARED_LINKER_FLAGS, then update this snippet
if(CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared-libasan")
endif()

set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} " -Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all -Wl,--version-script=${CMAKE_SOURCE_DIR}/amdocl/amdocl.map")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

add_definitions(-DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DCL_TARGET_OPENCL_VERSION=220 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

# FIXME: ROCclr_DIR and LIBROCclr_STATIC_DIR are for compatability with
# current build scripts. Instead they should install, or use
# CMAKE_PREFIX_PATH
#Uncomment once fixed
#find_package(ROCclr CONFIG
#  PATHS
#  ${ROCclr_DIR}
#  ${LIBROCclr_STATIC_DIR})

if(NOT ROCclr_FOUND)
  # FIXME: Make ROCclr find required and remove this.
  #
  # This is to break the dependency on patch to start producing find
  # module.
  include(${LIBROCclr_STATIC_DIR}/amdrocclr_staticTargets.cmake)
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/khronos)
include_directories(${CMAKE_SOURCE_DIR}/khronos/headers)
include_directories(${CMAKE_SOURCE_DIR}/khronos/headers/opencl2.2)
if(${USE_COMGR_LIBRARY} MATCHES "yes")
  add_definitions(-DUSE_COMGR_LIBRARY -DCOMGR_DYN_DLL)
  find_package(amd_comgr REQUIRED CONFIG
     PATHS
       /opt/rocm/
     PATH_SUFFIXES
      cmake/amd_comgr
      lib/cmake/amd_comgr
   )
   MESSAGE(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

  include_directories("$<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>")
endif()
include_directories(${ROCclr_DIR})
include_directories(${ROCclr_DIR}/include)
include_directories(${ROCclr_DIR}/compiler/lib)
include_directories(${ROCclr_DIR}/compiler/lib/include)
include_directories(${ROCclr_DIR}/compiler/lib/backends/common)

include_directories(${ROCclr_DIR}/elf/utils/libelf)
include_directories(${ROCclr_DIR}/elf/utils/common)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("-Wno-ignored-attributes")
endif()

if(UNIX)
    set(ADDITIONAL_LIBRARIES glibc_functions.cpp)
else()
    set(ADDITIONAL_LIBRARIES "")
endif()

set(amdocl64_src
  cl_memobj.cpp
  cl_program.cpp
  cl_sdi_amd.cpp
  cl_device.cpp
  cl_debugger_amd.cpp
  cl_svm.cpp
  cl_sampler.cpp
  cl_thread_trace_amd.cpp
  cl_counter.cpp
  cl_d3d10.cpp
  cl_d3d9.cpp
  cl_command.cpp
  cl_platform_amd.cpp
  cl_event.cpp
  cl_lqdflash_amd.cpp
  cl_kernel_info_amd.cpp
  cl_execute.cpp
  cl_gl.cpp
  cl_icd.cpp
  cl_pipe.cpp
  cl_d3d11.cpp
  cl_context.cpp
  cl_profile_amd.cpp
  cl_p2p_amd.cpp
  ${ADDITIONAL_LIBRARIES}
)

add_library(amdocl64 SHARED ${amdocl64_src})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(amdocl64 amdrocclr_static Threads::Threads dl)
