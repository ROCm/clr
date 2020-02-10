
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# Some linker flags, -Wl,--unresolved-symbols=report-all being one of them, requires -shared-libasan if using -fsanitize=address.
# If -Wl,--unresolved-symbols=report-all is removed from CMAKE_SHARED_LINKER_FLAGS, then update this snippet
if(CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared-libasan")
endif()

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all -Wl,--version-script=${CMAKE_SOURCE_DIR}/amdocl/amdocl.map")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(ROCT REQUIRED)
find_package(ROCR REQUIRED)

add_definitions(-DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

include_directories(${ROCR_INCLUDES})

if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")
	  include_directories(${LLVM_INCLUDES})
 endif() # if (DEFINED LLVM_INCLUDES AND NOT ${LLVM_INCLUDES} STREQUAL "")

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
else()
	#  include_directories(${CMAKE_SOURCE_DIR}/compiler/driver/src)
endif()
include_directories(${VDI_DIR})
include_directories(${VDI_DIR}/include)
include_directories(${VDI_DIR}/compiler/lib)
include_directories(${VDI_DIR}/compiler/lib/include)
include_directories(${VDI_DIR}/compiler/lib/backends/common)
#include_directories(${VDI_DIR}/compiler/lib/loaders)
include_directories(${VDI_DIR}/elf/utils/libelf)
include_directories(${VDI_DIR}/elf/utils/common)
#include_directories(${VDI_DIR}/compiler/tools)

if(UNIX)
    set(ADDITIONAL_LIBRARIES glibc_functions.cpp)
else()
    set(ADDITIONAL_LIBRARIES "")
endif()

add_library(amdocl64_obj OBJECT
  cl_memobj.cpp
  cl_program.cpp
  cl_sdi_amd.cpp
  cl_device.cpp
  cl_debugger_amd.cpp
  cl_svm.cpp
  cl_sampler.cpp
  cl_thread_trace_amd.cpp
  cl_object.cpp
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
  #  $<TARGET_OBJECTS:oclruntime>
  #$<TARGET_OBJECTS:oclrocm>
  ${ADDITIONAL_LIBRARIES}
)

add_library(amdocl64 SHARED
     $<TARGET_OBJECTS:amdocl64_obj>
)
add_library(amdocl64_static STATIC
     $<TARGET_OBJECTS:amdocl64_obj>
)

add_library(amdvdi_static STATIC IMPORTED)
set_target_properties(amdvdi_static PROPERTIES IMPORTED_LOCATION "${LIBVDI_STATIC_DIR}/libamdvdi_static.a")

#add_library(oclelf STATIC IMPORTED)
#set_target_properties(oclelf PROPERTIES IMPORTED_LOCATION "${LIBOCLELF_STATIC_PATH}/liboclelf.a")
target_link_libraries(amdocl64_static amdvdi_static pthread dl ${ROCT_LIBRARIES} ${ROCR_LIBRARIES})

target_link_libraries(amdocl64 amdvdi_static pthread dl ${ROCT_LIBRARIES} ${ROCR_LIBRARIES})
