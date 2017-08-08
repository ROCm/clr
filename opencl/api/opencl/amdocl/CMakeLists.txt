set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all -Wl,--version-script=${CMAKE_SOURCE_DIR}/api/opencl/amdocl/amdocl.map")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-DLINUX -D__x86_64__ -D__AMD64__ -DUNIX_OS -DqLittleEndian -DOPENCL_MAJOR=2 -DOPENCL_MINOR=0 -DWITH_AQL -DWITH_ONLINE_COMPILER -DATI_OS_LINUX -DATI_ARCH_X86 -DLITTLEENDIAN_CPU -DATI_BITS_64 -DATI_COMP_GCC -DWITH_HSA_DEVICE -DWITH_TARGET_AMDGCN -DOPENCL_EXPORTS -DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS -DVEGA10_ONLY=false -DWITH_LIGHTNING_COMPILER)

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

include_directories(${CMAKE_SOURCE_DIR}/runtime)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers)
include_directories(${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.0)
include_directories(${CMAKE_SOURCE_DIR}/compiler/driver/src)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/include)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/backends/common)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/libelf)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/common)
include_directories(${CMAKE_SOURCE_DIR}/compiler/tools)
 
add_library(amdocl64 SHARED
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
  $<TARGET_OBJECTS:oclruntime>
  $<TARGET_OBJECTS:oclrocm>
)

target_link_libraries(amdocl64 opencl_driver oclelf pthread dl ${ROCR_LIBRARIES})
