cmake_minimum_required(VERSION 3.4.3)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
  set(PROJ_VERSION VERSION 1.5.0)
endif()

project(OpenCL-ROCm)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TARGETS_TO_BUILD "AMDGPU" CACHE STRING "")
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")

# override default option value in library and driver
set(GENERIC_IS_ZERO ON CACHE BOOL ON FORCE)

add_subdirectory(compiler/llvm)

find_package(LLVM REQUIRED CONFIG PATHS ${CMAKE_BINARY_DIR}/compiler/llvm NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
# TODO: add find_package for Clang and lld, and also use LLVM/Clang variables got from their config
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/tools/clang/include)
include_directories(${CMAKE_BINARY_DIR}/compiler/llvm/tools/clang/include)
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/tools/lld/include)

# TODO: move AMDGPU.h header to include folder
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/lib/Target/AMDGPU)
include_directories(${CMAKE_BINARY_DIR}/compiler/llvm/lib/Target/AMDGPU)

add_subdirectory(compiler/driver)

set(BUILD_HC_LIB OFF CACHE BOOL "")
set(ROCM_DEVICELIB_INCLUDE_TESTS OFF CACHE BOOL "")
set(AMDGCN_TARGETS_LIB_LIST "AMDGCN_LIB_TARGETS")
set(AMDGPU_TARGET_TRIPLE "amdgcn-amd-amdhsa-amdgizcl")
add_subdirectory(library/amdgcn)

add_subdirectory(compiler/lib/loaders/elf/utils/libelf)

find_package(ROCR REQUIRED)
add_subdirectory(runtime)

set(OPENCL_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers
  ${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.2)

add_subdirectory(api/opencl/amdocl)
add_subdirectory(api/opencl/khronos/icd)

add_subdirectory(tools/clinfo)

install(PROGRAMS $<TARGET_FILE:OpenCL>
        DESTINATION lib)

install(DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.2"
        DESTINATION include
        USE_SOURCE_PERMISSIONS
        PATTERN cl_egl.h EXCLUDE)
