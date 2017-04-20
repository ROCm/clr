cmake_minimum_required(VERSION 3.4.3)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
  set(PROJ_VERSION VERSION 1.5.0)
endif()

project(OpenCL-ROCm)

set(LLVM_INCLUDE_TESTS CACHE BOOL OFF)
set(LLVM_TARGETS_TO_BUILD "AMDGPU" "X86" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS CACHE STRING "clang")

add_subdirectory(compiler/llvm)

find_package(LLVM REQUIRED CONFIG PATHS ${CMAKE_BINARY_DIR}/compiler/llvm NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
# TODO: add find_package(Clang) and use LLVM/Clang variables got from their config
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/tools/clang/include)
include_directories(${CMAKE_BINARY_DIR}/compiler/llvm/tools/clang/include)

# TODO: move AMDGPU.h header to include folder
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/lib/Target/AMDGPU)

add_subdirectory(compiler/driver)

set(CMAKE_C_COMPILER "${PROJECT_BINARY_DIR}/compiler/llvm/bin/clang")
set(BUILD_HC_LIB CACHE BOOL OFF)
add_subdirectory(library/amdgcn)
add_dependencies(ocml_lib_bc clang llvm-link llvm-objdump lld)