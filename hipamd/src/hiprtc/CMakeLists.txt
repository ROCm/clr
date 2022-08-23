# Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This project builds hiprtc
# If ever this is to be a different lib living in different folder
# Please read this part
# Depends on: rocclr, so find_package(rocclr) will be required
# Building hip header requires hip include folders with hip_version.h

cmake_minimum_required(VERSION 3.16.1)
option(BUILD_SHARED_LIBS "Build the shared library" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(BUILD_SHARED_LIBS)
  add_library(hiprtc SHARED)
  # Windows doesn't have a strip utility, so CMAKE_STRIP won't be set.
  if((CMAKE_BUILD_TYPE STREQUAL "Release") AND NOT ("${CMAKE_STRIP}" STREQUAL ""))
    add_custom_command(TARGET hiprtc POST_BUILD COMMAND ${CMAKE_STRIP} $<TARGET_FILE:hiprtc>)
  endif()
else()
  add_library(hiprtc STATIC $<TARGET_OBJECTS:rocclr>)
endif()

set_target_properties(hiprtc PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(WIN32)
  if(${HIP_LIB_VERSION_MAJOR} VERSION_GREATER 9)
    set(HIP_MAJOR_VERSION "${HIP_LIB_VERSION_MAJOR}")
  else()
    set(HIP_MAJOR_VERSION "0${HIP_LIB_VERSION_MAJOR}")
  endif()
  set(HIP_MINOR_VERSION "0${HIP_LIB_VERSION_MINOR}")
endif()

if(BUILD_SHARED_LIBS)
  if(WIN32)
    set_target_properties(hiprtc PROPERTIES
      OUTPUT_NAME "hiprtc${HIP_MAJOR_VERSION}${HIP_MINOR_VERSION}"
      ARCHIVE_OUTPUT_NAME "hiprtc")
  else()
    set_target_properties(hiprtc PROPERTIES
      VERSION ${HIP_LIB_VERSION_STRING}
      SOVERSION ${HIP_LIB_VERSION_MAJOR})
  endif()
endif()

# Create HIPRTC object library
if(BUILD_SHARED_LIBS)
  add_library(hiprtcobject OBJECT hiprtc.cpp hiprtcComgrHelper.cpp hiprtcInternal.cpp)
endif()

set_target_properties(hiprtcobject PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(hiprtcobject
  PRIVATE
  ${HIP_COMMON_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include)

if(BUILD_SHARED_LIBS)
  if(WIN32)
    target_sources(hiprtc PRIVATE hiprtc.def)
  else()
    target_link_libraries(hiprtcobject PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hiprtc.map.in")
    set_target_properties(hiprtcobject PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/hiprtc.map.in")
  endif()
endif()

if(WIN32)
  target_link_libraries(hiprtc PRIVATE Dbghelp.lib)
endif()

target_link_libraries(hiprtcobject PRIVATE ${CMAKE_DL_LIBS})

if(BUILD_SHARED_LIBS)
  target_link_libraries(hiprtcobject PRIVATE rocclr)
else()
  target_compile_definitions(hiprtcobject PRIVATE $<TARGET_PROPERTY:rocclr,COMPILE_DEFINITIONS>)
  target_include_directories(hiprtcobject PRIVATE $<TARGET_PROPERTY:rocclr,INCLUDE_DIRECTORIES>)
endif()

target_compile_definitions(hiprtcobject PRIVATE __HIP_PLATFORM_AMD__)

add_to_config(_versionInfo HIP_PACKAGING_VERSION_PATCH)
add_to_config(_versionInfo CPACK_DEBIAN_PACKAGE_RELEASE)
add_to_config(_versionInfo CPACK_RPM_PACKAGE_RELEASE)

add_to_config(_versionInfo HIP_VERSION_MAJOR)
add_to_config(_versionInfo HIP_VERSION_MINOR)
add_to_config(_versionInfo HIP_VERSION_PATCH)
add_to_config(_versionInfo HIP_VERSION_GITHASH)

# Enable preprocessed hiprtc-builtins library
include(HIPRTC RESULT_VARIABLE HIPRTC_CMAKE)
# Requires clang and llvm-mc to create this library.
find_package(LLVM REQUIRED CONFIG PATHS ${ROCM_PATH}/llvm)
find_package(Clang REQUIRED CONFIG PATHS ${ROCM_PATH}/llvm)
set(HIPRTC_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip_rtc_gen")
set(HIPRTC_GEN_HEADER "${HIPRTC_GEN_DIR}/hipRTC_header.h")
set(HIPRTC_GEN_MCIN "${HIPRTC_GEN_DIR}/hipRTC_header.mcin")
set(HIPRTC_GEN_PREPROCESSED "${HIPRTC_GEN_DIR}/hipRTC")
set(HIPRTC_GEN_OBJ "${HIPRTC_GEN_DIR}/hipRTC_header${CMAKE_CXX_OUTPUT_EXTENSION}")
set(HIPRTC_WARP_FUNCS "${PROJECT_SOURCE_DIR}/include/hip/amd_detail/amd_warp_functions.h")

# Generate required HIPRTC files.
FILE(MAKE_DIRECTORY ${HIPRTC_GEN_DIR})
generate_hiprtc_header("${HIPRTC_GEN_HEADER}")
generate_hiprtc_mcin("${HIPRTC_GEN_MCIN}" "${HIPRTC_GEN_PREPROCESSED}")

# Generate HIPRTC Builtins Preprocessed Object.
# Note: second command appends define macros at build time.
# FIXME: --hip-version forced to 3.6 to use clang headers, until Windows versioning is fixed.
add_custom_command(
  OUTPUT ${HIPRTC_GEN_PREPROCESSED}
  COMMAND $<TARGET_FILE:clang> -O3 --rocm-path=${PROJECT_SOURCE_DIR}/include/.. -std=c++17 -nogpulib --hip-version=3.6 -isystem ${HIP_COMMON_INCLUDE_DIR} -isystem ${PROJECT_SOURCE_DIR}/include -isystem ${PROJECT_BINARY_DIR}/include -isystem ${CMAKE_CURRENT_SOURCE_DIR}/include --cuda-device-only -D__HIPCC_RTC__ -x hip ${HIPRTC_GEN_HEADER} -E -o ${HIPRTC_GEN_PREPROCESSED}
  COMMAND ${CMAKE_COMMAND} -DHIPRTC_ADD_MACROS=1 -DHIPRTC_WARP_HEADER_FILE=${HIPRTC_WARP_FUNCS} -DHIPRTC_PREPROCESSED_FILE=${HIPRTC_GEN_PREPROCESSED} -P ${HIPRTC_CMAKE}
  DEPENDS clang ${HIPRTC_GEN_HEADER})
add_custom_command(
  OUTPUT ${HIPRTC_GEN_OBJ}
  COMMAND $<TARGET_FILE:llvm-mc> -o ${HIPRTC_GEN_OBJ} ${HIPRTC_GEN_MCIN} --filetype=obj
  DEPENDS llvm-mc ${HIPRTC_GEN_PREPROCESSED} ${HIPRTC_GEN_MCIN})

# Create hiprtc-builtins library.
add_library(hiprtc-builtins ${HIPRTC_GEN_OBJ})
set_target_properties(hiprtc-builtins PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  LINKER_LANGUAGE CXX
  VERSION ${HIP_LIB_VERSION_STRING})

# Windows and Linux have different naming conventions.
if(WIN32)
  # Windows uses DEF file to determine which symbols to expose.
  target_sources(hiprtc-builtins PRIVATE hiprtc-builtins.def)
  set_target_properties(hiprtc-builtins PROPERTIES
    OUTPUT_NAME "hiprtc-builtins${HIP_MAJOR_VERSION}${HIP_MINOR_VERSION}"
    ARCHIVE_OUTPUT_NAME "hiprtc-builtins")
  # Since ${HIPRTC_GEN_OBJ} was manually generated with llvm-mc, /MT did not embed
  # libcmt.lib inside of the obj. So we need to manually set it as defaultlib.
  target_link_options(hiprtc-builtins PRIVATE "LINKER:/DEFAULTLIB:libcmt")
else()
  # SOVERSION is only supported on Linux.
  set_target_properties(hiprtc-builtins PROPERTIES
    OUTPUT_NAME "hiprtc-builtins"
    SOVERSION ${HIP_LIB_VERSION_MAJOR})
endif()

# Test the header file works with simple compilation.
add_custom_command(
  OUTPUT ${HIPRTC_GEN_DIR}/tmp.bc
  COMMAND $<TARGET_FILE:clang> -O3 --rocm-path=${PROJECT_SOURCE_DIR}/include/.. -std=c++14 -nogpulib -nogpuinc -emit-llvm -c -isystem ${HIP_COMMON_INCLUDE_DIR} -isystem ${PROJECT_BINARY_DIR}/include -isystem ${CMAKE_CURRENT_SOURCE_DIR}/include --cuda-device-only -D__HIPCC_RTC__ --offload-arch=gfx906 -x hip-cpp-output ${HIPRTC_GEN_PREPROCESSED} -o ${HIPRTC_GEN_DIR}/tmp.bc
  DEPENDS clang ${HIPRTC_GEN_PREPROCESSED})

target_link_libraries(hiprtcobject PRIVATE ${HIPRTC_GEN_OBJ})
target_compile_definitions(hiprtcobject PRIVATE __HIP_ENABLE_RTC)
target_link_libraries(hiprtc PRIVATE hiprtcobject)

# As a temporary workaround adding hiprtc sources to amdhip64 using target_sources, to avoid
# jenkins failure. Once, jenkins upgrades to 3.21 or higher, hiprtcobject can be appended to
# HIPRTC_OBJECTS below which links to amdhip64
if(NOT WIN32)
  target_sources(amdhip64 PRIVATE hiprtc.cpp hiprtcComgrHelper.cpp hiprtcInternal.cpp)
endif()

list(APPEND HIPRTC_OBJECTS ${HIPRTC_GEN_OBJ})
set(HIPRTC_OBJECTS ${HIPRTC_OBJECTS} PARENT_SCOPE)

add_dependencies(hiprtc hiprtc-builtins)
install(TARGETS hiprtc-builtins
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

INSTALL(TARGETS hiprtc
  EXPORT hiprtc-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
