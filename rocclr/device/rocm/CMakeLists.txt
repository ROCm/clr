# Copyright (C) 2017-2021 Advanced Micro Devices, Inc. All Rights Reserved.
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

add_library(oclrocm OBJECT
  roccounters.cpp
  rocprintf.cpp
  rocprogram.cpp
  rocmemory.cpp
  rocdevice.cpp
  rocblit.cpp
  rockernel.cpp
  rocvirtual.cpp
  rocglinterop.cpp
  rocappprofile.cpp
  rocsettings.cpp
  rocschedcl.cpp
  rocsignal.cpp
)

target_include_directories(oclrocm
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    # GL and EGL headers
    $<BUILD_INTERFACE:${OPENCL_DIR}/khronos/headers>
    $<BUILD_INTERFACE:${OPENCL_DIR}/khronos/headers/opencl2.2>
    $<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>
  PRIVATE
    ${OPENCL_DIR}
    ${PROJECT_SOURCE_DIR}/compiler/lib
    ${PROJECT_SOURCE_DIR}/compiler/lib/include
    ${PROJECT_SOURCE_DIR}/compiler/lib/backends/common
    ${PROJECT_SOURCE_DIR}/elf
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ROCM_OCL_INCLUDES}
    $<TARGET_PROPERTY:hsa-runtime64::hsa-runtime64,INTERFACE_INCLUDE_DIRECTORIES>)

option(BUILD_HMM "Build HMM support" OFF)
if (BUILD_HMM)
  target_compile_definitions(oclrocm
    PRIVATE AMD_HMM_SUPPORT)
endif()

if(USE_COMGR_LIBRARY)
  if(${BUILD_SHARED_LIBS})
    target_compile_definitions(oclrocm PRIVATE USE_COMGR_LIBRARY COMGR_DYN_DLL)
  else()
    target_compile_definitions(oclrocm PRIVATE USE_COMGR_LIBRARY)
  endif()
endif()

if (UNIX)
  find_library(LIBNUMA numa)
  if (LIBNUMA)
    target_compile_definitions(oclrocm PRIVATE ROCCLR_SUPPORT_NUMA_POLICY)
    message(STATUS "Found: ${LIBNUMA}")
  endif()
endif()

set_target_properties(oclrocm PROPERTIES POSITION_INDEPENDENT_CODE ON)
