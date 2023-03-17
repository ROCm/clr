# Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

project(clr)
cmake_minimum_required(VERSION 3.16.8)
include(GNUInstallDirs)

##########
# Defaults
##########
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
endif()
message(STATUS "ROCM Installation path(ROCM_PATH): ${ROCM_PATH}")

if(UNIX)
    set(CLR_DEFAULT_INSTALL_PREFIX "${ROCM_PATH}")
endif()
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CLR_DEFAULT_INSTALL_PREFIX} CACHE PATH "Installation path for HIP and OCL" FORCE)
endif()
set(CPACK_INSTALL_PREFIX ${CLR_DEFAULT_INSTALL_PREFIX} CACHE PATH "Package Installation path for HIP and OCL")

if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    message(STATUS "HIP and OCL will be installed in: " ${CMAKE_INSTALL_PREFIX})
else()
    message(FATAL_ERROR "Don't know where to install HIP and OCL. Please specify absolute path using -DCMAKE_INSTALL_PREFIX")
endif()

set(CPACK_SET_DESTDIR ON CACHE BOOL "Installer package will install HIP and OCL to CMAKE_INSTALL_PREFIX instead of CPACK_PACKAGING_INSTALL_PREFIX")
if (NOT CPACK_SET_DESTDIR)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Default installation path of HIP and OCL installer package")
endif()

option(CLR_BUILD_HIP "Build HIP" ON)
option(CLR_BUILD_OCL "Build OCL" OFF)

#############
# Build steps
#############
if(CLR_BUILD_HIP OR CLR_BUILD_OCL)
    add_subdirectory(rocclr)
endif()
if(CLR_BUILD_HIP)
    add_subdirectory(hipamd)
endif()
if(CLR_BUILD_OCL)
    add_subdirectory(opencl)
endif()
