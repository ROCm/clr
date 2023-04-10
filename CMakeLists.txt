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

##########
# Defaults
##########
option(CLR_BUILD_HIP "Build HIP" OFF)
option(CLR_BUILD_OCL "Build OCL" OFF)

#############
# Build steps
#############
if(CLR_BUILD_HIP)
    # Determine HIP_PLATFORM
    set(__HIPCONFIG_EXECUTABLE__ ${HIP_COMMON_DIR}/bin/hipconfig)
    if(NOT DEFINED HIP_PLATFORM)
        if(NOT DEFINED ENV{HIP_PLATFORM})
            execute_process(COMMAND ${__HIPCONFIG_EXECUTABLE__} --platform
                OUTPUT_VARIABLE HIP_PLATFORM
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        else()
            set(HIP_PLATFORM $ENV{HIP_PLATFORM} CACHE STRING "HIP Platform")
        endif()
    endif()
endif()

if((CLR_BUILD_HIP AND HIP_PLATFORM STREQUAL "amd") OR CLR_BUILD_OCL)
    add_subdirectory(rocclr)
elseif(HIP_PLATFORM STREQUAL "amd")
    message(FATAL_ERROR "Please enable building of one or more of the below runtimes:\n- HIP (-DCLR_BUILD_HIP=ON)\n- OpenCL (-DCLR_BUILD_OCL-ON)")
endif()
if(CLR_BUILD_HIP)
    add_subdirectory(hipamd)
endif()
if(CLR_BUILD_OCL)
    add_subdirectory(opencl)
endif()
