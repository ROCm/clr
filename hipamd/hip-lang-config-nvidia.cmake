# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

add_library(hip-lang::device INTERFACE IMPORTED)
add_library(hip-lang::host INTERFACE IMPORTED)
add_library(hip-lang::amdhip64 INTERFACE IMPORTED)

# Find the hip-lang config file path with symlinks resolved
# RealPath: /opt/rocm-ver/lib/cmake/hip-lang/hip-lang-config.cmake
# Go 4 level up to get IMPORT PREFIX
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" REALPATH)
get_filename_component(_IMPORT_PREFIX "${_DIR}/../../../../" ABSOLUTE)


set_target_properties(hip-lang::device PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$<$<COMPILE_LANGUAGE:HIP>:${_IMPORT_PREFIX}/include>"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<$<COMPILE_LANGUAGE:HIP>:${_IMPORT_PREFIX}/include>"
)

set_target_properties(hip-lang::amdhip64 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$<$<COMPILE_LANGUAGE:HIP>:${_IMPORT_PREFIX}/include>"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<$<COMPILE_LANGUAGE:HIP>:${_IMPORT_PREFIX}/include>"
)

set(_CMAKE_HIP_DEVICE_RUNTIME_TARGET "hip-lang::device")
