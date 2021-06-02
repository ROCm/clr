# Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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

if(AMD_UGL_FOUND)
  return()
endif()

find_path(AMD_UGL_INCLUDE_DIR GL/glx.h
  HINTS
    ${AMD_DRIVERS_PATH}
  PATHS
    # p4 repo layout
    ${CMAKE_SOURCE_DIR}/drivers
    ${CMAKE_SOURCE_DIR}/../drivers
    ${CMAKE_SOURCE_DIR}/../../drivers
    # github ent repo layout
    ${CMAKE_SOURCE_DIR}/drivers/drivers
    ${CMAKE_SOURCE_DIR}/../drivers/drivers
    ${CMAKE_SOURCE_DIR}/../../drivers/drivers
  PATH_SUFFIXES
    ugl/inc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD_UGL
  "\nAMD UGL not found"
  AMD_UGL_INCLUDE_DIR)
mark_as_advanced(AMD_UGL_INCLUDE_DIR)

set(AMD_UGL_INCLUDE_DIRS ${AMD_UGL_INCLUDE_DIR} ${ROCCLR_SRC_DIR}/device/gpu/gslbe/src/rt)
mark_as_advanced(AMD_UGL_INCLUDE_DIRS)
