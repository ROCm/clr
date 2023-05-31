# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.16.8)

set(OPENCL ${PROJECT_NAME})
set(OPENCL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(OPENCL_WRAPPER_DIR ${OPENCL_BUILD_DIR}/wrapper_dir)
set(OPENCL_WRAPPER_INC_DIR ${OPENCL_WRAPPER_DIR}/include/CL)
set(OPENCL_WRAPPER_BIN_DIR ${OPENCL_WRAPPER_DIR}/bin)
set(OPENCL_WRAPPER_LIB_DIR ${OPENCL_WRAPPER_DIR}/lib)

#Function to generate header template file
function(create_header_template)
    file(WRITE ${OPENCL_WRAPPER_DIR}/header.hpp.in "/*
    Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the \"Software\"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */

#ifndef @include_guard@
#define @include_guard@

#ifndef ROCM_HEADER_WRAPPER_WERROR
#define ROCM_HEADER_WRAPPER_WERROR @deprecated_error@
#endif
#if ROCM_HEADER_WRAPPER_WERROR  /* ROCM_HEADER_WRAPPER_WERROR 1 */
#error \"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with CL\"
#else    /* ROCM_HEADER_WRAPPER_WERROR 0 */
#if defined(__GNUC__)
#warning \"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with CL\"
#else
#pragma message(\"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with CL\")
#endif
#endif  /* ROCM_HEADER_WRAPPER_WERROR */

@include_statements@

#endif")
endfunction()


#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${OPENCL_WRAPPER_INC_DIR})
  set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/khronos/headers/opencl2.2/CL )
  #find all header files from CL folder
  file(GLOB include_files ${CMAKE_CURRENT_SOURCE_DIR}/khronos/headers/opencl2.2/CL/*)
  #remove files that are not required in package
  list(REMOVE_ITEM include_files ${HEADER_DIR}/cl_egl.h ${HEADER_DIR}/cl_dx9_media_sharing.h ${HEADER_DIR}/cl_d3d11.h ${HEADER_DIR}/cl_d3d10.h)
  #Generate wrapper header files
  foreach(header_file ${include_files})
    # set include guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "${include_guard}OPENCL_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "${include_statements}#include \"../../../${CMAKE_INSTALL_INCLUDEDIR}/CL/${file_name}\"\n")
    configure_file(${OPENCL_WRAPPER_DIR}/header.hpp.in ${OPENCL_WRAPPER_INC_DIR}/${file_name})
    unset(include_guard)
    unset(include_statements)
  endforeach()

endfunction()

#function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${OPENCL_WRAPPER_BIN_DIR})
  set(file_name "clinfo")
  add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${OPENCL_WRAPPER_BIN_DIR}/${file_name})
endfunction()

#function to create symlink to libraries
function(create_library_symlink)
  if(BUILD_ICD)
    file(MAKE_DIRECTORY ${OPENCL_WRAPPER_LIB_DIR})
    set(LIB_OPENCL "libOpenCL.so")
    set(MAJ_VERSION "${OPENCL_LIB_VERSION_MAJOR}")
    set(SO_VERSION "${OPENCL_LIB_VERSION_STRING}")
    set(library_files "${LIB_OPENCL}"  "${LIB_OPENCL}.${MAJ_VERSION}" "${LIB_OPENCL}.${SO_VERSION}")

    foreach(file_name ${library_files})
      add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${OPENCL_WRAPPER_LIB_DIR}/${file_name})
    endforeach()
  endif()
  if(BUILD_SHARED_LIBS)
    set(LIB_AMDDOC "libamdocl64.so")
  else()
    set(LIB_AMDDOC "libamdocl64.a")
  endif()
  set(file_name "${LIB_AMDDOC}")
  add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${OPENCL_WRAPPER_DIR}/${file_name})
endfunction()

#Creater a template for header file
create_header_template()
#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${OPENCL_WRAPPER_INC_DIR} DESTINATION ${OPENCL}/include COMPONENT dev)
# Create symlink to binaries
create_binary_symlink()
install(DIRECTORY ${OPENCL_WRAPPER_BIN_DIR}  DESTINATION ${OPENCL} COMPONENT binary)

option(BUILD_SHARED_LIBS "Build the shared library" ON)
# Create symlink to libraries
create_library_symlink()
if(BUILD_ICD)
  install(DIRECTORY ${OPENCL_WRAPPER_LIB_DIR}  DESTINATION ${OPENCL} COMPONENT icd)
endif()
if(BUILD_SHARED_LIBS)
  install(FILES ${OPENCL_WRAPPER_DIR}/libamdocl64.so  DESTINATION ${OPENCL}/lib COMPONENT binary)
else()
  install(FILES ${OPENCL_WRAPPER_DIR}/libamdocl64.a  DESTINATION ${OPENCL}/lib COMPONENT binary)
endif()
