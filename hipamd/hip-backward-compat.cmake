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

set(HIP_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(HIP_WRAPPER_DIR ${HIP_BUILD_DIR}/wrapper_dir)
set(HIP_WRAPPER_INC_DIR ${HIP_WRAPPER_DIR}/include/hip)
set(HIP_WRAPPER_BIN_DIR ${HIP_WRAPPER_DIR}/bin)
set(HIP_WRAPPER_LIB_DIR ${HIP_WRAPPER_DIR}/lib)
set(HIP_WRAPPER_CMAKE_DIR ${HIP_WRAPPER_DIR}/cmake)
set(HIP_WRAPPER_FINDHIP_DIR ${HIP_WRAPPER_DIR}/FindHIP)
set(HIP_SRC_INC_DIR ${HIP_SRC_PATH}/include/hip)
set(HIP_SRC_BIN_DIR ${HIP_SRC_PATH}/bin)
set(HIP_INFO_FILE ".hipInfo")

#Function to set actual file contents in wrapper files
#Some components grep for the contents in the file
function(set_file_contents input_file)
    set(hashzero_check "#if 0
/* The following is a copy of the original file for the benefit of build systems which grep for values
 * in this file rather than preprocess it. This is just for backward compatibility */")

    file(READ ${input_file} file_contents)
    set(hash_endif "#endif")
    get_filename_component(file_name ${input_file} NAME)
    configure_file(${HIP_SRC_PATH}/header_template.hpp.in ${HIP_WRAPPER_INC_DIR}/${file_name})
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
#create respecitve folder in /opt/rocm/hip
  file(MAKE_DIRECTORY ${HIP_WRAPPER_INC_DIR}/amd_detail)
  file(MAKE_DIRECTORY ${HIP_WRAPPER_INC_DIR}/nvidia_detail)

  #find all header files from include/hip
  file(GLOB include_files ${HIP_BUILD_DIR}/include/hip/*.h)
  #Convert the list of files into #includes
  foreach(header_file ${include_files})
    # set include guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "HIP_WRAPPER_INCLUDE_HIP_${INC_GAURD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "#include \"../../../${CMAKE_INSTALL_INCLUDEDIR}/hip/${file_name}\"\n")
    if(${file_name} STREQUAL "hip_version.h")
      set_file_contents(${header_file})
    else()
      configure_file(${HIP_SRC_PATH}/header_template.hpp.in ${HIP_WRAPPER_INC_DIR}/${file_name})
    endif()
  endforeach()

  #find all header files from include/hip/amd_detail
  file(GLOB include_files ${HIP_SRC_INC_DIR}/amd_detail/*)
  #Convert the list of files into #includes
  foreach(header_file ${include_files})
    # set include guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "HIP_WRAPPER_INCLUDE_HIP_AMD_DETAIL_${INC_GAURD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "#include \"../../../../${CMAKE_INSTALL_INCLUDEDIR}/hip/amd_detail/${file_name}\"\n")

    configure_file(${HIP_SRC_PATH}/header_template.hpp.in ${HIP_WRAPPER_INC_DIR}/amd_detail/${file_name})
  endforeach()

  #find all header files from include/hip/nvidia_detail
  file(GLOB include_files ${HIP_SRC_INC_DIR}/nvidia_detail/*)
  #Convert the list of files into #includes
  foreach(header_file ${include_files})
    # set include guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "HIP_WRAPPER_INCLUDE_HIP_NVIDIA_DETAIL_${INC_GAURD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "#include \"../../../../${CMAKE_INSTALL_INCLUDEDIR}/hip/nvidia_detail/${file_name}\"\n")

    configure_file(${HIP_SRC_PATH}/header_template.hpp.in ${HIP_WRAPPER_INC_DIR}/nvidia_detail/${file_name})
  endforeach()

endfunction()

#function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${HIP_WRAPPER_BIN_DIR})
  #get all  binaries
  file(GLOB binary_files ${HIP_SRC_BIN_DIR}/*)
  #Add .hipVersion to binary list
  set(binary_files "${binary_files}" ".hipVersion")
  foreach(binary_file ${binary_files})
    get_filename_component(file_name ${binary_file} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${HIP_WRAPPER_BIN_DIR}/${file_name})
  endforeach()

  unset(binary_files)
  file(GLOB binary_files ${HIP_BUILD_DIR}/bin/*)
  foreach(binary_file ${binary_files})
    get_filename_component(file_name ${binary_file} NAME)
    if(WIN32)
      add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${HIP_WRAPPER_BIN_DIR}/${file_name})

    else()
      if( NOT ${file_name} MATCHES ".bat$")
        add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${HIP_WRAPPER_BIN_DIR}/${file_name})
      endif()#end of bat file check
    endif()#end of OS check
  endforeach()
endfunction()

#function to create symlink to libraries
function(create_library_symlink)
  file(MAKE_DIRECTORY ${HIP_WRAPPER_LIB_DIR})
  if(BUILD_SHARED_LIBS)
    set(LIB_AMDHIP "libamdhip64.so")
    set(MAJ_VERSION "${HIP_LIB_VERSION_MAJOR}")
    set(SO_VERSION "${HIP_LIB_VERSION_STRING}")
    set(library_files "${LIB_AMDHIP}"  "${LIB_AMDHIP}.${MAJ_VERSION}" "${LIB_AMDHIP}.${SO_VERSION}")
    set(LIB_HIPRTC "libhiprtc-builtins.so")
    set(library_files "${library_files}" "${LIB_HIPRTC}"  "${LIB_HIPRTC}.${MAJ_VERSION}" "${LIB_HIPRTC}.${SO_VERSION}" )    
    set(LIB_RTC "libhiprtc.so")
    set(library_files "${library_files}" "${LIB_RTC}"  "${LIB_RTC}.${MAJ_VERSION}" "${LIB_RTC}.${SO_VERSION}" )
  else()
    set(library_files "libamdhip64.a")
  endif()

  foreach(file_name ${library_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${HIP_WRAPPER_LIB_DIR}/${file_name})
  endforeach()
  #Add symlink for .hipInfo
  set(file_name ${HIP_INFO_FILE})
  add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${HIP_WRAPPER_LIB_DIR}/${file_name})
endfunction()

function(create_cmake_symlink)
  file(MAKE_DIRECTORY ${HIP_WRAPPER_CMAKE_DIR}/hip)

  #create symlink to all hip config files
  file(GLOB config_files ${HIP_BUILD_DIR}/hip-config*)
  foreach(config_name ${config_files})
    get_filename_component(file_name ${config_name} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../../../${CMAKE_INSTALL_LIBDIR}/cmake/hip/${file_name} ${HIP_WRAPPER_CMAKE_DIR}/hip/${file_name})
  endforeach()
  unset(config_files)

  #create symlink to hip-lang
  file(MAKE_DIRECTORY ${HIP_WRAPPER_CMAKE_DIR}/hip-lang)
  file(GLOB config_files ${HIP_BUILD_DIR}/src/hip-lang-config*)
  foreach(config_name ${config_files})
    get_filename_component(file_name ${config_name} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../../../${CMAKE_INSTALL_LIBDIR}/cmake/hip-lang/${file_name} ${HIP_WRAPPER_CMAKE_DIR}/hip-lang/${file_name})
  endforeach()
  unset(config_files)

  #create symlink to hiprtc config files
  file(MAKE_DIRECTORY ${HIP_WRAPPER_CMAKE_DIR}/hiprtc)
  file(GLOB config_files ${HIP_BUILD_DIR}/hiprtc-config*)
  foreach(config_name ${config_files})
    get_filename_component(file_name ${config_name} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../../../${CMAKE_INSTALL_LIBDIR}/cmake/hiprtc/${file_name} ${HIP_WRAPPER_CMAKE_DIR}/hiprtc/${file_name})
  endforeach()
  unset(config_files)

  #create symlink to FindHIP
  file(MAKE_DIRECTORY ${HIP_WRAPPER_FINDHIP_DIR}/FindHIP)
  file(GLOB config_files ${HIP_BUILD_DIR}/cmake/FindHIP/*.cmake)
  foreach(config_name ${config_files})
    get_filename_component(file_name ${config_name} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../../${CMAKE_INSTALL_LIBDIR}/cmake/hip/FindHIP/${file_name} ${HIP_WRAPPER_FINDHIP_DIR}/FindHIP/${file_name})
  endforeach()
  unset(config_files)

  file(GLOB config_files ${HIP_BUILD_DIR}/cmake/*.cmake)
  foreach(config_name ${config_files})
    get_filename_component(file_name ${config_name} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/cmake/hip/${file_name} ${HIP_WRAPPER_FINDHIP_DIR}/${file_name})
  endforeach()
  unset(config_files)

endfunction()

#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${HIP_WRAPPER_INC_DIR} DESTINATION hip/include  COMPONENT dev)
# Create symlink to binaries
create_binary_symlink()
install(DIRECTORY ${HIP_WRAPPER_BIN_DIR} DESTINATION hip COMPONENT dev)

option(BUILD_SHARED_LIBS "Build the shared library" ON)
# Create symlink to library files
create_library_symlink()
if(HIP_PLATFORM STREQUAL "amd" )
  if(BUILD_SHARED_LIBS)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libamdhip64.so DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libamdhip64.so.${HIP_LIB_VERSION_MAJOR} DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libamdhip64.so.${HIP_LIB_VERSION_STRING} DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libhiprtc-builtins.so DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libhiprtc-builtins.so.${HIP_LIB_VERSION_MAJOR} DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libhiprtc-builtins.so.${HIP_LIB_VERSION_STRING} DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libhiprtc.so DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libhiprtc.so.${HIP_LIB_VERSION_MAJOR} DESTINATION hip/lib COMPONENT binary)
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libhiprtc.so.${HIP_LIB_VERSION_STRING} DESTINATION hip/lib COMPONENT binary)

  else()
    install(FILES ${HIP_WRAPPER_LIB_DIR}/libamdhip64.a DESTINATION hip/lib COMPONENT binary)
  endif()#End BUILD_SHARED_LIBS
endif()#End HIP_PLATFORM AMD
#install hipInfo
install(FILES ${HIP_WRAPPER_LIB_DIR}/${HIP_INFO_FILE} DESTINATION hip/lib COMPONENT binary)
#create symlink to cmake files
create_cmake_symlink()
install(DIRECTORY ${HIP_WRAPPER_CMAKE_DIR} DESTINATION hip/lib COMPONENT binary)
install(DIRECTORY ${HIP_WRAPPER_FINDHIP_DIR}/ DESTINATION hip/cmake COMPONENT dev)
