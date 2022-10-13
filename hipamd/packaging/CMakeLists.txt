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

cmake_minimum_required(VERSION 3.16.8)

#set components for HIP
set(CPACK_COMPONENTS_ALL binary dev doc samples runtime-nvidia)

###############Install Required files for all compnents########

#Enable Component Install
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)

###Set License####
set(CPACK_RESOURCE_FILE_LICENSE ${hip_SOURCE_DIR}/LICENSE.txt)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT binary)
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

#Begin binary files install
if(HIP_PLATFORM STREQUAL "amd" )
   if(BUILD_SHARED_LIBS)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libamdhip64.so DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libamdhip64.so.${HIP_LIB_VERSION_MAJOR} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libamdhip64.so.${HIP_LIB_VERSION_STRING} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc.so DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc.so.${HIP_LIB_VERSION_MAJOR} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc.so.${HIP_LIB_VERSION_STRING} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc-builtins.so DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc-builtins.so.${HIP_LIB_VERSION_MAJOR} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
      install(FILES ${CMAKE_BINARY_DIR}/lib/libhiprtc-builtins.so.${HIP_LIB_VERSION_STRING} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
   else()
      install(FILES ${CMAKE_BINARY_DIR}/lib/libamdhip64.a DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
   endif()#End BUILD_SHARED_LIBS

#TODO:This do not belong in BINARY package.
#Keeping it as is for now
install(FILES ${CMAKE_BINARY_DIR}/.hipInfo DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)

install(FILES ${CMAKE_BINARY_DIR}/hip-config.cmake ${CMAKE_BINARY_DIR}/hip-config-version.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip COMPONENT binary)
install ( EXPORT hip-targets  FILE hip-targets.cmake NAMESPACE hip:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip  COMPONENT binary)

install(FILES ${CMAKE_BINARY_DIR}/src/hip-lang-config.cmake ${CMAKE_BINARY_DIR}/src/hip-lang-config-version.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip-lang COMPONENT binary)
install ( EXPORT hip-lang-targets  FILE hip-lang-targets.cmake NAMESPACE hip-lang:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip-lang  COMPONENT binary)

install(FILES ${CMAKE_BINARY_DIR}/hiprtc-config.cmake ${CMAKE_BINARY_DIR}/hiprtc-config-version.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiprtc COMPONENT binary)
install ( EXPORT hiprtc-targets  FILE hiprtc-targets.cmake NAMESPACE hiprtc:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiprtc  COMPONENT binary)

endif()#End HIP_PLATFORM = "amd"
#End bianry files install

#Begin dev files install
if(WIN32)
  install(DIRECTORY ${HIP_COMMON_DIR}/bin DESTINATION . COMPONENT dev
          USE_SOURCE_PERMISSIONS)
else()
  install(DIRECTORY ${HIP_COMMON_DIR}/bin DESTINATION .  COMPONENT dev
          USE_SOURCE_PERMISSIONS
          DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
          PATTERN *.bat EXCLUDE)
endif()

install(DIRECTORY ${hip_SOURCE_DIR}/bin DESTINATION . COMPONENT dev
        USE_SOURCE_PERMISSIONS
        DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(DIRECTORY ${HIP_COMMON_DIR}/include DESTINATION . COMPONENT dev)
install(DIRECTORY ${hip_SOURCE_DIR}/include/hip/amd_detail
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip COMPONENT dev)
install(DIRECTORY ${hip_SOURCE_DIR}/include/hip/nvidia_detail
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip COMPONENT dev)
install(FILES ${CMAKE_BINARY_DIR}/include/hip/amd_detail/hip_prof_str.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip/amd_detail COMPONENT dev)
install(FILES ${CMAKE_BINARY_DIR}/include/hip/hip_version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip COMPONENT dev)
install(FILES ${CMAKE_BINARY_DIR}/.hipVersion DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT dev)
install(DIRECTORY ${HIP_COMMON_DIR}/cmake/ DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip COMPONENT dev)
#End dev files install

#Begin doc files install
find_program(DOXYGEN_EXE doxygen)
if(DOXYGEN_EXE)
    add_custom_target(build_doxygen ALL
                  COMMAND HIP_PATH=${HIP_COMMON_DIR} doxygen ${HIP_COMMON_DIR}/docs/doxygen-input/doxy.cfg)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/RuntimeAPI/html
            DESTINATION ${CMAKE_INSTALL_DOCDIR}/RuntimeAPI COMPONENT doc)
endif()
#End doc files install

#Begin samples files install
install(DIRECTORY ${HIP_COMMON_DIR}/samples DESTINATION ${CMAKE_INSTALL_DATADIR}/hip COMPONENT samples)
#End samples files install


##################################
# Packaging steps COMMON Variables
##################################
set(CPACK_SET_DESTDIR TRUE)

set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_CONTACT "HIP Support <hip.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HIP:Heterogenous-computing Interface for Portability")
set(CPACK_PACKAGE_VERSION_MAJOR ${HIP_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${HIP_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${HIP_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_PACKAGING_VERSION_PATCH})
set(CPACK_GENERATOR "TGZ;DEB;RPM" CACHE STRING "Package types to build")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt")
if (CPACK_RPM_PACKAGE_RELEASE MATCHES "local" )
  #If building locally default value will cause build failure
  #DEBUG SYMBOL pacaking require SOURCE_DIR to be small
  set(CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX ${CPACK_INSTALL_PREFIX})
endif()
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

set(CPACK_SOURCE_GENERATOR "TGZ")


#Begin Binary Packaging setting

set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_RPM "ON")

set(CPACK_DEBIAN_BINARY_PACKAGE_NAME "hip-runtime-amd")
set(CPACK_RPM_BINARY_PACKAGE_NAME "hip-runtime-amd")

set(CPACK_COMPONENT_BINARY_DESCRIPTION "HIP:Heterogenous-computing Interface for Portability [RUNTIME - AMD]")
if(FILE_REORG_BACKWARD_COMPATIBILITY)
#This is used for softlinking hip-target files
   configure_file(hip-runtime-amd.postinst ${CMAKE_CURRENT_BINARY_DIR}/binary/postinst @ONLY)
   configure_file(hip-runtime-amd.prerm    ${CMAKE_CURRENT_BINARY_DIR}/binary/prerm @ONLY)
   set(CPACK_DEBIAN_BINARY_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/binary/postinst;${CMAKE_CURRENT_BINARY_DIR}/binary/prerm")
endif()
set(CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS "hsa-rocr-dev (>= 1.3), rocminfo,  comgr (>= 2.0), rocm-llvm, libc6, rocm-core")
set(CPACK_DEBIAN_BINARY_PACKAGE_PROVIDES "hip-rocclr (= ${CPACK_PACKAGE_VERSION})")
set(CPACK_DEBIAN_BINARY_PACKAGE_REPLACES "hip-rocclr (= ${CPACK_PACKAGE_VERSION})")

set(CPACK_RPM_BINARY_PACKAGE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
if(FILE_REORG_BACKWARD_COMPATIBILITY)
   set(CPACK_RPM_BINARY_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/binary/postinst")
   set(CPACK_RPM_BINARY_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/binary/prerm")
endif()
string(REPLACE "-" "_" HIP_BASE_VERSION ${CPACK_PACKAGE_VERSION})
set(CPACK_RPM_BINARY_PACKAGE_REQUIRES "hsa-rocr-dev >= 1.3, rocminfo,  comgr >= 2.0, rocm-llvm, rocm-core")
set(CPACK_RPM_BINARY_PACKAGE_PROVIDES "hip-rocclr = ${HIP_BASE_VERSION}")
set(CPACK_RPM_BINARY_PACKAGE_OBSOLETES "hip-rocclr = ${HIP_BASE_VERSION}")
#End Binary Packaging setting

#Begin dev Packaging setting
set(CPACK_DEV_DEB "ON")
set(CPACK_DEV_RPM "ON")

set(CPACK_DEBIAN_DEV_PACKAGE_NAME "hip-dev")
set(CPACK_RPM_DEV_PACKAGE_NAME "hip-devel")

set(CPACK_COMPONENT_DEV_DESCRIPTION "HIP: Heterogenous-computing Interface for Portability [DEVELOPMENT]")

configure_file(hip-devel.postinst ${CMAKE_CURRENT_BINARY_DIR}/dev/postinst @ONLY)
configure_file(hip-devel.prerm    ${CMAKE_CURRENT_BINARY_DIR}/dev/prerm @ONLY)
set(CPACK_DEBIAN_DEV_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/dev/postinst;${CMAKE_CURRENT_BINARY_DIR}/dev/prerm")

set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "perl (>= 5.0), liburi-encode-perl, libfile-basedir-perl, libfile-copy-recursive-perl, libfile-find-rule-perl, libfile-listing-perl, libfile-which-perl, libc6, file, rocm-core")
set(CPACK_DEBIAN_DEV_PACKAGE_PROVIDES "hip-base")
set(CPACK_DEBIAN_DEV_PACKAGE_REPLACES "hip-base")

set(CPACK_RPM_DEV_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/dev/postinst")
set(CPACK_RPM_DEV_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/dev/prerm")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "perl >= 5.0, perl-File-Which, perl-File-Listing, perl-File-BaseDir, perl-File-Find-Rule, perl-URI-Encode, file, rocm-core")

set(CPACK_RPM_DEV_PACKAGE_PROVIDES "hip-base")
set(CPACK_RPM_DEV_PACKAGE_OBSOLETES "hip-base")
#End dev Packaging setting

#Begin doc Packaging setting
set(CPACK_DOC_DEB "ON")
set(CPACK_DOC_RPM "ON")
set(CPACK_DEBIAN_DOC_PACKAGE_NAME "hip-doc")
set(CPACK_RPM_DOC_PACKAGE_NAME "hip-doc")
set(CPACK_COMPONENT_DOC_DESCRIPTION "HIP: Heterogenous-computing Interface for Portability [DOCUMENTATION]")

set(CPACK_DEBIAN_DOC_PACKAGE_DEPENDS "hip-dev (= ${CPACK_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}), rocm-core")
set(CPACK_DEBIAN_DOC_PACKAGE_PROVIDES "hip-doc")

string(REPLACE "-" "_" HIP_BASE_VERSION ${CPACK_PACKAGE_VERSION})
set(CPACK_RPM_DOC_PACKAGE_REQUIRES "hip-devel = ${HIP_BASE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}, rocm-core")

#End doc Packaging setting

#Begin samples Packaging setting
set(CPACK_SAMPLES_DEB "ON")
set(CPACK_SAMPLES_RPM "ON")
set(CPACK_DEBIAN_SAMPLES_PACKAGE_NAME "hip-samples")
set(CPACK_RPM_SAMPLES_PACKAGE_NAME "hip-samples")
set(CPACK_COMPONENT_SAMPLES_DESCRIPTION "HIP: Heterogenous-computing Interface for Portability [SAMPLES]")
set(CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS "hip-dev (= ${CPACK_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}), rocm-core")
set(CPACK_DEBIAN_SAMPLES_PACKAGE_PROVIDES "hip-samples")

set(CPACK_RPM_SAMPLES_PACKAGE_REQUIRES "hip-devel = ${HIP_BASE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}, rocm-core")
#End samples Packaging setting

#Begin runtime-nvidia Packaging setting
set(CPACK_RUNTIME-NVIDIA_DEB "ON")
set(CPACK_RUNTIME-NVIDIA_RPM "ON")
set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_NAME "hip-runtime-nvidia")
set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_NAME "hip-runtime-nvidia")
set(CPACK_COMPONENT_RUNTIME-NVIDIA_DESCRIPTION "HIP: Heterogenous-computing Interface for Portability [RUNTIME-NVIDIA]")

set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_DEPENDS "cuda (>= 7.5), rocm-core")
set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_PROVIDES "hip-nvcc")
set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_REPLACES "hip-nvcc")

set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_PROVIDES "hip-nvcc")
set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_OBSOLETES "hip-nvcc")
set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_REQUIRES "cuda >= 7.5, rocm-core")

# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)

    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_BINARY_PACKAGE_REQUIRES ${CPACK_RPM_BINARY_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS ${CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DOC_PACKAGE_REQUIRES ${CPACK_RPM_DOC_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DOC_PACKAGE_DEPENDS ${CPACK_DEBIAN_DOC_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_SAMPLES_PACKAGE_REQUIRES ${CPACK_RPM_SAMPLES_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS ${CPACK_DEBIAN_SAMPLES_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_DEPENDS})
endif()

include(CPack)
