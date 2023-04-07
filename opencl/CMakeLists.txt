cmake_minimum_required(VERSION 3.5.1)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
  set(PROJ_VERSION VERSION 1.5.0)
endif()

project(opencl)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default libdir to be "lib" for ROCm, distros will override this anyway:
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")

include(GNUInstallDirs)

option(BUILD_TESTS "Enable building OpenCL tests" OFF)
option(BUILD_ICD "Enable building OpenCL ICD Loader" ON)
option(EMU_ENV "Enable building for emulation environment" OFF)
# Disable file reorg backward compatibilty for ASAN build
if(NOT ENABLE_ASAN_PACKAGING)
  option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorganization backward compatibility" ON)
endif()

set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_LIST_DIR}/khronos/headers/opencl2.2" CACHE PATH "")
if(BUILD_ICD)
  add_subdirectory(khronos/icd)
else()
  find_package(OpenCL REQUIRED)
endif()
add_subdirectory(amdocl)
add_subdirectory(tools/clinfo)
add_subdirectory(tools/cltrace)
if(BUILD_TESTS)
  add_subdirectory(tests/ocltst)
endif()

###--- Packaging ------------------------------------------------------------###

# DEV package
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/khronos/headers/opencl2.2/CL"
        DESTINATION include
        COMPONENT DEV
        USE_SOURCE_PERMISSIONS
        PATTERN cl_d3d10.h EXCLUDE
        PATTERN cl_d3d11.h EXCLUDE
        PATTERN cl_dx9_media_sharing.h EXCLUDE
        PATTERN cl_egl.h EXCLUDE)

#############################
# Packaging steps
#############################
if(NOT WIN32)
find_package(ROCM QUIET CONFIG PATHS /opt/rocm)
if(ROCM_FOUND)
  include(ROCMSetupVersion)
  rocm_setup_version( VERSION "2.0.0" )
else()
  set(PROJECT_VERSION "2.0.0")
endif()

#set a name for icd file
set(OPENCL_AMD_ICD_FILE "amdocl64.icd")
if (DEFINED ROCM_PATCH_VERSION)
  # set unique name for ICD file for each jenkins build
  # Use ENV variable CPACK_RPM_PACKAGE_RELEASE, which is having build number
  set(PACKAGE_RELEASE_VERSION "")
  if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(PACKAGE_RELEASE_VERSION $ENV{CPACK_RPM_PACKAGE_RELEASE})
  endif()
  if(PACKAGE_RELEASE_VERSION)
    # Replace "." to "_" in package version string. So file name will have .icd as extension
    string(REPLACE "." "_" PACKAGE_RELEASE_VERSION ${PACKAGE_RELEASE_VERSION})
  else()
    # set a default value
    set(PACKAGE_RELEASE_VERSION "9999")
  endif()
  set(OPENCL_AMD_ICD_FILE "amdocl64_${ROCM_PATCH_VERSION}_${PACKAGE_RELEASE_VERSION}.icd")
endif()

if(BUILD_ICD)
  get_target_property(OPENCL_LIB_VERSION_MAJOR OpenCL SOVERSION)
  get_target_property(OPENCL_LIB_VERSION_STRING OpenCL VERSION)
endif()

#Set Package Version
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
  message("Using CPACK_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION}")
endif()

set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Package Installation path for OpenCL")
#ROCM_PATH is needed to create symlink of libraries
if(NOT DEFINED ROCM_PATH)
  string(REPLACE "/opencl" "" ROCM_PATH ${CPACK_PACKAGING_INSTALL_PREFIX})
endif()
message (STATUS "ROCM Installation path(ROCM_PATH): ${ROCM_PATH}")

#Package: rocm-opencl,rocm-opencl-dev/devel,rocm-ocl-icd

if(BUILD_ICD)
  set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/rocm-ocl-icd)
  configure_file(packaging/rocm-ocl-icd.postinst ${BUILD_DIR}/postinst @ONLY)
  configure_file(packaging/rocm-ocl-icd.prerm ${BUILD_DIR}/prerm @ONLY)
  configure_file(packaging/rocm-ocl-icd.rpm_post ${BUILD_DIR}/rpm_post @ONLY)
  configure_file(packaging/rocm-ocl-icd.rpm_postun ${BUILD_DIR}/rpm_postun @ONLY)
endif()

add_subdirectory(packaging)

#File reorg Backward compatibility function
if(FILE_REORG_BACKWARD_COMPATIBILITY)
  include(opencl-backward-compat.cmake)
endif()

endif()
