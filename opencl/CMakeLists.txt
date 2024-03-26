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
option(BUILD_ICD "Enable building OpenCL ICD Loader" OFF)
option(EMU_ENV "Enable building for emulation environment" OFF)
option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorganization backward compatibility" OFF)
option(ENABLE_ADDRESS_SANITIZER "Option to enable ASAN build" OFF)

# Add flags to generate PDB files with full symbolic information
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG:FULL")
endif()

# Address sanitizer options
if(ENABLE_ADDRESS_SANITIZER)
  message(STATUS "Building ocltst tests with Address Sanitizer options")
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan -g -gz")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan -g -gz")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags -fuse-ld=lld -fsanitize=address -shared-libasan -g -gz -Wl,--build-id=sha1 -L${ROCM_PATH}/lib/asan")
endif()

set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_LIST_DIR}/khronos/headers/opencl2.2" CACHE PATH "")
if(BUILD_ICD)
  add_subdirectory(khronos/icd)
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
        PATTERN cl_d3d10.h EXCLUDE
        PATTERN cl_d3d11.h EXCLUDE
        PATTERN cl_dx9_media_sharing.h EXCLUDE
        PATTERN cl_egl.h EXCLUDE
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )

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

set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/opencl)
configure_file(packaging/rocm-opencl.postinst ${BUILD_DIR}/postinst @ONLY)
configure_file(packaging/rocm-opencl.prerm ${BUILD_DIR}/prerm @ONLY)
configure_file(packaging/rocm-opencl.rpm_post ${BUILD_DIR}/rpm_post @ONLY)
configure_file(packaging/rocm-opencl.rpm_postun ${BUILD_DIR}/rpm_postun @ONLY)

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
# To enable/disable #error in wrapper header files
    if(NOT DEFINED ROCM_HEADER_WRAPPER_WERROR)
      if(DEFINED ENV{ROCM_HEADER_WRAPPER_WERROR})
          set(ROCM_HEADER_WRAPPER_WERROR "$ENV{ROCM_HEADER_WRAPPER_WERROR}"
                  CACHE STRING "Header wrapper warnings as errors.")
      else()
          set(ROCM_HEADER_WRAPPER_WERROR "OFF" CACHE STRING "Header wrapper warnings as errors.")
      endif()
  endif()
  if(ROCM_HEADER_WRAPPER_WERROR)
      set(deprecated_error 1)
  else()
      set(deprecated_error 0)
  endif()

  include(opencl-backward-compat.cmake)
endif()

endif()
