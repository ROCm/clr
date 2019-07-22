cmake_minimum_required(VERSION 3.4.3)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
  set(PROJ_VERSION VERSION 1.5.0)
endif()

# Build ROCm-OpenCL with ccache if the package is present.
set(ROCM_OPENCL_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(ROCM_OPENCL_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(FATAL_ERROR "Unable to find the program ccache. Set ROCM_OPENCL_CCACHE_BUILD to OFF")
  endif()
  set(LLVM_CCACHE_BUILD ON CACHE BOOL "")
  set(ROCM_DEVICE_LIBS_CCACHE_BUILD ON CACHE BOOL "")
  set(ROCM_OPENCL_DRIVER_CCACHE_BUILD ON CACHE BOOL "")
endif()

project(OpenCL-ROCm)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TARGETS_TO_BUILD "AMDGPU" CACHE STRING "")
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")

# override default option value in library and driver
set(GENERIC_IS_ZERO ON CACHE BOOL ON FORCE)

add_subdirectory(compiler/llvm EXCLUDE_FROM_ALL)

find_package(LLVM REQUIRED CONFIG PATHS ${CMAKE_BINARY_DIR}/compiler/llvm NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
# TODO: add find_package for Clang and lld, and also use LLVM/Clang variables got from their config
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/tools/clang/include)
include_directories(${CMAKE_BINARY_DIR}/compiler/llvm/tools/clang/include)
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/tools/lld/include)

# TODO: move AMDGPU.h header to include folder
include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/lib/Target/AMDGPU)
include_directories(${CMAKE_BINARY_DIR}/compiler/llvm/lib/Target/AMDGPU)

if(${USE_COMGR_LIBRARY} MATCHES "yes")
  set(COMGR_DYN_DLL "yes")
  add_definitions(-DCOMGR_DYN_DLL="yes")
  include_directories(${CMAKE_SOURCE_DIR}/compiler/llvm/include)
  add_definitions(-DUSE_COMGR_LIBRARY)
else()
  add_subdirectory(compiler/driver EXCLUDE_FROM_ALL)
endif()

set(BUILD_HC_LIB OFF CACHE BOOL "")
set(ROCM_DEVICELIB_INCLUDE_TESTS OFF CACHE BOOL "")
set(AMDGCN_TARGETS_LIB_LIST "AMDGCN_LIB_TARGETS")
set(AMDGCN_TARGETS_LIB_DEPS "AMDGCN_DEP_TARGETS")
set(AMDGPU_TARGET_TRIPLE "amdgcn-amd-amdhsa")
add_subdirectory(library/amdgcn EXCLUDE_FROM_ALL)

add_subdirectory(compiler/lib/loaders/elf/utils/libelf)

find_package(ROCT REQUIRED)
find_package(ROCR REQUIRED)
add_subdirectory(runtime)

set(OPENCL_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers
  ${CMAKE_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.2)

add_subdirectory(api/opencl/amdocl)
add_subdirectory(api/opencl/khronos/icd)

add_subdirectory(tools/clinfo)

install(PROGRAMS $<TARGET_FILE:clang> $<TARGET_FILE:lld>
        DESTINATION bin/x86_64
        COMPONENT libraries)

install(PROGRAMS $<TARGET_FILE:OpenCL>
        DESTINATION lib/x86_64
        COMPONENT applications)

install(PROGRAMS $<TARGET_LINKER_FILE:OpenCL>
        DESTINATION lib/x86_64
        COMPONENT libraries)

install(DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.2/CL"
        DESTINATION include
        COMPONENT libraries
        USE_SOURCE_PERMISSIONS
        PATTERN cl_egl.h EXCLUDE)

foreach(AMDGCN_LIB_TARGET ${AMDGCN_LIB_TARGETS})
  get_target_property(lib_file_name ${AMDGCN_LIB_TARGET} ARCHIVE_OUTPUT_NAME)
  get_target_property(lib_file_path ${AMDGCN_LIB_TARGET} ARCHIVE_OUTPUT_DIRECTORY)
  install(FILES ${lib_file_path}/${lib_file_name}.amdgcn.bc
        DESTINATION lib/x86_64/bitcode
        COMPONENT libraries)
endforeach()

## CPack standard variables
set ( CPACK_PACKAGE_NAME "rocm-opencl" )
set ( CPACK_PACKAGE_FILE_NAME "rocm-opencl-1.2.0" )
set ( CPACK_PACKAGE_VENDOR "AMD" )
set ( CPACK_PACKAGE_VERSION_MAJOR "1" )
set ( CPACK_PACKAGE_VERSION_MINOR "2" )
set ( CPACK_PACKAGE_VERSION_PATCH "0" )
set ( CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc." )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime" )

set ( CPACK_GENERATOR RPM DEB )
set ( CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm/opencl" )

## Debian package values

set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "AMD" )
set ( CPACK_DEBIAN_PACKAGE_DEPENDS "" )
set ( CPACK_DEB_COMPONENT_INSTALL ON)
set ( CPACK_DEBIAN_applications_PACKAGE_NAME "rocm-opencl" )
set ( CPACK_DEBIAN_applications_FILE_NAME "rocm-opencl-1.2.0.deb" )
set ( CPACK_DEBIAN_libraries_PACKAGE_NAME "rocm-opencl-dev" )
set ( CPACK_DEBIAN_libraries_FILE_NAME "rocm-opencl-dev-1.2.0.deb" )
set ( CPACK_DEBIAN_libraries_PACKAGE_DEPENDS "rocm-opencl" )
set ( CPACK_DEBIAN_applications_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/DEB/postinst;${CMAKE_CURRENT_SOURCE_DIR}/DEB/prerm" )

## RPM package values

set ( CPACK_RPM_COMPONENT_INSTALL ON)
set ( CPACK_RPM_applications_PACKAGE_NAME "rocm-opencl" )
set ( CPACK_RPM_applications_FILE_NAME "rocm-opencl-1.2.0.rpm" )
set ( CPACK_RPM_libraries_PACKAGE_NAME "rocm-opencl-devel" )
set ( CPACK_RPM_libraries_FILE_NAME "rocm-opencl-devel-1.2.0.rpm" )
set ( CPACK_RPM_libraries_PACKAGE_DEPENDS "rocm-opencl" )
set ( CPACK_RPM_applications_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post" )
set ( CPACK_RPM_applications_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun" )

set ( CPACK_COMPONENTS_ALL applications libraries )
include ( CPack )
