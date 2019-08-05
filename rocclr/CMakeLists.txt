cmake_minimum_required(VERSION 3.4.3)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
  set(PROJ_VERSION VERSION 1.5.0)
endif()

# FIXME: Remove following line after enabling COMGR by default
set(USE_COMGR_LIBRARY "no" CACHE STRING "Do not enable COMGR by default")

# Build ROCm-OpenCL with ccache if the package is present.
set(ROCM_OPENCL_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(ROCM_OPENCL_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(FATAL_ERROR "Unable to find the program ccache. Set ROCM_OPENCL_CCACHE_BUILD to OFF")
  endif()

  # FIXME: Remove following if block after enabling COMGR by default
  if (${USE_COMGR_LIBRARY} STREQUAL "no")
    set(LLVM_CCACHE_BUILD ON CACHE BOOL "")
    set(ROCM_DEVICE_LIBS_CCACHE_BUILD ON CACHE BOOL "")
    set(ROCM_OPENCL_DRIVER_CCACHE_BUILD ON CACHE BOOL "")
  endif() # if (${USE_COMGR_LIBRARY} STREQUAL "no")
endif()

project(OpenCL-ROCm)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.2" CACHE PATH "")

find_package(ROCT REQUIRED)
find_package(ROCR REQUIRED)

# FIXME: Remove following if block after enabling COMGR by default
if (${USE_COMGR_LIBRARY} STREQUAL "no")
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

  set(BUILD_HC_LIB OFF CACHE BOOL "")
  set(ROCM_DEVICELIB_INCLUDE_TESTS OFF CACHE BOOL "")
  set(AMDGCN_TARGETS_LIB_LIST "AMDGCN_LIB_TARGETS")
  set(AMDGCN_TARGETS_LIB_DEPS "AMDGCN_DEP_TARGETS")
  set(AMDGPU_TARGET_TRIPLE "amdgcn-amd-amdhsa")
  add_subdirectory(library/amdgcn EXCLUDE_FROM_ALL)
  add_subdirectory(compiler/driver EXCLUDE_FROM_ALL)

  install(PROGRAMS $<TARGET_FILE:clang> $<TARGET_FILE:lld>
          DESTINATION bin/x86_64
          COMPONENT DEV)

  foreach(AMDGCN_LIB_TARGET ${AMDGCN_LIB_TARGETS})
    get_target_property(lib_file_name ${AMDGCN_LIB_TARGET} ARCHIVE_OUTPUT_NAME)
    get_target_property(lib_file_path ${AMDGCN_LIB_TARGET} ARCHIVE_OUTPUT_DIRECTORY)
    install(FILES ${lib_file_path}/${lib_file_name}.amdgcn.bc
          DESTINATION lib/x86_64/bitcode
          COMPONENT DEV)
  endforeach()
endif() # if (${USE_COMGR_LIBRARY} STREQUAL "no")

if(${USE_COMGR_LIBRARY} MATCHES "yes")
  set(COMGR_DYN_DLL "yes")
  add_definitions(-DCOMGR_DYN_DLL)
  add_definitions(-DUSE_COMGR_LIBRARY)
  if( ${BUILD_HIP} MATCHES "yes")
    add_subdirectory(api/hip)
  endif()
endif()

add_subdirectory(api/opencl/amdocl)
add_subdirectory(compiler/lib/loaders/elf/utils/libelf)
add_subdirectory(runtime)
add_subdirectory(tools/clinfo)

set(BUILD_SHARED_LIBS "Build shared libs" ON)
add_subdirectory(api/opencl/khronos/icd)

###--- Packaging ------------------------------------------------------------###

# MAIN package
install(PROGRAMS $<TARGET_FILE:clinfo>
        DESTINATION bin/x86_64
        COMPONENT MAIN)
install(PROGRAMS $<TARGET_FILE:amdocl64>
        DESTINATION lib/x86_64
        COMPONENT MAIN)
install(PROGRAMS $<TARGET_FILE:OpenCL>
        DESTINATION lib/x86_64
        COMPONENT MAIN)
install(PROGRAMS $<TARGET_SONAME_FILE:OpenCL>
        DESTINATION lib/x86_64
        COMPONENT MAIN)

# DEV package
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/api/opencl/khronos/headers/opencl2.2/CL"
        DESTINATION include
        COMPONENT DEV
        USE_SOURCE_PERMISSIONS
        PATTERN cl_d3d10.h EXCLUDE
        PATTERN cl_d3d11.h EXCLUDE
        PATTERN cl_dx9_media_sharing.h EXCLUDE
        PATTERN cl_egl.h EXCLUDE)
install(PROGRAMS $<TARGET_LINKER_FILE:OpenCL>
        DESTINATION lib/x86_64
        COMPONENT DEV)

# Generic CPACK variables
set(CPACK_GENERATOR "DEB;RPM" CACHE STRING "Default packaging generators")

set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_VENDOR "AMD")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm/opencl")

set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "0")

# Debian CPACK variables
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_MAIN_FILE_NAME "rocm-opencl-1.2.0.deb")
set(CPACK_DEBIAN_MAIN_PACKAGE_NAME "rocm-opencl")
set(CPACK_DEBIAN_MAIN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/DEB/postinst;${CMAKE_CURRENT_SOURCE_DIR}/DEB/prerm")

set(CPACK_DEBIAN_DEV_FILE_NAME "rocm-opencl-dev-1.2.0.deb")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "rocm-opencl-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "rocm-opencl")

# RPM CPACK variables
set(CPACK_RPM_COMPONENT_INSTALL ON)

set(CPACK_RPM_MAIN_FILE_NAME "rocm-opencl-1.2.0.rpm")
set(CPACK_RPM_MAIN_PACKAGE_NAME "rocm-opencl")
set(CPACK_RPM_MAIN_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_post")
set(CPACK_RPM_MAIN_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RPM/rpm_postun")

set(CPACK_RPM_DEV_FILE_NAME "rocm-opencl-devel-1.2.0.rpm")
set(CPACK_RPM_DEV_PACKAGE_NAME "rocm-opencl-devel")
set(CPACK_RPM_DEV_PACKAGE_DEPENDS "rocm-opencl")

set(CPACK_COMPONENTS_ALL MAIN DEV)
include(CPack)
