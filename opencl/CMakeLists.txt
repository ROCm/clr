cmake_minimum_required(VERSION 3.4.3)

if (POLICY CMP0048)
	  cmake_policy(SET CMP0048 NEW)
	    set(PROJ_VERSION VERSION 1.5.0)
endif()

project(opencl)

#example command:
#cmake -DVDI_DIR=/vdi -DUSE_COMGR_LIBRARY=yes -DLIBVDI_STATIC_DIR=/build ..

set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/khronos/headers/opencl2.2" CACHE PATH "")
set(BUILD_SHARED_LIBS "Build shared libs" ON)

add_subdirectory(khronos/icd)
add_subdirectory(amdocl)
add_subdirectory(tools/clinfo)

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
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/khronos/headers/opencl2.2/CL"
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
