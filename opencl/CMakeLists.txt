cmake_minimum_required(VERSION 3.5.1)

if (POLICY CMP0048)
	  cmake_policy(SET CMP0048 NEW)
	    set(PROJ_VERSION VERSION 1.5.0)
endif()

project(opencl)

#example command:
#cmake -DROCclr_DIR=/rocclr -DUSE_COMGR_LIBRARY=yes -DLIBROCclr_STATIC_DIR=/build ..

set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/khronos/headers/opencl2.2" CACHE PATH "")
set(BUILD_SHARED_LIBS "Build shared libs" ON)

add_subdirectory(khronos/icd)
add_subdirectory(amdocl)
add_subdirectory(tools/clinfo)
add_subdirectory(tools/cltrace)

# get date information based on UTC
# use the last two digits of year + week number + day in the week as OPENCL_VERSION_GITDATE
# use the commit date, instead of build date
# add xargs to remove strange trailing newline character
execute_process(COMMAND git show -s --format=@%ct
    COMMAND xargs
    COMMAND date -f - --utc +%y%U%w
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE OPENCL_VERSION_GITDATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# get commit short hash
execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE OPENCL_VERSION_GITHASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(OPENCL_VERSION_MAJOR "2")
set(OPENCL_VERSION_MINOR "0")
set(OPENCL_VERSION_PATCH ${OPENCL_VERSION_GITDATE})

get_target_property(OPENCL_LIB_VERSION_MAJOR OpenCL SOVERSION)
get_target_property(OPENCL_LIB_VERSION_STRING OpenCL VERSION)

###--- Packaging ------------------------------------------------------------###

# MAIN package
install(PROGRAMS $<TARGET_FILE:clinfo>
        DESTINATION bin
        COMPONENT MAIN)
install(PROGRAMS $<TARGET_FILE:cltrace>
        DESTINATION lib
        COMPONENT MAIN)
install(PROGRAMS $<TARGET_FILE:amdocl64>
        DESTINATION lib
        COMPONENT MAIN)
install(PROGRAMS $<TARGET_FILE:OpenCL>
        DESTINATION lib
        COMPONENT MAIN)
install(PROGRAMS $<TARGET_SONAME_FILE:OpenCL>
        DESTINATION lib
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
        DESTINATION lib
        COMPONENT DEV)

#############################
# Packaging steps
#############################
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm/opencl" CACHE PATH "Package Installation path for OpenCL")

#ROCM_PATH is needed to create symlink of libraries
if (NOT DEFINED ROCM_PATH )
     string(REPLACE "/opencl" "" ROCM_PATH ${CPACK_PACKAGING_INSTALL_PREFIX})
endif ()
message (STATUS "ROCM Installation path(ROCM_PATH): ${ROCM_PATH}")

# Package: rocm-opencl
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/rocm-opencl)
configure_file(packaging/rocm-opencl.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
configure_file(packaging/rocm-opencl.postinst ${BUILD_DIR}/postinst @ONLY)
configure_file(packaging/rocm-opencl.prerm ${BUILD_DIR}/prerm @ONLY)
configure_file(packaging/rocm-opencl.rpm_post ${BUILD_DIR}/rpm_post @ONLY)
configure_file(packaging/rocm-opencl.rpm_postun ${BUILD_DIR}/rpm_postun @ONLY)
add_custom_target(pkg_rocm_opencl COMMAND ${CMAKE_COMMAND} .
    COMMAND rm -rf *.deb *.rpm *.tar.gz
    COMMAND make package
    COMMAND cp *.deb ${PROJECT_BINARY_DIR}
    COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${BUILD_DIR}
    DEPENDS amdocl64 OpenCL)

# Package: rocm-opencl-dev
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/rocm-opencl-dev)
configure_file(packaging/rocm-opencl-dev.txt ${BUILD_DIR}/CMakeLists.txt @ONLY)
add_custom_target(pkg_rocm_opencl_dev  COMMAND ${CMAKE_COMMAND} .
    COMMAND rm -rf *.deb *.rpm *.tar.gz
    COMMAND make package
    COMMAND cp *.deb ${PROJECT_BINARY_DIR}
    COMMAND cp *.rpm ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${BUILD_DIR}
    DEPENDS amdocl64 OpenCL)

# Package: all
if(POLICY CMP0037)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0037 OLD)
endif()
file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/fixnames
	CONTENT "pwd; for i in *.deb; do mv \"\$i\" \"\${i/.deb/-${OPENCL_VERSION_GITHASH}-amd64.deb}\" ; done
	for i in *.rpm ; do mv \"\$i\" \"\${i/.rpm/-${OPENCL_VERSION_GITHASH}.x86_64.rpm}\" ; done
")
add_custom_target(package
    COMMAND bash ${PROJECT_BINARY_DIR}/fixnames
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS pkg_rocm_opencl pkg_rocm_opencl_dev)
if(POLICY CMP0037)
    cmake_policy(POP)
endif()


