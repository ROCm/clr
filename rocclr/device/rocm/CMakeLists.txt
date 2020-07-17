add_library(oclrocm OBJECT
  roccounters.cpp
  rocprintf.cpp
  rocprogram.cpp
  rocmemory.cpp
  rocdevice.cpp
  rocblit.cpp
  rockernel.cpp
  rocvirtual.cpp
  rocglinterop.cpp
  rocappprofile.cpp
  rocsettings.cpp
  rocschedcl.cpp
  rochostcall.cpp
)

target_include_directories(oclrocm
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    # GL and EGL headers
    $<BUILD_INTERFACE:${OPENCL_DIR}/khronos/headers>
    $<BUILD_INTERFACE:${OPENCL_DIR}/khronos/headers/opencl2.2>
    $<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>
  PRIVATE
    ${OPENCL_DIR}
    ${PROJECT_SOURCE_DIR}/compiler/lib
    ${PROJECT_SOURCE_DIR}/compiler/lib/include
    ${PROJECT_SOURCE_DIR}/compiler/lib/backends/common
    ${PROJECT_SOURCE_DIR}/compiler/lib/loaders/elf/utils/common
    ${PROJECT_SOURCE_DIR}/compiler/lib/loaders/elf/utils/libelf
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ROCM_OCL_INCLUDES}
    $<TARGET_PROPERTY:hsa-runtime64::hsa-runtime64,INTERFACE_INCLUDE_DIRECTORIES>)

option(BUILD_HMM "Build HMM support" OFF)
if (BUILD_HMM)
  target_compile_definitions(oclrocm
    PRIVATE AMD_HMM_SUPPORT)
endif()

if(USE_COMGR_LIBRARY)
  target_compile_definitions(oclrocm
    PRIVATE USE_COMGR_LIBRARY)
endif()

set_target_properties(oclrocm PROPERTIES POSITION_INDEPENDENT_CODE ON)

