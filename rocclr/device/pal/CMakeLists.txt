###
file(STRINGS palcdefs PAL_MAJOR_VERSION REGEX "^PAL_MAJOR_VERSION = [0-9]+")
string(REGEX REPLACE "PAL_MAJOR_VERSION = " "" PAL_MAJOR_VERSION ${PAL_MAJOR_VERSION})

file(STRINGS palcdefs GPUOPEN_MAJOR_VERSION REGEX "^GPUOPEN_MAJOR_VERSION = [0-9]+")
string(REGEX REPLACE "GPUOPEN_MAJOR_VERSION = " "" GPUOPEN_MAJOR_VERSION ${GPUOPEN_MAJOR_VERSION})

### Create PAL backend library for ROCclr
add_library(rocclrpal OBJECT "")

# Add defines necessary for PAL build
set(PAL_CLIENT_INTERFACE_MAJOR_VERSION ${PAL_MAJOR_VERSION}
    CACHE STRING "PAL major interface: ${PAL_MAJOR_VERSION}" FORCE)
set(GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION ${GPUOPEN_MAJOR_VERSION}
    CACHE STRING "GPU open major interface: ${GPUOPEN_MAJOR_VERSION}" FORCE)
set(GPUOPEN_CLIENT_INTERFACE_MINOR_VERSION "0" CACHE STRING "GPU open minor interface: 0" FORCE)

set(PAL_CLIENT "OCL" CACHE STRING "Build PAL OCL support" FORCE)

set(PAL_BUILD_GPUOPEN ON CACHE BOOL "Build GPUOpen support")
set(PAL_BUILD_VIDEO OFF CACHE BOOL "Don't build PAL with Video support")
set(PAL_BUILD_SCPC OFF CACHE BOOL "Don't build SCPC")

set(PAL_CLOSED_SOURCE ON CACHE BOOL "Build closed source PAL")

set(PAL_BUILD_POLARIS22 ON CACHE BOOL "Build PAL with Polaris support")
set(PAL_BUILD_GFX9 ON CACHE BOOL "Build PAL with GFX9 support")
set(PAL_BUILD_OSS4 ON CACHE BOOL "Build PAL with OSS4 support")
set(PAL_BUILD_VEGA12 ON CACHE BOOL "Build PAL with VEGA12 support")
set(PAL_BUILD_VEGA20 ON CACHE BOOL "Build PAL with GFX9 support")
set(PAL_BUILD_RAVEN1 ON CACHE BOOL "Build PAL with Raven1 support")
set(PAL_BUILD_RAVEN2 ON CACHE BOOL "Build PAL with Raven2 support")
set(PAL_BUILD_RENOIR ON CACHE BOOL "Build PAL with Renoir support")

set(PAL_BUILD_GFX10 ON CACHE BOOL "Build PAL with GFX10 support")
set(PAL_BUILD_NAVI12 ON CACHE BOOL "Build PAL with Navi12 support")
set(PAL_BUILD_NAVI14 ON CACHE BOOL "Build PAL with Navi14 support")

option(PAL_DEVELOPER_BUILD "No developer build" OFF)

set(PYTHON_EXECUTABLE /usr/bin/python3)

# Build PAL . . .
set(GLOBAL_ROOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(palbe)

# Setup defines for the client compilation
target_compile_definitions(rocclrpal PRIVATE
    PAL_CLIENT_INTERFACE_MAJOR_VERSION=${PAL_CLIENT_INTERFACE_MAJOR_VERSION})
target_compile_definitions(rocclrpal PRIVATE
    GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION=${GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION})
target_compile_definitions(rocclrpal PRIVATE
    GPUOPEN_CLIENT_INTERFACE_MINOR_VERSION=${GPUOPEN_CLIENT_INTERFACE_MINOR_VERSION})

target_compile_definitions(rocclrpal PRIVATE PAL_CLIENT_OCL=1)
target_compile_definitions(rocclrpal PRIVATE PAL_GPUOPEN_OCL=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_GPUOPEN=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_SCPC=0)
target_compile_definitions(rocclrpal PRIVATE PAL_CLOSED_SOURCE=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_GPUOPEN=1)

# GFX9 targets
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_POLARIS22=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_GFX9=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_OSS4=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_VEGA12=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_VEGA20=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_RAVEN1=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_RAVEN2=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_RENOIR=1)

# GFX10 targets
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_GFX10=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_NAVI12=1)
target_compile_definitions(rocclrpal PRIVATE PAL_BUILD_NAVI14=1)

target_compile_definitions(rocclrpal PRIVATE OPENCL_MAINLINE=1)

if(${USE_COMGR_LIBRARY} MATCHES "yes")
  target_compile_definitions(rocclrpal PRIVATE USE_COMGR_LIBRARY)
endif()

if(UNIX)
  target_compile_definitions(rocclrpal PRIVATE PAL_AMDGPU_BUILD)
endif()

target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compiler/sc/HSAIL/include)
target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../asic_reg)

target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/pal/palbe)
target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/pal/palbe/inc)
target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/pal/palbe/inc/core)
target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/pal/palbe/inc/util)
target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/pal/palbe/shared/gpuopen/inc)
target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device/pal/palbe/shared/gpuopen/third_party/metrohash/src)
target_include_directories(rocclrpal PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_sources(rocclrpal PRIVATE
  palappprofile.cpp
  palblit.cpp
  palconstbuf.cpp
  palcounters.cpp
  paldebugmanager.cpp
  paldevice.cpp
  paldeviced3d10.cpp
  paldeviced3d11.cpp
  paldeviced3d9.cpp
  paldevicegl.cpp
  palgpuopen.cpp
  palkernel.cpp
  palmemory.cpp
  palprintf.cpp
  palprogram.cpp
  palresource.cpp
  palschedcl.cpp
  palsettings.cpp
  palthreadtrace.cpp
  paltimestamp.cpp
  palvirtual.cpp
)

set_target_properties(rocclrpal PROPERTIES POSITION_INDEPENDENT_CODE ON)


