if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DDEBUG)
endif()

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic -Wl,--unresolved-symbols=report-all -Wl,--version-script=${CMAKE_SOURCE_DIR}/tools/cltrace/cltrace.map")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/khronos/headers/opencl2.2)

# FIXME: Remove this
include_directories(${ROCclr_DIR}/include)

add_definitions(-DHAVE_CL2_HPP -DCL_TARGET_OPENCL_VERSION=220)

add_library(cltrace SHARED cltrace.cpp cltrace.def)
target_include_directories(cltrace
  PRIVATE
    $<TARGET_PROPERTY:amdrocclr_static,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(cltrace OpenCL)
