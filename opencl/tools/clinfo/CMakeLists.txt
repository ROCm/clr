set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${OPENCL_ICD_LOADER_HEADERS_DIR})

add_definitions(-DHAVE_CL2_HPP -DCL_TARGET_OPENCL_VERSION=220)

add_executable(clinfo clinfo.cpp)

target_link_libraries(clinfo OpenCL)
