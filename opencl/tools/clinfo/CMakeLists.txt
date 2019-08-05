set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${OPENCL_ICD_LOADER_HEADERS_DIR})

add_definitions(-DHAVE_CL2_HPP)

add_executable(clinfo clinfo.cpp)

target_link_libraries(clinfo OpenCL)
