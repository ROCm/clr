
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(clinfo clinfo.cpp)

target_link_libraries(clinfo OpenCL amdocl)
