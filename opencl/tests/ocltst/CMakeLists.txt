cmake_minimum_required(VERSION 3.5.1)

set(OCLTST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_subdirectory(module/common)
add_subdirectory(log)
add_subdirectory(env)
add_subdirectory(module/gl)
add_subdirectory(module/perf)
add_subdirectory(module/runtime)
