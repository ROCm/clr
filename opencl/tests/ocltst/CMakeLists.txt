cmake_minimum_required(VERSION 3.5.1)

set(OCLTST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenGL)
find_package(GLEW)

add_subdirectory(module/common)
add_subdirectory(log)
add_subdirectory(env)
if(OPENGL_FOUND AND GLEW_FOUND)
    add_subdirectory(module/gl)
endif()
add_subdirectory(module/perf)
add_subdirectory(module/runtime)
