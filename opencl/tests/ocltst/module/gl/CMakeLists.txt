add_library(oclgl SHARED
    OCLGLBuffer.cpp
    OCLGLBufferMultipleQueues.cpp
    OCLGLDepthBuffer.cpp
    OCLGLDepthTex.cpp
    OCLGLFenceSync.cpp
    OCLGLMsaaTexture.cpp
    OCLGLMultiContext.cpp
    OCLGLTexture.cpp
    TestList.cpp
    $<TARGET_OBJECTS:Common>)

set_target_properties(oclgl PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(oclgl
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(oclgl
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(oclgl
    PRIVATE
        OpenCL
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES})

add_custom_command(
    TARGET oclgl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/oclgl.exclude
            ${CMAKE_BINARY_DIR}/tests/ocltst/ogl.exclude)

add_dependencies(ocltst oclgl)
