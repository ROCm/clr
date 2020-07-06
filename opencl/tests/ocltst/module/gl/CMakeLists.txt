set(TESTS
    OCLGLBuffer
    OCLGLBufferMultipleQueues
    OCLGLDepthBuffer
    OCLGLDepthTex
    OCLGLFenceSync
    OCLGLMsaaTexture
    OCLGLMultiContext
    OCLGLTexture
)

add_library(oclgl SHARED
    TestList.cpp
    $<TARGET_OBJECTS:Common>)

foreach(TEST ${TESTS})
    target_sources(oclgl
        PRIVATE
            ${TEST}.cpp)
endforeach()

set_target_properties(oclgl PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    EXCLUDE_FROM_ALL TRUE)

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

add_custom_target(test.ocltst.oclgl
    COMMAND
        ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
        $<TARGET_FILE:ocltst> -m $<TARGET_FILE:oclgl> -A oclgl.exclude
    DEPENDS
        ocltst oclgl amdocl64
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)

foreach(TEST ${TESTS})
    add_custom_target(test.ocltst.oclgl.${TEST}
        COMMAND
            ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
            $<TARGET_FILE:ocltst> -m $<TARGET_FILE:oclgl> -t ${TEST}
        DEPENDS
            ocltst oclgl amdocl64
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}/tests/ocltst
        USES_TERMINAL)
endforeach()
