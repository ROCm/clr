set(TESTS
    OCLAsyncMap
    OCLAsyncTransfer
    OCLAtomicCounter
    OCLBlitKernel
    OCLBufferFromImage
    OCLCPUGuardPages
    OCLCreateBuffer
    OCLCreateContext
    OCLCreateImage
    OCLDeviceAtomic
    OCLDeviceQueries
    OCLDynamic
    OCLDynamicBLines
    OCLGenericAddressSpace
    OCLGetQueueThreadID
    OCLGlobalOffset
    OCLImage2DFromBuffer
    OCLImageCopyPartial
    OCLKernelBinary
    OCLLDS32K
    OCLLinearFilter
    OCLLiquidFlash
    OCLMapCount
    OCLMemDependency
    OCLMemObjs
    OCLMemoryInfo
    OCLMultiQueue
    OCLOfflineCompilation
    OCLP2PBuffer
    OCLPartialWrkgrp
    OCLPerfCounters
    OCLPersistent
    OCLPinnedMemory
    OCLPlatformAtomics
    OCLProgramScopeVariables
    OCLReadWriteImage
    OCLRTQueue
    OCLSDI
    OCLSemaphore
    OCLStablePState
    OCLSVM
    OCLThreadTrace
    OCLUnalignedCopy
)

add_library(oclruntime SHARED
    TestList.cpp
    $<TARGET_OBJECTS:Common>)

foreach(TEST ${TESTS})
    target_sources(oclruntime
        PRIVATE
            ${TEST}.cpp)
endforeach()

set_target_properties(oclruntime PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    EXCLUDE_FROM_ALL TRUE)

target_compile_definitions(oclruntime
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(oclruntime
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(oclruntime
    PRIVATE
        OpenCL)

if(OPENGL_FOUND AND GLEW_FOUND)
    target_link_libraries(oclruntime
        PRIVATE
            ${GLEW_LIBRARIES}
            ${OPENGL_LIBRARIES})
endif()

add_custom_command(
    TARGET oclruntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/oclruntime.exclude
            ${CMAKE_BINARY_DIR}/tests/ocltst/oclruntime.exclude)

add_custom_target(test.ocltst.oclruntime
    COMMAND
        ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
        $<TARGET_FILE:ocltst> -m $<TARGET_FILE:oclruntime> -A oclruntime.exclude
    DEPENDS
        ocltst oclruntime amdocl64
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)

foreach(TEST ${TESTS})
    add_custom_target(test.ocltst.oclruntime.${TEST}
        COMMAND
            ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
            $<TARGET_FILE:ocltst> -m $<TARGET_FILE:oclruntime> -t ${TEST}
        DEPENDS
            ocltst oclruntime amdocl64
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}/tests/ocltst
        USES_TERMINAL)
endforeach()
