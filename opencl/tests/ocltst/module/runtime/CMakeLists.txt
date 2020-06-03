add_library(oclruntime SHARED
    OCLAsyncMap.cpp
    OCLAsyncTransfer.cpp
    OCLAtomicCounter.cpp
    OCLBlitKernel.cpp
    OCLBufferFromImage.cpp
    OCLCPUGuardPages.cpp
    OCLCreateBuffer.cpp
    OCLCreateContext.cpp
    OCLCreateImage.cpp
    OCLDeviceAtomic.cpp
    OCLDeviceQueries.cpp
    OCLDynamic.cpp
    OCLDynamicBLines.cpp
    OCLGenericAddressSpace.cpp
    OCLGetQueueThreadID.cpp
    OCLGlobalOffset.cpp
    OCLImage2DFromBuffer.cpp
    OCLImageCopyPartial.cpp
    OCLKernelBinary.cpp
    OCLLDS32K.cpp
    OCLLinearFilter.cpp
    OCLLiquidFlash.cpp
    OCLMapCount.cpp
    OCLMemDependency.cpp
    OCLMemObjs.cpp
    OCLMemoryInfo.cpp
    OCLMultiQueue.cpp
    OCLOfflineCompilation.cpp
    OCLP2PBuffer.cpp
    OCLPartialWrkgrp.cpp
    OCLPerfCounters.cpp
    OCLPersistent.cpp
    OCLPinnedMemory.cpp
    OCLPlatformAtomics.cpp
    OCLProgramScopeVariables.cpp
    OCLReadWriteImage.cpp
    OCLRTQueue.cpp
    OCLSDI.cpp
    OCLSemaphore.cpp
    OCLStablePState.cpp
    OCLSVM.cpp
    OCLThreadTrace.cpp
    OCLUnalignedCopy.cpp
    TestList.cpp
    $<TARGET_OBJECTS:Common>)

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

add_dependencies(ocltst oclruntime)

add_custom_target(test_oclruntime
    COMMAND
        ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
        $<TARGET_FILE:ocltst> -m $<TARGET_FILE:oclruntime> -A oclruntime.exclude
    DEPENDS
        ocltst oclruntime amdocl64
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)
