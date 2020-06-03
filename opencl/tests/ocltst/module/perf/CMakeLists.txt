set(OCLPERF_SOURCES
    OCLPerf3DImageWriteSpeed.cpp
    OCLPerfAES256.cpp
    OCLPerfAtomicSpeed20.cpp
    OCLPerfAtomicSpeed.cpp
    OCLPerfBufferCopyOverhead.cpp
    OCLPerfBufferCopySpeed.cpp
    OCLPerfBufferReadSpeed.cpp
    OCLPerfBufferWriteSpeed.cpp
    OCLPerfCommandQueue.cpp
    OCLPerfConcurrency.cpp
    OCLPerfCPUMemSpeed.cpp
    OCLPerfDeviceConcurrency.cpp
    OCLPerfDeviceEnqueue2.cpp
    OCLPerfDeviceEnqueue.cpp
    OCLPerfDeviceEnqueueEvent.cpp
    OCLPerfDeviceEnqueueSier.cpp
    OCLPerfDevMemReadSpeed.cpp
    OCLPerfDevMemWriteSpeed.cpp
    OCLPerfDispatchSpeed.cpp
    OCLPerfDoubleDMA.cpp
    OCLPerfDoubleDMASeq.cpp
    OCLPerfFillBuffer.cpp
    OCLPerfFillImage.cpp
    OCLPerfFlush.cpp
    OCLPerfGenericBandwidth.cpp
    OCLPerfGenoilSiaMiner.cpp
    OCLPerfImageCopyCorners.cpp
    OCLPerfImageCopySpeed.cpp
    OCLPerfImageCreate.cpp
    OCLPerfImageMapUnmap.cpp
    OCLPerfImageReadSpeed.cpp
    OCLPerfImageReadsRGBA.cpp
    OCLPerfImageReadWrite.cpp
    OCLPerfImageSampleRate.cpp
    OCLPerfImageWriteSpeed.cpp
    OCLPerfKernelArguments.cpp
    OCLPerfKernelThroughput.cpp
    OCLPerfLDSLatency.cpp
    OCLPerfLDSReadSpeed.cpp
    OCLPerfMandelbrot.cpp
    OCLPerfMapBufferReadSpeed.cpp
    OCLPerfMapBufferWriteSpeed.cpp
    OCLPerfMapImageReadSpeed.cpp
    OCLPerfMapImageWriteSpeed.cpp
    OCLPerfMatrixTranspose.cpp
    OCLPerfMemCombine.cpp
    OCLPerfMemCreate.cpp
    OCLPerfMemLatency.cpp
    OCLPerfPinnedBufferReadSpeed.cpp
    OCLPerfPinnedBufferWriteSpeed.cpp
    OCLPerfPipeCopySpeed.cpp
    OCLPerfProgramGlobalRead.cpp
    OCLPerfProgramGlobalWrite.cpp
    OCLPerfSampleRate.cpp
    OCLPerfScalarReplArrayElem.cpp
    OCLPerfSdiP2PCopy.cpp
    OCLPerfSHA256.cpp
    OCLPerfSVMAlloc.cpp
    OCLPerfSVMKernelArguments.cpp
    OCLPerfSVMMap.cpp
    OCLPerfSVMMemcpy.cpp
    OCLPerfSVMMemFill.cpp
    OCLPerfSVMSampleRate.cpp
    OCLPerfTextureMemLatency.cpp
    OCLPerfUAVReadSpeed.cpp
    OCLPerfUAVReadSpeedHostMem.cpp
    OCLPerfUAVWriteSpeedHostMem.cpp
    OCLPerfUncoalescedRead.cpp
    OCLPerfVerticalFetch.cpp
    TestList.cpp)

if(OPENGL_FOUND AND GLEW_FOUND)
    set(OCLPERF_SOURCES
        ${OCLPERF_SOURCES}
        OCLPerfSepia.cpp)
endif()

add_library(oclperf SHARED
    ${OCLPERF_SOURCES}
    $<TARGET_OBJECTS:Common>)

set_target_properties(oclperf PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    EXCLUDE_FROM_ALL TRUE)

target_compile_definitions(oclperf
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(oclperf
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(oclperf
    PRIVATE
        OpenCL)

if(OPENGL_FOUND AND GLEW_FOUND)
    target_link_libraries(oclperf
        PRIVATE
            ${GLEW_LIBRARIES}
            ${OPENGL_LIBRARIES})
endif()

add_custom_command(
    TARGET oclperf POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/oclperf.exclude
            ${CMAKE_BINARY_DIR}/tests/ocltst/oclperf.exclude)

add_dependencies(ocltst oclperf)

add_custom_target(test_oclperf
    COMMAND
        ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
        $<TARGET_FILE:ocltst> -m $<TARGET_FILE:oclperf> -A oclperf.exclude
    DEPENDS
        ocltst oclperf amdocl64
    WORKING_DIRECTORY
        ${CMAKE_BINARY_DIR}/tests/ocltst
    USES_TERMINAL)
