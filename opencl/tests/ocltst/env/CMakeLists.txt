add_executable(ocltst)

target_sources(ocltst PRIVATE
    ${OCLTST_DIR}/env/oclsysinfo.cpp
    ${OCLTST_DIR}/env/ocltst.cpp
    ${OCLTST_DIR}/env/pfm.cpp
    ${OCLTST_DIR}/env/Timer.cpp
    ${OCLTST_DIR}/module/common/BaseTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLThread.cpp
    ${OCLTST_DIR}/module/common/OCLWrapper.cpp)

# Windows compatibilty logic
if (WIN32)
    target_sources(ocltst PRIVATE
        ${OCLTST_DIR}/env/getopt.cpp
        ${OCLTST_DIR}/env/ServiceCode.cpp
        ${OCLTST_DIR}/env/window.cpp)
endif()


set_target_properties(ocltst PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst)

target_compile_definitions(ocltst
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(ocltst
    PRIVATE
        $<TARGET_PROPERTY:Common,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(ocltst
    PRIVATE
        OpenCL
        TestLog)

INSTALL(TARGETS ocltst DESTINATION tests/ocltst COMPONENT ocltst)

