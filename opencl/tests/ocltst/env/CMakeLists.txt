add_executable(ocltst
    ${OCLTST_DIR}/env/oclsysinfo.cpp
    ${OCLTST_DIR}/env/ocltst.cpp
    ${OCLTST_DIR}/env/pfm.cpp
    ${OCLTST_DIR}/env/Timer.cpp
    ${OCLTST_DIR}/module/common/BaseTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLTestImp.cpp
    ${OCLTST_DIR}/module/common/OCLThread.cpp
    ${OCLTST_DIR}/module/common/OCLWrapper.cpp)

set_target_properties(ocltst PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    EXCLUDE_FROM_ALL TRUE)

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
