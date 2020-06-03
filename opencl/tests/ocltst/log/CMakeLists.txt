add_library(TestLog SHARED
    ${OCLTST_DIR}/log/oclTestLog.cpp)

set_target_properties(TestLog PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/ocltst
    EXCLUDE_FROM_ALL TRUE)

target_include_directories(TestLog
    PRIVATE
        ${OCLTST_DIR}/include
    PUBLIC
        ${OCLTST_DIR}/log)
