add_executable(clinfo clinfo.cpp)

target_compile_definitions(clinfo PRIVATE CL_TARGET_OPENCL_VERSION=220 HAVE_CL2_HPP)

target_include_directories(clinfo PRIVATE ${OPENCL_ICD_LOADER_HEADERS_DIR})

target_link_libraries(clinfo OpenCL::OpenCL)

INSTALL(TARGETS clinfo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
