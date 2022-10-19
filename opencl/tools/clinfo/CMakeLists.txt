add_executable(clinfo clinfo.cpp)

target_compile_definitions(clinfo PRIVATE CL_TARGET_OPENCL_VERSION=220 HAVE_CL2_HPP)

target_include_directories(clinfo PRIVATE  "../../khronos/headers/opencl2.2/CL" "../../khronos/headers/opencl2.2/CL/.."  "../../khronos/headers/opencl2.2/CL/../.." "../../khronos/headers/opencl2.2/CL/../../.." "../../khronos/headers/opencl2.2/CL/../../../..")

target_link_libraries(clinfo OpenCL)

set_target_properties(clinfo PROPERTIES INSTALL_RPATH "${CPACK_PACKAGING_INSTALL_PREFIX}/lib")

INSTALL(TARGETS clinfo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
