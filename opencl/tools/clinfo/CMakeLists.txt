add_executable(clinfo clinfo.cpp)

target_compile_definitions(clinfo PRIVATE CL_TARGET_OPENCL_VERSION=220 HAVE_CL2_HPP)

target_include_directories(clinfo PRIVATE  "../../khronos/headers/opencl2.2/CL" "../../khronos/headers/opencl2.2/CL/.."  "../../khronos/headers/opencl2.2/CL/../.." "../../khronos/headers/opencl2.2/CL/../../.." "../../khronos/headers/opencl2.2/CL/../../../..")

target_link_libraries(clinfo OpenCL)

INSTALL(TARGETS clinfo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
