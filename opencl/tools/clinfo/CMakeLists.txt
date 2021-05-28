add_executable(clinfo clinfo.cpp)

target_link_libraries(clinfo OpenCL amdocl)

INSTALL(TARGETS clinfo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
