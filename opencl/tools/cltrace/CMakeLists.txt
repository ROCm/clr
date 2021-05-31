add_library(cltrace SHARED cltrace.cpp)

if(WIN32)
  target_sources(cltrace PRIVATE cltrace.def)
else()
  target_link_options(cltrace PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/cltrace.map")
  set_target_properties(cltrace PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/cltrace.map")
endif()

target_link_libraries(cltrace OpenCL amdocl)

INSTALL(TARGETS cltrace
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
