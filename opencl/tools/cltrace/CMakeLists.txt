set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(cltrace SHARED cltrace.cpp)

if(WIN32)
  target_sources(amdocl PRIVATE cltrace.def)
else()
  set_target_properties(amdocl PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/cltrace.map")
endif()

target_link_libraries(cltrace OpenCL amdocl)
