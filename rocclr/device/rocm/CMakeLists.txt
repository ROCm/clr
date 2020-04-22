include_directories(${CMAKE_CURRENT_SOURCE_DIR}/compiler/lib/loaders/elf/utils/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/compiler/lib/loaders/elf/utils/libelf)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ROCM_OCL_INCLUDES})

if(${USE_COMGR_LIBRARY} MATCHES "yes")
  add_definitions(-DUSE_COMGR_LIBRARY)
endif()

add_library(oclrocm OBJECT
  roccounters.cpp
  rocprintf.cpp
  rocprogram.cpp
  rocmemory.cpp
  rocdevice.cpp
  rocblit.cpp
  rockernel.cpp
  rocvirtual.cpp
  rocglinterop.cpp
  rocappprofile.cpp
  rocsettings.cpp
  rocschedcl.cpp
  rochcmessages.cpp
  rochcprintf.cpp
  rochostcall.cpp
)
set_target_properties(oclrocm PROPERTIES POSITION_INDEPENDENT_CODE ON)

