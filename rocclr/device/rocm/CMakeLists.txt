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


target_include_directories(oclrocm
  PUBLIC
    $<TARGET_PROPERTY:amd_comgr,INTERFACE_INCLUDE_DIRECTORIES>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler/lib/loaders/elf/utils/common
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler/lib/loaders/elf/utils/libelf
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ROCM_OCL_INCLUDES}
    ${ROCR_INCLUDES})


if(${USE_COMGR_LIBRARY} MATCHES "yes")
  target_compile_definitions(oclrocm
    PRIVATE USE_COMGR_LIBRARY)
endif()

set_target_properties(oclrocm PROPERTIES POSITION_INDEPENDENT_CODE ON)

