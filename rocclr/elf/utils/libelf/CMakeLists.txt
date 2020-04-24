
file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

add_library(oclelf_obj OBJECT ${sources})
set_target_properties(oclelf_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(oclelf_obj
  PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/../common
     ${CMAKE_CURRENT_SOURCE_DIR}/../common/win32/sys)

target_compile_definitions(oclelf_obj
  PUBLIC
    BSD_LIBELF USE_MEMFILE)
