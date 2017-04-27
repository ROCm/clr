set(headers
  irif.amdgcn.inc
  ockl.amdgcn.inc
  ocml.amdgcn.inc
  opencl.amdgcn.inc
  oclc_unsafe_math_on.amdgcn.inc
  oclc_unsafe_math_off.amdgcn.inc
  oclc_finite_only_on.amdgcn.inc
  oclc_finite_only_off.amdgcn.inc
  oclc_correctly_rounded_sqrt_on.amdgcn.inc
  oclc_correctly_rounded_sqrt_off.amdgcn.inc
  oclc_daz_opt_off.amdgcn.inc
  oclc_daz_opt_on.amdgcn.inc
  oclc_isa_version_700.amdgcn.inc
  oclc_isa_version_701.amdgcn.inc
  oclc_isa_version_800.amdgcn.inc
  oclc_isa_version_801.amdgcn.inc
  oclc_isa_version_802.amdgcn.inc
  oclc_isa_version_803.amdgcn.inc
  oclc_isa_version_804.amdgcn.inc
  oclc_isa_version_810.amdgcn.inc
  oclc_isa_version_900.amdgcn.inc
  oclc_isa_version_901.amdgcn.inc
)


include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/common)
include_directories(${CMAKE_SOURCE_DIR}/compiler/lib/loaders/elf/utils/libelf)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


add_library(oclrocm OBJECT
  rocprintf.cpp
  rocprogram.cpp
  rocmemory.cpp
  rocdevice.cpp
  roccompiler.cpp
  rocblit.cpp
  rockernel.cpp
  rocvirtual.cpp
  rocglinterop.cpp
  rocappprofile.cpp
  rocsettings.cpp
  libraries.amdgcn.inc
  opencl1.2-c.amdgcn.inc
  opencl2.0-c.amdgcn.inc
  ${headers}
)
set_target_properties(oclrocm PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_dependencies(oclrocm libhsail-includes)

foreach(header ${headers})
  string(REGEX REPLACE "([^._-]*)(.*).amdgcn.inc" "${CMAKE_BINARY_DIR}/library/amdgcn/\\1/\\1\\2.lib.bc" bclib ${header}) 
  string(REGEX REPLACE ".inc$" "" target ${header})
  string(REGEX REPLACE "[\\.-]" "_" symbol ${target})
  string(REGEX REPLACE ".amdgcn.inc$" "" lib_bc ${header})

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${target}.inc
    COMMAND bc2h ${bclib} ${CMAKE_CURRENT_BINARY_DIR}/${target}.inc ${symbol}
    DEPENDS ${bclib} ${lib_bc}_lib_bc
    COMMENT "Generating ${target}.inc"
  )
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/${target}.inc)

  add_custom_target(${target}.inc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${target}.inc)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${target}.inc PROPERTIES HEADER_FILE_ONLY TRUE)

  string(REGEX REPLACE ".amdgcn.inc$" "" lib_bc ${header})
  #add_dependencies(${target}.inc ${lib_bc}_lib_bc)
endforeach()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/libraries.amdgcn.inc "// Automatically generated file; DO NOT EDIT.\n")

foreach(file ${headers})
  string(REGEX REPLACE "(.*).inc$" "#include \"\\1.inc\"" line ${file})
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/libraries.amdgcn.inc "${line}\n")
endforeach()

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/libraries.amdgcn.inc
  "static inline std::pair<const void*, size_t>\nget_oclc_isa_version(uint gfxip)\n{\nswitch (gfxip) {\n")
foreach(file ${headers})
  string(REGEX MATCH "oclc_isa_version_(.*).amdgcn.inc" isa ${file})
  if (NOT "${isa}" STREQUAL "")
    string(REGEX REPLACE "oclc_isa_version_(.*).amdgcn.inc" "\\1" gfxip ${file})
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/libraries.amdgcn.inc
      "case ${gfxip}: return std::make_pair(oclc_isa_version_${gfxip}_amdgcn, oclc_isa_version_${gfxip}_amdgcn_size); break;\n")
  endif()
endforeach()
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/libraries.amdgcn.inc
  "default: return std::make_pair((const void*)0,(size_t)0);\n}\n}\n")

foreach(file ${headers})
  string(REGEX MATCH "oclc_(.*)_on.amdgcn.inc" is_control ${file})
  if (NOT "${is_control}" STREQUAL "")
    string(REGEX REPLACE "oclc_(.*)_on.amdgcn.inc" "\\1" function ${file})
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/libraries.amdgcn.inc
      "static inline std::pair<const char*, size_t> get_oclc_${function}(bool on)\n{ return std::make_pair("
                "(const char*)(on ? oclc_${function}_on_amdgcn : oclc_${function}_off_amdgcn),"
                "on ? oclc_${function}_on_amdgcn_size : oclc_${function}_off_amdgcn_size);}\n")
  endif()
endforeach()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.amdgcn.pch
  COMMAND clang -cc1 -x cl-header -triple amdgcn-amd-amdhsa-opencl -Werror -O3 -Dcl_khr_fp64 -Dcl_khr_fp16 -DNDEBUG -cl-std=CL1.2 -emit-pch -o ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.amdgcn.pch < ${CMAKE_SOURCE_DIR}/compiler/llvm/tools/clang/lib/Headers/opencl-c.h 
  DEPENDS clang ${CMAKE_SOURCE_DIR}/compiler/llvm/tools/clang/lib/Headers/opencl-c.h
  COMMENT "Generating opencl1.2-c.amdgcn.pch"
)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.amdgcn.inc
  COMMAND bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.amdgcn.pch ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.amdgcn.inc opencl1_2_c_amdgcn
  DEPENDS bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl1.2-c.amdgcn.pch
  COMMENT "Generating opencl1.2-c.amdgcn.inc"
)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.inc ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.pch)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.pch
  COMMAND clang -cc1 -x cl-header -triple amdgcn-amd-amdhsa-opencl -Werror -O3 -Dcl_khr_fp64 -Dcl_khr_fp16 -DNDEBUG -cl-std=CL2.0 -emit-pch -o ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.pch < ${CMAKE_SOURCE_DIR}/compiler/llvm/tools/clang/lib/Headers/opencl-c.h 
  DEPENDS clang ${CMAKE_SOURCE_DIR}/compiler/llvm/tools/clang/lib/Headers/opencl-c.h
  COMMENT "Generating opencl2.0-c.amdgcn.pch"
)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.inc
  COMMAND bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.pch ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.inc opencl2_0_c_amdgcn
  DEPENDS bc2h ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.pch
  COMMENT "Generating opencl2.0-c.amdgcn.inc"
)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.inc ${CMAKE_CURRENT_BINARY_DIR}/opencl2.0-c.amdgcn.pch)

