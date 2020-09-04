#-------------------------------------elf_test--------------------------------------#
cmake_minimum_required(VERSION 3.5.1)
# This is unit test for amd::Elf.
# The test is on top of rocclr, so rocclr must be built and installed firstly.
# This file is seperate from cmake file of rocclr to prevent interference. 

find_package(amd_comgr REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/amd_comgr
    lib/cmake/amd_comgr)

find_package(hsa-runtime64 REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    cmake/hsa-runtime64)

find_package(Threads REQUIRED)

# Look for ROCclr which contains elfio
find_package(ROCclr REQUIRED CONFIG
  PATHS
    /opt/rocm
    /opt/rocm/rocclr)

add_executable(elf_test main.cpp)
set_target_properties(
    elf_test PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_include_directories(elf_test
  PRIVATE
    $<TARGET_PROPERTY:amdrocclr_static,INTERFACE_INCLUDE_DIRECTORIES>)

add_definitions(-DUSE_COMGR_LIBRARY -DCOMGR_DYN_DLL -DWITH_LIGHTNING_COMPILER -DDEBUG)

target_link_libraries(elf_test PRIVATE amdrocclr_static)

#-------------------------------------elf_test--------------------------------------#