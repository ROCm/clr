# - Try to find the ROCm Runtime.
# Once done this will define
#  ROCR_FOUND - System has the ROCR installed
#  ROCR_INCLUDE_DIRS - The ROCR include directories
#  ROCR_LIBRARIES - The libraries needed to use the ROCR

find_path(ROCR_INCLUDE_DIR hsa.h
          HINTS /opt/rocm/include /opt/rocm/hsa/include
          PATH_SUFFIXES hsa)

find_library(ROCR_LIBRARY hsa-runtime64
             HINTS /opt/rocm/lib  /opt/rocm/hsa/lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ROCR_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ROCR  DEFAULT_MSG
                                  ROCR_LIBRARY ROCR_INCLUDE_DIR)

mark_as_advanced(ROCR_INCLUDE_DIR ROCR_LIBRARY)

set(ROCR_LIBRARIES ${ROCR_LIBRARY})
set(ROCR_INCLUDE_DIRS ${ROCR_INCLUDE_DIR})
