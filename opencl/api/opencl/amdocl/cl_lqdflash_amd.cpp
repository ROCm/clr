//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//

#include "cl_common.hpp"

#include "cl_lqdflash_amd.h"

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup AMD_Extensions
 *  @{
 *
 */

RUNTIME_ENTRY_RET(cl_file_amd, clCreateFileObjectAMD, (
    cl_context context,
    cl_file_flags_amd,
    cl_char * file_name,
    cl_int *errcode_ret))
{
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_file_amd) 0;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clRetainFileObjectAMD, (
    cl_file_amd file))
{
    return CL_INVALID_FILE_OBJECT_AMD;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clReleaseFileObjectAMD, (
    cl_file_amd file))
{
    return CL_INVALID_FILE_OBJECT_AMD;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEnqueueWriteBufferFromFileAMD, (
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t buffer_offset,
    size_t cb,
    cl_file_amd file,
    size_t file_offset,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event))
{
    return CL_INVALID_FILE_OBJECT_AMD;
}
RUNTIME_EXIT

