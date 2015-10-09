#ifndef __CL_LQDFLASH_AMD_H
#define __CL_LQDFLASH_AMD_H

#include "CL/cl_ext.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

extern CL_API_ENTRY cl_file_amd CL_API_CALL
clCreateFileObjectAMD(
    cl_context context,
    cl_file_flags_amd flags,
    cl_char * file_name,
    cl_int * errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainFileObjectAMD(
    cl_file_amd file) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseFileObjectAMD(
    cl_file_amd file) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBufferFromFileAMD(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t buffer_offset,
    size_t cb,
    cl_file_amd file,
    size_t file_offset,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event) CL_EXT_SUFFIX__VERSION_1_2;

#ifdef __cplusplus
} /*extern "C"*/
#endif /*__cplusplus*/

#endif
