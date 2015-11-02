//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//

#include "cl_common.hpp"
#include <CL/cl_ext.h>

#include "platform/object.hpp"

#include "cl_lqdflash_amd.h"

#if defined __linux__
typedef wchar_t char_t;
#endif // __linux__

#if !defined(OPENCL_MAINLINE) && !defined(BUILD_HSA_TARGET) && defined(_WIN32)
#define WITH_LIQUID_FLASH 1
#endif // _WIN32

#if defined(WITH_LIQUID_FLASH)
#include "lf.h"
#endif // WITH_LIQUID_FLASH


namespace amd {

LiquidFlashFile::~LiquidFlashFile()
{
    close();
}

bool
LiquidFlashFile::open()
{
#if defined WITH_LIQUID_FLASH
    lf_status err;
    lf_file_flags flags;

    switch (flags_) {
    case CL_FILE_READ_ONLY_AMD:  flags = LF_READ;          break;
    case CL_FILE_WRITE_ONLY_AMD: flags = LF_WRITE;         break;
    case CL_FILE_READ_WRITE_AMD: flags = LF_READ|LF_WRITE; break;
    }

    handle_ = lfOpenFile(name_, flags, &err);
    if (err != lf_success) {
        return false;
    }

    if (lfGetFileBlockSize((lf_file)handle_, &blockSize_) != lf_success) {
        return false;
    }
    return true;
#else
    return false;
#endif // WITH_LIQUID_FLASH
}

void
LiquidFlashFile::close()
{
#if defined WITH_LIQUID_FLASH
    if (handle_ != NULL) {
        lfReleaseFile((lf_file)handle_);
        handle_ = NULL;
    }
#endif // WITH_LIQUID_FLASH
}

bool
LiquidFlashFile::readBlock(
    void* dst,
    uint64_t fileOffset,
    uint64_t bufferOffset,
    uint64_t size) const
{
#if defined WITH_LIQUID_FLASH
    lf_region_descriptor    region =
        { fileOffset / blockSize(), bufferOffset / blockSize(), size / blockSize() };
    lf_status status = lfReadFile(dst, size, handle_, 1, &region, NULL);
    if (lf_success == status) {
        return true;
    }
    else {
        return false;
    }
#else
    return false;
#endif // WITH_LIQUID_FLASH
}

} // namesapce amd

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup AMD_Extensions
 *  @{
 *
 */

RUNTIME_ENTRY_RET(cl_file_amd, clCreateFileObjectAMD, (
    cl_context context,
    cl_file_flags_amd flags,
    const wchar_t* file_name,
    cl_int* errcode_ret))
{
    amd::LiquidFlashFile* file = new amd::LiquidFlashFile(file_name, flags);

    if (file == NULL) {
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        return (cl_file_amd)0;
    }

    if (!file->open()) {
        *not_null(errcode_ret) = CL_INVALID_VALUE;
        delete file;
        return (cl_file_amd)0;
    }

    *not_null(errcode_ret) = CL_SUCCESS;
    return as_cl(file);
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clRetainFileObjectAMD, (
    cl_file_amd file))
{
    if (!is_valid(file)) {
        return CL_INVALID_FILE_OBJECT_AMD;
    }
    as_amd(file)->retain();
    return CL_SUCCESS;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clReleaseFileObjectAMD, (
    cl_file_amd file))
{
    if (!is_valid(file)) {
        return CL_INVALID_FILE_OBJECT_AMD;
    }
    as_amd(file)->release();
    return CL_SUCCESS;
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
    if (!is_valid(command_queue)) {
        return CL_INVALID_COMMAND_QUEUE;
    }

    if (!is_valid(buffer)) {
        return CL_INVALID_MEM_OBJECT;
    }
    amd::Buffer* dstBuffer = as_amd(buffer)->asBuffer();
    if (dstBuffer == NULL) {
        return CL_INVALID_MEM_OBJECT;
    }

    if (dstBuffer->getMemFlags() &
        (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)) {
        return CL_INVALID_OPERATION;
    }

    amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
    if (NULL == queue) {
        return CL_INVALID_COMMAND_QUEUE;
    }
    amd::HostQueue& hostQueue = *queue;

    if(hostQueue.context() != dstBuffer->getContext()) {
        return CL_INVALID_CONTEXT;
    }

    if (!is_valid(file)) {
        return CL_INVALID_FILE_OBJECT_AMD;
    }

    amd::LiquidFlashFile* amdFile  = as_amd(file);
    amd::Coord3D    dstOffset(buffer_offset, 0, 0);
    amd::Coord3D    dstSize(cb, 1, 1);

    if ((!dstBuffer->validateRegion(dstOffset, dstSize)) ||
        // LF library supports aligned sizes only
        ((buffer_offset % amdFile->blockSize()) != 0) ||
        ((cb % amdFile->blockSize()) != 0) ||
        ((file_offset % amdFile->blockSize()) != 0))  {
        return CL_INVALID_VALUE;
    }

    amd::Command::EventWaitList eventWaitList;
    cl_int err = amd::clSetEventWaitList(eventWaitList,
        hostQueue.context(), num_events_in_wait_list, event_wait_list);
    if (err != CL_SUCCESS){
        return err;
    }

    amd::WriteBufferFromFileCommand *command = new amd::WriteBufferFromFileCommand(
        hostQueue, eventWaitList,
        *dstBuffer, dstOffset, dstSize, amdFile, file_offset);
    if (command == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Make sure we have memory for the command execution
    if (!command->validateMemory()) {
        delete command;
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    command->enqueue();
    if (blocking_write) {
        command->awaitCompletion();
    }

    *not_null(event) = as_cl(&command->event());
    if (event == NULL) {
        command->release();
    }
    return CL_SUCCESS;
}
RUNTIME_EXIT

