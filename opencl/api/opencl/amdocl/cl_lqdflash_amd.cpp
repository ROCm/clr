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

#if defined(_WIN32) && !defined(_LP64)
#define WITH_LIQUID_FLASH 1
#endif // _WIN32

#if defined WITH_LIQUID_FLASH
#include "lf.h"
#endif // WITH_LIQUID_FLASH


namespace amd {

class LiquidFlashFile : public RuntimeObject
{
private:
    const wchar_t*    name_;
    cl_file_flags_amd flags_;
    void*             handle_;

public:
    LiquidFlashFile(const wchar_t* name, cl_file_flags_amd flags)
      : name_(name), flags_(flags), handle_(NULL) { }

    ~LiquidFlashFile();

    bool open();
    void close();

    size_t blockSize() const;

    size_t readBlocks(
        void* dst,
        uint32_t count,
        const uint64_t* file_offsets,
        const uint64_t* buffer_offsets,
        const uint64_t* sizes);

    virtual ObjectType objectType() const {return ObjectTypeLiquidFlashFile;}
};

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
    if (err == lf_success) {
        return true;
    }
#endif // WITH_LIQUID_FLASH
    return false;
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

size_t
LiquidFlashFile::blockSize() const
{
#if defined WITH_LIQUID_FLASH
    if (handle_ != NULL) {
        lf_uint32 blockSize;
        if (lfGetFileBlockSize((lf_file)handle_, &blockSize) == lf_success) {
            return blockSize;
        }
    }
#endif // WITH_LIQUID_FLASH
    return 0;
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
    return CL_INVALID_FILE_OBJECT_AMD;
}
RUNTIME_EXIT

