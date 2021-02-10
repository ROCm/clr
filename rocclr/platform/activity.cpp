/* Copyright (c) 2019-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "platform/activity.hpp"

ACTIVITY_PROF_INSTANCES();

#define CASE_STRING(X, C)  case X: case_string = #C ;break;

const char* getOclCommandKindString(uint32_t op) {
  const char* case_string;

  switch(static_cast<cl_command_type>(op)) {
    CASE_STRING(0, InternalMarker)
    CASE_STRING(CL_COMMAND_MARKER, Marker)
    CASE_STRING(CL_COMMAND_NDRANGE_KERNEL, KernelExecution)
    CASE_STRING(CL_COMMAND_READ_BUFFER, CopyDeviceToHost)
    CASE_STRING(CL_COMMAND_WRITE_BUFFER, CopyHostToDevice)
    CASE_STRING(CL_COMMAND_COPY_BUFFER, CopyDeviceToDevice)
    CASE_STRING(CL_COMMAND_READ_BUFFER_RECT, CopyDeviceToHost2D)
    CASE_STRING(CL_COMMAND_WRITE_BUFFER_RECT, CopyHostToDevice2D)
    CASE_STRING(CL_COMMAND_COPY_BUFFER_RECT, CopyDeviceToDevice2D)
    CASE_STRING(CL_COMMAND_FILL_BUFFER, FillBuffer)
    CASE_STRING(CL_COMMAND_TASK, Task)
    CASE_STRING(CL_COMMAND_NATIVE_KERNEL, NativeKernel)
    CASE_STRING(CL_COMMAND_READ_IMAGE, ReadImage)
    CASE_STRING(CL_COMMAND_WRITE_IMAGE, WriteImage)
    CASE_STRING(CL_COMMAND_COPY_IMAGE, CopyImage)
    CASE_STRING(CL_COMMAND_COPY_IMAGE_TO_BUFFER, CopyImageToBuffer)
    CASE_STRING(CL_COMMAND_COPY_BUFFER_TO_IMAGE, CopyBufferToImage)
    CASE_STRING(CL_COMMAND_MAP_BUFFER, MapBuffer)
    CASE_STRING(CL_COMMAND_MAP_IMAGE, MapImage)
    CASE_STRING(CL_COMMAND_UNMAP_MEM_OBJECT, UnmapMemObject)
    CASE_STRING(CL_COMMAND_ACQUIRE_GL_OBJECTS, AcquireGLObjects)
    CASE_STRING(CL_COMMAND_RELEASE_GL_OBJECTS, ReleaseGLObjects)
    CASE_STRING(CL_COMMAND_USER, User)
    CASE_STRING(CL_COMMAND_BARRIER, Barrier)
    CASE_STRING(CL_COMMAND_MIGRATE_MEM_OBJECTS, MigrateMemObjects)
    CASE_STRING(CL_COMMAND_FILL_IMAGE, FillImage)
    CASE_STRING(CL_COMMAND_SVM_FREE, SvmFree)
    CASE_STRING(CL_COMMAND_SVM_MEMCPY, SvmMemcpy)
    CASE_STRING(CL_COMMAND_SVM_MEMFILL, SvmMemFill)
    CASE_STRING(CL_COMMAND_SVM_MAP, SvmMap)
    CASE_STRING(CL_COMMAND_SVM_UNMAP, SvmUnmap)
    default: case_string = "Unknown command type";
  };
  return case_string;
};
