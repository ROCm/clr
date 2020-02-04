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
    CASE_STRING(CL_COMMAND_NDRANGE_KERNEL, KernelExecution)
    CASE_STRING(CL_COMMAND_READ_BUFFER, CopyDeviceToHost)
    CASE_STRING(CL_COMMAND_WRITE_BUFFER, CopyoHostToDevice)
    CASE_STRING(CL_COMMAND_COPY_BUFFER, CopyDeviceToDevice)
    CASE_STRING(CL_COMMAND_READ_BUFFER_RECT, CopyDeviceToHost2D)
    CASE_STRING(CL_COMMAND_WRITE_BUFFER_RECT, CopyoHostToDevice2D)
    CASE_STRING(CL_COMMAND_COPY_BUFFER_RECT, CopyDeviceToDevice2D)
    CASE_STRING(CL_COMMAND_FILL_BUFFER, FillBuffer)
    default: case_string = "Unknown command type";
  };
  return case_string;
};
