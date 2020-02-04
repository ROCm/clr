/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

#include "cl_common.hpp"

RUNTIME_ENTRY_RET(cl_key_amd, clCreateKeyAMD,
                  (cl_platform_id platform, void(CL_CALLBACK* destructor)(void*),
                   cl_int* errcode_ret)) {
  cl_key_amd key = amd::ObjectMetadata::createKey(destructor);

  *not_null(errcode_ret) = amd::ObjectMetadata::check(key) ? CL_SUCCESS : CL_OUT_OF_RESOURCES;

  return key;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clObjectGetValueForKeyAMD, (void* object, cl_key_amd key, void** ret_val)) {
  if (ret_val == NULL) {
    return CL_INVALID_VALUE;
  }
  *ret_val = NULL;

  if (!amd::RuntimeObject::isValidHandle(object)) {
    return CL_INVALID_OBJECT_AMD;
  }
  if (!amd::ObjectMetadata::check(key)) {
    return CL_INVALID_KEY_AMD;
  }

  amd::ObjectMetadata& metadata =
      amd::RuntimeObject::fromHandle<amd::RuntimeObject>(object)->metadata();

  void* value = metadata.getValueForKey(key);
  if (value == NULL) {
    return CL_INVALID_KEY_AMD;
  }

  *ret_val = value;
  return CL_SUCCESS;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clObjectSetValueForKeyAMD, (void* object, cl_key_amd key, void* value)) {
  if (!amd::RuntimeObject::isValidHandle(object)) {
    return CL_INVALID_OBJECT_AMD;
  }
  if (!amd::ObjectMetadata::check(key)) {
    return CL_INVALID_KEY_AMD;
  }
  if (value == NULL) {
    return CL_INVALID_VALUE;
  }

  amd::ObjectMetadata& metadata =
      amd::RuntimeObject::fromHandle<amd::RuntimeObject>(object)->metadata();

  metadata.setValueForKey(key, value);
  return CL_SUCCESS;
}
RUNTIME_EXIT
