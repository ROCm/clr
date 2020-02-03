//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

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
