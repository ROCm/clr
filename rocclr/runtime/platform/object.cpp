//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "platform/object.hpp"

#include <cstring>

namespace amd {

Atomic<ObjectMetadata::Key> ObjectMetadata::nextKey_ = 1;


ObjectMetadata::Destructor ObjectMetadata::destructors_[OCL_MAX_KEYS] = {NULL};


bool ObjectMetadata::check(Key key) { return key > 0 && key <= OCL_MAX_KEYS; }

ObjectMetadata::Key ObjectMetadata::createKey(Destructor destructor) {
  Key key = nextKey_++;

  if (!check(key)) {
    return 0;
  }

  destructors_[key - 1] = destructor;
  return key;
}

ObjectMetadata::~ObjectMetadata() {
  if (!values_) {
    return;
  }

  for (size_t i = 0; i < OCL_MAX_KEYS; ++i) {
    if (values_[i] && destructors_[i]) {
      destructors_[i](values_[i]);
    }
  }

  delete[] values_;
}

void* ObjectMetadata::getValueForKey(Key key) const {
  if (!values_ || !check(key)) {
    return NULL;
  }

  return values_[key - 1];
}

bool ObjectMetadata::setValueForKey(Key key, Value value) {
  if (!check(key)) {
    return false;
  }

  while (!values_) {
    Value* values = new Value[OCL_MAX_KEYS];
    memset(values, '\0', sizeof(Value) * OCL_MAX_KEYS);

    if (!values_.compareAndSet(NULL, values)) {
      delete[] values;
    }
  }

  size_t index = key - 1;
  Value prev = AtomicOperation::swap(value, &values_[index]);
  if (prev && destructors_[index] != NULL) {
    destructors_[index](prev);
  }

  return true;
}

}  // namespace amd
