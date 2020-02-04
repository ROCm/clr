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
