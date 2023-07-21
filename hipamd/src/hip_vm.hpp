/* Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.

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

#ifndef HIP_SRC_HIP_VM_H
#define HIP_SRC_HIP_VM_H

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"

#include "platform/object.hpp"

hipError_t ihipFree(void* ptr);

namespace hip {

struct MemMapAllocUserData {
  void* ptr_;       // Original pointer of the allocation
  size_t size_;     // Aligned size of the allocation
  amd::Memory* va_; // Memory object for the virtual address

  MemMapAllocUserData(void* ptr, size_t size, amd::Memory* va) : ptr_(ptr), size_(size), va_(va) {}
};

class GenericAllocation : public amd::RuntimeObject {
  void* ptr_;                          //<! Device ptr
  size_t size_;                        //<! Allocated size
  hipMemAllocationProp properties_;    //<! Allocation Properties

public:
  GenericAllocation(void* ptr, size_t size, const hipMemAllocationProp& prop)
                     : ptr_(ptr), size_(size), properties_(prop) {}
  ~GenericAllocation() {
    hipError_t err = ihipFree(ptr_);
  }

  const hipMemAllocationProp& GetProperties() const { return properties_; }
  hipMemGenericAllocationHandle_t asMemGenericAllocationHandle() {
    return reinterpret_cast<hipMemGenericAllocationHandle_t>(this);
  }
  amd::Memory& asAmdMemory() {
    size_t discardOffset;
    return *getMemoryObject(genericAddress(), discardOffset);
  }
  void* genericAddress() const { return ptr_; }

  virtual ObjectType objectType() const { return ObjectTypeVMMAlloc; }
};
};

#endif //HIP_SRC_HIP_VM_H
