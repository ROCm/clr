//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef ALLOC_HPP_
#define ALLOC_HPP_

#include "top.hpp"

namespace amd {

class AlignedMemory : public AllStatic {
 public:
  static void* allocate(size_t size, size_t alignment);

  static void deallocate(void* ptr);
};

class GuardedMemory : public AllStatic {
 public:
  static void* allocate(size_t size, size_t alignment, size_t guardSize);

  static void deallocate(void* ptr);
};

}  // namespace amd

#endif /*ALLOC_HPP_*/
