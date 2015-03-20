//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "sync.hpp"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Threading.h"


static llvm::sys::MutexImpl mtx;

namespace amdcl
{

void acquire_global_lock() {
  if (llvm::llvm_is_multithreaded())
    mtx.acquire();
}

void release_global_lock() {
  if (llvm::llvm_is_multithreaded())
    mtx.release();
}

} // namespace amdcl
