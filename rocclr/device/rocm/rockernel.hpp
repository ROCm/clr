/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#pragma once

#include <memory>
#include "acl.h"
#include "rocprogram.hpp"
#include "top.hpp"
#include "rocprintf.hpp"

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

#define MAX_INFO_STRING_LEN 0x40

class Kernel : public device::Kernel {
 public:
  Kernel(std::string name, Program* prog, const uint64_t& kernelCodeHandle,
         const uint32_t workgroupGroupSegmentByteSize,
         const uint32_t workitemPrivateSegmentByteSize, const uint32_t kernargSegmentByteSize,
         const uint32_t kernargSegmentAlignment);

  Kernel(std::string name, Program* prog);

  ~Kernel() {}

  //! Initializes the metadata required for this kernel
  virtual bool init() = 0;

  const Program* program() const { return static_cast<const Program*>(&prog_); }
};

class HSAILKernel : public roc::Kernel {
 public:
  HSAILKernel(std::string name, Program* prog, const uint64_t& kernelCodeHandle,
              const uint32_t workgroupGroupSegmentByteSize,
              const uint32_t workitemPrivateSegmentByteSize,
              const uint32_t kernargSegmentByteSize,
              const uint32_t kernargSegmentAlignment)
   : roc::Kernel(name, prog, kernelCodeHandle, workgroupGroupSegmentByteSize,
                 workitemPrivateSegmentByteSize, kernargSegmentByteSize, kernargSegmentAlignment) {
  }

  //! Initializes the metadata required for this kernel
  virtual bool init() final;
};

class LightningKernel : public roc::Kernel {
 public:
  LightningKernel(std::string name, Program* prog, const uint64_t& kernelCodeHandle,
                  const uint32_t workgroupGroupSegmentByteSize,
                  const uint32_t workitemPrivateSegmentByteSize,
                  const uint32_t kernargSegmentByteSize,
                  const uint32_t kernargSegmentAlignment)
   : roc::Kernel(name, prog, kernelCodeHandle, workgroupGroupSegmentByteSize,
                 workitemPrivateSegmentByteSize, kernargSegmentByteSize, kernargSegmentAlignment) {
  }

  LightningKernel(std::string name, Program* prog)
   : roc::Kernel(name, prog) {}

  //! Initializes the metadata required for this kernel
  virtual bool init() final;
};

}  // namespace roc

#endif  // WITHOUT_HSA_BACKEND
