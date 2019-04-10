//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
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

  const uint64_t& KernelCodeHandle() { return kernelCodeHandle_; }

  const uint32_t WorkgroupGroupSegmentByteSize() const { return workgroupGroupSegmentByteSize_; }

  const uint32_t workitemPrivateSegmentByteSize() const { return workitemPrivateSegmentByteSize_; }

  const uint32_t KernargSegmentByteSize() const { return kernargSegmentByteSize_; }

  const uint8_t KernargSegmentAlignment() const { return kernargSegmentAlignment_; }

  ~Kernel() {}

  //! Initializes the metadata required for this kernel
  virtual bool init() = 0;

  const Program* program() const { return static_cast<const Program*>(&prog_); }

 protected:
//  Program* program_;                      //!< The roc::Program context
  uint64_t kernelCodeHandle_;             //!< Kernel code handle (aka amd_kernel_code_t)
  uint32_t workgroupGroupSegmentByteSize_;
  uint32_t workitemPrivateSegmentByteSize_;
  uint32_t kernargSegmentByteSize_;
  uint32_t kernargSegmentAlignment_;
  size_t kernelDirectiveOffset_;
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
