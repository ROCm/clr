/* Copyright (c) 2024 Advanced Micro Devices, Inc.

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

#include "device/pal/paldefs.hpp"
#include "palDeveloperHooks.h"

namespace amd::pal {

class Device;
class VirtualGPU;
class HSAILKernel;

// ================================================================================================
class ICaptureMgr {
 public:
  virtual bool Update(Pal::IPlatform* platform) = 0;

  virtual void PreDispatch(VirtualGPU* gpu,
                           const HSAILKernel& kernel,
                           size_t x, size_t y, size_t z) = 0;
  virtual void PostDispatch(VirtualGPU* gpu) = 0;

  virtual void FinishRGPTrace(VirtualGPU* gpu, bool aborted) = 0;

  virtual void WriteBarrierStartMarker(const VirtualGPU* gpu,
                                       const Pal::Developer::BarrierData& data) const = 0;
  virtual void WriteBarrierEndMarker(const VirtualGPU* gpu,
                                     const Pal::Developer::BarrierData& data) const = 0;

  virtual bool RegisterTimedQueue(uint32_t queue_id,
                                  Pal::IQueue* iQueue, bool* debug_vmid) const = 0;

  virtual Pal::Result TimedQueueSubmit(Pal::IQueue* queue, uint64_t cmdId,
                                       const Pal::SubmitInfo& submitInfo) const = 0;

  virtual uint64_t AddElfBinary(const void* exe_binary, size_t exe_binary_size,
                                const void* elf_binary, size_t elf_binary_size,
                                Pal::IGpuMemory* pGpuMemory, size_t offset) = 0;
};

} // namespace amd::pal
