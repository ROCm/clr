/* Copyright (c) 2017-present Advanced Micro Devices, Inc.

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

#ifndef WITHOUT_HSA_BACKEND

#include "top.hpp"
#include "hsa.h"

/*! \addtogroup HSA
 *  @{
 */

namespace roc {

//! Pro Device Interface
class IProDevice : public amd::HeapObject {
public:
  static IProDevice* Init(uint32_t bus, uint32_t device, uint32_t func);

  virtual void* AllocDmaBuffer(hsa_agent_t agent, size_t size, void** host_ptr) const = 0;
  virtual void FreeDmaBuffer(void* ptr) const = 0;
  virtual void GetAsicIdAndRevisionId(uint32_t* asic_id, uint32_t* rev_id) const = 0;

  IProDevice() {}
  virtual ~IProDevice() {}
};

}  // namespace roc

/**
 * @}
 */
#endif /*WITHOUT_HSA_BACKEND*/
