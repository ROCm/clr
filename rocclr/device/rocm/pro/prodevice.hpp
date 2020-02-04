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

#include "profuncs.hpp"
#include "prodriver.hpp"
#include "thread/monitor.hpp"
#include <unordered_map>

/*! \addtogroup HSA
 *  @{
 */

//! HSA Device Implementation
namespace roc {

class ProDevice : public IProDevice {
public:
  static bool DrmInit();

  ProDevice()
    : file_desc_(0)
    , major_ver_(0)
    , minor_ver_(0)
    , dev_handle_(nullptr)
    , alloc_ops_(nullptr) {}
  virtual ~ProDevice() override;

  bool Create(uint32_t bus, uint32_t device, uint32_t func);

  virtual void* AllocDmaBuffer(
      hsa_agent_t agent, size_t size, void** host_ptr) const override;
  virtual void FreeDmaBuffer(void* ptr) const override;
  virtual void GetAsicIdAndRevisionId(uint32_t* asic_id, uint32_t* rev_id) const override
  {
    *asic_id    = gpu_info_.asic_id;
    *rev_id     = gpu_info_.pci_rev_id;
  }

private:
  static void*          lib_drm_handle_;
  static bool           initialized_;
  static drm::Funcs     funcs_;
  const drm::Funcs& Funcs() const { return funcs_; }

  int32_t               file_desc_;   //!< File descriptor for the device
  uint32_t              major_ver_;   //!< Major driver version
  uint32_t              minor_ver_;   //!< Minor driver version
  amdgpu_device_handle  dev_handle_;  //!< AMD gpu device handle
  amdgpu_gpu_info       gpu_info_;    //!< GPU info structure
  amdgpu_heap_info      heap_info_;   //!< Information about memory
  mutable std::unordered_map<void*, std::pair<amdgpu_bo_handle, uint32_t>> allocs_; //!< Alloced memory mapping
  amd::Monitor*         alloc_ops_;   //!< Serializes memory allocations/destructions
};

}  // namespace roc

/**
 * @}
 */
#endif /*WITHOUT_HSA_BACKEND*/
