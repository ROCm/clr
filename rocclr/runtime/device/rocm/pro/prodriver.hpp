//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//

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
