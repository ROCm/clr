//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CPUKERNEL_HPP_
#define CPUKERNEL_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include <amdocl/cl_kernel.h>

#include "device/cpu/cpumapping.hpp"

//! \namespace cpu CPU Device Implementation
namespace cpu {

//! \class CPU kernel
class Kernel : public device::Kernel {
 private:
  const void* entryPoint_;  //!< entry for the kernel

  std::vector<std::pair<size_t, size_t> > args_;
  std::vector<std::pair<HCtoDCmap, size_t> > HCtoDCmaps_;
  std::vector<HCtoDCmap> internal_maps_;

 public:
  uint nature_;       //!< kernel's nature
  uint privateSize_;  //!< WorkItem's private memory size (in bytes)

 private:
  //! Disable default copy constructor
  Kernel(const Kernel&);
  //! Disable operator=
  Kernel& operator=(const Kernel&);

 public:
  void addArg(size_t size, size_t alignment) {
    args_.push_back(std::pair<size_t, size_t>(size, alignment));
  }

  size_t getArgSize(int argIndex) const { return args_[argIndex].first; }

  size_t getArgAlignment(int argIndex) const { return args_[argIndex].second; }

  void addInternalMap(HCtoDCmap* new_map) {
    if (new_map != NULL) {
      internal_maps_.push_back(*new_map);
      this->addInternalMap(new_map->internal_field_map);
      this->addInternalMap(new_map->next_field_map);
    } else
      return;
  }

  void addHCtoDCmap(HCtoDCmap* new_map) {
    if (new_map != NULL) {
      if (HCtoDCmaps_.size() > 0)
        HCtoDCmaps_.push_back(std::pair<HCtoDCmap, size_t>(*new_map, HCtoDCmaps_.back().second));
      else
        HCtoDCmaps_.push_back(std::pair<HCtoDCmap, size_t>(*new_map, 0));
    } else
      return;
  }

  HCtoDCmap getHCtoDCmap(int mapIndex) const { return HCtoDCmaps_[mapIndex].first; }


  uint getArgNumber() { return HCtoDCmaps_.size(); }

  //! Default constructor
  Kernel(const std::string& name)
      : device::Kernel(name), entryPoint_(NULL), nature_(0), privateSize_(CLK_PRIVATE_MEMORY_SIZE) {
    workGroupInfo_.size_ = CPU_MAX_WORKGROUP_SIZE;
  }

  //! Default destructor
  ~Kernel() {}

  //! Returns the CPU kernel entry point
  const void* getEntryPoint() const { return entryPoint_; }

  //! Sets the CPU kernel entry point
  void setEntryPoint(const void* entryPoint) { entryPoint_ = entryPoint; }

  //! Returns true if the kernel has a call to barrier
  bool hasBarrier() const { return 0 != (nature_ & KN_HAS_BARRIER); }

  //! Returns the private memory size of a single WorkItem
  uint getWorkItemPrivateMemSize() const { return privateSize_; }
};

}  // namespace cpu

#endif  // CPUKERNEL_HPP_
