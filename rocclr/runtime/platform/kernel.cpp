//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "platform/kernel.hpp"
#include "platform/program.hpp"
#include "os/alloc.hpp"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/sampler.hpp"

namespace amd {

Kernel::Kernel(Program& program, const Symbol& symbol, const std::string& name)
    : program_(program), symbol_(symbol), name_(name) {
  const KernelSignature& s = signature();
  size_t stackSize = s.paramsSize();
  parameters_ = new (s) KernelParameters(s);
  fixme_guarantee(parameters_ != NULL && "out of memory");
  name_ += '\0';
}

Kernel::~Kernel() {
  // Release kernel object itself
  delete parameters_;
}

const device::Kernel* Kernel::getDeviceKernel(const Device& device, bool noAlias) const {
  return symbol_.getDeviceKernel(device, noAlias);
}

const KernelSignature& Kernel::signature() const { return symbol_.signature(); }

bool KernelParameters::check() {
  if (validated_) {
    return true;
  }

  for (size_t i = 0; i < signature_.numParameters(); ++i) {
    if (!test(i)) {
      return false;
    }
  }
  validated_ = true;

  return true;
}

size_t KernelParameters::localMemSize(size_t minDataTypeAlignment) const {
  size_t memSize = 0;

  for (size_t i = 0; i < signature_.numParameters(); ++i) {
    const KernelParameterDescriptor& desc = signature_.at(i);
    if (desc.size_ == 0) {
      memSize = alignUp(memSize, minDataTypeAlignment) +
          *reinterpret_cast<const size_t*>(values_ + desc.offset_);
    }
  }
  return memSize;
}

void KernelParameters::set(size_t index, size_t size, const void* value, bool svmBound) {
  const KernelParameterDescriptor& desc = signature_.at(index);

  void* param = values_ + desc.offset_;
  assert((desc.type_ == T_POINTER || value != NULL || desc.size_ == 0) &&
         "not a valid local mem arg");

  uint32_t uint32_value = 0;
  uint64_t uint64_value = 0;

  if (desc.type_ == T_POINTER && desc.size_ != 0) {
    if (svmBound) {
      LP64_SWITCH(uint32_value, uint64_value) = (LP64_SWITCH(uint32_t, uint64_t))value;
      svmBound_[index] = true;
    } else if ((value == NULL) || (static_cast<const cl_mem*>(value) == NULL)) {
      LP64_SWITCH(uint32_value, uint64_value) = 0;
    } else {
      // convert cl_mem to amd::Memory*
      LP64_SWITCH(uint32_value, uint64_value) =
          (uintptr_t)as_amd(*static_cast<const cl_mem*>(value));
    }
  } else if (desc.type_ == T_SAMPLER) {
    // convert cl_sampler to amd::Sampler*
    amd::Sampler* sampler = as_amd(*static_cast<const cl_sampler*>(value));
    LP64_SWITCH(uint32_value, uint64_value) = (uintptr_t)sampler;
  } else if (desc.type_ == T_QUEUE) {
    // convert cl_command_queue to amd::DeviceQueue*
    amd::DeviceQueue* queue = as_amd(*static_cast<const cl_command_queue*>(value))->asDeviceQueue();
    LP64_SWITCH(uint32_value, uint64_value) = (uintptr_t)queue;
  } else
    switch (desc.size_) {
      case 1:
        uint32_value = *static_cast<const uint8_t*>(value);
        break;
      case 2:
        uint32_value = *static_cast<const uint16_t*>(value);
        break;
      case 4:
        uint32_value = *static_cast<const uint32_t*>(value);
        break;
      case 8:
        uint64_value = *static_cast<const uint64_t*>(value);
        break;
      default:
        break;
    }

  switch (desc.size_) {
    case 0 /*local mem*/:
      *static_cast<size_t*>(param) = size;
      break;
    case sizeof(uint32_t):
      *static_cast<uint32_t*>(param) = uint32_value;
      break;
    case sizeof(uint64_t):
      *static_cast<uint64_t*>(param) = uint64_value;
      break;
    default:
      ::memcpy(param, value, size);
      break;
  }

  defined_[index] = true;
}

address KernelParameters::capture(const Device& device) {
  const size_t stackSize = signature_.paramsSize();
  //! Information about which arguments are SVM pointers is stored after
  // the actual parameters, but only if the device has any SVM capability
  const size_t svmInfoSize =
      device.info().svmCapabilities_ ? signature_.numParameters() * sizeof(bool) : 0;
  const size_t execInfoSize = getNumberOfSvmPtr() * sizeof(void*);
  address mem = (address)AlignedMemory::allocate(stackSize + svmInfoSize + execInfoSize,
                                                 PARAMETERS_MIN_ALIGNMENT);

  address last = mem + stackSize;
  if (mem != NULL) {
    ::memcpy(mem, values_, stackSize);

    for (size_t i = 0; i < signature_.numParameters(); ++i) {
      const KernelParameterDescriptor& desc = signature_.at(i);
      if (desc.type_ == T_POINTER && desc.size_ != 0 && !svmBound_[i]) {
        Memory* memArg = *(Memory**)(mem + desc.offset_);
        if (memArg != NULL) {
          memArg->retain();
        }
      } else if (desc.type_ == T_SAMPLER) {
        Sampler* samplerArg = *(Sampler**)(mem + desc.offset_);
        if (samplerArg != NULL) {
          samplerArg->retain();
        }
      } else if (desc.type_ == T_QUEUE) {
        DeviceQueue* queue = *(DeviceQueue**)(mem + desc.offset_);
        if (queue != NULL) {
          queue->retain();
        }
      }
    }
    ::memcpy(last, svmBound_, svmInfoSize);
    last += svmInfoSize;

    if (0 != execInfoSize) {
      ::memcpy(last, &execSvmPtr_[0], execInfoSize);
    }
    execInfoOffset_ = stackSize + svmInfoSize;
  }

  return mem;
}

bool KernelParameters::boundToSvmPointer(const Device& device, const_address capturedParameter,
                                         size_t index) const {
  if (!device.info().svmCapabilities_) {
    return false;
  }
  //! Information about which arguments are SVM pointers is stored after
  // actual parameters
  const bool* svmBound = reinterpret_cast<const bool*>(capturedParameter + signature_.paramsSize());
  return svmBound[index];
}

void KernelParameters::release(address mem, const amd::Device& device) const {
  if (mem == NULL) {
    // nothing to do!
    return;
  }

  for (size_t i = 0; i < signature_.numParameters(); ++i) {
    const KernelParameterDescriptor& desc = signature_.at(i);
    if (desc.type_ == T_POINTER && desc.size_ != 0 && !boundToSvmPointer(device, mem, i)) {
      Memory* memArg = *(Memory**)(mem + desc.offset_);
      if (memArg != NULL) {
        memArg->release();
      }
    } else if (desc.type_ == T_SAMPLER) {
      Sampler* samplerArg = *(Sampler**)(mem + desc.offset_);
      if (samplerArg != NULL) {
        samplerArg->release();
      }
    } else if (desc.type_ == T_QUEUE) {
      DeviceQueue* queue = *(DeviceQueue**)(mem + desc.offset_);
      if (queue != NULL) {
        queue->release();
      }
    }
  }

  AlignedMemory::deallocate(mem);
}


KernelSignature::KernelSignature(const std::vector<KernelParameterDescriptor>& params,
                                 const std::string& attrib)
    : params_(params), paramsSize_(0), attributes_(attrib) {
  if (params.size() > 0) {
    KernelParameterDescriptor last = params.back();

    size_t lastSize = last.size_;
    if (lastSize == 0 /* local mem */) {
      lastSize = sizeof(cl_mem);
    }
    paramsSize_ = last.offset_ + alignUp(lastSize, sizeof(intptr_t));
  }
}

}  // namespace amd
