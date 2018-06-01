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
  parameters_ = new (signature()) KernelParameters(const_cast<KernelSignature&>(signature()));
  fixme_guarantee(parameters_ != NULL && "out of memory");
  name_ += '\0';
}

Kernel::Kernel(const Kernel& rhs)
    : program_(rhs.program_()), symbol_(rhs.symbol_), name_(rhs.name_) {
  parameters_ = new(signature()) KernelParameters(*rhs.parameters_);
  fixme_guarantee(parameters_ != NULL && "out of memory");
}

Kernel::~Kernel() {
  // Release kernel object itself
  delete parameters_;
}

const device::Kernel* Kernel::getDeviceKernel(const Device& device) const {
  return symbol_.getDeviceKernel(device);
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
  KernelParameterDescriptor& desc = signature_.params()[index];

  void* param = values_ + desc.offset_;
  assert((desc.type_ == T_POINTER || value != NULL || desc.size_ == 0) &&
         "not a valid local mem arg");

  uint32_t uint32_value = 0;
  uint64_t uint64_value = 0;

  if (desc.type_ == T_POINTER && desc.size_ != 0) {
    if (svmBound) {
      desc.info_.rawPointer_ = true;
      LP64_SWITCH(uint32_value, uint64_value) = *(LP64_SWITCH(uint32_t*, uint64_t*))value;
      memoryObjects_[desc.info_.arrayIndex_] = amd::MemObjMap::FindMemObj(
        *reinterpret_cast<const void* const*>(value));
    } else if ((value == NULL) || (static_cast<const cl_mem*>(value) == NULL)) {
      desc.info_.rawPointer_ = false;
      LP64_SWITCH(uint32_value, uint64_value) = 0;
      memoryObjects_[desc.info_.arrayIndex_] = nullptr;
    } else {
      desc.info_.rawPointer_ = false;
      // convert cl_mem to amd::Memory*
      memoryObjects_[desc.info_.arrayIndex_] = as_amd(*static_cast<const cl_mem*>(value));
    }
  } else if (desc.type_ == T_SAMPLER) {
    // convert cl_sampler to amd::Sampler*
    samplerObjects_[desc.info_.arrayIndex_] =
      as_amd(*static_cast<const cl_sampler*>(value));
  } else if (desc.type_ == T_QUEUE) {
    // convert cl_command_queue to amd::DeviceQueue*
    queueObjects_[desc.info_.arrayIndex_] =
      as_amd(*static_cast<const cl_command_queue*>(value))->asDeviceQueue();
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

  desc.info_.defined_ = true;
}

address KernelParameters::capture(const Device& device, cl_int* error) {
  *error = CL_SUCCESS;
  //! Information about which arguments are SVM pointers is stored after
  // the actual parameters, but only if the device has any SVM capability
  const size_t execInfoSize = getNumberOfSvmPtr() * sizeof(void*);

  address mem = reinterpret_cast<address>(AlignedMemory::allocate(
    totalSize_ + execInfoSize, PARAMETERS_MIN_ALIGNMENT));

  if (mem != nullptr) {
    ::memcpy(mem, values_, totalSize_);

    for (size_t i = 0; i < signature_.numParameters(); ++i) {
      const KernelParameterDescriptor& desc = signature_.at(i);
      if (desc.type_ == T_POINTER && desc.size_ != 0) {
        Memory* memArg = memoryObjects_[desc.info_.arrayIndex_];
        if (memArg != nullptr) {
          memArg->retain();
          device::Memory* devMem = memArg->getDeviceMemory(device);
          if (nullptr == devMem) {
            LogPrintfError("Can't allocate memory size - 0x%08X bytes!", memArg->getSize());
            *error = CL_MEM_OBJECT_ALLOCATION_FAILURE;
            AlignedMemory::deallocate(mem);
            return nullptr;
          }
          // Write GPU VA addreess to the arguments
          if (!desc.info_.rawPointer_) {
            *reinterpret_cast<uintptr_t*>(mem + desc.offset_) = static_cast<uintptr_t>
              (devMem->virtualAddress());
          }
        } else if (desc.info_.rawPointer_) {
          if (!device.isFineGrainedSystem(true)) {
          }
        }
      } else if (desc.type_ == T_SAMPLER) {
        Sampler* samplerArg = samplerObjects_[desc.info_.arrayIndex_];
        if (samplerArg != nullptr) {
          samplerArg->retain();
          // todo: It's uint64_t type
          *reinterpret_cast<uintptr_t*>(mem + desc.offset_) = static_cast<uintptr_t>(
            samplerArg->getDeviceSampler(device)->hwSrd());
        }
      } else if (desc.type_ == T_QUEUE) {
        DeviceQueue* queue = queueObjects_[desc.info_.arrayIndex_];
        if (queue != nullptr) {
          queue->retain();
          // todo: It's uint64_t type
          *reinterpret_cast<uintptr_t*>(mem + desc.offset_) = 0;
        }
      }
    }

    execInfoOffset_ = totalSize_;
    address last = mem + execInfoOffset_;
    if (0 != execInfoSize) {
      ::memcpy(last, &execSvmPtr_[0], execInfoSize);
    }
  } else {
    *error = CL_OUT_OF_HOST_MEMORY;
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
  if (mem == nullptr) {
    // nothing to do!
    return;
  }

  amd::Memory* const* memories = reinterpret_cast<amd::Memory* const*>(mem + memoryObjOffset());
  for (uint32_t i = 0; i < signature_.numMemories(); ++i) {
    Memory* memArg = memories[i];
    if (memArg != nullptr) {
      memArg->release();
    }
  }
  if (signature_.numSamplers() > 0) {
    amd::Sampler* const* samplers = reinterpret_cast<amd::Sampler* const*>(mem + samplerObjOffset());
    for (uint32_t i = 0; i < signature_.numSamplers(); ++i) {
      Sampler* samplerArg = samplers[i];
      if (samplerArg != nullptr) {
        samplerArg->release();
      }
    }
  }
  if (signature_.numQueues() > 0) {
    amd::DeviceQueue* const* queues = reinterpret_cast<amd::DeviceQueue* const*>(mem + queueObjOffset());
    for (uint32_t i = 0; i < signature_.numQueues(); ++i) {
      DeviceQueue* queue = queues[i];
      if (queue != nullptr) {
        queue->release();
      }
    }
  }

  AlignedMemory::deallocate(mem);
}

KernelSignature::KernelSignature(const std::vector<KernelParameterDescriptor>& params,
  const std::string& attrib)
  : params_(params)
  , attributes_(attrib)
  , paramsSize_(0)
  , numMemories_(0)
  , numSamplers_(0)
  , numQueues_(0) {
  size_t maxOffset = 0;
  size_t last = 0;
  // Find the last entry
  for (size_t i = 0; i < params.size(); ++i) {
    const KernelParameterDescriptor& desc = params[i];
    // Serach for the max offset, since due to the pass by reference
    // we can't rely on the last argument as the max offset
    if (maxOffset < desc.offset_) {
      maxOffset = desc.offset_;
      last = i;
    }
    // Collect all OCL memory objects
    if (desc.type_ == T_POINTER && desc.size_ != 0) {
      params_[i].info_.arrayIndex_ = numMemories_;
      ++numMemories_;
    }
    // Collect all OCL sampler objects
    else if (desc.type_ == T_SAMPLER) {
      params_[i].info_.arrayIndex_ = numSamplers_;
      ++numSamplers_;
    }
    // Collect all OCL queues
    else if (desc.type_ == T_QUEUE) {
      params_[i].info_.arrayIndex_ = numQueues_   ;
      ++numQueues_;
    }
  }

  if (params.size() > 0) {
    size_t lastSize = params[last].size_;
    if (lastSize == 0 /* local mem */) {
      lastSize = sizeof(cl_mem);
    }
    paramsSize_ = params[last].offset_ + alignUp(lastSize, sizeof(intptr_t));
  }
}
}  // namespace amd
