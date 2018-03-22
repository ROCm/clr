//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/pal/palconstbuf.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palsettings.hpp"

namespace pal {

// ================================================================================================
ManagedBuffer::ManagedBuffer(VirtualGPU& gpu, uint32_t size)
    : gpu_(gpu)
    , buffers_(MaxNumberOfBuffers)
    , activeBuffer_(0)
    , size_(size)
    , wrtOffset_(0)
    , lastWrtSize_(0)
    , wrtAddress_(nullptr) {}

// ================================================================================================
ManagedBuffer::~ManagedBuffer() {
  for (auto it : buffers_) {
    if (it->data() != nullptr) {
      it->unmap(&gpu_);
    }
    delete it;
  }
  amd::AlignedMemory::deallocate(sysMemCopy_);
}

// ================================================================================================
bool ManagedBuffer::create(Resource::MemoryType type, bool constbuf) {
  if (constbuf) {
    // Create sysmem copy for the constant buffer
    sysMemCopy_ = reinterpret_cast<address>(amd::AlignedMemory::allocate(
        gpu_.dev().info().maxParameterSize_, 256));
    if (sysMemCopy_ == nullptr) {
      LogPrintfError("We couldn't allocate sysmem copy for constant buffer, size(%d)!", size_);
      return false;
    }
    memset(sysMemCopy_, 0, gpu_.dev().info().maxParameterSize_);
  }

  for (uint i = 0; i < buffers_.size(); ++i) {
    buffers_[i] = new Memory(const_cast<pal::Device&>(gpu_.dev()), size_);
    if (nullptr == buffers_[i] || !buffers_[i]->create(type)) {
      LogPrintfError("We couldn't create HW constant buffer, size(%d)!", size_);
      return false;
    }
    void* wrtAddress = buffers_[i]->map(&gpu_);
    if (wrtAddress == nullptr) {
        LogPrintfError("We couldn't map HW constant buffer, size(%d)!", size_);
        return false;
    }
    // Make sure OCL touches every buffer in the queue to avoid delays on the first submit
    uint dummy = 0;
    static constexpr bool Wait = true;
    // Write 0 for the buffer paging by VidMM
    buffers_[i]->writeRawData(gpu_, 0, sizeof(dummy), &dummy, Wait);
  }
  wrtAddress_ = buffers_[activeBuffer_]->data();
  return true;
}

// ================================================================================================
bool ManagedBuffer::uploadDataToHw(uint32_t size) {
  static constexpr uint32_t HwCbAlignment = 256;

  // Align copy size on the vector's boundary
  uint32_t count = amd::alignUp(size, 16);
  wrtOffset_ += lastWrtSize_;

  // Check if CB has enough space for copy
  if ((wrtOffset_ + count) > size_) {
    // Get the next buffer in the list
    ++activeBuffer_;
    activeBuffer_ %= MaxNumberOfBuffers;
    // Make sure the buffer isn't busy
    buffers_[activeBuffer_]->wait(gpu_);
    wrtAddress_ = buffers_[activeBuffer_]->data();
    wrtOffset_ = 0;
    lastWrtSize_ = 0;
  }

  // Update memory with new CB data
  memcpy((reinterpret_cast<char*>(wrtAddress_) + wrtOffset_), sysMemCopy_, count);

  // Adjust the size by the HW CB buffer alignment
  lastWrtSize_ = amd::alignUp(size, HwCbAlignment);
  return true;
}

}  // namespace pal
