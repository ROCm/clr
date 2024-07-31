/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include "platform/commandqueue.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocsched.hpp"
#include "utils/debug.hpp"
#include <algorithm>

namespace amd::roc {
DmaBlitManager::DmaBlitManager(VirtualGPU& gpu, Setup setup)
    : HostBlitManager(gpu, setup),
      MinSizeForPinnedTransfer(dev().settings().pinnedMinXferSize_),
      completeOperation_(false),
      context_(nullptr),
      sdmaEngineRetainCount_(0) {
        dev().getSdmaRWMasks(&sdmaEngineReadMask_, &sdmaEngineWriteMask_);
      }

inline void DmaBlitManager::synchronize() const {
  if (syncOperation_) {
    gpu().releaseGpuMemoryFence();
    gpu().releasePinnedMem();
  }
}

inline Memory& DmaBlitManager::gpuMem(device::Memory& mem) const {
  return static_cast<Memory&>(mem);
}

// ================================================================================================
bool DmaBlitManager::readMemoryStaged(Memory& srcMemory, void* dstHost, Memory& xferBuf,
                                      size_t origin, size_t& offset, size_t& totalSize,
                                      size_t xferSize) const {
  const_address src = srcMemory.getDeviceMemory();
  address staging = xferBuf.getDeviceMemory();

  // Copy data from device to host
  src += origin + offset;
  address dst = reinterpret_cast<address>(dstHost) + offset;
  bool ret = hsaCopyStaged(src, dst, totalSize, staging, false);

  return ret;
}

// ================================================================================================
bool DmaBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                const amd::Coord3D& origin, const amd::Coord3D& size,
                                bool entire, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation
  gpu().releaseGpuMemoryFence(kSkipCpuWait);

  // Use host copy if memory has direct access
  if (setup_.disableReadBuffer_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
    gpu().Barriers().WaitCurrent();
    return HostBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
  } else {
    size_t srcSize = size[0];
    size_t offset = 0;
    size_t pinSize = dev().settings().pinnedXferSize_;
    pinSize = std::min(pinSize, srcSize);

    // Check if a pinned transfer can be executed
    if (pinSize && (srcSize > MinSizeForPinnedTransfer)) {
      // Align offset to 4K boundary
      char* tmpHost = const_cast<char*>(
          amd::alignDown(reinterpret_cast<const char*>(dstHost), PinnedMemoryAlignment));

      // Find the partial size for unaligned copy
      size_t partial = reinterpret_cast<const char*>(dstHost) - tmpHost;

      amd::Memory* pinned = nullptr;
      bool first = true;
      size_t tmpSize;
      size_t pinAllocSize;

      // Copy memory, using pinning
      while (srcSize > 0) {
        // If it's the first iterarion, then readjust the copy size
        // to include alignment
        if (first) {
          pinAllocSize = amd::alignUp(pinSize + partial, PinnedMemoryAlignment);
          tmpSize = std::min(pinAllocSize - partial, srcSize);
          first = false;
        } else {
          tmpSize = std::min(pinSize, srcSize);
          pinAllocSize = amd::alignUp(tmpSize, PinnedMemoryAlignment);
          partial = 0;
        }
        amd::Coord3D dst(partial, 0, 0);
        amd::Coord3D srcPin(origin[0] + offset, 0, 0);
        amd::Coord3D copySizePin(tmpSize, 0, 0);
        size_t partial2;

        // Allocate a GPU resource for pinning
        pinned = pinHostMemory(tmpHost, pinAllocSize, partial2);
        if (pinned != nullptr) {
          // Get device memory for this virtual device
          Memory* dstMemory = dev().getRocMemory(pinned);
          const KernelBlitManager *kb = dynamic_cast<const KernelBlitManager*>(this);
          if (!kb->copyBuffer(gpuMem(srcMemory), *dstMemory, srcPin, dst,
                              copySizePin)) {
            LogWarning("DmaBlitManager::readBuffer failed a pinned copy!");
            gpu().addPinnedMem(pinned);
            break;
          }
          gpu().addPinnedMem(pinned);
        } else {
          LogWarning("DmaBlitManager::readBuffer failed to pin a resource!");
          break;
        }
        srcSize -= tmpSize;
        offset += tmpSize;
        tmpHost = reinterpret_cast<char*>(tmpHost) + tmpSize + partial;
      }
    }

    if (0 != srcSize) {
      Memory& xferBuf = dev().xferRead().acquire();

      // Read memory using a staging resource
      if (!readMemoryStaged(gpuMem(srcMemory), dstHost, xferBuf, origin[0], offset, srcSize,
                            srcSize)) {
        LogError("DmaBlitManager::readBuffer failed!");
        return false;
      }

      dev().xferRead().release(gpu(), xferBuf);
    }
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::readBufferRect(device::Memory& srcMemory, void* dstHost,
                                    const amd::BufferRect& bufRect,
                                    const amd::BufferRect& hostRect,
                                    const amd::Coord3D& size, bool entire,
                                    amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation
  gpu().releaseGpuMemoryFence();

  // Use host copy if memory has direct access
  if (setup_.disableReadBufferRect_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
    gpu().Barriers().WaitCurrent();
    return HostBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire, copyMetadata);
  } else {
    Memory& xferBuf = dev().xferRead().acquire();
    address staging = xferBuf.getDeviceMemory();
    const_address src = gpuMem(srcMemory).getDeviceMemory();

    size_t srcOffset;
    size_t dstOffset;

    for (size_t z = 0; z < size[2]; ++z) {
      for (size_t y = 0; y < size[1]; ++y) {
        srcOffset = bufRect.offset(0, y, z);
        dstOffset = hostRect.offset(0, y, z);

        // Copy data from device to host - line by line
        address dst = reinterpret_cast<address>(dstHost) + dstOffset;
        bool retval = hsaCopyStaged(src + srcOffset, dst, size[0], staging, false);
        if (!retval) {
          return retval;
        }
      }
    }
    dev().xferRead().release(gpu(), xferBuf);
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::readImage(device::Memory& srcMemory, void* dstHost,
                               const amd::Coord3D& origin, const amd::Coord3D& size,
                               size_t rowPitch, size_t slicePitch,
                               bool entire, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation
  gpu().releaseGpuMemoryFence();

  if (setup_.disableReadImage_) {
    return HostBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                      entire, copyMetadata);
  } else {
    //! @todo Add HW accelerated path
    return HostBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                      entire, copyMetadata);
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::writeMemoryStaged(const void* srcHost, Memory& dstMemory, Memory& xferBuf,
                                       size_t origin, size_t& offset, size_t& totalSize,
                                       size_t xferSize) const {
  address dst = dstMemory.getDeviceMemory();
  address staging = xferBuf.getDeviceMemory();

  // Copy data from host to device
  dst += origin + offset;
  const_address src = reinterpret_cast<const_address>(srcHost) + offset;
  bool retval = hsaCopyStaged(src, dst, totalSize, staging, true);

  return retval;
}

// ================================================================================================
bool DmaBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                 const amd::Coord3D& origin, const amd::Coord3D& size,
                                 bool entire, amd::CopyMetadata copyMetadata) const {
  // Use host copy if memory has direct access
  if (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess() ||
      gpuMem(dstMemory).IsPersistentDirectMap()) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
  } else {
    // HSA copy functionality with a possible async operation
    gpu().releaseGpuMemoryFence(kSkipCpuWait);

    size_t dstSize = size[0];
    size_t tmpSize = 0;
    size_t offset = 0;
    size_t pinSize = dev().settings().pinnedXferSize_;
    pinSize = std::min(pinSize, dstSize);

    // Check if a pinned transfer can be executed
    if (pinSize && (dstSize > MinSizeForPinnedTransfer)) {
      // Align offset to 4K boundary
      char* tmpHost = const_cast<char*>(
          amd::alignDown(reinterpret_cast<const char*>(srcHost), PinnedMemoryAlignment));

      // Find the partial size for unaligned copy
      size_t partial = reinterpret_cast<const char*>(srcHost) - tmpHost;

      amd::Memory* pinned = nullptr;
      bool first = true;
      size_t tmpSize;
      size_t pinAllocSize;

      // Copy memory, using pinning
      while (dstSize > 0) {
        // If it's the first iterarion, then readjust the copy size
        // to include alignment
        if (first) {
          pinAllocSize = amd::alignUp(pinSize + partial, PinnedMemoryAlignment);
          tmpSize = std::min(pinAllocSize - partial, dstSize);
          first = false;
        } else {
          tmpSize = std::min(pinSize, dstSize);
          pinAllocSize = amd::alignUp(tmpSize, PinnedMemoryAlignment);
          partial = 0;
        }
        amd::Coord3D src(partial, 0, 0);
        amd::Coord3D dstPin(origin[0] + offset, 0, 0);
        amd::Coord3D copySizePin(tmpSize, 0, 0);
        size_t partial2;

        // Allocate a GPU resource for pinning
        pinned = pinHostMemory(tmpHost, pinAllocSize, partial2);

        if (pinned != nullptr) {
          // Get device memory for this virtual device
          Memory* srcMemory = dev().getRocMemory(pinned);
          const KernelBlitManager *kb = dynamic_cast<const KernelBlitManager*>(this);
          if (!kb->copyBuffer(*srcMemory, gpuMem(dstMemory), src, dstPin,
                              copySizePin)) {
            LogWarning("DmaBlitManager::writeBuffer failed a pinned copy!");
            gpu().addPinnedMem(pinned);
            break;
          }
          gpu().addPinnedMem(pinned);
        } else {
          LogWarning("DmaBlitManager::writeBuffer failed to pin a resource!");
          break;
        }
        dstSize -= tmpSize;
        offset += tmpSize;
        tmpHost = reinterpret_cast<char*>(tmpHost) + tmpSize + partial;
      }
    }

    if (dstSize != 0) {
      Memory& xferBuf = dev().xferWrite().acquire();

      // Write memory using a staging resource
      if (!writeMemoryStaged(srcHost, gpuMem(dstMemory), xferBuf, origin[0], offset, dstSize,
                             dstSize)) {
        LogError("DmaBlitManager::writeBuffer failed!");
        return false;
      }

      gpu().addXferWrite(xferBuf);
    }
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::writeBufferRect(const void* srcHost, device::Memory& dstMemory,
                                     const amd::BufferRect& hostRect,
                                     const amd::BufferRect& bufRect, const amd::Coord3D& size,
                                     bool entire, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation
  gpu().releaseGpuMemoryFence();

  // Use host copy if memory has direct access
  if (setup_.disableWriteBufferRect_ || dstMemory.isHostMemDirectAccess() ||
      gpuMem(dstMemory).IsPersistentDirectMap()) {
    return HostBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                            copyMetadata);
  } else {
    Memory& xferBuf = dev().xferWrite().acquire();
    address staging = xferBuf.getDeviceMemory();
    address dst = static_cast<roc::Memory&>(dstMemory).getDeviceMemory();

    size_t srcOffset;
    size_t dstOffset;

    for (size_t z = 0; z < size[2]; ++z) {
      for (size_t y = 0; y < size[1]; ++y) {
        srcOffset = hostRect.offset(0, y, z);
        dstOffset = bufRect.offset(0, y, z);

        // Copy data from host to device - line by line
        const_address src = reinterpret_cast<const_address>(srcHost) + srcOffset;
        bool retval = hsaCopyStaged(src, dst + dstOffset, size[0], staging, true);
        if (!retval) {
          return retval;
        }
      }
    }
    gpu().addXferWrite(xferBuf);
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                const amd::Coord3D& origin, const amd::Coord3D& size,
                                size_t rowPitch, size_t slicePitch, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation
  gpu().releaseGpuMemoryFence();

  if (setup_.disableWriteImage_) {
    return HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                       entire, copyMetadata);
  } else {
    //! @todo Add HW accelerated path
    return HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                       entire, copyMetadata);
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
  if (setup_.disableCopyBuffer_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached() &&
      (dev().agent_profile() != HSA_PROFILE_FULL) && dstMemory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size, false,
                                       copyMetadata);
  } else {
    return hsaCopy(gpuMem(srcMemory), gpuMem(dstMemory), srcOrigin, dstOrigin, size, copyMetadata);
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::copyBufferRect(device::Memory& srcMemory, device::Memory& dstMemory,
                                    const amd::BufferRect& srcRect, const amd::BufferRect& dstRect,
                                    const amd::Coord3D& size, bool entire,
                                    amd::CopyMetadata copyMetadata) const {
  if (setup_.disableCopyBufferRect_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached() &&
       dstMemory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    return HostBlitManager::copyBufferRect(srcMemory, dstMemory, srcRect, dstRect, size, entire,
                                           copyMetadata);
  } else {
    gpu().releaseGpuMemoryFence(kSkipCpuWait);

    void* src = gpuMem(srcMemory).getDeviceMemory();
    void* dst = gpuMem(dstMemory).getDeviceMemory();

    // Detect the agents for memory allocations
    const hsa_agent_t srcAgent =
        (srcMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();
    const hsa_agent_t dstAgent =
        (dstMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();

    bool isSubwindowRectCopy = true;
    hsa_amd_copy_direction_t direction = hsaHostToHost;

    hsa_agent_t agent = dev().getBackendDevice();
    //Determine copy direction
    if (srcMemory.isHostMemDirectAccess() && !dstMemory.isHostMemDirectAccess()) {
      direction = hsaHostToDevice;
    } else if (!srcMemory.isHostMemDirectAccess() && dstMemory.isHostMemDirectAccess()) {
      direction = hsaDeviceToHost;
    } else if (!srcMemory.isHostMemDirectAccess() && !dstMemory.isHostMemDirectAccess()) {
      direction = hsaDeviceToDevice;
    }

    hsa_pitched_ptr_t srcMem = { (reinterpret_cast<address>(src) + srcRect.offset(0, 0, 0)),
                                srcRect.rowPitch_,
                                srcRect.slicePitch_ };

    hsa_pitched_ptr_t dstMem = { (reinterpret_cast<address>(dst) + dstRect.offset(0, 0, 0)),
                                dstRect.rowPitch_,
                                dstRect.slicePitch_ };

    hsa_dim3_t dim = { static_cast<uint32_t>(size[0]),
                      static_cast<uint32_t>(size[1]),
                      static_cast<uint32_t>(size[2]) };
    hsa_dim3_t offset = { 0, 0 ,0 };


    if ((srcRect.rowPitch_ % 4 != 0)    ||
        (srcRect.slicePitch_ % 4 != 0)  ||
        (dstRect.rowPitch_ % 4 != 0)    ||
        (dstRect.slicePitch_ % 4 != 0)) {
      isSubwindowRectCopy = false;
    }

    HwQueueEngine engine = HwQueueEngine::Unknown;
    if ((srcAgent.handle == dev().getCpuAgent().handle) &&
        (dstAgent.handle != dev().getCpuAgent().handle)) {
      engine = HwQueueEngine::SdmaWrite;
    } else if ((srcAgent.handle != dev().getCpuAgent().handle) &&
              (dstAgent.handle == dev().getCpuAgent().handle)) {
      engine = HwQueueEngine::SdmaRead;
    }

    auto wait_events = gpu().Barriers().WaitingSignal(engine);

    if (isSubwindowRectCopy ) {
      hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp());

      // Copy memory line by line
      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
              "HSA Async Copy Rect dst=0x%zx, src=0x%zx, wait_event=0x%zx "
              "completion_signal=0x%zx", dstMem.base, srcMem.base,
              (wait_events.size() != 0) ? wait_events[0].handle : 0, active.handle);

      hsa_status_t status = hsa_amd_memory_async_copy_rect(&dstMem, &offset,
          &srcMem, &offset, &dim, agent, direction, wait_events.size(), wait_events.data(),
          active);
      if (status != HSA_STATUS_SUCCESS) {
        gpu().Barriers().ResetCurrentSignal();
        LogPrintfError("DMA buffer failed with code %d", status);
        return false;
      }
    } else {
      // Fall to line by line copies
      const hsa_signal_value_t kInitVal = size[2] * size[1];
      hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitVal, gpu().timestamp());

      for (size_t z = 0; z < size[2]; ++z) {
        for (size_t y = 0; y < size[1]; ++y) {
          size_t srcOffset = srcRect.offset(0, y, z);
          size_t dstOffset = dstRect.offset(0, y, z);

          // Copy memory line by line
          ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
                  "HSA Async Copy wait_event=0x%zx, completion_signal=0x%zx",
                  (wait_events.size() != 0) ? wait_events[0].handle : 0, active.handle);
          hsa_status_t status = hsa_amd_memory_async_copy(
              (reinterpret_cast<address>(dst) + dstOffset), dstAgent,
              (reinterpret_cast<const_address>(src) + srcOffset), srcAgent,
              size[0], wait_events.size(), wait_events.data(), active);
          if (status != HSA_STATUS_SUCCESS) {
            gpu().Barriers().ResetCurrentSignal();
            LogPrintfError("DMA buffer failed with code %d", status);
            return false;
          }
        }
      }
    }
  }

  return true;
}

// ================================================================================================
bool DmaBlitManager::copyImageToBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire, size_t rowPitch,
                                       size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation, hence make sure GPU is done
  gpu().releaseGpuMemoryFence();

  bool result = false;

  if (setup_.disableCopyImageToBuffer_) {
    result = HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
  } else {
    Image& srcImage = static_cast<roc::Image&>(srcMemory);
    Buffer& dstBuffer = static_cast<roc::Buffer&>(dstMemory);
    address dstHost = reinterpret_cast<address>(dstBuffer.getDeviceMemory()) + dstOrigin[0];

    // Use ROCm path for a transfer.
    // Note: it doesn't support SDMA
    hsa_ext_image_region_t image_region;
    image_region.offset.x = srcOrigin[0];
    image_region.offset.y = srcOrigin[1];
    image_region.offset.z = srcOrigin[2];
    image_region.range.x = size[0];
    image_region.range.y = size[1];
    image_region.range.z = size[2];

    hsa_status_t status = hsa_ext_image_export(gpu().gpu_device(), srcImage.getHsaImageObject(),
                                               dstHost, rowPitch, slicePitch, &image_region);
    result = (status == HSA_STATUS_SUCCESS) ? true : false;

    // hsa_ext_image_export need a system scope fence
    gpu().addSystemScope();

    // Check if a HostBlit transfer is required
    if (completeOperation_ && !result) {
      result = HostBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                  entire, rowPitch, slicePitch, copyMetadata);
    }
  }

  return result;
}

// ================================================================================================
bool DmaBlitManager::copyBufferToImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin,
                                       const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire, size_t rowPitch,
                                       size_t slicePitch, amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation, hence make sure GPU is done
  gpu().releaseGpuMemoryFence();

  bool result = false;

  if (setup_.disableCopyBufferToImage_) {
    result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
  } else {
    Buffer& srcBuffer = static_cast<roc::Buffer&>(srcMemory);
    Image& dstImage = static_cast<roc::Image&>(dstMemory);

    // Use ROC path for a transfer
    // Note: it doesn't support SDMA
    address srcHost = reinterpret_cast<address>(srcBuffer.getDeviceMemory()) + srcOrigin[0];

    hsa_ext_image_region_t image_region;
    image_region.offset.x = dstOrigin[0];
    image_region.offset.y = dstOrigin[1];
    image_region.offset.z = dstOrigin[2];
    image_region.range.x = size[0];
    image_region.range.y = size[1];
    image_region.range.z = size[2];

    hsa_status_t status = hsa_ext_image_import(gpu().gpu_device(), srcHost, rowPitch, slicePitch,
                                               dstImage.getHsaImageObject(), &image_region);
    result = (status == HSA_STATUS_SUCCESS) ? true : false;

    // hsa_ext_image_import need a system scope fence
    gpu().addSystemScope();

    // Check if a HostBlit tran sfer is required
    if (completeOperation_ && !result) {
      result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                  entire, rowPitch, slicePitch, copyMetadata);
    }
  }

  return result;
}

// ================================================================================================
bool DmaBlitManager::copyImage(device::Memory& srcMemory, device::Memory& dstMemory,
                               const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                               const amd::Coord3D& size, bool entire,
                               amd::CopyMetadata copyMetadata) const {
  // HSA copy functionality with a possible async operation, hence make sure GPU is done
  gpu().releaseGpuMemoryFence();

  bool result = false;

  if (setup_.disableCopyImage_) {
    return HostBlitManager::copyImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                      copyMetadata);
  } else {
    //! @todo Add HW accelerated path
    return HostBlitManager::copyImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                      copyMetadata);
  }

  return result;
}

// ================================================================================================
bool DmaBlitManager::hsaCopy(const Memory& srcMemory, const Memory& dstMemory,
                             const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                             const amd::Coord3D& size, amd::CopyMetadata copyMetadata) const {
  address src = reinterpret_cast<address>(srcMemory.getDeviceMemory());
  address dst = reinterpret_cast<address>(dstMemory.getDeviceMemory());

  gpu().releaseGpuMemoryFence(kSkipCpuWait);

  src += srcOrigin[0];
  dst += dstOrigin[0];

  // Just call copy function for full profile
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (dev().agent_profile() == HSA_PROFILE_FULL) {
    // Stall GPU, sicne CPU copy is possible
    gpu().Barriers().WaitCurrent();
    status = hsa_memory_copy(dst, src, size[0]);
    if (status != HSA_STATUS_SUCCESS) {
      LogPrintfError("Hsa copy of data failed with code %d", status);
    }
    return (status == HSA_STATUS_SUCCESS);
  }

  hsa_agent_t srcAgent;
  hsa_agent_t dstAgent;

  if (&srcMemory.dev() == &dstMemory.dev()) {
    // Detect the agents for memory allocations
    srcAgent =
      (srcMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();
    dstAgent =
      (dstMemory.isHostMemDirectAccess()) ? dev().getCpuAgent() : dev().getBackendDevice();
  }
  else {
    srcAgent = srcMemory.dev().getBackendDevice();
    dstAgent = dstMemory.dev().getBackendDevice();
  }

  // This workaround is needed for performance to get around the slowdown
  // caused to SDMA engine powering down if its not active. Forcing agents
  // to amdgpu device causes rocr to take blit path internally.
  if (size[0] <= dev().settings().sdmaCopyThreshold_) {
    srcAgent = dstAgent = dev().getBackendDevice();
  }

  uint32_t copyMask = 0;
  uint32_t freeEngineMask = 0;
  bool kUseRegularCopyApi = 0;
  constexpr size_t kRetainCountThreshold = 8;
  bool forceSDMA = (copyMetadata.copyEnginePreference_ ==
                      amd::CopyMetadata::CopyEnginePreference::SDMA);
  HwQueueEngine engine = HwQueueEngine::Unknown;

  // Determine engine and assign a copy mask for the new versatile ROCr API
  // If engine preferred is SDMA, assign the SdmaWrite path
  if ((srcAgent.handle == dev().getCpuAgent().handle) &&
      (dstAgent.handle != dev().getCpuAgent().handle)) {
    engine = HwQueueEngine::SdmaWrite;
    copyMask = kUseRegularCopyApi ? 0 : dev().fetchSDMAMask(this, false);
    if (copyMask == 0) {
      // Track the HtoD copies and increment the count. The last used SDMA engine might be busy
      // and using it everytime can cause contention. When the count exceeds the threshold,
      // reset it so as to check the engine status and fetch the new mask.
      sdmaEngineRetainCount_ = (sdmaEngineRetainCount_ > kRetainCountThreshold)
                               ? 0 : sdmaEngineRetainCount_++;
    }
  } else if ((srcAgent.handle != dev().getCpuAgent().handle) &&
             (dstAgent.handle == dev().getCpuAgent().handle)) {
    engine = HwQueueEngine::SdmaRead;
    copyMask = kUseRegularCopyApi ? 0 : dev().fetchSDMAMask(this, true);
    if (copyMask == 0 && sdmaEngineRetainCount_ > 0) {
      // Track the DtoH copies and decrement the count.
      sdmaEngineRetainCount_--;
    }
  }

  if (engine == HwQueueEngine::Unknown && forceSDMA) {
    engine = HwQueueEngine::SdmaRead;
    copyMask = kUseRegularCopyApi ? 0 : dev().fetchSDMAMask(this, true);
  }

  // Check if host wait has to be forced
  bool forceHostWait = forceHostWaitFunc(size[0]);

  auto wait_events = gpu().Barriers().WaitingSignal(engine);
  hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp(),
                                                      forceHostWait);

  if (!kUseRegularCopyApi && engine != HwQueueEngine::Unknown) {
    if (copyMask == 0) {
      if (sdmaEngineRetainCount_) {
        // Check if there a recently used SDMA engine for the stream
        copyMask = gpu().getLastUsedSdmaEngine();
        ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Last copy mask 0x%x", copyMask);
        copyMask &= (engine == HwQueueEngine::SdmaRead ?
                    sdmaEngineReadMask_ : sdmaEngineWriteMask_);
      }
      if (copyMask == 0) {
        // Check SDMA engine status
        status = hsa_amd_memory_copy_engine_status(dstAgent, srcAgent, &freeEngineMask);
        ClPrint(amd::LOG_DEBUG, amd::LOG_COPY, "Query copy engine status %x, "
                "free_engine mask 0x%x", status, freeEngineMask);
        // Return a mask with the rightmost bit set
        copyMask = freeEngineMask - (freeEngineMask & (freeEngineMask - 1));
        gpu().setLastUsedSdmaEngine(copyMask);
      }
    }

    if (copyMask != 0 && status == HSA_STATUS_SUCCESS) {
      // Copy on the first available free engine if ROCr returns a valid mask
      hsa_amd_sdma_engine_id_t copyEngine = static_cast<hsa_amd_sdma_engine_id_t>(copyMask);

      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
              "HSA Async Copy on copy_engine=0x%x, dst=0x%zx, src=0x%zx, "
              "size=%ld, forceSDMA=%d, wait_event=0x%zx, completion_signal=0x%zx", copyEngine,
              dst, src, size[0], forceSDMA, (wait_events.size() != 0) ? wait_events[0].handle : 0,
              active.handle);

      status = hsa_amd_memory_async_copy_on_engine(dst, dstAgent, src, srcAgent,
                                                  size[0], wait_events.size(),
                                                  wait_events.data(), active, copyEngine,
                                                  forceSDMA);
    } else {
      kUseRegularCopyApi = true;
    }
  }

  if (engine == HwQueueEngine::Unknown || kUseRegularCopyApi) {
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
            "HSA Async Copy dst=0x%zx, src=0x%zx, size=%ld, wait_event=0x%zx, "
            "completion_signal=0x%zx",
            dst, src, size[0], (wait_events.size() != 0) ? wait_events[0].handle : 0,
            active.handle);

    status = hsa_amd_memory_async_copy(dst, dstAgent, src, srcAgent,
        size[0], wait_events.size(), wait_events.data(), active);
  }

  if (status == HSA_STATUS_SUCCESS) {
    gpu().addSystemScope();
  } else {
    gpu().Barriers().ResetCurrentSignal();
    LogPrintfError("HSA copy failed with code %d, falling to Blit copy", status);
  }

  return (status == HSA_STATUS_SUCCESS);
}

// ================================================================================================
bool DmaBlitManager::hsaCopyStaged(const_address hostSrc, address hostDst, size_t size,
                                   address staging, bool hostToDev) const {
  // Stall GPU, sicne CPU copy is possible
  gpu().releaseGpuMemoryFence();

  // No allocation is necessary for Full Profile
  hsa_status_t status;
  if (dev().agent_profile() == HSA_PROFILE_FULL) {
    status = hsa_memory_copy(hostDst, hostSrc, size);
    if (status != HSA_STATUS_SUCCESS) {
      LogPrintfError("Hsa copy of data failed with code %d", status);
    }
    return (status == HSA_STATUS_SUCCESS);
  }

  size_t totalSize = size;
  size_t offset = 0;

  address hsaBuffer = staging;

  // Allocate requested size of memory
  while (totalSize > 0) {
    size = std::min(totalSize, dev().settings().stagedXferSize_);

    // Copy data from Host to Device
    if (hostToDev) {
      // This workaround is needed for performance to get around the slowdown
      // caused to SDMA engine powering down if its not active. Forcing agents
      // to amdgpu device causes rocr to take blit path internally.
      const hsa_agent_t srcAgent =
          (size <= dev().settings().sdmaCopyThreshold_) ? dev().getBackendDevice() :
                                                          dev().getCpuAgent();

      HwQueueEngine engine = HwQueueEngine::Unknown;
      if (srcAgent.handle == dev().getBackendDevice().handle) {
        engine = HwQueueEngine::SdmaWrite;
      }
      gpu().Barriers().SetActiveEngine(engine);
      auto wait_events = gpu().Barriers().WaitingSignal(engine);
      hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp());

      memcpy(hsaBuffer, hostSrc + offset, size);
      status = hsa_amd_memory_async_copy(
          hostDst + offset, dev().getBackendDevice(), hsaBuffer, srcAgent, size,
          wait_events.size(), wait_events.data(), active);
      ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
          "HSA Async Copy staged H2D dst=0x%zx, src=0x%zx, size=%ld, completion_signal=0x%zx",
          hostDst + offset, hsaBuffer, size, active.handle);

      if (status != HSA_STATUS_SUCCESS) {
        gpu().Barriers().ResetCurrentSignal();
        LogPrintfError("Hsa copy from host to device failed with code %d", status);
        return false;
      }
      gpu().Barriers().WaitCurrent();
      totalSize -= size;
      offset += size;
      continue;
    }

    // This workaround is needed for performance to get around the slowdown
    // caused to SDMA engine powering down if its not active. Forcing agents
    // to amdgpu device causes rocr to take blit path internally.
    const hsa_agent_t dstAgent =
        (size <= dev().settings().sdmaCopyThreshold_) ? dev().getBackendDevice() :
                                                        dev().getCpuAgent();

    HwQueueEngine engine = HwQueueEngine::Unknown;
    if (dstAgent.handle == dev().getBackendDevice().handle) {
      engine = HwQueueEngine::SdmaRead;
    }
    gpu().Barriers().SetActiveEngine(engine);
    auto wait_events = gpu().Barriers().WaitingSignal(engine);
    hsa_signal_t active = gpu().Barriers().ActiveSignal(kInitSignalValueOne, gpu().timestamp());

    // Copy data from Device to Host
    status = hsa_amd_memory_async_copy(
        hsaBuffer, dstAgent, hostSrc + offset, dev().getBackendDevice(), size,
        wait_events.size(), wait_events.data(), active);
    ClPrint(amd::LOG_DEBUG, amd::LOG_COPY,
            "HSA Async Copy staged D2H dst=0x%zx, src=0x%zx, size=%ld, completion_signal=0x%zx",
            hsaBuffer, hostSrc + offset, size, active.handle);

    if (status == HSA_STATUS_SUCCESS) {
      gpu().Barriers().WaitCurrent();
      memcpy(hostDst + offset, hsaBuffer, size);
    } else {
      gpu().Barriers().ResetCurrentSignal();
      LogPrintfError("Hsa copy from device to host failed with code %d", status);
      return false;
    }
    totalSize -= size;
    offset += size;
  }

  gpu().addSystemScope();

  return true;
}

// ================================================================================================
KernelBlitManager::KernelBlitManager(VirtualGPU& gpu, Setup setup)
    : DmaBlitManager(gpu, setup),
      program_(nullptr),
      xferBufferSize_(0),
      lockXferOps_("Transfer Ops Lock", true) {
  for (uint i = 0; i < BlitTotal; ++i) {
    kernels_[i] = nullptr;
  }

  completeOperation_ = false;
}

KernelBlitManager::~KernelBlitManager() {
  for (uint i = 0; i < NumBlitKernels(); ++i) {
    if (nullptr != kernels_[i]) {
      kernels_[i]->release();
    }
  }

  dev().resetSDMAMask(this);

  if (nullptr != program_) {
    program_->release();
  }

  if (nullptr != context_) {
    // Release a dummy context
    context_->release();
  }
}

bool KernelBlitManager::create(amd::Device& device) {
  if (!DmaBlitManager::create(device)) {
    return false;
  }

  if (!createProgram(static_cast<Device&>(device))) {
    return false;
  }
  return true;
}

// ================================================================================================
bool KernelBlitManager::createProgram(Device& device) {
  if (device.blitProgram() == nullptr) {
    if (!device.createBlitProgram()) {
      return false;
    }
  }

  std::vector<amd::Device*> devices;
  devices.push_back(&device);

  // Save context and program for this device
  context_ = device.blitProgram()->context_;
  context_->retain();
  program_ = device.blitProgram()->program_;
  program_->retain();

  bool result = false;
  do {
    // Create kernel objects for all blits
    for (uint i = 0; i < NumBlitKernels(); ++i) {
      const amd::Symbol* symbol = program_->findSymbol(BlitName[i]);
      if (symbol == nullptr) {
        // Not all blit kernels are needed in some setup, so continue with the rest
        continue;
      }
      kernels_[i] = new amd::Kernel(*program_, *symbol, BlitName[i]);
      if (kernels_[i] == nullptr) {
        break;
      }
      // Validate blit kernels for the scratch memory usage (pre SI)
      if (!device.validateKernel(*kernels_[i], &gpu())) {
        break;
      }
    }

    result = true;
  } while (!result);

  return result;
}

// The following data structures will be used for the view creations.
// Some formats has to be converted before a kernel blit operation
struct FormatConvertion {
  uint32_t clOldType_;
  uint32_t clNewType_;
};

// The list of rejected data formats and corresponding conversion
static constexpr FormatConvertion RejectedData[] = {
    {CL_UNORM_INT8, CL_UNSIGNED_INT8},       {CL_UNORM_INT16, CL_UNSIGNED_INT16},
    {CL_SNORM_INT8, CL_UNSIGNED_INT8},       {CL_SNORM_INT16, CL_UNSIGNED_INT16},
    {CL_HALF_FLOAT, CL_UNSIGNED_INT16},      {CL_FLOAT, CL_UNSIGNED_INT32},
    {CL_SIGNED_INT8, CL_UNSIGNED_INT8},      {CL_SIGNED_INT16, CL_UNSIGNED_INT16},
    {CL_UNORM_INT_101010, CL_UNSIGNED_INT8}, {CL_SIGNED_INT32, CL_UNSIGNED_INT32}};

// The list of rejected channel's order and corresponding conversion
static constexpr FormatConvertion RejectedOrder[] = {
    {CL_A, CL_R},        {CL_RA, CL_RG},      {CL_LUMINANCE, CL_R}, {CL_INTENSITY, CL_R},
    {CL_RGB, CL_RGBA},   {CL_BGRA, CL_RGBA},  {CL_ARGB, CL_RGBA},   {CL_sRGB, CL_RGBA},
    {CL_sRGBx, CL_RGBA}, {CL_sRGBA, CL_RGBA}, {CL_sBGRA, CL_RGBA},  {CL_DEPTH, CL_R}};

const uint RejectedFormatDataTotal = sizeof(RejectedData) / sizeof(FormatConvertion);
const uint RejectedFormatChannelTotal = sizeof(RejectedOrder) / sizeof(FormatConvertion);

// ================================================================================================
bool KernelBlitManager::copyBufferToImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                          const amd::Coord3D& srcOrigin,
                                          const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                          bool entire, size_t rowPitch, size_t slicePitch,
                                          amd::CopyMetadata copyMetadata) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  amd::Image* dstImage = static_cast<amd::Image*>(dstMemory.owner());
  size_t imgRowPitch = size[0] * dstImage->getImageFormat().getElementSize();
  size_t imgSlicePitch = imgRowPitch * size[1];

  if (setup_.disableCopyBufferToImage_) {
    result = HostBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                entire, rowPitch, slicePitch, copyMetadata);
    synchronize();
    return result;
  }
  // Check if buffer is in system memory with direct access
  else if (srcMemory.isHostMemDirectAccess() &&
           (((rowPitch == 0) && (slicePitch == 0)) ||
            ((rowPitch == imgRowPitch) && ((slicePitch == 0) || (slicePitch == imgSlicePitch))))) {
    // First attempt to do this all with DMA,
    // but there are restriciton with older hardware
    if (dev().settings().imageDMA_) {
      result = DmaBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                 entire, rowPitch, slicePitch, copyMetadata);
      if (result) {
        synchronize();
        return result;
      }
    }
  }

  if (!result) {
    result = copyBufferToImageKernel(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                     rowPitch, slicePitch, copyMetadata);
  }

  synchronize();

  return result;
}

// ================================================================================================
void CalcRowSlicePitches(uint64_t* pitch, const int32_t* copySize, size_t rowPitch,
                         size_t slicePitch, const Memory& mem) {
  amd::Image* image = static_cast<amd::Image*>(mem.owner());
  uint32_t memFmtSize = image->getImageFormat().getElementSize();
  bool img1Darray = (mem.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? true : false;

  if (rowPitch == 0) {
    pitch[0] = copySize[0];
  } else {
    pitch[0] = rowPitch / memFmtSize;
  }
  if (slicePitch == 0) {
    pitch[1] = pitch[0] * (img1Darray ? 1 : copySize[1]);
  } else {
    pitch[1] = slicePitch / memFmtSize;
  }
  assert((pitch[0] <= pitch[1]) && "rowPitch must be <= slicePitch");

  if (img1Darray) {
    // For 1D array rowRitch = slicePitch
    pitch[0] = pitch[1];
  }
}

// ================================================================================================
bool KernelBlitManager::copyBufferToImageKernel(device::Memory& srcMemory,
                                                device::Memory& dstMemory,
                                                const amd::Coord3D& srcOrigin,
                                                const amd::Coord3D& dstOrigin,
                                                const amd::Coord3D& size, bool entire,
                                                size_t rowPitch, size_t slicePitch,
                                                amd::CopyMetadata copyMetadata) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  bool rejected = false;
  Memory* dstView = &gpuMem(dstMemory);
  bool result = false;
  amd::Image* dstImage = static_cast<amd::Image*>(dstMemory.owner());
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  amd::Image::Format newFormat(dstImage->getImageFormat());
  bool swapLayer =
    (dstImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10);

  // Find unsupported formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == newFormat.image_channel_data_type) {
      newFormat.image_channel_data_type = RejectedData[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // Find unsupported channel's order
  for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
    if (RejectedOrder[i].clOldType_ == newFormat.image_channel_order) {
      newFormat.image_channel_order = RejectedOrder[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // If the image format was rejected, then attempt to create a view
  if (rejected &&
      // todo ROC runtime has a problem with a view for this format
      (dstImage->getImageFormat().image_channel_data_type != CL_UNORM_INT_101010)) {
    dstView = createView(gpuMem(dstMemory), newFormat, CL_MEM_WRITE_ONLY);
    if (dstView != nullptr) {
      rejected = false;
    }
  }

  // Fall into the host path if the image format was rejected
  if (rejected) {
    return DmaBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                             entire, rowPitch, slicePitch, copyMetadata);
  }

  // Use a common blit type with three dimensions by default
  uint blitType = BlitCopyBufferToImage;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  if (dstImage->getDims() == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (dstImage->getDims() == 2) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
    // Swap the Y and Z components, apparently gfx10 HW expects
    // layer in Z
    if (swapLayer) {
      globalWorkSize[2] = globalWorkSize[1];
      globalWorkSize[1] = 1;
      localWorkSize[2] = localWorkSize[1];
      localWorkSize[1] = 1;
    }
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // Program kernels arguments for the blit operation
  cl_mem mem = as_cl<amd::Memory>(srcMemory.owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstView->owner());
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);
  uint32_t memFmtSize = dstImage->getImageFormat().getElementSize();
  uint32_t components = dstImage->getImageFormat().getNumChannels();

  // 1 element granularity for writes by default
  int32_t granularity = 1;
  if (memFmtSize == 2) {
    granularity = 2;
  } else if (memFmtSize >= 4) {
    granularity = 4;
  }
  CondLog(((srcOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
  uint64_t srcOrg[4] = {srcOrigin[0] / granularity, srcOrigin[1], srcOrigin[2], 0};
  setArgument(kernels_[blitType], 2, sizeof(srcOrg), srcOrg);

  int32_t dstOrg[4] = {(int32_t)dstOrigin[0], (int32_t)dstOrigin[1], (int32_t)dstOrigin[2], 0};
  int32_t copySize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  if (swapLayer) {
    dstOrg[2] = dstOrg[1];
    dstOrg[1] = 0;
    copySize[2] = copySize[1];
    copySize[1] = 1;
  }

  setArgument(kernels_[blitType], 3, sizeof(dstOrg), dstOrg);
  setArgument(kernels_[blitType], 4, sizeof(copySize), copySize);

  // Program memory format
  uint multiplier = memFmtSize / sizeof(uint32_t);
  multiplier = (multiplier == 0) ? 1 : multiplier;
  uint32_t format[4] = {components, memFmtSize / components, multiplier, 0};
  setArgument(kernels_[blitType], 5, sizeof(format), format);

  // Program row and slice pitches
  uint64_t pitch[4] = {0};
  CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, gpuMem(dstMemory));
  setArgument(kernels_[blitType], 6, sizeof(pitch), pitch);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyImageToBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                          const amd::Coord3D& srcOrigin,
                                          const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                          bool entire, size_t rowPitch, size_t slicePitch,
                                          amd::CopyMetadata copyMetadata) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  size_t imgRowPitch = size[0] * srcImage->getImageFormat().getElementSize();
  size_t imgSlicePitch = imgRowPitch * size[1];

  if (setup_.disableCopyImageToBuffer_) {
    result = DmaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                               entire, rowPitch, slicePitch, copyMetadata);
    synchronize();
    return result;
  }
  // Check if buffer is in system memory with direct access
  else if (dstMemory.isHostMemDirectAccess() &&
           (((rowPitch == 0) && (slicePitch == 0)) ||
            ((rowPitch == imgRowPitch) && ((slicePitch == 0) || (slicePitch == imgSlicePitch))))) {
    // First attempt to do this all with DMA,
    // but there are restriciton with older hardware
    // If the dest buffer is external physical(SDI), copy two step as
    // single step SDMA is causing corruption and the cause is under investigation
    if (dev().settings().imageDMA_) {
      result = DmaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                                 entire, rowPitch, slicePitch, copyMetadata);
      if (result) {
        synchronize();
        return result;
      }
    }
  }

  if (!result) {
    result = copyImageToBufferKernel(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                     rowPitch, slicePitch, copyMetadata);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyImageToBufferKernel(device::Memory& srcMemory,
                                                device::Memory& dstMemory,
                                                const amd::Coord3D& srcOrigin,
                                                const amd::Coord3D& dstOrigin,
                                                const amd::Coord3D& size, bool entire,
                                                size_t rowPitch, size_t slicePitch,
                                                amd::CopyMetadata copyMetadata) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  bool rejected = false;
  Memory* srcView = &gpuMem(srcMemory);
  bool result = false;
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  amd::Image::Format newFormat(srcImage->getImageFormat());
  bool swapLayer =
    (srcImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10);

  // Find unsupported formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == newFormat.image_channel_data_type) {
      newFormat.image_channel_data_type = RejectedData[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // Find unsupported channel's order
  for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
    if (RejectedOrder[i].clOldType_ == newFormat.image_channel_order) {
      newFormat.image_channel_order = RejectedOrder[i].clNewType_;
      rejected = true;
      break;
    }
  }

  // If the image format was rejected, then attempt to create a view
  if (rejected &&
      // todo ROC runtime has a problem with a view for this format
      (srcImage->getImageFormat().image_channel_data_type != CL_UNORM_INT_101010)) {
    srcView = createView(gpuMem(srcMemory), newFormat, CL_MEM_READ_ONLY);
    if (srcView != nullptr) {
      rejected = false;
    }
  }

  // Fall into the host path if the image format was rejected
  if (rejected) {
    return DmaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                             entire, rowPitch, slicePitch, copyMetadata);
  }

  uint blitType = BlitCopyImageToBuffer;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  // Find the current blit type
  if (srcImage->getDims() == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (srcImage->getDims() == 2) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
    // Swap the Y and Z components, apparently gfx10 HW expects
    // layer in Z
    if (swapLayer) {
      globalWorkSize[2] = globalWorkSize[1];
      globalWorkSize[1] = 1;
      localWorkSize[2] = localWorkSize[1];
      localWorkSize[1] = 1;
    }
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // Program kernels arguments for the blit operation
  cl_mem mem = as_cl<amd::Memory>(srcView->owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstMemory.owner());
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);

  // Update extra paramters for USHORT and UBYTE pointers.
  // Only then compiler can optimize the kernel to use
  // UAV Raw for other writes
  setArgument(kernels_[blitType], 2, sizeof(cl_mem), &mem);
  setArgument(kernels_[blitType], 3, sizeof(cl_mem), &mem);

  int32_t srcOrg[4] = {(int32_t)srcOrigin[0], (int32_t)srcOrigin[1], (int32_t)srcOrigin[2], 0};
  int32_t copySize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  if (swapLayer) {
    srcOrg[2] = srcOrg[1];
    srcOrg[1] = 0;
    copySize[2] = copySize[1];
    copySize[1] = 1;
  }

  setArgument(kernels_[blitType], 4, sizeof(srcOrg), srcOrg);
  uint32_t memFmtSize = srcImage->getImageFormat().getElementSize();
  uint32_t components = srcImage->getImageFormat().getNumChannels();

  // 1 element granularity for writes by default
  int32_t granularity = 1;
  if (memFmtSize == 2) {
    granularity = 2;
  } else if (memFmtSize >= 4) {
    granularity = 4;
  }
  CondLog(((dstOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
  uint64_t dstOrg[4] = {dstOrigin[0] / granularity, dstOrigin[1], dstOrigin[2], 0};
  setArgument(kernels_[blitType], 5, sizeof(dstOrg), dstOrg);
  setArgument(kernels_[blitType], 6, sizeof(copySize), copySize);

  // Program memory format
  uint multiplier = memFmtSize / sizeof(uint32_t);
  multiplier = (multiplier == 0) ? 1 : multiplier;
  uint32_t format[4] = {components, memFmtSize / components, multiplier, 0};
  setArgument(kernels_[blitType], 7, sizeof(format), format);

  // Program row and slice pitches
  uint64_t pitch[4] = {0};
  CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, gpuMem(srcMemory));
  setArgument(kernels_[blitType], 8, sizeof(pitch), pitch);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyImage(device::Memory& srcMemory, device::Memory& dstMemory,
                                  const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                  const amd::Coord3D& size, bool entire,
                                  amd::CopyMetadata copyMetadata) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  Memory* srcView = &gpuMem(srcMemory);
  Memory* dstView = &gpuMem(dstMemory);
  amd::Image* srcImage = static_cast<amd::Image*>(srcMemory.owner());
  amd::Image* dstImage = static_cast<amd::Image*>(dstMemory.owner());
  amd::Image::Format srcFormat(srcImage->getImageFormat());
  amd::Image::Format dstFormat(dstImage->getImageFormat());
  bool srcRejected = false, dstRejected = false;
  // Find unsupported source formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == srcFormat.image_channel_data_type) {
      srcFormat.image_channel_data_type = RejectedData[i].clNewType_;
      srcRejected = true;
      break;
    }
  }

  // Search for the rejected source channel's order only if the format was rejected
  // Note: Image blit is independent from the channel order
  if (srcRejected) {
    for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
      if (RejectedOrder[i].clOldType_ == srcFormat.image_channel_order) {
        srcFormat.image_channel_order = RejectedOrder[i].clNewType_;
        srcRejected = true;
        break;
      }
    }
  }

  // Find unsupported destination formats
  for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
    if (RejectedData[i].clOldType_ == dstFormat.image_channel_data_type) {
      dstFormat.image_channel_data_type = RejectedData[i].clNewType_;
      dstRejected = true;
      break;
    }
  }

  // Search for the rejected destionation channel's order only if the format was rejected
  // Note: Image blit is independent from the channel order
  if (dstRejected) {
    for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
      if (RejectedOrder[i].clOldType_ == dstFormat.image_channel_order) {
        dstFormat.image_channel_order = RejectedOrder[i].clNewType_;
        break;
      }
    }
  }

  if (srcFormat.image_channel_order != dstFormat.image_channel_order ||
      srcFormat.image_channel_data_type != dstFormat.image_channel_data_type) {
    //Give hint if any related test fails
    LogPrintfInfo("srcFormat(order=0x%xh, type=0x%xh) != dstFormat(order=0x%xh, type=0x%xh)",
                  srcFormat.image_channel_order, srcFormat.image_channel_data_type,
                  dstFormat.image_channel_order, dstFormat.image_channel_data_type);
  }
  // Attempt to create a view if the format was rejected
  if (srcRejected) {
    srcView = createView(gpuMem(srcMemory), srcFormat, CL_MEM_READ_ONLY);
    if (srcView != nullptr) {
      srcRejected = false;
    }
  }

  if (dstRejected) {
    dstView = createView(gpuMem(dstMemory), dstFormat, CL_MEM_WRITE_ONLY);
    if (dstView != nullptr) {
      dstRejected = false;
    }
  }

  // Fall into the host path for the copy if the image format was rejected
  if (srcRejected || dstRejected) {
    result = DmaBlitManager::copyImage(srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
                                       copyMetadata);
    synchronize();
  }

  uint blitType = BlitCopyImage;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  dim = 3;
  // Find the current blit type
  if ((srcImage->getDims() == 1) || (dstImage->getDims() == 1)) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if ((srcImage->getDims() == 2) || (dstImage->getDims() == 2)) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // The current OpenCL spec allows "copy images from a 1D image
  // array object to a 1D image array object" only.
  if ((gpuMem(srcMemory).owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
      (gpuMem(dstMemory).owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY)) {
    blitType = BlitCopyImage1DA;
  }

  // Program kernels arguments for the blit operation
  cl_mem mem = as_cl<amd::Memory>(srcView->owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstView->owner());
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);

  // Program source origin
  int32_t srcOrg[4] = {(int32_t)srcOrigin[0], (int32_t)srcOrigin[1], (int32_t)srcOrigin[2], 0};
  if ((srcImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10)) {
    srcOrg[3] = 1;
  }
  setArgument(kernels_[blitType], 2, sizeof(srcOrg), srcOrg);

  // Program destinaiton origin
  int32_t dstOrg[4] = {(int32_t)dstOrigin[0], (int32_t)dstOrigin[1], (int32_t)dstOrigin[2], 0};
  if ((dstImage->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10)) {
    dstOrg[3] = 1;
  }
  setArgument(kernels_[blitType], 3, sizeof(dstOrg), dstOrg);

  int32_t copySize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  setArgument(kernels_[blitType], 4, sizeof(copySize), copySize);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  synchronize();

  return result;
}

// ================================================================================================
void FindPinSize(size_t& pinSize, const amd::Coord3D& size, size_t& rowPitch, size_t& slicePitch,
                 const Memory& mem) {
  amd::Image* image = static_cast<amd::Image*>(mem.owner());
  pinSize = size[0] * image->getImageFormat().getElementSize();
  if ((rowPitch == 0) || (rowPitch == pinSize)) {
    rowPitch = 0;
  } else {
    pinSize = rowPitch;
  }

  // Calculate the pin size, which should be equal to the copy size
  for (uint i = 1; i < image->getDims(); ++i) {
    pinSize *= size[i];
    if (i == 1) {
      if ((slicePitch == 0) || (slicePitch == pinSize)) {
        slicePitch = 0;
      } else {
        if (mem.owner()->getType() != CL_MEM_OBJECT_IMAGE1D_ARRAY) {
          pinSize = slicePitch;
        } else {
          pinSize = slicePitch * size[i];
        }
      }
    }
  }
}

// ================================================================================================
bool KernelBlitManager::readImage(device::Memory& srcMemory, void* dstHost,
                                  const amd::Coord3D& origin, const amd::Coord3D& size,
                                  size_t rowPitch, size_t slicePitch, bool entire,
                                  amd::CopyMetadata copyMetadata) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableReadImage_ || (srcMemory.isHostMemDirectAccess() &&
                                  !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch,
                                        entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize;
    FindPinSize(pinSize, size, rowPitch, slicePitch, gpuMem(srcMemory));

    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(dstHost, pinSize, partial);

    if (amdMemory == nullptr) {
      // Force SW copy
      result =
          DmaBlitManager::readImage(srcMemory, dstHost, origin, size, rowPitch, slicePitch, entire,
                                    copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D dstOrigin(partial);

    // Get device memory for this virtual device
    Memory* dstMemory = dev().getRocMemory(amdMemory);

    // Copy image to buffer
    result = copyImageToBuffer(srcMemory, *dstMemory, origin, dstOrigin, size, entire, rowPitch,
                               slicePitch, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                   const amd::Coord3D& origin, const amd::Coord3D& size,
                                   size_t rowPitch, size_t slicePitch, bool entire,
                                   amd::CopyMetadata copyMetadata) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableWriteImage_ || dstMemory.isHostMemDirectAccess()) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                         entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize;
    FindPinSize(pinSize, size, rowPitch, slicePitch, gpuMem(dstMemory));

    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(srcHost, pinSize, partial);

    if (amdMemory == nullptr) {
      // Force SW copy
      result = DmaBlitManager::writeImage(srcHost, dstMemory, origin, size, rowPitch, slicePitch,
                                          entire, copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D srcOrigin(partial);

    // Get device memory for this virtual device
    Memory* srcMemory = dev().getRocMemory(amdMemory);

    // Copy image to buffer
    result = copyBufferToImage(*srcMemory, dstMemory, srcOrigin, origin, size, entire, rowPitch,
                               slicePitch, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::copyBufferRect(device::Memory& srcMemory, device::Memory& dstMemory,
                                       const amd::BufferRect& srcRectIn,
                                       const amd::BufferRect& dstRectIn,
                                       const amd::Coord3D& sizeIn,
                                       bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  bool rejected = false;

  // Fall into the ROC path for rejected transfers
  if (dev().info().pcie_atomics_ && (setup_.disableCopyBufferRect_ ||
      srcMemory.isHostMemDirectAccess() || dstMemory.isHostMemDirectAccess())) {
    result = DmaBlitManager::copyBufferRect(srcMemory, dstMemory, srcRectIn, dstRectIn, sizeIn, entire,
                                           copyMetadata);

    if (result) {
      synchronize();
      return result;
    }
  }

  uint blitType = BlitCopyBufferRect;
  size_t dim = 3;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  const static uint CopyRectAlignment[3] = {16, 4, 1};

  uint i;
  for (i = 0; i < sizeof(CopyRectAlignment) / sizeof(uint); i++) {
    bool aligned;
    // Check source alignments
    aligned = ((srcRectIn.rowPitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((srcRectIn.slicePitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((srcRectIn.start_ % CopyRectAlignment[i]) == 0);

    // Check destination alignments
    aligned &= ((dstRectIn.rowPitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((dstRectIn.slicePitch_ % CopyRectAlignment[i]) == 0);
    aligned &= ((dstRectIn.start_ % CopyRectAlignment[i]) == 0);

    // Check copy size alignment in the first dimension
    aligned &= ((sizeIn[0] % CopyRectAlignment[i]) == 0);

    if (aligned) {
      if (CopyRectAlignment[i] != 1) {
        blitType = BlitCopyBufferRectAligned;
      }
      break;
    }
  }

  amd::BufferRect srcRect;
  amd::BufferRect dstRect;
  amd::Coord3D size(sizeIn[0], sizeIn[1], sizeIn[2]);

  srcRect.rowPitch_ = srcRectIn.rowPitch_ / CopyRectAlignment[i];
  srcRect.slicePitch_ = srcRectIn.slicePitch_ / CopyRectAlignment[i];
  srcRect.start_ = srcRectIn.start_ / CopyRectAlignment[i];
  srcRect.end_ = srcRectIn.end_ / CopyRectAlignment[i];

  dstRect.rowPitch_ = dstRectIn.rowPitch_ / CopyRectAlignment[i];
  dstRect.slicePitch_ = dstRectIn.slicePitch_ / CopyRectAlignment[i];
  dstRect.start_ = dstRectIn.start_ / CopyRectAlignment[i];
  dstRect.end_ = dstRectIn.end_ / CopyRectAlignment[i];

  size.c[0] /= CopyRectAlignment[i];

  // Program the kernel's workload depending on the transfer dimensions
  if ((size[1] == 1) && (size[2] == 1)) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = 1;
    globalWorkSize[2] = 1;
    localWorkSize[0] = 256;
    localWorkSize[1] = 1;
    localWorkSize[2] = 1;
  } else if (size[2] == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = 1;
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
  } else {
    globalWorkSize[0] = amd::alignUp(size[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }


  // Program kernels arguments for the blit operation
  cl_mem mem = as_cl<amd::Memory>(srcMemory.owner());
  setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem);
  mem = as_cl<amd::Memory>(dstMemory.owner());
  setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem);
  uint64_t src[4] = {srcRect.rowPitch_, srcRect.slicePitch_, srcRect.start_, 0};
  setArgument(kernels_[blitType], 2, sizeof(src), src);
  uint64_t dst[4] = {dstRect.rowPitch_, dstRect.slicePitch_, dstRect.start_, 0};
  setArgument(kernels_[blitType], 3, sizeof(dst), dst);
  uint64_t copySize[4] = {size[0], size[1], size[2], CopyRectAlignment[i]};
  setArgument(kernels_[blitType], 4, sizeof(copySize), copySize);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);

  if (amd::IS_HIP) {
    // Update the command type for ROC profiler
    if (srcMemory.isHostMemDirectAccess()) {
      gpu().SetCopyCommandType(CL_COMMAND_WRITE_BUFFER_RECT);
    }
    if (dstMemory.isHostMemDirectAccess()) {
      gpu().SetCopyCommandType(CL_COMMAND_READ_BUFFER_RECT);
    }
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                   const amd::Coord3D& origin, const amd::Coord3D& size,
                                   bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableReadBuffer_ || (srcMemory.isHostMemDirectAccess() &&
      !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = size[0];
    // Check if a pinned transfer can be executed with a single pin
    if ((pinSize <= dev().settings().pinnedXferSize_) && (pinSize > MinSizeForPinnedTransfer)) {
      size_t partial;
      amd::Memory* amdMemory = pinHostMemory(dstHost, pinSize, partial);

      if (amdMemory == nullptr) {
        // Force SW copy
        result = DmaBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire,
                                            copyMetadata);
        synchronize();
        return result;
      }

      // Readjust host mem offset
      amd::Coord3D dstOrigin(partial);

      // Get device memory for this virtual device
      Memory* dstMemory = dev().getRocMemory(amdMemory);

      // Copy image to buffer
      result = copyBuffer(srcMemory, *dstMemory, origin, dstOrigin, size, entire, copyMetadata);

      // Add pinned memory for a later release
      gpu().addPinnedMem(amdMemory);
    } else {
      result = DmaBlitManager::readBuffer(srcMemory, dstHost, origin, size, entire, copyMetadata);
    }
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::readBufferRect(device::Memory& srcMemory, void* dstHost,
                                       const amd::BufferRect& bufRect,
                                       const amd::BufferRect& hostRect, const amd::Coord3D& size,
                                       bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableReadBufferRect_ ||
      (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
                                             copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = hostRect.start_ + hostRect.end_;
    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(dstHost, pinSize, partial);

    if (amdMemory == nullptr) {
      // Force SW copy
      result = DmaBlitManager::readBufferRect(srcMemory, dstHost, bufRect, hostRect, size, entire,
                                              copyMetadata);
      synchronize();
      return result;
    }

    // Readjust host mem offset
    amd::BufferRect rect;
    rect.rowPitch_ = hostRect.rowPitch_;
    rect.slicePitch_ = hostRect.slicePitch_;
    rect.start_ = hostRect.start_ + partial;
    rect.end_ = hostRect.end_;

    // Get device memory for this virtual device
    Memory* dstMemory = dev().getRocMemory(amdMemory);

    // Copy image to buffer
    result = copyBufferRect(srcMemory, *dstMemory, bufRect, rect, size, entire, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                    const amd::Coord3D& origin, const amd::Coord3D& size,
                                    bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess() ||
      gpuMem(dstMemory).IsPersistentDirectMap()) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = size[0];

    // Check if a pinned transfer can be executed with a single pin
    if ((pinSize <= dev().settings().pinnedXferSize_) && (pinSize > MinSizeForPinnedTransfer)) {
      size_t partial;
      amd::Memory* amdMemory = pinHostMemory(srcHost, pinSize, partial);

      if (amdMemory == nullptr) {
        // Force SW copy
        result = DmaBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
        synchronize();
        return result;
      }

      // Readjust destination offset
      const amd::Coord3D srcOrigin(partial);

      // Get device memory for this virtual device
      Memory* srcMemory = dev().getRocMemory(amdMemory);

      // Copy buffer rect
      result = copyBuffer(*srcMemory, dstMemory, srcOrigin, origin, size, entire, copyMetadata);

      // Add pinned memory for a later release
      gpu().addPinnedMem(amdMemory);
    } else {
      result = DmaBlitManager::writeBuffer(srcHost, dstMemory, origin, size, entire, copyMetadata);
    }
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::writeBufferRect(const void* srcHost, device::Memory& dstMemory,
                                        const amd::BufferRect& hostRect,
                                        const amd::BufferRect& bufRect, const amd::Coord3D& size,
                                        bool entire, amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host copy if memory has direct access
  if (setup_.disableWriteBufferRect_ || dstMemory.isHostMemDirectAccess() ||
      gpuMem(dstMemory).IsPersistentDirectMap()) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                              copyMetadata);
    synchronize();
    return result;
  } else {
    size_t pinSize = hostRect.start_ + hostRect.end_;
    size_t partial;
    amd::Memory* amdMemory = pinHostMemory(srcHost, pinSize, partial);

    if (amdMemory == nullptr) {
      // Force DMA copy with staging
      result = DmaBlitManager::writeBufferRect(srcHost, dstMemory, hostRect, bufRect, size, entire,
                                               copyMetadata);
      synchronize();
      return result;
    }

    // Readjust destination offset
    const amd::Coord3D srcOrigin(partial);

    // Get device memory for this virtual device
    Memory* srcMemory = dev().getRocMemory(amdMemory);

    // Readjust host mem offset
    amd::BufferRect rect;
    rect.rowPitch_ = hostRect.rowPitch_;
    rect.slicePitch_ = hostRect.slicePitch_;
    rect.start_ = hostRect.start_ + partial;
    rect.end_ = hostRect.end_;

    // Copy buffer rect
    result = copyBufferRect(*srcMemory, dstMemory, rect, bufRect, size, entire, copyMetadata);

    // Add pinned memory for a later release
    gpu().addPinnedMem(amdMemory);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::fillBuffer(device::Memory& memory, const void* pattern, size_t patternSize,
                                   const amd::Coord3D& surface, const amd::Coord3D& origin,
                                   const amd::Coord3D& size, bool entire, bool forceBlit) const {

  guarantee(size[0] > 0 && size[1] > 0 && size[2] > 0, "Dimension cannot be 0");

  if (size[1] == 1 && size[2] == 1) {
    return fillBuffer1D(memory, pattern, patternSize, surface, origin, size, entire, forceBlit);
  } else if (size[2] == 1) {
    return fillBuffer2D(memory, pattern, patternSize, surface, origin, size, entire, forceBlit);
  } else {
    bool ret_val = true;
    amd::Coord3D my_origin(origin);
    amd::Coord3D my_region{surface[1], surface[2], size[2]};
    amd::BufferRect rect;
    rect.create(static_cast<size_t*>(my_origin), static_cast<size_t*>(my_region), surface[0], 0);
    for (size_t slice = 0; slice < size[2]; ++slice) {
      const size_t row_offset = rect.offset(0, 0, slice);
      amd::Coord3D new_origin(row_offset, origin[1], origin[2]);
      ret_val |= fillBuffer2D(memory, pattern, patternSize, surface, new_origin, size, entire,
                              forceBlit);
    }
    return ret_val;
  }
}

// ================================================================================================
bool KernelBlitManager::fillBuffer1D(device::Memory& memory, const void* pattern,
                                     size_t patternSize, const amd::Coord3D& surface,
                                     const amd::Coord3D& origin, const amd::Coord3D& size,
                                     bool entire, bool forceBlit) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;

  // Use host fill if memory has direct access
  if (setup_.disableFillBuffer_ || (!forceBlit && memory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::fillBuffer(memory, pattern, patternSize, size, origin, size, entire);
    synchronize();
    return result;
  } else {
    // Pack the fill buffer info, that handles unaligned memories.
    std::vector<FillBufferInfo> packed_vector{};
    FillBufferInfo::PackInfo(memory, size[0], origin[0], pattern, patternSize, packed_vector);

    size_t overall_offset = origin[0];
    for (auto& packed_obj: packed_vector) {
      constexpr uint32_t kFillType = FillBufferAligned;
      uint32_t kpattern_size = (packed_obj.pattern_expanded_) ?
                                HostBlitManager::FillBufferInfo::kExtendedSize : patternSize;
      size_t kfill_size = packed_obj.fill_size_ / kpattern_size;
      size_t koffset = overall_offset;
      overall_offset += packed_obj.fill_size_;

      size_t globalWorkOffset[3] = {0, 0, 0};
      uint32_t alignment = (kpattern_size & 0xf) == 0 ? 2 * sizeof(uint64_t) :
                           (kpattern_size & 0x7) == 0 ? sizeof(uint64_t) :
                           (kpattern_size & 0x3) == 0 ? sizeof(uint32_t) :
                           (kpattern_size & 0x1) == 0 ? sizeof(uint16_t) : sizeof(uint8_t);
      // Program kernels arguments for the fill operation
      cl_mem mem = as_cl<amd::Memory>(memory.owner());
      setArgument(kernels_[kFillType], 0, sizeof(cl_mem), &mem, koffset);
      const size_t localWorkSize = 256;
      size_t globalWorkSize =
          std::min(dev().settings().limit_blit_wg_ * localWorkSize, kfill_size);
      globalWorkSize = amd::alignUp(globalWorkSize, localWorkSize);

      auto constBuf = gpu().allocKernArg(kCBSize, kCBAlignment);

      // If pattern has been expanded, use the expanded pattern, otherwise use the default pattern.
      if (packed_obj.pattern_expanded_) {
        memcpy(constBuf, &packed_obj.expanded_pattern_, kpattern_size);
      } else {
        memcpy(constBuf, pattern, kpattern_size);
      }
      constexpr bool kDirectVa = true;
      setArgument(kernels_[kFillType], 1, sizeof(cl_mem), constBuf, 0, nullptr, kDirectVa);

      // Adjust the pattern size in the copy type size
      kpattern_size /= alignment;
      setArgument(kernels_[kFillType], 2, sizeof(uint32_t), &kpattern_size);
      setArgument(kernels_[kFillType], 3, sizeof(alignment), &alignment);

      // Calculate max id
      kfill_size = memory.virtualAddress() + koffset + kfill_size * kpattern_size * alignment;
      setArgument(kernels_[kFillType], 4, sizeof(kfill_size), &kfill_size);
      uint32_t next_chunk = globalWorkSize * kpattern_size;
      setArgument(kernels_[kFillType], 5, sizeof(uint32_t), &next_chunk);

      // Create ND range object for the kernel's execution
      amd::NDRangeContainer ndrange(1, globalWorkOffset, &globalWorkSize, &localWorkSize);

      // Execute the blit
      address parameters = captureArguments(kernels_[kFillType]);
      result = gpu().submitKernelInternal(ndrange, *kernels_[kFillType], parameters, nullptr);
      releaseArguments(parameters);
    }
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::fillBuffer2D(device::Memory& memory, const void* pattern,
                                     size_t patternSize, const amd::Coord3D& surface,
                                     const amd::Coord3D& origin, const amd::Coord3D& size,
                                     bool entire, bool forceBlit) const {

  amd::ScopedLock k(lockXferOps_);
  bool result = false;

    // Use host fill if memory has direct access
  if (setup_.disableFillBuffer_ || (!forceBlit && memory.isHostMemDirectAccess())) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::fillBuffer(memory, pattern, patternSize, size, origin, size, entire);
    synchronize();
    return result;
  } else {
    uint fillType = FillBufferAligned2D;
    uint64_t fillSizeX = (size[0]/patternSize) == 0 ? 1 : (size[0]/patternSize);
    uint64_t fillSizeY = size[1];

    size_t globalWorkOffset[3] = {0, 0, 0};
    size_t globalWorkSize[3] = {amd::alignUp(fillSizeX, 16),
                                amd::alignUp(fillSizeY, 16), 1};
    size_t localWorkSize [3] = {16, 16, 1};

    uint32_t alignment = (patternSize & 0x7) == 0 ?
                          sizeof(uint64_t) :
                          (patternSize & 0x3) == 0 ?
                          sizeof(uint32_t) :
                          (patternSize & 0x1) == 0 ?
                          sizeof(uint16_t) : sizeof(uint8_t);

    cl_mem mem = as_cl<amd::Memory>(memory.owner());
     if (alignment == sizeof(uint64_t)) {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), &mem);
    } else if (alignment == sizeof(uint32_t)) {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), &mem);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), nullptr);
    } else if (alignment == sizeof(uint16_t)) {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), &mem);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), nullptr);
    } else {
      setArgument(kernels_[fillType], 0, sizeof(cl_mem), &mem);
      setArgument(kernels_[fillType], 1, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 2, sizeof(cl_mem), nullptr);
      setArgument(kernels_[fillType], 3, sizeof(cl_mem), nullptr);
    }

    // Get constant buffer to allow multipel fills
    auto constBuf = gpu().allocKernArg(kCBSize, kCBAlignment);
    memcpy(constBuf, pattern, patternSize);

    constexpr bool kDirectVa = true;
    setArgument(kernels_[fillType], 4, sizeof(cl_mem), constBuf, 0, nullptr, kDirectVa);

    uint64_t mem_origin = static_cast<uint64_t>(origin[0]);
    uint64_t width = static_cast<uint64_t>(size[0]);
    uint64_t height = static_cast<uint64_t>(size[1]);
    uint64_t pitch = static_cast<uint64_t>(surface[0]);

    patternSize/= alignment;
    mem_origin /= alignment;
    pitch /= alignment;

    setArgument(kernels_[fillType], 5, sizeof(uint32_t), &patternSize);
    setArgument(kernels_[fillType], 6, sizeof(mem_origin), &mem_origin);
    setArgument(kernels_[fillType], 7, sizeof(width), &width);
    setArgument(kernels_[fillType], 8, sizeof(height), &height);
    setArgument(kernels_[fillType], 9, sizeof(pitch), &pitch);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(2, globalWorkOffset, globalWorkSize, localWorkSize);

    // Execute the blit
    address parameters = captureArguments(kernels_[fillType]);
    result = gpu().submitKernelInternal(ndrange, *kernels_[fillType], parameters, nullptr);
    releaseArguments(parameters);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::fillBuffer3D(device::Memory& memory, const void* pattern,
                                     size_t patternSize, const amd::Coord3D& surface,
                                     const amd::Coord3D& origin, const amd::Coord3D& size,
                                     bool entire, bool forceBlit) const {
  ShouldNotReachHere();
  return false;
}
// ================================================================================================
bool KernelBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                   const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                   const amd::Coord3D& sizeIn, bool entire,
                                   amd::CopyMetadata copyMetadata) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  bool p2p = false;
  uint32_t blit_wg_ = dev().settings().limit_blit_wg_;

  if (&gpuMem(srcMemory).dev() != &gpuMem(dstMemory).dev()) {
    if (sizeIn[0] > dev().settings().sdma_p2p_threshold_) {
      p2p = true;
    } else {
      constexpr uint32_t kLimitWgForKernelP2p = 16;
      blit_wg_ = kLimitWgForKernelP2p;
    }
  }

  bool asan = false;
  bool ipcShared = srcMemory.owner()->ipcShared() || dstMemory.owner()->ipcShared();
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  asan = true;
#endif
#endif

  bool useShaderCopyPath = setup_.disableHwlCopyBuffer_ ||
                           (!srcMemory.isHostMemDirectAccess() &&
                            !dstMemory.isHostMemDirectAccess() &&
                            !(p2p || asan) && !ipcShared &&
                            !(copyMetadata.copyEnginePreference_ ==
                              amd::CopyMetadata::CopyEnginePreference::SDMA));

  if (!useShaderCopyPath) {
    if (amd::IS_HIP) {
      // Update the command type for ROC profiler
      if (srcMemory.isHostMemDirectAccess()) {
        gpu().SetCopyCommandType(CL_COMMAND_WRITE_BUFFER);
      }
      if (dstMemory.isHostMemDirectAccess()) {
        gpu().SetCopyCommandType(CL_COMMAND_READ_BUFFER);
      }
    }
    result = DmaBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, sizeIn, entire,
                                        copyMetadata);
  }

  if (!result) {
    constexpr uint32_t kBlitType = BlitCopyBuffer;
    constexpr uint32_t kMaxAlignment = 2 * sizeof(uint64_t);
    amd::Coord3D size(sizeIn[0]);

    // Check alignments for source and destination
    bool aligned = ((srcOrigin[0] % kMaxAlignment) == 0) && ((dstOrigin[0] % kMaxAlignment) == 0);
    uint32_t aligned_size = (aligned) ? kMaxAlignment : sizeof(uint32_t);

    // Setup copy size accordingly to the alignment
    uint32_t remainder = size[0] % aligned_size;
    size.c[0] /= aligned_size;
    size.c[0] += (remainder != 0) ? 1 : 0;

    // Program the dispatch dimensions
    const size_t localWorkSize = (aligned) ? 512 : 1024;
    size_t globalWorkSize = std::min(blit_wg_ * localWorkSize, size[0]);
    globalWorkSize = amd::alignUp(globalWorkSize, localWorkSize);

    // Program kernels arguments for the blit operation
    cl_mem mem = as_cl<amd::Memory>(srcMemory.owner());
    // Program source origin
    uint64_t srcOffset = srcOrigin[0];
    setArgument(kernels_[kBlitType], 0, sizeof(cl_mem), &mem, srcOffset, &srcMemory);
    mem = as_cl<amd::Memory>(dstMemory.owner());
    // Program destinaiton origin
    uint64_t dstOffset = dstOrigin[0];
    setArgument(kernels_[kBlitType], 1, sizeof(cl_mem), &mem, dstOffset, &dstMemory);

    uint64_t copySize = sizeIn[0];
    setArgument(kernels_[kBlitType], 2, sizeof(copySize), &copySize);

    setArgument(kernels_[kBlitType], 3, sizeof(remainder), &remainder);
    setArgument(kernels_[kBlitType], 4, sizeof(aligned_size), &aligned_size);

    // End pointer is the aligned copy size and destination offset
    uint64_t end_ptr = dstMemory.virtualAddress() + dstOffset + sizeIn[0] - remainder;

    setArgument(kernels_[kBlitType], 5, sizeof(end_ptr), &end_ptr);

    uint32_t next_chunk = globalWorkSize;
    setArgument(kernels_[kBlitType], 6, sizeof(next_chunk), &next_chunk);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(1, nullptr, &globalWorkSize, &localWorkSize);

    // Execute the blit
    address parameters = captureArguments(kernels_[kBlitType]);
    result = gpu().submitKernelInternal(ndrange, *kernels_[kBlitType], parameters, nullptr);
    releaseArguments(parameters);
  }

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::fillImage(device::Memory& memory, const void* pattern,
                                  const amd::Coord3D& origin, const amd::Coord3D& size,
                                  bool entire) const {

  guarantee((dev().info().imageSupport_ != false), "Image not supported on this device");

  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  constexpr size_t kFillImageThreshold = 256 * 256;

  // Use host fill if memory has direct access and image is small
  if (setup_.disableFillImage_ ||
      (gpuMem(memory).isHostMemDirectAccess() &&
      (size.c[0] * size.c[1] * size.c[2]) <= kFillImageThreshold)) {
    // Stall GPU before CPU access
    gpu().releaseGpuMemoryFence();
    result = HostBlitManager::fillImage(memory, pattern, origin, size, entire);
    synchronize();
    return result;
  }

  uint fillType;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];
  Memory* memView = &gpuMem(memory);
  amd::Image* image = static_cast<amd::Image*>(memory.owner());
  amd::Image::Format newFormat(image->getImageFormat());
  bool swapLayer =
    (image->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) && (dev().isa().versionMajor() >= 10);

  // Program the kernels workload depending on the fill dimensions
  fillType = FillImage;
  dim = 3;

  void* newpattern = const_cast<void*>(pattern);
  uint32_t iFillColor[4];

  bool rejected = false;

  // For depth, we need to create a view
  if (newFormat.image_channel_order == CL_sRGBA) {
    // Find unsupported data type
    for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
      if (RejectedData[i].clOldType_ == newFormat.image_channel_data_type) {
        newFormat.image_channel_data_type = RejectedData[i].clNewType_;
        rejected = true;
        break;
      }
    }

    if (newFormat.image_channel_order == CL_sRGBA) {
      // Converting a linear RGB floating-point color value to a 8-bit unsigned integer sRGB value
      // because hw is not support write_imagef for sRGB.
      float* fColor = static_cast<float*>(newpattern);
      iFillColor[0] = sRGBmap(fColor[0]);
      iFillColor[1] = sRGBmap(fColor[1]);
      iFillColor[2] = sRGBmap(fColor[2]);
      iFillColor[3] = (uint32_t)(fColor[3] * 255.0f);
      newpattern = static_cast<void*>(&iFillColor[0]);
      for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
        if (RejectedOrder[i].clOldType_ == newFormat.image_channel_order) {
          newFormat.image_channel_order = RejectedOrder[i].clNewType_;
          rejected = true;
          break;
        }
      }
    }
  }
  // If the image format was rejected, then attempt to create a view
  if (rejected) {
    memView = createView(gpuMem(memory), newFormat, CL_MEM_WRITE_ONLY);
    if (memView != nullptr) {
      rejected = false;
    }
  }

  if (rejected) {
    return DmaBlitManager::fillImage(memory, pattern, origin, size, entire);
  }

  // Perform workload split to allow multiple operations in a single thread
  globalWorkSize[0] = (size[0] + TransferSplitSize - 1) / TransferSplitSize;
  // Find the current blit type
  if (image->getDims() == 1) {
    globalWorkSize[0] = amd::alignUp(globalWorkSize[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (image->getDims() == 2) {
    globalWorkSize[0] = amd::alignUp(globalWorkSize[0], 16);
    globalWorkSize[1] = amd::alignUp(size[1], 16);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = localWorkSize[1] = 16;
    localWorkSize[2] = 1;
    // Swap the Y and Z components, apparently gfx10 HW expects
    // layer in Z
    if (swapLayer) {
      globalWorkSize[2] = globalWorkSize[1];
      globalWorkSize[1] = 1;
      localWorkSize[2] = localWorkSize[1];
      localWorkSize[1] = 1;
    }
  } else {
    globalWorkSize[0] = amd::alignUp(globalWorkSize[0], 8);
    globalWorkSize[1] = amd::alignUp(size[1], 8);
    globalWorkSize[2] = amd::alignUp(size[2], 4);
    localWorkSize[0] = localWorkSize[1] = 8;
    localWorkSize[2] = 4;
  }

  // Program kernels arguments for the blit operation
  cl_mem mem = as_cl<amd::Memory>(memView->owner());
  setArgument(kernels_[fillType], 0, sizeof(cl_mem), &mem);
  setArgument(kernels_[fillType], 1, sizeof(float[4]), newpattern);
  setArgument(kernels_[fillType], 2, sizeof(int32_t[4]), newpattern);
  setArgument(kernels_[fillType], 3, sizeof(uint32_t[4]), newpattern);

  int32_t fillOrigin[4] = {(int32_t)origin[0], (int32_t)origin[1], (int32_t)origin[2], 0};
  int32_t fillSize[4] = {(int32_t)size[0], (int32_t)size[1], (int32_t)size[2], 0};
  if (swapLayer) {
    fillOrigin[2] = fillOrigin[1];
    fillOrigin[1] = 0;
    fillSize[2] = fillSize[1];
    fillSize[1] = 1;
  }
  setArgument(kernels_[fillType], 4, sizeof(fillOrigin), fillOrigin);
  setArgument(kernels_[fillType], 5, sizeof(fillSize), fillSize);

  // Find the type of image
  uint32_t type = 0;
  switch (newFormat.image_channel_data_type) {
    case CL_SNORM_INT8:
    case CL_SNORM_INT16:
    case CL_UNORM_INT8:
    case CL_UNORM_INT16:
    case CL_UNORM_SHORT_565:
    case CL_UNORM_SHORT_555:
    case CL_UNORM_INT_101010:
    case CL_HALF_FLOAT:
    case CL_FLOAT:
      type = 0;
      break;
    case CL_SIGNED_INT8:
    case CL_SIGNED_INT16:
    case CL_SIGNED_INT32:
      type = 1;
      break;
    case CL_UNSIGNED_INT8:
    case CL_UNSIGNED_INT16:
    case CL_UNSIGNED_INT32:
      type = 2;
      break;
  }
  setArgument(kernels_[fillType], 6, sizeof(type), &type);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[fillType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[fillType], parameters, nullptr);
  releaseArguments(parameters);

  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::streamOpsWrite(device::Memory& memory, uint64_t value,
                                       size_t offset, size_t sizeBytes) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  uint blitType = StreamOpsWrite;
  size_t dim = 1;
  size_t globalWorkOffset[1] = { 0 };
  size_t globalWorkSize[1] = { 1 };
  size_t localWorkSize[1] = { 1 };
  // Program kernels arguments for the write operation
  cl_mem mem = as_cl<amd::Memory>(memory.owner());
  bool is32BitWrite = (sizeBytes == sizeof(uint32_t)) ? true : false;
  // Program kernels arguments for the write operation
  if (is32BitWrite) {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 2, sizeof(uint32_t), &value);
  } else {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 2, sizeof(uint64_t), &value);
  }
  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);
  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);
  synchronize();
  return result;
}

// ================================================================================================
bool KernelBlitManager::streamOpsWait(device::Memory& memory, uint64_t value, size_t offset,
                                      size_t sizeBytes, uint64_t flags, uint64_t mask) const {
  amd::ScopedLock k(lockXferOps_);
  bool result = false;
  uint blitType = StreamOpsWait;
  size_t dim = 1;

  size_t globalWorkOffset[1] = { 0 };
  size_t globalWorkSize[1] = { 1 };
  size_t localWorkSize[1] = { 1 };

  // Program kernels arguments for the wait operation
  cl_mem mem = as_cl<amd::Memory>(memory.owner());
  bool is32BitWait = (sizeBytes == sizeof(uint32_t)) ? true : false;
  // Program kernels arguments for the wait operation
  if (is32BitWait) {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 2, sizeof(uint32_t), &value);
    setArgument(kernels_[blitType], 3, sizeof(uint32_t), &flags);
    setArgument(kernels_[blitType], 4, sizeof(uint32_t), &mask);
  } else {
    setArgument(kernels_[blitType], 0, sizeof(cl_mem), nullptr);
    setArgument(kernels_[blitType], 1, sizeof(cl_mem), &mem, offset);
    setArgument(kernels_[blitType], 2, sizeof(uint64_t), &value);
    setArgument(kernels_[blitType], 3, sizeof(uint64_t), &flags);
    setArgument(kernels_[blitType], 4, sizeof(uint64_t), &mask);
  }

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);
  synchronize();

  return result;
}

// ================================================================================================
bool KernelBlitManager::initHeap(device::Memory* heap_to_initialize, device::Memory* initial_blocks,
                                 uint heap_size, uint number_of_initial_blocks) const {
  bool result;
  // Clear memory to 0 for device library logic and set
  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {256};
  size_t localWorkSize[1] = {256};

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(1, globalWorkOffset, globalWorkSize, localWorkSize);
  uint blitType = InitHeap;
  uint64_t management_heap_va = heap_to_initialize->virtualAddress();
  uint64_t initial_heap_va = 0;
  if (initial_blocks != nullptr) {
    initial_heap_va = initial_blocks->virtualAddress();
  }
  setArgument(kernels_[blitType], 0, sizeof(cl_ulong), &management_heap_va);
  setArgument(kernels_[blitType], 1, sizeof(cl_ulong), &initial_heap_va);
  setArgument(kernels_[blitType], 2, sizeof(uint), &heap_size);
  setArgument(kernels_[blitType], 3, sizeof(uint), &number_of_initial_blocks);
  address parameters = captureArguments(kernels_[blitType]);
  result = gpu().submitKernelInternal(ndrange, *kernels_[blitType], parameters, nullptr);
  releaseArguments(parameters);
  synchronize();

  return result;
}

// ================================================================================================

amd::Memory* DmaBlitManager::pinHostMemory(const void* hostMem, size_t pinSize,
                                           size_t& partial) const {
  size_t pinAllocSize;
  const static bool SysMem = true;
  amd::Memory* amdMemory;

  // Align offset to 4K boundary
  char* tmpHost = const_cast<char*>(
      amd::alignDown(reinterpret_cast<const char*>(hostMem), PinnedMemoryAlignment));

  // Find the partial size for unaligned copy
  partial = reinterpret_cast<const char*>(hostMem) - tmpHost;

  // Recalculate pin memory size
  pinAllocSize = amd::alignUp(pinSize + partial, PinnedMemoryAlignment);

  amdMemory = gpu().findPinnedMem(tmpHost, pinAllocSize);

  if (nullptr != amdMemory) {
    return amdMemory;
  }

  amdMemory = new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, pinAllocSize);
  amdMemory->setVirtualDevice(&gpu());
  if ((amdMemory != nullptr) && !amdMemory->create(tmpHost, SysMem)) {
    DevLogPrintfError("Buffer create failed, Buffer: 0x%x \n", amdMemory);
    amdMemory->release();
    return nullptr;
  }

  // Get device memory for this virtual device
  // @note: This will force real memory pinning
  Memory* srcMemory = dev().getRocMemory(amdMemory);

  if (srcMemory == nullptr) {
    // Release all pinned memory and attempt pinning again
    gpu().releasePinnedMem();
    srcMemory = dev().getRocMemory(amdMemory);
    if (srcMemory == nullptr) {
      // Release memory
      amdMemory->release();
      amdMemory = nullptr;
    }
  }

  return amdMemory;
}

bool DmaBlitManager::forceHostWaitFunc(size_t copy_size) const {
  // 10us wait is true for all other targets.
  bool forceHostWait = true;
  // Based on the profiled results, do not wait for copy size > 24 KB.
  static constexpr size_t kGfx90aCopyThreshold = 24;

  if ((dev().isa().versionMajor() == 9 && dev().isa().versionMinor() == 0
       && dev().isa().versionStepping() == 10) && (copy_size >= kGfx90aCopyThreshold * Ki)) {
    // Check if this is gfx90a and restrict small copy to 24K.
    forceHostWait = false;
  } else if ((dev().isa().versionMajor() == 9) && (dev().isa().versionMinor() == 4)
              && (dev().isa().versionStepping() == 0 || dev().isa().versionStepping() == 1
                  || dev().isa().versionStepping() == 2)) {
    // for gfx940, gfx941, gfx942, dependency signal resolution is fast, so no Host wait at all.
    forceHostWait = false;
  }

  return forceHostWait;
}

Memory* KernelBlitManager::createView(const Memory& parent, cl_image_format format,
                                      cl_mem_flags flags) const {
  assert((parent.owner()->asBuffer() == nullptr) && "View supports images only");
  amd::Image* parentImage = static_cast<amd::Image*>(parent.owner());
  auto parent_dev_image = static_cast<Image*>(parentImage->getDeviceMemory(dev()));
  amd::Image* image = parent_dev_image->FindView(format);
  if (image == nullptr) {
    image = parentImage->createView(parent.owner()->getContext(), format, &gpu(), 0, flags,
                                    false, true);
    if (image == nullptr) {
      LogError("[OCL] Fail to allocate view of image object");
      return nullptr;
    }
    if (!parent_dev_image->AddView(image)) {
      // Another thread already added a view
      image->release();
      image = parent_dev_image->FindView(format);
    }
  }
  auto dev_image = static_cast<Image*>(image->getDeviceMemory(dev()));
  return dev_image;
}

address KernelBlitManager::captureArguments(const amd::Kernel* kernel) const {
  return kernel->parameters().values();
}

void KernelBlitManager::releaseArguments(address args) const {
}

// ================================================================================================
bool KernelBlitManager::runScheduler(uint64_t vqVM, amd::Memory* schedulerParam,
                                     hsa_queue_t* schedulerQueue,
                                     hsa_signal_t& schedulerSignal,
                                     uint threads) {
  size_t globalWorkOffset[1] = {0};
  size_t globalWorkSize[1] = {threads};
  size_t localWorkSize[1] = {1};

  amd::NDRangeContainer ndrange(1, globalWorkOffset, globalWorkSize, localWorkSize);

  device::Kernel* devKernel = const_cast<device::Kernel*>(kernels_[Scheduler]->getDeviceKernel(dev()));
  Kernel& gpuKernel = static_cast<Kernel&>(*devKernel);

  SchedulerParam* sp = reinterpret_cast<SchedulerParam*>(schedulerParam->getHostMem());
  memset(sp, 0, sizeof(SchedulerParam));

  Memory* schedulerMem = dev().getRocMemory(schedulerParam);
  sp->kernarg_address = reinterpret_cast<uint64_t>(schedulerMem->getDeviceMemory());
  sp->thread_counter = 0;
  sp->child_queue = reinterpret_cast<uint64_t>(schedulerQueue);
  sp->complete_signal = schedulerSignal;

  hsa_signal_store_relaxed(schedulerSignal, kInitSignalValueOne);


  sp->vqueue_header = vqVM;

  sp->parentAQL = sp->kernarg_address + sizeof(SchedulerParam);

  if (dev().info().maxEngineClockFrequency_ > 0) {
    sp->eng_clk = (1000 * 1024) / dev().info().maxEngineClockFrequency_;
  }

  // Use a device side global atomics to workaround the reliance of PCIe 3 atomics
  sp->write_index = hsa_queue_load_write_index_relaxed(schedulerQueue);

  cl_mem mem = as_cl<amd::Memory>(schedulerParam);
  setArgument(kernels_[Scheduler], 0, sizeof(cl_mem), &mem);

  address parameters = captureArguments(kernels_[Scheduler]);

  if (!gpu().submitKernelInternal(ndrange, *kernels_[Scheduler],
                                  parameters, nullptr, 0, nullptr, &sp->scheduler_aql)) {
    return false;
  }
  releaseArguments(parameters);

  if (!WaitForSignal(schedulerSignal)) {
    LogWarning("Failed schedulerSignal wait");
    return false;
  }

  return true;
}

// ================================================================================================
bool KernelBlitManager::RunGwsInit(
  uint32_t value) const {
  amd::ScopedLock k(lockXferOps_);

  if (dev().settings().gwsInitSupported_ == false) {
    LogError("GWS Init is not supported on this target");
    return false;
  }

  size_t globalWorkOffset[1] = { 0 };
  size_t globalWorkSize[1] = { 1 };
  size_t localWorkSize[1] = { 1 };

  // Program kernels arguments
  setArgument(kernels_[GwsInit], 0, sizeof(uint32_t), &value);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(1, globalWorkOffset, globalWorkSize, localWorkSize);

  // Execute the blit
  address parameters = captureArguments(kernels_[GwsInit]);

  bool result = gpu().submitKernelInternal(ndrange, *kernels_[GwsInit], parameters, nullptr);

  releaseArguments(parameters);

  return result;
}

}  // namespace amd::roc
