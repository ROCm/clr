//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#include "platform/commandqueue.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "utils/debug.hpp"

namespace roc {


void
FindPinSize(
    size_t& pinSize, const amd::Coord3D& size,
    size_t& rowPitch, size_t& slicePitch, const Image& image)
{
    size_t elementSize = image.owner()->asImage()->getImageFormat().getElementSize();
    pinSize = size[0] * elementSize;
    if ((rowPitch == 0) || (rowPitch == pinSize)) {
        rowPitch = 0;
    }
    else {
        pinSize = rowPitch;
    }

    // Calculate the pin size, which should be equal to the copy size
    for (uint i = 1; i < 3; ++i) {
        pinSize *= size[i];
        if (i == 1) {
            if ((slicePitch == 0) || (slicePitch == pinSize)) {
                slicePitch = 0;
            }
            else {
                if (image.getHsaImageDescriptor().geometry != HSA_EXT_IMAGE_GEOMETRY_1DA) {
                    pinSize = slicePitch;
                }
                else {
                    pinSize = slicePitch * size[i];
                }
            }
        }
    }
}

HsaBlitManager::HsaBlitManager(device::VirtualDevice& vDev, Setup setup)
    : HostBlitManager(vDev, setup),
    roc_device_(reinterpret_cast<const roc::Device &>(dev_)) {
  completion_signal_.handle = 0;
}

bool HsaBlitManager::hsaCopy(const void *hostSrc, void *hostDst,
                             uint32_t size, bool hostToDev) const {

  // No allocation is necessary for Full Profile 
  hsa_status_t status;
  if (roc_device_.agent_profile() == HSA_PROFILE_FULL) {
    status = hsa_memory_copy(hostDst, hostSrc, size);
    if (status != HSA_STATUS_SUCCESS) {
      LogPrintfError("Hsa copy of data failed with code %d", status);
    }
    return (status == HSA_STATUS_SUCCESS);
  }

  // Allocate requested size of memory
  size_t align = 0x04;
  bool atomics = false;
  void *hsaBuffer = NULL;
  hsaBuffer = roc_device_.hostAlloc(size, align, false);
  if (hsaBuffer == NULL) {
    LogError("Hsa buffer allocation failed with code");
    return false;
  }

  const hsa_signal_value_t kInitVal = 1;
  hsa_signal_store_relaxed(completion_signal_, kInitVal);

  // Copy data from Host to Device
  if (hostToDev) {
    memcpy(hsaBuffer, hostSrc, size);
    status = hsa_amd_memory_async_copy(
        hostDst, roc_device_.getBackendDevice(), hsaBuffer,
        roc_device_.getCpuAgent(), size, 0, NULL, completion_signal_);
    if (status == HSA_STATUS_SUCCESS) {
      hsa_signal_value_t val =
        hsa_signal_wait_acquire(completion_signal_, HSA_SIGNAL_CONDITION_EQ, 0,
        uint64_t(-1), HSA_WAIT_STATE_ACTIVE);

      if (val != (kInitVal - 1)) {
        LogError("Async copy failed");
        status = HSA_STATUS_ERROR;
      }
    }
    else {
      LogPrintfError("Hsa copy from host to device failed with code %d", status);
    }

    roc_device_.hostFree(hsaBuffer, size);
    return (status == HSA_STATUS_SUCCESS);
  }

  // Copy data from Device to Host
  status = hsa_amd_memory_async_copy(hsaBuffer, roc_device_.getCpuAgent(),
                                     hostSrc, roc_device_.getBackendDevice(),
                                     size, 0, NULL, completion_signal_);
  if (status == HSA_STATUS_SUCCESS) {
    hsa_signal_value_t val = hsa_signal_wait_acquire(
      completion_signal_, HSA_SIGNAL_CONDITION_EQ, 0, uint64_t(-1),
      HSA_WAIT_STATE_ACTIVE);

    if (val != (kInitVal - 1)) {
      LogError("Async copy failed");
      status = HSA_STATUS_ERROR;
    }

    if (status == HSA_STATUS_SUCCESS) {
      memcpy(hostDst, hsaBuffer, size);
    }
  } else {
    LogPrintfError("Hsa copy from device to host failed with code %d", status);
  }
  
  roc_device_.hostFree(hsaBuffer, size);
  return (status == HSA_STATUS_SUCCESS);
}

bool HsaBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                const amd::Coord3D& origin,
                                const amd::Coord3D& size, bool entire) const {
  hsa_memory_register(dstHost, size[0]);
  void* src = static_cast<roc::Memory&>(srcMemory).getDeviceMemory();

  // Copy data from device to host
  const void *srcDev = reinterpret_cast<const_address>(src) + origin[0];
  bool retval = hsaCopy(srcDev, dstHost, size[0], false);

  hsa_memory_deregister(dstHost, size[0]);
  return retval;
}

bool HsaBlitManager::readBufferRect(device::Memory& srcMemory, void* dst,
                                    const amd::BufferRect& bufRect,
                                    const amd::BufferRect& hostRect,
                                    const amd::Coord3D& size,
                                    bool entire) const {
  void* src = static_cast<roc::Memory&>(srcMemory).getDeviceMemory();

  size_t srcOffset;
  size_t dstOffset;

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      srcOffset = bufRect.offset(0, y, z);
      dstOffset = hostRect.offset(0, y, z);

      // Copy data from device to host - line by line
      void *dstHost = reinterpret_cast<address>(dst) + dstOffset;
      const void *srcDev = reinterpret_cast<const_address>(src) + srcOffset;
      bool retval = hsaCopy(srcDev, dstHost, size[0], false);
      if (!retval) {
        return retval;
      }
    }
  }

  return true;
}

static bool hsaCopyImageToBuffer(hsa_agent_t agent,
                                 hsa_ext_image_t srcImage,
                                 void* dstBuffer, const amd::Coord3D& srcOrigin,
                                 const amd::Coord3D& dstOrigin,
                                 const amd::Coord3D& size, bool entire,
                                 size_t rowPitch, size_t slicePitch) {
  hsa_ext_image_region_t image_region;
  image_region.offset.x = srcOrigin[0];
  image_region.offset.y = srcOrigin[1];
  image_region.offset.z = srcOrigin[2];
  image_region.range.x = size[0];
  image_region.range.y = size[1];
  image_region.range.z = size[2];

  char *dstHost = ((char*)dstBuffer) + dstOrigin[0];

  hsa_status_t status = hsa_ext_image_export(agent, srcImage, dstHost, rowPitch,
                                             slicePitch, &image_region);
  return (status == HSA_STATUS_SUCCESS);
}

bool HsaBlitManager::readImage(device::Memory& srcMemory, void* dstHost,
                               const amd::Coord3D& origin,
                               const amd::Coord3D& size, size_t rowPitch,
                               size_t slicePitch, bool entire) const {
  roc::Image* srcImage = (roc::Image*)&srcMemory;

  void* svmDstHost = NULL;
  size_t pinSize = 0;
  FindPinSize(pinSize, size, rowPitch, slicePitch, *srcImage);

  hsa_agent_t agent = gpu().gpu_device();

  hsa_status_t status = hsa_amd_memory_lock(dstHost, pinSize,
      &agent, 1, &svmDstHost);

  if (status != HSA_STATUS_SUCCESS) {
      return false;
  }

  bool retval = hsaCopyImageToBuffer(agent, srcImage->getHsaImageObject(),
                              svmDstHost, origin, amd::Coord3D(0), size, entire,
                              rowPitch, slicePitch);
  hsa_amd_memory_unlock(dstHost);
  return retval;
}

bool HsaBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                 const amd::Coord3D& origin,
                                 const amd::Coord3D& size, bool entire) const {
  hsa_memory_register(const_cast<void*>(srcHost), size[0]);
  void* dst = static_cast<roc::Memory&>(dstMemory).getDeviceMemory();

  // Copy data from host to device
  void *dstDev = reinterpret_cast<address>(dst) + origin[0];
  bool retval = hsaCopy(srcHost, dstDev, size[0], true);
  
  hsa_memory_deregister(const_cast<void*>(srcHost), size[0]);
  return retval;
}

bool HsaBlitManager::writeBufferRect(const void* src,
                                     device::Memory& dstMemory,
                                     const amd::BufferRect& hostRect,
                                     const amd::BufferRect& bufRect,
                                     const amd::Coord3D& size,
                                     bool entire) const {
  void* dst = static_cast<roc::Memory&>(dstMemory).getDeviceMemory();

  size_t srcOffset;
  size_t dstOffset;

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      srcOffset = hostRect.offset(0, y, z);
      dstOffset = bufRect.offset(0, y, z);

      // Copy data from host to device - line by line
      void *dstDev = reinterpret_cast<address>(dst) + dstOffset;
      const void *srcHost = reinterpret_cast<const_address>(src) + srcOffset;
      bool retval = hsaCopy(srcHost, dstDev, size[0], true);
      if (!retval) {
        return retval;
      }
    }
  }

  return true;
}

bool hsaCopyBufferToImage(hsa_agent_t agent, const void* srcBuffer,
                          hsa_ext_image_t dstImage,
                          const amd::Coord3D& srcOrigin,
                          const amd::Coord3D& dstOrigin,
                          const amd::Coord3D& size, bool entire,
                          size_t rowPitch, size_t slicePitch) {
  char* srcHost = ((char*)srcBuffer) + srcOrigin[0];

  hsa_ext_image_region_t image_region;
  image_region.offset.x = dstOrigin[0];
  image_region.offset.y = dstOrigin[1];
  image_region.offset.z = dstOrigin[2];
  image_region.range.x = size[0];
  image_region.range.y = size[1];
  image_region.range.z = size[2];

  hsa_status_t status = hsa_ext_image_import(
      agent, srcHost, rowPitch, slicePitch, dstImage, &image_region);
  return (status == HSA_STATUS_SUCCESS);
}

bool HsaBlitManager::writeImage(const void* srcHost, device::Memory& dstMemory,
                                const amd::Coord3D& origin,
                                const amd::Coord3D& size, size_t rowPitch,
                                size_t slicePitch, bool entire) const {
  roc::Image* image = (roc::Image*)&dstMemory;

  void* svmSrcHost = NULL;
  size_t pinSize = 0;
  FindPinSize(pinSize, size, rowPitch, slicePitch, *image);

  hsa_agent_t agent = gpu().gpu_device();

  hsa_status_t status = hsa_amd_memory_lock(const_cast<void*>(srcHost), pinSize,
      &agent, 1, &svmSrcHost);

  if (status != HSA_STATUS_SUCCESS) {
      return false;
  }

  bool retval = hsaCopyBufferToImage(agent, svmSrcHost,
                              image->getHsaImageObject(), amd::Coord3D(0),
                              origin, size, entire, rowPitch, slicePitch);

  hsa_amd_memory_unlock(const_cast<void*>(srcHost));

  return retval;
}

bool HsaBlitManager::copyBuffer(device::Memory& srcMemory,
                                device::Memory& dstMemory,
                                const amd::Coord3D& srcOrigin,
                                const amd::Coord3D& dstOrigin,
                                const amd::Coord3D& size, bool entire) const {
  void* src = static_cast<roc::Memory&>(srcMemory).getDeviceMemory();
  void* dst = static_cast<roc::Memory&>(dstMemory).getDeviceMemory();

  if (srcMemory.isHostMemDirectAccess() && dstMemory.isHostMemDirectAccess()) {
    if (srcMemory.owner()->getMemFlags() & CL_MEM_USE_HOST_PTR) {
      src = srcMemory.owner()->getHostMem();
    }

    if (dstMemory.owner()->getMemFlags() & CL_MEM_USE_HOST_PTR) {
      dst = dstMemory.owner()->getHostMem();
    }
  }

  const hsa_agent_t src_agent = (srcMemory.isHostMemDirectAccess())
                                    ? roc_device_.getCpuAgent()
                                    : roc_device_.getBackendDevice();

  const hsa_agent_t dst_agent = (dstMemory.isHostMemDirectAccess())
                                    ? roc_device_.getCpuAgent()
                                    : roc_device_.getBackendDevice();

  // Straight forward buffer copy
  const hsa_signal_value_t kInitVal = 1;
  hsa_signal_store_relaxed(completion_signal_, kInitVal);
  hsa_status_t status = hsa_amd_memory_async_copy(
      (reinterpret_cast<address>(dst) + dstOrigin[0]), dst_agent,
      (reinterpret_cast<const_address>(src) + srcOrigin[0]), src_agent, size[0],
      0, NULL, completion_signal_);
  if (status != HSA_STATUS_SUCCESS) {
    LogPrintfError("DMA buffer failed with code %d", status);
    return false;
  }

  hsa_signal_value_t val =
      hsa_signal_wait_acquire(completion_signal_, HSA_SIGNAL_CONDITION_EQ, 0,
                              uint64_t(-1), HSA_WAIT_STATE_ACTIVE);

  if (val != (kInitVal - 1)) {
    LogError("Async copy failed");
    return false;
  }

  return true;
}

bool HsaBlitManager::copyBufferRect(device::Memory& srcMemory,
                                    device::Memory& dstMemory,
                                    const amd::BufferRect& srcRect,
                                    const amd::BufferRect& dstRect,
                                    const amd::Coord3D& size,
                                    bool entire) const {
  void* src = static_cast<roc::Memory&>(srcMemory).getDeviceMemory();
  void* dst = static_cast<roc::Memory&>(dstMemory).getDeviceMemory();

  const hsa_signal_value_t kInitVal = size[2] * size[1];
  hsa_signal_store_relaxed(completion_signal_, kInitVal);

  for (size_t z = 0; z < size[2]; ++z) {
    for (size_t y = 0; y < size[1]; ++y) {
      size_t srcOffset = srcRect.offset(0, y, z);
      size_t dstOffset = dstRect.offset(0, y, z);

      // Copy memory line by line
      hsa_status_t status = hsa_amd_memory_async_copy(
          (reinterpret_cast<address>(dst) + dstOffset),
          roc_device_.getBackendDevice(),
          (reinterpret_cast<const_address>(src) + srcOffset),
          roc_device_.getBackendDevice(), size[0], 0, NULL,
          completion_signal_);
      if (status != HSA_STATUS_SUCCESS) {
        LogPrintfError("DMA buffer failed with code %d", status);
        return false;
      }
    }
  }

  hsa_signal_value_t val =
    hsa_signal_wait_acquire(completion_signal_, HSA_SIGNAL_CONDITION_EQ,
    0, uint64_t(-1), HSA_WAIT_STATE_ACTIVE);

  if (val != 0) {
    LogError("Async copy failed");
    return false;
  }

  return true;
}

bool HsaBlitManager::copyImageToBuffer(device::Memory& srcMemory,
                                       device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin,
                                       const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire,
                                       size_t rowPitch,
                                       size_t slicePitch) const {
  roc::Image& srcImage = (roc::Image&)srcMemory;
  roc::Buffer& dstBuffer = (roc::Buffer&)dstMemory;

  return hsaCopyImageToBuffer(gpu().gpu_device(), srcImage.getHsaImageObject(),
                              dstBuffer.getDeviceMemory(), srcOrigin, dstOrigin,
                              size, entire, rowPitch, slicePitch);
}

bool HsaBlitManager::copyBufferToImage(device::Memory& srcMemory,
                                       device::Memory& dstMemory,
                                       const amd::Coord3D& srcOrigin,
                                       const amd::Coord3D& dstOrigin,
                                       const amd::Coord3D& size, bool entire,
                                       size_t rowPitch,
                                       size_t slicePitch) const {
  roc::Buffer& srcBuffer = (roc::Buffer&)srcMemory;
  roc::Image& dstImage = (roc::Image&)dstMemory;

  return hsaCopyBufferToImage(gpu().gpu_device(), srcBuffer.getDeviceMemory(),
                              dstImage.getHsaImageObject(), srcOrigin,
                              dstOrigin, size, entire, rowPitch, slicePitch);
}

bool HsaBlitManager::copyImage(device::Memory& srcMemory,
                               device::Memory& dstMemory,
                               const amd::Coord3D& srcOrigin,
                               const amd::Coord3D& dstOrigin,
                               const amd::Coord3D& size, bool entire) const {
  if (srcMemory.isHostMemDirectAccess() &&
    dstMemory.isHostMemDirectAccess()) {
    return device::HostBlitManager::copyImage(
      srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire);
  }

  roc::Image *srcImage = (roc::Image *)&srcMemory;
  roc::Image *dstImage = (roc::Image *)&dstMemory;

  hsa_dim3_t src_offset = { 0 };
  src_offset.x = srcOrigin[0];
  src_offset.y = srcOrigin[1];
  src_offset.z = srcOrigin[2];

  hsa_dim3_t dst_offset = { 0 };
  dst_offset.x = dstOrigin[0];
  dst_offset.y = dstOrigin[1];
  dst_offset.z = dstOrigin[2];

  hsa_dim3_t copy_size = { 0 };
  copy_size.x = size[0];
  copy_size.y = size[1];
  copy_size.z = size[2];

  hsa_status_t status = hsa_ext_image_copy(
    gpu().gpu_device(), srcImage->getHsaImageObject(), &src_offset,
    dstImage->getHsaImageObject(), &dst_offset, &copy_size);
  return (status == HSA_STATUS_SUCCESS);
}

bool HsaBlitManager::fillBuffer(device::Memory& memory, const void* pattern,
                                size_t patternSize, const amd::Coord3D& origin,
                                const amd::Coord3D& size, bool entire) const {
  void* fillMem = static_cast<roc::Memory&>(memory).getDeviceMemory();

  size_t offset = origin[0];
  size_t fillSize = size[0];

  if ((fillSize % patternSize) != 0) {
    LogError("Misaligned buffer size and pattern size!");
  }

  // Fill the buffer memory with a pattern
  for (size_t i = 0; i < (fillSize / patternSize); i++) {
    void *dstDev = reinterpret_cast<address>(fillMem) + offset;
    bool retval = hsaCopy(pattern, dstDev, patternSize, true);
    if (!retval) {
      LogError("DMA buffer failed with code");
      return retval;
    }

    offset += patternSize;
  }

  return true;
}

bool HsaBlitManager::fillImage(device::Memory& memory, const void* pattern,
                               const amd::Coord3D& origin,
                               const amd::Coord3D& size, bool entire) const {
  if (memory.isHostMemDirectAccess()) {
    return device::HostBlitManager::fillImage(memory, pattern, origin, size, entire);
  }

  roc::Image *image = (roc::Image*)&memory;
  hsa_ext_image_region_t image_region;
  image_region.offset.x = origin[0];
  image_region.offset.y = origin[1];
  image_region.offset.z = origin[2];
  image_region.range.x = size[0];
  image_region.range.y = size[1];
  image_region.range.z = size[2];

  hsa_status_t status = hsa_ext_image_clear(
    gpu().gpu_device(), image->getHsaImageObject(),
    pattern, &image_region);
  return (status == HSA_STATUS_SUCCESS);
}

static void
CalcRowSlicePitches(
    cl_ulong* pitch, const cl_int* copySize,
    size_t rowPitch, size_t slicePitch, const Memory& mem)
{
    const roc::Image &hsaImage = static_cast< const roc::Image &>(mem);
    bool img1Darray =
        (mem.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? true : false;
    size_t memFmtSize = mem.owner()->asImage()->getImageFormat().getElementSize();

    if (rowPitch == 0) {
        pitch[0] = copySize[0];
    }
    else {
        pitch[0] = rowPitch / memFmtSize;
    }
    if (slicePitch == 0) {
        pitch[1] = pitch[0] * (img1Darray ? 1 : copySize[1]);
    }
    else {
        pitch[1] = slicePitch / memFmtSize;
    }
    assert((pitch[0] <= pitch[1]) && "rowPitch must be <= slicePitch");

    if (img1Darray) {
        // For 1D array rowRitch = slicePitch
        pitch[0] = pitch[1];
    }
}

KernelBlitManager::KernelBlitManager(device::VirtualDevice& vDev, Setup setup)
    : HsaBlitManager(vDev, setup),
      context_(NULL),
      program_(NULL)
{
    for (uint i = 0; i < BlitTotal; ++i) {
        kernels_[i] = NULL;
    }
}

KernelBlitManager::~KernelBlitManager()
{
    for (uint i = 0; i < BlitTotal; ++i) {
        if (NULL != kernels_[i]) {
            kernels_[i]->release();
        }
    }

    if (NULL != program_) {
        program_->release();
    }

    if (NULL != context_) {
        // Release a dummy context
        context_->release();
    }
}

bool
KernelBlitManager::readBuffer(
    device::Memory& srcMemory,
    void*       dstHost,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire) const
{
    //if (setup_.disableReadBuffer_ || srcMemory.isHostMemDirectAccess()) {
    //    return device::HostBlitManager::readBuffer(srcMemory, dstHost, origin,
    //                                      size, entire);
    //}
    // Exercise HSA path for now.
    return HsaBlitManager::readBuffer(srcMemory, dstHost, origin,
      size, entire);

    amd::Buffer *dstMemory = new (*context_) amd::Buffer(
        *context_, CL_MEM_USE_HOST_PTR, size[0]);

    if (!dstMemory->create(const_cast<void *>(dstHost))) {
        LogError("[OCL] Fail to create mem object for destination");
        return false;
    }

    device::Memory *devDstMemory = dstMemory->getDeviceMemory(dev_);
    if (devDstMemory== NULL) {
        LogError("[OCL] Fail to create device mem object for destination");
        return false;
    }

    bool result = copyBuffer(
        srcMemory, *devDstMemory, origin, amd::Coord3D(0), size, entire);

    // Wait for the transfer to finish so that we could safely release the
    // destination memory object.
    // TODO: we could remove this if issue on implicit memory registration is
    // fixed by KFD, so that we could pass the pattern as SVM.
    gpu().releaseGpuMemoryFence();

    dstMemory->release();

    return result;
}

bool
KernelBlitManager::readBufferRect(
    device::Memory& srcMemory,
    void*       dstHost,
    const amd::BufferRect&   bufRect,
    const amd::BufferRect&   hostRect,
    const amd::Coord3D& size,
    bool        entire) const
{
  //  if (setup_.disableReadBufferRect_ || srcMemory.isHostMemDirectAccess()) {
		//return device::HostBlitManager::readBufferRect(
  //          srcMemory, dstHost, bufRect, hostRect, size, entire);
  //  }

    // Exercise HSA path for now.
    return HsaBlitManager::readBufferRect(
      srcMemory, dstHost, bufRect, hostRect, size, entire);

    size_t  dstSize = hostRect.start_ + hostRect.end_;
    amd::Buffer *dstMemory =
        new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, dstSize);

    if (!dstMemory->create(const_cast<void *>(dstHost))) {
        LogError("[OCL] Fail to create mem object for destination");
        return false;
    }

    device::Memory *devDstMemory = dstMemory->getDeviceMemory(dev_);
    if (devDstMemory== NULL) {
        LogError("[OCL] Fail to create device mem object for destination");
        return false;
    }

    bool result = copyBufferRect(
        srcMemory, *devDstMemory, bufRect, hostRect, size, entire);

    // Wait for the transfer to finish so that we could safely release the
    // destination memory object.
    // TODO: we could remove this if issue on implicit memory registration is
    // fixed by KFD, so that we could pass the pattern as SVM.
    gpu().releaseGpuMemoryFence();

    dstMemory->release();

    return result;
}

void
FindLinearSize(
    size_t& linearSize, const amd::Coord3D& size,
    size_t& rowPitch, size_t& slicePitch, const device::Memory& mem)
{
    const roc::Image &image = static_cast<const roc::Image &>(mem);
    size_t elementSize = mem.owner()->asImage()->getImageFormat().getElementSize();

    linearSize = size[0] * elementSize;
    if ((rowPitch == 0) || (rowPitch == linearSize)) {
        rowPitch = 0;
    }
    else {
        linearSize = rowPitch;
    }

    // Calculate the pin size, which should be equal to the copy size
    for (uint i = 1; i < mem.owner()->asImage()->getDims(); ++i) {
        linearSize *= size[i];
        if (i == 1) {
            if ((slicePitch == 0) || (slicePitch == linearSize)) {
                slicePitch = 0;
            }
            else {
                if (mem.owner()->getType() != CL_MEM_OBJECT_IMAGE1D_ARRAY) {
                    linearSize = slicePitch;
                }
                else {
                    linearSize = slicePitch * size[i];
                }
            }
        }
    }
}

// The following data structures will be used for the view creations.
// Some formats has to be converted before a kernel blit operation
struct FormatConvertion {
    cl_uint clOldType_;
    cl_uint clNewType_;
};

// The list of rejected data formats and corresponding conversion
static const FormatConvertion RejectedData[] =
{
    { CL_UNORM_INT8,            CL_UNSIGNED_INT8  },
    { CL_UNORM_INT16,           CL_UNSIGNED_INT16 },
    { CL_SNORM_INT8,            CL_UNSIGNED_INT8  },
    { CL_SNORM_INT16,           CL_UNSIGNED_INT16 },
    { CL_HALF_FLOAT,            CL_UNSIGNED_INT16 },
    { CL_FLOAT,                 CL_UNSIGNED_INT32 },
    { CL_SIGNED_INT8,           CL_UNSIGNED_INT8  },
    { CL_SIGNED_INT16,          CL_UNSIGNED_INT16 },
    { CL_UNORM_INT_101010, CL_UNSIGNED_INT8 },
    { CL_SIGNED_INT32,          CL_UNSIGNED_INT32 }
};

// The list of rejected channel's order and corresponding conversion
static const FormatConvertion RejectedOrder[] =
{
    { CL_A,                     CL_R  },
    { CL_RA,                    CL_RG },
    { CL_LUMINANCE,             CL_R  },
    { CL_INTENSITY,             CL_R },
    { CL_RGB, CL_RGBA },
    { CL_BGRA,                  CL_RGBA },
    { CL_ARGB,                  CL_RGBA },
    { CL_sRGB, CL_RGBA },
    { CL_sRGBx, CL_RGBA },
    { CL_sRGBA, CL_RGBA },
    { CL_sBGRA, CL_RGBA },
    { CL_DEPTH, CL_R}
};

const uint RejectedFormatDataTotal =
        sizeof(RejectedData) / sizeof(FormatConvertion);
const uint RejectedFormatChannelTotal =
        sizeof(RejectedOrder) / sizeof(FormatConvertion);

amd::Image::Format
KernelBlitManager::filterFormat(amd::Image::Format oldFormat) const
{
    cl_image_format newFormat;
    newFormat.image_channel_data_type = oldFormat.image_channel_data_type;
    newFormat.image_channel_order = oldFormat.image_channel_order;

    // Find unsupported formats
    for (uint i = 0; i < RejectedFormatDataTotal; ++i) {
        if (RejectedData[i].clOldType_ == oldFormat.image_channel_data_type) {
            newFormat.image_channel_data_type = RejectedData[i].clNewType_;
            break;
        }
    }

    // Find unsupported channel's order
    for (uint i = 0; i < RejectedFormatChannelTotal; ++i) {
        if (RejectedOrder[i].clOldType_ == oldFormat.image_channel_order) {
            newFormat.image_channel_order = RejectedOrder[i].clNewType_;
            break;
        }
    }

    return amd::Image::Format(newFormat);
}

device::Memory *
KernelBlitManager::createImageView(
        device::Memory &parent,
        amd::Image::Format newFormat) const
{
    amd::Image *image =
        parent.owner()->asImage()->createView(parent.owner()->getContext(), newFormat, &gpu());

    if (image == NULL) {
        LogError("[OCL] Fail to allocate view of image object");
        return NULL;
    }

    Image* devImage = new roc::Image(static_cast<const Device &>(dev_), *image);
    if (devImage == NULL) {
        LogError("[OCL] Fail to allocate device mem object for the view");
        image->release();
        return NULL;
    }

    if (!devImage->createView(static_cast<roc::Image &>(parent))) {
        LogError("[OCL] Fail to create device mem object for the view");
        delete devImage;
        image->release();
        return NULL;
    }

    image->replaceDeviceMemory(&dev_, devImage);

    return devImage;
}

bool
KernelBlitManager::readImage(
    device::Memory& srcMemory,
    void*       dstHost,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    size_t      rowPitch,
    size_t      slicePitch,
    bool        entire) const
{
  return HsaBlitManager::readImage(
    srcMemory, dstHost, origin, size, rowPitch, slicePitch, entire);
}

bool
KernelBlitManager::writeBuffer(
    const void* srcHost,
    device::Memory& dstMemory,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire) const
{
  //  if (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess()) {
		//return device::HostBlitManager::writeBuffer(srcHost, dstMemory, origin, size,
  //                                         entire);
  //  }

    // Exercise HSA path for now.
    return HsaBlitManager::writeBuffer(srcHost, dstMemory, origin, size,
      entire);

    amd::Buffer *srcMemory =
        new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, size[0]);

    if (!srcMemory->create(const_cast<void *>(srcHost))) {
        LogError("[OCL] Fail to create mem object for destination");
        return false;
    }

    device::Memory *devSrcMemory = srcMemory->getDeviceMemory(dev_);
    if (devSrcMemory== NULL) {
        LogError("[OCL] Fail to create device mem object for destination");
        return false;
    }

    bool result =
        copyBuffer(*devSrcMemory, dstMemory, amd::Coord3D(0), origin, size, entire);

    // Wait for the transfer to finish so that we could safely release the
    // source memory object.
    // TODO: we could remove this if issue on implicit memory registration is
    // fixed by KFD, so that we could pass the pattern as SVM.
    gpu().releaseGpuMemoryFence();

    srcMemory->release();

    return result;
}

bool
KernelBlitManager::writeBufferRect(
    const void* srcHost,
    device::Memory& dstMemory,
    const amd::BufferRect&   hostRect,
    const amd::BufferRect&   bufRect,
    const amd::Coord3D& size,
    bool        entire) const
{
  //  if (setup_.disableWriteBufferRect_ || dstMemory.isHostMemDirectAccess()) {
		//return device::HostBlitManager::writeBufferRect(
  //          srcHost, dstMemory, hostRect, bufRect, size, entire);
  //  }

    // Exercise HSA path for now.
    return HsaBlitManager::writeBufferRect(
      srcHost, dstMemory, hostRect, bufRect, size, entire);

    size_t  srcSize = hostRect.start_ + hostRect.end_;
    amd::Buffer *srcMemory =
        new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, srcSize);

    if (!srcMemory->create(const_cast<void *>(srcHost))) {
        LogError("[OCL] Fail to create mem object for destination");
        return false;
    }

    device::Memory *devSrcMemory = srcMemory->getDeviceMemory(dev_);
    if (devSrcMemory== NULL) {
        LogError("[OCL] Fail to create device mem object for destination");
        return false;
    }

    bool result = copyBufferRect(
        *devSrcMemory, dstMemory, hostRect, bufRect, size, entire);

    // Wait for the transfer to finish so that we could safely release the
    // destination memory object.
    // TODO: we could remove this if issue on implicit memory registration is
    // fixed by KFD, so that we could pass the pattern as SVM.
    gpu().releaseGpuMemoryFence();

    srcMemory->release();

    return result;
}

bool
KernelBlitManager::writeImage(
    const void* srcHost,
    device::Memory& dstMemory,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    size_t      rowPitch,
    size_t      slicePitch,
    bool        entire) const
{
  return HsaBlitManager::writeImage(
    srcHost, dstMemory, origin, size, rowPitch, slicePitch, entire);
}

bool
KernelBlitManager::copyBuffer(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& sizeIn,
    bool        entire) const
{
  //  if (setup_.disableCopyBuffer_ ||
  //      (srcMemory.isHostMemDirectAccess()  &&
  //       dstMemory.isHostMemDirectAccess())) {
		//return HsaBlitManager::copyBuffer(
  //          srcMemory, dstMemory, srcOrigin, dstOrigin, sizeIn, entire);
  //  }

    // Exercise HSA path for now.
    return HsaBlitManager::copyBuffer(
      srcMemory, dstMemory, srcOrigin, dstOrigin, sizeIn, entire);

    uint    blitType = BlitCopyBuffer;
    size_t  dim = 1;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    size_t  globalWorkSize = 0;
    size_t  localWorkSize = 0;

    const static uint CopyBuffAlignment[3] = { 16, 4, 1 };
    amd::Coord3D size(sizeIn[0], sizeIn[1], sizeIn[2]);

    bool aligned;
    uint i;
    for (i = 0; i < 3; ++i) {
        // Check source alignments
        aligned = ((srcOrigin[0] % CopyBuffAlignment[i]) == 0);
        // Check destination alignments
        aligned &= ((dstOrigin[0] % CopyBuffAlignment[i]) == 0);
        // Check copy size alignment in the first dimension
        aligned &= ((sizeIn[0] % CopyBuffAlignment[i]) == 0);

        if (aligned) {
            if (CopyBuffAlignment[i] != 1) {
                blitType = BlitCopyBufferAligned;
            }
            break;
        }
    }

    cl_uint remain;
    if (blitType == BlitCopyBufferAligned) {
        size.c[0] /= CopyBuffAlignment[i];
    }
    else {
        remain = size[0] % 4;
        size.c[0] /= 4;
        size.c[0] += 1;
    }

    // Program the dispatch dimensions
    localWorkSize = 256;
    globalWorkSize = amd::alignUp(size[0] , 256);

    // Program kernels arguments for the blit operation
    cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(srcMemory.owner()));
    kernels_[blitType]->parameters().set(0, sizeof(cl_mem), &clmem);
    clmem = ((cl_mem) as_cl<amd::Memory>(dstMemory.owner()));
    kernels_[blitType]->parameters().set(1, sizeof(cl_mem), &clmem);
    // Program source origin
    cl_ulong  srcOffset = srcOrigin[0] / CopyBuffAlignment[i];
    kernels_[blitType]->parameters().set(2, sizeof(srcOffset), &srcOffset);

    // Program destination origin
    cl_ulong  dstOffset = dstOrigin[0] / CopyBuffAlignment[i];
    kernels_[blitType]->parameters().set(3, sizeof(dstOffset), &dstOffset);

    cl_ulong  copySize = size[0];
    kernels_[blitType]->parameters().set(4, sizeof(copySize), &copySize);

    if (blitType == BlitCopyBufferAligned) {
        cl_int  alignment = CopyBuffAlignment[i];
        kernels_[blitType]->parameters().set(5, sizeof(alignment), &alignment);
    }
    else {
        kernels_[blitType]->parameters().set(5, sizeof(remain), &remain);
    }

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(
        1, globalWorkOffset, &globalWorkSize, &localWorkSize);

    // Execute the blit
    address parameters = kernels_[blitType]->parameters().capture(dev_);
    bool result = gpu().submitKernelInternal(
        ndrange, *kernels_[blitType], parameters, NULL);
    kernels_[blitType]->parameters().release(const_cast<address>(parameters), dev_);
    return result;
}

bool
KernelBlitManager::copyBufferRect(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::BufferRect&   srcRectIn,
    const amd::BufferRect&   dstRectIn,
    const amd::Coord3D& sizeIn,
    bool        entire) const
{
  //  if (setup_.disableCopyBuffer_ ||
  //      (srcMemory.isHostMemDirectAccess() && dstMemory.isHostMemDirectAccess())) {
		//return HsaBlitManager::copyBufferRect(
  //          srcMemory, dstMemory, srcRectIn, dstRectIn, sizeIn, entire);
  //  }

    // Exercise HSA path for now.
    return HsaBlitManager::copyBufferRect(
      srcMemory, dstMemory, srcRectIn, dstRectIn, sizeIn, entire);

    uint    blitType = BlitCopyBufferRect;
    size_t  dim = 3;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    size_t  globalWorkSize[3];
    size_t  localWorkSize[3];

    const static uint CopyRectAlignment[3] = { 16, 4, 1 };

    bool aligned;
    uint i;
    for (i = 0; i < sizeof(CopyRectAlignment) / sizeof(uint); i++) {
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
    amd::Coord3D    size(sizeIn[0], sizeIn[1], sizeIn[2]);

    srcRect.rowPitch_      = srcRectIn.rowPitch_ / CopyRectAlignment[i];
    srcRect.slicePitch_    = srcRectIn.slicePitch_ / CopyRectAlignment[i];
    srcRect.start_         = srcRectIn.start_ / CopyRectAlignment[i];
    srcRect.end_           = srcRectIn.end_ / CopyRectAlignment[i];

    dstRect.rowPitch_      = dstRectIn.rowPitch_ / CopyRectAlignment[i];
    dstRect.slicePitch_    = dstRectIn.slicePitch_ / CopyRectAlignment[i];
    dstRect.start_         = dstRectIn.start_ / CopyRectAlignment[i];
    dstRect.end_           = dstRectIn.end_ / CopyRectAlignment[i];

    size.c[0] /= CopyRectAlignment[i];

    // Program the kernel's workload depending on the transfer dimensions
    if ((size[1] == 1) && (size[2] == 1)) {
        globalWorkSize[0] = amd::alignUp(size[0], 256);
        globalWorkSize[1] = 1;
        globalWorkSize[2] = 1;
        localWorkSize[0] = 256;
        localWorkSize[1] = 1;
        localWorkSize[2] = 1;
    }
    else if (size[2] == 1) {
        globalWorkSize[0] = amd::alignUp(size[0], 16);
        globalWorkSize[1] = amd::alignUp(size[1], 16);
        globalWorkSize[2] = 1;
        localWorkSize[0] = localWorkSize[1] = 16;
        localWorkSize[2] = 1;
    }
    else {
        globalWorkSize[0] = amd::alignUp(size[0], 8);
        globalWorkSize[1] = amd::alignUp(size[1], 8);
        globalWorkSize[2] = amd::alignUp(size[2], 4);
        localWorkSize[0] = localWorkSize[1] = 8;
        localWorkSize[2] = 4;
    }


    // Program kernels arguments for the blit operation
    cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(srcMemory.owner()));
    kernels_[blitType]->parameters().set(0, sizeof(cl_mem), &clmem);
    clmem = ((cl_mem) as_cl<amd::Memory>(dstMemory.owner()));
    kernels_[blitType]->parameters().set(1, sizeof(cl_mem), &clmem);
    cl_ulong  src[4] = {srcRect.rowPitch_,
                        srcRect.slicePitch_,
                        srcRect.start_, 0 };
    kernels_[blitType]->parameters().set(2, sizeof(src), src);
    cl_ulong  dst[4] = {dstRect.rowPitch_,
                        dstRect.slicePitch_,
                        dstRect.start_, 0 };
    kernels_[blitType]->parameters().set(3, sizeof(dst), dst);
    cl_ulong  copySize[4] = {size[0],
                             size[1],
                             size[2],
                             CopyRectAlignment[i] };
    kernels_[blitType]->parameters().set(4, sizeof(copySize), copySize);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(dim,
        globalWorkOffset, globalWorkSize, localWorkSize);

    // Execute the blit
    address parameters = kernels_[blitType]->parameters().capture(dev_);
    bool result = gpu().submitKernelInternal(
        ndrange, *kernels_[blitType], parameters, NULL);
    kernels_[blitType]->parameters().release(const_cast<address>(parameters), dev_);
    return result;
}

bool
KernelBlitManager::copyImageToBuffer(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    bool        entire,
    size_t      rowPitch,
    size_t      slicePitch) const
{
  if (dstMemory.isHostMemDirectAccess()) {
    return HsaBlitManager::copyImageToBuffer(srcMemory, dstMemory, srcOrigin,
                                             dstOrigin, size, entire, rowPitch,
                                             slicePitch);
  }

  amd::Image::Format oldFormat = srcMemory.owner()->asImage()->getImageFormat();
  amd::Image::Format newFormat = filterFormat(oldFormat);
  bool useView = false;

  device::Memory* srcView = &srcMemory;
  if (oldFormat != newFormat) {
    srcView = createImageView(srcMemory, newFormat);
    useView = true;
  }

  roc::Image& srcImage = static_cast<roc::Image&>(*srcView);

  amd::Image* image = srcImage.owner()->asImage();
  uint blitType = 0;
  blitType = BlitCopyImageToBuffer;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  const size_t imageDims = srcImage.owner()->asImage()->getDims();
  dim = 3;
  // Find the current blit type
  if (imageDims == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (imageDims == 2) {
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

  // Program kernels arguments for the blit operation
  cl_mem clmem = ((cl_mem)as_cl<amd::Memory>(srcImage.owner()));
  kernels_[blitType]->parameters().set(0, sizeof(cl_mem), &clmem);
  clmem = ((cl_mem)as_cl<amd::Memory>(dstMemory.owner()));
  kernels_[blitType]->parameters().set(1, sizeof(cl_mem), &clmem);

  // Update extra paramters for USHORT and UBYTE pointers.
  // Only then compiler can optimize the kernel to use
  // UAV Raw for other writes
  kernels_[blitType]->parameters().set(2, sizeof(cl_mem), &clmem);
  kernels_[blitType]->parameters().set(3, sizeof(cl_mem), &clmem);

  cl_int srcOrg[4] = {(cl_int)srcOrigin[0], (cl_int)srcOrigin[1],
                      (cl_int)srcOrigin[2], 0};
  cl_int copySize[4] = {(cl_int)size[0], (cl_int)size[1], (cl_int)size[2], 0};

  kernels_[blitType]->parameters().set(4, sizeof(srcOrg), srcOrg);

  const size_t elementSize =
      srcImage.owner()->asImage()->getImageFormat().getElementSize();
  const size_t numChannels =
      srcImage.owner()->asImage()->getImageFormat().getNumChannels();

  // 1 element granularity for writes by default
  cl_int granularity = 1;
  if (elementSize == 2) {
    granularity = 2;
  } else if (elementSize >= 4) {
    granularity = 4;
  }
  CondLog(((dstOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
  cl_ulong dstOrg[4] = {dstOrigin[0] / granularity, dstOrigin[1], dstOrigin[2],
                        0};
  kernels_[blitType]->parameters().set(5, sizeof(dstOrg), dstOrg);
  kernels_[blitType]->parameters().set(6, sizeof(copySize), copySize);

  // Program memory format
  uint multiplier = elementSize / sizeof(uint32_t);
  multiplier = (multiplier == 0) ? 1 : multiplier;
  cl_uint format[4] = {(cl_uint)numChannels,
                       (cl_uint)(elementSize / numChannels), multiplier, 0};
  kernels_[blitType]->parameters().set(7, sizeof(format), format);

  // Program row and slice pitches
  cl_ulong pitch[4] = {0};
  CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, srcImage);
  kernels_[blitType]->parameters().set(8, sizeof(pitch), pitch);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize,
                                localWorkSize);

  // Execute the blit
  address parameters = kernels_[blitType]->parameters().capture(dev_);
  bool result = gpu().submitKernelInternal(ndrange, *kernels_[blitType],
                                           parameters, NULL);
  kernels_[blitType]->parameters().release(const_cast<address>(parameters),
                                           dev_);

  if (useView) {
    srcView->owner()->release();
  }

  return result;
}

bool KernelBlitManager::copyBufferToImage(device::Memory& srcMemory,
                                          device::Memory& dstMemory,
                                          const amd::Coord3D& srcOrigin,
                                          const amd::Coord3D& dstOrigin,
                                          const amd::Coord3D& size, bool entire,
                                          size_t rowPitch,
                                          size_t slicePitch) const {
  if (srcMemory.isHostMemDirectAccess()) {
    return HsaBlitManager::copyBufferToImage(srcMemory, dstMemory, srcOrigin,
                                             dstOrigin, size, entire, rowPitch,
                                             slicePitch);
  }

  amd::Image::Format oldFormat = dstMemory.owner()->asImage()->getImageFormat();
  amd::Image::Format newFormat = filterFormat(oldFormat);
  bool useView = false;

  device::Memory* dstView = &dstMemory;
  if (oldFormat != newFormat) {
    dstView = createImageView(dstMemory, newFormat);
    useView = true;
  }

  roc::Image& dstImage = static_cast<roc::Image&>(*dstView);

  // Use a common blit type with three dimensions by default
  uint blitType = BlitCopyBufferToImage;
  size_t dim = 0;
  size_t globalWorkOffset[3] = {0, 0, 0};
  size_t globalWorkSize[3];
  size_t localWorkSize[3];

  // Program the kernels workload depending on the blit dimensions
  const size_t imageDims = dstImage.owner()->asImage()->getDims();
  dim = 3;
  if (imageDims == 1) {
    globalWorkSize[0] = amd::alignUp(size[0], 256);
    globalWorkSize[1] = amd::alignUp(size[1], 1);
    globalWorkSize[2] = amd::alignUp(size[2], 1);
    localWorkSize[0] = 256;
    localWorkSize[1] = localWorkSize[2] = 1;
  } else if (imageDims == 2) {
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

  // Program kernels arguments for the blit operation
  cl_mem clmem = ((cl_mem)as_cl<amd::Memory>(srcMemory.owner()));
  kernels_[blitType]->parameters().set(0, sizeof(cl_mem), &clmem);
  clmem = ((cl_mem)as_cl<amd::Memory>(dstImage.owner()));
  kernels_[blitType]->parameters().set(1, sizeof(cl_mem), &clmem);

  const size_t elementSize =
      dstImage.owner()->asImage()->getImageFormat().getElementSize();
  const size_t numChannels =
      dstImage.owner()->asImage()->getImageFormat().getNumChannels();

  // 1 element granularity for writes by default
  cl_int granularity = 1;
  if (elementSize == 2) {
    granularity = 2;
  } else if (elementSize >= 4) {
    granularity = 4;
  }
  CondLog(((srcOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
  cl_ulong srcOrg[4] = {srcOrigin[0] / granularity, srcOrigin[1], srcOrigin[2],
                        0};
  kernels_[blitType]->parameters().set(2, sizeof(srcOrg), srcOrg);

  cl_int dstOrg[4] = {(cl_int)dstOrigin[0], (cl_int)dstOrigin[1],
                      (cl_int)dstOrigin[2], 0};
  cl_int copySize[4] = {(cl_int)size[0], (cl_int)size[1], (cl_int)size[2], 0};

  kernels_[blitType]->parameters().set(3, sizeof(dstOrg), dstOrg);
  kernels_[blitType]->parameters().set(4, sizeof(copySize), copySize);

  // Program memory format
  uint multiplier = elementSize / sizeof(uint32_t);
  multiplier = (multiplier == 0) ? 1 : multiplier;
  cl_uint format[4] = {(cl_uint)numChannels,
                       (cl_uint)(elementSize / numChannels), multiplier, 0};
  kernels_[blitType]->parameters().set(5, sizeof(format), format);

  // Program row and slice pitches
  cl_ulong pitch[4] = {0};
  CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, dstImage);
  kernels_[blitType]->parameters().set(6, sizeof(pitch), pitch);

  // Create ND range object for the kernel's execution
  amd::NDRangeContainer ndrange(dim, globalWorkOffset, globalWorkSize,
                                localWorkSize);

  // Execute the blit
  address parameters = kernels_[blitType]->parameters().capture(dev_);
  bool result = gpu().submitKernelInternal(ndrange, *kernels_[blitType],
                                           parameters, NULL);
  kernels_[blitType]->parameters().release(const_cast<address>(parameters),
                                           dev_);

  if (useView) {
    dstView->owner()->release();
  }

  return result;
}

bool
KernelBlitManager::copyImage(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    bool        entire) const
{
  return HsaBlitManager::copyImage(
    srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire);
}

bool
KernelBlitManager::fillBuffer(
    device::Memory& memory,
    const void* pattern,
    size_t      patternSize,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire
    ) const
{
    if (setup_.disableFillBuffer_ || memory.isHostMemDirectAccess()) {
        return HostBlitManager::fillBuffer(memory, pattern, patternSize, origin,
                                           size, entire);
    }

    uint    fillType = FillBuffer;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    cl_ulong  fillSize = size[0] / patternSize;
    size_t  globalWorkSize = amd::alignUp(fillSize, 256);
    size_t  localWorkSize = 256;
    bool    dwordAligned =
        ((patternSize % sizeof(uint32_t)) == 0) ? true : false;

    // Program kernels arguments for the fill operation
    if (dwordAligned) {
        kernels_[fillType]->parameters().set(0, sizeof(cl_mem), NULL);
        cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(memory.owner()));
        kernels_[fillType]->parameters().set(1, sizeof(cl_mem), &clmem);
    }
    else {
        cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(memory.owner()));
        kernels_[fillType]->parameters().set(0, sizeof(cl_mem), &clmem);
        kernels_[fillType]->parameters().set(1, sizeof(cl_mem), NULL);
    }

    amd::Buffer *fillMemory =
        new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, patternSize);

    if (!fillMemory->create(const_cast<void *>(pattern))) {
        LogError("[OCL] Fail to create mem object for destination");
        return false;
    }

    if (fillMemory->getDeviceMemory(dev_) == NULL) {
        LogError("[OCL] Fail to create device mem object for destination");
        return false;
    }

    cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(fillMemory));
    kernels_[fillType]->parameters().set(2, sizeof(cl_mem), &clmem);
    cl_ulong  offset = origin[0];
    if (dwordAligned) {
        patternSize /= sizeof(uint32_t);
        offset /= sizeof(uint32_t);
    }
    kernels_[fillType]->parameters().set(3, sizeof(cl_uint), &patternSize);
    kernels_[fillType]->parameters().set(4, sizeof(offset), &offset);
    kernels_[fillType]->parameters().set(5, sizeof(fillSize), &fillSize);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(1,
        globalWorkOffset, &globalWorkSize, &localWorkSize);

    // Execute the blit
    address parameters = kernels_[fillType]->parameters().capture(dev_);
    bool result = gpu().submitKernelInternal(
        ndrange, *kernels_[fillType], parameters, NULL);
    kernels_[fillType]->parameters().release(const_cast<address>(parameters), dev_);

    // Wait for the transfer to finish so that we could safely release the
    // fill memory object.
    // TODO: we could remove this if issue on implicit memory registration is
    // fixed by KFD, so that we could pass the pattern as SVM.
    gpu().releaseGpuMemoryFence();

    fillMemory->release();

    return result;
}

bool
KernelBlitManager::fillImage(
    device::Memory&     memory,
    const void* pattern,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire
    ) const
{
  return HsaBlitManager::fillImage(memory, pattern, origin, size, entire);
}

bool
KernelBlitManager::create(amd::Device& device)
{
    if (!HsaBlitManager::create(device)) {
        return false;
    }
    if (!createProgram(static_cast<Device&>(device))) {
        return false;
    }

    return true;
}

bool
KernelBlitManager::createProgram(Device& device)
{
    // Save context and program for this device
    context_ = device.blitProgram()->context_;
    context_->retain();
    program_ = device.blitProgram()->program_;
    program_->retain();

    bool result = false;
    do {
        // Create kernel objects for all blits
        for (uint i = 0; i < BlitTotal; ++i) {
            const amd::Symbol* symbol = program_->findSymbol(BlitName[i]);
            if (symbol == NULL) {
                break;
            }
            kernels_[i] = new amd::Kernel(*program_, *symbol, BlitName[i]);
            if (kernels_[i] == NULL) {
                break;
            }
        }

        result = true;
    } while(!result);

    return result;
}

} // namespace roc
