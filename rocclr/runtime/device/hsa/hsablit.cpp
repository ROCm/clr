//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#include "platform/commandqueue.hpp"
#include "device/hsa/hsadevice.hpp"
#include "device/hsa/hsablit.hpp"
#include "device/hsa/hsamemory.hpp"
#include "device/hsa/hsavirtual.hpp"
#include "device/hsa/oclhsa_common.hpp"
#include "utils/debug.hpp"

namespace oclhsa {
HsaBlitManager::HsaBlitManager(device::VirtualDevice& vDev, Setup setup)
    : HostBlitManager(vDev, setup)
{ }

bool
HsaBlitManager::readBuffer(
    device::Memory& srcMemory,
    void*       dstHost,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    if (setup_.disableReadBuffer_ || srcMemory.isHostMemDirectAccess()) {
        return HostBlitManager::readBuffer(
            srcMemory, dstHost, origin, size, entire);
    }

    void *src = static_cast<oclhsa::Memory&>(srcMemory).getDeviceMemory();

    // Copy memory
    HsaStatus status = hsacoreapi->HsaCopyMemory(
        dstHost, reinterpret_cast<const_address>(src) + origin[0], size[0]);
    if (status != kHsaStatusSuccess) {
      LogPrintfError("DMA buffer failed with code %d", status);
      return false;
    }
    return true;
}

bool
HsaBlitManager::readBufferRect(
    device::Memory& srcMemory,
    void*       dstHost,
    const amd::BufferRect&   bufRect,
    const amd::BufferRect&   hostRect,
    const amd::Coord3D& size,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    if (setup_.disableReadBufferRect_ || srcMemory.isHostMemDirectAccess()) {
        return HostBlitManager::readBufferRect(
            srcMemory, dstHost, bufRect, hostRect, size, entire);
    }

    void *src = static_cast<oclhsa::Memory&>(srcMemory).getDeviceMemory();

    size_t  srcOffset;
    size_t  dstOffset;

    for (size_t z = 0; z < size[2]; ++z) {
        for (size_t y = 0; y < size[1]; ++y) {
            srcOffset   = bufRect.offset(0, y, z);
            dstOffset   = hostRect.offset(0, y, z);

            // Copy memory line by line
            HsaStatus status =
                hsacoreapi->HsaCopyMemory(
                    (reinterpret_cast<address>(dstHost) + dstOffset),
                    (reinterpret_cast<const_address>(src) + srcOffset),
                    size[0]);

            if (status != kHsaStatusSuccess) {
              LogPrintfError("DMA buffer failed with code %d", status);
              return false;
            }
        }
    }

    return true;
}

bool
HsaBlitManager::readImage(
    device::Memory& srcMemory,
    void*       dstHost,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    size_t      rowPitch,
    size_t      slicePitch,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    oclhsa::Image &image = static_cast<oclhsa::Image&>(srcMemory);

    const uint8_t *src = static_cast<const uint8_t*>(image.getDeviceMemory());
    uint8_t* dst = static_cast<uint8_t*>(dstHost);

    const amd::Coord3D srcOffset = origin;
    const amd::Coord3D dstOffset = amd::Coord3D(0);

    size_t srcRowPitch = image.getDeviceRowPitchSize();
    size_t srcSlicePitch = image.getDeviceSlicePitchSize();

    size_t elementSize =
      srcMemory.owner()->asImage()->getImageFormat().getElementSize();
    size_t dstRowPitch =
        (rowPitch == 0) ? (size[0] * elementSize) : rowPitch;
    size_t dstSlicePitch =
        (slicePitch == 0) ? (size[1] * dstRowPitch) : slicePitch;

    const amd::Coord3D& sizeToCopy = size;

    return importExportImage(
        dst, src, dstOffset, dstRowPitch, dstSlicePitch, srcOffset, srcRowPitch,
        srcSlicePitch, sizeToCopy, elementSize);
}

bool
HsaBlitManager::writeBuffer(
    const void* srcHost,
    device::Memory& dstMemory,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    if (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess()) {
        return HostBlitManager::writeBuffer(
            srcHost, dstMemory, origin, size, entire);
    }

    void *dst = static_cast<oclhsa::Memory&>(dstMemory).getDeviceMemory();

    // Copy memory
    HsaStatus status =
        hsacoreapi->HsaCopyMemory(
            reinterpret_cast<address>(dst) + origin[0], srcHost, size[0]);

    if (status != kHsaStatusSuccess) {
      LogPrintfError("DMA buffer failed with code %d", status);
      return false;
    }

    return true;
}

bool
HsaBlitManager::writeBufferRect(
    const void* srcHost,
    device::Memory& dstMemory,
    const amd::BufferRect&   hostRect,
    const amd::BufferRect&   bufRect,
    const amd::Coord3D& size,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    if (setup_.disableWriteBufferRect_ || dstMemory.isHostMemDirectAccess()) {
        return HostBlitManager::writeBufferRect(
            srcHost, dstMemory, hostRect, bufRect, size, entire);
    }

    void *dst = static_cast<oclhsa::Memory&>(dstMemory).getDeviceMemory();

    size_t  srcOffset;
    size_t  dstOffset;

    for (size_t z = 0; z < size[2]; ++z) {
        for (size_t y = 0; y < size[1]; ++y) {
            srcOffset   = hostRect.offset(0, y, z);
            dstOffset   = bufRect.offset(0, y, z);

            // Copy memory line by line
            HsaStatus status =
                hsacoreapi->HsaCopyMemory(
                    (reinterpret_cast<address>(dst) + dstOffset),
                    (reinterpret_cast<const_address>(srcHost) + srcOffset),
                    size[0]);

            if (status != kHsaStatusSuccess) {
              LogPrintfError("DMA buffer failed with code %d", status);
              return false;
            }
        }
    }

    return true;
}

bool
HsaBlitManager::writeImage(
    const void* srcHost,
    device::Memory& dstMemory,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    size_t      rowPitch,
    size_t      slicePitch,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    oclhsa::Image &image = static_cast<oclhsa::Image&>(dstMemory);

    const uint8_t* src = static_cast<const uint8_t*>(srcHost);
    uint8_t *dst = static_cast<uint8_t*>(image.getDeviceMemory());

    const amd::Coord3D srcOffset = amd::Coord3D(0);
    const amd::Coord3D dstOffset = origin;

    size_t elementSize =
      dstMemory.owner()->asImage()->getImageFormat().getElementSize();
    size_t srcRowPitch =
        (rowPitch == 0) ? (size[0] * elementSize) : rowPitch;
    size_t srcSlicePitch =
        (slicePitch == 0) ? (size[1] * srcRowPitch) : slicePitch;

    size_t dstRowPitch = image.getDeviceRowPitchSize();
    size_t dstSlicePitch = image.getDeviceSlicePitchSize();

    const amd::Coord3D& sizeToCopy = size;

    return importExportImage(
        dst, src, dstOffset, dstRowPitch, dstSlicePitch, srcOffset, srcRowPitch,
        srcSlicePitch, sizeToCopy, elementSize);
}

bool
HsaBlitManager::copyBuffer(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    if (setup_.disableCopyBuffer_ ||
        (srcMemory.isHostMemDirectAccess() &&
         dstMemory.isHostMemDirectAccess())) {
        return HostBlitManager::copyBuffer(
            srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire);
    }

    void *src = static_cast<oclhsa::Memory&>(srcMemory).getDeviceMemory();
    void *dst = static_cast<oclhsa::Memory&>(dstMemory).getDeviceMemory();

    // Straight forward buffer copy
    HsaStatus status =
        hsacoreapi->HsaCopyMemory(
            (reinterpret_cast<address>(dst) + dstOrigin[0]),
            (reinterpret_cast<const_address>(src) + srcOrigin[0]),
            size[0]);

    if (status != kHsaStatusSuccess) {
      LogPrintfError("DMA buffer failed with code %d", status);
      return false;
    }

    return true;
}

bool
HsaBlitManager::copyBufferRect(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::BufferRect&   srcRect,
    const amd::BufferRect&   dstRect,
    const amd::Coord3D& size,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    if (setup_.disableCopyBuffer_ ||
        (srcMemory.isHostMemDirectAccess() &&
         dstMemory.isHostMemDirectAccess())) {
        return HostBlitManager::copyBufferRect(
            srcMemory, dstMemory, srcRect, dstRect, size, entire);
    }

    void *src = static_cast<oclhsa::Memory&>(srcMemory).getDeviceMemory();
    void *dst = static_cast<oclhsa::Memory&>(dstMemory).getDeviceMemory();

    for (size_t z = 0; z < size[2]; ++z) {
        for (size_t y = 0; y < size[1]; ++y) {
            size_t srcOffset = srcRect.offset(0, y, z);
            size_t dstOffset = dstRect.offset(0, y, z);

            // Copy memory line by line
            HsaStatus status =
                hsacoreapi->HsaCopyMemory(
                    (reinterpret_cast<address>(dst) + dstOffset),
                    (reinterpret_cast<const_address>(src) + srcOffset),
                    size[0]);

            if (status != kHsaStatusSuccess) {
              LogPrintfError("DMA buffer failed with code %d", status);
              return false;
            }
        }
    }

    return true;
}

bool
HsaBlitManager::copyImageToBuffer(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    bool        entire,
    size_t      rowPitch,
    size_t      slicePitch) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    oclhsa::Image& srcImage = static_cast<oclhsa::Image&>(srcMemory);
    oclhsa::Buffer& destBuff = static_cast<oclhsa::Buffer&>(dstMemory);

    const uint8_t *src = static_cast<const uint8_t*>(srcImage.getDeviceMemory());
    uint8_t* dst = static_cast<uint8_t*>(destBuff.getDeviceMemory());

    size_t elementSize =
        srcMemory.owner()->asImage()->getImageFormat().getElementSize();
    size_t dstRowPitch = size[0] * elementSize;
    size_t dstSlicePitch = size[1] * dstRowPitch;

    size_t srcRowPitch = srcImage.getDeviceRowPitchSize();
    size_t srcSlicePitch = srcImage.getDeviceSlicePitchSize();

    return importExportImage(
        dst, src, dstOrigin, dstRowPitch, dstSlicePitch, srcOrigin, srcRowPitch,
        srcSlicePitch, size, elementSize);
}

bool
HsaBlitManager::copyBufferToImage(
    device::Memory&     srcMemory,
    device::Memory&     dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    bool        entire,
    size_t      rowPitch,
    size_t      slicePitch) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    oclhsa::Buffer& srcBuff = static_cast<oclhsa::Buffer&>(srcMemory);
    oclhsa::Image& dstImage = static_cast<oclhsa::Image&>(dstMemory);

    const uint8_t *src = static_cast<const uint8_t*>(srcBuff.getDeviceMemory());
    uint8_t* dst = static_cast<uint8_t*>(dstImage.getDeviceMemory());

    size_t elementSize =
        dstMemory.owner()->asImage()->getImageFormat().getElementSize();
    size_t srcRowPitch = size[0] * elementSize;
    size_t srcSlicePitch = size[1] * srcRowPitch;

    size_t dstRowPitch = dstImage.getDeviceRowPitchSize();
    size_t dstSlicePitch = dstImage.getDeviceSlicePitchSize();

    return importExportImage(
        dst, src, dstOrigin, dstRowPitch, dstSlicePitch, srcOrigin, srcRowPitch,
        srcSlicePitch, size, elementSize);
}

bool
HsaBlitManager::copyImage(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    bool        entire) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    oclhsa::Image& srcImage = static_cast<oclhsa::Image&>(srcMemory);
    oclhsa::Image& destImage = static_cast<oclhsa::Image&>(dstMemory);

    const uint8_t *src = static_cast<const uint8_t*>(srcImage.getDeviceMemory());
    uint8_t* dst = static_cast<uint8_t*>(destImage.getDeviceMemory());

    size_t srcRowPitch = srcImage.getDeviceRowPitchSize();
    size_t srcSlicePitch = srcImage.getDeviceSlicePitchSize();

    size_t dstRowPitch = destImage.getDeviceRowPitchSize();
    size_t dstSlicePitch = destImage.getDeviceSlicePitchSize();

    size_t elementSize =
        srcMemory.owner()->asImage()->getImageFormat().getElementSize();

    return importExportImage(
        dst, src, dstOrigin, dstRowPitch, dstSlicePitch, srcOrigin, srcRowPitch,
        srcSlicePitch, size, elementSize);
}

bool
HsaBlitManager::fillBuffer(
    device::Memory& memory,
    const void* pattern,
    size_t      patternSize,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire
    ) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    if (setup_.disableFillBuffer_ || memory.isHostMemDirectAccess()) {
        return HostBlitManager::fillBuffer(memory, pattern, patternSize,
                                           origin, size, entire);
    }

    void *fillMem = static_cast<oclhsa::Memory&>(memory).getDeviceMemory();

    size_t  offset      = origin[0];
    size_t  fillSize    = size[0];

    if ((fillSize % patternSize) != 0) {
        LogError("Misaligned buffer size and pattern size!");
    }

    // Fill the buffer memory with a pattern
    for (size_t i = 0; i < (fillSize / patternSize); i++) {
        HsaStatus status =
            hsacoreapi->HsaCopyMemory(
                (reinterpret_cast<address>(fillMem) + offset),
                (reinterpret_cast<const_address>(pattern)),
                patternSize);

        if (status != kHsaStatusSuccess) {
            LogPrintfError("DMA buffer failed with code %d", status);
            return false;
        }

        offset += patternSize;
    }

    return true;
}

bool
HsaBlitManager::fillImage(
    device::Memory&     memory,
    const void* pattern,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire
    ) const
{
    // Wait on the last outstanding kernel.
    gpu().releaseGpuMemoryFence();

    oclhsa::Image& image = static_cast<oclhsa::Image&>(memory);

    void *fillMem = image.getDeviceMemory();

    size_t elementSize =
        memory.owner()->asImage()->getImageFormat().getElementSize();

    float fillValue[4];
    memset(fillValue, 0, sizeof(fillValue));
    memory.owner()->asImage()->getImageFormat().formatColor(
        pattern, fillValue);

    size_t rowPitchSize = image.getDeviceRowPitchSize();
    size_t slicePitchSize = image.getDeviceSlicePitchSize();

    size_t offset  = origin[0] * elementSize;

    // Adjust offset with Y dimension
    offset += rowPitchSize * origin[1];

    // Adjust offset with Z dimension
    offset += slicePitchSize * origin[2];

    size_t offsetOrg = offset;

    // Fill the image memory with a pattern
    for (size_t slice = 0; slice < size[2]; ++slice) {
        offset = offsetOrg + slice * slicePitchSize;

        for (size_t rows = 0; rows < size[1]; ++rows) {
            size_t  pixOffset = offset;

            // Copy memory pixel by pixel
            for (size_t column = 0; column < size[0]; ++column) {
                HsaStatus status =
                    hsacoreapi->HsaCopyMemory(
                        (reinterpret_cast<address>(fillMem) + pixOffset),
                        (reinterpret_cast<const_address>(fillValue)),
                        elementSize);

                if (status != kHsaStatusSuccess) {
                    LogPrintfError("DMA buffer failed with code %d", status);
                    return false;
                }

                pixOffset += elementSize;
            }

            offset += rowPitchSize;
        }
    }

    return true;
}

bool
HsaBlitManager::importExportImage(
    uint8_t* dst,
    const uint8_t* src,
    const amd::Coord3D& dstOffset,
    size_t dstRowPitch,
    size_t dstSlicePitch,
    const amd::Coord3D& srcOffset,
    size_t srcRowPitch,
    size_t srcSlicePitch,
    const amd::Coord3D& sizeToCopy,
    size_t elementSize) const
{
    for (size_t zDim = 0; zDim < sizeToCopy[2]; ++zDim) {
        for (size_t yDim = 0; yDim < sizeToCopy[1]; ++yDim) {
            size_t srcImgOffset =
                srcOffset[0] * elementSize + (srcOffset[1] + yDim) * srcRowPitch +
                (srcOffset[2] + zDim) * srcSlicePitch;
            size_t dstImgOffset =
                dstOffset[0] * elementSize + (dstOffset[1] + yDim) * dstRowPitch +
                (dstOffset[2] + zDim) * dstSlicePitch;
            HsaStatus status = hsacoreapi->HsaCopyMemory(
                dst + dstImgOffset, src + srcImgOffset, sizeToCopy[0]*elementSize);

            if (status != kHsaStatusSuccess) {
              LogPrintfError("DMA import/export image failed with code %d", status);
              return false;
            }
        }
    }

    return true;
}

static void
CalcRowSlicePitches(
    cl_ulong* pitch, const cl_int* copySize,
    size_t rowPitch, size_t slicePitch, const Memory& mem)
{
    const oclhsa::Image &hsaImage = static_cast< const oclhsa::Image &>(mem);
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
    if (setup_.disableReadBuffer_ || srcMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::readBuffer(srcMemory, dstHost, origin,
                                          size, entire);
    }

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
    if (setup_.disableReadBufferRect_ || srcMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::readBufferRect(
            srcMemory, dstHost, bufRect, hostRect, size, entire);
    }

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
    const oclhsa::Image &image = static_cast<const oclhsa::Image &>(mem);
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
    { CL_SIGNED_INT32,          CL_UNSIGNED_INT32 }
};

// The list of rejected channel's order and corresponding conversion
static const FormatConvertion RejectedOrder[] =
{
    { CL_A,                     CL_R  },
    { CL_RA,                    CL_RG },
    { CL_LUMINANCE,             CL_R  },
    { CL_INTENSITY,             CL_R },
    { CL_BGRA,                  CL_RGBA },
    { CL_ARGB,                  CL_RGBA }
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

    Image* devImage = new oclhsa::Image(static_cast<const Device &>(dev_), *image);
    if (devImage == NULL) {
        LogError("[OCL] Fail to allocate device mem object for the view");
        image->release();
        return NULL;
    }

    if (!devImage->createView(static_cast<oclhsa::Image &>(parent))) {
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
    if (setup_.disableReadImage_ || srcMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::readImage(
            srcMemory, dstHost, origin, size, rowPitch, slicePitch, entire);
    }

    size_t linearSize = 0;
    FindLinearSize(linearSize, size, rowPitch, slicePitch, srcMemory);
    amd::Buffer *dstMemory =
        new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, linearSize);

    if (!dstMemory->create(const_cast<void *>(dstHost))) {
        LogError("[OCL] Fail to create mem object for destination");
        return false;
    }

    device::Memory *devDstMemory = dstMemory->getDeviceMemory(dev_);
    if (devDstMemory== NULL) {
        LogError("[OCL] Fail to create device mem object for destination");
        return false;
    }

    bool result = copyImageToBuffer(
        srcMemory, *devDstMemory, origin, amd::Coord3D(0), size, entire, rowPitch,
        slicePitch);

    // Wait for the transfer to finish so that we could safely release the
    // destination memory object.
    // TODO: we could remove this if issue on implicit memory registration is
    // fixed by KFD, so that we could pass the pattern as SVM.
    gpu().releaseGpuMemoryFence();

    dstMemory->release();

    return result;
}

bool
KernelBlitManager::writeBuffer(
    const void* srcHost,
    device::Memory& dstMemory,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    bool        entire) const
{
    if (setup_.disableWriteBuffer_ || dstMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::writeBuffer(srcHost, dstMemory, origin, size,
                                           entire);
    }

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
    if (setup_.disableWriteBufferRect_ || dstMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::writeBufferRect(
            srcHost, dstMemory, hostRect, bufRect, size, entire);
    }

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
    if (setup_.disableWriteImage_ || dstMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::writeImage(
            srcHost, dstMemory, origin, size, rowPitch, slicePitch, entire);
    }

    size_t linearSize = 0;
    FindLinearSize(linearSize, size, rowPitch, slicePitch, dstMemory);
    amd::Buffer *srcMemory =
        new (*context_) amd::Buffer(*context_, CL_MEM_USE_HOST_PTR, linearSize);

    if (!srcMemory->create(const_cast<void *>(srcHost))) {
        LogError("[OCL] Fail to create mem object for destination");
        return false;
    }

    device::Memory *devSrcMemory = srcMemory->getDeviceMemory(dev_);
    if (devSrcMemory== NULL) {
        LogError("[OCL] Fail to create device mem object for destination");
        return false;
    }

    bool result = copyBufferToImage(
        *devSrcMemory, dstMemory, amd::Coord3D(0), origin, size, entire,
        rowPitch, slicePitch);

    // Wait for the transfer to finish so that we could safely release the
    // destination memory object.
    // TODO: we could remove this if issue on implicit memory registration is
    // fixed by KFD, so that we could pass the pattern as SVM.
    gpu().releaseGpuMemoryFence();

    srcMemory->release();

    return result;
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
    if (setup_.disableCopyBuffer_ ||
        srcMemory.isHostMemDirectAccess() ||
        dstMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::copyBuffer(
            srcMemory, dstMemory, srcOrigin, dstOrigin, sizeIn, entire);
    }

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

    // Program destinaiton origin
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
    if (setup_.disableCopyBuffer_ ||
        (srcMemory.isHostMemDirectAccess() && dstMemory.isHostMemDirectAccess())) {
        return HsaBlitManager::copyBufferRect(
            srcMemory, dstMemory, srcRectIn, dstRectIn, sizeIn, entire);
    }

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
    cl_ulong    src[4] = { srcRect.rowPitch_,
                           srcRect.slicePitch_,
                           srcRect.start_, 0 };
    kernels_[blitType]->parameters().set(2, sizeof(src), src);
    cl_ulong    dst[4] = { dstRect.rowPitch_,
                           dstRect.slicePitch_,
                           dstRect.start_, 0 };
    kernels_[blitType]->parameters().set(3, sizeof(dst), dst);
    cl_ulong    copySize[4] = { size[0],
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
    if (srcMemory.isHostMemDirectAccess() && dstMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::copyImageToBuffer(
            srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
            rowPitch, slicePitch);
    }

    amd::Image::Format oldFormat = srcMemory.owner()->asImage()->getImageFormat();
    amd::Image::Format newFormat = filterFormat(oldFormat);
    bool useView = false;

    device::Memory *srcView = &srcMemory;
    if (oldFormat != newFormat) {
        srcView = createImageView(srcMemory, newFormat);
        useView = true;
    }

    oclhsa::Image &srcImage = static_cast<oclhsa::Image &>(*srcView);

    amd::Image * image = srcImage.owner()->asImage();
    uint blitType = 0;
    blitType = BlitCopyImageToBuffer;
    size_t  dim = 0;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    size_t  globalWorkSize[3];
    size_t  localWorkSize[3];

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
    }
    else if (imageDims == 2) {
        globalWorkSize[0] = amd::alignUp(size[0], 16);
        globalWorkSize[1] = amd::alignUp(size[1], 16);
        globalWorkSize[2] = amd::alignUp(size[2], 1);
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
    cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(srcImage.owner()));
    kernels_[blitType]->parameters().set(0, sizeof(cl_mem), &clmem);
    clmem = ((cl_mem) as_cl<amd::Memory>(dstMemory.owner()));
    kernels_[blitType]->parameters().set(1, sizeof(cl_mem), &clmem);

    // Update extra paramters for USHORT and UBYTE pointers.
    // Only then compiler can optimize the kernel to use
    // UAV Raw for other writes
    kernels_[blitType]->parameters().set(2, sizeof(cl_mem), &clmem);
    kernels_[blitType]->parameters().set(3, sizeof(cl_mem), &clmem);

    cl_int   srcOrg[4] = { (cl_int)srcOrigin[0],
                           (cl_int)srcOrigin[1],
                           (cl_int)srcOrigin[2], 0 };
    cl_int copySize[4] = { (cl_int)size[0],
                           (cl_int)size[1],
                           (cl_int)size[2], 0 };

    kernels_[blitType]->parameters().set(4, sizeof(srcOrg), srcOrg);

    const size_t elementSize =
        srcImage.owner()->asImage()->getImageFormat().getElementSize();
    const size_t numChannels =
        srcImage.owner()->asImage()->getImageFormat().getNumChannels();

    // 1 element granularity for writes by default
    cl_int  granularity = 1;
    if (elementSize == 2) {
        granularity = 2;
    }
    else if (elementSize >= 4) {
        granularity = 4;
    }
    CondLog(((dstOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
    cl_ulong    dstOrg[4] = { dstOrigin[0] / granularity,
                              dstOrigin[1],
                              dstOrigin[2],
                              0 };
    kernels_[blitType]->parameters().set(5, sizeof(dstOrg), dstOrg);
    kernels_[blitType]->parameters().set(6, sizeof(copySize), copySize);

    // Program memory format
    uint multiplier = elementSize / sizeof(uint32_t);
    multiplier = (multiplier == 0) ? 1 : multiplier;
    cl_uint format[4] = { (cl_uint)numChannels,
                          (cl_uint)(elementSize / numChannels),
                          multiplier, 0 };
    kernels_[blitType]->parameters().set(7, sizeof(format), format);

    // Program row and slice pitches
    cl_ulong    pitch[4] = { 0 };
    CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, srcImage);
    kernels_[blitType]->parameters().set(8, sizeof(pitch), pitch);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(dim,
        globalWorkOffset, globalWorkSize, localWorkSize);

    // Execute the blit
    address parameters = kernels_[blitType]->parameters().capture(dev_);
    bool result = gpu().submitKernelInternal(
        ndrange, *kernels_[blitType], parameters, NULL);
    kernels_[blitType]->parameters().release(const_cast<address>(parameters), dev_);

    if (useView) {
        srcView->owner()->release();
    }

    return result;
}

bool
KernelBlitManager::copyBufferToImage(
    device::Memory&     srcMemory,
    device::Memory&     dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    bool        entire,
    size_t      rowPitch,
    size_t      slicePitch) const
{
    if (srcMemory.isHostMemDirectAccess() && dstMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::copyBufferToImage(
            srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire,
            rowPitch, slicePitch);
    }

    amd::Image::Format oldFormat = dstMemory.owner()->asImage()->getImageFormat();
    amd::Image::Format newFormat = filterFormat(oldFormat);
    bool useView = false;

    device::Memory *dstView = &dstMemory;
    if (oldFormat != newFormat) {
        dstView = createImageView(dstMemory, newFormat);
        useView = true;
    }

    oclhsa::Image &dstImage = static_cast<oclhsa::Image &>(*dstView);

    // Use a common blit type with three dimensions by default
    uint    blitType = BlitCopyBufferToImage;
    size_t  dim = 0;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    size_t  globalWorkSize[3];
    size_t  localWorkSize[3];

    // Program the kernels workload depending on the blit dimensions
    const size_t imageDims = dstImage.owner()->asImage()->getDims();
    dim = 3;
    if (imageDims == 1) {
        globalWorkSize[0] = amd::alignUp(size[0], 256);
        globalWorkSize[1] = amd::alignUp(size[1], 1);
        globalWorkSize[2] = amd::alignUp(size[2], 1);
        localWorkSize[0] = 256;
        localWorkSize[1] = localWorkSize[2] = 1;
    }
    else if (imageDims == 2) {
        globalWorkSize[0] = amd::alignUp(size[0], 16);
        globalWorkSize[1] = amd::alignUp(size[1], 16);
        globalWorkSize[2] = amd::alignUp(size[2], 1);
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
    clmem = ((cl_mem) as_cl<amd::Memory>(dstImage.owner()));
    kernels_[blitType]->parameters().set(1, sizeof(cl_mem), &clmem);

    const size_t elementSize =
        dstImage.owner()->asImage()->getImageFormat().getElementSize();
    const size_t numChannels =
        dstImage.owner()->asImage()->getImageFormat().getNumChannels();

    // 1 element granularity for writes by default
    cl_int  granularity = 1;
    if (elementSize == 2) {
        granularity = 2;
    }
    else if (elementSize >= 4) {
        granularity = 4;
    }
    CondLog(((srcOrigin[0] % granularity) != 0), "Unaligned offset in blit!");
    cl_ulong    srcOrg[4] = { srcOrigin[0] / granularity,
                              srcOrigin[1],
                              srcOrigin[2], 0 };
    kernels_[blitType]->parameters().set(2, sizeof(srcOrg), srcOrg);

    cl_int   dstOrg[4] = { (cl_int)dstOrigin[0],
                           (cl_int)dstOrigin[1],
                           (cl_int)dstOrigin[2], 0 };
    cl_int copySize[4] = { (cl_int)size[0],
                           (cl_int)size[1],
                           (cl_int)size[2], 0 };

    kernels_[blitType]->parameters().set(3, sizeof(dstOrg), dstOrg);
    kernels_[blitType]->parameters().set(4, sizeof(copySize), copySize);

    // Program memory format
    uint multiplier = elementSize / sizeof(uint32_t);
    multiplier = (multiplier == 0) ? 1 : multiplier;
    cl_uint  format[4] = { (cl_uint)numChannels,
                           (cl_uint)(elementSize / numChannels),
                           multiplier, 0 };
    kernels_[blitType]->parameters().set(5, sizeof(format), format);

    // Program row and slice pitches
    cl_ulong    pitch[4] = { 0 };
    CalcRowSlicePitches(pitch, copySize, rowPitch, slicePitch, dstImage);
    kernels_[blitType]->parameters().set(6, sizeof(pitch), pitch);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(dim,
        globalWorkOffset, globalWorkSize, localWorkSize);

    // Execute the blit
    address parameters = kernels_[blitType]->parameters().capture(dev_);
    bool result = gpu().submitKernelInternal(
        ndrange, *kernels_[blitType], parameters, NULL);
    kernels_[blitType]->parameters().release(const_cast<address>(parameters), dev_);

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
    if (srcMemory.isHostMemDirectAccess() &&
        dstMemory.isHostMemDirectAccess()) {
        return HsaBlitManager::copyImage(
            srcMemory, dstMemory, srcOrigin, dstOrigin, size, entire);
    }

    amd::Image::Format srcOldFormat = srcMemory.owner()->asImage()->getImageFormat();
    amd::Image::Format srcNewFormat = filterFormat(srcOldFormat);
    bool useSrcView = false;

    device::Memory *srcView = &srcMemory;
    if (srcOldFormat != srcNewFormat) {
        srcView = createImageView(srcMemory, srcNewFormat);
        useSrcView = true;
    }

    oclhsa::Image &srcImage = static_cast<oclhsa::Image &>(*srcView);

    amd::Image::Format dstOldFormat = srcMemory.owner()->asImage()->getImageFormat();
    amd::Image::Format dstNewFormat = filterFormat(dstOldFormat);
    bool useDstView = false;

    device::Memory *dstView = &dstMemory;
    if (dstOldFormat != dstNewFormat) {
        dstView = createImageView(dstMemory, dstNewFormat);
        useDstView = true;
    }

    oclhsa::Image &dstImage = static_cast<oclhsa::Image &>(*dstView);

    uint    blitType = BlitCopyImage;
    size_t  dim = 0;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    size_t  globalWorkSize[3];
    size_t  localWorkSize[3];

    // Program the kernels workload depending on the blit dimensions
    dim = 3;
    // Find the current blit type
    const size_t srcDimSize = srcImage.owner()->asImage()->getDims();
    const size_t dstDimSize = dstImage.owner()->asImage()->getDims();
    if ((srcDimSize == 1) ||
        (dstDimSize == 1)) {
        globalWorkSize[0] = amd::alignUp(size[0], 256);
        globalWorkSize[1] = amd::alignUp(size[1], 1);
        globalWorkSize[2] = amd::alignUp(size[2], 1);
        localWorkSize[0] = 256;
        localWorkSize[1] = localWorkSize[2] = 1;
    }
    else if ((srcDimSize == 2) ||
             (dstDimSize == 2)) {
        globalWorkSize[0] = amd::alignUp(size[0], 16);
        globalWorkSize[1] = amd::alignUp(size[1], 16);
        globalWorkSize[2] = amd::alignUp(size[2], 1);
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

    // The current OpenCL spec allows "copy images from a 1D image
    // array object to a 1D image array object" only.
    if ((srcImage.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY) ||
        (dstImage.owner()->getType() == CL_MEM_OBJECT_IMAGE1D_ARRAY)) {
        blitType = BlitCopyImage1DA;
    }

    // Program kernels arguments for the blit operation
    cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(srcImage.owner()));
    kernels_[blitType]->parameters().set(0, sizeof(cl_mem), &clmem);
    clmem = ((cl_mem) as_cl<amd::Memory>(dstImage.owner()));
    kernels_[blitType]->parameters().set(1, sizeof(cl_mem), &clmem);

    // Program source origin
    cl_int  srcOrg[4] = { (cl_int)srcOrigin[0],
                          (cl_int)srcOrigin[1],
                          (cl_int)srcOrigin[2], 0 };

    kernels_[blitType]->parameters().set(2, sizeof(srcOrg), srcOrg);

    // Program destination origin
    cl_int  dstOrg[4] = { (cl_int)dstOrigin[0],
                          (cl_int)dstOrigin[1],
                          (cl_int)dstOrigin[2], 0 };
    kernels_[blitType]->parameters().set(3, sizeof(dstOrg), dstOrg);

    cl_int  copySize[4] = { (cl_int)size[0],
                            (cl_int)size[1],
                            (cl_int)size[2], 0 };
    kernels_[blitType]->parameters().set(4, sizeof(copySize), copySize);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(
        dim, globalWorkOffset, globalWorkSize, localWorkSize);

    // Execute the blit
    address parameters = kernels_[blitType]->parameters().capture(dev_);
    bool result = gpu().submitKernelInternal(
        ndrange, *kernels_[blitType], parameters, NULL);
    kernels_[blitType]->parameters().release(const_cast<address>(parameters), dev_);

    if (useSrcView) {
        srcView->owner()->release();
    }

    if (useDstView) {
        dstView->owner()->release();
    }

    return result;
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
        return HsaBlitManager::fillBuffer(
            memory, pattern, patternSize, origin, size, entire);
    }

    uint    fillType = FillBuffer;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    cl_ulong    fillSize = size[0] / patternSize;
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
    cl_ulong    offset = origin[0];
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
    if (memory.isHostMemDirectAccess()) {
        return HsaBlitManager::fillImage(memory, pattern, origin, size, entire);
    }

    amd::Image *image = memory.owner()->asImage();

    uint    fillType;
    size_t  dim = 0;
    size_t  globalWorkOffset[3] = { 0, 0, 0 };
    size_t  globalWorkSize[3];
    size_t  localWorkSize[3];

    // Program the kernels workload depending on the fill dimensions
    fillType = FillImage;
    dim = 3;
    // Find the current blit type
    const size_t dimSize = image->getDims();
    if (dimSize == 1) {
        globalWorkSize[0] = amd::alignUp(size[0], 256);
        globalWorkSize[1] = amd::alignUp(size[1], 1);
        globalWorkSize[2] = amd::alignUp(size[2], 1);
        localWorkSize[0] = 256;
        localWorkSize[1] = localWorkSize[2] = 1;
    }
    else if (dimSize == 2) {
        globalWorkSize[0] = amd::alignUp(size[0], 16);
        globalWorkSize[1] = amd::alignUp(size[1], 16);
        globalWorkSize[2] = amd::alignUp(size[2], 1);
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
    cl_mem clmem = ((cl_mem) as_cl<amd::Memory>(memory.owner()));
    kernels_[fillType]->parameters().set(0, sizeof(cl_mem), &clmem);
    kernels_[fillType]->parameters().set(1, sizeof(cl_float4), pattern);
    kernels_[fillType]->parameters().set(2, sizeof(cl_int4), pattern);
    kernels_[fillType]->parameters().set(3, sizeof(cl_uint4), pattern);

    cl_int fillOrigin[4] = { (cl_int)origin[0],
                             (cl_int)origin[1],
                             (cl_int)origin[2], 0 };
    cl_int   fillSize[4] = { (cl_int)size[0],
                             (cl_int)size[1],
                             (cl_int)size[2], 0 };
    kernels_[fillType]->parameters().set(4, sizeof(fillOrigin), fillOrigin);
    kernels_[fillType]->parameters().set(5, sizeof(fillSize), fillSize);

    // Find the type of image
    uint32_t    type = 0;
    amd::Image::Format format(image->getImageFormat());
    switch (format.image_channel_data_type) {
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
    kernels_[fillType]->parameters().set(6, sizeof(type), &type);

    // Create ND range object for the kernel's execution
    amd::NDRangeContainer ndrange(dim,
        globalWorkOffset, globalWorkSize, localWorkSize);

    // Execute the blit
    address parameters = kernels_[fillType]->parameters().capture(dev_);
    bool result = gpu().submitKernelInternal(
        ndrange, *kernels_[fillType], parameters, NULL);
    kernels_[fillType]->parameters().release(const_cast<address>(parameters), dev_);

    return result;
}

bool
KernelBlitManager::create(amd::Device& device)
{
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

} // namespace oclhsa
