//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/gpu/gpuconstbuf.hpp"
#include "device/gpu/gpuvirtual.hpp"
#include "device/gpu/gpudevice.hpp"
#include "device/gpu/gpusettings.hpp"

namespace gpu {

ConstBuffer::ConstBuffer(
    VirtualGPU&     gpu,
    size_t          size)
    : Resource(const_cast<gpu::Device&>(gpu.dev()), size, CM_SURF_FMT_RGBA32F)
    , gpu_(gpu)
    , size_(size * VectorSize)
    , wrtOffset_(0)
    , lastWrtSize_(0)
    , wrtAddress_(NULL)
{
}

ConstBuffer::~ConstBuffer()
{
    if (wrtAddress_ != NULL) {
        unmap(&gpu_);
    }

    amd::AlignedMemory::deallocate(sysMemCopy_);
}

bool
ConstBuffer::create()
{
    // Create sysmem copy for the constant buffer
    sysMemCopy_ = reinterpret_cast<address>(amd::AlignedMemory::allocate(size_, 256));
    if (sysMemCopy_ == NULL) {
        LogPrintfError("We couldn't allocate sysmem copy for constant buffer,\
            size(%d)!", size_);
        return false;
    }
    memset(sysMemCopy_, 0, size_);

    if (!Resource::create(Resource::RemoteUSWC)) {
        LogPrintfError("We couldn't create HW constant buffer, size(%d)!", size_);
        return false;
    }

    // Constant buffer warm-up
    warmUpRenames(gpu_);

    wrtAddress_ = map(&gpu_, Resource::Discard);
    if (wrtAddress_ == NULL) {
        LogPrintfError("We couldn't map HW constant buffer, size(%d)!", size_);
        return false;
    }

    return true;
}

bool
ConstBuffer::uploadDataToHw(size_t size)
{
    static const size_t HwCbAlignment = 256;

    // Align copy size on the vector's boundary
    size_t count = amd::alignUp(size, VectorSize);
    wrtOffset_ += lastWrtSize_;

    // Check if CB has enough space for copy
    if ((wrtOffset_ + count) > size_) {
        if (wrtAddress_ != NULL) {
            unmap(&gpu_);
        }
        wrtAddress_ = map(&gpu_, Resource::Discard);
        wrtOffset_ = 0;
        lastWrtSize_ = 0;
    }

    // Update memory with new CB data
    memcpy((reinterpret_cast<char*>(wrtAddress_) + wrtOffset_), sysMemCopy_, count);

    // Adjust the size by the HW CB buffer alignment
    lastWrtSize_ = amd::alignUp(size, HwCbAlignment);
    return true;
}

} // namespace gpu
