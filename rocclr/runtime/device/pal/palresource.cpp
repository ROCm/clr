// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "utils/flags.hpp"
#include "thread/monitor.hpp"
#include "device/pal/palresource.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palblit.hpp"
#include "device/pal/paltimestamp.hpp"
#include "thread/atomic.hpp"
#include "hsa_ext_image.h"
#ifdef _WIN32
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#endif // _WIN32
#include <GL/gl.h>
#include "GL/glATIInternal.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace pal {

GpuMemoryReference*
GpuMemoryReference::Create(
    const Device&                 dev,
    const Pal::GpuMemoryCreateInfo& createInfo)
{
    Pal::Result result;
    size_t gpuMemSize = dev.iDev()->GetGpuMemorySize(createInfo, &result);
    if (result != Pal::Result::Success) {
        return nullptr;
    }

    GpuMemoryReference*  memRef = new (gpuMemSize) GpuMemoryReference();
    if (memRef != nullptr) {
        result = dev.iDev()->CreateGpuMemory(createInfo, &memRef[1], &memRef->gpuMem_);
        if (result != Pal::Result::Success) {
            memRef->release();
            return nullptr;
        }
    }
    // Update free memory size counters
    const_cast<Device&>(dev).updateFreeMemory(
        createInfo.heaps[0], createInfo.size, false);
    return memRef;
}

GpuMemoryReference*
GpuMemoryReference::Create(
    const Device&   dev,
    const Pal::PinnedGpuMemoryCreateInfo& createInfo)
{
    Pal::Result result;
    size_t gpuMemSize = dev.iDev()->GetPinnedGpuMemorySize(createInfo, &result);
    if (result != Pal::Result::Success) {
        return nullptr;
    }

    GpuMemoryReference*  memRef = new (gpuMemSize) GpuMemoryReference();
    Pal::VaRange vaRange = Pal::VaRange::Default;
    if (memRef != nullptr) {
        result = dev.iDev()->CreatePinnedGpuMemory(createInfo,
            &memRef[1], &memRef->gpuMem_);
        if (result != Pal::Result::Success) {
            memRef->release();
            return nullptr;
        }
    }
    // Update free memory size counters
    const_cast<Device&>(dev).updateFreeMemory(
        Pal::GpuHeap::GpuHeapGartCacheable, createInfo.size, false);
    return memRef;
}

GpuMemoryReference*
GpuMemoryReference::Create(
    const Device&   dev,
    const Pal::ExternalResourceOpenInfo& openInfo)
{
    Pal::Result result;
    size_t gpuMemSize = dev.iDev()->GetExternalSharedGpuMemorySize(&result);
    if (result != Pal::Result::Success) {
        return nullptr;
    }

    Pal::GpuMemoryCreateInfo    createInfo = {};
    GpuMemoryReference*  memRef = new (gpuMemSize) GpuMemoryReference();
    if (memRef != nullptr) {
        result = dev.iDev()->OpenExternalSharedGpuMemory(
            openInfo, &memRef[1], &createInfo, &memRef->gpuMem_);
        if (result != Pal::Result::Success) {
            memRef->release();
            return nullptr;
        }
    }

    return memRef;
}

GpuMemoryReference*
GpuMemoryReference::Create(
    const Device&   dev,
    const Pal::ExternalImageOpenInfo& openInfo,
    Pal::ImageCreateInfo* imgCreateInfo,
    Pal::IImage**   image)
{
    Pal::Result result;
    size_t gpuMemSize = 0;
    size_t imageSize = 0;
    if (Pal::Result::Success != dev.iDev()->GetExternalSharedImageSizes(
        openInfo, &imageSize, &gpuMemSize, imgCreateInfo)) {
        return nullptr;
    }

    Pal::GpuMemoryCreateInfo    createInfo = {};
    GpuMemoryReference*  memRef = new (gpuMemSize) GpuMemoryReference();
    char* imgMem = new char [imageSize];
    if (memRef != nullptr) {
        result = dev.iDev()->OpenExternalSharedImage(
            openInfo, imgMem, &memRef[1], &createInfo, image, &memRef->gpuMem_);
        if (result != Pal::Result::Success) {
            memRef->release();
            return nullptr;
        }
    }

    return memRef;
}

GpuMemoryReference::GpuMemoryReference()
    : gpuMem_(nullptr)
    , cpuAddress_(nullptr)
{
}

GpuMemoryReference::~GpuMemoryReference()
{
    if (cpuAddress_ != nullptr) {
        iMem()->Unmap();
    }
    if (0 != iMem()) {
        iMem()->Destroy();
        gpuMem_ = nullptr;
    }
}

Resource::Resource(
    const Device&   gpuDev,
    size_t          size)
    : elementSize_(0)
    , gpuDevice_(gpuDev)
    , mapCount_(0)
    , address_(nullptr)
    , offset_(0)
    , curRename_(0)
    , memRef_(nullptr)
    , viewOwner_(nullptr)
    , pinOffset_(0)
    , gpu_(nullptr)
    , image_(nullptr)
    , hwSrd_(0)
{
    // Fill resource descriptor fields
    desc_.state_     = 0;
    desc_.type_      = Empty;
    desc_.width_     = amd::alignUp(size,
        Pal::Formats::BytesPerPixel(Pal::ChNumFormat::X32_Uint)) /
        Pal::Formats::BytesPerPixel(Pal::ChNumFormat::X32_Uint);
    desc_.height_    = 1;
    desc_.depth_     = 1;
    desc_.mipLevels_ = 1;
    desc_.format_.image_channel_order = CL_R;
    desc_.format_.image_channel_data_type = CL_FLOAT;
    desc_.flags_     = 0;
    desc_.pitch_     = 0;
    desc_.slice_     = 0;
    desc_.cardMemory_ = true;
    desc_.dimSize_   = 1;
    desc_.buffer_    = true;
    desc_.imageArray_ = false;
    desc_.topology_  = CL_MEM_OBJECT_BUFFER;
    desc_.SVMRes_    = false;
    desc_.scratch_   = false;
    desc_.isAllocExecute_ = false;
    desc_.baseLevel_ = 0;
}

Resource::Resource(
    const Device&   gpuDev,
    size_t          width,
    size_t          height,
    size_t          depth,
    cl_image_format format,
    cl_mem_object_type  imageType,
    uint            mipLevels)
    : elementSize_(0)
    , gpuDevice_(gpuDev)
    , mapCount_(0)
    , address_(nullptr)
    , offset_(0)
    , curRename_(0)
    , memRef_(nullptr)
    , viewOwner_(nullptr)
    , pinOffset_(0)
    , gpu_(nullptr)
    , image_(nullptr)
    , hwSrd_(0)
{
    // Fill resource descriptor fields
    desc_.state_     = 0;
    desc_.type_      = Empty;
    desc_.width_     = width;
    desc_.height_    = height;
    desc_.depth_     = depth;
    desc_.mipLevels_ = mipLevels;
    desc_.format_    = format;
    desc_.flags_     = 0;
    desc_.pitch_     = 0;
    desc_.slice_     = 0;
    desc_.cardMemory_ = true;
    desc_.buffer_     = false;
    desc_.imageArray_ = false;
    desc_.topology_  = imageType;
    desc_.SVMRes_ = false;
    desc_.scratch_ = false;
    desc_.isAllocExecute_ = false;
    desc_.baseLevel_ = 0;

    switch (imageType) {
    case CL_MEM_OBJECT_IMAGE2D:
        desc_.dimSize_   = 2;
        break;
    case CL_MEM_OBJECT_IMAGE3D:
        desc_.dimSize_   = 3;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        desc_.dimSize_   = 3;
        desc_.imageArray_ = true;
        break;
    case CL_MEM_OBJECT_IMAGE1D:
        desc_.dimSize_   = 1;
        break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        desc_.dimSize_   = 2;
        desc_.imageArray_ = true;
        break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        desc_.dimSize_   = 1;
        break;
    default:
        desc_.dimSize_   = 1;
        LogError("Unknown image type!");
        break;
    }
}

Resource::~Resource()
{
    Pal::GpuHeap heap = Pal::GpuHeapCount;
    switch (memoryType()) {
    case Persistent:
        heap = Pal::GpuHeapLocal;
        break;
    case RemoteUSWC:
        heap = Pal::GpuHeapGartUswc;
        break;
    case Pinned:
    case Remote:
        heap = Pal::GpuHeapGartCacheable;
        break;
    case Shader:
    case BusAddressable:
    case ExternalPhysical:
        // Fall through to process the memory allocation ...
    case Local:
        heap = Pal::GpuHeapInvisible;
        break;
    default:
        heap = Pal::GpuHeapLocal;
        break;
    }
    if ((memRef_ != nullptr) && (heap != Pal::GpuHeapCount)) {
        // Update free memory size counters
        const_cast<Device&>(dev()).updateFreeMemory(
            heap, iMem()->Desc().size, true);
    }

    free();

    if ((nullptr != image_) && ((memoryType() != ImageView) ||
        //! @todo PAL doesn't allow an SRD view creation with different pixel size
        (elementSize() != viewOwner_->elementSize()))) {
        image_->Destroy();
        delete [] reinterpret_cast<char*>(image_);
    }
}

static uint32_t GetHSAILImageFormatType(const cl_image_format& format)
{
    static const uint32_t  FormatType[] = {
        HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32,
        HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT,
        HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 };

    uint    idx = format.image_channel_data_type - CL_SNORM_INT8;
    assert((idx <= (CL_UNORM_INT24 - CL_SNORM_INT8)) && "Out of range format channel!");
    return FormatType[idx];
}

static uint32_t GetHSAILImageOrderType(const cl_image_format& format)
{
    static const uint32_t  OrderType[] = {
        HSA_EXT_IMAGE_CHANNEL_ORDER_R,
        HSA_EXT_IMAGE_CHANNEL_ORDER_A,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RG,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RA,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGB,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA,
        HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA,
        HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB,
        HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY,
        HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RX,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGX,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX,
        HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH,
        HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA,
        HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR };

    uint    idx = format.image_channel_order - CL_R;
    assert((idx <= (CL_ABGR - CL_R)) && "Out of range format order!");
    return OrderType[idx];
}

void
Resource::memTypeToHeap(Pal::GpuMemoryCreateInfo* createInfo)
{
    createInfo->heapCount = 1;
    switch (memoryType()) {
    case Persistent:
        createInfo->heaps[0] = Pal::GpuHeapLocal;
        break;
    case RemoteUSWC:
        createInfo->heaps[0] = Pal::GpuHeapGartUswc;
        desc_.cardMemory_ = false;
        break;
    case Remote:
        createInfo->heaps[0] = Pal::GpuHeapGartCacheable;
        desc_.cardMemory_ = false;
        break;
    case Shader:
    case BusAddressable:
    case ExternalPhysical:
        // Fall through to process the memory allocation ...
    case Local:
        createInfo->heapCount = 2;
        createInfo->heaps[0] = Pal::GpuHeapInvisible;
        createInfo->heaps[1] = Pal::GpuHeapLocal;
        break;
    default:
        createInfo->heaps[0] = Pal::GpuHeapLocal;
        break;    
    }
}

bool
Resource::create(MemoryType memType, CreateParams* params)
{
    static const Pal::gpusize MaxGpuAlignment = 64 * Ki;
    const   amd::HostMemoryReference* hostMemRef = nullptr;
    bool    imageCreateView = false;
    uint    hostMemOffset = 0;
    bool    foundCalRef = false;
    bool    viewDefined = false;
    uint    viewLayer = 0;
    uint    viewLevel = 0;
    uint    viewFlags = 0;
    Pal::SubresId    ImgSubresId = { Pal::ImageAspect::Color, 0, 0 };
    Pal::SubresRange ImgSubresRange = { ImgSubresId, 1, 1 };
    Pal::ChannelMapping channels;
    Pal::ChNumFormat format = dev().getPalFormat(desc().format_, &channels);

    // This is a thread safe operation
    const_cast<Device&>(dev()).initializeHeapResources();

    amd::ScopedLock lk(dev().lockPAL());

    if (memType == Shader) {
        // force to use remote memory for HW DEBUG or use
        // local memory once we determine if FGS is supported
        // memType = (!dev().settings().enableHwDebug_) ? Local : RemoteUSWC;
        memType = RemoteUSWC;
    }

    // Get the element size
    elementSize_ = Pal::Formats::BytesPerPixel(format);
    desc_.type_ = memType;
    if (memType == Scratch) {
        // use local memory for scratch buffer unless it is using HW DEBUG
        desc_.type_ = (!dev().settings().enableHwDebug_) ? Local : RemoteUSWC;
        desc_.scratch_ = true;
    }

    // Force remote allocation if it was requested in the settings
    if (dev().settings().remoteAlloc_ &&
        ((memoryType() == Local) ||
         (memoryType() == Persistent))) {
        if (dev().settings().apuSystem_ && dev().settings().viPlus_) {
            desc_.type_ = Remote;
        }
        else {
            desc_.type_ = RemoteUSWC;
        }
    }

    if (dev().settings().disablePersistent_ && (memoryType() == Persistent)) {
        desc_.type_ = RemoteUSWC;
    }

    if (params != nullptr) {
        gpu_ = params->gpu_;
    }

    Pal::Result result;

#ifdef _WIN32
    if ((memoryType() == OGLInterop) ||
        (memoryType() == D3D9Interop) ||
        (memoryType() == D3D10Interop) ||
        (memoryType() == D3D11Interop)) {
        Pal::ExternalResourceOpenInfo openInfo = {};
        uint misc = 0;
        uint layer = 0;
        uint mipLevel = 0;
        InteropType type = InteropTypeless;

        if (memoryType() == OGLInterop) {
            OGLInteropParams* oglRes = reinterpret_cast<OGLInteropParams*>(params);
            assert(oglRes->glPlatformContext_ && "We don't have OGL context!");
            switch (oglRes->type_) {
            case InteropVertexBuffer:
                glType_ = GL_RESOURCE_ATTACH_VERTEXBUFFER_AMD;
                break;
            case InteropRenderBuffer:
                glType_ = GL_RESOURCE_ATTACH_RENDERBUFFER_AMD;
                break;
            case InteropTexture:
            case InteropTextureViewLevel:
            case InteropTextureViewCube:
                glType_ = GL_RESOURCE_ATTACH_TEXTURE_AMD;
                break;
            default:
                LogError("Unknown OGL interop type!");
                return false;
                break;
            }
            glPlatformContext_ = oglRes->glPlatformContext_;
            glDeviceContext_ = oglRes->glDeviceContext_;
            layer = oglRes->layer_;
            type = oglRes->type_;
            mipLevel = oglRes->mipLevel_;

            if (!dev().resGLAssociate(oglRes->glPlatformContext_, oglRes->handle_,
                    glType_, &openInfo.hExternalResource, &glInteropMbRes_, &offset_)) {
                return false;
            }
        }
        else {
            D3DInteropParams* d3dRes = reinterpret_cast<D3DInteropParams*>(params);
            openInfo.hExternalResource = d3dRes->handle_;
            misc = d3dRes->misc;
            layer = d3dRes->layer_;
            type = d3dRes->type_;
            mipLevel = d3dRes->mipLevel_;
        }
        //! @todo PAL query for image/buffer object doesn't work properly!
#if 0
        bool    isImage = false;
        if (Pal::Result::Success != 
            dev().iDev()->DetermineExternalSharedResourceType(openInfo, &isImage)) {
            return false;
        }
#endif // 0
        if (desc().buffer_ || misc) {
            memRef_ = GpuMemoryReference::Create(dev(), openInfo);
            if (nullptr == memRef_) {
                return false;
            }

            if (misc) {
                Pal::ImageCreateInfo    imgCreateInfo = {};
                Pal::ExternalImageOpenInfo imgOpenInfo = {};
                imgOpenInfo.resourceInfo = openInfo;
                imgOpenInfo.swizzledFormat.format = format;
                imgOpenInfo.swizzledFormat.swizzle = channels;
                imgOpenInfo.flags.formatChangeSrd = true;
                imgOpenInfo.usage.shaderRead = true;
                imgOpenInfo.usage.shaderWrite = true;
                size_t imageSize;
                size_t gpuMemSize;

                if (Pal::Result::Success != dev().iDev()->GetExternalSharedImageSizes(
                    imgOpenInfo, &imageSize, &gpuMemSize, &imgCreateInfo)) {
                    return false;
                }

                Pal::gpusize    viewOffset = 0;
                imgCreateInfo.flags.shareable = false;
                imgCreateInfo.imageType = Pal::ImageType::Tex2d;
                imgCreateInfo.extent.width  = desc().width_;
                imgCreateInfo.extent.height = desc().height_;
                imgCreateInfo.extent.depth  = desc().depth_;
                imgCreateInfo.arraySize     = 1;
                imgCreateInfo.flags.formatChangeSrd = true;
                imgCreateInfo.usageFlags.shaderRead = true;
                imgCreateInfo.usageFlags.shaderWrite = true;
                imgCreateInfo.swizzledFormat.format  = format;
                imgCreateInfo.swizzledFormat.swizzle = channels;
                imgCreateInfo.mipLevels = 1;
                imgCreateInfo.samples   = 1;
                imgCreateInfo.fragments = 1;
                imgCreateInfo.tiling    = Pal::ImageTiling::Linear;

                switch (misc) {
                case 1:     // NV12 format
                    switch (layer) {
                    case -1:
                        break;
                    case 0:
                        break;
                    case 1:
                        // Y - plane size to the offset
                        // NV12 format. UV is 2 times smaller plane Y
                        viewOffset = 2 * imgCreateInfo.rowPitch * desc().height_;
                        imgCreateInfo.depthPitch = imgCreateInfo.rowPitch * desc().height_;
                        break;
                    default:
                        LogError("Unknown Interop View Type");
                        return false;
                    }
                    break;
                case 2:     // YV12 format
                    switch (layer) {
                    case -1:
                        break;
                    case 0:
                        break;
                    case 1:
                        // Y - plane size to the offset
                        // YV12 format. U is 4 times smaller plane than Y
                        viewOffset = 2 * imgCreateInfo.rowPitch * desc().height_;
                        imgCreateInfo.rowPitch >>= 1;
                        break;
                    case 2:
                        // Y + U plane sizes to the offest.
                        // U plane is 4 times smaller than Y and U == V
                        viewOffset = 5 * imgCreateInfo.rowPitch *  desc().height_ / 2;
                        imgCreateInfo.rowPitch >>= 1;
                        break;
                    default:
                        LogError("Unknown Interop View Type");
                        return false;
                    }
                    imgCreateInfo.depthPitch = imgCreateInfo.rowPitch * desc().height_;
                    break;
                default:
                    LogError("Unknown Interop View Type");
                    return false;
                }

                imageSize = dev().iDev()->GetImageSize(imgCreateInfo, &result);
                if (result != Pal::Result::Success) {
                    return false;
                }

                char* memImg = new char[imageSize];
                if (memImg != nullptr) {
                    result = dev().iDev()->CreateImage(imgCreateInfo, memImg, &image_);
                    if (result != Pal::Result::Success) {
                        delete memImg;
                        return false;
                    }
                }
                result = image_->BindGpuMemory(iMem(), viewOffset);
                offset_ = static_cast<size_t>(viewOffset);
                hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
                if ((0 == hwSrd_) && (memoryType() != ImageView)) {
                    return false;
                }
                Pal::ImageViewInfo viewInfo = {};
                viewInfo.viewType = Pal::ImageViewType::Tex2d;
                viewInfo.pImage = image_;
                viewInfo.swizzledFormat.format = format;
                viewInfo.swizzledFormat.swizzle = channels;
                viewInfo.subresRange = ImgSubresRange;
                dev().iDev()->CreateImageViewSrds(1, &viewInfo, hwState_);

                hwState_[8] = GetHSAILImageFormatType(desc().format_);
                hwState_[9] = GetHSAILImageOrderType(desc().format_);
                hwState_[10] = static_cast<uint32_t>(desc().width_);
                hwState_[11] = 0;   // one extra reserved field in the argument
            }
        }
        else if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
            memRef_ = GpuMemoryReference::Create(dev(), openInfo);
            if (nullptr == memRef_) {
                return false;
            }
            Pal::BufferViewInfo viewInfo = {};
            viewInfo.gpuAddr = memRef_->iMem()->Desc().gpuVirtAddr + offset();
            viewInfo.range = memRef_->iMem()->Desc().size;
            viewInfo.stride = elementSize();
            viewInfo.swizzledFormat.format = format;
            viewInfo.swizzledFormat.swizzle = channels;
            hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
            if ((0 == hwSrd_) && (memoryType() != ImageView)) {
                return false;
            }

            dev().iDev()->CreateTypedBufferViewSrds(1, &viewInfo, hwState_);
            hwState_[8] = GetHSAILImageFormatType(desc().format_);
            hwState_[9] = GetHSAILImageOrderType(desc().format_);
            hwState_[10] = static_cast<uint32_t>(desc().width_);
            hwState_[11] = 0;   // one extra reserved field in the argument
        }
        else {
            Pal::ExternalImageOpenInfo imgOpenInfo = {};
            Pal::ImageCreateInfo    imgCreateInfo = {};
            imgOpenInfo.resourceInfo = openInfo;
            imgOpenInfo.swizzledFormat.format = format;
            imgOpenInfo.swizzledFormat.swizzle = channels;
            imgOpenInfo.flags.formatChangeSrd = true;
            imgOpenInfo.usage.shaderRead = true;
            imgOpenInfo.usage.shaderWrite = true;
            memRef_ = GpuMemoryReference::Create(
                dev(), imgOpenInfo, &imgCreateInfo, &image_);
            if (nullptr == memRef_) {
                return false;
            }

            hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
            if ((0 == hwSrd_) && (memoryType() != ImageView)) {
                return false;
            }
            Pal::ImageViewInfo viewInfo = {};
            viewInfo.viewType = Pal::ImageViewType::Tex2d;
            switch (imgCreateInfo.imageType) {
            case Pal::ImageType::Tex3d:
                viewInfo.viewType = Pal::ImageViewType::Tex3d;
                break;
            case Pal::ImageType::Tex1d:
                viewInfo.viewType = Pal::ImageViewType::Tex1d;
                break;
            }
            viewInfo.pImage = image_;
            viewInfo.swizzledFormat.format = format;
            viewInfo.swizzledFormat.swizzle = channels;
            if ((type == InteropTextureViewLevel) ||
                (type == InteropTextureViewCube)) {
                ImgSubresRange.startSubres.mipLevel = mipLevel;
                if (type == InteropTextureViewCube) {
                    ImgSubresRange.startSubres.arraySlice = layer;
                    viewInfo.viewType = Pal::ImageViewType::Tex2d;
                }
            }
            if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
                ImgSubresRange.numSlices = desc_.height_;
            }
            if (desc().topology_ == CL_MEM_OBJECT_IMAGE2D_ARRAY) {
                ImgSubresRange.numSlices = desc_.depth_;
            }
            ImgSubresRange.numMips = desc().mipLevels_;
            viewInfo.subresRange = ImgSubresRange;

            dev().iDev()->CreateImageViewSrds(1, &viewInfo, hwState_);

            hwState_[8] = GetHSAILImageFormatType(desc().format_);
            hwState_[9] = GetHSAILImageOrderType(desc().format_);
            hwState_[10] = static_cast<uint32_t>(desc().width_);
            hwState_[11] = 0;   // one extra reserved field in the argument
        }
        return true;
    }
#endif // _WIN32

    if (!desc_.buffer_) {
        if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
            if (memoryType() == ImageBuffer) {
                ImageBufferParams* imageBuffer = reinterpret_cast<ImageBufferParams*>(params);
                viewOwner_ = imageBuffer->resource_;
                memRef_ = viewOwner_->memRef_;
                memRef_->retain();
                desc_.cardMemory_ = viewOwner_->desc().cardMemory_;
            }
            else {
                Pal::GpuMemoryCreateInfo createInfo = {};
                createInfo.size = desc().width_ * elementSize();
                // @todo 64K alignment is too big
                createInfo.size = amd::alignUp(createInfo.size, MaxGpuAlignment);
                createInfo.alignment = MaxGpuAlignment;
                createInfo.vaRange = Pal::VaRange::Default;
                createInfo.priority  = Pal::GpuMemPriority::Normal;
                memTypeToHeap(&createInfo);
                // createInfo.priority;
                memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment);
                if (nullptr == memRef_) {
                    memRef_ = GpuMemoryReference::Create(dev(), createInfo);
                    if (nullptr == memRef_) {
                        LogError("Failed PAL memory allocation!");
                        return false;
                    }
                }
            }
            Pal::BufferViewInfo viewInfo = {};
            viewInfo.gpuAddr = memRef_->iMem()->Desc().gpuVirtAddr + offset();
            viewInfo.range = memRef_->iMem()->Desc().size;
            viewInfo.stride = elementSize();
            viewInfo.swizzledFormat.format = format;
            viewInfo.swizzledFormat.swizzle = channels;
            //viewInfo.channels = channels;
            hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
            if ((0 == hwSrd_) && (memoryType() != ImageView)) {
                return false;
            }

            dev().iDev()->CreateTypedBufferViewSrds(1, &viewInfo, hwState_);
            hwState_[8] = GetHSAILImageFormatType(desc().format_);
            hwState_[9] = GetHSAILImageOrderType(desc().format_);
            hwState_[10] = static_cast<uint32_t>(desc().width_);
            hwState_[11] = 0;   // one extra reserved field in the argument
            return true;
        }

        Pal::ImageViewInfo viewInfo = {};
        Pal::ImageCreateInfo    imgCreateInfo = {};
        Pal::GpuMemoryRequirements req = {};
        char* memImg;
        imgCreateInfo.imageType = Pal::ImageType::Tex2d;
        viewInfo.viewType = Pal::ImageViewType::Tex2d;
        imgCreateInfo.extent.width  = desc_.width_;
        imgCreateInfo.extent.height = desc_.height_;
        imgCreateInfo.extent.depth  = desc_.depth_;
        imgCreateInfo.arraySize     = 1;

        switch (desc_.topology_) {
        case CL_MEM_OBJECT_IMAGE3D:
            imgCreateInfo.imageType = Pal::ImageType::Tex3d;
            viewInfo.viewType = Pal::ImageViewType::Tex3d;
            break;
        case CL_MEM_OBJECT_IMAGE1D:
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            imgCreateInfo.imageType = Pal::ImageType::Tex1d;
            viewInfo.viewType = Pal::ImageViewType::Tex1d;
            break;
        }
        if (desc_.topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
            ImgSubresRange.numSlices = imgCreateInfo.arraySize = desc_.height_;
            imgCreateInfo.extent.depth = desc_.height_;
            imgCreateInfo.extent.height = 1;
        }
        if (desc_.topology_ == CL_MEM_OBJECT_IMAGE2D_ARRAY) {
            ImgSubresRange.numSlices = imgCreateInfo.arraySize = desc_.depth_;
        }

        if (memoryType() == ImageView) {
            ImageViewParams* imageView = reinterpret_cast<ImageViewParams*>(params);
            ImgSubresRange.startSubres.mipLevel = imageView->level_;
            desc_.baseLevel_ = imageView->level_;
            ImgSubresRange.startSubres.arraySlice = imageView->layer_;
            viewOwner_  = imageView->resource_;
            image_ = viewOwner_->image_;
            offset_ = viewOwner_->offset_;
        }
        else if (memoryType() == ImageBuffer) {
            ImageBufferParams* imageBuffer = reinterpret_cast<ImageBufferParams*>(params);
            viewOwner_  = imageBuffer->resource_;
        }
        ImgSubresRange.numMips = desc().mipLevels_;

        if ((memoryType() != ImageView) ||
            //! @todo PAL doesn't allow an SRD view creation with different pixel size
            (elementSize() != viewOwner_->elementSize())) {
            imgCreateInfo.flags.formatChangeSrd = true;
            imgCreateInfo.usageFlags.shaderRead = true;
            imgCreateInfo.usageFlags.shaderWrite =
                (format == Pal::ChNumFormat::X8Y8Z8W8_Srgb) ? false : true;
            imgCreateInfo.swizzledFormat.format = format;
            imgCreateInfo.swizzledFormat.swizzle = channels;
            imgCreateInfo.mipLevels     = (desc_.mipLevels_) ? desc_.mipLevels_ : 1;
            imgCreateInfo.samples   = 1;
            imgCreateInfo.fragments = 1;
            Pal::ImageTiling    tiling =  Pal::ImageTiling::Optimal;

            if (((memoryType() == Persistent) && 
                 dev().settings().linearPersistentImage_) ||
                (memoryType() == ImageBuffer)) {
                tiling    = Pal::ImageTiling::Linear;
            }
            else if (memoryType() == ImageView) {
                tiling = viewOwner_->image_->GetImageCreateInfo().tiling;
            }

            if (memoryType() == ImageBuffer) {
                uint32_t    rowPitch;
                if ((params->owner_ != NULL) && params->owner_->asImage() &&
                    (params->owner_->asImage()->getRowPitch() != 0)) {
                    rowPitch = params->owner_->asImage()->getRowPitch() / elementSize();
                }
                else {
                    rowPitch = desc().width_;
                }
                // Make sure the row pitch is aligned to pixels
                imgCreateInfo.rowPitch = elementSize() *
                    amd::alignUp(rowPitch, dev().info().imagePitchAlignment_);
                imgCreateInfo.depthPitch = imgCreateInfo.rowPitch * desc().height_;
            }
            imgCreateInfo.tiling    = tiling;

            size_t imageSize = dev().iDev()->GetImageSize(imgCreateInfo, &result);
            if (result != Pal::Result::Success) {
                return false;
            }

            memImg = new char[imageSize];
            if (memImg != nullptr) {
                result = dev().iDev()->CreateImage(imgCreateInfo, memImg, &image_);
                if (result != Pal::Result::Success) {
                    delete memImg;
                    return false;
                }
            }
            image_->GetGpuMemoryRequirements(&req);
            // createInfo.priority;
        }

        if ((memoryType() != ImageView) && (memoryType() != ImageBuffer)) {
            Pal::GpuMemoryCreateInfo createInfo = {};
            createInfo.size =  amd::alignUp(req.size, MaxGpuAlignment);
            createInfo.alignment = std::max(req.alignment, MaxGpuAlignment);
            createInfo.vaRange = Pal::VaRange::Default;
            createInfo.priority  = Pal::GpuMemPriority::Normal;
            memTypeToHeap(&createInfo);

            memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment);
            if (nullptr == memRef_) {
                memRef_ = GpuMemoryReference::Create(dev(), createInfo);
                if (nullptr == memRef_) {
                    LogError("Failed PAL memory allocation!");
                    return false;
                }
            }
        }
        else {
            memRef_ = viewOwner_->memRef_;
            memRef_->retain();
            desc_.cardMemory_ = viewOwner_->desc().cardMemory_;
            if (req.size > viewOwner_->iMem()->Desc().size) {
                LogWarning("Image is bigger than the original mem object!");
            }
        }

        result = image_->BindGpuMemory(memRef_->gpuMem_, offset_);
        if (result != Pal::Result::Success) {
            return false;
        }

        hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
        if ((0 == hwSrd_) && (memoryType() != ImageView)) {
            return false;
        }
        viewInfo.pImage = image_;
        viewInfo.swizzledFormat.format = format;
        viewInfo.swizzledFormat.swizzle = channels;
        viewInfo.subresRange = ImgSubresRange;
        dev().iDev()->CreateImageViewSrds(1, &viewInfo, hwState_);

        hwState_[8] = GetHSAILImageFormatType(desc().format_);
        hwState_[9] = GetHSAILImageOrderType(desc().format_);
        hwState_[10] = static_cast<uint32_t>(desc().width_);
        hwState_[11] = 0;   // one extra reserved field in the argument
        return true;
    }

    if (memoryType() == View) {
        // Save the offset in the global heap
        ViewParams* view = reinterpret_cast<ViewParams*>(params);
        offset_ = view->offset_;

        // Make sure parent was provided
        if (nullptr != view->resource_) {
            viewOwner_ = view->resource_;
            offset_ += viewOwner_->offset();

            if (viewOwner_->isMemoryType(Pinned)) {
                address_ = viewOwner_->data() + view->offset_;
            }
            pinOffset_ = viewOwner_->pinOffset();
            memRef_ = viewOwner_->memRef_;
            memRef_->retain();
            desc_.cardMemory_ = viewOwner_->desc().cardMemory_;
        }
        else {
            desc_.type_ = Empty;
        }
        return true;
    }

    if (memoryType() == Pinned) {
        PinnedParams*   pinned = reinterpret_cast<PinnedParams*>(params);
        uint        allocSize = static_cast<uint>(pinned->size_);
        void*       pinAddress;
        hostMemRef  = pinned->hostMemRef_;
        pinAddress  = address_ = hostMemRef->hostMem();
        // assert((allocSize == (desc().width_ * elementSize())) && "Sizes don't match");
        if (desc().topology_ == CL_MEM_OBJECT_BUFFER) {
            // Allign offset to 4K boundary (Vista/Win7 limitation)
            char* tmpHost = const_cast<char*>(
                amd::alignDown(reinterpret_cast<const char*>(address_),
                PinnedMemoryAlignment));

            // Find the partial size for unaligned copy
            hostMemOffset = static_cast<uint>(
                reinterpret_cast<const char*>(address_) - tmpHost);

            pinOffset_ = hostMemOffset;

            pinAddress = tmpHost;

            if (hostMemOffset != 0) {
                allocSize += hostMemOffset;
            }
            allocSize = amd::alignUp(allocSize, PinnedMemoryAlignment);
//            hostMemOffset &= ~(0xff);
        }
        else if (desc().topology_ == CL_MEM_OBJECT_IMAGE2D) {
            //! @todo: Width has to be aligned for 3D.
            //! Need to be replaced with a compute copy
            // Width aligned by 8 texels
            if (((desc().width_ % 0x8) != 0) ||
                // Pitch aligned by 64 bytes
                (((desc().width_ * elementSize()) % 0x40) != 0)) {
                return false;
            }
        }
        else {
            //! @todo GSL doesn't support pinning with resAlloc_
            return false;
        }

        // Ensure page alignment
        if ((uint64_t)(pinAddress) & (amd::Os::pageSize() - 1)) {
            return false;
        }
        Pal::PinnedGpuMemoryCreateInfo createInfo = {};
        createInfo.pSysMem = pinAddress;
        createInfo.size = allocSize;
        createInfo.vaRange = Pal::VaRange::Default;
        memRef_ = GpuMemoryReference::Create(dev(), createInfo);
        if (nullptr == memRef_) {
            LogError("Failed PAL memory allocation!");
            pinOffset_ = 0;
            return false;
        }
        desc_.cardMemory_ = false;
        return true;
    }

    Pal::GpuMemoryCreateInfo createInfo = {};
    createInfo.size = desc().width_ * elementSize_;
    // @todo 64K alignment is too big
    createInfo.size = amd::alignUp(createInfo.size, MaxGpuAlignment);
    createInfo.alignment = MaxGpuAlignment;
    createInfo.vaRange = Pal::VaRange::Default;
    createInfo.priority  = Pal::GpuMemPriority::Normal;
    memTypeToHeap(&createInfo);
    // createInfo.priority;
    memRef_ = dev().resourceCache().findGpuMemory(&desc_, createInfo.size, createInfo.alignment);
    if (nullptr == memRef_) {
        memRef_ = GpuMemoryReference::Create(dev(), createInfo);
        if (nullptr == memRef_) {
            LogError("Failed PAL memory allocation!");
            return false;
        }
    }

    return true;
}

void
Resource::free()
{
    if (memRef_ == nullptr) {
        return;
    }

    // Sanity check for the map calls
    if (mapCount_ != 0) {
        LogWarning("Resource wasn't unlocked, but destroyed!");
    }
    const bool wait = (memoryType() != ImageView) &&
                      (memoryType() != ImageBuffer) &&
                      (memoryType() != View);

    // Check if resource could be used in any queue(thread)
    if (gpu_ == nullptr) {
        Device::ScopedLockVgpus lock(dev());

        if (renames_.size() == 0) {
            // Destroy GSL resource
            if (iMem() != 0) {
                // Release all virtual memory objects on all virtual GPUs
                for (uint idx = 0; idx < dev().vgpus().size(); ++idx) {
                    // Ignore the transfer queue,
                    // since it releases resources after every operation
                    if (dev().vgpus()[idx] != dev().xferQueue()) {
                        dev().vgpus()[idx]->releaseMemory(iMem(), wait);
                    }
                }

                //! @note: This is a workaround for bad applications that
                //! don't unmap memory
                if (mapCount_ != 0) {
                    unmap(nullptr);
                }

                // Add resource to the cache
                if (!dev().resourceCache().addGpuMemory(&desc_, memRef_)) {
                    palFree();
                }
            }
        }
        else {
            renames_[curRename_]->cpuAddress_ = 0;
            for (size_t i = 0; i < renames_.size(); ++i) {
                memRef_ = renames_[i];
                // Destroy GSL resource
                if (iMem() != 0) {
                    // Release all virtual memory objects on all virtual GPUs
                    for (uint idx = 0; idx < dev().vgpus().size(); ++idx) {
                        // Ignore the transfer queue,
                        // since it releases resources after every operation
                        if (dev().vgpus()[idx] != dev().xferQueue()) {
                            dev().vgpus()[idx]->releaseMemory(iMem());
                        }
                    }
                    palFree();
                }
            }
        }
    }
    else {
        if (renames_.size() == 0) {
            // Destroy GSL resource
            if (iMem() != 0) {
                // Release virtual memory object on the specified virtual GPU
                gpu_->releaseMemory(iMem(), wait);
                palFree();
            }
        }
        else for (size_t i = 0; i < renames_.size(); ++i) {
            memRef_ = renames_[i];
            // Destroy GSL resource
            if (iMem() != 0) {
                // Release virtual memory object on the specified virtual GPUs
                gpu_->releaseMemory(iMem());
                palFree();
            }
        }
    }

    // Free SRD for images
    if (!desc().buffer_) {
        dev().srds().freeSrdSlot(hwSrd_);
    }
}

void
Resource::writeRawData(
    VirtualGPU& gpu,
    size_t      offset,
    size_t      size,
    const void* data,
    bool        waitForEvent) const
{
    GpuEvent    event;

    // Write data size bytes to surface
    // size needs to be DWORD aligned
    assert((size & 3) == 0);
    gpu.eventBegin(MainEngine);
    gpu.queue(MainEngine).addCmdMemRef(iMem());
    gpu.iCmd()->CmdUpdateMemory(*iMem(), offset, size, reinterpret_cast<const uint32_t*>(data));
    gpu.eventEnd(MainEngine, event);

    setBusy(gpu, event);
    // Update the global GPU event
    gpu.setGpuEvent(event, false);

    if (waitForEvent) {
        // Wait for event to complete
        gpu.waitForEvent(&event);
    }
}
static const Pal::ChNumFormat ChannelFmt(uint bytesPerElement)
{
    if (bytesPerElement == 16) {
        return Pal::ChNumFormat::X32Y32Z32W32_Uint;
    }
    else if (bytesPerElement == 8) {
        return Pal::ChNumFormat::X32Y32_Uint;
    }
    else if (bytesPerElement == 4) {
        return Pal::ChNumFormat::X32_Uint;
    }
    else if (bytesPerElement == 2) {
        return Pal::ChNumFormat::X16_Uint;
    }
    else {
        return Pal::ChNumFormat::X8_Uint;
    }
}

bool
Resource::partialMemCopyTo(
    VirtualGPU& gpu,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size,
    Resource& dstResource,
    bool enableCopyRect,
    bool flushDMA,
    uint bytesPerElement) const
{
    GpuEvent    event;
    bool        result = true;
    EngineType  activeEngineID = gpu.engineID_;
    static const bool waitOnBusyEngine = true;

    assert(!(desc().cardMemory_ && dstResource.desc().cardMemory_) &&
        "Unsupported configuraiton!");
    gpu.engineID_ = SdmaEngine;

    // Wait for the resources, since runtime may use async transfers
    wait(gpu, waitOnBusyEngine);
    dstResource.wait(gpu, waitOnBusyEngine);

    size_t     calSrcOrigin[3], calDstOrigin[3], calSize[3];
    calSrcOrigin[0] = srcOrigin[0] + pinOffset();
    calSrcOrigin[1] = srcOrigin[1];
    calSrcOrigin[2] = srcOrigin[2];
    calDstOrigin[0] = dstOrigin[0] + dstResource.pinOffset();
    calDstOrigin[1] = dstOrigin[1];
    calDstOrigin[2] = dstOrigin[2];
    calSize[0] = size[0];
    calSize[1] = size[1];
    calSize[2] = size[2];

    if (gpu.validateSdmaOverlap(*this, dstResource)) {
        gpu.flushDMA(SdmaEngine);
    }

    Pal::ImageLayout imgLayout = {};
    gpu.eventBegin(gpu.engineID_);
    gpu.queue(gpu.engineID_).addCmdMemRef(iMem());
    gpu.queue(gpu.engineID_).addCmdMemRef(dstResource.iMem());
    if (desc().buffer_ && !dstResource.desc().buffer_) {
        Pal::SubresId    ImgSubresId = { Pal::ImageAspect::Color, dstResource.desc().baseLevel_, 0 };
        Pal::MemoryImageCopyRegion copyRegion = {};
        copyRegion.imageSubres = ImgSubresId;
        copyRegion.imageOffset.x = calDstOrigin[0];
        copyRegion.imageOffset.y = calDstOrigin[1];
        copyRegion.imageOffset.z = calDstOrigin[2];
        copyRegion.imageExtent.width = calSize[0];
        copyRegion.imageExtent.height = calSize[1];
        copyRegion.imageExtent.depth = calSize[2];
        copyRegion.numSlices = 1;
        copyRegion.gpuMemoryOffset = calSrcOrigin[0] + offset();
        copyRegion.gpuMemoryRowPitch = (calSrcOrigin[1]) ? calSrcOrigin[1] :
            calSize[0] * dstResource.elementSize();
        copyRegion.gpuMemoryDepthPitch = (calSrcOrigin[2]) ? calSrcOrigin[2] :
            copyRegion.gpuMemoryRowPitch * calSize[1];
        // Make sure linear pitch in bytes is 4 bytes aligned
        if (((copyRegion.gpuMemoryRowPitch % 4) != 0) ||
            // another DRM restriciton... SI has 4 pixels
            (copyRegion.gpuMemoryOffset % 4 != 0) ||
            (dev().settings().sdamPageFaultWar_ && 
             (copyRegion.imageOffset.x % dstResource.elementSize() != 0))) {
            result = false;
        }
        else {
            gpu.iCmd()->CmdCopyMemoryToImage(*iMem(), *dstResource.image_,
                imgLayout, 1, &copyRegion);
        }
    }
    else if (!desc().buffer_ && dstResource.desc().buffer_) {
        Pal::MemoryImageCopyRegion copyRegion = {};
        Pal::SubresId    ImgSubresId = { Pal::ImageAspect::Color, desc().baseLevel_, 0 };
        copyRegion.imageSubres = ImgSubresId;
        copyRegion.imageOffset.x = calSrcOrigin[0];
        copyRegion.imageOffset.y = calSrcOrigin[1];
        copyRegion.imageOffset.z = calSrcOrigin[2];
        copyRegion.imageExtent.width = calSize[0];
        copyRegion.imageExtent.height = calSize[1];
        copyRegion.imageExtent.depth = calSize[2];
        copyRegion.numSlices = 1;
        copyRegion.gpuMemoryOffset = calDstOrigin[0] + dstResource.offset();
        copyRegion.gpuMemoryRowPitch = (calDstOrigin[1]) ? calDstOrigin[1] :
            calSize[0] * elementSize();
        copyRegion.gpuMemoryDepthPitch = (calDstOrigin[2]) ? calDstOrigin[2] :
            copyRegion.gpuMemoryRowPitch * calSize[1];
        // Make sure linear pitch in bytes is 4 bytes aligned
        if (((copyRegion.gpuMemoryRowPitch % 4) != 0) ||
            // another DRM restriciton... SI has 4 pixels
            (copyRegion.gpuMemoryOffset % 4 != 0) ||
            (dev().settings().sdamPageFaultWar_ &&
             (copyRegion.imageOffset.x % elementSize() != 0))) {
            result = false;
        }
        else {
            gpu.iCmd()->CmdCopyImageToMemory(*image_, imgLayout,
                *dstResource.iMem(), 1, &copyRegion);
        }
    }
    else {
        if (enableCopyRect) {
            Pal::TypedBufferCopyRegion copyRegion = {};
            Pal::ChannelMapping channels = { Pal::ChannelSwizzle::X, Pal::ChannelSwizzle::Y,
                Pal::ChannelSwizzle::Z, Pal::ChannelSwizzle::W };
            copyRegion.srcBuffer.swizzledFormat.format = ChannelFmt(bytesPerElement);
            copyRegion.srcBuffer.swizzledFormat.swizzle = channels;
            copyRegion.srcBuffer.offset = calSrcOrigin[0] + offset();
            copyRegion.srcBuffer.rowPitch = calSrcOrigin[1];
            copyRegion.srcBuffer.depthPitch = calSrcOrigin[2];
            copyRegion.extent.width = calSize[0] / bytesPerElement;
            copyRegion.extent.height = calSize[1];
            copyRegion.extent.depth = calSize[2];
            copyRegion.dstBuffer.swizzledFormat.format = ChannelFmt(bytesPerElement);
            copyRegion.dstBuffer.swizzledFormat.swizzle = channels;
            copyRegion.dstBuffer.offset = calDstOrigin[0] + dstResource.offset();
            copyRegion.dstBuffer.rowPitch = calDstOrigin[1];
            copyRegion.dstBuffer.depthPitch = calDstOrigin[2];
            gpu.iCmd()->CmdCopyTypedBuffer(*iMem(), *dstResource.iMem(),
                1, &copyRegion);
        }
        else {
            Pal::MemoryCopyRegion copyRegion = {};
            copyRegion.srcOffset = calSrcOrigin[0] + offset();
            copyRegion.dstOffset = calDstOrigin[0] + dstResource.offset();
            copyRegion.copySize = calSize[0];
            gpu.iCmd()->CmdCopyMemory(*iMem(), *dstResource.iMem(),
                1, &copyRegion);
        }
    }

    gpu.eventEnd(gpu.engineID_, event);

    if (result) {
        // Mark source and destination as busy
        setBusy(gpu, event);
        dstResource.setBusy(gpu, event);

        // Update the global GPU event
        gpu.setGpuEvent(event, flushDMA);
    }

    // Restore the original engine
    gpu.engineID_ = activeEngineID;

    return result;
}

void
Resource::setBusy(
    VirtualGPU& gpu,
    GpuEvent    gpuEvent
    ) const
{
    gpu.assignGpuEvent(iMem(), gpuEvent);

    // If current resource is a view, then update the parent event as well
    if (viewOwner_ != nullptr) {
        viewOwner_->setBusy(gpu, gpuEvent);
    }
}

void
Resource::wait(VirtualGPU& gpu, bool waitOnBusyEngine) const
{
    GpuEvent*   gpuEvent = gpu.getGpuEvent(iMem());

    // Check if we have to wait unconditionally
    if (!waitOnBusyEngine ||
        // or we have to wait only if another engine was used on this resource
        (waitOnBusyEngine && (gpuEvent->engineId_ != gpu.engineID_))) {
        gpu.waitForEvent(gpuEvent);
    }

    // If current resource is a view and not in the global heap,
    // then wait for the parent event as well
    if (viewOwner_ != nullptr) {
        viewOwner_->wait(gpu, waitOnBusyEngine);
    }
}

bool
Resource::hostWrite(
    VirtualGPU*         gpu,
    const void*         hostPtr,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    uint                flags,
    size_t              rowPitch,
    size_t              slicePitch)
{
    void*   dst;

    size_t  startLayer  = origin[2];
    size_t  numLayers   = size[2];
    if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
        startLayer  = origin[1];
        numLayers   = size[1];
    }

    // Get physical GPU memmory
    dst = map(gpu, flags, startLayer, numLayers);
    if (nullptr == dst) {
        LogError("Couldn't map GPU memory for host write");
        return false;
    }

    if (1 == desc().dimSize_) {
        size_t  copySize = (desc().buffer_) ? size[0] : size[0] * elementSize_;

        // Update the pointer
        dst = static_cast<void*>(static_cast<char*>(dst) + origin[0]);

        // Copy memory
        amd::Os::fastMemcpy(dst, hostPtr, copySize);
    }
    else {
        size_t srcOffs = 0;
        size_t dstOffsBase = origin[0] * elementSize_;
        size_t dstOffs;

        // Make sure we use the right pitch if it's not specified
        if (rowPitch == 0) {
            rowPitch = size[0] * elementSize_;
        }

        // Make sure we use the right slice if it's not specified
        if (slicePitch == 0) {
            slicePitch = size[0] * size[1] * elementSize_;
        }

        // Adjust the destination offset with Y dimension
        dstOffsBase += desc().pitch_ * origin[1] * elementSize_;

        // Adjust the destination offset with Z dimension
        dstOffsBase += desc().slice_ * origin[2] * elementSize_;

        // Copy memory slice by slice
        for (size_t slice = 0; slice < size[2]; ++slice) {
            dstOffs = dstOffsBase + slice * desc().slice_ * elementSize_;
            srcOffs = slice * slicePitch;

            // Copy memory line by line
            for (size_t row = 0; row < size[1]; ++row) {
                // Copy memory
                amd::Os::fastMemcpy(
                    (reinterpret_cast<address>(dst) + dstOffs),
                    (reinterpret_cast<const_address>(hostPtr) + srcOffs),
                    size[0] * elementSize_);

                dstOffs += desc().pitch_ * elementSize_;
                srcOffs += rowPitch;
            }
        }
    }

    // Unmap GPU memory
    unmap(gpu);

    return true;
}

bool
Resource::hostRead(
    VirtualGPU*         gpu,
    void*               hostPtr,
    const amd::Coord3D& origin,
    const amd::Coord3D& size,
    size_t              rowPitch,
    size_t              slicePitch)
{
    void*   src;

    size_t  startLayer  = origin[2];
    size_t  numLayers   = size[2];
    if (desc().topology_ == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
        startLayer  = origin[1];
        numLayers   = size[1];
    }

    // Get physical GPU memmory
    src = map(gpu, ReadOnly, startLayer, numLayers);
    if (nullptr == src) {
        LogError("Couldn't map GPU memory for host read");
        return false;
    }

    if (1 == desc().dimSize_) {
        size_t  copySize = (desc().buffer_) ? size[0] : size[0] * elementSize_;

        // Update the pointer
        src = static_cast<void*>(static_cast<char*>(src) + origin[0]);

        // Copy memory
        amd::Os::fastMemcpy(hostPtr, src, copySize);
    }
    else {
        size_t srcOffsBase = origin[0] * elementSize_;
        size_t srcOffs;
        size_t dstOffs = 0;

        // Make sure we use the right pitch if it's not specified
        if (rowPitch == 0) {
            rowPitch = size[0] * elementSize_;
        }

        // Make sure we use the right slice if it's not specified
        if (slicePitch == 0) {
            slicePitch = size[0] * size[1] * elementSize_;
        }

        // Adjust destination offset with Y dimension
        srcOffsBase += desc().pitch_ * origin[1] * elementSize_;

        // Adjust the destination offset with Z dimension
        srcOffsBase += desc().slice_ * origin[2] * elementSize_;

        // Copy memory line by line
        for (size_t slice = 0; slice < size[2]; ++slice) {
            srcOffs = srcOffsBase + slice * desc().slice_ * elementSize_;
            dstOffs = slice * slicePitch;

            // Copy memory line by line
            for (size_t row = 0; row < size[1]; ++row) {
                // Copy memory
                amd::Os::fastMemcpy(
                    (reinterpret_cast<address>(hostPtr) + dstOffs),
                    (reinterpret_cast<const_address>(src) + srcOffs),
                    size[0] * elementSize_);

                srcOffs += desc().pitch_ * elementSize_;
                dstOffs += rowPitch;
            }
        }
    }

    // Unmap GPU memory
    unmap(gpu);

    return true;
}

void*
Resource::gpuMemoryMap(size_t* pitch, uint flags, Pal::IGpuMemory* resource) const
{
    if (desc_.cardMemory_ && !isPersistentDirectMap()) {
        // @todo remove const cast
        Unimplemented();
        return nullptr;
//        return const_cast<Device&>(dev()).resMapLocal(*pitch, resource, flags);
    }
    else {
        amd::ScopedLock lk(dev().lockPAL());
        void*   address;
        if (image_ != nullptr) {
            constexpr  Pal::SubresId ImgSubresId = { Pal::ImageAspect::Color, 0, 0 };
            Pal::SubresLayout layout;
            image_->GetSubresourceLayout(ImgSubresId, &layout);
            *pitch = layout.rowPitch / elementSize();
        }
        *pitch = desc().width_;
        if (Pal::Result::Success == resource->Map(&address)) {
            return address;
        }
        else {
            LogError("PAL GpuMemory->Map() failed!");
            return nullptr;
        }
    }
}

void
Resource::gpuMemoryUnmap(Pal::IGpuMemory* resource) const
{
    if (desc_.cardMemory_ && !isPersistentDirectMap()) {
        // @todo remove const cast
        Unimplemented();
//        const_cast<Device&>(dev()).resUnmapLocal(resource);
    }
    else {
        Pal::Result result = resource->Unmap();
        if (Pal::Result::Success != result) {
            LogError("PAL GpuMemory->Unmap() failed!");
        }
    }
}

bool
Resource::glAcquire()
{
    bool retVal = true;
    if (desc().type_ == OGLInterop) {
        retVal = dev().resGLAcquire(glPlatformContext_, glInteropMbRes_, glType_);
    }
    return retVal;
}

bool
Resource::glRelease()
{
    bool retVal = true;
    if (desc().type_ == OGLInterop) {
        retVal = dev().resGLRelease(glPlatformContext_,glInteropMbRes_, glType_);
    }
    return retVal;
}
void
Resource::palFree() const
{
    amd::ScopedLock lk(dev().lockPAL());

    if (desc().type_ == OGLInterop) {
        dev().resGLFree(glPlatformContext_, glInteropMbRes_, glType_);
    }
    memRef_->release();
}

bool
Resource::isMemoryType(MemoryType memType) const
{
    if (memoryType() == memType) {
        return true;
    }
    else if (memoryType() == View) {
        return viewOwner_->isMemoryType(memType);
    }

    return false;
}

bool
Resource::isPersistentDirectMap() const
{
    bool directMap = ((memoryType() == Resource::Persistent) &&
        (desc().dimSize_ < 3) && !desc().imageArray_);

    // If direct map is possible, then validate it with the current tiling
    if (directMap && desc().tiled_) {
        //!@note IOL for Linux doesn't support tiling aperture
        // and runtime doesn't force linear images in persistent
        directMap = IS_WINDOWS && !dev().settings().linearPersistentImage_;
    }

    return directMap;
}

void*
Resource::map(VirtualGPU* gpu, uint flags, uint startLayer, uint numLayers)
{
    if (isMemoryType(Pinned)) {
        // Check if we have to wait
        if (!(flags & NoWait)) {
            if (gpu != nullptr) {
                wait(*gpu);
            }
        }
        return address_;
    }

    if (flags & ReadOnly) {
        assert(!(flags & Discard) && "We can't use lock discard with read only!");
    }

    if (flags & WriteOnly) {
    }

    // Check if use map discard
    if (flags & Discard) {
        if (gpu != nullptr) {
            // If we use a new renamed allocation, then skip the wait
            if (rename(*gpu)) {
                flags |= NoWait;
            }
        }
    }

    // Check if we have to wait
    if (!(flags & NoWait)) {
        if (gpu != nullptr) {
            wait(*gpu);
        }
    }

    // Check if memory wasn't mapped yet
    if (++mapCount_ == 1) {
        if ((desc().dimSize_ == 3) || desc().imageArray_ ||
            ((desc().type_ == ImageView) && viewOwner_->mipMapped())) {
            // Save map info for multilayer map/unmap
            startLayer_ = startLayer;
            numLayers_  = numLayers;
            mapFlags_   = flags;
            // Map with layers
            address_ = mapLayers(gpu, flags);
        }
        else {
            // Map current resource
            address_ = gpuMemoryMap(&desc_.pitch_, flags, iMem());
            if (address_ == nullptr) {
                LogError("cal::ResMap failed!");
                --mapCount_;
                return nullptr;
            }
        }
    }

    //! \note the atomic operation with counter doesn't
    // guarantee that the address will be valid,
    // since PAL could still process the first map
    if (address_ == nullptr) {
        for (uint i = 0; address_ == NULL && i < 10; ++i) {
            amd::Os::sleep(1);
        }
        assert((address_ != nullptr) && "Multiple maps failed!");
    }

    return address_;
}

void*
Resource::mapLayers(VirtualGPU* gpu, uint    flags)
{
    size_t srcOffs = 0;
    size_t dstOffs = 0;
    Pal::IGpuMemory*  sliceResource = 0;
    PalGpuMemoryType palDim = PAL_TEXTURE_2D;
    size_t layers = desc().depth_;
    size_t height = desc().height_;

    // Use 1D layers
    if (CL_MEM_OBJECT_IMAGE1D_ARRAY == desc().topology_) {
        palDim = PAL_TEXTURE_1D;
        height = 1;
        layers = desc().height_;
    }

    desc_.pitch_ = desc().width_;
    desc_.slice_ = desc().pitch_ * height;
    address_ = new char [desc().slice_ * layers * elementSize()];
    if (nullptr == address_) {
        return nullptr;
    }

    // Check if map is write only
    if (flags & WriteOnly) {
        return address_;
    }

    if (numLayers_ != 0) {
        layers = startLayer_ + numLayers_;
    }

    dstOffs = startLayer_ * desc().slice_ * elementSize();

    // Loop through all layers
    for (uint i = startLayer_; i < layers; ++i) {
  //      gslResource3D   gslSize;
        size_t          calOffset;
        void*           sliceAddr;
        size_t          pitch;
        Unimplemented();
        // Allocate a layer from the image
    //    gslSize.width   = desc().width_;
        //gslSize.height  = height;
        //gslSize.depth   = 1;
        calOffset       = 0;
/*
        sliceResource = dev().resAllocView(
            iMem(), gslSize,
            calOffset, desc().format_, desc().channelOrder_, palDim,
            0, i, CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER);
        if (0 == sliceResource) {
            LogError("Map layer. resAllocSliceView failed!");
            return nullptr;
        }
*/
        // Map 2D layer
        sliceAddr = gpuMemoryMap(&pitch, ReadOnly, sliceResource);
        if (sliceAddr == nullptr) {
            LogError("Map layer. CalResMap failed!");
            return nullptr;
        }

        srcOffs = 0;
        // Copy memory line by line
        for (size_t rows = 0; rows < height; ++rows) {
            // Copy memory
            amd::Os::fastMemcpy(
                (reinterpret_cast<address>(address_) + dstOffs),
                (reinterpret_cast<const_address>(sliceAddr) + srcOffs),
                desc().width_ * elementSize_);

            dstOffs += desc().pitch_ * elementSize();
            srcOffs += pitch * elementSize();
        }

        // Unmap a layer
        gpuMemoryUnmap(sliceResource);
        //dev().resFree(sliceResource);
    }

    return address_;
}

void
Resource::unmap(VirtualGPU* gpu)
{
    if (isMemoryType(Pinned)) {
        return;
    }

    // Decrement map counter
    int count = --mapCount_;

    // Check if it's the last unmap
    if (count == 0) {
        if ((desc().dimSize_ == 3) || desc().imageArray_ ||
            ((desc().type_ == ImageView) && viewOwner_->mipMapped())) {
            // Unmap layers
            unmapLayers(gpu);
        }
        else {
            // Unmap current resource
            gpuMemoryUnmap(iMem());
        }
        address_ = nullptr;
    }
    else if (count < 0) {
        LogError("dev().serialCalResUnmap failed!");
        ++mapCount_;
        return;
    }
}

void
Resource::unmapLayers(VirtualGPU* gpu)
{
    size_t srcOffs = 0;
    size_t dstOffs = 0;
    PalGpuMemoryType palDim = PAL_TEXTURE_2D;
    Pal::IGpuMemory*  sliceResource = nullptr;
    uint        layers = desc().depth_;
    uint        height = desc().height_;

    // Use 1D layers
    if (CL_MEM_OBJECT_IMAGE1D_ARRAY == desc().topology_) {
        palDim = PAL_TEXTURE_1D;
        height = 1;
        layers = desc().height_;
    }

    if (numLayers_ != 0) {
        layers = startLayer_ + numLayers_;
    }

    srcOffs = startLayer_ * desc().slice_ * elementSize();

    // Check if map is write only
    if (!(mapFlags_ & ReadOnly)) {
        // Loop through all layers
        for (uint i = startLayer_; i < layers; ++i) {
             Unimplemented();
//            gslResource3D   gslSize;
            size_t          calOffset;
            void*           sliceAddr;
            size_t          pitch;

            // Allocate a layer from the image
            //gslSize.width   = desc().width_;
            //gslSize.height  = height;
            //gslSize.depth   = 1;
            calOffset       = 0;
            /*sliceResource = dev().resAllocView(
                iMem(), gslSize,
                calOffset, desc().format_, desc().channelOrder_, palDim,
                0, i, CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER);
            if (0 == sliceResource) {
                LogError("Unmap layer. resAllocSliceView failed!");
                return;
            }
*/
            // Map a layer
            sliceAddr = gpuMemoryMap(&pitch, WriteOnly, sliceResource);
            if (sliceAddr == nullptr) {
                LogError("Unmap layer. CalResMap failed!");
                return;
            }

            dstOffs = 0;
            // Copy memory line by line
            for (size_t rows = 0; rows < height; ++rows) {
                // Copy memory
                amd::Os::fastMemcpy(
                    (reinterpret_cast<address>(sliceAddr) + dstOffs),
                    (reinterpret_cast<const_address>(address_) + srcOffs),
                    desc().width_ * elementSize_);

                dstOffs += pitch * elementSize();
                srcOffs += desc().pitch_ * elementSize();
            }

            // Unmap a layer
            gpuMemoryUnmap(sliceResource);
            //dev().resFree(sliceResource);
        }
    }

    // Destroy the mapped memory
    delete [] reinterpret_cast<char*>(address_);
}

void
Resource::setActiveRename(VirtualGPU& gpu, GpuMemoryReference* rename)
{
    // Copy the unique GSL data
    memRef_  = rename;
    address_ = rename->cpuAddress_;
}

bool
Resource::getActiveRename(VirtualGPU& gpu, GpuMemoryReference** rename)
{
    // Copy the old data to the rename descriptor
    *rename = memRef_;
    return true;
}

bool
Resource::rename(VirtualGPU& gpu, bool force)
{
    GpuEvent*   gpuEvent = gpu.getGpuEvent(iMem());
    if (!gpuEvent->isValid() && !force) {
        return true;
    }

    bool useNext = false;
    uint    resSize = desc().width_ * ((desc().height_) ? desc().height_ : 1) *
        elementSize_;

    // Rename will work with real GSL resources
    if (((memoryType() != Local) &&
         (memoryType() != Persistent) &&
         (memoryType() != Remote) &&
         (memoryType() != RemoteUSWC)) ||
         (dev().settings().maxRenames_ == 0)) {
        return false;
    }

    // If the resource for renaming is too big, then lets check the current status first
    // at the cost of an extra flush
    if (resSize >= (dev().settings().maxRenameSize_ / dev().settings().maxRenames_)) {
        if (gpu.isDone(gpuEvent)) {
            return true;
        }
    }

    // Save the first
    if (renames_.size() == 0) {
        GpuMemoryReference* rename;
        if (mapCount_ > 0) {
            memRef_->cpuAddress_ = address_;
        }
        if (!getActiveRename(gpu, &rename)) {
            return false;
        }

        curRename_ = renames_.size();
        renames_.push_back(rename);
    }

    // Can we use a new rename?
    if ((renames_.size() <= dev().settings().maxRenames_) &&
        ((renames_.size() * resSize) <= dev().settings().maxRenameSize_)) {
        GpuMemoryReference* rename;

        // Create a new GSL allocation
        if (create(memoryType())) {
            if (mapCount_ > 0) {
                assert(!desc().cardMemory_ && "Unsupported memory type!");
                memRef_->cpuAddress_ = gpuMemoryMap(&desc_.pitch_, 0, iMem());
                if (memRef_->cpuAddress_ == nullptr) {
                    LogError("gslMap fails on rename!");
                }
                address_ = memRef_->cpuAddress_;
            }
            if (getActiveRename(gpu, &rename)) {
                curRename_ = renames_.size();
                renames_.push_back(rename);
            }
            else {
                memRef_->release();
                useNext = true;
            }
        }
        else {
            useNext = true;
        }
    }
    else {
        useNext  = true;
    }

    if (useNext) {
        // Get the last submitted
        curRename_++;
        if (curRename_ >= renames_.size()) {
            curRename_ = 0;
        }
        setActiveRename(gpu, renames_[curRename_]);
        return false;
    }

    return true;
}

void
Resource::warmUpRenames(VirtualGPU& gpu)
{
    for (uint i = 0; i < dev().settings().maxRenames_; ++i) {
        uint    dummy = 0;
        const bool NoWait = false;
        // Write 0 for the buffer paging by VidMM
        writeRawData(gpu, 0, sizeof(dummy), &dummy, NoWait);
        const bool Force = true;
        rename(gpu, Force);
    }
}

ResourceCache::~ResourceCache()
{
    free();
}

//! \note the cache works in FILO mode
bool
ResourceCache::addGpuMemory(
    Resource::Descriptor* desc, GpuMemoryReference* ref)
{
    amd::ScopedLock l(&lockCacheOps_);
    bool result = false;
    size_t  size = ref->iMem()->Desc().size;

    // Make sure current allocation isn't bigger than cache
    if (((desc->type_ == Resource::Local) ||
         (desc->type_ == Resource::Persistent) ||
         (desc->type_ == Resource::Remote) ||
         (desc->type_ == Resource::RemoteUSWC)) &&
         (size < cacheSizeLimit_) &&
         !desc->SVMRes_) {
        // Validate the cache size limit. Loop until we have enough space
        while ((cacheSize_ + size) > cacheSizeLimit_) {
            removeLast();
        }
        Resource::Descriptor* descCached = new Resource::Descriptor;
        if (descCached != nullptr) {
            // Copy the original desc to the cached version
            memcpy(descCached, desc, sizeof(Resource::Descriptor));

            // Add the current resource to the cache
            resCache_.push_front(std::make_pair(descCached, ref));
            cacheSize_ += size;
            result  = true;
        }
    }

    return result;
}

GpuMemoryReference*
ResourceCache::findGpuMemory(
    Resource::Descriptor* desc, Pal::gpusize size, Pal::gpusize alignment)
{
    amd::ScopedLock l(&lockCacheOps_);
    GpuMemoryReference* ref = nullptr;

    // Early exit if resource is too big
    if (size >= cacheSizeLimit_ || desc->SVMRes_) {
        //! \note we may need to free the cache here to reduce memory pressure
        return ref;
    }

    // Serach the right resource through the cache list
    for (const auto& it: resCache_) {
        Resource::Descriptor*  entry = it.first;
        size_t sizeRes = it.second->iMem()->Desc().size;
        // Find if we can reuse this entry
        if ((entry->type_ == desc->type_) &&
            (entry->flags_ == desc->flags_) &&
            (size <= sizeRes) &&
            (size > (sizeRes >> 2)) &&
            ((it.second->iMem()->Desc().gpuVirtAddr % alignment) == 0) &&
            (entry->isAllocExecute_  == desc->isAllocExecute_)) {
                ref = it.second;
                delete it.first;
                // Remove the found etry from the cache
                resCache_.remove(it);
                cacheSize_ -= sizeRes;
                break;
        }
    }

    return ref;
}

bool
ResourceCache::free(size_t minCacheEntries)
{
    amd::ScopedLock l(&lockCacheOps_);
    bool result = false;

    if (minCacheEntries < resCache_.size()) {
        if (static_cast<int>(cacheSize_) > 0) {
            result = true;
        }
        // Clear the cache
        while (static_cast<int>(cacheSize_) > 0) {
            removeLast();
        }
        CondLog((cacheSize_ != 0), "Incorrect size for cache release!");
    }
    return result;
}

void
ResourceCache::removeLast()
{
    std::pair<Resource::Descriptor*, GpuMemoryReference*> entry;
    entry = resCache_.back();
    resCache_.pop_back();

    size_t  size = entry.second->iMem()->Desc().size;

    // Delete Descriptor
    delete entry.first;

    // Destroy GSL resource
    entry.second->release();
    cacheSize_ -= size;
}

} // namespace pal
