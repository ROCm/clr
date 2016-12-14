//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef WITHOUT_HSA_BACKEND

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "CL/cl_ext.h"

#include "utils/util.hpp"
#include "device/device.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocglinterop.hpp"
#include "thread/monitor.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "api/opencl/amdocl/cl_gl_amd.hpp"

namespace roc {

/////////////////////////////////roc::Memory//////////////////////////////
Memory::Memory(const roc::Device &dev, amd::Memory &owner)
    : device::Memory(owner),
      dev_(dev),
      deviceMemory_(NULL),
      kind_(MEMORY_KIND_NORMAL)
{
}

Memory::~Memory()
{
    dev_.removeVACache(this);
}

bool
Memory::allocateMapMemory(size_t allocationSize)
{
    assert(mapMemory_ == NULL);

    void *mapData = NULL;

    amd::Memory* mapMemory = dev_.findMapTarget(owner()->getSize());

    if (mapMemory == nullptr) {
        // Create buffer object to contain the map target.
        mapMemory =
          new(owner()->getContext()) amd::Buffer(
          owner()->getContext(), CL_MEM_ALLOC_HOST_PTR, owner()->getSize());

        if ((mapMemory == NULL) || (!mapMemory->create())) {
            LogError("[OCL] Fail to allocate map target object");
            dev_.hostFree(mapData);
            if (mapMemory) {
              mapMemory->release();
            }
            return false;
        }

        roc::Memory* hsaMapMemory = reinterpret_cast<roc::Memory *>(
            mapMemory->getDeviceMemory(dev_));
        if (hsaMapMemory == nullptr) {
        	 mapMemory->release();
        	 return false;
        }
    }

    mapMemory_ = mapMemory;

    return true;
}

void*
Memory::allocMapTarget(
    const amd::Coord3D &origin,
    const amd::Coord3D &region,
    uint    mapFlags,
    size_t *rowPitch,
    size_t *slicePitch)
{
    // Map/Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());

    incIndMapCount();

    // If the device backing storage is direct accessible, use it.
    if (isHostMemDirectAccess()) {
        if (owner()->getHostMem() != nullptr) {
            return (static_cast<char *>(owner()->getHostMem()) + origin[0]);
        }

        return (static_cast<char *>(deviceMemory_) + origin[0]);
    }

    // Otherwise, check for host memory.
    void *hostMem = owner()->getHostMem();
    if (hostMem != NULL) {
        return (static_cast<char *>(hostMem) + origin[0]);
    }

    // Allocate one if needed.
    if (indirectMapCount_ == 1) {
        if (!allocateMapMemory(owner()->getSize())) {
            decIndMapCount();
            return NULL;
        }
    }
    else {
        // Did the map resource allocation fail?
        if (mapMemory_ == NULL) {
            LogError("Could not map target resource");
            return NULL;
        }
    }

    roc::Memory* hsaMapMemory = reinterpret_cast<roc::Memory *>(
        mapMemory_->getDeviceMemory(dev_));
    return reinterpret_cast<address>(hsaMapMemory->getDeviceMemory()) + origin[0];
}

void
Memory::decIndMapCount()
{
    // Map/Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());

    if (indirectMapCount_ == 0) {
        LogError("decIndMapCount() called when indirectMapCount_ already zero");
        return;
    }

    // Decrement the counter and release indirect map if it's the last op
    if (--indirectMapCount_ == 0 &&
        mapMemory_ != NULL) {
        if (!dev_.addMapTarget(mapMemory_)) {
            // Release the buffer object containing the map data.
            mapMemory_->release();
        }
        mapMemory_ = nullptr;
    }
}

void *
Memory::cpuMap(
    device::VirtualDevice& vDev,
    uint flags,
    uint startLayer,
    uint numLayers,
    size_t* rowPitch,
    size_t* slicePitch)
{
    // Create the map target.
    void * mapTarget =
        allocMapTarget(amd::Coord3D(0), amd::Coord3D(0), 0, rowPitch, slicePitch);

    assert(mapTarget != NULL);

    if (!isHostMemDirectAccess()) {
        if (!vDev.blitMgr().readBuffer(
            *this, mapTarget, amd::Coord3D(0), amd::Coord3D(size()), true)) {
            decIndMapCount();
            return NULL;
        }
    }

    return mapTarget;
}

void
Memory::cpuUnmap(device::VirtualDevice& vDev)
{
    if (!isHostMemDirectAccess()) {
        if (!vDev.blitMgr().writeBuffer(
            mapMemory_->getHostMem(), *this, amd::Coord3D(0),
            amd::Coord3D(size()), true)) {
            LogError("[OCL] Fail sync the device memory on cpuUnmap");
        }
    }

    decIndMapCount();
}

// Setup an interop buffer (dmabuf handle) as an OpenCL buffer
bool Memory::createInteropBuffer(GLenum targetType, int miplevel, size_t* metadata_size, const hsa_amd_image_descriptor_t** metadata)
{
#if defined(_WIN32)
  return false;
#else
  assert(owner()->isInterop() && "Object is not an interop object.");
  
  mesa_glinterop_export_in in;
  mesa_glinterop_export_out out;

  in.size=sizeof(mesa_glinterop_export_in);
  out.size=sizeof(mesa_glinterop_export_out);

  if(owner()->getMemFlags() & CL_MEM_READ_ONLY)
    in.access=MESA_GLINTEROP_ACCESS_READ_ONLY;
  else if(owner()->getMemFlags() & CL_MEM_WRITE_ONLY)
    in.access=MESA_GLINTEROP_ACCESS_WRITE_ONLY;
  else
    in.access=MESA_GLINTEROP_ACCESS_READ_WRITE;

  in.target = targetType;
  in.obj=owner()->getInteropObj()->asGLObject()->getGLName();
  in.miplevel=miplevel;
  in.out_driver_data_size=0;
  in.out_driver_data=NULL;

  if(!dev_.mesa().Export(in, out))
    return false;
  
  size_t size;
  hsa_agent_t agent=dev_.getBackendDevice();
  hsa_status_t status=hsa_amd_interop_map_buffer(1, &agent, out.dmabuf_fd, 0, &size, &deviceMemory_, metadata_size, (const void**)metadata);
  close(out.dmabuf_fd);

  if(status!=HSA_STATUS_SUCCESS)
    return false;

  kind_=MEMORY_KIND_INTEROP;
  assert(deviceMemory_!=NULL && "Interop map failed to produce a pointer!");

  return true;
#endif
}

void Memory::destroyInteropBuffer()
{
  assert(kind_==MEMORY_KIND_INTEROP && "Memory must be interop type.");
  hsa_amd_interop_unmap_buffer(deviceMemory_);
  deviceMemory_=NULL;
}

/////////////////////////////////roc::Buffer//////////////////////////////

Buffer::Buffer(const roc::Device &dev, amd::Memory &owner)
    : roc::Memory(dev, owner)
{}

Buffer::~Buffer()
{
    destroy();
}

void
Buffer::destroy()
{
    if (owner()->parent()  != NULL)  {
        return;
    }

    if(kind_==MEMORY_KIND_INTEROP)
    {
      destroyInteropBuffer();
      return;
    }

    const cl_mem_flags memFlags = owner()->getMemFlags();

    if ((deviceMemory_  != nullptr) &&
        (deviceMemory_ != owner()->getHostMem())) {
        // if they are identical, the host pointer will be
        // deallocated later on => avoid double deallocation
        if (isHostMemDirectAccess()) {
            if (memFlags & CL_MEM_USE_HOST_PTR) {
                if (dev_.agent_profile() != HSA_PROFILE_FULL) {
                    hsa_amd_memory_unlock(owner()->getHostMem());
                }
            }
        }
        else {
            dev_.memFree(deviceMemory_, size());
        }
    }

    if (memFlags & CL_MEM_USE_HOST_PTR) {
        if (dev_.agent_profile() == HSA_PROFILE_FULL) {
            hsa_memory_deregister(owner()->getHostMem(), size());
        }
    }
}

bool
Buffer::create()
{
    //Interop buffer
    if(owner()->isInterop())
      return createInteropBuffer(GL_ARRAY_BUFFER, 0, NULL, NULL);

    if (owner()->parent()) {
        // Sub-Buffer creation.
        roc::Memory *parentBuffer =
          static_cast<roc::Memory *>(owner()->parent()->getDeviceMemory(dev_));

        if (parentBuffer == NULL) {
            LogError("[OCL] Fail to allocate parent buffer");
            return false;
        }

        const size_t offset = owner()->getOrigin();
        deviceMemory_ =
            static_cast<char *>(parentBuffer->getDeviceMemory()) + offset;

        flags_ |= SubMemoryObject;
        flags_ |=
            parentBuffer->isHostMemDirectAccess() ? HostMemoryDirectAccess : 0;
        return true;
    }

    // Allocate backing storage in device local memory unless UHP or AHP are set
    const cl_mem_flags memFlags = owner()->getMemFlags();
    if (!(memFlags & (CL_MEM_USE_HOST_PTR |
        CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_PERSISTENT_MEM_AMD))) {
        deviceMemory_ = dev_.deviceLocalAlloc(size());

        if (deviceMemory_ == NULL) {
            // TODO: device memory is not enabled yet.
            // Fallback to system memory if exist.

            flags_ |= HostMemoryDirectAccess;
            if (dev_.agent_profile() == HSA_PROFILE_FULL &&
                owner()->getHostMem() != NULL) {
                deviceMemory_ = owner()->getHostMem();
                assert(
                    amd::isMultipleOf(
                    deviceMemory_,
                    static_cast<size_t>(dev_.info().memBaseAddrAlign_)));
                return true;
            }

            deviceMemory_ = dev_.hostAlloc(size(), 1, false);
        }

        assert(
            amd::isMultipleOf(
            deviceMemory_,
            static_cast<size_t>(dev_.info().memBaseAddrAlign_)));

        if (deviceMemory_ && (memFlags & CL_MEM_COPY_HOST_PTR)) {
            // To avoid recurssive call to Device::createMemory, we perform
            // data transfer to the view of the buffer.
            amd::Buffer *bufferView = new (owner()->getContext()) amd::Buffer(
                *owner(), 0, owner()->getOrigin(), owner()->getSize());
            bufferView->create();

            roc::Buffer *devBufferView =
                new roc::Buffer(dev_, *bufferView);
            devBufferView->deviceMemory_ = deviceMemory_;

            bufferView->replaceDeviceMemory(&dev_, devBufferView);

            bool ret = dev_.xferMgr().writeBuffer(
                owner()->getHostMem(), *devBufferView, amd::Coord3D(0),
                amd::Coord3D(size()), true);

            if (!ret) {
                dev_.memFree(deviceMemory_, size());
                deviceMemory_ = NULL;
            }

            bufferView->release();
            return ret;
        }

        return deviceMemory_ != NULL;
    }
    else if (memFlags & CL_MEM_USE_PERSISTENT_MEM_AMD) {
        deviceMemory_ = dev_.hostAlloc(size(), 1, false);
        if (deviceMemory_ != nullptr) {
            if (owner()->getHostMem() != nullptr) {
                memcpy(deviceMemory_, owner()->getHostMem(), size());
            }
            flags_ |= HostMemoryDirectAccess;
        }
        return deviceMemory_ != nullptr;
    }

    assert(owner()->getHostMem() != NULL);

    flags_ |= HostMemoryDirectAccess;

    if (dev_.agent_profile() == HSA_PROFILE_FULL) {
      deviceMemory_ = owner()->getHostMem();

      if (memFlags & CL_MEM_USE_HOST_PTR) {
        hsa_memory_register(deviceMemory_, size());
      }

      return deviceMemory_ != NULL;
    }

    if (owner()->getSvmPtr() != owner()->getHostMem()) {
        if (memFlags & CL_MEM_USE_HOST_PTR) {
            hsa_agent_t agent = dev_.getBackendDevice();
            hsa_status_t status = hsa_amd_memory_lock(
                owner()->getHostMem(), owner()->getSize(), &agent, 1, &deviceMemory_);
            if (status != HSA_STATUS_SUCCESS) {
                deviceMemory_ = nullptr;
            }
        }
        else {
            deviceMemory_ = owner()->getHostMem();
        }
    }
    else {
        deviceMemory_ = owner()->getHostMem();
    }

    return deviceMemory_ != NULL;
}

/////////////////////////////////roc::Image//////////////////////////////
typedef struct ChannelOrderMap {
    uint32_t cl_channel_order;
    hsa_ext_image_channel_order_t hsa_channel_order;
} ChannelOrderMap;

typedef struct ChannelTypeMap {
    uint32_t cl_channel_type;
    hsa_ext_image_channel_type_t hsa_channel_type;
} ChannelTypeMap;

static const ChannelOrderMap kChannelOrderMapping[] = {
    { CL_R, HSA_EXT_IMAGE_CHANNEL_ORDER_R },
    { CL_A, HSA_EXT_IMAGE_CHANNEL_ORDER_A },
    { CL_RG, HSA_EXT_IMAGE_CHANNEL_ORDER_RG },
    { CL_RA, HSA_EXT_IMAGE_CHANNEL_ORDER_RA },
    { CL_RGB, HSA_EXT_IMAGE_CHANNEL_ORDER_RGB },
    { CL_RGBA, HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA },
    { CL_BGRA, HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA },
    { CL_ARGB, HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB },
    { CL_INTENSITY, HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY },
    { CL_LUMINANCE, HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE },
    { CL_Rx, HSA_EXT_IMAGE_CHANNEL_ORDER_RX },
    { CL_RGx, HSA_EXT_IMAGE_CHANNEL_ORDER_RGX },
    { CL_RGBx, HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX },
    { CL_DEPTH, HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH },
    { CL_DEPTH_STENCIL, HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL },
    { CL_sRGB, HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB },
    { CL_sRGBx, HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX },
    { CL_sRGBA, HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA },
    { CL_sBGRA, HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA },
    { CL_ABGR, HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR },
};

static const ChannelTypeMap kChannelTypeMapping[] = {
    {CL_SNORM_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8},
    {CL_SNORM_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16},
    {CL_UNORM_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8},
    {CL_UNORM_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16},
    {CL_UNORM_SHORT_565, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565},
    {CL_UNORM_SHORT_555, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555},
    {CL_UNORM_INT_101010, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010},
    {CL_SIGNED_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8},
    {CL_SIGNED_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16},
    {CL_SIGNED_INT32, HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32},
    {CL_UNSIGNED_INT8, HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8},
    {CL_UNSIGNED_INT16, HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16},
    {CL_UNSIGNED_INT32, HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32},
    {CL_HALF_FLOAT, HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT},
    {CL_FLOAT, HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT},
    {CL_UNORM_INT24, HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24},
};


static hsa_access_permission_t
GetHsaAccessPermission(const cl_mem_flags flags) {
  if(flags & CL_MEM_READ_ONLY)
        return HSA_ACCESS_PERMISSION_RO;
  else if(flags & CL_MEM_WRITE_ONLY)
        return HSA_ACCESS_PERMISSION_WO;
  else
    return HSA_ACCESS_PERMISSION_RW;
}

Image::Image(const roc::Device& dev, amd::Memory& owner) :
    roc::Memory(dev, owner)
{
    flags_ &= (~HostMemoryDirectAccess & ~HostMemoryRegistered);
    populateImageDescriptor();
    hsaImageObject_.handle = 0;
    originalDeviceMemory_ = NULL;
}

void
Image::populateImageDescriptor()
{
    amd::Image* image = owner()->asImage();

    // build HSA runtime image descriptor
    imageDescriptor_.width = image->getWidth();
    imageDescriptor_.height = image->getHeight();
    imageDescriptor_.depth = image->getDepth();
    imageDescriptor_.array_size = 0;

    switch (image->getType())
    {
    case CL_MEM_OBJECT_IMAGE1D:
        imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_1D;
        imageDescriptor_.height = 1;
        imageDescriptor_.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_1DB;
        imageDescriptor_.height = 1;
        imageDescriptor_.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        //@todo - arraySize = height ?!
        imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_1DA;
        imageDescriptor_.height = 1;
        imageDescriptor_.array_size = image->getHeight();
        break;
    case CL_MEM_OBJECT_IMAGE2D:
        imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
        imageDescriptor_.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        //@todo - arraySize = depth ?!
        imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
        imageDescriptor_.depth = 1;
        imageDescriptor_.array_size = image->getDepth();
        break;
    case CL_MEM_OBJECT_IMAGE3D:
        imageDescriptor_.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
        break;
    }

    const int kChannelOrderCount =
        sizeof(kChannelOrderMapping) / sizeof(ChannelOrderMap);
    for (int i = 0; i < kChannelOrderCount; i++) {
      if (image->getImageFormat().image_channel_order ==
          kChannelOrderMapping[i].cl_channel_order) {
        imageDescriptor_.format.channel_order =
            kChannelOrderMapping[i].hsa_channel_order;
        break;
      }
    }

    const int kChannelTypeCount =
        sizeof(kChannelTypeMapping) / sizeof(ChannelTypeMap);
    for (int i = 0; i < kChannelTypeCount; i++) {
        if (image->getImageFormat().image_channel_data_type ==
            kChannelTypeMapping[i].cl_channel_type) {
            imageDescriptor_.format.channel_type =
                kChannelTypeMapping[i].hsa_channel_type;
            break;
        }
    }

    permission_ =
      GetHsaAccessPermission(owner()->getMemFlags());
}

bool
Image::createInteropImage()
{
  auto obj=owner()->getInteropObj()->asGLObject();
  assert(obj->getCLGLObjectType()!=CL_GL_OBJECT_BUFFER && "Non-image OpenGL object used with interop image API.");
  
  const hsa_amd_image_descriptor_t* meta;
  size_t size=0;
  
  GLenum glTarget = obj->getGLTarget();
  if (glTarget == GL_TEXTURE_CUBE_MAP) {
    glTarget = obj->getCubemapFace();
  }
  if(!createInteropBuffer(glTarget, obj->getGLMipLevel(), &size, &meta))
  {
    assert(false && "Failed to map image buffer.");
    return false;
  }
  MAKE_SCOPE_GUARD(BufferGuard, [&](){ destroyInteropBuffer(); });

  amdImageDesc_=(hsa_amd_image_descriptor_t*)malloc(size);
  if(amdImageDesc_==NULL)
    return false;
  MAKE_SCOPE_GUARD(DescGuard, [&](){ free(amdImageDesc_); amdImageDesc_=NULL; });

  memcpy(amdImageDesc_, meta, size);

  image_metadata desc;
  if(!desc.create(amdImageDesc_))
    return false;

  if(!desc.setMipLevel(obj->getGLMipLevel()))
    return false;

  if (obj->getGLTarget()==GL_TEXTURE_CUBE_MAP)
    desc.setFace(obj->getCubemapFace());
  
  originalDeviceMemory_=deviceMemory_;

  hsa_status_t err=hsa_amd_image_create(dev_.getBackendDevice(), &imageDescriptor_, amdImageDesc_, originalDeviceMemory_, permission_, &hsaImageObject_);
  if(err!=HSA_STATUS_SUCCESS)
    return false;
  
  BufferGuard.Dismiss();
  DescGuard.Dismiss();
  return true;
}

bool
Image::create()
{
    if (owner()->parent()) {
        // Image view creation
        roc::Memory *parent =
          static_cast<roc::Memory *>(owner()->parent()->getDeviceMemory(dev_));

        if (parent == NULL) {
            LogError("[OCL] Fail to allocate parent image");
            return false;
        }

        return createView(*parent);
    }

    //Interop image
    if(owner()->isInterop())
      return createInteropImage();

    // Get memory size requirement for device specific image.
    hsa_status_t status = hsa_ext_image_data_get_info(
        dev_.getBackendDevice(), &imageDescriptor_,
        permission_, &deviceImageInfo_);

    if (status != HSA_STATUS_SUCCESS) {
        LogError("[OCL] Fail to allocate image memory");
        return false;
    }

    // roc::Device::hostAlloc and deviceLocalAlloc implementation does not
    // support alignment larger than HSA memory region allocation granularity.
    // In this case, the user manages the alignment.
    const size_t alloc_size =
      (deviceImageInfo_.alignment <= dev_.alloc_granularity())
      ? deviceImageInfo_.size
      : deviceImageInfo_.size + deviceImageInfo_.alignment;

    if (!(owner()->getMemFlags() & CL_MEM_ALLOC_HOST_PTR)) {
      originalDeviceMemory_ = dev_.deviceLocalAlloc(alloc_size);
    }

    if (originalDeviceMemory_ == NULL) {
        originalDeviceMemory_ =
          dev_.hostAlloc(alloc_size, 1, false);
    }

    deviceMemory_ = reinterpret_cast<void *>(
      amd::alignUp(reinterpret_cast<uintptr_t>(originalDeviceMemory_),
      deviceImageInfo_.alignment));

    assert(amd::isMultipleOf(
      deviceMemory_, static_cast<size_t>(deviceImageInfo_.alignment)));

    status = hsa_ext_image_create(
      dev_.getBackendDevice(), &imageDescriptor_, deviceMemory_,
        permission_, &hsaImageObject_);

    if (status != HSA_STATUS_SUCCESS) {
        LogError("[OCL] Fail to allocate image memory");
        return false;
    }

    return true;
}

bool
Image::createView(Memory &parent)
{
    deviceMemory_ = parent.getDeviceMemory();

    originalDeviceMemory_ = (parent.owner()->asBuffer() != NULL)
                        ? deviceMemory_
                        : static_cast<Image &>(parent).originalDeviceMemory_;

    kind_=parent.getKind();

    hsa_status_t status;
    if(kind_==MEMORY_KIND_INTEROP)
      status = hsa_amd_image_create(dev_.getBackendDevice(), &imageDescriptor_, amdImageDesc_, deviceMemory_, permission_, &hsaImageObject_);
    else
      status= hsa_ext_image_create(dev_.getBackendDevice(), &imageDescriptor_, deviceMemory_, permission_, &hsaImageObject_);

    if (status != HSA_STATUS_SUCCESS) {
        LogError("[OCL] Fail to allocate image memory");
        return false;
    }

    return true;
}

void*
Image::allocMapTarget(
    const amd::Coord3D& origin,
    const amd::Coord3D& region,
    uint    mapFlags,
    size_t* rowPitch,
    size_t* slicePitch)
{
    amd::ScopedLock lock(owner()->lockMemoryOps());

    incIndMapCount();

    void* pHostMem = owner()->getHostMem();

    amd::Image* image = owner()->asImage();

    size_t elementSize = image->getImageFormat().getElementSize();

    size_t  offset = origin[0] * elementSize;

    if (pHostMem == NULL) {
        if (indirectMapCount_ == 1) {
            if (!allocateMapMemory(owner()->getSize())) {
                decIndMapCount();
                return NULL;
            }
        }
        else {
            // Did the map resource allocation fail?
            if (mapMemory_ == NULL) {
                LogError("Could not map target resource");
                return NULL;
            }
        }

        pHostMem = mapMemory_->getHostMem();

        *rowPitch = region[0] * elementSize;

        size_t slicePitchTmp = 0;

        if (imageDescriptor_.geometry == HSA_EXT_IMAGE_GEOMETRY_1DA) {
            slicePitchTmp = *rowPitch;
        }
        else {
            slicePitchTmp = *rowPitch * region[1];
        }
        if (slicePitch != NULL) {
            *slicePitch = slicePitchTmp;
        }

        return pHostMem;
    }

    // Adjust offset with Y dimension
    offset += image->getRowPitch() * origin[1];

    // Adjust offset with Z dimension
    offset += image->getSlicePitch() * origin[2];

    *rowPitch = image->getRowPitch();
    if (slicePitch != NULL) {
        *slicePitch = image->getSlicePitch();
    }

    return (static_cast<uint8_t*>(pHostMem)+offset);
}

Image::~Image()
{
    destroy();
}

void
Image::destroy()
{
  if (hsaImageObject_.handle != 0) {
      hsa_status_t status =
          hsa_ext_image_destroy(dev_.getBackendDevice(), hsaImageObject_);
      assert(status == HSA_STATUS_SUCCESS);
  }

  if (owner()->parent() != NULL) {
      return;
  }

  if(kind_==MEMORY_KIND_INTEROP)
  {
    free(amdImageDesc_);
    amdImageDesc_=NULL;
    destroyInteropBuffer();
    return;
  }

  if (originalDeviceMemory_ != NULL) {
      dev_.memFree(originalDeviceMemory_, deviceImageInfo_.size);
  }
}
}
#endif  // WITHOUT_HSA_BACKEND
