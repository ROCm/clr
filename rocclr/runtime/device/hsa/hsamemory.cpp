//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef WITHOUT_FSA_BACKEND

#include "CL/cl_ext.h"

#include "device/device.hpp"
#include "device/hsa/hsamemory.hpp"
#include "device/hsa/hsadevice.hpp"
#include "device/hsa/hsablit.hpp"
#include "device/hsa/oclhsa_common.hpp"
#include "thread/monitor.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"

namespace oclhsa {

/////////////////////////////////oclhsa::Memory//////////////////////////////
Memory::Memory(const oclhsa::Device &dev, amd::Memory &owner)
    : device::Memory(owner),
      dev_(dev),
      deviceMemory_(NULL),
      interopType_(InteropNone)
{
}

Memory::~Memory()
{}

bool
Memory::allocateMapMemory(size_t allocationSize)
{
    assert(mapMemory_ == NULL);

    void *mapData = NULL;

    // Use/reuse system memory from HSA system memory pool as backing
    // storage of the map target.
    if (kHsaStatusSuccess !=
        servicesapi->HsaAllocateSystemMemory(
            owner()->getSize(), 0, kHsaSystemMemoryTypeDefault, &mapData)) {
        LogError("[OCL] Fail to allocate the backing storage for map target");
        return false;
    }

    // Create buffer object to contain the map target.
    amd::Memory *mapMemory =
        new(owner()->getContext()) amd::Buffer(
            owner()->getContext(), CL_MEM_USE_HOST_PTR, owner()->getSize());

    if ((mapMemory == NULL) || (!mapMemory->create(mapData))) {
        LogError("[OCL] Fail to allocate map target object");
        servicesapi->HsaFreeSystemMemory(mapData);
        if (mapMemory) {
            mapMemory->release();
        }
        return false;
    }

    mapMemory_ = mapMemory;

    return true;
}

void 
Memory::freeMapMemory()
{
    // Return the memory to HSA system memory pool.
    assert(mapMemory_ != NULL);
    servicesapi->HsaFreeSystemMemory(mapMemory_->getHostMem());

    // Release the buffer object containing the map data.
    mapMemory_->release();
    mapMemory_ = NULL;
}

void *
Memory::allocMapTarget(const amd::Coord3D &origin,
                       const amd::Coord3D &region,
                       size_t *rowPitch,
                       size_t *slicePitch) 
{
    // Map/Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());

    incIndMapCount();

    // If the device backing storage is direct accessible, use it.
    if (isHostMemDirectAccess()) {
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

    return (static_cast<char *>(mapMemory_->getHostMem()) + origin[0]);
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
        freeMapMemory();
    }
}

void *
Memory::cpuMap(
      device::VirtualDevice& vDev,
      uint flags,
      uint startLayer,
      uint numLayers,
      size_t* rowPitch,
      size_t* slicePitch
      )
{
    // Create the map target.
    void * mapTarget =
        allocMapTarget(amd::Coord3D(0), amd::Coord3D(0), rowPitch, slicePitch);

    // Sync to map target if no direct access.
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
    // Sync to device backing storage if no direct access.
    if (!isHostMemDirectAccess()) {
        if (!vDev.blitMgr().writeBuffer(
                mapMemory_->getHostMem(), *this, amd::Coord3D(0),
                amd::Coord3D(size()), true)) {
            LogError("[OCL] Fail sync the device memory on cpuUnmap");
        }
    }

    decIndMapCount();
}

void Memory::destroyInterop()
{
    HsaStatus status;
#ifdef _WIN32
    if (interopType_ == InteropD3D10) {
        HsaStatus status = hsacoreapi->HsaUnmapD3D10Resource(
        dev_.getBackendDevice(), d3d10Resource_);
        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaUnmapD3D10Resource");
            return;
        }
    }

    else if (interopType_ == InteropD3D11) {
        HsaStatus status = hsacoreapi->HsaUnmapD3D11Resource(
        dev_.getBackendDevice(), d3d11Resource_);
        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaUnmapD3D11Resource");
            return;
        }
    }
#endif

    if (interopType_ == InteropGL) {
        void * glContext =owner()->getContext().info().hCtx_;
        status = hsacoreapi->HsaReleaseGLResources( dev_.getBackendDevice(),
                                                        glContext,
                                                        &glResource_,
                                                        1);
        if (kHsaStatusSuccess != status) {
            LogError("[OCL] Fail on HsaReleaseGLResources");
        }

        status = hsacoreapi->HsaUnmapGLResource(
            dev_.getBackendDevice(), glContext, &glResource_);

        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaUnmapGLResource");
            return;
        }
    }
}

bool
Memory::isHsaLocalMemory() const {
    if (owner()->isInterop()) {
        return true;
    }
    else {
        if (amd::Is64Bits()) {
            uint64_t addr = reinterpret_cast<uint64_t>(deviceMemory_);

            // Fast check: in 64 bits, CPU can only access the high area
            // (VA[63:47] == 0x1FFFF) and low area (VA[63:47 == 0).
            // Reference: GFXIP7_ShaderIO_Delt.doc
            addr >>= 47;  // discard least significant 47 bits
            return (addr != 0x1FFFF && addr != 0);
        }
        else {
            const HsaMemoryDescriptor &memDesc =
                dev_.getBackendDevice()->memory_descriptors[0];

            if (memDesc.heap_type == kHsaHeapTypeFrameBufferPrivate) {
                const uintptr_t addr =
                    reinterpret_cast<uintptr_t>(deviceMemory_);
                const uintptr_t gpuvmBase = memDesc.virtual_base_address;
                const size_t size = memDesc.size_in_bytes;
                return (addr >= gpuvmBase && addr < (gpuvmBase + size));
            }
        }
    }
    return false;
}

/////////////////////////////////oclhsa::Buffer//////////////////////////////

Buffer::Buffer(const oclhsa::Device &dev, amd::Memory &owner)
    : oclhsa::Memory(dev, owner)
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

    if (owner()->isInterop()) {
        destroyInterop();
        return;
    }

    if (isHostMemoryRegistered()) {
        hsacoreapi->HsaDeregisterSystemMemory(deviceMemory_);
    }
    else {
        if (!isHostMemDirectAccess()) {
            hsacoreapi->HsaFreeDeviceMemory(deviceMemory_);
        }
        else if (deviceMemory_ != owner()->getHostMem()) {
            // if they are identical, the host pointer will be
            // deallocated later on => avoid double deallocation
            hsacoreapi->HsaAmdFreeSystemMemory(deviceMemory_);
        }
    }
}

bool Buffer::createInterop()
{
    amd::InteropObject *interopObject = owner()->getInteropObj();

#ifdef _WIN32
    if (interopObject->asD3D10Object() != NULL) {
        amd::D3D10Object *d3d10Object = interopObject->asD3D10Object();
        // 1. Get the D3D11 resource
        ID3D10Resource *resource = d3d10Object->getD3D10Resource();
        ID3D10Buffer *d3d10Buffer = static_cast<ID3D10Buffer *>(resource);

        HsaStatus status = hsacoreapi->HsaMapD3D10Buffer(
        dev_.getBackendDevice(), d3d10Buffer, &deviceMemory_);
        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaMapD3D10Buffer");
            return false;
        }
        interopType_ = InteropD3D10;
        d3d10Resource_ = d3d10Buffer;
    }

    if (interopObject->asD3D11Object() != NULL) {
        amd::D3D11Object *d3d11Object = interopObject->asD3D11Object();
        // 1. Get the D3D11 resource
        ID3D11Resource *resource = d3d11Object->getD3D11Resource();
        ID3D11Buffer *d3d11Buffer = static_cast<ID3D11Buffer *>(resource);

        HsaStatus status = hsacoreapi->HsaMapD3D11Buffer(
            dev_.getBackendDevice(), d3d11Buffer, &deviceMemory_);
        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaMapD3D10Buffer");
            return false;
        }
        interopType_ = InteropD3D11;
        d3d11Resource_ = d3d11Buffer;
    }
#endif

    if (interopObject->asBufferGL()) {
        amd::BufferGL *buffer_gl = interopObject->asBufferGL();
        HsaGLResource gl_resource = {0};
        gl_resource.name = buffer_gl->getGLName();
        gl_resource.type = buffer_gl->getGLInternalFormat();

        void * glContext =owner()->getContext().info().hCtx_;
        HsaStatus status = hsacoreapi->HsaMapGLBuffer(
            dev_.getBackendDevice(), glContext, &gl_resource, &deviceMemory_);
        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaMapGLBuffer");
            return false;
        }

        status = hsacoreapi->HsaAcquireGLResources( dev_.getBackendDevice(),
                                                            glContext,
                                                           &gl_resource,
                                                           1);

        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaAcquireGLResources");
            return false;
        }
        interopType_ = InteropGL;
        glResource_ = gl_resource;
    }
    return true;
}

bool
Buffer::create()
{
    if (owner()->parent()) {
        // Sub-Buffer creation.
        oclhsa::Memory *parentBuffer =
          static_cast<oclhsa::Memory *>(owner()->parent()->getDeviceMemory(dev_));

        if (parentBuffer == NULL) {
            LogError("[OCL] Fail to allocate parent buffer");
            return false;
        }

        const size_t offset = owner()->getOrigin();
        deviceMemory_ =
            static_cast<char *>(parentBuffer->getDeviceMemory()) + offset;

        void* parentHostPtr = parentBuffer->owner()->getHostMem();
        if (parentHostPtr) {
            owner()->setHostMem(static_cast<char *>(parentHostPtr) + offset);
        }

        flags_ |= owner()->parent()->getMemFlags();
        return true;
    }

    // Allocate backing storage in device local memory unless UHP or AHP are set
    const cl_mem_flags memFlags = owner()->getMemFlags();
    if (!(memFlags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))) {
        bool useDeviceMemory = dev_.settings().enableLocalMemory_;
        size_t alignment = static_cast<size_t>(dev_.info().memBaseAddrAlign_);
        if (useDeviceMemory) {
            hsacoreapi->HsaAllocateDeviceMemory(
                size(), alignment, dev_.getBackendDevice(), &deviceMemory_);
            if (deviceMemory_ && (memFlags & CL_MEM_COPY_HOST_PTR)) {
                bool ret = dev_.xferMgr().writeBuffer(owner()->getHostMem(), *this,
                             amd::Coord3D(0), amd::Coord3D(size()), true);
                if (!ret) {
                    hsacoreapi->HsaFreeDeviceMemory(deviceMemory_);
                    deviceMemory_ = NULL;
                }
                return ret;
            }
            // if device memory is depleted, do not fall back to system memory
            return deviceMemory_ != NULL;
        }
        else if (!(owner()->getHostMem())) {
            flags_ |= HostMemoryDirectAccess;
            deviceMemory_ = dev_.hostAlloc(size(), alignment);
            // no need to copy - otherwise, the host pointer will not be NULL
            return deviceMemory_ != NULL;
        }
    }

    flags_ |= HostMemoryDirectAccess;
    void* hostMem = owner()->getHostMem();
    assert(hostMem);
    // If there is a host ptr, then register it only if it was not allocated,
    // (=> allocated by us)
    if (!(owner()->getHostMemRef()->alloced())) {
        // Reuse existing host memory for the backing storage and register it.
        //
        // SVM precludes a possible 64-bits optimization in which host buffers
        // allocated by the user (UHP) in the default, coherent space could be
        // mapped into the non-coherent space by means of CreateFileMapping/mmap
        // without copying any data (the "device memory" would be the
        // non-coherent buffer).
        // The optimization cannot be applied because regular buffers allocated
        // using UHP are expected to have same characteristics as the original
        // buffer, i.e., if the original buffer supports atomics then the
        // corresponding OpenCL buffer will support atomics too.
        flags_ |= HostMemoryRegistered;
        if (hsacoreapi->HsaRegisterSystemMemory(hostMem, size()) != kHsaStatusSuccess) {
            LogError("[OCL] Failed to register system memory");
            return false;
        }
    }
    deviceMemory_ = hostMem;
    return true;
}

bool
Buffer::recreate(size_t newSize, size_t newAlignment, bool forceSystem) {
    const size_t memFlag = static_cast<size_t>(owner()->getMemFlags());
    if ((memFlag & CL_MEM_ALLOC_HOST_PTR) ||
        (memFlag & CL_MEM_USE_HOST_PTR) ||
        !dev_.settings().enableLocalMemory_) {
        forceSystem = true;
    }

    void *newDeviceMemory = NULL;
    uint hostDirectAccess = 0;

    if (forceSystem) {
        newDeviceMemory = dev_.hostAlloc(newSize, newAlignment);
        if (newDeviceMemory == NULL) {
            LogError("[OCL] Fail to reallocate system memory");
            return false;
        }

        // Copy the old data to the new memory location.
        if (!dev_.xferMgr().readBuffer(*this, newDeviceMemory,
                                       amd::Coord3D(0),
                                       amd::Coord3D(size()),
                                       true)) {
            LogError("[OCL] Fail to copy the current value");
            dev_.hostFree(newDeviceMemory);
            newDeviceMemory = NULL;
            return false;
        }

        hostDirectAccess = HostMemoryDirectAccess;
    }
    else {
        hsacoreapi->HsaAllocateDeviceMemory(
            newSize, newAlignment, dev_.getBackendDevice(), &newDeviceMemory);

        if (newDeviceMemory == NULL) {
            LogError("[OCL] Fail to reallocate device local memory");
            return false;
        }

        assert(
            amd::isMultipleOf(static_cast<char *>(newDeviceMemory),
            newAlignment));

        // Copy the old data to the new memory location.
        if (!dev_.xferMgr().readBuffer(
                *this, newDeviceMemory, amd::Coord3D(0), amd::Coord3D(size()),
                true)) {
            LogError("[OCL] Fail to copy the current value");
            hsacoreapi->HsaFreeDeviceMemory(newDeviceMemory);
            newDeviceMemory = NULL;
            return false;
        }
    }

    destroy();

    deviceMemory_ = newDeviceMemory;

    if ((memFlag & CL_MEM_ALLOC_HOST_PTR) &&
        (owner()->getContext().devices().size() == 1)) {
        owner()->setHostMem(deviceMemory_);
    }

    flags_ &= (~HostMemoryDirectAccess & ~HostMemoryRegistered);
    flags_ |= hostDirectAccess;

    return true;
}

/////////////////////////////////oclhsa::Image//////////////////////////////

Image::Image(const oclhsa::Device& dev, amd::Memory& owner) :
    oclhsa::Memory(dev, owner)
{
    flags_ &= (~HostMemoryDirectAccess & ~HostMemoryRegistered);
    populateImageDescriptor();
}

struct ImageFormatLayout {
    cl_image_format clFormat;
    HsaImageFormat hsaFormat;
};

static const ImageFormatLayout
    ImageFormatLayoutMap[] = {
        { { CL_R, CL_UNORM_INT8 }, HSA_IMAGE_FMT_R8_UNORM  },
        { { CL_R, CL_UNORM_INT16}, HSA_IMAGE_FMT_R16_UNORM },
        { { CL_R, CL_SNORM_INT8 }, HSA_IMAGE_FMT_R8_SNORM  },
        { { CL_R, CL_SNORM_INT16}, HSA_IMAGE_FMT_R16_SNORM },
        { { CL_R, CL_SIGNED_INT8}, HSA_IMAGE_FMT_R8_SINT   },
        { { CL_R, CL_SIGNED_INT16}, HSA_IMAGE_FMT_R16_SINT},
        { { CL_R, CL_SIGNED_INT32}, HSA_IMAGE_FMT_R32_SINT},
        { { CL_R, CL_UNSIGNED_INT8},HSA_IMAGE_FMT_R8_UINT },
        { { CL_R, CL_UNSIGNED_INT16}, HSA_IMAGE_FMT_R16_UINT}, 
        { { CL_R, CL_UNSIGNED_INT32}, HSA_IMAGE_FMT_R32_UINT},
        { { CL_R, CL_HALF_FLOAT}, HSA_IMAGE_FMT_R_HALFFLOAT},
        { { CL_R, CL_FLOAT }, HSA_IMAGE_FMT_R_FLOAT},
        { { CL_A, CL_UNORM_INT8 }, HSA_IMAGE_FMT_A8_UNORM},
        { { CL_A, CL_UNORM_INT16 }, HSA_IMAGE_FMT_A16_UNORM},
        { { CL_A, CL_SNORM_INT8 }, HSA_IMAGE_FMT_A8_SNORM},
        { { CL_A, CL_SNORM_INT16 }, HSA_IMAGE_FMT_A16_SNORM},
        { { CL_A, CL_SIGNED_INT8 }, HSA_IMAGE_FMT_A8_SINT},
        { { CL_A, CL_SIGNED_INT16 },HSA_IMAGE_FMT_A16_SINT},
        { { CL_A, CL_SIGNED_INT32}, HSA_IMAGE_FMT_A32_SINT},
        { { CL_A, CL_UNSIGNED_INT8 },HSA_IMAGE_FMT_A8_UINT},
        { { CL_A, CL_UNSIGNED_INT16}, HSA_IMAGE_FMT_A16_UINT}, 
        { { CL_A, CL_UNSIGNED_INT32}, HSA_IMAGE_FMT_A32_UINT},
        { { CL_A, CL_HALF_FLOAT}, HSA_IMAGE_FMT_A_HALFFLOAT},
        { { CL_A, CL_FLOAT}, HSA_IMAGE_FMT_A_FLOAT},
        { { CL_RG,CL_UNORM_INT8}, HSA_IMAGE_FMT_R8G8_UNORM},
        { { CL_RG,CL_UNORM_INT16},HSA_IMAGE_FMT_R16G16_UNORM},
        { { CL_RG,CL_SNORM_INT8}, HSA_IMAGE_FMT_R8G8_SNORM},
        { { CL_RG,CL_SNORM_INT16},HSA_IMAGE_FMT_R16G16_SNORM},
        { { CL_RG,CL_SIGNED_INT8},HSA_IMAGE_FMT_R8G8_SINT},
        { { CL_RG,CL_SIGNED_INT16},HSA_IMAGE_FMT_R16G16_SINT},
        { { CL_RG,CL_SIGNED_INT32},HSA_IMAGE_FMT_R32G32_SINT},
        { { CL_RG,CL_UNSIGNED_INT8},HSA_IMAGE_FMT_R8G8_UINT},
        { { CL_RG,CL_UNSIGNED_INT16},HSA_IMAGE_FMT_R16G16_UINT},
        { { CL_RG,CL_UNSIGNED_INT32},HSA_IMAGE_FMT_R32G32_UINT},
        { { CL_RG,CL_HALF_FLOAT},HSA_IMAGE_FMT_RG_HALFFLOAT},
        { { CL_RG,CL_FLOAT},HSA_IMAGE_FMT_RG_FLOAT},
        { { CL_RA,CL_UNORM_INT8}, HSA_IMAGE_FMT_R8A8_UNORM},
        { { CL_RA,CL_UNORM_INT16},HSA_IMAGE_FMT_R16A16_UNORM},
        { { CL_RA,CL_SNORM_INT8}, HSA_IMAGE_FMT_R8A8_SNORM},
        { { CL_RA,CL_SNORM_INT16},HSA_IMAGE_FMT_R16A16_SNORM},
        { { CL_RA,CL_SIGNED_INT8},HSA_IMAGE_FMT_R8A8_SINT},
        { { CL_RA,CL_SIGNED_INT16},HSA_IMAGE_FMT_R16A16_SINT},
        { { CL_RA,CL_SIGNED_INT32},HSA_IMAGE_FMT_R32A32_SINT},
        { { CL_RA,CL_UNSIGNED_INT8},HSA_IMAGE_FMT_R8A8_UINT},
        { { CL_RA,CL_UNSIGNED_INT16},HSA_IMAGE_FMT_R16A16_UINT},
        { { CL_RA,CL_UNSIGNED_INT32},HSA_IMAGE_FMT_R32A32_UINT},
        { { CL_RA,CL_HALF_FLOAT},HSA_IMAGE_FMT_RA_HALFFLOAT},
        { { CL_RA,CL_FLOAT},HSA_IMAGE_FMT_RA_FLOAT},
        { { CL_RGBA,CL_UNORM_INT8}, HSA_IMAGE_FMT_R8G8B8A8_UNORM},
        { { CL_RGBA,CL_UNORM_INT16},HSA_IMAGE_FMT_R16G16B16A16_UNORM},
        { { CL_RGBA,CL_SNORM_INT8}, HSA_IMAGE_FMT_R8G8B8A8_SNORM},
        { { CL_RGBA,CL_SNORM_INT16},HSA_IMAGE_FMT_R16G16B16A16_SNORM},
        { { CL_RGBA,CL_SIGNED_INT8},HSA_IMAGE_FMT_R8G8B8A8_SINT},
        { { CL_RGBA,CL_SIGNED_INT16},HSA_IMAGE_FMT_R16G16B16A16_SINT},
        { { CL_RGBA,CL_SIGNED_INT32},HSA_IMAGE_FMT_R32G32B32A32_SINT},
        { { CL_RGBA,CL_UNSIGNED_INT8},HSA_IMAGE_FMT_R8G8B8A8_UINT},
        { { CL_RGBA,CL_UNSIGNED_INT16},HSA_IMAGE_FMT_R16G16B16A16_UINT},
        { { CL_RGBA,CL_UNSIGNED_INT32},HSA_IMAGE_FMT_R32G32B32A32_UINT},
        { { CL_RGBA,CL_HALF_FLOAT},HSA_IMAGE_FMT_RGBA_HALFFLOAT},
        { { CL_RGBA,CL_FLOAT},HSA_IMAGE_FMT_RGBA_FLOAT},
        { { CL_ARGB,CL_UNORM_INT8},HSA_IMAGE_FMT_A8R8G8B8_UNORM},
        { { CL_ARGB,CL_SNORM_INT8},HSA_IMAGE_FMT_A8R8G8B8_SNORM},
        { { CL_ARGB,CL_SIGNED_INT8},HSA_IMAGE_FMT_A8R8G8B8_SINT},
        { { CL_ARGB,CL_UNSIGNED_INT8},HSA_IMAGE_FMT_A8R8G8B8_UINT},
        { { CL_BGRA,CL_UNORM_INT8},HSA_IMAGE_FMT_B8G8R8A8_UNORM},
        { { CL_BGRA,CL_SNORM_INT8},HSA_IMAGE_FMT_B8G8R8A8_SNORM},
        { { CL_BGRA,CL_SIGNED_INT8},HSA_IMAGE_FMT_B8G8R8A8_SINT},
        { { CL_BGRA,CL_UNSIGNED_INT8},HSA_IMAGE_FMT_B8G8R8A8_UINT},
        { {CL_LUMINANCE,CL_SNORM_INT8}, HSA_IMAGE_FMT_L8_SNORM},
        { {CL_LUMINANCE,CL_SNORM_INT16},HSA_IMAGE_FMT_L16_SNORM},
        { {CL_LUMINANCE,CL_UNORM_INT8},HSA_IMAGE_FMT_L8_UNORM},
        { {CL_LUMINANCE,CL_UNORM_INT16},HSA_IMAGE_FMT_L16_UNORM},
        { {CL_LUMINANCE,CL_HALF_FLOAT},HSA_IMAGE_FMT_L_HALFFLOAT},
        { {CL_LUMINANCE,CL_FLOAT},HSA_IMAGE_FMT_L_FLOAT},
        { {CL_INTENSITY,CL_SNORM_INT8}, HSA_IMAGE_FMT_I8_SNORM},
        { {CL_INTENSITY,CL_SNORM_INT16},HSA_IMAGE_FMT_I16_SNORM},
        { {CL_INTENSITY,CL_UNORM_INT8},HSA_IMAGE_FMT_I8_UNORM},
        { {CL_INTENSITY,CL_UNORM_INT16},HSA_IMAGE_FMT_I16_UNORM},
        { {CL_INTENSITY,CL_HALF_FLOAT},HSA_IMAGE_FMT_I_HALFFLOAT},
        { {CL_INTENSITY,CL_FLOAT},HSA_IMAGE_FMT_I_FLOAT},
        { {CL_RGB, CL_UNORM_SHORT_565},HSA_IMAGE_FMT_R5G6B5_UNORM},
        { {CL_RGB, CL_UNORM_SHORT_555},HSA_IMAGE_FMT_R5G5B5_UNORM},
        { {CL_RGB, CL_UNORM_INT_101010},HSA_IMAGE_FMT_R10G10B10_UNORM}
};

void
Image::populateImageDescriptor()
{
    amd::Image* image = owner()->asImage();

    // build HSA runtime image descriptor
    imageDescriptor_.width = image->getWidth();
    imageDescriptor_.height = image->getHeight();
    imageDescriptor_.depth = image->getDepth();
    imageDescriptor_.arraySize = 0;

    // Device specific image does not require rowpitch/slicepitch information.
    // Only image buffer is required to specify rowpitch size.
    imageDescriptor_.rowPitchInBytes = 0;
    imageDescriptor_.slicePitchInBytes = 0;

    switch (image->getType())
    {
    case CL_MEM_OBJECT_IMAGE1D:
        imageDescriptor_.geometry = HSA_GEOMETRY_1D;
        imageDescriptor_.height = 1;
        imageDescriptor_.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        imageDescriptor_.geometry = HSA_GEOMETRY_1DBuffer;
        imageDescriptor_.height = 1;
        imageDescriptor_.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        //@todo - arraySize = height ?!
        imageDescriptor_.geometry = HSA_GEOMETRY_1DArray;
        imageDescriptor_. height = 1;
        imageDescriptor_.arraySize = image->getHeight();
        break;
    case CL_MEM_OBJECT_IMAGE2D:
        imageDescriptor_.geometry = HSA_GEOMETRY_2D;
        imageDescriptor_.depth = 1;
        break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        //@todo - arraySize = depth ?!
        imageDescriptor_.geometry = HSA_GEOMETRY_2DArray;
        imageDescriptor_.depth = 1;
        imageDescriptor_.arraySize = image->getDepth();
        break;
    case CL_MEM_OBJECT_IMAGE3D:
        imageDescriptor_.geometry = HSA_GEOMETRY_3D;
        break;
    }

    for (uint i = 0; i < sizeof(ImageFormatLayoutMap) / sizeof(ImageFormatLayout); ++i) {
        if ((image->getImageFormat().image_channel_data_type ==
            ImageFormatLayoutMap[i].clFormat.image_channel_data_type) &&
            (image->getImageFormat().image_channel_order ==
            ImageFormatLayoutMap[i].clFormat.image_channel_order)) {
                imageDescriptor_.format = ImageFormatLayoutMap[i].hsaFormat;
        }
    }
}

bool Image::createInterop() {
    amd::ScopedLock lock(owner()->lockMemoryOps());
    amd::InteropObject *interopObject = owner()->getInteropObj();
    void *hsaImageObjectInterop = NULL;
    size_t hsaImageObjectInteropSize = 0;
#ifdef _WIN32
    if (interopObject->asD3D10Object()) {
        amd::D3D10Object *d3d10Object = interopObject->asD3D10Object();
        // 1. Get the D3D11 resource
        ID3D10Resource *resource = d3d10Object->getD3D10Resource();
        HsaStatus status = hsacoreapi->HsaMapD3D10Texture(
          dev_.getBackendDevice(), resource, &hsaImageObjectInterop,
          &hsaImageObjectInteropSize, kHsaMapFlagsReadWrite);
        if (status != kHsaStatusSuccess || hsaImageObjectInteropSize == 0 ) {
            LogError("[OCL] Fail on HsaMapD3D10Texture");
            return false;
        }
        interopType_ = InteropD3D10;
        d3d10Resource_ = resource;
    }

    if (interopObject->asD3D11Object()) {
        amd::D3D11Object *d3d11Object = interopObject->asD3D11Object();
        
        // 1. Get the D3D11 resource
        ID3D11Resource *resource = d3d11Object->getD3D11Resource();
        HsaStatus status = hsacoreapi->HsaMapD3D11Texture(
          dev_.getBackendDevice(), resource, &hsaImageObjectInterop,
          &hsaImageObjectInteropSize, kHsaMapFlagsReadWrite,
          d3d11Object->getPlane());
        if (status != kHsaStatusSuccess || hsaImageObjectInteropSize == 0 ) {
            LogError("[OCL] Fail on HsaMapD3D11Texture");
            return false;
        }
        interopType_ = InteropD3D11;
        d3d11Resource_ = resource;
    }
#endif

    if (interopObject->asGLObject()) {
        amd::GLObject* gl_object = interopObject->asGLObject();
        HsaGLResource gl_resource = {0};
        gl_resource.name = gl_object->getGLName();
        if (gl_object->getGLTarget() != GL_TEXTURE_CUBE_MAP) {
            gl_resource.type = gl_object->getGLTarget();
        }
        else {
            gl_resource.type = gl_object->getCubemapFace();
        }
        gl_resource.mipmap_level = gl_object->getGLMipLevel();

        void * glContext =owner()->getContext().info().hCtx_;

        // Get the texture SRD.
        HsaStatus status = hsacoreapi->HsaMapGLTexture(
          dev_.getBackendDevice(), glContext, &gl_resource,
          &hsaImageObjectInterop, &hsaImageObjectInteropSize);
        if (status != kHsaStatusSuccess || hsaImageObjectInteropSize == 0) {
            LogError("[OCL] Fail on HsaMapGLTexture");
            return false;
        }

        status = hsacoreapi->HsaAcquireGLResources( dev_.getBackendDevice(),
                                                    glContext,
                                                    &gl_resource,
                                                    1);

        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail on HsaAcquireGLResources");
            return false;
        }

        // Get the flat address for texture buffer.
        if (owner()->getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
            // Map the texture buffer resource as buffer.
            HsaStatus status = hsacoreapi->HsaMapGLBuffer(
                dev_.getBackendDevice(), glContext, &gl_resource,
                &deviceMemory_);
            if (status != kHsaStatusSuccess) {
                LogError("[OCL] Fail on HsaMapGLBuffer");
                return false;
            }
            // Sanity check.
            assert((deviceMemory_ != NULL) &&
                   "deviceMemory_ should not be \
                   NULL upon successful return from HsaMapGLBuffer");
        }

        interopType_ = InteropGL;
        glResource_ = gl_resource;
    }

    // Populate HSA specific information to the interop image object.
    HsaStatus status = hsacoreapi->HsaAmdCreateDeviceImageView(
        &imageDescriptor_, hsaImageObjectInterop, hsaImageObject_);
    if (status != kHsaStatusSuccess) {
        LogError("[OCL] Fail to tranform interop image SRD");
        return false;
    }
    return true;
}

bool Image::create()
{
    if (owner()->parent()) {
        // Image view creation
        oclhsa::Image *parentImage =
          static_cast<oclhsa::Image *>(owner()->parent()->getDeviceMemory(dev_));

        if (parentImage == NULL) {
            LogError("[OCL] Fail to allocate parent image");
            return false;
        }

        return createView(*parentImage);
    }

    amd::ScopedLock lock(owner()->lockMemoryOps());

    // Get memory size requirement for device specific image.
    HsaStatus status = hsacoreapi->HsaGetDeviceImageInfo(
        dev_.getBackendDevice(), &imageDescriptor_,
        &deviceImageInfo_);

    if (status != kHsaStatusSuccess) {
        LogError("[OCL] Fail to allocate image memory");
        return false;
    }

    if (dev_.settings().enableLocalMemory_) {
        status = hsacoreapi->HsaAllocateDeviceMemory(
            deviceImageInfo_.imageSizeInBytes,
            deviceImageInfo_.imageAlignmentInBytes,
            dev_.getBackendDevice(),
            &deviceMemory_);
    } else {
        status = servicesapi->HsaAllocateSystemMemory(
            deviceImageInfo_.imageSizeInBytes,
            deviceImageInfo_.imageAlignmentInBytes,
            kHsaSystemMemoryTypeDefault,
            &deviceMemory_);
    }

    if (status != kHsaStatusSuccess) {
        LogError("[OCL] Fail to allocate image memory");
        return false;
    }

    assert(amd::isMultipleOf(
        deviceMemory_, deviceImageInfo_.imageAlignmentInBytes));

    status = hsacoreapi->HsaCreateDeviceImage(
        dev_.getBackendDevice(), &imageDescriptor_,
        deviceMemory_, &hsaImageObject_[0]);

    return true;
}

bool
Image::createView(Image &parent)
{
    amd::ScopedLock lock(owner()->lockMemoryOps());

    if (parent.owner()->asBuffer()) {
        // Get new texture SRD since parent is a buffer.
        deviceMemory_ = parent.getDeviceMemory();

        // Force device specific image implementation to use rowpitch size.
        amd::Image* image = owner()->asImage();
        imageDescriptor_.rowPitchInBytes = image->getRowPitch();

        HsaStatus status = hsacoreapi->HsaCreateDeviceImage(
            dev_.getBackendDevice(), &imageDescriptor_,
            deviceMemory_, &hsaImageObject_[0]);

        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail to create HSA image object");
            return false;
        }
    } else {
        // Get the view of the existing parent's SRD based on the child's image
        // descriptor.
        HsaStatus status = hsacoreapi->HsaAmdCreateDeviceImageView(
            &imageDescriptor_, parent.getHsaImageObjectAddress(),
            &hsaImageObject_[0]);
        if (status != kHsaStatusSuccess) {
            LogError("[OCL] Fail to get view of parent image");
            return false;
        }
    }

    return true;
}

void* Image::allocMapTarget(const amd::Coord3D& origin,
    const amd::Coord3D& region,
    size_t* rowPitch,
    size_t* slicePitch)
{
    amd::ScopedLock lock(owner()->lockMemoryOps());

    incIndMapCount();

    void* pHostMem = owner()->getHostMem();

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
    }

    amd::Image* image = owner()->asImage();

    size_t elementSize = image->getImageFormat().getElementSize();

    size_t  offset  = origin[0] * elementSize;

    // Adjust offset with Y dimension
    offset += image->getRowPitch() * origin[1];

    // Adjust offset with Z dimension
    offset += image->getSlicePitch() * origin[2];

    *rowPitch = image->getRowPitch();
    if (slicePitch != NULL)
        *slicePitch = image->getSlicePitch();

    return (static_cast<uint8_t*>(pHostMem) + offset);
}

Image::~Image()
{
    destroy();
}

void
Image::destroy()
{
    if (owner()->parent() != NULL) {
        return;
    }

    if (owner()->isInterop()) {
        destroyInterop();
        return;
    }

    if (dev_.settings().enableLocalMemory_) {
        hsacoreapi->HsaFreeDeviceMemory(deviceMemory_);
    }
    else {
        servicesapi->HsaFreeSystemMemory(deviceMemory_);
    }
}
}
#endif  // WITHOUT_FSA_BACKEND
