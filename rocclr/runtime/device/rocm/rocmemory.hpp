//
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#include "top.hpp"
#include "platform/memory.hpp"
#include "utils/debug.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocglinterop.hpp"

namespace roc {
class Memory : public device::Memory {
 public:
    enum MEMORY_KIND { MEMORY_KIND_NORMAL=0, MEMORY_KIND_LOCK, MEMORY_KIND_GART, MEMORY_KIND_INTEROP };

    Memory(const roc::Device &dev, amd::Memory &owner);

    virtual ~Memory();

    // Getter for deviceMemory_.
    void *getDeviceMemory() const { return deviceMemory_; }

    // Gets a pointer to a region of host-visible memory for use as the target
    // of an indirect map for a given memory object
    virtual void *allocMapTarget(const amd::Coord3D &origin,
                                 const amd::Coord3D &region,
                                 uint   mapFlags,
                                 size_t *rowPitch,
                                 size_t *slicePitch);

    // Create device memory according to OpenCL memory flag.
    virtual bool create() = 0;

    // Pins system memory associated with this memory object.
    virtual bool pinSystemMemory(void *hostPtr, // System memory address
                                 size_t size    // Size of allocated system memory
                                 ) {
        Unimplemented();
        return true;
    }
  
    // Immediate blocking write from device cache to owners's backing store.
    // Marks owner as "current" by resetting the last writer to NULL.
    virtual void syncHostFromCache(SyncFlags syncFlags = SyncFlags())
    {
        // Need to revisit this when multi-devices is supported.
    }

    // Releases indirect map surface
    void releaseIndirectMap() { decIndMapCount(); }

    //! Map the device memory to CPU visible
    virtual void* cpuMap(
        device::VirtualDevice& vDev,    //!< Virtual device for map operaiton
        uint flags = 0,         //!< flags for the map operation
        // Optimization for multilayer map/unmap
        uint startLayer = 0,    //!< Start layer for multilayer map
        uint numLayers = 0,     //!< End layer for multilayer map
        size_t* rowPitch = NULL,//!< Row pitch for the device memory
        size_t* slicePitch = NULL   //!< Slice pitch for the device memory
        );

    //! Unmap the device memory
    virtual void cpuUnmap(
        device::VirtualDevice& vDev     //!< Virtual device for unmap operaiton
        );

    //Mesa has already decomressed if needed and also does acquire at the start of every command batch.
    virtual bool processGLResource(GLResourceOP operation) { return true; }

    // Accessors for indirect map memory object
    amd::Memory *mapMemory() const { return mapMemory_; }

    MEMORY_KIND getKind() const { return kind_; }

 protected:

    bool allocateMapMemory(size_t allocationSize);

    // Decrement map count
    virtual void decIndMapCount();

    // Free / deregister device memory.
    virtual void destroy() = 0;

    // Place interop object into HSA's flat address space
    bool createInteropBuffer(GLenum targetType, int miplevel, size_t* metadata_size, const hsa_amd_image_descriptor_t** metadata);

    void destroyInteropBuffer();

    // Pointer to the device associated with this memory object.
    const roc::Device &dev_;

    // Pointer to the device memory. This could be in system or device local mem.
    void* deviceMemory_;

    // Track if this memory is interop, lock, gart, or normal.
    MEMORY_KIND kind_;

   private:
    // Disable copy constructor
    Memory(const Memory &);

    // Disable operator=
    Memory &operator=(const Memory &);

};

class Buffer : public roc::Memory {
 public:
    Buffer(const roc::Device &dev, amd::Memory &owner);

    virtual ~Buffer();

    // Create device memory according to OpenCL memory flag.
    virtual bool create();

    // Recreate the device memory using new size and alignment.
    bool recreate(size_t newSize, size_t newAlignment, bool forceSystem);

 private:
    // Disable copy constructor
    Buffer(const Buffer &);

    // Disable operator=
    Buffer &operator=(const Buffer &);

    // Free device memory.
    void destroy();
};

class Image : public roc::Memory
{
public:
    Image(const roc::Device& dev, amd::Memory& owner);

    virtual ~Image();

    //! Create device memory according to OpenCL memory flag.
    virtual bool create();

    //! Create an image view
    bool createView(Memory &parent);

    //! Gets a pointer to a region of host-visible memory for use as the target
    //! of an indirect map for a given memory object
    virtual void* allocMapTarget(
        const amd::Coord3D& origin,
        const amd::Coord3D& region,
        uint    mapFlags,
        size_t* rowPitch,
        size_t* slicePitch);

    size_t getDeviceDataSize() { return deviceImageInfo_.size; }
    size_t getDeviceDataAlignment() { return deviceImageInfo_.alignment; }

    hsa_ext_image_t getHsaImageObject() { return hsaImageObject_; }
    const hsa_ext_image_descriptor_t& getHsaImageDescriptor() const { return imageDescriptor_; }
private:
    //! Disable copy constructor
    Image(const Buffer&);

    //! Disable operator=
    Image& operator=(const Buffer&);

    // Setup an interop image
    bool createInteropImage();

    // Free / deregister device memory.
    void destroy();

    void populateImageDescriptor();

    hsa_ext_image_descriptor_t imageDescriptor_;
    hsa_access_permission_t permission_;
    hsa_ext_image_data_info_t deviceImageInfo_;
    hsa_ext_image_t hsaImageObject_;
    hsa_amd_image_descriptor_t* amdImageDesc_;

    const void* hsaImageData_;
};

}
#endif

