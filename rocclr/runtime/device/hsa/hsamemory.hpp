#ifndef HSAMEMORY_HPP_
#define HSAMEMORY_HPP_

#include "top.hpp"
#include "platform/memory.hpp"
#include "utils/debug.hpp"
#include "hsadevice.hpp"
#include "services.h"
#ifdef _WIN32
#include "amdocl/cl_d3d11_amd.hpp"
#endif
#include "amdocl/cl_gl_amd.hpp"
#include "hsainterop.h"

namespace oclhsa {

enum InteropType {
    InteropNone = 0,
    InteropD3D9 = 1,
    InteropD3D10 = 2,
    InteropD3D11 = 3,
    InteropGL = 4
};

class Memory : public device::Memory {
 public:
  Memory(const oclhsa::Device &dev, amd::Memory &owner);

  virtual ~Memory();

  // Getter for deviceMemory_.
  void *getDeviceMemory() const { return deviceMemory_; }

  // Gets a pointer to a region of host-visible memory for use as the target
  // of an indirect map for a given memory object
  virtual void *allocMapTarget(const amd::Coord3D &origin,
                               const amd::Coord3D &region,
                               size_t *rowPitch,
                               size_t *slicePitch);

  // Create device memory according to OpenCL memory flag.
  virtual bool create() = 0;
  virtual bool createInterop() = 0;

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

  bool processGLResource (GLResourceOP operation) { return true;}

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

  bool isHsaLocalMemory() const;

  // Accessors for indirect map memory object
  amd::Memory *mapMemory() const { return mapMemory_; }

 protected:
  bool allocateMapMemory(size_t allocationSize);

  void freeMapMemory();

  // Decrement map count
  virtual void decIndMapCount();

  // Free / deregister device memory.
  virtual void destroy() = 0;

  //This function is called in the destructor ~Buffer() and ~Image(),
  //since InteropObject belonging to owner() is destroyed before
  //the destructor is called, we use the cached values of
  //interopType and Resource in this function.
  virtual void destroyInterop();

  // Pointer to the device associated with this memory object.
  const oclhsa::Device &dev_;

  // Pointer to the device memory. This could be in system or device local mem.
  void* deviceMemory_;

  InteropType interopType_;
#ifdef _WIN32
  ID3D10Resource* d3d10Resource_;
  ID3D11Resource* d3d11Resource_;
#endif
  HsaGLResource glResource_;

 private:
  // Disable copy constructor
  Memory(const Memory &);

  // Disable operator=
  Memory &operator=(const Memory &);
};



class Buffer : public oclhsa::Memory {
 public:
    Buffer(const oclhsa::Device &dev, amd::Memory &owner);

    virtual ~Buffer();

    // Create device memory according to OpenCL memory flag.
    virtual bool create();

    // Recreate the device memory using new size and alignment.
    bool recreate(size_t newSize, size_t newAlignment, bool forceSystem);

    //! Create a interop memory
    bool createInterop();

 private:
    // Disable copy constructor
    Buffer(const Buffer &);

    // Disable operator=
    Buffer &operator=(const Buffer &);

    // Free / deregister device memory.
    void destroy();
};

class Image : public oclhsa::Memory
{
public:
    Image(const oclhsa::Device& dev, amd::Memory& owner);

    virtual ~Image();

    //! Create device memory according to OpenCL memory flag.
    virtual bool create();

    //! Create an image view
    bool createView(Image &image);

    virtual bool createInterop();

    //! Gets a pointer to a region of host-visible memory for use as the target
    //! of an indirect map for a given memory object
    virtual void* allocMapTarget(const amd::Coord3D& origin,
        const amd::Coord3D& region,
        size_t* rowPitch,
        size_t* slicePitch);

    size_t getDeviceRowPitchSize() { return deviceImageInfo_.rowPitchInBytes; }
    size_t getDeviceSlicePitchSize() { return deviceImageInfo_.slicePitchInBytes; }
    size_t getDeviceDataSize() { return deviceImageInfo_.imageSizeInBytes; }
    size_t getDeviceDataAlignment() { return deviceImageInfo_.imageAlignmentInBytes; }

    void* getHsaImageObjectAddress() { return &hsaImageObject_[0];}
    size_t getHsaImageObjectSizeInBytes() {return sizeof(hsaImageObject_); }

private:
    //! Disable copy constructor
    Image(const Buffer&);

    //! Disable operator=
    Image& operator=(const Buffer&);

    // Free / deregister device memory.
    void destroy();

    void populateImageDescriptor();

    HsaImageDescriptor imageDescriptor_;
    HsaDeviceImageInfo deviceImageInfo_;
    uint8_t hsaImageObject_[HSA_IMAGE_OBJECT_SIZE];
};

}
#endif
