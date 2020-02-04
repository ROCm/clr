/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#ifndef GPUMEMORY_HPP_
#define GPUMEMORY_HPP_

#include "top.hpp"
#include "thread/atomic.hpp"
#include "device/gpu/gpuresource.hpp"
#include <map>

/*! \addtogroup GPU
 *  @{
 */
namespace device {
class Memory;
}

//! GPU Device Implementation
namespace gpu {

class Device;
class Heap;
class Resource;
class Memory;
class VirtualGPU;

//! GPU memory object.
//  Wrapper that can contain a heap block or an interop buffer/image.
class Memory : public device::Memory, public Resource {
 public:
  enum InteropType {
    InteropNone = 0,         //!< None interop memory
    InteropHwEmulation = 1,  //!< Uses HW emulaiton with calMemCopy
    InteropDirectAccess = 2  //!< Uses direct access to the interop surface
  };

  //! Constructor (with owner)
  Memory(const Device& gpuDev, amd::Memory& owner, size_t size = 0);

  //! Constructor (nonfat version for local scratch mem use without heap block)
  Memory(const Device& gpuDev, size_t size);

  //! Constructor memory for buffer (without global heap allocaton)
  Memory(const Device& gpuDev,  //!< GPU device object
         amd::Memory& owner,    //!< Abstraction layer memory object
         size_t width,          //!< Memory width
         cmSurfFmt format       //!< CAL format
         );

  //! Constructor memory for buffer (without global heap allocaton)
  Memory(const Device& gpuDev,  //!< GPU device object
         size_t size,           //!< Memory object size
         size_t width,          //!< Memory width
         cmSurfFmt format       //!< CAL format
         );

  //! Constructor memory for images (without global heap allocaton)
  Memory(const Device& gpuDev,          //!< GPU device object
         amd::Memory& owner,            //!< Abstraction layer memory object
         size_t width,                  //!< Allocated memory width
         size_t height,                 //!< Allocated memory height
         size_t depth,                  //!< Allocated memory depth
         cmSurfFmt format,              //!< Memory format
         gslChannelOrder chOrder,       //!< Channel order
         cl_mem_object_type imageType,  //!< CL image type
         uint mipLevels                 //!< The number of mip levels
         );

  //! Constructor memory for images (without global heap allocaton)
  Memory(const Device& gpuDev,          //!< GPU device object
         size_t size,                   //!< Memory object size
         size_t width,                  //!< Allocated memory width
         size_t height,                 //!< Allocated memory height
         size_t depth,                  //!< Allocated memory depth
         cmSurfFmt format,              //!< Memory format
         gslChannelOrder chOrder,       //!< Channel order
         cl_mem_object_type imageType,  //!< CL image type
         uint mipLevels                 //!< The number of mip levels
         );

  //! Default destructor
  ~Memory();

  //! Creates the interop memory
  bool createInterop(InteropType type  //!< The interop type
                     );

  //! Overloads the resource create method
  virtual bool create(Resource::MemoryType memType,          //!< Memory type
                      Resource::CreateParams* params = NULL  //!< Prameters for create
                      );

  //! Allocate memory for API-level maps
  virtual void* allocMapTarget(const amd::Coord3D& origin,  //!< The map location in memory
                               const amd::Coord3D& region,  //!< The map region in memory
                               uint mapFlags,               //!< Map flags
                               size_t* rowPitch = NULL,     //!< Row pitch for the mapped memory
                               size_t* slicePitch = NULL    //!< Slice for the mapped memory
                               );

  //! Pins system memory associated with this memory object
  virtual bool pinSystemMemory(void* hostPtr,  //!< System memory address
                               size_t size     //!< Size of allocated system memory
                               );

  //! Releases indirect map surface
  virtual void releaseIndirectMap() { decIndMapCount(); }

  //! Map the device memory to CPU visible
  virtual void* cpuMap(device::VirtualDevice& vDev,  //!< Virtual device for map operaiton
                       uint flags = 0,               //!< flags for the map operation
                       // Optimization for multilayer map/unmap
                       uint startLayer = 0,       //!< Start layer for multilayer map
                       uint numLayers = 0,        //!< End layer for multilayer map
                       size_t* rowPitch = NULL,   //!< Row pitch for the device memory
                       size_t* slicePitch = NULL  //!< Slice pitch for the device memory
                       );

  //! Unmap the device memory
  virtual void cpuUnmap(device::VirtualDevice& vDev  //!< Virtual device for unmap operaiton
                        );

  //! Updates device memory from the owner's host allocation
  void syncCacheFromHost(VirtualGPU& gpu,  //!< Virtual GPU device object
                         //! Synchronization flags
                         device::Memory::SyncFlags syncFlags = device::Memory::SyncFlags());

  //! Updates the owner's host allocation from device memory
  virtual void syncHostFromCache(
      //! Synchronization flags
      device::Memory::SyncFlags syncFlags = device::Memory::SyncFlags());

  //! Creates a view from current resource
  virtual Memory* createBufferView(
      amd::Memory& subBufferOwner  //!< The abstraction layer subbuf owner
      );

  virtual uint64_t virtualAddress() const override { return (vmAddress() + pinOffset()); }

  //! Allocates host memory for synchronization with MGPU context
  void mgpuCacheWriteBack();

  //! Accessors for indirect map memory object
  Memory* mapMemory() const;

  //! Returns the interop memory for this memory object
  Memory* interop() const { return interopMemory_; }

  //! Gets interop type for this memory object
  InteropType interopType() const { return interopType_; }

  //! Sets interop type for this memory object
  void setInteropType(InteropType type) { interopType_ = type; }

  //! Set the owner
  void setOwner(amd::Memory* owner) { owner_ = owner; }

  // Decompress GL depth-stencil/MSAA resources for CL access
  // Invalidates any FBOs the resource may be bound to, otherwise the GL driver may crash.
  virtual bool processGLResource(GLResourceOP operation);

  //! Returns the interop resource for this memory object
  const Memory* parent() const { return parent_; }

  //! Returns TRUE if direct map is acceaptable. The method detects
  //! forced USWC memory on APU and will cause a switch to
  //! indirect map for allocations with a possibility of host read
  bool isDirectMap() {
    return (isCacheable() || !isHostMemDirectAccess() ||
            (owner()->getMemFlags() &
             (CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY)));
  }

 protected:
  //! Decrement map count
  void decIndMapCount();

  //! Initialize the object members
  void init();

 private:
  //! Disable copy constructor
  Memory(const Memory&);

  //! Disable operator=
  Memory& operator=(const Memory&);

  InteropType interopType_;  //!< Interop type
  Memory* interopMemory_;    //!< interop memory

  Memory* pinnedMemory_;  //!< Memory used as pinned system memory
  const Memory* parent_;  //!< Parent memory object
};

class Buffer : public gpu::Memory {
 public:
  //! Buffer constructor
  Buffer(const Device& gpuDev,  //!< GPU device object
         amd::Memory& owner,    //!< Abstraction layer memory object
         size_t size            //!< Buffer size
         )
      : gpu::Memory(gpuDev, owner, amd::alignUp(size, ElementSize) / ElementSize, ElementType) {}

  //! Creates a view from current resource
  virtual Memory* createBufferView(
      amd::Memory& subBufferOwner  //!< The abstraction layer subbuf owner
      ) const;

 private:
  //! Disable copy constructor
  Buffer(const Buffer&);

  //! Disable operator=
  Buffer& operator=(const Buffer&);

  //! The size of buffer element in bytes
  static const size_t ElementSize = 4;

  //! The type of buffer element
  static const cmSurfFmt ElementType = CM_SURF_FMT_R32I;
};

class Image : public gpu::Memory {
 public:
  //! Image constructor
  Image(const Device& gpuDev,          //!< GPU device object
        amd::Memory& owner,            //!< Abstraction layer memory object
        size_t width,                  //!< Allocated memory width
        size_t height,                 //!< Allocated memory height
        size_t depth,                  //!< Allocated memory depth
        cmSurfFmt format,              //!< Memory format
        gslChannelOrder chOrder,       //!< Channel order
        cl_mem_object_type imageType,  //!< CL image type
        uint mipLevels                 //!< The number of mip levels
        )
      : gpu::Memory(gpuDev, owner, width, height, depth, format, chOrder, imageType, mipLevels) {}

  //! Image constructor
  Image(const Device& gpuDev,          //!< GPU device object
        size_t size,                   //!< Memory size
        size_t width,                  //!< Allocated memory width
        size_t height,                 //!< Allocated memory height
        size_t depth,                  //!< Allocated memory depth
        cmSurfFmt format,              //!< Memory format
        gslChannelOrder chOrder,       //!< Channel order
        cl_mem_object_type imageType,  //!< CL image type
        uint mipLevels                 //!< The number of mip levels
        )
      : gpu::Memory(gpuDev, size, width, height, depth, format, chOrder, imageType, mipLevels) {}

  //! Allocate memory for API-level maps
  virtual void* allocMapTarget(const amd::Coord3D& origin,  //!< The map location in memory
                               const amd::Coord3D& region,  //!< The map region in memory
                               uint mapFlags,               //!< Map flags
                               size_t* rowPitch = NULL,     //!< Row pitch for the mapped memory
                               size_t* slicePitch = NULL    //!< Slice for the mapped memory
                               );

 private:
  //! Disable copy constructor
  Image(const Image&);

  //! Disable operator=
  Image& operator=(const Image&);
};

}  // namespace gpu

#endif  // GPUMEMORY_HPP_
