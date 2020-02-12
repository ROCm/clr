/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

#ifndef BLIT_HPP_
#define BLIT_HPP_

#include "top.hpp"
#include "platform/command.hpp"
#include "device/device.hpp"

/*! \addtogroup GPU Blit Implementation
 *  @{
 */

//! GPU Blit Manager Implementation
namespace device {

//! Blit Manager Abstraction class
class BlitManager : public amd::HeapObject {
 public:
  //! HW accelerated setup
  union Setup {
    struct {
      uint disableReadBuffer_ : 1;
      uint disableReadBufferRect_ : 1;
      uint disableReadImage_ : 1;
      uint disableWriteBuffer_ : 1;
      uint disableWriteBufferRect_ : 1;
      uint disableWriteImage_ : 1;
      uint disableCopyBuffer_ : 1;
      uint disableCopyBufferRect_ : 1;
      uint disableCopyImageToBuffer_ : 1;
      uint disableCopyBufferToImage_ : 1;
      uint disableCopyImage_ : 1;
      uint disableFillBuffer_ : 1;
      uint disableFillImage_ : 1;
      uint disableCopyBufferToImageOpt_ : 1;
      uint disableHwlCopyBuffer_ : 1;
    };
    uint32_t value_;
    Setup() : value_(0) {}
    void disableAll() { value_ = 0xffffffff; }
  };

 public:
  //! Constructor
  BlitManager(Setup setup = Setup()  //!< Specifies HW accelerated blits
              )
      : setup_(setup), syncOperation_(false) {}

  //! Destructor
  virtual ~BlitManager() {}

  //! Creates HostBlitManager object
  virtual bool create(amd::Device& device) { return true; }

  //! Copies a buffer object to system memory
  virtual bool readBuffer(Memory& srcMemory,           //!< Source memory object
                          void* dstHost,               //!< Destination host memory
                          const amd::Coord3D& origin,  //!< Source origin
                          const amd::Coord3D& size,    //!< Size of the copy region
                          bool entire = false          //!< Entire buffer will be updated
                          ) const = 0;

  //! Copies a buffer object to system memory
  virtual bool readBufferRect(Memory& srcMemory,                //!< Source memory object
                              void* dstHost,                    //!< Destinaiton host memory
                              const amd::BufferRect& bufRect,   //!< Source rectangle
                              const amd::BufferRect& hostRect,  //!< Destination rectangle
                              const amd::Coord3D& size,         //!< Size of the copy region
                              bool entire = false               //!< Entire buffer will be updated
                              ) const = 0;

  //! Copies an image object to system memory
  virtual bool readImage(Memory& srcMemory,           //!< Source memory object
                         void* dstHost,               //!< Destination host memory
                         const amd::Coord3D& origin,  //!< Source origin
                         const amd::Coord3D& size,    //!< Size of the copy region
                         size_t rowPitch,             //!< Row pitch for host memory
                         size_t slicePitch,           //!< Slice pitch for host memory
                         bool entire = false          //!< Entire buffer will be updated
                         ) const = 0;

  //! Copies system memory to a buffer object
  virtual bool writeBuffer(const void* srcHost,         //!< Source host memory
                           Memory& dstMemory,           //!< Destination memory object
                           const amd::Coord3D& origin,  //!< Destination origin
                           const amd::Coord3D& size,    //!< Size of the copy region
                           bool entire = false          //!< Entire buffer will be updated
                           ) const = 0;

  //! Copies system memory to a buffer object
  virtual bool writeBufferRect(const void* srcHost,              //!< Source host memory
                               Memory& dstMemory,                //!< Destination memory object
                               const amd::BufferRect& hostRect,  //!< Destination rectangle
                               const amd::BufferRect& bufRect,   //!< Source rectangle
                               const amd::Coord3D& size,         //!< Size of the copy region
                               bool entire = false               //!< Entire buffer will be updated
                               ) const = 0;

  //! Copies system memory to an image object
  virtual bool writeImage(const void* srcHost,         //!< Source host memory
                          Memory& dstMemory,           //!< Destination memory object
                          const amd::Coord3D& origin,  //!< Destination origin
                          const amd::Coord3D& size,    //!< Size of the copy region
                          size_t rowPitch,             //!< Row pitch for host memory
                          size_t slicePitch,           //!< Slice pitch for host memory
                          bool entire = false          //!< Entire buffer will be updated
                          ) const = 0;

  //! Copies a buffer object to another buffer object
  virtual bool copyBuffer(Memory& srcMemory,              //!< Source memory object
                          Memory& dstMemory,              //!< Destination memory object
                          const amd::Coord3D& srcOrigin,  //!< Source origin
                          const amd::Coord3D& dstOrigin,  //!< Destination origin
                          const amd::Coord3D& size,       //!< Size of the copy region
                          bool entire = false             //!< Entire buffer will be updated
                          ) const = 0;

  //! Copies a buffer object to another buffer object
  virtual bool copyBufferRect(Memory& srcMemory,               //!< Source memory object
                              Memory& dstMemory,               //!< Destination memory object
                              const amd::BufferRect& srcRect,  //!< Source rectangle
                              const amd::BufferRect& dstRect,  //!< Destination rectangle
                              const amd::Coord3D& size,        //!< Size of the copy region
                              bool entire = false              //!< Entire buffer will be updated
                              ) const = 0;

  //! Copies an image object to a buffer object
  virtual bool copyImageToBuffer(Memory& srcMemory,              //!< Source memory object
                                 Memory& dstMemory,              //!< Destination memory object
                                 const amd::Coord3D& srcOrigin,  //!< Source origin
                                 const amd::Coord3D& dstOrigin,  //!< Destination origin
                                 const amd::Coord3D& size,       //!< Size of the copy region
                                 bool entire = false,            //!< Entire buffer will be updated
                                 size_t rowPitch = 0,            //!< Pitch for buffer
                                 size_t slicePitch = 0           //!< Slice for buffer
                                 ) const = 0;

  //! Copies a buffer object to an image object
  virtual bool copyBufferToImage(Memory& srcMemory,              //!< Source memory object
                                 Memory& dstMemory,              //!< Destination memory object
                                 const amd::Coord3D& srcOrigin,  //!< Source origin
                                 const amd::Coord3D& dstOrigin,  //!< Destination origin
                                 const amd::Coord3D& size,       //!< Size of the copy region
                                 bool entire = false,            //!< Entire buffer will be updated
                                 size_t rowPitch = 0,            //!< Pitch for buffer
                                 size_t slicePitch = 0           //!< Slice for buffer
                                 ) const = 0;

  //! Copies an image object to another image object
  virtual bool copyImage(Memory& srcMemory,              //!< Source memory object
                         Memory& dstMemory,              //!< Destination memory object
                         const amd::Coord3D& srcOrigin,  //!< Source origin
                         const amd::Coord3D& dstOrigin,  //!< Destination origin
                         const amd::Coord3D& size,       //!< Size of the copy region
                         bool entire = false             //!< Entire buffer will be updated
                         ) const = 0;

  //! Fills a buffer memory with a pattern data
  virtual bool fillBuffer(Memory& memory,              //!< Memory object to fill with pattern
                          const void* pattern,         //!< Pattern data
                          size_t patternSize,          //!< Pattern size
                          const amd::Coord3D& origin,  //!< Destination origin
                          const amd::Coord3D& size,    //!< Size of the copy region
                          bool entire = false          //!< Entire buffer will be updated
                          ) const = 0;

  //! Fills an image memory with a pattern data
  virtual bool fillImage(Memory& dstMemory,           //!< Memory object to fill with pattern
                         const void* pattern,         //!< Pattern data
                         const amd::Coord3D& origin,  //!< Destination origin
                         const amd::Coord3D& size,    //!< Size of the copy region
                         bool entire = false          //!< Entire buffer will be updated
                         ) const = 0;

  //! Enables synchronization on blit operations
  void enableSynchronization() { syncOperation_ = true; }
  
  //! Returns Xfer queue lock
  virtual amd::Monitor* lockXfer() const { return nullptr; }

 protected:
  const Setup setup_;   //!< HW accelerated blit requested
  bool syncOperation_;  //!< Blit operations are synchronized

 private:
  //! Disable copy constructor
  BlitManager(const BlitManager&);

  //! Disable operator=
  BlitManager& operator=(const BlitManager&);
};

//! Host Blit Manager
class HostBlitManager : public device::BlitManager {
 public:
  //! Constructor
  HostBlitManager(VirtualDevice& vdev,   //!< Virtual GPU to be used for blits
                  Setup setup = Setup()  //!< Specifies HW accelerated blits
                  );

  //! Destructor
  virtual ~HostBlitManager() {}

  //! Creates HostBlitManager object
  virtual bool create(amd::Device& device) { return true; }

  //! Copies a buffer object to system memory
  virtual bool readBuffer(device::Memory& srcMemory,   //!< Source memory object
                          void* dstHost,               //!< Destination host memory
                          const amd::Coord3D& origin,  //!< Source origin
                          const amd::Coord3D& size,    //!< Size of the copy region
                          bool entire = false          //!< Entire buffer will be updated
                          ) const;

  //! Copies a buffer object to system memory
  virtual bool readBufferRect(device::Memory& srcMemory,        //!< Source memory object
                              void* dstHost,                    //!< Destinaiton host memory
                              const amd::BufferRect& bufRect,   //!< Source rectangle
                              const amd::BufferRect& hostRect,  //!< Destination rectangle
                              const amd::Coord3D& size,         //!< Size of the copy region
                              bool entire = false               //!< Entire buffer will be updated
                              ) const;

  //! Copies an image object to system memory
  virtual bool readImage(device::Memory& srcMemory,   //!< Source memory object
                         void* dstHost,               //!< Destination host memory
                         const amd::Coord3D& origin,  //!< Source origin
                         const amd::Coord3D& size,    //!< Size of the copy region
                         size_t rowPitch,             //!< Row pitch for host memory
                         size_t slicePitch,           //!< Slice pitch for host memory
                         bool entire = false          //!< Entire buffer will be updated
                         ) const;

  //! Copies system memory to a buffer object
  virtual bool writeBuffer(const void* srcHost,         //!< Source host memory
                           device::Memory& dstMemory,   //!< Destination memory object
                           const amd::Coord3D& origin,  //!< Destination origin
                           const amd::Coord3D& size,    //!< Size of the copy region
                           bool entire = false          //!< Entire buffer will be updated
                           ) const;

  //! Copies system memory to a buffer object
  virtual bool writeBufferRect(const void* srcHost,              //!< Source host memory
                               device::Memory& dstMemory,        //!< Destination memory object
                               const amd::BufferRect& hostRect,  //!< Destination rectangle
                               const amd::BufferRect& bufRect,   //!< Source rectangle
                               const amd::Coord3D& size,         //!< Size of the copy region
                               bool entire = false               //!< Entire buffer will be updated
                               ) const;

  //! Copies system memory to an image object
  virtual bool writeImage(const void* srcHost,         //!< Source host memory
                          device::Memory& dstMemory,   //!< Destination memory object
                          const amd::Coord3D& origin,  //!< Destination origin
                          const amd::Coord3D& size,    //!< Size of the copy region
                          size_t rowPitch,             //!< Row pitch for host memory
                          size_t slicePitch,           //!< Slice pitch for host memory
                          bool entire = false          //!< Entire buffer will be updated
                          ) const;

  //! Copies a buffer object to another buffer object
  virtual bool copyBuffer(device::Memory& srcMemory,      //!< Source memory object
                          device::Memory& dstMemory,      //!< Destination memory object
                          const amd::Coord3D& srcOrigin,  //!< Source origin
                          const amd::Coord3D& dstOrigin,  //!< Destination origin
                          const amd::Coord3D& size,       //!< Size of the copy region
                          bool entire = false             //!< Entire buffer will be updated
                          ) const;

  //! Copies a buffer object to another buffer object
  virtual bool copyBufferRect(device::Memory& srcMemory,       //!< Source memory object
                              device::Memory& dstMemory,       //!< Destination memory object
                              const amd::BufferRect& srcRect,  //!< Source rectangle
                              const amd::BufferRect& dstRect,  //!< Destination rectangle
                              const amd::Coord3D& size,        //!< Size of the copy region
                              bool entire = false              //!< Entire buffer will be updated
                              ) const;

  //! Copies an image object to a buffer object
  virtual bool copyImageToBuffer(device::Memory& srcMemory,      //!< Source memory object
                                 device::Memory& dstMemory,      //!< Destination memory object
                                 const amd::Coord3D& srcOrigin,  //!< Source origin
                                 const amd::Coord3D& dstOrigin,  //!< Destination origin
                                 const amd::Coord3D& size,       //!< Size of the copy region
                                 bool entire = false,            //!< Entire buffer will be updated
                                 size_t rowPitch = 0,            //!< Pitch for buffer
                                 size_t slicePitch = 0           //!< Slice for buffer
                                 ) const;

  //! Copies a buffer object to an image object
  virtual bool copyBufferToImage(device::Memory& srcMemory,      //!< Source memory object
                                 device::Memory& dstMemory,      //!< Destination memory object
                                 const amd::Coord3D& srcOrigin,  //!< Source origin
                                 const amd::Coord3D& dstOrigin,  //!< Destination origin
                                 const amd::Coord3D& size,       //!< Size of the copy region
                                 bool entire = false,            //!< Entire buffer will be updated
                                 size_t rowPitch = 0,            //!< Pitch for buffer
                                 size_t slicePitch = 0           //!< Slice for buffer
                                 ) const;

  //! Copies an image object to another image object
  virtual bool copyImage(device::Memory& srcMemory,      //!< Source memory object
                         device::Memory& dstMemory,      //!< Destination memory object
                         const amd::Coord3D& srcOrigin,  //!< Source origin
                         const amd::Coord3D& dstOrigin,  //!< Destination origin
                         const amd::Coord3D& size,       //!< Size of the copy region
                         bool entire = false             //!< Entire buffer will be updated
                         ) const;

  //! Fills a buffer memory with a pattern data
  virtual bool fillBuffer(device::Memory& memory,      //!< Memory object to fill with pattern
                          const void* pattern,         //!< Pattern data
                          size_t patternSize,          //!< Pattern size
                          const amd::Coord3D& origin,  //!< Destination origin
                          const amd::Coord3D& size,    //!< Size of the copy region
                          bool entire = false          //!< Entire buffer will be updated
                          ) const;

  //! Fills an image memory with a pattern data
  virtual bool fillImage(device::Memory& dstMemory,   //!< Memory object to fill with pattern
                         const void* pattern,         //!< Pattern data
                         const amd::Coord3D& origin,  //!< Destination origin
                         const amd::Coord3D& size,    //!< Size of the copy region
                         bool entire = false          //!< Entire buffer will be updated
                         ) const;

  uint32_t sRGBmap(float fc) const;

 protected:
  VirtualDevice& vDev_;     //!< Virtual device object
  const amd::Device& dev_;  //!< Physical device

 private:
  //! Disable copy constructor
  HostBlitManager(const HostBlitManager&);

  //! Disable operator=
  HostBlitManager& operator=(const HostBlitManager&);
};

/*@}*/} // namespace device

#endif /*BLIT_HPP_*/
