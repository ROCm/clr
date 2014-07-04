//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef HSABLIT_HPP_
#define HSABLIT_HPP_

#include "top.hpp"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "device/device.hpp"
#include "device/blit.hpp"

/*! \addtogroup HSA Blit Implementation
 *  @{
 */

//! HSA Blit Manager Implementation
namespace oclhsa {

class Device;
class Kernel;
class Memory;
class VirtualGPU;

//! DMA Blit Manager
class HsaBlitManager : public device::HostBlitManager
{
public:
    //! Constructor
    HsaBlitManager(
        device::VirtualDevice& vdev,        //!< Virtual GPU to be used for blits
        Setup setup = Setup() //!< Specifies HW accelerated blits
        );

    //! Destructor
    virtual ~HsaBlitManager() { }

    //! Creates HostBlitManager object
    virtual bool create(amd::Device& device) { return true; }

    //! Copies a buffer object to system memory
    virtual bool readBuffer(
        device::Memory& srcMemory,      //!< Source memory object
        void*       dstHost,            //!< Destination host memory
        const amd::Coord3D& origin,     //!< Source origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies a buffer object to system memory
    virtual bool readBufferRect(
        device::Memory& srcMemory,          //!< Source memory object
        void*       dstHost,                //!< Destinaiton host memory
        const amd::BufferRect&  bufRect,    //!< Source rectangle
        const amd::BufferRect&  hostRect,   //!< Destination rectangle
        const amd::Coord3D&     size,       //!< Size of the copy region
        bool        entire = false          //!< Entire buffer will be updated
        ) const;

    //! Copies an image object to system memory
    virtual bool readImage(
        device::Memory& srcMemory,      //!< Source memory object
        void*       dstHost,            //!< Destination host memory
        const amd::Coord3D& origin,     //!< Source origin
        const amd::Coord3D& size,       //!< Size of the copy region
        size_t      rowPitch,           //!< Row pitch for host memory
        size_t      slicePitch,         //!< Slice pitch for host memory
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies system memory to a buffer object
    virtual bool writeBuffer(
        const void* srcHost,            //!< Source host memory
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies system memory to a buffer object
    virtual bool writeBufferRect(
        const void* srcHost,                //!< Source host memory
        device::Memory& dstMemory,          //!< Destination memory object
        const amd::BufferRect&  hostRect,   //!< Destination rectangle
        const amd::BufferRect&  bufRect,    //!< Source rectangle
        const amd::Coord3D&     size,       //!< Size of the copy region
        bool        entire = false          //!< Entire buffer will be updated
        ) const;

    //! Copies system memory to an image object
    virtual bool writeImage(
        const void* srcHost,            //!< Source host memory
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        size_t      rowPitch,           //!< Row pitch for host memory
        size_t      slicePitch,         //!< Slice pitch for host memory
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies a buffer object to another buffer object
    virtual bool copyBuffer(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies a buffer object to another buffer object
    virtual bool copyBufferRect(
        device::Memory& srcMemory,          //!< Source memory object
        device::Memory& dstMemory,          //!< Destination memory object
        const amd::BufferRect&  srcRect,    //!< Source rectangle
        const amd::BufferRect&  dstRect,    //!< Destination rectangle
        const amd::Coord3D&     size,       //!< Size of the copy region
        bool        entire = false          //!< Entire buffer will be updated
        ) const;

    //! Copies an image object to a buffer object
    virtual bool copyImageToBuffer(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false,     //!< Entire buffer will be updated
        size_t      rowPitch = 0,       //!< Pitch for buffer
        size_t      slicePitch = 0      //!< Slice for buffer
        ) const;

    //! Copies a buffer object to an image object
    virtual bool copyBufferToImage(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false,     //!< Entire buffer will be updated
        size_t      rowPitch = 0,       //!< Pitch for buffer
        size_t      slicePitch = 0      //!< Slice for buffer
        ) const;

    //! Copies an image object to another image object
    virtual bool copyImage(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Fills a buffer memory with a pattern data
    virtual bool fillBuffer(
        device::Memory& memory,         //!< Memory object to fill with pattern
        const void* pattern,            //!< Pattern data
        size_t      patternSize,        //!< Pattern size
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Fills an image memory with a pattern data
    virtual bool fillImage(
        device::Memory& dstMemory,      //!< Memory object to fill with pattern
        const void* pattern,            //!< Pattern data
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

protected:
    //! Returns the virtual GPU object
    VirtualGPU& gpu() const { return static_cast<VirtualGPU&>(vDev_); }

private:
    //! Disable copy constructor
    HsaBlitManager(const HsaBlitManager&);

    //! Disable operator=
    HsaBlitManager& operator=(const HsaBlitManager&);

    bool importExportImage(
        uint8_t* dst,
        const uint8_t* src,
        const amd::Coord3D& dstOffset,
        size_t dstRowPitch,
        size_t dstSlicePitch,
        const amd::Coord3D& srcOffset,
        size_t srcRowPitch,
        size_t srcSlicePitch,
        const amd::Coord3D& sizeToCopy,
        size_t elementSize) const;
};

//! Kernel Blit Manager
class KernelBlitManager : public HsaBlitManager
{
public:
    enum {
        BlitCopyImage = 0,
        BlitCopyImage1DA,
        BlitCopyImageToBuffer,
        BlitCopyBufferToImage,
        BlitCopyBufferRect,
        BlitCopyBufferRectAligned,
        BlitCopyBuffer,
        BlitCopyBufferAligned,
        FillBuffer,
        FillImage,
        BlitTotal
    };

    //! Constructor
    KernelBlitManager(
        device::VirtualDevice& vdev,        //!< Virtual GPU to be used for blits
        Setup setup = Setup() //!< Specifies HW accelerated blits
        );

    //! Destructor
    virtual ~KernelBlitManager();

    //! Creates HostBlitManager object
    virtual bool create(amd::Device& device);

    //! Copies a buffer object to system memory
    virtual bool readBuffer(
        device::Memory& srcMemory,      //!< Source memory object
        void*       dstHost,            //!< Destination host memory
        const amd::Coord3D& origin,     //!< Source origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies a buffer object to system memory
    virtual bool readBufferRect(
        device::Memory& srcMemory,          //!< Source memory object
        void*       dstHost,                //!< Destinaiton host memory
        const amd::BufferRect&  bufRect,    //!< Source rectangle
        const amd::BufferRect&  hostRect,   //!< Destination rectangle
        const amd::Coord3D&     size,       //!< Size of the copy region
        bool        entire = false          //!< Entire buffer will be updated
        ) const;

    //! Copies an image object to system memory
    virtual bool readImage(
        device::Memory& srcMemory,      //!< Source memory object
        void*       dstHost,            //!< Destination host memory
        const amd::Coord3D& origin,     //!< Source origin
        const amd::Coord3D& size,       //!< Size of the copy region
        size_t      rowPitch,           //!< Row pitch for host memory
        size_t      slicePitch,         //!< Slice pitch for host memory
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies system memory to a buffer object
    virtual bool writeBuffer(
        const void* srcHost,            //!< Source host memory
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies system memory to a buffer object
    virtual bool writeBufferRect(
        const void* srcHost,                //!< Source host memory
        device::Memory& dstMemory,          //!< Destination memory object
        const amd::BufferRect&  hostRect,   //!< Destination rectangle
        const amd::BufferRect&  bufRect,    //!< Source rectangle
        const amd::Coord3D&     size,       //!< Size of the copy region
        bool        entire = false          //!< Entire buffer will be updated
        ) const;

    //! Copies system memory to an image object
    virtual bool writeImage(
        const void* srcHost,            //!< Source host memory
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        size_t      rowPitch,           //!< Row pitch for host memory
        size_t      slicePitch,         //!< Slice pitch for host memory
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies a buffer object to another buffer object
    virtual bool copyBuffer(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Copies a buffer object to another buffer object
    virtual bool copyBufferRect(
        device::Memory& srcMemory,          //!< Source memory object
        device::Memory& dstMemory,          //!< Destination memory object
        const amd::BufferRect&  srcRect,    //!< Source rectangle
        const amd::BufferRect&  dstRect,    //!< Destination rectangle
        const amd::Coord3D&     size,       //!< Size of the copy region
        bool        entire = false          //!< Entire buffer will be updated
        ) const;

    //! Copies an image object to a buffer object
    virtual bool copyImageToBuffer(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false,     //!< Entire buffer will be updated
        size_t      rowPitch = 0,       //!< Pitch for buffer
        size_t      slicePitch = 0      //!< Slice for buffer
        ) const;

    //! Copies a buffer object to an image object
    virtual bool copyBufferToImage(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false,     //!< Entire buffer will be updated
        size_t      rowPitch = 0,       //!< Pitch for buffer
        size_t      slicePitch = 0      //!< Slice for buffer
        ) const;

    //! Copies an image object to another image object
    virtual bool copyImage(
        device::Memory& srcMemory,      //!< Source memory object
        device::Memory& dstMemory,      //!< Destination memory object
        const amd::Coord3D& srcOrigin,  //!< Source origin
        const amd::Coord3D& dstOrigin,  //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Fills a buffer memory with a pattern data
    virtual bool fillBuffer(
        device::Memory& memory,         //!< Memory object to fill with pattern
        const void* pattern,            //!< Pattern data
        size_t      patternSize,        //!< Pattern size
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

    //! Fills an image memory with a pattern data
    virtual bool fillImage(
        device::Memory& dstMemory,      //!< Memory object to fill with pattern
        const void* pattern,            //!< Pattern data
        const amd::Coord3D& origin,     //!< Destination origin
        const amd::Coord3D& size,       //!< Size of the copy region
        bool        entire = false      //!< Entire buffer will be updated
        ) const;

private:
    //! Disable copy constructor
    KernelBlitManager(const KernelBlitManager&);

    //! Disable operator=
    KernelBlitManager& operator=(const KernelBlitManager&);

    //! Creates a program for all blit operations
    bool createProgram(
        Device& device                  //!< Device object
        );

    amd::Image::Format filterFormat(amd::Image::Format oldFormat) const;

    device::Memory *createImageView(
        device::Memory &parent,
        amd::Image::Format newFormat) const;

    amd::Context *context_;              //!< A dummy context
    amd::Program *program_;              //!< GPU program obejct
    amd::Kernel *kernels_[BlitTotal];    //!< GPU kernels for blit
};

static const char* BlitName[KernelBlitManager::BlitTotal] = {
    "copyImage",
    "copyImage1DA",
    "copyImageToBuffer",
    "copyBufferToImage",
    "copyBufferRect",
    "copyBufferRectAligned",
    "copyBuffer",
    "copyBufferAligned",
    "fillBuffer",
    "fillImage"
    };

/*@}*/
} // namespace oclhsa

#endif /*HSABLIT_HPP_*/
