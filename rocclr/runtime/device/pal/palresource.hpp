//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "platform/command.hpp"
#include "platform/program.hpp"
#include "device/pal/paldefs.hpp"

//! \namespace pal PAL Resource Implementation
namespace pal {

class Device;
class VirtualGPU;

/*! \addtogroup PAL PAL Resource Implementation
 *  @{
 */

class GpuMemoryReference : public amd::ReferenceCountedObject
{
public:
    static GpuMemoryReference* Create(
        const Device&   dev,
        const Pal::GpuMemoryCreateInfo& createInfo);

    static GpuMemoryReference* Create(
        const Device&   dev,
        const Pal::PinnedGpuMemoryCreateInfo& createInfo);

    static GpuMemoryReference* Create(
        const Device&   dev,
        const Pal::ExternalResourceOpenInfo& openInfo);

    static GpuMemoryReference* Create(
        const Device&   dev,
        const Pal::ExternalImageOpenInfo& openInfo,
        Pal::ImageCreateInfo* imgCreateInfo,
        Pal::IImage**   image);

    //! Default constructor
    GpuMemoryReference();

    //! Get PAL memory object
    Pal::IGpuMemory* iMem() const { return gpuMem_; }

    Pal::IGpuMemory*    gpuMem_;        //!< PAL GPU memory object
    void*               cpuAddress_;    //!< CPU address of this memory

protected:
    //! Default destructor
    ~GpuMemoryReference();

private:
    //! Disable copy constructor
    GpuMemoryReference(const GpuMemoryReference&);

    //! Disable operator=
    GpuMemoryReference& operator=(const GpuMemoryReference&);
};

//! GPU resource
class Resource : public amd::HeapObject
{
public:
    enum InteropType {
        InteropTypeless         = 0,
        InteropVertexBuffer,
        InteropIndexBuffer,
        InteropRenderBuffer,
        InteropTexture,
        InteropTextureViewLevel,
        InteropTextureViewCube,
        InteropSurface
    };

    struct CreateParams : public amd::StackObject {
        amd::Memory*    owner_;     //!< Resource's owner
        VirtualGPU*     gpu_;       //!< Resource won't be shared between multiple queues
        CreateParams(): owner_(NULL), gpu_(NULL) {}
    };

    struct PinnedParams : public CreateParams {
        const amd::HostMemoryReference* hostMemRef_;//!< System memory pointer for pinning
        size_t          size_;      //!< System memory size
    };

    struct ViewParams : public CreateParams {
        size_t          offset_;    //!< Alias resource offset
        size_t          size_;      //!< Alias resource size
        const Resource* resource_;  //!< Parent resource for the view creation
        const void*     memory_;
    };

    struct ImageViewParams : public CreateParams {
        size_t          level_;     //!< Image mip level for a new view
        size_t          layer_;     //!< Image layer for a new view
        const Resource* resource_;  //!< Parent resource for the view creation
        const void*     memory_;
    };

    struct ImageBufferParams : public CreateParams {
        const Resource* resource_;  //!< Parent resource for the image creation
        const void*     memory_;
    };

    struct OGLInteropParams : public CreateParams {
        InteropType type_;      //!< OGL resource type
        uint        handle_;    //!< OGL resource handle
        uint        mipLevel_;  //!< Texture mip level
        uint        layer_;     //!< Texture layer
        void*       glPlatformContext_;
        void*       glDeviceContext_;
        uint        flags_;
    };

#ifdef _WIN32
    struct D3DInteropParams : public CreateParams {
        InteropType type_;      //!< D3D resource type
        void*       iDirect3D_; //!< D3D resource interface object
        void*       handle_;    //!< D3D resource handle
        uint        mipLevel_;  //!< Texture mip level
        int         layer_;     //!< Texture layer
        uint        misc;       //!< miscellaneous cases
    };
#endif // _WIN32

    //! Resource memory
    enum MemoryType
    {
        Empty   = 0x0,      //!< resource is empty
        Local,              //!< resource in local memory
        Persistent,         //!< resource in persistent memory
        Remote,             //!< resource in nonlocal memory
        RemoteUSWC,         //!< resource in nonlocal memory
        Pinned,             //!< resource in pinned system memory
        View,               //!< resource is an alias
        OGLInterop,         //!< resource is an OGL memory object
        D3D10Interop,       //!< resource is a D3D10 memory object
        D3D11Interop,       //!< resource is a D3D11 memory object
        ImageView,          //!< resource is a view to some image
        ImageBuffer,        //!< resource is an image view of a buffer
        BusAddressable,     //!< resource is a bus addressable memory
        ExternalPhysical,   //!< resource is an external physical memory
        D3D9Interop,        //!< resource is a D3D9 memory object
        Scratch,            //!< resource is scratch memory
        Shader,             //!< resource is a shader
    };

    //! Resource map flags
    enum MapFlags
    {
        Discard     = 0x00000001,   //!< discard lock
        NoOverwrite = 0x00000002,   //!< lock with no overwrite
        ReadOnly    = 0x00000004,   //!< lock for read only operation
        WriteOnly   = 0x00000008,   //!< lock for write only operation
        NoWait      = 0x00000010,   //!< lock with no wait
    };

    //! Resource descriptor
    struct Descriptor : public amd::HeapObject
    {
        MemoryType  type_;          //!< Memory type
        size_t      width_;         //!< Resource width
        size_t      height_;        //!< Resource height
        size_t      depth_;         //!< Resource depth
        uint        baseLevel_;     //!< The base level for the view
        uint        mipLevels_;     //!< Number of mip levels 
        uint        flags_;         //!< Resource flags, used in creation
        size_t      pitch_;         //!< Resource pitch, valid if locked
        size_t      slice_;         //!< Resource slice, valid if locked
        cl_image_format format_;    //!< CL image format
        cl_mem_object_type topology_;//!< CL mem object type
        union {
            struct {
                uint    dimSize_        : 2;    //!< Dimension size
                uint    cardMemory_     : 1;    //!< GSL resource is in video memory
                uint    imageArray_     : 1;    //!< GSL resource is an array of images
                uint    buffer_         : 1;    //!< GSL resource is a buffer
                uint    tiled_          : 1;    //!< GSL resource is tiled
                uint    SVMRes_         : 1;    //!< SVM flag to the cal resource
                uint    scratch_        : 1;    //!< Scratch buffer
                uint    isAllocExecute_ : 1;    //!< SVM resource allocation attribute for shader\cmdbuf
            };
            uint    state_;
        };
    };

    //! Constructor of 1D Resource object
    Resource(
        const Device& gpuDev,       //!< GPU device object
        size_t        size          //!< Resource size
        );

    //! Constructor of Image Resource object
    Resource(
        const Device& gpuDev,       //!< GPU device object
        size_t        width,        //!< resource width
        size_t        height,       //!< resource height
        size_t        depth,        //!< resource depth
        cl_image_format   format,   //!< resource format
        cl_mem_object_type  imageType,  //!< CL image type
        uint          mipLevels = 1 //!< Number of mip levels
        );

    //! Destructor of the resource
    virtual ~Resource();

    /*! \brief Creates a CAL object, associated with the resource
     *
     *  \return True if we succesfully created a CAL resource
     */
    virtual bool create(
        MemoryType  memType,        //!< memory type
        CreateParams*   params = 0  //!< special parameters for resource allocation
        );

    /*! \brief Copies a subregion of memory from one resource to another
     *
     *  This is a general copy from anything to anything (as long as it fits).
     *  All positions and sizes are given in bytes. Note, however, that only
     *  a subset of this general interface is currently implemented.
     *
     *  \return true if successful
     */
    bool partialMemCopyTo(
        VirtualGPU&  gpu,               //!< Virtual GPU device object
        const amd::Coord3D& srcOrigin,  //!< Origin of the source region
        const amd::Coord3D& dstOrigin,  //!< Origin of the destination region
        const amd::Coord3D& size,       //!< Size of the region to copy
        Resource& dstResource,          //!< Destination resource
        bool enableRectCopy = false,    //!< Rectangular DMA support
        bool flushDMA = false,          //!< Flush DMA if requested
        uint bytesPerElement = 1        //!< Bytes Per Element
        ) const;

    /*! \brief Copies size/4 DWORD of memory to a surface
     *
     *  This is a raw copy to any surface using a CP packet.
     *  Size needs to be atleast a DWORD or multiple
     *
     */
    void writeRawData(
        VirtualGPU& gpu,            //!< Virtual GPU device object
        size_t      offset,         //!< Offset for in the buffer for data
        size_t      size,           //!< Size in bytes of data to be copied(multiple of DWORDS)
        const void* data,           //!< Data to be copied
        bool        waitForEvent    //!< Wait for event complete
        ) const;

    //! Returns the offset in GPU memory for aliases
    size_t offset() const { return offset_; }

    //! Returns the pinned memory offset
    uint64_t pinOffset() const { return pinOffset_; }

    //! Returns the GPU device that owns this resource
    const Device& dev() const { return gpuDevice_; }

    //! Returns the descriptor for resource
    const Descriptor& desc() const { return desc_; }

    //! Returns the PAL memory object
    Pal::IGpuMemory* iMem() const { return memRef_->iMem(); }

    //! Returns global memory offset
    uint64_t vmAddress() const { return iMem()->Desc().gpuVirtAddr + offset_; }

    //! Returns global memory offset
    uint64_t vmSize() const { return iMem()->Desc().size - offset_; }

    //! Returns global memory offset
    bool mipMapped() const { return (desc().mipLevels_ > 1) ? true : false; }

    //! Checks if persistent memory can have a direct map
    bool isPersistentDirectMap() const;

    /*! \brief Locks the resource and returns a physical pointer
     *
     *  \note This operation stalls HW pipeline!
     *
     *  \return Pointer to the physical memory
     */
    void* map(
        VirtualGPU* gpu,        //!< Virtual GPU device object
        uint flags = 0,         //!< flags for the map operation
        // Optimization for multilayer map/unmap
        uint startLayer = 0,    //!< Start layer for multilayer map
        uint numLayers = 0      //!< End layer for multilayer map
        );

    //! Unlocks the resource if it was locked
    void unmap(
        VirtualGPU* gpu         //!< Virtual GPU device object
        );

    //! Marks the resource as busy
    void setBusy(
        VirtualGPU& gpu,        //!< Virtual GPU device object
        GpuEvent  calEvent      //!< CAL event
        ) const;

    //! Wait for the resource
    void wait(
        VirtualGPU& gpu,                //!< Virtual GPU device object
        bool    waitOnBusyEngine = false//!< Wait only if engine has changed
        ) const;

    //! Performs host write to the resource GPU memory
    bool hostWrite(
        VirtualGPU* gpu,            //!< Virtual GPU device object
        const void* hostPtr,        //!< Host pointer to the SRC data
        const amd::Coord3D& origin, //!< Offsets for the update
        const amd::Coord3D& size,   //!< The number of bytes to write
        uint        flags = 0,      //!< Map flags
        size_t      rowPitch = 0,   //!< Raw data row pitch
        size_t      slicePitch = 0  //!< Raw data slice pitch
        );

    //! Performs host read from the resource GPU memory
    bool hostRead(
        VirtualGPU* gpu,            //!< Virtual GPU device object
        void*       hostPtr,        //!< Host pointer to the DST data
        const amd::Coord3D& origin, //!< Offsets for the update
        const amd::Coord3D& size,   //!< The number of bytes to write
        size_t      rowPitch = 0,   //!< Raw data row pitch
        size_t      slicePitch = 0  //!< Raw data slice pitch
        );

    //! Warms up the rename list for this resource
    void warmUpRenames(VirtualGPU& gpu);

    //! Gets the resource element size
    uint elementSize() const { return elementSize_; }

    //! Get the mapped address of this resource
    address data() const { return reinterpret_cast<address>(address_); }

    //! Frees all allocated CAL memories and resources,
    //! associated with this objects. And also destroys all rename structures
    //! Note: doesn't destroy the object itself
    void free();

    //! Return memory type
    MemoryType      memoryType() const { return desc().type_; }

    //! Retunrs true if memory type matches specified
    bool isMemoryType(MemoryType memType) const;

    //! Returns TRUE if resource was allocated as cacheable
    bool isCacheable() const
        { return (isMemoryType(Remote) || isMemoryType(Pinned)) ? true : false; }

    bool gslGLAcquire() ;
    bool gslGLRelease() ;

    //! Returns HW state for the resource (used for images only)
    const void*   hwState() const { return hwState_; }

    //! Returns CPU HW SRD for the resource (used for images only)
    uint64_t    hwSrd() const { return hwSrd_; }

    uint numComponents() const {
        return Pal::Formats::NumComponents(image_->GetImageCreateInfo().format.chFmt); }

protected:
    uint    elementSize_;   //!< Size of a single element in bytes

private:
    //! Disable copy constructor
    Resource(const Resource&);

    //! Disable operator=
    Resource& operator=(const Resource&);

    typedef std::vector<GpuMemoryReference*> RenameList;

    //! Rename current resource
    bool rename(
        VirtualGPU& gpu,                //!< Virtual GPU device object
        bool        force = false       //!< Force renaming
        );

    //! Sets the rename as active
    void setActiveRename(
        VirtualGPU& gpu,                //!< Virtual GPU device object
        GpuMemoryReference* rename    //!< new active rename
        );

    //! Gets the active rename
    bool getActiveRename(
        VirtualGPU& gpu,                //!< Virtual GPU device object
        GpuMemoryReference** rename   //!< Saved active rename
        );

    /*! \brief Locks the resource with layers and returns a physical pointer
     *
     *  \return Pointer to the physical memory
     */
    void* mapLayers(
        VirtualGPU* gpu,            //!< Virtual GPU device object
        uint        flags = 0       //!< flags for the map operation
        );

    //! Unlocks the resource with layers if it was locked
    void unmapLayers(
        VirtualGPU* gpu             //!< Virtual GPU device object
        );

    //! Calls GSL to map a resource
    void* gpuMemoryMap(
        size_t* pitch,              //!< Pitch value for the image
        uint    flags,              //!< Map flags
        Pal::IGpuMemory* resource   //!< GSL memory object
        ) const;

    //! Uses GSL to unmap a resource
    void gpuMemoryUnmap(
        Pal::IGpuMemory* resource   //!< GSL memory object
        ) const;

    //! Fress all GSL resources associated with OCL resource
    void gslFree() const;

    //! Converts Resource memory type to the PAL heaps
    void memTypeToHeap(
        Pal::GpuMemoryCreateInfo* createInfo    //!< Memory create info
        );

    const Device&   gpuDevice_;     //!< GPU device
    Descriptor      desc_;          //!< Descriptor for this resource
    amd::Atomic<int> mapCount_;     //!< Total number of maps
    void*           address_;       //!< Physical address of this resource
    size_t          offset_;        //!< Resource offset
    size_t          curRename_;     //!< Current active rename in the list
    RenameList      renames_;       //!< Rename resource list
    GpuMemoryReference* memRef_;    //!< GSL resource reference
    const Resource* viewOwner_;     //!< GPU resource, which owns this view
    uint64_t        pinOffset_;     //!< Pinned memory offset
    void*           glInteropMbRes_;//!< Mb Res handle
    uint32_t        glType_;        //!< GL interop type
    void*           glPlatformContext_;
    void*           glDeviceContext_;

    // Optimization for multilayer map/unmap
    uint            startLayer_;    //!< Start layer for map/unmapLayer
    uint            numLayers_;     //!< Number of layers for map/unmapLayer
    uint            mapFlags_;      //!< Map flags for map/umapLayer

    //! @note: This field is necessary for the thread safe release only
    VirtualGPU*     gpu_;           //!< Resource will be used only on this queue
    Pal::IImage*    image_;     //!< PAL image object

    uint32_t*       hwState_;       //!< HW state for image object
    uint64_t        hwSrd_;         //!< GPU pointer to HW SRD
};

class ResourceCache : public amd::HeapObject
{
public:
    //! Default constructor
    ResourceCache(size_t cacheSizeLimit)
        : lockCacheOps_("PAL resource cache", true)
        , cacheSize_(0)
        , cacheSizeLimit_(cacheSizeLimit)
        {}

    //! Default destructor
    ~ResourceCache();

    //! Adds a CAL resource to the cache
    bool addGpuMemory(
        Resource::Descriptor*   desc,   //!< Resource descriptor - cache key
        GpuMemoryReference*     ref     //!< Resource reference
        );

    //! Finds a CAL resource from the cache
    GpuMemoryReference* findGpuMemory(
        Resource::Descriptor* desc, //!< Resource descriptor - cache key
        Pal::gpusize  size,
        Pal::gpusize  alignment
        );

    //! Destroys cache
    bool free(size_t minCacheEntries = 0);

private:
    //! Disable copy constructor
    ResourceCache(const ResourceCache&);

    //! Disable operator=
    ResourceCache& operator=(const ResourceCache&);

    //! Removes one last entry from the cache
    void removeLast();

    amd::Monitor    lockCacheOps_;  //!< Lock to serialise cache access

    size_t  cacheSize_;         //!< Current cache size in bytes
    size_t  cacheSizeLimit_;    //!< Cache size limit in bytes

    //! CAL resource cache
    std::list<std::pair<Resource::Descriptor*, GpuMemoryReference*> >    resCache_;
};

/*@}*/} // namespace pal
