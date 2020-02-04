/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#include "device/device.hpp"

#if defined(ATI_OS_WIN)
#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>
#endif
#include <GL/gl.h>
#include "GL/glATIInternal.h"

#include "os/os.hpp"
#include "utils/flags.hpp"
#include "thread/monitor.hpp"
#include "device/gpu/gpuresource.hpp"
#include "device/gpu/gpudevice.hpp"
#include "device/gpu/gpublit.hpp"
#include "device/gpu/gputimestamp.hpp"
#include "thread/atomic.hpp"
#include "hsa_ext_image.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace gpu {

GslResourceReference::GslResourceReference(const Device& gpuDev, gslMemObject gslResource,
                                           gslMemObject gslResOriginal)
    : device_(gpuDev), resource_(gslResource), resOriginal_(gslResOriginal), cpuAddress_(NULL) {}

GslResourceReference::~GslResourceReference() {
  if (cpuAddress_ != NULL) {
    device_.resUnmapRemote(gslResource());
  }
  if (0 != gslResource()) {
    device_.resFree(gslResource());
    resource_ = NULL;
  }

  if (0 != gslOriginal()) {
    device_.resFree(gslOriginal());
    resOriginal_ = NULL;
  }
}

Resource::Resource(const Device& gpuDev, size_t width, cmSurfFmt format)
    : elementSize_(0),
      gpuDevice_(gpuDev),
      mapCount_(0),
      address_(NULL),
      offset_(0),
      curRename_(0),
      gslRef_(NULL),
      viewOwner_(NULL),
      hbOffset_(0),
      hbSize_(0),
      pinOffset_(0),
      glInterop_(0),
      gpu_(NULL) {
  // Fill GSL descriptor fields
  cal_.type_ = Empty;
  cal_.width_ = width;
  cal_.height_ = 1;
  cal_.depth_ = 1;
  cal_.mipLevels_ = 1;
  cal_.format_ = format;
  cal_.flags_ = 0;
  cal_.pitch_ = 0;
  cal_.slice_ = 0;
  cal_.channelOrder_ = GSL_CHANNEL_ORDER_REPLICATE_R;
  cal_.dimension_ = GSL_MOA_BUFFER;
  cal_.cardMemory_ = true;
  cal_.dimSize_ = 1;
  cal_.buffer_ = true;
  cal_.imageArray_ = false;
  cal_.imageType_ = 0;
  cal_.skipRsrcCache_ = false;
  cal_.scratch_ = false;
  cal_.isAllocSVM_ = false;
  cal_.isAllocExecute_ = false;
}

Resource::Resource(const Device& gpuDev, size_t width, size_t height, size_t depth,
                   cmSurfFmt format, gslChannelOrder chOrder, cl_mem_object_type imageType,
                   uint mipLevels)
    : elementSize_(0),
      gpuDevice_(gpuDev),
      mapCount_(0),
      address_(NULL),
      offset_(0),
      curRename_(0),
      gslRef_(NULL),
      viewOwner_(NULL),
      hbOffset_(0),
      hbSize_(0),
      pinOffset_(0),
      glInterop_(0),
      gpu_(NULL) {
  // Fill GSL descriptor fields
  cal_.type_ = Empty;
  cal_.width_ = width;
  cal_.height_ = height;
  cal_.depth_ = depth;
  cal_.mipLevels_ = mipLevels;
  cal_.format_ = format;
  cal_.flags_ = 0;
  cal_.pitch_ = 0;
  cal_.slice_ = 0;
  cal_.channelOrder_ = chOrder;
  cal_.cardMemory_ = true;
  cal_.buffer_ = false;
  cal_.imageArray_ = false;
  cal_.imageType_ = imageType;
  cal_.skipRsrcCache_ = false;
  cal_.scratch_ = false;
  cal_.isAllocSVM_ = false;
  cal_.isAllocExecute_ = false;

  switch (imageType) {
    case CL_MEM_OBJECT_IMAGE2D:
      cal_.dimension_ = GSL_MOA_TEXTURE_2D;
      cal_.dimSize_ = 2;
      break;
    case CL_MEM_OBJECT_IMAGE3D:
      cal_.dimension_ = GSL_MOA_TEXTURE_3D;
      cal_.dimSize_ = 3;
      break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      cal_.dimension_ = GSL_MOA_TEXTURE_2D_ARRAY;
      cal_.dimSize_ = 3;
      cal_.imageArray_ = true;
      break;
    case CL_MEM_OBJECT_IMAGE1D:
      cal_.dimension_ = GSL_MOA_TEXTURE_1D;
      cal_.dimSize_ = 1;
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      cal_.dimension_ = GSL_MOA_TEXTURE_1D_ARRAY;
      cal_.dimSize_ = 2;
      cal_.imageArray_ = true;
      break;
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      cal_.dimension_ = GSL_MOA_TEXTURE_BUFFER;
      cal_.dimSize_ = 1;
      break;
    default:
      cal_.dimSize_ = 1;
      LogError("Unknown image type!");
      break;
  }
}

Resource::~Resource() { free(); }

static uint32_t GetHSAILImageFormatType(cmSurfFmt format) {
  uint32_t formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8;

  switch (format) {
    case CM_SURF_FMT_sR8:
    case CM_SURF_FMT_sRG8:
    case CM_SURF_FMT_sRGBA8:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8;
      break;
    case CM_SURF_FMT_sU16:
    case CM_SURF_FMT_sUV16:
    case CM_SURF_FMT_sUVWQ16:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16;
      break;
    case CM_SURF_FMT_INTENSITY8:
    case CM_SURF_FMT_RG8:
    case CM_SURF_FMT_RGBA8:
    case CM_SURF_FMT_RGBX8UI:
    case CM_SURF_FMT_RGBA8_SRGB:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8;
      break;
    case CM_SURF_FMT_R16:
    case CM_SURF_FMT_RG16:
    case CM_SURF_FMT_RGBA16:
    case CM_SURF_FMT_DEPTH16:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16;
      break;
    case CM_SURF_FMT_BGR10_X2:
    case CM_SURF_FMT_RGB10_X2:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010;
      break;
    case CM_SURF_FMT_sR8I:
    case CM_SURF_FMT_sRG8I:
    case CM_SURF_FMT_sRGBA8I:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
      break;
    case CM_SURF_FMT_sR16I:
    case CM_SURF_FMT_sRG16I:
    case CM_SURF_FMT_sRGBA16I:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
      break;
    case CM_SURF_FMT_sR32I:
    case CM_SURF_FMT_sRG32I:
    case CM_SURF_FMT_sRGBA32I:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
      break;
    case CM_SURF_FMT_R8I:
    case CM_SURF_FMT_RG8I:
    case CM_SURF_FMT_RGBA8UI:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
      break;
    case CM_SURF_FMT_R16I:
    case CM_SURF_FMT_RG16I:
    case CM_SURF_FMT_RGBA16UI:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
      break;
    case CM_SURF_FMT_R32I:
    case CM_SURF_FMT_RG32I:
    case CM_SURF_FMT_RGBA32UI:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
      break;
    case CM_SURF_FMT_R16F:
    case CM_SURF_FMT_RG16F:
    case CM_SURF_FMT_RGBA16F:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
      break;
    case CM_SURF_FMT_R32F:
    case CM_SURF_FMT_RG32F:
    case CM_SURF_FMT_RGBA32F:
    case CM_SURF_FMT_DEPTH32F:
    case CM_SURF_FMT_DEPTH32F_X24_STEN8:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT;
      break;
    case CM_SURF_FMT_DEPTH24_STEN8:
      formatType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24;
      break;
    default:
      assert(false);
  }

  return formatType;
}

static uint32_t GetHSAILImageOrderType(gslChannelOrder chOrder, cmSurfFmt format) {
  uint32_t orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_A;

  switch (chOrder) {
    case GSL_CHANNEL_ORDER_R:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
      break;
    case GSL_CHANNEL_ORDER_A:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_A;
      break;
    case GSL_CHANNEL_ORDER_RG:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_RG;
      break;
    case GSL_CHANNEL_ORDER_RA:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_RA;
      break;
    case GSL_CHANNEL_ORDER_RGB:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_RGB;
      break;
    case GSL_CHANNEL_ORDER_RGBA:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
      break;
    case GSL_CHANNEL_ORDER_BGRA:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA;
      break;
    case GSL_CHANNEL_ORDER_ARGB:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB;
      break;
    case GSL_CHANNEL_ORDER_SRGB:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB;
      break;
    case GSL_CHANNEL_ORDER_SRGBX:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX;
      break;
    case GSL_CHANNEL_ORDER_SRGBA:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA;
      break;
    case GSL_CHANNEL_ORDER_SBGRA:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA;
      break;
    case GSL_CHANNEL_ORDER_INTENSITY:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY;
      break;
    case GSL_CHANNEL_ORDER_LUMINANCE:
      orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE;
      break;
    case GSL_CHANNEL_ORDER_REPLICATE_R:
      if ((format == CM_SURF_FMT_DEPTH32F_X24_STEN8) || (format == CM_SURF_FMT_DEPTH24_STEN8)) {
        orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL;
      } else {
        orderType = HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH;
      }
      break;
    default:
      assert(false);
  }

  return orderType;
}

bool Resource::create(MemoryType memType, CreateParams* params) {
  bool calRes = false;
  gslMemObject gslResource = 0;
  gslMemObject gslResOriginal = 0;
  const amd::HostMemoryReference* hostMemRef = NULL;
  bool imageCreateView = false;
  CALuint hostMemOffset = 0;
  bool foundCalRef = false;
  bool viewDefined = false;
  uint viewLayer = 0;
  uint viewLevel = 0;
  uint viewFlags = 0;
  gslResource3D viewSize = {0};
  size_t viewOffset = 0;
  cmSurfFmt viewSurfFmt;
  gslChannelOrder viewChannelOrder = GSL_CHANNEL_ORDER_UNSPECIFIED;
  gslMemObjectAttribType viewResType;
  CALresourceDesc desc;
  uint64 bytePitch = (uint64)-1;
  bool useRowPitch = false;
  bool mipLevelPitchPad = false;

  desc.vaBase = 0;
  desc.minAlignment = 0;
  desc.isAllocExecute = false;
  desc.isAllocSVM = false;
  desc.section = GSL_SECTION_REGULAR;
  if (NULL != params && NULL != params->owner_) {  // make sure params not NULL
    mcaddr svmPtr = reinterpret_cast<mcaddr>(params->owner_->getSvmPtr());
    desc.vaBase = (svmPtr == 1) ? 0 : svmPtr;
    // Dont cache coarse\fine grain svm resource as these may not be released
    // and allocations may fail since there is limited space for coarse\fine grainbuffers
    cal_.skipRsrcCache_ = (svmPtr != 0);
    desc.section = (svmPtr != 0) ? GSL_SECTION_SVM : GSL_SECTION_REGULAR;

    if (params->owner_->getMemFlags() & CL_MEM_SVM_ATOMICS) {
      desc.section = GSL_SECTION_SVM_ATOMICS;
    }

    if (dev().settings().svmFineGrainSystem_ &&
        (desc.section == GSL_SECTION_SVM || desc.section == GSL_SECTION_SVM_ATOMICS)) {
      cal_.isAllocSVM_ = desc.isAllocSVM = true;
    }
  }

  if (memType == Shader) {
    if (dev().settings().svmFineGrainSystem_) {
      cal_.isAllocExecute_ = desc.isAllocExecute = true;
      cal_.isAllocSVM_ = desc.isAllocSVM = true;
    }
    // force to use remote memory for HW DEBUG or use
    // local memory once we determine if FGS is supported
    memType = (!dev().settings().enableHwDebug_) ? Local : RemoteUSWC;
  }

  // This is a thread safe operation
  const_cast<Device&>(dev()).initializeHeapResources();

  // Get the element size
  elementSize_ = static_cast<CALuint>(memoryFormatSize(cal()->format_).size_);
  cal_.type_ = memType;
  if (memType == Scratch) {
    // use local memory for scratch buffer unless it is using HW DEBUG
    cal_.type_ = (!dev().settings().enableHwDebug_) ? Local : RemoteUSWC;
    cal_.scratch_ = true;
  }

  // Force remote allocation if it was requested in the settings
  if (dev().settings().remoteAlloc_ && ((memoryType() == Local) || (memoryType() == Persistent))) {
    if (dev().settings().apuSystem_ && dev().settings().viPlus_) {
      cal_.type_ = Remote;
    } else {
      cal_.type_ = RemoteUSWC;
    }
  }

  if (dev().settings().disablePersistent_ && (memoryType() == Persistent)) {
    cal_.type_ = RemoteUSWC;
  }

  if (cal()->buffer_) {
    // Force linear tiling for buffer alloctions
    cal_.flags_ |= CAL_RESALLOC_GLOBAL_BUFFER;
  }

  if (params != NULL) {
    gpu_ = params->gpu_;
  }

  switch (memoryType()) {
    case Heap:
      gslResource = dev().resGetHeap(0);
      if (gslResource == 0) {
        return false;
      }
      calRes = true;
      cal_.width_ = static_cast<size_t>(gslResource->getPitch());
      cal_.pitch_ = static_cast<size_t>(gslResource->getPitch());
      break;
    case Persistent:
      if (dev().settings().linearPersistentImage_) {
        // Force linear tiling for image allocations in persistent
        cal_.flags_ |= CAL_RESALLOC_GLOBAL_BUFFER;
      }
    // Fall through ...
    case RemoteUSWC:
    case Remote:
    case Shader:
    case BusAddressable:
    case ExternalPhysical:
    // Fall through to process the memory allocation ...
    case Local: {
      if (cal()->buffer_) {
        //! @todo Remove alignment.
        //! GSL asserts in mem copy with an unaligned size
        cal_.width_ = amd::alignUp(cal_.width_, 64);
        if ((desc.section == GSL_SECTION_SVM || desc.section == GSL_SECTION_SVM_ATOMICS)) {
          cal_.width_ = amd::alignUp(cal_.width_, 64 * Ki / sizeof(uint32_t));
        }
      }

      desc.dimension = cal()->dimension_;
      desc.size.width = cal()->width_;
      desc.size.height = cal()->height_;
      desc.size.depth = cal()->depth_;
      desc.format = cal()->format_;
      desc.channelOrder = cal()->channelOrder_;
      desc.flags = cal()->flags_;
      desc.mipLevels = cal()->mipLevels_;
      desc.systemMemory = NULL;

      uint allocAttempt = 0;
      do {
        // Find a type for allocation
        if (memoryType() == Persistent) {
          desc.type = GSL_MOA_MEMORY_CARD_LOCKABLE;
        } else if (memoryType() == Remote) {
          desc.type = GSL_MOA_MEMORY_REMOTE_CACHEABLE;
        } else if (memoryType() == RemoteUSWC) {
          desc.type = GSL_MOA_MEMORY_AGP;
        } else if (memoryType() == BusAddressable) {
          desc.type = GSL_MOA_MEMORY_CARD_BUS_ADDRESSABLE;
        } else if (memoryType() == ExternalPhysical) {
          desc.type = GSL_MOA_MEMORY_CARD_EXTERNAL_PHYSICAL;
          cl_bus_address_amd bus_address =
              (reinterpret_cast<amd::Buffer*>(params->owner_))->busAddress();
          desc.busAddress[0] = bus_address.surface_bus_address;
          desc.busAddress[1] = bus_address.marker_bus_address;
        } else {
          desc.type = GSL_MOA_MEMORY_CARD_EXT_NONEXT;
        }

        // Check resource cache first for an appropriate resource
        gslRef_ = dev().resourceCache().findCalResource(&cal_);
        if (memType == Scratch) {
          if ((dev().settings().hsail_) || (dev().settings().oclVersion_ >= OpenCL20)) {
            desc.minAlignment = 64 * Ki;
          } else {
            desc.vaBase = static_cast<mcaddr>(0x100000000ULL);
          }
        } else if ((gslRef_ != NULL) && (!dev().settings().use64BitPtr_)) {
          // Make sure runtime didn't pick a resource with > 4GB address
          if ((cal()->dimension_ == GSL_MOA_BUFFER) &&
              (static_cast<uint64_t>(gslRef_->gslResource()->getSurfaceAddress() +
                                     gslRef_->gslResource()->getSurfaceSize()) >
               (uint64_t(4) * Gi))) {
            gslRef_->release();
            gslRef_ = NULL;
          }
        }
        // Try to allocate memory if we couldn't find a cached resource
        if (gslRef_ == NULL) {
          // Allocate memory
          gslResource = dev().resAlloc(&desc);
          if (gslResource != 0) {
            calRes = true;
          }
        } else {
          calRes = true;
          gslResource = gslRef_->gslOriginal();
          foundCalRef = true;
        }

        // If GSL fails allocation then try other heaps
        if (!calRes) {
          // Free cache if we failed allocation
          if (dev().resourceCache().free()) {
            // We freed something - attempt to allocate memory again
            continue;
          }

          // Local to Persistent
          if (memoryType() == Local) {
            cal_.type_ = Persistent;
          }
          // Don't switch to USWC if persistent memory was explicitly asked
          else if ((allocAttempt > 0) && (memoryType() == Persistent)) {
            cal_.type_ = RemoteUSWC;
          }
          // Remote cacheable to uncacheable
          else if (memoryType() == Remote) {
            cal_.type_ = RemoteUSWC;
          } else {
            break;
          }
          allocAttempt++;
        }
      } while (!calRes);
    } break;
    case Pinned: {
      PinnedParams* pinned = reinterpret_cast<PinnedParams*>(params);
      CALuint allocSize = static_cast<CALuint>(pinned->size_);
      void* pinAddress;
      hostMemRef = pinned->hostMemRef_;
      pinAddress = address_ = hostMemRef->hostMem();

      // Use untiled allocation
      cal_.flags_ |= CAL_RESALLOC_GLOBAL_BUFFER;

      desc.size.width = cal()->width_;

      if (cal()->dimension_ == GSL_MOA_BUFFER) {
        // Allign offset to 4K boundary (Vista/Win7 limitation)
        char* tmpHost = const_cast<char*>(
            amd::alignDown(reinterpret_cast<const char*>(address_), PinnedMemoryAlignment));

        // Find the partial size for unaligned copy
        hostMemOffset = static_cast<CALuint>(reinterpret_cast<const char*>(address_) - tmpHost);

        pinOffset_ = hostMemOffset & 0xff;

        pinAddress = tmpHost;
        // Align width to avoid GSL useless assert with a view
        if (hostMemOffset != 0) {
          desc.size.width += hostMemOffset / elementSize();
          desc.size.width = amd::alignUp(desc.size.width, 64);
        }
        hostMemOffset &= ~(0xff);
      } else if (cal()->dimension_ == GSL_MOA_TEXTURE_2D) {
        //! @todo: Width has to be aligned for 3D.
        //! Need to be replaced with a compute copy
        // Width aligned by 8 texels
        if (((cal()->width_ % 0x8) != 0) ||
            // Pitch aligned by 64 bytes
            (((cal()->width_ * elementSize()) % 0x40) != 0)) {
          return false;
        }
      } else {
        //! @todo GSL doesn't support pinning with resAlloc_
        return false;
      }

      // Fill the GSL desc info structure
      desc.dimension = cal()->dimension_;
      desc.type = GSL_MOA_MEMORY_SYSTEM;
      desc.size.height = cal()->height_;
      desc.size.depth = cal()->depth_;
      desc.format = cal()->format_;
      desc.channelOrder = cal()->channelOrder_;
      desc.mipLevels = 0;
      desc.systemMemory = reinterpret_cast<CALvoid*>(pinAddress);
      desc.flags = 0;

      // Ensure page alignment
      if ((CALuint64)desc.systemMemory & (amd::Os::pageSize() - 1)) {
        return false;
      }

      gslResource = dev().resAlloc(&desc);
      if (gslResource != 0) {
        calRes = true;
      } else {
        pinOffset_ = 0;
      }
    } break;
    case View: {
      // Save the offset in the global heap
      ViewParams* view = reinterpret_cast<ViewParams*>(params);
      offset_ = view->offset_;

      // Make sure parent was provided
      if (NULL != view->resource_) {
        viewOwner_ = view->resource_;
        uint64 bytePitch = (view->size_ + viewOwner_->pinOffset());
        viewSize.width = bytePitch / elementSize();
        viewSize.height = 1;
        viewSize.depth = 1;
        viewOffset = static_cast<CALuint>(offset() / elementSize());

        gslResource = dev().resAllocView(view->resource_->gslResource(), viewSize, viewOffset,
                                         cal()->format_, GSL_CHANNEL_ORDER_REPLICATE_R,
                                         cal()->dimension_, 0, 0, cal()->flags_, bytePitch);
        if (gslResource != 0) {
          calRes = true;
        }

        if (viewOwner_->isMemoryType(Pinned)) {
          address_ = viewOwner_->data() + offset();
        }
        pinOffset_ = viewOwner_->pinOffset();
      } else {
        cal_.type_ = Empty;
      }
    } break;
    case ImageView: {
      ImageViewParams* imageView = reinterpret_cast<ImageViewParams*>(params);
      imageCreateView = true;
      viewLayer = imageView->layer_;
      viewLevel = imageView->level_;
      gslResource = imageView->resource_->gslResource();
      viewOwner_ = imageView->resource_;
      if ((viewLevel != 0) || viewOwner_->mipMapped()) {
        viewFlags |= CAL_RESALLOCSLICEVIEW_LEVEL;
      }
      if ((viewOwner_->viewOwner_ != NULL) && viewOwner_->viewOwner_->mipMapped()) {
        mipLevelPitchPad = true;
      }

      if (viewLayer != 0) {
        viewFlags |= CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER;
      }
      calRes = true;
    } break;
    case ImageBuffer: {
      ImageBufferParams* imageBuffer = reinterpret_cast<ImageBufferParams*>(params);
      imageCreateView = true;
      gslResource = imageBuffer->resource_->gslResource();
      viewOwner_ = imageBuffer->resource_;
      calRes = true;
      useRowPitch = true;
    } break;
    case OGLInterop: {
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
      CALGSLDevice::GLResAssociate resData = {0};
      resData.GLContext = oglRes->glPlatformContext_;
      resData.GLdeviceContext = oglRes->glDeviceContext_;
      resData.name = oglRes->handle_;
      resData.type = glType_;
      // We need not pass any flags down to OGL for interop and there is no need to
      // pass down resData.flags field

      if (dev().resGLAssociate(resData)) {
        gslResource = resData.memObject;
        glInteropMbRes_ = resData.mbResHandle;
        glInterop_ = resData.mem_base;
        calRes = true;
      }

      // Check if we have to create a view
      if (calRes && ((oglRes->type_ == InteropTextureViewLevel) ||
                     (oglRes->type_ == InteropTextureViewCube))) {
        imageCreateView = true;
        viewLayer = oglRes->layer_;
        viewLevel = oglRes->mipLevel_;

        // Find the view parameters
        if (InteropTextureViewLevel == oglRes->type_) {
          viewFlags |= CAL_RESALLOCSLICEVIEW_LEVEL;
        } else if (InteropTextureViewCube == oglRes->type_) {
          viewFlags |= CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER;
        } else {
          LogError("Unknown Interop View Type");
        }
      }
    } break;
#ifdef _WIN32
    case D3D9Interop:
    case D3D10Interop:
    case D3D11Interop: {
      D3DInteropParams* d3dRes = reinterpret_cast<D3DInteropParams*>(params);
      desc.dimension = cal()->dimension_;
      desc.size.width = cal()->width_;
      desc.size.height = cal()->height_;
      desc.size.depth = cal()->depth_;
      desc.format = cal()->format_;
      desc.channelOrder = cal()->channelOrder_;
      desc.flags = cal()->flags_;
      desc.mipLevels = 0;
      desc.systemMemory = NULL;
      switch (d3dRes->misc) {
        case 1:  // NV12 format
        case 2:  // YV12 format
          // Readjust the size to the original NV12/YV12 size, since runtime
          // creates an interop for all planes
          switch (d3dRes->layer_) {
            case 0:
              desc.size.height = 3 * desc.size.height / 2;
              break;
            case 1:
            case 2:
              // Force R8 format for the interop allocation by default
              if (1 == d3dRes->misc) {
                desc.format = CM_SURF_FMT_R8;
                desc.channelOrder = GSL_CHANNEL_ORDER_R;
              }
              desc.size.width = 2 * desc.size.width;
              desc.size.height = 3 * desc.size.height;
              break;
            default:
              break;
          }
          break;
        default:
          break;
      }

      // Create an interop GSL object
      gslResource =
          dev().resMapD3DResource(&desc, (CALuint64)d3dRes->handle_, (memoryType() != D3D9Interop));
      if (gslResource != 0) {
        calRes = true;
      } else {
        return false;
      }


      // Check if we have to create a view
      if (calRes && ((d3dRes->type_ == InteropTextureViewLevel) ||
                     (d3dRes->type_ == InteropTextureViewCube))) {
        imageCreateView = true;
        viewLayer = d3dRes->layer_;
        viewLevel = d3dRes->mipLevel_;

        // Find the view parameters
        if (InteropTextureViewLevel == d3dRes->type_) {
          viewFlags |= CAL_RESALLOCSLICEVIEW_LEVEL;
        } else if (InteropTextureViewCube == d3dRes->type_) {
          viewFlags |= CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER;
        } else {
          LogError("Unknown Interop View Type");
        }
      }

      switch (d3dRes->misc) {
        case 0:
          break;
        case 1:  // NV12 format
        case 2:  // YV12 format
          // Create a view for the specified plane
          viewDefined = true;
          viewSize.width = cal()->width_;
          viewSize.height = cal()->height_;
          viewSize.depth = 1;
          bytePitch = static_cast<size_t>(gslResource->getPitch());
          viewOffset = 0;
          viewSurfFmt = cal()->format_;
          viewChannelOrder = cal()->channelOrder_;
          switch (d3dRes->layer_) {
            case -1:
              bytePitch *= elementSize();
              break;
            case 0:
              bytePitch *= elementSize();
              break;
            case 1:
              // Y - plane size to the offset
              viewOffset = bytePitch * viewSize.height * 2;
              if (d3dRes->misc == 2) {
                // YV12 format U is 2 times smaller plane
                bytePitch /= 2;
              }
              break;
            case 2:
              // Y + U plane sizes to the offest.
              // U plane is 4 times smaller than Y => 5/2
              viewOffset = bytePitch * viewSize.height * 5 / 2;
              // V is 2 times smaller plane
              bytePitch /= 2;
              break;
            default:
              LogError("Unknown Interop View Type");
              calRes = false;
              break;
          }
          break;
        case 3:
          break;
        default:
          LogError("Unknown Interop View Type");
          calRes = false;
      }
    } break;
#endif  // _WIN32
    default:
      LogWarning("Resource::create() called with unknown memory type");
      return false;
      break;
  }

  // Create a view for interop, since the original buffer may have different format
  // than the global buffer and GSL mem copy will fail
  bool interopBufView =
      cal()->buffer_ && ((memoryType() == D3D10Interop) || (memoryType() == OGLInterop) ||
                         (memoryType() == D3D11Interop));

  bool ignoreParentHandle = ((memoryType() == ImageView) || (memoryType() == ImageBuffer));

  // Create imageview if it was requested
  if (calRes && (imageCreateView || interopBufView || hostMemOffset || viewDefined)) {
    gslResOriginal = gslResource;

    // Disable tiling if it's a buffer view
    if (interopBufView || hostMemOffset) {
      viewFlags = CAL_RESALLOCVIEW_GLOBAL_BUFFER;
    }

    viewResType = cal()->dimension_;
    if (!viewDefined) {
      viewSize.width = cal()->width_ + (pinOffset() / elementSize());
      viewSize.height = cal()->height_;
      viewSize.depth = cal()->depth_;
      viewOffset = hostMemOffset / static_cast<CALuint>(elementSize());
      viewSurfFmt = cal()->format_;
      viewChannelOrder = cal()->channelOrder_;
    }

    if (useRowPitch && (params->owner_ != NULL) && params->owner_->asImage() &&
        (params->owner_->asImage()->getRowPitch() != 0)) {
      bytePitch = params->owner_->asImage()->getRowPitch();
    }

    // Allocate a view resource object
    gslResource =
        dev().resAllocView(gslResOriginal, viewSize, viewOffset, viewSurfFmt, viewChannelOrder,
                           viewResType, viewLevel, viewLayer, viewFlags, bytePitch);

    if (gslResource == 0) {
      // If we don't have to keep the parent handle,
      // then destroy the original resource
      if (!ignoreParentHandle) {
        dev().resFree(gslResOriginal);
        gslResOriginal = 0;
      }
      LogError("ResAlloc failed!");
      return false;
    }

    if (ignoreParentHandle) {
      gslResOriginal = 0;
    }
  }

  if (!calRes) {
    if (gslResource != 0) {
      dev().resFree(gslResource);
    }
    if (memoryType() != Pinned) {
      LogError("calResAlloc failed!");
    }
    return false;
  }

  // Find memory location
  switch (gslResource->getAttribs().location) {
    case GSL_MOA_MEMORY_CARD:
    case GSL_MOA_MEMORY_CARD_EXT:
    case GSL_MOA_MEMORY_CARD_LOCKABLE:
    case GSL_MOA_MEMORY_CARD_EXT_NONEXT:
    case GSL_MOA_MEMORY_CARD_BUS_ADDRESSABLE:
      cal_.cardMemory_ = true;
      break;
    default:
      cal_.cardMemory_ = false;
      break;
  }

  gslMemObjectAttribTiling tiling = gslResource->getAttribs().tiling;
  cal_.tiled_ = (GSL_MOA_TILING_LINEAR != tiling) && (GSL_MOA_TILING_LINEAR_GENERAL != tiling);

  // Get the heap block offset
  hbOffset_ = gslResource->getSurfaceAddress() - dev().heap().baseAddress();
  hbSize_ = static_cast<uint64_t>(gslResource->getSurfaceSize());

  if (!dev().settings().use64BitPtr_ &&
      !((memType == Scratch) || ((memType == View) && viewOwner_->cal()->scratch_))) {
    // Make sure runtime doesn't go over the address space limit for buffers
    if ((memoryType() != Heap) && (cal()->dimension_ == GSL_MOA_BUFFER) &&
        ((hbOffset_ + hbSize_) > (uint64_t(4) * Gi))) {
      if (cal_.cardMemory_) {
        LogPrintfError("Out of 4GB address space. Base: 0x%016llX, size: 0x%016llX!", hbOffset_,
                       hbSize_);

        dev().resFree(gslResource);
        //! @note: A workaround for a Windows delay on memory destruction
        //! Runtime submits a fake memory fill to force KMD to return
        //! the freed memory ranges
        if (IS_WINDOWS) {
          uint32_t pattern = 0;
          Memory* dummy = reinterpret_cast<Memory*>(dev().dummyPage()->getDeviceMemory(dev()));
          dev().xferMgr().fillBuffer(*dummy, &pattern, sizeof(uint32_t), amd::Coord3D(0),
                                     amd::Coord3D(sizeof(uint32_t)));
        }
        if ((gslResOriginal != 0) && !ignoreParentHandle) {
          dev().resFree(gslResOriginal);
          gslResOriginal = 0;
        }
        return false;
      } else {
        LogWarning("Out of 4GB address space for AHP/UHP!");
      }
    }
  }

  if (!foundCalRef) {
    gslRef_ = new GslResourceReference(dev(), gslResource, gslResOriginal);
    if (gslRef_ == NULL) {
      LogError("Memory allocation failure!");
      dev().resFree(gslResource);
      return false;
    }
  }

  if ((dev().settings().hsail_ || (dev().settings().oclVersion_ >= OpenCL20)) && !cal()->buffer_) {
    hwSrd_ = dev().srds().allocSrdSlot(reinterpret_cast<address*>(&hwState_));
    if (0 == hwSrd_) {
      return false;
    }
    dev().fillImageHwState(gslResource, hwState_, 8 * sizeof(uint32_t));
    hwState_[8] = GetHSAILImageFormatType(cal()->format_);
    hwState_[9] = GetHSAILImageOrderType(cal()->channelOrder_, cal()->format_);
    hwState_[10] = static_cast<uint32_t>(cal()->width_);
    if (memoryType() == ImageView) {
      // Workaround for depth view, change tileIndex to the parent for depth view
      if (viewChannelOrder == GSL_CHANNEL_ORDER_REPLICATE_R) {
        if ((hwState_[3] & 0x1f00000) == 0xe00000) {
          hwState_[3] = (hwState_[3] & 0xfe0fffff) | (viewOwner_->hwState_[3] & 0x1f00000);
        }
      }
      // Update the POW2_PAD flag, otherwise HW uses a wrong pitch value
      if ((viewFlags & CAL_RESALLOCSLICEVIEW_LEVEL) || mipLevelPitchPad) {
        hwState_[3] |= (viewOwner_->hwState_[3] & 0x2000000);
      }
    }
    hwState_[11] = 0;  // one extra reserved field in the argument
  }

  if (desc.section == GSL_SECTION_SVM || desc.section == GSL_SECTION_SVM_ATOMICS) {
    params->owner_->setSvmPtr(reinterpret_cast<void*>(gslResource->getSurfaceAddress()));
  }

  return true;
}

void Resource::free() {
  if (gslRef_ == NULL) {
    return;
  }

  // Sanity check for the map calls
  if (mapCount_ != 0) {
    LogWarning("Resource wasn't unlocked, but destroyed!");
  }
  const bool wait = (memoryType() != ImageView) && (memoryType() != ImageBuffer);

  // Check if resource could be used in any queue(thread)
  if (gpu_ == NULL) {
    Device::ScopedLockVgpus lock(dev());

    if (renames_.size() == 0) {
      // Destroy GSL resource
      if (gslResource() != 0) {
        // Release all virtual memory objects on all virtual GPUs
        for (uint idx = 0; idx < dev().vgpus().size(); ++idx) {
          // Ignore the transfer queue,
          // since it releases resources after every operation
          if (dev().vgpus()[idx] != dev().xferQueue()) {
            dev().vgpus()[idx]->releaseMemory(gslResource(), wait);
          }
        }

        //! @note: This is a workaround for bad applications that
        //! don't unmap memory
        if (mapCount_ != 0) {
          unmap(NULL);
        }

        // Add resource to the cache
        if (!dev().resourceCache().addCalResource(&cal_, gslRef_)) {
          gslFree();
        }
      }
    } else {
      renames_[curRename_]->cpuAddress_ = 0;
      for (size_t i = 0; i < renames_.size(); ++i) {
        gslRef_ = renames_[i];
        // Destroy GSL resource
        if (gslResource() != 0) {
          // Release all virtual memory objects on all virtual GPUs
          for (uint idx = 0; idx < dev().vgpus().size(); ++idx) {
            // Ignore the transfer queue,
            // since it releases resources after every operation
            if (dev().vgpus()[idx] != dev().xferQueue()) {
              dev().vgpus()[idx]->releaseMemory(gslResource());
            }
          }
          gslFree();
        }
      }
    }
  } else {
    if (renames_.size() == 0) {
      // Destroy GSL resource
      if (gslResource() != 0) {
        // Release virtual memory object on the specified virtual GPU
        gpu_->releaseMemory(gslResource(), wait);
        gslFree();
      }
    } else
      for (size_t i = 0; i < renames_.size(); ++i) {
        gslRef_ = renames_[i];
        // Destroy GSL resource
        if (gslResource() != 0) {
          // Release virtual memory object on the specified virtual GPUs
          gpu_->releaseMemory(gslResource());
          gslFree();
        }
      }
  }

  // Free SRD for images
  if ((dev().settings().hsail_ || (dev().settings().oclVersion_ >= OpenCL20)) && !cal()->buffer_) {
    dev().srds().freeSrdSlot(hwSrd_);
  }
}

void Resource::writeRawData(VirtualGPU& gpu, size_t size, const void* data,
                            bool waitForEvent) const {
  GpuEvent event;

  // Write data size bytes to surface
  // size needs to be DWORD aligned
  assert((size & 3) == 0);
  gpu.eventBegin(MainEngine);
  gslResource()->writeDataRaw(gpu.cs(), size, data, true);
  gpu.eventEnd(MainEngine, event);

  setBusy(gpu, event);
  // Update the global GPU event
  gpu.setGpuEvent(event, false);

  if (waitForEvent) {
    // Wait for event to complete
    gpu.waitForEvent(&event);
  }
}

bool Resource::partialMemCopyTo(VirtualGPU& gpu, const amd::Coord3D& srcOrigin,
                                const amd::Coord3D& dstOrigin, const amd::Coord3D& size,
                                Resource& dstResource, bool enableCopyRect, bool flushDMA,
                                uint bytesPerElement) const {
  GpuEvent event;
  bool result;
  CALuint syncFlags = CAL_MEMCOPY_SYNC;
  EngineType activeEngineID = gpu.engineID_;
  static const bool waitOnBusyEngine = true;
  // \note timing issues in Linux with sync mode
  bool flush = true;

  // Check if runtime can use async memory copy,
  // even if a caller didn't request async
  // Keep ASYNC if profiling is disabled or sdma profiling is possible
  if ((!gpu.profiling() || dev().settings().sdmaProfiling_) &&
      (!cal()->cardMemory_ || !dstResource.cal()->cardMemory_)) {
    // Switch to SDMA engine
    gpu.engineID_ = SdmaEngine;
    syncFlags = CAL_MEMCOPY_ASYNC;
    flush = false;
  }

  // Wait for the resources, since runtime may use async transfers
  wait(gpu, waitOnBusyEngine);
  dstResource.wait(gpu, waitOnBusyEngine);

  size_t calSrcOrigin[3], calDstOrigin[3], calSize[3];
  calSrcOrigin[0] = srcOrigin[0] + pinOffset();
  calSrcOrigin[1] = srcOrigin[1];
  calSrcOrigin[2] = srcOrigin[2];
  calDstOrigin[0] = dstOrigin[0] + dstResource.pinOffset();
  calDstOrigin[1] = dstOrigin[1];
  calDstOrigin[2] = dstOrigin[2];
  calSize[0] = size[0];
  calSize[1] = size[1];
  calSize[2] = size[2];

  result = gpu.copyPartial(event, gslResource(), calSrcOrigin, dstResource.gslResource(),
                           calDstOrigin, calSize, static_cast<CALmemcopyflags>(syncFlags),
                           enableCopyRect, bytesPerElement);

  if (result) {
    // Mark source and destination as busy
    setBusy(gpu, event);
    dstResource.setBusy(gpu, event);

    // Update the global GPU event
    gpu.setGpuEvent(event, (flush | flushDMA));
  }

  // Restore the original engine
  gpu.engineID_ = activeEngineID;

  return result;
}

void Resource::setBusy(VirtualGPU& gpu, GpuEvent gpuEvent) const {
  gpu.assignGpuEvent(gslResource(), gpuEvent);

  // If current resource is a view, then update the parent event as well
  if (viewOwner_ != NULL) {
    viewOwner_->setBusy(gpu, gpuEvent);
  }
}

void Resource::wait(VirtualGPU& gpu, bool waitOnBusyEngine) const {
  GpuEvent* gpuEvent = gpu.getGpuEvent(gslResource());

  // Check if we have to wait unconditionally
  if (!waitOnBusyEngine ||
      // or we have to wait only if another engine was used on this resource
      (waitOnBusyEngine && (gpuEvent->engineId_ != gpu.engineID_))) {
    gpu.waitForEvent(gpuEvent);
  }

  // If current resource is a view and not in the global heap,
  // then wait for the parent event as well
  if ((viewOwner_ != NULL) && (viewOwner_ != &dev().globalMem())) {
    viewOwner_->wait(gpu, waitOnBusyEngine);
  }
}

bool Resource::hostWrite(VirtualGPU* gpu, const void* hostPtr, const amd::Coord3D& origin,
                         const amd::Coord3D& size, uint flags, size_t rowPitch, size_t slicePitch) {
  void* dst;

  size_t startLayer = origin[2];
  size_t numLayers = size[2];
  if (cal()->dimension_ == GSL_MOA_TEXTURE_1D_ARRAY) {
    startLayer = origin[1];
    numLayers = size[1];
  }

  // Get physical GPU memmory
  dst = map(gpu, flags, startLayer, numLayers);
  if (NULL == dst) {
    LogError("Couldn't map GPU memory for host write");
    return false;
  }

  if (1 == cal()->dimSize_) {
    size_t copySize = (cal()->buffer_) ? size[0] : size[0] * elementSize_;

    // Update the pointer
    dst = static_cast<void*>(static_cast<char*>(dst) + origin[0]);

    // Copy memory
    amd::Os::fastMemcpy(dst, hostPtr, copySize);
  } else {
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
    dstOffsBase += cal()->pitch_ * origin[1] * elementSize_;

    // Adjust the destination offset with Z dimension
    dstOffsBase += cal()->slice_ * origin[2] * elementSize_;

    // Copy memory slice by slice
    for (size_t slice = 0; slice < size[2]; ++slice) {
      dstOffs = dstOffsBase + slice * cal()->slice_ * elementSize_;
      srcOffs = slice * slicePitch;

      // Copy memory line by line
      for (size_t row = 0; row < size[1]; ++row) {
        // Copy memory
        amd::Os::fastMemcpy((reinterpret_cast<address>(dst) + dstOffs),
                            (reinterpret_cast<const_address>(hostPtr) + srcOffs),
                            size[0] * elementSize_);

        dstOffs += cal()->pitch_ * elementSize_;
        srcOffs += rowPitch;
      }
    }
  }

  // Unmap GPU memory
  unmap(gpu);

  return true;
}

bool Resource::hostRead(VirtualGPU* gpu, void* hostPtr, const amd::Coord3D& origin,
                        const amd::Coord3D& size, size_t rowPitch, size_t slicePitch) {
  void* src;

  size_t startLayer = origin[2];
  size_t numLayers = size[2];
  if (cal()->dimension_ == GSL_MOA_TEXTURE_1D_ARRAY) {
    startLayer = origin[1];
    numLayers = size[1];
  }

  // Get physical GPU memmory
  src = map(gpu, ReadOnly, startLayer, numLayers);
  if (NULL == src) {
    LogError("Couldn't map GPU memory for host read");
    return false;
  }

  if (1 == cal()->dimSize_) {
    size_t copySize = (cal()->buffer_) ? size[0] : size[0] * elementSize_;

    // Update the pointer
    src = static_cast<void*>(static_cast<char*>(src) + origin[0]);

    // Copy memory
    amd::Os::fastMemcpy(hostPtr, src, copySize);
  } else {
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
    srcOffsBase += cal()->pitch_ * origin[1] * elementSize_;

    // Adjust the destination offset with Z dimension
    srcOffsBase += cal()->slice_ * origin[2] * elementSize_;

    // Copy memory line by line
    for (size_t slice = 0; slice < size[2]; ++slice) {
      srcOffs = srcOffsBase + slice * cal()->slice_ * elementSize_;
      dstOffs = slice * slicePitch;

      // Copy memory line by line
      for (size_t row = 0; row < size[1]; ++row) {
        // Copy memory
        amd::Os::fastMemcpy((reinterpret_cast<address>(hostPtr) + dstOffs),
                            (reinterpret_cast<const_address>(src) + srcOffs),
                            size[0] * elementSize_);

        srcOffs += cal()->pitch_ * elementSize_;
        dstOffs += rowPitch;
      }
    }
  }

  // Unmap GPU memory
  unmap(gpu);

  return true;
}

void* Resource::gslMap(size_t* pitch, gslMapAccessType flags, gslMemObject resource) const {
  if (cal_.cardMemory_ || cal_.tiled_) {
    // @todo remove const cast
    return const_cast<Device&>(dev()).resMapLocal(*pitch, resource, flags);
  } else {
    return dev().resMapRemote(*pitch, resource, flags);
  }
}

void Resource::gslUnmap(gslMemObject resource) const {
  if (cal_.cardMemory_) {
    // @todo remove const cast
    const_cast<Device&>(dev()).resUnmapLocal(resource);
  } else {
    dev().resUnmapRemote(resource);
  }
}

bool Resource::gslGLAcquire() {
  bool retVal = true;
  if (cal()->type_ == OGLInterop) {
    retVal = dev().resGLAcquire(glPlatformContext_, glInteropMbRes_, glType_);
  }
  return retVal;
}

bool Resource::gslGLRelease() {
  bool retVal = true;
  if (cal()->type_ == OGLInterop) {
    retVal = dev().resGLRelease(glPlatformContext_, glInteropMbRes_, glType_);
  }
  return retVal;
}
void Resource::gslFree() const {
  if (cal()->type_ == OGLInterop) {
    if (0 == gslRef_->resOriginal_) {
      dev().resGLFree(glPlatformContext_, glDeviceContext_, gslRef_->resource_, glInterop_,
                      glInteropMbRes_, glType_);
      gslRef_->resource_ = 0;
    } else {
      dev().resFree(gslRef_->resource_);
      gslRef_->resource_ = 0;
      dev().resGLFree(glPlatformContext_, glDeviceContext_, gslRef_->resOriginal_, glInterop_,
                      glInteropMbRes_, glType_);
      gslRef_->resOriginal_ = 0;
    }
  }
  gslRef_->release();
}

bool Resource::isMemoryType(MemoryType memType) const {
  if (memoryType() == memType) {
    return true;
  } else if (memoryType() == View) {
    return viewOwner_->isMemoryType(memType);
  }

  return false;
}

bool Resource::isPersistentDirectMap() const {
  bool directMap =
      ((memoryType() == Resource::Persistent) && (cal()->dimSize_ < 3) && !cal()->imageArray_);

  // If direct map is possible, then validate it with the current tiling
  if (directMap && cal()->tiled_) {
    //!@note IOL for Linux doesn't support tiling aperture
    // and runtime doesn't force linear images in persistent
    directMap = IS_WINDOWS && !dev().settings().linearPersistentImage_;
  }

  return directMap;
}

void* Resource::map(VirtualGPU* gpu, uint flags, uint startLayer, uint numLayers) {
  if (isMemoryType(Pinned)) {
    // Check if we have to wait
    if (!(flags & NoWait)) {
      if (gpu != NULL) {
        wait(*gpu);
      }
    }
    return address_;
  }

  gslMapAccessType mapFlags = GSL_MAP_READ_WRITE;

  if (flags & ReadOnly) {
    assert(!(flags & Discard) && "We can't use lock discard with read only!");
    mapFlags = GSL_MAP_READ_ONLY;
  }

  if (flags & WriteOnly) {
    mapFlags = GSL_MAP_WRITE_ONLY;
  }

  // Check if use map discard
  if (flags & Discard) {
    mapFlags = GSL_MAP_WRITE_ONLY;
    if (gpu != NULL) {
      // If we use a new renamed allocation, then skip the wait
      if (rename(*gpu)) {
        flags |= NoWait;
      }
    }
  }

  // Check if we have to wait
  if (!(flags & NoWait)) {
    if (gpu != NULL) {
      wait(*gpu);
    }
  }

  // Check if memory wasn't mapped yet
  if (++mapCount_ == 1) {
    if ((cal()->dimSize_ == 3) || cal()->imageArray_ ||
        ((cal()->type_ == ImageView) && viewOwner_->mipMapped())) {
      // Save map info for multilayer map/unmap
      startLayer_ = startLayer;
      numLayers_ = numLayers;
      mapFlags_ = mapFlags;
      // Map with layers
      address_ = mapLayers(gpu, mapFlags);
    } else {
      // Map current resource
      address_ = gslMap(&cal_.pitch_, mapFlags, gslResource());
      if (address_ == NULL) {
        LogError("cal::ResMap failed!");
        --mapCount_;
        return NULL;
      }
    }
  }

  //! \note the atomic operation with counter doesn't
  // guarantee that the address will be valid,
  // since GSL could still process the first map
  if (address_ == NULL) {
    for (uint i = 0; address_ == NULL && i < 10; ++i) {
      amd::Os::sleep(1);
    }
    assert((address_ != NULL) && "Multiple maps failed!");
  }

  return address_;
}

void* Resource::mapLayers(VirtualGPU* gpu, CALuint flags) {
  size_t srcOffs = 0;
  size_t dstOffs = 0;
  gslMemObject sliceResource = 0;
  gslMemObjectAttribType gslDim = GSL_MOA_TEXTURE_2D;
  size_t layers = cal()->depth_;
  size_t height = cal()->height_;

  // Use 1D layers
  if (GSL_MOA_TEXTURE_1D_ARRAY == cal()->dimension_) {
    gslDim = GSL_MOA_TEXTURE_1D;
    height = 1;
    layers = cal()->height_;
  }

  cal_.pitch_ = cal()->width_;
  cal_.slice_ = cal()->pitch_ * height;
  address_ = new char[cal()->slice_ * layers * elementSize()];
  if (NULL == address_) {
    return NULL;
  }

  // Check if map is write only
  if (flags == GSL_MAP_WRITE_ONLY) {
    return address_;
  }

  if (numLayers_ != 0) {
    layers = startLayer_ + numLayers_;
  }

  dstOffs = startLayer_ * cal()->slice_ * elementSize();

  // Loop through all layers
  for (uint i = startLayer_; i < layers; ++i) {
    gslResource3D gslSize;
    size_t calOffset;
    void* sliceAddr;
    size_t pitch;

    // Allocate a layer from the image
    gslSize.width = cal()->width_;
    gslSize.height = height;
    gslSize.depth = 1;
    calOffset = 0;
    sliceResource =
        dev().resAllocView(gslResource(), gslSize, calOffset, cal()->format_, cal()->channelOrder_,
                           gslDim, 0, i, CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER);
    if (0 == sliceResource) {
      LogError("Map layer. resAllocSliceView failed!");
      return NULL;
    }

    // Map 2D layer
    sliceAddr = gslMap(&pitch, GSL_MAP_READ_ONLY, sliceResource);
    if (sliceAddr == NULL) {
      LogError("Map layer. CalResMap failed!");
      return NULL;
    }

    srcOffs = 0;
    // Copy memory line by line
    for (size_t rows = 0; rows < height; ++rows) {
      // Copy memory
      amd::Os::fastMemcpy((reinterpret_cast<address>(address_) + dstOffs),
                          (reinterpret_cast<const_address>(sliceAddr) + srcOffs),
                          cal()->width_ * elementSize_);

      dstOffs += cal()->pitch_ * elementSize();
      srcOffs += pitch * elementSize();
    }

    // Unmap a layer
    gslUnmap(sliceResource);
    dev().resFree(sliceResource);
  }

  return address_;
}

void Resource::unmap(VirtualGPU* gpu) {
  if (isMemoryType(Pinned)) {
    return;
  }

  // Decrement map counter
  int count = --mapCount_;

  // Check if it's the last unmap
  if (count == 0) {
    if ((cal()->dimSize_ == 3) || cal()->imageArray_ ||
        ((cal()->type_ == ImageView) && viewOwner_->mipMapped())) {
      // Unmap layers
      unmapLayers(gpu);
    } else {
      // Unmap current resource
      gslUnmap(gslResource());
    }
    address_ = NULL;
  } else if (count < 0) {
    LogError("dev().serialCalResUnmap failed!");
    ++mapCount_;
    return;
  }
}

void Resource::unmapLayers(VirtualGPU* gpu) {
  size_t srcOffs = 0;
  size_t dstOffs = 0;
  gslMemObjectAttribType gslDim = GSL_MOA_TEXTURE_2D;
  gslMemObject sliceResource = NULL;
  CALuint layers = cal()->depth_;
  CALuint height = cal()->height_;

  // Use 1D layers
  if (GSL_MOA_TEXTURE_1D_ARRAY == cal()->dimension_) {
    gslDim = GSL_MOA_TEXTURE_1D;
    height = 1;
    layers = cal()->height_;
  }

  if (numLayers_ != 0) {
    layers = startLayer_ + numLayers_;
  }

  srcOffs = startLayer_ * cal()->slice_ * elementSize();

  // Check if map is write only
  if (!(mapFlags_ == GSL_MAP_READ_ONLY)) {
    // Loop through all layers
    for (uint i = startLayer_; i < layers; ++i) {
      gslResource3D gslSize;
      size_t calOffset;
      void* sliceAddr;
      size_t pitch;

      // Allocate a layer from the image
      gslSize.width = cal()->width_;
      gslSize.height = height;
      gslSize.depth = 1;
      calOffset = 0;
      sliceResource = dev().resAllocView(gslResource(), gslSize, calOffset, cal()->format_,
                                         cal()->channelOrder_, gslDim, 0, i,
                                         CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER);
      if (0 == sliceResource) {
        LogError("Unmap layer. resAllocSliceView failed!");
        return;
      }

      // Map a layer
      sliceAddr = gslMap(&pitch, GSL_MAP_WRITE_ONLY, sliceResource);
      if (sliceAddr == NULL) {
        LogError("Unmap layer. CalResMap failed!");
        return;
      }

      dstOffs = 0;
      // Copy memory line by line
      for (size_t rows = 0; rows < height; ++rows) {
        // Copy memory
        amd::Os::fastMemcpy((reinterpret_cast<address>(sliceAddr) + dstOffs),
                            (reinterpret_cast<const_address>(address_) + srcOffs),
                            cal()->width_ * elementSize_);

        dstOffs += pitch * elementSize();
        srcOffs += cal()->pitch_ * elementSize();
      }

      // Unmap a layer
      gslUnmap(sliceResource);
      dev().resFree(sliceResource);
    }
  }

  // Destroy the mapped memory
  delete[] reinterpret_cast<char*>(address_);
}

void Resource::setActiveRename(VirtualGPU& gpu, GslResourceReference* rename) {
  // Copy the unique GSL data
  gslRef_ = rename;
  address_ = rename->cpuAddress_;

  hbOffset_ = rename->gslResource()->getSurfaceAddress() - dev().heap().baseAddress();
}

bool Resource::getActiveRename(VirtualGPU& gpu, GslResourceReference** rename) {
  // Copy the old data to the rename descriptor
  *rename = gslRef_;
  return true;
}

bool Resource::rename(VirtualGPU& gpu, bool force) {
  GpuEvent* gpuEvent = gpu.getGpuEvent(gslResource());
  if (!gpuEvent->isValid() && !force) {
    return true;
  }

  bool useNext = false;
  CALuint resSize = cal()->width_ * ((cal()->height_) ? cal()->height_ : 1) * elementSize_;

  // Rename will work with real GSL resources
  if (((memoryType() != Local) && (memoryType() != Persistent) && (memoryType() != Remote) &&
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
    GslResourceReference* rename;
    if (mapCount_ > 0) {
      gslRef_->cpuAddress_ = address_;
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
    GslResourceReference* rename;

    // Create a new GSL allocation
    if (create(memoryType())) {
      if (mapCount_ > 0) {
        assert(!cal()->cardMemory_ && "Unsupported memory type!");
        gslRef_->cpuAddress_ = dev().resMapRemote(cal_.pitch_, gslResource(), GSL_MAP_READ_WRITE);
        if (gslRef_->cpuAddress_ == NULL) {
          LogError("gslMap fails on rename!");
        }
        address_ = gslRef_->cpuAddress_;
      }
      if (getActiveRename(gpu, &rename)) {
        curRename_ = renames_.size();
        renames_.push_back(rename);
      } else {
        gslRef_->release();
        useNext = true;
      }
    } else {
      useNext = true;
    }
  } else {
    useNext = true;
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

void Resource::warmUpRenames(VirtualGPU& gpu) {
  for (uint i = 0; i < dev().settings().maxRenames_; ++i) {
    // EPR #411675 - On Kaveri, benchmark "photo editing" of PCMarks takes longer time
    // if writing 0 for the buffer paging by VidMM is excuted. Not sure how PCMarks measures it.
    // Disable this code for apu
    if (!dev().settings().apuSystem_) {
      uint dummy = 0;
      const bool NoWait = false;
      // Write 0 for the buffer paging by VidMM
      writeRawData(gpu, sizeof(dummy), &dummy, NoWait);
    }
    const bool Force = true;
    rename(gpu, Force);
  }
}

ResourceCache::~ResourceCache() { free(); }

//! \note the cache works in FILO mode
bool ResourceCache::addCalResource(Resource::CalResourceDesc* desc, GslResourceReference* ref) {
  amd::ScopedLock l(&lockCacheOps_);
  bool result = false;
  size_t size = getResourceSize(desc);

  // Make sure current allocation isn't bigger than cache
  if (((desc->type_ == Resource::Local) || (desc->type_ == Resource::Persistent) ||
       (desc->type_ == Resource::Remote) || (desc->type_ == Resource::RemoteUSWC)) &&
      (size < cacheSizeLimit_) && !desc->skipRsrcCache_) {
    // Validate the cache size limit. Loop until we have enough space
    while ((cacheSize_ + size) > cacheSizeLimit_) {
      removeLast();
    }
    Resource::CalResourceDesc* descCached = new Resource::CalResourceDesc;
    if (descCached != NULL) {
      // Copy the original desc to the cached version
      memcpy(descCached, desc, sizeof(Resource::CalResourceDesc));

      // Add the current resource to the cache
      resCache_.push_front({descCached, ref});
      cacheSize_ += size;
      if (desc->type_ == Resource::Local) {
          lclCacheSize_ += size;
      }
      result = true;
    }
  }

  return result;
}

GslResourceReference* ResourceCache::findCalResource(Resource::CalResourceDesc* desc) {
  amd::ScopedLock l(&lockCacheOps_);
  GslResourceReference* ref = NULL;
  size_t size = getResourceSize(desc);

  // Early exit if resource is too big or it is for scratch buffer
  if (size >= cacheSizeLimit_ || desc->skipRsrcCache_ || desc->scratch_) {
    //! \note we may need to free the cache here to reduce memory pressure
    return ref;
  }

  // Serach the right resource through the cache list
  for (const auto& it : resCache_) {
    Resource::CalResourceDesc* entry = it.first;
    // Find if we can reuse this entry
    if ((entry->dimension_ == desc->dimension_) && (entry->type_ == desc->type_) &&
        (entry->width_ == desc->width_) && (entry->height_ == desc->height_) &&
        (entry->depth_ == desc->depth_) && (entry->channelOrder_ == desc->channelOrder_) &&
        (entry->format_ == desc->format_) && (entry->flags_ == desc->flags_) &&
        (entry->mipLevels_ == desc->mipLevels_) && (entry->isAllocSVM_ == desc->isAllocSVM_) &&
        (entry->isAllocExecute_ == desc->isAllocExecute_)) {
      ref = it.second;
      cacheSize_ -= size;
      if (entry->type_ == Resource::Local) {
          lclCacheSize_ -= size;
      }
      delete it.first;
      // Remove the found etry from the cache
      resCache_.remove(it);
      break;
    }
  }

  return ref;
}

bool ResourceCache::free(size_t minCacheEntries) {
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

size_t ResourceCache::getResourceSize(Resource::CalResourceDesc* desc) {
  // Find the total amount of elements
  size_t size =
      desc->width_ * ((desc->height_) ? desc->height_ : 1) * ((desc->depth_) ? desc->depth_ : 1);

  // Find total size in bytes
  size *= static_cast<size_t>(memoryFormatSize(desc->format_).size_);

  return size;
}

void ResourceCache::removeLast() {
  std::pair<Resource::CalResourceDesc*, GslResourceReference*> entry;
  entry = resCache_.back();
  resCache_.pop_back();

  size_t size = getResourceSize(entry.first);

  cacheSize_ -= size;
  if (entry.first->type_ == Resource::Local) {
      lclCacheSize_ -= size;
  }

  // Delete CalResourceDesc
  delete entry.first;

  // Destroy GSL resource
  entry.second->release();
}

}  // namespace gpu
