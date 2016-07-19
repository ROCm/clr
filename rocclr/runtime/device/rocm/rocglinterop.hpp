//
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#ifdef _WIN32
//GLX header cannot be included in Windows due to X11 header dependency
#define MESA_GLINTEROP_NO_GLX
#include "device/rocm/mesa_glinterop.h"
//Give GLX parameters void* size
typedef void Display;
typedef void* GLXContext;
#undef MESA_GLINTEROP_NO_GLX
#else
#include "device/rocm/mesa_glinterop.h"
#endif

#include "device/rocm/rocregisters.hpp"
#include "hsa_ext_amd.h"

#include <atomic>

namespace roc
{
  
  //Specific typed container for version 1
  typedef struct metadata_amd_ci_vi_s {
    uint32_t version; // Must be 1
    uint32_t vendorID; // AMD | CZ
    SQ_IMG_RSRC_WORD0 word0;
    SQ_IMG_RSRC_WORD1 word1;
    SQ_IMG_RSRC_WORD2 word2;
    SQ_IMG_RSRC_WORD3 word3;
    SQ_IMG_RSRC_WORD4 word4;
    SQ_IMG_RSRC_WORD5 word5;
    SQ_IMG_RSRC_WORD6 word6;
    SQ_IMG_RSRC_WORD7 word7;
    uint32_t mip_offsets[0]; //Mip level offset bits [39:8] for each level (if any)
  } metadata_amd_ci_vi_t;

  class image_metadata
  {
  private:
    metadata_amd_ci_vi_t* data;

    image_metadata(const image_metadata&)=delete;
    image_metadata& operator=(const image_metadata&)=delete;

  public:
    image_metadata() : data(nullptr) {}
    ~image_metadata() { data=nullptr; }

    bool create(hsa_amd_image_descriptor_t* image_desc)
    {
      if((image_desc->version!=1) || ((image_desc->deviceID>>16)!=0x1002)) return false;
      data=reinterpret_cast<metadata_amd_ci_vi_t*>(image_desc);
      return true;
    }

    bool setMipLevel(uint32_t level)
    {
      if(level>data->word3.bits.last_level)
        return false;
      data->word3.bits.base_level=level;
      data->word3.bits.last_level=level;
      return true;
    }

    bool setLayer(uint32_t layer)
    {
      data->word3.bits.type=SQ_RSRC_IMG_2D_ARRAY;
      data->word5.bits.last_array=layer;
      data->word5.bits.base_array=layer;
      return true;
    }

    bool setFace(GLenum face)
    {
      int index=face-GL_TEXTURE_CUBE_MAP_POSITIVE_X;
      if(index<0 || index>5)
        return false;
      if(data->word3.bits.type!=SQ_RSRC_IMG_CUBE)
        return false;
      return setLayer(index);
    }
  };

  class MesaInterop
  {
  public:

    enum MESA_INTEROP_KIND { MESA_INTEROP_NONE=0, MESA_INTEROP_GLX=1, MESA_INTEROP_EGL=2 };

    union DisplayHandle
    {
      Display* glxDisplay;
      EGLDisplay eglDisplay;
    };

    union ContextHandle
    {
      GLXContext glxContext;
      EGLContext eglContext;
    };

    //True if the configuration supports the indicated interop ability.
    static bool Supported();

    MesaInterop() { kind=MESA_INTEROP_NONE; }
    MesaInterop(const MesaInterop& rhs) { *this=rhs; }
    ~MesaInterop() { Unbind(); }

    const MesaInterop& operator=(const MesaInterop& rhs)
    {
      display=rhs.display;
      context=rhs.context;
      kind=rhs.kind;
      if(kind!=MESA_INTEROP_NONE)
        refCount++;
	  return *this;
    }

    /*
    Loads Mesa interop APIs and sets this interface object to use the indicated
    subsystem (GLX/EGL).  Returns true if the required subsystem is found.
    */
    bool Bind(MESA_INTEROP_KIND Kind, const DisplayHandle& Display, const ContextHandle& Context);

    /*
    Releases use of Mesa interop APIs.
    Used to check for bad load/unload sequences.
    */
    void Unbind()
    {
      if(kind==MESA_INTEROP_NONE) return;
      assert(refCount>0 && "Invalid refCount in MesaInterop.");
      refCount--;
      kind=MESA_INTEROP_NONE;
    }

    bool GetInfo(mesa_glinterop_device_info& info) const;

    bool Export (mesa_glinterop_export_in& in, mesa_glinterop_export_out& out) const;

  private:
    static std::atomic<uint32_t> refCount;

    DisplayHandle display;
    ContextHandle context;
    MESA_INTEROP_KIND kind;
  };

}

#endif  /*WITHOUT_HSA_BACKEND*/

