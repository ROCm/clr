//
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

#ifndef _WIN32
#include <GL/glx.h>
#include <EGL/egl.h>
#else
#include <GL/gl.h>
#include <EGL/egl.h>
typedef _XDisplay Display;
typedef __GLXcontextRec* GLXContext;
#endif

#include "device/rocm/mesa_glinterop.h"
#include "device/rocm/rocregisters.hpp"
#include "hsa_ext_amd.h"

namespace roc {

// Specific typed container for version 1
typedef struct metadata_amd_ci_vi_s {
  uint32_t version;   // Must be 1
  uint32_t vendorID;  // AMD | CZ
  SQ_IMG_RSRC_WORD0 word0;
  SQ_IMG_RSRC_WORD1 word1;
  SQ_IMG_RSRC_WORD2 word2;
  SQ_IMG_RSRC_WORD3 word3;
  SQ_IMG_RSRC_WORD4 word4;
  SQ_IMG_RSRC_WORD5 word5;
  SQ_IMG_RSRC_WORD6 word6;
  SQ_IMG_RSRC_WORD7 word7;
  uint32_t mip_offsets[0];  // Mip level offset bits [39:8] for each level (if any)
} metadata_amd_ci_vi_t;

class image_metadata {
 private:
  metadata_amd_ci_vi_t* data;

  image_metadata(const image_metadata&) = delete;
  image_metadata& operator=(const image_metadata&) = delete;

 public:
  image_metadata() : data(nullptr) {}
  ~image_metadata() { data = nullptr; }

  bool create(hsa_amd_image_descriptor_t* image_desc) {
    if ((image_desc->version != 1) || ((image_desc->deviceID >> 16) != 0x1002)) return false;
    data = reinterpret_cast<metadata_amd_ci_vi_t*>(image_desc);
    return true;
  }

  bool setMipLevel(uint32_t level) {
    if (level > data->word3.bits.last_level) return false;
    data->word3.bits.base_level = level;
    data->word3.bits.last_level = level;
    return true;
  }

  bool setLayer(uint32_t layer) {
    data->word3.bits.type = SQ_RSRC_IMG_2D_ARRAY;
    data->word5.bits.last_array = layer;
    data->word5.bits.base_array = layer;
    return true;
  }

  bool setFace(GLenum face) {
    int index = face - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    if (index < 0 || index > 5) return false;
    if (data->word3.bits.type != SQ_RSRC_IMG_CUBE) return false;
    return setLayer(index);
  }
};

namespace MesaInterop {
enum MESA_INTEROP_KIND { MESA_INTEROP_NONE = 0, MESA_INTEROP_GLX = 1, MESA_INTEROP_EGL = 2 };

union DisplayHandle {
  Display* glxDisplay;
  EGLDisplay eglDisplay;
  DisplayHandle() {}
  DisplayHandle(Display* display) : glxDisplay(display) {}
  DisplayHandle(EGLDisplay display) : eglDisplay(display) {}
};

union ContextHandle {
  GLXContext glxContext;
  EGLContext eglContext;
  ContextHandle() {}
  ContextHandle(GLXContext context) : glxContext(context) {}
  ContextHandle(EGLContext context) : eglContext(context) {}
};

// True if the build supports Mesa interop.
bool Supported();

// Returns true if the required subsystem is supported on the GL device.
// Must be called at least once, may be called multiple times.
bool Init(MESA_INTEROP_KIND Kind);

bool GetInfo(mesa_glinterop_device_info& info, MESA_INTEROP_KIND Kind, const DisplayHandle display,
             const ContextHandle context);

bool Export(mesa_glinterop_export_in& in, mesa_glinterop_export_out& out, MESA_INTEROP_KIND Kind,
            const DisplayHandle display, const ContextHandle context);
}
}

#endif /*WITHOUT_HSA_BACKEND*/
