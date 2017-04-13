//
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef WITHOUT_HSA_BACKEND

#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "device/rocm/rocglinterop.hpp"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace roc {

#if !defined(_WIN32)
static PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC GlxInfo = nullptr;
static PFNMESAGLINTEROPGLXEXPORTOBJECTPROC GlxExport = nullptr;
static PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC EglInfo = nullptr;
static PFNMESAGLINTEROPEGLEXPORTOBJECTPROC EglExport = nullptr;
#endif

std::atomic<uint32_t> MesaInterop::refCount(0);

bool MesaInterop::Supported() {
#ifdef _WIN32
  return false;
#else
  return true;
#endif
}

// Attempt to locate Mesa interop APIs.  Return which of glx/egl are supported.
bool MesaInterop::Bind(MESA_INTEROP_KIND Kind, const DisplayHandle& Display,
                       const ContextHandle& Context) {
#if defined(_WIN32)
  return false;
#else
  if (Kind == MESA_INTEROP_NONE) return false;

  if (kind != MESA_INTEROP_NONE) {
    LogError("Error - MesaInterop Bind while already bound.");
    return false;
  }

  void* glxinfo = dlsym(RTLD_DEFAULT, "MesaGLInteropGLXQueryDeviceInfo");
  void* eglinfo = dlsym(RTLD_DEFAULT, "MesaGLInteropEGLQueryDeviceInfo");

  if (((glxinfo != GlxInfo) || (eglinfo != EglInfo)) && (refCount != 0))
    LogWarning("Warning - Mesa changed while holding interop contexts.");

  GlxInfo = (PFNMESAGLINTEROPGLXQUERYDEVICEINFOPROC)glxinfo;
  EglInfo = (PFNMESAGLINTEROPEGLQUERYDEVICEINFOPROC)eglinfo;

  GlxExport =
      (PFNMESAGLINTEROPGLXEXPORTOBJECTPROC)dlsym(RTLD_DEFAULT, "MesaGLInteropGLXExportObject");
  EglExport =
      (PFNMESAGLINTEROPEGLEXPORTOBJECTPROC)dlsym(RTLD_DEFAULT, "MesaGLInteropEGLExportObject");

  uint32_t ret = MESA_INTEROP_NONE;
  if (GlxInfo && GlxExport) ret |= MESA_INTEROP_GLX;
  if (EglInfo && EglExport) ret |= MESA_INTEROP_EGL;

  kind = MESA_INTEROP_KIND(ret & Kind);
  display = Display;
  context = Context;

  if (kind != MESA_INTEROP_NONE) {
    refCount++;
    return true;
  }
  return false;

#endif
}

bool MesaInterop::GetInfo(mesa_glinterop_device_info& info) const {
#ifdef _WIN32
  return false;
#else
  switch (kind) {
    case MESA_INTEROP_GLX:
      return GlxInfo(display.glxDisplay, context.glxContext, &info) == MESA_GLINTEROP_SUCCESS;
    case MESA_INTEROP_EGL:
      return EglInfo(display.eglDisplay, context.eglContext, &info) == MESA_GLINTEROP_SUCCESS;
    default:
      return false;
  }
#endif
}

bool MesaInterop::Export(mesa_glinterop_export_in& in, mesa_glinterop_export_out& out) const {
#ifdef _WIN32
  return false;
#else
  switch (kind) {
    case MESA_INTEROP_GLX:
      return GlxExport(display.glxDisplay, context.glxContext, &in, &out) == MESA_GLINTEROP_SUCCESS;
    case MESA_INTEROP_EGL:
      return EglExport(display.eglDisplay, context.eglContext, &in, &out) == MESA_GLINTEROP_SUCCESS;
    default:
      return false;
  }
#endif
}
}

#endif  // WITHOUT_HSA_BACKEND
