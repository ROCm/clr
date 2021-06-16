/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "top.hpp"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "cl_gl_amd.hpp"
#include "cl_common.hpp"
#include <GL/gl.h>
#include <GL/glext.h>

#ifndef _WIN32
#include <GL/glx.h>
#endif

namespace amd {
static std::once_flag interopOnce;
}

#ifdef _WIN32
// Creates Device and GL Contexts to be associated with hip Context.
// Refer to bool OCLGLCommon::initializeGLContext(OCLGLHandle& hGL) in OCLGLCommonWindows.cpp
static inline bool getDeviceGLContext(HDC& dc, HGLRC& glrc) {
  BOOL glErr = FALSE;
  DISPLAY_DEVICE dispDevice;
  DWORD deviceNum;
  int pfmt;
  PIXELFORMATDESCRIPTOR pfd;
  pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 24;
  pfd.cRedBits = 8;
  pfd.cRedShift = 0;
  pfd.cGreenBits = 8;
  pfd.cGreenShift = 0;
  pfd.cBlueBits = 8;
  pfd.cBlueShift = 0;
  pfd.cAlphaBits = 8;
  pfd.cAlphaShift = 0;
  pfd.cAccumBits = 0;
  pfd.cAccumRedBits = 0;
  pfd.cAccumGreenBits = 0;
  pfd.cAccumBlueBits = 0;
  pfd.cAccumAlphaBits = 0;
  pfd.cDepthBits = 24;
  pfd.cStencilBits = 8;
  pfd.cAuxBuffers = 0;
  pfd.iLayerType = PFD_MAIN_PLANE;
  pfd.bReserved = 0;
  pfd.dwLayerMask = 0;
  pfd.dwVisibleMask = 0;
  pfd.dwDamageMask = 0;

  dispDevice.cb = sizeof(DISPLAY_DEVICE);
  for (deviceNum = 0; EnumDisplayDevices(nullptr, deviceNum, &dispDevice, 0); deviceNum++) {
    if (dispDevice.StateFlags & DISPLAY_DEVICE_MIRRORING_DRIVER) {
      continue;
    }

    dc = CreateDC(nullptr, dispDevice.DeviceName, nullptr, nullptr);
    if (!dc) {
      continue;
    }

    pfmt = ChoosePixelFormat(dc, &pfd);
    if (pfmt == 0) {
      LogError("Failed choosing the requested PixelFormat.\n");
      return false;
    }

    glErr = SetPixelFormat(dc, pfmt, &pfd);
    if (glErr == FALSE) {
      LogError("Failed to set the requested PixelFormat.\n");
      return false;
    }

    DWORD err = GetLastError();

    glrc = wglCreateContext(dc);
    if (nullptr == glrc) {
      err = GetLastError();
      LogPrintfError("\n wglCreateContext() failed error: 0x%x\n", err);
      return false;
    }

    glErr = wglMakeCurrent(dc, glrc);
    if (FALSE == glErr) {
      LogError("\n wglMakeCurrent() failed\n");
      return false;
    }

    return true;
  }
  return false;
}
#else
struct OCLGLHandle_ {
  static Display* display;
  static XVisualInfo* vInfo;
  static int referenceCount;
  GLXContext context;
  Window window;
  Colormap cmap;
};


#define DEFAULT_OPENGL  "libGL.so"
Display* OCLGLHandle_::display = nullptr;
XVisualInfo* OCLGLHandle_::vInfo = nullptr;
int OCLGLHandle_::referenceCount = 0;

static inline bool getDeviceGLContext(OCLGLHandle_* hGL) {

  amd::GLFunctions* glenv_;
  void* h = amd::Os::loadLibrary(DEFAULT_OPENGL);

  if (h && (glenv_ = new amd::GLFunctions(h, false))) {
    LogError("\n couldn't load opengl\n");
    return false;
  }


  hGL->display = XOpenDisplay(nullptr);
  if (hGL->display == nullptr) {
    printf("XOpenDisplay() failed\n");
    return false;
  }

  if (hGL->vInfo == nullptr) {
    int dblBuf[] = {GLX_RGBA, GLX_RED_SIZE,   1,  GLX_GREEN_SIZE,   1,   GLX_BLUE_SIZE,
                  1,        GLX_DEPTH_SIZE, 12, GLX_DOUBLEBUFFER, None};

    hGL->vInfo = glenv_->glXChooseVisual_(hGL->display, DefaultScreen(hGL->display), dblBuf);
    if (hGL->vInfo == nullptr) {
      printf("glXChooseVisual() failed\n");
      return false;
    }
  }
  hGL->referenceCount++;

  hGL->context = glenv_->glXCreateContext_(hGL->display, hGL->vInfo, None, True);
  if (hGL->context == nullptr) {
    printf("glXCreateContext() failed\n");
    return false;
  }

  return true;
}
#endif

// Sets up GL context association with amd context.
// NOTE: Refer to Context setup code in OCLTestImp.cpp
void setupGLInteropOnce() {
  amd::Context* amdContext = hip::getCurrentDevice()->asContext();

#ifdef _WIN32
  HDC dc;
  HGLRC glrc;

  if (!getDeviceGLContext(dc, glrc)) {
    LogError(" Context setup failed \n");
    return;
  }

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)AMD_PLATFORM,
                                        CL_GL_CONTEXT_KHR,
                                        (cl_context_properties)glrc,
                                        CL_WGL_HDC_KHR,
                                        (cl_context_properties)dc,
                                        0};



#else
  OCLGLHandle_* hGL = new OCLGLHandle_;

  hGL->context = nullptr;
  hGL->window = 0;
  hGL->cmap = 0;

  if (!getDeviceGLContext(hGL)) {
    LogError(" Context setup failed \n");
    return;
  }

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)AMD_PLATFORM,
                                        CL_GL_CONTEXT_KHR,
                                        (cl_context_properties)hGL->context,
                                        CL_GLX_DISPLAY_KHR,
                                        (cl_context_properties)hGL->display,
                                        0};
#endif

  amd::Context::Info info;
  if (CL_SUCCESS != amd::Context::checkProperties(properties, &info)) {
    LogError("Context setup failed \n");
    return;
  }

  amdContext->setInfo(info);
  if (CL_SUCCESS != amdContext->create(properties)) {
    LogError("Context setup failed \n");
  }
}

static inline hipError_t hipSetInteropObjects(int num_objects, void** mem_objects,
                                              std::vector<amd::Memory*>& interopObjects) {
  if ((num_objects == 0 && mem_objects != nullptr) || (num_objects != 0 && mem_objects == nullptr)) {
    return hipErrorUnknown;
  }

  while (num_objects-- > 0) {
    void* obj = *mem_objects++;
    if (obj == nullptr) {
      return hipErrorInvalidResourceHandle;
    }

    amd::Memory* mem = reinterpret_cast<amd::Memory*>(obj);

    if (mem->getInteropObj() == nullptr) {
      return hipErrorInvalidResourceHandle;
    }

    interopObjects.push_back(mem);
  }
  return hipSuccess;
}

// NOTE: This method cooresponds to OpenCL functionality in clGetGLContextInfoKHR()
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
  HIP_INIT_API(hipGLGetDevices, pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList);

  std::call_once(amd::interopOnce, setupGLInteropOnce);

  static const bool VALIDATE_ONLY = true;
  if (deviceList == hipGLDeviceListNextFrame) {
    LogError(" hipGLDeviceListNextFrame not supported yet.\n");
    HIP_RETURN(hipErrorNotSupported);
  }
  if (pHipDeviceCount == nullptr || pHipDevices == nullptr || hipDeviceCount == 0) {
    LogError(" Invalid Argument \n");
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipDeviceCount = std::min(hipDeviceCount, static_cast<unsigned int>(g_devices.size()));

  amd::Context::Info info = hip::getCurrentDevice()->asContext()->info();
  if (!(info.flags_ & amd::Context::GLDeviceKhr)) {
    LogError("Failed : Invalid Shared Group Reference \n");
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pHipDeviceCount = 0;
  switch (deviceList) {
    case hipGLDeviceListCurrentFrame:
      for (int i = 0; i < hipDeviceCount; ++i) {
        const std::vector<amd::Device*>& devices = g_devices[i]->devices();
        if (devices.size() > 0 &&
            devices[0]->bindExternalDevice(info.flags_, info.hDev_, info.hCtx_, VALIDATE_ONLY)) {
          pHipDevices[0] = i;
          *pHipDeviceCount = 1;
          break;
        }
      }
      break;

    case hipGLDeviceListAll: {
      int foundDeviceCount = 0;
      for (int i = 0; i < hipDeviceCount; ++i) {
        const std::vector<amd::Device*>& devices = g_devices[i]->devices();
        if (devices.size() > 0 &&
            devices[0]->bindExternalDevice(info.flags_, info.hDev_, info.hCtx_, VALIDATE_ONLY)) {
          pHipDevices[foundDeviceCount++] = i;
          break;
        }
      }

      *pHipDeviceCount = foundDeviceCount;
    } break;

    default:
      LogWarning("Invalid deviceList value");
      HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(*pHipDeviceCount > 0 ? hipSuccess : hipErrorNoDevice);
}

static inline void clearGLErrors(const amd::Context& amdContext) {
  GLenum glErr, glLastErr = GL_NO_ERROR;
  while (1) {
    glErr = amdContext.glenv()->glGetError_();
    if (glErr == GL_NO_ERROR || glErr == glLastErr) {
      break;
    }
    glLastErr = glErr;
    LogWarning("GL error");
  }
}

static inline GLenum checkForGLError(const amd::Context& amdContext) {
  GLenum glRetErr = GL_NO_ERROR;
  GLenum glErr;
  while (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
    glRetErr = glErr;  // Just return the last GL error
    LogWarning("Check GL error");
  }
  return glRetErr;
}

hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags) {
  HIP_INIT_API(hipGraphicsGLRegisterBuffer, resource, buffer, flags);

  if (!((flags == hipGraphicsRegisterFlagsNone) || (flags == hipGraphicsRegisterFlagsReadOnly) ||
        (flags == hipGraphicsRegisterFlagsWriteDiscard) ||
        (flags == hipGraphicsRegisterFlagsSurfaceLoadStore) ||
        (flags == hipGraphicsRegisterFlagsTextureGather))) {
    LogError("invalid parameter \"flags\"");
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::BufferGL* pBufferGL = nullptr;
  GLenum glErr;
  GLenum glTarget = GL_ARRAY_BUFFER;
  GLint gliSize = 0;
  GLint gliMapped = 0;

  amd::Context& amdContext = *(hip::getCurrentDevice()->asContext());

  // Add this scope to bound the scoped lock
  {
    amd::GLFunctions::SetIntEnv ie(amdContext.glenv());
    if (!ie.isValid()) {
      LogWarning("\"amdContext\" is not created from GL context or share list \n");
      HIP_RETURN(hipErrorUnknown);
    }

    // Verify GL buffer object
    clearGLErrors(amdContext);
    if ((GL_FALSE == amdContext.glenv()->glIsBuffer_(buffer)) ||
        (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_()))) {
      LogWarning("\"buffer\" is not a GL buffer object \n");
      HIP_RETURN(hipErrorInvalidResourceHandle);
    }

    // Check if size is available - data store is created
    amdContext.glenv()->glBindBuffer_(glTarget, buffer);
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetBufferParameteriv_(glTarget, GL_BUFFER_SIZE, &gliSize);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      LogWarning("cannot get the GL buffer size \n");
      HIP_RETURN(hipErrorInvalidResourceHandle);
    }
    if (gliSize == 0) {
      LogWarning("the GL buffer's data store is not created \n");
      HIP_RETURN(hipErrorInvalidResourceHandle);
    }

  }  // Release scoped lock

  // Now create BufferGL object
  pBufferGL = new (amdContext) amd::BufferGL(amdContext, flags, gliSize, 0, buffer);

  if (!pBufferGL) {
    LogWarning("cannot create object of class BufferGL");
    HIP_RETURN(hipErrorUnknown);
  }

  if (!pBufferGL->create()) {
    pBufferGL->release();
    HIP_RETURN(hipErrorUnknown);
  }

  // Create interop object
  if (pBufferGL->getInteropObj() == nullptr) {
    LogWarning("cannot create object of class BufferGL");
    HIP_RETURN(hipErrorUnknown);
  }

  // Fixme: If more than one device is present in the context, we choose the first device.
  // We should come up with a more elegant solution to handle this.
  assert(amdContext.devices().size() == 1);

  const auto it = amdContext.devices().cbegin();
  const amd::Device& dev = *(*it);

  device::Memory* mem = pBufferGL->getDeviceMemory(dev);
  if (nullptr == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", pBufferGL->getSize());
    HIP_RETURN(hipErrorUnknown);
  }
  mem->processGLResource(device::Memory::GLDecompressResource);

  *resource = reinterpret_cast<hipGraphicsResource*>(pBufferGL);

  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream) {
  HIP_INIT_API(hipGraphicsMapResources, count, resources, stream);
  amd::Context* amdContext = hip::getCurrentDevice()->asContext();
  if (!amdContext || !amdContext->glenv()) {
    HIP_RETURN(hipErrorUnknown);
  }
  clearGLErrors(*amdContext);
  amdContext->glenv()->glFinish_();
  if (checkForGLError(*amdContext) != GL_NO_ERROR) {
    HIP_RETURN(hipErrorUnknown);
  }

  amd::HostQueue* queue = hip::getQueue(stream);
  if (nullptr == queue) {
    HIP_RETURN(hipErrorUnknown);
  }
  amd::HostQueue& hostQueue = *queue;

  if (!hostQueue.context().glenv() || !hostQueue.context().glenv()->isAssociated()) {
    LogWarning("\"amdContext\" is not created from GL context or share list");
    HIP_RETURN(hipErrorUnknown);
  }

  std::vector<amd::Memory*> memObjects;
  hipError_t err = hipSetInteropObjects(count, reinterpret_cast<void**>(resources), memObjects);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }

  amd::Command::EventWaitList nullWaitList;

  //! Now create command and enqueue
  amd::AcquireExtObjectsCommand* command = new amd::AcquireExtObjectsCommand(
      hostQueue, nullWaitList, count, memObjects, CL_COMMAND_ACQUIRE_GL_OBJECTS);
  if (command == nullptr) {
    HIP_RETURN(hipErrorUnknown);
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    HIP_RETURN(hipErrorUnknown);
  }

  command->enqueue();

  // *not_null(event) = as_cl(&command->event());
  if (as_cl(&command->event()) == nullptr) {
    command->release();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource) {
  HIP_INIT_API(hipGraphicsResourceGetMappedPointer, devPtr, size, resource);
  amd::Context* amdContext = hip::getCurrentDevice()->asContext();
  if (!amdContext || !amdContext->glenv()) {
    HIP_RETURN(hipErrorUnknown);
  }

  // Fixme: If more than one device is present in the context, we choose the first device.
  // We should come up with a more elegant solution to handle this.
  assert(amdContext->devices().size() == 1);

  const auto it = amdContext->devices().cbegin();

  amd::Device* curDev = *it;
  amd::Memory* amdMem = reinterpret_cast<amd::Memory*>(resource);
  *size = amdMem->getSize();

  // Interop resources don't have svm allocations they are added to
  // amd::MemObjMap using device virtual address during creation.
  device::Memory* mem = reinterpret_cast<device::Memory*>(amdMem->getDeviceMemory(*curDev));
  *devPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(mem->virtualAddress()));

  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream) {
  HIP_INIT_API(hipGraphicsUnmapResources, count, resources, stream);
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }

  // Wait for the current host queue
  hip::getQueue(stream)->finish();

  amd::HostQueue* queue = hip::getQueue(stream);
  if (nullptr == queue) {
    HIP_RETURN(hipErrorUnknown);
  }
  amd::HostQueue& hostQueue = *queue;

  std::vector<amd::Memory*> memObjects;
  hipError_t err = hipSetInteropObjects(count, reinterpret_cast<void**>(resources), memObjects);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }

  amd::Command::EventWaitList nullWaitList;

  // Now create command and enqueue
  amd::ReleaseExtObjectsCommand* command = new amd::ReleaseExtObjectsCommand(
      hostQueue, nullWaitList, count, memObjects, CL_COMMAND_RELEASE_GL_OBJECTS);
  if (command == nullptr) {
    HIP_RETURN(hipErrorUnknown);
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    HIP_RETURN(hipErrorUnknown);
  }

  command->enqueue();

  if (as_cl(&command->event()) == nullptr) {
    command->release();
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource) {
  HIP_INIT_API(hipGraphicsUnregisterResource, resource);

  amd::BufferGL* pBufferGL = reinterpret_cast<amd::BufferGL*>(resource);
  delete pBufferGL;

  HIP_RETURN(hipSuccess);
}
