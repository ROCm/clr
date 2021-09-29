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


namespace amd {
static std::once_flag interopOnce;
}
// Sets up GL context association with amd context.
// NOTE: Refer to Context setup code in OCLTestImp.cpp
void setupGLInteropOnce() {
  amd::Context* amdContext = hip::getCurrentDevice()->asContext();

//current context will be read in amdContext->create
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)AMD_PLATFORM,
                                        ROCCLR_HIP_GL_CONTEXT_KHR,
                                        (cl_context_properties) nullptr,
#ifdef _WIN32
                                        ROCCLR_HIP_WGL_HDC_KHR,
                                        (cl_context_properties) nullptr,
#else
                                        ROCCLR_HIP_GLX_DISPLAY_KHR,
                                        (cl_context_properties) nullptr,
#endif
                                        0};

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
