//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef INTEROP_H_
#define INTEROP_H_

namespace amd {

//! Forward declarations of interop classes
class GLObject;
class BufferGL;

#ifdef _WIN32
class D3D10Object;
class D3D11Object;
class D3D9Object;
#endif  //_WIN32

//! Base object providing common map/unmap interface for interop objects
class InteropObject {
 public:
  //! Virtual destructor to get rid of linux warning
  virtual ~InteropObject() {}

  // Static cast functions for interop objects
  virtual GLObject* asGLObject() { return NULL; }
  virtual BufferGL* asBufferGL() { return NULL; }

#ifdef _WIN32
  virtual D3D10Object* asD3D10Object() { return NULL; }
  virtual D3D11Object* asD3D11Object() { return NULL; }
  virtual D3D9Object* asD3D9Object() { return NULL; }
#endif  //_WIN32

  // On acquire copy data from original resource to shared resource
  virtual bool copyOrigToShared() { return true; }
  // On release copy data from shared copy to the original resource
  virtual bool copySharedToOrig() { return true; }
};

}  // namespace amd

#endif  //! INTEROP_H_
