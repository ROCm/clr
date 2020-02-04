/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

#ifndef GPU_THREAD_TRACE_HPP_
#define GPU_THREAD_TRACE_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "device/gpu/gpudevice.hpp"
#include "shadertracebuffer/ShaderTraceBufferObject.h"

#include <vector>
namespace gpu {

class VirtualGPU;

class CalThreadTraceReference : public amd::ReferenceCountedObject {
 public:
  //! Default constructor
  CalThreadTraceReference(VirtualGPU& gpu,                //!< Virtual GPU device object
                          gslQueryObject gslThreadTrace)  //!< GSL query thread trace object
      : gpu_(gpu),
        threadTrace_(gslThreadTrace) {}

  //! Get GSL thread race object
  gslQueryObject gslThreadTrace() const { return threadTrace_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return gpu_; }

 protected:
  //! Default destructor
  ~CalThreadTraceReference();

 private:
  //! Disable copy constructor
  CalThreadTraceReference(const CalThreadTraceReference&);

  //! Disable operator=
  CalThreadTraceReference& operator=(const CalThreadTraceReference&);

  VirtualGPU& gpu_;             //!< The virtual GPU device object
  gslQueryObject threadTrace_;  //!< GSL thread trace query object
};

//! ThreadTrace implementation on GPU
class ThreadTrace : public device::ThreadTrace {
 public:
  //! Destructor for the GPU ThreadTrace object
  virtual ~ThreadTrace();

  //! Creates the current object
  bool create(CalThreadTraceReference* calRef  //!< Reference ThreadTrace
              );

  //! Returns the GPU device, associated with the current object
  const Device& dev() const { return gpuDevice_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return gpu_; }

  //! Constructor for the GPU ThreadTrace object
  ThreadTrace(Device& device,   //!< A GPU device object
              VirtualGPU& gpu,  //!< Virtual GPU device object
              uint amdThreadTraceMemObjsNum)
      : gpuDevice_(device),
        gpu_(gpu),
        calRef_(NULL),
        index_(0),
        amdThreadTraceMemObjsNum_(amdThreadTraceMemObjsNum) {
    threadTraceBufferObjs_ = new gslShaderTraceBufferObject[amdThreadTraceMemObjsNum];
    for (uint i = 0; i < amdThreadTraceMemObjsNum; ++i) {
      threadTraceBufferObjs_[i] = gpu.cs()->createShaderTraceBuffer();
    }
  }

  //! Returns the specific information about the thread trace object
  bool info(uint infoType,  //!< The type of returned information
            uint* info,     //!< The returned information
            uint infoSize   //!< The size of returned information
            ) const;

  //! Set the ThreadTrace memory buffer size
  void setMemBufferSizeTT(uint memBufferSizeTT) { memBufferSizeTT_ = memBufferSizeTT; }

  //! Set isNewBufferBinded_ to true/false if new buffer was binded/unbinded respectively
  void setNewBufferBinded(bool isNewBufferBinded) { isNewBufferBinded_ = isNewBufferBinded; }

  //! Attach gslMemObject to the TreadTrace buffer
  void attachMemToThreadTraceBuffer();

  void setMemObj(size_t memObjSize, std::vector<amd::Memory*> memObj) {
    memObj_ = memObj;
    memBufferSizeTT_ = memObjSize;
  }
  //! Get GSL thread trace object
  gslQueryObject gslThreadTrace() const { return threadTrace_; }

  //! Get GSL Thread Trace Buffer objects
  gslShaderTraceBufferObject* getThreadTraceBufferObjects() { return threadTraceBufferObjs_; }

 private:
  //! Disable default copy constructor
  ThreadTrace(const ThreadTrace&);

  //! Disable default operator=
  ThreadTrace& operator=(const ThreadTrace&);

  //! Retrieve gslMemoryObject
  gslMemObject getCurrentGslMemObject(amd::Memory*);

  const Device& gpuDevice_;  //!< The backend device

  VirtualGPU& gpu_;  //!< The virtual GPU device object

  CalThreadTraceReference* calRef_;  //!< Reference ThreadTrace
  gslShaderTraceBufferObject*
      threadTraceBufferObjs_;         //!< The buffer object for Thread Trace recording
  uint index_;                        //!< ThreadTrace index in the CAL container
  uint memBufferSizeTT_;              //!< ThreadTrace memory buffer size
  std::vector<amd::Memory*> memObj_;  //!< ThreadTrace memory object
  gslQueryObject threadTrace_;        //!< GSL thread trace query object
  uint amdThreadTraceMemObjsNum_;  //!< ThreadTrace memory object`s number (should be equal to the
                                   //!SE number)
  bool isNewBufferBinded_;  //!< The indicator if new buffer was binded to the ThreadTrace object
  bool isBufferOnSubmit_;   //!< The indicator if "new buffer on submit" mode is used
};

}  // namespace gpu

#endif  // GPU_THREAD_TRACE_HPP_
