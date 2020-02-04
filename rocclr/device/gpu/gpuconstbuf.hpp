/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

#ifndef GPUCONSTBUF_HPP_
#define GPUCONSTBUF_HPP_

#include "device/gpu/gpumemory.hpp"

//! \namespace gpu GPU Resource Implementation
namespace gpu {

//! Cconstant buffer
class ConstBuffer : public Memory {
 public:
  //! Vector size of the constant buffer
  static const size_t VectorSize = 16;

  //! Constructor for the ConstBuffer class
  ConstBuffer(VirtualGPU& gpu,  //!< Virtual GPU device object
              size_t size       //!< size of the constant buffer in vectors
              );

  //! Destructor for the ConstBuffer class
  ~ConstBuffer();

  //! Creates the real HW constant buffer
  bool create();

  /*! \brief Uploads current constant buffer data from sysMemCopy_ to HW
   *
   *  \return True if the data upload was succesful
   */
  bool uploadDataToHw(size_t size  //!< real data size for upload
                      );

  //! Returns a pointer to the system memory copy for CB
  address sysMemCopy() const { return sysMemCopy_; }

  //! Returns CB size
  size_t size() const { return size_; }

  //! Returns current write offset for the constant buffer
  size_t wrtOffset() const { return wrtOffset_; }

  //! Returns last write size for the constant buffer
  size_t lastWrtSize() const { return lastWrtSize_; }

 private:
  //! Disable copy constructor
  ConstBuffer(const ConstBuffer&);

  //! Disable operator=
  ConstBuffer& operator=(const ConstBuffer&);

  VirtualGPU& gpu_;     //!< Virtual GPU object
  address sysMemCopy_;  //!< System memory copy
  size_t size_;         //!< Constant buffer size
  size_t wrtOffset_;    //!< Current write offset
  size_t lastWrtSize_;  //!< Last write size
  void* wrtAddress_;    //!< Write address in CB
};


/*@}*/} // namespace gpu

#endif /*GPUCONSTBUF_HPP_*/
