//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "device/pal/palmemory.hpp"

//! \namespace pal PAL Resource Implementation
namespace pal {

//! Managed buffer (staging or constant)
class ManagedBuffer : public amd::HeapObject {
 public:
  //! Constructor for the ConstBuffer class
  ManagedBuffer(VirtualGPU& gpu,    //!< Virtual GPU device object
                uint32_t    size    //!< size of the managed buffers in bytes
                );

  //! Destructor for the ConstBuffer class
  ~ManagedBuffer();

  //! Creates the real HW constant buffer
  bool create(Resource::MemoryType type, bool constBuf = false);

  /*! \brief Uploads current constant buffer data from sysMemCopy_ to HW
   *
   *  \return True if the data upload was succesful
   */
  bool uploadDataToHw(uint32_t size  //!< real data size for upload
                      );

  //! Returns a pointer to the system memory copy for CB
  address sysMemCopy() const { return sysMemCopy_; }

  //! Returns CB size
  uint32_t size() const { return size_; }

  //! Returns current write offset for the constant buffer
  uint32_t wrtOffset() const { return wrtOffset_; }

  //! Returns last write size for the constant buffer
  uint32_t lastWrtSize() const { return lastWrtSize_; }

  Memory* activeMemory() const { return buffers_[activeBuffer_]; }

  uint64_t vmAddress() const { return buffers_[activeBuffer_]->vmAddress(); }

 private:
  //! The maximum number of the managed buffers
  static constexpr uint32_t MaxNumberOfBuffers = 3;

  //! Disable copy constructor
  ManagedBuffer(const ManagedBuffer&) = delete;

  //! Disable operator=
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  VirtualGPU& gpu_;                 //!< Virtual GPU object
  std::vector<Memory*>  buffers_;   //!< Buffers for management
  uint32_t  activeBuffer_;          //!< Current active buffer
  address   sysMemCopy_;            //!< System memory copy
  uint32_t  size_;                  //!< Constant buffer size
  uint32_t  wrtOffset_;             //!< Current write offset
  uint32_t  lastWrtSize_;           //!< Last write size
  void*     wrtAddress_;            //!< Write address in CB
};

/*@}*/} // namespace pal
