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

#ifndef GPUCOUNTERS_HPP_
#define GPUCOUNTERS_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "device/gpu/gpudevice.hpp"

namespace gpu {

class VirtualGPU;

class CalCounterReference : public amd::ReferenceCountedObject {
 public:
  //! Default constructor
  CalCounterReference(VirtualGPU& gpu,  //!< Virtual GPU device object
                      gslQueryObject gslCounter)
      : gpu_(gpu), counter_(gslCounter), results_(NULL) {}

  //! Get CAL counter
  gslQueryObject gslCounter() const { return counter_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return gpu_; }

  //! Increases the results array for this CAL counter(container)
  bool growResultArray(uint maxIndex  //!< the maximum HW counter index in the CAL counter
                       );

  //! Returns the CAL counter results
  uint64_t* results() const { return results_; }

 protected:
  //! Default destructor
  ~CalCounterReference();

 private:
  //! Disable copy constructor
  CalCounterReference(const CalCounterReference&);

  //! Disable operator=
  CalCounterReference& operator=(const CalCounterReference&);

  VirtualGPU& gpu_;         //!< The virtual GPU device object
  gslQueryObject counter_;  //!< GSL object counter
  uint64_t* results_;       //!< CAL counter results
};

//! Performance counter implementation on GPU
class PerfCounter : public device::PerfCounter {
 public:
  //! The performance counter info
  struct Info : public amd::EmbeddedObject {
    uint blockIndex_;    //!< Index of the block to configure
    uint counterIndex_;  //!< Index of the hardware counter
    uint eventIndex_;    //!< Event you wish to count with the counter
  };

  //! The PerfCounter flags
  enum Flags { BeginIssued = 0x00000001, EndIssued = 0x00000002, ResultReady = 0x00000004 };

  //! Constructor for the GPU PerfCounter object
  PerfCounter(const Device& device,   //!< A GPU device object
              const VirtualGPU& gpu,  //!< Virtual GPU device object
              uint32_t blockIndex,     //!< HW block index
              uint32_t counterIndex,   //!< Counter index within the block
              uint32_t eventIndex)     //!< Event index for profiling
      : gpuDevice_(device),
        gpu_(gpu),
        calRef_(NULL),
        flags_(0),
        counter_(0),
        index_(0) {
    info_.blockIndex_ = blockIndex;
    info_.counterIndex_ = counterIndex;
    info_.eventIndex_ = eventIndex;
  }

  //! Destructor for the GPU PerfCounter object
  virtual ~PerfCounter();

  //! Creates the current object
  bool create(CalCounterReference* calRef  //!< Reference counter
              );

  //! Returns the specific information about the counter
  uint64_t getInfo(uint64_t infoType  //!< The type of returned information
                   ) const;

  //! Returns the GPU device, associated with the current object
  const Device& dev() const { return gpuDevice_; }

  //! Returns the virtual GPU device
  const VirtualGPU& gpu() const { return gpu_; }

  //! Returns the CAL performance counter descriptor
  const Info* info() const { return &info_; }

  //! Returns the Info structure for performance counter
  gslQueryObject gslCounter() const { return counter_; }

 private:
  //! Disable default copy constructor
  PerfCounter(const PerfCounter&);

  //! Disable default operator=
  PerfCounter& operator=(const PerfCounter&);

  const Device& gpuDevice_;  //!< The backend device
  const VirtualGPU& gpu_;    //!< The virtual GPU device object

  CalCounterReference* calRef_;  //!< Reference counter
  uint flags_;                   //!< The perfcounter object state
  Info info_;                    //!< The info structure for perfcounter
  gslQueryObject counter_;       //!< GSL counter object
  uint index_;                   //!< Counter index in the CAL container
};

}  // namespace gpu

#endif  // GPUCOUNTERS_HPP_
