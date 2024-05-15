/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_FAT_BINARY_HPP
#define HIP_FAT_BINARY_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"

// Forward declaration for Unique FD
struct UniqueFD;

namespace hip {

//Fat Binary Per Device info
class FatBinaryDeviceInfo {
public:
  FatBinaryDeviceInfo (const void* binary_image, size_t binary_size, size_t binary_offset)
                      : binary_image_(binary_image), binary_size_(binary_size),
                        binary_offset_(binary_offset), program_(nullptr),
                        add_dev_prog_(false), prog_built_(false) {}

  ~FatBinaryDeviceInfo();

private:
  const void* binary_image_; // binary image ptr
  size_t binary_size_;       // binary image size
  size_t binary_offset_;     // image offset from original

  amd::Program* program_;    // reinterpreted as hipModule_t
  friend class FatBinaryInfo;

  //Control Variables
  bool add_dev_prog_;
  bool prog_built_;
};


// Fat Binary Info
class FatBinaryInfo {
public:
  FatBinaryInfo(const char* fname, const void* image);
  ~FatBinaryInfo();

  // Loads Fat binary from file or image, unbundles COs for devices.
  hipError_t ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>& devices);

  /**
     *  @brief Extract code object from fatbin using comgr unbundling action via calling
     *         CodeObject::extractCodeObjectFromFatBinaryUsingComgr
     *
     *  @param[in]  data the bundle data(fatbin or loaded module data). It can be in uncompressed,
     *              compressed and even SPIR-V(to be supported later) mode.
     *  @param[in]  devices devices whose code objects will be extracted.
     *  Returned error code
     *
     *  @return #hipSuccess, #hipErrorNoBinaryForGpu, #hipErrorInvalidValue
     *
     *  @see CodeObject::extractCodeObjectFromFatBinaryUsingComgr()
     */
  hipError_t ExtractFatBinaryUsingCOMGR(const void* data,
                                              const std::vector<hip::Device*>& devices);
  hipError_t ExtractFatBinary(const std::vector<hip::Device*>& devices);
  hipError_t AddDevProgram(const int device_id);
  hipError_t BuildProgram(const int device_id);

  // Device Id bounds check
  inline void DeviceIdCheck(const int device_id) const {
    guarantee(device_id >= 0, "Invalid DeviceId less than 0");
    guarantee(static_cast<size_t>(device_id) < fatbin_dev_info_.size(), "Invalid DeviceId, greater than no of fatbin device info!");
  }

  // Getter Methods
  amd::Program* GetProgram(int device_id) {
    DeviceIdCheck(device_id);
    return fatbin_dev_info_[device_id]->program_;
  }

  hipModule_t Module(int device_id) const {
    DeviceIdCheck(device_id);
    return reinterpret_cast<hipModule_t>(as_cl(fatbin_dev_info_[device_id]->program_));
  }

  hipError_t GetModule(int device_id, hipModule_t* hmod) const {
    DeviceIdCheck(device_id);
    *hmod = reinterpret_cast<hipModule_t>(as_cl(fatbin_dev_info_[device_id]->program_));
    return hipSuccess;
  }

private:
  std::string fname_;        //!< File name
  amd::Os::FileDesc fdesc_;  //!< File descriptor
  size_t fsize_;             //!< Total file size
  size_t foffset_;           //!< File Offset where the fat binary is present.

  // Even when file is passed image will be mmapped till ~desctructor.
  const void* image_;        //!< Image
  bool image_mapped_;        //!< flag to detect if image is mapped

  // Only used for FBs where image is directly passed
  std::string uri_;          //!< Uniform resource indicator

  // Per Device Info, like corresponding binary ptr, size.
  std::vector<FatBinaryDeviceInfo*> fatbin_dev_info_;

  std::shared_ptr<UniqueFD> ufd_; //!< Unique file descriptor
};

}; // namespace hip

#endif // HIP_FAT_BINARY_HPP
