/*
Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip_fatbin.hpp"

#include <unordered_map>
#include "hip_code_object.hpp"
#include "hip_platform.hpp"

namespace hip {

FatBinaryDeviceInfo::~FatBinaryDeviceInfo() {
  if (program_ != nullptr) {
    program_->unload();
    program_->release();
    program_ = nullptr;
  }
}

FatBinaryInfo::FatBinaryInfo(const char* fname, const void* image) : fdesc_(amd::Os::FDescInit()),
                             fsize_(0), foffset_(0), image_(image), image_mapped_(false),
                             uri_(std::string()) {

  if (fname != nullptr) {
    fname_ = std::string(fname);
  } else {
    fname_ = std::string();
  }

  fatbin_dev_info_.resize(g_devices.size(), nullptr);
}

FatBinaryInfo::~FatBinaryInfo() {
  // Different devices in the same model have the same binary_image_
  std::set<const void*> toDelete;
  // Release per device fat bin info.
  for (auto* fbd: fatbin_dev_info_) {
    if (fbd != nullptr) {
      if (fbd->binary_image_ && fbd->binary_offset_ == 0 && fbd->binary_image_ != image_) {
        // binary_image_ was allocated in CodeObject::extractCodeObjectFromFatBinary
        toDelete.insert(fbd->binary_image_);
      }
      delete fbd;
    }
  }

  for (auto itemData : toDelete) {
    LogPrintfInfo("~FatBinaryInfo(%p) will delete binary_image_ %p", this, itemData);
    delete[] reinterpret_cast<const char*>(itemData);
  }

  // Using COMGR Unbundler
  if (ufd_ && amd::Os::isValidFileDesc(ufd_->fdesc_)) {
    // Check for ufd_ != nullptr, since sometimes, we never create unique_file_desc.
    if (ufd_->fsize_ && image_mapped_
          && !amd::Os::MemoryUnmapFile(image_, ufd_->fsize_)) {
      LogPrintfError("Cannot unmap file for fdesc: %d fsize: %d", ufd_->fdesc_, ufd_->fsize_);
      assert(false);
    }
    if (!PlatformState::instance().CloseUniqueFileHandle(ufd_)) {
      LogPrintfError("Cannot close file for fdesc: %d", ufd_->fdesc_);
      assert(false);
    }
  }

  fname_ = std::string();
  fdesc_ = amd::Os::FDescInit();
  fsize_ = 0;
  image_ = nullptr;
  uri_ = std::string();

  if (0 == PlatformState::instance().UfdMapSize()) {
    LogError("All Unique FDs are closed");
  }
}

void ListAllDeviceWithNoCOFromBundle(const std::unordered_map<std::string,
                                     std::pair<size_t, size_t>> unique_isa_names) {
  LogError("Missing CO for these ISAs - ");
  for (const auto& unique_isa : unique_isa_names) {
    if (unique_isa.second.first == 0) {
      LogPrintfError("     %s", unique_isa.first.c_str());
    }
  }
}

hipError_t FatBinaryInfo::ExtractFatBinary(const std::vector<hip::Device*>& devices) {
  hipError_t hip_status = hipSuccess;
  // If image was passed as a pointer to our hipMod* api, we can try to extract the file name
  // if it was mapped by the app. Otherwise use the COMGR data API.
  if (fname_.size() == 0) {
    if (image_ == nullptr) {
      LogError("Both Filename and image cannot be null");
      return hipErrorInvalidValue;
    }

    if(!amd::Os::FindFileNameFromAddress(image_, &fname_, &foffset_)) {
      fname_ = std::string("");
      foffset_ = 0;
    }
  }

  // If file name & path are available (or it is passed to you), then get the file desc to use
  // COMGR file slice APIs.
  if (fname_.size() > 0) {
    // Get File Handle & size of the file.
    ufd_ = PlatformState::instance().GetUniqueFileHandle(fname_.c_str());
    if (ufd_ == nullptr) {
      return hipErrorFileNotFound;
    }

    // If the file name exists but the file size is 0, the something wrong with the file or its path
    if (ufd_->fsize_ == 0) {
      return hipErrorInvalidValue;
    }

    // If image_ is nullptr, then file path is passed via hipMod* APIs, so map the file.
    if (image_ == nullptr) {
      if(!amd::Os::MemoryMapFileDesc(ufd_->fdesc_, ufd_->fsize_, foffset_, &image_)) {
        LogError("Cannot map the file descriptor");
        PlatformState::instance().CloseUniqueFileHandle(ufd_);
        return hipErrorInvalidValue;
      }
      image_mapped_ = true;
    }
  }

  // At this line, image should be a valid ptr.
  guarantee(image_ != nullptr, "Image cannot be nullptr, file:%s did not map for some reason",
                                fname_.c_str());

  do {
    std::vector<std::pair<const void*, size_t>> code_objs;
    // Copy device names
    std::vector<std::string> device_names;
    device_names.reserve(devices.size());
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      device_names.push_back(devices[dev_idx]->devices()[0]->isa().isaName());
    }
    hip_status = CodeObject::extractCodeObjectFromFatBinary(
      image_, 0, device_names, code_objs);
    if (hip_status == hipErrorNoBinaryForGpu || hip_status == hipSuccess) {
      for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        if (code_objs[dev_idx].first) {
            fatbin_dev_info_[devices[dev_idx]->deviceId()]
            = new FatBinaryDeviceInfo(code_objs[dev_idx].first, code_objs[dev_idx].second, 0);

            fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_
            = new amd::Program(*devices[dev_idx]->asContext());
          if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == NULL) {
            break;
          }
        }
        else {
          // This is the case of hipErrorNoBinaryForGpu which will finally fail app
          LogPrintfError("Cannot find CO in the bundle %s for ISA: %s", fname_.c_str(),
                         device_names[dev_idx].c_str());
        }
      }
    }
    else if (hip_status == hipErrorInvalidKernelFile) {
      hip_status = hipSuccess;
      // If the image ptr is not clang offload bundle then just directly point the image.
      for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        fatbin_dev_info_[devices[dev_idx]->deviceId()] =
            new FatBinaryDeviceInfo(image_, CodeObject::ElfSize(image_), 0);
        fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ =
            new amd::Program(*devices[dev_idx]->asContext());
        if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == nullptr) {
          hip_status = hipErrorOutOfMemory;
          break;
        }
      }
    }
    else {
      LogPrintfError(
        "CodeObject::extractCodeObjectFromFatBinary failed with status %d\n",
        hip_status);
    }
  } while(0);

  // Clean up file and memory resouces if hip_status failed for some reason.
  if (hip_status != hipSuccess && hip_status != hipErrorInvalidKernelFile) {
    if (image_mapped_) {
      if (!amd::Os::MemoryUnmapFile(image_, fsize_))
        guarantee(false, "Cannot unmap the file");

      image_ = nullptr;
      image_mapped_ = false;
    }

    if (amd::Os::isValidFileDesc(fdesc_)) {
      guarantee(fsize_ > 0, "Size has to greater than 0 too");
      if (!amd::Os::CloseFileHandle(fdesc_))
        guarantee(false, "Cannot close the file handle");

      fdesc_ = 0;
      fsize_ = 0;
    }
  }
  return hip_status;
}

hipError_t FatBinaryInfo::AddDevProgram(const int device_id) {
  // Device Id bounds Check
  DeviceIdCheck(device_id);

  FatBinaryDeviceInfo* fbd_info = fatbin_dev_info_[device_id];
  if (fbd_info == nullptr) {
    return hipErrorInvalidKernelFile;
  }

  // If fat binary was already added, skip this step and return success
  if (fbd_info->add_dev_prog_ == false) {
    amd::Context* ctx = g_devices[device_id]->asContext();
    if (CL_SUCCESS != fbd_info->program_->addDeviceProgram(*ctx->devices()[0],
                                          fbd_info->binary_image_,
                                          fbd_info->binary_size_, false,
                                          nullptr, nullptr, fdesc_,
                                          fbd_info->binary_offset_, uri_)) {
      return hipErrorInvalidKernelFile;
    }
    fbd_info->add_dev_prog_ = true;
  }
  return hipSuccess;
}

hipError_t FatBinaryInfo::BuildProgram(const int device_id) {

  // Device Id Check and Add DeviceProgram if not added so far
  DeviceIdCheck(device_id);
  IHIP_RETURN_ONFAIL(AddDevProgram(device_id));

  // If Program was already built skip this step and return success
  FatBinaryDeviceInfo* fbd_info = fatbin_dev_info_[device_id];
  if (fbd_info->prog_built_ == false) {
    if(CL_SUCCESS != fbd_info->program_->build(g_devices[device_id]->devices(),
                                               nullptr, nullptr, nullptr,
                                               kOptionChangeable, kNewDevProg)) {
      return hipErrorNoBinaryForGpu;
    }
    fbd_info->prog_built_ = true;
  }

  if (!fbd_info->program_->load()) {
    return hipErrorNoBinaryForGpu;
  }
  return hipSuccess;
}

} //namespace : hip
