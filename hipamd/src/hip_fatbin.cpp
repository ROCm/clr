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
#include "comgrctx.hpp"

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
        toDelete.insert(fbd->binary_image_);
      }
      delete fbd;
    }
  }
  for (auto itemData : toDelete) {
    LogPrintfInfo("~FatBinaryInfo(%p) will delete binary_image_ %p", this, itemData);
    delete[] reinterpret_cast<const char*>(itemData);
  }
  if (!HIP_USE_RUNTIME_UNBUNDLER) {
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

  } else {
    // Using Runtime Unbundler
    if (amd::Os::isValidFileDesc(fdesc_)) {
      if (fsize_ && !amd::Os::MemoryUnmapFile(image_, fsize_)) {
        LogPrintfError("Cannot unmap file for fdesc: %d fsize: %d", fdesc_, fsize_);
        assert(false);
      }
      if (!amd::Os::CloseFileHandle(fdesc_)) {
        LogPrintfError("Cannot close file for fdesc: %d", fdesc_);
        assert(false);
      }
    }

    fname_ = std::string();
    fdesc_ = amd::Os::FDescInit();
    fsize_ = 0;
    image_ = nullptr;
    uri_ = std::string();
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

hipError_t FatBinaryInfo::ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>& devices) {
  amd_comgr_data_t data_object {0};
  amd_comgr_status_t comgr_status = AMD_COMGR_STATUS_SUCCESS;
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
    bool isCompressed = false;
    // If the image ptr is not clang offload bundle then just directly point the image.
    if (!CodeObject::IsClangOffloadMagicBundle(image_, isCompressed)) {
      for (size_t dev_idx=0; dev_idx < devices.size(); ++dev_idx) {
        fatbin_dev_info_[devices[dev_idx]->deviceId()]
          = new FatBinaryDeviceInfo(image_, CodeObject::ElfSize(image_), 0);
        fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_
          = new amd::Program(*devices[dev_idx]->asContext());
        if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == nullptr) {
          hip_status = hipErrorOutOfMemory;
          break;
        }
      }
      break;
    }
    if (isCompressed || HIP_ALWAYS_USE_NEW_COMGR_UNBUNDLING_ACTION) {
      size_t major = 0, minor = 0;
      amd::Comgr::get_version(&major, &minor);
      if (major >= 2 && minor >= 8) {
        hip_status = ExtractFatBinaryUsingCOMGR(image_, devices);
        break;
      } else if (isCompressed) {
        LogPrintfError(
          "comgr %zu.%zu cannot support commpressed mode which need comgr 2.8+", major, minor);
        hip_status = hipErrorNotSupported;
        break;
      } else if (HIP_ALWAYS_USE_NEW_COMGR_UNBUNDLING_ACTION) {
        HIP_ALWAYS_USE_NEW_COMGR_UNBUNDLING_ACTION = false;
        LogInfo("HIP_ALWAYS_USE_NEW_COMGR_UNBUNDLING_ACTION = true only works on comgr 2.8+");
      }
    }
    // Create a data object, if it fails return error
    if ((comgr_status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &data_object))
                        != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Creating data object failed with status %d ", comgr_status);
      hip_status = hipErrorInvalidValue;
      break;
    }

#if !defined(_WIN32)
    // Using the file descriptor and file size, map the data object.
    if (amd::Os::isValidFileDesc(fdesc_)) {
      guarantee(fsize_ > 0, "Cannot have a file size of 0, fdesc: %d fname: %s",
                             fdesc_, fname_.c_str());
      if ((comgr_status = amd_comgr_set_data_from_file_slice(data_object, fdesc_, foffset_,
                          fsize_)) != AMD_COMGR_STATUS_SUCCESS) {
        LogPrintfError("Setting data from file slice failed with status %d ", comgr_status);
        hip_status = hipErrorInvalidValue;
        break;
      }
    } else
#endif
    if (image_ != nullptr) {
      // Using the image ptr, map the data object.
      if ((comgr_status = amd_comgr_set_data(data_object, 4096,
                          reinterpret_cast<const char*>(image_))) != AMD_COMGR_STATUS_SUCCESS) {
        LogPrintfError("Setting data from file slice failed with status %d ", comgr_status);
        hip_status = hipErrorInvalidValue;
        break;
      }
    } else {
      guarantee(false, "Cannot have both fname_ and image_ as nullptr");
    }

    // Find the unique number of ISAs needed for this COMGR query.
    std::unordered_map<std::string, std::pair<size_t, size_t>> unique_isa_names;
    for (auto device : devices) {
      std::string device_name = device->devices()[0]->isa().isaName();
      unique_isa_names.insert({device_name, std::make_pair<size_t, size_t>(0,0)});
    }

    // Create a query list using COMGR info for unique ISAs.
    std::vector<amd_comgr_code_object_info_t> query_list_array;
    query_list_array.reserve(unique_isa_names.size());
    for (const auto &isa_name : unique_isa_names) {
      auto &item = query_list_array.emplace_back();
      item.isa = isa_name.first.c_str();
      item.size = 0;
      item.offset = 0;
    }

    // Look up the code object info passing the query list.
    if ((comgr_status = amd_comgr_lookup_code_object(data_object, query_list_array.data(),
                        unique_isa_names.size())) != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Setting data from file slice failed with status %d ", comgr_status);
      hip_status = hipErrorInvalidValue;
      break;
    }

    for (const auto &item : query_list_array) {
      auto unique_it = unique_isa_names.find(item.isa);
      guarantee(unique_isa_names.cend() != unique_it, "Cannot find unique isa ");
      unique_it->second = std::pair<size_t, size_t>
                            (static_cast<size_t>(item.size),
                             static_cast<size_t>(item.offset));
    }

    for (auto device : devices) {
      std::string device_name = device->devices()[0]->isa().isaName();
      auto dev_it = unique_isa_names.find(device_name);
      // If the size is 0, then COMGR API could not find the CO for this GPU device/ISA
      if (dev_it->second.first == 0) {
        LogPrintfError("Cannot find CO in the bundle %s for ISA: %s",
                        fname_.c_str(), device_name.c_str());
        hip_status = hipErrorNoBinaryForGpu;
        ListAllDeviceWithNoCOFromBundle(unique_isa_names);
        break;
      }
      guarantee(unique_isa_names.cend() != dev_it,
                "Cannot find the device name in the unique device name");
      fatbin_dev_info_[device->deviceId()]
        = new FatBinaryDeviceInfo(reinterpret_cast<address>(const_cast<void*>(image_))
                                  + dev_it->second.second, dev_it->second.first,
                                                           dev_it->second.second);
      fatbin_dev_info_[device->deviceId()]->program_
        = new amd::Program(*(device->asContext()));
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

  if (data_object.handle) {
    if ((comgr_status = amd_comgr_release_data(data_object)) != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Releasing COMGR data failed with status %d ", comgr_status);
      return hipErrorInvalidValue;
    }
  }

  return hip_status;
}

hipError_t FatBinaryInfo::ExtractFatBinary(const std::vector<hip::Device*>& devices) {
  if (!HIP_USE_RUNTIME_UNBUNDLER) {
    return ExtractFatBinaryUsingCOMGR(devices);
  }

  hipError_t hip_error = hipSuccess;
  std::vector<std::pair<const void*, size_t>> code_objs;

  // Copy device names for Extract Code object File
  std::vector<std::string> device_names;
  device_names.reserve(devices.size());
  for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
    device_names.push_back(devices[dev_idx]->devices()[0]->isa().isaName());
  }

  // We are given file name, get the file desc and file size
  if (fname_.size() > 0) {
    // Get File Handle & size of the file.
    if (!amd::Os::GetFileHandle(fname_.c_str(), &fdesc_, &fsize_)) {
      return hipErrorFileNotFound;
    }
    if (fsize_ == 0) {
      return hipErrorInvalidImage;
    }

    // Extract the code object from file
    hip_error = CodeObject::ExtractCodeObjectFromFile(fdesc_, fsize_, &image_,
                device_names, code_objs);

  } else if (image_ != nullptr) {
    // We are directly given image pointer directly, try to extract file desc & file Size
    hip_error = CodeObject::ExtractCodeObjectFromMemory(image_,
                device_names, code_objs, uri_);
  } else {
    return hipErrorInvalidValue;
  }

  if (hip_error == hipErrorNoBinaryForGpu) {
    if (fname_.size() > 0) {
      LogPrintfError("hipErrorNoBinaryForGpu: Couldn't find binary for file: %s", fname_.c_str());
    } else {
      LogPrintfError("hipErrorNoBinaryForGpu: Couldn't find binary for ptr: 0x%x", image_);
    }

    // For the condition: unable to find code object for all devices,
    // still extract available images to those devices owning them.
    // This helps users to work with ROCm if there is any supported
    // GFX on system.
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      if (code_objs[dev_idx].first) {
        // Calculate the offset wrt binary_image and the original image
        size_t offset_l
          = (reinterpret_cast<address>(const_cast<void*>(code_objs[dev_idx].first))
              - reinterpret_cast<address>(const_cast<void*>(image_)));

        fatbin_dev_info_[devices[dev_idx]->deviceId()]
          = new FatBinaryDeviceInfo(code_objs[dev_idx].first, code_objs[dev_idx].second, offset_l);

        fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_
          = new amd::Program(*devices[dev_idx]->asContext());
        if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == NULL) {
          break;
        }
      }
    }

    return hip_error;
  }

  if (hip_error == hipErrorInvalidKernelFile) {
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      // the image type is no CLANG_OFFLOAD_BUNDLER, image for current device directly passed
      fatbin_dev_info_[devices[dev_idx]->deviceId()]
        = new FatBinaryDeviceInfo(image_, CodeObject::ElfSize(image_), 0);
    }
  } else if(hip_error == hipSuccess) {
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      // Calculate the offset wrt binary_image and the original image
      size_t offset_l
        = (reinterpret_cast<address>(const_cast<void*>(code_objs[dev_idx].first))
            - reinterpret_cast<address>(const_cast<void*>(image_)));

      fatbin_dev_info_[devices[dev_idx]->deviceId()]
        = new FatBinaryDeviceInfo(code_objs[dev_idx].first, code_objs[dev_idx].second, offset_l);
    }
  }

  for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
    fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_
       = new amd::Program(*devices[dev_idx]->asContext());
    if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == NULL) {
      return hipErrorOutOfMemory;
    }
  }

  return hipSuccess;
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

// ================================================================================================
hipError_t FatBinaryInfo::ExtractFatBinaryUsingCOMGR(const void *data,
    const std::vector<hip::Device*>& devices) {
  hipError_t hip_status = hipSuccess;
  // At this line, image should be a valid ptr.
  guarantee(data != nullptr, "Image cannot be nullptr");

  do {
    std::vector<std::pair<const void*, size_t>> code_objs;
    // Copy device names
    std::vector<std::string> device_names;
    device_names.reserve(devices.size());
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      device_names.push_back(devices[dev_idx]->devices()[0]->isa().isaName());
    }

    hip_status = CodeObject::extractCodeObjectFromFatBinaryUsingComgr(data, 0,
      device_names, code_objs);
    if (hip_status == hipErrorNoBinaryForGpu || hip_status == hipSuccess) {
      for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        if (code_objs[dev_idx].first) {
          fatbin_dev_info_[devices[dev_idx]->deviceId()] =
              new FatBinaryDeviceInfo(code_objs[dev_idx].first, code_objs[dev_idx].second, 0);

          fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ =
              new amd::Program(*devices[dev_idx]->asContext());
          if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == NULL) {
            break;
          }
        } else {
          // This is the case of hipErrorNoBinaryForGpu which will finally fail app on device
          // without code object
          LogPrintfError("Cannot find CO in the bundle %s for ISA: %s", fname_.c_str(),
                         device_names[dev_idx].c_str());
        }
      }
    } else if (hip_status == hipErrorInvalidKernelFile) {
      hip_status = hipSuccess;
      // If the image ptr is not clang offload bundle then just directly point the image.
      for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        fatbin_dev_info_[devices[dev_idx]->deviceId()] =
            new FatBinaryDeviceInfo(data, CodeObject::ElfSize(data), 0);
        fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ =
            new amd::Program(*devices[dev_idx]->asContext());
        if (fatbin_dev_info_[devices[dev_idx]->deviceId()]->program_ == nullptr) {
          hip_status = hipErrorOutOfMemory;
          break;
        }
      }
    } else {
      LogPrintfError(
        "CodeObject::extractCodeObjectFromFatBinaryUsingComgr failed with status %d\n",
                     hip_status);
    }
  } while (0);

  return hip_status;
}

} //namespace : hip
