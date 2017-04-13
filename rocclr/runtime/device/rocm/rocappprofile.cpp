//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//


#ifndef WITHOUT_HSA_BACKEND

#include "top.hpp"
#include "device/device.hpp"
#include "device/appprofile.hpp"
#include "device/rocm/rocappprofile.hpp"

#include <algorithm>

amd::AppProfile* rocCreateAppProfile() {
  amd::AppProfile* appProfile = new roc::AppProfile;

  if ((appProfile == nullptr) || !appProfile->init()) {
    return nullptr;
  }

  return appProfile;
}

namespace roc {

bool AppProfile::ParseApplicationProfile() {
  std::string appName("Explorer");

  std::transform(appName.begin(), appName.end(), appName.begin(), ::tolower);
  std::transform(appFileName_.begin(), appFileName_.end(), appFileName_.begin(), ::tolower);

  if (appFileName_.compare(appName) == 0) {
    gpuvmHighAddr_ = false;
    profileOverridesAllSettings_ = true;
  }

  return true;
}
}

#endif
