//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//


#ifndef WITHOUT_FSA_BACKEND

#include "top.hpp"
#include "device/device.hpp"
#include "device/appprofile.hpp"
#include "device/hsa/hsaappprofile.hpp"

#include <algorithm>

amd::AppProfile* oclhsaCreateAppProfile()
{
    amd::AppProfile* appProfile = new oclhsa::AppProfile;

    if ((appProfile == NULL) || !appProfile->init()) {
        return NULL;
    }

    return appProfile;
}

namespace oclhsa {

bool AppProfile::ParseApplicationProfile()
{
    std::string appName("Explorer");

    std::transform(appName.begin(), appName.end(), appName.begin(), ::tolower);
    std::transform(appFileName_.begin(), appFileName_.end(), appFileName_.begin(), ::tolower);

    if (appFileName_.compare(appName) == 0 ) {
        hsaDeviceHint_ = CL_HSA_DISABLED_AMD;
        gpuvmHighAddr_ = false;
        noHsaInit_ = true;
        profileOverridesAllSettings_ = true;
    }

    // Setting both bits is invalid, make it niether.
    if (hsaDeviceHint_ & CL_HSA_ENABLED_AMD
        && hsaDeviceHint_ & CL_HSA_DISABLED_AMD) {
        hsaDeviceHint_ = 0;
    }

    if (noHsaInit_) {
        // If no HSA initialization, then force hint flag to non-HSA device.
        // Even if this is not forced, the device selection logic will endure it.
        // After all hint flags are treated as hint only - depending on
        // availibility.
        hsaDeviceHint_ = CL_HSA_DISABLED_AMD;
    }

    return true;
}

}

#endif
