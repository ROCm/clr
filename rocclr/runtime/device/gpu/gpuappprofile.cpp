//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#include "top.hpp"
#include "utils/debug.hpp"
#include "device/appprofile.hpp"
#include "device/gpu/gpuappprofile.hpp"

namespace gpu {

AppProfile::AppProfile()
    : amd::AppProfile()
    , enableHighPerformanceState_(IS_LINUX ? false : true)
    , reportAsOCL12Device_(false)
{
    propertyDataMap_.insert(DataMap::value_type("HighPerfState",
        PropertyData(DataType_Boolean, &enableHighPerformanceState_)));

    propertyDataMap_.insert(DataMap::value_type("OCL12Device",
        PropertyData(DataType_Boolean, &reportAsOCL12Device_)));
}

bool AppProfile::ParseApplicationProfile()
{
    amd::ADL* adl = new amd::ADL;

    if ((adl == NULL) || !adl->init()) {
        delete adl;
        return false;
    }

    ADLApplicationProfile* pProfile = NULL;

    // Apply blb configurations
    int result = adl->adl2ApplicationProfilesProfileOfApplicationx2Search(
        adl->adlContext(), wsAppFileName_.c_str(), NULL, NULL,
        L"OCL", &pProfile);

    delete adl;

    if (pProfile == NULL) {
        return false;
    }

    PropertyRecord* firstProperty = pProfile->record;
    uint32_t valueOffset = 0;

    for (int index = 0; index < pProfile->iCount; index++) {
        PropertyRecord* profileProperty = reinterpret_cast<PropertyRecord*>
            ((reinterpret_cast<char*>(firstProperty)) + valueOffset);

        // Get property name
        char* propertyName = profileProperty->strName;
        auto entry = propertyDataMap_.find(std::string(propertyName));
        if (entry == propertyDataMap_.end()) {
            // unexpected name
            valueOffset += (sizeof(PropertyRecord) + profileProperty->iDataSize - 4);
            continue;
        }

        // Get the property value
        switch (entry->second.type_) {
        case DataType_Boolean:
            *(reinterpret_cast<bool*>(entry->second.data_)) =
                profileProperty->uData[0] ? true : false;
            break;
        default:
            break;
        }
        valueOffset += (sizeof(PropertyRecord) + profileProperty->iDataSize - 4);
    }

    free(pProfile);
    return true;
}

}

