//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#include "top.hpp"
#include "utils/debug.hpp"
#include "device/appprofile.hpp"
#include "device/gpu/gpuappprofile.hpp"

namespace gpu {

AppProfile::AppProfile():amd::AppProfile(),
                          enableHighPerformanceState_(true),
                          reportAsOCL12Device_(false)
{
    propertyDatatypeMap_.insert(DatatypeMap::value_type("HighPerfState",
                                                        DataType_Boolean));
    boolPropertyMap_.insert(BoolMap::value_type("HighPerfState",
                                                &enableHighPerformanceState_));

    propertyDatatypeMap_.insert(DatatypeMap::value_type("OCL12Device",
                                                        DataType_Boolean));
    boolPropertyMap_.insert(BoolMap::value_type("OCL12Device",
                                                &reportAsOCL12Device_));
}

bool AppProfile::ParseApplicationProfile()
{
    amd::ADL *adl = new amd::ADL;

    if (!adl->init()) {
        delete adl;
        return false;
    }

    int result = ADL_ERR_NOT_INIT;
    ADLApplicationProfile *pProfile = NULL;

    //
    // Apply blb configurations
    //
    result = adl->adl2ApplicationProfilesProfileOfApplicationx2Search(adl->adlContext(),
                                                                       wsAppFileName_.c_str(),
                                                                       NULL,
                                                                       NULL,
                                                                       L"OCL",
                                                                       &pProfile);

    delete adl;

    if (pProfile == NULL) {
        return false;
    }

    PropertyRecord *firstProperty = pProfile->record;
    PropertyRecord *profileProperty = NULL;
    uint32_t valueOffset = 0;
    for (int index = 0; index < pProfile->iCount; index++) {
        profileProperty = reinterpret_cast<PropertyRecord*>
                            ((reinterpret_cast<char*>(firstProperty)) + valueOffset);

        //
        // Get property name
        //
        char* propertyName = profileProperty->strName;
        DatatypeMap::const_iterator propertyDatatypeMapIt =
                                        propertyDatatypeMap_.find(std::string(propertyName));
        if (propertyDatatypeMapIt == propertyDatatypeMap_.end())
        {
            valueOffset += (sizeof(PropertyRecord) + profileProperty->iDataSize - 4);
            continue; // unexpected name.
        }

        DataTypes dataType = propertyDatatypeMapIt->second;
        switch(dataType) {
        case DataType_Boolean:
            {
                unsigned char propertyValue = profileProperty->uData[0];
                BoolMap::iterator boolPropertyMapIt =
                                    boolPropertyMap_.find(std::string(propertyName));
                if (boolPropertyMapIt != boolPropertyMap_.end()) {
                    *(boolPropertyMapIt->second) = propertyValue ? true : false;
                }
            }
            break;
        default:
            break;
        }
        valueOffset += (sizeof(PropertyRecord) + profileProperty->iDataSize - 4);
    }

    free(pProfile);
    pProfile = NULL;
    return true;
}

}

