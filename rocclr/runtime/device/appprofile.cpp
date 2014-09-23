//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#include "top.hpp"
#include "os/os.hpp"
#include "utils/flags.hpp"
#include "appprofile.hpp"
#include <cstdlib>

static void* __stdcall adlMallocCallback(int n)
{
    return malloc(n);
}

#define GETPROCADDRESS(_adltype_, _adlfunc_) (_adltype_)amd::Os::getSymbol(adlHandle_, #_adlfunc_);

namespace amd {

ADL::ADL() : adlHandle_(NULL),
               adlContext_(NULL)
{
    adl2MainControlCreate = NULL;
    adl2MainControlDestroy = NULL;
    adl2ConsoleModeFileDescriptorSet = NULL;
    adl2MainControlRefresh = NULL;
    adl2ApplicationProfilesSystemReload = NULL;
    adl2ApplicationProfilesProfileOfApplicationx2Search = NULL;
}

ADL::~ADL()
{
    if (adl2MainControlDestroy != NULL) {
        adl2MainControlDestroy(adlContext_);
    }
    adlContext_ = NULL;
}

bool ADL::init()
{
    if (!adlHandle_) {
        adlHandle_ = amd::Os::loadLibrary("atiadl" LP64_SWITCH(LINUX_SWITCH("xx", "xy"), "xx"));
    }

    if (!adlHandle_) {
        return false;
    }

    adl2MainControlCreate = GETPROCADDRESS(Adl2MainControlCreate, ADL2_Main_Control_Create);
    adl2MainControlDestroy = GETPROCADDRESS(Adl2MainControlDestroy, ADL2_Main_Control_Destroy);
    adl2ConsoleModeFileDescriptorSet = GETPROCADDRESS(Adl2ConsoleModeFileDescriptorSet, ADL2_ConsoleMode_FileDescriptor_Set);
    adl2MainControlRefresh = GETPROCADDRESS(Adl2MainControlRefresh, ADL2_Main_Control_Refresh);
    adl2ApplicationProfilesSystemReload = GETPROCADDRESS(Adl2ApplicationProfilesSystemReload,
                                                         ADL2_ApplicationProfiles_System_Reload);
    adl2ApplicationProfilesProfileOfApplicationx2Search = GETPROCADDRESS(Adl2ApplicationProfilesProfileOfApplicationx2Search,
                                                                         ADL2_ApplicationProfiles_ProfileOfAnApplicationX2_Search);

    if (adl2MainControlCreate == NULL
     || adl2MainControlDestroy == NULL
     || adl2MainControlRefresh == NULL
     || adl2ApplicationProfilesSystemReload == NULL
     || adl2ApplicationProfilesProfileOfApplicationx2Search == NULL) {
        return false;
    }

    int result = adl2MainControlCreate(adlMallocCallback, 1, &adlContext_);
    if (result != ADL_OK) {
        // ADL2 is expected to return ADL_ERR_NO_XDISPLAY in Linux Console mode environment
        if (result == ADL_ERR_NO_XDISPLAY) {
            if(adl2ConsoleModeFileDescriptorSet == NULL
            || adl2ConsoleModeFileDescriptorSet(adlContext_, ADL_UNSET) != ADL_OK) {
                return false;
            }
            adl2MainControlRefresh(adlContext_);
        }
        else {
            return false;
        }
    }

    result = adl2ApplicationProfilesSystemReload(adlContext_);
    if (result != ADL_OK) {
        return false;
    }

    return true;
}

AppProfile::AppProfile(): hsaDeviceHint_(0),
                          gpuvmHighAddr_(false),
                          noHsaInit_(false),
                          profileOverridesAllSettings_(false)
{
    appFileName_ = amd::Os::getAppFileName();
    propertyDataMap_.insert(DataMap::value_type("BuildOptsAppend",
        PropertyData(DataType_String, &buildOptsAppend_)));
}

AppProfile::~AppProfile()
{

}

bool AppProfile::init()
{
    if (appFileName_.empty()){
        return false;
    }

    // Convert appName to wide char for X2_Search ADL interface
    size_t strLength = appFileName_.length() + 1;
    wchar_t *appName = new wchar_t[strLength];

    size_t success = mbstowcs(appName, appFileName_.c_str(), strLength);
    if (success > 0) {
        // mbstowcs was able to convert to wide character successfully.
        appName[strLength - 1] = L'\0';
    }

    wsAppFileName_ = appName;

    delete appName;

    ParseApplicationProfile();

    return true;
}

cl_device_type AppProfile::ApplyHsaDeviceHintFlag(const cl_device_type& type)
{
    cl_device_type ret_type = type;

    bool isHsaHintSpecified = (type & (CL_HSA_ENABLED_AMD|CL_HSA_DISABLED_AMD))
                                != 0;
    // Apply app profile hsa device hint only if
    // HSA_RUNTIME is not set/defined *and*
    // no hsa hint flag already specified.
    // OR
    // Profile overridess all other settings (HSA_RUNTIME and hint flags).
    if ( profileOverridesAllSettings_
      || (flagIsDefault(HSA_RUNTIME) && !isHsaHintSpecified)) {
        // Clear current hsa hint.
        ret_type = type & ~(CL_HSA_ENABLED_AMD | CL_HSA_DISABLED_AMD);
        // Apply hsa hint from app profile.
        return (ret_type | hsaDeviceHint_);
    }

    // Do not apply app profile hsa device hint.
    return type;
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
    const int BUFSIZE = 1024;
    wchar_t wbuffer[BUFSIZE];
    char buffer[2 * BUFSIZE];

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
        case DataType_String: {
            assert((size_t)(profileProperty->iDataSize) < sizeof(wbuffer) - 2 &&
                "app profile string too long");
            memcpy(wbuffer, profileProperty->uData, profileProperty->iDataSize);
            wbuffer[profileProperty->iDataSize / 2] = L'\0';
            size_t len = wcstombs(buffer, wbuffer, sizeof(buffer));
            assert(len < sizeof(buffer) - 1 && "app profile string too long");
            *(reinterpret_cast<std::string*>(entry->second.data_)) = buffer;
            break;
        }
        default:
            break;
        }
        valueOffset += (sizeof(PropertyRecord) + profileProperty->iDataSize - 4);
    }

    free(pProfile);
    return true;
}

}
