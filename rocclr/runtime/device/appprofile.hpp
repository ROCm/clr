//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef APPPROFILE_HPP_
#define APPPROFILE_HPP_

#include "adl.h"

#include <string>

namespace amd {

class ADL {
public:
    ADL();
    ~ADL();

    bool init();

    void* adlHandle() const { return adlHandle_; };
    ADL_CONTEXT_HANDLE adlContext() const { return adlContext_; }

    typedef int (*Adl2MainControlCreate)(ADL_MAIN_MALLOC_CALLBACK callback,
                                            int iEnumConnectedAdapters,
                                            ADL_CONTEXT_HANDLE* context);
    typedef int (*Adl2MainControlDestroy)(ADL_CONTEXT_HANDLE context);
    typedef int (*Adl2ConsoleModeFileDescriptorSet)(ADL_CONTEXT_HANDLE context, int fileDescriptor);
    typedef int (*Adl2MainControlRefresh)(ADL_CONTEXT_HANDLE context);
    typedef int (*Adl2ApplicationProfilesSystemReload)(ADL_CONTEXT_HANDLE context);
    typedef int (*Adl2ApplicationProfilesProfileOfApplicationx2Search)(ADL_CONTEXT_HANDLE context,
                                                                              const wchar_t* fileName,
                                                                              const wchar_t* path,
                                                                              const wchar_t* version,
                                                                              const wchar_t* appProfileArea,
                                                                              ADLApplicationProfile** lppProfile);

    Adl2MainControlCreate adl2MainControlCreate;
    Adl2MainControlDestroy adl2MainControlDestroy;
    Adl2ConsoleModeFileDescriptorSet adl2ConsoleModeFileDescriptorSet;
    Adl2MainControlRefresh adl2MainControlRefresh;
    Adl2ApplicationProfilesSystemReload adl2ApplicationProfilesSystemReload;
    Adl2ApplicationProfilesProfileOfApplicationx2Search adl2ApplicationProfilesProfileOfApplicationx2Search;

private:
    void* adlHandle_;
    ADL_CONTEXT_HANDLE adlContext_;
};

class AppProfile {
public:
    AppProfile();
    virtual ~AppProfile();

    bool init();

    cl_device_type ApplyHsaDeviceHintFlag(const cl_device_type& type);
    bool IsHsaInitDisabled() { return noHsaInit_; }

protected:
    std::string appFileName_; // without extension
    std::wstring wsAppFileName_;

    virtual bool ParseApplicationProfile() { return true; }

    cl_device_type hsaDeviceHint_;  // valid values: CL_HSA_ENABLED_AMD
                                    // or CL_HSA_DISABLED_AMD
    bool gpuvmHighAddr_;  // Currently not used.
    bool noHsaInit_;      // Do not even initialize HSA.
    bool profileOverridesAllSettings_; // Overrides hint flags and env.var.
};

}
#endif

