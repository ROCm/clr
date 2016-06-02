//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef APPPROFILE_HPP_
#define APPPROFILE_HPP_

#include <map>
#include <string>

namespace amd {

class AppProfile {
public:
    AppProfile();
    virtual ~AppProfile();

    bool init();

    cl_device_type ApplyHsaDeviceHintFlag(const cl_device_type& type);
    bool IsHsaInitDisabled() { return noHsaInit_; }
    const std::string& GetBuildOptsAppend() const { return buildOptsAppend_; }
protected:
    enum DataTypes
    {
        DataType_Unknown = 0,
        DataType_Boolean,
        DataType_String,
    };

    struct PropertyData {
        PropertyData(DataTypes type, void* data): type_(type), data_(data) {}
        DataTypes   type_;  //!< Data type
        void*       data_;  //!< Pointer to the data
    };

    typedef std::map<std::string, PropertyData> DataMap;

    DataMap propertyDataMap_;
    std::string appFileName_; // without extension
    std::wstring wsAppFileName_;

    virtual bool ParseApplicationProfile();

    cl_device_type hsaDeviceHint_;  // valid values: CL_HSA_ENABLED_AMD
                                    // or CL_HSA_DISABLED_AMD
    bool gpuvmHighAddr_;  // Currently not used.
    bool noHsaInit_;      // Do not even initialize HSA.
    bool profileOverridesAllSettings_; // Overrides hint flags and env.var.
    std::string buildOptsAppend_;
};

}
#endif

