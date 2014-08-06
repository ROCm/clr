//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef GPUAPPPROFILE_HPP_
#define GPUAPPPROFILE_HPP_

#include <string>
#include <map>

namespace gpu {

class AppProfile : public amd::AppProfile
{
public:
    AppProfile();

    //! return the value of enableHighPerformanceState_
    bool enableHighPerformanceState() const { return enableHighPerformanceState_; }
    bool reportAsOCL12Device() const { return reportAsOCL12Device_; }

protected:
    //! parse application profile based on application file name
    virtual bool ParseApplicationProfile();

private:
    enum DataTypes
    {
        DataType_Unknown = 0,
        DataType_Boolean,
    };

    struct PropertyData {
        PropertyData(DataTypes type, void* data): type_(type), data_(data) {}
        DataTypes   type_;  //!< Data type
        void*       data_;  //!< Pointer to the data
    };

    typedef std::map<std::string, PropertyData> DataMap;

    DataMap propertyDataMap_;

    bool enableHighPerformanceState_;
    bool reportAsOCL12Device_;
};

}

#endif
