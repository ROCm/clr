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

  const std::string& GetBuildOptsAppend() const { return buildOptsAppend_; }

  const std::string& appFileName() const { return appFileName_; }

 protected:
  enum DataTypes {
    DataType_Unknown = 0,
    DataType_Boolean,
    DataType_String,
  };

  struct PropertyData {
    PropertyData(DataTypes type, void* data) : type_(type), data_(data) {}
    DataTypes type_;  //!< Data type
    void* data_;      //!< Pointer to the data
  };

  typedef std::map<std::string, PropertyData> DataMap;

  DataMap propertyDataMap_;
  std::string appFileName_;  // without extension
  std::wstring wsAppFileName_;

  virtual bool ParseApplicationProfile();

  bool gpuvmHighAddr_;                // Currently not used.
  bool profileOverridesAllSettings_;  // Overrides hint flags and env.var.
  std::string buildOptsAppend_;
};
}
#endif
