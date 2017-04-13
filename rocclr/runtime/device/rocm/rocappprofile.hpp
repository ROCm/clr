//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

class AppProfile : public amd::AppProfile {
 public:
  AppProfile() : amd::AppProfile() {}

 protected:
  //! parse application profile based on application file name
  virtual bool ParseApplicationProfile();
};
}

#endif
