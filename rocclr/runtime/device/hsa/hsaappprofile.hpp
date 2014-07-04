//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef HSAAPPPROFILE_HPP_
#define HSAAPPPROFILE_HPP_


#ifndef WITHOUT_FSA_BACKEND

namespace oclhsa {

class AppProfile : public amd::AppProfile
{
public:
    AppProfile(): amd::AppProfile() {}

protected:
    //! parse application profile based on application file name
    virtual bool ParseApplicationProfile();
};

}

#endif

#endif

