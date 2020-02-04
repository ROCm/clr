/* Copyright (c) 2014-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "top.hpp"
#include "os/os.hpp"
#include "utils/flags.hpp"
#include "appprofile.hpp"
#if !defined(WITH_LIGHTNING_COMPILER)
#include "adl.h"
#endif // !defined(WITH_LIGHTNING_COMPILER)
#include <cstdlib>
#include <cstring>

#if defined(WITH_LIGHTNING_COMPILER)
typedef void* ADLApplicationProfile;
int SearchProfileOfAnApplication(const wchar_t* fileName, ADLApplicationProfile** lppProfile)
{
  return 0;
}
#define __stdcall
#endif // defined(WITH_LIGHTNING_COMPILER)

#ifdef BRAHMA
extern int SearchProfileOfAnApplication(const wchar_t* fileName,
                                        ADLApplicationProfile** lppProfile);
#endif  // BRAHMA

static void* __stdcall adlMallocCallback(int n) { return malloc(n); }

#define GETPROCADDRESS(_adltype_, _adlfunc_) (_adltype_) amd::Os::getSymbol(adlHandle_, #_adlfunc_);

namespace amd {

#if !defined(BRAHMA) && !defined(WITH_LIGHTNING_COMPILER)

class ADL {
 public:
  ADL();
  ~ADL();

  bool init();

  void* adlHandle() const { return adlHandle_; };
  ADL_CONTEXT_HANDLE adlContext() const { return adlContext_; }

  typedef int (*Adl2MainControlCreate)(ADL_MAIN_MALLOC_CALLBACK callback,
                                       int iEnumConnectedAdapters, ADL_CONTEXT_HANDLE* context);
  typedef int (*Adl2MainControlDestroy)(ADL_CONTEXT_HANDLE context);
  typedef int (*Adl2ConsoleModeFileDescriptorSet)(ADL_CONTEXT_HANDLE context, int fileDescriptor);
  typedef int (*Adl2MainControlRefresh)(ADL_CONTEXT_HANDLE context);
  typedef int (*Adl2ApplicationProfilesSystemReload)(ADL_CONTEXT_HANDLE context);
  typedef int (*Adl2ApplicationProfilesProfileOfApplicationx2Search)(
      ADL_CONTEXT_HANDLE context, const wchar_t* fileName, const wchar_t* path,
      const wchar_t* version, const wchar_t* appProfileArea, ADLApplicationProfile** lppProfile);

  Adl2MainControlCreate adl2MainControlCreate;
  Adl2MainControlDestroy adl2MainControlDestroy;
  Adl2ConsoleModeFileDescriptorSet adl2ConsoleModeFileDescriptorSet;
  Adl2MainControlRefresh adl2MainControlRefresh;
  Adl2ApplicationProfilesSystemReload adl2ApplicationProfilesSystemReload;
  Adl2ApplicationProfilesProfileOfApplicationx2Search
      adl2ApplicationProfilesProfileOfApplicationx2Search;

 private:
  void* adlHandle_;
  ADL_CONTEXT_HANDLE adlContext_;
};

ADL::ADL() : adlHandle_(NULL), adlContext_(NULL) {
  adl2MainControlCreate = NULL;
  adl2MainControlDestroy = NULL;
  adl2ConsoleModeFileDescriptorSet = NULL;
  adl2MainControlRefresh = NULL;
  adl2ApplicationProfilesSystemReload = NULL;
  adl2ApplicationProfilesProfileOfApplicationx2Search = NULL;
}

ADL::~ADL() {
  if (adl2MainControlDestroy != NULL) {
    adl2MainControlDestroy(adlContext_);
  }
  adlContext_ = NULL;
}

bool ADL::init() {
  if (!adlHandle_) {
    adlHandle_ = amd::Os::loadLibrary("atiadl" LP64_SWITCH(LINUX_SWITCH("xx", "xy"), "xx"));
  }

  if (!adlHandle_) {
    return false;
  }

  adl2MainControlCreate = GETPROCADDRESS(Adl2MainControlCreate, ADL2_Main_Control_Create);
  adl2MainControlDestroy = GETPROCADDRESS(Adl2MainControlDestroy, ADL2_Main_Control_Destroy);
  adl2ConsoleModeFileDescriptorSet =
      GETPROCADDRESS(Adl2ConsoleModeFileDescriptorSet, ADL2_ConsoleMode_FileDescriptor_Set);
  adl2MainControlRefresh = GETPROCADDRESS(Adl2MainControlRefresh, ADL2_Main_Control_Refresh);
  adl2ApplicationProfilesSystemReload =
      GETPROCADDRESS(Adl2ApplicationProfilesSystemReload, ADL2_ApplicationProfiles_System_Reload);
  adl2ApplicationProfilesProfileOfApplicationx2Search =
      GETPROCADDRESS(Adl2ApplicationProfilesProfileOfApplicationx2Search,
                     ADL2_ApplicationProfiles_ProfileOfAnApplicationX2_Search);

  if (adl2MainControlCreate == NULL || adl2MainControlDestroy == NULL ||
      adl2MainControlRefresh == NULL || adl2ApplicationProfilesSystemReload == NULL ||
      adl2ApplicationProfilesProfileOfApplicationx2Search == NULL) {
    return false;
  }

  int result = adl2MainControlCreate(adlMallocCallback, 1, &adlContext_);
  if (result != ADL_OK) {
    // ADL2 is expected to return ADL_ERR_NO_XDISPLAY in Linux Console mode environment
    if (result == ADL_ERR_NO_XDISPLAY) {
      if (adl2ConsoleModeFileDescriptorSet == NULL ||
          adl2ConsoleModeFileDescriptorSet(adlContext_, ADL_UNSET) != ADL_OK) {
        return false;
      }
      adl2MainControlRefresh(adlContext_);
    } else {
      return false;
    }
  }

  // Reload is disabled in ADL with the change list 1198904 and ticket
  // SWDEV-59442 - The ADL_ApplicationProfiles_System_Reload Function is not Re-entrant
  // Returned value is ADL_ERR_NOT_SUPPORTED on Windows.
  adl2ApplicationProfilesSystemReload(adlContext_);

  return true;
}

#endif  // BRAHMA

AppProfile::AppProfile() : gpuvmHighAddr_(false), profileOverridesAllSettings_(false) {
  amd::Os::getAppPathAndFileName(appFileName_, appPathAndFileName_);
  propertyDataMap_.insert(
      DataMap::value_type("BuildOptsAppend", PropertyData(DataType_String, &buildOptsAppend_)));
}

AppProfile::~AppProfile() {}

bool AppProfile::init() {
  if (appFileName_.empty()) {
    return false;
  }

  // Convert appName to wide char for X2_Search ADL interface
  size_t strLength = appFileName_.length() + 1;
  size_t strPathLength = appPathAndFileName_.length() + 1;
  wchar_t* appName = new wchar_t[strPathLength];

  size_t success = mbstowcs(appName, appFileName_.c_str(), strLength);
  if (success > 0) {
    // mbstowcs was able to convert to wide character successfully.
    appName[strLength - 1] = L'\0';
  }

  wsAppFileName_ = appName;

  success = mbstowcs(appName, appPathAndFileName_.c_str(), strPathLength);
  if (success > 0) {
    // mbstowcs was able to convert to wide character successfully.
    appName[strPathLength - 1] = L'\0';
  }

  wsAppPathAndFileName_ = appName;

  delete[] appName;

  ParseApplicationProfile();

  return true;
}

bool AppProfile::ParseApplicationProfile() {
  ADLApplicationProfile* pProfile = NULL;

#if !defined(BRAHMA) && !defined(WITH_LIGHTNING_COMPILER)
  amd::ADL* adl = new amd::ADL;

  if ((adl == NULL) || !adl->init()) {
    delete adl;
    return false;
  }

  // Apply blb configurations
  int result = adl->adl2ApplicationProfilesProfileOfApplicationx2Search(
      adl->adlContext(), wsAppFileName_.c_str(), NULL, NULL, L"OCL", &pProfile);

  delete adl;

#else  // BRAHMA

  if (!SearchProfileOfAnApplication(wsAppFileName_.c_str(), &pProfile)) {
    return false;
  }

#endif  // BRAHMA

  if (pProfile == NULL) {
    return false;
  }

#if !defined(WITH_LIGHTNING_COMPILER)
  PropertyRecord* firstProperty = pProfile->record;
  uint32_t valueOffset = 0;
  const int BUFSIZE = 1024;
  wchar_t wbuffer[BUFSIZE];
  char buffer[2 * BUFSIZE];

  for (int index = 0; index < pProfile->iCount; index++) {
    PropertyRecord* profileProperty =
        reinterpret_cast<PropertyRecord*>((reinterpret_cast<char*>(firstProperty)) + valueOffset);

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
        *(reinterpret_cast<bool*>(entry->second.data_)) = profileProperty->uData[0] ? true : false;
        break;
      case DataType_String: {
        assert((size_t)(profileProperty->iDataSize) < sizeof(wbuffer) - 2 &&
               "app profile string too long");
        memset(wbuffer, 0, sizeof(wbuffer));
        memcpy(wbuffer, profileProperty->uData, profileProperty->iDataSize);
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
#endif // !defined(WITH_LIGHTNING_COMPILER)
  return true;
}
}
