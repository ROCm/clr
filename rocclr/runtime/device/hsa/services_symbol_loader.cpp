//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

// Implementation of the the loading of dll and loading of all the exported
// function symbols.

#include "device/hsa/services_symbol_loader.hpp"

#include "runtime/thread/thread.hpp"
#include "runtime/utils/debug.hpp"
#include "runtime/os/os.hpp"

#include <stdlib.h>

#include <string>

ServicesApiSymbols* ServicesApiSymbols::instance_ = NULL;
// services_dll_handle_ is defined in ServicesApiSymbols class.
// This macro must be used only in member functions of ServicesApiSymbols
// class.
#define LOADSYMBOL(api)                                                        \
  api = (pfn_ ## api) amd::Os::getSymbol(services_dll_handle_, # api);         \
  if (api == NULL) {                                                           \
    amd::log_printf(amd::LOG_ERROR, __FILE__, __LINE__,                        \
                  "amd::Os::getSymbol() for exported func " # api " failed."); \
    amd::Os::unloadLibrary(services_dll_handle_);                              \
    abort();                                                                   \
  }

ServicesApiSymbols::ServicesApiSymbols()
    : services_dll_name_(SERVICES_DLL_NAME) {
  services_dll_handle_ = amd::Os::loadLibrary(services_dll_name_.c_str());
  if (services_dll_handle_ == NULL) {
// Do not print, otherwise tests fail when HSA core and services DLLs are 
// not installed, in which case only ORCA stack is initialized and it is 
// not an error
//   amd::log_printf(amd::LOG_INFO, __FILE__, __LINE__,
//"Cannot load hsa servicese dll. HSA DLLs may not be installed on the machine."
//" OpenCL requirement, returning without error.");
    return;
  }

  LOADSYMBOL(HsaGetServicesApiTable)
}

ServicesApiSymbols::~ServicesApiSymbols() {
    if (services_dll_handle_) {
    amd::Os::unloadLibrary(services_dll_handle_);
    services_dll_handle_ = NULL;
  }
}
