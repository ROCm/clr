//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

// Implementation of the the loading of dll and loading of all the exported
// function symbols.


#include "runtime/device/hsa/hsacore_symbol_loader.hpp"

#include "runtime/thread/thread.hpp"
#include "runtime/utils/debug.hpp"
#include "runtime/os/os.hpp"

#include <stdlib.h>
#include <string>

HsacoreApiSymbols* HsacoreApiSymbols::instance_ = NULL;
// hsacore_dll_handle_ is defined in HsacoreApiSymbols class.
// This macro must be used only in member functions of HsacoreApiSymbols
// class.
#define LOADSYMBOL(api)                                                    \
  api = (pfn_ ## api) amd::Os::getSymbol(hsacore_dll_handle_, # api);      \
  if (api == NULL) {                                                       \
    amd::log_printf(amd::LOG_ERROR, __FILE__, __LINE__,                    \
              "amd::Os::getSymbol() for exported func " # api " failed."); \
    amd::Os::unloadLibrary(hsacore_dll_handle_);                           \
    abort();                                                               \
  }

HsacoreApiSymbols::HsacoreApiSymbols()
    : hsacore_dll_name_(HSACORE_DLL_NAME) {
  hsacore_dll_handle_ = amd::Os::loadLibrary(hsacore_dll_name_.c_str()); 
  if( hsacore_dll_handle_ == NULL) {
  // Do not print, otherwise tests fail when HSA core and services DLLs are 
  // not installed, in which case only ORCA stack is initialized and it is 
  // not an error..
  //amd::log_printf(amd::LOG_INFO, __FILE__, __LINE__,
  //   "Cannot load hsa core dll. HSA DLLs may not be installed on the machine."
  //   " OpenCL requirement, returning without error.");
    return;
  }

  LOADSYMBOL(HsaGetCoreApiTable)
}

HsacoreApiSymbols::~HsacoreApiSymbols() {
  if (hsacore_dll_handle_) {
    amd::Os::unloadLibrary(hsacore_dll_handle_);
    hsacore_dll_handle_ = NULL;
  }
}

