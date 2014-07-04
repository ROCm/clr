//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef _OPENCL_RUNTIME_DEVICE_HSA_SERVICES_SYMBOL_LOADER_HPP_
#define _OPENCL_RUNTIME_DEVICE_HSA_SERVICES_SYMBOL_LOADER_HPP_

// File: services_symbol_loader.hpp
// The main purpose of this file (class ServicesApiSymbols), is to load the HSA
// API function symbol HsaGetServicesApiTable() from hsaservices DLL/so module.
// This function outputs HsaServicesApiTable which has pointers to the rest of the
// hsaservices API functions, which should be used to invoke the API functions.

#include "services.h"
#include "hsainterop.h"
#include "hsaagent.h"

#include <string>

// In case of change in the name of hsaservices dll name, change the
// #define SERVICES_DLL_NAME  value. this is the only place the DLL name should
// be changed or referred to.
#define SERVICES_DLL_NAME "hsaservices" LP64_ONLY("64")

// Convention: The typedefed function name must be prefixed with pfn_ indicating
// it as pointer-to-function.
typedef HsaStatus (*pfn_HsaGetServicesApiTable)(const HsaServicesApiTable **api_table);

// Singleton ServicesApiSymbols class contains the module handle and loaded
// symbols of one accessor API accessor function.
// To call hsaservices API funciton, instance of this class must be used.
// Example:
//    // In initialization code
//    const HsaServicesApiTable *servicesapi = NULL;
//    ServicesApiSymbols::Instance().HsaGetServicesApiTable(&servicesapi);
//    ...
//    ...
//    // Calling the services api.
//    servicesapi->HsaGetDevices(...);
//    servicesapi->HsaRegisterMemory(...);
class ServicesApiSymbols {
 public:
  // Only the access function symbol is loaded, which in turn has pointers to
  // rest of the hsaservices api.
  pfn_HsaGetServicesApiTable HsaGetServicesApiTable;
  static ServicesApiSymbols& Instance() {
    if (instance_ == NULL) {
        instance_ = new ServicesApiSymbols();
    }

    return *instance_;
  }
  static void teardown(){
       if (instance_ != NULL){
           delete instance_;
       }
  }
  static bool IsDllLoaded(){
     return Instance().services_dll_handle_ ? true : false;
  };


 private:

  static ServicesApiSymbols* instance_;
  // Force singleton pattern.
  explicit ServicesApiSymbols();
  ~ServicesApiSymbols();
  ServicesApiSymbols(const ServicesApiSymbols &) {}
  const ServicesApiSymbols &operator=(const ServicesApiSymbols &) {
    return *this;
  }

  // Data.
  void *services_dll_handle_;
  const std::string services_dll_name_;
};
#endif  // _OPENCL_RUNTIME_DEVICE_HSA_SERVICES_SYMBOL_LOADER_HPP_
