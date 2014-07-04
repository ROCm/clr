//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef _OPENCL_RUNTIME_DEVICE_HSA_HSACORE_SYMBOL_LOADER_HPP_
#define _OPENCL_RUNTIME_DEVICE_HSA_HSACORE_SYMBOL_LOADER_HPP_

// File: hsacore_symbol_loader.hpp
// The main purpose of this file (class HsacoreApiSymbols), is to load the HSA
// API function symbol HsaGetCoreApiTable() from hsacore DLL/so module.
// This function outputs HsaCoreApiTable which has pointers to the rest of the 
// hsacore API functions, which should be used to invoke the API functions.

#include "newcore.h"
#include "hsacoreagent.h"

#include <string>

// In case of change in the name of hsacore dll name, change the
// #define HSACORE_DLL_NAME  value. this is the only place the DLL name should
// be changed or referred to.
#define HSACORE_DLL_NAME "newhsacore" LP64_ONLY("64")

// Convention: The typedefed function name must be prefixed with pfn_ indicating
// it as pointer-to-function.
typedef HsaStatus (*pfn_HsaGetCoreApiTable)(const HsaCoreApiTable **api_table);


// Singleton HsacoreApiSymbols class contains the module handle and loaded
// symbols of one accessor API accessor function.
// To call hsacore API funciton, instance of this class must be used.
// Example:
//    // In initialization code
//    const HsaCoreApiTable *hsacoreapi = NULL;
//    HsacoreApiSymbols::Instance().HsaGetCoreApiTable(&hsacoreapi);
//    ...
//    ...
//    // Calling the core api.
//    hsacoreapi->HsaGetDevices(...);
//    hsacoreapi->HsaRegisterMemory(...);
class HsacoreApiSymbols {
 public:
  // Only the access function symbol is loaded, which in turn has pointers to
  // rest of the hsacore api.
  pfn_HsaGetCoreApiTable HsaGetCoreApiTable;

   static HsacoreApiSymbols &Instance() {
    if (instance_ == NULL) {
	instance_ = new HsacoreApiSymbols();
    }
    return *instance_;
   }
   static void teardown(){
       if (instance_ != NULL){
           delete instance_;
       }
   }
   static bool IsDllLoaded() {
     return Instance().hsacore_dll_handle_ ? true : false;
   };

 private:

  static HsacoreApiSymbols* instance_;
  // Force singleton pattern.export LD_LIBRAR
  explicit HsacoreApiSymbols();
  ~HsacoreApiSymbols();
  HsacoreApiSymbols(const HsacoreApiSymbols &) {}
  const HsacoreApiSymbols &operator=(const HsacoreApiSymbols &) {return *this; }

  // Data.
  void *hsacore_dll_handle_;
  const std::string hsacore_dll_name_;
};
#endif  // header guard
