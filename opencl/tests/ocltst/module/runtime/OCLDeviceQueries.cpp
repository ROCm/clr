/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

#include "OCLDeviceQueries.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "CL/cl_ext.h"

struct AMDDeviceInfo {
  const char* targetName_;        //!< Target name
  const char* machineTarget_;     //!< Machine target
  cl_uint simdPerCU_;             //!< Number of SIMDs per CU
  cl_uint simdWidth_;             //!< Number of workitems processed per SIMD
  cl_uint simdInstructionWidth_;  //!< Number of instructions processed per SIMD
  cl_uint memChannelBankWidth_;   //!< Memory channel bank width
  cl_uint localMemSizePerCU_;     //!< Local memory size per CU
  cl_uint localMemBanks_;         //!< Number of banks of local memory
  cl_uint gfxipMajor_;            //!< GFXIP major number
  cl_uint gfxipMinor_;            //!< GFXIP minor number
};

static const cl_uint Ki = 1024;
static const AMDDeviceInfo DeviceInfo[] = {
    /* CAL_TARGET_CAYMAN */
    {"Cayman", "cayman", 1, 16, 4, 256, 32 * Ki, 32, 5, 0},
    /* CAL_TARGET_TAHITI */
    {"Tahiti", "tahiti", 4, 16, 1, 256, 64 * Ki, 32, 6, 0},
    /* CAL_TARGET_PITCAIRN */
    {"Pitcairn", "pitcairn", 4, 16, 1, 256, 64 * Ki, 32, 6, 0},
    /* CAL_TARGET_CAPEVERDE */
    {"Capeverde", "capeverde", 4, 16, 1, 256, 64 * Ki, 32, 6, 0},
    /* CAL_TARGET_DEVASTATOR */
    {"Devastator", "trinity", 1, 16, 4, 256, 32 * Ki, 32, 5, 0},
    /* CAL_TARGET_SCRAPPER */
    {"Scrapper", "trinity", 1, 16, 4, 256, 32 * Ki, 32, 5, 0},
    /* CAL_TARGET_OLAND */ {"Oland", "oland", 4, 16, 1, 256, 64 * Ki, 32, 6, 0},
    /* CAL_TARGET_BONAIRE */
    {"Bonaire", "bonaire", 4, 16, 1, 256, 64 * Ki, 32, 7, 2},
    /* CAL_TARGET_SPECTRE */
    {"Spectre", "spectre", 4, 16, 1, 256, 64 * Ki, 32, 7, 1},
    /* CAL_TARGET_SPOOKY */
    {"Spooky", "spooky", 4, 16, 1, 256, 64 * Ki, 32, 7, 1},
    /* CAL_TARGET_KALINDI */
    {"Kalindi", "kalindi", 4, 16, 1, 256, 64 * Ki, 32, 7, 2},
    /* CAL_TARGET_HAINAN */
    {"Hainan", "hainan", 4, 16, 1, 256, 64 * Ki, 32, 6, 0},
    /* CAL_TARGET_HAWAII */
    {"Hawaii", "hawaii", 4, 16, 1, 256, 64 * Ki, 32, 7, 2},
    /* CAL_TARGET_ICELAND */
    {"Iceland", "iceland", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* CAL_TARGET_TONGA */ {"Tonga", "tonga", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* CAL_TARGET_MULLINS */
    {"Mullins", "mullins", 4, 16, 1, 256, 64 * Ki, 32, 7, 2},
    /* CAL_TARGET_FIJI */ {"Fiji", "fiji", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* CAL_TARGET_CARRIZO */
    {"Carrizo", "carrizo", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* CAL_TARGET_CARRIZO */
    {"Bristol Ridge", "carrizo", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* CAL_TARGET_Ellesmere */
    {"Ellesmere", "ellesmere", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* CAL_TARGET_BAFFIN */
    {"Baffin", "baffin", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* ROCM Kaveri */ {"gfx700", "gfx700", 4, 16, 1, 256, 64 * Ki, 32, 7, 1},
    /* ROCM Hawaii */ {"gfx701", "gfx701", 4, 16, 1, 256, 64 * Ki, 32, 7, 2},
    /* ROCM Kabini */ {"gfx703", "gfx703", 4, 16, 1, 256, 64 * Ki, 32, 7, 2},
    /* ROCM Iceland */ {"gfx800", "gfx800", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* ROCM Carrizo */ {"gfx801", "gfx801", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* ROCM Tonga */ {"gfx802", "gfx802", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* ROCM Fiji  */ {"gfx803", "gfx803", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* Vega10 */ {"gfx900", "gfx900", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* CAL_TARGET_STONEY */
    {"Stoney", "stoney", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* CAL_TARGET_LEXA */
    {"gfx804", "gfx804", 4, 16, 1, 256, 64 * Ki, 32, 8, 0},
    /* Vega10_XNACK */ {"gfx901", "gfx901", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* Raven */ {"gfx902", "gfx902", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* Raven_XNACK */ {"gfx903", "gfx903", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* Vega12      */ {"gfx904", "gfx904", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* Vega12_XNACK */ {"gfx905", "gfx905", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* Vega20 */ {"gfx906", "gfx906", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* Vega20_XNACK */ {"gfx907", "gfx907", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* MI100 */ {"gfx908", "gfx908", 4, 16, 1, 256, 64 * Ki, 32, 9, 0},
    /* Navi10 */ {"gfx1010", "gfx1010", 4, 32, 1, 256, 64 * Ki, 32, 10, 1},
    /* Navi12 */ {"gfx1011", "gfx1011", 4, 32, 1, 256, 64 * Ki, 32, 10, 1},
    /* Navi14 */ {"gfx1012", "gfx1012", 4, 32, 1, 256, 64 * Ki, 32, 10, 1},
    /* Navi21 */   { "gfx1030",     "gfx1030",     4, 32, 1, 256, 64 * Ki, 32, 10, 3 },
    /* Navi22 */   { "gfx1031",     "gfx1031",     4, 32, 1, 256, 64 * Ki, 32, 10, 3 },
    /* Navi23 */   { "gfx1032",     "gfx1032",     4, 32, 1, 256, 64 * Ki, 32, 10, 3 },
    /* Van Gogh */ { "gfx1033",     "gfx1033",     4, 32, 1, 256, 64 * Ki, 32, 10, 3 },
    /* Rembrandt */{ "gfx1035",     "gfx1035",     4, 32, 1, 256, 64 * Ki, 32, 10, 3 },
};

const int DeviceInfoSize = sizeof(DeviceInfo) / sizeof(AMDDeviceInfo);

OCLDeviceQueries::OCLDeviceQueries() {
  _numSubTests = 1;
  failed_ = false;
}

OCLDeviceQueries::~OCLDeviceQueries() {}

void OCLDeviceQueries::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  char name[1024] = {0};
  size_t size = 0;

  if (deviceId >= deviceCount_) {
    failed_ = true;
    return;
  }
  cl_uint value;
  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE,
                                     sizeof(deviceType), &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }

  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 1024,
                            name, &size);
  if (!strstr(name, "cl_amd_device_attribute_query")) {
    printf("AMD device attribute  extension is required for this test!\n");
    failed_ = true;
    return;
  }

  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_NAME,
                                     sizeof(name), name, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_NAME failed");

  std::string str = name;
  int id = 0;
  bool deviceFound = false;
  for (int i = 0; i < DeviceInfoSize; ++i) {
    if (0 == str.find(DeviceInfo[i].targetName_)) {
      deviceFound = true;
      id = i;
      break;
    }
  }
  CHECK_RESULT(deviceFound != true, "Device %s is not supported", name);

  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId],
                                     CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD,
                                     sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS),
               "CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD failed");
  CHECK_RESULT((value != DeviceInfo[id].simdPerCU_),
               "CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD failed");

  error_ =
      _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_SIMD_WIDTH_AMD,
                                sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_SIMD_WIDTH_AMD failed");
  CHECK_RESULT((value != DeviceInfo[id].simdWidth_),
               "CL_DEVICE_SIMD_WIDTH_AMD failed");

  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId],
                                     CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD,
                                     sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS),
               "CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD failed");
  CHECK_RESULT((value != DeviceInfo[id].simdInstructionWidth_),
               "CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD failed");

  error_ = _wrapper->clGetDeviceInfo(
      devices_[deviceId], CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD,
      sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS),
               "CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD failed");
  CHECK_RESULT((value != DeviceInfo[id].memChannelBankWidth_),
               "CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD failed");

  error_ = _wrapper->clGetDeviceInfo(
      devices_[deviceId], CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD,
      sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS),
               "CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD failed");
  CHECK_RESULT((value != DeviceInfo[id].localMemSizePerCU_),
               "CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD failed");

  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId],
                                     CL_DEVICE_LOCAL_MEM_BANKS_AMD,
                                     sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_LOCAL_MEM_BANKS_AMD failed");
  CHECK_RESULT((value != DeviceInfo[id].localMemBanks_),
               "CL_DEVICE_LOCAL_MEM_BANKS_AMD failed");

  error_ =
      _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_GFXIP_MAJOR_AMD,
                                sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_GFXIP_MAJOR_AMD failed");
  CHECK_RESULT((value != DeviceInfo[id].gfxipMajor_),
               "CL_DEVICE_GFXIP_MAJOR_AMD failed");

  error_ =
      _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_GFXIP_MINOR_AMD,
                                sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_GFXIP_MINOR_AMD failed");

  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId],
                                     CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD,
                                     sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS),
               "CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD failed");
  CHECK_RESULT((value == 0), "CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD failed");

  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId],
                                     CL_DEVICE_WAVEFRONT_WIDTH_AMD,
                                     sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_WAVEFRONT_WIDTH_AMD failed");
  CHECK_RESULT((value == 0), "CL_DEVICE_WAVEFRONT_WIDTH_AMD failed");

  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId],
                                     CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD,
                                     sizeof(cl_uint), &value, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS),
               "CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD failed");
  CHECK_RESULT((value == 0), "CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD failed");
}

static void CL_CALLBACK notify_callback(cl_event event,
                                        cl_int event_command_exec_status,
                                        void* user_data) {}

void OCLDeviceQueries::run(void) {
  if (failed_) {
    return;
  }
}

unsigned int OCLDeviceQueries::close(void) { return OCLTestImp::close(); }
