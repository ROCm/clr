#ifndef _OPENCL_RUNTIME_DEVICE_HSA_HSADEFS_HPP_
#define _OPENCL_RUNTIME_DEVICE_HSA_HSADEFS_HPP_

#ifndef WITHOUT_FSA_BACKEND

namespace oclhsa {

typedef uint HsaDeviceId;

struct AMDDeviceInfo {
    HsaDeviceId hsaDeviceId_;               //!< Machine id
    const char* targetName_;            //!< Target name for compilation
    const char* machineTarget_;         //!< Machine target
    uint        simdPerCU_;             //!< Number of SIMDs per CU
    uint        simdWidth_;             //!< Number of workitems processed per SIMD
    uint        simdInstructionWidth_;  //!< Number of instructions processed per SIMD
    uint        memChannelBankWidth_;   //!< Memory channel bank width
    uint        localMemSizePerCU_;     //!< Local memory size per CU
    uint        localMemBanks_;         //!< Number of banks of local memory
};

//The device ID must match with the device's index into DeviceInfo
const HsaDeviceId HSA_SPECTRE_ID = 0;
const HsaDeviceId HSA_SPOOKY_ID = 1;
const HsaDeviceId HSA_TONGA_ID = 2;
const HsaDeviceId HSA_CARRIZO_ID = 3;
const HsaDeviceId HSA_ICELAND_ID = 4;
const HsaDeviceId HSA_INVALID_DEVICE_ID = -1;

static const AMDDeviceInfo DeviceInfoTable[] = {
                          //  targetName  machineTarget
/* TARGET_KAVERI_SPECTRE */   {HSA_SPECTRE_ID, "Spectre", "Spectre", 4, 16, 1, 256, 64 * Ki, 32 },
/* TARGET_KAVERI_SPOOKY */    {HSA_SPOOKY_ID, "Spooky", "Spooky", 4, 16, 1, 256, 64 * Ki, 32 },
/* TARGET_TONGA */            {HSA_TONGA_ID, "Tonga", "Tonga", 4, 16, 1, 256, 64 * Ki, 32},
/* TARGET_CARRIZO */          {HSA_CARRIZO_ID, "Carrizo", "Carrizo", 4, 16, 1, 256, 64 * Ki, 32},
/* TARGET_ICELAND */          {HSA_ICELAND_ID, "Topaz", "Topaz", 4, 16, 1, 256, 64 * Ki, 32}
};


}
#endif
#endif