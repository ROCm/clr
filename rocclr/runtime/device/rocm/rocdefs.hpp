#pragma once

#ifndef WITHOUT_HSA_BACKEND

namespace roc {

typedef uint HsaDeviceId;

struct AMDDeviceInfo {
    HsaDeviceId hsaDeviceId_;           //!< Machine id
    const char* targetName_;            //!< Target name for compilation
    const char* machineTarget_;         //!< Machine target
    uint        simdPerCU_;             //!< Number of SIMDs per CU
    uint        simdWidth_;             //!< Number of workitems processed per SIMD
    uint        simdInstructionWidth_;  //!< Number of instructions processed per SIMD
    uint        memChannelBankWidth_;   //!< Memory channel bank width
    uint        localMemSizePerCU_;     //!< Local memory size per CU
    uint        localMemBanks_;         //!< Number of banks of local memory
    uint        pciDeviceId;            //!< PCIe device id
};

//The device ID must match with the device's index into DeviceInfo
const HsaDeviceId HSA_SPECTRE_ID = 0;
const HsaDeviceId HSA_SPOOKY_ID = 1;
const HsaDeviceId HSA_TONGA_ID = 2;
const HsaDeviceId HSA_CARRIZO_ID = 3;
const HsaDeviceId HSA_ICELAND_ID = 4;
const HsaDeviceId HSA_FIJI_ID = 5;
const HsaDeviceId HSA_HAWAII_ID = 6;
const HsaDeviceId HSA_ELLESMERE_ID = 7;
const HsaDeviceId HSA_BAFFIN_ID = 8;
const HsaDeviceId HSA_INVALID_DEVICE_ID = -1;

static const AMDDeviceInfo DeviceInfo[] = {
                          //  targetName  machineTarget
    /* TARGET_KAVERI_SPECTRE */   {HSA_SPECTRE_ID, "Spectre", "Spectre", 4, 16, 1, 256, 64 * Ki, 32, 0 },
    /* TARGET_KAVERI_SPOOKY */    {HSA_SPOOKY_ID, "Spooky", "Spooky", 4, 16, 1, 256, 64 * Ki, 32, 0 },
    /* TARGET_TONGA */            {HSA_TONGA_ID, "Tonga", "Tonga", 4, 16, 1, 256, 64 * Ki, 32, 0},
    /* TARGET_CARRIZO */          {HSA_CARRIZO_ID, "Carrizo", "Carrizo", 4, 16, 1, 256, 64 * Ki, 32, 0},
    /* TARGET_ICELAND */          {HSA_ICELAND_ID, "Topaz", "Topaz", 4, 16, 1, 256, 64 * Ki, 32, 0},
    /* TARGET_FIJI */             {HSA_FIJI_ID, "Fiji", "Fiji", 4, 16, 1, 256, 64 * Ki, 32, 0 },
    /* TARGET HAWAII */           {HSA_HAWAII_ID, "Hawaii", "Hawaii", 4, 16, 1, 256, 64 * Ki, 32, 0 },
    /* TARGET ELLESMERE */        {HSA_ELLESMERE_ID, "Ellesmere", "Ellesmere", 4, 16, 1, 256, 64 * Ki, 32, 0 },
    /* TARGET BAFFIN */           {HSA_BAFFIN_ID, "Baffin", "Baffin", 4, 16, 1, 256, 64 * Ki, 32, 0 }
};

}
#endif

