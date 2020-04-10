 /* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#include "inifile.h"
#include "ini_export.h"
#include "ini_values.h"

#include "gsl_enum.h"

extern gslMemObjectAttribTiling g_CALBETiling_Tiled;

void
getConfigFromFile(gslStaticRuntimeConfig&  scfg,
                  gslDynamicRuntimeConfig& dcfg)
{
    const char* calIniFile = getenv("CAL_INI_FILE");
    IniFile iniFile(cmString(calIniFile ? calIniFile : INI_FILE));

    CALboolean dumpIL = CAL_FALSE;
    CALboolean dumpISA = CAL_FALSE;
    CALboolean macro                  = CAL_TRUE;
    CALboolean micro                  = CAL_TRUE;
    CALboolean breakonload            = CAL_FALSE;
    CALint     useRectPrim            = 0;
    CALboolean forceRemoteMemory      = CAL_FALSE;
    CALboolean disableAsyncDma        = CAL_FALSE;
    CALboolean disableVM              = CAL_FALSE;


    dcfg.bEmulator.hasValue               = ATIGL_TRUE;
    dcfg.DropFlush.hasValue               = ATIGL_TRUE;
    dcfg.EnableCommandbufferDump.hasValue = ATIGL_TRUE;
    dcfg.WaitForIdleAfterSubmit.hasValue  = ATIGL_TRUE;
    dcfg.FlushAfterRender.hasValue        = ATIGL_TRUE;
    dcfg.nPatchDumpLevel.hasValue         = ATIGL_TRUE;

    cmString commandbufferDumpFilename;

    iniFile.getValue(section, CAL_EMULATOR,               (CALboolean*) &dcfg.bEmulator.value);
    iniFile.getValue(section, CAL_ENABLE_FORCE_ASIC_ID,   (CALboolean*) &dcfg.forceAsicID.hasValue);
    iniFile.getValue(section, CAL_FORCE_ASIC_ID,          (CALint*)     &dcfg.forceAsicID.value);
    iniFile.getValue(section, CAL_DROPFLUSH,              (CALboolean*) &dcfg.DropFlush.value);
    iniFile.getValue(section, CAL_ENABLEPACKETDUMP,       (CALboolean*) &dcfg.EnableCommandbufferDump.value);

    // Check if location string is longer than 128 then assign, if not default location will be C:\packet.txt in gsl_ctx.cpp: gsCtxManager::PacketDump()
    uintp length = commandbufferDumpFilename.length();
    if (length > 0 && length < sizeof(dcfg.CommandbufferDumpFilename))
        strcpy(dcfg.CommandbufferDumpFilename, commandbufferDumpFilename.c_str());

    iniFile.getValue(section, CAL_ENABLEPATCHDUMP,        (CALint*)     &dcfg.nPatchDumpLevel.value);
    iniFile.getValue(section, CAL_ENABLEMACROTILE,        (CALboolean*) &macro);
    iniFile.getValue(section, CAL_ENABLEMICROTILE,        (CALboolean*) &micro);
    iniFile.getValue(section, CAL_BREAK_ON_LOAD,          (CALboolean*) &breakonload);
    iniFile.getValue(section, CAL_FORCE_REMOTE_MEMORY,    (CALboolean*) &forceRemoteMemory);
    iniFile.getValue(section, CAL_DISABLE_ASYNC_DMA,      (CALboolean*) &disableAsyncDma);
    iniFile.getValue(section, CAL_WAITFORIDLEAFTERSUBMIT, (CALboolean*) &dcfg.WaitForIdleAfterSubmit.value);
    iniFile.getValue(section, CAL_ENABLE_DUMP_IL, (CALboolean*) &dumpIL);
    iniFile.getValue(section, CAL_ENABLE_DUMP_ISA, (CALboolean*) &dumpISA);
    iniFile.getValue(section, CAL_ENABLE_FLUSH_AFTER_RENDER, (CALboolean*) &dcfg.FlushAfterRender.value);
    iniFile.getValue(section, CAL_DISABLE_VM,             (CALboolean*) &disableVM);

    if (disableVM)
    {
        scfg.VMMode = GSL_CONFIG_VM_MODE_FORCE_OFF;
    }

    if (!macro && !micro)
    {
        g_CALBETiling_Tiled = GSL_MOA_TILING_LINEAR;
    }

    if (breakonload)
    {
        #ifndef ATI_OS_LINUX
        __debugbreak();
        #endif
    }

    switch (forceRemoteMemory)
    {
    case 1:
        //
        // Also set linear, due to CAL expectations about different memory regions
        //
        g_CALBETiling_Tiled = GSL_MOA_TILING_LINEAR;
        break;

    default:
        break;
    }

    if (disableAsyncDma)
    {
        dcfg.drmdmaMode.hasValue = ATIGL_TRUE;
        dcfg.drmdmaMode.value = GSL_CONFIG_DRMDMA_MODE_FORCE_OFF;
    }
}

