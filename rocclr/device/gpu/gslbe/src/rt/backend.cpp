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

#include "gsl_ctx.h"
#include "GSLContext.h"
#include "backend.h"
#include "GSLDevice.h"
#include "amuABI.h"

bool
getFuncInfoFromImage(CALimage image, CALfuncInfo *pFuncInfo)
{
    if (image == 0)
    {
        return false;
    }

    if (pFuncInfo == 0)
    {
        return false;
    }

    //Initialize the pFuncInfo
    pFuncInfo->maxScratchRegsNeeded = 0;
    pFuncInfo->numSharedGPRUser     = 0;
    pFuncInfo->numSharedGPRTotal    = 0;
    pFuncInfo->numThreadPerGroup    = 0;
    pFuncInfo->numThreadPerGroupX   = 0;
    pFuncInfo->numThreadPerGroupY   = 0;
    pFuncInfo->numThreadPerGroupZ   = 0;
    pFuncInfo->totalNumThreadGroup  = 0;
    pFuncInfo->numWavefrontPerSIMD  = 0;
    pFuncInfo->setBufferForNumGroup = false;
    pFuncInfo->wavefrontSize        = 0;
    pFuncInfo->numGPRsAvailable     = 0;
    pFuncInfo->numGPRsUsed          = 0;
    pFuncInfo->numSGPRsAvailable    = 0;
    pFuncInfo->numSGPRsUsed         = 0;
    pFuncInfo->numVGPRsAvailable    = 0;
    pFuncInfo->numVGPRsUsed         = 0;
    pFuncInfo->LDSSizeAvailable     = 0;
    pFuncInfo->LDSSizeUsed          = 0;
    pFuncInfo->stackSizeAvailable   = 0;
    pFuncInfo->stackSizeUsed        = 0;

    //read data from image file
    AMUabiMultiBinary mb;
    amuABIMultiBinaryCreate(&mb);
    if (!amuABIMultiBinaryUnpack(mb, (void*) image))
    {
        amuABIMultiBinaryDestroy(mb);
        return false;
    }

    unsigned int encodingCount;
    if (!amuABIMultiBinaryGetEncodingCount(&encodingCount, mb))
    {
        amuABIMultiBinaryDestroy(mb);
        return false;
    }

    AMUabiEncoding encoding;
    //get encoding info for the first encoding
    if ((encodingCount > 0)&& !amuABIMultiBinaryGetEncoding( &encoding, mb, 0))
    {
        amuABIMultiBinaryDestroy(mb);
        return false;
    }

    unsigned int machine, type;
    if (!amuABIEncodingGetSignature(&machine, &type, encoding))
    {
        amuABIMultiBinaryDestroy(mb);
        return false;
    }

    if (!amuABIMultiBinaryFindEncoding(&encoding, mb, machine, type))
    {
        amuABIMultiBinaryDestroy(mb);
        return false;
    }

    unsigned int progInfosCount = 0;
    CALProgramInfoEntry* pInfos = 0;
    if (!amuABIEncodingGetProgInfos(&progInfosCount, &pInfos, encoding))
    {
        amuABIMultiBinaryDestroy(mb);
        return false;
    }

    for (CALuint i =0; i < progInfosCount; i++)
    {
        switch(pInfos[i].address)
        {
        case AMU_ABI_CS_MAX_SCRATCH_REGS:
            pFuncInfo->maxScratchRegsNeeded = pInfos[i].value;
            break;
        case AMU_ABI_CS_NUM_SHARED_GPR_USER:
            pFuncInfo->numSharedGPRUser     = pInfos[i].value;
            break;
        case AMU_ABI_CS_NUM_SHARED_GPR_TOTAL:
            pFuncInfo->numSharedGPRTotal    = pInfos[i].value;
            break;
        case AMU_ABI_ECS_SETUP_MODE:
            break;
        case AMU_ABI_NUM_THREAD_PER_GROUP:
            pFuncInfo->numThreadPerGroup    = pInfos[i].value;
            break;
        case AMU_ABI_NUM_THREAD_PER_GROUP_X:
            pFuncInfo->numThreadPerGroupX   = pInfos[i].value;
            break;
        case AMU_ABI_NUM_THREAD_PER_GROUP_Y:
            pFuncInfo->numThreadPerGroupY   = pInfos[i].value;
            break;
        case AMU_ABI_NUM_THREAD_PER_GROUP_Z:
            pFuncInfo->numThreadPerGroupZ   = pInfos[i].value;
            break;
        case AMU_ABI_TOTAL_NUM_THREAD_GROUP:
            pFuncInfo->totalNumThreadGroup  = pInfos[i].value;
            break;
        case AMU_ABI_NUM_WAVEFRONT_PER_SIMD:
        case AMU_ABI_MAX_WAVEFRONT_PER_SIMD: //CAL_USE_SC_PRM
            pFuncInfo->numWavefrontPerSIMD  = pInfos[i].value;
            break;
        case AMU_ABI_IS_MAX_NUM_WAVE_PER_SIMD:
            break;
        case AMU_ABI_SET_BUFFER_FOR_NUM_GROUP:
            pFuncInfo->setBufferForNumGroup = (0 != pInfos[i].value) ? true : false;
            break;
        case AMU_ABI_WAVEFRONT_SIZE:
            pFuncInfo->wavefrontSize        = pInfos[i].value;
            break;
        case AMU_ABI_NUM_GPR_AVAIL:
            pFuncInfo->numGPRsAvailable     = pInfos[i].value;
            break;
        case AMU_ABI_NUM_GPR_USED:
            pFuncInfo->numGPRsUsed          = pInfos[i].value;
            break;
        case AMU_ABI_LDS_SIZE_AVAIL:
            pFuncInfo->LDSSizeAvailable     = pInfos[i].value;
            break;
        case AMU_ABI_LDS_SIZE_USED:
            pFuncInfo->LDSSizeUsed          = pInfos[i].value;
            break;
        case AMU_ABI_STACK_SIZE_AVAIL:
            pFuncInfo->stackSizeAvailable   = pInfos[i].value;
            break;
        case AMU_ABI_STACK_SIZE_USED:
            pFuncInfo->stackSizeUsed        = pInfos[i].value;
            break;
        case AMU_ABI_SI_NUM_SGPRS_AVAIL:
            pFuncInfo->numSGPRsAvailable    = pInfos[i].value;
            break;
        case AMU_ABI_SI_NUM_SGPRS:
            pFuncInfo->numSGPRsUsed         = pInfos[i].value;
            break;
        case AMU_ABI_SI_NUM_VGPRS_AVAIL:
            pFuncInfo->numVGPRsAvailable    = pInfos[i].value;
            break;
        case AMU_ABI_SI_NUM_VGPRS:
            pFuncInfo->numVGPRsUsed         = pInfos[i].value;
            break;

        default:
            //GSLAssert(0 && "Unknown address in program info");
            break;
        }
    }

    amuABIEncodingGetScratchRegisterCount(&pFuncInfo->maxScratchRegsNeeded, encoding);

    amuABIMultiBinaryDestroy(mb);

    return true;
}

gslMemObjectAttribTiling g_CALBETiling_Tiled  = GSL_MOA_TILING_TILED;
