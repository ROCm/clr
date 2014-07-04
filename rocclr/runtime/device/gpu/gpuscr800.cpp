//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/gpu/gpukernel.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>

#include "acl.h"
#define R900_BUILD 1
#include "SCShadersR800.h"
#include "r8xx_r9xx_merged__offset.h"
#include "r8xx_r9xx_merged__typedef.h"

namespace gpu {

#define NUM_R800_CS_INFOS (0x22+SC_R800_MAX_UAV+                                                            \
                           3+       /* globalReturnBuffer flag plus numUavs and numGlobalReturnBuffers */   \
                           1+       /* extendedCaching flag                                            */   \
                           3+       /* globalReturnBuffer sizes for dword, shorts and bytes            */   \
                           3*SC_R800_MAX_UAV+   /* offsetmap, cached and uncached fetch consts         */   \
                           2*SC_R800_MAX_UAV+   /* 64- and 128-bit cached fetch consts                 */   \
                           2*R800_GLOBAL_RTN_BUF_LAST   /* global return buffer fetch consts and type  */ )

struct Options {
    uint numClauseTemps_;
    uint numGPRs_;
    uint numThreads_;
    uint numStackEntries_;
    uint ldsSize_;

    Options(CALtarget target) {
        numClauseTemps_ = 4;

        switch (target) {
        case CAL_TARGET_DEVASTATOR:
        case CAL_TARGET_SCRAPPER:
        case CAL_TARGET_CAYMAN:
        case CAL_TARGET_KAUAI:
            numClauseTemps_     = 0;
            numStackEntries_    = 512;
            numThreads_         = 248;
            break;
        case CAL_TARGET_SUPERSUMO:
        case CAL_TARGET_TURKS:
        case CAL_TARGET_REDWOOD:
            numStackEntries_    = 256;
            numThreads_         = 248;
            break;
        case CAL_TARGET_WRESTLER:
        case CAL_TARGET_SUMO:
        case CAL_TARGET_CAICOS:
        case CAL_TARGET_CEDAR:
            numStackEntries_    = 256;
            numThreads_         = 192;
            break;
        case CAL_TARGET_CYPRESS:
        case CAL_TARGET_BARTS:
        case CAL_TARGET_JUNIPER:
            numStackEntries_    = 512;
            numThreads_         = 248;
            break;
        default:
            numStackEntries_    = 512;
            numThreads_         = 248;
            LogError("Unknown ASIC type");
        }

        numGPRs_ = 256 - 2 * numClauseTemps_;
        ldsSize_ = 32*1024;
    }
private:
    Options();
    Options(const Options&);
    Options& operator=(const Options&);
};

static const uint UncachedFetchConst[SC_R800_MAX_UAV] =
    { 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 173 };

static const uint CachedFetchConst[SC_R800_MAX_UAV] =
    { 144, 145, 146, 148, 149, 150, 151, 152, 0, 0, 0, 153 };

static const uint GlobalReturnFetchConst[R800_GLOBAL_RTN_BUF_LAST] =
    { 165, 166, 167, 168, 169, 170, 171, 172 };

static const uint GlobalReturnBufferType[R800_GLOBAL_RTN_BUF_LAST] =
    { AMU_ABI_UAV_FORMAT_TYPELESS, AMU_ABI_UAV_FORMAT_FLOAT,
      AMU_ABI_UAV_FORMAT_UNORM, AMU_ABI_UAV_FORMAT_SNORM, AMU_ABI_UAV_FORMAT_UINT,
      AMU_ABI_UAV_FORMAT_SINT, AMU_ABI_UAV_FORMAT_SHORT, AMU_ABI_UAV_FORMAT_BYTE };

static const uint CachedFetchConst64[SC_R800_MAX_UAV] =
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174 };

static const uint CachedFetchConst128[SC_R800_MAX_UAV] =
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175 };

bool
NullKernel::r800CreateHwInfo(const void* shader, AMUabiAddEncoding& encoding)
{
    CALProgramInfoEntry* newInfos;
    const Options options(nullDev().calTarget());
    uint i = 0;
    uint numShaderEngines = 1;
    if ((nullDev().calTarget() == CAL_TARGET_CAYMAN) ||
        (nullDev().calTarget() == CAL_TARGET_CYPRESS) ||
        (nullDev().calTarget() == CAL_TARGET_BARTS)) {
        numShaderEngines = 2;
    }

    uint infoCount = NUM_R800_CS_INFOS;
    SC_R800CSHWSHADER* cShader = (SC_R800CSHWSHADER *)shader;
    if (cShader->u32NumThreadPerGroup == 0) {
        return false;
    }
    newInfos = new CALProgramInfoEntry[infoCount];
    encoding.progInfos = newInfos;
    if (encoding.progInfos == 0) {
        infoCount = 0;
        return false;
    }
    memset(newInfos, 0, infoCount * sizeof(CALProgramInfoEntry));

    newInfos[i].address = mmSQ_PGM_START_LS;
    newInfos[i].value = 0x0;
    i++;
    newInfos[i].address = mmSQ_PGM_RESOURCES_LS;
    cShader->sqPgmResourcesCs.bits.UNCACHED_FIRST_INST  = 1;
    cShader->sqPgmResourcesCs.bits.PRIME_CACHE_ENABLE   = 1;
    cShader->sqPgmResourcesCs.bits.PRIME_CACHE_ON_CONST = 0;
    newInfos[i].value = cShader->sqPgmResourcesCs.u32All;
    i++;
    newInfos[i].address = mmSQ_PGM_RESOURCES_2_LS;
    newInfos[i].value = cShader->sqPgmResources2Cs.u32All;
    i++;

    newInfos[i].address = mmSPI_THREAD_GROUPING;
    regSPI_THREAD_GROUPING spi_thread_grouping;
    spi_thread_grouping.u32All = 0;
    spi_thread_grouping.bits.PS_GROUPING = 0;
    spi_thread_grouping.bits.VS_GROUPING = 0;
    spi_thread_grouping.bits.ES_GROUPING = 0;
    spi_thread_grouping.bits.GS_GROUPING = 0;
    // dyn_gpr_mgmt if CS_GROUPING = 1.
    spi_thread_grouping.bits.CS_GROUPING = 0;
    newInfos[i].value = spi_thread_grouping.u32All;
    i++;

    const unsigned int numSharedGPR   = cShader->u32NumSharedGprTotal;
    newInfos[i].address = mmSQ_DYN_GPR_CNTL_PS_FLUSH_REQ;
    regSQ_DYN_GPR_CNTL_PS_FLUSH_REQ sq_dyn_gpr_cntl_ps_flush_req;
    sq_dyn_gpr_cntl_ps_flush_req.u32All                  = 0;
    sq_dyn_gpr_cntl_ps_flush_req.bits.RING0_OFFSET       = numSharedGPR;
    newInfos[i].value = sq_dyn_gpr_cntl_ps_flush_req.u32All;
    i++;

    const unsigned int numClauseTemps = options.numClauseTemps_;
    const unsigned int MaxNumGPRsAvail = options.numGPRs_;
    newInfos[i].address = mmSQ_GPR_RESOURCE_MGMT_1;
    regSQ_GPR_RESOURCE_MGMT_1 sq_gpr_resource_mgmt_1;
    sq_gpr_resource_mgmt_1.u32All                    = 0;
    sq_gpr_resource_mgmt_1.bits.NUM_CLAUSE_TEMP_GPRS = numClauseTemps;
    newInfos[i].value = sq_gpr_resource_mgmt_1.u32All;
    i++;

    newInfos[i].address = mmSQ_GPR_RESOURCE_MGMT_3__EG;
    regSQ_GPR_RESOURCE_MGMT_3__EG sq_gpr_resource_mgmt_3;
    sq_gpr_resource_mgmt_3.u32All           = 0;
    {
        const unsigned int numWavefrontPerSIMD = 1 ; // ??  cShader->u32NumWavefrontPerSIMD;
        if ((cShader->u32NumSharedGprUser != cShader->u32NumSharedGprTotal)) // cShader->bIsMaxNumWavePerSIMD)
        {
            // if running with a barrier, need to limit the number of wavefronts on a SIMD.
            // force max wavefronts run on a simd by adjusting the num_es_gprs pool that all es programs can
            // allocate from. (# of gprs the program uses * numWavefrontsPerSIMD)
            sq_gpr_resource_mgmt_3.bits.NUM_LS_GPRS = cShader->sqPgmResourcesCs.bits.NUM_GPRS * numWavefrontPerSIMD;
        }
        else
        {
            sq_gpr_resource_mgmt_3.bits.NUM_LS_GPRS = MaxNumGPRsAvail - numSharedGPR;
        }
    }
    newInfos[i].value = sq_gpr_resource_mgmt_3.u32All;
    i++;

    newInfos[i].address = mmSPI_GPR_MGMT;
    regSPI_GPR_MGMT        spi_gpr_mgmt;
    spi_gpr_mgmt.u32All            = 0;
    {
        const unsigned int numWavefrontPerSIMD = 1 ; // ??  cShader->u32NumWavefrontPerSIMD;
        if ((cShader->u32NumSharedGprUser != cShader->u32NumSharedGprTotal)) // cShader->bIsMaxNumWavePerSIMD)
        {
            // if running with a barrier, need to limit the number of wavefronts on a SIMD.
            // force max wavefronts run on a simd by adjusting the num_es_gprs pool that all es programs can
            // allocate from. (# of gprs the program uses * numWavefrontsPerSIMD)
            spi_gpr_mgmt.bits.NUM_LS_GPRS = (cShader->sqPgmResourcesCs.bits.NUM_GPRS * numWavefrontPerSIMD) >> 3;
        }
        else
        {
            spi_gpr_mgmt.bits.NUM_LS_GPRS = (MaxNumGPRsAvail - numSharedGPR) >> 3;
        }
    }
    newInfos[i].value = spi_gpr_mgmt.u32All;
    i++;


    newInfos[i].address = mmSPI_WAVE_MGMT_1;
    regSPI_WAVE_MGMT_1     spi_wave_mgmt_1;
    spi_wave_mgmt_1.u32All = 0;
    newInfos[i].value = spi_wave_mgmt_1.u32All;
    i++;

    newInfos[i].address = mmSPI_WAVE_MGMT_2;
    regSPI_WAVE_MGMT_2     spi_wave_mgmt_2;
    spi_wave_mgmt_2.u32All = 0;
    spi_wave_mgmt_2.bits.NUM_CS_WAVES_ONE_RING = (options.numThreads_) >> 3;
    newInfos[i].value = spi_wave_mgmt_2.u32All;
    i++;

    newInfos[i].address = mmSQ_THREAD_RESOURCE_MGMT__EG;
    regSQ_THREAD_RESOURCE_MGMT__EG sq_thread_resource_mgmt;
    sq_thread_resource_mgmt.u32All              = 0;
    sq_thread_resource_mgmt.bits.NUM_PS_THREADS = 0;
    sq_thread_resource_mgmt.bits.NUM_VS_THREADS = 0;
    sq_thread_resource_mgmt.bits.NUM_GS_THREADS = 0;
    sq_thread_resource_mgmt.bits.NUM_ES_THREADS = 0;
    newInfos[i].value = sq_thread_resource_mgmt.u32All;
    i++;

    newInfos[i].address = mmSQ_THREAD_RESOURCE_MGMT_2__EG;
    regSQ_THREAD_RESOURCE_MGMT_2__EG sq_thread_resource_mgmt_2;
    sq_thread_resource_mgmt_2.u32All              = 0;
    sq_thread_resource_mgmt_2.bits.NUM_HS_THREADS = 0;
    sq_thread_resource_mgmt_2.bits.NUM_LS_THREADS = options.numThreads_;
    newInfos[i].value = sq_thread_resource_mgmt_2.u32All;
    i++;

    regSPI_COMPUTE_INPUT_CNTL spi_dompute_input_cntl;
    spi_dompute_input_cntl.u32All = 0;
    spi_dompute_input_cntl.bits.DISABLE_INDEX_PACK = 1;
    spi_dompute_input_cntl.bits.TID_IN_GROUP_ENA = 1;
    spi_dompute_input_cntl.bits.TGID_ENA = 1;
    newInfos[i].address = mmSPI_COMPUTE_INPUT_CNTL;
    newInfos[i].value = spi_dompute_input_cntl.u32All;
    i++;

    newInfos[i].address = mmSQ_LDS_ALLOC;
    newInfos[i].value = cShader->sqLdsAllocCs.u32All;
    i++;

    //This is information passed from SC to GSL, there is no valid address, so make up one.
    newInfos[i].address = AMU_ABI_CS_MAX_SCRATCH_REGS;
    newInfos[i].value = cShader->MaxScratchRegsNeeded;
    i++;
    newInfos[i].address = AMU_ABI_CS_NUM_SHARED_GPR_USER;
    newInfos[i].value = cShader->u32NumSharedGprUser;
    i++;
    newInfos[i].address = AMU_ABI_CS_NUM_SHARED_GPR_TOTAL;
    newInfos[i].value = cShader->u32NumSharedGprTotal;
    i++;
    newInfos[i].address = AMU_ABI_NUM_THREAD_PER_GROUP;
    newInfos[i].value = cShader->u32NumThreadPerGroup;
    i++;
    newInfos[i].address = AMU_ABI_NUM_THREAD_PER_GROUP_X;
    newInfos[i].value = cShader->u32NumThreadPerGroup_x;
    i++;
    newInfos[i].address = AMU_ABI_NUM_THREAD_PER_GROUP_Y;
    newInfos[i].value = cShader->u32NumThreadPerGroup_y;
    i++;
    newInfos[i].address = AMU_ABI_NUM_THREAD_PER_GROUP_Z;
    newInfos[i].value = cShader->u32NumThreadPerGroup_z;
    i++;
    newInfos[i].address = AMU_ABI_TOTAL_NUM_THREAD_GROUP;
    newInfos[i].value = cShader->u32TotalNumThreadGroup;
    i++;
    newInfos[i].address = AMU_ABI_NUM_WAVEFRONT_PER_SIMD;
    newInfos[i].value = 1;
    i++;
    newInfos[i].address = AMU_ABI_IS_MAX_NUM_WAVE_PER_SIMD;
    newInfos[i].value = 0;         // ??
    i++;
    newInfos[i].address = AMU_ABI_SET_BUFFER_FOR_NUM_GROUP;
    newInfos[i].value = cShader->bSetBufferForNumGroup;
    i++;
    newInfos[i].address = AMU_ABI_RAT_OP_IS_USED;
    newInfos[i].value = cShader->u32RatOpIsUsed;
    i++;
    newInfos[i].address = AMU_ABI_RAT_ATOMIC_OP_IS_USED;
    newInfos[i].value = cShader->u32RatAtomicOpIsUsed;
    i++;
    newInfos[i].address = AMU_ABI_WAVEFRONT_SIZE;
    newInfos[i].value = nullDev().hwInfo()->simdWidth_ * 4;
    i++;
    newInfos[i].address = AMU_ABI_NUM_GPR_AVAIL;
    newInfos[i].value = options.numGPRs_;
    i++;
    newInfos[i].address = AMU_ABI_NUM_GPR_USED;
    newInfos[i].value = cShader->sqPgmResourcesCs.bits.NUM_GPRS;
    i++;
    newInfos[i].address = AMU_ABI_LDS_SIZE_AVAIL;
    newInfos[i].value = options.ldsSize_;
    i++;
    newInfos[i].address = AMU_ABI_LDS_SIZE_USED;
    newInfos[i].value = cShader->sqLdsAllocCs.bits.SIZE;
    i++;
    newInfos[i].address = AMU_ABI_STACK_SIZE_AVAIL;
    newInfos[i].value = options.numStackEntries_;
    i++;
    newInfos[i].address = AMU_ABI_STACK_SIZE_USED;
    newInfos[i].value = cShader->sqPgmResourcesCs.bits.STACK_SIZE;
    i++;

    for (unsigned int j = 0;j <SC_R800_MAX_UAV; j++)
    {
        unsigned int bufferSize = cShader->scUavRtnBufInfoTbl[j].stride;

        bufferSize *= 4; // convert from DWORDS to bytes

        //
        // multiply by the maximum number of threads in flight at one time
        //
        // 256 waves * 64 threads/wave * 2 shader engines (for 870)
        //
        bufferSize *= nullDev().hwInfo()->simdWidth_ * 4; // threads/wave
        bufferSize *= 256 * 4; // maximum number of waves

        bufferSize *= numShaderEngines;

        newInfos[i].address = AMU_ABI_SET_BUFFER_FOR_UAV_RET_BUFFER0 + j;
        newInfos[i].value = bufferSize;
        i++;
    }
    newInfos[i].address = AMU_ABI_GLOBAL_RETURN_BUFFER;
    newInfos[i].value = true;
    i++;
    // Always use extended caching with global return buffer
    newInfos[i].address = AMU_ABI_EXTENDED_CACHING;
    newInfos[i].value = true;
    i++;
    newInfos[i].address = AMU_ABI_NUM_GLOBAL_UAV;
    newInfos[i].value = SC_R800_MAX_UAV;
    i++;
    newInfos[i].address = AMU_ABI_NUM_GLOBAL_RETURN_BUFFER;
    newInfos[i].value = R800_GLOBAL_RTN_BUF_LAST;
    i++;
    {
        unsigned int bufferSize = cShader->u32GlobalRtnBufSlot;

        bufferSize *= 4; // convert from DWORDS to bytes

        //
        // multiply by the maximum number of threads in flight at one time
        //
        // 256 waves * 64 threads/wave * 2 shader engines (for 870)
        //
        bufferSize *= nullDev().hwInfo()->simdWidth_ * 4; // threads/wave
        bufferSize *= 256 * 4; // maximum number of waves

        bufferSize *= numShaderEngines;

        newInfos[i].address = AMU_ABI_GLOBAL_RETURN_BUFFER_SIZE;
        newInfos[i].value = bufferSize;
        i++;
    }
    {
        unsigned int bufferSize = cShader->u32GlobalRtnBufSlotShort;

        bufferSize *= 4; // convert from DWORDS to bytes

        //
        // multiply by the maximum number of threads in flight at one time
        //
        // 256 waves * 64 threads/wave * 2 shader engines (for 870)
        //
        bufferSize *= nullDev().hwInfo()->simdWidth_ * 4; // threads/wave
        bufferSize *= 256 * 4; // maximum number of waves

        bufferSize *= numShaderEngines;

        newInfos[i].address = AMU_ABI_GLOBAL_RETURN_BUFFER_SIZE_SHORT;
        newInfos[i].value = bufferSize;
        i++;
    }
    {
        unsigned int bufferSize = cShader->u32GlobalRtnBufSlotByte;

        bufferSize *= 4; // convert from DWORDS to bytes

        //
        // multiply by the maximum number of threads in flight at one time
        //
        // 256 waves * 64 threads/wave * 2 shader engines (for 870)
        //
        bufferSize *= nullDev().hwInfo()->simdWidth_ * 4;  // threads/wave
        bufferSize *= 256 * 4; // maximum number of waves

        bufferSize *= numShaderEngines;

        newInfos[i].address = AMU_ABI_GLOBAL_RETURN_BUFFER_SIZE_BYTE;
        newInfos[i].value = bufferSize;
        i++;
    }
    for (unsigned int j = 0; j < SC_R800_MAX_UAV; j++)
    {
        newInfos[i].address = AMU_ABI_OFFSET_TO_UAV0+j;
        newInfos[i].value = j;
        i++;
    }
    for (unsigned int j = 0; j < SC_R800_MAX_UAV; j++)
    {
        // Set up UAV->fetch constant mapping for uncached
        newInfos[i].address = AMU_ABI_UNCACHED_FETCH_CONST_UAV0+j;
        newInfos[i].value = UncachedFetchConst[j];
        i++;
    }
    for (unsigned int j = 0; j < SC_R800_MAX_UAV; j++)
    {
        newInfos[i].address = AMU_ABI_CACHED_FETCH_CONST_UAV0+j;
        newInfos[i].value = CachedFetchConst[j];
        i++;
    }
    for (unsigned int j = 0; j < R800_GLOBAL_RTN_BUF_LAST; j++)
    {
        newInfos[i].address = AMU_ABI_GLOBAL_RETURN_FETCH_CONST0+j;
        newInfos[i].value = GlobalReturnFetchConst[j];
        i++;
    }
    for (unsigned int j = 0; j < R800_GLOBAL_RTN_BUF_LAST; j++)
    {
        newInfos[i].address = AMU_ABI_GLOBAL_RETURN_BUFFER_TYPE0+j;
        newInfos[i].value = GlobalReturnBufferType[j];
        i++;
    }
    for (unsigned int j = 0; j < SC_R800_MAX_UAV; j++)
    {
        newInfos[i].address = AMU_ABI_CACHED_FETCH_CONST64_UAV0+j;
        newInfos[i].value = CachedFetchConst64[j];
        i++;
    }
    for (unsigned int j = 0; j < SC_R800_MAX_UAV; j++)
    {
        newInfos[i].address = AMU_ABI_CACHED_FETCH_CONST128_UAV0+j;
        newInfos[i].value = CachedFetchConst128[j];
        i++;
    }

    assert(i == infoCount);
    encoding.progInfosCount    = infoCount;

    encoding.uavMask.mask[0] = cShader->u32RatOpIsUsed;
    encoding.textData  = HWSHADER_Get(cShader, hShaderMemHandle);
    encoding.textSize  = cShader->CodeLenInByte;
    instructionCnt_ = encoding.textSize / sizeof(uint32_t);
    encoding.scratchRegisterCount = cShader->MaxScratchRegsNeeded;

    uint bufferSize = 0;
    bufferSize = cShader->u32GlobalRtnBufSlot +
        cShader->u32GlobalRtnBufSlotShort + cShader->u32GlobalRtnBufSlotByte;
    bufferSize *= 4; // convert from DWORDS to bytes

    //
    // multiply by the maximum number of threads in flight at one time
    //
    // 256 waves * 64 threads/wave * 2 shader engines (for 870)
    //
    bufferSize *= nullDev().hwInfo()->simdWidth_ * 4; // threads/wave
    bufferSize *= 256 * 4; // maximum number of waves

    bufferSize *= numShaderEngines;
    encoding.UAVReturnBufferTotalSize = bufferSize;

    return true;
}

} // namespace gpu

