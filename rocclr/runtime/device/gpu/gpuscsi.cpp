//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpukernel.hpp"
#include "acl.h"
#include "SCShadersSi.h"
#include "si_ci_merged_offset.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>

namespace gpu {

bool
NullKernel::siCreateHwInfo(const void* shader, AMUabiAddEncoding& encoding)
{
    static const uint NumSiCsInfos = (70 + 5 + 1 + 32 + 6);
    CALProgramInfoEntry* newInfos;
    uint i = 0;
    uint infoCount = NumSiCsInfos;
    const SC_SI_HWSHADER_CS* cShader = reinterpret_cast<const SC_SI_HWSHADER_CS*>(shader);
    newInfos = new CALProgramInfoEntry[infoCount];
    encoding.progInfos = newInfos;
    if (encoding.progInfos == 0) {
        infoCount = 0;
        return false;
    }
    newInfos[i].address = AMU_ABI_USER_ELEMENT_COUNT;
    newInfos[i].value = cShader->common.userElementCount;
    i++;
    for (unsigned int j = 0; j < cShader->common.userElementCount; j++) {
        newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD0 + 4*j;
        newInfos[i].value = HWSHADER_Get(cShader, common.pUserElements)[j].dataClass;
        i++;
        newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD1 + 4*j;
        newInfos[i].value = HWSHADER_Get(cShader, common.pUserElements)[j].apiSlot;
        i++;
        newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD2 + 4*j;
        newInfos[i].value = HWSHADER_Get(cShader, common.pUserElements)[j].startUserReg;
        i++;
        newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD3 + 4*j;
        newInfos[i].value = HWSHADER_Get(cShader, common.pUserElements)[j].userRegCount;
        i++;
    }

    newInfos[i].address = AMU_ABI_SI_NUM_VGPRS;
    newInfos[i].value = cShader->common.numVgprs;
    i++;
    newInfos[i].address = AMU_ABI_SI_NUM_SGPRS;
    newInfos[i].value = cShader->common.numSgprs;
    i++;
    newInfos[i].address = AMU_ABI_SI_NUM_SGPRS_AVAIL;
    newInfos[i].value = 104-2; //512;//options.NumSGPRsAvailable;
    i++;
    newInfos[i].address = AMU_ABI_SI_NUM_VGPRS_AVAIL;
    newInfos[i].value = 256;//options.NumVGPRsAvailable;
    i++;

    newInfos[i].address = AMU_ABI_SI_FLOAT_MODE;
    newInfos[i].value = cShader->common.floatMode;
    i++;
    newInfos[i].address = AMU_ABI_SI_IEEE_MODE;
    newInfos[i].value = cShader->common.bIeeeMode;
    i++;

    newInfos[i].address = AMU_ABI_SI_SCRATCH_SIZE;
    newInfos[i].value = cShader->common.scratchSize;;
    i++;

    newInfos[i].address = mmCOMPUTE_PGM_RSRC2;
    newInfos[i].value = cShader->computePgmRsrc2.u32All;
    i++;

    newInfos[i].address = AMU_ABI_NUM_THREAD_PER_GROUP_X;
    newInfos[i].value = cShader->numThreadX;
    i++;
    newInfos[i].address = AMU_ABI_NUM_THREAD_PER_GROUP_Y;
    newInfos[i].value = cShader->numThreadY;
    i++;
    newInfos[i].address = AMU_ABI_NUM_THREAD_PER_GROUP_Z;
    newInfos[i].value = cShader->numThreadZ;
    i++;

    newInfos[i].address = AMU_ABI_ORDERED_APPEND_ENABLE;
    newInfos[i].value = cShader->bOrderedAppendEnable;
    i++;

    newInfos[i].address = AMU_ABI_RAT_OP_IS_USED;
    newInfos[i].value = cShader->common.uavResourceUsage[0];
    i++;

    for (unsigned int j = 0; j < ((SC_MAX_UAV + 31) / 32); j++) {
        newInfos[i].address = AMU_ABI_UAV_RESOURCE_MASK_0 + j;
        newInfos[i].value = cShader->common.uavResourceUsage[j];
        i++;
    }

    newInfos[i].address = AMU_ABI_NUM_WAVEFRONT_PER_SIMD;      // Setting the same as for scWrapR800Info
    newInfos[i].value = 1;
    i++;

    newInfos[i].address = AMU_ABI_WAVEFRONT_SIZE;
    newInfos[i].value = nullDev().hwInfo()->simdWidth_ * 4;   //options.WavefrontSize;
    i++;

    newInfos[i].address = AMU_ABI_LDS_SIZE_AVAIL;
    newInfos[i].value = 32*1024;    //options.LDSSize;
    i++;

    newInfos[i].address = AMU_ABI_LDS_SIZE_USED;
    newInfos[i].value = 64 * 4 * cShader->computePgmRsrc2.bits.LDS_SIZE;
    i++;

    infoCount = i;
    assert((i + 4 * (16 - cShader->common.userElementCount)) == NumSiCsInfos);
    encoding.progInfosCount    = infoCount;

    CALUavMask uavMask;
    memcpy(uavMask.mask, cShader->common.uavResourceUsage, sizeof(CALUavMask));
    encoding.uavMask    = uavMask;
    encoding.textData   = HWSHADER_Get(cShader, common.hShaderMemHandle);
    encoding.textSize   = cShader->common.codeLenInByte;
    instructionCnt_     = encoding.textSize / sizeof(uint32_t);
    encoding.scratchRegisterCount  = cShader->common.scratchSize;
    encoding.UAVReturnBufferTotalSize  = 0;

    return true;
}

bool
HSAILKernel::aqlCreateHWInfo(const void* shader, size_t shaderSize)
{
    // Copy the shader_isa into a buffer
    hwMetaData_ = new char[shaderSize];
    if (hwMetaData_ == NULL) {
        return false;
    }
    memcpy(hwMetaData_, shader, shaderSize);

    SC_SI_HWSHADER_CS* siMetaData = reinterpret_cast<SC_SI_HWSHADER_CS*>(hwMetaData_);

    // Code to patch the pointers in the shader object.
    // Must be preferably done in the compiler library
    size_t offset = siMetaData->common.uSizeInBytes;
    if (siMetaData->common.u32PvtDataSizeInBytes > 0) {
        siMetaData->common.pPvtData =
            reinterpret_cast<SC_BYTE *>(
            reinterpret_cast<char *>(siMetaData) + offset);
        offset += siMetaData->common.u32PvtDataSizeInBytes;
    }
    if (siMetaData->common.codeLenInByte > 0) {
        siMetaData->common.hShaderMemHandle =
            reinterpret_cast<char *>(siMetaData) + offset;
        offset += siMetaData->common.codeLenInByte;
    }

    char* headerBaseAddress =
        reinterpret_cast<char*>(siMetaData->common.hShaderMemHandle);
    hsa_ext_code_descriptor_t* hcd =
        reinterpret_cast<hsa_ext_code_descriptor_t*>(headerBaseAddress);
    amd_kernel_code_t* akc = reinterpret_cast<amd_kernel_code_t*>(
        headerBaseAddress + hcd->code.handle);

    address codeStartAddress = reinterpret_cast<address>(akc);
    address codeEndAddress = reinterpret_cast<address>(hcd) + siMetaData->common.codeLenInByte;
    uint64_t codeSize = codeEndAddress - codeStartAddress;
    code_ = new gpu::Memory(dev(), amd::alignUp(codeSize, gpu::ConstBuffer::VectorSize));
    // Initialize kernel ISA code
    if ((code_ != NULL) && code_->create(Resource::Local)) {
        address cpuCodePtr = static_cast<address>(code_->map(NULL, Resource::WriteOnly));
        // Copy only amd_kernel_code_t
        memcpy(cpuCodePtr, codeStartAddress, codeSize);
        code_->unmap(NULL);
    }
    else {
        LogError("Failed to allocate ISA code!");
        return false;
    }
    cpuAqlCode_ = akc;

    assert((akc->workitem_private_segment_byte_size & 3) == 0 &&
        "Scratch must be DWORD aligned");
    workGroupInfo_.scratchRegs_ =
        amd::alignUp(akc->workitem_private_segment_byte_size, 16) / sizeof(uint);
    workGroupInfo_.availableSGPRs_ = dev().gslCtx()->getNumSGPRsAvailable();
    workGroupInfo_.availableVGPRs_ = dev().gslCtx()->getNumVGPRsAvailable();
    workGroupInfo_.preferredSizeMultiple_ = dev().getAttribs().wavefrontSize;
    workGroupInfo_.privateMemSize_ = akc->workitem_private_segment_byte_size;
    workGroupInfo_.localMemSize_ =
    workGroupInfo_.usedLDSSize_ = akc->workgroup_group_segment_byte_size;
    workGroupInfo_.usedSGPRs_ = akc->wavefront_sgpr_count;
    workGroupInfo_.usedStackSize_ = 0;
    workGroupInfo_.usedVGPRs_ = akc->workitem_vgpr_count;
    workGroupInfo_.wavefrontPerSIMD_ = dev().getAttribs().wavefrontSize;

    return true;
}
} // namespace gpu
