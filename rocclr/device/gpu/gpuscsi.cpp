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

#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gpuprogram.hpp"
#include "device/gpu/gpukernel.hpp"
#include "acl.h"
#include "SCShadersSi.h"
#include "si_ci_vi_merged_offset.h"
#include "si_ci_vi_merged_registers.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include "amd_hsa_loader.hpp"

namespace gpu {

bool NullKernel::siCreateHwInfo(const void* shader, AMUabiAddEncoding& encoding) {
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
    newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD0 + 4 * j;
    newInfos[i].value = HWSHADER_Get(cShader, common.pUserElements)[j].dataClass;
    i++;
    newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD1 + 4 * j;
    newInfos[i].value = HWSHADER_Get(cShader, common.pUserElements)[j].apiSlot;
    i++;
    newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD2 + 4 * j;
    newInfos[i].value = HWSHADER_Get(cShader, common.pUserElements)[j].startUserReg;
    i++;
    newInfos[i].address = AMU_ABI_USER_ELEMENTS_0_DWORD3 + 4 * j;
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
  newInfos[i].value = SI_sgprs_avail;  // 512;//options.NumSGPRsAvailable;
  i++;
  newInfos[i].address = AMU_ABI_SI_NUM_VGPRS_AVAIL;
  newInfos[i].value = SI_vgprs_avail;  // options.NumVGPRsAvailable;
  i++;

  newInfos[i].address = AMU_ABI_SI_FLOAT_MODE;
  newInfos[i].value = cShader->common.floatMode;
  i++;
  newInfos[i].address = AMU_ABI_SI_IEEE_MODE;
  newInfos[i].value = cShader->common.bIeeeMode;
  i++;

  newInfos[i].address = AMU_ABI_SI_SCRATCH_SIZE;
  newInfos[i].value = cShader->common.scratchSize;
  ;
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

  newInfos[i].address = AMU_ABI_NUM_WAVEFRONT_PER_SIMD;  // Setting the same as for scWrapR800Info
  newInfos[i].value = 1;
  i++;

  newInfos[i].address = AMU_ABI_WAVEFRONT_SIZE;
  newInfos[i].value = nullDev().hwInfo()->simdWidth_ * 4;  // options.WavefrontSize;
  i++;

  newInfos[i].address = AMU_ABI_LDS_SIZE_AVAIL;
  newInfos[i].value = SI_ldssize_avail;  // options.LDSSize;
  i++;

  COMPUTE_PGM_RSRC2 computePgmRsrc2;
  computePgmRsrc2.u32All = cShader->computePgmRsrc2.u32All;

  newInfos[i].address = AMU_ABI_LDS_SIZE_USED;
  newInfos[i].value = 64 * 4 * computePgmRsrc2.bits.LDS_SIZE;
  i++;

  infoCount = i;
  assert((i + 4 * (16 - cShader->common.userElementCount)) == NumSiCsInfos);
  encoding.progInfosCount = infoCount;

  encoding.textData = HWSHADER_Get(cShader, common.hShaderMemHandle);
  encoding.textSize = cShader->common.codeLenInByte;
  instructionCnt_ = encoding.textSize / sizeof(uint32_t);
  encoding.scratchRegisterCount = cShader->common.scratchSize;
  encoding.UAVReturnBufferTotalSize = 0;

  return true;
}

bool HSAILKernel::aqlCreateHWInfo(amd::hsa::loader::Symbol* sym) {
  if (!sym) {
    return false;
  }
  uint64_t akc_addr = 0;
  if (!sym->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, reinterpret_cast<void*>(&akc_addr))) {
    return false;
  }
  amd_kernel_code_t* akc = reinterpret_cast<amd_kernel_code_t*>(akc_addr);
  cpuAqlCode_ = akc;
  if (!sym->GetInfo(HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_SIZE,
                    reinterpret_cast<void*>(&codeSize_))) {
    return false;
  }
  size_t akc_align = 0;
  if (!sym->GetInfo(HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_ALIGN,
                    reinterpret_cast<void*>(&akc_align))) {
    return false;
  }

  // Allocate HW resources for the real program only
  if (!prog().isNull()) {
    code_ = new gpu::Memory(dev(), amd::alignUp(codeSize_, akc_align));
    // Initialize kernel ISA code
    if (code_ && code_->create(Resource::Shader)) {
      address cpuCodePtr = static_cast<address>(code_->map(NULL, Resource::WriteOnly));
      // Copy only amd_kernel_code_t
      memcpy(cpuCodePtr, reinterpret_cast<address>(akc), codeSize_);
      code_->unmap(NULL);
    } else {
      LogError("Failed to allocate ISA code!");
      return false;
    }
  }

  assert((akc->workitem_private_segment_byte_size & 3) == 0 && "Scratch must be DWORD aligned");
  workGroupInfo_.scratchRegs_ =
      amd::alignUp(akc->workitem_private_segment_byte_size, 16) / sizeof(uint);
  workGroupInfo_.privateMemSize_ = akc->workitem_private_segment_byte_size;
  workGroupInfo_.availableLDSSize_ = dev().info().localMemSize_;
  workGroupInfo_.localMemSize_ = workGroupInfo_.usedLDSSize_ =
      akc->workgroup_group_segment_byte_size;
  workGroupInfo_.usedSGPRs_ = akc->wavefront_sgpr_count;
  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.usedVGPRs_ = akc->workitem_vgpr_count;

  if (!prog().isNull()) {
    workGroupInfo_.availableSGPRs_ = dev().gslCtx()->getNumSGPRsAvailable();
    workGroupInfo_.availableVGPRs_ = dev().gslCtx()->getNumVGPRsAvailable();
    workGroupInfo_.preferredSizeMultiple_ = dev().getAttribs().wavefrontSize;
    workGroupInfo_.wavefrontPerSIMD_ = dev().getAttribs().wavefrontSize;
  } else {
    workGroupInfo_.availableSGPRs_ = 104;
    workGroupInfo_.availableVGPRs_ = 256;
    workGroupInfo_.preferredSizeMultiple_ = workGroupInfo_.wavefrontPerSIMD_ = 64;
  }
  return true;
}
}  // namespace gpu
