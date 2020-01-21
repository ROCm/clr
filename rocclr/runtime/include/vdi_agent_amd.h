//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef _VDI_AGENT_AMD_H
#define _VDI_AGENT_AMD_H

#include <CL/cl.h>
#include "amdocl/cl_icd_amd.h"

#define cl_amd_agent 1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef const struct _vdi_agent vdi_agent;

#define VDI_AGENT_VERSION_1_0 100

/* Context Callbacks */

typedef void(CL_CALLBACK* acContextCreate_fn)(vdi_agent* /* agent */, cl_context /* context */);

typedef void(CL_CALLBACK* acContextFree_fn)(vdi_agent* /* agent */, cl_context /* context */);

/* Command Queue Callbacks */

typedef void(CL_CALLBACK* acCommandQueueCreate_fn)(vdi_agent* /* agent */,
                                                   cl_command_queue /* queue */);

typedef void(CL_CALLBACK* acCommandQueueFree_fn)(vdi_agent* /* agent */,
                                                 cl_command_queue /* queue */);

/* Event Callbacks */

typedef void(CL_CALLBACK* acEventCreate_fn)(vdi_agent* /* agent */, cl_event /* event */,
                                            cl_command_type /* type */);

typedef void(CL_CALLBACK* acEventFree_fn)(vdi_agent* /* agent */, cl_event /* event */);

typedef void(CL_CALLBACK* acEventStatusChanged_fn)(vdi_agent* /* agent */, cl_event /* event */,
                                                   cl_int /* execution_status */,
                                                   cl_long /* epoch_time_stamp */);

/* Memory Object Callbacks */

typedef void(CL_CALLBACK* acMemObjectCreate_fn)(vdi_agent* /* agent */, cl_mem /* memobj */);

typedef void(CL_CALLBACK* acMemObjectFree_fn)(vdi_agent* /* agent */, cl_mem /* memobj */);

typedef void(CL_CALLBACK* acMemObjectAcquired_fn)(vdi_agent* /* agent */, cl_mem /* memobj */,
                                                  cl_device_id /* device */,
                                                  cl_long /* elapsed_time */);

/* Sampler Callbacks */

typedef void(CL_CALLBACK* acSamplerCreate_fn)(vdi_agent* /* agent */, cl_sampler /* sampler */);

typedef void(CL_CALLBACK* acSamplerFree_fn)(vdi_agent* /* agent */, cl_sampler /* sampler */);

/* Program Callbacks */

typedef void(CL_CALLBACK* acProgramCreate_fn)(vdi_agent* /* agent */, cl_program /* program */);

typedef void(CL_CALLBACK* acProgramFree_fn)(vdi_agent* /* agent */, cl_program /* program */);

typedef void(CL_CALLBACK* acProgramBuild_fn)(vdi_agent* /* agent */, cl_program /* program */);

/* Kernel Callbacks */

typedef void(CL_CALLBACK* acKernelCreate_fn)(vdi_agent* /* agent */, cl_kernel /* kernel */);

typedef void(CL_CALLBACK* acKernelFree_fn)(vdi_agent* /* agent */, cl_kernel /* kernel */);

typedef void(CL_CALLBACK* acKernelSetArg_fn)(vdi_agent* /* agent */, cl_kernel /* kernel */,
                                             cl_int /* arg_index */, size_t /* size */,
                                             const void* /* value_ptr */);

typedef struct _vdi_agent_callbacks {
  /* Context Callbacks */
  acContextCreate_fn ContextCreate;
  acContextFree_fn ContextFree;

  /* Command Queue Callbacks */
  acCommandQueueCreate_fn CommandQueueCreate;
  acCommandQueueFree_fn CommandQueueFree;

  /* Event Callbacks */
  acEventCreate_fn EventCreate;
  acEventFree_fn EventFree;
  acEventStatusChanged_fn EventStatusChanged;

  /* Memory Object Callbacks */
  acMemObjectCreate_fn MemObjectCreate;
  acMemObjectFree_fn MemObjectFree;
  acMemObjectAcquired_fn MemObjectAcquired;

  /* Sampler Callbacks */
  acSamplerCreate_fn SamplerCreate;
  acSamplerFree_fn SamplerFree;

  /* Program Callbacks */
  acProgramCreate_fn ProgramCreate;
  acProgramFree_fn ProgramFree;
  acProgramBuild_fn ProgramBuild;

  /* Kernel Callbacks */
  acKernelCreate_fn KernelCreate;
  acKernelFree_fn KernelFree;
  acKernelSetArg_fn KernelSetArg;

} vdi_agent_callbacks;

typedef cl_uint vdi_agent_capability_action;

#define VDI_AGENT_ADD_CAPABILITIES 0x0
#define VDI_AGENT_RELINQUISH_CAPABILITIES 0x1

typedef struct _vdi_agent_capabilities {
  cl_bitfield canGenerateContextEvents : 1;
  cl_bitfield canGenerateCommandQueueEvents : 1;
  cl_bitfield canGenerateEventEvents : 1;
  cl_bitfield canGenerateMemObjectEvents : 1;
  cl_bitfield canGenerateSamplerEvents : 1;
  cl_bitfield canGenerateProgramEvents : 1;
  cl_bitfield canGenerateKernelEvents : 1;

} vdi_agent_capabilities;

struct _vdi_agent {
  cl_int(CL_API_CALL* GetVersionNumber)(vdi_agent* /* agent */, cl_int* /* version_ret */);

  cl_int(CL_API_CALL* GetPlatform)(vdi_agent* /* agent */, cl_platform_id* /* platform_id_ret */);

  cl_int(CL_API_CALL* GetTime)(vdi_agent* /* agent */, cl_long* /* time_nanos */);

  cl_int(CL_API_CALL* SetCallbacks)(vdi_agent* /* agent */,
                                    const vdi_agent_callbacks* /* callbacks */, size_t /* size */);


  cl_int(CL_API_CALL* GetPotentialCapabilities)(vdi_agent* /* agent */,
                                                vdi_agent_capabilities* /* capabilities */);

  cl_int(CL_API_CALL* GetCapabilities)(vdi_agent* /* agent */,
                                       vdi_agent_capabilities* /* capabilities */);

  cl_int(CL_API_CALL* SetCapabilities)(vdi_agent* /* agent */,
                                       const vdi_agent_capabilities* /* capabilities */,
                                       vdi_agent_capability_action /* action */);


  cl_int(CL_API_CALL* GetICDDispatchTable)(vdi_agent* /* agent */,
                                           cl_icd_dispatch_table* /* table */, size_t /* size */);

  cl_int(CL_API_CALL* SetICDDispatchTable)(vdi_agent* /* agent */,
                                           const cl_icd_dispatch_table* /* table */,
                                           size_t /* size */);

  /* add Kernel/Program helper functions, etc... */
};

extern cl_int CL_CALLBACK vdiAgent_OnLoad(vdi_agent* /* agent */);

extern void CL_CALLBACK vdiAgent_OnUnload(vdi_agent* /* agent */);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _VDI_AGENT_AMD_H */
