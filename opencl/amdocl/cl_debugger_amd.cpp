/* Copyright (c) 2014-present Advanced Micro Devices, Inc.

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

#include "cl_common.hpp"
#include "cl_debugger_amd.h"

#include <cstring>

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup AMD_Extensions
 *  @{
 *
 */

/*! \brief Set up the the dispatch call back function
 *
 *  \param device  specifies the device to be used
 *
 *  \param preDispatchFunction  is the function to be called before dispatching the kernel
 *
 *  \param postDispatchFunction  is the function to be called after kernel execution
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgSetCallBackFunctionsAMD,
              (cl_device_id device, cl_PreDispatchCallBackFunctionAMD preDispatchFunction,
               cl_PostDispatchCallBackFunctionAMD postDispatchFunction)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->setCallBackFunctions(preDispatchFunction, postDispatchFunction);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Set up the arguments of the dispatch call back function
 *
 *  \param device  specifies the device to be used
 *
 *  \param preDispatchArgs  is the arguments for the pre-dispatch callback function
 *
 *  \param postDispatchArgs  is the arguments for the post-dispatch callback function
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgSetCallBackArgumentsAMD,
              (cl_device_id device, void* preDispatchArgs, void* postDispatchArgs)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->setCallBackArguments(preDispatchArgs, postDispatchArgs);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Invalidate all cache on the device.
 *
 *  \param device  specifies the device to be used
 *
 *  \param mask    is the mask to specify which cache to be flush/invalidate
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgFlushCacheAMD, (cl_device_id device, cl_dbg_gpu_cache_mask_amd mask)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->flushCache(mask.ui32All);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Set up an exception policy in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param policy  specifies the exception policy, which includes the exception mask,
 *                 wave action, host action, wave mode.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the policy is not specified (NULL)
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgSetExceptionPolicyAMD,
              (cl_device_id device, cl_dbg_exception_policy_amd* policy)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == policy) {
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->setExceptionPolicy(policy);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Get the exception policy in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param policy  is a pointer to the memory where the policy is returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the policy storage is not specified
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgGetExceptionPolicyAMD,
              (cl_device_id device, cl_dbg_exception_policy_amd* policy)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == policy) {
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->getExceptionPolicy(policy);

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Set up the kernel execution mode in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param mode    specifies the kernel execution mode, which indicate whether single
 *                 step mode is used, how many CUs are reserved.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the mode is not specified, ie, has a NULL value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgSetKernelExecutionModeAMD,
              (cl_device_id device, cl_dbg_kernel_exec_mode_amd* mode)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == mode) {
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->setKernelExecutionMode(mode);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Get the kernel execution mode in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param mode    is a pointer to the memory where the exectuion mode is returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the mode storage is not specified, ie, has a NULL value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgGetKernelExecutionModeAMD,
              (cl_device_id device, cl_dbg_kernel_exec_mode_amd* mode)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == mode) {
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->getKernelExecutionMode(mode);

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Create a trap event for debugging
 *
 *  \param device  specifies the device to be used
 *
 *  \param autoReset   is the auto reset flag
 *
 *  \param pDebugEvent returns the debug event to be used for exception notification
 *
 *  \param pEventId    is the event ID, which is not used at this moment
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pDebugEvent value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_OUT_OF_RESOURCES if fails to create the event
 */
RUNTIME_ENTRY(cl_int, clHwDbgCreateEventAMD, (cl_device_id device, bool autoReset,
                                              cl_dbg_event_amd* pDebugEvent, cl_uint* pEventId)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == pDebugEvent) {
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  // set it to zero for now - not used by OpenCL
  *pEventId = 0;
  *pDebugEvent = debugManager->createDebugEvent(autoReset);

  return (NULL == pDebugEvent) ? CL_OUT_OF_RESOURCES : CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Wait for a debug event to be signaled
 *
 *  \param device  specifies the device to be used
 *
 *  \param pDebugEvent is the debug event to be waited for
 *
 *  \param pEventId    is the event ID, which is not used at this moment
 *
 *  \param timeOut     is the duration for waiting
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pDebugEvent value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_EVENT_TIMEOUT_AMD if timeout occurs
 */
RUNTIME_ENTRY(cl_int, clHwDbgWaitEventAMD, (cl_device_id device, cl_dbg_event_amd pDebugEvent,
                                            cl_uint pEventId, cl_uint timeOut)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (0 == pDebugEvent) {
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  return debugManager->waitDebugEvent(pDebugEvent, timeOut);
}
RUNTIME_EXIT

/*! \brief Destroy a trap event for debugging
 *
 *  \param device  specifies the device to be used
 *
 *  \param pDebugEvent is the debug event to be waited for
 *
 *  \param pEventId    is the event ID, which is not used at this moment
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pDebugEvent value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgDestroyEventAMD,
              (cl_device_id device, cl_dbg_event_amd* pDebugEvent, cl_uint* pEventId)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == pDebugEvent) {
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->destroyDebugEvent(pDebugEvent);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Register the debugger on a device
 *
 *  \param context specifies the context for the debugger
 *
 *  \param device  specifies the device to be used
 *
 *  \param pMessageStorge specifies the memory for trap message passing between KMD and OCL runtime
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_CONTEXT if the context is not valid
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pMEssageStorge value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_OUT_OF_RESOURCES if a host queue cannot be created for the debugger
 */
RUNTIME_ENTRY(cl_int, clHwDbgRegisterDebuggerAMD,
              (cl_context context, cl_device_id device, volatile void* pMessageStorage)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (!is_valid(context)) {
    return CL_INVALID_CONTEXT;
  }

  if (NULL == pMessageStorage) {
    return CL_INVALID_VALUE;
  }

  if (NULL == as_amd(device)->hwDebugMgr()) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  return as_amd(device)->hwDebugManagerInit(as_amd(context),
                                            reinterpret_cast<uintptr_t>(pMessageStorage));
}
RUNTIME_EXIT


/*! \brief Unregister the debugger on a device
 *
 *  \param device  specifies the device to be used
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgUnregisterDebuggerAMD, (cl_device_id device)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->unregisterDebugger();

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Setup the pointer of the acl_binary to be used by the debugger
 *
 *  \param device  specifies the device to be used
 *
 *  \param aclBinary  specifies the ACL binary to be used
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the aclBinary is not provided
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgSetAclBinaryAMD, (cl_device_id device, void* aclBinary)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == aclBinary) {
    LogWarning("clHwDbgSetAclBinaryAMD: Invalid ACL binary argument.");
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->setAclBinary(aclBinary);

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Control the execution of wavefront on the GPU
 *
 *  \param device       specifies the device to be used
 *
 *  \param action       specifies the wave action - halt, resume, kill, debug
 *
 *  \param mode         specifies the wave mode
 *
 *  \param trapId       specifies the trap ID, which should be 0x7
 *
 *  \param waveAddress  specifies the wave address for the wave control
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the waveMsg is not provided, invalid action or mode value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgWaveControlAMD,
              (cl_device_id device, cl_dbg_waves_action_amd action, cl_dbg_wave_mode_amd mode,
               cl_uint trapId, cl_dbg_wave_addr_amd waveAddress)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  //  validate the passing arguments
  //
  if (action < 0 || action >= CL_DBG_WAVES_MAX) {
    LogWarning("clHwDbgWaveControlAMD: Invalid wave action argument");
    return CL_INVALID_VALUE;
  }

  if ((mode != CL_DBG_WAVEMODE_SINGLE) && (mode != CL_DBG_WAVEMODE_BROADCAST) &&
      (mode != CL_DBG_WAVEMODE_BROADCAST_CU)) {
    LogWarning("clHwDbgWaveControlAMD: Invalid wave mode argument");
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->wavefrontControl(action, mode, trapId, (void*)&waveAddress);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Set watch points on memory address ranges to generate exception events
 *
 *  \param device          specifies the device to be used
 *
 *  \param numWatchPoints  specifies the number of watch points
 *
 *  \param watchMode       is the array of watch mode for the watch points
 *
 *  \param watchAddress    is the array of watch address for the watch points
 *
 *  \param watchMask       is the array of mask for the watch points
 *
 *  \param watchEvent      is the array of event for the watch points
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the number of points <= 0, or other parameters is not specified
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgAddressWatchAMD,
              (cl_device_id device, cl_uint numWatchPoints,
               cl_dbg_address_watch_mode_amd* watchMode, void** watchAddress, cl_ulong* watchMask,
               cl_dbg_event_amd* watchEvent)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  //  validate the passing arguments
  if (numWatchPoints <= 0) {
    LogWarning("clHwDbgAddressWatchAMD: Invalid number of watch points argument");
    return CL_INVALID_VALUE;
  }

  if (NULL == watchMode) {
    LogWarning("clHwDbgAddressWatchAMD: Watch mode argument");
    return CL_INVALID_VALUE;
  }

  if (NULL == watchAddress) {
    LogWarning("clHwDbgAddressWatchAMD: Watch address argument");
    return CL_INVALID_VALUE;
  }

  if (NULL == watchMask) {
    LogWarning("clHwDbgAddressWatchAMD: Watch mask argument");
    return CL_INVALID_VALUE;
  }

  // TODO: WC - confirm how the watch event is used.
  //
  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->setAddressWatch(numWatchPoints, watchAddress, watchMask,
                                reinterpret_cast<cl_ulong*>(watchMode), watchEvent);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Get the AQL packet information for kernel dispatch
 *
 *  \param device  specifies the device to be used
 *
 *  \param aqlPacket     specifies the AQL packet
 *
 *  \param aqlCodeInfo   specifies the kernel code and its size
 *
 *  \param packetInfo    points to the memory for the packet information to be returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgGetAqlPacketInfoAMD,
              (cl_device_id device, const void* aqlCodeInfo, cl_aql_packet_info_amd* packetInfo)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->getPacketAmdInfo(aqlCodeInfo, packetInfo);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Get the dispatch debug information
 *
 *  \param device  specifies the device to be used
 *
 *  \param debugInfo  points to the memory for the debug information to be returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgGetDispatchDebugInfoAMD,
              (cl_device_id device, cl_dispatch_debug_info_amd* debugInfo)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == debugInfo) {
    LogWarning("clHwDbgGetDispatchDebugInfoAMD: Invalid debug information pointer.");
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->getDispatchDebugInfo((void*)debugInfo);

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Map the video memory for the kernel code to allow host access
 *
 *  \param device          specifies the device to be used
 *
 *  \param aqlCodeInfo   specifies the kernel code and its size
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgMapKernelCodeAMD, (cl_device_id device, void* aqlCodeInfo)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->mapKernelCode(aqlCodeInfo);

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Unmap the video memory for the kernel code
 *
 *  \param device          specifies the device to be used (no needed, just to be consistent)
 *
 *  \param aqlCodeAddress  is the memory points to the mapped memory address for the kernel code
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgUnmapKernelCodeAMD, (cl_device_id device, cl_ulong* aqlCodeAddress)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == aqlCodeAddress) {
    LogWarning("clHwDbgUnmapKernelCodeAMD: Invalid AQL code address argument.");
    return CL_INVALID_VALUE;
  }

  // Shader buffer is always pinned to host memory so there is no need to unmap the memory.
  // Just set it to 0 to avoid unwanted access
  *aqlCodeAddress = 0;

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Map the scratch ring's memory to allow CPU access
 *
 *  \param device  specifies the device to be used
 *
 *  \param scratchRingAddr  is the memory points to the returned host memory address for scratch
 * ring
 *
 *  \param scratchRingSize  returns the size of the scratch ring
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgMapScratchRingAMD,
              (cl_device_id device, cl_ulong* scratchRingAddr, cl_uint* scratchRingSize)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  debugManager->mapScratchRing(scratchRingAddr, scratchRingSize);

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Unmap the shader scratch ring's video memory
 *
 *  \param device           specifies the device to be used (no needed, just to be consistent)
 *
 *  \param scratchRingAddr  is the memory points to the mapped memory address for scratch ring
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgUnmapScratchRingAMD,
              (cl_device_id device, cl_ulong* scratchRingAddr)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (NULL == scratchRingAddr) {
    LogWarning("clHwDbgUnmapScratchRingAMD: Invalid scratch ring address argument.");
    return CL_INVALID_VALUE;
  }

  // Scratch ring buffer is always pinned to host memory so there is no need to unmap the memory.
  // Just set it to NULL to avoid unwanted access
  *scratchRingAddr = 0;

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Get the memory object associated with the kernel parameter
 *
 *  \param device     specifies the device to be used
 *
 *  \param paramIdx   is the index of of the kernel argument
 *
 *  \param paramMem   is pointer of the memory associated with the kernel argument to be returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the paramIdx is less than zero, or the paramMem has NULL value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_INVALID_KERNEL_ARGS if it fails to get the memory object for the kernel argument
 */
RUNTIME_ENTRY(cl_int, clHwDbgGetKernelParamMemAMD,
              (cl_device_id device, cl_uint paramIdx, cl_mem* paramMem)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::Device* amdDevice = as_amd(device);

  if (paramIdx < 0) {
    LogWarning("clHwDbgGetKernelParamMemAMD: Invalid parameter index argument.");
    return CL_INVALID_VALUE;
  }

  if (NULL == paramMem) {
    LogWarning("clHwDbgGetKernelParamMemAMD: Invalid parameter member object argument.");
    return CL_INVALID_VALUE;
  }

  amd::HwDebugManager* debugManager = amdDevice->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  *paramMem = debugManager->getKernelParamMem(paramIdx);

  return (*paramMem == 0) ? CL_INVALID_KERNEL_ARGS : CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Set value of a global memory object
 *
 *  \param device      specifies the device to be used
 *
 *  \param memObject   is the memory object handle to be assigned the value specified in srcMem.
 *
 *  \param offset      is offset of the memory object
 *
 *  \param srcMem      points to the memory which contains the values to be assigned to the memory
 *
 *  \param size        size (in bytes) of the srcMem
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if memObj or srcPtr has NULL value, size <= 0 or offset < 0
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgSetGlobalMemoryAMD,
              (cl_device_id device, cl_mem memObject, cl_uint offset, void* srcMem, cl_uint size)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  if (0 > offset || 0 >= size) {
    return CL_INVALID_VALUE;
  }

  amd::Memory* globalMem = as_amd(memObject);
  debugManager->setGlobalMemory(globalMem, offset, srcMem, size);

  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! \brief Install the trap handler of a given type
 *
 *  \param device      specifies the device to be used
 *
 *  \param trapType    is the type of trap handler
 *
 *  \param trapHandler is the pointer of trap handler (TBA)
 *
 *  \param trapBuffer  is the pointer of trap handler buffer (TMA)
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
RUNTIME_ENTRY(cl_int, clHwDbgInstallTrapAMD, (cl_device_id device, cl_dbg_trap_type_amd trapType,
                                              cl_mem trapHandler, cl_mem trapBuffer)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  amd::HwDebugManager* debugManager = as_amd(device)->hwDebugMgr();
  if (NULL == debugManager) {
    return CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD;
  }

  amd::Memory* pTrapHandler = as_amd(trapHandler);
  amd::Memory* pTrapBuffer = as_amd(trapBuffer);
  debugManager->installTrap(trapType, pTrapHandler, pTrapBuffer);

  return CL_SUCCESS;
}

RUNTIME_EXIT


/*! @}
 *  @}
 */
