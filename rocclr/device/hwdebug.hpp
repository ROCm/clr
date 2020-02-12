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

#ifndef HWDEBUG_H_
#define HWDEBUG_H_

#include "device.hpp"
#include "amdocl/cl_debugger_amd.h"

static const int TbaStartOffset = 256;

static const int RtTrapBufferWaveSize = 64;
static const int RtTrapBufferSeNum = 4;
static const int RtTrapBufferShNum = 2;
static const int RtTrapBufferCuNum = 16;
static const int RtTrapBufferSimdNum = 4;
static const int RtTrapBufferWaveNum = 16;
static const int RtTrapBufferTotalWaveNum =
    ((RtTrapBufferSeNum) * (RtTrapBufferShNum) * (RtTrapBufferCuNum) * (RtTrapBufferSimdNum) *
     (RtTrapBufferWaveNum));


/*!  \brief Debug trap handler location in the runtime trap buffer
 *
 *   This enumeration is used to indicate the location where the debug
 *   trap handler and debug trap buffer are set in the device trap buffer.
 */
enum DebugTrapLocation {
  kDebugTrapHandlerLocation = 0,  //! Debug Trap handler location, this location must be 0
  kDebugTrapBufferLocation = 1,   //! Debug Trap buffer location, this location must be 1
  kDebugTrapLocationMax = 2
};


/*!  \brief This structure is for the debug info in each kernel dispatch.
 *
 *   Contains the memory descriptor information of the scratch memory and the global
 *   memory
 */
struct DispatchDebugInfo {
  uint32_t scratchMemoryDescriptor_[4];  //! Scratch memory descriptor
  uint32_t globalMemoryDescriptor_[4];   //! Global memory descriptor
};

/*!  \brief Trap handler descriptor
 *
 *   The trap handler descriptor contains the details of a given trap handler.
 */
struct TrapHandlerInfo {
  amd::Memory* trapHandler_;  //!< Device memory for the trap handler
  amd::Memory* trapBuffer_;   //!< Device memory for the trap buffer
};

/*!  \brief Structure of the runtime trap handler buffer, which includes the following
 *   information: information of the runtime trap handler and buffer, information of
 *   the level-2 trap handlers and buffers.
 */
struct RuntimeTrapInfo {
  TrapHandlerInfo trap_;  //!< Structure of the address of all trap handlers
  uint32_t dispatchId_;   //!< Dispatch ID that signals the shader event
  uint32_t vgpr_backup_[RtTrapBufferTotalWaveNum][RtTrapBufferWaveSize];
  //!< Buffer to backup the VGPR used by the runtime trap handler
};

/**
 * Opaque pointer to trap event
 */
typedef uintptr_t DebugEvent;

namespace amd {


class Context;
class Device;
class HostQueue;


/*! \class HwDebugManager
 *
 *  \brief The device interface class for the hardware debug manager
 */
class HwDebugManager {
 public:
  //! Constructor for the Hardware Debug Manager
  HwDebugManager(amd::Device* device);

  //! Destructor for Hardware Debug Manager
  virtual ~HwDebugManager();

  //!  Setup the call back function pointer
  void setCallBackFunctions(cl_PreDispatchCallBackFunctionAMD preDispatchFn,
                            cl_PostDispatchCallBackFunctionAMD postDispatchFn);

  //!  Setup the call back argument pointers
  void setCallBackArguments(void* preDispatchArgs, void* postDispatchArgs);

  //!  Get dispatch debug info
  void getDispatchDebugInfo(void* debugInfo) const;

  //!  Set the kernel code address and its size
  void setKernelCodeInfo(address aqlCodeAddr, uint32_t aqlCodeSize);

  //!  Get the scratch ring
  void setScratchRing(address scratchRingAddr, uint32_t scratchRingSize);

  //!  Map the scratch ring for host access
  void mapScratchRing(uint64_t* scratchRingAddr, uint32_t* scratchRingSize) const;

  //!  Retrieve the pre-dispatch callback function
  cl_PreDispatchCallBackFunctionAMD preDispatchCallBackFunc() const {
    return preDispatchCallBackFunc_;
  }

  //!  Retrieve the post-dispatch callback function
  cl_PostDispatchCallBackFunctionAMD postDispatchCallBackFunc() const {
    return postDispatchCallBackFunc_;
  }

  //!  Retrieve the pre-dispatch callback function arguments
  void* preDispatchCallBackArgs() const { return preDispatchCallBackArgs_; }

  //!  Retrieve the post-dispatch callback function arguments
  void* postDispatchCallBackArgs() const { return postDispatchCallBackArgs_; }

  //!  Retrieve the memory pointer of the runtime trap handler code
  device::Memory* runtimeTBA() const { return runtimeTBA_; }

  //!  Retrieve the memory pointer of the runtime trap handler buffer
  device::Memory* runtimeTMA() const { return runtimeTMA_; }

  //!  Set exception policy
  void setExceptionPolicy(void* policy);

  //!  Get exception policy
  void getExceptionPolicy(void* policy) const;

  //!  Set the kernel execution mode
  void setKernelExecutionMode(void* mode);

  //!  Get the kernel execution mode
  void getKernelExecutionMode(void* mode) const;

  //!  Setup the pointer to the aclBinary within the debug manager
  void setAclBinary(void* aclBinary);

  //!  Allocate storage to keep the memory pointers of the kernel parameters
  void allocParamMemList(uint32_t numParams);

  //!  Assign the kernel parameter memory
  void assignKernelParamMem(uint32_t paramIdx, amd::Memory* mem);

  //!  Get kernel parameter memory object
  cl_mem getKernelParamMem(uint32_t paramIdx) const;

  //!  Install trap handler
  void installTrap(cl_dbg_trap_type_amd trapType, amd::Memory* pTrapHandler,
                   amd::Memory* pTrapBuffer);

  //!  Flush cache
  virtual void flushCache(uint32_t mask) = 0;

  //!  Create the debug event
  virtual DebugEvent createDebugEvent(const bool autoReset) = 0;

  //!  Wait for the debug event
  virtual int32_t waitDebugEvent(DebugEvent pEvent, uint32_t timeOut) const = 0;

  //!  Destroy the debug event
  virtual void destroyDebugEvent(DebugEvent* pEvent) = 0;

  //!  Register the debugger
  virtual int32_t registerDebugger(amd::Context* context, uintptr_t pMessageStorage) = 0;

  //!  Unregister the debugger
  virtual void unregisterDebugger() = 0;

  //!  Send the wavefront control cmmand
  virtual void wavefrontControl(uint32_t waveAction, uint32_t waveMode, uint32_t trapId,
                                void* waveAddr) const = 0;

  //!  Set address watching point
  virtual void setAddressWatch(uint32_t numWatchPoints, void** watchAddress, uint64_t* watchMask,
                               uint64_t* watchMode, DebugEvent* event) = 0;

  //!  Map the shader (AQL code) for host access
  virtual void mapKernelCode(void* aqlCodeInfo) const = 0;

  //!  Get the packet information for dispatch
  virtual void getPacketAmdInfo(const void* aqlCodeInfo, void* packetInfo) const = 0;

  //!  Set global memory values
  virtual void setGlobalMemory(amd::Memory* memObj, uint32_t offset, void* srcPtr,
                               uint32_t size) = 0;

  //!  Execute the post-dispatch callback function
  virtual void executePostDispatchCallBack() = 0;

  //!  Execute the pre-dispatch callback function
  virtual void executePreDispatchCallBack(void* aqlPacket, void* toolInfo) = 0;

 protected:
  //!  Return the context
  const amd::Context* context() const { return context_; }

  //!  Get the debug device
  const amd::Device* device() const { return device_; }

  //!  Return the register flag
  bool isRegistered() const { return isRegistered_; }

 protected:
  const amd::Context* context_;  ///< context that used to create host queue for the debugger
  amd::Device* device_;          ///< Device to run the debugger

  cl_PreDispatchCallBackFunctionAMD preDispatchCallBackFunc_;  //!< pre-dispatch callback function
  cl_PostDispatchCallBackFunctionAMD
      postDispatchCallBackFunc_;    //!< post-dispatch callback function
  void* preDispatchCallBackArgs_;   //!< pre-dispatch callback function arguments
  void* postDispatchCallBackArgs_;  //!< post-dispatch callback function arguments

  DispatchDebugInfo debugInfo_;  //!< Debug setting/information for kernel dispatch
  amd::Memory* rtTrapInfo_[kDebugTrapLocationMax];  //!< Device trap buffer, to store various trap
                                                    //!handlers on the device

  amd::Memory** paramMemory_;  //!< list of memory pointers for kernel parameters
  uint32_t numParams_;         //!< number of kernel parameters

  void* aclBinary_;  //!< ACL binary

  address aqlCodeAddr_;   //!< The mapped AQL code to allow host access
  uint32_t aqlCodeSize_;  //!< The size of the AQL code info

  address scratchRingAddr_;   //!< The mapped address of the scratch buffer
  uint32_t scratchRingSize_;  //!< The size of the scratch ring

  bool isRegistered_;  //! flag to indicate the debugger has been registered

  cl_dbg_exception_policy_amd excpPolicy_;  //!< exception policy
  cl_dbg_kernel_exec_mode_amd execMode_;    //!< kernel execution mode
  RuntimeTrapInfo rtTrapHandlerInfo_;       //!< Runtime trap information

  //!  Runtime Trap handler pointer (TBA) & its buffer (TMA)
  device::Memory* runtimeTBA_;  //! runtime trap handler pointer
  device::Memory* runtimeTMA_;  //! runtime trap handler buffer
};


/**@}*/

/**
 * @}
 */
}  // namespace amd

#endif  // HWDEBUG_H_
