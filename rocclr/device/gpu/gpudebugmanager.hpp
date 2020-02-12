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

#ifndef HWDBG_DEBUGMANAGER_H__
#define HWDBG_DEBUGMANAGER_H__

#include "gpuvirtual.hpp"
#include "gpudebugger.hpp"

namespace gpu {

class GpuDebugManager;
class Device;
class Memory;


/*!  \brief Debug Manager Class
 *
 *    The debug manager class is used to pass all the trap info to the
 *    kernel dispatch and then the kernel execution can use such trap information
 *    for kernel execution. This class contains the trap handler and shader event
 *    objects. The trap handler is setup by users and passed to the kernel dispatch.
 *    The shader event is to receive interrupts from the GPU and then users can
 *    perform various operations.
 *
 *    This class also provides the interface for setting up the pre-dispatch
 *    callback functions used by the profiler and debugger. It also provides
 *    a way to retrieve various debug information for the kernel execution.
 *
 */
class GpuDebugManager : public amd::HwDebugManager {
 public:
  //!  Constructor of the debug manager class
  GpuDebugManager(amd::Device* device);

  //!  Destructor of the debug manager class
  ~GpuDebugManager();

  //!  Get the single instance of the GpuDebugManager class
  static GpuDebugManager* getDefaultInstance();

  //!  Destroy the GpuDebugManager class object
  static void destroyInstances();

  //!  Flush cache
  void flushCache(uint32_t mask);

  //!  Create the debug event
  DebugEvent createDebugEvent(const bool autoReset);

  //!  Wait for the debug event
  int32_t waitDebugEvent(DebugEvent pEvent, uint32_t timeOut) const;

  //!  Destroy the debug event
  void destroyDebugEvent(DebugEvent* pEvent);

  //!  Register the debugger
  int32_t registerDebugger(amd::Context* context, uintptr_t messageStorage);

  //!  Unregister the debugger
  void unregisterDebugger();

  //!  Send the wavefront control cmmand
  void wavefrontControl(uint32_t waveAction, uint32_t waveMode, uint32_t trapId,
                        void* waveAddr) const;

  //!  Set address watching point
  void setAddressWatch(uint32_t numWatchPoints, void** watchAddress, uint64_t* watchMask,
                       uint64_t* watchMode, DebugEvent* pEvent);

  //!  Map the kernel code for host access
  void mapKernelCode(void* aqlCodeInfo) const;

  //!  Get the packet information for dispatch
  void getPacketAmdInfo(const void* aqlCodeInfo, void* packetInfo) const;

  //!  Set global memory values
  void setGlobalMemory(amd::Memory* memObj, uint32_t offset, void* srcPtr, uint32_t size);

  //!  Execute the post-dispatch callback function
  void executePostDispatchCallBack();

  //!  Execute the pre-dispatch callback function
  void executePreDispatchCallBack(void* aqlPacket, void* toolInfo);

 private:
  //!  Setup trap handler info for kernel execution
  void setupTrapInformation(DebugToolInfo* toolInfo);

  //!  Create runtime trap handler
  int32_t createRuntimeTrapHandler();

 protected:
  const VirtualGPU* vGpu() const { return vGpu_; }

 private:
  const gpu::Device* device() const { return reinterpret_cast<const gpu::Device*>(device_); }

  VirtualGPU* vGpu_;  //!< the virtual GPU

  uintptr_t debugMessages_;  //!< Pointer to a SHARED_DEBUG_MESSAGES pass to the KMD

  HwDbgAddressWatch* addressWatch_;  //!< Address watch data
  size_t addressWatchSize_;          //!< Size of address watch data

  //!  Arguments used by the callback function
  void* oclEventHandle_;                           //!< event handler
  const hsa_kernel_dispatch_packet_t* aqlPacket_;  //!< AQL packet
};

}  // namespace gpu

#endif  // HWDBG_DEBUGMANAGER_H__
