//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//
#pragma once

#include "device/pal/palvirtual.hpp"
#include "device/pal/paldebugger.hpp"

namespace pal {

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
    cl_int waitDebugEvent(DebugEvent pEvent, uint32_t timeOut) const;

    //!  Destroy the debug event
    void destroyDebugEvent(DebugEvent* pEvent);

    //!  Register the debugger
    cl_int registerDebugger(amd::Context*context, uintptr_t messageStorage);

    //!  Unregister the debugger
    void unregisterDebugger();

    //!  Send the wavefront control cmmand
    void wavefrontControl(uint32_t waveAction,
                            uint32_t waveMode,
                            uint32_t trapId,
                            void*  waveAddr) const;

    //!  Set address watching point
    void setAddressWatch(uint32_t numWatchPoints,
                           void** watchAddress,
                           uint64_t* watchMask,
                           uint64_t* watchMode,
                           DebugEvent* pEvent);

    //!  Map the kernel code for host access
    void mapKernelCode(void* aqlCodeInfo) const;

    //!  Get the packet information for dispatch
    void getPacketAmdInfo(const void* aqlCodeInfo, void* packetInfo) const;

    //!  Set global memory values
    void setGlobalMemory(amd::Memory* memObj, uint32_t offset, void* srcPtr, uint32_t size);

    //!  Execute the post-dispatch callback function
    void executePostDispatchCallBack();

    //!  Execute the pre-dispatch callback function
    void executePreDispatchCallBack(void*   aqlPacket,
                                    void*   toolInfo);

protected:
    const VirtualGPU*    vGpu() const { return vGpu_; }

private:
    //!  Setup trap handler info for kernel execution
    void setupTrapInformation(DebugToolInfo* toolInfo);

    //!  Create runtime trap handler
    cl_int createRuntimeTrapHandler();

    const pal::Device*   device() const {
        return reinterpret_cast<const pal::Device *>(device_); }

    VirtualGPU*         vGpu_;              //!< the virtual GPU
    uintptr_t           debugMessages_;     //!< Pointer to a SHARED_DEBUG_MESSAGES pass to the KMD
    HwDbgAddressWatch*  addressWatch_;      //!< Address watch data
    size_t              addressWatchSize_;  //!< Size of address watch data
    //!  Arguments used by the callback function
    void*                                 oclEventHandle_;     //!< event handler
    const hsa_kernel_dispatch_packet_t*   aqlPacket_;          //!< AQL packet
};

}  // namespace pal
