/* ========================================================================
   Copyright (c) 2014 Advanced Micro Devices, Inc.  All rights reserved.
  ========================================================================*/

#ifndef HWDEBUG_H_
#define HWDEBUG_H_

#include "amdocl/cl_debugger_amd.h"

#define TBA_START_OFFSET 256

/**
 *******************************************************************************
 * @brief Debug information required by the AMD debugger
 *        This might have to be moved to a private header. We could provide
 *        these services as a seperate dll.
 * @details The information is populated by the function oclGetDebugInfo
 *******************************************************************************
 */
struct PacketAmdInfo
{
    uint32_t trapReservedVgprIndex;   //!< reserved VGPR index, -1 when they are not valid
    uint32_t scratchBufferWaveOffset; //!< scratch buffer wave offset, -1 when no scratch buffer
    void *pointerToIsaBuffer;         //!< pointer to the buffer containing ISA
    size_t sizeOfIsaBuffer;           //!< size of the ISA buffer
    uint32_t numberOfVgprs;           //!< number of VGPRs used by the kernel
    uint32_t numberOfSgprs;           //!< number of SGPRs used by the kernel
    size_t sizeOfStaticGroupMemory;   //!< Static local memory used by the kernel
};

//! Cache mask for invalidation
struct HwDbgGpuCacheMask
{
    union {
        struct {
            uint32_t sqICache   : 1;    //!< Instruction cache
            uint32_t sqKCache   : 1;    //!< Data cache
            uint32_t tcL1       : 1;    //!< tcL1 cache
            uint32_t tcL2       : 1;    //!< tcL2 cache
            uint32_t reserved   : 28;
        };
        uint32_t ui32All;
    };
};



/**
 * Opaque pointer to trap event
 */
typedef uint64_t DebugEvent;        //! opaque pointer to trap event

namespace amd {

/*! \class HwDebugManager
 *
 *  \brief The device interface class for the hardware debug manager
 */
class HwDebugManager
{
public:

    //! Constructor for the Hardware Debug Manager
    HwDebugManager() : isRegistered_(false), useHwDebug_(false) {}

    //! Destructor for Hardware Debug Manager
    ~HwDebugManager() {};

    //!  Setup the call back function pointer
    virtual void setCallBackFunctions(cl_PreDispatchCallBackFunctionAMD  preDispatchFn,
                                      cl_PostDispatchCallBackFunctionAMD postDispatchFn) = 0;

    //!  Setup the call back argument pointers
    virtual void setCallBackArguments(void *preDispatchArgs, void *postDispatchArgs) = 0;

    //!  Flush cache
    virtual cl_int flushCache(uint32_t mask) = 0;

    //!  Set exception policy
    virtual cl_int setExceptionPolicy(void *policy) = 0;

    //!  Get exception policy
    virtual cl_int getExceptionPolicy(void *policy) const = 0;

    //!  Set the kernel execution mode
    virtual cl_int setKernelExecutionMode(void *mode) = 0;

    //!  Get the kernel execution mode
    virtual cl_int getKernelExecutionMode(void *mode) const = 0;

    //!  Create the debug event
    virtual DebugEvent createDebugEvent(const bool autoReset) = 0;

    //!  Wait for the debug event
    virtual cl_int waitDebugEvent(DebugEvent pEvent, uint32_t  timeOut) const = 0;

    //!  Destroy the debug event
    virtual cl_int destroyDebugEvent(DebugEvent pEvent) = 0;

    //!  Register the debugger
    virtual cl_int registerDebugger(amd::Context *context, uintptr_t pMessageStorage) = 0;

    //!  Call KMD to register the debugger
    virtual cl_int registerDebuggerOnQueue(device::VirtualDevice *vDevice) = 0;

    //!  Unregister the debugger
    virtual cl_int unregisterDebugger() = 0;

    //!  Setup the pointer to the aclBinary within the debug manager
    virtual void setAclBinary(void *aclBinary) = 0;

    //!  Send the wavefront control cmmand
    virtual cl_int wavefrontControl(uint32_t waveAction,
                                    uint32_t waveMode,
                                    uint32_t trapId,
                                    void * waveAddr) const = 0;

    //!  Set address watching point
    virtual cl_int setAddressWatch(uint32_t numWatchPoints,
                                   void ** watchAddress,
                                   uint64_t * watchMask,
                                   uint64_t * watchMode,
                                   DebugEvent * event) = 0;

    //!  Get the packet information for dispatch
    virtual cl_int getPacketAmdInfo(const void * aqlCodeInfo,
                                    void * packetInfo) const = 0;

    //!  Get dispatch debug info
    virtual cl_int getDispatchDebugInfo(void * debugInfo) const = 0;

    //!  Map the AQL code for host access
    virtual cl_int mapKernelCode(uint64_t *aqlCode, uint32_t *aqlCodeSize) const = 0;

    //!  Map the scratch ring for host access
    virtual cl_int mapScratchRing(uint64_t *scratchRingAddr, uint32_t *scratchRingSize) const = 0;

    //!  Set global memory values
    virtual cl_int setGlobalMemory(void * memObj,
                                   uint32_t offset,
                                   void * srcPtr,
                                   uint32_t size) = 0;

    //!  Set kernel parameter memory object list
    virtual cl_int setKernelParamMemList(void ** paramMem, uint32_t numParams) = 0;

    //!  Get kernel parameter memory object
    virtual uint64_t getKernelParamMem(uint32_t paramIdx) const = 0;

    //!  Set the kernel code address and its size
    virtual void setKernelCodeInfo(address aqlCodeAddr, uint32_t aqlCodeSize) = 0;

    //!  Get the scratch ring
    virtual void setScratchRing(address scratchRingAddr, uint32_t scratchRingSize) = 0;

    //!  Retrieve the pre-dispatch callback function
    virtual cl_PreDispatchCallBackFunctionAMD getPreDispatchCallBackFunction() const = 0;

    //!  Retrieve the post-dispatch callback function
    virtual void * getPreDispatchCallBackArguments() const = 0;

    //!  Retrieve the pre-dispatch callback function arguments
    virtual cl_PostDispatchCallBackFunctionAMD getPostDispatchCallBackFunction() const = 0;

    //!  Retrieve the post-dispatch callback function arguments
    virtual void * getPostDispatchCallBackArguments() const = 0;

    //!  Set the register flag
    void setRegisterFlag(bool regFlag) { isRegistered_ = regFlag; }

    //!  Set the use of HW DEBUG flag
    void setUseHwDebugFlag(bool flag) { useHwDebug_ = flag; }

    //!  Return the register flag
    bool isRegistered() const { return isRegistered_; }

    //!  Return the use of HW DEBUG flag
    bool useHwDebug() const { return useHwDebug_; }


protected:
    bool isRegistered_;     //! flag to indicate the debugger has been registered
    bool useHwDebug_;       //! flag to indicate the HW DEBUG is using
};


/**@}*/

/**
 * @}
 */
} // namespace amd

#endif  // HWDEBUG_H_
