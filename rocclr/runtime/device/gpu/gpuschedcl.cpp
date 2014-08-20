//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

namespace gpu {

#define SCHEDULER_KERNEL(...) #__VA_ARGS__

const char* SchedulerSourceCode = SCHEDULER_KERNEL(
\n
//! AmdAqlWrap slot state
enum AqlWrapState {
    AQL_WRAP_FREE = 0,
    AQL_WRAP_RESERVED,
    AQL_WRAP_READY,
    AQL_WRAP_MARKER,
    AQL_WRAP_BUSY,
    AQL_WRAP_DONE
};

//! Profiling states
enum ProfilingState {
    PROFILING_COMMAND_START = 0,
    PROFILING_COMMAND_END,
    PROFILING_COMMAND_COMPLETE
};

typedef struct _HsaAqlDispatchPacket {
    uint    mix;
    ushort  workgroup_size[3];
    ushort  reserved2;
    uint    grid_size[3];
    uint    private_segment_size_bytes;
    uint    group_segment_size_bytes;
    ulong   kernel_object_address;
    ulong   kernel_arg_address;
    ulong   reserved3;
    ulong   completion_signal;
} HsaAqlDispatchPacket;

typedef struct _AmdVQueueHeader {
    uint    aql_slot_num;       //!< [LRO/SRO] The total number of the AQL slots (multiple of 64).
    uint    event_slot_num;     //!< [LRO] The number of kernel events in the events buffer
    ulong   event_slot_mask;    //!< [LRO] A pointer to the allocation bitmask array for the events
    ulong   event_slots;        //!< [LRO] Pointer to a buffer for the events.
                                // Array of event_slot_num entries of AmdEvent
    ulong   aql_slot_mask;      //!< [LRO/SRO]A pointer to the allocation bitmask for aql_warp slots
    uint    command_counter;    //!< [LRW] The global counter for the submitted commands into the queue
    uint    wait_size;          //!< [LRO] The wait list size (in clk_event_t)
    uint    arg_size;           //!< [LRO] The size of argument buffer (in bytes)
    uint    reserved0;          //!< For the future usage
    ulong   kernel_table;       //!< [LRO] Pointer to an array with all kernel objects (ulong for each entry)
    uint    reserved[2];        //!< For the future usage
} AmdVQueueHeader;

typedef struct _AmdAqlWrap {
    uint state;             //!< [LRW/SRW] The current state of the AQL wrapper:  FREE, RESERVED, READY,
                            // MARKER, BUSY and DONE. The block could be returned back to a free state.
    uint enqueue_flags;     //!< [LWO/SRO] Contains the flags for the kernel execution start –
        // (kernel_enqueue_flags_t)
        // CLK_ENQUEUE_FLAGS_NO_WAIT – we just start processing
        // CLK_ENQUEUE_FLAGS_WAIT_KERNEL – check if parent_wrap->state is done and then start processing
        // CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP - currently == WAIT_KERNEL
    uint command_id;        //!< [LWO/SRO] The unique command ID
    uint child_counter;     //!< [LRW/SRW] Counter that determine the launches of child kernels.
                            // It’s incremented on the
                            // start and decremented on the finish. The parent kernel can be considered as
                            // done when the value is 0 and the state is DONE
    ulong completion;       //!< [LWO/SRO] CL event for the current execution (clk_event_t)
    ulong parent_wrap;      //!< [LWO/SRO] Pointer to the parent AQL wrapper (AmdAqlWrap*)
    ulong wait_list;        //!< [LRO/SRO] Pointer to an array of clk_event_t objects (64 bytes default)
    uint wait_num;          //!<  [LWO/SRO] The number of cl_event_wait objects
    uint reserved[5];       //!< For the future usage
    HsaAqlDispatchPacket aql;  //!< [LWO/SRO] AQL packet – 64 bytes AQL packet
} AmdAqlWrap;

typedef struct _AmdEvent {
    uint state;             //!< [LRO/SRW] Event state: START, END, COMPLETE
    uint counter;           //!< [LRW] Event retain/release counter. 0 means the event is free
    ulong timer[3];         //!< [LRO/SWO] Timer values for profiling for each state
} AmdEvent;

typedef struct _SchedulerParam {
    uint    signal;         //!< Signal to stop the child queue
    uint    eng_clk;        //!< Engine clock in Mhz
    ulong   hw_queue;       //!< Address to HW queue
    ulong   hsa_queue;      //!< Address to HSA dummy queue
    uint    launch;         //!< Child launch status
    uint    scratchSize;    //!< Scratch buffer size
    ulong   scratch;        //!< GPU address to the scratch buffer
    uint    numMaxWaves;    //!< Num max waves on the asic
    uint    releaseHostCP;  //!< Releases CP on the host queue
    ulong   parentAQL;      //!< Host parent AmdAqlWrap packet
    uint    dedicatedQueue; //!< Scheduler uses a dedicated queue
    uint    scratchOffset;  //!< Scratch buffer offset
} SchedulerParam;

typedef struct _HwDispatch {
    uint    startExe;       // REWIND execution
    uint    condExe0;       // 0xC0032200 -- TYPE 3, COND_EXEC
    uint    condExe1;       // 0x00000204 ----
    uint    condExe2;       // 0x00000000 ----
    uint    condExe3;       // 0x00000000 ----
    uint    condExe4;       // 0x00000000 ----
    uint    packet0;        // 0xC0067602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (6 values)
    uint    offset0;        // 0x00000204 ---- OFFSET
    uint    startX;         // 0x00000000 ---- COMPUTE_START_X: START = 0x0
    uint    startY;         // 0x00000000 ---- COMPUTE_START_Y: START = 0x0
    uint    startZ;         // 0x00000000 ---- COMPUTE_START_Z: START = 0x0
    uint    wrkGrpSizeX;    // 0x00000000 ---- COMPUTE_NUM_THREAD_X: NUM_THREAD_FULL = 0x0, NUM_THREAD_PARTIAL = 0x0
    uint    wrkGrpSizeY;    // 0x00000000 ---- COMPUTE_NUM_THREAD_Y: NUM_THREAD_FULL = 0x0, NUM_THREAD_PARTIAL = 0x0
    uint    wrkGrpSizeZ;    // 0x00000000 ---- COMPUTE_NUM_THREAD_Z: NUM_THREAD_FULL = 0x0, NUM_THREAD_PARTIAL = 0x0
    uint    packet1;        // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
    uint    offset1;        // 0x0000020C ---- OFFSET
    uint    isaLo;          // 0x00000000 ---- COMPUTE_PGM_LO: DATA = 0x0
    uint    isaHi;          // 0x00000000 ---- COMPUTE_PGM_HI: DATA = 0x0, INST_ATC__CI__VI = 0x0
    uint    packet2;        // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
    uint    offset2;        // 0x00000212 ---- OFFSET
    uint    resource1;      // 0x00000000 ---- COMPUTE_PGM_RSRC1: VGPRS = 0x0, SGPRS = 0x0, PRIORITY = 0x0, FLOAT_MODE = 0x0, PRIV = 0x0, DX10_CLAMP = 0x0, DEBUG_MODE = 0x0, IEEE_MODE = 0x0, BULKY__CI__VI = 0x0, CDBG_USER__CI__VI = 0x0
    uint    resource2;      // 0x00000000 ---- COMPUTE_PGM_RSRC2: SCRATCH_EN = 0x0, USER_SGPR = 0x0, TRAP_PRESENT = 0x0, TGID_X_EN = 0x0, TGID_Y_EN = 0x0, TGID_Z_EN = 0x0, TG_SIZE_EN = 0x0, TIDIG_COMP_CNT = 0x0, EXCP_EN_MSB__CI__VI = 0x0, LDS_SIZE = 0x0, EXCP_EN = 0x0
    uint    packet3;        // 0xC0067602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (6 values)
    uint    offset3;        // 0x00000215 ---- OFFSET
    uint    pad31;          // 0x00000000 ---- COMPUTE_RESOURCE_LIMITS: WAVES_PER_SH = 0x0, TG_PER_CU = 0x0, LOCK_THRESHOLD = 0x0, SIMD_DEST_CNTL = 0x0, FORCE_SIMD_DIST__CI__VI = 0x0, CU_GROUP_COUNT__CI__VI = 0x0
    uint    pad32;          // 0xFFFFFFFF ---- COMPUTE_STATIC_THREAD_MGMT_SE0: SH0_CU_EN = 0xFFFF, SH1_CU_EN = 0xFFFF
    uint    pad33;          // 0xFFFFFFFF ---- COMPUTE_STATIC_THREAD_MGMT_SE1: SH0_CU_EN = 0xFFFF, SH1_CU_EN = 0xFFFF
    uint    ringSize;       // 0x00000000 ---- COMPUTE_TMPRING_SIZE: WAVES = 0x0, WAVESIZE = 0x0
    uint    pad34;          // 0xFFFFFFFF ---- COMPUTE_STATIC_THREAD_MGMT_SE2__CI__VI: SH0_CU_EN = 0xFFFF, SH1_CU_EN = 0xFFFF
    uint    pad35;          // 0xFFFFFFFF ---- COMPUTE_STATIC_THREAD_MGMT_SE3__CI__VI: SH0_CU_EN = 0xFFFF, SH1_CU_EN = 0xFFFF
    uint    user0;          // 0xC0047602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (4 values)
    uint    offsUser0;      // 0x00000240 ---- OFFSET
    uint    scratchLo;      // 0x00000000 ---- COMPUTE_USER_DATA_0: DATA = 0x0
    uint    scratchHi;      // 0x80000000 ---- COMPUTE_USER_DATA_1: DATA = 0x80000000
    uint    scratchSize;    // 0x00000000 ---- COMPUTE_USER_DATA_2: DATA = 0x0
    uint    padUser;        // 0x00EA7FAC ---- COMPUTE_USER_DATA_3: DATA = 0xEA7FAC
    uint    user1;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
    uint    offsUser1;      // 0x00000244 ---- OFFSET
    uint    aqlPtrLo;       // 0x00000000 ---- COMPUTE_USER_DATA_4: DATA = 0x0
    uint    aqlPtrHi;       // 0x00000000 ---- COMPUTE_USER_DATA_5: DATA = 0x0
    uint    user2;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
    uint    offsUser2;      // 0x00000246 ---- OFFSET
    uint    hsaQueueLo;     // 0x00000000 ---- COMPUTE_USER_DATA_6: DATA = 0x0
    uint    hsaQueueHi;     // 0x00000000 ---- COMPUTE_USER_DATA_7: DATA = 0x0
    uint    user3;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
    uint    offsUser3;      // 0x00000246 ---- OFFSET
    uint    argsLo;         // 0x00000000 ---- COMPUTE_USER_DATA_8: DATA = 0x0
    uint    argsHi;         // 0x00000000 ---- COMPUTE_USER_DATA_9: DATA = 0x0
    uint    copyData;       // 0xC0044000 -- TYPE 3, COPY_DATA
    uint    copyDataFlags;  // 0x00000405 ---- srcSel 0x5, destSel 0x4, countSel 0x0, wrConfirm 0x0, engineSel 0x0
    uint    scratchAddrLo;  // 0x000201C4 ---- srcAddressLo
    uint    scratchAddrHi;  // 0x00000000 ---- srcAddressHi
    uint    shPrivateLo;    // 0x00002580 ---- dstAddressLo
    uint    shPrivateHi;    // 0x00000000 ---- dstAddressHi
    uint    user4;          // 0xC0027602 -- TYPE 3, SET_SH_REG, TYPE:COMPUTE (2 values)
    uint    offsUser4;      // 0x00000248 ---- OFFSET
    uint    scratchOffs;    // 0x00000000 ---- COMPUTE_USER_DATA_10: DATA = 0x0
    uint    privSize;       // 0x00000030 ---- COMPUTE_USER_DATA_11: DATA = 0x30
    uint    packet4;        // 0xC0031502 -- TYPE 3, DISPATCH_DIRECT, TYPE:COMPUTE
    uint    glbSizeX;       // 0x00000000
    uint    glbSizeY;       // 0x00000000
    uint    glbSizeZ;       // 0x00000000
    uint    padd41;         // 0x00000021
} HwDispatch;

const uint ResumeExecution = 0x80000000; // 0x81000000
const uint StallExecution = 0x00000000; // 0x01000000
const uint WavefrontSize = 64;
const uint MaxWaveSize = 0x400;

static inline void
dispatch(
    volatile __global HwDispatch*   dispatch,
    __global HsaAqlDispatchPacket*  aqlPkt,
    ulong   scratch,
    ulong   hsaQueue,
    uint    scratchSize,
    uint    scratchOffset,
    uint    numMaxWaves)
{
    const uint UsrRegOffset = 0x240;
    const uint Pm4Nop = 0xC0001002;
    const uint Pm4UserRegs = 0xC0007602;
    const uint Pm4CopyReg = 0xC0044000;

    // Wait for CP idle isn't necessary if CP waits for child
    // while (atomic_and(&dispatch->startExe, 0xffffffff) != StallExecution) {}

    uint    usrRegCnt = 0;

    dispatch->wrkGrpSizeX   = aqlPkt->workgroup_size[0];
    dispatch->wrkGrpSizeY   = aqlPkt->workgroup_size[1];
    dispatch->wrkGrpSizeZ   = aqlPkt->workgroup_size[2];
    // ISA address
    __global uchar* kernelObj = (__global uchar*)aqlPkt->kernel_object_address;
    ulong   isa  = aqlPkt->kernel_object_address + *((__global uint*)(kernelObj + 0x10));
    dispatch->isaLo = (uint)(isa >> 8);
    dispatch->isaHi = (uint)(isa >> 40);

    // Program PGM resource registers
    dispatch->resource1 = *((__global uint*)(kernelObj + 0x30));
    dispatch->resource2 = *((__global uint*)(kernelObj + 0x34));
    uint    flags       = *((__global uint*)(kernelObj + 0x38));
    uint    privateSize = *((__global uint*)(kernelObj + 0x50));

    uint ldsSize = aqlPkt->group_segment_size_bytes +
        *((__global uint*)(kernelObj + 0x54));
    // Align up the LDS blocks 128 * 4(in DWORDs)
    uint ldsBlocks = (ldsSize + 511) >> 9;
    dispatch->resource2 |= (ldsBlocks << 15);

    // Workaround for compiler bug
    dispatch->scratchLo = (flags & 1);
    // privSegEna = (flags & 1);
    if (flags & 0x1) {
        uint    waveSize = privateSize * WavefrontSize;
        // 256 DWRODs is the minimum for SQ
        waveSize = max(MaxWaveSize, waveSize);
        uint    numWaves = scratchSize / waveSize;
        numWaves = min(numWaves, numMaxWaves);
        dispatch->ringSize = numWaves;
        dispatch->ringSize |= (waveSize >> 10) << 12;
        dispatch->user0 = Pm4UserRegs | (4 << 16);
        dispatch->scratchLo = (uint)scratch;
        dispatch->scratchHi = ((uint)(scratch >> 32)) | 0x80000000; // Enables swizzle
        dispatch->scratchSize = scratchSize;
        usrRegCnt += 4;
    }
    else {
        dispatch->ringSize = 0;
        dispatch->user0 = Pm4Nop | (4 << 16);
    }

    // dispatchEna = (flags & 0x2);
    dispatch->user1 = (flags & 0x2) ? (Pm4UserRegs | (2 << 16)) : (Pm4Nop | (2 << 16));
    dispatch->offsUser1 = UsrRegOffset + usrRegCnt;
    usrRegCnt += (flags & 0x2) ? 2 : 0;
    ulong  gpuAqlPtr = (ulong)aqlPkt;
    dispatch->aqlPtrLo = (uint)gpuAqlPtr;
    dispatch->aqlPtrHi = (uint)(gpuAqlPtr >> 32);

    // queuePtr = (flags & 0x4);
    if (flags & 0x4) {
        dispatch->user2 = Pm4UserRegs | (2 << 16);
        dispatch->offsUser2 = UsrRegOffset + usrRegCnt;
        usrRegCnt += 2;
        dispatch->hsaQueueLo = (uint)hsaQueue;
        dispatch->hsaQueueHi = (uint)(hsaQueue >> 32);
    }
    else {
        dispatch->user2 = Pm4Nop | (2 << 16);
    }

    // kernelArgEna = (flags & 0x8);
    dispatch->user3 = (flags & 0x8) ? (Pm4UserRegs | (2 << 16)) : (Pm4Nop | (2 << 16));
    dispatch->offsUser3 = UsrRegOffset + usrRegCnt;
    usrRegCnt += (flags & 0x8) ? 2 : 0;
    dispatch->argsLo = (uint)aqlPkt->kernel_arg_address;
    dispatch->argsHi = (uint)(aqlPkt->kernel_arg_address >> 32);

    // flatScratchEna = (flags & 0x20);
    if (flags & 0x20) {
        dispatch->copyData = Pm4CopyReg;
        dispatch->scratchAddrLo = (uint)((scratch - scratchOffset) >> 16);
        dispatch->offsUser4 = UsrRegOffset + usrRegCnt;
        dispatch->scratchOffs = scratchOffset;
        dispatch->privSize = privateSize;
    }
    else {
        dispatch->copyData = Pm4Nop | (8 << 16);
    }

    dispatch->glbSizeX  = aqlPkt->grid_size[0];
    dispatch->glbSizeY  = aqlPkt->grid_size[1];
    dispatch->glbSizeZ  = aqlPkt->grid_size[2];
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Resume the execution
    dispatch->startExe = ResumeExecution;
}

static inline bool
checkWaitEvents(__global AmdEvent** events, uint numEvents)
{
    for (uint i = 0; i < numEvents; ++i) {
        if (atomic_load_explicit(
            (__global atomic_uint*)(&events[i]->state),
            memory_order_acquire, memory_scope_device) != CL_COMPLETE) {
            return false;
        }
    }
    return true;
}


// release slot in a bitmask controlled resource i is the slot number
static inline void
release_slot(__global uint* restrict mask, uint i)
{
    /* uint b = ~(1UL << (i & 0x1f)); */
    uint b = ~amd_bfm(1U, i);
    __global atomic_uint *p = (__global atomic_uint *)(mask + (i >> 5));
    uint vv;
    uint v = atomic_load_explicit(p, memory_order_acquire, memory_scope_device);
    for (;;) {
        vv = v & b;
        if (atomic_compare_exchange_strong_explicit(p, &v, vv,
            memory_order_acq_rel, memory_order_acquire, memory_scope_device)) {
            break;
        }
    }
}

static inline void
releaseEvent(__global AmdEvent* ev, __global uint* emask, __global AmdEvent* eb)
{
    uint c = atomic_fetch_sub_explicit((__global atomic_uint *)&ev->counter, 1U,
        memory_order_acq_rel, memory_scope_device);
    if (c == 1U) {
        uint i = ev - eb;
        release_slot(emask, i);
    }
}

static inline void
releaseWaitEvents(__global AmdEvent** events, uint numEvents,
    __global uint* emask, __global AmdEvent* eb)
{
    for (uint i = 0; i < numEvents; ++i) {
        releaseEvent(events[i], emask, eb);
    }
}

static inline uint
min_command(uint slot_num, __global AmdAqlWrap* wraps)
{
    uint minCommand = 0xffffffff;
    for (uint idx = 0; idx < slot_num; ++idx) {
        __global AmdAqlWrap* disp = (__global AmdAqlWrap*)&wraps[idx];
        uint slotState = atomic_load_explicit((__global atomic_uint*)(&disp->state),
                memory_order_acquire, memory_scope_device);
        if ((slotState != AQL_WRAP_FREE) && (slotState != AQL_WRAP_RESERVED)) {
            minCommand = min(disp->command_id, minCommand);
        }
    }
    return minCommand;
}

extern ulong __hsail_get_clock();   // Declaration is required

__kernel void
scheduler(
    __global    AmdVQueueHeader*    queue,
    __global    SchedulerParam*     params,
    uint                            paramIdx)
{
    __global  SchedulerParam* param = &params[paramIdx];
    volatile __global HwDispatch* hwDisp =
            (volatile __global HwDispatch*)param->hw_queue;
    __global AmdAqlWrap*    hostParent = (__global AmdAqlWrap*)(param->parentAQL);
    __global uint*          counter = (__global uint*)(&hostParent->child_counter);
    __global uint*          signal = (__global uint*)(&param->signal);
    __global AmdAqlWrap*    wraps = (__global AmdAqlWrap*)&queue[1];
    __global uint*          amask = (__global uint *)queue->aql_slot_mask;

    //! @todo This is an unexplained behavior.
    //! The scheduler can be launched one more time after termination.
    if (1 == atomic_load_explicit((__global atomic_uint*)&param->releaseHostCP,
        memory_order_acquire, memory_scope_device)) {
        return;
    }

    uint launch = 0;
    uint loop = 1;

    do {
        uint mask = atomic_load_explicit((__global atomic_uint*)(&amask[get_group_id(0)]),
                memory_order_acquire, memory_scope_device);

        int baseIdx = get_group_id(0) * 32;
        while (mask != 0) {
            uint sIdx = ctz(mask);
            uint idx = baseIdx + sIdx;
            mask &= ~(1 << sIdx);
            __global AmdAqlWrap* disp = (__global AmdAqlWrap*)&wraps[idx];
            uint slotState = atomic_load_explicit((__global atomic_uint*)(&disp->state),
                memory_order_acquire, memory_scope_device);
            __global AmdAqlWrap* parent = (__global AmdAqlWrap*)(disp->parent_wrap);
            __global AmdEvent* event = (__global AmdEvent*)(disp->completion);

            // Check if the current slot is ready for processing
            if (slotState == AQL_WRAP_READY) {
                if (launch == 0) {
                    launch = atomic_load_explicit((__global atomic_uint*)&param->launch,
                        memory_order_acquire, memory_scope_device);
                }
                if (launch == 0) {
                    // Attempt to find a new disaptch if nothing was launched yet
                    uint parentState = atomic_load_explicit(
                        (__global atomic_uint*)(&parent->state),
                        memory_order_acquire, memory_scope_device);
                    uint enqueueFlags = atomic_load_explicit(
                        (__global atomic_uint*)(&disp->enqueue_flags),
                        memory_order_acquire, memory_scope_device);

                    // Check the launch flags
                    if (((enqueueFlags == CLK_ENQUEUE_FLAGS_WAIT_KERNEL) ||
                         (enqueueFlags == CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP)) &&
                        (parentState != AQL_WRAP_DONE)) {
                        continue;
                    }

                    // Check if the wait list is COMPLETE
                    launch = checkWaitEvents(
                        (__global AmdEvent**)(disp->wait_list), disp->wait_num);

                    if (launch == 0) continue;

                    uint tmp = 0;
                    if (atomic_compare_exchange_strong_explicit(
                        (__global atomic_uint*)&param->launch, &tmp, launch,
                        memory_order_acq_rel, memory_order_acquire, memory_scope_device)) {
                        if (event != 0) {
                            event->timer[PROFILING_COMMAND_START] =
                                (__hsail_get_clock() * (ulong)param->eng_clk) >> 10;
                        }
                        // Launch child kernel ....
                        dispatch(hwDisp, &disp->aql, param->scratch, param->hsa_queue,
                            param->scratchSize, param->scratchOffset, param->numMaxWaves);
                        disp->state = AQL_WRAP_BUSY;
                        releaseWaitEvents((__global AmdEvent**)(disp->wait_list),
                            disp->wait_num, (__global uint*)queue->event_slot_mask,
                            (__global AmdEvent*)queue->event_slots);
                        break;
                    }
                }
            }
            else if (slotState == AQL_WRAP_MARKER) {
                bool complete = false;
                if (disp->wait_num == 0) {
                    uint minCommand = min_command(queue->aql_slot_num, wraps);
                    if (disp->command_id == minCommand) {
                        complete = true;
                    }
                }
                else {
                    // Check if the wait list is COMPLETE
                    if (checkWaitEvents(
                        (__global AmdEvent**)(disp->wait_list), disp->wait_num)) {
                        complete = true;
                        releaseWaitEvents((__global AmdEvent**)(disp->wait_list),
                            disp->wait_num, (__global uint*)queue->event_slot_mask,
                            (__global AmdEvent*)queue->event_slots);
                    }
                }
                if (complete) {
                    // Decrement the child execution counter on the parent
                    atomic_fetch_sub_explicit(
                        (__global atomic_uint*)&parent->child_counter,
                        1, memory_order_acq_rel, memory_scope_device);
                    event->state = CL_COMPLETE;
                    disp->state = AQL_WRAP_FREE;
                    release_slot(amask, idx);
                    releaseEvent(event, (__global uint*)queue->event_slot_mask,
                        (__global AmdEvent*)queue->event_slots);
                }
            }
            else if (slotState == AQL_WRAP_DONE) {
                // Was CL_EVENT requested?
                if (event != 0) {
                    // The current dispatch doesn't have any outstanding children
                    if (disp->child_counter == 0) {
                        event->state = CL_COMPLETE;
                        event->timer[PROFILING_COMMAND_END] =
                        event->timer[PROFILING_COMMAND_COMPLETE] =
                            (__hsail_get_clock() * (ulong)param->eng_clk) >> 10;
                    }
                    else {
                        event->timer[PROFILING_COMMAND_END] =
                            (__hsail_get_clock() * (ulong)param->eng_clk) >> 10;
                    }
                    releaseEvent(event, (__global uint *)queue->event_slot_mask,
                        (__global AmdEvent *)queue->event_slots);
                }
                // The current dispatch doesn't have any outstanding children
                if (disp->child_counter == 0) {
                    // Decrement the child execution counter on the parent
                    atomic_fetch_sub_explicit(
                        (__global atomic_uint*)&parent->child_counter,
                        1, memory_order_acq_rel, memory_scope_device);
                    disp->state = AQL_WRAP_FREE;
                    release_slot(amask, idx);
                }
            }
            else if (slotState == AQL_WRAP_BUSY) {
                disp->state = AQL_WRAP_DONE;
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        launch = atomic_load_explicit((__global atomic_uint*)&param->launch,
            memory_order_acquire, memory_scope_device);

        if (param->dedicatedQueue) {
            loop = atomic_load_explicit((__global atomic_uint*)signal,
                memory_order_acquire, memory_scope_device);
        }
        else {
            loop = atomic_load_explicit((__global atomic_uint*)counter,
                memory_order_acquire, memory_scope_device);
        }

    } while ((launch == 0) && (loop != 0));

    if (loop == 0) {
        //! \todo Write deadcode to the template, but somehow
        //! the scheduler will be launched one more time.
        hwDisp->packet0 = 0xdeadc0de;
        hwDisp[1].condExe0 = 0xdeadc0de;
        hwDisp[1].condExe1 = 0xdeadc0de;
        hwDisp[1].condExe2 = 0xdeadc0de;
        hwDisp[1].condExe3 = 0xdeadc0de;
        atomic_store_explicit((__global atomic_uint*)signal,
            0, memory_order_release, memory_scope_device);
        barrier(CLK_GLOBAL_MEM_FENCE);
        atomic_store_explicit((__global atomic_uint*)&hwDisp->startExe,
            ResumeExecution, memory_order_release, memory_scope_device);
        atomic_store_explicit((__global atomic_uint*)&param->releaseHostCP,
            1, memory_order_release, memory_scope_device);
    }
}
\n
\n
);

} // namespace gpu
