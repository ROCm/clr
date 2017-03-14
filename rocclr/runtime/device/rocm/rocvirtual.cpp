//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocblit.hpp"
#include "platform/kernel.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"
#include "SCHSAInterface.h"
#include "amd_hsa_kernel_code.h"

#include <fstream>
#include <vector>
#include <string>

/**
* HSA image object size in bytes (see HSAIL spec)
*/
#define HSA_IMAGE_OBJECT_SIZE 48

/**
* HSA image object alignment in bytes (see HSAIL spec)
*/
#define HSA_IMAGE_OBJECT_ALIGNMENT  16

/**
* HSA sampler object size in bytes (see HSAIL spec)
*/
#define HSA_SAMPLER_OBJECT_SIZE 32

/**
* HSA sampler object alignment in bytes (see HSAIL spec)
*/
#define HSA_SAMPLER_OBJECT_ALIGNMENT  16

namespace roc {
// (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) invalidates I, K and L1
// (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE) invalidates L1, L2 and flushes L2

static const uint16_t kDispatchPacketHeaderNoSync =
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kDispatchPacketHeader =
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kBarrierPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kBarrierPacketAcquireHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const uint16_t kBarrierPacketReleaseHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

static const hsa_barrier_and_packet_t kBarrierAcquirePacket =
    {kBarrierPacketAcquireHeader,0,0,0,0,0,0,0,0,0};

static const hsa_barrier_and_packet_t kBarrierReleasePacket =
    {kBarrierPacketReleaseHeader,0,0,0,0,0,0,0,0,0};

double Timestamp::ticksToTime_=0;

/**
* Set the ocl correlation handle (essentially the cl_event handle)
* to correlate the cl kernel launch and HSA kernel dispatch
*/
typedef hsa_status_t
  (*hsa_ext_tools_set_correlation_handle)(const hsa_agent_t agent,
                                        void *correlation_handle);
static void SetOclCorrelationHandle(void *tools_lib, const hsa_agent_t agent, void *handle) {
  hsa_ext_tools_set_correlation_handle func =
    (hsa_ext_tools_set_correlation_handle)amd::Os::getSymbol(tools_lib, "hsa_ext_tools_set_correlation_handler");
  if (func) {
    func(agent, handle);
  }

  return;
}

bool
VirtualGPU::MemoryDependency::create(size_t numMemObj)
{
    if (numMemObj > 0) {
        // Allocate the array of memory objects for dependency tracking
        memObjectsInQueue_ = new MemoryState[numMemObj];
        if (nullptr == memObjectsInQueue_) {
            return false;
        }
        memset(memObjectsInQueue_, 0, sizeof(MemoryState) * numMemObj);
        maxMemObjectsInQueue_ = numMemObj;
    }

    return true;
}

void
VirtualGPU::MemoryDependency::validate(
    VirtualGPU&     gpu,
    const Memory*   memory,
    bool            readOnly)
{
    bool    flushL1Cache = false;

    if (maxMemObjectsInQueue_ == 0) {
        // Sync AQL packets
        gpu.setAqlHeader(kDispatchPacketHeader);
        return;
    }

    uint64_t curStart = reinterpret_cast<uint64_t>(memory->getDeviceMemory());
    uint64_t curEnd = curStart + memory->size();

    // Loop through all memory objects in the queue and find dependency
    // @note don't include objects from the current kernel
    for (size_t j = 0; j < endMemObjectsInQueue_; ++j) {
        // Check if the queue already contains this mem object and
        // GPU operations aren't readonly
        uint64_t busyStart = memObjectsInQueue_[j].start_;
        uint64_t busyEnd = memObjectsInQueue_[j].end_;

        // Check if the start inside the busy region
        if ((((curStart >= busyStart) && (curStart < busyEnd)) ||
            // Check if the end inside the busy region
             ((curEnd > busyStart) && (curEnd <= busyEnd)) ||
            // Check if the start/end cover the busy region
             ((curStart <= busyStart) && (curEnd >= busyEnd))) &&
            // If the buys region was written or the current one is for write
            (!memObjectsInQueue_[j].readOnly_ || !readOnly)) {
            flushL1Cache = true;
            break;
        }
    }

    // Did we reach the limit?
    if (maxMemObjectsInQueue_ <= (numMemObjectsInQueue_ + 1)) {
        flushL1Cache = true;
    }

    if (flushL1Cache) {
        // Sync AQL packets
        gpu.setAqlHeader(kDispatchPacketHeader);

        // Clear memory dependency state
        const static bool All = true;
        clear(!All);
    }

    // Insert current memory object into the queue always,
    // since runtime calls flush before kernel execution and it has to keep
    // current kernel in tracking
    memObjectsInQueue_
        [numMemObjectsInQueue_].start_ = curStart;
    memObjectsInQueue_
        [numMemObjectsInQueue_].end_ = curEnd;
    memObjectsInQueue_
        [numMemObjectsInQueue_].readOnly_ = readOnly;
    numMemObjectsInQueue_++;
}

void
VirtualGPU::MemoryDependency::clear(bool all)
{
    if (numMemObjectsInQueue_ > 0) {
        size_t  i, j;
        if (all) {
            endMemObjectsInQueue_ = numMemObjectsInQueue_;
        }

        // Preserve all objects from the current kernel
        for (i = 0, j = endMemObjectsInQueue_; j < numMemObjectsInQueue_; i++, j++) {
            memObjectsInQueue_[i].start_ = memObjectsInQueue_[j].start_;
            memObjectsInQueue_[i].end_ = memObjectsInQueue_[j].end_;
            memObjectsInQueue_[i].readOnly_ = memObjectsInQueue_[j].readOnly_;
        }
        // Clear all objects except current kernel
        memset(&memObjectsInQueue_[i], 0, sizeof(amd::Memory*) * numMemObjectsInQueue_);
        numMemObjectsInQueue_ -= endMemObjectsInQueue_;
        endMemObjectsInQueue_ = 0;
    }
}

bool
VirtualGPU::processMemObjects(
    const amd::Kernel&  kernel,
    const_address       params)
{
    static const bool NoAlias = true;
    const Kernel& hsaKernel = static_cast<const Kernel&>
        (*(kernel.getDeviceKernel(dev(), NoAlias)));
    const amd::KernelSignature& signature = kernel.signature();
    const amd::KernelParameters& kernelParams = kernel.parameters();

    // AQL packets
    setAqlHeader(kDispatchPacketHeaderNoSync);

    // Mark the tracker with a new kernel,
    // so we can avoid checks of the aliased objects
    memoryDependency().newKernel();

    bool deviceSupportFGS = 0 != dev().isFineGrainedSystem(true);
    bool supportFineGrainedSystem = deviceSupportFGS;
    FGSStatus status = kernelParams.getSvmSystemPointersSupport();
    switch (status) {
        case FGS_YES:
            if (!deviceSupportFGS) {
                return false;
            }
            supportFineGrainedSystem = true;
            break;
        case FGS_NO:
            supportFineGrainedSystem = false;
            break;
        case FGS_DEFAULT:
        default:
            break;
    }

    size_t count = kernelParams.getNumberOfSvmPtr();
    size_t execInfoOffset = kernelParams.getExecInfoOffset();
    bool sync = true;

    amd::Memory* memory = nullptr;
    //get svm non arugment information
    void* const* svmPtrArray =
        reinterpret_cast<void* const*>(params + execInfoOffset);
    for (size_t i = 0; i < count; i++) {
        memory =  amd::SvmManager::FindSvmBuffer(svmPtrArray[i]);
        if (nullptr == memory) {
            if (!supportFineGrainedSystem) {
                return false;
            }
            else if (sync) {
                // Sync AQL packets
                setAqlHeader(kDispatchPacketHeader);
                // Clear memory dependency state
                const static bool All = true;
                memoryDependency().clear(!All);
                continue;
            }
        }
        else {
            Memory* rocMemory = static_cast<Memory*>(memory->getDeviceMemory(dev()));
            if (nullptr != rocMemory) {
                // Synchronize data with other memory instances if necessary
                rocMemory->syncCacheFromHost(*this);

                const static bool IsReadOnly = false;
                // Validate SVM passed in the non argument list
                memoryDependency().validate(*this, rocMemory, IsReadOnly);
            }
            else {
                return false;
            }
        }
    }

    // Check all parameters for the current kernel
    for (size_t i = 0; i < signature.numParameters(); ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        const Kernel::Argument*  arg = hsaKernel.hsailArgAt(i);
        Memory* memory = nullptr;
        bool    readOnly = false;
        amd::Memory* svmMem = nullptr;

        // Find if current argument is a buffer
        if ((desc.type_ == T_POINTER) && (arg->addrQual_ != ROC_ADDRESS_LOCAL)) {
            if (kernelParams.boundToSvmPointer(dev(), params, i)) {
                svmMem = amd::SvmManager::FindSvmBuffer(
                    *reinterpret_cast<void* const*>(params + desc.offset_));
                if (!svmMem) {
                    // Sync AQL packets
                    setAqlHeader(kDispatchPacketHeader);
                    // Clear memory dependency state
                    const static bool All = true;
                    memoryDependency().clear(!All);
                    continue;
                }
            }

            if (*reinterpret_cast<amd::Memory* const*>
                    (params + desc.offset_) != nullptr) {
                if (nullptr == svmMem) {
                    memory = static_cast<Memory*>((*reinterpret_cast<amd::Memory* const*>
                            (params + desc.offset_))->getDeviceMemory(dev()));
                }
                else {
                    memory = static_cast<Memory*>(svmMem->getDeviceMemory(dev()));
                }
                // Don't sync for internal objects,
                // since they are not shared between devices
                if (memory->owner()->getVirtualDevice() == nullptr) {
                    // Synchronize data with other memory instances if necessary
                    memory->syncCacheFromHost(*this);
                }
            }

            if (memory != nullptr) {
                readOnly |= (arg->access_ == ROC_ACCESS_TYPE_RO);
                // Validate memory for a dependency in the queue
                memoryDependency().validate(*this, memory, readOnly);
            }
        }
    }

    if (hsaKernel.program()->hasGlobalStores()) {
        // Sync AQL packets
        setAqlHeader(kDispatchPacketHeader);
        // Clear memory dependency state
        const static bool All = true;
        memoryDependency().clear(!All);
    }

    return true;
}

template<typename AqlPacket>
bool
VirtualGPU::dispatchGenericAqlPacket(
    AqlPacket* packet,
    bool blocking)
{
    const uint32_t queueSize = gpu_queue_->size;
    const uint32_t queueMask = queueSize - 1;

    //Check for queue full and wait if needed.
    uint64_t index = hsa_queue_load_write_index_relaxed(gpu_queue_);
    uint64_t read = hsa_queue_load_read_index_relaxed(gpu_queue_);
    hsa_signal_t signal;

    // TODO: placeholder to setup the kernel to populate start and end timestamp.
    if (timestamp_ != nullptr) {
        // Find signal slot
        ProfilingSignal* profilingSignal = &signal_pool_[index & queueMask];
        // Make sure we save the old results in the TS structure
        if (profilingSignal->ts_ != nullptr) {
            profilingSignal->ts_->checkGpuTime();
        }
        // Update the new TS with the signal info
        timestamp_->setProfilingSignal(profilingSignal);
        packet->completion_signal = profilingSignal->signal_;
        profilingSignal->ts_ = timestamp_;
        timestamp_->setAgent(gpu_device_);
    }

    if (blocking || (index - read) == queueMask) {
        if (packet->completion_signal.handle == 0) {
            packet->completion_signal = barrier_signal_;
        }
        signal = packet->completion_signal;
        // Initialize signal for a wait
        hsa_signal_store_relaxed(signal, InitSignalValue);
        blocking = true;
    }

    //Insert packet
    ((AqlPacket*)(
        gpu_queue_->base_address))[index & queueMask] = *packet;
    hsa_queue_store_write_index_release(gpu_queue_, index + 1);
    hsa_signal_store_relaxed(gpu_queue_->doorbell_signal, index);

    //Wait on signal ?
    if (blocking) {
        if (hsa_signal_wait_acquire(
            signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
            HSA_WAIT_STATE_BLOCKED) != 0) {
            LogPrintfError("Failed signal [0x%lx] wait", signal.handle);
            return false;
        }

        // Release the pool, since runtime just drained the entire queue
        resetKernArgPool();
    }

    return true;
}

bool
VirtualGPU::dispatchAqlPacket(
    hsa_kernel_dispatch_packet_t* packet,
    bool blocking)
{
  return dispatchGenericAqlPacket(packet, blocking);
}

bool
VirtualGPU::dispatchAqlPacket(
    hsa_barrier_and_packet_t* packet,
    bool blocking)
{
  return dispatchGenericAqlPacket(packet, blocking);
}

void
VirtualGPU::dispatchBarrierPacket(const hsa_barrier_and_packet_t* packet)
{
    assert(packet->completion_signal.handle != 0);
    const uint32_t queueSize = gpu_queue_->size;
    const uint32_t queueMask = queueSize - 1;

    uint64_t index = hsa_queue_load_write_index_relaxed(gpu_queue_);
    ((hsa_barrier_and_packet_t*)(
        gpu_queue_->base_address))[index&queueMask] = *packet;

    hsa_queue_store_write_index_relaxed(gpu_queue_, index + 1);

    hsa_signal_store_relaxed(gpu_queue_->doorbell_signal, index);
}

/**
 * @brief Waits on an outstanding kernel without regard to how
 * it was dispatched - with or without a signal
 *
 * @return bool true if Wait returned successfully, false
 * otherwise
 */
bool VirtualGPU::releaseGpuMemoryFence() {
    // Return if there is no pending dispatch
    if (!hasPendingDispatch_) {
      return false;
    }

    // Initialize signal for the barrier packet.
    hsa_signal_store_relaxed(barrier_signal_, InitSignalValue);

    // Dispatch barrier packet into the queue and wait till it finishes.
    dispatchBarrierPacket(&barrier_packet_);
    if (hsa_signal_wait_acquire(
        barrier_signal_, HSA_SIGNAL_CONDITION_EQ, 0, uint64_t(-1),
        HSA_WAIT_STATE_BLOCKED) != 0) {
        LogError("Barrier packet submission failed");
        return false;
    }

    hasPendingDispatch_ = false;

    // Release all transfer buffers on this command queue
    releaseXferWrite();

    // Release all memory dependencies
    memoryDependency().clear();

    // Release the pool, since runtime just completed a barrier
    resetKernArgPool();

    return true;
}

VirtualGPU::VirtualGPU(Device &device)
    : device::VirtualDevice(device)
    , roc_device_(device)
    , index_(device.numOfVgpus_++) // Virtual gpu unique index incrementing
{
    gpu_device_ = device.getBackendDevice();
    printfdbg_ = nullptr;

    // Initialize the last signal and dispatch flags
    timestamp_ = nullptr;
    hasPendingDispatch_ = false;
    tools_lib_ = nullptr;

    kernarg_pool_base_ = nullptr;
    kernarg_pool_size_ = 0;
    kernarg_pool_cur_offset_ = 0;
    aqlHeader_ = kDispatchPacketHeaderNoSync;
    barrier_signal_.handle = 0;
}

VirtualGPU::~VirtualGPU()
{
    releasePinnedMem();

    if (timestamp_ != nullptr) {
        delete timestamp_;
        timestamp_ = nullptr;
        LogError("There was a timestamp that was not used; deleting.");
    }
    if (printfdbg_ != nullptr){
        delete printfdbg_;
        printfdbg_ = nullptr;
    }

    tools_lib_ = nullptr;
    --roc_device_.numOfVgpus_; // Virtual gpu unique index decrementing
}

bool
VirtualGPU::create(bool profilingEna)
{
    // Set the event handle to the tools lib if the env var
    // Load the library using its advertised "soname"
    std::string lib_name = amd::Os::getEnvironment("HSA_TOOLS_LIB");
    if (lib_name != "") {
#if defined(_WIN32) || defined(__CYGWIN__)
        const char *tools_lib_name = "hsa-runtime-tools" LP64_SWITCH("", "64") ".dll";
#else
        const char *tools_lib_name = "libhsa-runtime-tools" LP64_SWITCH("", "64") ".so.1";
#endif
        tools_lib_ = amd::Os::loadLibrary(tools_lib_name);
    }

    // Checking Virtual gpu unique index for ROCm backend
    if (index() > device().settings().commandQueues_) {
        return false;
    }

    uint32_t queue_max_packets = 0;
    if (HSA_STATUS_SUCCESS !=
        hsa_agent_get_info(
        gpu_device_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max_packets)) {
        return false;
    }

    //Pick a reasonable queue size
    uint32_t queue_size = 1024;
    queue_size = (queue_max_packets < queue_size) ? queue_max_packets : queue_size;
    while (hsa_queue_create(gpu_device_,
        queue_size, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, UINT_MAX, UINT_MAX,
        &gpu_queue_) != HSA_STATUS_SUCCESS) {
        queue_size >>= 1;
        if (queue_size < 64) {
            return false;
        }
    }

    if (!initPool(dev().settings().kernargPoolSize_, (profilingEna) ? queue_size : 0)) {
        LogError("Couldn't allocate arguments/signals for the queue");
        return false;
    }

    device::BlitManager::Setup  blitSetup;
    blitMgr_ = new KernelBlitManager(*this, blitSetup);
    if ((nullptr == blitMgr_) || !blitMgr_->create(roc_device_)) {
        LogError("Could not create BlitManager!");
        return false;
    }

    // Create signal for the barrier packet.
    hsa_signal_t signal = { 0 };
    if (HSA_STATUS_SUCCESS !=
        hsa_signal_create(InitSignalValue, 0, nullptr, &signal)) {
        return false;
    }
    barrier_signal_ = signal;

    // Initialize barrier packet.
    memset(&barrier_packet_, 0, sizeof(barrier_packet_));
    barrier_packet_.header = kBarrierPacketHeader;
    barrier_packet_.completion_signal = barrier_signal_;

    // Create a object of PrintfDbg
    printfdbg_ = new PrintfDbg(roc_device_);
    if (nullptr == printfdbg_) {
        LogError("\nCould not create printfDbg Object!");
        return false;
    }

    // Initialize timestamp conversion factor
    if (Timestamp::getGpuTicksToTime() == 0) {
        uint64_t frequency;
        hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
        Timestamp::setGpuTicksToTime(1e9/double(frequency));
    }

    if (!memoryDependency().create(GPU_NUM_MEM_DEPENDENCY)) {
        LogError("Could not create the array of memory objects!");
        return false;
    }

    return true;
}

bool
VirtualGPU::terminate()
{
    delete blitMgr_;

    // Release the resources of signal
    releaseGpuMemoryFence();
    hsa_status_t err = hsa_queue_destroy(gpu_queue_);
    if (err !=  HSA_STATUS_SUCCESS) {
        return false;
    }

    if (barrier_signal_.handle != 0) {
        hsa_signal_destroy(barrier_signal_);
    }

    if (tools_lib_) {
        amd::Os::unloadLibrary(tools_lib_);
        tools_lib_ = nullptr;
    }

    destroyPool();

    return true;
}

bool
VirtualGPU::initPool(size_t kernarg_pool_size, uint signal_pool_count)
{
    kernarg_pool_size_ = kernarg_pool_size;
    kernarg_pool_base_ = reinterpret_cast<char *>(
        roc_device_.hostAlloc(kernarg_pool_size_, 1, true));
    if (kernarg_pool_base_ == nullptr) {
        return false;
    }

    if (signal_pool_count != 0) {
        signal_pool_.resize(signal_pool_count);
        for (uint i = 0; i < signal_pool_count; ++i) {
            ProfilingSignal profilingSignal;
            if (HSA_STATUS_SUCCESS != hsa_signal_create(
                0, 0, nullptr, &profilingSignal.signal_)) {
                return false;
            }
            signal_pool_[i] = profilingSignal;
        }
    }

    return true;
}

void
VirtualGPU::destroyPool() {
    if (kernarg_pool_base_ != nullptr) {
        roc_device_.hostFree(kernarg_pool_base_, kernarg_pool_size_);
    }

    if (signal_pool_.size() > 0) {
        for (uint i = 0; i < signal_pool_.size(); ++i) {
            hsa_signal_destroy(signal_pool_[i].signal_);
        }
    }
}

void*
VirtualGPU::allocKernArg(size_t size, size_t alignment)
{
    char* result = nullptr;
    do {
        result = amd::alignUp(kernarg_pool_base_ + kernarg_pool_cur_offset_, alignment);
        const size_t pool_new_usage = (result + size) - kernarg_pool_base_;
        if (pool_new_usage <= kernarg_pool_size_) {
            kernarg_pool_cur_offset_ = pool_new_usage;
            return result;
        }
        else {
            //! We run out of the arguments space!
            //! That means the app didn't call clFlush/clFinish for very long time.
            //! We can issue a barrier to avoid expensive extra memory allocations.

            // Initialize signal for the barrier packet.
            hsa_signal_store_relaxed(barrier_signal_, InitSignalValue);

            // Dispatch barrier packet into the queue and wait till it finishes.
            dispatchBarrierPacket(&barrier_packet_);
            if (hsa_signal_wait_acquire(
                barrier_signal_, HSA_SIGNAL_CONDITION_EQ, 0, uint64_t(-1),
                HSA_WAIT_STATE_BLOCKED) != 0) {
                LogError("Kernel arguments reset failed");
            }

            resetKernArgPool();
        }
    } while (true);

    return result;
}

/* profilingBegin, when profiling is enabled, creates a timestamp to save in
* virtualgpu's timestamp_, and calls start() to get the current host
* timestamp.
*/
void VirtualGPU::profilingBegin(amd::Command &command, bool drmProfiling)
{
    if (command.profilingInfo().enabled_) {
        if (timestamp_ != nullptr) {
            LogWarning("Trying to create a second timestamp in VirtualGPU. \
                        This could have unintended consequences.");
            return;
        }
        timestamp_ = new Timestamp;
        timestamp_->start();
    }
}

/* profilingEnd, when profiling is enabled, checks to see if a signal was
* created for whatever command we are running and calls end() to get the
* current host timestamp if no signal is available. It then saves the pointer
* timestamp_ to the command's data.
*/
void VirtualGPU::profilingEnd(amd::Command &command)
{
    if (command.profilingInfo().enabled_) {
        if (timestamp_->getProfilingSignal() == nullptr) {
            timestamp_->end();
        }
        command.setData(reinterpret_cast<void*>(timestamp_));
        timestamp_ = nullptr;
    }
}

struct DestroySampler : public std::binary_function<hsa_ext_sampler_t,
                                                    hsa_agent_t, bool> {
    bool operator() (hsa_ext_sampler_t &sampler,
                     hsa_agent_t agent) const {
        hsa_status_t status = hsa_ext_sampler_destroy(agent, sampler);
        return status == HSA_STATUS_SUCCESS;
    }
};

void VirtualGPU::updateCommandsState(amd::Command *list)
{
    Timestamp *ts = nullptr;

    amd::Command* current = list;
    amd::Command* next = nullptr;

    if (current == nullptr) {
        return;
    }

    uint64_t endTimeStamp = 0;
    uint64_t startTimeStamp = endTimeStamp;

    if (current->profilingInfo().enabled_) {
        // TODO: use GPU timestamp when available.
        endTimeStamp = amd::Os::timeNanos();
        startTimeStamp = endTimeStamp;

        // This block gets the first valid timestamp from the first command
        // that has one. This timestamp is used below to mark any command that
        // came before it to start and end with this first valid start time.
        current = list;
        while (current != nullptr) {
            if (current->data() != nullptr) {
                ts = reinterpret_cast<Timestamp*>(current->data());
                    startTimeStamp = ts->getStart();
                    endTimeStamp = ts->getStart();
                break;
            }
            current = current->getNext();
        }
    }

    // Iterate through the list of commands, and set timestamps as appropriate
    // Note, if a command does not have a timestamp, it does one of two things:
    // - if the command (without a timestamp), A, precedes another command, C,
    // that _does_ contain a valid timestamp, command A will set RUNNING and
    // COMPLETE with the RUNNING (start) timestamp from command C. This would
    // also be true for command B, which is between A and C. These timestamps
    // are actually retrieved in the block above (startTimeStamp, endTimeStamp).
    // - if the command (without a timestamp), C, follows another command, A,
    // that has a valid timestamp, command C will be set RUNNING and COMPLETE
    // with the COMPLETE (end) timestamp of the previous command, A. This is
    // also true for any command B, which falls between A and C.
    current = list;
    while (current != nullptr) {
        if (current->profilingInfo().enabled_) {
            if (current->data() != nullptr) {
                // Since this is a valid command to get a timestamp, we use the
                // timestamp provided by the runtime (saved in the data())
                ts = reinterpret_cast<Timestamp*>(current->data());
                    startTimeStamp = ts->getStart();
                    endTimeStamp = ts->getEnd();
                delete ts;
                current->setData(nullptr);
            }
            else {
                // If we don't have a command that contains a valid timestamp,
                // we simply use the end timestamp of the previous command.
                // Note, if this is a command before the first valid timestamp,
                // this will be equal to the start timestamp of the first valid
                // timestamp at this point.
                startTimeStamp = endTimeStamp;
            }
        }

        if (current->status() == CL_SUBMITTED) {
            current->setStatus(CL_RUNNING, startTimeStamp);
            current->setStatus(CL_COMPLETE, endTimeStamp);
        }
        else if (current->status() != CL_COMPLETE) {
            LogPrintfError("Unexpected command status - %d.", current->status());
        }

        next = current->getNext();
        current->release();
        current = next;
    }

    // Release the sampler handles allocated for the various
    // on one or more kernel submissions
    std::for_each(samplerList_.begin(),
      samplerList_.end(),
      std::bind2nd<DestroySampler>(DestroySampler(), gpu_device_));
    samplerList_.clear();

    return;
}

void VirtualGPU::submitReadMemory(amd::ReadMemoryCommand &cmd)
{
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    profilingBegin(cmd);

    size_t offset = 0;
    // Find if virtual address is a CL allocation
    device::Memory* hostMemory = dev().findMemoryFromVA(cmd.destination(), &offset);

    Memory* devMem = dev().getRocMemory(&cmd.source());
    // Synchronize data with other memory instances if necessary
    devMem->syncCacheFromHost(*this);

    void *dst = cmd.destination();
    amd::Coord3D size = cmd.size();

    //! @todo: add multi-devices synchronization when supported.

    cl_command_type type = cmd.type();
    bool result = false;
    bool imageBuffer = false;

    // Force buffer read for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_READ_IMAGE) &&
        (cmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        type = CL_COMMAND_READ_BUFFER;
        imageBuffer = true;
    }

    switch (type) {
        case CL_COMMAND_READ_BUFFER: {
            amd::Coord3D    origin(cmd.origin()[0]);
            if (imageBuffer) {
                size_t  elemSize =
                  cmd.source().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0] *= elemSize;
            }
            if (hostMemory != nullptr) {
                // Accelerated transfer without pinning
                amd::Coord3D dstOrigin(offset);
                result = blitMgr().copyBuffer(*devMem, *hostMemory,
                    origin, dstOrigin, size, cmd.isEntireMemory());
            }
            else {
                result = blitMgr().readBuffer(
                        *devMem, dst, origin, size,
                        cmd.isEntireMemory());
            }
            break;
        }
        case CL_COMMAND_READ_BUFFER_RECT: {
            result = blitMgr().readBufferRect(
                        *devMem, dst, cmd.bufRect(), cmd.hostRect(), size,
                        cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_READ_IMAGE: {
            result = blitMgr().readImage(
              *devMem, dst, cmd.origin(), size, cmd.rowPitch(),
              cmd.slicePitch(), cmd.isEntireMemory());
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    if (!result) {
        LogError("submitReadMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }

    profilingEnd(cmd);
}

void VirtualGPU::submitWriteMemory(amd::WriteMemoryCommand &cmd)
{
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    profilingBegin(cmd);

    size_t offset = 0;
    // Find if virtual address is a CL allocation
    device::Memory* hostMemory = dev().findMemoryFromVA(cmd.source(), &offset);

    Memory* devMem = dev().getRocMemory(&cmd.destination());

    // Synchronize memory from host if necessary
    device::Memory::SyncFlags syncFlags;
    syncFlags.skipEntire_ = cmd.isEntireMemory();
    devMem->syncCacheFromHost(*this, syncFlags);

    const char* src = static_cast<const char*>(cmd.source());
    amd::Coord3D size = cmd.size();

    //! @todo add multi-devices synchronization when supported.

    cl_command_type type = cmd.type();
    bool result = false;
    bool imageBuffer = false;

    // Force buffer write for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_WRITE_IMAGE) &&
        (cmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        type = CL_COMMAND_WRITE_BUFFER;
        imageBuffer = true;
    }

    switch (type) {
        case CL_COMMAND_WRITE_BUFFER: {
            amd::Coord3D    origin(cmd.origin()[0]);
            if (imageBuffer) {
                size_t  elemSize =
                  cmd.destination().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0] *= elemSize;
            }
            if (hostMemory != nullptr) {
                // Accelerated transfer without pinning
                amd::Coord3D srcOrigin(offset);
                result = blitMgr().copyBuffer(*hostMemory, *devMem,
                    srcOrigin, origin, size, cmd.isEntireMemory());
            }
            else {
                result = blitMgr().writeBuffer(
                        src, *devMem , origin, size,
                        cmd.isEntireMemory());
            }
            break;
        }
        case CL_COMMAND_WRITE_BUFFER_RECT: {
            result = blitMgr().writeBufferRect(
                        src, *devMem, cmd.hostRect(), cmd.bufRect(), size,
                        cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_WRITE_IMAGE: {
            result = blitMgr().writeImage(
              src, *devMem, cmd.origin(), size, cmd.rowPitch(),
              cmd.slicePitch(), cmd.isEntireMemory());
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    if (!result) {
        LogError("submitWriteMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }
    else {
        cmd.destination().signalWrite(&dev());
    }

    profilingEnd(cmd);
}

void VirtualGPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd)
{
    // in-order semantics: previous commands need to be done before we start
    releaseGpuMemoryFence();

    profilingBegin(cmd);
    const std::vector<void*>& svmPointers = cmd.svmPointers();
    if (cmd.pfnFreeFunc() == nullptr) {
        // pointers allocated using clSVMAlloc
        for (cl_uint i = 0; i < svmPointers.size(); i++) {
            amd::SvmBuffer::free(cmd.context(), svmPointers[i]);
        }
    }
    else {
        cmd.pfnFreeFunc()(as_cl(cmd.queue()->asCommandQueue()), svmPointers.size(),
                (void**) (&(svmPointers[0])), cmd.userData());
    }
    profilingEnd(cmd);
}

void VirtualGPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd)
{
    // in-order semantics: previous commands need to be done before we start
    releaseGpuMemoryFence();
    profilingBegin(cmd);
    amd::SvmBuffer::memFill(cmd.dst(), cmd.src(), cmd.srcSize(), 1);
    profilingEnd(cmd);
}

void VirtualGPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd)
{
    // in-order semantics: previous commands need to be done before we start
    releaseGpuMemoryFence();
    profilingBegin(cmd);
    amd::SvmBuffer::memFill(cmd.dst(), cmd.pattern(), cmd.patternSize(), cmd.times());
    profilingEnd(cmd);
}

void VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand &cmd)
{
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    profilingBegin(cmd);

    Memory* srcDevMem = dev().getRocMemory(&cmd.source());
    Memory* dstDevMem = dev().getRocMemory(&cmd.destination());

    // Synchronize source and destination memory
    device::Memory::SyncFlags syncFlags;
    syncFlags.skipEntire_ = cmd.isEntireMemory();
    dstDevMem->syncCacheFromHost(*this, syncFlags);
    srcDevMem->syncCacheFromHost(*this);

    amd::Coord3D size = cmd.size();

    cl_command_type type = cmd.type();
    bool result = false;
    bool srcImageBuffer = false;
    bool dstImageBuffer = false;

    // Force buffer copy for IMAGE1D_BUFFER
    if (cmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        srcImageBuffer = true;
        type = CL_COMMAND_COPY_BUFFER;
    }
    if (cmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        dstImageBuffer = true;
        type = CL_COMMAND_COPY_BUFFER;
    }

    switch (cmd.type()) {
        case CL_COMMAND_COPY_BUFFER: {
            amd::Coord3D    srcOrigin(cmd.srcOrigin()[0]);
            amd::Coord3D    dstOrigin(cmd.dstOrigin()[0]);

            if (srcImageBuffer) {
                const size_t elemSize =
                    cmd.source().asImage()->getImageFormat().getElementSize();
                srcOrigin.c[0] *= elemSize;
                if (dstImageBuffer) {
                    dstOrigin.c[0] *= elemSize;
                }
                size.c[0] *= elemSize;
            }
            else if (dstImageBuffer) {
                const size_t elemSize =
                    cmd.destination().asImage()->getImageFormat().getElementSize();
                dstOrigin.c[0] *= elemSize;
                size.c[0] *= elemSize;
            }

            result = blitMgr().copyBuffer(
                        *srcDevMem, *dstDevMem, srcOrigin,
                        dstOrigin, size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_BUFFER_RECT: {
            result = blitMgr().copyBufferRect(
                        *srcDevMem, *dstDevMem, cmd.srcRect(),
                        cmd.dstRect(), size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_IMAGE: {
            result = blitMgr().copyImage(
              *srcDevMem, *dstDevMem, cmd.srcOrigin(),
              cmd.dstOrigin(), size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
            result = blitMgr().copyImageToBuffer(
              *srcDevMem, *dstDevMem, cmd.srcOrigin(),
              cmd.dstOrigin(), size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
            result = blitMgr().copyBufferToImage(
              *srcDevMem, *dstDevMem, cmd.srcOrigin(),
              cmd.dstOrigin(), size, cmd.isEntireMemory());
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    if (!result) {
        LogError("submitCopyMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }

    cmd.destination().signalWrite(&dev());

    profilingEnd(cmd);
}

void VirtualGPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd)
{
    // No fence is needed since this is a no-op: the
    // command will be completed only after all the
    // previous commands are complete
    profilingBegin(cmd);
    profilingEnd(cmd);
}

void VirtualGPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd)
{
    // No fence is needed since this is a no-op: the
    // command will be completed only after all the
    // previous commands are complete
    profilingBegin(cmd);
    profilingEnd(cmd);
}

void VirtualGPU::submitMapMemory(amd::MapMemoryCommand &cmd)
{
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    profilingBegin(cmd);

    //! @todo add multi-devices synchronization when supported.

    roc::Memory* devMemory = reinterpret_cast<roc::Memory *>(
        cmd.memory().getDeviceMemory(dev(), false));

    cl_command_type type = cmd.type();
    bool imageBuffer = false;

    // Save map requirement.
    cl_map_flags mapFlag = cmd.mapFlags();

    // Treat no map flag as read-write.
    if (mapFlag == 0) {
        mapFlag = CL_MAP_READ | CL_MAP_WRITE;
    }

    devMemory->saveMapInfo(cmd.mapPtr(), cmd.origin(), cmd.size(),
        mapFlag, cmd.isEntireMemory());

    // Sync to the map target.
    // If we have host memory, use it
    if (devMemory->owner()->getHostMem() != nullptr) {
        // Target is the backing store, so just ensure that owner is up-to-date
        devMemory->owner()->cacheWriteBack();

        if (devMemory->isHostMemDirectAccess()) {
            // Add memory to VA cache, so rutnime can detect direct access to VA
            dev().addVACache(devMemory);
        }
    }
    else if (mapFlag & (CL_MAP_READ | CL_MAP_WRITE)) {
        bool result = false;
        roc::Memory *hsaMemory = static_cast<roc::Memory *>(devMemory);

        amd::Memory* mapMemory = hsaMemory->mapMemory();
        void *hostPtr = mapMemory == nullptr ?
            hsaMemory->owner()->getHostMem() :
            mapMemory->getHostMem();

        if (type == CL_COMMAND_MAP_BUFFER) {
            amd::Coord3D    origin(cmd.origin()[0]);
            amd::Coord3D    size(cmd.size()[0]);
            amd::Coord3D    dstOrigin(cmd.origin()[0], 0, 0);
            if (imageBuffer) {
                size_t  elemSize =
                  cmd.memory().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0] *= elemSize;
            }

            if (mapMemory != nullptr) {
                roc::Memory *hsaMapMemory = static_cast<roc::Memory *>(
                    mapMemory->getDeviceMemory(dev(), false));
                result = blitMgr().copyBuffer(*hsaMemory,
                    *hsaMapMemory, origin, dstOrigin,
                    size, cmd.isEntireMemory());
            }
            else {
                result = blitMgr().readBuffer(
                    *hsaMemory, static_cast<char *>(hostPtr)+origin[0],
                    origin, size, cmd.isEntireMemory());
            }
        }
        else if (type == CL_COMMAND_MAP_IMAGE) {
            amd::Image* image = cmd.memory().asImage();
            if (mapMemory != nullptr) {
                roc::Memory *mapMemory = static_cast<roc::Memory *>(
                    devMemory->mapMemory()->getDeviceMemory(dev(), false));
                result = blitMgr().copyImageToBuffer(
                  *hsaMemory, *mapMemory, cmd.origin(),
                  amd::Coord3D(0,0,0), cmd.size(), cmd.isEntireMemory());
            }
            else {
                result = blitMgr().readImage(
                    *hsaMemory, hostPtr, amd::Coord3D(0),
                    image->getRegion(), image->getRowPitch(),
                    image->getSlicePitch(), true);
            }
        }
        else {
            ShouldNotReachHere();
        }

        if (!result) {
            LogError("submitMapMemory failed!");
            cmd.setStatus(CL_OUT_OF_RESOURCES);
        }
    }

    profilingEnd(cmd);
}

void VirtualGPU::submitUnmapMemory(amd::UnmapMemoryCommand &cmd)
{
    roc::Memory* devMemory = static_cast<roc::Memory*>(
        cmd.memory().getDeviceMemory(dev(), false));

    const device::Memory::WriteMapInfo* mapInfo =
        devMemory->writeMapInfo(cmd.mapPtr());
    if (nullptr == mapInfo) {
        LogError("Unmap without map call");
        return;
    }
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();
    profilingBegin(cmd);

    // Force buffer write for IMAGE1D_BUFFER
    bool imageBuffer = (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER);

    // We used host memory
    if (devMemory->owner()->getHostMem() != nullptr) {
        if (mapInfo->isUnmapWrite()) {
            // Target is the backing store, so sync
            devMemory->owner()->signalWrite(nullptr);
            devMemory->syncCacheFromHost(*this);
        }
        if (devMemory->isHostMemDirectAccess()) {
            // Remove memory from VA cache
            dev().removeVACache(devMemory);
        }
    }
    else if (mapInfo->isUnmapWrite()) {
        // Commit the changes made by the user.
        if (!devMemory->isHostMemDirectAccess()) {
            bool result = false;

            if (cmd.memory().asImage() && !imageBuffer) {
                amd::Image *image = cmd.memory().asImage();
                amd::Memory* mapMemory = devMemory->mapMemory();
                if (devMemory->mapMemory() != nullptr) {
                    roc::Memory *mapMemory = static_cast<roc::Memory *>(
                        devMemory->mapMemory()->getDeviceMemory(dev(), false));
                    result = blitMgr().copyBufferToImage(
                      *mapMemory, *devMemory, amd::Coord3D(0,0,0),
                      mapInfo->origin_, mapInfo->region_, mapInfo->isEntire());
                }
                else {
                    void *hostPtr = devMemory->owner()->getHostMem();

                    result = blitMgr().writeImage(
                        hostPtr, *devMemory,
                        amd::Coord3D(0), image->getRegion(),
                        image->getRowPitch(), image->getSlicePitch(), true);
                }
            }
            else {
                amd::Coord3D origin(mapInfo->origin_[0]);
                amd::Coord3D size(mapInfo->region_[0]);
                if (imageBuffer) {
                    size_t  elemSize =
                        cmd.memory().asImage()->getImageFormat().getElementSize();
                    origin.c[0] *= elemSize;
                    size.c[0] *= elemSize;
                }
                if (devMemory->mapMemory() != nullptr) {
                    roc::Memory *mapMemory = static_cast<roc::Memory *>(
                        devMemory->mapMemory()->getDeviceMemory(dev(), false));

                    result = blitMgr().copyBuffer(
                        *mapMemory, *devMemory,
                        mapInfo->origin_,
                        mapInfo->origin_,
                        mapInfo->region_,
                        mapInfo->isEntire());
                }
                else {
                    result = blitMgr().writeBuffer(
                        cmd.mapPtr(), *devMemory, origin, size);
                }
            }
            if (!result) {
                LogError("submitMapMemory failed!");
                cmd.setStatus(CL_OUT_OF_RESOURCES);
            }
        }

        cmd.memory().signalWrite(&dev());
    }

    devMemory->clearUnmapInfo(cmd.mapPtr());

    profilingEnd(cmd);
}

void VirtualGPU::submitFillMemory(amd::FillMemoryCommand &cmd)
{
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    profilingBegin(cmd);

    Memory* memory = dev().getRocMemory(&cmd.memory());

    bool    entire = cmd.isEntireMemory();
    // Synchronize memory from host if necessary
    device::Memory::SyncFlags syncFlags;
    syncFlags.skipEntire_ = entire;
    memory->syncCacheFromHost(*this, syncFlags);

    cl_command_type type = cmd.type();
    bool result = false;
    bool imageBuffer = false;
    float fillValue[4];

    // Force fill buffer for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_FILL_IMAGE) &&
        (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        type = CL_COMMAND_FILL_BUFFER;
        imageBuffer = true;
    }

    // Find the the right fill operation
    switch (type) {
        case CL_COMMAND_FILL_BUFFER: {
            const void* pattern = cmd.pattern();
            size_t  patternSize = cmd.patternSize();
            amd::Coord3D    origin(cmd.origin()[0]);
            amd::Coord3D    size(cmd.size()[0]);
            // Reprogram fill parameters if it's an IMAGE1D_BUFFER object
            if (imageBuffer) {
                size_t  elemSize =
                    cmd.memory().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0] *= elemSize;
                memset(fillValue, 0, sizeof(fillValue));
                cmd.memory().asImage()->getImageFormat().formatColor(pattern,
                                                                     fillValue);
                pattern = fillValue;
                patternSize = elemSize;
            }
            result = blitMgr().fillBuffer(
                *memory, pattern, patternSize, origin, size, entire);
            break;
        }
        case CL_COMMAND_FILL_IMAGE: {
            result = blitMgr().fillImage(
              *memory, cmd.pattern(), cmd.origin(), cmd.size(), entire);
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    if (!result) {
        LogError("submitFillMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }

    cmd.memory().signalWrite(&dev());

    profilingEnd(cmd);
}

void VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand &vcmd)
{
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    profilingBegin(vcmd);

    for (auto itr : vcmd.memObjects()) {
        // Find device memory
        Memory* memory = dev().getRocMemory(&(*itr));

        if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_HOST) {
            memory->mgpuCacheWriteBack();
        }
        else if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) {
            // Synchronize memory from host if necessary.
            // The sync function will perform memory migration from
            // another device if necessary
            device::Memory::SyncFlags syncFlags;
            memory->syncCacheFromHost(*this, syncFlags);
        }
        else {
            LogWarning("Unknown operation for memory migration!");
        }
    }

    profilingEnd(vcmd);
}

/*! \brief Writes to the buffer and increments the write pointer to the
 *         buffer. Also, ensures that the argument is written to an
 *         aligned memory as specified. Return the new write pointer.
 *
 * @param dst The write pointer to the buffer
 * @param src The source pointer
 * @param size The size in bytes to copy
 * @param alignment The alignment to follow while writing to the buffer
 */
static inline address
addArg(address dst, const void* src, size_t size, uint32_t alignment)
{
    dst = amd::alignUp(dst, alignment);
    ::memcpy(dst, src, size);
    return dst + size;
}

static inline address
addArg(address dst, const void* src, size_t size)
{
    assert(size < UINT32_MAX);
    return addArg(dst, src, size, size);
}

 //Over rides the workgroup size fields in the packet with runtime/compiler set sizes
 void setRuntimeCompilerLocalSize(hsa_kernel_dispatch_packet_t& dispatchPacket,
     amd::NDRangeContainer sizes,
     const size_t* compile_size,
     const roc::Device &dev){
     //Todo (sramalin) need to check if compile_size is set to 0 if dimension is not valid
    // else this error check is incorrect
    if (compile_size[0] || compile_size[1] || compile_size[2]) {
        dispatchPacket.workgroup_size_x = sizes.dimensions()>0 ? compile_size[0] : 1;
        dispatchPacket.workgroup_size_y = sizes.dimensions()>1 ? compile_size[1] : 1;
        dispatchPacket.workgroup_size_z = sizes.dimensions()>2 ? compile_size[2] : 1;
    }
    else {
        //Runtime must set the group size
        dispatchPacket.workgroup_size_x = 1;
        dispatchPacket.workgroup_size_y = 1;
        dispatchPacket.workgroup_size_z = 1;

       if (sizes.dimensions() == 1) {
            dispatchPacket.workgroup_size_x = dev.settings().maxWorkGroupSize_;
        }
        else if (sizes.dimensions() == 2) {
            dispatchPacket.workgroup_size_x = dev.settings().maxWorkGroupSize2DX_;
            dispatchPacket.workgroup_size_y = dev.settings().maxWorkGroupSize2DY_;
        }
        else if (sizes.dimensions() == 3) {
            dispatchPacket.workgroup_size_x = dev.settings().maxWorkGroupSize3DX_;
            dispatchPacket.workgroup_size_y = dev.settings().maxWorkGroupSize3DY_;
            dispatchPacket.workgroup_size_z = dev.settings().maxWorkGroupSize3DZ_;
        }
    }
}

 static void
fillSampleDescriptor(
    hsa_ext_sampler_descriptor_t& samplerDescriptor,
   const amd::Sampler& sampler)
 {
     samplerDescriptor.filter_mode = sampler.filterMode() == CL_FILTER_NEAREST ?
     HSA_EXT_SAMPLER_FILTER_MODE_NEAREST : HSA_EXT_SAMPLER_FILTER_MODE_LINEAR;
     samplerDescriptor.coordinate_mode = sampler.normalizedCoords() ?
        HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED :
        HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED;
     switch (sampler.addressingMode()) {
     case CL_ADDRESS_CLAMP_TO_EDGE:
         samplerDescriptor.address_mode =
            HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
         break;
     case CL_ADDRESS_REPEAT:
         samplerDescriptor.address_mode =
            HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT;
         break;
     case CL_ADDRESS_CLAMP:
         samplerDescriptor.address_mode =
            HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER;
         break;
     case CL_ADDRESS_MIRRORED_REPEAT:
         samplerDescriptor.address_mode =
            HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
         break;
     case CL_ADDRESS_NONE:
         samplerDescriptor.address_mode =
            HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED;
         break;
     default:
        return;
     }
   }

bool
VirtualGPU::submitKernelInternal(
    const amd::NDRangeContainer& sizes,
    const amd::Kernel& kernel,
    const_address parameters,
    void *eventHandle)
{
    if (tools_lib_) {
        SetOclCorrelationHandle(tools_lib_, this->gpu_device_, eventHandle);
    }

    device::Kernel *devKernel = const_cast<device::Kernel*>(
        kernel.getDeviceKernel(dev()));
    Kernel &gpuKernel = static_cast<Kernel &>(*devKernel);

    const size_t compilerLdsUsage = gpuKernel.WorkgroupGroupSegmentByteSize();
    size_t ldsUsage = compilerLdsUsage;

    // Check memory dependency and SVM objects
    if (!processMemObjects(kernel, parameters)) {
        LogError("Wrong memory objects!");
        return false;
    }

    // Init PrintfDbg object if printf is enabled.
    bool printfEnabled = (gpuKernel.printfInfo().size() > 0) ? true : false;
    if (!printfDbg()->init(printfEnabled)){
        LogError("\nPrintfDbg object initialization failed!");
        return false;
    }

    const amd::KernelSignature& signature = kernel.signature();
    const amd::KernelParameters& kernelParams = kernel.parameters();

    size_t newOffset[3] = {0, 0, 0};
    size_t newGlobalSize[3] = {0, 0, 0};

    int dim = -1;
    int iteration = 1;
    size_t globalStep = 0;
    for (uint i = 0; i < sizes.dimensions(); i++) {
        newGlobalSize[i] = sizes.global()[i];
        newOffset[i] = sizes.offset()[i];
    }

    if (gpuKernel.isInternalKernel()) {
        // Calculate new group size for each submission
        for (uint i = 0; i < sizes.dimensions(); i++) {
            if (sizes.global()[i] > static_cast<size_t>(0xffffffff)) {
                dim = i;
                iteration = sizes.global()[i] / 0xC0000000
                            + ((sizes.global()[i] % 0xC0000000) ? 1: 0);
                globalStep = (sizes.global()[i] / sizes.local()[i]) / iteration
                             * sizes.local()[dim];
                if (timestamp_ != nullptr) {
                    timestamp_->setSplittedDispatch();
                }
                break;
            }
        }
    }

    for (int j = 0; j < iteration; j++) {
        // Reset global size for dimension dim if split is needed
        if (dim != -1) {
            newOffset[dim] = sizes.offset()[dim] + globalStep * j;
            if (((newOffset[dim] + globalStep) < sizes.global()[dim]) &&
                (j != (iteration - 1))) {
                newGlobalSize[dim] = globalStep;
            }
            else {
                newGlobalSize[dim] = sizes.global()[dim] - newOffset[dim];
            }
        }

        // Find all parameters for the current kernel

        // Allocate buffer to hold kernel arguments
        address argBuffer = (address)allocKernArg(
            gpuKernel.KernargSegmentByteSize(),
            gpuKernel.KernargSegmentAlignment());

        if (argBuffer == nullptr) {
            LogError("Out of memory");
            return false;
        }

        address argPtr = argBuffer;
        for (auto arg : gpuKernel.hsailArgs()) {
            const_address   srcArgPtr = nullptr;
            if (arg->index_ != uint(-1)) {
                srcArgPtr = parameters + signature.at(arg->index_).offset_;
            }

            // Handle the hidden arguments first, as they do not have a
            // matching parameter in the OCL signature (not a valid arg->index_)
            switch (arg->type_) {
            case ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_X: {
                size_t offset_x = sizes.dimensions() >= 1 ? newOffset[0] : 0;
                assert(arg->size_ == sizeof(offset_x) && "check the sizes");
                argPtr = addArg(argPtr, &offset_x, arg->size_, arg->alignment_);
                break;
            }
            case ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Y: {
                size_t offset_y = sizes.dimensions() >= 2 ? newOffset[1] : 0;
                assert(arg->size_ == sizeof(offset_y) && "check the sizes");
                argPtr = addArg(argPtr, &offset_y, arg->size_, arg->alignment_);
                break;
            }
            case ROC_ARGTYPE_HIDDEN_GLOBAL_OFFSET_Z: {
                size_t offset_z = sizes.dimensions() == 3 ? newOffset[2] : 0;
                assert(arg->size_ == sizeof(offset_z) && "check the sizes");
                argPtr = addArg(argPtr, &offset_z, arg->size_, arg->alignment_);
                break;
            }
            case ROC_ARGTYPE_HIDDEN_PRINTF_BUFFER: {
                address bufferPtr = printfDbg()->dbgBuffer();
                assert(arg->size_ == sizeof(bufferPtr) && "check the sizes");
                argPtr = addArg(argPtr, &bufferPtr, arg->size_, arg->alignment_);
                break;
            }
            case ROC_ARGTYPE_HIDDEN_DEFAULT_QUEUE:
            case ROC_ARGTYPE_HIDDEN_COMPLETION_ACTION:
            case ROC_ARGTYPE_HIDDEN_NONE: {
                void* zero = 0;
                assert(arg->size_ <= sizeof(zero) && "check the sizes");
                argPtr = addArg(argPtr, &zero, arg->size_, arg->alignment_);
                break;
            }
            case ROC_ARGTYPE_POINTER: {
                if (arg->addrQual_ == ROC_ADDRESS_LOCAL) {
                    // Align the LDS on the alignment requirement of type pointed to
                    ldsUsage = amd::alignUp(ldsUsage, arg->pointeeAlignment_);
                    argPtr = addArg(argPtr, &ldsUsage, arg->size_, arg->alignment_);
                    ldsUsage += *reinterpret_cast<const size_t *>(srcArgPtr);
                    break;
                }
                assert((arg->addrQual_ == ROC_ADDRESS_GLOBAL
                     || arg->addrQual_ == ROC_ADDRESS_CONSTANT)
                         && "Unsupported address qualifier");
                if (kernelParams.boundToSvmPointer(dev(), parameters, arg->index_)) {
                    argPtr = addArg(argPtr, srcArgPtr, arg->size_, arg->alignment_);
                    break;
                }
                amd::Memory* mem = *reinterpret_cast<amd::Memory* const*>(srcArgPtr);
                if (mem == nullptr) {
                    argPtr = addArg(argPtr, srcArgPtr, arg->size_, arg->alignment_);
                    break;
                }

                Memory *devMem = static_cast<Memory *>(mem->getDeviceMemory(dev()));
                //! @todo add multi-devices synchronization when supported.
                void* globalAddress = devMem->getDeviceMemory();
                argPtr = addArg(argPtr, &globalAddress, arg->size_, arg->alignment_);

                //! @todo Compiler has to return read/write attributes
                if ((mem->getMemFlags() & CL_MEM_READ_ONLY) == 0) {
                    mem->signalWrite(&dev());
                }
                break;
            }
            case ROC_ARGTYPE_REFERENCE: {
                void *mem = allocKernArg(arg->size_, arg->alignment_);
                if (mem == nullptr) {
                    LogError("Out of memory");
                    return false;
                }
                memcpy(mem, srcArgPtr, arg->size_);
                argPtr = addArg(argPtr, &mem, sizeof(void*));
                break;
            }
            case ROC_ARGTYPE_VALUE:
                argPtr = addArg(argPtr, srcArgPtr, arg->size_, arg->alignment_);
                break;
            case ROC_ARGTYPE_IMAGE: {
                amd::Memory* mem = *reinterpret_cast<amd::Memory* const*>(srcArgPtr);
                Image* image = static_cast<Image *>(mem->getDeviceMemory(dev()));
                if (image == nullptr) {
                    LogError("Kernel image argument is not an image object");
                    return false;
                }

                if (dev().settings().enableImageHandle_) {
                    const uint64_t image_srd = image->getHsaImageObject().handle;
                    assert(amd::isMultipleOf(image_srd, sizeof(image_srd)));
                    argPtr = addArg(argPtr, &image_srd, sizeof(image_srd));
                }
                else {
                    // Image arguments are of size 48 bytes and are aligned to 16 bytes
                    argPtr = addArg(argPtr, (void *)image->getHsaImageObject().handle,
                        HSA_IMAGE_OBJECT_SIZE, HSA_IMAGE_OBJECT_ALIGNMENT);
                }

                //! @todo Compiler has to return read/write attributes
                if ((mem->getMemFlags() & CL_MEM_READ_ONLY) == 0) {
                    mem->signalWrite(&dev());
                }
                break;
            }
            case ROC_ARGTYPE_SAMPLER: {
                amd::Sampler* sampler = *reinterpret_cast<amd::Sampler* const*>(srcArgPtr);
                if (sampler == nullptr) {
                    LogError("Kernel sampler argument is not an sampler object");
                    return false;
                }

                hsa_ext_sampler_descriptor_t samplerDescriptor;
                fillSampleDescriptor(samplerDescriptor, *sampler);

                hsa_ext_sampler_t hsa_sampler;
                hsa_status_t status = hsa_ext_sampler_create(dev().getBackendDevice(),
                    &samplerDescriptor, &hsa_sampler);
                if (status != HSA_STATUS_SUCCESS) {
                    LogError("Error creating device sampler object!");
                    return false;
                }

                if (dev().settings().enableImageHandle_) {
                    uint64_t sampler_srd = hsa_sampler.handle;
                    argPtr = addArg(argPtr, &sampler_srd, sizeof(sampler_srd));
                    samplerList_.push_back(hsa_sampler);
                    // TODO: destroy sampler.
                }
                else {
                    argPtr = amd::alignUp(argPtr, HSA_SAMPLER_OBJECT_ALIGNMENT);

                    memcpy(argPtr, (void*)hsa_sampler.handle, HSA_SAMPLER_OBJECT_SIZE);
                    argPtr += HSA_SAMPLER_OBJECT_SIZE;
                    hsa_ext_sampler_destroy(dev().getBackendDevice(), hsa_sampler);
                }
                break;
            }
            default:
                return false;
            }
        }

        // Check there is no arguments' buffer overflow
        assert(argPtr <= argBuffer + gpuKernel.KernargSegmentByteSize());

        // Check for group memory overflow
        //! @todo Check should be in HSA - here we should have at most an assert
        assert(roc_device_.info().localMemSizePerCU_ > 0);
        if (ldsUsage > roc_device_.info().localMemSizePerCU_) {
            LogError("No local memory available\n");
            return false;
        }

        //Initialize the dispatch Packet
        hsa_kernel_dispatch_packet_t dispatchPacket;
        memset(&dispatchPacket, 0, sizeof(dispatchPacket));

        dispatchPacket.kernel_object = gpuKernel.KernelCodeHandle();

        dispatchPacket.header = aqlHeader_;
        dispatchPacket.setup |= sizes.dimensions() << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        dispatchPacket.grid_size_x = sizes.dimensions()>0 ? newGlobalSize[0] : 1;
        dispatchPacket.grid_size_y = sizes.dimensions()>1 ? newGlobalSize[1] : 1;
        dispatchPacket.grid_size_z = sizes.dimensions()>2 ? newGlobalSize[2] : 1;

        const size_t* compile_size = devKernel->workGroupInfo()->compileSize_;
        if (sizes.local().product() != 0) {
            dispatchPacket.workgroup_size_x = sizes.dimensions()>0 ? sizes.local()[0] : 1;
            dispatchPacket.workgroup_size_y = sizes.dimensions()>1 ? sizes.local()[1] : 1;
            dispatchPacket.workgroup_size_z = sizes.dimensions()>2 ? sizes.local()[2] : 1;
        } else {
            amd::NDRangeContainer  tmpSizes(sizes.dimensions(),
                &newOffset[0], &newGlobalSize[0],
                &(const_cast<amd::NDRangeContainer&>(sizes).local()[0]));

            setRuntimeCompilerLocalSize(dispatchPacket, tmpSizes, compile_size, dev());
        }
        dispatchPacket.kernarg_address = argBuffer;
        dispatchPacket.group_segment_size = ldsUsage;
        dispatchPacket.private_segment_size = devKernel->workGroupInfo()->privateMemSize_;

        //Dispatch the packet
        if (!dispatchAqlPacket(&dispatchPacket, GPU_FLUSH_ON_EXECUTION)){
            return false;
        }
    }

    // Mark the flag indicating if a dispatch is outstanding.
    // We are not waiting after every dispatch.
    hasPendingDispatch_ = true;

    // Output printf buffer
    if(!printfDbg()->output(*this, printfEnabled, gpuKernel.printfInfo())){
        LogError("\nCould not print data from the printf buffer!");
        return false;
    }
    return true;
}
/**
 * @brief Api to dispatch a kernel for execution. The implementation
 * parses the input object, an instance of virtual command to obtain
 * the parameters of global size, work group size, offsets of work
 * items, enable/disable profiling, etc.
 *
 * It also parses the kernel arguments buffer to inject into Hsa Runtime
 * the list of kernel parameters.
 */
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand &vcmd) {
    profilingBegin(vcmd);

    // Submit kernel to HW
    if (!submitKernelInternal(
            vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
            static_cast<void *>(as_cl(&vcmd.event())))) {
        LogError("AQL dispatch failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
}

void VirtualGPU::submitNativeFn(amd::NativeFnCommand &cmd) {
  // std::cout<<__FUNCTION__<<" not implemented"<<"*********"<<std::endl;
}

void VirtualGPU::submitMarker(amd::Marker &cmd) {
  // std::cout<<__FUNCTION__<<" not implemented"<<"*********"<<std::endl;
}

void VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& vcmd)
{
  profilingBegin(vcmd);
  auto fence=kBarrierAcquirePacket;
  dispatchAqlPacket(&fence, false);
  profilingEnd(vcmd);
}

void VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& vcmd)
{
  profilingBegin(vcmd);
  auto fence=kBarrierReleasePacket;
  dispatchAqlPacket(&fence, false);
  profilingEnd(vcmd);
}

void VirtualGPU::flush(amd::Command *list, bool wait)
{
    releaseGpuMemoryFence();
    updateCommandsState(list);
    // Release all pinned memory
    releasePinnedMem();
}

void
VirtualGPU::addXferWrite(Memory& memory)
{
    if (xferWriteBuffers_.size() > 7) {
        dev().xferWrite().release(*this, *xferWriteBuffers_.front());
        xferWriteBuffers_.erase(xferWriteBuffers_.begin());
    }

    // Delay destruction
    xferWriteBuffers_.push_back(&memory);
}

void
VirtualGPU::releaseXferWrite()
{
    for (auto& memory : xferWriteBuffers_) {
        dev().xferWrite().release(*this, *memory);
    }
    xferWriteBuffers_.resize(0);
}

void
VirtualGPU::addPinnedMem(amd::Memory* mem)
{
    if (nullptr == findPinnedMem(mem->getHostMem(), mem->getSize())) {
        if (pinnedMems_.size() > 7) {
            pinnedMems_.front()->release();
            pinnedMems_.erase(pinnedMems_.begin());
        }

        // Delay destruction
        pinnedMems_.push_back(mem);
    }
}

void
VirtualGPU::releasePinnedMem()
{
    for (auto& amdMemory : pinnedMems_) {
        amdMemory->release();
    }
    pinnedMems_.resize(0);
}

amd::Memory*
VirtualGPU::findPinnedMem(void* addr, size_t size)
{
    for (auto& amdMemory : pinnedMems_) {
        if ((amdMemory->getHostMem() == addr) && (size <= amdMemory->getSize())) {
            return amdMemory;
        }
    }
    return nullptr;
}

void
VirtualGPU::enableSyncBlit() const
{
    blitMgr_->enableSynchronization();
}
}  // End of roc namespace
