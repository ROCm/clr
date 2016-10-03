//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

/*!
 * \file command.cpp
 * \brief  Definitions for Event, Command and HostQueue objects.
 *
 * \author Laurent Morichetti (laurent.morichetti@amd.com)
 * \date   October 2008
 */

#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "device/device.hpp"
#include "platform/context.hpp"
#include "platform/kernel.hpp"
#include "thread/monitor.hpp"
#include "platform/memory.hpp"
#include "platform/agent.hpp"
#include "os/alloc.hpp"

#include <cstring>
#include <algorithm>

namespace amd {

Event::Event(HostQueue& queue)
    : callbacks_(NULL)
    , status_(CL_INT_MAX)
    , profilingInfo_(
        queue.properties().test(CL_QUEUE_PROFILING_ENABLE)
        || Agent::shouldPostEventEvents())
{ notified_.clear(); }

Event::Event()
    : callbacks_(NULL)
    , status_(CL_SUBMITTED)
{ notified_.clear(); }

Event::~Event()
{
    CallBackEntry* callback = callbacks_;
    while (callback != NULL) {
        CallBackEntry* next = callback->next_;
        delete callback;
        callback = next;
    }
}

uint64_t
Event::recordProfilingInfo(cl_int status, uint64_t timeStamp)
{
    if (timeStamp == 0) {
        timeStamp = Os::timeNanos();
    }
    switch (status) {
    case CL_QUEUED:
        profilingInfo_.queued_ = timeStamp;
        break;
    case CL_SUBMITTED:
        profilingInfo_.submitted_ = timeStamp;
        break;
    case CL_RUNNING:
        profilingInfo_.start_ = timeStamp;
        break;
    default:
        profilingInfo_.end_ = timeStamp;
        if (profilingInfo_.callback_ != NULL) {
            profilingInfo_.callback_->callback(timeStamp - profilingInfo_.start_);
        }
        break;
    }
    return timeStamp;
}

bool
Event::setStatus(cl_int status, uint64_t timeStamp)
{
    assert(status <= CL_QUEUED && "invalid status");

    cl_int currentStatus = status_;
    if (currentStatus <= CL_COMPLETE || currentStatus <= status) {
        // We can only move forward in the execution status.
        return false;
    }

    if (profilingInfo().enabled_) {
        timeStamp = recordProfilingInfo(status, timeStamp);
    }

    if (!make_atomic(status_).compareAndSet(currentStatus, status)) {
        // Somebody else beat us to it, let them deal with the release/signal.
        return false;
    }

    if (callbacks_ != (CallBackEntry*)0) {
        processCallbacks(status);
    }

    if (Agent::shouldPostEventEvents() && command().type() != 0) {
        Agent::postEventStatusChanged(
            as_cl(this), status, timeStamp + Os::offsetToEpochNanos());
    }

    if (status <= CL_COMPLETE) {
        // Before we notify the waiters that this event reached the CL_COMPLETE
        // status, we release all the resources associated with this instance.
        releaseResources();

        // Broadcast all the waiters.
        if (referenceCount() > 1) {
            signal();
        }
        release();
    }

    return true;
}


bool
Event::setCallback(cl_int status, Event::CallBackFunction callback, void* data)
{
    assert(status >= CL_COMPLETE && status <= CL_QUEUED && "invalid status");

    CallBackEntry* entry = new CallBackEntry(status, callback, data);
    if (entry == NULL) {
        return false;
    }

    entry->next_ = callbacks_;
    while (!callbacks_.compare_exchange_weak(entry->next_, entry))
        ; // Someone else is also updating the head of the linked list! reload.

    // Check if the event has already reached 'status'
    if (status_ <= status && entry->callback_ != CallBackFunction(0)) {
        if (entry->callback_.exchange(NULL) != NULL) {
            callback(as_cl(this), status, entry->data_);
        }
    }

    return true;
}


void
Event::processCallbacks(cl_int status) const
{
    cl_event event = const_cast<cl_event>(as_cl(this));
    const cl_int mask = (status > CL_COMPLETE) ? status : CL_COMPLETE;

    // For_each callback:
    CallBackEntry* entry;
    for (entry = callbacks_; entry != NULL; entry = entry->next_) {
        // If the entry's status matches the mask,
        if (entry->status_ == mask && entry->callback_ != CallBackFunction(0)) {
            // invoke the callback function.
            CallBackFunction callback = entry->callback_.exchange(NULL);
            if (callback != NULL) {
                callback(event, status, entry->data_);
            }
        }
    }
}

bool
Event::awaitCompletion()
{
    if (status_ > CL_COMPLETE) {
        // Notifies current command queue about waiting
        if (!notifyCmdQueue()) {
            return false;
        }

        ScopedLock lock(lock_);

        // Wait until the status becomes CL_COMPLETE or negative.
        while (status_ > CL_COMPLETE) {
            lock_.wait();
        }
    }

    return status_ == CL_COMPLETE;
}

bool
Event::notifyCmdQueue()
{
    HostQueue* queue = command().queue();
    if ((NULL != queue) && !notified_.test_and_set()) {
        // Make sure the queue is draining the enqueued commands.
        amd::Command* command = new amd::Marker(*queue, false, nullWaitList, this);
        if (command == NULL) {
            notified_.clear();
            return false;
        }
        command->enqueue();
        command->release();
    }
    return true;
}

const Event::EventWaitList Event::nullWaitList(0);

Command::Command(
    HostQueue& queue,
    cl_command_type type,
    const EventWaitList& eventWaitList) :
        Event(queue), queue_(&queue), next_(NULL), type_(type),
        exception_(0), data_(NULL), eventWaitList_(eventWaitList)
{
    // Retain the commands from the event wait list.
    std::for_each(
        eventWaitList.begin(),
        eventWaitList.end(),
        std::mem_fun(&Command::retain));
}

void
Command::releaseResources()
{
    const Command::EventWaitList& events = eventWaitList();

    // Release the commands from the event wait list.
    std::for_each(
        events.begin(),
        events.end(),
        std::mem_fun(&Command::release));
}

void
Command::enqueue()
{
    assert(queue_ != NULL && "Cannot be enqueued");

    if (Agent::shouldPostEventEvents() && type_ != 0) {
        Agent::postEventCreate(as_cl(static_cast<Event*>(this)), type_);
    }

    queue_->append(*this);
    queue_->flush();
    if (queue_->device().settings().waitCommand_ && (type_ != 0)) {
        awaitCompletion();
    }
}

const Context& 
Command::context() const
{
    return queue_->context();
}

NDRangeKernelCommand::NDRangeKernelCommand(
    HostQueue& queue,
    const EventWaitList& eventWaitList,
    Kernel& kernel,
    const NDRangeContainer& sizes) :
        Command(queue, CL_COMMAND_NDRANGE_KERNEL, eventWaitList),
        kernel_(kernel), sizes_(sizes)
{
    parameters_ = kernel.parameters().capture(queue.device());
    auto& device = queue.device();
    auto devKernel = const_cast<device::Kernel*>(kernel.getDeviceKernel(device));
    profilingInfo_.setCallback(devKernel->getProfilingCallback(queue.vdev()));
    fixme_guarantee(parameters_ != NULL && "out of memory");
    kernel_.retain();
}

void NDRangeKernelCommand::releaseResources() {
    kernel_.parameters().release(parameters_, queue()->device());
    DEBUG_ONLY(parameters_ = NULL);
    kernel_.release();
    Command::releaseResources();
}

NativeFnCommand::NativeFnCommand(
    HostQueue& queue, const EventWaitList& eventWaitList,
    void (CL_CALLBACK *nativeFn)(void*), const void* args, size_t argsSize,
    size_t numMemObjs, const cl_mem* memObjs, const void** memLocs) :
        Command(queue, CL_COMMAND_NATIVE_KERNEL, eventWaitList),
        nativeFn_(nativeFn), argsSize_(argsSize)
{
    args_ = new char[argsSize_];
    if (args_ == NULL) {
        return;
    }
    ::memcpy(args_, args, argsSize_);

    memObjects_.resize(numMemObjs);
    memOffsets_.resize(numMemObjs);
    for (size_t i = 0; i < numMemObjs; ++i) {
        Memory* obj = as_amd(memObjs[i]);

        obj->retain();
        memObjects_[i] = obj;
        memOffsets_[i] = (const_address) memLocs[i] - (const_address) args;
    }
}

cl_int
NativeFnCommand::invoke()
{
    size_t numMemObjs = memObjects_.size();
    for (size_t i = 0; i < numMemObjs; ++i) {
        void* hostMemPtr = memObjects_[i]->getHostMem();
        if (hostMemPtr == NULL) {
            return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }
        *reinterpret_cast<void **>(&args_[memOffsets_[i]]) = hostMemPtr;
    }
    nativeFn_(args_);
    return CL_SUCCESS;
}

bool
OneMemoryArgCommand::validateMemory()
 {
    if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        device::Memory* mem = memory_->getDeviceMemory(queue()->device());
        if (NULL == mem) {
            LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                memory_->getSize());
            return false;
        }
    }
    return true;
}

bool
TwoMemoryArgsCommand::validateMemory()
{
    if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        device::Memory* mem = memory1_->getDeviceMemory(queue()->device());
        if (NULL == mem) {
            LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                memory1_->getSize());
            return false;
        }
        mem = memory2_->getDeviceMemory(queue()->device());
        if (NULL == mem) {
            LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                memory2_->getSize());
            return false;
        }
    }
    return true;
}
bool
ReadMemoryCommand::isEntireMemory() const
{
    return source().isEntirelyCovered(origin(), size());
}

bool
WriteMemoryCommand::isEntireMemory() const
{
    return destination().isEntirelyCovered(origin(), size());
}

bool
SvmMapMemoryCommand::isEntireMemory() const
{
    return getSvmMem()->isEntirelyCovered(origin(), size());
}

bool
FillMemoryCommand::isEntireMemory() const
{
    return memory().isEntirelyCovered(origin(), size());
}

bool
CopyMemoryCommand::isEntireMemory() const
{
    bool result = false;

    switch (type()) {
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
        Coord3D imageSize(size()[0] * size()[1] * size()[2] *
            source().asImage()->getImageFormat().getElementSize());
        result = source().isEntirelyCovered(srcOrigin(), size()) &&
            destination().isEntirelyCovered(dstOrigin(), imageSize);
    }
        break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
        Coord3D imageSize(size()[0] * size()[1] * size()[2] *
            destination().asImage()->getImageFormat().getElementSize());
        result = source().isEntirelyCovered(srcOrigin(), imageSize) &&
            destination().isEntirelyCovered(dstOrigin(), size());
    }
        break;
    case CL_COMMAND_COPY_BUFFER_RECT: {
        Coord3D rectSize(size()[0] * size()[1] * size()[2]);
        Coord3D srcOffs(srcRect().start_);
        Coord3D dstOffs(dstRect().start_);
        result = source().isEntirelyCovered(srcOffs, rectSize) &&
            destination().isEntirelyCovered(dstOffs, rectSize);
    }
        break;
    default:
        result = source().isEntirelyCovered(srcOrigin(), size()) &&
            destination().isEntirelyCovered(dstOrigin(), size());
        break;
    }
    return result;
}

bool
MapMemoryCommand::isEntireMemory() const
{
    return memory().isEntirelyCovered(origin(), size());
}

void
UnmapMemoryCommand::releaseResources()
{
    if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        //! @todo This is a workaround to a deadlock on indirect map release.
        //! Remove this code when CAL will have a refcounter on memory.
        //! decIndMapCount() has to go back to submitUnmapMemory()
        device::Memory* mem = memory_->getDeviceMemory(queue()->device());
        if (NULL != mem) {
            mem->releaseIndirectMap();
        }
    }
    OneMemoryArgCommand::releaseResources();
}

bool
MigrateMemObjectsCommand::validateMemory()
{
    if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        std::vector<amd::Memory*>::const_iterator   itr;
        for (itr = memObjects_.begin(); itr != memObjects_.end(); itr++) {
            device::Memory* mem = (*itr)->getDeviceMemory(queue()->device());
            if (NULL == mem) {
                LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                    (*itr)->getSize());
                return false;
            }
        }
    }

    return true;
}

cl_int
NDRangeKernelCommand::validateMemory()
{
    const amd::Device& device = queue()->device();
    if (device.info().type_ & CL_DEVICE_TYPE_GPU) {
        // Validate the kernel before submission
        if (!queue()->device().validateKernel(kernel(), queue()->vdev())) {
            return CL_OUT_OF_RESOURCES;
        }

        const amd::KernelSignature& signature = kernel().signature();
        for (uint i = 0; i != signature.numParameters(); ++i) {
            const amd::KernelParameterDescriptor& desc = signature.at(i);
            // Check if it's a memory object
            if ((desc.type_ == T_POINTER) && (desc.size_ != 0)) {
                amd::Memory* amdMemory;
                if (kernel().parameters().boundToSvmPointer(device,
                                                            parameters_,
                                                            i)) {
                    //find the real mem object from svm ptr from the list
                    amdMemory = amd::SvmManager::FindSvmBuffer(
                        *reinterpret_cast<void* const*>(parameters() + desc.offset_));
                }
                else {
                    amdMemory = *reinterpret_cast<amd::Memory* const*>
                        (parameters() + desc.offset_);
                }
                if (amdMemory != NULL) {
                    if (desc.addressQualifier_ == CL_KERNEL_ARG_ADDRESS_CONSTANT) {
                        // Make sure argument size isn't bigger than the device limit
                        if (amdMemory->getSize() > device.info().maxConstantBufferSize_) {
                            LogPrintfError("HW constant buffer is too big (0x%X bytes)!",
                                amdMemory->getSize());
                            return CL_OUT_OF_RESOURCES;
                        }
                    }
                    device::Memory* mem =
                        amdMemory->getDeviceMemory(device);
                    if (!kernel().getDeviceKernel(
                            device)->validateMemory(i, amdMemory)) {
                        if (device.reallocMemory(*amdMemory)) {
                            mem = amdMemory->getDeviceMemory(device);
                        }
                        else {
                            mem = NULL;
                        }
                    }
                    if (NULL == mem) {
                        LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                            amdMemory->getSize());
                        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
                    }
                }
            }
        }
    }
    return CL_SUCCESS;
}

bool ExtObjectsCommand::validateMemory()
{
   bool retVal = true;
   if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        for(std::vector<amd::Memory*>::const_iterator itr = memObjects_.begin();
            itr != memObjects_.end(); itr++) {
            device::Memory* mem = (*itr)->getDeviceMemory(queue()->device());
            if (NULL == mem) {
                LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                    (*itr)->getSize());
                return false;
            }
            retVal = processGLResource(mem);
        }
    }
    return retVal;
}

bool AcquireExtObjectsCommand::processGLResource(device::Memory * mem)
{
    return mem->processGLResource(device::Memory::GLDecompressResource);
}

bool ReleaseExtObjectsCommand::processGLResource(device::Memory * mem)
{
    return mem->processGLResource(device::Memory::GLInvalidateFBO);
}

bool
MakeBuffersResidentCommand::validateMemory()
{
    if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        for(std::vector<amd::Memory*>::const_iterator itr = memObjects_.begin();
            itr != memObjects_.end(); itr++) {
            device::Memory* mem = (*itr)->getDeviceMemory(queue()->device());
            if (NULL == mem) {
                LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                    (*itr)->getSize());
                return false;
            }
        }
    }

    return true;

}
bool
ThreadTraceMemObjectsCommand::validateMemory()
{
    if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        for(std::vector<amd::Memory*>::const_iterator itr = memObjects_.begin();
            itr != memObjects_.end(); itr++) {
            device::Memory* mem = (*itr)->getDeviceMemory(queue()->device());
            if (NULL == mem) {
                std::vector<amd::Memory*>::const_iterator   tmpItr;
                for (tmpItr = memObjects_.begin(); tmpItr != itr; tmpItr++) {
                    device::Memory* tmpMem = (*tmpItr)->getDeviceMemory(queue()->device());
                    delete tmpMem;
                }
                LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                    (*itr)->getSize());
                return false;
            }
        }
    }

    return true;
}

void
TransferBufferFileCommand::releaseResources()
{
    for (uint i = 0; i < NumStagingBuffers; ++i) {
        if (NULL != staging_[i]) {
            staging_[i]->release();
        }
    }

    // Call the parent
    OneMemoryArgCommand::releaseResources();
}

void
TransferBufferFileCommand::submit(device::VirtualDevice& device)
{
    device::Memory* mem = memory_->getDeviceMemory(queue()->device());
    if (memory_->getMemFlags() & (CL_MEM_USE_HOST_PTR |
        CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_PERSISTENT_MEM_AMD)) {
        void* srcDstBuffer = mem->cpuMap(device);
        // Make HD transfer to the host accessible memory
        bool writeBuffer(type() == CL_COMMAND_WRITE_BUFFER_FROM_FILE_AMD);
        if (!file()->transferBlock(writeBuffer, srcDstBuffer, mem->size(),
                                   fileOffset(), origin()[0], size()[0])) {
            setStatus(CL_INVALID_OPERATION);
            return;
        }
        mem->cpuUnmap(device);
    }
    else {
        device.submitTransferBufferFromFile(*this);
    }
}

bool
TransferBufferFileCommand::validateMemory()
{
    if (queue()->device().info().type_ & CL_DEVICE_TYPE_GPU) {
        // Check if the destination buffer has direct host access
        if (!(memory_->getMemFlags() & (CL_MEM_USE_HOST_PTR |
                CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_PERSISTENT_MEM_AMD))) {
            // Allocate staging buffers
            for (uint i = 0; i < NumStagingBuffers; ++i) {
                staging_[i] = new (memory_->getContext())
                    Buffer(memory_->getContext(),
                        StagingBufferMemType, StagingBufferSize);
                if (NULL == staging_[i] || !staging_[i]->create(nullptr)) {
                    return false;
                }
                device::Memory* mem = staging_[i]->getDeviceMemory(queue()->device());
                if (NULL == mem) {
                    LogPrintfError("Can't allocate staging buffer - 0x%08X bytes!",
                        staging_[i]->getSize());
                    return false;
                }
            }
        }

        device::Memory* mem = memory_->getDeviceMemory(queue()->device());
        if (NULL == mem) {
            LogPrintfError("Can't allocate memory size - 0x%08X bytes!",
                memory_->getSize());
            return false;
        }
    }
    return true;
}

} // namespace amd
