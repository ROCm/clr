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

/*!
 * \file command.cpp
 * \brief  Definitions for Event, Command and HostQueue objects.
 *
 * \author Laurent Morichetti
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
    : callbacks_(NULL),
      status_(CL_INT_MAX),
      profilingInfo_(queue.properties().test(CL_QUEUE_PROFILING_ENABLE) ||
                     Agent::shouldPostEventEvents()) {
  notified_.clear();
}

Event::Event() : callbacks_(NULL), status_(CL_SUBMITTED) { notified_.clear(); }

Event::~Event() {
  CallBackEntry* callback = callbacks_;
  while (callback != NULL) {
    CallBackEntry* next = callback->next_;
    delete callback;
    callback = next;
  }
}

uint64_t Event::recordProfilingInfo(int32_t status, uint64_t timeStamp) {
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
        profilingInfo_.callback_->callback(timeStamp - profilingInfo_.start_,
            profilingInfo_.waves_);
      }
      break;
  }
  return timeStamp;
}

// Global epoch time since the first processed command
uint64_t epoch = 0;

bool Event::setStatus(int32_t status, uint64_t timeStamp) {
  assert(status <= CL_QUEUED && "invalid status");

  int32_t currentStatus = status_;
  if (currentStatus <= CL_COMPLETE || currentStatus <= status) {
    // We can only move forward in the execution status.
    return false;
  }

  if (profilingInfo().enabled_) {
    timeStamp = recordProfilingInfo(status, timeStamp);
    if (epoch == 0) {
      epoch = profilingInfo().queued_;
    }
  }

  if (!make_atomic(status_).compareAndSet(currentStatus, status)) {
    // Somebody else beat us to it, let them deal with the release/signal.
    return false;
  }

  if (callbacks_ != (CallBackEntry*)0) {
    processCallbacks(status);
  }

  if (Agent::shouldPostEventEvents() && command().type() != 0) {
    Agent::postEventStatusChanged(as_cl(this), status, timeStamp + Os::offsetToEpochNanos());
  }

  if (status <= CL_COMPLETE) {
    // Before we notify the waiters that this event reached the CL_COMPLETE
    // status, we release all the resources associated with this instance.
    releaseResources();

    activity_.ReportEventTimestamps(*this);
    // Broadcast all the waiters.
    if (referenceCount() > 1) {
      signal();
    }

    ClPrint(LOG_DEBUG, LOG_CMD, "command %p complete (Wall: %ld, CPU: %ld, GPU: %ld us)",
      &command(),
      ((profilingInfo().end_ - epoch) / 1000),
      ((profilingInfo().submitted_ - profilingInfo().queued_) / 1000),
      ((profilingInfo().end_ - profilingInfo().start_) / 1000));

    release();
  }

  return true;
}


bool Event::setCallback(int32_t status, Event::CallBackFunction callback, void* data) {
  assert(status >= CL_COMPLETE && status <= CL_QUEUED && "invalid status");

  CallBackEntry* entry = new CallBackEntry(status, callback, data);
  if (entry == NULL) {
    return false;
  }

  entry->next_ = callbacks_;
  while (!callbacks_.compare_exchange_weak(entry->next_, entry))
    ;  // Someone else is also updating the head of the linked list! reload.

  // Check if the event has already reached 'status'
  if (status_ <= status && entry->callback_ != CallBackFunction(0)) {
    if (entry->callback_.exchange(NULL) != NULL) {
      callback(as_cl(this), status, entry->data_);
    }
  }

  return true;
}


void Event::processCallbacks(int32_t status) const {
  cl_event event = const_cast<cl_event>(as_cl(this));
  const int32_t mask = (status > CL_COMPLETE) ? status : CL_COMPLETE;

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

bool Event::awaitCompletion() {
  if (status_ > CL_COMPLETE) {
    // Notifies current command queue about waiting
    if (!notifyCmdQueue()) {
      return false;
    }

    ClPrint(LOG_DEBUG, LOG_WAIT, "waiting for event %p to complete, current status %d", this, status_);
    auto* queue = command().queue();
    if ((queue != nullptr) && queue->vdev()->ActiveWait()) {
      while (status_ > CL_COMPLETE) {
        amd::Os::yield();
      }
    } else {
      ScopedLock lock(lock_);

      // Wait until the status becomes CL_COMPLETE or negative.
      while (status_ > CL_COMPLETE) {
        lock_.wait();
      }
    }

    ClPrint(LOG_DEBUG, LOG_WAIT, "event %p wait completed", this);
  }

  return status_ == CL_COMPLETE;
}

bool Event::notifyCmdQueue() {
  HostQueue* queue = command().queue();
  if ((NULL != queue) && !notified_.test_and_set()) {
    // Make sure the queue is draining the enqueued commands.
    amd::Command* command = new amd::Marker(*queue, false, nullWaitList, this);
    if (command == NULL) {
      notified_.clear();
      return false;
    }
    ClPrint(LOG_DEBUG, LOG_CMD, "queue marker to command queue: %p", queue);
    command->enqueue();
    command->release();
  }
  return true;
}

const Event::EventWaitList Event::nullWaitList(0);

Command::Command(HostQueue& queue, cl_command_type type,
                 const EventWaitList& eventWaitList, uint32_t commandWaitBits)
    : Event(queue),
      queue_(&queue),
      next_(NULL),
      type_(type),
      exception_(0),
      data_(NULL),
      eventWaitList_(eventWaitList),
      commandWaitBits_(commandWaitBits) {
  // Retain the commands from the event wait list.
  std::for_each(eventWaitList.begin(), eventWaitList.end(), std::mem_fun(&Command::retain));
  if (type != 0) activity_.Initialize(type, queue.vdev()->index(), queue.device().index());
}

void Command::releaseResources() {
  const Command::EventWaitList& events = eventWaitList();

  // Release the commands from the event wait list.
  std::for_each(events.begin(), events.end(), std::mem_fun(&Command::release));
}

void Command::enqueue() {
  assert(queue_ != NULL && "Cannot be enqueued");

  if (Agent::shouldPostEventEvents() && type_ != 0) {
    Agent::postEventCreate(as_cl(static_cast<Event*>(this)), type_);
  }

  ClPrint(LOG_DEBUG, LOG_CMD, "command is enqueued: %p", this);
  queue_->append(*this);
  if (IS_HIP) {
    queue_->setLastQueuedCommand(this);
  }
  queue_->flush();
  if ((queue_->device().settings().waitCommand_ && (type_ != 0)) ||
      ((commandWaitBits_ & 0x2) != 0)) {
    awaitCompletion();
  }
}

const Context& Command::context() const { return queue_->context(); }

NDRangeKernelCommand::NDRangeKernelCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                                           Kernel& kernel, const NDRangeContainer& sizes,
                                           uint32_t sharedMemBytes, uint32_t extraParam,
                                           uint32_t gridId, uint32_t numGrids,
                                           uint64_t prevGridSum, uint64_t allGridSum, uint32_t firstDevice) :
    Command(queue, CL_COMMAND_NDRANGE_KERNEL, eventWaitList, AMD_SERIALIZE_KERNEL),
    kernel_(kernel),
    sizes_(sizes),
    sharedMemBytes_(sharedMemBytes),
    extraParam_(extraParam),
    gridId_(gridId),
    numGrids_(numGrids),
    prevGridSum_(prevGridSum),
    allGridSum_(allGridSum),
    firstDevice_(firstDevice) {
  auto& device = queue.device();
  auto devKernel = const_cast<device::Kernel*>(kernel.getDeviceKernel(device));
  profilingInfo_.setCallback(devKernel->getProfilingCallback(
    queue.vdev()), devKernel->getWavesPerSH(queue.vdev()));
  kernel_.retain();
}

void NDRangeKernelCommand::releaseResources() {
  kernel_.parameters().release(parameters_, queue()->device());
  DEBUG_ONLY(parameters_ = NULL);
  kernel_.release();
  Command::releaseResources();
}

NativeFnCommand::NativeFnCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                                 void(CL_CALLBACK* nativeFn)(void*), const void* args,
                                 size_t argsSize, size_t numMemObjs, const cl_mem* memObjs,
                                 const void** memLocs)
    : Command(queue, CL_COMMAND_NATIVE_KERNEL, eventWaitList),
      nativeFn_(nativeFn),
      argsSize_(argsSize) {
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
    memOffsets_[i] = (const_address)memLocs[i] - (const_address)args;
  }
}

int32_t NativeFnCommand::invoke() {
  size_t numMemObjs = memObjects_.size();
  for (size_t i = 0; i < numMemObjs; ++i) {
    void* hostMemPtr = memObjects_[i]->getHostMem();
    if (hostMemPtr == NULL) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
    *reinterpret_cast<void**>(&args_[memOffsets_[i]]) = hostMemPtr;
  }
  nativeFn_(args_);
  return CL_SUCCESS;
}

bool OneMemoryArgCommand::validateMemory() {
  // Runtime disables deferred memory allocation for single device.
  // Hence ignore memory validations
  if (queue()->context().devices().size() == 1) {
    return true;
  }
  device::Memory* mem = memory_->getDeviceMemory(queue()->device());
  if (NULL == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", memory_->getSize());
    return false;
  }
  return true;
}

bool TwoMemoryArgsCommand::validateMemory() {
  // Runtime disables deferred memory allocation for single device.
  // Hence ignore memory validations
  if (queue()->context().devices().size() == 1) {
    return true;
  }
  device::Memory* mem = memory1_->getDeviceMemory(queue()->device());
  if (NULL == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", memory1_->getSize());
    return false;
  }
  mem = memory2_->getDeviceMemory(queue()->device());
  if (NULL == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", memory2_->getSize());
    return false;
  }
  return true;
}
bool ReadMemoryCommand::isEntireMemory() const {
  return source().isEntirelyCovered(origin(), size());
}

bool WriteMemoryCommand::isEntireMemory() const {
  return destination().isEntirelyCovered(origin(), size());
}

bool SvmMapMemoryCommand::isEntireMemory() const {
  return getSvmMem()->isEntirelyCovered(origin(), size());
}

bool FillMemoryCommand::isEntireMemory() const {
  return memory().isEntirelyCovered(origin(), size());
}

bool CopyMemoryCommand::isEntireMemory() const {
  bool result = false;

  switch (type()) {
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
      Coord3D imageSize(size()[0] * size()[1] * size()[2] *
                        source().asImage()->getImageFormat().getElementSize());
      result = source().isEntirelyCovered(srcOrigin(), size()) &&
          destination().isEntirelyCovered(dstOrigin(), imageSize);
    } break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
      Coord3D imageSize(size()[0] * size()[1] * size()[2] *
                        destination().asImage()->getImageFormat().getElementSize());
      result = source().isEntirelyCovered(srcOrigin(), imageSize) &&
          destination().isEntirelyCovered(dstOrigin(), size());
    } break;
    case CL_COMMAND_COPY_BUFFER_RECT: {
      Coord3D rectSize(size()[0] * size()[1] * size()[2]);
      Coord3D srcOffs(srcRect().start_);
      Coord3D dstOffs(dstRect().start_);
      result = source().isEntirelyCovered(srcOffs, rectSize) &&
          destination().isEntirelyCovered(dstOffs, rectSize);
    } break;
    default:
      result = source().isEntirelyCovered(srcOrigin(), size()) &&
          destination().isEntirelyCovered(dstOrigin(), size());
      break;
  }
  return result;
}

bool MapMemoryCommand::isEntireMemory() const {
  return memory().isEntirelyCovered(origin(), size());
}

void UnmapMemoryCommand::releaseResources() {
  //! @todo This is a workaround to a deadlock on indirect map release.
  //! Remove this code when CAL will have a refcounter on memory.
  //! decIndMapCount() has to go back to submitUnmapMemory()
  device::Memory* mem = memory_->getDeviceMemory(queue()->device());
  if (NULL != mem) {
    mem->releaseIndirectMap();
  }

  OneMemoryArgCommand::releaseResources();
}

bool MigrateMemObjectsCommand::validateMemory() {
  // Runtime disables deferred memory allocation for single device.
  // Hence ignore memory validations
  if (queue()->context().devices().size() == 1) {
    return true;
  }
  for (const auto& it : memObjects_) {
    device::Memory* mem = it->getDeviceMemory(queue()->device());
    if (NULL == mem) {
      LogPrintfError("Can't allocate memory size - 0x%08X bytes!", it->getSize());
      return false;
    }
  }
  return true;
}

int32_t NDRangeKernelCommand::captureAndValidate() {
  const amd::Device& device = queue()->device();
  // Validate the kernel before submission
  if (!queue()->device().validateKernel(kernel(), queue()->vdev(), cooperativeGroups())) {
    return CL_OUT_OF_RESOURCES;
  }

  int32_t error;
  uint64_t lclMemSize = kernel().getDeviceKernel(device)->workGroupInfo()->localMemSize_;
  parameters_ = kernel().parameters().capture(device, sharedMemBytes_ + lclMemSize, &error);
  return error;
}

bool ExtObjectsCommand::validateMemory() {
  // Always process GL objects, even if deferred allocations are disabled,
  // because processGLResource() calls OGL Acquire().
  bool retVal = true;
  for (const auto& it : memObjects_) {
    device::Memory* mem = it->getDeviceMemory(queue()->device());
    if (NULL == mem) {
      LogPrintfError("Can't allocate memory size - 0x%08X bytes!", it->getSize());
      return false;
    }
    retVal = processGLResource(mem);
  }
  return retVal;
}

bool AcquireExtObjectsCommand::processGLResource(device::Memory* mem) {
  return mem->processGLResource(device::Memory::GLDecompressResource);
}

bool ReleaseExtObjectsCommand::processGLResource(device::Memory* mem) {
  return mem->processGLResource(device::Memory::GLInvalidateFBO);
}

bool MakeBuffersResidentCommand::validateMemory() {
  // Runtime disables deferred memory allocation for single device.
  // Hence ignore memory validations
  if (queue()->context().devices().size() == 1) {
    return true;
  }
  for (const auto& it : memObjects_) {
    device::Memory* mem = it->getDeviceMemory(queue()->device());
    if (NULL == mem) {
      LogPrintfError("Can't allocate memory size - 0x%08X bytes!", it->getSize());
      return false;
    }
  }
  return true;
}

bool ThreadTraceMemObjectsCommand::validateMemory() {
  // Runtime disables deferred memory allocation for single device.
  // Hence ignore memory validations
  if (queue()->context().devices().size() == 1) {
    return true;
  }
  for (auto it = memObjects_.cbegin(); it != memObjects_.cend(); it++) {
    device::Memory* mem = (*it)->getDeviceMemory(queue()->device());
    if (NULL == mem) {
      for (auto tmpIt = memObjects_.cbegin(); tmpIt != it; tmpIt++) {
        device::Memory* tmpMem = (*tmpIt)->getDeviceMemory(queue()->device());
        delete tmpMem;
      }
      LogPrintfError("Can't allocate memory size - 0x%08X bytes!", (*it)->getSize());
      return false;
    }
  }
  return true;
}

void TransferBufferFileCommand::releaseResources() {
  for (uint i = 0; i < NumStagingBuffers; ++i) {
    if (NULL != staging_[i]) {
      staging_[i]->release();
    }
  }

  // Call the parent
  OneMemoryArgCommand::releaseResources();
}

void TransferBufferFileCommand::submit(device::VirtualDevice& device) {
  device::Memory* mem = memory_->getDeviceMemory(queue()->device());
  if (memory_->getMemFlags() &
      (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_PERSISTENT_MEM_AMD)) {
    void* srcDstBuffer = nullptr;
    if (memory_->getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
      // Lock protected multiple maps for persistent memory
      amd::ScopedLock lock(mem->owner()->lockMemoryOps());
      srcDstBuffer = mem->cpuMap(device);
    } else {
      srcDstBuffer = mem->cpuMap(device);
    }
    // Make HD transfer to the host accessible memory
    bool writeBuffer(type() == CL_COMMAND_READ_SSG_FILE_AMD);
    if (!file()->transferBlock(writeBuffer, srcDstBuffer, mem->size(), fileOffset(), origin()[0],
                               size()[0])) {
      setStatus(CL_INVALID_OPERATION);
      return;
    }
    if (memory_->getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
      // Lock protected multiple maps for persistent memory
      amd::ScopedLock lock(mem->owner()->lockMemoryOps());
      mem->cpuUnmap(device);
    } else {
      mem->cpuUnmap(device);
    }
  } else {
    device.submitTransferBufferFromFile(*this);
  }
}

bool TransferBufferFileCommand::validateMemory() {
  // Check if the destination buffer has direct host access
  if (!(memory_->getMemFlags() &
        (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_PERSISTENT_MEM_AMD))) {
    // Allocate staging buffers
    for (uint i = 0; i < NumStagingBuffers; ++i) {
      staging_[i] = new (memory_->getContext())
          Buffer(memory_->getContext(), StagingBufferMemType, StagingBufferSize);
      if (NULL == staging_[i] || !staging_[i]->create(nullptr)) {
        DevLogPrintfError("Staging Create failed, Staging[%d]: 0x%x", i, staging_[i]);
        return false;
      }
      device::Memory* mem = staging_[i]->getDeviceMemory(queue()->device());
      if (NULL == mem) {
        LogPrintfError("Can't allocate staging buffer - 0x%08X bytes!", staging_[i]->getSize());
        return false;
      }
    }
  }

  device::Memory* mem = memory_->getDeviceMemory(queue()->device());
  if (NULL == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", memory_->getSize());
    return false;
  }
  return true;
}

bool CopyMemoryP2PCommand::validateMemory() {
  const std::vector<Device*>& devices = memory1_->getContext().devices();
  if (devices.size() != 1) {
    LogError("Can't allocate memory object for P2P extension");
    return false;
  }
  device::Memory* mem = memory1_->getDeviceMemory(*devices[0]);
  if (nullptr == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", memory1_->getSize());
    return false;
  }
  const std::vector<Device*>& devices2 = memory2_->getContext().devices();
  if (devices2.size() != 1) {
    LogError("Can't allocate memory object for P2P extension");
    return false;
  }
  mem = memory2_->getDeviceMemory(*devices2[0]);
  if (nullptr == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", memory2_->getSize());
    return false;
  }
  bool p2pStaging = false;
  // Validate P2P memories on the current device, if any of them is null, then it's p2p staging
  if ((nullptr == memory1_->getDeviceMemory(queue()->device())) ||
      (nullptr == memory2_->getDeviceMemory(queue()->device()))) {
    p2pStaging = true;
  }

  if (devices[0]->P2PStage() != nullptr && p2pStaging) {
    amd::ScopedLock lock(devices[0]->P2PStageOps());
    // Make sure runtime allocates memory on every device
    for (uint d = 0; d < devices[0]->GlbCtx().devices().size(); ++d) {
      device::Memory* mem = devices[0]->P2PStage()->getDeviceMemory(*devices[0]->GlbCtx().devices()[d]);
      if (nullptr == mem) {
        DevLogPrintfError("Cannot get P2P stage Device Memory for device: 0x%x \n",
                          devices[0]->GlbCtx().devices()[d]);
        return false;
      }
    }
  }
  return true;
}

}  // namespace amd
