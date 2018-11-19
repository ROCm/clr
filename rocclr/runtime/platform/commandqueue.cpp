//
// Copyright (c) 2012 Advanced Micro Devices, Inc. All rights reserved.
//

#include "commandqueue.hpp"
#include "thread/monitor.hpp"
#include "device/device.hpp"
#include "platform/context.hpp"

/*!
 * \file commandQueue.cpp
 * \brief  Definitions for HostQueue object.
 *
 * \author Laurent Morichetti (laurent.morichetti@amd.com)
 * \date   October 2008
 */

namespace amd {

HostQueue::HostQueue(Context& context, Device& device, cl_command_queue_properties properties,
                     uint queueRTCUs, Priority priority)
    : CommandQueue(context, device, properties,
                   device.info().queueProperties_ | CL_QUEUE_COMMAND_INTERCEPT_ENABLE_AMD,
                   queueRTCUs, priority),
      lastEnqueueCommand_(nullptr) {
  if (thread_.state() >= Thread::INITIALIZED) {
    ScopedLock sl(queueLock_);
    thread_.start(this);
    queueLock_.wait();
  }
}

bool HostQueue::terminate() {
  if (Os::isThreadAlive(thread_)) {
    Command* marker = nullptr;

    // Send a finish if the queue is still accepting commands.
    { ScopedLock sl(queueLock_);
      if (thread_.acceptingCommands_) {
        marker = new Marker(*this, false);
        if (marker != nullptr) {
          append(*marker);
          queueLock_.notify();
        }
      }
    }
    if (marker != nullptr) {
      marker->awaitCompletion();
      marker->release();
    }

    // Wake-up the command loop, so it can exit
    { ScopedLock sl(queueLock_);
      thread_.acceptingCommands_ = false;
      queueLock_.notify();
    }

    // FIXME_lmoriche: fix termination handshake
    while (thread_.state() < Thread::FINISHED) {
      Os::yield();
    }
  }

  if (Agent::shouldPostCommandQueueEvents()) {
    Agent::postCommandQueueFree(as_cl(this->asCommandQueue()));
  }

  return true;
}

void HostQueue::finish() {
  // Send a finish to make sure we finished all commands
  Command* command = new Marker(*this, false);
  if (command == NULL) {
    return;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();
}

void HostQueue::loop(device::VirtualDevice* virtualDevice) {
  cl_int(CL_CALLBACK * commandIntercept)(cl_event, cl_int*) =
      properties().test(CL_QUEUE_COMMAND_INTERCEPT_ENABLE_AMD) ? context().info().commandIntercept_
                                                               : NULL;

  // Notify the caller that the queue is ready to accept commands.
  {
    ScopedLock sl(queueLock_);
    thread_.acceptingCommands_ = true;
    queueLock_.notify();
  }
  // Create a command batch with all the commands present in the queue.
  Command* head = NULL;
  Command* tail = NULL;
  while (true) {
    // Get one command from the queue
    Command* command = queue_.dequeue();
    if (command == NULL) {
      ScopedLock sl(queueLock_);
      while ((command = queue_.dequeue()) == NULL) {
        if (!thread_.acceptingCommands_) {
          return;
        }
        queueLock_.wait();
      }
    }

    command->retain();

    // Process the command's event wait list.
    const Command::EventWaitList& events = command->eventWaitList();
    bool dependencyFailed = false;

    for (const auto& it : events) {
      // Only wait if the command is enqueued into another queue.
      if (it->command().queue() != this) {
        virtualDevice->flush(head, true);
        tail = head = NULL;
        dependencyFailed |= !it->awaitCompletion();
      }
    }

    // Insert the command to the linked list.
    if (NULL == head) {  // if the list is empty
      head = tail = command;
    } else {
      tail->setNext(command);
      tail = command;
    }

    if (dependencyFailed) {
      command->setStatus(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
      continue;
    }

    command->setStatus(CL_SUBMITTED);

    cl_int result;
    if ((commandIntercept != NULL) && commandIntercept(as_cl<Event>(command), &result)) {
      // The command was handled by the callback.
      command->setStatus(CL_RUNNING, command->profilingInfo().submitted_);
      command->setStatus(result);
      continue;
    }

    // Submit to the device queue.
    command->submit(*virtualDevice);

    // if we are in intercept mode or this is a user invisible marker command
    if ((0 == command->type()) || (commandIntercept != NULL)) {
      virtualDevice->flush(head);
      tail = head = NULL;
    }
  }  // while (true) {
}

void HostQueue::append(Command& command) {
  // We retain the command here. It will be released when its status
  // changes to CL_COMPLETE
  command.retain();
  command.setStatus(CL_QUEUED);
  queue_.enqueue(&command);
}

bool HostQueue::isEmpty() {
  // Get a snapshot of queue size
  return queue_.empty();
}

void HostQueue::setLastQueuedCommand(Command* lastCommand) {
  // Set last submitted command
  ScopedLock sl(queueLock_);
  if (lastEnqueueCommand_ != nullptr) {
    lastEnqueueCommand_->release();
  }
  lastEnqueueCommand_ = lastCommand;
  if (lastCommand != nullptr) {
    lastEnqueueCommand_->retain();
  }
}

Command* HostQueue::getLastQueuedCommand(bool retain) {
  // Get last submitted command
  ScopedLock sl(queueLock_);
  if (lastEnqueueCommand_ == nullptr) {
    return nullptr;
  }

  if (retain) {
    lastEnqueueCommand_->retain();
  }
  return lastEnqueueCommand_;
}

DeviceQueue::~DeviceQueue() {
  delete virtualDevice_;
  ScopedLock lock(context().lock());
  context().removeDeviceQueue(device(), this);
}

bool DeviceQueue::create() {
  static const bool InteropQueue = true;
  const bool defaultDeviceQueue = properties().test(CL_QUEUE_ON_DEVICE_DEFAULT);
  bool result = false;

  virtualDevice_ = device().createVirtualDevice(this);
  if (virtualDevice_ != NULL) {
    result = true;
    context().addDeviceQueue(device(), this, defaultDeviceQueue);
  }

  return result;
}

}  // namespace amd {
