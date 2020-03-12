/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

/*! \file command.hpp
 *  \brief  Declarations for Event, Command and HostQueue objects.
 *
 *  \author Laurent Morichetti
 *  \date   October 2008
 */

#ifndef COMMAND_HPP_
#define COMMAND_HPP_

#include "top.hpp"
#include "thread/monitor.hpp"
#include "thread/thread.hpp"
#include "platform/agent.hpp"
#include "platform/object.hpp"
#include "platform/context.hpp"
#include "platform/ndrange.hpp"
#include "platform/kernel.hpp"
#include "device/device.hpp"
#include "utils/concurrent.hpp"
#include "platform/memory.hpp"
#include "platform/perfctr.hpp"
#include "platform/threadtrace.hpp"
#include "platform/activity.hpp"

#include "CL/cl_ext.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <vector>

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Commands Event, Commands and Command-Queue
 *  @{
 */

class Command;
class HostQueue;

/*! \brief Encapsulates the status of a command.
 *
 *  \details An event object encapsulates the status of a Command
 *  it is associated with and can be used to synchronize operations
 *  in a Context.
 */
class Event : public RuntimeObject {
  typedef void(CL_CALLBACK* CallBackFunction)(cl_event event, int32_t command_exec_status,
                                              void* user_data);

  struct CallBackEntry : public HeapObject {
    struct CallBackEntry* next_;  //!< the next entry in the callback list.

    std::atomic<CallBackFunction> callback_;  //!< callback function pointer.
    void* data_;                              //!< user data passed to the callback function.
    int32_t status_;                           //!< execution status triggering the callback.

    CallBackEntry(int32_t status, CallBackFunction callback, void* data)
        : callback_(callback), data_(data), status_(status) {}
  };

 public:
  typedef std::vector<Event*> EventWaitList;

 private:
  Monitor lock_;

  std::atomic<CallBackEntry*> callbacks_;  //!< linked list of callback entries.
  volatile int32_t status_;                 //!< current execution status.
  std::atomic_flag notified_;              //!< Command queue was notified

 protected:
  static const EventWaitList nullWaitList;

  struct ProfilingInfo {
    ProfilingInfo(bool enabled = false) : enabled_(enabled), waves_(0) {
      if (enabled) {
        clear();
        callback_ = NULL;
      }
    }

    uint64_t queued_;
    uint64_t submitted_;
    uint64_t start_;
    uint64_t end_;
    bool enabled_;    //!< Profiling enabled for the wave limiter
    uint32_t waves_;  //!< The number of waves used in a dispatch
    ProfilingCallback* callback_;
    void clear() {
      queued_ = 0ULL;
      submitted_ = 0ULL;
      start_ = 0ULL;
      end_ = 0ULL;
    }
    void setCallback(ProfilingCallback* callback, uint32_t waves) {
      if (callback == NULL) {
        return;
      }
      enabled_ = true;
      waves_ = waves;
      clear();
      callback_ = callback;
    }

  } profilingInfo_;

  activity_prof::ActivityProf activity_;  //!< Activity profiling

  //! Construct a new event.
  Event();

  //! Construct a new event associated to the given command \a queue.
  Event(HostQueue& queue);

  //! Destroy the event.
  virtual ~Event();

  //! Release the resources associated with this event.
  virtual void releaseResources() {}

  //! Record the profiling info for the given change of \a status.
  //  If the given \a timeStamp is 0 and profiling is enabled,
  //  use the current host clock time instead.
  uint64_t recordProfilingInfo(int32_t status, uint64_t timeStamp = 0);

  //! Process the callbacks for the given \a status change.
  void processCallbacks(int32_t status) const;

 public:
  //! Return the context for this event.
  virtual const Context& context() const = 0;

  //! Return the command this event is associated with.
  inline Command& command();
  inline const Command& command() const;

  //! Return the profiling info.
  const ProfilingInfo& profilingInfo() const { return profilingInfo_; }

  //! Return this command's execution status.
  int32_t status() const { return status_; }

  //! Insert the given \a callback into the callback stack.
  bool setCallback(int32_t status, CallBackFunction callback, void* data);

  /*! \brief Set the event status.
   *
   *  \details If the status becomes CL_COMPLETE, notify all threads
   *  awaiting this command's completion.  If the given \a timeStamp is 0
   *  and profiling is enabled, use the current host clock time instead.
   *
   *  \see amd::Event::awaitCompletion
   */
  bool setStatus(int32_t status, uint64_t timeStamp = 0);

  //! Signal all threads waiting on this event.
  void signal() {
    ScopedLock lock(lock_);
    lock_.notifyAll();
  }

  /*! \brief Suspend the current thread until the status of the Command
   *  associated with this event changes to CL_COMPLETE. Return true if the
   *  command successfully completed.
   */
  virtual bool awaitCompletion();

  /*! \brief Notifies current command queue about execution status
   */
  bool notifyCmdQueue();

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeEvent; }
};

/*! \brief An operation that is submitted to a command queue.
 *
 *  %Command is the abstract base type of all OpenCL operations
 *  submitted to a HostQueue for execution. Classes derived from
 *  %Command must implement the submit() function.
 *

 */
class Command : public Event {
 private:
  //! The command queue this command is enqueue into. NULL if not yet enqueue.
  HostQueue* queue_;
  //! Next GPU command in the queue list
  Command* next_;

  const cl_command_type type_;  //!< This command's OpenCL type.
  volatile int32_t exception_;   //!< The first raised exception.
  void* data_;

 protected:
  //! The Events that need to complete before this command is submitted.
  EventWaitList eventWaitList_;

  //! Force await completion of previous command
  //! 0x1 - wait before enqueue, 0x2 - wait after, 0x3 - wait both.
  uint32_t commandWaitBits_;

  //! Construct a new command of the given OpenCL type.
  Command(HostQueue& queue, cl_command_type type, const EventWaitList& eventWaitList = nullWaitList,
          uint32_t commandWaitBits = 0);

  //! Construct a new command of the given OpenCL type.
  Command(cl_command_type type)
      : Event(),
        queue_(NULL),
        next_(NULL),
        type_(type),
        exception_(0),
        data_(NULL),
        eventWaitList_(nullWaitList),
        commandWaitBits_(0) {}

  bool terminate() {
    if (Agent::shouldPostEventEvents() && type() != 0) {
      Agent::postEventFree(as_cl(static_cast<Event*>(this)));
    }
    return true;
  }

 public:
  //! Return the queue this command is enqueued into.
  HostQueue* queue() const { return queue_; }

  //! Enqueue this command into the associated command queue.
  void enqueue();

  //! Return the event encapsulating this command's status.
  const Event& event() const { return *this; }
  Event& event() { return *this; }

  //! Return the list of events this command needs to wait on before dispatch
  const EventWaitList& eventWaitList() const { return eventWaitList_; }

  //! Return this command's OpenCL type.
  cl_command_type type() const { return type_; }

  //! Return the first raised exception or 0 if none.
  int32_t exception() const { return exception_; }

  //! Set the exception for this command.
  void setException(int32_t exception) { exception_ = exception; }

  //! Return the opaque, device specific data for this command.
  void* data() const { return data_; }

  //! Set the opaque, device specific data for this command.
  void setData(void* data) { data_ = data; }

  /*! \brief The execution engine for this command.
   *
   *  \details All derived class must implement this virtual function.
   *
   *  \note This function will execute in the command queue thread.
   */
  virtual void submit(device::VirtualDevice& device) = 0;

  //! Release the resources associated with this event.
  virtual void releaseResources();

  //! Set the next GPU command
  void setNext(Command* next) { next_ = next; }

  //! Get the next GPU command
  Command* getNext() const { return next_; }

  //! Return the context for this event.
  virtual const Context& context() const;

  //! Get command wait bits
  uint32_t getWaitBits() const { return commandWaitBits_; }
};

class UserEvent : public Command {
  const Context& context_;

 public:
  UserEvent(Context& context) : Command(CL_COMMAND_USER), context_(context) {
    setStatus(CL_SUBMITTED);
  }

  virtual void submit(device::VirtualDevice& device) { ShouldNotCallThis(); }

  virtual const Context& context() const { return context_; }
};

class ClGlEvent : public Command {
 private:
  const Context& context_;
  bool waitForFence();

 public:
  ClGlEvent(Context& context) : Command(CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR), context_(context) {
    setStatus(CL_SUBMITTED);
  }

  virtual void submit(device::VirtualDevice& device) { ShouldNotCallThis(); }

  bool awaitCompletion() { return waitForFence(); }

  virtual const Context& context() const { return context_; }
};

inline Command& Event::command() { return *static_cast<Command*>(this); }

inline const Command& Event::command() const { return *static_cast<const Command*>(this); }

class Kernel;
class NDRangeContainer;

//! A memory command that holds a single memory object reference.
//
class OneMemoryArgCommand : public Command {
 protected:
  Memory* memory_;

 public:
  OneMemoryArgCommand(HostQueue& queue, cl_command_type type, const EventWaitList& eventWaitList,
                      Memory& memory)
      : Command(queue, type, eventWaitList, AMD_SERIALIZE_COPY), memory_(&memory) {
    memory_->retain();
  }

  virtual void releaseResources() {
    memory_->release();
    DEBUG_ONLY(memory_ = NULL);
    Command::releaseResources();
  }

  bool validateMemory();
};

//! A memory command that holds a single memory object reference.
//
class TwoMemoryArgsCommand : public Command {
 protected:
  Memory* memory1_;
  Memory* memory2_;

 public:
  TwoMemoryArgsCommand(HostQueue& queue, cl_command_type type, const EventWaitList& eventWaitList,
                       Memory& memory1, Memory& memory2)
      : Command(queue, type, eventWaitList, AMD_SERIALIZE_COPY),
        memory1_(&memory1),
        memory2_(&memory2) {
    memory1_->retain();
    memory2_->retain();
  }

  virtual void releaseResources() {
    memory1_->release();
    memory2_->release();
    DEBUG_ONLY(memory1_ = memory2_ = NULL);
    Command::releaseResources();
  }

  bool validateMemory();
};

/*!  \brief     A generic read memory command.
 *
 *   \details   Used for operations on both buffers and images. Backends
 *              are expected to handle any required translation. Buffers
 *              are treated as 1D structures so origin_[0] and size_[0]
 *              are equivalent to offset_ and count_ respectively.
 *
 *   @todo Find a cleaner way of merging the row and slice pitch concepts at this level.
 *
 */

class ReadMemoryCommand : public OneMemoryArgCommand {
 private:
  Coord3D origin_;     //!< Origin of the region to read.
  Coord3D size_;       //!< Size of the region to read.
  void* hostPtr_;      //!< The host pointer destination.
  size_t rowPitch_;    //!< Row pitch (for image operations)
  size_t slicePitch_;  //!< Slice pitch (for image operations)

  BufferRect bufRect_;   //!< Buffer rectangle information
  BufferRect hostRect_;  //!< Host memory rectangle information

 public:
  //! Construct a new ReadMemoryCommand
  ReadMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& memory, Coord3D origin, Coord3D size, void* hostPtr,
                    size_t rowPitch = 0, size_t slicePitch = 0)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(rowPitch),
        slicePitch_(slicePitch) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  //! Construct a new ReadMemoryCommand
  ReadMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& memory, Coord3D origin, Coord3D size, void* hostPtr,
                    const BufferRect& bufRect, const BufferRect& hostRect)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(0),
        slicePitch_(0),
        bufRect_(bufRect),
        hostRect_(hostRect) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitReadMemory(*this); }

  //! Return the memory object to read from.
  Memory& source() const { return *memory_; }
  //! Return the host memory to write to
  void* destination() const { return hostPtr_; }

  //! Return the origin of the region to read
  const Coord3D& origin() const { return origin_; }
  //! Return the size of the region to read
  const Coord3D& size() const { return size_; }
  //! Return the row pitch
  size_t rowPitch() const { return rowPitch_; }
  //! Return the slice pitch
  size_t slicePitch() const { return slicePitch_; }

  //! Return the buffer rectangle information
  const BufferRect& bufRect() const { return bufRect_; }
  //! Return the host rectangle information
  const BufferRect& hostRect() const { return hostRect_; }

  //! Return true if the entire memory object is read.
  bool isEntireMemory() const;
};

/*! \brief      A generic write memory command.
 *
 *  \details    Used for operations on both buffers and images. Backends
 *              are expected to handle any required translations. Buffers
 *              are treated as 1D structures so origin_[0] and size_[0]
 *              are equivalent to offset_ and count_ respectively.
 */

class WriteMemoryCommand : public OneMemoryArgCommand {
 private:
  Coord3D origin_;       //!< Origin of the region to write to.
  Coord3D size_;         //!< Size of the region to write to.
  const void* hostPtr_;  //!< The host pointer source.
  size_t rowPitch_;      //!< Row pitch (for image operations)
  size_t slicePitch_;    //!< Slice pitch (for image operations)

  BufferRect bufRect_;   //!< Buffer rectangle information
  BufferRect hostRect_;  //!< Host memory rectangle information

 public:
  WriteMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                     Memory& memory, Coord3D origin, Coord3D size, const void* hostPtr,
                     size_t rowPitch = 0, size_t slicePitch = 0)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(rowPitch),
        slicePitch_(slicePitch) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  WriteMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                     Memory& memory, Coord3D origin, Coord3D size, const void* hostPtr,
                     const BufferRect& bufRect, const BufferRect& hostRect)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(0),
        slicePitch_(0),
        bufRect_(bufRect),
        hostRect_(hostRect) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitWriteMemory(*this); }

  //! Return the host memory to read from
  const void* source() const { return hostPtr_; }
  //! Return the memory object to write to.
  Memory& destination() const { return *memory_; }

  //! Return the region origin
  const Coord3D& origin() const { return origin_; }
  //! Return the region size
  const Coord3D& size() const { return size_; }
  //! Return the row pitch
  size_t rowPitch() const { return rowPitch_; }
  //! Return the slice pitch
  size_t slicePitch() const { return slicePitch_; }

  //! Return the buffer rectangle information
  const BufferRect& bufRect() const { return bufRect_; }
  //! Return the host rectangle information
  const BufferRect& hostRect() const { return hostRect_; }

  //! Return true if the entire memory object is written.
  bool isEntireMemory() const;
};

/*! \brief      A generic fill memory command.
 *
 *  \details    Used for operations on both buffers and images. Backends
 *              are expected to handle any required translations. Buffers
 *              are treated as 1D structures so origin_[0] and size_[0]
 *              are equivalent to offset_ and count_ respectively.
 */

class FillMemoryCommand : public OneMemoryArgCommand {
 public:
  const static size_t MaxFillPatterSize = sizeof(double[16]);

 private:
  Coord3D origin_;                   //!< Origin of the region to write to.
  Coord3D size_;                     //!< Size of the region to write to.
  char pattern_[MaxFillPatterSize];  //!< The fill pattern
  size_t patternSize_;               //!< Pattern size

 public:
  FillMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& memory, const void* pattern, size_t patternSize, Coord3D origin,
                    Coord3D size)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        patternSize_(patternSize) {
    // Sanity checks
    assert(pattern != NULL && "pattern cannot be null");
    assert(size.c[0] > 0 && "invalid");
    memcpy(pattern_, pattern, patternSize);
  }

  virtual void submit(device::VirtualDevice& device) { device.submitFillMemory(*this); }

  //! Return the pattern memory to fill with
  const void* pattern() const { return reinterpret_cast<const void*>(pattern_); }
  //! Return the pattern size
  const size_t patternSize() const { return patternSize_; }
  //! Return the memory object to write to.
  Memory& memory() const { return *memory_; }

  //! Return the region origin
  const Coord3D& origin() const { return origin_; }
  //! Return the region size
  const Coord3D& size() const { return size_; }

  //! Return true if the entire memory object is written.
  bool isEntireMemory() const;
};

/*! \brief      A generic copy memory command
 *
 *  \details    Used for both buffers and images. Backends are expected
 *              to handle any required translation. Buffers are treated
 *              as 1D structures so origin_[0] and size_[0] are
 *              equivalent to offset_ and count_ respectively.
 */

class CopyMemoryCommand : public TwoMemoryArgsCommand {
 private:
  Coord3D srcOrigin_;  //!< Origin of the source region.
  Coord3D dstOrigin_;  //!< Origin of the destination region.
  Coord3D size_;       //!< Size of the region to copy.

  BufferRect srcRect_;  //!< Source buffer rectangle information
  BufferRect dstRect_;  //!< Destination buffer rectangle information

 public:
  CopyMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& srcMemory, Memory& dstMemory, Coord3D srcOrigin, Coord3D dstOrigin,
                    Coord3D size)
      : TwoMemoryArgsCommand(queue, cmdType, eventWaitList, srcMemory, dstMemory),
        srcOrigin_(srcOrigin),
        dstOrigin_(dstOrigin),
        size_(size) {
    // Sanity checks
    assert(size.c[0] > 0 && "invalid");
  }

  CopyMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& srcMemory, Memory& dstMemory, Coord3D srcOrigin, Coord3D dstOrigin,
                    Coord3D size, const BufferRect& srcRect, const BufferRect& dstRect)
      : TwoMemoryArgsCommand(queue, cmdType, eventWaitList, srcMemory, dstMemory),
        srcOrigin_(srcOrigin),
        dstOrigin_(dstOrigin),
        size_(size),
        srcRect_(srcRect),
        dstRect_(dstRect) {
    // Sanity checks
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitCopyMemory(*this); }

  //! Return the host memory to read from
  Memory& source() const { return *memory1_; }
  //! Return the memory object to write to.
  Memory& destination() const { return *memory2_; }

  //! Return the source origin
  const Coord3D& srcOrigin() const { return srcOrigin_; }
  //! Return the offset in bytes in the destination.
  const Coord3D& dstOrigin() const { return dstOrigin_; }
  //! Return the number of bytes to copy.
  const Coord3D& size() const { return size_; }

  //! Return the source buffer rectangle information
  const BufferRect& srcRect() const { return srcRect_; }
  //! Return the destination buffer rectangle information
  const BufferRect& dstRect() const { return dstRect_; }

  //! Return true if the both memories are is read/written in their entirety.
  bool isEntireMemory() const;
};

/*! \brief  A generic map memory command. Makes a memory object accessible to the host.
 *
 * @todo:dgladdin   Need to think more about how the pitch parameters operate in
 *                  the context of unified buffer/image commands.
 */

class MapMemoryCommand : public OneMemoryArgCommand {
 private:
  cl_map_flags mapFlags_;  //!< Flags controlling the map.
  bool blocking_;          //!< True for blocking maps
  Coord3D origin_;         //!< Origin of the region to map.
  Coord3D size_;           //!< Size of the region to map.
  const void* mapPtr_;     //!< Host-space pointer that the object is currently mapped at

 public:
  //! Construct a new MapMemoryCommand
  MapMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                   Memory& memory, cl_map_flags mapFlags, bool blocking, Coord3D origin,
                   Coord3D size, size_t* imgRowPitch = nullptr, size_t* imgSlicePitch = nullptr,
                   void* mapPtr = nullptr)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        mapFlags_(mapFlags),
        blocking_(blocking),
        origin_(origin),
        size_(size),
        mapPtr_(mapPtr) {
    // Sanity checks
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitMapMemory(*this); }

  //! Read the memory object
  Memory& memory() const { return *memory_; }
  //! Read the map control flags
  cl_map_flags mapFlags() const { return mapFlags_; }
  //! Read the origin
  const Coord3D& origin() const { return origin_; }
  //! Read the size
  const Coord3D& size() const { return size_; }
  //! Read the blocking flag
  bool blocking() const { return blocking_; }
  //! Returns true if the entire memory object is mapped
  bool isEntireMemory() const;
  //! Read the map pointer
  const void* mapPtr() const { return mapPtr_; }
};


/*! \brief  A generic unmap memory command.
 *
 * @todo:dgladdin   Need to think more about how the pitch parameters operate in
 *                  the context of unified buffer/image commands.
 */

class UnmapMemoryCommand : public OneMemoryArgCommand {
 private:
  //! Host-space pointer that the object is currently mapped at
  void* mapPtr_;

 public:
  //! Construct a new MapMemoryCommand
  UnmapMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                     Memory& memory, void* mapPtr)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory), mapPtr_(mapPtr) {}

  virtual void submit(device::VirtualDevice& device) { device.submitUnmapMemory(*this); }

  virtual void releaseResources();

  //! Read the memory object
  Memory& memory() const { return *memory_; }
  //! Read the map pointer
  void* mapPtr() const { return mapPtr_; }
};

/*! \brief      Migrate memory objects command.
 *
 *  \details    Used for operations on both buffers and images. Backends
 *              are expected to handle any required translations.
 */
class MigrateMemObjectsCommand : public Command {
 private:
  cl_mem_migration_flags migrationFlags_;  //!< Migration flags
  std::vector<amd::Memory*> memObjects_;   //!< The list of memory objects

 public:
  //! Construct a new AcquireExtObjectsCommand
  MigrateMemObjectsCommand(HostQueue& queue, cl_command_type type,
                           const EventWaitList& eventWaitList,
                           const std::vector<amd::Memory*>& memObjects,
                           cl_mem_migration_flags flags)
      : Command(queue, type, eventWaitList), migrationFlags_(flags) {
    for (const auto& it : memObjects) {
      it->retain();
      memObjects_.push_back(it);
    }
  }

  virtual void submit(device::VirtualDevice& device) { device.submitMigrateMemObjects(*this); }

  //! Release all resources associated with this command
  void releaseResources() {
    for (const auto& it : memObjects_) {
      it->release();
    }
    Command::releaseResources();
  }

  //! Returns the migration flags
  cl_mem_migration_flags migrationFlags() const { return migrationFlags_; }
  //! Returns the number of memory objects in the command
  uint32_t numMemObjects() const { return (uint32_t)memObjects_.size(); }
  //! Returns a pointer to the memory objects
  const std::vector<amd::Memory*>& memObjects() const { return memObjects_; }

  bool validateMemory();
};

//! To execute a kernel on a specific device.
class NDRangeKernelCommand : public Command {
 private:
  Kernel& kernel_;
  NDRangeContainer sizes_;
  address parameters_;      //!< Pointer to the kernel argumets
  // The below fields are specific to the HIP functionality
  uint32_t sharedMemBytes_; //!< Size of reserved shared memory
  uint32_t extraParam_;     //!< Extra flags for the kernel launch
  uint32_t gridId_;         //!< Grid ID in the multi GPU kernel launch
  uint32_t numGrids_;       //!< Total number of grids in multi GPU launch
  uint64_t prevGridSum_;    //!< A sum of previous grids to the current launch
  uint64_t allGridSum_;     //!< A sum of all grids in multi GPU launch
  uint32_t firstDevice_;    //!< Device index of the first device in the grid

 public:
  enum {
    CooperativeGroups = 0x01,
    CooperativeMultiDeviceGroups = 0x02,
  };

  //! Construct an ExecuteKernel command
  NDRangeKernelCommand(HostQueue& queue, const EventWaitList& eventWaitList, Kernel& kernel,
                       const NDRangeContainer& sizes, uint32_t sharedMemBytes = 0,
                       uint32_t extraParam = 0, uint32_t gridId = 0, uint32_t numGrids = 0,
                       uint64_t prevGridSum = 0, uint64_t allGridSum = 0, uint32_t firstDevice = 0);

  virtual void submit(device::VirtualDevice& device) { device.submitKernel(*this); }

  //! Release all resources associated with this command (
  void releaseResources();

  //! Return the kernel.
  const Kernel& kernel() const { return kernel_; }

  //! Return the parameters given to this kernel.
  const_address parameters() const { return parameters_; }

  //! Return the kernel NDRange.
  const NDRangeContainer& sizes() const { return sizes_; }

  //! Return the shared memory size
  uint32_t sharedMemBytes() const { return sharedMemBytes_; }

  //! Return the cooperative groups mode
  bool cooperativeGroups() const { return (extraParam_ & CooperativeGroups) ? true : false; }

  //! Return the cooperative multi device groups mode
  bool cooperativeMultiDeviceGroups() const {
    return (extraParam_ & CooperativeMultiDeviceGroups) ? true : false;
  }

  //! Return the current grid ID for multidevice launch
  uint32_t gridId() const { return gridId_; }

  //! Return the number of launched grids
  uint32_t numGrids() const { return numGrids_; }

  //! Return the total workload size for up to the current
  uint64_t prevGridSum() const { return prevGridSum_; }

  //! Return the total workload size for all GPUs
  uint64_t allGridSum() const { return allGridSum_; }

  //! Return the index of the first device in multi GPU launch
  uint64_t firstDevice() const { return firstDevice_; }

  //! Set the local work size.
  void setLocalWorkSize(const NDRange& local) { sizes_.local() = local; }

  int32_t captureAndValidate();
};

class NativeFnCommand : public Command {
 private:
  void(CL_CALLBACK* nativeFn_)(void*);

  char* args_;
  size_t argsSize_;

  std::vector<Memory*> memObjects_;
  std::vector<size_t> memOffsets_;

 public:
  NativeFnCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                  void(CL_CALLBACK* nativeFn)(void*), const void* args, size_t argsSize,
                  size_t numMemObjs, const cl_mem* memObjs, const void** memLocs);

  ~NativeFnCommand() { delete[] args_; }

  void releaseResources() {
    std::for_each(memObjects_.begin(), memObjects_.end(), std::mem_fun(&Memory::release));
    Command::releaseResources();
  }

  virtual void submit(device::VirtualDevice& device) { device.submitNativeFn(*this); }

  int32_t invoke();
};

class Marker : public Command {
 public:
  //! Create a new Marker
  Marker(HostQueue& queue, bool userVisible, const EventWaitList& eventWaitList = nullWaitList,
         const Event* waitingEvent = NULL)
      : Command(queue, userVisible ? CL_COMMAND_MARKER : 0, eventWaitList),
        waitingEvent_(waitingEvent) {}

  //! The actual command implementation.
  virtual void submit(device::VirtualDevice& device) { device.submitMarker(*this); }

  const Event* waitingEvent() const { return waitingEvent_; }

 private:
  const Event* waitingEvent_;  //!< Waiting event associated with the marker
};

/*! \brief  Maps CL objects created from external ones and syncs the contents (blocking).
 *
 */

class ExtObjectsCommand : public Command {
 private:
  std::vector<amd::Memory*> memObjects_;  //!< The list of Memory based classes

 public:
  //! Construct a new AcquireExtObjectsCommand
  ExtObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList, uint32_t num_objects,
                    const std::vector<amd::Memory*>& memoryObjects, cl_command_type type)
      : Command(queue, type, eventWaitList) {
    for (const auto& it : memoryObjects) {
      it->retain();
      memObjects_.push_back(it);
    }
  }

  //! Release all resources associated with this command
  void releaseResources() {
    for (const auto& it : memObjects_) {
      it->release();
    }
    Command::releaseResources();
  }

  //! Get number of GL objects
  uint32_t getNumObjects() { return (uint32_t)memObjects_.size(); }
  //! Get pointer to GL object list
  const std::vector<amd::Memory*>& getMemList() const { return memObjects_; }
  bool validateMemory();
  virtual bool processGLResource(device::Memory* mem) = 0;
};

class AcquireExtObjectsCommand : public ExtObjectsCommand {
 public:
  //! Construct a new AcquireExtObjectsCommand
  AcquireExtObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                           uint32_t num_objects, const std::vector<amd::Memory*>& memoryObjects,
                           cl_command_type type)
      : ExtObjectsCommand(queue, eventWaitList, num_objects, memoryObjects, type) {}

  virtual void submit(device::VirtualDevice& device) { device.submitAcquireExtObjects(*this); }

  virtual bool processGLResource(device::Memory* mem);
};

class ReleaseExtObjectsCommand : public ExtObjectsCommand {
 public:
  //! Construct a new ReleaseExtObjectsCommand
  ReleaseExtObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                           uint32_t num_objects, const std::vector<amd::Memory*>& memoryObjects,
                           cl_command_type type)
      : ExtObjectsCommand(queue, eventWaitList, num_objects, memoryObjects, type) {}

  virtual void submit(device::VirtualDevice& device) { device.submitReleaseExtObjects(*this); }

  virtual bool processGLResource(device::Memory* mem);
};

class PerfCounterCommand : public Command {
 public:
  typedef std::vector<PerfCounter*> PerfCounterList;

  enum State {
    Begin = 0,  //!< Issue a begin command
    End = 1     //!< Issue an end command
  };

  //! Construct a new PerfCounterCommand
  PerfCounterCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                     const PerfCounterList& counterList, State state)
      : Command(queue, 1, eventWaitList), counterList_(counterList), state_(state) {
    for (uint i = 0; i < counterList_.size(); ++i) {
      counterList_[i]->retain();
    }
  }

  void releaseResources() {
    for (uint i = 0; i < counterList_.size(); ++i) {
      counterList_[i]->release();
    }
    Command::releaseResources();
  }

  //! Gets the number of PerfCounter objects
  size_t getNumCounters() const { return counterList_.size(); }

  //! Gets the list of all counters
  const PerfCounterList& getCounters() const { return counterList_; }

  //! Gets the performance counter state
  State getState() const { return state_; }

  //! Process the command on the device queue
  virtual void submit(device::VirtualDevice& device) { device.submitPerfCounter(*this); }

 private:
  PerfCounterList counterList_;  //!< The list of performance counters
  State state_;                  //!< State of the issued command
};

/*! \brief      Thread Trace memory objects command.
 *
 *  \details    Used for bindig memory objects to therad trace mechanism.
 */
class ThreadTraceMemObjectsCommand : public Command {
 public:
  //! Construct a new ThreadTraceMemObjectsCommand
  ThreadTraceMemObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                               size_t numMemoryObjects, const cl_mem* memoryObjects,
                               size_t sizeMemoryObject, ThreadTrace& threadTrace,
                               cl_command_type type)
      : Command(queue, type, eventWaitList),
        sizeMemObjects_(sizeMemoryObject),
        threadTrace_(threadTrace) {
    memObjects_.resize(numMemoryObjects);
    for (size_t i = 0; i < numMemoryObjects; ++i) {
      Memory* obj = as_amd(memoryObjects[i]);
      obj->retain();
      memObjects_[i] = obj;
    }
    threadTrace_.retain();
  }
  //! Release all resources associated with this command
  void releaseResources() {
    threadTrace_.release();
    for (const auto& itr : memObjects_) {
      itr->release();
    }
    Command::releaseResources();
  }

  //! Get number of CL memory objects
  uint32_t getNumObjects() { return (uint32_t)memObjects_.size(); }

  //! Get pointer to CL memory object list
  const std::vector<amd::Memory*>& getMemList() const { return memObjects_; }

  //! Submit command to bind memory object to the Thread Trace mechanism
  virtual void submit(device::VirtualDevice& device) { device.submitThreadTraceMemObjects(*this); }

  //! Return the thread trace object.
  ThreadTrace& getThreadTrace() const { return threadTrace_; }

  //! Get memory object size
  const size_t getMemoryObjectSize() const { return sizeMemObjects_; }

  //! Validate memory bound to the thread thrace
  bool validateMemory();

 private:
  std::vector<amd::Memory*> memObjects_;  //!< The list of memory objects,bound to the thread trace
  size_t sizeMemObjects_;     //!< The size of each memory object from memObjects_ list (all memory
                              //! objects have the smae size)
  ThreadTrace& threadTrace_;  //!< The Thread Trace object
};

/*! \brief      Thread Trace command.
 *
 *  \details    Used for issue begin/end/pause/resume for therad trace object.
 */
class ThreadTraceCommand : public Command {
 private:
  void* threadTraceConfig_;

 public:
  enum State {
    Begin = 0,  //!< Issue a begin command
    End = 1,    //!< Issue an end command
    Pause = 2,  //!< Issue a pause command
    Resume = 3  //!< Issue a resume command
  };

  //! Construct a new ThreadTraceCommand
  ThreadTraceCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                     const void* threadTraceConfig, ThreadTrace& threadTrace, State state,
                     cl_command_type type)
      : Command(queue, type, eventWaitList), threadTrace_(threadTrace), state_(state) {
    const unsigned int size = *static_cast<const unsigned int*>(threadTraceConfig);
    threadTraceConfig_ = static_cast<void*>(new char[size]);
    if (threadTraceConfig_) {
      memcpy(threadTraceConfig_, threadTraceConfig, size);
    }
    threadTrace_.retain();
  }

  //! Release all resources associated with this command
  void releaseResources() {
    threadTrace_.release();
    Command::releaseResources();
  }

  //! Get the thread trace object
  ThreadTrace& getThreadTrace() const { return threadTrace_; }

  //! Get the thread trace command state
  State getState() const { return state_; }

  //! Process the command on the device queue
  virtual void submit(device::VirtualDevice& device) { device.submitThreadTrace(*this); }
  // Accessor methods
  void* threadTraceConfig() const { return threadTraceConfig_; }

 private:
  ThreadTrace& threadTrace_;  //!< The list of performance counters
  State state_;               //!< State of the issued command
};

class SignalCommand : public OneMemoryArgCommand {
 private:
  uint32_t markerValue_;
  uint64_t markerOffset_;

 public:
  SignalCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                Memory& memory, uint32_t value, uint64_t offset = 0)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        markerValue_(value),
        markerOffset_(offset) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSignal(*this); }

  const uint32_t markerValue() { return markerValue_; }
  Memory& memory() { return *memory_; }
  const uint64_t markerOffset() { return markerOffset_; }
};

class MakeBuffersResidentCommand : public Command {
 private:
  std::vector<amd::Memory*> memObjects_;
  cl_bus_address_amd* busAddresses_;

 public:
  MakeBuffersResidentCommand(HostQueue& queue, cl_command_type type,
                             const EventWaitList& eventWaitList,
                             const std::vector<amd::Memory*>& memObjects,
                             cl_bus_address_amd* busAddr)
      : Command(queue, type, eventWaitList), busAddresses_(busAddr) {
    for (const auto& it : memObjects) {
      it->retain();
      memObjects_.push_back(it);
    }
  }

  virtual void submit(device::VirtualDevice& device) { device.submitMakeBuffersResident(*this); }

  void releaseResources() {
    for (const auto& it : memObjects_) {
      it->release();
    }
    Command::releaseResources();
  }

  bool validateMemory();
  const std::vector<amd::Memory*>& memObjects() const { return memObjects_; }
  cl_bus_address_amd* busAddress() const { return busAddresses_; }
};

//! A deallocation command used to free SVM or system pointers.
class SvmFreeMemoryCommand : public Command {
 public:
  typedef void(CL_CALLBACK* freeCallBack)(cl_command_queue, uint32_t, void**, void*);

 private:
  std::vector<void*> svmPointers_;  //!< List of pointers to deallocate
  freeCallBack pfnFreeFunc_;        //!< User-defined deallocation callback
  void* userData_;                  //!< Data passed to user-defined callback

 public:
  SvmFreeMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, uint32_t numSvmPointers,
                       void** svmPointers, freeCallBack pfnFreeFunc, void* userData)
      : Command(queue, CL_COMMAND_SVM_FREE, eventWaitList),
        //! We copy svmPointers since it can be reused/deallocated after
        //  command creation
        svmPointers_(svmPointers, svmPointers + numSvmPointers),
        pfnFreeFunc_(pfnFreeFunc),
        userData_(userData) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmFreeMemory(*this); }

  std::vector<void*>& svmPointers() { return svmPointers_; }

  freeCallBack pfnFreeFunc() const { return pfnFreeFunc_; }

  void* userData() const { return userData_; }
};

//! A copy command where the origin and destination memory locations are SVM
// pointers.
class SvmCopyMemoryCommand : public Command {
 private:
  void* dst_;        //!< Destination pointer
  const void* src_;  //!< Source pointer
  size_t srcSize_;   //!< Size (in bytes) of the source buffer

 public:
  SvmCopyMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, void* dst,
                       const void* src, size_t srcSize)
      : Command(queue, CL_COMMAND_SVM_MEMCPY, eventWaitList),
        dst_(dst),
        src_(src),
        srcSize_(srcSize) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmCopyMemory(*this); }

  void* dst() const { return dst_; }

  const void* src() const { return src_; }

  size_t srcSize() const { return srcSize_; }
};

//! A fill command where the pattern and destination memory locations are SVM
// pointers.
class SvmFillMemoryCommand : public Command {
 private:
  void* dst_;                                           //!< Destination pointer
  char pattern_[FillMemoryCommand::MaxFillPatterSize];  //!< The fill pattern
  size_t patternSize_;                                  //!< Pattern size
  size_t times_;                                        //!< Number of times to fill the
  //   destination buffer with the source buffer

 public:
  SvmFillMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, void* dst,
                       const void* pattern, size_t patternSize, size_t size)
      : Command(queue, CL_COMMAND_SVM_MEMFILL, eventWaitList),
        dst_(dst),
        patternSize_(patternSize),
        times_(size / patternSize) {
    assert(amd::isMultipleOf(size, patternSize));
    //! We copy the pattern buffer since it can be reused/deallocated after
    //  command creation
    memcpy(pattern_, pattern, patternSize);
  }

  virtual void submit(device::VirtualDevice& device) { device.submitSvmFillMemory(*this); }

  void* dst() const { return dst_; }

  const char* pattern() const { return pattern_; }

  size_t patternSize() const { return patternSize_; }

  size_t times() const { return times_; }
};

/*! \brief A map memory command where the pointer to be mapped is a SVM shared
 * buffer
 */
class SvmMapMemoryCommand : public Command {
 private:
  Memory* svmMem_;  //!< the pointer to the amd::Memory object corresponding the svm pointer mapped
  Coord3D size_;    //!< the map size
  Coord3D origin_;  //!< the origin of the mapped svm pointer shift from the beginning of svm space
                    //! allocated
  cl_map_flags flags_;  //!< map flags
  void* svmPtr_;

 public:
  SvmMapMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, Memory* svmMem,
                      const size_t size, const size_t offset, cl_map_flags flags, void* svmPtr)
      : Command(queue, CL_COMMAND_SVM_MAP, eventWaitList),
        svmMem_(svmMem),
        size_(size),
        origin_(offset),
        flags_(flags),
        svmPtr_(svmPtr) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmMapMemory(*this); }

  Memory* getSvmMem() const { return svmMem_; }

  Coord3D size() const { return size_; }

  cl_map_flags mapFlags() const { return flags_; }

  Coord3D origin() const { return origin_; }

  void* svmPtr() const { return svmPtr_; }

  bool isEntireMemory() const;
};

/*! \brief An unmap memory command where the unmapped pointer is a SVM shared
 * buffer
 */
class SvmUnmapMemoryCommand : public Command {
 private:
  Memory* svmMem_;  //!< the pointer to the amd::Memory object corresponding the svm pointer mapped
  void* svmPtr_;    //!< SVM pointer

 public:
  SvmUnmapMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, Memory* svmMem,
                        void* svmPtr)
      : Command(queue, CL_COMMAND_SVM_UNMAP, eventWaitList), svmMem_(svmMem), svmPtr_(svmPtr) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmUnmapMemory(*this); }

  Memory* getSvmMem() const { return svmMem_; }

  void* svmPtr() const { return svmPtr_; }
};

/*! \brief      A generic transfer memory from/to file command.
 *
 *  \details    Currently supports buffers only. Buffers
 *              are treated as 1D structures so origin_[0] and size_[0]
 *              are equivalent to offset_ and count_ respectively.
 */
class TransferBufferFileCommand : public OneMemoryArgCommand {
 public:
  static const uint NumStagingBuffers = 2;
  static const size_t StagingBufferSize = 4 * Mi;
  static const uint StagingBufferMemType = CL_MEM_USE_PERSISTENT_MEM_AMD;

 protected:
  const Coord3D origin_;                     //!< Origin of the region to write to
  const Coord3D size_;                       //!< Size of the region to write to
  LiquidFlashFile* file_;                    //!< The file object for data read
  size_t fileOffset_;                        //!< Offset in the file for data read
  amd::Memory* staging_[NumStagingBuffers];  //!< Staging buffers for transfer

 public:
  TransferBufferFileCommand(cl_command_type type, HostQueue& queue,
                            const EventWaitList& eventWaitList, Memory& memory,
                            const Coord3D& origin, const Coord3D& size, LiquidFlashFile* file,
                            size_t fileOffset)
      : OneMemoryArgCommand(queue, type, eventWaitList, memory),
        origin_(origin),
        size_(size),
        file_(file),
        fileOffset_(fileOffset) {
    // Sanity checks
    assert(size.c[0] > 0 && "invalid");
    for (uint i = 0; i < NumStagingBuffers; ++i) {
      staging_[i] = NULL;
    }
  }

  virtual void releaseResources();

  virtual void submit(device::VirtualDevice& device);

  //! Return the memory object to write to
  Memory& memory() const { return *memory_; }

  //! Return the host memory to read from
  LiquidFlashFile* file() const { return file_; }

  //! Returns file offset
  size_t fileOffset() const { return fileOffset_; }

  //! Return the region origin
  const Coord3D& origin() const { return origin_; }
  //! Return the region size
  const Coord3D& size() const { return size_; }

  //! Return the staging buffer for transfer
  Memory& staging(uint i) const { return *staging_[i]; }

  bool validateMemory();
};

/*! \brief      A P2P copy memory command
 *
 *  \details    Used for buffers only. Backends are expected
 *              to handle any required translation. Buffers are treated
 *              as 1D structures so origin_[0] and size_[0] are
 *              equivalent to offset_ and count_ respectively.
 */

class CopyMemoryP2PCommand : public CopyMemoryCommand {
 public:
  CopyMemoryP2PCommand(HostQueue& queue, cl_command_type cmdType,
                       const EventWaitList& eventWaitList, Memory& srcMemory, Memory& dstMemory,
                       Coord3D srcOrigin, Coord3D dstOrigin, Coord3D size)
      : CopyMemoryCommand(queue, cmdType, eventWaitList, srcMemory, dstMemory, srcOrigin, dstOrigin,
                          size) {}

  virtual void submit(device::VirtualDevice& device) { device.submitCopyMemoryP2P(*this); }

  bool validateMemory();
};

/*! @}
 *  @}
 */

}  // namespace amd

#endif /*COMMAND_HPP_*/
