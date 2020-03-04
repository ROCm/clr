/* Copyright (c) 2019-present Advanced Micro Devices, Inc.

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

#include "utils/debug.hpp"
#include "top.hpp"
#include "utils/flags.hpp"

#include "rochostcall.hpp"
#include "rochcmessages.hpp"

#include "os/os.hpp"
#include "thread/monitor.hpp"
#include "utils/util.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"

#include "rochostcall.hpp"

#include <hsa.h>

#include <assert.h>
#include <string.h>
#include <set>

namespace {  // anonymous

enum SignalValue { SIGNAL_DONE = 0, SIGNAL_INIT = 1 };

/** \brief Packet payload
 *
 *  Contains 64 slots of 8 ulongs each, one for each workitem in the
 *  wave. A slot with index \c i contains valid data if the
 *  corresponding bit in PacketHeader::activemask is set.
 */
struct Payload {
  uint64_t slots[64][8];
};

/** Packet header */
struct PacketHeader {
  /** Tagged pointer to the next packet in an intrusive stack */
  uint64_t next_;
  /** Bitmask that represents payload slots with valid data */
  uint64_t activemask_;
  /** Service ID requested by the wave */
  uint32_t service_;
  /** Control bits.
   *  \li 0: \c READY flag. Indicates packet awaiting a host response.
   */
  uint32_t control_;
};

static_assert(std::is_standard_layout<PacketHeader>::value,
              "the hostcall packet must be useable from other languages");

/** Field offsets in the packet control field */
enum ControlOffset {
  CONTROL_OFFSET_READY_FLAG = 0,
  CONTROL_OFFSET_RESERVED0 = 1,
};

/** Field widths in the packet control field */
enum ControlWidth {
  CONTROL_WIDTH_READY_FLAG = 1,
  CONTROL_WIDTH_RESERVED0 = 31,
};

/** \brief Shared buffer submitting hostcall requests.
 *
 *  Holds hostcall packets requested by all kernels executing on the
 *  same device queue. Each hostcall buffer is associated with at most
 *  one device queue.
 *
 *  Packets in the buffer are accessed using 64-bit tagged pointers to mitigate
 *  the ABA problem in lock-free stacks. The index_mask is used to extract the
 *  lower bits of the pointer, which form the index into the packet array. The
 *  remaining higher bits define a tag that is incremented on every pop from a
 *  stack.
 */
class HostcallBuffer {
  /** Array of packet headers */
  PacketHeader* headers_;
  /** Array of packet payloads */
  Payload* payloads_;
  /** Signal used by kernels to indicate new work */
  hsa_signal_t doorbell_;
  /** Stack of free packets. Uses tagged pointers. */
  uint64_t free_stack_;
  /** Stack of ready packets. Uses tagged pointers */
  uint64_t ready_stack_;
  /** Mask for accessing the packet index in the tagged pointer. */
  uint64_t index_mask_;

  PacketHeader* getHeader(uint64_t ptr) const;
  Payload* getPayload(uint64_t ptr) const;

 public:
  void processPackets(MessageHandler& messages);
  void initialize(uint32_t num_packets);
  void setDoorbell(hsa_signal_t doorbell) { doorbell_ = doorbell; };
};

static_assert(std::is_standard_layout<HostcallBuffer>::value,
              "the hostcall buffer must be useable from other languages");

};  // namespace

PacketHeader* HostcallBuffer::getHeader(uint64_t ptr) const {
  return headers_ + (ptr & index_mask_);
}

Payload* HostcallBuffer::getPayload(uint64_t ptr) const {
  return payloads_ + (ptr & index_mask_);
}

static uint32_t setControlField(uint32_t control, uint8_t offset, uint8_t width, uint32_t value) {
  uint32_t mask = ~(((1 << width) - 1) << offset);
  control &= mask;
  return control | (value << offset);
}

static uint32_t resetReadyFlag(uint32_t control) {
  return setControlField(control, CONTROL_OFFSET_READY_FLAG, CONTROL_WIDTH_READY_FLAG, 0);
}

/** \brief Signature for pointer accepted by the function call service.
 *  \param output Pointer to output arguments.
 *  \param input Pointer to input arguments.
 *
 *  The function can accept up to seven 64-bit arguments via the
 *  #input pointer, and can produce up to two 64-bit arguments via the
 *  #output pointer. The contents of these arguments are defined by
 *  the function being invoked.
 */
typedef void (*HostcallFunctionCall)(uint64_t* output, const uint64_t* input);

static void handlePayload(MessageHandler& messages, uint32_t service, uint64_t* payload) {
  switch (service) {
    case SERVICE_FUNCTION_CALL: {
      uint64_t output[2];
      auto fptr = reinterpret_cast<HostcallFunctionCall>(payload[0]);
      fptr(output, payload + 1);
      memcpy(payload, output, sizeof(output));
      return;
    }
    case SERVICE_PRINTF:
      if (!messages.handlePayload(service, payload)) {
        ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "Hostcall: invalid request for service \"%d\".",
                service);
        amd::report_fatal(__FILE__, __LINE__, "Hostcall: invalid service request.");
      }
      return;
    default:
      ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "Hostcall: no handler found for service ID \"%d\".",
              service);
      amd::report_fatal(__FILE__, __LINE__, "Hostcall service not supported.");
      return;
  }
}

void HostcallBuffer::processPackets(MessageHandler& messages) {
  // Grab the entire ready stack and set the top to 0. New requests from the
  // device will continue pushing on the stack while we process the packets that
  // we have grabbed.
  uint64_t ready_stack = __atomic_exchange_n(&ready_stack_, 0, std::memory_order_acquire);
  if (!ready_stack) {
    return;
  }

  // Each wave can submit at most one packet at a time. The ready stack cannot
  // contain multiple packets from the same wave, so consuming ready packets in
  // a latest-first order does not affect ordering of hostcall within a wave.
  for (decltype(ready_stack) iter = ready_stack, next = 0; iter; iter = next) {
    auto header = getHeader(iter);
    // Remember the next packet pointer, because we will no longer own the
    // current packet at the end of this loop.
    next = header->next_;

    auto service = header->service_;
    auto payload = getPayload(iter);
    auto activemask = header->activemask_;
    while (activemask) {
      auto wi = amd::leastBitSet(activemask);
      activemask ^= static_cast<decltype(activemask)>(1) << wi;
      auto slot = payload->slots[wi];
      handlePayload(messages, service, slot);
    }

    __atomic_store_n(&header->control_, resetReadyFlag(header->control_),
                     std::memory_order_release);
  }
}

static uintptr_t getHeaderStart() {
  return amd::alignUp(sizeof(HostcallBuffer), alignof(PacketHeader));
}

static uintptr_t getPayloadStart(uint32_t num_packets) {
  auto header_start = getHeaderStart();
  auto header_end = header_start + sizeof(PacketHeader) * num_packets;
  return amd::alignUp(header_end, alignof(Payload));
}

size_t getHostcallBufferSize(uint32_t num_packets) {
  size_t buffer_size = getPayloadStart(num_packets);
  buffer_size += num_packets * sizeof(Payload);
  return buffer_size;
}

uint32_t getHostcallBufferAlignment() { return alignof(Payload); }

static uint64_t getIndexMask(uint32_t num_packets) {
  // The number of packets is at least equal to the maximum number of waves
  // supported by the device. That means we do not need to account for the
  // border cases where num_packets is zero or one.
  assert(num_packets > 1);
  if (!amd::isPowerOfTwo(num_packets)) {
    num_packets = amd::nextPowerOfTwo(num_packets);
  }
  return num_packets - 1;
}

void HostcallBuffer::initialize(uint32_t num_packets) {
  auto base = reinterpret_cast<uint8_t*>(this);
  headers_ = reinterpret_cast<PacketHeader*>((base + getHeaderStart()));
  payloads_ = reinterpret_cast<Payload*>((base + getPayloadStart(num_packets)));
  index_mask_ = getIndexMask(num_packets);

  // The null pointer is identical to (uint64_t)0. When using tagged pointers,
  // the tag and the index part of the array must never be zero at the same
  // time. In the initialized free stack, headers[1].next points to headers[0],
  // which has index 0. We initialize this pointer to have a tag of 1.
  uint64_t next = index_mask_ + 1;

  // Initialize the free stack.
  headers_[0].next_ = 0;
  for (uint32_t ii = 1; ii != num_packets; ++ii) {
    headers_[ii].next_ = next;
    next = ii;
  }
  free_stack_ = next;
  ready_stack_ = 0;
}

/** \brief Manage a unique listener thread and its associated buffers.
 */
class HostcallListener {
  std::set<HostcallBuffer*> buffers_;
  hsa_signal_t doorbell_;
  MessageHandler messages_;

  class Thread : public amd::Thread {
   public:
    Thread() : amd::Thread("Hostcall Listener Thread", CQ_THREAD_STACK_SIZE) {}

    //! The hostcall listener thread entry point.
    void run(void* data) {
      auto listener = reinterpret_cast<HostcallListener*>(data);
      listener->consumePackets();
    }
  } thread_;  //!< The hostcall listener thread.

  void consumePackets();

 public:
  /** \brief Add a buffer to the listener.
   *
   *  Behaviour is undefined if:
   *  - hostcall_initialize_buffer() was not invoked successfully on
   *    the buffer prior to registration.
   *  - The same buffer is registered with multiple listeners.
   *  - The same buffer is associated with more than one hardware queue.
   */
  void addBuffer(HostcallBuffer* buffer);

  /** \brief Remove a buffer that is no longer in use.
   *
   *  The buffer can be reused after removal.  Behaviour is undefined if the
   *  buffer is freed without first removing it.
   */
  void removeBuffer(HostcallBuffer* buffer);

  /* \brief Return true if no buffers are registered.
  */
  bool idle() const {
    return buffers_.empty();
  }

  void terminate();
  bool initialize();
};

HostcallListener* hostcallListener = nullptr;
amd::Monitor listenerLock("Hostcall listener lock");

void HostcallListener::consumePackets() {
  uint64_t signal_value = SIGNAL_INIT;
  uint64_t timeout = 1024 * 1024;

  while (true) {
    while (true) {
      uint64_t new_value = hsa_signal_wait_acquire(doorbell_, HSA_SIGNAL_CONDITION_NE, signal_value, timeout,
                                                   HSA_WAIT_STATE_BLOCKED);
      if (new_value != signal_value) {
        signal_value = new_value;
        break;
      }
    }

    if (signal_value == SIGNAL_DONE) {
      ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Hostcall listener received SIGNAL_DONE");
      return;
    }

    amd::ScopedLock lock{listenerLock};

    for (auto ii : buffers_) {
      ii->processPackets(messages_);
    }
  }

  return;
}

void HostcallListener::terminate() {
  if (!amd::Os::isThreadAlive(thread_)) {
    return;
  }

  hsa_signal_store_release(doorbell_, SIGNAL_DONE);

  // FIXME_lmoriche: fix termination handshake
  while (thread_.state() < Thread::FINISHED) {
    amd::Os::yield();
  }

  hsa_signal_destroy(doorbell_);
}

void HostcallListener::addBuffer(HostcallBuffer* buffer) {
  assert(buffers_.count(buffer) == 0 && "buffer already present");
  buffer->setDoorbell(doorbell_);
  buffers_.insert(buffer);
}

void HostcallListener::removeBuffer(HostcallBuffer* buffer) {
  assert(buffers_.count(buffer) != 0 && "unknown buffer");
  buffers_.erase(buffer);
}

bool HostcallListener::initialize() {
  auto status = hsa_signal_create(SIGNAL_INIT, 0, NULL, &doorbell_);
  if (status != HSA_STATUS_SUCCESS) {
    return false;
  }

  // If the listener thread was not successfully initialized, clean
  // everything up and bail out.
  if (thread_.state() < Thread::INITIALIZED) {
    hsa_signal_destroy(doorbell_);
    return false;
  }

  thread_.start(this);
  return true;
}

bool enableHostcalls(void* bfr, uint32_t numPackets, const void* queue) {
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  buffer->initialize(numPackets);

  amd::ScopedLock lock(listenerLock);
  if (!hostcallListener) {
    hostcallListener = new HostcallListener();
    if (!hostcallListener->initialize()) {
      ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
              "Failed to launch hostcall listener");
      delete hostcallListener;
      hostcallListener = nullptr;
      return false;
    }
    ClPrint(amd::LOG_INFO, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
            "Launched hostcall listener at %p", hostcallListener);
  }
  hostcallListener->addBuffer(buffer);
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Registered hostcall buffer %p with listener %p", buffer,
          hostcallListener);
  return true;
}

void disableHostcalls(void* bfr, const void* queue) {
  amd::ScopedLock lock(listenerLock);
  if (!hostcallListener) {
    return;
  }
  assert(bfr && "expected a hostcall buffer");
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  hostcallListener->removeBuffer(buffer);

  if (hostcallListener->idle()) {
    hostcallListener->terminate();
    delete hostcallListener;
    hostcallListener = nullptr;
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Terminated hostcall listener");
  }
}
