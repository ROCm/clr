/* Copyright (c) 2021-present Advanced Micro Devices, Inc.

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

#include "device/device.hpp"
#include "palsignal.hpp"
#include "paldevice.hpp"
#include "os/os.hpp"

#include <functional>

namespace pal {

Signal::~Signal() {
  dev_->context().svmFree(amdSignal_);
}

bool Signal::Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) {
  dev_ = static_cast<const pal::Device*>(&dev);
  ws_ = ws;

  void* buffer = dev_->context().svmAlloc(sizeof(amd_signal_t), alignof(amd_signal_t),
                                          CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS);
  if (!buffer) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
            "Failed to create amd_signal_t buffer");
    return false;
  }
  std::memset(buffer, 0, sizeof(amd_signal_t));

  amdSignal_ = new (buffer) amd_signal_t();
  amdSignal_->value = init;

  return true;
}

uint64_t Signal::Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) {
  auto cmp = [](device::Signal::Condition c) -> std::function<bool(uint64_t, uint64_t)> {
    switch (c) {
      case device::Signal::Condition::Eq:
        return [](auto ls, auto rs) { return ls == rs; };
      case device::Signal::Condition::Ne:
        return [](auto ls, auto rs) { return ls != rs; };
      case device::Signal::Condition::Lt:
        return [](auto ls, auto rs) { return ls < rs; };
      case device::Signal::Condition::Gte:
        return [](auto ls, auto rs) { return ls >= rs; };
    };
    ShouldNotReachHere();
    return [](auto ls, auto rs) { return false; };
  } (c);

  if (ws_ == device::Signal::WaitState::Blocked) {
    guarantee(false, "Unimplemented");
  } else if (ws_ == device::Signal::WaitState::Active) {
    auto start = amd::Os::timeNanos();
    while (true) {
      auto end = amd::Os::timeNanos();
      auto duration = 1000 * (end - start); // convert to us
      if (duration >= timeout) {
        return -1;
      }

      if (!cmp(amdSignal_->value, value)) {
        amd::Os::yield();
        continue;
      }

      std::atomic_thread_fence(std::memory_order_acquire);
      return amdSignal_->value;
    }
  }

  return -1;
}

void Signal::Reset(uint64_t value) {
  amdSignal_->value = value;
}

};
