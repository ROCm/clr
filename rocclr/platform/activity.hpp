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

#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#include "thread/monitor.hpp"

#define USE_PROF_API 1

#if USE_PROF_API
enum OpId { OP_ID_DISPATCH = 0, OP_ID_COPY = 1, OP_ID_BARRIER = 2, OP_ID_NUMBER = 3 };

#include "prof_protocol.h"

// Statically allocated table of callbacks and global unique ID of each operation
#define ACTIVITY_PROF_INSTANCES()                                                                  \
  namespace activity_prof {                                                                        \
  CallbacksTable::table_t CallbacksTable::table_{};                                                \
  std::atomic<record_id_t> ActivityProf::globe_record_id_(0);                                      \
  }  // activity_prof

namespace activity_prof {
typedef activity_correlation_id_t record_id_t;
typedef activity_op_t op_id_t;
typedef uint32_t command_id_t;

typedef activity_id_callback_t id_callback_fun_t;
typedef activity_async_callback_t callback_fun_t;
typedef void* callback_arg_t;

// Activity callbacks table
class CallbacksTable {
 public:
  struct table_t {
    id_callback_fun_t id_callback;
    callback_fun_t op_callback;
    callback_arg_t arg;
    std::atomic<bool> enabled[OP_ID_NUMBER];
  };

  // Initialize record id callback and activity callback
  static void init(const id_callback_fun_t& id_callback, const callback_fun_t& op_callback,
                   const callback_arg_t& arg) {
    table_.id_callback = id_callback;
    table_.op_callback = op_callback;
    table_.arg = arg;
  }

  static bool SetEnabled(const op_id_t& op_id, const bool& enable) {
    bool ret = true;
    if (op_id < OP_ID_NUMBER) {
      table_.enabled[op_id].store(enable, std::memory_order_release);
    } else {
      ret = false;
    }
    return ret;
  }

  static bool IsEnabled(const op_id_t& op_id) {
    return table_.enabled[op_id].load(std::memory_order_acquire);
  }

  static id_callback_fun_t get_id_callback() { return table_.id_callback; }
  static callback_fun_t get_op_callback() { return table_.op_callback; }
  static callback_arg_t get_arg() { return table_.arg; }

 private:
  static table_t table_;
};

// Activity profile class
class ActivityProf {
 public:
  // Domain ID
  static const int ACTIVITY_DOMAIN_ID = ACTIVITY_DOMAIN_HIP_VDI;

  ActivityProf() : command_id_(0), queue_id_(0), device_id_(0), record_id_(0), enabled_(false) {}

  // Initialization
  void Initialize(const command_id_t command_id, const uint32_t queue_id,
                  const uint32_t device_id) {
    activity_op_t op_id = (command_id == CL_COMMAND_NDRANGE_KERNEL) ? OP_ID_DISPATCH : OP_ID_COPY;
    enabled_ = CallbacksTable::IsEnabled(op_id);
    if (IsEnabled()) {
      command_id_ = command_id;
      queue_id_ = queue_id;
      device_id_ = device_id;
      record_id_ = globe_record_id_.fetch_add(1, std::memory_order_relaxed);
      (CallbacksTable::get_id_callback())(record_id_);
    }
  }

  template <class T> inline void ReportEventTimestamps(T& obj, const size_t bytes = 0) {
    if (IsEnabled()) {
      uint64_t start = obj.profilingInfo().start_;
      uint64_t end = obj.profilingInfo().end_;
      callback(start, end, bytes);
    }
  }

  bool IsEnabled() const { return enabled_; }

 private:
  // Activity callback routine
  void callback(const uint64_t begin_ts, const uint64_t end_ts, const size_t bytes) {
    activity_op_t op_id = (command_id_ == CL_COMMAND_NDRANGE_KERNEL) ? OP_ID_DISPATCH : OP_ID_COPY;
    activity_record_t record {
        ACTIVITY_DOMAIN_ID,            // domain id
        (activity_kind_t)command_id_,  // activity kind
        op_id,                         // operation id
        record_id_,                    // activity correlation id
        begin_ts,                      // begin timestamp, ns
        end_ts,                        // end timestamp, ns
        static_cast<int>(device_id_),  // device id
        queue_id_,                     // queue id
        bytes                          // copied data size, for memcpy
    };
    (CallbacksTable::get_op_callback())(op_id, &record, CallbacksTable::get_arg());
  }

  command_id_t command_id_; //!< Command ID, executed on the queue
  uint32_t queue_id_;       //!< Queue ID, associated with this command
  uint32_t device_id_;      //!< Device ID, associated with this command
  record_id_t record_id_;   //!< Uniqueue execution ID(counter) of this command
  bool enabled_;            //!< Activity profiling is enabled

  // Global record ID
  static std::atomic<record_id_t> globe_record_id_; //!< GLobal counter of all executed commands
};

}  // namespace activity_prof

#else
#define ACTIVITY_PROF_INSTANCES()

namespace activity_prof {
typedef uint32_t op_id_t;
typedef uint32_t command_id_t;

typedef void* id_callback_fun_t;
typedef void* callback_fun_t;
typedef void* callback_arg_t;

struct CallbacksTable {
  static void init(const id_callback_fun_t& id_callback, const callback_fun_t& op_callback,
                   const callback_arg_t& arg) {}
  static bool SetEnabled(const op_id_t& op_id, const bool& enable) { return false; }
};

class ActivityProf {
 public:
  ActivityProf() {}
  inline void Initialize(const command_id_t command_id, const uint32_t queue_id,
                         const uint32_t device_id) {}
  template <class T> inline void ReportEventTimestamps(T& obj, const size_t bytes = 0) {}
  inline bool IsEnabled() { return false; }
};

}  // namespace activity_prof

#endif

const char* getOclCommandKindString(uint32_t op);
