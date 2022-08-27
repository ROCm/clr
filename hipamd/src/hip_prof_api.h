/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc.

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

#ifndef HIP_SRC_HIP_PROF_API_H
#define HIP_SRC_HIP_PROF_API_H

#include <atomic>
#include <cassert>
#include <iostream>
#include <shared_mutex>
#include <utility>

#if USE_PROF_API
#include "hip/amd_detail/hip_prof_str.h"
#include "platform/prof_protocol.h"

// HIP API callbacks spawner object macro
#define HIP_CB_SPAWNER_OBJECT(CB_ID) \
  api_callbacks_spawner_t<HIP_API_ID_##CB_ID> \
    __api_tracer([=](auto &api_data) constexpr { INIT_CB_ARGS_DATA(CB_ID, api_data); });

class api_callbacks_table_t {
 public:
  api_callbacks_table_t() = default;

  bool set_activity(hip_api_id_t id, activity_sync_callback_t function, void* arg) {
    if (id < HIP_API_ID_FIRST || id > HIP_API_ID_LAST)
      return false;

    auto& entry = callbacks_table_[id];
    std::unique_lock lock(entry.mutex);

    /* 'function != nullptr' indicates it is activity register call,
       increment should happen only once but client is free to call
       register CB multiple times for same API id hence the check

       'function == nullptr' indicates it is de-register call and
       decrement should happen only once hence the check. */

    if (function != nullptr) {
      if (entry.activity.first == nullptr)
        enabled_api_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
      if (entry.activity.first != nullptr)
        enabled_api_count_.fetch_sub(1, std::memory_order_relaxed);
    }
    entry.activity = {function, arg};

    return true;
  }

  bool set_callback(hip_api_id_t id, activity_rtapi_callback_t function, void* arg) {
    if (id < HIP_API_ID_FIRST || id > HIP_API_ID_LAST)
      return false;

    auto& entry = callbacks_table_[id];
    std::unique_lock lock(entry.mutex);
    entry.user_callback = {function, arg};

    return true;
  }

  auto get(hip_api_id_t id) {
    assert(id >= HIP_API_ID_FIRST && id <= HIP_API_ID_LAST && "invalid callback id");

    auto& entry = callbacks_table_[id];
    std::shared_lock lock(entry.mutex);
    return std::make_pair(entry.user_callback, entry.activity);
  }

  bool is_enabled() const {
    return enabled_api_count_.load(std::memory_order_relaxed) > 0;
  }

 private:
  std::atomic<uint32_t> enabled_api_count_{0};

  // HIP API callbacks table
  struct {
    std::shared_mutex mutex;
    std::pair<activity_sync_callback_t, void*> activity;
    std::pair<activity_rtapi_callback_t, void*> user_callback;
  } callbacks_table_[HIP_API_ID_LAST + 1]{};
};

extern api_callbacks_table_t callbacks_table;

template <hip_api_id_t ID>
class api_callbacks_spawner_t {
 public:
  template <typename Functor>
  constexpr api_callbacks_spawner_t(Functor init_cb_args_data) : record_()
  {
    static_assert(ID >= HIP_API_ID_FIRST && ID <= HIP_API_ID_LAST, "invalid callback id");
    if (!callbacks_table.is_enabled()) return;

    std::tie(user_callback_, activity_) = callbacks_table.get(ID);
    if (activity_.first == nullptr)
      return;

    api_data_.phase = ACTIVITY_API_PHASE_ENTER;
    activity_.first(ID, &record_, &api_data_, activity_.second);
    activity_prof::correlation_id = api_data_.correlation_id;

    if (user_callback_.first) {
      init_cb_args_data(api_data_);
      user_callback_.first(ACTIVITY_DOMAIN_HIP_API, ID, &api_data_, user_callback_.second);
    }
  }

  ~api_callbacks_spawner_t() {
    if (activity_.first == nullptr)
      return;

    api_data_.phase = ACTIVITY_API_PHASE_EXIT;
    if (user_callback_.first != nullptr)
      user_callback_.first(ACTIVITY_DOMAIN_HIP_API, ID, &api_data_, user_callback_.second);

    activity_prof::correlation_id = 0;
    activity_.first(ID, &record_, &api_data_, activity_.second);
  }

 private:
  std::pair<activity_rtapi_callback_t /* function */, void * /* arg */> user_callback_;
  std::pair<activity_sync_callback_t /* function */, void * /* arg */> activity_;
  activity_record_t record_;
  union {
    hip_api_data_t api_data_;
  };
};

template <>
class api_callbacks_spawner_t<HIP_API_ID_NONE> {
 public:
  template <typename Functor>
  api_callbacks_spawner_t(Functor) {}
};

#else

#define HIP_CB_SPAWNER_OBJECT(x) do {} while(false)

class api_callbacks_table_t {
 public:
  bool set_activity(hip_api_id_t, activity_sync_callback_t, void*) { return false; }
  bool set_callback(hip_api_id_t, activity_rtapi_callback_t, void*) { return false; }
};

#endif

#endif  // HIP_SRC_HIP_PROF_API_H
