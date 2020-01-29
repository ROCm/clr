//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "os/os.hpp"
#include "platform/perfctr.hpp"
#include "device/gpu/gpudefs.hpp"
#include "device/gpu/gputimestamp.hpp"
#include "device/gpu/gpuvirtual.hpp"
#include "device/gpu/gpucounters.hpp"

namespace gpu {

TimeStamp::TimeStamp(const VirtualGPU& gpu, gslMemObject gslMem, uint memOffset, address cpuAddr)
    : gpu_(gpu), gslMem_(gslMem), memOffset_(memOffset) {
  values_ = reinterpret_cast<volatile uint64_t*>(cpuAddr + memOffset);
}

TimeStamp::~TimeStamp() {}

void TimeStamp::begin(bool sdma) {
  if (!flags_.beginIssued_) {
    gpu().rs()->writeTimer(gpu().cs(), sdma, gslMem_,
                           memOffset_ + CommandStartTime * sizeof(uint64_t));
    flags_.beginIssued_ = true;
  }
}

void TimeStamp::end(bool sdma) {
  CondLog(!flags_.beginIssued_, "We didn't issue a begin operation!");
  gpu().rs()->writeTimer(gpu().cs(), sdma, gslMem_, memOffset_ + CommandEndTime * sizeof(uint64_t));
  flags_.endIssued_ = true;
  flags_.sdma_ = sdma;
}

inline void SetValue(uint64_t* time, uint64_t val, double nanos) {
  *time = static_cast<uint64_t>(static_cast<double>(val) * nanos);
}

void TimeStamp::value(uint64_t* startTime, uint64_t* endTime) {
  CondLog(!flags_.endIssued_, "We didn't send the counter end operation!");
  const double NanoSecondsPerTick = gpu_.dev().getAttribs().nanoSecondsPerTick;

  SetValue(startTime, values_[CommandStartTime], NanoSecondsPerTick);
  SetValue(endTime, values_[CommandEndTime], NanoSecondsPerTick);
}

TimeStampCache::~TimeStampCache() {
  // Release all time stamp objects from the cache
  for (uint i = 0; i < freedTS_.size(); ++i) {
    delete freedTS_[i];
  }
  freedTS_.clear();

  // Release all memory objects
  for (uint i = 0; i < tsBuf_.size(); ++i) {
    tsBuf_[i]->unmap(&gpu_);
    delete tsBuf_[i];
  }
  tsBuf_.clear();
}

TimeStamp* TimeStampCache::allocTimeStamp() {
  TimeStamp* ts = NULL;
  if (0 != freedTS_.size()) {
    ts = freedTS_.back();
    freedTS_.pop_back();
  }

  if (NULL == ts) {
    if ((tsBufCpu_ == NULL) || ((tsOffset_ + TimerSlotSize) > TimerBufSize)) {
      Memory* buf = new Memory(gpu_.dev(), TimerBufSize);
      if (buf == NULL || !buf->create(Resource::Remote)) {
        return NULL;
      }
      tsBufCpu_ = reinterpret_cast<address>(buf->map(&gpu_));
      memset(tsBufCpu_, 0, TimerBufSize);
      tsOffset_ = 0;
      tsBuf_.push_back(buf);
    }
    // Allocate a TimeStamp object
    ts = new TimeStamp(gpu_, tsBuf_[(tsBuf_.size() - 1)]->gslResource(), tsOffset_, tsBufCpu_);
    // Create a timestamp
    if (ts == NULL) {
      return NULL;
    }
    tsOffset_ += TimerSlotSize;
  }

  // Set this timestamp into DRM profile mode if it was requested
  ts->clearStates();

  return ts;
}

}  // namespace gpu
