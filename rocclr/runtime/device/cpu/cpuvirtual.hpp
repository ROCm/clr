//
// Copyright (c) 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CPUVIRTUAL_HPP_
#define CPUVIRTUAL_HPP_

#include "top.hpp"
#include "device/device.hpp"
#include "thread/atomic.hpp"
#include "thread/thread.hpp"
#include "platform/ndrange.hpp"

//! \namespace cpu CPU Device Implementation
namespace cpu {

class WorkerThread;
class Device;

class VirtualCPU : public device::VirtualDevice {
 private:
  WorkerThread** cores_;                         //!< Pointer to array of Worker threads
  static amd::Atomic<size_t> numWorkerThreads_;  //!< Current Worker Threads number
  bool acceptingCommands_;

 public:
  VirtualCPU(cpu::Device& device);
  ~VirtualCPU();
  bool terminate();

  WorkerThread* getWorkerThread(size_t id) { return cores_[id]; }

  bool acceptingCommands() const { return acceptingCommands_; }

  virtual void submitReadMemory(amd::ReadMemoryCommand& command);
  virtual void submitWriteMemory(amd::WriteMemoryCommand& command);
  virtual void submitCopyMemory(amd::CopyMemoryCommand& command);
  virtual void submitMapMemory(amd::MapMemoryCommand& command);
  virtual void submitUnmapMemory(amd::UnmapMemoryCommand& command);
  virtual void submitKernel(amd::NDRangeKernelCommand& command);
  virtual void submitNativeFn(amd::NativeFnCommand& command);
  virtual void submitMarker(amd::Marker& command);
  virtual void submitFillMemory(amd::FillMemoryCommand& command);
  virtual void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd) {}
  virtual void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd);
  virtual void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd);
  virtual void submitPerfCounter(amd::PerfCounterCommand& cmd);
  virtual void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd);
  virtual void submitThreadTrace(amd::ThreadTraceCommand& cmd);
  virtual void flush(amd::Command* list = NULL, bool wait = false);
  virtual void submitSignal(amd::SignalCommand& cmd);
  virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& cmd);
  virtual void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  virtual void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  virtual void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  virtual void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  virtual void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);

  virtual void computeLocalSizes(amd::NDRangeKernelCommand& command, amd::NDRange& local);

  static bool fillImage(amd::Image& image, address fillMem, const void* pattern,
                        const amd::Coord3D& origin, const amd::Coord3D& region, size_t rowPitch,
                        size_t slicePitch, size_t elementSize);
};

}  // namespace cpu

#endif  // CPUVIRTUAL_HPP_
