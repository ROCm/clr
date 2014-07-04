//
// Copyright 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpuvirtual.hpp"
#include "device/cpu/cpudevice.hpp"
#include "device/cpu/cpucommand.hpp"
#include "device/blit.hpp"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "os/os.hpp"

namespace cpu {

amd::Atomic<size_t> VirtualCPU::numWorkerThreads_(0);

VirtualCPU::VirtualCPU(Device& device)
    : device::VirtualDevice(device), acceptingCommands_(false)
{
    const size_t numCores = device.info().maxComputeUnits_;

    if ((numWorkerThreads_ += numCores) >= Device::getMaxWorkerThreadsNumber()) {
        numWorkerThreads_ -= numCores;
        cores_ = NULL;
        return;
    }

    cores_ = new(std::nothrow) WorkerThread*[numCores];
    if (cores_ == NULL) {
        return;
    }

    // Clear memory for the worker threads
    memset(cores_, 0, numCores * sizeof(WorkerThread*));

#if defined(__linux__)
    const bool isNuma =
#if defined(NUMA_SUPPORT)
        device.getNumaMask() == NULL;
#else
        false;
#endif // NUMA_SUPPORT
    const amd::Os::ThreadAffinityMask* affinityMask = isNuma ? NULL :
#else
    const amd::Os::ThreadAffinityMask* affinityMask =
#endif
        device.getWorkerThreadsAffinity();

    uint coreId = affinityMask != NULL ? affinityMask->getFirstSet() : (uint)-1;

    for (size_t i = 0; i < numCores; ++i) {
        WorkerThread* thread = cores_[i] = new WorkerThread(device);
        if (thread == NULL) {
            for (size_t j = 0; j < i; ++j) {
                cores_[j]->resume();
            }
            return;
        }

        if (thread->state() != amd::Thread::INITIALIZED) {
            return;
        }

#if defined(__linux__)
        if (!isNuma) {
            if (coreId == (uint)-1) {
                thread->setAffinity((uint) i);
            }
            else {
                thread->setAffinity(coreId);
                coreId = affinityMask->getNextSet(coreId);
            }
        }
#else // On Windows we set an affinity mask and not a specific ID.
        if (coreId != (uint)-1) {
            thread->setAffinity(*affinityMask);
        }
#endif
        thread->start();
    }

    blitMgr_ = new device::HostBlitManager(*this);
    if ((NULL == blitMgr_) || !blitMgr_->create(device)) {
        LogError("Could not create BlitManager!");
        return;
    }

    acceptingCommands_ = true;
}

VirtualCPU::~VirtualCPU()
{
    if (cores_ == NULL) {
        return;
    }

    delete blitMgr_;

    const size_t numCores = device().info().maxComputeUnits_;
    for (size_t i = 0; i < numCores; ++i) {
        delete cores_[i];
    }
    numWorkerThreads_ -= numCores;
    delete[] cores_;
}

bool
VirtualCPU::terminate()
{
    if (cores_ == NULL) {
        return true;
    }

    const size_t numCores = device().info().maxComputeUnits_;
    for (size_t i = 0; i < numCores; ++i) {
        if (cores_[i]) {
            cores_[i]->terminate();
        }
    }
    return true;
}

void
VirtualCPU::submitReadMemory(amd::ReadMemoryCommand& vcmd)
{
    vcmd.setStatus(CL_RUNNING);

    bool result = false;
    device::Memory memory(vcmd.source());

    // Ensure memory up-to-date
    vcmd.source().cacheWriteBack();

    switch (vcmd.type()) {
    case CL_COMMAND_READ_BUFFER:
        result = blitMgr().readBuffer(memory, vcmd.destination(),
            vcmd.origin(), vcmd.size(), vcmd.isEntireMemory());
        break;
    case CL_COMMAND_READ_BUFFER_RECT:
        result = blitMgr().readBufferRect(memory,
            vcmd.destination(), vcmd.bufRect(), vcmd.hostRect(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    case CL_COMMAND_READ_IMAGE:
        result = blitMgr().readImage(memory, vcmd.destination(),
            vcmd.origin(), vcmd.size(), vcmd.rowPitch(), vcmd.slicePitch(),
            vcmd.isEntireMemory());
        break;
    default:
        LogError("Unsupported type for the read command");
        break;
    }

    if (!result) {
        LogError("submitReadMemory failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        vcmd.setStatus(CL_COMPLETE);
    }
}

void
VirtualCPU::submitWriteMemory(amd::WriteMemoryCommand& vcmd)
{
    vcmd.setStatus(CL_RUNNING);

    bool result = false;
    device::Memory memory(vcmd.destination());

    // Ensure memory up-to-date
    vcmd.destination().cacheWriteBack();

    // Process different write commands
    switch (vcmd.type()) {
    case CL_COMMAND_WRITE_BUFFER:
        result = blitMgr().writeBuffer(vcmd.source(), memory,
            vcmd.origin(), vcmd.size(), vcmd.isEntireMemory());
        break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
        result = blitMgr().writeBufferRect(vcmd.source(), memory,
            vcmd.hostRect(), vcmd.bufRect(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    case CL_COMMAND_WRITE_IMAGE:
        result = blitMgr().writeImage(vcmd.source(), memory,
            vcmd.origin(), vcmd.size(), vcmd.rowPitch(), vcmd.slicePitch(),
            vcmd.isEntireMemory());
        break;
    default:
        LogError("Unsupported type for the write command");
        break;
    }

    // Mark cache as clean (CPU works directly on backing store)
    vcmd.destination().signalWrite(NULL);

    if (!result) {
        LogError("submitWriteMemory failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        vcmd.setStatus(CL_COMPLETE);
    }
}


void
VirtualCPU::submitCopyMemory(amd::CopyMemoryCommand& vcmd)
{
    vcmd.setStatus(CL_RUNNING);

    // Ensure memory up-to-date
    vcmd.source().cacheWriteBack();
    vcmd.destination().cacheWriteBack();

    // Translate memory references and ensure cache up-to-date
    device::Memory dstMemory(vcmd.destination());
    device::Memory srcMemory(vcmd.source());

    bool result = false;

    // Check if HW can be used for memory copy
    switch (vcmd.type()) {
    case CL_COMMAND_COPY_BUFFER:
        result = blitMgr().copyBuffer(srcMemory, dstMemory,
            vcmd.srcOrigin(), vcmd.dstOrigin(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    case CL_COMMAND_COPY_BUFFER_RECT:
        result = blitMgr().copyBufferRect(srcMemory, dstMemory,
            vcmd.srcRect(), vcmd.dstRect(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
        result = blitMgr().copyImageToBuffer(srcMemory, dstMemory,
            vcmd.srcOrigin(), vcmd.dstOrigin(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
        result = blitMgr().copyBufferToImage(srcMemory, dstMemory,
            vcmd.srcOrigin(), vcmd.dstOrigin(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    case CL_COMMAND_COPY_IMAGE:
        result = blitMgr().copyImage(srcMemory, dstMemory,
            vcmd.srcOrigin(), vcmd.dstOrigin(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    default:
        LogError("Unsupported command type for memory copy!");
        break;
    }

    // Mark cache as clean (CPU works directly on backing store)
    vcmd.destination().signalWrite(NULL);

    if (!result) {
        LogError("submitCopyMemory failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        vcmd.setStatus(CL_COMPLETE);
    }
}

void
VirtualCPU::submitMapMemory(amd::MapMemoryCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);

    if (cmd.mapFlags() & CL_MAP_READ
     || cmd.mapFlags() & CL_MAP_WRITE) {
         LogInfo("cpu::VirtualCPU::submitMapMemory() CL_MAP_READ and CL_MAP_WRITE ignored");
    }

    // Ensure memory up-to-date
    cmd.memory().cacheWriteBack();

    cmd.setStatus(CL_COMPLETE);
}

void
VirtualCPU::submitUnmapMemory(amd::UnmapMemoryCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);

    // Mark cache as clean (CPU works directly on backing store)
    cmd.memory().signalWrite(NULL);

    //! @todo:dgladdin: strictly speaking we should check that the mem object was mapped
    cmd.setStatus(CL_COMPLETE);
}

void
VirtualCPU::submitFillMemory(amd::FillMemoryCommand& vcmd)
{
    vcmd.setStatus(CL_RUNNING);

    device::Memory memory(vcmd.memory());

    vcmd.memory().cacheWriteBack();

    bool result = false;

    // Find the the right fill operation
    switch (vcmd.type()) {
    case CL_COMMAND_FILL_BUFFER:
        result = blitMgr().fillBuffer(memory, vcmd.pattern(),
            vcmd.patternSize(), vcmd.origin(), vcmd.size(),
            vcmd.isEntireMemory());
        break;
    case CL_COMMAND_FILL_IMAGE:
        result = blitMgr().fillImage(memory, vcmd.pattern(),
            vcmd.origin(), vcmd.size(), vcmd.isEntireMemory());
        break;
    default:
        LogError("Unsupported command type for FillMemory!");
        break;
    }

    vcmd.memory().signalWrite(NULL);

    if (!result) {
        LogError("submitFillMemory failed!");
        vcmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        vcmd.setStatus(CL_COMPLETE);
    }
}

//! Helper function for forcing a cache sync for all kernel parameters
static void syncAllParams(amd::NDRangeKernelCommand& cmd)
{
    const amd::Kernel& kernel = cmd.kernel();
    const amd::KernelParameters& kernelParam = kernel.parameters();
    const amd::KernelSignature& signature = kernel.signature();
    const amd::Device& device = cmd.queue()->device();

    for (size_t i = 0; i < signature.numParameters(); ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        if (desc.type_ == T_POINTER && desc.size_ > 0 &&
            !kernelParam.boundToSvmPointer(device, cmd.parameters(), i)) {
            address ptr = (address) (cmd.parameters() + desc.offset_);
            amd::Memory* memArg = *(amd::Memory**)ptr;

            if (memArg != NULL) {
                memArg->cacheWriteBack();
                memArg->signalWrite(NULL);
            }
        }
    }
}

void
VirtualCPU::computeLocalSizes(amd::NDRangeKernelCommand& command,
                             amd::NDRange& local) {
  bool uniformSize = (OPENCL_MAJOR < 2) ||
    command.kernel().getDeviceKernel(device())->getUniformWorkGroupSize();

  const amd::NDRangeContainer& sizes = command.sizes();
  const size_t numCores = device().info().maxComputeUnits_;

  const size_t globalSize1D = sizes.global().product();
  const size_t targetNumOperations = 
    std::min(globalSize1D, numCores * 4);
  size_t localSize1D = 
    std::min(globalSize1D / targetNumOperations,
             device().info().maxWorkGroupSize_);
  
  for (size_t i = 0; i < local.dimensions(); ++i) {
    const size_t globalSize = sizes.global()[i];
    size_t localSize =
      std::min(std::min(localSize1D, globalSize),
               device().info().maxWorkItemSizes_[i]);
    
    // local must exactly divide global if uniform size is required
    // For non uniform size, we could use the work group size hint
    if (uniformSize && globalSize % localSize != 0) {
      while (true) {
        //! @todo: lmoriche: find a better way
        if (globalSize % localSize == 0) break;
        --localSize;
      }
    }
    local[i] = localSize;
    localSize1D /= localSize;
  }

  command.setLocalWorkSize(local);
}


static
amd::NDRange computeRemainders(const amd::NDRange& global,
                               const amd::NDRange& local)
{
  amd::NDRange remainders(local.dimensions());

  for (size_t i = 0; i < local.dimensions(); ++i) {
    remainders[i] = (global[i]  % local[i] != 0) ? 1 : 0;
  }

  return remainders;
}

void
VirtualCPU::submitKernel(amd::NDRangeKernelCommand& command)
{
    const amd::NDRangeContainer& sizes = command.sizes();
    const size_t numCores = device().info().maxComputeUnits_;

    amd::NDRange local = sizes.local();

    if (local == 0) {
      computeLocalSizes(command, local);
    }
    amd::NDRange remainders = computeRemainders(sizes.global(), local);

    // number of groups in each dimensions
    const amd::NDRange numGroups = (sizes.global() / local) + remainders;

    size_t numOperations = numGroups.product();
    if (numOperations == 0) {
        command.setStatus(CL_COMPLETE);
        return;
    }

    syncAllParams(command);
    // retain the command here instead of retaining in NDRangeKernelBatch' ctor
    command.retain();

    size_t batchCount = std::min(numOperations, numCores);
    NDRangeKernelBatch batch(command, *this, numGroups, batchCount);

    Operation::Counter counter(command, batchCount);
    command.setData(&counter);

    for (size_t coreId = 0; coreId < batchCount; ++coreId) {
        batch.setCoreId(coreId);
        cores_[coreId]->enqueue(batch);
        cores_[coreId]->flush();
    }

    command.awaitCompletion();
    command.release();
}

void
VirtualCPU::submitNativeFn(amd::NativeFnCommand& command)
{
    NativeFn fn(command);
    cores_[0]->enqueue(fn);
    cores_[0]->flush();
    command.awaitCompletion();
}

void
VirtualCPU::submitMarker(amd::Marker& command)
{
    command.setStatus(CL_COMPLETE);
}

void
VirtualCPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd)
{
    //! @todo [odintsov]: create an AcquireExtObjectsOperation and enqueue it
    //!  to a core when a core scheduler is around.
    //
    // cores_[0]->enqueue(new AcquireExtObjectsOperation(cmd));
    // the code below will be moved to AcquireExtObjectsOperation::execute()
    cmd.setStatus(CL_RUNNING);

    //
    // AcquireExtObjects execution starts here
    //
    bool bError = false;

    //! Go through ext objects by one and call member function to execute
    //! a sequence of external graphics API commands for each external object
    for(std::vector<amd::Memory*>::const_iterator itr = cmd.getMemList().begin();
        itr != cmd.getMemList().end(); itr++) {
        if(*itr) {
            bError |= !((*itr)->mapExtObjectInCQThread());
        }
    }
    if(bError) {
        cmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        cmd.setStatus(CL_COMPLETE);
    }
}

void
VirtualCPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd)
{
    //! @todo [odintsov]: create a ReleaseExtObjectsOperation and enqueue it
    //! to a core when a core scheduler is around.
    //
    // cores_[i]->enqueue(new ReleaseExtObjectsOperation(cmd));
    // the code below will be moved to ReleaseExtObjectsOperation::execute()
    cmd.setStatus(CL_RUNNING);

    bool bError = false;

    for(std::vector<amd::Memory*>::const_iterator itr = cmd.getMemList().begin();
        itr != cmd.getMemList().end(); itr++) {
        if(*itr) {
            bError |= !((*itr)->unmapExtObjectInCQThread());
        }
    }
    if(bError) {
        cmd.setStatus(CL_INVALID_OPERATION);
    }
    else {
        cmd.setStatus(CL_COMPLETE);
    }
}

void VirtualCPU::submitPerfCounter(amd::PerfCounterCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);
    LogError("We don't support HW perf counters on CPU");
    cmd.setStatus(CL_INVALID_OPERATION);
}

void VirtualCPU::submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);
    LogError("We don't support thread trace on CPU");
    cmd.setStatus(CL_INVALID_OPERATION);
}

void VirtualCPU::submitThreadTrace(amd::ThreadTraceCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);
    LogError("We don't support thread trace on CPU");
    cmd.setStatus(CL_INVALID_OPERATION);
}

void
VirtualCPU::flush(amd::Command* list, bool wait)
{
    amd::Command* head = list;

    // Release all commands from the link list
    while (head != NULL) {
        amd::Command * it = head->getNext();
        head->release();
        head = it;
    }
}

#if cl_amd_open_video
void VirtualCPU::submitRunVideoProgram(amd::RunVideoProgramCommand& cmd)
{
    cmd.setStatus(CL_INVALID_OPERATION);
}

void VirtualCPU::submitSetVideoSession(amd::SetVideoSessionCommand& cmd)
{
    cmd.setStatus(CL_INVALID_OPERATION);
}
#endif // cl_amd_open_video

void
VirtualCPU::submitSignal(amd::SignalCommand & cmd)
{
    cmd.setStatus(CL_INVALID_OPERATION);
}

void
VirtualCPU::submitMakeBuffersResident(amd::MakeBuffersResidentCommand & cmd)
{
    cmd.setStatus(CL_INVALID_OPERATION);
}

void
VirtualCPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);
    if (cmd.pfnFreeFunc() == NULL) {
        // pointers allocated using clSVMAlloc
        for (cl_uint i = 0; i < cmd.svmPointers().size(); i++) {
            amd::SvmBuffer::free(cmd.context(), cmd.svmPointers()[i]);
        }
    }
    else {
        cmd.pfnFreeFunc()(as_cl(cmd.queue()->asCommandQueue()), cmd.svmPointers().size(),
                (void**) (&(cmd.svmPointers()[0])), cmd.userData());
    }
    cmd.setStatus(CL_COMPLETE);
}

void
VirtualCPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);
    amd::SvmBuffer::memFill(cmd.dst(), cmd.src(), cmd.srcSize(), 1);
    cmd.setStatus(CL_COMPLETE);
}

void
VirtualCPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd)
{
    cmd.setStatus(CL_RUNNING);
    amd::SvmBuffer::memFill(cmd.dst(), cmd.pattern(), cmd.patternSize(), cmd.times());
    cmd.setStatus(CL_COMPLETE);
}

void
VirtualCPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd)
{
    cmd.setStatus(CL_COMPLETE);
}

void
VirtualCPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd)
{
    cmd.setStatus(CL_COMPLETE);
}

} // namespace cpu
