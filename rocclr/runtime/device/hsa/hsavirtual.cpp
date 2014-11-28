//
// Copyright (c) 2013 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/hsa/hsadevice.hpp"
#include "device/hsa/hsavirtual.hpp"
#include "device/hsa/hsakernel.hpp"
#include "device/hsa/hsamemory.hpp"
#include "device/hsa/oclhsa_common.hpp"
#include "device/hsa/hsacounters.hpp"
#include "device/hsa/hsablit.hpp"

#include "platform/kernel.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include "platform/sampler.hpp"
#include "utils/debug.hpp"

#include "newcore.h"
#include "services.h"
#include "hsainterop.h"

#ifdef _WIN32
#include "amdocl/cl_d3d10_amd.hpp"
#endif  // _WIN32

#include "amdocl/cl_gl_amd.hpp"

#include <fstream>
#include <vector>

namespace oclhsa {

Timestamp::~Timestamp() {
    if (signal_ != 0) {
        hsacoreapi->HsaDestroySignal(signal_);
    }
}

HsaSignal Timestamp::createSignal() {
    start_ = 0;
    end_ = 0;

    HsaStatus status = hsacoreapi->HsaCreateSignal(&signal_);
    if (status != kHsaStatusSuccess) {
        LogError("HsaCreateSignal failed, could not create signal for timestamp");
        return 0;
    }
    return signal_;
}

void Timestamp::start() {
    start_ = amd::Os::timeNanos();
    signal_ = 0;
}

void Timestamp::end() {
    end_ = amd::Os::timeNanos();
}

/**
 * @brief Waits on an outstanding kernel without regard to how
 * it was dispatched - with or without a signal
 *
 * @return bool true if Wait returned successfully, false
 * otherwise
 */
bool VirtualGPU::releaseGpuMemoryFence() {
   
    // Return if there is no pending dispatch
    if (!hasPendingDispatch_) {
      return false;
    }

    // Reset the wait on  dispatch flag
    HsaStatus status;
    hasPendingDispatch_ = false;

    // This is the first call to wait on a kernel, issue
    // a End Of Pipe - Release_Mem command
    HsaQueue *hsaQueue;
    hsaQueue = (lastSubmitQueue_ == kHsaQueueTypeCompute) ?
                                    gpu_queue_ : interopQueue_;
    if (hsaQueue != NULL) {
        status = hsacoreapi->HsaAmdReleaseGpuFence(hsaQueue);
        if (status == kHsaStatusSuccess) {
            return true;
        }
    }

    LogError("Call to HsaAmdReleaseGpuFence() failed.\n");
    return false;
}

VirtualGPU::VirtualGPU(Device &device)
    : device::VirtualDevice(device), oclhsa_device_(device)
{
    lastSubmitQueue_ = static_cast<HsaQueueType>(0xFFFF);
    gpu_device_ = const_cast<HsaDevice *>(device.getBackendDevice());
    interopQueue_ = NULL;
    timestamp_ = NULL;

    // Initialize the last signal and dispatch flags
    hasPendingDispatch_ = false;
}

VirtualGPU::~VirtualGPU()
{
    if (timestamp_ != NULL) {
        delete timestamp_;
        timestamp_ = NULL;
        LogError("There was a timestamp that was not used; deleting.");
    }
}

/* profilingBegin, when profiling is enabled, creates a timestamp to save in
 * virtualgpu's timestamp_, and calls start() to get the current host
 * timestamp.
 */
void VirtualGPU::profilingBegin(amd::Command &command, bool drmProfiling)
{
    if (command.profilingInfo().enabled_) {
        if (timestamp_ != NULL) {
            LogWarning("Trying to create a second timestamp in VirtualGPU. \
                        This could have unintended consequences.");
            return;
        }
        timestamp_ = new Timestamp;
        timestamp_->start();
  }
}

/* profilingEnd, when profiling is enabled, checks to see if a signal was
 * created for whatever command we are running and calls end() to get the
 * current host timestamp if no signal is available. It then saves the pointer
 * timestamp_ to the command's data.
 */
void VirtualGPU::profilingEnd(amd::Command &command)
{
    if (command.profilingInfo().enabled_) {
        if (timestamp_->getSignal() == 0) {
            timestamp_->end();
        }
        command.setData(reinterpret_cast<void*>(timestamp_));
        timestamp_ = NULL;
    }
}

bool VirtualGPU::profilingCollectResults(amd::Command *list)
{
    uint32_t cmdType;
    HsaAmdProfileObject profileObj;
    Timestamp *ts = NULL;
    HsaStatus status;

    amd::Command* current = list;
    amd::Command* next = NULL;

    // If the command list is, empty then exit
    if (current == NULL) {
        return true;
    }

    // Determine profiling has been enabled.
    if (!current->profilingInfo().enabled_) {
        return false;
    }

    // This block gets the current device and system clock counters, and uses
    // the delta between the two to adjust the device clock to the host domain.
    uint64_t endTimeStampGPU = 0;
    uint64_t endTimeStamp = 0;
    // Device frequency
    double deviceNsPerTick = 0;
    HsaDeviceClockCounterInfo clockCounterInfo;
    if (kHsaStatusSuccess == hsacoreapi->HsaDeviceGetClockCounters(gpu_device_, &clockCounterInfo)) {
    // Device frequency
        deviceNsPerTick = 1000000000.0 /
                          clockCounterInfo.device_clock_frequency_hz;
        endTimeStampGPU = clockCounterInfo.device_clock_counter * deviceNsPerTick;
        // keep this order of operations for accuracy
        endTimeStamp = clockCounterInfo.system_clock_counter *
                    (1000000000.0 / clockCounterInfo.system_clock_frequency_hz);
    } else {
        LogWarning("Could not get device/system counters. Device times could be off.");
        endTimeStamp = amd::Os::timeNanos();
    }

    uint64_t startTimeStamp = endTimeStamp;
    uint64_t readjustTimeGPU = 0;
    if (endTimeStampGPU != 0) {
        readjustTimeGPU = endTimeStampGPU - endTimeStamp;
    }

    // This block gets the first valid timestamp from the first command that has
    // one. This timestamp is used below to mark any command that came before
    // it to start and end with this first valid start time.
    current = list;
    while (current != NULL) {
        cmdType = current->type();
        if (current->data() != NULL) {
            ts = reinterpret_cast<Timestamp*>(current->data());
            if (ts->getSignal() != 0) {
                status = hsacoreapi->HsaAmdGetProfileObject(ts->getSignal(), &profileObj);
                if (status != kHsaStatusSuccess) {
                    LogError("Error reading profile data.");
                    continue;
                }
                startTimeStamp = *profileObj.launch_time_ * deviceNsPerTick;
                startTimeStamp -= readjustTimeGPU;
                endTimeStamp = startTimeStamp;
            } else {
                startTimeStamp = ts->getStart();
                endTimeStamp = ts->getStart();
            }
            break;
        }
        current = current->getNext();
    }

    // Iterate through the list of commands, and set timestamps as appropriate
    // Note, if a command does not have a timestamp, it does one of two things:
    // - if the command (without a timestamp), A, precedes another command, C,
    // that _does_ contain a valid timestamp, command A will set RUNNING and
    // COMPLETE with the RUNNING (start) timestamp from command C. This would
    // also be true for command B, which is between A and C. These timestamps
    // are actually retrieved in the block above (startTimeStamp, endTimeStamp).
    // - if the command (without a timestamp), C, follows another command, A,
    // that has a valid timestamp, command C will be set RUNNING and COMPLETE
    // with the COMPLETE (end) timestamp of the previous command, A. This is
    // also true for any command B, which falls between A and C.
    current = list;
    while (current != NULL) {
        cmdType = current->type();
        if (current->data() != NULL) {
            // Since this is a valid command to get a timestamp, we use the
            // timestamp provided by the runtime (saved in the data())
            ts = reinterpret_cast<Timestamp*>(current->data());
            if (ts->getSignal() != 0) {
                status = hsacoreapi->HsaAmdGetProfileObject(ts->getSignal(), &profileObj);
                if (status != kHsaStatusSuccess) {
                    LogError("Error reading profile data.");
                    continue;
                }
                startTimeStamp = *profileObj.launch_time_ * deviceNsPerTick;
                endTimeStamp = *profileObj.completion_time_ * deviceNsPerTick;
                startTimeStamp -= readjustTimeGPU;
                endTimeStamp -= readjustTimeGPU;
            } else {
                startTimeStamp = ts->getStart();
                endTimeStamp = ts->getEnd();
            }
            delete ts;
            current->setData(NULL);
        } else {
            // If we don't have a command that contains a valid timestamp, we
            // simply use the end timestamp of the previous command.
            // Note, if this is a command before the first valid timestamp,
            // this will be equal to the start timestamp of the first valid
            // timestamp at this point.
            startTimeStamp = endTimeStamp;
        }

        if (current->status() == CL_SUBMITTED) {
            current->setStatus(CL_RUNNING, startTimeStamp);
            current->setStatus(CL_COMPLETE, endTimeStamp);
        }
        else if (current->status() != CL_COMPLETE) {
            LogPrintfError("Unexpected command status - %d.", current->status());
        }

        next = current->getNext();
        current->release();
        current = next;
    }

    // Release the memory blocks allocated for the various
    // struct arguments of one or more kernel submissions
    std::for_each(kernelArgList_.begin(),
                  kernelArgList_.end(),
                  std::ptr_fun(servicesapi->HsaFreeSystemMemory));
    kernelArgList_.clear();
    
    // Reset the queue parameter
    lastSubmitQueue_ = static_cast<HsaQueueType>(0xFFFF);

    // Return True so that OpenCL commands are
    // not processed again
    return true;
}

bool
VirtualGPU::create(HsaQueueType queueType)
{
    //context was created with d3d11 or d3d10 or gl
    //extension enabled, RT still needs to create
    //two queues even for an interop application.
    bool isInterop = (queueType == kHsaQueueTypeInterop);
    if (kHsaStatusSuccess !=
        hsacoreapi->HsaCreateUserModeQueue(gpu_device_,
                                           NULL,
                                           0,
                                           kHsaQueueTypeCompute,
                                           kHsaQueuePriorityMaximum,
                                           kHsaQueueFractionTen,
                                           &gpu_queue_)) {
        LogError("Error creating hsa queue");
        return false;
    }

    if ((dev().settings().enableLocalMemory_ || isInterop) &&
        kHsaStatusSuccess !=
            hsacoreapi->HsaCreateUserModeQueue(gpu_device_,
                                               NULL,
                                               0,
                                               kHsaQueueTypeInterop,
                                               kHsaQueuePriorityMaximum,
                                               kHsaQueueFractionTen,
                                               &interopQueue_)) {
        LogError("Error creating hsa interop queue");
        return false;
    }

    device::BlitManager::Setup  blitSetup;
    blitMgr_ = new KernelBlitManager(*this, blitSetup);
    if ((NULL == blitMgr_) || !blitMgr_->create(oclhsa_device_)) {
        LogError("Could not create BlitManager!");
        return false;
    }

    return true;
}

bool
VirtualGPU::terminate()
{
    delete blitMgr_;

    // Release the resources of signal
    releaseGpuMemoryFence();

    // Close the user mode queue
    if (interopQueue_) {
        hsacoreapi->HsaDestroyUserModeQueue(interopQueue_);
    }
    hsacoreapi->HsaDestroyUserModeQueue(gpu_queue_);

    return true;
}

void VirtualGPU::submitReadMemory(amd::ReadMemoryCommand &cmd)
{
    device::Memory *devMem = cmd.source().getDeviceMemory(dev());
    void *dst = cmd.destination();
    amd::Coord3D size = cmd.size();

    //! @todo: add multi-devices synchronization when supported.

    cl_command_type type = cmd.type();
    bool result = false;
    bool imageBuffer = false;

    // Force buffer read for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_READ_IMAGE) &&
        (cmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        type = CL_COMMAND_READ_BUFFER;
        imageBuffer = true;
    }

    profilingBegin(cmd);

    switch (type) {
        case CL_COMMAND_READ_BUFFER: {
            amd::Coord3D    origin(cmd.origin()[0]);
            if (imageBuffer) {
                size_t  elemSize =
                    cmd.source().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0]   *= elemSize;
            }
            result = blitMgr().readBuffer(
                        *devMem, dst, origin, size,
                        cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_READ_BUFFER_RECT: {
            result = blitMgr().readBufferRect(
                        *devMem, dst, cmd.bufRect(), cmd.hostRect(), size,
                        cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_READ_IMAGE: {
            result = blitMgr().readImage(
                        *devMem, dst, cmd.origin(), size, cmd.rowPitch(),
                        cmd.slicePitch(), cmd.isEntireMemory());
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    profilingEnd(cmd);

    if (!result) {
        LogError("submitReadMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }
}

void VirtualGPU::submitWriteMemory(amd::WriteMemoryCommand &cmd)
{
    device::Memory *devMem = cmd.destination().getDeviceMemory(dev());
    const char *src = static_cast<const char *>(cmd.source());
    amd::Coord3D size = cmd.size();

    //! @todo add multi-devices synchronization when supported.

    cl_command_type type = cmd.type();
    bool result = false;
    bool imageBuffer = false;

    // Force buffer write for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_WRITE_IMAGE) &&
        (cmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        type = CL_COMMAND_WRITE_BUFFER;
        imageBuffer = true;
    }

    profilingBegin(cmd);

    switch (type) {
        case CL_COMMAND_WRITE_BUFFER: {
            amd::Coord3D    origin(cmd.origin()[0]);
            if (imageBuffer) {
                size_t  elemSize =
                    cmd.destination().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0]   *= elemSize;
            }
            result = blitMgr().writeBuffer(
                        src, *devMem , origin, size,
                        cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_WRITE_BUFFER_RECT: {
            result = blitMgr().writeBufferRect(
                        src, *devMem, cmd.hostRect(), cmd.bufRect(), size,
                        cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_WRITE_IMAGE: {
            result = blitMgr().writeImage(
                        src, *devMem, cmd.origin(), size, cmd.rowPitch(),
                        cmd.slicePitch(), cmd.isEntireMemory());
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    if (!result) {
        LogError("submitWriteMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }
    else {
        cmd.destination().signalWrite(&dev());
    }

    profilingEnd(cmd);
}

void VirtualGPU::submitCopyMemory(amd::CopyMemoryCommand &cmd)
{
    device::Memory *srcDevMem = cmd.source().getDeviceMemory(dev());
    device::Memory *destDevMem = cmd.destination().getDeviceMemory(dev());
    amd::Coord3D size = cmd.size();

    //! @todo add multi-devices synchronization when supported.

    cl_command_type type = cmd.type();
    bool result = false;
    bool srcImageBuffer = false;
    bool dstImageBuffer = false;

    // Force buffer copy for IMAGE1D_BUFFER
    if (cmd.source().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        srcImageBuffer = true;
        type = CL_COMMAND_COPY_BUFFER;
    }
    if (cmd.destination().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
        dstImageBuffer = true;
        type = CL_COMMAND_COPY_BUFFER;
    }

    profilingBegin(cmd);

    switch (cmd.type()) {
        case CL_COMMAND_COPY_BUFFER: {
            amd::Coord3D    srcOrigin(cmd.srcOrigin()[0]);
            amd::Coord3D    dstOrigin(cmd.dstOrigin()[0]);

            if (srcImageBuffer) {
                const size_t elemSize =
                    cmd.source().asImage()->getImageFormat().getElementSize();
                srcOrigin.c[0] *= elemSize;
                if (dstImageBuffer) {
                    dstOrigin.c[0] *= elemSize;
                }
                size.c[0] *= elemSize;
            }
            else if (dstImageBuffer) {
                const size_t elemSize =
                    cmd.destination().asImage()->getImageFormat().getElementSize();
                dstOrigin.c[0] *= elemSize;
                size.c[0] *= elemSize;
            }

            result = blitMgr().copyBuffer(
                        *srcDevMem, *destDevMem, srcOrigin,
                        dstOrigin, size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_BUFFER_RECT: {
            result = blitMgr().copyBufferRect(
                        *srcDevMem, *destDevMem, cmd.srcRect(),
                        cmd.dstRect(), size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_IMAGE: {
            result = blitMgr().copyImage(
                        *srcDevMem, *destDevMem, cmd.srcOrigin(),
                        cmd.dstOrigin(), size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_IMAGE_TO_BUFFER: {
            result = blitMgr().copyImageToBuffer(
                        *srcDevMem, *destDevMem, cmd.srcOrigin(),
                        cmd.dstOrigin(), size, cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
            result = blitMgr().copyBufferToImage(
                        *srcDevMem, *destDevMem, cmd.srcOrigin(),
                        cmd.dstOrigin(), size, cmd.isEntireMemory());
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    if (!result) {
        LogError("submitCopyMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }

    profilingEnd(cmd);

    cmd.destination().signalWrite(&dev());
}

void VirtualGPU::submitMapMemory(amd::MapMemoryCommand &cmd)
{
    //! @todo add multi-devices synchronization when supported.

    profilingBegin(cmd);

    device::Memory *devMemory = cmd.memory().getDeviceMemory(dev(), false);

    cl_command_type type = cmd.type();
    bool imageBuffer = false;

    // Force buffer read for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_MAP_IMAGE) &&
        (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        type = CL_COMMAND_MAP_BUFFER;
        imageBuffer = true;
    }

    cl_map_flags mapFlag = cmd.mapFlags();

    // Treat no map flag as read-write.
    if (mapFlag == 0) {
        mapFlag = CL_MAP_READ | CL_MAP_WRITE;
    }

    // Save map write requirement.
    if (mapFlag & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)) {
        devMemory->saveMapInfo(cmd.origin(), cmd.size(),
            mapFlag, cmd.isEntireMemory());
    }

    // Sync to the map target.
    if ((!devMemory->isHostMemDirectAccess()) &&
        (mapFlag & (CL_MAP_READ | CL_MAP_WRITE))) {
        bool result = false;

        oclhsa::Memory *hsaMemory = static_cast<oclhsa::Memory *>(devMemory);

        amd::Memory* mapMemory = hsaMemory->mapMemory();
        void *hostPtr = mapMemory == NULL ?
                              hsaMemory->owner()->getHostMem() :
                              mapMemory->getHostMem();

        if (type == CL_COMMAND_MAP_BUFFER) {
            amd::Coord3D    origin(cmd.origin()[0]);
            amd::Coord3D    size(cmd.size()[0]);
            if (imageBuffer) {
                size_t  elemSize =
                    cmd.memory().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0]   *= elemSize;
            }
            result = blitMgr().readBuffer(
                        *hsaMemory,
                        static_cast<char *>(hostPtr) + origin[0],
                        origin,
                        size,
                        cmd.isEntireMemory());
        }
        else if (type == CL_COMMAND_MAP_IMAGE) {
          amd::Image* image = cmd.memory().asImage();
          result = blitMgr().readImage(
                      *hsaMemory, hostPtr, amd::Coord3D(0),
                      image->getRegion(), image->getRowPitch(),
                      image->getSlicePitch(), true);
        }
        else {
            ShouldNotReachHere();
        }

        if (!result) {
            LogError("submitMapMemory failed!");
            cmd.setStatus(CL_OUT_OF_RESOURCES);
        }
    }

    profilingEnd(cmd);
}

void VirtualGPU::submitUnmapMemory(amd::UnmapMemoryCommand &cmd)
{
    profilingBegin(cmd);

    device::Memory *devMemory = cmd.memory().getDeviceMemory(dev(), false);

    // Force buffer write for IMAGE1D_BUFFER
    bool imageBuffer =
        (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER);

    if (devMemory->isUnmapWrite()) {
        // Commit the changes made by the user.
        if (!devMemory->isHostMemDirectAccess()) {
            bool result = false;

            if (cmd.memory().asImage() && !imageBuffer) {
                amd::Image *image = cmd.memory().asImage();
                result = blitMgr().writeImage(
                            cmd.mapPtr(), *devMemory,
                            devMemory->writeMapInfo()->origin_,
                            devMemory->writeMapInfo()->region_,
                            image->getRowPitch(), image->getSlicePitch());
            }
            else {
                amd::Coord3D origin(devMemory->writeMapInfo()->origin_[0]);
                amd::Coord3D size(devMemory->writeMapInfo()->region_[0]);
                if (imageBuffer) {
                    size_t  elemSize =
                        cmd.memory().asImage()->getImageFormat().getElementSize();
                    origin.c[0] *= elemSize;
                    size.c[0]   *= elemSize;
                }
                result = blitMgr().writeBuffer(
                            cmd.mapPtr(), *devMemory,
                            origin,
                            size);
            }

            if (!result) {
                LogError("submitMapMemory failed!");
                cmd.setStatus(CL_OUT_OF_RESOURCES);
            }
        }

        devMemory->clearUnmapFlags();

        cmd.memory().signalWrite(&dev());
    }

    profilingEnd(cmd);
}

void VirtualGPU::submitFillMemory(amd::FillMemoryCommand &cmd)
{
    device::Memory *devMemory = cmd.memory().getDeviceMemory(dev(), false);

    //! @todo add multi-devices synchronization when supported.

    cl_command_type type = cmd.type();
    bool result = false;
    bool imageBuffer = false;
    float fillValue[4];

    // Force fill buffer for IMAGE1D_BUFFER
    if ((type == CL_COMMAND_FILL_IMAGE) &&
        (cmd.memory().getType() == CL_MEM_OBJECT_IMAGE1D_BUFFER)) {
        type = CL_COMMAND_FILL_BUFFER;
        imageBuffer = true;
    }

    profilingBegin(cmd);

    // Find the the right fill operation
    switch (type) {
        case CL_COMMAND_FILL_BUFFER: {
            const void* pattern = cmd.pattern();
            size_t  patternSize = cmd.patternSize();
            amd::Coord3D    origin(cmd.origin()[0]);
            amd::Coord3D    size(cmd.size()[0]);
            // Reprogram fill parameters if it's an IMAGE1D_BUFFER object
            if (imageBuffer) {
                size_t  elemSize =
                    cmd.memory().asImage()->getImageFormat().getElementSize();
                origin.c[0] *= elemSize;
                size.c[0]   *= elemSize;
                memset(fillValue, 0, sizeof(fillValue));
                cmd.memory().asImage()->getImageFormat().formatColor(pattern, fillValue);
                pattern = fillValue;
                patternSize = elemSize;
            }
            result = blitMgr().fillBuffer(
                        *devMemory, pattern, patternSize, origin, size,
                        cmd.isEntireMemory());
            break;
        }
        case CL_COMMAND_FILL_IMAGE: {
            result = blitMgr().fillImage(
                        *devMemory, cmd.pattern(), cmd.origin(), cmd.size(),
                        cmd.isEntireMemory());
            break;
        }
        default:
            ShouldNotReachHere();
            break;
    }

    if (!result) {
        LogError("submitFillMemory failed!");
        cmd.setStatus(CL_OUT_OF_RESOURCES);
    }

    cmd.memory().signalWrite(&dev());

    profilingEnd(cmd);
}

void VirtualGPU::submitMigrateMemObjects(amd::MigrateMemObjectsCommand &vcmd)
{
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    profilingBegin(vcmd);

    std::vector<amd::Memory *>::const_iterator itr;

    for (itr = vcmd.memObjects().begin();
         itr != vcmd.memObjects().end();
         itr++) {
        // Find device memory
        device::Memory *m = (*itr)->getDeviceMemory(dev());
        oclhsa::Memory *memory = static_cast<oclhsa::Memory *>(m);

        if (vcmd.migrationFlags() & CL_MIGRATE_MEM_OBJECT_HOST) {
            //! @todo revisit this when multi devices is supported.
        } else if (vcmd.migrationFlags() &
                   CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) {
            //! @todo revisit this when multi devices is supported.
        } else {
            LogWarning("Unknown operation for memory migration!");
        }
    }

    profilingEnd(vcmd);
}

HsaStatus VirtualGPU::getDispatchConfig(uint32_t lds_size,
                                        bool profile_enable,
                                        HsaDispatchConfig* config,
                                        const amd::NDRangeContainer& sizes,
                                        const amd::Kernel&  kernel)
{
    uint32_t idx;
    uint32_t dimensions;
    
    //Used to detect whether runtime implemetation should
    //set up the work group size
    bool overrideLwgSize = true;

    device::Kernel *devKernel = const_cast<device::Kernel *>
                              (kernel.getDeviceKernel(dev()));

    // Initialize the work grid parameter
    for (idx = 0; idx < 3; idx++) {
        config->local_work_size.dimension[idx] = 1;
        config->global_work_size.dimension[idx] = 1;
        config->global_work_offset.dimension[idx] = 0;
    }

    // Retrieve user provided work grid values
    dimensions = sizes.dimensions();
    amd::NDRange local(sizes.local());
    amd::NDRange global(sizes.global());
    amd::NDRange offset(sizes.offset());

    // Update the work grid with user provided values
    for (idx = 0; idx < dimensions; idx++) {
        config->global_work_size.dimension[idx] = global[idx];

        config->global_work_offset.dimension[idx] = offset[idx];

        //if reqd_work_group_size is set use that
        //otherwise use the ones passed into NDRange
        //In both cases, no need to further override work group size
        if (devKernel->workGroupInfo()->compileSize_[idx]) {
            config->local_work_size.dimension[idx] =
                          devKernel->workGroupInfo()->compileSize_[idx];
            overrideLwgSize = false;
        }
        else if (local[idx]) {
            config->local_work_size.dimension[idx] = local[idx];
            overrideLwgSize = false;
        }
    }

    //If true, set work group sizes
    if (overrideLwgSize) {
        if (dimensions == 1) {
            config->local_work_size.dimension[0] =
                                    dev().settings().maxWorkGroupSize_;
        }
        else if (dimensions == 2) {
            config->local_work_size.dimension[0] =
                                    dev().settings().maxWorkGroupSize2DX_;
            config->local_work_size.dimension[1] =
                                    dev().settings().maxWorkGroupSize2DY_;
        }
        else if (dimensions == 3) {
            config->local_work_size.dimension[0] =
                                    dev().settings().maxWorkGroupSize3DX_;
            config->local_work_size.dimension[1] =
                                    dev().settings().maxWorkGroupSize3DY_;
            config->local_work_size.dimension[2] =
                                    dev().settings().maxWorkGroupSize3DZ_;
        }
        else {
            assert("Invalid Work Dimensions");
        }
    }
    // Update Local Data Store and Profiling parameters
    config->lds_size = lds_size;
    config->work_dimensions = dimensions;
    config->profile = profile_enable;
    return kHsaStatusSuccess;
}

HsaStatus VirtualGPU::synchronizeInterQueueKernels(HsaQueue *dispatchQueue) {

    // Determine current kernel type based on queue used to submit
    HsaQueueType currQueue = (dispatchQueue == gpu_queue_) ?
                                      kHsaQueueTypeCompute : kHsaQueueTypeInterop;

    // An outstanding kernel exists, a new one can be submitted
    // as long as it belongs to the same class of queue type
    if (lastSubmitQueue_ == currQueue) {
        return kHsaStatusSuccess;
    }

    // If there is no outstanding kernel, a new one can be
    // submitted unconditionally
    if (lastSubmitQueue_ == 0xFFFF) {
        lastSubmitQueue_ = currQueue;
        return kHsaStatusSuccess;
    }

    // Current kernel submit cannot occur until all outstanding
    // kernels on the queue type have completed.
    releaseGpuMemoryFence();
    lastSubmitQueue_ = currQueue;
    return kHsaStatusSuccess;
}

/*! \brief Writes to the buffer and incrememts the write pointer to the
 *         buffer. Also, ensures that the argument is written to an
 *         aligned memory as specified
 *
 * @param dst The write pointer to the buffer
 * @param src The source pointer
 * @param size The size in bytes to copy
 * @param alignment The alignment to follow while writing to the buffer
 */
static void
addArg(unsigned char** dst, const void* src,
                               size_t size, uint32_t alignment)
{
    *dst = amd::alignUp(*dst, alignment);
    memcpy(*dst, src, size);
    *dst += size;
}

static inline void
addArg(unsigned char** dst, const void* src, size_t size)
{
    assert(size < UINT32_MAX);
    addArg(dst, src, size, size);
}

static void
fillSampleDescriptor(HsaSamplerDescriptor& samplerDescriptor,
                     const amd::Sampler& sampler)
{
    samplerDescriptor.filterType = sampler.filterMode() == CL_FILTER_NEAREST ?
            HSA_SAMP_FILTER_NEAREST : HSA_SAMP_FILTER_LINEAR;
    samplerDescriptor.coordinateMode = sampler.normalizedCoords() ?
            HSA_SAMP_COORDINATE_NORMALIZED : HSA_SAMP_COORDINATE_UNNORMALIZED;
    HsaSamplerAddressMode mode = HSA_SAMP_ADDRESS_NONE;
    switch (sampler.addressingMode()) {
    case CL_ADDRESS_CLAMP_TO_EDGE:
        mode = HSA_SAMP_ADDRESS_CLAMPEDGE;
        break;
    case CL_ADDRESS_REPEAT:
        mode = HSA_SAMP_ADDRESS_WRAP;
        break;
    case CL_ADDRESS_CLAMP:
        mode = HSA_SAMP_ADDRESS_CLAMPBORDER;
        break;
    case CL_ADDRESS_MIRRORED_REPEAT:
        mode = HSA_SAMP_ADDRESS_MIRROR;
        break;
    case CL_ADDRESS_NONE:
        mode = HSA_SAMP_ADDRESS_MIRRORONCE;
        break;
    default:
        return;
    }
    samplerDescriptor.addressModeX = mode;
    samplerDescriptor.addressModeY = mode;
    samplerDescriptor.addressModeZ = mode;
}

bool
VirtualGPU::submitKernelInternal(
    const amd::NDRangeContainer& sizes,
    const amd::Kernel& kernel,
    const_address parameters,
    void *eventHandle)
{
    device::Kernel *devKernel = const_cast<device::Kernel *>
                          (kernel.getDeviceKernel(dev()));
    Kernel &gpuKernel = static_cast<Kernel &>(*devKernel);
    HsaKernelCode *kernelCode = const_cast<HsaKernelCode *>(gpuKernel.kernelCode());
    const size_t compilerLdsUsage = kernelCode->workgroup_group_segment_byte_size;
    size_t ldsUsage = compilerLdsUsage;
    bool useInteropQueue = false;

    // Allocate buffer to hold kernel arguments
    address argBuffer = NULL;
    HsaStatus status = servicesapi->HsaAllocateSystemMemory(
        kernelCode->kernarg_segment_byte_size, 256,
        kHsaSystemMemoryTypeUncached, reinterpret_cast<void**>(&argBuffer));
    if (status != kHsaStatusSuccess) {
        LogError("Out of memory");
        return false;
    }
    kernelArgList_.push_back(argBuffer);
    address argPtr = argBuffer;

    // The HLC generates 3 additional arguments for the global offsets
    for (uint j = 0; j < Kernel::ExtraArguments; ++j) {
        const size_t offset = j < sizes.dimensions() ? sizes.offset()[j] : 0;
        addArg(&argPtr, &offset, sizeof(size_t));
    }

    const amd::KernelSignature& signature = kernel.signature();
    const amd::KernelParameters& kernelParams = kernel.parameters();

    // Find all parameters for the current kernel
    for (uint i = 0; i != signature.numParameters(); ++i) {
        const HsailKernelArg* arg = gpuKernel.hsailArgAt(i);
        const_address srcArgPtr = parameters + signature.at(i).offset_;

        if (arg->type_ == HSAIL_ARGTYPE_POINTER ) {
            const size_t size = sizeof(size_t);
            if (arg->addrQual_ == HSAIL_ADDRESS_LOCAL) {
                ldsUsage = amd::alignUp(ldsUsage, arg->alignment_);  //!< do we need this?
                addArg(&argPtr, &ldsUsage, size);
                ldsUsage += *reinterpret_cast<const size_t *>(srcArgPtr);
                continue;
            }
            assert((arg->addrQual_ == HSAIL_ADDRESS_GLOBAL) &&
                   "Unsupported address qualifier");
            if (kernelParams.boundToSvmPointer(dev(), parameters, i)) {
                addArg(&argPtr, srcArgPtr, size);
                continue;
            }
            amd::Memory* mem = *reinterpret_cast<amd::Memory* const*>(srcArgPtr);
            if (mem == NULL) {
                addArg(&argPtr, srcArgPtr, size);
                continue;
            }

            Memory *devMem = static_cast<Memory *>(mem->getDeviceMemory(dev()));
            //! @todo add multi-devices synchronization when supported.
            void* globalAddress = devMem->getDeviceMemory();
            addArg(&argPtr, &globalAddress, size);

            //! @todo Compiler has to return read/write attributes
            const cl_mem_flags flags = mem->getMemFlags();
            if (!flags || (flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY))) {
                mem->signalWrite(&dev());
            }

            useInteropQueue |= devMem->isHsaLocalMemory();
        }
        else if (arg->type_ == HSAIL_ARGTYPE_VALUE) {
            if (arg->dataType_ == HSAIL_DATATYPE_STRUCT) {
                void *mem = NULL;
                if (kHsaStatusSuccess != servicesapi->HsaAllocateSystemMemory(
                      arg->size_, 0, kHsaSystemMemoryTypeUncached, &mem)) {
                    LogError("Out of memory");
                    return false;
                }
                memcpy(mem, srcArgPtr, arg->size_);
                addArg(&argPtr, &mem, sizeof(void*));
                kernelArgList_.push_back(mem);
                continue;
            }
            for (uint e = 0; e < arg->numElem_; ++e) {
                addArg(&argPtr, srcArgPtr, arg->size_);
                srcArgPtr += arg->size_;
            }
        }
        else if (arg->type_ ==  HSAIL_ARGTYPE_IMAGE) {
            amd::Memory* mem = *reinterpret_cast<amd::Memory* const*>(srcArgPtr);
            Image* image = static_cast<Image *>(mem->getDeviceMemory(dev()));
            if (image == NULL) {
                LogError( "Kernel image argument is not an image object");
                return false;
            }

            // Image arguments are of size 48 bytes and are aligned to 16 bytes
            addArg(&argPtr, image->getHsaImageObjectAddress(),
                HSA_IMAGE_OBJECT_SIZE, HSA_IMAGE_OBJECT_ALIGNMENT);

            //! @todo Compiler has to return read/write attributes
            const cl_mem_flags flags = mem->getMemFlags();
            if (!flags || (flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY))) {
                mem->signalWrite(&dev());
            }

            useInteropQueue |= image->isHsaLocalMemory();
        }
        else {
            assert((arg->type_ == HSAIL_ARGTYPE_SAMPLER) &&
                   "Unsupported address type");
            amd::Sampler* sampler = *reinterpret_cast<amd::Sampler* const*>(srcArgPtr);
            if (sampler == NULL) {
                LogError("Kernel sampler argument is not an sampler object");
                return false;
            }

            HsaSamplerDescriptor samplerDescriptor;
            fillSampleDescriptor(samplerDescriptor, *sampler);

            argPtr = amd::alignUp(argPtr, HSA_SAMPLER_OBJECT_ALIGNMENT);
            status = hsacoreapi->HsaCreateDeviceSampler(dev().getBackendDevice(),
                &samplerDescriptor, argPtr);
            if (status != kHsaStatusSuccess) {
                LogError("Error creating device sampler object!");
                return false;
            }
            argPtr += HSA_SAMPLER_OBJECT_SIZE;
        }
    }

    // Check there is no arguments' buffer overflow
    assert(argPtr <= argBuffer + kernelCode->kernarg_segment_byte_size);

    // Check for group memory overflow
    //! @todo Check should be in HSA - here we should have at most an assert
    if (ldsUsage > gpu_device_->group_memory_size) {
        LogError("No local memory available\n");
        return false;
    }

    HsaQueue *queue = useInteropQueue ? interopQueue_ : gpu_queue_;

    // Set the acl_binary and ocl event for possible debugger use
    if (eventHandle != NULL) {
        const HsaDevice *device = queue->device;
        servicesapi->HsaDebuggerCorrelationHandler(device, eventHandle);
        assert(gpuKernel.brig()->loadmap_section != NULL);
        void * acl_binary =
            reinterpret_cast<aclBinary*>(gpuKernel.brig()->loadmap_section);
        servicesapi->HsaSetAclBinary(device,
          const_cast<aclBinary*>(gpuKernel.program()->binaryElf()));
    }

    // Obtain handle to an instance of Dispatch configuration object
    HsaDispatchConfig config;
    bool profilingEnable = timestamp_ != NULL;
    status = getDispatchConfig(ldsUsage - compilerLdsUsage, profilingEnable,
                               &config, sizes, kernel);
    if (status != kHsaStatusSuccess) {
        LogError("Call to HsaPopulateDispatchConfig failed.\n");
        return false;
    }

    // Determine if enqueue must wait on last kernel submit
    status = synchronizeInterQueueKernels(queue);
    if (status != kHsaStatusSuccess) {
        LogError("synchronizeInterQueueKernels failed");
        return false;
    }

    // Create a signal object to monitor kernel completion when needed
    HsaSignal signal = profilingEnable ? timestamp_->createSignal() : 0;
    status = servicesapi->HsaDispatchKernel(queue, signal, kernelCode, &config,
                                            (uint64_t*)argBuffer, 1);
    if (status != kHsaStatusSuccess) {
        LogError("Call to HsaDispatchKernel failed.\n");
        return false;
    }

    // Mark the flag indicating if a dispatch is outstanding
    hasPendingDispatch_ = true;
    return true;
}
/**
 * @brief Api to dispatch a kernel for execution. The implementation
 * parses the input object, an instance of virtual command to obtain
 * the parameters of global size, work group size, offsets of work
 * items, enable/disable profiling, etc.
 *
 * It also parses the kernel arguments buffer to inject into Hsa Runtime
 * the list of kernel parameters.
 */
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand &vcmd) {
    profilingBegin(vcmd);

    // Submit kernel to HW
    if (!submitKernelInternal(
            vcmd.sizes(), vcmd.kernel(), vcmd.parameters(),
            static_cast<void *>(as_cl(&vcmd.event())))) {
        vcmd.setStatus(CL_INVALID_OPERATION);
    }

    profilingEnd(vcmd);
}

void VirtualGPU::submitNativeFn(amd::NativeFnCommand &cmd) {
  // std::cout<<__FUNCTION__<<" not implemented"<<"*********"<<std::endl;
}

void VirtualGPU::submitMarker(amd::Marker &cmd) {
  // std::cout<<__FUNCTION__<<" not implemented"<<"*********"<<std::endl;
}

void VirtualGPU::submitAcquireExtObjects(amd::AcquireExtObjectsCommand &vcmd)
{
  // Wait on a kernel if one is outstanding
  releaseGpuMemoryFence();

  profilingBegin(vcmd);

#ifdef _WIN32
  std::vector<amd::Memory *>::const_iterator it = vcmd.getMemList().begin();
  amd::InteropObject *interop;
  std::vector<ID3D10Resource *> d3d10Resources;
  std::vector<ID3D11Resource *> d3d11Resources;
  amd::D3D10Object *d3d10Obj;
  amd::D3D11Object *d3d11Obj;

  for (std::vector<amd::Memory *>::const_iterator it =
         vcmd.getMemList().begin();
       it != vcmd.getMemList().end(); it++) {
    // amd::Memory object should never be NULL
    assert(*it && "Memory object for interop is NULL");

    device::Memory *m = (*it)->getDeviceMemory(dev());
    oclhsa::Memory *memory = static_cast<oclhsa::Memory *>(m);

    interop = (*it)->getInteropObj();
    // [TODO]: Check if this is need in case of HSA.

    if (interop) {
      d3d10Obj = interop->asD3D10Object();
      if (d3d10Obj != NULL) {
        if (d3d10Obj->getD3D10ResOrig() != NULL) {
          // Resource is a shared copy of original resource
          // Need to copy data from original resource
          d3d10Obj->copyOrigToShared();
        }
        assert(d3d10Obj->getD3D10Resource() != NULL);
        d3d10Resources.push_back(d3d10Obj->getD3D10Resource());
      }

      d3d11Obj = interop->asD3D11Object();
      if (d3d11Obj != NULL) {
        if (d3d11Obj->getD3D11ResOrig() != NULL) {
          // Resource is a shared copy of original resource
          // Need to copy data from original resource
          d3d11Obj->copyOrigToShared();
        }
        assert(d3d11Obj->getD3D11Resource() != NULL);
        d3d11Resources.push_back(d3d11Obj->getD3D11Resource());
      }
    }

  } //end of for loop

  if (!d3d10Resources.empty()) {
    HsaStatus status = hsacoreapi->HsaAcquireD3D10Resources(gpu_device_,
                                                             &d3d10Resources[0],
                                                             d3d10Resources.size());
    if (status != kHsaStatusSuccess) {
      LogError("HsaAcquireD3D10Resources - failed");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }
  }

  if (!d3d11Resources.empty()) {
    HsaStatus status = hsacoreapi->HsaAcquireD3D11Resources(gpu_device_,
                                                             &d3d11Resources[0],
                                                             d3d11Resources.size());
    if (status != kHsaStatusSuccess) {
      LogError("HsaAcquireD3D11Resources - failed");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }
  }
#endif

  profilingEnd(vcmd);
}

void VirtualGPU::submitReleaseExtObjects(amd::ReleaseExtObjectsCommand &vcmd) {
  
  // Wait on a kernel if one is outstanding
  releaseGpuMemoryFence();

  profilingBegin(vcmd);
  std::vector<amd::Memory *>::const_iterator it = vcmd.getMemList().begin();

  amd::InteropObject *interop;

#ifdef _WIN32
  std::vector<ID3D10Resource *> d3d10Resources;
  std::vector<ID3D11Resource *> d3d11Resources;

  amd::D3D10Object *d3d10Obj;
  amd::D3D11Object *d3d11Obj;

  for (std::vector<amd::Memory *>::const_iterator it =
         vcmd.getMemList().begin();
       it != vcmd.getMemList().end(); it++) {
    // amd::Memory object should never be NULL
    assert(*it && "Memory object for interop is NULL");

    device::Memory *m = (*it)->getDeviceMemory(dev());
    oclhsa::Memory *memory = static_cast<oclhsa::Memory *>(m);
    interop = (*it)->getInteropObj();

    if (interop) {
      d3d10Obj = interop->asD3D10Object();
      if (d3d10Obj != NULL) {
        if (d3d10Obj->getD3D10ResOrig() != NULL) {
          // Resource is a shared copy of original resource
          // Need to copy data from original resource
          d3d10Obj->copySharedToOrig();
        }
        assert(d3d10Obj->getD3D10Resource() != NULL);
        d3d10Resources.push_back(d3d10Obj->getD3D10Resource());
      }

      d3d11Obj = interop->asD3D11Object();
      if (d3d11Obj != NULL) {
        if (d3d11Obj->getD3D11ResOrig() != NULL) {
          // Resource is a shared copy of original resource
          // Need to copy data from original resource
          d3d11Obj->copySharedToOrig();
        }
        assert(d3d11Obj->getD3D11Resource() != NULL);
        d3d11Resources.push_back(d3d11Obj->getD3D11Resource());
      }
    }
  }

  if (!d3d10Resources.empty()) {
    HsaStatus status = hsacoreapi->HsaReleaseD3D10Resources(gpu_device_,
                                                             &d3d10Resources[0],
                                                             d3d10Resources.size());
    if (status != kHsaStatusSuccess) {
      LogError("HsaReleaseD3D10Resources - failed");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }
  }

  if (!d3d11Resources.empty()) {
    HsaStatus status = hsacoreapi->HsaReleaseD3D11Resources(gpu_device_,
                                                             &d3d11Resources[0],
                                                             d3d11Resources.size());
    if (status != kHsaStatusSuccess) {
      LogError("HsaReleaseD3D11Resources - failed");
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }
  }
#endif  // _WIN32

  profilingEnd(vcmd);
}

void
VirtualGPU::submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd)
{
    // in-order semantics: previous commands need to be done before we start
    releaseGpuMemoryFence();

    profilingBegin(cmd);
    const std::vector<void*>& svmPointers = cmd.svmPointers();
    if (cmd.pfnFreeFunc() == NULL) {
        // pointers allocated using clSVMAlloc
        for (cl_uint i = 0; i < svmPointers.size(); i++) {
            amd::SvmBuffer::free(cmd.context(), svmPointers[i]);
        }
    }
    else {
        cmd.pfnFreeFunc()(as_cl(cmd.queue()->asCommandQueue()), svmPointers.size(),
                (void**) (&(svmPointers[0])), cmd.userData());
    }
    profilingEnd(cmd);
}

void
VirtualGPU::submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd)
{
    releaseGpuMemoryFence();
    profilingBegin(cmd);
    SvmBuffer::memFill(cmd.dst(), cmd.src(), cmd.srcSize(), 1);
    profilingEnd(cmd);
}

void
VirtualGPU::submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd)
{
    releaseGpuMemoryFence();
    profilingBegin(cmd);
    SvmBuffer::memFill(cmd.dst(), cmd.pattern(), cmd.patternSize(), cmd.times());
    profilingEnd(cmd);
}

void
VirtualGPU::submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd)
{
    // no fence is needed since this is a no-op: the command will be completed
    // only after all the previous commands are complete
    profilingBegin(cmd);
    profilingEnd(cmd);
}

void
VirtualGPU::submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd)
{
    // no fence is needed since this is a no-op: the command will be completed
    // only after all the previous commands are complete
    profilingBegin(cmd);
    profilingEnd(cmd);
}

void VirtualGPU::submitPerfCounter(amd::PerfCounterCommand &vcmd) {
  
    // Wait on a kernel if one is outstanding
    releaseGpuMemoryFence();

    HsaPmu hsaPmu = NULL;
    HsaStatus status;
    const amd::PerfCounterCommand::PerfCounterList counters = vcmd.getCounters();
    for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
        amd::PerfCounter* amdCounter =
            static_cast<amd::PerfCounter*>(counters[i]);
        const PerfCounter* counter =
            reinterpret_cast<const PerfCounter*>(amdCounter->getDeviceCounter());

        // Make sure we have a valid gpu performance counter
        if (NULL == counter) {
            if (hsaPmu == NULL) {
                status = servicesapi->HsaCreatePmu(gpu_device_, &hsaPmu);
                if (status != kHsaStatusSuccess) {
                    LogError("HsaCreatePmu - failed");
                    vcmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }
            }

            amd::PerfCounter::Properties prop = amdCounter->properties();
            PerfCounter* hsaCounter = new PerfCounter(
                                                     gpu_device_,
                                                     *this,
                                                     prop[CL_PERFCOUNTER_GPU_BLOCK_INDEX],
                                                     prop[CL_PERFCOUNTER_GPU_COUNTER_INDEX],
                                                     prop[CL_PERFCOUNTER_GPU_EVENT_INDEX]);
            if (NULL == hsaCounter) {
                LogError("We failed to allocate memory for the GPU perfcounter");
                vcmd.setStatus(CL_INVALID_OPERATION);
                return;
            }
            else if (hsaCounter->create(hsaPmu)) {
                amdCounter->setDeviceCounter(hsaCounter);
            }
            else {
                LogPrintfError("We failed to allocate a perfcounter in Hsa.\
                              Block: %d, counter: #d, event: %d",
                              hsaCounter->info()->blockIndex_,
                              hsaCounter->info()->counterIndex_,
                              hsaCounter->info()->eventIndex_);
                delete hsaCounter;
                vcmd.setStatus(CL_INVALID_OPERATION);
                return;
            }
            counter = NULL;
        }
    }

    if (vcmd.getState() == amd::PerfCounterCommand::Begin) {
        hsaPmu = NULL;
        for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
            amd::PerfCounter* amdCounter =
                static_cast<amd::PerfCounter*>(counters[i]);
            const PerfCounter* counter =
                static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());
 
            if (hsaPmu != counter->getCounterPmu()) {
                hsaPmu = counter->getCounterPmu();
                status = servicesapi->HsaPmuBegin(hsaPmu, gpu_queue_, true);
                if (status != kHsaStatusSuccess) {
                    LogError("HsaPmuBegin failed");
                    vcmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }
            }
        }
    }
    else if (vcmd.getState() == amd::PerfCounterCommand::End) {
        hsaPmu = NULL;
        for (uint i = 0; i < vcmd.getNumCounters(); ++i) {
            amd::PerfCounter* amdCounter =
                static_cast<amd::PerfCounter*>(counters[i]);
            const PerfCounter* counter =
                static_cast<const PerfCounter*>(amdCounter->getDeviceCounter());
 
            if (hsaPmu != counter->getCounterPmu()) {
                hsaPmu = counter->getCounterPmu();
                status = servicesapi->HsaPmuEnd(hsaPmu, gpu_queue_);
                if (status != kHsaStatusSuccess) {
                    LogError("HsaPmuEnd failed");
                    vcmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }

                status = servicesapi->HsaPmuWaitForCompletion(hsaPmu, HSA_TIMEOUT_INFINITE);
                if (status != kHsaStatusSuccess) {
                    LogError("HsaPmuWaitForCompletion failed");
                    vcmd.setStatus(CL_INVALID_OPERATION);
                    return;
                }
            }
        }
    }
    else {
        LogError("Unsupported performance counter state");
        vcmd.setStatus(CL_INVALID_OPERATION);
        return;
    }
}

void VirtualGPU::flush(amd::Command *list, bool wait) {

    /**
     * VT TODO temporarily setting the status complete at flush
     * This is not the correct way of handling completion, the
     * correct way is to either register a callback that sets
     * command status or tie-in event from higher levels to HSA
     * Event. There are no known thread safety issues if an HSA
     * event is exposed to OCL level and mapped to its event
     *
     * list->setStatus(CL_COMPLETE);
     */
    amd::Command *current = list;

    // Query the status of openCL kernel task i.e. is still
    // running or has completed.
    releaseGpuMemoryFence();

    // If profiling is enabled collect the results
    if (profilingCollectResults(list)) {
      return;
    }

    // The openCL task has completed successfully
    while (current != NULL) {

        // @note: Currently Commands coming into Hsa Runtime
        // already have their status set as CL_SUBMITTED
        // SUBMITTED -> RUNNING -> COMPLETE
        if (current->status() == CL_SUBMITTED) {
            current->setStatus(CL_RUNNING);
            current->setStatus(CL_COMPLETE);
        }
        else if (current->status() == CL_RUNNING) {
            current->setStatus(CL_COMPLETE);
        }

        // Get the next command in the list for updates and free current.
        amd::Command *next = current->getNext();
        current->release();
        current = next;
    }

    // Release the memory blocks allocated for the various
    // struct arguments of one or more kernel submissions
    std::for_each(kernelArgList_.begin(),
                  kernelArgList_.end(),
                  std::ptr_fun(servicesapi->HsaFreeSystemMemory));
    kernelArgList_.clear();

    // Reset the queue parameter
    lastSubmitQueue_ = static_cast<HsaQueueType>(0xFFFF);
}
}  // End of oclhsa namespace
