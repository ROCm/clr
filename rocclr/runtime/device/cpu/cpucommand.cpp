//
// Copyright 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpucommand.hpp"
#include "device/cpu/cpubuiltins.hpp"
#include "device/cpu/cpudevice.hpp"
#include "device/cpu/cputables.hpp"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "platform/sampler.hpp"
#include "thread/thread.hpp"
#include "os/os.hpp"
#include "utils/util.hpp"
#include "utils/options.hpp"

#include <amdocl/cl_kernel.h>
#include <algorithm>

namespace cpu {

#define CPU_WORKER_THREAD_TOTAL_STACK_SIZE (CPU_WORKER_THREAD_STACK_SIZE + \
    CLK_PRIVATE_MEMORY_SIZE * (CPU_MAX_WORKGROUP_SIZE + 1))

WorkerThread::WorkerThread(const cpu::Device& device) :
    Thread("CPU Worker Thread", CPU_WORKER_THREAD_TOTAL_STACK_SIZE),
    queueLock_("WorkerThread::queueLock"), waitingOp_(0), terminated_(false)
{
    localDataSize_ = (size_t) device.info().localMemSize_;
    localDataStorage_ = (address) amd::AlignedMemory::allocate(
        localDataSize_ + __CPU_SCRATCH_SIZE, sizeof(cl_long16));

#if defined(__linux__) && defined(NUMA_SUPPORT)
    const nodemask_t* numaMask = device.getNumaMask();
    if (numaMask != NULL) {
        numa_bind(numaMask);
    }
#endif
}

WorkerThread::~WorkerThread()
{
    guarantee(Thread::current() != this && "thread suicide!");
    amd::AlignedMemory::deallocate(localDataStorage_);
}

bool
WorkerThread::terminate()
{
    terminated_ = true;

    if (Thread::current() != this) {
        // FIXME_lmoriche: fix termination handshake
        while (state() < Thread::FINISHED) {
            flush();
            amd::Os::yield();
        }
    }

    return true;
}

void 
WorkerThread::enqueue(Operation& op)
{
    while (waitingOp_ != 0) {
        amd::Os::yield();
    }
    op.clone(operation());
    ++waitingOp_;
}

void
WorkerThread::loop()
{
    baseWorkItemsStack_ = amd::alignDown(stackBase() -
        CPU_WORKER_THREAD_STACK_SIZE, CLK_PRIVATE_MEMORY_SIZE);
#if defined(WIN32)
    amd::Os::touchStackPages(baseWorkItemsStack_, amd::Os::currentStackPtr());
#endif // WINDOWS
    Operation *op = operation();

    queueLock_.lock();
    while (true) {
        while (waitingOp_ == 0) {
            if (terminated_) {
                break;
            }
            queueLock_.wait();
        }
        if (terminated_) {
            break;
        }
        op->command().setStatus(CL_RUNNING);
        op->execute();
        op->cleanup();
        --waitingOp_;
    }
    queueLock_.unlock();
}

void
NativeFn::execute()
{
    cl_int status = static_cast<amd::NativeFnCommand&>(command()).invoke();
    command().setStatus(status);
}

static void
nop() { /*Do nothing*/ }


template <NDRangeKernelBatch::ExecutionNature NATURE,
          NDRangeKernelBatch::ExecutionOrder ORDER =
              NDRangeKernelBatch::ORDER_DEFAULT>
class NDRangeKernelBatchMode : public NDRangeKernelBatch
{
private:
    void executeWorkGroup(WorkGroup& wg)
    {
        if (NATURE == NATURE_WG_LEVEL_EXEC) {
            wg.executeWorkItem();
        }
        else if ((NATURE == NATURE_1_WORK_ITEM) ||
                 (wg.getNumWorkItems() == 1)) {
            wg.executeWorkItem();
        }
        else {
            wg.getBaseWorkItem()->setNext(&wg.getWorkerThread().mainFiber());
            if (NATURE == NATURE_WITHOUT_BARRIER) {
                wg.executeWithoutBarrier();
            }
            else { // NATURE == NATURE_WITH_BARRIER
                wg.executeWithBarrier();
            }
        }
        // Yield at the end of each workgroup to avoid starving GPU device
        amd::Os::yield();
    }

public:
    void executeMode(WorkGroup& wg)
    {
        const amd::NDRange& offset =
            static_cast<amd::NDRangeKernelCommand&>(command_).sizes().offset();
        WorkItem* workItem0 = wg.getBaseWorkItem();
        clk_builtins_t tableTask;
        size_t prevOpId = 0, opId = (size_t)-1;

        if (NATURE == NATURE_1_WORK_ITEM) {
            tableTask = Builtins::dispatchTable_;

            // If local size == 1 then barrier() becomes a nop.
            tableTask.barrier_ptr = (void (*)(cl_mem_fence_flags)) nop;
            workItem0->infoBlock().builtins = &tableTask;
            workItem0->setNext(&wg.getWorkerThread().mainFiber());
        }

        while (getNextOperationId(opId)) {
            workItem0->incrementGroupId(groupIds_, offset, opId - prevOpId);
            uint workDims = workItem0->infoBlock().work_dim;
            size_t numWorkItems = workItem0->infoBlock().local_size[0] *
              (workDims >= 2 ? workItem0->infoBlock().local_size[1] : 1) *
              (workDims >= 3 ? workItem0->infoBlock().local_size[2] : 1);
            wg.setNumWorkItems(numWorkItems);
            if(numWorkItems == 1) {
              tableTask = Builtins::dispatchTable_;
              tableTask.barrier_ptr = (void (*)(cl_mem_fence_flags)) nop;
              workItem0->infoBlock().builtins = &tableTask;
              workItem0->setNext(&wg.getWorkerThread().mainFiber());
              executeWorkGroup(wg);
              tableTask.barrier_ptr = &WorkItem::barrier;
            } else {
              executeWorkGroup(wg);
            }
            prevOpId = opId;
        }

//#define DISABLE_TASK_STEALING
#if !defined(DISABLE_TASK_STEALING) && 0
        size_t maxId = numCores_;
        size_t stolenId = coreId_ + 1;
        NDRangeKernelBatch* workingBatch = this;
        size_t numStolenIds = 1;
        const size_t maxStealingSize = 3;
        const size_t minAdaptiveStealingDiff = numCores_ * maxStealingSize;
    
        while (true) {
            for (; stolenId < maxId; ++stolenId) {
                WorkerThread* worker = virtualDevice_.getWorkerThread(stolenId);

                // In case were we have less operations than Worker Threads
                if (worker->isOperationValid()) {
                    workingBatch = static_cast<NDRangeKernelBatch*>(
                        worker->operation());

                    numStolenIds =
                        workingBatch->getNextOperationIds(opId, numStolenIds);
                    if (numStolenIds > 0) {
                        do {
                            for (size_t i = 0; i < numStolenIds; ++i) {
                                workItem0->setGroupId(groupIds_, offset, opId);
                                executeWorkGroup(wg);
                                opId += numCores_;
                            }

                            // adaptive stealing
                            if (numWorkGroups_ - opId > minAdaptiveStealingDiff) {
                                numStolenIds = maxStealingSize;
                            }
                            else {
                                while (workingBatch->getNextOperationId(opId)) {
                                    workItem0->setGroupId(groupIds_, offset, opId);
                                    executeWorkGroup(wg);
                                }
                                break;
                            }
                            numStolenIds = workingBatch->getNextOperationIds(
                                opId, numStolenIds);
                        } while (numStolenIds > 0);
                    }
                    numStolenIds = 1;
                }
            } // for (stolenId..maxId)

            if (stolenId == coreId_) {
                break;
            }

            stolenId = 0;
            maxId = coreId_;
        } // while (true)
#endif
    }
};


inline bool
NDRangeKernelBatch::getNextOperationId(size_t& opId)
{
    if (currentOpId_ >= numWorkGroups_) {
        return false;
    }
    opId = amd::AtomicOperation::add(numCores_, &currentOpId_);
    return opId < numWorkGroups_;
}


inline size_t
NDRangeKernelBatch::getNextOperationIds(size_t& opId, size_t count)
{
    size_t topId = numCores_ * count;
    if (currentOpId_ >= numWorkGroups_) {
        return 0;
    }

    opId = amd::AtomicOperation::add(topId, &currentOpId_);
    const size_t numWorkGroups = numWorkGroups_;
    if (opId >= numWorkGroups) {
        return 0;
    }

    topId += opId;
    if (topId >= (numWorkGroups + numCores_)) {
        count -= (topId - numWorkGroups) / numCores_;
    }

    return count;
}

// Process the parameters, allocate LDS.
bool
NDRangeKernelBatch::patchParameters(
    const cpu::Kernel& cpuKernel,
    address params,
    address& localMemPtr,
    const address localMemLimit,
    size_t localMemSize) const
{
    amd::NDRangeKernelCommand& command =
        static_cast<amd::NDRangeKernelCommand&>(command_);

    const amd::Device& device = command.queue()->device();

    const amd::Kernel& kernel = command.kernel();
    const amd::KernelSignature& signature = kernel.signature();
    const amd::KernelParameters& kernelParam = kernel.parameters();

    const_address cmdParams = command.parameters();

    unsigned effectiveOffset = 0;

    // DD -- on CPU device, real effective offset is NATIVELY aligned
    // Here all source arguments are in place, so we're safe just iterating
    for (size_t i = 0; i < signature.numParameters(); ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        const void* cmdParam = cmdParams + desc.offset_;
        void    *param;
        size_t   prmSize = cpuKernel.getArgSize(i);

        // Align i'th parameter on multiple of its size. Parameter size is power of 2.
        size_t alignment = cpuKernel.getArgAlignment(i);
        effectiveOffset = amd::alignUp(effectiveOffset, std::min(alignment, size_t(16)));
        param = params + effectiveOffset;
        if (desc.size_ == 0) {
            // __local memory parameter
            localMemPtr = amd::alignUp(localMemPtr, sizeof(cl_long16));

            size_t length = *static_cast<const size_t*>(cmdParam);
            *static_cast<void**>(param) = localMemPtr;
            localMemPtr += length;

            if (localMemPtr > localMemLimit) {
                command.setException(CL_MEM_OBJECT_ALLOCATION_FAILURE);
                return false;
            }
        }
        else if (desc.type_ == T_POINTER) {
            // __global memory parameter
            cl_mem_object_type pointer_type = CL_MEM_OBJECT_BUFFER;
            if (kernelParam.boundToSvmPointer(device, cmdParams, i)) {
                *reinterpret_cast<void**>(param) =
                        *reinterpret_cast<void* const *>(cmdParam);
            }
            else {
                void* hostMemPtr = NULL;
                amd::Memory* memArg =
                    *reinterpret_cast<amd::Memory* const *>(cmdParam);
                if (memArg != NULL) {
                    hostMemPtr = memArg->getHostMem();
                    if (hostMemPtr == NULL) {
                        command.setException(CL_MEM_OBJECT_ALLOCATION_FAILURE);
                        return false;
                    }
                    pointer_type = memArg->getType();
                }
                // For images on CPU devices, pass "struct {int4 p0; int4 p1}".
                // That allows an obvious implementation for
                // __amdil_get_image[23]d_params[01].
                // That makes the rest of the .bc implementation for
                // images relatively straight forward.
                if (pointer_type == CL_MEM_OBJECT_IMAGE1D ||
                    pointer_type == CL_MEM_OBJECT_IMAGE2D ||
                    pointer_type == CL_MEM_OBJECT_IMAGE3D ||
                    pointer_type == CL_MEM_OBJECT_IMAGE1D_ARRAY  ||
                    pointer_type == CL_MEM_OBJECT_IMAGE1D_BUFFER ||
                    pointer_type == CL_MEM_OBJECT_IMAGE2D_ARRAY) {
                    amd::Image::Impl& impl = memArg->asImage()->getImpl();
                    impl.reserved_ = hostMemPtr;
                    *reinterpret_cast<void**>(param) = (void*)&impl;
                } else {
                    *reinterpret_cast<void**>(param) = hostMemPtr;
                }
            }
        }
        else if (desc.type_ == T_SAMPLER) {
            // Switch from an Amd::Sampler to the 32bit integer
            // variable that is a clk_sampler.
            amd::Sampler* samplerArg =
                *reinterpret_cast<amd::Sampler* const *>(cmdParam);
            *reinterpret_cast<uint32_t*>(param) = (uint32_t)samplerArg->state();
        }
        else {
            //Using HCtoDCmap
            HCtoDCmap arg_map = cpuKernel.getHCtoDCmap(i);
            unsigned int arg_offset = effectiveOffset;
            int err_code = 0;
            int inStruct = 0;
            int sys_64bit = LP64_SWITCH(0, 1);  // Mapping only required for 32 bit targets
            if (CPU_USE_ALIGNMENT_MAP == 0 && !sys_64bit) {
               effectiveOffset += arg_map.copy_params(param, cmdParam, arg_offset, err_code, inStruct);
               if (err_code) {
                   return false;
               }
               prmSize = arg_map.dc_size;
            }
            else {
                ::memcpy(param, cmdParam, desc.size_);
            }
        }
        effectiveOffset += prmSize;
    }

    localMemPtr = amd::alignUp(localMemPtr, sizeof(cl_long16));
    if ((localMemPtr + localMemSize) > localMemLimit) {
        command.setException(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        return false;
    }

    return true;
}

void
NDRangeKernelBatch::execute()
{
    amd::NDRangeKernelCommand& command =
        static_cast<amd::NDRangeKernelCommand&>(command_);

    const cpu::Kernel& kernel = static_cast<const cpu::Kernel&>(
                  *command.kernel().getDeviceKernel(command.queue()->device()));

    WorkerThread& thread = *WorkerThread::current();

    const size_t numWorkItems = command.sizes().local().product();

    address params = thread.baseWorkItemsStack();
    address baseLocalMemPtr = thread.localDataStorage();
    address patchedLocalMemPtr = thread.localDataStorage() + __CPU_SCRATCH_SIZE;
    if (!patchParameters(kernel, params,
        patchedLocalMemPtr, patchedLocalMemPtr + thread.localDataSize(),
        kernel.workGroupInfo()->localMemSize_)) {
        return;
    }

    WorkItem* workItem0 = ::new((WorkItem*)params - 1) WorkItem(
        command.sizes(), baseLocalMemPtr, patchedLocalMemPtr);

    WorkGroup wg(command, kernel, thread, params, workItem0, numWorkItems);

    if (numWorkItems == 1) {
        static_cast<NDRangeKernelBatchMode<NATURE_1_WORK_ITEM>*>(this)->
            executeMode(wg);
    }
    else if (kernel.hasBarrier()) {
        static_cast<NDRangeKernelBatchMode<NATURE_WITH_BARRIER>*>(this)->
            executeMode(wg);
    }
    else {
        static_cast<NDRangeKernelBatchMode<NATURE_WITHOUT_BARRIER>*>(this)->
            executeMode(wg);
    }
}

void
WorkGroup::executeWorkItem()
{
    callKernel((kernelentrypoint_t)kernel_.getEntryPoint(), workItem0_->nativeStackPtr());
}

void
WorkGroup::executeWithBarrier()
{
    kernelentrypoint_t entryPoint = (kernelentrypoint_t)kernel_.getEntryPoint();

    workingFiber_ = workItem0_;
    address workGroupStackPtr = workItem0_->nativeStackPtr();

    // Save the current stack context in case we execute a barrier.
    volatile size_t threadCounter = 0;
    bool barrier = !thread_.mainFiber().save();

    size_t tid = threadCounter++;
    WorkItem* workItem = (WorkItem*)((char*) workItem0_
        - tid * CLK_PRIVATE_MEMORY_SIZE);

    if (barrier) {
        WorkItem* prev = (WorkItem*)((char*) workItem
            + CLK_PRIVATE_MEMORY_SIZE);

        WINDOWS_ONLY(amd::Os::touchStackPages(
            (address) (workItem + 1), (address) prev));
        ::memcpy(workItem, prev, sizeof(WorkItem));

        clk_thread_info_block_t& tib = workItem->infoBlock();
        ++tib.local_id[0];
        if (unlikely(tib.local_id[0] >= tib.local_size[0])) {
            //
            // Compiling for Windows 64bit (only in release) introduces a bug,
            // which uses the same register for saving threadCounter and the
            // 0 value. Therefore "tib.local_id[i] = 0" was actually translated
            // to "tib.local_id[0] = threadCounter". To avoid this issue, and
            // still be able to store a 0 into tib.local_id[i], we trick the
            // compiler, by using the value in tib.local_id[3], which is always
            // initialized to 0.
            //
            tib.local_id[0] = tib.local_id[3];

            ++tib.local_id[1];
            if (unlikely(tib.local_id[1] >= tib.local_size[1])) {
                tib.local_id[1] = tib.local_id[3];

                ++tib.local_id[2];
            }
        }

        // Link the previous workitem to this one.
        prev->setNext(workItem);
        // If this is the last workitem, complete the ring.
        if (tid >= numWorkItems_ - 1) {
            workItem->setNext(workItem0_);
        }
    }

    // Execute thread0

    address workItemStackPtr = workItem->nativeStackPtr();
    callKernelProtectedReturn(entryPoint, workItemStackPtr);

    // Check if thread0 executed a barrier()
    if (threadCounter > 1) {
        workItem = (WorkItem*)workingFiber_;
        workingFiber_ = workingFiber_->next();

        tid = ((address)workItem0_ - (address)workItem)
            / CLK_PRIVATE_MEMORY_SIZE;
        if (tid == (numWorkItems_ - 1)) {
            // If we get here, we are done!
            return;
        }
        if (workItem->next() == &thread_.mainFiber()) {
            // Detected a deadlock
            command_.setException(CL_INVALID_KERNEL);
            return;
        }

        // Schedule the next workitem.
        workItem->next()->restore();
        ShouldNotReachHere();
    }

    // Execute thread1...threadN
    callKernelRange(entryPoint, workItemStackPtr, workItem->infoBlock());
}

void
WorkGroup::executeWithoutBarrier()
{
    kernelentrypoint_t entryPoint = (kernelentrypoint_t)kernel_.getEntryPoint();
    address workItemStackPtr = workItem0_->nativeStackPtr();

    // Execute thread0
    callKernel(entryPoint, workItemStackPtr);

    // Execute thread1...threadN
    callKernelRange(entryPoint, workItemStackPtr, workItem0_->infoBlock());
}

void
WorkGroup::callKernelRange(kernelentrypoint_t entryPoint,
                           address stackPtr,
                           clk_thread_info_block_t& tib)
{
    while (true) {
        ++tib.local_id[0];
        if (unlikely(tib.local_id[0] >= tib.local_size[0])) {
            tib.local_id[0] = 0;

            ++tib.local_id[1];
            if (unlikely(tib.local_id[1] >= tib.local_size[1])) {
                tib.local_id[1] = 0;

                ++tib.local_id[2];
                if (unlikely(tib.local_id[2] >= tib.local_size[2])) {
                    tib.local_id[2] = 0;

                    return;
                }
            }
        }

        callKernel(entryPoint, stackPtr);
    }
}

WorkItem::WorkItem(const amd::NDRangeContainer& sizes,
                   void* scratchMemPtr,
                   void* localMemPtr)
{
    const amd::NDRange& local = sizes.local();
    const amd::NDRange& global = sizes.global();
    const amd::NDRange& offset = sizes.offset();
    const size_t dims = sizes.dimensions();

    tib_.builtins = &Builtins::dispatchTable_;
    tib_.local_mem_base = localMemPtr;
    tib_.local_scratch = scratchMemPtr;
    tib_.table_base = (const void *)cpuTables;
    tib_.work_dim = (cl_uint) sizes.dimensions();

    for (size_t i = 0; i < dims; ++i) {
        tib_.global_offset[i] = offset[i];
        tib_.global_size[i] = global[i];
        tib_.local_size[i] =  local[i];
        tib_.enqueued_local_size[i] =  local[i];
        tib_.local_id[i] = 0;
        tib_.group_id[i] = 0;
    }

    // Fill the remaining dimensions.
    for (size_t i = dims; i < sizeof(tib_.global_size)/sizeof(size_t); ++i) {
        tib_.global_offset[i] =  0;
        tib_.global_size[i] = 1;
        tib_.local_size[i] =  1;
        tib_.enqueued_local_size[i] =  1;
        tib_.local_id[i] = 0;
        tib_.group_id[i] = 0;
    }
}

ALWAYSINLINE void
WorkItem::setGroupId(
    const amd::NDRange& rangeLimits, 
    const amd::NDRange& offset, 
    size_t n)
{
    const size_t dims = rangeLimits.dimensions();
    for (size_t i = 0; i < dims; ++i) {
        size_t lim = rangeLimits[i];
        size_t& val = tib_.group_id[i];
        val = n;
        if (n < lim) {
            tib_.global_offset[i] =
              offset[i] + val * tib_.enqueued_local_size[i];
            tib_.local_id[i] = 0;
            tib_.local_size[i] =
              std::min(tib_.enqueued_local_size[i],
                       tib_.global_size[i] - (val * tib_.enqueued_local_size[i]));

            ++i;
            for (; i < dims; ++i) {
                tib_.global_offset[i] = offset[i];
                tib_.local_id[i] = 0;
                tib_.group_id[i] = 0;
            }
            break;
        }
        else {
            n /= lim;
            val -= n * lim;
            tib_.global_offset[i] =
              offset[i] + val * tib_.enqueued_local_size[i];
            tib_.local_id[i] = 0;
            tib_.local_size[i] =
              std::min(tib_.enqueued_local_size[i],
                       tib_.global_size[i] - (val * tib_.enqueued_local_size[i]));
        }
    }
}

ALWAYSINLINE void
WorkItem::incrementGroupId(
    const amd::NDRange& rangeLimits, 
    const amd::NDRange& offset, 
    size_t n)
{
    const size_t dims = rangeLimits.dimensions();
    for (size_t i = 0; i < dims; ++i) {
        size_t lim = rangeLimits[i];
        size_t& val = tib_.group_id[i];
        val += n;
        if (val < lim) {
            tib_.global_offset[i] =
              offset[i] + val * tib_.enqueued_local_size[i];
            tib_.local_id[i] = 0;
            tib_.local_size[i] =
              std::min(tib_.enqueued_local_size[i],
                       tib_.global_size[i] - (val * tib_.enqueued_local_size[i]));
            break;
        }
        else {
            n = val / lim;
            val -= n * lim;
            tib_.global_offset[i] =
              offset[i] + val * tib_.enqueued_local_size[i];
            tib_.local_id[i] = 0;
            tib_.local_size[i] =
              std::min(tib_.enqueued_local_size[i],
                       tib_.global_size[i] - (val * tib_.enqueued_local_size[i]));
        }
    }
}

void
WorkItem::barrier(cl_mem_fence_flags flags)
{
    WorkItem* workItem = WorkItem::current();
    workItem->swap(workItem->next());
}

void Operation::cleanup()
{
    cl_int lastException = command().exception();
    cl_int status = (lastException != 0) ? lastException : CL_COMPLETE;

    Counter* counter = reinterpret_cast<Counter*>(command().data());
    if (counter == NULL) {
        command().setStatus(status);
    }
    else if (counter->decrement() == 0) {
        counter->event().setStatus(status);
    }
}

} // namespace cpu
