//
// Copyright 2010 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef OPERATION_HPP_
#define OPERATION_HPP_

#include "top.hpp"
#include "device/cpu/cpudevice.hpp"
#include "device/cpu/cpukernel.hpp"
#include "platform/command.hpp"
#include "thread/thread.hpp"
#include "os/os.hpp"
#include "amdocl/cl_kernel.h"
#include "device/cpu/ring.hpp"

#if defined(ATI_ARCH_ARM)
#include <setjmp.h>
#endif // ATI_ARCH_ARM

namespace cpu {

/*! \addtogroup CPU
 *  @{
 *
 *  \addtogroup CPUExec Execution environment
 *  @{
 */

//! A saved stack context
class StackContext : public amd::StackObject
{
private:
#if defined(ATI_ARCH_ARM)
    jmp_buf env_;
#elif defined(_WIN64)
    intptr_t __declspec(align(16)) regs_[32];
#else // !_WIN64
    intptr_t regs_[LP64_SWITCH(6,8)];
#endif // !_WIN64

public:
    //! Save the stack context. Return 0 if returning directly.
    inline intptr_t setjmp();

    //! Restore the stack context
    inline void longjmp(intptr_t val) const;
};

//! A thread fiber
class Fiber : public amd::StackObject
{
private:
    //! Next fiber in the thread.
    Fiber* next_;

    //! This fiber's saved state.
    StackContext context_;

public:
    //! Construct a new Fiber
    Fiber() : next_(NULL) { }

    //! Return the next fiber in the current thread.
    const Fiber* next() const { return next_; }
    //! Set the next fiber in the current thread.
    void setNext(Fiber* next) { next_ = next; }

    //! Save the state of this fiber. Return true if directly returning.
    ALWAYSINLINE bool save() { return context_.setjmp() == 0; }
    //! Restore this fiber from the saved context.
    void restore() const { context_.longjmp(1); }

    //! Switch to the given fiber.
    void swap(const Fiber* fiber) { if (save()) { fiber->restore(); } }
};



//! A CPU core operation (enqueued in the worker thread queue)
class Operation :  public amd::HeapObject
{
public:
    //! An atomic counter
    class Counter
    {
        // FIXME_lmoriche: recycle the counters, implement a thread local pool.
    private:
        amd::Event& event_;
        //! The atomic counter value.
        amd::Atomic<size_t> counter_;

    public:
        //! Initialize the counter with the given initial value.
        Counter(amd::Event& event, size_t initialValue) :
            event_(event), counter_(initialValue) { }
        //! Return the event associated with this counter.
        amd::Event& event() { return event_; }
        //! Decrement the counter and return the new value.
        size_t decrement() { return --counter_; }
    };

protected:
	amd::Command& command_; 

public:
    Operation(amd::Command& command) : command_(command)
    { }

    virtual ~Operation() {};

    virtual void clone(Operation* buf) = 0;

    void cleanup();

	amd::Command& command() { return command_;}

    virtual void execute() = 0;
};

/*! @}
 *  \defgroup CPUOperations Operations
 *  @{
 */

//! A work item instance
class WorkItem : public Fiber
{
private:
    //! Thread info block (must be the last field).
    clk_thread_info_block_t tib_;

private:
    //! Cannot be deleted (allocated with placement new).
    void operator delete(void*) { ShouldNotCallThis(); }

public:
    //! Initialize this workgroup.
    WorkItem(const amd::NDRangeContainer& size, void* localMemPtr);

    //! Return the current WorkItem (based of the current stack pointer).
    static WorkItem* current() {
        return (WorkItem*)amd::alignUp((intptr_t) amd::Os::currentStackPtr(),
            CLK_PRIVATE_MEMORY_SIZE) - 1;
    }

    clk_thread_info_block_t& infoBlock() { return tib_; }

    //! Return the native stack pointer base for this workitem.
    address nativeStackPtr() const {
        address newSp = amd::alignDown((address) this - CPUKERNEL_STACK_ALIGN,
                                       CPUKERNEL_STACK_ALIGN);
        WINDOWS_ONLY(NOT_WIN64(newSp += sizeof(void*)));
        return newSp;
    }

    //! These functions are mapping "n" from 1d index to the required dimension
    inline void setGroupId(
        const amd::NDRange& rangeLimits, 
        const amd::NDRange& offset, 
        size_t n);
    inline void incrementGroupId(
        const amd::NDRange& rangeLimits, 
        const amd::NDRange& offset, 
        size_t n);

    //! Execute a thread synchronization barrier.
    static void barrier(cl_mem_fence_flags flags);
};

typedef void (*kernelentrypoint_t)(const void*);

//! Execute a workgroup (work-items).
class WorkGroup
{
private:
    amd::NDRangeKernelCommand& command_;
    const cpu::Kernel& kernel_;
    WorkerThread& thread_;
    address params_;
    WorkItem* const workItem0_;
    const Fiber* workingFiber_;
    size_t numWorkItems_;

public:
    WorkGroup(
        amd::NDRangeKernelCommand& parent,
        const cpu::Kernel& kernel, 
        WorkerThread& thread,
        address params,
        WorkItem* workItem0,
        const size_t numWorkItems) :
            command_(parent), 
            kernel_(kernel), 
            thread_(thread),
            params_(params),
            workItem0_(workItem0),
            numWorkItems_(numWorkItems)
    { }

    WorkItem* getBaseWorkItem() { return workItem0_; }
    WorkerThread& getWorkerThread() { return thread_; }

    void executeWorkItem(); // In case of 1 WorkItem
    void executeWithBarrier();
    void executeWithoutBarrier();

    void setNumWorkItems(size_t workItems) { numWorkItems_ = workItems; }
    size_t getNumWorkItems() { return numWorkItems_; }
private:
    void callKernelRange(
        kernelentrypoint_t entryPoint,
        address stackPtr,
        clk_thread_info_block_t& tib);
    inline void callKernel(
        kernelentrypoint_t entryPoint,
        address stackPtr);
    inline void callKernelProtectedReturn(
        kernelentrypoint_t entryPoint,
        address stackPtr);
};

class NDRangeKernelBatch : public Operation
{
protected:
    size_t coreId_;
    const size_t numWorkGroups_;
    const size_t numCores_;
    volatile size_t currentOpId_;
    const amd::NDRange groupIds_; //!< Number of groups in each dimensions
    VirtualCPU& virtualDevice_;

public:
    enum ExecutionOrder {
        ORDER_DEFAULT,
        ORDER_ROUND_ROBIN = ORDER_DEFAULT,
        //ORDER_LINEAR
    };

    enum ExecutionNature {
        NATURE_WITH_BARRIER,
        NATURE_WITHOUT_BARRIER,
        NATURE_1_WORK_ITEM,
        NATURE_WG_LEVEL_EXEC
    };

    NDRangeKernelBatch(
        amd::NDRangeKernelCommand& parent,
        VirtualCPU& virtualDevice,
        const amd::NDRange& groupIds, size_t numCores) :
            Operation(parent), 
            coreId_(0),
            numWorkGroups_(groupIds.product()),
            numCores_(numCores),
            currentOpId_(0),
            groupIds_(groupIds), 
            virtualDevice_(virtualDevice)
    { }

    virtual void clone(Operation* buf)
    {
        ::new(buf) NDRangeKernelBatch(static_cast<amd::NDRangeKernelCommand&>(command_), 
            virtualDevice_, groupIds_, numCores_);
        static_cast<NDRangeKernelBatch*>(buf)->setCoreId(coreId_);
    }

    virtual void execute();

    void setCoreId(size_t coreId) { coreId_ = coreId; currentOpId_ = coreId; }

    inline bool getNextOperationId(size_t& opId);
    inline size_t getNextOperationIds(size_t& opId, size_t count);

private:
    bool patchParameters(
        const cpu::Kernel& kernel,
        address params,
        address& localMemPtr,
        const address localMemLimit,
        size_t localMemSize) const;
};

class NativeFn : public Operation
{
public:
    NativeFn(amd::NativeFnCommand& parent) : Operation(parent)
    { }

    virtual void clone(Operation* buf)
    {
        ::new(buf) NativeFn(static_cast<amd::NativeFnCommand&>(command_));
    }

    virtual void execute();
};
#ifndef MAX
#define MAX(x,y) ((x)>=(y) ?(x) : (y))
#endif //MAX

#define MAX_OPERATION_ALLOC_SIZE (MAX(sizeof(NDRangeKernelBatch), sizeof(NativeFn)))

//! A thread bound to a cpu core.
class WorkerThread : public amd::Thread
{
private:
	Fiber mainFiber_; //!< main fiber for this worker thread.

	amd::Monitor queueLock_; //!< lock protecting the queue.
    volatile int waitingOp_;
	bool terminated_; //!< true if the thread is shutting down.
	
	//! Local memory storage
	address localDataStorage_;
	//! Size of the local memory.
	size_t localDataSize_;

    char operation_[MAX_OPERATION_ALLOC_SIZE];

    address baseWorkItemsStack_;
private:
	//! Awaits operations and execute them as they become ready.
	void loop();

public:
	//! Construct a new WorkerThread.
	WorkerThread(const cpu::Device& device);
	//! Destroy the worker thread.
	virtual ~WorkerThread();
	//! Cleanup the thread before termination.
	bool terminate();

	//! Return the main fiber for this thread.
	Fiber& mainFiber() { return mainFiber_; }
	//! Return the LDS for this thread
	address localDataStorage() const { return localDataStorage_; }
	//! Return the size of the local memory for this thread.
	size_t localDataSize() const { return localDataSize_; }

    address baseWorkItemsStack() { return baseWorkItemsStack_; }

    Operation* operation() { return reinterpret_cast<Operation*>(operation_); }
    bool isOperationValid() { return waitingOp_ > 0; }

	//! Enqueue a new operation to execute in this thread.
	void enqueue(Operation& op);
	//! Signal to start processing the commands in the queue.
	void flush() { amd::ScopedLock sl(queueLock_); queueLock_.notify(); }

	//! This thread's execution engine.
	void run(void* data) {
		loop();
	}

	//! Return the currently executing WorkerThread's instance.
	static WorkerThread* current()
	{
		return static_cast<WorkerThread*>(Thread::current());
	}
};

/*! @}
 *  @}
 */

extern "C" intptr_t _StackContext_setjmp(intptr_t* regs);

#if !defined(ATI_ARCH_ARM)
ALWAYSINLINE
#endif
intptr_t
StackContext::setjmp()
{
#if defined(ATI_ARCH_ARM)
    return ::setjmp(env_);
#else
    return _StackContext_setjmp(regs_);
#endif
}

extern "C" void _StackContext_longjmp(const intptr_t* env, intptr_t val);

ALWAYSINLINE void
StackContext::longjmp(intptr_t val) const
{
#if defined(ATI_ARCH_ARM)
    return ::longjmp(*const_cast<jmp_buf*>(&env_), val);
#else
    return _StackContext_longjmp(regs_, val);
#endif
}



extern "C" void _WorkGroup_callKernel( 
    address params,
    kernelentrypoint_t entryPoint,
    address stackPtr);

extern "C" void _WorkGroup_callKernelProtectedReturn( 
    address params,
    kernelentrypoint_t entryPoint,
    address stackPtr);


ALWAYSINLINE void
WorkGroup::callKernel(
    kernelentrypoint_t entryPoint,
    address stackPtr)
{
    _WorkGroup_callKernel(params_, entryPoint, stackPtr);
}

// This version support the case of changing the stack for fibers.
ALWAYSINLINE void
WorkGroup::callKernelProtectedReturn(
    kernelentrypoint_t entryPoint,
    address stackPtr)
{
    _WorkGroup_callKernelProtectedReturn(params_, entryPoint, stackPtr);
}


} // namespace cpu

#endif /*OPERATION_HPP_*/
