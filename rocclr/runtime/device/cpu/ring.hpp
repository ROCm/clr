//
// Copyright 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef RING_BUFFER_HPP
#define RING_BUFFER_HPP

#include "top.hpp"
#include "thread/atomic.hpp"
#include "os/alloc.hpp"
// @brief Block-free ring buffer implemenation
// @brief THE RING BUFFER SUPPORTS MULTIPLE CONSUMERS AND A SINGLE PRODUCER.
// @tparam T Object-type to be saved within the ring buffer.
namespace amd{
    template <typename T>
    class RingBuffer
    {
    public:
        ///////////////////////////////////////
        // public initialization and cleanup //
        ///////////////////////////////////////
        RingBuffer();
        ~RingBuffer() { cleanup();}

        bool 
        initialize(unsigned short ringBufferSize);
        //////////////////////
        // public interface //
        ///////////////////////

        bool getNext(T & obj);
        bool insert(const T & obj);
    private:
        struct ABACounter{
            unsigned short tranactionId;
            unsigned short consumerIndex;
        };

        union Consumer {
            ABACounter abaCounter;
            volatile int32_t interlockedVar;
        };

        bool canInsert();

        template <typename T2> T2 incrementIndex(T2  index) {
            index++;
            if (index == ringBufferSize_)
                index = 0;
            return index;
        }
        ////////////////////////////////////
        // read only cache line for       //
        // producer and consumer threads  //
        ////////////////////////////////////
        T * ringBuffer_;
        unsigned short ringBufferSize_;

        char cachePad1_[64];
        /////////////////////////////////////
        // read/write cache line for       //
        // producer thread                 //
        /////////////////////////////////////
        //! producer is an index in the ring buffer array
        volatile int32_t producer_;
        //! caches the amount of free space in the buffer. reduces cache misses
        //! by reducing access to 'm_consumer' from producer thread 
        int32_t freeSpace_;

        char cachePad2_[64];
        /////////////////////////////////////
        // read/write cache line for       //
        // consumer threads                //
        /////////////////////////////////////
        volatile Consumer consumer_;

        /////////////////////////////////////
        // save the thread that inserts    //
        // for checking multiple producers //
        /////////////////////////////////////
        Thread *producerThread_;

        void cleanup();
        // do not allow copying
        RingBuffer(const RingBuffer&);
        RingBuffer& operator=(const RingBuffer&);
    };

    template <typename T>
    RingBuffer<T>::RingBuffer() :
        ringBuffer_(NULL),
        ringBufferSize_(0),
        producer_(0),
        freeSpace_(0),
        producerThread_(NULL)
    { 
        consumer_.interlockedVar = 0;
    }

    template <typename T>
    bool 
    RingBuffer<T>::initialize(unsigned short  ringBufferSize)
    {
        bool retVal = false;

        cleanup();
        ringBuffer_ = new T [ringBufferSize];
        if (ringBuffer_)
        {
            ringBufferSize_ = ringBufferSize;
            retVal = true;
        }
        return retVal;
    }

    template <typename T>
    void 
    RingBuffer<T>::cleanup()
    {
        if (ringBuffer_)
        {
            delete [] ringBuffer_;
            ringBuffer_ = NULL;
        }
        producer_ = 0;
        consumer_.interlockedVar = 0;
        freeSpace_ = 0;
        ringBufferSize_ = 0;
    }

    template <typename T>
    bool 
    RingBuffer<T>::insert(const T & obj)
    {
#ifdef DEBUG
        // if this is the 1st insert, set producerThread_
        if (NULL == producerThread_) {
            producerThread_ = Thread::current();
        } else {
            assert(Thread::current() == producerThread_ && "not a single writer");
        }
#endif //DEBUG
        bool retVal = false;
        if (canInsert())
        {
            ringBuffer_[producer_] = obj;

            producer_ = incrementIndex(producer_);
            retVal = true;
        }
        return retVal;
    }

    template <typename T>
    bool 
    RingBuffer<T>::getNext( T & obj)
    {

        Consumer consumer;
        consumer.interlockedVar = consumer_.interlockedVar;
        //cache the producer variable on the stack  
        int producer = producer_;
        //while the buffer is not empty
        while (producer != consumer.abaCounter.consumerIndex)
        {
            obj = ringBuffer_[consumer.abaCounter.consumerIndex];
            Consumer newConsumer;
            newConsumer.abaCounter.consumerIndex = incrementIndex(consumer.abaCounter.consumerIndex);
            newConsumer.abaCounter.tranactionId = consumer.abaCounter.tranactionId+1;

            if (consumer.interlockedVar == amd::AtomicOperation::compareAndSwap(consumer.interlockedVar,
                &(consumer_.interlockedVar),newConsumer.interlockedVar))
            {
                return true;
            }

            consumer.interlockedVar = consumer_.interlockedVar;
            producer = producer_;
        }
        return false;
    }

    template <typename T>
    bool 
    RingBuffer<T>::canInsert()
    {
        if (freeSpace_ > 1)
        {
            freeSpace_--;
            return true;
        }
        //cache the volatile variable on the stack;
        int32_t consumer = consumer_.abaCounter.consumerIndex;

        //there will alway be one unused cell in the array
        //to distinguish between the case it is completely full and completely empty
        freeSpace_ = consumer - producer_ - 1  ;
        if ( freeSpace_ <= -1 )
        {
            freeSpace_ = ringBufferSize_ + freeSpace_;
        }
        return (freeSpace_ > 0) ;
    }

}//NAMESPACE AMD

#endif // RING_BUFFER_HPP
