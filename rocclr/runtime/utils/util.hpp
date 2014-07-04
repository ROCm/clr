//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include "top.hpp"
#include "thread/atomic.hpp"

#include <string>

namespace amd {

/*! \addtogroup Utils Utilities
 *  @{
 */

//! \brief Check if the given value \a val is a power of 2.
template <typename T>
static inline bool
isPowerOfTwo(T val)
{
    return (val & (val - 1)) == 0;
}

//! \cond ignore

// Compute the next power of 2 helper.
template <uint N>
struct NextPowerOfTwoFunction
{
    template <typename T>
    static T compute(T val)
    {
        val = NextPowerOfTwoFunction<N/2>::compute(val);
        return (val >> N) | val;
    }
};

// Specialized version for <1> to break the recursion.
template <>
struct NextPowerOfTwoFunction<1>
{
    template <typename T>
    static T compute(T val) { return (val >> 1) | val; }
};

template <uint N, int S>
struct NextPowerOfTwoHelper
{
    static const uint prev = NextPowerOfTwoHelper<N, S / 2>::value;
    static const uint value = (prev >> S) | prev;
};
template <uint N>
struct NextPowerOfTwoHelper<N, 1>
{
    static const int value = (N >> 1) | N;
};

template <uint N>
struct NextPowerOfTwo
{
    static const uint value = NextPowerOfTwoHelper<N-1, 16>::value + 1;
};

//! \endcond

/*! \brief Return the next power of two for a value of type T.
 *
 *  The compute function is (with n = sizeof(T)*8):
 *
 *    val = (val >> 1) | val;
 *    val = (val >> 2) | val;
 *    ...
 *    val = (val >> n/4) | val;
 *    val = (val >> n/2) | val;
 *
 *  The next power of two is: 1+compute(val-1)
 */
template <typename T>
inline T
nextPowerOfTwo(T val)
{
    return NextPowerOfTwoFunction<sizeof(T)*4>::compute(val - 1) + 1;
}

// Compute log2(N)
template <uint N>
struct Log2
{
    static const uint value = Log2<N/2>::value + 1;
};

// Break the recursion
template <>
struct Log2<1>
{
    static const uint value = 0;
};

/*! \brief Return the log2 for a value of type T.
 *
 * The compute function is (with n = sizeof(T)*8):
 *
 *   uint l = 0;
 *   if (val >= 1 << n/2) { val >>= n/2; l |= n/2; }
 *   if (val >= 1 << n/4) { val >>= n/4; l |= n/4; }
 *   ...
 *   if (val >= 1 << 2) { val >>= 2; l |= 2; }
 *   if (val >= 1 << 1) { l |= 1; }
 *   return l;
 */
template <uint N>
struct Log2Function
{
    template <typename T>
    static uint compute(T val)
    {
        uint l = 0;
        if (val >= T(1) << N) {
            val >>= N; l = N;
        }
        return l + Log2Function<N/2>::compute(val);
    }
};

template <>
struct Log2Function<1>
{
    template <typename T>
    static uint compute(T val) {
        return (val >= T(1)<<1) ? 1 : 0;
    }
};

// log2 helper function
template <typename T>
inline uint
log2(T val)
{
    return Log2Function<sizeof(T)*4>::compute(val);
}

template <typename T>
inline T
alignDown(T value, size_t alignment)
{
    return (T) (value & ~(alignment - 1));
}

template <typename T>
inline T*
alignDown(T* value, size_t alignment)
{
    return (T*) alignDown((intptr_t) value, alignment);
}

template <typename T>
inline T
alignUp(T value, size_t alignment)
{
    return alignDown((T) (value + alignment - 1), alignment);
}

template <typename T>
inline T*
alignUp(T* value, size_t alignment)
{
    return (T*) alignDown((intptr_t) (value + alignment - 1), alignment);
}

template<typename T>
inline bool isMultipleOf(T value, size_t alignment)
{
    if (isPowerOfTwo(alignment)) {
        // fast path, using logical operators
        return alignUp(value, alignment) == value;
    }
    return value % alignment == 0;
}

template<typename T>
inline bool isMultipleOf(T* value, size_t alignment)
{
    intptr_t ptr  = reinterpret_cast<intptr_t>(value);
    return isMultipleOf(ptr, alignment);
}

template <class T, class AllocClass = HeapObject>
struct SimplyLinkedNode : public AllocClass
{
    typedef SimplyLinkedNode<T, AllocClass> Node;

protected:
    Atomic<Node*> next_; /*!< \brief The next element. */
    T volatile item_;

public:
    //! \brief Return the next element in the linked-list.
    Node* next() const { return next_; }
    //! \brief Return the item.
    T item() const { return item_; }

    //! \brief Set the next element pointer.
    void setNext(Node* next) { next_ = next; }
    //! \brief Set the item.
    void setItem(T item) { item_ = item; }

    //! \brief Swap the next element pointer.
    Node* swapNext(Node* next) { return next_.swap(next); }

    //! \brief Compare and set the next element pointer.
    bool compareAndSetNext(Node* compare, Node* next)
    {
        return next_.compareAndSet(compare, next);
    }
};

/* For the implementation of a doubly-linked list, check:
 *        Lock-Free and Practical
 *        Deques and Doubly Linked
 *        Lists using Single-Word
 *        Compare-And-Swap
 *
 *        Hakan Sundell, Philippas Tsigas
 *        Department of Computing Science
 *        Chalmers Univ. of Technol. and Goteborg Univ.
 */

template <class T, class AllocClass = HeapObject>
struct DoublyLinkedNode
{
    typedef SimplyLinkedNode<T, AllocClass> Node;

protected:
    Atomic<Node*> prev_; //!< The previous element.
    Atomic<Node*> next_; //!< The next element.
    T volatile item_;

public:
    //! \brief Return the previous element in the linked-list.
    Node* prev() const { return prev_; }
    //! \brief Return the next element in the linked-list.
    Node* next() const { return next_; }
    //! \brief Return the item.
    T item() const { return item_; }

    //! \brief Set the previous element pointer.
    void setPrev(Node* prev) { prev_ = prev; }
    //! \brief Set the next element pointer.
    void setNext(Node* next) { next_ = next; }
    //! \brief Set the item.
    void setItem(T item) { item_ = item; }

    //! \brief Swap the previous element pointer.
    Node* swapPrev(Node* prev)
    {
        return prev_.swap(prev);
    }
    //! \brief Swap the next element pointer.
    Node*  swapNext( Node* next)
    {
        return next_.swap(next);
    }

    //! \brief Compare and set the previous element pointer.
    bool compareAndSetPrev(Node* compare, Node* prev)
    {
        return prev_.compareAndSet(compare, prev, false, false);
    }
    //! \brief Compare and set the next element pointer.
    bool compareAndSetNext(Node* compare, Node* next)
    {
        return next_.compareAndSet(compare, next, false, false);
    }
};

template <class Reference, class Value>
struct DeviceMap {
    Reference   ref_;
    Value       value_;
};


inline uint
countBitsSet32(uint32_t value)
{
#if __GNUC__ >= 4
    return (uint)__builtin_popcount(value);
#else
    value = value - ((value >> 1) & 0x55555555);
    value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
    return (uint)(((value + (value >> 4) & 0xF0F0F0F) * 0x1010101) >> 24);
#endif
}

inline uint
countBitsSet64(uint64_t value)
{
#if __GNUC__ >= 4
    return (uint)__builtin_popcountll(value);
#else
    value = value - ((value >> 1) & 0x5555555555555555ULL);
    value = (value & 0x3333333333333333ULL) + ((value >> 2) & 0x3333333333333333ULL);
    value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (uint)((uint64_t)(value * 0x0101010101010101ULL) >> 56);
#endif
}

inline uint
leastBitSet32(uint32_t value)
{
#if defined(_WIN32)
    unsigned long idx;
    return _BitScanForward(&idx, (unsigned long)value) ? idx : (uint)-1;
#else
    return value ? __builtin_ctz(value) : (uint)-1;
#endif
}

inline uint
leastBitSet64(uint64_t value)
{
#if defined(_WIN64)
    unsigned long idx;
    return _BitScanForward64(&idx, (unsigned __int64)value) ? idx : (uint)-1;
#elif defined (__GNUC__)
    return value ? __builtin_ctzll(value) : (uint)-1;
#else
    static const uint8_t lookup67[67+1] = {
        64,  0,  1, 39,  2, 15, 40, 23,
        3, 12, 16, 59, 41, 19, 24, 54,
        4, -1, 13, 10, 17, 62, 60, 28,
        42, 30, 20, 51, 25, 44, 55, 47,
        5, 32, -1, 38, 14, 22, 11, 58,
        18, 53, 63,  9, 61, 27, 29, 50,
        43, 46, 31, 37, 21, 57, 52,  8,
        26, 49, 45, 36, 56,  7, 48, 35,
        6, 34, 33, -1
    };

    return (uint)lookup67[((int64_t)value & -(int64_t)value) % 67];
#endif
}

template <typename T>
inline uint countBitsSet(T value)
{
    return (sizeof(T) == 8) ? countBitsSet64((uint64_t)value) : 
        countBitsSet32((uint32_t)value);
}

template <typename T>
inline uint leastBitSet(T value)
{
    return (sizeof(T) == 8) ? leastBitSet64((uint64_t)value) : 
        leastBitSet32((uint32_t)value);
}

static inline bool Is32Bits() {
    return LP64_SWITCH(true, false);
}

static inline bool Is64Bits() {
    return LP64_SWITCH(false, true);
}
/*@}*/} // namespace amd

#endif /*UTIL_HPP_*/
