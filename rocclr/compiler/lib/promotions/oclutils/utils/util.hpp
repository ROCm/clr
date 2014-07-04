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

//! \cond ignore
template <int N>
struct PairElement;

template <>
struct PairElement<0>
{
    template <class F, class S>
    static inline F& get(pair<F,S>& p) { return p.first; }
    template <class F, class S>
    static inline const F& get(const pair<F,S>& p) { return p.first; }
};

template <>
struct PairElement<1>
{
    template <class F, class S>
    static inline S& get(pair<F,S>& p) { return p.second; }
    template <class F, class S>
    static inline const S& get(const pair<F,S>& p) { return p.second; }
};

// Forward declaration of the tuple_elements container class.
template <class H, class T>
struct TupleElementsContainer;

/*! \brief Return the type of the Nth element in the tuple.
 */
template <int N, class T>
struct TupleElementType
{
    typedef typename T::tail_t next_element;
    typedef typename TupleElementType<N-1,next_element>::type type;
};

// break the recursion
template <class T>
struct TupleElementType<0,T>
{
    typedef Null next_element;
    typedef typename T::head_t type;
};

/*! \brief Helper struct to extract the Nth element from a tuple
 */
template <int N>
struct TupleElementGetter
{
    template <class R, class H, class T>
    static R get(TupleElementsContainer<H,T>& t)
    {
        return TupleElementGetter<N-1>::template get<R>(t.tail);
    }
    template <class R, class H, class T>
    static R get(const TupleElementsContainer<H,T>& t)
    {
        return TupleElementGetter<N-1>::template get<R>(t.tail);
    }
};

// break the recursion
template <>
struct TupleElementGetter<0>
{
    template <class R, class H, class T>
    static R get(TupleElementsContainer<H,T>& t)
    {
        return t.head;
    }
    template <class R, class H, class T>
    static R get(const TupleElementsContainer<H,T>& t)
    {
        return t.head;
    }
};

/*! \brief Return the Nth element in the tuple.
 */
template <int N, class H, class T>
inline typename TupleElementType<N,TupleElementsContainer<H,T> >::type&
getTupleElement(TupleElementsContainer<H,T>& t)
{
    return TupleElementGetter<N>::template get<
        typename TupleElementType<N,TupleElementsContainer<H,T> >::type&,
        H,T>(t);
}

template <int N, class H, class T>
inline const typename TupleElementType<N,TupleElementsContainer<H,T> >::type&
getTupleElement(const TupleElementsContainer<H, T>& t)
{
    return TupleElementGetter<N>::template get<
        const typename TupleElementType<N,TupleElementsContainer<H,T> >::type&,
        H,T>(t);
}

/*! \brief The tuple elements struct
 */
template <class H, class T>
struct TupleElementsContainer
{
    typedef H head_t;
    typedef T tail_t;

    head_t head; tail_t tail;

    TupleElementsContainer() : head(), tail() { }

    template <class T0, class T1, class T2, class T3>
    TupleElementsContainer(T0& t0, T1& t1, T2& t2, T3& t3)
        : head(t0), tail(t1, t2, t3, null())
    { }

    template <class HH, class TT>
    TupleElementsContainer& operator= (const TupleElementsContainer<HH,TT>& t)
    {
        head = t.head;
        tail = t.tail;
        return *this;
    }

    template <class F, class S>
    TupleElementsContainer& operator= (const pair<F,S>& p)
    {
        head = p.first;
        tail.head = p.second;
        return *this;
    }

    template<int N>
    typename TupleElementType<N, TupleElementsContainer>::type&
    get() { return getTupleElement<N>(*this); }
};

// break the recursion
template <class H>
struct TupleElementsContainer<H, Null>
{
    typedef H head_t;
    typedef Null tail_t;

    H head;

    TupleElementsContainer() : head() { }

    template <class T0>
    TupleElementsContainer(T0& t0, const Null&, const Null&, const Null&)
        : head(t0)
    { }

    template <class HH>
    TupleElementsContainer& operator = (
        const TupleElementsContainer<HH,Null>& t)
    {
        head = t.head;
        return *this;
    }

    template<int N>
    typename TupleElementType<N, TupleElementsContainer>::type&
    get() { return getTupleElement<N>(*this); }
};

/*! \brief Rebind the TupleElementsContainer type.
 */
template <class T0, class T1, class T2, class T3>
struct TupleElementsBinder
{
    typedef TupleElementsContainer<
        T0, typename TupleElementsBinder<T1, T2, T3, Null>::type
    > type;
};

// break the recursion
template<>
struct TupleElementsBinder<Null, Null, Null, Null>
{ typedef Null type; };
//! \endcond

/*! \brief A simple N-element (1 to 4) tuple.
 */
template <class T0 = Null, class T1 = Null, class T2 = Null, class T3 = Null>
class tuple : public TupleElementsBinder<T0, T1, T2, T3>::type
{
private:
    typedef typename TupleElementsBinder<T0, T1, T2, T3>::type base_t;

public:
    tuple() { }
    tuple(T0 t0) : base_t(t0, null(), null(), null()) { }
    tuple(T0 t0, T1 t1) : base_t(t0, t1, null(), null()) { }
    tuple(T0 t0, T1 t1, T2 t2) : base_t(t0, t1, t2, null()) { }
    tuple(T0 t0, T1 t1, T2 t2, T3 t3) : base_t(t0, t1, t2, t3) { }

    template <class H, class T>
    tuple(const TupleElementsContainer<H,T>& te) : base_t(te)
    { }

    template <class H, class T>
    tuple& operator = (const TupleElementsContainer<H,T>& te)
    {
        base_t::operator = (te);
        return *this;
    }

    template <class F, class S>
    tuple& operator = (const pair<F,S>& p)
    {
        base_t::operator = (p);
        return *this;
    }
};

// tuple / pair element getters.

template <int N, class H, class T>
inline typename TupleElementType<N, TupleElementsContainer<H,T> >::type&
get(TupleElementsContainer<H,T>& te)
{
    return getTupleElement<N>(te);
}

template <int N, class H, class T>
inline const typename TupleElementType<N, TupleElementsContainer<H,T> >::type&
get(const TupleElementsContainer<H,T>& te)
{
    return getTupleElement<N>(te);
}

template <int N, class F, class S>
inline typename TupleElementType<N, tuple<F,S> >::type&
get(pair<F,S>& p)
{
    return PairElement<N>::get(p);
}

template <int N, class F, class S>
inline const typename TupleElementType<N, tuple<F,S> >::type&
get(const pair<F,S>& p)
{
    return PairElement<N>::get(p);
}

// Some tuple helpers (make_tuple() and tie())

template <class T0>
inline tuple<T0>
make_tuple(const T0& t0)
{
    return tuple<T0>(t0);
}

template <class T0, class T1>
inline tuple<T0, T1>
make_tuple(const T0& t0, const T1& t1)
{
    return tuple<T0, T1>(t0, t1);
}

template <class T0, class T1, class T2>
inline tuple<T0, T1, T2>
make_tuple(const T0& t0, const T1& t1, const T2& t2)
{
    return tuple<T0, T1, T2>(t0, t1, t2);
}

template <class T0, class T1, class T2, class T3>
inline tuple<T0, T1, T2, T3>
make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3)
{
    return tuple<T0, T1, T2, T3>(t0, t1, t2, t3);
}

template <class T0>
inline tuple<T0&>
tie(T0& t0)
{
    return tuple<T0&>(t0);
}

template <class T0, class T1>
inline tuple<T0&, T1&>
tie(T0& t0, T1& t1)
{
    return tuple<T0&, T1&>(t0, t1);
}

template <class T0, class T1, class T2>
inline tuple<T0&, T1&, T2&>
tie(T0& t0, T1& t1, T2& t2)
{
    return tuple<T0&, T1&, T2&>(t0, t1, t2);
}

template <class T0, class T1, class T2, class T3>
inline tuple<T0&, T1&, T2&, T3&>
tie(T0& t0, T1& t1, T2& t2, T3& t3)
{
    return tuple<T0&, T1&, T2&, T3&>(t0, t1, t2, t3);
}

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

/*@}*/} // namespace amd

#endif /*UTIL_HPP_*/
