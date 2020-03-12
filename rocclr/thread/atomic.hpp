/* Copyright (c) 2009-present Advanced Micro Devices, Inc.

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

/*! \file atomic.hpp
 *  \brief  Declarations for Memory order access and Atomic operations.
 *
 *  \author Laurent Morichetti
 *  \date   October 2008
 */

#ifndef ATOMIC_HPP_
#define ATOMIC_HPP_

#include "top.hpp"

#include <type_traits>

#ifdef _WIN32
#include <intrin.h>
#elif defined(ATI_ARCH_X86)
#include <emmintrin.h>
#include <xmmintrin.h>
#endif  // !_WIN32

#include <atomic>
#include <utility>

namespace amd {

/*! \addtogroup Threads
 *  @{
 *
 *  \addtogroup Atomic Atomic Operations
 *  @{
 */

/*! \brief Static functions for atomic operations.
 */
class AtomicOperation : AllStatic {
 private:
  //! Template to specialize atomic intrinsics on register size.
  template <int N> struct Intrinsics {
    /*! \brief %Atomic add.
     *
     *  Atomically add \a inc to \a *dest and return the prior value.
     */
    template <typename T> static inline T add(T increment, volatile T* dest);

    /*! \brief %Atomic exchange.
     *
     *  Atomically exchange value with *dest and return the prior value.
     */
    template <typename T> static inline T swap(T value, volatile T* dest);

    /*! \brief %Atomic compare and exchange.
     *
     *  Atomically compare and xchge value with *dest if *dest == compare.
     *  Return the prior value.
     */
    template <typename T> static inline T compareAndSwap(T compare, volatile T* dest, T value);

    /*! \brief %Atomic increment.
     *
     *  Atomically increment *dest and return the prior value.
     */
    template <typename T> static inline T increment(volatile T* dest);

    /*! \brief %Atomic exchange.
     *
     *  Atomically decrement *dest and return the prior value.
     */
    template <typename T> static inline T decrement(volatile T* dest);

    /*! \brief %Atomic or.
     *
     *  Atomically or \a mask to \a *dest and return the prior value.
     */
    template <typename T> static inline T _or(T mask, volatile T* dest);

    /*! \brief %Atomic and.
     *
     *  Atomically and \a mask to \a *dest and return the prior value.
     */
    template <typename T> static inline T _and(T mask, volatile T* dest);
  };

 public:
  /*! \brief %Atomic add.
   *
   *  Atomically add \a inc to \a *dest and return the prior value.
   */
  template <typename T> static T add(T inc, volatile T* dest) {
    return Intrinsics<sizeof(T)>::add((T)inc, dest);
  }

  /*! \brief %Atomic exchange.
   *
   *  Atomically exchange value with *dest and return the prior value.
   */
  template <typename T> static T swap(T value, volatile T* dest) {
    return Intrinsics<sizeof(T)>::swap(value, dest);
  }

  /*! \brief %Atomic compare and exchange.
   *
   *  Atomically compare and exchange value with *dest if *dest == compare.
   *  Return the prior value.
   */
  template <typename T> static T compareAndSwap(T compare, volatile T* dest, T value) {
    return Intrinsics<sizeof(T)>::compareAndSwap(compare, dest, value);
  }

  /*! \brief %Atomic increment.
   *
   *  Atomically increment *dest and return the prior value.
   */
  template <typename T> static T increment(volatile T* dest) {
    return Intrinsics<sizeof(T)>::increment(dest);
  }

  /*! \brief %Atomic decrement.
   *
   *  Atomically decrement *dest and return the prior value.
   */
  template <typename T> static T decrement(volatile T* dest) {
    return Intrinsics<sizeof(T)>::decrement(dest);
  }

  /*! \brief %Atomic or.
   *
   *  Atomically or \a mask to \a *dest and return the prior value.
   */
  template <typename T> static T _or(T mask, volatile T* dest) {
    return Intrinsics<sizeof(T)>::_or((T)mask, dest);
  }

  /*! \brief %Atomic and.
   *
   *  Atomically or \a mask to \a *dest and return the prior value.
   */
  template <typename T> static T _and(T mask, volatile T* dest) {
    return Intrinsics<sizeof(T)>::_and((T)mask, dest);
  }
};

/*@}*/

#if defined(_MSC_VER)

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<4>::add(T increment, volatile T* dest) {
  return (T)_InterlockedExchangeAdd((volatile long*)dest, (long)increment);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<4>::swap(T value, volatile T* dest) {
  return (T)_InterlockedExchange((volatile long*)dest, (long)value);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<4>::compareAndSwap(T compare, volatile T* dest, T value) {
  return (T)_InterlockedCompareExchange((volatile long*)dest, (long)value, (long)compare);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<4>::increment(volatile T* dest) {
  return (T)(_InterlockedIncrement((volatile long*)dest) - 1L);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<4>::decrement(volatile T* dest) {
  return (T)(_InterlockedDecrement((volatile long*)dest) + 1L);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<4>::_or(T mask, volatile T* dest) {
  return (T)_InterlockedOr((volatile long*)dest, (long)mask);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<4>::_and(T mask, volatile T* dest) {
  return (T)_InterlockedAnd((volatile long*)dest, (long)mask);
}

#ifdef _WIN64

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<8>::add(T increment, volatile T* dest) {
  return (T)_InterlockedExchangeAdd64((volatile __int64*)dest, (__int64)increment);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<8>::swap(T value, volatile T* dest) {
  return (T)_InterlockedExchange64((volatile __int64*)dest, (__int64)value);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<8>::compareAndSwap(T compare, volatile T* dest, T value) {
  return (T)_InterlockedCompareExchange64((volatile __int64*)dest, (__int64)value,
                                          (__int64)compare);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<8>::increment(volatile T* dest) {
  return (T)(_InterlockedIncrement64((volatile __int64*)dest) - 1LL);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<8>::decrement(volatile T* dest) {
  return (T)(_InterlockedDecrement64((volatile __int64*)dest) + 1LL);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<8>::_or(T mask, volatile T* dest) {
  return (T)_InterlockedOr64((volatile long*)dest, (long)mask);
}

template <>
template <typename T>
inline T AtomicOperation::Intrinsics<8>::_and(T mask, volatile T* dest) {
  return (T)_InterlockedAnd64((volatile long*)dest, (long)mask);
}

#endif  // _LP64

#elif defined(__GNUC__)

template <int N>
template <typename T>
inline T AtomicOperation::Intrinsics<N>::add(T inc, volatile T* dest) {
  return __sync_fetch_and_add(dest, inc);
}

template <int N>
template <typename T>
inline T AtomicOperation::Intrinsics<N>::swap(T value, volatile T* dest) {
  return __sync_lock_test_and_set(dest, value);
}

template <int N>
template <typename T>
inline T AtomicOperation::Intrinsics<N>::compareAndSwap(T compare, volatile T* dest, T value) {
  return __sync_val_compare_and_swap(dest, compare, value);
}

template <int N>
template <typename T>
inline T AtomicOperation::Intrinsics<N>::increment(volatile T* dest) {
  return add(T(1), dest);
}

template <int N>
template <typename T>
inline T AtomicOperation::Intrinsics<N>::decrement(volatile T* dest) {
  return add(T(-1), dest);
}

template <int N>
template <typename T>
inline T AtomicOperation::Intrinsics<N>::_or(T mask, volatile T* dest) {
  return __sync_fetch_and_or(dest, mask);
}

template <int N>
template <typename T>
inline T AtomicOperation::Intrinsics<N>::_and(T mask, volatile T* dest) {
  return __sync_fetch_and_and(dest, mask);
}

#else
#error Unimplemented
#endif

/*! \addtogroup Atomic Atomic Operations
 *  @{
 */

/*! \brief A variable of type T with atomic properties.
 */
template <typename T> class Atomic {
 private:
  typedef typename std::remove_volatile<
      typename std::remove_pointer<typename std::remove_reference<T>::type>::type>::type value_type;
  typename std::add_volatile<T>::type value_;  //!< \brief The variable.

 public:
  //! Construct a new %Atomic variable of type T.
  Atomic() : value_(T(0)) {}
  //! Construct a new %Atomic variable of type T from \a value.
  Atomic(T value) : value_(value) {}
  //! Construct a new %Atomic variable of type T from another %Atomic.
  Atomic(const Atomic<T>& atomic) : value_(atomic.value_) {}
  //! Copy value into this %Atomic variable.
  Atomic<T>& operator=(T value) {
    value_ = value;
    return *this;
  }

  //! Return the %Atomic variable value.
  operator T() const { return T(value_); }
  //! Return the %Atomic variable value.
  T operator->() const { return T(value_); }
  //! Return the %Atomic variable's address.
  typename std::add_pointer<typename std::add_volatile<value_type>::type>::type operator&() {
    return &value_;
  }

  //! Atomically add \a inc to this variable.
  Atomic<T>& operator+=(value_type inc) {
    if (std::is_pointer<T>::value) {
      inc *= sizeof(typename std::remove_pointer<T>::type);
    }
    AtomicOperation::add(inc, &value_);
    return *this;
  }

  //! Atomically subtract \a inc to this variable.
  Atomic<T>& operator-=(value_type inc) {
    value_type modifier = 0;
    if (std::is_pointer<T>::value) {
      inc *= sizeof(typename std::remove_pointer<T>::type);
    }
    AtomicOperation::add(modifier - inc, &value_);
    return *this;
  }

  //! Atomically OR \a value to this variable.
  Atomic<T>& operator|=(value_type mask) {
    AtomicOperation::_or(mask, &value_);
    return *this;
  }

  //! Atomically AND \a value to this variable.
  Atomic<T>& operator&=(value_type mask) {
    AtomicOperation::_and(mask, &value_);
    return *this;
  }

  //! Atomically increment this variable and return its new value.
  typename std::remove_reference<T>::type operator++() {
    if (std::is_pointer<T>::value) {
      value_type inc = static_cast<value_type>(sizeof(typename std::remove_pointer<T>::type));
      return AtomicOperation::add(inc, &value_) + 1;
    } else {
      return AtomicOperation::increment(&value_) + 1;
    }
  }

  //! Atomically decrement this variable and return its new value.
  typename std::remove_reference<T>::type operator--() {
    if (std::is_pointer<T>::value) {
      value_type inc =
          static_cast<value_type>(-static_cast<typename std::make_signed<value_type>::type>(
              sizeof(typename std::remove_pointer<T>::type)));
      return AtomicOperation::add(inc, &value_) - 1;
    } else {
      return AtomicOperation::decrement(&value_) - 1;
    }
  }

  //! Atomically increment this variable and return its previous value.
  typename std::remove_reference<T>::type operator++(int) {
    if (std::is_pointer<T>::value) {
      value_type inc = static_cast<value_type>(sizeof(typename std::remove_pointer<T>::type));
      return AtomicOperation::add(inc, &value_);
    } else {
      return AtomicOperation::increment(&value_);
    }
  }

  //! Atomically decrement this variable and return its previous value.
  T operator--(int) {
    if (std::is_pointer<T>::value) {
      value_type inc =
          static_cast<value_type>(-static_cast<typename std::make_signed<value_type>::type>(
              sizeof(typename std::remove_pointer<T>::type)));
      return AtomicOperation::add(inc, &value_);
    } else {
      return AtomicOperation::decrement(&value_);
    }
  }

  /*! \brief Atomically compare this variable with \a compare and set
   *  to value if equals
   */
  bool compareAndSet(T compare, T value) {
    return compare == AtomicOperation::compareAndSwap(compare, &value_, value);
  }

  //! Atomically set this variable to \a value and return its previous value.
  T swap(T value) { return AtomicOperation::swap(value, &value_); }

  /*! \brief Execute a stores fence followed by a store to this variable.
   *
   *  This storeRelease operation ensures that all store to memory operations
   *  preceding this function will be globally visible before the update to
   *  this variable's value.
   */
  void storeRelease(T value) {
    std::atomic_thread_fence(std::memory_order_release);
    value_ = value;
  }

  /*! \brief Execute a load from this variable followed by a loads fence.
   *
   *  This loadAcquire operation ensures that all load from memory operations
   *  following this function will be globally visible after the read from
   *  this variable's value.
   */
  T loadAcquire() const {
    T value = value_;
    std::atomic_thread_fence(std::memory_order_acquire);
    return value;
  }
};

//! Helper function to tie an Atomic<T&> to a variable of type T.
template <typename T> inline Atomic<T&> make_atomic(T& t) { return Atomic<T&>(t); }


/*! @}
 *  @}
 */

}  // namespace amd

#endif /*ATOMIC_HPP_*/
