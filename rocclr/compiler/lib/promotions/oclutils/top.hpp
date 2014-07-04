//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef TOP_HPP_
#define TOP_HPP_

#if defined(ATI_ARCH_ARM)
# define __EXPORTED_HEADERS__ 1
#endif /*ATI_ARCH_ARM*/

#ifdef _WIN32
#define NOMINMAX 1
#define WIN32_LEAN_AND_MEAN 1
#endif /*_WIN32*/

#include "utils/macros.hpp"
#if 0 // FIXME_lmoriche
#include "CL/opencl.h"
#include "amdocl/cl_open_video_amd.h"
#endif

#ifdef _WIN32
# include <cstdlib>
#else /*!_WIN32*/
# include <inttypes.h>
#endif /*!_WIN32*/

#if !defined(ATI_ARCH_ARM)
#include <xmmintrin.h>
#endif /*!ATI_ARCH_ARM*/
#include <cstddef>
#include <new>

typedef unsigned char* address;
typedef const unsigned char* const_address;
typedef void * pointer;
typedef const void * const_pointer;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef const char* cstring;


#ifdef _WIN32
#if _MSC_VER >= 1600
# include <stdint.h>
#else // _MSC_VER < 1600
typedef signed   __int8  int8_t;
typedef unsigned __int8  uint8_t;
typedef signed   __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef signed   __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef signed   __int64 int64_t;
typedef unsigned __int64 uint64_t;
#endif // _MSC_VER < 1600
#ifndef _WIN64
typedef int32_t ssize_t;
#else // _WIN64
typedef int64_t ssize_t;
#endif // _WIN64
#endif /*_WIN32*/

#ifdef _WIN32
# define SIZE_T_FMT "%Iu"
# define PTR_FMT    "0x%p"
# define snprintf sprintf_s
#else /*!_WIN32*/
# define SIZE_T_FMT "%zu"
# define PTR_FMT    "%p"
#endif /*!_WIN32*/

typedef uint32_t cl_mem_fence_flags;

//! \cond ignore
#define _BAD_INT32  0xBAADBAAD
#define _BAD_INT64  0XBAADBAADBAADBAADLL
#define _BAD_INTPTR LP64_SWITCH(_BAD_INT32,_BAD_INT64)

const pointer badPointer = (pointer)(intptr_t) _BAD_INTPTR;
const address badAddress = (address)(intptr_t) _BAD_INTPTR;
//! \endcond

const size_t Ki = 1024;
const size_t Mi = Ki*Ki;
const size_t Gi = Ki*Ki*Ki;

const size_t K = 1000;
const size_t M = K*K;
const size_t G = K*K*K;

#include "utils/debug.hpp"

//! \addtogroup Utils

//! Namespace for AMD's OpenCL platform
namespace amd {/*@{*/

//! \brief The default Null object type (!= void*);
struct Null {};

//! \brief Return a const Null object (null)
inline const Null null() { return Null(); }

/*! \brief A struct to hold 2 objects of arbitrary type.
 */
template <typename F, typename S>
struct pair
{
    F first;  /*!< \brief first element. */
    S second; /*!< \brief second element. */

    pair() : first(), second() { }
    pair(const F& f, const S& s) : first(f), second(s) { }

    template <typename FF, typename SS>
    pair(const pair<FF,SS>& p) : first(p.first), second(p.second) { }
};

template<typename F, typename S>
inline pair<F,S>
make_pair(F first, S second)
{
    return pair<F,S>(first, second);
}

/*! \brief Equivalent to a namespace (All member functions are static).
 */
class AllStatic
{
WINDOWS_SWITCH(public,private):
    AllStatic() { ShouldNotCallThis(); }
    AllStatic(const AllStatic&) { ShouldNotCallThis(); }
    ~AllStatic() { ShouldNotCallThis(); }
};

/*! \brief For embedded objects.
 */
class EmbeddedObject
{
WINDOWS_SWITCH(public,private):
    void* operator new(size_t) { ShouldNotCallThis(); return badPointer;  }
    void operator delete(void *) { ShouldNotCallThis(); }
};

/*! \brief For stack allocated objects.
 */
class StackObject
{
WINDOWS_SWITCH(public,private):
    void* operator new(size_t) { ShouldNotCallThis(); return badPointer;  }
    void operator delete(void *) { ShouldNotCallThis(); }
};

/*! \brief for objects allocated in a dedicate memory pool.
  the standard 'new' should not be called, 
  only the in place version 'new (allocation_pointer) <class>()'
  , delete should only invoke the destructors and not release memory
 */
class MemoryPoolObject
{
public:
	void* operator new(size_t) { ShouldNotCallThis(); return badPointer;  }
	void* operator new(size_t size,void * address) { return address; }
	void operator delete(void *) {  }
	void operator delete( void *,void * address) { }
};

/*! \brief For objects allocated on the C-heap.
 */
class HeapObject
{
public:
    void* operator new(size_t size);
    void operator delete(void* obj);
};

/*! \brief For all reference counted objects.
 */
class ReferenceCountedObject
{
    volatile uint referenceCount_;

protected:
    virtual ~ReferenceCountedObject() { }
    virtual bool terminate() { return true; }

public:
    ReferenceCountedObject() : referenceCount_(1) { }

    void* operator new(size_t size) { return ::operator new(size); }
    void operator delete(void* p) { return ::operator delete(p); }

    uint referenceCount() const { return referenceCount_; }

    uint retain();
    uint release();
};

/*@}*/} // namespace amd

#ifdef FOR_DOXYGEN_ONLY
namespace std
{
template<class F, class S> struct pair { F first; S second; };
template<class T> struct vector { public: T data; };
template<class T> class deque { public: T data; };
template<class T> class list { public: T data; };
template<class T> class slist { public: T data; };
template<class T> class set { public: T data; };
template<class Key, class Data> class map { public: Key key; Data data; };
}
#endif // FOR_DOXYGEN_ONLY

#undef min // using std::min
#undef max // using std::max

#endif /*TOP_HPP_*/

