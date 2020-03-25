/* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#ifndef OBJECT_HPP_
#define OBJECT_HPP_

#include "top.hpp"
#include "os/alloc.hpp"
#include "thread/monitor.hpp"
#include "utils/util.hpp"


#define KHR_CL_TYPES_DO(F)                                                                         \
  /* OpenCL type          Runtime type */                                                          \
  F(cl_context, Context)                                                                           \
  F(cl_event, Event)                                                                               \
  F(cl_command_queue, CommandQueue)                                                                \
  F(cl_kernel, Kernel)                                                                             \
  F(cl_program, Program)                                                                           \
  F(cl_device_id, Device)                                                                          \
  F(cl_mem, Memory)                                                                                \
  F(cl_sampler, Sampler)

#define AMD_CL_TYPES_DO(F)                                                                         \
  F(cl_counter_amd, Counter)                                                                       \
  F(cl_perfcounter_amd, PerfCounter)                                                               \
  F(cl_threadtrace_amd, ThreadTrace)                                                               \
  F(cl_file_amd, LiquidFlashFile)


#define CL_TYPES_DO(F)                                                                             \
  KHR_CL_TYPES_DO(F)                                                                               \
  AMD_CL_TYPES_DO(F)

// Forward declare ::cl_* types and amd::Class types
//

#define DECLARE_CL_TYPES(CL, AMD)                                                                  \
  namespace amd {                                                                                  \
  class AMD;                                                                                       \
  }

CL_TYPES_DO(DECLARE_CL_TYPES);

#undef DECLARE_CL_TYPES

typedef struct _cl_icd_dispatch cl_icd_dispatch;

#define DECLARE_CL_TYPES(CL, AMD)                                                                  \
  typedef struct _##CL {                                                                           \
    cl_icd_dispatch* dispatch;                                                             \
  } * CL;

AMD_CL_TYPES_DO(DECLARE_CL_TYPES);

#undef DECLARE_CL_TYPES

namespace amd {

// Define the cl_*_type tokens for type checking.
//

#define DEFINE_CL_TOKENS(CL, ignored) T##CL,

enum cl_token { Tinvalid = 0, CL_TYPES_DO(DEFINE_CL_TOKENS) numTokens };

#undef DEFINE_CL_TOKENS

const size_t RuntimeObjectAlignment = NextPowerOfTwo<numTokens>::value;

//! \cond ignore
template <typename T> struct as_internal { typedef void type; };

template <typename T> struct as_external { typedef void type; };

template <typename T> struct class_token { static const cl_token value = Tinvalid; };

#define DEFINE_CL_TRAITS(CL, AMD)                                                                  \
                                                                                                   \
  template <> struct class_token<AMD> { static const cl_token value = T##CL; };                    \
                                                                                                   \
  template <> struct as_internal<_##CL> { typedef AMD type; };                                     \
  template <> struct as_internal<const _##CL> { typedef AMD const type; };                         \
                                                                                                   \
  template <> struct as_external<AMD> { typedef _##CL type; };                                     \
  template <> struct as_external<const AMD> { typedef _##CL const type; };

CL_TYPES_DO(DEFINE_CL_TRAITS);

#undef DEFINE_CL_TRAITS
//! \endcond

struct ICDDispatchedObject {
  static cl_icd_dispatch icdVendorDispatch_[];
  const cl_icd_dispatch* const dispatch_;

 protected:
  ICDDispatchedObject() : dispatch_(icdVendorDispatch_) {}

 public:
  static bool isValidHandle(const void* handle) { return handle != NULL; }

  const void* handle() const { return static_cast<const ICDDispatchedObject*>(this); }
  void* handle() { return static_cast<ICDDispatchedObject*>(this); }

  template <typename T> static const T* fromHandle(const void* handle) {
    return static_cast<const T*>(reinterpret_cast<const ICDDispatchedObject*>(handle));
  }
  template <typename T> static T* fromHandle(void* handle) {
    return static_cast<T*>(reinterpret_cast<ICDDispatchedObject*>(handle));
  }
};

/*! \brief For all OpenCL/Runtime objects.
 */
class RuntimeObject : public ReferenceCountedObject, public ICDDispatchedObject {
 public:
  enum ObjectType {
    ObjectTypeContext = 0,
    ObjectTypeDevice = 1,
    ObjectTypeMemory = 2,
    ObjectTypeKernel = 3,
    ObjectTypeCounter = 4,
    ObjectTypePerfCounter = 5,
    ObjectTypeEvent = 6,
    ObjectTypeProgram = 7,
    ObjectTypeQueue = 8,
    ObjectTypeSampler = 9,
    ObjectTypeThreadTrace = 10,
    ObjectTypeLiquidFlashFile = 11
  };

  virtual ObjectType objectType() const = 0;
};

template <typename T> class SharedReference : public EmbeddedObject {
 private:
  T& reference_;

 private:
  // do not copy shared references.
  SharedReference<T>& operator=(const SharedReference<T>& sref);

 public:
  explicit SharedReference(T& reference) : reference_(reference) { reference_.retain(); }

  ~SharedReference() { reference_.release(); }

  T& operator()() const { return reference_; }
};

/*! \brief A 1,2 or 3D coordinate.
 *!
 *! Note, dimensionality is only defined for sizes, and is given by the number
 *! of non-zero elements. (i.e. a 1D line is not the same as a 2D plane with width 1)
 */

struct Coord3D {
  size_t c[3];

  Coord3D(size_t d0, size_t d1 = 0, size_t d2 = 0) {
    c[0] = d0;
    c[1] = d1;
    c[2] = d2;
  }
  const size_t& operator[](size_t idx) const {
    assert(idx < 3);
    return c[idx];
  }
  bool operator==(const Coord3D& rhs) const {
    return c[0] == rhs.c[0] && c[1] == rhs.c[1] && c[2] == rhs.c[2];
  }
  explicit operator size_t*() {
    return &c[0];
  }
};

}  // namespace amd

template <typename CL> typename amd::as_internal<CL>::type* as_amd(CL* cl_obj) {
  return cl_obj == NULL ? NULL
                        : amd::RuntimeObject::fromHandle<typename amd::as_internal<CL>::type>(
                              static_cast<void*>(cl_obj));
}

template <typename AMD> typename amd::as_external<AMD>::type* as_cl(AMD* amd_obj) {
  return amd_obj == NULL ? NULL
                         : static_cast<typename amd::as_external<AMD>::type*>(amd_obj->handle());
}

template <typename CL> bool is_valid(CL* handle) {
  return amd::as_internal<CL>::type::isValidHandle(handle);
}

#endif /*OBJECT_HPP_*/
