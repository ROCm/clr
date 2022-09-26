/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#ifndef HIPRTC_SRC_HIP_INTERNAL_H
#define HIPRTC_SRC_HIP_INTERNAL_H

#include "hip_internal.hpp"

#if __linux__
#include <cstdlib>

#if HIPRTC_USE_CXXABI
#include <cxxabi.h>

#define DEMANGLE abi::__cxa_demangle

#else
extern "C" char * __cxa_demangle(const char *mangled_name, char *output_buffer,
                                size_t *length, int *status);

#define DEMANGLE __cxa_demangle
#endif //HIPRTC_USE_CXXABI

#elif defined(_WIN32)
#include <Windows.h>
#include <DbgHelp.h>

#define UNDECORATED_SIZE 4096

#endif // __linux__

// This macro should be called at the beginning of every HIP RTC API.
#define HIPRTC_INIT_API(...)                                 \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s ( %s )", __func__, ToString( __VA_ARGS__ ).c_str()); \
  amd::Thread* thread = amd::Thread::current();              \
  if (!VDI_CHECK_THREAD(thread)) {                           \
    HIPRTC_RETURN(HIPRTC_ERROR_INTERNAL_ERROR);              \
  }                                                          \
  HIP_INIT_VOID();

#define HIPRTC_RETURN(ret)             \
  hiprtc::tls.last_rtc_error_ = ret;        \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s", __func__, \
          hiprtcGetErrorString(hiprtc::tls.last_rtc_error_));                 \
  return hiprtc::tls.last_rtc_error_;


#endif // HIPRTC_SRC_HIP_INTERNAL_H
