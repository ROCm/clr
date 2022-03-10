/*
Copyright (c) 2022 - Present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_version.h>


#ifdef HIPRTC_USE_EXCEPTIONS
#include <exception>
#endif
#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "top.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "utils/macros.hpp"

#ifdef __HIP_ENABLE_RTC
extern "C" {
extern const char __hipRTC_header[];
extern unsigned __hipRTC_header_size;
}
#endif

#include "hiprtcComgrHelper.hpp"


namespace hiprtc {
namespace internal {
template <typename T> inline std::string ToString(T v) {
  std::ostringstream ss;
  ss << v;
  return ss.str();
}

inline std::string ToString() { return (""); }

template <typename T, typename... Args> inline std::string ToString(T first, Args... args) {
  return ToString(first) + ", " + ToString(args...);
}
}  // namespace internal
}  // namespace hiprtc

#define HIPRTC_INIT_API(...)                                                                       \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s ( %s )", __func__,                                      \
          hiprtc::internal::ToString(__VA_ARGS__).c_str());

#define HIPRTC_RETURN(ret)                                                                         \
  hiprtc::g_lastRtcError = (ret);                                                                  \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s", __func__,                                \
          hiprtcGetErrorString(hiprtc::g_lastRtcError));                                           \
  return hiprtc::g_lastRtcError;


namespace hiprtc {

static void crashWithMessage(std::string message) {
#ifdef HIPRTC_USE_EXCEPTIONS
  throw std::runtime_error(message);
#else
  guarantee(false, message.c_str());
#endif
}

struct Settings {
  bool dumpISA{false};
  bool offloadArchProvided{false};
};

class RTCProgram {
  static amd::Monitor lock_;
  static std::once_flag initialized;

  std::string name;
  Settings settings;

  std::string isa;
  std::string buildLog;

  std::vector<char> executable;

  std::map<std::string, std::string> strippedNames;
  std::map<std::string, std::string> demangledNames;
  std::string sourceCode;
  std::string sourceName;

  std::vector<std::string> compileOptions;
  std::vector<std::string> linkOptions;
  std::vector<std::string> exeOptions;

  amd_comgr_data_set_t compileInput;
  amd_comgr_data_set_t linkInput;
  amd_comgr_data_set_t execInput;

  bool dumpIsa();
  bool findIsa();

  bool addSource_impl();
  bool addBuiltinHeader();
  bool transformOptions();

  RTCProgram() = delete;
  RTCProgram(RTCProgram&) = delete;
  RTCProgram& operator=(RTCProgram&) = delete;

 public:
  RTCProgram(std::string);

  // Converters
  inline static hiprtcProgram as_hiprtcProgram(RTCProgram* p) {
    return reinterpret_cast<hiprtcProgram>(p);
  }
  inline static RTCProgram* as_RTCProgram(hiprtcProgram& p) {
    return reinterpret_cast<RTCProgram*>(p);
  }

  bool addSource(const std::string& source, const std::string& name);
  bool addHeader(const std::string& source, const std::string& name);
  bool compile(const std::vector<std::string>& options);
  bool getDemangledName(const char* name_expression, const char** loweredName);
  bool trackMangledName(std::string& name);

  const std::vector<char>& getExec() const { return executable; }
  size_t getExecSize() const { return executable.size(); }
  const std::string& getLog() const { return buildLog; }
  size_t getLogSize() const { return buildLog.size(); }

  ~RTCProgram() {
    amd::Comgr::destroy_data_set(compileInput);
    amd::Comgr::destroy_data_set(linkInput);
    amd::Comgr::destroy_data_set(execInput);
  }
};
}  // namespace hiprtc
