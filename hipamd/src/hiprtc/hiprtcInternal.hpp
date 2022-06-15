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
protected:
  // Lock and control variables
  static amd::Monitor lock_;
  static std::once_flag initialized_;

  RTCProgram(std::string name);
  ~RTCProgram() {
    amd::Comgr::destroy_data_set(exec_input_);
  }

  // Member Functions
  bool findIsa();
  
  // Data Members  
  std::string name_;  
  std::string isa_;
  std::string build_log_;
  std::vector<char> executable_;

  amd_comgr_data_set_t exec_input_;
  std::vector<std::string> exe_options_;
};

class RTCCompileProgram : public RTCProgram {

  // Private Data Members
  Settings settings_;

  std::string source_code_;
  std::string source_name_;
  std::map<std::string, std::string> stripped_names_;
  std::map<std::string, std::string> demangled_names_;
  
  std::vector<std::string> compile_options_;
  std::vector<std::string> link_options_;

  amd_comgr_data_set_t compile_input_;
  amd_comgr_data_set_t link_input_;

  bool fgpu_rdc_;
  std::vector<char> LLVMBitcode_;

  // Private Member functions
  bool addSource_impl();
  bool addBuiltinHeader();
  bool transformOptions();

  RTCCompileProgram() = delete;
  RTCCompileProgram(RTCCompileProgram&) = delete;
  RTCCompileProgram& operator=(RTCCompileProgram&) = delete;

 public:
  RTCCompileProgram(std::string);
  ~RTCCompileProgram() {
    amd::Comgr::destroy_data_set(compile_input_);
    amd::Comgr::destroy_data_set(link_input_);
  }

  // Converters
  inline static hiprtcProgram as_hiprtcProgram(RTCCompileProgram* p) {
    return reinterpret_cast<hiprtcProgram>(p);
  }
  inline static RTCCompileProgram* as_RTCCompileProgram(hiprtcProgram& p) {
    return reinterpret_cast<RTCCompileProgram*>(p);
  }

  // Public Member Functions
  bool addSource(const std::string& source, const std::string& name);
  bool addHeader(const std::string& source, const std::string& name);
  bool compile(const std::vector<std::string>& options, bool fgpu_rdc);
  bool getMangledName(const char* name_expression, const char** loweredName);
  bool trackMangledName(std::string& name);
  void stripNamedExpression(std::string& named_expression);

  bool GetBitcode(char* bitcode);
  bool GetBitcodeSize(size_t* bitcode_size);
  // Public Getter/Setters
  const std::vector<char>& getExec() const { return executable_; }
  size_t getExecSize() const { return executable_.size(); }
  const std::string& getLog() const { return build_log_; }
  size_t getLogSize() const { return build_log_.size(); }
};

// Linker Arguments passed via hipLinkCreate
struct LinkArguments {
  unsigned int max_registers_;
  unsigned int threads_per_block_;
  float wall_time_;
  size_t info_log_size_;
  char* info_log_;
  size_t error_log_size_;
  char* error_log_;
  unsigned int optimization_level_;
  unsigned int target_from_hip_context_;
  unsigned int jit_target_;
  unsigned int fallback_strategy_;
  int generate_debug_info_;
  long log_verbose_;
  int generate_line_info_;
  unsigned int cache_mode_;
  bool sm3x_opt_;
  bool fast_compile_;
  const char** global_symbol_names_;
  void** global_symbol_addresses_;
  unsigned int global_symbol_count_;
  int lto_;
  int ftz_;
  int prec_div_;
  int prec_sqrt_;
  int fma_;
};

class RTCLinkProgram : public RTCProgram {

  // Private Member Functions (forbid these function calls)
  RTCLinkProgram() = delete;
  RTCLinkProgram(RTCLinkProgram&) = delete;
  RTCLinkProgram& operator=(RTCLinkProgram&) = delete;

  amd_comgr_data_kind_t GetCOMGRDataKind(hiprtcJITInputType input_type);

  // Linker Argumenets at hipLinkCreate
  LinkArguments link_args_;

  // Private Data Members
  amd_comgr_data_set_t link_input_;
  std::vector<std::string> link_options_;
public:
  RTCLinkProgram(std::string name);
  ~RTCLinkProgram() {
    amd::Comgr::destroy_data_set(link_input_);
  }
  // Public Member Functions
  bool AddLinkerOptions(unsigned int num_options, hiprtcJIT_option* options_ptr,
                        void** options_vals_ptr);
  bool AddLinkerFile(std::string file_path, hiprtcJITInputType input_type);
  bool AddLinkerData(void* image_ptr, size_t image_size, std::string link_file_name,
                    hiprtcJITInputType input_type);
  bool LinkComplete(void** bin_out, size_t* size_out);
};

}  // namespace hiprtc
