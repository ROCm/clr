/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hiprtcInternal.hpp"

#include <fstream>
#include <streambuf>
#include <vector>

#include <sys/stat.h>

#include "vdi_common.hpp"
#include "utils/flags.hpp"

namespace hiprtc {
using namespace helpers;
std::unordered_set<RTCLinkProgram*>RTCLinkProgram::linker_set_;

std::vector<std::string> getLinkOptions(const LinkArguments& args) {
  std::vector<std::string> res;

  {  // process optimization level
    std::string opt("-O");
    opt += std::to_string(args.optimization_level_);
    res.push_back(opt);
  }

  const auto irArgCount = args.linker_ir2isa_args_count_;
  if (irArgCount > 0) {
    res.reserve(irArgCount);
    const auto irArg = args.linker_ir2isa_args_;
    for (size_t i = 0; i < irArgCount; i++) {
      res.emplace_back(std::string(irArg[i]));
    }
  }
  return res;
}

// RTC Program Member Functions
RTCProgram::RTCProgram(std::string name) : name_(name) {
  constexpr bool kComgrVersioned = true;
  std::call_once(amd::Comgr::initialized, amd::Comgr::LoadLib, kComgrVersioned);
  if (amd::Comgr::create_data_set(&exec_input_) != AMD_COMGR_STATUS_SUCCESS) {
    crashWithMessage("Failed to allocate internal hiprtc structure");
  }
}

bool RTCProgram::findIsa() {

#ifdef BUILD_SHARED_LIBS
  const char* libName;
#ifdef _WIN32
  std::string dll_name = std::string("amdhip64_" + std::to_string(HIP_VERSION_MAJOR) + ".dll");
  libName = dll_name.c_str();
#else
  libName = "libamdhip64.so";
#endif

  void* handle = amd::Os::loadLibrary(libName);

  if (!handle) {
    LogInfo("hip runtime failed to load using dlopen");
    build_log_ +=
        "hip runtime failed to load.\n"
        "Error: Please provide architecture for which code is to be "
        "generated.\n";
    return false;
  }

  void* sym_hipGetDevice = amd::Os::getSymbol(handle, "hipGetDevice");
  void* sym_hipGetDeviceProperties =
      amd::Os::getSymbol(handle, "hipGetDevicePropertiesR0600");  // Try to find the new symbol
  if (sym_hipGetDeviceProperties == nullptr) {
    sym_hipGetDeviceProperties =
        amd::Os::getSymbol(handle, "hipGetDeviceProperties");  // Fall back to old one
  }

  if (sym_hipGetDevice == nullptr || sym_hipGetDeviceProperties == nullptr) {
    LogInfo("ISA cannot be found to dlsym failure");
    build_log_ +=
        "ISA cannot be found from hip runtime.\n"
        "Error: Please provide architecture for which code is to be "
        "generated.\n";
    return false;
  }

  hipError_t (*dyn_hipGetDevice)(int*) = reinterpret_cast<hipError_t (*)(int*)>(sym_hipGetDevice);

  hipError_t (*dyn_hipGetDeviceProperties)(hipDeviceProp_t*, int) =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_t*, int)>(sym_hipGetDeviceProperties);

  int device;
  hipError_t status = dyn_hipGetDevice(&device);
  if (status != hipSuccess) {
    return false;
  }
  hipDeviceProp_t props;
  status = dyn_hipGetDeviceProperties(&props, device);
  if (status != hipSuccess) {
    return false;
  }
  isa_ = "amdgcn-amd-amdhsa--";
  isa_.append(props.gcnArchName);

  amd::Os::unloadLibrary(handle);
  return true;

#else
  int device;
  hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) {
    return false;
  }
  hipDeviceProp_t props;
  status = hipGetDeviceProperties(&props, device);
  if (status != hipSuccess) {
    return false;
  }
  isa_ = "amdgcn-amd-amdhsa--";
  isa_.append(props.gcnArchName);

  return true;
#endif
}

// RTC Compile Program Member Functions
void RTCProgram::AppendOptions(const std::string app_env_var, std::vector<std::string>* options) {
  if (options == nullptr) {
    LogError("Append options passed is nullptr.");
    return;
  }

  std::stringstream ss(app_env_var);
  std::istream_iterator<std::string> begin{ss}, end;
  options->insert(options->end(), begin, end);
}

// RTC Compile Program Member Functions
RTCCompileProgram::RTCCompileProgram(std::string name_) : RTCProgram(name_), fgpu_rdc_(false) {
  if ((amd::Comgr::create_data_set(&compile_input_) != AMD_COMGR_STATUS_SUCCESS) ||
      (amd::Comgr::create_data_set(&link_input_) != AMD_COMGR_STATUS_SUCCESS)) {
    crashWithMessage("Failed to allocate internal hiprtc structure");
  }
  // Add internal header
  if (!addBuiltinHeader()) {
    crashWithMessage("Unable to add internal header");
  }

  // Add compile options
  const std::string hipVerOpt{"--hip-version=" + std::to_string(HIP_VERSION_MAJOR) + '.' +
                              std::to_string(HIP_VERSION_MINOR) + '.' +
                              std::to_string(HIP_VERSION_PATCH)};
  const std::string hipVerMajor{"-DHIP_VERSION_MAJOR=" + std::to_string(HIP_VERSION_MAJOR)};
  const std::string hipVerMinor{"-DHIP_VERSION_MINOR=" + std::to_string(HIP_VERSION_MINOR)};
  const std::string hipVerPatch{"-DHIP_VERSION_PATCH=" + std::to_string(HIP_VERSION_PATCH)};

  compile_options_.reserve(20);  // count of options below
  compile_options_.push_back("-O3");

  if (!(GPU_ENABLE_WGP_MODE)) {
    compile_options_.push_back("-mcumode");
  }

  compile_options_.push_back(hipVerOpt);
  compile_options_.push_back(hipVerMajor);
  compile_options_.push_back(hipVerMinor);
  compile_options_.push_back(hipVerPatch);
  compile_options_.push_back("-D__HIPCC_RTC__");
  compile_options_.push_back("-include");
  compile_options_.push_back("hiprtc_runtime.h");
  compile_options_.push_back("-std=c++14");
  compile_options_.push_back("-nogpuinc");
  compile_options_.push_back("-Wno-gnu-line-marker");
  compile_options_.push_back("-Wno-missing-prototypes");
#ifdef _WIN32
  compile_options_.push_back("-target");
  compile_options_.push_back("x86_64-pc-windows-msvc");
  compile_options_.push_back("-fms-extensions");
  compile_options_.push_back("-fms-compatibility");
#endif
  AppendCompileOptions();
}

bool RTCCompileProgram::addSource(const std::string& source, const std::string& name) {
  if (source.size() == 0 || name.size() == 0) {
    LogError("Error in hiprtc: source or name is of size 0 in addSource");
    return false;
  }
  source_code_ += source;
  source_name_ = name;
  return true;
}

// addSource_impl is a different function because we need to add source when we track mangled
// objects
bool RTCCompileProgram::addSource_impl() {
  std::vector<char> vsource(source_code_.begin(), source_code_.end());
  if (!addCodeObjData(compile_input_, vsource, source_name_, AMD_COMGR_DATA_KIND_SOURCE)) {
    return false;
  }
  return true;
}

bool RTCCompileProgram::addHeader(const std::string& source, const std::string& name) {
  if (source.size() == 0 || name.size() == 0) {
    LogError("Error in hiprtc: source or name is of size 0 in addHeader");
    return false;
  }
  std::vector<char> vsource(source.begin(), source.end());
  if (!addCodeObjData(compile_input_, vsource, name, AMD_COMGR_DATA_KIND_INCLUDE)) {
    return false;
  }
  return true;
}

bool RTCCompileProgram::addBuiltinHeader() {
  std::vector<char> source(__hipRTC_header, __hipRTC_header + __hipRTC_header_size);
  std::string name{"hiprtc_runtime.h"};
  if (!addCodeObjData(compile_input_, source, name, AMD_COMGR_DATA_KIND_INCLUDE)) {
    return false;
  }
  return true;
}

bool RTCCompileProgram::findExeOptions(const std::vector<std::string>& options,
                                        std::vector<std::string>& exe_options) {
  for (size_t i = 0; i < options.size(); ++i) {
    // -mllvm options passed by the app such as "-mllvm" "-amdgpu-early-inline-all=true"
    if (options[i] == "-mllvm") {
      if (options.size() == (i + 1)) {
        LogInfo(
            "-mllvm option passed by the app, it comes as a pair but there is no option after "
            "this");
        return false;
      }
      exe_options.push_back(options[i]);
      exe_options.push_back(options[i + 1]);
    }
    // Options like -Rpass=inline
    if (options[i].find("-Rpass=") == 0) {
      exe_options.push_back(options[i]);
    }
  }
  return true;
}

bool RTCCompileProgram::transformOptions(std::vector<std::string>& compile_options) {
  auto getValueOf = [](const std::string& option) {
    std::string res;
    auto f = std::find(option.begin(), option.end(), '=');
    if (f != option.end()) res = std::string(f + 1, option.end());
    return res;
  };

  for (auto& i : compile_options) {
    if (i == "-hip-pch") {
      LogInfo(
          "-hip-pch is deprecated option, has no impact on execution of new hiprtc programs, it "
          "can be removed");
      i.clear();
      continue;
    }
    // Some rtc samples use --gpu-architecture
    if (i.rfind("--gpu-architecture=", 0) == 0) {
      LogInfo("--gpu-architecture is nvcc option, transforming it to --offload-arch option");
      auto val = getValueOf(i);
      i = "--offload-arch=" + val;
      continue;
    }
  }

  // Removed consumed options
  compile_options.erase(
      std::remove(compile_options.begin(), compile_options.end(), std::string("")),
      compile_options.end());

  if (auto res = std::find_if(
          compile_options.begin(), compile_options.end(),
          [](const std::string& str) { return str.find("--offload-arch=") != std::string::npos; });
      res != compile_options.end()) {
    auto isaName = getValueOf(*res);
    isa_ = "amdgcn-amd-amdhsa--" + isaName;
    settings_.offloadArchProvided = true;
    return true;
  }
  // App has not provided the gpu archiecture, need to find it
  return findIsa();
}

amd::Monitor RTCProgram::lock_("HIPRTC Program", true);

bool RTCCompileProgram::compile(const std::vector<std::string>& options, bool fgpu_rdc) {
  if (!addSource_impl()) {
    LogError("Error in hiprtc: unable to add source code");
    return false;
  }

  fgpu_rdc_ = fgpu_rdc;

  // Append compile options
  std::vector<std::string> compileOpts(compile_options_);
  compileOpts.reserve(compile_options_.size() + options.size() + 2);
  compileOpts.insert(compileOpts.end(), options.begin(), options.end());

  if (!transformOptions(compileOpts)) {
    LogError("Error in hiprtc: unable to transform options");
    return false;
  }

  if (fgpu_rdc_) {
    if (!compileToBitCode(compile_input_, isa_, compileOpts, build_log_, LLVMBitcode_)) {
      LogError("Error in hiprtc: unable to compile source to bitcode");
      return false;
    }
  } else {
    LogInfo("Using the new path of comgr");
    if (!compileToExecutable(compile_input_, isa_, compileOpts, link_options_, build_log_,
                             executable_)) {
      LogError("Failing to compile to realloc");
      return false;
    }
  }

  if (!mangled_names_.empty()) {
    auto& compile_step_output = fgpu_rdc_ ? LLVMBitcode_ : executable_;
    if (!fillMangledNames(compile_step_output, mangled_names_, fgpu_rdc_)) {
      LogError("Error in hiprtc: unable to fill mangled names");
      return false;
    }
  }

  return true;
}


void RTCCompileProgram::stripNamedExpression(std::string& strippedName) {
  if (strippedName.back() == ')') {
    strippedName.pop_back();
    strippedName.erase(0, strippedName.find('('));
  }
  if (strippedName.front() == '&') {
    strippedName.erase(0, 1);
  }

}

bool RTCCompileProgram::trackMangledName(std::string& name) {
  amd::ScopedLock lock(lock_);

  if (name.size() == 0) return false;

  std::string strippedName = name;
  stripNamedExpression(strippedName);

  mangled_names_.insert(std::pair<std::string, std::string>(strippedName, ""));

  std::string gcn_expr = "__amdgcn_name_expr_";
  std::string size = std::to_string(mangled_names_.size());
  const auto var1{"\n static __device__ const void* " + gcn_expr + size + "[]= {\"" + strippedName + "\", (void*)&" + strippedName + "};"};
  const auto var2{"\n static auto __amdgcn_name_expr_stub_" + size + " = " + gcn_expr + size + ";\n"};
  const auto code{var1 + var2};

  source_code_ += code;
  return true;
}

bool RTCCompileProgram::getMangledName(const char* name_expression, const char** loweredName) {
  std::string strippedName = name_expression;
  stripNamedExpression(strippedName);

  if (auto dres = mangled_names_.find(strippedName); dres != mangled_names_.end()) {
    if (dres->second.size() != 0) {
      *loweredName = dres->second.c_str();
      return true;
    } else
      return false;
  }
  return false;
}

bool RTCCompileProgram::GetBitcode(char* bitcode) {
  if (!fgpu_rdc_ || LLVMBitcode_.size() <= 0) {
    return false;
  }

  std::copy(LLVMBitcode_.begin(), LLVMBitcode_.end(), bitcode);
  return true;
}

bool RTCCompileProgram::GetBitcodeSize(size_t* bitcode_size) {
  if (!fgpu_rdc_ || LLVMBitcode_.size() <= 0) {
    return false;
  }

  *bitcode_size = LLVMBitcode_.size();
  return true;
}

// RTC Link Program Member Functions
RTCLinkProgram::RTCLinkProgram(std::string name) : RTCProgram(name) {
  if (amd::Comgr::create_data_set(&link_input_) != AMD_COMGR_STATUS_SUCCESS) {
    crashWithMessage("Failed to allocate internal hiprtc structure");
  }
  amd::ScopedLock lock(lock_);
  linker_set_.insert(this);
}

bool RTCLinkProgram::isLinkerValid(RTCLinkProgram* link_program) {
  amd::ScopedLock lock(lock_);
  if (linker_set_.find(link_program) == linker_set_.end()) {
    return false;
  }
  return true;
}

bool RTCLinkProgram::AddLinkerOptions(unsigned int num_options, hiprtcJIT_option* options_ptr,
                                      void** options_vals_ptr) {
  for (size_t opt_idx = 0; opt_idx < num_options; ++opt_idx) {
    switch (options_ptr[opt_idx]) {
      case HIPRTC_JIT_MAX_REGISTERS:
        link_args_.max_registers_ = *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_THREADS_PER_BLOCK:
        link_args_.threads_per_block_ =
            *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_WALL_TIME:
        link_args_.wall_time_ = *(reinterpret_cast<long*>(options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_INFO_LOG_BUFFER: {
        if (options_vals_ptr[opt_idx] == nullptr) {
          LogError("Options value can not be nullptr");
          return false;
        }
        link_args_.info_log_ = (reinterpret_cast<char*>(options_vals_ptr[opt_idx]));
        break;
      }
      case HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
        link_args_.info_log_size_ = (reinterpret_cast<size_t>(options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_ERROR_LOG_BUFFER: {
        if (options_vals_ptr[opt_idx] == nullptr) {
          LogError("Options value can not be nullptr");
          return false;
        }
        link_args_.error_log_ = reinterpret_cast<char*>(options_vals_ptr[opt_idx]);
        break;
      }
      case HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
        link_args_.error_log_size_ = (reinterpret_cast<size_t>(options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_OPTIMIZATION_LEVEL:
        link_args_.optimization_level_ =
            *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_TARGET_FROM_HIPCONTEXT:
        link_args_.target_from_hip_context_ =
            *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_TARGET:
        link_args_.jit_target_ = *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_FALLBACK_STRATEGY:
        link_args_.fallback_strategy_ =
            *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_GENERATE_DEBUG_INFO:
        link_args_.generate_debug_info_ = *(reinterpret_cast<int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_LOG_VERBOSE:
        link_args_.log_verbose_ = reinterpret_cast<size_t>(options_vals_ptr[opt_idx]);
        break;
      case HIPRTC_JIT_GENERATE_LINE_INFO:
        link_args_.generate_line_info_ = *(reinterpret_cast<int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_CACHE_MODE:
        link_args_.cache_mode_ = *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_NEW_SM3X_OPT:
        link_args_.sm3x_opt_ = *(reinterpret_cast<bool*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_FAST_COMPILE:
        link_args_.fast_compile_ = *(reinterpret_cast<bool*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_GLOBAL_SYMBOL_NAMES: {
        if (options_vals_ptr[opt_idx] == nullptr) {
          LogError("Options value can not be nullptr");
          return false;
        }
        link_args_.global_symbol_names_ = reinterpret_cast<const char**>(options_vals_ptr[opt_idx]);
        break;
      }
      case HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS: {
        if (options_vals_ptr[opt_idx] == nullptr) {
          LogError("Options value can not be nullptr");
          return false;
        }
        link_args_.global_symbol_addresses_ = reinterpret_cast<void**>(options_vals_ptr[opt_idx]);
        break;
      }
      case HIPRTC_JIT_GLOBAL_SYMBOL_COUNT:
        link_args_.global_symbol_count_ =
            *(reinterpret_cast<unsigned int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_LTO:
        link_args_.lto_ = *(reinterpret_cast<int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_FTZ:
        link_args_.ftz_ = *(reinterpret_cast<int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_PREC_DIV:
        link_args_.prec_div_ = *(reinterpret_cast<int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_PREC_SQRT:
        link_args_.prec_sqrt_ = *(reinterpret_cast<int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_FMA:
        link_args_.fma_ = *(reinterpret_cast<int*>(&options_vals_ptr[opt_idx]));
        break;
      case HIPRTC_JIT_IR_TO_ISA_OPT_EXT: {
        if (options_vals_ptr[opt_idx] == nullptr) {
          LogError("Options value can not be nullptr");
          return false;
        }
        link_args_.linker_ir2isa_args_ = reinterpret_cast<const char**>(options_vals_ptr[opt_idx]);
        break;
      }
      case HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT:
        link_args_.linker_ir2isa_args_count_ = reinterpret_cast<size_t>(options_vals_ptr[opt_idx]);
        break;
      default:
        break;
    }
  }

  return true;
}

amd_comgr_data_kind_t RTCLinkProgram::GetCOMGRDataKind(hiprtcJITInputType input_type) {
  amd_comgr_data_kind_t data_kind = AMD_COMGR_DATA_KIND_UNDEF;

  // Map the hiprtc input type to comgr data kind
  switch (input_type) {
    case HIPRTC_JIT_INPUT_LLVM_BITCODE:
      data_kind = AMD_COMGR_DATA_KIND_BC;
      break;
    case HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE:
      data_kind =
          HIPRTC_USE_RUNTIME_UNBUNDLER ? AMD_COMGR_DATA_KIND_BC : AMD_COMGR_DATA_KIND_BC_BUNDLE;
      break;
    case HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE:
      data_kind = AMD_COMGR_DATA_KIND_AR_BUNDLE;
      break;
    default:
      LogError("Cannot find the corresponding comgr data kind");
      break;
  }

  return data_kind;
}

bool RTCLinkProgram::AddLinkerDataImpl(std::vector<char>& link_data, hiprtcJITInputType input_type,
                                       std::string& link_file_name) {
  std::vector<char> llvm_bitcode;
  // If this is bundled bitcode then unbundle this.
  if (HIPRTC_USE_RUNTIME_UNBUNDLER && input_type == HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE) {
    if (!findIsa()) {
      return false;
    }

    size_t co_offset = 0;
    size_t co_size = 0;
    if (!UnbundleBitCode(link_data, isa_, co_offset, co_size)) {
      LogError("Error in hiprtc: unable to unbundle the llvm bitcode");
      return false;
    }

    llvm_bitcode.assign(link_data.begin() + co_offset, link_data.begin() + co_offset + co_size);
  } else {
    llvm_bitcode.assign(link_data.begin(), link_data.end());
  }

  amd_comgr_data_kind_t data_kind;
  if ((data_kind = GetCOMGRDataKind(input_type)) == AMD_COMGR_DATA_KIND_UNDEF) {
    LogError("Cannot find the correct COMGR data kind");
    return false;
  }

  if (!addCodeObjData(link_input_, llvm_bitcode, link_file_name, data_kind)) {
    LogError("Error in hiprtc: unable to add linked code object");
    return false;
  }

  return true;
}

bool RTCLinkProgram::AddLinkerFile(std::string file_path, hiprtcJITInputType input_type) {
  std::ifstream file_stream{file_path, std::ios_base::in | std::ios_base::binary};
  if (!file_stream.good()) {
    return false;
  }

  file_stream.seekg(0, std::ios::end);
  std::streampos file_size = file_stream.tellg();
  file_stream.seekg(0, std::ios::beg);

  // Read the file contents
  std::vector<char> link_file_info(file_size);
  file_stream.read(link_file_info.data(), file_size);

  file_stream.close();

  std::string link_file_name("LinkerProgram");

  return AddLinkerDataImpl(link_file_info, input_type, link_file_name);
}

bool RTCLinkProgram::AddLinkerData(void* image_ptr, size_t image_size, std::string link_file_name,
                                   hiprtcJITInputType input_type) {
  char* image_char_buf = reinterpret_cast<char*>(image_ptr);
  std::vector<char> bundled_llvm_bitcode(image_char_buf, image_char_buf + image_size);

  return AddLinkerDataImpl(bundled_llvm_bitcode, input_type, link_file_name);
}

bool RTCLinkProgram::LinkComplete(void** bin_out, size_t* size_out) {
  if (!findIsa()) {
    return false;
  }

  AppendLinkerOptions();

  std::vector<char> linked_llvm_bitcode;
  if (!linkLLVMBitcode(link_input_, isa_, link_options_, build_log_, linked_llvm_bitcode)) {
    LogError("Error in hiprtc: unable to add device libs to linked bitcode");
    return false;
  }

  std::string linkedFileName = "LLVMBitcode.bc";
  if (!addCodeObjData(exec_input_, linked_llvm_bitcode, linkedFileName, AMD_COMGR_DATA_KIND_BC)) {
    LogError("Error in hiprtc: unable to add linked bitcode");
    return false;
  }

  std::vector<std::string> exe_options = getLinkOptions(link_args_);
  LogPrintfInfo("Exe options forwarded to compiler: %s",
                [&]() {
                  std::string ret;
                  for (const auto& i : exe_options) {
                    ret += i;
                    ret += " ";
                  }
                  return ret;
                }()
                    .c_str());
  if (!createExecutable(exec_input_, isa_, exe_options, build_log_, executable_)) {
    LogPrintfInfo("Error in hiprtc: unable to create exectuable: %s", build_log_.c_str());
    return false;
  }

  *size_out = executable_.size();
  *bin_out = executable_.data();

  return true;
}

}  // namespace hiprtc
