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

#include <hip/hiprtc.h>
#include "hiprtcInternal.hpp"

namespace hiprtc {
thread_local hiprtcResult g_lastRtcError = HIPRTC_SUCCESS;
}

const char* hiprtcGetErrorString(hiprtcResult x) {
  switch (x) {
    case HIPRTC_SUCCESS:
      return "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
      return "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
      return "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
      return "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
      return "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
      return "HIPRTC_ERROR_INTERNAL_ERROR";
    default:
      LogPrintfError("Invalid HIPRTC error code: %d \n", x);
      return nullptr;
  };

  return nullptr;
}


hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src, const char* name,
                                 int numHeaders, const char** headers, const char** headerNames) {
  HIPRTC_INIT_API(prog, src, name, numHeaders, headers, headerNames);

  if (prog == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_PROGRAM);
  }
  if (numHeaders < 0) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  if (numHeaders && (headers == nullptr || headerNames == nullptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  std::string progName;

  if (name) {
    progName = name;
  }

  auto* rtcProgram = new hiprtc::RTCProgram(progName);
  if (rtcProgram == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);
  }

  if (!rtcProgram->addSource(std::string(src), std::string("CompileSource"))) {
    delete rtcProgram;
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  for (int i = 0; i < numHeaders; i++) {
    if (!rtcProgram->addHeader(std::string(headers[i]), std::string(headerNames[i]))) {
      delete rtcProgram;
      HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
    }
  }

  *prog = hiprtc::RTCProgram::as_hiprtcProgram(rtcProgram);

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char** options) {
  HIPRTC_INIT_API(prog, numOptions, options);

  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(prog);

  std::vector<std::string> opt;
  opt.reserve(numOptions);
  for (int i = 0; i < numOptions; i++) {
    opt.push_back(std::string(options[i]));
  }

  if (!rtcProgram->compile(opt)) {
    HIPRTC_RETURN(HIPRTC_ERROR_COMPILATION);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char* name_expression) {
  HIPRTC_INIT_API(prog, name_expression);

  if (name_expression == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(prog);
  std::string name = name_expression;
  if (!rtcProgram->trackMangledName(name)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog, const char* name_expression,
                                  const char** loweredName) {
  HIPRTC_INIT_API(prog, name_expression, loweredName);

  if (name_expression == nullptr || loweredName == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(prog);

  if (!rtcProgram->getDemangledName(name_expression, loweredName)) {
    return HIPRTC_RETURN(HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) {
  HIPRTC_INIT_API(prog);
  if (prog == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(*prog);
  delete rtcProgram;
  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* binarySizeRet) {
  HIPRTC_INIT_API(prog, binarySizeRet);

  if (binarySizeRet == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(prog);

  *binarySizeRet = rtcProgram->getExecSize();

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* binaryMem) {
  HIPRTC_INIT_API(prog, binaryMem);

  if (binaryMem == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(prog);
  auto binary = rtcProgram->getExec();
  ::memcpy(binaryMem, binary.data(), binary.size());

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* dst) {
  HIPRTC_INIT_API(prog, dst);
  if (dst == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(prog);
  auto log = rtcProgram->getLog();
  ::memcpy(dst, log.data(), log.size());

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet) {
  HIPRTC_INIT_API(prog, logSizeRet);
  if (logSizeRet == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCProgram::as_RTCProgram(prog);

  *logSizeRet = rtcProgram->getLogSize();

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcVersion(int* major, int* minor) {
  HIPRTC_INIT_API(major, minor);

  if (major == nullptr || minor == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  // TODO add actual version, what do these numbers mean
  *major = 9;
  *minor = 0;

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}
