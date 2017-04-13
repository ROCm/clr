//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "top.hpp"
#include "utils/debug.hpp"
#include "os/os.hpp"

#if !defined(LOG_LEVEL)
#include "utils/flags.hpp"
#endif

#include <cstdlib>
#include <cstdio>
#include <cstdarg>

#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

namespace amd {

//! \cond ignore
extern "C" void breakpoint(void) {
#ifdef _MSC_VER
  DebugBreak();
#endif  // _MSC_VER
}
//! \endcond

void report_fatal(const char* file, int line, const char* message) {
  // FIXME_lmoriche: Obfuscate the message string
  fprintf(stderr, "%s:%d: %s\n", file, line, message);
  ::abort();
}

void report_warning(const char* message) { fprintf(stderr, "Warning: %s\n", message); }

void log_entry(LogLevel level, const char* file, int line, const char* message) {
  if (level == LOG_NONE) {
    return;
  }
  fprintf(stderr, ":%d:%s:%d: %s\n", level, file, line, message);
}

void log_timestamped(LogLevel level, const char* file, int line, const char* message) {
  static bool gotstart = false;  // not thread-safe, but not scary if fails
  static uint64_t start;

  if (!gotstart) {
    start = Os::timeNanos();
    gotstart = true;
  }

  uint64_t time = Os::timeNanos() - start;
  if (level == LOG_NONE) {
    return;
  }
#if 0
    fprintf(stderr, ":%d:%s:%d: (%010lld) %s\n", level, file, line, time, message);
#else  // if you prefer fixed-width fields
  fprintf(stderr, ":% 2d:%15s:% 5d: (%010lld) %s\n", level, file, line, time / 100ULL,
          message);  // timestamp is 100ns units
#endif
}

void log_printf(LogLevel level, const char* file, int line, const char* format, ...) {
  va_list ap;

  va_start(ap, format);
  char message[1024];
  vsprintf(message, format, ap);
  va_end(ap);

  fprintf(stderr, ":%d:%s:%d: %s\n", level, file, line, message);
}

}  // namespace amd
