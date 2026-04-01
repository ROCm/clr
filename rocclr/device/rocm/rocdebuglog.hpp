/* Copyright (c) 2025 Advanced Micro Devices, Inc.

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

#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/syscall.h>

namespace hip_debug {

inline int& logEnabled() {
  static int e = -1;
  return e;
}

inline FILE*& logFile() {
  static FILE* f = nullptr;
  return f;
}

inline pthread_once_t& onceCtrl() {
  static pthread_once_t o = PTHREAD_ONCE_INIT;
  return o;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
inline void initLog() {
  const char* env = getenv("HIP_DEBUG_LOG");
  if (!env || env[0] == '\0' || (env[0] == '0' && env[1] == '\0')) {
    logEnabled() = 0;
    return;
  }

  char path[512];
  if (strcmp(env, "1") == 0) {
    snprintf(path, sizeof(path), "/tmp/hip_debug_%d.log", getpid());
  } else {
    snprintf(path, sizeof(path), "%.*s", (int)(sizeof(path) - 1), env);
    char* pct = strstr(path, "%d");
    if (pct) {
      char tmp[512];
      *pct = '\0';
      snprintf(tmp, sizeof(tmp), "%s%d%s", path, getpid(), pct + 2);
      snprintf(path, sizeof(path), "%s", tmp);
    }
  }

  logFile() = fopen(path, "a");
  if (logFile()) {
    logEnabled() = 1;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    fprintf(logFile(), "[%ld.%06ld] === HIP Debug Log opened (pid=%d) ===\n",
            (long)ts.tv_sec, ts.tv_nsec / 1000, getpid());
    fflush(logFile());
  } else {
    logEnabled() = 0;
  }
}
#pragma GCC diagnostic pop

inline void dlog(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
inline void dlog(const char* fmt, ...) {
  pthread_once(&onceCtrl(), initLog);
  if (logEnabled() <= 0) return;
  FILE* f = logFile();
  if (!f) return;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  fprintf(f, "[%ld.%06ld] ", (long)ts.tv_sec, ts.tv_nsec / 1000);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  va_end(ap);
  fflush(f);
}

}  // namespace hip_debug

#define HIP_DLOG(fmt, ...) hip_debug::dlog(fmt, ##__VA_ARGS__)
