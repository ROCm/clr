//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef DEBUG_HPP_
#define DEBUG_HPP_


#include <cassert>

//! \addtogroup Utils

namespace amd { /*@{*/

enum LogLevel { LOG_NONE = 0, LOG_ERROR = 1, LOG_WARNING = 2, LOG_INFO = 3, LOG_DEBUG = 4 };

//! \cond ignore
extern "C" void breakpoint();
//! \endcond

//! \brief Report a Fatal exception message and abort.
extern void report_fatal(const char* file, int line, const char* message);

//! \brief Display a warning message.
extern void report_warning(const char* message);

//! \brief Insert a log entry.
extern void log_entry(LogLevel level, const char* file, int line, const char* messsage);

//! \brief Insert a timestamped log entry.
extern void log_timestamped(LogLevel level, const char* file, int line, const char* messsage);

//! \brief Insert a printf-style log entry.
extern void log_printf(LogLevel level, const char* file, int line, const char* format, ...);

/*@}*/} // namespace amd

#if __INTEL_COMPILER

// Disable ICC's warning #279: controlling expression is constant
// (0!=1 && "msg")
//          ^
#pragma warning(disable : 279)

#endif  // __INTEL_COMPILER

//! \brief Abort the program if the invariant \a cond is false.
#define guarantee(cond)                                                                            \
  if (!(cond)) {                                                                                   \
    amd::report_fatal(__FILE__, __LINE__, "guarantee(" XSTR(cond) ")");                            \
    amd::breakpoint();                                                                             \
  }

#define fixme_guarantee(cond) guarantee(cond)

//! \brief Abort the program with a fatal error message.
#define fatal(msg)                                                                                 \
  do {                                                                                             \
    assert(false && msg);                                                                          \
  } while (0)


//! \brief Display a warning message.
inline void warning(const char* msg) { amd::report_warning(msg); }

/*! \brief Abort the program with a "ShouldNotReachHere" message.
 *  \hideinitializer
 */
#define ShouldNotReachHere() fatal("ShouldNotReachHere()")

/*! \brief Abort the program with a "ShouldNotCallThis" message.
 *  \hideinitializer
 */
#define ShouldNotCallThis() fatal("ShouldNotCallThis()")

/*! \brief Abort the program with an "Unimplemented" message.
 *  \hideinitializer
 */
#define Unimplemented() fatal("Unimplemented()")

/*! \brief Display an "Untested" warning message.
 *  \hideinitializer
 */
#ifndef NDEBUG
#define Untested(msg) warning("Untested(\"" msg "\")")
#else /*NDEBUG*/
#define Untested(msg) (void)(0)
#endif /*NDEBUG*/

#ifdef DEBUG
#define Log(level, msg)                                                                            \
  do {                                                                                             \
    if (LOG_LEVEL >= level) {                                                                      \
      amd::log_entry(level, __FILE__, __LINE__, msg);                                              \
    }                                                                                              \
  } while (false)
#else  // !DEBUG
#define Log(level, msg) (void)(0)
#endif  // !DEBUG

#ifdef DEBUG
#define LogTS(level, msg)                                                                          \
  do {                                                                                             \
    if (LOG_LEVEL >= level) {                                                                      \
      amd::log_timestamped(level, __FILE__, __LINE__, msg);                                        \
    }                                                                                              \
  } while (false)
#else  // !DEBUG
#define Log(level, msg) (void)(0)
#endif  // !DEBUG

#ifdef DEBUG
#define Logf(level, format, ...)                                                                   \
  do {                                                                                             \
    if (LOG_LEVEL >= level) {                                                                      \
      amd::log_printf(level, __FILE__, __LINE__, format, __VA_ARGS__);                             \
    }                                                                                              \
  } while (false)
#else  // !DEBUG
#define Logf(level, format, ...) (void)(0)
#endif  // !DEBUG

#define CondLog(cond, msg)                                                                         \
  do {                                                                                             \
    if (false DEBUG_ONLY(|| (cond))) {                                                             \
      Log(amd::LOG_INFO, msg);                                                                     \
    }                                                                                              \
  } while (false)

#ifdef DEBUG
#define LogGuarantee(cond, level, msg)                                                             \
  do {                                                                                             \
    if (LOG_LEVEL >= level) {                                                                      \
      guarantee(cond);                                                                             \
    }                                                                                              \
  } while (false)
#else  // !DEBUG
#define LogGuarantee(cond, level, msg) (void)(0)
#endif  // !DEBUG

#define LogInfo(msg) Log(amd::LOG_INFO, msg)
#define LogError(msg) Log(amd::LOG_ERROR, msg)
#define LogWarning(msg) Log(amd::LOG_WARNING, msg)

#define LogTSInfo(msg) LogTS(amd::LOG_INFO, msg)
#define LogTSError(msg) LogTS(amd::LOG_ERROR, msg)
#define LogTSWarning(msg) LogTS(amd::LOG_WARNING, msg)

#define LogPrintfDebug(format, ...) Logf(amd::LOG_DEBUG, format, __VA_ARGS__)
#define LogPrintfError(format, ...) Logf(amd::LOG_ERROR, format, __VA_ARGS__)
#define LogPrintfWarning(format, ...) Logf(amd::LOG_WARNING, format, __VA_ARGS__)
#define LogPrintfInfo(format, ...) Logf(amd::LOG_INFO, format, __VA_ARGS__)

#define DebugInfoGuarantee(cond) LogGuarantee(cond, amd::LOG_INFO, "Warning")

#endif /*DEBUG_HPP_*/
