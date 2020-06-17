/* Copyright (c) 2020-present Advanced Micro Devices, Inc.

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

/** \file Format string processing for printf based on hostcall messages.
 */

#include <assert.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

static void checkPrintf(int* outCount, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int retval = vprintf(fmt, args);
  *outCount = retval < 0 ? retval : *outCount + retval;
}

static int countStars(const std::string& spec) {
  int stars = 0;
  for (auto c : spec) {
    if (c == '*') {
      ++stars;
    }
  }
  return stars;
}

template <typename... Args>
static const uint64_t* consumeInteger(int* outCount, const std::string& spec, const uint64_t* ptr,
                                      Args... args) {
  checkPrintf(outCount, spec.c_str(), args..., ptr[0]);
  return ptr + 1;
}

template <typename... Args>
static const uint64_t* consumeFloatingPoint(int* outCount, const std::string& spec,
                                            const uint64_t* ptr, Args... args) {
  double d;
  memcpy(&d, ptr, 8);
  checkPrintf(outCount, spec.c_str(), args..., d);
  return ptr + 1;
}

template <typename... Args>
static const uint64_t* consumeCstring(int* outCount, const std::string& spec, const uint64_t* ptr,
                                      Args... args) {
  auto str = reinterpret_cast<const char*>(ptr);
  checkPrintf(outCount, spec.c_str(), args..., str);
  return ptr + (strlen(str) + 7) / 8;
}

template <typename... Args>
static const uint64_t* consumePointer(int* outCount, const std::string& spec, const uint64_t* ptr,
                                      Args... args) {
  auto vptr = reinterpret_cast<void*>(*ptr);
  checkPrintf(outCount, spec.c_str(), args..., vptr);
  return ptr + 1;
}

template <typename... Args>
static const uint64_t* consumeArgument(int* outCount, const std::string& spec, const uint64_t* ptr,
                                       const uint64_t* end, Args... args) {
  switch (spec.back()) {
    case 'd':
    case 'i':
    case 'o':
    case 'u':
    case 'x':
    case 'X':
    case 'c':
      return consumeInteger(outCount, spec, ptr, args...);
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A':
      return consumeFloatingPoint(outCount, spec, ptr, args...);
    case 's':
      return consumeCstring(outCount, spec, ptr, args...);
    case 'p':
      return consumePointer(outCount, spec, ptr, args...);
    case 'n':
      return ptr + 1;
  }

  // Undefined behaviour with an unknown flag
  return end;
}

static const uint64_t* processSpec(int* outCount, const std::string& spec, const uint64_t* ptr,
                                   const uint64_t* end) {
  auto stars = countStars(spec);
  assert(stars < 3 && "cannot have more than two placeholders");
  switch (stars) {
    case 0:
      return consumeArgument(outCount, spec, ptr, end);
    case 1:
      // Undefined behaviour if there are not enough arguments.
      if (end - ptr < 2) {
        return end;
      }
      return consumeArgument(outCount, spec, ptr + 1, end, ptr[0]);
    case 2:
      // Undefined behaviour if there are not enough arguments.
      if (end - ptr < 3) {
        return end;
      }
      return consumeArgument(outCount, spec, ptr + 2, end, ptr[0], ptr[1]);
  }

  // Undefined behaviour if three are more than two stars.
  return end;
}

/** \brief Process a printf message using the system printf function.
 * \param begin Start of the uint64_t array containing the message.
 * \param end   One past the last element in the array.
 * \return An integer that satisfies the POSIX return value for printf.
 *
 * The message has the following format:
 *  - uint64_t version, required to be zero.
 *  - Format string padded to an 8 byte boundary.
 *  - Sequence of arguments
 *    - Each int/float/pointer argument occupies one uint64_t location.
 *    - Each string argument is padded to an 8 byte boundary.
 *
 * The format() function extracts the format string, and then
 * extracts further arguments based on the format string. It breaks
 * up the format string at the format specifiers and invokes the
 * system printf() function multiple times:
 * - A format specifier and its corresponding arguments are passed to
 *   a separate printf() call.
 * - Slices between the format specifiers are passed to additional
 *   printf() calls interleaved with the specifiers.
 *
 * Limitations:
 * - Behaviour is undefined with wide characters and strings.
 * - %n specifier is ignored and the corresponding argument is skipped.
 */
static int format(const uint64_t* begin, const uint64_t* end) {
  const char convSpecifiers[] = "diouxXfFeEgGaAcspn";
  auto ptr = begin;

  const std::string fmt(reinterpret_cast<const char*>(ptr));
  ptr += (fmt.length() + 7 + 1) / 8;  // the extra '1' is for the null

  int outCount = 0;
  size_t point = 0;
  while (true) {
    // Each segment of the format string delineated by [mark,
    // point) is handled seprately.
    auto mark = point;
    point = fmt.find('%', point);

    // Two different cases where a literal segment is printed out.
    // 1. When the point reaches the end of the format string.
    // 2. When the point is at the start of a format specifier.
    if (point == std::string::npos) {
      checkPrintf(&outCount, "%s", &fmt[mark]);
      return outCount;
    }
    checkPrintf(&outCount, "%.*s", (int)(point - mark), &fmt[mark]);
    if (outCount < 0) {
      return outCount;
    }

    mark = point;
    ++point;

    // Handle the simplest specifier, '%%'.
    if (fmt[point] == '%') {
      checkPrintf(&outCount, "%%");
      if (outCount < 0) {
        return outCount;
      }
      ++point;
      continue;
    }

    // Before processing the specifier, check if we have run out
    // of arguments.
    if (ptr == end) {
      return outCount;
    }

    // Undefined behaviour if we don't see a conversion specifier.
    point = fmt.find_first_of(convSpecifiers, point);
    if (point == std::string::npos) {
      return outCount;
    }
    ++point;

    // [mark,point) now contains a complete specifier.
    const std::string spec(fmt, mark, point - mark);
    ptr = processSpec(&outCount, spec, ptr, end);
    if (outCount < 0) {
      return outCount;
    }
  }
}

void handlePrintf(uint64_t* output, const uint64_t* input, uint64_t len) {
  auto version = *input;
  if (version != 0) {
    *output = -1;
    return;
  }

  *output = format(input + 1, input + len);
}
