//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpubuiltins.hpp"
#include "device/cpu/cpucommand.hpp"

#include <amdocl/cl_kernel.h>
#include <cstdio>  // for printf
#include <stdarg.h>

#define BUF_SIZE_PRINTF 4095
// In the current implementation of printf in gcc 4.5.2 runtime libraries,inf/infinity and nan are
// not supported
// The [-]infinity value is printed as [-]1.#INF00
// The [-]nan value is printed as [-]1.#INF00
// bufOutUpdate converts the all printed instanced of [-]1.#INF00 to inf,and
//                          all printed instanced of [-]1.#IND00 to nan
void bufOutUpdate(std::string& sBufOut, const char* strToReplace, const char* strReplace) {
  size_t foundIdx = 0;
  while ((foundIdx = sBufOut.find(strToReplace, foundIdx)) != std::string::npos) {
    sBufOut.replace(foundIdx, strlen(strToReplace), strReplace, strlen(strReplace));
    foundIdx += 3;
  }
}
int cpuprintf(const char* format, ...) {
  char cBufOut[BUF_SIZE_PRINTF];
  std::string sBufOut;
  va_list args;
  va_start(args, format);
  // write to the buffer
  vsprintf(cBufOut, format, args);
  sBufOut = cBufOut;

  // convert to correct infinity/nan representation
  bufOutUpdate(sBufOut, "1.#INF00", "inf");
  bufOutUpdate(sBufOut, "1.#IND00", "nan");
  bufOutUpdate(sBufOut, "1.#QNAN0", "nan");
  int ret = amd::Os::printf("%s", sBufOut.c_str());
  fflush(stdout);
  va_end(args);
  return ret;
}
namespace cpu {

const clk_builtins_t Builtins::dispatchTable_ = {
    /* Synchronization functions */
    &WorkItem::barrier,
    /* AMD Only builtins: FIXME_lmoriche: remove or add an extension */
    NULL, cpuprintf};

}  // namespace cpu
