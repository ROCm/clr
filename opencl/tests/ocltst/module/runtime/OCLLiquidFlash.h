/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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

#ifndef _OCL_LIQUID_FLASH_H_
#define _OCL_LIQUID_FLASH_H_

#include "OCLTestImp.h"

class OCLLiquidFlash : public OCLTestImp {
 public:
  OCLLiquidFlash();
  virtual ~OCLLiquidFlash();

 public:
  virtual void open(unsigned int test, char* units, double& conversion,
                    unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int testID_;
  cl_ulong maxSize_;
#ifdef CL_VERSION_2_0
  cl_file_amd amdFile_;
#endif
  bool direct_;
  size_t BufferSize;
  int NumChunks;
  int NumIter;
  int NumStages;
#ifdef CL_VERSION_2_0
  clCreateSsgFileObjectAMD_fn createFile;
  clRetainSsgFileObjectAMD_fn retainFile;
  clReleaseSsgFileObjectAMD_fn releaseFile;
  clEnqueueReadSsgFileAMD_fn writeBufferFromFile;
#endif
};

#endif  // _OCL_LIQUID_FLASH_H_
