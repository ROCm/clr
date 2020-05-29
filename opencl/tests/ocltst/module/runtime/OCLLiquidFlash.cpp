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

#include "OCLLiquidFlash.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "CL/cl.h"

const static size_t ChunkSize = 256 * 1024;
const static int NumSizes = 5;
const static int NumChunksArray[NumSizes] = {1, 4, 16, 32, 56};
const static size_t MaxSubTests = 4 * NumSizes;
const static char* BinFileName = "LiquidFlash.bin";
const static int NumIterArray[NumSizes] = {20, 15, 10, 10, 10};
const static int NumStagesArray[NumSizes] = {2, 2, 4, 4, 4};

OCLLiquidFlash::OCLLiquidFlash() {
#ifdef CL_VERSION_2_0
  _numSubTests = MaxSubTests;
  failed_ = false;
  maxSize_ = 0;
  direct_ = false;
  amdFile_ = NULL;
#else
  _numSubTests = 0;
  failed_ = false;
  maxSize_ = 0;
  direct_ = false;
#endif
}

OCLLiquidFlash::~OCLLiquidFlash() {}

void OCLLiquidFlash::open(unsigned int test, char* units, double& conversion,
                          unsigned int deviceId) {
#ifdef CL_VERSION_2_0
  failed_ = false;
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;
  char name[1024] = {0};
  size_t size = 0;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 1024,
                            name, &size);

  if (!strstr(name, "cl_amd_liquid_flash")) {
    printf("Liquid flash extension is required for this test!\n");
    failed_ = true;
    return;
  }

  NumChunks = NumChunksArray[testID_ / 4];
  NumIter = NumIterArray[testID_ / 4];
  NumStages = NumStagesArray[testID_ / 4];
  BufferSize = NumChunks * ChunkSize * sizeof(cl_uint);
  direct_ = ((testID_ % 4) < 3) ? true : false;
  createFile =
      (clCreateSsgFileObjectAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform_, "clCreateSsgFileObjectAMD");
  retainFile =
      (clRetainSsgFileObjectAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform_, "clRetainSsgFileObjectAMD");
  releaseFile =
      (clReleaseSsgFileObjectAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform_, "clReleaseSsgFileObjectAMD");
  writeBufferFromFile =
      (clEnqueueReadSsgFileAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform_, "clEnqueueReadSsgFileAMD");
  if (createFile == NULL || retainFile == NULL || releaseFile == NULL ||
      writeBufferFromFile == NULL) {
    testDescString = "Failed to initialize LiquidFlash extension!\n";
    failed_ = true;
    return;
  }

  size_t chunkSize = ChunkSize;
  std::ofstream fs;
  fs.open(BinFileName, std::fstream::binary);

  if (fs.is_open()) {
    // allocate memory for file content
    cl_uint* buffer = new cl_uint[chunkSize];
    for (cl_uint i = 0; i < chunkSize; ++i) {
      buffer[i] = i;
    }
    for (int i = 0; i < NumChunks; ++i) {
      fs.write(reinterpret_cast<char*>(buffer), chunkSize * sizeof(cl_uint));
    }
    delete[] buffer;
  }
  fs.close();

  std::string str(BinFileName);
  std::wstring wc(str.length(), L' ');
  // Copy string to wstring.
  std::copy(str.begin(), str.end(), wc.begin());

  amdFile_ = createFile(context_, CL_FILE_READ_ONLY_AMD, wc.c_str(), &error_);
  if (error_ != CL_SUCCESS) {
    printf(
        "Create file failed. Liquid flash support is required for this "
        "test!\n");
    failed_ = true;
    return;
  }

  cl_mem buf = NULL;
  if (direct_) {
    cl_uint subTest = testID_ % 4;
    cl_uint memFlags = (subTest == 0)
                           ? CL_MEM_USE_PERSISTENT_MEM_AMD
                           : ((subTest == 1) ? CL_MEM_ALLOC_HOST_PTR : 0);
    buf = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY | memFlags,
                                   BufferSize, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS),
                 "clEnqueueWriteBufferFromFileAMD() failed");
  } else {
    for (int i = 0; i < NumStages; ++i) {
      buf = _wrapper->clCreateBuffer(context_,
                                     CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                     BufferSize / NumStages, NULL, &error_);
      CHECK_RESULT((error_ != CL_SUCCESS),
                   "clEnqueueWriteBufferFromFileAMD() failed");
      buffers_.push_back(buf);
    }

    buf = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, BufferSize,
                                   NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS),
                 "clEnqueueWriteBufferFromFileAMD() failed");
  }
  buffers_.push_back(buf);
#endif
}

void OCLLiquidFlash::run(void) {
#ifdef CL_VERSION_2_0
  if (failed_) {
    return;
  }
  size_t finalBuf = (direct_) ? 0 : NumStages;

  cl_uint* buffer = new cl_uint[NumChunks * ChunkSize];
  size_t iterSize = BufferSize / NumStages;
  memset(buffer, 0, BufferSize);
  if (direct_) {
    error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers_[0],
                                            CL_TRUE, 0, BufferSize, buffer, 0,
                                            NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
  } else {
    for (int i = 0; i < NumStages; ++i) {
      error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId],
                                              buffers_[i], CL_TRUE, 0, iterSize,
                                              buffer, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
    }
    error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId],
                                            buffers_[finalBuf], CL_TRUE, 0,
                                            BufferSize, buffer, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");
  }

  CPerfCounter timer;

  double sec = 0.;

  for (int i = 0; i < NumIter; ++i) {
    timer.Reset();
    timer.Start();
    if (direct_) {
      error_ = writeBufferFromFile(
          cmdQueues_[_deviceId], buffers_[0], CL_FALSE, 0 /*buffer_offset*/,
          BufferSize, amdFile_ /*file*/, 0 /*file_offset*/, 0, NULL, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "writeBufferFromFile() failed");
    } else {
      for (int i = 0; i < NumStages; ++i) {
        error_ = writeBufferFromFile(
            cmdQueues_[_deviceId], buffers_[i], CL_FALSE, 0 /*buffer_offset*/,
            iterSize, amdFile_ /*file*/, iterSize * i /*file_offset*/, 0, NULL,
            NULL);
        CHECK_RESULT((error_ != CL_SUCCESS), "writeBufferFromFile() failed");

        error_ = _wrapper->clEnqueueCopyBuffer(
            cmdQueues_[_deviceId], buffers_[i], buffers_[NumStages], 0,
            iterSize * i, iterSize, 0, NULL, NULL);
        CHECK_RESULT((error_ != CL_SUCCESS), "CopyBuffer() failed");
        _wrapper->clFlush(cmdQueues_[_deviceId]);
      }
    }
    _wrapper->clFinish(cmdQueues_[_deviceId]);
    timer.Stop();
    double cur = timer.GetElapsedTime();
    if (i == 0) {
      sec = cur;
    } else {
      sec = std::min(cur, sec);
    }
  }

  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId],
                                         buffers_[finalBuf], CL_TRUE, 0,
                                         BufferSize, buffer, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "Validation failed!");

  for (int c = 0; c < NumChunks; ++c) {
    for (cl_uint i = 0; i < ChunkSize; ++i) {
      if (buffer[c * ChunkSize + i] != i) {
        CHECK_RESULT(false, "Validation failed!");
      }
    }
  }
  delete[] buffer;

  static const char* MemTypeStr[] = {"Visible  ", "Remote   ", "Invisible",
                                     "Staging"};
  _perfInfo = (float)BufferSize / ((float)sec * 1024.f * 1024.f);
  std::stringstream str;
  str << "WriteBufferFromFile performance (";
  str << BufferSize / (1024 * 1024);
  str << " MB of " << MemTypeStr[testID_ % 4] << ") transfer speed (MB/s):";
  testDescString = str.str();
#endif
}

unsigned int OCLLiquidFlash::close(void) {
#ifdef CL_VERSION_2_0
  if (!failed_) {
    if (amdFile_ != NULL) {
      releaseFile(amdFile_);
    }
    if (remove(BinFileName) != 0) {
    }
  }
  return OCLTestImp::close();
#else
  return CL_SUCCESS;
#endif
}
