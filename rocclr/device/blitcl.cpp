/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

namespace device {

#define BLIT_KERNELS(...) #__VA_ARGS__

const char* BlitLinearSourceCode = BLIT_KERNELS(
  // Extern
  extern void __amd_fillBufferAligned(__global uchar*, __global ushort*, __global uint*,
                                      __global ulong*, __constant uchar*, uint, ulong, ulong);

  extern void __amd_fillBufferAligned2D(__global uchar*, __global ushort*, __global uint*,
                                        __global ulong*, __constant uchar*, uint, ulong, ulong,
                                        ulong, ulong);

  extern void __amd_copyBuffer(__global uchar*, __global uchar*, ulong, ulong, ulong, uint);

  extern void __amd_copyBufferAligned(__global uint*, __global uint*, ulong, ulong, ulong, uint);

  extern void __amd_copyBufferRect(__global uchar*, __global uchar*, ulong4, ulong4, ulong4);

  extern void __amd_copyBufferRectAligned(__global uint*, __global uint*, ulong4, ulong4, ulong4);

  extern void __amd_streamOpsWrite(__global uint*, __global ulong*, ulong, ulong);

  extern void __amd_streamOpsWait(__global uint*, __global ulong*, ulong, ulong, ulong);

  extern void __ockl_dm_init_v1(ulong, ulong, uint, uint);

  extern void __ockl_gws_init(uint nwm1, uint rid);

  // Implementation
  __kernel void __amd_rocclr_fillBufferAligned(__global uchar* bufUChar,
                                               __global ushort* bufUShort,
                                               __global uint* bufUInt,
                                               __global ulong* bufULong,
                                               __constant uchar* pattern,
                                               uint patternSize, ulong offset,
                                               ulong size) {
    __amd_fillBufferAligned(bufUChar, bufUShort, bufUInt, bufULong,
                            pattern, patternSize, offset, size);
  }

  __kernel void __amd_rocclr_fillBufferAligned2D(__global uchar* bufUChar,
                                                 __global ushort* bufUShort,
                                                 __global uint* bufUInt,
                                                 __global ulong* bufULong,
                                                 __constant uchar* pattern,
                                                 uint patternSize,
                                                 ulong offset,
                                                 ulong width,
                                                 ulong height,
                                                 ulong pitch) {
    __amd_fillBufferAligned2D(bufUChar, bufUShort, bufUInt, bufULong,
                              pattern, patternSize, offset, width, height,
                              pitch);
  }

  __kernel void __amd_rocclr_copyBuffer(__global uchar* srcI, __global uchar* dstI,
                                        ulong srcOrigin, ulong dstOrigin, ulong size,
                                        uint remain) {
    __amd_copyBuffer(srcI, dstI, srcOrigin, dstOrigin, size, remain);
  }

  __kernel void __amd_rocclr_copyBufferAligned(__global uint* src, __global uint* dst,
                                               ulong srcOrigin, ulong dstOrigin, ulong size,
                                               uint alignment) {
    __amd_copyBufferAligned(src, dst, srcOrigin, dstOrigin, size, alignment);
  }

  __kernel void __amd_rocclr_copyBufferRect(__global uchar* src, __global uchar* dst,
                                            ulong4 srcRect, ulong4 dstRect, ulong4 size) {
    __amd_copyBufferRect(src, dst, srcRect, dstRect, size);
  }

  __kernel void __amd_rocclr_copyBufferRectAligned(__global uint* src, __global uint* dst,
                                                   ulong4 srcRect, ulong4 dstRect, ulong4 size) {
    __amd_copyBufferRectAligned(src, dst, srcRect, dstRect, size);
  }
);

const char* HipExtraSourceCode = BLIT_KERNELS(
  __kernel void __amd_rocclr_streamOpsWrite(__global uint* ptrInt, __global ulong* ptrUlong,
                                            ulong value, ulong sizeBytes) {
    __amd_streamOpsWrite(ptrInt, ptrUlong, value, sizeBytes);
  }

  __kernel void __amd_rocclr_streamOpsWait(__global uint* ptrInt, __global ulong* ptrUlong,
                                           ulong value, ulong flags, ulong mask) {
    __amd_streamOpsWait(ptrInt, ptrUlong, value, flags, mask);
  }

  __kernel void __amd_rocclr_initHeap(ulong heap_to_initialize, ulong initial_blocks,
                                      uint heap_size, uint number_of_initial_blocks) {
    __ockl_dm_init_v1(heap_to_initialize, initial_blocks, heap_size, number_of_initial_blocks);
  }

  __kernel void __amd_rocclr_gwsInit(uint value) {
    __ockl_gws_init(value, 0);
  }
);

const char* BlitImageSourceCode = BLIT_KERNELS(
  // Extern
  extern void __amd_fillImage(__write_only image2d_array_t, float4, int4, uint4, int4, int4,
                              uint);

  extern void __amd_copyImage(__read_only image2d_array_t, __write_only image2d_array_t, int4,
                              int4, int4);

  extern void __amd_copyImage1DA(__read_only image2d_array_t, __write_only image2d_array_t, int4,
                                 int4, int4);

  extern void __amd_copyBufferToImage(__global uint*, __write_only image2d_array_t, ulong4,
                                      int4, int4, uint4, ulong4);

  extern void __amd_copyImageToBuffer(__read_only image2d_array_t, __global uint*,
                                      __global ushort*, __global uchar*, int4, ulong4, int4,
                                      uint4, ulong4);

  __kernel void __amd_rocclr_fillImage(__write_only image2d_array_t image, float4 patternFLOAT4,
                                       int4 patternINT4, uint4 patternUINT4, int4 origin,
                                       int4 size, uint type) {
    __amd_fillImage(image, patternFLOAT4, patternINT4, patternUINT4, origin, size, type);
  }

  __kernel void __amd_rocclr_copyImage(__read_only image2d_array_t src,
                                       __write_only image2d_array_t dst, int4 srcOrigin,
                                       int4 dstOrigin, int4 size) {
    __amd_copyImage(src, dst, srcOrigin, dstOrigin, size);
  }

  __kernel void __amd_rocclr_copyImage1DA(__read_only image2d_array_t src,
                                          __write_only image2d_array_t dst, int4 srcOrigin,
                                          int4 dstOrigin, int4 size) {
    __amd_copyImage1DA(src, dst, srcOrigin, dstOrigin, size);
  }

  __kernel void __amd_rocclr_copyBufferToImage(__global uint* src,
                                               __write_only image2d_array_t dst, ulong4 srcOrigin,
                                               int4 dstOrigin, int4 size, uint4 format,
                                               ulong4 pitch) {
    __amd_copyBufferToImage(src, dst, srcOrigin, dstOrigin, size, format, pitch);
  }

  __kernel void __amd_rocclr_copyImageToBuffer(__read_only image2d_array_t src,
                                               __global uint* dstUInt,
                                               __global ushort* dstUShort,
                                               __global uchar* dstUChar,
                                               int4 srcOrigin, ulong4 dstOrigin, int4 size,
                                               uint4 format, ulong4 pitch) {
    __amd_copyImageToBuffer(src, dstUInt, dstUShort, dstUChar, srcOrigin, dstOrigin, size, format,
                            pitch);
  }

);

}  // namespace device
