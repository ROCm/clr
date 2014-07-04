//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

namespace device {

#define BLIT_KERNELS(...) #__VA_ARGS__

const char* BlitSourceCode = BLIT_KERNELS(
\n
__kernel void copyBufferToImage(
    __global    uint*       src,
    __write_only image2d_array_t  dst,
    int4        srcOrigin,
    int4        dstOrigin,
    int4        size,
    int4        format,
    int4        pitch)
{
    uint     idxSrc;
    int4     coordsDst;
    uint4    pixel;
    __global uint*   srcUInt = src;
    __global ushort* srcUShort = (__global ushort*)src;
    __global uchar*  srcUChar  = (__global uchar*)src;
    ushort   tmpUShort;
    uint     tmpUInt;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    idxSrc = (coordsDst.z * pitch.y +
       coordsDst.y * pitch.x + coordsDst.x) *
       format.z + srcOrigin.x;

    coordsDst.x += dstOrigin.x;
    coordsDst.y += dstOrigin.y;
    coordsDst.z += dstOrigin.z;

    // Check components
    switch (format.x) {
    case 1:
        // Check size
        switch (format.y) {
        case 1:
            pixel.x = (uint)srcUChar[idxSrc];
            break;
        case 2:
            pixel.x = (uint)srcUShort[idxSrc];
            break;
        case 4:
            pixel.x = srcUInt[idxSrc];
            break;
        }
    break;
    case 2:
        // Check size
        switch (format.y) {
        case 1:
            tmpUShort = srcUShort[idxSrc];
            pixel.x = (uint)(tmpUShort & 0xff);
            pixel.y = (uint)(tmpUShort >> 8);
            break;
        case 2:
            tmpUInt = srcUInt[idxSrc];
            pixel.x = (tmpUInt & 0xffff);
            pixel.y = (tmpUInt >> 16);
            break;
        case 4:
            pixel.x = srcUInt[idxSrc++];
            pixel.y = srcUInt[idxSrc];
            break;
        }
    break;
    case 4:
        // Check size
        switch (format.y) {
        case 1:
            tmpUInt = srcUInt[idxSrc];
            pixel.x = tmpUInt & 0xff;
            pixel.y = (tmpUInt >> 8) & 0xff;
            pixel.z = (tmpUInt >> 16) & 0xff;
            pixel.w = (tmpUInt >> 24) & 0xff;
            break;
        case 2:
            tmpUInt = srcUInt[idxSrc++];
            pixel.x = tmpUInt & 0xffff;
            pixel.y = (tmpUInt >> 16);
            tmpUInt = srcUInt[idxSrc];
            pixel.z = tmpUInt & 0xffff;
            pixel.w = (tmpUInt >> 16);
            break;
        case 4:
            pixel.x = srcUInt[idxSrc++];
            pixel.y = srcUInt[idxSrc++];
            pixel.z = srcUInt[idxSrc++];
            pixel.w = srcUInt[idxSrc];
            break;
        }
    break;
    }
    // Write the final pixel
    write_imageui(dst, coordsDst, pixel);
}
\n
__kernel void copyImageToBuffer(
    __read_only image2d_array_t   src,
    __global    uint*       dstUInt,
    __global    ushort*     dstUShort,
    __global    uchar*      dstUChar,
    int4        srcOrigin,
    int4        dstOrigin,
    int4        size,
    int4        format,
    int4        pitch)
{
    uint     idxDst;
    int4     coordsSrc;
    uint4    texel;

    coordsSrc.x = get_global_id(0);
    coordsSrc.y = get_global_id(1);
    coordsSrc.z = get_global_id(2);
    coordsSrc.w = 0;

    if ((coordsSrc.x >= size.x) ||
        (coordsSrc.y >= size.y) ||
        (coordsSrc.z >= size.z)) {
        return;
    }

    idxDst = (coordsSrc.z * pitch.y + coordsSrc.y * pitch.x +
        coordsSrc.x) * format.z + dstOrigin.x;

    coordsSrc.x += srcOrigin.x;
    coordsSrc.y += srcOrigin.y;
    coordsSrc.z += srcOrigin.z;

    texel = read_imageui(src, coordsSrc);

    // Check components
    switch (format.x) {
    case 1:
        // Check size
        switch (format.y) {
        case 1:
            dstUChar[idxDst] = (uchar)texel.x;
            break;
        case 2:
            dstUShort[idxDst] = (ushort)texel.x;
            break;
        case 4:
            dstUInt[idxDst] = texel.x;
            break;
        }
    break;
    case 2:
        // Check size
        switch (format.y) {
        case 1:
            dstUShort[idxDst] = (ushort)texel.x |
               ((ushort)texel.y << 8);
            break;
        case 2:
            dstUInt[idxDst] = texel.x | (texel.y << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst] = texel.y;
            break;
        }
    break;
    case 4:
        // Check size
        switch (format.y) {
        case 1:
            dstUInt[idxDst] = (uint)texel.x |
               (texel.y << 8) |
               (texel.z << 16) |
               (texel.w << 24);
            break;
        case 2:
            dstUInt[idxDst++] = texel.x | (texel.y << 16);
            dstUInt[idxDst] = texel.z | (texel.w << 16);
            break;
        case 4:
            dstUInt[idxDst++] = texel.x;
            dstUInt[idxDst++] = texel.y;
            dstUInt[idxDst++] = texel.z;
            dstUInt[idxDst] = texel.w;
            break;
        }
    break;
    }
}
\n
__kernel void copyImage(
    __read_only  image2d_array_t src,
    __write_only image2d_array_t dst,
    int4       srcOrigin,
    int4       dstOrigin,
    int4       size)
{
    int4  coordsDst;
    int4  coordsSrc;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    coordsSrc = srcOrigin + coordsDst;
    coordsDst += dstOrigin;

    uint4  texel;
    texel = read_imageui(src, coordsSrc);
    write_imageui(dst, coordsDst, texel);
}
\n
__kernel void copyImage1DA(
    __read_only  image2d_array_t  src,
    __write_only image2d_array_t  dst,
    int4       srcOrigin,
    int4       dstOrigin,
    int4       size)
{
    int4  coordsDst;
    int4  coordsSrc;

    coordsDst.x = get_global_id(0);
    coordsDst.y = get_global_id(1);
    coordsDst.z = get_global_id(2);
    coordsDst.w = 0;

    if ((coordsDst.x >= size.x) ||
        (coordsDst.y >= size.y) ||
        (coordsDst.z >= size.z)) {
        return;
    }

    coordsSrc = srcOrigin + coordsDst;
    coordsDst += dstOrigin;
    if (srcOrigin.w != 0) {
       coordsSrc.z = coordsSrc.y;
       coordsSrc.y = 0;
    }
    if (dstOrigin.w != 0) {
       coordsDst.z = coordsDst.y;
       coordsDst.y = 0;
    }

    uint4  texel;
    texel = read_imageui(src, coordsSrc);
    write_imageui(dst, coordsDst, texel);
}
\n
__kernel void copyBufferRect(
    __global   uchar*  src,
    __global   uchar*  dst,
    uint4      srcRect,
    uint4      dstRect,
    uint4      size)
{
    uint x = (uint)get_global_id(0);
    uint y = (uint)get_global_id(1);
    uint z = (uint)get_global_id(2);

    if ((x >= size.x) ||
        (y >= size.y) ||
        (z >= size.z)) {
        return;
    }

    uint offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
    uint offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

    dst[offsDst] = src[offsSrc];
}
\n
__kernel void copyBufferRectAligned(
    __global   uint*  src,
    __global   uint*  dst,
    uint4      srcRect,
    uint4      dstRect,
    uint4      size)
{
    uint x = (uint)get_global_id(0);
    uint y = (uint)get_global_id(1);
    uint z = (uint)get_global_id(2);

    if ((x >= size.x) ||
        (y >= size.y) ||
        (z >= size.z)) {
        return;
    }

    uint offsSrc = srcRect.z + x + y * srcRect.x + z * srcRect.y;
    uint offsDst = dstRect.z + x + y * dstRect.x + z * dstRect.y;

    if (size.w == 16) {
        __global uint4* src4 = (__global uint4*)src;
        __global uint4* dst4 = (__global uint4*)dst;
        dst4[offsDst] = src4[offsSrc];
    }
    else {
        dst[offsDst] = src[offsSrc];
    }
}
\n
__kernel void copyBuffer(
    __global   uchar*  src,
    __global   uchar*  dst,
    int       srcOrigin,
    int       dstOrigin,
    uint      size)
{
    uint id = (uint)get_global_id(0);

    if (id >= size) {
        return;
    }

    uint offsSrc = id + srcOrigin;
    uint offsDst = id + dstOrigin;

    dst[offsDst] = src[offsSrc];
}
\n
__kernel void copyBufferAligned(
    __global   uint*  src,
    __global   uint*  dst,
    int       srcOrigin,
    int       dstOrigin,
    uint      size,
    uint      alignment)
{
    uint id = (uint)get_global_id(0);

    if (id >= size) {
        return;
    }

    uint offsSrc = id + srcOrigin;
    uint offsDst = id + dstOrigin;

    if (alignment == 16) {
        __global uint4* src4 = (__global uint4*)src;
        __global uint4* dst4 = (__global uint4*)dst;
        dst4[offsDst] = src4[offsSrc];
    }
    else {
        dst[offsDst] = src[offsSrc];
    }
}
\n
__kernel void fillBuffer(
    __global   uchar*  bufUChar,
    __global   uint*   bufUInt,
    __constant uchar*  pattern,
    uint       patternSize,
    uint       offset,
    uint       size)
{
    uint id = (uint)get_global_id(0);

    if (id >= size) {
        return;
    }

    if (bufUInt) {
       __global uint* element = &bufUInt[offset + id * patternSize];
       __constant uint*  pt = (__constant uint*)pattern;

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pt[i];
        }
    }
    else {
        __global uchar* element = &bufUChar[offset + id * patternSize];

        for (uint i = 0; i < patternSize; ++i) {
            element[i] = pattern[i];
        }
    }
}
\n
__kernel void fillImage(
    __write_only image2d_array_t  image,
    float4     patternFLOAT4,
    int4       patternINT4,
    uint4      patternUINT4,
    int4       origin,
    int4       size,
    uint       type)
{
    int4  coords;

    coords.x = get_global_id(0);
    coords.y = get_global_id(1);
    coords.z = get_global_id(2);
    coords.w = 0;

    if ((coords.x >= size.x) ||
        (coords.y >= size.y) ||
        (coords.z >= size.z)) {
        return;
    }

    coords += origin;

    // Check components
    switch (type) {
    case 0:
        write_imagef(image, coords, patternFLOAT4);
        break;
    case 1:
        write_imagei(image, coords, patternINT4);
        break;
    case 2:
        write_imageui(image, coords, patternUINT4);
        break;
    }
}
\n
\n
);

} // namespace device
