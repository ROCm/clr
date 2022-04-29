/*
Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef AMD_HIP_MATH_CONSTANTS_H
#define AMD_HIP_MATH_CONSTANTS_H
#define HIP_INF_F            __int_as_float(0x7f800000U)
#define HIP_NAN_F            __int_as_float(0x7fffffffU)
#define HIP_MIN_DENORM_F     __int_as_float(0x00000001U)
#define HIP_MAX_NORMAL_F     __int_as_float(0x7f7fffffU)
#define HIP_NEG_ZERO_F       __int_as_float(0x80000000U)
#define HIP_ZERO_F           0.0F
#define HIP_ONE_F            1.0F
#define HIP_SQRT_HALF_F      0.707106781F
#define HIP_SQRT_HALF_HI_F   0.707106781F
#define HIP_SQRT_HALF_LO_F   1.210161749e-08F
#define HIP_SQRT_TWO_F       1.414213562F
#define HIP_THIRD_F          0.333333333F
#define HIP_PIO4_F           0.785398163F
#define HIP_PIO2_F           1.570796327F
#define HIP_3PIO4_F          2.356194490F
#define HIP_2_OVER_PI_F      0.636619772F
#define HIP_SQRT_2_OVER_PI_F 0.797884561F
#define HIP_PI_F             3.141592654F
#define HIP_L2E_F            1.442695041F
#define HIP_L2T_F            3.321928094F
#define HIP_LG2_F            0.301029996F
#define HIP_LGE_F            0.434294482F
#define HIP_LN2_F            0.693147181F
#define HIP_LNT_F            2.302585093F
#define HIP_LNPI_F           1.144729886F
#define HIP_TWO_TO_M126_F    1.175494351e-38F
#define HIP_TWO_TO_126_F     8.507059173e37F
#define HIP_NORM_HUGE_F      3.402823466e38F
#define HIP_TWO_TO_23_F      8388608.0F
#define HIP_TWO_TO_24_F      16777216.0F
#define HIP_TWO_TO_31_F      2147483648.0F
#define HIP_TWO_TO_32_F      4294967296.0F
#define HIP_REMQUO_BITS_F    3U
#define HIP_REMQUO_MASK_F    (~((~0U)<<HIPRT_REMQUO_BITS_F))
#define HIP_TRIG_PLOSS_F     105615.0F
#endif
