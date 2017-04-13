//
// Copyright 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef CPUFEAT_HPP
#define CPUFEAT_HPP

#define CPUFEAT_CX_SSE3 (1 << 0)
#define CPUFEAT_CX_SSSE3 (1 << 9)
#define CPUFEAT_CX_CMPXCHG16B (1 << 13)
#define CPUFEAT_CX_SSE4_1 (1 << 19)
#define CPUFEAT_CX_SSE4_2 (1 << 20)
#define CPUFEAT_CX_POPCNT (1 << 23)
#define CPUFEAT_CX_AES (1 << 25)
#define CPUFEAT_CX_OSXSAVE (1 << 27)
#define CPUFEAT_CX_AVX (1 << 28)

#define INTEL_CPUFEAT_CX_FMA3 (1 << 12)

#define AMD_CPUFEAT_CX_FMA4 (1 << 16)
#define AMD_CPUFEAT_CX_XOP (1 << 11)
#define AMD_CPUFEAT_CX_SSE4A (1 << 6)

#define CPUFEAT_DX_SSE (1 < 25)
#define CPUFEAT_DX_SSE2 (1 << 26)

#endif  // CPUFEAT_HPP
