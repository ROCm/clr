//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef LIBRARY_H_
#define LIBRARY_H_

#include <vector>
#include <string>
namespace amd {

typedef enum _library_selector {
    LibraryUndefined = 0,
    GPU_Library_7xx,
    GPU_Library_Evergreen,
    GPU_Library_SI,
    CPU_Library_Generic,
    CPU_Library_AVX,
    CPU_Library_FMA4,
    GPU_Library_Generic,
    CPU64_Library_Generic,
    CPU64_Library_AVX,
    CPU64_Library_FMA4,
    GPU64_Library_Evergreen,
    GPU64_Library_SI,
    GPU64_Library_Generic,
    GPU_Library_CI,
    GPU64_Library_CI,
    GPU_Library_HSAIL,
    LibraryTotal
} LibrarySelector;

/** Integrated Bitcode Libararies **/
class LibraryDescriptor {
public:
    enum {MAX_NUM_LIBRARY_DESCS = 11};

    const char* start;
    size_t      size;
};

int getLibDescs (
    LibrarySelector     LibType,      // input
    LibraryDescriptor*  LibDesc,      // output
    int&                LibDescSize   // output -- LibDesc[0:LibDescSize-1]
);

static const char* amdRTFuns[] = {
      "__amdrt_div_i64",
      "__amdrt_div_u64",
      "__amdrt_mod_i64",
      "__amdrt_mod_u64",
      "__amdrt_cvt_f64_to_u64",
      "__amdrt_cvt_f32_to_u64"
};
} //amd

#endif // LIBRARY_H_

