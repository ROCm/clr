//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef BUILTINS_HPP_
#define BUILTINS_HPP_

#include "top.hpp"
#include "amdocl/cl_kernel.h"

namespace cpu {

struct Builtins : public amd::AllStatic
{
    static const clk_builtins_t dispatchTable_;
};

} // namespace cpu

#endif /*BUILTINS_HPP_*/
