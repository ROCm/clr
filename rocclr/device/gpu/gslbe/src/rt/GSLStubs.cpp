 /* Copyright (c) 2008-present Advanced Micro Devices, Inc.

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

#include "os_if.h"
#include "osws_if.h"

#include "atidefines.h"
#include "atitypes.h"
#include "scl_types.h"
#include "SCInterface.h"

//
//  This file represents the entry points that are stubbed out to satisfy the
//  linker, but aren't used in the runtime of GSL operations.
//


enum fsComponentType {
    FS_BYTE,
    FS_UNSIGNED_BYTE,
    FS_SHORT,
    FS_UNSIGNED_SHORT,
    FS_INT,
    FS_UNSIGNED_INT,
    FS_FLOAT,
    FS_FLOAT16,
};

enum fsInstrSet {
    FS_INSTR_KHAN,   ///< Generate Khan based instruction set
    FS_INSTR_PELE    ///< Generate Pele based instruction set
};

enum fsUsage {
    FS_USAGE_HW,     ///< An actual hardware stream
    FS_USAGE_SW      ///< A place holder stream (to support SW path)
};

struct fsInstr {
    fsUsage         usage;      ///< How the stream is going to be used (place holder or actual hardware stream)
    uint32          components; ///< Number of components to the input vector
    fsComponentType type;       ///< Type of each component
    bool32          normalize;  ///< Should the components be normalized to the -1..1 range
    uint32          stride;     ///< Stride between vectors
    uint32          ivmOffset;  ///< location in input vector memory
};


sclHandle CONV
sclInit(const sclShaderConstantAddress* shaderStateConstTable,
        const sclProfile&               profile,
        const sclLimits&                fpLimits,
        const sclLimits&                vpLimits)
{
    return 0;
}

void CONV
sclDestroy(sclHandle hSCL)
{
}

sclProgram* CONV
sclCompile(sclHandle                hSCL,
          const sclInputShader&    shader,
          const sclCompilerParams& params,
          const sclLimits&         limits)
{
    return 0;
}

sclProgramPair* CONV
sclLink(sclHandle                   hSCL,
       const sclInputMultShaderPair    *shader,
       const sclCompilerParams&    params,
       const sclLimits&            fpLimits,
       const sclLimits&            vpLimits)
{
    return 0;
}

void CONV
sclFreeProgram(sclHandle   hSCL,
              sclProgram* program)
{
}

sclShaderReplaceHandle CONV
sclRegisterShaderString(sclHandle             hSCL,
                       const sclInputShader& src,
                       const sclInputShader& dst)
{
    return 0;
}

void CONV
sclUnregisterShaderString(sclHandle              hSCL,
                         sclShaderReplaceHandle hReplacement)
{
}

bool32 CONV
fsCompile(fsInstrSet     instrSet,
          uint32         instrCount,
          const fsInstr* instr,
          void*&         binary,
          uint32&        length,
          bool32         dumpShader,
          bool32         doCacheOpt,
          const sclCompilerParamTessellation& tessParams)
{
    return ATIGL_TRUE;
}


void CONV
fsFreeBinary(void* binary)
{
}


void CONV
oswsInit(HOSInstance hOSInst)
{
    //
    // do nothing...
    //
}

void CONV
oswsExit()
{
    //
    //  do nothing...
    //
}

