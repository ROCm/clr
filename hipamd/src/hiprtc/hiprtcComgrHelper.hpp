/*
Copyright (c) 2022 - Present Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <vector>
#include <string>

#include "vdi_common.hpp"
#include "utils/debug.hpp"
#include "device/comgrctx.hpp"

namespace hiprtc {
namespace helpers {
bool addCodeObjData(amd_comgr_data_set_t& input, const std::vector<char>& source,
                    const std::string& name, const amd_comgr_data_kind_t type);
bool extractBuildLog(amd_comgr_data_set_t dataSet, std::string& buildLog);
bool extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
                           const amd_comgr_data_kind_t dataKind, std::vector<char>& bin);
bool createAction(amd_comgr_action_info_t& action, std::vector<std::string>& options,
                  const std::string& isa,
                  const amd_comgr_language_t lang = AMD_COMGR_LANGUAGE_NONE);
bool compileToBitCode(const amd_comgr_data_set_t compileInputs, const std::string& isa,
                      std::vector<std::string>& compileOptions, std::string& buildLog,
                      std::vector<char>& LLVMBitcode);
bool linkLLVMBitcode(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                     std::vector<std::string>& linkOptions, std::string& buildLog,
                     std::vector<char>& LinkedLLVMBitcode);
bool createExecutable(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                      std::vector<std::string>& exeOptions, std::string& buildLog,
                      std::vector<char>& executable);
bool dumpIsaFromBC(const amd_comgr_data_set_t isaInputs, const std::string& isa,
                   std::vector<std::string>& exeOptions, std::string name, std::string& buildLog);
bool demangleName(const std::string& mangledName, std::string& demangledName);
std::string handleMangledName(std::string loweredName);
bool fillMangledNames(std::vector<char>& executable, std::vector<std::string>& mangledNames);
bool getDemangledNames(const std::vector<std::string>& mangledNames,
                     std::map<std::string, std::string>& demangledNames);
}  // namespace helpers
}  // namespace hiprtc
