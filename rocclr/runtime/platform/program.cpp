//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "device/appprofile.hpp"
#include "platform/program.hpp"
#include "platform/context.hpp"
#include "utils/options.hpp"
#include "utils/libUtils.h"
#include "utils/bif_section_labels.hpp"
#include "acl.h"

#include <cstdlib> // for malloc
#include <cstring> // for strcmp
#include <utility>

namespace amd {

Program::~Program()
{
    // Destroy all device programs
    deviceprograms_t::const_iterator it, itEnd;
    for (it = devicePrograms_.begin(), itEnd = devicePrograms_.end();
         it != itEnd; ++it) {
        delete it->second;
    }

    for (devicebinary_t::const_iterator IT = binary_.begin(), IE = binary_.end();
         IT != IE;  ++IT) {
        const binary_t& Bin = IT->second;
        if (Bin.first) {
            delete [] Bin.first;
        }
    }

    delete symbolTable_;
    //! @todo Make sure we have destroyed all CPU specific objects
}

const Symbol*
Program::findSymbol(const char* kernelName) const
{
    symbols_t::const_iterator it = symbolTable_->find(kernelName);
    return (it == symbolTable_->end()) ? NULL : &it->second;
}

cl_int
Program::addDeviceProgram(Device& device, const void* image, size_t length,
    amd::option::Options* options)
{
    if (image != NULL &&
        !aclValidateBinaryImage(image, length,
            isSPIRV_?BINARY_TYPE_SPIRV:BINARY_TYPE_ELF|BINARY_TYPE_LLVM)) {
        return CL_INVALID_BINARY;
    }

    // Check if the device is already associated with this program
    if (deviceList_.find(&device) != deviceList_.end()) {
        return CL_INVALID_VALUE;
    }

    Device& rootDev = device.rootDevice();

    // if the rootDev is already associated with a program
    if (devicePrograms_[&rootDev] != NULL) {
        return CL_SUCCESS;
    }
    bool emptyOptions = false;
    amd::option::Options emptyOpts;
    if (options == NULL) {
        options = &emptyOpts;
        emptyOptions = true;
    }
    if (image != NULL && length != 0 && aclValidateBinaryImage(image, length, BINARY_TYPE_ELF)) {
        acl_error errorCode;
        aclBinary *binary = aclReadFromMem(image, length, &errorCode);
        if (errorCode != ACL_SUCCESS) {
            return CL_INVALID_BINARY;
        }
        const oclBIFSymbolStruct* symbol = findBIF30SymStruct(symOpenclCompilerOptions);
        assert(symbol && "symbol not found");
        std::string symName = std::string(symbol->str[bif::PRE]) + std::string(symbol->str[bif::POST]);
        size_t symSize = 0;
        const void *opts = aclExtractSymbol(device.compiler(),
            binary, &symSize, aclCOMMENT, symName.c_str(), &errorCode);
        // if we have options from binary and input options was not specified
        if (opts != NULL && emptyOptions) {
            std::string sBinOptions = std::string((char*)opts, symSize);
            if (!amd::option::parseAllOptions(sBinOptions, *options)) {
                programLog_ = options->optionsLog();
                LogError("Parsing compilation options from binary failed.");
                return CL_INVALID_COMPILER_OPTIONS;
            }
        }
        options->oVariables->Legacy = isAMDILTarget(*aclutGetTargetInfo(binary));
    }
    options->oVariables->BinaryIsSpirv = isSPIRV_;
    device::Program* program = rootDev.createProgram(options);
    if (program == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }

    if (image != NULL) {
        uint8_t* memory = binary(rootDev).first;
        // clone 'binary' (it is owned by the host thread).
        if (memory == NULL) {
            memory = new (std::nothrow) uint8_t[length];
            if (memory == NULL) {
                delete program;
                return CL_OUT_OF_HOST_MEMORY;
            }

            ::memcpy(memory, image, length);

            // Save the original image
            binary_[&rootDev] = std::make_pair(memory, length);
        }

        if (!program->setBinary(reinterpret_cast<char *>(memory), length)) {
            delete program;
            return CL_INVALID_BINARY;
        }
    }

    devicePrograms_[&rootDev] = program;

    deviceList_.insert(&device);
    return CL_SUCCESS;
}

device::Program*
Program::getDeviceProgram(const Device& device) const
{
    deviceprograms_t::const_iterator it =
        devicePrograms_.find(&device.rootDevice());
    if (it == devicePrograms_.end()) {
        return NULL;
    }
    return it->second;
}

Monitor
Program::buildLock_("OCL build program", true);

cl_int
Program::compile(
    const std::vector<Device*>& devices,
    size_t numHeaders,
    const std::vector<const Program*>& headerPrograms,
    const char** headerIncludeNames,
    const char* options,
    void (CL_CALLBACK * notifyFptr)(cl_program, void *),
    void* data,
    bool optionChangable)
{
    ScopedLock sl(buildLock_);

    cl_int  retval = CL_SUCCESS;

    // Clear the program object
    clear();

    // Process build options.
    std::string cppstr(options ? options : "");

    // if there is a -ignore-env,  adjust options.
    if (cppstr.size() > 0) {
        // Set the options to be the string after -ignore-env
        size_t pos = cppstr.find("-ignore-env");
        if (pos != std::string::npos) {
            cppstr = cppstr.substr(pos+sizeof("-ignore-env"));
            optionChangable = false;
        }
    }
    option::Options parsedOptions;
    if (!ParseAllOptions(cppstr, parsedOptions, optionChangable)) {
        programLog_ = parsedOptions.optionsLog();
        LogError("Parsing compile options failed.");
        return CL_INVALID_COMPILER_OPTIONS;
    }

    std::vector<const std::string*> headers(numHeaders);
    for (size_t i = 0; i < numHeaders; ++i) {
        const std::string& header = headerPrograms[i]->sourceCode();
        headers[i] = &header;
    }

    // Compile the program programs associated with the given devices.
    std::vector<Device*>::const_iterator it;
    for (it = devices.begin(); it != devices.end(); ++it) {
        device::Program* devProgram = getDeviceProgram(**it);
        if (devProgram == NULL) {
            const binary_t& bin = binary(**it);
            retval = addDeviceProgram(**it, bin.first, bin.second, &parsedOptions);
            if (retval != CL_SUCCESS) {
                return retval;
            }
            devProgram = getDeviceProgram(**it);
        }

        if (devProgram->type() == device::Program::TYPE_INTERMEDIATE ||
            isSPIRV_) {
           continue;
        }
        // We only build a Device-Program once
        if (devProgram->buildStatus() != CL_BUILD_NONE) {
            continue;
        }
        if (sourceCode_.empty()) {
            return CL_INVALID_OPERATION;
        }
        cl_int result = devProgram->compile(
            sourceCode_, headers,
            headerIncludeNames,
            options,
            &parsedOptions);

        // Check if the previous device failed a build
        if ((result != CL_SUCCESS) && (retval != CL_SUCCESS)) {
            retval = CL_INVALID_OPERATION;
        }
        // Update the returned value with a build error
        else if (result != CL_SUCCESS) {
            retval = result;
        }
    }

    if (notifyFptr != NULL) {
        notifyFptr(as_cl(this), data);
    }

    return retval;
}

cl_int
Program::link(
    const std::vector<Device*>& devices,
    size_t numInputs,
    const std::vector<Program*>& inputPrograms,
    const char* options,
    void (CL_CALLBACK * notifyFptr)(cl_program, void *),
    void* data,
    bool optionChangable)
{
    ScopedLock sl(buildLock_);
    cl_int  retval = CL_SUCCESS;

    if (symbolTable_ == NULL) {
        symbolTable_ = new symbols_t;
        if (symbolTable_ == NULL) {
            return CL_OUT_OF_HOST_MEMORY;
        }
    }

    // Clear the program object
    clear();

    // Process build options.
    std::string cppstr(options ? options : "");

    // if there is a -ignore-env,  adjust options.
    if (cppstr.size() > 0) {
        // Set the options to be the string after -ignore-env
        size_t pos = cppstr.find("-ignore-env");
        if (pos != std::string::npos) {
            cppstr = cppstr.substr(pos+sizeof("-ignore-env"));
            optionChangable = false;
        }
    }
    option::Options parsedOptions;
    if (!ParseAllOptions(cppstr, parsedOptions, optionChangable, true)) {
        programLog_ = parsedOptions.optionsLog();
        LogError("Parsing link options failed.");
        return CL_INVALID_LINKER_OPTIONS;
    }

    // Link the program programs associated with the given devices.
    std::vector<Device*>::const_iterator it;
    for (it = devices.begin(); it != devices.end(); ++it) {
        // find the corresponding device program in each input program
        std::vector<device::Program*> inputDevPrograms(numInputs);
        bool found = false;
        for (size_t i = 0; i < numInputs; ++i) {
            Program& inputProgram = *inputPrograms[i];
            if (inputProgram.isSPIRV_) {
                parsedOptions.oVariables->BinaryIsSpirv = inputProgram.isSPIRV_;
            }
            deviceprograms_t inputDevProgs = inputProgram.devicePrograms();
            deviceprograms_t::const_iterator findIt = inputDevProgs.find(*it);
            if (findIt == inputDevProgs.end()) {
                if (found) break;
                continue;
            }
            inputDevPrograms[i] = findIt->second;
            device::Program::binary_t binary = inputDevPrograms[i]->binary();
            // Check the binary's target for the first found device program.
            // TODO: Revise these binary's target checks
            // and possibly remove them after switching to HSAIL by default.
            if (!found && binary.first != NULL && binary.second > 0) {
                acl_error errorCode = ACL_SUCCESS;
                void *mem = const_cast<void*>(binary.first);
                aclBinary* aclBin = aclReadFromMem(mem, binary.second, &errorCode);
                if (errorCode != ACL_SUCCESS) {
                    LogWarning("Error while linking: Could not read from raw binary.");
                    return CL_INVALID_BINARY;
                }
                if (isHSAILTarget(*aclutGetTargetInfo(aclBin))) {
                    parsedOptions.oVariables->Frontend = "clang";
                } else if (isAMDILTarget(*aclutGetTargetInfo(aclBin))) {
                    parsedOptions.oVariables->Frontend = "edg";
                }
            }
            found = true;
        }
        if (inputDevPrograms.size() == 0) {
            continue;
        }
        if (inputDevPrograms.size() < numInputs) {
            return CL_INVALID_VALUE;
        }

        device::Program* devProgram = getDeviceProgram(**it);
        if (devProgram == NULL) {
            const binary_t& bin = binary(**it);
            retval = addDeviceProgram(**it, bin.first, bin.second, &parsedOptions);
            if (retval != CL_SUCCESS) {
                return retval;
            }
            devProgram = getDeviceProgram(**it);
        }

        // We only build a Device-Program once
        if (devProgram->buildStatus() != CL_BUILD_NONE) {
            continue;
        }
        cl_int result = devProgram->link(
            inputDevPrograms, options, &parsedOptions);

        // Check if the previous device failed a build
        if ((result != CL_SUCCESS) && (retval != CL_SUCCESS)) {
            retval = CL_INVALID_OPERATION;
        }
        // Update the returned value with a build error
        else if (result != CL_SUCCESS) {
            retval = result;
        }
    }

    if (retval != CL_SUCCESS) {
        return retval;
    }

    // Rebuild the symbol table
    deviceprograms_t::iterator sit;
    for (sit = devicePrograms_.begin(); sit != devicePrograms_.end(); ++sit) {
        const Device& device = *sit->first;
        const device::Program& program = *sit->second;

        const device::Program::kernels_t& kernels = program.kernels();
        device::Program::kernels_t::const_iterator kit;
        for (kit = kernels.begin(); kit != kernels.end(); ++kit) {
            const std::string& name = kit->first;
            const device::Kernel* devKernel = kit->second;

            Symbol& symbol = (*symbolTable_)[name];
            if (!symbol.setDeviceKernel(device, devKernel)) {
                retval = CL_LINK_PROGRAM_FAILURE;
            }
        }
    }

    // Create a string with all kernel names from the program
    if (kernelNames_.length() == 0) {
        amd::Program::symbols_t::const_iterator it;
        for (it = symbols().begin(); it != symbols().end(); ++it) {
            if (it != symbols().begin()) {
                kernelNames_.append(1, ';');
            }
            kernelNames_.append(it->first.c_str());
        }
    }

    if (notifyFptr != NULL) {
        notifyFptr(as_cl(this), data);
    }

    return retval;
}

cl_int
Program::build(
    const std::vector<Device*>& devices,
    const char* options,
    void (CL_CALLBACK * notifyFptr)(cl_program, void *),
    void* data,
    bool optionChangable)
{
    ScopedLock sl(buildLock_);
    cl_int  retval = CL_SUCCESS;

    if (symbolTable_ == NULL) {
        symbolTable_ = new symbols_t;
        if (symbolTable_ == NULL) {
             return CL_OUT_OF_HOST_MEMORY;
        }
    }

    // Clear the program object
    clear();

    // Process build options.
    std::string cppstr(options ? options : "");

    // if there is a -ignore-env,  adjust options.
    if (cppstr.size() > 0) {
        // Set the options to be the string after -ignore-env
        size_t pos = cppstr.find("-ignore-env");
        if (pos != std::string::npos) {
            cppstr = cppstr.substr(pos+sizeof("-ignore-env"));
            optionChangable = false;
        }
    }
    option::Options parsedOptions;
    if (!ParseAllOptions(cppstr, parsedOptions, optionChangable)) {
        programLog_ = parsedOptions.optionsLog();
        LogError("Parsing compile options failed.");
        return CL_INVALID_COMPILER_OPTIONS;
    }

    // Build the program programs associated with the given devices.
    std::vector<Device*>::const_iterator it;
    for (it = devices.begin(); it != devices.end(); ++it) {
        device::Program* devProgram = getDeviceProgram(**it);
        if (devProgram == NULL) {
            const binary_t& bin = binary(**it);
            if (sourceCode_.empty() && (bin.first == NULL)) {
                retval = false;
                continue;
            }
            retval = addDeviceProgram(**it, bin.first, bin.second, &parsedOptions);
            if (retval != CL_SUCCESS) {
                return retval;
            }
            devProgram = getDeviceProgram(**it);
        }

        parsedOptions.oVariables->AssumeAlias = true;

        // We only build a Device-Program once
        if (devProgram->buildStatus() != CL_BUILD_NONE) {
            continue;
        }
        cl_int result = devProgram->build(sourceCode_, options, &parsedOptions);

        // Check if the previous device failed a build
        if ((result != CL_SUCCESS) && (retval != CL_SUCCESS)) {
            retval = CL_INVALID_OPERATION;
        }
        // Update the returned value with a build error
        else if (result != CL_SUCCESS) {
            retval = result;
        }
    }

    if (retval != CL_SUCCESS) {
        return retval;
    }

    // Rebuild the symbol table
    deviceprograms_t::iterator sit;
    for (sit = devicePrograms_.begin(); sit != devicePrograms_.end(); ++sit) {
        const Device& device = *sit->first;
        const device::Program& program = *sit->second;

        const device::Program::kernels_t& kernels = program.kernels();
        device::Program::kernels_t::const_iterator kit;
        for (kit = kernels.begin(); kit != kernels.end(); ++kit) {
            const std::string& name = kit->first;
            const device::Kernel* devKernel = kit->second;

            Symbol& symbol = (*symbolTable_)[name];
            if (!symbol.setDeviceKernel(device, devKernel)) {
                retval = CL_BUILD_PROGRAM_FAILURE;
            }
        }
    }

    // Create a string with all kernel names from the program
    if (kernelNames_.length() == 0) {
        amd::Program::symbols_t::const_iterator it;
        for (it = symbols().begin(); it != symbols().end(); ++it) {
            if (it != symbols().begin()) {
                kernelNames_.append(1, ';');
            }
            kernelNames_.append(it->first.c_str());
        }
    }

    if (notifyFptr != NULL) {
        notifyFptr(as_cl(this), data);
    }

    return retval;
}

void
Program::clear()
{
    deviceprograms_t::iterator sit;

    // Destroy old programs if we have any
    for (sit = devicePrograms_.begin(); sit != devicePrograms_.end(); ++sit) {
        // Destroy device program
        delete sit->second;
    }

    devicePrograms_.clear();
    deviceList_.clear();
    if (symbolTable_) symbolTable_->clear();
    kernelNames_.clear();
}

int
Program::GetOclCVersion(const char* clVer) {
    // default version
    int version = 12;
    if (clVer == NULL) {
        return version;
    }
    std::string clStd(clVer);
    if (clStd.size() != 5) {
        return version;
    }
    clStd.erase(0,2);
    clStd.erase(1,1);
    return std::stoi(clStd);
}

bool
Program::ParseAllOptions(const std::string& options, option::Options& parsedOptions, bool optionChangable, bool linkOptsOnly) {
    std::string allOpts = options;
    if (optionChangable) {
        if (linkOptsOnly) {
            if (AMD_OCL_LINK_OPTIONS != NULL) {
                allOpts.append(" ");
                allOpts.append(AMD_OCL_LINK_OPTIONS);
            }
            if (AMD_OCL_LINK_OPTIONS_APPEND != NULL) {
                allOpts.append(" ");
                allOpts.append(AMD_OCL_LINK_OPTIONS_APPEND);
            }
        } else {
            if (AMD_OCL_BUILD_OPTIONS != NULL) {
                allOpts.append(" ");
                allOpts.append(AMD_OCL_BUILD_OPTIONS);
            }
            if (!Device::appProfile()->GetBuildOptsAppend().empty()) {
                allOpts.append(" ");
                allOpts.append(Device::appProfile()->GetBuildOptsAppend());
            }
            if (AMD_OCL_BUILD_OPTIONS_APPEND != NULL) {
                allOpts.append(" ");
                allOpts.append(AMD_OCL_BUILD_OPTIONS_APPEND);
            }
        }
    }
    return amd::option::parseAllOptions(allOpts, parsedOptions, linkOptsOnly);
}

bool
Symbol::setDeviceKernel(
    const Device& device,
    const device::Kernel* func,
    bool    noAlias)
{
    // FIXME_lmoriche: check that the signatures are compatible
    if (deviceKernels_.size() == 0 || device.type() == CL_DEVICE_TYPE_CPU) {
        signature_ = func->signature();
    }

    if (noAlias) {
        deviceKernels_[&device] = func;
    }
    else {
        devKernelsNoOpt_[&device] = func;
    }
    return true;
}

const device::Kernel*
Symbol::getDeviceKernel(const Device& device, bool noAlias) const
{
    const devicekernels_t*    devKernels =
         (noAlias) ? &deviceKernels_ : &devKernelsNoOpt_;
    devicekernels_t::const_iterator itEnd = devKernels->end();
    devicekernels_t::const_iterator it = devKernels->find(&device);
    if (it != itEnd) {
        return it->second;
    }

    for (it = devKernels->begin(); it != itEnd; ++it) {
        if (it->first->isAncestor(&device)) {
            return it->second;
        }
    }

    return NULL;
}

} // namespace amd
