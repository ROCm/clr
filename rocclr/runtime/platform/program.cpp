//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//
#include "top.hpp"
#include "device/appprofile.hpp"
#include "platform/program.hpp"
#include "platform/context.hpp"
#include "utils/options.hpp"
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
    for (it = devProgramsNoOpt_.begin(), itEnd = devProgramsNoOpt_.end();
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
Program::addDeviceProgram(Device& device, const void* image, size_t length, int oclVer)
{
    if (image != NULL &&
        !aclValidateBinaryImage(image, length,
            isIL_?BINARY_TYPE_SPIRV:BINARY_TYPE_ELF|BINARY_TYPE_LLVM)) {
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

    device::Program* program = rootDev.createProgram(oclVer);
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

    program = rootDev.createProgram(oclVer);
    if (program == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }
    devProgramsNoOpt_[&rootDev] = program;

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

inline static int
GetOclCVersion(const char* clVer)
{
    std::string clStd(clVer);

    if (clStd == "CL1.0") {
        return 100;
    }
    else if (clStd == "CL1.1") {
        return 110;
    }
    else if (clStd == "CL1.2") {
        return 120;
    }
    else {
        if (clStd != "CL2.0") {
            LogError("Unsupported OCL C version!");
        }
        return 200;
    }
}

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
    option::Options parsedOptions;
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
    if (optionChangable) {
        if (AMD_OCL_BUILD_OPTIONS != NULL) {
            // Override options.
            cppstr = AMD_OCL_BUILD_OPTIONS;
        }
        if (!Device::appProfile()->GetBuildOptsAppend().empty()) {
          cppstr.append(" ");
          cppstr.append(Device::appProfile()->GetBuildOptsAppend());
        }
        if (AMD_OCL_BUILD_OPTIONS_APPEND != NULL) {
            cppstr.append(" ");
            cppstr.append(AMD_OCL_BUILD_OPTIONS_APPEND);
        }
    }
    if (!option::parseAllOptions(cppstr, parsedOptions)) {
        programLog_ = parsedOptions.optionsLog();
        return CL_INVALID_COMPILER_OPTIONS;
    }
    programLog_ = parsedOptions.optionsLog();

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
            const int oclVer = GetOclCVersion(parsedOptions.oVariables->CLStd);
            retval = addDeviceProgram(**it, bin.first, bin.second, oclVer);
            if (retval != CL_SUCCESS) {
                return retval;
            }
            devProgram = getDeviceProgram(**it);
        }

        if (devProgram->type() == device::Program::TYPE_INTERMEDIATE) {
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
    option::Options parsedOptions;
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
    if (optionChangable) {
        if (AMD_OCL_LINK_OPTIONS != NULL) {
            // Override options.
            cppstr = AMD_OCL_LINK_OPTIONS;
        }
        if (AMD_OCL_LINK_OPTIONS_APPEND != NULL) {
            cppstr.append(" ");
            cppstr.append(AMD_OCL_LINK_OPTIONS_APPEND);
        }
    }
    if (!option::parseLinkOptions(cppstr, parsedOptions)) {
        programLog_ = parsedOptions.optionsLog();
        return CL_INVALID_LINKER_OPTIONS;
    }
    programLog_ = parsedOptions.optionsLog();

    // Link the program programs associated with the given devices.
    std::vector<Device*>::const_iterator it;
    for (it = devices.begin(); it != devices.end(); ++it) {
        // find the corresponding device program in each input program
        std::vector<device::Program*> inputDevPrograms(numInputs);
        bool found = false;
        int maxOclVer = GetOclCVersion(parsedOptions.oVariables->CLStd);
        for (size_t i = 0; i < numInputs; ++i) {
            Program& inputProgram = *inputPrograms[i];
            deviceprograms_t inputDevProgs = inputProgram.devicePrograms();
            deviceprograms_t::const_iterator findIt = inputDevProgs.find(*it);
            if (findIt == inputDevProgs.end()) {
                if (found) break;
                continue;
            }
            found = true;
            inputDevPrograms[i] = findIt->second;
            size_t pos = inputDevPrograms[i]->compileOptions().find("-cl-std=");
            if (pos != std::string::npos) {
                std::string clStd =
                    inputDevPrograms[i]->compileOptions().substr((pos+8), 5);
                int oclVer = GetOclCVersion(clStd.c_str());
                maxOclVer = (maxOclVer > oclVer) ? maxOclVer : oclVer;
            }

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
            retval = addDeviceProgram(**it, bin.first, bin.second, maxOclVer);
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
    option::Options parsedOptions;
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
    if (optionChangable) {
        if (AMD_OCL_BUILD_OPTIONS != NULL) {
            // Override options.
            cppstr = AMD_OCL_BUILD_OPTIONS;
        }
        if (!Device::appProfile()->GetBuildOptsAppend().empty()) {
          cppstr.append(" ");
          cppstr.append(Device::appProfile()->GetBuildOptsAppend());
        }
        if (AMD_OCL_BUILD_OPTIONS_APPEND != NULL) {
            cppstr.append(" ");
            cppstr.append(AMD_OCL_BUILD_OPTIONS_APPEND);
        }
    }
    if (!option::parseAllOptions(cppstr, parsedOptions)) {
        programLog_ = parsedOptions.optionsLog();
        return CL_INVALID_BUILD_OPTIONS;
    }
    programLog_ = parsedOptions.optionsLog();

    // Build the program programs associated with the given devices.
    std::vector<Device*>::const_iterator it;
    for (it = devices.begin(); it != devices.end(); ++it) {
        device::Program* devProgram = getDeviceProgram(**it);
        if (devProgram == NULL) {
            const binary_t& bin = binary(**it);
            const int oclVer = GetOclCVersion(parsedOptions.oVariables->CLStd);
            if (sourceCode_.empty() && (bin.first == NULL)) {
                retval = false;
                continue;
            }
            retval = addDeviceProgram(**it, bin.first, bin.second, oclVer);
            if (retval != CL_SUCCESS) {
                return retval;
            }
            devProgram = getDeviceProgram(**it);
        }

        parsedOptions.oVariables->AssumeAlias = (*it)->settings().assumeAliases_;

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

bool
Program::buildNoOpt(const Device& device, const std::string& kernelName)
{
    ScopedLock sl(buildLock_);
    // Don't allow multiple builds of program without optimizations
    if (!firstBuildNoOpt_) {
        return false;
    }
    firstBuildNoOpt_ = false;

    symbols_t::const_iterator it = symbolTable_->find(kernelName);
    assert((it != symbolTable_->end()) && "Kernel must be valid at this time");
    const Symbol& progSymbol = it->second;

    // Check if program already has unoptimized kernel
    device::Kernel* devKernel = const_cast<device::Kernel*>
        (progSymbol.getDeviceKernel(device, false));
    if (devKernel != NULL) {
        return true;
    }

    // Find the original program for build options string
    deviceprograms_t::const_iterator pit = devicePrograms_.find(&device);
    assert((pit != devicePrograms_.end()) && "Program must be valid at this time");
    device::Program* orgProgram = pit->second;

    // Process build options.
    option::Options parsedOptions;
    std::string cppstr(orgProgram->compileOptions());
    if (AMD_OCL_BUILD_OPTIONS != NULL) {
      // Override options.
      cppstr = AMD_OCL_BUILD_OPTIONS;
    }
    if (!Device::appProfile()->GetBuildOptsAppend().empty()) {
      cppstr.append(" ");
      cppstr.append(Device::appProfile()->GetBuildOptsAppend());
    }
    if (AMD_OCL_BUILD_OPTIONS_APPEND != NULL) {
      cppstr.append(" ");
      cppstr.append(AMD_OCL_BUILD_OPTIONS_APPEND);
    }

    if (!option::parseAllOptions(cppstr, parsedOptions)) {
        return false;
    }
    parsedOptions.optionsLog();

    parsedOptions.oVariables->AssumeAlias = true;
    parsedOptions.oVariables->ForceLLVM = true;

    // Find the program without optimizaiton
    pit = devProgramsNoOpt_.find(&device);

    // Update the symbol table
    if (pit != devProgramsNoOpt_.end()) {
        device::Program& program = *pit->second;
        const device::Program::binary_t& progBinary = orgProgram->binary();

        if (!program.setBinary(reinterpret_cast<char *>(const_cast<void*>
                (progBinary.first)), progBinary.second)) {
            return false;
        }

        // Force recompilation from the binary only
        if (CL_SUCCESS != program.build("", orgProgram->compileOptions().c_str(),
            &parsedOptions)) {
            return false;
        }

        const device::Program::kernels_t& kernels = program.kernels();
        device::Program::kernels_t::const_iterator kit;
        for (kit = kernels.begin(); kit != kernels.end(); ++kit) {
            const std::string& name = kit->first;
            const device::Kernel* devKernel = kit->second;

            symbols_t::iterator sit = symbolTable_->find(name);
            Symbol& symbol = sit->second;
            if (!symbol.setDeviceKernel(device, devKernel, false)) {
                return false;
            }
        }
    }

    return true;
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
    for (sit = devProgramsNoOpt_.begin(); sit != devProgramsNoOpt_.end(); ++sit) {
        // Destroy device program
        delete sit->second;
    }
    devicePrograms_.clear();
    devProgramsNoOpt_.clear();
    deviceList_.clear();
    if (symbolTable_) symbolTable_->clear();
    kernelNames_.clear();
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
