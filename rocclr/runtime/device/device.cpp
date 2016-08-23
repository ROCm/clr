//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/device.hpp"
#include "thread/atomic.hpp"
#include "thread/monitor.hpp"

#if defined(WITH_HSA_DEVICE)
#include "device/rocm/rocdevice.hpp"
extern amd::AppProfile* rocCreateAppProfile();
#endif

#if defined(WITH_CPU_DEVICE)
#include "device/cpu/cpudevice.hpp"
#endif // WITH_CPU_DEVICE

#if defined(WITH_PAL_DEVICE)
//namespace pal {
extern bool PalDeviceLoad();
extern void PalDeviceUnload();
//}
#endif // WITH_PAL_DEVICE

#if defined(WITH_GPU_DEVICE)
extern bool DeviceLoad();
extern void DeviceUnload();
#endif // WITH_GPU_DEVICE

#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "thread/monitor.hpp"
#include "amdocl/cl_common.hpp"
#include "utils/options.hpp"
#include "utils/versions.hpp"    // AMD_PLATFORM_INFO

#if defined(HAVE_BLOWFISH_H)
#include "blowfish/oclcrypt.hpp"
#endif

#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"
#include "spirv/spirvUtils.h"

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <set>
#include <algorithm>



namespace device {
extern const char* BlitSourceCode;
}

namespace amd {

std::vector<Device*> *Device::devices_ = NULL;
bool Device::isHsaDeviceAvailable_ = false;
bool Device::isGpuDeviceAvailable_ = false;
AppProfile Device::appProfile_;

#if defined(WITH_HSA_DEVICE)
AppProfile* Device::rocAppProfile_ = NULL;
#endif

amd::Monitor SvmManager::AllocatedLock_("Guards SVM allocation list");
std::map<uintptr_t, amd::Memory*> SvmManager::svmBufferMap_;

size_t
SvmManager::size()
{
    amd::ScopedLock lock(AllocatedLock_);
    return svmBufferMap_.size();
}

void
SvmManager::AddSvmBuffer(const void* k, amd::Memory* v)
{
    amd::ScopedLock lock(AllocatedLock_);
    svmBufferMap_.insert(std::pair<uintptr_t, amd::Memory*>(reinterpret_cast<uintptr_t>(k), v));
}

void
SvmManager::RemoveSvmBuffer(const void* k)
{
    amd::ScopedLock lock(AllocatedLock_);
    svmBufferMap_.erase(reinterpret_cast<uintptr_t>(k));
}

amd::Memory*
SvmManager::FindSvmBuffer(const void* k)
{
    amd::ScopedLock lock(AllocatedLock_);
    uintptr_t key = reinterpret_cast<uintptr_t>(k);
    std::map<uintptr_t, amd::Memory*>::iterator it = svmBufferMap_.upper_bound(key);
    if (it == svmBufferMap_.begin()) {
        return NULL;
    }

    --it;
    amd::Memory* mem = it->second;
    if (key >= it->first && key < (it->first + mem->getSize())) {
        //the k is in the range
        return mem;
    }
    else {
        return NULL;
    }
}


Device::BlitProgram::~BlitProgram()
{
    if (program_ != NULL) {
        program_->release();
    }
}

bool
Device::BlitProgram::create(amd::Device* device,
    const char* extraKernels, const char* extraOptions)
{
    std::vector<amd::Device*> devices;
    devices.push_back(device);
    std::string kernels(device::BlitSourceCode);

    if (extraKernels != NULL) {
        kernels += extraKernels;
    }

    // Create a program with all blit kernels
    program_ = new Program(*context_, kernels.c_str());
    if (program_ == NULL) {
        return false;
    }

    // Build all kernels
#if defined(WITH_LIGHTNING_COMPILER)
    std::string opt = "";
#else // !defined(WITH_LIGHTNING_COMPILER)
    std::string opt = "-Wf,--force_disable_spir -fno-lib-no-inline "\
        "-fno-sc-keep-calls -cl-internal-kernel ";
#endif // !defined(WITH_LIGHTNING_COMPILER)

    if (extraOptions != NULL) {
        opt += extraOptions;
    }
#if !defined(WITH_LIGHTNING_COMPILER)
    if (!GPU_DUMP_BLIT_KERNELS) {
        opt += " -fno-enable-dump";
    }
#endif // !defined(WITH_LIGHTNING_COMPILER)
    if (CL_SUCCESS != program_->build(devices, opt.c_str(),
        NULL, NULL, GPU_DUMP_BLIT_KERNELS)) {
        return false;
    }

    return true;
}

bool
Device::init()
{
    assert(!Runtime::initialized() && "initialize only once");
    bool ret = false;
    devices_ = NULL;
    appProfile_.init();


// IMPORTANT: Note that we are initialiing HSA stack first and then
// GPU stack. The order of initialization is signiicant and if changed
// amd::Device::registerDevice() must be accordingly modified.
#if defined(WITH_HSA_DEVICE)
    rocAppProfile_ = rocCreateAppProfile();
    if (rocAppProfile_ && !rocAppProfile_->IsHsaInitDisabled()) {
        // Return value of roc::Device::init()
        // If returned false, error initializing HSA stack.
        // If returned true, either HSA not installed or HSA stack
        //                   successfully initialized.
        if (!roc::Device::init() ) {
            // abort() commentted because this is the only indication
            // that KFD is not installed.
            // Ignore the failure and assume KFD is not installed.
            //abort();
        }
        ret |= roc::NullDevice::init();
    }
#endif // WITH_HSA_DEVICE
#if defined(WITH_GPU_DEVICE)
    if (GPU_ENABLE_PAL != 1) {
        ret |= DeviceLoad();
    }
#endif // WITH_GPU_DEVICE
#if defined(WITH_PAL_DEVICE)
    if (GPU_ENABLE_PAL != 0) {
        ret |= PalDeviceLoad();
    }
#endif // WITH_PAL_DEVICE
#if defined(WITH_CPU_DEVICE)
    ret |= cpu::Device::init();
#endif // WITH_CPU_DEVICE
    return ret;
}

void
Device::tearDown()
{
    if (devices_ != NULL) {
        for (uint i = 0; i < devices_->size(); ++i) {
            delete devices_->at(i);
        }
        devices_->clear();
        delete devices_;
    }
#if defined(WITH_HSA_DEVICE)
    if (rocAppProfile_ && !rocAppProfile_->IsHsaInitDisabled()) {
        roc::Device::tearDown();
        delete rocAppProfile_;
        rocAppProfile_ = NULL;
    }
#endif // WITH_HSA_DEVICE
#if defined(WITH_GPU_DEVICE)
    if (GPU_ENABLE_PAL != 1) {
        DeviceUnload();
    }
#endif // WITH_GPU_DEVICE
#if defined(WITH_PAL_DEVICE)
    if (GPU_ENABLE_PAL != 0) {
        PalDeviceUnload();
    }
#endif // WITH_PAL_DEVICE
#if defined(WITH_CPU_DEVICE)
    cpu::Device::tearDown();
#endif // WITH_CPU_DEVICE
}

Device::Device(Device* parent)
    : settings_(NULL)
    , online_(true)
    , blitProgram_(NULL)
    , hwDebugMgr_(NULL)
    , parent_(parent)
    , vaCacheAccess_(nullptr)
    , vaCacheMap_(nullptr)
{
    memset(&info_, '\0', sizeof(info_));
    if (parent_ != NULL) {
        parent_->retain();
    }
}

Device::~Device()
{
    CondLog((vaCacheMap_ != nullptr) &&
        (vaCacheMap_->size() != 0), "Application didn't unmap all host memory!");
    delete vaCacheMap_;
    delete vaCacheAccess_;

    // Destroy device settings
    if (settings_ != NULL) {
        delete settings_;
    }

    if (parent_ != NULL) {
        parent_->release();
    }
    else {
        if (info_.extensions_ != NULL) {
            delete [] info_.extensions_;
        }
    }

    if (info_.partitionCreateInfo_.type_.byCounts_ &&
        info_.partitionCreateInfo_.byCounts_.countsList_ != NULL) {
        delete [] info_.partitionCreateInfo_.byCounts_.countsList_;
    }
}

bool
Device::create()
{
    vaCacheAccess_ = new amd::Monitor("VA Cache Ops Lock", true);
    if (NULL == vaCacheAccess_) {
        return false;
    }
    vaCacheMap_ = new std::map<uintptr_t, device::Memory*>();
    if (NULL == vaCacheMap_) {
        return false;
    }
    return true;
}

bool
Device::isAncestor(const Device* sub) const
{
    for (const Device* d = sub->parent_; d != NULL; d = d->parent_) {
        if (d == this) {
            return true;
        }
    }
    return false;
}

bool Device::IsHsaCapableDevice() {
    static bool init = false;
    typedef std::vector<std::string> ListType;
    typedef std::vector<std::string>::const_iterator  ListIterType;
    static ListType hsaNames;

    if (!init) {
        hsaNames.push_back("Spectre");
        hsaNames.push_back("Spooky");
        init = true;
    }

    for (ListIterType itr = hsaNames.begin(); itr != hsaNames.end(); itr++) {
        if ((*itr) == info_.name_  // If name is in list then HSA capable.
        || (info_.type_ & CL_HSA_ENABLED_AMD)) { // HSA enabled is HSA capable.
            return true;
        }
    }

    return false;
}


void
Device::registerDevice()
{
    assert(Runtime::singleThreaded() && "this is not thread-safe");

    static bool nonhsaDefaultIsAssigned = false;
    static bool hsaDefaultIsAssigned = false;

    if (devices_ == NULL) {
        devices_ = new std::vector<Device*>;
    }

    if (info_.available_) {
        bool dvHsaEnabled = (info_.type_ & CL_HSA_ENABLED_AMD) != 0;
        if (IsHsaCapableDevice()) {
            if (dvHsaEnabled && !hsaDefaultIsAssigned) {
                hsaDefaultIsAssigned = true;
                info_.type_ |= CL_DEVICE_TYPE_DEFAULT;
                isHsaDeviceAvailable_ = true;
            }
        }
        if (!dvHsaEnabled && !nonhsaDefaultIsAssigned) {
            nonhsaDefaultIsAssigned = true;
            info_.type_ |= CL_DEVICE_TYPE_DEFAULT;
            isGpuDeviceAvailable_ = true;
        }
    }
    devices_->push_back(this);
}

void
Device::addVACache(device::Memory* memory) const
{
    // Make sure system memory has direct access
    if (memory->isHostMemDirectAccess()) {
        // VA cache access must be serialised
        amd::ScopedLock lk(*vaCacheAccess_);
        void*   start = memory->owner()->getHostMem();
        size_t  offset;
        device::Memory*   doubleMap = findMemoryFromVA(start, &offset);

        if (doubleMap == nullptr) {
            // Insert the new entry
            vaCacheMap_->insert(std::pair<uintptr_t, device::Memory*>
                (reinterpret_cast<uintptr_t>(start), memory));
        }
        else {
            LogError("Unexpected double map() call from the app!");
        }
    }
}

void
Device::removeVACache(const device::Memory* memory) const
{
    // Make sure system memory has direct access
    if (memory->isHostMemDirectAccess() && memory->owner()) {
        // VA cache access must be serialised
        amd::ScopedLock lk(*vaCacheAccess_);
        void*   start = memory->owner()->getHostMem();
        vaCacheMap_->erase(reinterpret_cast<uintptr_t>(start));
    }
}

device::Memory*
Device::findMemoryFromVA(const void* ptr, size_t* offset) const
{
    // VA cache access must be serialised
    amd::ScopedLock lk(*vaCacheAccess_);

    uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
    std::map<uintptr_t, device::Memory*>::iterator it = vaCacheMap_->upper_bound(
        reinterpret_cast<uintptr_t>(ptr));
    if (it == vaCacheMap_->begin()) {
        return nullptr;
    }

    --it;
    device::Memory* mem = it->second;
    if (key >= it->first && key < (it->first + mem->size())) {
        // ptr is in the range
        *offset = key - it->first;
        return mem;
    }
    return nullptr;
}

bool IsHsaRequested(cl_device_type requestedType) {
// Depending on HSA_RUNTIME and hint flags CL_HSA_XXXXX_AMD,
// decide to use Gpu or HSA device.

    if (requestedType == CL_DEVICE_TYPE_ALL) {
        // hint flags are masked by ALL (0xFFFFFFFF) so HSA_RUNTIME decides.
        return HSA_RUNTIME;
    }

    bool hsabit = (requestedType & CL_HSA_ENABLED_AMD) != 0;
    bool nonhsabit = (requestedType & CL_HSA_DISABLED_AMD) != 0;

    // Condition where both hsabit and nonhsabit set
    // is checked in getDevices() and
    // numDevices(). So that condition will never occur in this code path
    if (hsabit ^ nonhsabit) {
        // Eithere ENABLE or DISABLE HSA specified via hint flags.
        return (requestedType & CL_HSA_ENABLED_AMD) != 0;
    } else {
        // Niethre ENABLE nor DISABLE HSA specified via hint flags.
        return HSA_RUNTIME;
    }
}

bool Device::IsTypeMatching(cl_device_type type, bool offlineDevices) {
    if (!(isOnline() || offlineDevices)) {
        return false;
    }

    cl_device_type hintAndDefaultMask = (CL_HSA_ENABLED_AMD
                        | CL_HSA_DISABLED_AMD | CL_DEVICE_TYPE_DEFAULT);
    if ((info_.type_ & type & ~hintAndDefaultMask)
    || (info_.type_ & type & CL_DEVICE_TYPE_DEFAULT)) {

        // HSA and ORCA stack will never be running side-by-side.
        // Hence device selection is mute.
        // Instead of removing the code just returning true always.
        // To reinstate, simply remove this "return true;" statement.
        return true;

        bool dvHsaEnabled = (info_.type_ & CL_HSA_ENABLED_AMD) != 0;
        bool isHsaReq = IsHsaRequested(type);

        // If not HSA capable device then always a match.
        if (!IsHsaCapableDevice()) {
            return true;
        }

        // ASSUMPTION: In the following two if statements,
        // assumption is, either HSA or GPU stacks always available
        // for HSA capable devcie.

        // Requested stack not available - ignore hint.
        if (isHsaReq && !isHsaDeviceAvailable_) {
            return isGpuDeviceAvailable_;
        }
        // Requested stack not available - ignore hint.
        if (!isHsaReq && !isGpuDeviceAvailable_) {
            return isHsaDeviceAvailable_;
        }

        // If HSA capable device then request type(hsa or not) must
        // match device hsa enablement(HSA enabled or not).
        if (!(dvHsaEnabled ^ isHsaReq)) {
            return true;
        }
    }
    return false;
}

std::vector<Device*>
Device::getDevices(cl_device_type type, bool offlineDevices)
{
    std::vector<Device*> result;

    if (devices_ == NULL) {
        return result;
    }

#if defined(WITH_HSA_DEVICE)
    type = rocAppProfile_->ApplyHsaDeviceHintFlag(type);
#endif

    // Create the list of available devices
    for (device_iterator it = devices_->begin(); it != devices_->end(); ++it) {
        // Check if the device type is matched
        if ((*it)->IsTypeMatching(type, offlineDevices)) {
          result.push_back(*it);
        }
    }

    return result;
}

size_t
Device::numDevices(cl_device_type type, bool offlineDevices)
{
    size_t result = 0;

    if (devices_ == NULL) {
        return 0;
    }

#if defined(WITH_HSA_DEVICE)
    type = rocAppProfile_->ApplyHsaDeviceHintFlag(type);
#endif

    for (device_iterator it = devices_->begin(); it != devices_->end(); ++it) {
        // Check if the device type is matched
        if ((*it)->IsTypeMatching(type, offlineDevices)) {
            ++result;
        }
    }

    return result;
}

bool
Device::getDeviceIDs(
    cl_device_type  deviceType,
    cl_uint         numEntries,
    cl_device_id*   devices,
    cl_uint*        numDevices,
    bool            offlineDevices)
{
    if (numDevices != NULL && devices == NULL) {
        *numDevices =
            (cl_uint)amd::Device::numDevices(deviceType, offlineDevices);
        return (*numDevices > 0) ? true : false;
    }
    assert(devices != NULL && "check the code above");

    std::vector<amd::Device*> ret =
        amd::Device::getDevices(deviceType, offlineDevices);
    if (ret.size() == 0) {
        *not_null(numDevices) = 0;
        return false;
    }

    std::vector<amd::Device*>::iterator it = ret.begin();
    cl_uint count = std::min(numEntries, (cl_uint)ret.size());

    while (count--) {
        *devices++ = as_cl(*it++);
        --numEntries;
    }
    while (numEntries--) {
        *devices++ = (cl_device_id) 0;
    }

    *not_null(numDevices) = (cl_uint)ret.size();
    return true;
}

char*
Device::getExtensionString()
{
    std::stringstream   extStream;
    size_t  size;
    char*   result = NULL;

    // Generate the extension string
    for (uint i = 0; i < ClExtTotal; ++i) {
        if (settings().checkExtension(i)) {
            extStream << OclExtensionsString[i];
        }
    }

    size = extStream.str().size() + 1;

    // Create a single string with all extensions
    result = new char[size];
    if (result != NULL) {
        memcpy(result, extStream.str().data(), (size - 1));
        result[size - 1] = 0;
    }

    return result;
}

void*
Device::allocMapTarget(
    amd::Memory&    mem,
    const amd::Coord3D& origin,
    const amd::Coord3D& region,
    uint    mapFlags,
    size_t* rowPitch,
    size_t* slicePitch)
{
    // Translate memory references
    device::Memory* devMem = mem.getDeviceMemory(*this);
    if (devMem == NULL) {
        LogError("allocMapTarget failed. Can't allocate video memory");
        return NULL;
    }

    // Pass request over to memory
    return devMem->allocMapTarget(origin, region, mapFlags, rowPitch, slicePitch);
}

} // namespace amd

namespace device {

Settings::Settings()
{
    assert((ClExtTotal < (8 * sizeof(extensions_))) && "Too many extensions!");
    extensions_          = 0;
    partialDispatch_     = false;
    supportRA_           = true;
    customHostAllocator_ = false;
    waitCommand_         = AMD_OCL_WAIT_COMMAND;
    supportDepthsRGB_    = false;
    enableHwDebug_       = false;
}

bool
Kernel::createSignature(const parameters_t& params)
{
    std::stringstream attribs;
    if (workGroupInfo_.compileSize_[0] != 0) {
        attribs << "reqd_work_group_size(";
        for (size_t i = 0; i < 3; ++i) {
            if (i != 0) {
                attribs << ",";
            }

            attribs << workGroupInfo_.compileSize_[i];
        }
        attribs << ")";
    }
    if (workGroupInfo_.compileSizeHint_[0] != 0) {
        attribs << " work_group_size_hint(";
        for (size_t i = 0; i < 3; ++i) {
            if (i != 0) {
                attribs << ",";
            }

            attribs << workGroupInfo_.compileSizeHint_[i];
        }
        attribs << ")";
    }

    if (!workGroupInfo_.compileVecTypeHint_.empty()) {
      attribs << " vec_type_hint("
              << workGroupInfo_.compileVecTypeHint_
              << ")";
    }

    // Destroy old signature if it was allocated before
    // (offline devices path)
    delete signature_;
    signature_ = new amd::KernelSignature(params, attribs.str());
    if (NULL != signature_) {
        return true;
    }
    return false;
}

Kernel::~Kernel()
{
    delete signature_;
}

std::string
Kernel::openclMangledName(const std::string& name)
{
    const oclBIFSymbolStruct* bifSym = findBIF30SymStruct(symOpenclKernel);
    assert(bifSym && "symbol not found");
    return std::string("&") + bifSym->str[bif::PRE] + name + bifSym->str[bif::POST];
}

void
Memory::saveMapInfo(
    const void*         mapAddress,
    const amd::Coord3D  origin,
    const amd::Coord3D  region,
    uint                mapFlags,
    bool                entire,
    amd::Image*         baseMip)
{
    // Map/Unmap must be serialized.
    amd::ScopedLock lock(owner()->lockMemoryOps());

    WriteMapInfo info = {};
    WriteMapInfo* pInfo = &info;
    auto it = writeMapInfo_.find(mapAddress);
    if (it != writeMapInfo_.end()) {
        LogWarning("Double map of the same or overlapped region!");
        pInfo = &it->second;
    }

    if (mapFlags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)) {
        pInfo->origin_ = origin;
        pInfo->region_ = region;
        pInfo->entire_ = entire;
        pInfo->unmapWrite_ = true;
    }
    if (mapFlags & CL_MAP_READ) {
        pInfo->unmapRead_ = true;
    }
    pInfo->baseMip_ = baseMip;

    // Insert into the map if it's the first region
    if (++pInfo->count_ == 1) {
        writeMapInfo_.insert(std::pair<const void*, WriteMapInfo>(mapAddress, info));
    }

}

Program::Program(amd::Device& device)
    : device_(device)
    , type_(TYPE_NONE)
    , clBinary_(NULL)
    , llvmBinary_()
    , elfSectionType_(amd::OclElf::LLVMIR)
    , compileOptions_()
    , linkOptions_()
    , lastBuildOptionsArg_()
    , buildStatus_(CL_BUILD_NONE)
    , buildError_(CL_SUCCESS)
    , globalVariableTotalSize_(0)
    , programOptions(NULL)
{ }

Program::~Program()
{
   clear();
}

void
Program::clear()
{
    // Destroy all device kernels
    kernels_t::const_iterator it;
    for (it = kernels_.begin(); it != kernels_.end(); ++it) {
        delete it->second;
    }
    kernels_.clear();
}

bool
Program::initBuild(amd::option::Options* options)
{
    programOptions = options;

    if (options->oVariables->DumpFlags > 0) {
      static amd::Atomic<unsigned> build_num = 0;
      options->setBuildNo(build_num++);
    }
    buildLog_.clear();
    if (!initClBinary()) {
        return false;
    }
    return true;
}

bool
Program::finiBuild(bool isBuildGood)
{
   return true;
}

cl_int
Program::compile(const std::string& sourceCode,
                 const std::vector<const std::string*>& headers,
                 const char** headerIncludeNames,
                 const char* origOptions,
                 amd::option::Options* options)
{
    uint64_t start_time = 0;
    if (options->oVariables->EnableBuildTiming) {
        buildLog_ = "\nStart timing major build components.....\n\n";
        start_time = amd::Os::timeNanos();
    }

    lastBuildOptionsArg_ = origOptions ? origOptions : "";
    if (options) {
        compileOptions_ = options->origOptionStr;
    }

    buildStatus_ = CL_BUILD_IN_PROGRESS;
    if (!initBuild(options)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ = "Internal error: Compilation init failed.";
        }
    }

    if (options->oVariables->FP32RoundDivideSqrt &&
        !(device().info().singleFPConfig_
          & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
        buildStatus_ = CL_BUILD_ERROR;
        buildLog_ += "Error: -cl-fp32-correctly-rounded-divide-sqrt "\
            "specified without device support";
    }

    // Compile the source code if any
    if ((buildStatus_ == CL_BUILD_IN_PROGRESS) &&
        !sourceCode.empty() &&
        !compileImpl(sourceCode, headers, headerIncludeNames, options)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ = "Internal error: Compilation failed.";
        }
    }

    setType(TYPE_COMPILED);

    if ((buildStatus_ == CL_BUILD_IN_PROGRESS) &&
        !createBinary(options)) {
        buildLog_ += "Internal Error: creating OpenCL binary failed!\n";
    }

    if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ = "Internal error: Compilation fini failed.";
        }
    }

    if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
        buildStatus_ = CL_BUILD_SUCCESS;
    }
    else {
        buildError_  = CL_COMPILE_PROGRAM_FAILURE;
    }

    if (options->oVariables->EnableBuildTiming) {
        std::stringstream tmp_ss;
        tmp_ss << "\nTotal Compile Time: "
               << (amd::Os::timeNanos() - start_time)/1000ULL
               << " us\n";
        buildLog_ += tmp_ss.str();
    }

    if (options->oVariables->BuildLog && !buildLog_.empty()) {
        if (strcmp(options->oVariables->BuildLog, "stderr") == 0) {
            fprintf(stderr, "%s\n", options->optionsLog().c_str());
            fprintf(stderr, "%s\n", buildLog_.c_str());
        }
        else if (strcmp(options->oVariables->BuildLog, "stdout") == 0) {
            printf("%s\n", options->optionsLog().c_str());
            printf("%s\n", buildLog_.c_str());
        }
        else {
            std::fstream f;
            std::stringstream tmp_ss;
            std::string logs = options->optionsLog() + buildLog_;
            tmp_ss << options->oVariables->BuildLog
                   << "." << options->getBuildNo();
            f.open(tmp_ss.str().c_str(),
                   (std::fstream::out | std::fstream::binary));
            f.write(logs.data(), logs.size());
            f.close();
        }
        LogError(buildLog_.c_str());
    }

    return buildError();
}

cl_int Program::link(const std::vector<Program*>& inputPrograms,
                     const char* origLinkOptions,
                     amd::option::Options* linkOptions)
{
    lastBuildOptionsArg_ = origLinkOptions ? origLinkOptions : "";
    if (linkOptions) {
        linkOptions_ = linkOptions->origOptionStr;
    }

    buildStatus_ = CL_BUILD_IN_PROGRESS;

    amd::option::Options options;
    if (!getCompileOptionsAtLinking(inputPrograms, linkOptions)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ += "Internal error: Get compile options failed.";
        }
    }
    else {
        if (!amd::option::parseAllOptions(compileOptions_, options)) {
            buildStatus_ = CL_BUILD_ERROR;
            buildLog_ += options.optionsLog();
            LogError("Parsing compile options failed.");
        }
    }

    uint64_t start_time = 0;
    if (options.oVariables->EnableBuildTiming) {
        buildLog_ = "\nStart timing major build components.....\n\n";
        start_time = amd::Os::timeNanos();
    }

    // initBuild() will clear buildLog_, so store it in a temporary variable
    std::string tmpBuildLog = buildLog_;

    if ((buildStatus_ == CL_BUILD_IN_PROGRESS)
        && !initBuild(&options)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ += "Internal error: Compilation init failed.";
        }
    }

    buildLog_ += tmpBuildLog;

    if (options.oVariables->FP32RoundDivideSqrt &&
        !(device().info().singleFPConfig_
          & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
        buildStatus_ = CL_BUILD_ERROR;
        buildLog_ += "Error: -cl-fp32-correctly-rounded-divide-sqrt "\
            "specified without device support";
    }

    bool createLibrary
      = linkOptions ? linkOptions->oVariables->clCreateLibrary : false;
    if ((buildStatus_ == CL_BUILD_IN_PROGRESS)
        && !linkImpl(inputPrograms, &options, createLibrary)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ += "Internal error: Link failed.\n";
            buildLog_ += "Make sure the system setup is correct.";
        }
    }

    if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ = "Internal error: Compilation fini failed.";
        }
    }

    if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
        buildStatus_ = CL_BUILD_SUCCESS;
    }
    else {
        buildError_  = CL_LINK_PROGRAM_FAILURE;
    }

    if (options.oVariables->EnableBuildTiming) {
        std::stringstream tmp_ss;
        tmp_ss << "\nTotal Link Time: "
               << (amd::Os::timeNanos() - start_time)/1000ULL
               << " us\n";
        buildLog_ += tmp_ss.str();
    }

    if (options.oVariables->BuildLog && !buildLog_.empty()) {
        if (strcmp(options.oVariables->BuildLog, "stderr") == 0) {
            fprintf(stderr, "%s\n", options.optionsLog().c_str());
            fprintf(stderr, "%s\n", buildLog_.c_str());
        }
        else if (strcmp(options.oVariables->BuildLog, "stdout") == 0) {
            printf("%s\n", options.optionsLog().c_str());
            printf("%s\n", buildLog_.c_str());
        }
        else {
            std::fstream f;
            std::stringstream tmp_ss;
            std::string logs = options.optionsLog() + buildLog_;
            tmp_ss << options.oVariables->BuildLog
                   << "." << options.getBuildNo();
            f.open(tmp_ss.str().c_str(),
                   (std::fstream::out | std::fstream::binary));
            f.write(logs.data(), logs.size());
            f.close();
        }
    }

    if (!buildLog_.empty()) {
        LogError(buildLog_.c_str());
    }

    return buildError();
}

cl_int
Program::build(const std::string& sourceCode,
               const char* origOptions,
               amd::option::Options* options)
{
    uint64_t start_time = 0;
    if (options->oVariables->EnableBuildTiming) {
        buildLog_ = "\nStart timing major build components.....\n\n";
        start_time = amd::Os::timeNanos();
    }

    lastBuildOptionsArg_ = origOptions ? origOptions : "";
    if (options) {
        compileOptions_ = options->origOptionStr;
    }

    buildStatus_ = CL_BUILD_IN_PROGRESS;
    if (!initBuild(options)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ = "Internal error: Compilation init failed.";
        }
    }

    if (options->oVariables->FP32RoundDivideSqrt &&
        !(device().info().singleFPConfig_ & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)) {
        buildStatus_ = CL_BUILD_ERROR;
        buildLog_ += "Error: -cl-fp32-correctly-rounded-divide-sqrt "\
            "specified without device support";
    }

    // Compile the source code if any
    std::vector<const std::string*> headers;
    if ((buildStatus_ == CL_BUILD_IN_PROGRESS) &&
        !sourceCode.empty() && !compileImpl(sourceCode, headers, NULL, options)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ = "Internal error: Compilation failed.";
        }
    }

    if ((buildStatus_ == CL_BUILD_IN_PROGRESS) && !linkImpl(options)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ += "Internal error: Link failed.\n";
            buildLog_ += "Make sure the system setup is correct.";
        }
    }

    if (!finiBuild(buildStatus_ == CL_BUILD_IN_PROGRESS)) {
        buildStatus_ = CL_BUILD_ERROR;
        if (buildLog_.empty()) {
            buildLog_ = "Internal error: Compilation fini failed.";
        }
    }

    if (buildStatus_ == CL_BUILD_IN_PROGRESS) {
        buildStatus_ = CL_BUILD_SUCCESS;
    }
    else {
        buildError_  = CL_BUILD_PROGRAM_FAILURE;
    }

    if (options->oVariables->EnableBuildTiming) {
        std::stringstream tmp_ss;
        tmp_ss << "\nTotal Build Time: "
               << (amd::Os::timeNanos() - start_time)/1000ULL
               << " us\n";
        buildLog_ += tmp_ss.str();
    }

    if (options->oVariables->BuildLog && !buildLog_.empty()) {
        if (strcmp(options->oVariables->BuildLog, "stderr") == 0) {
            fprintf(stderr, "%s\n", options->optionsLog().c_str());
            fprintf(stderr, "%s\n", buildLog_.c_str());
        }
        else if (strcmp(options->oVariables->BuildLog, "stdout") == 0) {
            printf("%s\n", options->optionsLog().c_str());
            printf("%s\n", buildLog_.c_str());
        }
        else {
            std::fstream f;
            std::stringstream tmp_ss;
            std::string logs = options->optionsLog() + buildLog_;
            tmp_ss << options->oVariables->BuildLog << "." << options->getBuildNo();
            f.open(tmp_ss.str().c_str(), (std::fstream::out | std::fstream::binary));
            f.write(logs.data(), logs.size());
            f.close();
        }
    }

    if (!buildLog_.empty()) {
        LogError(buildLog_.c_str());
    }

    return buildError();
}

bool
Program::getCompileOptionsAtLinking(const std::vector<Program*>& inputPrograms,
                                    const amd::option::Options* linkOptions)
{
    amd::option::Options compileOptions;
    std::vector<device::Program*>::const_iterator it
        = inputPrograms.begin();
    std::vector<device::Program*>::const_iterator itEnd
        = inputPrograms.end();
    for (size_t i = 0; it != itEnd; ++it, ++i) {
        Program* program = *it;

        amd::option::Options compileOptions2;
        amd::option::Options* thisCompileOptions
            = i == 0 ? &compileOptions : &compileOptions2;
        if (!amd::option::parseAllOptions(program->compileOptions_,
                                          *thisCompileOptions)) {
            buildLog_ += thisCompileOptions->optionsLog();
            LogError("Parsing compile options failed.");
            return false;
        }

        if (i == 0)
            compileOptions_ = program->compileOptions_;

        // if we are linking a program executable, and if "program" is a
        // compiled module or a library created with "-enable-link-options",
        // we can overwrite "program"'s compile options with linking options
        if (!linkOptions_.empty()
            && !linkOptions->oVariables->clCreateLibrary) {
            bool linkOptsCanOverwrite = false;
            if (program->type() != TYPE_LIBRARY) {
                linkOptsCanOverwrite = true;
            }
            else {
                amd::option::Options thisLinkOptions;
                if (!amd::option::parseLinkOptions(program->linkOptions_,
                                                   thisLinkOptions)) {
                    buildLog_ += thisLinkOptions.optionsLog();
                    LogError("Parsing link options failed.");
                    return false;
                }
                if (thisLinkOptions.oVariables->clEnableLinkOptions)
                    linkOptsCanOverwrite = true;
            }
            if (linkOptsCanOverwrite) {
                if (!thisCompileOptions->setOptionVariablesAs(*linkOptions)) {
                    buildLog_ += thisCompileOptions->optionsLog();
                    LogError("Setting link options failed.");
                    return false;
                }
            }
            if (i == 0)
                compileOptions_ += " " + linkOptions_;
        }
        // warn if input modules have inconsistent compile options
        if (i > 0) {
            if (!compileOptions.equals(*thisCompileOptions,
                                       true/*ignore clc options*/)) {
                buildLog_ += "Warning: Input OpenCL binaries has inconsistent"
                             " compile options. Using compile options from"
                             " the first input binary!\n";
            }
        }
    }
    return true;
}

bool
Program::initClBinary(char* binaryIn, size_t size)
{
    if (!initClBinary()) {
        return false;
    }

    // Save the original binary that isn't owned by ClBinary
    clBinary()->saveOrigBinary(binaryIn, size);

    char* bin = binaryIn;
    size_t sz = size;

    //unencrypted
    int encryptCode = 0;
    char* decryptedBin = NULL;

#if !defined(WITH_LIGHTNING_COMPILER)
    bool isSPIRV = isSPIRVMagic(binaryIn, size);
    if (isSPIRV || isBcMagic(binaryIn))
    {
        acl_error err = ACL_SUCCESS;
        aclBinaryOptions binOpts = {0};
        binOpts.struct_size = sizeof(binOpts);
        binOpts.elfclass
            = (info().arch_id == aclX64 || info().arch_id == aclAMDIL64 ||
               info().arch_id == aclHSAIL64)
              ? ELFCLASS64 : ELFCLASS32;
        binOpts.bitness = ELFDATA2LSB;
        binOpts.alloc = &::malloc;
        binOpts.dealloc = &::free;
        aclBinary* aclbin_v30 = aclBinaryInit(sizeof(aclBinary), &info(), &binOpts, &err);
        if (err != ACL_SUCCESS) {
            LogWarning("aclBinaryInit failed");
            aclBinaryFini(aclbin_v30);
            return false;
        }
        err = aclInsertSection(device().compiler(), aclbin_v30, binaryIn, size,
                isSPIRV?aclSPIRV:aclSPIR);
        if (ACL_SUCCESS != err) {
            LogWarning("aclInsertSection failed");
            aclBinaryFini(aclbin_v30);
            return false;
        }
        if (info().arch_id == aclHSAIL || info().arch_id == aclHSAIL64) {
            err = aclWriteToMem(aclbin_v30, reinterpret_cast<void**>(&bin), &sz);
            if (err != ACL_SUCCESS) {
                LogWarning("aclWriteToMem failed");
                aclBinaryFini(aclbin_v30);
                return false;
            }
            aclBinaryFini(aclbin_v30);
        }
        else {
            aclBinary* aclbin_v21 = aclCreateFromBinary(aclbin_v30,aclBIFVersion21);
            err = aclWriteToMem(aclbin_v21, reinterpret_cast<void**>(&bin), &sz);
            if (err != ACL_SUCCESS) {
                LogWarning("aclWriteToMem failed");
                aclBinaryFini(aclbin_v30);
                aclBinaryFini(aclbin_v21);
                return false;
            }
            aclBinaryFini(aclbin_v30);
            aclBinaryFini(aclbin_v21);
        }
    }
    else
#endif // defined(WITH_LIGHTNING_COMPILER)
    {
        size_t decryptedSize;
        if (!clBinary()->decryptElf(binaryIn,size,
                &decryptedBin,&decryptedSize,&encryptCode)) {
            return false;
        }
        if (decryptedBin != NULL) {
            // It is decrypted binary.
            bin = decryptedBin;
            sz  = decryptedSize;
        }

        if (!isElf(bin)) {
            // Invalid binary.
            if (decryptedBin != NULL) {
                delete [] decryptedBin;
            }
            return false;
        }
    }

    clBinary()->setFlags(encryptCode);

    return clBinary()->setBinary(bin, sz, (decryptedBin != NULL));
}


bool
Program::setBinary(char* binaryIn, size_t size)
{
    if (!initClBinary(binaryIn, size)) {
        return false;
    }

#if defined(WITH_LIGHTNING_COMPILER)
    if (!clBinary()->setElfIn(ELFCLASS64)) {
#else // !defined(WITH_LIGHTNING_COMPILER)
    if (!clBinary()->setElfIn(ELFCLASS32)) {
#endif // !defined(WITH_LIGHTNING_COMPILER)
        LogError("Setting input OCL binary failed");
        return false;
    }
    uint16_t type;
    if (!clBinary()->elfIn()->getType(type)) {
        LogError("Bad OCL Binary: error loading ELF type!");
        return false;
    }
    switch (type) {
        case ET_NONE:
        {
            setType(TYPE_NONE);
            break;
        }
        case ET_REL:
        {
            if (clBinary()->isSPIR() || clBinary()->isSPIRV()) {
                setType(TYPE_INTERMEDIATE);
            } else {
                setType(TYPE_COMPILED);
            }
            break;
        }
        case ET_DYN:
        {
            setType(TYPE_LIBRARY);
            break;
        }
        case ET_EXEC:
        {
            setType(TYPE_EXECUTABLE);
            break;
        }
        default:
            LogError("Bad OCL Binary: bad ELF type!");
            return false;
    }

    clBinary()->loadCompileOptions(compileOptions_);
    clBinary()->loadLinkOptions(linkOptions_);
    clBinary()->resetElfIn();
    return true;
}

bool
Program::createBIFBinary(aclBinary* bin)
{
#if defined(WITH_LIGHTNING_COMPILER)
    assert(!"FIXME_Wilkin");
    return false;
#else // defined(WITH_LIGHTNING_COMPILER)
    acl_error err;
    char *binaryIn = NULL;
    size_t size;
    err = aclWriteToMem(bin, reinterpret_cast<void**>(&binaryIn), &size);
    if (err != ACL_SUCCESS) {
        LogWarning("aclWriteToMem failed");
        return false;
    }
    clBinary()->saveBIFBinary(binaryIn, size);
    aclFreeMem(bin, binaryIn);
    return true;
#endif // defined(WITH_LIGHTNING_COMPILER)
}

ClBinary::ClBinary(const amd::Device& dev, BinaryImageFormat bifVer)
    : dev_(dev)
    , binary_(NULL)
    , size_(0)
    , flags_(0)
    , origBinary_(NULL)
    , origSize_(0)
    , encryptCode_ (0)
    , elfIn_(NULL)
    , elfOut_(NULL)
    , format_(bifVer)
{
}

ClBinary::~ClBinary()
{
    release();

    if (elfIn_) {
        delete elfIn_;
    }
    if (elfOut_) {
        delete elfOut_;
    }
}

std::string
ClBinary::getBIFSymbol(unsigned int symbolID) const
{
    size_t nSymbols = 0;
    // Due to PRE & POST defines in bif_section_labels.hpp conflict with
    // PRE & POST struct members in sp3-si-chip-registers.h
    // unable to include bif_section_labels.hpp in device.hpp
    //! @todo: resolve conflict by renaming defines,
    // then include bif_section_labels.hpp in device.hpp &
    // use oclBIFSymbolID instead of unsigned int as a parameter
    const oclBIFSymbolID symID = static_cast<oclBIFSymbolID>(symbolID);
    switch (format_) {
    case BIF_VERSION2: {
        nSymbols = sizeof(BIF20)/sizeof(oclBIFSymbolStruct);
        const oclBIFSymbolStruct* symb = findBIFSymbolStruct(BIF20, nSymbols, symID);
        assert(symb && "BIF20 symbol with symbolID not found");
        if (symb) {
            return std::string(symb->str[bif::PRE]) + std::string(symb->str[bif::POST]);
        }
        break;
    }
    case BIF_VERSION3: {
        nSymbols = sizeof(BIF30)/sizeof(oclBIFSymbolStruct);
        const oclBIFSymbolStruct* symb = findBIFSymbolStruct(BIF30, nSymbols, symID);
        assert(symb && "BIF30 symbol with symbolID not found");
        if (symb) {
            return std::string(symb->str[bif::PRE]) + std::string(symb->str[bif::POST]);
        }
        break;
    }
    default:
        assert(0 && "unexpected BIF type");
        return "";
    }
    return "";
}

void
ClBinary::init(amd::option::Options* optionsObj, bool amdilRequired)
{
    // option has higher priority than environment variable.
    if ((flags_ & BinarySourceMask) != BinaryRemoveSource) {
        // set to zero
        flags_    = (flags_ & (~BinarySourceMask));

        flags_ |= (optionsObj->oVariables->BinSOURCE
            ? BinarySaveSource : BinaryNoSaveSource);
    }

    if ((flags_ & BinaryLlvmirMask) != BinaryRemoveLlvmir) {
        // set to zero
        flags_ = (flags_ & (~BinaryLlvmirMask));

        flags_ |= (optionsObj->oVariables->BinLLVMIR
            ? BinarySaveLlvmir : BinaryNoSaveLlvmir);
    }

    // If amdilRequired is true, force to save AMDIL (for correctness)
    if ((flags_ & BinaryAmdilMask) != BinaryRemoveAmdil ||
         amdilRequired) {
        // set to zero
        flags_ = (flags_ & (~BinaryAmdilMask));
        flags_ |= ((optionsObj->oVariables->BinAMDIL || amdilRequired)
            ? BinarySaveAmdil : BinaryNoSaveAmdil);
    }

    if ((flags_ & BinaryIsaMask) != BinaryRemoveIsa) {
        // set to zero
        flags_ = (flags_ & (~BinaryIsaMask));
        flags_ |= ((optionsObj->oVariables->BinEXE)
            ? BinarySaveIsa : BinaryNoSaveIsa);
    }

    if ((flags_ & BinaryASMask) != BinaryRemoveAS) {
        // set to zero
        flags_ = (flags_ & (~BinaryASMask));
        flags_ |= ((optionsObj->oVariables->BinAS)
            ? BinarySaveAS : BinaryNoSaveAS);
    }
}

bool
ClBinary::isRecompilable(std::string& llvmBinary,
                         amd::OclElf::oclElfPlatform thePlatform)
{
    /* It is recompilable if there is llvmir that was generated for
       the same platform (CPU or GPU) and with the same bitness.

       Note: the bitness has been checked in initClBinary(), no need
             to check it here.
     */
    if (llvmBinary.empty() ) {
        return false;
    }

    uint16_t elf_target;
    amd::OclElf::oclElfPlatform platform;
    if (elfIn()->getTarget(elf_target, platform)){
        if (platform == thePlatform){
            return true;
        }
        if ((platform == amd::OclElf::COMPLIB_PLATFORM) &&
            (((thePlatform == amd::OclElf::CAL_PLATFORM) &&
              ((elf_target == (uint16_t)EM_AMDIL) ||
               (elf_target == (uint16_t)EM_HSAIL) ||
               (elf_target == (uint16_t)EM_HSAIL_64))) ||
             ((thePlatform == amd::OclElf::CPU_PLATFORM) &&
              ((elf_target == (uint16_t)EM_386) ||
               (elf_target == (uint16_t)EM_X86_64))))){
            return true;
        }
    }

    return false;
}

void
ClBinary::release()
{
    if (isBinaryAllocated() && (binary_ != NULL)) {
        delete [] binary_;
        binary_ = NULL;
        flags_ &= ~BinaryAllocated;
    }
}

void
ClBinary::saveBIFBinary(char* binaryIn, size_t size)
{
    char *image = new char[size];
    memcpy(image, binaryIn, size);

    setBinary(image, size, true);
    return;
}

bool
ClBinary::createElfBinary(bool doencrypt, Program::type_t type)
{
#if 0
        if (!saveISA() && !saveAMDIL() && !saveLLVMIR() && !saveSOURCE()) {
            return true;
        }
#endif
    release();

    size_t imageSize;
    char*  image;
    assert (elfOut_ && "elfOut_ should be initialized in ClBinary::data()");

    // Insert Version string that builds this binary into .comment section
    const device::Info&  devInfo = dev_.info();
    std::string buildVerInfo("@(#) ");
    if (devInfo.version_ != NULL) {
        buildVerInfo.append(devInfo.version_);
        buildVerInfo.append(".  Driver version: ");
        buildVerInfo.append(devInfo.driverVersion_);
    }
    else {
        // char OpenCLVersion[256];
        // size_t sz;
        // cl_int ret= clGetPlatformInfo(AMD_PLATFORM, CL_PLATFORM_VERSION, 256, OpenCLVersion, &sz);
        // if (ret == CL_SUCCESS) {
        //     buildVerInfo.append(OpenCLVersion, sz);
        // }

        // If CAL is unavailable, just hard-code the OpenCL driver version
        buildVerInfo.append("OpenCL 1.1" AMD_PLATFORM_INFO);
    }

    elfOut_->addSection(amd::OclElf::COMMENT, buildVerInfo.data(), buildVerInfo.size());
    switch (type) {
        case Program::TYPE_NONE:
        {
            elfOut_->setType(ET_NONE);
            break;
        }
        case Program::TYPE_COMPILED:
        {
            elfOut_->setType(ET_REL);
            break;
        }
        case Program::TYPE_LIBRARY:
        {
            elfOut_->setType(ET_DYN);
            break;
        }
        case Program::TYPE_EXECUTABLE:
        {
            elfOut_->setType(ET_EXEC);
            break;
        }
        default:
            assert(0 && "unexpected elf type");
    }

    if (!elfOut_->dumpImage(&image, &imageSize)) {
        return false;
    }

#if defined(HAVE_BLOWFISH_H)
    if (doencrypt) {
        // Increase the size by 64 to accomodate extra headers
        int outBufSize = (int)(imageSize + 64);
        char * outBuf = new char[outBufSize];
        if (outBuf == NULL) {
            return false;
        }
        memset(outBuf, '\0', outBufSize);

        int outBytes = 0;
        bool success = amd::oclEncrypt(0, image, imageSize, outBuf, outBufSize, &outBytes);
        delete [] image;
        if (!success) {
			delete [] outBuf;
            return false;
        }
        image = outBuf;
        imageSize = outBytes;
    }
#endif

    setBinary(image, imageSize, true);
    return true;
}

Program::binary_t ClBinary::data() const
{
    return std::make_pair(binary_, size_);
}

bool
ClBinary::setBinary(char* theBinary, size_t theBinarySize, bool allocated)
{
    release();

    size_   = theBinarySize;
    binary_ = theBinary;
    if (allocated) {
        flags_ |= BinaryAllocated;
    }
    return true;
}

void
ClBinary::setFlags(int encryptCode)
{
    encryptCode_ = encryptCode;
    if (encryptCode != 0) {
        flags_ = (flags_ & (~(BinarySourceMask | BinaryLlvmirMask |
                              BinaryAmdilMask  | BinaryIsaMask |
                              BinaryASMask)));
        flags_ |= (BinaryRemoveSource | BinaryRemoveLlvmir |
                   BinaryRemoveAmdil  | BinarySaveIsa |
                   BinaryRemoveAS);
    }
}

bool
ClBinary::decryptElf(char* binaryIn, size_t size,
                     char** decryptBin, size_t* decryptSize, int* encryptCode)
{
    *decryptBin = NULL;
#if defined(HAVE_BLOWFISH_H)
	int outBufSize = 0;
    if (amd::isEncryptedBIF(binaryIn, (int)size, &outBufSize)) {
        char* outBuf = new (std::nothrow) char[outBufSize];
        if (outBuf == NULL) {
            return false;
        }

        // Decrypt
		int outDataSize = 0;
        if (!amd::oclDecrypt(binaryIn, (int)size, outBuf, outBufSize, &outDataSize)) {
            delete [] outBuf;
            return false;
        }

        *decryptBin = reinterpret_cast<char*>(outBuf);
        *decryptSize = outDataSize;
		*encryptCode = 1;
    }
#endif
    return true;
}

bool
ClBinary::setElfIn(unsigned char eclass)
{
    if (elfIn_) return true;

    if (binary_ == NULL) {
       return false;
    }
    elfIn_ = new amd::OclElf(eclass, binary_, size_, NULL, ELF_C_READ);
    if ( (elfIn_ == NULL)|| elfIn_->hasError() ) {
        if (elfIn_) {
            delete elfIn_;
            elfIn_ = NULL;
        }
        LogError("Creating input ELF object failed");
        return false;
    }

    return true;
}

void ClBinary::resetElfIn()
{
    if (elfIn_) {
        delete elfIn_;
        elfIn_ = NULL;
    }
}

bool ClBinary::setElfOut(unsigned char eclass, const char* outFile)
{
    elfOut_ = new amd::OclElf(eclass, NULL, 0, outFile, ELF_C_WRITE);
    if ( (elfOut_ == NULL) || elfOut_->hasError() )  {
        if (elfOut_) {
            delete elfOut_;
            elfOut_ = NULL;
        }
        LogError("Creating ouput ELF object failed");
        return false;
    }

    return setElfTarget();
}

void ClBinary::resetElfOut()
{
    if (elfOut_) {
        delete elfOut_;
        elfOut_ = NULL;
    }
}

bool
ClBinary::loadLlvmBinary(std::string& llvmBinary, amd::OclElf::oclElfSections& elfSectionType) const
{
    // Check if current binary already has LLVMIR
    char *section = NULL;
    size_t sz = 0;
    const amd::OclElf::oclElfSections SectionTypes[] =
            {amd::OclElf::LLVMIR, amd::OclElf::SPIR, amd::OclElf::SPIRV};

    for (int i = 0; i < 3; ++i){
      if (elfIn_->getSection(SectionTypes[i], &section, &sz) && section && sz > 0) {
          llvmBinary.append(section, sz);
          elfSectionType = SectionTypes[i];
          return true;
      }
    }

    return false;
}

bool ClBinary::loadCompileOptions(std::string& compileOptions) const
{
    char *options = NULL;
    size_t sz;
    compileOptions.clear();
    if (elfIn_->getSymbol(amd::OclElf::COMMENT,
        getBIFSymbol(symOpenclCompilerOptions).c_str(), &options, &sz)) {
        if (sz > 0) {
            compileOptions.append(options, sz);
        }
        return true;
    }
    return false;
}

bool ClBinary::loadLinkOptions(std::string& linkOptions) const
{
    char *options = NULL;
    size_t sz;
    linkOptions.clear();
    if (elfIn_->getSymbol(amd::OclElf::COMMENT,
        getBIFSymbol(symOpenclLinkerOptions).c_str(), &options, &sz)) {
        if (sz > 0) {
            linkOptions.append(options, sz);
        }
        return true;
    }
    return false;
}

void ClBinary::storeCompileOptions(const std::string& compileOptions)
{
    elfOut()->addSymbol(amd::OclElf::COMMENT,
        getBIFSymbol(symOpenclCompilerOptions).c_str(),
        compileOptions.c_str(), compileOptions.length());
}

void ClBinary::storeLinkOptions(const std::string& linkOptions)
{
    elfOut()->addSymbol(amd::OclElf::COMMENT,
        getBIFSymbol(symOpenclLinkerOptions).c_str(),
        linkOptions.c_str(), linkOptions.length());
}

bool
ClBinary::isSPIR() const
{
    char *section = NULL;
    size_t sz = 0;
    if (elfIn_->getSection(amd::OclElf::LLVMIR, &section, &sz) && section && sz > 0)
        return false;

    if (elfIn_->getSection(amd::OclElf::SPIR, &section, &sz) && section && sz > 0)
        return true;

    return false;
}

bool
ClBinary::isSPIRV() const
{
    char *section = NULL;
    size_t sz = 0;

    if (elfIn_->getSection(amd::OclElf::SPIRV, &section, &sz) && section && sz > 0) {
        return true;
    }
    return false;
}

cl_device_partition_property
PartitionType::toCL() const
{
    static cl_device_partition_property conv[] = {
        CL_DEVICE_PARTITION_EQUALLY,
        CL_DEVICE_PARTITION_BY_COUNTS,
        CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN
    };
    return conv[amd::leastBitSet(value_)];
}

size_t
PartitionType::toCL(cl_device_partition_property* types) const
{
    size_t i = 0;
    if (equally_) {
        types[i++] = CL_DEVICE_PARTITION_EQUALLY;
    }
    if (byCounts_) {
        types[i++] = CL_DEVICE_PARTITION_BY_COUNTS;
    }
    if (byAffinityDomain_) {
        types[i++] = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
    }
    return i;
}

cl_device_affinity_domain
AffinityDomain::toCL() const
{
    return (cl_device_affinity_domain)value_;
}

#ifdef cl_ext_device_fission

cl_device_partition_property_ext
PartitionType::toCLExt() const
{
    static cl_device_partition_property_ext conv[] = {
        CL_DEVICE_PARTITION_EQUALLY_EXT,
        CL_DEVICE_PARTITION_BY_COUNTS_EXT,
        CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT
    };
    return conv[amd::leastBitSet(value_)];
}

size_t
PartitionType::toCLExt(cl_device_partition_property_ext* types) const
{
    size_t i = 0;
    if (equally_) {
        types[i++] = CL_DEVICE_PARTITION_EQUALLY_EXT;
    }
    if (byCounts_) {
        types[i++] = CL_DEVICE_PARTITION_BY_COUNTS_EXT;
    }
    if (byAffinityDomain_) {
        types[i++] = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT;
    }
    return i;
}

cl_device_partition_property_ext
AffinityDomain::toCLExt() const
{
    static cl_device_partition_property_ext conv[] = {
        CL_AFFINITY_DOMAIN_NUMA_EXT,
        CL_AFFINITY_DOMAIN_L4_CACHE_EXT,
        CL_AFFINITY_DOMAIN_L3_CACHE_EXT,
        CL_AFFINITY_DOMAIN_L2_CACHE_EXT,
        CL_AFFINITY_DOMAIN_L1_CACHE_EXT,
        CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT
    };
    return conv[amd::leastBitSet(value_)];
}

size_t
AffinityDomain::toCLExt(cl_device_partition_property_ext* affinities) const
{
    size_t i = 0;
    if (numa_) {
        affinities[i++] = CL_AFFINITY_DOMAIN_NUMA_EXT;
    }
    if (cacheL4_) {
        affinities[i++] = CL_AFFINITY_DOMAIN_L4_CACHE_EXT;
    }
    if (cacheL3_) {
        affinities[i++] = CL_AFFINITY_DOMAIN_L3_CACHE_EXT;
    }
    if (cacheL2_) {
        affinities[i++] = CL_AFFINITY_DOMAIN_L2_CACHE_EXT;
    }
    if (cacheL1_) {
        affinities[i++] = CL_AFFINITY_DOMAIN_L1_CACHE_EXT;
    }
    if (next_) {
        affinities[i++] = CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT;
    }
    return i;
}

#endif // cl_ext_device_fission

} // namespace device
