//
// Copyright 2011 Advanced Micro Devices, Inc. All rights reserved.
//

#include "device/cpu/cpudevice.hpp"
#include "device/cpu/cpuprogram.hpp"
#include "utils/versions.hpp"

#include "amdocl/cl_common.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#if defined(__linux__)
#if !defined(ATI_ARCH_ARM)
#include <sys/sysinfo.h>
#endif // ATI_ARCH_ARM
#include <unistd.h>
#endif

#if defined(_WIN32)
# include <windows.h>
# include <intrin.h>

extern BOOL (WINAPI *pfnGetNumaNodeProcessorMaskEx)(USHORT,PGROUP_AFFINITY);
#endif // _WIN32

namespace cpu {

aclCompiler* Device::compiler_;

size_t Device::maxWorkerThreads_ = (size_t)-1;

Device::~Device()
{
#if defined(__linux__) && defined(NUMA_SUPPORT)
    if (getNumaMask() != NULL) {
        if (numaMask_ != NULL) {
            delete numaMask_;
        }
    }
    else
#endif
    if (workerThreadsAffinity_ != NULL) {
        delete workerThreadsAffinity_;
    }
}
void
Device::tearDown()
{
  aclCompilerFini(compiler_);
}
bool
Device::init()
{
    // Allow disabling of the CPU device
    if (CPU_MAX_COMPUTE_UNITS == 0)
        return false;

    const char *library = getenv("COMPILER_LIBRARY");
    aclCompilerOptions opts = {
        sizeof(aclCompilerOptions_0_8),
        library,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        &::malloc,
        &::free
    };
    compiler_ = aclCompilerInit(NULL, NULL);

    device::Info info;
    ::memset(&info, '\0', sizeof(info));

    info.type_ = CL_DEVICE_TYPE_CPU;
    info.vendorId_ = 0x1002;

    int systemProcessorCount = amd::Os::processorCount();
    info.maxComputeUnits_ = systemProcessorCount;
    if (!flagIsDefault(CPU_MAX_COMPUTE_UNITS)) {
        if ((CPU_MAX_COMPUTE_UNITS <= 0) || (CPU_MAX_COMPUTE_UNITS > systemProcessorCount))
            info.maxComputeUnits_ = systemProcessorCount;
        else
            info.maxComputeUnits_ = CPU_MAX_COMPUTE_UNITS;
    }

    info.maxWorkItemDimensions_ = 3;
    info.maxWorkGroupSize_ = CPU_MAX_WORKGROUP_SIZE;
    info.maxWorkItemSizes_[0] = info.maxWorkGroupSize_;
    info.maxWorkItemSizes_[1] = info.maxWorkGroupSize_;
    info.maxWorkItemSizes_[2] = info.maxWorkGroupSize_;

    info.addressBits_ = LP64_SWITCH(32,64);


    if (CPU_IMAGE_SUPPORT) {
        info.imageSupport_      = CL_TRUE;
        info.maxReadImageArgs_  = MaxReadImage;
        info.maxWriteImageArgs_ = MaxWriteImage;
        info.image2DMaxWidth_   = 8 * Ki;
        info.image2DMaxHeight_  = 8 * Ki;
        info.image3DMaxWidth_   = 2 * Ki;
        info.image3DMaxHeight_  = 2 * Ki;
        info.image3DMaxDepth_   = 2 * Ki;
        info.maxSamplers_       = MaxSamplers;

        // OpenCL 1.2 device info fields
        info.imageMaxBufferSize_ = 64 * Ki;
        info.imageMaxArraySize_  = 2 * Ki;

        info.imagePitchAlignment_       = 0;
        info.imageBaseAddressAlignment_ = 0;
        info.bufferFromImageSupport_    = CL_FALSE;
    }

    info.maxParameterSize_ = 4*Ki;

    info.memBaseAddrAlign_ = 8 * (flagIsDefault(MEMOBJ_BASE_ADDR_ALIGN) ?
        sizeof(cl_long16) : MEMOBJ_BASE_ADDR_ALIGN);
    info.minDataTypeAlignSize_ = sizeof(cl_long16);

    info.singleFPConfig_ =
        CL_FP_DENORM | CL_FP_INF_NAN |
        CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO |
        CL_FP_ROUND_TO_INF | CL_FP_FMA;

    info.doubleFPConfig_ = info.singleFPConfig_;
    info.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;

    info.affinityDomain_.value_ = 0;
    info.affinityDomain_.next_ = 1;

    info.globalMemCacheType_ = CL_READ_WRITE_CACHE;

#if defined(__linux__)

    info.globalMemCacheLineSize_ = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    info.globalMemCacheSize_ = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    info.affinityDomain_.cacheL1_ = 1;

    if (sysconf(_SC_LEVEL2_CACHE_SIZE) > 0) {
        info.affinityDomain_.cacheL2_ = 1;
    }
    if (sysconf(_SC_LEVEL3_CACHE_SIZE) > 0) {
        info.affinityDomain_.cacheL3_ = 1;
    }
    if (sysconf(_SC_LEVEL4_CACHE_SIZE) > 0) {
        info.affinityDomain_.cacheL4_ = 1;
    }

#if defined(NUMA_SUPPORT)
    if (numa_available() != -1 && numa_max_node() => 0) {
        info.affinityDomain_.numa_ = 1;
    }
#endif

#else // win32

    DWORD length = 0;
    ::GetLogicalProcessorInformation(NULL, &length);

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer =
        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION) malloc(length);

    if (buffer != NULL && ::GetLogicalProcessorInformation(buffer, &length)) {
        bool found = false;
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr, limit =
            &buffer[length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)];
        for (ptr = buffer; ptr < limit; ++ptr) {
            PCACHE_DESCRIPTOR cache = &ptr->Cache;
            if (ptr->Relationship == RelationCache && cache->Type != CacheInstruction) {
                info.affinityDomain_.value_ |= 
                    (device::AffinityDomain::AFFINITY_DOMAIN_L1_CACHE << 1) >>
                    cache->Level;

                if (!found && cache->Level == 1) {
                    info.globalMemCacheLineSize_ = cache->LineSize;
                    info.globalMemCacheSize_ = cache->Size;
                    found = true;
                }
            }
        }
    }

    free(buffer);

    ULONG highestNuma = 0;
    if (::GetNumaHighestNodeNumber(&highestNuma) && highestNuma != 0) {
        info.affinityDomain_.numa_ = 1;
    }

#endif

    uintptr_t virtualMemSize;

#if defined(__linux__)
#if !defined(ATI_ARCH_ARM)
    struct sysinfo si;

    if (sysinfo(&si) != 0) {
        return false;
    }
    if (si.mem_unit == 0) {
        // Linux kernels prior to 2.3.23 return sizes in bytes.
        si.mem_unit = 1;
    }
    info.globalMemSize_ = (cl_ulong) si.totalram * si.mem_unit;
#else
    info.globalMemSize_ = 0;
#endif
    virtualMemSize = (uintptr_t) info.globalMemSize_;
#else
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof (statex);

    if (GlobalMemoryStatusEx (&statex) == 0) {
        return false;
    }
    info.globalMemSize_ = (cl_ulong) statex.ullTotalPhys;
    virtualMemSize =
        (uintptr_t) std::min(statex.ullTotalPageFile, statex.ullTotalVirtual);
#endif

    maxWorkerThreads_ = (size_t) (virtualMemSize / 
        (uintptr_t) ((CPU_WORKER_THREAD_STACK_SIZE + 
        CLK_PRIVATE_MEMORY_SIZE * (CPU_MAX_WORKGROUP_SIZE + 1))) * 
        7 / 10);

#if defined(_LP64)
    // Cap at 8TiB for 64-bit
    const cl_ulong maxGlobalMemSize = 8ULL*Ki*Gi;
#elif defined(_WIN32)
    // Cap at 2GiB (see http://msdn.microsoft.com/en-us/library/aa366778.aspx)
    const cl_ulong maxGlobalMemSize = 2ULL*Gi;
#else // linux
    // Cap at 3.5GiB
    const cl_ulong maxGlobalMemSize = 3584ULL*Mi;
#endif
    info.globalMemSize_ = std::min(info.globalMemSize_, maxGlobalMemSize);

    info.maxMemAllocSize_ = info.globalMemSize_ * CPU_MAX_ALLOC_PERCENT / 100;
    if (flagIsDefault(CPU_MAX_ALLOC_PERCENT)) {
        const cl_ulong minAllocSize = LP64_SWITCH(1ULL*Gi, 2ULL*Gi);
        info.maxMemAllocSize_ = std::max(info.maxMemAllocSize_,
            std::min(info.globalMemSize_, minAllocSize));
    }

    info.maxConstantBufferSize_ = 64*Ki;
    info.maxConstantArgs_ = 8;

    info.localMemType_ = CL_GLOBAL;
    info.localMemSize_ = std::max((cl_ulong)32*Ki, info.globalMemCacheSize_/2);

    info.errorCorrectionSupport_ = CL_FALSE;
    info.hostUnifiedMemory_ = CL_TRUE;
    info.profilingTimerResolution_ = (size_t)amd::Os::timerResolutionNanos();
    info.profilingTimerOffset_ = amd::Os::offsetToEpochNanos();
    info.littleEndian_ = CL_TRUE;
    info.available_ = CL_TRUE;
    info.compilerAvailable_ = CL_TRUE;
    info.linkerAvailable_ = CL_TRUE;

    info.executionCapabilities_ = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
    info.svmCapabilities_ = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER |
            CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
            CL_DEVICE_SVM_FINE_GRAIN_SYSTEM |
            CL_DEVICE_SVM_ATOMICS;
    info.preferredPlatformAtomicAlignment_ = 0;
    info.preferredGlobalAtomicAlignment_ = 0;
    info.preferredLocalAtomicAlignment_ = 0;
    info.queueProperties_ = CL_QUEUE_PROFILING_ENABLE;

    info.platform_ = AMD_PLATFORM;

#if defined(__linux__)

    std::ifstream ifs("/proc/cpuinfo", std::ios::in);
    if (ifs.is_open()) {
        std::string line;
        bool vendor = false;
        bool name = false;
        bool freq = false;

        while (std::getline(ifs, line) && !(vendor && name && freq)) {
            if (!vendor && (line.find("vendor_id\t: ")
                            != std::string::npos)) {
                ::strcpy(
                    info.vendor_,
                    line.substr(line.find_first_of(':') + 2).c_str());
                vendor = true;
            }
            else if (!name && (line.find("model name\t: ") != std::string::npos
                     || line.find("Processor\t: ") != std::string::npos)) {
                ::strcpy(
                    info.name_,
                    line.substr(line.find_first_of(':') + 2).c_str());
                name = true;
            }
            else if (!freq && (line.find("cpu MHz\t\t: ")
                            != std::string::npos)) {
                info.maxClockFrequency_ =
                    ::atoi(line.substr(line.find_first_of(':') + 2).c_str());
                freq = true;
            }
        }
        ifs.close();
    }

#elif defined(_WIN32)

    int CPUInfo[4] = {-1};
    int nRet = 0;
    unsigned    nIds, nExIds, i;

    // cpuid with an InfoType argument of 0 returns the number of
    // valid Ids in CPUInfo[0] and the CPU identification string in
    // the other three array elements. The CPU identification string is
    // not in linear order. The code below arranges the information
    // in a human readable form.
    amd::Os::cpuid(CPUInfo, 0);
    nIds = CPUInfo[0];
    memset(info.vendor_, 0, sizeof(info.vendor_));
    *((int*)(info.vendor_+0)) = CPUInfo[1];
    *((int*)(info.vendor_+4)) = CPUInfo[3];
    *((int*)(info.vendor_+8)) = CPUInfo[2];

    // Calling cpuid with 0x80000000 as the InfoType argument
    // gets the number of valid extended IDs.
    amd::Os::cpuid(CPUInfo, 0x80000000);
    nExIds = CPUInfo[0];
    memset(info.name_, 0, sizeof(info.name_));
    sprintf(info.name_, "Unknown Processor");

    // Get the information associated with each extended ID.
    for (i=0x80000000; i<=nExIds; ++i)
    {
        amd::Os::cpuid(CPUInfo, i);
        // Interpret CPU brand string and cache information.
        if  (i == 0x80000002)
            memcpy(info.name_, CPUInfo, sizeof(CPUInfo));
        else if  (i == 0x80000003)
            memcpy(info.name_ + 16, CPUInfo, sizeof(CPUInfo));
        else if  (i == 0x80000004)
            memcpy(info.name_ + 32, CPUInfo, sizeof(CPUInfo));
    }


    info.maxClockFrequency_ = 0;
    HKEY hKey;

    // Open the key
    if (RegOpenKeyEx(
            HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\\",
            0, KEY_QUERY_VALUE, &hKey) == ERROR_SUCCESS) {

        // Read the value
        DWORD dwLen = 4;
        RegQueryValueEx(
            hKey, "~MHz", NULL, NULL,
            (LPBYTE)&info.maxClockFrequency_, &dwLen);

        // Cleanup and return
        RegCloseKey(hKey);
    }

#else
    ::strcpy(info.name_, "Unknown Processor");
    ::strcpy(info.vendor_, "Unknown Vendor");
    info.maxClockFrequency_ = 0;
#endif

#define OPENCL_VERSION_STR XSTR(OPENCL_MAJOR) "." XSTR(OPENCL_MINOR)

    info.profile_ = "FULL_PROFILE";
    info.version_ = "OpenCL " OPENCL_VERSION_STR " " AMD_PLATFORM_INFO;
    info.oclcVersion_ = "OpenCL C " OPENCL_VERSION_STR " ";
    info.spirVersions_ = "1.2";

#if cl_amd_open_video
    info.openVideo_ = CL_FALSE;
#endif // cl_amd_open_video

    info.partitionCreateInfo_.type_.value_ = 0;
    info.partitionProperties_.value_ = 0;
    if (info.maxComputeUnits_ > 1) {
        info.partitionProperties_.equally_  = 1;
        info.partitionProperties_.byCounts_ = 1;
        if (info.affinityDomain_.value_ != 0) {
            info.partitionProperties_.byAffinityDomain_ = 1;
        }
    }
    else {
        info.affinityDomain_.value_ = 0;
    }

    // Copy the name into the boardName data member for CPU implementation.
//   ::strncpy(info.boardName_, info.name_, sizeof(info.boardName_));
    memset(info.boardName_, 0, sizeof(info.boardName_));

    Device* device = new Device();

    if (device == NULL || !device->create()) {
        delete device;
        return false;
    }

    ::snprintf(info.driverVersion_, sizeof(info.driverVersion_) - 1,
        "%s (%s%s%s)", AMD_BUILD_STRING,
#if defined(ATI_ARCH_X86)
        "sse2",
#else // !ATI_ARCH_X86
        "",
#endif // !ATI_ARCH_X86
        device->hasAVXInstructions() ? ",avx" : "",
        device->hasFMA4Instructions() ? ",fma4" : "");

    // These will need to change for AVX2
    info.preferredVectorWidthChar_ = 16;
    info.preferredVectorWidthShort_ = 8;
    info.preferredVectorWidthInt_ = 4;
    info.preferredVectorWidthLong_ = 2;
    if (device->hasAVXInstructions()) {
        info.preferredVectorWidthFloat_ = 8;
        info.preferredVectorWidthDouble_ = 4;
    } else {
        info.preferredVectorWidthFloat_ = 4;
        info.preferredVectorWidthDouble_ = 2;
    }
    info.preferredVectorWidthHalf_ = 0; // no half support

    // Same here, will need to change for AVX2
    info.nativeVectorWidthChar_ = 16;
    info.nativeVectorWidthShort_ = 8;
    info.nativeVectorWidthInt_ = 4;
    info.nativeVectorWidthLong_ = 2;
    if (device->hasAVXInstructions()) {
        info.nativeVectorWidthFloat_ = 8;
        info.nativeVectorWidthDouble_ = 4;
    } else {
        info.nativeVectorWidthFloat_ = 4;
        info.nativeVectorWidthDouble_ = 2;
    }
    info.nativeVectorWidthHalf_ = 0; // no half support

    // Find all supported device extensions
    info.extensions_ = device->getExtensionString();

    // OpenCL 1.2 device info fields
    info.builtInKernels_ = "";
    info.preferredInteropUserSync_ = true;
    info.printfBufferSize_ = 64*Ki;

    info.maxPipePacketSize_ = info.maxMemAllocSize_;
    info.maxPipeActiveReservations_ = 16;
    info.maxPipeArgs_ = 16;
    info.maxReadWriteImageArgs_ = MaxReadWriteImage;

    // Max size should not be bigger than 1.75 GB
    const cl_ulong maxSize = std::min(static_cast<cl_ulong>((Gi/4)*7),
                                      info.maxMemAllocSize_);
    info.maxGlobalVariableSize_ = static_cast<size_t>(maxSize);
    info.globalVariablePreferredTotalSize_ = static_cast<size_t>(maxSize);

    device->info_ = info;
    device->registerDevice();

    return true;
}

bool
Device::create()
{
    // Create CPU settings
    settings_ = new cpu::Settings();
    cpu::Settings* cpuSettings = reinterpret_cast<cpu::Settings*>(settings_);

    if ((cpuSettings == NULL) || !cpuSettings->create()) {
        return false;
    }

#if defined(ATI_ARCH_X86)
    // Check that we have at least SSE2
    if (settings().cpuFeatures_ == 0) {
        return false;
    }
#endif

    return true;
}

bool
Device::initSubDevice(
    device::Info& info, 
    cl_uint maxComputeUnits, 
    const device::CreateSubDevicesInfo& create_info)
{
    if (workerThreadsAffinity_ == NULL) {
        workerThreadsAffinity_ = new amd::Os::ThreadAffinityMask;
        if (workerThreadsAffinity_ == NULL) {
            return false;
        }
    }

    info_ = info;
    info_.maxComputeUnits_ = maxComputeUnits;
    info_.partitionCreateInfo_ = create_info.p_;
    if (create_info.p_.type_.value_ == device::PartitionType::BY_COUNTS) {
        cl_uint* countsList = new cl_uint[create_info.p_.byCounts_.listSize_];
        if (countsList == NULL) {
            return false;
        }
        for (size_t i = 0; i < create_info.p_.byCounts_.listSize_; ++i) {
            countsList[i] = create_info.countsListAt(i);
        }
        info_.partitionCreateInfo_.byCounts_.countsList_ = countsList;
    }

    // The device cannot be partitioned further
    if (maxComputeUnits == 1) {
        info_.partitionProperties_.value_ = 0;
        info_.affinityDomain_.value_ = 0;
    }
    return true;
}

void
Device::setWorkerThreadsAffinity(
    cl_uint numWorkerThreads, 
    const amd::Os::ThreadAffinityMask* threadsAffinityMask,
    uint& baseCoreId)
{
    uint coreId = baseCoreId;
    if (threadsAffinityMask == NULL) {
        for (cl_uint i = 0; i < numWorkerThreads; ++i) {
            ++coreId;
            workerThreadsAffinity_->set(coreId);
        }
    }
    else { // Already has affinity, so filter accordingly
        for (cl_uint i = 0; i < numWorkerThreads; ++i) {
            coreId = threadsAffinityMask->getNextSet(coreId);
            workerThreadsAffinity_->set(coreId);
        }
    }
    baseCoreId = coreId;
}

cl_int
Device::createSubDevices(
    device::CreateSubDevicesInfo& create_info,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    switch (create_info.p_.type_.value_) {
    case device::PartitionType::EQUALLY:
        return partitionEqually(
            create_info, num_entries, devices, num_devices);

    case device::PartitionType::BY_COUNTS:
        return partitionByCounts(
            create_info, num_entries, devices, num_devices);

    case device::PartitionType::BY_AFFINITY_DOMAIN:
        if (info_.affinityDomain_.value_ == 0) {
            return CL_DEVICE_PARTITION_FAILED;
        }

        if (create_info.p_.byAffinityDomain_.next_) {
            create_info.p_.byAffinityDomain_.next_ = 0;
            create_info.p_.byAffinityDomain_.value_ =
                (1 << amd::leastBitSet(info_.affinityDomain_.value_));
        }
        else {
            if ((create_info.p_.byAffinityDomain_.value_ &
                info_.affinityDomain_.value_) == 0) {
                return CL_INVALID_VALUE;
            }
        }

        if (create_info.p_.byAffinityDomain_.numa_) {
            return partitionByAffinityDomainNUMA(
                create_info, num_entries, devices, num_devices);
        }
        else {
            return partitionByAffinityDomainCacheLevel(
                create_info, num_entries, devices, num_devices);
        }
    default:
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

cl_int
Device::partitionEqually(
    const device::CreateSubDevicesInfo& create_info,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    cl_uint subComputeUnits =
        (cl_uint)create_info.p_.equally_.numComputeUnits_;
    if (subComputeUnits == 0) {
        return CL_INVALID_VALUE;
    }

    cl_uint numSubDevices = info_.maxComputeUnits_ / subComputeUnits;
    if (numSubDevices == 0) {
        return CL_DEVICE_PARTITION_FAILED;
    }

    if (num_devices != NULL) {
        *num_devices = numSubDevices;
    }

    if (devices != NULL) {
        if (num_entries < numSubDevices) {
            return CL_INVALID_VALUE;
        }
        uint coreId = (uint)-1;
        while (numSubDevices-- > 0) {
            Device* device = new Device(this);
            if (device == NULL) {
                return CL_OUT_OF_HOST_MEMORY;
            }

            if (!device->create() ||
                !device->initSubDevice(info_, subComputeUnits, create_info)) {
                    device->release();
                    return CL_OUT_OF_HOST_MEMORY;
            }

            device->setWorkerThreadsAffinity(
                subComputeUnits, workerThreadsAffinity_, coreId);
            *devices++ = as_cl(static_cast<amd::Device*>(device));
        }
    }
    
    return CL_SUCCESS;
}

cl_int
Device::partitionByCounts(
    const device::CreateSubDevicesInfo& create_info,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    cl_uint maxComputeUnits = 0;
    cl_uint numSubDevices = (cl_uint)create_info.p_.byCounts_.listSize_;
    for (size_t i = (size_t)numSubDevices; i > 0; --i) {
        maxComputeUnits += create_info.countsListAt(i);
    }
    if (numSubDevices == 0 || maxComputeUnits > info_.maxComputeUnits_) {
        return CL_INVALID_DEVICE_PARTITION_COUNT;
    }

    if (num_devices != NULL) {
        *num_devices = numSubDevices;
    }

    if (devices != NULL) {
        if (num_entries < numSubDevices) {
            return CL_INVALID_VALUE;
        }
        uint coreId = (uint)-1;
        while (numSubDevices-- > 0) {
            Device* device = new Device(this);
            if (device == NULL) {
                return CL_OUT_OF_HOST_MEMORY;
            }

            cl_uint subComputeUnits =
                create_info.countsListAt((size_t)numSubDevices);
            if (!device->create() ||
                !device->initSubDevice(info_, subComputeUnits, create_info)) {
                    device->release();
                    return CL_OUT_OF_HOST_MEMORY;
            }

            device->setWorkerThreadsAffinity(
                subComputeUnits, workerThreadsAffinity_, coreId);
            *devices++ = as_cl(static_cast<amd::Device*>(device));
        }
    }

    return CL_SUCCESS;
}

cl_int
Device::partitionByAffinityDomainNUMA(
    const device::CreateSubDevicesInfo& create_info,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    cl_uint numSubDevices = 0;

#if defined(__linux__)
#if !defined(NUMA_SUPPORT)
    return CL_INVALID_VALUE;
#else
    int highestNuma = numa_max_node();
    if (highestNuma < 0) {
        return CL_INVALID_VALUE;
    }

    numSubDevices = (cl_uint)highestNuma;
    if (devices != NULL) {
        for (int node = 0; node <= highestNuma; ++node) {
            cl_uint subComputeUnits = 0;
            int len = 1;
            while (true) {
                ulong* cpus = alloca(sizeof(ulong)*len);
                if (numa_node_to_cpus(node, cpus, len * sizeof(ulong)) < 0) {
                    if (errno != ERANGE) {
                        return CL_INVALID_VALUE;
                    }
                    len *= 2;
                }
                else {
                    len *= sizeof(ulong) * 8;
                    for (int i = 0; i < len; i++) {
                        if (test_bit(i, cpus)) {
                            ++subComputeUnits;
                        }
                    }
                    break;
                }
            }

            if (subComputeUnits == 0) {
                return CL_INVALID_VALUE;
            }

            Device* device = new Device(this);
            if (device == NULL) {
                return CL_OUT_OF_HOST_MEMORY;
            }

            if (!device->create() || NULL == (device->numaMask_ = new nodemask_t)) {
                device->release();
                return CL_OUT_OF_HOST_MEMORY;
            }


            if (!device->initSubDevice(
                info_, subComputeUnits, create_info)) {
                    delete device->numaMask_;
                    device->numaMask_ = NULL;
                    device->release();
                    return CL_OUT_OF_HOST_MEMORY;
            }

            nodemask_zero(device->numaMask_);
            nodemask_set(device->numaMask_, node);
            // Need to remove this domain type
            device->info_.affinityDomain_.numa_ = 0;
            *devices++ = as_cl(static_cast<amd::Device*>(device));
        }
    }
#endif // NUMA_SUPPORT

#else // win32
    GROUP_AFFINITY numaNodeMask;
    ULONG highestNuma = 0;
    if (!::GetNumaHighestNodeNumber(&highestNuma)) {
        return CL_INVALID_VALUE;
    }

    for (ULONG node = 0; node <= highestNuma; ++node) {
        if (pfnGetNumaNodeProcessorMaskEx != NULL) {
            if (!pfnGetNumaNodeProcessorMaskEx((USHORT)node, &numaNodeMask)) {
                // Highet NUMA node number is not guaranteed to be the
                // number of nodes.
                continue;
            }
        }
        else {
            ULONGLONG tmpMask;
            if (!::GetNumaNodeProcessorMask((UCHAR)node, &tmpMask)) {
                // Highet NUMA node number is not guaranteed to be the
                // number of nodes.
                continue;
            }
            numaNodeMask.Group = 0;
            numaNodeMask.Mask = (KAFFINITY)tmpMask;
        }

        if (workerThreadsAffinity_ != NULL) {
            workerThreadsAffinity_->adjust(0, numaNodeMask.Mask);
        }
        if (numaNodeMask.Mask == 0) {
            continue;
        }

        if (devices != NULL) {
            Device* device = new Device(this);
            if (device == NULL) {
                return CL_OUT_OF_HOST_MEMORY;
            }

            if (!device->create() || !device->initSubDevice(info_, 
                (cl_uint)amd::countBitsSet(numaNodeMask.Mask), create_info)) {
                    device->release();
                    return CL_OUT_OF_HOST_MEMORY;
            }

            device->workerThreadsAffinity_->set(
                numaNodeMask.Group, numaNodeMask.Mask);
            // Need to remove this domain type
            device->info_.affinityDomain_.numa_ = 0;
            *devices++ = as_cl(static_cast<amd::Device*>(device));
        }
        numSubDevices++;
    }

#endif // win32

    if (num_devices != NULL) {
        *num_devices = numSubDevices;
    }

    // Could not get a processor mask for any of the nodes
    if (numSubDevices == 0) {
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

#if defined(__linux__)
static bool
readFileString(const char* file, char* buf, size_t bufSize)
{
    int fd = open(file, O_RDONLY);
    if (fd < 0) {
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }

    if ((size_t)st.st_size < bufSize) {
        bufSize = (size_t)st.st_size;
    }

    ssize_t n = read(fd, buf, bufSize);
    close(fd);

    if (n <= 0) {
        return false;
    }

    if (n >= (ssize_t)bufSize) {
        n = (ssize_t)bufSize - 1;
    }
    buf[n] = '\0';
    return true;
}

static void
parseSharedCpuMap(const char* cpuMap, cpu_set_t& mask)
{
    CPU_ZERO(&mask);
    uint32_t* bits = (uint32_t*)mask.__bits;
    const char* s = cpuMap + strlen(cpuMap);
    while (true) { 
        s = (const char*)memrchr(cpuMap, ',', s - cpuMap);
        if (!s) {
            s = cpuMap;
        }
        else {
            s++;
        }

        *bits++ = strtoul(s, NULL, 16);

        if (s == cpuMap) {
            return;
        }

        --s;
    }
}
#endif // linux

cl_int
Device::partitionByAffinityDomainCacheLevel(
    const device::CreateSubDevicesInfo& create_info,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
    cl_uint cacheLevel = 0;
    switch (create_info.p_.byAffinityDomain_.value_) {
    case device::AffinityDomain::AFFINITY_DOMAIN_L4_CACHE:
        cacheLevel = 4;
        break;
    case device::AffinityDomain::AFFINITY_DOMAIN_L3_CACHE:
        cacheLevel = 3;
        break;
    case device::AffinityDomain::AFFINITY_DOMAIN_L2_CACHE:
        cacheLevel = 2;
        break;
    case device::AffinityDomain::AFFINITY_DOMAIN_L1_CACHE:
        cacheLevel = 1;
        break;
    default:
        return CL_INVALID_VALUE;
    }

    const uint negAffinityDomain =
        ~create_info.p_.byAffinityDomain_.value_;
    cl_uint numSubDevices = 0;

#if defined(__linux__)

    amd::Os::ThreadAffinityMask affinityMask;
    if (workerThreadsAffinity_ != NULL) {
        affinityMask = *workerThreadsAffinity_;
    }
    else {
        for (uint cpuId = 0; cpuId < (uint)info_.maxComputeUnits_; ++cpuId) {
            affinityMask.set(cpuId);
        }
    }

    amd::Os::ThreadAffinityMask currentMask;
    char buf[1024];
    for (uint cpuId = affinityMask.getFirstSet(); 
         cpuId != (uint)-1; 
         cpuId = affinityMask.getNextSet(cpuId)) {

        sprintf(buf, 
            "/sys/devices/system/cpu/cpu%u/cache/index%u/shared_cpu_map", 
            cpuId, cacheLevel);

        if (!readFileString(buf, buf, sizeof(buf))) {
            return CL_INVALID_VALUE;
        }

        parseSharedCpuMap(buf, currentMask.getNative());
        affinityMask.adjust(currentMask.getNative());
        if (currentMask.isEmpty()) {
            continue;
        }

        cl_uint maxComputeUnits;
        if (cacheLevel > 1) {
            maxComputeUnits = 0;
            amd::Os::ThreadAffinityMask currentMaskSub;
            cl_uint cacheLevelSub = cacheLevel - 1;
            for (uint cpuIdSub = affinityMask.getFirstSet(); 
                 cpuIdSub != (uint)-1; 
                 cpuIdSub = affinityMask.getNextSet(cpuIdSub)) {

                sprintf(buf, 
                    "/sys/devices/system/cpu/cpu%u/cache/index%u/shared_cpu_map", 
                    cpuIdSub, cacheLevelSub);

                if (!readFileString(buf, buf, sizeof(buf))) {
                    return CL_INVALID_VALUE;
                }

                parseSharedCpuMap(buf, currentMaskSub.getNative());
                currentMask.adjust(currentMaskSub.getNative());
                if (!currentMaskSub.isEmpty()) {
                    ++maxComputeUnits;
                }
            }

            if (maxComputeUnits == 0) {
                continue;
            }
        }
        else {
            maxComputeUnits = 1;
        }

        if (devices != NULL) {
            Device* device = new Device(this);
            if (device == NULL) {
                return CL_OUT_OF_HOST_MEMORY;
            }

            if (!device->create() ||
                !device->initSubDevice(info_, maxComputeUnits, create_info)) {
                device->release();
                return CL_OUT_OF_HOST_MEMORY;
            }

            device->workerThreadsAffinity_->set(currentMask.getNative());
            // Need to remove this domain type
            device->info_.affinityDomain_.value_ &= negAffinityDomain;
            *devices++ = as_cl(static_cast<amd::Device*>(device));
        }
        numSubDevices++;
        affinityMask.clear(currentMask.getNative());
    }

#else // win32
    DWORD length = 0;
    ::GetLogicalProcessorInformation(NULL, &length);

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer =
        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION) malloc(length);

    if (buffer != NULL && ::GetLogicalProcessorInformation(buffer, &length)) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr, limit =
            &buffer[length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)];

        for (ptr = buffer; ptr < limit; ++ptr) {
            PCACHE_DESCRIPTOR cache = &ptr->Cache;
            if (ptr->Relationship == RelationCache && cache->Type != CacheInstruction) {
                if (cache->Level == cacheLevel) {
                    KAFFINITY affinityMask = (KAFFINITY)ptr->ProcessorMask;
                    if (workerThreadsAffinity_ != NULL) {
                        workerThreadsAffinity_->adjust(0, affinityMask);
                    }
                    if (affinityMask == 0) {
                        continue;
                    }

                    cl_uint maxComputeUnits;
                    if (cacheLevel > 1) {
                        maxComputeUnits = 0;
                        cl_uint cacheLevelSub = cacheLevel - 1;
                        for (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION
                            ptrSub = buffer; ptrSub < limit; ++ptrSub) {

                            PCACHE_DESCRIPTOR cacheSub = &ptrSub->Cache;
                            if (ptrSub->Relationship == RelationCache &&
                                cacheSub->Type != CacheInstruction) {
                                if (cacheSub->Level == cacheLevelSub &&
                                    ((affinityMask & (KAFFINITY)ptrSub->ProcessorMask) != 0)) {
                                    ++maxComputeUnits;
                                }
                            }
                        }

                        if (maxComputeUnits == 0) {
                            continue;
                        }
                    }
                    else {
                        maxComputeUnits = 1;
                    }

                    if (devices != NULL) {
                        Device* device = new Device(this);
                        if (device == NULL) {
                            free(buffer);
                            return CL_OUT_OF_HOST_MEMORY;
                        }

                        if (!device->create() || !device->initSubDevice(info_, 
                            maxComputeUnits, create_info)) {
                            free(buffer);
                            device->release();
                            return CL_OUT_OF_HOST_MEMORY;
                        }

                        device->workerThreadsAffinity_->set(0, affinityMask);
                        // Need to remove this domain type
                        device->info_.affinityDomain_.value_ &= negAffinityDomain;
                        *devices++ = as_cl(static_cast<amd::Device*>(device));
                    }
                    numSubDevices++;
                    if (numSubDevices >= info_.maxComputeUnits_) {
                        break;
                    }
                }
            }
        }
    }

    free(buffer);

#endif

    if (num_devices != NULL) {
        *num_devices = numSubDevices;
    }

    if (numSubDevices == 0) {
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

device::Program*
Device::createProgram(int oclVer)
{
    Program* cpuProgram = new Program(*this);
    if (cpuProgram == NULL) {
        LogError("We failed memory allocation for program!");
    }

    return cpuProgram;
}

void*
Device::allocMapTarget(
    amd::Memory&        mem,
    const amd::Coord3D& origin,
    const amd::Coord3D& region,
    size_t*             rowPitch,
    size_t*             slicePitch)
{
    if (mem.asImage() != NULL) {
        amd::Image * image = mem.asImage();
        size_t elementSize = image->getImageFormat().getElementSize();
        size_t rp = image->getRowPitch();
        size_t sp = image->getSlicePitch();
        *rowPitch = rp;
        if (slicePitch) {
            *slicePitch = sp;
        }
        return (address) image->getHostMem()
            + (origin[0] * elementSize + origin[1] * rp + origin[2] * sp);
    }
    else if (mem.asBuffer() != NULL) {
        return (address) mem.getHostMem() + origin[0];
    }

    return NULL;
}

void
Device::freeMapTarget(amd::Memory& mem, void* target)
{
    // nop for CPU
}

} // namespace cpu
