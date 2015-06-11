//
// Copyright (c) 2008 Advanced Micro Devices, Inc. All rights reserved.
//


#ifndef WITHOUT_FSA_BACKEND


#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"
#include "thread/monitor.hpp"
#include "CL/cl_ext.h"

#include "newcore.h"

#include "amdocl/cl_common.hpp"
#include "device/hsa/hsadevice.hpp"
#include "device/hsa/hsavirtual.hpp"
#include "device/hsa/hsaprogram.hpp"
#include "device/hsa/hsablit.hpp"
#include "device/hsa/hsacompilerlib.hpp"
#include "device/hsa/hsamemory.hpp"
#include "hsacore_symbol_loader.hpp"
#include "device/hsa/oclhsa_common.hpp"
#include "kv_id.h"
#include "vi_id.h"
#include "cz_id.h"
#include "hsainterop.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include "CL/cl_gl.h"

#ifdef _WIN32
#include "CL/cl_d3d10.h"
#endif  // _WIN32

#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#endif  // WITHOUT_FSA_BACKEND

const HsaCoreApiTable *hsacoreapi = NULL;
const HsaServicesApiTable *servicesapi = NULL;
#define OPENCL_VERSION_STR XSTR(OPENCL_MAJOR) "." XSTR(OPENCL_MINOR)

#ifndef WITHOUT_FSA_BACKEND
namespace device {
extern const char* BlitSourceCode;
}

namespace oclhsa {

aclCompiler* NullDevice::compilerHandle_;
bool oclhsa::Device::isHsaInitialized_ = false;
const bool oclhsa::Device::offlineDevice_ = false;
const bool oclhsa::NullDevice::offlineDevice_= true;

static HsaDeviceId getHsaDeviceId(const HsaDevice *device) {
  /*
   * Use the device id to determine the ASIC family
   */
  switch (device->device_id) {
    case DEVICE_ID_SPECTRE_MOBILE:
    case DEVICE_ID_SPECTRE_DESKTOP:
    case DEVICE_ID_SPECTRE_LITE_MOBILE_1309:
    case DEVICE_ID_SPECTRE_LITE_MOBILE_130A:
    case DEVICE_ID_SPECTRE_SL_MOBILE_130B:
    case DEVICE_ID_SPECTRE_MOBILE_130C:
    case DEVICE_ID_SPECTRE_LITE_MOBILE_130D:
    case DEVICE_ID_SPECTRE_SL_MOBILE_130E:
    case DEVICE_ID_SPECTRE_DESKTOP_130F:
    case DEVICE_ID_SPECTRE_WORKSTATION_1310:
    case DEVICE_ID_SPECTRE_WORKSTATION_1311:
    case DEVICE_ID_SPECTRE_LITE_DESKTOP_1313:
    case DEVICE_ID_SPECTRE_SL_DESKTOP_1315:
    case DEVICE_ID_SPECTRE_SL_MOBILE_1318:
    case DEVICE_ID_SPECTRE_SL_EMBEDDED_131B:
    case DEVICE_ID_SPECTRE_EMBEDDED_131C:
    case DEVICE_ID_SPECTRE_LITE_EMBEDDED_131D:
        return HSA_SPECTRE_ID;
    case DEVICE_ID_SPOOKY_MOBILE:
    case DEVICE_ID_SPOOKY_DESKTOP:
    case DEVICE_ID_SPOOKY_DESKTOP_1312:
    case DEVICE_ID_SPOOKY_DESKTOP_1316:
    case DEVICE_ID_SPOOKY_MOBILE_1317:
		return HSA_SPOOKY_ID;
    case DEVICE_ID_VI_TONGA_P_6920:
    case DEVICE_ID_VI_TONGA_P_6921:
    case DEVICE_ID_VI_TONGA_P_6928:
    case DEVICE_ID_VI_TONGA_P_692B:
    case DEVICE_ID_VI_TONGA_P_692F:
    case DEVICE_ID_VI_TONGA_P_6938:
    case DEVICE_ID_VI_TONGA_P_6939:
		return HSA_TONGA_ID;
    case DEVICE_ID_CZ_9870:
    case DEVICE_ID_CZ_9874:
    case DEVICE_ID_CZ_9875:
    case DEVICE_ID_CZ_9876:
    case DEVICE_ID_CZ_9877:
		return HSA_CARRIZO_ID;
    case DEVICE_ID_VI_ICELAND_M_6900:
    case DEVICE_ID_VI_ICELAND_M_6901:
    case DEVICE_ID_VI_ICELAND_M_6902:
    case DEVICE_ID_VI_ICELAND_M_6903:
    case DEVICE_ID_VI_ICELAND_M_6907:
		return HSA_ICELAND_ID;
    default:
        return HSA_INVALID_DEVICE_ID;
  }
}
bool NullDevice::create(const AMDDeviceInfo& deviceInfo) {
    online_ = false;
    deviceInfo_ = deviceInfo;
    // Mark the device as GPU type
    info_.type_     = CL_DEVICE_TYPE_GPU | CL_HSA_ENABLED_AMD;
    info_.vendorId_ = 0x1002;

    settings_ = new Settings();
    oclhsa::Settings* hsaSettings = static_cast<oclhsa::Settings*>(settings_);
    if ((hsaSettings == NULL) ||
        // @Todo sramalin Use double precision from constsant
        !hsaSettings->create((true) & 0x1)) {
            LogError("Error creating settings for NULL HSA device");
            return false;
    }
    // Report the device name
    ::strcpy(info_.name_, deviceInfo_.machineTarget_);
    info_.extensions_ = getExtensionString();
    info_.maxWorkGroupSize_ = hsaSettings->maxWorkGroupSize_;
    ::strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");
    info_.oclcVersion_ = "OpenCL C " OPENCL_VERSION_STR " ";
    std::string driverVersion = AMD_BUILD_STRING;
    driverVersion.append(" (HSA)");
    strcpy(info_.driverVersion_, driverVersion.c_str());
    info_.version_ = "OpenCL " OPENCL_VERSION_STR " ";
    return true;
}

Device::Device(const HsaDevice *bkendDevice)
    : _bkendDevice(bkendDevice),  context_(NULL), xferQueue_(NULL)
{
}

Device::~Device()
{
    // Destroy transfer queue
    if (xferQueue_ && xferQueue_->terminate()) {
        delete xferQueue_;
        xferQueue_ = NULL;
    }

    if (blitProgram_) {
        delete blitProgram_;
        blitProgram_ = NULL;
    }

    if (context_ != NULL) {
        context_->release();
    }

    if (info_.extensions_) {
        delete[]info_.extensions_;
        info_.extensions_ = NULL;
    }

    if (settings_) {
        delete settings_;
        settings_ = NULL;
    }
}
bool NullDevice::initCompiler(bool isOffline) {
     // Initializes g_complibModule and g_complibApi if they were not initialized
    if( g_complibModule == NULL ){
        if (!LoadCompLib(isOffline)) {
            if (!isOffline) {
                LogError("Error - could not find the compiler library");
            }
            return false;
        }
    }
    //Initialize the compiler handle if has already not been initialized
    //This is destroyed in Device::teardown
    acl_error error;
    if (!compilerHandle_) {
        compilerHandle_ = g_complibApi._aclCompilerInit(NULL, &error);
        if (error != ACL_SUCCESS) {
            LogError("Error initializing the compiler handle");
            return false;
        }
    }
    return true;
}

bool NullDevice::destroyCompiler() {
    if (compilerHandle_ != NULL) {
        acl_error error = g_complibApi._aclCompilerFini(compilerHandle_);
        if (error != ACL_SUCCESS) {
            LogError("Error closing the compiler");
            return false;
        }
    }
    if( g_complibModule != NULL ){
        UnloadCompLib();
    }
    return true;
}

void NullDevice::tearDown() {
    destroyCompiler();
}
bool NullDevice::init() {
    //Initialize the compiler
    if (!initCompiler(offlineDevice_)){
        return false;
    }
    //If there is an HSA enabled device online then skip any offline device
    std::vector<Device*> devices;
    devices = getDevices(CL_DEVICE_TYPE_GPU | CL_HSA_ENABLED_AMD, false);

    //Load the offline devices
    //Iterate through the set of available offline devices
    for (uint id = 0; id < sizeof(DeviceInfoTable)/sizeof(AMDDeviceInfo); id++) {
        bool isOnline = false;
        //Check if the particular device is online
        for (unsigned int i=0; i< devices.size(); i++) {
            if (static_cast<NullDevice*>(devices[i])->deviceInfo_.hsaDeviceId_ ==
                DeviceInfoTable[id].hsaDeviceId_){
                    isOnline = true;
            }
        }
        if (isOnline) {
            continue;
        }
        NullDevice* nullDevice = new NullDevice();
        if (!nullDevice->create(DeviceInfoTable[id])) {
            LogError("Error creating new instance of Device.");
            delete nullDevice;
            return false;
        }
        nullDevice->registerDevice();
    }
    return true;
}
NullDevice::~NullDevice() {
        if (info_.extensions_) {
            delete[]info_.extensions_;
            info_.extensions_ = NULL;
        }

        if (settings_) {
            delete settings_;
            settings_ = NULL;
        }
}
bool Device::init() {
    // Assumption: init() will be called by ocl only once at the start of program
    // with a matching tearDown() when program exits.
    // TODO(papte) Check if init(),
    // tearDown(), init(), tearDown()  repeat sequence is possible in one session
    // (process lifetime).  If so we will be calling LoadLibrary() and
    // FreeLibrary() ifcn the similar repeat sequence.  Investigate the effect of
    // this on the HSA Device and Core runtime's initialzers, where the device list
    // is generated in the runtime.
#ifdef BUILD_STATIC_HSA
    HsaGetCoreApiTable(&hsacoreapi);
    HsaGetServicesApiTable(&servicesapi);
#else
    bool core_dll_loaded = HsacoreApiSymbols::Instance().IsDllLoaded();
    bool service_dll_loaded = ServicesApiSymbols::Instance().IsDllLoaded();

    if (!core_dll_loaded  && !service_dll_loaded ) {
        // Both DLLs are not loaded, assume HSA not installed on a non-HSA
        // machine, returning true.
        LogInfo("HSA stack not available.");
        return true; // Return true, indicating nothing is wrong and 
                     // assuming HSA not installed.
    } else if (core_dll_loaded ^ service_dll_loaded) {
        // If Only one of the two HSA DLLs failed, then its an ERROR.
        LogError("One of the HSA libraies, core or services failed to load.\n");
        return false;
    } else {
        // Both DLLs loaded, continue initializing HSA stack.
        LogInfo("Initializing HSA stack.");
    }
  
  // First thing first, initialize hsacoreapi and servicesapi to call core and 
  // services API respectively.
  HsacoreApiSymbols::Instance().HsaGetCoreApiTable(&hsacoreapi);
  ServicesApiSymbols::Instance().HsaGetServicesApiTable(&servicesapi);
#endif
    isHsaInitialized_ = false;
    if (hsacoreapi->HsaAmdInitialize() != kHsaStatusSuccess) {
        // Either an error in HSA core initialization or 
        // KFD not installed on the machine.
        // Return without error, so OpenCL can continue without HSA stack.
        return true;
    }
    isHsaInitialized_ = true;

  // Initialize the structure used to configure the
  // behavior of Hsa Runtime
  // TODO (PA) : verify if this ito be called or not.
  // Latest code does not call.
  // SetHsaEnvConfig();

    //Initialize the compiler
    if (!initCompiler(offlineDevice_)){
        return false;
    }

    const HsaDevice *devices = NULL;
    unsigned num_devices = 0;

    // Initialize the Hsa Service layer
    servicesapi->HsaInitServices(128);

    HsaStatus status = hsacoreapi->HsaGetDevices(&num_devices, &devices);
    if (status != kHsaStatusSuccess) {
        LogPrintfError(
            "in %s(), Call to newcore HsaGetDevices() failed, HsaStatus: %d",
            __FUNCTION__, status);
        return false;
    }

    for (unsigned int i = 0; i < num_devices; i++) {
        Device *oclhsa_device = new Device(&devices[i]);
        if (!oclhsa_device) {
            LogError("Error creating new instance of Device on then heap.");
            return false;
        }
        HsaDeviceId deviceId = getHsaDeviceId(&devices[i]);
        if (deviceId == HSA_INVALID_DEVICE_ID) {
            LogError(" Invalid HSA device");
            return false;
        }
        //Find device id in the table
        unsigned sizeOfTable = sizeof(DeviceInfoTable)/sizeof(AMDDeviceInfo);
        uint id;
        for (id = 0; id < sizeOfTable; id++) {
            if (DeviceInfoTable[id].hsaDeviceId_ == deviceId){
                break;
            }
        }
        //If the AmdDeviceInfo for the HsaDevice Id could not be found return false
        if (id == sizeOfTable) {
            return false;
        }
        oclhsa_device->deviceInfo_ = DeviceInfoTable[id];

        if (!oclhsa_device->mapHSADeviceToOpenCLDevice(&devices[i])) {
            LogError("Failed mapping of HsaDevice to Device.");
            return false;
        }

        if (!oclhsa_device->create()) {
            LogError("Error creating new instance of Device.");
            return false;
        }
        oclhsa_device->registerDevice();  // no return code for this function
    }
    return true;
}

void
Device::tearDown()
{
    if (isHsaInitialized_) {
        if (servicesapi != NULL && servicesapi->HsaDestroyServices != NULL) {
            servicesapi->HsaDestroyServices();
        }
        hsacoreapi->HsaAmdShutdown();
    }
  NullDevice::tearDown();
  HsacoreApiSymbols::teardown();
  ServicesApiSymbols::teardown();
}

bool
Device::create()
{
    amd::Context::Info  info = {0};
    std::vector<amd::Device*> devices;
    devices.push_back(this);

    // Create a dummy context
    context_ = new amd::Context(devices, info);
    if (context_ == NULL) {
        return false;
    }

    blitProgram_ = new BlitProgram(context_);
    // Create blit programs
    if (blitProgram_ == NULL || !blitProgram_->create(this)) {
        delete blitProgram_;
        blitProgram_ = NULL;
        LogError("Couldn't create blit kernels!");
        return false;
    }

    return true;
}

oclhsa::Memory*
Device::getOclHsaMemory(amd::Memory* mem) const
{
    return static_cast<oclhsa::Memory*>(mem->getDeviceMemory(*this));
}

device::Program*
NullDevice::createProgram(int oclVer) {
    return new oclhsa::FSAILProgram(*this);
}

device::Program*
Device::createProgram(int oclVer) {
    return new oclhsa::FSAILProgram(*this);
}

cl_device_svm_capabilities
Device::getSvmCapabilities(const HsaDevice* device)
{
    // KV supports all types of SVM
    if (device->device_id >= DEVICE_ID_SPECTRE_MOBILE &&
        device->device_id <= DEVICE_ID_SPECTRE_EMBEDDED_131C) {

        cl_bitfield atomics = CL_DEVICE_SVM_ATOMICS;
        // Atomics are allowed in 32 bits if a environment variable is set
        if (Is32Bits() && !settings().enableSvm32BitsAtomics_) {
            atomics = 0;
        }
        return CL_DEVICE_SVM_COARSE_GRAIN_BUFFER |
                CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
                CL_DEVICE_SVM_FINE_GRAIN_SYSTEM |
                atomics;
    }
    // Devices such as Bonaire enable some HSA features but they do not include
    // CL_DEVICE_SVM_FINE_GRAIN_SYSTEM (because of addresses above 2^40) or
    // CL_DEVICE_SVM_ATOMICS capabilities.
    return CL_DEVICE_SVM_COARSE_GRAIN_BUFFER |
            CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
}

bool
Device::mapHSADeviceToOpenCLDevice(const HsaDevice *dev)
{
    // Create HSA settings
    settings_ = new Settings();
    oclhsa::Settings* hsaSettings = static_cast<oclhsa::Settings*>(settings_);
    if ((hsaSettings == NULL) ||
        !hsaSettings->create((dev->is_double_precision) & 0x1)) {
        return false;
    }
    // Report the device name
    ::strcpy(info_.name_, deviceInfo_.machineTarget_);
    strcpy(info_.boardName_, dev->device_name);

    if (dev->number_cache_descriptors != 0) {
        HsaCacheDescriptor* cacheDesc = dev->cache_descriptors;
        info_.globalMemCacheLineSize_ = cacheDesc->cache_line_size;
        info_.globalMemCacheSize_ = cacheDesc->cache_size * Ki;

        info_.globalMemCacheType_ = (cacheDesc->cache_type.value == 0) ?
             CL_NONE : CL_READ_WRITE_CACHE;
    }
    else {
        info_.globalMemCacheType_ = CL_NONE;
        info_.globalMemCacheLineSize_ = 0;
        info_.globalMemCacheSize_ = 0;
    }

    // Map HSA device types to OCL device types.
    // if (dev->device_type == kHsaDeviceTypeThroughput)
    info_.type_ = CL_DEVICE_TYPE_GPU | CL_HSA_ENABLED_AMD;

    info_.maxComputeUnits_ = dev->number_compute_units;
    info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
    info_.deviceTopology_.pcie.bus = (dev->location_id&(0xFF<<8))>>8;
    info_.deviceTopology_.pcie.device = (dev->location_id&(0x1F<<3))>>3;
    info_.deviceTopology_.pcie.function = (dev->location_id&0x07);
    info_.extensions_ = getExtensionString();
    info_.nativeVectorWidthDouble_ =
    info_.preferredVectorWidthDouble_ = (settings().doublePrecision_) ? 1 : 0;

    info_.maxWorkGroupSize_ = dev->wave_front_size * dev->max_waves_per_simd;
    info_.maxClockFrequency_ = dev->max_clock_rate_of_f_compute;
    //info_.imageSupport_ = dev->is_image_support;
    info_.imageSupport_ = false;

    info_.localMemSizePerCU_ = dev->group_memory_size;

    if (populateOCLDeviceConstants() == false) {
        return false;
    }

    // Populate the single config setting.
    info_.singleFPConfig_ = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO |
        CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

    if (hsaSettings->doublePrecision_) {
        info_.doubleFPConfig_ = info_.singleFPConfig_ | CL_FP_DENORM;
        info_.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }

    info_.svmCapabilities_ = getSvmCapabilities(dev);
    info_.preferredPlatformAtomicAlignment_ = 0;
    info_.preferredGlobalAtomicAlignment_ = 0;
    info_.preferredLocalAtomicAlignment_ = 0;

    return true;
}

static bool
isFrameBufferDescriptor(HsaMemoryDescriptor &desc)
{
  return (desc.heap_type == kHsaHeapTypeFrameBufferPrivate);
}

bool
Device::populateOCLDeviceConstants()
{
    info_.available_ = true;
    /*info_.maxWorkGroupSize_ = 256;*/
    info_.maxWorkItemDimensions_ = 3;

    // Get frame buffer memory descriptor.
    HsaMemoryDescriptor *memDescBegin = _bkendDevice->memory_descriptors;
    HsaMemoryDescriptor *memDescEnd =
        memDescBegin + _bkendDevice->number_memory_descriptors;
    HsaMemoryDescriptor *hsaFbDesc =
        std::find_if(memDescBegin, memDescEnd, isFrameBufferDescriptor);

    if ((hsaFbDesc != memDescEnd) && (hsaFbDesc->size_in_bytes > 0)) {
        // Device local memory exists. Populate OpenCL info field with
        // attributes of HSA GPU local memory descriptor.
        info_.globalMemSize_ = hsaFbDesc->size_in_bytes;

        info_.maxMemAllocSize_ =
            std::max(std::min(cl_ulong(1 * Gi), info_.globalMemSize_ / 4),
                     cl_ulong(128 * Mi));

        // Make sure the max allocation size is not larger than the available
        // memory size.
        info_.maxMemAllocSize_ =
            std::min(info_.maxMemAllocSize_, info_.globalMemSize_);
    }
    else {
        // The HSA device backend does not have local memory, so we use system
        // memory as default.
        info_.globalMemSize_ = Os::getPhysicalMemSize();
        if (info_.globalMemSize_ == 0) {
            return false;
        }

        // Cap global memory
#if defined (_LP64)
        // Cap at 8TiB for 64-bit
        const cl_ulong maxGlobalMemSize = 8ULL * Ki * Gi;
#elif defined (_WIN32)
        // Cap at 2GiB (see http://msdn.microsoft.com/en-us/library/aa366778.aspx)
        const cl_ulong maxGlobalMemSize = 2ULL * Gi;
#else  // linux
           // Cap at 3.5GiB
        const cl_ulong maxGlobalMemSize = 3584ULL * Mi;
#endif
        info_.globalMemSize_ = std::min(info_.globalMemSize_, maxGlobalMemSize);

        info_.maxMemAllocSize_ =
            info_.globalMemSize_ * CPU_MAX_ALLOC_PERCENT / 100;
        if (flagIsDefault(CPU_MAX_ALLOC_PERCENT)) {
            const cl_ulong minAllocSize = LP64_SWITCH(1ULL * Gi, 2ULL * Gi);
            info_.maxMemAllocSize_ = std::max(info_.maxMemAllocSize_,
                std::min(info_.globalMemSize_, minAllocSize));
        }
    }

    /*make sure we don't run anything over 8 params for now*/
    info_.maxParameterSize_ = 1024;  // [TODO]: CAL stack values: 1024*
                                   // constant
    info_.maxWorkItemSizes_[0] = 256;
    info_.maxWorkItemSizes_[1] = 256;
    info_.maxWorkItemSizes_[2] = 256;

    info_.nativeVectorWidthChar_ = info_.preferredVectorWidthChar_ = 4;
    info_.nativeVectorWidthShort_ = info_.preferredVectorWidthShort_ = 2;
    info_.nativeVectorWidthInt_ = info_.preferredVectorWidthInt_ = 1;
    info_.nativeVectorWidthLong_ = info_.preferredVectorWidthLong_ = 1;
    info_.nativeVectorWidthFloat_ = info_.preferredVectorWidthFloat_ = 1;

    info_.localMemSize_ = 32 * 1024;
    info_.hostUnifiedMemory_ = CL_TRUE;
    info_.memBaseAddrAlign_ = 8 * (flagIsDefault(MEMOBJ_BASE_ADDR_ALIGN) ?
                                 sizeof(cl_long16) : MEMOBJ_BASE_ADDR_ALIGN);
    info_.minDataTypeAlignSize_ = sizeof(cl_long16);

    info_.maxConstantArgs_ = 8;
    info_.maxConstantBufferSize_ = 64 * 1024;
    info_.localMemType_ = CL_LOCAL;
    info_.errorCorrectionSupport_ = false;
    info_.profilingTimerResolution_ = 1;
    info_.littleEndian_ = true;
    info_.compilerAvailable_ = true;
    info_.executionCapabilities_ = CL_EXEC_KERNEL;
    info_.queueProperties_ = CL_QUEUE_PROFILING_ENABLE;
    info_.platform_ = AMD_PLATFORM;
    info_.profile_ = "FULL_PROFILE";
    strcpy(info_.vendor_, "Advanced Micro Devices, Inc.");

    info_.addressBits_ = LP64_SWITCH(32, 64);
    info_.maxSamplers_ = 16;
    info_.maxReadImageArgs_ = 128;
    info_.maxWriteImageArgs_ = 8;
    info_.maxReadWriteImageArgs_ = 64;
    info_.image2DMaxWidth_ = 16 * 1024;
    info_.image2DMaxHeight_ = 16 * 1024;
    info_.image3DMaxWidth_ = 2 * 1024;
    info_.image3DMaxHeight_ = 2 * 1024;
    info_.image3DMaxDepth_ = 2 * 1024;
    info_.imageMaxArraySize_ = 2 * 1024;
    info_.imageMaxBufferSize_ = 64 * 1024;
    info_.imagePitchAlignment_ = 256;
    info_.imageBaseAddressAlignment_ = 256;
    info_.imageMaxArraySize_ = 2048;
    info_.imageMaxBufferSize_ = 65536;
    info_.bufferFromImageSupport_ = CL_TRUE;
    info_.oclcVersion_ = "OpenCL C " OPENCL_VERSION_STR " ";
    std::string driverVersion = AMD_BUILD_STRING;
    driverVersion.append(" (HSA)");
    strcpy(info_.driverVersion_, driverVersion.c_str());
    info_.version_ = "OpenCL " OPENCL_VERSION_STR " ";

    info_.builtInKernels_ = "";
    info_.linkerAvailable_ = true;
    info_.preferredInteropUserSync_ = true;
    info_.printfBufferSize_ = 1000 * 1024;
    info_.vendorId_ = 0x1002; // from gpudevice

    info_.maxGlobalVariableSize_ = static_cast<size_t>(info_.maxMemAllocSize_);
    info_.globalVariablePreferredTotalSize_ =
                                   static_cast<size_t>(info_.globalMemSize_);
    return true;
}

device::VirtualDevice*
Device::createVirtualDevice(amd::CommandQueue* queue)
{
    bool interopQueue = (queue != NULL) &&
            (0 != (queue->context().info().flags_ &
            (amd::Context::GLDeviceKhr |
             amd::Context::D3D10DeviceKhr |
             amd::Context::D3D11DeviceKhr)));

    // Initialization of heap and other resources occur during the command
    // queue creation time.
    HsaQueueType type = kHsaQueueTypeCompute;
    if (interopQueue) {
        type = kHsaQueueTypeInterop;
    }

    VirtualGPU *virtualDevice = new VirtualGPU(*this);

    if (!virtualDevice->create(type)) {
        delete virtualDevice;
        virtualDevice = NULL;
    }

    return virtualDevice;
}

bool
Device::globalFreeMemory(size_t *freeMemory) const
{
    return false;
}

bool
Device::bindExternalDevice(
    intptr_t    type,
    void*       gfxDevice,
    void*       gfxContext,
    bool        validateOnly)
{
    switch (type) {
#ifdef _WIN32
    case CL_CONTEXT_D3D10_DEVICE_KHR:
        if (kHsaStatusSuccess != hsacoreapi->HsaBeginD3D10Interop(
            _bkendDevice, reinterpret_cast<ID3D10Device *>(gfxDevice))) {
            LogError("Failed HsaBeginD3D10Interop()");
            return false;
        }
        break;
    case CL_CONTEXT_D3D11_DEVICE_KHR:
        if (kHsaStatusSuccess != hsacoreapi->HsaBeginD3D11Interop(
            _bkendDevice, reinterpret_cast<ID3D11Device *>(gfxDevice))) {
            LogError("Failed HsaBeginD3D11Interop()");
            return false;
        }
        break;
#endif  // _WIN32
    case CL_GL_CONTEXT_KHR:
        if (kHsaStatusSuccess != hsacoreapi->HsaBeginGLInterop(
            _bkendDevice, reinterpret_cast<GLvoid *>(gfxContext))) {
            LogError("Failed HsaBeginGLInterop()");
            return false;
        }
        break;
    default:
        LogError("Unknown external device!");
        return false;
    }

    if (validateOnly) {
        return unbindExternalDevice(type, gfxDevice, gfxContext, validateOnly);
    }
    return true;
}

bool
Device::unbindExternalDevice(
    intptr_t    type,
    void*       gfxDevice,
    void*       gfxContext,
    bool        validateOnly)
{
    switch (type) {
#ifdef _WIN32
    case CL_CONTEXT_D3D10_DEVICE_KHR:
        if (kHsaStatusSuccess != hsacoreapi->HsaEndD3D10Interop(
            _bkendDevice, reinterpret_cast<ID3D10Device *>(gfxDevice))) {
            LogError("Failed HsaEndD3D10Interop()");
            return false;
        }
        break;
    case CL_CONTEXT_D3D11_DEVICE_KHR:
        if (kHsaStatusSuccess != hsacoreapi->HsaEndD3D11Interop(
            _bkendDevice, reinterpret_cast<ID3D11Device *>(gfxDevice))) {
            LogError("Failed HsaEndD3D11Interop()");
            return false;
        }
        break;
#endif  // _WIN32
    case CL_GL_CONTEXT_KHR:
        if (kHsaStatusSuccess != hsacoreapi->HsaEndGLInterop(
            _bkendDevice, reinterpret_cast<GLvoid *>(gfxContext))) {
            LogError("Failed HsaEndGLInterop()");
            return false;
        }
        break;
    default:
        LogError("Unknown external device!");
        return false;
    }

    return true;
}

device::Memory*
Device::createMemory(amd::Memory &owner) const
{
    oclhsa::Memory* memory = NULL;

    if (owner.asBuffer()) {
        memory = new oclhsa::Buffer(*this, owner);
    }
    else if (owner.asImage()) {
        memory = new oclhsa::Image(*this, owner);
    }
    else {
        LogError("Unknown memory type");
    }

    if (memory == NULL) {
      return NULL;
    }

    bool result = false;
    if (owner.isInterop() && (owner.parent() == NULL)) {
        result = memory->createInterop();
    }
    else {
        result = memory->create();
    }

    if (!result) {
        delete memory;
        return NULL;
    }

    if (!memory->isHostMemDirectAccess() && owner.asImage() &&
                 owner.parent() == NULL &&
                 (owner.getMemFlags() &
                    (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR))) {
        // To avoid recurssive call to Device::createMemory, we perform
        // data transfer to the view of the image.
        amd::Image *imageView =
            owner.asImage()->createView(
                owner.getContext(), owner.asImage()->getImageFormat(), xferQueue());

        if (imageView == NULL) {
            LogError("[OCL] Fail to allocate view of image object");
            return NULL;
        }

        Image* devImageView =
            new oclhsa::Image(static_cast<const Device &>(*this), *imageView);
        if (devImageView == NULL) {
            LogError("[OCL] Fail to allocate device mem object for the view");
            imageView->release();
            return NULL;
        }

        if (devImageView != NULL &&
            !devImageView->createView(static_cast<oclhsa::Image &>(*memory))) {
            LogError("[OCL] Fail to create device mem object for the view");
            delete devImageView;
            imageView->release();
            return NULL;
        }

        imageView->replaceDeviceMemory(this, devImageView);

        result = xferMgr().writeImage(
                    owner.getHostMem(),
                    *devImageView, 
                    amd::Coord3D(0),
                    imageView->getRegion(),
                    imageView->getRowPitch(),
                    imageView->getSlicePitch(),
                    true);

        imageView->release();
    }

    if (!result) {
        delete memory;
        return NULL;
    }

    return memory;
}

void*
Device::hostAlloc(size_t size, size_t alignment, bool atomics) const
{
    void* ret;
    alignment = std::max(alignment, static_cast<size_t>(info_.memBaseAddrAlign_));
    assert(amd::isMultipleOf(alignment, info_.memBaseAddrAlign_));
    HsaAmdSystemMemoryType type = amd::Is64Bits() && atomics
        ? kHsaAmdSystemMemoryTypeCoherent : kHsaAmdSystemMemoryTypeDefault;
    hsacoreapi->HsaAmdAllocateSystemMemory(size, alignment, type, &ret);
    return ret;
}

void
Device::hostFree(void* ptr, size_t size) const
{
    hsacoreapi->HsaAmdFreeSystemMemory(ptr);
}

void*
Device::svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags, void* svmPtr) const
{
    bool atomics = (flags & CL_MEM_SVM_ATOMICS) != 0;
    return hostAlloc(size, alignment, atomics);
}

void
Device::svmFree(void* ptr) const
{
    hostFree(ptr);
}

VirtualGPU*
Device::xferQueue() const
{
    if (!xferQueue_) {
        // Create virtual device for internal memory transfer
        Device* thisDevice = const_cast<Device*>(this);
        thisDevice->xferQueue_ = reinterpret_cast<VirtualGPU*>(
                thisDevice->createVirtualDevice());
        if (!xferQueue_) {
            LogError("Couldn't create the device transfer manager!");
        }
    }
    return xferQueue_;
}

}
#endif  // WITHOUT_FSA_BACKEND
