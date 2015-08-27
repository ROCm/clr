#include "gsl_ctx.h"
#include "GSLDevice.h"
#include "EventQueue.h"
#include "ini_export.h"
#include "GSLContext.h"
#include "cm_if.h"
#include "utils/flags.hpp"
#include "query/QueryObject.h"
#include "memory/MemObject.h"
#include "sampler/SamplerObject.h"
#include "texture/TextureResourceObject.h"

extern gslMemObjectAttribTiling g_CALBETiling_Tiled;

void
CALGSLDevice::resFree(gslMemObject mem) const
{
    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());
    m_cs->destroyMemObject(mem);
}

void CALGSLDevice::Initialize()
{
    m_adp = 0;
    m_cs = 0;
    m_rs = 0;
    m_textureResource = 0;
    m_textureSampler = 0;
    m_target = (CALtarget)0xffffffff;
    m_srcDRMDMAMem = NULL ;
    m_dstDRMDMAMem = NULL ;

    m_nativeDisplayHandle = NULL;
    m_deviceMode = GSL_DEVICE_MODE_GFX;

    m_gpuIndex = 0;
    m_usePerVPUAdapterModel = false;
    m_chainIndex = 0;
    m_vpuMask = 1;
    m_PerformLazyDeviceInit = false;
    m_computeRing = false;
    m_isComputeRingIDForced = false;
    m_forcedComputeEngineID = GSL_ENGINEID_INVALID;
    gslDeviceOps_ = NULL;
}

CALGSLDevice::CALGSLDevice()
{
    Initialize();
}

CALGSLDevice::~CALGSLDevice()
{
    assert(m_adp == 0);    /// CALBE client must call close explicitly. Check that here

    delete gslDeviceOps_;

    switch(m_deviceMode)
    {
    case GSL_DEVICE_MODE_NONE:
    case GSL_DEVICE_MODE_CONSOLE:
        delete static_cast<char*>(m_nativeDisplayHandle);
        break;
    case GSL_DEVICE_MODE_GFX:
        break;
    }
}


gsl::gsAdaptor*
CALGSLDevice::getNative() const
{
    return m_adp;
}

uint32
CALGSLDevice::getMaxTextureSize() const
{
    return static_cast<uint32>(m_maxtexturesize);
}

void
CALGSLDevice::getAttribs_int(gsl::gsCtx* cs)
{
    m_attribs.struct_size = sizeof(CALdeviceattribs);

    m_attribs.target = m_target;

    gslMemInfo memInfo;
    cs->getMemInfo(&memInfo, GSL_MEMINFO_BASIC);

    m_attribs.localRAM = (uint32)((memInfo.cardMemTotalBytes + memInfo.cardExtMemTotalBytes) / (1024 * 1024));
    m_attribs.uncachedRemoteRAM = (uint32)(memInfo.agpMemTotalBytes / (1024 * 1024));
    m_attribs.cachedRemoteRAM = (uint32)(memInfo.agpMemTotalCacheableBytes / (1024 * 1024));
    m_attribs.totalVisibleHeap = (uint32) (memInfo.cardMemTotalBytes / (1024 * 1024));
    m_attribs.totalInvisibleHeap = (uint32) (memInfo.cardExtMemTotalBytes / (1024 * 1024));
    m_attribs.totalDirectHeap = (uint32) (memInfo.directTotalBytes / (1024 * 1024));
    m_attribs.totalCoherentHeap = (uint32) (memInfo.coherentTotalBytes / (1024 * 1024));
    m_attribs.totalRemoteSharedHeap = (uint32) (memInfo.sharedTotalBytes / (1024 * 1024));
    m_attribs.totalCachedRemoteSharedHeap = (uint32) (memInfo.sharedCacheableTotalBytes / (1024 * 1024));
    m_attribs.totalSDIHeap = (uint32) (memInfo.busAddressableTotalBytes / (1024 * 1024));

    m_attribs.engineClock = cs->getMaxEngineClock();
    m_attribs.memoryClock = cs->getMaxMemoryClock();
    m_attribs.numberOfSIMD = cs->getNumSIMD();
    m_attribs.numberOfCUsperShaderArray = cs->getNumCUsPerShaderArray();
    m_attribs.wavefrontSize = cs->getWaveFrontSize();
    m_attribs.doublePrecision = cs->getIsDoublePrecisionSupported();
    m_attribs.memBusWidth = cs->getVramBitWidth();
    m_attribs.numMemBanks = cs->getVramBanks();
    m_attribs.isWorkstation = cs->getIsWorkstation();

    m_attribs.numberOfShaderEngines = cs->getNumShaderEngines();
    m_attribs.pciTopologyInformation = m_adp->getLocationId();

    const uint8* boardName = cs->getString(GSL_GS_RENDERER);
    ::strncpy(m_attribs.boardName, (char*)boardName, CAL_ASIC_INFO_MAX_LEN * sizeof(char));

    m_attribs.counterFreq = cs->getCounterFreq();
    m_attribs.nanoSecondsPerTick = 1000000000.0 / cs->getCounterFreq();
    m_attribs.longIdleDetect = cs->getLongIdleDetect();
    m_attribs.svmAtomics = m_adp->pAsicInfo->svmAtomics;

    m_attribs.vaStart = static_cast<CALuint64>(m_adp->pAsicInfo->vaStart);
    m_attribs.vaEnd   = static_cast<CALuint64>(m_adp->pAsicInfo->vaEnd);
    m_attribs.numOfVpu = m_adp->pAsicInfo->numberOfVPU;
    m_attribs.isOpenCL200Device = m_adp->pAsicInfo->bIsOpen2Device;
    m_attribs.isSVMFineGrainSystem = m_adp->pAsicInfo->svmFineGrainSystem;
    m_attribs.isWDDM2Enabled = m_adp->pAsicInfo->vaAvailable && m_adp->pAsicInfo->bNoVATranslation;
}

bool
CALGSLDevice::open(uint32 gpuIndex, bool enableHighPerformanceState, bool reportAsOCL12Device)
{
    gslDeviceOps_ = new amd::Monitor("GSL Device Ops Lock", true);
    if (NULL == gslDeviceOps_) {
        return false;
    }

    unsigned int chainIndex = 0;
#ifdef ATI_OS_WIN
    m_gpuIndex = gpuIndex;
    m_usePerVPUAdapterModel = true;
    m_PerformLazyDeviceInit = true;
#else
    void * nativeHandle;
    gslDeviceMode deviceMode;
    gsAdaptor::getDeviceInitData(gpuIndex, &deviceMode, &chainIndex, &nativeHandle);

    m_nativeDisplayHandle = nativeHandle;
    m_deviceMode = deviceMode;
#endif
    m_chainIndex = chainIndex;
    m_vpuMask  = 1 << chainIndex;

    //
    // CALBE is required to explicitly manage multiple opens and closes
    // assert on the condition for correct usage
    //
    assert(m_adp == 0);

    memset(&m_dcfg, 0, sizeof(m_dcfg));

    extern void getConfigFromFile(gslStaticRuntimeConfig &scfg, gslDynamicRuntimeConfig &dcfg);
    getConfigFromFile(m_scfg, m_dcfg);

    m_scfg.UsePerVPUAdapterModel.hasValue = true;
    m_scfg.UsePerVPUAdapterModel.value = m_usePerVPUAdapterModel;

    m_scfg.DX10SamplerResources.hasValue = true;
    m_scfg.DX10SamplerResources.value = true;

    m_scfg.vpuMask.hasValue = true;
    m_scfg.vpuMask.value = m_vpuMask;

    m_scfg.bEnableHighPerformanceState.hasValue = true;
    m_scfg.bEnableHighPerformanceState.value = enableHighPerformanceState;

    m_dcfg.disableMarkUsedInCmdBuf.hasValue = true;
    m_dcfg.disableMarkUsedInCmdBuf.value    = false;

    // Enable immediate memory release
    m_dcfg.immediateMemoryRelease.hasValue = true;
    m_dcfg.immediateMemoryRelease.value    = true;

    m_dcfg.bEnableSvm.hasValue = true;
    m_dcfg.bEnableSvm.value    = reportAsOCL12Device ? false : OPENCL_MAJOR >= 2;

    m_dcfg.bEnableFlatAddressing.hasValue = true;
#if defined(ATI_BITS_32) && defined(ATI_OS_LINUX)
    m_dcfg.bEnableFlatAddressing.value    = false;
#else
    m_dcfg.bEnableFlatAddressing.value    = reportAsOCL12Device ? false : (OPENCL_MAJOR >= 2);
#endif

    //we can use environment variable CAL_ENABLE_ASYNC_DMA to force dma on or off when we need it
    char *s = NULL;
    if((s = getenv("CAL_ENABLE_ASYNC_DMA")))
    {
        m_dcfg.drmdmaMode.hasValue = true;
        m_dcfg.drmdmaMode.value = (atoi(s) == 0) ? GSL_CONFIG_DRMDMA_MODE_FORCE_OFF : GSL_CONFIG_DRMDMA_MODE_DEFAULT;
    }

    // Use GPU_USE_SYNC_OBJECTS to force syncobject on or off when we need it
    m_dcfg.syncObjectMode.hasValue = true;
    m_dcfg.syncObjectMode.value = (GPU_USE_SYNC_OBJECTS) ?
        GSL_CONFIG_SYNCOBJECT_MODE_ON : GSL_CONFIG_SYNCOBJECT_MODE_OFF;

    // Use GPU_IFH_MODE to test with IFH mode enabled
    m_dcfg.DropFlush.hasValue = true;
    m_dcfg.DropFlush.value = (GPU_IFH_MODE == 1);

    // Enable TC compatible htile mode. It's HW feature for VI+ and controlled in HWL.
    // Depth interop doesn't support TC compatible htile mode, but OCL needs correct tiling setup.
    m_dcfg.bEnableTCCompatibleHtile.hasValue = true;
    m_dcfg.bEnableTCCompatibleHtile.value = true;

    int32 asic_id = 0;
    if (!SetupAdapter(asic_id))
    {
        return false;
    }

    if (!SetupContext(asic_id))
    {
        return false;
    }

    if (m_PerformLazyDeviceInit)
    {
        // close the adaptor
        gsAdaptor::closeAdaptor(m_adp);
        m_adp = 0;
    }
    else
    {
        PerformFullInitialization();
    }

    return true;
}

void
CALGSLDevice::close()
{
    if (m_cs != NULL)
    {
        m_cs->Flush();
    }

    if (m_dstDRMDMAMem)
    {
        resFree(m_dstDRMDMAMem);
        m_dstDRMDMAMem = NULL ;
    }
    if (m_srcDRMDMAMem)
    {
        resFree(m_srcDRMDMAMem);
        m_srcDRMDMAMem = NULL ;
    }

    if (m_cs != NULL)
    {
        m_cs->destroyTextureResource(m_textureResource);
        m_cs->destroySampler(m_textureSampler);
        m_cs->destroyQuery(m_mapQuery);
        m_cs->destroyQuery(m_mapDMAQuery);

        m_cs->setRenderState(0);
        m_cs->destroyRenderState(m_rs);
        m_cs->destroySubAllocDesc();
        m_rs = 0;

        m_adp->deleteContext(m_cs);

        m_cs = 0;
    }

    if (m_adp != NULL)
    {
        gsAdaptor::closeAdaptor(m_adp);
        m_adp = 0;
    }
}

void
CALGSLDevice::PerformAdapterInitialization() const
{
    CALGSLDevice* mutable_this = const_cast<CALGSLDevice*>(this);
    mutable_this->PerformAdapterInitialization_int(false);
}

void
CALGSLDevice::PerformFullInitialization() const
{
    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    CALGSLDevice* mutable_this = const_cast<CALGSLDevice*>(this);
    mutable_this->PerformFullInitialization_int();
}

bool
CALGSLDevice::SetupAdapter(int32 &asic_id)
{
    PerformAdapterInitialization_int(true);

    if (m_adp == 0)
    {
        return false;
    }

    asic_id = m_adp->getAsicID();

    if ((asic_id < GSL_ATIASIC_ID_TAHITI_P) ||
        (asic_id == GSL_ATIASIC_ID_DEVASTATOR) ||
        (asic_id == GSL_ATIASIC_ID_SCRAPPER))
    {
        LogPrintfInfo("Unsupported legacy ASIC(%d) found!\n", asic_id);
        // close the adaptor
        gsAdaptor::closeAdaptor(m_adp);
        m_adp = 0;
        return false;
    }

    bool hasDmaEngine = m_adp->findDMAEngine();
    bool hasComputeEngine = m_adp->findComputeEngine();

    m_canDMA = hasDmaEngine;

    m_adp->queryAvailableEngines(&m_nEngines, m_engines);

    //Disable DRMDMA on CFX mode for linux on all GPUs.
#ifdef ATI_OS_LINUX
    if ((m_adp->getNumLinkedVPUs() > 1) && !DRMDMA_FOR_LNX_CF)
    {
        m_canDMA = ATIGL_FALSE;
    }
#endif

    if (asic_id < GSL_ATIASIC_ID_TAHITI_P ||
        asic_id == GSL_ATIASIC_ID_DEVASTATOR ||
        asic_id == GSL_ATIASIC_ID_SCRAPPER)
    {
        m_computeRing = false;
    }
    else
    {
        m_computeRing = true;
    }

    if (!flagIsDefault(GPU_NUM_COMPUTE_RINGS))
    {
        m_computeRing = (GPU_NUM_COMPUTE_RINGS != 0);
    }

    if ((!flagIsDefault(GPU_SELECT_COMPUTE_RINGS_ID)) && (m_computeRing))
    {
        gslEngineID engineID;
        engineID = static_cast<gslEngineID>(GPU_SELECT_COMPUTE_RINGS_ID + GSL_ENGINEID_COMPUTE0);
        if ((engineID >= GSL_ENGINEID_COMPUTE0) && (engineID <= GSL_ENGINEID_COMPUTE7))
        {
            for (uint i = 0; i < m_nEngines; ++i) {
                if (m_engines[i].id == engineID){
                    m_isComputeRingIDForced = true;
                    m_forcedComputeEngineID = engineID;
                    break;
                }
            }
        }
    }

    if (m_computeRing && !hasComputeEngine)
    {
        return false;
    }

    return true;
}

bool
CALGSLDevice::SetupContext(int32 &asic_id)
{
    gsl::gsCtx* temp_cs = m_adp->createComputeContext(m_computeRing ? (m_isComputeRingIDForced ? m_forcedComputeEngineID :
                                                     getFirstAvailableComputeEngineID()) : GSL_ENGINEID_3DCOMPUTE0,
                                                     m_canDMA ? GSL_ENGINEID_DRMDMA0 : GSL_ENGINEID_INVALID, true);
    temp_cs->getMainSubCtx()->setVPUMask(m_vpuMask);

    m_maxtexturesize = temp_cs->getMaxTextureSize();

    switch (asic_id)
    {
    case GSL_ATIASIC_ID_R870:
        m_target = CAL_TARGET_CYPRESS;
        m_elfmachine = ED_ATI_CAL_MACHINE_CYPRESS_ISA;
        break;
    case GSL_ATIASIC_ID_R830:
        m_target = CAL_TARGET_JUNIPER;
        m_elfmachine = ED_ATI_CAL_MACHINE_JUNIPER_ISA;
        break;
    case GSL_ATIASIC_ID_REDWOOD:
        m_target = CAL_TARGET_REDWOOD;
        m_elfmachine = ED_ATI_CAL_MACHINE_REDWOOD_ISA;
        break;
    case GSL_ATIASIC_ID_CEDAR:
        m_target = CAL_TARGET_CEDAR;
        m_elfmachine = ED_ATI_CAL_MACHINE_CEDAR_ISA;
        break;
    case GSL_ATIASIC_ID_CAYMAN:
        m_target = CAL_TARGET_CAYMAN;
        m_elfmachine = ED_ATI_CAL_MACHINE_CAYMAN_ISA;
        break;
    case GSL_ATIASIC_ID_BARTS:
        m_target = CAL_TARGET_BARTS;
        m_elfmachine = ED_ATI_CAL_MACHINE_BARTS_ISA;
        break;
    case GSL_ATIASIC_ID_TURKS:
        m_target = CAL_TARGET_TURKS;
        m_elfmachine = ED_ATI_CAL_MACHINE_TURKS_ISA;
        break;
    case GSL_ATIASIC_ID_CAICOS:
        m_target = CAL_TARGET_CAICOS;
        m_elfmachine = ED_ATI_CAL_MACHINE_CAICOS_ISA;
        break;
    case GSL_ATIASIC_ID_SUMO:
        m_target = CAL_TARGET_SUMO;
        m_elfmachine = ED_ATI_CAL_MACHINE_SUMO_ISA;
        break;
    case GSL_ATIASIC_ID_SUPERSUMO:
        m_target = CAL_TARGET_SUPERSUMO;
        m_elfmachine = ED_ATI_CAL_MACHINE_SUPERSUMO_ISA;
        break;
    case GSL_ATIASIC_ID_WRESTLER:
        m_target = CAL_TARGET_WRESTLER;
        m_elfmachine = ED_ATI_CAL_MACHINE_WRESTLER_ISA;
        break;
    case GSL_ATIASIC_ID_TAHITI_P:
        m_target = CAL_TARGET_TAHITI;
        m_elfmachine = ED_ATI_CAL_MACHINE_TAHITI_ISA;
        break;
    case GSL_ATIASIC_ID_PITCAIRN_PM:
        m_target = CAL_TARGET_PITCAIRN;
        m_elfmachine = ED_ATI_CAL_MACHINE_PITCAIRN_ISA;
        break;
    case GSL_ATIASIC_ID_CAPEVERDE_M:
        m_target = CAL_TARGET_CAPEVERDE;
        m_elfmachine = ED_ATI_CAL_MACHINE_CAPEVERDE_ISA;
        break;
    case GSL_ATIASIC_ID_DEVASTATOR:
        m_target = CAL_TARGET_DEVASTATOR;
        m_elfmachine = ED_ATI_CAL_MACHINE_DEVASTATOR_ISA;
        break;
    case GSL_ATIASIC_ID_SCRAPPER:
        m_target = CAL_TARGET_SCRAPPER;
        m_elfmachine = ED_ATI_CAL_MACHINE_SCRAPPER_ISA;
        break;
    case GSL_ATIASIC_ID_OLAND_M:
        m_target = CAL_TARGET_OLAND;
        m_elfmachine = ED_ATI_CAL_MACHINE_OLAND_ISA;
        break;
    case GSL_ATIASIC_ID_HAINAN_M:
        m_target = CAL_TARGET_HAINAN;
        m_elfmachine = ED_ATI_CAL_MACHINE_HAINAN_ISA;
        break;
    case GSL_ATIASIC_ID_BONAIRE_M:
        m_target = CAL_TARGET_BONAIRE;
        m_elfmachine = ED_ATI_CAL_MACHINE_BONAIRE_ISA;
        break;
    case GSL_ATIASIC_ID_SPECTRE:
        m_target = CAL_TARGET_SPECTRE;
        m_elfmachine = ED_ATI_CAL_MACHINE_SPECTRE_ISA;
        break;
    case GSL_ATIASIC_ID_SPOOKY:
        m_target = CAL_TARGET_SPOOKY;
        m_elfmachine = ED_ATI_CAL_MACHINE_SPOOKY_ISA;
        break;
    case GSL_ATIASIC_ID_KALINDI:
        m_target = CAL_TARGET_KALINDI;
        m_elfmachine = ED_ATI_CAL_MACHINE_KALINDI_ISA;
        break;
    case GSL_ATIASIC_ID_HAWAII_P:
        m_target = CAL_TARGET_HAWAII;
        m_elfmachine = ED_ATI_CAL_MACHINE_HAWAII_ISA;
        break;
    case GSL_ATIASIC_ID_ICELAND_M:
        m_target = CAL_TARGET_ICELAND;
        m_elfmachine = ED_ATI_CAL_MACHINE_ICELAND_ISA;
        break;
    case GSL_ATIASIC_ID_TONGA_P:
        m_target = CAL_TARGET_TONGA;
        m_elfmachine = ED_ATI_CAL_MACHINE_TONGA_ISA;
        break;
    case GSL_ATIASIC_ID_GODAVARI:
        m_target = CAL_TARGET_GODAVARI;
        m_elfmachine = ED_ATI_CAL_MACHINE_GODAVARI_ISA;
        break;
    case GSL_ATIASIC_ID_FIJI_P:
        m_target = CAL_TARGET_FIJI;
        m_elfmachine = ED_ATI_CAL_MACHINE_FIJI_ISA;
        break;
    case GSL_ATIASIC_ID_CARRIZO:
        m_target = CAL_TARGET_CARRIZO;
        m_elfmachine = ED_ATI_CAL_MACHINE_CARRIZO_ISA;
        break;
    case GSL_ATIASIC_ID_ELLESMERE:
        m_target = CAL_TARGET_ELLESMERE;
        m_elfmachine = ED_ATI_CAL_MACHINE_ELLESMERE_ISA;
        break;
    case GSL_ATIASIC_ID_BAFFIN:
        m_target = CAL_TARGET_BAFFIN;
        m_elfmachine = ED_ATI_CAL_MACHINE_BAFFIN_ISA;
        break;
  case GSL_ATIASIC_ID_GREENLAND:
        m_target = CAL_TARGET_GREENLAND;
        m_elfmachine = ED_ATI_CAL_MACHINE_GREENLAND_ISA;
        break;
  case GSL_ATIASIC_ID_STONEY:
        m_target = CAL_TARGET_STONEY;
        m_elfmachine = ED_ATI_CAL_MACHINE_STONEY_ISA;
        break;
    default:
        // 6XX is not supported
        m_adp->deleteContext(temp_cs);
        gsAdaptor::closeAdaptor(m_adp);
        m_adp = 0;

        assert(0);
        return false;
    }

    //cache device details
    getAttribs_int(temp_cs);
    temp_cs->getMemInfo(&m_memInfo, GSL_MEMINFO_BASIC);

    m_vmMode = temp_cs->getVMMode();

    m_adp->deleteContext(temp_cs);

    return true;
}

void
CALGSLDevice::PerformAdapterInitialization_int(bool initLite)
{
    if (m_adp == 0)
    {
        if (m_usePerVPUAdapterModel)
        {
            m_adp = gsAdaptor::openAdaptorByIndex<gsl::gsAdaptor>(m_gpuIndex, &m_scfg, &m_dcfg, initLite);
        }
        else
        {
            m_adp = gsAdaptor::openAdaptor(m_nativeDisplayHandle, m_chainIndex, &m_scfg, &m_dcfg);
        }

        assert(m_adp != 0);
    }
}

void
CALGSLDevice::PerformFullInitialization_int()
{
    if (m_adp == 0)
    {
        PerformAdapterInitialization_int(false);
    }

    if (m_cs == 0)
    {
        m_cs = m_adp->createComputeContext(m_computeRing ? (m_isComputeRingIDForced ? m_forcedComputeEngineID :
                                          getFirstAvailableComputeEngineID()) : GSL_ENGINEID_3DCOMPUTE0,
                                          m_canDMA ? GSL_ENGINEID_DRMDMA0 : GSL_ENGINEID_INVALID, false);
        m_cs->getMainSubCtx()->setVPUMask(m_vpuMask);

        //
        // Check if the command stream has a DMA connection and allow DMA if there
        // is a connection and we can actually DMA
        //
        bool dmaConnection = m_cs->getDrmDma0Ctx() && m_cs->getDrmDma0Ctx()->ioInfo.iolConnection;
        m_allowDMA = (dmaConnection && m_canDMA);

        m_rs = m_cs->createRenderState();
        m_cs->setRenderState(m_rs);

        m_cs->Flush();

        m_cs->createSubAllocDesc();

        m_mapQuery = m_cs->createQuery(GSL_SYNC_ATI);
        m_mapDMAQuery = m_cs->createQuery(GSL_DRMDMA_SYNC_ATI);

        // Allocate 1x1 FART and Vid memory for DMA flush
        CALresourceDesc desc;
        memset(&desc, 0, sizeof(CALresourceDesc));
        desc.type           = GSL_MOA_MEMORY_AGP;
        desc.size.width     = 1;
        desc.size.height    = 1;
        desc.format         = CM_SURF_FMT_R32F;
        desc.channelOrder   = GSL_CHANNEL_ORDER_R;
        desc.dimension      = GSL_MOA_TEXTURE_2D;
        m_srcDRMDMAMem      = resAlloc(&desc);

        desc.type           = GSL_MOA_MEMORY_CARD_EXT_NONEXT;
        m_dstDRMDMAMem      = resAlloc(&desc);

        m_cs->setDMAFlushBuf(m_srcDRMDMAMem, m_dstDRMDMAMem, 4 /*size of CM_SURF_FMT_R32F*/);

        m_PerformLazyDeviceInit = false;

        m_uavInCB = m_cs->getIsUAVInCB();
        m_textureResource = m_cs->createTextureResource();
        m_textureSampler = m_cs->createSampler();
    }
}

void
Wait(gsl::gsCtx* cs, gslQueryTarget target, gslQueryObject object)
{
    uint64 param;

    uint32 mask = (target == GSL_DRMDMA_SYNC_ATI) ? GSL_ENGINE_MASK(GSL_ENGINEID_DRMDMA0) | GSL_ENGINE_MASK(GSL_ENGINEID_DRMDMA1) : GSL_ENGINEMASK_ALL_BUT_UVD_VCE;

    object->BeginQuery(cs, target, 0, mask);
    object->EndQuery(cs, 0);
    object->GetResult(cs, &param);

    assert(param == 1);
}

bool
CALGSLDevice::ResolveAperture(const gslMemObjectAttribTiling tiling) const
{
    // Don't ask for aperture if the tiling is linear.
    if ((GSL_MOA_TILING_LINEAR == tiling) ||
        (GSL_MOA_TILING_LINEAR_GENERAL == tiling))
    {
        return false;
    }

    // Use aperture.
    return true;
}

gslMemObject
CALGSLDevice::resAlloc(const CALresourceDesc* desc) const
{
    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    assert(m_cs != 0);
    gslMemObject    mem = 0;
    uint32  flags = desc->flags;

    gslMemObjectAttribs attribs(
        GSL_MOA_TEXTURE_1D,      // type
        GSL_MOA_MEMORY_CARD_EXT_NONEXT,     // location  XXX
        (flags & CAL_RESALLOC_GLOBAL_BUFFER) ? GSL_MOA_TILING_LINEAR : g_CALBETiling_Tiled,     // tiling
        GSL_MOA_DISPLAYABLE_NO,  // displayable
        ATIGL_FALSE,             // mipmap
        1,                       // samples
        0,                       // cpu_address
        GSL_MOA_SIGNED_NO,       // signed_format
        GSL_MOA_FORMAT_DERIVED,  // numFormat
        DRIVER_MODULE_GLL,       // module
        GSL_ALLOCATION_INSTANCED // alloc_type
    );

    attribs.location = desc->type;
    attribs.vaBase   = desc->vaBase;
    attribs.section = desc->section;
    attribs.isAllocExecute = desc->isAllocExecute;
    attribs.minAlignment = desc->minAlignment;

    //!@note GSL asserts with tiled 1D images of any type.
    if ((desc->dimension == GSL_MOA_BUFFER) ||
        (desc->dimension == GSL_MOA_TEXTURE_1D) ||
        (desc->dimension == GSL_MOA_TEXTURE_1D_ARRAY) ||
        (desc->dimension == GSL_MOA_TEXTURE_BUFFER))
    {
        attribs.tiling  = GSL_MOA_TILING_LINEAR;
    }

    if (desc->type == GSL_MOA_MEMORY_SYSTEM)
    {
        // CPU addres and size for pinning
        attribs.cpu_address = desc->systemMemory;
        attribs.size = desc->systemMemorySize;

        if ((desc->size.width % 64) == 0)
        {
            attribs.tiling = GSL_MOA_TILING_LINEAR;
        }
        else
        {
            // Use linear general if width isn't aligned
            attribs.tiling = GSL_MOA_TILING_LINEAR_GENERAL;
        }
    }
    else if (desc->type == GSL_MOA_MEMORY_CARD_EXTERNAL_PHYSICAL)
    {
        attribs.cpu_address = (void*)desc->busAddress;
    }

    // Don't ask for aperture if the tiling is linear.
    attribs.useAperture = ResolveAperture(attribs.tiling);

    attribs.channelOrder = desc->channelOrder;
    attribs.type = desc->dimension;
    if (desc->mipLevels > 1) {
        attribs.levels = desc->mipLevels;
        attribs.mipmap = true;
    }
    switch (desc->dimension)
    {
        case GSL_MOA_BUFFER:
            mem = m_cs->createMemObject1D(desc->format, desc->size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_1D:
            mem = m_cs->createMemObject1D(desc->format, desc->size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_2D:
            mem = m_cs->createMemObject2D(desc->format, desc->size.width, (uint32)desc->size.height, &attribs);
            break;
        case GSL_MOA_TEXTURE_3D:
            mem = m_cs->createMemObject3D(desc->format, desc->size.width,
                (uint32)desc->size.height, (uint32)desc->size.depth, &attribs);
            break;
        case GSL_MOA_TEXTURE_BUFFER:
            attribs.type = GSL_MOA_TEXTURE_BUFFER;
            mem = m_cs->createMemObject1D(desc->format, desc->size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_1D_ARRAY:
            mem = m_cs->createMemObject3D(desc->format, desc->size.width,
                1, (uint32)desc->size.height, &attribs);
            break;
        case GSL_MOA_TEXTURE_2D_ARRAY:
            mem = m_cs->createMemObject3D(desc->format, desc->size.width,
                (uint32)desc->size.height, (uint32)desc->size.depth, &attribs);
            break;
        default:
            break;
    }

#ifdef ATI_OS_WIN
    if ((desc->section == GSL_SECTION_SVM || desc->section == GSL_SECTION_SVM_ATOMICS) && mem == NULL) {
        //svm allocation failure, try one more time after wait.
        Wait(m_cs, GSL_SYNC_ATI, m_mapQuery);
        mem = m_cs->createMemObject1D(desc->format, desc->size.width, &attribs);
    }
#endif

    return mem;
}

gslMemObject
CALGSLDevice::resAllocView(gslMemObject res, gslResource3D size, size_t offset, cmSurfFmt format,
    gslChannelOrder channelOrder, gslMemObjectAttribType resType, uint32 level, uint32 layer, uint32 flags,
    uint64 bytePitch) const
{
    assert(m_cs != 0);

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    gslMemObjectAttribs attribs(
        GSL_MOA_TEXTURE_2D,      // type. Filled in below based on the base type.
        GSL_MOA_MEMORY_ALIAS,    // location. Filled in below based on the base location.
        GSL_MOA_TILING_LINEAR, // tiling. Filled in below based on the flags passed in.
        GSL_MOA_DISPLAYABLE_NO,  // displayable
        ATIGL_FALSE,             // mipmap
        1,                       // samples
        0,                       // cpu_address
        GSL_MOA_SIGNED_NO,       // signed_format
        GSL_MOA_FORMAT_DERIVED,  // numFormat
        DRIVER_MODULE_GLL,       // module
        GSL_ALLOCATION_INSTANCED // alloc_type
    );
    attribs.bytePitch = bytePitch;
    attribs.section = res->getAttribs().section;
    attribs.isAllocExecute = res->getAttribs().isAllocExecute;

    // Need to get the alignment info from hwl.
    // Not sure hwl is correct though. Linear aligned 256b, tiled 8kb according to the address library.
    uint32 alignment;
    switch (flags & ~CAL_RESALLOCSLICEVIEW_LEVEL_AND_LAYER)
    {
    case CAL_RESALLOCSLICEVIEW_LINEAR_ALIGNED:
        alignment = 256;
        attribs.tiling = GSL_MOA_TILING_LINEAR;
        break;
    case CAL_RESALLOCSLICEVIEW_LINEAR_UNALIGNED:
        alignment = 1;
        attribs.tiling = GSL_MOA_TILING_LINEAR_GENERAL;
        break;
    default:
        alignment = 8192;
        // GSL asserts if this tiled mode is differnt from the original surface.
        // (For example, original is GSL_MOA_TILING_MACRO and the new one is GSL_MOA_TILING_TILED)
        // Use the original mode for view allocation.
        attribs.tiling = res->getAttribs().tiling;
        if (attribs.tiling == GSL_MOA_TILING_LINEAR || attribs.tiling == GSL_MOA_TILING_LINEAR_GENERAL)
        {
            alignment = 256;
        }
        break;
    };

    // Check any alignment restrictions.
    uint64 resPitch = res->getPitch();
    cmSurfFmt baseFormat = res->getFormat();
    uint32 elementSize = cmGetSurfElementSize(static_cast<cmSurfFmt>(baseFormat));
    uint64 offsetInBytes = static_cast<uint64>(offset) * elementSize;
    if (offsetInBytes % alignment)
    {
        return 0; //offset doesn't match alignment requirements.
    }

    // alias has same location as the base resource.
    attribs.type         = res->getAttribs().type;
    attribs.location     = res->getAttribs().location;
    attribs.displayable  = res->getAttribs().displayable;
    attribs.channelOrder = channelOrder;

    gslMemObject mo = NULL, levelobject = res;

    bool levelLayer = false;
    if (flags & CAL_RESALLOCSLICEVIEW_LEVEL)
    {
        const gsSubImageParam levelParam(level);
        levelobject = m_cs->createSubMemObject(res, GSL_LEVEL, levelParam);
        attribs.bytePitch = static_cast<size_t>(levelobject->getPitch()) *
            (levelobject->getBitsPerElement() / 8);
        levelLayer = true;
    }
    if (flags & CAL_RESALLOCSLICEVIEW_LAYER)
    {
        const gsSubImageParam layerParam(layer);
        mo = m_cs->createSubMemObject(levelobject, GSL_LAYER, layerParam);
        if (levelobject != res)
        {
            m_cs->destroyMemObject(levelobject);
        }
        levelobject = mo;
        levelLayer = true;
    }

    if (levelLayer) {
        // If level/layer object was created, then don't need an extra view
        return levelobject;
    }

    attribs.type = resType;
    switch (resType)
    {
        case GSL_MOA_BUFFER:
            mo = m_cs->createOffsetMemObject1D(levelobject, offsetInBytes, format,
                size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_1D:
            mo = m_cs->createOffsetMemObject1D(levelobject, offsetInBytes, format,
                size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_2D:
            mo = m_cs->createOffsetMemObject2D(levelobject, offsetInBytes, format,
                size.width, (uint32)size.height, &attribs);
            break;
        case GSL_MOA_TEXTURE_3D:
            mo = m_cs->createOffsetMemObject3D(levelobject, offsetInBytes, format,
                size.width, (uint32)size.height, (uint32)size.depth, &attribs);
            break;
        case GSL_MOA_TEXTURE_BUFFER:
            mo = m_cs->createOffsetMemObject1D(levelobject, offsetInBytes, format,
                size.width, &attribs);
            break;
        case GSL_MOA_TEXTURE_1D_ARRAY:
            mo = m_cs->createOffsetMemObject3D(levelobject, offsetInBytes, format,
                size.width, 1, (uint32)size.height, &attribs);
            break;
        case GSL_MOA_TEXTURE_2D_ARRAY:
            mo = m_cs->createOffsetMemObject3D(levelobject, offsetInBytes, format,
                size.width, (uint32)size.height, (uint32)size.depth, &attribs);
            break;
        default:
            break;
    }

    if (levelobject != res)
    {
        m_cs->destroyMemObject(levelobject);
    }

    return mo;
}

enum MemMap_DMA
{
    MemMap_DMA_None,
    MemMap_DMA_DRMDMA,
    MemMap_DMA_CPDMA
};

typedef struct _GSLDeviceMemMap_
{
    gslMemObject    mem;
    MemMap_DMA      dma;
    uint32          flags;
    bool32          lockable;
} GSLDeviceMemMap;


bool
CALGSLDevice::resMapLocal(void*&            pPtr,
                             size_t&           pitch,
                             gslMemObject      mem,
                             gslMapAccessType  flags,
                             bool              isHwDebug)
{
    assert(m_cs != 0);
    assert(mem != 0);

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    //
    // Allocate map structure for the unmap call
    //

    GSLDeviceMemMap* memMap = (GSLDeviceMemMap*)malloc(sizeof(GSLDeviceMemMap));

    if (memMap == NULL)
    {
        return false;
    }

    gslMemObjectAttribLocation location = mem->getAttribs().location;

    gslMemObject newMemDest = 0;
    gslMemObject newMemSrc = 0;
    bool needDestroy = 0;

    uint64 width = mem->getRectWidth();

    intp height = mem->getRectHeight();

    cmSurfFmt format = mem->getFormat();

    gslMemObjectAttribType   dstType   = mem->getAttribs().type;

    gslMemObjectAttribs attribsDest(
        dstType,                                        // type
        GSL_MOA_MEMORY_REMOTE_CACHEABLE,                // location
        GSL_MOA_TILING_LINEAR,                          // tiling
        GSL_MOA_DISPLAYABLE_NO,                         // displayable
        ATIGL_FALSE,                                    // mipmap
        1,                                              // samples
        0,                                              // cpu_address
        GSL_MOA_SIGNED_NO,                              // signed_format
        GSL_MOA_FORMAT_DERIVED,                         // numFormat
        DRIVER_MODULE_GLL,                              // module
        GSL_ALLOCATION_INSTANCED                        // alloc_type
    );

    attribsDest.channelOrder = mem->getAttribs().channelOrder;

    //
    // DMA 1D surfaces
    //
    MemMap_DMA dma = MemMap_DMA_DRMDMA;

    if (location == GSL_MOA_MEMORY_CARD_LOCKABLE)
    {
        //
        // direct lock
        //

        dma = MemMap_DMA_None;
        memMap->lockable = ATIGL_TRUE;

        // Get tiling mode and resolve the aperture settings.
        bool useAperture;
        gslMemObjectAttribTiling tiling = mem->getAttribs().tiling;
        useAperture = ResolveAperture(tiling);

        pPtr = mem->map(m_cs, GSL_MAP_NOSYNC, GSL_GPU_0, false, useAperture);

        if (pPtr == NULL)
        {
            free(memMap);
            return false;
        }

        //
        // obtain the pitch of the buffer
        //
        uint64 tmppitch = mem->getPitch();

        pitch = static_cast<size_t>(tmppitch);

        m_hack.insert(std::pair<gslMemObject, intp>(mem, (intp) memMap));
    }
    else
    {
        memMap->lockable = ATIGL_FALSE;

        //
        // Create the target destination buffer
        //

        memMap->mem = m_cs->createMemObject2D(format, width, (uint32)height, &attribsDest);

        if (memMap->mem == NULL)
        {
            attribsDest.location = GSL_MOA_MEMORY_AGP;
            memMap->mem = m_cs->createMemObject2D(format, width, (uint32)height, &attribsDest);
            if (memMap->mem == NULL)
            {
                free(memMap);
                return false;
            }
        }

        //
        // set the pointer to it as the return buffer
        //
        void* tmp = memMap->mem->map(m_cs, GSL_MAP_NOSYNC, GSL_GPU_0, false, false);

        if (tmp == 0)
        {
            m_cs->destroyMemObject(memMap->mem);
            free(memMap);
            return false;
        }

        pPtr = tmp;

        //
        // obtain the pitch of the temporary buffer
        //
        uint64 tmppitch = memMap->mem->getPitch();

        pitch = static_cast<size_t>(tmppitch);

        uint64 surfaceSize;

        // avoid using CPDMA, which may cause deadlock with HW Debug
        uint32 copyFlag = (isHwDebug) ? CAL_MEMCOPY_ASYNC : 0;
        CopyType copy = GetCopyType(mem, memMap->mem, 0, 0, m_allowDMA, copyFlag, surfaceSize, 0, 0);

        switch (copy)
        {
        case USE_CPDMA:
            dma = MemMap_DMA_CPDMA;
            break;

        case USE_DRMDMA:
            dma = MemMap_DMA_DRMDMA;
            break;

        default:
            dma = MemMap_DMA_None;
            break;
         }


        //
        // For write only cases, we don't care about the data
        //
        switch (dma)
        {
        case MemMap_DMA_DRMDMA:
            if (flags != GSL_MAP_WRITE_ONLY)
            {
                PerformDMACopy(mem, memMap->mem, (cmSurfFmt)format, CAL_MEMCOPY_SYNC, isHwDebug);
                //
                // Flush then wait
                //
                m_cs->Flush();
#ifdef USE_3D_SYNC
                Wait(m_cs, GSL_SYNC_ATI, m_mapQuery);
#else
                Wait(m_cs, GSL_DRMDMA_SYNC_ATI, m_mapDMAQuery);
#endif
            }
            break;

        case MemMap_DMA_CPDMA:
            memMap->mem->unmap(m_cs);
            m_cs->destroyMemObject(memMap->mem);
            memMap->mem = NULL;

            pPtr = mem->map(m_cs, flags, GSL_GPU_0, true, false);

            if (pPtr == NULL)
            {
                assert(0);
                free(memMap);
                return false;
            }
            break;

        case MemMap_DMA_None:
            assert(0);
            break;
        }

        //
        // XXX - lock free?
        m_hack.insert(std::pair<gslMemObject, intp>(mem, (intp) memMap));

        if (needDestroy)
        {
            m_cs->destroyMemObject(newMemSrc);
            m_cs->destroyMemObject(newMemDest);
        }
    }

    memMap->dma   = dma;
    memMap->flags = flags;

    return true;
}

bool
CALGSLDevice::resUnmapLocal(gslMemObject mem, bool isHwDebug)
{
    assert(m_cs != 0);

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    //
    // Find the pairing
    //

    Hack::iterator iter = m_hack.find(mem);
    if (iter == m_hack.end())
    {
        mem->unmap(m_cs);
        return true;
    }

    GSLDeviceMemMap* memMap = (GSLDeviceMemMap*)iter->second;
    m_hack.erase(iter);

    if (memMap->lockable)
    {
        //
        // direct unlock
        //
        mem->unmap(m_cs);
    }
    else
    {
        //
        // Handle the different map cases.  For readonly cases, we can forgo the
        // copy back
        //
        //
        // 770 flushes denorms to 0 during the copy. To be consistent with other platforms, we
        // alias the memory as uint32 when doing the copies.
        //

        cmSurfFmt format = mem->getFormat();

        switch (memMap->dma)
        {
        case MemMap_DMA_CPDMA:
            mem->unmap(m_cs);
            //
            // Flush then wait
            //
            m_cs->Flush();
            Wait(m_cs, GSL_SYNC_ATI, m_mapQuery);
            break;

        case MemMap_DMA_DRMDMA:
            memMap->mem->unmap(m_cs);

            if (memMap->flags != GSL_MAP_READ_ONLY)
            {
                if (PerformDMACopy(memMap->mem, mem, format, CAL_MEMCOPY_SYNC, isHwDebug) == false)
                {
                    assert(0);
                }

                //
                // Flush then wait
                //
                m_cs->Flush();
#ifdef USE_3D_SYNC
                Wait(m_cs, GSL_SYNC_ATI, m_mapQuery);
#else
                Wait(m_cs, GSL_DRMDMA_SYNC_ATI, m_mapDMAQuery);
#endif
            }
            m_cs->destroyMemObject(memMap->mem);
            break;

        case MemMap_DMA_None:
            assert(0);
            break;
        }
    }

    free(memMap);

    return true;
}

gslMemObject
CALGSLDevice::resGetHeap(size_t size) const
{
    assert(m_cs != 0);

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    gslMemObjectAttribs attribs(
        GSL_MOA_VIRTUAL_HEAP,           // type
        GSL_MOA_MEMORY_SYSTEM,          // location
        GSL_MOA_TILING_LINEAR,          // tiling
        GSL_MOA_DISPLAYABLE_NO,         // displayable
        ATIGL_FALSE,                    // mipmap
        1,                              // samples
        0,                              // cpu_address
        GSL_MOA_SIGNED_NO,              // signed_format
        GSL_MOA_FORMAT_DERIVED,         // numFormat
        DRIVER_MODULE_GLL,              // module
        GSL_ALLOCATION_INSTANCED,       // alloc_type
        0,                              // channel_order
        0                               // size of cpu_address
    );

    gslMemObject rval = m_cs->createMemObject1D(CM_SURF_FMT_R32I, size, &attribs);

    return rval;
}

bool
CALGSLDevice::resMapRemote(void*& pPtr,
                               size_t& pitch,
                               gslMemObject mem,
                               gslMapAccessType flags) const
{
    assert(m_cs != 0);
    assert(mem != 0);

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    pPtr = mem->map(m_cs, GSL_MAP_NOSYNC, GSL_GPU_0, false, false);
    if (pPtr == NULL)
    {
        return false;
    }

    uint64 tmppitch = mem->getPitch();

    pitch = static_cast<size_t>(tmppitch);

    return true;
}

bool
CALGSLDevice::resUnmapRemote(gslMemObject mem) const
{
    assert(m_cs != 0);

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    mem->unmap(m_cs);

    return true;
}

bool
CALGSLDevice::PerformDMACopy(gslMemObject srcMem, gslMemObject destMem, cmSurfFmt format, CALuint flags, bool isHwDebug)
{
    assert(m_cs != 0);

    uint64 surfaceSize = srcMem->getSurfaceSize();
    uint64 dstSize = destMem->getSurfaceSize();

    //
    // XXX -- this is somewhat lame.  Need the actual amount of data
    // to copy.  Not the surface sizes.  Since one is linear and one
    // could be tiled.  The smaller one should contain the size we need.
    //

    surfaceSize = (surfaceSize > dstSize) ? dstSize : surfaceSize;

    uint32 mode;

    if (isHwDebug) {
        mode = 0;   // Cannot use any sync flag to avoid possible deadlock due to halted wave
    }
    else {
        switch (flags)
        {
        case CAL_MEMCOPY_SYNC:
             mode = GSL_SYNCUPLOAD_SYNC_WAIT | GSL_SYNCUPLOAD_SYNC_START;
             break;

        case CAL_MEMCOPY_ASYNC:
             assert(0);
             //
             // XXX -- not currently supported so fall through
             //

        case CAL_MEMCOPY_DEFAULT:
        default:
             mode = GSL_SYNCUPLOAD_SYNC_START;
             break;
        }
    }

    m_cs->DMACopy(srcMem, 0, destMem, 0, surfaceSize, mode, NULL);

    return true;
}

void
CALGSLDevice::resCopy(gslMemObject srcRes, gslMemObject dstRes, uint32 flags) const
{
    assert(m_cs != 0);
    assert(srcRes != 0);
    assert(dstRes != 0);

    //! @note: GSL device isn't thread safe
    amd::ScopedLock k(gslDeviceOps());

    uint64 surfaceSize;

    CopyType type = GetCopyType(srcRes, dstRes, 0, 0, m_allowDMA, 0, surfaceSize, 0, 0);

    if (type == USE_DRMDMA)
    {
        m_cs->DMACopy(srcRes, 0, dstRes, 0, surfaceSize, GSL_SYNCUPLOAD_SYNC_WAIT, NULL);
        m_cs->Flush();
        Wait(m_cs, GSL_DRMDMA_SYNC_ATI, m_mapDMAQuery);
    }
    else if (type == USE_CPDMA)
    {
        m_cs->syncUploadRaw(srcRes, 0, dstRes, 0, surfaceSize, 0);
        m_cs->Flush();
        Wait(m_cs, GSL_SYNC_ATI, m_mapQuery);
    }
    else
    {
        assert(0 && "No copy engine is being used");
    }
}

#define CPDMA_THRESHOLD 131072

CopyType
CALGSLDevice::GetCopyType(
    gslMemObject srcMem,
    gslMemObject destMem,
    size_t* srcOffset,
    size_t* destOffset,
    bool allowDMA,
    uint32 flags,
    uint64& surfaceSize,
    size_t size,
    bool enableCopyRect) const
{
    CopyType    type = USE_NONE;
    intp        bppSrc = 0;
    intp        bppDst = 0;

    gslMemObjectAttribTiling srcTiling = srcMem->getAttribs().tiling;
    gslMemObjectAttribTiling dstTiling = destMem->getAttribs().tiling;
    gslMemObjectAttribType   srcType   = srcMem->getAttribs().type;
    gslMemObjectAttribType   dstType   = destMem->getAttribs().type;
    uint64                   srcSize   = srcMem->getSurfaceSize();
    uint64                   dstSize   = destMem->getSurfaceSize();

    surfaceSize = (srcSize > dstSize) ? dstSize : srcSize;

    if( size != 0)
      srcSize = (srcSize > size) ? size : srcSize;

    if(allowDMA == false) {
        if(((srcTiling != GSL_MOA_TILING_LINEAR) && (srcTiling != GSL_MOA_TILING_LINEAR_GENERAL)) ||
          ((dstTiling != GSL_MOA_TILING_LINEAR) && (dstTiling != GSL_MOA_TILING_LINEAR_GENERAL))) {
                type = USE_NONE;
                return type;
        }
    }


	// CPDMA isnt possible for anything other than a 1D_TEXURE or a BUFFER as it does a blind blob copy without regards to padding
    bool isCPDMApossible = ((srcTiling == GSL_MOA_TILING_LINEAR) || srcTiling == GSL_MOA_TILING_LINEAR_GENERAL) &&
                           ((dstTiling == GSL_MOA_TILING_LINEAR) || dstTiling == GSL_MOA_TILING_LINEAR_GENERAL) &&
                           (dstType == GSL_MOA_TEXTURE_1D || dstType == GSL_MOA_BUFFER) &&
                           (srcType == dstType);
    //
    // Use CPDMA for transfers < 128KB
    //
    if(isCPDMApossible && (((flags != CAL_MEMCOPY_ASYNC) && (srcSize <= CPDMA_THRESHOLD) && !enableCopyRect) ||
         (allowDMA == false)) )
    {
        type = USE_CPDMA;
    }
    // ### Check for Particular kind of DRMDMA here
    else if (allowDMA &&
            (((srcType == GSL_MOA_TEXTURE_2D) && (dstType == GSL_MOA_BUFFER)) ||
            ((dstType == GSL_MOA_TEXTURE_2D) && (srcType == GSL_MOA_BUFFER))))
    {
        uint64 pitch;
        uint64 linearBytePitch = 0;
        if ((srcTiling != GSL_MOA_TILING_LINEAR) &&
            (dstTiling == GSL_MOA_TILING_LINEAR))
        {
            bppSrc = srcMem->getBitsPerElement();
            pitch  = srcMem->getPitch();
            linearBytePitch = size * (bppSrc / 8);

            // Make sure linear pitch in bytes is 128 bytes aligned
            // Note: Cypress restriction, Cayman should have 4 bytes
            if (((linearBytePitch % 0x80) == 0) &&
                // another DRM restriciton... Cayman has 4 pixels
                (srcOffset[0] % 8 == 0))
            {
                type = USE_DRMDMA_T2L;
            }
            else
            {
                type = USE_NONE;
            }
        }
        else if ((srcTiling == GSL_MOA_TILING_LINEAR) &&
                 (dstTiling != GSL_MOA_TILING_LINEAR))
        {
            bppDst = destMem->getBitsPerElement();
            pitch  = destMem->getPitch();
            linearBytePitch = size * (bppDst / 8);

            // Make sure linear pitch in bytes is 128 bytes aligned
            // Note: Cypress restriction, Cayman should have 4 bytes
            if (((linearBytePitch % 0x80) == 0) &&
                // another DRM restriciton... Cayman has 4 pixels
                (destOffset[0] % 8 == 0))
            {
                type = USE_DRMDMA_L2T;
            }
            else
            {
                type = USE_NONE;
            }
        }
        else
        {
            type = USE_NONE;
        }
    }
    else if (dstType == srcType)
    {
        type = USE_DRMDMA;
    }

    //
    // Machine specific issues here
    //
    if (allowDMA && ((m_target == CAL_TARGET_CAYMAN) ||
              (m_target == CAL_TARGET_DEVASTATOR) ||
              (m_target == CAL_TARGET_SCRAPPER)
             ) &&
             ((bppSrc == 128 || bppDst == 128) &&
              ((srcTiling != GSL_MOA_TILING_LINEAR) ||
               (dstTiling != GSL_MOA_TILING_LINEAR)))
            )
    {
        type = USE_NONE;
    }

    return type;
}

uint32
CALGSLDevice::calcScratchBufferSize(uint32 regNum) const
{
    gslProgramTargetEnum target = GSL_COMPUTE_PROGRAM;

    // Determine the scratch size we need to allocate.
    cmScratchSpaceNeededPerShaderStage scratchSpacePerShaderStage;
    memset(&scratchSpacePerShaderStage, 0, sizeof(scratchSpacePerShaderStage));
    uint32 scratchBufferSizes[gslProgramTarget_COUNT];
    memset(scratchBufferSizes, 0, sizeof(scratchBufferSizes));
    uint32 enabledShadersFlag = 0;

    if (!uavInCB())
    {
        enabledShadersFlag |= CM_FRAGMENT_SHADER_BIT;
        scratchSpacePerShaderStage.scratchSpace[CM_FRAGMENT_SHADER] = regNum;
        target = GSL_FRAGMENT_PROGRAM;
    }
    else
    {
        enabledShadersFlag |= CM_COMPUTE_SHADER_BIT;
        scratchSpacePerShaderStage.scratchSpace[CM_COMPUTE_SHADER] = regNum;
    }

    m_cs->CalcAllScratchBufferSizes(enabledShadersFlag, scratchSpacePerShaderStage,
                                scratchBufferSizes);
    return scratchBufferSizes[target];
}

void
CALGSLDevice::convertInputChannelOrder(intp*channelOrder) const
{
    // set default to indicate that we don't want to override the channel order.
    // set all order to zero to indicate default.
    channelSwizzle chanSwiz = {SWIZZLE_ZERO,
                               SWIZZLE_ZERO,
                               SWIZZLE_ZERO,
                               SWIZZLE_ZERO};

    switch (*channelOrder) {
    case GSL_CHANNEL_ORDER_R:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_ZERO;
        chanSwiz.b = SWIZZLE_ZERO;
        chanSwiz.a = SWIZZLE_ONE;
        break;

    case GSL_CHANNEL_ORDER_A:
        chanSwiz.r = SWIZZLE_ZERO;
        chanSwiz.g = SWIZZLE_ZERO;
        chanSwiz.b = SWIZZLE_ZERO;
        chanSwiz.a = SWIZZLE_COMPONENT0;
        break;

    case GSL_CHANNEL_ORDER_RG:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_ZERO;
        chanSwiz.a = SWIZZLE_ONE;
        break;

    case GSL_CHANNEL_ORDER_RA:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_ZERO;
        chanSwiz.b = SWIZZLE_ZERO;
        chanSwiz.a = SWIZZLE_COMPONENT1;
        break;

    case GSL_CHANNEL_ORDER_RGB:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_COMPONENT2;
        chanSwiz.a = SWIZZLE_ONE;
        break;

    case GSL_CHANNEL_ORDER_RGBA:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_COMPONENT2;
        chanSwiz.a = SWIZZLE_COMPONENT3;
        break;

    case GSL_CHANNEL_ORDER_ARGB:
        chanSwiz.r = SWIZZLE_COMPONENT1;
        chanSwiz.g = SWIZZLE_COMPONENT2;
        chanSwiz.b = SWIZZLE_COMPONENT3;
        chanSwiz.a = SWIZZLE_COMPONENT0;
        break;

    case GSL_CHANNEL_ORDER_BGRA:
        chanSwiz.r = SWIZZLE_COMPONENT2;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_COMPONENT0;
        chanSwiz.a = SWIZZLE_COMPONENT3;
        break;

    case GSL_CHANNEL_ORDER_SRGB:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_COMPONENT2;
        chanSwiz.a = SWIZZLE_ONE;
        break;

    case GSL_CHANNEL_ORDER_SRGBX:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_COMPONENT2;
        chanSwiz.a = SWIZZLE_ONE;
        break;

    case GSL_CHANNEL_ORDER_SRGBA:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_COMPONENT2;
        chanSwiz.a = SWIZZLE_COMPONENT3;
        break;

    case GSL_CHANNEL_ORDER_SBGRA:
        chanSwiz.r = SWIZZLE_COMPONENT2;
        chanSwiz.g = SWIZZLE_COMPONENT1;
        chanSwiz.b = SWIZZLE_COMPONENT0;
        chanSwiz.a = SWIZZLE_COMPONENT3;
        break;

    case GSL_CHANNEL_ORDER_REPLICATE_R:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT0;
        chanSwiz.b = SWIZZLE_COMPONENT0;
        chanSwiz.a = SWIZZLE_COMPONENT0;
        break;

    case GSL_CHANNEL_ORDER_INTENSITY:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT0;
        chanSwiz.b = SWIZZLE_COMPONENT0;
        chanSwiz.a = SWIZZLE_COMPONENT0;
        break;

    case GSL_CHANNEL_ORDER_LUMINANCE:
        chanSwiz.r = SWIZZLE_COMPONENT0;
        chanSwiz.g = SWIZZLE_COMPONENT0;
        chanSwiz.b = SWIZZLE_COMPONENT0;
        chanSwiz.a = SWIZZLE_ONE;
        break;
    default: assert(0); break;
    };

    *channelOrder = *(uint32 *)&chanSwiz;
}

void
CALGSLDevice::fillImageHwState(gslMemObject mem, void* hwState, uint32 hwStateSize) const
{
    amd::ScopedLock k(gslDeviceOps());
    intp channelOrder = mem->getAttribs().channelOrder;
    convertInputChannelOrder(&channelOrder);
    m_textureResource->updateDepthTextureParam(mem);
    m_textureResource->getTextureSrd(m_cs, mem, reinterpret_cast<const char*>(&channelOrder),
        hwState, hwStateSize);
}

void
CALGSLDevice::fillSamplerHwState(bool unnorm, uint32 min, uint32 mag, uint32 addr,
    float minLod, float maxLod, void* hwState, uint32 hwStateSize) const
{
    amd::ScopedLock k(gslDeviceOps());
    m_textureSampler->setUnnormalizedMode(m_cs, unnorm);
    m_textureSampler->setMinFilter(m_cs, static_cast<gslTexParameterParamMinFilter>(min));
    m_textureSampler->setMagFilter(m_cs, static_cast<gslTexParameterParamMagFilter>(mag));
    m_textureSampler->setWrap(m_cs, GSL_TEXTURE_WRAP_S, static_cast<gslTexParameterParamWrap>(addr));
    m_textureSampler->setWrap(m_cs, GSL_TEXTURE_WRAP_T, static_cast<gslTexParameterParamWrap>(addr));
    m_textureSampler->setWrap(m_cs, GSL_TEXTURE_WRAP_R, static_cast<gslTexParameterParamWrap>(addr));
    m_textureSampler->setMinLOD(m_cs, static_cast<float32>(minLod));
    m_textureSampler->setMaxLOD(m_cs, static_cast<float32>(maxLod));

    m_textureSampler->getSamplerSrd(m_cs, hwState, hwStateSize);
}
