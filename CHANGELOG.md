# Change Log for HIP

Full documentation for HIP is available at [docs.amd.com](https://docs.amd.com/)

## (Unreleased) HIP 5.7 (For ROCm 5.7)

### Optimizations

### Added
- Added meta_group_size/rank for getting the number of tiles and rank of a tile in the partition
- Added new APIs supporting Windows only, under development on Linux

    - hipMallocMipmappedArray for allocating a mipmapped array on the device

    - hipFreeMipmappedArray for freeing a mipmapped array on the device

    - hipGetMipmappedArrayLevel for getting a mipmap level of a HIP mipmapped array

    - hipMipmappedArrayCreate for creating a mipmapped array

    - hipMipmappedArrayDestroy for destroy a mipmapped array

    - hipMipmappedArrayGetLevel for getting a mipmapped array on a mipmapped level

- Added new HIP APIs, still under developement

    - hipGraphAddExternalSemaphoresWaitNode for creating a external semaphor wait node and adds it to a graph

    - hipGraphAddExternalSemaphoresSignalNode for creating a external semaphor signal node and adds it to a graph

    - hipGraphExternalSemaphoresSignalNodeSetParams for updating node parameters in the external semaphore signal node
    - hipGraphExternalSemaphoresWaitNodeSetParams for updating node parameters in the external semaphore wait node

    - hipGraphExternalSemaphoresSignalNodeGetParams for returning external semaphore signal node params

    - hipGraphExternalSemaphoresWaitNodeGetParams for returning external semaphore wait node params

    - hipGraphExecExternalSemaphoresSignalNodeSetParams for updating node parameters in the external semaphore signal node in the given graphExec

    - hipGraphExecExternalSemaphoresWaitNodeSetParams for updating node parameters in the external semaphore wait node in the given graphExec

### Changed

### Fixed

## HIP 5.6 (For ROCm 5.6)

### Optimizations
- Consolidation of hipamd, rocclr and OpenCL projects in clr
- Optimized lock for graph global capture mode

### Added
- Added hipRTC support for amd_hip_fp16
- Added hipStreamGetDevice implementation to get the device assocaited with the stream
- Added HIP_AD_FORMAT_SIGNED_INT16 in hipArray formats
- hipArrayGetInfo for getting information about the specified array
- hipArrayGetDescriptor for getting 1D or 2D array descriptor
- hipArray3DGetDescriptor to get 3D array descriptor

### Changed
- hipMallocAsync to return success for zero size allocation to match hipMalloc
- Separation of hipcc perl binaries from HIP project to hipcc project. hip-devel package depends on newly added hipcc package
- Consolidation of hipamd, ROCclr, and OpenCL repositories into a single repository called clr. Instructions are updated to build HIP from sources in the HIP Installation guide
- Removed hipBusBandwidth and hipCommander samples from hip-tests

### Fixed
- Fixed regression in hipMemCpyParam3D when offset is applied

### Known Issues
- Limited testing on xnack+ configuration
  - Multiple HIP tests failures (gpuvm fault or hangs)

### Upcoming changes in future release
- Removal of gcnarch from hipDeviceProp_t structure
- Addition of new fields in hipDeviceProp_t structure
  - maxTexture1D
  - maxTexture2D
  - maxTexture1DLayered
  - maxTexture2DLayered
  - sharedMemPerMultiprocessor
  - deviceOverlap
  - asyncEngineCount
  - surfaceAlignment
  - unifiedAddressing
  - computePreemptionSupported
  - uuid
- Removal of deprecated code
  - hip-hcc codes from hip code tree
- Correctness of hipArray usage in HIP APIs
- HIPMEMCPY_3D fields correction (unsigned int -> size_t)
