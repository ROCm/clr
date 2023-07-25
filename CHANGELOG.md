# Change Log for HIP

Full documentation for HIP is available at [docs.amd.com](https://docs.amd.com/)

## HIP 5.7 (For ROCm 5.7)

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

### Changed

### Fixed

### Known Issues
- HIP memory type enum values currently don't support equivalent value to cudaMemoryTypeUnregistered, due to HIP functionality backward compatibility.
- HIP API hipPointerGetAttributes could return invalid value in case the input memory pointer was not allocated through any HIP API on device or host.

### Upcoming changes in ROCm 6.0 release
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
  - hostRegisterSupported
  - uuid
- Removal of deprecated code
  -hip-hcc codes from hip code tree
- Correct hipArray usage in HIP APIs such as hipMemcpyAtoH and hipMemcpyHtoA
- HIPMEMCPY_3D fields correction to avoid truncation of "size_t" to "unsigned int" inside hipMemcpy3D()
- Renaming of 'memoryType' in hipPointerAttribute_t structure to 'type'
- Correct hipGetLastError to return the last error instead of last API call's return code
- Update hipExternalSemaphoreHandleDesc to add "unsigned int reserved[16]"
- Correct handling of flag values in hipIpcOpenMemHandle for hipIpcMemLazyEnablePeerAccess
- Remove hiparray* and make it opaque with hipArray_t

## HIP 5.6.1 (For ROCm 5.6.1)
### Fixed
- Enabled xnack+ check in HIP catch2 tests hang while tests execution
- Memory leak when code object files are loaded/unloaded via hipModuleLoad/hipModuleUnload APIs
- Fixed a crash happening while using hipGraphAddMemFreeNode

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
- hipSetDevice and hipSetDeviceFlags APIs return hipErrorInvalidDevice instead of hipErrorNoDevice, on a system without GPU
- Known memory leak when code object files are loaded/unloaded via hipModuleLoad/hipModuleUnload APIs. Issue will be fixed in future release

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
  - hostRegisterSupported
  - uuid
- Removal of deprecated code
  -hip-hcc codes from hip code tree
- Correct hipArray usage in HIP APIs such as hipMemcpyAtoH and hipMemcpyHtoA
- HIPMEMCPY_3D fields correction to avoid truncation of "size_t" to "unsigned int" inside hipMemcpy3D()
- Renaming of 'memoryType' in hipPointerAttribute_t structure to 'type'
- Correct hipGetLastError to return the last error instead of last API call's return code
- Update hipExternalSemaphoreHandleDesc to add "unsigned int reserved[16]"
- Correct handling of flag values in hipIpcOpenMemHandle for hipIpcMemLazyEnablePeerAccess
- Remove hiparray* and make it opaque with hipArray_t
