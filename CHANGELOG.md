# Change Log for HIP

Full documentation for HIP is available at [docs.amd.com](https://docs.amd.com/)

## HIP 5.6 (For ROCm 5.6)

### Optimizations
- Consolidation of hipamd, rocclr and OpenCL projects in clr
- Optimized lock for graph global capture mode

### Added
- Added hipRTC support for amd_hip_fp16
- Added hipStreamGetDevice implementation to get the device assocaited with the stream
- Added HIP_AD_FORMAT_SIGNED_INT16 in hipArray formats

### Changed
- hipMallocAsync to return success for zero size allocation to match hipMalloc
- Separation of hipcc perl binaries from HIP project to hipcc project. hip-devel package depends on newly added hipcc package.
- Removed hipBusBandwidth and hipCommander samples from hip-tests

### Fixed
- Fixed regression in hipMemCpyParam3D when offset is applied

### Upcoming changes in future release
- Removal of gcnarch from hipDeviceProp_t structure
- Removal of deprecated stuff such as hip-hcc codes from hip code tree
- Correctness of hipArray usage in HIP APIs
- HIPMEMCPY_3D fields correction (unsigned int -> size_t)
- 
