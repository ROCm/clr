#include "hip_conversions.hpp"

hipError_t ihipMemcpy3D_validate(const hipMemcpy3DParms* p);

hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

hipError_t ihipMemcpyCommand(amd::Command*& command, void* dst, const void* src, size_t sizeBytes,
                             hipMemcpyKind kind, hip::Stream& stream, bool isAsync = false);

void ihipHtoHMemcpy(void* dst, const void* src, size_t sizeBytes, hip::Stream& stream);

bool IsHtoHMemcpy(void* dst, const void* src, hipMemcpyKind kind);

hipError_t ihipLaunchKernel_validate(hipFunction_t f, uint32_t globalWorkSizeX,
                                     uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                     uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                     uint32_t sharedMemBytes, void** kernelParams, void** extra,
                                     int deviceId, uint32_t params);

hipError_t ihipMemset_validate(void* dst, int64_t value, size_t valueSize, size_t sizeBytes);

hipError_t ihipMemset3D_validate(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                 size_t sizeBytes);

hipError_t ihipLaunchKernelCommand(amd::Command*& command, hipFunction_t f,
                                   uint32_t globalWorkSizeX, uint32_t globalWorkSizeY,
                                   uint32_t globalWorkSizeZ, uint32_t blockDimX, uint32_t blockDimY,
                                   uint32_t blockDimZ, uint32_t sharedMemBytes,
                                   hip::Stream* stream, void** kernelParams, void** extra,
                                   hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags,
                                   uint32_t params, uint32_t gridId, uint32_t numGrids,
                                   uint64_t prevGridSum, uint64_t allGridSum, uint32_t firstDevice);

hipError_t ihipMemcpy3DCommand(amd::Command*& command, const hipMemcpy3DParms* p,
                               hip::Stream* stream);

hipError_t ihipMemsetCommand(std::vector<amd::Command*>& commands, void* dst, int64_t value,
                             size_t valueSize, size_t sizeBytes, hip::Stream* stream);

hipError_t ihipMemset3DCommand(std::vector<amd::Command*>& commands, hipPitchedPtr pitchedDevPtr,
                               int value, hipExtent extent, hip::Stream* stream, size_t elementSize = 1);

hipError_t ihipMemcpySymbol_validate(const void* symbol, size_t sizeBytes, size_t offset,
                                     size_t& sym_size, hipDeviceptr_t& device_ptr);

hipError_t ihipMemcpyAtoDValidate(hipArray* srcArray, void* dstDevice, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t dstRowPitch, size_t dstSlicePitch, amd::Memory*& dstMemory,
                                  amd::Image*& srcImage, amd::BufferRect& srcRect,
                                  amd::BufferRect& dstRect);

hipError_t ihipMemcpyDtoAValidate(void* srcDevice, hipArray* dstArray, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, amd::Image*& dstImage,
                                  amd::Memory*& srcMemory, amd::BufferRect& dstRect,
                                  amd::BufferRect& srcRect);

hipError_t ihipMemcpyDtoDValidate(void* srcDevice, void* dstDevice, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, size_t dstRowPitch,
                                  size_t dstSlicePitch, amd::Memory*& srcMemory,
                                  amd::Memory*& dstMemory, amd::BufferRect& srcRect,
                                  amd::BufferRect& dstRect);


hipError_t ihipMemcpyDtoHValidate(void* srcDevice, void* dstHost, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, size_t dstRowPitch,
                                  size_t dstSlicePitch, amd::Memory*& srcMemory,
                                  amd::BufferRect& srcRect, amd::BufferRect& dstRect);

hipError_t ihipMemcpyHtoDValidate(const void* srcHost, void* dstDevice, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, size_t dstRowPitch,
                                  size_t dstSlicePitch, amd::Memory*& dstMemory,
                                  amd::BufferRect& srcRect, amd::BufferRect& dstRect);


hipError_t ihipMemcpyAtoAValidate(hipArray* srcArray, hipArray* dstArray, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  amd::Image*& srcImage, amd::Image*& dstImage);


hipError_t ihipMemcpyHtoAValidate(const void* srcHost, hipArray* dstArray, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t srcRowPitch, size_t srcSlicePitch, amd::Image*& dstImage,
                                  amd::BufferRect& srcRect);

hipError_t ihipMemcpyAtoHValidate(hipArray* srcArray, void* dstHost, amd::Coord3D& srcOrigin,
                                  amd::Coord3D& dstOrigin, amd::Coord3D& copyRegion,
                                  size_t dstRowPitch, size_t dstSlicePitch, amd::Image*& srcImage,
                                  amd::BufferRect& dstRect);

hipError_t ihipGraphMemsetParams_validate(const hipMemsetParams* pNodeParams);
