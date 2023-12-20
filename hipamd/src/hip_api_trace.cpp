/*
    Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
   */

// #include "hip_api_trace.hpp"
#include <hip/amd_detail/hip_api_trace.hpp>

#include "hip_internal.hpp"
namespace hip {
// HIP Internal APIs
hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                     hipStream_t* stream);
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                      hipStream_t stream);
void** __hipRegisterFatBinary(const void* data);
void __hipRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction,
                           const char* deviceName, unsigned int threadLimit, uint3* tid, uint3* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize);
void __hipRegisterManagedVar(void* hipModule, void** pointer, void* init_value, const char* name,
                             size_t size, unsigned align);
void __hipRegisterSurface(void** modules, void* var, char* hostVar, char* deviceVar, int type,
                          int ext);
void __hipRegisterTexture(void** modules, void* var, char* hostVar, char* deviceVar, int type,
                          int norm, int ext);
void __hipRegisterVar(void** modules, void* var, char* hostVar, char* deviceVar, int ext,
                      size_t size, int constant, int global);
void __hipUnregisterFatBinary(void** modules);
const char* hipApiName(uint32_t id);
hipError_t hipArray3DCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray);
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array);
hipError_t hipArrayCreate(hipArray_t* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);
hipError_t hipArrayDestroy(hipArray_t array);
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray_t array);
hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags,
                           hipArray_t array);
hipError_t hipBindTexture(size_t* offset, const textureReference* tex, const void* devPtr,
                          const hipChannelFormatDesc* desc, size_t size);
hipError_t hipBindTexture2D(size_t* offset, const textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch);
hipError_t hipBindTextureToArray(const textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc);
hipError_t hipBindTextureToMipmappedArray(const textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc);
hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop);
hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* properties);
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);
hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const struct hipResourceViewDesc* pResViewDesc);
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
hipError_t hipCtxDestroy(hipCtx_t ctx);
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig);
hipError_t hipCtxGetCurrent(hipCtx_t* ctx);
hipError_t hipCtxGetDevice(hipDevice_t* device);
hipError_t hipCtxGetFlags(unsigned int* flags);
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
hipError_t hipCtxPopCurrent(hipCtx_t* ctx);
hipError_t hipCtxPushCurrent(hipCtx_t ctx);
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
hipError_t hipCtxSynchronize(void);
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem);
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem);
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device);
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit);
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device);
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice,
                                    int dstDevice);
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device);
hipError_t hipDeviceGraphMemTrim(int device);
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
hipError_t hipDeviceReset(void);
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value);
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool);
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipDrvGetErrorName(hipError_t hipError, const char** errorString);
hipError_t hipDrvGetErrorString(hipError_t hipError, const char** errorString);
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy);
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy);
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream);
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes,
                                      void** data, hipDeviceptr_t ptr);
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
hipError_t hipEventQuery(hipEvent_t event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype,
                                        uint32_t* hopcount);
hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                              void** args, size_t sharedMemBytes, hipStream_t stream,
                              hipEvent_t startEvent, hipEvent_t stopEvent, int flags);
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                              unsigned int flags);
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask);
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask);
hipError_t hipExternalMemoryGetMappedBuffer(void** devPtr, hipExternalMemory_t extMem,
                                            const hipExternalMemoryBufferDesc* bufferDesc);
hipError_t hipFree(void* ptr);
hipError_t hipFreeArray(hipArray_t array);
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream);
hipError_t hipFreeHost(void* ptr);
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);
hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc);
hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func);
hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value);
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);
hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config);
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList);
hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipGetDeviceFlags(unsigned int* flags);
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int deviceId);
hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device);
const char* hipGetErrorName(hipError_t hip_error);
const char* hipGetErrorString(hipError_t hipError);
hipError_t hipGetLastError(void);
hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,
                                     hipMipmappedArray_const_t mipmappedArray, unsigned int level);
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol);
hipError_t hipGetSymbolSize(size_t* size, const void* symbol);
hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* texref);
hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject);
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph);
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies);
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies);
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event);
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event);
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams);
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams);
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   hipMemAllocNodeParams* pNodeParams);
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  void* dev_ptr);
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams);
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind);
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind);
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind);
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams);
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph);
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph);
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);
hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags);
hipError_t hipGraphDestroy(hipGraph_t graph);
hipError_t hipGraphDestroyNode(hipGraphNode_t node);
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph);
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event);
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event);
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams);
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams);
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams);
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind);
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind);
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset, hipMemcpyKind kind);
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams);
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out);
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges);
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes);
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams);
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams);
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags);
hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst);
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value);
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams);
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value);
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams);
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr);
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams);
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams);
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind);
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind);
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind);
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams);
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams);
hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph);
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies);
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes);
hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled);
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType);
hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled);
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count);
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies);
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count,
                                    unsigned int flags);
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags);
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags);
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream);
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource);
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource,
                                                unsigned int arrayIndex, unsigned int mipLevel);
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream);
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void* ptr);
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
hipError_t hipHostUnregister(void* hostPtr);
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out,
                                   const hipExternalMemoryHandleDesc* memHandleDesc);
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc);
hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                 const hipGraphNode_t* dependencies, size_t numDependencies,
                                 const HIP_MEMSET_NODE_PARAMS* memsetParams, hipCtx_t ctx);
hipError_t hipInit(unsigned int flags);
hipError_t hipIpcCloseMemHandle(void* devPtr);
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);
const char* hipKernelNameRef(const hipFunction_t f);
const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream);
hipError_t hipLaunchByPtr(const void* func);
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream);
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                                 unsigned int flags);
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData);
hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                           void** args, size_t sharedMemBytes, hipStream_t stream);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);
hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags);
hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height, unsigned int flags);
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream);
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream);
hipError_t hipMallocHost(void** ptr, size_t size);
hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags);
hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,
                                   const struct hipChannelFormatDesc* desc, struct hipExtent extent,
                                   unsigned int numLevels, unsigned int flags);
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
hipError_t hipMemAddressFree(void* devPtr, size_t size);
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags);
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device);
hipError_t hipMemAllocHost(void** ptr, size_t size);
hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes);
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags);
hipError_t hipMemExportToShareableHandle(void* shareableHandle,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType handleType,
                                         unsigned long long flags);
hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr);
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option);
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                   hipMemGenericAllocationHandle_t handle);
hipError_t hipMemGetInfo(size_t* free, size_t* total);
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle,
                                           hipMemAllocationHandleType shHandleType);
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags);
hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count,
                               hipStream_t stream);
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props);
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr);
hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool,
                                             hipMemAllocationHandleType handle_type,
                                             unsigned int flags);
hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                               hipMemLocation* location);
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle,
                                               hipMemAllocationHandleType handle_type,
                                               unsigned int flags);
hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool,
                                   hipMemPoolPtrExportData* export_data);
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list,
                               size_t count);
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold);
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream);
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size);
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count);
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                    hipMemRangeAttribute* attributes, size_t num_attributes,
                                    const void* dev_ptr, size_t count);
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr);
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
hipError_t hipMemUnmap(void* ptr, size_t size);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                size_t hOffset, size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                     size_t hOffset, size_t width, size_t height,
                                     hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream);
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p);
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream);
hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray, size_t srcOffset, size_t count);
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream);
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind);
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                               hipMemcpyKind kind);
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost, size_t count);
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy);
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream);
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                         size_t sizeBytes);
hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream);
hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind);
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind);
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hipStream_t stream);
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,
                            hipStream_t stream);
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent);
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                            hipStream_t stream);
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count);
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count,
                             hipStream_t stream);
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream);
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count);
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count,
                            hipStream_t stream);
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels);
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);
hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name);
hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name);
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams);
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices, unsigned int flags);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues);
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f,
                                                              int blockSize,
                                                              size_t dynSharedMemPerBlk);
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags);
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, hipFunction_t f,
                                                   size_t dynSharedMemPerBlk, int blockSizeLimit);
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                                            hipFunction_t f,
                                                            size_t dynSharedMemPerBlk,
                                                            int blockSizeLimit, unsigned int flags);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                        int blockSize, size_t dynSharedMemPerBlk);
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* f,
                                                                 int blockSize,
                                                                 size_t dynSharedMemPerBlk,
                                                                 unsigned int flags);
hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, const void* f,
                                             size_t dynSharedMemPerBlk, int blockSizeLimit);
hipError_t hipPeekAtLastError(void);
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr);
hipError_t hipProfilerStart();
hipError_t hipProfilerStop();
hipError_t hipRuntimeGetVersion(int* runtimeVersion);
hipError_t hipSetDevice(int deviceId);
hipError_t hipSetDeviceFlags(unsigned flags);
hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream);
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags);
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length,
                                   unsigned int flags);
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId);
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out);
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device);
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
hipError_t hipStreamQuery(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags);
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask);
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask);
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags);
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags);
hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject, const HIP_RESOURCE_DESC* pResDesc,
                              const HIP_TEXTURE_DESC* pTexDesc,
                              const HIP_RESOURCE_VIEW_DESC* pResViewDesc);
hipError_t hipTexObjectDestroy(hipTextureObject_t texObject);
hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, hipTextureObject_t texObject);
hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                           hipTextureObject_t texObject);
hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, hipTextureObject_t texObject);
hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef);
hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode* pam, const textureReference* texRef,
                                   int dim);
hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode* pfm, const textureReference* texRef);
hipError_t hipTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef);
hipError_t hipTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels,
                              const textureReference* texRef);
hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef);
hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray, const textureReference* texRef);
hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode* pfm,
                                        const textureReference* texRef);
hipError_t hipTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef);
hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp,
                                        const textureReference* texRef);
hipError_t hipTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr,
                               size_t bytes);
hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr, size_t Pitch);
hipError_t hipTexRefSetAddressMode(textureReference* texRef, int dim,
                                   enum hipTextureAddressMode am);
hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags);
hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor);
hipError_t hipTexRefSetFilterMode(textureReference* texRef, enum hipTextureFilterMode fm);
hipError_t hipTexRefSetFlags(textureReference* texRef, unsigned int Flags);
hipError_t hipTexRefSetFormat(textureReference* texRef, hipArray_Format fmt,
                              int NumPackedComponents);
hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso);
hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef, enum hipTextureFilterMode fm);
hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias);
hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp,
                                        float maxMipMapLevelClamp);
hipError_t hipTexRefSetMipmappedArray(textureReference* texRef,
                                      struct hipMipmappedArray* mipmappedArray, unsigned int Flags);
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode);
hipError_t hipUnbindTexture(const textureReference* tex);
hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                               unsigned int initialRefcount, unsigned int flags);
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count);
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count);
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                          const hipExternalSemaphoreWaitParams* paramsArray,
                                          unsigned int numExtSems, hipStream_t stream);
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc);
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                      uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                      uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                      uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                      hipStream_t hStream, void** kernelParams, void** extra,
                                      hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flag);
hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                      uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                      uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                      uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                      hipStream_t hStream, void** kernelParams, void** extra,
                                      hipEvent_t startEvent, hipEvent_t stopEvent);
hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                                 size_t offset, hipMemcpyKind kind);
hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                   hipMemcpyKind kind);
hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                           size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                    size_t hOffset, size_t width, size_t height,
                                    hipMemcpyKind kind);
hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p);
hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync_spt(void* dst, int value, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height);
hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value, size_t width, size_t height,
                                hipStream_t stream);
hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                hipStream_t stream);
hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent);
hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                              hipStream_t stream);
hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms* p, hipStream_t stream);
hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, hipMemcpyKind kind,
                                hipStream_t stream);
hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t sizeBytes,
                                        size_t offset, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t sizeBytes,
                                      size_t offset, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                  size_t hOffset, size_t count, hipMemcpyKind kind);
hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                         size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                         size_t height, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset,
                                       const void* src, size_t spitch, size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream);
hipError_t hipStreamQuery_spt(hipStream_t stream);
hipError_t hipStreamSynchronize_spt(hipStream_t stream);
hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority);
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags);
hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags);
hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback,
                                    void* userData, unsigned int flags);
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream = NULL);
hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim, dim3 blockDim,
                                          void** kernelParams, uint32_t sharedMemBytes,
                                          hipStream_t hStream);

hipError_t hipLaunchKernel_spt(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                               void** args, size_t sharedMemBytes, hipStream_t stream);

hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode);
hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph);
hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                       unsigned long long* pId);
hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,
                                          hipStreamCaptureStatus* captureStatus_out,
                                          unsigned long long* id_out, hipGraph_t* graph_out,
                                          const hipGraphNode_t** dependencies_out,
                                          size_t* numDependencies_out);
hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void* userData);
int hipGetStreamDeviceId(hipStream_t stream);
hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const HIP_MEMCPY3D* copyParams, hipCtx_t ctx);
hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol);
hipError_t hipGraphAddExternalSemaphoresSignalNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams);
hipError_t hipGraphAddExternalSemaphoresWaitNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams);
hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams);
hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams);
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind);
hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out);
hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams);
hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out);
hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams);
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices, unsigned int flags);
}  // namespace hip

void UpdateHipCompilerDispatchTable(HipCompilerDispatchTable* ptrCompilerDispatchTable) {
  ptrCompilerDispatchTable->size = sizeof(HipCompilerDispatchTable);
  ptrCompilerDispatchTable->__hipPopCallConfiguration_fn = hip::__hipPopCallConfiguration;
  ptrCompilerDispatchTable->__hipPushCallConfiguration_fn = hip::__hipPushCallConfiguration;
  ptrCompilerDispatchTable->__hipRegisterFatBinary_fn = hip::__hipRegisterFatBinary;
  ptrCompilerDispatchTable->__hipRegisterFunction_fn = hip::__hipRegisterFunction;
  ptrCompilerDispatchTable->__hipRegisterManagedVar_fn = hip::__hipRegisterManagedVar;
  ptrCompilerDispatchTable->__hipRegisterSurface_fn = hip::__hipRegisterSurface;
  ptrCompilerDispatchTable->__hipRegisterTexture_fn = hip::__hipRegisterTexture;
  ptrCompilerDispatchTable->__hipRegisterVar_fn = hip::__hipRegisterVar;
  ptrCompilerDispatchTable->__hipUnregisterFatBinary_fn = hip::__hipUnregisterFatBinary;
}

void UpdateHipDispatchTable(HipDispatchTable* ptrDispatchTable) {
  ptrDispatchTable->size = sizeof(HipDispatchTable);
  ptrDispatchTable->hipApiName_fn = hip::hipApiName;
  ptrDispatchTable->hipArray3DCreate_fn = hip::hipArray3DCreate;
  ptrDispatchTable->hipArray3DGetDescriptor_fn = hip::hipArray3DGetDescriptor;
  ptrDispatchTable->hipArrayCreate_fn = hip::hipArrayCreate;
  ptrDispatchTable->hipArrayDestroy_fn = hip::hipArrayDestroy;
  ptrDispatchTable->hipArrayGetDescriptor_fn = hip::hipArrayGetDescriptor;
  ptrDispatchTable->hipArrayGetInfo_fn = hip::hipArrayGetInfo;
  ptrDispatchTable->hipBindTexture_fn = hip::hipBindTexture;
  ptrDispatchTable->hipBindTexture2D_fn = hip::hipBindTexture2D;
  ptrDispatchTable->hipBindTextureToArray_fn = hip::hipBindTextureToArray;
  ptrDispatchTable->hipBindTextureToMipmappedArray_fn = hip::hipBindTextureToMipmappedArray;
  ptrDispatchTable->hipChooseDevice_fn = hip::hipChooseDevice;
  ptrDispatchTable->hipChooseDeviceR0000_fn = hip::hipChooseDeviceR0000;
  ptrDispatchTable->hipConfigureCall_fn = hip::hipConfigureCall;
  ptrDispatchTable->hipCreateSurfaceObject_fn = hip::hipCreateSurfaceObject;
  ptrDispatchTable->hipCreateTextureObject_fn = hip::hipCreateTextureObject;
  ptrDispatchTable->hipCtxCreate_fn = hip::hipCtxCreate;
  ptrDispatchTable->hipCtxDestroy_fn = hip::hipCtxDestroy;
  ptrDispatchTable->hipCtxDisablePeerAccess_fn = hip::hipCtxDisablePeerAccess;
  ptrDispatchTable->hipCtxEnablePeerAccess_fn = hip::hipCtxEnablePeerAccess;
  ptrDispatchTable->hipCtxGetApiVersion_fn = hip::hipCtxGetApiVersion;
  ptrDispatchTable->hipCtxGetCacheConfig_fn = hip::hipCtxGetCacheConfig;
  ptrDispatchTable->hipCtxGetCurrent_fn = hip::hipCtxGetCurrent;
  ptrDispatchTable->hipCtxGetDevice_fn = hip::hipCtxGetDevice;
  ptrDispatchTable->hipCtxGetFlags_fn = hip::hipCtxGetFlags;
  ptrDispatchTable->hipCtxGetSharedMemConfig_fn = hip::hipCtxGetSharedMemConfig;
  ptrDispatchTable->hipCtxPopCurrent_fn = hip::hipCtxPopCurrent;
  ptrDispatchTable->hipCtxPushCurrent_fn = hip::hipCtxPushCurrent;
  ptrDispatchTable->hipCtxSetCacheConfig_fn = hip::hipCtxSetCacheConfig;
  ptrDispatchTable->hipCtxSetCurrent_fn = hip::hipCtxSetCurrent;
  ptrDispatchTable->hipCtxSetSharedMemConfig_fn = hip::hipCtxSetSharedMemConfig;
  ptrDispatchTable->hipCtxSynchronize_fn = hip::hipCtxSynchronize;
  ptrDispatchTable->hipDestroySurfaceObject_fn = hip::hipDestroySurfaceObject;
  ptrDispatchTable->hipDestroyTextureObject_fn = hip::hipDestroyTextureObject;
  ptrDispatchTable->hipDeviceCanAccessPeer_fn = hip::hipDeviceCanAccessPeer;
  ptrDispatchTable->hipDeviceComputeCapability_fn = hip::hipDeviceComputeCapability;
  ptrDispatchTable->hipDeviceDisablePeerAccess_fn = hip::hipDeviceDisablePeerAccess;
  ptrDispatchTable->hipDeviceEnablePeerAccess_fn = hip::hipDeviceEnablePeerAccess;
  ptrDispatchTable->hipDeviceGet_fn = hip::hipDeviceGet;
  ptrDispatchTable->hipDeviceGetAttribute_fn = hip::hipDeviceGetAttribute;
  ptrDispatchTable->hipDeviceGetByPCIBusId_fn = hip::hipDeviceGetByPCIBusId;
  ptrDispatchTable->hipDeviceGetCacheConfig_fn = hip::hipDeviceGetCacheConfig;
  ptrDispatchTable->hipDeviceGetDefaultMemPool_fn = hip::hipDeviceGetDefaultMemPool;
  ptrDispatchTable->hipDeviceGetGraphMemAttribute_fn = hip::hipDeviceGetGraphMemAttribute;
  ptrDispatchTable->hipDeviceGetLimit_fn = hip::hipDeviceGetLimit;
  ptrDispatchTable->hipDeviceGetMemPool_fn = hip::hipDeviceGetMemPool;
  ptrDispatchTable->hipDeviceGetName_fn = hip::hipDeviceGetName;
  ptrDispatchTable->hipDeviceGetP2PAttribute_fn = hip::hipDeviceGetP2PAttribute;
  ptrDispatchTable->hipDeviceGetPCIBusId_fn = hip::hipDeviceGetPCIBusId;
  ptrDispatchTable->hipDeviceGetSharedMemConfig_fn = hip::hipDeviceGetSharedMemConfig;
  ptrDispatchTable->hipDeviceGetStreamPriorityRange_fn = hip::hipDeviceGetStreamPriorityRange;
  ptrDispatchTable->hipDeviceGetUuid_fn = hip::hipDeviceGetUuid;
  ptrDispatchTable->hipDeviceGraphMemTrim_fn = hip::hipDeviceGraphMemTrim;
  ptrDispatchTable->hipDevicePrimaryCtxGetState_fn = hip::hipDevicePrimaryCtxGetState;
  ptrDispatchTable->hipDevicePrimaryCtxRelease_fn = hip::hipDevicePrimaryCtxRelease;
  ptrDispatchTable->hipDevicePrimaryCtxReset_fn = hip::hipDevicePrimaryCtxReset;
  ptrDispatchTable->hipDevicePrimaryCtxRetain_fn = hip::hipDevicePrimaryCtxRetain;
  ptrDispatchTable->hipDevicePrimaryCtxSetFlags_fn = hip::hipDevicePrimaryCtxSetFlags;
  ptrDispatchTable->hipDeviceReset_fn = hip::hipDeviceReset;
  ptrDispatchTable->hipDeviceSetCacheConfig_fn = hip::hipDeviceSetCacheConfig;
  ptrDispatchTable->hipDeviceSetGraphMemAttribute_fn = hip::hipDeviceSetGraphMemAttribute;
  ptrDispatchTable->hipDeviceSetLimit_fn = hip::hipDeviceSetLimit;
  ptrDispatchTable->hipDeviceSetMemPool_fn = hip::hipDeviceSetMemPool;
  ptrDispatchTable->hipDeviceSetSharedMemConfig_fn = hip::hipDeviceSetSharedMemConfig;
  ptrDispatchTable->hipDeviceSynchronize_fn = hip::hipDeviceSynchronize;
  ptrDispatchTable->hipDeviceTotalMem_fn = hip::hipDeviceTotalMem;
  ptrDispatchTable->hipDriverGetVersion_fn = hip::hipDriverGetVersion;
  ptrDispatchTable->hipDrvGetErrorName_fn = hip::hipDrvGetErrorName;
  ptrDispatchTable->hipDrvGetErrorString_fn = hip::hipDrvGetErrorString;
  ptrDispatchTable->hipDrvGraphAddMemcpyNode_fn = hip::hipDrvGraphAddMemcpyNode;
  ptrDispatchTable->hipDrvMemcpy2DUnaligned_fn = hip::hipDrvMemcpy2DUnaligned;
  ptrDispatchTable->hipDrvMemcpy3D_fn = hip::hipDrvMemcpy3D;
  ptrDispatchTable->hipDrvMemcpy3DAsync_fn = hip::hipDrvMemcpy3DAsync;
  ptrDispatchTable->hipDrvPointerGetAttributes_fn = hip::hipDrvPointerGetAttributes;
  ptrDispatchTable->hipEventCreate_fn = hip::hipEventCreate;
  ptrDispatchTable->hipEventCreateWithFlags_fn = hip::hipEventCreateWithFlags;
  ptrDispatchTable->hipEventDestroy_fn = hip::hipEventDestroy;
  ptrDispatchTable->hipEventElapsedTime_fn = hip::hipEventElapsedTime;
  ptrDispatchTable->hipEventQuery_fn = hip::hipEventQuery;
  ptrDispatchTable->hipEventRecord_fn = hip::hipEventRecord;
  ptrDispatchTable->hipEventSynchronize_fn = hip::hipEventSynchronize;
  ptrDispatchTable->hipExtGetLinkTypeAndHopCount_fn = hip::hipExtGetLinkTypeAndHopCount;
  ptrDispatchTable->hipExtLaunchKernel_fn = hip::hipExtLaunchKernel;
  ptrDispatchTable->hipExtLaunchMultiKernelMultiDevice_fn = hip::hipExtLaunchMultiKernelMultiDevice;
  ptrDispatchTable->hipExtMallocWithFlags_fn = hip::hipExtMallocWithFlags;
  ptrDispatchTable->hipExtStreamCreateWithCUMask_fn = hip::hipExtStreamCreateWithCUMask;
  ptrDispatchTable->hipExtStreamGetCUMask_fn = hip::hipExtStreamGetCUMask;
  ptrDispatchTable->hipExternalMemoryGetMappedBuffer_fn = hip::hipExternalMemoryGetMappedBuffer;
  ptrDispatchTable->hipFree_fn = hip::hipFree;
  ptrDispatchTable->hipFreeArray_fn = hip::hipFreeArray;
  ptrDispatchTable->hipFreeAsync_fn = hip::hipFreeAsync;
  ptrDispatchTable->hipFreeHost_fn = hip::hipFreeHost;
  ptrDispatchTable->hipFreeMipmappedArray_fn = hip::hipFreeMipmappedArray;
  ptrDispatchTable->hipFuncGetAttribute_fn = hip::hipFuncGetAttribute;
  ptrDispatchTable->hipFuncGetAttributes_fn = hip::hipFuncGetAttributes;
  ptrDispatchTable->hipFuncSetAttribute_fn = hip::hipFuncSetAttribute;
  ptrDispatchTable->hipFuncSetCacheConfig_fn = hip::hipFuncSetCacheConfig;
  ptrDispatchTable->hipFuncSetSharedMemConfig_fn = hip::hipFuncSetSharedMemConfig;
  ptrDispatchTable->hipGLGetDevices_fn = hip::hipGLGetDevices;
  ptrDispatchTable->hipGetChannelDesc_fn = hip::hipGetChannelDesc;
  ptrDispatchTable->hipGetDevice_fn = hip::hipGetDevice;
  ptrDispatchTable->hipGetDeviceCount_fn = hip::hipGetDeviceCount;
  ptrDispatchTable->hipGetDeviceFlags_fn = hip::hipGetDeviceFlags;
  ptrDispatchTable->hipGetDevicePropertiesR0600_fn = hip::hipGetDevicePropertiesR0600;
  ptrDispatchTable->hipGetErrorName_fn = hip::hipGetErrorName;
  ptrDispatchTable->hipGetErrorString_fn = hip::hipGetErrorString;
  ptrDispatchTable->hipGetLastError_fn = hip::hipGetLastError;
  ptrDispatchTable->hipGetMipmappedArrayLevel_fn = hip::hipGetMipmappedArrayLevel;
  ptrDispatchTable->hipGetSymbolAddress_fn = hip::hipGetSymbolAddress;
  ptrDispatchTable->hipGetSymbolSize_fn = hip::hipGetSymbolSize;
  ptrDispatchTable->hipGetTextureAlignmentOffset_fn = hip::hipGetTextureAlignmentOffset;
  ptrDispatchTable->hipGetTextureObjectResourceDesc_fn = hip::hipGetTextureObjectResourceDesc;
  ptrDispatchTable->hipGetTextureObjectResourceViewDesc_fn =
      hip::hipGetTextureObjectResourceViewDesc;
  ptrDispatchTable->hipGetTextureObjectTextureDesc_fn = hip::hipGetTextureObjectTextureDesc;
  ptrDispatchTable->hipGetTextureReference_fn = hip::hipGetTextureReference;
  ptrDispatchTable->hipGraphAddChildGraphNode_fn = hip::hipGraphAddChildGraphNode;
  ptrDispatchTable->hipGraphAddDependencies_fn = hip::hipGraphAddDependencies;
  ptrDispatchTable->hipGraphAddEmptyNode_fn = hip::hipGraphAddEmptyNode;
  ptrDispatchTable->hipGraphAddEventRecordNode_fn = hip::hipGraphAddEventRecordNode;
  ptrDispatchTable->hipGraphAddEventWaitNode_fn = hip::hipGraphAddEventWaitNode;
  ptrDispatchTable->hipGraphAddHostNode_fn = hip::hipGraphAddHostNode;
  ptrDispatchTable->hipGraphAddKernelNode_fn = hip::hipGraphAddKernelNode;
  ptrDispatchTable->hipGraphAddMemAllocNode_fn = hip::hipGraphAddMemAllocNode;
  ptrDispatchTable->hipGraphAddMemFreeNode_fn = hip::hipGraphAddMemFreeNode;
  ptrDispatchTable->hipGraphAddMemcpyNode_fn = hip::hipGraphAddMemcpyNode;
  ptrDispatchTable->hipGraphAddMemcpyNode1D_fn = hip::hipGraphAddMemcpyNode1D;
  ptrDispatchTable->hipGraphAddMemcpyNodeFromSymbol_fn = hip::hipGraphAddMemcpyNodeFromSymbol;
  ptrDispatchTable->hipGraphAddMemcpyNodeToSymbol_fn = hip::hipGraphAddMemcpyNodeToSymbol;
  ptrDispatchTable->hipGraphAddMemsetNode_fn = hip::hipGraphAddMemsetNode;
  ptrDispatchTable->hipGraphChildGraphNodeGetGraph_fn = hip::hipGraphChildGraphNodeGetGraph;
  ptrDispatchTable->hipGraphClone_fn = hip::hipGraphClone;
  ptrDispatchTable->hipGraphCreate_fn = hip::hipGraphCreate;
  ptrDispatchTable->hipGraphDebugDotPrint_fn = hip::hipGraphDebugDotPrint;
  ptrDispatchTable->hipGraphDestroy_fn = hip::hipGraphDestroy;
  ptrDispatchTable->hipGraphDestroyNode_fn = hip::hipGraphDestroyNode;
  ptrDispatchTable->hipGraphEventRecordNodeGetEvent_fn = hip::hipGraphEventRecordNodeGetEvent;
  ptrDispatchTable->hipGraphEventRecordNodeSetEvent_fn = hip::hipGraphEventRecordNodeSetEvent;
  ptrDispatchTable->hipGraphEventWaitNodeGetEvent_fn = hip::hipGraphEventWaitNodeGetEvent;
  ptrDispatchTable->hipGraphEventWaitNodeSetEvent_fn = hip::hipGraphEventWaitNodeSetEvent;
  ptrDispatchTable->hipGraphExecChildGraphNodeSetParams_fn =
      hip::hipGraphExecChildGraphNodeSetParams;
  ptrDispatchTable->hipGraphExecDestroy_fn = hip::hipGraphExecDestroy;
  ptrDispatchTable->hipGraphExecEventRecordNodeSetEvent_fn =
      hip::hipGraphExecEventRecordNodeSetEvent;
  ptrDispatchTable->hipGraphExecEventWaitNodeSetEvent_fn = hip::hipGraphExecEventWaitNodeSetEvent;
  ptrDispatchTable->hipGraphExecHostNodeSetParams_fn = hip::hipGraphExecHostNodeSetParams;
  ptrDispatchTable->hipGraphExecKernelNodeSetParams_fn = hip::hipGraphExecKernelNodeSetParams;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParams_fn = hip::hipGraphExecMemcpyNodeSetParams;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParams1D_fn = hip::hipGraphExecMemcpyNodeSetParams1D;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParamsFromSymbol_fn =
      hip::hipGraphExecMemcpyNodeSetParamsFromSymbol;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParamsToSymbol_fn =
      hip::hipGraphExecMemcpyNodeSetParamsToSymbol;
  ptrDispatchTable->hipGraphExecMemsetNodeSetParams_fn = hip::hipGraphExecMemsetNodeSetParams;
  ptrDispatchTable->hipGraphExecUpdate_fn = hip::hipGraphExecUpdate;
  ptrDispatchTable->hipGraphGetEdges_fn = hip::hipGraphGetEdges;
  ptrDispatchTable->hipGraphGetNodes_fn = hip::hipGraphGetNodes;
  ptrDispatchTable->hipGraphGetRootNodes_fn = hip::hipGraphGetRootNodes;
  ptrDispatchTable->hipGraphHostNodeGetParams_fn = hip::hipGraphHostNodeGetParams;
  ptrDispatchTable->hipGraphHostNodeSetParams_fn = hip::hipGraphHostNodeSetParams;
  ptrDispatchTable->hipGraphInstantiate_fn = hip::hipGraphInstantiate;
  ptrDispatchTable->hipGraphInstantiateWithFlags_fn = hip::hipGraphInstantiateWithFlags;
  ptrDispatchTable->hipGraphKernelNodeCopyAttributes_fn = hip::hipGraphKernelNodeCopyAttributes;
  ptrDispatchTable->hipGraphKernelNodeGetAttribute_fn = hip::hipGraphKernelNodeGetAttribute;
  ptrDispatchTable->hipGraphKernelNodeGetParams_fn = hip::hipGraphKernelNodeGetParams;
  ptrDispatchTable->hipGraphKernelNodeSetAttribute_fn = hip::hipGraphKernelNodeSetAttribute;
  ptrDispatchTable->hipGraphKernelNodeSetParams_fn = hip::hipGraphKernelNodeSetParams;
  ptrDispatchTable->hipGraphLaunch_fn = hip::hipGraphLaunch;
  ptrDispatchTable->hipGraphMemAllocNodeGetParams_fn = hip::hipGraphMemAllocNodeGetParams;
  ptrDispatchTable->hipGraphMemFreeNodeGetParams_fn = hip::hipGraphMemFreeNodeGetParams;
  ptrDispatchTable->hipGraphMemcpyNodeGetParams_fn = hip::hipGraphMemcpyNodeGetParams;
  ptrDispatchTable->hipGraphMemcpyNodeSetParams_fn = hip::hipGraphMemcpyNodeSetParams;
  ptrDispatchTable->hipGraphMemcpyNodeSetParams1D_fn = hip::hipGraphMemcpyNodeSetParams1D;
  ptrDispatchTable->hipGraphMemcpyNodeSetParamsFromSymbol_fn =
      hip::hipGraphMemcpyNodeSetParamsFromSymbol;
  ptrDispatchTable->hipGraphMemcpyNodeSetParamsToSymbol_fn =
      hip::hipGraphMemcpyNodeSetParamsToSymbol;
  ptrDispatchTable->hipGraphMemsetNodeGetParams_fn = hip::hipGraphMemsetNodeGetParams;
  ptrDispatchTable->hipGraphMemsetNodeSetParams_fn = hip::hipGraphMemsetNodeSetParams;
  ptrDispatchTable->hipGraphNodeFindInClone_fn = hip::hipGraphNodeFindInClone;
  ptrDispatchTable->hipGraphNodeGetDependencies_fn = hip::hipGraphNodeGetDependencies;
  ptrDispatchTable->hipGraphNodeGetDependentNodes_fn = hip::hipGraphNodeGetDependentNodes;
  ptrDispatchTable->hipGraphNodeGetEnabled_fn = hip::hipGraphNodeGetEnabled;
  ptrDispatchTable->hipGraphNodeGetType_fn = hip::hipGraphNodeGetType;
  ptrDispatchTable->hipGraphNodeSetEnabled_fn = hip::hipGraphNodeSetEnabled;
  ptrDispatchTable->hipGraphReleaseUserObject_fn = hip::hipGraphReleaseUserObject;
  ptrDispatchTable->hipGraphRemoveDependencies_fn = hip::hipGraphRemoveDependencies;
  ptrDispatchTable->hipGraphRetainUserObject_fn = hip::hipGraphRetainUserObject;
  ptrDispatchTable->hipGraphUpload_fn = hip::hipGraphUpload;
  ptrDispatchTable->hipGraphicsGLRegisterBuffer_fn = hip::hipGraphicsGLRegisterBuffer;
  ptrDispatchTable->hipGraphicsGLRegisterImage_fn = hip::hipGraphicsGLRegisterImage;
  ptrDispatchTable->hipGraphicsMapResources_fn = hip::hipGraphicsMapResources;
  ptrDispatchTable->hipGraphicsResourceGetMappedPointer_fn =
      hip::hipGraphicsResourceGetMappedPointer;
  ptrDispatchTable->hipGraphicsSubResourceGetMappedArray_fn =
      hip::hipGraphicsSubResourceGetMappedArray;
  ptrDispatchTable->hipGraphicsUnmapResources_fn = hip::hipGraphicsUnmapResources;
  ptrDispatchTable->hipGraphicsUnregisterResource_fn = hip::hipGraphicsUnregisterResource;
  ptrDispatchTable->hipHostAlloc_fn = hip::hipHostAlloc;
  ptrDispatchTable->hipHostFree_fn = hip::hipHostFree;
  ptrDispatchTable->hipHostGetDevicePointer_fn = hip::hipHostGetDevicePointer;
  ptrDispatchTable->hipHostGetFlags_fn = hip::hipHostGetFlags;
  ptrDispatchTable->hipHostMalloc_fn = hip::hipHostMalloc;
  ptrDispatchTable->hipHostRegister_fn = hip::hipHostRegister;
  ptrDispatchTable->hipHostUnregister_fn = hip::hipHostUnregister;
  ptrDispatchTable->hipImportExternalMemory_fn = hip::hipImportExternalMemory;
  ptrDispatchTable->hipImportExternalSemaphore_fn = hip::hipImportExternalSemaphore;
  ptrDispatchTable->hipInit_fn = hip::hipInit;
  ptrDispatchTable->hipIpcCloseMemHandle_fn = hip::hipIpcCloseMemHandle;
  ptrDispatchTable->hipIpcGetEventHandle_fn = hip::hipIpcGetEventHandle;
  ptrDispatchTable->hipIpcGetMemHandle_fn = hip::hipIpcGetMemHandle;
  ptrDispatchTable->hipIpcOpenEventHandle_fn = hip::hipIpcOpenEventHandle;
  ptrDispatchTable->hipIpcOpenMemHandle_fn = hip::hipIpcOpenMemHandle;
  ptrDispatchTable->hipKernelNameRef_fn = hip::hipKernelNameRef;
  ptrDispatchTable->hipKernelNameRefByPtr_fn = hip::hipKernelNameRefByPtr;
  ptrDispatchTable->hipLaunchByPtr_fn = hip::hipLaunchByPtr;
  ptrDispatchTable->hipLaunchCooperativeKernel_fn = hip::hipLaunchCooperativeKernel;
  ptrDispatchTable->hipLaunchCooperativeKernelMultiDevice_fn =
      hip::hipLaunchCooperativeKernelMultiDevice;
  ptrDispatchTable->hipLaunchHostFunc_fn = hip::hipLaunchHostFunc;
  ptrDispatchTable->hipLaunchKernel_fn = hip::hipLaunchKernel;
  ptrDispatchTable->hipMalloc_fn = hip::hipMalloc;
  ptrDispatchTable->hipMalloc3D_fn = hip::hipMalloc3D;
  ptrDispatchTable->hipMalloc3DArray_fn = hip::hipMalloc3DArray;
  ptrDispatchTable->hipMallocArray_fn = hip::hipMallocArray;
  ptrDispatchTable->hipMallocAsync_fn = hip::hipMallocAsync;
  ptrDispatchTable->hipMallocFromPoolAsync_fn = hip::hipMallocFromPoolAsync;
  ptrDispatchTable->hipMallocHost_fn = hip::hipMallocHost;
  ptrDispatchTable->hipMallocManaged_fn = hip::hipMallocManaged;
  ptrDispatchTable->hipMallocMipmappedArray_fn = hip::hipMallocMipmappedArray;
  ptrDispatchTable->hipMallocPitch_fn = hip::hipMallocPitch;
  ptrDispatchTable->hipMemAddressFree_fn = hip::hipMemAddressFree;
  ptrDispatchTable->hipMemAddressReserve_fn = hip::hipMemAddressReserve;
  ptrDispatchTable->hipMemAdvise_fn = hip::hipMemAdvise;
  ptrDispatchTable->hipMemAllocHost_fn = hip::hipMemAllocHost;
  ptrDispatchTable->hipMemAllocPitch_fn = hip::hipMemAllocPitch;
  ptrDispatchTable->hipMemCreate_fn = hip::hipMemCreate;
  ptrDispatchTable->hipMemExportToShareableHandle_fn = hip::hipMemExportToShareableHandle;
  ptrDispatchTable->hipMemGetAccess_fn = hip::hipMemGetAccess;
  ptrDispatchTable->hipMemGetAddressRange_fn = hip::hipMemGetAddressRange;
  ptrDispatchTable->hipMemGetAllocationGranularity_fn = hip::hipMemGetAllocationGranularity;
  ptrDispatchTable->hipMemGetAllocationPropertiesFromHandle_fn =
      hip::hipMemGetAllocationPropertiesFromHandle;
  ptrDispatchTable->hipMemGetInfo_fn = hip::hipMemGetInfo;
  ptrDispatchTable->hipMemImportFromShareableHandle_fn = hip::hipMemImportFromShareableHandle;
  ptrDispatchTable->hipMemMap_fn = hip::hipMemMap;
  ptrDispatchTable->hipMemMapArrayAsync_fn = hip::hipMemMapArrayAsync;
  ptrDispatchTable->hipMemPoolCreate_fn = hip::hipMemPoolCreate;
  ptrDispatchTable->hipMemPoolDestroy_fn = hip::hipMemPoolDestroy;
  ptrDispatchTable->hipMemPoolExportPointer_fn = hip::hipMemPoolExportPointer;
  ptrDispatchTable->hipMemPoolExportToShareableHandle_fn = hip::hipMemPoolExportToShareableHandle;
  ptrDispatchTable->hipMemPoolGetAccess_fn = hip::hipMemPoolGetAccess;
  ptrDispatchTable->hipMemPoolGetAttribute_fn = hip::hipMemPoolGetAttribute;
  ptrDispatchTable->hipMemPoolImportFromShareableHandle_fn =
      hip::hipMemPoolImportFromShareableHandle;
  ptrDispatchTable->hipMemPoolImportPointer_fn = hip::hipMemPoolImportPointer;
  ptrDispatchTable->hipMemPoolSetAccess_fn = hip::hipMemPoolSetAccess;
  ptrDispatchTable->hipMemPoolSetAttribute_fn = hip::hipMemPoolSetAttribute;
  ptrDispatchTable->hipMemPoolTrimTo_fn = hip::hipMemPoolTrimTo;
  ptrDispatchTable->hipMemPrefetchAsync_fn = hip::hipMemPrefetchAsync;
  ptrDispatchTable->hipMemPtrGetInfo_fn = hip::hipMemPtrGetInfo;
  ptrDispatchTable->hipMemRangeGetAttribute_fn = hip::hipMemRangeGetAttribute;
  ptrDispatchTable->hipMemRangeGetAttributes_fn = hip::hipMemRangeGetAttributes;
  ptrDispatchTable->hipMemRelease_fn = hip::hipMemRelease;
  ptrDispatchTable->hipMemRetainAllocationHandle_fn = hip::hipMemRetainAllocationHandle;
  ptrDispatchTable->hipMemSetAccess_fn = hip::hipMemSetAccess;
  ptrDispatchTable->hipMemUnmap_fn = hip::hipMemUnmap;
  ptrDispatchTable->hipMemcpy_fn = hip::hipMemcpy;
  ptrDispatchTable->hipMemcpy2D_fn = hip::hipMemcpy2D;
  ptrDispatchTable->hipMemcpy2DAsync_fn = hip::hipMemcpy2DAsync;
  ptrDispatchTable->hipMemcpy2DFromArray_fn = hip::hipMemcpy2DFromArray;
  ptrDispatchTable->hipMemcpy2DFromArrayAsync_fn = hip::hipMemcpy2DFromArrayAsync;
  ptrDispatchTable->hipMemcpy2DToArray_fn = hip::hipMemcpy2DToArray;
  ptrDispatchTable->hipMemcpy2DToArrayAsync_fn = hip::hipMemcpy2DToArrayAsync;
  ptrDispatchTable->hipMemcpy3D_fn = hip::hipMemcpy3D;
  ptrDispatchTable->hipMemcpy3DAsync_fn = hip::hipMemcpy3DAsync;
  ptrDispatchTable->hipMemcpyAsync_fn = hip::hipMemcpyAsync;
  ptrDispatchTable->hipMemcpyAtoH_fn = hip::hipMemcpyAtoH;
  ptrDispatchTable->hipMemcpyDtoD_fn = hip::hipMemcpyDtoD;
  ptrDispatchTable->hipMemcpyDtoDAsync_fn = hip::hipMemcpyDtoDAsync;
  ptrDispatchTable->hipMemcpyDtoH_fn = hip::hipMemcpyDtoH;
  ptrDispatchTable->hipMemcpyDtoHAsync_fn = hip::hipMemcpyDtoHAsync;
  ptrDispatchTable->hipMemcpyFromArray_fn = hip::hipMemcpyFromArray;
  ptrDispatchTable->hipMemcpyFromSymbol_fn = hip::hipMemcpyFromSymbol;
  ptrDispatchTable->hipMemcpyFromSymbolAsync_fn = hip::hipMemcpyFromSymbolAsync;
  ptrDispatchTable->hipMemcpyHtoA_fn = hip::hipMemcpyHtoA;
  ptrDispatchTable->hipMemcpyHtoD_fn = hip::hipMemcpyHtoD;
  ptrDispatchTable->hipMemcpyHtoDAsync_fn = hip::hipMemcpyHtoDAsync;
  ptrDispatchTable->hipMemcpyParam2D_fn = hip::hipMemcpyParam2D;
  ptrDispatchTable->hipMemcpyParam2DAsync_fn = hip::hipMemcpyParam2DAsync;
  ptrDispatchTable->hipMemcpyPeer_fn = hip::hipMemcpyPeer;
  ptrDispatchTable->hipMemcpyPeerAsync_fn = hip::hipMemcpyPeerAsync;
  ptrDispatchTable->hipMemcpyToArray_fn = hip::hipMemcpyToArray;
  ptrDispatchTable->hipMemcpyToSymbol_fn = hip::hipMemcpyToSymbol;
  ptrDispatchTable->hipMemcpyToSymbolAsync_fn = hip::hipMemcpyToSymbolAsync;
  ptrDispatchTable->hipMemcpyWithStream_fn = hip::hipMemcpyWithStream;
  ptrDispatchTable->hipMemset_fn = hip::hipMemset;
  ptrDispatchTable->hipMemset2D_fn = hip::hipMemset2D;
  ptrDispatchTable->hipMemset2DAsync_fn = hip::hipMemset2DAsync;
  ptrDispatchTable->hipMemset3D_fn = hip::hipMemset3D;
  ptrDispatchTable->hipMemset3DAsync_fn = hip::hipMemset3DAsync;
  ptrDispatchTable->hipMemsetAsync_fn = hip::hipMemsetAsync;
  ptrDispatchTable->hipMemsetD16_fn = hip::hipMemsetD16;
  ptrDispatchTable->hipMemsetD16Async_fn = hip::hipMemsetD16Async;
  ptrDispatchTable->hipMemsetD32_fn = hip::hipMemsetD32;
  ptrDispatchTable->hipMemsetD32Async_fn = hip::hipMemsetD32Async;
  ptrDispatchTable->hipMemsetD8_fn = hip::hipMemsetD8;
  ptrDispatchTable->hipMemsetD8Async_fn = hip::hipMemsetD8Async;
  ptrDispatchTable->hipMipmappedArrayCreate_fn = hip::hipMipmappedArrayCreate;
  ptrDispatchTable->hipMipmappedArrayDestroy_fn = hip::hipMipmappedArrayDestroy;
  ptrDispatchTable->hipMipmappedArrayGetLevel_fn = hip::hipMipmappedArrayGetLevel;
  ptrDispatchTable->hipModuleGetFunction_fn = hip::hipModuleGetFunction;
  ptrDispatchTable->hipModuleGetGlobal_fn = hip::hipModuleGetGlobal;
  ptrDispatchTable->hipModuleGetTexRef_fn = hip::hipModuleGetTexRef;
  ptrDispatchTable->hipModuleLaunchCooperativeKernel_fn = hip::hipModuleLaunchCooperativeKernel;
  ptrDispatchTable->hipModuleLaunchCooperativeKernelMultiDevice_fn =
      hip::hipModuleLaunchCooperativeKernelMultiDevice;
  ptrDispatchTable->hipModuleLaunchKernel_fn = hip::hipModuleLaunchKernel;
  ptrDispatchTable->hipModuleLoad_fn = hip::hipModuleLoad;
  ptrDispatchTable->hipModuleLoadData_fn = hip::hipModuleLoadData;
  ptrDispatchTable->hipModuleLoadDataEx_fn = hip::hipModuleLoadDataEx;
  ptrDispatchTable->hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn =
      hip::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
  ptrDispatchTable->hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn =
      hip::hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  ptrDispatchTable->hipModuleOccupancyMaxPotentialBlockSize_fn =
      hip::hipModuleOccupancyMaxPotentialBlockSize;
  ptrDispatchTable->hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn =
      hip::hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
  ptrDispatchTable->hipModuleUnload_fn = hip::hipModuleUnload;
  ptrDispatchTable->hipOccupancyMaxActiveBlocksPerMultiprocessor_fn =
      hip::hipOccupancyMaxActiveBlocksPerMultiprocessor;
  ptrDispatchTable->hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn =
      hip::hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  ptrDispatchTable->hipOccupancyMaxPotentialBlockSize_fn = hip::hipOccupancyMaxPotentialBlockSize;
  ptrDispatchTable->hipPeekAtLastError_fn = hip::hipPeekAtLastError;
  ptrDispatchTable->hipPointerGetAttribute_fn = hip::hipPointerGetAttribute;
  ptrDispatchTable->hipPointerGetAttributes_fn = hip::hipPointerGetAttributes;
  ptrDispatchTable->hipPointerSetAttribute_fn = hip::hipPointerSetAttribute;
  ptrDispatchTable->hipProfilerStart_fn = hip::hipProfilerStart;
  ptrDispatchTable->hipProfilerStop_fn = hip::hipProfilerStop;
  ptrDispatchTable->hipRuntimeGetVersion_fn = hip::hipRuntimeGetVersion;
  ptrDispatchTable->hipSetDevice_fn = hip::hipSetDevice;
  ptrDispatchTable->hipSetDeviceFlags_fn = hip::hipSetDeviceFlags;
  ptrDispatchTable->hipSetupArgument_fn = hip::hipSetupArgument;
  ptrDispatchTable->hipSignalExternalSemaphoresAsync_fn = hip::hipSignalExternalSemaphoresAsync;
  ptrDispatchTable->hipStreamAddCallback_fn = hip::hipStreamAddCallback;
  ptrDispatchTable->hipStreamAttachMemAsync_fn = hip::hipStreamAttachMemAsync;
  ptrDispatchTable->hipStreamBeginCapture_fn = hip::hipStreamBeginCapture;
  ptrDispatchTable->hipStreamCreate_fn = hip::hipStreamCreate;
  ptrDispatchTable->hipStreamCreateWithFlags_fn = hip::hipStreamCreateWithFlags;
  ptrDispatchTable->hipStreamCreateWithPriority_fn = hip::hipStreamCreateWithPriority;
  ptrDispatchTable->hipStreamDestroy_fn = hip::hipStreamDestroy;
  ptrDispatchTable->hipStreamEndCapture_fn = hip::hipStreamEndCapture;
  ptrDispatchTable->hipStreamGetCaptureInfo_fn = hip::hipStreamGetCaptureInfo;
  ptrDispatchTable->hipStreamGetCaptureInfo_v2_fn = hip::hipStreamGetCaptureInfo_v2;
  ptrDispatchTable->hipStreamGetDevice_fn = hip::hipStreamGetDevice;
  ptrDispatchTable->hipStreamGetFlags_fn = hip::hipStreamGetFlags;
  ptrDispatchTable->hipStreamGetPriority_fn = hip::hipStreamGetPriority;
  ptrDispatchTable->hipStreamIsCapturing_fn = hip::hipStreamIsCapturing;
  ptrDispatchTable->hipStreamQuery_fn = hip::hipStreamQuery;
  ptrDispatchTable->hipStreamSynchronize_fn = hip::hipStreamSynchronize;
  ptrDispatchTable->hipStreamUpdateCaptureDependencies_fn = hip::hipStreamUpdateCaptureDependencies;
  ptrDispatchTable->hipStreamWaitEvent_fn = hip::hipStreamWaitEvent;
  ptrDispatchTable->hipStreamWaitValue32_fn = hip::hipStreamWaitValue32;
  ptrDispatchTable->hipStreamWaitValue64_fn = hip::hipStreamWaitValue64;
  ptrDispatchTable->hipStreamWriteValue32_fn = hip::hipStreamWriteValue32;
  ptrDispatchTable->hipStreamWriteValue64_fn = hip::hipStreamWriteValue64;
  ptrDispatchTable->hipTexObjectCreate_fn = hip::hipTexObjectCreate;
  ptrDispatchTable->hipTexObjectDestroy_fn = hip::hipTexObjectDestroy;
  ptrDispatchTable->hipTexObjectGetResourceDesc_fn = hip::hipTexObjectGetResourceDesc;
  ptrDispatchTable->hipTexObjectGetResourceViewDesc_fn = hip::hipTexObjectGetResourceViewDesc;
  ptrDispatchTable->hipTexObjectGetTextureDesc_fn = hip::hipTexObjectGetTextureDesc;
  ptrDispatchTable->hipTexRefGetAddress_fn = hip::hipTexRefGetAddress;
  ptrDispatchTable->hipTexRefGetAddressMode_fn = hip::hipTexRefGetAddressMode;
  ptrDispatchTable->hipTexRefGetFilterMode_fn = hip::hipTexRefGetFilterMode;
  ptrDispatchTable->hipTexRefGetFlags_fn = hip::hipTexRefGetFlags;
  ptrDispatchTable->hipTexRefGetFormat_fn = hip::hipTexRefGetFormat;
  ptrDispatchTable->hipTexRefGetMaxAnisotropy_fn = hip::hipTexRefGetMaxAnisotropy;
  ptrDispatchTable->hipTexRefGetMipMappedArray_fn = hip::hipTexRefGetMipMappedArray;
  ptrDispatchTable->hipTexRefGetMipmapFilterMode_fn = hip::hipTexRefGetMipmapFilterMode;
  ptrDispatchTable->hipTexRefGetMipmapLevelBias_fn = hip::hipTexRefGetMipmapLevelBias;
  ptrDispatchTable->hipTexRefGetMipmapLevelClamp_fn = hip::hipTexRefGetMipmapLevelClamp;
  ptrDispatchTable->hipTexRefSetAddress_fn = hip::hipTexRefSetAddress;
  ptrDispatchTable->hipTexRefSetAddress2D_fn = hip::hipTexRefSetAddress2D;
  ptrDispatchTable->hipTexRefSetAddressMode_fn = hip::hipTexRefSetAddressMode;
  ptrDispatchTable->hipTexRefSetArray_fn = hip::hipTexRefSetArray;
  ptrDispatchTable->hipTexRefSetBorderColor_fn = hip::hipTexRefSetBorderColor;
  ptrDispatchTable->hipTexRefSetFilterMode_fn = hip::hipTexRefSetFilterMode;
  ptrDispatchTable->hipTexRefSetFlags_fn = hip::hipTexRefSetFlags;
  ptrDispatchTable->hipTexRefSetFormat_fn = hip::hipTexRefSetFormat;
  ptrDispatchTable->hipTexRefSetMaxAnisotropy_fn = hip::hipTexRefSetMaxAnisotropy;
  ptrDispatchTable->hipTexRefSetMipmapFilterMode_fn = hip::hipTexRefSetMipmapFilterMode;
  ptrDispatchTable->hipTexRefSetMipmapLevelBias_fn = hip::hipTexRefSetMipmapLevelBias;
  ptrDispatchTable->hipTexRefSetMipmapLevelClamp_fn = hip::hipTexRefSetMipmapLevelClamp;
  ptrDispatchTable->hipTexRefSetMipmappedArray_fn = hip::hipTexRefSetMipmappedArray;
  ptrDispatchTable->hipThreadExchangeStreamCaptureMode_fn = hip::hipThreadExchangeStreamCaptureMode;
  ptrDispatchTable->hipUnbindTexture_fn = hip::hipUnbindTexture;
  ptrDispatchTable->hipUserObjectCreate_fn = hip::hipUserObjectCreate;
  ptrDispatchTable->hipUserObjectRelease_fn = hip::hipUserObjectRelease;
  ptrDispatchTable->hipUserObjectRetain_fn = hip::hipUserObjectRetain;
  ptrDispatchTable->hipWaitExternalSemaphoresAsync_fn = hip::hipWaitExternalSemaphoresAsync;
  ptrDispatchTable->hipCreateChannelDesc_fn = hip::hipCreateChannelDesc;
  ptrDispatchTable->hipCreateSurfaceObject_fn = hip::hipCreateSurfaceObject;
  ptrDispatchTable->hipExtModuleLaunchKernel_fn = hip::hipExtModuleLaunchKernel;
  ptrDispatchTable->hipHccModuleLaunchKernel_fn = hip::hipHccModuleLaunchKernel;
  ptrDispatchTable->hipMemcpy_spt_fn = hip::hipMemcpy_spt;
  ptrDispatchTable->hipMemcpyToSymbol_spt_fn = hip::hipMemcpyToSymbol_spt;
  ptrDispatchTable->hipMemcpyFromSymbol_spt_fn = hip::hipMemcpyFromSymbol_spt;
  ptrDispatchTable->hipMemcpy2D_spt_fn = hip::hipMemcpy2D_spt;
  ptrDispatchTable->hipMemcpy2DFromArray_spt_fn = hip::hipMemcpy2DFromArray_spt;
  ptrDispatchTable->hipMemcpy3D_spt_fn = hip::hipMemcpy3D_spt;
  ptrDispatchTable->hipMemset_spt_fn = hip::hipMemset_spt;
  ptrDispatchTable->hipMemsetAsync_spt_fn = hip::hipMemsetAsync_spt;
  ptrDispatchTable->hipMemset2D_spt_fn = hip::hipMemset2D_spt;
  ptrDispatchTable->hipMemset2DAsync_spt_fn = hip::hipMemset2DAsync_spt;
  ptrDispatchTable->hipMemset3DAsync_spt_fn = hip::hipMemset3DAsync_spt;
  ptrDispatchTable->hipMemset3D_spt_fn = hip::hipMemset3D_spt;
  ptrDispatchTable->hipMemcpyAsync_spt_fn = hip::hipMemcpyAsync_spt;
  ptrDispatchTable->hipMemcpy3DAsync_spt_fn = hip::hipMemcpy3DAsync_spt;
  ptrDispatchTable->hipMemcpy2DAsync_spt_fn = hip::hipMemcpy2DAsync_spt;
  ptrDispatchTable->hipMemcpyFromSymbolAsync_spt_fn = hip::hipMemcpyFromSymbolAsync_spt;
  ptrDispatchTable->hipMemcpyToSymbolAsync_spt_fn = hip::hipMemcpyToSymbolAsync_spt;
  ptrDispatchTable->hipMemcpyFromArray_spt_fn = hip::hipMemcpyFromArray_spt;
  ptrDispatchTable->hipMemcpy2DToArray_spt_fn = hip::hipMemcpy2DToArray_spt;
  ptrDispatchTable->hipMemcpy2DFromArrayAsync_spt_fn = hip::hipMemcpy2DFromArrayAsync_spt;
  ptrDispatchTable->hipMemcpy2DToArrayAsync_spt_fn = hip::hipMemcpy2DToArrayAsync_spt;
  ptrDispatchTable->hipStreamQuery_spt_fn = hip::hipStreamQuery_spt;
  ptrDispatchTable->hipStreamSynchronize_spt_fn = hip::hipStreamSynchronize_spt;
  ptrDispatchTable->hipStreamGetPriority_spt_fn = hip::hipStreamGetPriority_spt;
  ptrDispatchTable->hipStreamWaitEvent_spt_fn = hip::hipStreamWaitEvent_spt;
  ptrDispatchTable->hipStreamGetFlags_spt_fn = hip::hipStreamGetFlags_spt;
  ptrDispatchTable->hipStreamAddCallback_spt_fn = hip::hipStreamAddCallback_spt;
  ptrDispatchTable->hipEventRecord_spt_fn = hip::hipEventRecord_spt;
  ptrDispatchTable->hipLaunchCooperativeKernel_spt_fn = hip::hipLaunchCooperativeKernel_spt;
  ptrDispatchTable->hipLaunchKernel_spt_fn = hip::hipLaunchKernel_spt;
  ptrDispatchTable->hipGraphLaunch_spt_fn = hip::hipGraphLaunch_spt;
  ptrDispatchTable->hipStreamBeginCapture_spt_fn = hip::hipStreamBeginCapture_spt;
  ptrDispatchTable->hipStreamEndCapture_spt_fn = hip::hipStreamEndCapture_spt;
  ptrDispatchTable->hipStreamIsCapturing_spt_fn = hip::hipStreamIsCapturing_spt;
  ptrDispatchTable->hipStreamGetCaptureInfo_spt_fn = hip::hipStreamGetCaptureInfo_spt;
  ptrDispatchTable->hipStreamGetCaptureInfo_v2_spt_fn = hip::hipStreamGetCaptureInfo_v2_spt;
  ptrDispatchTable->hipLaunchHostFunc_spt_fn = hip::hipLaunchHostFunc_spt;
  ptrDispatchTable->hipGetStreamDeviceId_fn = hip::hipGetStreamDeviceId;
  ptrDispatchTable->hipDrvGraphAddMemsetNode_fn = hip::hipDrvGraphAddMemsetNode;
  ptrDispatchTable->hipGetDevicePropertiesR0000_fn = hip::hipGetDevicePropertiesR0000;
}

namespace hip {
namespace {

void ToolCompilerInitImpl(std::string name, HipCompilerDispatchTable* ptr) {
  // implementation of tool init
}
void ToolInit(std::string name, HipCompilerDispatchTable* ptr) {
  static auto _once = std::once_flag{};
  std::call_once(_once, ToolCompilerInitImpl, name, ptr);
}
HipCompilerDispatchTable& GetHipCompilerDispatchTableImpl() {
  // using a static inside a function prevents static initialization fiascos
  static auto _v = HipCompilerDispatchTable{};
  _v.size = sizeof(HipCompilerDispatchTable);
  return _v;
}
HipCompilerDispatchTable* LoadInitialHipCompilerDispatchTable() {
  auto& hip_Compiler_Dispatch_table_ = hip::GetHipCompilerDispatchTableImpl();

  // 2. Change all the function pointers to point to the HIP runtime implementation functions
  UpdateHipCompilerDispatchTable(&hip_Compiler_Dispatch_table_);

  // 3. Profiler Registration
  ToolInit("HipCompilerDispatchTable", &hip_Compiler_Dispatch_table_);
  return &hip_Compiler_Dispatch_table_;
}

void ToolInitImpl(std::string name, HipDispatchTable* ptr) {
  // implementation of tool init
}
void ToolInit(std::string name, HipDispatchTable* ptr) {
  static auto _once = std::once_flag{};
  std::call_once(_once, ToolInitImpl, name, ptr);
}
HipDispatchTable& GetHipDispatchTableImpl() {
  // using a static inside a function prevents static initialization fiascos
  static auto _v = HipDispatchTable{};
  _v.size = sizeof(HipDispatchTable);
  return _v;
}
HipDispatchTable* LoadInitialHipDispatchTable() {
  //   1. Initialize the HIP runtime optimize later
  auto& hip_dispatch_table_ = hip::GetHipDispatchTableImpl();

  // 2. Change all the function pointers to point to the HIP runtime implementation functions
  UpdateHipDispatchTable(&hip_dispatch_table_);

  // 3. Profiler Registration
  ToolInit("HipDispatchTable", &hip_dispatch_table_);
  return &hip_dispatch_table_;
}
}  // namespace
const HipDispatchTable* GetHipDispatchTable() {
  static auto _v = LoadInitialHipDispatchTable();
  return _v;
}
const HipCompilerDispatchTable* GetHipCompilerDispatchTable() {
  static auto _v = LoadInitialHipCompilerDispatchTable();
  return _v;
}
}  // namespace hip
