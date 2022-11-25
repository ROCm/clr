/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "hip_graph_internal.hpp"
#include "platform/command.hpp"
#include "hip_conversions.hpp"
#include "hip_platform.hpp"
#include "hip_event.hpp"
#include "top.hpp"

std::vector<hip::Stream*> g_captureStreams;
amd::Monitor g_captureStreamsLock{"StreamCaptureGlobalList"};

inline hipError_t ihipGraphAddNode(hipGraphNode_t graphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies) {
  graph->AddNode(graphNode);
  for (size_t i = 0; i < numDependencies; i++) {
    if (!hipGraphNode::isNodeValid(pDependencies[i])) {
      return hipErrorInvalidValue;
    }
    pDependencies[i]->AddEdge(graphNode);
  }
  return hipSuccess;
}

hipError_t ihipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipKernelNodeParams* pNodeParams) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pNodeParams == nullptr ||
      pNodeParams->func == nullptr) {
    return hipErrorInvalidValue;
  }

  if (!ihipGraph::isGraphValid(graph)) {
    return hipErrorInvalidValue;
  }

  // If neither 'kernelParams' or 'extra' are provided or if both are provided, return error
  if ((pNodeParams->kernelParams == nullptr && pNodeParams->extra == nullptr) ||
      (pNodeParams->kernelParams != nullptr && pNodeParams->extra != nullptr)) {
    return hipErrorInvalidValue;
  }

  hipError_t status = hipGraphKernelNode::validateKernelParams(pNodeParams);
  if (hipSuccess != status) {
    return status;
  }

  size_t globalWorkSizeX = static_cast<size_t>(pNodeParams->gridDim.x) * pNodeParams->blockDim.x;
  size_t globalWorkSizeY = static_cast<size_t>(pNodeParams->gridDim.y) * pNodeParams->blockDim.y;
  size_t globalWorkSizeZ = static_cast<size_t>(pNodeParams->gridDim.z) * pNodeParams->blockDim.z;
  if (globalWorkSizeX > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeY > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeZ > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidConfiguration;
  }

  *pGraphNode = new hipGraphKernelNode(pNodeParams);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return status;
}

hipError_t ihipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemcpy3DParms* pCopyParams) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pCopyParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipError_t status = ihipMemcpy3D_validate(pCopyParams);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hipGraphMemcpyNode(pCopyParams);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return status;
}

hipError_t ihipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    return hipErrorInvalidValue;
  }
  hipError_t status = ihipMemcpy_validate(dst, src, count, kind);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hipGraphMemcpyNode1D(dst, src, count, kind);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return status;
}

hipError_t ihipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemsetParams* pMemsetParams) {
  if (pGraphNode == nullptr || graph == nullptr || pMemsetParams == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pMemsetParams->height == 0) {
    return hipErrorInvalidValue;
  }
  // The element size must be 1, 2, or 4 bytes
  if (pMemsetParams->elementSize != sizeof(int8_t) &&
      pMemsetParams->elementSize != sizeof(int16_t) &&
      pMemsetParams->elementSize != sizeof(int32_t)) {
    return hipErrorInvalidValue;
  }

  hipError_t status;
  status = ihipGraphMemsetParams_validate(pMemsetParams);
  if (status != hipSuccess) {
    return status;
  }
  if (pMemsetParams->height == 1) {
    status =
        ihipMemset_validate(pMemsetParams->dst, pMemsetParams->value, pMemsetParams->elementSize,
                            pMemsetParams->width * pMemsetParams->elementSize);
  } else {
    auto sizeBytes = pMemsetParams->width * pMemsetParams->height * 1;
    status = ihipMemset3D_validate(
        {pMemsetParams->dst, pMemsetParams->pitch, pMemsetParams->width, pMemsetParams->height},
        pMemsetParams->value, {pMemsetParams->width, pMemsetParams->height, 1}, sizeBytes);
  }
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hipGraphMemsetNode(pMemsetParams);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return status;
}

hipError_t capturehipLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                  dim3& blockDim, void**& args, size_t& sharedMemBytes) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node kernel launch on stream : %p", stream);

  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipKernelNodeParams nodeParams;
  nodeParams.func = const_cast<void*>(hostFunction);
  nodeParams.blockDim = blockDim;
  nodeParams.extra = nullptr;
  nodeParams.gridDim = gridDim;
  nodeParams.kernelParams = args;
  nodeParams.sharedMemBytes = sharedMemBytes;

  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddKernelNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &nodeParams);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t ihipExtLaunchKernel(hipStream_t stream, hipFunction_t f, uint32_t globalWorkSizeX,
                               uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                               uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                               uint32_t localWorkSizeZ, size_t sharedMemBytes, void** kernelParams,
                               void** extra, hipEvent_t startEvent, hipEvent_t stopEvent,
                               uint32_t flags) {
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);

  hipGraphNode_t pGraphNode;
  hipError_t status;
  if (startEvent != nullptr) {
    pGraphNode = new hipGraphEventRecordNode(startEvent);
    status = ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                              s->GetLastCapturedNodes().size());
    if (status != hipSuccess) {
      return status;
    }
    s->SetLastCapturedNode(pGraphNode);
  }
  hipKernelNodeParams nodeParams;
  nodeParams.func = f;
  nodeParams.blockDim = dim3(localWorkSizeX, localWorkSizeY, localWorkSizeZ);
  nodeParams.extra = extra;
  nodeParams.gridDim = dim3(globalWorkSizeX / localWorkSizeX, globalWorkSizeY / localWorkSizeY,
                            globalWorkSizeZ / localWorkSizeZ);
  nodeParams.kernelParams = kernelParams;
  nodeParams.sharedMemBytes = sharedMemBytes;
  status =
      ihipGraphAddKernelNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &nodeParams);

  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  if (stopEvent != nullptr) {
    pGraphNode = new hipGraphEventRecordNode(stopEvent);
    status = ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                              s->GetLastCapturedNodes().size());
    if (status != hipSuccess) {
      return status;
    }
    s->SetLastCapturedNode(pGraphNode);
  }
  return hipSuccess;
}

hipError_t capturehipExtModuleLaunchKernel(hipStream_t& stream, hipFunction_t& f,
                                           uint32_t& globalWorkSizeX, uint32_t& globalWorkSizeY,
                                           uint32_t& globalWorkSizeZ, uint32_t& localWorkSizeX,
                                           uint32_t& localWorkSizeY, uint32_t& localWorkSizeZ,
                                           size_t& sharedMemBytes, void**& kernelParams,
                                           void**& extra, hipEvent_t& startEvent,
                                           hipEvent_t& stopEvent, uint32_t& flags) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node Ext Module launch kernel on stream : %p", stream);
  return ihipExtLaunchKernel(stream, f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                             localWorkSizeX, localWorkSizeY, localWorkSizeZ, sharedMemBytes,
                             kernelParams, extra, startEvent, stopEvent, flags);
}

hipError_t capturehipExtLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                     dim3& blockDim, void**& args, size_t& sharedMemBytes,
                                     hipEvent_t& startEvent, hipEvent_t& stopEvent, int& flags) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node Ext kernel launch on stream : %p", stream);
  return ihipExtLaunchKernel(
      stream, reinterpret_cast<hipFunction_t>(const_cast<void*>(hostFunction)),
      gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z, blockDim.x,
      blockDim.y, blockDim.z, sharedMemBytes, args, nullptr, startEvent, stopEvent, flags);
}

hipError_t capturehipModuleLaunchKernel(hipStream_t& stream, hipFunction_t& f, uint32_t& gridDimX,
                                        uint32_t& gridDimY, uint32_t& gridDimZ, uint32_t& blockDimX,
                                        uint32_t& blockDimY, uint32_t& blockDimZ,
                                        uint32_t& sharedMemBytes, void**& kernelParams,
                                        void**& extra) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node module launch kernel launch on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipKernelNodeParams nodeParams;
  nodeParams.func = f;
  nodeParams.blockDim = {blockDimX, blockDimY, blockDimZ};
  nodeParams.extra = extra;
  nodeParams.gridDim = {gridDimX, gridDimY, gridDimZ};
  nodeParams.kernelParams = kernelParams;
  nodeParams.sharedMemBytes = sharedMemBytes;

  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddKernelNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &nodeParams);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy3DAsync(hipStream_t& stream, const hipMemcpy3DParms*& p) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memcpy3D on stream : %p",
          stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy2DAsync(hipStream_t& stream, void*& dst, size_t& dpitch,
                                   const void*& src, size_t& spitch, size_t& width, size_t& height,
                                   hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memcpy2D on stream : %p",
          stream);
  if (dst == nullptr || src == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;

  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.kind = kind;
  p.srcPtr.ptr = const_cast<void*>(src);
  p.srcPtr.pitch = spitch;
  p.srcArray = nullptr;  // Ignored.

  p.dstPtr.ptr = const_cast<void*>(dst);
  p.dstPtr.pitch = dpitch;
  p.dstArray = nullptr;  // Ignored.

  p.extent = {width, height, 1};

  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy2DFromArrayAsync(hipStream_t& stream, void*& dst, size_t& dpitch,
                                            hipArray_const_t& src, size_t& wOffsetSrc,
                                            size_t& hOffsetSrc, size_t& width, size_t& height,
                                            hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node Memcpy2DFromArray on stream : %p", stream);
  if (src == nullptr || dst == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.srcPos = {wOffsetSrc, hOffsetSrc, 0};
  p.kind = kind;
  p.srcPtr.ptr = nullptr;
  p.srcArray = const_cast<hipArray*>(src);  // Ignored.

  p.kind = kind;
  p.dstPtr.ptr = dst;
  p.dstArray = nullptr;  // Ignored.
  p.dstPtr.pitch = dpitch;
  p.extent = {width / hip::getElementSize(p.srcArray), height, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyFromArrayAsync(hipStream_t& stream, void*& dst, hipArray_const_t& src,
                                          size_t& wOffsetSrc, size_t& hOffsetSrc, size_t& count,
                                          hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node Memcpy2DFromArray on stream : %p", stream);
  if (src == nullptr || dst == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.srcPos = {wOffsetSrc, hOffsetSrc, 0};
  p.kind = kind;
  p.srcPtr.ptr = nullptr;
  p.srcArray = const_cast<hipArray*>(src);

  p.kind = kind;
  p.dstPtr.ptr = dst;
  p.dstArray = nullptr;  // Ignored.
  p.dstPtr.pitch = 0;
  const size_t arrayHeight = (src->height != 0) ? src->height : 1;
  const size_t widthInBytes = count / arrayHeight;
  const size_t height = (count / src->width) / hip::getElementSize(src);
  p.extent = {widthInBytes / hip::getElementSize(p.srcArray), height, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy2DToArrayAsync(hipStream_t& stream, hipArray*& dst, size_t& wOffset,
                                          size_t& hOffset, const void*& src, size_t& spitch,
                                          size_t& width, size_t& height, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node Memcpy2DFromArray on stream : %p", stream);
  if (src == nullptr || dst == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.dstPos = {wOffset, hOffset, 0};
  p.kind = kind;
  p.dstPtr.ptr = nullptr;
  p.dstArray = dst;  // Ignored.

  p.kind = kind;
  p.srcPtr.ptr = const_cast<void*>(src);
  p.srcArray = nullptr;  // Ignored.
  p.srcPtr.pitch = spitch;
  p.extent = {width / hip::getElementSize(p.dstArray), height, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyToArrayAsync(hipStream_t& stream, hipArray_t& dst, size_t& wOffset,
                                        size_t& hOffset, const void*& src, size_t& count,
                                        hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node Memcpy2DFromArray on stream : %p", stream);
  if (src == nullptr || dst == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.dstPos = {wOffset, hOffset, 0};
  p.kind = kind;
  p.dstPtr.ptr = nullptr;
  p.dstArray = dst;  // Ignored.

  p.kind = kind;
  p.srcPtr.ptr = const_cast<void*>(src);
  p.srcArray = nullptr;  // Ignored.
  p.srcPtr.pitch = 0;
  const size_t arrayHeight = (dst->height != 0) ? dst->height : 1;
  const size_t widthInBytes = count / arrayHeight;
  const size_t height = (count / dst->width) / hip::getElementSize(dst);
  p.extent = {widthInBytes / hip::getElementSize(p.dstArray), height, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyParam2DAsync(hipStream_t& stream, const hip_Memcpy2D*& pCopy) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyParam2D on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.srcArray = pCopy->srcArray;
  p.srcPos = {pCopy->srcXInBytes, pCopy->srcY, 0};
  p.srcPtr.pitch = pCopy->srcPitch;
  if (pCopy->srcDevice != nullptr) {
    p.srcPtr.ptr = pCopy->srcDevice;
  }
  if (pCopy->srcHost != nullptr) {
    p.srcPtr.ptr = const_cast<void*>(pCopy->srcHost);
  }
  p.dstArray = pCopy->dstArray;
  p.dstPos = {pCopy->dstXInBytes, pCopy->dstY, 0};
  p.dstPtr.pitch = pCopy->srcPitch;
  if (pCopy->dstDevice != nullptr) {
    p.dstPtr.ptr = pCopy->dstDevice;
  }
  if (pCopy->dstHost != nullptr) {
    p.dstPtr.ptr = const_cast<void*>(pCopy->dstHost);
  }
  p.extent = {pCopy->WidthInBytes, pCopy->Height, 1};
  if (pCopy->srcMemoryType == hipMemoryTypeHost && pCopy->dstMemoryType == hipMemoryTypeDevice) {
    p.kind = hipMemcpyHostToDevice;
  } else if (pCopy->srcMemoryType == hipMemoryTypeDevice &&
             pCopy->dstMemoryType == hipMemoryTypeHost) {
    p.kind = hipMemcpyDeviceToHost;
  } else if (pCopy->srcMemoryType == hipMemoryTypeDevice &&
             pCopy->dstMemoryType == hipMemoryTypeDevice) {
    p.kind = hipMemcpyDeviceToDevice;
  }
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyAtoHAsync(hipStream_t& stream, void*& dstHost, hipArray*& srcArray,
                                     size_t& srcOffset, size_t& ByteCount) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyParam2D on stream : %p", stream);
  if (srcArray == nullptr || dstHost == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.srcArray = srcArray;
  p.srcPos = {srcOffset, 0, 0};
  p.dstPtr.ptr = dstHost;
  p.extent = {ByteCount / hip::getElementSize(p.srcArray), 1, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyHtoAAsync(hipStream_t& stream, hipArray*& dstArray, size_t& dstOffset,
                                     const void*& srcHost, size_t& ByteCount) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyParam2D on stream : %p", stream);
  if (dstArray == nullptr || srcHost == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.dstArray = dstArray;
  p.dstPos = {dstOffset, 0, 0};
  p.srcPtr.ptr = const_cast<void*>(srcHost);
  p.extent = {ByteCount / hip::getElementSize(p.dstArray), 1, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy(hipStream_t stream, void* dst, const void* src, size_t sizeBytes,
                            hipMemcpyKind kind) {
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  std::vector<hipGraphNode_t> pDependencies = s->GetLastCapturedNodes();
  size_t numDependencies = s->GetLastCapturedNodes().size();
  hipGraph_t graph = s->GetCaptureGraph();
  hipError_t status = ihipMemcpy_validate(dst, src, sizeBytes, kind);
  if (status != hipSuccess) {
    return status;
  }
  hipGraphNode_t node = new hipGraphMemcpyNode1D(dst, src, sizeBytes, kind);
  status = ihipGraphAddNode(node, graph, pDependencies.data(), numDependencies);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(node);
  return hipSuccess;
}

hipError_t capturehipMemcpyAsync(hipStream_t& stream, void*& dst, const void*& src,
                                 size_t& sizeBytes, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memcpy1D on stream : %p",
          stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  return capturehipMemcpy(stream, dst, src, sizeBytes, kind);
}

hipError_t capturehipMemcpyHtoDAsync(hipStream_t& stream, hipDeviceptr_t& dstDevice, void*& srcHost,
                                     size_t& ByteCount, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node MemcpyHtoD on stream : %p",
          stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  return capturehipMemcpy(stream, dstDevice, srcHost, ByteCount, kind);
}

hipError_t capturehipMemcpyDtoDAsync(hipStream_t& stream, hipDeviceptr_t& dstDevice,
                                     hipDeviceptr_t& srcDevice, size_t& ByteCount,
                                     hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node hipMemcpyDtoD on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  return capturehipMemcpy(stream, dstDevice, srcDevice, ByteCount, kind);
}

hipError_t capturehipMemcpyDtoHAsync(hipStream_t& stream, void*& dstHost, hipDeviceptr_t& srcDevice,
                                     size_t& ByteCount, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node hipMemcpyDtoH on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  return capturehipMemcpy(stream, dstHost, srcDevice, ByteCount, kind);
}

hipError_t capturehipMemcpyFromSymbolAsync(hipStream_t& stream, void*& dst, const void*& symbol,
                                           size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyFromSymbolNode on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode =
      new hipGraphMemcpyNodeFromSymbol(dst, symbol, sizeBytes, offset, kind);
  status = ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                            s->GetLastCapturedNodes().size());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyToSymbolAsync(hipStream_t& stream, const void*& symbol, const void*& src,
                                         size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyToSymbolNode on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;
  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode = new hipGraphMemcpyNodeToSymbol(symbol, src, sizeBytes, offset, kind);
  status = ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                            s->GetLastCapturedNodes().size());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemsetAsync(hipStream_t& stream, void*& dst, int& value, size_t& valueSize,
                                 size_t& sizeBytes) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset1D on stream : %p",
          stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hipMemsetParams memsetParams = {0};
  memsetParams.dst = dst;
  memsetParams.value = value;
  memsetParams.elementSize = valueSize;
  memsetParams.width = sizeBytes / valueSize;
  memsetParams.height = 1;

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddMemsetNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &memsetParams);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemset2DAsync(hipStream_t& stream, void*& dst, size_t& pitch, int& value,
                                   size_t& width, size_t& height) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset2D on stream : %p",
          stream);
  hipMemsetParams memsetParams = {0};
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  memsetParams.dst = dst;
  memsetParams.value = value;
  memsetParams.width = width;
  memsetParams.height = height;
  memsetParams.pitch = pitch;
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddMemsetNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &memsetParams);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemset3DAsync(hipStream_t& stream, hipPitchedPtr& pitchedDevPtr, int& value,
                                   hipExtent& extent) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset3D on stream : %p",
          stream);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t capturehipEventRecord(hipStream_t& stream, hipEvent_t& event) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node EventRecord on stream : %p, Event %p", stream, event);
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  e->StartCapture(stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  s->SetCaptureEvent(event);
  std::vector<hipGraphNode_t> lastCapturedNodes = s->GetLastCapturedNodes();
  if (!lastCapturedNodes.empty()) {
    e->SetNodesPrevToRecorded(lastCapturedNodes);
  }
  return hipSuccess;
}

hipError_t capturehipStreamWaitEvent(hipEvent_t& event, hipStream_t& stream, unsigned int& flags) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node StreamWaitEvent on stream : %p, Event %p", stream,
          event);
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  if (event == nullptr || stream == nullptr) {
    return hipErrorInvalidValue;
  }
  if (!s->IsOriginStream()) {
    s->SetCaptureGraph(reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->GetCaptureGraph());
    s->SetCaptureId(reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->GetCaptureID());
    s->SetCaptureMode(reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->GetCaptureMode());
    s->SetParentStream(e->GetCaptureStream());
    reinterpret_cast<hip::Stream*>(s->GetParentStream())->SetParallelCaptureStream(stream);
  }
  s->AddCrossCapturedNode(e->GetNodesPrevToRecorded());
  return hipSuccess;
}

hipError_t capturehipLaunchHostFunc(hipStream_t& stream, hipHostFn_t& fn, void*& userData) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset2D on stream : %p",
          stream);
  if (fn == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hipHostNodeParams hostParams = {0};
  hostParams.fn = fn;
  hostParams.userData = userData;
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode = new hipGraphHostNode(&hostParams);
  hipError_t status =
      ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                       s->GetLastCapturedNodes().size());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t hipStreamIsCapturing_common(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  if (pCaptureStatus == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  if (hip::Stream::StreamCaptureBlocking() == true && stream == nullptr) {
    return hipErrorStreamCaptureImplicit;
  }
  if (stream == nullptr) {
    *pCaptureStatus = hipStreamCaptureStatusNone;
  } else {
    *pCaptureStatus = reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus();
  }
  return hipSuccess;
}

hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  HIP_INIT_API(hipStreamIsCapturing, stream, pCaptureStatus);
  HIP_RETURN(hipStreamIsCapturing_common(stream, pCaptureStatus));
}

hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  HIP_INIT_API(hipStreamIsCapturing, stream, pCaptureStatus);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamIsCapturing_common(stream, pCaptureStatus));
}

hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  HIP_INIT_API(hipThreadExchangeStreamCaptureMode, mode);

  if (mode == nullptr || *mode < hipStreamCaptureModeGlobal ||
      *mode > hipStreamCaptureModeRelaxed) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto oldMode = hip::tls.stream_capture_mode_;
  hip::tls.stream_capture_mode_ = *mode;
  *mode = oldMode;

  HIP_RETURN_DURATION(hipSuccess);
}

hipError_t hipStreamBeginCapture_common(hipStream_t stream, hipStreamCaptureMode mode) {
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  // capture cannot be initiated on legacy stream
  if (stream == nullptr) {
    return hipErrorStreamCaptureUnsupported;
  }
  if (mode < hipStreamCaptureModeGlobal ||
      mode > hipStreamCaptureModeRelaxed) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // It can be initiated if the stream is not already in capture mode
  if (s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
    return hipErrorIllegalState;
  }

  s->SetCaptureGraph(new ihipGraph());
  s->SetCaptureId();
  s->SetCaptureMode(mode);
  s->SetOriginStream();
  if (mode != hipStreamCaptureModeRelaxed) {
    hip::tls.capture_streams_.push_back(s);
  }
  if (mode == hipStreamCaptureModeGlobal) {
    amd::ScopedLock lock(g_captureStreamsLock);
    g_captureStreams.push_back(s);
  }
  return hipSuccess;
}

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  HIP_INIT_API(hipStreamBeginCapture, stream, mode);
  HIP_RETURN_DURATION(hipStreamBeginCapture_common(stream, mode));
}

hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode) {
  HIP_INIT_API(hipStreamBeginCapture, stream, mode);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipStreamBeginCapture_common(stream, mode));
}

hipError_t hipStreamEndCapture_common(hipStream_t stream, hipGraph_t* pGraph) {
  if (pGraph == nullptr) {
    return hipErrorInvalidValue;
  }
  if (stream == nullptr) {
    return hipErrorIllegalState;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // Capture status must be active before endCapture can be initiated
  if (s->GetCaptureStatus() == hipStreamCaptureStatusNone) {
    return hipErrorIllegalState;
  }
  // Capture must be ended on the same stream in which it was initiated
  if (!s->IsOriginStream()) {
    return hipErrorStreamCaptureUnmatched;
  }
  // If mode is not hipStreamCaptureModeRelaxed, hipStreamEndCapture must be called on the stream
  // from the same thread
  const auto& it = std::find(hip::tls.capture_streams_.begin(),
                    hip::tls.capture_streams_.end(), s);
  if (s->GetCaptureMode() != hipStreamCaptureModeRelaxed) {
    if (it == hip::tls.capture_streams_.end()) {
      return hipErrorStreamCaptureWrongThread;
    }
    hip::tls.capture_streams_.erase(it);
  }
  if (s->GetCaptureMode() == hipStreamCaptureModeGlobal) {
    amd::ScopedLock lock(g_captureStreamsLock);
    g_captureStreams.erase(std::find(g_captureStreams.begin(), g_captureStreams.end(), s));
  }
  // If capture was invalidated, due to a violation of the rules of stream capture
  if (s->GetCaptureStatus() == hipStreamCaptureStatusInvalidated) {
    *pGraph = nullptr;
    return hipErrorStreamCaptureInvalidated;
  }
  // check if all parallel streams have joined
  // Nodes that are removed from the dependency set via API hipStreamUpdateCaptureDependencies do
  // not result in hipErrorStreamCaptureUnjoined
  // add temporary node to check if all parallel streams have joined
  hipGraphNode_t pGraphNode;
  pGraphNode = new hipGraphEmptyNode();
  hipError_t status =
      ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                       s->GetLastCapturedNodes().size());

  if (s->GetCaptureGraph()->GetLeafNodeCount() > 1) {
    std::vector<hipGraphNode_t> leafNodes = s->GetCaptureGraph()->GetLeafNodes();
    const std::vector<hipGraphNode_t>& removedDepNodes = s->GetRemovedDependencies();
    bool foundInRemovedDep = false;
    for (auto leafNode : leafNodes) {
      for (auto node : removedDepNodes) {
        if (node == leafNode) {
          foundInRemovedDep = true;
        }
      }
    }
    // remove temporary node
    s->GetCaptureGraph()->RemoveNode(pGraphNode);
    if (foundInRemovedDep == false) {
      return hipErrorStreamCaptureUnjoined;
    }
  } else {
    // remove temporary node
    s->GetCaptureGraph()->RemoveNode(pGraphNode);
  }
  *pGraph = s->GetCaptureGraph();
  // end capture on all streams/events part of graph capture
  return s->EndCapture();
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  HIP_INIT_API(hipStreamEndCapture, stream, pGraph);
  HIP_RETURN_DURATION(hipStreamEndCapture_common(stream, pGraph));
}

hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph) {
  HIP_INIT_API(hipStreamEndCapture, stream, pGraph);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipStreamEndCapture_common(stream, pGraph));
}

hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  HIP_INIT_API(hipGraphCreate, pGraph, flags);
  if ((pGraph == nullptr) || (flags != 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraph = new ihipGraph();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphDestroy(hipGraph_t graph) {
  HIP_INIT_API(hipGraphDestroy, graph);
  if (graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // if graph is not valid its destroyed already
  if (!ihipGraph::isGraphValid(graph)) {
    HIP_RETURN(hipErrorIllegalState);
  }
  delete graph;
  HIP_RETURN(hipSuccess);
}


hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphAddKernelNode, pGraphNode, graph, pDependencies, numDependencies,
               pNodeParams);
  if (pGraphNode == nullptr || graph == nullptr || pNodeParams == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN_DURATION(
      ihipGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams));
}

hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams) {
  HIP_INIT_API(hipGraphAddMemcpyNode, pGraphNode, graph, pDependencies, numDependencies,
               pCopyParams);
  if (pGraphNode == nullptr || graph == nullptr || pCopyParams == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN_DURATION(
      ihipGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams));
}

hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNode1D, pGraphNode, graph, pDependencies, numDependencies, dst, src,
               count, kind);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN_DURATION(ihipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies,
                                               dst, src, count, kind));
}

hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParams1D, node, dst, src, count, kind);
  if (node == nullptr || dst == nullptr || src == nullptr || count == 0 || src == dst) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNode1D*>(node)->SetParams(dst, src, count, kind));
}

hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParams1D, hGraphExec, node, dst, src, count, kind);
  if (hGraphExec == nullptr || node == nullptr || dst == nullptr || src == nullptr || count == 0 ||
      src == dst) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNode1D*>(clonedNode)->SetParams(dst, src, count, kind));
}

hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams) {
  HIP_INIT_API(hipGraphAddMemsetNode, pGraphNode, graph, pDependencies, numDependencies,
               pMemsetParams);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN_DURATION(
      ihipGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams));
}

hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies) {
  HIP_INIT_API(hipGraphAddEmptyNode, pGraphNode, graph, pDependencies, numDependencies);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraphNode = new hipGraphEmptyNode();
  hipError_t status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}

hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph) {
  HIP_INIT_API(hipGraphAddChildGraphNode, pGraphNode, pDependencies, numDependencies, childGraph);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || childGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraphNode = new hipChildGraphNode(childGraph);
  hipError_t status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}

hipError_t ihipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph) {
  if (pGraphExec == nullptr || graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  std::unordered_map<Node, Node> clonedNodes;
  hipGraph_t clonedGraph = graph->clone(clonedNodes);
  if (clonedGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  std::vector<std::vector<Node>> parallelLists;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists;
  std::unordered_set<hipUserObject*> graphExeUserObj;
  clonedGraph->GetRunList(parallelLists, nodeWaitLists);
  std::vector<Node> levelOrder;
  clonedGraph->LevelOrder(levelOrder);
  clonedGraph->GetUserObjs(graphExeUserObj);
  *pGraphExec = new hipGraphExec(levelOrder, parallelLists, nodeWaitLists, clonedNodes, graphExeUserObj);
  if (*pGraphExec != nullptr) {
    return (*pGraphExec)->Init();
  } else {
    return hipErrorOutOfMemory;
  }
}

hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  HIP_INIT_API(hipGraphInstantiate, pGraphExec, graph);
  HIP_RETURN_DURATION(ihipGraphInstantiate(pGraphExec, graph));
}

hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags) {
  HIP_INIT_API(hipGraphInstantiateWithFlags, pGraphExec, graph, flags);
  if (pGraphExec == nullptr || graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // invalid flag check
  if (flags != 0 && flags != hipGraphInstantiateFlagAutoFreeOnLaunch) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // enable when change is merged to hip
  // if (flags == hipGraphInstantiateFlagAutoFreeOnLaunch) {
  // Free any unfreed memory allocations before the graph is relaunched
  //}
  HIP_RETURN_DURATION(ihipGraphInstantiate(pGraphExec, graph));
}

hipError_t hipGraphExecDestroy(hipGraphExec_t pGraphExec) {
  HIP_INIT_API(hipGraphExecDestroy, pGraphExec);
  if (pGraphExec == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  delete pGraphExec;
  HIP_RETURN(hipSuccess);
}

hipError_t ihipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  return graphExec->Run(stream);
}

hipError_t hipGraphLaunch_common( hipGraphExec_t graphExec, hipStream_t stream) {
  if (graphExec == nullptr || !hip::isValid(stream) || !hipGraphExec::isGraphExecValid(graphExec)) {
    return hipErrorInvalidValue;
  }
  return ihipGraphLaunch(graphExec, stream);
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphLaunch, graphExec, stream);
  HIP_RETURN_DURATION(hipGraphLaunch_common(graphExec, stream));
}

hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphLaunch, graphExec, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipGraphLaunch_common(graphExec, stream));
}

hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
  HIP_INIT_API(hipGraphGetNodes, graph, nodes, numNodes);
  if (graph == nullptr || numNodes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<hipGraphNode_t>& graphNodes = graph->GetNodes();
  if (nodes == nullptr) {
    *numNodes = graphNodes.size();
    HIP_RETURN(hipSuccess);
  } else if (*numNodes <= graphNodes.size()) {
    for (int i = 0; i < *numNodes; i++) {
      nodes[i] = graphNodes[i];
    }
  } else {
    for (int i = 0; i < graphNodes.size(); i++) {
      nodes[i] = graphNodes[i];
    }
    for (int i = graphNodes.size(); i < *numNodes; i++) {
      nodes[i] = nullptr;
    }
    *numNodes = graphNodes.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes) {
  HIP_INIT_API(hipGraphGetRootNodes, graph, pRootNodes, pNumRootNodes);

  if (graph == nullptr || pNumRootNodes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<hipGraphNode_t> nodes = graph->GetRootNodes();
  if (pRootNodes == nullptr) {
    *pNumRootNodes = nodes.size();
    HIP_RETURN(hipSuccess);
  } else if (*pNumRootNodes <= nodes.size()) {
    for (int i = 0; i < *pNumRootNodes; i++) {
      pRootNodes[i] = nodes[i];
    }
  } else {
    for (int i = 0; i < nodes.size(); i++) {
      pRootNodes[i] = nodes[i];
    }
    for (int i = nodes.size(); i < *pNumRootNodes; i++) {
      pRootNodes[i] = nullptr;
    }
    *pNumRootNodes = nodes.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphKernelNodeGetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphKernelNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                       const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphKernelNodeSetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr || pNodeParams->func == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphKernelNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphMemcpyNodeGetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphMemcpyNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value) {
  HIP_INIT_API(hipGraphKernelNodeSetAttribute, hNode, attr, value);
  if (hNode == nullptr || value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (attr != hipKernelNodeAttributeAccessPolicyWindow && attr != hipKernelNodeAttributeCooperative ) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphKernelNode*>(hNode)->SetAttrParams(attr, value));
}

hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value) {
  HIP_INIT_API(hipGraphKernelNodeGetAttribute, hNode, attr, value);
  if (hNode == nullptr || value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (attr != hipKernelNodeAttributeAccessPolicyWindow && attr != hipKernelNodeAttributeCooperative ) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphKernelNode*>(hNode)->GetAttrParams(attr, value));
}

hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParams, hGraphExec, node, pNodeParams);
  if (hGraphExec == nullptr || node == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (ihipMemcpy3D_validate(pNodeParams) != hipSuccess) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Check if pNodeParams passed is a empty struct
  if (((pNodeParams->srcArray == 0) && (pNodeParams->srcPtr.ptr == nullptr)) ||
      ((pNodeParams->dstArray == 0) && (pNodeParams->dstPtr.ptr == nullptr))) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNode*>(clonedNode)->SetParams(pNodeParams));
}

hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams) {
  HIP_INIT_API(hipGraphMemsetNodeGetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphMemsetNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams) {
  HIP_INIT_API(hipGraphMemsetNodeSetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphMemsetNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams) {
  HIP_INIT_API(hipGraphExecMemsetNodeSetParams, hGraphExec, node, pNodeParams);
  if (hGraphExec == nullptr || node == nullptr || pNodeParams == nullptr ||
      pNodeParams->dst == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (ihipGraphMemsetParams_validate(pNodeParams) != hipSuccess) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphMemsetNode*>(clonedNode)->SetParams(pNodeParams));
}

hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies) {
  HIP_INIT_API(hipGraphAddDependencies, graph, from, to, numDependencies);
  if (graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (numDependencies == 0) {
    HIP_RETURN(hipSuccess);
  } else if (from == nullptr || to == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  for (size_t i = 0; i < numDependencies; i++) {
    // When the same node is specified for both from and to
    if (from[i] == nullptr || to[i] == nullptr || from[i] == to[i] ||
        !hipGraphNode::isNodeValid(to[i]) || !hipGraphNode::isNodeValid(from[i]) ||
        // making sure the nodes blong to the graph
        to[i]->GetParentGraph() != graph || from[i]->GetParentGraph() != graph) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  for (size_t i = 0; i < numDependencies; i++) {
    // When the same edge added from->to return invalid value
    const std::vector<Node>& edges = from[i]->GetEdges();
    for (auto edge : edges) {
      if (edge == to[i]) {
        HIP_RETURN(hipErrorInvalidValue);
      }
    }
    from[i]->AddEdge(to[i]);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphExecKernelNodeSetParams, hGraphExec, node, pNodeParams);
  if (hGraphExec == nullptr || node == nullptr || pNodeParams == nullptr ||
      pNodeParams->func == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphKernelNode*>(clonedNode)->SetParams(pNodeParams));
}

hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
  HIP_INIT_API(hipGraphChildGraphNodeGetGraph, node, pGraph);
  if (node == nullptr || pGraph == nullptr || !hipGraphNode::isNodeValid(node)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraph = reinterpret_cast<hipGraphNode*>(node)->GetChildGraph();
  if (*pGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph) {
  HIP_INIT_API(hipGraphExecChildGraphNodeSetParams, hGraphExec, node, childGraph);
  if (hGraphExec == nullptr || node == nullptr || childGraph == nullptr ||
      !ihipGraph::isGraphValid(childGraph)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (childGraph == node->GetParentGraph()) {
    HIP_RETURN(hipErrorUnknown);
  }

// Validate whether the topology of node and childGraph matches
  std::vector<Node> childGraphNodes1;
  node->LevelOrder(childGraphNodes1);

  std::vector<Node> childGraphNodes2;
  childGraph->LevelOrder(childGraphNodes2);

  if (childGraphNodes1.size() != childGraphNodes2.size()) {
    HIP_RETURN(hipErrorUnknown);
  }
  // Validate if the node insertion order matches
  else {
    for (std::vector<Node>::size_type i = 0; i != childGraphNodes1.size(); i++) {
      if (childGraphNodes1[i]->GetType() != childGraphNodes2[i]->GetType()) {
        HIP_RETURN(hipErrorUnknown);
      }
    }
  }

  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipChildGraphNode*>(clonedNode)->SetParams(childGraph));
}

hipError_t hipStreamGetCaptureInfo_common(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  if (pCaptureStatus == nullptr || !hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  if (stream == nullptr) {
    return hipErrorUnknown;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  *pCaptureStatus = s->GetCaptureStatus();
  if (*pCaptureStatus == hipStreamCaptureStatusActive && pId != nullptr) {
    *pId = s->GetCaptureID();
  }
  return hipSuccess;
}

hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  HIP_INIT_API(hipStreamGetCaptureInfo, stream, pCaptureStatus, pId);
  HIP_RETURN(hipStreamGetCaptureInfo_common(stream, pCaptureStatus, pId));
}

hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  HIP_INIT_API(hipStreamGetCaptureInfo, stream, pCaptureStatus, pId);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamGetCaptureInfo_common(stream, pCaptureStatus, pId));
}

hipError_t hipStreamGetCaptureInfo_v2_common(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  if (captureStatus_out == nullptr) {
    return hipErrorInvalidValue;
  }
  if (stream == nullptr) {
    return hipErrorUnknown;
  }
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  *captureStatus_out = s->GetCaptureStatus();
  if (*captureStatus_out == hipStreamCaptureStatusActive) {
    if (id_out != nullptr) {
      *id_out = s->GetCaptureID();
    }
    if (graph_out != nullptr) {
      *graph_out = s->GetCaptureGraph();
    }
    if (dependencies_out != nullptr) {
      *dependencies_out = s->GetLastCapturedNodes().data();
    }
    if (numDependencies_out != nullptr) {
      *numDependencies_out = s->GetLastCapturedNodes().size();
    }
  }
  return hipSuccess;
}

hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  HIP_INIT_API(hipStreamGetCaptureInfo_v2, stream, captureStatus_out, id_out, graph_out,
               dependencies_out, numDependencies_out);
  HIP_RETURN(hipStreamGetCaptureInfo_v2_common(stream, captureStatus_out, id_out, graph_out,
             dependencies_out, numDependencies_out));
}

hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  HIP_INIT_API(hipStreamGetCaptureInfo_v2, stream, captureStatus_out, id_out, graph_out,
               dependencies_out, numDependencies_out);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamGetCaptureInfo_v2_common(stream, captureStatus_out, id_out, graph_out,
             dependencies_out, numDependencies_out));
}

hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags) {
  HIP_INIT_API(hipStreamUpdateCaptureDependencies, stream, dependencies, numDependencies, flags);
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  if (s->GetCaptureStatus() == hipStreamCaptureStatusNone) {
    HIP_RETURN(hipErrorIllegalState);
  }
  if ((numDependencies > 0 && dependencies == nullptr) ||
      (flags != 0 && flags != hipStreamAddCaptureDependencies &&
       flags != hipStreamSetCaptureDependencies)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  std::vector<hipGraphNode_t> depNodes;
  for (int i = 0; i < numDependencies; i++) {
    depNodes.push_back(dependencies[i]);
  }
  if (flags == hipStreamAddCaptureDependencies) {
    s->AddCrossCapturedNode(depNodes);
  } else if (flags == hipStreamSetCaptureDependencies) {
    bool replace = true;
    s->AddCrossCapturedNode(depNodes, replace);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies) {
  HIP_INIT_API(hipGraphRemoveDependencies, graph, from, to, numDependencies);
  if (graph == nullptr || from == nullptr || to == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  for (size_t i = 0; i < numDependencies; i++) {
    if (from[i]->RemoveEdge(to[i]) == false) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges) {
  HIP_INIT_API(hipGraphGetEdges, graph, from, to, numEdges);
  if (graph == nullptr || numEdges == nullptr || (from == nullptr && to != nullptr) ||
      (to == nullptr && from != nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<std::pair<Node, Node>> edges = graph->GetEdges();
  // returns only the number of edges in numEdges when from and to are null
  if (from == nullptr && to == nullptr) {
    *numEdges = edges.size();
    HIP_RETURN(hipSuccess);
  } else if (*numEdges <= edges.size()) {
    for (int i = 0; i < *numEdges; i++) {
      from[i] = edges[i].first;
      to[i] = edges[i].second;
    }
  } else {
    for (int i = 0; i < edges.size(); i++) {
      from[i] = edges[i].first;
      to[i] = edges[i].second;
    }
    // If numEdges > actual number of edges, the remaining entries in from and to will be set to
    // NULL
    for (int i = edges.size(); i < *numEdges; i++) {
      from[i] = nullptr;
      to[i] = nullptr;
    }
    *numEdges = edges.size();
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies) {
  HIP_INIT_API(hipGraphNodeGetDependencies, node, pDependencies, pNumDependencies);
  if (node == nullptr || pNumDependencies == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<hipGraphNode_t>& dependencies = node->GetDependencies();
  if (pDependencies == NULL) {
    *pNumDependencies = dependencies.size();
    HIP_RETURN(hipSuccess);
  } else if (*pNumDependencies <= dependencies.size()) {
    for (int i = 0; i < *pNumDependencies; i++) {
      pDependencies[i] = dependencies[i];
    }
  } else {
    for (int i = 0; i < dependencies.size(); i++) {
      pDependencies[i] = dependencies[i];
    }
    // pNumDependencies > actual number of dependencies, the remaining entries in pDependencies will
    // be set to NULL
    for (int i = dependencies.size(); i < *pNumDependencies; i++) {
      pDependencies[i] = nullptr;
    }
    *pNumDependencies = dependencies.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes) {
  HIP_INIT_API(hipGraphNodeGetDependentNodes, node, pDependentNodes, pNumDependentNodes);
  if (node == nullptr || pNumDependentNodes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<hipGraphNode_t>& dependents = node->GetEdges();
  if (pDependentNodes == NULL) {
    *pNumDependentNodes = dependents.size();
    HIP_RETURN(hipSuccess);
  } else if (*pNumDependentNodes <= dependents.size()) {
    for (int i = 0; i < *pNumDependentNodes; i++) {
      pDependentNodes[i] = dependents[i];
    }
  } else {
    for (int i = 0; i < dependents.size(); i++) {
      pDependentNodes[i] = dependents[i];
    }
    // pNumDependentNodes > actual number of dependents, the remaining entries in pDependentNodes
    // will be set to NULL
    for (int i = dependents.size(); i < *pNumDependentNodes; i++) {
      pDependentNodes[i] = nullptr;
    }
    *pNumDependentNodes = dependents.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
  HIP_INIT_API(hipGraphNodeGetType, node, pType);
  if (node == nullptr || pType == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pType = node->GetType();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  HIP_INIT_API(hipGraphDestroyNode, node);
  if (node == nullptr || !hipGraphNode::isNodeValid(node)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  node->GetParentGraph()->RemoveNode(node);
  HIP_RETURN(hipSuccess);
}


hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  HIP_INIT_API(hipGraphClone, pGraphClone, originalGraph);
  if (originalGraph == nullptr || pGraphClone == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!ihipGraph::isGraphValid(originalGraph)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraphClone = originalGraph->clone();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph) {
  HIP_INIT_API(hipGraphNodeFindInClone, pNode, originalNode, clonedGraph);
  if (pNode == nullptr || originalNode == nullptr || clonedGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (clonedGraph->getOriginalGraph() == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (auto node : clonedGraph->GetNodes()) {
    if (node->GetID() == originalNode->GetID()) {
      *pNode = node;
      HIP_RETURN(hipSuccess);
    }
  }
  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNodeFromSymbol, pGraphNode, graph, pDependencies, numDependencies,
               dst, symbol, count, offset, kind);
  if (graph == nullptr || pGraphNode == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || dst == nullptr ||
      !ihipGraph::isGraphValid(graph)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  *pGraphNode = new hipGraphMemcpyNodeFromSymbol(dst, symbol, count, offset, kind);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}

hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParamsFromSymbol, node, dst, symbol, count, offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  if (node == nullptr || dst == nullptr || count == 0 || symbol == dst) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNodeFromSymbol*>(node)->SetParams(dst, symbol, count,
                                                                              offset, kind));
}

hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParamsFromSymbol, hGraphExec, node, dst, symbol, count,
               offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  if (hGraphExec == nullptr || node == nullptr || dst == nullptr || count == 0 || symbol == dst) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNodeFromSymbol*>(clonedNode)
                 ->SetParams(dst, symbol, count, offset, kind));
}

hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNodeToSymbol, pGraphNode, graph, pDependencies, numDependencies,
               symbol, src, count, offset, kind);
  if (pGraphNode == nullptr || graph == nullptr || src == nullptr ||
      !ihipGraph::isGraphValid(graph) || (pDependencies == nullptr && numDependencies > 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;
  hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  *pGraphNode = new hipGraphMemcpyNodeToSymbol(symbol, src, count, offset, kind);
  if (*pGraphNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}

hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParamsToSymbol, symbol, src, count, offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  if (node == nullptr || src == nullptr || count == 0 || symbol == src) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNodeToSymbol*>(node)->SetParams(symbol, src, count,
                                                                            offset, kind));
}


hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset,
                                                   hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParamsToSymbol, hGraphExec, node, symbol, src, count,
               offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  if (hGraphExec == nullptr || src == nullptr || node == nullptr || count == 0 || src == symbol) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNodeToSymbol*>(clonedNode)
                 ->SetParams(symbol, src, count, offset, kind));
}

hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event) {
  HIP_INIT_API(hipGraphAddEventRecordNode, pGraphNode, graph, pDependencies, numDependencies,
               event);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraphNode = new hipGraphEventRecordNode(event);
  hipError_t status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}

hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  HIP_INIT_API(hipGraphEventRecordNodeGetEvent, node, event_out);
  if (node == nullptr || event_out == nullptr || node->GetType() != hipGraphNodeTypeEventRecord) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphEventRecordNode*>(node)->GetParams(event_out);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  HIP_INIT_API(hipGraphEventRecordNodeSetEvent, node, event);
  if (node == nullptr || event == nullptr || node->GetType() != hipGraphNodeTypeEventRecord) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphEventRecordNode*>(node)->SetParams(event));
}

hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event) {
  HIP_INIT_API(hipGraphExecEventRecordNodeSetEvent, hGraphExec, hNode, event);
  if (hGraphExec == nullptr || hNode == nullptr || event == nullptr ||
      hNode->GetType() != hipGraphNodeTypeEventRecord) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(hNode);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphEventRecordNode*>(clonedNode)->SetParams(event));
}

hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event) {
  HIP_INIT_API(hipGraphAddEventWaitNode, pGraphNode, graph, pDependencies, numDependencies, event);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraphNode = new hipGraphEventWaitNode(event);
  hipError_t status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}

hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  HIP_INIT_API(hipGraphEventWaitNodeGetEvent, node, event_out);
  if (node == nullptr || event_out == nullptr || node->GetType() != hipGraphNodeTypeWaitEvent) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphEventWaitNode*>(node)->GetParams(event_out);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  HIP_INIT_API(hipGraphEventWaitNodeSetEvent, node, event);
  if (node == nullptr || event == nullptr || node->GetType() != hipGraphNodeTypeWaitEvent) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphEventWaitNode*>(node)->SetParams(event));
}

hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event) {
  HIP_INIT_API(hipGraphExecEventWaitNodeSetEvent, hGraphExec, hNode, event);
  if (hGraphExec == nullptr || hNode == nullptr || event == nullptr ||
      (hNode->GetType() != hipGraphNodeTypeWaitEvent)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(hNode);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphEventRecordNode*>(clonedNode)->SetParams(event));
}

hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphAddHostNode, pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  if (pGraphNode == nullptr || graph == nullptr || pNodeParams == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pNodeParams->fn == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pGraphNode = new hipGraphHostNode(pNodeParams);
  hipError_t status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}

hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphHostNodeGetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr || !hipGraphNode::isNodeValid(node)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphHostNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphHostNodeSetParams, node, pNodeParams);
  if (pNodeParams == nullptr || pNodeParams->fn == nullptr ||
      node == nullptr || !hipGraphNode::isNodeValid(node)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphHostNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphExecHostNodeSetParams, hGraphExec, node, pNodeParams);
  if (hGraphExec == nullptr || pNodeParams == nullptr || pNodeParams->fn == nullptr || node == nullptr || !hipGraphNode::isNodeValid(node)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hipGraphHostNode*>(clonedNode)->SetParams(pNodeParams));
}

hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out) {
  HIP_INIT_API(hipGraphExecUpdate, hGraphExec, hGraph, hErrorNode_out, updateResult_out);
  // parameter check
  if (hGraphExec == nullptr || hGraph == nullptr || hErrorNode_out == nullptr ||
      updateResult_out == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  std::vector<Node> newGraphNodes;
  hGraph->LevelOrder(newGraphNodes);
  std::vector<Node>& oldGraphExecNodes = hGraphExec->GetNodes();
  if (newGraphNodes.size() != oldGraphExecNodes.size()) {
    *updateResult_out = hipGraphExecUpdateErrorTopologyChanged;
    HIP_RETURN(hipErrorGraphExecUpdateFailure);
  }
  for (std::vector<Node>::size_type i = 0; i != newGraphNodes.size(); i++) {
    if (newGraphNodes[i]->GetType() == oldGraphExecNodes[i]->GetType()) {
      hipError_t status = oldGraphExecNodes[i]->SetParams(newGraphNodes[i]);
      if (status != hipSuccess) {
        *hErrorNode_out = newGraphNodes[i];
        if (status == hipErrorInvalidDeviceFunction) {
          *updateResult_out = hipGraphExecUpdateErrorUnsupportedFunctionChange;
        } else if (status == hipErrorInvalidValue || status == hipErrorInvalidDevicePointer) {
          *updateResult_out = hipGraphExecUpdateErrorParametersChanged;
        } else {
          *updateResult_out = hipGraphExecUpdateErrorNotSupported;
        }
        HIP_RETURN(hipErrorGraphExecUpdateFailure);
      }
    } else {
      *hErrorNode_out = newGraphNodes[i];
      *updateResult_out = hipGraphExecUpdateErrorNodeTypeChanged;
      HIP_RETURN(hipErrorGraphExecUpdateFailure);
    }
  }
  *updateResult_out = hipGraphExecUpdateSuccess;
  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  HIP_INIT_API(hipDeviceGetGraphMemAttribute, device, attr, value);
  if ((static_cast<size_t>(device) >= g_devices.size()) || device < 0 || value == nullptr) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  // later use this to access memory pool
  auto* deviceHandle = g_devices[device]->devices()[0];
  switch (attr) {
    case hipGraphMemAttrUsedMemCurrent:
      *reinterpret_cast<int32_t*>(value) = 0;
      break;
    case hipGraphMemAttrUsedMemHigh:
      *reinterpret_cast<int32_t*>(value) = 0;
      break;
    case hipGraphMemAttrReservedMemCurrent:
      *reinterpret_cast<int32_t*>(value) = 0;
      break;
    case hipGraphMemAttrReservedMemHigh:
      *reinterpret_cast<int32_t*>(value) = 0;
      break;
    default:
      return HIP_RETURN(hipErrorInvalidValue);
  }
  return HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  HIP_INIT_API(hipDeviceSetGraphMemAttribute, device, attr, value);
  if ((static_cast<size_t>(device) >= g_devices.size()) || device < 0 || value == nullptr) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  // later use this to access memory pool
  auto* deviceHandle = g_devices[device]->devices()[0];
  switch (attr) {
    case hipGraphMemAttrUsedMemHigh:
      break;
    case hipGraphMemAttrReservedMemHigh:
      break;
    default:
      return HIP_RETURN(hipErrorInvalidValue);
  }
  return HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGraphMemTrim(int device) {
  HIP_INIT_API(hipDeviceGraphMemTrim, device);
  if ((static_cast<size_t>(device) >= g_devices.size()) || device < 0) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  // not implemented yet
  return HIP_RETURN(hipSuccess);
}

hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) {
  HIP_INIT_API(hipUserObjectCreate, object_out, ptr, destroy, initialRefcount, flags);
  if (object_out == nullptr || flags != hipUserObjectNoDestructorSync || initialRefcount == 0 || destroy == nullptr || initialRefcount > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *object_out = new hipUserObject(destroy, ptr, flags);
  //! Creating object adds one reference.
  if (initialRefcount > 1) {
    (*object_out)->increaseRefCount(static_cast<const unsigned int>(initialRefcount - 1));
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
  HIP_INIT_API(hipUserObjectRelease, object, count);
  if (object == nullptr || count == 0 || count > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (object->referenceCount() < count || !hipUserObject::isUserObjvalid(object)) {
    HIP_RETURN(hipSuccess);
  }
  object->decreaseRefCount(count);
  HIP_RETURN(hipSuccess);
}

hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
  HIP_INIT_API(hipUserObjectRetain, object, count);
  if (object == nullptr || count == 0 || count > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!hipUserObject::isUserObjvalid(object)) {
    HIP_RETURN(hipSuccess);
  }
  object->increaseRefCount(count);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count, unsigned int flags) {
  HIP_INIT_API(hipGraphRetainUserObject, graph, object, count, flags);
  hipError_t status = hipSuccess;
  if (graph == nullptr || object == nullptr || count == 0 || count > INT_MAX ||
      (flags != 0 && flags != hipGraphUserObjectMove)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!hipUserObject::isUserObjvalid(object) && !graph->isUserObjGraphValid(object)) {
    HIP_RETURN(hipSuccess);
  }
  if (flags != hipGraphUserObjectMove) {
    status = hipUserObjectRetain(object, count);
    if (status != hipSuccess) {
      HIP_RETURN(status);
    }
  } else {
    //! if flag is UserObjMove delete userobj from list
    hipUserObject::removeUSerObj(object);
  }
  graph->addUserObjGraph(object);
  HIP_RETURN(status);
}

hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
  HIP_INIT_API(hipGraphReleaseUserObject, graph, object, count);
  if (graph == nullptr || object == nullptr || count == 0 || count > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!graph->isUserObjGraphValid(object) || object->referenceCount() < count) {
    HIP_RETURN(hipSuccess);
  }
  //! Obj is being destroyed
  unsigned int releaseCount = (object->referenceCount() < count) ? object->referenceCount() : count;
  if (object->referenceCount() == releaseCount) {
    graph->RemoveUserObjGraph(object);
  }
  hipError_t status = hipUserObjectRelease(object, count);
  HIP_RETURN(status);
}

hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst) {
  HIP_INIT_API(hipGraphKernelNodeCopyAttributes, hSrc, hDst);
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t ihipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags) {
  if (graph == nullptr || path == nullptr) {
    return hipErrorInvalidValue;
  }
  std::ofstream fout;
  fout.open(path, std::ios::out);
  if (fout.fail()) {
    ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] Error during opening of file : %s", path);
    return hipErrorOperatingSystem;
  }
  fout << "digraph dot {" << std::endl;
  graph->GenerateDOT(fout, (hipGraphDebugDotFlags)flags);
  fout << "}" << std::endl;
  fout.close();
  return hipSuccess;
}

hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags) {
  HIP_INIT_API(hipGraphDebugDotPrint, graph, path, flags);
  HIP_RETURN(ihipGraphDebugDotPrint(graph, path, flags));
}

hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled) {
  HIP_INIT_API(hipGraphNodeSetEnabled, hGraphExec, hNode, isEnabled);
   if (hGraphExec == nullptr || hNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(hNode);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  clonedNode->SetEnabled(isEnabled);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled) {
  HIP_INIT_API(hipGraphNodeGetEnabled, hGraphExec, hNode, isEnabled);
  if (hGraphExec == nullptr || hNode == nullptr || isEnabled == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNode_t clonedNode = hGraphExec->GetClonedNode(hNode);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *isEnabled = clonedNode->GetEnabled();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphUpload, graphExec, stream);
  HIP_RETURN(hipErrorNotSupported);
}
