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

thread_local std::vector<hipStream_t> g_captureStreams;
std::unordered_map<amd::Command*, hipGraphExec_t> hipGraphExec::activeGraphExec_;

inline void ihipGraphAddNode(hipGraphNode_t graphNode, hipGraph_t graph,
                            const hipGraphNode_t* pDependencies, size_t numDependencies) {
  graph->AddNode(graphNode);
  for (size_t i = 0; i < numDependencies; i++) {
    pDependencies[i]->AddEdge(graphNode);
  }
}

hipError_t ihipValidateKernelParams(const hipKernelNodeParams* pNodeParams) {
  hipFunction_t func = nullptr;
  hipError_t status =
      PlatformState::instance().getStatFunc(&func, pNodeParams->func, ihipGetDevice());
  if ((status != hipSuccess) || (func == nullptr)) {
    return hipErrorInvalidDeviceFunction;
  }
  size_t globalWorkSizeX = static_cast<size_t>(pNodeParams->gridDim.x) * pNodeParams->blockDim.x;
  size_t globalWorkSizeY = static_cast<size_t>(pNodeParams->gridDim.y) * pNodeParams->blockDim.y;
  size_t globalWorkSizeZ = static_cast<size_t>(pNodeParams->gridDim.z) * pNodeParams->blockDim.z;
  if (globalWorkSizeX > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeY > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeZ > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidConfiguration;
  }
  status = ihipLaunchKernel_validate(
      func, static_cast<uint32_t>(globalWorkSizeX), static_cast<uint32_t>(globalWorkSizeY),
      static_cast<uint32_t>(globalWorkSizeZ), pNodeParams->blockDim.x, pNodeParams->blockDim.y,
      pNodeParams->blockDim.z, pNodeParams->sharedMemBytes, pNodeParams->kernelParams,
      pNodeParams->extra, ihipGetDevice(), 0);
  if (status != hipSuccess) {
    return status;
  }
  return hipSuccess;
}

hipError_t ihipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipKernelNodeParams* pNodeParams) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pNodeParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipError_t status = ihipValidateKernelParams(pNodeParams);
  if (hipSuccess != status) {
    return status;
  }
  hipFunction_t func = nullptr;
  status = PlatformState::instance().getStatFunc(&func, pNodeParams->func, ihipGetDevice());
  if ((status != hipSuccess) || (func == nullptr)) {
    return hipErrorInvalidDeviceFunction;
  }
  *pGraphNode = new hipGraphKernelNode(pNodeParams, func);
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return hipSuccess;
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
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return hipSuccess;
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
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return hipSuccess;
}

hipError_t ihipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemsetParams* pMemsetParams) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pMemsetParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipError_t status;
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
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  return hipSuccess;
}

hipError_t capturehipLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                  dim3& blockDim, void**& args, size_t& sharedMemBytes) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node kernel launch on stream : %p", stream);
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

hipError_t capturehipMemcpy3DAsync(hipStream_t& stream, const hipMemcpy3DParms*& p) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memcpy3D on stream : %p",
          stream);
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

hipError_t capturehipMemcpyAsync(hipStream_t& stream, void*& dst, const void*& src,
                                 size_t& sizeBytes, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memcpy1D on stream : %p",
          stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraph_t graph = nullptr;
  std::vector<hipGraphNode_t> pDependencies = s->GetLastCapturedNodes();
  size_t numDependencies = s->GetLastCapturedNodes().size();
  graph = s->GetCaptureGraph();
  hipError_t status = ihipMemcpy_validate(dst, src, sizeBytes, kind);
  if (status != hipSuccess) {
    return status;
  }
  hipGraphNode_t node = new hipGraphMemcpyNode1D(dst, src, sizeBytes, kind);
  ihipGraphAddNode(node, graph, pDependencies.data(), numDependencies);
  s->SetLastCapturedNode(node);
  return hipSuccess;
}

hipError_t capturehipMemcpyFromSymbolAsync(hipStream_t& stream, void*& dst, const void*& symbol,
                                           size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyFromSymbolNode on stream : %p", stream);
  return hipSuccess;
}

hipError_t capturehipMemcpyToSymbolAsync(hipStream_t& stream, const void*& symbol, const void*& src,
                                         size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyToSymbolNode on stream : %p", stream);
  return hipSuccess;
}

hipError_t capturehipMemsetAsync(hipStream_t& stream, void*& dst, int& value, size_t& valueSize,
                                 size_t& sizeBytes) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset1D on stream : %p",
          stream);

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
  return hipSuccess;
}

hipError_t capturehipEventRecord(hipStream_t& stream, hipEvent_t& event) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node EventRecord on stream : %p, Event %p", stream, event);
  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  e->StartCapture(stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
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

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  if (event == nullptr || stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!s->IsOriginStream()) {
    s->SetCaptureGraph(reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->GetCaptureGraph());
    s->SetCaptureMode(reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->GetCaptureMode());
    s->SetParentStream(e->GetCaptureStream());
  }
  s->AddCrossCapturedNode(e->GetNodesPrevToRecorded());
  g_captureStreams.push_back(stream);
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  HIP_INIT_API(hipStreamIsCapturing, stream, pCaptureStatus);
  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pCaptureStatus = reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus();
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  HIP_INIT_API(hipStreamBeginCapture, stream, mode);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // capture cannot be initiated on legacy stream
  // It can be initiated if the stream is not already in capture mode
  if (stream == nullptr || s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  s->SetCaptureGraph(new ihipGraph());
  s->SetCaptureMode(mode);
  s->SetOriginStream();
  g_captureStreams.push_back(stream);
  HIP_RETURN_DURATION(hipSuccess);
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  HIP_INIT_API(hipStreamEndCapture, stream, pGraph);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // Capture must be ended on the same stream in which it was initiated
  if (!s->IsOriginStream()) {
    HIP_RETURN(hipErrorStreamCaptureUnmatched);
  }
  // If mode is not hipStreamCaptureModeRelaxed, hipStreamEndCapture must be called on the stream
  // from the same thread
  if (s->GetCaptureMode() != hipStreamCaptureModeRelaxed &&
      std::find(g_captureStreams.begin(), g_captureStreams.end(), stream) ==
          g_captureStreams.end()) {
    HIP_RETURN(hipErrorStreamCaptureWrongThread);
  }
  // If capture was invalidated, due to a violation of the rules of stream capture
  if (s->GetCaptureStatus() == hipStreamCaptureStatusInvalidated) {
    *pGraph = nullptr;
    HIP_RETURN(hipErrorStreamCaptureInvalidated);
  }
  // check if all parallel streams have joined
  // Nodes that are removed from the dependency set via API hipStreamUpdateCaptureDependencies do
  // not result in hipErrorStreamCaptureUnjoined
  if (s->GetCaptureGraph()->GetLeafNodeCount() != 1) {
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
    if (foundInRemovedDep == false) {
      return hipErrorStreamCaptureUnjoined;
    }
  }
  *pGraph = s->GetCaptureGraph();
  // end capture on all streams/events part of graph capture
  HIP_RETURN_DURATION(s->EndCapture());
}

hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  HIP_INIT_API(hipGraphCreate, pGraph, flags);
  *pGraph = new ihipGraph();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphDestroy(hipGraph_t graph) {
  HIP_INIT_API(hipGraphDestroy, graph);
  if (graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  delete graph;
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphAddKernelNode, pGraphNode, graph, pDependencies, numDependencies,
               pNodeParams);
  HIP_RETURN_DURATION(
      ihipGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams));
}

hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams) {
  HIP_INIT_API(hipGraphAddMemcpyNode, pGraphNode, graph, pDependencies, numDependencies,
               pCopyParams);

  HIP_RETURN_DURATION(
      ihipGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams));
}

hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNode1D, pGraphNode, graph, pDependencies, numDependencies, dst, src,
               count, kind);

  HIP_RETURN_DURATION(ihipGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies,
                                               dst, src, count, kind));
}

hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParams1D, node, dst, src, count, kind);
  reinterpret_cast<hipGraphMemcpyNode1D*>(node)->SetParams(dst, src, count, kind);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParams1D, hGraphExec, node, dst, src, count, kind);
  HIP_RETURN(
      reinterpret_cast<hipGraphMemcpyNode1D*>(node)->SetCommandParams(dst, src, count, kind));
}

hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams) {
  HIP_INIT_API(hipGraphAddMemsetNode, pGraphNode, graph, pDependencies, numDependencies,
               pMemsetParams);

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
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(hipSuccess);
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
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(hipSuccess);
}

hipError_t ihipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  std::vector<std::vector<Node>> parallelLists;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists;
  graph->GetRunList(parallelLists, nodeWaitLists);
  std::vector<Node> levelOrder;
  graph->LevelOrder(levelOrder);
  *pGraphExec = new hipGraphExec(levelOrder, parallelLists, nodeWaitLists);
  if (*pGraphExec != nullptr) {
    return (*pGraphExec)->Init();
  } else {
    return hipErrorOutOfMemory;
  }
}

hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  HIP_INIT_API(hipGraphInstantiate, pGraphExec, graph);
  HIP_RETURN_DURATION(ihipGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize));
}

hipError_t hipGraphExecDestroy(hipGraphExec_t pGraphExec) {
  HIP_INIT_API(hipGraphExecDestroy, pGraphExec);
  if (pGraphExec == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  delete pGraphExec;
  HIP_RETURN(hipSuccess);
}

hipError_t ihipGraphlaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  return graphExec->Run(stream);
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphLaunch, graphExec, stream);
  if (graphExec == nullptr || stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN_DURATION(ihipGraphlaunch(graphExec, stream));
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
  } else if (*numNodes < graphNodes.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  for (int i = 0; i < graphNodes.size(); i++) {
    nodes[i] = graphNodes[i];
  }
  for (int i = graphNodes.size(); i < *numNodes; i++) {
    nodes[i] = nullptr;
  }
  *numNodes = graphNodes.size();
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
  } else if (*pNumRootNodes < nodes.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  for (int i = 0; i < nodes.size(); i++) {
    pRootNodes[i] = nodes[i];
  }
  for (int i = nodes.size(); i < *pNumRootNodes; i++) {
    pRootNodes[i] = nullptr;
  }
  *pNumRootNodes = nodes.size();
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
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphKernelNode*>(node)->SetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphMemcpyNodeGetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphMemcpyNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParams, node, pNodeParams);
  if (node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hipGraphMemcpyNode*>(node)->SetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParams, hGraphExec, node, pNodeParams);
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNode*>(node)->SetCommandParams(pNodeParams));
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
  reinterpret_cast<hipGraphMemsetNode*>(node)->SetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
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
    from[i]->AddEdge(to[i]);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphExecKernelNodeSetParams, hGraphExec, node, pNodeParams);
  if (hGraphExec == nullptr || node == nullptr || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  return reinterpret_cast<hipGraphKernelNode*>(node)->SetCommandParams(pNodeParams);
}

hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
  HIP_INIT_API(hipGraphChildGraphNodeGetGraph, node, pGraph);
  if (node == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraph = reinterpret_cast<hipChildGraphNode*>(node)->GetChildGraph();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph) {
  HIP_INIT_API(hipGraphExecChildGraphNodeSetParams, hGraphExec, node, childGraph);
  if (hGraphExec == nullptr || node == nullptr || childGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  HIP_INIT_API(hipStreamGetCaptureInfo, stream, pCaptureStatus, pId);
  if (pCaptureStatus == nullptr || pId == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (stream == nullptr) {
    HIP_RETURN(hipErrorStreamCaptureImplicit);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  *pCaptureStatus = s->GetCaptureStatus();
  *pId = s->GetCaptureID();
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  HIP_INIT_API(hipStreamGetCaptureInfo_v2, stream, captureStatus_out, id_out, graph_out,
               dependencies_out, numDependencies_out);
  if (stream == nullptr) {
    HIP_RETURN(hipErrorStreamCaptureImplicit);
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
    if (dependencies_out != nullptr && numDependencies_out != nullptr) {
      *dependencies_out = s->GetLastCapturedNodes().data();
      *numDependencies_out = s->GetLastCapturedNodes().size();
    }
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags) {
  HIP_INIT_API(hipStreamUpdateCaptureDependencies, stream, dependencies, numDependencies, flags);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  if (s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
    return hipErrorIllegalState;
  }
  if (numDependencies > 0 && dependencies == nullptr) {
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
    from[i]->RemoveEdge(to[i]);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges) {
  HIP_INIT_API(hipGraphGetEdges, graph, from, to, numEdges);
  if (graph == nullptr || numEdges == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<std::pair<Node, Node>> edges = graph->GetEdges();
  // returns only the number of edges in numEdges when from and to are null
  if (from == nullptr && to == nullptr) {
    *numEdges = edges.size();
    return hipSuccess;
  } else if (*numEdges < edges.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  for (int i = 0; i < edges.size(); i++) {
    from[i] = edges[i].first;
    to[i] = edges[i].second;
  }
  // If numEdges > actual number of edges, the remaining entries in from and to will be set to NULL
  for (int i = edges.size(); i < *numEdges; i++) {
    from[i] = nullptr;
    to[i] = nullptr;
  }
  *numEdges = edges.size();
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
  } else if (*pNumDependencies < dependencies.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (int i = 0; i < dependencies.size(); i++) {
    pDependencies[i] = dependencies[i];
  }
  // pNumDependencies > actual number of dependencies, the remaining entries in pDependencies will
  // be set to NULL
  for (int i = dependencies.size(); i < *pNumDependencies; i++) {
    pDependencies[i] = nullptr;
  }
  *pNumDependencies = dependencies.size();
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
  } else if (*pNumDependentNodes < dependents.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (int i = 0; i < dependents.size(); i++) {
    pDependentNodes[i] = dependents[i];
  }
  // pNumDependentNodes > actual number of dependents, the remaining entries in pDependentNodes will
  // be set to NULL
  for (int i = dependents.size(); i < *pNumDependentNodes; i++) {
    pDependentNodes[i] = nullptr;
  }
  *pNumDependentNodes = dependents.size();
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
  if (node == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  node->GetParentGraph()->RemoveNode(node);
  // Takescare of removing its dependencies and dependent nodes
  delete node;
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  HIP_INIT_API(hipGraphClone, pGraphClone, originalGraph);
  if (originalGraph == nullptr || pGraphClone == nullptr) {
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
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hipGraphMemcpyNodeFromSymbol(dst, symbol, count, offset, kind);
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParamsFromSymbol, node, dst, symbol, count, offset, kind);
  reinterpret_cast<hipGraphMemcpyNodeFromSymbol*>(node)->SetParams(dst, symbol, count, offset,
                                                                   kind);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParamsFromSymbol, hGraphExec, node, dst, symbol, count,
               offset, kind);
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNodeFromSymbol*>(node)->SetCommandParams(
      dst, symbol, count, offset, kind));
}

hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNodeToSymbol, pGraphNode, graph, pDependencies, numDependencies,
               symbol, src, count, offset, kind);
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;
  hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hipGraphMemcpyNodeToSymbol(symbol, src, count, offset, kind);
  ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParamsToSymbol, symbol, src, count, offset, kind);
  reinterpret_cast<hipGraphMemcpyNodeToSymbol*>(node)->SetParams(symbol, src, count, offset, kind);
  HIP_RETURN(hipSuccess);
}


hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset,
                                                   hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParamsToSymbol, hGraphExec, node, symbol, src, count,
               offset, kind);
  HIP_RETURN(reinterpret_cast<hipGraphMemcpyNodeToSymbol*>(node)->SetCommandParams(
      symbol, src, count, offset, kind));
}
