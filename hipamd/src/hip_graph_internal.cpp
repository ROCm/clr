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
#include <queue>

#define CASE_STRING(X, C)                                                                          \
  case X:                                                                                          \
    case_string = #C;                                                                              \
    break;
namespace {
const char* GetGraphNodeTypeString(uint32_t op) {
  const char* case_string;
  switch (static_cast<hipGraphNodeType>(op)) {
    CASE_STRING(hipGraphNodeTypeKernel, KernelNode)
    CASE_STRING(hipGraphNodeTypeMemcpy, MemcpyNode)
    CASE_STRING(hipGraphNodeTypeMemset, MemsetNode)
    CASE_STRING(hipGraphNodeTypeHost, HostNode)
    CASE_STRING(hipGraphNodeTypeGraph, GraphNode)
    CASE_STRING(hipGraphNodeTypeEmpty, EmptyNode)
    CASE_STRING(hipGraphNodeTypeWaitEvent, WaitEventNode)
    CASE_STRING(hipGraphNodeTypeEventRecord, EventRecordNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreSignal, ExtSemaphoreSignalNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreWait, ExtSemaphoreWaitNode)
    CASE_STRING(hipGraphNodeTypeMemAlloc, MemAllocNode)
    CASE_STRING(hipGraphNodeTypeMemFree, MemFreeNode)
    CASE_STRING(hipGraphNodeTypeMemcpyFromSymbol, MemcpyFromSymbolNode)
    CASE_STRING(hipGraphNodeTypeMemcpyToSymbol, MemcpyToSymbolNode)
    default:
      case_string = "Unknown node type";
  };
  return case_string;
};
}

namespace hip {
std::unordered_map<GraphExec *, std::pair<hip::Stream *, bool>>
    GraphExecStatus_ ROCCLR_INIT_PRIORITY(101);
amd::Monitor GraphExecStatusLock_ ROCCLR_INIT_PRIORITY(101){
    "Guards graph execution state"};

int GraphNode::nextID = 0;
int Graph::nextID = 0;
std::unordered_set<GraphNode*> GraphNode::nodeSet_;
amd::Monitor GraphNode::nodeSetLock_{"Guards global node set"};
std::unordered_set<Graph*> Graph::graphSet_;
amd::Monitor Graph::graphSetLock_{"Guards global graph set"};
std::unordered_set<GraphExec*> GraphExec::graphExecSet_;
amd::Monitor GraphExec::graphExecSetLock_{"Guards global exec graph set"};
std::unordered_set<UserObject*> UserObject::ObjectSet_;
amd::Monitor UserObject::UserObjectLock_{"Guards global user object"};

hipError_t GraphMemcpyNode1D::ValidateParams(void* dst, const void* src, size_t count,
                                                hipMemcpyKind kind) {
  hipError_t status = ihipMemcpy_validate(dst, src, count, kind);
  if (status != hipSuccess) {
    return status;
  }
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);

  if ((srcMemory == nullptr) && (dstMemory != nullptr)) {  // host to device
    if ((kind != hipMemcpyHostToDevice) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {  // device to host
    if ((kind != hipMemcpyDeviceToHost) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  }

  return hipSuccess;
}

hipError_t GraphMemcpyNode::ValidateParams(const hipMemcpy3DParms* pNodeParams) {
  hipError_t status;
  status = ihipMemcpy3D_validate(pNodeParams);
  if (status != hipSuccess) {
    return status;
  }

  const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(*pNodeParams);
  status = ihipDrvMemcpy3D_validate(&pCopy);
  if (status != hipSuccess) {
    return status;
  }
  return hipSuccess;
}

bool Graph::isGraphValid(Graph* pGraph) {
  amd::ScopedLock lock(graphSetLock_);
  if (graphSet_.find(pGraph) == graphSet_.end()) {
    return false;
  }
  return true;
}

void Graph::AddNode(const Node& node) {
  vertices_.emplace_back(node);
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Add %s(%p)",
          GetGraphNodeTypeString(node->GetType()), node);
  node->SetParentGraph(this);
}

void Graph::RemoveNode(const Node& node) {
  vertices_.erase(std::remove(vertices_.begin(), vertices_.end(), node), vertices_.end());
  delete node;
}

// root nodes are all vertices with 0 in-degrees
std::vector<Node> Graph::GetRootNodes() const {
  std::vector<Node> roots;
  for (auto entry : vertices_) {
    if (entry->GetInDegree() == 0) {
      roots.push_back(entry);
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Root node: %s(%p)",
              GetGraphNodeTypeString(entry->GetType()), entry);
    }
  }
  return roots;
}

// leaf nodes are all vertices with 0 out-degrees
std::vector<Node> Graph::GetLeafNodes() const {
  std::vector<Node> leafNodes;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      leafNodes.push_back(entry);
    }
  }
  return leafNodes;
}

size_t Graph::GetLeafNodeCount() const {
  int numLeafNodes = 0;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      numLeafNodes++;
    }
  }
  return numLeafNodes;
}

std::vector<std::pair<Node, Node>> Graph::GetEdges() const {
  std::vector<std::pair<Node, Node>> edges;
  for (const auto& i : vertices_) {
    for (const auto& j : i->GetEdges()) {
      edges.push_back(std::make_pair(i, j));
    }
  }
  return edges;
}

void Graph::GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                               std::vector<Node>& singleList,
                               std::vector<std::vector<Node>>& parallelLists,
                               std::unordered_map<Node, std::vector<Node>>& dependencies) {
  // Mark the current node as visited.
  visited[v] = true;
  singleList.push_back(v);
  // Recurse for all the vertices adjacent to this vertex
  for (auto& adjNode : v->GetEdges()) {
    if (!visited[adjNode]) {
      // For the parallel list nodes add parent as the dependency
      if (singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] For %s(%p) - add parent as dependency %s(%p)",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
      GetRunListUtil(adjNode, visited, singleList, parallelLists, dependencies);
    } else {
      for (auto& list : parallelLists) {
        // Merge singleList when adjNode matches with the first element of the list in existing
        // lists
        if (adjNode == list[0]) {
          for (auto k = singleList.rbegin(); k != singleList.rend(); ++k) {
            list.insert(list.begin(), *k);
          }
          singleList.erase(singleList.begin(), singleList.end());
        }
      }
      // If the list cannot be merged with the existing list add as dependancy
      if (!singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] For %s(%p) - add dependency %s(%p)",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
    }
  }
  if (!singleList.empty()) {
    parallelLists.push_back(singleList);
    singleList.erase(singleList.begin(), singleList.end());
  }
}
// The function to do Topological Sort.
// It uses recursive GetRunListUtil()
void Graph::GetRunList(std::vector<std::vector<Node>>& parallelLists,
                           std::unordered_map<Node, std::vector<Node>>& dependencies) {
  std::vector<Node> singleList;

  // Mark all the vertices as not visited
  std::unordered_map<Node, bool> visited;
  for (auto node : vertices_) visited[node] = false;

  // Call the recursive helper function for all vertices one by one
  for (auto node : vertices_) {
    // If the node has embedded child graph
    node->GetRunList(parallelLists, dependencies);
    if (visited[node] == false) {
      GetRunListUtil(node, visited, singleList, parallelLists, dependencies);
    }
  }
  for (size_t i = 0; i < parallelLists.size(); i++) {
    for (size_t j = 0; j < parallelLists[i].size(); j++) {
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] List %d - %s(%p)", i + 1,
              GetGraphNodeTypeString(parallelLists[i][j]->GetType()), parallelLists[i][j]);
    }
  }
}
bool Graph::TopologicalOrder(std::vector<Node>& TopoOrder) {
  std::queue<Node> q;
  std::unordered_map<Node, int> inDegree;
  for (auto entry : vertices_) {
    if (entry->GetInDegree() == 0) {
      q.push(entry);
    }
    inDegree[entry] = entry->GetInDegree();
  }
  while (!q.empty())
  {
    Node node = q.front();
    TopoOrder.push_back(node);
    q.pop();
    for (auto edge : node->GetEdges()) {
      inDegree[edge]--;
      if (inDegree[edge] == 0) {
        q.push(edge);
      }
    }
  }
  if (GetNodeCount() == TopoOrder.size()) {
    return true;
  }
  return false;
}
Graph* Graph::clone(std::unordered_map<Node, Node>& clonedNodes) const {
  Graph* newGraph = new Graph(device_, this);
  for (auto entry : vertices_) {
    GraphNode* node = entry->clone();
    node->SetParentGraph(newGraph);
    newGraph->vertices_.push_back(node);
    clonedNodes[entry] = node;
  }

  std::vector<Node> clonedEdges;
  std::vector<Node> clonedDependencies;
  for (auto node : vertices_) {
    const std::vector<Node>& edges = node->GetEdges();
    clonedEdges.clear();
    for (auto edge : edges) {
      clonedEdges.push_back(clonedNodes[edge]);
    }
    clonedNodes[node]->SetEdges(clonedEdges);
  }
  for (auto node : vertices_) {
    const std::vector<Node>& dependencies = node->GetDependencies();
    clonedDependencies.clear();
    for (auto dep : dependencies) {
      clonedDependencies.push_back(clonedNodes[dep]);
    }
    clonedNodes[node]->SetDependencies(clonedDependencies);
  }
  for (auto userObj : graphUserObj_) {
    userObj->retain();
    newGraph->graphUserObj_.insert(userObj);
  }
  return newGraph;
}

Graph* Graph::clone() const {
  std::unordered_map<Node, Node> clonedNodes;
  return clone(clonedNodes);
}

bool GraphExec::isGraphExecValid(GraphExec* pGraphExec) {
  amd::ScopedLock lock(graphExecSetLock_);
  if (graphExecSet_.find(pGraphExec) == graphExecSet_.end()) {
    return false;
  }
  return true;
}

hipError_t GraphExec::CreateStreams(uint32_t num_streams) {
  parallel_streams_.reserve(num_streams);
  for (uint32_t i = 0; i < num_streams; ++i) {
    auto stream = new hip::Stream(hip::getCurrentDevice(),
                                  hip::Stream::Priority::Normal, hipStreamNonBlocking);
    if (stream == nullptr || !stream->Create()) {
      if (stream != nullptr) {
        hip::Stream::Destroy(stream);
      }
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create parallel stream!");
      return hipErrorOutOfMemory;
    }
    parallel_streams_.push_back(stream);
  }
  // Don't wait for other streams to finish.
  // Capture stream is to capture AQL packet.
  capture_stream_ = hip::getNullStream(false);
  return hipSuccess;
}

hipError_t GraphExec::Init() {
  hipError_t status = hipSuccess;
  size_t min_num_streams = 1;

  for (auto& node : topoOrder_) {
    status = node->GetNumParallelStreams(min_num_streams);
    if (status != hipSuccess) {
      return status;
    }
  }
  status = CreateStreams(parallelLists_.size() - 1 + min_num_streams);
  if (status != hipSuccess) {
    return status;
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    // For graph nodes capture AQL packets to dispatch them directly during graph launch.
    status = CaptureAQLPackets();
  }
  instantiateDeviceId_ = hip::getCurrentDevice()->deviceId();
  return status;
}

void GetKernelArgSizeForGraph(std::vector<std::vector<Node>>& parallelLists,
                              size_t& kernArgSizeForGraph) {
  // GPU packet capture is enabled for kernel nodes. Calculate the kernel
  // arg size required for all graph kernel nodes to allocate
  for (const auto& list : parallelLists) {
    for (auto& node : list) {
      if (node->GetType() == hipGraphNodeTypeKernel &&
          !reinterpret_cast<hip::GraphKernelNode*>(node)->HasHiddenHeap()) {
        kernArgSizeForGraph += reinterpret_cast<hip::GraphKernelNode*>(node)->GetKerArgSize();
      } else if (node->GetType() == hipGraphNodeTypeGraph) {
        auto& childParallelLists = reinterpret_cast<hip::ChildGraphNode*>(node)->GetParallelLists();
        if (childParallelLists.size() == 1) {
          GetKernelArgSizeForGraph(childParallelLists, kernArgSizeForGraph);
        }
      }
    }
  }
}

hipError_t AllocKernelArgForGraph(std::vector<hip::Node>& topoOrder, hip::Stream* capture_stream,
                                  hip::GraphExec* graphExec) {
  hipError_t status = hipSuccess;
  for (auto& node : topoOrder) {
    if (node->GetType() == hipGraphNodeTypeKernel &&
        !reinterpret_cast<hip::GraphKernelNode*>(node)->HasHiddenHeap()) {
      auto kernelNode = reinterpret_cast<hip::GraphKernelNode*>(node);
      // From the kernel pool allocate the kern arg size required for the current kernel node.
      address kernArgOffset = nullptr;
      if (kernelNode->GetKernargSegmentByteSize()) {
        kernArgOffset = graphExec->allocKernArg(kernelNode->GetKernargSegmentByteSize(),
                                                kernelNode->GetKernargSegmentAlignment());
        if (kernArgOffset == nullptr) {
          return hipErrorMemoryAllocation;
        }
      }
      // Form GPU packet capture for the kernel node.
      kernelNode->CaptureAndFormPacket(capture_stream, kernArgOffset);
    } else if (node->GetType() == hipGraphNodeTypeGraph) {
      auto childNode = reinterpret_cast<hip::ChildGraphNode*>(node);
      auto& childParallelLists = childNode->GetParallelLists();
      if (childParallelLists.size() == 1) {
        childNode->SetGraphCaptureStatus(true);
        status =
            AllocKernelArgForGraph(childNode->GetChildGraphNodeOrder(), capture_stream, graphExec);
        if (status != hipSuccess) {
          return status;
        }
      }
    }
  }
  return status;
}

hipError_t GraphExec::CaptureAQLPackets() {
  hipError_t status = hipSuccess;
  if (parallelLists_.size() == 1) {
    size_t kernArgSizeForGraph = 0;
    GetKernelArgSizeForGraph(parallelLists_, kernArgSizeForGraph);
    auto device = g_devices[ihipGetDevice()]->devices()[0];
    if (kernArgSizeForGraph != 0) {
      if (device->info().largeBar_) {
        kernarg_pool_graph_ =
            reinterpret_cast<address>(device->deviceLocalAlloc(kernArgSizeForGraph));
        device_kernarg_pool_ = true;
      } else {
        kernarg_pool_graph_ = reinterpret_cast<address>(
            device->hostAlloc(kernArgSizeForGraph, 0, amd::Device::MemorySegment::kKernArg));
      }

      if (kernarg_pool_graph_ == nullptr) {
        return hipErrorMemoryAllocation;
      }
      kernarg_pool_size_graph_ = kernArgSizeForGraph;
    }
    status = AllocKernelArgForGraph(topoOrder_, capture_stream_, this);
    if (status != hipSuccess) {
      return status;
    }

    if (device_kernarg_pool_) {
      auto kernArgImpl = device->settings().kernel_arg_impl_;

      if (kernArgImpl == KernelArgImpl::DeviceKernelArgsHDP) {
        *device->info().hdpMemFlushCntl = 1u;
        auto kSentinel = *reinterpret_cast<volatile int*>(device->info().hdpMemFlushCntl);
      } else if (kernArgImpl == KernelArgImpl::DeviceKernelArgsReadback &&
                 kernarg_pool_size_graph_ != 0) {
        address dev_ptr = kernarg_pool_graph_ + kernarg_pool_size_graph_;
        auto kSentinel = *reinterpret_cast<volatile unsigned char*>(dev_ptr - 1);
        _mm_sfence();
        *(dev_ptr - 1) = kSentinel;
        _mm_mfence();
        kSentinel = *reinterpret_cast<volatile unsigned char*>(dev_ptr - 1);
      }
    }
  }
  return status;
}

hipError_t GraphExec::UpdateAQLPacket(hip::GraphKernelNode* node) {
  if (parallelLists_.size() == 1) {
    size_t pool_new_usage = 0;
    address result = nullptr;
    if (!kernarg_graph_.empty()) {
      // 1. Allocate memory for the kernel args
      size_t kernArgSizeForNode = 0;
      kernArgSizeForNode = node->GetKerArgSize();

      result = amd::alignUp(kernarg_graph_.back() + kernarg_graph_cur_offset_,
                            node->GetKernargSegmentAlignment());
      pool_new_usage = (result + kernArgSizeForNode) - kernarg_graph_.back();
    }
    if (pool_new_usage != 0 && pool_new_usage <= kernarg_graph_size_) {
      kernarg_graph_cur_offset_ = pool_new_usage;
    } else {
      address kernarg_graph;
      auto device = g_devices[ihipGetDevice()]->devices()[0];
      if (device->info().largeBar_) {
        kernarg_graph = reinterpret_cast<address>(device->deviceLocalAlloc(kernarg_graph_size_));
      } else {
        kernarg_graph = reinterpret_cast<address>(
            device->hostAlloc(kernarg_graph_size_, 0, amd::Device::MemorySegment::kKernArg));
      }
      kernarg_graph_.push_back(kernarg_graph);
      kernarg_graph_cur_offset_ = 0;

      // 1. Allocate memory for the kernel args
      size_t kernArgSizeForNode = 0;
      kernArgSizeForNode = node->GetKerArgSize();
      result = amd::alignUp(kernarg_graph_.back() + kernarg_graph_cur_offset_,
                            node->GetKernargSegmentAlignment());
      const size_t pool_new_usage = (result + kernArgSizeForNode) - kernarg_graph_.back();
      if (pool_new_usage <= kernarg_graph_size_) {
        kernarg_graph_cur_offset_ = pool_new_usage;
      }
    }

    // 2. copy kernel args / create new AQL packet
    node->CaptureAndFormPacket(capture_stream_, result);
  }
  return hipSuccess;
}

hipError_t FillCommands(std::vector<std::vector<Node>>& parallelLists,
                        std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
                        std::vector<Node>& topoOrder, Graph* clonedGraph,
                        amd::Command*& graphStart, amd::Command*& graphEnd, hip::Stream* stream) {
  hipError_t status = hipSuccess;
  for (auto& node : topoOrder) {
    // TODO: clone commands from next launch
    status = node->CreateCommand(node->GetQueue());
    if (status != hipSuccess) return status;
    amd::Command::EventWaitList waitList;
    for (auto depNode : nodeWaitLists[node]) {
      for (auto command : depNode->GetCommands()) {
        waitList.push_back(command);
      }
    }
    node->UpdateEventWaitLists(waitList);
  }

  std::vector<Node> rootNodes = clonedGraph->GetRootNodes();
  ClPrint(amd::LOG_INFO, amd::LOG_CODE,
          "[hipGraph] RootCommand get launched on stream %p", stream);

  for (auto& root : rootNodes) {
    //If rootnode is launched on to the same stream dont add dependency
    if (root->GetQueue() != stream) {
      if (graphStart == nullptr) {
        graphStart = new amd::Marker(*stream, false, {});
        if (graphStart == nullptr) {
          return hipErrorOutOfMemory;
        }
      }
      amd::Command::EventWaitList waitList;
      waitList.push_back(graphStart);
      auto commands = root->GetCommands();
      if (!commands.empty()) {
        commands[0]->updateEventWaitList(waitList);
      }
    }
  }

  // graphEnd ensures next enqueued ones start after graph is finished (all parallel branches)
  amd::Command::EventWaitList graphLastCmdWaitList;
  std::vector<Node> leafNodes = clonedGraph->GetLeafNodes();

  for (auto& leaf : leafNodes) {
    // If leaf node is launched on to the same stream dont add dependency
    if (leaf->GetQueue() != stream) {
      amd::Command::EventWaitList waitList;
      waitList.push_back(graphEnd);
      auto commands = leaf->GetCommands();
      if (!commands.empty()) {
        graphLastCmdWaitList.push_back(commands.back());
      }
    }
  }
  if (!graphLastCmdWaitList.empty()) {
    graphEnd = new amd::Marker(*stream, false, graphLastCmdWaitList);
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[hipGraph] EndCommand will get launched on stream %p", stream);
    if (graphEnd == nullptr) {
      graphStart->release();
      return hipErrorOutOfMemory;
    }
  }
  return hipSuccess;
}

void UpdateStream(std::vector<std::vector<Node>>& parallelLists, hip::Stream* stream,
                 GraphExec* ptr) {
  int i = 0;
  for (const auto& list : parallelLists) {
    // first parallel list will be launched on the same queue as parent
    if (i == 0) {
      for (auto& node : list) {
        node->SetStream(stream, ptr);
      }
    } else {  // New stream for parallel branches
      hip::Stream* stream = ptr->GetAvailableStreams();
      for (auto& node : list) {
        node->SetStream(stream, ptr);
      }
    }
    i++;
  }
}

hipError_t EnqueueGraphWithSingleList(std::vector<hip::Node>& topoOrder, hip::Stream* hip_stream,
                                      hip::GraphExec* graphExec) {
  // Accumulate command tracks all the AQL packet batch that we submit to the HW. For now
  // we track only kernel nodes.
  amd::AccumulateCommand* accumulate = nullptr;
  hipError_t status = hipSuccess;
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    accumulate = new amd::AccumulateCommand(*hip_stream, {}, nullptr);
  }
  for (int i = 0; i < topoOrder.size(); i++) {
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE && topoOrder[i]->GetType() == hipGraphNodeTypeKernel &&
        !reinterpret_cast<hip::GraphKernelNode*>(topoOrder[i])->HasHiddenHeap()) {
      if (topoOrder[i]->GetEnabled()) {
        hip_stream->vdev()->dispatchAqlPacket(topoOrder[i]->GetAqlPacket(),
                                              topoOrder[i]->GetKernelName(),
                                              accumulate);
      }
    } else {
      topoOrder[i]->SetStream(hip_stream, graphExec);
      status = topoOrder[i]->CreateCommand(topoOrder[i]->GetQueue());
      topoOrder[i]->EnqueueCommands(hip_stream);
    }
  }

  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    accumulate->enqueue();
    accumulate->release();
  }
  return status;
}

hipError_t GraphExec::Run(hipStream_t graph_launch_stream) {
  hipError_t status = hipSuccess;

  hip::Stream* launch_stream = hip::getStream(graph_launch_stream);

  if (flags_ & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (!topoOrder_.empty()) {
      topoOrder_[0]->GetParentGraph()->FreeAllMemory(launch_stream);
    }
  }

  // If this is a repeat launch, make sure corresponding MemFreeNode exists for a MemAlloc node
  if (repeatLaunch_ == true) {
    for (auto& node : topoOrder_) {
      if (node->GetType() == hipGraphNodeTypeMemAlloc &&
          static_cast<GraphMemAllocNode*>(node)->IsActiveMem() == true) {
        return hipErrorInvalidValue;
      }
    }
  } else {
    repeatLaunch_ = true;
  }

  if (parallelLists_.size() == 1 &&
      instantiateDeviceId_ == launch_stream->DeviceId()) {
    status = EnqueueGraphWithSingleList(topoOrder_, launch_stream, this);
  } else if (parallelLists_.size() == 1 &&
             instantiateDeviceId_ != launch_stream->DeviceId()) {
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->SetStream(launch_stream, this);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      topoOrder_[i]->EnqueueCommands(launch_stream);
    }
  } else {
    UpdateStream(parallelLists_, launch_stream, this);
    amd::Command* rootCommand = nullptr;
    amd::Command* endCommand = nullptr;
    status = FillCommands(parallelLists_, nodeWaitLists_, topoOrder_, clonedGraph_, rootCommand,
                          endCommand, launch_stream);
    if (status != hipSuccess) {
      return status;
    }
    if (rootCommand != nullptr) {
      rootCommand->enqueue();
      rootCommand->release();
    }
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->EnqueueCommands(topoOrder_[i]->GetQueue());
    }
    if (endCommand != nullptr) {
      endCommand->enqueue();
      endCommand->release();
    }
  }
  amd::ScopedLock lock(GraphExecStatusLock_);
  GraphExecStatus_[this] = std::make_pair(launch_stream, false);
  ResetQueueIndex();
  return status;
}
void ReleaseGraphExec(int deviceId) {
  // Release all graph exec objects destroyed by user.
  amd::ScopedLock lock(GraphExecStatusLock_);
  for (auto itr = GraphExecStatus_.begin(); itr != GraphExecStatus_.end();) {
    auto pair = itr->second;
    if (pair.first->DeviceId() == deviceId) {
      if (pair.second == true) {
        ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] Release GraphExec");
        (itr->first)->release();
      }
      GraphExecStatus_.erase(itr++);
    } else {
      itr++;
    }
  }
}
void ReleaseGraphExec(hip::Stream* stream) {
  amd::ScopedLock lock(GraphExecStatusLock_);
  for (auto itr = GraphExecStatus_.begin(); itr != GraphExecStatus_.end();) {
    auto pair = itr->second;
    if (pair.first == stream) {
      if (pair.second == true) {
        ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] Release GraphExec");
        (itr->first)->release();
      }
      GraphExecStatus_.erase(itr++);
      break;
    } else {
      ++itr;
    }
  }
}
}  // namespace hip
