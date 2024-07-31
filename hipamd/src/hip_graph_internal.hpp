/* Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc.

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

#pragma once
#include <algorithm>
#include <queue>
#include <stack>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_graph_helper.hpp"
#include "hip_event.hpp"
#include "hip_platform.hpp"
#include "hip_mempool_impl.hpp"
#include "hip_vm.hpp"

typedef struct ihipExtKernelEvents {
  hipEvent_t startEvent_;
  hipEvent_t stopEvent_;
} ihipExtKernelEvents;

namespace hip {
struct Graph;
struct GraphNode;
struct GraphExec;
struct UserObject;
typedef GraphNode* Node;
hipError_t FillCommands(std::vector<std::vector<Node>>& parallelLists,
                        std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
                        std::vector<Node>& topoOrder, Graph* clonedGraph, amd::Command*& graphStart,
                        amd::Command*& graphEnd, hip::Stream* stream);
void UpdateStream(std::vector<std::vector<Node>>& parallelLists, hip::Stream* stream,
                  GraphExec* ptr);
hipError_t EnqueueGraphWithSingleList(std::vector<hip::Node>& topoOrder, hip::Stream* hip_stream,
                                      hip::GraphExec* graphExec = nullptr);
struct UserObject : public amd::ReferenceCountedObject {
  typedef void (*UserCallbackDestructor)(void* data);
  static std::unordered_set<UserObject*> ObjectSet_;
  static amd::Monitor UserObjectLock_;

 public:
  UserObject(UserCallbackDestructor callback, void* data, unsigned int flags)
      : ReferenceCountedObject(), callback_(callback), data_(data), flags_(flags) {
    amd::ScopedLock lock(UserObjectLock_);
    ObjectSet_.insert(this);
  }

  virtual ~UserObject() {
    amd::ScopedLock lock(UserObjectLock_);
    if (callback_ != nullptr) {
      callback_(data_);
    }
    ObjectSet_.erase(this);
  }

  void increaseRefCount(const unsigned int refCount) {
    for (uint32_t i = 0; i < refCount; i++) {
      retain();
    }
  }

  void decreaseRefCount(const unsigned int refCount) {
    assert((refCount <= referenceCount()) && "count is bigger than refcount");
    for (uint32_t i = 0; i < refCount; i++) {
      release();
    }
  }

  static bool isUserObjvalid(UserObject* pUsertObj) {
    auto it = ObjectSet_.find(pUsertObj);
    if (it == ObjectSet_.end()) {
      return false;
    }
    return true;
  }

  static void removeUSerObj(UserObject* pUsertObj) {
    amd::ScopedLock lock(UserObjectLock_);
    auto it = ObjectSet_.find(pUsertObj);
    if (it != ObjectSet_.end()) {
      ObjectSet_.erase(it);
    }
  }

 private:
  UserCallbackDestructor callback_;
  void* data_;
  unsigned int flags_;
  //! Disable default operator=
  UserObject& operator=(const UserObject&) = delete;
  //! Disable copy constructor
  UserObject(const UserObject& obj) = delete;
};

struct hipGraphNodeDOTAttribute {
 protected:
  std::string style_;
  std::string shape_;
  std::string label_;

  hipGraphNodeDOTAttribute(std::string style, std::string shape, std::string label) {
    style_ = style;
    shape_ = shape;
    label_ = label;
  }

  hipGraphNodeDOTAttribute() {
    style_ = "solid";
    shape_ = "rectangle";
    label_ = "";
  }

  hipGraphNodeDOTAttribute(const hipGraphNodeDOTAttribute& node) {
    style_ = node.style_;
    shape_ = node.shape_;
    label_ = node.label_;
  }

  void SetStyle(std::string style) { style_ = style; }

  void SetShape(std::string shape) { shape_ = shape; }

  virtual std::string GetShape(hipGraphDebugDotFlags flag) { return shape_; }

  void SetLabel(std::string label) { label_ = label; }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) { return label_; }

  virtual void PrintAttributes(std::ostream& out, hipGraphDebugDotFlags flag) {
    out << "[";
    out << "style";
    out << "=\"";
    out << style_;
    out << "\"";
    out << "shape";
    out << "=\"";
    out << GetShape(flag);
    out << "\"";
    out << "label";
    out << "=\"";
    out << GetLabel(flag);
    out << "\"";
    out << "];";
  }
};

struct GraphNode : public hipGraphNodeDOTAttribute {
 protected:
  hip::Stream* stream_ = nullptr;
  unsigned int id_;
  hipGraphNodeType type_;
  std::vector<amd::Command*> commands_;
  std::vector<Node> edges_;
  std::vector<Node> dependencies_;
  bool visited_;
  // count of in coming edge  s
  size_t inDegree_;
  // count of outgoing edges
  size_t outDegree_;
  static int nextID;
  struct Graph* parentGraph_;
  static std::unordered_set<GraphNode*> nodeSet_;
  static amd::Monitor nodeSetLock_;
  unsigned int isEnabled_;
  uint8_t gpuPacket_[64];  //!< GPU Packet to enqueue during graph launch
  std::string capturedKernelName_;

 public:
  GraphNode(hipGraphNodeType type, std::string style = "", std::string shape = "",
               std::string label = "")
      : type_(type),
        visited_(false),
        inDegree_(0),
        outDegree_(0),
        id_(nextID++),
        parentGraph_(nullptr),
        isEnabled_(1),
        hipGraphNodeDOTAttribute(style, shape, label) {
    amd::ScopedLock lock(nodeSetLock_);
    nodeSet_.insert(this);
  }
  /// Copy Constructor
  GraphNode(const GraphNode& node) : hipGraphNodeDOTAttribute(node) {
    type_ = node.type_;
    inDegree_ = node.inDegree_;
    outDegree_ = node.outDegree_;
    visited_ = false;
    id_ = node.id_;
    parentGraph_ = nullptr;
    amd::ScopedLock lock(nodeSetLock_);
    nodeSet_.insert(this);
    isEnabled_ = node.isEnabled_;
  }

  virtual ~GraphNode() {
    for (auto node : edges_) {
      node->RemoveDependency(this);
    }
    for (auto node : dependencies_) {
      node->RemoveEdge(this);
    }
    amd::ScopedLock lock(nodeSetLock_);
    nodeSet_.erase(this);
  }

  // check node validity
  static bool isNodeValid(GraphNode* pGraphNode) {
    amd::ScopedLock lock(nodeSetLock_);
    if (pGraphNode == nullptr || nodeSet_.find(pGraphNode) == nodeSet_.end()) {
      return false;
    }
    return true;
  }
  // Return gpu packet address to update with actual packet under capture.
  uint8_t* GetAqlPacket() { return gpuPacket_; }
  void SetKernelName(const std::string& kernelName) { capturedKernelName_ = kernelName; }
  const std::string& GetKernelName() const { return capturedKernelName_; }

  hip::Stream* GetQueue() const { return stream_; }

  virtual void SetStream(hip::Stream* stream, GraphExec* ptr = nullptr) {
    stream_ = stream;
  }
  /// Create amd::command for the graph node
  virtual hipError_t CreateCommand(hip::Stream* stream) {
    commands_.clear();
    stream_ = stream;
    return hipSuccess;
  }
  /// Return node unique ID
  int GetID() const { return id_; }
  /// Returns command for graph node
  virtual std::vector<amd::Command*>& GetCommands() { return commands_; }
  /// Returns graph node type
  hipGraphNodeType GetType() const { return type_; }
  /// Clone graph node
  virtual GraphNode* clone() const = 0;
  /// Returns graph node indegree
  size_t GetInDegree() const { return inDegree_; }
  /// Updates indegree of the node
  void SetInDegree(size_t inDegree) { inDegree_ = inDegree; }
  /// Returns graph node outdegree
  size_t GetOutDegree() const { return outDegree_; }
  ///  Updates outdegree of the node
  void SetOutDegree(size_t outDegree) { outDegree_ = outDegree; }
  /// Returns graph node dependencies
  const std::vector<Node>& GetDependencies() const { return dependencies_; }
  /// Update graph node dependecies
  void SetDependencies(std::vector<Node>& dependencies) {
    for (auto entry : dependencies) {
      dependencies_.push_back(entry);
    }
  }
  /// Add graph node dependency
  void AddDependency(const Node& node) { dependencies_.push_back(node); }
  /// Remove graph node dependency
  void RemoveDependency(const Node& node) {
    dependencies_.erase(std::remove(dependencies_.begin(), dependencies_.end(), node),
                        dependencies_.end());
  }
  void RemoveEdge(const Node& childNode) {
    edges_.erase(std::remove(edges_.begin(), edges_.end(), childNode), edges_.end());
  }
  /// Return graph node children
  const std::vector<Node>& GetEdges() const { return edges_; }
  /// Updates graph node children
  void SetEdges(std::vector<Node>& edges) {
    for (auto entry : edges) {
      edges_.push_back(entry);
    }
  }
  /// Add edge, update parent node outdegree, child node indegree and dependency
  void AddEdge(const Node& childNode) {
    edges_.push_back(childNode);
    outDegree_++;
    childNode->SetInDegree(childNode->GetInDegree() + 1);
    childNode->AddDependency(this);
  }
  /// Remove edge, update parent node outdegree, child node indegree and dependency
  bool RemoveUpdateEdge(const Node& childNode) {
    // std::remove changes the end() hence saving it before hand for validation
    auto currEdgeEnd = edges_.end();
    auto it = std::remove(edges_.begin(), edges_.end(), childNode);
    if (it == currEdgeEnd) {
      // Should come here if childNode is not present in the edge list
      return false;
    }
    edges_.erase(it, edges_.end());
    outDegree_--;
    childNode->SetInDegree(childNode->GetInDegree() - 1);
    childNode->RemoveDependency(this);
    return true;
  }
  /// Get Runlist of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual void GetRunList(std::vector<std::vector<Node>>& parallelList,
                          std::unordered_map<Node, std::vector<Node>>& dependencies) {}
  /// Get topological sort of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual bool TopologicalOrder(std::vector<Node>& TopoOrder) { return true; }
  /// Update waitlist of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual void UpdateEventWaitLists(amd::Command::EventWaitList waitList) {
    for (auto command : commands_) {
      command->updateEventWaitList(waitList);
    }
  }
  virtual hipError_t GetNumParallelStreams(size_t &num) { return hipSuccess; }
  /// Enqueue commands part of the node
  virtual void EnqueueCommands(hip::Stream* stream) {
    // If the node is disabled it becomes empty node. To maintain ordering just enqueue marker.
    // Node can be enabled/disabled only for kernel, memcpy and memset nodes.
    if (!isEnabled_ &&
        (type_ == hipGraphNodeTypeKernel || type_ == hipGraphNodeTypeMemcpy ||
         type_ == hipGraphNodeTypeMemset)) {
      amd::Command::EventWaitList waitList;
      if (!commands_.empty()) {
        waitList = commands_[0]->eventWaitList();
      }
      amd::Command* command = new amd::Marker(*stream, !kMarkerDisableFlush, waitList);
      command->enqueue();
      command->release();
      return;
    }
    for (auto& command : commands_) {
      command->enqueue();
      command->release();
    }
  }
  Graph* GetParentGraph() { return parentGraph_; }
  virtual Graph* GetChildGraph() { return nullptr; }
  void SetParentGraph(Graph* graph) { parentGraph_ = graph; }
  virtual hipError_t SetParams(GraphNode* node) { return hipSuccess; }
  virtual void GenerateDOT(std::ostream& fout, hipGraphDebugDotFlags flag) {}
  virtual void GenerateDOTNode(size_t graphId, std::ostream& fout, hipGraphDebugDotFlags flag) {
    fout << "\n";
    std::string nodeName = "graph_" + std::to_string(graphId) + "_node_" + std::to_string(GetID());
    fout << "\"" << nodeName << "\"";
    PrintAttributes(fout, flag);
    fout << "\n";
  }
  virtual void GenerateDOTNodeEdges(size_t graphId, std::ostream& fout,
                                    hipGraphDebugDotFlags flag) {
    for (auto node : edges_) {
      std::string toNodeName =
          "graph_" + std::to_string(graphId) + "_node_" + std::to_string(node->GetID());
      std::string fromNodeName =
          "graph_" + std::to_string(graphId) + "_node_" + std::to_string(GetID());
      fout << "\"" << fromNodeName << "\" -> \"" << toNodeName << "\"" << std::endl;
    }
  }
  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    return (std::to_string(id_) + "\n" + label_);
  }
  unsigned int GetEnabled() const { return isEnabled_; }
  void SetEnabled(unsigned int isEnabled) { isEnabled_ = isEnabled; }
};

struct Graph {
  std::vector<Node> vertices_;
  const Graph* pOriginalGraph_ = nullptr;
  static std::unordered_set<Graph*> graphSet_;
  static amd::Monitor graphSetLock_;
  std::unordered_set<UserObject*> graphUserObj_;
  unsigned int id_;
  static int nextID;
  hip::Device* device_;       //!< HIP device object
  hip::MemoryPool* mem_pool_; //!< Memory pool, associated with this graph
  std::unordered_set<GraphNode*> capturedNodes_;
  bool graphInstantiated_;

 public:
  Graph(hip::Device* device, const Graph* original = nullptr)
      : pOriginalGraph_(original)
      , id_(nextID++)
      , device_(device) {
    amd::ScopedLock lock(graphSetLock_);
    graphSet_.insert(this);
    mem_pool_ = device->GetGraphMemoryPool();
    mem_pool_->retain();
    graphInstantiated_ = false;
  }

  ~Graph() {
    for (auto node : vertices_) {
      delete node;
    }
    amd::ScopedLock lock(graphSetLock_);
    graphSet_.erase(this);
    for (auto userobj : graphUserObj_) {
      userobj->release();
    }
    if (mem_pool_ != nullptr) {
      mem_pool_->release();
    }

  }

  void AddManualNodeDuringCapture(GraphNode* node) { capturedNodes_.insert(node); }

  std::unordered_set<GraphNode*> GetManualNodesDuringCapture() { return capturedNodes_; }

  void RemoveManualNodesDuringCapture() {
    capturedNodes_.erase(capturedNodes_.begin(), capturedNodes_.end());
  }

  /// Return graph unique ID
  int GetID() const { return id_; }

  // check graphs validity
  static bool isGraphValid(Graph* pGraph);

  /// add node to the graph
  void AddNode(const Node& node);
  void RemoveNode(const Node& node);
  /// Returns root nodes, all vertices with 0 in-degrees
  std::vector<Node> GetRootNodes() const;
  /// Returns leaf nodes, all vertices with 0 out-degrees
  std::vector<Node> GetLeafNodes() const;
  /// Returns number of leaf nodes
  size_t GetLeafNodeCount() const;
  /// Returns total numbers of nodes in the graph
  size_t GetNodeCount() const { return vertices_.size(); }
  /// returns all the nodes in the graph
  const std::vector<Node>& GetNodes() const { return vertices_; }
  /// returns all the edges in the graph
  std::vector<std::pair<Node, Node>> GetEdges() const;
  // returns the original graph ptr if cloned
  const Graph* getOriginalGraph() const { return pOriginalGraph_; }
  // Add user obj resource to graph
  void addUserObjGraph(UserObject* pUserObj) {
    amd::ScopedLock lock(graphSetLock_);
    graphUserObj_.insert(pUserObj);
  }
  // Check user obj resource from graph is valid
  bool isUserObjGraphValid(UserObject* pUserObj) {
    if (graphUserObj_.find(pUserObj) == graphUserObj_.end()) {
      return false;
    }
    return true;
  }
  // Delete user obj resource from graph
  void RemoveUserObjGraph(UserObject* pUserObj) { graphUserObj_.erase(pUserObj); }

  void GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                      std::vector<Node>& singleList, std::vector<std::vector<Node>>& parallelLists,
                      std::unordered_map<Node, std::vector<Node>>& dependencies);
  void GetRunList(std::vector<std::vector<Node>>& parallelLists,
                  std::unordered_map<Node, std::vector<Node>>& dependencies);
  bool TopologicalOrder(std::vector<Node>& TopoOrder);

  Graph* clone(std::unordered_map<Node, Node>& clonedNodes) const;
  Graph* clone() const;
  void GenerateDOT(std::ostream& fout, hipGraphDebugDotFlags flag) {
    fout << "subgraph cluster_" << GetID() << " {" << std::endl;
    fout << "label=\"graph_" << GetID() <<"\"graph[style=\"dashed\"];\n";
    for (auto node : vertices_) {
      node->GenerateDOTNode(GetID(), fout, flag);
    }
    fout << "\n";
    for (auto& node : vertices_) {
      node->GenerateDOTNodeEdges(GetID(), fout, flag);
    }
    fout << "}" << std::endl;
    for (auto node : vertices_) {
      node->GenerateDOT(fout, flag);
    }
  }

  void* AllocateMemory(size_t size, hip::Stream* stream, void* dptr) const {
    auto ptr = mem_pool_->AllocateMemory(size, stream, dptr);
    return ptr;
  }

  void* ReserveAddress(size_t size) const {
    void* startAddress = nullptr;
    void* ptr;
    const auto& dev_info = g_devices[0]->devices()[0]->info();

    size = amd::alignUp(size, dev_info.virtualMemAllocGranularity_);
    // Single virtual alloc would reserve for all devices.
    ptr = g_devices[0]->devices()[0]->virtualAlloc(startAddress, size,
            dev_info.virtualMemAllocGranularity_);
    if (ptr == nullptr) {
      LogError("Failed to reserve Virtual Address");
    }

    return ptr;
  }

  void FreeAddress(void* ptr) const {
    // Single Free would free for all devices.
    g_devices[0]->devices()[0]->virtualFree(ptr);
  }

  void FreeMemory(void* dev_ptr, hip::Stream* stream) const {
    size_t offset = 0;
    auto memory = getMemoryObject(dev_ptr, offset);
    if (memory != nullptr) {
      auto device_id = memory->getUserData().deviceId;
      if (!g_devices[device_id]->FreeMemory(memory, stream)) {
        LogError("Memory didn't belong to any pool!");
      }
    }
  }

  bool ProbeMemory(void* dev_ptr) const {
    size_t offset = 0;
    auto memory = getMemoryObject(dev_ptr, offset);
    if (memory != nullptr) {
      return mem_pool_->IsBusyMemory(memory);
    }
    return false;
  }

  void FreeAllMemory(hip::Stream* stream) {
    mem_pool_->FreeAllMemory(stream);
  }

  bool IsGraphInstantiated() const {
    return graphInstantiated_;
  }

  void SetGraphInstantiated(bool graphInstantiate) {
    graphInstantiated_ = graphInstantiate;
  }
};
struct GraphKernelNode;
struct GraphExec : public amd::ReferenceCountedObject {
  std::vector<std::vector<Node>> parallelLists_;
  // Topological order of the graph doesn't include nodes embedded as part of the child graph
  std::vector<Node> topoOrder_;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists_;
  struct Graph* clonedGraph_;
  std::vector<hip::Stream*> parallel_streams_;
  hip::Stream* capture_stream_;
  uint currentQueueIndex_;
  std::unordered_map<Node, Node> clonedNodes_;
  amd::Command* lastEnqueuedCommand_;
  static std::unordered_set<GraphExec*> graphExecSet_;
  static amd::Monitor graphExecSetLock_;
  uint64_t flags_ = 0;
  bool repeatLaunch_ = false;
  // Graph Kernel arg vars
  bool device_kernarg_pool_ = false;
  address kernarg_pool_graph_ = nullptr;
  uint32_t kernarg_pool_size_graph_ = 0;
  uint32_t kernarg_pool_cur_graph_offset_ = 0;
  std::vector<address> kernarg_graph_;
  uint32_t kernarg_graph_cur_offset_ = 0;
  uint32_t kernarg_graph_size_ = 128 * Ki;
  int instantiateDeviceId_ = -1;

 public:
  GraphExec(std::vector<Node>& topoOrder, std::vector<std::vector<Node>>& lists,
            std::unordered_map<Node, std::vector<Node>>& nodeWaitLists, struct Graph*& clonedGraph,
            std::unordered_map<Node, Node>& clonedNodes, uint64_t flags = 0)
      : ReferenceCountedObject(),
        parallelLists_(lists),
        topoOrder_(topoOrder),
        nodeWaitLists_(nodeWaitLists),
        clonedGraph_(clonedGraph),
        clonedNodes_(clonedNodes),
        lastEnqueuedCommand_(nullptr),
        currentQueueIndex_(0),
        flags_(flags) {
    amd::ScopedLock lock(graphExecSetLock_);
    graphExecSet_.insert(this);
  }

  ~GraphExec() {
    // new commands are launched for every launch they are destroyed as and when command is
    // terminated after it complete execution
    for (auto stream : parallel_streams_) {
      if (stream != nullptr) {
        hip::Stream::Destroy(stream);
      }
    }
    // Release the kernel arg memory.
    auto device = g_devices[ihipGetDevice()]->devices()[0];
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      if (kernarg_pool_size_graph_ != 0) {
        device->hostFree(kernarg_pool_graph_, kernarg_pool_size_graph_);
        for (auto& element : kernarg_graph_) {
          device->hostFree(element, kernarg_graph_size_);
        }
      }
    }
    amd::ScopedLock lock(graphExecSetLock_);
    graphExecSet_.erase(this);
    delete clonedGraph_;
  }

  Node GetClonedNode(Node node) {
    Node clonedNode;
    if (clonedNodes_.find(node) == clonedNodes_.end()) {
      return nullptr;
    } else {
      clonedNode = clonedNodes_[node];
    }
    return clonedNode;
  }

  address allocKernArg(size_t size, size_t alignment) {
    assert(alignment != 0);
    address result = nullptr;
    result = amd::alignUp(kernarg_pool_graph_ + kernarg_pool_cur_graph_offset_, alignment);
    const size_t pool_new_usage = (result + size) - kernarg_pool_graph_;
    if (pool_new_usage <= kernarg_pool_size_graph_) {
      kernarg_pool_cur_graph_offset_ = pool_new_usage;
    }
    return result;
  }

  // check executable graphs validity
  static bool isGraphExecValid(GraphExec* pGraphExec);

  std::vector<Node>& GetNodes() { return topoOrder_; }

  hip::Stream* GetAvailableStreams() {
    if (currentQueueIndex_ < parallel_streams_.size()) {
      return parallel_streams_[currentQueueIndex_++];
    }
    return nullptr;
  }

  void ResetQueueIndex() { currentQueueIndex_ = 0; }
  hipError_t Init();
  hipError_t CreateStreams(uint32_t num_streams);
  hipError_t Run(hipStream_t stream);
  // Capture GPU Packets from graph commands
  hipError_t CaptureAQLPackets();
  hipError_t UpdateAQLPacket(hip::GraphKernelNode* node);
  using KernelArgImpl = device::Settings::KernelArgImpl;
};

struct ChildGraphNode : public GraphNode {
  struct Graph* childGraph_;
  std::vector<Node> childGraphNodeOrder_;
  std::vector<std::vector<Node>> parallelLists_;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists_;
  amd::Command* lastEnqueuedCommand_;
  amd::Command* startCommand_;
  amd::Command* endCommand_;
  bool graphCaptureStatus_;
 public:
  ChildGraphNode(Graph* g) : GraphNode(hipGraphNodeTypeGraph, "solid", "rectangle") {
    childGraph_ = g->clone();
    lastEnqueuedCommand_ = nullptr;
    startCommand_ = nullptr;
    endCommand_ = nullptr;
    graphCaptureStatus_ = false;
  }

  ~ChildGraphNode() { delete childGraph_; }

  ChildGraphNode(const ChildGraphNode& rhs) : GraphNode(rhs) {
    childGraph_ = rhs.childGraph_->clone();
    graphCaptureStatus_ = rhs.graphCaptureStatus_;
  }

  GraphNode* clone() const override {
    return new ChildGraphNode(static_cast<ChildGraphNode const&>(*this));
  }

  Graph* GetChildGraph() override { return childGraph_; }

  void SetGraphCaptureStatus(bool status) { graphCaptureStatus_ = status; }

  bool GetGraphCaptureStatus() { return graphCaptureStatus_; }

  std::vector<Node>& GetChildGraphNodeOrder() {
    return childGraphNodeOrder_;
  }

  std::vector<std::vector<Node>>& GetParallelLists() {
    return parallelLists_;
  }


  hipError_t GetNumParallelStreams(size_t &num) override {
    if (false == TopologicalOrder(childGraphNodeOrder_)) {
      return hipErrorInvalidValue;
    }
    for (auto& node : childGraphNodeOrder_) {
      if (hipSuccess != node->GetNumParallelStreams(num)) {
        return hipErrorInvalidValue;
      }
    }
    // returns total number of parallel queues required for child graph nodes to be launched
    // first parallel list will be launched on the same queue as parent
    num += (parallelLists_.size() - 1);
    return hipSuccess;
  }

  void SetStream(hip::Stream* stream, GraphExec* ptr = nullptr) override {
    stream_ = stream;
    UpdateStream(parallelLists_, stream, ptr);
  }

  // For nodes that are dependent on the child graph node waitlist is the last node of the first
  // parallel list
  std::vector<amd::Command*>& GetCommands() override {
    return parallelLists_[0].back()->GetCommands();
  }

  // Create child graph node commands and set waitlists
  hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    startCommand_ = nullptr;
    endCommand_ = nullptr;
    if (!graphCaptureStatus_) {
      status = FillCommands(parallelLists_, nodeWaitLists_, childGraphNodeOrder_, childGraph_,
                            startCommand_, endCommand_, stream);
    }
    return status;
  }

  //
  void UpdateEventWaitLists(amd::Command::EventWaitList waitList) override {
    if (startCommand_ != nullptr) {
      startCommand_->updateEventWaitList(waitList);
    }
  }

  void GetRunList(std::vector<std::vector<Node>>& parallelList,
                  std::unordered_map<Node, std::vector<Node>>& dependencies) override {
    childGraph_->GetRunList(parallelLists_, nodeWaitLists_);
  }
  bool TopologicalOrder(std::vector<Node>& TopoOrder) override {
    return childGraph_->TopologicalOrder(TopoOrder);
  }
  void EnqueueCommands(hip::Stream* stream) override {
    if (graphCaptureStatus_) {
      hipError_t status =
          EnqueueGraphWithSingleList(childGraphNodeOrder_, stream);
    } else {
      // enqueue child graph start command
      if (startCommand_ != nullptr) {
        startCommand_->enqueue();
        startCommand_->release();
      }
      // enqueue nodes in child graph in level order
      for (auto& node : childGraphNodeOrder_) {
        node->EnqueueCommands(stream);
      }
      // enqueue child graph end command
      if (endCommand_ != nullptr) {
        endCommand_->enqueue();
        endCommand_->release();
      }
    }
  }

  hipError_t SetParams(const Graph* childGraph) {
    const std::vector<Node>& newNodes = childGraph->GetNodes();
    const std::vector<Node>& oldNodes = childGraph_->GetNodes();
    for (std::vector<Node>::size_type i = 0; i != newNodes.size(); i++) {
      hipError_t status = oldNodes[i]->SetParams(newNodes[i]);
      if (status != hipSuccess) {
        return status;
      }
    }
    return hipSuccess;
  }

  hipError_t SetParams(GraphNode* node) override {
    const ChildGraphNode* childGraphNode = static_cast<ChildGraphNode const*>(node);
    return SetParams(childGraphNode->childGraph_);
  }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    return std::to_string(GetID()) + "\n" + "graph_" + std::to_string(childGraph_->GetID());
  }

  virtual void GenerateDOT(std::ostream& fout, hipGraphDebugDotFlags flag) override {
    childGraph_->GenerateDOT(fout, flag);
  }
};

class GraphKernelNode : public GraphNode {
  hipKernelNodeParams kernelParams_;   //!< Kernel node parameters
  unsigned int numParams_;             //!< No. of kernel params as part of signature
  hipKernelNodeAttrValue kernelAttr_;  //!< Kernel node attributes
  unsigned int kernelAttrInUse_;       //!< Kernel attributes in use
  ihipExtKernelEvents kernelEvents_;   //!< Events for Ext launch kernel
  size_t alignedKernArgSize_;          //!< Aligned size required for kernel args
  size_t kernargSegmentByteSize_;      //!< Kernel arg segment byte size
  size_t kernargSegmentAlignment_;     //!< Kernel arg segment alignment
  bool hasHiddenHeap_;                 //!< Kernel has hidden heap(device side allocation)


 public:
  size_t GetKerArgSize() const { return alignedKernArgSize_; }
  size_t GetKernargSegmentByteSize() const { return kernargSegmentByteSize_; }
  size_t GetKernargSegmentAlignment() const { return kernargSegmentAlignment_; }
  bool HasHiddenHeap() const { return hasHiddenHeap_; }

  void PrintAttributes(std::ostream& out, hipGraphDebugDotFlags flag) override {
    out << "[";
    out << "style";
    out << "=\"";
    out << style_;
    (flag == hipGraphDebugDotFlagsKernelNodeParams ||
      flag == hipGraphDebugDotFlagsKernelNodeAttributes) ?
      out << "\n" : out << "\"";
    out << "shape";
    out << "=\"";
    out << GetShape(flag);
    out << "\"";
    out << "label";
    out << "=\"";
    out << GetLabel(flag);
    out << "\"";
    out << "];";
    }

  void CaptureAndFormPacket(hip::Stream* capture_stream, address kernArgOffset) {
    hipError_t status = CreateCommand(capture_stream);
    for (auto& command : commands_) {
      reinterpret_cast<amd::NDRangeKernelCommand*>(command)->setCapturingState(
          true, GetAqlPacket(), kernArgOffset);

      // Enqueue command to capture GPU Packet. The packet is not submitted to the device.
      // The packet is stored in gpuPacket_ and submitted during graph launch.
      command->submit(*(command->queue())->vdev());
      // Need to ensure if the command is NDRangeKernelCommand if we capture non kernel nodes
      SetKernelName(reinterpret_cast<amd::NDRangeKernelCommand*>(command)->kernel().name());
      command->release();
    }
  }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    hipFunction_t func = getFunc(kernelParams_, ihipGetDevice());
    hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
    std::string label;
    char buffer[500];
    if (flag == hipGraphDebugDotFlagsVerbose) {
      sprintf(buffer,
              "{\n%s\n| {ID | %d | %s\\<\\<\\<(%u,%u,%u),(%u,%u,%u),%u\\>\\>\\>}\n| {{node "
              "handle | func handle} | {%p | %p}}\n| {accessPolicyWindow | {base_ptr | num_bytes | "
              "hitRatio | hitProp | missProp} | {%p | %zu | %f | %d | %d}}\n| {cooperative | "
              "%u}\n| {priority | %d}\n}",
              label_.c_str(), GetID(), function->name().c_str(), kernelParams_.gridDim.x,
              kernelParams_.gridDim.y, kernelParams_.gridDim.z, kernelParams_.blockDim.x,
              kernelParams_.blockDim.y, kernelParams_.blockDim.z,
              kernelParams_.sharedMemBytes, this, kernelParams_.func,
              kernelAttr_.accessPolicyWindow.base_ptr, kernelAttr_.accessPolicyWindow.num_bytes,
              kernelAttr_.accessPolicyWindow.hitRatio, kernelAttr_.accessPolicyWindow.hitProp,
              kernelAttr_.accessPolicyWindow.missProp, kernelAttr_.cooperative,
              kernelAttr_.priority);
      label = buffer;
    }
    else if (flag == hipGraphDebugDotFlagsKernelNodeAttributes) {
      sprintf(buffer,
              "{\n%s\n| {ID | %d | %s}\n"
              "| {accessPolicyWindow | {base_ptr | num_bytes | "
              "hitRatio | hitProp | missProp} |\n| {%p | %zu | %f | %d | %d}}\n| {cooperative | "
              "%u}\n| {priority | %d}\n}",
              label_.c_str(), GetID(), function->name().c_str(),
              kernelAttr_.accessPolicyWindow.base_ptr, kernelAttr_.accessPolicyWindow.num_bytes,
              kernelAttr_.accessPolicyWindow.hitRatio, kernelAttr_.accessPolicyWindow.hitProp,
              kernelAttr_.accessPolicyWindow.missProp, kernelAttr_.cooperative,
              kernelAttr_.priority);
      label = buffer;
    }
    else if (flag == hipGraphDebugDotFlagsKernelNodeParams) {
      sprintf(buffer, "%d\n%s\n\\<\\<\\<(%u,%u,%u),(%u,%u,%u),%u\\>\\>\\>",
              GetID(), function->name().c_str(), kernelParams_.gridDim.x,
              kernelParams_.gridDim.y, kernelParams_.gridDim.z,
              kernelParams_.blockDim.x, kernelParams_.blockDim.y,
              kernelParams_.blockDim.z, kernelParams_.sharedMemBytes);
      label = buffer;
    }
    else {
      label = std::to_string(GetID()) + "\n" + function->name() + "\n";
    }
    return label;
  }

  std::string GetShape(hipGraphDebugDotFlags flag) override {
    if (flag == hipGraphDebugDotFlagsKernelNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }

  static hipFunction_t getFunc(const hipKernelNodeParams& params, unsigned int device) {
    hipFunction_t func = nullptr;
    hipError_t status = PlatformState::instance().getStatFunc(&func, params.func, device);
    if (status == hipErrorInvalidSymbol) {
      // capturehipExtModuleLaunchKernel() mixes host function with hipFunction_t, so we convert
      // here. If it's wrong, later functions will fail
      func = static_cast<hipFunction_t>(params.func);
    } else if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] getStatFunc() failed with err %d", status);
    }
    return func;
  }

  hipError_t copyParams(const hipKernelNodeParams* pNodeParams) {
    hasHiddenHeap_ = false;
    hipFunction_t func = getFunc(*pNodeParams, ihipGetDevice());
    if (!func) {
      return hipErrorInvalidDeviceFunction;
    }
    hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
    amd::Kernel* kernel = function->kernel();
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      auto device = g_devices[ihipGetDevice()]->devices()[0];
      device::Kernel* devKernel = const_cast<device::Kernel*>(kernel->getDeviceKernel(*device));
      kernargSegmentByteSize_ = devKernel->KernargSegmentByteSize();
      kernargSegmentAlignment_ = devKernel->KernargSegmentAlignment();
      alignedKernArgSize_ =
          amd::alignUp(devKernel->KernargSegmentByteSize(), devKernel->KernargSegmentAlignment());
    }
    const amd::KernelSignature& signature = kernel->signature();
    numParams_ = signature.numParameters();

    // Allocate/assign memory if params are passed part of 'kernelParams'
    if (pNodeParams->kernelParams != nullptr) {
      kernelParams_.kernelParams = (void**)malloc(numParams_ * sizeof(void*));
      if (kernelParams_.kernelParams == nullptr) {
        return hipErrorOutOfMemory;
      }

      for (uint32_t i = 0; i < numParams_; ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        kernelParams_.kernelParams[i] = malloc(desc.size_);
        if (kernelParams_.kernelParams[i] == nullptr) {
          return hipErrorOutOfMemory;
        }
        ::memcpy(kernelParams_.kernelParams[i], (pNodeParams->kernelParams[i]), desc.size_);
      }
      for (uint32_t i = signature.numParameters(); i < signature.numParametersAll(); ++i) {
        if (signature.at(i).info_.oclObject_ == amd::KernelParameterDescriptor::HiddenHeap) {
          hasHiddenHeap_ = true;
        }
      }
    }

    // Allocate/assign memory if params are passed as part of 'extra'
    else if (pNodeParams->extra != nullptr) {
      // 'extra' is a struct that contains the following info: {
      // HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
      // HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargs_size,
      // HIP_LAUNCH_PARAM_END }
      unsigned int numExtra = 5;
      kernelParams_.extra = (void**)malloc(numExtra * sizeof(void*));
      if (kernelParams_.extra == nullptr) {
        return hipErrorOutOfMemory;
      }
      kernelParams_.extra[0] = pNodeParams->extra[0];
      size_t kernargs_size = *((size_t*)pNodeParams->extra[3]);
      kernelParams_.extra[1] = malloc(kernargs_size);
      if (kernelParams_.extra[1] == nullptr) {
        return hipErrorOutOfMemory;
      }
      kernelParams_.extra[2] = pNodeParams->extra[2];
      kernelParams_.extra[3] = malloc(sizeof(void*));
      if (kernelParams_.extra[3] == nullptr) {
        return hipErrorOutOfMemory;
      }
      *((size_t*)kernelParams_.extra[3]) = kernargs_size;
      ::memcpy(kernelParams_.extra[1], (pNodeParams->extra[1]), kernargs_size);
      kernelParams_.extra[4] = pNodeParams->extra[4];
    }
    return hipSuccess;
  }

  GraphKernelNode(const hipKernelNodeParams* pNodeParams, const ihipExtKernelEvents* pEvents)
      : GraphNode(hipGraphNodeTypeKernel, "bold", "octagon", "KERNEL") {
    kernelParams_ = *pNodeParams;
    kernelEvents_ = { 0 };
    if (pEvents != nullptr) {
      kernelEvents_ = *pEvents;
    }
    if (copyParams(pNodeParams) != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to copy params");
    }
    memset(&kernelAttr_, 0, sizeof(kernelAttr_));
    kernelAttrInUse_ = 0;
    hasHiddenHeap_ = false;
  }

  ~GraphKernelNode() { freeParams(); }

  void freeParams() {
    // Deallocate memory allocated for kernargs passed via 'kernelParams'
    if (kernelParams_.kernelParams != nullptr) {
      for (size_t i = 0; i < numParams_; ++i) {
        if (kernelParams_.kernelParams[i] != nullptr) {
          free(kernelParams_.kernelParams[i]);
        }
        kernelParams_.kernelParams[i] = nullptr;
      }
      free(kernelParams_.kernelParams);
      kernelParams_.kernelParams = nullptr;
    }
    // Deallocate memory allocated for kernargs passed via 'extra'
    else if (kernelParams_.extra != nullptr) {
      free(kernelParams_.extra[1]);
      free(kernelParams_.extra[3]);
      memset(kernelParams_.extra, 0, 5 * sizeof(kernelParams_.extra[0]));  // 5 items
      free(kernelParams_.extra);
      kernelParams_.extra = nullptr;
    }
  }

  GraphKernelNode(const GraphKernelNode& rhs) : GraphNode(rhs) {
    kernelParams_ = rhs.kernelParams_;
    kernelEvents_ = rhs.kernelEvents_;
    hipError_t status = copyParams(&rhs.kernelParams_);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to allocate memory to copy params");
    }
    memset(&kernelAttr_, 0, sizeof(kernelAttr_));
    kernelAttrInUse_ = 0;
    status = CopyAttr(&rhs);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to during copy attrs");
    }
  }

  GraphNode* clone() const override {
    return new GraphKernelNode(static_cast<GraphKernelNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipFunction_t func = nullptr;
    hipError_t status = validateKernelParams(&kernelParams_, &func,
                                             stream ? hip::getDeviceID(stream->context()) : -1);
    if (hipSuccess != status) {
      return status;
    }
    status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    status = ihipLaunchKernelCommand(
        command, func, kernelParams_.gridDim.x * kernelParams_.blockDim.x,
        kernelParams_.gridDim.y * kernelParams_.blockDim.y,
        kernelParams_.gridDim.z * kernelParams_.blockDim.z, kernelParams_.blockDim.x,
        kernelParams_.blockDim.y, kernelParams_.blockDim.z, kernelParams_.sharedMemBytes,
        stream, kernelParams_.kernelParams, kernelParams_.extra, kernelEvents_.startEvent_,
        kernelEvents_.stopEvent_, 0, 0, 0, 0, 0, 0, 0);
    commands_.emplace_back(command);
    return status;
  }

  void GetParams(hipKernelNodeParams* params) { *params = kernelParams_; }

  hipError_t SetParams(const hipKernelNodeParams* params) {
    // updates kernel params
    hipError_t status = validateKernelParams(params);
    if (hipSuccess != status) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to validateKernelParams");
      return status;
    }
    if ((kernelParams_.kernelParams && kernelParams_.kernelParams == params->kernelParams) ||
         (kernelParams_.extra && kernelParams_.extra == params->extra)) {
      // params is copied from kernelParams_ and then updated, so just copy it back
      kernelParams_ = *params;
      return status;
    }
    freeParams();
    kernelParams_ = *params;
    status = copyParams(params);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to set params");
    }
    return status;
  }

  hipError_t SetAttrParams(hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* params) {
    hipDeviceProp_t prop = {0};
    hipError_t status = ihipGetDeviceProperties(&prop, ihipGetDevice());
    if (hipSuccess != status){
      return status;
    }
    int accessPolicyMaxWindowSize = prop.accessPolicyMaxWindowSize;
    // updates kernel attr params
    if (attr == hipKernelNodeAttributeAccessPolicyWindow) {
      if (params->accessPolicyWindow.hitRatio > 1 ||
          params->accessPolicyWindow.hitRatio < 0) {
        return hipErrorInvalidValue;
      }

      if (params->accessPolicyWindow.missProp == hipAccessPropertyPersisting) {
        return hipErrorInvalidValue;
      }
      if (params->accessPolicyWindow.num_bytes > 0 && params->accessPolicyWindow.hitRatio == 0) {
        return hipErrorInvalidValue;
      }

      // need to check against accessPolicyMaxWindowSize from device
      // accessPolicyMaxWindowSize not implemented on the device side yet
      if (params->accessPolicyWindow.num_bytes > accessPolicyMaxWindowSize) {
        return hipErrorInvalidValue;
      }

      kernelAttr_.accessPolicyWindow.base_ptr = params->accessPolicyWindow.base_ptr;
      kernelAttr_.accessPolicyWindow.hitProp = params->accessPolicyWindow.hitProp;
      kernelAttr_.accessPolicyWindow.hitRatio = params->accessPolicyWindow.hitRatio;
      kernelAttr_.accessPolicyWindow.missProp = params->accessPolicyWindow.missProp;
      kernelAttr_.accessPolicyWindow.num_bytes = params->accessPolicyWindow.num_bytes;
    } else if (attr == hipKernelNodeAttributeCooperative) {
      kernelAttr_.cooperative = params->cooperative;
    } else if (attr == hipLaunchAttributePriority) {
      if (params->priority < hip::Stream::Priority::Low ||
          params->priority > hip::Stream::Priority::High){
        return hipErrorInvalidValue;
      }
      kernelAttr_.priority = params->priority;
    }

    kernelAttrInUse_ = attr;
    return hipSuccess;
  }
  hipError_t GetAttrParams(hipKernelNodeAttrID attr, hipKernelNodeAttrValue* params) {
    // Get kernel attr params
    if (kernelAttrInUse_ != 0 && kernelAttrInUse_ != attr) return hipErrorInvalidValue;
    if (attr == hipKernelNodeAttributeAccessPolicyWindow) {
      params->accessPolicyWindow.base_ptr = kernelAttr_.accessPolicyWindow.base_ptr;
      params->accessPolicyWindow.hitProp = kernelAttr_.accessPolicyWindow.hitProp;
      params->accessPolicyWindow.hitRatio = kernelAttr_.accessPolicyWindow.hitRatio;
      params->accessPolicyWindow.missProp = kernelAttr_.accessPolicyWindow.missProp;
      params->accessPolicyWindow.num_bytes = kernelAttr_.accessPolicyWindow.num_bytes;
    } else if (attr == hipKernelNodeAttributeCooperative) {
      params->cooperative = kernelAttr_.cooperative;
    } else if (attr == hipLaunchAttributePriority) {
      params->priority = kernelAttr_.priority;
    }
    return hipSuccess;
  }
  hipError_t CopyAttr(const GraphKernelNode* srcNode) {
    if (kernelAttrInUse_ == 0 && srcNode->kernelAttrInUse_ == 0) {
      return hipSuccess;
    }
    if (kernelAttrInUse_ != 0 && srcNode->kernelAttrInUse_ != kernelAttrInUse_) {
      return hipErrorInvalidContext;
    }
    kernelAttrInUse_ = srcNode->kernelAttrInUse_;
    switch (srcNode->kernelAttrInUse_) {
      case hipKernelNodeAttributeAccessPolicyWindow:
        kernelAttr_.accessPolicyWindow.base_ptr = srcNode->kernelAttr_.accessPolicyWindow.base_ptr;
        kernelAttr_.accessPolicyWindow.hitProp = srcNode->kernelAttr_.accessPolicyWindow.hitProp;
        kernelAttr_.accessPolicyWindow.hitRatio = srcNode->kernelAttr_.accessPolicyWindow.hitRatio;
        kernelAttr_.accessPolicyWindow.missProp = srcNode->kernelAttr_.accessPolicyWindow.missProp;
        kernelAttr_.accessPolicyWindow.num_bytes =
            srcNode->kernelAttr_.accessPolicyWindow.num_bytes;
        break;
      case hipKernelNodeAttributeCooperative:
        kernelAttr_.cooperative = srcNode->kernelAttr_.cooperative;
        break;
      case hipLaunchAttributePriority:
        kernelAttr_.priority = srcNode->kernelAttr_.priority;
        break;
      default:
        return hipErrorInvalidValue;
    }
    return hipSuccess;
  }

  hipError_t SetParams(GraphNode* node) override {
    const GraphKernelNode* kernelNode = static_cast<GraphKernelNode const*>(node);
    return SetParams(&kernelNode->kernelParams_);
  }

  static hipError_t validateKernelParams(const hipKernelNodeParams* pNodeParams,
                                         hipFunction_t* ptrFunc = nullptr, int devId = -1) {
    devId = devId == -1 ? ihipGetDevice() : devId;
    hipFunction_t func = getFunc(*pNodeParams, devId);
    if (!func) {
      return hipErrorInvalidDeviceFunction;
    }

    size_t globalWorkSizeX = static_cast<size_t>(pNodeParams->gridDim.x) * pNodeParams->blockDim.x;
    size_t globalWorkSizeY = static_cast<size_t>(pNodeParams->gridDim.y) * pNodeParams->blockDim.y;
    size_t globalWorkSizeZ = static_cast<size_t>(pNodeParams->gridDim.z) * pNodeParams->blockDim.z;

    hipError_t status = ihipLaunchKernel_validate(
        func, static_cast<uint32_t>(globalWorkSizeX), static_cast<uint32_t>(globalWorkSizeY),
        static_cast<uint32_t>(globalWorkSizeZ), pNodeParams->blockDim.x, pNodeParams->blockDim.y,
        pNodeParams->blockDim.z, pNodeParams->sharedMemBytes, pNodeParams->kernelParams,
        pNodeParams->extra, devId, 0);
    if (status != hipSuccess) {
      return status;
    }

    if (ptrFunc) *ptrFunc = func;
    return hipSuccess;
  }
};

class GraphMemcpyNode : public GraphNode {
 protected:
  hipMemcpy3DParms copyParams_;

 public:
  GraphMemcpyNode(const hipMemcpy3DParms* pCopyParams)
      : GraphNode(hipGraphNodeTypeMemcpy, "solid", "trapezium", "MEMCPY") {
    if (pCopyParams) {
      copyParams_ = *pCopyParams;
    }
  }
  ~GraphMemcpyNode() {}

  GraphMemcpyNode(const GraphMemcpyNode& rhs) : GraphNode(rhs) {
    copyParams_ = rhs.copyParams_;
  }

  GraphNode* clone() const override {
    return new GraphMemcpyNode(static_cast<GraphMemcpyNode const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) override {
    if ((copyParams_.kind == hipMemcpyHostToHost || copyParams_.kind == hipMemcpyDefault)
      && IsHtoHMemcpy(copyParams_.dstPtr.ptr, copyParams_.srcPtr.ptr)) {
      return hipSuccess;
    }
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    status = ihipMemcpy3DCommand(command, &copyParams_, stream);
    commands_.emplace_back(command);
    return status;
  }

  virtual void EnqueueCommands(hip::Stream* stream) override {
    if ( (copyParams_.kind == hipMemcpyHostToHost || copyParams_.kind == hipMemcpyDefault) &&
          isEnabled_ && IsHtoHMemcpy(copyParams_.dstPtr.ptr, copyParams_.srcPtr.ptr)) {
      ihipHtoHMemcpy(copyParams_.dstPtr.ptr, copyParams_.srcPtr.ptr,
                     copyParams_.extent.width * copyParams_.extent.height *
                     copyParams_.extent.depth, *stream);
      return;
    }
    GraphNode::EnqueueCommands(stream);
  }

  void GetParams(hipMemcpy3DParms* params) {
    std::memcpy(params, &copyParams_, sizeof(hipMemcpy3DParms));
  }

  virtual hipMemcpyKind GetMemcpyKind() const { return copyParams_.kind; };

  hipError_t SetParams(const hipMemcpy3DParms* params) {
    hipError_t status = ValidateParams(params);
    if (status != hipSuccess) {
      return status;
    }
    std::memcpy(&copyParams_, params, sizeof(hipMemcpy3DParms));
    return hipSuccess;
  }

  virtual hipError_t SetParams(GraphNode* node) override {
    const GraphMemcpyNode* memcpyNode = static_cast<GraphMemcpyNode const*>(node);
    return SetParams(&memcpyNode->copyParams_);
  }
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t ValidateParams(const hipMemcpy3DParms* pNodeParams);

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    size_t offset = 0;
    const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(copyParams_);
    hipMemoryType srcMemoryType = pCopy.srcMemoryType;
    if (srcMemoryType == hipMemoryTypeUnified) {
      srcMemoryType =
          getMemoryObject(pCopy.srcDevice, offset) ? hipMemoryTypeDevice : hipMemoryTypeHost;
    }
    offset = 0;
    hipMemoryType dstMemoryType = pCopy.dstMemoryType;
    if (dstMemoryType == hipMemoryTypeUnified) {
      dstMemoryType =
          getMemoryObject(pCopy.dstDevice, offset) ? hipMemoryTypeDevice : hipMemoryTypeHost;
    }

    // If {src/dst}MemoryType is hipMemoryTypeHost, check if the memory was prepinned.
    // In that case upgrade the copy type to hipMemoryTypeDevice to avoid extra pinning.
    offset = 0;
    if (srcMemoryType == hipMemoryTypeHost) {
      amd::Memory* mem = getMemoryObject(pCopy.srcHost, offset);
      srcMemoryType = mem ? hipMemoryTypeDevice : hipMemoryTypeHost;
    }
    if (dstMemoryType == hipMemoryTypeHost) {
      amd::Memory* mem = getMemoryObject(pCopy.dstHost, offset);
      dstMemoryType = mem ? hipMemoryTypeDevice : hipMemoryTypeHost;
    }
    std::string memcpyDirection;
    if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeDevice)) {
      // Host to Device.
      memcpyDirection = "HtoD";
    } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeHost)) {
      // Device to Host.
      memcpyDirection = "DtoH";
    } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeDevice)) {
      // Device to Device.
      memcpyDirection = "DtoD";
    } else if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeArray)) {
      memcpyDirection = "HtoA";
    } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeHost)) {
      // Image to Host.
      memcpyDirection = "AtoH";
    } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeArray)) {
      // Device to Image.
      memcpyDirection = "DtoA";
    } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeDevice)) {
      // Image to Device.
      memcpyDirection = "AtoD";
    } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeArray)) {
      memcpyDirection = "AtoA";
    }
    std::string label;
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      char buffer[500];
      sprintf(
          buffer,
          "{\n%s\n| {{ID | node handle} | {%u | %p}}\n| {kind | %s}\n| {{srcPtr | dstPtr} | "
          "{pitch "
          "| ptr | xsize | ysize | pitch | ptr | xsize | size} | {%zu | %p | %zu | %zu | %zu | %p "
          "| %zu "
          "| %zu}}\n| {{srcPos | {{x | %zu} | {y | %zu} | {z | %zu}}} | {dstPos | {{x | %zu} | {y "
          "| "
          "%zu} | {z | %zu}}} | {Extent | {{Width | %zu} | {Height | %zu} | {Depth | %zu}}}}\n}",
          label_.c_str(), GetID(), this, memcpyDirection.c_str(), copyParams_.srcPtr.pitch,
          copyParams_.srcPtr.ptr, copyParams_.srcPtr.xsize, copyParams_.srcPtr.ysize,
          copyParams_.dstPtr.pitch, copyParams_.dstPtr.ptr, copyParams_.dstPtr.xsize,
          copyParams_.dstPtr.ysize, copyParams_.srcPos.x, copyParams_.srcPos.y,
          copyParams_.srcPos.z, copyParams_.dstPos.x, copyParams_.dstPos.y,
          copyParams_.dstPos.z, copyParams_.extent.width, copyParams_.extent.height,
          copyParams_.extent.depth);
      label = buffer;
    } else {
      label = std::to_string(GetID()) + "\nMEMCPY\n(" + memcpyDirection + ")";
    }
    return label;
  }
  std::string GetShape(hipGraphDebugDotFlags flag) override {
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }
};

class GraphMemcpyNode1D : public GraphMemcpyNode {
 protected:
  void* dst_;
  const void* src_;
  size_t count_;
  hipMemcpyKind kind_;

 public:
  GraphMemcpyNode1D(void* dst, const void* src, size_t count, hipMemcpyKind kind,
                       hipGraphNodeType type = hipGraphNodeTypeMemcpy)
      : GraphMemcpyNode(nullptr),
        dst_(dst),
        src_(src),
        count_(count),
        kind_(kind) {}

  ~GraphMemcpyNode1D() {}

  GraphNode* clone() const override {
    return new GraphMemcpyNode1D(static_cast<GraphMemcpyNode1D const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) override {
    if ((kind_ == hipMemcpyHostToHost || kind_ == hipMemcpyDefault) && IsHtoHMemcpy(dst_, src_)) {
      return hipSuccess;
    }
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command = nullptr;
    status = ihipMemcpyCommand(command, dst_, src_, count_, kind_, *stream);
    commands_.emplace_back(command);
    return status;
  }

  virtual void EnqueueCommands(hip::Stream* stream) override {
    bool isH2H = false;
    if ((kind_ == hipMemcpyHostToHost || kind_ == hipMemcpyDefault) && IsHtoHMemcpy(dst_, src_)) {
      isH2H = true;
    }
    if (!isH2H) {
      if (commands_.empty()) return;
      // commands_ should have just 1 item
      assert(commands_.size() == 1 && "Invalid command size in GraphMemcpyNode1D");
    }
    if (isEnabled_) {
      //HtoH
      if (isH2H) {
        ihipHtoHMemcpy(dst_, src_, count_, *stream);
        return;
      }
      amd::Command* command = commands_[0];
      amd::HostQueue* cmdQueue = command->queue();

      if (cmdQueue == stream) {
        command->enqueue();
        command->release();
        return;
      }

      amd::Command::EventWaitList waitList;
      amd::Command* depdentMarker = nullptr;
      amd::Command* cmd = stream->getLastQueuedCommand(true);
      if (cmd != nullptr) {
        waitList.push_back(cmd);
        amd::Command* depdentMarker = new amd::Marker(*cmdQueue, true, waitList);
        if (depdentMarker != nullptr) {
          depdentMarker->enqueue();  // Make sure command synced with last command of queue
          depdentMarker->release();
        }
        cmd->release();
      }
      command->enqueue();
      command->release();

      cmd = cmdQueue->getLastQueuedCommand(true);  // should be command
      if (cmd != nullptr) {
        waitList.clear();
        waitList.push_back(cmd);
        amd::Command* depdentMarker = new amd::Marker(*stream, true, waitList);
        if (depdentMarker != nullptr) {
          depdentMarker->enqueue();  // Make sure future commands of queue synced with command
          depdentMarker->release();
        }
        cmd->release();
      }
    } else {
      amd::Command::EventWaitList waitList;
      amd::Command* command = new amd::Marker(*stream, !kMarkerDisableFlush, waitList);
      command->enqueue();
      command->release();
    }
  }

  hipMemcpyKind GetMemcpyKind() const override {
    return kind_;
  }

  hipError_t SetParams(void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    hipError_t status = ValidateParams(dst, src, count, kind);
    if (status != hipSuccess) {
      return status;
    }
    dst_ = dst;
    src_ = src;
    count_ = count;
    kind_ = kind;
    return hipSuccess;
  }

  virtual hipError_t SetParams(GraphNode* node) override {
    const GraphMemcpyNode1D* memcpy1DNode = static_cast<GraphMemcpyNode1D const*>(node);
    return SetParams(memcpy1DNode->dst_, memcpy1DNode->src_, memcpy1DNode->count_,
                     memcpy1DNode->kind_);
  }
  static hipError_t ValidateParams(void* dst, const void* src, size_t count, hipMemcpyKind kind);
  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    size_t sOffsetOrig = 0;
    amd::Memory* origSrcMemory = getMemoryObject(src_, sOffsetOrig);
    size_t dOffsetOrig = 0;
    amd::Memory* origDstMemory = getMemoryObject(dst_, dOffsetOrig);

    size_t sOffset = 0;
    amd::Memory* srcMemory = getMemoryObject(src_, sOffset);
    size_t dOffset = 0;
    amd::Memory* dstMemory = getMemoryObject(dst_, dOffset);
    std::string memcpyDirection;
    if ((srcMemory == nullptr) && (dstMemory != nullptr)) {  // host to device
      memcpyDirection = "HtoD";
    } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {  // device to host
      memcpyDirection = "DtoH";
    } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
      memcpyDirection = "DtoD";
    } else {
      if (kind_ == hipMemcpyHostToDevice) {
        memcpyDirection = "HtoD";
      } else if (kind_ == hipMemcpyDeviceToHost) {
        memcpyDirection = "DtoH";
      }
    }
    std::string label;
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      char buffer[500];
      sprintf(
          buffer,
          "{\n%s\n| {{ID | node handle} | {%u | %p}}\n| {kind | %s}\n| {{srcPtr | dstPtr} | "
          "{pitch "
          "| ptr | xsize | ysize | pitch | ptr | xsize | size} | {%zu | %p | %zu | %zu | %zu | %p "
          "| %zu "
          "| %zu}}\n| {{srcPos | {{x | %zu} | {y | %zu} | {z | %zu}}} | {dstPos | {{x | %zu} | {y "
          "| "
          "%zu} | {z | %zu}}} | {Extent | {{Width | %zu} | {Height | %zu} | {Depth | %zu}}}}\n}",
          label_.c_str(), GetID(), this, memcpyDirection.c_str(), (size_t)0, src_, (size_t)0,
          (size_t)0, (size_t)0, dst_, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (size_t)0,
          (size_t)0, (size_t)0, (size_t)0, count_, (size_t)1, (size_t)1);
      label = buffer;
    } else {
      label = std::to_string(GetID()) + "\n" + label_ + "\n(" + memcpyDirection + "," +
          std::to_string(count_) + ")";
    }
    return label;
  }
  std::string GetShape(hipGraphDebugDotFlags flag) override {
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }
};

class GraphMemcpyNodeFromSymbol : public GraphMemcpyNode1D {
  const void* symbol_;
  size_t offset_;

 public:
  GraphMemcpyNodeFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                               hipMemcpyKind kind)
      : GraphMemcpyNode1D(dst, nullptr, count, kind, hipGraphNodeTypeMemcpy),
        symbol_(symbol),
        offset_(offset) {}

  ~GraphMemcpyNodeFromSymbol() {}

  GraphNode* clone() const override {
    return new GraphMemcpyNodeFromSymbol(
        static_cast<GraphMemcpyNodeFromSymbol const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command = nullptr;
    size_t sym_size = 0;
    hipDeviceptr_t device_ptr = nullptr;

    status = ihipMemcpySymbol_validate(symbol_, count_, offset_, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
    }
    status = ihipMemcpyCommand(command, dst_, device_ptr, count_, kind_, *stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.emplace_back(command);
    return status;
  }

  hipError_t SetParams(void* dst, const void* symbol, size_t count, size_t offset,
                       hipMemcpyKind kind, bool isExec = false) {
    if (isExec) {
      size_t discardOffset = 0;
      amd::Memory *memObj = getMemoryObject(dst, discardOffset);
      if (memObj != nullptr) {
        amd::Memory *memObjOri = getMemoryObject(dst_, discardOffset);
        if (memObjOri != nullptr) {
          if (memObjOri->getUserData().deviceId != memObj->getUserData().deviceId) {
            return hipErrorInvalidValue;
          }
        }
      }
    }
    size_t sym_size = 0;
    hipDeviceptr_t device_ptr = nullptr;
    // check to see if dst is also a symbol (hip negative test case)
    hipError_t status = ihipMemcpySymbol_validate(dst, count, offset, sym_size, device_ptr);
    if (status == hipSuccess) {
      return hipErrorInvalidValue;
    }
    status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
    }

    size_t dOffset = 0;
    amd::Memory* dstMemory = getMemoryObject(dst, dOffset);
    if (dstMemory == nullptr && kind != hipMemcpyDeviceToHost && kind != hipMemcpyDefault) {
      return hipErrorInvalidMemcpyDirection;
    } else if (dstMemory != nullptr && dstMemory->getMemFlags() == 0 &&
               kind != hipMemcpyDeviceToDevice && kind != hipMemcpyDeviceToDeviceNoCU
               && kind != hipMemcpyDefault) {
      return hipErrorInvalidMemcpyDirection;
    } else if (kind == hipMemcpyHostToHost || kind == hipMemcpyHostToDevice) {
      return hipErrorInvalidMemcpyDirection;
    }

    dst_ = dst;
    symbol_ = symbol;
    count_ = count;
    offset_ = offset;
    kind_ = kind;
    return hipSuccess;
  }

  virtual hipError_t SetParams(GraphNode* node) override {
    const GraphMemcpyNodeFromSymbol* memcpyNode =
        static_cast<GraphMemcpyNodeFromSymbol const*>(node);
    return SetParams(memcpyNode->dst_, memcpyNode->symbol_, memcpyNode->count_, memcpyNode->offset_,
                     memcpyNode->kind_);
  }
};
class GraphMemcpyNodeToSymbol : public GraphMemcpyNode1D {
  const void* symbol_;
  size_t offset_;

 public:
  GraphMemcpyNodeToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                             hipMemcpyKind kind)
      : GraphMemcpyNode1D(nullptr, src, count, kind, hipGraphNodeTypeMemcpy),
        symbol_(symbol),
        offset_(offset) {}

  ~GraphMemcpyNodeToSymbol() {}

  GraphNode* clone() const override {
    return new GraphMemcpyNodeToSymbol(static_cast<GraphMemcpyNodeToSymbol const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command = nullptr;
    size_t sym_size = 0;
    hipDeviceptr_t device_ptr = nullptr;

    status = ihipMemcpySymbol_validate(symbol_, count_, offset_, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
    }
    status = ihipMemcpyCommand(command, device_ptr, src_, count_, kind_, *stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.emplace_back(command);
    return status;
  }

  hipError_t SetParams(const void* symbol, const void* src, size_t count, size_t offset,
                       hipMemcpyKind kind, bool isExec = false) {
    if (isExec) {
      size_t discardOffset = 0;
      amd::Memory *memObj = getMemoryObject(src, discardOffset);
      if (memObj != nullptr) {
        amd::Memory *memObjOri = getMemoryObject(src_, discardOffset);
        if (memObjOri != nullptr) {
          if (memObjOri->getUserData().deviceId != memObj->getUserData().deviceId) {
            return hipErrorInvalidValue;
          }
        }
      }
    }
    size_t sym_size = 0;
    hipDeviceptr_t device_ptr = nullptr;
    // check to see if src is also a symbol (hip negative test case)
    hipError_t status = ihipMemcpySymbol_validate(src, count, offset, sym_size, device_ptr);
    if (status == hipSuccess) {
      return hipErrorInvalidValue;
    }
    status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
    }
    size_t dOffset = 0;
    amd::Memory* srcMemory = getMemoryObject(src, dOffset);
    cl_mem_flags srcFlag = 0;
    if (srcMemory != nullptr) {
      srcFlag = srcMemory->getMemFlags();
      if (!IS_LINUX) {
        srcFlag &= ~ROCCLR_MEM_INTERPROCESS;
      }
    }
    if (srcMemory == nullptr && kind != hipMemcpyHostToDevice && kind != hipMemcpyDefault) {
      return hipErrorInvalidValue;
    } else if (srcMemory != nullptr && srcFlag == 0 &&
               kind != hipMemcpyDeviceToDevice && kind != hipMemcpyDeviceToDeviceNoCU
               && kind != hipMemcpyDefault) {
      return hipErrorInvalidValue;
    } else if (kind == hipMemcpyHostToHost || kind == hipMemcpyDeviceToHost) {
      return hipErrorInvalidValue;
    }
    symbol_ = symbol;
    src_ = src;
    count_ = count;
    offset_ = offset;
    kind_ = kind;
    return hipSuccess;
  }

  virtual hipError_t SetParams(GraphNode* node) override {
    const GraphMemcpyNodeToSymbol* memcpyNode =
        static_cast<GraphMemcpyNodeToSymbol const*>(node);
    return SetParams(memcpyNode->src_, memcpyNode->symbol_, memcpyNode->count_, memcpyNode->offset_,
                     memcpyNode->kind_);
  }
};
class GraphMemsetNode : public GraphNode {
  hipMemsetParams memsetParams_;
  size_t depth_ = 1;
 public:
  GraphMemsetNode(const hipMemsetParams* pMemsetParams, size_t depth = 1)
      : GraphNode(hipGraphNodeTypeMemset, "solid", "invtrapezium", "MEMSET") {
    memsetParams_ = *pMemsetParams;
    depth_ = depth;
    size_t sizeBytes = 0;
    if (memsetParams_.height == 1) {
      sizeBytes = memsetParams_.width * memsetParams_.elementSize;
    } else {
      sizeBytes = memsetParams_.width * memsetParams_.height * depth_ * memsetParams_.elementSize;
    }
  }

  ~GraphMemsetNode() { }
  // Copy constructor
  GraphMemsetNode(const GraphMemsetNode& memsetNode) : GraphNode(memsetNode) {
    memsetParams_ = memsetNode.memsetParams_;
    depth_ = memsetNode.depth_;
  }

  GraphNode* clone() const override {
    return new GraphMemsetNode(static_cast<GraphMemsetNode const&>(*this));
  }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    std::string label;
    if (flag == hipGraphDebugDotFlagsMemsetNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      char buffer[500];
      sprintf(buffer,
              "{\n%s\n| {{ID | node handle | dptr | pitch | value | elementSize | width | "
              "height | depth} | {%u | %p | %p | %zu | %u | %u | %zu | %zu | %zu}}}",
              label_.c_str(), GetID(), this, memsetParams_.dst, memsetParams_.pitch,
              memsetParams_.value, memsetParams_.elementSize, memsetParams_.width,
              memsetParams_.height, depth_);
      label = buffer;
    } else {
      size_t sizeBytes;
      if (memsetParams_.height == 1) {
        sizeBytes = memsetParams_.width * memsetParams_.elementSize;
      } else {
        sizeBytes = memsetParams_.width * memsetParams_.height * depth_ * memsetParams_.elementSize;
      }
      label = std::to_string(GetID()) + "\n" + label_ + "\n(" +
          std::to_string(memsetParams_.value) + "," + std::to_string(sizeBytes) + ")";
    }
    return label;
  }

  std::string GetShape(hipGraphDebugDotFlags flag) override {
    if (flag == hipGraphDebugDotFlagsMemsetNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    if (memsetParams_.height == 1) {
      size_t sizeBytes = memsetParams_.width * memsetParams_.elementSize;
      hipError_t status = ihipMemsetCommand(commands_, memsetParams_.dst, memsetParams_.value,
                                            memsetParams_.elementSize, sizeBytes, stream);
    } else {
      hipError_t status = ihipMemset3DCommand(
          commands_,
          {memsetParams_.dst, memsetParams_.pitch, memsetParams_.width * memsetParams_.elementSize,
           memsetParams_.height},
          memsetParams_.value,
          {memsetParams_.width * memsetParams_.elementSize, memsetParams_.height, depth_}, stream,
          memsetParams_.elementSize);
    }
    return status;
  }

  void GetParams(hipMemsetParams* params) {
    std::memcpy(params, &memsetParams_, sizeof(hipMemsetParams));
  }

  void GetParams(HIP_MEMSET_NODE_PARAMS* params) {
    params->dst = memsetParams_.dst;
    params->elementSize = memsetParams_.elementSize;
    params->height = memsetParams_.height;
    params->pitch = memsetParams_.pitch;
    params->value = memsetParams_.value;
    params->width = memsetParams_.width;
  }

  hipError_t SetParamsInternal(const hipMemsetParams* params, bool isExec, size_t depth = 1) {
    hipError_t hip_error = hipSuccess;
    hip_error = ihipGraphMemsetParams_validate(params);
    if (hip_error != hipSuccess) {
      return hip_error;
    }
    if (depth == 0) {
      return hipErrorInvalidValue;
    }
    if (isExec) {
      size_t discardOffset = 0;
      amd::Memory *memObj = getMemoryObject(params->dst, discardOffset);
      if (memObj != nullptr) {
        amd::Memory *memObjOri = getMemoryObject(memsetParams_.dst, discardOffset);
        if (memObjOri != nullptr) {
          if (memObjOri->getUserData().deviceId != memObj->getUserData().deviceId) {
            return hipErrorInvalidValue;
          }
        }
      }
    }
    size_t sizeBytes;
    if (params->height == 1) {
      // 1D - for hipGraphMemsetNodeSetParams & hipGraphExecMemsetNodeSetParams, They return
      // invalid value if new width is more than actual allocation.
      size_t discardOffset = 0;
      amd::Memory *memObj = getMemoryObject(params->dst, discardOffset);
      if (memObj != nullptr) {
        if (params->width * params->elementSize > memObj->getSize()) {
          return hipErrorInvalidValue;
        }
       }
      sizeBytes = params->width * params->elementSize;
      hip_error = ihipMemset_validate(params->dst, params->value, params->elementSize, sizeBytes);
    } else {
      if (isExec) {
        // 2D - hipGraphExecMemsetNodeSetParams returns invalid value if new width or new height is
        // not same as what memset node is added with.
        if (memsetParams_.width * memsetParams_.elementSize != params->width * params->elementSize
         || memsetParams_.height != params->height || depth != depth_) {
          return hipErrorInvalidValue;
        }
      } else {
        // 2D - hipGraphMemsetNodeSetParams returns invalid value if new width or new height is
        // greter than actual allocation.
        size_t discardOffset = 0;
        amd::Memory *memObj = getMemoryObject(params->dst, discardOffset);
        if (memObj != nullptr) {
          if (params->width * params->elementSize > memObj->getUserData().width_
           || params->height > memObj->getUserData().height_
           || depth > memObj->getUserData().depth_) {
            return hipErrorInvalidValue;
           }
        }
       }
      sizeBytes = params->width * params->elementSize * params->height * depth;
      hip_error = ihipMemset3D_validate(
          {params->dst, params->pitch, params->width * params->elementSize, params->height},
          params->value, {params->width * params->elementSize, params->height, depth}, sizeBytes);
    }
    if (hip_error != hipSuccess) {
      return hip_error;
    }
    std::memcpy(&memsetParams_, params, sizeof(hipMemsetParams));
    depth_ = depth;
    return hipSuccess;
  }

  hipError_t SetParams(const hipMemsetParams* params, bool isExec = false, size_t depth = 1) {
    return SetParamsInternal(params, isExec, depth);
  }

  hipError_t SetParams(const HIP_MEMSET_NODE_PARAMS* params, bool isExec = false, size_t depth = 1) {
    hipMemsetParams pmemsetParams;
    pmemsetParams.dst = params->dst;
    pmemsetParams.elementSize = params->elementSize;
    pmemsetParams.height = params->height;
    pmemsetParams.pitch = params->pitch;
    pmemsetParams.value = params->value;
    pmemsetParams.width = params->width;
    return SetParamsInternal(&pmemsetParams, isExec, depth);
  }
  hipError_t SetParams(GraphNode* node) override {
    const GraphMemsetNode* memsetNode = static_cast<GraphMemsetNode const*>(node);
    return SetParams(&memsetNode->memsetParams_, false, memsetNode->depth_);
  }
};

class GraphEventRecordNode : public GraphNode {
  hipEvent_t event_;

 public:
  GraphEventRecordNode(hipEvent_t event)
      : GraphNode(hipGraphNodeTypeEventRecord, "solid", "rectangle", "EVENT_RECORD"),
        event_(event) {}
  ~GraphEventRecordNode() {}

  GraphNode* clone() const override {
    return new GraphEventRecordNode(static_cast<GraphEventRecordNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    hip::Event* e = reinterpret_cast<hip::Event*>(event_);
    commands_.reserve(1);
    amd::Command* command = nullptr;
    status = e->recordCommand(command, stream);
    commands_.emplace_back(command);
    return status;
  }

  void EnqueueCommands(hip::Stream* stream) override {
    if (!commands_.empty()) {
      hip::Event* e = reinterpret_cast<hip::Event*>(event_);
      // command release during enqueueRecordCommand
      hipError_t status = e->enqueueRecordCommand(
            reinterpret_cast<hipStream_t>(stream), commands_[0], true);
      if (status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] Enqueue event record command failed for node %p - status %d", this,
                status);
      }
    }
  }

  void GetParams(hipEvent_t* event) const { *event = event_; }

  hipError_t SetParams(hipEvent_t event) {
    event_ = event;
    return hipSuccess;
  }

  hipError_t SetParams(GraphNode* node) override {
    const GraphEventRecordNode* eventRecordNode =
        static_cast<GraphEventRecordNode const*>(node);
    return SetParams(eventRecordNode->event_);
  }
};

class GraphEventWaitNode : public GraphNode {
  hipEvent_t event_;

 public:
  GraphEventWaitNode(hipEvent_t event)
      : GraphNode(hipGraphNodeTypeWaitEvent, "solid", "rectangle", "EVENT_WAIT"),
        event_(event) {}
  ~GraphEventWaitNode() {}

  GraphNode* clone() const override {
    return new GraphEventWaitNode(static_cast<GraphEventWaitNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    hip::Event* e = reinterpret_cast<hip::Event*>(event_);
    commands_.reserve(1);
    amd::Command* command;
    status = e->streamWaitCommand(command, stream);
    commands_.emplace_back(command);
    return status;
  }

  void EnqueueCommands(hip::Stream* stream) override {
    if (!commands_.empty()) {
      hip::Event* e = reinterpret_cast<hip::Event*>(event_);
      hipError_t status =
        e->enqueueStreamWaitCommand(reinterpret_cast<hipStream_t>(stream), commands_[0]);
      if (status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] Enqueue stream wait command failed for node %p - status %d", this,
                status);
      }
      commands_[0]->release();
    }
  }

  void GetParams(hipEvent_t* event) const { *event = event_; }

  hipError_t SetParams(hipEvent_t event) {
    event_ = event;
    return hipSuccess;
  }

  hipError_t SetParams(GraphNode* node) override {
    const GraphEventWaitNode* eventWaitNode = static_cast<GraphEventWaitNode const*>(node);
    return SetParams(eventWaitNode->event_);
  }
};

class GraphHostNode : public GraphNode {
  hipHostNodeParams NodeParams_;

 public:
  GraphHostNode(const hipHostNodeParams* NodeParams)
      : GraphNode(hipGraphNodeTypeHost, "solid", "rectangle", "HOST") {
    NodeParams_ = *NodeParams;
  }
  ~GraphHostNode() { }

  GraphHostNode(const GraphHostNode& hostNode) : GraphNode(hostNode) {
    NodeParams_ = hostNode.NodeParams_;
  }

  GraphNode* clone() const override {
    return new GraphHostNode(static_cast<GraphHostNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    amd::Command::EventWaitList waitList;
    commands_.reserve(1);
    amd::Command* command = new amd::Marker(*stream, !kMarkerDisableFlush, waitList);
    commands_.emplace_back(command);
    return hipSuccess;
  }

  static void Callback(cl_event event, cl_int command_exec_status, void* user_data) {
    hipHostNodeParams* NodeParams = reinterpret_cast<hipHostNodeParams*>(user_data);
    NodeParams->fn(NodeParams->userData);
  }

  void EnqueueCommands(hip::Stream* stream) override {
    if (!commands_.empty()) {
      if (!commands_[0]->setCallback(CL_COMPLETE, GraphHostNode::Callback, &NodeParams_)) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed during setCallback");
      }
      commands_[0]->enqueue();
      // Add the new barrier to stall the stream, until the callback is done
      amd::Command::EventWaitList eventWaitList;
      eventWaitList.push_back(commands_[0]);
      amd::Command* block_command =
          new amd::Marker(*commands_[0]->queue(), !kMarkerDisableFlush, eventWaitList);
      if (block_command == nullptr) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed during block command creation");
      }
      block_command->enqueue();
      block_command->release();
      commands_[0]->release();
    }
  }

  void GetParams(hipHostNodeParams* params) {
    std::memcpy(params, &NodeParams_, sizeof(hipHostNodeParams));
  }
  hipError_t SetParams(const hipHostNodeParams* params) {
    std::memcpy(&NodeParams_, params, sizeof(hipHostNodeParams));
    return hipSuccess;
  }

  hipError_t SetParams(GraphNode* node) override {
    const GraphHostNode* hostNode = static_cast<GraphHostNode const*>(node);
    return SetParams(&hostNode->NodeParams_);
  }
};

class GraphEmptyNode : public GraphNode {
 public:
  GraphEmptyNode() : GraphNode(hipGraphNodeTypeEmpty, "solid", "rectangle", "EMPTY") {}
  ~GraphEmptyNode() {}

  GraphNode* clone() const override {
    return new GraphEmptyNode(static_cast<GraphEmptyNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    amd::Command::EventWaitList waitList;
    commands_.reserve(1);
    amd::Command* command = new amd::Marker(*stream, !kMarkerDisableFlush, waitList);
    commands_.emplace_back(command);
    return hipSuccess;
  }
};

// ================================================================================================
class GraphMemAllocNode final : public GraphNode {
  hipMemAllocNodeParams node_params_;  // Node parameters for memory allocation
  amd::Memory* va_ = nullptr;         // Memory object, which holds a virtual address

  // Derive the new class for VirtualMapCommand,
  // so runtime can allocate memory during the execution of command
  class VirtualMemAllocNode : public amd::VirtualMapCommand {
   public:
    VirtualMemAllocNode(amd::HostQueue& queue, const amd::Event::EventWaitList& eventWaitList,
                        amd::Memory* va, size_t size, amd::Memory* memory, Graph* graph)
        : VirtualMapCommand(queue, eventWaitList, va->getSvmPtr(), size, memory),
          va_(va), graph_(graph) {}

    virtual void submit(device::VirtualDevice& device) final {
      // Remove VA reference from the global mapping. Runtime has to keep a dummy reference for
      // validation logic during the capture or creation of the nodes
      if (amd::MemObjMap::FindMemObj(va_->getSvmPtr())) {
        amd::MemObjMap::RemoveMemObj(va_->getSvmPtr());
      }
      // Allocate real memory for mapping
      const auto& dev_info = queue()->device().info();
      auto aligned_size = amd::alignUp(size_, dev_info.virtualMemAllocGranularity_);
      auto dptr = graph_->AllocateMemory(aligned_size, static_cast<hip::Stream*>(queue()), nullptr);
      if (dptr == nullptr) {
        setStatus(CL_INVALID_OPERATION);
        return;
      }
      size_t offset = 0;
      // Get memory object associated with the real allocation
      memory_ = getMemoryObject(dptr, offset);
      // Retain memory object because command release will release it
      memory_->retain();
      size_ = aligned_size;
      // Execute the original mapping command
      VirtualMapCommand::submit(device);
      amd::Memory* vaddr_sub_obj = amd::MemObjMap::FindMemObj(va_->getSvmPtr());
      assert(vaddr_sub_obj != nullptr);
      queue()->device().SetMemAccess(vaddr_sub_obj->getSvmPtr(), aligned_size,
                                     amd::Device::VmmAccess::kReadWrite);
      va_->retain();
      ClPrint(amd::LOG_INFO, amd::LOG_MEM_POOL,
              "Graph MemAlloc execute [%p-%p], %p", vaddr_sub_obj->getSvmPtr(),
              reinterpret_cast<char*>(vaddr_sub_obj->getSvmPtr()) + aligned_size, memory());
    }

   private:
    amd::Memory* va_;   // Memory object with the new virtual address for mapping
    Graph* graph_;  // Graph which allocates/maps memory
  };

 public:
  GraphMemAllocNode(const hipMemAllocNodeParams* node_params)
      : GraphNode(hipGraphNodeTypeMemAlloc, "solid", "rectangle", "MEM_ALLOC") {
    node_params_ = *node_params;
  }

  GraphMemAllocNode(const GraphMemAllocNode& rhs)
      : GraphNode(rhs) {
    node_params_ = rhs.node_params_;
    if (HIP_MEM_POOL_USE_VM) {
      assert(rhs.va_ != nullptr && "Graph MemAlloc runtime can't clone an invalid node!");
      va_ = rhs.va_;
      va_->retain();
    }
  }

  virtual ~GraphMemAllocNode() final {
    if (va_ != nullptr) {
      va_->release();
    }
  }

  virtual GraphNode* clone() const final {
    return new GraphMemAllocNode(static_cast<GraphMemAllocNode const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) final {
    auto error = GraphNode::CreateCommand(stream);
    if (!HIP_MEM_POOL_USE_VM) {
      auto ptr = Execute(stream_);
    } else {
      auto graph = GetParentGraph();
      if (graph != nullptr) {
        assert(va_ != nullptr && "Runtime can't create a command for an invalid node!");
        stream->GetDevice()->GetGraphMemoryPool()->SetGraphInUse();
        // Create command for memory mapping
        auto cmd = new VirtualMemAllocNode(*stream, amd::Event::EventWaitList{},
            va_, node_params_.bytesize, nullptr, graph);
        commands_.push_back(cmd);
        size_t offset = 0;
        // Check if memory was already added after first reserve
        if (getMemoryObject(node_params_.dptr, offset) == nullptr) {
          // Map VA in the accessible space because the graph execution still has
          // pointers validation and must find a valid object
          // @note: Memory can be released outside of the graph and
          // runtime can't keep a valid mapping since it doesn't know if the graph will
          // be executed again
          amd::MemObjMap::AddMemObj(node_params_.dptr, va_);
        }
        ClPrint(amd::LOG_INFO, amd::LOG_MEM_POOL, "Graph MemAlloc create: %p",
            node_params_.dptr);
      }
    }
    return error;
  }

  void* ReserveAddress() {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      node_params_.dptr = graph->ReserveAddress(node_params_.bytesize);
      if (node_params_.dptr != nullptr) {
        // Find VA and map in the accessible space so capture can find a valid object
        va_ = amd::MemObjMap::FindVirtualMemObj(node_params_.dptr);
        amd::MemObjMap::AddMemObj(node_params_.dptr, va_);
      }
      ClPrint(amd::LOG_INFO, amd::LOG_MEM_POOL, "Graph MemAlloc reserve VA: %p",
          node_params_.dptr);
    }
    return node_params_.dptr;
  }

  void* Execute(hip::Stream* stream = nullptr) {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      // The node creation requires to return a valid address, however FreeNode can't
      // free memory on creation because it doesn't have any execution point yet. Thus
      // the code below makes sure memory won't be recreated on the first execution of the graph
      if ((node_params_.dptr == nullptr) || !graph->ProbeMemory(node_params_.dptr)) {
        auto dptr = graph->AllocateMemory(node_params_.bytesize, stream, node_params_.dptr);
        if ((node_params_.dptr != nullptr) && (node_params_.dptr != dptr)) {
          LogPrintfError("Ptr mismatch in graph mem alloc %p != %p", node_params_.dptr, dptr);
        }
        node_params_.dptr = dptr;
      }
    }
    return node_params_.dptr;
  }

  bool IsActiveMem() {
    auto graph = GetParentGraph();
    return graph->ProbeMemory(node_params_.dptr);
  }

  void GetParams(hipMemAllocNodeParams* params) const {
    std::memcpy(params, &node_params_, sizeof(hipMemAllocNodeParams));
  }
};

// ================================================================================================
class GraphMemFreeNode : public GraphNode {
  void* device_ptr_;    // Device pointer of the freed memory

  // Derive the new class for VirtualMap command, since runtime has to free
  // real allocation after unmap is complete
  class VirtualMemFreeNode : public amd::VirtualMapCommand {
   public:
    VirtualMemFreeNode(Graph* graph, int device_id, amd::HostQueue& queue,
        const amd::Event::EventWaitList& eventWaitList, void* ptr, size_t size,
        amd::Memory* memory) : VirtualMapCommand(queue, eventWaitList, ptr, size, memory)
        , graph_(graph), device_id_(device_id) {}

    virtual void submit(device::VirtualDevice& device) final {
      // Find memory object before unmap logic
      auto vaddr_sub_obj = amd::MemObjMap::FindMemObj(ptr());
      assert(vaddr_sub_obj != nullptr);
      amd::Memory* phys_mem_obj = vaddr_sub_obj->getUserData().phys_mem_obj;
      assert(phys_mem_obj != nullptr);
      auto vaddr_mem_obj = amd::MemObjMap::FindVirtualMemObj(ptr());
      assert(vaddr_mem_obj != nullptr);
      VirtualMapCommand::submit(device);
      if (!AMD_DIRECT_DISPATCH) {
        // Update the current device, since hip event, used in mem pools, requires device
        hip::setCurrentDevice(device_id_);
      }
      // Free virtual address
      vaddr_mem_obj->release();
      // Release the allocation back to graph's pool
      graph_->FreeMemory(phys_mem_obj->getSvmPtr(), static_cast<hip::Stream*>(queue()));
      amd::MemObjMap::AddMemObj(ptr(), vaddr_mem_obj);
      ClPrint(amd::LOG_INFO, amd::LOG_MEM_POOL, "Graph MemFree execute: %p, %p",
          ptr(), vaddr_sub_obj);
    }

   private:
    Graph* graph_;  // Graph, which has the execution of this command
    int device_id_;     // Device ID where this command is executed
  };

 public:
  GraphMemFreeNode(void* dptr)
    : GraphNode(hipGraphNodeTypeMemFree, "solid", "rectangle", "MEM_FREE")
    , device_ptr_(dptr) {}
  GraphMemFreeNode(const GraphMemFreeNode& rhs) : GraphNode(rhs) {
    device_ptr_ = rhs.device_ptr_;
  }

  virtual GraphNode* clone() const final {
    return new GraphMemFreeNode(static_cast<GraphMemFreeNode const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) final {
    auto error = GraphNode::CreateCommand(stream);
    if (!HIP_MEM_POOL_USE_VM) {
      Execute(stream_);
    } else {
      auto graph = GetParentGraph();
      if (graph != nullptr) {
        const auto& dev_info = stream->device().info();
        auto va = amd::MemObjMap::FindVirtualMemObj(device_ptr_);
        // Unmap virtual address from memory
        amd::Command* cmd = new VirtualMemFreeNode(graph, stream->DeviceId(), *stream,
            amd::Command::EventWaitList{}, device_ptr_,
            amd::alignUp(va->getSize(), dev_info.virtualMemAllocGranularity_), nullptr);
        commands_.push_back(cmd);
        ClPrint(amd::LOG_INFO, amd::LOG_MEM_POOL, "Graph FreeMem create: %p", device_ptr_);
      }
    }
    return error;
  }

  void Execute(hip::Stream* stream) {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      graph->FreeMemory(device_ptr_, stream);
    }
  }

  void GetParams(void** params) const {
    *params = device_ptr_;
  }
};

class GraphDrvMemcpyNode : public GraphNode {
  HIP_MEMCPY3D copyParams_;

 public:
  GraphDrvMemcpyNode(const HIP_MEMCPY3D* pCopyParams)
      : GraphNode(hipGraphNodeTypeMemcpy, "solid", "trapezium", "MEMCPY") {
    copyParams_ = *pCopyParams;
  }
  ~GraphDrvMemcpyNode() {}

  GraphDrvMemcpyNode(const GraphDrvMemcpyNode& rhs) : GraphNode(rhs) {
    copyParams_ = rhs.copyParams_;
  }

  GraphNode* clone() const override {
    return new GraphDrvMemcpyNode(static_cast<GraphDrvMemcpyNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) override {
    if(copyParams_.srcMemoryType == hipMemoryTypeHost &&
       copyParams_.dstMemoryType == hipMemoryTypeHost &&
       IsHtoHMemcpy(copyParams_.dstHost, copyParams_.srcHost)) {
      return hipSuccess;
    }
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    status = ihipGetMemcpyParam3DCommand(command, &copyParams_, stream);
    commands_.emplace_back(command);
    return status;
  }

  void EnqueueCommands(hip::Stream* stream) override {
    bool isHtoH = false;
    if(copyParams_.srcMemoryType == hipMemoryTypeHost &&
       copyParams_.dstMemoryType == hipMemoryTypeHost &&
       IsHtoHMemcpy(copyParams_.dstHost, copyParams_.srcHost)) {
      isHtoH = true;
    }
    if (isEnabled_ && isHtoH) {
      ihipHtoHMemcpy(copyParams_.dstHost, copyParams_.srcHost,
                     copyParams_.WidthInBytes * copyParams_.Height *
                     copyParams_.Depth, *stream);
      return;
    }
    GraphNode::EnqueueCommands(stream);
  }

  void GetParams(HIP_MEMCPY3D* params) {
    std::memcpy(params, &copyParams_, sizeof(HIP_MEMCPY3D));
  }
  hipError_t SetParams(const HIP_MEMCPY3D* params) {
    hipError_t status = ValidateParams(params);
    if (status != hipSuccess) {
      return status;
    }
    std::memcpy(&copyParams_, params, sizeof(HIP_MEMCPY3D));
    return hipSuccess;
  }
  hipError_t SetParams(GraphNode* node) override {
    const GraphDrvMemcpyNode* memcpyNode = static_cast<GraphDrvMemcpyNode const*>(node);
    return SetParams(&memcpyNode->copyParams_);
  }
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t ValidateParams(const HIP_MEMCPY3D* pNodeParams) {
    hipError_t status = ihipDrvMemcpy3D_validate(pNodeParams);
    if (status != hipSuccess) {
      return status;
    }
    return hipSuccess;
  }

};

class hipGraphExternalSemSignalNode : public GraphNode {
  hipExternalSemaphoreSignalNodeParams externalSemaphorNodeParam_;

 public:
  hipGraphExternalSemSignalNode(const hipExternalSemaphoreSignalNodeParams* pNodeParams)
    : GraphNode(hipGraphNodeTypeExtSemaphoreSignal, "solid", "rectangle",
                   "EXTERNAL_SEMAPHORE_SIGNAL") {
        externalSemaphorNodeParam_ = *pNodeParams;
  }

  hipGraphExternalSemSignalNode(const hipGraphExternalSemSignalNode& rhs)
    : GraphNode(rhs) {
    externalSemaphorNodeParam_ = rhs.externalSemaphorNodeParam_;
  }

  ~hipGraphExternalSemSignalNode() {}

  GraphNode* clone() const {
    return new hipGraphExternalSemSignalNode(
                static_cast<hipGraphExternalSemSignalNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    unsigned int numExtSems = externalSemaphorNodeParam_.numExtSems;
    commands_.reserve(numExtSems);
    for (unsigned int i = 0; i < numExtSems; i++) {
      if (externalSemaphorNodeParam_.extSemArray[i] != nullptr) {
        amd::ExternalSemaphoreCmd* command = new amd::ExternalSemaphoreCmd(*stream,
                                      externalSemaphorNodeParam_.extSemArray[i],
                                      externalSemaphorNodeParam_.paramsArray[i].params.fence.value,
                                      amd::ExternalSemaphoreCmd::COMMAND_SIGNAL_EXTSEMAPHORE);
        if (command == nullptr) {
          return hipErrorOutOfMemory;
        }
        commands_.emplace_back(command);
      } else {
        return hipErrorInvalidValue;
      }
    }
    return hipSuccess;
  }

  void GetParams(hipExternalSemaphoreSignalNodeParams* pNodeParams) const {
    std::memcpy(pNodeParams, &externalSemaphorNodeParam_,
                sizeof(hipExternalSemaphoreSignalNodeParams));
  }

  hipError_t SetParams(const hipExternalSemaphoreSignalNodeParams* pNodeParams) {
    std::memcpy(&externalSemaphorNodeParam_, pNodeParams,
                sizeof(hipExternalSemaphoreSignalNodeParams));
    return hipSuccess;
  }
};

class hipGraphExternalSemWaitNode : public GraphNode {
  hipExternalSemaphoreWaitNodeParams externalSemaphorNodeParam_;

 public:
  hipGraphExternalSemWaitNode(const hipExternalSemaphoreWaitNodeParams* pNodeParams)
    : GraphNode(hipGraphNodeTypeExtSemaphoreWait, "solid",
                   "rectangle", "EXTERNAL_SEMAPHORE_WAIT") {
        externalSemaphorNodeParam_ = *pNodeParams;
  }

  hipGraphExternalSemWaitNode(const hipGraphExternalSemWaitNode& rhs) : GraphNode(rhs) {
    externalSemaphorNodeParam_ = rhs.externalSemaphorNodeParam_;
  }
  ~hipGraphExternalSemWaitNode() {}

  GraphNode* clone() const {
    return new hipGraphExternalSemWaitNode(static_cast<hipGraphExternalSemWaitNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;

    }
    unsigned int numExtSems = externalSemaphorNodeParam_.numExtSems;
    commands_.reserve(numExtSems);
    for (unsigned int i = 0; i < numExtSems; i++) {
      if (externalSemaphorNodeParam_.extSemArray[i] != nullptr) {
        amd::ExternalSemaphoreCmd* command = new amd::ExternalSemaphoreCmd(*stream,
                                    externalSemaphorNodeParam_.extSemArray[i],
                                    externalSemaphorNodeParam_.paramsArray[i].params.fence.value,
                                    amd::ExternalSemaphoreCmd::COMMAND_WAIT_EXTSEMAPHORE);
        if (command == nullptr) {
          return hipErrorOutOfMemory;
        }
        commands_.emplace_back(command);
      } else {
        return hipErrorInvalidValue;
      }
    }
    return hipSuccess;
  }

  void GetParams(hipExternalSemaphoreWaitNodeParams* pNodeParams) const {
    std::memcpy(pNodeParams, &externalSemaphorNodeParam_,
                sizeof(hipExternalSemaphoreWaitNodeParams));
  }

  hipError_t SetParams(const hipExternalSemaphoreWaitNodeParams* pNodeParams) {
    std::memcpy(&externalSemaphorNodeParam_, pNodeParams,
                sizeof(hipExternalSemaphoreWaitNodeParams));
    return hipSuccess;
  }
};

}  // namespace hip
