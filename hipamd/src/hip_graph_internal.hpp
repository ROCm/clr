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

typedef hipGraphNode* Node;
hipError_t FillCommands(std::vector<std::vector<Node>>& parallelLists,
                        std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
                        std::vector<Node>& levelOrder, std::vector<amd::Command*>& rootCommands,
                        amd::Command*& endCommand, hip::Stream* stream);
void UpdateStream(std::vector<std::vector<Node>>& parallelLists, hip::Stream* stream,
                  hipGraphExec* ptr);

struct hipUserObject : public amd::ReferenceCountedObject {
  typedef void (*UserCallbackDestructor)(void* data);
  static std::unordered_set<hipUserObject*> ObjectSet_;
  static amd::Monitor UserObjectLock_;

 public:
  hipUserObject(UserCallbackDestructor callback, void* data, unsigned int flags)
      : ReferenceCountedObject(), callback_(callback), data_(data), flags_(flags) {
    amd::ScopedLock lock(UserObjectLock_);
    ObjectSet_.insert(this);
  }

  virtual ~hipUserObject() {
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

  static bool isUserObjvalid(hipUserObject* pUsertObj) {
    auto it = ObjectSet_.find(pUsertObj);
    if (it == ObjectSet_.end()) {
      return false;
    }
    return true;
  }

  static void removeUSerObj(hipUserObject* pUsertObj) {
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
  hipUserObject& operator=(const hipUserObject&) = delete;
  //! Disable copy constructor
  hipUserObject(const hipUserObject& obj) = delete;
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

struct hipGraphNode : public hipGraphNodeDOTAttribute {
 protected:
  hip::Stream* stream_ = nullptr;
  uint32_t level_;
  unsigned int id_;
  hipGraphNodeType type_;
  std::vector<amd::Command*> commands_;
  std::vector<Node> edges_;
  std::vector<Node> dependencies_;
  bool visited_;
  // count of in coming edges
  size_t inDegree_;
  // count of outgoing edges
  size_t outDegree_;
  static int nextID;
  struct ihipGraph* parentGraph_;
  static std::unordered_set<hipGraphNode*> nodeSet_;
  static amd::Monitor nodeSetLock_;
  unsigned int isEnabled_;

 public:
  hipGraphNode(hipGraphNodeType type, std::string style = "", std::string shape = "",
               std::string label = "")
      : type_(type),
        level_(0),
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
  hipGraphNode(const hipGraphNode& node) : hipGraphNodeDOTAttribute(node) {
    level_ = node.level_;
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

  virtual ~hipGraphNode() {
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
  static bool isNodeValid(hipGraphNode* pGraphNode) {
    amd::ScopedLock lock(nodeSetLock_);
    if (pGraphNode == nullptr || nodeSet_.find(pGraphNode) == nodeSet_.end()) {
      return false;
    }
    return true;
  }

  hip::Stream* GetQueue() { return stream_; }

  virtual void SetStream(hip::Stream* stream, hipGraphExec* ptr = nullptr) {
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
  /// Returns graph node in coming edges
  uint32_t GetLevel() const { return level_; }
  /// Set graph node level
  void SetLevel(uint32_t level) { level_ = level; }
  /// Clone graph node
  virtual hipGraphNode* clone() const = 0;
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
  /// Update level, for existing edges
  void UpdateEdgeLevel() {
    for (auto edge : edges_) {
      edge->SetLevel(std::max(edge->GetLevel(), GetLevel() + 1));
      edge->UpdateEdgeLevel();
    }
  }
  void ReduceEdgeLevel() {
    for (auto edge: edges_) {
      edge->SetLevel(std::min(edge->GetLevel(),GetLevel() + 1));
      edge->ReduceEdgeLevel();
    }
  }
  /// Add edge, update parent node outdegree, child node indegree, level and dependency
  void AddEdge(const Node& childNode) {
    edges_.push_back(childNode);
    outDegree_++;
    childNode->SetInDegree(childNode->GetInDegree() + 1);
    childNode->SetLevel(std::max(childNode->GetLevel(), GetLevel() + 1));
    childNode->UpdateEdgeLevel();
    childNode->AddDependency(this);
  }
  /// Remove edge, update parent node outdegree, child node indegree, level and dependency
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
    const std::vector<Node>& dependencies = childNode->GetDependencies();
    int32_t level = 0;
    int32_t parentLevel = 0;
    uint32_t origLevel = 0;
    for (auto parent : dependencies) {
      parentLevel = parent->GetLevel();
      level = std::max(level, (parentLevel + 1));
    }
    origLevel = childNode->GetLevel();
    childNode->SetLevel(level);
    if (level < origLevel) {
      childNode->ReduceEdgeLevel();
    }
    return true;
  }
  /// Get Runlist of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual void GetRunList(std::vector<std::vector<Node>>& parallelList,
                          std::unordered_map<Node, std::vector<Node>>& dependencies) {}
  /// Get levelorder of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual void LevelOrder(std::vector<Node>& levelOrder) {}
  /// Update waitlist of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual void UpdateEventWaitLists(amd::Command::EventWaitList waitList) {
    for (auto command : commands_) {
      command->updateEventWaitList(waitList);
    }
  }
  virtual size_t GetNumParallelStreams() { return 0; }
  /// Enqueue commands part of the node
  virtual void EnqueueCommands(hipStream_t stream) {
    // If the node is disabled it becomes empty node. To maintain ordering just enqueue marker.
    // Node can be enabled/disabled only for kernel, memcpy and memset nodes.
    if (!isEnabled_ &&
        (type_ == hipGraphNodeTypeKernel || type_ == hipGraphNodeTypeMemcpy ||
         type_ == hipGraphNodeTypeMemset)) {
      amd::Command::EventWaitList waitList;
      hip::Stream* hip_stream = hip::getStream(stream);
      amd::Command* command = new amd::Marker(*hip_stream, !kMarkerDisableFlush, waitList);
      command->enqueue();
      command->release();
      return;
    }
    for (auto& command : commands_) {
      command->enqueue();
      command->release();
    }
  }
  ihipGraph* GetParentGraph() { return parentGraph_; }
  virtual ihipGraph* GetChildGraph() { return nullptr; }
  void SetParentGraph(ihipGraph* graph) { parentGraph_ = graph; }
  virtual hipError_t SetParams(hipGraphNode* node) { return hipSuccess; }
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
  virtual std::string GetLabel(hipGraphDebugDotFlags flag) { return (std::to_string(id_) + "\n" + label_); }
  unsigned int GetEnabled() const { return isEnabled_; }
  void SetEnabled(unsigned int isEnabled) { isEnabled_ = isEnabled; }
};

struct ihipGraph {
  std::vector<Node> vertices_;
  const ihipGraph* pOriginalGraph_ = nullptr;
  static std::unordered_set<ihipGraph*> graphSet_;
  static amd::Monitor graphSetLock_;
  std::unordered_set<hipUserObject*> graphUserObj_;
  unsigned int id_;
  static int nextID;
  hip::Device* device_;       //!< HIP device object
  hip::MemoryPool* mem_pool_; //!< Memory pool, associated with this graph
  std::unordered_set<hipGraphNode*> capturedNodes_;
  bool graphInstantiated_;

 public:
  ihipGraph(hip::Device* device, const ihipGraph* original = nullptr)
      : pOriginalGraph_(original)
      , id_(nextID++)
      , device_(device) {
    amd::ScopedLock lock(graphSetLock_);
    graphSet_.insert(this);
    mem_pool_ = device->GetGraphMemoryPool();
    mem_pool_->retain();
    graphInstantiated_ = false;
  }

  ~ihipGraph() {
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

  void AddManualNodeDuringCapture(hipGraphNode* node) { capturedNodes_.insert(node); }

  std::unordered_set<hipGraphNode*> GetManualNodesDuringCapture() { return capturedNodes_; }

  void RemoveManualNodesDuringCapture() {
    capturedNodes_.erase(capturedNodes_.begin(), capturedNodes_.end());
  }

  /// Return graph unique ID
  int GetID() const { return id_; }

  // check graphs validity
  static bool isGraphValid(ihipGraph* pGraph);

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
  const ihipGraph* getOriginalGraph() const { return pOriginalGraph_; }
  // Add user obj resource to graph
  void addUserObjGraph(hipUserObject* pUserObj) {
    amd::ScopedLock lock(graphSetLock_);
    graphUserObj_.insert(pUserObj);
  }
  // Check user obj resource from graph is valid
  bool isUserObjGraphValid(hipUserObject* pUserObj) {
    if (graphUserObj_.find(pUserObj) == graphUserObj_.end()) {
      return false;
    }
    return true;
  }
  // Delete user obj resource from graph
  void RemoveUserObjGraph(hipUserObject* pUserObj) { graphUserObj_.erase(pUserObj); }

  void GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                      std::vector<Node>& singleList, std::vector<std::vector<Node>>& parallelLists,
                      std::unordered_map<Node, std::vector<Node>>& dependencies);
  void GetRunList(std::vector<std::vector<Node>>& parallelLists,
                  std::unordered_map<Node, std::vector<Node>>& dependencies);
  void LevelOrder(std::vector<Node>& levelOrder);
  void GetUserObjs(std::unordered_set<hipUserObject*>& graphExeUserObjs) {
    for (auto userObj : graphUserObj_) {
      userObj->retain();
      graphExeUserObjs.insert(userObj);
    }
  }
  ihipGraph* clone(std::unordered_map<Node, Node>& clonedNodes) const;
  ihipGraph* clone() const;
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

  void FreeAllMemory() {
    mem_pool_->FreeAllMemory();
  }

  bool IsGraphInstantiated() const {
    return graphInstantiated_;
  }

  void SetGraphInstantiated(bool graphInstantiate) {
    graphInstantiated_ = graphInstantiate;
  }
};

struct hipGraphExec {
  std::vector<std::vector<Node>> parallelLists_;
  // level order of the graph doesn't include nodes embedded as part of the child graph
  std::vector<Node> levelOrder_;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists_;
  std::vector<hip::Stream*> parallel_streams_;
  uint currentQueueIndex_;
  std::unordered_map<Node, Node> clonedNodes_;
  amd::Command* lastEnqueuedCommand_;
  static std::unordered_set<hipGraphExec*> graphExecSet_;
  std::unordered_set<hipUserObject*> graphExeUserObj_;
  static amd::Monitor graphExecSetLock_;
  uint64_t flags_ = 0;
  bool repeatLaunch_ = false;
 public:
  hipGraphExec(std::vector<Node>& levelOrder, std::vector<std::vector<Node>>& lists,
               std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
               std::unordered_map<Node, Node>& clonedNodes,
               std::unordered_set<hipUserObject*>& userObjs,
               uint64_t flags = 0)
      : parallelLists_(lists),
        levelOrder_(levelOrder),
        nodeWaitLists_(nodeWaitLists),
        clonedNodes_(clonedNodes),
        lastEnqueuedCommand_(nullptr),
        graphExeUserObj_(userObjs),
        currentQueueIndex_(0),
        flags_(flags) {
    amd::ScopedLock lock(graphExecSetLock_);
    graphExecSet_.insert(this);
  }

  ~hipGraphExec() {
    // new commands are launched for every launch they are destroyed as and when command is
    // terminated after it complete execution
    for (auto stream : parallel_streams_) {
      if (stream != nullptr) {
        stream->release();
      }
    }
    for (auto it = clonedNodes_.begin(); it != clonedNodes_.end(); it++) delete it->second;
    amd::ScopedLock lock(graphExecSetLock_);
    for (auto userobj : graphExeUserObj_) {
      userobj->release();
    }
    graphExecSet_.erase(this);
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

  // check executable graphs validity
  static bool isGraphExecValid(hipGraphExec* pGraphExec);

  std::vector<Node>& GetNodes() { return levelOrder_; }

  hip::Stream* GetAvailableStreams() { return parallel_streams_[currentQueueIndex_++]; }
  void ResetQueueIndex() { currentQueueIndex_ = 0; }
  hipError_t Init();
  hipError_t CreateStreams(uint32_t num_streams);
  hipError_t Run(hipStream_t stream);
};

struct hipChildGraphNode : public hipGraphNode {
  struct ihipGraph* childGraph_;
  std::vector<Node> childGraphlevelOrder_;
  std::vector<std::vector<Node>> parallelLists_;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists_;
  amd::Command* lastEnqueuedCommand_;

 public:
  hipChildGraphNode(ihipGraph* g) : hipGraphNode(hipGraphNodeTypeGraph, "solid", "rectangle") {
    childGraph_ = g->clone();
    lastEnqueuedCommand_ = nullptr;
  }

  ~hipChildGraphNode() { delete childGraph_; }

  hipChildGraphNode(const hipChildGraphNode& rhs) : hipGraphNode(rhs) {
    childGraph_ = rhs.childGraph_->clone();
  }

  hipGraphNode* clone() const {
    return new hipChildGraphNode(static_cast<hipChildGraphNode const&>(*this));
  }

  ihipGraph* GetChildGraph() { return childGraph_; }

  size_t GetNumParallelStreams() {
    LevelOrder(childGraphlevelOrder_);
    size_t num = 0;
    for (auto& node : childGraphlevelOrder_) {
      num += node->GetNumParallelStreams();
    }
    // returns total number of parallel queues required for child graph nodes to be launched
    // first parallel list will be launched on the same queue as parent
    return num + (parallelLists_.size() - 1);
  }

  void SetStream(hip::Stream* stream, hipGraphExec* ptr = nullptr) {
    stream_ = stream;
    UpdateStream(parallelLists_, stream, ptr);
  }

  // For nodes that are dependent on the child graph node waitlist is the last node of the first
  // parallel list
  std::vector<amd::Command*>& GetCommands() { return parallelLists_[0].back()->GetCommands(); }

  // Create child graph node commands and set waitlists
  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(2);
    std::vector<amd::Command*> rootCommands;
    amd::Command* endCommand = nullptr;
    status = FillCommands(parallelLists_, nodeWaitLists_, childGraphlevelOrder_, rootCommands,
                          endCommand, stream);
    for (auto& cmd : rootCommands) {
      commands_.push_back(cmd);
    }
    if (endCommand != nullptr) {
      commands_.push_back(endCommand);
    }
    return status;
  }

  //
  void UpdateEventWaitLists(amd::Command::EventWaitList waitList) {
    parallelLists_[0].front()->UpdateEventWaitLists(waitList);
  }

  void GetRunList(std::vector<std::vector<Node>>& parallelList,
                  std::unordered_map<Node, std::vector<Node>>& dependencies) {
    childGraph_->GetRunList(parallelLists_, nodeWaitLists_);
  }

  void LevelOrder(std::vector<Node>& levelOrder) { childGraph_->LevelOrder(levelOrder); }

  void EnqueueCommands(hipStream_t stream) {
    // enqueue child graph start command
    if (commands_.size() == 1) {
      commands_[0]->enqueue();
      commands_[0]->release();
    }
    // enqueue nodes in child graph in level order
    for (auto& node : childGraphlevelOrder_) {
      node->EnqueueCommands(stream);
    }
    // enqueue child graph end command
    if (commands_.size() == 2) {
      commands_[1]->enqueue();
      commands_[1]->release();
    }
  }

  hipError_t SetParams(const ihipGraph* childGraph) {
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

  hipError_t SetParams(hipGraphNode* node) {
    const hipChildGraphNode* childGraphNode = static_cast<hipChildGraphNode const*>(node);
    return SetParams(childGraphNode->childGraph_);
  }

  std::string GetLabel(hipGraphDebugDotFlags flag) {
    return std::to_string(GetID()) + "\n" + "graph_" + std::to_string(childGraph_->GetID());
  }

  virtual void GenerateDOT(std::ostream& fout, hipGraphDebugDotFlags flag) {
    childGraph_->GenerateDOT(fout, flag);
  }
};

class hipGraphKernelNode : public hipGraphNode {
  hipKernelNodeParams* pKernelParams_;
  unsigned int numParams_;
  hipKernelNodeAttrValue kernelAttr_;
  unsigned int kernelAttrInUse_;

 public:
    void PrintAttributes(std::ostream& out, hipGraphDebugDotFlags flag) {
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

  std::string GetLabel(hipGraphDebugDotFlags flag) {
    hipFunction_t func = getFunc(*pKernelParams_, ihipGetDevice());
    hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
    std::string label;
    char buffer[500];
    if (flag == hipGraphDebugDotFlagsVerbose) {
      sprintf(buffer,
              "{\n%s\n| {ID | %d | %s\\<\\<\\<(%u,%u,%u),(%u,%u,%u),%u\\>\\>\\>}\n| {{node "
              "handle | func handle} | {%p | %p}}\n| {accessPolicyWindow | {base_ptr | num_bytes | "
              "hitRatio | hitProp | missProp} | {%p | %ld | %f | %d | %d}}\n| {cooperative | "
              "%u}\n| {priority | 0}\n}",
              label_.c_str(), GetID(), function->name().c_str(), pKernelParams_->gridDim.x,
              pKernelParams_->gridDim.y, pKernelParams_->gridDim.z, pKernelParams_->blockDim.x,
              pKernelParams_->blockDim.y, pKernelParams_->blockDim.z,
              pKernelParams_->sharedMemBytes, this, pKernelParams_->func,
              kernelAttr_.accessPolicyWindow.base_ptr, kernelAttr_.accessPolicyWindow.num_bytes,
              kernelAttr_.accessPolicyWindow.hitRatio, kernelAttr_.accessPolicyWindow.hitProp,
              kernelAttr_.accessPolicyWindow.missProp, kernelAttr_.cooperative);
      label = buffer;
    }
    else if (flag == hipGraphDebugDotFlagsKernelNodeAttributes) {
      sprintf(buffer,
              "{\n%s\n| {ID | %d | %s}\n"
              "| {accessPolicyWindow | {base_ptr | num_bytes | "
              "hitRatio | hitProp | missProp} |\n| {%p | %ld | %f | %d | %d}}\n| {cooperative | "
              "%u}\n| {priority | 0}\n}",
              label_.c_str(), GetID(), function->name().c_str(),
              kernelAttr_.accessPolicyWindow.base_ptr, kernelAttr_.accessPolicyWindow.num_bytes,
              kernelAttr_.accessPolicyWindow.hitRatio, kernelAttr_.accessPolicyWindow.hitProp,
              kernelAttr_.accessPolicyWindow.missProp, kernelAttr_.cooperative);
      label = buffer;
    }
    else if (flag == hipGraphDebugDotFlagsKernelNodeParams) {
      sprintf(buffer, "%d\n%s\n\\<\\<\\<(%u,%u,%u),(%u,%u,%u),%u\\>\\>\\>",
              GetID(), function->name().c_str(), pKernelParams_->gridDim.x,
              pKernelParams_->gridDim.y, pKernelParams_->gridDim.z,
              pKernelParams_->blockDim.x, pKernelParams_->blockDim.y,
              pKernelParams_->blockDim.z, pKernelParams_->sharedMemBytes);
      label = buffer;
    }
    else {
      label = std::to_string(GetID()) + "\n" + function->name() + "\n";
    }
    return label;
  }

  std::string GetShape(hipGraphDebugDotFlags flag) {
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
      ClPrint(amd::LOG_INFO, amd::LOG_CODE,
              "[hipGraph] capturehipExtModuleLaunchKernel() should be called", status);
    } else if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] getStatFunc() failed with err %d", status);
    }
    return func;
  }

  hipError_t copyParams(const hipKernelNodeParams* pNodeParams) {
    hipFunction_t func = getFunc(*pNodeParams, ihipGetDevice());
    if (!func) {
      return hipErrorInvalidDeviceFunction;
    }
    hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
    amd::Kernel* kernel = function->kernel();
    const amd::KernelSignature& signature = kernel->signature();
    numParams_ = signature.numParameters();

    // Allocate/assign memory if params are passed part of 'kernelParams'
    if (pNodeParams->kernelParams != nullptr) {
      pKernelParams_->kernelParams = (void**)malloc(numParams_ * sizeof(void*));
      if (pKernelParams_->kernelParams == nullptr) {
        return hipErrorOutOfMemory;
      }

      for (uint32_t i = 0; i < numParams_; ++i) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        pKernelParams_->kernelParams[i] = malloc(desc.size_);
        if (pKernelParams_->kernelParams[i] == nullptr) {
          return hipErrorOutOfMemory;
        }
        ::memcpy(pKernelParams_->kernelParams[i], (pNodeParams->kernelParams[i]), desc.size_);
      }
    }

    // Allocate/assign memory if params are passed as part of 'extra'
    else if (pNodeParams->extra != nullptr) {
      // 'extra' is a struct that contains the following info: {
      // HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
      // HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargs_size,
      // HIP_LAUNCH_PARAM_END }
      unsigned int numExtra = 5;
      pKernelParams_->extra = (void**)malloc(numExtra * sizeof(void*));
      if (pKernelParams_->extra == nullptr) {
        return hipErrorOutOfMemory;
      }
      pKernelParams_->extra[0] = pNodeParams->extra[0];
      size_t kernargs_size = *((size_t*)pNodeParams->extra[3]);
      pKernelParams_->extra[1] = malloc(kernargs_size);
      if (pKernelParams_->extra[1] == nullptr) {
        return hipErrorOutOfMemory;
      }
      pKernelParams_->extra[2] = pNodeParams->extra[2];
      pKernelParams_->extra[3] = malloc(sizeof(void*));
      if (pKernelParams_->extra[3] == nullptr) {
        return hipErrorOutOfMemory;
      }
      *((size_t*)pKernelParams_->extra[3]) = kernargs_size;
      ::memcpy(pKernelParams_->extra[1], (pNodeParams->extra[1]), kernargs_size);
      pKernelParams_->extra[4] = pNodeParams->extra[4];
    }
    return hipSuccess;
  }

  hipGraphKernelNode(const hipKernelNodeParams* pNodeParams)
      : hipGraphNode(hipGraphNodeTypeKernel, "bold", "octagon", "KERNEL") {
    pKernelParams_ = new hipKernelNodeParams(*pNodeParams);
    if (copyParams(pNodeParams) != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to copy params");
    }
    memset(&kernelAttr_, 0, sizeof(kernelAttr_));
    kernelAttrInUse_ = 0;
  }

  ~hipGraphKernelNode() { freeParams(); }

  void freeParams() {
    // Deallocate memory allocated for kernargs passed via 'kernelParams'
    if (pKernelParams_->kernelParams != nullptr) {
      for (size_t i = 0; i < numParams_; ++i) {
        if (pKernelParams_->kernelParams[i] != nullptr) {
          free(pKernelParams_->kernelParams[i]);
        }
        pKernelParams_->kernelParams[i] = nullptr;
      }
      free(pKernelParams_->kernelParams);
      pKernelParams_->kernelParams = nullptr;
    }
    // Deallocate memory allocated for kernargs passed via 'extra'
    else {
      free(pKernelParams_->extra[1]);
      free(pKernelParams_->extra[3]);
      memset(pKernelParams_->extra, 0, 5 * sizeof(pKernelParams_->extra[0]));  // 5 items
      free(pKernelParams_->extra);
      pKernelParams_->extra = nullptr;
    }
    delete pKernelParams_;
    pKernelParams_ = nullptr;
  }

  hipGraphKernelNode(const hipGraphKernelNode& rhs) : hipGraphNode(rhs) {
    pKernelParams_ = new hipKernelNodeParams(*rhs.pKernelParams_);
    hipError_t status = copyParams(rhs.pKernelParams_);
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

  hipGraphNode* clone() const {
    return new hipGraphKernelNode(static_cast<hipGraphKernelNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipFunction_t func = nullptr;
    hipError_t status = validateKernelParams(pKernelParams_, &func,
                                             stream ? hip::getDeviceID(stream->context()) : -1);
    if (hipSuccess != status) {
      return status;
    }
    status = hipGraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    status = ihipLaunchKernelCommand(
        command, func, pKernelParams_->gridDim.x * pKernelParams_->blockDim.x,
        pKernelParams_->gridDim.y * pKernelParams_->blockDim.y,
        pKernelParams_->gridDim.z * pKernelParams_->blockDim.z, pKernelParams_->blockDim.x,
        pKernelParams_->blockDim.y, pKernelParams_->blockDim.z, pKernelParams_->sharedMemBytes,
        stream, pKernelParams_->kernelParams, pKernelParams_->extra, nullptr, nullptr, 0, 0, 0, 0, 0,
        0, 0);
    commands_.emplace_back(command);
    return status;
  }

  void GetParams(hipKernelNodeParams* params) { *params = *pKernelParams_; }

  hipError_t SetParams(const hipKernelNodeParams* params) {
    // updates kernel params
    hipError_t status = validateKernelParams(params);
    if (hipSuccess != status) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to validateKernelParams");
      return status;
    }
    if (pKernelParams_ &&
        ((pKernelParams_->kernelParams && pKernelParams_->kernelParams == params->kernelParams) ||
         (pKernelParams_->extra && pKernelParams_->extra == params->extra))) {
      // params is copied from pKernelParams_ and then updated, so just copy it back
      *pKernelParams_ = *params;
      return status;
    }
    freeParams();
    pKernelParams_ = new hipKernelNodeParams(*params);
    status = copyParams(params);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to set params");
    }
    return status;
  }

  hipError_t SetAttrParams(hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* params) {
    // updates kernel attr params
    if (attr == hipKernelNodeAttributeAccessPolicyWindow) {
      if (params->accessPolicyWindow.hitRatio > 1) {
        return hipErrorInvalidValue;
      }
      if (params->accessPolicyWindow.missProp == hipAccessPropertyPersisting) {
        return hipErrorInvalidValue;
      }
      if (params->accessPolicyWindow.num_bytes > 0 && params->accessPolicyWindow.hitRatio == 0) {
        return hipErrorInvalidValue;
      }
      kernelAttr_.accessPolicyWindow.base_ptr = params->accessPolicyWindow.base_ptr;
      kernelAttr_.accessPolicyWindow.hitProp = params->accessPolicyWindow.hitProp;
      kernelAttr_.accessPolicyWindow.hitRatio = params->accessPolicyWindow.hitRatio;
      kernelAttr_.accessPolicyWindow.missProp = params->accessPolicyWindow.missProp;
      kernelAttr_.accessPolicyWindow.num_bytes = params->accessPolicyWindow.num_bytes;
    } else if (attr == hipKernelNodeAttributeCooperative) {
      kernelAttr_.cooperative = params->cooperative;
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
    }
    return hipSuccess;
  }
  hipError_t CopyAttr(const hipGraphKernelNode* srcNode) {
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
      default:
        return hipErrorInvalidValue;
    }
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphKernelNode* kernelNode = static_cast<hipGraphKernelNode const*>(node);
    return SetParams(kernelNode->pKernelParams_);
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

class hipGraphMemcpyNode : public hipGraphNode {
  hipMemcpy3DParms* pCopyParams_;

 public:
  hipGraphMemcpyNode(const hipMemcpy3DParms* pCopyParams)
      : hipGraphNode(hipGraphNodeTypeMemcpy, "solid", "trapezium", "MEMCPY") {
    pCopyParams_ = new hipMemcpy3DParms(*pCopyParams);
  }
  ~hipGraphMemcpyNode() { delete pCopyParams_; }

  hipGraphMemcpyNode(const hipGraphMemcpyNode& rhs) : hipGraphNode(rhs) {
    pCopyParams_ = new hipMemcpy3DParms(*rhs.pCopyParams_);
  }

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNode(static_cast<hipGraphMemcpyNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    if (IsHtoHMemcpy(pCopyParams_->dstPtr.ptr, pCopyParams_->srcPtr.ptr, pCopyParams_->kind)) {
      return hipSuccess;
    }
    hipError_t status = hipGraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    status = ihipMemcpy3DCommand(command, pCopyParams_, stream);
    commands_.emplace_back(command);
    return status;
  }

  void EnqueueCommands(hipStream_t stream) override {
    if (isEnabled_ && IsHtoHMemcpy(pCopyParams_->dstPtr.ptr, pCopyParams_->srcPtr.ptr, pCopyParams_->kind)) {
      ihipHtoHMemcpy(pCopyParams_->dstPtr.ptr, pCopyParams_->srcPtr.ptr,
                     pCopyParams_->extent.width * pCopyParams_->extent.height *
                     pCopyParams_->extent.depth, *hip::getStream(stream));
      return;
    }
    hipGraphNode::EnqueueCommands(stream);
  }

  void GetParams(hipMemcpy3DParms* params) {
    std::memcpy(params, pCopyParams_, sizeof(hipMemcpy3DParms));
  }
  hipError_t SetParams(const hipMemcpy3DParms* params) {
    hipError_t status = ValidateParams(params);
    if (status != hipSuccess) {
      return status;
    }
    std::memcpy(pCopyParams_, params, sizeof(hipMemcpy3DParms));
    return hipSuccess;
  }
  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphMemcpyNode* memcpyNode = static_cast<hipGraphMemcpyNode const*>(node);
    return SetParams(memcpyNode->pCopyParams_);
  }
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t ValidateParams(const hipMemcpy3DParms* pNodeParams);

  std::string GetLabel(hipGraphDebugDotFlags flag) {
    size_t offset = 0;
    const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(*pCopyParams_);
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
          label_.c_str(), GetID(), this, memcpyDirection.c_str(), pCopyParams_->srcPtr.pitch,
          pCopyParams_->srcPtr.ptr, pCopyParams_->srcPtr.xsize, pCopyParams_->srcPtr.ysize,
          pCopyParams_->dstPtr.pitch, pCopyParams_->dstPtr.ptr, pCopyParams_->dstPtr.xsize,
          pCopyParams_->dstPtr.ysize, pCopyParams_->srcPos.x, pCopyParams_->srcPos.y,
          pCopyParams_->srcPos.z, pCopyParams_->dstPos.x, pCopyParams_->dstPos.y,
          pCopyParams_->dstPos.z, pCopyParams_->extent.width, pCopyParams_->extent.height,
          pCopyParams_->extent.depth);
      label = buffer;
    } else {
      label = std::to_string(GetID()) + "\nMEMCPY\n(" + memcpyDirection + ")";
    }
    return label;
  }
  std::string GetShape(hipGraphDebugDotFlags flag) {
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }
};

class hipGraphMemcpyNode1D : public hipGraphNode {
 protected:
  void* dst_;
  const void* src_;
  size_t count_;
  hipMemcpyKind kind_;

 public:
  hipGraphMemcpyNode1D(void* dst, const void* src, size_t count, hipMemcpyKind kind,
                       hipGraphNodeType type = hipGraphNodeTypeMemcpy)
      : hipGraphNode(type, "solid", "trapezium", "MEMCPY"),
        dst_(dst),
        src_(src),
        count_(count),
        kind_(kind) {}

  ~hipGraphMemcpyNode1D() {}

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNode1D(static_cast<hipGraphMemcpyNode1D const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) {
    if (IsHtoHMemcpy(dst_, src_, kind_)) {
      return hipSuccess;
    }
    hipError_t status = hipGraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command = nullptr;
    status = ihipMemcpyCommand(command, dst_, src_, count_, kind_, *stream);
    commands_.emplace_back(command);
    return status;
  }

  void EnqueueCommands(hipStream_t stream) {
    bool isH2H = IsHtoHMemcpy(dst_, src_, kind_);
    if (!isH2H) {
      if (commands_.empty()) return;
      // commands_ should have just 1 item
      assert(commands_.size() == 1 && "Invalid command size in hipGraphMemcpyNode1D");
    }
    if (isEnabled_) {
      //HtoH
      if (isH2H) {
        ihipHtoHMemcpy(dst_, src_, count_, *hip::getStream(stream));
        return;
      }
      amd::Command* command = commands_[0];
      amd::HostQueue* cmdQueue = command->queue();
      hip::Stream* hip_stream = hip::getStream(stream);

      if (cmdQueue == hip_stream) {
        command->enqueue();
        command->release();
        return;
      }

      amd::Command::EventWaitList waitList;
      amd::Command* depdentMarker = nullptr;
      amd::Command* cmd = hip_stream->getLastQueuedCommand(true);
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
        amd::Command* depdentMarker = new amd::Marker(*hip_stream, true, waitList);
        if (depdentMarker != nullptr) {
          depdentMarker->enqueue();  // Make sure future commands of queue synced with command
          depdentMarker->release();
        }
        cmd->release();
      }
    } else {
      amd::Command::EventWaitList waitList;
      hip::Stream* hip_stream = hip::getStream(stream);
      amd::Command* command = new amd::Marker(*hip_stream, !kMarkerDisableFlush, waitList);
      command->enqueue();
      command->release();
    }
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

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphMemcpyNode1D* memcpy1DNode = static_cast<hipGraphMemcpyNode1D const*>(node);
    return SetParams(memcpy1DNode->dst_, memcpy1DNode->src_, memcpy1DNode->count_,
                     memcpy1DNode->kind_);
  }
  static hipError_t ValidateParams(void* dst, const void* src, size_t count, hipMemcpyKind kind);
  std::string GetLabel(hipGraphDebugDotFlags flag) {
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
      sprintf(buffer,
              "{\n%s\n| {{ID | node handle} | {%u | %p}}\n| {kind | %s}\n| {{srcPtr | dstPtr} | "
              "{pitch "
              "| ptr | xsize | ysize | pitch | ptr | xsize | size} | {%zu | %p | %zu | %zu | %zu | %p "
              "| %zu "
              "| %zu}}\n| {{srcPos | {{x | %zu} | {y | %zu} | {z | %zu}}} | {dstPos | {{x | %zu} | {y "
              "| "
              "%zu} | {z | %zu}}} | {Extent | {{Width | %zu} | {Height | %zu} | {Depth | %zu}}}}\n}",
              label_.c_str(), GetID(), this, memcpyDirection.c_str(), (size_t)0,
              src_, (size_t)0, (size_t)0, (size_t)0, dst_, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (size_t)0, count_, (size_t)1, (size_t)1);
      label = buffer;
    } else {
      label = std::to_string(GetID()) + "\n" + label_ + "\n(" + memcpyDirection + "," +
          std::to_string(count_) + ")";
    }
    return label;
  }
  std::string GetShape(hipGraphDebugDotFlags flag) {
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }
};

class hipGraphMemcpyNodeFromSymbol : public hipGraphMemcpyNode1D {
  const void* symbol_;
  size_t offset_;

 public:
  hipGraphMemcpyNodeFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                               hipMemcpyKind kind)
      : hipGraphMemcpyNode1D(dst, nullptr, count, kind, hipGraphNodeTypeMemcpy),
        symbol_(symbol),
        offset_(offset) {}

  ~hipGraphMemcpyNodeFromSymbol() {}

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNodeFromSymbol(
        static_cast<hipGraphMemcpyNodeFromSymbol const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
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
                       hipMemcpyKind kind) {
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
    if (dstMemory == nullptr && kind != hipMemcpyHostToDevice) {
      return hipErrorInvalidMemcpyDirection;
    } else if (dstMemory != nullptr && kind != hipMemcpyDeviceToDevice) {
      return hipErrorInvalidMemcpyDirection;
    } else if (kind == hipMemcpyHostToHost || kind == hipMemcpyDeviceToHost) {
      return hipErrorInvalidMemcpyDirection;
    }

    dst_ = dst;
    symbol_ = symbol;
    count_ = count;
    offset_ = offset;
    kind_ = kind;
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphMemcpyNodeFromSymbol* memcpyNode =
        static_cast<hipGraphMemcpyNodeFromSymbol const*>(node);
    return SetParams(memcpyNode->dst_, memcpyNode->symbol_, memcpyNode->count_, memcpyNode->offset_,
                     memcpyNode->kind_);
  }
};
class hipGraphMemcpyNodeToSymbol : public hipGraphMemcpyNode1D {
  const void* symbol_;
  size_t offset_;

 public:
  hipGraphMemcpyNodeToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                             hipMemcpyKind kind)
      : hipGraphMemcpyNode1D(nullptr, src, count, kind, hipGraphNodeTypeMemcpy),
        symbol_(symbol),
        offset_(offset) {}

  ~hipGraphMemcpyNodeToSymbol() {}

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNodeToSymbol(static_cast<hipGraphMemcpyNodeToSymbol const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
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
                       hipMemcpyKind kind) {
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
    if (srcMemory == nullptr && kind != hipMemcpyHostToDevice) {
      return hipErrorInvalidValue;
    } else if (srcMemory != nullptr && kind != hipMemcpyDeviceToDevice) {
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

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphMemcpyNodeToSymbol* memcpyNode =
        static_cast<hipGraphMemcpyNodeToSymbol const*>(node);
    return SetParams(memcpyNode->src_, memcpyNode->symbol_, memcpyNode->count_, memcpyNode->offset_,
                     memcpyNode->kind_);
  }
};

class hipGraphMemsetNode : public hipGraphNode {
  hipMemsetParams* pMemsetParams_;

 public:
  hipGraphMemsetNode(const hipMemsetParams* pMemsetParams)
      : hipGraphNode(hipGraphNodeTypeMemset, "solid", "invtrapezium", "MEMSET") {
    pMemsetParams_ = new hipMemsetParams(*pMemsetParams);
    size_t sizeBytes = 0;
    if (pMemsetParams_->height == 1) {
      sizeBytes = pMemsetParams_->width * pMemsetParams_->elementSize;
    } else {
      sizeBytes = pMemsetParams_->width * pMemsetParams_->height * pMemsetParams_->elementSize;
    }
  }

  ~hipGraphMemsetNode() { delete pMemsetParams_; }
  // Copy constructor
  hipGraphMemsetNode(const hipGraphMemsetNode& memsetNode) : hipGraphNode(memsetNode) {
    pMemsetParams_ = new hipMemsetParams(*memsetNode.pMemsetParams_);
  }

  hipGraphNode* clone() const {
    return new hipGraphMemsetNode(static_cast<hipGraphMemsetNode const&>(*this));
  }

  std::string GetLabel(hipGraphDebugDotFlags flag) {
    std::string label;
    if (flag == hipGraphDebugDotFlagsMemsetNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      char buffer[500];
      sprintf(buffer,
              "{\n%s\n| {{ID | node handle | dptr | pitch | value | elementSize | width | "
              "height} | {%u | %p | %p | %zu | %u | %u | %zu | %zu}}}",
              label_.c_str(), GetID(), this, pMemsetParams_->dst, pMemsetParams_->pitch,
              pMemsetParams_->value, pMemsetParams_->elementSize, pMemsetParams_->width,
              pMemsetParams_->height);
      label = buffer;
    } else {
      size_t sizeBytes;
      if (pMemsetParams_->height == 1) {
        sizeBytes = pMemsetParams_->width * pMemsetParams_->elementSize;
      } else {
        sizeBytes = pMemsetParams_->width * pMemsetParams_->height * pMemsetParams_->elementSize;
      }
      label = std::to_string(GetID()) + "\n" + label_ + "\n(" +
          std::to_string(pMemsetParams_->value) + "," + std::to_string(sizeBytes) + ")";
    }
    return label;
  }

  std::string GetShape(hipGraphDebugDotFlags flag) {
    if (flag == hipGraphDebugDotFlagsMemsetNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    if (pMemsetParams_->height == 1) {
      size_t sizeBytes = pMemsetParams_->width * pMemsetParams_->elementSize;
      hipError_t status = ihipMemsetCommand(commands_, pMemsetParams_->dst, pMemsetParams_->value,
                                            pMemsetParams_->elementSize, sizeBytes, stream);
    } else {
      hipError_t status = ihipMemset3DCommand(
          commands_,
          {pMemsetParams_->dst, pMemsetParams_->pitch, pMemsetParams_->width * pMemsetParams_->elementSize,
           pMemsetParams_->height},
          pMemsetParams_->value, {pMemsetParams_->width * pMemsetParams_->elementSize, pMemsetParams_->height, 1}, stream, pMemsetParams_->elementSize);
    }
    return status;
  }

  void GetParams(hipMemsetParams* params) {
    std::memcpy(params, pMemsetParams_, sizeof(hipMemsetParams));
  }

  hipError_t SetParams(const hipMemsetParams* params, bool isExec = false) {
    hipError_t hip_error = hipSuccess;
    hip_error = ihipGraphMemsetParams_validate(params);
    if (hip_error != hipSuccess) {
      return hip_error;
    }
    if (isExec) {
      size_t discardOffset = 0;
      amd::Memory *memObj = getMemoryObject(params->dst, discardOffset);
      if (memObj != nullptr) {
        amd::Memory *memObjOri = getMemoryObject(pMemsetParams_->dst, discardOffset);
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
        if (pMemsetParams_->width * pMemsetParams_->elementSize != params->width * params->elementSize
         || pMemsetParams_->height != params->height) {
          return hipErrorInvalidValue;
        }
      } else {
        // 2D - hipGraphMemsetNodeSetParams returns invalid value if new width or new height is
        // greter than actual allocation.
        size_t discardOffset = 0;
        amd::Memory *memObj = getMemoryObject(params->dst, discardOffset);
        if (memObj != nullptr) {
          if (params->width * params->elementSize > memObj->getUserData().width_
           || params->height > memObj->getUserData().height_) {
            return hipErrorInvalidValue;
           }
        }
       }
      sizeBytes = params->width * params->elementSize * params->height * 1;
      hip_error =
          ihipMemset3D_validate({params->dst, params->pitch, params->width * params->elementSize, params->height},
                                params->value, {params->width * params->elementSize, params->height, 1}, sizeBytes);
    }
    if (hip_error != hipSuccess) {
      return hip_error;
    }
    std::memcpy(pMemsetParams_, params, sizeof(hipMemsetParams));
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphMemsetNode* memsetNode = static_cast<hipGraphMemsetNode const*>(node);
    return SetParams(memsetNode->pMemsetParams_);
  }
};

class hipGraphEventRecordNode : public hipGraphNode {
  hipEvent_t event_;

 public:
  hipGraphEventRecordNode(hipEvent_t event)
      : hipGraphNode(hipGraphNodeTypeEventRecord, "solid", "rectangle", "EVENT_RECORD"),
        event_(event) {}
  ~hipGraphEventRecordNode() {}

  hipGraphNode* clone() const {
    return new hipGraphEventRecordNode(static_cast<hipGraphEventRecordNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
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

  void EnqueueCommands(hipStream_t stream) {
    if (!commands_.empty()) {
      hip::Event* e = reinterpret_cast<hip::Event*>(event_);
      // command release during enqueueRecordCommand
      hipError_t status = e->enqueueRecordCommand(stream, commands_[0], true);
      if (status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] enqueue event record command failed for node %p - status %d\n", this,
                status);
      }
    }
  }

  void GetParams(hipEvent_t* event) const { *event = event_; }

  hipError_t SetParams(hipEvent_t event) {
    event_ = event;
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphEventRecordNode* eventRecordNode =
        static_cast<hipGraphEventRecordNode const*>(node);
    return SetParams(eventRecordNode->event_);
  }
};

class hipGraphEventWaitNode : public hipGraphNode {
  hipEvent_t event_;

 public:
  hipGraphEventWaitNode(hipEvent_t event)
      : hipGraphNode(hipGraphNodeTypeWaitEvent, "solid", "rectangle", "EVENT_WAIT"),
        event_(event) {}
  ~hipGraphEventWaitNode() {}

  hipGraphNode* clone() const {
    return new hipGraphEventWaitNode(static_cast<hipGraphEventWaitNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
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

  void EnqueueCommands(hipStream_t stream) {
    if (!commands_.empty()) {
      hip::Event* e = reinterpret_cast<hip::Event*>(event_);
      hipError_t status = e->enqueueStreamWaitCommand(stream, commands_[0]);
      if (status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] enqueue stream wait command failed for node %p - status %d\n", this,
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

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphEventWaitNode* eventWaitNode = static_cast<hipGraphEventWaitNode const*>(node);
    return SetParams(eventWaitNode->event_);
  }
};

class hipGraphHostNode : public hipGraphNode {
  hipHostNodeParams* pNodeParams_;

 public:
  hipGraphHostNode(const hipHostNodeParams* pNodeParams)
      : hipGraphNode(hipGraphNodeTypeHost, "solid", "rectangle", "HOST") {
    pNodeParams_ = new hipHostNodeParams(*pNodeParams);
  }
  ~hipGraphHostNode() { delete pNodeParams_; }

  hipGraphHostNode(const hipGraphHostNode& hostNode) : hipGraphNode(hostNode) {
    pNodeParams_ = new hipHostNodeParams(*hostNode.pNodeParams_);
  }

  hipGraphNode* clone() const {
    return new hipGraphHostNode(static_cast<hipGraphHostNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
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
    hipHostNodeParams* pNodeParams = reinterpret_cast<hipHostNodeParams*>(user_data);
    pNodeParams->fn(pNodeParams->userData);
  }

  void EnqueueCommands(hipStream_t stream) {
    if (!commands_.empty()) {
      if (!commands_[0]->setCallback(CL_COMPLETE, hipGraphHostNode::Callback, pNodeParams_)) {
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
    std::memcpy(params, pNodeParams_, sizeof(hipHostNodeParams));
  }
  hipError_t SetParams(const hipHostNodeParams* params) {
    std::memcpy(pNodeParams_, params, sizeof(hipHostNodeParams));
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphHostNode* hostNode = static_cast<hipGraphHostNode const*>(node);
    return SetParams(hostNode->pNodeParams_);
  }
};

class hipGraphEmptyNode : public hipGraphNode {
 public:
  hipGraphEmptyNode() : hipGraphNode(hipGraphNodeTypeEmpty, "solid", "rectangle", "EMPTY") {}
  ~hipGraphEmptyNode() {}

  hipGraphNode* clone() const {
    return new hipGraphEmptyNode(static_cast<hipGraphEmptyNode const&>(*this));
  }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = hipGraphNode::CreateCommand(stream);
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

class hipGraphMemAllocNode : public hipGraphNode {
  hipMemAllocNodeParams node_params_; // Node parameters for memory allocation

 public:
  hipGraphMemAllocNode(const hipMemAllocNodeParams* node_params)
      : hipGraphNode(hipGraphNodeTypeMemAlloc, "solid", "rectangle", "MEM_ALLOC") {
        node_params_ = *node_params;
      }
  ~hipGraphMemAllocNode() {}

  hipGraphNode* clone() const {
    return new hipGraphMemAllocNode(static_cast<hipGraphMemAllocNode const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) {
    auto error = hipGraphNode::CreateCommand(stream);
    auto ptr = Execute(stream_);
    return error;
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

class hipGraphMemFreeNode : public hipGraphNode {
  void* device_ptr_;    // Device pointer of the freed memory

 public:
  hipGraphMemFreeNode(void* dptr)
    : hipGraphNode(hipGraphNodeTypeMemFree, "solid", "rectangle", "MEM_FREE")
    , device_ptr_(dptr) {}
  ~hipGraphMemFreeNode() {}

  hipGraphNode* clone() const {
    return new hipGraphMemFreeNode(static_cast<hipGraphMemFreeNode const&>(*this));
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) {
    auto error = hipGraphNode::CreateCommand(stream);
    Execute(stream_);
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
