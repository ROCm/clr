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

typedef hipGraphNode* Node;
hipError_t ihipValidateKernelParams(const hipKernelNodeParams* pNodeParams);
hipError_t FillCommands(std::vector<std::vector<Node>>& parallelLists,
                        std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
                        std::vector<Node>& levelOrder, amd::Command*& rootCommand,
                        amd::Command*& endCommand, amd::HostQueue* queue);
void UpdateQueue(std::vector<std::vector<Node>>& parallelLists, amd::HostQueue*& queue,
                 hipGraphExec* ptr);
struct hipGraphNode {
 protected:
  amd::HostQueue* queue_;
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

 public:
  hipGraphNode(hipGraphNodeType type)
      : type_(type),
        level_(0),
        visited_(false),
        inDegree_(0),
        outDegree_(0),
        id_(nextID++),
        parentGraph_(nullptr) {}
  /// Copy Constructor
  hipGraphNode(const hipGraphNode& node) {
    level_ = node.level_;
    type_ = node.type_;
    inDegree_ = node.inDegree_;
    outDegree_ = node.outDegree_;
    visited_ = false;
    id_ = node.id_;
    parentGraph_ = nullptr;
  }

  virtual ~hipGraphNode() {
    for (auto node : edges_) {
      node->RemoveDependency(this);
    }
    for (auto node : dependencies_) {
      node->RemoveEdge(this);
    }
  }

  amd::HostQueue* GetQueue() { return queue_; }

  virtual void SetQueue(amd::HostQueue* queue, hipGraphExec* ptr = nullptr) { queue_ = queue; }
  /// Create amd::command for the graph node
  virtual hipError_t CreateCommand(amd::HostQueue* queue) {
    commands_.clear();
    queue_ = queue;
    return hipSuccess;
  }
  /// Method to release amd::command part of node
  virtual void ReleaseCommand() {
    for (auto command : commands_) {
      command->release();
    }
    commands_.clear();
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
  bool RemoveEdge(const Node& childNode) {
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
    const std::vector<Node>& dependencies = childNode->GetDependencies();
    int32_t level = 0;
    int32_t parentLevel = 0;
    for (auto parent : dependencies) {
      parentLevel = parent->GetLevel();
      level = std::max(level, (parentLevel + 1));
    }
    childNode->SetLevel(level);
    childNode->RemoveDependency(this);
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
  virtual size_t GetNumParallelQueues() { return 0; }
  /// Enqueue commands part of the node
  virtual void EnqueueCommands(hipStream_t stream) {
    for (auto& command : commands_) {
      command->enqueue();
      command->release();
    }
  }
  ihipGraph* GetParentGraph() { return parentGraph_; }
  void SetParentGraph(ihipGraph* graph) { parentGraph_ = graph; }
  virtual hipError_t SetParams(hipGraphNode* node) { return hipSuccess; }
};

struct ihipGraph {
  std::vector<Node> vertices_;

 public:
  ihipGraph() {}
  ~ihipGraph() {
    for (auto node : vertices_) {
      delete node;
    }
  }
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
  void GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                      std::vector<Node>& singleList, std::vector<std::vector<Node>>& parallelList,
                      std::unordered_map<Node, std::vector<Node>>& dependencies);
  void GetRunList(std::vector<std::vector<Node>>& parallelList,
                  std::unordered_map<Node, std::vector<Node>>& dependencies);
  void LevelOrder(std::vector<Node>& levelOrder);
  ihipGraph* clone(std::unordered_map<Node, Node>& clonedNodes) const;
  ihipGraph* clone() const;
};

struct hipGraphExec {
  std::vector<std::vector<Node>> parallelLists_;
  // level order of the graph doesn't include nodes embedded as part of the child graph
  std::vector<Node> levelOrder_;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists_;
  std::vector<amd::HostQueue*> parallelQueues_;
  uint currentQueueIndex_;
  std::unordered_map<Node, Node> clonedNodes_;
  amd::Command* lastEnqueuedCommand_;

 public:
  hipGraphExec(std::vector<Node>& levelOrder, std::vector<std::vector<Node>>& lists,
               std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
               std::unordered_map<Node, Node>& clonedNodes)
      : parallelLists_(lists),
        levelOrder_(levelOrder),
        nodeWaitLists_(nodeWaitLists),
        clonedNodes_(clonedNodes),
        lastEnqueuedCommand_(nullptr),
        currentQueueIndex_(0) {}

  ~hipGraphExec() {
    // new commands are launched for every launch they are destroyed as and when command is
    // terminated after it complete execution
    for (auto queue : parallelQueues_) {
      queue->release();
    }
    for (auto it = clonedNodes_.begin(); it != clonedNodes_.end(); it++) delete it->second;
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

  std::vector<Node>& GetNodes() { return levelOrder_; }

  amd::HostQueue* GetAvailableQueue() { return parallelQueues_[currentQueueIndex_++]; }
  void ResetQueueIndex() { currentQueueIndex_ = 0; }
  hipError_t Init();
  hipError_t CreateQueues(size_t numQueues);
  hipError_t Run(hipStream_t stream);
};
struct hipChildGraphNode : public hipGraphNode {
  struct ihipGraph* childGraph_;
  std::vector<Node> childGraphlevelOrder_;
  std::vector<std::vector<Node>> parallelLists_;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists_;
  amd::Command* lastEnqueuedCommand_;

 public:
  hipChildGraphNode(ihipGraph* g) : hipGraphNode(hipGraphNodeTypeGraph) {
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

  size_t GetNumParallelQueues() {
    LevelOrder(childGraphlevelOrder_);
    size_t num = 0;
    for (auto& node : childGraphlevelOrder_) {
      num += node->GetNumParallelQueues();
    }
    // returns total number of parallel queues required for child graph nodes to be launched
    // first parallel list will be launched on the same queue as parent
    return num + (parallelLists_.size() - 1);
  }

  void SetQueue(amd::HostQueue* queue, hipGraphExec* ptr = nullptr) {
    queue_ = queue;
    UpdateQueue(parallelLists_, queue, ptr);
  }

  // For nodes that are dependent on the child graph node waitlist is the last node of the first
  // parallel list
  std::vector<amd::Command*>& GetCommands() { return parallelLists_[0].back()->GetCommands(); }

  // Create child graph node commands and set waitlists
  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(2);
    amd::Command* rootCommand = nullptr;
    amd::Command* endCommand = nullptr;
    status = FillCommands(parallelLists_, nodeWaitLists_, childGraphlevelOrder_, rootCommand,
                          endCommand, queue);
    if (rootCommand != nullptr) {
      commands_.push_back(rootCommand);
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
    }
    // enqueue nodes in child graph in level order
    for (auto& node : childGraphlevelOrder_) {
      node->EnqueueCommands(stream);
    }
    // enqueue child graph end command
    if (commands_.size() == 2) {
      commands_[1]->enqueue();
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
};

class hipGraphKernelNode : public hipGraphNode {
  hipKernelNodeParams* pKernelParams_;
  hipFunction_t func_;

 public:
  hipGraphKernelNode(const hipKernelNodeParams* pNodeParams, const hipFunction_t func)
      : hipGraphNode(hipGraphNodeTypeKernel) {
    pKernelParams_ = new hipKernelNodeParams(*pNodeParams);
    func_ = func;
  }
  ~hipGraphKernelNode() { delete pKernelParams_; }
  hipGraphKernelNode(const hipGraphKernelNode& rhs) : hipGraphNode(rhs) {
    pKernelParams_ = new hipKernelNodeParams(*rhs.pKernelParams_);
    func_ = rhs.func_;
  }
  hipGraphNode* clone() const {
    return new hipGraphKernelNode(static_cast<hipGraphKernelNode const&>(*this));
  }
  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    status = ihipLaunchKernelCommand(
        command, func_, pKernelParams_->gridDim.x * pKernelParams_->blockDim.x,
        pKernelParams_->gridDim.y * pKernelParams_->blockDim.y,
        pKernelParams_->gridDim.z * pKernelParams_->blockDim.z, pKernelParams_->blockDim.x,
        pKernelParams_->blockDim.y, pKernelParams_->blockDim.z, pKernelParams_->sharedMemBytes,
        queue, pKernelParams_->kernelParams, pKernelParams_->extra, nullptr, nullptr, 0, 0, 0, 0, 0,
        0, 0);
    commands_.emplace_back(command);
    return status;
  }
  void GetParams(hipKernelNodeParams* params) {
    std::memcpy(params, pKernelParams_, sizeof(hipKernelNodeParams));
  }
  hipError_t SetParams(const hipKernelNodeParams* params) {
    // updates kernel params
    hipError_t status = ihipValidateKernelParams(params);
    if (hipSuccess != status) {
      return status;
    }
    if (params->func != pKernelParams_->func) {
      hipFunction_t func = nullptr;
      hipError_t status =
          PlatformState::instance().getStatFunc(&func, params->func, ihipGetDevice());
      if ((status != hipSuccess) || (func == nullptr)) {
        return hipErrorInvalidDeviceFunction;
      }
      func_ = func;
    }
    std::memcpy(pKernelParams_, params, sizeof(hipKernelNodeParams));
    return status;
  }
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t SetCommandParams(const hipKernelNodeParams* params) {
    if (params->func != pKernelParams_->func) {
      hipFunction_t func = nullptr;
      hipError_t status =
          PlatformState::instance().getStatFunc(&func, params->func, ihipGetDevice());
      if ((status != hipSuccess) || (func == nullptr)) {
        return hipErrorInvalidDeviceFunction;
      }
      func_ = func;
    }
    // updates kernel params
    hipError_t status = ihipValidateKernelParams(params);
    if (hipSuccess != status) {
      return status;
    }
    size_t globalWorkOffset[3] = {0};
    size_t globalWorkSize[3] = {params->gridDim.x, params->gridDim.y, params->gridDim.z};
    size_t localWorkSize[3] = {params->blockDim.x, params->blockDim.y, params->blockDim.z};
    reinterpret_cast<amd::NDRangeKernelCommand*>(commands_[0])
        ->setSizes(globalWorkOffset, globalWorkSize, localWorkSize);
    reinterpret_cast<amd::NDRangeKernelCommand*>(commands_[0])
        ->setSharedMemBytes(params->sharedMemBytes);
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphKernelNode* kernelNode = static_cast<hipGraphKernelNode const*>(node);
    return SetParams(kernelNode->pKernelParams_);
  }
};

class hipGraphMemcpyNode : public hipGraphNode {
  hipMemcpy3DParms* pCopyParams_;

 public:
  hipGraphMemcpyNode(const hipMemcpy3DParms* pCopyParams) : hipGraphNode(hipGraphNodeTypeMemcpy) {
    pCopyParams_ = new hipMemcpy3DParms(*pCopyParams);
  }
  ~hipGraphMemcpyNode() { delete pCopyParams_; }

  hipGraphMemcpyNode(const hipGraphMemcpyNode& rhs) : hipGraphNode(rhs) {
    pCopyParams_ = new hipMemcpy3DParms(*rhs.pCopyParams_);
  }

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNode(static_cast<hipGraphMemcpyNode const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    status = ihipMemcpy3DCommand(command, pCopyParams_, queue);
    commands_.emplace_back(command);
    return status;
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
  hipError_t SetCommandParams(const hipMemcpy3DParms* pNodeParams);
  hipError_t ValidateParams(const hipMemcpy3DParms* pNodeParams);
};

class hipGraphMemcpyNode1D : public hipGraphNode {
 protected:
  void* dst_;
  const void* src_;
  size_t count_;
  hipMemcpyKind kind_;

 public:
  hipGraphMemcpyNode1D(void* dst, const void* src, size_t count, hipMemcpyKind kind,
                       hipGraphNodeType type = hipGraphNodeTypeMemcpy1D)
      : hipGraphNode(type), dst_(dst), src_(src), count_(count), kind_(kind) {}

  ~hipGraphMemcpyNode1D() {}

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNode1D(static_cast<hipGraphMemcpyNode1D const&>(*this));
  }

  virtual hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command = nullptr;
    status = ihipMemcpyCommand(command, dst_, src_, count_, kind_, *queue);
    commands_.emplace_back(command);
    return status;
  }

  void EnqueueCommands(hipStream_t stream) {
    if (!commands_.empty()) {
      for (auto& command : commands_) {
        command->enqueue();
      }
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
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t SetCommandParams(void* dst, const void* src, size_t count, hipMemcpyKind kind);
  hipError_t ValidateParams(void* dst, const void* src, size_t count, hipMemcpyKind kind);
};

class hipGraphMemcpyNodeFromSymbol : public hipGraphMemcpyNode1D {
  const void* symbol_;
  size_t offset_;

 public:
  hipGraphMemcpyNodeFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                               hipMemcpyKind kind)
      : hipGraphMemcpyNode1D(dst, nullptr, count, kind, hipGraphNodeTypeMemcpyFromSymbol),
        symbol_(symbol),
        offset_(offset) {}

  ~hipGraphMemcpyNodeFromSymbol() {}

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNodeFromSymbol(
        static_cast<hipGraphMemcpyNodeFromSymbol const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
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
    status = ihipMemcpyCommand(command, dst_, device_ptr, count_, kind_, *queue);
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

    hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
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
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t SetCommandParams(void* dst, const void* symbol, size_t count, size_t offset,
                              hipMemcpyKind kind) {
    size_t sym_size = 0;
    hipDeviceptr_t device_ptr = nullptr;

    hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
    }
    return hipGraphMemcpyNode1D::SetCommandParams(dst, device_ptr, count, kind);
  }
};
class hipGraphMemcpyNodeToSymbol : public hipGraphMemcpyNode1D {
  const void* symbol_;
  size_t offset_;

 public:
  hipGraphMemcpyNodeToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                             hipMemcpyKind kind)
      : hipGraphMemcpyNode1D(nullptr, src, count, kind, hipGraphNodeTypeMemcpyToSymbol),
        symbol_(symbol),
        offset_(offset) {}

  ~hipGraphMemcpyNodeToSymbol() {}

  hipGraphNode* clone() const {
    return new hipGraphMemcpyNodeToSymbol(static_cast<hipGraphMemcpyNodeToSymbol const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
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
    status = ihipMemcpyCommand(command, device_ptr, src_, count_, kind_, *queue);
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

    hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
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
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t SetCommandParams(const void* symbol, const void* src, size_t count, size_t offset,
                              hipMemcpyKind kind) {
    size_t sym_size = 0;
    hipDeviceptr_t device_ptr = nullptr;

    hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
    if (status != hipSuccess) {
      return status;
    }
    return hipGraphMemcpyNode1D::SetCommandParams(device_ptr, src, count, kind);
  }
};

class hipGraphMemsetNode : public hipGraphNode {
  hipMemsetParams* pMemsetParams_;

 public:
  hipGraphMemsetNode(const hipMemsetParams* pMemsetParams) : hipGraphNode(hipGraphNodeTypeMemset) {
    pMemsetParams_ = new hipMemsetParams(*pMemsetParams);
  }
  ~hipGraphMemsetNode() { delete pMemsetParams_; }
  // Copy constructor
  hipGraphMemsetNode(const hipGraphMemsetNode& memsetNode) : hipGraphNode(memsetNode) {
    pMemsetParams_ = new hipMemsetParams(*memsetNode.pMemsetParams_);
  }

  hipGraphNode* clone() const {
    return new hipGraphMemsetNode(static_cast<hipGraphMemsetNode const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    if (pMemsetParams_->height == 1) {
      return ihipMemsetCommand(commands_, pMemsetParams_->dst, pMemsetParams_->value,
                               pMemsetParams_->elementSize,
                               pMemsetParams_->width * pMemsetParams_->elementSize, queue);
    } else {
      return ihipMemset3DCommand(commands_,
                                 {pMemsetParams_->dst, pMemsetParams_->pitch, pMemsetParams_->width,
                                  pMemsetParams_->height},
                                 pMemsetParams_->elementSize,
                                 {pMemsetParams_->width, pMemsetParams_->height, 1}, queue);
    }
    return hipSuccess;
  }

  void GetParams(hipMemsetParams* params) {
    std::memcpy(params, pMemsetParams_, sizeof(hipMemsetParams));
  }
  hipError_t SetParams(const hipMemsetParams* params) {
    hipError_t hip_error = hipSuccess;
    hip_error = ihipMemset_validate(params->dst, params->value, params->elementSize,
                                    params->width * params->elementSize);
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
      : hipGraphNode(hipGraphNodeTypeEventRecord), event_(event) {}
  ~hipGraphEventRecordNode() {}

  hipGraphNode* clone() const {
    return new hipGraphEventRecordNode(static_cast<hipGraphEventRecordNode const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    hip::Event* e = reinterpret_cast<hip::Event*>(event_);
    commands_.reserve(1);
    amd::Command* command;
    status = e->recordCommand(command, queue);
    commands_.emplace_back(command);
    return status;
  }

  void EnqueueCommands(hipStream_t stream) {
    if (!commands_.empty()) {
      hip::Event* e = reinterpret_cast<hip::Event*>(event_);
      hipError_t status = e->enqueueRecordCommand(stream, commands_[0], true);
      if (status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] enqueue event record command failed for node %p - status %d\n", this,
                status);
      }
    }
  }

  void GetParams(hipEvent_t* event) { *event = event_; }

  hipError_t SetParams(hipEvent_t event) {
    event_ = event;
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphEventRecordNode* eventRecordNode =
        static_cast<hipGraphEventRecordNode const*>(node);
    return SetParams(eventRecordNode->event_);
  }
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t SetCommandParams(hipEvent_t event) {
    amd::HostQueue* queue;
    if (!commands_.empty()) {
      queue = commands_[0]->queue();
      commands_[0]->release();
    }
    commands_.clear();
    return CreateCommand(queue);
  }
};

class hipGraphEventWaitNode : public hipGraphNode {
  hipEvent_t event_;

 public:
  hipGraphEventWaitNode(hipEvent_t event)
      : hipGraphNode(hipGraphNodeTypeWaitEvent), event_(event) {}
  ~hipGraphEventWaitNode() {}

  hipGraphNode* clone() const {
    return new hipGraphEventWaitNode(static_cast<hipGraphEventWaitNode const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    hip::Event* e = reinterpret_cast<hip::Event*>(event_);
    commands_.reserve(1);
    amd::Command* command;
    status = e->streamWaitCommand(command, queue);
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
    }
  }

  void GetParams(hipEvent_t* event) { *event = event_; }

  hipError_t SetParams(hipEvent_t event) {
    event_ = event;
    return hipSuccess;
  }

  hipError_t SetParams(hipGraphNode* node) {
    const hipGraphEventWaitNode* eventWaitNode = static_cast<hipGraphEventWaitNode const*>(node);
    return SetParams(eventWaitNode->event_);
  }
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t SetCommandParams(hipEvent_t event) {
    amd::HostQueue* queue;
    if (!commands_.empty()) {
      queue = commands_[0]->queue();
      commands_[0]->release();
    }
    commands_.clear();
    return CreateCommand(queue);
  }
};

class hipGraphHostNode : public hipGraphNode {
  hipHostNodeParams* pNodeParams_;

 public:
  hipGraphHostNode(const hipHostNodeParams* pNodeParams) : hipGraphNode(hipGraphNodeTypeHost) {
    pNodeParams_ = new hipHostNodeParams(*pNodeParams);
  }
  ~hipGraphHostNode() { delete pNodeParams_; }

  hipGraphHostNode(const hipGraphHostNode& hostNode) : hipGraphNode(hostNode) {
    pNodeParams_ = new hipHostNodeParams(*hostNode.pNodeParams_);
  }

  hipGraphNode* clone() const {
    return new hipGraphHostNode(static_cast<hipGraphHostNode const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    amd::Command::EventWaitList waitList;
    commands_.reserve(1);
    amd::Command* command = new amd::Marker(*queue, !kMarkerDisableFlush, waitList);
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
  // ToDo: use this when commands are cloned and command params are to be updated
  hipError_t SetCommandParams(const hipHostNodeParams* params);
};

class hipGraphEmptyNode : public hipGraphNode {
 public:
  hipGraphEmptyNode() : hipGraphNode(hipGraphNodeTypeEmpty) {}
  ~hipGraphEmptyNode() {}

  hipGraphNode* clone() const {
    return new hipGraphEmptyNode(static_cast<hipGraphEmptyNode const&>(*this));
  }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    hipError_t status = hipGraphNode::CreateCommand(queue);
    if (status != hipSuccess) {
      return status;
    }
    amd::Command::EventWaitList waitList;
    commands_.reserve(1);
    amd::Command* command = new amd::Marker(*queue, !kMarkerDisableFlush, waitList);
    commands_.emplace_back(command);
    return hipSuccess;
  }
};
