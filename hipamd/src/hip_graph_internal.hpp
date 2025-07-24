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
#include "platform/graph_vmheap.hpp"

typedef struct ihipExtKernelEvents {
  hipEvent_t startEvent_;
  hipEvent_t stopEvent_;
} ihipExtKernelEvents;

namespace hip {
class Graph;
class GraphNode;
class GraphExec;
class UserObject;
class GraphKernelNode;
namespace ga {
class GraphAnalysis;
}
typedef GraphNode* Node;

class UserObject : public amd::ReferenceCountedObject {
  typedef void (*UserCallbackDestructor)(void* data);

 public:
  // Graphs owns this user object.
  // In case if User object is about to be deleted (last release()), Pointer refering to it
  // should be cleared from Graph's list of user object.
  std::unordered_set<Graph*> owning_graphs_;

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
    owning_graphs_.clear();
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
    amd::ScopedLock lock(UserObjectLock_);
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
  static std::unordered_set<UserObject*> ObjectSet_;
  static amd::Monitor UserObjectLock_;
};

class hipGraphNodeDOTAttribute {
 protected:
  const char* style_;
  const char* shape_;
  const char* label_;

  hipGraphNodeDOTAttribute(const char* style, const char* shape, const char* label) {
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

  virtual const char* GetShape(hipGraphDebugDotFlags flag) { return shape_; }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) { return label_; }

  virtual void PrintAttributes(std::ostream& out, hipGraphDebugDotFlags flag) {}
};

class GraphKernelArgManager : public amd::ReferenceCountedObject,
                              public amd::GraphKernelArgManager {
 public:
  GraphKernelArgManager() : amd::ReferenceCountedObject() {}
  ~GraphKernelArgManager() {
    //! Release the kernel arg pools
    if (device_ != nullptr) {
      for (auto& element : kernarg_graph_) {
        device_->hostFree(element.kernarg_pool_addr_, element.kernarg_pool_size_);
      }
      kernarg_graph_.clear();
    }
  }

  // Allocate kernel arg pool on device for the given size.
  bool AllocGraphKernargPool(size_t pool_size, amd::Device* device);

  // Allocate kernel args from current chunck for given size and alignment.
  // If kernel arg pool is full allocate new chunck and alloc kern args from new pool.
  address AllocKernArg(size_t size, size_t alignment) override;

  // Do HDP flush/When HDP flush register is invalid fallback to Readback
  void ReadBackOrFlush();

 private:
  struct KernelArgPoolGraph {
    KernelArgPoolGraph(address base_addr, size_t size)
        : kernarg_pool_addr_(base_addr), kernarg_pool_size_(size), kernarg_pool_offset_(0) {}
    address kernarg_pool_addr_;   //! Base address of the kernel arg pool
    size_t kernarg_pool_size_;    //! Size of the pool
    size_t kernarg_pool_offset_;  //! Current offset in the kernel arg alloc
  };
  bool device_kernarg_pool_ = false;  //! Indicate if kernel pool in device mem
  amd::Device* device_ = nullptr;     //! Device from where kernel arguments are allocated
  std::vector<KernelArgPoolGraph> kernarg_graph_;  //! Vector of allocated kernarg pool
  using KernelArgImpl = device::Settings::KernelArgImpl;
};

class GraphNode : public hipGraphNodeDOTAttribute {
 public:
  GraphNode(hipGraphNodeType type, const char* style = "", const char* shape = "",
            const char* label = "")
      : type_(type),
        visited_(false),
        inDegree_(0),
        outDegree_(0),
        id_(nextID++),
        parentGraph_(nullptr),
        isEnabled_(1),
        dev_id_(ihipGetDevice()),
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
    dev_id_ = node.dev_id_;
  }

  virtual ~GraphNode() {
    for (auto node : edges_) {
      node->RemoveDependency(this);
    }
    for (auto node : dependencies_) {
      node->RemoveEdge(this);
    }
    for (auto packet : gpuPackets_) {
      delete[] packet;
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
  std::vector<uint8_t*>& GetAqlPackets() { return gpuPackets_; }
  void SetKernelName(const std::string& kernelName) { capturedKernelName_ = kernelName; }
  const std::string& GetKernelName() const { return capturedKernelName_; }
  size_t GetKerArgSize() const { return alignedKernArgSize_; }
  size_t GetKernargSegmentByteSize() const { return kernargSegmentByteSize_; }
  size_t GetKernargSegmentAlignment() const { return kernargSegmentAlignment_; }
  hipError_t CaptureAndFormPacket(GraphKernelArgManager* kernArgMgr) {
    auto capture_stream = hip::getNullStream(g_devices[dev_id_]->devices()[0]->context(), false);
    hipError_t status = CreateCommand(capture_stream);
    if (status != hipSuccess) {
      return status;
    }

    gpuPackets_.clear();
    for (auto& command : commands_) {
      command->setPktCapturingState(true, &gpuPackets_, kernArgMgr, &capturedKernelName_);
      // Enqueue command to capture GPU Packet. The packet is not submitted to the device.
      // The packet is stored in gpuPacket_ and submitted during graph launch.
      command->submit(*(command->queue())->vdev());
      command->release();
    }
    // Commands are captured and released. Clear them from the object.
    commands_.clear();

    return status;
  }
  hip::Stream* GetQueue() const { return stream_; }

  virtual std::pair<std::set<void*>, std::set<void*>> Values() const { return {}; }

  virtual void UpdatePtr(void* old_ptr, void** new_ptr) {}

  virtual void SetStream(hip::Stream* stream) { stream_ = stream; }
  //! Updates the grpah node with the execution stream
  void SetStream(
    const std::vector<hip::Stream*>& streams  //!< A pool of streams to use in graph's execution
    ) {
    assert(stream_id_ != -1 && "Stream ID wasn't initialized");
    stream_ = streams[stream_id_];
    // Reset the launch ID after the stream assignment
    launch_id_ = -1;
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
  void AddDependency(const Node& node) {
    dependencies_.push_back(node);
    inDegree_++;
  }
  /// Remove graph node dependency
  void RemoveDependency(const Node& node) {
    dependencies_.erase(std::remove(dependencies_.begin(), dependencies_.end(), node),
                        dependencies_.end());
    inDegree_--;
  }
  void RemoveEdge(const Node& childNode) {
    edges_.erase(std::remove(edges_.begin(), edges_.end(), childNode), edges_.end());
    outDegree_--;
  }
  void AddEdge(const Node& childNode) {
    edges_.push_back(childNode);
    outDegree_++;
  }
  /// Add edge, update parent node outdegree, child node indegree and dependency
  void AddEdgeDep(const Node& childNode) {
    AddEdge(childNode);
    childNode->AddDependency(this);
  }
  /// Remove edge, update parent node outdegree, child node indegree and dependency
  bool RemoveEdgeDep(const Node& childNode) {
    // std::remove changes the end() hence saving it before hand for validation
    auto currEdgeEnd = edges_.end();
    auto it = std::remove(edges_.begin(), edges_.end(), childNode);
    if (it == currEdgeEnd) {
      // Should come here if childNode is not present in the edge list
      return false;
    }
    edges_.erase(it, edges_.end());
    outDegree_--;
    childNode->RemoveDependency(this);
    return true;
  }
  /// Return graph node children
  const std::vector<Node>& GetEdges() const { return edges_; }
  /// Updates graph node children
  void SetEdges(std::vector<Node>& edges) {
    for (auto entry : edges) {
      edges_.push_back(entry);
    }
  }
  /// Get topological sort of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual bool TopologicalOrder(std::vector<Node>& TopoOrder) { return true; }
  /// Update waitlist of the nodes embedded as part of the graphnode(e.g. ChildGraph)
  virtual void UpdateEventWaitLists(const amd::Command::EventWaitList& waitList) {
    for (auto command : commands_) {
      command->updateEventWaitList(waitList);
    }
  }
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
  Graph* GetParentGraph() const { return parentGraph_; }
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
  // Returns true if capture is enabled for the current node.
  virtual bool GraphCaptureEnabled() {
    bool isGraphCapture = false;
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      switch (GetType()) {
        case hipGraphNodeTypeMemset:
          isGraphCapture = true;
          break;
        default:
          break;
      }
    }
    return isGraphCapture;
  }
  virtual void PrintAttributes(std::ostream& out, hipGraphDebugDotFlags flag) override {
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
    if (DEBUG_HIP_GRAPH_DOT_PRINT) {
      out << "\nStreamId:" << stream_id_;
      out << "\nSignalIsRequired: " << ((signal_is_required_) ? "true" : "false");
      out << "\nDeviceId:" << dev_id_;
    }
    out << "\"";
    out << "];";
  }
  void SetDeviceId(int id) { dev_id_ = id; }
  int GetDeviceId() const { return dev_id_; }

 protected:
  // Declare Graph and GraphExec as friends of node for simpler access to GraphNode fields
  friend class Graph;
  friend class GraphExec;
  hip::Stream* stream_ = nullptr;
  unsigned int id_;
  hipGraphNodeType type_;
  std::vector<amd::Command*> commands_;
  std::vector<Node> edges_;
  std::vector<Node> dependencies_;
  bool visited_;
  size_t inDegree_;         //!< count of in coming edges (@todo: remove, it's dependencies_.size())
  size_t outDegree_;        //!< count of outgoing edges (@todo: remove, it's edges_.size())
  int32_t stream_id_ = -1;  //! Stream ID on which this node will be executed
  int32_t launch_id_ = -1;  //! Launch ID of this node in the entire graph execution sequence
  static int nextID;
  Graph* parentGraph_;
  static std::unordered_set<GraphNode*> nodeSet_;
  static amd::Monitor nodeSetLock_;
  static amd::Monitor WorkerThreadLock_;
  unsigned int isEnabled_;
  bool signal_is_required_ = false;   //!< This node requires a signal on the command
  std::vector<uint8_t*> gpuPackets_;  //!< GPU Packet to enqueue during graph launch
  std::string capturedKernelName_;
  size_t alignedKernArgSize_ = 256;       //!< Aligned size required for kernel args
  size_t kernargSegmentByteSize_ = 512;   //!< Kernel arg segment byte size
  size_t kernargSegmentAlignment_ = 256;  //!< Kernel arg segment alignment
  int dev_id_;  //!< Device Id when node is created(dev id from capture stream/current device
                //!< when explicitly added)
};

class Graph {
 public:
  //!< Contains mem alloc dptrs whose corresponding free node is not added to the graph.
  std::unordered_set<void*> memAllocNodePtrs_;
  static std::unordered_set<Graph*> graphSet_;
  static amd::Monitor graphSetLock_;
  Graph(hip::Device* device, const Graph* original = nullptr)
      : pOriginalGraph_(original), id_(nextID++), device_(device), graph_analysis_(nullptr) {
    amd::ScopedLock lock(graphSetLock_);
    graphSet_.insert(this);
    mem_pool_ = device->GetGraphMemoryPool();
    graphInstantiated_ = false;
    roots_.resize(DEBUG_HIP_FORCE_GRAPH_QUEUES);
    leafs_.resize(DEBUG_HIP_FORCE_GRAPH_QUEUES);
    wait_order_.resize(DEBUG_HIP_FORCE_GRAPH_QUEUES);
  }
  void RemoveUserObjectFromOwingGraphs(UserObject* uObj) {
    for (auto& g : uObj->owning_graphs_) {
      if (g != this) {
        g->RemoveUserObjGraph(uObj);
      }
    }
  }

  ~Graph();

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
    graphUserObj_.insert({pUserObj, 0});
  }
  // Increments graphUserObj_.second.
  void IncrementGraphUserObjRefCount(UserObject* pUserObj, unsigned int count) {
    amd::ScopedLock lock(graphSetLock_);
    auto it = graphUserObj_.find(pUserObj);
    if (it != graphUserObj_.end()) {
      it->second += count;
    }
  }
  // Decrements graphUserObj_.second.
  void DecrementGraphUserObjRefCount(UserObject* pUserObj, unsigned int count) {
    amd::ScopedLock lock(graphSetLock_);
    auto it = graphUserObj_.find(pUserObj);
    if (it != graphUserObj_.end()) {
      it->second -= count;
    }
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

  //! Schedules one node on a vitual stream.
  //! It will also process the nodes in edges, using recursion
  void ScheduleOneNode(
    Node node,      //!< Node for scheduling on a virtual stream
    int stream_id   //!< Current active virtual stream to use for scheduling
    );

  //! Schedules all nodes in the graph into different streams
  void ScheduleNodes();

  //! Update streams for the graph execution
  void UpdateStreams(
    hip::Stream* launch_stream, //!< Launch stream from the application
    const std::vector<hip::Stream*>& parallel_stream  //!< The list of parallel streams
  );

  //! Runs one node on the assigned stream
  bool RunOneNode(
    Node node,    //!< Node for the execution on GPU
    bool wait     //!< Wait dependencies
    );

  //! Runs all nodes from the execution graph on the assigned streams
  bool RunNodes(
    int32_t base_stream = 0,  //!< The base stream to run the graph on
    const std::vector<hip::Stream*>* streams = nullptr,  //!< Streams to run the graph
    const amd::Command::EventWaitList* parent_waitlist = nullptr //!< Parent Graph waitlist
  );

  bool TopologicalOrder(std::vector<Node>& TopoOrder);

  void clone(Graph* newGraph, bool cloneNodes = false) const;
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

  bool IsValidAllocation(void* ptr) const {
    return mem_pool_->IsValidAllocation(ptr);
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

    // Set Access to read write for all devices.
    for (size_t dev_idx = 0; dev_idx < g_devices.size(); ++dev_idx) {
      amd::Device* device = g_devices[dev_idx]->devices()[0];
      device->SetMemAccess(ptr, size, amd::Device::VmmAccess::kReadWrite);
    }

    return ptr;
  }

  void FreeAddress(void* ptr) const {
    // Single Free would free for all devices.
    g_devices[0]->devices()[0]->virtualFree(ptr);
  }

  amd::Command* GetSlabMapCommand(amd::GraphSlab* slab, amd::HostQueue& queue) {
    return mem_pool_->GetSlabMapCommand(slab, queue);
  }

  amd::Command* GetSlabUnmapCommand(amd::GraphSlab* slab, amd::HostQueue& queue) {
    return mem_pool_->GetSlabUnmapCommand(slab, queue);
  }

  amd::Command* GetAllocationCommand(amd::GraphSlab* slab, amd::HostQueue& queue, amd::Command* slab_map_command, size_t size, size_t offset) {
    return mem_pool_->GetAllocationCommand(slab, queue, slab_map_command, size, offset);
  }

  amd::GraphSlab* AllocateSlab(size_t size, size_t num_peers, size_t slab_id) const {
    return mem_pool_->AllocateSlab(size, num_peers, slab_id);
  }

  amd::GraphSlab* FetchSlab(size_t slab_id) {
    return mem_pool_->FetchSlab(slab_id);
  }

  void FreeSlab(amd::GraphSlab* slab) const {
    mem_pool_->FreeSlab(slab);
  }

  void* AllocateTemporaryHandle() const {
    return mem_pool_->AllocateTemporaryHandle();
  }

  void InvalidateTemporaryHandle() const {
    mem_pool_->InvalidateTemporaryHandle();
  }

  void FreeAllMemory(hip::Stream* stream) const {
    mem_pool_->FreeAllMemory(*stream);
  }

  bool IsGraphInstantiated() const {
    return graphInstantiated_;
  }

  void SetGraphInstantiated(bool graphInstantiate) {
    graphInstantiated_ = graphInstantiate;
  }

  //! returns count of unreleased memalloc nodes
  uint32_t GetMemAllocNodeCount() const { return memalloc_nodes_; }
  //! Increments the graph memory alloc node count
  void IncrementMemAllocNodeCount() { memalloc_nodes_++; }
  //! Decrements the graph memory alloc node count
  void DecrementMemAllocNodeCount() { memalloc_nodes_--; }
  //! returns device object
  hip::Device* Device() { return device_; }

  bool RunGraphAnalysis();
  
  void InvalidateGraphAnalysis();

  void AllocationSizeChanged(GraphNode* node, size_t previous_size, size_t new_size);

  void UpdatePtrs(void* old_ptr, void** new_ptr);

 protected:
  int max_streams_ = 0;  //!< Maximum number of streams used in the graph launch

 private:
  friend class GraphExec;
  std::vector<Node> vertices_;
  const Graph* pOriginalGraph_ = nullptr;
  //!< graphUserObj_.second stores refcount owned by this graph for user object,
  std::unordered_map<UserObject*, int> graphUserObj_;
  unsigned int id_;
  static int nextID;
  uint32_t memalloc_nodes_ = 0;  //!< Count of unreleased Memalloc nodes
  std::vector<Node> roots_;      //!< Root nodes, used in parallel launches
  std::vector<Node> leafs_;      //!< The list of leaf nodes on every parallel stream
  //!< Used as a temporary storage for the waiting nodes
  //!< to reduce the stack pressure in recursion
  std::vector<Node> wait_order_;
  std::vector<hip::Stream*> streams_;  //!< The list of streams, used in the execution
  int32_t current_id_ = 0;             //!< The current node ID in the graph execution sequence
  hip::Device* device_;                //!< HIP device object
  amd::GraphVmHeapArray* mem_pool_;    //!< Memory pool, associated with this graph
  std::unordered_set<GraphNode*> capturedNodes_;
  bool graphInstantiated_;
  std::unordered_map<Node, Node> clonedNodes_;
  ga::GraphAnalysis* graph_analysis_;
};

class GraphExec : public amd::ReferenceCountedObject, public Graph {
 public:
  static std::unordered_set<GraphExec*> graphExecSet_;
  static amd::Monitor graphExecSetLock_;
  GraphExec(uint64_t flags = 0)
      : ReferenceCountedObject(), Graph(hip::getCurrentDevice()), flags_(flags) {
    amd::ScopedLock lock(graphExecSetLock_);
    graphExecSet_.insert(this);
  }

  ~GraphExec() {
    for (auto stream : parallel_streams_) {
      if (stream != nullptr) {
        constexpr bool kForceDestroy = true;
        hip::Stream::Destroy(stream, kForceDestroy);
      }
    }
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      if (kernArgManager_ != nullptr) {
        kernArgManager_->release();
      }
    }
    if (instantiateDeviceId_ != -1) {
      static_cast<ReferenceCountedObject*>(g_devices[instantiateDeviceId_])->release();
    }
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

  //! Check if kernel node has hidden heap
  bool HasHiddenHeap() const { return hasHiddenHeap_; }
  //! Graph has nodes that require hidden heap.
  void SetHiddenHeap() { hasHiddenHeap_ = true; }

  //! Check executable graphs validity
  static bool isGraphExecValid(GraphExec* pGraphExec);
  std::vector<Node>& GetNodes() { return topoOrder_; }
  uint64_t GetFlags() const { return flags_; }
  hipError_t Init();
  hipError_t CreateStreams(uint32_t num_streams);
  hipError_t Run(hip::Stream* stream);
  // Capture GPU Packets from graph commands
  hipError_t CaptureAQLPackets();
  hipError_t UpdateAQLPacket(hip::GraphNode* node);
  // Kenrel arg manger is for the entire graph.
  // Child graph also shares the same kernel arg manager object. some apps have 100's of
  // child graph nodes and each child graph has only one node.
  void SetKernelArgManager(GraphKernelArgManager* kernArgManager) {
    kernArgManager_ = kernArgManager;
  }
  GraphKernelArgManager* GetKernelArgManager() {
    return kernArgManager_;
  }
  static void DecrementRefCount(cl_event event, cl_int command_exec_status, void* user_data);
  hipError_t AllocKernelArgForGraphNode();
  void GetKernelArgSizeForGraph(size_t& kernArgSizeForGraph);
  hipError_t EnqueueGraphWithSingleList(hip::Stream* hip_stream);
  bool TopologicalOrder() { return Graph::TopologicalOrder(topoOrder_); }

 protected:
  //! Topological order of the graph doesn't include nodes embedded as part of the child graph
  std::vector<Node> topoOrder_;
  std::vector<hip::Stream*> parallel_streams_;
  uint64_t flags_ = 0;
  GraphKernelArgManager* kernArgManager_ = nullptr;  //!< Kernel Arg manager for graph.
  int instantiateDeviceId_ = -1;
  bool hasHiddenHeap_ = false;  //!< Hidden heap indicator for Kernel node
  bool repeatLaunch_ = false;
};

class ChildGraphNode : public GraphNode, public GraphExec {
 public:
  ChildGraphNode(Graph* g) : GraphNode(hipGraphNodeTypeGraph, "solid", "rectangle"), GraphExec() {
    g->clone(this);
    graphCaptureStatus_ = false;
  }

  ChildGraphNode(const ChildGraphNode& rhs) : GraphNode(rhs), GraphExec() {
    rhs.Graph::clone(this);
    graphCaptureStatus_ = rhs.graphCaptureStatus_;
  }

  GraphNode* clone() const override { return new ChildGraphNode(*this); }

  Graph* GetChildGraph() override { return this; }

  void SetGraphCaptureStatus(bool status) { graphCaptureStatus_ = status; }

  bool GetGraphCaptureStatus() { return graphCaptureStatus_; }

  std::vector<Node>& GetChildGraphNodeOrder() {
    return topoOrder_;
  }

  void SetStream(hip::Stream* stream) override {
    stream_ = stream;
  }

  bool TopologicalOrder(std::vector<Node>& TopoOrder) override {
    return Graph::TopologicalOrder(TopoOrder);
  }

  void EnqueueCommands(hip::Stream* stream) override {
    if (graphCaptureStatus_) {
      hipError_t status = EnqueueGraphWithSingleList(stream);
    } else if (max_streams_ == 1) {
      for (int i = 0; i < topoOrder_.size(); i++) {
        topoOrder_[i]->SetStream(stream_);
        hipError_t status =
            topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
        topoOrder_[i]->EnqueueCommands(stream_);
      }
    }
  }

  hipError_t SetParams(const Graph* childGraph) {
    const std::vector<Node>& newNodes = childGraph->GetNodes();
    const std::vector<Node>& oldNodes = Graph::GetNodes();
    for (std::vector<Node>::size_type i = 0; i != newNodes.size(); i++) {
      hipError_t status = oldNodes[i]->SetParams(newNodes[i]);
      if (status != hipSuccess) {
        return status;
      }
    }

    auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

    return hipSuccess;
  }

  hipError_t SetParams(GraphNode* node) override {
    const ChildGraphNode* childGraphNode = static_cast<ChildGraphNode const*>(node);
    return SetParams((Graph*)this);
  }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    return std::to_string(GraphNode::GetID()) + "\n" + "graph_" + std::to_string(Graph::GetID());
  }

  virtual void GenerateDOT(std::ostream& fout, hipGraphDebugDotFlags flag) override {
    Graph::GenerateDOT(fout, flag);
  }

 private:
  bool graphCaptureStatus_;
};

class GraphKernelNode : public GraphNode {
  hipKernelNodeParams kernelParams_;   //!< Kernel node parameters
  unsigned int numParams_;             //!< No. of kernel params as part of signature
  hipKernelNodeAttrValue kernelAttr_;  //!< Kernel node attributes
  unsigned int kernelAttrInUse_;       //!< Kernel attributes in use
  ihipExtKernelEvents kernelEvents_;   //!< Events for Ext launch kernel
  bool hasHiddenHeap_;                 //!< Kernel has hidden heap(device side allocation)
  int coopKernel_;                     //!< Launch cooperative kernel

 public:
  bool HasHiddenHeap() const { return hasHiddenHeap_; }
  void EnqueueCommands(hip::Stream* stream) override {
    // If the node is disabled it becomes empty node. To maintain ordering just enqueue marker.
    // Node can be enabled/disabled only for kernel, memcpy and memset nodes.
    if (!isEnabled_) {
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
      hipFunction_t func = getFunc(kernelParams_, dev_id_);
      hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
      amd::Kernel* kernel = function->kernel();
      amd::ScopedLock lock(function->dflock_);
      command->enqueue();
      command->release();
    }
  }

  virtual std::pair<std::set<void*>, std::set<void*>> Values() const final {
    std::set<void*> defs;
    std::set<void*> uses;

    amd::Kernel* kernel = hip::DeviceFunc::asFunction(getFunc(kernelParams_, dev_id_))->kernel();
    auto device = g_devices[dev_id_]->devices()[0];
    device::Kernel* devKernel = const_cast<device::Kernel*>(kernel->getDeviceKernel(*device));
    auto signature = devKernel->signature();
    auto kernel_param_descriptors = signature.params();

    std::vector<bool> read_only(numParams_);
    size_t count = 0;
    for (auto& param : kernel_param_descriptors) {
      if (param.info_.hidden_) {
	continue;
      }
      // FIXME: it seems that isReadOnlyByCompiler is wrong, at least it misses the case like this:
      // kernel(void *p)
      //   q = p
      //   *q = 0
      // p here would get marked isReadOnlyByCompiler, but not readOnly_. Is this intentional?

      //read_only[count++] = param.accessQualifier_ == CL_KERNEL_ARG_ACCESS_READ_ONLY;
      //read_only[count++] = param.info_.readOnly_ || param.info_.isReadOnlyByCompiler;
      read_only[count++] = param.info_.readOnly_;
    }

    auto graph = GetParentGraph();
    for (unsigned int i = 0; i < numParams_; ++i) {
      size_t offset = 0;
      auto ptr = *reinterpret_cast<void**>(kernelParams_.kernelParams[i]);
      auto memory = getMemoryObject(ptr, offset);
      if (memory != nullptr || (graph != nullptr && graph->IsValidAllocation(ptr))) {
	if (read_only[i]) {
	  uses.insert(ptr);
	} else {
	  uses.insert(ptr);
	  defs.insert(ptr);
	}
      }
    }

    return {defs, uses};
  }

  virtual void UpdatePtr(void* old_ptr, void** new_ptr) final {
    amd::Kernel* kernel = hip::DeviceFunc::asFunction(getFunc(kernelParams_, dev_id_))->kernel();
    const amd::KernelSignature& signature = kernel->signature();
    for (unsigned int i = 0; i < numParams_; ++i) {
      auto ptr = *reinterpret_cast<void**>(kernelParams_.kernelParams[i]);
      if (ptr == old_ptr) {
        const amd::KernelParameterDescriptor& desc = signature.at(i);
        ::memcpy(kernelParams_.kernelParams[i], new_ptr, desc.size_);
      }
    }
  }

  void PrintAttributes(std::ostream& out, hipGraphDebugDotFlags flag) override {
    out << "[";
    out << "style";
    out << "=\"";
    out << style_;
    (flag == hipGraphDebugDotFlagsKernelNodeParams ||
     flag == hipGraphDebugDotFlagsKernelNodeAttributes)
        ? out << "\n"
        : out << "\"";
    out << "shape";
    out << "=\"";
    out << GetShape(flag);
    out << "\"";
    out << "label";
    out << "=\"";
    out << GetLabel(flag);
    if (DEBUG_HIP_GRAPH_DOT_PRINT) {
      out << "StreamId:" << stream_id_;
      out << "\nSignalIsRequired: " << ((signal_is_required_) ? "true" : "false");
      out << "\nDeviceId:" << dev_id_;
    }
    out << "\"";
    out << "];";
  }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    hipFunction_t func = getFunc(kernelParams_, dev_id_);
    hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
    std::string label;
    char buffer[4096];
    if (flag == hipGraphDebugDotFlagsVerbose) {
      sprintf(buffer,
              "{\n%s\n| {ID | %d | %s\\<\\<\\<(%u,%u,%u),(%u,%u,%u),%u\\>\\>\\>}\n| {{node "
              "handle | func handle} | {%p | %p}}\n| {accessPolicyWindow | {base_ptr | num_bytes | "
              "hitRatio | hitProp | missProp} | {%p | %zu | %f | %d | %d}}\n| {cooperative | "
              "%u}\n| {priority | %d}\n}",
              label_, GetID(), function->name().c_str(), kernelParams_.gridDim.x,
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
              label_, GetID(), function->name().c_str(),
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

  const char* GetShape(hipGraphDebugDotFlags flag) override {
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
    hipFunction_t func = getFunc(*pNodeParams, dev_id_);
    if (!func) {
      return hipErrorInvalidDeviceFunction;
    }
    hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
    amd::Kernel* kernel = function->kernel();
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      auto device = g_devices[dev_id_]->devices()[0];
      device::Kernel* devKernel = const_cast<device::Kernel*>(kernel->getDeviceKernel(*device));
      kernargSegmentByteSize_ = devKernel->KernargSegmentByteSize();
      kernargSegmentAlignment_ = devKernel->KernargSegmentAlignment();
      alignedKernArgSize_ =
          amd::alignUp(devKernel->KernargSegmentByteSize(), devKernel->KernargSegmentAlignment());
    }
    const amd::KernelSignature& signature = kernel->signature();
    numParams_ = signature.numParameters();

    // Copy gridDim, blockDim, sharedMemBytes and func
    kernelParams_ = *pNodeParams;

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

  GraphKernelNode(const hipKernelNodeParams* pNodeParams, const ihipExtKernelEvents* pEvents,
                  int coopKernel = 0)
      : GraphNode(hipGraphNodeTypeKernel, "bold", "octagon", "KERNEL") {
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
    coopKernel_ = coopKernel;
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
    coopKernel_ = rhs.coopKernel_;
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

  GraphNode* clone() const override { return new GraphKernelNode(*this); }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipFunction_t func = getFunc(kernelParams_, dev_id_);
    if (!func) {
      return hipErrorInvalidDeviceFunction;
    }
    hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
    amd::Kernel* kernel = function->kernel();
    amd::ScopedLock lock(function->dflock_);
    hipError_t status = validateKernelParams(&kernelParams_, func, dev_id_);
    if (hipSuccess != status) {
      return status;
    }
    status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command;
    uint32_t flags = 0;
    if (DEBUG_HIP_FORCE_ASYNC_QUEUE) {
      // If there is one dependency, but many edges, then execute this node in any order
      if (((dependencies_.size() == 1) && (dependencies_[0]->GetEdges().size() > 1) &&
          (DEBUG_HIP_FORCE_GRAPH_QUEUES == 1))) {
        // Makes sure the first node in the edges will have a barrier always
        if (dependencies_[0]->GetEdges()[0] != this) {
          flags = hipExtAnyOrderLaunch;
        }
      }
    }

    amd::HIPLaunchParams launch_params(kernelParams_.gridDim.x, kernelParams_.gridDim.y,
                                       kernelParams_.gridDim.z, kernelParams_.blockDim.x,
                                       kernelParams_.blockDim.y, kernelParams_.blockDim.z,
                                       kernelParams_.sharedMemBytes);

    if (!launch_params.IsValidConfig()) {
      return hipErrorInvalidConfiguration;
    }

    status = ihipLaunchKernelCommand(
        command, func, launch_params, stream, kernelParams_.kernelParams, kernelParams_.extra,
        kernelEvents_.startEvent_, kernelEvents_.stopEvent_, flags, coopKernel_, 0, 0, 0, 0, 0);
    if (signal_is_required_) {
      // Optimize the barriers by adding a signal into the dispatch packet directly
      command->SetProfiling();
    }
    commands_.emplace_back(command);
    return status;
  }

  void GetParams(hipKernelNodeParams* params) { *params = kernelParams_; }

  hipError_t SetParams(const hipKernelNodeParams* params) {
    hipFunction_t func = getFunc(kernelParams_, dev_id_);
    if (!func) {
      return hipErrorInvalidDeviceFunction;
    }
    // updates kernel params
    hipError_t status = validateKernelParams(params, func, dev_id_);
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
    status = copyParams(params);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to set params");
    } else {
      //auto graph = GetParentGraph();
      //if (graph != nullptr) {
      //  graph->InvalidateGraphAnalysis();
      //}
    }
    return status;
  }

  hipError_t SetAttrParams(hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* params) {
    hipDeviceProp_t prop = {0};
    hipError_t status = ihipGetDeviceProperties(&prop, dev_id_);
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
                                         hipFunction_t func, int devId) {

    amd::HIPLaunchParams launch_params(pNodeParams->gridDim.x, pNodeParams->gridDim.y,
                                       pNodeParams->gridDim.z, pNodeParams->blockDim.x,
                                       pNodeParams->blockDim.y, pNodeParams->blockDim.z,
                                       pNodeParams->sharedMemBytes);

    if (!launch_params.IsValidConfig()) {
      HIP_RETURN(hipErrorInvalidConfiguration);
    }

    hipError_t status = ihipLaunchKernel_validate(func, launch_params, pNodeParams->kernelParams,
                                                  pNodeParams->extra, devId, 0);
    if (status != hipSuccess) {
      return status;
    }
    return hipSuccess;
  }

  virtual bool GraphCaptureEnabled() override {
    bool isGraphCapture = false;
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      // Disable capture for cooperative kernels
      if (!coopKernel_) {
        isGraphCapture = true;
      }
    }
    return isGraphCapture;
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

  GraphNode* clone() const override { return new GraphMemcpyNode(*this); }

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
    auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}
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
      char buffer[4096];
      sprintf(
          buffer,
          "{\n%s\n| {{ID | node handle} | {%u | %p}}\n| {kind | %s}\n| {{srcPtr | dstPtr} | "
          "{pitch "
          "| ptr | xsize | ysize | pitch | ptr | xsize | size} | {%zu | %p | %zu | %zu | %zu | %p "
          "| %zu "
          "| %zu}}\n| {{srcPos | {{x | %zu} | {y | %zu} | {z | %zu}}} | {dstPos | {{x | %zu} | {y "
          "| "
          "%zu} | {z | %zu}}} | {Extent | {{Width | %zu} | {Height | %zu} | {Depth | %zu}}}}\n}",
          label_, GetID(), this, memcpyDirection.c_str(), copyParams_.srcPtr.pitch,
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
  const char* GetShape(hipGraphDebugDotFlags flag) override {
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }
  virtual bool GraphCaptureEnabled() override {
    bool isGraphCapture = false;
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      switch (copyParams_.kind) {
        case hipMemcpyDeviceToDevice:
          isGraphCapture = true;
          break;
        default:
          break;
      }
    }
    return isGraphCapture;
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

  GraphMemcpyNode1D(const GraphMemcpyNode1D& rhs) : GraphMemcpyNode(rhs) {
    dst_ = rhs.dst_;
    src_ = rhs.src_;
    count_ = rhs.count_;
    kind_ = rhs.kind_;
  }

  GraphNode* clone() const override { return new GraphMemcpyNode1D(*this); }

  virtual std::pair<std::set<void*>, std::set<void*>> Values() const final {
    std::set<void*> defs;
    std::set<void*> uses;
    auto graph = GetParentGraph();
    {
    size_t offset = 0;
    auto memory = getMemoryObject(dst_, offset);
    if (memory != nullptr || (graph != nullptr && graph->IsValidAllocation(dst_))) {
      defs.insert(dst_);
      uses.insert(dst_);
    }
    }
    {
    size_t offset = 0;
    auto memory = getMemoryObject(src_, offset);
    if (memory != nullptr || (graph != nullptr && graph->IsValidAllocation(const_cast<void*>(src_)))) {
      defs.insert(const_cast<void*>(src_));
      uses.insert(const_cast<void*>(src_));
    }
    }
    return {defs, uses};
  }

  virtual void UpdatePtr(void* old_ptr, void** new_ptr) final {
    if (dst_ == old_ptr) {
      dst_ = *new_ptr;
    }
    if (src_ == old_ptr) {
      src_ = *new_ptr;
    }
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = hip::GraphMemcpyNode1D::ValidateParams(dst_, src_, count_, kind_);
    if (status != hipSuccess) {
      return status;
    }

    if ((kind_ == hipMemcpyHostToHost || kind_ == hipMemcpyDefault) && IsHtoHMemcpy(dst_, src_)) {
      return hipSuccess;
    }
    status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    commands_.reserve(1);
    amd::Command* command = nullptr;
    if (!AMD_DIRECT_DISPATCH) {
      WorkerThreadLock_.lock();
    }
    status = ihipMemcpyCommand(command, dst_, src_, count_, kind_, *stream);
    hip::MemcpyType type = ihipGetMemcpyType(src_, dst_, kind_);
    if (type == hipCopyBuffer) {
      amd::CopyMemoryCommand* cpycmd = reinterpret_cast<amd::CopyMemoryCommand*>(command);
      amd::CopyMetadata copyMetadata = cpycmd->copyMetadata();
      copyMetadata.copyEnginePreference_ = amd::CopyMetadata::CopyEnginePreference::BLIT;
      cpycmd->SetCopyMetadata(copyMetadata);
    }
    if (!AMD_DIRECT_DISPATCH) {
      WorkerThreadLock_.unlock();
    }
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
    dst_ = dst;
    src_ = src;
    count_ = count;
    kind_ = kind;

    auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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
      char buffer[4096];
      sprintf(
          buffer,
          "{\n%s\n| {{ID | node handle} | {%u | %p}}\n| {kind | %s}\n| {{srcPtr | dstPtr} | "
          "{pitch "
          "| ptr | xsize | ysize | pitch | ptr | xsize | size} | {%zu | %p | %zu | %zu | %zu | %p "
          "| %zu "
          "| %zu}}\n| {{srcPos | {{x | %zu} | {y | %zu} | {z | %zu}}} | {dstPos | {{x | %zu} | {y "
          "| "
          "%zu} | {z | %zu}}} | {Extent | {{Width | %zu} | {Height | %zu} | {Depth | %zu}}}}\n}",
          label_, GetID(), this, memcpyDirection.c_str(), (size_t)0, src_, (size_t)0,
          (size_t)0, (size_t)0, dst_, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (size_t)0,
          (size_t)0, (size_t)0, (size_t)0, count_, (size_t)1, (size_t)1);
      label = buffer;
    } else {
      label = std::to_string(GetID()) + "\n" + label_ + "\n(" + memcpyDirection + "," +
          std::to_string(count_) + ")";
    }
    return label;
  }
  const char* GetShape(hipGraphDebugDotFlags flag) override {
    if (flag == hipGraphDebugDotFlagsMemcpyNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      return "record";
    } else {
      return shape_;
    }
  }
  virtual bool GraphCaptureEnabled() override {
    bool isGraphCapture = false;
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      hip::MemcpyType type = ihipGetMemcpyType(src_, dst_, kind_);
      switch (type) {
        case hipCopyBuffer:
          isGraphCapture = true;
          break;
        default:
          break;
      }
    }
    return isGraphCapture;
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

  GraphMemcpyNodeFromSymbol(const GraphMemcpyNodeFromSymbol& rhs) : GraphMemcpyNode1D(rhs) {
     symbol_ = rhs.symbol_;
     offset_ = rhs.offset_;
  }

  GraphNode* clone() const override { return new GraphMemcpyNodeFromSymbol(*this); }

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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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

  GraphMemcpyNodeToSymbol(const GraphMemcpyNodeToSymbol& rhs) : GraphMemcpyNode1D(rhs) {
     symbol_ = rhs.symbol_;
     offset_ = rhs.offset_;
  }

  GraphNode* clone() const override { return new GraphMemcpyNodeToSymbol(*this); }

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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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
  size_t arrWidth_ = 1;
  size_t arrHeight_ = 1;
 public:
  GraphMemsetNode(const hipMemsetParams* pMemsetParams, size_t depth = 1, size_t arrWidth = 1,
                  size_t arrHeight = 1)
      : GraphNode(hipGraphNodeTypeMemset, "solid", "invtrapezium", "MEMSET") {
    memsetParams_ = *pMemsetParams;
    depth_ = depth;
    arrWidth_ = arrWidth;
    arrHeight_ = arrHeight;
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
    arrWidth_ = memsetNode.arrWidth_;
    arrHeight_ = memsetNode.arrHeight_;
  }

  GraphNode* clone() const override { return new GraphMemsetNode(*this); }

  virtual std::string GetLabel(hipGraphDebugDotFlags flag) override {
    std::string label;
    if (flag == hipGraphDebugDotFlagsMemsetNodeParams || flag == hipGraphDebugDotFlagsVerbose) {
      char buffer[4096];
      sprintf(buffer,
              "{\n%s\n| {{ID | node handle | dptr | pitch | value | elementSize | width | "
              "height | depth} | {%u | %p | %p | %zu | %u | %u | %zu | %zu | %zu}}}",
              label_, GetID(), this, memsetParams_.dst, memsetParams_.pitch,
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

  const char* GetShape(hipGraphDebugDotFlags flag) override {
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
    if (memsetParams_.height == 1 && depth_ == 1) {
      size_t sizeBytes = memsetParams_.width * memsetParams_.elementSize;
      hipError_t status = ihipMemsetCommand(commands_, memsetParams_.dst, memsetParams_.value,
                                            memsetParams_.elementSize, sizeBytes, stream);
    } else {
      hipError_t status = ihipMemset3DCommand(
          commands_,
          {memsetParams_.dst, memsetParams_.pitch, arrWidth_ * memsetParams_.elementSize,
           arrHeight_},
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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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

  GraphEventRecordNode(const GraphEventRecordNode& rhs) : GraphNode(rhs) {
    event_ = rhs.event_;
  }

  GraphNode* clone() const override { return new GraphEventRecordNode(*this); }

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
      hipError_t status = e->enqueueRecordCommand(stream, commands_[0]);
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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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

  GraphEventWaitNode(const GraphEventWaitNode& rhs) : GraphNode(rhs) {
    event_ = rhs.event_;
  }

  GraphNode* clone() const override { return new GraphEventWaitNode(*this); }

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
      commands_[0]->enqueue();
      commands_[0]->release();
    }
  }

  void GetParams(hipEvent_t* event) const { *event = event_; }

  hipError_t SetParams(hipEvent_t event) {
    event_ = event;

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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

  GraphNode* clone() const override { return new GraphHostNode(*this); }

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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

    return hipSuccess;
  }

  hipError_t SetParams(GraphNode* node) override {
    const GraphHostNode* hostNode = static_cast<GraphHostNode const*>(node);
    return SetParams(&hostNode->NodeParams_);
  }
};

// ================================================================================================
class GraphEmptyNode : public GraphNode {
 public:
  GraphEmptyNode() : GraphNode(hipGraphNodeTypeEmpty, "solid", "rectangle", "EMPTY") {}
  ~GraphEmptyNode() {}

  GraphNode* clone() const override { return new GraphEmptyNode(*this); }

  hipError_t CreateCommand(hip::Stream* stream) override {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    // If just one stream was forced for the execution, then the barrier can be skipped
    if (DEBUG_HIP_FORCE_GRAPH_QUEUES != 1) {
      amd::Command::EventWaitList waitList;
      commands_.reserve(1);
      amd::Command* command = new amd::Marker(*stream, !kMarkerDisableFlush, waitList);
      commands_.emplace_back(command);
    }
    return hipSuccess;
  }
};

// ================================================================================================
class GraphMemFreeNode : public GraphNode {
  void* device_ptr_;    // Device pointer of the freed memory
  size_t slab_id_ = 0;
  amd::GraphSlab* slab_ = nullptr;
  bool forwards_memory_ = false;

 public:
  GraphMemFreeNode(void* dptr)
    : GraphNode(hipGraphNodeTypeMemFree, "solid", "rectangle", "MEM_FREE")
    , device_ptr_(dptr) {}
  GraphMemFreeNode(const GraphMemFreeNode& rhs) : GraphNode(rhs) {
    device_ptr_ = rhs.device_ptr_;
  }

  virtual GraphNode* clone() const final { return new GraphMemFreeNode(*this); }

  virtual std::pair<std::set<void*>, std::set<void*>> Values() const final {
    return {{}, {device_ptr_}};
  }

  virtual void UpdatePtr(void* old_ptr, void** new_ptr) final {
    if (device_ptr_ == old_ptr) {
      device_ptr_ = *new_ptr;
    }
  }

  void SetSlabId(size_t slab_id) {
    slab_id_ = slab_id;
  }

  void ForwardsMemory() {
    forwards_memory_ = true;
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) final {
    auto error = GraphNode::CreateCommand(stream);
    if (!HIP_MEM_POOL_USE_VM) {
      Execute(stream_);
    } else {
      size_t offset = 0;
      auto memory = getMemoryObject(device_ptr_, offset);
      if (memory && !forwards_memory_) {
	  amd::MemObjMap::RemoveMemObj(device_ptr_);
	  memory->release();
      }

      auto graph = GetParentGraph();
      if (graph != nullptr) {
	slab_ = graph->FetchSlab(slab_id_);
	assert(slab_ != nullptr);
	if (slab_->DecrementRefcount() == 0) {
	  auto unmap_command = graph->GetSlabUnmapCommand(slab_, *stream);
	  if (nullptr != unmap_command) {
	    commands_.push_back(unmap_command);
	  }
	}
	graph->DecrementMemAllocNodeCount();
      }
    }
    return error;
  }

  void Execute(hip::Stream* stream) {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      size_t offset = 0;
      auto memory = getMemoryObject(device_ptr_, offset);
      if (memory != nullptr && memory->getUserData().data != nullptr) {
	slab_ = reinterpret_cast<amd::GraphSlab*>(memory->getUserData().data);
	auto unmap_command = graph->GetSlabUnmapCommand(slab_, *stream);
	if (nullptr != unmap_command) {
	  unmap_command->enqueue();
	  unmap_command->awaitCompletion();
	  unmap_command->release();
	}
      }
    }
  }

  void GetParams(void** params) const {
    *params = device_ptr_;
  }

  hipError_t SetParams(hipMemFreeNodeParams *params) {
    size_t offset = 0;
    auto memory = getMemoryObject(params->dptr, offset);
    if (memory == nullptr) {
      if (HIP_MEM_POOL_USE_VM) {
	// When VM is on the address must be valid and may point to a VA object
	memory = amd::MemObjMap::FindVirtualMemObj(params->dptr);
      }
      if (memory == nullptr) {
	return hipErrorInvalidValue;
      }
    }

    // FIXME: more verification. Technically, as it is now, this breaks the invariant
    // that every alloc node has a corresponding free node since here dptr could be
    // any (valid) memory object

    device_ptr_ = params->dptr;

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

    return hipSuccess;
  }
};

// ================================================================================================
class GraphMemAllocNode final : public GraphNode {
  hipMemAllocNodeParams node_params_;  // Node parameters for memory allocation
  std::vector<GraphMemAllocNode*> peers_;
  amd::GraphSlab* slab_ = nullptr;
  size_t slab_id_;
  size_t slab_size_;
  size_t slab_offset_;
  void* temporary_handle_ = nullptr;

 public:
  GraphMemAllocNode(const hipMemAllocNodeParams* node_params)
      : GraphNode(hipGraphNodeTypeMemAlloc, "solid", "rectangle", "MEM_ALLOC") {
    node_params_ = *node_params;
  }

  GraphMemAllocNode(const GraphMemAllocNode& rhs)
      : GraphNode(rhs) {
    node_params_ = rhs.node_params_;
    temporary_handle_ = rhs.temporary_handle_;
    slab_ = rhs.slab_;
    if (HIP_MEM_POOL_USE_VM) {
      assert(rhs.node_params_.dptr != nullptr && "Graph MemAlloc runtime can't clone an invalid node!");
      node_params_.dptr = rhs.node_params_.dptr;
    }
  }

  virtual ~GraphMemAllocNode() final {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      if (slab_ != nullptr) {
	graph->FreeSlab(slab_);
      } else {
	graph->InvalidateTemporaryHandle();
      }
    }
  }

  virtual GraphNode* clone() const final { return new GraphMemAllocNode(*this); }

  virtual std::pair<std::set<void*>, std::set<void*>> Values() const final {
    return {{node_params_.dptr}, {}};
  }

  size_t Bytesize() const {
    return node_params_.bytesize;
  }

  void SetSlabInfo(size_t slab_id, size_t slab_size, std::vector<GraphMemAllocNode*> peers, size_t slab_offset) {
    slab_ = nullptr;
    slab_id_ = slab_id;
    slab_size_ = slab_size;
    peers_ = peers;
    slab_offset_ = slab_offset;
  }

  void InvalidateSlab() {
    slab_ = nullptr;
  }

  void ReleaseSlab() {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      if (slab_ && slab_->Busy()) {
	graph->FreeSlab(slab_);
	for (auto peer : peers_) {
	  peer->InvalidateSlab();
	}
      }
    }
  }

  virtual hipError_t CreateCommand(hip::Stream* stream) final {
    auto error = GraphNode::CreateCommand(stream);
    if (!HIP_MEM_POOL_USE_VM) {
      auto ptr = Execute(stream_);
    } else {
      auto graph = GetParentGraph();

      if (graph != nullptr) {
	slab_ = graph->AllocateSlab(slab_size_, peers_.size(), slab_id_);
	assert(slab_ != nullptr);
	amd::Command *slab_map_command = nullptr;
	if (!slab_->IsMapped()) {
	  slab_map_command = graph->GetSlabMapCommand(slab_, *stream);
	}

	node_params_.dptr = (void*) ((size_t) slab_->MemPtr() + slab_offset_);
	graph->UpdatePtrs(temporary_handle_, &node_params_.dptr);

	if (slab_map_command != nullptr) {
	  commands_.push_back(slab_map_command);
	}
	commands_.push_back(graph->GetAllocationCommand(slab_, *stream, slab_map_command, node_params_.bytesize, slab_offset_));

	graph->IncrementMemAllocNodeCount();
      }
    }
    return error;
  }

  void* ReserveAddress() {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      temporary_handle_ = graph->AllocateTemporaryHandle();
      node_params_.dptr = temporary_handle_;
    }
    return node_params_.dptr;
  }

  void* Execute(hip::Stream* stream = nullptr) {
    auto graph = GetParentGraph();
    if (graph != nullptr) {
      slab_size_ = node_params_.bytesize;
      slab_offset_ = 0;
      slab_ = graph->AllocateSlab(slab_size_, 1, 0);

      auto slab_map_command = graph->GetSlabMapCommand(slab_, *stream);
      if (slab_map_command) {
	slab_map_command->enqueue();
      }
      node_params_.dptr = (void*) ((size_t) slab_->MemPtr() + slab_offset_);
      auto allocation_command = graph->GetAllocationCommand(slab_, *stream, slab_map_command, node_params_.bytesize, slab_offset_);
      assert(allocation_command);
      allocation_command->enqueue();

      if (slab_map_command) {
	slab_map_command->awaitCompletion();
	slab_map_command->release();
      }
      allocation_command->awaitCompletion();
      allocation_command->release();
    }
    return node_params_.dptr;
  }

  void GetParams(hipMemAllocNodeParams* params) const {
    std::memcpy(params, &node_params_, sizeof(hipMemAllocNodeParams));
  }

  hipError_t SetParamsInternal(hipMemAllocNodeParams *params) {
    auto graph = GetParentGraph();
    if (graph == nullptr) {
      return hipSuccess;
    }

    size_t prev_bytesize = node_params_.bytesize;

    if (!HIP_MEM_POOL_USE_VM) {
      graph->FreeSlab(slab_);
      node_params_.dptr = Execute();
    } else {
      // Two cases

      if (slab_ == nullptr) {
	// 1. The node hasn't been run yet, in which case do nothing
	std::memcpy(&node_params_, params, sizeof(hipMemAllocNodeParams));
      } else {
	// 2. The node has been run, release the slab through the leader 
	ReleaseSlab();
	std::memcpy(&node_params_, params, sizeof(hipMemAllocNodeParams));
      }
      node_params_.dptr = temporary_handle_;
    }

    //graph->InvalidateGraphAnalysis();

    if (node_params_.bytesize != prev_bytesize) {
      graph->AllocationSizeChanged(this, prev_bytesize, node_params_.bytesize);
    }

    return hipSuccess;
  }

  hipError_t SetParams(hipMemAllocNodeParams *params) {
    if (params->bytesize == 0 ||
	params->poolProps.allocType != hipMemAllocationTypePinned ||
	params->poolProps.location.type != hipMemLocationTypeDevice) {
      params->dptr = nullptr;
      return hipErrorInvalidValue;
    }
    if (params->poolProps.location.type == hipMemLocationTypeDevice) {
      if (params->poolProps.location.id < 0 ||
	  params->poolProps.location.id >= g_devices.size()) {
	return hipErrorInvalidValue;
      }
    }

    return SetParamsInternal(params);
  }
};

// ================================================================================================
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

  GraphNode* clone() const override { return new GraphDrvMemcpyNode(*this); }

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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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

  GraphNode* clone() const override { return new hipGraphExternalSemSignalNode(*this); }

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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

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

  GraphNode* clone() const override { return new hipGraphExternalSemWaitNode(*this); }

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

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

    return hipSuccess;
  }
};

class hipGraphBatchMemOpNode : public GraphNode {
  hipBatchMemOpNodeParams batchMemOpNodeParam_;

 public:
  hipGraphBatchMemOpNode(const hipBatchMemOpNodeParams* pNodeParams)
      : GraphNode(hipGraphNodeTypeBatchMemOp, "solid", "rectangle", "BATCH_MEM_OP_NODE") {
    batchMemOpNodeParam_ = *pNodeParams;
  }

  hipGraphBatchMemOpNode(const hipGraphBatchMemOpNode& rhs) : GraphNode(rhs) {
    batchMemOpNodeParam_ = rhs.batchMemOpNodeParam_;
  }
  ~hipGraphBatchMemOpNode() {}

  GraphNode* clone() const override { return new hipGraphBatchMemOpNode(*this); }

  hipError_t CreateCommand(hip::Stream* stream) {
    hipError_t status = GraphNode::CreateCommand(stream);
    if (status != hipSuccess) {
      return status;
    }
    amd::Command::EventWaitList waitList;
    amd::BatchMemoryOperationCommand* command = new amd::BatchMemoryOperationCommand(
        *stream, ROCCLR_COMMAND_BATCH_STREAM, batchMemOpNodeParam_.count,
        batchMemOpNodeParam_.flags, waitList, batchMemOpNodeParam_.paramArray,
        sizeof(hipStreamBatchMemOpParams));
    if (command == nullptr) {
      return hipErrorOutOfMemory;
    }
    commands_.emplace_back(command);
    return hipSuccess;
  }

  void GetParams(hipBatchMemOpNodeParams* pNodeParams) const {
    std::memcpy(pNodeParams, &batchMemOpNodeParam_, sizeof(hipBatchMemOpNodeParams));
  }

  hipError_t SetParams(const hipBatchMemOpNodeParams* pNodeParams) {
    std::memcpy(&batchMemOpNodeParam_, pNodeParams, sizeof(hipBatchMemOpNodeParams));

    //auto graph = GetParentGraph();
    //if (graph != nullptr) {
    //  graph->InvalidateGraphAnalysis();
    //}

    return hipSuccess;
  }
};

}  // namespace hip
