#include "hip_graph_modifier.hpp"
#include "hip/hip_runtime_api.h"
#include "hip_graph_fuse_recorder.hpp"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "hip_graph_internal.hpp"
#include <yaml-cpp/yaml.h>


namespace {
void loadExternalSymbols(const std::vector<std::pair<std::string, std::string>>& fusedGroups) {
  HIP_INIT_VOID();
  for (auto& [symbolName, imagePath] : fusedGroups) {
    hip::PlatformState::instance().loadExternalSymbol(symbolName, imagePath);
  }
}

dim3 max(dim3& one, dim3& two) {
  return dim3(std::max(one.x, two.x), std::max(one.y, two.y), std::max(one.z, two.z));
}


class FusionGroup : public hip::GraphNode {
 public:
  FusionGroup() : hip::GraphNode(hipGraphNodeTypeKernel) {
    semaphore_ = hip::PlatformState::instance().getSemaphore();
  }
  virtual ~FusionGroup() { delete fusedNode_; };
  void addNode(hip::GraphKernelNode* node) { fusee_.push_back(node); }
  std::vector<hip::GraphKernelNode*>& getNodes() { return fusee_; }
  hip::GraphKernelNode* getHead() { return fusee_.empty() ? nullptr : fusee_.front(); }
  hip::GraphKernelNode* getTail() { return fusee_.empty() ? nullptr : fusee_.back(); }

  hip::GraphKernelNode* getFusedNode() { return fusedNode_; }
  void generateNode(void* functionHandle);
  hipKernelNodeParams* getNodeParams() { return &fusedNodeParams_; }

 private:
  hip::GraphNode* clone() const override { return nullptr; }
  std::vector<hip::GraphKernelNode*> fusee_{};
  hip::GraphKernelNode* fusedNode_;
  hipKernelNodeParams fusedNodeParams_{};
  std::vector<void*> kernelArgs_{};
  std::vector<void*> hiddenkernelArgs_{};
  void* semaphore_{};
};

void FusionGroup::generateNode(void* functionHandle) {
  fusedNodeParams_.blockDim = dim3(0, 0, 0);
  fusedNodeParams_.gridDim = dim3(0, 0, 0);
  fusedNodeParams_.sharedMemBytes = 0;
  fusedNodeParams_.func = functionHandle;

  for (auto* node : fusee_) {
    auto nodeParams = hip::GraphFuseRecorder::getKernelNodeParams(node);
    fusedNodeParams_.blockDim = max(fusedNodeParams_.blockDim, nodeParams.blockDim);
    fusedNodeParams_.gridDim = max(fusedNodeParams_.gridDim, nodeParams.gridDim);
    fusedNodeParams_.sharedMemBytes =
        std::max(fusedNodeParams_.sharedMemBytes, nodeParams.sharedMemBytes);

    auto* kernel = hip::GraphFuseRecorder::getDeviceKernel(nodeParams);
    const auto numKernelArgs = kernel->signature().numParametersAll();

    for (size_t i = 0; i < numKernelArgs; ++i) {
      if (kernel->signature().at(i).info_.hidden_) hiddenkernelArgs_.push_back(nodeParams.kernelParams[i]);
      else kernelArgs_.push_back(nodeParams.kernelParams[i]); 
    }
    guarantee(nodeParams.extra == nullptr,
              "current implementation does not support `extra` params");
  }
  kernelArgs_.push_back(&semaphore_);
  kernelArgs_.reserve(kernelArgs_.size() + hiddenkernelArgs_.size());
  kernelArgs_.insert(kernelArgs_.end(), hiddenkernelArgs_.begin(), hiddenkernelArgs_.end());

  fusedNodeParams_.kernelParams = kernelArgs_.data();
  fusedNodeParams_.extra = nullptr;

  hipError_t status = hip::GraphKernelNode::validateKernelParams(&fusedNodeParams_);
  if (hipSuccess != status) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "`validateKernelParams` during fusion");
  }
  fusedNode_ = new hip::GraphKernelNode(&fusedNodeParams_, nullptr);
}
}  // namespace


namespace hip {
bool GraphModifier::isSubstitutionStateQueried_{false};
bool GraphModifier::isSubstitutionSwitchedOn_{false};
GraphModifier::CounterType GraphModifier::instanceCounter_{};
hip::ExternalCOs::SymbolTableType GraphModifier::symbolTable_{};
std::vector<GraphModifier::GraphDescription> GraphModifier::descriptions_{};

bool GraphModifier::isInputOk() {
  auto* env = getenv("AMD_FUSION_MANIFEST");
  if (env == nullptr) {
    std::stringstream msg;
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS,
            "fusion manifest is not specified; cannot proceed fusion substitution");
    return false;
  }

  std::string manifestPathName(env);
  std::filesystem::path filePath(manifestPathName);
  if (!std::filesystem::exists(filePath)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "cannot open fusion manifest file: %s",
            manifestPathName.c_str());
    return false;
  }

  std::vector<std::pair<std::string, std::string>> fusedGroups{};
  std::vector<std::vector<std::vector<size_t>>> executionOrders{};
  try {
    const auto& manifest = YAML::LoadFile(manifestPathName);
    const auto& graphs = manifest["graphs"];
    for (const auto& graph : graphs) {
      GraphModifier::GraphDescription descr{};
      for (const auto& group : graph["groups"]) {
        const auto& groupName = group["name"].as<std::string>();
        const auto& location = group["location"].as<std::string>();
        fusedGroups.push_back(std::make_pair(groupName, location));
        descr.groupSymbols_.push_back(groupName);
      }
      std::vector<std::vector<size_t>> order;
      for (const auto& sequence : graph["executionOrder"]) {
        order.push_back(sequence.as<std::vector<size_t>>());
      }
      descr.executionGroups_ = std::move(order);
      descriptions_.push_back(descr);
    }
  } catch (const YAML::ParserException& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion manifest: %s", ex.what());
    return false;
  } catch (const std::runtime_error& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion manifest: %s", ex.what());
    return false;
  }

  loadExternalSymbols(fusedGroups);
  GraphModifier::symbolTable_ = PlatformState::instance().getExternalSymbolTable();

  auto isOk = PlatformState::instance().initSemaphore();
  if (!isOk) {
    return false;
  }

  ClPrint(amd::LOG_INFO, amd::LOG_ALWAYS, "graph fuse substitution is enabled");
  return true;
}

bool GraphModifier::isSubstitutionOn() {
  static amd::Monitor lock_;
  amd::ScopedLock lock(lock_);
  if (!isSubstitutionStateQueried_) {
    isSubstitutionSwitchedOn_ = GraphModifier::isInputOk();
    isSubstitutionStateQueried_ = true;
  }
  return isSubstitutionSwitchedOn_;
}

void GraphModifier::Finalizer::operator()(size_t* instanceCounter) const {
  for (auto& description : GraphModifier::descriptions_) {
    for (auto& group : description.fusionGroups_) {
      delete group;
    }
  }
  delete instanceCounter;
}

GraphModifier::GraphModifier(hip::Graph*& graph) : graph_(graph) {
  amd::ScopedLock lock(fclock_);
  if (GraphModifier::instanceCounter_ == nullptr) {
    GraphModifier::instanceCounter_ = GraphModifier::CounterType{new size_t};
    *(GraphModifier::instanceCounter_) = 0;
  }
  instanceId_ = (*GraphModifier::instanceCounter_)++;
}

GraphModifier::~GraphModifier() {}

bool GraphModifier::check() {
  size_t numNodes{0};
  const auto& executionGroups_ = currDescription.executionGroups_;
  std::for_each(executionGroups_.begin(), executionGroups_.end(),
                [&numNodes](auto& group) { numNodes += group.size(); });
  const auto& originalGraphNodes = graph_->GetNodes();
  return numNodes == originalGraphNodes.size();
}

std::vector<hip::GraphNode*> GraphModifier::collectNodes(const std::vector<Node>& originalNodes) {
  size_t nodeCounter{0};
  std::vector<hip::GraphNode*> nodes{};
  const auto& executionGroups = currDescription.executionGroups_;
  for (const auto& group : executionGroups) {
    if (group.size() == 1) {
      const auto nodeNumber = group[0];
      guarantee(nodeNumber == nodeCounter, "the execution order must be correct");
      nodes.push_back(originalNodes[nodeCounter]);
      ++nodeCounter;
    } else {
      FusionGroup* fusionGroup = new FusionGroup{};
      currDescription.fusionGroups_.emplace_back(fusionGroup);

      for (const auto& nodeNumber : group) {
        guarantee(nodeNumber == nodeCounter, "the execution order must be correct");
        auto originalNode = originalNodes[nodeNumber];
        auto* kernelNode = dynamic_cast<hip::GraphKernelNode*>(originalNode);
        fusionGroup->addNode(kernelNode);
        ++nodeCounter;
      }
      nodes.push_back(fusionGroup);
    }
  }
  return nodes;
}

void GraphModifier::generateFusedNodes() {
  for (size_t groupId = 0; groupId < currDescription.fusionGroups_.size(); ++groupId) {
    std::string groupKey = currDescription.groupSymbols_.at(groupId);
    auto [funcHandle, fusedKernel] = GraphModifier::symbolTable_.at(groupKey);

    auto* group = dynamic_cast<FusionGroup*>(currDescription.fusionGroups_.at(groupId));
    group->generateNode(funcHandle);
  }
}

void GraphModifier::run() {
  amd::ScopedLock lock(fclock_);
  currDescription = descriptions_[instanceId_];

  auto isOk = check();
  if (!isOk) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "substitution of graph %zu failed consistency check",
            instanceId_);
    return;
  }

  const auto& originalGraphNodes = graph_->GetNodes();
  auto nodes = collectNodes(originalGraphNodes);
  generateFusedNodes();

  for (size_t i = 0; i < currDescription.fusionGroups_.size(); ++i) {
    auto* group = dynamic_cast<FusionGroup*>(currDescription.fusionGroups_.at(i));
    auto* fusedNode = group->getFusedNode();

    auto* groupHead = group->getHead();
    if (groupHead) {
      const auto& dependencies = groupHead->GetDependencies();
      std::vector<Node> additionalEdges{fusedNode};
      for (const auto& dependency : dependencies) {
        dependency->RemoveUpdateEdge(groupHead);
        dependency->AddEdge(fusedNode);
      }
    }

    auto* groupTail = group->getTail();
    if (groupTail) {
      const auto& edges = groupTail->GetEdges();
      for (const auto& edge : edges) {
        groupTail->RemoveUpdateEdge(edge);
        fusedNode->AddEdge(edge);
      }
    }

    auto& fusee = group->getNodes();
    for (auto node : fusee) {
      graph_->RemoveNode(node);
    }
    graph_->AddNode(fusedNode);
  }
}
}  // namespace hip
