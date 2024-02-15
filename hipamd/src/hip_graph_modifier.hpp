#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "hip_graph_internal.hpp"
#include "hip_code_object.hpp"


namespace hip {
class GraphModifier {
  amd::Monitor fclock_{"Guards Graph Modifier object", true};

 public:
  GraphModifier(hip::Graph*& graph);
  ~GraphModifier();
  void run();
  static bool isSubstitutionOn();

 private:
  struct GraphDescription {
    std::vector<std::string> groupSymbols_{};
    std::vector<std::vector<size_t>> executionGroups_{};
    std::vector<hip::GraphNode*> fusionGroups_{};
  };

  static bool isInputOk();
  bool check();
  std::vector<Node> collectNodes(const std::vector<Node>& originalNodes);
  void generateFusedNodes();
  void adjustNodeLevels();

  hip::Graph*& graph_;
  size_t instanceId_{};
  GraphDescription currDescription{};

  struct Finalizer {
    void operator()(size_t*) const;
  };
  using CounterType = std::unique_ptr<size_t, Finalizer>;

  static CounterType instanceCounter_;
  static bool isSubstitutionStateQueried_;
  static bool isSubstitutionSwitchedOn_;
  static hip::ExternalCOs::SymbolTableType symbolTable_;
  static std::vector<GraphDescription> descriptions_;
};
}  // namespace hip
