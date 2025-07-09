/* Copyright (c) 2025 Advanced Micro Devices, Inc.

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

#include <fstream>
#include "hip_graph_internal.hpp"

namespace hip {

struct DominatorTreeNode {
  enum Type {
    Entry,
    Exit,
    Node,
  };
  Type type_;
  GraphNode *node_;

  // Needed for set operations
  bool operator<(const DominatorTreeNode& other) const {
    return type_ == Type::Node && other.type_ == Type::Node ? node_ < other.node_ : type_ < other.type_;
  }

  bool operator==(const DominatorTreeNode& other) const {
    return type_ == Type::Node && other.type_ == Type::Node ? node_ == other.node_ : type_ == other.type_;
  }
};

class DominatorTreeBuilder {
public:
  DominatorTreeBuilder(std::vector<GraphNode*> roots, std::vector<GraphNode*> &nodes, bool is_postdominator=false)
    : roots_(roots),
      nodes_(nodes),
      entry_node_({DominatorTreeNode::Type::Entry, nullptr}),
      exit_node_({DominatorTreeNode::Type::Exit, nullptr}),
      reverse_arrival_times_(nodes.size() + 3),
      labels_(nodes.size() + 3),
      sdom_(nodes.size() + 3),
      idom_(nodes.size() + 3),
      parents_(nodes.size() + 3),
      reverse_graph_(nodes.size() + 3),
      dsu_(nodes.size() + 3),
      buckets_(nodes.size() + 3),
      arrival_time_(0),
      is_postdominator_(is_postdominator) {
      
	for (size_t i = 0; i < nodes.size() + 3; ++i) {
	  sdom_[i] = i;
	  idom_[i] = i;
	  dsu_[i] = i;
	  labels_[i] = i;
	}
	arrival_times_[entry_node_] = 0;
	arrival_times_[exit_node_] = 0;
	for (auto node : nodes) {
	  arrival_times_[{DominatorTreeNode::Type::Node, node}] = 0;
	}
      }

  void Run() {
    DepthFirstSearch({DominatorTreeNode::Type::Entry, nullptr});

    for (size_t i = nodes_.size() + 2; i >= 1; --i) {
      for (int j = 0; j < reverse_graph_[i].size(); ++j) {
	sdom_[i] = std::min(sdom_[i], sdom_[Find(reverse_graph_[i][j])]);
      }
      if (i > 1) {
	buckets_[sdom_[i]].push_back(i);
      }

      for (size_t j = 0; j < buckets_[i].size(); ++j) {
	size_t w = buckets_[i][j];
	size_t v = Find(w);
	if (sdom_[v] == sdom_[w]) {
	  idom_[w] = sdom_[w];
	} else {
	  idom_[w] = v;
	}
      }
      if (i > 1) {
	Union(parents_[i], i);
      }
    }
  }

  std::pair<std::map<DominatorTreeNode, std::vector<DominatorTreeNode>>, std::map<DominatorTreeNode, DominatorTreeNode>> GetTree() {
    std::map<DominatorTreeNode, std::vector<DominatorTreeNode>> tree;
    std::map<DominatorTreeNode, DominatorTreeNode> parents;
    tree[entry_node_] = {};
    tree[exit_node_] = {};
    for (auto node : nodes_) {
      tree[{DominatorTreeNode::Type::Node, node}] = {};
    }

    for (size_t i = 2; i <= nodes_.size() + 2; ++i) {
      if (idom_[i] != sdom_[i]) {
	idom_[i] = idom_[idom_[i]];
      }
      parents[reverse_arrival_times_[i]] = reverse_arrival_times_[idom_[i]];
      tree[reverse_arrival_times_[idom_[i]]].push_back(reverse_arrival_times_[i]);
    }
    return {tree, parents};
  }
private:
  void DepthFirstSearch(DominatorTreeNode node) {
    ++arrival_time_;
    arrival_times_[node] = arrival_time_;
    reverse_arrival_times_[arrival_time_] = node;
    labels_[arrival_time_] = arrival_time_;
    sdom_[arrival_time_] = arrival_time_;
    parents_[arrival_time_] = arrival_time_;
    dsu_[arrival_time_] = arrival_time_;

    if (node == entry_node_) {
      // This is a dud entry node, search from actual roots
      for (auto root : roots_) {
	DominatorTreeNode root_node = {DominatorTreeNode::Type::Node, root};
	DepthFirstSearch(root_node);
	parents_[arrival_times_[root_node]] = arrival_times_[node];
	reverse_graph_[arrival_times_[root_node]].push_back(arrival_times_[node]);
      }
    } else if (node.type_ == DominatorTreeNode::Type::Node) {
      auto children = is_postdominator_ ? node.node_->GetDependencies() : node.node_->GetEdges();
      if (!children.empty()) {
	// This is not a leaf node
	for (auto edge : children) {
	  DominatorTreeNode edge_node = {DominatorTreeNode::Type::Node, edge};
	  if (arrival_times_[edge_node] == 0) {
	    DepthFirstSearch(edge_node);
	    parents_[arrival_times_[edge_node]] = arrival_times_[node];
	  }
	  reverse_graph_[arrival_times_[edge_node]].push_back(arrival_times_[node]);
	}
      } else {
	// This is a leaf node, process dud exit node
	if (arrival_times_[exit_node_] == 0) {
	  DepthFirstSearch(exit_node_);
	  parents_[arrival_times_[exit_node_]] = arrival_times_[node];
	}
	reverse_graph_[arrival_times_[exit_node_]].push_back(arrival_times_[node]);
      }
    }
  }

  size_t Find(size_t u, size_t x = 0) {
    if (u == dsu_[u]) {
      return x != 0 ? (size_t) -1 : u;
    }

    size_t v = Find(dsu_[u], x + 1);
    if (v == (size_t) -1) {
      return u;
    }
    if (sdom_[labels_[dsu_[u]]] < sdom_[labels_[u]]) {
      labels_[u] = labels_[dsu_[u]];
    }
    dsu_[u] = v;
    return x != 0 ? v : labels_[u];
  }

  void Union(size_t u, size_t v) {
    dsu_[v] = u;
  }

  std::vector<GraphNode*> roots_;
  std::vector<GraphNode*> &nodes_;

  DominatorTreeNode entry_node_;
  DominatorTreeNode exit_node_;

  std::map<DominatorTreeNode, size_t> arrival_times_;
  std::vector<DominatorTreeNode> reverse_arrival_times_;
  std::vector<size_t> labels_;
  std::vector<size_t> sdom_;
  std::vector<size_t> idom_;
  std::vector<size_t> parents_;
  std::vector<std::vector<size_t>> reverse_graph_;
  std::vector<size_t> dsu_;
  std::vector<std::vector<size_t>> buckets_;
  size_t arrival_time_;
  bool is_postdominator_;
};

class DominatorTree {
public:
  void Build(Graph* graph) {
    auto roots = graph->GetRootNodes();
    auto leaves = graph->GetLeafNodes();
    auto nodes = graph->GetNodes();

    {
      DominatorTreeBuilder dtb(roots, nodes);
      dtb.Run();
      auto t = dtb.GetTree();
      dominator_tree_ = t.first;
      parents_ = t.second;
    }

    {
      DominatorTreeBuilder dtb(leaves, nodes, true);
      dtb.Run();
      auto t = dtb.GetTree();
      postdominator_tree_ = t.first;
      postdominator_parents_ = t.second;
    }
  }

  bool Dominates(GraphNode* node1, GraphNode* node2) {
    DominatorTreeNode actual_node1;
    if (node1 == nullptr) {
      actual_node1 = {DominatorTreeNode::Type::Entry, nullptr};
    } else {
      actual_node1 = {DominatorTreeNode::Type::Node, node1};
    }
    DominatorTreeNode actual_node2;
    if (node2 == nullptr) {
      actual_node2 = {DominatorTreeNode::Type::Entry, nullptr};
    } else {
      actual_node2 = {DominatorTreeNode::Type::Node, node2};
    }
    return Dominates(actual_node1, actual_node2);
  }

  GraphNode* FirstCommonDominator(GraphNode* node1, GraphNode* node2, bool post=false) {
    // Null encodes ENTRY
    if (node1 == nullptr || node2 == nullptr) {
      return nullptr;
    }
    DominatorTreeNode current_node = {DominatorTreeNode::Type::Node, node1};
    DominatorTreeNode search_node = {DominatorTreeNode::Type::Node, node2};
    // Enough to check that current_node dominates node2 since we're traversing domination chain of node1
    while (!Dominates(current_node, search_node, post)) {
      current_node = parents_[current_node];
    }
    assert(current_node.type_ != DominatorTreeNode::Type::Exit);
    if (current_node.type_ == DominatorTreeNode::Type::Entry) {
      return nullptr;
    }
    return current_node.node_;
  }

  GraphNode* FirstCommonPostdominator(GraphNode* node1, GraphNode* node2) {
    return FirstCommonDominator(node1, node2, true);
  }

  GraphNode* ImmediateDominator(GraphNode* node) {
    if (node == nullptr) {
      return nullptr;
    }
    return parents_[{DominatorTreeNode::Type::Node, node}].node_;
  }
private:
  bool Dominates(DominatorTreeNode node1, DominatorTreeNode node2, bool post=false) {
    return DepthFirstSearch(node1, node2, post);
  }

  bool DepthFirstSearch(DominatorTreeNode node1, DominatorTreeNode node2, bool post) {
    if (node1 == node2) return true;
    for (auto child : (post ? postdominator_tree_ : dominator_tree_)[node1]) {
      auto res = DepthFirstSearch(child, node2, post);
      if (res) return true;
    }
    return false;
  }

  std::map<DominatorTreeNode, std::vector<DominatorTreeNode>> dominator_tree_;
  std::map<DominatorTreeNode, DominatorTreeNode> parents_;
  std::map<DominatorTreeNode, std::vector<DominatorTreeNode>> postdominator_tree_;
  std::map<DominatorTreeNode, DominatorTreeNode> postdominator_parents_;
};

class GraphAnalysis {
  struct Value {
    void* val_;
    std::map<GraphNode*, std::set<GraphNode*>> def_chains_;
    GraphNode* first_def_;

    // Needed for set operations
    bool operator<(const Value& other) const {
      return val_ < other.val_;
    }
  };

  typedef std::map<GraphNode*, std::set<GraphNode*>> CoarseValues;

  struct Coallocation {
    GraphNode* node_;
    std::vector<GraphNode*> objects_;
  };

  enum class AllocationHeuristic {
    Greedy = 0,
  };

  struct AllocatorAction {
    enum class Type {
      Allocate = 0,
      Free = 1,
    };

    Type type_;
    GraphNode* node_;
  };

public:
  bool Run(Graph* graph) {
    dt_.Build(graph);
    GetValues(graph);

    // Pass 1: remove unnecessary dependencies
    bool modified = MoveByDependencies(graph);
    modified |= RemoveUselessEdges();
    
    // Pass 2: coallocation
    dt_.Build(graph);
    FindCoallocatedObjects(graph);
    CreateAllocationSchedule(AllocationHeuristic::Greedy);

    return modified;
  }
private:
  void GetValues(Graph* graph) {
    auto path_exists = [](GraphNode* s, GraphNode* t) -> bool {
      if (s == nullptr) {
	return true;
      }

      std::function<bool(GraphNode*, GraphNode*, std::set<GraphNode*>&)> DFS = [&](GraphNode* s, GraphNode* t, std::set<GraphNode*>& visited) -> bool {
	if (s == t) {
	  return true;
	}

	for (auto edge : s->GetEdges()) {
	  if (visited.find(edge) != visited.end()) {
	    continue;
	  }
	  auto found = DFS(edge, t, visited);
	  if (found) {
	    return true;
	  }
	  visited.insert(edge);
	}
	return false;
      };

      std::set<GraphNode*> visited;
      return DFS(s, t, visited);
    };

    std::map<GraphNode*, std::set<void*>> uses_without_defs;

    auto nodes = graph->GetNodes();
    for (auto node : nodes) {
      auto dependencies = node->Values();
      auto defs = dependencies.first;
      auto uses = dependencies.second;

      for (auto def : defs) {
	Value dep_value = {def, {}, nullptr};
	auto existing_value = values_.find(dep_value);
	if (existing_value == values_.end()) {
	  // First time seeing the value
	  dep_value.def_chains_[node] = {};
	  dep_value.first_def_ = node;
	  values_.insert(dep_value);
	} else {
	  // Find the "latest" def
          GraphNode* latest_def = nullptr;
	  for (auto def_chain : existing_value->def_chains_) {
	    if (path_exists(node, def_chain.first)) {
	      continue;
	    }
	    if (path_exists(latest_def, def_chain.first)) {
	      latest_def = def_chain.first;
	    }
	  }

	  Value new_value { existing_value->val_, existing_value->def_chains_, existing_value->first_def_ };
	  if (latest_def == nullptr) {
	    // This is the new first def
	    new_value.first_def_ = node;
	    new_value.def_chains_[node] = {};
	  } else {
	    // There may be some uses that belong to the new def
	    std::set<GraphNode*> old_defs;
	    std::set<GraphNode*> new_defs;
	    for (auto use : new_value.def_chains_[latest_def]) {
	      if (dt_.Dominates(node, use)) {
		new_defs.insert(use);
	      } else {
		old_defs.insert(use);
	      }
	    }
	    new_value.def_chains_[latest_def] = old_defs;
	    new_value.def_chains_[node] = new_defs;
	  }
	  values_.erase(existing_value);
	  values_.insert(new_value);
	}
      }

      for (auto use : uses) {
	Value dep_value = {use, {}, nullptr};
	auto existing_value = values_.find(dep_value);

	if (existing_value == values_.end()) {
	  if (uses_without_defs.find(node) == uses_without_defs.end()) {
	    uses_without_defs[node] = {use};
	  } else {
	    uses_without_defs[node].insert(use);
	  }
	  continue;
	}

	// Find the "latest" def
	GraphNode* latest_def = nullptr;
	for (auto def_chain : existing_value->def_chains_) {
	  if (path_exists(node, def_chain.first)) {
	    continue;
	  }
	  if (path_exists(latest_def, def_chain.first)) {
	    latest_def = def_chain.first;
	  }
	}

	if (latest_def == nullptr) {
	  if (uses_without_defs.find(node) == uses_without_defs.end()) {
	    uses_without_defs[node] = {use};
	  } else {
	    uses_without_defs[node].insert(use);
	  }
	} else {
	  Value new_value { existing_value->val_, existing_value->def_chains_, existing_value->first_def_ };
	  new_value.def_chains_[latest_def].insert(node);
	  values_.erase(existing_value);
	  values_.insert(new_value);
	}
      }
    }

    // Try to salvage uses that did not have defs (i.e. due to the iteration order of the graph)
    // FIXME: code duplication
    for (auto e : uses_without_defs) {
      auto node = e.first;
      auto& uses = e.second;
      for (auto use : uses) {
	Value dep_value = {use, {}, nullptr};
	auto existing_value = values_.find(dep_value);
	GraphNode* latest_def = nullptr;

	if (existing_value != values_.end()) {
	  // Find the "latest" def
	  for (auto def_chain : existing_value->def_chains_) {
	    if (path_exists(node, def_chain.first)) {
	      continue;
	    }
	    if (path_exists(latest_def, def_chain.first)) {
	      latest_def = def_chain.first;
	    }
	  }
	} else {
	  LogPrintfError("Only one use for value: %p %ld", use, node->GetID());
	  dep_value.first_def_ = node;
	  values_.insert(dep_value);
	  continue;
	}

	Value new_value { existing_value->val_, existing_value->def_chains_, existing_value->first_def_ };
	if (latest_def == nullptr) {
	  new_value.def_chains_[node] = new_value.def_chains_[new_value.first_def_];
	  new_value.first_def_ = node;
	} else {
	  new_value.def_chains_[latest_def].insert(node);
	}
	values_.erase(existing_value);
	values_.insert(new_value);
      }
    }

    for (auto& value : values_) {
      for (auto& def_chain : value.def_chains_) {
	if (def_use_chains_.find(def_chain.first) == def_use_chains_.end()) {
	  def_use_chains_[def_chain.first] = def_chain.second;
	} else {
	  for (auto u : def_chain.second) {
	    def_use_chains_[def_chain.first].insert(u);
	  }
	}

	for (auto u : def_chain.second) {
	  if (use_def_chains_.find(u) == use_def_chains_.end()) {
	    use_def_chains_[u] = {def_chain.first};
	  } else {
	    use_def_chains_[u].insert(def_chain.first);
	  }
	}
      }
    }
  }

  bool MoveByDependencies(Graph* graph) {
    // Store a copy of the graph to later check if we made any changes
    std::map<GraphNode*, std::vector<GraphNode*>> graph_copy;
    for (auto node : graph->GetNodes()) {
      std::vector<GraphNode*> edges;
      for (auto edge : node->GetEdges()) {
	edges.push_back(edge);
      }
      graph_copy[node] = edges;
    }

    // Unlink the whole graph
    for (auto node : graph->GetNodes()) {
      auto parents = node->GetDependencies();
      auto children = node->GetEdges();
      for (auto parent : parents) {
	node->RemoveDependency(parent);
      }
      for (auto child : children) {
	node->RemoveEdge(child);
      }
    }

    // Link dependencies directly to use
    for (auto def_chain : def_use_chains_) {
      auto def = def_chain.first;
      for (auto use : def_chain.second) {
	def->AddEdgeDep(use);
      }
    }

    // Check if we made any changes
    for (auto node : graph->GetNodes()) {
      if (graph_copy[node].size() != node->GetEdges().size()) {
	return true;
      }
      for (auto edge : node->GetEdges()) {
	if (std::find(graph_copy[node].begin(), graph_copy[node].end(), edge) == graph_copy[node].end()) {
	  return true;
	}
      }
    }
    return false;
  }

  bool RemoveUselessEdges() {
    std::function<bool(GraphNode*, GraphNode*, std::map<GraphNode*, bool>&)> DFS = [&](GraphNode* s, GraphNode* t, std::map<GraphNode*, bool>& visited) -> bool {
      if (visited[s]) {
	return false;
      }
      if (s == t) {
	return true;
      }

      visited[s] = true;
      for (auto use : def_use_chains_[s]) {
	auto found = DFS(use, t, visited);
	if (found) {
	  return true;
	}
      }
      return false;
    };

    // Removes direct edges that can be reached through other children, e.g.
    // A -> B; A -> C; B -> C becomes A -> B; B -> C
    bool modified = false;
    for (auto& def_chain : def_use_chains_) {
      auto def = def_chain.first;
      auto uses = def_chain.second;
      for (auto use : uses) {
	bool can_reach = false;
	for (auto use2 : uses) {
	  if (use == use2) {
	    continue;
	  }

	  std::map<GraphNode*, bool> visited;
	  for (auto def : def_use_chains_) {
	    visited[def.first] = false;
	  }
	  if (DFS(use2, use, visited)) {
	    can_reach = true;
	    break;
	  }
	}
	if (can_reach) {
	  def->RemoveEdgeDep(use);
	  modified = true;
	}
      }
    }
    return modified;
  }

  void FindCoallocatedObjects(Graph* graph) {
    std::map<GraphNode*, size_t> distances;
    auto longest_path = [&](GraphNode* search) {
      // Iteration in topological order + dynamic programming
      // Only works because this is a DAG
      std::map<GraphNode*, size_t> length_to;
      for (auto node : graph->GetNodes()) {
	length_to[node] = 0;
      }
      for (auto node : graph->GetNodes()) {
	for (auto edge : node->GetEdges()) {
	  length_to[edge] = std::max(length_to[edge], length_to[node] + 1);
	}
      }
      return length_to[search];
    };

    for (auto use_chain : use_def_chains_) {
      if (use_chain.second.size() <= 1) {
	continue;
      }

      auto node = use_chain.first;

      std::vector<GraphNode*> coallocated;
      for (auto& value : values_) {
	if (value.def_chains_.find(node) != value.def_chains_.end() && value.first_def_->GetType() == hipGraphNodeTypeMemAlloc) {
	  coallocated.push_back(value.first_def_);
	}
      }
      if (coallocated.size() > 1) {
	coallocations_.push_back({node, coallocated});
	distances[node] = longest_path(node);
      }
    }


    // Sort in ascending order based on the longest distance from the entry
    std::sort(coallocations_.begin(), coallocations_.end(), [&](Coallocation& l, Coallocation& r) {
	return distances[l.node_] < distances[r.node_];
    });
  }

  void CreateAllocationSchedule(AllocationHeuristic heuristic) {
    switch (heuristic) {
      case (AllocationHeuristic::Greedy): {
	CreateAllocationScheduleGreedy();
	break;
      }
      default: {
	 LogPrintfError("Creating allocation schedule with heuristic unimplemented: %d", (int) heuristic);
	 break;
      }
    }
  }

  void CreateAllocationScheduleGreedy() {
    auto get_lifetime = [&](void* ptr, GraphNode* node) -> size_t {
      // This should be a shortest path algorithm in the presense of loops
      // But since there are no loop, it can be a simple DFS
      std::map<GraphNode*, bool> visited;
      for (auto def : def_use_chains_) {
	visited[def.first] = false;
      }
      std::function<size_t(void*, GraphNode*, size_t)> DFS = [&](void* ptr, GraphNode* node, size_t distance) -> size_t {
	if (node->GetType() == hipGraphNodeTypeMemFree) {
	  void* free_ptr = nullptr;
	  dynamic_cast<GraphMemFreeNode*>(node)->GetParams(&free_ptr);
	  if (free_ptr == ptr) {
	    return distance;
	  }
	}

	for (auto use : def_use_chains_[node]) {
	  if (visited[use]) {
	    continue;
	  }
	  auto dist = DFS(ptr, use, distance + 1);
	  if (dist != (size_t) -1) {
	    return dist;
	  }
	  visited[use] = true;
	}
	return (size_t) -1;
      };

      return DFS(ptr, node, 0);
    };

    std::map<GraphNode*, size_t> lifetimes;
    for (auto& value : values_) {
      if (value.first_def_->GetType() == hipGraphNodeTypeMemAlloc) {
	lifetimes[value.first_def_] = get_lifetime(value.val_, value.first_def_);
      }
    }

    std::map<GraphNode*, size_t> latest_coallocation;
    for (size_t i = 0; i < coallocations_.size(); ++i) {
      for (auto object : coallocations_[i].objects_) {
	latest_coallocation[object] = i;
      }
    }

    auto CanFree = [&](GraphNode* alloc_node, GraphNode* use_node) {
      auto this_value = *values_.begin();
      for (auto value : values_) {
	if (value.first_def_ == alloc_node) {
	  this_value = value;
	  break;
	}
      }

      std::map<GraphNode*, bool> visited;
      for (auto& def : def_use_chains_) {
	visited[def.first] = false;
      }

      std::function<bool(Value&, GraphNode*)> DFS = [&](Value& value, GraphNode* node) -> bool {
	if (value.def_chains_.find(node) != value.def_chains_.end()) {
	  return true;
	}

	for (auto use : def_use_chains_[node]) {
	  if (visited[use]) {
	    continue;
	  }
	  auto found = DFS(value, use);
	  if (found) {
	    return true;
	  }
	  visited[use] = true;
	}
	return false;
      };

      return !DFS(this_value, use_node);
    };

    std::vector<GraphNode*> heap_slots;
    auto Allocated = [&](GraphNode* object) -> bool {
      return std::find(heap_slots.begin(), heap_slots.end(), object) != heap_slots.end();
    };

    std::vector<AllocatorAction> schedule;
    for (size_t i = 0; i < coallocations_.size(); ++i) {
      auto& coallocated = coallocations_[i].objects_;
      // Sort by longest lived first
      std::sort(coallocated.begin(), coallocated.end(), [&](GraphNode* l, GraphNode* r) {
        return (lifetimes[l] == lifetimes[r] && latest_coallocation[l] > latest_coallocation[r]) || lifetimes[l] < lifetimes[r];
      });

      for (auto object : coallocated) {
	if (Allocated(object)) {
	  continue;
	}

        size_t my_heap_slot = heap_slots.size();
        for (size_t j = 0; j < heap_slots.size(); ++j) {
          if (CanFree(heap_slots[j], coallocations_[i].node_)) {
            schedule.push_back({AllocatorAction::Type::Free, heap_slots[j]});
            my_heap_slot = j;
            break;
          }
        }

        if (my_heap_slot == heap_slots.size()) {
          heap_slots.push_back(object);
        } else {
          heap_slots[my_heap_slot] = object;
        }
        schedule.push_back({AllocatorAction::Type::Allocate, object});
      }
    }
  }

  DominatorTree dt_;
  std::set<Value> values_;
  CoarseValues def_use_chains_;
  CoarseValues use_def_chains_;
  std::vector<Coallocation> coallocations_;
};

} // namespace hip
