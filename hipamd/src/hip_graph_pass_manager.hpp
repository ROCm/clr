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
namespace ga {

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

struct Coallocation {
  GraphNode* node_;
  std::vector<GraphNode*> objects_;
};

struct AllocatorAction {
  enum class Type {
    Allocate = 0,
    Free = 1,
  };

  Type type_;
  GraphNode* node_;
  size_t offset_;
  std::vector<size_t> dependencies_;
};

struct Value {
  void* val_;
  std::map<GraphNode*, std::set<GraphNode*>> def_chains_;
  GraphNode* first_def_;
  GraphNode* free_node_;

  // Needed for set operations
  bool operator<(const Value& other) const {
    return val_ < other.val_;
  }
};

typedef std::map<GraphNode*, std::set<GraphNode*>> CoarseValues;

class AllocationSchedulerGreedy {
  // idx is index in the allocation schedule
  struct SlotDependency {
    size_t idx;
    size_t offset;
    size_t size;
  };

  struct HeapSlot {
    size_t offset;
    size_t size;
    GraphNode* node;
    HeapSlot* prev;
    HeapSlot* next;
    // Frees that have to be executed before all/part of the slot becomes available
    std::vector<SlotDependency> dependencies;
  };
public:
  AllocationSchedulerGreedy(std::set<Value>& values, CoarseValues& def_use_chains, std::vector<Coallocation> coallocations)
    : values_(values),
      def_use_chains_(def_use_chains),
      coallocations_(coallocations),
      heap_slots_(new HeapSlot({0, (size_t) -1, nullptr, nullptr, nullptr, {}})) {

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

    for (auto& value : values_) {
      if (value.first_def_->GetType() == hipGraphNodeTypeMemAlloc) {
	lifetimes_[value.first_def_] = get_lifetime(value.val_, value.first_def_);
      }
    }

    for (size_t i = 0; i < coallocations_.size(); ++i) {
      for (auto object : coallocations_[i].objects_) {
	latest_coallocation_[object] = i;
      }
    }
  }

  ~AllocationSchedulerGreedy() {
    auto curr_slot = heap_slots_;
    do {
      auto tmp = curr_slot;
      curr_slot = curr_slot->next;
      delete tmp;
    } while (curr_slot != nullptr);
  }

  std::vector<AllocatorAction> Make() {
    for (size_t i = 0; i < coallocations_.size(); ++i) {
      auto& coallocated = coallocations_[i].objects_;
      // Sort by longest lived first
      std::sort(coallocated.begin(), coallocated.end(), [&](GraphNode* l, GraphNode* r) {
        return (lifetimes_[l] == lifetimes_[r] && latest_coallocation_[l] > latest_coallocation_[r]) || lifetimes_[l] < lifetimes_[r];
      });

      //CleanSlots(coallocations_[i].node_);

      for (auto object : coallocated) {
	if (Allocated(object)) {
	  continue;
	}

	size_t my_offset = 0;
	auto slot = heap_slots_;
	do {
	  if (Fits(object, slot) && (Empty(slot) || FreeSlot(coallocations_[i].node_, &slot))) {
	    AllocateInSlot(object, slot);
	    my_offset = slot->offset;
	    break;
	  }

	  slot = slot->next;
	} while (slot != nullptr);

	// Get fine-grained dependencies for this allocation
	// Iterate through slot dependencies backwards (i.e. latest first) until the whole allocation range has been covered
	std::vector<size_t> dependencies;
	std::vector<std::pair<size_t, size_t>> uncovered_range = {{my_offset, my_offset + slot->size}};
	for (auto it = slot->dependencies.rbegin(); it != slot->dependencies.rend(); ++it) {
	  std::vector<std::pair<size_t, size_t>> new_uncovered_range;
	  for (auto& range : uncovered_range) {
	    if (it->offset <= range.first && it->offset + it->size >= range.second) {
	      // (1) Dependency covers the range completely, do nothing
	      dependencies.push_back(it->idx);
	    } else if (it->offset <= range.first && it->offset + it->size < range.second) {
	      // (2) Dependency begins before the range and ends somewhere in the middle of it
	      new_uncovered_range.push_back({it->offset + it->size, range.second});
	      dependencies.push_back(it->idx);
	    } else if (it->offset > range.first && it->offset < range.second && it->offset + it->size >= range.second) {
	      // (3) Dependency begins inside the range and ends after the range
	      new_uncovered_range.push_back({range.first, it->offset});
	      dependencies.push_back(it->idx);
	    } else if (it->offset > range.first && it->offset + it->size < range.second) {
	      // (4) Dependency is inside the range and does not cover it completely, must split in two
	      new_uncovered_range.push_back({range.first, it->offset});
	      new_uncovered_range.push_back({it->offset + it->size, range.second});
	      dependencies.push_back(it->idx);
	    }
	  }
	  if (new_uncovered_range.empty()) {
	    break;
	  }
	  uncovered_range = new_uncovered_range;
	}

	schedule_.push_back({AllocatorAction::Type::Allocate, object, my_offset, dependencies});
      }
    }

    CleanSlots(nullptr);

    return schedule_;
  }
private:
  void CleanSlots(GraphNode* current_node) {
    auto slot = heap_slots_;
    do {
      bool empty = false;
      if (Empty(slot)) {
	empty = true;
      } else if (!current_node || CanFree(slot->node, current_node)) {
	schedule_.push_back({AllocatorAction::Type::Free, slot->node, slot->offset, {}});
	slot->node = nullptr;
	empty = true;
      }
      if (empty) {
	slot = Coalesce(slot);
      }
      slot = slot->next;
    } while (slot != nullptr);
  }

  bool FreeSlot(GraphNode* current_node, HeapSlot** slot) {
    if (!CanFree((*slot)->node, current_node)) {
      return false;
    }
    (*slot)->dependencies.push_back({schedule_.size(), (*slot)->offset, (*slot)->size});
    schedule_.push_back({AllocatorAction::Type::Free, (*slot)->node, (*slot)->offset});
    (*slot)->node = nullptr;
    *slot = Coalesce(*slot);
    return true;
  }

  void AllocateInSlot(GraphNode* object, HeapSlot* slot) {
    size_t size = dynamic_cast<GraphMemAllocNode*>(object)->Bytesize();
    size_t remainder = slot->size - size;
    HeapSlot *remainder_slot = new HeapSlot({slot->offset + size, remainder, nullptr, slot, slot->next, {}});
    if (slot->next) {
      slot->next->prev = remainder_slot;
    }
    slot->next = remainder_slot;
    slot->size = size;
    slot->node = object;
  }

  HeapSlot* Coalesce(HeapSlot* slot) {
    auto curr_slot = slot;
    // Coalesce with previous
    if (curr_slot->prev && Empty(curr_slot->prev)) {
      curr_slot->prev->size += curr_slot->size;
      curr_slot->prev->next = curr_slot->next;
      if (curr_slot->next) {
	curr_slot->next->prev = curr_slot->prev;
      }
      curr_slot->prev->dependencies.insert(curr_slot->prev->dependencies.end(), curr_slot->dependencies.begin(), curr_slot->dependencies.end());
      auto tmp = curr_slot;
      curr_slot = curr_slot->prev;
      delete tmp;
    }
    // Coalesce with next
    if (curr_slot->next && Empty(curr_slot->next)) {
      curr_slot->size += curr_slot->next->size;
      if (curr_slot->next->next) {
	curr_slot->next->next->prev = curr_slot;
      }
      curr_slot->dependencies.insert(curr_slot->dependencies.end(), curr_slot->next->dependencies.begin(), curr_slot->next->dependencies.end());
      auto tmp = curr_slot->next;
      curr_slot->next = curr_slot->next->next;
      delete tmp;
    }
    return curr_slot;
  }

  bool CanFree(GraphNode* alloc_node, GraphNode* use_node) {
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

  bool Allocated(GraphNode* object) {
    auto slot = heap_slots_;
    do {
      if (slot->node == object) {
	return true;
      }
      slot = slot->next;
    } while (slot != nullptr);
    return false;
  }

  bool Fits(GraphNode* object, HeapSlot* heap_slot) {
    size_t size = dynamic_cast<GraphMemAllocNode*>(object)->Bytesize();
    return heap_slot->size >= size;
  }

  bool Empty(HeapSlot* heap_slot) {
    return heap_slot->node == nullptr;
  }

  std::set<Value>& values_;
  CoarseValues& def_use_chains_;
  std::vector<Coallocation> coallocations_;

  std::map<GraphNode*, size_t> lifetimes_;
  std::map<GraphNode*, size_t> latest_coallocation_;
  HeapSlot* heap_slots_;
  std::vector<AllocatorAction> schedule_;
};

class GraphAnalysis {
  struct AllocationSchedule {
    std::vector<AllocatorAction> actions_;
    std::vector<GraphMemAllocNode*> alloc_nodes_;
    size_t slab_size_;
  };

  enum class AllocationHeuristic {
    Greedy = 0,
  };

public:
  bool Run(Graph* graph) {
    if (needs_reschedule_) {
      bool simple_offset_scale = SimpleOffsetScale();
      if (!simple_offset_scale) {
	all_schedules_ = CreateAllocationSchedule(AllocationHeuristic::Greedy);
	AddCoallocationEdges();
      }
      SetSlabInfo();
      return !simple_offset_scale;
    } else if (graph == graph_) {
      return false;
    }

    graph_ = graph;

    dt_.Build(graph_);
    GetValues();

    // Pass 1: remove unnecessary dependencies
    bool modified = MoveByDependencies();
    modified |= RemoveUselessEdges();
    
    // Pass 2: coallocation
    FindCoallocatedObjects();
    all_schedules_ = CreateAllocationSchedule(AllocationHeuristic::Greedy);
    AddCoallocationEdges();
    SetSlabInfo();

    return modified;
  }

  void Invalidate() {
    graph_ = nullptr;
  }

  void AllocationSizeChanged(GraphNode* node, size_t previous_size, size_t new_size) {
    changed_node_sizes_[node] = {previous_size, new_size};
    needs_reschedule_ = true;
  }

private:
  void GetValues() {
    values_ = {};
    def_use_chains_ = {};
    use_def_chains_ = {};
    coallocations_ = {};

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

    auto nodes = graph_->GetNodes();
    for (auto node : nodes) {
      auto dependencies = node->Values();
      auto defs = dependencies.first;
      auto uses = dependencies.second;

      for (auto def : defs) {
	Value dep_value = {def, {}, nullptr, nullptr};
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

	  Value new_value { existing_value->val_, existing_value->def_chains_, existing_value->first_def_, existing_value->free_node_ };
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
	Value dep_value = {use, {}, nullptr, nullptr};
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
	  Value new_value { existing_value->val_, existing_value->def_chains_, existing_value->first_def_, existing_value->free_node_ };
	  if (node->GetType() == hipGraphNodeTypeMemFree) {
	    new_value.free_node_ = node;
	  }
	  if (node->GetType() == hipGraphNodeTypeMemFree && !new_value.def_chains_[latest_def].empty()) {
	    // Special case for free nodes: all preceding uses become defs
	    for (auto use : new_value.def_chains_[latest_def]) {
	      if (new_value.def_chains_.find(use) == new_value.def_chains_.end()) {
		new_value.def_chains_[use] = {node};
	      } else {
		new_value.def_chains_[use].insert(node);
	      }
	    }
	  } else {
	    new_value.def_chains_[latest_def].insert(node);
	  }
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
	Value dep_value = {use, {}, nullptr, nullptr};
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

	Value new_value { existing_value->val_, existing_value->def_chains_, existing_value->first_def_, existing_value->free_node_ };
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

  bool MoveByDependencies() {
    // Store a copy of the graph to later check if we made any changes
    std::map<GraphNode*, std::vector<GraphNode*>> graph_copy;
    for (auto node : graph_->GetNodes()) {
      std::vector<GraphNode*> edges;
      for (auto edge : node->GetEdges()) {
	edges.push_back(edge);
      }
      graph_copy[node] = edges;
    }

    // Unlink the whole graph
    for (auto node : graph_->GetNodes()) {
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
    for (auto node : graph_->GetNodes()) {
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

  bool SimpleOffsetScale() {
    for (auto& schedule : all_schedules_) {
      double offset_scale = 0.0;
      for (auto node : schedule.alloc_nodes_) {
	double this_scale;
	if (changed_node_sizes_.find(node) == changed_node_sizes_.end()) {
	  this_scale = 1.0;
	} else {
	  auto changed_size = changed_node_sizes_[node];
	  this_scale = ((double) changed_size.second) / changed_size.first;
	}
	if (offset_scale == 0.0) {
	  offset_scale = this_scale;
	} else if (offset_scale != this_scale) {
	  return false;
	}
      }

      for (auto& action : schedule.actions_) {
	action.offset_ *= offset_scale;
      }
      schedule.slab_size_ *= offset_scale;
    }

    return true;
  }

  void AddCoallocationEdges() {
    for (auto& schedule : all_schedules_) {
      for (auto& action : schedule.actions_) {
	if (action.type_ == AllocatorAction::Type::Allocate) {
	  auto node = action.node_;
	  for (auto dep : action.dependencies_) {
	    auto dep_node = schedule.actions_[dep].node_;
	    dep_node->AddEdgeDep(node);
	    if (def_use_chains_.find(dep_node) == def_use_chains_.end()) {
	      def_use_chains_[dep_node] = {node};
	    } else {
	      def_use_chains_[dep_node].insert(node);
	    }
	  }
	}
      }
    }

    RemoveUselessEdges();
  }

  void SetSlabInfo() {
    for (size_t k = 0; k < all_schedules_.size(); ++k) {
      auto& schedule = all_schedules_[k];

      for (auto& action : schedule.actions_) {
	if (action.type_ == AllocatorAction::Type::Allocate) {
	  auto alloc_node = dynamic_cast<GraphMemAllocNode*>(action.node_);
	  alloc_node->SetSlabInfo(k, schedule.slab_size_, schedule.alloc_nodes_, action.offset_);
	} else {
	  auto free_node = dynamic_cast<GraphMemFreeNode*>(action.node_);
	  free_node->SetSlabId(k);
	}
      }
    }

    changed_node_sizes_ = {};
    needs_reschedule_ = false;
  }

  void FindCoallocatedObjects() {
    struct CoallocationNode {
      std::set<size_t> edges;
    };
    std::vector<CoallocationNode> nodes;

    auto path_exists = [&](size_t s, size_t t) -> bool {
      std::vector<bool> visited(nodes.size(), false);

      std::function<bool(size_t, size_t)> DFS = [&](size_t s, size_t t) -> bool {
	if (s == t) {
	  return true;
	}

	visited[s] = true;
	for (auto e : nodes[s].edges) {
	  if (!visited[e]) {
	    auto found = DFS(e, t);
	    if (found) {
	      return true;
	    }
	  }
	}
	return false;
      };

      return DFS(s, t);
    };

    std::map<GraphNode*, size_t> distances;
    auto longest_path = [&](GraphNode* search) {
      // Iteration in topological order + dynamic programming
      // Only works because this is a DAG
      std::map<GraphNode*, size_t> length_to;
      for (auto node : graph_->GetNodes()) {
	length_to[node] = 0;
      }
      for (auto node : graph_->GetNodes()) {
	for (auto edge : node->GetEdges()) {
	  length_to[edge] = std::max(length_to[edge], length_to[node] + 1);
	}
      }
      return length_to[search];
    };

    std::vector<Coallocation> individual_coallocations;
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
	individual_coallocations.push_back({node, coallocated});
	distances[node] = longest_path(node);
      }
    }

    // Build a graph of coallocations, where E = {(u, v) | there exists GraphNode n s.t. n in u and n in v}
    nodes = std::vector<CoallocationNode>(individual_coallocations.size());
    for (size_t i = 0; i < individual_coallocations.size(); ++i) {
      for (size_t j = i + 1; j < individual_coallocations.size(); ++j) {
	std::set<GraphNode*> node_set;
	node_set.insert(individual_coallocations[i].objects_.begin(), individual_coallocations[i].objects_.end());
	node_set.insert(individual_coallocations[j].objects_.begin(), individual_coallocations[j].objects_.end());
	if (node_set.size() < individual_coallocations[i].objects_.size() + individual_coallocations[j].objects_.size()) {
	  nodes[i].edges.insert(j);
	  nodes[j].edges.insert(i);
	}
      }
    }

    // Build a graph of strongly connected components from the previous graph
    std::vector<bool> is_scc(nodes.size(), false);
    std::vector<std::vector<size_t>> scc;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (is_scc[i]) {
	continue;
      }

      std::vector<size_t> this_scc;
      this_scc.push_back(i);
      for (size_t j = i + 1; j < nodes.size(); ++j) {
	if (!is_scc[j] && path_exists(i, j)) {
	  is_scc[j] = true;
	  this_scc.push_back(j);
	}
      }

      scc.push_back(this_scc);
    }

    // Each strongly connected component should be scheduled separately
    coallocations_ = std::vector<std::vector<Coallocation>>(scc.size());
    for (size_t i = 0; i < scc.size(); ++i) {
      for (auto e : scc[i]) {
	coallocations_[i].push_back(individual_coallocations[e]);
      }
    }

    // Sort in ascending order based on the longest distance from the entry
    for (auto& coallocation : coallocations_) {
      std::sort(coallocation.begin(), coallocation.end(), [&](Coallocation& l, Coallocation& r) {
	  return distances[l.node_] < distances[r.node_];
      });
    }
  }

  std::vector<AllocationSchedule> CreateAllocationSchedule(AllocationHeuristic heuristic) {
    switch (heuristic) {
      case (AllocationHeuristic::Greedy): {
	return CreateAllocationScheduleGreedy();
	break;
      }
      default: {
	 LogPrintfError("Creating allocation schedule with heuristic unimplemented: %d", (int) heuristic);
	 break;
      }
    }
    return {};
  }

  std::vector<AllocationSchedule> CreateAllocationScheduleGreedy() {
    std::vector<AllocationSchedule> all_schedules;
    for (auto& coallocation : coallocations_) {
      AllocationSchedulerGreedy scheduler(values_, def_use_chains_, coallocation);
      auto actions = scheduler.Make();

      std::vector<GraphMemAllocNode*> alloc_nodes;
      size_t slab_size = 0;

      for (auto& e : actions) {
	if (e.type_ == AllocatorAction::Type::Allocate) {
	  auto alloc_node = dynamic_cast<GraphMemAllocNode*>(e.node_);
	  alloc_nodes.push_back(alloc_node);
	  slab_size = std::max(slab_size, e.offset_ + alloc_node->Bytesize());
	} else if (e.type_ == AllocatorAction::Type::Free) {
	  for (auto& value : values_) {
	    if (value.first_def_ == e.node_) {
	      e.node_ = value.free_node_;
	      break;
	    }
	  }
	}
      }

      all_schedules.push_back({actions, alloc_nodes, slab_size});
    }
    return all_schedules;
  }

  Graph* graph_;

  DominatorTree dt_;
  std::set<Value> values_;
  CoarseValues def_use_chains_;
  CoarseValues use_def_chains_;
  std::vector<std::vector<Coallocation>> coallocations_;
  std::vector<AllocationSchedule> all_schedules_;
  std::map<GraphNode*, std::pair<size_t, size_t>> changed_node_sizes_;
  bool needs_reschedule_ = false;
};

} // namespace ga
} // namespace hip
