#pragma once

#include <map>

namespace OpsFusion {
namespace Pass {
namespace detail {

using node_t = int;
using AdjList = std::map<node_t, std::vector<node_t>>;

// Convert an AdjList from "node -> child_nodes" to "node -> parent_nodes"
static AdjList child_graph_to_parent_graph(const AdjList &child_graph) {
  AdjList parent_graph;
  for (const auto &[key, value] : child_graph) {
    parent_graph[key] = {};
  }

  for (const auto &[node, children] : child_graph) {
    for (auto child : children) {
      parent_graph[child].push_back(node);
    }
  }

  return parent_graph;
}

class AdjGraph {
public:
  AdjGraph() = default;
  AdjGraph(AdjList child_graph) : child_graph_(std::move(child_graph)) {
    parent_graph_ = child_graph_to_parent_graph(child_graph_);
  }

  // Get Graph Inputs : nodes with no parents
  std::vector<node_t> get_graph_inputs() const {
    std::vector<node_t> inputs;
    for (const auto &[node, parents] : parent_graph_) {
      if (parents.empty()) {
        inputs.push_back(node);
      }
    }
    return inputs;
  }

  const std::vector<node_t> &get_children(node_t node) const {
    return child_graph_.at(node);
  }

  const std::vector<node_t> &get_parents(node_t node) const {
    return parent_graph_.at(node);
  }

private:
  AdjList child_graph_;
  AdjList parent_graph_;
};

} // namespace detail
} // namespace Pass
} // namespace OpsFusion
