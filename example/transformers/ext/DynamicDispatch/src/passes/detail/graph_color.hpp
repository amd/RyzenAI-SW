#pragma once
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adjacency_graph.hpp"

namespace OpsFusion {
namespace Pass {
namespace detail {

using label_t = int;

static constexpr label_t UNDEFINED_LABEL = -1;

enum class VisitStatus { UNVISITED, VISITED };

// Check if all nodes in the
static bool all_labelled(const std::map<node_t, label_t> &node_labels,
                         const std::vector<node_t> &nodes) {
  for (auto node : nodes) {
    if (node_labels.at(node) == UNDEFINED_LABEL) {
      return false;
    }
  }
  return true;
}

static label_t get_unused_label(std::stack<label_t> &used_labels,
                                label_t &label_idx) {
  if (used_labels.empty()) {
    auto new_label = label_idx++;
    return new_label;
  }

  auto used_label = used_labels.top();
  used_labels.pop();
  return used_label;
}

static std::map<node_t, label_t> color_graph(const AdjList &graph) {
  std::queue<node_t> queue;
  std::stack<label_t> unused_labels;
  label_t label_id = 0;

  AdjGraph adj_graph(graph);

  // Visit status
  std::unordered_map<node_t, VisitStatus> visit_status;
  for (const auto &[node, next_nodes] : graph) {
    visit_status[node] = VisitStatus::UNVISITED;
  }

  // Result
  std::map<node_t, label_t> node_labels;
  for (const auto &[node, next_nodes] : graph) {
    node_labels[node] = UNDEFINED_LABEL;
  }

  std::vector<node_t> inputs = adj_graph.get_graph_inputs();
  for (auto input : inputs) {
    queue.push(input);
    visit_status[input] = VisitStatus::VISITED;
  }

  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();

    // If all parents are labelled, label the current node
    //   Otherwise, ask it to wait in the queue end.
    const auto &parents = adj_graph.get_parents(node);
    if (all_labelled(node_labels, parents)) {
      label_t label = get_unused_label(unused_labels, label_id);
      node_labels[node] = label;
    } else {
      queue.push(node);
      continue;
    }

    // Now that this node is labelled, check if all other children of its
    // parents are also labelled. If so, the parent can release its label
    // for others to use.
    for (auto parent : parents) {
      const auto &children = adj_graph.get_children(parent);
      if (all_labelled(node_labels, children)) {
        unused_labels.push(node_labels[parent]);
      }
    }

    // Push the children of current node to queue
    const auto &children = adj_graph.get_children(node);
    for (auto child : children) {
      if (visit_status.at(child) == VisitStatus::UNVISITED) {
        queue.push(child);
        visit_status.at(child) = VisitStatus::VISITED;
      }
    }
  }

  return node_labels;
}

} // namespace detail
} // namespace Pass
} // namespace OpsFusion

template <typename V>
static std::ostream &operator<<(std::ostream &os, const std::vector<V> &vec) {
  for (const auto &v : vec) {
    os << v << ", ";
  }
  return os;
}

template <typename K, typename V>
static std::ostream &operator<<(std::ostream &os, const std::map<K, V> &dict) {
  for (const auto &[k, v] : dict) {
    os << k << " : " << v << "\n";
  }
  return os;
}

template <typename K, typename V>
static std::ostream &operator<<(std::ostream &os,
                                const std::map<K, std::vector<V>> &dict) {
  for (const auto &[k, v] : dict) {
    os << k << " : " << v << "\n";
  }
  return os;
}

// Test Linear chain
static void test1() {
  std::cout << "TEST 1 " << std::endl;
  OpsFusion::Pass::detail::AdjList child_graph = {
      {0, {1}}, {1, {2}}, {2, {3}}, {3, {}}};

  OpsFusion::Pass::detail::AdjGraph graph(child_graph);

  std::cout << graph.get_graph_inputs() << std::endl;

  auto label_nodes = OpsFusion::Pass::detail::color_graph(child_graph);
  std::cout << "Label & Nodes :\n" << label_nodes << std::endl;
}

// Skip connection
static void test2() {
  std::cout << "TEST 2 " << std::endl;
  OpsFusion::Pass::detail::AdjList child_graph = {
      {1, {2}}, {2, {3, 5}}, {3, {4}}, {4, {5}}, {5, {}}};

  OpsFusion::Pass::detail::AdjGraph graph(child_graph);

  std::cout << graph.get_graph_inputs() << std::endl;

  auto label_nodes = OpsFusion::Pass::detail::color_graph(child_graph);
  std::cout << "Label & Nodes :\n" << label_nodes << std::endl;
}
