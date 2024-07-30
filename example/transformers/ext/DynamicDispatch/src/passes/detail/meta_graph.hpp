#pragma once

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>

#include <ops/op_interface.hpp>
#include <unordered_set>
#include <utils/tfuncs.hpp>

namespace OpsFusion {
namespace Pass {
namespace detail {

/*
  Graph API based on Metadata Structure.
  This class will act as a base API to treat Metadata as a graph.
  This will be useful if some passes require to traverse through the metadata
  like a graph.
*/

class MetaGraph {
public:
  MetaGraph() = default;
  MetaGraph(OpsFusion::Metadata meta) : meta_(std::move(meta)) {
    fill_node_inputs_outputs();
  }

  /// @brief Get the input tensor names of the graph
  /// @return
  const std::vector<std::string> &get_input_tensors() const {
    return MAP_AT(meta_.fused_tensors, "in").packed_tensors;
  }

  /// @brief Get the output tensor names of the graph
  /// @return
  const std::vector<std::string> &get_output_tensors() const {
    return MAP_AT(meta_.fused_tensors, "out").packed_tensors;
  }

  /// @brief Get the input tensor names of an Op in the graph
  /// @param op_name Name of the Op
  /// @return
  const std::vector<std::string> &
  get_op_inputs(const std::string &op_name) const {
    return MAP_AT(node_inputs_, op_name);
  }

  /// @brief Get the input tensor names of an Op in the graph
  /// @param op_name Name of the Op
  /// @return
  const std::vector<std::string> &
  get_op_outputs(const std::string &op_name) const {
    return MAP_AT(node_outputs_, op_name);
  }

private:
  /// @brief Finds the input/output tensor names of each Op in the graph and
  /// cache it for later access.
  void fill_node_inputs_outputs() {
    std::unordered_set<std::string> visited_tensors;
    for (const auto &op_info : meta_.op_list) {
      node_inputs_[op_info.name] = {};
      node_outputs_[op_info.name] = {};

      // TODO : Accessing Op just to identify inputs/outputs is significant
      // overhead. This can be resolved by updating meta structure.
      auto op = OpBuilder::create(op_info.name, op_info, meta_.tensor_map);
      auto tensors = MetaUtils::collect_op_tensors(meta_, op_info);
      auto args_map = DD_INVOKE_OPMETHOD(get_buffer_reqs, op.get(), op_info,
                                         tensors, tensors, op_info.attr);
      for (const auto &arg : args_map) {
        if (arg.arg_type == OpArgMap::OpArgType::INPUT) {
          const auto &tensor_name = ARRAY_AT(op_info.args, arg.onnx_arg_idx);
          node_inputs_[op_info.name].push_back(tensor_name);
        } else if (arg.arg_type == OpArgMap::OpArgType::OUTPUT) {
          const auto &tensor_name = ARRAY_AT(op_info.args, arg.onnx_arg_idx);
          node_outputs_[op_info.name].push_back(tensor_name);
        }
      }
    }
  }

private:
  OpsFusion::Metadata meta_;

  /// @brief Map of OpName --> Input tensor names
  std::map<std::string, std::vector<std::string>> node_inputs_;

  /// @brief Map of OpName --> Output tensor names
  std::map<std::string, std::vector<std::string>> node_outputs_;
};

} // namespace detail
} // namespace Pass
} // namespace OpsFusion
