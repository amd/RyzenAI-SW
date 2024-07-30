#pragma once

#include <op_fuser/fuse_types.hpp>
#include <ops/op_interface.hpp>
#include <utils/tfuncs.hpp>

namespace OpsFusion {

class MetaUtils {
public:
  /// @brief Get a summary of the metdata like number of each ops, total memory
  /// consumption etc.
  static std::string get_summary(const Metadata &meta);

  /// @brief Get the input tensors of the metadata
  static std::vector<Tensor> get_input_tensors(const Metadata &meta);

  /// @brief Get the output tensors of the metadata
  static std::vector<Tensor> get_output_tensors(const Metadata &meta);

  /// @brief Get all the const tensors of the metadata
  static std::vector<Tensor> get_const_tensors(const Metadata &meta);

  /// @brief Get number of input tensors of the metadata
  static size_t get_num_inputs(const Metadata &meta);

  /// @brief Get number of output tensors of the metadata
  static size_t get_num_outputs(const Metadata &meta);

  /// @brief Load all the constant data from binary files associated with an op
  static std::map<std::string, std::vector<char>>
  load_op_const_buffers(const Metadata &meta, const Metadata::OpInfo &op_info);

  /// @brief Get a list of args of an Op as Tensors. If const_buffer_ptrs are
  /// passed, Tensor::data will point to a valid buffer in const_buffer_ptrs,
  /// otherwise, Tensor::data will be nullptr by default.
  /// MetaUtils::load_op_const_buffers() can be used to get const buffers for an
  /// op.
  static std::vector<Tensor> collect_op_tensors(
      const Metadata &meta, const Metadata::OpInfo &op_info,
      const std::map<std::string, void *> &const_buffer_ptrs = {});

private:
  static std::vector<Tensor> get_tensors(const Metadata &meta,
                                         OpArgMap::OpArgType arg_type);
  static size_t get_num_tensors(const Metadata &meta,
                                OpArgMap::OpArgType arg_type);
};
} // namespace OpsFusion
