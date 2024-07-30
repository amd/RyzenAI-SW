#include <sstream>
#include <utils/meta_utils.hpp>

namespace OpsFusion {

std::string MetaUtils::get_summary(const Metadata &meta) {
  std::ostringstream oss;
  oss << "Summary of Metadata\n";
  oss << "-------------------\n";
  oss << "Total number of Ops : " << meta.op_list.size() << "\n";
  // OpCount
  std::map<std::string, size_t> op_count;
  for (const auto &op : meta.op_list) {
    op_count[op.type]++;
  }
  for (const auto &[op_type, cnt] : op_count) {
    oss << "  #" << op_type << " : " << cnt << "\n";
  }

  // MemReqs
  size_t total_mem = std::accumulate(
      meta.fused_tensors.begin(), meta.fused_tensors.end(), size_t{0},
      [](size_t accum, const auto &item) { return accum + item.second.size; });

  oss << "\n";
  oss << "Total Device Memmory (B) : " << total_mem << "\n";
  oss << "  Input Memory (B) : " << meta.fused_tensors.at("in").size << "\n";
  oss << "  Output Memory (B) : " << meta.fused_tensors.at("out").size << "\n";
  oss << "  Scratch Memory (B) : " << meta.fused_tensors.at("scratch").size
      << "\n";
  oss << "  Const Memory (B) : " << meta.fused_tensors.at("const").size << "\n";
  oss << "  SuperKernel Memory (B) : "
      << meta.fused_tensors.at("super_instr").size << "\n";
  oss << "-------------------\n";

  return oss.str();
}

size_t MetaUtils::get_num_inputs(const Metadata &meta) {
  return MetaUtils::get_num_tensors(meta, OpArgMap::OpArgType::INPUT);
}

size_t MetaUtils::get_num_outputs(const Metadata &meta) {
  return MetaUtils::get_num_tensors(meta, OpArgMap::OpArgType::OUTPUT);
}

std::vector<Tensor> MetaUtils::get_input_tensors(const Metadata &meta) {
  return MetaUtils::get_tensors(meta, OpArgMap::OpArgType::INPUT);
}

std::vector<Tensor> MetaUtils::get_output_tensors(const Metadata &meta) {
  return MetaUtils::get_tensors(meta, OpArgMap::OpArgType::OUTPUT);
}

std::vector<Tensor> MetaUtils::get_const_tensors(const Metadata &meta) {
  return MetaUtils::get_tensors(meta, OpArgMap::OpArgType::CONST_INPUT);
}

std::vector<Tensor> MetaUtils::get_tensors(const Metadata &meta,
                                           OpArgMap::OpArgType arg_type) {
  std::vector<Tensor> res;
  const auto tensor_name = convert_argtype_to_string(arg_type);
  for (const auto &inp : meta.fused_tensors.at(tensor_name).packed_tensors) {
    const auto &tensor = meta.tensor_map.at(inp);
    Tensor t{/*data*/ nullptr,
             /*shape*/ tensor.shape,
             /*dtype*/ tensor.dtype};
    res.push_back(t);
  }
  return res;
}

size_t MetaUtils::get_num_tensors(const Metadata &meta,
                                  OpArgMap::OpArgType arg_type) {
  const auto tensor_name = convert_argtype_to_string(arg_type);
  return meta.fused_tensors.at(tensor_name).packed_tensors.size();
}

std::map<std::string, std::vector<char>>
MetaUtils::load_op_const_buffers(const Metadata &meta,
                                 const Metadata::OpInfo &op_info) {
  std::map<std::string, std::vector<char>> const_buffers;
  for (auto &tensor_name : op_info.args) {
    const auto &tinfo = MAP_AT(meta.tensor_map, tensor_name);
    if (tinfo.parent_name == "const") {
      DOD_ASSERT(!tinfo.file_name.empty(),
                 dod_format("Tensor:{} is mapped to constant, but no "
                            "associated filename provided",
                            tensor_name));

      auto const_buffer = read_bin_file(tinfo.file_name);
      DOD_ASSERT(const_buffer.size() == tinfo.file_size,
                 dod_format("Const tensor size doesn't match.\n  Tensor: "
                            "{}\n  Size in JSON: {}\n  Size of file: {}",
                            tensor_name, tinfo.file_size, const_buffer.size()));

      const_buffers[tensor_name] = std::move(const_buffer);
    }
  }
  return const_buffers;
}

std::vector<Tensor> MetaUtils::collect_op_tensors(
    const Metadata &meta, const Metadata::OpInfo &op_info,
    const std::map<std::string, void *> &const_buffer_ptrs) {
  std::vector<Tensor> tensors;
  bool enable_real_const_buffer_ptr = !const_buffer_ptrs.empty();
  for (auto &tensor_name : op_info.args) {
    const auto &tinfo = MAP_AT(meta.tensor_map, tensor_name);

    void *tensor_ptr = nullptr;
    if (enable_real_const_buffer_ptr && tinfo.parent_name == "const") {
      auto const_buffer_ptr = MAP_AT(const_buffer_ptrs, tensor_name);
      tensor_ptr = const_buffer_ptr;
    }

    tensors.push_back({tensor_ptr, meta.tensor_map.at(tensor_name).shape,
                       meta.tensor_map.at(tensor_name).dtype});
  }
  return tensors;
}

} // namespace OpsFusion
