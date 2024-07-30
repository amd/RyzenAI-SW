#include <unordered_set>

#include <op_fuser/fuse_types.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "detail/graph_color.hpp"
#include "detail/meta_graph.hpp"
#include "passes.hpp"

/*
1. When this pass is called, it assumes that an initial buffer analysis is
already done and the buffer shapes for all scratch buffer is already computed in
meta.

2. As the name specifies, this optim is only for buffers in scratch space. Not
in buffers in input/output/const/super-param

3. TODO : With current meta structure, it is hard to differentiate b/w input &
output tensors of an op from the meta itself. Changing meta structure might need
a larger change in both DD & vaip. So for now, this impl will rely on
op->get_buff_reqs() to identify inputs & outputs.

There are mutliple better solutions here.
    3.a. (P0) Differentiate ip & op at meta level itself.
    3.b. (P1) Create non-DAG graph of tensor connections.
*/

using namespace OpsFusion::Pass::detail;

static constexpr size_t TENSOR_PACK_ALIGNMENT = 4; // Bytes

namespace OpsFusion {

static std::map<std::string, int> create_tensor_id_map(const Metadata &meta) {
  std::map<std::string, int> tensor_id_map;
  int id = 0;
  for (const auto &[t_name, t_info] : meta.tensor_map) {
    if (t_info.parent_name != "const") {
      tensor_id_map[t_name] = id++;
    }
  }
  return tensor_id_map;
}

static AdjList
create_tensor_connectivity(const Metadata &meta,
                           const std::map<std::string, int> &tensor_id_map) {
  MetaGraph meta_graph(meta);

  AdjList adj_list;
  std::unordered_set<std::string> visited_tensors;
  for (const auto &op_info : meta.op_list) {
    const auto &op_inputs = meta_graph.get_op_inputs(op_info.name);
    const auto &op_outputs = meta_graph.get_op_outputs(op_info.name);
    for (const auto &op_output : op_outputs) {
      auto op_output_id = MAP_AT(tensor_id_map, op_output);
      if (visited_tensors.find(op_output) == visited_tensors.end()) {
        visited_tensors.insert(op_output);
        adj_list[op_output_id] = {};
      }
      for (const auto &op_input : op_inputs) {
        auto op_input_id = MAP_AT(tensor_id_map, op_input);
        adj_list[op_input_id].push_back(op_output_id);
      }
    }
  }

  RYZENAI_LOG_TRACE(dod_format("AdjList :\n{}", adj_list));

  return adj_list;
}

static void
remove_io_from_labels(std::map<node_t, label_t> &node_labels,
                      const std::map<std::string, node_t> &tensor_id_map,
                      const Metadata &meta) {
  RYZENAI_LOG_TRACE(dod_format("Labels :\n{}", node_labels));

  const auto &in_tensors = MAP_AT(meta.fused_tensors, "in").packed_tensors;
  for (const auto &tname : in_tensors) {
    auto tid = MAP_AT(tensor_id_map, tname);
    node_labels.erase(tid);
  }

  const auto &out_tensors = MAP_AT(meta.fused_tensors, "out").packed_tensors;
  for (const auto &tname : out_tensors) {
    auto tid = MAP_AT(tensor_id_map, tname);
    node_labels.erase(tid);
  }

  RYZENAI_LOG_TRACE(dod_format("Labels after removing IO:\n{}", node_labels));
}

static std::map<label_t, std::vector<node_t>>
node_labels_to_label_nodes(const std::map<node_t, label_t> &src) {
  std::map<label_t, std::vector<node_t>> dst;
  for (const auto &[key, value] : src) {
    dst[value] = {};
  }

  for (const auto &[key, value] : src) {
    dst[value].push_back(key);
  }

  RYZENAI_LOG_TRACE(dod_format("Label and Nodes :\n{}", dst));

  return dst;
}

template <typename Key, typename Value>
static std::map<Value, Key> reverse_1to1_map(const std::map<Key, Value> &dict) {
  std::map<Value, Key> reverse_dict;
  for (const auto &[key, val] : dict) {
    reverse_dict[val] = key;
  }
  return reverse_dict;
}

static std::map<label_t, size_t> compute_size_for_label(
    const std::map<label_t, std::vector<node_t>> &label_nodes,
    const std::map<int, std::string> &id_tensor_map, const Metadata &meta) {
  RYZENAI_LOG_TRACE("Computing Size for each label ... START");
  std::map<label_t, size_t> label_size;
  for (const auto &[label, tids] : label_nodes) {
    RYZENAI_LOG_TRACE(dod_format("  label : {}", label));
    size_t bucket_size = 0;
    for (auto tid : tids) {
      const auto &tname = MAP_AT(id_tensor_map, tid);
      const auto &tinfo = MAP_AT(meta.tensor_map, tname);
      bucket_size = std::max(bucket_size, tinfo.size_in_bytes);
      RYZENAI_LOG_TRACE(dod_format("    tid:{}, tensor:{}, size:{}", tid, tname,
                                   tinfo.size_in_bytes));
    }
    label_size[label] =
        Utils::align_to_next(bucket_size, TENSOR_PACK_ALIGNMENT);
  }

  RYZENAI_LOG_TRACE(dod_format("Label Size :\n{}", label_size));
  RYZENAI_LOG_TRACE("Computing Size for each label ... END");
  return label_size;
}

static size_t compute_total_size(const std::map<label_t, size_t> &label_size,
                                 size_t alignment = TENSOR_PACK_ALIGNMENT) {
  size_t total_size = 0;
  for (const auto &[label, size] : label_size) {
    total_size += size;
  }
  return total_size;
}

static std::map<label_t, size_t>
compute_label_offsets(const std::map<label_t, size_t> &label_size) {
  std::map<label_t, size_t> label_offsets;
  size_t total_size = 0;
  for (const auto &[label, size] : label_size) {
    label_offsets[label] = total_size;
    total_size += size;
  }

  RYZENAI_LOG_TRACE(dod_format("label offsets: \n{}", label_offsets));
  return label_offsets;
}

static void update_meta_scratch_space(
    Metadata &meta, const std::map<std::string, node_t> &tensor_id_map,
    const std::map<node_t, label_t> &node_labels,
    const std::map<label_t, size_t> &label_offsets, size_t total_size) {

  RYZENAI_LOG_TRACE("Patching meta scratch space ... START");

  const size_t max_tensor_padding_sz = meta.max_tensor_padding_sz;
  auto &scratch_buffer = MAP_AT(meta.fused_tensors, "scratch");

  scratch_buffer.size = Utils::align_to_next(total_size + max_tensor_padding_sz,
                                             TENSOR_PACK_ALIGNMENT);

  RYZENAI_LOG_TRACE(
      dod_format("Total Optimized Scratch Space : {}", scratch_buffer.size));

  for (const auto &tname : scratch_buffer.packed_tensors) {
    auto tid = MAP_AT(tensor_id_map, tname);
    auto label = MAP_AT(node_labels, tid);
    auto new_offset = MAP_AT(label_offsets, label);
    auto &tinfo = MAP_AT(meta.tensor_map, tname);
    auto old_offset = tinfo.offset;
    // have a fixed offset to support padding of input tensors in scratch
    // assumes these are just used for alignment/data read patterns and
    // not used for computation
    tinfo.offset = new_offset + max_tensor_padding_sz;

    RYZENAI_LOG_TRACE(
        dod_format("tid:{}, label:{}, orig_offset:{} --> new_offset:{}", tid,
                   label, old_offset, new_offset + max_tensor_padding_sz));
  }
  RYZENAI_LOG_TRACE("Patching meta scratch space ... END");
}

void optimize_scratch_buffer(Metadata &meta) {
  RYZENAI_LOG_TRACE("Buffer Reuse ... START");
  auto tensor_id_map = create_tensor_id_map(meta);
  auto adj_list = create_tensor_connectivity(meta, tensor_id_map);
  auto node_labels = color_graph(adj_list);
  remove_io_from_labels(node_labels, tensor_id_map, meta);
  auto label_nodes = node_labels_to_label_nodes(node_labels);
  auto id_tensor_map = reverse_1to1_map(tensor_id_map);
  auto label_size = compute_size_for_label(label_nodes, id_tensor_map, meta);
  auto total_scratch_size = compute_total_size(label_size);
  auto label_offsets = compute_label_offsets(label_size);
  update_meta_scratch_space(meta, tensor_id_map, node_labels, label_offsets,
                            total_scratch_size);
  RYZENAI_LOG_TRACE(MetaUtils::get_summary(meta));
  RYZENAI_LOG_TRACE("Buffer Reuse ... END");
}

} // namespace OpsFusion
