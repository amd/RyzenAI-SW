#include <op_fuser/fuse_types.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "detail/graph_color.hpp"
#include "detail/meta_graph.hpp"
#include "passes.hpp"

static constexpr size_t TENSOR_PACK_ALIGNMENT = 4; // Bytes

namespace OpsFusion {

// Apply top-level validity checks on arg_map returned by the op.
static void validate_op_reqs(const std::vector<OpArgMap> &arg_map) {
  // Check 1 : Number of super kernel instrns should be <= 1
  auto num_super_instrns =
      std::count_if(arg_map.begin(), arg_map.end(), [](const OpArgMap &arg) {
        return arg.arg_type == OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT;
      });
  DOD_ASSERT(num_super_instrns <= 1,
             OpsFusion::dod_format(
                 "Number of super instruction buffers should be <= 1, given {}",
                 num_super_instrns));

  // Check 2 : Multiple const buffer with same xrt_arg_id is not allowed.
  std::vector<size_t> xrt_arg_ids;
  for (const auto &arg : arg_map) {
    if (arg.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      xrt_arg_ids.push_back(arg.xrt_arg_idx);
    }
  }
  std::set unique_xrt_arg_ids(xrt_arg_ids.begin(), xrt_arg_ids.end());
  DOD_ASSERT(
      xrt_arg_ids.size() == unique_xrt_arg_ids.size(),
      "Multiple const args with same xrt_arg_id is not allowed."
      "Can multiple const args be replaced with a single combined arg ?"
      "If not, this requires multi-const-arg feature to be enabled in DoD.");
}

// This function handles the I/O buffer requests from the operator.
// 1. For I/O buffer reqs from the kernel, DD allows an operator to request for
// a buffer size different from what is there in the original model, provided
// the requested buffer is always equal-to/larger-than the size in the model.
// 2. If two operators request for different sizes for a tensor shared by them,
// DD allocates the max of them.

using IOBufferInfo = std::pair<size_t, size_t>;
static void handle_io_tensors(const OpsFusion::Metadata::OpInfo &op_info,
                              const OpArgMap &req,
                              const OpsFusion::Metadata &meta,
                              std::map<std::string, IOBufferInfo> &io_bufs) {
  auto buf_name = ARRAY_AT(op_info.args, req.onnx_arg_idx);
  auto size_in_meta = MAP_AT(meta.tensor_map, buf_name).size_in_bytes;
  auto size_in_op = req.size;
  auto padding_offset = req.padding_offset;
  DOD_ASSERT(size_in_op >= size_in_meta + padding_offset,
             OpsFusion::dod_format(
                 "Size of IO buffer required by op ({}) is less "
                 "than the size in the model ({}) with padding ({})"
                 " for the node: {}",
                 size_in_op, size_in_meta, padding_offset, op_info.name));
  if (size_in_op > size_in_meta) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dod_format("[WARNING] Size of IO buffer required by "
                              "op spec ({}) is higher "
                              "than the size in the model ({})"
                              " for the node: {}",
                              size_in_op, size_in_meta, op_info.name));
  }

  if (io_bufs.end() == io_bufs.find(buf_name)) {
    io_bufs[buf_name] = std::make_pair(size_in_op, padding_offset);
  }

  io_bufs[buf_name].first = std::max(io_bufs[buf_name].first, size_in_op);
  if (io_bufs[buf_name].second != 0 && padding_offset != 0) {
    DOD_THROW_IF(io_bufs[buf_name].second == padding_offset,
                 "Different padding offset required for same IO buffer!");
  }
  io_bufs[buf_name].second = std::max(io_bufs[buf_name].second, padding_offset);
}

static void
update_io_buffers(OpsFusion::Metadata &meta,
                  const std::map<std::string, IOBufferInfo> &io_bufs) {

  const size_t max_tensor_padding_sz = meta.max_tensor_padding_sz;

  RYZENAI_LOG_TRACE("  Update IO pack buffer sizes and offsets");
  for (auto &[name, tensor_info] : meta.fused_tensors) {
    if (name == "const" || name == "super_instr") {
      continue;
    }
    // this fixes up tensor offset for packed/fused "in", "out", "scratch"
    // tensors "in" and "out" are input/output of subgraph
    // "scratch" tensor is intermediate outputs of subgraph
    size_t tensor_size = max_tensor_padding_sz;
    for (const auto &sub_tensor_name : tensor_info.packed_tensors) {
      auto &tinfo = MAP_AT(meta.tensor_map, sub_tensor_name);
      auto [sub_tensor_size, _sub_tensor_padding] =
          MAP_AT(io_bufs, sub_tensor_name);
      // For now ignore padding for inputs of later tensors are in scratch pad
      // (space optimization) have first tensor with an offset of
      // max_tensor_padding_sz i.e. will read adjacent tensors in
      // input/scratch BO
      tinfo.offset = tensor_size;
      tinfo.size_in_bytes = sub_tensor_size;
      tensor_size += sub_tensor_size;
      tensor_size = Utils::align_to_next(tensor_size, TENSOR_PACK_ALIGNMENT);
    }
    tensor_info.size = tensor_size;
  }
}

static void
update_superkernel_buffers(OpsFusion::Metadata &meta,
                           const std::vector<size_t> &super_instr_bufs) {
  RYZENAI_LOG_TRACE("  Update super buffer sizes and offsets");
  DOD_ASSERT(super_instr_bufs.size() == meta.op_list.size(),
             OpsFusion::dod_format("#Ops({}) != #SuperInstrs({})",
                                   meta.op_list.size(),
                                   super_instr_bufs.size()));
  size_t tensor_size = 0;
  for (size_t i = 0; i < meta.op_list.size(); ++i) {
    const auto &op_info = meta.op_list[i];
    auto op_size = super_instr_bufs[i];
    meta.super_instr_map[op_info.name] = {/*offset*/ tensor_size,
                                          /*size*/ op_size};
    tensor_size += op_size;
    tensor_size = Utils::align_to_next(tensor_size, TENSOR_PACK_ALIGNMENT);
  }
  MAP_AT(meta.fused_tensors, "super_instr").size = tensor_size;
}

static void update_op_scratch_buffers(OpsFusion::Metadata &meta,
                                      const std::vector<size_t> &scratch_bufs) {
  RYZENAI_LOG_TRACE("  Update scratch buffer sizes and offsets");
  DOD_ASSERT(scratch_bufs.size() == meta.op_list.size(),
             OpsFusion::dod_format("#Ops({}) != #Scratch({})",
                                   meta.op_list.size(), scratch_bufs.size()));

  size_t max_op_scratch_pad_size = 0;
  meta.scratch_op_set.clear();

  // NOTE: assumption is each op will be sequentially executed
  //       so this scratch pad can be reused
  //       idea is to place at end of intermediate buffers
  for (size_t i = 0; i < meta.op_list.size(); ++i) {
    const auto &op_info = meta.op_list[i];
    auto op_size = scratch_bufs[i];
    if (op_size != 0) {
      meta.scratch_op_set.insert(op_info.name);
      max_op_scratch_pad_size = std::max(max_op_scratch_pad_size, op_size);
    }
  }

  // Need to maintain this, since it will be used for BO size
  meta.max_op_scratch_pad_size =
      Utils::align_to_next(max_op_scratch_pad_size, TENSOR_PACK_ALIGNMENT);

  auto intermediate_scratch_size = MAP_AT(meta.fused_tensors, "scratch").size;

  MAP_AT(meta.fused_tensors, "scratch").size =
      Utils::align_to_next(intermediate_scratch_size, TENSOR_PACK_ALIGNMENT);

  RYZENAI_LOG_TRACE(OpsFusion::dod_format(
      "    curr_scratch_buf_size : {}, max_op_scratch_pad_size : {}",
      MAP_AT(meta.fused_tensors, "scratch").size, max_op_scratch_pad_size));
}

static void update_const_buffers(
    OpsFusion::Metadata &meta,
    const std::vector<std::vector<std::pair<size_t, size_t>>> &const_bufs) {
  RYZENAI_LOG_TRACE("  Update Const buffer sizes and offsets");
  size_t const_tensor_size = 0;
  for (size_t i = 0; i < meta.op_list.size(); ++i) {
    const auto &op_info = meta.op_list[i];
    for (const auto &[xrt_arg_id, buf_size] : const_bufs[i]) {
      meta.const_map[op_info.name] = {/*offset*/ const_tensor_size,
                                      /*size*/ buf_size};
      const_tensor_size += buf_size;
      const_tensor_size =
          Utils::align_to_next(const_tensor_size, TENSOR_PACK_ALIGNMENT);
    }
  }
  meta.fused_tensors["const"].size = const_tensor_size;
}

// This pass do an initial buffer analysis for all the tensors based on the
// requirements from the op interface.
void analyze_buffer_reqs(Metadata &meta) {
  RYZENAI_LOG_TRACE("Analyzing Buffer Reqs ... START");

  // io bufname -> [size, padding]
  std::map<std::string, IOBufferInfo> io_bufs;
  std::vector<size_t> super_instr_bufs;
  std::vector<size_t> scratch_bufs;

  // [#ops x #consts x (xrt_arg_id, size)]
  std::vector<std::vector<std::pair<size_t, size_t>>> const_bufs;

  size_t max_tensor_padding_sz = 0;

  // Collect All Ops Buffer Reqs
  for (const auto &op_info : meta.op_list) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "    Analyze - OpName : {}, OpType : {}", op_info.name, op_info.type));

    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);
    auto tensors = MetaUtils::collect_op_tensors(meta, op_info);
    auto buf_reqs = DD_INVOKE_OPMETHOD(get_buffer_reqs, op.get(), op_info,
                                       tensors, tensors, op_info.attr);

    RYZENAI_LOG_TRACE(OpsFusion::dod_format("      Op Buffer Reqs\n{}",
                                            cvt_to_string(buf_reqs)));

    validate_op_reqs(buf_reqs);

    std::vector<std::pair<size_t, size_t>> consts_in_req;
    size_t super_instr_sz = 0;
    size_t scratch_sz = 0;
    for (const auto &req : buf_reqs) {
      if (req.arg_type == OpArgMap::OpArgType::INPUT ||
          req.arg_type == OpArgMap::OpArgType::OUTPUT) {
        handle_io_tensors(op_info, req, meta, io_bufs);
        max_tensor_padding_sz =
            std::max(max_tensor_padding_sz, req.padding_offset);
      } else if (req.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
        consts_in_req.emplace_back(req.xrt_arg_idx, req.size);
      } else if (req.arg_type ==
                 OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT) {
        super_instr_sz = req.size;
      } else if (req.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
        scratch_sz = req.size;
      } else {
        DOD_THROW(dod_format("Unhandled OpArgType in buffer requirements of "
                             "node:{}, node type:{}",
                             op_info.name, op_info.type));
      }
    } // for req

    const_bufs.push_back(std::move(consts_in_req));
    super_instr_bufs.push_back(super_instr_sz);
    scratch_bufs.push_back(scratch_sz);
  } // for op

  meta.max_tensor_padding_sz =
      Utils::align_to_next(max_tensor_padding_sz, TENSOR_PACK_ALIGNMENT);

  update_io_buffers(meta, io_bufs);
  update_superkernel_buffers(meta, super_instr_bufs);
  update_const_buffers(meta, const_bufs);
  update_op_scratch_buffers(meta, scratch_bufs);

  RYZENAI_LOG_TRACE("Analyzing Buffer Reqs ... END");
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("\nMeta Summary after Analysis:\n{}",
                                          MetaUtils::get_summary(meta)));
}

} // namespace OpsFusion
