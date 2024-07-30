#pragma once

#include <any>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

#include "fuse_types.hpp"
#include <ops/op_builder.hpp>
#include <ops/op_interface.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>

#include "txn/txn_utils.hpp"

using json = nlohmann::json;

namespace OpsFusion {

static std::ostream &operator<<(std::ostream &os, const Metadata &m) {
  for (const auto &opinfo : m.op_list) {
    os << opinfo.name << " : " << opinfo.type << " : ";
    for (auto &arg : opinfo.args) {
      os << arg << " ";
    }
    os << std::endl;
  }

  for (const auto &[name, tinfo] : m.fused_tensors) {
    os << name << " : " << tinfo.size << ", " << tinfo.arg_idx << std::endl;
  }

  for (const auto &[name, off_info] : m.tensor_map) {
    os << name << " : " << off_info.parent_name << ", " << off_info.offset
       << ", " << off_info.arg_idx << std::endl;
  }

  for (const auto &[name, span_info] : m.super_instr_map) {
    os << "Super Kernel Instr Span : " << name << " : " << span_info.offset
       << ", " << span_info.size << std::endl;
  }
  for (const auto &[key, span_info] : m.const_map) {
    os << "const Span : " << key << " - " << span_info.offset << ", "
       << span_info.size << std::endl;
  }

  return os;
}

using txn_vec_t = std::vector<uint8_t>;

template <typename T>
static T json_get(const json &js, const std::string &key, const T &value) {
  return js.find(key) != js.end() ? js.at(key).template get<T>() : value;
}

static std::map<std::string, std::any> extract_op_attrs(const json &op_info) {
  std::map<std::string, std::any> attrs;
  if (op_info.find("attrs") == op_info.end()) {
    return attrs;
  }

  for (const auto &[attr_name, attr_info] : op_info.at("attrs").items()) {
    const std::string dtype = attr_info.at("type").template get<std::string>();
    const std::vector<std::string> values =
        attr_info.at("value").template get<std::vector<std::string>>();

    if (dtype == "float") {
      attrs[attr_name] =
          for_each(values, [](const auto &s) { return std::stof(s); });
    } else if (dtype == "int") {
      attrs[attr_name] =
          for_each(values, [](const auto &s) { return std::stoi(s); });
    } else if (dtype == "str") {
      attrs[attr_name] = values;
    } else {
      DOD_THROW(OpsFusion::dod_format("Unsupported dtype for attrs in JSON: {}",
                                      dtype));
    }
  }
  return attrs;
}

static std::map<std::string, std::any> load_aux_info(const json &aux_info) {
  std::map<std::string, std::any> res;

  // Original outputs
  {
    if (aux_info.find("original_outputs") != aux_info.end()) {
      std::map<std::string, Tensor> tensors;
      for (const auto &[name, tinfo] :
           aux_info.at("original_outputs").items()) {
        Tensor tensor{nullptr,
                      tinfo.at("shape").template get<std::vector<size_t>>(),
                      tinfo.at("dtype").template get<std::string>()};
        tensors[name] = tensor;
      }
      res["original_outputs"] = std::any(tensors);
    }
  }

  // Original Inputs
  {
    if (aux_info.find("original_inputs") != aux_info.end()) {
      std::map<std::string, Tensor> tensors;
      for (const auto &[name, tinfo] : aux_info.at("original_inputs").items()) {
        Tensor tensor{nullptr,
                      tinfo.at("shape").template get<std::vector<size_t>>(),
                      tinfo.at("dtype").template get<std::string>()};
        tensors[name] = tensor;
      }
      res["original_inputs"] = std::any(tensors);
    }
  }

  return res;
}

static Metadata load_meta_json(const std::string &meta_json) {
  RYZENAI_LOG_TRACE("Loading the meta.json ...");
  Metadata meta;
  std::ifstream ifs(meta_json);
  DOD_ASSERT(ifs.is_open(),
             OpsFusion::dod_format("Couldn't open JSON : {}", meta_json));

  // TODO : Nothing is caught while parse error
  json data;
  try {
    data = json::parse(ifs, nullptr, true);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    DOD_THROW(OpsFusion::dod_format("Failed to parse JSON: {} (Detail: {})",
                                    meta_json, e.what()));
  }
  RYZENAI_LOG_TRACE("Loading the meta.json ... DONE");

  meta.json_path = meta_json;
  // oplist
  for (const auto &opinfo : data.at("op_list")) {
    meta.op_list.push_back(
        {opinfo.at("name").template get<std::string>(),
         opinfo.at("type").template get<std::string>(),
         opinfo.at("args").template get<std::vector<std::string>>(),
         {}});
    meta.op_list.back().attr = extract_op_attrs(opinfo);
  }

  // tensor info
  for (const auto &[name, tinfo] : data.at("fused_tensors").items()) {
    meta.fused_tensors[name] = {
        tinfo.at("buffer_size").template get<size_t>(),
        tinfo.at("xrt_arg_id").template get<size_t>(),
        tinfo.at("packed_tensors").template get<std::vector<std::string>>()};
  }

  // tensor_map
  for (const auto &[name, offset_info] : data.at("tensor_map").items()) {
    meta.tensor_map[name] = {
        offset_info.at("packed_buffer_label").template get<std::string>(),
        offset_info.at("offset").template get<size_t>(),
        offset_info.at("xrt_arg_id").template get<size_t>(),
        offset_info.at("dtype").template get<std::string>(),
        offset_info.at("shape").template get<std::vector<size_t>>(),
        offset_info.at("size_in_bytes").template get<size_t>(),
        json_get<std::string>(offset_info, "file_name", ""),
        json_get<size_t>(offset_info, "file_size", 0)};
  }

  if (data.find("aux_info") != data.end()) {
    meta.aux_info = load_aux_info(data.at("aux_info"));
  }

  RYZENAI_LOG_TRACE("Filling Metadata ... DONE");
  return meta;
}

static txn_vec_t generate_fused_ops(const Metadata &meta,
                                    const std::pair<size_t, size_t> &op_range) {
  RYZENAI_LOG_TRACE("Get ops txn ...");
  std::vector<txn_vec_t> txn_vecs;
  size_t const num_ops = op_range.second > op_range.first
                             ? (op_range.second - op_range.first)
                             : (1);
  txn_vecs.reserve(num_ops);

  for (auto ind = op_range.first; ind < op_range.second; ind++) {
    const auto &op_info = meta.op_list.at(ind);
    RYZENAI_LOG_TRACE(
        OpsFusion::dod_format("Get ops txn for op:{}", op_info.name));
    auto op = OpBuilder::create(op_info.name, op_info, meta.tensor_map);

    auto const_buffers = MetaUtils::load_op_const_buffers(meta, op_info);
    std::map<std::string, void *> const_buf_ptrs;
    for (auto &[name, buffer] : const_buffers) {
      const_buf_ptrs[name] = buffer.data();
    }

    std::vector<Tensor> tensors =
        MetaUtils::collect_op_tensors(meta, op_info, const_buf_ptrs);

    auto txn_vec = DD_INVOKE_OPMETHOD(get_transaction_bin, op.get(), op_info,
                                      tensors, tensors, op_info.attr);
    auto args_map = DD_INVOKE_OPMETHOD(get_buffer_reqs, op.get(), op_info,
                                       tensors, tensors, op_info.attr);
    utils::txn_util patched_txn(txn_vec);
    patched_txn.patch(op_info, meta, args_map);
    txn_vecs.push_back(std::move(patched_txn.to_vector()));
  }
  auto fused_txn = utils::txn_util::fuse_txns(txn_vecs);
  RYZENAI_LOG_TRACE("Get ops txn ... DONE");
  return fused_txn;
}

} // namespace OpsFusion
