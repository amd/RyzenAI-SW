#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <ops/record_timer/record_timer.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

using json = nlohmann::json;

namespace OpsFusion {

namespace profile_ids {
static uint32_t timer_id = 0;
}

static inline std::string shape_to_string(const std::vector<size_t> &shape) {
  std::string shape_str = "";
  for (auto it = shape.begin(); it != shape.end(); it++) {
    shape_str += std::to_string(*it);
    if (std::next(it) != shape.end()) {
      shape_str += "x";
    }
  }
  return shape_str;
}

static std::tuple<std::string, std::string, size_t>
get_dtype_shape_from_op(const Metadata &meta, const Metadata::OpInfo &op) {
  const auto args = op.args;
  const auto t_map = meta.tensor_map;

  std::string shape_str = "";
  std::string dtype_str = "";
  auto num_args = args.size();

  for (auto it = args.begin(); it != args.end(); it++) {
    auto arg_info = t_map.find(*it);
    shape_str += shape_to_string(arg_info->second.shape);
    dtype_str += arg_info->second.dtype;
    if (std::next(it) != args.end()) {
      shape_str += "_";
      dtype_str += "_";
    }
  }
  return std::make_tuple(shape_str, dtype_str, num_args);
}

static json get_op_prop(const Metadata &meta, const Metadata::OpInfo &op) {
  auto [shape, dtype, num_args] = get_dtype_shape_from_op(meta, op);
  json op_prop = {
      {{"name", "dtype"},
       {"type", "string"},
       {"tooltip", ""},
       {"value", dtype}},
      {{"name", "shape"},
       {"type", "string"},
       {"tooltip", ""},
       {"value", shape}},
      {{"name", "pdi_id"},
       {"type", "int"},
       {"tooltip", ""},
       {"value", std::to_string(op.pdi_id)}},
      {{"name", "num_args"},
       {"type", "int"},
       {"tooltip", ""},
       {"value", std::to_string(num_args)}},
  };
  return op_prop;
}

static void insert_timer_op_in_meta(Metadata &meta, const std::string &op_name,
                                    uint8_t pdi_id, uint32_t timer_id) {

  std::map<std::string, std::any> attr;
  attr["timer_id"] = profile_ids::timer_id;
  attr["op_name"] = op_name;
  Metadata::OpInfo timer_info = {
      "timer_id_" + profile_ids::timer_id, "RECORD_TIMER", {}, attr, pdi_id};
  meta.op_list.emplace_back(timer_info);
}

static void insert_timer_info_in_dd_json(json &dd_ts, const json &op_prop,
                                         const Metadata::OpInfo &op,
                                         uint32_t timer_id, bool start,
                                         uint32_t parent_id) {
  std::string parent_pdi_subgraph =
      "pdi_partition_" + std::to_string(op.pdi_id);
  dd_ts["events"].push_back({{"id", profile_ids::timer_id},
                             {"name", op.name},
                             {"op_type", op.type},
                             {"start", start ? true : false},
                             {"type", "layer"},
                             {"parent", parent_pdi_subgraph},
                             {"properties", op_prop}});
}

Metadata insert_record_timer_nodes(const Metadata &meta,
                                   uint32_t profile_level) {
  RYZENAI_LOG_TRACE("Insert Record Timer nodes: Init");
  const std::string dd_timestamp_fname = "dd_timestamp_info.json";
  Metadata record_timer_meta = meta;
  record_timer_meta.op_list.clear();
  ryzenai::record_timer timer_op();
  json dd_ts;

  // if timer_id != 0, create new dd_json, else append to the existing file
  if (profile_ids::timer_id != 0) {
    std::ifstream ifs(dd_timestamp_fname);
    dd_ts = json::parse(ifs);
  } else {
    dd_ts["events"] = json::array();
  }

  // insert subgraph timer start
  if (profile_level >= 1) {
    insert_timer_op_in_meta(record_timer_meta,
                            "subgraph_" + meta.json_path + "__start", 0x0,
                            profile_ids::timer_id);
    dd_ts["events"].push_back({{"id", profile_ids::timer_id},
                               {"name", meta.json_path},
                               {"op_type", "subgraph"},
                               {"start", true},
                               {"type", "subgraph"}});

    profile_ids::timer_id++;
  }

  // iterate over pdi partitions and insert profile points
  for (size_t part = 0; part < meta.partitions.size(); part++) {
    auto pdi_parent_timer_id = profile_ids::timer_id;
    if (profile_level >= 2) {

      // insert pdi_subgraph_start
      insert_timer_op_in_meta(record_timer_meta,
                              "pdi_partition_" + std::to_string(part), part,
                              profile_ids::timer_id);
      dd_ts["events"].push_back(
          {{"id", profile_ids::timer_id},
           {"name", "pdi_partition_" + std::to_string(part)},
           {"op_type", "pdi_partition"},
           {"start", true},
           {"parent", meta.json_path},
           {"type", "pdi_partition"}});
      profile_ids::timer_id++;
    }

    for (size_t i = meta.partitions.at(part).op_range.first;
         i < meta.partitions.at(part).op_range.second; ++i) {
      const auto &op = meta.op_list.at(i);
      json op_prop = get_op_prop(meta, op);
      // Add start timer
      if (profile_level >= 3) {
        insert_timer_op_in_meta(record_timer_meta, op.name + "__start",
                                op.pdi_id, profile_ids::timer_id);
        insert_timer_info_in_dd_json(dd_ts, op_prop, op, profile_ids::timer_id,
                                     true, pdi_parent_timer_id);
        profile_ids::timer_id++;
      }
      record_timer_meta.op_list.emplace_back(op);
      // Add end timer
      if (profile_level >= 3) {
        insert_timer_op_in_meta(record_timer_meta, op.name + "__end", op.pdi_id,
                                profile_ids::timer_id);
        insert_timer_info_in_dd_json(dd_ts, op_prop, op, profile_ids::timer_id,
                                     false, pdi_parent_timer_id);
        profile_ids::timer_id++;
      }
    }

    if (profile_level >= 2) {
      // insert pdi_subgraph_end
      insert_timer_op_in_meta(record_timer_meta,
                              "pdi_partition_" + std::to_string(part), part,
                              profile_ids::timer_id);
      dd_ts["events"].push_back(
          {{"id", profile_ids::timer_id},
           {"name", "pdi_partition_" + std::to_string(part)},
           {"op_type", "pdi_partition"},
           {"start", false},
           {"parent", meta.json_path},
           {"type", "pdi_partition"}});
      profile_ids::timer_id++;
    }
  }

  if (profile_level >= 1) {
    // insert subgraph timer end
    insert_timer_op_in_meta(record_timer_meta,
                            "subgraph_" + meta.json_path + "__end", 0x0,
                            profile_ids::timer_id);
    dd_ts["events"].push_back({{"id", profile_ids::timer_id},
                               {"name", meta.json_path},
                               {"op_type", "subgraph"},
                               {"start", false},
                               {"type", "subgraph"}});

    profile_ids::timer_id++;
  }

  const std::string ai_analyzer_metafile = "dd_timestamp_info.json";
  std::ofstream jsonf(ai_analyzer_metafile);
  jsonf << std::setw(4) << dd_ts << std::endl;

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Inserted {} timers, ai_analyzer_meta_file: {}",
                            profile_ids::timer_id, ai_analyzer_metafile));
  RYZENAI_LOG_TRACE("Insert Record Timer nodes: Done");

  return record_timer_meta;
}
} // namespace OpsFusion
