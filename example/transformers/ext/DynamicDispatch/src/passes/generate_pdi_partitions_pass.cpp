#include <iostream>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

void generate_pdi_partitions_pass(Metadata &meta, bool eager_mode) {

  std::vector<Partition> partitions;

  if (0 == meta.op_list.size()) {
    meta.partitions = partitions;
    return;
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Generate PDI Partitions Pass, eager_mode {}",
                            static_cast<std::uint32_t>(eager_mode)));

  std::set<std::uint8_t> unique_pdi_ids;

  Partition partition;

  size_t start_op_id = 0;
  auto curr_pdi_id = meta.op_list.at(0).pdi_id;

  partition.pdi_id = curr_pdi_id;
  unique_pdi_ids.insert(curr_pdi_id);

  RYZENAI_LOG_TRACE(OpsFusion::dod_format("\top_name {} pdi_id {}",
                                          meta.op_list[0].name,
                                          (std::uint32_t)partition.pdi_id));

  for (size_t op_id = 1; op_id < meta.op_list.size(); op_id++) {
    curr_pdi_id = meta.op_list.at(op_id).pdi_id;

    RYZENAI_LOG_TRACE(OpsFusion::dod_format("\top_name {} pdi_id {}",
                                            meta.op_list[op_id].name,
                                            (std::uint32_t)curr_pdi_id));

    if ((partition.pdi_id != curr_pdi_id) || eager_mode) {
      partition.op_range = std::make_pair(start_op_id, op_id);
      partitions.push_back(partition);

      start_op_id = op_id;
      partition.pdi_id = curr_pdi_id;
      unique_pdi_ids.insert(curr_pdi_id);
    }
  }

  partition.op_range = std::make_pair(start_op_id, meta.op_list.size());
  partitions.push_back(partition);

  if (unique_pdi_ids.end() != unique_pdi_ids.find(OpsFusion::CONTROL_PDI_ID)) {
    DOD_THROW(OpsFusion::dod_format(
        "Found CONTROL_PDI_ID - this does not belong to any kernel!"));
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Num PDI Partitions {} : ", partitions.size()));
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Generate PDI partitions: DONE"));

  meta.partitions = std::move(partitions);
}

} // namespace OpsFusion
