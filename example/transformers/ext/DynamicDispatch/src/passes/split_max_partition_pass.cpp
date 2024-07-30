#include <iostream>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

bool split_max_partition_pass(
    Metadata &meta, const std::vector<std::vector<uint8_t>> fused_instr_vec,
    size_t limit) {
  size_t max_instr_size = 0;
  size_t max_partition_size = 0;
  size_t max_partition_idx = 0;

  constexpr bool SPLIT = true;
  constexpr bool NOT_SPLIT = false;

  bool need_to_split = false;

  for (size_t partition_idx = 0; partition_idx < meta.partitions.size();
       partition_idx++) {
    const auto &partition = meta.partitions.at(partition_idx);
    size_t curr_instr_size = fused_instr_vec.at(partition_idx).size();
    size_t curr_partition_size =
        partition.op_range.second - partition.op_range.first;

    // choose partition that generated biggest instruction size, and has more
    // than 1 op for now we only partition on op boundaries
    if ((max_instr_size < curr_instr_size) && (limit < curr_instr_size) &&
        (1 < curr_partition_size)) {
      max_instr_size = curr_instr_size;
      max_partition_idx = partition_idx;
      max_partition_size = curr_partition_size;

      need_to_split = true;
    }

    if ((limit < curr_instr_size) && (1 == curr_partition_size)) {
      // TODO: add support if a single op exceeds instr BO size
      return NOT_SPLIT;
    }
  }

  if (!need_to_split) {
    // we are done
    return SPLIT;
  }

  const size_t num_partitions = meta.partitions.size();

  Partition partition_to_split = meta.partitions.at(max_partition_idx);
  Partition left = partition_to_split;
  Partition right = partition_to_split;

  // Note: range is of form [first, last)
  left.op_range.second = left.op_range.first + max_partition_size / 2;
  right.op_range.first = left.op_range.second;

  // shift every element in array up by 1 uptil partition we are splitting
  meta.partitions.emplace_back(Partition{});

  for (size_t partition_idx = num_partitions; partition_idx > max_partition_idx;
       partition_idx--) {
    meta.partitions.at(partition_idx) = meta.partitions.at(partition_idx - 1);
  }

  // lastly create new partition,
  meta.partitions.at(max_partition_idx) = left;
  meta.partitions.at(max_partition_idx + 1) = right;

  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Splitting max partition: DONE"));

  return SPLIT;
}

} // namespace OpsFusion
