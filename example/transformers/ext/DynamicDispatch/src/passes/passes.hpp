
#pragma once

#include <op_fuser/fuse_types.hpp>

namespace OpsFusion {

void assign_pdi_id_pass(const OpPDIMap &op_pdi_map, Metadata &meta);
Metadata insert_pm_swap_nodes(const Metadata &meta);
Metadata insert_record_timer_nodes(const Metadata &meta,
                                   uint32_t profile_level);
void generate_pdi_partitions_pass(Metadata &meta, bool eager_mode);
void analyze_buffer_reqs(Metadata &meta);
void optimize_scratch_buffer(Metadata &meta);
bool split_max_partition_pass(
    Metadata &meta, const std::vector<std::vector<uint8_t>> fused_instr_vec,
    size_t limit);

} // namespace OpsFusion
