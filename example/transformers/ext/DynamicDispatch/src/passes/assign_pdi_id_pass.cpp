#include <iostream>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

void assign_pdi_id_pass(const OpPDIMap &op_pdi_map, Metadata &meta) {

  std::set<std::uint8_t> unique_pdi_ids;

  for (size_t i = 0; i < meta.op_list.size(); ++i) {
    auto &op = meta.op_list.at(i);
    std::uint8_t pdi_id = OpsFusion::get_pdi_id(op_pdi_map, op.type);
    op.pdi_id = pdi_id;
    unique_pdi_ids.insert(pdi_id);
  }

  constexpr std::uint8_t DEFAULT_PDI_ID = 0;

  size_t num_unique_pdi_ids = unique_pdi_ids.size();

  // only have control ops
  // e.g. want to profile sequence of PM loads
  bool use_default_pdi_id =
      (num_unique_pdi_ids == 1) &&
      (unique_pdi_ids.end() != unique_pdi_ids.find(OpsFusion::CONTROL_PDI_ID));

  if (use_default_pdi_id) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format("Using Default PDI ID: {}",
                                            (std::uint32_t)DEFAULT_PDI_ID));
    for (size_t i = 0; i < meta.op_list.size(); ++i) {
      auto &op = meta.op_list.at(i);
      op.pdi_id = DEFAULT_PDI_ID;
    }
  }

  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Assign PDI IDs: DONE"));
}

} // namespace OpsFusion
