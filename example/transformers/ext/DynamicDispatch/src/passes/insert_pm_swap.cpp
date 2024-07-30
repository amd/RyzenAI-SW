#include <iostream>

#include <op_fuser/fuse_types.hpp>
#include <ops/op_builder.hpp>
#include <ops/pm_load/pm_load.hpp>
#include <utils/meta_utils.hpp>

#include "passes.hpp"

namespace OpsFusion {

Metadata insert_pm_swap_nodes(const Metadata &meta) {
  Metadata pm_swap_meta = meta;
  // clear op_list and rebuild the list by iterating through meta.
  pm_swap_meta.op_list.clear();
  std::string curr_pm_id = "";
  constexpr bool load_xrt = false;
  ryzenai::pm_load pm_op(load_xrt);
  for (size_t i = 0; i < meta.op_list.size(); ++i) {
    const auto &op = meta.op_list.at(i);
    const auto &args = op.args;
    auto &op_type = op.type;
    auto &op_dtype = meta.tensor_map.at(args[0]).dtype;
    auto &xclbin_mdata = pm_op.get_op_xclbin_meta(op_type, op_dtype);
    if (xclbin_mdata.pm_elf_fname != curr_pm_id) {
      RYZENAI_LOG_TRACE(
          OpsFusion::dod_format("OP: {}, PM ID change from {} to {}", op.type,
                                curr_pm_id, xclbin_mdata.pm_elf_fname));
      curr_pm_id = xclbin_mdata.pm_elf_fname;

      std::map<std::string, std::any> attr;
      attr["op_type"] = op_type;
      attr["op_dtype"] = op_dtype;
      Metadata::OpInfo pm_op_info = {
          "pm_load_" + op.name, "PM_LOAD", {}, attr, op.pdi_id};
      pm_swap_meta.op_list.emplace_back(pm_op_info);
    }
    RYZENAI_LOG_INFO(
        OpsFusion::dod_format("OP: {}, PM ID: {}", op.type, curr_pm_id));
    pm_swap_meta.op_list.emplace_back(op);
  }

  std::cout << "Meta after pass" << std::endl;
  for (auto &op_info : pm_swap_meta.op_list) {
    std::cout << op_info.name << std::endl;
  }

  return pm_swap_meta;
}
} // namespace OpsFusion
