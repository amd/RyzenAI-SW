#pragma once

#include <ops/op_interface.hpp>

namespace ryzenai {

struct overlay_pm_meta {
  struct pkt_switch_meta {
    uint8_t pkt_id;
    uint8_t col;
    uint8_t dma_ch_num;
  };
  uint8_t num_cols;
  std::vector<pkt_switch_meta> pkt_sw_meta_;
};

struct op_xclbin_meta {
  std::string xclbin_name;
  std::string pm_elf_fname;
};

class pm_load : public OpInterface {
private:
  static const std::map<std::string, overlay_pm_meta> overlay_meta_;
  static const std::map<std::string, op_xclbin_meta> op_xclbin_meta_;
  const std::vector<uint8_t>
  get_pm_bin(const std::map<std::string, std::any> &attr);

public:
  pm_load(bool load_xrt = false);
  void execute(std::string op_name, std::string dtype);
  const std::vector<uint8_t>
  get_transaction_bin(std::vector<Tensor> &input, std::vector<Tensor> &output,
                      const std::map<std::string, std::any> &attr) override;
  const overlay_pm_meta &get_overlay_meta(const std::string &xclbin_name) const;
  const op_xclbin_meta &get_op_xclbin_meta(const std::string &op_name,
                                           const std::string &dtype) const;
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr) override;
  const std::vector<uint8_t>
  get_super_kernel_params(std::vector<Tensor> &input,
                          std::vector<Tensor> &output,
                          const std::map<std::string, std::any> &attr) override;
};

} // namespace ryzenai
