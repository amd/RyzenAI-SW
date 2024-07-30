#pragma once

#include <ops/op_interface.hpp>

namespace ryzenai {

template <typename InT, typename OutT> class square : public OpInterface {
private:
  size_t num_elems = 32;

  xrt::bo input_bo_;
  xrt::bo output_bo_;
  xrt::bo instr_bo_;

  void setup_instr_bo();

public:
  square(bool load_xrt = false);
  void
  initialize_const_params(void *dest, const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;
};
} // namespace ryzenai
