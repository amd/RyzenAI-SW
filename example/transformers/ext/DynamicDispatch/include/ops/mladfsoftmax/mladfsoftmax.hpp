#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {

/*
 * mladf_softmax is a class to offload matrix
 * Attention masking and Softmax to AIE. this class uses lite runtime stack to
 * interface with XRT
 */
template <typename InT, typename WtT, typename OutT>
class mladf_softmax : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_operand_header;
  // 4096x4096
  std::map<std::string, std::vector<std::vector<size_t>>> default_shapes_;

  std::vector<size_t> kernel_x_shape_;
  std::vector<size_t> kernel_z_shape_;
  /*Kernel shape selected in runtime*/
  size_t operand_size_in_bytes_;
  size_t const_input_size_;
  /* xrt context handle */
  // xrt_context *xrt_ctx_;
  // static instruction_registry instr_reg_add_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for rtp */
  xrt::bo rtp_bo_;
  /* XRT BO for tiled LHS matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled RHS matrix */
  xrt::bo b_bo_;
  /* XRT BO for tiled OUT matrix */
  xrt::bo c_bo_;
  /* size for activation dtype*/
  int operand_dtype_size_;

  /* variables to store profile data */
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t b_copy_time_;
  int64_t b_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t rtp_copy_time_;
  int64_t rtp_sync_time_;
  int64_t run_aie_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t adf_softmax_id_;
  static uint64_t adf_softmax_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string operand_dtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix,
                            const std::vector<size_t> &dimensions);

public:
  mladf_softmax(const std::string &operand_dtype, bool load_xrt);

  void debug(bool enable);

  void set_params(const std::string &model_name, std::vector<size_t> a_shape);
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

  const std::vector<uint8_t>
  get_super_kernel_params(std::vector<Tensor> &input,
                          std::vector<Tensor> &output,
                          const std::map<std::string, std::any> &attr = {});

  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;
};

} // namespace ryzenai
