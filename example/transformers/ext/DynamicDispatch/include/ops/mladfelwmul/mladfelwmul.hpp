#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {

/*
 * ml_adf_elw_mul is an experimental class to offload uint16_t * uint16_t matrix
 * multiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */
template <typename InT, typename WtT, typename OutT>
class ml_adf_elw_mul : public OpInterface {
private:
  std::string DPU_DIR;
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_c_header;
  std::map<std::string, std::vector<std::vector<size_t>>> default_shapes_;
  std::map<std::string, std::vector<std::vector<size_t>>> raw_shapes_;
  /* base elwmul being offloaded to AIE */
  // store padded tensor shape
  std::vector<size_t> kernel_x_shape_;
  std::vector<size_t> kernel_y_shape_;
  std::vector<size_t> kernel_z_shape_;
  /*Kernel shape selected in runtime*/
  /* Max Kernel M size supported for a given model*/
  int KERNEL_M_MAX;
  // store original tensor shape
  std::vector<size_t> a_shape_;
  std::vector<size_t> c_shape_;
  std::vector<size_t> w_shape_;

  std::vector<size_t> w_padded_shape_;

  /* xrt context handle */
  // xrt_context *xrt_ctx_;
  // static instruction_registry instr_reg_add_;
  static std::once_flag instr_reg_flag_;
  xrt::bo params_bo_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo b_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* size for input activation dtype*/
  int a_dtype_size_;
  /* size for weights dtype*/
  int b_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  /* variables to store profile data */
  int64_t params_copy_time_;
  int64_t params_sync_time_;
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t b_copy_time_;
  int64_t b_format_time_;
  int64_t b_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t ml_adf_elw_mul_id_;
  static uint64_t ml_adf_elw_mul_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;

  bool is_1x4096x4096 = false;
  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix,
                            const std::vector<size_t> &dimensions);
  std::vector<size_t> map_padded_shape(const std::vector<size_t> &dims);

public:
  ml_adf_elw_mul(const std::string &a_dtype, const std::string &b_dtype,
                 const std::string &c_dtype, bool load_xrt);

  void
  initialize_const_params(void *dest, const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  void debug(bool enable);
  void set_params(const std::string &model_name,
                  std::vector<size_t> input_shape);

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
