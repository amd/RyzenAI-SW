#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {

template <typename InT, typename OutT> class slice : public OpInterface {
private:
  std::string design_param_;
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_c_header;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> default_shapes_;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> raw_shapes_;
  /* M x K dimension of base slice being offloaded to AIE */
  int64_t kernel_x_shape_[2];
  /*Kernel shape selected in runtime*/
  /* actual M x K of matrix A */
  int64_t a_shape_[2];
  /* actual M x K of matrix C */
  int64_t c_shape_[2];
  // static instruction_registry instr_reg_add_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* size for input activation dtype*/
  int a_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  /* variables to store profile data */
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t slice_id_;
  static uint64_t slice_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;
  size_t slice_idx_;

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, int h, int w, int c);
  std::tuple<int, int, int> map_padded_shape(int H, int W, int C);

public:
  slice(const std::string &a_dtype, const std::string &c_dtype, bool load_xrt,
        const std::map<std::string, std::any> &attr);

  void initialize_const_params(void *dest,
                               const std::vector<Tensor> &const_params);

  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  void debug(bool enable);

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
