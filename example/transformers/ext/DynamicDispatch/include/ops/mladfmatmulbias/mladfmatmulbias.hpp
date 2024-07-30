#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {

template <typename InT, typename WtT, typename AccT, typename OutT = AccT>
class mladfmatmulbias : public OpInterface {
private:
  // additional member variables grows from matmulbias
  /* use AVX or not */
  bool use_avx;
  /* bytes required for params */
  int params_bytes;
  /*group size selected for this instantiation */
  int grp_size_;
  /* singed or unsigned */
  int sign;
  /* Temporary CPU buffer to hold accumulation */
  std::vector<AccT> c_acc_vec_;

  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<mladf_matrix_shapes>> default_shapes_;

  std::string DPU_DIR;
  /* M x K dimension of base matmul being offloaded to AIE */
  int64_t kernel_x_shape_[2];
  /* K x N dimension of base matmul being offloaded to AIE */
  int64_t kernel_y_shape_[2];
  /* M x N dimension of base matmul being offloaded to AIE */
  int64_t kernel_z_shape_[2];
  /*Kernel shape selected in runtime*/
  int64_t kernel_x_rows;
  /* Max Kernel M size supported for a given model*/
  int KERNEL_M_MAX;
  /* actual M x K of matrix A */
  int64_t a_shape_[2];
  /* actual M x N of matrix C */
  int64_t c_shape_[2];
  /* actual K x N of matrix W */
  int64_t w_shape_[2];
  /* padded shape of weight matrix */
  int64_t w_padded_shape_[2];
  // static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* vector of XRT BOs for tiled and reformtted weight matrix */
  std::vector<xrt::bo> weights_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_token_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_token_;
  // xrt::bo c_bo_token_;
  /* size for activation dtype */
  int a_dtype_size_;
  /* size for weights dtype*/
  int b_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;

  /* variables to store profile data */
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
  uint64_t mladfmatmulbias_id_;
  static uint64_t mladfmatmulbias_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  bool initialized_;
  void setup_instr_registry();
  std::string
  get_instr_key(std::string prefix, int m, int k, int n,
                int grp_size = 0 /* additional arg for group size*/);

public:
  mladfmatmulbias(const std::string &a_dtype, const std::string &b_dtype,
                  const std::string &c_dtype, bool load_xrt);
  void
  initialize_const_params(void *dest, const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  void execute(std::vector<xrt::bo> &input,
               std::vector<xrt::bo> &output) override;
  std::vector<xrt::bo> get_inputs(int M);
  std::vector<xrt::bo> get_outputs(int M);
  std::vector<xrt::bo> get_const();
  void set_shape(std::vector<int> a_shape);

  void debug(bool enable);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;
  void set_kernel_shapes_kn_mladf();
  // void initialize_weights_int4_mladf(const std::vector<Tensor>
  // &const_params);
  void set_kernel_shapes_m_mladf(int64_t input_m);
  void run_aie(InT *a, xrt::bo &w_bo, int64_t *input_shape);
};

} // namespace ryzenai
