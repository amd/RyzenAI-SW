#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {

template <typename InT, typename WtT, typename OutT>
class bmm : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<matrix_shapes>> default_shapes_;
  std::map<std::string, std::vector<matrix_shapes>> raw_shapes_;

  /* M x K dimension of base bmm being offloaded to AIE */
  int64_t kernel_x_shape_[2];
  /* K x N dimension of base bmm being offloaded to AIE */
  int64_t kernel_y_shape_[2];
  /* M x N dimension of base bmm being offloaded to AIE */
  int64_t kernel_z_shape_[2];
  /*Kernel shape selected in runtime*/
  int64_t kernel_x_rows;
  /* Max Kernel M size supported for a given model*/
  int KERNEL_M_MAX;
  /* actual M x K of matrix A */
  int64_t a_shape_[2];
  /* actual M x N of matrix A */
  int64_t c_shape_[2];
  /* actual K x N of matrix A */
  int64_t w_shape_[2];
  /* padded shape of weight matrix */
  int64_t w_padded_shape_[2];
  /* xrt context handle */
  // xrt_context *xrt_ctx_;
  // static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT Instruction BO */
  xrt::bo instr_bo;
  int instr_bo_words;
  xrt::kernel kernel_;
  xrt::run run;
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
  uint64_t bmm_id_;
  static uint64_t bmm_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;

  void set_kernel_shapes();
  void setup_instr_registry();
  std::string get_instr_key(std::string prefix, size_t m, size_t k, size_t n);
  std::tuple<size_t, size_t> map_padded_shape(size_t M, size_t N);

public:
  bmm(const std::string &a_dtype, const std::string &b_dtype,
      const std::string &c_dtype, bool load_xrt);
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void execute(std::vector<xrt::bo> &input,
               std::vector<xrt::bo> &output) override;
  void debug(bool enable);
  std::vector<xrt::bo> allocate_inputs();
  std::vector<xrt::bo> allocate_outputs();

  void set_params(const std::string &model_name,
                  std::vector<size_t> input_shape);

  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;

  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;
};

} // namespace ryzenai
