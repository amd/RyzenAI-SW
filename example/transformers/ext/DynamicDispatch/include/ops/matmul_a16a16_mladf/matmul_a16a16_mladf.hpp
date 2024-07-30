
#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>

namespace ryzenai {

/*
 * ipu_wrapper is an experimental class to offload int8_t * int8_t matrix
 * multiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */
template <typename InT, typename WtT, typename OutT>
class matmul_a16a16_mladf : public OpInterface {
private:
  std::string DPU_DIR;
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<matrix_shapes>> default_shapes_;
  std::map<std::string, std::vector<matrix_shapes>> raw_shapes_;

  /* M x K dimension of base matmul being offloaded to AIE */
  int64_t kernel_x_shape_[2];
  /* K x N dimension of base matmul being offloaded to AIE */
  int64_t kernel_y_shape_[2];
  /* M x N dimension of base matmul being offloaded to AIE */
  int64_t kernel_z_shape_[2];
  /*Kernel shape selected in runtime*/
  int64_t kernel_x_rows;
  /* Max Kernel M size supported for a given model*/
  int KERNEL_M_MAX = 4096;
  /* Max Kernel K size supported for a given model*/
  int KERNEL_K_MAX = 4096;
  /* Max Kernel N size supported for a given model*/
  int KERNEL_N_MAX = 4096;
  /* actual param shape of matmul */
  int64_t param_shape_[1];
  /* actual M x K of matrix A */
  int64_t a_shape_[2];
  /* actual M x N of matrix A */
  int64_t c_shape_[2];
  /* actual K x N of matrix A */
  int64_t w_shape_[2];
  /* padded shape of weight matrix */
  int64_t w_padded_shape_[2];
  /* xrt context handle */
  //   xrt_context *xrt_ctx_;
  //   static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled param */
  xrt::bo param_bo_;
  /* XRT BO for tiled activation matrix */
  xrt::bo a_bo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo b_bo_;
  /* XRT BO for tiled output matrix */
  xrt::bo c_bo_;
  /* size for param dtype*/
  int param_dtype_size_;
  /* size for input activation dtype*/
  int a_dtype_size_;
  /* size for weight activation dtype*/
  int b_dtype_size_;
  /* size for output activation dtype*/
  int c_dtype_size_;
  /* variables to store profile data */
  int64_t param_copy_time_;
  int64_t param_sync_time_;
  int64_t a_copy_time_;
  int64_t a_sync_time_;
  int64_t b_copy_time_;
  int64_t b_sync_time_;
  int64_t c_copy_time_;
  int64_t c_sync_time_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t matmul_a16a16_mladf_id_;
  static uint64_t matmul_a16a16_mladf_count;
  /* debug flag */
  bool debug_ = false;
  /*xclbin and mc_code selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;

  /* Utility function to set the kernel shape based on the weights dimensions
   * Pick kernel shape using weight matrix size
   * Select OPT shapes when a_type is int8
   * Select Llamav2 shapes when a_type is int16
   * Need to fix this to pick shapes independent of the datatype*/
  void set_kernel_shapes();

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(std::string prefix, int m, int k, int n);
  std::tuple<int, int> map_padded_shape(int M, int N);

public:
  /*
   * matmul_a16a16_mladf class constructor
   *
   * @param kernel_x_shape tuple containing of M x K dimension base matmul
   * supported on IPU
   * @param kernel_y_shape tuple containing of K x N dimension base matmul
   * supported on IPU
   *
   * NOTE: If the input shape has a smaller M dimension than the kernel
   * shape initialized here, the execute function can transparently
   * call a smaller GeMM to reduce padding overhead. The kernel
   * shape passed here should have largest supported M dimension.
   *
   */
  matmul_a16a16_mladf(const std::string &a_dtype, const std::string &b_dtype,
                      const std::string &c_dtype, bool load_xrt);
  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  void debug(bool enable);
  void set_params(const std::string &model_name,
                  std::vector<size_t> input_shape,
                  std::vector<size_t> weight_shape);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;
  void initialize_wts(void *dest, const std::vector<Tensor> &const_params,
                      const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(void *dest, const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
};

} // namespace ryzenai
