#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>
#include <tuple>

namespace ryzenai {

namespace xcom {

// TO DO: add zero point/scale here as necessary
struct conv_qdq_info_t {
  std::int32_t act_zero_point;
  float act_scale;
  std::int32_t weight_zero_point;
  float weight_scale;
  std::int32_t bias_zero_point;
  float bias_scale;
  std::int32_t out_zero_point;
  float out_scale;
  std::int16_t H;
  std::int16_t W;
  std::int16_t C_in;
  std::int16_t C_out;
  std::int8_t kernel_size_x;
  std::int8_t kernel_size_y;
  std::int8_t stride_x;
  std::int8_t stride_y;
  std::int32_t unused_[5];
};

static_assert(sizeof(conv_qdq_info_t) == 64);

// initialize_const_params, get_buffer_reqs seem to rely on shape
// info passed through std::vector<Tensor> object
// These will be parameters which run-time can not configure
struct conv_shape_t {
  int64_t H;
  int64_t W;
  int64_t C_in;
  int64_t C_out;

  friend bool operator<(const conv_shape_t &lhs, const conv_shape_t &rhs) {
    return std::tie(lhs.H, lhs.W, lhs.C_in, lhs.C_out) <
           std::tie(rhs.H, rhs.W, rhs.C_in, rhs.C_out);
  }
};

// Minimal info to determine if conv layer is supported
struct conv_static_params_t {
  conv_shape_t shape_info;
  int64_t kernel_x;
  int64_t kernel_y;
  int64_t stride_x;
  int64_t stride_y;
  int64_t pad_left;
  int64_t pad_right;
  int64_t pad_top;
  int64_t pad_bottom;
  bool bias;
};

struct conv_params_t {
  conv_static_params_t static_params;
  // whether output of conv layer goes through relu
  bool relu;
  // QDQ params
  int32_t zero_point;
  int32_t scale;
};

/*
 * conv2d is an experimental class to offload int8_t * int8_t, int16_t * int8_t
 * 2D convolution to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU.
 */
template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
class conv2d : public OpInterface {
private:
  /* XCOMPILER layer parameters that kernel consumes, size in bytes*/
  static constexpr int64_t LAYER_PARAM_SIZE = 64;
  /* Kernel shape for 2D convolution - e.g. 1x1 or 3x3 typically*/
  int64_t kernel_x_dim_;
  int64_t kernel_y_dim_;
  /* stride along input*/
  int64_t stride_x_;
  int64_t stride_y_;
  /*This actual number of output channels*/
  int64_t num_output_channels_;
  /* in NCHW format - without padding*/
  int64_t activation_shape_[4];
  /*pad for left, right (X dim) and top, down (Y dim)*/
  int64_t pads_[4];
  /* in NCHW format - input shape with padding */
  int64_t input_padded_shape_[4];
  /* num_output_channelsxkernel_x_dimxkernel_y_dimxnum_input_channels*/
  int64_t weights_shape_[4];
  /*will be NUM_OUTPUT_PADDEDxKERNEL_X_PADDEDxKERNEL_Y_PADDEDxNUM_INPUT_PADDED*/
  int64_t weights_padded_shape_[4];
  /*1D dim bias padded_shape*/
  int64_t bias_padded_shape[1];
  /*in NCHW format - without padding*/
  int64_t output_shape_[4];
  /*in NCHW format - includes padding*/
  int64_t output_padded_shape_[4];
  /* Add bias to output */
  bool bias_en_;
  /* Have relu as activation function */
  bool relu_en_;

  // QDQ related params
  int32_t zeropoint;
  int32_t scale;

  using op_type = std::string;
  using raw_to_padded_shape_map = std::map<conv_shape_t, conv_shape_t>;
  std::map<std::string, raw_to_padded_shape_map> supported_shapes_;

  // cache state after call to set_params
  conv_shape_t padded_shape_info_;

  static std::once_flag instr_reg_flag_;
  /* XRT BO for padded activation */
  xrt::bo input_bo_;
  /* XRT BO for layer params and tiled weights/bias */
  xrt::bo param_bo_;
  /* XRT BO for padded output */
  xrt::bo output_bo_;
  /* XRT BO for intermediate output within a op */
  xrt::bo scratch_bo_;
  /* size for input activation dtype*/
  int64_t a_dtype_size_;
  /* size for weights dtype*/
  int64_t b_dtype_size_;
  /* size for bias dtype*/
  int64_t bias_dtype_size_;
  /* size for output activation dtype*/
  int64_t c_dtype_size_;
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
  uint64_t id_;
  static uint64_t count_;

  /*xclbin and transaction bin selection variables*/
  std::string a_dtype_;
  std::string b_dtype_;
  std::string bias_dtype_;
  std::string c_dtype_;
  std::string txn_fname_prefix_;

  std::string XCLBIN_FNAME_;

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();

  std::string get_instr_key(const std::string &prefix,
                            const conv_shape_t &shape_info) const;
  conv_shape_t get_padded_shape(const conv_shape_t &shape_info) const;
  const std::vector<std::uint8_t>
  get_kernel_params(const std::string &param_key) const;

public:
  conv2d(const std::string &a_dtype, const std::string &b_dtype,
         const std::string &bias_dtype, const std::string &c_dtype,
         bool load_xrt);
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;

  void initialize_const_params(
      void *dest, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;

  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override;
  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);

  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;
  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;

  void set_params(const conv_params_t &params);
};

} // namespace xcom

} // namespace ryzenai
