#pragma once

#include "ops/op_interface.hpp"
#include "ops/ops_common.hpp"
#include <utils/txn_container.hpp>

namespace ryzenai {
template <typename InT, typename WtT, typename OutT>
class lstm : public OpInterface {
private:
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<matrix_shapes>> default_shapes_;

  std::string DPU_DIR;
  /* Input dimension of base lstm being offloaded to AIE */
  int64_t kernelInputShape_[3];
  /* Weight dimension of base lstm being offloaded to AIE */
  int64_t kernelWeightShape_[3];
  /* Output dimension of base lstm being offloaded to AIE */
  int64_t kernelOutputShape_[3];

  /* Sandip TBD : Max Kernel parameters should be defined here and should be
   * checked in code */

  /* actual input matrix */
  int64_t inputShape_[3];
  /* actual output matrix */
  int64_t outputShape_[3];
  /* actual weight matrix inserted */
  int64_t weightShape_[3];

  int modelNum_;

  //  static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo ifmBo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo constBo_;
  /* XRT BO for tiled output matrix */
  xrt::bo ofmBo_;
  /* size for input activation dtype*/

  int ifmDtypeSize_;
  /* size for weights dtype*/
  int weightDtypeSize_;
  /* size for output activation dtype*/
  int ofmDtypeSize_;
  /* variables to store profile data */
  int64_t ifmCopyTime_;
  int64_t ifmSyncTime_;
  int64_t weightCopyTime_;
  int64_t weightFormatTime_;
  int64_t weightSyncTime_;
  int64_t ofmCopyTime_;
  int64_t ofmSyncTime_;
  int64_t run_aie_time_;
  int64_t cpu_acc_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;
  static std::once_flag logger_flag_;
  uint64_t lstm_id_;
  static uint64_t lstm_count;
  /* debug flag */
  bool debug_ = false;

  /*xclbin and mc_code selection variables*/
  std::string ifmDtype_;
  std::string weightDtype_;
  std::string ofmDtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;

  void setup_instr_registry();
  std::string get_instr_key(std::string prefix, int Mi0, int Mi1, int Mi2);

public:
  lstm(const std::string &a_dtype, const std::string &b_dtype,
       const std::string &c_dtype, bool load_xrt);
  void
  initialize_const_params(void *dest, const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  void debug(bool enable);
  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) override;
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;
  void set_params(const int modelNum, std::vector<size_t> input_shape,
                  std::vector<size_t> weight_shape,
                  std::vector<size_t> output_shape);
};

} // namespace ryzenai
