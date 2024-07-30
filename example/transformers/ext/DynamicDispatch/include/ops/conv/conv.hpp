#pragma once

#include "ops/op_interface.hpp"
#include "ops/ops_common.hpp"
#include <utils/txn_container.hpp>

namespace ryzenai {
template <typename InT, typename WtT, typename OutT>
class conv : public OpInterface {
  using WtsListType = std::vector<std::vector<std::vector<std::vector<WtT>>>>;

private:
  std::map<std::string, std::string> xclbin_a_header;
  std::map<std::string, std::string> xclbin_b_header;
  std::map<std::string, std::string> xclbin_acc_header;
  std::map<std::string, std::string> txnbin_a_header;
  std::map<std::string, std::string> txnbin_b_header;
  std::map<std::string, std::string> txnbin_acc_header;
  std::map<std::string, std::vector<conv_shapes>> default_shapes_;

  std::map<std::string, std::any> attr_;
  int groupId_;
  uint8_t foldWts_;
  int zp_;
  std::vector<uint8_t> lp;
  std::string convData_;

  /* Input dimension of base conv being offloaded to AIE */
  int64_t kernelInputShape_[4];
  /* Weight dimension of base conv being offloaded to AIE */
  int64_t kernelWeightShape_[4];
  /* Output dimension of base conv being offloaded to AIE */
  int64_t kernelOutputShape_[4];

  /* Sandip TBD : Max Kernel parameters should be defined here and should be
   * checked in code */

  /* actual input matrix */
  int64_t inputShape_[3];
  /* actual output matrix */
  int64_t outputShape_[3];
  /* actual weight matrix inserted */
  int64_t weightShape_[4];
  /* Group ID is used to check whether the operator is DWC or not. greater than
   * 1 signifies it is DWC operator */

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
  uint64_t conv_id_;
  static uint64_t conv_count;
  /* debug flag */
  bool debug_ = false;

  /*xclbin and mc_code selection variables*/
  std::string ifmDtype_;
  std::string weightDtype_;
  std::string ofmDtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;

  // TODO: remove this after rt_constant padding is applied to all models
  bool useTxnBinWithZp_;
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;

  void setup_instr_registry();
  std::string get_instr_key(std::string prefix, int64_t zp, int64_t F,
                            int64_t K, int64_t N);
  std::vector<int32_t> qdq_header(int64_t *qdq, int32_t qdqParams[],
                                  int32_t ofm_height, int32_t ofm_width,
                                  int32_t ofm_depth_start,
                                  int32_t ofm_depth_end);
  /* qdq header for convA16W16 */
  std::vector<int32_t> qdq_header(int64_t *qdq, int32_t ofm_height,
                                  int32_t ofm_width, int32_t ofm_depth_start,
                                  int32_t ofm_depth_end);
  int64_t
  ConcatenateWeightParams(void *dest, const std::vector<WtsListType> &wts_list,
                          const std::vector<std::vector<int32_t>> &qdq_list,
                          /* uint8_t *lp, */ int ifm_depth, int cstride,
                          int ksize_x, int ksize_y);
  /* Concat weight params for convA16W16 */
  int64_t
  ConcatenateWeightParams(void *dest, const std::vector<WtsListType> &wts_list,
                          const std::vector<uint8_t> &conv_lp,
                          const std::vector<std::vector<int32_t>> &qdq_list,
                          /* uint8_t *lp, */ int ifm_depth, int cstride,
                          int ksize_x, int ksize_y);
  WtsListType TransformWts(const WtsListType &wts, uint8_t ksize_x,
                           uint8_t ksize_y, uint8_t wts_zp);
  WtsListType TransformWtsWithZp(const WtsListType &wtsOld);
  int64_t ConcatenateWeightParams_dwc(
      void *dest, const std::vector<WtsListType> &wts_list,
      const std::vector<std::vector<int32_t>> &qdq_list, int ifm_depth,
      int cstride, int ksize_x, int ksize_y);
  void WriteToFile(void *src, uint64_t length);
  void initialize_const_params_conv(void *dest,
                                    const std::vector<Tensor> &const_params);
  void initialize_const_params_dwc(void *dest,
                                   const std::vector<Tensor> &const_params);
  void dumpBinary(void *src, size_t length, std::string &filePath);
  std::vector<uint8_t> get_transaction_bin();

public:
  conv(const std::string &a_dtype, const std::string &b_dtype,
       const std::string &c_dtype, bool load_xrt,
       const std::map<std::string, std::any> &attr = {});
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
  const std::map<std::string, std::any> &get_attr() const override {
    return attr_;
  }
  void set_params(const std::string &modelName, bool useTxnBinWithZp);
  Transaction txn_handler;
};

} // namespace ryzenai
