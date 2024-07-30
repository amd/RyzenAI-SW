/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#ifndef _WIN32
#include <cmath>
#endif

#include <iomanip>
#include <iterator>
#include <string>

#include <ops/lstm/lstm.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

namespace ryzenai {
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string lstm<InT, WtT, OutT>::get_instr_key(std::string prefix, int m,
                                                int k, int n) {
  return "lstm_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k) +
         "_" + std::to_string(n);
}

/*
 * lstm class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base lstm
 * supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base lstm
 * supported on IPU
 *
 */
template <typename InT, typename WtT, typename OutT>
lstm<InT, WtT, OutT>::lstm(const std::string &ifmDtype,
                           const std::string &weightDtype,
                           const std::string &ofmDtype, bool load_xrt) {
  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_b_header = {{"uint16", "w16"}};
  txnbin_acc_header = {{"uint16", "c16"}};

  modelNum_ = 320;
  DPU_DIR =
      OpInterface::get_dod_base_dir() + "//transaction//" + "stx" + "//lstm//";

  ifmDtype_ = ifmDtype;
  weightDtype_ = weightDtype;
  ofmDtype_ = ofmDtype;
  ifmDtypeSize_ = sizeof(InT);
  weightDtypeSize_ = sizeof(WtT);
  ofmDtypeSize_ = sizeof(OutT);

  lstm_id_ = lstm_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + "\\xclbin\\stx\\4x2_pso2_" +
      std::to_string(modelNum_) + "_model_lstm_a16w16_qdq.xclbin";

  // RYZENAI_LOG_TRACE(OpsFusion::dod_format("xclbin fname : {}",
  // XCLBIN_FNAME));
  txn_fname_prefix_ = "lstm_" + txnbin_a_header.at(ifmDtype_) +
                      txnbin_b_header.at(weightDtype_) +
                      txnbin_acc_header.at(ofmDtype_);

  default_shapes_["lstm_a16w16c16"] = std::vector<matrix_shapes>{};

  /* Shapes for PSO2-320 */
  default_shapes_["lstm_a16w16c16"].emplace_back(80, 1, 64);

  /* Shapes for PSO2-640 */
  default_shapes_["lstm_a16w16c16"].emplace_back(160, 1, 64);

  /* Shapes for PSO2-1280 */
  default_shapes_["lstm_a16w16c16"].emplace_back(320, 1, 64);

  weightShape_[0] = 1;
  weightShape_[1] = 1;
  weightShape_[2] = 672512;

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  ifmCopyTime_ = 0;
  ifmSyncTime_ = 0;
  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;
  ofmCopyTime_ = 0;
  ofmSyncTime_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  /* TODO :: Attribute Parsing required ???*/

  std::call_once(logger_flag_, []() {
    std::string header = "lstm_id (Mi0 Mi1 Mi2 Mo0, Mo1, Mo2) Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[lstm] ID: " + std::to_string(lstm_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
                    ifmDtype_ + ", " + weightDtype_ + ", " + ofmDtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::set_params(const int model_num,
                                      std::vector<size_t> input_shape,
                                      std::vector<size_t> weight_shape,
                                      std::vector<size_t> output_shape) {
  modelNum_ = model_num;
  inputShape_[0] = input_shape.at(0);
  inputShape_[1] = input_shape.at(1);
  inputShape_[2] = input_shape.at(2);

  weightShape_[0] = weight_shape.at(0);
  weightShape_[1] = weight_shape.at(1);
  weightShape_[2] = weight_shape.at(2);

  outputShape_[0] = output_shape.at(0);
  outputShape_[1] = output_shape.at(1);
  outputShape_[2] = output_shape.at(2);

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + "\\xclbin\\stx\\4x2_pso2_" +
      std::to_string(modelNum_) + "_model_lstm_a16w16_qdq.xclbin";

  RYZENAI_LOG_TRACE(OpsFusion::dod_format("xclbin fname : {}", XCLBIN_FNAME));
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every lstm performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU lstm implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("LSTM initialize_const_params(ptr) ...");

  auto testDataFolder = OpInterface::get_dod_base_dir() +
                        "\\tests\\cpp\\unit_tests\\testDataMladf\\lstm_" +
                        std::to_string(modelNum_);

  auto fileName = testDataFolder + "\\" + "wts" + ".bin";

  std::vector<WtT> weights = OpsFusion::read_bin_file<WtT>(
      fileName); //= (WtT*)const_params.at(0).data;

  int weightsSize =
      weightShape_[0] * weightShape_[1] * weightShape_[2] * sizeof(WtT);
  memcpy((void *)dest, (void *)weights.data(), weightsSize);

  RYZENAI_LOG_TRACE("LSTM initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("lstm initialize_const_params ...");

  // weightShape_[0] = 1; //const_params.at(0).shape.at(0);
  // weightShape_[1] = 448; //const_params.at(0).shape.at(1);
  // weightShape_[2] = 512; //const_params.at(0).shape.at(2);

  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;

  /* Create input/output BOs */
  const int CONST_BO_SIZE =
      (weightShape_[0] * weightShape_[1] * weightShape_[2] * weightDtypeSize_);
  const int IFM_BO_SIZE =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  const int OFM_BO_SIZE =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);
  RYZENAI_LOG_TRACE("lstm: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE));
  constBo_ =
      xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  ifmBo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  ofmBo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));

  auto weightCopyStart = GET_ELAPSED_TIME_NS();
  auto weightFormatStart = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = constBo_.map<WtT *>();

  initialize_const_params(b_bo_map, const_params);
  auto weightFormatStop = GET_ELAPSED_TIME_NS();
  weightFormatTime_ += weightFormatStop - weightFormatStart;
  auto weightCopyStop = GET_ELAPSED_TIME_NS();
  auto weightSyncStart = GET_ELAPSED_TIME_NS();
  constBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto weightSyncStop = GET_ELAPSED_TIME_NS();
  weightCopyTime_ = weightCopyStop - weightCopyStart;
  weightSyncTime_ = weightSyncStop - weightSyncStart;
  RYZENAI_LOG_TRACE("lstm initialize_const_params ... DONE");
}
/*
 * perform lstm c = a * w. w is stored in the object with initilize_weights
 * method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of lstm
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                   std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("lstm execute ...");

  ifmCopyTime_ = 0;
  ifmSyncTime_ = 0;
  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;
  ofmCopyTime_ = 0;
  ofmSyncTime_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  inputShape_[0] = input.at(0).shape.at(0);
  inputShape_[1] = input.at(0).shape.at(1);
  inputShape_[2] = input.at(0).shape.at(2);
  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  ifmBo_.write(input.at(0).data);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  ifmCopyTime_ = a_copy_stop - a_copy_start;

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();
  ifmSyncTime_ = a_sync_stop - a_sync_start;

  xrt::bo instr_bo; // = nullptr;
  xrt::bo param_bo;

  auto instr_bo_key = get_instr_key(txn_fname_prefix_, inputShape_[0],
                                    inputShape_[1], inputShape_[2]);
  instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  xrt::run run;
  // launch the lstm kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for lstm that supports transaction binary flow
  run = kernel_(2, instr_bo, instr_bo_words,
                constBo_.address() + DDR_AIE_ADDR_OFFSET,
                ifmBo_.address() + DDR_AIE_ADDR_OFFSET,
                ofmBo_.address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  run.wait2();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += run_aie_stop - run_aie_start;

  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  ofmSyncTime_ += c_sync_stop - c_sync_start;

  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  ofmBo_.read(output.at(0).data);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  ofmCopyTime_ = c_copy_stop - c_copy_start;
  /*
    RYZENAI_LOG_INFO(
        std::to_string(matmul_id_) + " " + std::to_string(a_shape_[0]) + " " +
        std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
        std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1])
    + " " + std::to_string(kernel_y_shape_[1]) + " " + std::to_string(exec_end -
    exec_start) + " " + std::to_string(num_run_aie_) + " " +
    std::to_string(run_aie_time_) + " " + std::to_string(ifmCopyTime_) + " " +
    std::to_string(ifmCopyTime_) + " " + std::to_string(ofmCopyTime_) + " " +
    std::to_string(ofmCopyTime_) + " " + std::to_string((double)run_aie_time_ /
    num_run_aie_) + "\n");
  */
  RYZENAI_LOG_TRACE("lstm execute ... DONE");
}

/*
 * method to set debug flag
 *
 * When the debug flag is set, execute method will write input, weights and
 * output matricies to a filed. the filename will be
 * ryzenai_qlinear2_<execute_num>_<matrix>.txt
 *
 * @param debug bool value to enable disable debug feature. turned off by
 * default
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> lstm<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  inputShape_[0] = input.at(0).shape[0];
  inputShape_[1] = input.at(0).shape[1];
  inputShape_[2] = input.at(0).shape[2];
  std::string txn_key =
      "lstm_" + txn_fname_prefix_ + "_" + std::to_string(inputShape_[0]) + "_" +
      std::to_string(inputShape_[1]) + "_" + std::to_string(inputShape_[2]);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> txnData((std::istreambuf_iterator<char>(txn_stream)),
                               std::istreambuf_iterator<char>());
  return txnData;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> lstm<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {

  inputShape_[0] = input.at(0).shape[0];
  inputShape_[1] = input.at(0).shape[1];
  inputShape_[2] = input.at(0).shape[2];
  outputShape_[0] = input.at(7).shape[0];
  outputShape_[1] = input.at(7).shape[1];
  outputShape_[2] = input.at(7).shape[2];

  size_t const_params_bo_size =
      (weightShape_[0] * weightShape_[1] * weightShape_[2] *
       weightDtypeSize_); // totalWtsSize;
  size_t ifm_bo_size =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  size_t ofm_bo_size =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);
  RYZENAI_LOG_TRACE("lstm: IFM_BO_SIZE:" + std::to_string(ifm_bo_size) +
                    " CONST_BO_SIZE:" + std::to_string(const_params_bo_size) +
                    " OFM_BO_SIZE:" + std::to_string(ofm_bo_size));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 7, 0, ofm_bo_size}};

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("lstm Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag lstm<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t lstm<InT, WtT, OutT>::lstm_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag lstm<InT, WtT, OutT>::instr_reg_flag_;

template class lstm<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
