/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

#include <utils/instruction_registry.hpp>
#include <utils/txn_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/maskedsoftmax/maskedsoftmax.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

namespace ryzenai {

namespace {
std::string getXCLBinName() {
  return OpInterface::get_dod_base_dir() +
         LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;
}

size_t tuple_product(const std::tuple<int, int, int> &tuple) {
  return std::get<0>(tuple) * std::get<1>(tuple) * std::get<2>(tuple);
}

} // namespace

static std::tuple<int, int, int> extract_BMK(const Tensor &input) {
  int B = 0;
  int M = 0;
  int K = 0;
  if (input.shape.size() != 3) {
    throw std::runtime_error(
        "Masked SoftMax expects a rank 3 tensor [Batch,Rows,Cols]");
  }
  B = input.shape.at(0);
  M = input.shape.at(1);
  K = input.shape.at(2);
  return std::make_tuple(B, M, K);
}

template <typename LhsT, typename MaskT, typename OutT>
bool masked_softmax<LhsT, MaskT, OutT>::isSupportedShape(
    const Tensor &operand) {
  const auto &supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  const auto shapeOperand = extract_BMK(operand);
  for (const auto &supported : supported_shapes) {
    if (supported == shapeOperand) {
      return true;
    }
  }
  return false;
}

template <typename LhsT, typename MaskT, typename OutT>
std::once_flag masked_softmax<LhsT, MaskT, OutT>::logger_flag_;

template <typename LhsT, typename MaskT, typename OutT>
uint64_t masked_softmax<LhsT, MaskT, OutT>::masked_softmax_count = 0;

template <typename LhsT, typename MaskT, typename OutT>
std::once_flag masked_softmax<LhsT, MaskT, OutT>::instr_reg_flag_;

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename LhsT, typename MaskT, typename OutT>
std::string masked_softmax<LhsT, MaskT, OutT>::get_instr_key(std::string prefix,
                                                             int batch, int m,
                                                             int k) {
  return "maskedsoftmax_" + prefix + "_" + std::to_string(batch) + "_" +
         std::to_string(m) + "_" + std::to_string(k);
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::tuple<int, int, int>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto tensor = supported_shapes[i];
    auto key = get_instr_key(txn_fname_prefix_, get<0>(tensor), get<1>(tensor),
                             get<2>(tensor));
    instructions.push_back(std::make_pair(key, false));
  }

  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}
template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  get_inputs();

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
}

template <typename LhsT, typename MaskT, typename OutT>
masked_softmax<LhsT, MaskT, OutT>::masked_softmax(
    const std::string &operand_dtype, bool load_xrt) {
  if (operand_dtype != "bfloat16")
    throw std::runtime_error(
        "MaskedSoftMax only supportes homogeneous bfloat16 data type "
        "for activation, mask and result");
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(LhsT);

  txnbin_operand_header = {{"bfloat16", "a16"}};

  default_shapes_["maskedsoftmax_a16"] =
      std::vector<std::tuple<int, int, int>>();
  default_shapes_["maskedsoftmax_a16"].push_back(
      std::make_tuple(32, 2048, 2048));

  masked_softmax_id_ = masked_softmax_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName();

  txn_fname_prefix_ =
      "maskedsoftmax_" + txnbin_operand_header.at(operand_dtype_);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  // TODO not really sure we need this member
  kernel_x_shape_[0] = 32;
  kernel_x_shape_[1] = 2048;
  kernel_x_shape_[2] = 2048;
  auto shapeOperand = std::make_tuple(32, 2048, 2048);
  operand_size_in_bytes_ = tuple_product(shapeOperand) * sizeof(LhsT);
  mask_size_in_bytes_ =
      tuple_product(std::make_tuple(1, 2048, 2048)) * sizeof(MaskT);

  std::call_once(logger_flag_, []() {
    std::string header = "masked_softmax_id Batch M N Execute"
                         "time(ns) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "Mask_copy_time(ns) Mask_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE(
      "[MASKEDSOFTMAX] ID: " + std::to_string(masked_softmax_id_) +
      ", XCLBIN: " + XCLBIN_FNAME + ", (operand_dtype, b_dtype, c_dtype): (" +
      operand_dtype_ + ", " + operand_dtype_ + ", " + operand_dtype_ + ")");
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::execute(std::vector<Tensor> &input,
                                                std::vector<Tensor> &output) {

  // The first data is a and second data is b
  LhsT *a = (LhsT *)input.at(0).data;
  MaskT *b = (MaskT *)input.at(1).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  if (!isSupportedShape(input.at(0))) {
    throw std::runtime_error("Unsupported shape for masked softmax");
  }
  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  LhsT *a_bo_map = a_bo_.map<LhsT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes_);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // b_bo copy
  int64_t b_copy_start = GET_ELAPSED_TIME_NS();
  MaskT *b_bo_map = b_bo_.map<MaskT *>();
  memcpy((void *)b_bo_map, (void *)b, mask_size_in_bytes_);
  int64_t b_copy_stop = GET_ELAPSED_TIME_NS();

  // b_bo sync
  int64_t b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t b_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = b_copy_stop - b_copy_start;
  b_sync_time_ = b_sync_stop - b_sync_start;
  std::vector<xrt::bo> inputs = {a_bo_, b_bo_};
  std::vector<xrt::bo> outputs = {c_bo_};
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  // do we really need to sync before? c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  execute(inputs, outputs);
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;
  num_run_aie_++;

  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;

  // copy c_bo to host memory
  auto aie_out = (OutT *)output.at(0).data;
  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  OutT *c_bo_map = c_bo_.map<OutT *>();
  memcpy((void *)aie_out, (void *)c_bo_map, operand_size_in_bytes_);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(masked_softmax_id_) + " " +
      std::to_string(kernel_x_shape_[0]) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_x_shape_[2]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(b_copy_time_) + " " + std::to_string(b_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::execute(std::vector<xrt::bo> &input,
                                                std::vector<xrt::bo> &output) {
  // prepare inst_bo and param_bo
  const auto instr_bo_key =
      get_instr_key(txn_fname_prefix_, kernel_x_shape_[0], kernel_x_shape_[1],
                    kernel_x_shape_[2]);
  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;

  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  // do we really need to sync before? c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                input[0].address() + DDR_AIE_ADDR_OFFSET,
                input[1].address() + DDR_AIE_ADDR_OFFSET,
                output[0].address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  run.wait2();
}
template <typename LhsT, typename MaskT, typename OutT>
std::vector<xrt::bo> masked_softmax<LhsT, MaskT, OutT>::get_inputs() {
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));

  b_bo_ = xrt::bo(xrt_ctx_->get_device(), mask_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  return {a_bo_, b_bo_};
}

template <typename LhsT, typename MaskT, typename OutT>
const std::vector<uint8_t>
masked_softmax<LhsT, MaskT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  const auto [B, M, K] = extract_BMK(input.at(0));
  std::string txn_key = get_instr_key(txn_fname_prefix_, B, M, K);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename MaskT, typename OutT>
const std::vector<uint8_t>
masked_softmax<InT, MaskT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  return {};
}

template <typename LhsT, typename MaskT, typename OutT>
std::vector<OpArgMap> masked_softmax<LhsT, MaskT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  const auto shapeOperand = extract_BMK(input.at(0));
  const auto shapeMask = extract_BMK(input.at(1));
  const auto shapeResult = extract_BMK(output.at(0));

  if ((shapeOperand != shapeResult)) {
    throw std::runtime_error("mismatch shape of activation and result not "
                             "supported for masked softmax\n");
  }
  if (std::get<1>(shapeResult) != std::get<1>(shapeMask) ||
      std::get<2>(shapeResult) != std::get<2>(shapeMask)) {
    throw std::runtime_error("Mismatched shape of mask and activation/result "
                             "not supported for masked softmax");
  }
  const auto numElementsOperand = tuple_product(shapeOperand);
  const auto numElementsMask = tuple_product(shapeMask);
  size_t input_1_bo_size = (numElementsOperand * sizeof(LhsT));
  size_t input_2_bo_size = (numElementsMask * sizeof(MaskT));
  size_t output_bo_size = (numElementsOperand * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size},
  };
  return arg_map;
}

template class masked_softmax<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
