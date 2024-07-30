/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <iostream>
#include <map>
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

// Subroutines to read the transaction binary
#include "op_buf.hpp"
#include "op_types.h"

#include <utils/instruction_registry.hpp>
#include <utils/txn_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/elwmul/elwmul.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

#include "../ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {

namespace {
std::string getXCLBinName() {
  return OpInterface::get_dod_base_dir() +
         LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;
}
// Function to calculate the maximum pairwise product
auto max_pairwise_product(const tuple<int, int> &t) {
  return get<0>(t) * get<1>(t);
}

// Function to calculate the maximum pairwise product of a vector of tuples
int max_pairwise_product(const vector<tuple<int, int>> &supportedMatrixShapes) {
  auto max_product_iter =
      max_element(supportedMatrixShapes.begin(), supportedMatrixShapes.end(),
                  [](const auto &t1, const auto &t2) {
                    return max_pairwise_product(t1) < max_pairwise_product(t2);
                  });
  return max_pairwise_product(*max_product_iter);
}

} // namespace

static std::tuple<int, int> extract_MK(const std::vector<Tensor> &inputs) {
  int M = 0;
  int K = 0;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
    K = inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 3) {
    if (inputs.at(0).shape.at(0) != 1)
      std::runtime_error("Only batch size of 1 supported for elementwise mul");
    M = inputs.at(0).shape.at(1);
    K = inputs.at(0).shape.at(2);
  }
  return std::make_tuple(M, K);
}

template <typename LhsT, typename RhsT, typename OutT>
bool elw_mul<LhsT, RhsT, OutT>::isSupportedShape(const Tensor &operand) {
  const auto &supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (const auto &supported : supported_shapes) {
    if (std::get<0>(supported) == operand.shape.at(0) &&
        std::get<1>(supported) == operand.shape.at(1))
      return true;
  }
  return false;
}

template <typename LhsT, typename RhsT, typename OutT>
std::once_flag elw_mul<LhsT, RhsT, OutT>::logger_flag_;

template <typename LhsT, typename RhsT, typename OutT>
uint64_t elw_mul<LhsT, RhsT, OutT>::elw_mul_count = 0;

template <typename LhsT, typename RhsT, typename OutT>
std::once_flag elw_mul<LhsT, RhsT, OutT>::instr_reg_flag_;

template <typename LhsT, typename RhsT, typename OutT>
void elw_mul<LhsT, RhsT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename LhsT, typename RhsT, typename OutT>
std::string elw_mul<LhsT, RhsT, OutT>::get_instr_key(std::string prefix, int m,
                                                     int k) {
  // NOTE the need of that first "elwmul_" is weird....
  //  it seems that the first "elwmul_" indicates a higher level folder?
  return "elwmul_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k);
}

template <typename LhsT, typename RhsT, typename OutT>
void elw_mul<LhsT, RhsT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::tuple<int, int>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes[i];
    auto key = get_instr_key(txn_fname_prefix_, get<0>(mat), get<1>(mat));
    instructions.push_back(std::make_pair(key, false));
  }

  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}

template <typename LhsT, typename RhsT, typename OutT>
elw_mul<LhsT, RhsT, OutT>::elw_mul(const std::string &operand_dtype,
                                   bool load_xrt) {
  if (operand_dtype != "bfloat16")
    std::runtime_error("Elwmul only supportes homogeneous bfloat16 data type "
                       "for all operands and result");
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(LhsT);

  txnbin_operand_header = {{"bfloat16", "a16"}};

  default_shapes_["elwmul_a16a16"] = std::vector<std::tuple<int, int>>();
  default_shapes_["elwmul_a16a16"].push_back(std::make_tuple(1, 11008));
  default_shapes_["elwmul_a16a16"].push_back(std::make_tuple(128, 11008));
  default_shapes_["elwmul_a16a16"].push_back(std::make_tuple(256, 11008));
  default_shapes_["elwmul_a16a16"].push_back(std::make_tuple(512, 11008));
  default_shapes_["elwmul_a16a16"].push_back(std::make_tuple(1024, 11008));
  default_shapes_["elwmul_a16a16"].push_back(std::make_tuple(2048, 11008));

  elw_mul_id_ = elw_mul_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName();

  txn_fname_prefix_ = "elwmul_" + txnbin_operand_header.at(operand_dtype_) +
                      txnbin_operand_header.at(operand_dtype_);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
    // preempting bo creation with largest shape for unit testing
    const size_t operand_size_in_bytes =
        max_pairwise_product(default_shapes_["elwmul_a16a16"]) *
        operand_dtype_size_;
    a_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes,
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    b_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes,
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    c_bo_ = xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes,
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = "elw_mul_id M N kernel_m kernel_n num_aie_runs Execute"
                         "time(us) run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "B_copy_time(ns) B_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[MUL] ID: " + std::to_string(elw_mul_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (operand_dtype, b_dtype, c_dtype): (" + operand_dtype_ +
                    ", " + operand_dtype_ + ", " + operand_dtype_ + ")");
}
template <typename LhsT, typename RhsT, typename OutT>
void elw_mul<LhsT, RhsT, OutT>::execute(std::vector<xrt::bo> &inputs,
                                        std::vector<xrt::bo> &outputs) {
  // prepare inst_bo and param_bo
  const auto instr_bo_key =
      get_instr_key(txn_fname_prefix_, kernel_x_shape_[0], kernel_x_shape_[1]);
  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;

  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  // do we really need to sync before? c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                inputs[0].address() + DDR_AIE_ADDR_OFFSET,
                inputs[1].address() + DDR_AIE_ADDR_OFFSET,
                outputs[0].address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  run.wait2();
}
template <typename LhsT, typename RhsT, typename OutT>
void elw_mul<LhsT, RhsT, OutT>::set_kernel_shape(std::vector<size_t> shape) {
  // TODO not really sure we need this member
  kernel_x_shape_[0] = shape.at(0);
  kernel_x_shape_[1] = shape.at(1);
}

template <typename LhsT, typename RhsT, typename OutT>
void elw_mul<LhsT, RhsT, OutT>::execute(std::vector<Tensor> &input,
                                        std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 2) {
    throw std::runtime_error("elwmul IPU Wrapper expect to have two inputs.");
  }
  const int a_idx = 0;
  // The first data is a and second data is b
  LhsT *a = (LhsT *)input.at(a_idx).data;
  RhsT *b = (RhsT *)input.at(a_idx + 1).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  if (!isSupportedShape(input.at(0)))
    std::runtime_error("Unsupported shape for elemwise mul");

  // TODO not really sure we need this member
  set_kernel_shape(input.at(0).shape);

  const auto operand_size_in_bytes =
      input.at(0).shape.at(0) * input.at(0).shape.at(1) * sizeof(LhsT);
  RYZENAI_LOG_TRACE("elwmul: operand_size_in_bytes:" +
                    std::to_string(operand_size_in_bytes));

  // TODO this trace looks repetitive w.r.t. the previous
  /* Create input/output BOs */
  RYZENAI_LOG_TRACE(
      "elwmul: A_BO_SIZE:" + std::to_string(operand_size_in_bytes) +
      " B_BO_SIZE:" + std::to_string(operand_size_in_bytes) +
      " C_BO_SIZE:" + std::to_string(operand_size_in_bytes));

  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  LhsT *a_bo_map = a_bo_.map<LhsT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // b_bo copy
  int64_t b_copy_start = GET_ELAPSED_TIME_NS();
  RhsT *b_bo_map = b_bo_.map<RhsT *>();
  memcpy((void *)b_bo_map, (void *)b, operand_size_in_bytes);
  int64_t b_copy_stop = GET_ELAPSED_TIME_NS();

  // b_bo sync
  int64_t b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t b_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = b_copy_stop - b_copy_start;
  b_sync_time_ = b_sync_stop - b_sync_start;

  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  std::vector<xrt::bo> inputs = {a_bo_, b_bo_};
  std::vector<xrt::bo> outputs = {c_bo_};
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
  memcpy((void *)aie_out, (void *)c_bo_map, operand_size_in_bytes);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(elw_mul_id_) + " " + std::to_string(kernel_x_shape_[0]) +
      " " + std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_x_shape_[0]) + " " +
      std::to_string(kernel_x_shape_[1]) + " " + std::to_string(num_run_aie_) +
      " " + std::to_string(exec_end - exec_start) + " " +
      std::to_string(run_aie_time_) + " " + std::to_string(a_copy_time_) + " " +
      std::to_string(a_sync_time_) + " " + std::to_string(b_copy_time_) + " " +
      std::to_string(b_sync_time_) + " " + std::to_string(c_copy_time_) + " " +
      std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename LhsT, typename RhsT, typename OutT>
const std::vector<uint8_t> elw_mul<LhsT, RhsT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K] = extract_MK(input);
  std::string txn_key = get_instr_key(txn_fname_prefix_, M, K);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename RhsT, typename OutT>
const std::vector<uint8_t> elw_mul<InT, RhsT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  return {};
}

template <typename LhsT, typename RhsT, typename OutT>
std::vector<OpArgMap> elw_mul<LhsT, RhsT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto M1 = input.at(0).shape.at(1); // [1xMxN : 1x512x768]
  auto N1 = input.at(0).shape.at(2);
  auto M2 = input.at(1).shape.at(1); // [1xMxN : 1x512x768]
  auto N2 = input.at(1).shape.at(2);
  auto M3 = output.at(0).shape.at(1); // [1xMxN : 1x512x768]
  auto N3 = output.at(0).shape.at(2);

  if ((M1 != M2) || (N1 != N2) || (N3 != N2) || (M3 != M1)) {
    throw std::runtime_error(
        "Dimensions of all tensors should be equal for eltwise mul op\n");
  }
  size_t input_1_bo_size = (M1 * N1 * sizeof(LhsT));
  size_t input_2_bo_size = (M1 * N1 * sizeof(RhsT));
  size_t output_bo_size = (M1 * N1 * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size},
  };
  return arg_map;
}

template class elw_mul<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
