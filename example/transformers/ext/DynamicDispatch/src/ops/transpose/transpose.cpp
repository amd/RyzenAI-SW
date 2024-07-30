/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <any>
#include <iostream>
#include <map>
#include <sstream>
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

#include <ops/op_interface.hpp>
#include <ops/transpose/transpose.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {
template <typename T>
static std::tuple<T, T, T> extract_HMK(const std::vector<Tensor> &inputs) {
  T H = 0;
  T M = 0;
  T K = 0;
  if (inputs.at(0).shape.size() == 3) {
    H = inputs.at(0).shape.at(0);
    M = inputs.at(0).shape.at(1);
    K = inputs.at(0).shape.at(2);
  }
  return std::make_tuple(H, M, K);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag transpose<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t transpose<InT, WtT, OutT>::transpose_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag transpose<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void transpose<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string transpose<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                     std::vector<size_t> &mat) {
  std::string out_str = "transpose_" + prefix;
  for (size_t i = 0; i < mat.size(); i++) {
    out_str += "_" + std::to_string(mat[i]);
  }
  return out_str;
}

template <typename InT, typename WtT, typename OutT>
void transpose<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::vector<size_t>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat);
    auto param_key = get_instr_key(param_fname_prefix_, mat) + "_param";
    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }

  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
transpose<InT, WtT, OutT>::transpose(const std::string &a_dtype,
                                     const std::string &b_dtype,
                                     const std::string &c_dtype,
                                     bool load_xrt) {

  txnbin_a_header = {{"uint16", "a16"}, {"uint8", "a8"}};

  default_shapes_["transpose_a16a16"] = std::vector<std::vector<size_t>>();

  default_shapes_["transpose_a16a16"].push_back(
      std::vector<size_t>{1, 3136, 128});
  default_shapes_["transpose_a16a16"].push_back(
      std::vector<size_t>{1, 784, 256});
  default_shapes_["transpose_a16a16"].push_back(
      std::vector<size_t>{1, 196, 512});
  default_shapes_["transpose_a16a16"].push_back(
      std::vector<size_t>{1, 49, 1024});
  default_shapes_["transpose_a16a16"].push_back(
      std::vector<size_t>{64, 49, 128});
  default_shapes_["transpose_a16a16"].push_back(
      std::vector<size_t>{16, 49, 256});
  default_shapes_["transpose_a16a16"].push_back(
      std::vector<size_t>{4, 49, 512});

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  transpose_id_ = transpose_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + ryzenai::PSF_A8W8_QDQ_XCLBIN_PATH;

  if (a_dtype_ == "uint16") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSI_A16W8_QDQ_XCLBIN_PATH;
  }

  txn_fname_prefix_ = "transpose_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_a_header.at(a_dtype_);

  param_fname_prefix_ = "transpose_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_a_header.at(a_dtype_);

  KERNEL_M_MAX = 512;

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = "transpose_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[ADD] ID: " + std::to_string(transpose_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void transpose<InT, WtT, OutT>::set_params(const std::string &model_name,
                                           std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "PSI") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSI_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  w_shape_[0] = input_shape.at(0) * input_shape.at(1);
  w_shape_[1] = input_shape.at(2);

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void transpose<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params) {
  RYZENAI_LOG_TRACE("Transpose initialize_const_params(ptr) ...");

  RYZENAI_LOG_TRACE("Transpose initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void transpose<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params) {

  // if (const_params.size() != 1) {
  //   throw std::runtime_error("TRANSPOSE IPU Wrapper expect to have one
  //   constant.");
  // }

  kernel_x_shape_[0] = w_shape_[0];
  kernel_x_shape_[1] = w_shape_[1];

  kernel_y_shape_[0] = 0;
  kernel_y_shape_[1] = 0;

  kernel_z_shape_[0] = w_shape_[0];
  kernel_z_shape_[1] = w_shape_[1];

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input/output BOs */
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  RYZENAI_LOG_TRACE("TRANPOSE: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(0) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), 10,
                  XRT_BO_FLAGS_HOST_ONLY, // dummy BO not used
                  xrt_ctx_->get_kernel().group_id(8));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  // copy b_bo
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;

  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  initialize_const_params(b_bo_map, const_params);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += b_format_stop - b_format_start;
  b_copy_time_ = b_copy_stop - b_copy_start;

  // sync b_bo
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ = b_sync_stop - b_sync_start;
}

template <typename InT, typename WtT, typename OutT>
void transpose<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                        std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error(
        "TRANSPOSE IPU Wrapper expect to have one inputs.");
  }
  const int a_idx = 0;
  // The first data is a
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  auto H = input.at(a_idx).shape.at(0);
  auto M = input.at(a_idx).shape.at(1);
  auto K = input.at(a_idx).shape.at(2);

  int a_size = H * M * K * sizeof(InT);
  RYZENAI_LOG_TRACE("TRANSPOSE: a_size:" + std::to_string(a_size));
  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, a_size);

  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // prepare inst_bo and param_bo
  std::vector<size_t> param_shape = {H, M, K};
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, param_shape);
  auto param_bo_key =
      get_instr_key(param_fname_prefix_, param_shape) + "_param";
  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  const xrt::bo &param_bo = instr_reg_.get_param_bo(param_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                param_bo.address() + DDR_AIE_ADDR_OFFSET, 0);
  run.wait2();
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
  memcpy((void *)aie_out, (void *)c_bo_map, H * M * K * sizeof(OutT));
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(transpose_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> transpose<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [H, M, K] = extract_HMK<size_t>(input);
  std::vector<size_t> param_shape = {H, M, K};
  std::string txn_key = get_instr_key(txn_fname_prefix_, param_shape);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> transpose<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [H, M, K] = extract_HMK<size_t>(input);
  std::vector<size_t> param_shape = {H, M, K};
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      get_instr_key(param_fname_prefix_, param_shape) + "_param";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  Transaction &txn = Transaction::getInstance();
  std::string param_string = txn.get_txn_str(param_key);
  std::istringstream params_stream(param_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(params_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> transpose<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto H1 = input.at(0).shape.at(0);
  auto M1 = input.at(0).shape.at(1); // [1xMxN : 1x512x768]
  auto N1 = input.at(0).shape.at(2);

  // Transpose does not require any const bo by default.
  // If we are not passing const BO to to arg map, we are seeing error.
  // To avoid this we have added dummy buffer.
  size_t const_params_bo_size = 16; // Dummy Buffer
  size_t input_1_bo_size = (H1 * M1 * N1 * sizeof(InT));

  size_t output_bo_size = (H1 * M1 * N1 * sizeof(OutT));
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0,
       const_params_bo_size}, // Dummy allocation
      {OpArgMap::OpArgType::OUTPUT, 0, 2, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  return arg_map;
}

template class transpose<uint16_t, int8_t, uint16_t>;

} // namespace ryzenai
