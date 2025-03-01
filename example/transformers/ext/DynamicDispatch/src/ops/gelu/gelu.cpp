/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <any>
#include <fstream>
#include <iostream>
#include <map>
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

#include "ops/ops_common/gelu_lut_bf16_512.h"
#include "ops/ops_common/matmul_matrix.hpp"
// #include "ops/ops_common/silu_lut_bf16_512.h"
#include <ops/gelu/gelu.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include <xaiengine.h>
using namespace matmul_matrix;
namespace ryzenai {
static std::tuple<int, int> extract_MK(const std::vector<Tensor> &inputs) {
  int M = 0;
  int K = 0;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
    K = inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 3) {
    M = inputs.at(0).shape.at(1);
    K = inputs.at(0).shape.at(2);
  } else if (inputs.at(0).shape.size() == 4) {
    if (inputs.at(0).shape.at(1) == inputs.at(0).shape.at(2)) { // NHWC
      M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1) *
          inputs.at(0).shape.at(2);
      K = inputs.at(0).shape.at(3);
    } else { // NCHW
      M = inputs.at(0).shape.at(2) * inputs.at(0).shape.at(3);
      K = inputs.at(0).shape.at(1);
    }
  }
  return std::make_tuple(M, K);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<int, int> gelu<InT, WtT, OutT>::map_padded_shape(int M, int N) {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<std::tuple<int, int>> &supported_shapes = iter->second;
  int Mo = M;
  int No = N;
  int fidx = 0;
  int f_found = 0;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes[i];
    if (M == (int)get<0>(mat) && N == (int)get<1>(mat)) {
      fidx = i;
      f_found = 1;
      break;
    }
  }
  if (f_found == 1) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<std::tuple<int, int>> &actual_shapes = iter->second;
    auto mat = actual_shapes[fidx];
    Mo = (int)get<0>(mat);
    No = (int)get<1>(mat);
  } else {
    throw std::runtime_error("Can not find the shape");
  }
  // std::cout << Mo << ' ' << No << std::endl;
  return std::make_tuple(Mo, No);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag gelu<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t gelu<InT, WtT, OutT>::gelu_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag gelu<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string gelu<InT, WtT, OutT>::get_instr_key(std::string prefix, int m,
                                                int k) {
  // return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + ".bin";
  return "gelu_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> gelu<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K] = extract_MK(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  std::string txn_key = get_instr_key(txn_fname_prefix_, Mo, Ko);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> gelu<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K] = extract_MK(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  // TODO: Add check to validate tensor shapes
  std::string param_key = get_instr_key(param_fname_prefix_, Mo, Ko) + "_param";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  Transaction &txn = Transaction::getInstance();
  std::string param_string = txn.get_txn_str(param_key);
  std::istringstream params_stream(param_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(params_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> gelu<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // input --> [input, gamma, beta, output]
  // Check if IO buffers have batch.
  auto [M, N] = extract_MK(input);
  auto [Mo, No] = map_padded_shape(M, N);

  int const gelu_lut_ab_size = sizeof(lnr_lutab);
  int const gelu_lut_cd_size = sizeof(lnr_lutcd);

  size_t const_params_bo_size =
      (gelu_lut_ab_size + gelu_lut_cd_size +
       matmul_matrix::QDQparam_size * sizeof(int16_t));
  size_t input_bo_size = (Mo * No * sizeof(InT));
  size_t output_bo_size = (Mo * No * sizeof(OutT));
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 2, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;

  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::tuple<int, int>> &supported_shapes = iter->second;
    for (size_t i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes[i];
      auto key = get_instr_key(mkey, get<0>(mat), get<1>(mat));
      auto param_key = get_instr_key(mkey, get<0>(mat), get<1>(mat)) + "_param";
      instructions.push_back(std::make_pair(key, false));
      layer_params.push_back(std::make_pair(param_key, false));
    }
  }

  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
gelu<InT, WtT, OutT>::gelu(const std::string &a_dtype,
                           const std::string &b_dtype,
                           const std::string &c_dtype, bool load_xrt,
                           const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"bfloat16", "abf16"}, {"uint16", "a16"}};

  txnbin_acc_header = {{"bfloat16", "accbf16"}, {"uint16", "acc16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["gelu_4x2_a16accbf16"] = std::vector<std::tuple<int, int>>();

  default_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(64, 5120));
  default_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(256, 5120));
  default_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(1024, 2560));
  default_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(4096, 1280));

  default_shapes_["gelu_4x4_a16accbf16"] = std::vector<std::tuple<int, int>>();

  default_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(64, 5120));
  default_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(256, 5120));
  default_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(1024, 2560));
  default_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(4096, 1280));

  // raw shape is the actual shape from ONNX, sequence needs to match with
  // default_shape
  raw_shapes_["gelu_4x2_a16accbf16"] = std::vector<std::tuple<int, int>>();

  raw_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(64, 5120));
  raw_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(256, 5120));
  raw_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(1024, 2560));
  raw_shapes_["gelu_4x2_a16accbf16"].push_back(std::make_tuple(4096, 1280));

  raw_shapes_["gelu_4x4_a16accbf16"] = std::vector<std::tuple<int, int>>();

  raw_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(64, 5120));
  raw_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(256, 5120));
  raw_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(1024, 2560));
  raw_shapes_["gelu_4x4_a16accbf16"].push_back(std::make_tuple(4096, 1280));

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  gelu_id_ = gelu_count++;
  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + ryzenai::PSF_A8W8_QDQ_XCLBIN_PATH;

  if (c_dtype_ == "uint16") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSJ_A16W8_QDQ_XCLBIN_PATH;
  }

  design_param_ = "";
  if (attr.count("design_param") &&
      attr.at("design_param").type() == typeid(std::vector<string>)) {
    const auto &design_param_vector =
        std::any_cast<const std::vector<string> &>(attr.at("design_param"));

    if (design_param_vector.size() == 1) {
      design_param_ = design_param_vector[0];
    } else {
      std::cout
          << "Design Format attribute does not have the expected number of "
             "elements.Number of passed : design_param_vector.size(), "
             "Expected:1"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("iConv: DesignFormat: " + design_param_);
  }

  txn_fname_prefix_ = "gelu_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_acc_header.at(c_dtype_);

  param_fname_prefix_ = "gelu_4x2_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "gelu_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

    param_fname_prefix_ = "gelu_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_acc_header.at(c_dtype_);
  }

  KERNEL_M_MAX = 512;

  w_shape_[0] = KERNEL_M_MAX;

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
    std::string header = "gelu_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[Gelu] ID: " + std::to_string(gelu_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
                    a_dtype + ", " + b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::set_params(const std::string &model_name,
                                      std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "PSR") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSR_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "4x4PSR") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSR4x4_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  // for memory allocation
  auto [M, N] = map_padded_shape(input_shape.at(0), input_shape.at(1));
  w_shape_[0] = M;
  w_shape_[1] = N;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Gelu initialize_const_params(ptr) ...");

  DOD_THROW_IF((const_params.size() != 1) ||
                   (const_params.at(0).shape.size() != 1),
               OpsFusion::dod_format("Unsupported const spec for Gelu\n") +
                   OpsFusion::dod_format(
                       "(Details : #const params == 1 ({}), Const param1 dim "
                       "== 1 ({})",
                       const_params.size(), const_params.at(0).shape.size()));

  const int qdq_params_idx = 0;

  auto qdq_params = (int16_t *)const_params.at(qdq_params_idx).data;

  // SW convert scale to 1/scale and bfloat16 for Q

  auto w_dest = (WtT *)dest;
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int16_t);

  int const gelu_size_lutab = sizeof(lnr_lutab);
  int const gelu_size_lutcd = sizeof(lnr_lutcd);

  // int const gelu_offset = silu_size_lutab + silu_size_lutcd;

  memcpy((void *)(static_cast<int8_t *>(dest)), (void *)lnr_lutab,
         gelu_size_lutab);
  memcpy((void *)(static_cast<int8_t *>(dest) + gelu_size_lutab),
         (void *)lnr_lutcd, gelu_size_lutcd);

  auto offset = gelu_size_lutab + gelu_size_lutcd;
  memcpy((void *)(static_cast<int8_t *>(dest) + offset), (void *)qdq_params,
         qdq_params_size);

  RYZENAI_LOG_TRACE("Gelu initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // Check the number of inputs
  if (const_params.size() != 1) {
    throw std::runtime_error("Gelu IPU Wrapper expect to have one constant.");
  }

  int const gelu_lut_ab_size = sizeof(lnr_lutab);
  int const gelu_lut_cd_size = sizeof(lnr_lutcd);

  auto lut_size = gelu_lut_ab_size + gelu_lut_cd_size;

  // Init the BO size
  kernel_x_shape_[0] = w_shape_[0];
  kernel_x_shape_[1] = w_shape_[1];
  ;
  kernel_z_shape_[0] = w_shape_[0];
  kernel_z_shape_[1] = w_shape_[1];
  ;

  // Create input/output BOs
  const int B_BO_SIZE =
      lut_size + matmul_matrix::QDQparam_size * sizeof(int16_t);
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  RYZENAI_LOG_TRACE("Gelu: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
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
void gelu<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                   std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error("Gelu IPU Wrapper expect to have one input.");
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

  a_shape_[0] = input.at(a_idx).shape.at(0);
  a_shape_[1] = input.at(a_idx).shape.at(1);

  c_shape_[0] = a_shape_[0];
  c_shape_[1] = a_shape_[1];

  kernel_x_rows = a_shape_[0];

  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  int a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  RYZENAI_LOG_TRACE("Gelu: a_size:" + std::to_string(a_size));
  memcpy((void *)a_bo_map, (void *)a, a_size);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // prepare inst_bo and param_bo
  auto [M, K] = map_padded_shape(a_shape_[0], a_shape_[1]);
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, M, K);
  auto param_bo_key = get_instr_key(param_fname_prefix_, M, K) + "_param";
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
  memcpy((void *)aie_out, (void *)c_bo_map,
         c_shape_[0] * c_shape_[1] * sizeof(OutT));
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(gelu_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1]) +
      " " + std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template class gelu<uint16_t, uint16_t, uint16_t>;
} // namespace ryzenai
