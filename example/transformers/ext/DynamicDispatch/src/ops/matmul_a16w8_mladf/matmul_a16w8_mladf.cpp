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

#include <ops/matmul_a16w8_mladf/matmul_a16w8_mladf.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

#include <utils/instruction_registry.hpp>

// AIE Driver header
#include "xaiengine.h"

// Headers for BFP matrix formatting
#include "ops/ops_common/matmul_a16w8_mladf_matrix.hpp"
using namespace matmul_a16w8_mladf_matrix;
namespace ryzenai {

static std::tuple<int, int, int>
extract_MKN(const std::vector<Tensor> &inputs) {
  // inputs[0] --> input
  // inputs[1] --> wts

  int M;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
  } else if (inputs.at(0).shape.size() == 3) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1);
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }
  int K = inputs.at(1).shape.at(0);
  int N = inputs.at(1).shape.at(1);

  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<int, int>
matmul_a16w8_mladf<InT, WtT, OutT>::map_padded_shape(int M, int N) {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<matrix_shapes> &supported_shapes = iter->second;
  int Mo = M;
  int No = N;
  int fidx = 0;
  int f_found = 0;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    if ((M == mat.M) && (N == mat.N)) {
      fidx = i;
      f_found = 1;
      break;
    }
  }
  if (f_found == 1) {
    auto iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<matrix_shapes> &actual_shapes = iter->second;
    auto mat = actual_shapes.at(fidx);
    Mo = (int)mat.M;
    No = (int)mat.K;
  } else {
    throw std::runtime_error("Can not find the shape");
  }
  return std::make_tuple(Mo, No);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag matmul_a16w8_mladf<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t matmul_a16w8_mladf<InT, WtT, OutT>::matmul_mladf_count = 0;

template <typename InT, typename WtT, typename OutT>
char matmul_a16w8_mladf<InT, WtT, OutT>::kernel_params_[64];

// template <typename InT, typename WtT, typename OutT>
// instruction_registry matmul_mladf<InT, WtT, OutT>::instr_reg_;

template <typename InT, typename WtT, typename OutT>
std::once_flag matmul_a16w8_mladf<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string
matmul_a16w8_mladf<InT, WtT, OutT>::get_instr_key(std::string prefix, int m,
                                                  int k, int n) {
  auto instr_key = prefix + "_" + std::to_string(m) + "_" + std::to_string(k) +
                   "_" + std::to_string(n);
  return instr_key;
}

template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  // GEMM
  txn_fname_prefix_ = "gemm_mladf_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);

    auto key =
        "gemm_mladf_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
matmul_a16w8_mladf<InT, WtT, OutT>::matmul_a16w8_mladf(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt) {

  matmul_a16w8_mladf<InT, WtT, OutT>::txnbin_a_header = {{"uint16", "a16"},
                                                         {"int16", "a16"}};

  txnbin_b_header = {{"int8", "w8"}, {"uint8", "w8"}};

  txnbin_acc_header = {{"int32", "acc32"},
                       {"uint16", "acc16"},
                       {"int16", "acc16"},
                       {"uint8", "acc8"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["gemm_mladf_a16w8acc16"] = std::vector<matrix_shapes>();
  default_shapes_["gemm_mladf_a16w16acc16"] = std::vector<matrix_shapes>();

  default_shapes_["gemm_mladf_a16w8acc16"].emplace_back(4096, 512, 512);
  default_shapes_["gemm_mladf_a16w16acc16"].emplace_back(4096, 512, 4096);
  default_shapes_["gemm_mladf_a16w16acc16"].emplace_back(4096, 4096, 512);

  // raw shape is the actual shape from ONNX
  raw_shapes_["gemm_mladf_a8w8acc8"] = std::vector<matrix_shapes>();
  raw_shapes_["gemm_mladf_a16w8acc16"] = std::vector<matrix_shapes>();

  raw_shapes_["gemm_mladf_a16w8acc16"].emplace_back(4096, 512, 512);
  raw_shapes_["gemm_mladf_a16w16acc16"].emplace_back(4096, 512, 4096);
  raw_shapes_["gemm_mladf_a16w16acc16"].emplace_back(4096, 4096, 512);

  DPU_DIR = OpInterface::get_dod_base_dir() + "//transaction//" + "stx";

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  param_dtype_size_ = sizeof(uint32_t);
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  matmul_mladf_id_ = matmul_mladf_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                             ryzenai::MLADF_4x2_GEMM_A16W8_XCLBIN_PATH;

  if ((b_dtype_ == "uint16") || (b_dtype_ == "int16")) {
    XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                   ryzenai::MLADF_4x2_GEMM_A16W16_XCLBIN_PATH;
  }
  txn_fname_prefix_ = "gemm_mladf_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }
  KERNEL_M_MAX = 4096;

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
    std::string header =
        "matmul_mladf_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[OP] ID: " + std::to_string(matmul_mladf_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if ((b_dtype_ == "int8") || (b_dtype_ == "uint8")) {
    XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                   ryzenai::MLADF_4x2_GEMM_A16W8_XCLBIN_PATH;
  } else if ((b_dtype_ == "int16") || (b_dtype_ == "uint16")) {
    XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                   ryzenai::MLADF_4x2_GEMM_A16W16_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }
  auto [M, K] = map_padded_shape(input_shape.at(0), input_shape.at(1));
  KERNEL_M_MAX = M;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  RYZENAI_LOG_TRACE("GEMM_MLADF: w_shape0:" + std::to_string(w_shape_[0]) +
                    " w_shape1:" + std::to_string(w_shape_[1]));
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  kernel_x_shape_[1] = w_shape_[0];
  kernel_y_shape_[0] = w_shape_[0];
  kernel_y_shape_[1] = w_shape_[1];
  kernel_z_shape_[1] = w_shape_[1];
}

template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::initialize_wts(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("matmul_a16w8_mladf initialize_const_params(ptr) ...");
  DOD_THROW_IF(
      (const_params.size() < 3) || (const_params.at(0).shape.size() != 2) ||
          (const_params.at(1).shape.size() != 1),
      OpsFusion::dod_format("Unsupported const spec for Matmul Mladf\n") +
          OpsFusion::dod_format(
              "(Details : #const params == 2 ({}), Const param1 dim == 2 ({}), "
              "Const param2 dim == 1 ({})",
              const_params.size(), const_params.at(0).shape.size(),
              const_params.at(1).shape.size()));

  const int w_idx = 0, qdq_idx = 1, qdq_param_idx = 2;
  // The first data is Weight
  auto weights = (WtT *)const_params.at(w_idx).data;
  std::vector<size_t> shape = const_params.at(w_idx).shape;
  int size_weight = shape[0] * shape[1] * b_dtype_size_;
  w_shape_[0] = shape[0];
  w_shape_[1] = shape[1];

  set_kernel_shapes();

  auto qdq = (int64_t *)const_params.at(qdq_idx).data;
  auto qdq_params = (int32_t *)const_params.at(qdq_param_idx).data;

  int write_offset = 0;

  std::vector<WtT> buf(w_shape_[0] * w_shape_[1]);
  matmul_a16w8_mladf_matrix::WgtMatrix<WtT, Ksubv, Nsubv> W(
      w_shape_[0], w_shape_[1], buf.data());
  for (int r = 0; r < w_shape_[0]; ++r) {
    for (int c = 0; c < w_shape_[1]; ++c) {
      W.at(r, c) = weights[(r * w_shape_[1]) + c];
    }
  }

  auto total_size =
      matmul_a16w8_mladf_matrix::Ksubv * matmul_a16w8_mladf_matrix::Nsubv;
  auto qdq_size = matmul_a16w8_mladf_matrix::Nsubv * sizeof(int64_t);
  auto qdq_params_size =
      matmul_a16w8_mladf_matrix::QDQparam_size * sizeof(int32_t);
  //// WGT + Bias
  { // This section of the code interleaves bias with weights Nsubv of bias
    // with every K x N

    for (int N_shard = 0;
         N_shard < (w_shape_[1]) / (matmul_a16w8_mladf_matrix::Nsubv);
         N_shard++) {
      for (int K_shard = 0;
           K_shard < (w_shape_[0]) / (matmul_a16w8_mladf_matrix::Ksubv);
           K_shard++) {
        // WTS
        memcpy((void *)(reinterpret_cast<int8_t *>(dest) + write_offset),
               (void *)&buf[(N_shard * w_shape_[0] *
                             matmul_a16w8_mladf_matrix::Nsubv) +
                            (K_shard * total_size)],
               (total_size));
        write_offset += total_size;
        // BIAS
        memcpy((void *)(reinterpret_cast<int8_t *>(dest) + write_offset),
               (void *)&qdq[N_shard * matmul_a16w8_mladf_matrix::Nsubv],
               qdq_size);
        write_offset += qdq_size;
        // QDQ params
        memcpy((void *)(reinterpret_cast<int8_t *>(dest) + write_offset),
               (void *)qdq_params, qdq_params_size);
        write_offset += qdq_params_size;
      }
    }
  }
  RYZENAI_LOG_TRACE("matmul_a16w8_mladf initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  initialize_wts(dest, const_params);
  const Tensor &kernel_params = const_params.at(3);
  size_t params_bytes = kernel_params.shape[0] * sizeof(int32_t);
  memcpy(kernel_params_, kernel_params.data, params_bytes);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
matmul_a16w8_mladf<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  std::vector<uint8_t> params(64);
  memcpy(params.data(), kernel_params_, 64);
  return params;
}

// For MATMUL_Mladf: weight + qdq + qdq_params
template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // Check the number of inputs
  if (const_params.size() != 3) {
    throw std::runtime_error("MATMUL_Mladf expect to have three constant.");
  }
  const int w_idx = 0, qdq_idx = 1;
  // The first data is Weight
  // auto weight = (int8_t*)const_params.at(w_idx).data;
  std::vector<size_t> shape = const_params.at(w_idx).shape;
  int size_weight = shape[0] * shape[1] * b_dtype_size_;
  w_shape_[0] = shape[0];
  w_shape_[1] = shape[1];

  shape = const_params.at(qdq_idx).shape;
  int num_sv =
      ((w_shape_[0]) * (w_shape_[1])) /
      ((matmul_a16w8_mladf_matrix::Nsubv) * (matmul_a16w8_mladf_matrix::Ksubv));
  int size_interleaved_qdq =
      (matmul_a16w8_mladf_matrix::Nsubv * sizeof(int64_t) +
       matmul_a16w8_mladf_matrix::QDQparam_size * sizeof(int32_t)) *
      num_sv;
  // Init the BO size
  set_kernel_shapes();

  // Create input/output BOs
  const int PARAM_BO_SIZE = (16 * param_dtype_size_);
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const int B_BO_SIZE = size_weight + size_interleaved_qdq;
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  RYZENAI_LOG_TRACE("GEMM_MLADF: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  param_bo_ =
      xrt::bo(xrt_ctx_->get_device(), PARAM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  // copy b_bo
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  initialize_wts(b_bo_map, const_params);
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

// matmul mladf
template <typename InT, typename WtT, typename OutT>
void matmul_a16w8_mladf<InT, WtT, OutT>::execute(
    const std::vector<Tensor> &input, std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 2) {
    throw std::runtime_error("MATMUL_Mladf expect to have two input.");
  }
  const int param_idx = 0, a_idx = 1;
  // The first data is a
  InT *a = (InT *)input.at(a_idx).data;
  uint32_t *param = (uint32_t *)input.at(param_idx).data;

  param_copy_time_ = 0;
  param_sync_time_ = 0;
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

  param_shape_[0] = input.at(param_idx).shape.at(0);

  a_shape_[0] = input.at(a_idx).shape.at(0);
  a_shape_[1] = input.at(a_idx).shape.at(1);

  c_shape_[0] = a_shape_[0];
  c_shape_[1] = w_shape_[1];

  auto [M, K] = map_padded_shape(a_shape_[0], a_shape_[1]);
  kernel_x_rows = M;

  // param_bo copy
  int64_t param_copy_start = GET_ELAPSED_TIME_NS();
  uint32_t *param_bo_map = param_bo_.map<uint32_t *>();
  int param_size = param_shape_[0] * sizeof(uint32_t);
  memcpy((void *)param_bo_map, (void *)param, param_size);
  int64_t param_copy_stop = GET_ELAPSED_TIME_NS();

  // param_bo sync
  int64_t param_sync_start = GET_ELAPSED_TIME_NS();
  param_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t param_sync_stop = GET_ELAPSED_TIME_NS();

  param_copy_time_ = param_copy_stop - param_copy_start;
  param_sync_time_ = param_sync_stop - param_sync_start;

  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  int a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  memcpy((void *)a_bo_map, (void *)a, a_size);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // prepare inst_bo and param_bo
  auto instr_bo_key = "gemm_mladf_" + txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);
  const xrt::bo instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  uint32_t *param_map = param_bo_.map<uint32_t *>();
  for (int i = 0; i < 16; i++)
    std::cout << "params[" << i << "]=" << (uint32_t)(param_map[i])
              << std::endl;

  run = kernel_(2, instr_bo, instr_bo_words,
                param_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET, 0);
  std::cout << "Kernel Run" << std::endl;
  run.wait2();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;
  num_run_aie_++;
  std::cout << "Kernel Run Finished" << std::endl;

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
      std::to_string(matmul_mladf_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
matmul_a16w8_mladf<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  std::string txn_key =
      "gemm_mladf_" + get_instr_key(txn_fname_prefix_, Mo, Ko, N);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());
  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> matmul_a16w8_mladf<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // input --> [input, weights, qdq_c0, qdq_c1c2, kernel_params, output]

  if (input.size() != 6) {
    throw std::runtime_error(
        "MATMUL_MLADF : Incorrect number of tensors received");
  }
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);

  int size_weight = K * N * b_dtype_size_;
  int num_sv = (K * N) / ((matmul_a16w8_mladf_matrix::Nsubv) *
                          (matmul_a16w8_mladf_matrix::Ksubv));
  int size_interleaved_qdq =
      (matmul_a16w8_mladf_matrix::Nsubv * sizeof(int64_t) +
       matmul_a16w8_mladf_matrix::QDQparam_size * sizeof(int32_t)) *
      num_sv;

  size_t PARAM_BO_SIZE = (16 * sizeof(uint32_t));
  size_t B_BO_SIZE = size_weight + size_interleaved_qdq;
  size_t A_BO_SIZE = (Mo * Ko * sizeof(InT));
  size_t C_BO_SIZE = (Mo * N * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 0, 4, 0, PARAM_BO_SIZE},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, B_BO_SIZE},
      {OpArgMap::OpArgType::INPUT, 2, 0, 0, A_BO_SIZE},
      {OpArgMap::OpArgType::OUTPUT, 3, 5, 0, C_BO_SIZE}};
  return arg_map;
};

template class matmul_a16w8_mladf<uint16_t, uint8_t, uint16_t>;

} // namespace ryzenai
