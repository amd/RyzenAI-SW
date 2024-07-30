/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <any>
#include <iostream>
#include <map>
#include <numeric>
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

#include <ops/mladfelwadd/mladfelwadd.hpp>
#include <ops/op_interface.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>
#include <utils/txn_container.hpp>
#include <xrt_context/xrt_context.hpp>
// #include <utils/utils.h>

// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {
static void process_shape(std::vector<size_t> &input_shape) {
  if (!input_shape.empty() && input_shape[0] == 1) {
    input_shape.erase(input_shape.begin());
  }
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_add<InT, WtT, OutT>::determine_weight_repeat() {
  wgt_repeat = true;
  if (kernel_y_shape_.size() == 1 ||
      std::all_of(kernel_y_shape_.begin() + 1, kernel_y_shape_.end(),
                  [](size_t value) { return value != 1; })) {
    wgt_repeat = false;
  }
}

template <typename InT, typename WtT, typename OutT>
std::string ml_adf_elw_add<InT, WtT, OutT>::get_instr_key(
    std::string prefix, const std::vector<size_t> &dimensions) {
  std::string key = "mladfelwadd_" + prefix;
  for (const int &dim : dimensions) {
    key += "_" + std::to_string(dim);
  }
  return key;
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_add<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::vector<size_t>> &supported_shapes = iter->second;
    for (int i = 0; i < supported_shapes.size(); i++) {
      auto shape = supported_shapes[i];
      auto key = get_instr_key(mkey, shape);
      instructions.push_back(std::make_pair(key, false));
    }
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
ml_adf_elw_add<InT, WtT, OutT>::ml_adf_elw_add(const std::string &a_dtype,
                                               const std::string &b_dtype,
                                               const std::string &c_dtype,
                                               bool load_xrt) {
  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_b_header = {{"uint16", "w16"}};
  txnbin_c_header = {{"uint16", "acc16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["mladfelwadd_a16w16acc16"] =
      std::vector<std::vector<size_t>>();

  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {128, 512, 512, 128, 512, 512});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {256, 256, 256, 256, 256, 256});

  default_shapes_["mladfelwadd_a16w16acc16"].push_back({4096, 512, 512});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {128, 256, 256, 128, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {128, 512, 512, 128, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {256, 128, 128, 256, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {256, 256, 256, 256, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {256, 512, 512, 256, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {512, 64, 64, 512, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {512, 128, 128, 512, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {512, 256, 256, 512, 1, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back({512, 4096, 512, 1});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {512, 128, 128, 512, 128, 128});
  default_shapes_["mladfelwadd_a16w16acc16"].push_back(
      {512, 64, 64, 512, 64, 64});

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  ml_adf_elw_add_id_ = ml_adf_elw_add_count++;

  std::string XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                             ryzenai::MLADF_4x2_ELWADD_A16W16_QDQ_XCLBIN_PATH;

  txn_fname_prefix_ = "mladfelwadd_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_c_header.at(c_dtype_);

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
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = "elw_add_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[ADD] ID: " + std::to_string(ml_adf_elw_add_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_add<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "PST" || model_name == "PSS") {
    XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                   ryzenai::MLADF_4x2_ELWADD_A16W16_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  kernel_x_shape_ = input_shape;
  kernel_z_shape_ = input_shape;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_add<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("MLADFElwadd initialize_const_params(ptr) ...");
  // Assume non-broacasting weight is non-const in PSS and PST model.
  // If weight is non-const, just return.
  if (const_params.size() == 1)
    return;
  DOD_THROW_IF(
      (const_params.size() != 2),
      OpsFusion::dod_format("Unsupported const spec for MLADFElwadd\n") +
          OpsFusion::dod_format("(Details : #const params == 2 or 1({})",
                                const_params.size()));
  kernel_y_shape_ = const_params.at(0).shape;
  process_shape(kernel_y_shape_);
  int wgt_sz = std::accumulate(kernel_y_shape_.begin(), kernel_y_shape_.end(),
                               size_t{1}, std::multiplies{}) *
               sizeof(WtT);
  determine_weight_repeat();
  if (wgt_repeat) {
    auto expand_vector = [](const WtT *original, size_t M) {
      std::vector<WtT> expanded(2 * M);
      for (size_t i = 0; i < M; ++i) {
        expanded[2 * i] = original[i];
        expanded[2 * i + 1] = original[i];
      }
      return expanded;
    };

    std::vector<WtT> w_expand = expand_vector(
        static_cast<const WtT *>(const_params.at(0).data), kernel_y_shape_[0]);
    memcpy((char *)dest, (char *)w_expand.data(), wgt_sz * 2);
  } else {
    memcpy((char *)dest, (char *)const_params.at(0).data, wgt_sz);
  }

  RYZENAI_LOG_TRACE("MLADFElwadd initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_add<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  DOD_THROW_IF(
      (const_params.size() != 2),
      OpsFusion::dod_format("Unsupported const spec for MLADFElwadd\n") +
          OpsFusion::dod_format("(Details : #const params == 2 ({})",
                                const_params.size()));

  kernel_y_shape_ = const_params.at(0).shape;

  bool wgt_repeat = true;
  // no_broadcast or outer_broadcast don't need repeat weight
  if ((kernel_y_shape_.size() == 1 &&
       kernel_y_shape_[0] == kernel_x_shape_[0]) ||
      std::all_of(kernel_y_shape_.begin() + 1, kernel_y_shape_.end(),
                  [](size_t value) { return value != 1; })) {
    wgt_repeat = false;
  }

  int B_BO_SIZE =
      std::accumulate(kernel_y_shape_.begin(), kernel_y_shape_.end(), size_t{1},
                      std::multiplies{}) *
      b_dtype_size_;
  if (wgt_repeat) {
    B_BO_SIZE *= 2;
  }

  const int A_BO_SIZE =
      std::accumulate(kernel_x_shape_.begin(), kernel_x_shape_.end(), size_t{1},
                      std::multiplies{}) *
      a_dtype_size_;
  const int C_BO_SIZE =
      std::accumulate(kernel_z_shape_.begin(), kernel_z_shape_.end(), size_t{1},
                      std::multiplies{}) *
      c_dtype_size_;

  const int P_BO_SIZE = const_params.at(1).shape[0];

  RYZENAI_LOG_TRACE("MLADFElwadd: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE) +
                    " P_BO_SIZE:" + std::to_string(P_BO_SIZE));

  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  p_bo_ = xrt::bo(xrt_ctx_->get_device(), P_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  b_copy_time_ = 0;
  b_sync_time_ = 0;
  p_copy_time_ = 0;
  p_sync_time_ = 0;
  // b_bo copy
  int8_t *b_bo_map = b_bo_.map<int8_t *>();
  int64_t b_copy_start = GET_ELAPSED_TIME_NS();
  initialize_const_params(b_bo_map, const_params);
  int64_t b_copy_stop = GET_ELAPSED_TIME_NS();
  // b_bo sync
  int64_t b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t b_sync_stop = GET_ELAPSED_TIME_NS();

  int8_t *p_bo_map = p_bo_.map<int8_t *>();
  // p_bo copy
  int64_t p_copy_start = GET_ELAPSED_TIME_NS();
  memcpy((void *)p_bo_map, (void *)const_params.at(1).data, P_BO_SIZE);
  int64_t p_copy_stop = GET_ELAPSED_TIME_NS();
  // p_bo sync
  int64_t p_sync_start = GET_ELAPSED_TIME_NS();
  p_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t p_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = b_copy_stop - b_copy_start;
  b_sync_time_ = b_sync_stop - b_sync_start;

  p_copy_time_ = p_copy_stop - p_copy_start;
  p_sync_time_ = p_sync_stop - p_sync_start;

  RYZENAI_LOG_TRACE("MLADFElwadd initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_add<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                             std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("MLADFElwadd execute ...");

  const int a_idx = 0;
  InT *a = (InT *)input.at(0).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;

  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  a_shape_ = input.at(0).shape;
  c_shape_ = a_shape_;

  int a_size = std::accumulate(a_shape_.begin(), a_shape_.end(), size_t{1},
                               std::multiplies{}) *
               sizeof(InT);
  // a_bo copy
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  // a_size: copy original shape of data from input tensor into bo
  memcpy((void *)a_bo_map, (void *)a, a_size);
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // prepare inst_bo and param_bo
  std::vector<size_t> merged_kernel_shape;
  merged_kernel_shape.reserve(kernel_x_shape_.size() + kernel_y_shape_.size());
  merged_kernel_shape.insert(merged_kernel_shape.end(), kernel_x_shape_.begin(),
                             kernel_x_shape_.end());
  merged_kernel_shape.insert(merged_kernel_shape.end(), kernel_y_shape_.begin(),
                             kernel_y_shape_.end());

  auto instr_bo_key = get_instr_key(txn_fname_prefix_, merged_kernel_shape);
  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  ;
  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                p_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET, 0);
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
  auto c_size = std::accumulate(c_shape_.begin(), c_shape_.end(), size_t{1},
                                std::multiplies{}) *
                c_dtype_size_;
  memcpy((void *)aie_out, (void *)c_bo_map, c_size);
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(ml_adf_elw_add_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> ml_adf_elw_add<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // maybe need to preprocess shape to remove batch dimension
  auto a_shape = input.at(0).shape;
  auto wgt_shape = input.at(1).shape;
  // assgin kernel shape for usage in initialize_const_params
  kernel_x_shape_ = a_shape;
  kernel_y_shape_ = wgt_shape;
  process_shape(a_shape);
  process_shape(wgt_shape);

  std::vector<size_t> merged_kernel_shape;
  merged_kernel_shape.reserve(a_shape.size() + wgt_shape.size());
  merged_kernel_shape.insert(merged_kernel_shape.end(), a_shape.begin(),
                             a_shape.end());
  merged_kernel_shape.insert(merged_kernel_shape.end(), wgt_shape.begin(),
                             wgt_shape.end());

  std::string txn_key = get_instr_key(txn_fname_prefix_, merged_kernel_shape);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
ml_adf_elw_add<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {

  auto const_QDQ_param = input.at(2);

  DOD_THROW_IF(
      (const_QDQ_param.dtype != "int8"),
      OpsFusion::dod_format("Invalid dtype: expected int8 for MLADFElwadd\n") +
          OpsFusion::dod_format("(Details : type is ({})",
                                const_QDQ_param.dtype));

  DOD_THROW_IF(
      (const_QDQ_param.shape.size() != 1 || const_QDQ_param.shape[0] != 24),
      OpsFusion::dod_format(
          "Invalid shape: expected shape[0] to be 24 for MLADFElwadd\n"));

  size_t total_bytes = const_QDQ_param.shape[0] * sizeof(int8_t);
  std::vector<uint8_t> data(total_bytes);

  std::memcpy(data.data(), const_QDQ_param.data, total_bytes);

  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> ml_adf_elw_add<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  kernel_x_shape_ = input.at(0).shape;
  kernel_y_shape_ = input.at(1).shape;

  process_shape(kernel_x_shape_);
  process_shape(kernel_y_shape_);

  size_t input_1_bo_size =
      std::accumulate(kernel_x_shape_.begin(), kernel_x_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(InT);
  size_t const_bo_size =
      std::accumulate(kernel_y_shape_.begin(), kernel_y_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(WtT);
  determine_weight_repeat();
  if (wgt_repeat)
    const_bo_size *= 2;
  size_t output_bo_size =
      std::accumulate(kernel_x_shape_.begin(), kernel_x_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(OutT);
  // QDQ param size
  size_t param_size = input.at(2).shape[0];
  std::vector<OpArgMap> arg_map;
  if (kernel_x_shape_ == kernel_y_shape_) {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 2, 0, 0, input_1_bo_size},
        {OpArgMap::OpArgType::INPUT, 1, 1, 0, const_bo_size},
        {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 0, 2, 0, param_size},
        {OpArgMap::OpArgType::OUTPUT, 3, 3, 0, output_bo_size}};
  } else {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 2, 0, 0, input_1_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_bo_size},
        {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 0, 2, 0, param_size},
        {OpArgMap::OpArgType::OUTPUT, 3, 3, 0, output_bo_size}};
  }

  return arg_map;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag ml_adf_elw_add<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t ml_adf_elw_add<InT, WtT, OutT>::ml_adf_elw_add_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag ml_adf_elw_add<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_add<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template class ml_adf_elw_add<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
