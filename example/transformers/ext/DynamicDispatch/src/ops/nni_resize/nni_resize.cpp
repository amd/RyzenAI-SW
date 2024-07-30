/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <ops/nni_resize/nni_resize.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

namespace ryzenai {

static std::tuple<int, int, int>
extract_HWC(const std::vector<Tensor> &inputs) {
  int H = 0;
  int W = 0;
  int C = 0;
  if (inputs.at(0).shape.size() == 3) {
    H = inputs.at(0).shape.at(0);
    W = inputs.at(0).shape.at(1);
    C = inputs.at(0).shape.at(2);
  } else if (inputs.at(0).shape.size() == 4) {
    if (inputs.at(0).shape.at(2) == inputs.at(0).shape.at(3)) {
      H = inputs.at(0).shape.at(2);
      W = inputs.at(0).shape.at(3);
      C = inputs.at(0).shape.at(1);
    } else {
      H = inputs.at(0).shape.at(1);
      W = inputs.at(0).shape.at(2);
      C = inputs.at(0).shape.at(3);
    }
  }
  return std::make_tuple(H, W, C);
}

template <typename InT, typename OutT>
std::once_flag nni_resize<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t nni_resize<InT, OutT>::nni_resize_count = 0;

template <typename InT, typename OutT>
std::once_flag nni_resize<InT, OutT>::instr_reg_flag_;

template <typename InT, typename OutT>
void nni_resize<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename OutT>
std::tuple<int, int, int> nni_resize<InT, OutT>::map_padded_shape(int H, int W,
                                                                  int C) {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<std::tuple<int, int, int>> &supported_shapes = iter->second;
  int Ho = H;
  int Wo = W;
  int Co = C;
  int fidx = 0;
  int f_found = 0;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes[i];
    if (H == (int)get<0>(mat) && W == (int)get<1>(mat) &&
        C == (int)get<2>(mat)) {
      fidx = i;
      f_found = 1;
      break;
    }
  }
  if (f_found == 1) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<std::tuple<int, int, int>> &actual_shapes = iter->second;
    auto mat = actual_shapes[fidx];
    Ho = (int)get<0>(mat);
    Wo = (int)get<1>(mat);
    Co = (int)get<2>(mat);
  } else {
    throw std::runtime_error("Can not find the shape");
  }
  return std::make_tuple(Ho, Wo, Co);
}

template <typename InT, typename OutT>
std::string nni_resize<InT, OutT>::get_instr_key(std::string prefix, int h,
                                                 int w, int c) {
  return "nni_resize_" + prefix + "_" + std::to_string(h) + "_" +
         std::to_string(w) + "_" + std::to_string(c);
}

template <typename InT, typename OutT>
void nni_resize<InT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::tuple<int, int, int>> &supported_shapes = iter->second;
    for (int i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes[i];
      auto key = get_instr_key(mkey, get<0>(mat), get<1>(mat), get<2>(mat));
      auto param_key =
          get_instr_key(mkey, get<0>(mat), get<1>(mat), get<2>(mat)) + "_param";
      instructions.push_back(std::make_pair(key, false));
      layer_params.push_back(std::make_pair(param_key, false));
    }
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}

template <typename InT, typename OutT>
nni_resize<InT, OutT>::nni_resize(const std::string &a_dtype,
                                  const std::string &c_dtype, bool load_xrt,
                                  const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_c_header = {{"uint16", "acc16"}};

  default_shapes_["nni_resize_4x4_a16acc16"] =
      std::vector<std::tuple<int, int, int>>();
  default_shapes_["nni_resize_4x4_a16acc16"].push_back(
      std::make_tuple(8, 8, 1280));
  default_shapes_["nni_resize_4x4_a16acc16"].push_back(
      std::make_tuple(16, 16, 1280));
  default_shapes_["nni_resize_4x4_a16acc16"].push_back(
      std::make_tuple(32, 32, 640));

  raw_shapes_["nni_resize_4x4_a16acc16"] =
      std::vector<std::tuple<int, int, int>>();
  raw_shapes_["nni_resize_4x4_a16acc16"].push_back(std::make_tuple(8, 8, 1280));
  raw_shapes_["nni_resize_4x4_a16acc16"].push_back(
      std::make_tuple(16, 16, 1280));
  raw_shapes_["nni_resize_4x4_a16acc16"].push_back(
      std::make_tuple(32, 32, 640));

  a_dtype_ = a_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  c_dtype_size_ = sizeof(OutT);
  nni_resize_id_ = nni_resize_count++;

  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + ryzenai::PSR4x4_A16W8_QDQ_XCLBIN_PATH;

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

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "nni_resize_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_c_header.at(c_dtype_);

    param_fname_prefix_ = "nni_resize_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_c_header.at(c_dtype_);
  } else {
    throw std::runtime_error("NNI_RESIZE IPU Wrapper only support 4x4 design.");
  }

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header =
        "nni_resize_id H W C kernel_h kernel_w kernel_c Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[NNI_RESIZE] ID: " + std::to_string(nni_resize_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, c_dtype): (" +
                    a_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename OutT>
void nni_resize<InT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params) {
  RYZENAI_LOG_TRACE("NNI_resize initialize_const_params(ptr) ...");

  RYZENAI_LOG_TRACE("NNI_resize initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename OutT>
void nni_resize<InT, OutT>::execute(const std::vector<Tensor> &input,
                                    std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error(
        "NNI_RESIZE IPU Wrapper expect to have one input.");
  }
  const int a_idx = 0;
  const int c_idx = 0;
  // The first data is a
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  a_shape_[0] = input.at(a_idx).shape.at(0);
  a_shape_[1] = input.at(a_idx).shape.at(1);
  a_shape_[2] = input.at(a_idx).shape.at(2);

  auto [H, W, C] = map_padded_shape(a_shape_[0], a_shape_[1], a_shape_[2]);

  c_shape_[0] = output.at(c_idx).shape.at(0);
  c_shape_[1] = output.at(c_idx).shape.at(1);
  c_shape_[2] = output.at(c_idx).shape.at(2);

  kernel_x_shape_[0] = H;
  kernel_x_shape_[1] = W;
  kernel_x_shape_[2] = C;

  const auto a_bo_size_in_bytes = kernel_x_shape_[0] * kernel_x_shape_[1] *
                                  kernel_x_shape_[2] * a_dtype_size_;
  const auto c_bo_size_in_bytes = kernel_x_shape_[0] * kernel_x_shape_[1] *
                                  kernel_x_shape_[2] * 4 * c_dtype_size_;

  /* Create input/output BOs */
  const int A_BO_SIZE = a_bo_size_in_bytes;
  const int C_BO_SIZE = c_bo_size_in_bytes;
  RYZENAI_LOG_TRACE("NNI_RESIZE: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  int a_size = a_shape_[0] * a_shape_[1] * a_shape_[2] * sizeof(InT);
  RYZENAI_LOG_TRACE("NNI_RESIZE: a_size:" + std::to_string(a_size));
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
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, H, W, C);
  auto param_bo_key = get_instr_key(param_fname_prefix_, H, W, C) + "_param";
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
                a_bo_.address() + DDR_AIE_ADDR_OFFSET, 0,
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
         c_shape_[0] * c_shape_[1] * c_shape_[2] * sizeof(OutT));
  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(nni_resize_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(a_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename OutT>
const std::vector<uint8_t> nni_resize<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [H, W, C] = extract_HWC(input);
  auto [Ho, Wo, Co] = map_padded_shape(H, W, C);
  std::string txn_key = get_instr_key(txn_fname_prefix_, Ho, Wo, Co);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename OutT>
const std::vector<uint8_t> nni_resize<InT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [H, W, C] = extract_HWC(input);
  auto [Ho, Wo, Co] = map_padded_shape(H, W, C);
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      get_instr_key(param_fname_prefix_, Ho, Wo, Co) + "_param";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  Transaction &txn = Transaction::getInstance();
  std::string param_string = txn.get_txn_str(param_key);
  std::istringstream params_stream(param_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(params_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename OutT>
std::vector<OpArgMap> nni_resize<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [H, W, C] = extract_HWC(input);

  auto [Ho, Wo, Co] = map_padded_shape(H, W, C);
  auto out_shape = Ho * Wo * Co * 4;
  size_t const_params_bo_size = 16; // Dummy Buffer
  size_t input_bo_size = (Ho * Wo * Co * sizeof(InT));
  size_t output_bo_size = (out_shape * sizeof(OutT));
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0,
       const_params_bo_size}, // Dummy allocation
      {OpArgMap::OpArgType::OUTPUT, 0, 2, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}}; // TODO: This will be removed after the recent
                            // changes from main are merged into computex

  return arg_map;
}

template class nni_resize<uint16_t, uint16_t>;

} // namespace ryzenai
