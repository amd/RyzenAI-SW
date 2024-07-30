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
#include <xrt_context/xrt_context.hpp>

#include "utils/txn_container.hpp"
#include <ops/conv2matmul/conv2matmul.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/txn_container.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {

static std::tuple<int, int, int> extract_MKN(const std::vector<Tensor> &inputs,
                                             string input_format) {
  int M;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
  } else if (inputs.at(0).shape.size() == 3) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 4) { // has batch_dim
    if (input_format == "NHWC") {              // NHWC
      M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1) *
          inputs.at(0).shape.at(2);
    } else { // NCHW
      M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(2) *
          inputs.at(0).shape.at(3);
    }
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }

  int K = inputs.at(1).shape.at(1);
  int N = inputs.at(1).shape.at(0);

  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<int, int, int>
conv2matmul<InT, WtT, OutT>::map_padded_shape(int M, int K, int N) {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<matrix_shapes> &supported_shapes = iter->second;
  int Mo = M;
  int Ko = K;
  int No = N;
  int fidx = 0;
  int f_found = 0;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    if (M == mat.M && K == mat.K && N == mat.N) {
      fidx = i;
      f_found = 1;
      break;
    }
  }

  if (f_found == 1) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<matrix_shapes> &actual_shapes = iter->second;
    auto mat = actual_shapes.at(fidx);
    Mo = (int)mat.M;
    Ko = (int)mat.K;
    No = (int)mat.N;
  } else {
    throw std::runtime_error("Can not find the shape");
  }
  // std::cout << Mo << ' ' << No << std::endl;
  return std::make_tuple(Mo, Ko, No);
}

/*
 * conv2matmul is an experimental class to offload int8_t * int8_t matrix
 * multiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */

/* Utility function to set the kernel shape based on the weights dimensions
 * Pick kernel shape using weight matrix size
 * Select OPT shapes when a_type is int8
 * Select Llamav2 shapes when a_type is int16
 * Need to fix this to pick shapes independent of the datatype*/
template <typename InT, typename WtT, typename OutT>
void conv2matmul<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  kernel_x_shape_[1] = w_shape_[0];
  kernel_y_shape_[0] = w_shape_[0];
  kernel_y_shape_[1] = w_shape_[1];
  kernel_z_shape_[1] = w_shape_[1];
}

/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void conv2matmul<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key =
        "conv2gemm_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    auto param_key = "conv2gemm_" +
                     get_instr_key(param_fname_prefix_, mat.M, mat.K, mat.N) +
                     "_param";
    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  instr_reg_.add_layer_params(layer_params);
}
template <typename InT, typename WtT, typename OutT>
std::string conv2matmul<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                       int m, int k, int n) {
  return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
         std::to_string(n);
}

template <typename InT, typename WtT, typename OutT>
conv2matmul<InT, WtT, OutT>::conv2matmul(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"uint16", "a16"}};

  txnbin_b_header = {{"uint8", "w8"}};

  txnbin_acc_header = {{"uint16", "acc16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["conv2gemm_4x2_a16w8acc16"] = std::vector<matrix_shapes>{};

  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(64, 2560, 1280);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 640, 1280);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 1280, 64);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 1920, 1280);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 2560, 1280);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 320, 640);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 640, 64);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 960, 640);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 1280, 640);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 1920, 640);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(4096, 640, 320);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(4096, 960, 320);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(64, 1280, 64);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(128, 1040, 64);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(128, 1040, 64);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(4096, 320, 64);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(64, 1280, 320);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(64, 1280, 640);
  default_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(64, 1280, 1280);

  default_shapes_["conv2gemm_4x4_a16w8acc16"] = std::vector<matrix_shapes>{};

  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(64, 2560, 1280);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 640, 1280);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 1280, 64);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 1920, 1280);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 2560, 1280);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 320, 640);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 640, 64);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 960, 640);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 1280, 640);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 1920, 640);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(4096, 640, 320);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(4096, 960, 320);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 64);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(96, 1024, 64);
  //  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(128, 1040, 64);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(4096, 320, 64);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 320);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 640);
  default_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 1280);

  // raw shape is the actual shape from ONNX
  raw_shapes_["conv2gemm_4x2_a16w8acc16"] = std::vector<matrix_shapes>{};

  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(64, 2560, 1280);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 640, 1280);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 1280, 64);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 1920, 1280);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(256, 2560, 1280);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 320, 640);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 640, 64);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 960, 640);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 1280, 640);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1024, 1920, 640);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(4096, 640, 320);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(4096, 960, 320);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(64, 1280, 64);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(77, 1024, 64);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(128, 1024, 64);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(4096, 320, 64);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1, 1280, 320);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1, 1280, 640);
  raw_shapes_["conv2gemm_4x2_a16w8acc16"].emplace_back(1, 1280, 1280);

  raw_shapes_["conv2gemm_4x4_a16w8acc16"] = std::vector<matrix_shapes>{};

  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(64, 2560, 1280);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 640, 1280);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 1280, 64);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 1920, 1280);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(256, 2560, 1280);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 320, 640);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 640, 64);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 960, 640);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 1280, 640);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1024, 1920, 640);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(4096, 640, 320);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(4096, 960, 320);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 64);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(77, 1024, 64);
  //  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(128, 1024, 64);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(4096, 320, 64);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1, 1280, 320);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1, 1280, 640);
  raw_shapes_["conv2gemm_4x4_a16w8acc16"].emplace_back(1, 1280, 1280);

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  conv2matmul_id_ = conv2matmul_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + ryzenai::PSR_A16W8_QDQ_XCLBIN_PATH;

  RYZENAI_LOG_TRACE(OpsFusion::dod_format("xclbin fname : {}", XCLBIN_FNAME));

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

  txn_fname_prefix_ = "conv2gemm_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  param_fname_prefix_ = "conv2gemm_4x2_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "conv2gemm_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
    param_fname_prefix_ = "conv2gemm_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_acc_header.at(c_dtype_);

    if (attr.count("input_shape") &&
        attr.at("input_shape").type() == typeid(std::vector<int>)) {
      const auto &input_shape_vector =
          std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

      if (input_shape_vector.size() == 4) {
        inputShape_[0] = input_shape_vector[0];
        inputShape_[1] = input_shape_vector[1];
        inputShape_[2] = input_shape_vector[2];
        inputShape_[3] = input_shape_vector[3];
      } else {
        std::cout
            << "Input Shape attribute does not have the expected number of "
               "elements.Number of passed : input_shape_vector.size(), "
               "Expected:4"
            << std::endl;
      }
      RYZENAI_LOG_TRACE(
          "iConv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
          std::to_string(input_shape_vector[1]) + ", " +
          std::to_string(input_shape_vector[2]) + ", " +
          std::to_string(input_shape_vector[3]));
    } else {
      std::cout << "Input Shape attribute not found or not of correct type."
                << std::endl;
    }
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("txn_fname_prefix : {}", txn_fname_prefix_));
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("param_fname_prefix : {}", param_fname_prefix_));

  if (attr.count("input_format") &&
      attr.at("input_format").type() == typeid(std::vector<string>)) {
    const auto &input_format_vector =
        std::any_cast<const std::vector<string> &>(attr.at("input_format"));

    if (input_format_vector.size() == 1) {
      input_format_ = input_format_vector[0];
    } else {
      std::cout
          << "Input Format attribute does not have the expected number of "
             "elements.Number of passed : input_format_vector.size(), "
             "Expected:4"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("iConv: InputFormat: " + input_format_);
  } else {
    std::cout << "Input Format attribute not found or not of correct type."
              << std::endl;
  }

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
        "conv2matmul_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[CONV2GEMM] ID: " + std::to_string(conv2matmul_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul<InT, WtT, OutT>::set_params(const std::string &model_name,
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

  auto [M, K, N] =
      map_padded_shape(input_shape.at(0), input_shape.at(1), input_shape.at(2));
  KERNEL_M_MAX = M;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv2Matmul initialize_const_params(ptr) ...");

  DOD_THROW_IF(
      (const_params.size() != 3),
      OpsFusion::dod_format("Unsupported const spec for Conv2Matmul\n") +
          OpsFusion::dod_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  auto K_orig = const_params.at(0).shape.at(1);
  w_shape_[0] = const_params.at(0).shape.at(1); // K
  w_shape_[1] = const_params.at(0).shape.at(0); //

  auto weights = (WtT *)const_params.at(0).data;

  auto qdq = (int64_t *)const_params.at(1).data;
  auto qdq_params = (int32_t *)const_params.at(2).data;

  auto Ksubv = matmul_matrix::Ksubv_PSR;
  auto Msubv = matmul_matrix::Msubv_PSR;
  auto Nsubv = matmul_matrix::Nsubv_PSR;
  if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
    set_kernel_shapes();
  } else {
    if (w_shape_[0] == 1024 && w_shape_[1] == 64) { // PSR padding K case
      w_shape_[0] = 1040;
    }
    set_kernel_shapes();
  }
  std::vector<WtT> buf(w_shape_[0] * w_shape_[1]);
  if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
    int M = inputShape_[2] * inputShape_[3];

    SUBV_T key = {M, w_shape_[0], w_shape_[1]};
    auto subv_mode = search_subv_mode(key);

    format_wgt_trans<WtT>(weights, buf.data(), subv_mode, w_shape_[0],
                          w_shape_[1], K_orig);
    SUBV_T subv = get_subv(subv_mode);
    Msubv = subv[0];
    Ksubv = subv[1];
    Nsubv = subv[2];
  } else {

    if (w_shape_[1] > 640) {
      Nsubv = matmul_matrix::Nsubv_PSR_LARGE;
    }

    if (w_shape_[1] > 640) {
      matmul_matrix::WgtMatrix<WtT, matmul_matrix::Ksubv_PSR,
                               matmul_matrix::Nsubv_PSR_LARGE>
          W(w_shape_[0], w_shape_[1], buf.data());
      for (int r = 0; r < K_orig; ++r) {
        for (int c = 0; c < w_shape_[1]; ++c) {
          W.at(r, c) = weights[(c * K_orig) + r]; // transpose
        }
      }
    } else {
      matmul_matrix::WgtMatrix<WtT, matmul_matrix::Ksubv_PSR,
                               matmul_matrix::Nsubv_PSR>
          W(w_shape_[0], w_shape_[1], buf.data());
      for (int r = 0; r < K_orig; ++r) {
        for (int c = 0; c < w_shape_[1]; ++c) {
          W.at(r, c) = weights[(c * K_orig) + r]; // transpose
        }
      }
    }
  }

  // padding Msubv and Nsubv
  qdq_params[qdq_Mv_idx] = Msubv;
  qdq_params[qdq_Nv_idx] = Nsubv;

  auto total_size = Ksubv * Nsubv;
  auto qdq_size = Nsubv * sizeof(int64_t);
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  //// WGT + Bias(all zeros)
  { // This section of the code interleaves bias with weights Nsubv of bias
    // with every K x N
    int write_offset = 0;
    for (int N_shard = 0; N_shard < (w_shape_[1]) / (Nsubv); N_shard++) {
      for (int K_shard = 0; K_shard < (w_shape_[0]) / (Ksubv); K_shard++) {
        memcpy((void *)(static_cast<uint8_t *>(dest) + write_offset),
               (void *)&buf[(N_shard * w_shape_[0] * Nsubv) +
                            (K_shard * total_size)],
               (total_size));
        write_offset += total_size;
        memcpy((void *)(static_cast<uint8_t *>(dest) + write_offset),
               (void *)&qdq[N_shard * Nsubv], qdq_size);
        write_offset += qdq_size;
      }
    }
    memcpy((void *)(static_cast<WtT *>(dest) + write_offset),
           (void *)qdq_params, qdq_params_size);
  }

  RYZENAI_LOG_TRACE("Conv2Matmul initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv2Matmul initialize_const_params ...");

  DOD_THROW_IF(
      (const_params.size() != 3) || (const_params.at(0).shape.size() != 2),
      OpsFusion::dod_format("Unsupported const spec for Cpnv2Matmul\n") +
          OpsFusion::dod_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  w_shape_[0] = const_params.at(0).shape.at(1); // K
  w_shape_[1] = const_params.at(0).shape.at(0); // N
  int Ksubv;
  if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
    set_kernel_shapes();
    int M = inputShape_[2] * inputShape_[3];

    SUBV_T key = {M, w_shape_[0], w_shape_[1]};
    auto subv_mode = search_subv_mode(key);
    SUBV_T subv = get_subv(subv_mode);
    Ksubv = subv[1];
  } else {
    if (w_shape_[0] == 1024 && w_shape_[1] == 64) { // PSR padding K case
      w_shape_[0] = 1040;
    }
    set_kernel_shapes();
    Ksubv = matmul_matrix::Ksubv_PSR;
  }
  // qdqc
  int size_interleaved_qdq =
      w_shape_[0] * w_shape_[1] / Ksubv * sizeof(int64_t);

  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t);

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input/output BOs */
  const int B_BO_SIZE =
      (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_ +
       size_interleaved_qdq);
  const int A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const int C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);

  RYZENAI_LOG_TRACE("CONV2GEMM: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  initialize_const_params(b_bo_map, const_params);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += b_format_stop - b_format_start;
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = b_copy_stop - b_copy_start;
  b_sync_time_ = b_sync_stop - b_sync_start;
  RYZENAI_LOG_TRACE("Conv2Matmul initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                          std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Conv2Matmul execute ...");

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  int64_t exec_start = GET_ELAPSED_TIME_NS();
  if (input.at(0).shape.size() == 4) {
    a_shape_[0] = input.at(0).shape.at(0) * input.at(0).shape.at(1) *
                  input.at(0).shape.at(2);
    a_shape_[1] = input.at(0).shape.at(3);
  } else if (input.at(0).shape.size() == 3) {
    a_shape_[0] = input.at(0).shape.at(0) * input.at(0).shape.at(1);
    a_shape_[1] = input.at(0).shape.at(2);
  } else if (input.at(0).shape.size() == 2) {
    a_shape_[0] = input.at(0).shape.at(0);
    a_shape_[1] = input.at(0).shape.at(1);
  } else {
    throw std::runtime_error("Conv2Matmul : Invalid shape received for input");
  }

  RYZENAI_LOG_TRACE("CONV2GEMM: a_shape_[0]:" + std::to_string(a_shape_[0]) +
                    " a_shape_[1]:" + std::to_string(a_shape_[1]) +
                    " w_shape_[1]:" + std::to_string(w_shape_[1]));

  c_shape_[0] = a_shape_[0];
  c_shape_[1] = w_shape_[1];
  auto aie_out = (OutT *)output.at(0).data;
  auto a = (InT *)input.at(0).data;

  auto [M, K, N] = map_padded_shape(a_shape_[0], a_shape_[1], w_shape_[1]);
  kernel_x_rows = M;

  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  if (K == a_shape_[1]) {
    memcpy((void *)a_bo_map, (void *)a,
           (a_shape_[0] * a_shape_[1] * a_dtype_size_));
  } else {
    for (int i = 0; i < a_shape_[0]; i++) {
      memcpy((void *)&a_bo_map[i * K], (void *)&a[i * a_shape_[1]],
             (a_shape_[1] * a_dtype_size_));
    }
  }
  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // INIT with zeros
  auto instr_bo_key = "conv2gemm_" + txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);
  auto param_bo_key = "conv2gemm_" + param_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]) + "_param";
  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  const xrt::bo &param_bo = instr_reg_.get_param_bo(param_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;
  // launch the GEMM kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for GEMM that supports transaction binary flow
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                param_bo.address() + DDR_AIE_ADDR_OFFSET, 0);
  run.wait2();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  OutT *c_bo_map = c_bo_.map<OutT *>();
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;
  run_aie_time_ += run_aie_stop - run_aie_start;
  memcpy((void *)aie_out, (void *)c_bo_map,
         (c_shape_[0] * c_shape_[1] * c_dtype_size_));
  int64_t exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(conv2matmul_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
  RYZENAI_LOG_TRACE("Conv2Matmul execute ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> conv2matmul<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K, N] = extract_MKN(input, input_format_);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  std::string txn_key =
      "conv2gemm_" + get_instr_key(txn_fname_prefix_, Mo, Ko, No);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());
  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> conv2matmul<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto [M, K, N] = extract_MKN(input, input_format_);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      "conv2gemm_" + get_instr_key(param_fname_prefix_, Mo, Ko, No) + "_param";
  Transaction &txn = Transaction::getInstance();
  std::string param_string = txn.get_txn_str(param_key);
  std::istringstream params_stream(param_string, std::ios::binary);
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Super kernel params name : {}", param_key));
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(params_stream)),
                            std::istreambuf_iterator<char>());

  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> conv2matmul<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  auto [M, K, N] = extract_MKN(input, input_format_);

  auto [Mo, Ko, No] = map_padded_shape(M, K, N);

  int Ksubv;
  if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
    SUBV_T key = {M, K, N};
    auto subv_mode = search_subv_mode(key);
    SUBV_T subv = get_subv(subv_mode);
    Ksubv = subv[1];
  } else {
    Ksubv = matmul_matrix::Ksubv_PSR;
  }

  // qdqc
  int size_interleaved_qdq = Ko * No / Ksubv * sizeof(int64_t);
  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t);

  size_t const_params_bo_size =
      (Ko * No * b_dtype_size_) + size_interleaved_qdq;
  size_t input_bo_size = (Mo * Ko * a_dtype_size_);
  size_t output_bo_size = (Mo * No * c_dtype_size_);
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 4, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Conv2Matmul Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag conv2matmul<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t conv2matmul<InT, WtT, OutT>::conv2matmul_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag conv2matmul<InT, WtT, OutT>::instr_reg_flag_;

template class conv2matmul<uint16_t, uint8_t, uint16_t>;

} // namespace ryzenai
