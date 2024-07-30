/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <any>
#include <array>
#include <fstream>
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

#include <utils/instruction_registry.hpp>
#include <utils/txn_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/iconv/iconv.hpp>
#include <ops/op_interface.hpp>
#include <txn_helper/txn_helper.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>
// AIE Driver header
#include "xaiengine.h"

#include <ops/ops_common/iconv_matrix.hpp>

using namespace iconv_matrix;

namespace ryzenai {

static std::tuple<size_t, size_t, size_t> extract_CYX(const Tensor &tensor,
                                                      string input_format) {
  size_t C, Y, X;
  if (tensor.shape.size() == 3) { // HWC
    C = tensor.shape.at(2);
    Y = static_cast<size_t>(sqrt(tensor.shape.at(0) * tensor.shape.at(1)));
    X = Y;
  } else if (tensor.shape.size() == 4) {
    if (input_format == "NHWC") { // NHWC
      Y = tensor.shape.at(1);
      X = tensor.shape.at(2);
      C = tensor.shape.at(3);
    } else { // NCHW
      C = tensor.shape.at(1);
      Y = tensor.shape.at(2);
      X = tensor.shape.at(3);
    }
  } else {
    throw std::runtime_error("iCONV : Invalid shape received for Matrix");
  }

  return std::make_tuple(C, Y, X);
}

template <typename InT, typename WtT, typename OutT>
std::vector<size_t>
iconv<InT, WtT, OutT>::map_padded_shape(std::vector<size_t> &in) {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<std::vector<size_t>> &supported_shapes = iter->second;
  int fidx = 0;
  int f_found = 0;
  std::vector<size_t> out_mat;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    int sum_diff = 0;
    int diff = 0;
    for (int j = 0; j < in.size(); j++) {
      diff = mat[j] - in[j];
      diff = diff > 0 ? diff : (-diff);
      sum_diff += diff;
    }
    if (sum_diff == 0) {
      fidx = i;
      f_found = 1;
      break;
    }
  }
  if (f_found == 1) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<std::vector<size_t>> &actual_shapes = iter->second;
    out_mat = actual_shapes.at(fidx);
  } else {
    throw std::runtime_error("Can not find the shape");
  }
  // std::cout << Mo << ' ' << No << std::endl;
  return out_mat;
}

template <typename InT, typename WtT, typename OutT>
std::string iconv<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                 std::vector<size_t> &mat) {
  std::string out_str = "iconv_" + prefix;
  for (int i = 0; i < mat.size(); i++) {
    out_str += "_" + std::to_string(mat[i]);
  }
  return out_str;
}

template <typename InT, typename WtT, typename OutT>
void iconv<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::vector<size_t>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
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
iconv<InT, WtT, OutT>::iconv(const std::string &a_dtype,
                             const std::string &b_dtype,
                             const std::string &c_dtype, bool load_xrt,
                             const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"uint16", "a16"}};

  txnbin_b_header = {{"uint8", "w8"}};

  txnbin_acc_header = {{"uint16", "acc16"}};
  // ci, yi, xi, co, yo, xo, ky, kx
  default_shapes_["iconv_4x2_a16w8acc16"] = std::vector<std::vector<size_t>>();
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{128, 56, 56, 256, 28, 28, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{256, 28, 28, 512, 14, 14, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{512, 14, 14, 1024, 8, 7, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{128, 56, 56, 128, 56, 56, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{256, 28, 28, 256, 32, 28, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{512, 14, 14, 512, 16, 14, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1024, 7, 7, 1024, 8, 7, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{8, 230, 116, 128, 56, 56, 7, 4});

  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 8, 8, 1280, 8, 8, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 8, 8, 1280, 8, 8, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 1280, 32, 32, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{960, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 640, 64, 64, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{960, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 32, 32, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 16, 16, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 8, 8, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 32, 64, 64, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{16, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{16, 64, 64, 320, 64, 64, 3, 3});

  default_shapes_["iconv_4x4_a16w8acc16"] = std::vector<std::vector<size_t>>();
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 8, 8, 1280, 8, 8, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 8, 8, 1280, 8, 8, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 16, 16, 1280, 16, 16, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 1280, 32, 32, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{960, 32, 32, 640, 32, 32, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 640, 64, 64, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{960, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 32, 32, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 16, 16, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 8, 8, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 4, 64, 64, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{16, 64, 64, 320, 64, 64, 3, 3});
  default_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{16, 64, 64, 320, 64, 64, 3, 3});

  raw_shapes_["iconv_4x2_a16w8acc16"] = std::vector<std::vector<size_t>>();
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{128, 56, 56, 256, 28, 28, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{256, 28, 28, 512, 14, 14, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{512, 14, 14, 1024, 7, 7, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{128, 56, 56, 128, 56, 56, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{256, 28, 28, 256, 28, 28, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{512, 14, 14, 512, 14, 14, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1024, 7, 7, 1024, 7, 7, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{3, 224, 224, 128, 56, 56, 7, 7});

  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 8, 8, 1280, 8, 8, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 8, 8, 1280, 8, 8, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 1280, 32, 32, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{960, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 640, 64, 64, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{960, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 32, 32, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 16, 16, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 8, 8, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 4, 64, 64, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{16, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x2_a16w8acc16"].push_back(
      std::vector<size_t>{4, 64, 64, 320, 64, 64, 3, 3});

  raw_shapes_["iconv_4x4_a16w8acc16"] = std::vector<std::vector<size_t>>();
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 8, 8, 1280, 8, 8, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 8, 8, 1280, 8, 8, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{2560, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 16, 16, 1280, 16, 16, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 1280, 32, 32, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1920, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{960, 32, 32, 640, 32, 32, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 640, 64, 64, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{960, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 320, 32, 32, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{640, 32, 32, 640, 16, 16, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{1280, 16, 16, 1280, 8, 8, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{320, 64, 64, 4, 64, 64, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{16, 64, 64, 320, 64, 64, 3, 3});
  raw_shapes_["iconv_4x4_a16w8acc16"].push_back(
      std::vector<size_t>{4, 64, 64, 320, 64, 64, 3, 3});

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  iconv_id_ = iconv_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + ryzenai::PSF_A8W8_QDQ_XCLBIN_PATH;

  if (a_dtype_ == "uint16")
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSJ_A16W8_QDQ_XCLBIN_PATH;

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

  txn_fname_prefix_ = "iconv_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  param_fname_prefix_ = "iconv_4x2_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "iconv_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
    param_fname_prefix_ = "iconv_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_acc_header.at(c_dtype_);
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("txn_fname_prefix : {}", txn_fname_prefix_));
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("param_fname_prefix : {}", param_fname_prefix_));

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  KERNEL_M_MAX = 1024;

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

  if (attr.count("strides") &&
      attr.at("strides").type() == typeid(std::vector<int>)) {
    const auto &strides_vector =
        std::any_cast<const std::vector<int> &>(attr.at("strides"));

    if (strides_vector.size() == 2) {
      strides_[0] = strides_vector[0];
      strides_[1] = strides_vector[1];
    } else {
      std::cout
          << "Strides attribute does not have the expected number of "
             "elements.Number of passed : strides_vector.size(), Expected:2"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("iConv: Strides: " + std::to_string(strides_vector[0]) +
                      ", " + std::to_string(strides_vector[1]));
  } else {
    std::cout << "Strides attribute not found or not of correct type."
              << std::endl;
  }

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
             "elements.Number of passed : input_shape_vector.size(), Expected:4"
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
             "Expected:1"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("iConv: InputFormat: " + input_format_);
  } else {
    std::cout << "Input Format attribute not found or not of correct type."
              << std::endl;
  }

  std::call_once(logger_flag_, []() {
    std::string header = "iconv_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[iCONV] ID: " + std::to_string(iconv_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
                    a_dtype_ + ", " + b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void iconv<InT, WtT, OutT>::set_params(const std::string &model_name,
                                       std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "PSI") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSI_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "PSR") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSR_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "4x4PSR") {
    XCLBIN_FNAME =
        OpInterface::get_dod_base_dir() + ryzenai::PSR4x4_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  kernel_x_shape_[0] = input_shape.at(0); // YI
  kernel_x_shape_[1] = input_shape.at(1); // XI
  kernel_z_shape_[0] = input_shape.at(2); // YO
  kernel_z_shape_[1] = input_shape.at(3); // XO

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void iconv<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("iconv initialize_const_params(ptr) ...");

  DOD_THROW_IF((const_params.size() != 3),
               OpsFusion::dod_format("Unsupported const spec for iconv\n") +
                   OpsFusion::dod_format("(Details : #const params == 3 ({})",
                                         const_params.size()));

  auto CO = const_params.at(0).shape.at(0);
  auto CI = const_params.at(0).shape.at(1);
  auto KY = const_params.at(0).shape.at(2);
  auto KX = const_params.at(0).shape.at(3);

  kernel_y_shape_[0] = KY; // KY
  kernel_y_shape_[1] = KX; // KX

  if (strides_[0] == 4) { // special case for 7x7 CONV in PSI
    CI = 8;
    KX = 4;
  }

  auto CO_padded = CO;
  auto CI_padded = CI;

  auto w_dest = (WtT *)dest;
  auto weights = (WtT *)const_params.at(0).data;

  auto qdq = (int64_t *)const_params.at(1).data;
  auto qdq_params = (int32_t *)const_params.at(2).data;

  if (strides_[0] == 1 && (CI == 1)) { // DWC
    iconv_matrix::DwcWgtTensor<WtT, C_IN_SPLIT_DWC> W(CO, KY, KX, w_dest);
    format_dwc_wgt(W, weights, qdq, qdq_params[qdq_c1_idx],
                   qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                   qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
  } else if (strides_[0] == 1 || strides_[0] == 2) {      // CONV
    if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design

      if (CI == 4) {
        CI_padded = 16;
      }
      auto split_mode = search_subv_mode(inputShape_[3]);
      if (split_mode == 0) {
        constexpr SUBV_T subv = get_subv(0);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        CO_padded = std::max((int)CO, Cos * 4);
        iconv_matrix::ConvWgtTensor<WtT, Cos, Cis> W(CO_padded, CI_padded, KY,
                                                     KX, w_dest);
        format_conv_wgt(W, weights, CO, CI, qdq, qdq_params[qdq_c1_idx],
                        qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                        qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
      } else if (split_mode == 1) {
        constexpr SUBV_T subv = get_subv(1);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        CO_padded = std::max((int)CO, Cos * 4);
        iconv_matrix::ConvWgtTensor<WtT, Cos, Cis> W(CO_padded, CI_padded, KY,
                                                     KX, w_dest);

        format_conv_wgt(W, weights, CO, CI, qdq, qdq_params[qdq_c1_idx],
                        qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                        qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
      } else if (split_mode == 2) {
        constexpr SUBV_T subv = get_subv(2);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        CO_padded = std::max((int)CO, Cos * 4);
        iconv_matrix::ConvWgtTensor<WtT, Cos, Cis> W(CO_padded, CI_padded, KY,
                                                     KX, w_dest);
        format_conv_wgt(W, weights, CO, CI, qdq, qdq_params[qdq_c1_idx],
                        qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                        qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
      } else if (split_mode == 3) {
        constexpr SUBV_T subv = get_subv(3);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        CO_padded = std::max((int)CO, Cos * 4);
        iconv_matrix::ConvWgtTensor<WtT, Cos, Cis> W(CO_padded, CI_padded, KY,
                                                     KX, w_dest);
        format_conv_wgt(W, weights, CO, CI, qdq, qdq_params[qdq_c1_idx],
                        qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                        qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
      } else {
        std::cout << "ERROR: Unsupported split mode" << std::endl;
      }
    } else {                  // 4x2 mode
      if (strides_[0] == 1) { // PSR 3x3 stride1
        if (inputShape_[3] == 8) {
          iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV> W(
              CO, CI, KY, KX, w_dest);
          format_conv_wgt(W, weights, CO, CI, qdq, qdq_params[qdq_c1_idx],
                          qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                          qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
        } else {
          if (CO == 4) {
            CO_padded = 32;
          }
          if (CI == 4) {
            CI_padded = 16;
          }
          iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV_PSR,
                                      C_IN_SPLIT_CONV_PSR>
              W(CO_padded, CI_padded, KY, KX, w_dest);
          // uint32_t wgt_zp = qdq_params[qdq_wgt_zp_idx];
          format_conv_wgt(W, weights, CO, CI, qdq, qdq_params[qdq_c1_idx],
                          qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                          qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
        }
      } else { // 3x3 stride2
        iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV> W(
            CO, CI, KY, KX, w_dest);
        format_conv_wgt(W, weights, CO, CI, qdq, qdq_params[qdq_c1_idx],
                        qdq_params[qdq_c2_idx], qdq_params[qdq_Stdm_idx],
                        qdq_params[qdq_Sout_idx], qdq_params[qdq_wgt_zp_idx]);
      }
    }
  } else { // CONV7
    int Ci_no_fold = 3;
    int Ky_no_fold = 7;
    int Kx_no_fold = 7;

    int fold_factor = 2;
    int Ci_gran = 4;

    WtT wgt_zp = qdq_params[qdq_wgt_zp_idx];
    iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV7, C_IN_SPLIT_CONV7> W(
        CO, CI, KY, KX, w_dest);
    fold_conv_wgt<WtT>(weights, wgt_zp, CO, Ci_no_fold, Ky_no_fold, Kx_no_fold,
                       fold_factor, Ci_gran, W);

    for (int c = 0; c < W.Co; ++c) {
      W.set_qdq_c0(c, qdq[c]);
    }

    W.set_qdq_c1(qdq_params[qdq_c1_idx]);
    W.set_qdq_c2(qdq_params[qdq_c2_idx]);
    W.set_shift_tdm(qdq_params[qdq_Stdm_idx]);
    W.set_shift_res(qdq_params[qdq_Sout_idx]);
    W.set_zp_wgt(qdq_params[qdq_wgt_zp_idx]);
  }
  const_pad_ = uint16_t(qdq_params[qdq_ifm_zp_idx]);
  RYZENAI_LOG_TRACE("iConv initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void iconv<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("iConv initialize_const_params ...");

  DOD_THROW_IF((const_params.size() != 3),
               OpsFusion::dod_format("Unsupported const spec for iConv\n") +
                   OpsFusion::dod_format("(Details : #const params == 3 ({})",
                                         const_params.size()));

  auto CO = const_params.at(0).shape.at(0);
  auto CI = const_params.at(0).shape.at(1);
  auto KY = const_params.at(0).shape.at(2);
  auto KX = const_params.at(0).shape.at(3);

  if (strides_[0] == 1 && (CI == 1)) {
    CI = CO;
  }

  std::vector<size_t> input_shape = {CI,
                                     static_cast<size_t>(kernel_x_shape_[0]),
                                     static_cast<size_t>(kernel_x_shape_[1]),
                                     CO,
                                     static_cast<size_t>(kernel_z_shape_[0]),
                                     static_cast<size_t>(kernel_z_shape_[1]),
                                     KY,
                                     KX};
  auto mapped_shape = map_padded_shape(input_shape);
  CI = mapped_shape[0];
  CO = mapped_shape[3];
  KY = mapped_shape[6];
  KX = mapped_shape[7];

  int wgt_size;
  if (strides_[0] == 1 && (const_params.at(0).shape.at(1) == 1)) { // DWC
    wgt_size =
        iconv_matrix::DwcWgtTensor<WtT, C_IN_SPLIT_DWC>::size(CO, KY, KX);
  } else if (strides_[0] == 1 || strides_[0] == 2) {      //
    if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
      auto split_mode = search_subv_mode(kernel_x_shape_[1]);
      if (split_mode == 0) {
        constexpr SUBV_T subv = get_subv(0);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else if (split_mode == 1) {
        constexpr SUBV_T subv = get_subv(1);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else if (split_mode == 2) {
        constexpr SUBV_T subv = get_subv(2);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else if (split_mode == 3) {
        constexpr SUBV_T subv = get_subv(3);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else {
        std::cout << "ERROR: Unsupported split mode" << std::endl;
      }
    } else {
      if (strides_[0] == 1) { // PSR 3x3 stride1
        if (kernel_x_shape_[1] == 8)
          wgt_size = iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV,
                                                 C_IN_SPLIT_CONV>::size(CO, CI,
                                                                        KY, KX);
        else
          wgt_size =
              iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV_PSR,
                                          C_IN_SPLIT_CONV_PSR>::size(CO, CI, KY,
                                                                     KX);
      } else { // 3x3 stride2
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV,
                                        C_IN_SPLIT_CONV>::size(CO, CI, KY, KX);
      }
    }
  } else { // CONV7
    wgt_size =
        iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV7,
                                    C_IN_SPLIT_CONV7>::size(CO, CI, KY, KX);
  }
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;

  /* Create input/output BOs */
  const int B_BO_SIZE = wgt_size;
  const int A_BO_SIZE =
      (CI * mapped_shape[1] * mapped_shape[2] * a_dtype_size_);
  const int C_BO_SIZE =
      (CO * mapped_shape[4] * mapped_shape[5] * c_dtype_size_);

  RYZENAI_LOG_TRACE("iCONV: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
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
  RYZENAI_LOG_TRACE("iConv initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void iconv<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                    std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("iConv execute ...");

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  int64_t exec_start = GET_ELAPSED_TIME_NS();
  auto [CI, YI, XI] = extract_CYX(input.at(0), input_format_);
  auto [CO, YO, XO] = extract_CYX(output.at(0), input_format_);
  std::vector<size_t> param_shape = {CI,
                                     YI,
                                     XI,
                                     CO,
                                     YO,
                                     XO,
                                     static_cast<size_t>(kernel_y_shape_[0]),
                                     static_cast<size_t>(kernel_y_shape_[1])};
  auto mapped_shape = map_padded_shape(param_shape);

  InT *a = (InT *)input.at(0).data;

  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();

  if (CI == 3) { // special case for PSI 7x7 CONV
    int Sx_no_fold = 4;
    int pad_no_fold = 3;

    int fold_factor = 2;
    int Ci_gran = 4;
    int Xi_gran = 4;
    ActTensor<InT> X(mapped_shape[0], mapped_shape[1], mapped_shape[2],
                     a_bo_map);
    fold_conv_ifm<InT>(a, const_pad_, CI, YI, XI, XO, kernel_y_shape_[1],
                       Sx_no_fold, pad_no_fold, fold_factor, Ci_gran, Xi_gran,
                       X);
  } else if (CI == 4) { // special case for PSR
    ActTensor<InT> X(mapped_shape[0], mapped_shape[1], mapped_shape[2],
                     a_bo_map);
    format_conv_ifm<InT>(a, const_pad_, CI, X);
  } else {
    memcpy((void *)a_bo_map, (void *)a, (CI * YI * XI * a_dtype_size_));
  }

  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  auto instr_bo_key = get_instr_key(txn_fname_prefix_, mapped_shape);
  auto param_bo_key =
      get_instr_key(param_fname_prefix_, mapped_shape) + "_param";

  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  const xrt::bo &param_bo = instr_reg_.get_param_bo(param_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);

  // Ignore instruction key from registry since const padding instruction is
  // required.
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(instr_bo_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  uint32_t zp = uint16_t(const_pad_);
  uint32_t pad_val = zp | (zp << 16);
  std::vector<uint8_t> txn_w_pad;
  if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 4);
  } else {
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 2);
  }

  aiectrl::op_buf i_buf;
  i_buf.addOP(aiectrl::transaction_op(txn_w_pad.data()));
  size_t i_bo_words = i_buf.ibuf_.size();
  xrt::bo i_bo =
      xrt::bo(xrt_ctx_->get_context(), i_bo_words, xrt::bo::flags::cacheable,
              xrt_ctx_->get_kernel().group_id(1));
  i_bo.write(i_buf.ibuf_.data());
  i_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  i_bo_words = i_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;
  // launch the kernel
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for GEMM that supports transaction binary flow
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, i_bo, i_bo_words, c_bo_.address() + DDR_AIE_ADDR_OFFSET,
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

  OutT *aie_out = (OutT *)output.at(0).data;
  if (CO == 4) { // special case for PSR
    //   ActTensor<OutT> Out(mapped_shape[3], mapped_shape[4], mapped_shape[5],
    //       c_bo_map);
    //   format_conv_ofm<OutT>(aie_out, CO, Out);
    for (int i = 0; i < YO * XO; i++) {
      memcpy((void *)&aie_out[i * CO], (void *)&c_bo_map[i * mapped_shape[3]],
             (CO * c_dtype_size_));
    }

  } else {
    memcpy((void *)aie_out, (void *)c_bo_map, (CO * YO * XO * c_dtype_size_));
  }
  int64_t exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(iconv_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1]) +
      " " + std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
  RYZENAI_LOG_TRACE("iConv execute ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void iconv<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> iconv<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto KY = input.at(1).shape.at(2);
  auto KX = input.at(1).shape.at(3);
  auto [CI, YI, XI] = extract_CYX(input.at(0), input_format_);
  auto [CO, YO, XO] = extract_CYX(input.at(4), input_format_);
  std::vector<size_t> input_shape = {CI, YI, XI, CO, YO, XO, KY, KX};
  auto param_shape = map_padded_shape(input_shape);
  std::string txn_key = get_instr_key(txn_fname_prefix_, param_shape);

  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  // assume input.at(3) is qdq_params
  auto qdq_param = (int32_t *)input.at(3).data;
  uint32_t zp = uint16_t(qdq_param[qdq_ifm_zp_idx]);
  uint32_t pad_val = zp | (zp << 16);
  std::vector<uint8_t> txn_w_pad;
  if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 4);
  } else {
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 2);
  }
  return txn_w_pad;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> iconv<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  auto KY = input.at(1).shape.at(2);
  auto KX = input.at(1).shape.at(3);
  auto [CI, YI, XI] = extract_CYX(input.at(0), input_format_);
  auto [CO, YO, XO] = extract_CYX(input.at(4), input_format_);
  std::vector<size_t> input_shape = {CI, YI, XI, CO, YO, XO, KY, KX};
  auto param_shape = map_padded_shape(input_shape);
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      get_instr_key(param_fname_prefix_, param_shape) + "_param";
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
std::vector<OpArgMap> iconv<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // input --> [input, weights, qdq, qdq_params, output]
  // Check if IO buffers have batch.
  auto KY = input.at(1).shape.at(2);
  auto KX = input.at(1).shape.at(3);
  auto [CI, YI, XI] = extract_CYX(input.at(0), input_format_);
  auto [CO, YO, XO] = extract_CYX(input.at(4), input_format_);
  std::vector<size_t> input_shape = {CI, YI, XI, CO, YO, XO, KY, KX};
  auto param_shape = map_padded_shape(input_shape);
  CI = param_shape[0];
  YI = param_shape[1];
  XI = param_shape[2];
  CO = param_shape[3];
  YO = param_shape[4];
  XO = param_shape[5];
  KY = param_shape[6];
  KX = param_shape[7];

  int wgt_size;
  if (strides_[0] == 1 &&
      (CI == 128 || CI == 256 || CI == 512 || CI == 1024)) { // DWC
    wgt_size =
        iconv_matrix::DwcWgtTensor<WtT, C_IN_SPLIT_DWC>::size(CO, KY, KX);
  } else if (strides_[0] == 1 || strides_[0] == 2) {      // CONV
    if (design_param_.find("4x4") != std::string::npos) { // PSR 4x4 design
      auto split_mode = search_subv_mode(XI);
      if (split_mode == 0) {
        constexpr SUBV_T subv = get_subv(0);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else if (split_mode == 1) {
        constexpr SUBV_T subv = get_subv(1);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else if (split_mode == 2) {
        constexpr SUBV_T subv = get_subv(2);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else if (split_mode == 3) {
        constexpr SUBV_T subv = get_subv(3);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        int Cop = std::max((int)CO, Cos * 4);
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, Cos, Cis>::size(Cop, CI, KY, KX);
      } else {
        std::cout << "ERROR: Unsupported split mode" << std::endl;
      }
    } else {
      if (strides_[0] == 1) { // PSR 3x3 stride1
        if (XI == 8)
          wgt_size = iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV,
                                                 C_IN_SPLIT_CONV>::size(CO, CI,
                                                                        KY, KX);
        else
          wgt_size =
              iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV_PSR,
                                          C_IN_SPLIT_CONV_PSR>::size(CO, CI, KY,
                                                                     KX);
      } else { // 3x3 stride2
        wgt_size =
            iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV,
                                        C_IN_SPLIT_CONV>::size(CO, CI, KY, KX);
      }
    }
  } else { // CONV7
    wgt_size =
        iconv_matrix::ConvWgtTensor<WtT, C_OUT_SPLIT_CONV7,
                                    C_IN_SPLIT_CONV7>::size(CO, CI, KY, KX);
  }

  size_t const_params_bo_size = wgt_size;
  size_t input_bo_size = (CI * YI * XI * a_dtype_size_);
  size_t output_bo_size = (CO * YO * XO * c_dtype_size_);
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 4, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("iConv Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag iconv<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t iconv<InT, WtT, OutT>::iconv_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag iconv<InT, WtT, OutT>::instr_reg_flag_;

template class iconv<uint16_t, uint8_t, uint16_t>;

} // namespace ryzenai
