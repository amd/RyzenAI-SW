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

#ifndef _WIN32
#include <cmath>
#endif

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

#include <utils/instruction_registry.hpp>
#include <utils/txn_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/op_interface.hpp>
#include <ops/xcom/conv/conv.hpp>
#include <ops/xcom/conv/weight_shuffle.hpp>
#include <ops/xcom/ddr_buffer_info.hpp>
#include <txn_helper/txn_helper.hpp>
#include <utils/logging.hpp>

namespace ryzenai {

namespace xcom {

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
std::once_flag conv2d<InT, WtT, BiasT, OutT, DWC>::logger_flag_;

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
uint64_t conv2d<InT, WtT, BiasT, OutT, DWC>::count_ = 0;

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
std::once_flag conv2d<InT, WtT, BiasT, OutT, DWC>::instr_reg_flag_;

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
std::string conv2d<InT, WtT, BiasT, OutT, DWC>::get_instr_key(
    const std::string &prefix, const conv_shape_t &shape_info) const {

  return "xcomconv2d_" + prefix + "_" + std::to_string(shape_info.H) + "x" +
         std::to_string(shape_info.W) + "x" + std::to_string(shape_info.C_in) +
         "_" + std::to_string(shape_info.C_out);
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
void conv2d<InT, WtT, BiasT, OutT, DWC>::setup_instr_registry() {

  constexpr bool is_qdq = std::is_same_v<BiasT, std::int32_t>;

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  for (const auto &[mkey, raw_to_padded_shape] : supported_shapes_) {
    for (const auto &[raw_shape, padded_shape] : raw_to_padded_shape) {
      auto key = get_instr_key(mkey, padded_shape);
      instructions.push_back(std::make_pair(key, false));
      if constexpr (is_qdq) {
        layer_params.push_back(std::make_pair(key + "_param", false));
        layer_params.push_back(
            std::make_pair(key + "_weight_format_param", false));
        layer_params.push_back(std::make_pair(key + "_ddr_buffer_info", false));
      }
    }
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
  if constexpr (is_qdq) {
    instr_reg_.add_layer_params(layer_params);
  }
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
conv2d<InT, WtT, BiasT, OutT, DWC>::conv2d(const std::string &a_dtype,
                                           const std::string &b_dtype,
                                           const std::string &bias_dtype,
                                           const std::string &c_dtype,
                                           bool load_xrt) {

  // TO DO: expand based on supported types
  const std::map<std::string, std::string> txnbin_a_header = {
      {"int8", "a8"}, {"uint16", "au16"}};
  const std::map<std::string, std::string> txnbin_b_header = {{"int8", "w8"},
                                                              {"uint8", "wu8"}};
  const std::map<std::string, std::string> txnbin_c_header = {
      {"int8", "out8"}, {"uint16", "outu16"}};

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  bias_dtype_ = bias_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  bias_dtype_size_ = sizeof(BiasT);
  c_dtype_size_ = sizeof(OutT);

  // there are some rules on what this string can be
  const std::string op_name = DWC ? "xcomdwc_" : "xcomconv2d_";

  txn_fname_prefix_ = op_name + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_c_header.at(c_dtype_);

  // default shape is the padded shaped used in AIE for BO allocation
#define _CONV(h, w, c_in, c_out)                                               \
  { h, w, c_in, c_out }

  if constexpr (DWC) {
    // in case different sizes are supported
    supported_shapes_[txn_fname_prefix_ + "_1x1_s1"] =
        raw_to_padded_shape_map{};
    supported_shapes_.at(txn_fname_prefix_ + "_1x1_s1")[_CONV(64, 64, 64, 64)] =
        _CONV(64, 64, 64, 64);
  } else {

    bool is_a8w8out8 =
        (a_dtype_ == "int8") && (b_dtype_ == "int8") && (c_dtype_ == "int8");
    bool is_au16wu8outu16 = (a_dtype_ == "uint16") && (b_dtype_ == "uint8") &&
                            (c_dtype_ == "uint16");

    supported_shapes_[txn_fname_prefix_ + "_1x1_s1"] =
        raw_to_padded_shape_map{};
    supported_shapes_[txn_fname_prefix_ + "_3x3_s1"] =
        raw_to_padded_shape_map{};
    supported_shapes_[txn_fname_prefix_ + "_3x3_s2"] =
        raw_to_padded_shape_map{};

    if (is_a8w8out8) {
      supported_shapes_.at(txn_fname_prefix_ +
                           "_1x1_s1")[_CONV(320, 320, 48, 24)] =
          _CONV(320, 320, 48, 24);
    } else if (is_au16wu8outu16) {
      supported_shapes_.at(txn_fname_prefix_ + "_1x1_s1")[_CONV(64, 64, 4, 4)] =
          _CONV(64, 64, 4, 4);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_1x1_s1")[_CONV(256, 256, 512, 256)] =
          _CONV(256, 256, 512, 256);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_1x1_s1")[_CONV(512, 512, 256, 128)] =
          _CONV(512, 512, 256, 128);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_1x1_s1")[_CONV(256, 256, 128, 256)] =
          _CONV(256, 256, 128, 256);
      supported_shapes_.at(txn_fname_prefix_ + "_1x1_s1")[_CONV(64, 64, 8, 8)] =
          _CONV(64, 64, 8, 8);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_1x1_s1")[_CONV(128, 128, 256, 512)] =
          _CONV(128, 128, 256, 512);

      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(64, 64, 512, 512)] =
          _CONV(64, 64, 512, 512);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(64, 64, 4, 512)] =
          _CONV(64, 64, 4, 512);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(128, 128, 512, 512)] =
          _CONV(128, 128, 512, 512);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(256, 256, 512, 512)] =
          _CONV(256, 256, 512, 512);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(256, 256, 512, 256)] =
          _CONV(256, 256, 512, 256);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(256, 256, 256, 256)] =
          _CONV(256, 256, 256, 256);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(512, 512, 256, 256)] =
          _CONV(512, 512, 256, 256);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(512, 512, 256, 128)] =
          _CONV(512, 512, 256, 128);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(512, 512, 128, 128)] =
          _CONV(512, 512, 128, 128);
      // Needs depadding - should only be at end of graph
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(512, 512, 128, 4)] =
          _CONV(512, 512, 128, 3);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(256, 256, 128, 256)] =
          _CONV(256, 256, 128, 256);
      // Needs padding - should only be at beginning of graph
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(512, 512, 4, 128)] =
          _CONV(512, 512, 3, 128);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(64, 64, 512, 8)] =
          _CONV(64, 64, 512, 8);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s1")[_CONV(128, 128, 256, 512)] =
          _CONV(128, 128, 256, 512);

      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s2")[_CONV(513, 513, 128, 128)] =
          _CONV(513, 513, 128, 128);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s2")[_CONV(257, 257, 256, 256)] =
          _CONV(257, 257, 256, 256);
      supported_shapes_.at(txn_fname_prefix_ +
                           "_3x3_s2")[_CONV(129, 129, 512, 512)] =
          _CONV(129, 129, 512, 512);

    } else {
      DOD_THROW("Unsupported shapes/data for XCOM::CONV2D!");
    }
  }

#undef _CONV

  count_++;
  id_ = count_;

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
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
    std::string header = "xcom-conv2d M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[XCOM-CONV] ID: " + std::to_string(id_) + ", XCLBIN: " +
                    XCLBIN_FNAME_ + ", (a_dtype, b_dtype, c_dtype): (" +
                    a_dtype_ + ", " + b_dtype_ + ", " + c_dtype_ + ")");
}

std::ostream &operator<<(std::ostream &os, const conv_shape_t &shape_info) {
  return os << "H: " << shape_info.H << ", W: " << shape_info.W
            << ", C_in: " << shape_info.C_in << ", C_out: " << shape_info.C_out;
}

std::ostream &operator<<(std::ostream &os, const WGTShuffleParam &param) {
  return os << "input_ic_: " << param.input_ic_ << "\n"
            << "output_c_: " << param.output_c_ << "\n"
            << "align_oc_: " << param.align_oc_ << "\n"
            << "align_ic_: " << param.align_ic_ << "\n"
            << "kernel_h_: " << param.kernel_h_ << "\n"
            << "kernel_w_: " << param.kernel_w_ << "\n"
            << "align_kernel_w_: " << param.align_kernel_w_ << "\n"
            << "stride_w_: " << param.stride_w_ << "\n"
            << "pad_left: " << param.pad_left << "\n"
            << "chl_augmentation_opt: " << param.chl_augmentation_opt << "\n"
            << "mode: " << param.mode << "\n"
            << "oc_mt: " << param.oc_mt << "\n"
            << "oc_per_aie: " << param.oc_per_aie << "\n"
            << "ic_per_aie: " << param.ic_per_aie << "\n"
            << "OCPf: " << param.OCPf << "\n"
            << "BIAS_DUP_NUM: " << param.BIAS_DUP_NUM << "\n"
            << "iter_ocg_: " << param.iter_ocg_ << "\n"
            << "iter_icg_: " << param.iter_icg_ << "\n"
            << "tile_ocg_: " << param.tile_ocg_ << "\n"
            << "tile_icg_: " << param.tile_icg_ << "\n"
            << "OCp: " << param.OCp << "\n"
            << "ICp: " << param.ICp << "\n"
            << "is_first_conv: " << param.is_first_conv << "\n"
            << "split_num: " << param.split_num << "\n"
            << "RowNum: " << param.RowNum << "\n"
            << "enable_col_num: " << param.enable_col_num << "\n"
            << "x_zp: " << param.x_zp << "\n"
            << "y_zp: " << param.y_zp << "\n"
            << "w_zp: " << param.w_zp << "\n"
            << "prelu_in: " << param.prelu_in << "\n"
            << "prelu_shift: " << param.prelu_shift << "\n"
            << "tile_scale: " << param.tile_scale << "\n"
            << "in_width: " << param.in_width << "\n"
            << "wgt_width: " << param.wgt_width << "\n"
            << "x_s: " << param.x_s << "\n"
            << "y_s: " << param.y_s << "\n"
            << "w_s: " << param.w_s << "\n"
            << "wgt_is_int8: " << (std::uint32_t)param.wgt_is_int8 << "\n"
            << "wgt_is_uint8: " << (std::uint32_t)param.wgt_is_uint8 << "\n"
            << "is_prelu: " << (std::uint32_t)param.is_prelu << "\n"
            << "is_fused_with_tile: " << (std::uint32_t)param.is_fused_with_tile
            << "\n";
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
conv_shape_t conv2d<InT, WtT, BiasT, OutT, DWC>::get_padded_shape(
    const conv_shape_t &shape_info) const {

  auto txn_fname_prefix =
      txn_fname_prefix_ + "_" + std::to_string(kernel_y_dim_) + "x" +
      std::to_string(kernel_x_dim_) + "_s" + std::to_string(stride_x_);
  auto it = supported_shapes_.find(txn_fname_prefix);
  const auto &raw_shape_map = it->second;

  if (raw_shape_map.end() == raw_shape_map.find(shape_info)) {
    DOD_THROW(OpsFusion::dod_format("Unsupported shape {}", shape_info));
  }

  return raw_shape_map.at(shape_info);
}

// NOTE: this is mainly to set state variables for standalone unit tests - NOT
// through fusion_rt flow!!
template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
void conv2d<InT, WtT, BiasT, OutT, DWC>::set_params(
    const conv_params_t &params) {

  activation_shape_[0] = 1; // hardcode N to be 1
  activation_shape_[1] = params.static_params.shape_info.H;
  activation_shape_[2] = params.static_params.shape_info.W;
  activation_shape_[3] = params.static_params.shape_info.C_in;

  kernel_x_dim_ = params.static_params.kernel_x;
  kernel_y_dim_ = params.static_params.kernel_y;
  num_output_channels_ = params.static_params.shape_info.C_out;
  bias_en_ = params.static_params.bias;
  relu_en_ = params.relu;

  stride_x_ = params.static_params.stride_x;
  stride_y_ = params.static_params.stride_y;

  DOD_ASSERT(stride_x_ == stride_y_, "stride x and y do not match!");

  padded_shape_info_ = get_padded_shape(params.static_params.shape_info);
  input_padded_shape_[0] = 1;
  input_padded_shape_[1] = padded_shape_info_.H;
  input_padded_shape_[2] = padded_shape_info_.W;
  input_padded_shape_[3] = padded_shape_info_.C_in;

  // TO DO: this should be moved to where weights is populated, since
  // weights might be replicated etc.
  weights_padded_shape_[0] = padded_shape_info_.C_out;
  weights_padded_shape_[1] = params.static_params.kernel_x;
  weights_padded_shape_[2] = params.static_params.kernel_y;
  weights_padded_shape_[3] = padded_shape_info_.C_in;

  output_padded_shape_[0] = 1;
  output_padded_shape_[1] =
      padded_shape_info_.H / params.static_params.stride_y;
  output_padded_shape_[2] =
      padded_shape_info_.W / params.static_params.stride_x;
  output_padded_shape_[3] = padded_shape_info_.C_out;

  // For now assume these are the same - if not will need to do extra
  // zero-padding
  DOD_ASSERT(
      num_output_channels_ == padded_shape_info_.C_out,
      "Padded num output channel does not match original output channels");

  // padding on inner dims can not be fused, would need CPU to intervene
  DOD_ASSERT(input_padded_shape_[3] == activation_shape_[3],
             "Padding in inner dim");
  DOD_ASSERT(input_padded_shape_[2] == activation_shape_[2],
             "Padding in inner dim");

  if constexpr (std::is_same_v<BiasT, int32_t>) {
    XCLBIN_FNAME_ = ryzenai::XCOM_4x4_Q_XCLBIN_PATH;
  } else {
    XCLBIN_FNAME_ = ryzenai::XCOM_4x4_XCLBIN_PATH;
  }

  RYZENAI_LOG_TRACE("XCOM::CONV2D::set_params: xclbin name " + XCLBIN_FNAME_);

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
const std::vector<std::uint8_t>
conv2d<InT, WtT, BiasT, OutT, DWC>::get_kernel_params(
    const std::string &param_key) const {
  // std::cout << "Accessing param_key: " << param_key << std::endl;
  constexpr bool is_qdq = std::is_same_v<BiasT, std::int32_t>;

  if constexpr (is_qdq) {
    Transaction &txn = Transaction::getInstance();
    std::string param_string = txn.get_txn_str(param_key);
    std::istringstream params_stream(param_string, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(params_stream)),
                              std::istreambuf_iterator<char>());

    return data;
  } else {
    std::vector<std::uint8_t> data;
    return data;
  }
}

// this will mainly be called through fusion rt flow
// should be called after generating fused txn, so expect any state populated
// there to to be maintained
template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
void conv2d<InT, WtT, BiasT, OutT, DWC>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv2d initialize_const_params(ptr) ...");

  constexpr std::uint32_t ACT_SCALE_INDEX = 0;
  constexpr std::uint32_t ACT_ZP_INDEX = 1;
  constexpr std::uint32_t WEIGHT_INDEX = 2;
  constexpr std::uint32_t WEIGHT_SCALE_INDEX = 3;
  constexpr std::uint32_t WEIGHT_ZP_INDEX = 4;
  constexpr std::uint32_t BIAS_INDEX = 5;
  constexpr std::uint32_t OUT_SCALE_INDEX = 8;
  constexpr std::uint32_t OUT_ZP_INDEX = 9;

  const auto &input_shape =
      std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
  const auto &output_shape =
      std::any_cast<const std::vector<int> &>(attr.at("output_shape"));
  const auto &kernel_shape =
      std::any_cast<const std::vector<int> &>(attr.at("kernel_shape"));
  const auto &strides =
      std::any_cast<const std::vector<int> &>(attr.at("strides"));

  conv_qdq_info_t qdq_params;
  using scale_t = float;
  using zp_t = std::uint16_t;
  using weight_zp_t = std::uint8_t;

  qdq_params.act_scale =
      *static_cast<scale_t *>(const_params.at(ACT_SCALE_INDEX).data);
  qdq_params.act_zero_point =
      *static_cast<zp_t *>(const_params.at(ACT_ZP_INDEX).data);
  qdq_params.weight_scale =
      *static_cast<scale_t *>(const_params.at(WEIGHT_SCALE_INDEX).data);
  qdq_params.weight_zero_point =
      *static_cast<weight_zp_t *>(const_params.at(WEIGHT_ZP_INDEX).data);
  qdq_params.out_scale =
      *static_cast<scale_t *>(const_params.at(OUT_SCALE_INDEX).data);
  qdq_params.out_zero_point =
      *static_cast<zp_t *>(const_params.at(OUT_ZP_INDEX).data);

  constexpr std::uint32_t TENSOR_SHAPE_C_INDEX = 1;
  constexpr std::uint32_t TENSOR_SHAPE_H_INDEX = 2;
  constexpr std::uint32_t TENSOR_SHAPE_W_INDEX = 3;
  qdq_params.C_in = input_shape[TENSOR_SHAPE_C_INDEX];
  qdq_params.C_out = output_shape[TENSOR_SHAPE_C_INDEX];
  qdq_params.H = input_shape[TENSOR_SHAPE_H_INDEX];
  qdq_params.W = input_shape[TENSOR_SHAPE_W_INDEX];
  qdq_params.kernel_size_x = kernel_shape[0];
  qdq_params.kernel_size_y = kernel_shape[1];
  qdq_params.stride_x = strides[0];
  qdq_params.stride_y = strides[1];

  if constexpr (std::is_same_v<BiasT, int32_t> && !DWC) {
    // QDQ conv
    const conv_qdq_info_t *conv_qdq_params_p = &qdq_params;
    padded_shape_info_.H = conv_qdq_params_p->H;
    padded_shape_info_.W = conv_qdq_params_p->W;
    padded_shape_info_.C_in = conv_qdq_params_p->C_in;
    padded_shape_info_.C_out = conv_qdq_params_p->C_out;

    auto prefix = txn_fname_prefix_ + "_" +
                  std::to_string(conv_qdq_params_p->kernel_size_y) + "x" +
                  std::to_string(conv_qdq_params_p->kernel_size_x) + "_s" +
                  std::to_string(conv_qdq_params_p->stride_x);

    const std::string layer_param_key =
        get_instr_key(prefix, padded_shape_info_) + "_param";
    const std::string weight_shuffle_param_key =
        get_instr_key(prefix, padded_shape_info_) + "_weight_format_param";

    const auto layer_param_bvec = get_kernel_params(layer_param_key);
    DOD_ASSERT(layer_param_bvec.size() == LAYER_PARAM_SIZE,
               "Unexpected layer param size");
    const auto weight_shuffle_bvec =
        get_kernel_params(weight_shuffle_param_key);
    DOD_ASSERT(weight_shuffle_bvec.size() == sizeof(WGTShuffleParam),
               "Unexpected weight shuffle param size");

    std::vector<std::int32_t> layer_info_buf(layer_param_bvec.size() /
                                             sizeof(std::int32_t));

    memcpy(layer_info_buf.data(), layer_param_bvec.data(),
           layer_param_bvec.size());

    WGTShuffleParam param;
    memcpy(&param, weight_shuffle_bvec.data(), sizeof(WGTShuffleParam));

    param.x_zp = conv_qdq_params_p->act_zero_point;
    param.y_zp = conv_qdq_params_p->out_zero_point;
    param.w_zp = conv_qdq_params_p->weight_zero_point;
    param.x_s = 1.0f / conv_qdq_params_p->act_scale;
    param.y_s = 1.0f / conv_qdq_params_p->out_scale;
    param.w_s = 1.0f / conv_qdq_params_p->weight_scale;

    const Tensor &weight_tensor = const_params.at(WEIGHT_INDEX);
    const Tensor &bias_tensor = const_params.at(BIAS_INDEX);

    // TODO: does this need to be padded up??
    size_t weight_tensor_volume =
        std::accumulate(weight_tensor.shape.begin(), weight_tensor.shape.end(),
                        size_t{1}, std::multiplies{});

    size_t bias_tensor_volume =
        std::accumulate(bias_tensor.shape.begin(), bias_tensor.shape.end(),
                        size_t{1}, std::multiplies{});

    // pad up to multiple of 4
    size_t bias_tensor_volume_padded = 4 * ((bias_tensor_volume + 3) / 4);

    std::vector<char> weights(weight_tensor_volume * sizeof(WtT));
    // weight_tensor.data will be in [C_out, C_in, ky, kx] format
    // need to reformat this to be [C_out, ky, kx, C_in]
    constexpr std::uint32_t WEIGHT_TENSOR_SHAPE_COUT_IDX = 0;
    constexpr std::uint32_t WEIGHT_TENSOR_SHAPE_CIN_IDX = 1;
    constexpr std::uint32_t WEIGHT_TENSOR_SHAPE_KY_IDX = 2;
    constexpr std::uint32_t WEIGHT_TENSOR_SHAPE_KX_IDX = 3;
    DOD_ASSERT(weight_tensor.shape.size() == 4, "Expect 4 dim tensor");
    if (weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_KY_IDX) == 1 &&
        weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_KX_IDX) == 1) {
      // 1x1 kernel - copy over as is since [C_in, 1, 1]
      // can be interpreted as [1, 1, C_in]
      memcpy(weights.data(), weight_tensor.data,
             weight_tensor_volume * sizeof(WtT));
    } else {
      WtT const *src_p = static_cast<WtT *>(weight_tensor.data);
      WtT *dst_p = static_cast<WtT *>((void *)weights.data());

      size_t src_index = 0;
      size_t dest_stride_l =
          weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_CIN_IDX);
      size_t dest_stride_k =
          weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_KX_IDX) *
          weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_CIN_IDX);
      size_t dest_stride_i =
          weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_KY_IDX) *
          weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_KX_IDX) *
          weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_CIN_IDX);

      for (size_t i = 0;
           i < weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_COUT_IDX); i++) {
        for (size_t j = 0;
             j < weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_CIN_IDX); j++) {
          for (size_t k = 0;
               k < weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_KY_IDX); k++) {
            for (size_t l = 0;
                 l < weight_tensor.shape.at(WEIGHT_TENSOR_SHAPE_KX_IDX); l++) {
              // dst[i][k][l][j] = src[i][j][k][l]
              dst_p[i * dest_stride_i + k * dest_stride_k + l * dest_stride_l +
                    j] = *src_p++;
            }
          }
        }
      }
    }

    std::vector<BiasT> bias_data(bias_tensor_volume_padded, 0);
    memcpy(bias_data.data(), bias_tensor.data,
           bias_tensor_volume * sizeof(BiasT));

    const size_t BIAS_SIZE = bias_tensor_volume * sizeof(BiasT);

    std::uint8_t *dest_ptr = static_cast<std::uint8_t *>(dest);

    memcpy(dest_ptr, bias_data.data(), BIAS_SIZE);

    dest_ptr += BIAS_SIZE;

    std::vector<char> shuffled_weight_data =
        qdq_conv_data_shuffle(weights, bias_data, layer_info_buf, param);
    // for qconv, buffer layout will be [bias, layer_params, weights_data]
    // and shuffled_weight_data should be [layer_params, weights_data]

    const size_t SHUFFLED_WEIGHT_SIZE =
        shuffled_weight_data.size() * sizeof(char);

    memcpy(dest_ptr, shuffled_weight_data.data(), SHUFFLED_WEIGHT_SIZE);

  } else {
    // TO DO: how to populate layer param/weights/bias into param BO
    // currently just hardcoded path and ignore passed in vector

    const std::string param_path =
        OpInterface::get_dod_base_dir() + "/bin/conv_case/param.bin";

    std::vector<std::uint8_t> param_bin_vec =
        OpsFusion::read_bin_file<std::uint8_t>(param_path);

    memcpy(dest, param_bin_vec.data(), param_bin_vec.size());
  }

  RYZENAI_LOG_TRACE("Conv2d initialize_const_params(ptr) ... DONE");
}

// this will mainly be called through unit test
template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
void conv2d<InT, WtT, BiasT, OutT, DWC>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  if (const_params.size() != 2 && const_params.size() != 3) {
    throw std::runtime_error(
        "Conv2d IPU Wrapper expect to have 2 or 3 constants "
        "[qdq_params, weights, bias (optional)] passed.");
  }

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input (activations), weight, scratch, output BO */
  const int64_t INPUT_BO_SIZE =
      input_padded_shape_[0] * input_padded_shape_[1] * input_padded_shape_[2] *
      input_padded_shape_[3] * a_dtype_size_;

  int64_t PARAM_BO_SIZE = conv2d::LAYER_PARAM_SIZE +
                          weights_padded_shape_[0] * weights_padded_shape_[1] *
                              weights_padded_shape_[2] *
                              weights_padded_shape_[3] * b_dtype_size_ +
                          bias_padded_shape[0] * b_dtype_size_;

  // HACK - HARD code this value for now
  // TO DO: add another file to read from
  PARAM_BO_SIZE = 163840;

  // TO DO: add another file to read from
  const int64_t SCRATCH_BO_SIZE = 2097152;

  const int64_t OUTPUT_BO_SIZE =
      output_padded_shape_[0] * output_padded_shape_[1] *
      output_padded_shape_[2] * output_padded_shape_[3] * c_dtype_size_;

  RYZENAI_LOG_TRACE(
      "XCOM::CONV2D: INPUT_BO_SIZE:" + std::to_string(INPUT_BO_SIZE) +
      " PARAM_BO_SIZE:" + std::to_string(PARAM_BO_SIZE) +
      " SCRATCH_BO_SIZE:" + std::to_string(SCRATCH_BO_SIZE) +
      " OUTPUT_BO_SIZE:" + std::to_string(OUTPUT_BO_SIZE));

  input_bo_ =
      xrt::bo(xrt_ctx_->get_device(), INPUT_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(8));

  param_bo_ =
      xrt::bo(xrt_ctx_->get_device(), PARAM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(8));

  scratch_bo_ =
      xrt::bo(xrt_ctx_->get_device(), SCRATCH_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(8));

  output_bo_ =
      xrt::bo(xrt_ctx_->get_device(), OUTPUT_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(8));

  // populate weights BO
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;

  auto b_copy_start = GET_ELAPSED_TIME_NS();
  std::uint8_t *param_bo_ptr = param_bo_.map<std::uint8_t *>();
  initialize_const_params(param_bo_ptr, const_params);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = b_copy_stop - b_copy_start;
  b_format_time_ = b_copy_time_;

  // sync b_bo
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  param_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ = b_sync_stop - b_sync_start;
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
void conv2d<InT, WtT, BiasT, OutT, DWC>::execute(
    const std::vector<Tensor> &input, std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    DOD_THROW("Conv2D IPU Wrapper expect to have one input.");
  }

  if (output.size() != 1) {
    DOD_THROW("Conv2D IPU Wrapper expect to have one output.");
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

  int64_t exec_start = GET_ELAPSED_TIME_NS();

  const int64_t INPUT_BO_SIZE =
      input_padded_shape_[0] * input_padded_shape_[1] * input_padded_shape_[2] *
      input_padded_shape_[3] * a_dtype_size_;

  RYZENAI_LOG_TRACE("Conv2D: INPUT_BO_SIZE:" + std::to_string(INPUT_BO_SIZE));
  // populate activations
  int64_t a_copy_start = GET_ELAPSED_TIME_NS();
  std::uint8_t *input_bo_ptr = input_bo_.map<std::uint8_t *>();
  constexpr std::uint32_t act_index = 0;
  // The first data is a and second data is b
  std::uint8_t *a = (std::uint8_t *)input.at(act_index).data;
  memcpy((void *)input_bo_ptr, (void *)a, INPUT_BO_SIZE);

  int64_t a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  int64_t a_sync_start = GET_ELAPSED_TIME_NS();
  input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  int64_t a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = a_copy_stop - a_copy_start;
  a_sync_time_ = a_sync_stop - a_sync_start;

  // prepare inst_bo
  auto txn_fname_prefix =
      txn_fname_prefix_ + "_" + std::to_string(kernel_y_dim_) + "x" +
      std::to_string(kernel_x_dim_) + "_s" + std::to_string(stride_x_);
  auto instr_bo_key = get_instr_key(txn_fname_prefix, padded_shape_info_);

  const xrt::bo &instr_bo = instr_reg_.get_instr_bo(instr_bo_key).second;
  int instr_bo_words = instr_bo.size() / sizeof(int);
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  output_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                input_bo_.address() + DDR_AIE_ADDR_OFFSET,
                param_bo_.address() + DDR_AIE_ADDR_OFFSET,
                output_bo_.address() + DDR_AIE_ADDR_OFFSET,
                scratch_bo_.address() + DDR_AIE_ADDR_OFFSET,
                0); // TO DO: this is shim-dma control packets
                    // will need to remap arg_idx 5 to 4 when converting from
                    // DPU sequence to TXN sequence
  run.wait2();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += run_aie_stop - run_aie_start;
  num_run_aie_++;

  // sync output activation to host memory
  int64_t c_sync_start = GET_ELAPSED_TIME_NS();
  output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  int64_t c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += c_sync_stop - c_sync_start;

  int64_t c_copy_start = GET_ELAPSED_TIME_NS();
  // copy c_bo to host memory
  auto aie_out = (std::uint8_t *)output.at(0).data;
  std::uint8_t *output_ptr = output_bo_.map<std::uint8_t *>();
  const int64_t OUTPUT_BO_SIZE =
      output_padded_shape_[0] * output_padded_shape_[1] *
      output_padded_shape_[2] * output_padded_shape_[3] * c_dtype_size_;

  memcpy((void *)aie_out, (void *)output_ptr, OUTPUT_BO_SIZE);

  int64_t c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = c_copy_stop - c_copy_start;
  int64_t exec_end = c_copy_stop;

  RYZENAI_LOG_INFO(
      std::to_string(id_) + " " + std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
const std::vector<uint8_t>
conv2d<InT, WtT, BiasT, OutT, DWC>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {

  // shapes will be in [NCHW] format
  const auto &input_shape =
      std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
  const auto &output_shape =
      std::any_cast<const std::vector<int> &>(attr.at("output_shape"));
  const auto &kernel_shape =
      std::any_cast<const std::vector<int> &>(attr.at("kernel_shape"));
  const auto &strides =
      std::any_cast<const std::vector<int> &>(attr.at("strides"));

  // TODO: see if these need to configurable
  // always assume bias is enabled
  // bias_en_ = true;
  // always assume relu is enabled
  // relu_en_ = true;
  constexpr std::uint32_t TENSOR_SHAPE_C_INDEX = 1;
  constexpr std::uint32_t TENSOR_SHAPE_H_INDEX = 2;
  constexpr std::uint32_t TENSOR_SHAPE_W_INDEX = 3;
  //[H_in, W_in, C_in, C_out]
  conv_shape_t shape_info = {input_shape.at(TENSOR_SHAPE_H_INDEX),
                             input_shape.at(TENSOR_SHAPE_W_INDEX),
                             input_shape.at(TENSOR_SHAPE_C_INDEX),
                             output_shape.at(TENSOR_SHAPE_C_INDEX)};

  auto prefix = txn_fname_prefix_ + "_" + std::to_string(kernel_shape.at(0)) +
                "x" + std::to_string(kernel_shape.at(1)) + "_s" +
                std::to_string(strides.at(0));

  std::string txn_key = get_instr_key(prefix, shape_info);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(txn_stream)),
                            std::istreambuf_iterator<char>());

  constexpr bool is_qdq = std::is_same_v<BiasT, std::int32_t>;

  if constexpr (is_qdq) {
    // need to pre-pend zero point
    // Note: this index is based on activations being first input
    constexpr std::uint32_t ACT_ZP_INDEX = 2;
    using zp_t = std::uint16_t;
    std::uint32_t act_zero_point =
        *static_cast<zp_t *>(input.at(ACT_ZP_INDEX).data);

    std::uint32_t pad_val = (act_zero_point << 16) | act_zero_point;
    // these values are dependent on design
    // direction will be mm2s i.e. 1
    // assumption is same padding value is used for entire overlay
    constexpr std::uint8_t NUM_CHAN = 4;
    constexpr std::uint8_t NUM_COLS = 4;
    data =
        ryzenai::prepend_mtile_const_pad_txn(data, pad_val, NUM_CHAN, NUM_COLS);
  }

  return data;
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
const std::vector<uint8_t>
conv2d<InT, WtT, BiasT, OutT, DWC>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {

  // TO DO: for xcompiler, only have layer params in weights buffer, this might
  // not be needed
  return {};
}

template <typename InT, typename WtT, typename BiasT, typename OutT, bool DWC>
std::vector<OpArgMap> conv2d<InT, WtT, BiasT, OutT, DWC>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {

  // shapes will be in [NCHW] format
  const auto &input_shape =
      std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
  const auto &output_shape =
      std::any_cast<const std::vector<int> &>(attr.at("output_shape"));
  const auto &kernel_shape =
      std::any_cast<const std::vector<int> &>(attr.at("kernel_shape"));
  const auto &strides =
      std::any_cast<const std::vector<int> &>(attr.at("strides"));

  constexpr std::uint32_t TENSOR_SHAPE_C_INDEX = 1;
  constexpr std::uint32_t TENSOR_SHAPE_H_INDEX = 2;
  constexpr std::uint32_t TENSOR_SHAPE_W_INDEX = 3;
  //[H_in, W_in, C_in, C_out]
  conv_shape_t shape_info = {input_shape.at(TENSOR_SHAPE_H_INDEX),
                             input_shape.at(TENSOR_SHAPE_W_INDEX),
                             input_shape.at(TENSOR_SHAPE_C_INDEX),
                             output_shape.at(TENSOR_SHAPE_C_INDEX)};

  auto prefix = txn_fname_prefix_ + "_" + std::to_string(kernel_shape.at(0)) +
                "x" + std::to_string(kernel_shape.at(1)) + "_s" +
                std::to_string(strides.at(0));

  const std::string ddr_buffer_info_key =
      get_instr_key(prefix, shape_info) + "_ddr_buffer_info";

  const auto ddr_buffer_info_bvec = get_kernel_params(ddr_buffer_info_key);
  DOD_ASSERT(ddr_buffer_info_bvec.size() == sizeof(ddr_buffer_info_s),
             "Unexpected ddr buffer info size");

  ddr_buffer_info_s ddr_buffer_info;
  memcpy(&ddr_buffer_info, ddr_buffer_info_bvec.data(),
         sizeof(ddr_buffer_info_s));

  const size_t INPUT_BO_SIZE = ddr_buffer_info.ifm_size;
  const size_t PARAM_BO_SIZE = ddr_buffer_info.param_size;
  const size_t SCRATCH_BO_SIZE = ddr_buffer_info.inter_size;
  const size_t OUTPUT_BO_SIZE = ddr_buffer_info.ofm_size;

  const size_t INPUT_OFFSET = ddr_buffer_info.ifm_addr;

  // QDQ conv
  if constexpr (std::is_same_v<BiasT, std::int32_t> && !DWC) {

    const std::string layer_param_key =
        get_instr_key(prefix, shape_info) + "_param";
    const std::string weight_shuffle_param_key =
        get_instr_key(prefix, shape_info) + "_weight_format_param";

    const auto layer_param_bvec = get_kernel_params(layer_param_key);
    DOD_ASSERT(layer_param_bvec.size() == LAYER_PARAM_SIZE,
               "Unexpected layer param size");
    const auto weight_shuffle_bvec =
        get_kernel_params(weight_shuffle_param_key);
    DOD_ASSERT(weight_shuffle_bvec.size() == sizeof(WGTShuffleParam),
               "Unexpected weight shuffle param size");

    std::vector<std::int32_t> layer_info_buf(layer_param_bvec.size() /
                                             sizeof(std::int32_t));

    // Need to read actual params, since right now we do not have a way
    // to estimate buffer size needed, so do a dummy weight shuffle
    // TODO: use/add API to will just figure out size needed
    memcpy(layer_info_buf.data(), layer_param_bvec.data(),
           layer_param_bvec.size());

    WGTShuffleParam param;
    memcpy(&param, weight_shuffle_bvec.data(), sizeof(WGTShuffleParam));

    constexpr std::uint32_t WEIGHT_INDEX = 3;
    constexpr std::uint32_t BIAS_INDEX = 6;

    const auto &weight_tensor = input.at(WEIGHT_INDEX);
    const auto &bias_tensor = input.at(BIAS_INDEX);

    size_t weight_tensor_volume =
        std::accumulate(weight_tensor.shape.begin(), weight_tensor.shape.end(),
                        size_t{1}, std::multiplies{});

    size_t bias_tensor_volume =
        std::accumulate(bias_tensor.shape.begin(), bias_tensor.shape.end(),
                        size_t{1}, std::multiplies{});

    // pad up to multiple of 4
    size_t bias_tensor_volume_padded = 4 * ((bias_tensor_volume + 3) / 4);

    std::vector<char> weights(weight_tensor_volume * sizeof(WtT));

    std::vector<BiasT> bias_data(bias_tensor_volume_padded, 0);

    const size_t BIAS_SIZE = bias_tensor_volume * sizeof(BiasT);

    std::vector<char> shuffled_weight_data =
        qdq_conv_data_shuffle(weights, bias_data, layer_info_buf, param);

    // for qconv, buffer layout will be [bias, layer_params, weights_data]
    // and shuffled_weight_data should be [layer_params, weights_data]
    const size_t SHUFFLED_WEIGHT_SIZE =
        shuffled_weight_data.size() * sizeof(char);

    size_t CALC_PARAM_BO_SIZE = BIAS_SIZE + SHUFFLED_WEIGHT_SIZE;
    DOD_ASSERT(CALC_PARAM_BO_SIZE <= PARAM_BO_SIZE,
               "Reformatted weights buffer too small compared to meta!");
  }

  RYZENAI_LOG_TRACE("XCOM::CONV2D:get_buffer_reqs INPUT_BO_SIZE:" +
                    std::to_string(INPUT_BO_SIZE) +
                    " INPUT_OFFSET: " + std::to_string(INPUT_OFFSET) +
                    " PARAM_BO_SIZE:" + std::to_string(PARAM_BO_SIZE) +
                    " SCRATCH_BO_SIZE:" + std::to_string(SCRATCH_BO_SIZE) +
                    " OUTPUT_BO_SIZE:" + std::to_string(OUTPUT_BO_SIZE));

  //[OpArgType,  xrt_arg_idx, onnx_arg_idx, offset_into_bo, size_in_bytes]
  // NOTE: removed scratch pad since it seems unused in txn
  // HACK: attempt at adding padding
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, INPUT_BO_SIZE, INPUT_OFFSET},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 3, 0, PARAM_BO_SIZE},
      {OpArgMap::OpArgType::OUTPUT, 2, 11, 0, OUTPUT_BO_SIZE}};
  // TO DO: how to expose buffer requirements for scratch buffer and control
  // packet? control packets will need patching as transactions get fused

  return arg_map;
}

// regular conv
template class conv2d<std::int8_t, std::int8_t, std::int8_t, std::int8_t,
                      false>;
template class conv2d<std::int8_t, std::int8_t, std::int8_t, std::int8_t, true>;

// qconv
template class conv2d<std::uint16_t, std::uint8_t, std::int32_t, std::uint16_t,
                      false>;
template class conv2d<std::uint16_t, std::uint8_t, std::int32_t, std::uint16_t,
                      true>;

} // namespace xcom
} // namespace ryzenai
