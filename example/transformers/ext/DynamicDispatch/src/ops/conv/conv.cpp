/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#ifndef _WIN32
#include <cmath>
#endif

#include <iomanip>
#include <iterator>
#include <string>

#include <ops/conv/conv.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

namespace ryzenai {
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<conv_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key =
        "conv_" + get_instr_key(txn_fname_prefix_, mat.Z, mat.F, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  instr_reg_.setup_hw_ctx(xrt_ctx_);
  instr_reg_.add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string conv<InT, WtT, OutT>::get_instr_key(std::string prefix, int64_t zp,
                                                int64_t F, int64_t K,
                                                int64_t N) {
  if (zp == NO_ZP) {
    return prefix + "_" + std::to_string(F) + "_" + std::to_string(K) + "_" +
           std::to_string(N);
  } else {
    return prefix + "_" + std::to_string(zp) + "_" + std::to_string(F) + "_" +
           std::to_string(K) + "_" + std::to_string(N);
  }
}

static std::string GetParamKey(std::string prefix, int64_t zp, int64_t K,
                               int64_t N, int64_t F0) {
  return prefix + "_" + std::to_string(zp) + "_" + std::to_string(F0) + "_" +
         std::to_string(K) + "_" + std::to_string(N);
}

/*
 * conv class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base conv
 * supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base conv
 * supported on IPU
 *
 */
template <typename InT, typename WtT, typename OutT>
conv<InT, WtT, OutT>::conv(const std::string &ifmDtype,
                           const std::string &weightDtype,
                           const std::string &ofmDtype, bool load_xrt,
                           const std::map<std::string, std::any> &attr)
    : attr_(attr) {

  /* By default use txn binaries without zp */
  this->useTxnBinWithZp_ = false;

  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_b_header = {{"uint8", "w8"}, {"uint16", "w16"}};
  txnbin_acc_header = {{"uint16", "c16"}};
  xclbin_a_header = {{"int16", "a16"}, {"int8", "a8"}};
  xclbin_b_header = {{"int8", "w8"}, {"uint16", "w16"}};
  xclbin_acc_header = {{"int16", "acc16"}, {"int8", "acc8"}};

  ifmDtype_ = ifmDtype;
  weightDtype_ = weightDtype;
  ofmDtype_ = ofmDtype;
  ifmDtypeSize_ = sizeof(InT);
  weightDtypeSize_ = sizeof(WtT);
  ofmDtypeSize_ = sizeof(OutT);

  conv_id_ = conv_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dod_base_dir() + "\\xclbin\\stx\\ConvDwc.xclbin";

  RYZENAI_LOG_TRACE(OpsFusion::dod_format("xclbin fname : {}", XCLBIN_FNAME));
  txn_fname_prefix_ = "conv_" + txnbin_a_header.at(ifmDtype_) +
                      txnbin_b_header.at(weightDtype_) +
                      txnbin_acc_header.at(ofmDtype_);

  default_shapes_["conv_a16w8c16"] = std::vector<conv_shapes>{};

  /* Shapes for PSI */
  /* TODO: after runtime const padding the first argument below is not needed
   * and should be removed */
  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 3, 512, 512);
  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 3, 1024, 1024);
  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 7, 4, 128);
  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 3, 128, 128);
  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 3, 256, 256);

  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 3, 256, 512);

  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 3, 128, 256);
  default_shapes_["conv_a16w8c16"].emplace_back(NO_ZP, 3, 512, 1024);

  /* PST Shapes */
  default_shapes_["conv_a16w8c16"].emplace_back(29706, 1, 8, 8);
  default_shapes_["conv_a16w8c16"].emplace_back(19793, 1, 256, 512);
  default_shapes_["conv_a16w8c16"].emplace_back(20675, 1, 128, 256);
  default_shapes_["conv_a16w8c16"].emplace_back(699, 3, 512, 512);

  /* PSS Shapes */
  default_shapes_["conv_a16w8c16"].emplace_back(37147, 1, 512, 256);

#if 0
  /* Under development */

  default_shapes_["conv_a16w8c16"].emplace_back(37529, 1, 256, 128); /* PSS 29*/
  default_shapes_["conv_a16w8c16"].emplace_back(41972, 3, 256, 256); /* PSS 28 */
#endif
  /* Shapes for PSO2 320 */
  default_shapes_["conv_a16w16c16"] = std::vector<conv_shapes>{};

  default_shapes_["conv_a16w16c16"].emplace_back(40597, 3, 8, 16);
  default_shapes_["conv_a16w16c16"].emplace_back(32705, 3, 16, 32);
  default_shapes_["conv_a16w16c16"].emplace_back(36423, 1, 32, 16);
  default_shapes_["conv_a16w16c16"].emplace_back(33409, 3, 16, 32);
  default_shapes_["conv_a16w16c16"].emplace_back(29586, 1, 32, 128);
  default_shapes_["conv_a16w16c16"].emplace_back(25513, 1, 128, 16);
  default_shapes_["conv_a16w16c16"].emplace_back(31530, 3, 16, 32);
  default_shapes_["conv_a16w16c16"].emplace_back(32591, 1, 32, 128);
  default_shapes_["conv_a16w16c16"].emplace_back(31990, 1, 128, 32);
  default_shapes_["conv_a16w16c16"].emplace_back(35326, 1, 48, 256);
  default_shapes_["conv_a16w16c16"].emplace_back(34702, 1, 256, 32);
  default_shapes_["conv_a16w16c16"].emplace_back(30051, 3, 32, 48);
  default_shapes_["conv_a16w16c16"].emplace_back(35719, 1, 48, 256);
  default_shapes_["conv_a16w16c16"].emplace_back(26536, 1, 256, 64);
  default_shapes_["conv_a16w16c16"].emplace_back(22444, 3, 64, 80);
  default_shapes_["conv_a16w16c16"].emplace_back(32234, 1, 80, 512);
  default_shapes_["conv_a16w16c16"].emplace_back(33891, 1, 512, 64);
  default_shapes_["conv_a16w16c16"].emplace_back(33497, 3, 64, 80);
  default_shapes_["conv_a16w16c16"].emplace_back(31960, 1, 80, 512);
  default_shapes_["conv_a16w16c16"].emplace_back(33774, 1, 512, 16);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  ifmCopyTime_ = 0;
  ifmSyncTime_ = 0;
  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;
  ofmCopyTime_ = 0;
  ofmSyncTime_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  /* Attribute Parsing */
  if (attr.count("group") &&
      attr.at("group").type() == typeid(std::vector<int>)) {
    const auto &group_vector =
        std::any_cast<const std::vector<int> &>(attr.at("group"));
    for (const auto &group_id : group_vector) {
      groupId_ = group_id;
    }
  } else {
    std::cout << "Group ID not found or not of correct type." << std::endl;
  }

  if (attr.count("zero_point") &&
      attr.at("zero_point").type() == typeid(std::vector<int>)) {
    const auto &zp_vector =
        std::any_cast<const std::vector<int> &>(attr.at("zero_point"));
    for (const auto &zp : zp_vector) {
      zp_ = zp;
    }
  } else {
    std::cout << "Zero Point not found or not of correct type." << std::endl;
  }

  if (attr.count("weight_shape") &&
      attr.at("weight_shape").type() == typeid(std::vector<int>)) {
    const auto &weight_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("weight_shape"));

    if (weight_shape_vector.size() >= 4) {
      weightShape_[0] = weight_shape_vector[2];
      weightShape_[1] = weight_shape_vector[3];
      weightShape_[2] = weight_shape_vector[1];
      weightShape_[3] = weight_shape_vector[0];
    } else {
      std::cout << "Weight Shape attribute does not have enough elements."
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: WeightShape: " + std::to_string(weight_shape_vector[0]) + ", " +
        std::to_string(weight_shape_vector[1]) + ", " +
        std::to_string(weight_shape_vector[2]) + ", " +
        std::to_string(weight_shape_vector[3]));
  } else {
    std::cout << "Weight Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 4) {
      inputShape_[0] = input_shape_vector[1];
      if ((inputShape_[0] <= 4) && (zp_ == 29172)) {
        /* This is a specific work around required for layer 1 of PSI model */
        inputShape_[0] = 4;
      }
      inputShape_[1] = input_shape_vector[2];
      inputShape_[2] = input_shape_vector[3];
    } else {
      std::cout
          << "Input Shape attribute does not have the expected number of "
             "elements.Number of passed : input_shape_vector.size(), Expected:4"
          << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
        std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]) + ", " +
        std::to_string(input_shape_vector[3]));
  } else {
    std::cout << "Input Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 4) {
      outputShape_[0] = output_shape_vector[1];
      outputShape_[1] = output_shape_vector[2];
      outputShape_[2] = output_shape_vector[3];
    } else {
      std::cout << "Output Shape attribute does not have the expected number "
                   "of elements.Number of passed : input_shape_vector.size(), "
                   "Expected:4"
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: OutputShape: " + std::to_string(output_shape_vector[0]) + ", " +
        std::to_string(output_shape_vector[1]) + ", " +
        std::to_string(output_shape_vector[2]) + ", " +
        std::to_string(output_shape_vector[3]));
  } else {
    std::cout << "Output Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("width") &&
      attr.at("width").type() == typeid(std::vector<int>)) {
    const auto &width_vector =
        std::any_cast<const std::vector<int> &>(attr.at("width"));
    for (const auto &width : width_vector) {
      convData_ = "convData" + std::to_string(width) + "_layer";
    }
  } else {
    convData_ = "convData_layer";
  }

  if (inputShape_[0] >= 8) {
    kernelInputShape_[0] = inputShape_[1];
    kernelInputShape_[1] = inputShape_[0] / 8;
    kernelInputShape_[2] = inputShape_[2];
    kernelInputShape_[3] = 8;
  } else {
    kernelInputShape_[0] = inputShape_[1];
    kernelInputShape_[1] = inputShape_[0] / 4;
    kernelInputShape_[2] = inputShape_[2];
    kernelInputShape_[3] = 4;
  }

  kernelWeightShape_[0] = weightShape_[3];
  kernelWeightShape_[1] = weightShape_[2];
  kernelWeightShape_[2] = weightShape_[0];
  kernelWeightShape_[3] = weightShape_[1];

  kernelOutputShape_[0] = outputShape_[1];
  kernelOutputShape_[1] = outputShape_[0] / 8;
  kernelOutputShape_[2] = outputShape_[2];
  kernelOutputShape_[3] = 8;

  lp.resize(64);
  std::string lp_key = GetParamKey(convData_, zp_, inputShape_[0],
                                   outputShape_[0], weightShape_[0]) +
                       "_lp";
  std::string lp_binary = txn_handler.get_txn_str(lp_key);
  txn_handler.GetBinData(lp_binary, lp, false);
  foldWts_ = lp[19];

  std::call_once(logger_flag_, []() {
    std::string header = "conv_id (Mi0 Mi1 F0 F1 K N Mo0 Mo1) Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[Conv] ID: " + std::to_string(conv_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
                    ifmDtype_ + ", " + weightDtype_ + ", " + ofmDtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::WriteToFile(void *src, uint64_t length) {
  uint8_t *dataPtr = (uint8_t *)src;
  std::string testDataFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" + "GeneratedWeights";
  std::string fileName = testDataFolder + "\\" +
                         GetParamKey("wtsGenerated", zp_, inputShape_[0],
                                     outputShape_[0], weightShape_[0]) +
                         ".txt";
  std::ofstream wts32_fp(fileName);

  for (int i = 0; 4 * i + 3 < length; i++) {
    // print 4 nibbles (in reverse order)
    for (int j = 3; j >= 0; j--) {
      wts32_fp << setw(1) << hex << ((dataPtr[4 * i + j] & 0xF0) >> 4)
               << setw(0);
      wts32_fp << setw(1) << hex << (dataPtr[4 * i + j] & 0x0F) << setw(0);
    }
    wts32_fp << endl;
  }
  wts32_fp.close();
}

/* Concat weight params for convA16W8 */
template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::dumpBinary(void *src, size_t length,
                                      std::string &filePath) {
  std::ofstream ofs(filePath, std::ios::binary);
  size_t chunk_size = 1024;
  char *ptr = (char *)src;
  for (int i = 0; i < length / chunk_size; ++i) {
    ofs.write((char *)src, 1024);
    ptr += chunk_size;
  }
  ofs.write(ptr, length % chunk_size);
}

template <typename InT, typename WtT, typename OutT>
int64_t conv<InT, WtT, OutT>::ConcatenateWeightParams(
    void *dest, const std::vector<WtsListType> &wts_list,
    const std::vector<std::vector<int32_t>> &qdq_list, /* uint8_t *lp,*/
    int ifm_depth, int cstride, int ksize_x, int ksize_y) {

  auto concatenateWeightParamsLength = 0;
  uint8_t *dstWeightBuffer = (uint8_t *)dest;

  for (size_t it = 0; it < wts_list.size(); ++it) {
    std::vector<WtT> w_vals;

    memcpy(dstWeightBuffer, lp.data(), 64 * sizeof(uint8_t));
    concatenateWeightParamsLength += 64 * sizeof(uint8_t);
    dstWeightBuffer += 64 * sizeof(uint8_t);

    int istride = std::min(ifm_depth, cstride);
    int ostride = 8;
    int Cout = wts_list[it].size();
    int Cin = wts_list[it][0].size();

    auto wtsListMember = wts_list[it];
    for (int o = 0; o < Cout; o += ostride) {
      for (int i = 0; i < Cin; i += istride) {
        for (int y = 0; y < ksize_y; ++y) {
          for (int x = 0; x < ksize_x; ++x) {
            for (int i_idx = i; i_idx < i + istride; ++i_idx) {
              for (int o_idx = 0; o_idx < 8; ++o_idx) {
                auto w_val = wts_list[it][o + o_idx][i_idx][y][x];
                w_vals.push_back(w_val);
              }
            }
          }
        }
      }
    }

    memcpy(dstWeightBuffer, w_vals.data(), sizeof(WtT) * w_vals.size());
    concatenateWeightParamsLength += sizeof(WtT) * w_vals.size();
    dstWeightBuffer += sizeof(WtT) * w_vals.size();

    memcpy(dstWeightBuffer, qdq_list[it].data(),
           sizeof(int32_t) * qdq_list[it].size());
    concatenateWeightParamsLength += sizeof(int32_t) * qdq_list[it].size();
    dstWeightBuffer += sizeof(int32_t) * qdq_list[it].size();
  }
  return concatenateWeightParamsLength;
}

/* Concat weight params for convA16W16 */
template <typename InT, typename WtT, typename OutT>
int64_t conv<InT, WtT, OutT>::ConcatenateWeightParams(
    void *dest, const std::vector<WtsListType> &wts_list,
    const std::vector<uint8_t> &conv_lp,
    const std::vector<std::vector<int32_t>> &qdq_list,
    /* uint8_t *lp,*/
    int ifm_depth, int cstride, int ksize_x, int ksize_y) {

  auto concatenateWeightParamsLength = 0;
  uint8_t *dstWeightBuffer = (uint8_t *)dest;
  for (size_t it = 0; it < wts_list.size(); ++it) {
    int id = 0;
    std::vector<WtT> w_vals;

    // dd lp
    memcpy(dstWeightBuffer, lp.data(), 64 * sizeof(uint8_t));
    concatenateWeightParamsLength += 64 * sizeof(uint8_t);
    dstWeightBuffer += 64 * sizeof(uint8_t);

    // conv lp
    memcpy(dstWeightBuffer, conv_lp.data(), 64 * sizeof(uint8_t));
    concatenateWeightParamsLength += 64 * sizeof(uint8_t);
    dstWeightBuffer += 64 * sizeof(uint8_t);

    // qdq
    memcpy(dstWeightBuffer, qdq_list[it].data(),
           sizeof(int32_t) * qdq_list[it].size());
    concatenateWeightParamsLength += sizeof(int32_t) * qdq_list[it].size();
    dstWeightBuffer += sizeof(int32_t) * qdq_list[it].size();

    int istride = std::min(ifm_depth, cstride);
    int ostride = 8;
    int Cout = wts_list[it].size();
    int Cin = wts_list[it][0].size();

    auto wtsListMember = wts_list[it];
    for (int o = 0; o < Cout; o += ostride) {
      for (int i = 0; i < Cin; i += istride) {
        for (int y = 0; y < ksize_y; ++y) {
          for (int x = 0; x < ksize_x; ++x) {
            for (int i_idx = i; i_idx < i + istride; ++i_idx) {
              for (int o_idx = 0; o_idx < 8; ++o_idx) {
                auto w_val = wts_list[it][o + o_idx][i_idx][y][x];
                w_vals.push_back(w_val);
              }
            }
          }
        }
      }
    }

    memcpy(dstWeightBuffer, w_vals.data(), sizeof(WtT) * w_vals.size());
    concatenateWeightParamsLength += sizeof(WtT) * w_vals.size();
    dstWeightBuffer += sizeof(WtT) * w_vals.size();
  }
  return concatenateWeightParamsLength;
}

template <typename InT, typename WtT, typename OutT>
std::vector<int32_t> conv<InT, WtT, OutT>::qdq_header(
    int64_t *qdq, int32_t qdqParams[], int32_t ofm_height, int32_t ofm_width,
    int32_t ofm_depth_start, int32_t ofm_depth_end) {
  int64_t *c0 = qdq;
  int32_t c1 = qdqParams[0];
  int32_t c2 = qdqParams[1];
  int32_t output_depth_gran = 16;

  std::vector<int32_t> header;
  for (int32_t i = ofm_depth_start; i < ofm_depth_end; i++) {
    int64_t ci = c0[i];
    int64_t temp = static_cast<uint64_t>(ci) & 0x00000000FFFFFFFF;
    header.push_back(static_cast<int32_t>(temp));
    temp = (static_cast<uint64_t>(ci) >> 32) & 0x00000000FFFFFFFF;
    header.push_back(static_cast<int32_t>(temp));
  }

  std::vector<int32_t> qdq_config{c1, c2};
  header.insert(header.end(), qdq_config.begin(), qdq_config.end());
  return header;
}

/* qdq header for convA16W16 */
template <typename InT, typename WtT, typename OutT>
std::vector<int32_t>
conv<InT, WtT, OutT>::qdq_header(int64_t *qdq, int32_t ofm_height,
                                 int32_t ofm_width, int32_t ofm_depth_start,
                                 int32_t ofm_depth_end) {
  int64_t *c0 = qdq;
  int32_t output_depth_gran = 16;
  int32_t ofm_count = (ofm_depth_end - ofm_depth_start) / output_depth_gran;

  std::vector<int32_t> header;
  for (int32_t i = ofm_depth_start; i < ofm_depth_end; i++) {
    int64_t ci = c0[i];
    int64_t temp = static_cast<uint64_t>(ci) & 0x00000000FFFFFFFF;
    header.push_back(static_cast<int32_t>(temp));
    temp = (static_cast<uint64_t>(ci) >> 32) & 0x00000000FFFFFFFF;
    header.push_back(static_cast<int32_t>(temp));
  }

  return header;
}
/* Below is required for zero padding for 7x7 kernel with stride 4 only */
template <typename InT, typename WtT, typename OutT>
std::vector<std::vector<std::vector<std::vector<WtT>>>>
conv<InT, WtT, OutT>::TransformWts(const WtsListType &wts, uint8_t ksize_x,
                                   uint8_t ksize_y, uint8_t wts_zp) {
  WtsListType wt_pad(wts.size(),
                     std::vector<std::vector<std::vector<WtT>>>(
                         wts[0].size() + 1,
                         std::vector<std::vector<WtT>>(
                             ksize_y, std::vector<WtT>(ksize_x + 1, wts_zp))));

  for (size_t i = 0; i < wts.size(); i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t y = 0; y < ksize_y; y++) {
        for (size_t x = 0; x < ksize_x; x++) {
          wt_pad[i][j][y][x] = wts[i][j][y][x];
        }
      }
    }
  }

  for (size_t i = 0; i < wts.size(); i++) {
    for (size_t y = 0; y < ksize_y; y++) {
      for (size_t x = 0; x < ksize_x; x++) {
        wt_pad[i][3][y][x] = 0;
      }
    }
  }

  WtsListType wt_new(wts.size(),
                     std::vector<std::vector<std::vector<WtT>>>(
                         (wts[0].size() + 1) * 2,
                         std::vector<std::vector<WtT>>(
                             ksize_y, std::vector<WtT>((ksize_x + 1) / 2, 0))));

  for (size_t i = 0; i < wts.size(); i++) {
    for (size_t k = 0; k < (ksize_x + 1) / 2; k++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t y = 0; y < ksize_y; y++) {
          wt_new[i][j][y][k] = wt_pad[i][j][y][2 * k];
          wt_new[i][j + 4][y][k] = wt_pad[i][j][y][2 * k + 1];
        }
      }
    }
  }

  return wt_new;
}

/* if the convolution kernel is 1x1 and the number of channels is less than 64,
 * than the ifm is zero padded. Accordingly the weights should be padded with zp
 * value in the layer parameter */
template <typename InT, typename WtT, typename OutT>
std::vector<std::vector<std::vector<std::vector<WtT>>>>
conv<InT, WtT, OutT>::TransformWtsWithZp(const WtsListType &wtsOld) {
  WtsListType wts(
      kernelWeightShape_[0],
      vector<vector<vector<WtT>>>(
          64, vector<vector<WtT>>(kernelWeightShape_[2],
                                  vector<WtT>(kernelWeightShape_[3]))));

  // Copy existing elements from 'wtsOld' to 'wts'
  for (size_t i = 0; i < wtsOld.size(); ++i) {
    for (size_t j = 0; j < wtsOld[i].size(); ++j) {
      wts[i][j] = wtsOld[i][j];
    }
  }

  auto wts_zp = lp[46];
  // Initialize the new members with 'zp'
  for (size_t i = 0; i < wts.size(); ++i) {
    for (size_t j = wtsOld[i].size(); j < wts[i].size(); ++j) {
      for (size_t k = 0; k < wts[i][j].size(); ++k) {
        for (size_t l = 0; l < wts[i][j][k].size(); ++l) {
          wts[i][j][k][l] = wts_zp;
        }
      }
    }
  }
  return wts;
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every conv performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU conv implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::initialize_const_params_conv(
    void *dest, const std::vector<Tensor> &const_params) {
  auto ksize_x = weightShape_[0];
  auto ksize_y = weightShape_[1];
  auto stride = (1 << lp[5]);

  auto ofm_depth = outputShape_[0];
  auto ifm_sv_depth = int(lp[2] * 8);
  auto ofm_sv_depth = int(lp[3] * 8);
  auto ofm_sv_width = int((lp[0] - ksize_x + 1) / stride);
  auto ofm_sv_height = int((lp[1] - ksize_y + 1) / stride);

  auto ofm_depth_iters = int(lp[17]);
  auto num_adf_rows = int((ofm_depth / ofm_sv_depth) / ofm_depth_iters);
  auto ch_in_depth_split = int(lp[24]);
  auto num_wt_streams = num_adf_rows;
  auto cout_per_ch_iter = int(ofm_sv_depth * num_adf_rows);
  auto cout_per_stream = int(cout_per_ch_iter / num_wt_streams);
  auto ifm_depth_iters = lp[18];
  auto dp = int(ifm_depth_iters * ifm_sv_depth / ch_in_depth_split);
  auto cstride = (int)8;

  auto wtsIn = (WtT *)const_params.at(0).data;
  auto wtsInShape = const_params.at(0).shape;
  auto wtsInType = const_params.at(0).dtype;
  WtsListType wts;

  WtsListType wtsTensor(
      wtsInShape[0],
      vector<vector<vector<WtT>>>(
          wtsInShape[1],
          vector<vector<WtT>>(wtsInShape[2], vector<WtT>(wtsInShape[3]))));

  for (int i = 0; i < wtsInShape[0]; ++i) {
    for (int j = 0; j < wtsInShape[1]; ++j) {
      for (int k = 0; k < wtsInShape[2]; ++k) {
        for (int l = 0; l < wtsInShape[3]; ++l) {
          wtsTensor[i][j][k][l] =
              wtsIn[(i * wtsInShape[1] * wtsInShape[2] * wtsInShape[3]) +
                    (j * wtsInShape[2] * wtsInShape[3]) + (k * wtsInShape[3]) +
                    l];
        }
      }
    }
  }
  wts = wtsTensor;

  /* Zero padding applicalbe for 7x7/s4*/
  if (foldWts_) {
    auto wts_zp = lp[46];
    wts = TransformWts(wts, ksize_x, ksize_y, wts_zp);
  }

  if ((kernelWeightShape_[2] == 1) && (kernelWeightShape_[3] == 1) &&
      (kernelWeightShape_[1] < 64)) {
    wts = TransformWtsWithZp(wts);
  }

  auto qdqSize = outputShape_[0];
  std::vector<int64_t> qdq;
  qdq.resize(qdqSize);
  std::string qdq_key = GetParamKey(convData_, zp_, inputShape_[0],
                                    outputShape_[0], weightShape_[0]) +
                        "_qdq";
  std::string qdq_binary = txn_handler.get_txn_str(qdq_key);
  txn_handler.GetBinData(qdq_binary, qdq, false);

  int qdqParamsSize = 3;
  std::vector<int32_t> qdqParams;
  qdqParams.resize(qdqParamsSize);

  std::vector<uint8_t> conv_lp;
  conv_lp.resize(64);

  if (wtsInType == "uint16") {
    std::string conv_lp_key = GetParamKey(convData_, zp_, inputShape_[0],
                                          outputShape_[0], weightShape_[0]) +
                              "_convlp";
    std::string conv_lp_params_binary = txn_handler.get_txn_str(conv_lp_key);
    txn_handler.GetBinData(conv_lp_params_binary, conv_lp, false);
  } else {
    std::string qdq_key_params = GetParamKey(convData_, zp_, inputShape_[0],
                                             outputShape_[0], weightShape_[0]) +
                                 "_qdq_params";
    std::string qdq_params_binary = txn_handler.get_txn_str(qdq_key_params);
    txn_handler.GetBinData(qdq_params_binary, qdqParams, false);
  }

  std::vector<WtsListType> wts_list;
  std::vector<std::vector<int32_t>> qdq_list;

  for (int32_t och_iter = 0; och_iter < ofm_depth_iters; och_iter++) {
    auto start_och = och_iter * cout_per_ch_iter;
    auto end_och = (och_iter + 1) * cout_per_ch_iter;
    auto wt_och_iter =
        WtsListType(wts.begin() + start_och, wts.begin() + end_och);

    for (int32_t kk = 0; kk < ch_in_depth_split; kk++) {
      for (int32_t wt_strms = 0; wt_strms < num_wt_streams; wt_strms++) {
        auto start_wt_strms = wt_strms * cout_per_stream;
        auto end_wt_strms = (wt_strms + 1) * cout_per_stream;
        auto wt_strm_data = WtsListType(wt_och_iter.begin() + start_wt_strms,
                                        wt_och_iter.begin() + end_wt_strms);

        int start = 0;
        int end = static_cast<int>(
            ceil(ifm_depth_iters / static_cast<double>(ch_in_depth_split)));

        for (int32_t ich_iter = start; ich_iter < end; ich_iter++) {
          if (wtsInType == "uint16") {
            qdq_list.push_back(qdq_header(
                qdq.data(), ofm_sv_height, ofm_sv_width,
                och_iter * cout_per_ch_iter + wt_strms * cout_per_stream,
                och_iter * cout_per_ch_iter + end_wt_strms));
          } else {
            qdq_list.push_back(qdq_header(
                qdq.data(), qdqParams.data(), ofm_sv_height, ofm_sv_width,
                och_iter * cout_per_ch_iter + wt_strms * cout_per_stream,
                och_iter * cout_per_ch_iter + end_wt_strms));
          }

          WtsListType sub_tensor;
          for (const auto &o : wt_strm_data) {
            std::vector<std::vector<std::vector<WtT>>> wts_o;
            for (int i = kk * dp + ich_iter * ifm_sv_depth;
                 i < kk * dp + (ich_iter + 1) * ifm_sv_depth &&
                 i < static_cast<int>(o.size());
                 i++) {
              wts_o.push_back(o[i]);
            }
            sub_tensor.push_back(wts_o);
          }
          wts_list.push_back(sub_tensor);
        }
      }
    }
  }

  auto concatenateWeightParamsLength = 0;
  if (wtsInType == "uint16") {
    concatenateWeightParamsLength = ConcatenateWeightParams(
        dest, wts_list, conv_lp, qdq_list, wts[0].size(), cstride,
        wts[0][0][0].size(), wts[0][0].size());
  } else {
    concatenateWeightParamsLength =
        ConcatenateWeightParams(dest, wts_list, qdq_list, wts[0].size(),
                                cstride, wts[0][0][0].size(), wts[0][0].size());
  }

  if (debug_ == true) {
    WriteToFile(dest, concatenateWeightParamsLength);
  }
}

template <typename InT, typename WtT, typename OutT>
int64_t conv<InT, WtT, OutT>::ConcatenateWeightParams_dwc(
    void *dest, const std::vector<WtsListType> &wts_list,
    const std::vector<std::vector<int32_t>> &qdq_list, int ifm_depth,
    int cstride, int ksize_x, int ksize_y) {

  auto concatenateWeightParamsLength = 0;
  WtT *dstWeightBuffer = (WtT *)dest;
  for (size_t it = 0; it < wts_list.size(); ++it) {
    std::vector<WtT> w_vals;

    memcpy(dstWeightBuffer, lp.data(), 64 * sizeof(WtT));
    concatenateWeightParamsLength += 64 * sizeof(WtT);
    dstWeightBuffer += 64 * sizeof(WtT);

    int istride = std::min(ifm_depth, cstride);
    int ostride = 8;
    int Cout = wts_list[it].size();
    int Cin = wts_list[it][0].size();

    auto wtsListMember = wts_list[it];
    for (int o = 0; o < Cout; o += ostride) {
      for (int i = 0; i < Cin; i += istride) {
        for (int y = 0; y < ksize_y; ++y) {
          for (int x = 0; x < ksize_x; ++x) {
            for (int o_idx = 0; o_idx < 8; ++o_idx) {
              auto w_val = wts_list[it][o + o_idx][i][y][x];
              w_vals.push_back(w_val);
            }
          }
        }
      }
    }

    int zeroPadForAlignmentLength = (64 - (w_vals.size() % 64));

    memcpy(dstWeightBuffer, w_vals.data(), sizeof(WtT) * w_vals.size());
    concatenateWeightParamsLength +=
        sizeof(WtT) * (w_vals.size() + zeroPadForAlignmentLength);
    dstWeightBuffer +=
        sizeof(WtT) * (w_vals.size() + zeroPadForAlignmentLength);

    int qdqSizeInBytes = qdq_list[it].size() * sizeof(int32_t);
    zeroPadForAlignmentLength = (64 - (qdqSizeInBytes % 64));

    memcpy(dstWeightBuffer, qdq_list[it].data(),
           sizeof(int32_t) * qdq_list[it].size());
    concatenateWeightParamsLength +=
        sizeof(int32_t) * qdq_list[it].size() + zeroPadForAlignmentLength;
    dstWeightBuffer +=
        sizeof(int32_t) * qdq_list[it].size() + zeroPadForAlignmentLength;
  }
  return concatenateWeightParamsLength;
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every dwconv performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU dwconv implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::initialize_const_params_dwc(
    void *dest, const std::vector<Tensor> &const_params) {
  auto ksize_x = weightShape_[0];
  auto ksize_y = weightShape_[1];
  auto stride = (1 << lp[5]);

  auto ofm_depth = outputShape_[0];
  auto ifm_depth = ofm_depth;

  auto ifm_sv_depth = int(lp[2] * 8);
  auto ofm_sv_depth = int(lp[2] * 8);
  auto ofm_sv_width = int((lp[0] - ksize_x + 1) / stride);
  auto ofm_sv_height = int((lp[1] - ksize_y + 1) / stride);

  auto ifm_depth_iters = int(ifm_depth / ifm_sv_depth);
  auto rep_count = int(lp[8]);
  auto cstride = (int)8;

  auto wtsIn = (WtT *)const_params.at(0).data;
  auto wtsInShape = const_params.at(0).shape;
  WtsListType wts;

  WtsListType wtsTensor(
      wtsInShape[0],
      vector<vector<vector<WtT>>>(
          wtsInShape[1],
          vector<vector<WtT>>(wtsInShape[2], vector<WtT>(wtsInShape[3]))));

  for (int i = 0; i < wtsInShape[0]; ++i) {
    for (int j = 0; j < wtsInShape[1]; ++j) {
      for (int k = 0; k < wtsInShape[2]; ++k) {
        for (int l = 0; l < wtsInShape[3]; ++l) {
          wtsTensor[i][j][k][l] =
              wtsIn[(i * wtsInShape[1] * wtsInShape[2] * wtsInShape[3]) +
                    (j * wtsInShape[2] * wtsInShape[3]) + (k * wtsInShape[3]) +
                    l];
        }
      }
    }
  }
  wts = wtsTensor;

  auto qdqSize = outputShape_[0];
  std::vector<int64_t> qdq;
  qdq.resize(qdqSize);
  std::string qdq_key = GetParamKey(convData_, zp_, inputShape_[0],
                                    outputShape_[0], weightShape_[0]) +
                        "_qdq";
  std::string qdq_binary = txn_handler.get_txn_str(qdq_key);
  txn_handler.GetBinData(qdq_binary, qdq, false);

  int qdqParamsSize = 3;
  std::vector<int32_t> qdqParams;
  qdqParams.resize(qdqParamsSize);
  std::string qdq_key_params = GetParamKey(convData_, zp_, inputShape_[0],
                                           outputShape_[0], weightShape_[0]) +
                               "_qdq_params";
  std::string qdq_params_binary = txn_handler.get_txn_str(qdq_key_params);
  txn_handler.GetBinData(qdq_params_binary, qdqParams, false);

  std::vector<WtsListType> wts_list;
  std::vector<std::vector<int32_t>> qdq_list;
  for (int32_t rep = 0; rep < rep_count; rep++) {
    for (int32_t i = 0; i < ifm_depth_iters; i++) {
      auto start_och = i * ofm_sv_depth;
      auto end_och = (i + 1) * ofm_sv_depth;
      auto wt_och_iter =
          WtsListType(wts.begin() + start_och, wts.begin() + end_och);
      wts_list.push_back(wt_och_iter);
      qdq_list.push_back(qdq_header(qdq.data(), qdqParams.data(), ofm_sv_height,
                                    ofm_sv_width, start_och, end_och));
    }
  }

  auto concatenateWeightParamsLength = ConcatenateWeightParams_dwc(
      dest, wts_list, qdq_list, ifm_depth, cstride, ksize_x, ksize_y);
  if (debug_ == true) {
    WriteToFile(dest, concatenateWeightParamsLength);
  }
}

template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv initialize_const_params(ptr) ...");

  if (groupId_ == 1) {
    initialize_const_params_conv(dest, const_params);
  } else {
    initialize_const_params_dwc(dest, const_params);
  }

  RYZENAI_LOG_TRACE("Conv initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv initialize_const_params ...");

  /* Get buffer sizes required for this operators. We are not using input and
   * output tenosrs in get_buffer_req(). So calling with dummy tensors */
  std::vector<Tensor> input;
  std::vector<Tensor> output;
  size_t CONST_BO_SIZE, IFM_BO_SIZE, OFM_BO_SIZE;
  CONST_BO_SIZE = IFM_BO_SIZE = OFM_BO_SIZE = 0;
  auto args_map_list = this->get_buffer_reqs(input, output, attr);
  for (const auto &args_map : args_map_list) {
    if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      CONST_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::INPUT) {
      IFM_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::OUTPUT) {
      OFM_BO_SIZE = args_map.size;
    }
  }
  RYZENAI_LOG_TRACE("Conv: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE));
  constBo_ =
      xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  ifmBo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  ofmBo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));

  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;
  auto weightCopyStart = GET_ELAPSED_TIME_NS();
  auto weightFormatStart = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = constBo_.map<WtT *>();

  initialize_const_params(b_bo_map, const_params, attr);
  auto weightFormatStop = GET_ELAPSED_TIME_NS();
  weightFormatTime_ += weightFormatStop - weightFormatStart;
  auto weightCopyStop = GET_ELAPSED_TIME_NS();
  auto weightSyncStart = GET_ELAPSED_TIME_NS();
  constBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto weightSyncStop = GET_ELAPSED_TIME_NS();
  weightCopyTime_ = weightCopyStop - weightCopyStart;
  weightSyncTime_ = weightSyncStop - weightSyncStart;
  RYZENAI_LOG_TRACE("Conv initialize_const_params ... DONE");
}
/*
 * perform conv c = a * w. w is stored in the object with initilize_weights
 * method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of conv
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                   std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Conv execute ...");

  ifmBo_.write(input.at(0).data);
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto txnData = this->get_transaction_bin();

  aiectrl::op_buf i_buf;
  i_buf.addOP(aiectrl::transaction_op(txnData.data()));
  size_t instr_bo_words = i_buf.ibuf_.size();
  xrt::bo instr_bo =
      xrt::bo(xrt_ctx_->get_context(), instr_bo_words,
              xrt::bo::flags::cacheable, xrt_ctx_->get_kernel().group_id(1));
  instr_bo.write(i_buf.ibuf_.data());
  instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  xrt::run run;

  int64_t run_aie_start = GET_ELAPSED_TIME_NS();
  /* kernel call for Conv that supports transaction binary flow. For single
   * convolution there can't be any time requirement of scratch pad buffer. So
   * in below executiion scratch pad is not used */
  run = kernel_(2, instr_bo, instr_bo_words,
                constBo_.address() + DDR_AIE_ADDR_OFFSET,
                ifmBo_.address() + DDR_AIE_ADDR_OFFSET,
                ofmBo_.address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  run.wait2();
  int64_t run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += run_aie_stop - run_aie_start;

  /* sync output activation to host memory */
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofmBo_.read(output.at(0).data);

  RYZENAI_LOG_TRACE("Conv execute ... DONE");
}

/*
 * method to set debug flag
 *
 * @param debug bool value to enable disable debug feature. turned off by
 * default
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

/**
 * Helper function to read txn binary from file, embed zp in it (if rt_const_pad
 * is true) and return it
 * */
template <typename InT, typename WtT, typename OutT>
std::vector<uint8_t> conv<InT, WtT, OutT>::get_transaction_bin() {
  std::string txn_key =
      "conv_" + txn_fname_prefix_ + "_" +
      (this->useTxnBinWithZp_ ? (std::to_string(zp_) + "_") : "") +
      std::to_string(weightShape_[0]) + "_" + std::to_string(inputShape_[0]) +
      "_" + std::to_string(outputShape_[0]);
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> txnData((std::istreambuf_iterator<char>(txn_stream)),
                               std::istreambuf_iterator<char>());

  if (!this->useTxnBinWithZp_) {
    // Runtime constant padding
    uint32_t zp = uint16_t(zp_);
    uint32_t pad_val = zp | (zp << 16);
    auto paddedTxnData = prepend_mtile_const_pad_txn(txnData, pad_val, 6, 2);
    if (this->debug_) {
      // Dump paddedTxnData
      std::string filePath = OpInterface::get_dod_base_dir() + "\\" + "tests" +
                             "\\" + "cpp" + "\\" + "unit_tests" + "\\" +
                             "testDataMladf" + "\\" + "GeneratedWeights" +
                             GetParamKey("padded_conv", zp_, inputShape_[0],
                                         outputShape_[0], weightShape_[0]) +
                             ".bin";
      if (!paddedTxnData.empty()) {
        dumpBinary(paddedTxnData.data(),
                   paddedTxnData.size() * sizeof(paddedTxnData[0]), filePath);
      }
    }
    return paddedTxnData;
  }
  return txnData;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> conv<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename OutT>
void conv<InT, WtT, OutT>::set_params(const std::string &modelName,
                                      bool useTxnBinWithZp) {
  this->useTxnBinWithZp_ = useTxnBinWithZp;
  std::string XCLBIN_FNAME;
  if (modelName == "psi") {
    XCLBIN_FNAME = OpInterface::get_dod_base_dir() +
                   "\\xclbin\\stx\\ConvDwcGap_Psi.xclbin";
  } else if ((modelName == "pst") || (modelName == "pss")) {
    if (zp_ == 699) { /* Temporary until xclbin is merged */
      XCLBIN_FNAME =
          OpInterface::get_dod_base_dir() +
          "\\xclbin\\stx\\tempXclbinFiles\\conv_699_3_512_512.xclbin";
    } else {
      XCLBIN_FNAME =
          OpInterface::get_dod_base_dir() + "\\xclbin\\stx\\ConvPssPst.xclbin";
    }
  } else if ((modelName == "pso2") || (modelName == "pso2640") ||
             (modelName == "pso21280") || (modelName == "pso22560")) {
    XCLBIN_FNAME = OpInterface::get_dod_base_dir() + "\\xclbin\\" + "stx" +
                   "\\ConvPso2.xclbin";
  }
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> conv<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  int totalWtsSize = 0;

  if (groupId_ == 1) {
    int size_interleaved_qdq = 0;
    if (std::is_same<WtT, uint16_t>::value) {
      size_interleaved_qdq =
          (64 + 64 + 8 * 8 * lp[3]) * (lp[17]) * (lp[18]) * (lp[25]);
    } else {
      size_interleaved_qdq =
          (64 + 8 * 8 * lp[3] + 8) * (lp[17]) * (lp[18]) * (lp[25]);
    }
    if (foldWts_ != 1) {
      if (std::is_same<WtT, uint16_t>::value) {
        totalWtsSize = (kernelWeightShape_[0] * kernelWeightShape_[1] *
                            kernelWeightShape_[2] * kernelWeightShape_[3] *
                            weightDtypeSize_ +
                        size_interleaved_qdq);
      } else {
        if ((kernelWeightShape_[2] == 1) && (kernelWeightShape_[3] == 1)) {
          uint32_t thirdDim;
          if (kernelWeightShape_[1] < 64) {
            thirdDim = 64;
          } else {
            thirdDim = kernelWeightShape_[1];
          }
          totalWtsSize =
              (kernelWeightShape_[0] * thirdDim * kernelWeightShape_[2] *
                   kernelWeightShape_[3] * weightDtypeSize_ +
               size_interleaved_qdq);
        } else {
          totalWtsSize = (kernelWeightShape_[0] * kernelWeightShape_[1] *
                              kernelWeightShape_[2] * kernelWeightShape_[3] *
                              weightDtypeSize_ +
                          size_interleaved_qdq);
        }
      }
    } else {
      totalWtsSize = (kernelWeightShape_[0] * (kernelWeightShape_[1] + 1) *
                          kernelWeightShape_[2] * (kernelWeightShape_[3] + 1) *
                          weightDtypeSize_ +
                      size_interleaved_qdq);
    }
  } else {
    /* TBD: Here wts data type is not considered and assumed it is
     * uint8_t. Need to modify below calculation for other weight data type */
    int wtsLPSizePerSV =
        64 + kernelWeightShape_[2] * kernelWeightShape_[3] * lp[2] * 8;
    wtsLPSizePerSV = wtsLPSizePerSV + (64 - (wtsLPSizePerSV % 64));
    int qdqSizePerSV = 8 * 8 * lp[2] + 8;
    qdqSizePerSV = qdqSizePerSV + (64 - (qdqSizePerSV % 64));
    totalWtsSize = (wtsLPSizePerSV + qdqSizePerSV) * lp[8] *
                   kernelInputShape_[1] * kernelInputShape_[3] / (lp[2] * 8);
  }

  size_t const_params_bo_size = totalWtsSize;
  size_t ifm_bo_size =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  size_t ofm_bo_size =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);
  RYZENAI_LOG_TRACE("Conv: IFM_BO_SIZE:" + std::to_string(ifm_bo_size) +
                    " CONST_BO_SIZE:" + std::to_string(const_params_bo_size) +
                    " OFM_BO_SIZE:" + std::to_string(ofm_bo_size));
  /* conv operator is used in concate as well. Here we are assuming that if each
   * conv layer's output is supposed to spill over in DDR, than each layer needs
   * ifm_bo_size + ofm_bo_size scratch buffer. */
  size_t scratch_bo_size = ifm_bo_size + ofm_bo_size;

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, ofm_bo_size},
      {OpArgMap::OpArgType::SCRATCH_PAD, 3, 0, 0, scratch_bo_size}};

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Conv Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag conv<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t conv<InT, WtT, OutT>::conv_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag conv<InT, WtT, OutT>::instr_reg_flag_;

template class conv<uint16_t, uint8_t, uint16_t>;
template class conv<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
