/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#include <iomanip>
#include <iterator>
#include <string>

#include <ops/maxpool/maxpool.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

namespace ryzenai {
/*
 * Max Pool (maxpool) class constructor
 */
template <typename InT, typename OutT>
maxpool<InT, OutT>::maxpool(const std::string &ifmDtype,
                            const std::string &ofmDtype,
                            const std::map<std::string, std::any> &attr) {
  ifmDtype_ = ifmDtype;
  ofmDtype_ = ofmDtype;
  ifmDtypeSize_ = sizeof(InT);
  ofmDtypeSize_ = sizeof(OutT);

  maxpool_id_ = maxpool_count++;

  /* Params based on attributes */
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

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 4) {
      inputShape_[0] = input_shape_vector[1];
      inputShape_[1] = input_shape_vector[2];
      inputShape_[2] = input_shape_vector[3];
    } else {
      std::cout << "Input Shape attribute does not have the expected number of "
                   "elements."
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Maxpool: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
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
                   "of elements."
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Maxpool: OutputShape: " + std::to_string(output_shape_vector[0]) +
        ", " + std::to_string(output_shape_vector[1]) + ", " +
        std::to_string(output_shape_vector[2]) + ", " +
        std::to_string(output_shape_vector[3]));
  } else {
    std::cout << "Output Shape attribute not found or not of correct type."
              << std::endl;
  }

  std::call_once(logger_flag_, []() {
    std::string header = "Maxpool_id (K Mi0 Mi1) Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[Maxpool] ID: " + std::to_string(maxpool_id_) +
                    ", (a_dtype, c_dtype): (" + ifmDtype_ + ", " + ofmDtype_ +
                    ")");
}

template <typename InT, typename OutT>
void maxpool<InT, OutT>::set_params(const std::string &modelName) {}

template <typename InT, typename OutT>
std::vector<OpArgMap> maxpool<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  size_t const_params_bo_size = 0;
  size_t ifm_bo_size =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  size_t ofm_bo_size =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, ofm_bo_size}};

  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Maxpool Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename OutT>
std::once_flag maxpool<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t maxpool<InT, OutT>::maxpool_count = 0;

template <typename InT, typename OutT>
std::once_flag maxpool<InT, OutT>::instr_reg_flag_;

template class maxpool<uint16_t, uint16_t>;
} // namespace ryzenai
