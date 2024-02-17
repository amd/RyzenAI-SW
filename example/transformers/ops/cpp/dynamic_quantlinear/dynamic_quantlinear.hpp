/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __DYQLINEAR_H__
#define __DYQLINEAR_H__

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>

// tvm headers
#include <dlpack/dlpack.h>

// xrt headers
#include <xrt/xrt_bo.h>

#include "../qlinear/qlinear.hpp"
#include "logging.h"
#include "threadpool.h"
#include "utils.h"

namespace fs = std::filesystem;

namespace ryzenai {

template <typename InT, typename OutT>
class dynamicquantlinear : public qlinear<int8_t, int8_t, int32_t> {
private:
  int64_t _a_shape[2];
  int64_t _b_shape[2];
  float scale_wts_;
  float requantize_out_scale_;
  void input_quant(int8_t *output,
                                                  const InT *input,
                                                  std::tuple<int, int> &shape,
                                                  float scale);
  void output_dequant(
      OutT *output, int32_t *input, std::tuple<int, int> &shape, float scale);
  void log_profile_summary();
public:
  dynamicquantlinear(const std::vector<std::string> &kernel_libs,
                     const std::vector<std::tuple<int, int>> &x_shapes,
                     const std::tuple<int, int> &y_shape, float scale_weights,
                     float requantize_out_scale, size_t nworkers, int num_dlls,
                     int dll_switch_limit);
  dynamicquantlinear(const std::vector<std::string> &kernel_libs,
                     const std::vector<std::tuple<int, int>> &x_shapes,
                     const std::tuple<int, int> &y_shape, float scale_weights,
                     float requantize_out_scale, size_t nworkers, int num_dlls,
                     int dll_switch_limit, const std::string &log_file);
  dynamicquantlinear(const std::string &kernel_lib,
                     const std::tuple<int, int> &x_shape,
                     const std::tuple<int, int> &y_shape, float scale_weights,
                     float requantize_out_scale, size_t nworkers);
  dynamicquantlinear(const std::string &kernel_lib,
                     const std::tuple<int, int> &x_shape,
                     const std::tuple<int, int> &y_shape, float scale_weights,
                     float requantize_out_scale, size_t nworkers,
                     const std::string &log_file);
  ~dynamicquantlinear();
  void
  initialize_weights_bin(std::string bin_file,
                                             std::tuple<int, int> &wt_shape);

  void
  initialize_weights_data(InT *wts,
                                              std::tuple<int, int> &wt_shape);
  void execute_aie(const InT *a,
                                                  std::tuple<int, int> &a_shape,
                                                  OutT *c);
};

template <typename InT, typename OutT>
dynamicquantlinear<InT, OutT>::dynamicquantlinear(
    const std::vector<std::string> &kernel_libs,
    const std::vector<std::tuple<int, int>> &x_shapes,
    const std::tuple<int, int> &y_shape, float scale_weights,
    float requantize_out_scale, size_t nworkers, int num_dlls,
    int dll_switch_limit)
    : qlinear<int8_t, int8_t, int32_t>(kernel_libs, x_shapes, y_shape, nworkers,
                                       num_dlls, dll_switch_limit,
                                       "log_linear_cpp_PROFILE.csv") {
  scale_wts_ = scale_weights;
  requantize_out_scale_ = requantize_out_scale;
}

template <typename InT, typename OutT>
dynamicquantlinear<InT, OutT>::dynamicquantlinear(
    const std::vector<std::string> &kernel_libs,
    const std::vector<std::tuple<int, int>> &x_shapes,
    const std::tuple<int, int> &y_shape, float scale_weights,
    float requantize_out_scale, size_t nworkers, int num_dlls,
    int dll_switch_limit, const std::string &log_file)
    : qlinear<int8_t, int8_t, int32_t>(kernel_libs, x_shapes, y_shape, nworkers,
                                       num_dlls, dll_switch_limit,
                                       "qlinear.csv") {
  scale_wts_ = scale_weights;
  requantize_out_scale_ = requantize_out_scale;
}
template <typename InT, typename OutT>
dynamicquantlinear<InT, OutT>::dynamicquantlinear(
    const std::string &kernel_lib, const std::tuple<int, int> &x_shape,
    const std::tuple<int, int> &y_shape, float scale_weights,
    float requantize_out_scale, size_t nworkers)
    : qlinear<int8_t, int8_t, int32_t>(kernel_lib, x_shape, y_shape, nworkers,
                                       "log_linear_cpp_PROFILE.csv") {
  scale_wts_ = scale_weights;
  requantize_out_scale_ = requantize_out_scale;
}

template <typename InT, typename OutT>
dynamicquantlinear<InT, OutT>::dynamicquantlinear(
    const std::string &kernel_lib, const std::tuple<int, int> &x_shape,
    const std::tuple<int, int> &y_shape, float scale_weights,
    float requantize_out_scale, size_t nworkers, const std::string &log_file)
    : qlinear<int8_t, int8_t, int32_t>(kernel_lib, x_shape, y_shape, nworkers,
                                       "qlinear.csv") {
  scale_wts_ = scale_weights;
  requantize_out_scale_ = requantize_out_scale;
}
template <typename InT, typename OutT>
void dynamicquantlinear<InT, OutT>::initialize_weights_data(
    InT *wts, std::tuple<int, int> &wt_shape) {
  std::vector<int8_t> data(std::get<0>(wt_shape) * std::get<1>(wt_shape));

  _b_shape[0] = std::get<0>(wt_shape);
  _b_shape[1] = std::get<1>(wt_shape);
  for (int i = 0; i < std::get<0>(wt_shape); i++) {
    for (int j = 0; j < std::get<1>(wt_shape); j++)
      data[i * std::get<1>(wt_shape) + j] =
          std::round(wts[i * std::get<1>(wt_shape) + j] / scale_wts_);
  }

  initialize_weights(data.data(), wt_shape);
}
template <typename InT, typename OutT>
void dynamicquantlinear<InT, OutT>::initialize_weights_bin(
    std::string bin_file, std::tuple<int, int> &wt_shape) {
  unsigned int size = (unsigned int)fs::file_size(bin_file);
  auto wts = (int8_t *)malloc(size);
  _b_shape[0] = std::get<0>(wt_shape);
  _b_shape[1] = std::get<1>(wt_shape);
  auto infile = std::ifstream(bin_file, std::ios::in | std::ios::binary);
  for (unsigned i = 0; infile.read(&((char *)wts)[i], sizeof(int8_t)); i++)
    ;
  initialize_weights(wts, wt_shape);
  free(wts);
}

template <typename InT, typename OutT>
dynamicquantlinear<InT, OutT>::~dynamicquantlinear() {}

template <typename InT, typename OutT>
void dynamicquantlinear<InT, OutT>::input_quant(int8_t *output,
                                                const InT *input,
                                                std::tuple<int, int> &shape,
                                                float scale) {
  for (int i = 0; i < std::get<0>(shape); i++) {
    for (int j = 0; j < std::get<1>(shape); j++) {
      auto tmp =
          (int32_t)std::round((input[i * std::get<1>(shape) + j] / scale));
      int8_t data = std::min(std::max(tmp, -128), 127);
      output[i * std::get<1>(shape) + j] = data;
    }
  }
}
template <typename InT, typename OutT>
void dynamicquantlinear<InT, OutT>::output_dequant(OutT *output, int32_t *input,
                                                   std::tuple<int, int> &shape,
                                                   float scale) {
  for (int i = 0; i < std::get<0>(shape); i++) {
    for (int j = 0; j < std::get<1>(shape); j++) {
      output[i * std::get<1>(shape) + j] =
          input[i * std::get<1>(shape) + j] * scale;
    }
  }
}
template <typename InT, typename OutT>
void dynamicquantlinear<InT, OutT>::execute_aie(const InT *a,
                                                std::tuple<int, int> &a_shape,
                                                OutT *c) {
  std::vector<int8_t> input;
  _a_shape[0] = std::get<0>(a_shape);
  _a_shape[1] = std::get<1>(a_shape);
  input.resize(std::get<0>(a_shape) * std::get<1>(a_shape));
  InT max = Utils::abs_max<InT>(a, (int)(_a_shape[0] * _a_shape[1]));
  float x_scale = (float)(max / 128);
  std::vector<int32_t> output;
  output.resize(std::get<0>(a_shape) * _b_shape[1]);
  int64_t input_quant_start = GET_ELAPSED_TIME_NS();
  input_quant(input.data(), a, a_shape, x_scale);
  int64_t input_quant_stop = GET_ELAPSED_TIME_NS();

  auto c_shape = std::make_tuple(std::get<0>(a_shape), (int)_b_shape[1]);
  int64_t exec_start = GET_ELAPSED_TIME_NS();
  execute(input.data(), a_shape, output.data());
  int64_t exec_stop = GET_ELAPSED_TIME_NS();

  int64_t output_dequant_start = GET_ELAPSED_TIME_NS();

  output_dequant(c, output.data(), c_shape,
                 x_scale * scale_wts_ * requantize_out_scale_);

  int64_t output_dequant_stop = GET_ELAPSED_TIME_NS();

  int64_t input_quant_total = input_quant_stop - input_quant_start;
  int64_t output_dequant_total = output_dequant_stop - output_dequant_start;
  int64_t exec_total = exec_stop - exec_start;
  RYZENAI_LOG_INFO(
      std::to_string(_a_shape[0]) + " " + std::to_string(_a_shape[1]) + " " +
      std::to_string(_b_shape[1]) + " " + std::to_string(input_quant_total) +
      " " + std::to_string(exec_total) + " " +
      std::to_string(output_dequant_total) + " ");
}

} // namespace ryzenai

#endif /* __DYQLINEAR_H__ */
