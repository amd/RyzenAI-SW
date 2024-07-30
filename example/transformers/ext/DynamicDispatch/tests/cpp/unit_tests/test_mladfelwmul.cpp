/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/mladfelwmul/mladfelwmul.hpp>

#include "test_common.hpp"
using namespace matmul_matrix;

static size_t layer_params_size = 22;

static uint32_t convert_float_to_qint(float in_f) {
  union {
    float f;
    uint32_t i;
  } u;
  u.f = in_f;
  u.i &= 0x7fffffff; // Remove sign bit
  return u.i;
}

static int get_shift_from_int32_rep(uint32_t rep) {
  // Equivalent to struct.calcsize('i') in Python, which typically returns 4
  int shift = 127 - (((rep >> 23) & 255) + 1) + (8 * sizeof(int) - 2);
  return shift;
}

std::tuple<int, int, int, int> compute_qdq(double ifm1_scale, double ifm2_scale,
                                           double ofm_scale, int ofm_zero_point,
                                           int ifm1_zero_point,
                                           int ifm2_zero_point) {
  float C0 = static_cast<float>(ifm1_scale * ifm2_scale / ofm_scale);
  uint32_t c0_qint = convert_float_to_qint(C0);
  int c0_shift = get_shift_from_int32_rep(c0_qint);
  int coeff0 = static_cast<int>(C0 * std::pow(2, c0_shift));

  float C1 = C0 * ifm1_zero_point * ifm2_zero_point + ofm_zero_point;
  uint32_t c1_qint = convert_float_to_qint(C1);
  int c1_shift = get_shift_from_int32_rep(c1_qint);
  int coeff1 = static_cast<int>(C1 * std::pow(2, c1_shift));

  return std::make_tuple(c0_shift, coeff0, c1_shift, coeff1);
}

static double round_half_to_even(double value) {
  double integral_part;
  double fractional_part = modf(value, &integral_part);
  double nearest_even;
  if (fractional_part > 0.5 || fractional_part < 0.5) {
    nearest_even = std::round(value);
  } else {
    if (std::fmod(integral_part, 2.0) == 0) {
      nearest_even = integral_part;
    } else {
      nearest_even = integral_part + 1.0;
    }
  }
  return nearest_even;
}

static uint16_t round_srs_to_uint16(double x) {
  return static_cast<uint16_t>(
      saturate<int32_t>((int32_t)round_half_to_even(x), 0, UINT16_MAX));
}

static int32_t round_srs_to_int32(double x) {
  return static_cast<int32_t>(
      saturate<int32_t>((int32_t)round_half_to_even(x), INT32_MIN, INT32_MAX));
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = int16_t>
int test_mladfelwmul(std::vector<size_t> shape, bool debug = false,
                     const std::string &a_dtype = "int16",
                     const std::string &b_dtype = "int16",
                     const std::string &c_dtype = "int16",
                     const std::string &model_name = "PST") {
  int err_count = 0;
  size_t M = shape[0];
  size_t K = std::accumulate(shape.begin() + 1, shape.end(), size_t{1},
                             std::multiplies{});
  std::vector<size_t> a_shape = shape;
  std::vector<size_t> b_shape = shape;
  std::fill(b_shape.begin() + 1, b_shape.end(), 1);
  std::vector<size_t> layer_params_shape = {layer_params_size};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(M * 1);
  std::vector<OuT> cpu_out(M * K);

  std::vector<OuT> aie_out(M * K);
  std::vector<char> layer_params(layer_params_size);
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_out.data());

  srand(0xABCD);

  initialize_random<InT>(a, M * K, 65535, 1);
  initialize_random<WgT>(b, M * 1, 65535, 1);
  double ifm1_scale = 0.000763;
  double ifm2_scale = 0.000113;
  double ofm_scale = 0.0008906;
  int ofm_zero_point = 41733;
  int ifm1_zero_point = 19916;
  int ifm2_zero_point = 42933;

  // This function mimic  kernel run on aie.
  auto qdq_params =
      compute_qdq(ifm1_scale, ifm2_scale, ofm_scale, ofm_zero_point,
                  ifm1_zero_point, ifm2_zero_point);
  int c0_shift, coeff0, c1_shift, coeff1;
  std::tie(c0_shift, coeff0, c1_shift, coeff1) = qdq_params;
  int32_t sum_shift = c0_shift > c1_shift ? 2 : 0;
  int32_t final_shift = c0_shift > c1_shift ? (c0_shift - c1_shift - 2) : 0;
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      int64_t temp_a = static_cast<int64_t>(a.at(r * K + c));
      int64_t temp_b = static_cast<int64_t>(b.at(r));
      int64_t izp1_sum = temp_a * temp_b;
      izp1_sum -= ifm2_zero_point * temp_a;
      izp1_sum -= ifm1_zero_point * temp_b;
      izp1_sum =
          round_srs_to_int32(izp1_sum * 1.0 / double(std::pow(2, sum_shift)));
      int64_t c0_sum = static_cast<int64_t>(izp1_sum * coeff0);
      int64_t sum_ =
          round_srs_to_int32(c0_sum * 1.0 / double(std::pow(2, final_shift)));
      int64_t res = static_cast<int64_t>(sum_ + coeff1);
      cpu_out.at(r * K + c) =
          round_srs_to_uint16(res * 1.0 / double(std::pow(2, c1_shift)));
    }
  }
  // set value of kernel params tensor
  uint32_t kernel_sv_size = 4096;
  layer_params[0] = kernel_sv_size & 0xFF;
  layer_params[1] = (kernel_sv_size >> 8) & 0xFF;
  layer_params[2] = (kernel_sv_size >> 16) & 0xFF;
  layer_params[3] = (kernel_sv_size >> 24) & 0xFF;

  layer_params[4] = coeff0 & 0xFF;
  layer_params[5] = (coeff0 >> 8) & 0xFF;
  layer_params[6] = (coeff0 >> 16) & 0xFF;
  layer_params[7] = (coeff0 >> 24) & 0xFF;

  layer_params[8] = coeff1 & 0xFF;
  layer_params[9] = (coeff1 >> 8) & 0xFF;
  layer_params[10] = (coeff1 >> 16) & 0xFF;
  layer_params[11] = (coeff1 >> 24) & 0xFF;

  layer_params[12] = ifm1_zero_point & 0xFF;
  layer_params[13] = (ifm1_zero_point >> 8) & 0xFF;

  layer_params[14] = ifm2_zero_point & 0xFF;
  layer_params[15] = (ifm2_zero_point >> 8) & 0xFF;

  layer_params[16] = static_cast<uint8_t>(c0_shift);
  layer_params[17] = static_cast<uint8_t>(c1_shift);

  layer_params[18] = 0;
  layer_params[19] = 0;

  uint32_t ifmsv_size = M * K;
  uint32_t kernel_iters = ifmsv_size / (4096 * 8);
  layer_params[20] = kernel_iters & 0xFF;
  layer_params[21] = (kernel_iters >> 8) & 0xFF;

  ryzenai::ml_adf_elw_mul mladfelwmul_ =
      ryzenai::ml_adf_elw_mul<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false);

  std::vector<Tensor> const_Tensor;
  Tensor layer_params_tensor = {layer_params.data(), layer_params_shape,
                                "uint8_t"};
  Tensor b_T = {b.data(), b_shape, b_dtype};
  const_Tensor.push_back(b_T);
  const_Tensor.push_back(layer_params_tensor);

  std::vector<Tensor> input_Tensor;
  Tensor a_T = {a.data(), a_shape, a_dtype};
  Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  mladfelwmul_.debug(debug);
  mladfelwmul_.set_params(model_name, a_shape);
  mladfelwmul_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladfelwmul_.execute(input_Tensor, output_Tensor));
#else
  mladfelwmul_.execute(input_Tensor, output_Tensor);
#endif
  err_count = check_result(cpu_Q_Y, aie_Y);
  return err_count;
}

// MLADFElWMUL
TEST(PSS_mladfelwmul_A16, Kernel1) {
  int err_count = test_mladfelwmul<uint16_t, uint16_t, uint16_t>(
      {1, 4096, 4096}, false, "uint16", "uint16", "uint16", "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
