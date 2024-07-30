#pragma once

#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <algorithm>
using namespace matmul_matrix;
namespace mladfelwmul_helpers {

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

std::tuple<int, int, int, int>
compute_qdq(double ifm1_scale, double ifm2_scale, double ofm_scale,
            int ifm1_zero_point, int ifm2_zero_point, int ofm_zero_point) {
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

// Function to assign qdq_params
void assign_qdq_params(std::vector<int8_t> &qdq_params, int coeff0, int coeff1,
                       int ifm1_zero_point, int ifm2_zero_point, int c0_shift,
                       int c1_shift, uint32_t ifmsv_size) {
  uint32_t kernel_sv_size = 4096;
  uint32_t kernel_iters = ifmsv_size / (4096 * 8);
  qdq_params[0] = kernel_sv_size & 0xFF;
  qdq_params[1] = (kernel_sv_size >> 8) & 0xFF;
  qdq_params[2] = (kernel_sv_size >> 16) & 0xFF;
  qdq_params[3] = (kernel_sv_size >> 24) & 0xFF;

  qdq_params[4] = coeff0 & 0xFF;
  qdq_params[5] = (coeff0 >> 8) & 0xFF;
  qdq_params[6] = (coeff0 >> 16) & 0xFF;
  qdq_params[7] = (coeff0 >> 24) & 0xFF;

  qdq_params[8] = coeff1 & 0xFF;
  qdq_params[9] = (coeff1 >> 8) & 0xFF;
  qdq_params[10] = (coeff1 >> 16) & 0xFF;
  qdq_params[11] = (coeff1 >> 24) & 0xFF;

  qdq_params[12] = ifm1_zero_point & 0xFF;
  qdq_params[13] = (ifm1_zero_point >> 8) & 0xFF;

  qdq_params[14] = ifm2_zero_point & 0xFF;
  qdq_params[15] = (ifm2_zero_point >> 8) & 0xFF;

  qdq_params[16] = static_cast<uint8_t>(c0_shift);
  qdq_params[17] = static_cast<uint8_t>(c1_shift);

  qdq_params[18] = 0;
  qdq_params[19] = 0;

  qdq_params[20] = kernel_iters & 0xFF;
  qdq_params[21] = (kernel_iters >> 8) & 0xFF;
}

// Function to compute CPU output
template <typename InT, typename WgT, typename OuT>
void compute_cpu_output(const std::vector<InT> &a, const std::vector<WgT> &b,
                        std::vector<OuT> &cpu_out, size_t M, size_t K,
                        int coeff0, int coeff1, int ifm1_zero_point,
                        int ifm2_zero_point, int c0_shift, int c1_shift) {
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
}

void process_shape(std::vector<size_t> &input_shape) {
  if (input_shape.size() != 1 && !input_shape.empty() && input_shape[0] == 1) {
    input_shape.erase(input_shape.begin());
  }
}

} // namespace mladfelwmul_helpers
