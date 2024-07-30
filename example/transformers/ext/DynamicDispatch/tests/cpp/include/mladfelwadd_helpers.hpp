#pragma once

#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <algorithm>
using namespace matmul_matrix;
namespace mladfelwadd_helpers {

typedef enum {
  undefined_broadcast = -1,
  outer_broadcast = 0,
  inner_broadcast = 1,
  scalar_broadcast = 2,
  no_broadcast = 3
} broadcast_type;

broadcast_type
determine_broadcast_type(const std::vector<std::vector<size_t>> &inputs_shape) {
  if (inputs_shape.size() < 2) {
    return undefined_broadcast;
  }

  const auto &shape1 = inputs_shape[0];
  const auto &shape2 = inputs_shape[1];

  if (shape1 == shape2) {
    return no_broadcast;
  }

  if (shape2.size() == shape1.size() && shape2[0] == shape1[0] &&
      std::all_of(shape2.begin() + 1, shape2.end(),
                  [](size_t dim) { return dim == 1; })) {
    return inner_broadcast;
  }

  if (shape2.size() == 1 && shape2[0] == shape1.back()) {
    return outer_broadcast;
  }
  return undefined_broadcast;
}

// Function to round half to even
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

// Function to round and saturate to uint16
static uint16_t round_srs_to_uint16(double x) {
  return static_cast<uint16_t>(
      saturate<int32_t>((int32_t)round_half_to_even(x), 0, UINT16_MAX));
}

// Function to compute qdq parameters
std::tuple<int, int, int, int8_t, int8_t, int8_t, int8_t>
compute_qdq(double ifm1_scale, double ifm2_scale, double ofm_scale,
            int ifm1_zero_point, int ifm2_zero_point, int ofm_zero_point) {
  double ifm1_shift = std::floor(-std::log2(ifm1_scale / ofm_scale) + 31);
  double ifm2_shift = std::floor(-std::log2(ifm2_scale / ofm_scale) + 31);
  double signed_zp = ofm_zero_point -
                     (ifm1_scale * ifm1_zero_point / ofm_scale) -
                     (ifm2_scale * ifm2_zero_point / ofm_scale);
  double zero_point_shift = std::floor(-std::log2(std::abs(signed_zp)) + 31);
  double ofm_shift = std::max({ifm1_shift, ifm2_shift, zero_point_shift});

  int this_ifm1_coeff =
      static_cast<int>(ifm1_scale / ofm_scale * std::pow(2, ifm1_shift));
  int this_ifm2_coeff =
      static_cast<int>(ifm2_scale / ofm_scale * std::pow(2, ifm2_shift));
  int this_zero_point_coeff =
      static_cast<int>(signed_zp * std::pow(2, zero_point_shift));

  int8_t this_ofm_shift = ofm_shift;
  int8_t this_ifm1_shift = ofm_shift - ifm1_shift;
  int8_t this_ifm2_shift = ofm_shift - ifm2_shift;
  int8_t this_zero_point_shift = ofm_shift - zero_point_shift;
  return std::make_tuple(this_ifm1_coeff, this_ifm2_coeff,
                         this_zero_point_coeff, this_ofm_shift, this_ifm1_shift,
                         this_ifm2_shift, this_zero_point_shift);
}

// Function to assign qdq_params
void assign_qdq_params(std::vector<int8_t> &qdq_params, int ifm1_coeff,
                       int ifm2_coeff, int zero_point_coeff, int8_t ofm_shift,
                       int8_t ifm1_shift, int8_t ifm2_shift,
                       int8_t zero_point_shift, uint32_t ifmsv_size,
                       broadcast_type bd_type) {
  uint32_t kernel_sv_size = 4096;
  uint32_t kernel_iters = ifmsv_size / (4096 * 8);

  if (bd_type == no_broadcast) {
    kernel_sv_size = 1024;
    kernel_iters = ifmsv_size / (4096 / 4 * 8);
  }

  qdq_params[0] = kernel_sv_size & 0xFF;
  qdq_params[1] = (kernel_sv_size >> 8) & 0xFF;
  qdq_params[2] = (kernel_sv_size >> 16) & 0xFF;
  qdq_params[3] = (kernel_sv_size >> 24) & 0xFF;

  qdq_params[20] = kernel_iters & 0xFF;
  qdq_params[21] = (kernel_iters >> 8) & 0xFF;

  qdq_params[4] = ifm1_coeff & 0xFF;
  qdq_params[5] = (ifm1_coeff >> 8) & 0xFF;
  qdq_params[6] = (ifm1_coeff >> 16) & 0xFF;
  qdq_params[7] = (ifm1_coeff >> 24) & 0xFF;

  qdq_params[8] = ifm2_coeff & 0xFF;
  qdq_params[9] = (ifm2_coeff >> 8) & 0xFF;
  qdq_params[10] = (ifm2_coeff >> 16) & 0xFF;
  qdq_params[11] = (ifm2_coeff >> 24) & 0xFF;

  qdq_params[12] = zero_point_coeff & 0xFF;
  qdq_params[13] = (zero_point_coeff >> 8) & 0xFF;
  qdq_params[14] = (zero_point_coeff >> 16) & 0xFF;
  qdq_params[15] = (zero_point_coeff >> 24) & 0xFF;

  qdq_params[16] = int8_t(ofm_shift) & 0xFF;
  qdq_params[17] = int8_t(ifm1_shift) & 0xFF;
  qdq_params[18] = int8_t(ifm2_shift) & 0xFF;
  qdq_params[19] = int8_t(zero_point_shift) & 0xFF;
}

// Function to compute CPU output
template <typename InT, typename WgT, typename OuT>
void compute_cpu_output(const std::vector<InT> &a, const std::vector<WgT> &b,
                        std::vector<OuT> &cpu_out, size_t M, size_t K,
                        int ifm1_coeff, int ifm2_coeff, int zero_point_coeff,
                        int8_t ofm_shift, int8_t ifm1_shift, int8_t ifm2_shift,
                        int8_t zero_point_shift, broadcast_type bd_type,
                        int alpha = 1) {
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < K; ++c) {
      double temp_a = a.at(r * K + c) * std::pow(2, ifm1_shift);
      double temp_b;
      if (bd_type == inner_broadcast) {
        temp_b = b.at(r) * std::pow(2, ifm2_shift);
      } else if (bd_type == outer_broadcast) {
        temp_b = b.at(c) * std::pow(2, ifm2_shift);
      } else if (bd_type == no_broadcast) {
        temp_b = b.at(r * K + c) * std::pow(2, ifm2_shift);
      } else {
        throw std::invalid_argument("Invalid broadcast type");
      }

      double temp_c = temp_a * ifm1_coeff + alpha * temp_b * ifm2_coeff +
                      (zero_point_coeff * std::pow(2, zero_point_shift));
      cpu_out.at(r * K + c) =
          round_srs_to_uint16(temp_c * 1.0 / double(std::pow(2, ofm_shift)));
    }
  }
}

void process_shape(std::vector<size_t> &input_shape) {
  if (!input_shape.empty() && input_shape[0] == 1) {
    input_shape.erase(input_shape.begin());
  }
}

} // namespace mladfelwadd_helpers
