#pragma once

#include "test_common.hpp"

namespace maskedsoftmax_helpers {
std::vector<float>
golden_maskedsoftmax(const std::tuple<size_t, size_t, size_t> &shape,
                     const std::vector<uint16_t> &a,
                     const std::vector<uint16_t> &mask,
                     const float pre_mask_scale) {
  auto [B, M, K] = shape;
  std::vector<float> cpu_float(B * M * K);
  for (int batch = 0; batch < B; batch++) {
    for (int m = 0; m < M; m++) {
      const auto partial2dIndex = m * K;
      const auto partial3dIndex = batch * (M * K) + partial2dIndex;
      // compute runningSum to use in softmax dividend
      float runSum = 0;
      // Masking and exponentiating
      for (int k = 0; k < K; k++) {
        cpu_float.at(partial3dIndex + k) = std::exp(
            dd::bfloat16_to_float(a.at(partial3dIndex + k) * pre_mask_scale) +
            dd::bfloat16_to_float(mask.at(partial2dIndex + k)));
        runSum += cpu_float.at(partial3dIndex + k);
      }
      // Softmaxing
      for (int k = 0; k < K; k++) {
        cpu_float.at(partial3dIndex + k) /= runSum;
      }
    }
  }
  return cpu_float;
}
} // namespace maskedsoftmax_helpers
