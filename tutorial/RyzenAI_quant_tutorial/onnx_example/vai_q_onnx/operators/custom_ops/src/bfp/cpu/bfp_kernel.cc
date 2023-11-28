//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <cstdint>
#include <cmath>
#include "bfp/cpu/bfp_kernel.h"

uint32_t __float_as_uint(float x) {
  return *reinterpret_cast<uint32_t*>(&x);
}

float __uint_as_float(uint32_t x) {
  return *reinterpret_cast<float*>(&x);
}

uint32_t GetExponentCPU(float v) {
  // Get the biased exponent.
  uint32_t uint_v = *reinterpret_cast<uint32_t*>(&v);
  // Shift away mantissa bits.
  return (uint_v & 0x7f800000) >> 23;
}

// Get a unsinged value of the max biased exponent.
uint32_t GetMaxExponentCPU(const float* input, int n) {
  uint32_t max_exp = 0;
  for (int i = 0; i < n; i++) {
    max_exp = std::max(max_exp, GetExponentCPU(input[i]));
  }
  return max_exp;
}

void BFPCPUKernel(const float* input,
                  float* output,
                  int n,
                  int index,
                  int stride,
                  int bit_width,
                  int rounding_mode) {
  uint32_t shared_exp = 0;
  // Loop over block to find shared exponent.
  for (int i = index; i < n; i += stride) {
    uint32_t exp = GetExponentCPU(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents.
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }
  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = std::pow(2.0, shared_exp_value - (m_bits - 1));
  auto max_v = std::pow(2.0, shared_exp_value + 1) - scale;
  for (int i = index; i < n; i += stride) {
    // Output +-0/NaN/Inf as is.
    uint32_t exp = GetExponentCPU(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      // Round half to even.
      auto x = std::nearbyintf(input[i] / scale) * scale;
      // Clamp(x, min_v, max_v)
      output[i] = std::max(-max_v, std::min(x, max_v));
    }
  }
}

void LaunchBFPCPUKernel(const float* input,
                        float* output,
                        int n,
                        int bit_width,
                        int block_size,
                        int rounding_mode) {
  int num_blocks = n / block_size;
  for (int index = 0; index < num_blocks; index++) {
    BFPCPUKernel(input, output, n, index, num_blocks, bit_width, rounding_mode);
  }
}

void BFPCPUKernelV2(const float* input,
                    float* output,
                    int offset,
                    int bit_width,
                    int block_size,
                    int rounding_mode) {
  uint32_t shared_exp = 0;
  // Loop over block to find shared exponent.
  for (int i = 0; i < block_size; i++) {
    uint32_t exp = GetExponentCPU(input[offset + i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents.
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = std::pow(2.0, shared_exp_value - (m_bits - 1));
  auto max_v = std::pow(2.0, shared_exp_value + 1) - scale;

  for (int i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    uint32_t exp = GetExponentCPU(input[offset + i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      // Round half to even.
      auto x = std::nearbyintf(input[i] / scale) * scale;
      // Clamp(x, min_v, max_v)
      output[i] = std::max(-max_v, std::min(x, max_v));
    }
  }
}

void LaunchBFPCPUKernelV2(const float* input,
                          float* output,
                          int n,
                          int bit_width,
                          int block_size,
                          int rounding_mode) {
  int num_blocks = n / block_size;
  for (int index = 0; index < num_blocks; index++) {
    BFPCPUKernel(input, output, index * block_size + block_size,
                 index * block_size, 1, bit_width, rounding_mode);
  }
}

void LaunchBFPPrimeCPUKernel(const float* input,
                             float* output,
                             const int n,
                             const int bit_width,
                             const int block_size,
                             const int sub_block_size,
                             const int sub_block_shift_bits,
                             const int rounding_mode) {

  int num_blocks = n / block_size;
  for (int index = 0; index < num_blocks; index++) {
    BFPPrimeCPUKernel(input, output, n, index * block_size/*offset*/, 1/*stride*/,
        bit_width, block_size, sub_block_size, sub_block_shift_bits,
        rounding_mode);
  }
}

// Notable things:
// 1. +-INF are converted to NaNs
// 2. All subnormal numbers are flushed to zeros.
// 3. When the shared exponent is 2^w - 1, all k values in a block are NaNs
void BFPPrimeCPUKernel(const float* input,
                       float* output,
                       const int n,
                       const int offset,
                       const int stride,
                       const int bit_width,
                       const int block_size,
                       const int sub_block_size,
                       const int sub_block_shift_bits,
                       const int rounding_mode) {

  // Mantissa bits of float32.
  const uint32_t m_float = 23;
  // Mantissa bits of bfp, sign: 1 bit, exponent: 8 bits.
  const uint32_t m_bfp = bit_width - 9;
  const uint32_t exp_bias = 127;

  uint32_t shared_exp = GetMaxExponentCPU(input + offset, block_size);

  for (int i = 0; i < block_size / sub_block_size; i++) {
    uint32_t max_sub_exp = GetMaxExponentCPU(
        input + offset + i * sub_block_size, sub_block_size);

    // Compute sub-block shifts. Each sub-block shift is the difference between
    // the shared exponent and the maximum exponent in the sub-block,
    // upper bounded by 2^d - 1.
    uint32_t shift;
    uint32_t shift_upper_bound = (1 << sub_block_shift_bits) - 1;
    if (shared_exp - max_sub_exp > shift_upper_bound) {
      shift = shift_upper_bound;
    } else {
      shift = shared_exp - max_sub_exp;
    }

    for (int j = 0; j < sub_block_size; j++) {
      auto idx = offset + i * sub_block_size + j;
      uint32_t input_x = __float_as_uint(input[idx]);
      uint32_t exp = (input_x & 0x7f800000) >> m_float;
      uint32_t mantissa;
      if (exp == 0) {
        // Subnormals are flushed to zero.
        mantissa = 0;
      } else {
        // Add leading 1.
        mantissa = (input_x & 0x7fffff) | (1 << m_float);
      }
      // Right shift mantissa by the exponent difference + the mantissa bitwidth difference
      uint32_t num_bits_shifting = shared_exp - shift - exp + m_float - m_bfp;
      if (num_bits_shifting >= 32) {
        // Shift a number of bits equal or greater than the bit width is undefined behavior.
        num_bits_shifting = 31;
      }
      mantissa >>= num_bits_shifting;
      // Do not round up if mantissa is all 1s.
      if (rounding_mode == 0 && mantissa != ((1 << (m_bfp + 1)) - 1)) {
        mantissa += 1;
      }
      mantissa >>= 1;
      int sign = input_x & 0x80000000 ? -1 : 1;
      // If any number in a block is +-Inf/NaN,
      // then every number in the output is a NaN.
      if (shared_exp == 0xff) {
        output[idx] = __uint_as_float(0x7fffffff);
      } else {
        // v = (−1)^s * 2^(E - bias) * 2^(-D) * 2^(1-m) * M
        output[idx] = sign * std::pow(2.0,
            static_cast<int>(shared_exp - exp_bias - shift + 1 - m_bfp)) * static_cast<int>(mantissa);
      }
    }
  }
}
