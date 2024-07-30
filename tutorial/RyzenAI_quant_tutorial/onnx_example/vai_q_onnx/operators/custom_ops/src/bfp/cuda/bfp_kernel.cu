//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "bfp/cuda/bfp_kernel.h"
#include <stdio.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#else
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#endif

// Get a unsinged value of the biased exponent.
__device__ uint32_t GetExponent(float v) {
  uint32_t uint_v = __float_as_uint(v);
  // Shift away mantissa bits.
  return (uint_v & 0x7f800000) >> 23;
}

// Get a unsinged value of the max biased exponent.
__device__ uint32_t GetMaxExponent(const float* input, int n) {
  uint32_t max_exp = 0;
  for (int i = 0; i < n; i++) {
    max_exp = max(max_exp, GetExponent(input[i]));
  }
  return max_exp;
}

__device__ uint32_t GetMantissa(float v) {
  uint32_t uint_v = *reinterpret_cast<uint32_t*>(&v);
  return uint_v & 0x7fffff;
}

__device__ uint32_t Rand(int seed) {
  curandState state;
  curand_init(static_cast<uint64_t>(clock()) + seed, 0, 0, &state);
  return curand(&state);
}

__device__ float RandUniform(int seed) {
  curandState state;
  curand_init(static_cast<uint64_t>(clock()) + seed, 0, 0, &state);
  return curand_uniform(&state);
}

__device__ float StochsticRound(float v, int seed) {
  return floorf(v + RandUniform(seed));
  //return __float2int_rd(v + RandUniform(seed));
}

__global__ void BFPCUDAKernel(int bit_width,
                              int n,
                              const float* input,
                              float* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  uint32_t shared_exp = 0;
  for (int i = index; i < n; i += stride) {
    uint32_t exp = GetExponent(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Max exponent as shared exponent.
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
    auto exp = GetExponent(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      // Round half to even.
      auto x = std::nearbyintf(input[i] / scale) * scale;
      // Clamp(x, min_v, max_v)
      output[i] = max(-max_v, min(x, max_v));
    }
  }
}

void LaunchBFPCUDAKernel(const float* input,
                         float* output,
                         int n,
                         int bit_width,
                         int block_size) {
  BFPCUDAKernel<<<n / block_size, 1>>>(bit_width, n, input, output);
}

__global__ void BFPCUDAKernelV2(const float* input,
                                float* output,
                                const int num_threads,
                                const int axis_size,
                                const int bit_width,
                                const int block_size,
                                const int rounding_mode) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= num_threads) {
    return;
  }

  const int axis_blocks = axis_size / block_size;
  const int block_index = index % axis_blocks;
  const int axis_index = index / axis_blocks;
  //if (threadIdx.x >= 1) {
  //  printf("index=%d, axis_blocks=%d, block_index=%d, axis_index=%d\n",
  //      index, axis_blocks, block_index, axis_index);
  //}

  int offset = axis_index * axis_size + block_index * block_size;
  // Loop over bounding box to find shared exponent
  uint32_t shared_exp = 0;
  for (int i = 0; i < block_size; i++) {
    uint32_t exp = GetExponent(input[offset + i]);
    if (exp == 0xff) {
      exp = 0;
    }
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }
  
  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = pow(2.0, shared_exp_value - (m_bits - 1));
  auto max_v = pow(2.0, shared_exp_value + 1) - scale;
  for (int i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    auto index = offset + i;
    auto exp = GetExponent(input[index]);
    if (exp == 0xff) {
      output[index] = input[index];
    } else {
      auto scaled_x = input[index] / scale;
      float rounded_scaled_x;
      // 0: Round half to even.
      // 1: Stochstic rounding.
      if (rounding_mode == 0) {
        rounded_scaled_x = nearbyintf(scaled_x);
      } else if (rounding_mode == 1) {
        rounded_scaled_x = StochsticRound(scaled_x, index);
        //printf("index=%d, scaled_x=%f, rounded_scaled_x=%f, nearbyintf=%f\n", index, scaled_x, rounded_scaled_x, nearbyintf(scaled_x));
      }
      // Clamp(x, min_v, max_v)
      output[index] = max(-max_v, min(rounded_scaled_x * scale, max_v));
    }
  }
}

void LaunchBFPCUDAKernelV2(
    const float* input,
    float* output,
    const int n,
    const int axis_size,
    const int bit_width,
    const int block_size,
    const int rounding_mode) {
  
  const int threads = n / block_size;
  
  int threads_per_block = 256;
  int blocks = static_cast<int>(
      ceil(static_cast<float>(threads) / threads_per_block));
  BFPCUDAKernelV2<<<blocks, threads_per_block>>>(
    input,
    output,
    threads,
    axis_size,
    bit_width,
    block_size,
    rounding_mode
  );
}

// Notable things:
// 1. +-INF are converted to NaNs
// 2. All subnormal numbers are flushed to zeros.
// 3. When the shared exponent is 2^w - 1, all k values in a block are NaNs
__global__ void BFPPrimeCUDAKernel(const float* input,
                                   float* output,
                                   const int num_threads,
                                   const int axis_size,
                                   const int bit_width,
                                   const int block_size,
                                   const int sub_block_size,
                                   const int sub_block_shift_bits,
                                   const int rounding_mode) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= num_threads) {
    return;
  }

  const int axis_blocks = axis_size / block_size;
  const int block_index = index % axis_blocks;
  const int axis_index = index / axis_blocks;

  //if (threadIdx.x >= 1) {
  //  printf("index=%d, axis_blocks=%d, block_index=%d, axis_index=%d\n",
  //      index, axis_blocks, block_index, axis_index);
  //}

  // Mantissa bits of float32.
  const uint32_t m_float = 23;
  // Mantissa bits of bfp, sign: 1 bit, exponent: 8 bits.
  const uint32_t m_bfp = bit_width - 9;
  const uint32_t exp_bias = 127;

  int offset = axis_index * axis_size + block_index * block_size;
  uint32_t shared_exp = GetMaxExponent(input + offset, block_size);

  for (int i = 0; i < block_size / sub_block_size; i++) {
    uint32_t max_sub_exp = GetMaxExponent(
        input + offset + i * sub_block_size, sub_block_size);

    // Sub-block shift is the difference between the shared exponent and
    // the maximum exponent in the sub-block, upper bounded by 2^d - 1.
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
      if (rounding_mode != 0 && mantissa != ((1 << (m_bfp + 1)) - 1)) {
        // 0: Round toward zero
        // 1: Round half away from zero
        // 2: Stochstic rounding
        if (rounding_mode == 1) {
          mantissa += 1;
        } else if (rounding_mode == 2) {
          if (num_bits_shifting < 1 + m_float) {
            uint32_t mask = (1 << (num_bits_shifting + 1)) - 1;
            uint32_t p = Rand(idx);
            if ((p & mask) < (input_x & mask)) {
              mantissa += 2;
            }
          }
        }
      }

      // Truncate the last bit
      mantissa >>= 1;
      int sign = input_x & 0x80000000 ? -1 : 1;
      // If any number in a block is +-Inf/NaN,
      // then every number in the output is a NaN.
      if (shared_exp == 0xff) {
        output[idx] = __uint_as_float(0x7fffffff);
      } else {
        // v = (−1)^s * 2^(E - bias) * 2^(-D) * 2^(1-m) * M
        output[idx] = sign * std::pow(2.0,
            static_cast<int>(shared_exp - exp_bias - shift + 1 - m_bfp)) * mantissa;
      }
    }
  }
}

void LaunchBFPPrimeCUDAKernel(
    const float* input,
    float* output,
    const int n,
    const int axis_size,
    const int bit_width,
    const int block_size,
    const int sub_block_size,
    const int sub_block_shift_bits,
    const int rounding_mode) {

  const int threads = n / block_size;

  int threads_per_block = 256;
  int blocks = static_cast<int>(
      ceil(static_cast<float>(threads) / threads_per_block));

  BFPPrimeCUDAKernel<<<blocks, threads_per_block>>>(
    input,
    output,
    threads,
    axis_size,
    bit_width,
    block_size,
    sub_block_size,
    sub_block_shift_bits,
    rounding_mode
  );
}
