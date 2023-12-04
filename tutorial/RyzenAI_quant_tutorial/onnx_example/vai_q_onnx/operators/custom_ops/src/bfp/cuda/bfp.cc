//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "bfp/bfp.h"
#include "bfp/cuda/bfp_kernel.h"
#include "bfp/cpu/bfp_kernel.h"

void to_bfp(const Ort::Value& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      int64_t rounding_mode,
                      Ort::Value& out) {
  const float* input = tensor.GetTensorData<float>();
  float* output = out.GetTensorMutableData<float>();
  size_t element_count = out.GetTensorTypeAndShapeInfo().GetElementCount();

  LaunchBFPCUDAKernel(input, output, element_count, bit_width, block_size);
}

void to_bfp_v2(const Ort::Value& tensor,
                         int64_t bit_width,
                         int64_t block_size,
                         int64_t rounding_mode,
                         Ort::Value& out) {
  const float* input = tensor.GetTensorData<float>();
  float* output = out.GetTensorMutableData<float>();
  size_t element_count = out.GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<int64_t> dimensions = tensor.GetTensorTypeAndShapeInfo().GetShape();

  int axis_size = dimensions[dimensions.size() - 1];
  LaunchBFPCUDAKernelV2(input, output, element_count, axis_size, bit_width,
      block_size, rounding_mode);
}

void to_bfp_prime_cuda(const Ort::Value& tensor,
                                 int64_t bit_width,
                                 int64_t block_size,
                                 int64_t sub_block_size,
                                 int64_t sub_block_shift_bits,
                                 int64_t rounding_mode,
                                 Ort::Value& out) {

  const float* input = tensor.GetTensorData<float>();
  float* output = out.GetTensorMutableData<float>();
  size_t element_count = out.GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<int64_t> dimensions = tensor.GetTensorTypeAndShapeInfo().GetShape();

  int axis_size = dimensions[dimensions.size() - 1];
  LaunchBFPPrimeCUDAKernel(
      input, output, element_count, axis_size, bit_width, block_size,
      sub_block_size, sub_block_shift_bits, rounding_mode);
}

void to_bfp_prime_shared(const Ort::Value& tensor,
                                   int64_t bit_width,
                                   int64_t block_size,
                                   int64_t sub_block_size,
                                   int64_t sub_block_shift_bits,
                                   int64_t rounding_mode,
                                   Ort::Value& out) {

  const float* input = tensor.GetTensorData<float>();
  float* output = out.GetTensorMutableData<float>();
  size_t element_count = out.GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<int64_t> dimensions = tensor.GetTensorTypeAndShapeInfo().GetShape();

  int axis_size = dimensions[dimensions.size() - 1];
  LaunchBFPPrimeCUDAKernel(input, output, element_count, axis_size,
      bit_width, block_size, sub_block_size, sub_block_shift_bits,
      rounding_mode);
}
