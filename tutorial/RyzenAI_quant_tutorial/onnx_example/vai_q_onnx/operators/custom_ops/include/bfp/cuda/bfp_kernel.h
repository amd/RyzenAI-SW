//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef _INCLUDE_CUDA_BFP_KERNEL_H_
#define _INCLUDE_CUDA_BFP_KERNEL_H_

void LaunchBFPCUDAKernel(const float* input,
                         float* output,
                         int n,
                         int bit_width,
                         int block_size);

void LaunchBFPCUDAKernelV2(const float* input,
                           float* output,
                           const int n,
                           const int axis_size,
                           const int bit_width,
                           const int block_size,
                           const int rounding_mode);

void LaunchBFPPrimeCUDAKernel(const float* input,
                              float* output,
                              const int n,
                              const int axis_size,
                              const int bit_width,
                              const int block_size,
                              const int sub_block_size,
                              const int sub_block_shift_bits,
                              const int rounding_mode);

#endif // _INCLUDE_CUDA_BFP_KERNEL_H_
