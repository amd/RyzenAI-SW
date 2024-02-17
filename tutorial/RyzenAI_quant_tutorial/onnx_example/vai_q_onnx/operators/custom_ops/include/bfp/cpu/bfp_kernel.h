//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#ifndef _INCLUDE_CPU_BFP_H_
#define _INCLUDE_CPU_BFP_H_

void LaunchBFPCPUKernel(const float* input,
                        float* output,
                        int n,
                        int bit_width,
                        int block_size,
                        int rounding_mode);

void LaunchBFPCPUKernelV2(const float* input,
                          float* output,
                          int n,
                          int bit_width,
                          int block_size,
                          int rounding_mode);

void BFPPrimeCPUKernel(const float* input,
                       float* output,
                       const int n,
                       const int offset,
                       const int stride,
                       const int bit_width,
                       const int block_size,
                       const int sub_block_size,
                       const int sub_block_shift_bits,
                       const int rounding_mode);

void LaunchBFPPrimeCPUKernel(const float* input,
                             float* output,
                             int n,
                             const int bit_width,
                             const int block_size,
                             const int sub_block_size,
                             const int sub_block_shift_bits,
                             const int rounding_mode);

#endif // _INCLUDE_CPU_BFP_H_
