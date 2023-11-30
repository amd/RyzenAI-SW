//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef _NNDCT_CPU_MATH_H_
#define _NNDCT_CPU_MATH_H_

template<typename Dtype>
void cpu_set(const int n, Dtype* data, Dtype val);

template<typename Dtype>
void cpu_max(const int n, const Dtype* src, Dtype& dst);

template<typename Dtype>
void cpu_pow(const int n, Dtype* data, Dtype pow);

template<typename Dtype>
void cpu_min(const int n, const Dtype* src, Dtype& dst);

template<typename Dtype>
void cpu_sub(const int n, const Dtype* src, Dtype* dst);

template<typename Dtype>
void cpu_sum(const int n, const Dtype* src, Dtype& dst);

template<typename Dtype>
void cpu_scale_inplace(const int n, 
                       Dtype* data, 
                       Dtype scale);

template<typename Dtype>
void cpu_scale(const int n, 
               const Dtype* src, 
               Dtype* dst, 
               Dtype scale);

#endif //_NNDCT_CPU_MATH_H_

