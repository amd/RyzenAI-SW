//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef _NNDCT_FIX_KERELS_H_
#define _NNDCT_FIX_KERELS_H_

template<typename Dtype>
void cuda_sigmoid_table_lookup(const int N, 
                               const Dtype* input, 
                               const Dtype* table,
                               Dtype* output,
                               int fragpos);  

template<typename Dtype>
void cuda_sigmoid_simulation(const int N, 
                               const Dtype* input, 
                               Dtype* output,
                               int fragpos);  

template<typename Dtype>
void cuda_tanh_simulation(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            int fragpos);  

template<typename Dtype>
void cuda_tanh_table_lookup(const int N, 
                            const Dtype* input, 
                            const Dtype* table,
                            Dtype* output,
                            int fragpos);  

template<typename Dtype>
void cuda_fix_neuron_v1(const int N, 
                        const Dtype* src,
                        const Dtype* fragpos, 
                        Dtype* dst, 
                        int val_min,
                        int val_max, 
                        int keep_scale, 
                        int method);

template<typename Dtype>
void cuda_vai_round(const int N, 
                    const Dtype* src,
                    Dtype* dst, 
                    int method);

template<typename Dtype>
void cuda_fix_neuron_v2(const int N, 
                        const Dtype* src,
                        Dtype* dst, 
                        int val_min,
                        int val_max, 
                        Dtype val_amp, 
                        int zero_point,
                        int keep_scale, 
                        int method);

template<typename Dtype>
void cuda_fix_neuron_v2_2d(const int N_row, 
                           const int N_col, 
                           const Dtype* src, 
                           Dtype* dst, 
                           int val_min,
                           int val_max, 
                           const Dtype* scale, 
                           const int* zero_point,
                           int keep_scale, 
                           int method);

template<typename Dtype>
void cuda_diff_S(const int N, 
                 const Dtype* src, 
                 Dtype* buffer, 
                 Dtype* output, 
                 int bitwidth, 
                 int range, 
                 int method);

template<typename Dtype>
void cuda_softmax_exp_approximate(const int N,
                            const Dtype* input,
                            Dtype* output);  

template<typename Dtype>
void cuda_softmax_lod(const int N,
                            const Dtype* input,
                            Dtype* output);  

template<typename Dtype>
void cuda_softmax_simulation_part_1(const int N,
                            const Dtype* input,
                            Dtype* output); 

template<typename Dtype>
void cuda_softmax_simulation_part_2(const int N,
                            const Dtype* sum,
                            Dtype* output);  

template<typename Dtype>
void cuda_sigmoid_table_lookup_aie2(const int N, 
                               const Dtype* input, 
                               Dtype* output,
                               int fragpos);

template<typename Dtype>
void cuda_tanh_table_lookup_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            int fragpos); 

template<typename Dtype>
void cuda_exp_appr_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output,
                            const int bit_width);  
template<typename Dtype>
void cuda_log_softmax_fast_ln(const int N,
                            const Dtype* input,
                            Dtype* output);  

template<typename Dtype>
void cuda_log_softmax_sub(const int N,
                            const Dtype* input,
                            Dtype* output,
                            const Dtype* sub);  

template<typename Dtype>
void cuda_aie_sqrt(const int N,
                   const Dtype* input,
                   Dtype* output); 

template<typename Dtype>
void cuda_aie_isqrt(const int N,
                    const Dtype* input,
                    Dtype* output); 

template<typename Dtype>
void cuda_layernorm_isqrt(const int N,
                          const Dtype* input,
                          Dtype* output); 

template<typename Dtype>
void cuda_layernorm_invsqrt(const int N,
                            const Dtype* input,
                            Dtype* output); 
template<typename Dtype>
void cuda_inverse_aie2(const int N, 
                            const Dtype* input, 
                            Dtype* output); 

#endif //_NNDCT_FIX_KERELS_H_
