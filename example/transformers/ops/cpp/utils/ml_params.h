/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __ML_PARAMS_H__
#define __ML_PARAMS_H__

#include <stdint.h>
namespace ryzenai {
union MLKernelControl {
  uint32_t control;
  struct {
    uint32_t zero_init : 1;
    uint32_t sign_N : 1;
    uint32_t sign_O : 1;
    uint32_t reserved3 : 3;
    uint32_t skip_casc_in : 1;
    uint32_t skip_casc_out : 1;
    uint32_t sign_W : 1;
    uint32_t sign_A : 1;
    uint32_t out_64 : 1;
    uint32_t reserved10 : 13;
    uint32_t norm_ch_g : 8;
  } parts;
};

struct MLKernelParams {
  uint8_t Kx_g;
  uint8_t Ky_g;
  uint8_t Ci_g;
  int8_t S_g;
  uint8_t N_g;
  uint8_t X_g;
  uint8_t Y_g;
  uint8_t Co_g;
  uint16_t inner_g;
  uint16_t outer_g;
  int8_t shift_tdm;
  int8_t shift_res;
  int8_t shift_norm;
  int8_t shift_bias;

  uint16_t step_Kx;
  uint16_t step_Ky;
  uint16_t step_Ci;
  uint16_t step_Xi;
  uint16_t step_Yi;
  uint16_t step_Xo;
  uint16_t step_Yo;
  uint16_t step_Co;
  int param_value;
  MLKernelControl ctrl;

  MLKernelParams() {
    Kx_g = 0;
    Ky_g = 0;
    Ci_g = 0;
    S_g = 0;
    N_g = 16;
    X_g = 4;
    Y_g = 8;
    Co_g = 0;
    inner_g = 16;
    outer_g = 32;
    shift_tdm = 12;
    shift_res = 12;
    shift_norm = 0;
    shift_bias = 0;
    step_Kx = 1024;
    step_Ky = 8;
    step_Ci = 0;
    step_Xi = 512;
    step_Yi = 8;
    step_Xo = 512;
    step_Yo = 8;
    step_Co = 0;
    param_value = 0;
    ctrl.control = 773;
  }

  void update_params(int Y, int N, int X, int Ygran, int Ngran, int Xgran) {
    // compute derived params
    Y_g = Y / Ygran;
    N_g = N / Ngran;
    X_g = X / Xgran;
    inner_g = N_g;
    outer_g = Y_g * X_g;
    step_Xi = Y * 8;
    step_Kx = N * 8;
    step_Yi = Ygran;
    step_Xo = Y * 8;
    step_Yo = Ygran;
  }
};
} // namespace ryzenai
#endif //__ML_PARAMS_H__
