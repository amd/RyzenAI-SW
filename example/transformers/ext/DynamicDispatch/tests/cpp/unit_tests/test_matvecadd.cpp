/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/matvecadd/matvecadd.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_matvecadd(size_t M, size_t K, bool debug = false,
                   const std::string &a_dtype = "int16",
                   const std::string &b_dtype = "int8",
                   const std::string &c_dtype = "int32",
                   const std::string &model_name = "PSF") {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {1, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};
  std::vector<size_t> qdq_params_4x4_shape = {QDQparam_size * 2};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(1 * K);
  std::vector<WgT> b(M * K);
  std::vector<OuT> cpu_out(M * K);
  std::vector<OuT> cpu_q_out(M * K);
  std::vector<OuT> aie_out(M * K);
  std::vector<uint16_t> qdq_params_4x4(QDQparam_size * 2);
  std::vector<int32_t> qdq_params(QDQparam_size);
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Y(M, K, cpu_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_q_out.data());

  srand(0xABCD);

  uint16_t matA_zero_point = 2;
  float matA_scale = 2.0;

  uint16_t matB_zero_point = 2;
  float matB_scale = 2.0;

  // quantize output for uint16 output
  float matC_scale = 0.1;
  uint16_t sc_out = float2bfloat(1.0 / matC_scale); // bfloat16;
  OuT matC_zero_point = 4451;

  initialize_random<InT>(a, 1 * K, 4600, 4300);
  initialize_random<WgT>(b, M * K, 1200, 800);
  matA_zero_point = 4451;
  matA_scale = 0.001;
  matB_zero_point = 1000;
  matB_scale = 0.0002;

#ifdef RANDOM_DATA
  qdq_params[0] = matA_zero_point;
  qdq_params[1] = float_to_bfloat16(matA_scale);
  qdq_params[2] = matB_zero_point;
  qdq_params[3] = float_to_bfloat16(matB_scale);
  qdq_params[4] = matC_zero_point;
  qdq_params[5] = sc_out;

  qdq_params_4x4[0] = matA_zero_point;
  qdq_params_4x4[1] = float_to_bfloat16(matA_scale);
  qdq_params_4x4[2] = matB_zero_point;
  qdq_params_4x4[3] = float_to_bfloat16(matB_scale);
  qdq_params_4x4[4] = matC_zero_point;
  qdq_params_4x4[5] = sc_out;

  std::vector<OuT> a_dq(1 * K);
  std::vector<OuT> b_dq(M * K);

  dequant_to_bfloat(a, a_dq, matA_zero_point, matA_scale);
  dequant_to_bfloat(b, b_dq, matB_zero_point, matB_scale);

  // compute golden
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < K; ++c) {
      cpu_out.at(r * K + c) =
          float_to_bfloat16(bfloat16_to_float(a_dq.at(c)) +
                            bfloat16_to_float(b_dq.at(r * K + c)));
    }
  }

  quant_bfloat_to_uint16(cpu_Y, sc_out, matC_zero_point, cpu_Q_Y);

#endif
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }

  ryzenai::matvec_add matvecadd_ = ryzenai::matvec_add<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  std::vector<Tensor> const_Tensor;
  if (model_name == "4x4PSR") {
    const_Tensor = {{qdq_params_4x4.data(), qdq_params_4x4_shape, "uint16"}};
  } else {
    const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};
  }

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), b_shape, b_dtype};
  struct Tensor c_T = {aie_out.data(), b_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  matvecadd_.debug(debug);
  matvecadd_.set_params(model_name, b_shape);
  matvecadd_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(matvecadd_.execute(input_Tensor, output_Tensor));
#else
  matvecadd_.execute(input_Tensor, output_Tensor);
#endif

  err_count = check_add_result(cpu_Q_Y, aie_Y, 0.01);

  return err_count;
}

// MatvecADD
TEST(PSR_MATVECADD_Testa16, Kernel1) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      64, 1280, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_MATVECADD_Testa16, Kernel2) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      256, 1280, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_MATVECADD_Testa16, Kernel3) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      1024, 640, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_MATVECADD_Testa16, Kernel4) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      4096, 320, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// MatvecADD 4x4
TEST(C4PSR_MATVECADD_Testa16, Kernel1) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      64, 1280, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_MATVECADD_Testa16, Kernel2) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      256, 1280, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_MATVECADD_Testa16, Kernel3) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      1024, 640, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_MATVECADD_Testa16, Kernel4) {
  int err_count = test_matvecadd<uint16_t, uint16_t, uint16_t>(
      4096, 320, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
