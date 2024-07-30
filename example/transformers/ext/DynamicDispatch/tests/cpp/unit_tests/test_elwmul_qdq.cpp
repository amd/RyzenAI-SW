/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/elwmul_qdq/elwmul_qdq.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_elwmul_qdq(size_t M, size_t K, bool debug = false,
                    const std::string &a_dtype = "int16",
                    const std::string &b_dtype = "int8",
                    const std::string &c_dtype = "int32",
                    const std::string &model_name = "PSF") {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(M * K);
  std::vector<OuT> cpu_out(M * K);
  std::vector<OuT> cpu_q_out(M * K);
  std::vector<OuT> aie_out(M * K);
  std::vector<int32_t> qdq_params(QDQparam_size);
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Y(M, K, cpu_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_q_out.data());

  srand(0xABCD);
  initialize_random<InT>(a, M * K, 16, 0);
  initialize_random<WgT>(b, M * K, 16, 0);

  int32_t matA_zero_point = 2;
  float matA_scale = 2.0;
  int32_t is_matA_uint16 = 1; // is_matA_uint16 = 0, input a is bf16

  int32_t matB_zero_point = 2;
  float matB_scale = 2.0;

  // quantize output for uint16 output
  float matC_scale = 0.1;
  int16_t sc_out = float2bfloat(1.0 / matC_scale); // bfloat16;
  OuT matC_zero_point = 4451;
  int32_t is_matC_uint16 = 0; // is_matC_uint16 = 1, output c is uint16

  if (a_dtype == "uint16" || a_dtype == "bfloat16") {
    initialize_random<InT>(a, M * K, 4600, 4300);
    initialize_random<WgT>(b, M * K, 1200, 800);
    matA_zero_point = 4451;
    matA_scale = 0.001;
    matB_zero_point = 1000;
    matB_scale = 0.0002;
  }

  if (a_dtype == "bfloat16") {
    is_matA_uint16 = 0;
  }

  if (c_dtype == "uint16") {
    is_matC_uint16 = 1;
  }

#ifdef RANDOM_DATA
  qdq_params[0] = float_to_bfloat16(matA_scale);
  qdq_params[1] = matA_zero_point;
  qdq_params[2] = float_to_bfloat16(matB_scale);
  qdq_params[3] = matB_zero_point;
  qdq_params[4] = (int32_t)sc_out;
  qdq_params[5] = (int32_t)matC_zero_point;
  qdq_params[6] = is_matA_uint16;
  qdq_params[7] = is_matC_uint16;

  std::vector<OuT> a_dq(M * K);
  std::vector<OuT> b_dq(M * K);

  dequant_to_bfloat(a, a_dq, matA_zero_point, matA_scale);
  dequant_to_bfloat(b, b_dq, matB_zero_point, matB_scale);

  if (is_matA_uint16 == 0) { // matA is bf16
    memcpy((void *)a.data(), (void *)a_dq.data(), M * K * sizeof(OuT));
  }

  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      cpu_out.at(r * K + c) =
          float_to_bfloat16(bfloat16_to_float(a_dq.at(r * K + c)) *
                            bfloat16_to_float(b_dq.at(r * K + c)));
    }
  }

  if (c_dtype == "uint16") {
    quant_bfloat_to_uint16(cpu_Y, sc_out, matC_zero_point, cpu_Q_Y);
    // q_bfloat2uint16(Out, float2bfloat(matC_scale), zp_out, cpu_Y);
  }

#endif
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::elwmul_qdq elwmul_qdq_ = ryzenai::elwmul_qdq<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), a_shape, b_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  elwmul_qdq_.debug(debug);
  elwmul_qdq_.set_params(model_name, a_shape);
  elwmul_qdq_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(elwmul_qdq_.execute(input_Tensor, output_Tensor));
#else
  elwmul_qdq_.execute(input_Tensor, output_Tensor);
#endif
  if (c_dtype == "uint16") {
    err_count = check_add_result(cpu_Q_Y, aie_Y, 0.01);
  } else {
    err_count = check_add_result_bfloat16<OuT>(cpu_out, aie_out, a_shape, 0.01);
  }
  return err_count;
}

// elwmul_qdq
TEST(PSR_ELWMULQDQ_Testa16w8, Kernel1) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      64, 5120, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_ELWMULQDQ_Testa16w8, Kernel2) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      256, 5120, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_ELWMULQDQ_Testa16w8, Kernel3) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      1024, 2560, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_ELWMULQDQ_Testa16w8, Kernel4) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      4096, 1280, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// 4x4 elwmul_qdq
TEST(C4PSR_ELWMULQDQ_Testa16w8, Kernel1) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      64, 5120, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_ELWMULQDQ_Testa16w8, Kernel2) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      256, 5120, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_ELWMULQDQ_Testa16w8, Kernel3) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      1024, 2560, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_ELWMULQDQ_Testa16w8, Kernel4) {
  int err_count = test_elwmul_qdq<uint16_t, uint16_t, uint16_t>(
      4096, 1280, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
