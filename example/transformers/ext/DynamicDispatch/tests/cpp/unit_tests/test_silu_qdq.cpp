/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/silu_qdq/silu_qdq.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_silu_qdq(int M, int N, bool debug = false,
                  const std::string &a_dtype = "int16",
                  const std::string &b_dtype = "int16",
                  const std::string &c_dtype = "int16",
                  const std::string &model_name = "PSF") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> cpu_q_out(M * N);
  std::vector<OutT> aie_out(M * N);
  std::vector<int16_t> qdq_params(QDQparam_size);

  RowMajorMatrix<OutT> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OutT> cpu_q_Y(M, N, cpu_q_out.data());
  RowMajorMatrix<OutT> aie_Y(M, N, aie_out.data());
  RowMajorMatrix<InT> inputMat(M, N, a.data());

#ifdef RANDOM_DATA
  int32_t is_output_uint16 = 0;

  if (c_dtype == "uint16") {
    is_output_uint16 = 1;
  }

  float sc_float = 0.01;
  int16_t sc_out = 1.0 / sc_float; // bfloat16
  OutT zp_out = 129;

  srand(0xABCD);
  initialize_random_bfloat16(a, M * N, -20, 20);

  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold = bfloat16_to_float(inputMat.at(r, c));
      cpu_Y.at(r, c) = float_to_bfloat16(silu_golden(in_gold, r, c));
    }
  }
  // quant_bfloat_to_uint16(cpu_Y, sc_out, zp_out, cpu_q_Y);
  quant_bfloat16_to_int16(cpu_Y, cpu_q_Y, sc_out, zp_out);

  qdq_params[0] = zp_out; // for silu
  qdq_params[1] = float_to_bfloat16(sc_out);
  qdq_params[2] = 1; // out_quant_enable
  qdq_params[3] = 0;
  qdq_params[4] = 0;
  qdq_params[5] = 0; // if 1, enalbe de-quant at input

#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::silu_qdq silu_qdq_ =
      ryzenai::silu_qdq<InT, WgT, OutT>(a_dtype, b_dtype, c_dtype, false, attr);

  silu_qdq_.debug(debug);
  silu_qdq_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int16"}};

  silu_qdq_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(silu_qdq_.execute(input_Tensor, output_Tensor));
#else
  silu_qdq_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  err_count = check_add_result(cpu_q_Y, aie_Y, 0.1);

  return err_count;
}

// PSR 4x2
TEST(PSR_SILUQDQ_Testa16, Kernel1) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      320, 1024, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel2) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      320, 4096, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel3) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      640, 256, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel4) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      640, 1024, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel5) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      640, 4096, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel6) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      960, 1024, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel7) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      960, 4096, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel8) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1280, 64, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel9) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1280, 256, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel10) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1280, 1024, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel11) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1920, 256, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel12) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1920, 1024, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel13) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      2560, 64, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_SILUQDQ_Testa16, Kernel14) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      2560, 256, false, "bfloat16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSR 4x4 -- Need to update txn/param files for the combined xclbin
TEST(C4PSR_SILUQDQ_Testa16, Kernel1) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      320, 1024, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel2) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      320, 4096, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel3) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      640, 256, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel4) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      640, 1024, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel5) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      640, 4096, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel6) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      960, 1024, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel7) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      960, 4096, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel8) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1280, 64, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel9) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1280, 256, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel10) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1280, 1024, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel11) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1920, 256, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel12) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1920, 1024, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel13) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      2560, 64, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel14) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      2560, 256, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_SILUQDQ_Testa16, Kernel15) {
  int err_count = test_silu_qdq<uint16_t, uint16_t, uint16_t>(
      1280, 1, false, "bfloat16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
