/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/gelu/gelu.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_gelu(int M, int N, bool debug = false,
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
  std::vector<float> a_dq(M * N);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> aie_out(M * N);
  std::vector<int16_t> qdq_params(QDQparam_size);

  RowMajorMatrix<OutT> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OutT> aie_Y(M, N, aie_out.data());
  RowMajorMatrix<InT> inputMat(M, N, a.data());

#ifdef RANDOM_DATA
  int32_t is_input_uint16 = 0;

  if (a_dtype == "uint16") {
    is_input_uint16 = 1;
  }

  uint16_t in_dq_zero_point = 4250;
  float in_dq_scale = 0.001;

  srand(0xABCD);
  if (is_input_uint16 == 1) {
    initialize_random<InT>(a, M * N, 4600, 4000);
  } else {
    throw std::runtime_error("Gelu not supported datatype.");
  }

  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold = (inputMat.at(r, c) - in_dq_zero_point) * in_dq_scale;
      cpu_Y.at(r, c) = float_to_bfloat16(gelu_golden(in_gold));
    }
  }

  qdq_params[0] = 0; // for silu
  qdq_params[1] = 0;
  qdq_params[2] = 0; // out_quant_enable
  qdq_params[3] = in_dq_zero_point;
  qdq_params[4] = float_to_bfloat16(in_dq_scale);
  qdq_params[5] = 1; // if 1, enalbe de-quant at input

#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::gelu gelu_ =
      ryzenai::gelu<InT, WgT, OutT>(a_dtype, b_dtype, c_dtype, false, attr);

  gelu_.debug(debug);
  gelu_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int16"}};

  gelu_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(gelu_.execute(input_Tensor, output_Tensor));
#else
  gelu_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  err_count = check_result_bfloat(cpu_Y, aie_Y, 0.01);

  return err_count;
}

// PSR 4x2
TEST(PSR_GELU_Testa16, Kernel1) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      64, 5120, false, "uint16", "uint16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GELU_Testa16, Kernel2) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      256, 5120, false, "uint16", "uint16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GELU_Testa16, Kernel3) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      1024, 2560, false, "uint16", "uint16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GELU_Testa16, Kernel4) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      4096, 1280, false, "uint16", "uint16", "bfloat16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSR 4x4 -- Need to update txn/param files for the combined xclbin
TEST(C4PSR_GELU_Testa16, Kernel1) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      64, 5120, false, "uint16", "uint16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GELU_Testa16, Kernel2) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      256, 5120, false, "uint16", "uint16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GELU_Testa16, Kernel3) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      1024, 2560, false, "uint16", "uint16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GELU_Testa16, Kernel4) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      4096, 1280, false, "uint16", "uint16", "bfloat16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
