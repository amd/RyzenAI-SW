/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/matmulbias/matmulbias.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"

using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_matmulbias(int M, int K, int N, bool debug = false,
                    const std::string &a_dtype = "int8",
                    const std::string &b_dtype = "int8",
                    const std::string &c_dtype = "int8", bool bias = false,
                    bool gelu = false) {
  int err_count = 0;
  if (a_dtype == "int16") {
    int constexpr Msubv = 32;
    int constexpr Ksubv = 128;
    int constexpr Nsubv = 64;
  } else if (a_dtype == "int8") {
    int constexpr Msubv = 64;
    int constexpr Ksubv = 128;
    int constexpr Nsubv = 64;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }

  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ks, Ns};
  std::vector<size_t> bias_shape = {1, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<WgT> biasv(1 * N);

  std::vector<OuT> cpu_out(M * N, garbage_value);
  std::vector<OuT> aie_out(M * N, garbage_value);

  ActMatrix<InT, Msubv, Ksubv> X(M, K, a.data());
  RowMajorMatrix<WgT> W(K, N, b.data());
  BiaVector<InT, 1, Nsubv> B(N, biasv.data());
  OutMatrix<OuT, Msubv, Nsubv> cpu_Y(M, N, cpu_out.data());
  OutMatrix<OuT, Msubv, Nsubv> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, -16, 16);
  initialize_random<WgT>(b, K * N, 16);
  init_random(B, -16, 16);

  cpu_matmul<ActMatrix<InT, Msubv, Ksubv>, RowMajorMatrix<WgT>,
             BiaVector<InT, 1, Nsubv>, OutMatrix<OuT, Msubv, Nsubv>>(
      X, W, B, cpu_Y, c_dtype);

  ryzenai::matmulbias matmulbias_ =
      ryzenai::matmulbias<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, true);

  matmulbias_.debug(debug);
  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype},
                  {biasv.data(), bias_shape, b_dtype}};

  matmulbias_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(matmulbias_.execute(input_Tensor, output_Tensor));
#else
  matmulbias_.execute(input_Tensor, output_Tensor);
#endif

  err_count = check_result(cpu_Y, aie_Y);

  return err_count;
}

//// GEMM BIAS a8w8
// TEST(GEMM_Testa8w8_BIAS, Kernel1) {
//   int err_count = test_matmulbias<int8_t, int8_t, int8_t>(
//       512, 1152, 1152, false, "int8", "int8", "int8");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(GEMM_Testa8w8_BIAS, Kernel2) {
//   int err_count = test_matmulbias<int8_t, int8_t, int8_t>(
//       512, 768, 1152, false, "int8", "int8", "int8");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(GEMM_Testa8w8_BIAS, Kernel3) {
//   int err_count = test_matmulbias<int8_t, int8_t, int8_t>(
//       512, 512, 1152, false, "int8", "int8", "int8");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(GEMM_Testa8w8_BIAS, Kernel4) {
//   int err_count = test_matmulbias<int8_t, int8_t, int8_t>(
//       512, 768, 768, false, "int8", "int8", "int8");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(GEMM_Testa8w8_BIAS, Kernel5) {
//   int err_count = test_matmulbias<int8_t, int8_t, int8_t>(
//       512, 3072, 768, false, "int8", "int8", "int8");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(GEMM_Testa8w8_BIAS, Kernel6) {
//   int err_count = test_matmulbias<int8_t, int8_t, int8_t>(
//       512, 768, 128, false, "int8", "int8", "int8");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
