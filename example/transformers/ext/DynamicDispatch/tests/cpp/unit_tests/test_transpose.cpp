/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/transpose/transpose.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_transpose(size_t H, size_t M, size_t K, bool debug = false,
                   const std::string &a_dtype = "int16",
                   const std::string &b_dtype = "int8",
                   const std::string &c_dtype = "int32",
                   const std::string &model_name = "PSF") {
  int err_count = 0;
  size_t Hs = static_cast<size_t>(H);
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Hs, Ms, Ks};

  int P, Q, R, S; // 4D transpose
  if (H == 1) {   // transpose before win_attn
    P = std::sqrt(M) / 7;
    Q = 7;
    R = P;
    S = K * 7;
  } else {
    P = std::sqrt(H);
    Q = P;
    R = 7;
    S = K * 7;
  }

  std::vector<InT> a(H * M * K);
  std::vector<OuT> cpu_out(H * M * K);
  std::vector<OuT> aie_out(H * M * K);

  initialize_random<InT>(a, H * M * K, 255, 0);

#ifdef RANDOM_DATA
  // p, q, r, s -> p, r, q, s
  // compute golden
  for (int p = 0; p < P; p++) {
    for (int r = 0; r < R; r++) {
      for (int q = 0; q < Q; q++) {
        for (int s = 0; s < S; s++) {
          cpu_out[p * R * Q * S + r * Q * S + q * S + s] =
              a[p * R * Q * S + q * R * S + r * S + s];
        }
      }
    }
  }
#else

#endif
  ryzenai::transpose transpose_ =
      ryzenai::transpose<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false);

  std::vector<Tensor> const_Tensor;

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  transpose_.debug(debug);
  transpose_.set_params(model_name, a_shape);
  transpose_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(transpose_.execute(input_Tensor, output_Tensor));
#else
  transpose_.execute(input_Tensor, output_Tensor);
#endif

  err_count = 0;
  int max_diff = 0;
  float L2_norm = 0;
  for (int r = 0; r < H * M * K; ++r) {
    int32_t diff = std::abs(cpu_out[r] - aie_out[r]);
    L2_norm += ((float)diff * (float)diff);
    if (diff > max_diff)
      max_diff = diff;
    if (diff > 0) {
      // std::cout << "ERROR: Y[" << r << ", " << c << "]: "
      //           << "Expected: " << int(cpu_Y.at(r, c)) << ", "
      //           << "Received: " << int(aie_Y.at(r, c)) << ", "
      //           << "Diff: " << int(diff) << "\n";
      err_count++;
    }
  }
  L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;

  return err_count;
}

// TRANSPOSE
TEST(PSI_TRANS_Testa16w8, Kernel1) {
  int err_count = test_transpose<uint16_t, int8_t, uint16_t>(
      1, 3136, 128, false, "uint16", "int8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_TRANS_Testa16w8, Kernel2) {
  int err_count = test_transpose<uint16_t, int8_t, uint16_t>(
      1, 784, 256, false, "uint16", "int8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_TRANS_Testa16w8, Kernel3) {
  int err_count = test_transpose<uint16_t, int8_t, uint16_t>(
      1, 196, 512, false, "uint16", "int8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_TRANS_Testa16w8, Kernel4) {
  int err_count = test_transpose<uint16_t, int8_t, uint16_t>(
      1, 49, 1024, false, "uint16", "int8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_TRANS_Testa16w8, Kernel5) {
  int err_count = test_transpose<uint16_t, int8_t, uint16_t>(
      64, 49, 128, false, "uint16", "int8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_TRANS_Testa16w8, Kernel6) {
  int err_count = test_transpose<uint16_t, int8_t, uint16_t>(
      16, 49, 256, false, "uint16", "int8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_TRANS_Testa16w8, Kernel7) {
  int err_count = test_transpose<uint16_t, int8_t, uint16_t>(
      4, 49, 512, false, "uint16", "int8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
