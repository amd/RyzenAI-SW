/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include "enable_perf.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/bmm/bmm.hpp>
#include <vector>

#define RANDOM_DATA

using namespace ryzenai;
using namespace matmul_matrix;

template <typename Tx, typename Tw, typename Ty>
void cpu_bmm(Tx X, Tw W, Ty Y, bool trans) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      float acc = 0.0;
      for (int k = 0; k < X.num_cols; ++k) {
        // acc += X.at(r, k) * W.at(k, c);
        float fx = dd::bfloat16_to_float(X.at(r, k));
        float fw = 0.0;
        if (trans) {
          fw = dd::bfloat16_to_float(W.at(c, k));
        } else {
          fw = dd::bfloat16_to_float(W.at(k, c));
        }
        acc += fx * fw;
      }
      Y.at(r, c) = float_to_bfloat16(acc);
    }
  }
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_bmm(int M, int K, int N, int B = 32, bool debug = false,
             const std::string &a_dtype = "uint16_t",
             const std::string &b_dtype = "uint16_t",
             const std::string &c_dtype = "uint16_t",
             const std::string &model_name = "BMM", bool trans = true) {
  int BM = M * B;
  int err_count = 0;

  size_t Ms = static_cast<size_t>(BM);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {B * Ks, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<InT> a(BM * K);
  std::vector<WgT> b(B * K * N);
  std::vector<uint16_t> cpu_out(BM * N);
  std::vector<OuT> aie_out(BM * N, garbage_value);

  RowMajorMatrix<InT> X(BM, K, a.data());
  RowMajorMatrix<WgT> *W;

  if (trans == true) {
    W = new RowMajorMatrix<WgT>(B * N, K, b.data());
  } else {
    W = new RowMajorMatrix<WgT>(B * K, N, b.data());
  }

  RowMajorMatrix<uint16_t> cpu_Y(BM, N, cpu_out.data());
  RowMajorMatrix<OuT> aie_Y(BM, N, aie_out.data());

  srand(0xABCD);
  dd::initialize_random_bfloat16(a, 1.5);
  dd::initialize_random_bfloat16(b, 1.5);
  ryzenai::bmm bmm_ =
      ryzenai::bmm<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false);
  bmm_.debug(debug);
  bmm_.set_params(model_name, a_shape);
  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype}};
  bmm_.initialize_const_params(const_Tensor);
  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS({
    bmm_.initialize_const_params(const_Tensor);
    bmm_.execute(input_Tensor, output_Tensor);
  });
#else
  bmm_.execute(input_Tensor, output_Tensor);
#endif

  for (int i = 0; i < B; i++) {
    RowMajorMatrix<InT> XX(M, K, a.data() + i * M * K);
    RowMajorMatrix<InT> *WW;
    if (trans == true) {
      WW = new RowMajorMatrix<WgT>(N, K, b.data() + i * K * N);
    } else {
      WW = new RowMajorMatrix<WgT>(K, N, b.data() + i * K * N);
    }
    RowMajorMatrix<uint16_t> cpu_YY(M, N, cpu_out.data() + i * M * N);
    cpu_bmm<RowMajorMatrix<InT>, RowMajorMatrix<WgT>, RowMajorMatrix<OuT>>(
        XX, *WW, cpu_YY, trans);
    std::cout << ".";
  }
  std::cout << std::endl;
  err_count =
      check_add_result_bfloat16<OuT>(cpu_out, aie_out, aie_out_shape, 0.75);
  return err_count;
}

// BMM a16w16
TEST(BMM_Testa16w16_2048_128_2048A, Kernel1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      2048, 128, 2048, 32, false, "uint16_t", "uint16_t", "uint16_t", "BMM",
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// BMM a16w16
TEST(BMM_Testa16w16_2048_128_2048B, Kernel2) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      2048, 128, 2048, 32, false, "uint16_t", "uint16_t", "uint16_t", "BMM",
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// BMM a16w16
TEST(BMM_Testa16w16_2048_2048_128A, Kernel3) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      2048, 2048, 128, 32, false, "uint16_t", "uint16_t", "uint16_t", "BMM",
      false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// BMM a16w16
TEST(BMM_Testa16w16_2048_2048_128B, Kernel4) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      2048, 2048, 128, 32, false, "uint16_t", "uint16_t", "uint16_t", "BMM",
      false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
