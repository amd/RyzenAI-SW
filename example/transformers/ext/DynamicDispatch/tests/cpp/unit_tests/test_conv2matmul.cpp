/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/conv2matmul/conv2matmul.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_conv2matmul(int H, int W, int C, int N, bool debug = false,
                     const std::string &a_dtype = "int16",
                     const std::string &b_dtype = "int8",
                     const std::string &c_dtype = "int32",
                     const std::string &model_name = "PSF") {
  int err_count = 0;

  size_t Hs = static_cast<size_t>(H); // M = H*W
  size_t Ws = static_cast<size_t>(W);
  size_t Cs = static_cast<size_t>(C); // K
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {1, Hs, Ws, Cs};
  std::vector<size_t> b_shape = {Ns, Cs};
  std::vector<size_t> qdq_shape = {Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {1, Hs, Ws, Ns};

  int M, K;
  M = H * W;
  K = C;
  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<WgT> b_trans(K * N);
  std::vector<int64_t> qdq(1 * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> cpu_out(M * N);
  std::vector<OuT> cpu_out_qdq(M * N);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 2048);
  initialize_random<WgT>(b, K * N, 128, 0);
  initialize_random<int64_t>(qdq, 1 * N, 10, 0);
  uint32_t C1 = -11;
  uint32_t C2 = 3;
  uint32_t SQb = 0;
  uint32_t Sout = 16;
  uint32_t Stdm = 2;
  int64_t *C0_vec = (int64_t *)qdq.data();
  int64_t c0 = 0;
  int isint16 = 1;

#ifdef RANDOM_DATA
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv_act;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;
  qdq_params[qdq_isint16_idx] =
      isint16; // for PSH, user needs to set it based on Q datatype

  RowMajorMatrix<WgT> Wmat(K, N, b_trans.data());
  for (int r = 0; r < K; r++) {
    for (int c = 0; c < N; c++) {
      Wmat.at(r, c) = (WgT)(b[c * K + r]);
    }
  }
  cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>, RowMajorMatrix<int32_t>>(
      X, Wmat, cpu_Y, Stdm);
  qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>, RowMajorMatrix<OuT>>(
      X, cpu_Y, C2, C1, C0_vec, SQb, Sout, cpu_Y_qdq, "uint16");

#endif
  std::map<std::string, std::any> attr;
  attr["input_format"] = std::vector<string>{"NHWC"};

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["input_shape"] = std::vector<int>{1, C, H, W};
  }
  ryzenai::conv2matmul conv2matmul_ = ryzenai::conv2matmul<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  conv2matmul_.debug(debug);
  std::vector<size_t> param_shape = {static_cast<size_t>(M), Cs, Ns};
  conv2matmul_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype},
                  {qdq.data(), qdq_shape, "int64"},
                  {qdq_params.data(), qdq_params_shape, "int32"}};

  conv2matmul_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(conv2matmul_.execute(input_Tensor, output_Tensor));
#else
  conv2matmul_.execute(input_Tensor, output_Tensor);
#endif
  // read_bin_file(golden_out_name,
  // reinterpret_cast<char*>(cpu_out_qdq.data()));
  err_count = check_result(cpu_Y_qdq, aie_Y);

  return err_count;
}
// PSR : H, W, C, N
TEST(PSR_CONV2GEMM_Testa16w8, Kernel1) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 320, 64, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel2) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 77, 1024, 64, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel3) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 320, 640, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel4) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 640, 64, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel5) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 640, 1280, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel6) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 1280, 64, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel7) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      8, 8, 1280, 64, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel8) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      8, 8, 2560, 1280, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel9) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 2560, 1280, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel10) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 1920, 1280, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel11) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 1920, 640, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel12) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 1280, 640, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel13) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 960, 640, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel14) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 960, 320, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel15) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 640, 320, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSR GemmV with transB = 1
TEST(PSR_CONV2GEMM_Testa16w8, Kernel16) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 320, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel17) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 640, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_CONV2GEMM_Testa16w8, Kernel18) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 1280, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel1) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 320, 64, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel2) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 77, 1024, 64, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel3) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 320, 640, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel4) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 640, 64, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel5) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 640, 1280, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel6) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 1280, 64, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel7) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      8, 8, 1280, 64, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel8) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      8, 8, 2560, 1280, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel9) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 2560, 1280, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel10) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 1920, 1280, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel11) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 1920, 640, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel12) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 1280, 640, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel13) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 960, 640, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel14) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 960, 320, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel15) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 640, 320, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSR GemmV with transB = 1
TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel16) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 320, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel17) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 640, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_CONV2GEMM_Testa16w8, Kernel18) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 1280, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
