/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_a16w8_mladf_matrix.hpp"
#include <ops/matmul_a16w8_mladf/matmul_a16w8_mladf.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_a16w8_mladf_matrix;

template <typename InT = int16_t, typename WgT = int8_t, typename OuT = int16_t>
int test_matmul_a16w8_mladf(int M, int K, int N, const uint32_t sv_M,
                            const uint32_t sv_K, const uint32_t sv_N,
                            const uint32_t C1, const uint32_t C2,
                            const int32_t shift_gemm_out,
                            const int32_t shift_qdq_out, bool debug = false,
                            const std::string &a_dtype = "int16",
                            const std::string &b_dtype = "int8",
                            const std::string &c_dtype = "int16",
                            const std::string &model_name = "PSS") {
  int err_count = 0;
  const std::string rtp_dtype = "uint32";
  int Msubv_act = 0;
  if ((a_dtype == "uint16") || (a_dtype == "int16")) {
    Msubv_act = 16;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }
  int N_w = N;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ks, Ns};
  std::vector<size_t> rtp_shape = {16};
  std::vector<size_t> qdq_shape = {Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {Ms, Ns};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<uint32_t> rtp(16);
  std::vector<int64_t> qdq(1 * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> cpu_out(M * N_w);
  std::vector<OuT> cpu_out_qdq(M * N_w);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<OuT> cpu_Y(M, N_w, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N_w, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  // a16w16
  srand(0xABCD);
  init_random(X, 0, 4);
  initialize_random<WgT>(b, K * N, 4, 0);
  initialize_random<int64_t>(qdq, 1 * N, 4, 0);

  // generate qdq params
  float ifm_scale = 1.0;
  uint32_t ifm_zp = 32768;
  float wts_scale = 1.0;
  uint32_t wts_zp = 0;
  float ofm_scale = 1.0;
  uint32_t ofm_zp = 32768;

  float coeffs2 = (ifm_scale * wts_scale) / ofm_scale;
  uint32_t C2_shift = std::log2((std::pow(2, 8) - 1) / coeffs2);
  int64_t *C0_vec = (int64_t *)qdq.data();

  std::cout << "C1= " << C1 << " C2= " << C2 << " c2_shift= " << C2_shift
            << " coeffs2= " << coeffs2 << std::endl;

  if (b_dtype == "uint8") {
    // a16w8
    srand(0xABCD);
    init_random(X, 0, 4);
    initialize_random<WgT>(b, K * N, 4, 0);
    initialize_random<int64_t>(qdq, 1 * N, 8, 0);
  }
  rtp = {sv_M,
         sv_K,
         sv_N,
         0x2000,
         0x6000,
         0x3800,
         K / sv_K,
         static_cast<uint32_t>(shift_qdq_out),
         static_cast<uint32_t>(shift_gemm_out),
         0,
         0,
         0,
         0,
         0,
         0,
         0};
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;

  RowMajorMatrix<WgT> W(K, N_w, b.data());
  cpu_qdq_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>, RowMajorMatrix<OuT>>(
      X, W, cpu_Y, qdq, C1, C2, shift_gemm_out, shift_qdq_out, "uint16");

  ryzenai::matmul_a16w8_mladf matmul_a16w8_mladf_ =
      ryzenai::matmul_a16w8_mladf<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype,
                                                 false);
  matmul_a16w8_mladf_.debug(debug);
  matmul_a16w8_mladf_.set_params(model_name, a_shape);
  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype},
                  {qdq.data(), qdq_shape, "int64"},
                  {qdq_params.data(), qdq_params_shape, "int32"}};
  matmul_a16w8_mladf_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{rtp.data(), rtp_shape, rtp_dtype},
                  {a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(matmul_a16w8_mladf_.execute(input_Tensor, output_Tensor));
#else
  matmul_a16w8_mladf_.execute(input_Tensor, output_Tensor);
#endif

  err_count = check_result(cpu_Y, aie_Y);

  return err_count;
}

// PSS/PST a16w8
TEST(PSS_GEMM_Testa16w8, Kernel1) {
  int err_count = test_matmul_a16w8_mladf<uint16_t, uint8_t, uint16_t>(
      4096, 512, 512, 16, 128, 16, 2, 4, 0, 0, false, "uint16", "uint8",
      "uint16", "PSS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
