#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_a16a16_mladf_matrix.hpp"
#include <ops/matmul_a16a16_mladf/matmul_a16a16_mladf.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_a16a16_mladf_matrix;

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_matmul_a16a16_mladf(int M, int K, int N, const uint32_t sv_M,
                             const uint32_t sv_K, const uint32_t sv_N,
                             const int64_t C0_gen_C0, const int32_t C0_gen_C1,
                             const int32_t C1, int32_t C2,
                             const int16_t shift_gemm_out,
                             const int16_t shift_qdq_out, bool debug = false,
                             const std::string &a_dtype = "uint16",
                             const std::string &b_dtype = "uint16",
                             const std::string &c_dtype = "uint16",
                             const std::string &model_name = "PSS") {
  int err_count = 0;
  const std::string rtp_dtype = "uint32";
  if ((a_dtype != "uint16") && (a_dtype != "int16")) {
    throw std::invalid_argument("a_dtype is not supported");
  }
  if ((b_dtype != "uint16") && (b_dtype != "int16")) {
    throw std::invalid_argument("b_dtype is not supported");
  }
  if ((c_dtype != "uint16") && (c_dtype != "int16")) {
    throw std::invalid_argument("c_dtype is not supported");
  }
  int N_w = N;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ks, Ns};
  std::vector<size_t> rtp_shape = {16};
  std::vector<size_t> aie_out_shape = {Ms, Ns};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<uint32_t> rtp(16);
  std::vector<int32_t> cpu_out(M * N_w);
  std::vector<OuT> cpu_out_qdq(M * N_w);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<WgT> W(K, N_w, b.data());
  RowMajorMatrix<OuT> cpu_Y(M, N_w, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N_w, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 4);
  init_random(W, 0, 4);
  rtp = {sv_M,
         sv_K,
         sv_N,
         K / sv_K,
         0x2000,
         0x4800,
         0x3800,
         0x3C00,
         0x4000,
         0x4400,
         static_cast<uint32_t>(C0_gen_C0 & 0xffffffff),
         static_cast<uint32_t>((C0_gen_C0 >> 32) & 0xffffffff),
         static_cast<uint32_t>(C0_gen_C1),
         static_cast<uint32_t>(C1),
         static_cast<uint32_t>(C2),
         static_cast<uint32_t>(shift_gemm_out | (shift_qdq_out << 16))};

  cpu_qdq_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>, RowMajorMatrix<OuT>>(
      X, W, cpu_Y, C0_gen_C0, C0_gen_C1, C1, C2, shift_gemm_out, shift_qdq_out,
      "uint16");

  ryzenai::matmul_a16a16_mladf matmul_a16a16_mladf_ =
      ryzenai::matmul_a16a16_mladf<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype,
                                                  false);

  matmul_a16a16_mladf_.debug(debug);
  matmul_a16a16_mladf_.set_params(model_name, a_shape, b_shape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {};
  matmul_a16a16_mladf_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{rtp.data(), rtp_shape, rtp_dtype},
                  {a.data(), a_shape, a_dtype},
                  {b.data(), b_shape, b_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(matmul_a16a16_mladf_.execute(input_Tensor, output_Tensor));
#else
  matmul_a16a16_mladf_.execute(input_Tensor, output_Tensor);
#endif

  err_count = check_result(cpu_Y, aie_Y);

  return err_count;
}

// PSS
TEST(PSS_GEMM_Testa16a16, Kernel1) {
  int err_count = test_matmul_a16a16_mladf<uint16_t, uint16_t, uint16_t>(
      4096, 512, 4096, 16, 256, 8, 4, 1, 2, 1, 0, 0, false, "uint16", "uint16",
      "uint16", "PSS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSS_GEMM_Testa16a16, Kernel2) {
  int err_count = test_matmul_a16a16_mladf<uint16_t, uint16_t, uint16_t>(
      4096, 4096, 512, 16, 256, 8, 2, 3, 4, 2, 0, 0, false, "uint16", "uint16",
      "uint16", "PSS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PST
TEST(PST_GEMM_Testa16a16, Kernel1) {
  int err_count = test_matmul_a16a16_mladf<uint16_t, uint16_t, uint16_t>(
      4096, 512, 4096, 16, 256, 8, 0, 0, 0, 1, 0, 0, false, "uint16", "uint16",
      "uint16", "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PST_GEMM_Testa16a16, Kernel2) {
  int err_count = test_matmul_a16a16_mladf<uint16_t, uint16_t, uint16_t>(
      4096, 4096, 512, 16, 256, 8, 0, 0, 0, 1, 0, 0, false, "uint16", "uint16",
      "uint16", "PST");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
