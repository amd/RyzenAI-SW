/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/matmul/matmul.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_matmul(int M, int K, int N, int shape_format = 0, bool debug = false,
                const std::string &a_dtype = "int16",
                const std::string &b_dtype = "int8",
                const std::string &c_dtype = "int32",
                const std::string &model_name = "PSF") {
  int err_count = 0;
  int Msubv_act = 0;
  int Nsubv_act = Nsubv;
  int Ksubv_act = Ksubv;
  if (a_dtype == "uint16") {
    Msubv_act = 32;
  } else if (a_dtype == "uint8") {
    Msubv_act = 64;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }
  if (K % Ksubv_PSR == 0) {
    Msubv_act = Msubv_PSR;
    Ksubv_act = Ksubv_PSR;
    Nsubv_act = Nsubv_PSR;
    if (N > 640) {
      Nsubv_act = Nsubv_PSR_LARGE;
    }
  }

  if (model_name == "4x4PSR") {
    SUBV_T key = {M, K, N};
    auto subv_mode = search_subv_mode(key);
    SUBV_T subv = get_subv(subv_mode);
    Msubv_act = subv[0];
    Ksubv_act = subv[1];
    Nsubv_act = subv[2];
  }

  int N_w = N;
  // if (N_w < Nsubv * 2) {
  //   N_w = Nsubv * 2; // This is the miminum N
  // }
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ks, Ns};
  std::vector<size_t> qdq_shape = {Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {Ms, Ns};

  int P;
  if (shape_format == 1) { // mimic the tensor 4D shape
    P = sqrt(M);
    size_t Ps = static_cast<size_t>(P);
    a_shape = {1, Ps, Ps, Ks};
    aie_out_shape = {1, Ps, Ps, Ns};
  }

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<int64_t> qdq(1 * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> cpu_out(M * N_w);
  std::vector<OuT> cpu_out_qdq(M * N_w);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N_w, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N_w, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 32);
  initialize_random<WgT>(b, K * N, 32, 0);
  initialize_random<int64_t>(qdq, 1 * N, 32, 0);

  uint32_t C1 = 0;
  uint32_t C2 = 10;
  uint32_t SQb = 0;
  uint32_t Sout = 13;
  uint32_t Stdm = 0;
  int64_t *C0_vec = (int64_t *)qdq.data();
  int64_t c0 = 0;
  int isint16 = 1;
  if (a_dtype == "uint16") {
    srand(0xABCD);
    init_random(X, 0, 2048);
    initialize_random<WgT>(b, K * N, 128, 0);
    initialize_random<int64_t>(qdq, 1 * N, 10, 0);
    c0 = 0;
    C1 = -11;
    SQb = 0;
    Sout = 16;
    Stdm = 2; // round(log2(K)) - 8;
    C2 = 3;   // 2 << Stdm;
  }
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

  RowMajorMatrix<WgT> W(K, N_w, b.data());
  if (a_dtype == "uint16") {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, Stdm, Msubv_act, Ksubv_act,
                                        Nsubv_act);
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_Y_qdq, "uint16");
  } else {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, "int32");
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_Y_qdq, "uint8");
  }
#else

std:
  string fld_name = "//bin_files//PSI_Matmul0";
  std::vector<uint32_t> aint(M * K);
  read_data_file<uint32_t>(OpInterface::get_dod_base_dir() + fld_name +
                               "//ifm.txt",
                           (uint32_t *)aint.data());
  for (int r = 0; r < M * K; r++) {
    a[r] = (InT)(aint[r]);
  }

  std::vector<uint32_t> bint(N * K);
  read_data_file<uint32_t>(OpInterface::get_dod_base_dir() + fld_name +
                               "//wgt.txt",
                           (uint32_t *)bint.data());
  for (int r = 0; r < N * K; r++) {
    b[r] = (InT)(bint[r]);
  }

  read_data_file<uint64_t>(OpInterface::get_dod_base_dir() + fld_name +
                               "//c0.txt",
                           (uint64_t *)qdq.data());

  read_data_file<uint32_t>(
      OpInterface::get_dod_base_dir() + fld_name + "//c1.txt", (uint32_t *)&C1);

  read_data_file<uint32_t>(
      OpInterface::get_dod_base_dir() + fld_name + "//c2.txt", (uint32_t *)&C2);

  read_data_file<uint32_t>(OpInterface::get_dod_base_dir() + fld_name +
                               "//shift_final.txt",
                           (uint32_t *)&Sout);

  read_data_file<uint32_t>(OpInterface::get_dod_base_dir() + fld_name +
                               "//shift_matmul.txt",
                           (uint32_t *)&Stdm);

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

  std::vector<uint32_t> outint(M * N);
  read_data_file<uint32_t>(OpInterface::get_dod_base_dir() + fld_name +
                               "//ofm.txt",
                           (uint32_t *)outint.data());
  for (int r = 0; r < N * M; r++) {
    cpu_out_qdq[r] = (OuT)(outint[r]);
  }
#endif
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["input_shape"] = std::vector<int>{1, P, P, N};
  }

  ryzenai::matmul matmul_ =
      ryzenai::matmul<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false, attr);

  matmul_.debug(debug);
  std::vector<size_t> param_shape = {Ms, Ks, Ns};
  matmul_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype},
                  {qdq.data(), qdq_shape, "int64"},
                  {qdq_params.data(), qdq_params_shape, "int32"}};

  matmul_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(matmul_.execute(input_Tensor, output_Tensor));
#else
  matmul_.execute(input_Tensor, output_Tensor);
#endif
  // read_bin_file(golden_out_name,
  // reinterpret_cast<char*>(cpu_out_qdq.data()));
  err_count = check_result(cpu_Y_qdq, aie_Y);

  return err_count;
}

// GEMM a8w8
TEST(PSF_GEMM_Testa8w8, Kernel1) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 1152, 1152, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSF_GEMM_Testa8w8, Kernel2) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 768, 1152, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSF_GEMM_Testa8w8, Kernel3) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 512, 1152, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSF_GEMM_Testa8w8, Kernel4) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 768, 768, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSF_GEMM_Testa8w8, Kernel5) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 3072, 768, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSF_GEMM_Testa8w8, Kernel6) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 768, 128, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSF_GEMM_Testa8w8, Kernel7) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 768, 3072, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSF_GEMM_Testa8w8, Kernel8) {
  int err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
      512, 768, 26, 0, false, "uint8", "uint8", "uint8", "PSF");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSJ
TEST(PSJ_GEMM_Testa16w8, Kernel1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      128, 1152, 1152, 0, false, "uint16", "uint8", "uint16", "PSJ");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSJ_GEMM_Testa16w8, Kernel2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      128, 768, 1152, 0, false, "uint16", "uint8", "uint16", "PSJ");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSJ_GEMM_Testa16w8, Kernel3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      128, 512, 1152, 0, false, "uint16", "uint8", "uint16", "PSJ");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSJ_GEMM_Testa16w8, Kernel4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      128, 768, 768, 0, false, "uint16", "uint8", "uint16", "PSJ");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSJ_GEMM_Testa16w8, Kernel5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      128, 3072, 768, 0, false, "uint16", "uint8", "uint16", "PSJ");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSJ_GEMM_Testa16w8, Kernel6) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      128, 768, 128, 0, false, "uint16", "uint8", "uint16", "PSJ");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSJ_GEMM_Testa16w8, Kernel7) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      128, 768, 3072, 0, false, "uint16", "uint8", "uint16", "PSJ");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSH
TEST(PSH_GEMM_Testa16w8, Kernel11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      512, 1152, 1152, 0, false, "uint16", "uint8", "uint16", "PSH");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSH_GEMM_Testa16w8, Kernel12) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      512, 768, 1152, 0, false, "uint16", "uint8", "uint16", "PSH");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSH_GEMM_Testa16w8, Kernel13) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      512, 512, 1152, 0, false, "uint16", "uint8", "uint16", "PSH");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSH_GEMM_Testa16w8, Kernel14) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      512, 768, 768, 0, false, "uint16", "uint8", "uint16", "PSH");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSH_GEMM_Testa16w8, Kernel15) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      512, 3072, 768, 0, false, "uint16", "uint8", "uint16", "PSH");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSH_GEMM_Testa16w8, Kernel16) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      512, 768, 128, 0, false, "uint16", "uint8", "uint16", "PSH");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSH_GEMM_Testa16w8, Kernel17) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      512, 768, 3072, 0, false, "uint16", "uint8", "uint16", "PSH");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSI
TEST(PSI_GEMM_Testa16w8, Kernel1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      49, 128, 128, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      49, 1024, 1024, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      49, 1024, 3072, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      49, 1024, 4096, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      49, 4096, 1024, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel6) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      196, 512, 512, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel7) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      196, 512, 1536, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel8) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      196, 512, 2048, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel9) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      196, 2048, 512, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel10) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      784, 256, 256, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      784, 256, 768, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel12) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      784, 256, 1024, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel13) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      784, 1024, 256, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel14) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      3136, 128, 128, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel15) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      3136, 128, 384, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel16) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      3136, 128, 512, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel17) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      3136, 512, 128, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_GEMM_Testa16w8, Kernel18) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1, 1024, 768, 0, false, "uint16", "uint8", "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSQ2_GEMM_Testa16w8, Kernel18) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      77, 1024, 1024, 0, false, "uint16", "uint8", "uint16", "PSQ2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSQ2_GEMM_Testa16w8, Kernel19) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      77, 4096, 1024, 0, false, "uint16", "uint8", "uint16", "PSQ2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSR 4x2
TEST(PSR_GEMM_Testa16w8, Kernel1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      64, 1280, 10240, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      64, 1280, 1280, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      64, 5120, 1280, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      256, 5120, 1280, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      256, 1280, 10240, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel6) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      256, 1280, 1280, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel7) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1024, 640, 5120, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel8) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1024, 2560, 640, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel9) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1024, 640, 640, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel10) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      4096, 1280, 320, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      4096, 320, 320, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_GEMM_Testa16w8, Kernel12) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      4096, 320, 2560, 1, false, "uint16", "uint8", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSR 4x4
TEST(C4PSR_GEMM_Testa16w8, Kernel1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      64, 1280, 10240, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      64, 1280, 1280, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      64, 5120, 1280, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      256, 5120, 1280, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      256, 1280, 10240, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel6) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      256, 1280, 1280, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel7) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1024, 640, 5120, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel8) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1024, 2560, 640, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel9) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1024, 640, 640, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel10) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      4096, 1280, 320, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      4096, 320, 320, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_GEMM_Testa16w8, Kernel12) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      4096, 320, 2560, 1, false, "uint16", "uint8", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
