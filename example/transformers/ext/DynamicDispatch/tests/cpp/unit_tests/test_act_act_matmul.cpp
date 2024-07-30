/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
// #include "ops/ops_common/matmul_matrix.hpp"
#include "ops/ops_common/mhagprb_matrix.hpp"
#include <ops/act_act_matmul_qdq/act_act_matmul_qdq.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
// using namespace matmul_matrix;
using namespace mhagprb_matrix;

template <typename InT = int16_t, typename WgT = int16_t,
          typename OuT = int16_t>
int test_act_act_matmul_qdq(int M, int K, int N, int shape_format = 0,
                            bool debug = false,
                            const std::string &a_dtype = "int16",
                            const std::string &b_dtype = "int16",
                            const std::string &c_dtype = "int32",
                            const std::string &model_name = "PSR") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);

#if 1
  size_t Stq, St, St_pad, Sq, Dt, Dh, Skv, H, B;

  H = 1;
  B = 1;

  Stq = Ms;
  St = Ms;
  Skv = 64;

  St_pad = St; // no padding in case of QKt MatMul and SMV Matmul for Self MHA
  Sq = 16;     // there is an equation in main_di_model.cpp
  Dh = 64;     // sub-volume K per head
  Dt = Dh * H; // total volume K across all heads
#endif
  size_t n_qdq_nodes;
  std::vector<size_t> full_shape = {Ms, Ks, Ns};
  std::vector<size_t> act1_shape = {Ms, Ks};
  std::vector<size_t> act2_shape = {Ns, Ks};
  std::vector<size_t> out_shape = {Ms, Ns};
  if (model_name == "4x4PSR")
    n_qdq_nodes = 1;
  else
    n_qdq_nodes = 6;
  std::vector<size_t> qdq_params_shape = {n_qdq_nodes, QDQparam_size};

  std::vector<InT> act1(M * K);
  std::vector<WgT> act2(N * K);
  std::vector<OuT> both_acts((M * K) + (N * K));
  std::vector<OuT> cpu_out(M * N);
  std::vector<OuT> model_out(M * N);
  std::vector<OuT> aie_out(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size * n_qdq_nodes);

  OutMatrix<OuT, 1, 1> model_Y(M, N, model_out.data());
  OutMatrix<OuT, 1, 1> aie_Y(M, N, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Y(M, N, cpu_out.data());

#ifdef RANDOM_DATA
  srand(0xABCD);
  initialize_random<InT>(act1, M * K, 16, 0);
  initialize_random<WgT>(act2, K * N, 16, 0);

  if (model_name == "PSR") {
    // MuL DQ - dummy, not used in kernel
    qdq_params[(16 * 0) + 0] = 0;
    qdq_params[(16 * 0) + 1] = float_to_bfloat16(1.0).value;
    // MuL Q - dummy, not used in kernel
    qdq_params[(16 * 1) + 0] = 0;
    qdq_params[(16 * 1) + 1] = float_to_bfloat16(1.0).value;

    // QKT
    *(int64_t *)(&qdq_params[(16 * 2) + 0]) = 681330421901376; // c0
    qdq_params[(16 * 2) + 2] = -322147399;                     // c1
    qdq_params[(16 * 2) + 3] = 1250944;                        // c2
    qdq_params[(16 * 2) + 4] = -304829643;                     // c3
    qdq_params[(16 * 2) + 5] = Sq;                             // M
    qdq_params[(16 * 2) + 6] = Skv;                            // N
    qdq_params[(16 * 2) + 7] = 0;                              // SQb
    qdq_params[(16 * 2) + 8] = 30;                             // Sout
    qdq_params[(16 * 2) + 9] = 7;                              // Stdm

    // SM *V
    *(int64_t *)(&qdq_params[(16 * 3) + 0]) = 8824815616000;
    qdq_params[(16 * 3) + 2] = -134639853;
    qdq_params[(16 * 3) + 3] = 2098688;
    qdq_params[(16 * 3) + 4] = 0;
    qdq_params[(16 * 3) + 5] = Sq;
    qdq_params[(16 * 3) + 6] = val_subv_cols;
    qdq_params[(16 * 3) + 7] = 0;
    qdq_params[(16 * 3) + 8] = 28;
    qdq_params[(16 * 3) + 9] = 9;

    // DQ before SM
    qdq_params[(16 * 4) + 0] = 35625;
    qdq_params[(16 * 4) + 1] =
        float_to_bfloat16(0.0003217620251234621 * 1.442695041).value;

    // Q after SM
    qdq_params[(16 * 5) + 0] = 0;
    qdq_params[(16 * 5) + 1] = float_to_bfloat16(1.0 / 0.000009475293).value;
  } else { // 4x4PSR
    // QKT
    *(int64_t *)(&qdq_params[(16 * 0) + 0]) = 681330421901376; // c0
    qdq_params[(16 * 0) + 2] = -322147399;                     // c1
    qdq_params[(16 * 0) + 3] = 1250944;                        // c2
    qdq_params[(16 * 0) + 4] = -304829643;                     // c3
    qdq_params[(16 * 0) + 5] = Sq;                             // M
    qdq_params[(16 * 0) + 6] = Skv;                            // N
    qdq_params[(16 * 0) + 7] = 0;                              // SQb
    qdq_params[(16 * 0) + 8] = 30;                             // Sout
    qdq_params[(16 * 0) + 9] = 7;                              // Stdm
  }

#else
  std::string fld_name;
  if (model_name == "PSR" && M == 256 && K == 64 && N == 256) {
    fld_name = "//self_qkt_m256//";
  } else if (model_name == "PSR" && M == 1024 && K == 64 && N == 1024) {
    fld_name = "//self_qkt_m1024//";
  } else if (model_name == "PSR" && M == 4096 && K == 64 && N == 4096) {
    fld_name = "//self_qkt_m4096//";
  } else if (model_name == "PSR" && M == 256 && K == 256 && N == 64) {
    fld_name = "//self_smv_m256//";
  } else if (model_name == "PSR" && M == 1024 && K == 1024 && N == 64) {
    fld_name = "//self_smv_m1024//";
  } else if (model_name == "PSR" && M == 4096 && K == 4096 && N == 64) {
    fld_name = "//self_smv_m4096//";
  } else if (model_name == "4x4PSR" && M == 256 && K == 64 && N == 256) {
    fld_name = "//m_256//";
  } else if (model_name == "4x4PSR" && M == 1024 && K == 64 && N == 1024) {
    fld_name = "//m_1024//";
  } else if (model_name == "4x4PSR" && M == 4096 && K == 64 && N == 4096) {
    fld_name = "//m_4096//";
  } else if (model_name == "4x4PSR" && M == 256 && K == 256 && N == 64) {
    fld_name = "//smxv_256x256x64//";
  } else if (model_name == "4x4PSR" && M == 1024 && K == 1024 && N == 64) {
    fld_name = "//smxv_1024x1024x64//";
  } else if (model_name == "4x4PSR" && M == 4096 && K == 4096 && N == 64) {
    fld_name = "//smxv_4096x4096x64//";
  }

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "ifm.bin",
                reinterpret_cast<char *>(both_acts.data()));

  memcpy(act1.data(), both_acts.data(), Ms * Ks * sizeof(InT));
  memcpy(act2.data(),
         reinterpret_cast<char *>(both_acts.data()) + Ms * Ks * sizeof(InT),
         Ns * Ks * sizeof(WgT));

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "ofm.bin",
                reinterpret_cast<char *>(model_out.data()));

#if 0
  // MuL DQ - dummy, not used in kernel
  qdq_params[(16 * 0) + 0] = 0;
  qdq_params[(16 * 0) + 1] = float_to_bfloat16(1.0).value;
  // MuL Q - dummy, not used in kernel
  qdq_params[(16 * 1) + 0] = 0;
  qdq_params[(16 * 1) + 1] = float_to_bfloat16(1.0).value;

    // QKT
  *(int64_t *)(&qdq_params[(16 * 2) + 0]) = 681330421901376; // c0
  qdq_params[(16 * 2) + 2] = -322147399;                     // c1
  qdq_params[(16 * 2) + 3] = 1250944;                        // c2
  qdq_params[(16 * 2) + 4] = -304829643;                     // c3
  qdq_params[(16 * 2) + 5] = Sq;                             // M
  qdq_params[(16 * 2) + 6] = Skv;                            // N
  qdq_params[(16 * 2) + 7] = 0;                              // SQb
  qdq_params[(16 * 2) + 8] = 30;                             // Sout
  qdq_params[(16 * 2) + 9] = 7;                              // Stdm

  // SM *V
  *(int64_t *)(&qdq_params[(16 * 3) + 0]) = 8824815616000;
  qdq_params[(16 * 3) + 2] = -134639853;
  qdq_params[(16 * 3) + 3] = 2098688;
  qdq_params[(16 * 3) + 4] = 0;
  qdq_params[(16 * 3) + 5] = Sq;
  qdq_params[(16 * 3) + 6] = val_subv_cols;
  qdq_params[(16 * 3) + 7] = 0;
  qdq_params[(16 * 3) + 8] = 28;
  qdq_params[(16 * 3) + 9] = 9;

  // DQ before SM
  qdq_params[(16 * 4) + 0] = 35625;
  qdq_params[(16 * 4) + 1] =
      float_to_bfloat16(0.0003217620251234621 * 1.442695041).value;

  // Q after SM
  qdq_params[(16 * 5) + 0] = 0;
  qdq_params[(16 * 5) + 1] = float_to_bfloat16(1.0 / 0.000009475293).value;
#endif

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "wgt.bin",
                reinterpret_cast<char *>(qdq_params.data()));

#endif
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }

  ryzenai::act_act_matmul act_act_matmul_ =
      ryzenai::act_act_matmul<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false,
                                             attr);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {act1.data(), act1_shape, a_dtype};
  struct Tensor b_T = {act2.data(), act2_shape, b_dtype};
  struct Tensor c_T = {aie_out.data(), out_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  act_act_matmul_.debug(debug);
  act_act_matmul_.set_params(model_name, full_shape);
  act_act_matmul_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(act_act_matmul_.execute(input_Tensor, output_Tensor));
#else
  act_act_matmul_.execute(input_Tensor, output_Tensor);
#endif

  float const max_pct_diff = 1.0;
#ifdef RANDOM_DATA
  float average_error_rate = check_result_mha(cpu_Y, aie_Y, max_pct_diff, 0);
#else
  float average_error_rate = check_result_mha(model_Y, aie_Y, max_pct_diff, 0);
#endif

  return err_count;
}

// Activation MatMul for PSR 4x2

// QKt for 256
TEST(PSR_actGEMM_Testa16w16, Kernel1) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      256, 64, 256, 1, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// QKt for 1024
TEST(PSR_actGEMM_Testa16w16, Kernel2) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      1024, 64, 1024, 1, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// QKt for 4096
TEST(PSR_actGEMM_Testa16w16, Kernel3) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      4096, 64, 4096, 1, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SMV for 256
TEST(PSR_actGEMM_Testa16w16, Kernel4) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      256, 256, 64, 1, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SMV for 1024
TEST(PSR_actGEMM_Testa16w16, Kernel5) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      1024, 1024, 64, 1, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SMV for 4096
TEST(PSR_actGEMM_Testa16w16, Kernel6) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      4096, 4096, 64, 1, false, "uint16", "uint16", "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Activation MatMul for PSR 4x4

// QKt for 256
TEST(C4PSR_actGEMM_Testa16w16, Kernel1) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      256, 64, 256, 1, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// QKt for 1024
TEST(C4PSR_actGEMM_Testa16w16, Kernel2) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      1024, 64, 1024, 1, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// QKt for 4096
TEST(C4PSR_actGEMM_Testa16w16, Kernel3) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      4096, 64, 4096, 1, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SMV for 256
TEST(C4PSR_actGEMM_Testa16w16, Kernel4) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      256, 256, 64, 1, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SMV for 1024
TEST(C4PSR_actGEMM_Testa16w16, Kernel5) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      1024, 1024, 64, 1, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SMV for 4096
TEST(C4PSR_actGEMM_Testa16w16, Kernel6) {
  int err_count = test_act_act_matmul_qdq<uint16_t, uint16_t, uint16_t>(
      4096, 4096, 64, 1, false, "uint16", "uint16", "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
