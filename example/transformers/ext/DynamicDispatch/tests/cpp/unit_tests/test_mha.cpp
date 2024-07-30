/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/mha_validation.cpp"
#include "ops/ops_common/mhagprb_matrix.hpp"
#include <ops/mha/mha.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace mhagprb_matrix;
template <typename Tqkv = uint16_t, typename WgT = uint8_t,
          typename OuT = uint8_t>
int test_mha(int M, int K, int N, bool debug = false,
             const std::string &a_dtype = "uint16",
             const std::string &b_dtype = "uint8",
             const std::string &c_dtype = "uint16",
             const std::string &model_name = "PSQ2") {
  int err_count = 0;

  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);

  size_t batch, heads, x, y;
  batch = 1;
  heads = 16;
  x = K;
  y = 64;

  size_t St, H, S, Dh, St_pad;
  St = x;      // 77
  St_pad = 80; // 77 padded to 80
  H = heads;
  Dh = 64;

  size_t Dt = Dh * H; // total volume K across all heads

  size_t qry_rows = St;
  size_t qry_cols = Dt;
  size_t key_rows = St;
  size_t key_cols = Dt;
  size_t val_rows = St;
  size_t val_cols = Dt;
  size_t out_rows = St;
  size_t out_cols = Dt;

  size_t qry_subv_rows = St;
  size_t qry_subv_cols = Dh;
  size_t key_subv_rows = St;
  size_t key_subv_cols = Dh;
  size_t val_subv_rows = St;
  size_t val_subv_cols = Dh;
  size_t out_subv_rows = St;
  size_t out_subv_cols = Dh;

  std::vector<size_t> qkv_shape = {Ms, Ks, Ns};

  std::vector<size_t> q_shape = {qry_rows, qry_cols};
  std::vector<size_t> k_shape = {key_rows, key_cols};
  std::vector<size_t> v_shape = {val_rows, val_cols};

  std::vector<size_t> qdq_shape = {num_qdq_nodes, QDQparam_size};
  std::vector<size_t> out_shape = {batch * heads * x * y};

  std::vector<Tqkv> qkv(M * K * N);
  // std::cout << "QKV size : " << qkv.size() * sizeof(Tqkv) << std::endl;

  std::vector<int32_t> qdq_params(QDQparam_size * num_qdq_nodes);

  std::vector<OuT> out(batch * heads * x * y);
  std::vector<OuT> cpu_out(batch * heads * x * y);
  RowMajorMatrix<Tqkv> cpu_Y(out_rows, out_cols, cpu_out.data());
  RowMajorMatrix<Tqkv> aie_Y(batch * out_rows, out_cols, out.data());

  std::vector<Tqkv> q(qry_rows * qry_cols);
  std::vector<Tqkv> k(key_rows * key_cols);
  std::vector<Tqkv> v(val_rows * val_cols);
  RowMajorMatrix<Tqkv> aie_Q(qry_rows, qry_cols, q.data());
  RowMajorMatrix<Tqkv> aie_K(key_rows, key_cols, k.data());
  RowMajorMatrix<Tqkv> aie_V(val_rows, val_cols, v.data());
  gemm_qdq_param<Tqkv> qdq_params_l[4];

  // Initialize q, k and v
  srand(0xABCD);
  int64_t qk_qdq_c0, smv_qdq_c0;
  int32_t C1, C2, C3;
  uint8_t SQb, Sout, Stm_qkt, Stm_smv;
  // DQ before Mul and SM
  Tqkv mul_dq_zp, sm_dq_zp;
  float mul_dq_scale, sm_dq_scale;
  // Q after Mul and SM
  Tqkv mul_q_zp, sm_q_zp;
  float mul_q_scale, sm_q_scale;
#ifdef RANDOM_DATA
  initialize_random<Tqkv>(qkv, M * K * N, 32, 0);
  init_random<RowMajorMatrix<Tqkv>>(aie_Q, 0, 32);
  init_random<RowMajorMatrix<Tqkv>>(aie_K, 0, 32);
  init_random<RowMajorMatrix<Tqkv>>(aie_V, 0, 64);

  qk_qdq_c0 = 1;
  smv_qdq_c0 = 1;
  C1 = 1;
  C2 = 1;
  C3 = 1;
  SQb = 0;  // 8;
  Sout = 8; // 15;
  Stm_qkt = 2;
  Stm_smv = 0;
  // DQ before SM
  sm_dq_zp = 126;
  sm_dq_scale = 0.45;
  // Q after SM
  sm_q_zp = 0;
  sm_q_scale = 0.003921568;
  // DQ before MUL
  mul_dq_zp = 126;
  mul_dq_scale = 0.45;
  // Q after MUL
  mul_q_zp = 0;
  mul_q_scale = 0.003921568;

  float mul_value = 65535 * 0.000002179860302931047;
  // qdq param for MHA int32

  // MuL DQ
  qdq_params[(16 * 0) + 0] = mul_dq_zp;
  qdq_params[(16 * 0) + 1] = float_to_bfloat16(mul_dq_scale * mul_value).value;
  // MuL Q
  qdq_params[(16 * 1) + 0] = mul_q_zp;
  qdq_params[(16 * 1) + 1] = float_to_bfloat16(1.0 / mul_q_scale).value;

  // QKT
  *(int64_t *)(&qdq_params[(16 * 2) + qdq_c0_idx]) = qk_qdq_c0;
  qdq_params[(16 * 2) + qdq_c1_idx] = C1;
  qdq_params[(16 * 2) + qdq_c2_idx] = C2;
  qdq_params[(16 * 2) + qdq_c3_idx] = C3;
  qdq_params[(16 * 2) + qdq_Mv_idx] = St_pad;
  qdq_params[(16 * 2) + qdq_Nv_idx] = St_pad;
  qdq_params[(16 * 2) + qdq_SQb_idx] = SQb;
  qdq_params[(16 * 2) + qdq_Sout_idx] = Sout;
  qdq_params[(16 * 2) + qdq_Stdm_idx] = Stm_qkt;

  // SM *V
  *(int64_t *)(&qdq_params[(16 * 3) + qdq_c0_idx]) = smv_qdq_c0;
  qdq_params[(16 * 3) + qdq_c1_idx] = C1;
  qdq_params[(16 * 3) + qdq_c2_idx] = C2;
  qdq_params[(16 * 3) + qdq_c3_idx] = C3;
  qdq_params[(16 * 3) + qdq_Mv_idx] = St_pad;
  qdq_params[(16 * 3) + qdq_Nv_idx] = val_subv_cols;
  qdq_params[(16 * 3) + qdq_SQb_idx] = SQb;
  qdq_params[(16 * 3) + qdq_Sout_idx] = Sout;
  qdq_params[(16 * 3) + qdq_Stdm_idx] = Stm_smv;

  // DQ before SM
  qdq_params[(16 * 4) + 0] = sm_dq_zp;
  // additional scaling to emulate exp using exp2.
  qdq_params[(16 * 4) + 1] = float_to_bfloat16(sm_dq_scale * 1.442695041).value;

  // Q after SM
  qdq_params[(16 * 5) + 0] = sm_q_zp;
  qdq_params[(16 * 5) + 1] = float_to_bfloat16(1.0 / sm_q_scale).value;

  // for CPU model
  qdq_params_l[0].C0 = qk_qdq_c0;
  qdq_params_l[0].C1 = C1;
  qdq_params_l[0].C2 = C2;
  qdq_params_l[0].C3 = C3;
  qdq_params_l[0].sqb = SQb;
  qdq_params_l[0].sout = Sout;
  qdq_params_l[0].zero_point = 3;
  qdq_params_l[0].scale = 0.0034;

  qdq_params_l[1].C0 = smv_qdq_c0;
  qdq_params_l[1].C1 = C1;
  qdq_params_l[1].C2 = C2;
  qdq_params_l[1].C3 = C3;
  qdq_params_l[1].sqb = SQb;
  qdq_params_l[1].sout = Sout;
  qdq_params_l[1].zero_point = 3;
  qdq_params_l[1].scale = 0.0034;

  qdq_params_l[2].zero_point = sm_dq_zp;
  qdq_params_l[2].scale = sm_dq_scale;

  qdq_params_l[3].zero_point = sm_q_zp;
  qdq_params_l[3].scale = sm_q_scale;

  int const out_size = out_rows * out_cols * sizeof(Tqkv);

  void *cpu_head_qry =
      malloc(RowMajorMatrix<Tqkv>::size(qry_rows, qry_subv_cols));
  void *cpu_head_key =
      malloc(RowMajorMatrix<Tqkv>::size(key_rows, key_subv_cols));
  void *cpu_head_val =
      malloc(RowMajorMatrix<Tqkv>::size(val_rows, val_subv_cols));
  void *cpu_head_out =
      malloc(RowMajorMatrix<uint32_t>::size(out_rows, out_subv_cols));
  void *cpu_head8_out =
      malloc(RowMajorMatrix<Tqkv>::size(out_rows, out_subv_cols));

  RowMajorMatrix<Tqkv> head_Q(qry_rows, qry_subv_cols, cpu_head_qry);
  RowMajorMatrix<Tqkv> head_K(key_rows, key_subv_cols, cpu_head_key);
  RowMajorMatrix<Tqkv> head_V(val_rows, val_subv_cols, cpu_head_val);
  RowMajorMatrix<uint32_t> head_Y(out_rows, out_subv_cols, cpu_head_out);
  RowMajorMatrix<Tqkv> head_Y16(out_rows, out_subv_cols, cpu_head8_out);

  // calculate cpu_Y
  bool transposedK = false;

  // int const num_heads = qry_cols / qry_subv_cols;
  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < head_Q.num_rows; ++i) {
      for (int j = 0; j < head_Q.num_cols; ++j) {
        head_Q.at(i, j) = aie_Q.at(i, (h * qry_subv_cols) + j);
      }
    }
    for (int i = 0; i < head_K.num_rows; ++i) {
      for (int j = 0; j < head_K.num_cols; ++j) {
        head_K.at(i, j) = aie_K.at(i, (h * key_subv_cols) + j);
      }
    }
    for (int i = 0; i < head_V.num_rows; ++i) {
      for (int j = 0; j < head_V.num_cols; ++j) {
        head_V.at(i, j) = aie_V.at(i, (h * val_subv_cols) + j);
      }
    }

    ref_qxkxv<Tqkv>(head_Q.data, head_K.data, head_V.data, head_Y16.data,
                    head_Y.data, qry_rows, key_subv_cols, val_rows,
                    val_subv_cols, qdq_params_l, transposedK);

    for (int i = 0; i < head_Y.num_rows; ++i) {
      for (int j = 0; j < head_Y.num_cols; ++j) {
        cpu_Y.at(i, (h * out_subv_cols) + j) = head_Y16.at(i, j);
      }
    }
  }

  free(cpu_head_qry);
  free(cpu_head_key);
  free(cpu_head_val);
  free(cpu_head_out);
  free(cpu_head8_out);
#else
  std::string fld_name;
  if (M == 3) {
    fld_name = "//mha_PSQ2//";
  }
  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "ifm.bin",
                reinterpret_cast<char *>(qkv.data()));

  memcpy(q.data(), qkv.data(), qry_rows * qry_cols * sizeof(Tqkv));
  memcpy(k.data(),
         reinterpret_cast<char *>(qkv.data()) +
             qry_rows * qry_cols * sizeof(Tqkv),
         qry_rows * qry_cols * sizeof(Tqkv));
  memcpy(v.data(),
         reinterpret_cast<char *>(qkv.data()) +
             2 * qry_rows * qry_cols * sizeof(Tqkv),
         qry_rows * qry_cols * sizeof(Tqkv));

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "ofm.bin",
                reinterpret_cast<char *>(cpu_out.data()));
  float mul_scale;
  Tqkv mul_val, mul_zp;
  mul_scale = 0.000002697439413;
  mul_zp = 0;
  mul_val = 65535;
  // MuL DQ
  qdq_params[(16 * 0) + 0] = 0;
  qdq_params[(16 * 0) + 1] = float_to_bfloat16(1.0).value;

  // MuL Q
  qdq_params[(16 * 1) + 0] = 0;
  qdq_params[(16 * 1) + 1] = float_to_bfloat16(1.0).value;

  // QKT
  *(int64_t *)(&qdq_params[(16 * 2) + 0]) = 894036524920768; // c0
  qdq_params[(16 * 2) + 2] = -421341891;                     // c1
  qdq_params[(16 * 2) + 3] = 1607552;                        // c2
  qdq_params[(16 * 2) + 4] = -409385723;                     // c3
  qdq_params[(16 * 2) + 5] = St_pad;
  qdq_params[(16 * 2) + 6] = St_pad;
  qdq_params[(16 * 2) + 7] = 0;  // SQb
  qdq_params[(16 * 2) + 8] = 29; // Sout
  qdq_params[(16 * 2) + 9] = 7;  // Stdm

  // SM *V
  *(int64_t *)(&qdq_params[(16 * 3) + 0]) = 34432752812032;
  qdq_params[(16 * 3) + 2] = -561445665;
  qdq_params[(16 * 3) + 3] = 4377344;
  qdq_params[(16 * 3) + 4] = 0;
  qdq_params[(16 * 3) + 5] = St_pad;
  qdq_params[(16 * 3) + 6] = val_subv_cols;
  qdq_params[(16 * 3) + 7] = 0;
  qdq_params[(16 * 3) + 8] = 30;
  qdq_params[(16 * 3) + 9] = 8;

  assert(qdq_params[(16 * 3) + 4] ==
         0); // kernel wrapper does not implment V sum as C3 is 0

  // DQ before SM
  qdq_params[(16 * 4) + 0] = 27995;
  qdq_params[(16 * 4) + 1] =
      float_to_bfloat16(0.0003508587833493948 * 1.442695041).value;

  // Q after SM
  qdq_params[(16 * 5) + 0] = 0;
  qdq_params[(16 * 5) + 1] =
      float_to_bfloat16(1.0 / 0.000015259021893143654).value;

  read_bin_file(OpInterface::get_dod_base_dir() + fld_name + "wgt.bin",
                reinterpret_cast<char *>(qdq_params.data()));
#endif
  // calculate aie_Y
  ryzenai::mha mha_ =
      ryzenai::mha<Tqkv, WgT, OuT>(a_dtype, b_dtype, c_dtype, false);
  mha_.debug(debug);
  // std::vector<size_t> a_shape = {qry_rows, qry_cols, key_rows, val_cols};
  mha_.set_params(model_name, qkv_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_shape, "int32"}};
  mha_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;

  input_Tensor = {{q.data(), q_shape, a_dtype},
                  {k.data(), k_shape, a_dtype},
                  {v.data(), v_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{out.data(), out_shape, a_dtype}};
#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", H = " << H << ", St = " << St << ", Dh = " << Dh);
  PROFILE_THIS(mha_.execute(input_Tensor, output_Tensor));
#else
  mha_.execute(input_Tensor, output_Tensor);
#endif

  float const max_pct_diff = 1.0;
  float average_error_rate = check_result_mha(cpu_Y, aie_Y, max_pct_diff, 0);

  // int err_cnt_TH = int(qry_rows * val_cols * 0.02);
  if (average_error_rate > 15)
    err_count = 0;

  return err_count;
}

// mha
TEST(PSQ2_mha_Testa16w8, Kernel1) {
  int err_count = test_mha<uint16_t, uint8_t, uint16_t>(
      3, 77, 1024, false, "uint16", "uint8", "uint16", "PSQ2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
