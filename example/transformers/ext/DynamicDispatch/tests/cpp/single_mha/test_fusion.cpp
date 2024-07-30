#include <algorithm>
#include <iostream>
#include <limits>
#include <op_fuser/fusion_rt.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/gprb_validation.cpp"
#include "ops/ops_common/mha_validation.cpp"
#include "ops/ops_common/mhagprb_matrix.hpp"
#include <ops/mhagprb/mhagprb.hpp>

#include "../unit_tests/enable_perf.hpp"

#include "test_common.hpp"

static constexpr size_t gprb_vec64_size = 15;

using namespace mhagprb_matrix;
template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_mha(const std::string &meta_json, int St, int Di, int S, int D,
             bool debug = false, const std::string &a_dtype = "int8",
             const std::string &b_dtype = "int8",
             const std::string &c_dtype = "int8") {
  int err_count = 0;
  size_t const qry_rows = St;
  size_t const qry_cols = Di;
  size_t const key_rows = S;
  size_t const key_cols = Di;
  size_t const val_rows = S;
  size_t const val_cols = D;
  size_t const out_rows = St;
  size_t const out_cols = D;

  size_t H = 12;
  std::vector<size_t> q_shape = {1, qry_rows, qry_cols};
  std::vector<size_t> k_shape = {1, key_rows, key_cols};
  std::vector<size_t> v_shape = {1, val_rows, val_cols};
  std::vector<size_t> msk_shape = {1, 1, 1, key_rows};

  // gprb proj_mat (96x8) unit8
  std::vector<size_t> weight_shape = {gprb_rows, gprb_cols};
  // gprb.qdq_bias (8) + gprb_c0_scalar + QK_qdq_c0_scalar + SMV_qdq_c0_scalar
  std::vector<size_t> gprb_vec64_shape = {gprb_vec64_size};
  // gprb int32 bit qdq params (11) + model_a (12) + model_b (1) + model_c (1)
  std::vector<size_t> gprb_vec32_shape = {QDQparam_size * 2};
  // bias 12x512x512 uint8
  std::vector<size_t> bias_shape = {H, qry_rows, key_rows};
  // qdq param for MHA int32
  std::vector<size_t> qdq_shape = {num_qdq_nodes, QDQparam_size};
  std::vector<size_t> out_shape = {1, out_rows, out_cols};

  int weight_size = GPRB_buf_size; // mat+qdq+padding

  int bias_size = H * qry_rows * key_rows;

  std::vector<InT> q(qry_rows * qry_cols);
  std::vector<InT> k(key_rows * key_cols);
  std::vector<InT> v(val_rows * val_cols);
  std::vector<uint16_t> msk(1 * key_rows);
  std::vector<WgT> weight(gprb_rows * gprb_cols);
  std::vector<int64_t> gprb_vec64(gprb_vec64_size);
  std::vector<int32_t> gprb_vec32(QDQparam_size * 2);
  GprbParams<int64_t, InT, gprb_rows, gprb_cols, num_heads>
      gprb_dataqdq; // internal gprb structure
  std::vector<WgT> bias(H * qry_rows * key_rows);
  std::vector<int32_t> qdq_params(QDQparam_size * num_qdq_nodes);

  std::vector<OutT> out(out_rows * out_cols, garbage_value);

  RowMajorMatrix<InT> aie_Q(qry_rows, qry_cols, q.data());
  RowMajorMatrix<InT> aie_K(key_rows, key_cols, k.data());
  RowMajorMatrix<InT> aie_V(val_rows, val_cols, v.data());
  gemm_qdq_param<InT> qdq_params_l[4];

  // Initialize q, k and v
  srand(0xABCD);
  int64_t qk_qdq_c0, smv_qdq_c0;
  int32_t C1, C2, C3;
  uint8_t SQb, Sout, Stm_qkt, Stm_smv;
  // DQ before SM
  uint8_t dq_zp;
  float dq_scale;
  // Q after SM
  uint8_t q_zp;
  float q_scale;
  if constexpr (std::is_same_v<InT, uint16_t>) {
    init_random<RowMajorMatrix<InT>>(aie_Q, 0, 32);
    init_random<RowMajorMatrix<InT>>(aie_K, 0, 32);
    init_random<RowMajorMatrix<InT>>(aie_V, 120, 150);
    RowMajorMatrix<uint16_t> mskMatrix(1, key_rows, msk.data());
    init_random_bfloat16(mskMatrix, 0, 5);
    init_gprb_params(gprb_dataqdq);
    initialize_random<InT>(bias, H * qry_rows * key_rows, 128, 0);

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
    dq_zp = 126;
    dq_scale = 0.45;
    // Q after SM
    q_zp = 0;
    q_scale = 0.003921568;

  } else { // uint8
    init_random<RowMajorMatrix<InT>>(aie_Q, 0, 32);
    init_random<RowMajorMatrix<InT>>(aie_K, 0, 32);
    init_random<RowMajorMatrix<InT>>(aie_V, 0, 64);
    RowMajorMatrix<uint16_t> mskMatrix(1, key_rows, msk.data());
    init_random_bfloat16(mskMatrix, 0, 5);
    init_gprb_params(gprb_dataqdq);
    initialize_random<WgT>(bias, H * qry_rows * key_rows, 255, 0);

    qk_qdq_c0 = 1;
    smv_qdq_c0 = 1;
    C1 = 1;
    C2 = 1;
    C3 = 1;
    SQb = 0;  // 8;
    Sout = 8; // 15;
    Stm_qkt = 0;
    Stm_smv = 0;
    // DQ before SM
    dq_zp = 126;
    dq_scale = 0.45;
    // Q after SM
    q_zp = 0;
    q_scale = 0.003921568;
  }

  // gprb proj_mat (96x8) unit8
  for (int i = 0; i < gprb_rows * gprb_cols; ++i) {
    weight[i] = gprb_dataqdq.proj_mat[i];
  }
  // gprb.qdq_bias (8) + gprb_c0_scalar + QK_qdq_c0_scalar + SMV_qdq_c0_scalar
  for (int i = 0; i < gprb_cols; ++i) {
    gprb_vec64[i] = gprb_dataqdq.qdq_bias[i];
  }
  gprb_vec64[gprb_c0_scalar_idx] = gprb_dataqdq.c0;
  gprb_vec64[qk_qdq_c0_scalar_idx] = qk_qdq_c0;
  gprb_vec64[smv_qdq_c0_scalar_idx] = smv_qdq_c0;

  // gprb int32 bit qdq params (11) + model_a (12) + model_b (1) + model_c (1)
  gprb_vec32[qdq_c1_idx] = gprb_dataqdq.c1;
  gprb_vec32[qdq_c2_idx] = gprb_dataqdq.c2;
  gprb_vec32[qdq_c3_idx] = gprb_dataqdq.c3;
  gprb_vec32[qdq_Mv_idx] = gprb_dataqdq.M;
  gprb_vec32[qdq_Nv_idx] = gprb_dataqdq.N;
  gprb_vec32[qdq_SQb_idx] = gprb_dataqdq.shift_Qb;
  gprb_vec32[qdq_Sout_idx] = gprb_dataqdq.shift_Qout;
  gprb_vec32[gprb_act_scale_idx] = gprb_dataqdq.act_scale.value;
  gprb_vec32[gprb_act_zero_idx] = gprb_dataqdq.act_zero_point;
  gprb_vec32[gprb_wgt_scale_idx] = gprb_dataqdq.wgt_scale.value;
  gprb_vec32[gprb_wgt_zero_idx] = gprb_dataqdq.wgt_zero_point;
  for (int h = 0; h < num_heads; ++h) {
    gprb_vec32[gprb_model_a_idx + h] = gprb_dataqdq.model_a[h].value;
  }
  gprb_vec32[gprb_model_b_idx] = gprb_dataqdq.model_b.value;
  gprb_vec32[gprb_model_c_idx] = gprb_dataqdq.model_c.value;

  // qdq param for MHA int32
  // QKT
  *(int64_t *)(&qdq_params[(16 * 0) + qdq_c0_idx]) = qk_qdq_c0;
  qdq_params[(16 * 0) + qdq_c1_idx] = C1;
  qdq_params[(16 * 0) + qdq_c2_idx] = C2;
  qdq_params[(16 * 0) + qdq_c3_idx] = C3;
  qdq_params[(16 * 0) + qdq_Mv_idx] = qry_subv_rows;
  qdq_params[(16 * 0) + qdq_Nv_idx] = key_subv_rows;
  qdq_params[(16 * 0) + qdq_SQb_idx] = SQb;
  qdq_params[(16 * 0) + qdq_Sout_idx] = Sout;
  qdq_params[(16 * 0) + qdq_Stdm_idx] = Stm_qkt;

  // SM *V
  *(int64_t *)(&qdq_params[(16 * 1) + qdq_c0_idx]) = smv_qdq_c0;
  qdq_params[(16 * 1) + qdq_c1_idx] = C1;
  qdq_params[(16 * 1) + qdq_c2_idx] = C2;
  qdq_params[(16 * 1) + qdq_c3_idx] = C3;
  qdq_params[(16 * 1) + qdq_Mv_idx] = qry_subv_rows;
  qdq_params[(16 * 1) + qdq_Nv_idx] = val_subv_cols;
  qdq_params[(16 * 1) + qdq_SQb_idx] = SQb;
  qdq_params[(16 * 1) + qdq_Sout_idx] = Sout;
  qdq_params[(16 * 1) + qdq_Stdm_idx] = Stm_smv;

  // DQ before SM
  qdq_params[(16 * 2) + 0] = dq_zp;
  qdq_params[(16 * 2) + 1] = float_to_bfloat16(dq_scale).value;

  // Q after SM
  qdq_params[(16 * 3) + 0] = q_zp;
  qdq_params[(16 * 3) + 1] = float_to_bfloat16(1.0 / q_scale).value;

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

  qdq_params_l[2].zero_point = dq_zp;
  qdq_params_l[2].scale = dq_scale;

  qdq_params_l[3].zero_point = q_zp;
  qdq_params_l[3].scale = q_scale;

  // auto const ptr = reinterpret_cast<unsigned char *>(&gprb_dataqdq);
  // std::vector<uint8_t> weight(ptr, ptr + sizeof(gprb_dataqdq));

  int const out_size = out_rows * out_cols * sizeof(OutT);

  void *cpu_head_qry =
      malloc(RowMajorMatrix<InT>::size(qry_rows, qry_subv_cols));
  void *cpu_head_key =
      malloc(RowMajorMatrix<InT>::size(key_rows, key_subv_cols));
  void *cpu_head_val =
      malloc(RowMajorMatrix<InT>::size(val_rows, val_subv_cols));
  void *cpu_head_msk = malloc(RowMajorMatrix<uint16_t>::size(1, key_rows));
  void *cpu_head_scl = malloc(RowMajorMatrix<WgT>::size(qry_rows, key_rows));
  void *cpu_head_bias = malloc(RowMajorMatrix<float>::size(qry_rows, key_rows));
  void *cpu_head32_out =
      malloc(RowMajorMatrix<uint32_t>::size(out_rows, out_subv_cols));
  void *cpu_head_out =
      malloc(RowMajorMatrix<OutT>::size(out_rows, out_subv_cols));
  void *cpu_out = malloc(out_size);

  RowMajorMatrix<InT> head_Q(qry_rows, qry_subv_cols, cpu_head_qry);
  RowMajorMatrix<InT> head_K(key_rows, key_subv_cols, cpu_head_key);
  RowMajorMatrix<InT> head_V(val_rows, val_subv_cols, cpu_head_val);
  RowMajorMatrix<uint16_t> head_M(1, key_rows, cpu_head_msk);
  RowMajorMatrix<WgT> head_S(qry_rows, key_rows, cpu_head_scl);
  RowMajorMatrix<float> head_B(qry_rows, key_rows, cpu_head_bias);
  RowMajorMatrix<uint32_t> head_Y32(out_rows, out_subv_cols, cpu_head32_out);
  RowMajorMatrix<OutT> head_Y(out_rows, out_subv_cols, cpu_head_out);

  RowMajorMatrix<OutT> cpu_Y(out_rows, out_cols, cpu_out);
  RowMajorMatrix<OutT> aie_Y(out_rows, out_cols, out.data());

  // calculate cpu_Y
  bool transposedK = false;
  for (int i = 0; i < key_rows; ++i) {
    head_M.at(0, i) = msk[i];
  }

  // int const num_heads = qry_cols / qry_subv_cols;
  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < head_S.num_rows; ++i) {
      for (int j = 0; j < head_S.num_cols; ++j) {
        head_S.at(i, j) = bias[h * head_S.num_rows * head_S.num_cols +
                               i * head_S.num_cols + j]; // aie_S.at(h, i, j);
      }
    }
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
    // calculate_gprb<InT, WgT>(head_Q, head_S, aie_W, head_B);
    calculate_gprb<InT, InT, float, gprb_rows, gprb_cols, num_heads>(
        head_Q, head_S, gprb_dataqdq, head_B, h);
    ref_qxkxv<InT>(head_Q.data, head_K.data, head_V.data, head_B.data,
                   head_M.data, head_Y.data, head_Y32.data, qry_rows,
                   key_subv_cols, val_rows, val_subv_cols, qdq_params_l,
                   transposedK);
    for (int i = 0; i < head_Y32.num_rows; ++i) {
      for (int j = 0; j < head_Y32.num_cols; ++j) {
        cpu_Y.at(i, (h * out_subv_cols) + j) = head_Y.at(i, j);
      }
    }
  }

  free(cpu_head_qry);
  free(cpu_head_key);
  free(cpu_head_val);
  free(cpu_head_msk);
  free(cpu_head_scl);
  free(cpu_head_bias);
  free(cpu_head32_out);
  free(cpu_head_out);

  // calculate aie_Y
  ryzenai::mhagrpb mhagrpb_ =
      ryzenai::mhagrpb<InT, WgT, OutT>(a_dtype, b_dtype, c_dtype, true);
  mhagrpb_.debug(debug);

  std::vector<Tensor> const_Tensor;

  // weight_shape is smaller than the actual weight_size
  const_Tensor = {{weight.data(), weight_shape, "uint8"},
                  {gprb_vec64.data(), gprb_vec64_shape, "int64"},
                  {gprb_vec32.data(), gprb_vec32_shape, "int32"},
                  {bias.data(), bias_shape, b_dtype},
                  {qdq_params.data(), qdq_shape, "int32"}};

  {
    int i = 0;
    for (const auto &tensor : const_Tensor) {
      std::string filename = "test_mha/"s + std::to_string(i) + ".const";
      std::ofstream ofs(filename, std::ios::binary);
      ofs.write((char *)tensor.data,
                std::accumulate(tensor.shape.begin(), tensor.shape.end(),
                                size_t{1}, std::multiplies{}) *
                    Utils::get_size_of_type(tensor.dtype));
      ofs.close();
      ++i;
    }
  }

  // mhagrpb_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{q.data(), q_shape, a_dtype},
                  {k.data(), k_shape, a_dtype},
                  {v.data(), v_shape, a_dtype},
                  {msk.data(), msk_shape, "uint16"}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{out.data(), out_shape, c_dtype}};

  auto meta = OpsFusion::load_meta_json(meta_json);
  std::string xclbin_fname = PSF_A8W8_QDQ_XCLBIN_REL_PATH;
  OpsFusion::FusionRuntime rt(xclbin_fname);
  rt.init(meta);

#ifdef UNIT_TEST_PERF
  LOG_THIS("St = " << St << ", Di = " << Di << ", S = " << S << ", D = " << D);
  PROFILE_THIS(rt.execute(input_Tensor, output_Tensor));
#else
  rt.execute(input_Tensor, output_Tensor);
#endif

  float const max_pct_diff = 1.0;
  float average_error_rate = check_result(cpu_Y, aie_Y, max_pct_diff, 0);

  // int err_cnt_TH = int(qry_rows * val_cols * 0.02);
  if (average_error_rate > 15)
    err_count = 1;

  free(cpu_out);

  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ops_fusion.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;

  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = test_mha<uint8_t, uint8_t, uint8_t>(
        meta_json, 512, 1152, 512, 768, false, "uint8", "uint8", "uint8");
    std::cout << "Total errors : " << err_count << std::endl;

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
