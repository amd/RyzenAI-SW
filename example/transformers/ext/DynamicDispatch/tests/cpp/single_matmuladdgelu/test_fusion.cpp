#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/matmulgeluadd/matmulgeluadd.hpp>

#include "test_common.hpp"

using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
static int test_matmulgelu_fusion(const std::string &meta_json, size_t M,
                                  size_t K, size_t N, bool debug = false,
                                  const std::string &a_dtype = "int16",
                                  const std::string &b_dtype = "int8",
                                  const std::string &c_dtype = "int32") {
  int err_count = 0;
  int Msubv_act = 0;
  if (a_dtype == "uint16") {
    Msubv_act = 32;
  } else if (a_dtype == "uint8") {
    Msubv_act = 64;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }
  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> b_shape = {K, N};
  std::vector<size_t> qdq_shape = {N};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {1, M, N};
  std::vector<size_t> bias_shape = {0};
  if (a_dtype == "uint8")
    bias_shape = {N};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<int64_t> qdq(1 * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> gelu_qdq_params(QDQparam_size);
  // std::vector<InT> bias(1 * N);
  std::vector<int32_t> cpu_out(M * N, garbage_value);
  std::vector<OuT> cpu_out_qdq(M * N, garbage_value);
  std::vector<uint16_t> gelu_out_golden(M * N, garbage_value);
  std::vector<OuT> gelu_out_golden_quant(M * N, garbage_value);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<WgT> W(K, N, b.data());
  // BiasVector<InT, 1, Nsubv> B(N, bias.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N, cpu_out_qdq.data());
  // golden Q(X*W)
  RowMajorMatrix<uint16_t> gelu_out_gold_mat(M, N, gelu_out_golden.data());
  // golden Q(X*W)
  RowMajorMatrix<OuT> gelu_out_gold_quant_mat(M, N,
                                              gelu_out_golden_quant.data());

  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 16);
  initialize_random<WgT>(b, K * N, 16, 0);
  initialize_random<int64_t>(qdq, 1 * N, 255, 0);
  // init_random(B, -16, 16);

  int32_t C1 = 0;
  int32_t C2 = 20;
  int64_t *C0_vec = (int64_t *)qdq.data();
  uint8_t SQb = 0;
  uint8_t Sout = 13;
  int64_t c0 = 0;
  uint8_t Stdm = 0;

  if (a_dtype == "uint16") {
    srand(0xABCD);
    init_random(X, 0, 32);
    initialize_random<WgT>(b, K * N, 32, 0);
    initialize_random<int64_t>(qdq, 1 * N, 10, -10);
    c0 = 0;
    C1 = -11;
    SQb = 0;
    Sout = 11;
    Stdm = round(log2(K)) - 8;
    C2 = 2 << Stdm;
  }
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;                // C1
  qdq_params[qdq_c2_idx] = C2;                // C2
  qdq_params[qdq_c3_idx] = 0;                 // C3
  // qdq_params[5] = Msubv;          // M
  // qdq_params[6] = Nsubv;          // N
  qdq_params[qdq_SQb_idx] = SQb;   // Shift_Qb
  qdq_params[qdq_Sout_idx] = Sout; // Shift_ou
  qdq_params[qdq_Stdm_idx] = Stdm;

  // NOTE: Set the Q/DQ params here
  InT gelu_in_dq_zero_point = 138;
  float gelu_in_dq_scale = 0.02688;
  InT gelu_out_q_zero_point = 13;
  float gelu_out_q_scale = 0.01295;

  if constexpr (std::is_same_v<InT, uint16_t>) {
    gelu_in_dq_zero_point = 23799;
    gelu_in_dq_scale = 0.00101977;
    gelu_out_q_zero_point = 261;
    gelu_out_q_scale = 0.00065204;
  }
  // GELU IN dequant params
  gelu_qdq_params[gelu_dq_zp_idx] = gelu_in_dq_zero_point;
  gelu_qdq_params[gelu_dq_scale_idx] = float_to_bfloat16(gelu_in_dq_scale);
  // GELU OUT quant params
  gelu_qdq_params[gelu_q_zp_idx] = gelu_out_q_zero_point;
  gelu_qdq_params[gelu_q_scale_idx] = float_to_bfloat16(1.0 / gelu_out_q_scale);

  {
    std::string matmul_dir = "test_matmuladdgelu";
    if (a_dtype == "uint16")
      matmul_dir += "_a16w8";
    else
      matmul_dir += "_a8w8";
    std::ofstream wts_f(matmul_dir + "/0.const",
                        std::ios::out | std::ios::binary);
    std::ofstream qdq_f(matmul_dir + "/1.const",
                        std::ios::out | std::ios::binary);
    std::ofstream qdq_params_f(matmul_dir + "/2.const",
                               std::ios::out | std::ios::binary);
    std::ofstream gelu_qdq_params_f(matmul_dir + "/3.const",
                                    std::ios::out | std::ios::binary);
    wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
    qdq_f.write((char *)qdq.data(), qdq.size() * sizeof(int64_t));
    qdq_params_f.write((char *)qdq_params.data(),
                       qdq_params.size() * sizeof(int32_t));
    gelu_qdq_params_f.write((char *)gelu_qdq_params.data(),
                            gelu_qdq_params.size() * sizeof(int32_t));
  }
  std::string Ytype;
  if (a_dtype == "uint16") {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, Stdm, Msubv_act, Ksubv,
                                        Nsubv);
    Ytype = "uint16";
  } else {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, "int32");

    Ytype = "uint8";
  }
  qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>, RowMajorMatrix<OuT>>(
      X, cpu_Y, C2, C1, C0_vec, SQb, Sout, cpu_Y_qdq, Ytype);

  // Compute GELU golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold =
          (cpu_Y_qdq.at(r, c) - gelu_in_dq_zero_point) * gelu_in_dq_scale;
      gelu_out_gold_mat.at(r, c) = float_to_bfloat16(gelu_golden(in_gold));
    }
  }
  // Quantisze gelu output
  quant(gelu_out_gold_mat, gelu_out_gold_quant_mat, gelu_out_q_scale,
        gelu_out_q_zero_point, Ytype);

  std::string xclbin_fname;
  if (a_dtype == "uint16") {
    xclbin_fname = PSJ_A16W8_QDQ_XCLBIN_REL_PATH;
  } else {
    xclbin_fname = PSF_A8W8_QDQ_XCLBIN_REL_PATH;
  }
  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt(xclbin_fname);
  rt.init(meta);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

  rt.execute(input_Tensor, output_Tensor);

  err_count = check_result(gelu_out_gold_quant_mat, aie_Y);

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
    int err_count = 0;
    if (meta_json.find("a16w8") != string::npos) {
      err_count = test_matmulgelu_fusion<uint16_t, uint8_t, uint16_t>(
          meta_json, 128, 768, 3072, false, "uint16", "uint8", "uint16");
    } else {
      err_count = test_matmulgelu_fusion<uint8_t, uint8_t, uint8_t>(
          meta_json, 512, 768, 3072, false, "uint8", "uint8", "uint8");
    }
    std::cout << "Total error count : " << err_count << std::endl;
    if (err_count > 0) {
      std::cout << "MatMulGeluAdd test failed " << std::endl;
      return EXIT_FAILURE;
    }

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
