#include <algorithm>
#include <iostream>
#include <numeric>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include <fstream>

#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/elwadd/elwadd.hpp>

#include "test_common.hpp"

using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_elwadd(const std::string &meta_json, size_t M, size_t K,
                bool debug = false, const std::string &a_dtype = "int16",
                const std::string &b_dtype = "int8",
                const std::string &c_dtype = "int32") {
  int err_count = 0;

  std::vector<size_t> a_shape = {1, M, K};
  std::vector<size_t> b_shape = {M, K};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * K);
  std::vector<InT> b(M * K);
  std::vector<OuT> cpu_out(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);
  std::vector<int32_t> qdq_params(QDQparam_size);

  int32_t matA_zero_point;
  float matA_scale;
  int32_t matB_zero_point;
  float matB_scale;
  InT rand_max_a, rand_min_a, rand_max_b, rand_min_b;

  if constexpr (std::is_same_v<InT, uint8_t>) {
    matA_zero_point = 2;
    matA_scale = 2.0;
    matB_zero_point = 2;
    matB_scale = 2.0;
    rand_max_a = 16;
    rand_min_a = 0;
    rand_max_b = 16;
    rand_min_b = 0;
  } else if constexpr (std::is_same_v<InT, uint16_t>) {
    matA_zero_point = 4451;
    matA_scale = 0.001;
    matB_zero_point = 1000;
    matB_scale = 0.0002;
    rand_max_a = 4600;
    rand_min_a = 4300;
    rand_max_b = 1200;
    rand_min_b = 800;
  }

  initialize_random<InT>(a, M * K, rand_max_a, rand_min_a);
  initialize_random<InT>(b, M * K, rand_max_b, rand_min_b);

  qdq_params[0] = float_to_bfloat16(matA_scale);
  qdq_params[1] = matA_zero_point;
  qdq_params[2] = float_to_bfloat16(matB_scale);
  qdq_params[3] = matB_zero_point;

  std::vector<OuT> a_dq(M * K);
  std::vector<OuT> b_dq(M * K);

  dequant_to_bfloat(a, a_dq, matA_zero_point, matA_scale);
  dequant_to_bfloat(b, b_dq, matB_zero_point, matB_scale);

  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      cpu_out.at(r * K + c) =
          float_to_bfloat16(bfloat16_to_float(a_dq.at(r * K + c)) +
                            bfloat16_to_float(b_dq.at(r * K + c)));
    }
  }
  {
    std::string add_dir = "test_add";
    if (a_dtype == "uint16")
      add_dir += "_a16w8";
    else
      add_dir += "_a8w8";
    std::ofstream qdq_f(add_dir + "/0.const", std::ios::out | std::ios::binary);
    // std::ofstream beta_f(lrn_dir + "/1.const", std::ios::out |
    // std::ios::binary); std::ofstream qdq_f(lrn_dir + "/2.const",
    // std::ios::out | std::ios::binary); gamma_f.write((char
    // *)aie_gamma.data(), aie_gamma.size() * sizeof(WgT)); beta_f.write((char
    // *)aie_beta.data(), aie_beta.size() * sizeof(WgT));
    qdq_f.write((char *)qdq_params.data(), qdq_params.size() * sizeof(int32_t));
  }
  std::string xclbin_fname;
  if (a_dtype == "uint16")
    xclbin_fname = PSJ_A16W8_QDQ_XCLBIN_REL_PATH;
  else
    xclbin_fname = PSF_A8W8_QDQ_XCLBIN_REL_PATH;

  auto meta = OpsFusion::load_meta_json(meta_json);

  OpsFusion::FusionRuntime rt(xclbin_fname);
  rt.init(meta);

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), a_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  rt.execute(input_Tensor, output_Tensor);

  err_count = check_add_result_bfloat16<OuT>(cpu_out, aie_out, b_shape);

  size_t my_cnt = 0;
  float max_error = 0;
  for (int i = 0; i < cpu_out.size(); ++i) {
    max_error = std::max(max_error, std::abs(bfloat16_to_float(cpu_out[i]) -
                                             bfloat16_to_float(aie_out[i])));
  }
  std::cout << "Max error in float : " << max_error << std::endl;

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
      err_count = test_elwadd<uint16_t, uint16_t, uint16_t>(
          meta_json, 128, 768, false, "uint16", "uint16", "bfloat16");
    } else {
      err_count = test_elwadd<uint8_t, uint8_t, uint16_t>(
          meta_json, 512, 768, false, "uint8", "uint8", "bfloat16");
    }
    if (err_count > 1) {
      std::cout << "EltwiseAdd Test failed with err count : " << err_count
                << std::endl;
      return EXIT_FAILURE;
    } else {
      std::cout << "EltwiseADd Test Passed with err count : " << err_count
                << std::endl;
    }

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
