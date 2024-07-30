
#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>

#include "ops/ops_common/lrn_matrix.hpp"
#include <ops/layernorm/layernorm.hpp>

#include "test_common.hpp"

using namespace lrn_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_lrn(const std::string &meta_json, int M, int N, bool debug = false,
             const std::string &a_dtype = "int16",
             const std::string &b_dtype = "int16",
             const std::string &c_dtype = "int16") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> gamma_shape = {Ns};
  std::vector<size_t> beta_shape = {Ns};
  std::vector<size_t> aie_out_shape = {1, Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<float> gamma(N); // for CPU calculation
  std::vector<float> beta(N);  // for CPU calculation
  std::vector<WgT> aie_gamma(N);
  std::vector<WgT> aie_beta(N);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> aie_out(M * N, garbage_value);
  std::vector<int32_t> qdq_params(QDQparam_size);

  std::vector<WgT> b(2 * N);
  BiasVector<WgT, 1> bias(N, b.data());

  srand(0xABCD);
  initialize_random_bfloat16(a, M * N, -20, 20);
  initialize_random_bfloat16(b, 2 * N, -1, 1);
  // init_random_bias(bias, -2, 2); // float to bfloat16

  std::vector<std::vector<InT>> In(M);
  std::vector<std::vector<float>> Out(M);
  ActMatrix<OutT, 1, 1> cpu_Y(M, N, cpu_out.data());

  // initialize golden inputs
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      In[r].push_back(a[r * N + c]);
    }
  }

  // auto gamma_16 = OpsFusion::read_bin_file<uint16_t>("test_lrn/0.const");
  // auto beta_16 = OpsFusion::read_bin_file<uint16_t>("test_lrn/1.const");

  for (int c = 0; c < N; c++) {
    gamma[c] = (bfloat2float(bias.gamma(c)));
    beta[c] = (bfloat2float(bias.beta(c)));
    aie_gamma[c] = bias.gamma(c);
    aie_beta[c] = bias.beta(c);
    // std::cout << "gamm & beta : " << bias.gamma(c) << " " << bias.beta(c)
    //           << std::endl;
    // std::cout << "gamm & beta : " << bfloat2float(bias.gamma(c)) << " "
    //           << bfloat2float(bias.beta(c)) << std::endl;
  }

  // compute golden
  compute_lrn_bfloat16(In, gamma, beta, Out);

  // quantize output
  float sc_float = 0.1;
  int16_t sc_out = float2bfloat(1.0 / sc_float); // bfloat16
  InT zp_out = 129;
  qdq_params[0] = (int32_t)sc_out;
  qdq_params[1] = (int32_t)zp_out;

  {
    std::string lrn_dir = "test_lrn";
    if (c_dtype == "uint16")
      lrn_dir += "_a16w8";
    else
      lrn_dir += "_a8w8";
    std::ofstream gamma_f(lrn_dir + "/0.const",
                          std::ios::out | std::ios::binary);
    std::ofstream beta_f(lrn_dir + "/1.const",
                         std::ios::out | std::ios::binary);
    std::ofstream qdq_f(lrn_dir + "/2.const", std::ios::out | std::ios::binary);
    gamma_f.write((char *)aie_gamma.data(), aie_gamma.size() * sizeof(WgT));
    beta_f.write((char *)aie_beta.data(), aie_beta.size() * sizeof(WgT));
    qdq_f.write((char *)qdq_params.data(), qdq_params.size() * sizeof(int32_t));
  }

  if (c_dtype == "uint16") {
    q_bfloat2uint16(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  } else {
    q_bfloat2uint8(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  }

  std::string xclbin_fname;
  if (c_dtype == "uint16") {
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

  // compare results
  int max_error = 0;
  int error_limit = 40;
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      int32_t diff = std::abs(aie_out[r * N + c] - cpu_Y.at(r, c));
      if (diff > error_limit) {
        std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                  << "Expected: " << (int)cpu_Y.at(r, c) << ", "
                  << "Received: " << (int)aie_out[r * N + c] << "\n";
        err_count++;
      }
      max_error = (diff > max_error) ? diff : max_error;
    }
  }

  std::cout << "Maximum Difference : " << max_error << std::endl;

  if (max_error <= error_limit) {
    err_count = 0;
  }

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
      err_count = test_lrn<int16_t, int16_t, uint16_t>(
          meta_json, 128, 768, false, "bfloat16", "uint16", "uint16");
    } else {
      err_count = test_lrn<int16_t, int16_t, uint8_t>(
          meta_json, 512, 768, false, "bfloat16", "uint16", "uint8");
    }
    if (err_count > 0) {
      std::cout << "LRN test failed with err_count = " << err_count
                << std::endl;
      return EXIT_FAILURE;
    }
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
