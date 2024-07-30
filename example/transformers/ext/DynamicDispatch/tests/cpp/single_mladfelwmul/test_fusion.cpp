#include "mladfelwmul_helpers.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>

#include <fstream>

#include "ops/ops_common/matmul_matrix.hpp"

#include "test_common.hpp"
using namespace matmul_matrix;

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_mladfelwmul(const std::string &meta_json,
                     std::vector<std::vector<size_t>> inputs_shape,
                     bool debug = false, const std::string &a_dtype = "uint16",
                     const std::string &b_dtype = "uint16",
                     const std::string &c_dtype = "uint16") {
  int err_count = 0;

  std::vector<size_t> a_shape = inputs_shape.at(0);
  std::vector<size_t> b_shape = inputs_shape.at(1);

  size_t M = a_shape[0];
  size_t K = std::accumulate(a_shape.begin() + 1, a_shape.end(), size_t{1},
                             std::multiplies{});
  if (a_shape[0] == 1) {
    M = a_shape[1];
    K = std::accumulate(a_shape.begin() + 2, a_shape.end(), size_t{1},
                        std::multiplies{});
  }

  size_t b_size = std::accumulate(b_shape.begin(), b_shape.end(), size_t{1},
                                  std::multiplies{});
  size_t qdq_size = 22;
  std::vector<size_t> qdq_params_shape = {qdq_size};

  std::vector<InT> a(M * K);
  std::vector<InT> b(b_size);
  std::vector<OuT> cpu_out(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_out.data());
  std::vector<int8_t> qdq_params(qdq_size);

  initialize_random<InT>(a, M * K, 65535, 1);
  initialize_random<WgT>(b, b_size, 65535, 1);
  // wgt is const so wgt data need to be written to be read by load_const func
  // in fusion_rt

  // hardcode QDQ parameter
  int matA_zero_point = 41733;
  double matA_scale = 0.0008634580299258232;
  int matB_zero_point = 19916;
  double matB_scale = 0.0001138541119871661;
  int matC_zero_point = 42933;
  double matC_scale = 0.0008906692964956164;
  auto results = mladfelwmul_helpers::compute_qdq(
      matA_scale, matB_scale, matC_scale, matA_zero_point, matB_zero_point,
      matC_zero_point);

  int c0_shift, coeff0, c1_shift, coeff1;
  std::tie(c0_shift, coeff0, c1_shift, coeff1) = results;

  uint32_t ifmsv_size = M * K;
  mladfelwmul_helpers::assign_qdq_params(qdq_params, coeff0, coeff1,
                                         matA_zero_point, matB_zero_point,
                                         c0_shift, c1_shift, ifmsv_size);

  {
    // this mul_dir should be consistent with 'dir_name' in model.py
    std::string mul_dir = "test_mladfelwmul";
    std::ofstream wts_f(mul_dir + "/0.const", std::ios::out | std::ios::binary);
    confirmOpen(wts_f);
    wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
    wts_f.close();

    std::ofstream qdq_f(mul_dir + "/1.const", std::ios::out | std::ios::binary);
    confirmOpen(qdq_f);
    qdq_f.write((char *)qdq_params.data(), qdq_size);
    qdq_f.close();
  }

  // Compute CPU output
  auto b_expand = b;
  if (M == 4096 && K == 4096) {
    b_expand = std::vector<WgT>(4096, b[0]);
  }
  mladfelwmul_helpers::compute_cpu_output(a, b_expand, cpu_out, M, K, coeff0,
                                          coeff1, matA_zero_point,
                                          matB_zero_point, c0_shift, c1_shift);

  std::string xclbin_fname;
  if (a_dtype == "uint16")
    xclbin_fname = MLADF_4x2_ELWMUL_A16W16_QDQ_XCLBIN_PATH;
  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt(xclbin_fname);
  rt.init(meta);
  std::vector<Tensor> input_tensor;
  Tensor a_T = {a.data(), a_shape, a_dtype};
  Tensor b_T = {b.data(), b_shape, b_dtype};

  Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_tensor.push_back(a_T);

  std::vector<Tensor> output_tensor;
  output_tensor.push_back(c_T);
  rt.execute(input_tensor, output_tensor);

  err_count = check_result(cpu_Q_Y, aie_Y);

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

    err_count = test_mladfelwmul<uint16_t, uint16_t, uint16_t>(
        meta_json, {{1, 512, 256, 256}, {512, 1, 1}}, false, "uint16", "uint16",
        "uint16");
    if (err_count > 1) {
      std::cout << "EltwiseMul Test failed with err count : " << err_count
                << std::endl;
      return EXIT_FAILURE;
    } else {
      std::cout << "EltwiseMul Test Passed with err count : " << err_count
                << std::endl;
    }

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
