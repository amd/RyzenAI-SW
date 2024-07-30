#include "mladfelwadd_helpers.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>

#include "ops/ops_common/matmul_matrix.hpp"

#include "test_common.hpp"
using namespace matmul_matrix;

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_mladfelwadd(const std::string &meta_json,
                     std::vector<std::vector<size_t>> inputs_shape,
                     bool debug = false, const std::string &a_dtype = "uint16",
                     const std::string &b_dtype = "uint16",
                     const std::string &c_dtype = "uint16") {
  int err_count = 0;
  mladfelwadd_helpers::process_shape(inputs_shape.at(0));
  mladfelwadd_helpers::process_shape(inputs_shape.at(1));
  auto bd_type = mladfelwadd_helpers::determine_broadcast_type(inputs_shape);

  std::vector<size_t> a_shape = inputs_shape.at(0);
  std::vector<size_t> b_shape = inputs_shape.at(1);

  size_t M = a_shape[0];
  size_t K = std::accumulate(a_shape.begin() + 1, a_shape.end(), size_t{1},
                             std::multiplies{});

  size_t b_size = std::accumulate(b_shape.begin(), b_shape.end(), size_t{1},
                                  std::multiplies{});
  size_t qdq_size = 24;
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

  int32_t matA_zero_point = 41733;
  double matA_scale = 0.0008634580299258232;
  int32_t matB_zero_point = 19916;
  double matB_scale = 0.0001138541119871661;
  OuT matC_zero_point = 42933;
  double matC_scale = 0.0008906692964956164;

  // This function mimic  kernel run on aie. Please refer to for details:
  // https://gitenterprise.xilinx.com/AIELibs/mllib/blob/be1ed3964a5ddeb2e03b3357bd156fcf3bf41f61/internal/demo/win24/sd/kernels/python/operators/AddA16.py#L55
  auto results = mladfelwadd_helpers::compute_qdq(
      matA_scale, matB_scale, matC_scale, matA_zero_point, matB_zero_point,
      matC_zero_point);

  int ifm1_coeff, ifm2_coeff, zero_point_coeff;
  int8_t ofm_shift, ifm1_shift, ifm2_shift, zero_point_shift;
  std::tie(ifm1_coeff, ifm2_coeff, zero_point_coeff, ofm_shift, ifm1_shift,
           ifm2_shift, zero_point_shift) = results;

  uint32_t ifmsv_size = M * K;
  mladfelwadd_helpers::assign_qdq_params(
      qdq_params, ifm1_coeff, ifm2_coeff, zero_point_coeff, ofm_shift,
      ifm1_shift, ifm2_shift, zero_point_shift, ifmsv_size, bd_type);

  std::string xclbin_fname;
  if (a_dtype == "uint16")
    xclbin_fname = MLADF_4x2_ELWADD_A16W16_QDQ_XCLBIN_PATH;
  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt(xclbin_fname);
  rt.init(meta);

  std::vector<Tensor> input_tensor;
  Tensor a_T = {a.data(), a_shape, a_dtype};
  Tensor b_T = {b.data(), b_shape, b_dtype};

  Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_tensor.push_back(a_T);
  // Assume non-broacasting weight is non-const in PSS and PST model.
  // If weight is non-const, put it in input_Tensor
  if (a_shape == b_shape) {
    input_tensor.push_back(b_T);
  } else {
    std::vector<char> bin_data =
        OpsFusion::read_bin_file("test_mladfelwadd/0.const");
    memcpy((char *)b.data(), bin_data.data(), b_size * sizeof(WgT));
  }
  // Compute CPU output
  mladfelwadd_helpers::compute_cpu_output(
      a, b, cpu_out, M, K, ifm1_coeff, ifm2_coeff, zero_point_coeff, ofm_shift,
      ifm1_shift, ifm2_shift, zero_point_shift, bd_type);

  std::vector<Tensor> output_tensor;
  output_tensor.push_back(c_T);
  rt.execute(input_tensor, output_tensor);

  err_count = check_add_result(cpu_Q_Y, aie_Y, 0.01);

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

    // err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
    //     meta_json, {{128, 256, 256}, {128, 1, 1}}, false, "uint16", "uint16",
    //     "uint16");
    err_count = test_mladfelwadd<uint16_t, uint16_t, uint16_t>(
        meta_json, {{128, 512, 512}, {128, 512, 512}}, false, "uint16",
        "uint16", "uint16");
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
