#include "mladfsoftmax_helpers.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/mladfsoftmax/mladfsoftmax.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

using namespace matmul_matrix;

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_mladfsoftmax(const std::string &meta_json, std::vector<size_t> shape,
                      bool debug = false, const std::string &a_dtype = "uint16",
                      const std::string &mask_dtype = "uint8",
                      const std::string &c_dtype = "uint16") {
  size_t M = shape[0];
  size_t K = std::accumulate(shape.begin() + 1, shape.end(), size_t{1},
                             std::multiplies{});

  std::vector<size_t> a_shape = shape;

  std::vector<InT> a(M * K);
  std::vector<OuT> cpu_out(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_out.data());

  std::string xclbin_fname;
  if (a_dtype == "uint16") {
    xclbin_fname = MLADF_SOFTMAX_A16_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("Unsupported data type: " + a_dtype);
  }

  std::string data_path_prefix = "tests/cpp/unit_tests/testDataMladf/";
  // std::string model_part = "pss_softmax_4096_4096/";
  std::string model_part = "pst_softmax_4096_4096/";
  std::string a_bin_path = data_path_prefix + model_part + "ifm.bin";
  std::string ofm_bin_path = data_path_prefix + model_part + "ofm.bin";

  mladfsoftmax_helpers::read_bin_to_vector(a_bin_path, a);
  mladfsoftmax_helpers::read_bin_to_vector(ofm_bin_path, cpu_out);

  OpsFusion::FusionRuntime rt(xclbin_fname);
  auto meta = OpsFusion::load_meta_json(meta_json);
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  input_Tensors.push_back(a_T);

  std::vector<Tensor> output_Tensors;
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  output_Tensors.push_back(c_T);
  rt.execute(input_Tensors, output_Tensors);
  return check_result(cpu_Q_Y, aie_Y);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_mladfsoftmax.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  size_t M = 4096;
  size_t N = 4096;

  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_mladfsoftmax<uint16_t, uint8_t, uint16_t>(
        meta_json, {M, N}, false, "uint16", "uint8", "uint16");
    if (err_count > 0) {
      std::cout << "MLADFSOFTMAX test failed with err_count = " << err_count
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
