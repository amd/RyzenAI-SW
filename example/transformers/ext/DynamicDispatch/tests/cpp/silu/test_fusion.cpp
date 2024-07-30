#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <ops/silu/silu.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "test_common.hpp"

// y = x * sigmoid(x)
template <typename InT = uint16_t, typename OuT = uint16_t>
static int test_silu(const std::string &meta_json, size_t M, size_t N,
                     bool debug = false,
                     const std::string &x_dtype = "bfloat16",
                     const std::string &y_dtype = "bfloat16") {

  std::vector<size_t> x_shape = {1, M, N};
  std::vector<size_t> y_shape = {1, M, N};

  std::vector<InT> x(M * N);
  std::vector<OuT> y(M * N, garbage_value);
  std::vector<float> y_golden(M * N, garbage_value);

  srand(42);
  dd::initialize_random_bfloat16(x, 42);

  // compute golden
  for (int i = 0; i < M * N; ++i) {
    float xf = dd::bfloat16_to_float(x[i]);
    float sigmoid = 1.0f / (1.0f + std::exp(-xf));
    float intermediate = xf * sigmoid;
    y_golden[i] = intermediate;
  }

  const std::string xclbin_fname =
      LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_REL_PATH;

  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt(xclbin_fname);
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{x.data(), x_shape, x_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{y.data(), y_shape, y_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  return dd::count_errors_floatvsbfloat16(
      y_golden, y, y_shape, ryzenai::silu<uint16_t, uint16_t>::EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_silu.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t M = 1;
  size_t N = 11008;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_silu(meta_json, M, N, false);
    if (err_count > 0) {
      std::cout << "Silu test failed with err_count = " << err_count
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
