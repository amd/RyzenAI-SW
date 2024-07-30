
#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>

#include "ops/ops_common/lrn_matrix.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/elwmul/elwmul.hpp>

#include "test_common.hpp"

// z = x * y
template <typename LhsT = int16_t, typename RhsT = int16_t,
          typename OuT = int16_t>
static int test_elwmul(const std::string &meta_json, size_t M, size_t N,
                       bool debug = false,
                       const std::string &x_dtype = "bfloat16",
                       const std::string &y_dtype = "bfloat16",
                       const std::string &z_dtype = "bfloat16") {

  // Without the prefix 1 in the batch dimension this does not work
  std::vector<size_t> x_shape = {1, M, N};

  std::vector<LhsT> x(M * N);
  std::vector<LhsT> y(M * N);
  std::vector<OuT> z(M * N, garbage_value);
  std::vector<OuT> z_golden(M * N, garbage_value);

  srand(42);
  lrn_matrix::initialize_random_bfloat16(x, M * N, -42, 42);
  lrn_matrix::initialize_random_bfloat16(y, M * N, -42, 42);

  // compute golden
  for (int i = 0; i < M * N; ++i) {
    z_golden[i] =
        matmul_matrix::float2bfloat(matmul_matrix::bfloat16_to_float(x[i]) *
                                    matmul_matrix::bfloat16_to_float(y[i]));
  }

  const auto &xclbin_fname =
      LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_REL_PATH;
  OpsFusion::FusionRuntime rt(xclbin_fname);

  auto meta = OpsFusion::load_meta_json(meta_json);
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{x.data(), x_shape, x_dtype}, {y.data(), x_shape, y_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{z.data(), x_shape, z_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  float const EPSILON = 0.0;

  int err_count = 0;
  float err_max = 0;
  float err_min = 0;
  float err_total = 0;
  for (int i = 0; i < y.size(); i++) {
    float err = std::abs(matmul_matrix::bfloat16_to_float(z_golden[i]) -
                         matmul_matrix::bfloat16_to_float(z[i]));
    if (std::abs(err_max) < std::abs(err)) {
      err_max = err;
    }
    if (i == 0) {
      err_min = err;
    } else if (std::abs(err_min) > std::abs(err)) {
      err_min = err;
    }
    err_total += err;
    if (err > EPSILON) {
      std::cout << "x[" << i << "]: " << matmul_matrix::bfloat16_to_float(x[i])
                << std::endl;
      std::cout << "y[" << i << "]: " << matmul_matrix::bfloat16_to_float(y[i])
                << std::endl;
      std::cout << "Z[" << i << "]: ";
      std::cout << "Expected: " << matmul_matrix::bfloat16_to_float(z_golden[i])
                << ", "
                << "Received: " << matmul_matrix::bfloat16_to_float(z[i])
                << std::endl;
      err_count++;
    }
  }

  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_elwmul.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }
  std::runtime_error("HERE");
  std::cout << std::fixed;
  size_t M = 1;
  size_t N = 11008;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_elwmul(meta_json, M, N, false);
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
