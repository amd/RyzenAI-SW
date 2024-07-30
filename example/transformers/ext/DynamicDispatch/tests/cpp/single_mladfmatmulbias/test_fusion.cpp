#include <algorithm>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/mladf_matmul_matrix.hpp"
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ops/ops_common.hpp>

#include "test_common.hpp"

template <typename T>
static void initialize_random_mladf(std::vector<T> &vec, size_t size,
                                    int data_max, std::string dtype = "int8") {
  auto data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    if (dtype == "bfloat16") {
      vec[i] = ryzenai::rand_bfloat16(float(data_max));
    } else if (dtype == "uint4") {
      vec[i] = ryzenai::rand_uint4(data_max);
    } else if (dtype == "int4") {
      vec[i] = ryzenai::rand_int4(data_max);
    } else if (std::is_same<T, float>::value) {
      vec[i] = (2.0 * (rand() / (float)RAND_MAX) - 1.0) * data_max;
    } else {
      vec[i] = (T)(rand() % (data_max - data_min + 1)) + data_min;
    }
  }
}
template <typename InT = int16_t, typename WgT = int8_t, typename OuT = int16_t>
static int test_matmul(const std::string &meta_json, size_t M, size_t K,
                       size_t N, bool debug = false,
                       const std::string &a_dtype = "bfloat16",
                       const std::string &b_dtype = "uint4",
                       const std::string &c_dtype = "bfloat16") {
  int group_size = 128;

  std::vector<size_t> a_shape = {1, M, K};
  std::vector<size_t> b_shape = {1, K, N};
  std::vector<size_t> c_shape = {1, M, N};
  std::vector<InT> a(M * K);
  std::vector<float> bias(N);
  std::vector<float> scales(K * N / group_size);
  std::vector<WgT> b(K * N);
  std::vector<WgT> zeros(K * N / group_size);
  int32_t garbage_value = 0xCDCDCDCD;
  std::vector<OuT> c(M * N, garbage_value);
  std::vector<float> c_golden(M * N, garbage_value);

  srand(42);

  // Select the input data range for activations, weights, scales, and bias
  initialize_random_mladf<InT>(a, M * K, 100, "bfloat16");
  initialize_random_mladf<WgT>(b, K * N, 7, "uint4");
  initialize_random_mladf<WgT>(zeros, K * N / group_size, 7, "uint4");
  initialize_random_mladf<float>(bias, N, 1, "float32");
  initialize_random_mladf<float>(scales, K * N / group_size, 1, "float32");

  {

    std::string matmul_dir = "test_mladfmatmul";

    std::ofstream wts_f(matmul_dir + "/0.const",
                        std::ios::out | std::ios::binary);
    std::ofstream bias_f(matmul_dir + "/1.const",
                         std::ios::out | std::ios::binary);
    std::ofstream scales_f(matmul_dir + "/2.const",
                           std::ios::out | std::ios::binary);
    std::ofstream zeros_f(matmul_dir + "/3.const",
                          std::ios::out | std::ios::binary);

    wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
    bias_f.write((char *)bias.data(), bias.size() * sizeof(float));
    scales_f.write((char *)scales.data(), scales.size() * sizeof(float));
    zeros_f.write((char *)zeros.data(), zeros.size() * sizeof(WgT));
  }
  const std::string xclbin_fname =
      Utils::get_env_var("DOD_ROOT") +
      ryzenai::LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;

  auto meta = OpsFusion::load_meta_json(meta_json);

  OpsFusion::FusionRuntime rt(xclbin_fname);

  rt.init(meta);
  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{c.data(), c_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  // compute golden (slow computation, therefore in CI only for small shapes)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      c_golden[m * N + n] = bias[n];
      for (int k = 0; k < K; ++k) {
        float x = ryzenai::bfloat16_to_float(a[m * K + k]);
        int g_idx = (k / group_size);
        float y = (b[k * N + n] - zeros[g_idx * N + n]) *
                  ryzenai::bfloat16_rnd_even(scales[g_idx * N + n]);

        c_golden[m * N + n] +=
            ryzenai::bfloat16_rnd_even(x) * ryzenai::bfloat16_rnd_even(y);
      }
    }
  }

  float const EPSILON_MAX =
      4.0; // this is the tolerated max error, normalized by sqrt(K)
  float const EPSILON_MEAN =
      0.8; // this is the tolerated mean error, normalized by sqrt(K)
  int err_count = 0;
  float err_max = 0;
  float err_min = 0;
  float err_total = 0;
  float err_mean = 0;

  for (int i = 0; i < c.size(); i++) {
    float err = std::abs(ryzenai::bfloat16_rnd_even(c_golden[i]) -
                         ryzenai::bfloat16_to_float(c[i]));
    if (std::abs(err_max) < std::abs(err)) {
      err_max = err;
    }
    if (i == 0) {
      err_min = err;
    } else if (std::abs(err_min) > std::abs(err)) {
      err_min = err;
    }
    err_total += err;
    if (err > EPSILON_MAX * sqrt(K)) {
      err_count++;
      if (err_count < 16) {
        std::cout << std::dec << "c[" << i << "]: "
                  << "Err: " << err << ", "
                  << "Expected: " << ryzenai::bfloat16_rnd_even(c_golden[i])
                  << ", "
                  << "Received: " << ryzenai::bfloat16_to_float(c[i]) << "\n";
      }
    }
  }

  err_mean = err_total / c.size();
  printf("err_max: %.2f, target: %.2f\n", err_max, EPSILON_MAX * sqrt(K));
  printf("err_mean: %.2f, target: %.2f\n", err_mean, EPSILON_MEAN * sqrt(K));

  if (err_count > 0)
    std::cout << std::dec << std::fixed << std::setprecision(2)
              << err_count / c.size()
              << "\% of the values deviate more than allowed." << std::endl;
  bool max_error_violation =
      std::isnan(err_max) || err_max > EPSILON_MAX * sqrt(K);
  bool mean_error_violation =
      std::isnan(err_mean) || err_mean > EPSILON_MEAN * sqrt(K);
  return max_error_violation || mean_error_violation;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_mladfmatmul.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t M = 1;
  size_t K = 4096;
  size_t N = 4096;
  try {
    std::string meta_json = std::string(argv[1]);

    int err_count = 0;
    err_count = test_matmul(meta_json, M, K, N, false);
    if (err_count > 0) {
      std::cout << "single_mladfmatmul test failed with err_count = "
                << err_count << std::endl;
      return EXIT_FAILURE;
    }
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
