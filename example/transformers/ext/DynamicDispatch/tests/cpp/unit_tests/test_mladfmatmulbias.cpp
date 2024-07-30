/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "ops/ops_common/matrix_formatting.h"
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"

template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size, int data_max,
                              std::string dtype = "int8") {
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

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_matmul_mladf(int M, int K, int N, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "int4",
                      const std::string &c_dtype = "bfloat16",
                      int group_size = 128, bool compare_values = true) {

  if (b_dtype == "int4" || b_dtype == "uint4") {
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};
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
    initialize_random<InT>(a, M * K, 100, "bfloat16");
    initialize_random<WgT>(b, K * N, 7, b_dtype);
    initialize_random<WgT>(zeros, K * N / group_size, 7, b_dtype);
    initialize_random<float>(bias, N, 1);
    initialize_random<float>(scales, K * N / group_size, 1);

    ryzenai::mladfmatmulbias mladfmatmulbias_ =
        ryzenai::mladfmatmulbias<InT, WgT, OuT, OuT>(a_dtype, b_dtype, c_dtype,
                                                     true);

    mladfmatmulbias_.debug(debug);

    size_t Ms = static_cast<size_t>(M);
    size_t Ks = static_cast<size_t>(K);
    size_t Ns = static_cast<size_t>(N);

    std::vector<Tensor> const_Tensor;
    std::vector<size_t> v_shape_vec = {Ms, Ks};
    std::vector<size_t> b_shape_vec = {Ks, Ns};
    std::vector<size_t> size_shape = {static_cast<size_t>(group_size), 0};
    const_Tensor = {{b.data(), b_shape_vec, b_dtype},
                    {bias.data(), size_shape, a_dtype},
                    {scales.data(), size_shape, a_dtype},
                    {zeros.data(), b_shape_vec, b_dtype}};

    mladfmatmulbias_.initialize_const_params(const_Tensor);

    std::vector<Tensor> input_Tensor;
    std::vector<size_t> a_shape_vec = {Ms, Ks};

    input_Tensor = {{a.data(), a_shape_vec, a_dtype}};

    std::vector<Tensor> output_Tensor;
    std::vector<size_t> c_shape_vec = {Ms, Ns};
    output_Tensor = {{c.data(), c_shape_vec, c_dtype}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("M = " << M << ", K = " << K << ", N = " << N
                    << ", Gs = " << group_size);
    PROFILE_THIS(mladfmatmulbias_.execute(input_Tensor, output_Tensor));
#else

    mladfmatmulbias_.execute(input_Tensor, output_Tensor);
#endif
    if (!compare_values)
      return 0;
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
}

// Formal test

TEST(Qlinear_2Testw3a16, Kernel4mladf2) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf3) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf4) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf5) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22528, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf6) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 32768, true, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf9) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf10) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf11) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf12) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22528, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf13) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 32768, true, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf8) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf21) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf22) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf23) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 12288, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf24) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 22528, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf25) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 32768, true, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf20) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 11008, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// ------------- TEST END for mladfmatmulbias -------------
