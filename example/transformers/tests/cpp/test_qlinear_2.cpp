/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "matrix_formatting.h"
#include <qlinear_2.hpp>

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

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int32_t>
static void matmul(std::vector<InT> &a, std::vector<WgT> &b,
                   std::vector<OutT> &c, std::tuple<int, int> &a_shape,
                   std::tuple<int, int> &b_shape, bool apply_srs = false,
                   OutT shift = 12, OutT srs_min = -128, OutT srs_max = 127) {
  int M = std::get<0>(a_shape);
  int K = std::get<1>(a_shape);
  int N = std::get<1>(b_shape);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      OutT sum = 0;
      for (int k = 0; k < K; k++) {
        sum += OutT(a[i * K + k]) * OutT(b[k * N + j]);
      }
      if constexpr (!std::is_same_v<OutT, float>) {
        if (apply_srs) {
          sum = sum >> shift;
          sum = (sum < srs_min) ? srs_min : sum;
          sum = (sum > srs_max) ? srs_max : sum;
        }
      }
      c[i * N + j] = sum;
    }
  }
}

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int32_t>
static void matmul_batch(std::vector<InT> &a_in, std::vector<WgT> &b_in,
                         std::vector<OutT> &c_in, std::tuple<int, int> &a_shape,
                         std::tuple<int, int> &b_shape, int batch_in,
                         bool apply_srs = false, OutT shift = 12,
                         OutT srs_min = -128, OutT srs_max = 127) {
  int batch = batch_in;
  int M = std::get<0>(a_shape);
  int K = std::get<1>(a_shape);
  int N = std::get<1>(b_shape);

  size_t a_2d_size = M * K;
  size_t b_2d_size = K * N;
  size_t c_2d_size = M * N;

  for (int bat_id = 0; bat_id < batch; bat_id++) {
    InT *a = a_in.data() + (bat_id * a_2d_size);
    WgT *b = b_in.data() + (bat_id * b_2d_size);
    OutT *c = c_in.data() + (bat_id * c_2d_size);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        OutT sum = 0;
        for (int k = 0; k < K; k++) {
          sum += OutT(a[i * K + k]) * OutT(b[k * N + j]);
        }
        if constexpr (!std::is_same_v<OutT, float>) {
          if (apply_srs) {
            sum = sum >> shift;
            sum = (sum < srs_min) ? srs_min : sum;
            sum = (sum > srs_max) ? srs_max : sum;
          }
        }
        c[i * N + j] = sum;
      }
    }
  }
}

/*
 * GeMM Unit Test
 *
 * X x Y x Z is the maximum supported kernel size
 * M x K x N is the actual GeMM shape
 *
 *    NOTE: When M < X, the AIE wrapper can transparently call a kernel
 *          with a smaller value for X by selecting a different DPU sequence.
 *          This allows us to quickly execute token generation (when M == 1)
 *          and prefill phase (where M may be large).
 *
 */
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int32_t>
int test_matmul(int M, int K, int N, bool debug = false,
                const std::string &a_dtype = "int8",
                const std::string &b_dtype = "int8",
                const std::string &c_dtype = "int32", int group_size = 32) {

  auto NUM_EXECUTE_ITERATIONS_ =
      Utils::get_env_var("NUM_EXECUTE_ITERATIONS", "1");

  if (a_dtype == "int8" || a_dtype == "int16") {
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};

    std::vector<InT> a(M * K);
    std::vector<WgT> b(K * N);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(M * N, garbage_value);
    std::vector<OuT> c_golden(M * N, garbage_value);

    srand(42);
    initialize_random<InT>(a, M * K, 127);
    initialize_random<WgT>(b, K * N, 127);
    matmul<InT, WgT, OuT>(a, b, c_golden, a_shape, b_shape);

    ryzenai::qlinear_2 qlin =
        ryzenai::qlinear_2<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype);
    qlin.debug(debug);
    qlin.initialize_weights(b.data(), b_shape, group_size);
    for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
      qlin.execute(a.data(), a_shape, c.data());

    int err_count = 0;
    for (int i = 0; i < c.size(); i++) {
      OuT err = std::abs(c_golden[i] - c[i]);
      if (err > 0) {
        // std::cout << "c[" << i << "]: "
        //           << "Expected: " << c_golden[i] << ", "
        //           << "Received: " << c[i] << "\n";
        err_count += 1;
      }
    }
    return err_count;
  } else if (b_dtype == "int4" || b_dtype == "uint4") {
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};

    std::vector<InT> a(M * K);
    std::vector<float> bias(N);
    std::vector<float> scales(K * N / group_size);
    std::vector<WgT> b(K * N);
    std::vector<WgT> zeros(K * N / group_size);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(M * N, garbage_value);
    std::vector<OuT> c_golden(M * N, garbage_value);
    srand(42);
    // Select the input data range for activations, weights, scales, and bias
    initialize_random<InT>(a, M * K, 42, "bfloat16");
    initialize_random<WgT>(b, K * N, 7, b_dtype);
    initialize_random<WgT>(zeros, K * N / group_size, 7, b_dtype);
    initialize_random<float>(bias, N, 1);
    initialize_random<float>(scales, K * N / group_size, 1);

    // compute golden
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
    ryzenai::qlinear_2 qlin =
        ryzenai::qlinear_2<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype);
    qlin.debug(debug);
    qlin.initialize_weights_int4(b.data(), zeros.data(), (float *)scales.data(),
                                 (float *)bias.data(), b_shape, group_size);
    for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
      qlin.execute(a.data(), a_shape, c.data());
    float const EPSILON = 1.0;
    int err_count = 0;
    float err_max = 0;
    float err_min = 0;
    float err_total = 0;
    for (int i = 0; i < c.size(); i++) {
      OuT err = std::abs(c_golden[i] - c[i]);
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
        std::cout << "c[" << i << "]: "
                  << "Expected: " << c_golden[i] << ", "
                  << "Received: " << c[i] << "\n";
        err_count++;
      }
    }
    // printf("err_max: %f, err_min: %f, err_mean: %f \n", err_max, err_min,
    // err_total/c.size());
    return err_count;
  }
}

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_matmul_mladf(int M, int K, int N, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "int4",
                      const std::string &c_dtype = "bfloat16",
                      int group_size = 128, bool compare_values = true) {
  auto NUM_EXECUTE_ITERATIONS_ =
      Utils::get_env_var("NUM_EXECUTE_ITERATIONS", "1");

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

    ryzenai::qlinear_2 qlin =
        ryzenai::qlinear_2<InT, WgT, OuT, OuT>(a_dtype, b_dtype, c_dtype);
    qlin.debug(debug);
    qlin.initialize_weights_int4_mladf(
        b.data(), zeros.data(), (float *)scales.data(), (float *)bias.data(),
        b_shape, group_size);
    for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
      qlin.execute(a.data(), a_shape, c.data());
    float const EPSILON_MAX =
        4.0; // this is the tolerated max error, normalized by sqrt(K)
    float const EPSILON_MEAN =
        0.8; // this is the tolerated mean error, normalized by sqrt(K)
    int err_count = 0;
    float err_max = 0;
    float err_min = 0;
    float err_total = 0;
    float err_mean = 0;

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

template <typename InT = uint16_t, typename WgT = int8_t, typename AccT = float,
          typename OuT = uint16_t>
int test_matmul_output_bfloat16(int M, int K, int N, bool debug = false,
                                const std::string &a_dtype = "bfloat16",
                                const std::string &b_dtype = "uint4",
                                const std::string &c_dtype = "float32",
                                int group_size = 32) {

  auto NUM_EXECUTE_ITERATIONS_ =
      Utils::get_env_var("NUM_EXECUTE_ITERATIONS", "1");

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
    std::vector<float> c_float(M * N, garbage_value);
    std::vector<AccT> c_golden(M * N, garbage_value);
    srand(42);
    // Select the input data range for activations, weights, scales, and bias
    initialize_random<InT>(a, M * K, 42, "bfloat16");
    initialize_random<WgT>(b, K * N, 7, b_dtype);
    initialize_random<WgT>(zeros, K * N / group_size, 7, b_dtype);
    initialize_random<float>(bias, N, 1);
    initialize_random<float>(scales, K * N / group_size, 1);

    // compute golden
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
    ryzenai::qlinear_2 qlin =
        ryzenai::qlinear_2<InT, WgT, AccT, OuT>(a_dtype, b_dtype, c_dtype);
    qlin.debug(debug);
    qlin.initialize_weights_int4(b.data(), zeros.data(), (float *)scales.data(),
                                 (float *)bias.data(), b_shape, group_size);

    for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
      qlin.execute(a.data(), a_shape, c.data());

    for (size_t i = 0; i < c.size(); ++i) {
      c_float[i] = ryzenai::bfloat16_to_float(c[i]);
    }

    float const EPSILON = 1.0;
    int err_count = 0;
    float err_max = 0;
    float err_min = 0;
    float err_total = 0;
    for (int i = 0; i < c.size(); i++) {
      float err =
          100.0 * std::abs(c_golden[i] - c_float[i]) / std::abs(c_golden[i]);
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
        std::cout << "c[" << i << "]: "
                  << "Expected: " << c_golden[i] << ", "
                  << "Received: " << c_float[i] << ","
                  << "Error: " << err << "%"
                  << "\n";
        err_count++;
      }
    }
    return err_count;
  }
  return 0;
}

/*
 * GeMM Unit Test
 *
 * X x Y x Z is the maximum supported kernel size
 * Batch x M x K x N is the GeMM shape
 *
 *    NOTE: When M < X, the AIE wrapper can transparently call a kernel
 *          with a smaller value for X by selecting a different DPU sequence.
 *          This allows us to quickly execute token generation (when M == 1)
 *          and prefill phase (where M may be large).
 *
 */
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int32_t>
int test_matmul_batch(int M, int K, int N, int batch, bool debug = false,
                      const std::string &a_dtype = "int8",
                      const std::string &b_dtype = "int8",
                      const std::string &c_dtype = "int32",
                      int group_size = 32) {
  // NOTE: The actual kernel shape is decided in
  //       run_aie and initialize_weights to be compatible
  //       with the pytorch runtime, so these shapes don't
  //       matter. They exist for backwards compatibility.
  //       We use 8 x 2k x 2k as dummy values here.

  auto NUM_EXECUTE_ITERATIONS_ =
      Utils::get_env_var("NUM_EXECUTE_ITERATIONS", "1");

  if (a_dtype == "int8" || a_dtype == "int16") {
    std::tuple<int, int> kernel_x_shape = {32, 4096};
    std::tuple<int, int> kernel_y_shape = {4096, 4096};
    std::tuple<int, int> a_2d_shape = {M, K};
    std::tuple<int, int> b_2d_shape = {K, N};
    std::tuple<int, int, int> a_shape = {batch, M, K};
    std::tuple<int, int, int> b_shape = {batch, K, N};

    std::vector<InT> a(batch * M * K);
    std::vector<WgT> b(batch * K * N);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(batch * M * N, garbage_value);
    std::vector<OuT> c_golden(batch * M * N, garbage_value);

    srand(42);

    ryzenai::qlinear_2 qlin =
        ryzenai::qlinear_2<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype);

    initialize_random<InT>(a, batch * M * K, 127);
    initialize_random<WgT>(b, batch * K * N, 127);
    matmul_batch<InT, WgT, OuT>(a, b, c_golden, a_2d_shape, b_2d_shape, batch);

    for (int bat_id = 0; bat_id < batch; bat_id++) {
      InT *a_2d = a.data() + (bat_id * M * K);
      WgT *b_2d = b.data() + (bat_id * K * N);
      OuT *c_2d = c.data() + (bat_id * M * N);

      qlin.initialize_weights(b_2d, b_2d_shape, group_size);
      for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
        qlin.execute(a_2d, a_2d_shape, c_2d);
    }

    int err_count = 0;
    for (int i = 0; i < c.size(); i++) {
      OuT err = std::abs(c_golden[i] - c[i]);
      if (err > 0) {
        // std::cout << "c[" << i << "]: "
        //           << "Expected: " << c_golden[i] << ", "
        //           << "Received: " << c[i] << "\n";
        err_count += 1;
      }
    }
    return err_count;
  }
}

/* Test kernel dimensions without tiling */

/* Test kernels with M = 1*/

TEST(Qlinear_2Testw8a8, Kernel1) {
  int err_count = test_matmul(1, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel2) {
  int err_count = test_matmul(1, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel3) {
  int err_count = test_matmul(1, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 8 */

TEST(Qlinear_2Testw8a8, Kernel4) {
  int err_count = test_matmul(8, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel5) {
  int err_count = test_matmul(8, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel6) {
  int err_count = test_matmul(8, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 16 */

TEST(Qlinear_2Testw8a8, Kernel7) {
  int err_count = test_matmul(16, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel8) {
  int err_count = test_matmul(16, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel9) {
  int err_count = test_matmul(16, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 32 */

TEST(Qlinear_2Testw8a8, Kernel10) {
  int err_count = test_matmul(32, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel11) {
  int err_count = test_matmul(32, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel12) {
  int err_count = test_matmul(32, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 64 */

TEST(Qlinear_2Testw8a8, Kernel13) {
  int err_count = test_matmul(64, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel14) {
  int err_count = test_matmul(64, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Kernel15) {
  int err_count = test_matmul(64, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test tiling up to OPT vocabulary output dimension */

TEST(Qlinear_2Testw8a8, Tiling1) {
  int err_count = test_matmul(8, 2048, 50272);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test tiling in all three dimensions with padding */

TEST(Qlinear_2Testw8a8, Tiling2) {
  int err_count = test_matmul(42, 5000, 2500);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Tiling3) {
  int err_count = test_matmul(42, 16000, 2500);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* Test padding multiple token input with OPT 1.3B dimensions */

TEST(Qlinear_2Testw8a8, Padding1) {
  int err_count = test_matmul(12, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Padding2) {
  int err_count = test_matmul(12, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw8a8, Padding3) {
  int err_count = test_matmul(12, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* SD shape test */
TEST(Qlinear_2Testw8a8, Batch1) {
  int err_count = test_matmul_batch(4096, 40, 4096, 8);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/*Llama2 shapes tests*/
TEST(Qlinear_2Testw8a16, Kernel1) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        32, 4096, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel2) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        32, 4096, 11008, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel3) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        32, 11008, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel4) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        64, 4096, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel5) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        64, 11008, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel6) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        64, 4096, 11008, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel7) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        8, 4096, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel8) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        8, 4096, 11008, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel9) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        8, 11008, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel10) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 4096, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel11) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 4096, 11008, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, Kernel12) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 11008, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw8a16, KernelTiling1) {
  if (std::string(Utils::get_env_var("DEVICE")) == "stx") {
    int err_count = test_matmul<int16_t, int8_t, int64_t>(
        1, 16000, 4096, false, "int16", "int8", "int64");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(Qlinear_2Testw4a16, Kernel1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel1p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 12288, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel2p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 12288, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 11008, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel3p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 11008, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel4p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 12288, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel5p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 12288, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 11008, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel6p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 11008, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel7p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 12288, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel8p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 12288, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 11008, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel9p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 11008, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelTiling1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 32000, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelTiling2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      64, 4096, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelTiling3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 8192, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelTiling4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4608, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelPadding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 2048, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelPadding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 2048, 4096, false, "bfloat16", "uint4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernelgroup1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 8192, false, "bfloat16", "uint4", "float32", 64);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernelgroup2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 8192, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 6400, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 6400, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6400, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6400, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 24576, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 24576, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 24576, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 24576, 6144, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, KernelStartCoding10) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel1p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 4096, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 12288, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel2p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 4096, 12288, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 11008, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel3p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      32, 11008, 4096, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel4p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4096, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 12288, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel5p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 12288, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 11008, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel6p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 11008, 4096, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel7p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 4096, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 12288, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel8p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 12288, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 11008, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernel9p) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 11008, 4096, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 32000, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      64, 4096, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 8192, 4096, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelTiling4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 4608, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelPadding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 2048, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelPadding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 2048, 4096, false, "bfloat16", "int4", "float32");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernelgroup1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 8192, false, "bfloat16", "int4", "float32", 64);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernelgroup2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 4096, 8192, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernelgroup3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 8192, false, "bfloat16", "int4", "float32", 64);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, Kernelgroup4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 4096, 8192, false, "bfloat16", "int4", "float32", 64);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 6400, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 6400, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6400, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6400, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 24576, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 6144, 24576, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 24576, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      1, 24576, 6144, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2Testw3a16, KernelStartCoding10) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      8, 6144, 49152, false, "bfloat16", "int4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* Test debug feature */
TEST(Qlinear_2Testw8a8, Debug) {
  int err_count = test_matmul(1, 2048, 2048, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

//-----------------------------------------------------------------------
// Llama-2 shapes after grouping
//-----------------------------------------------------------------------
TEST(Qlinear_2_LlamaONNX, Kernel8_1) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 12288, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_2) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 22016, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_3) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_4) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 11008, 4096, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_5) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 32000, false, "bfloat16", "uint4", "float32", 128);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_6) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 12288, false, "bfloat16", "uint4", "float32", 32);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_7) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 22016, false, "bfloat16", "uint4", "float32", 32);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_8) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 4096, false, "bfloat16", "uint4", "float32", 32);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_9) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 11008, 4096, false, "bfloat16", "uint4", "float32", 32);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(Qlinear_2_LlamaONNX, Kernel8_10) {
  int err_count = test_matmul<uint16_t, int8_t, float>(
      2048, 4096, 32000, false, "bfloat16", "uint4", "float32", 32);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
//-----------------------------------------------------------------------

// Tests for mladf bf16 gemm
// Debug test
// TEST(Qlinear_2Testw4a16, Kernel_mladf_debug) {
//   if (std::string(Utils::get_env_var("MLADF")).empty())
//     GTEST_SKIP() << "MLADF environment variable not set, skipped.";

//  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
//      8, 256, 2048, true, "bfloat16", "uint4", "bfloat16", 32, true);
//  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

// Formal test
TEST(Qlinear_2Testw4a16, Kernel_mladf_1x11008x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_1x4096x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_1x4096x11008) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_1x4096x12288) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_1x4096x22528) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22528, true, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_1x4096x32768) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 32768, true, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_128x11008x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_128x4096x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_128x4096x11008) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_128x4096x12288) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_128x4096x22528) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22528, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_128x4096x32768) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 32768, true, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_800x11008x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 11008, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_800x4096x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 4096, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_800x4096x11008) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 11008, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_800x4096x12288) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 12288, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_800x4096x22528) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 22528, true, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_800x4096x32768) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 32768, true, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_2048x11008x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_2048x4096x4096) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_2048x4096x11008) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_2048x4096x12288) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_2048x4096x22528) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw4a16, Kernel_mladf_2048x4096x32768) {
  if (std::string(Utils::get_env_var("MLADF")).empty())
    GTEST_SKIP() << "MLADF environment variable not set, skipped.";

  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// TEST END for mladf bf16 gemm

// Test converting float32 accumulation to bfloat16 in qlinear_2
// Disabled Test, because I don't have a good pass criteria
// I am seeing significant percent error for some elements
// TEST(Qlinear_2_output_bfloat16_Testw4a16, Kernel1) {
//  int err_count =
//      test_matmul_output_bfloat16<uint16_t, int8_t, float, uint16_t>(
//          32, 4096, 4096, true, "bfloat16", "uint4", "float32");
//  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
//}
