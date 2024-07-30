/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "matrix_formatting.h"
#include <silu.hpp>

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

/*
 * silu Unit Test
 *
 * X is the maximum supported kernel size
 * M x N is the actual silu shape which must be linearized
 *
 */
template <typename InOutT = int16_t>
int test_silu(int M, int N, bool debug = false,
              const std::string &a_dtype = "bfloat16",
              const std::string &b_dtype = "bfloat16",
              const std::string &c_dtype = "bfloat16") {

  auto NUM_EXECUTE_ITERATIONS_ =
      Utils::get_env_var("NUM_EXECUTE_ITERATIONS", "1");

  assert(a_dtype == "bfloat16" &&
         "Currently only supporting bfloat16 -> silu -> bfloat16");
  assert(a_dtype == b_dtype &&
         "Currently only supporting homogeneous input and output data types");
  assert(a_dtype == c_dtype &&
         "Currently only supporting homogeneous input and output data types");
  std::tuple<int, int> a_shape = {M, N};

  std::vector<InOutT> a(M * N);
  std::vector<InOutT> b(M * N);
  int32_t garbage_value = 0xCDCDCDCD;
  std::vector<InOutT> c(M * N, garbage_value);
  std::vector<InOutT> c_golden(M * N, garbage_value);
  srand(42);

  // Select the input data range for activations
  initialize_random<InOutT>(a, M * N, 42, "bfloat16");

  // compute goldens
  for (int i = 0; i < M * N; ++i) {
    float x = ryzenai::bfloat16_to_float(a[i]);
    float sigmoid = 1.0f / (1.0f + std::exp(-x));
    float intermediate = x * sigmoid;
    c_golden[i] = ryzenai::float_to_bfloat16(intermediate);
  }

  ryzenai::silu siluKernel = ryzenai::silu<InOutT>(a_dtype, b_dtype, c_dtype);

  siluKernel.debug(debug);
  for (int i = 0; i < stoi(NUM_EXECUTE_ITERATIONS_); i++)
    siluKernel.execute(a.data(), b.data(), c.data(), a_shape);

  float const EPSILON =
      0; // 0.0861652 According to
         // https://confluence.amd.com/display/XDCG/Validation+of+SiLU+and+Elementwise+Mul+on+STX#ValidationofSiLUandElementwiseMulonSTX-AccuracytestforSiLU;
  int err_count = 0;
  float err_max = 0;
  float err_min = 0;
  float err_total = 0;
  for (int i = 0; i < c.size(); i++) {
    InOutT err = std::abs(ryzenai::bfloat16_to_float(c_golden[i]) -
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
    if (err > EPSILON && false) {
      std::cout << "a[" << i << "]: " << ryzenai::bfloat16_to_float(a[i])
                << "\n";
      std::cout << "c[" << i << "]: "
                << "Expected: " << ryzenai::bfloat16_to_float(c_golden[i])
                << ", "
                << "Received: " << ryzenai::bfloat16_to_float(c[i]) << "\n";
      err_count++;
    }
  }
  // printf("err_max: %f, err_min: %f, err_mean: %f \n", err_max, err_min,
  // err_total/c.size());
  return err_count;
}

/* Test kernel dimensions without tiling */
/*Llama2 shapes tests*/
TEST(siluNoPadding, Kernel_1x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    int err_count =
        test_silu<int16_t>(1, 11008, false, "bfloat16", "bfloat16", "bfloat16");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_128x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    int err_count = test_silu<int16_t>(128, 11008, false, "bfloat16",
                                       "bfloat16", "bfloat16");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_256x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    int err_count = test_silu<int16_t>(256, 11008, false, "bfloat16",
                                       "bfloat16", "bfloat16");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_512x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    int err_count = test_silu<int16_t>(512, 11008, false, "bfloat16",
                                       "bfloat16", "bfloat16");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_800x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    // Not supported
    EXPECT_THROW(std::ignore = test_silu<int16_t>(800, 11008, false, "bfloat16",
                                                  "bfloat16", "bfloat16"),
                 std::runtime_error);
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_1024x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    int err_count = test_silu<int16_t>(1024, 11008, false, "bfloat16",
                                       "bfloat16", "bfloat16");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_2000x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    // Not supported
    EXPECT_THROW(std::ignore = test_silu<int16_t>(
                     2000, 11008, false, "bfloat16", "bfloat16", "bfloat16"),
                 std::runtime_error);
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_2048x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    int err_count = test_silu<int16_t>(2048, 11008, false, "bfloat16",
                                       "bfloat16", "bfloat16");
    EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}

TEST(siluNoPadding, Kernel_123x11008) {
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    // Not supported
    EXPECT_THROW(std::ignore = test_silu<int16_t>(123, 11008, false, "bfloat16",
                                                  "bfloat16", "bfloat16"),
                 std::runtime_error);
  } else {
    GTEST_SKIP() << "Test not supported on " << Utils::get_env_var("DEVICE");
  }
}
