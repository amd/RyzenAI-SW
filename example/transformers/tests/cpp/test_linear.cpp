/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <gtest/gtest.h>
#include <iostream>

#include <linear.hpp>

using bfloat16 = int16_t;

bfloat16 toBfloat16(float f) {
  bfloat16 bf;
  std::memcpy(&bf, (uint8_t *)&f + 2, 2);
  return bf;
}

float toFloat(bfloat16 bf) {
  float f = 0;
  std::memcpy((uint8_t *)&f + 2, &bf, 2);
  return f;
}

void rand_init(bfloat16 *ptr, size_t size, float min, float max) {
  for (int i = 0; i < size; i++) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    r = (min - max) * r + max;
    ptr[i] = toBfloat16(r);
  }
}

void cpu_matmul(bfloat16 *a, bfloat16 *b, float *c, int M, int K, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += toFloat(a[i * K + k]) * toFloat(b[k * N + j]);
      }
      c[i * N + j] = sum;
    }
  }
}

// Test for run without padding/tiling etc
TEST(Linear, Basic) {
  const int M = 256;
  const int K = 2048;
  const int N = 2048;
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N, 0.0);

  // initialize random data
  rand_init(A.data(), M * K, -1, 1);
  rand_init(B.data(), K * N, -1, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(M, K), std::make_tuple(K, N));
  lin_bf16.initialize_weights(B.data(), std::make_tuple(K, N));
  lin_bf16.execute(A.data(), std::make_tuple(M, K), C.data());

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << "elements";
}

// Test for execute when a, b shapes are smaller than kernel shapes supported.
// Padding test
TEST(Linear, Padding) {
  const int M = 196;
  const int K = 512;
  const int N = 121;

  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N, 0.0);

  // initialize random data
  rand_init(A.data(), M * K, -1, 1);
  rand_init(B.data(), K * N, -1, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(256, 2048), std::make_tuple(2048, 2048));
  lin_bf16.initialize_weights(B.data(), b_shape);
  lin_bf16.execute(A.data(), a_shape, C.data());

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";
}

// Tiling test
TEST(Linear, MTiling) {
  const int M = 256 * 2;
  const int K = 2048;
  const int N = 2048;

  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N, 0.0);

  // initialize random data
  rand_init(A.data(), M * K, 0, 1);
  rand_init(B.data(), K * N, 0, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(256, 2048), std::make_tuple(2048, 2048));
  lin_bf16.initialize_weights(B.data(), b_shape);
  lin_bf16.execute(A.data(), a_shape, C.data());

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";
}

// Tiling test
TEST(Linear, NTiling) {
  const int M = 256;
  const int K = 2048;
  const int N = 2048 * 5;

  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N, 0.0);

  // initialize random data
  rand_init(A.data(), M * K, 0, 1);
  rand_init(B.data(), K * N, 0, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(256, 2048), std::make_tuple(2048, 2048));
  lin_bf16.initialize_weights(B.data(), b_shape);
  lin_bf16.execute(A.data(), a_shape, C.data());

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";
}

// Tiling test
TEST(Linear, KTiling) {
  const int M = 256;
  const int K = 2048 * 2;
  const int N = 2048;

  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N, 0.0);

  // initialize random data
  rand_init(A.data(), M * K, 0, 1);
  rand_init(B.data(), K * N, 0, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(256, 2048), std::make_tuple(2048, 2048));
  lin_bf16.initialize_weights(B.data(), b_shape);
  lin_bf16.execute(A.data(), a_shape, C.data());

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";
}

// Tiling test
TEST(Linear, AllTiling) {
  const int M = 256 * 2;
  const int K = 2048 * 2;
  const int N = 2048 * 3;

  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N);

  // initialize random data
  rand_init(A.data(), M * K, 0, 1);
  rand_init(B.data(), K * N, 0, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(256, 2048), std::make_tuple(2048, 2048));
  lin_bf16.initialize_weights(B.data(), b_shape);
  lin_bf16.execute(A.data(), a_shape, C.data());

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";
}

// Tiling & padding test
TEST(Linear, TilingPadding) {
  const int M = 233 * 2;
  const int K = 1920 * 5;
  const int N = 2300 * 2;

  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N, 0.0);

  // initialize random data
  rand_init(A.data(), M * K, 0, 1);
  rand_init(B.data(), K * N, 0, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(256, 2048), std::make_tuple(2048, 2048));
  lin_bf16.initialize_weights(B.data(), b_shape);
  lin_bf16.execute(A.data(), a_shape, C.data());

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";
}

// Stationary weights
TEST(Linear, StationaryWeightsBasic) {
  const int M = 256 * 2;
  const int K = 2048 * 2;
  const int N = 2048 * 2;

  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};
  // create vectors for input/output data;
  std::vector<bfloat16> A(M * K);
  std::vector<bfloat16> B(K * N);
  std::vector<float> C(M * N, 0.0);

  // initialize random data
  rand_init(A.data(), M * K, 0, 1);
  rand_init(B.data(), K * N, 0, 1);

  ryzenai::linear lin_bf16 =
      ryzenai::linear(std::make_tuple(256, 2048), std::make_tuple(2048, 2048));
  std::memset(C.data(), 0, C.size() * sizeof(bfloat16));
  auto first_run_start = std::chrono::high_resolution_clock::now();
  lin_bf16.initialize_weights(B.data(), b_shape);
  lin_bf16.execute(A.data(), a_shape, C.data());
  auto first_run_end = std::chrono::high_resolution_clock::now();

  // matmul on CPU
  std::vector<float> cpu_golden(M * N, 0.0);
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  int err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";

  // Change Matrix A but re-use Matrix B
  rand_init(A.data(), M * K, 0, 1);

  // Reinitialize C to 0
  std::memset(C.data(), 0, C.size() * sizeof(float));
  auto second_run_start = std::chrono::high_resolution_clock::now();
  lin_bf16.execute(A.data(), a_shape, C.data());
  auto second_run_end = std::chrono::high_resolution_clock::now();

  // matmul on CPU with new A
  cpu_matmul(A.data(), B.data(), cpu_golden.data(), M, K, N);

  // Validate data
  err_count = 0;
  for (int i = 0; i < M * N; i++) {
    auto *out = C.data();
    auto *golden = cpu_golden.data();

    // printf("Golden: %f, Out: %f\n", golden[i], out[i]);
    float p_err = std::abs((out[i] - golden[i]) / golden[i]);
    if (p_err > 0.05) {
      err_count++;
    }
  }

  EXPECT_TRUE(err_count == 0)
      << "More than 5 percent error for " << err_count << " elements";
}
