/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
 */

#include <gtest/gtest.h>
#include <iostream>

#include <qlinear.hpp>

template <typename T>
static void initialize_random(std::vector<T> &vector, size_t size,
                              int data_max) {
  auto ptr = vector.data();
  int data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    ptr[i] = (rand() % (data_max - data_min + 1)) + data_min;
  }
}

template <typename T>
static auto transpose(const std::vector<T> &src,
                      std::tuple<int, int> src_shape) {
  std::vector<T> dst(src.size());
  auto [R, C] = src_shape;

  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      dst[j * R + i] = src[i * C + j];
    }
  }
  std::tuple<int, int> dst_shape{C, R};
  return std::make_pair(dst, dst_shape);
}

template <typename InT, typename OutT>
static void matmul(std::vector<InT> &a, std::vector<InT> &b,
                   std::vector<OutT> &c, std::tuple<int, int> &a_shape,
                   std::tuple<int, int> &b_shape) {
  auto a_ptr = a.data();
  auto b_ptr = b.data();
  auto c_ptr = c.data();
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      OutT sum = 0;
      for (int k = 0; k < K; k++) {
        sum += ((OutT)a[i * K + k]) * ((OutT)b[k * N + j]);
      }
      c_ptr[i * N + j] = sum;
    }
  }
}

// Test 4x2kx2k dll
TEST(QlinearTest, QlinearBasic) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {4, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {4, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<0>(b_shape);

  std::vector<int8_t> a(M * K);
  std::vector<int8_t> b(K * N);
  std::vector<int32_t> c(M * N, 0);
  std::vector<int32_t> golden(M * N, 0);

  initialize_random<int8_t>(a, M * K, 127);
  initialize_random<int8_t>(b, K * N, 127);
  auto [bt, bt_shape] = transpose(b, b_shape);

  ryzenai::qlinear qlin = ryzenai::qlinear<int8_t, int8_t, int32_t>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, 1, "qlinear.txt");
  qlin.initialize_weights(bt.data(), bt_shape);
  qlin.execute(a.data(), a_shape, c.data());

  matmul<int8_t, int32_t>(a, bt, golden, a_shape, bt_shape);

  for (int i = 0; i < c.size(); i++) {
    // Divide by 1 is for requantizing from int32 to int32
    auto err = std::abs((golden.data()[i] / 1) - c.data()[i]);
    EXPECT_TRUE(err <= 4) << "Absolute error > 4";
  }
}

// Test 8x2kx2k dll
TEST(QlinearTest, QlinearBasic2) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_8x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {8, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {8, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<int8_t> a(M * K);
  std::vector<int8_t> b(K * N);
  std::vector<int32_t> c(M * N, 0);
  std::vector<int32_t> golden(M * N, 0);

  initialize_random<int8_t>(a, M * K, 5);
  initialize_random<int8_t>(b, K * N, 5);
  auto [bt, bt_shape] = transpose(b, b_shape);

  ryzenai::qlinear qlin = ryzenai::qlinear<int8_t, int8_t, int32_t>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, 1, "qlinear.txt");
  qlin.initialize_weights(bt.data(), bt_shape);
  qlin.execute(a.data(), a_shape, c.data());

  matmul<int8_t, int32_t>(a, bt, golden, a_shape, bt_shape);
  for (int i = 0; i < c.size(); i++) {
    // Divide by 1 is for requantizing from int32 to int32
    auto err = std::abs((golden.data()[i] / 1) - c.data()[i]);
    EXPECT_TRUE(err <= 4) << "Absolute error > 4";
  }
}

// Tiling with 4x2kx2k dll
TEST(QlinearTest, Tiling1) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {4, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {8, 2048};
  std::tuple<int, int> b_shape = {2048, 2048};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<int8_t> a(M * K);
  std::vector<int8_t> b(K * N);
  std::vector<int32_t> c(M * N, 0);
  std::vector<int32_t> golden(M * N, 0);

  initialize_random<int8_t>(a, M * K, 5);
  initialize_random<int8_t>(b, K * N, 5);
  auto [bt, bt_shape] = transpose(b, b_shape);

  ryzenai::qlinear qlin = ryzenai::qlinear<int8_t, int8_t, int32_t>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, 1, "qlinear.txt");
  qlin.initialize_weights(bt.data(), bt_shape);
  qlin.execute(a.data(), a_shape, c.data());

  matmul<int8_t, int32_t>(a, bt, golden, a_shape, bt_shape);
  for (int i = 0; i < c.size(); i++) {
    // Divide by 1 is for requantizing from int32 to int32
    auto err = std::abs((golden.data()[i] / 1) - c.data()[i]);
    EXPECT_TRUE(err <= 4) << "Absolute error > 4";
  }
}

// Tiling with 4x2kx2k dll
TEST(QlinearTest, Tiling2) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {4, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {8, 8192};
  std::tuple<int, int> b_shape = {2048, 8192};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<int8_t> a(M * K);
  std::vector<int8_t> b(K * N);
  std::vector<int32_t> c(M * N, 0);
  std::vector<int32_t> golden(M * N, 0);

  initialize_random<int8_t>(a, M * K, 5);
  initialize_random<int8_t>(b, K * N, 5);
  auto [bt, bt_shape] = transpose(b, b_shape);

  ryzenai::qlinear qlin = ryzenai::qlinear<int8_t, int8_t, int32_t>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, 1, "qlinear.txt");
  qlin.initialize_weights(bt.data(), bt_shape);
  qlin.execute(a.data(), a_shape, c.data());

  matmul<int8_t, int32_t>(a, bt, golden, a_shape, bt_shape);
  for (int i = 0; i < c.size(); i++) {
    // Divide by 1 is for requantizing from int32 to int32
    auto err = std::abs((golden.data()[i] / 1) - c.data()[i]);
    EXPECT_TRUE(err <= 4) << "Absolute error > 4";
  }
}

// Padding & Tiling with 4x2kx2k dll
TEST(QlinearTest, PaddingTiling) {

  const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";

  std::tuple<int, int> kernel_x_shape = {4, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {1, 8192};
  std::tuple<int, int> b_shape = {2048, 8192};
  auto M = std::get<0>(a_shape);
  auto K = std::get<1>(a_shape);
  auto N = std::get<1>(b_shape);

  std::vector<int8_t> a(M * K);
  std::vector<int8_t> b(K * N);
  std::vector<int32_t> c(M * N, 0);
  std::vector<int32_t> golden(M * N, 0);

  initialize_random<int8_t>(a, M * K, 5);
  initialize_random<int8_t>(b, K * N, 5);
  auto [bt, bt_shape] = transpose(b, b_shape);

  ryzenai::qlinear qlin = ryzenai::qlinear<int8_t, int8_t, int32_t>(
      aie_kernel_dll, kernel_x_shape, kernel_y_shape, 1, "qlinear.txt");
  qlin.initialize_weights(bt.data(), bt_shape);
  qlin.execute(a.data(), a_shape, c.data());

  matmul<int8_t, int32_t>(a, bt, golden, a_shape, bt_shape);
  for (int i = 0; i < c.size(); i++) {
    // Divide by 1 is for requantizing from int32 to int32
    auto err = std::abs((golden.data()[i] / 1) - c.data()[i]);
    EXPECT_TRUE(err <= 4) << "Absolute error > 4";
  }
}

// Padding & Tiling with 4x2kx2k dll and 8x2kx2k dlls
TEST(QlinearTest, PaddingTiling_2DLL) {

  const std::vector<std::string> aie_kernel_dlls = {
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll",
      Utils::get_dll_path() + "libGemmQnnAie_8x2048_2048x2048.dll",
  };

  std::vector<std::tuple<int, int>> kernel_x_shapes = {{4, 2048}, {8, 2048}};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  ryzenai::qlinear qlin = ryzenai::qlinear<int8_t, int8_t, int32_t>(
      aie_kernel_dlls, kernel_x_shapes, kernel_y_shape,
      /*num_workers*/ 1, /*num_dlls*/ 2, /*limit*/ 4, "qlinear.txt");

  {
    std::tuple<int, int> a_shape = {1, 8192};
    std::tuple<int, int> b_shape = {2048, 8192};
    auto M = std::get<0>(a_shape);
    auto K = std::get<1>(a_shape);
    auto N = std::get<1>(b_shape);

    std::vector<int8_t> a(M * K);
    std::vector<int8_t> b(K * N);
    std::vector<int32_t> c(M * N, 0);
    std::vector<int32_t> golden(M * N, 0);

    initialize_random<int8_t>(a, M * K, 5);
    initialize_random<int8_t>(b, K * N, 5);
    auto [bt, bt_shape] = transpose(b, b_shape);

    qlin.initialize_weights(bt.data(), bt_shape);
    qlin.execute(a.data(), a_shape, c.data());

    matmul<int8_t, int32_t>(a, bt, golden, a_shape, bt_shape);
    size_t err_count = 0;
    for (int i = 0; i < c.size(); i++) {
      // Divide by 1 is for requantizing from int32 to int32
      auto err = std::abs((golden.data()[i] / 1) - c.data()[i]);
      if (err >= 4) {
        err_count++;
      }
    }
    EXPECT_TRUE(err_count == 0) << "Absolute error > 4";
  }
  {
    std::tuple<int, int> a_shape = {8, 8192};
    std::tuple<int, int> b_shape = {2048, 8192};
    auto M = std::get<0>(a_shape);
    auto K = std::get<1>(a_shape);
    auto N = std::get<1>(b_shape);

    std::vector<int8_t> a(M * K);
    std::vector<int8_t> b(K * N);
    std::vector<int32_t> c(M * N, 0);
    std::vector<int32_t> golden(M * N, 0);

    const std::vector<std::string> aie_kernel_dlls = {
        Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll",
        Utils::get_dll_path() + "libGemmQnnAie_8x2048_2048x2048.dll",
    };

    auto [bt, bt_shape] = transpose(b, b_shape);

    qlin.initialize_weights(bt.data(), bt_shape);
    qlin.execute(a.data(), a_shape, c.data());

    matmul<int8_t, int32_t>(a, bt, golden, a_shape, bt_shape);
    size_t err_count = 0;
    for (int i = 0; i < c.size(); i++) {
      // Divide by 1 is for requantizing from int32 to int32
      auto err = std::abs((golden.data()[i] / 1) - c.data()[i]);
      if (err >= 4) {
        err_count++;
      }
    }
    EXPECT_TRUE(err_count == 0) << "Absolute error > 4";
  }
}
