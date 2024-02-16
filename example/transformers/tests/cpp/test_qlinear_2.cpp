/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
 */

#include <gtest/gtest.h>
#include <iostream>

#include <qlinear_2.hpp>

template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size, int data_max) {
  int data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    vec[i] = (rand() % (data_max - data_min + 1)) + data_min;
  }
}

template <typename InT, typename OutT, typename AccT = int>
static void matmul(std::vector<InT> &a, std::vector<InT> &b,
                   std::vector<OutT> &c, std::tuple<int, int> &a_shape,
                   std::tuple<int, int> &b_shape, bool apply_srs = false,
                   AccT shift = 12, AccT srs_min = -128, AccT srs_max = 127) {
  int M = std::get<0>(a_shape);
  int K = std::get<1>(a_shape);
  int N = std::get<1>(b_shape);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      AccT sum = 0;
      for (int k = 0; k < K; k++) {
        sum += AccT(a[i * K + k]) * AccT(b[k * N + j]);
      }
      if (apply_srs) {
        sum = sum >> shift;
        sum = (sum < srs_min) ? srs_min : sum;
        sum = (sum > srs_max) ? srs_max : sum;
      }
      c[i * N + j] = sum;
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
int test_matmul(int M, int K, int N, bool debug = false) {
  // NOTE: The actual kernel shape is decided in
  //       run_aie and initialize_weights to be compatible
  //       with the pytorch runtime, so these shapes don't
  //       matter. They exist for backwards compatibility.
  //       We use 8 x 2k x 2k as dummy values here.
  std::tuple<int, int> kernel_x_shape = {8, 2048};
  std::tuple<int, int> kernel_y_shape = {2048, 2048};
  std::tuple<int, int> a_shape = {M, K};
  std::tuple<int, int> b_shape = {K, N};

  std::vector<int8_t> a(M * K);
  std::vector<int8_t> b(K * N);
  int32_t garbage_value = 0xCDCDCDCD;
  std::vector<int32_t> c(M * N, garbage_value);
  std::vector<int32_t> c_golden(M * N, garbage_value);

  srand(42);
  initialize_random(a, M * K, 127);
  initialize_random(b, K * N, 127);
  matmul(a, b, c_golden, a_shape, b_shape);

  ryzenai::qlinear_2 qlin = ryzenai::qlinear_2<int8_t, int8_t, int32_t>(
      kernel_x_shape, kernel_y_shape);
  qlin.debug(debug);
  qlin.initialize_weights(b.data(), b_shape);
  qlin.execute(a.data(), a_shape, c.data());

  int err_count = 0;
  for (int i = 0; i < c.size(); i++) {
    int32_t err = std::abs(c_golden[i] - c[i]);
    if (err > 0) {
      std::cout << "c[" << i << "]: "
                << "Expected: " << c_golden[i] << ", "
                << "Received: " << c[i] << "\n";
      err_count += 1;
    }
  }
  return err_count;
}

/* Test kernel dimensions without tiling */

/* Test kernels with M = 1*/

TEST(Qlinear_2Test, Kernel1) {
  int err_count = test_matmul(1, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel2) {
  int err_count = test_matmul(1, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel3) {
  int err_count = test_matmul(1, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 8 */

TEST(Qlinear_2Test, Kernel4) {
  int err_count = test_matmul(8, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel5) {
  int err_count = test_matmul(8, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel6) {
  int err_count = test_matmul(8, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 16 */

TEST(Qlinear_2Test, Kernel7) {
  int err_count = test_matmul(16, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel8) {
  int err_count = test_matmul(16, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel9) {
  int err_count = test_matmul(16, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test kernels with M = 32 */

TEST(Qlinear_2Test, Kernel10) {
  int err_count = test_matmul(32, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel11) {
  int err_count = test_matmul(32, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Kernel12) {
  int err_count = test_matmul(32, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test tiling up to OPT vocabulary output dimension */

TEST(Qlinear_2Test, Tiling1) {
  int err_count = test_matmul(8, 2048, 50272);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test tiling in all three dimensions with padding */

TEST(Qlinear_2Test, Tiling2) {
  int err_count = test_matmul(42, 5000, 2500);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test padding multiple token input with OPT 1.3B dimensions */

TEST(Qlinear_2Test, Padding1) {
  int err_count = test_matmul(12, 2048, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Padding2) {
  int err_count = test_matmul(12, 2048, 8192);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Test, Padding3) {
  int err_count = test_matmul(12, 8192, 2048);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

/* Test debug feature */

TEST(Qlinear_2Test, Debug) {
  int err_count = test_matmul(1, 2048, 2048, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
